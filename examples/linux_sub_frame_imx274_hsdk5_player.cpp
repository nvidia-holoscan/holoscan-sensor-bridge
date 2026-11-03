/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// HSDK5 IMX274 sub-frame player (grayscale):
//   LinuxReceiverOp → CsiToBayerOp → ImageProcessorOp → SubFrameCombinerOp
//     → TensorExtractorOp → HolovizOp
//
// The sensor frame is transported and processed as horizontal slices of
// --sub-frame-rows rows each, so csi_to_bayer and image_processor run on a
// slice as soon as it lands instead of waiting for the whole frame.
// SubFrameCombinerOp reassembles the slices into a full-height [H, W, 1]
// uint16 Bayer frame and emits once the last row of a frame arrives;
// TensorExtractorOp then re-publishes it for HolovizOp as kR16Unorm.
//
// Note that the combiner emits whole frames, not bands, so this pipeline gives
// sub-frame transport and processing but full-frame display latency. Rendering
// each slice as it arrives is what SubFrameVisualizerOp is for.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <getopt.h>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/schema_identity.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/visualization/holoviz_op.hpp>
#include <holoscan/visualization/sink_desc.hpp>

#include <hololink/common/cuda_helper.hpp>
#include <hololink/common/tools.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/enumerator.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/core/logging.hpp>
#include <hololink/operators/csi_to_bayer/csi_to_bayer.hpp>
#include <hololink/operators/image_processor/image_processor.hpp>
#include <hololink/operators/linux_receiver/linux_receiver_op.hpp>
#include <hololink/operators/sub_frame_combiner/sub_frame_combiner.hpp>
#if HOLOLINK_HAS_SUB_FRAME_VISUALIZER
#include <hololink/operators/sub_frame_visualizer/sub_frame_visualizer.hpp>
#endif
#include <hololink/sensors/camera/imx274/native_imx274_sensor.hpp>

namespace {

// ---------------------------------------------------------------------------
// Re-publishes the combined [H, W, 1] Bayer frame as a MONO16 [H, W] image
// from this operator's own device pool, which HolovizOp's kR16Unorm layer
// accepts (it rejects Bayer encodings).
//
// The copy is required: forwarding the input tensor with a new descriptor
// would be a kPassThrough output, which has no pool allocation, so the
// compiler would assign no device to the extractor→holoviz channel and
// HolovizOp would be left without a CUDA compute stream. The runtime then
// refuses to bind the device tensor to HolovizOp's input
// (RUNTIME_PORT_BINDING_UNAVAILABLE). A kRuntimePool output with a device
// binding gives the channel a device, which provisions HolovizOp's stream.
// Cost: one D2D copy per frame.
// ---------------------------------------------------------------------------
class TensorExtractorOp : public holoscan::Operator<> {
public:
    /// \p frame_limit > 0 stops the session once that many frames have passed through.
    explicit TensorExtractorOp(int64_t frame_limit = 0)
        : frame_limit_(frame_limit)
    {
    }

    holoscan::Input<holoscan::schema::ImageT> input { "input" };
    // HolovizOp image layers take schema::ImageT: the uint16 frame plus its encoding, so the
    // wire format travels with the sample instead of being assumed by the sink.
    holoscan::Output<holoscan::schema::ImageT> output { "output" };

    void setup(holoscan::OperatorSpec& spec) override
    {
        static constexpr DLDataType kUInt16 { static_cast<uint8_t>(kDLUInt), 16, 1 };
        static constexpr std::size_t kMaxBytes = 4096ULL * 2160 * 2; // 4K single-channel u16

        (void)spec.input(input, "input")
            .queue_depth(1U)
            .tensor_field(holoscan::schema::ImageTensorFields::data);
        spec.output(output, "output")
            .max_emits_per_compute(1U)
            .produces_tensor(holoscan::TensorOutputSpec {
                .representation = holoscan::TensorRepresentation {
                    .memory_kind = holoscan::MemoryKind::kCudaDevice,
                    .dtype = kUInt16,
                    .rank = 2U,
                },
                .bounds = holoscan::TensorBounds {
                    .max_rank = 2U,
                    .max_byte_span = kMaxBytes,
                },
                .storage = holoscan::TensorOutputStorage::kRuntimePool,
            })
            .payload_options(holoscan::PayloadOptions {
                .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
            });
    }

    [[nodiscard]] holoscan::Contract contract() const override
    {
        holoscan::Contract c;
        c.trigger(holoscan::OnEach { input });
        return c;
    }

    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
        holoscan::ExecutionContext& context) override
    {
        auto sample = input.receive();
        if (!sample)
            return holoscan::make_unexpected(std::move(sample).error());
        if (frame_limit_ > 0 && ++frames_seen_ >= frame_limit_) {
            context.request_stop(); // this frame is still forwarded; the session ends after it
        }

        if (!sample->data.data)
            return holoscan::make_unexpected(holoscan::Error {
                holoscan::ErrorCode::kInvalidArgument, "image has no tensor" });

        const holoscan::Tensor& in = *sample->data.data;
        const auto shape = in.shape_span();
        if (shape.size() != 3) {
            std::cerr << "[tensor_extractor] expected rank-3 tensor, got rank "
                      << shape.size() << std::endl;
            return holoscan::make_unexpected(holoscan::Error {
                holoscan::ErrorCode::kInvalidArgument, "expected rank-3 [H, W, 1] tensor" });
        }

        const DLDataType dtype = in.dtype();
        if (dtype.code != kDLUInt || dtype.bits != 16) {
            std::cerr << "[tensor_extractor] expected uint16, got code "
                      << int(dtype.code) << " bits " << int(dtype.bits) << std::endl;
            return holoscan::make_unexpected(holoscan::Error {
                holoscan::ErrorCode::kInvalidArgument, "expected uint16 tensor" });
        }

        if (shape[2] != 1)
            return holoscan::make_unexpected(holoscan::Error {
                holoscan::ErrorCode::kInvalidArgument, "expected a single-channel tensor" });
        // MONO16 is published as {H, W}; the [H, W, 1] input is the same packed bytes.
        const std::array<int64_t, 2> out_shape { shape[0], shape[1] };
        const std::size_t nbytes = static_cast<std::size_t>(shape[0])
            * static_cast<std::size_t>(shape[1]) * static_cast<std::size_t>(shape[2])
            * sizeof(uint16_t);

        auto loan = output.allocate_tensor(
            holoscan::TensorLoanRequest { .shape = out_shape, .dtype = dtype });
        if (!loan) {
            if (loan.error().code == holoscan::ErrorCode::kResourceExhausted)
                return {}; // HolovizOp is behind; drop this frame
            std::cerr << "[tensor_extractor] allocate_tensor: "
                      << holoscan::error_code_name(loan.error().code) << ": "
                      << loan.error().message << std::endl;
            return holoscan::make_unexpected(std::move(loan).error());
        }

        auto writer = loan->write(context.cuda_stream());
        if (!writer)
            return holoscan::make_unexpected(std::move(writer).error());

        const cudaError_t err = cudaMemcpyAsync(writer->data(), in.data(), nbytes,
            cudaMemcpyDeviceToDevice, context.cuda_stream().get());
        if (err != cudaSuccess)
            return holoscan::make_unexpected(holoscan::Error {
                holoscan::ErrorCode::kFailure, cudaGetErrorString(err) });

        auto committed = std::move(*writer).commit();
        if (!committed)
            return holoscan::make_unexpected(std::move(committed).error());

        // emit() rejects an invalid capture_time; SubFrameCombinerOp forwards
        // the first sub-frame's, but fall back to the activation time anyway.
        holoscan::TimePoint capture = sample->metadata.capture_time();
        if (!capture.valid())
            capture = context.activation_time();

        holoscan::schema::ImageT descriptor {};
        descriptor.height = static_cast<int32_t>(shape[0]);
        descriptor.width = static_cast<int32_t>(shape[1]);
        descriptor.encoding = holoscan::schema::ImageEncoding_MONO16;
        descriptor.color_space = holoscan::schema::ColorSpace_UNSPECIFIED;
        auto emitted = output.emit_tensor(std::move(*loan), descriptor,
            holoscan::EmitOptions { .capture_time = capture });
        if (!emitted && emitted.error().code == holoscan::ErrorCode::kBackpressured)
            return {}; // consumer queue full under kDropOldest; not an error
        return emitted;
    }

private:
    int64_t frame_limit_ = 0;
    int64_t frames_seen_ = 0;
};

} // anonymous namespace

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

// --present-mode names. The first six map onto both HolovizOp's PresentMode and the HoloViz
// RHI's; the two shared modes exist only in the RHI, which SubFrameVisualizerOp drives directly.
constexpr const char* kPresentModeHelp
    = "auto|fifo|fifo-latest-ready|mailbox|immediate|fifo-relaxed|shared-demand-refresh|shared-continuous-refresh";

static bool is_shared_present_mode_name(const std::string& name)
{
    return name == "shared-demand-refresh" || name == "shared-continuous-refresh";
}

static std::optional<holoscan::visualization::PresentMode> parse_holoscan_present_mode(const std::string& name)
{
    using PM = holoscan::visualization::PresentMode;
    if (name == "auto")
        return PM::kAuto;
    if (name == "fifo")
        return PM::kFIFO;
    if (name == "fifo-latest-ready")
        return PM::kFIFOLatestReady;
    if (name == "mailbox")
        return PM::kMailbox;
    if (name == "immediate")
        return PM::kImmediate;
    if (name == "fifo-relaxed")
        return PM::kFIFORelaxed;
    return std::nullopt;
}

#if HOLOLINK_HAS_SUB_FRAME_VISUALIZER
static std::optional<holoviz::PresentMode> parse_holoviz_present_mode(const std::string& name)
{
    using PM = holoviz::PresentMode;
    if (name == "auto")
        return PM::kAuto;
    if (name == "fifo")
        return PM::kFIFO;
    if (name == "fifo-latest-ready")
        return PM::kFIFOLatestReady;
    if (name == "mailbox")
        return PM::kMailbox;
    if (name == "immediate")
        return PM::kImmediate;
    if (name == "fifo-relaxed")
        return PM::kFIFORelaxed;
    if (name == "shared-demand-refresh")
        return PM::kSharedDemandRefresh;
    if (name == "shared-continuous-refresh")
        return PM::kSharedContinuousRefresh;
    return std::nullopt;
}
#endif

static int run_player(int argc, char** argv, holoscan::ManagedShutdownSignals& signals)
{
    int32_t camera_mode = 0;
    int64_t frame_limit = 0;
    std::string hololink_ip = hololink::env_hololink_ip(0, "192.168.0.2");
    int32_t expander_configuration = 0;
    int cu_device_ordinal = 0;
    // The camera height must be evenly divisible by the sub-frame rows; 216
    // satisfies that for every IMX274 mode.
    uint32_t sub_frame_rows = 216;
    uint32_t receiver_pages = 0; // 0 = derive from the slice count
    bool sub_frame_display = false; // render bands as they arrive (SubFrameVisualizerOp)
    bool exclusive_display = false; // direct-to-display scanout instead of a window
    std::string display_name; // window monitor / exclusive DRM connector (substring); empty = default
    std::string present_mode_name = "auto";

    const struct option long_options[] = {
        { "help", no_argument, nullptr, 'h' },
        { "camera-mode", required_argument, nullptr, 0 },
        { "frame-limit", required_argument, nullptr, 0 },
        { "hololink", required_argument, nullptr, 0 },
        { "expander-configuration", required_argument, nullptr, 0 },
        { "sub-frame-rows", required_argument, nullptr, 0 },
        { "receiver-pages", required_argument, nullptr, 0 },
        { "sub-frame-display", no_argument, nullptr, 0 },
        { "exclusive-display", no_argument, nullptr, 0 },
        { "display-name", required_argument, nullptr, 0 },
        { "present-mode", required_argument, nullptr, 0 },
        { 0, 0, nullptr, 0 }
    };

    while (true) {
        int option_index = 0;
        const int c = getopt_long(argc, argv, "h", long_options, &option_index);
        if (c == -1)
            break;

        const std::string argument(optarg ? optarg : "");
        if (c == 0) {
            const struct option* cur = &long_options[option_index];
            if (cur->name == std::string("camera-mode")) {
                camera_mode = std::stoi(argument);
            } else if (cur->name == std::string("frame-limit")) {
                frame_limit = std::stoll(argument);
            } else if (cur->name == std::string("hololink")) {
                hololink_ip = argument;
            } else if (cur->name == std::string("expander-configuration")) {
                expander_configuration = std::stoul(argument);
            } else if (cur->name == std::string("sub-frame-rows")) {
                sub_frame_rows = std::stoul(argument);
            } else if (cur->name == std::string("receiver-pages")) {
                receiver_pages = std::stoul(argument);
            } else if (cur->name == std::string("sub-frame-display")) {
                sub_frame_display = true;
            } else if (cur->name == std::string("exclusive-display")) {
                exclusive_display = true;
            } else if (cur->name == std::string("display-name")) {
                display_name = argument;
            } else if (cur->name == std::string("present-mode")) {
                present_mode_name = argument;
            }
        } else if (c == '?') {
            return EXIT_FAILURE; // getopt_long has reported the unknown option or missing argument
        } else if (c == 'h') {
            std::cout
                << "Usage: " << argv[0] << " [options]\n"
                << "  --hololink <ip>               HSB board IP (default " << hololink_ip << ")\n"
                << "  --camera-mode <0|1|2>         IMX274 mode (default 0 = 3840x2160@60)\n"
                << "  --frame-limit <n>             Stop after N frames (0 = run forever)\n"
                << "  --expander-configuration <n>  Expander config (default 0)\n"
                << "  --sub-frame-rows <n>          Rows per sub-frame (default 216; even,\n"
                << "                                and must divide the camera height evenly)\n"
                << "  --receiver-pages <n>          Receiver buffer pages (default: 3 frames\n"
                << "                                of chunks, capped at 30)\n"
                << "  --sub-frame-display           Display each band as it arrives through\n"
                << "                                SubFrameVisualizerOp instead of assembling\n"
                << "                                full frames for HolovizOp\n"
                << "  --exclusive-display           Scan out directly to a display no compositor\n"
                << "                                owns (Linux DRM; run from a console with the\n"
                << "                                display manager stopped) instead of a window\n"
                << "  --display-name <name>         Monitor for the fullscreen window, or the DRM\n"
                << "                                connector (substring, e.g. DP-2) for\n"
                << "                                --exclusive-display; empty = default/first free\n"
                << "  --present-mode <mode>         " << kPresentModeHelp << "\n"
                << "                                (default auto = best tear-free mode). The\n"
                << "                                shared-* modes need --sub-frame-display and a\n"
                << "                                surface offering VK_KHR_shared_presentable_image\n"
                << "                                (typically --exclusive-display)\n";
            return EXIT_SUCCESS;
        }
    }

    if (!parse_holoscan_present_mode(present_mode_name) && !is_shared_present_mode_name(present_mode_name)) {
        std::cerr << "Unknown --present-mode '" << present_mode_name << "' (expected one of "
                  << kPresentModeHelp << ")" << std::endl;
        return EXIT_FAILURE;
    }
    if (is_shared_present_mode_name(present_mode_name) && !sub_frame_display) {
        std::cerr << "--present-mode " << present_mode_name
                  << " renders in place into a shared presentable image, which only the\n"
                     "--sub-frame-display path (SubFrameVisualizerOp) supports; HolovizOp offers\n"
                     "auto|fifo|fifo-latest-ready|mailbox|immediate|fifo-relaxed."
                  << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Initializing CUDA..." << std::endl;
    CudaCheck(cuInit(0));
    CUdevice cu_device;
    CudaCheck(cuDeviceGet(&cu_device, cu_device_ordinal));
    CUcontext cu_context;
    CudaCheck(cuDevicePrimaryCtxRetain(&cu_context, cu_device));
    cuCtxSetCurrent(cu_context);

    std::cout << "Enumerating HSB board at " << hololink_ip << "..." << std::endl;
    hololink::Metadata channel_metadata = hololink::Enumerator::find_channel(hololink_ip);
    auto hololink_channel = std::make_shared<hololink::DataChannel>(channel_metadata);

    std::cout << "Configuring IMX274 sensor (mode " << camera_mode << ")..." << std::endl;
    auto camera = std::make_shared<hololink::sensors::NativeImx274Sensor>(
        *hololink_channel, expander_configuration);
    auto imx274_mode = static_cast<hololink::sensors::imx274_mode::Mode>(camera_mode);
    camera->set_mode(imx274_mode);

    auto bayer_format = camera->get_bayer_format();
    auto pixel_format = camera->get_pixel_format();
    const auto frame_width = static_cast<uint32_t>(camera->get_width());
    const auto frame_height = static_cast<uint32_t>(camera->get_height());
    if (frame_width == 0 || frame_height == 0) {
        std::cerr << "Unsupported --camera-mode " << camera_mode << std::endl;
        CudaCheck(cuDevicePrimaryCtxRelease(cu_device));
        return EXIT_FAILURE;
    }

    if (sub_frame_rows == 0) {
        sub_frame_rows = frame_height;
    } else if (sub_frame_rows > frame_height) {
        std::cerr << "--sub-frame-rows " << sub_frame_rows
                  << " exceeds the camera height of " << frame_height << std::endl;
        CudaCheck(cuDevicePrimaryCtxRelease(cu_device));
        return EXIT_FAILURE;
    } else if (frame_height % sub_frame_rows != 0) {
        std::cerr << "Camera height " << frame_height
                  << " is not evenly divisible by --sub-frame-rows " << sub_frame_rows << std::endl;
        CudaCheck(cuDevicePrimaryCtxRelease(cu_device));
        return EXIT_FAILURE;
    } else if (sub_frame_rows % 2 != 0) {
        // ImageProcessorOp takes the color filter phase from each band's own first row, so
        // every band must start on an even image row.
        std::cerr << "--sub-frame-rows " << sub_frame_rows << " is odd; it must be even" << std::endl;
        CudaCheck(cuDevicePrimaryCtxRelease(cu_device));
        return EXIT_FAILURE;
    }
    const uint32_t slices_per_frame = frame_height / sub_frame_rows;

    // CSI-to-Bayer, sliced into sub_frame_rows-row chunks
    hololink::operators::CsiToBayerParams csi_params {
        .cuda_device_ordinal = cu_device_ordinal,
        .sub_frame_rows = sub_frame_rows,
        .bayer_format = bayer_format,
    };
    camera->configure_converter(csi_params.geometry);

    // In sub-frame mode the receiver delivers one slice at a time, so it is
    // sized to a slice rather than to the whole CSI frame. This mirrors what
    // CsiToBayerOp::configure() computes for frame_size_ once it starts.
    const size_t frame_size = csi_params.geometry.bytes_per_line * sub_frame_rows;

    // The receiver ring defaults to a single page. In sub-frame mode that means the
    // next slice overwrites the one still in flight, and the combiner then sees a
    // gap in sub_frame_offset and rejects the frame as non-contiguous. Size it for
    // at least three complete frames, as the other sub-frame players do; a frame
    // takes one more chunk than it has slices when the CSI data carries a header.
    static constexpr uint32_t kMaxPages = 30;
    if (receiver_pages == 0) {
        const size_t csi_length = csi_params.geometry.start_byte
            + static_cast<size_t>(csi_params.geometry.bytes_per_line) * frame_height
            + csi_params.geometry.trailing_bytes;
        const auto chunks_per_frame = static_cast<uint32_t>((csi_length + frame_size - 1) / frame_size);
        receiver_pages = std::min(chunks_per_frame * 3U, kMaxPages);
    }

    std::cout << "Sub-frame mode: " << slices_per_frame << " slices of "
              << sub_frame_rows << " rows, " << receiver_pages << " receiver pages"
              << std::endl;

    // Build the graph.
    holoscan::Graph graph { "linux_sub_frame_imx274_player" };

    // Receiver
    // The receiver takes its runtime dependencies (data channel, device start/stop) through
    // the constructor; only sizing goes through set_params().
    auto rx = graph.op<hololink::operators::LinuxReceiverOp>(
        "rx", hololink_channel, [camera] { camera->start(); },
        [camera] {
            camera->stop();
            // Let the egress buffer drain, as HSDK4's IMX274 driver did.
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        });

    auto csi = graph.op<hololink::operators::CsiToBayerOp>("csi_to_bayer");
    csi.set_params(csi_params);

    rx.set_params(hololink::operators::LinuxReceiverParams {
        .frame_context = cu_context,
        .frame_size = frame_size,
        .pages = receiver_pages,
        // Match queue_size to pages to reduce the chance of dropping a slice.
        .queue_size = receiver_pages,
        // Report "Ingress frame timeout" when no data comes, as the other sub-frame players do.
        .watch_ingress = true,
    });

    // Image processor: white balance on single-channel Bayer, per sub-frame
    auto ip = graph.op<hololink::operators::ImageProcessorOp>("image_processor");
    ip.set_params(hololink::operators::ImageProcessorParams {
        .pixel_format = int(pixel_format),
        .bayer_format = int(bayer_format),
        .optical_black = 50,
        .cuda_device_ordinal = cu_device_ordinal,
        .pixel_width = csi_params.geometry.pixel_width,
        .pixel_height = csi_params.geometry.pixel_height,
    });

    // Wire the pipeline. The slices run through the capture lane in step, so each queue up to
    // the combiner holds the slice being worked on and at most one more; keep them that shallow,
    // since every queue entry costs a 64 MiB output pool slot in the receiver, CsiToBayerOp and
    // ImageProcessorOp.
    const holoscan::ConnectionOptions sub_frame_edge { .queue_depth = 2U };
    graph.add_flow(rx->tensor, csi->input_tensor, sub_frame_edge);
    graph.add_flow(rx->metadata, csi->input_metadata, sub_frame_edge);
    graph.add_flow(csi->output_tensor, ip->input_tensor, sub_frame_edge);
    graph.add_flow(csi->output_metadata, ip->input_metadata, sub_frame_edge);
    if (sub_frame_display) {
#if HOLOLINK_HAS_SUB_FRAME_VISUALIZER
        // Sub-frame display: every band goes straight from the image processor to
        // SubFrameVisualizerOp, which draws it into the current swapchain image at its
        // sub_frame_offset and presents once the last band of the frame has landed. No
        // combiner, so display latency is bounded by the last band, not the whole frame.
        // The PTP synchronizer is not wired in this player (nullptr), so the FPO capture
        // scheduling thread stays off. --exclusive-display scans out straight to the DRM
        // connector; --present-mode shared-continuous-refresh additionally streams every band
        // into the single shared presentable image with no per-frame acquire/present.
        const holoviz::PresentMode sub_frame_present_mode
            = parse_holoviz_present_mode(present_mode_name).value_or(holoviz::PresentMode::kAuto);
        auto sub_frame_viz = graph.op<hololink::operators::SubFrameVisualizerOp>(
            "sub_frame_visualizer",
            frame_height,
            /*fullscreen=*/true,
            /*use_exclusive_display=*/exclusive_display,
            /*display_name=*/display_name,
            /*display_width=*/frame_width,
            /*display_height=*/frame_height,
            /*display_framerate=*/60'000U,
            /*window_title=*/std::string { "IMX274 Sub-Frame Player (bands)" },
            /*ptp_synchronizer=*/nullptr,
            /*present_mode=*/sub_frame_present_mode);
        // Freshest-frame delivery: the visualizer blocks on the swapchain at the display rate,
        // so when the camera outruns it the queue must shed bands, not stall for seconds and
        // then burst. Up to one frame's worth of bands (at most four, since each queued band
        // holds a 64 MiB ImageProcessorOp pool slot) buffered, oldest dropped first; the band
        // and its metadata queues shed alike, so they stay paired.
        const holoscan::ConnectionOptions band_edge { .queue_policy = holoscan::QueuePolicy::kDropOldest,
            .queue_depth = std::min(slices_per_frame, 4U) };
        graph.add_flow(ip->output_tensor, sub_frame_viz->input_tensor, band_edge);
        graph.add_flow(ip->output_metadata, sub_frame_viz->input_metadata, band_edge);
        if (frame_limit > 0) {
            std::cout << "Note: --frame-limit applies to the default (combiner + HolovizOp) path; "
                         "the --sub-frame-display path runs until the window closes\n";
        }

        // Same lane split as below: the visualizer blocks on the swapchain, and the band
        // processing can hold up whatever shares its lane, so keep both off the receiver's lane.
        graph.partition("receiver").add(rx);
        graph.partition("capture").add(csi, ip);
        graph.auto_partition_remaining();
#else
        std::cerr << "--sub-frame-display requires SubFrameVisualizerOp, which was not built "
                     "(HoloViz library not found when configuring hololink)"
                  << std::endl;
        CudaCheck(cuDevicePrimaryCtxRelease(cu_device));
        return EXIT_FAILURE;
#endif
    } else {
        // Reassemble the slices into a full-height frame
        auto combiner = graph.op<hololink::operators::SubFrameCombinerOp>("sub_frame_combiner");
        combiner.set_params(hololink::operators::SubFrameCombinerParams {
            .cuda_device_ordinal = cu_device_ordinal,
            .height = frame_height,
        });

        // Re-publish the uint16 Bayer frame as a MONO16 schema::ImageT in its own device pool
        auto extractor = graph.op<TensorExtractorOp>("tensor_extractor", frame_limit);

        // HolovizOp: display as single-channel uint16 grayscale (kR16Unorm)
        holoscan::visualization::LayerStack layers;
        layers.add_image_layer("video",
            holoscan::visualization::ImageLayerDesc {
                .format = holoscan::visualization::ImageFormat::kR16Unorm,
                .max_width = frame_width,
                .max_height = frame_height,
            });

        // Sink: a fullscreen window through the compositor, or (--exclusive-display) direct
        // scanout on a DRM connector no display server owns. Both take --present-mode.
        const holoscan::visualization::PresentMode holoviz_present_mode
            = parse_holoscan_present_mode(present_mode_name).value_or(holoscan::visualization::PresentMode::kAuto);
        holoscan::visualization::SinkDesc sink;
        if (exclusive_display) {
            sink = holoscan::visualization::ExclusiveDisplaySinkDesc {
                .display_name = display_name,
                .width = frame_width,
                .height = frame_height,
                .refresh_hz = 60U,
                .present_mode = holoviz_present_mode,
                // The extractor's uint16 data is linear sensor output: encode to sRGB on write.
                .framebuffer_srgb = true,
            };
        } else {
            sink = holoscan::visualization::WindowSinkDesc {
                .title = "IMX274 Sub-Frame Player (grayscale u16)",
                .width = frame_width,
                .height = frame_height,
                .present_mode = holoviz_present_mode,
                .fullscreen = true,
                .display_name = display_name,
                .close_on_escape = true,
                .framebuffer_srgb = true,
            };
        }

        auto viz = graph.op<holoscan::visualization::HolovizOp>("holoviz",
            std::move(layers),
            std::move(sink),
            // Render one frame per assembled frame arriving on layer 0 (the former fixed
            // 60 Hz clock trigger is gone from HolovizOp).
            holoscan::visualization::LayerTrigger {},
            std::chrono::milliseconds(100));

        graph.add_flow(ip->output_tensor, combiner->input_tensor, sub_frame_edge);
        graph.add_flow(ip->output_metadata, combiner->input_metadata, sub_frame_edge);
        // The display lane (below) runs at the display's rate. The combined frame goes there
        // through a one-frame queue that drops the oldest: the newest frame is shown, and the
        // capture lane never waits for the display.
        graph.add_flow(combiner->output_tensor, extractor->input,
            holoscan::ConnectionOptions { .queue_policy = holoscan::QueuePolicy::kDropOldest, .queue_depth = 1U });
        graph.add_flow(extractor->output, viz->image_layer_input(0),
            holoscan::ConnectionOptions { .queue_policy = holoscan::QueuePolicy::kDropOldest, .queue_depth = 1U });

        // Split the pipeline across three execution lanes. HolovizOp blocks on the
        // swapchain at the display refresh rate, and the band processing can hold up
        // whatever shares its lane; on the receiver's lane either stall is long enough
        // for the receiver to miss a chunk. Since the sub-frame row offset is derived
        // from the receiver's chunk counter, a missed chunk becomes a missing
        // sub-frame. The receiver runs on a lane of its own; everything else up to the
        // combiner runs at slice rate on another; the extractor and HolovizOp run at
        // frame rate. Author the slice-rate partitions exactly and let the remainder
        // (the extractor and HolovizOp, plus any operators HolovizOp embeds) fall into
        // its own lane.
        graph.partition("receiver").add(rx);
        graph.partition("capture").add(csi, ip, combiner);
        graph.auto_partition_remaining();
    }

    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

    holoscan::CompileOptions options;
    std::vector<std::pair<std::string_view, std::string_view>> tensor_outputs {
        { "rx", "tensor" }, { "csi_to_bayer", "tensor" }, { "image_processor", "tensor" }
    };
    if (!sub_frame_display) {
        tensor_outputs.insert(tensor_outputs.end(),
            { { "sub_frame_combiner", "tensor" }, { "tensor_extractor", "output" } });
    }
    for (const auto& [operator_path, output_port] : tensor_outputs) {
        options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement {
            .operator_path = operator_path,
            .output_port = output_port,
            .device = holoscan::DeviceId { cu_device_ordinal },
        });
    }

    auto plan = holoscan::compile(graph, std::move(options));
    if (!plan.ok()) {
        std::cerr << "Graph compilation failed:" << std::endl;
        for (const auto& diag : plan.diagnostics()) {
            std::cerr << "  [" << diag.token << "]";
            if (!diag.service.empty())
                std::cerr << " service=" << diag.service;
            if (!diag.endpoint.empty())
                std::cerr << " endpoint=" << diag.endpoint;
            if (!diag.required.empty())
                std::cerr << " required=" << diag.required;
            if (!diag.provided.empty())
                std::cerr << " provided=" << diag.provided;
            if (!diag.json_pointer.empty())
                std::cerr << " at=" << diag.json_pointer;
            std::cerr << std::endl;
        }
        CudaCheck(cuDevicePrimaryCtxRelease(cu_device));
        return EXIT_FAILURE;
    }

    std::cout << "Starting hardware..." << std::endl;
    auto hololink_dev = hololink_channel->hololink();
    hololink_dev->start();
    hololink_dev->reset();
    camera->setup_clock();
    camera->configure(imx274_mode);
    camera->set_digital_gain_reg(0x4);

    std::cout << "Running pipeline (Ctrl+C to stop)..." << std::endl;
    auto session = holoscan::run_async(plan, signals);

    // Block until the session ends (frame limit hit, error, or SIGINT).
    session.wait();
    const bool failed = session.termination() == holoscan::RunTermination::kFailed;

    for (const auto& diag : session.diagnostics()) {
        std::cerr << "Session diagnostic: "
                  << holoscan::format_diagnostic(diag, holoscan::ExecutionGraphView {})
                  << " vertex=" << diag.source.vertex.value
                  << std::endl;
    }

    hololink_dev->stop();
    CudaCheck(cuDevicePrimaryCtxRelease(cu_device));

    std::cout << "Done." << std::endl;
    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}

int main(int argc, char** argv)
{
    // First, before anything that may start a thread (CUDA, the board connection): with managed
    // signals, Ctrl+C stops the session cooperatively, so the camera and the board are stopped
    // on the way out.
    auto created_signals = holoscan::ManagedShutdownSignals::create();
    if (!created_signals) {
        std::cerr << holoscan::runtime_error_message(created_signals.error()) << std::endl;
        return EXIT_FAILURE;
    }
    auto signals = std::move(created_signals).value();

    int result = EXIT_FAILURE;
    try {
        result = run_player(argc, argv, signals);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    if (auto closed = signals.close(); !closed) {
        std::cerr << holoscan::runtime_error_message(closed.error()) << std::endl;
        return EXIT_FAILURE;
    }
    return result;
}
