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

// MAX96716A internal video pattern generator (VPG) live viewer over the software (Linux)
// receiver:
//   LinuxReceiverOp -> RgbReshapeOp -> HolovizOp
//
// No Hawk/Ar0234 sensor is involved -- this exercises only the MAX96716A deserializer's
// built-in test-pattern generator over the module's ported receiver
// (hololink::module::operators::LinuxReceiverOp, itself ported to HSDK5's Operator<Params>
// shape). RgbReshapeOp reshapes the receiver's flat RGB888 byte buffer into a (H, W, 3)
// schema::ImageT tensor Holoviz can render directly -- no CSI/Bayer conversion is involved.

#include <getopt.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <capture_metadata_traits.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/visualization/holoviz_op.hpp>

#include "hololink/module/adapter.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/hololink.hpp"
#include "hololink/module/operators/linux_receiver_op.hpp"
#include "hololink/module/operators/log_level.hpp"
#include "hololink/module/taurotech_da326/taurotech_da326.hpp"
#include "max96716a.hpp"
#include <hololink/common/cuda_helper.hpp>
#include <hololink/common/tools.hpp>
#include <hololink/core/logging.hpp>
#include <hololink/operators/capture_metadata_utils.hpp>

#include "player_support.hpp"

namespace hx = hololink::examples;

namespace {

using hololink::module::sensors::max96716a::Max96716a;

void write24(Max96716a& deserializer, uint16_t base, uint32_t value)
{
    deserializer.set_register(base + 0, static_cast<uint8_t>((value >> 16) & 0xFF));
    deserializer.set_register(base + 1, static_cast<uint8_t>((value >> 8) & 0xFF));
    deserializer.set_register(base + 2, static_cast<uint8_t>(value & 0xFF));
}

void write16(Max96716a& deserializer, uint16_t base, uint32_t value)
{
    deserializer.set_register(base + 0, static_cast<uint8_t>((value >> 8) & 0xFF));
    deserializer.set_register(base + 1, static_cast<uint8_t>(value & 0xFF));
}

void configure_vpg(Max96716a& deserializer, int32_t width, int32_t height,
    const std::string& pattern)
{
    const int32_t h_active = width;
    const int32_t h_blank = 280;
    const int32_t h_total = h_active + h_blank;
    const int32_t v_active = height;
    const int32_t v_blank = 45;
    const int32_t v_total = v_active + v_blank;

    const uint32_t vs_high = static_cast<uint32_t>(v_active * h_total);
    const uint32_t vs_low = static_cast<uint32_t>(v_blank * h_total);

    const uint32_t hs_high = 44;
    const uint32_t hs_low = static_cast<uint32_t>(h_total - hs_high);
    const uint32_t hs_cnt = static_cast<uint32_t>(v_total);

    const uint32_t de_high = static_cast<uint32_t>(h_active);
    const uint32_t de_low = static_cast<uint32_t>(h_blank);
    const uint32_t de_cnt = static_cast<uint32_t>(v_active);

    write24(deserializer, 0x0245, vs_high); // VS_HIGH_2
    write24(deserializer, 0x0248, vs_low); // VS_LOW_2
    write16(deserializer, 0x024E, hs_high); // HS_HIGH_1
    write16(deserializer, 0x0250, hs_low); // HS_LOW_1
    write16(deserializer, 0x0252, hs_cnt); // HS_CNT_1
    write16(deserializer, 0x0257, de_high); // DE_HIGH_1
    write16(deserializer, 0x0259, de_low); // DE_LOW_1
    write16(deserializer, 0x025B, de_cnt); // DE_CNT_1

    uint8_t patgen_mode;
    if (pattern == "checkerboard") {
        deserializer.set_register(0x025E, 0x00); // CHKR_COLOR_A_L
        deserializer.set_register(0x025F, 0x00); // CHKR_COLOR_A_M
        deserializer.set_register(0x0260, 0xFF); // CHKR_COLOR_A_M
        deserializer.set_register(0x0261, 0xFF); // CHKR_COLOR_B_L
        deserializer.set_register(0x0262, 0x00); // CHKR_COLOR_B_M
        deserializer.set_register(0x0263, 0x00); // CHKR_COLOR_B_H
        deserializer.set_register(0x0264, 16); // CHKR_RPT_A
        deserializer.set_register(0x0265, 16); // CHKR_RPT_B
        deserializer.set_register(0x0266, 8); // CHKR_ALT
        patgen_mode = 0b01;
    } else if (pattern == "gradient") {
        deserializer.set_register(0x025D, 0x01); // GRAD_INCR
        patgen_mode = 0b10;
    } else {
        throw std::runtime_error("Unknown VPG pattern: " + pattern);
    }

    deserializer.set_register(0x0240, 0xE3); // PATGEN_0
    deserializer.set_register(0x0241, static_cast<uint8_t>(patgen_mode << 4)); // PATGEN_1
}

struct RgbReshapeOpParams {
    int32_t width { 0 };
    int32_t height { 0 };
};

// Reshapes the flat RGB888 receiver buffer into a (H, W, 3) schema::ImageT tensor Holoviz can
// render directly. No Bayer/CSI conversion is needed -- the VPG already emits interleaved RGB.
// Equivalent to the Python InstrumentedTimeProfiler's `cp_frame[:pixels].reshape(height, width, 3)`.
class RgbReshapeOp : public holoscan::Operator<RgbReshapeOpParams> {
public:
    holoscan::Input<holoscan::Tensor> input_tensor { "tensor" };
    holoscan::Input<hololink::CaptureMetadataT> input_metadata { "metadata" };
    holoscan::Output<holoscan::schema::ImageT> output { "output" };

    void setup(holoscan::OperatorSpec& spec) override
    {
        static constexpr DLDataType kUInt8 { static_cast<uint8_t>(kDLUInt), 8, 1 };
        // Matches the receiver's own pool ceiling so this stage is never the tighter
        // bottleneck -- the receiver can't forward a frame larger than this anyway.
        static constexpr std::size_t kMaxBytes
            = hololink::module::operators::LinuxReceiverOp::kMaxFrameBytes;

        spec.input(input_tensor, "tensor").queue_depth(4U).accepts_projected_tensor();
        spec.input(input_metadata, "metadata").queue_depth(4U);
        spec.output(output, "output")
            .max_emits_per_compute(1U)
            .tensor_field(holoscan::schema::ImageTensorFields::data)
            .produces_tensor(holoscan::TensorOutputSpec {
                .representation = holoscan::TensorRepresentation {
                    .memory_kind = holoscan::MemoryKind::kCudaDevice,
                    .dtype = kUInt8,
                    .rank = 3U,
                },
                .bounds = holoscan::TensorBounds {
                    .max_rank = 3U,
                    .max_byte_span = kMaxBytes,
                },
                .storage = holoscan::TensorOutputStorage::kRuntimePool,
            });
    }

    [[nodiscard]] holoscan::Contract contract() const override
    {
        holoscan::Contract c;
        c.trigger(holoscan::OnAligned {
                      .ports = holoscan::ports(input_tensor, input_metadata),
                      .window = hololink::operators::kMetadataPairingWindow,
                  })
            .input(input_tensor, holoscan::Exact {})
            .input(input_metadata, holoscan::Exact {});
        return c;
    }

    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
        holoscan::ExecutionContext& context) override
    {
        auto result = compute_impl(context);
        if (!result && result.error().code == holoscan::ErrorCode::kNotReady) {
            return {};
        }
        return result;
    }

    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute_impl(
        holoscan::ExecutionContext& context)
    {
        const RgbReshapeOpParams& configuration = params();
        const int32_t width = configuration.width;
        const int32_t height = configuration.height;
        const std::size_t pixels
            = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U;

        // Consume the receiver's correlated metadata with each image, even though
        // the display stage only needs the reshaped RGB tensor.
        auto maybe_paired = hololink::operators::receive_paired_tensor_metadata(
            input_tensor, input_metadata);
        if (!maybe_paired)
            return holoscan::make_unexpected(std::move(maybe_paired).error());
        if (!*maybe_paired) {
            hx::count_mismatched_frame(mismatched_frames_, "RgbReshapeOp");
            return {};
        }
        auto& paired = **maybe_paired;

        const auto& sample = paired.tensor;
        const holoscan::Tensor& in = sample.data;
        if (static_cast<std::size_t>(in.nbytes()) < pixels) {
            // Short frame — drop it. Holoviz can't render a partial image.
            return {};
        }

        const std::array<int64_t, 3> out_shape { height, width, 3 };
        auto loan = output.allocate_tensor(holoscan::TensorLoanRequest {
            .shape = out_shape,
            .dtype = DLDataType { static_cast<uint8_t>(kDLUInt), 8, 1 },
        });
        if (!loan) {
            if (loan.error().code == holoscan::ErrorCode::kResourceExhausted)
                return {}; // HolovizOp is behind; drop this frame
            return holoscan::make_unexpected(std::move(loan).error());
        }

        auto writer = loan->write(context.cuda_stream());
        if (!writer)
            return holoscan::make_unexpected(std::move(writer).error());

        const cudaStream_t stream = context.cuda_stream().get();
        if (cudaMemcpyAsync(writer->data(), in.data(), pixels, cudaMemcpyDeviceToDevice, stream)
            != cudaSuccess) {
            return holoscan::make_unexpected(holoscan::Error { holoscan::ErrorCode::kFailure,
                "RgbReshapeOp: device-to-device copy failed" });
        }

        auto committed = std::move(*writer).commit();
        if (!committed)
            return holoscan::make_unexpected(std::move(committed).error());

        holoscan::TimePoint capture = sample.metadata.capture_time();
        if (!capture.valid())
            capture = context.activation_time();

        holoscan::schema::ImageT descriptor;
        descriptor.width = width;
        descriptor.height = height;
        descriptor.encoding = holoscan::schema::ImageEncoding_RGB8;

        auto emitted = output.emit_tensor(std::move(*loan), descriptor,
            holoscan::EmitOptions { .capture_time = capture });
        if (!emitted && emitted.error().code == holoscan::ErrorCode::kBackpressured)
            return {};
        return emitted;
    }

private:
    hololink::operators::DropCounter mismatched_frames_;
};

} // namespace

int main(int argc, char** argv)
{
    bool headless = false;
    bool fullscreen = false;
    int64_t frame_limit = 0;
    std::string hololink_ip = hololink::env_hololink_ip(0, "192.168.0.2");
    holoscan::LogLevel log_level = holoscan::LogLevel::INFO;
    bool log_level_given = false;
    int32_t window_width = 1920;
    int32_t window_height = 1080;
    std::string title = "MAX96716A VPG";
    int32_t vpg_width = 1920;
    int32_t vpg_height = 1080;
    std::string pattern = "checkerboard";
    std::string sensor = "left";
    bool skip_setup = false;
    bool skip_reset = false;

    const struct option long_options[] = {
        { "help", no_argument, nullptr, 'h' },
        { "headless", no_argument, nullptr, 0 },
        { "fullscreen", no_argument, nullptr, 0 },
        { "frame-limit", required_argument, nullptr, 0 },
        { "configuration", required_argument, nullptr, 0 },
        { "hololink", required_argument, nullptr, 0 },
        { "log-level", required_argument, nullptr, 0 },
        { "window-height", required_argument, nullptr, 0 },
        { "window-width", required_argument, nullptr, 0 },
        { "title", required_argument, nullptr, 0 },
        { "width", required_argument, nullptr, 0 },
        { "height", required_argument, nullptr, 0 },
        { "pattern", required_argument, nullptr, 0 },
        { "sensor", required_argument, nullptr, 0 },
        { "skip-setup", no_argument, nullptr, 0 },
        { "skip-reset", no_argument, nullptr, 0 },
        { 0, 0, nullptr, 0 }
    };

    while (true) {
        int option_index = 0;
        const int c = getopt_long(argc, argv, "h", long_options, &option_index);
        if (c == -1)
            break;
        const std::string argument(optarg ? optarg : "");
        if (c == 0) {
            const std::string name = long_options[option_index].name;
            if (name == "headless")
                headless = true;
            else if (name == "fullscreen")
                fullscreen = true;
            else if (name == "frame-limit")
                frame_limit = std::stoll(argument);
            else if (name == "configuration") {
                if (!hx::check_configuration(argument))
                    return EXIT_FAILURE;
            } else if (name == "hololink")
                hololink_ip = argument;
            else if (name == "window-height")
                window_height = std::stoi(argument);
            else if (name == "window-width")
                window_width = std::stoi(argument);
            else if (name == "title")
                title = argument;
            else if (name == "width")
                vpg_width = std::stoi(argument);
            else if (name == "height")
                vpg_height = std::stoi(argument);
            else if (name == "pattern")
                pattern = argument;
            else if (name == "sensor")
                sensor = argument;
            else if (name == "skip-setup")
                skip_setup = true;
            else if (name == "skip-reset")
                skip_reset = true;
            else if (name == "log-level") {
                log_level_given = true;
                const auto level = hx::parse_log_level(argument);
                if (!level)
                    throw std::runtime_error("Unhandled log level: " + argument);
                log_level = *level;
            }
        } else if (c == 'h') {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl
                      << "Stream the MAX96716A internal video pattern generator." << std::endl;
            return EXIT_SUCCESS;
        }
    }

    // Fail fast, before touching any hardware, rather than letting a bad resolution
    // surface as a cryptic kInvalidArgument (or garbage register writes in
    // configure_vpg()) deep inside the app. Checked separately from the byte-count
    // cap below: a negative width paired with a negative height would otherwise wrap
    // around in the unsigned multiplication below and silently pass that check.
    if (vpg_width <= 0 || vpg_height <= 0) {
        std::cerr << "Invalid resolution " << vpg_width << "x" << vpg_height
                  << ": --width/--height must both be positive." << std::endl;
        return EXIT_FAILURE;
    }
    const std::size_t rgb_frame_bytes
        = static_cast<std::size_t>(vpg_width) * static_cast<std::size_t>(vpg_height) * 3U;
    if (rgb_frame_bytes > hololink::module::operators::LinuxReceiverOp::kMaxFrameBytes) {
        std::cerr << "Requested resolution " << vpg_width << "x" << vpg_height << " ("
                  << rgb_frame_bytes << " bytes/frame) exceeds the maximum supported frame size ("
                  << hololink::module::operators::LinuxReceiverOp::kMaxFrameBytes
                  << " bytes). Reduce --width/--height." << std::endl;
        return EXIT_FAILURE;
    }

    // With managed signals, Ctrl+C stops the session cooperatively, so the board is stopped on
    // the way out. CUDA and the Adapter start threads, so the signals come first.
    return hx::run_with_managed_signals([&](holoscan::ManagedShutdownSignals& signals) {
        if (log_level_given) {
            // Without --log-level, the module's logger keeps the level HOLOLINK_LOG_LEVEL chose.
            hololink::module::operators::set_log_level(log_level);
        } else {
            holoscan::set_log_level(log_level);
        }

        CUdevice cu_device;
        std::shared_ptr<hololink::module::HololinkInterfaceV1> holo_iface;

        try {
            // CUDA section
            CudaCheck(cuInit(0));
            int cu_device_ordinal = 0;
            CudaCheck(cuDeviceGet(&cu_device, cu_device_ordinal));
            CUcontext cu_context;
            CudaCheck(cuDevicePrimaryCtxRetain(&cu_context, cu_device));
            cuCtxSetCurrent(cu_context);

            // Map sensor arg to SIF index (matches Python: left=0, right=1).
            int64_t sif = (sensor == "right") ? 1 : 0;

            auto& adapter = hololink::module::Adapter::get_adapter();
            hololink::module::EnumerationMetadata metadata
                = adapter.wait_for_channel(hololink_ip, std::chrono::seconds(30));
            hololink::module::EnumerationMetadata sif_metadata = metadata;
            adapter.use_sensor(sif_metadata, sif);

            holo_iface = hololink::module::HololinkInterfaceV1::get_service(metadata);
            holo_iface->start();
            auto board = hololink::module::taurotech_da326::TauroTechDa326InterfaceV1::get_service(metadata);

            auto legacy_holo = board->hololink();
            auto deserializer = std::make_shared<Max96716a>(legacy_holo);

            if (!skip_setup) {
                if (!skip_reset) {
                    holo_iface->reset();
                }

                board->release_reset();
                board->power_cycle();
                board->check_power();
                board->setup_clock();

                // Deser sanity check
                uint8_t dev_id = deserializer->get_register(Max96716a::DEV_ID_REG);
                if (dev_id != Max96716a::DEV_ID) {
                    throw std::runtime_error(fmt::format(
                        "Deserializer mismatch: expected 0x{:02X}, got 0x{:02X}",
                        Max96716a::DEV_ID, dev_id));
                }

                // MIPI output configuration — phy_2x4, MIPI_TX1/2 lane counts.
                deserializer->configure_video_pipe();

                // force_csi_out
                deserializer->set_register(0x0330, 0x84);
                configure_vpg(*deserializer, vpg_width, vpg_height, pattern);
            }

            // RGB888 = 3 bytes/pixel; pad generously for CSI-2 per-line framing overhead.
            const size_t frame_size = static_cast<size_t>(vpg_width) * 3 * vpg_height
                + 16 * vpg_height + 4096;
            const std::string window_title = title + " - " + sensor;

            // Build the graph.
            holoscan::Graph graph { "linux_max96716a_player" };
            holoscan::CompileOptions compile_options;

            holoscan::visualization::LayerStack layers;
            layers.add_image_layer("video",
                holoscan::visualization::ImageLayerDesc {
                    .format = holoscan::visualization::ImageFormat::kR8G8B8Unorm,
                    .max_width = static_cast<uint32_t>(vpg_width),
                    .max_height = static_cast<uint32_t>(vpg_height),
                });

            auto viz = hx::add_holoviz(graph, "holoviz", std::move(layers),
                hx::DisplayConfig {
                    .title = window_title,
                    .width = static_cast<uint32_t>(window_width),
                    .height = static_cast<uint32_t>(window_height),
                    .fullscreen = fullscreen,
                    .headless = headless,
                },
                compile_options, cu_device_ordinal);

            // VPG has no per-stream device to start/stop; pass no-op lambdas.
            auto receiver = graph.op<hololink::module::operators::LinuxReceiverOp>(
                "receiver",
                sif_metadata,
                [] {},
                [] {});
            receiver.set_params(hololink::module::operators::LinuxReceiverParams {
                .frame_context = cu_context,
                .frame_size = frame_size,
                // Stop after --frame-limit frames, as HSDK4's CountCondition on the receiver did.
                .frame_limit = frame_limit,
                .frame_limit_ends_run = true,
            });

            auto reshape = graph.op<RgbReshapeOp>("reshape");
            reshape.set_params(RgbReshapeOpParams {
                .width = vpg_width,
                .height = vpg_height,
            });

            graph.add_flow(receiver->tensor, reshape->input_tensor, hx::pipeline_edge());
            graph.add_flow(receiver->metadata, reshape->input_metadata, hx::pipeline_edge());
            graph.add_flow(reshape->output, viz->image_layer_input(0), hx::display_edge());

            graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

            compile_options.deployment.bind_tensor_output_device(
                holoscan::TensorOutputDevicePlacement {
                    .operator_path = "receiver",
                    .output_port = "tensor",
                    .device = holoscan::DeviceId { cu_device_ordinal },
                });
            compile_options.deployment.bind_tensor_output_device(
                holoscan::TensorOutputDevicePlacement {
                    .operator_path = "reshape",
                    .output_port = "output",
                    .device = holoscan::DeviceId { cu_device_ordinal },
                });

            const auto plan = holoscan::compile(graph, std::move(compile_options));
            if (!plan.ok()) {
                hx::print_compile_failure(plan);
                holo_iface->stop();
                CudaCheck(cuDevicePrimaryCtxRelease(cu_device));
                return EXIT_FAILURE;
            }

            auto session = holoscan::run_async(plan, signals);
            session.wait();
            const bool failed = hx::report_session(session, plan);

            holo_iface->stop();
            CudaCheck(cuDevicePrimaryCtxRelease(cu_device));
            return failed ? EXIT_FAILURE : EXIT_SUCCESS;
        } catch (const std::exception& e) {
            if (holo_iface) {
                holo_iface->stop();
            }
            HOLOSCAN_LOG_ERROR("Application failed: {}", e.what());
            return EXIT_FAILURE;
        }
    });
}
