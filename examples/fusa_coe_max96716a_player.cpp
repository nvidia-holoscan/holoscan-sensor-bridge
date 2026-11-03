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

// MAX96716A internal video pattern generator (VPG) live viewer over the FuSa CoE capture
// path, ported to Holoscan SDK 5.0's Graph / Operator<Params> shape:
//   FusaCoeCaptureOp -> RgbReshapeOp -> HolovizOp
//
// Near-identical to the already-ported examples/max96716a_player.cpp (RoCE sibling) --
// same RgbReshapeOp, same VPG/TauroTechDa326 setup -- with
// hololink::module::operators::FusaCoeCaptureOp swapped in for the receiver, exactly as the
// FuSa Hawk apps already demonstrate. The VPG has no CSI/Bayer stage (the deserializer's
// built-in test pattern is RGB888 already), so FusaCoeCaptureOp runs in its opaque
// byte-blob capture mode (frame_size_bytes set, no CFA/packetizer structure) -- mirrors the
// pre-port shell's own configure_frame_size() call.

#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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
#include "hololink/module/operators/fusa_coe_capture_op.hpp"
#include "hololink/module/taurotech_da326/taurotech_da326.hpp"
#include "max96716a.hpp"
#include <hololink/common/cuda_helper.hpp>
#include <hololink/common/holoargs.hpp>
#include <hololink/common/tools.hpp>
#include <hololink/core/logging.hpp>

#include "player_support.hpp"

namespace hx = hololink::examples;

namespace {

using hololink::module::sensors::max96716a::Max96716a;

std::vector<uint8_t> parse_mac_address(const std::string& mac_str)
{
    std::vector<uint8_t> mac;
    std::istringstream iss(mac_str);
    std::string byte_str;
    while (std::getline(iss, byte_str, ':')) {
        if (mac.size() >= 6) {
            throw std::invalid_argument("Too many bytes in MAC address");
        }
        if (byte_str.size() != 2) {
            throw std::invalid_argument("Each MAC byte must be exactly two hex digits");
        }
        uint16_t byte;
        std::istringstream hex_stream(byte_str);
        hex_stream >> std::hex >> byte;
        if (hex_stream.fail()) {
            throw std::invalid_argument("Invalid hex in MAC address");
        }
        mac.push_back(static_cast<uint8_t>(byte));
    }
    if (mac.size() != 6) {
        throw std::invalid_argument("MAC address must have exactly 6 bytes");
    }
    return mac;
}

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

    write24(deserializer, 0x0245, static_cast<uint32_t>(v_active * h_total)); // VS_HIGH_2
    write24(deserializer, 0x0248, static_cast<uint32_t>(v_blank * h_total)); // VS_LOW_2
    write16(deserializer, 0x024E, 44); // HS_HIGH_1
    write16(deserializer, 0x0250, static_cast<uint32_t>(h_total - 44)); // HS_LOW_1
    write16(deserializer, 0x0252, static_cast<uint32_t>(v_total)); // HS_CNT_1
    write16(deserializer, 0x0257, static_cast<uint32_t>(h_active)); // DE_HIGH_1
    write16(deserializer, 0x0259, static_cast<uint32_t>(h_blank)); // DE_LOW_1
    write16(deserializer, 0x025B, static_cast<uint32_t>(v_active)); // DE_CNT_1

    uint8_t patgen_mode;
    if (pattern == "checkerboard") {
        deserializer.set_register(0x025E, 0x00); // CHKR_COLOR_A_L
        deserializer.set_register(0x025F, 0x00); // CHKR_COLOR_A_M
        deserializer.set_register(0x0260, 0xFF); // CHKR_COLOR_A_H
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
// Identical to examples/max96716a_player.cpp's own RgbReshapeOp.
class RgbReshapeOp : public holoscan::Operator<RgbReshapeOpParams> {
public:
    holoscan::Input<holoscan::Tensor> input_tensor { "tensor" };
    holoscan::Output<holoscan::schema::ImageT> output { "output" };

    void setup(holoscan::OperatorSpec& spec) override
    {
        static constexpr DLDataType kUInt8 { static_cast<uint8_t>(kDLUInt), 8, 1 };
        // Matches the receiver's own pool ceiling so this stage is never the tighter
        // bottleneck -- the receiver can't forward a frame larger than this anyway.
        static constexpr std::size_t kMaxBytes
            = hololink::module::operators::FusaCoeCaptureOp::kMaxFrameBytes;

        spec.input(input_tensor, "tensor").queue_depth(4U).accepts_projected_tensor();
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
        c.trigger(holoscan::OnEach { input_tensor });
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

        auto sample = input_tensor.receive();
        if (!sample)
            return holoscan::make_unexpected(std::move(sample).error());

        const holoscan::Tensor& in = sample->data;
        if (static_cast<std::size_t>(in.nbytes()) < pixels)
            return {}; // Short frame -- drop it. Holoviz can't render a partial image.

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

        holoscan::TimePoint capture = sample->metadata.capture_time();
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
};

// Runs the player on the options main() parsed.
int run(hololink::args::VariablesMap& variables_map, holoscan::ManagedShutdownSignals& signals)
{
    const auto hololink_ip = variables_map["hololink"].as<std::string>();
    const auto sensor = variables_map["sensor"].as<std::string>();
    const auto pattern = variables_map["pattern"].as<std::string>();
    const auto title = variables_map["title"].as<std::string>();
    const auto interface_arg = variables_map["interface"].as<std::string>();
    const auto headless = variables_map["headless"].as<bool>();
    const auto skip_setup = variables_map["skip-setup"].as<bool>();
    const auto skip_reset = variables_map["skip-reset"].as<bool>();
    const int32_t window_width = variables_map["window-width"].as<int>();
    const int32_t window_height = variables_map["window-height"].as<int>();
    const int32_t vpg_width = variables_map["width"].as<int>();
    const int32_t vpg_height = variables_map["height"].as<int>();
    const uint32_t timeout = static_cast<uint32_t>(variables_map["timeout"].as<int>());
    const int64_t frame_limit = variables_map["frame-limit"].as<int>();

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
    // RGB888 = 3 bytes/pixel; pad generously for CSI-2 per-line framing overhead.
    const size_t frame_size = static_cast<size_t>(vpg_width) * 3 * vpg_height
        + 16 * static_cast<size_t>(vpg_height) + 4096;
    if (frame_size > hololink::module::operators::FusaCoeCaptureOp::kMaxFrameBytes) {
        std::cerr << "Requested resolution " << vpg_width << "x" << vpg_height << " ("
                  << frame_size << " bytes/frame) exceeds the maximum supported frame size ("
                  << hololink::module::operators::FusaCoeCaptureOp::kMaxFrameBytes
                  << " bytes). Reduce --width/--height." << std::endl;
        return EXIT_FAILURE;
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

        hololink::module::EnumerationMetadata adapter_metadata
            = hololink::module::Adapter::get_adapter().wait_for_channel(hololink_ip,
                std::chrono::seconds(30));

        hololink::module::EnumerationMetadata sif_metadata(adapter_metadata);
        hololink::module::Adapter::get_adapter().use_sensor(sif_metadata, sif);

        holo_iface = hololink::module::HololinkInterfaceV1::get_service(adapter_metadata);
        holo_iface->start();
        auto board = hololink::module::taurotech_da326::TauroTechDa326InterfaceV1::get_service(adapter_metadata);
        // sensor_holo provides I2C access for sensor/deserializer setup;
        // sensors will be migrated to HololinkInterfaceV1::get_i2c() separately.
        auto sensor_holo = board->hololink();
        auto deserializer = std::make_shared<Max96716a>(sensor_holo);

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

        const std::string interface = interface_arg.empty()
            ? sif_metadata.get<std::string>("interface", "")
            : interface_arg;
        const auto mac_str = sif_metadata.get<std::string>("mac_id", "");
        const auto mac_bytes = parse_mac_address(mac_str);

        const std::string window_title = title + " - " + sensor;

        // Build the graph.
        holoscan::Graph graph { "fusa_coe_max96716a_player" };
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
                .headless = headless,
            },
            compile_options, cu_device_ordinal);

        // VPG has no per-stream device to start/stop; pass no-op lambdas.
        auto capture = graph.op<hololink::module::operators::FusaCoeCaptureOp>("capture");
        capture.set_params(hololink::module::operators::FusaCoeCaptureParams {
            .metadata = sif_metadata,
            .device_start = [] {},
            .device_stop = [] {},
            .interface = interface,
            .mac_addr = mac_bytes,
            .timeout = timeout,
            .frame_size_bytes = static_cast<uint32_t>(frame_size),
            // Stop after --frame-limit frames, as HSDK4's CountCondition on FusaCoeCaptureOp did.
            .frame_limit = frame_limit,
            .frame_limit_ends_run = true,
            // Names this camera in the capture's stall messages.
            .name = std::string(capture.name()),
        });

        auto reshape = graph.op<RgbReshapeOp>("reshape");
        reshape.set_params(RgbReshapeOpParams {
            .width = vpg_width,
            .height = vpg_height,
        });

        graph.add_flow(capture->tensor, reshape->input_tensor, hx::pipeline_edge());
        graph.add_flow(reshape->output, viz->image_layer_input(0), hx::display_edge());

        graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

        compile_options.deployment.bind_tensor_output_device(
            holoscan::TensorOutputDevicePlacement {
                .operator_path = "capture",
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
            return -1;
        }

        auto session = holoscan::run_async(plan, signals);
        session.wait();
        const bool failed = hx::report_session(session, plan);

        holo_iface->stop();
        CudaCheck(cuDevicePrimaryCtxRelease(cu_device));
        return failed ? -1 : 0;
    } catch (const std::exception& e) {
        if (holo_iface) {
            holo_iface->stop();
        }
        HOLOSCAN_LOG_ERROR("Application failed: {}", e.what());
        return -1;
    }
}

} // namespace

int main(int argc, char** argv)
{
    using namespace hololink::args;
    OptionsDescription options_description("Allowed options");

    // clang-format off
    options_description.add_options()
        ("frame-limit", value<int>()->default_value(0), "Exit after this many frames")
        ("headless", bool_switch()->default_value(false), "Run in headless mode")
        ("hololink", value<std::string>()->default_value(hololink::env_hololink_ip(0, "192.168.0.2")), "IP address of Hololink board")
        ("interface", value<std::string>()->default_value(""), "Ethernet interface (empty = auto from enumeration)")
        ("timeout", value<int>()->default_value(1500), "Capture request timeout, in milliseconds")
        ("window-height", value<int>()->default_value(1080), "Window height in pixels")
        ("window-width", value<int>()->default_value(1920), "Window width in pixels")
        ("title", value<std::string>()->default_value("MAX96716A VPG (FuSa CoE)"), "Window title prefix")
        ("width", value<int>()->default_value(1920), "VPG output width in pixels")
        ("height", value<int>()->default_value(1080), "VPG output height in pixels")
        ("pattern", value<std::string>()->default_value("checkerboard"), "VPG pattern: checkerboard or gradient")
        ("sensor", value<std::string>()->default_value("left"), "Which SIF to capture: left or right")
        ("skip-setup", bool_switch()->default_value(false), "Skip board / deserializer initialization")
        ("skip-reset", bool_switch()->default_value(false), "Skip the Hololink reset step")
        ;
    // clang-format on

    auto variables_map = Parser().parse_command_line(argc, argv, options_description);

    // With managed signals, Ctrl+C stops the session cooperatively, so the board is stopped on
    // the way out.
    return hx::run_with_managed_signals(
        [&](holoscan::ManagedShutdownSignals& signals) { return run(variables_map, signals); });
}
