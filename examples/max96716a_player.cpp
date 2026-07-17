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

#include <getopt.h>

#include <cuda_runtime.h>

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gxf/std/allocator.hpp>
#include <gxf/std/tensor.hpp>

#include "hololink/module/adapter.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/hololink.hpp"
#include "hololink/module/operators/roce_receiver_op.hpp"
#include "hololink/module/taurotech_da326/taurotech_da326.hpp"
#include "max96716a.hpp"
#include <hololink/common/cuda_helper.hpp>
#include <hololink/common/tools.hpp>
#include <hololink/core/logging.hpp>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

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

/// Reshapes the flat uint8 receiver buffer into a (H, W, 3) device tensor so
/// Holoviz recognises it as a color image.
class RgbReshapeOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(RgbReshapeOp)
    RgbReshapeOp() = default;

    void setup(holoscan::OperatorSpec& spec) override
    {
        spec.input<holoscan::gxf::Entity>("input");
        spec.output<holoscan::gxf::Entity>("output");
        spec.param(allocator_, "allocator", "Allocator", "Allocator for output tensor.");
        spec.param(width_, "width", "Width", "Image width in pixels.");
        spec.param(height_, "height", "Height", "Image height in pixels.");
    }

    void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
        holoscan::ExecutionContext& context) override
    {
        auto maybe_entity = op_input.receive<holoscan::gxf::Entity>("input");
        if (!maybe_entity) {
            throw std::runtime_error("RgbReshapeOp: failed to receive input");
        }
        auto& in_entity = static_cast<nvidia::gxf::Entity&>(maybe_entity.value());
        auto maybe_in_tensor = in_entity.get<nvidia::gxf::Tensor>();
        if (!maybe_in_tensor) {
            throw std::runtime_error("RgbReshapeOp: input tensor not found");
        }
        auto in_tensor = maybe_in_tensor.value();

        const size_t pixels = static_cast<size_t>(width_.get()) * static_cast<size_t>(height_.get()) * 3u;
        if (in_tensor->size() < pixels) {
            return;
        }

        auto allocator_handle = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
            fragment()->executor().context(), allocator_.get()->gxf_cid());
        if (!allocator_handle) {
            throw std::runtime_error("RgbReshapeOp: failed to get allocator handle");
        }

        nvidia::gxf::Shape shape { static_cast<int32_t>(height_.get()),
            static_cast<int32_t>(width_.get()), 3 };
        auto out_message = CreateTensorMap(context.context(), allocator_handle.value(),
            { { std::string(""), nvidia::gxf::MemoryStorageType::kDevice, shape,
                nvidia::gxf::PrimitiveType::kUnsigned8, 0,
                nvidia::gxf::ComputeTrivialStrides(shape,
                    nvidia::gxf::PrimitiveTypeSize(nvidia::gxf::PrimitiveType::kUnsigned8)) } },
            false);
        if (!out_message) {
            throw std::runtime_error("RgbReshapeOp: failed to allocate output");
        }
        auto out_tensor = out_message.value().get<nvidia::gxf::Tensor>("");
        if (!out_tensor) {
            throw std::runtime_error("RgbReshapeOp: failed to get output tensor");
        }

        cudaError_t err = cudaMemcpy(out_tensor.value()->pointer(), in_tensor->pointer(),
            pixels, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error(fmt::format("RgbReshapeOp: cudaMemcpy failed: {}",
                cudaGetErrorString(err)));
        }

        auto result = holoscan::gxf::Entity(std::move(out_message.value()));
        op_output.emit(result, "output");
    }

private:
    holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
    holoscan::Parameter<int32_t> width_;
    holoscan::Parameter<int32_t> height_;
};

class HoloscanApplication : public holoscan::Application {
public:
    HoloscanApplication(bool headless, bool fullscreen, CUcontext cuda_context,
        int cuda_device_ordinal, hololink::module::EnumerationMetadata sif_metadata,
        size_t frame_size, int64_t frame_limit,
        int32_t window_height, int32_t window_width,
        const std::string& window_title,
        int32_t vpg_width, int32_t vpg_height)
        : headless_(headless)
        , fullscreen_(fullscreen)
        , cuda_context_(cuda_context)
        , cuda_device_ordinal_(cuda_device_ordinal)
        , sif_metadata_(std::move(sif_metadata))
        , frame_size_(frame_size)
        , frame_limit_(frame_limit)
        , window_height_(window_height)
        , window_width_(window_width)
        , window_title_(window_title)
        , vpg_width_(vpg_width)
        , vpg_height_(vpg_height)
    {
    }

    void compose() override
    {
        std::shared_ptr<holoscan::Condition> condition;
        if (frame_limit_) {
            condition = make_condition<holoscan::CountCondition>("count", frame_limit_);
        } else {
            condition = make_condition<holoscan::BooleanCondition>("ok", true);
        }

        // VPG has no per-stream device to start/stop; pass no-op lambdas.
        auto receiver_operator = make_operator<hololink::module::operators::RoceReceiverOp>("receiver",
            condition,
            holoscan::Arg("enumeration_metadata", sif_metadata_),
            holoscan::Arg("frame_context", cuda_context_),
            holoscan::Arg("frame_size", frame_size_),
            holoscan::Arg("device_start", std::function<void()>([] {})),
            holoscan::Arg("device_stop", std::function<void()>([] {})));

        // Reshape the flat receiver buffer into (H, W, 3) so Holoviz recognises it.
        auto reshape_pool = make_resource<holoscan::BlockMemoryPool>("reshape_pool",
            1,
            static_cast<size_t>(vpg_width_) * vpg_height_ * 3u,
            2);
        auto reshape_op = make_operator<RgbReshapeOp>("reshape",
            holoscan::Arg("allocator", reshape_pool),
            holoscan::Arg("width", vpg_width_),
            holoscan::Arg("height", vpg_height_));

        std::vector<holoscan::ops::HolovizOp::InputSpec> input_specs;
        input_specs.emplace_back("", holoscan::ops::HolovizOp::InputType::COLOR);
        auto visualizer = make_operator<holoscan::ops::HolovizOp>("holoviz",
            holoscan::Arg("framebuffer_srgb", true),
            holoscan::Arg("fullscreen", fullscreen_),
            holoscan::Arg("headless", headless_),
            holoscan::Arg("height", window_height_),
            holoscan::Arg("width", window_width_),
            holoscan::Arg("window_title", window_title_),
            holoscan::Arg("tensors", input_specs));

        add_flow(receiver_operator, reshape_op, { { "output", "input" } });
        add_flow(reshape_op, visualizer, { { "output", "receivers" } });
    }

private:
    const bool headless_;
    const bool fullscreen_;
    const CUcontext cuda_context_;
    const int cuda_device_ordinal_;
    hololink::module::EnumerationMetadata sif_metadata_;
    const size_t frame_size_;
    const int64_t frame_limit_;
    const int32_t window_height_;
    const int32_t window_width_;
    const std::string window_title_;
    const int32_t vpg_width_;
    const int32_t vpg_height_;
};

} // anonymous namespace

int main(int argc, char** argv)
{
    bool headless = false;
    bool fullscreen = false;
    int64_t frame_limit = 0;
    std::string configuration;
    std::string hololink_ip = hololink::env_hololink_ip(0, "192.168.0.2");
    holoscan::LogLevel log_level = holoscan::LogLevel::INFO;
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
            else if (name == "configuration")
                configuration = argument;
            else if (name == "hololink")
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
                if (argument == "trace" || argument == "TRACE")
                    log_level = holoscan::LogLevel::TRACE;
                else if (argument == "debug" || argument == "DEBUG")
                    log_level = holoscan::LogLevel::DEBUG;
                else if (argument == "info" || argument == "INFO")
                    log_level = holoscan::LogLevel::INFO;
                else if (argument == "warn" || argument == "WARN")
                    log_level = holoscan::LogLevel::WARN;
                else if (argument == "error" || argument == "ERROR")
                    log_level = holoscan::LogLevel::ERROR;
                else if (argument == "critical" || argument == "CRITICAL")
                    log_level = holoscan::LogLevel::CRITICAL;
                else if (argument == "off" || argument == "OFF")
                    log_level = holoscan::LogLevel::OFF;
                else
                    throw std::runtime_error("Unhandled log level: " + argument);
            }
        } else if (c == 'h') {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl
                      << "Stream the MAX96716A internal video pattern generator." << std::endl;
            return EXIT_SUCCESS;
        }
    }

    holoscan::set_log_level(log_level);

    // CUDA section
    CudaCheck(cuInit(0));
    int cu_device_ordinal = 0;
    CUdevice cu_device;
    CudaCheck(cuDeviceGet(&cu_device, cu_device_ordinal));
    CUcontext cu_context;
    CudaCheck(cuDevicePrimaryCtxRetain(&cu_context, cu_device));

    // Map sensor arg to SIF index (matches Python: left=0, right=1).
    int64_t sif = (sensor == "right") ? 1 : 0;

    hololink::module::EnumerationMetadata adapter_metadata
        = hololink::module::Adapter::get_adapter().wait_for_channel(hololink_ip,
            std::chrono::seconds(30));

    hololink::module::EnumerationMetadata sif_metadata(adapter_metadata);
    hololink::module::Adapter::get_adapter().use_sensor(sif_metadata, sif);

    auto holo_iface = hololink::module::HololinkInterfaceV1::get_service(adapter_metadata);
    holo_iface->start();
    auto board = hololink::module::taurotech_da326::TauroTechDa326InterfaceV1::get_service(adapter_metadata);
    // sensor_holo provides I2C access for sensor/deserializer setup;
    // sensors will be migrated to HololinkInterfaceV1::get_i2c() separately.
    auto sensor_holo = board->hololink();
    auto deserializer = std::make_shared<Max96716a>(sensor_holo);

    // RGB888 = 3 bytes/pixel; pad generously for CSI-2 per-line framing overhead.
    const size_t frame_size = static_cast<size_t>(vpg_width) * 3 * vpg_height
        + 16 * vpg_height + 4096;
    const std::string window_title = title + " - " + sensor;

    auto application = holoscan::make_application<HoloscanApplication>(headless, fullscreen,
        cu_context, cu_device_ordinal, sif_metadata,
        frame_size, frame_limit, window_height, window_width, window_title,
        vpg_width, vpg_height);
    application->config(configuration);

    try {
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

        application->run();
    } catch (...) {
        holo_iface->stop();
        CudaCheck(cuDevicePrimaryCtxRelease(cu_device));
        throw;
    }
    holo_iface->stop();

    CudaCheck(cuDevicePrimaryCtxRelease(cu_device));
    return 0;
}
