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

#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gxf/std/allocator.hpp>
#include <gxf/std/tensor.hpp>

#include "hololink/module/adapter.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/hololink.hpp"
#include "hololink/module/operators/fusa_coe_capture_op.hpp"
#include "hololink/module/taurotech_da326/taurotech_da326.hpp"
#include "max96716a.hpp"
#include <hololink/common/holoargs.hpp>
#include <hololink/common/tools.hpp>
#include <hololink/core/logging.hpp>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

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

    write24(deserializer, 0x0245, static_cast<uint32_t>(v_active * h_total));
    write24(deserializer, 0x0248, static_cast<uint32_t>(v_blank * h_total));
    write16(deserializer, 0x024E, 44);
    write16(deserializer, 0x0250, static_cast<uint32_t>(h_total - 44));
    write16(deserializer, 0x0252, static_cast<uint32_t>(v_total));
    write16(deserializer, 0x0257, static_cast<uint32_t>(h_active));
    write16(deserializer, 0x0259, static_cast<uint32_t>(h_blank));
    write16(deserializer, 0x025B, static_cast<uint32_t>(v_active));

    uint8_t patgen_mode;
    if (pattern == "checkerboard") {
        deserializer.set_register(0x025E, 0x00);
        deserializer.set_register(0x025F, 0x00);
        deserializer.set_register(0x0260, 0xFF);
        deserializer.set_register(0x0261, 0xFF);
        deserializer.set_register(0x0262, 0x00);
        deserializer.set_register(0x0263, 0x00);
        deserializer.set_register(0x0264, 16);
        deserializer.set_register(0x0265, 16);
        deserializer.set_register(0x0266, 8);
        patgen_mode = 0b01;
    } else if (pattern == "gradient") {
        deserializer.set_register(0x025D, 0x01);
        patgen_mode = 0b10;
    } else {
        throw std::runtime_error("Unknown VPG pattern: " + pattern);
    }

    deserializer.set_register(0x0240, 0xE3);
    deserializer.set_register(0x0241, static_cast<uint8_t>(patgen_mode << 4));
}

/// Reshapes the flat uint8 receiver buffer into a (H, W, 3) device tensor so
/// Holoviz recognises it as a color image.
class RgbReshapeOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(RgbReshapeOp)

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

/// FuSa CoE capture requires a device with start()/stop(); the VPG has none.
class VpgDevice {
public:
    void start() { }
    void stop() { }
};

class Application : public holoscan::Application {
public:
    Application(bool headless,
        hololink::module::EnumerationMetadata sif_metadata,
        const std::string& interface,
        uint32_t timeout,
        int32_t window_height, int32_t window_width,
        size_t frame_size,
        int64_t frame_limit,
        int32_t vpg_width, int32_t vpg_height,
        const std::string& window_title)
        : headless_(headless)
        , sif_metadata_(std::move(sif_metadata))
        , interface_(interface)
        , timeout_(timeout)
        , window_height_(window_height)
        , window_width_(window_width)
        , frame_size_(frame_size)
        , frame_limit_(frame_limit)
        , vpg_width_(vpg_width)
        , vpg_height_(vpg_height)
        , window_title_(window_title)
        , device_(std::make_shared<VpgDevice>())
    {
    }

    void compose() override
    {
        using namespace holoscan;

        std::shared_ptr<Condition> condition;
        if (frame_limit_) {
            condition = make_condition<CountCondition>("count", frame_limit_);
        } else {
            condition = make_condition<BooleanCondition>("ok", true);
        }

        auto fusa_coe_capture = make_operator<hololink::module::operators::FusaCoeCaptureOp>(
            "fusa_coe_capture",
            Arg("enumeration_metadata", sif_metadata_),
            Arg("interface", interface_.empty() ? sif_metadata_.get<std::string>("interface", "") : interface_),
            Arg("mac_addr", [&]() -> std::vector<uint8_t> { return parse_mac_address(sif_metadata_.get<std::string>("mac_id", "")); }()),
            Arg("timeout", timeout_),
            Arg("device_start", std::function<void()>([this] { device_->start(); })),
            Arg("device_stop", std::function<void()>([this] { device_->stop(); })),
            condition);
        fusa_coe_capture->configure_frame_size(static_cast<uint32_t>(frame_size_));

        auto reshape_pool = make_resource<BlockMemoryPool>("reshape_pool",
            /*storage_type=*/1, // device memory
            static_cast<size_t>(vpg_width_) * vpg_height_ * 3u,
            /*num_blocks=*/2);
        auto reshape_op = make_operator<RgbReshapeOp>("reshape",
            Arg("allocator", reshape_pool),
            Arg("width", vpg_width_),
            Arg("height", vpg_height_));

        std::vector<ops::HolovizOp::InputSpec> input_specs;
        input_specs.emplace_back("", ops::HolovizOp::InputType::COLOR);
        auto visualizer = make_operator<ops::HolovizOp>("holoviz",
            Arg("framebuffer_srgb", true),
            Arg("headless", headless_),
            Arg("height", window_height_),
            Arg("width", window_width_),
            Arg("window_title", window_title_),
            Arg("tensors", input_specs));

        add_flow(fusa_coe_capture, reshape_op, { { "output", "input" } });
        add_flow(reshape_op, visualizer, { { "output", "receivers" } });
    }

private:
    const bool headless_;
    hololink::module::EnumerationMetadata sif_metadata_;
    const std::string interface_;
    const uint32_t timeout_;
    const int32_t window_height_;
    const int32_t window_width_;
    const size_t frame_size_;
    const int64_t frame_limit_;
    const int32_t vpg_width_;
    const int32_t vpg_height_;
    const std::string window_title_;
    const std::shared_ptr<VpgDevice> device_;
};

} // anonymous namespace

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

    const auto hololink_ip = variables_map["hololink"].as<std::string>();
    const auto sensor = variables_map["sensor"].as<std::string>();
    const auto pattern = variables_map["pattern"].as<std::string>();
    const auto title = variables_map["title"].as<std::string>();
    const auto interface = variables_map["interface"].as<std::string>();
    const auto headless = variables_map["headless"].as<bool>();
    const auto skip_setup = variables_map["skip-setup"].as<bool>();
    const auto skip_reset = variables_map["skip-reset"].as<bool>();
    const int32_t window_width = variables_map["window-width"].as<int>();
    const int32_t window_height = variables_map["window-height"].as<int>();
    const int32_t vpg_width = variables_map["width"].as<int>();
    const int32_t vpg_height = variables_map["height"].as<int>();
    const uint32_t timeout = static_cast<uint32_t>(variables_map["timeout"].as<int>());
    const int64_t frame_limit = variables_map["frame-limit"].as<int>();

    try {
        const int64_t sif = (sensor == "right") ? 1 : 0;

        auto& adapter = hololink::module::Adapter::get_adapter();
        hololink::module::EnumerationMetadata adapter_metadata
            = adapter.wait_for_channel(hololink_ip, std::chrono::seconds(30));

        hololink::module::EnumerationMetadata sif_metadata(adapter_metadata);
        adapter.use_sensor(sif_metadata, sif);

        auto holo_iface = hololink::module::HololinkInterfaceV1::get_service(adapter_metadata);
        holo_iface->start();
        auto board = hololink::module::taurotech_da326::TauroTechDa326InterfaceV1::get_service(adapter_metadata);
        auto legacy_holo = board->hololink();
        auto deserializer = std::make_shared<Max96716a>(legacy_holo);

        const size_t frame_size = static_cast<size_t>(vpg_width) * 3 * vpg_height
            + 16 * vpg_height + 4096;
        const std::string window_title = title + " - " + sensor;

        auto application = holoscan::make_application<Application>(headless,
            sif_metadata, interface, timeout,
            window_height, window_width, frame_size, frame_limit,
            vpg_width, vpg_height, window_title);

        try {
            if (!skip_setup) {
                if (!skip_reset) {
                    holo_iface->reset();
                }
                board->release_reset();
                board->power_cycle();
                board->check_power();
                board->setup_clock();

                uint8_t dev_id = deserializer->get_register(Max96716a::DEV_ID_REG);
                if (dev_id != Max96716a::DEV_ID) {
                    throw std::runtime_error(fmt::format(
                        "Deserializer mismatch: expected 0x{:02X}, got 0x{:02X}",
                        Max96716a::DEV_ID, dev_id));
                }

                deserializer->configure_video_pipe();
                deserializer->set_register(0x0330, 0x84); // force_csi_out
                configure_vpg(*deserializer, vpg_width, vpg_height, pattern);
            }

            application->run();
        } catch (...) {
            holo_iface->stop();
            throw;
        }
        holo_iface->stop();
        return 0;
    } catch (const std::exception& e) {
        HOLOSCAN_LOG_ERROR("Application failed: {}", e.what());
        return -1;
    }
}
