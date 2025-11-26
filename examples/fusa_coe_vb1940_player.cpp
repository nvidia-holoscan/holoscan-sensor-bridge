/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/bayer_demosaic/bayer_demosaic.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include <hololink/common/holoargs.hpp>
#include <hololink/core/logging.hpp>
#include <hololink/core/networking.hpp>
#include <hololink/operators/fusa_coe_capture/fusa_coe_capture.hpp>
#include <hololink/operators/image_processor/image_processor.hpp>
#include <hololink/operators/packed_format_converter/packed_format_converter.hpp>
#include <hololink/sensors/camera/vb1940/native_vb1940_sensor.hpp>

hololink::core::MacAddress parse_mac_address(const std::string& mac_str)
{
    hololink::core::MacAddress mac;
    std::istringstream iss(mac_str);
    std::string byte_str;
    int i = 0;

    while (std::getline(iss, byte_str, ':')) {
        if (i >= 6)
            throw std::invalid_argument("Too many bytes in MAC address");

        if (byte_str.size() != 2)
            throw std::invalid_argument("Each byte must be exactly two hex digits");

        uint16_t byte;
        std::istringstream hex_stream(byte_str);
        hex_stream >> std::hex >> byte;
        if (hex_stream.fail())
            throw std::invalid_argument("Invalid hex in MAC address");

        mac[i++] = static_cast<uint8_t>(byte);
    }

    if (i != 6)
        throw std::invalid_argument("MAC address must have exactly 6 bytes");

    return mac;
}

class Application : public holoscan::Application {
public:
    Application(
        bool headless,
        bool fullscreen,
        std::vector<hololink::DataChannel>& hololink_channels,
        std::vector<std::shared_ptr<hololink::sensors::NativeVb1940Sensor>>& cameras,
        hololink::sensors::vb1940_mode::Mode camera_mode,
        uint32_t timeout,
        int frame_limit)
        : headless_(headless)
        , fullscreen_(fullscreen)
        , hololink_channels_(hololink_channels)
        , cameras_(cameras)
        , camera_mode_(camera_mode)
        , timeout_(timeout)
        , frame_limit_(frame_limit)
    {
        // Because we have stereo camera paths going into the same visualizer, don't
        // raise an error when each path presents metadata with the same names.
        enable_metadata(true);
        metadata_policy(holoscan::MetadataPolicy::kReject);
    }

    void compose() override
    {
        using namespace holoscan;

        auto metadata = hololink_channels_[0].enumeration_metadata();
        auto interface = metadata.get<std::string>("interface").value_or("");
        auto mac_str = metadata.get<std::string>("mac_id").value_or("");
        auto mac_bytes = parse_mac_address(mac_str);
        HOLOSCAN_LOG_INFO("Using interface={}, mac={}", interface, mac_str);

        // Holoviz is used to render the image(s).
        const float view_width = 1.0f / cameras_.size();
        std::vector<ops::HolovizOp::InputSpec> specs;
        for (uint32_t i = 0; i < cameras_.size(); ++i) {
            ops::HolovizOp::InputSpec::View view;
            view.offset_x_ = view_width * i;
            view.offset_y_ = 0;
            view.width_ = view_width;
            view.height_ = 1;
            ops::HolovizOp::InputSpec spec(std::to_string(i), ops::HolovizOp::InputType::COLOR);
            spec.views_.push_back(view);
            specs.push_back(spec);
        }
        uint32_t width = 800 * cameras_.size();
        uint32_t height = 620;
        auto visualizer = make_operator<holoscan::ops::HolovizOp>(
            "holoviz",
            Arg("tensors", specs),
            Arg("fullscreen", fullscreen_),
            Arg("headless", headless_),
            Arg("width", width),
            Arg("height", height),
            Arg("framebuffer_srgb", true));

        // Create the capture/conversion pipelines for each sensor.
        for (uint32_t i = 0; i < cameras_.size(); ++i) {
            std::shared_ptr<Condition> condition;
            if (frame_limit_) {
                condition = make_condition<CountCondition>("count", frame_limit_);
            } else {
                condition = make_condition<BooleanCondition>("ok", true);
            }

            cameras_[i]->set_mode(camera_mode_);
            auto pixel_format = cameras_[i]->get_pixel_format();
            auto bayer_format = cameras_[i]->get_bayer_format();
            auto width = cameras_[i]->get_width();
            auto height = cameras_[i]->get_height();
            auto name = std::to_string(i);

            // Capture from Fusa.
            auto fusa_coe_capture = make_operator<hololink::operators::FusaCoeCaptureOp>(
                fmt::format("fusa_coe_capture_{}", name),
                Arg("interface", interface),
                Arg("mac_addr", std::vector<uint8_t>(mac_bytes.begin(), mac_bytes.end())),
                Arg("hololink_channel", &hololink_channels_[i]),
                Arg("out_tensor_name", name),
                Arg("timeout", timeout_),
                Arg("device_start", std::function<void()>([this, i] { cameras_[i]->start(); })),
                Arg("device_stop", std::function<void()>([this, i] { cameras_[i]->stop(); })),
                condition);
            cameras_[i]->configure_converter(fusa_coe_capture);

            // Convert packed RAW to 16-bit Bayer.
            size_t size = width * height * 2; // 16-bit Bayer
            const int32_t storage_type_device_memory = 1;
            const size_t num_blocks = 2;
            auto packed_format_converter_pool = make_resource<holoscan::BlockMemoryPool>(
                fmt::format("packed_format_converter_pool_{}", name),
                storage_type_device_memory, size, num_blocks);

            auto packed_format_converter = make_operator<hololink::operators::PackedFormatConverterOp>(
                fmt::format("packed_format_converter_{}", name),
                Arg("allocator", packed_format_converter_pool),
                Arg("in_tensor_name", name));
            fusa_coe_capture->configure_converter(*packed_format_converter.get());

            // Perform basic ISP operations.
            auto image_processor = make_operator<hololink::operators::ImageProcessorOp>(
                fmt::format("image_processor_{}", name),
                Arg("optical_black", 0),
                Arg("bayer_format", static_cast<int>(bayer_format)),
                Arg("pixel_format", static_cast<int>(pixel_format)));

            // Bayer demosaic to RGBA buffer.
            size = width * height * 2 * 4; // 16-bit RGBA
            auto demosaic_pool = make_resource<holoscan::BlockMemoryPool>(
                fmt::format("demosaic_pool_{}", name),
                storage_type_device_memory, size, num_blocks);

            auto bayer_demosaic = make_operator<holoscan::ops::BayerDemosaicOp>(
                fmt::format("bayer_demosaic_{}", name),
                Arg("pool", demosaic_pool),
                Arg("generate_alpha", true),
                Arg("alpha_value", 65535),
                Arg("bayer_grid_pos", static_cast<int>(bayer_format)),
                Arg("out_tensor_name", name));

            add_flow(fusa_coe_capture, packed_format_converter, { { "output", "input" } });
            add_flow(packed_format_converter, image_processor, { { "output", "input" } });
            add_flow(image_processor, bayer_demosaic, { { "output", "receiver" } });
            add_flow(bayer_demosaic, visualizer, { { "transmitter", "receivers" } });
        }
    }

private:
    bool headless_;
    bool fullscreen_;
    std::vector<hololink::DataChannel>& hololink_channels_;
    std::vector<std::shared_ptr<hololink::sensors::NativeVb1940Sensor>>& cameras_;
    hololink::sensors::vb1940_mode::Mode camera_mode_;
    uint32_t timeout_;
    int frame_limit_;
};

int main(int argc, char** argv)
{
    using namespace hololink::args;
    OptionsDescription options_description("Allowed options");

    hololink::logging::hsb_log_level = hololink::logging::HSB_LOG_LEVEL_INFO;

    // clang-format off
    options_description.add_options()
        ("camera-mode", value<int>()->default_value(static_cast<int>(hololink::sensors::vb1940_mode::VB1940_MODE_2560X1984_30FPS)), "VB1940 mode (default: 0)")
        ("frame-limit", value<int>()->default_value(0), "Exit after receiving this many frames")
        ("fullscreen", bool_switch()->default_value(false), "Run in fullscreen mode")
        ("headless", bool_switch()->default_value(false), "Run in headless mode")
        ("hololink", value<std::string>()->default_value("192.168.0.2"), "IP address of Hololink board")
        ("sensor", value<uint32_t>()->default_value(-1), "Sensor to use (0 or 1, or -1 (default) for stereo mode)")
        ("timeout", value<int>()->default_value(1500), "Capture request timeout, in milliseconds")
        ;
    // clang-format on

    auto variables_map = Parser().parse_command_line(argc, argv, options_description);
    auto hololink = variables_map["hololink"].as<std::string>();
    auto sensor = variables_map["sensor"].as<uint32_t>();

    try {
        // Get a handle to the Hololink device.
        auto channel_metadata = hololink::Enumerator::find_channel(hololink);

        std::vector<hololink::Metadata> channel_metadatas;
        std::vector<hololink::DataChannel> hololink_channels;
        if (sensor == -1) {
            // Create separate channels for each sensor.
            channel_metadatas.push_back(hololink::Metadata(channel_metadata));
            hololink::DataChannel::use_sensor(channel_metadatas[0], 0);
            hololink_channels.push_back(hololink::DataChannel(channel_metadatas[0]));

            channel_metadatas.push_back(hololink::Metadata(channel_metadata));
            hololink::DataChannel::use_sensor(channel_metadatas[1], 1);
            hololink_channels.push_back(hololink::DataChannel(channel_metadatas[1]));
        } else if (sensor == 0 || sensor == 1) {
            channel_metadatas.push_back(hololink::Metadata(channel_metadata));
            hololink::DataChannel::use_sensor(channel_metadatas[0], sensor);
            hololink_channels.push_back(hololink::DataChannel(channel_metadatas[0]));
        } else {
            HOLOSCAN_LOG_ERROR("sensor value must be 0 or 1 (given {})", sensor);
            return -1;
        }

        // Start the HSB.
        std::shared_ptr<hololink::Hololink> hololink = hololink_channels[0].hololink();
        hololink->start();
        hololink->reset();

        // Convert camera_mode to enum.
        auto camera_mode = static_cast<hololink::sensors::vb1940_mode::Mode>(variables_map["camera-mode"].as<int>());

        // Get handles to and configure the camera(s).
        std::vector<std::shared_ptr<hololink::sensors::NativeVb1940Sensor>> cameras;
        for (uint32_t i = 0; i < hololink_channels.size(); i++) {
            auto camera = std::make_shared<hololink::sensors::NativeVb1940Sensor>(hololink_channels[i]);
            if (i == 0) {
                camera->setup_clock();
            }
            camera->configure(camera_mode);
            cameras.push_back(camera);
        }

        // Create and run the application.
        auto app = std::make_unique<Application>(
            variables_map["headless"].as<bool>(),
            variables_map["fullscreen"].as<bool>(),
            hololink_channels,
            cameras,
            camera_mode,
            variables_map["timeout"].as<int>(),
            variables_map["frame-limit"].as<int>());
        app->run();

        hololink->stop();

        return 0;
    } catch (const std::exception& e) {
        HOLOSCAN_LOG_ERROR("Application failed: {}", e.what());
        return -1;
    }
}
