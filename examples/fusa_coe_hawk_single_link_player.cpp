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

#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "hawk.hpp"
#include "hololink/module/adapter.hpp"
#include "hololink/module/csi_converter.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/hololink.hpp"
#include "hololink/module/operators/fusa_coe_capture_op.hpp"
#include "hololink/module/operators/packed_format_converter_op.hpp"
#include "hololink/module/taurotech_da326/taurotech_da326.hpp"
#include "max96716a.hpp"
#include <hololink/common/holoargs.hpp>
#include <hololink/common/tools.hpp>
#include <hololink/core/logging.hpp>
#include <hololink/operators/image_processor/image_processor.hpp>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/bayer_demosaic/bayer_demosaic.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

namespace {

using hololink::module::sensors::ar0234::Ar0234;
using hololink::module::sensors::hawk::Hawk;
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

std::vector<int> sensor_arg_to_sif(const std::string& sensor)
{
    if (sensor == "left")
        return { 0 };
    if (sensor == "right")
        return { 1 };
    if (sensor == "both")
        return { 0, 1 };
    throw std::runtime_error("Invalid --sensor value: " + sensor);
}

class Application : public holoscan::Application {
public:
    Application(bool headless, bool fullscreen,
        std::vector<hololink::module::EnumerationMetadata> sif_metadatas,
        std::shared_ptr<Hawk> hawk,
        std::vector<std::shared_ptr<Ar0234>> cameras,
        hololink::module::sensors::ar0234::ar0234_mode::Mode camera_mode,
        uint32_t timeout,
        int32_t width, int32_t height,
        int64_t frame_limit,
        const std::string& interface,
        std::vector<std::string> window_titles)
        : headless_(headless)
        , fullscreen_(fullscreen)
        , sif_metadatas_(std::move(sif_metadatas))
        , hawk_(std::move(hawk))
        , cameras_(std::move(cameras))
        , camera_mode_(camera_mode)
        , timeout_(timeout)
        , width_(width)
        , height_(height)
        , frame_limit_(frame_limit)
        , interface_(interface)
        , window_titles_(std::move(window_titles))
    {
        enable_metadata(true);
        metadata_policy(holoscan::MetadataPolicy::kReject);
    }

    void compose() override
    {
        using namespace holoscan;

        auto interface = interface_.empty()
            ? sif_metadatas_[0].get<std::string>("interface", "")
            : interface_;
        auto mac_str = sif_metadatas_[0].get<std::string>("mac_id", "");
        auto mac_bytes = parse_mac_address(mac_str);
        HOLOSCAN_LOG_INFO("Using interface={} mac={}", interface, mac_str);

        for (uint32_t i = 0; i < cameras_.size(); ++i) {
            compose_pipeline(i, mac_bytes, interface);
        }
    }

private:
    void compose_pipeline(uint32_t i, const std::vector<uint8_t>& mac_bytes,
        const std::string& interface)
    {
        using namespace holoscan;

        std::shared_ptr<Condition> condition;
        if (frame_limit_) {
            condition = make_condition<CountCondition>(fmt::format("count_{}", i), frame_limit_);
        } else {
            condition = make_condition<BooleanCondition>(fmt::format("ok_{}", i), true);
        }

        cameras_[i]->set_mode(camera_mode_);
        auto pixel_format = cameras_[i]->get_pixel_format();
        auto bayer_format = cameras_[i]->get_bayer_format();
        auto width = cameras_[i]->get_width();
        auto height = cameras_[i]->get_height();
        auto name = std::to_string(i);

        auto fusa_coe_capture = make_operator<hololink::module::operators::FusaCoeCaptureOp>(
            fmt::format("fusa_coe_capture_{}", name),
            Arg("enumeration_metadata", sif_metadatas_[i]),
            Arg("out_tensor_name", name),
            Arg("timeout", timeout_),
            Arg("interface", interface),
            Arg("mac_addr", mac_bytes),
            Arg("device_start", std::function<void()>([this] { hawk_->start(); })),
            Arg("device_stop", std::function<void()>([this] { hawk_->stop(); })),
            condition);
        {
            const auto apf = static_cast<hololink::module::csi::PixelFormat>(static_cast<int>(pixel_format));
            const uint32_t start_byte = fusa_coe_capture->receiver_start_byte();
            const uint32_t transmitted = fusa_coe_capture->transmitted_line_bytes(apf, static_cast<uint32_t>(width));
            const uint32_t received = fusa_coe_capture->received_line_bytes(transmitted);
            fusa_coe_capture->configure(start_byte, received, static_cast<uint32_t>(width), static_cast<uint32_t>(height), apf, 0u);
        }

        const int32_t storage_type_device_memory = 1;
        const size_t num_blocks = 4;
        size_t size = width * height * 2; // 16-bit Bayer
        auto packed_format_converter_pool = make_resource<BlockMemoryPool>(
            fmt::format("packed_format_converter_pool_{}", name),
            storage_type_device_memory, size, num_blocks);
        auto packed_format_converter = make_operator<hololink::module::operators::PackedFormatConverterOp>(
            fmt::format("packed_format_converter_{}", name),
            Arg("allocator", packed_format_converter_pool),
            Arg("in_tensor_name", name));
        fusa_coe_capture->configure_converter(*packed_format_converter.get());

        auto image_processor = make_operator<hololink::operators::ImageProcessorOp>(
            fmt::format("image_processor_{}", name),
            Arg("optical_black", 50),
            Arg("bayer_format", static_cast<int>(bayer_format)),
            Arg("pixel_format", static_cast<int>(pixel_format)));

        size = width * height * 2 * 4; // 16-bit RGBA
        auto demosaic_pool = make_resource<BlockMemoryPool>(
            fmt::format("demosaic_pool_{}", name),
            storage_type_device_memory, size, num_blocks);
        auto bayer_demosaic = make_operator<ops::BayerDemosaicOp>(
            fmt::format("bayer_demosaic_{}", name),
            Arg("pool", demosaic_pool),
            Arg("generate_alpha", true),
            Arg("alpha_value", 65535),
            Arg("bayer_grid_pos", static_cast<int>(bayer_format)),
            Arg("interpolation_mode", 0));

        auto visualizer = make_operator<ops::HolovizOp>(fmt::format("holoviz_{}", name),
            Arg("framebuffer_srgb", true),
            Arg("fullscreen", fullscreen_),
            Arg("headless", headless_),
            Arg("height", height_),
            Arg("width", width_),
            Arg("window_title", window_titles_[i]));

        add_flow(fusa_coe_capture, packed_format_converter, { { "output", "input" } });
        add_flow(packed_format_converter, image_processor, { { "output", "input" } });
        add_flow(image_processor, bayer_demosaic, { { "output", "receiver" } });
        add_flow(bayer_demosaic, visualizer, { { "transmitter", "receivers" } });
    }

    const bool headless_;
    const bool fullscreen_;
    std::vector<hololink::module::EnumerationMetadata> sif_metadatas_;
    std::shared_ptr<Hawk> hawk_;
    std::vector<std::shared_ptr<Ar0234>> cameras_;
    hololink::module::sensors::ar0234::ar0234_mode::Mode camera_mode_;
    const uint32_t timeout_;
    const int32_t width_;
    const int32_t height_;
    const int64_t frame_limit_;
    const std::string interface_;
    std::vector<std::string> window_titles_;
};

} // anonymous namespace

int main(int argc, char** argv)
{
    using namespace hololink::args;
    OptionsDescription options_description("Allowed options");

    // clang-format off
    options_description.add_options()
        ("camera-mode", value<int>()->default_value(
            static_cast<int>(hololink::module::sensors::ar0234::ar0234_mode::AR0234_MODE_1920X1200_60FPS)),
            "AR0234 mode")
        ("channel", value<std::string>()->default_value("A"), "GMSL link to enable: A or B")
        ("exposure", value<int>()->default_value(0x02DC), "AR0234 exposure register value")
        ("frame-limit", value<int>()->default_value(0), "Exit after this many frames")
        ("fullscreen", bool_switch()->default_value(false), "Run in fullscreen mode")
        ("headless", bool_switch()->default_value(false), "Run in headless mode")
        ("height", value<int>()->default_value(1080), "Window height")
        ("hololink", value<std::string>()->default_value(hololink::env_hololink_ip(0, "192.168.0.2")), "IP address of Hololink board")
        ("interface", value<std::string>()->default_value(""), "Ethernet interface (empty = auto)")
        ("pattern", value<int>()->default_value(0), "Sensor test pattern (0 = off)")
        ("sensor", value<std::string>()->default_value("left"),
            "Which sensor(s) to stream: 'left', 'right', or 'both' (opens one window per sensor)")
        ("skip-reset", bool_switch()->default_value(false), "Skip the Hololink reset step")
        ("timeout", value<int>()->default_value(1500), "Capture request timeout in milliseconds")
        ("title", value<std::string>()->default_value("AR0234 (FuSa CoE)"), "Window title prefix")
        ("width", value<int>()->default_value(1920), "Window width")
        ;
    // clang-format on

    auto variables_map = Parser().parse_command_line(argc, argv, options_description);

    const auto hololink_ip = variables_map["hololink"].as<std::string>();
    const auto sensor_arg = variables_map["sensor"].as<std::string>();
    const auto channel_arg = variables_map["channel"].as<std::string>();
    const auto interface = variables_map["interface"].as<std::string>();
    const auto title = variables_map["title"].as<std::string>();
    const auto headless = variables_map["headless"].as<bool>();
    const auto fullscreen = variables_map["fullscreen"].as<bool>();
    const auto skip_reset = variables_map["skip-reset"].as<bool>();
    const int64_t frame_limit = variables_map["frame-limit"].as<int>();
    const int32_t window_width = variables_map["width"].as<int>();
    const int32_t window_height = variables_map["height"].as<int>();
    const uint32_t timeout = static_cast<uint32_t>(variables_map["timeout"].as<int>());
    const uint16_t exposure = static_cast<uint16_t>(variables_map["exposure"].as<int>());
    const int32_t pattern = variables_map["pattern"].as<int>();
    const auto camera_mode_int = variables_map["camera-mode"].as<int>();

    try {
        auto sif_list = sensor_arg_to_sif(sensor_arg);
        const std::vector<std::string> sif_name = { "left", "right" };

        auto& adapter = hololink::module::Adapter::get_adapter();
        hololink::module::EnumerationMetadata adapter_metadata
            = adapter.wait_for_channel(hololink_ip, std::chrono::seconds(30));

        std::vector<hololink::module::EnumerationMetadata> sif_metadatas;
        for (int sif : sif_list) {
            hololink::module::EnumerationMetadata sif_meta(adapter_metadata);
            adapter.use_sensor(sif_meta, sif);
            sif_metadatas.push_back(std::move(sif_meta));
        }

        auto holo_iface = hololink::module::HololinkInterfaceV1::get_service(adapter_metadata);
        holo_iface->start();
        auto board = hololink::module::taurotech_da326::TauroTechDa326InterfaceV1::get_service(adapter_metadata);
        auto legacy_holo = board->hololink();
        auto deserializer = std::make_shared<Max96716a>(legacy_holo);
        auto hawk = std::make_shared<Hawk>(legacy_holo);
        auto camera_mode = static_cast<hololink::module::sensors::ar0234::ar0234_mode::Mode>(camera_mode_int);

        std::vector<std::shared_ptr<Ar0234>> cameras;
        std::vector<std::string> window_titles;
        auto hawk_sensors = hawk->sensors();
        for (int sif : sif_list) {
            cameras.push_back(hawk_sensors[sif]);
            window_titles.push_back(title + " - " + sif_name[sif]);
        }

        int32_t per_w = window_width;
        int32_t per_h = window_height;
        if (sif_list.size() > 1) {
            per_w /= 2;
            per_h /= 2;
        }

        auto application = holoscan::make_application<Application>(headless, fullscreen,
            sif_metadatas, hawk, std::move(cameras), camera_mode, timeout,
            per_w, per_h, frame_limit, interface, window_titles);

        try {
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

            auto link = (channel_arg == "B") ? Max96716a::GmslLink::LINK_B
                                             : Max96716a::GmslLink::LINK_A;
            deserializer->enable_link_exclusive(link);

            uint8_t ser_id = hawk->serializer().get_register(hololink::module::sensors::max9295d::Max9295d::DEV_ID_REG);
            if (ser_id != hololink::module::sensors::max9295d::Max9295d::DEV_ID) {
                throw std::runtime_error(fmt::format(
                    "Serializer mismatch: expected 0x{:02X}, got 0x{:02X}",
                    hololink::module::sensors::max9295d::Max9295d::DEV_ID, ser_id));
            }

            for (auto& s : hawk->sensors()) {
                uint16_t s_id = s->get_register(Ar0234::DEV_ID_REG);
                if (s_id != Ar0234::DEV_ID) {
                    throw std::runtime_error(fmt::format(
                        "Sensor mismatch at i2c=0x{:02X}: expected 0x{:04X}, got 0x{:04X}",
                        s->get_i2c_address(), Ar0234::DEV_ID, s_id));
                }
            }

            deserializer->configure_video_pipe();
            uint8_t left = deserializer->stream_id_to_pipe_mapping(
                link, 0, Max96716a::VideoPipe::PIPE_Y);
            uint8_t right = deserializer->stream_id_to_pipe_mapping(
                link, 2, Max96716a::VideoPipe::PIPE_Z);
            deserializer->set_register(Max96716a::VIDEO_PIPE_SEL,
                static_cast<uint8_t>(left | right));

            hawk->configure(camera_mode);
            if (pattern) {
                hawk->test_pattern(static_cast<uint16_t>(pattern));
            }
            hawk->set_exposure_reg(exposure);

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
