/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * IMX274 CoE capture player using the hololink_module API.
 *
 * Module port of `examples/fusa_coe_imx274_player.cpp`: FusaCoeCaptureOp
 * resolves CoeDataChannelInterfaceV1 + FrameMetadataInterfaceV1 from
 * enumeration metadata; Imx274Cam drives the sensor over the module V1
 * surface. Supports mono (sensor 0 or 1) or stereo (sensor -1).
 *
 * This is the HSB-Lite counterpart of module_fusa_coe_vb1940_player, and the
 * only player that exercises the CrossLink-NX D-PHY layout (ClnxMipiDphyV1,
 * reached over SPI rather than an APB window).
 */

#include <getopt.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/bayer_demosaic/bayer_demosaic.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include "hololink/module/adapter.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/hololink.hpp"
#include "hololink/module/mipi_dphy.hpp"
#include "hololink/module/networking.hpp"
#include "hololink/module/operators/csi_to_bayer_op.hpp"
#include "hololink/module/operators/fusa_coe_capture_op.hpp"
#include "hololink/module/operators/image_processor_op.hpp"
#include "hololink/module/operators/packed_format_converter_op.hpp"
#include "hololink/module/tools.hpp"

#include "hololink/module/sensors/imx274/imx274_cam.hpp"
#include "hololink/module/sensors/imx274/imx274_mode.hpp"

hololink::module::MacAddress parse_mac_address(const std::string& mac_str)
{
    hololink::module::MacAddress mac;
    std::istringstream iss(mac_str);
    std::string byte_str;
    int i = 0;

    while (std::getline(iss, byte_str, ':')) {
        if (i >= 6) {
            throw std::invalid_argument("Too many bytes in MAC address");
        }

        if (byte_str.size() != 2) {
            throw std::invalid_argument("Each byte must be exactly two hex digits");
        }

        uint16_t byte;
        std::istringstream hex_stream(byte_str);
        hex_stream >> std::hex >> byte;
        if (hex_stream.fail() || byte > 0xFF) {
            throw std::invalid_argument("Invalid hex in MAC address");
        }
        hex_stream >> std::ws;
        if (hex_stream.peek() != EOF) {
            throw std::invalid_argument("Invalid hex in MAC address");
        }

        mac[i++] = static_cast<uint8_t>(byte);
    }

    if (i != 6) {
        throw std::invalid_argument("MAC address must have exactly 6 bytes");
    }

    return mac;
}

struct Channel {
    hololink::module::EnumerationMetadata metadata;
    std::shared_ptr<hololink::module::sensors::imx274::Imx274Cam> camera;
    std::string tensor_name;
};

class Application : public holoscan::Application {
public:
    Application(
        bool headless,
        bool fullscreen,
        std::vector<Channel> channels,
        hololink::module::sensors::imx274::Imx274_Mode camera_mode,
        uint32_t timeout,
        int frame_limit,
        bool no_packetizer,
        uint8_t pattern,
        bool pattern_set)
        : headless_(headless)
        , fullscreen_(fullscreen)
        , channels_(std::move(channels))
        , camera_mode_(camera_mode)
        , timeout_(timeout)
        , frame_limit_(frame_limit)
        , no_packetizer_(no_packetizer)
        , pattern_(pattern)
        , pattern_set_(pattern_set)
    {
        enable_metadata(true);
        metadata_policy(holoscan::MetadataPolicy::kReject);
    }

    void compose() override
    {
        using namespace holoscan;

        const auto& base_metadata = channels_[0].metadata;
        const std::string interface = base_metadata.get<std::string>("interface");
        const std::string mac_str = base_metadata.get<std::string>("mac_id");
        const auto mac_bytes = parse_mac_address(mac_str);
        HOLOSCAN_LOG_INFO("Using interface={}, mac={}", interface, mac_str);

        const float view_width = 1.0f / channels_.size();
        std::vector<ops::HolovizOp::InputSpec> specs;
        for (uint32_t i = 0; i < channels_.size(); ++i) {
            ops::HolovizOp::InputSpec::View view;
            view.offset_x_ = view_width * i;
            view.offset_y_ = 0;
            view.width_ = view_width;
            view.height_ = 1;
            ops::HolovizOp::InputSpec spec(channels_[i].tensor_name, ops::HolovizOp::InputType::COLOR);
            spec.views_.push_back(view);
            specs.push_back(spec);
        }
        const uint32_t width = 800 * channels_.size();
        const uint32_t height = 620;
        auto visualizer = make_operator<holoscan::ops::HolovizOp>(
            "holoviz",
            Arg("tensors", specs),
            Arg("fullscreen", fullscreen_),
            Arg("headless", headless_),
            Arg("width", width),
            Arg("height", height),
            Arg("framebuffer_srgb", true));

        for (uint32_t i = 0; i < channels_.size(); ++i) {
            Channel& ch = channels_[i];
            ch.camera->configure(camera_mode_);
            ch.camera->set_digital_gain_reg(4);
            if (pattern_set_) {
                ch.camera->test_pattern_enable(pattern_);
            }

            const auto pixel_format = ch.camera->pixel_format();
            const auto bayer_format = ch.camera->bayer_format();
            const auto camera_width = ch.camera->width();
            const auto camera_height = ch.camera->height();
            const std::string& name = ch.tensor_name;

            // Every IMX274 mode is RAW_10 or RAW_12, both of which the
            // packetizer has a program for.
            const bool packetizer_enabled = !no_packetizer_;

            std::shared_ptr<Condition> condition;
            if (frame_limit_) {
                condition = make_condition<CountCondition>(
                    fmt::format("count_{}", name), frame_limit_);
            } else {
                condition = make_condition<BooleanCondition>(
                    fmt::format("ok_{}", name), true);
            }

            auto fusa_coe_capture = make_operator<hololink::module::operators::FusaCoeCaptureOp>(
                fmt::format("fusa_coe_capture_{}", name),
                condition,
                Arg("enumeration_metadata", ch.metadata),
                Arg("interface", interface),
                Arg("mac_addr", std::vector<uint8_t>(mac_bytes.begin(), mac_bytes.end())),
                Arg("out_tensor_name", name),
                Arg("timeout", timeout_),
                Arg("device_start", std::function<void()>([cam = ch.camera] { cam->start(); })),
                Arg("device_stop", std::function<void()>([cam = ch.camera] { cam->stop(); })));
            fusa_coe_capture->configure_format(pixel_format, bayer_format, packetizer_enabled);
            ch.camera->configure_converter(fusa_coe_capture);

            const size_t size = camera_width * camera_height * 2;
            constexpr int32_t storage_type_device_memory = 1;
            constexpr size_t num_blocks = 4;
            // Which converter is correct depends on the packetizer: enabled, the buffer
            // holds HSB-packed data; disabled, it is still CSI-2 exactly as the sensor
            // sent it.
            auto converter_pool = make_resource<holoscan::BlockMemoryPool>(
                fmt::format("converter_pool_{}", name),
                storage_type_device_memory, size, num_blocks);

            std::shared_ptr<holoscan::Operator> converter;
            if (packetizer_enabled) {
                auto packed_format_converter = make_operator<hololink::module::operators::PackedFormatConverterOp>(
                    fmt::format("packed_format_converter_{}", name),
                    Arg("allocator", converter_pool),
                    Arg("in_tensor_name", name));
                fusa_coe_capture->configure_converter(*packed_format_converter.get());
                converter = packed_format_converter;
            } else {
                auto csi_to_bayer = make_operator<hololink::module::operators::CsiToBayerOp>(
                    fmt::format("csi_to_bayer_{}", name),
                    Arg("allocator", converter_pool));
                fusa_coe_capture->configure_converter(*csi_to_bayer.get());
                converter = csi_to_bayer;
            }

            auto image_processor = make_operator<hololink::module::operators::ImageProcessorOp>(
                fmt::format("image_processor_{}", name),
                // Optical black value for IMX274 is 50.
                Arg("optical_black", 50),
                Arg("bayer_format", static_cast<int>(bayer_format)),
                Arg("pixel_format", static_cast<int>(pixel_format)));

            size_t demosaic_size = camera_width * camera_height * 2 * 4;
            auto demosaic_pool = make_resource<holoscan::BlockMemoryPool>(
                fmt::format("demosaic_pool_{}", name),
                storage_type_device_memory, demosaic_size, num_blocks);

            auto bayer_demosaic = make_operator<holoscan::ops::BayerDemosaicOp>(
                fmt::format("bayer_demosaic_{}", name),
                Arg("pool", demosaic_pool),
                Arg("generate_alpha", true),
                Arg("alpha_value", 65535),
                Arg("bayer_grid_pos", static_cast<int>(bayer_format)),
                Arg("out_tensor_name", name));

            add_flow(fusa_coe_capture, converter, { { "output", "input" } });
            add_flow(converter, image_processor, { { "output", "input" } });
            add_flow(image_processor, bayer_demosaic, { { "output", "receiver" } });
            add_flow(bayer_demosaic, visualizer, { { "transmitter", "receivers" } });
        }
    }

private:
    bool headless_;
    bool fullscreen_;
    std::vector<Channel> channels_;
    hololink::module::sensors::imx274::Imx274_Mode camera_mode_;
    uint32_t timeout_;
    int frame_limit_;
    bool no_packetizer_;
    uint8_t pattern_;
    bool pattern_set_;
};

/* std::stoll already rejects text that isn't a number, but a value that parses
 * and then doesn't fit its destination narrows silently -- a mistyped
 * --timeout -1 becomes 49 days. Bound it before the cast; main's existing
 * std::out_of_range handler turns the throw into usage output. */
static int64_t parse_bounded(const std::string& name, const std::string& argument,
    int64_t minimum, int64_t maximum)
{
    const int64_t value = std::stoll(argument);
    if ((value < minimum) || (value > maximum)) {
        throw std::out_of_range("--" + name + " must be in ["
            + std::to_string(minimum) + ", " + std::to_string(maximum) + "]");
    }
    return value;
}

static void print_usage(const char* argv0, const std::string& default_hololink_ip)
{
    std::cout << "Usage: " << argv0 << " [options]\n"
              << "  --hololink <ip>            IP of the HSB-Lite board (default "
              << default_hololink_ip << ")\n"
              << "  --module-dir <path>        Directory containing hololink_<UUID>.so\n"
              << "  --camera-mode <int>        IMX274 mode (1=1920x1080 60fps default, "
                 "0=3840x2160 60fps RAW10, 2=3840x2160 60fps RAW12)\n"
              << "  --sensor <int>             Sensor to use (0 or 1, or -1 for stereo)\n"
              << "  --pattern <int>            Enable the IMX274 test pattern with this value\n"
              << "  --timeout <ms>             CoE capture timeout in milliseconds "
                 "(default 1500)\n"
              << "  --frame-limit <int>        Exit after this many frames\n"
              << "  --headless                 Run holoviz without a display\n"
              << "  --fullscreen               Run holoviz fullscreen\n"
              << "  --no-packetizer            Capture unpacketized CSI-2 data instead of "
                 "HSB-packed data\n"
              << "  --discovery-timeout <s>    Seconds to wait for bootp (default 30)\n"
              << "  --log-level <level>        Holoscan log level\n";
}

int main(int argc, char** argv)
{
    const std::string default_hololink_ip = hololink::module::env_hololink_ip(0, "192.168.0.2");
    auto camera_mode = hololink::module::sensors::imx274::Imx274_Mode::IMX274_MODE_1920X1080_60FPS;
    bool headless = false;
    bool fullscreen = false;
    bool no_packetizer = false;
    int64_t frame_limit = 0;
    int32_t sensor = -1;
    uint32_t timeout = 1500;
    uint8_t pattern = 0;
    bool pattern_set = false;
    std::string hololink_ip = default_hololink_ip;
    std::string module_dir;
    std::chrono::seconds discovery_timeout(30);
    holoscan::LogLevel log_level = holoscan::LogLevel::INFO;

    const struct option long_options[] = {
        { "help", no_argument, nullptr, 'h' },
        { "camera-mode", required_argument, nullptr, 0 },
        { "headless", no_argument, nullptr, 0 },
        { "fullscreen", no_argument, nullptr, 0 },
        { "no-packetizer", no_argument, nullptr, 0 },
        { "frame-limit", required_argument, nullptr, 0 },
        { "hololink", required_argument, nullptr, 0 },
        { "sensor", required_argument, nullptr, 0 },
        { "pattern", required_argument, nullptr, 0 },
        { "timeout", required_argument, nullptr, 0 },
        { "module-dir", required_argument, nullptr, 0 },
        { "discovery-timeout", required_argument, nullptr, 0 },
        { "log-level", required_argument, nullptr, 0 },
        { 0, 0, nullptr, 0 },
    };

    try {
        while (true) {
            int option_index = 0;
            const int c = getopt_long(argc, argv, "h", long_options, &option_index);
            if (c == -1) {
                break;
            }
            const std::string argument(optarg ? optarg : "");
            if (c == 0) {
                const std::string name = long_options[option_index].name;
                if (name == "camera-mode") {
                    camera_mode = static_cast<hololink::module::sensors::imx274::Imx274_Mode>(
                        std::stoi(argument));
                } else if (name == "headless") {
                    headless = true;
                } else if (name == "fullscreen") {
                    fullscreen = true;
                } else if (name == "no-packetizer") {
                    no_packetizer = true;
                } else if (name == "frame-limit") {
                    frame_limit = parse_bounded(name, argument, 0,
                        std::numeric_limits<int>::max());
                } else if (name == "hololink") {
                    hololink_ip = argument;
                } else if (name == "sensor") {
                    sensor = std::stoi(argument);
                } else if (name == "pattern") {
                    pattern = static_cast<uint8_t>(parse_bounded(name, argument, 0,
                        std::numeric_limits<uint8_t>::max()));
                    pattern_set = true;
                } else if (name == "timeout") {
                    timeout = static_cast<uint32_t>(parse_bounded(name, argument, 0,
                        std::numeric_limits<uint32_t>::max()));
                } else if (name == "module-dir") {
                    module_dir = argument;
                } else if (name == "discovery-timeout") {
                    discovery_timeout = std::chrono::seconds(parse_bounded(
                        name, argument, 0, std::numeric_limits<int64_t>::max()));
                } else if (name == "log-level") {
                    if (argument == "trace" || argument == "TRACE") {
                        log_level = holoscan::LogLevel::TRACE;
                    } else if (argument == "debug" || argument == "DEBUG") {
                        log_level = holoscan::LogLevel::DEBUG;
                    } else if (argument == "info" || argument == "INFO") {
                        log_level = holoscan::LogLevel::INFO;
                    } else if (argument == "warn" || argument == "WARN") {
                        log_level = holoscan::LogLevel::WARN;
                    } else if (argument == "error" || argument == "ERROR") {
                        log_level = holoscan::LogLevel::ERROR;
                    } else if (argument == "critical" || argument == "CRITICAL") {
                        log_level = holoscan::LogLevel::CRITICAL;
                    } else if (argument == "off" || argument == "OFF") {
                        log_level = holoscan::LogLevel::OFF;
                    } else {
                        throw std::runtime_error("Unhandled log level \"" + argument + "\"");
                    }
                } else {
                    throw std::runtime_error("Unhandled option \"" + name + "\"");
                }
            } else if (c == 'h') {
                print_usage(argv[0], default_hololink_ip);
                return EXIT_SUCCESS;
            } else {
                throw std::runtime_error("Unhandled option");
            }
        }
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: " << e.what() << "\n";
        print_usage(argv[0], default_hololink_ip);
        return EXIT_FAILURE;
    } catch (const std::out_of_range& e) {
        std::cerr << "Error: " << e.what() << "\n";
        print_usage(argv[0], default_hololink_ip);
        return EXIT_FAILURE;
    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << "\n";
        print_usage(argv[0], default_hololink_ip);
        return EXIT_FAILURE;
    }

    holoscan::set_log_level(log_level);

    std::shared_ptr<hololink::module::HololinkInterfaceV1> hololink;
    bool hololink_started = false;

    try {
        auto& adapter = hololink::module::Adapter::get_adapter();
        if (!module_dir.empty()) {
            adapter.set_module_directory(std::filesystem::path(module_dir));
        }

        const hololink::module::EnumerationMetadata base_metadata
            = adapter.wait_for_channel(hololink_ip, discovery_timeout);

        hololink = hololink::module::HololinkInterfaceV1::get_service(base_metadata);
        if (hololink->start() != HOLOLINK_MODULE_OK) {
            throw std::runtime_error("HololinkInterface::start failed");
        }
        hololink_started = true;
        if (hololink->reset() != HOLOLINK_MODULE_OK) {
            throw std::runtime_error("HololinkInterface::reset failed");
        }

        if ((sensor == -1) || (sensor == 0) || (sensor == 1)) {
            constexpr unsigned MIPI_LANE_COUNT = 4;
            auto dphy = hololink::module::MipiDphyInterfaceV1::get_service(base_metadata);
            // Currently, each SIF corresponds with the physical MIPI
            const std::vector<unsigned> physical_mipi_interfaces = (sensor == -1)
                ? std::vector<unsigned> { 0u, 1u }
                : std::vector<unsigned> { static_cast<unsigned>(sensor) };
            for (unsigned physical_mipi_interface : physical_mipi_interfaces) {
                const auto status = dphy->program(physical_mipi_interface, MIPI_LANE_COUNT,
                    hololink::module::MipiDphyInterfaceV1::LINE_RATE_DESIGN_MAXIMUM);

                if (status != HOLOLINK_MODULE_OK) {
                    throw std::runtime_error(fmt::format(
                        "Could not program MIPI interface {}: status {}",
                        physical_mipi_interface, status));
                }
            }
        }

        // The I2C expander output selects which camera sits behind the bus, and
        // must match the data channel's sensor id -- otherwise --sensor 1 drives
        // sensor 0's I2C path.
        auto make_channel = [&adapter, &base_metadata](int sensor_id) {
            Channel ch;
            ch.metadata = base_metadata;
            adapter.use_sensor(ch.metadata, sensor_id);
            hololink::module::sensors::imx274::Imx274Cam::use_expander_configuration(
                ch.metadata, static_cast<uint32_t>(sensor_id));
            ch.camera = std::make_shared<hololink::module::sensors::imx274::Imx274Cam>(
                ch.metadata);
            ch.tensor_name = std::to_string(sensor_id);
            return ch;
        };

        std::vector<Channel> channels;
        if (sensor == -1) {
            channels.push_back(make_channel(0));
            channels.push_back(make_channel(1));
        } else if (sensor == 0 || sensor == 1) {
            channels.push_back(make_channel(sensor));
        } else {
            HOLOSCAN_LOG_ERROR("sensor value must be 0 or 1 (given {})", sensor);
            hololink->stop();
            return EXIT_FAILURE;
        }

        auto app = std::make_unique<Application>(
            headless, fullscreen, std::move(channels), camera_mode, timeout,
            static_cast<int>(frame_limit), no_packetizer, pattern, pattern_set);

        app->run();
        hololink->stop();
        hololink_started = false;
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        if (hololink && hololink_started) {
            try {
                hololink->stop();
            } catch (const std::exception& stop_error) {
                HOLOSCAN_LOG_ERROR(
                    "HololinkInterface::stop failed after application error: {}",
                    stop_error.what());
            }
        }
        HOLOSCAN_LOG_ERROR("Application failed: {}", e.what());
        return EXIT_FAILURE;
    }
}
