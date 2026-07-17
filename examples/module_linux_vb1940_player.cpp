/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Single-camera VB1940 player using the hololink_module API through the
 * SOFTWARE (Linux) receiver. Identical to
 * examples/module_vb1940_player.cpp except the receiver operator is
 * hololink::module::operators::LinuxReceiverOp instead of
 * RoceReceiverOp — so this example runs on hosts with no infiniband
 * device (HOLOLINK_BUILD_ROCE=OFF).
 */

#include <getopt.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include <cuda.h>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/bayer_demosaic/bayer_demosaic.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include "hololink/module/cuda_unique.hpp"
#include "hololink/module/tools.hpp"

#include "hololink/module/adapter.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/hololink.hpp"
#include "hololink/module/operators/csi_to_bayer_op.hpp"
#include "hololink/module/operators/image_processor_op.hpp"
#include "hololink/module/operators/linux_receiver_op.hpp"

#include "hololink/module/sensors/vb1940/vb1940_cam.hpp"

class HoloscanApplication : public holoscan::Application {
public:
    HoloscanApplication(bool headless, bool fullscreen,
        CUcontext cuda_context, int cuda_device_ordinal,
        hololink::module::EnumerationMetadata metadata,
        std::shared_ptr<hololink::module::sensors::vb1940::Vb1940Cam> camera,
        int64_t frame_limit)
        : headless_(headless)
        , fullscreen_(fullscreen)
        , cuda_context_(cuda_context)
        , cuda_device_ordinal_(cuda_device_ordinal)
        , metadata_(std::move(metadata))
        , camera_(std::move(camera))
        , frame_limit_(frame_limit)
    {
    }
    HoloscanApplication() = delete;

    void compose() override
    {
        std::shared_ptr<holoscan::Condition> condition;
        if (frame_limit_) {
            condition = make_condition<holoscan::CountCondition>("count", frame_limit_);
        } else {
            condition = make_condition<holoscan::BooleanCondition>("ok", true);
        }

        auto csi_to_bayer_pool = make_resource<holoscan::BlockMemoryPool>(
            "csi_to_bayer_pool",
            /*storage_type=*/1,
            /*block_size=*/camera_->width() * sizeof(uint16_t) * camera_->height(),
            /*num_blocks=*/4);
        auto csi_to_bayer_operator = make_operator<hololink::module::operators::CsiToBayerOp>(
            "csi_to_bayer",
            holoscan::Arg("allocator", csi_to_bayer_pool),
            holoscan::Arg("cuda_device_ordinal", cuda_device_ordinal_));
        camera_->configure_converter(csi_to_bayer_operator);

        const size_t frame_size = csi_to_bayer_operator->get_csi_length();
        auto receiver_operator = make_operator<hololink::module::operators::LinuxReceiverOp>(
            "receiver",
            condition,
            holoscan::Arg("enumeration_metadata", metadata_),
            holoscan::Arg("frame_context", cuda_context_),
            holoscan::Arg("frame_size", frame_size),
            holoscan::Arg("device_start", std::function<void()>([this] { camera_->start(); })),
            holoscan::Arg("device_stop", std::function<void()>([this] { camera_->stop(); })));

        const auto bayer_format = camera_->bayer_format();
        const auto pixel_format = camera_->pixel_format();
        auto image_processor_operator = make_operator<hololink::module::operators::ImageProcessorOp>(
            "image_processor",
            // Optical black value for VB1940 is 8 (RAW10).
            holoscan::Arg("optical_black", 8),
            holoscan::Arg("bayer_format", static_cast<int>(bayer_format)),
            holoscan::Arg("pixel_format", static_cast<int>(pixel_format)));

        constexpr uint32_t rgba_components_per_pixel = 4;
        auto bayer_pool = make_resource<holoscan::BlockMemoryPool>(
            "bayer_pool",
            /*storage_type=*/1,
            /*block_size=*/camera_->width() * rgba_components_per_pixel * sizeof(uint16_t) * camera_->height(),
            /*num_blocks=*/4);
        auto demosaic = make_operator<holoscan::ops::BayerDemosaicOp>(
            "demosaic",
            holoscan::Arg("pool", bayer_pool),
            holoscan::Arg("generate_alpha", true),
            holoscan::Arg("alpha_value", 65535),
            holoscan::Arg("bayer_grid_pos", static_cast<int>(bayer_format)),
            holoscan::Arg("interpolation_mode", 0));

        auto visualizer = make_operator<holoscan::ops::HolovizOp>(
            "holoviz",
            holoscan::Arg("fullscreen", fullscreen_),
            holoscan::Arg("headless", headless_),
            holoscan::Arg("framebuffer_srgb", true));

        add_flow(receiver_operator, csi_to_bayer_operator, { { "output", "input" } });
        add_flow(csi_to_bayer_operator, image_processor_operator, { { "output", "input" } });
        add_flow(image_processor_operator, demosaic, { { "output", "receiver" } });
        add_flow(demosaic, visualizer, { { "transmitter", "receivers" } });
    }

private:
    const bool headless_;
    const bool fullscreen_;
    const CUcontext cuda_context_;
    const int cuda_device_ordinal_;
    hololink::module::EnumerationMetadata metadata_;
    std::shared_ptr<hololink::module::sensors::vb1940::Vb1940Cam> camera_;
    const int64_t frame_limit_;
};

int main(int argc, char** argv)
{
    const std::string default_hololink_ip = hololink::module::env_hololink_ip(0, "192.168.0.2");
    auto camera_mode = hololink::module::sensors::vb1940::Vb1940_Mode::VB1940_MODE_2560X1984_30FPS;
    bool headless = false;
    bool fullscreen = false;
    int64_t frame_limit = 0;
    std::string hololink_ip = default_hololink_ip;
    std::string module_dir;
    std::chrono::seconds discovery_timeout(30);
    holoscan::LogLevel log_level = holoscan::LogLevel::INFO;

    const struct option long_options[] = {
        { "help", no_argument, nullptr, 'h' },
        { "camera-mode", required_argument, nullptr, 0 },
        { "headless", no_argument, nullptr, 0 },
        { "fullscreen", no_argument, nullptr, 0 },
        { "frame-limit", required_argument, nullptr, 0 },
        { "hololink", required_argument, nullptr, 0 },
        { "module-dir", required_argument, nullptr, 0 },
        { "discovery-timeout", required_argument, nullptr, 0 },
        { "log-level", required_argument, nullptr, 0 },
        { 0, 0, nullptr, 0 },
    };

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
                camera_mode = static_cast<hololink::module::sensors::vb1940::Vb1940_Mode>(
                    std::stoi(argument));
            } else if (name == "headless") {
                headless = true;
            } else if (name == "fullscreen") {
                fullscreen = true;
            } else if (name == "frame-limit") {
                frame_limit = std::stoll(argument);
            } else if (name == "hololink") {
                hololink_ip = argument;
            } else if (name == "module-dir") {
                module_dir = argument;
            } else if (name == "discovery-timeout") {
                discovery_timeout = std::chrono::seconds(std::stoll(argument));
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
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  --hololink <ip>            IP of the Leopard VB1940 board (default "
                      << default_hololink_ip << ")\n"
                      << "  --module-dir <path>        Directory containing hololink_<UUID>.so\n"
                      << "  --camera-mode <int>        VB1940 mode (0=2560x1984 30fps default, "
                         "1=1920x1080 30fps, 2=2560x1984 30fps 8-bit, 3=2560x1984 60fps)\n"
                      << "  --frame-limit <int>        Exit after this many frames\n"
                      << "  --headless                 Run holoviz without a display\n"
                      << "  --fullscreen               Run holoviz fullscreen\n"
                      << "  --discovery-timeout <s>    Seconds to wait for the bootp announcement (default 30)\n"
                      << "  --log-level <level>        Holoscan log level\n";
            return EXIT_SUCCESS;
        } else {
            throw std::runtime_error("Unhandled option");
        }
    }

    holoscan::set_log_level(log_level);
    std::cout << "Initializing." << std::endl;

    HOLOLINK_MODULE_CUDA_CHECK(cuInit(0));
    const int cu_device_ordinal = 0;
    CUdevice cu_device;
    HOLOLINK_MODULE_CUDA_CHECK(cuDeviceGet(&cu_device, cu_device_ordinal));
    CUcontext cu_context;
    HOLOLINK_MODULE_CUDA_CHECK(cuDevicePrimaryCtxRetain(&cu_context, cu_device));

    auto& adapter = hololink::module::Adapter::get_adapter();
    if (!module_dir.empty()) {
        adapter.set_module_directory(std::filesystem::path(module_dir));
    }

    hololink::module::EnumerationMetadata metadata
        = adapter.wait_for_channel(hololink_ip, discovery_timeout);

    // Get the driver object.
    auto camera = std::make_shared<hololink::module::sensors::vb1940::Vb1940Cam>(
        metadata);
    // Connect to the HSB unit.
    auto hololink = hololink::module::HololinkInterfaceV1::get_service(metadata);
    if (hololink->start() != HOLOLINK_MODULE_OK) {
        throw std::runtime_error("HololinkInterface::start failed");
    }
    // Drive the unit into a known state.
    if (hololink->reset() != HOLOLINK_MODULE_OK) {
        throw std::runtime_error("HololinkInterface::reset failed");
    }
    // Configure the camera (walks the secure-boot FSM the first time
    // after power-up, then applies the per-mode register table).
    camera->configure(camera_mode);
    // Run it.
    auto application = holoscan::make_application<HoloscanApplication>(
        headless, fullscreen, cu_context, cu_device_ordinal,
        metadata, camera, frame_limit);
    std::cout << "Calling run" << std::endl;
    application->run();
    // Stop streaming data and close up sockets.
    hololink->stop();
    HOLOLINK_MODULE_CUDA_CHECK(cuDevicePrimaryCtxRelease(cu_device));
    return 0;
}
