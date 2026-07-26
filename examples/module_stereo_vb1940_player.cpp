/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Stereo VB1940-AIO player using the hololink_module API.
 *
 * Topology: one Leopard VB1940-AIO board hosts two cameras on a single
 * HSB (sensor 0 = left, sensor 1 = right). Both data planes share the
 * board's HololinkInterface (lifecycle: start / reset / stop runs once
 * per board, not once per channel); each channel gets its own
 * Vb1940Cam pinned to a distinct I2C bus (Leopard wires camera 0 to
 * bus 1, camera 1 to bus 2). The Holoscan pipeline runs two parallel
 * RoceReceiverOp -> CsiToBayerOp -> ImageProcessorOp -> BayerDemosaicOp
 * legs into a single HolovizOp that shows the pair side-by-side.
 *
 * Defaults match `examples/single_network_stereo_vb1940_player.cpp`:
 * `VB1940_MODE_2560X1984_30FPS` (mode 0), one bootp-discovered IP for
 * the HSB, 2-up window split. Mirrors the structure of
 * `examples/module_vb1940_player.cpp` and
 * `examples/module_quad_imx274_player.cpp`.
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
#include <vector>

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
#include "hololink/module/operators/roce_receiver_op.hpp"
#include "hololink/module/ptp_pps_output.hpp"
#include "hololink/module/vsync.hpp"

#include "hololink/module/sensors/vb1940/vb1940_cam.hpp"

#include "vsync_start_op.hpp"

// Everything the application needs to drive one stereo channel
// end-to-end: the per-sensor enumeration metadata (one per call to
// Adapter::use_sensor), the I2C bus the camera lives on, and the
// Vb1940Cam handle. Both channels in a stereo pair hold the same
// HololinkInterface (the locator returns one instance per board); the
// application brings the shared board up once and configures each
// camera independently.
struct Channel {
    hololink::module::EnumerationMetadata metadata;
    std::shared_ptr<hololink::module::sensors::vb1940::Vb1940Cam> camera;
    std::string tensor_name;
};

class HoloscanStereoApplication : public holoscan::Application {
public:
    HoloscanStereoApplication(bool headless, CUcontext cuda_context, int cuda_device_ordinal,
        std::vector<Channel> channels, int64_t frame_limit,
        uint32_t window_height, uint32_t window_width,
        std::string window_title,
        std::shared_ptr<hololink::module::VsyncInterfaceV1> vsync)
        : headless_(headless)
        , cuda_context_(cuda_context)
        , cuda_device_ordinal_(cuda_device_ordinal)
        , channels_(std::move(channels))
        , frame_limit_(frame_limit)
        , window_height_(window_height)
        , window_width_(window_width)
        , window_title_(std::move(window_title))
        , vsync_(std::move(vsync))
    {
        if (channels_.size() != 2) {
            throw std::runtime_error(
                "module_stereo_vb1940_player requires exactly 2 channels");
        }
        // Both channels feed the same HolovizOp; reject duplicate
        // metadata keys rather than failing on the collision.
        enable_metadata(true);
        metadata_policy(holoscan::MetadataPolicy::kReject);
    }
    HoloscanStereoApplication() = delete;

    void compose() override
    {
        const size_t channel_count = channels_.size();

        // One condition per channel.
        std::vector<std::shared_ptr<holoscan::Condition>> conditions;
        for (size_t i = 0; i < channel_count; ++i) {
            if (frame_limit_) {
                conditions.push_back(make_condition<holoscan::CountCondition>(
                    "count_" + channels_[i].tensor_name, frame_limit_));
            } else {
                conditions.push_back(make_condition<holoscan::BooleanCondition>(
                    "ok_" + channels_[i].tensor_name, true));
            }
        }

        // Both stereo cameras share width / height / formats (same mode).
        const uint32_t camera_width = channels_[0].camera->width();
        const uint32_t camera_height = channels_[0].camera->height();
        const auto bayer_format = channels_[0].camera->bayer_format();
        const auto pixel_format = channels_[0].camera->pixel_format();

        auto csi_to_bayer_pool = make_resource<holoscan::BlockMemoryPool>(
            "csi_to_bayer_pool",
            /*storage_type=*/1,
            /*block_size=*/camera_width * sizeof(uint16_t) * camera_height,
            /*num_blocks=*/4 * channel_count);

        constexpr uint32_t rgba_components_per_pixel = 4;
        auto bayer_pool = make_resource<holoscan::BlockMemoryPool>(
            "bayer_pool",
            /*storage_type=*/1,
            /*block_size=*/camera_width * rgba_components_per_pixel * sizeof(uint16_t) * camera_height,
            /*num_blocks=*/4 * channel_count);

        std::vector<std::shared_ptr<hololink::module::operators::RoceReceiverOp>> receivers;
        std::vector<std::shared_ptr<hololink::module::operators::CsiToBayerOp>> csi_ops;
        std::vector<std::shared_ptr<hololink::module::operators::ImageProcessorOp>> image_processors;
        std::vector<std::shared_ptr<holoscan::ops::BayerDemosaicOp>> demosaics;
        std::vector<holoscan::ops::HolovizOp::InputSpec> tensor_specs;

        for (size_t i = 0; i < channel_count; ++i) {
            Channel& ch = channels_[i];

            auto csi_op = make_operator<hololink::module::operators::CsiToBayerOp>(
                "csi_to_bayer_" + ch.tensor_name,
                holoscan::Arg("allocator", csi_to_bayer_pool),
                holoscan::Arg("cuda_device_ordinal", cuda_device_ordinal_),
                holoscan::Arg("out_tensor_name", ch.tensor_name));
            ch.camera->configure_converter(csi_op);
            csi_ops.push_back(csi_op);

            const size_t frame_size = csi_op->get_csi_length();

            auto cam = ch.camera;
            auto receiver = make_operator<hololink::module::operators::RoceReceiverOp>(
                "receiver_" + ch.tensor_name,
                conditions[i],
                holoscan::Arg("enumeration_metadata", ch.metadata),
                holoscan::Arg("frame_context", cuda_context_),
                holoscan::Arg("frame_size", frame_size),
                holoscan::Arg("device_start", std::function<void()>([cam] { cam->start(); })),
                holoscan::Arg("device_stop", std::function<void()>([cam] { cam->stop(); })));
            receivers.push_back(receiver);

            auto image_proc = make_operator<hololink::module::operators::ImageProcessorOp>(
                "image_processor_" + ch.tensor_name,
                // Optical black value for VB1940 is 8 (RAW10).
                holoscan::Arg("optical_black", 8),
                holoscan::Arg("bayer_format", static_cast<int>(bayer_format)),
                holoscan::Arg("pixel_format", static_cast<int>(pixel_format)));
            image_processors.push_back(image_proc);

            auto demosaic = make_operator<holoscan::ops::BayerDemosaicOp>(
                "demosaic_" + ch.tensor_name,
                holoscan::Arg("pool", bayer_pool),
                holoscan::Arg("generate_alpha", true),
                holoscan::Arg("alpha_value", 65535),
                holoscan::Arg("bayer_grid_pos", static_cast<int>(bayer_format)),
                holoscan::Arg("interpolation_mode", 0),
                holoscan::Arg("in_tensor_name", ch.tensor_name),
                holoscan::Arg("out_tensor_name", ch.tensor_name));
            demosaics.push_back(demosaic);

            // Side-by-side: left occupies x=[0, 0.5], right x=[0.5, 1].
            holoscan::ops::HolovizOp::InputSpec spec(
                ch.tensor_name, holoscan::ops::HolovizOp::InputType::COLOR);
            holoscan::ops::HolovizOp::InputSpec::View view;
            view.offset_x_ = 0.5f * static_cast<float>(i);
            view.offset_y_ = 0.0f;
            view.width_ = 0.5f;
            view.height_ = 1.0f;
            spec.views_ = { view };
            tensor_specs.push_back(spec);
        }

        auto visualizer = make_operator<holoscan::ops::HolovizOp>(
            "holoviz",
            holoscan::Arg("headless", headless_),
            holoscan::Arg("framebuffer_srgb", true),
            holoscan::Arg("tensors", tensor_specs),
            holoscan::Arg("height", window_height_),
            holoscan::Arg("width", window_width_),
            holoscan::Arg("window_title", window_title_));

        for (size_t i = 0; i < channel_count; ++i) {
            add_flow(receivers[i], csi_ops[i], { { "output", "input" } });
            add_flow(csi_ops[i], image_processors[i], { { "output", "input" } });
            add_flow(image_processors[i], demosaics[i], { { "output", "receiver" } });
            add_flow(demosaics[i], visualizer, { { "transmitter", "receivers" } });
        }

        // One-shot operator that fires vsync->start() after every
        // operator's start() (and therefore every camera's bring-up)
        // has completed.
        auto vsync_starter = make_operator<VsyncStartOp>(
            "vsync_start",
            make_condition<holoscan::CountCondition>("vsync_start_count", 1),
            holoscan::Arg("vsync", vsync_));
        add_operator(vsync_starter);
    }

private:
    const bool headless_;
    const CUcontext cuda_context_;
    const int cuda_device_ordinal_;
    std::vector<Channel> channels_;
    const int64_t frame_limit_;
    const uint32_t window_height_;
    const uint32_t window_width_;
    const std::string window_title_;
    const std::shared_ptr<hololink::module::VsyncInterfaceV1> vsync_;
};

int main(int argc, char** argv)
{
    const std::string default_hololink_ip = hololink::module::env_hololink_ip(0, "192.168.0.2");
    auto camera_mode = hololink::module::sensors::vb1940::Vb1940_Mode::VB1940_MODE_2560X1984_30FPS;
    bool headless = false;
    int64_t frame_limit = 0;
    std::string hololink_ip = default_hololink_ip;
    std::string module_dir;
    uint32_t window_height = 2160 / 4;
    uint32_t window_width = 3840 / 3;
    std::string window_title;
    std::chrono::seconds discovery_timeout(30);
    holoscan::LogLevel log_level = holoscan::LogLevel::INFO;
    // Synchronized-shutter mode (PTP-PPS-driven VSYNC) is enabled by
    // default — that is the canonical stereo configuration. --no-sync
    // leaves both cameras in free-run mode and skips the
    // ptp_pps_output enable; --sync-frequency overrides the default
    // 30 Hz pulse rate (the legacy stereo player's default).
    bool disable_sync = false;
    unsigned sync_frequency_hz = 30;

    const struct option long_options[] = {
        { "help", no_argument, nullptr, 'h' },
        { "camera-mode", required_argument, nullptr, 0 },
        { "headless", no_argument, nullptr, 0 },
        { "frame-limit", required_argument, nullptr, 0 },
        { "hololink", required_argument, nullptr, 0 },
        { "module-dir", required_argument, nullptr, 0 },
        { "discovery-timeout", required_argument, nullptr, 0 },
        { "window-height", required_argument, nullptr, 0 },
        { "window-width", required_argument, nullptr, 0 },
        { "title", required_argument, nullptr, 0 },
        { "log-level", required_argument, nullptr, 0 },
        { "no-sync", no_argument, nullptr, 0 },
        { "sync-frequency", required_argument, nullptr, 0 },
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
            } else if (name == "frame-limit") {
                frame_limit = std::stoll(argument);
            } else if (name == "hololink") {
                hololink_ip = argument;
            } else if (name == "module-dir") {
                module_dir = argument;
            } else if (name == "discovery-timeout") {
                discovery_timeout = std::chrono::seconds(std::stoll(argument));
            } else if (name == "window-height") {
                window_height = static_cast<uint32_t>(std::stoul(argument));
            } else if (name == "window-width") {
                window_width = static_cast<uint32_t>(std::stoul(argument));
            } else if (name == "title") {
                window_title = argument;
            } else if (name == "no-sync") {
                disable_sync = true;
            } else if (name == "sync-frequency") {
                sync_frequency_hz = static_cast<unsigned>(std::stoul(argument));
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
                      << "  --hololink <ip>            IP of the Leopard VB1940-AIO board (default "
                      << default_hololink_ip << ")\n"
                      << "  --module-dir <path>        Directory containing hololink_<UUID>.so\n"
                      << "  --camera-mode <int>        VB1940 mode (0=2560x1984 30fps default, "
                         "1=1920x1080 30fps, 2=2560x1984 30fps 8-bit, 3=2560x1984 60fps)\n"
                      << "  --frame-limit <int>        Exit after this many frames per channel\n"
                      << "  --headless                 Run holoviz without a display\n"
                      << "  --window-height <int>      Display window height (default 540)\n"
                      << "  --window-width <int>       Display window width (default 1280)\n"
                      << "  --title <text>             Window title (default 'Holoviz - <ip>')\n"
                      << "  --discovery-timeout <s>    Seconds to wait for the bootp announcement (default 30)\n"
                      << "  --log-level <level>        Holoscan log level\n"
                      << "  --no-sync                  Run cameras in free-run; skip PTP-PPS VSYNC (default: sync on)\n"
                      << "  --sync-frequency <hz>      VSYNC pulse rate when sync is on (default 30; valid: 1, 10, 30, 60, 90, 120)\n";
            return EXIT_SUCCESS;
        } else {
            throw std::runtime_error("Unhandled option");
        }
    }

    holoscan::set_log_level(log_level);
    if (window_title.empty()) {
        window_title = "Holoviz - " + hololink_ip;
    }
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

    // One bootp announcement covers both cameras on a VB1940-AIO HSB;
    // Adapter::use_sensor re-points a per-camera clone at the right
    // sensor (and the supplement's update_metadata stamps the
    // per-sensor address fields).
    const hololink::module::EnumerationMetadata base_metadata
        = adapter.wait_for_channel(hololink_ip, discovery_timeout);

    // Both channels resolve to the same HololinkInterface instance
    // (same serial). Bring the shared board up once.
    auto hololink = hololink::module::HololinkInterfaceV1::get_service(base_metadata);
    if (hololink->start() != HOLOLINK_MODULE_OK) {
        throw std::runtime_error("HololinkInterface::start failed");
    }
    if (hololink->reset() != HOLOLINK_MODULE_OK) {
        throw std::runtime_error("HololinkInterface::reset failed");
    }

    std::shared_ptr<hololink::module::VsyncInterfaceV1> vsync = hololink->null_vsync();
    if (!disable_sync) {
        auto ptp_pps = hololink->ptp_pps_output();
        if (ptp_pps->enable(sync_frequency_hz) != HOLOLINK_MODULE_OK) {
            throw std::runtime_error("PtpPpsOutput::enable failed");
        }
        vsync = ptp_pps;
    }

    std::vector<Channel> channels;
    channels.reserve(2);
    {
        Channel left;
        left.metadata = base_metadata;
        adapter.use_sensor(left.metadata, /*sensor_number=*/0);
        left.camera = std::make_shared<hololink::module::sensors::vb1940::Vb1940Cam>(
            left.metadata, vsync);
        left.tensor_name = "left";
        channels.push_back(std::move(left));
    }
    {
        Channel right;
        right.metadata = base_metadata;
        adapter.use_sensor(right.metadata, /*sensor_number=*/1);
        right.camera = std::make_shared<hololink::module::sensors::vb1940::Vb1940Cam>(
            right.metadata, vsync);
        right.tensor_name = "right";
        channels.push_back(std::move(right));
    }

    // Vb1940Cam::configure is idempotent across the stereo pair —
    // shared board-level bring-up runs once even though we call it
    // for both cameras.
    for (Channel& ch : channels) {
        ch.camera->configure(camera_mode);
    }

    // Run it.
    auto application = holoscan::make_application<HoloscanStereoApplication>(
        headless, cu_context, cu_device_ordinal,
        std::move(channels), frame_limit,
        window_height, window_width, window_title,
        vsync);
    std::cout << "Calling run" << std::endl;
    application->run();

    hololink->stop();
    HOLOLINK_MODULE_CUDA_CHECK(cuDevicePrimaryCtxRelease(cu_device));
    return 0;
}
