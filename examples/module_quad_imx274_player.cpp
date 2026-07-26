/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * 4-channel IMX274 player driving the hololink_module V1 surface.
 *
 * Topology: 4 independent HSB-Lite data planes (each its own IMX274
 * camera) flowing into a single Holoscan pipeline (RoceReceiverOp →
 * CsiToBayerOp → ImageProcessorOp → BayerDemosaicOp → HolovizOp).
 *
 * Default IPs: 192.168.0.200 / .201 / .202 / .203 (devices 0..3). The
 * example pins the cameras to IMX274_MODE_1920X1080_60FPS (mode 1,
 * 1080p RAW10) so the four-channel aggregate fits in the data-plane
 * bandwidth budget; the 4K modes over four channels exceed what the
 * link can sustain.
 */

#include <getopt.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <set>
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

#include "hololink/module/sensors/imx274/imx274_cam.hpp"
#include "hololink/module/sensors/imx274/imx274_mode.hpp"

// Wait until every expected peer IP has been enumerated by the bootp
// listener. Returns the metadata in the same order. Each
// Adapter::wait_for_channel call is bounded by the remaining time
// against the cumulative deadline, so the total time to wait for all
// peers is bounded by `timeout` rather than 4 × `timeout`.
static std::vector<hololink::module::EnumerationMetadata>
enumerate(hololink::module::Adapter& adapter,
    const std::vector<std::string>& expected_peer_ips,
    std::chrono::milliseconds timeout)
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    std::vector<hololink::module::EnumerationMetadata> found;
    found.reserve(expected_peer_ips.size());
    for (const auto& ip : expected_peer_ips) {
        const auto now = std::chrono::steady_clock::now();
        const auto remaining = (now >= deadline)
            ? std::chrono::milliseconds(0)
            : std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
        found.push_back(adapter.wait_for_channel(ip, remaining));
    }
    return found;
}

// Everything the application needs to drive one data plane end-to-end:
// the enumeration metadata that identifies the supplement module, the
// per-board HololinkInterface (lifecycle: start / reset / stop), and
// the IMX274 driver instance for this data plane. Two channels that
// happen to share a board hold the same HololinkInterface shared_ptr
// (the service locator returns one instance per (module, instance_id)
// pair); the application treats them independently regardless.
struct Channel {
    hololink::module::EnumerationMetadata metadata;
    std::shared_ptr<hololink::module::HololinkInterfaceV1> hololink;
    std::shared_ptr<hololink::module::sensors::imx274::Imx274Cam> camera;
};

class HoloscanApplication : public holoscan::Application {
public:
    HoloscanApplication(bool headless, CUcontext cuda_context, int cuda_device_ordinal,
        std::vector<Channel> channels,
        hololink::module::sensors::imx274::Imx274_Mode camera_mode,
        int64_t frame_limit, uint32_t window_height, uint32_t window_width,
        std::string window_title)
        : headless_(headless)
        , cuda_context_(cuda_context)
        , cuda_device_ordinal_(cuda_device_ordinal)
        , channels_(std::move(channels))
        , camera_mode_(camera_mode)
        , frame_limit_(frame_limit)
        , window_height_(window_height)
        , window_width_(window_width)
        , window_title_(std::move(window_title))
    {
        enable_metadata(true);
        metadata_policy(holoscan::MetadataPolicy::kReject);
    }
    HoloscanApplication() = delete;

    void compose() override
    {
        const size_t channel_count = channels_.size();

        // One condition per channel.
        std::vector<std::shared_ptr<holoscan::Condition>> conditions;
        for (size_t i = 0; i < channel_count; ++i) {
            if (frame_limit_) {
                conditions.push_back(make_condition<holoscan::CountCondition>(
                    "count_" + std::to_string(i), frame_limit_));
            } else {
                conditions.push_back(make_condition<holoscan::BooleanCondition>(
                    "ok_" + std::to_string(i), true));
            }
        }

        // Sensors + buffers.
        for (Channel& ch : channels_) {
            ch.camera->set_mode(camera_mode_);
        }
        const uint32_t camera_width = channels_[0].camera->width();
        const uint32_t camera_height = channels_[0].camera->height();
        const auto bayer_format = channels_[0].camera->bayer_format();
        const auto pixel_format = channels_[0].camera->pixel_format();

        auto csi_to_bayer_pool = make_resource<holoscan::BlockMemoryPool>(
            "csi_pool",
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
            const std::string tensor_name = "cam" + std::to_string(i);
            Channel& ch = channels_[i];

            auto csi_op = make_operator<hololink::module::operators::CsiToBayerOp>(
                "csi_to_bayer_" + std::to_string(i),
                holoscan::Arg("allocator", csi_to_bayer_pool),
                holoscan::Arg("cuda_device_ordinal", cuda_device_ordinal_),
                holoscan::Arg("out_tensor_name", tensor_name));
            ch.camera->configure_converter(csi_op);
            csi_ops.push_back(csi_op);

            const size_t frame_size = csi_op->get_csi_length();

            auto cam = ch.camera;
            auto receiver = make_operator<hololink::module::operators::RoceReceiverOp>(
                "receiver_" + std::to_string(i),
                conditions[i],
                holoscan::Arg("enumeration_metadata", ch.metadata),
                holoscan::Arg("frame_context", cuda_context_),
                holoscan::Arg("frame_size", frame_size),
                holoscan::Arg("device_start", std::function<void()>([cam] { cam->start(); })),
                holoscan::Arg("device_stop", std::function<void()>([cam] { cam->stop(); })));
            receivers.push_back(receiver);

            auto image_proc = make_operator<hololink::module::operators::ImageProcessorOp>(
                "image_processor_" + std::to_string(i),
                holoscan::Arg("optical_black", 50), // IMX274 optical black
                holoscan::Arg("bayer_format", static_cast<int>(bayer_format)),
                holoscan::Arg("pixel_format", static_cast<int>(pixel_format)));
            image_processors.push_back(image_proc);

            auto demosaic = make_operator<holoscan::ops::BayerDemosaicOp>(
                "demosaic_" + std::to_string(i),
                holoscan::Arg("pool", bayer_pool),
                holoscan::Arg("generate_alpha", true),
                holoscan::Arg("alpha_value", 65535),
                holoscan::Arg("bayer_grid_pos", static_cast<int>(bayer_format)),
                holoscan::Arg("interpolation_mode", 0),
                holoscan::Arg("in_tensor_name", tensor_name),
                holoscan::Arg("out_tensor_name", tensor_name));
            demosaics.push_back(demosaic);

            // 2x2 grid: cam0 top-left, cam1 top-right, cam2 bottom-left, cam3 bottom-right.
            holoscan::ops::HolovizOp::InputSpec spec(
                tensor_name, holoscan::ops::HolovizOp::InputType::COLOR);
            holoscan::ops::HolovizOp::InputSpec::View view;
            view.offset_x_ = 0.5f * static_cast<float>(i % 2);
            view.offset_y_ = 0.5f * static_cast<float>(i / 2);
            view.width_ = 0.5f;
            view.height_ = 0.5f;
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
    }

private:
    const bool headless_;
    const CUcontext cuda_context_;
    const int cuda_device_ordinal_;
    std::vector<Channel> channels_;
    hololink::module::sensors::imx274::Imx274_Mode camera_mode_;
    const int64_t frame_limit_;
    const uint32_t window_height_;
    const uint32_t window_width_;
    const std::string window_title_;
};

int main(int argc, char** argv)
{
    // Defaults match the Python sibling at examples/module_imx274_player.py.
    auto camera_mode = hololink::module::sensors::imx274::Imx274_Mode::IMX274_MODE_1920X1080_60FPS;
    bool headless = false;
    int64_t frame_limit = 0;
    std::vector<std::string> device_ips {
        hololink::module::env_hololink_ip(0, "192.168.0.200"),
        hololink::module::env_hololink_ip(1, "192.168.0.201"),
        hololink::module::env_hololink_ip(2, "192.168.0.202"),
        hololink::module::env_hololink_ip(3, "192.168.0.203"),
    };
    std::string module_dir;
    uint32_t window_height = 2160 / 4;
    uint32_t window_width = 3840 / 3;
    std::string window_title("hololink_module — 4-channel IMX274");
    std::chrono::seconds discovery_timeout(30);

    const struct option long_options[] = {
        { "help", no_argument, nullptr, 'h' },
        { "headless", no_argument, nullptr, 0 },
        { "camera-mode", required_argument, nullptr, 0 },
        { "frame-limit", required_argument, nullptr, 0 },
        { "device-0", required_argument, nullptr, 0 },
        { "device-1", required_argument, nullptr, 0 },
        { "device-2", required_argument, nullptr, 0 },
        { "device-3", required_argument, nullptr, 0 },
        { "module-dir", required_argument, nullptr, 0 },
        { "discovery-timeout", required_argument, nullptr, 0 },
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
            if (name == "headless") {
                headless = true;
            } else if (name == "camera-mode") {
                camera_mode = static_cast<hololink::module::sensors::imx274::Imx274_Mode>(
                    std::stoi(argument));
            } else if (name == "frame-limit") {
                frame_limit = std::stoll(argument);
            } else if (name == "device-0") {
                device_ips[0] = argument;
            } else if (name == "device-1") {
                device_ips[1] = argument;
            } else if (name == "device-2") {
                device_ips[2] = argument;
            } else if (name == "device-3") {
                device_ips[3] = argument;
            } else if (name == "module-dir") {
                module_dir = argument;
            } else if (name == "discovery-timeout") {
                discovery_timeout = std::chrono::seconds(std::stoll(argument));
            }
        } else if (c == 'h') {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  --device-0 <ip>          Peer IP of device 0 (default 192.168.0.200)\n"
                      << "  --device-1 <ip>          Peer IP of device 1 (default 192.168.0.201)\n"
                      << "  --device-2 <ip>          Peer IP of device 2 (default 192.168.0.202)\n"
                      << "  --device-3 <ip>          Peer IP of device 3 (default 192.168.0.203)\n"
                      << "  --module-dir <path>      Directory containing hololink_<UUID>.so\n"
                      << "  --camera-mode <int>      IMX274 mode (1=1080p60 default, 0=4K60 RAW10, 2=4K60 RAW12)\n"
                      << "  --frame-limit <int>      Exit after this many frames per channel\n"
                      << "  --headless               Run holoviz without a display\n"
                      << "  --discovery-timeout <s>  Seconds to wait for bootp announcements (default 30)\n";
            return EXIT_SUCCESS;
        }
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

    const std::vector<hololink::module::EnumerationMetadata> metadatas
        = enumerate(adapter, device_ips, discovery_timeout);

    // Build a channel per device.
    std::vector<Channel> channels;
    channels.reserve(metadatas.size());
    for (const hololink::module::EnumerationMetadata& md : metadatas) {
        Channel ch;
        ch.metadata = md;
        // Get the driver object
        ch.camera = std::make_shared<hololink::module::sensors::imx274::Imx274Cam>(md);
        // Connect to the HSB unit
        ch.hololink = hololink::module::HololinkInterfaceV1::get_service(md);
        channels.push_back(std::move(ch));
    }

    // Bring each unit up; channels sharing a board share a HololinkInterface,
    // so start/reset run once per board.
    std::vector<std::shared_ptr<hololink::module::HololinkInterfaceV1>> started;
    std::set<hololink::module::HololinkInterfaceV1*> brought_up;
    for (Channel& ch : channels) {
        if (brought_up.insert(ch.hololink.get()).second) {
            if (ch.hololink->start() != HOLOLINK_MODULE_OK) {
                throw std::runtime_error("HololinkInterface::start failed");
            }
            started.push_back(ch.hololink);
            // Drive the unit into a known state
            if (ch.hololink->reset() != HOLOLINK_MODULE_OK) {
                throw std::runtime_error("HololinkInterface::reset failed");
            }
        }
        // Configure the camera
        ch.camera->configure(camera_mode);
        ch.camera->set_digital_gain_reg(0x4);
    }

    // Run it
    auto application = holoscan::make_application<HoloscanApplication>(
        headless, cu_context, cu_device_ordinal,
        std::move(channels), camera_mode, frame_limit,
        window_height, window_width, window_title);
    std::cout << "Calling run" << std::endl;
    application->run();

    // Stop streaming data and close up sockets
    for (auto& hololink : started) {
        hololink->stop();
    }
    HOLOLINK_MODULE_CUDA_CHECK(cuDevicePrimaryCtxRelease(cu_device));

    return 0;
}
