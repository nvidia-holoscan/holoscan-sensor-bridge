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

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <unistd.h>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/bayer_demosaic/bayer_demosaic.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include <hololink/common/holoargs.hpp>
#include <hololink/core/csi_controller.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/enumerator.hpp>
#include <hololink/core/tools.hpp>
#include <hololink/operators/csi_to_bayer/csi_to_bayer.hpp>
#include <hololink/operators/image_processor/image_processor.hpp>
#include <hololink/operators/roce_receiver/roce_receiver_op.hpp>
#include <hololink/sensors/camera/camera_sensor.hpp>
#include <hololink/sensors/camera/vb1940/native_vb1940_sensor.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

class HoloscanApplication : public holoscan::Application {
public:
    HoloscanApplication(
        bool headless,
        bool fullscreen,
        CUcontext cuda_context,
        int cuda_device_ordinal,
        hololink::DataChannel& hololink_channel,
        const std::string& ibv_name,
        uint32_t ibv_port,
        std::shared_ptr<hololink::sensors::NativeVb1940Sensor> camera,
        hololink::sensors::vb1940_mode::Mode camera_mode,
        int frame_limit)
        : headless_(headless)
        , fullscreen_(fullscreen)
        , cuda_context_(cuda_context)
        , cuda_device_ordinal_(cuda_device_ordinal)
        , hololink_channel_(hololink_channel)
        , ibv_name_(ibv_name)
        , ibv_port_(ibv_port)
        , camera_(camera)
        , camera_mode_(camera_mode)
        , frame_limit_(frame_limit)
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
        camera_->set_mode(camera_mode_);

        auto csi_to_bayer_pool = make_resource<BlockMemoryPool>(
            "pool",
            1, // storage_type of 1 is device memory
            camera_->get_width() * sizeof(uint16_t) * camera_->get_height(), // block_size
            2 // num_blocks
        );

        auto csi_to_bayer_operator = make_operator<hololink::operators::CsiToBayerOp>(
            "csi_to_bayer",
            Arg("allocator", csi_to_bayer_pool),
            Arg("cuda_device_ordinal", cuda_device_ordinal_));
        camera_->configure_converter(csi_to_bayer_operator);

        size_t frame_size = csi_to_bayer_operator->get_csi_length();
        auto receiver_operator = make_operator<hololink::operators::RoceReceiverOp>(
            "receiver",
            condition,
            Arg("frame_size", frame_size),
            Arg("frame_context", cuda_context_),
            Arg("ibv_name", ibv_name_),
            Arg("ibv_port", ibv_port_),
            Arg("hololink_channel", &hololink_channel_),
            Arg("device_start", std::function<void()>([this] {
                camera_->start();
            })),
            Arg("device_stop", std::function<void()>([this] {
                camera_->stop();
            })));

        auto pixel_format = static_cast<int>(camera_->get_pixel_format());
        auto bayer_format = static_cast<int>(camera_->get_bayer_format());
        auto image_processor_operator = make_operator<hololink::operators::ImageProcessorOp>(
            "image_processor",
            Arg("optical_black", 8), // Optical black value for vb1940 is 8 for RAW10
            Arg("bayer_format", bayer_format),
            Arg("pixel_format", pixel_format));

        const int rgba_components_per_pixel = 4;
        auto bayer_pool = make_resource<BlockMemoryPool>(
            "pool",
            1, // storage_type of 1 is device memory
            camera_->get_width() * rgba_components_per_pixel * sizeof(uint16_t) * camera_->get_height(), // block_size
            2 // num_blocks
        );

        auto demosaic = make_operator<holoscan::ops::BayerDemosaicOp>(
            "demosaic",
            Arg("pool", bayer_pool),
            Arg("generate_alpha", true),
            Arg("alpha_value", 65535),
            Arg("bayer_grid_pos", bayer_format),
            Arg("interpolation_mode", 0));

        auto visualizer = make_operator<holoscan::ops::HolovizOp>(
            "holoviz",
            Arg("fullscreen", fullscreen_),
            Arg("headless", headless_),
            Arg("framebuffer_srgb", true));

        add_flow(receiver_operator, csi_to_bayer_operator, { { "output", "input" } });
        add_flow(csi_to_bayer_operator, image_processor_operator, { { "output", "input" } });
        add_flow(image_processor_operator, demosaic, { { "output", "receiver" } });
        add_flow(demosaic, visualizer, { { "transmitter", "receivers" } });
    }

private:
    bool headless_;
    bool fullscreen_;
    CUcontext cuda_context_;
    int cuda_device_ordinal_;
    hololink::DataChannel& hololink_channel_;
    std::string ibv_name_;
    uint32_t ibv_port_;
    std::shared_ptr<hololink::sensors::NativeVb1940Sensor> camera_;
    hololink::sensors::vb1940_mode::Mode camera_mode_;
    int frame_limit_;
};

int main(int argc, char** argv)
{
    using namespace hololink::args;
    OptionsDescription options_description("Allowed options");

    // Select the first available device to be the default
    std::string default_ibv_name;
    auto infiniband_devices = hololink::core::infiniband_devices();
    if (!infiniband_devices.empty())
        default_ibv_name = infiniband_devices[0];

    // clang-format off
    options_description.add_options()
        ("camera-mode", value<int>()->default_value(static_cast<int>(hololink::sensors::vb1940_mode::VB1940_MODE_2560X1984_30FPS)), "VB1940 mode (default: 0)")
        ("headless", bool_switch()->default_value(false), "Run in headless mode")
        ("fullscreen", bool_switch()->default_value(false), "Run in fullscreen mode")
        ("frame-limit", value<int>()->default_value(0), "Exit after receiving this many frames")
        ("configuration", value<std::string>()->default_value("example_configuration.yaml"), "Configuration file")
        ("hololink", value<std::string>()->default_value("192.168.0.2"), "IP address of Hololink board")
        ("ibv-name", value<std::string>()->default_value(default_ibv_name), "IBV device to use")
        ("ibv-port", value<uint32_t>()->default_value(1), "Port number of IBV device (default: 1)")
        ("use-sensor", value<uint32_t>()->default_value(0), "Use the specific sensor (0 or 1, default=0)")
        ;
    // clang-format on

    auto variables_map = Parser().parse_command_line(argc, argv, options_description);
    int use_sensor = variables_map["use-sensor"].as<uint32_t>();
    if (use_sensor >= 2) {
        HOLOSCAN_LOG_ERROR("Invalid value use_sensor={}; must be 0 or 1.", use_sensor);
        return -1;
    }

    try {
        // Initialize CUDA
        cuInit(0);
        CUdevice cu_device;
        int cu_device_ordinal = 0;
        cuDeviceGet(&cu_device, cu_device_ordinal);
        CUcontext cu_context;
        cuDevicePrimaryCtxRetain(&cu_context, cu_device);

        // Get a handle to the Hololink device
        auto channel_metadata = hololink::Enumerator::find_channel(variables_map["hololink"].as<std::string>());
        hololink::DataChannel::use_sensor(channel_metadata, use_sensor);
        hololink::DataChannel hololink_channel(channel_metadata);

        // Get a handle to the camera
        auto camera = std::make_shared<hololink::sensors::NativeVb1940Sensor>(hololink_channel);

        // Convert camera_mode to proper enum
        auto camera_mode = static_cast<hololink::sensors::vb1940_mode::Mode>(variables_map["camera-mode"].as<int>());

        // Set up the application
        auto app = std::make_unique<HoloscanApplication>(
            variables_map["headless"].as<bool>(),
            variables_map["fullscreen"].as<bool>(),
            cu_context,
            cu_device_ordinal,
            hololink_channel,
            variables_map["ibv-name"].as<std::string>(),
            variables_map["ibv-port"].as<uint32_t>(),
            camera,
            camera_mode,
            variables_map["frame-limit"].as<int>());
        app->config(variables_map["configuration"].as<std::string>());

        // Run it
        auto hololink = hololink_channel.hololink();
        hololink->start();
        hololink->reset();
        hololink->write_uint32(0x8, 0x0); // Keep the sensor RESET at low
        camera->setup_clock();
        // Release the sensor RESET to high
        if (use_sensor == 0) {
            hololink->write_uint32(0x8, 0x1);
        } else if (use_sensor == 1) {
            hololink->write_uint32(0x8, 0x2);
        } else {
            // We already checked this above; so this is redundant.
            HOLOSCAN_LOG_ERROR("Invalid value use_sensor={}; must be 0 or 1.", use_sensor);
            return -1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        camera->get_register_32(0x0000); // DEVICE_MODEL_ID:"S940"(ASCII code:0x53393430)
        camera->get_register_32(0x0734); // EXT_CLOCK(25MHz = 0x017d7840)
        camera->configure(camera_mode);

        app->run();
        hololink->stop();

        cuDevicePrimaryCtxRelease(cu_device);
        return 0;
    } catch (const std::exception& e) {
        HOLOSCAN_LOG_ERROR("Application failed: {}", e.what());
        return -1;
    }
}
