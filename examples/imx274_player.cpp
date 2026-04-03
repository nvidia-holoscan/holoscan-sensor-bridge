/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <iostream>
#include <string>

#include <hololink/common/cuda_helper.hpp>
#include <hololink/common/tools.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/enumerator.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/core/logging.hpp>
#include <hololink/operators/csi_to_bayer/csi_to_bayer.hpp>
#include <hololink/operators/image_processor/image_processor.hpp>
#include <hololink/operators/roce_receiver/roce_receiver_op.hpp>
#include <hololink/sensors/camera/imx274/native_imx274_sensor.hpp>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/bayer_demosaic/bayer_demosaic.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

namespace {

class HoloscanApplication : public holoscan::Application {
public:
    explicit HoloscanApplication(bool headless, bool fullscreen, CUcontext cuda_context,
        int cuda_device_ordinal, hololink::DataChannel& hololink_channel,
        const std::string& ibv_name, uint32_t ibv_port,
        std::shared_ptr<hololink::sensors::NativeImx274Sensor> camera,
        hololink::sensors::imx274_mode::Mode camera_mode,
        int64_t frame_limit)
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
    HoloscanApplication() = delete;

    void compose() override
    {
        std::shared_ptr<holoscan::Condition> condition;
        if (frame_limit_) {
            condition = make_condition<holoscan::CountCondition>("count", frame_limit_);
        } else {
            condition = make_condition<holoscan::BooleanCondition>("ok", true);
        }
        camera_->set_mode(camera_mode_);

        auto csi_to_bayer_pool = make_resource<holoscan::BlockMemoryPool>("csi_to_bayer_pool",
            // storage_type of 1 is device memory
            1, // storage_type
            camera_->get_width() * sizeof(uint16_t) * camera_->get_height(), // block_size
            4 // num_blocks
        );
        auto csi_to_bayer_operator = make_operator<hololink::operators::CsiToBayerOp>(
            "csi_to_bayer", holoscan::Arg("allocator", csi_to_bayer_pool),
            holoscan::Arg("cuda_device_ordinal", cuda_device_ordinal_));
        camera_->configure_converter(csi_to_bayer_operator);

        const size_t frame_size = csi_to_bayer_operator->get_csi_length();
        auto receiver_operator = make_operator<hololink::operators::RoceReceiverOp>("receiver",
            condition, holoscan::Arg("frame_size", frame_size),
            holoscan::Arg("frame_context", cuda_context_), holoscan::Arg("ibv_name", ibv_name_),
            holoscan::Arg("ibv_port", ibv_port_),
            holoscan::Arg("hololink_channel", &hololink_channel_),
            holoscan::Arg("device_start", std::function<void()>([this] { camera_->start(); })),
            holoscan::Arg("device_stop", std::function<void()>([this] { camera_->stop(); })));

        auto bayer_format = camera_->get_bayer_format();
        auto pixel_format = camera_->get_pixel_format();
        auto image_processor_operator = make_operator<hololink::operators::ImageProcessorOp>(
            "image_processor",
            // Optical black value for imx274 is 50
            holoscan::Arg("optical_black", 50), holoscan::Arg("bayer_format", int(bayer_format)),
            holoscan::Arg("pixel_format", int(pixel_format)));

        const uint32_t rgba_components_per_pixel = 4;
        auto bayer_pool = make_resource<holoscan::BlockMemoryPool>("bayer_pool",
            // storage_type of 1 is device memory
            1, // storage_type
            camera_->get_width() * rgba_components_per_pixel * sizeof(uint16_t)
                * camera_->get_height(), // block_size
            4 // num_blocks
        );
        auto demosaic = make_operator<holoscan::ops::BayerDemosaicOp>("demosaic",
            holoscan::Arg("pool", bayer_pool), holoscan::Arg("generate_alpha", true),
            holoscan::Arg("alpha_value", 65535), holoscan::Arg("bayer_grid_pos", int(bayer_format)),
            holoscan::Arg("interpolation_mode", 0));

        auto visualizer = make_operator<holoscan::ops::HolovizOp>("holoviz",
            holoscan::Arg("fullscreen", fullscreen_), holoscan::Arg("headless", headless_),
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
    hololink::DataChannel& hololink_channel_;
    const std::string ibv_name_;
    const uint32_t ibv_port_;
    std::shared_ptr<hololink::sensors::NativeImx274Sensor> camera_;
    hololink::sensors::imx274_mode::Mode camera_mode_;
    const int64_t frame_limit_;
};

} // anonymous namespace

int main(int argc, char** argv)
{
    const std::string default_hololink_ip("192.168.0.2");
    auto camera_mode = hololink::sensors::imx274_mode::IMX274_MODE_1920X1080_60FPS;
    bool headless = false;
    bool fullscreen = false;
    int64_t frame_limit = 0;
    std::string configuration;
    std::string hololink_ip = default_hololink_ip;
    holoscan::LogLevel log_level = holoscan::LogLevel::INFO;
    uint32_t ibv_port = 1;
    int32_t expander_configuration = 0;
    int32_t pattern = 0;
    bool pattern_set = false;

    std::string ibv_name("roceP5p3s0f0");
    try {
        ibv_name = hololink::infiniband_devices()[0];
    } catch (const std::exception& e) {
        std::cerr << "Error getting IBV name: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    // Parse args
    const struct option long_options[] = { { "help", no_argument, nullptr, 'h' },
        { "camera-mode", required_argument, nullptr, 0 }, { "headless", no_argument, nullptr, 0 },
        { "fullscreen", no_argument, nullptr, 0 }, { "frame-limit", required_argument, nullptr, 0 },
        { "configuration", required_argument, nullptr, 0 },
        { "hololink", required_argument, nullptr, 0 },
        { "ibv-name", required_argument, nullptr, 0 },
        { "ibv-port", required_argument, nullptr, 0 },
        { "expander-configuration", required_argument, nullptr, 0 },
        { "pattern", required_argument, nullptr, 0 },
        { "log-level", required_argument, nullptr, 0 }, { 0, 0, nullptr, 0 } };
    while (true) {
        int option_index = 0;
        const int c = getopt_long(argc, argv, "h", long_options, &option_index);

        if (c == -1) {
            break;
        }

        const std::string argument(optarg ? optarg : "");
        if (c == 0) {
            const struct option* cur_option = &long_options[option_index];
            if (cur_option->name == std::string("camera-mode")) {
                camera_mode = static_cast<hololink::sensors::imx274_mode::Mode>(std::stoi(argument));
            } else if (cur_option->name == std::string("headless")) {
                headless = true;
            } else if (cur_option->name == std::string("fullscreen")) {
                fullscreen = true;
            } else if (cur_option->name == std::string("frame-limit")) {
                frame_limit = std::stoll(argument);
            } else if (cur_option->name == std::string("configuration")) {
                configuration = argument;
            } else if (cur_option->name == std::string("hololink")) {
                hololink_ip = argument;
            } else if (cur_option->name == std::string("log-level")) {
                if ((argument == "trace") || (argument == "TRACE")) {
                    log_level = holoscan::LogLevel::TRACE;
                } else if ((argument == "debug") || (argument == "DEBUG")) {
                    log_level = holoscan::LogLevel::DEBUG;
                } else if ((argument == "info") || (argument == "INFO")) {
                    log_level = holoscan::LogLevel::INFO;
                } else if ((argument == "warn") || (argument == "WARN")) {
                    log_level = holoscan::LogLevel::WARN;
                } else if ((argument == "error") || (argument == "ERROR")) {
                    log_level = holoscan::LogLevel::ERROR;
                } else if ((argument == "critical") || (argument == "CRITICAL")) {
                    log_level = holoscan::LogLevel::CRITICAL;
                } else if ((argument == "off") || (argument == "OFF")) {
                    log_level = holoscan::LogLevel::OFF;
                } else {
                    throw std::runtime_error(fmt::format("Unhandled log level \"{}\"", argument));
                }
            } else if (cur_option->name == std::string("ibv-name")) {
                ibv_name = argument;
            } else if (cur_option->name == std::string("ibv-port")) {
                ibv_port = std::stoul(argument);
            } else if (cur_option->name == std::string("expander-configuration")) {
                expander_configuration = std::stoul(argument);
            } else if (cur_option->name == std::string("pattern")) {
                pattern = std::stoi(argument);
                pattern_set = true;
            } else {
                throw std::runtime_error(fmt::format("Unhandled option \"{}\"", cur_option->name));
            }
        } else {
            switch (c) {
            case 'h':
                std::cout << "Usage: " << argv[0] << " [options]" << std::endl
                          << "Options:" << std::endl
                          << "  -h, --help     display this information" << std::endl
                          << "  --hololink     IP address of Hololink board (default `"
                          << default_hololink_ip << "`)" << std::endl
                          << std::endl;
                return EXIT_SUCCESS;

            default:
                throw std::runtime_error("Unhandled option ");
            }
        }
    }

    try {
        // set the Holoscan log level
        holoscan::set_log_level(log_level);

        std::cout << "Initializing." << std::endl;

        // Get a handle to the GPU
        CudaCheck(cuInit(0));
        int cu_device_ordinal = 0;
        CUdevice cu_device;
        CudaCheck(cuDeviceGet(&cu_device, cu_device_ordinal));
        CUcontext cu_context;
        CudaCheck(cuDevicePrimaryCtxRetain(&cu_context, cu_device));

        // Get a handle to the data source
        hololink::Metadata channel_metadata = hololink::Enumerator::find_channel(hololink_ip);

        hololink::DataChannel hololink_channel(channel_metadata);

        // Get a handle to the camera
        auto camera = std::make_shared<hololink::sensors::NativeImx274Sensor>(
            hololink_channel, expander_configuration);

        // Set up the application
        auto application = holoscan::make_application<HoloscanApplication>(headless, fullscreen,
            cu_context, cu_device_ordinal, hololink_channel, ibv_name, ibv_port, camera,
            camera_mode, frame_limit);
        application->config(configuration);

        // Run it.
        std::shared_ptr<hololink::Hololink> hololink = hololink_channel.hololink();
        hololink->start();
        hololink->reset();
        camera->setup_clock();
        camera->configure(camera_mode);
        camera->set_digital_gain_reg(4);
        if (pattern_set) {
            camera->test_pattern(pattern);
        }
        std::cout << "Calling run" << std::endl;
        application->run();
        hololink->stop();

        CudaCheck(cuDevicePrimaryCtxRelease(cu_device));

    } catch (std::exception& e) {
        std::cout << e.what() << std::endl;
        return -1;
    }

    return 0;
}
