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
#include <hololink/core/data_channel.hpp>
#include <hololink/core/enumerator.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/core/logging.hpp>
#include <hololink/operators/csi_to_bayer/csi_to_bayer.hpp>
#include <hololink/operators/image_processor/image_processor.hpp>
#include <hololink/operators/roce_receiver/roce_receiver_op.hpp>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/bayer_demosaic/bayer_demosaic.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include <pybind11/embed.h>
namespace py = pybind11;
using namespace pybind11::literals; // to bring in the `_a` literal

namespace {

class HoloscanApplication : public holoscan::Application {
public:
    explicit HoloscanApplication(bool headless, bool fullscreen, CUcontext cuda_context,
        int cuda_device_ordinal, hololink::DataChannel& hololink_channel,
        const std::string& ibv_name, uint32_t ibv_port, py::object camera, py::object camera_mode,
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
        // acquire the Python GIL to be able to call the Python functions of the camera device
        py::gil_scoped_acquire guard;

        std::shared_ptr<holoscan::Condition> condition;
        if (frame_limit_) {
            condition = make_condition<holoscan::CountCondition>("count", frame_limit_);
        } else {
            condition = make_condition<holoscan::BooleanCondition>("ok", true);
        }
        camera_.attr("set_mode")(camera_mode_);

        auto csi_to_bayer_pool = make_resource<holoscan::BlockMemoryPool>("csi_to_bayer_pool",
            // storage_type of 1 is device memory
            1, // storage_type
            camera_.attr("_width").cast<int>() * sizeof(uint16_t)
                * camera_.attr("_height").cast<int>(), // block_size
            2 // num_blocks
        );
        auto csi_to_bayer_operator = make_operator<hololink::operators::CsiToBayerOp>(
            "csi_to_bayer", holoscan::Arg("allocator", csi_to_bayer_pool),
            holoscan::Arg("cuda_device_ordinal", cuda_device_ordinal_));
        // Without this; pybind11 doesn't know how to get the right superclass
        // for configure_converter.
        std::shared_ptr<hololink::csi::CsiConverter> csi_converter = csi_to_bayer_operator;
        camera_.attr("configure_converter")(csi_converter);

        const size_t frame_size = csi_to_bayer_operator->get_csi_length();
        auto receiver_operator = make_operator<hololink::operators::RoceReceiverOp>("receiver",
            condition, holoscan::Arg("frame_size", frame_size),
            holoscan::Arg("frame_context", cuda_context_), holoscan::Arg("ibv_name", ibv_name_),
            holoscan::Arg("ibv_port", ibv_port_),
            holoscan::Arg("hololink_channel", &hololink_channel_),
            holoscan::Arg("device_start", std::function<void()>([this] {
                // acquire the Python GIL lock to be able to call the Python functions of the camera
                // device
                py::gil_scoped_acquire guard;
                camera_.attr("start")();
            })),
            holoscan::Arg("device_stop", std::function<void()>([this] {
                // acquire the Python GIL lock to be able to call the Python functions of the camera
                // device
                py::gil_scoped_acquire guard;
                camera_.attr("stop")();
            })));

        auto bayer_format = camera_.attr("bayer_format")()
                                .cast<hololink::csi::BayerFormat>();
        auto pixel_format
            = camera_.attr("pixel_format")().cast<hololink::csi::PixelFormat>();
        auto image_processor_operator = make_operator<hololink::operators::ImageProcessorOp>(
            "image_processor",
            // Optical black value for imx274 is 50
            holoscan::Arg("optical_black", 50), holoscan::Arg("bayer_format", int(bayer_format)),
            holoscan::Arg("pixel_format", int(pixel_format)));

        const uint32_t rgba_components_per_pixel = 4;
        auto bayer_pool = make_resource<holoscan::BlockMemoryPool>("bayer_pool",
            // storage_type of 1 is device memory
            1, // storage_type
            camera_.attr("_width").cast<int>() * rgba_components_per_pixel * sizeof(uint16_t)
                * camera_.attr("_height").cast<int>(), // block_size
            2 // num_blocks
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
    const py::object camera_;
    const py::object camera_mode_;
    const int64_t frame_limit_;
};

} // anonymous namespace

int main(int argc, char** argv)
{
    int32_t camera_mode = 0;
    bool headless = false;
    bool fullscreen = false;
    int64_t frame_limit = 0;
    std::string configuration;
    std::string hololink_ip = "192.168.0.2";
    holoscan::LogLevel log_level = holoscan::LogLevel::INFO;
    uint32_t ibv_port = 1;
    int32_t expander_configuration = 0;
    int32_t pattern = 0;
    bool pattern_set = false;

    std::string ibv_name("roceP5p3s0f0");
    try {
        std::vector<std::string> devices;
        for (auto const& dir_entry : std::filesystem::directory_iterator {
                 std::filesystem::path { "/sys/class/infiniband" } }) {
            devices.push_back(dir_entry.path().filename());
        }
        if (!devices.empty()) {
            std::sort(devices.begin(), devices.end());
            ibv_name = devices[0];
        }
    } catch (...) {
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
                camera_mode = std::stoi(argument);
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
                          << hololink_ip << "`" << std::endl
                          << std::endl;
                return EXIT_SUCCESS;

            default:
                throw std::runtime_error("Unhandled option ");
            }
        }
    }

    // The IMX274 camera is implemented in Python, start the interpreter and keep it alive
    py::scoped_interpreter guard {};

    try {
        // set the Holoscan log level
        holoscan::set_log_level(log_level);

        // set the Python log level as well (needed for the camera implementation only)
        py::module_ hololink_module = py::module_::import("hololink");
        py::module_ logging = py::module_::import("logging");
        int python_log_level = logging.attr("NOTSET").cast<int>();
        switch (log_level) {
        case holoscan::LogLevel::TRACE:
            python_log_level = logging.attr("DEBUG").cast<int>() - 5;
            break;
        case holoscan::LogLevel::DEBUG:
            python_log_level = logging.attr("DEBUG").cast<int>();
            break;
        case holoscan::LogLevel::INFO:
            python_log_level = logging.attr("INFO").cast<int>();
            break;
        case holoscan::LogLevel::WARN:
            python_log_level = logging.attr("WARN").cast<int>();
            break;
        case holoscan::LogLevel::ERROR:
            python_log_level = logging.attr("ERROR").cast<int>();
            break;
        case holoscan::LogLevel::CRITICAL:
            python_log_level = logging.attr("CRITICAL").cast<int>();
            break;
        case holoscan::LogLevel::OFF:
            python_log_level = logging.attr("CRITICAL").cast<int>() + 1;
            break;
        default:
            throw std::runtime_error("Unhandled log level");
        }
        hololink_module.attr("logging_level")(python_log_level);

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

        // Import the IMX274 sensor module and the IMX274 mode
        py::module_ imx274 = py::module_::import("hololink.sensors.imx274");
        py::object Imx274Cam = imx274.attr("dual_imx274").attr("Imx274Cam");
        py::object Imx274_Mode = imx274.attr("imx274_mode").attr("Imx274_Mode");

        // Get a handle to the camera
        py::object camera = Imx274Cam("hololink_channel"_a = hololink_channel,
            "expander_configuration"_a = expander_configuration);
        py::object imx274_mode_set = Imx274_Mode(camera_mode);
        // Set up the application
        auto application = holoscan::make_application<HoloscanApplication>(headless, fullscreen,
            cu_context, cu_device_ordinal, hololink_channel, ibv_name, ibv_port, camera,
            imx274_mode_set, frame_limit);
        application->config(configuration);

        // Run it.
        std::shared_ptr<hololink::Hololink> hololink = hololink_channel.hololink();
        hololink->start();
        hololink->reset();
        camera.attr("setup_clock")();
        camera.attr("configure")(imx274_mode_set);
        camera.attr("set_digital_gain_reg")(0x4);
        if (pattern_set) {
            camera.attr("test_pattern")(pattern);
        }
        std::cout << "Calling run" << std::endl;
        {
            // we need release the Python GIL before starting the application to make sure the
            // operators can call camera device functions
            py::gil_scoped_release release_guard;
            application->run();
        }
        hololink->stop();

        CudaCheck(cuDevicePrimaryCtxRelease(cu_device));

    } catch (std::exception& e) {
        std::cout << e.what() << std::endl;
        return -1;
    }

    return 0;
}
