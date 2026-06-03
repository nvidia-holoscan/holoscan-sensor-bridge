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

#include <iostream>
#include <string>

#include <hololink/common/cuda_helper.hpp>
#include <hololink/common/holoargs.hpp>
#include <hololink/common/tools.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/enumerator.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/core/logging.hpp>
#include <hololink/operators/csi_to_bayer/csi_to_bayer.hpp>
#include <hololink/operators/image_processor/image_processor.hpp>
#include <hololink/operators/roce_receiver/roce_receiver_op.hpp>
#include <hololink/operators/sub_frame_visualizer/sub_frame_visualizer.hpp>
#include <hololink/sensors/camera/vb1940/native_vb1940_sensor.hpp>

#include <holoscan/core/resources/gxf/cuda_stream_pool.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/operators/bayer_demosaic/bayer_demosaic.hpp>
#include <holoscan/version_config.hpp>

#define HOLOSCAN_VERSION \
    (HOLOSCAN_VERSION_MAJOR * 10000 + HOLOSCAN_VERSION_MINOR * 100 + HOLOSCAN_VERSION_PATCH)

namespace {

class HoloscanApplication : public holoscan::Application {
public:
    explicit HoloscanApplication(bool fullscreen, bool use_exclusive_display,
        CUcontext cuda_context, int cuda_device_ordinal, hololink::DataChannel& hololink_channel,
        const std::string& ibv_name, uint32_t ibv_port,
        const std::shared_ptr<hololink::Hololink::PtpSynchronizer>& ptp_synchronizer,
        const std::shared_ptr<hololink::sensors::NativeVb1940Sensor>& camera,
        hololink::sensors::vb1940_mode::Mode camera_mode,
        int64_t frame_limit, uint32_t sub_frame_rows)
        : fullscreen_(fullscreen)
        , use_exclusive_display_(use_exclusive_display)
        , cuda_context_(cuda_context)
        , cuda_device_ordinal_(cuda_device_ordinal)
        , hololink_channel_(hololink_channel)
        , ibv_name_(ibv_name)
        , ibv_port_(ibv_port)
        , ptp_synchronizer_(ptp_synchronizer)
        , camera_(camera)
        , camera_mode_(camera_mode)
        , frame_limit_(frame_limit)
        , sub_frame_rows_(sub_frame_rows)
    {
    }
    HoloscanApplication() = delete;

    void compose() override
    {
        auto memory_pool = make_resource<holoscan::RMMAllocator>("memory_pool",
            holoscan::Arg("device_memory_max_size", "1GB"));

        std::shared_ptr<holoscan::Condition> condition;
        if (frame_limit_) {
            condition = make_condition<holoscan::CountCondition>("count", frame_limit_);
        } else {
            condition = make_condition<holoscan::BooleanCondition>("ok", true);
        }
        camera_->set_mode(camera_mode_);

        const int64_t width = camera_->get_width();
        const int64_t height = camera_->get_height();

        uint32_t sub_frame_rows = sub_frame_rows_;
        if (sub_frame_rows == 0) {
            sub_frame_rows = height;
        } else if (sub_frame_rows > height) {
            throw std::runtime_error(fmt::format("Sub frame rows of {} is greater than camera height of {}", sub_frame_rows, height));
        }

        auto csi_to_bayer_operator = make_operator<hololink::operators::CsiToBayerOp>("csi_to_bayer",
            holoscan::Arg("allocator", memory_pool),
            holoscan::Arg("sub_frame_rows", sub_frame_rows),
            holoscan::Arg("cuda_device_ordinal", cuda_device_ordinal_));
        camera_->configure_converter(csi_to_bayer_operator);

        const size_t frame_size = csi_to_bayer_operator->get_sub_frame_size();
        constexpr uint32_t MAX_PAGES = 30; // To have enough buffers for at least 3 complete frames
        auto receiver_operator = make_operator<hololink::operators::RoceReceiverOp>("receiver",
            condition,
            holoscan::Arg("frame_size", frame_size),
            holoscan::Arg("frame_context", cuda_context_),
            holoscan::Arg("ibv_name", ibv_name_),
            holoscan::Arg("ibv_port", ibv_port_),
            holoscan::Arg("hololink_channel", &hololink_channel_),
            holoscan::Arg("device_start", std::function<void()>([this] {
                camera_->start();
            })),
            holoscan::Arg("device_stop", std::function<void()>([this] {
                camera_->stop();
            })),
            // Set the number of the pages to the maximum number of pages
            holoscan::Arg("pages", MAX_PAGES),
            // Also set the queue size to the number of pages to reduce the probability of dropping frames
            holoscan::Arg("queue_size", MAX_PAGES),
            // Trim the frame to the size of the received data (when doing sub-frame processing, not
            // each received packet might have the same size when then camera CSI data has a header)
            holoscan::Arg("trim", true),
            // Disable the frame ready condition to reduce latency
            holoscan::Arg("use_frame_ready_condition", false));

        auto bayer_format = static_cast<int>(camera_->get_bayer_format());
        auto pixel_format = static_cast<int>(camera_->get_pixel_format());
        auto image_processor_operator = make_operator<hololink::operators::ImageProcessorOp>("image_processor",
            // Optical black value for vb1940 is 8 for RAW10
            holoscan::Arg("optical_black", 8), holoscan::Arg("bayer_format", bayer_format),
            holoscan::Arg("pixel_format", pixel_format));

        auto demosaic = make_operator<holoscan::ops::BayerDemosaicOp>("demosaic",
            holoscan::Arg("pool", memory_pool), holoscan::Arg("generate_alpha", true),
            holoscan::Arg("alpha_value", 65535), holoscan::Arg("bayer_grid_pos", bayer_format),
            holoscan::Arg("interpolation_mode", 0));

        add_flow(receiver_operator, csi_to_bayer_operator, { { "output", "input" } });
        add_flow(csi_to_bayer_operator, image_processor_operator, { { "output", "input" } });
        add_flow(image_processor_operator, demosaic, { { "output", "receiver" } });

        auto sub_frame_visualizer = make_operator<hololink::operators::SubFrameVisualizerOp>("sub_frame_visualizer",
            holoscan::Arg("ptp_synchronizer", ptp_synchronizer_.get()),
            holoscan::Arg("fullscreen", fullscreen_),
            holoscan::Arg("use_exclusive_display", use_exclusive_display_),
            holoscan::Arg("full_frame_height", static_cast<uint32_t>(height)));

        if (fullscreen_ || use_exclusive_display_) {
            sub_frame_visualizer->add_arg(holoscan::Arg("display_width", static_cast<uint32_t>(width)));
            sub_frame_visualizer->add_arg(holoscan::Arg("display_height", static_cast<uint32_t>(height)));
        }

        add_flow(demosaic, sub_frame_visualizer, { { "transmitter", "input" } });
    }

private:
    const bool fullscreen_;
    const bool use_exclusive_display_;
    const CUcontext cuda_context_;
    const int cuda_device_ordinal_;
    hololink::DataChannel& hololink_channel_;
    const std::string ibv_name_;
    const uint32_t ibv_port_;
    const std::shared_ptr<hololink::Hololink::PtpSynchronizer> ptp_synchronizer_;
    const std::shared_ptr<hololink::sensors::NativeVb1940Sensor> camera_;
    const hololink::sensors::vb1940_mode::Mode camera_mode_;
    const int64_t frame_limit_;
    const uint32_t sub_frame_rows_;
};

} // anonymous namespace

int main(int argc, char** argv)
{
    using namespace hololink::args;
    OptionsDescription options_description("Allowed options");

    // Select the first available device to be the default
    std::string default_ibv_name;
    auto infiniband_devices = hololink::infiniband_devices();
    if (!infiniband_devices.empty())
        default_ibv_name = infiniband_devices[0];

    // clang-format off
    options_description.add_options()
        ("camera-mode", value<int>()->default_value(static_cast<int>(hololink::sensors::vb1940_mode::VB1940_MODE_2560X1984_60FPS)), "VB1940 mode (default: 3)")
        ("fullscreen", bool_switch()->default_value(true), "Run in fullscreen mode")
        ("exclusive-display", bool_switch()->default_value(false), "Run in exclusive display mode")
        ("frame-limit", value<int64_t>()->default_value(0), "Exit after receiving this many frames")
        ("sub-frame-rows", value<uint32_t>()->default_value(0), "Number of rows per sub-frame (0 = full camera height). Camera height must be evenly divisible by the sub frame rows.")
        ("configuration", value<std::string>()->default_value(""), "Configuration file")
        ("hololink", value<std::string>()->default_value("192.168.0.2"), "IP address of Hololink board")
        ("ibv-name", value<std::string>()->default_value(default_ibv_name), "IBV device to use")
        ("ibv-port", value<uint32_t>()->default_value(1), "Port number of IBV device (default: 1)")
        ("use-sensor", value<uint32_t>()->default_value(0), "Use the specific sensor (0 or 1, default=0)");
    // clang-format on

    auto variables_map = Parser().parse_command_line(argc, argv, options_description);
    int use_sensor = variables_map["use-sensor"].as<uint32_t>();
    if (use_sensor >= 2) {
        HOLOSCAN_LOG_ERROR("Invalid value use_sensor={}; must be 0 or 1.", use_sensor);
        return -1;
    }

    try {
        // Get a handle to the GPU
        CudaCheck(cuInit(0));
        int cu_device_ordinal = 0;
        CUdevice cu_device;
        CudaCheck(cuDeviceGet(&cu_device, cu_device_ordinal));
        CUcontext cu_context;
        CudaCheck(cuDevicePrimaryCtxRetain(&cu_context, cu_device));

        // Get a handle to the Hololink device
        auto channel_metadata = hololink::Enumerator::find_channel(variables_map["hololink"].as<std::string>());
        hololink::DataChannel::use_sensor(channel_metadata, use_sensor);
        hololink::DataChannel hololink_channel(channel_metadata);

        // Create a PTP synchronizer for single pulse, the delay is then set per frame in the SubFrameVisualizerOp
        auto ptp_synchronizer = hololink_channel.hololink()->ptp_pps_output(1);
        // Get a handle to the camera
        auto camera = std::make_shared<hololink::sensors::NativeVb1940Sensor>(hololink_channel, ptp_synchronizer);

        // Convert camera_mode to proper enum
        auto camera_mode = static_cast<hololink::sensors::vb1940_mode::Mode>(variables_map["camera-mode"].as<int>());

        // Set up the application
        auto application = holoscan::make_application<HoloscanApplication>(
            variables_map["fullscreen"].as<bool>(),
            variables_map["exclusive-display"].as<bool>(),
            cu_context,
            cu_device_ordinal,
            hololink_channel,
            variables_map["ibv-name"].as<std::string>(),
            variables_map["ibv-port"].as<uint32_t>(),
            std::static_pointer_cast<hololink::Hololink::PtpSynchronizer>(ptp_synchronizer),
            camera,
            camera_mode,
            variables_map["frame-limit"].as<int64_t>(),
            variables_map["sub-frame-rows"].as<uint32_t>());
        auto configuration = variables_map["configuration"].as<std::string>();
        if (!configuration.empty()) {
            application->config(configuration);
        }

        // Run it.
        std::shared_ptr<hololink::Hololink> hololink = hololink_channel.hololink();
        hololink->start();
        hololink->reset();
        // Wait for PTP synchronization
        hololink->ptp_synchronize();
        hololink->write_uint32(0x8, 0x0); // Keep the sensor RESET at low
        camera->setup_clock();
        // Release the sensor RESET to high
        if (use_sensor == 0) {
            hololink->write_uint32(0x8, 0x1);
        } else if (use_sensor == 1) {
            hololink->write_uint32(0x8, 0x2);
        } else {
            // We already checked this above; so this is redundant.
            throw std::runtime_error(fmt::format("Invalid value use_sensor={}; must be 0 or 1.", use_sensor));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        camera->configure(camera_mode);

        application->run();
        hololink->stop();

        CudaCheck(cuDevicePrimaryCtxRelease(cu_device));

    } catch (std::exception& e) {
        std::cout << e.what() << std::endl;
        return -1;
    }

    return 0;
}
