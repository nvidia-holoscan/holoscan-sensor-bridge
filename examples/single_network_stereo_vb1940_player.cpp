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

// Custom operator to print timestamps from tensor metadata
class TimestampPrinterOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(TimestampPrinterOp)

    TimestampPrinterOp() = default;

    void setup(holoscan::OperatorSpec& spec) override
    {
        spec.input<holoscan::gxf::Entity>("input");
        spec.output<holoscan::gxf::Entity>("output");
        spec.param(camera_name_, "camera_name", "Camera Name", "Name of the camera for logging", std::string("unknown"));
    }

    void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output, holoscan::ExecutionContext& context) override
    {
        auto maybe_entity = op_input.receive<holoscan::gxf::Entity>("input");
        if (!maybe_entity) {
            HOLOSCAN_LOG_WARN("Failed to receive entity");
            return;
        }
        frame_count_++;
        int64_t timestamp_s = metadata()->get<int64_t>("timestamp_s", 0);
        int64_t timestamp_ns = metadata()->get<int64_t>("timestamp_ns", 0);
        double total_seconds = timestamp_s + timestamp_ns / 1e9;
        HOLOSCAN_LOG_INFO("[{}] Frame {}: timestamp={:.9f}", camera_name_.get(), frame_count_, total_seconds);
        op_output.emit(maybe_entity.value(), "output");
    }

private:
    holoscan::Parameter<std::string> camera_name_;
    int frame_count_ = 0;
};

class HoloscanStereoApplication : public holoscan::Application {
public:
    HoloscanStereoApplication(
        bool headless,
        CUcontext cuda_context,
        int cuda_device_ordinal,
        const std::vector<std::shared_ptr<hololink::DataChannel>>& hololink_channels,
        const std::string& ibv_name,
        uint32_t ibv_port,
        const std::vector<std::shared_ptr<hololink::sensors::NativeVb1940Sensor>>& cameras,
        hololink::sensors::vb1940_mode::Mode camera_mode,
        int frame_limit,
        int window_height,
        int window_width,
        const std::string& window_title,
        bool print_timestamps)
        : headless_(headless)
        , cuda_context_(cuda_context)
        , cuda_device_ordinal_(cuda_device_ordinal)
        , hololink_channels_(hololink_channels)
        , ibv_name_(ibv_name)
        , ibv_port_(ibv_port)
        , cameras_(cameras)
        , camera_mode_(camera_mode)
        , frame_limit_(frame_limit)
        , window_height_(window_height)
        , window_width_(window_width)
        , window_title_(window_title)
        , print_timestamps_(print_timestamps)
    {
        // Validate input vectors have the same size
        if (hololink_channels.size() != cameras.size() || cameras.size() != 2) {
            throw std::runtime_error("Stereo mode requires exactly 2 cameras and 2 data channels");
        }

        // Set metadata policy for stereo (similar to Python version)
        // Because we have stereo camera paths going into the same visualizer, don't
        // raise an error when each path presents metadata with the same names.
        // Because we don't use that metadata, it's easiest to just ignore new items
        // with the same names as existing items.
        enable_metadata(true);
        metadata_policy(holoscan::MetadataPolicy::kReject);
    }

    void compose() override
    {
        using namespace holoscan;

        // Create conditions for left and right cameras
        std::vector<std::shared_ptr<Condition>> conditions;
        if (frame_limit_) {
            conditions.push_back(make_condition<CountCondition>("count_left", frame_limit_));
            conditions.push_back(make_condition<CountCondition>("count_right", frame_limit_));
        } else {
            conditions.push_back(make_condition<BooleanCondition>("ok_left", true));
            conditions.push_back(make_condition<BooleanCondition>("ok_right", true));
        }

        // Set mode for both cameras
        for (auto& camera : cameras_) {
            camera->set_mode(camera_mode_);
        }

        // Create shared memory pool for CSI to Bayer conversion (6 blocks for stereo)
        auto csi_to_bayer_pool = make_resource<BlockMemoryPool>(
            "csi_to_bayer_pool",
            1, // storage_type of 1 is device memory
            cameras_[0]->get_width() * sizeof(uint16_t) * cameras_[0]->get_height(), // block_size
            6 // num_blocks (increased for stereo)
        );

        // Create CSI to Bayer operators for both cameras
        std::vector<std::shared_ptr<hololink::operators::CsiToBayerOp>> csi_to_bayer_operators;
        std::vector<std::string> tensor_names = { "left", "right" };

        for (size_t i = 0; i < cameras_.size(); ++i) {
            auto csi_to_bayer_op = make_operator<hololink::operators::CsiToBayerOp>(
                "csi_to_bayer_" + tensor_names[i],
                Arg("allocator", csi_to_bayer_pool),
                Arg("cuda_device_ordinal", cuda_device_ordinal_),
                Arg("out_tensor_name", tensor_names[i]));
            cameras_[i]->configure_converter(csi_to_bayer_op);
            csi_to_bayer_operators.push_back(csi_to_bayer_op);
        }

        // Verify both cameras have the same frame size
        size_t frame_size = csi_to_bayer_operators[0]->get_csi_length();
        if (csi_to_bayer_operators[1]->get_csi_length() != frame_size) {
            throw std::runtime_error("Both cameras must have the same frame size");
        }

        // Create receiver operators for both cameras
        std::vector<std::shared_ptr<hololink::operators::RoceReceiverOp>> receiver_operators;
        for (size_t i = 0; i < cameras_.size(); ++i) {
            auto receiver_op = make_operator<hololink::operators::RoceReceiverOp>(
                "receiver_" + tensor_names[i],
                conditions[i],
                Arg("frame_size", frame_size),
                Arg("frame_context", cuda_context_),
                Arg("ibv_name", ibv_name_),
                Arg("ibv_port", ibv_port_),
                Arg("hololink_channel", hololink_channels_[i].get()),
                Arg("device_start", std::function<void()>([this, i] { cameras_[i]->start(); })),
                Arg("device_stop", std::function<void()>([this, i] { cameras_[i]->stop(); })));
            receiver_operators.push_back(receiver_op);
        }

        // Create image processor operators for both cameras
        std::vector<std::shared_ptr<hololink::operators::ImageProcessorOp>> image_processor_operators;
        for (size_t i = 0; i < cameras_.size(); ++i) {
            auto pixel_format = static_cast<int>(cameras_[i]->get_pixel_format());
            auto bayer_format = static_cast<int>(cameras_[i]->get_bayer_format());
            auto image_processor_op = make_operator<hololink::operators::ImageProcessorOp>(
                "image_processor_" + tensor_names[i],
                Arg("optical_black", 8), // Optical black value for vb1940 is 8 for RAW10
                Arg("bayer_format", bayer_format),
                Arg("pixel_format", pixel_format));
            image_processor_operators.push_back(image_processor_op);
        }

        // Create shared memory pool for Bayer demosaic (6 blocks for stereo)
        const int rgba_components_per_pixel = 4;
        auto bayer_pool = make_resource<BlockMemoryPool>(
            "bayer_pool",
            1, // storage_type of 1 is device memory
            cameras_[0]->get_width() * rgba_components_per_pixel * sizeof(uint16_t) * cameras_[0]->get_height(), // block_size
            6 // num_blocks (increased for stereo)
        );

        // Create Bayer demosaic operators for both cameras
        std::vector<std::shared_ptr<holoscan::ops::BayerDemosaicOp>> demosaic_operators;
        for (size_t i = 0; i < cameras_.size(); ++i) {
            auto bayer_format = static_cast<int>(cameras_[i]->get_bayer_format());
            auto demosaic_op = make_operator<holoscan::ops::BayerDemosaicOp>(
                "demosaic_" + tensor_names[i],
                Arg("pool", bayer_pool),
                Arg("generate_alpha", true),
                Arg("alpha_value", 65535),
                Arg("bayer_grid_pos", bayer_format),
                Arg("interpolation_mode", 0),
                Arg("in_tensor_name", tensor_names[i]),
                Arg("out_tensor_name", tensor_names[i]));
            demosaic_operators.push_back(demosaic_op);
        }

        using HolovizOp = holoscan::ops::HolovizOp;
        std::vector<HolovizOp::InputSpec> tensor_specs;

        HolovizOp::InputSpec left_spec;
        left_spec.tensor_name_ = "left";
        left_spec.type_ = HolovizOp::InputType::COLOR;
        HolovizOp::InputSpec::View left_view;
        left_view.offset_x_ = 0.0f;
        left_view.offset_y_ = 0.0f;
        left_view.width_ = 0.5f;
        left_view.height_ = 1.0f;
        left_spec.views_.push_back(left_view);
        tensor_specs.push_back(left_spec);

        HolovizOp::InputSpec right_spec;
        right_spec.tensor_name_ = "right";
        right_spec.type_ = HolovizOp::InputType::COLOR;
        HolovizOp::InputSpec::View right_view;
        right_view.offset_x_ = 0.5f;
        right_view.offset_y_ = 0.0f;
        right_view.width_ = 0.5f;
        right_view.height_ = 1.0f;
        right_spec.views_.push_back(right_view);
        tensor_specs.push_back(right_spec);

        auto visualizer = make_operator<HolovizOp>(
            "holoviz",
            Arg("headless", headless_),
            Arg("framebuffer_srgb", true),
            Arg("tensors", tensor_specs),
            Arg("height", (uint32_t)window_height_),
            Arg("width", (uint32_t)window_width_),
            Arg("window_title", window_title_));

        std::vector<std::shared_ptr<TimestampPrinterOp>> timestamp_printers;
        if (print_timestamps_) {
            for (size_t i = 0; i < cameras_.size(); ++i) {
                std::string camera_name = tensor_names[i];
                auto timestamp_printer = make_operator<TimestampPrinterOp>(
                    "timestamp_printer_" + camera_name,
                    Arg("camera_name", camera_name));
                timestamp_printers.push_back(timestamp_printer);
            }
        }

        for (size_t i = 0; i < cameras_.size(); ++i) {
            if (print_timestamps_) {
                add_flow(receiver_operators[i], timestamp_printers[i], { { "output", "input" } });
                add_flow(timestamp_printers[i], csi_to_bayer_operators[i], { { "output", "input" } });
            } else {
                add_flow(receiver_operators[i], csi_to_bayer_operators[i], { { "output", "input" } });
            }
            add_flow(csi_to_bayer_operators[i], image_processor_operators[i], { { "output", "input" } });
            add_flow(image_processor_operators[i], demosaic_operators[i], { { "output", "receiver" } });
            add_flow(demosaic_operators[i], visualizer, { { "transmitter", "receivers" } });
        }
    }

private:
    bool headless_;
    CUcontext cuda_context_;
    int cuda_device_ordinal_;
    std::vector<std::shared_ptr<hololink::DataChannel>> hololink_channels_;
    std::string ibv_name_;
    uint32_t ibv_port_;
    std::vector<std::shared_ptr<hololink::sensors::NativeVb1940Sensor>> cameras_;
    hololink::sensors::vb1940_mode::Mode camera_mode_;
    int frame_limit_;
    int window_height_;
    int window_width_;
    std::string window_title_;
    bool print_timestamps_;
};

int main(int argc, char** argv)
{
    using namespace hololink::args;
    OptionsDescription options_description("Stereo VB1940 Player Options");

    // Select the first available device to be the default
    std::string default_ibv_name;
    auto infiniband_devices = hololink::core::infiniband_devices();
    if (!infiniband_devices.empty())
        default_ibv_name = infiniband_devices[0];

    // clang-format off
    options_description.add_options()
        ("camera-mode", value<int>()->default_value(static_cast<int>(hololink::sensors::vb1940_mode::VB1940_MODE_2560X1984_30FPS)), "VB1940 mode (default: 0)")
        ("trigger", bool_switch()->default_value(false), "Run in trigger mode for camera synchronization")
        ("exp", value<int>()->default_value(256), "Set EXPOSURE duration in lines, RANGE(4 to 65535). Default line value is 29.70usec")
        ("gain", value<int>()->default_value(0), "Set Analog Gain, RANGE(0 to 12). Default is 0. Equation is (16/(16-gain))")
        ("frequency", value<int>()->default_value(30), "VSYNC frequency in Hz (10, 30, 60, 90, 120). Default is 30Hz")
        ("headless", bool_switch()->default_value(false), "Run in headless mode")
        ("frame-limit", value<int>()->default_value(0), "Exit after receiving this many frames")
        ("configuration", value<std::string>()->default_value("example_configuration.yaml"), "Configuration file")
        ("hololink", value<std::string>()->default_value("192.168.0.2"), "IP address of Hololink board")
        ("ibv-name", value<std::string>()->default_value(default_ibv_name), "IBV device to use")
        ("ibv-port", value<uint32_t>()->default_value(1), "Port number of IBV device (default: 1)")
        ("window-height", value<int>()->default_value(2160 / 4), "Set the height of the displayed window")
        ("window-width", value<int>()->default_value(3840 / 3), "Set the width of the displayed window")
        ("title", value<std::string>(), "Set the window title")
        ("print-time", bool_switch()->default_value(false), "Print timestamp information for received frames")
        ;
    // clang-format on

    auto variables_map = Parser().parse_command_line(argc, argv, options_description);

    try {
        // Initialize CUDA
        cuInit(0);
        CUdevice cu_device;
        int cu_device_ordinal = 0;
        cuDeviceGet(&cu_device, cu_device_ordinal);
        CUcontext cu_context;
        cuDevicePrimaryCtxRetain(&cu_context, cu_device);

        // Get handles to the Hololink devices
        std::vector<std::shared_ptr<hololink::DataChannel>> hololink_channels;
        hololink_channels.reserve(2);

        // Get a handle to data sources. First, find an enumeration packet from the IP address we want to use.
        auto channel_metadata = hololink::Enumerator::find_channel(variables_map["hololink"].as<std::string>());

        // Now make separate connection metadata for left and right; and set them to use sensor 0 and 1 respectively.
        hololink::Metadata channel_metadata_left(channel_metadata);
        hololink::DataChannel::use_sensor(channel_metadata_left, 0);
        auto data_channel_left = std::make_shared<hololink::DataChannel>(channel_metadata_left);
        hololink_channels.push_back(data_channel_left);
        auto hololink = data_channel_left->hololink();

        hololink::Metadata channel_metadata_right(channel_metadata);
        hololink::DataChannel::use_sensor(channel_metadata_right, 1);
        hololink_channels.push_back(std::make_shared<hololink::DataChannel>(channel_metadata_right));

        // Get handles to the cameras
        std::vector<std::shared_ptr<hololink::sensors::NativeVb1940Sensor>> cameras;
        cameras.reserve(2);

        auto vsync = hololink::Synchronizer::null_synchronizer();
        bool trigger_mode = variables_map["trigger"].as<bool>();
        if (trigger_mode) {
            auto frequency = static_cast<unsigned>(variables_map["frequency"].as<int>());
            vsync = hololink->ptp_pps_output(frequency);
        }
        for (size_t i = 0; i < hololink_channels.size(); ++i) {
            cameras.push_back(std::make_shared<hololink::sensors::NativeVb1940Sensor>(
                *hololink_channels[i], vsync));
        }

        // Convert camera_mode to proper enum
        auto camera_mode = static_cast<hololink::sensors::vb1940_mode::Mode>(variables_map["camera-mode"].as<int>());

        // Set window title
        std::string window_title = "Holoviz - " + variables_map["hololink"].as<std::string>();
        if (variables_map.count("title")) {
            window_title = variables_map["title"].as<std::string>();
        }

        // Set up the application
        auto app = std::make_unique<HoloscanStereoApplication>(
            variables_map["headless"].as<bool>(),
            cu_context,
            cu_device_ordinal,
            hololink_channels,
            variables_map["ibv-name"].as<std::string>(),
            variables_map["ibv-port"].as<uint32_t>(),
            cameras,
            camera_mode,
            variables_map["frame-limit"].as<int>(),
            variables_map["window-height"].as<int>(),
            variables_map["window-width"].as<int>(),
            window_title,
            variables_map["print-time"].as<bool>());
        app->config(variables_map["configuration"].as<std::string>());

        // Run it - using the same stereo initialization pattern as the publisher
        hololink->start();
        HOLOSCAN_LOG_INFO("Before hololink.reset()");
        hololink->reset();

        // Set RESET low for all sensors
        hololink->write_uint32(0x8, 0x0);
        cameras[0]->setup_clock(); // this also sets camera_right's clock
        // Release RESET for all sensors - use 0x3 for stereo mode
        hololink->write_uint32(0x8, 0x3);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Verify camera connections for both cameras
        for (auto& camera : cameras) {
            camera->get_register_32(0x0000); // DEVICE_MODEL_ID:"S940"(ASCII code:0x53393430)
            camera->get_register_32(0x0734); // EXT_CLOCK(25MHz = 0x017d7840)

            camera->configure(camera_mode);
            camera->set_analog_gain_reg(variables_map["gain"].as<int>());
            camera->set_exposure_reg(variables_map["exp"].as<int>());
        }

        // Read camera EEPROM calibration data for both cameras
        try {
            auto cal_eeprom = cameras[0]->get_rgb_calibration_data();
            HOLOSCAN_LOG_INFO("Camera calibration data retrieved successfully: \n{}", cal_eeprom.to_string());
        } catch (const std::exception& e) {
            HOLOSCAN_LOG_WARN("Failed to read camera calibration data: {}", e.what());
        }

        app->run();
        hololink->stop();

        cuDevicePrimaryCtxRelease(cu_device);
        return 0;
    } catch (const std::exception& e) {
        HOLOSCAN_LOG_ERROR("Application failed: {}", e.what());
        return -1;
    }
}
