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

#include <getopt.h>

#include <cuda.h>

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "hawk.hpp"
#include "hololink/module/adapter.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/hololink.hpp"
#include "hololink/module/operators/linux_receiver_op.hpp"
#include "hololink/module/taurotech_da326/taurotech_da326.hpp"
#include "max96716a.hpp"
#include <hololink/common/cuda_helper.hpp>
#include <hololink/common/tools.hpp>
#include <hololink/core/logging.hpp>
#include <hololink/operators/csi_to_bayer/csi_to_bayer.hpp>
#include <hololink/operators/image_processor/image_processor.hpp>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/bayer_demosaic/bayer_demosaic.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

namespace {

using hololink::module::sensors::ar0234::Ar0234;
using hololink::module::sensors::hawk::Hawk;
using hololink::module::sensors::max96716a::Max96716a;

class HoloscanApplication : public holoscan::Application {
public:
    HoloscanApplication(bool headless, bool fullscreen, CUcontext cuda_context,
        int cuda_device_ordinal,
        std::vector<hololink::module::EnumerationMetadata> sif_metadatas,
        std::shared_ptr<Hawk> camera,
        hololink::module::sensors::ar0234::ar0234_mode::Mode camera_mode,
        int64_t frame_limit,
        int32_t window_height, int32_t window_width,
        std::vector<std::string> window_titles)
        : headless_(headless)
        , fullscreen_(fullscreen)
        , cuda_context_(cuda_context)
        , cuda_device_ordinal_(cuda_device_ordinal)
        , sif_metadatas_(std::move(sif_metadatas))
        , camera_(std::move(camera))
        , camera_mode_(camera_mode)
        , frame_limit_(frame_limit)
        , window_height_(window_height)
        , window_width_(window_width)
        , window_titles_(std::move(window_titles))
    {
    }

    void compose() override
    {
        camera_->set_mode(camera_mode_);
        for (size_t idx = 0; idx < sif_metadatas_.size(); ++idx) {
            compose_pipeline(idx, sif_metadatas_[idx], window_titles_[idx]);
        }
    }

private:
    void compose_pipeline(size_t idx,
        const hololink::module::EnumerationMetadata& sif_metadata,
        const std::string& window_title)
    {
        const std::string suffix = std::to_string(idx);
        std::shared_ptr<holoscan::Condition> condition;
        if (frame_limit_) {
            condition = make_condition<holoscan::CountCondition>("count_" + suffix, frame_limit_);
        } else {
            condition = make_condition<holoscan::BooleanCondition>("ok_" + suffix, true);
        }

        auto csi_to_bayer_pool = make_resource<holoscan::BlockMemoryPool>("csi_pool_" + suffix,
            // storage_type of 1 is device memory
            1,
            camera_->width() * sizeof(uint16_t) * camera_->height(),
            2 // num_blocks
        );
        auto csi_to_bayer_operator = make_operator<hololink::operators::CsiToBayerOp>(
            "csi_to_bayer_" + suffix,
            holoscan::Arg("allocator", csi_to_bayer_pool),
            holoscan::Arg("cuda_device_ordinal", cuda_device_ordinal_));
        camera_->configure_converter(csi_to_bayer_operator);

        const size_t frame_size = csi_to_bayer_operator->get_csi_length();
        auto receiver_operator = make_operator<hololink::module::operators::LinuxReceiverOp>(
            "receiver_" + suffix, condition,
            holoscan::Arg("enumeration_metadata", sif_metadata),
            holoscan::Arg("frame_context", cuda_context_),
            holoscan::Arg("frame_size", frame_size),
            holoscan::Arg("device_start", std::function<void()>([this] { camera_->start(); })),
            holoscan::Arg("device_stop", std::function<void()>([this] { camera_->stop(); })));

        auto bayer_format = camera_->bayer_format();
        auto pixel_format = camera_->pixel_format();
        auto image_processor_operator = make_operator<hololink::operators::ImageProcessorOp>(
            "image_processor_" + suffix,
            holoscan::Arg("optical_black", 50),
            holoscan::Arg("bayer_format", static_cast<int>(bayer_format)),
            holoscan::Arg("pixel_format", static_cast<int>(pixel_format)));

        const uint32_t rgba_components_per_pixel = 4;
        auto bayer_pool = make_resource<holoscan::BlockMemoryPool>("bayer_pool_" + suffix,
            1,
            camera_->width() * rgba_components_per_pixel * sizeof(uint16_t) * camera_->height(),
            2);
        auto demosaic = make_operator<holoscan::ops::BayerDemosaicOp>("demosaic_" + suffix,
            holoscan::Arg("pool", bayer_pool),
            holoscan::Arg("generate_alpha", true),
            holoscan::Arg("alpha_value", 65535),
            holoscan::Arg("bayer_grid_pos", static_cast<int>(bayer_format)),
            holoscan::Arg("interpolation_mode", 0));

        auto visualizer = make_operator<holoscan::ops::HolovizOp>("holoviz_" + suffix,
            holoscan::Arg("framebuffer_srgb", true),
            holoscan::Arg("fullscreen", fullscreen_),
            holoscan::Arg("headless", headless_),
            holoscan::Arg("height", window_height_),
            holoscan::Arg("width", window_width_),
            holoscan::Arg("window_title", window_title));

        add_flow(receiver_operator, csi_to_bayer_operator, { { "output", "input" } });
        add_flow(csi_to_bayer_operator, image_processor_operator, { { "output", "input" } });
        add_flow(image_processor_operator, demosaic, { { "output", "receiver" } });
        add_flow(demosaic, visualizer, { { "transmitter", "receivers" } });
    }

    const bool headless_;
    const bool fullscreen_;
    const CUcontext cuda_context_;
    const int cuda_device_ordinal_;
    std::vector<hololink::module::EnumerationMetadata> sif_metadatas_;
    std::shared_ptr<Hawk> camera_;
    hololink::module::sensors::ar0234::ar0234_mode::Mode camera_mode_;
    const int64_t frame_limit_;
    const int32_t window_height_;
    const int32_t window_width_;
    std::vector<std::string> window_titles_;
};

// Sensor selector mirrors Python SENSOR_ARG_TO_SIF.
std::vector<int64_t> sensor_arg_to_sif(const std::string& sensor)
{
    if (sensor == "left")
        return { 0 };
    if (sensor == "right")
        return { 1 };
    if (sensor == "both")
        return { 0, 1 };
    throw std::runtime_error("Invalid --sensor value: " + sensor);
}

} // anonymous namespace

int main(int argc, char** argv)
{
    int32_t camera_mode_int = hololink::module::sensors::ar0234::ar0234_mode::AR0234_MODE_1920X1200_60FPS;
    bool headless = false;
    bool fullscreen = false;
    int64_t frame_limit = 0;
    std::string configuration;
    std::string hololink_ip = hololink::env_hololink_ip(0, "192.168.0.2");
    holoscan::LogLevel log_level = holoscan::LogLevel::INFO;
    int32_t window_width = 1920;
    int32_t window_height = 1080;
    std::string title = "AR0234";
    std::string sensor_arg = "left";
    std::string channel_arg = "A";
    uint16_t exposure = 0x02DC;
    int32_t pattern = 0;
    bool pattern_set = false;
    bool skip_setup = false;
    bool skip_reset = false;

    const struct option long_options[] = {
        { "help", no_argument, nullptr, 'h' },
        { "camera-mode", required_argument, nullptr, 0 },
        { "headless", no_argument, nullptr, 0 },
        { "fullscreen", no_argument, nullptr, 0 },
        { "frame-limit", required_argument, nullptr, 0 },
        { "configuration", required_argument, nullptr, 0 },
        { "hololink", required_argument, nullptr, 0 },
        { "log-level", required_argument, nullptr, 0 },
        { "window-height", required_argument, nullptr, 0 },
        { "window-width", required_argument, nullptr, 0 },
        { "title", required_argument, nullptr, 0 },
        { "sensor", required_argument, nullptr, 0 },
        { "channel", required_argument, nullptr, 0 },
        { "exposure", required_argument, nullptr, 0 },
        { "pattern", required_argument, nullptr, 0 },
        { "skip-setup", no_argument, nullptr, 0 },
        { "skip-reset", no_argument, nullptr, 0 },
        { 0, 0, nullptr, 0 }
    };

    while (true) {
        int option_index = 0;
        const int c = getopt_long(argc, argv, "h", long_options, &option_index);
        if (c == -1)
            break;
        const std::string argument(optarg ? optarg : "");
        if (c == 0) {
            const std::string name = long_options[option_index].name;
            if (name == "camera-mode")
                camera_mode_int = std::stoi(argument);
            else if (name == "headless")
                headless = true;
            else if (name == "fullscreen")
                fullscreen = true;
            else if (name == "frame-limit")
                frame_limit = std::stoll(argument);
            else if (name == "configuration")
                configuration = argument;
            else if (name == "hololink")
                hololink_ip = argument;
            else if (name == "window-height")
                window_height = std::stoi(argument);
            else if (name == "window-width")
                window_width = std::stoi(argument);
            else if (name == "title")
                title = argument;
            else if (name == "sensor")
                sensor_arg = argument;
            else if (name == "channel")
                channel_arg = argument;
            else if (name == "exposure")
                exposure = static_cast<uint16_t>(std::stoul(argument, nullptr, 0));
            else if (name == "pattern") {
                pattern = std::stoi(argument, nullptr, 0);
                pattern_set = true;
            } else if (name == "skip-setup")
                skip_setup = true;
            else if (name == "skip-reset")
                skip_reset = true;
            else if (name == "log-level") {
                if (argument == "trace" || argument == "TRACE")
                    log_level = holoscan::LogLevel::TRACE;
                else if (argument == "debug" || argument == "DEBUG")
                    log_level = holoscan::LogLevel::DEBUG;
                else if (argument == "info" || argument == "INFO")
                    log_level = holoscan::LogLevel::INFO;
                else if (argument == "warn" || argument == "WARN")
                    log_level = holoscan::LogLevel::WARN;
                else if (argument == "error" || argument == "ERROR")
                    log_level = holoscan::LogLevel::ERROR;
                else if (argument == "critical" || argument == "CRITICAL")
                    log_level = holoscan::LogLevel::CRITICAL;
                else if (argument == "off" || argument == "OFF")
                    log_level = holoscan::LogLevel::OFF;
                else
                    throw std::runtime_error("Unhandled log level: " + argument);
            }
        } else if (c == 'h') {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl
                      << "Stream from a Hawk module on one GMSL link." << std::endl;
            return EXIT_SUCCESS;
        }
    }

    holoscan::set_log_level(log_level);

    // CUDA
    CudaCheck(cuInit(0));
    int cu_device_ordinal = 0;
    CUdevice cu_device;
    CudaCheck(cuDeviceGet(&cu_device, cu_device_ordinal));
    CUcontext cu_context;
    CudaCheck(cuDevicePrimaryCtxRetain(&cu_context, cu_device));

    // hololink channel(s) setup — one SIF metadata per requested sensor.
    auto sif_list = sensor_arg_to_sif(sensor_arg);

    auto& adapter = hololink::module::Adapter::get_adapter();
    hololink::module::EnumerationMetadata metadata
        = adapter.wait_for_channel(hololink_ip, std::chrono::seconds(30));

    std::vector<hololink::module::EnumerationMetadata> sif_metadatas;
    for (auto sif : sif_list) {
        hololink::module::EnumerationMetadata sif_meta = metadata;
        adapter.use_sensor(sif_meta, sif);
        sif_metadatas.push_back(std::move(sif_meta));
    }

    auto holo_iface = hololink::module::HololinkInterfaceV1::get_service(metadata);
    holo_iface->start();
    auto board = hololink::module::taurotech_da326::TauroTechDa326InterfaceV1::get_service(metadata);

    auto legacy_holo = board->hololink();
    auto deserializer = std::make_shared<Max96716a>(legacy_holo);
    auto hawk = std::make_shared<Hawk>(legacy_holo, skip_setup);
    auto camera_mode = static_cast<hololink::module::sensors::ar0234::ar0234_mode::Mode>(camera_mode_int);

    // Build window titles (one per channel) — match Python's "left"/"right" labels.
    std::vector<std::string> sif_name = { "left", "right" };
    std::vector<std::string> window_titles;
    for (auto sif : sif_list) {
        window_titles.push_back(title + " - " + sif_name[sif]);
    }

    int32_t per_w = window_width;
    int32_t per_h = window_height;
    if (sif_list.size() > 1) {
        per_w /= 2;
        per_h /= 2;
    }

    auto application = holoscan::make_application<HoloscanApplication>(headless, fullscreen,
        cu_context, cu_device_ordinal, sif_metadatas, hawk, camera_mode,
        frame_limit, per_h, per_w, window_titles);
    application->config(configuration);

    try {
        if (!skip_setup) {
            if (!skip_reset) {
                holo_iface->reset();
            }

            board->release_reset();
            board->power_cycle();
            board->check_power();
            board->setup_clock();

            // Deser sanity check
            uint8_t dev_id = deserializer->get_register(Max96716a::DEV_ID_REG);
            if (dev_id != Max96716a::DEV_ID) {
                throw std::runtime_error(fmt::format(
                    "Deserializer mismatch: expected 0x{:02X}, got 0x{:02X}",
                    Max96716a::DEV_ID, dev_id));
            }

            // Only enable the link we care about
            auto link = (channel_arg == "B") ? Max96716a::GmslLink::LINK_B
                                             : Max96716a::GmslLink::LINK_A;
            deserializer->enable_link_exclusive(link);

            // Serializer sanity check
            uint8_t ser_id = hawk->serializer().get_register(hololink::module::sensors::max9295d::Max9295d::DEV_ID_REG);
            if (ser_id != hololink::module::sensors::max9295d::Max9295d::DEV_ID) {
                throw std::runtime_error(fmt::format(
                    "Serializer mismatch: expected 0x{:02X}, got 0x{:02X}",
                    hololink::module::sensors::max9295d::Max9295d::DEV_ID, ser_id));
            }

            // Sensor sanity check
            for (auto& sensor : hawk->sensors()) {
                uint16_t s_id = sensor->get_register(Ar0234::DEV_ID_REG);
                if (s_id != Ar0234::DEV_ID) {
                    throw std::runtime_error(fmt::format(
                        "Sensor mismatch at i2c=0x{:02X}: expected 0x{:04X}, got 0x{:04X}",
                        sensor->get_i2c_address(), Ar0234::DEV_ID, s_id));
                }
            }

            // Deserializer settings
            deserializer->configure_video_pipe();
            uint8_t left = deserializer->stream_id_to_pipe_mapping(
                link, 0, Max96716a::VideoPipe::PIPE_Y);
            uint8_t right = deserializer->stream_id_to_pipe_mapping(
                link, 2, Max96716a::VideoPipe::PIPE_Z);
            deserializer->set_register(Max96716a::VIDEO_PIPE_SEL,
                static_cast<uint8_t>(left | right));

            // Sensor programming
            hawk->configure(camera_mode);

            if (pattern_set) {
                hawk->test_pattern(static_cast<uint16_t>(pattern));
            }

            hawk->set_exposure_reg(exposure);
        }

        application->run();
    } catch (...) {
        holo_iface->stop();
        throw;
    }
    holo_iface->stop();

    CudaCheck(cuDevicePrimaryCtxRelease(cu_device));
    return 0;
}
