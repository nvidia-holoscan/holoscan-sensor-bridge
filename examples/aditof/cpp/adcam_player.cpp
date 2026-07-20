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
#include <hololink/operators/linux_receiver/linux_receiver_op.hpp>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/bayer_demosaic/bayer_demosaic.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include "adcam_lib.hpp"
#include "adcam_unpack_op.hpp"
#include "programmer.hpp"

namespace {

//==============================================================================
//  HoloscanApplication
//------------------------------------------------------------------------------
//  This class builds the Holoscan operator graph for:
//      • Receiving CSI/MIPI frames (ROCE or Linux path)
//      • Converting CSI → Bayer
//      • Unpacking ADI ToF 5‑byte/pixel format
//      • Visualizing Depth / ActiveBrightness / Confidence
//==============================================================================

class HoloscanApplication : public holoscan::Application {
public:
    //--------------------------------------------------------------------------
    // Constructor
    //--------------------------------------------------------------------------
    explicit HoloscanApplication(bool headless,
                                 bool fullscreen,
                                 CUcontext cuda_context,
                                 int cuda_device_ordinal,
                                 std::shared_ptr<hololink::DataChannel> hololink_channel,
                                 const std::string& ibv_name,
                                 uint32_t ibv_port,
                                 std::shared_ptr<hololink::sensors::Adcam> adcam_inst_,
                                 int64_t frame_limit)
        : headless_(headless)
        , fullscreen_(fullscreen)
        , cuda_context_(cuda_context)
        , cuda_device_ordinal_(cuda_device_ordinal)
        , hololink_channel_(hololink_channel)
        , ibv_name_(ibv_name)
        , ibv_port_(ibv_port)
        , adcam_inst(adcam_inst_)
        , frame_limit_(frame_limit)
    {}

    HoloscanApplication() = delete;

    //--------------------------------------------------------------------------
    // Compose: Build the Holoscan operator graph
    //--------------------------------------------------------------------------
    void compose() override {

        //======================================================================
        // 1. Frame limiting condition
        //======================================================================
        std::shared_ptr<holoscan::Condition> condition;
        if (frame_limit_) {
            condition = make_condition<holoscan::CountCondition>("count", frame_limit_);
        } else {
            condition = make_condition<holoscan::BooleanCondition>("ok", true);
        }
        //======================================================================
        // 2. Memory pool for CSI → Bayer conversion
        //======================================================================
        auto csi_to_bayer_pool = make_resource<holoscan::BlockMemoryPool>(
            "csi_to_bayer_pool",
            /*storage_type=*/1,   // device memory
            adcam_inst->get_width() * adcam_inst->get_pixel_size_bytes() * adcam_inst->get_height() * sizeof(uint16_t),
            /*num_blocks=*/2
        );

        //======================================================================
        // 3. CSI → Bayer operator
        //======================================================================
        auto csi_to_bayer_operator =
            make_operator<hololink::operators::CsiToBayerOp>(
                "csi_to_bayer",
                holoscan::Arg("allocator", csi_to_bayer_pool),
                holoscan::Arg("cuda_device_ordinal", cuda_device_ordinal_));

        // Converter interface
        std::shared_ptr<hololink::csi::CsiConverter> csi_converter =
            csi_to_bayer_operator;

        //======================================================================
        // 4. Camera initialization and configuration
        //======================================================================
        // Configure CSI converter and camera mode
        adcam_inst->configure_converter(csi_to_bayer_operator);
        adcam_inst->set_mipi();
        adcam_inst->set_mode();

        // Frame size after CSI conversion
        const size_t frame_size = csi_to_bayer_operator->get_csi_length();
        HOLOSCAN_LOG_INFO("Acquire frame of size {}", frame_size);
        // TODO [DBG] Full sizing breakdown for MP/QMP analysis.
        HOLOSCAN_LOG_INFO(
            "[DBG compose] mode={} mipi={}x{} pixel={}x{}"
            " mipi_frame_bytes={} pixel_frame_bytes={} csi_frame_bytes={}",
            adcam_inst->get_mode(), adcam_inst->get_width(),
            adcam_inst->get_height(), adcam_inst->get_abs_width(),
            adcam_inst->get_abs_height(),
            adcam_inst->get_width() * adcam_inst->get_height(),
            adcam_inst->get_abs_width() * adcam_inst->get_abs_height() * adcam_inst->get_pixel_size_bytes(),
            frame_size);

        //======================================================================
        // 5. Receiver operator (ROCE or Linux)
        //======================================================================
        std::shared_ptr<holoscan::Operator> receiver_operator;

        if (!ibv_name_.empty()) {
            // ---------------- ROCE path ----------------
            HOLOSCAN_LOG_DEBUG("Using ROCE operator to receive");

            receiver_operator =
                make_operator<hololink::operators::RoceReceiverOp>(
                    "receiver",
                    condition,
                    holoscan::Arg("frame_size", frame_size),
                    holoscan::Arg("frame_context", cuda_context_),
                    holoscan::Arg("ibv_name", ibv_name_),
                    holoscan::Arg("ibv_port", ibv_port_),
                    holoscan::Arg("hololink_channel", hololink_channel_.get()),
                    holoscan::Arg("device_start", std::function<void()>([this] {
                        adcam_inst->start();
                    })),
                    holoscan::Arg("device_stop", std::function<void()>([this] {
                        adcam_inst->stop();
                    }))
                );

        } else {
            // ---------------- Linux path ----------------
            HOLOSCAN_LOG_DEBUG("Using Linux operator to receive");

            receiver_operator =
                make_operator<hololink::operators::LinuxReceiverOp>(
                    "receiver",
                    condition,
                    holoscan::Arg("frame_size", frame_size),
                    holoscan::Arg("frame_context", cuda_context_),
                    holoscan::Arg("hololink_channel", hololink_channel_.get()),
                    holoscan::Arg("device_start", std::function<void()>([this] {
                        adcam_inst->start();
                    })),
                    holoscan::Arg("device_stop", std::function<void()>([this] {
                        adcam_inst->stop();
                    }))
                );
        }

        //======================================================================
        // 6. Memory pool for ADI ToF unpack operator
        //======================================================================
        auto device_allocator_adtf =
            make_resource<holoscan::BlockMemoryPool>(
                "ADTF_output_pool",
                /*storage_type=*/1,
                adcam_inst->get_width() * adcam_inst->get_pixel_size_bytes() * adcam_inst->get_height() * sizeof(uint16_t),
                /*num_blocks=*/8);

        //int num_planes = adcam_inst->get_mode() < 2 ? 2:3;
        int num_planes = adcam_inst->get_numPlane();
        
        //======================================================================
        // 7. ADI ToF unpack operator (5‑byte/pixel → Depth/AB/Conf)
        //======================================================================
        auto ADIToF_data =
            make_operator<hololink::operators::ADTFUnpackOp>(
                "ADIToF_data",
                holoscan::Arg("num_planes", num_planes),
                holoscan::Arg("width", (int)adcam_inst->get_abs_width()),
                holoscan::Arg("height", (int)adcam_inst->get_abs_height()),
                holoscan::Arg("allocator", device_allocator_adtf),
                holoscan::Arg("in_tensor_name", ""),
                holoscan::Arg("out_tensor_name", "output")); 

        //======================================================================
        // 8. Holoviz visualization setup
        //======================================================================

        std::shared_ptr<holoscan::ops::HolovizOp> visualizer;
        if (num_planes == 2)
        {
            // ----- Left: Depth -----
            holoscan::ops::HolovizOp::InputSpec left_spec{
                "Depth", holoscan::ops::HolovizOp::InputType::COLOR};
            left_spec.views_ = { {0.0f, 0.0f, 0.50f, 1.0f} };

            // ----- Center: ActiveBrightness -----
            holoscan::ops::HolovizOp::InputSpec center_spec{
                "ActiveBrightness", holoscan::ops::HolovizOp::InputType::COLOR};
            center_spec.views_ = { {0.51f, 0.0f, 0.51f, 1.0f} };

            const std::string window_title = "ADI ToF Player";

            visualizer =
                make_operator<holoscan::ops::HolovizOp>(
                    "holoviz",
                    holoscan::Arg("headless", headless_),
                    holoscan::Arg("framebuffer_srgb", true),
                    holoscan::Arg("tensors",
                        std::vector<holoscan::ops::HolovizOp::InputSpec>{
                            left_spec, center_spec}),
                    holoscan::Arg("window_title", window_title));
        }
        else
        {
            // ----- Left: Depth -----
            holoscan::ops::HolovizOp::InputSpec left_spec{
                "Depth", holoscan::ops::HolovizOp::InputType::COLOR};
            left_spec.views_ = { {0.0f, 0.0f, 0.33f, 1.0f} };

            // ----- Center: ActiveBrightness -----
            holoscan::ops::HolovizOp::InputSpec center_spec{
                "ActiveBrightness", holoscan::ops::HolovizOp::InputType::COLOR};
            center_spec.views_ = { {0.33f, 0.0f, 0.33f, 1.0f} };

            // ----- Right: Confidence -----
            holoscan::ops::HolovizOp::InputSpec right_spec{
                "Conf", holoscan::ops::HolovizOp::InputType::COLOR};
            right_spec.views_ = { {0.66f, 0.0f, 0.34f, 1.0f} };

            const std::string window_title = "ADI ToF Player";

            visualizer =
                make_operator<holoscan::ops::HolovizOp>(
                    "holoviz",
                    holoscan::Arg("headless", headless_),
                    holoscan::Arg("framebuffer_srgb", true),
                    holoscan::Arg("tensors",
                        std::vector<holoscan::ops::HolovizOp::InputSpec>{
                            left_spec, center_spec, right_spec}),
                    holoscan::Arg("window_title", window_title));

        }

        //======================================================================
        // 9. Connect operators (data flow graph)
        //======================================================================
        add_flow(receiver_operator, csi_to_bayer_operator, {{"output", "input"}});
        add_flow(csi_to_bayer_operator, ADIToF_data, {{"output", "input"}});
        add_flow(ADIToF_data, visualizer, {{"output", "receivers"}});
    }

private:
    //--------------------------------------------------------------------------
    // Member variables
    //--------------------------------------------------------------------------
    const bool headless_;
    const bool fullscreen_;
    const CUcontext cuda_context_;
    const int cuda_device_ordinal_;
    std::shared_ptr<hololink::DataChannel> hololink_channel_;
    const std::string ibv_name_;
    const uint32_t ibv_port_;
    std::shared_ptr<hololink::sensors::Adcam> adcam_inst;
    const int64_t frame_limit_;
};

} // anonymous namespace

int main(int argc, char** argv)
{
    //--------------------------------------------------------------------------
    // 1. Default configuration values
    //--------------------------------------------------------------------------
    int32_t adcam_mode = 6;
    int32_t do_reset = 0;
    int32_t do_capture = 0;
    std::string firmware_manifest;
    int32_t reset_pin = 0;
    int32_t num_planes = 3;
    int32_t tof_fps = 30;
    int32_t metadata_sz = 0;
    uint16_t mipi_lane_speed = MIPI_SPEED_2_5_GBPS; /* 2.5Gbps/lane */
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

    std::string ibv_name;  // autodetected below

    //--------------------------------------------------------------------------
    // 2. Attempt to auto-detect an InfiniBand device name
    //--------------------------------------------------------------------------
    try {
        auto devices = hololink::infiniband_devices();
        ibv_name = devices.size() > 0 ? devices[0] : "";
    } catch (const std::exception& e) {
        std::cerr << "Error getting IBV name: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    //--------------------------------------------------------------------------
    // 3. Command-line argument parsing
    //--------------------------------------------------------------------------
    const struct option long_options[] = {
        { "help", no_argument, nullptr, 'h' },
        { "captureMode", required_argument, nullptr, 0 },
        { "headless", no_argument, nullptr, 0 },
        { "fullscreen", no_argument, nullptr, 0 },
        { "frame-limit", required_argument, nullptr, 0 },
        { "configuration", required_argument, nullptr, 0 },
        { "hololink", required_argument, nullptr, 0 },
        { "ibv-name", required_argument, nullptr, 0 },
        { "ibv-port", required_argument, nullptr, 0 },
        { "resetAdcam", required_argument, nullptr, 0 },
        { "resetPin", required_argument, nullptr, 0 },
        { "firmwareUpdate", required_argument, nullptr, 0},
        { "capture", required_argument, nullptr, 0 },
        { "numPlanes", required_argument, nullptr, 0 },
        { "captureFps", required_argument, nullptr, 0 },
        { "metadata", required_argument, nullptr, 0 },
        { "maxMipi", required_argument, nullptr, 0 },
        { "log-level", required_argument, nullptr, 0 },
        { 0, 0, nullptr, 0 }
    };

    while (true) {
        int option_index = 0;
        int c = getopt_long(argc, argv, "h", long_options, &option_index);

        if (c == -1)
            break;  // no more options

        const std::string argument(optarg ? optarg : "");

        if (c == 0) {
            // Long option
            const struct option* opt = &long_options[option_index];

            if (opt->name == std::string("captureMode")) {
                adcam_mode = std::stoi(argument);

                if (adcam_mode < 0 || adcam_mode > 9) {
                    throw std::runtime_error(fmt::format(
                        "Unhandled captureMode \"{}\"", adcam_mode));
                }

            } else if (opt->name == std::string("resetPin")) {
                reset_pin = std::stoi(argument);

                if (reset_pin <0 || reset_pin > 31)
                {
                    throw std::runtime_error(fmt::format("Unhandled resetPin (0-31) \"{}\"", reset_pin));
                }
            } else if (opt->name == std::string("numPlanes")) {
                num_planes = std::stoi(argument);

                if (num_planes <2 || num_planes > 3)
                {
                    throw std::runtime_error(fmt::format("Unhandled numPlanes (2 (Depth + AB) or 3 (Depth + AB + Conf)) \"{}\"", num_planes));
                }
            } else if (opt->name == std::string("captureFps")) {
                tof_fps = std::stoi(argument);

                if (tof_fps <1 || tof_fps > 30)
                {
                    throw std::runtime_error(fmt::format("Unhandled captureFps (1-30) \"{}\"", tof_fps));
                }
            } else if (opt->name == std::string("metadata")) {
                metadata_sz = std::stoi(argument);

                if (metadata_sz <0 || metadata_sz > 256)
                {
                    throw std::runtime_error(fmt::format("Unhandled metadata size (1-256) \"{}\"", metadata_sz));
                }
            } else if (opt->name == std::string("maxMipi")) {
                mipi_lane_speed = (uint16_t) std::stoi(argument);

                if (mipi_lane_speed == 1000)
		{
			mipi_lane_speed = MIPI_SPEED_1GBPS;
		}
		else if (mipi_lane_speed == 1500)
		{
			mipi_lane_speed = MIPI_SPEED_1_5_GBPS;
		}
		else if (mipi_lane_speed == 2000)
		{
			mipi_lane_speed = MIPI_SPEED_2_0_GBPS;
		}
		else if (mipi_lane_speed == 2500)
		{
			mipi_lane_speed = MIPI_SPEED_2_5_GBPS;
		}
		else
                {
                    throw std::runtime_error(fmt::format("Unhandled maxMipi (1000/1500/2000/2500) in Mbps/lane\"{}\"", mipi_lane_speed));
                }
            } else if (opt->name == std::string("headless")) {
                headless = true;

            } else if (opt->name == std::string("fullscreen")) {
                fullscreen = true;

            } else if (opt->name == std::string("frame-limit")) {
                frame_limit = std::stoll(argument);

            } else if (opt->name == std::string("configuration")) {
                configuration = argument;

            } else if (opt->name == std::string("hololink")) {
                hololink_ip = argument;

            } else if (opt->name == std::string("ibv-name")) {
                ibv_name = argument;

            } else if (opt->name == std::string("ibv-port")) {
                ibv_port = std::stoul(argument);

            } else if (opt->name == std::string("resetAdcam")) {
                do_reset = std::stoi(argument);

            } else if (opt->name == std::string("firmwareUpdate")) {
                firmware_manifest = argument;

            } else if (opt->name == std::string("capture")) {
                do_capture = std::stoi(argument);

            } else if (opt->name == std::string("log-level")) {
                // Normalize log level
                std::string lvl = argument;
                std::transform(lvl.begin(), lvl.end(), lvl.begin(), ::tolower);

                if (lvl == "trace") log_level = holoscan::LogLevel::TRACE;
                else if (lvl == "debug") log_level = holoscan::LogLevel::DEBUG;
                else if (lvl == "info") log_level = holoscan::LogLevel::INFO;
                else if (lvl == "warn") log_level = holoscan::LogLevel::WARN;
                else if (lvl == "error") log_level = holoscan::LogLevel::ERROR;
                else if (lvl == "critical") log_level = holoscan::LogLevel::CRITICAL;
                else if (lvl == "off") log_level = holoscan::LogLevel::OFF;
                else throw std::runtime_error(fmt::format("Unhandled log level \"{}\"", argument));

            } else {
                throw std::runtime_error(fmt::format("Unhandled option \"{}\"", opt->name));
            }

        } else if (c == 'h') {
            // Help text
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  -h, --help           Show this help message\n"
                      << "  --hololink <ip>      Hololink board IP (default " << hololink_ip << ")\n"
                      << "  --capture <0/1>      Capture and display Adcam data\n"
                      << "  --captureMode <0-6>  Adcam Capture code (0-6), default 6\n"
                      << "  --numPlanes <2/3>    Adcam Capture planes (Depth, AB, Conf), default 3\n"
                      << "  --captureFps <1-30>  Adcam Capture FPS (some FPS may not work), default 30\n"                                            
                      << "  --resetAdcam <0/1>   Reset ADCAM module\n"
                      << "  --resetPin <0-31>    Reset ADCAM pin, refer readme, default 0\n"
                      << "  --metadata <0-256>   Metadata to be removed from MIPI receive, refer readme, default 0\n"
                      << "  --maxMipi  <1000/1050/2000/2500>  Max supported Mipi per lane speed (default 2.5Gbps, 1G/1.5G/2G supported)\n"
                      << "  --firmwareUpdate <manifest.yaml>  Update ADCAM firmware using the given manifest file\n"                      
                      ;
            return EXIT_SUCCESS;

        } else {
            throw std::runtime_error("Unhandled option");
        }
    }

    //--------------------------------------------------------------------------
    // 4. Main application logic
    //--------------------------------------------------------------------------
    try {
        // Set Holoscan logging level
        holoscan::set_log_level(log_level);
        std::cout << "Initializing." << std::endl;

        //--------------------------------------------------------------------------
        // 4.1 Initialize CUDA device
        //--------------------------------------------------------------------------
        CudaCheck(cuInit(0));

        int cu_device_ordinal = 0;
        CUdevice cu_device;
        CudaCheck(cuDeviceGet(&cu_device, cu_device_ordinal));

        CUcontext cu_context;
        CudaCheck(cuDevicePrimaryCtxRetain(&cu_context, cu_device));

        //--------------------------------------------------------------------------
        // 4.2 Discover Hololink channel
        //--------------------------------------------------------------------------
        hololink::Metadata channel_metadata =
            hololink::Enumerator::find_channel(hololink_ip);

        //hololink::DataChannel hololink_channel(channel_metadata);
        auto hololink_channel = std::make_shared<hololink::DataChannel>(channel_metadata);

        uint32_t bus = hololink::CAM_I2C_BUS;

        //--------------------------------------------------------------------------
        // 4.3 Create ADCAM instance
        //--------------------------------------------------------------------------
        auto adcam_inst =
            std::make_shared<hololink::sensors::Adcam>(hololink_channel, bus, channel_metadata, adcam_mode, num_planes, tof_fps, reset_pin, metadata_sz, mipi_lane_speed);

        //--------------------------------------------------------------------------
        // 4.4 Start Hololink and initialize camera
        //--------------------------------------------------------------------------
        auto hololink = hololink_channel->hololink();
        hololink->start();

        if (0){
            std::cout << "Doing GPIO profiling" << std::endl;
             std::this_thread::sleep_for(std::chrono::seconds(5));
            adcam_inst->profile_fpga_perf(reset_pin);
            std::cout << "Doing GPIO profiling done" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(5));
            hololink->stop();
            return EXIT_SUCCESS;
        }
        
        if (do_reset > 0)
        {
            adcam_inst->adcam_reset_power_on();
            //adcam_inst->adcam_hard_reset();

            // Print master and slave firmware versions in a single burst session
            adcam_inst->switch_from_standard_to_burst();
            auto print_fw_version = [](const std::string &label,
                                       const std::vector<uint8_t> &resp) {
                if (resp.size() >= 4) {
                    std::cout << label << " Firmware version = " << (int)resp[0]
                              << "." << (int)resp[1] << "." << (int)resp[2]
                              << "." << (int)resp[3] << std::endl;
                } else {
                    std::cerr << label
                              << " Firmware version: incomplete response"
                              << std::endl;
                }
            };
            print_fw_version("Master", adcam_inst->get_fw_version_burst_mode(
                                           GET_MASTER_FIRMWARE_COMMAND));
            print_fw_version("Slave", adcam_inst->get_fw_version_burst_mode(
                                          GET_SLAVE_FIRMWARE_COMMAND));
            adcam_inst->switch_from_burst_to_standard();
        }

        if (!firmware_manifest.empty()) {
            hololink::Programmer::Args args;
            args.manifest = firmware_manifest;
            args.hololink_ip = hololink_ip;
            args.log_level = hololink::logging::hsb_log_level;

            std::cout << "Initializing firmware update." << std::endl;

            hololink::Programmer programmer(args, args.manifest);
            programmer.fetch_manifest("hololink");
            programmer.check_eula();

            std::cout << "EULA accepted.." << std::endl;
            programmer.check_images();
            auto ok =
                programmer.program_and_verify_images(hololink, adcam_inst);
            hololink->stop();
            CudaCheck(cuDevicePrimaryCtxRelease(cu_device));
            return EXIT_SUCCESS;
        }
                
        adcam_inst->get_ChipID();

        if (adcam_inst->probe_adcam_adtf3175()) {
            std::cout << "ADTF3175 Found" << std::endl;
        } else {
            std::cout << "ADTF3175 NOT Found, reset and try again" << std::endl;
            hololink->stop();
            return EXIT_FAILURE;
        }

        adcam_inst->get_status();
        adcam_inst->get_imager_type_and_ccb_version();

        //--------------------------------------------------------------------------
        // 4.5 Create and run Holoscan application
        //--------------------------------------------------------------------------
        if (do_capture > 0)
        {        
        auto application = holoscan::make_application<HoloscanApplication>(
            headless, fullscreen,
            cu_context, cu_device_ordinal,
            hololink_channel, ibv_name, ibv_port,
            adcam_inst, frame_limit);

        // NO need to do reset, if needed, add this line
        //hololink->reset();
            std::cout << "Calling run" << std::endl;
            application->run();
        }

        //adcam_inst->get_status();
        //hololink->stop();

        // Release CUDA primary context
        CudaCheck(cuDevicePrimaryCtxRelease(cu_device));

    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
