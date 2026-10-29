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

/*
 * IMX258 camera over the PCIe endpoint on the MODULE path, receive only.
 *
 * Mirrors examples/module_imx274_player.cpp in shape, on the module V1 path:
 *
 *     hololink::module::Adapter (singleton; bootp listener fed by the kernel driver) + wait_for_channel(bdf)
 *     hololink::module::HololinkInterfaceV1        (served by the hsb_pcie module, in-process)
 *     hololink::module::sensors::imx258::Imx258Cam  (pure I2C through I2cInterfaceV1)
 *     hololink::module::operators::PcieReceiverOp  (module-native ring; no DataChannel)
 *
 * The camera streams RAW10 into the FPGA, the endpoint DMAs whole frames into GPU VRAM, and
 * the usual chain turns them into pictures:
 *
 *   PcieReceiverOp -> CsiToBayerOp -> ImageProcessorOp -> BayerDemosaicOp -> HolovizOp
 *
 * There is no transmit path here -- no HDMI output, no MemRd fetch.
 *
 *   module_pcie_imx258_player 0005:06:00.0
 *   module_pcie_imx258_player --hololink 0005:06:00.0 --skip-camera-initialization --frame-limit 4 --headless
 *   module_pcie_imx258_player --hololink 0005:06:00.0 --verify-registers --frame-limit 30 --headless
 *
 * The endpoint's PCI address (BDF) is required, as --hololink or as the one positional
 * argument -- the PCIe counterpart of the Ethernet --hololink <ip>: the hololink_pcie kernel
 * driver announces every bound endpoint over bootp on the loopback interface, keyed on the
 * FPGA function whose BAR0 the control plane lands on, and Adapter::wait_for_channel matches
 * on that exactly as it does for an IP address. There is no default: no BDF is right on
 * every host; the bdf attribute of each entry under /sys/class/hololink_pcie lists the bound
 * ones.
 *
 * Requires a bound endpoint provisioned with the hsb_pcie module UUID, the extended kernel
 * driver, the module .so on the module search path (--module-dir / HOLOLINK_MODULE_DIR) and a
 * camera. No daemon. Not a ctest.
 */

#include <getopt.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <cuda.h>

#include <fmt/format.h>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/bayer_demosaic/bayer_demosaic.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include "hololink/module/adapter.hpp"
#include "hololink/module/csi_converter.hpp"
#include "hololink/module/cuda_unique.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/hololink.hpp"
#include "hololink/module/logging.hpp"
#include "hololink/module/operators/csi_to_bayer_op.hpp"
#include "hololink/module/operators/image_processor_op.hpp"
#include "hololink/module/operators/pcie_receiver_op.hpp"

#include "hololink/module/operators/log_level.hpp"

#include "hololink/module/sensors/imx258/imx258_cam.hpp"
#include "hololink/module/sensors/imx258/imx258_mode.hpp"

namespace {

using hololink::module::sensors::imx258::Imx258_Mode;
using hololink::module::sensors::imx258::Imx258Cam;

/// Frame geometry, from the sensor's mode table; used directly only with
/// --skip-camera-initialization.
constexpr uint32_t IMAGE_WIDTH = 1920;
constexpr uint32_t IMAGE_HEIGHT = 1080;

struct Options {
    std::string device; ///< the endpoint's BDF: --hololink or the positional argument; required
    std::string module_dir;
    Imx258_Mode camera_mode { Imx258_Mode::IMX258_MODE_1920X1080_60FPS_1194MBPS };
    uint32_t sensor { 0 };
    uint32_t num_buffers { 4 };
    uint32_t chunk_pages { 1 };
    int32_t optical_black { 50 };
    int64_t frame_limit { 0 };
    bool headless { false };
    bool fullscreen { false };
    bool skip_camera_initialization { false };
    bool verify_registers { false };
    bool reset { false }; // opt-in: SYS_RST kills the sensor receive path until the design is reloaded
    std::chrono::seconds discovery_timeout { 30 };
    holoscan::LogLevel log_level { holoscan::LogLevel::INFO }; // main() applies HOLOLINK_LOG_LEVEL first
};

/* Ask the sensor itself whether it is streaming: REG_MODE_SELECT as latched and the SMIA
 * frame counter sampled twice. This separates "the camera is not emitting frames" from "the
 * FPGA is not receiving them" (PACK_SIF_TVLD_STAT), which a stalled receiver alone cannot. */
static void log_streaming_status(hololink::module::sensors::imx258::Imx258Cam& camera, const char* when)
{
    try {
        const auto s = camera.streaming_status();
        if (s.mode_select == 0x01 && s.advancing) {
            HSB_LOG_INFO("IMX258 streaming check {}: MODE_SELECT=0x01, FRM_CNT {:#04x} -> {:#04x}: "
                         "the sensor is emitting frames",
                when, s.frame_count_before, s.frame_count_after);
        } else {
            HSB_LOG_WARN("IMX258 streaming check {}: MODE_SELECT={:#04x}, FRM_CNT {:#04x} -> {:#04x}: "
                         "the sensor is NOT emitting frames ({})",
                when, s.mode_select, s.frame_count_before, s.frame_count_after,
                s.mode_select != 0x01 ? "streaming is not enabled in the sensor"
                                      : "streaming enabled but the frame counter does not advance");
        }
    } catch (const std::exception& e) {
        HSB_LOG_WARN("IMX258 streaming check {}: I2C read failed: {}", when, e.what());
    }
}

class HoloscanApplication : public holoscan::Application {
public:
    HoloscanApplication(const Options& options, CUcontext cuda_context, int cuda_device_ordinal,
        hololink::module::EnumerationMetadata metadata, std::shared_ptr<Imx258Cam> camera)
        : options_(options)
        , cuda_context_(cuda_context)
        , cuda_device_ordinal_(cuda_device_ordinal)
        , metadata_(std::move(metadata))
        , camera_(std::move(camera))
    {
    }
    HoloscanApplication() = delete;

    void compose() override
    {
        std::shared_ptr<holoscan::Condition> condition;
        if (options_.frame_limit) {
            condition = make_condition<holoscan::CountCondition>("count", options_.frame_limit);
        } else {
            condition = make_condition<holoscan::BooleanCondition>("ok", true);
        }

        // The receiver drives the sensor's lifecycle, as on RoCE: PcieReceiverOp::start()
        // programs and arms the ring and only then calls device_start, so no frame is
        // produced before there is somewhere to put it. Both are no-ops with
        // --skip-camera-initialization.
        auto device_start = std::function<void()>([this] {
            if (camera_) {
                camera_->start();
                log_streaming_status(*camera_, "after device_start");
            }
        });
        auto device_stop = std::function<void()>([this] {
            if (camera_) {
                camera_->stop();
            }
        });

        auto visualizer = make_operator<holoscan::ops::HolovizOp>("holoviz",
            holoscan::Arg("fullscreen", options_.fullscreen),
            holoscan::Arg("headless", options_.headless),
            holoscan::Arg("framebuffer_srgb", true));

        // Geometry from the camera when there is one, else the mode table's constants so
        // --skip-camera-initialization arms the very same ring a camera run would.
        const uint32_t width = camera_ ? camera_->width() : IMAGE_WIDTH;
        const uint32_t height = camera_ ? camera_->height() : IMAGE_HEIGHT;
        const auto bayer_format
            = camera_ ? camera_->bayer_format() : hololink::module::csi::BayerFormat::RGGB;
        const auto pixel_format
            = camera_ ? camera_->pixel_format() : hololink::module::csi::PixelFormat::RAW_10;

        auto csi_to_bayer_pool = make_resource<holoscan::BlockMemoryPool>("csi_to_bayer_pool",
            /*storage_type=*/1,
            /*block_size=*/width * sizeof(uint16_t) * height,
            /*num_blocks=*/4);
        auto csi_to_bayer_operator = make_operator<hololink::module::operators::CsiToBayerOp>(
            "csi_to_bayer",
            holoscan::Arg("allocator", csi_to_bayer_pool),
            holoscan::Arg("cuda_device_ordinal", cuda_device_ordinal_));
        if (camera_) {
            camera_->configure_converter(csi_to_bayer_operator);
        } else {
            Imx258Cam::configure_converter(*csi_to_bayer_operator, width, height);
        }

        // 640 + 2400 * 1080 = 2592640: the CSI converter owns the frame layout and the
        // receiver only needs its size, exactly as with RoceReceiverOp.
        const size_t frame_size = csi_to_bayer_operator->get_csi_length();
        auto receiver_operator = make_operator<hololink::module::operators::PcieReceiverOp>(
            "receiver",
            condition,
            holoscan::Arg("enumeration_metadata", metadata_),
            holoscan::Arg("frame_context", cuda_context_),
            holoscan::Arg("frame_size", frame_size),
            holoscan::Arg("device_start", device_start),
            holoscan::Arg("device_stop", device_stop),
            holoscan::Arg("sensor", static_cast<int32_t>(options_.sensor)),
            holoscan::Arg("pages", options_.num_buffers),
            holoscan::Arg("chunk_pages", options_.chunk_pages));

        auto image_processor_operator
            = make_operator<hololink::module::operators::ImageProcessorOp>("image_processor",
                holoscan::Arg("optical_black", options_.optical_black),
                holoscan::Arg("bayer_format", static_cast<int>(bayer_format)),
                holoscan::Arg("pixel_format", static_cast<int>(pixel_format)));

        constexpr uint32_t rgba_components_per_pixel = 4;
        auto bayer_pool = make_resource<holoscan::BlockMemoryPool>("bayer_pool",
            /*storage_type=*/1,
            /*block_size=*/width * rgba_components_per_pixel * sizeof(uint16_t) * height,
            /*num_blocks=*/4);
        auto demosaic = make_operator<holoscan::ops::BayerDemosaicOp>("demosaic",
            holoscan::Arg("pool", bayer_pool),
            holoscan::Arg("generate_alpha", true),
            holoscan::Arg("alpha_value", 65535),
            holoscan::Arg("bayer_grid_pos", static_cast<int>(bayer_format)),
            holoscan::Arg("interpolation_mode", 0));

        add_flow(receiver_operator, csi_to_bayer_operator, { { "output", "input" } });
        add_flow(csi_to_bayer_operator, image_processor_operator, { { "output", "input" } });
        add_flow(image_processor_operator, demosaic, { { "output", "receiver" } });
        add_flow(demosaic, visualizer, { { "transmitter", "receivers" } });
    }

private:
    const Options options_;
    const CUcontext cuda_context_;
    const int cuda_device_ordinal_;
    hololink::module::EnumerationMetadata metadata_;
    const std::shared_ptr<Imx258Cam> camera_;
};

void print_usage(const char* program)
{
    std::cout << "Usage: " << program << " [options] (--hololink DDDD:BB:DD.F | DDDD:BB:DD.F)\n"
              << "  IMX258 over the PCIe endpoint on the module path, receive only.\n\n"
              << "  --hololink DDDD:BB:DD.F     PCI address of the endpoint (the PCIe counterpart of the\n"
              << "                              Ethernet --hololink <ip>). Required; may be given as the\n"
              << "                              positional argument instead. `cat /sys/class/hololink_pcie/*/bdf`\n"
              << "                              lists the bound endpoints\n"
              << "  --module-dir <path>         Directory containing hololink_<UUID>.so\n"
              << "  --camera-mode <int>         IMX258 mode: 0=1080p60 at 1194.75 Mb/s/lane (default 0)\n"
              << "  --sensor <int>              Sensor index, also the MSI-X vector (default 0)\n"
              << "  --num-buffers <int>         Ring pages the endpoint cycles through (default 4)\n"
              << "  --chunk-pages <int>         Data TLP size in 128-byte pages (default 1)\n"
              << "  --optical-black <int>       ImageProcessorOp optical black level (default 50)\n"
              << "  --frame-limit <int>         Exit after this many frames\n"
              << "  --headless                  Run holoviz without a display\n"
              << "  --fullscreen                Run holoviz fullscreen\n"
              << "  --skip-camera-initialization\n"
              << "                              Arm the ring without touching the sensor: no I2C is\n"
              << "                              issued, so nothing streams unless it already was\n"
              << "  --verify-registers          Read every sensor register back after writing it\n"
              << "  --reset                     Call HololinkInterfaceV1::reset() after start(). This pulses\n"
              << "                              SYS_RST, and on the current bitfiles the sensor receive path\n"
              << "                              does not come back until the FPGA is reconfigured -- every\n"
              << "                              later run, this player or the reference one, then captures\n"
              << "                              nothing. Off by default; the design is left as found.\n"
              << "  --discovery-timeout <s>     Seconds to wait for the bootp announcement (default 30)\n"
              << "  --log-level <level>         Holoscan log level (default $HOLOLINK_LOG_LEVEL, else INFO)\n";
}

holoscan::LogLevel to_log_level(const std::string& name)
{
    if (name == "trace" || name == "TRACE") {
        return holoscan::LogLevel::TRACE;
    } else if (name == "debug" || name == "DEBUG") {
        return holoscan::LogLevel::DEBUG;
    } else if (name == "info" || name == "INFO") {
        return holoscan::LogLevel::INFO;
    } else if (name == "warn" || name == "WARN") {
        return holoscan::LogLevel::WARN;
    } else if (name == "error" || name == "ERROR") {
        return holoscan::LogLevel::ERROR;
    } else if (name == "critical" || name == "CRITICAL") {
        return holoscan::LogLevel::CRITICAL;
    } else if (name == "off" || name == "OFF") {
        return holoscan::LogLevel::OFF;
    }
    throw std::runtime_error("Unhandled log level \"" + name + "\"");
}

bool parse_options(int argc, char** argv, Options& options)
{
    static const struct option long_options[] = {
        { "help", no_argument, nullptr, 'h' },
        { "hololink", required_argument, nullptr, 0 },
        { "module-dir", required_argument, nullptr, 0 },
        { "camera-mode", required_argument, nullptr, 0 },
        { "sensor", required_argument, nullptr, 0 },
        { "num-buffers", required_argument, nullptr, 0 },
        { "chunk-pages", required_argument, nullptr, 0 },
        { "optical-black", required_argument, nullptr, 0 },
        { "frame-limit", required_argument, nullptr, 0 },
        { "headless", no_argument, nullptr, 0 },
        { "fullscreen", no_argument, nullptr, 0 },
        { "skip-camera-initialization", no_argument, nullptr, 0 },
        { "verify-registers", no_argument, nullptr, 0 },
        { "reset", no_argument, nullptr, 0 },
        { "discovery-timeout", required_argument, nullptr, 0 },
        { "log-level", required_argument, nullptr, 0 },
        { nullptr, 0, nullptr, 0 },
    };

    std::string from_option;
    while (true) {
        int option_index = 0;
        const int c = getopt_long(argc, argv, "h", long_options, &option_index);
        if (c == -1) {
            break;
        }
        const std::string argument(optarg ? optarg : "");
        if (c == 0) {
            const std::string name = long_options[option_index].name;
            if (name == "hololink") {
                from_option = argument;
            } else if (name == "module-dir") {
                options.module_dir = argument;
            } else if (name == "camera-mode") {
                const int mode = std::stoi(argument);
                if (mode < 0 || mode >= static_cast<int>(Imx258_Mode::Unknown)) {
                    throw std::runtime_error(fmt::format("--camera-mode must be in [0, {}]",
                        static_cast<int>(Imx258_Mode::Unknown) - 1));
                }
                options.camera_mode = static_cast<Imx258_Mode>(mode);
            } else if (name == "sensor") {
                options.sensor = std::stoul(argument);
            } else if (name == "num-buffers") {
                options.num_buffers = std::stoul(argument);
            } else if (name == "chunk-pages") {
                options.chunk_pages = std::stoul(argument);
            } else if (name == "optical-black") {
                options.optical_black = std::stoi(argument);
            } else if (name == "frame-limit") {
                options.frame_limit = std::stoll(argument);
            } else if (name == "headless") {
                options.headless = true;
            } else if (name == "fullscreen") {
                options.fullscreen = true;
            } else if (name == "skip-camera-initialization") {
                options.skip_camera_initialization = true;
            } else if (name == "verify-registers") {
                options.verify_registers = true;
            } else if (name == "reset") {
                options.reset = true;
            } else if (name == "discovery-timeout") {
                options.discovery_timeout = std::chrono::seconds(std::stoll(argument));
            } else if (name == "log-level") {
                options.log_level = to_log_level(argument);
            } else {
                throw std::runtime_error("Unhandled option \"" + name + "\"");
            }
        } else if (c == 'h') {
            print_usage(argv[0]);
            return false;
        } else {
            throw std::runtime_error("Unhandled option");
        }
    }

    // The BDF: --hololink, or the single positional argument, but not both and not neither.
    std::string positional;
    if (optind < argc) {
        positional = argv[optind++];
    }
    if (optind < argc) {
        throw std::runtime_error(fmt::format("unexpected argument '{}'", argv[optind]));
    }
    if (!from_option.empty() && !positional.empty()) {
        throw std::runtime_error(fmt::format(
            "the endpoint was given twice: --hololink {} and '{}'", from_option, positional));
    }
    options.device = !from_option.empty() ? from_option : positional;
    if (options.device.empty()) {
        throw std::runtime_error("the endpoint's PCI address is required: --hololink DDDD:BB:DD.F "
                                 "or DDDD:BB:DD.F as the positional argument (-h for usage)");
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    Options options;
    try {
        options.log_level = hololink::module::operators::default_log_level();
        if (!parse_options(argc, argv, options)) {
            return EXIT_SUCCESS;
        }
    } catch (const std::exception& e) {
        // A missing BDF, a non-numeric --sensor, an out-of-range --camera-mode, an unknown
        // --log-level: report it like every other failure below instead of leaving main()
        // through an uncaught exception.
        std::cerr << "module_pcie_imx258_player: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    hololink::module::operators::set_log_level(options.log_level);
    std::cout << "Initializing." << std::endl;

    HOLOLINK_MODULE_CUDA_CHECK(cuInit(0));
    const int cu_device_ordinal = 0;
    CUdevice cu_device;
    HOLOLINK_MODULE_CUDA_CHECK(cuDeviceGet(&cu_device, cu_device_ordinal));
    CUcontext cu_context;
    HOLOLINK_MODULE_CUDA_CHECK(cuDevicePrimaryCtxRetain(&cu_context, cu_device));

    std::shared_ptr<hololink::module::HololinkInterfaceV1> hololink;
    try {
        auto& adapter = hololink::module::Adapter::get_adapter();
        if (!options.module_dir.empty()) {
            adapter.set_module_directory(std::filesystem::path(options.module_dir));
        }

        // Discovery is bootp, exactly as for an Ethernet board: the kernel driver announces
        // the endpoint once a second and the hsb_pcie module keys the record on the BDF.
        hololink::module::EnumerationMetadata metadata
            = adapter.wait_for_channel(options.device, options.discovery_timeout);
        // Stamp the per-sensor fields (sensor, vp_address, pcie_vector) before the first
        // get_service, which caches the per-board impl configured from this metadata.
        adapter.use_sensor(metadata, options.sensor);
        std::cout << "Enumerated " << options.device
                  << " node=" << metadata.get<std::string>("pcie_device_node", "?")
                  << " uuid=" << metadata.get<std::string>("fpga_uuid", "?")
                  << " module=" << metadata.get<std::string>("module_name", "?") << std::endl;

        // Connect to the endpoint. No reset() unless asked: reset() pulses SYS_RST, and on the
        // current bitfiles the sensor receive path does not come back from that until the FPGA
        // is reconfigured -- observed on an AGX Orin, where one resetting run left every later
        // run (this player and the reference one) capturing nothing while the sensor kept
        // emitting frames. The reference PCIe tooling never writes SYS_RST; neither does this
        // player by default. Apart from releasing SW_SEN_RST (start() does that: on the CPNX
        // Versa it is the camera's hardware reset, and the sensor NAKs until it is released),
        // the design is left exactly as found.
        hololink = hololink::module::HololinkInterfaceV1::get_service(metadata);
        if (hololink->start() != HOLOLINK_MODULE_OK) {
            throw std::runtime_error("HololinkInterfaceV1::start failed");
        }
        if (options.reset) {
            HSB_LOG_WARN("--reset: pulsing SYS_RST; expect the sensor receive path to stay dead "
                         "until the FPGA is reconfigured");
            if (hololink->reset() != HOLOLINK_MODULE_OK) {
                throw std::runtime_error("HololinkInterfaceV1::reset failed");
            }
        }

        // Configure the sensor now; it is started from the receiver's device_start once the
        // ring is armed.
        std::shared_ptr<Imx258Cam> camera;
        if (!options.skip_camera_initialization) {
            camera = std::make_shared<Imx258Cam>(metadata);
            camera->configure(options.camera_mode, options.verify_registers);
        }

        auto application = holoscan::make_application<HoloscanApplication>(
            options, cu_context, cu_device_ordinal, metadata, camera);
        std::cout << "Calling run" << std::endl;
        application->run();
    } catch (const std::exception& e) {
        std::cerr << "module_pcie_imx258_player failed: " << e.what() << std::endl;
        if (hololink) {
            hololink->stop();
        }
        cuDevicePrimaryCtxRelease(cu_device);
        return EXIT_FAILURE;
    }

    // Stop streaming data and close the endpoint.
    hololink->stop();
    HOLOLINK_MODULE_CUDA_CHECK(cuDevicePrimaryCtxRelease(cu_device));
    return EXIT_SUCCESS;
}
