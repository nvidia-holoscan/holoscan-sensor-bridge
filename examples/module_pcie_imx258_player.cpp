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
 *   module_pcie_imx258_player                          (the single enumerated endpoint)
 *   module_pcie_imx258_player --hololink 0005:06:00.0  (that PCI function)
 *   module_pcie_imx258_player --skip-camera --frame-limit 4 --headless
 *   module_pcie_imx258_player --verify-registers --frame-limit 30 --headless
 *
 * --hololink takes the endpoint's PCI address (BDF) -- the PCIe counterpart of the Ethernet
 * --hololink <ip>: the channel is keyed on the FPGA function whose BAR0 the control plane
 * lands on, and the /dev/hololink_pcieN node is derived from it through sysfs. When omitted
 * and exactly one endpoint is enumerated, that one is used; several are listed and the run
 * fails, since no default BDF is right on every host.
 *
 * Requires a bound endpoint provisioned with the hsb_pcie module UUID, the extended kernel
 * driver, the module .so on the module search path (--module-dir / HOLOLINK_MODULE_DIR) and a
 * camera. No daemon. Not a ctest.
 */

#include <getopt.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

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
#include "hololink/module/pcie_enumerator.hpp"

#include "hololink/module/sensors/imx258/imx258_cam.hpp"
#include "hololink/module/sensors/imx258/imx258_mode.hpp"

namespace {

using hololink::module::sensors::imx258::Imx258_Mode;
using hololink::module::sensors::imx258::Imx258Cam;

/// Frame geometry, from the sensor's mode table; used directly only with --skip-camera.
constexpr uint32_t IMAGE_WIDTH = 1920;
constexpr uint32_t IMAGE_HEIGHT = 1080;
/// RAW10 lines of 1920 pixels are exactly 2400 bytes; 8-byte alignment is a no-op there.
constexpr uint32_t RAW10_LINE_ALIGNMENT = 8;

struct Options {
    std::string device; ///< --hololink: the endpoint's BDF; empty selects the single enumerated one
    std::string module_dir;
    Imx258_Mode camera_mode { Imx258_Mode::IMX258_MODE_1920X1080_60FPS_1194MBPS };
    uint32_t sensor { 0 };
    uint32_t num_buffers { 4 };
    uint32_t chunk_pages { 1 };
    int32_t optical_black { 50 };
    int64_t frame_limit { 0 };
    bool headless { false };
    bool fullscreen { false };
    bool skip_camera { false };
    bool verify_registers { false };
    bool camera_start_early { false };
    bool reset { false }; // opt-in: SYS_RST kills the sensor receive path until the design is reloaded
    std::chrono::seconds discovery_timeout { 0 };
    holoscan::LogLevel log_level { holoscan::LogLevel::INFO };
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

        // The camera drives the receiver's device lifecycle: PcieReceiverOp::start() programs
        // and arms the ring, and only then is device_start called, so no frame is produced
        // before there is somewhere to put it. Both are no-ops with --skip-camera.
        auto device_start = std::function<void()>([this] {
            if (camera_) {
                if (!options_.camera_start_early) {
                    camera_->start();
                }
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
        // --skip-camera arms the very same ring a camera run would.
        const uint32_t width = camera_ ? camera_->width() : IMAGE_WIDTH;
        const uint32_t height = camera_ ? camera_->height() : IMAGE_HEIGHT;
        const uint32_t raw_start_byte = camera_
            ? camera_->raw_start_byte()
            : hololink::module::sensors::imx258::RAW10_START_BYTE;
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

        // 640 + 2400 * 1080 = 2592640: the converter's CSI length and the receiver's DMA frame
        // size are the same arithmetic, and PcieReceiverOp checks they agree.
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
            holoscan::Arg("num_buffers", options_.num_buffers),
            holoscan::Arg("chunk_pages", options_.chunk_pages),
            holoscan::Arg("width", width),
            holoscan::Arg("height", height),
            holoscan::Arg("stride_pixels", width),
            holoscan::Arg("line_alignment", RAW10_LINE_ALIGNMENT),
            holoscan::Arg("raw_start_byte", raw_start_byte));

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
    std::cout << "Usage: " << program << " [options]\n"
              << "  IMX258 over the PCIe endpoint on the module path, receive only.\n\n"
              << "  --hololink DDDD:BB:DD.F     PCI address of the endpoint (the PCIe counterpart of the\n"
              << "                              Ethernet --hololink <ip>); default: the single enumerated\n"
              << "                              endpoint, or an error listing them when there are several\n"
              << "  --module-dir <path>         Directory containing hololink_<UUID>.so\n"
              << "  --camera-mode <int>         IMX258 mode: 0=1080p60 at 1194.75 Mb/s/lane (default 0)\n"
              << "  --sensor <int>              Sensor index, also the MSI-X vector (default 0)\n"
              << "  --num-buffers <int>         Ring slots (default 4)\n"
              << "  --chunk-pages <int>         Data TLP size in 128-byte pages (default 1)\n"
              << "  --optical-black <int>       ImageProcessorOp optical black level (default 50)\n"
              << "  --frame-limit <int>         Exit after this many frames\n"
              << "  --headless                  Run holoviz without a display\n"
              << "  --fullscreen                Run holoviz fullscreen\n"
              << "  --skip-camera               Arm the ring without touching the sensor\n"
              << "  --verify-registers          Read every sensor register back after writing it\n"
              << "  --reset                     Call HololinkInterfaceV1::reset() after start(). This pulses\n"
              << "                              SYS_RST, and on the current bitfiles the sensor receive path\n"
              << "                              does not come back until the FPGA is reconfigured -- every\n"
              << "                              later run, this player or the reference one, then captures\n"
              << "                              nothing. Off by default; the design is left as found.\n"
              << "  --camera-start-early        Start the sensor streaming right after configuring it,\n"
              << "                              before the ring is armed (the reference player's order);\n"
              << "                              default starts it once the receiver is armed\n"
              << "  --discovery-timeout <s>     Re-scan sysfs for up to <s> seconds when the endpoint is\n"
              << "                              not listed yet (default 0: one scan, fail immediately)\n"
              << "  --log-level <level>         Holoscan log level\n\n"
              << "  --skip-camera isolates the DMA path from sensor bring-up: the ring is\n"
              << "  programmed and armed but no I2C is issued, so nothing streams unless the\n"
              << "  sensor was already running.\n";
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
        { "skip-camera", no_argument, nullptr, 0 },
        { "verify-registers", no_argument, nullptr, 0 },
        { "camera-start-early", no_argument, nullptr, 0 },
        { "reset", no_argument, nullptr, 0 },
        { "discovery-timeout", required_argument, nullptr, 0 },
        { "log-level", required_argument, nullptr, 0 },
        { nullptr, 0, nullptr, 0 },
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
            if (name == "hololink") {
                options.device = argument;
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
            } else if (name == "skip-camera") {
                options.skip_camera = true;
            } else if (name == "verify-registers") {
                options.verify_registers = true;
            } else if (name == "camera-start-early") {
                options.camera_start_early = true;
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
    return true;
}

/**
 * @brief The BDF to drive: --hololink normalized (a /dev/hololink_pcieN node or a
 * /dev/hololink/by-bdf alias is accepted and resolved to its BDF), or -- when omitted -- the
 * single endpoint sysfs lists. Several endpoints are listed and the run fails: there is no
 * default BDF that is right across hosts. `rescan_window` re-runs the scan for an endpoint
 * that is still appearing.
 */
std::string select_endpoint(const std::string& requested, std::chrono::milliseconds rescan_window)
{
    namespace pcie = hololink::module::pcie;
    if (!requested.empty()) {
        const std::optional<std::string> bdf = pcie::bdf_for_peer(requested);
        if (!bdf) {
            throw std::runtime_error(fmt::format(
                "--hololink '{}' is not a PCI address DDDD:BB:DD.F (nor a /dev/hololink_pcieN node "
                "or /dev/hololink/by-bdf/<bdf> alias of a bound endpoint)",
                requested));
        }
        return *bdf;
    }
    const auto deadline = std::chrono::steady_clock::now() + rescan_window;
    std::vector<pcie::PcieDeviceInfo> devices;
    while (true) {
        devices = pcie::scan_pcie_devices();
        if (!devices.empty() || std::chrono::steady_clock::now() >= deadline) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    if (devices.empty()) {
        throw std::runtime_error(fmt::format(
            "no hololink_pcie endpoint found under {}; bind the endpoint to the hololink_pcie "
            "driver, or name one with --hololink DDDD:BB:DD.F",
            pcie::DEFAULT_SYSFS_CLASS));
    }
    if (devices.size() > 1) {
        std::string listing;
        for (const auto& d : devices) {
            listing += fmt::format("\n  {}  ({}, uuid {})", d.bdf, d.device_node, d.uuid_text);
        }
        throw std::runtime_error(fmt::format(
            "{} hololink_pcie endpoints found; pick one with --hololink DDDD:BB:DD.F:{}",
            devices.size(), listing));
    }
    return devices[0].bdf;
}

/**
 * @brief Wait for the kernel driver's bootp announcement of `bdf` and return its metadata.
 *
 * Discovery is bootp-only, exactly as for Ethernet boards: the hololink_pcie kernel driver
 * publishes one announcement per bound endpoint on the loopback interface every second, the
 * Adapter's listener parses it, the hsb_pcie module (selected by the board UUID) restores
 * peer_ip from the bootp client address to the BDF, and Adapter::wait_for_channel matches on
 * that. Nothing to scan or replay; the first announcement arrives within one cadence.
 * `timeout` (--discovery-timeout) bounds the wait, with a floor of three cadences.
 */
hololink::module::EnumerationMetadata discover_device(hololink::module::Adapter& adapter,
    const std::string& bdf, std::chrono::milliseconds timeout)
{
    constexpr auto MIN_WAIT = std::chrono::milliseconds(3000);
    const auto wait = std::max(timeout, MIN_WAIT);
    return adapter.wait_for_channel(bdf, wait);
}

} // namespace

int main(int argc, char** argv)
{
    Options options;
    try {
        if (!parse_options(argc, argv, options)) {
            return EXIT_SUCCESS;
        }
    } catch (const std::exception& e) {
        // A non-numeric --sensor, an out-of-range --camera-mode, an unknown --log-level:
        // report it like every other failure below instead of leaving main() through an
        // uncaught exception.
        std::cerr << "module_pcie_imx258_player: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    holoscan::set_log_level(options.log_level);
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

        // Discovery: sysfs scan through the Adapter, keyed on the endpoint's PCI function.
        const std::string bdf = select_endpoint(options.device, options.discovery_timeout);
        hololink::module::EnumerationMetadata metadata
            = discover_device(adapter, bdf, options.discovery_timeout);
        // Stamp the per-sensor fields (sensor, vp_address, pcie_vector) before the first
        // get_service, which caches the per-board impl configured from this metadata.
        adapter.use_sensor(metadata, options.sensor);
        std::cout << "Enumerated " << bdf
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

        std::shared_ptr<Imx258Cam> camera;
        if (!options.skip_camera) {
            camera = std::make_shared<Imx258Cam>(metadata);
            camera->configure(options.camera_mode, options.verify_registers);
            if (options.camera_start_early) {
                // Reference-player order: the register table ends with MODE_SELECT=1, so
                // the sensor is already in HS before the ring is armed (frames until then
                // are simply not captured: VIP_MASK is 0).
                camera->start();
                log_streaming_status(*camera, "after early start");
            }
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
