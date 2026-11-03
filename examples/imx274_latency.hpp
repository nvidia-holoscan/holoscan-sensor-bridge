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

// IMX274 latency measurement shared by linux_imx274_latency and roce_imx274_latency, the C++ ports
// of the HSDK4 Python tools linux_imx274_latency.py and imx274_latency.py. The pipeline is the
// player's,
//   receiver -> CsiToBayerOp -> ImageProcessorOp -> BayerDemosaicOp -> HolovizOp,
// and each frame is timed from the sensor's first byte until its demosaiced image, done on the
// GPU, is handed to HolovizOp. HSDK4 timed until HolovizOp had rendered the frame; HSDK5's
// HolovizOp reports nothing after rendering into a window, so the last two metrics end at the
// display hand-off and the report says so.

#ifndef HOLOLINK_EXAMPLES_IMX274_LATENCY_HPP
#define HOLOLINK_EXAMPLES_IMX274_LATENCY_HPP

#include <getopt.h>
#include <time.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <cuda.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/vision/bayer_demosaic_op.hpp>
#include <holoscan/visualization/holoviz_op.hpp>

#include <capture_metadata_traits.hpp>
#include <hololink/common/cuda_helper.hpp>
#include <hololink/common/tools.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/enumerator.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/core/logging_internal.hpp>
#include <hololink/core/on_scope_exit.hpp>
#include <hololink/operators/capture_metadata_utils.hpp>
#include <hololink/operators/csi_to_bayer/csi_to_bayer.hpp>
#include <hololink/operators/image_processor/image_processor.hpp>
#include <hololink/sensors/camera/imx274/native_imx274_sensor.hpp>

#include "player_support.hpp"

namespace hololink::examples::imx274_latency {

// ---- Options ------------------------------------------------------------------------------------

struct Options {
    int camera_mode = 0;
    bool headless = false;
    bool fullscreen = false;
    int64_t frame_limit = 200;
    std::string hololink_ip = hololink::env_hololink_ip(0, "192.168.0.2");
    int expander_configuration = 0;
    std::optional<int> pattern;
    bool skip_reset = false;
    int64_t log_interval = 0;
    // roce_imx274_latency only
    bool roce = false;
    std::string ibv_name; // empty: the receiver uses the device on the board's network interface
    uint32_t ibv_port = 1;
};

// Parses the HSDK4 Python tool's command line (argparse's: usage and an error on stderr and exit
// status 2 for a bad option, help on stdout for -h), and sets the log levels from --log-level,
// Python's numbers (10 DEBUG, 20 INFO, ...) or a level name.
inline Options parse_options(int argc, char** argv, bool roce)
{
    Options options;
    options.roce = roce;
    const std::string program = argv[0];
    const std::string usage = "usage: " + program
        + " [-h] [--camera-mode {0,1,2}] [--headless] [--fullscreen] [--frame-limit FRAME_LIMIT]"
          " [--configuration CONFIGURATION] [--hololink HOLOLINK] [--log-level LOG_LEVEL]"
          " [--expander-configuration {0,1}] [--pattern {0,1,2,3,4,5,6,7,8,9,10,11}] [--skip-reset]"
          " [--log-interval LOG_INTERVAL]"
        + (roce ? " [--ibv-name IBV_NAME] [--ibv-port IBV_PORT]" : "");
    const auto fail = [&](const std::string& message) {
        std::cerr << usage << "\n"
                  << program << ": error: " << message << std::endl;
        std::exit(2);
    };
    const auto integer = [&](const char* option, const std::string& value) -> int64_t {
        try {
            std::size_t used = 0;
            const int64_t result = std::stoll(value, &used);
            if (used == value.size()) {
                return result;
            }
        } catch (const std::exception&) {
        }
        fail(fmt::format("argument {}: invalid int value: '{}'", option, value));
        return 0;
    };

    holoscan::LogLevel log_level = holoscan::LogLevel::INFO;

    enum : int {
        kCameraMode = 1,
        kHeadless,
        kFullscreen,
        kFrameLimit,
        kConfiguration,
        kHololink,
        kLogLevel,
        kExpanderConfiguration,
        kPattern,
        kSkipReset,
        kLogInterval,
        kIbvName,
        kIbvPort,
    };
    std::vector<struct option> long_options = {
        { "help", no_argument, nullptr, 'h' },
        { "camera-mode", required_argument, nullptr, kCameraMode },
        { "headless", no_argument, nullptr, kHeadless },
        { "fullscreen", no_argument, nullptr, kFullscreen },
        { "frame-limit", required_argument, nullptr, kFrameLimit },
        { "configuration", required_argument, nullptr, kConfiguration },
        { "hololink", required_argument, nullptr, kHololink },
        { "log-level", required_argument, nullptr, kLogLevel },
        { "expander-configuration", required_argument, nullptr, kExpanderConfiguration },
        { "pattern", required_argument, nullptr, kPattern },
        { "skip-reset", no_argument, nullptr, kSkipReset },
        { "log-interval", required_argument, nullptr, kLogInterval },
    };
    if (roce) {
        long_options.push_back({ "ibv-name", required_argument, nullptr, kIbvName });
        long_options.push_back({ "ibv-port", required_argument, nullptr, kIbvPort });
    }
    long_options.push_back({ nullptr, 0, nullptr, 0 });

    opterr = 0;
    while (true) {
        const int c = getopt_long(argc, argv, ":h", long_options.data(), nullptr);
        if (c == -1) {
            break;
        }
        const std::string value(optarg ? optarg : "");
        switch (c) {
        case 'h':
            std::cout << usage << "\n\n"
                      << "options:\n"
                      << "  -h, --help            show this help message and exit\n"
                      << "  --camera-mode {0,1,2}\n"
                      << "                        0:IMX274_MODE_3840X2160_60FPS 1:IMX274_MODE_1920X1080_60FPS\n"
                      << "                        2:IMX274_MODE_3840X2160_30FPS_12BITS\n"
                      << "  --headless            Run in headless mode\n"
                      << "  --fullscreen          Run in fullscreen mode\n"
                      << "  --frame-limit FRAME_LIMIT\n"
                      << "                        Exit after receiving this many frames (0: until Ctrl+C\n"
                      << "                        or the window closes)\n"
                      << "  --configuration CONFIGURATION\n"
                      << "                        Configuration file (ignored by Holoscan SDK 5)\n"
                      << "  --hololink HOLOLINK   IP address of Hololink board\n"
                      << "  --log-level LOG_LEVEL\n"
                      << "                        Logging level to display\n"
                      << "  --expander-configuration {0,1}\n"
                      << "                        I2C Expander configuration\n"
                      << "  --pattern {0,1,2,3,4,5,6,7,8,9,10,11}\n"
                      << "                        Configure to display a test pattern.\n"
                      << "  --skip-reset          Don't call reset on the hololink device.\n"
                      << "  --log-interval LOG_INTERVAL\n"
                      << "                        Log progress every this many frames (0: never)\n";
            if (roce) {
                std::cout << "  --ibv-name IBV_NAME   IBV device to use (default: the device on the\n"
                          << "                        board's network interface)\n"
                          << "  --ibv-port IBV_PORT   Port number of IBV device (with --ibv-name)\n";
            }
            std::exit(EXIT_SUCCESS);
        case kCameraMode:
            options.camera_mode = static_cast<int>(integer("--camera-mode", value));
            if (options.camera_mode < 0 || options.camera_mode > 2) {
                fail(fmt::format("argument --camera-mode: invalid choice: {} (choose from 0, 1, 2)", value));
            }
            break;
        case kHeadless:
            options.headless = true;
            break;
        case kFullscreen:
            options.fullscreen = true;
            break;
        case kFrameLimit:
            options.frame_limit = integer("--frame-limit", value);
            if (options.frame_limit < 0) {
                fail(fmt::format("argument --frame-limit: must not be negative: {}", value));
            }
            break;
        case kConfiguration:
            if (!check_configuration(value)) {
                std::exit(EXIT_FAILURE);
            }
            break;
        case kHololink:
            options.hololink_ip = value;
            break;
        case kLogLevel:
            if (const auto level = parse_log_level(value)) {
                log_level = *level;
            } else {
                log_level = log_level_from_python(static_cast<int>(integer("--log-level", value)));
            }
            break;
        case kExpanderConfiguration:
            options.expander_configuration = static_cast<int>(integer("--expander-configuration", value));
            if (options.expander_configuration != 0 && options.expander_configuration != 1) {
                fail(fmt::format("argument --expander-configuration: invalid choice: {} (choose from 0, 1)", value));
            }
            break;
        case kPattern:
            options.pattern = static_cast<int>(integer("--pattern", value));
            if (*options.pattern < 0 || *options.pattern > 11) {
                fail(fmt::format("argument --pattern: invalid choice: {} (choose from 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "
                                 "10, 11)",
                    value));
            }
            break;
        case kSkipReset:
            options.skip_reset = true;
            break;
        case kLogInterval:
            options.log_interval = integer("--log-interval", value);
            if (options.log_interval < 0) {
                fail(fmt::format("argument --log-interval: must not be negative: {}", value));
            }
            break;
        case kIbvName:
            options.ibv_name = value;
            break;
        case kIbvPort:
            options.ibv_port = static_cast<uint32_t>(integer("--ibv-port", value));
            break;
        case ':':
            fail(fmt::format("argument {}: expected one argument", argv[optind - 1]));
            break;
        default:
            fail(fmt::format("unrecognized arguments: {}", argv[optind - 1]));
            break;
        }
    }
    if (optind < argc) {
        fail(fmt::format("unrecognized arguments: {}", argv[optind]));
    }
    holoscan::set_log_level(log_level);
    set_hsb_log_level(log_level);
    return options;
}

// ---- Recording ----------------------------------------------------------------------------------

// One frame's timestamps, in seconds since the epoch.
struct FrameTimes {
    int64_t frame_number = 0;
    // The HSB received the frame's first byte.
    double frame_start_s = 0;
    // The HSB sent the frame's metadata packet, right after its last byte.
    double frame_end_s = 0;
    // The receiver's thread woke up to the completed frame.
    double received_s = 0;
    // CsiToBayerOp, the operator after the receiver, started on the frame.
    double operator_s = 0;
    // The demosaiced image, done on the GPU, was handed to HolovizOp.
    double complete_s = 0;
};

class LatencyRecorder {
public:
    void record(const FrameTimes& times)
    {
        std::lock_guard lock(mutex_);
        frames_.push_back(times);
    }

    [[nodiscard]] std::vector<FrameTimes> frames() const
    {
        std::lock_guard lock(mutex_);
        return frames_;
    }

private:
    mutable std::mutex mutex_;
    std::vector<FrameTimes> frames_;
};

struct DisplayHandoffParams {
    std::shared_ptr<LatencyRecorder> recorder;
    int64_t frame_limit = 0; // 0: no limit
    int64_t log_interval = 0; // 0: no progress lines
};

// Takes each demosaiced image as it goes to HolovizOp, with its frame's metadata, and records the
// frame's timestamps once the GPU has finished the image; completes the run after frame_limit
// frames. It runs on BayerDemosaicOp's thread (see run()), so waiting for the GPU here holds back
// the demosaic, not the display.
// The players' receivers count --frame-limit instead, as HSDK4 did, and the run then completes by
// itself; SDK 5.0.0.5 doesn't complete a run that way through the demosaic's fan-out (5.0.0.7
// does), so here the hand-off counts.
class DisplayHandoffOp : public holoscan::Operator<DisplayHandoffParams> {
public:
    holoscan::Input<holoscan::schema::ImageT> image { "image" };
    holoscan::Input<hololink::CaptureMetadataT> metadata { "metadata" };

    void setup(holoscan::OperatorSpec& spec) override
    {
        (void)spec.input(image, "image").queue_depth(4U).tensor_field(holoscan::schema::ImageTensorFields::data);
        spec.input(metadata, "metadata").queue_depth(4U);
    }

    [[nodiscard]] holoscan::Contract contract() const override
    {
        holoscan::Contract c;
        c.trigger(holoscan::OnAligned {
                      .ports = holoscan::ports(image, metadata),
                      .window = hololink::operators::kMetadataPairingWindow,
                  })
            .input(image, holoscan::Exact {})
            .input(metadata, holoscan::Exact {});
        return c;
    }

    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override
    {
        auto maybe_paired = hololink::operators::receive_paired_tensor_metadata(image, metadata);
        if (!maybe_paired) {
            return holoscan::make_unexpected(std::move(maybe_paired).error());
        }
        if (!*maybe_paired) {
            count_mismatched_frame(mismatched_frames_, "DisplayHandoffOp");
            return {};
        }
        auto& paired = **maybe_paired;
        // The image arrives as soon as BayerDemosaicOp has queued its GPU work; the frame is
        // complete once that work is done.
        if (auto ready = paired.tensor.data.data->synchronize_for_host(); !ready) {
            return holoscan::make_unexpected(std::move(ready).error());
        }
        const double complete_s = now_s();
        const hololink::CaptureMetadataT& frame = paired.metadata.data;
        const auto& p = params();
        p.recorder->record(FrameTimes {
            .frame_number = frame.frame_number,
            .frame_start_s = seconds(frame.timestamp_s, frame.timestamp_ns),
            .frame_end_s = seconds(frame.metadata_s, frame.metadata_ns),
            .received_s = seconds(frame.received_s, frame.received_ns),
            .operator_s = static_cast<double>(frame.operator_entry_ns) * 1e-9,
            .complete_s = complete_s,
        });
        ++frames_;
        if (p.log_interval > 0 && frames_ % p.log_interval == 0) {
            HSB_LOG_INFO("{} frames, latest frame_number={}", frames_, frame.frame_number);
        }
        if (p.frame_limit > 0 && frames_ == p.frame_limit) {
            context.request_completion();
        }
        return {};
    }

private:
    static double seconds(int64_t s, int64_t ns) { return static_cast<double>(s) + static_cast<double>(ns) * 1e-9; }

    static double now_s()
    {
        struct timespec now;
        clock_gettime(CLOCK_REALTIME, &now);
        return seconds(now.tv_sec, now.tv_nsec);
    }

    int64_t frames_ = 0;
    hololink::operators::DropCounter mismatched_frames_;
};

// ---- Report -------------------------------------------------------------------------------------

namespace detail {

    // Python's repr of a float: the shortest representation that reads back the same, with ".0"
    // on whole numbers.
    inline std::string python_float(double value)
    {
        std::string text = fmt::format("{}", value);
        if (text.find_first_of(".en") == std::string::npos) {
            text += ".0";
        }
        return text;
    }

    // Python's round(value, 4).
    inline double round4(double value) { return std::nearbyint(value * 1e4) / 1e4; }

    // Python's datetime.fromtimestamp(seconds).isoformat(), in local time.
    inline std::string isoformat(double seconds)
    {
        const double whole = std::floor(seconds);
        const auto t = static_cast<time_t>(whole);
        const auto microseconds = static_cast<int>(std::lround((seconds - whole) * 1e6));
        struct tm local;
        localtime_r(&t, &local);
        char text[64];
        std::strftime(text, sizeof(text), "%Y-%m-%dT%H:%M:%S", &local);
        return microseconds ? fmt::format("{}.{:06}", text, microseconds) : std::string(text);
    }

    inline void report_metric(const char* title, const char* metric, const std::vector<double>& values)
    {
        double min = values[0];
        double max = values[0];
        double sum = 0;
        for (double value : values) {
            min = std::min(min, value);
            max = std::max(max, value);
            sum += value;
        }
        HSB_LOG_INFO("{}", title);
        HSB_LOG_INFO("{:<30}{:<15}{:<15}{:<15}", metric, python_float(min), python_float(max),
            python_float(sum / static_cast<double>(values.size())));
    }

} // namespace detail

// Logs the HSDK4 tool's report over the settled frames: all but the first and last five.
// Returns the exit status: 1 when fewer than 100 frames settled.
inline int report(const std::vector<FrameTimes>& frames)
{
    const std::vector<FrameTimes> settled = frames.size() > 10
        ? std::vector<FrameTimes>(frames.begin() + 5, frames.end() - 5)
        : std::vector<FrameTimes> {};
    if (settled.size() < 100) {
        HSB_LOG_ERROR("Only {} settled timestamps; need at least 100.", settled.size());
        return EXIT_FAILURE;
    }

    std::vector<double> frame_time, transfer, operator_latency, processing, overall;
    for (const FrameTimes& t : settled) {
        const double frame_time_dt = t.frame_end_s - t.frame_start_s;
        const double cpu_latency_dt = t.received_s - t.frame_end_s;
        const double operator_latency_dt = t.operator_s - t.received_s;
        const double processing_time_dt = t.complete_s - t.operator_s;
        const double overall_time_dt = t.complete_s - t.frame_start_s;
        frame_time.push_back(detail::round4(frame_time_dt));
        transfer.push_back(detail::round4(cpu_latency_dt));
        operator_latency.push_back(detail::round4(operator_latency_dt));
        processing.push_back(detail::round4(processing_time_dt));
        overall.push_back(detail::round4(overall_time_dt));
        HSB_LOG_DEBUG("** Frame Information  for Frame Number = {}**", t.frame_number);
        HSB_LOG_DEBUG("Frame Start          : {}", detail::isoformat(t.frame_start_s));
        HSB_LOG_DEBUG("Frame End            : {}", detail::isoformat(t.frame_end_s));
        HSB_LOG_DEBUG("Received Timestamp   : {}", detail::isoformat(t.received_s));
        HSB_LOG_DEBUG("Operator Timestamp   : {}", detail::isoformat(t.operator_s));
        HSB_LOG_DEBUG("Complete Timestamp   : {}", detail::isoformat(t.complete_s));
        HSB_LOG_DEBUG("Frame Time (dt)      : {:.6f} s", frame_time_dt);
        HSB_LOG_DEBUG("CPU Latency (dt)     : {:.6f} s", cpu_latency_dt);
        HSB_LOG_DEBUG("Operator Latency (dt): {:.6f} s", operator_latency_dt);
        HSB_LOG_DEBUG("Processing Time (dt) : {:.6f} s", processing_time_dt);
        HSB_LOG_DEBUG("Overall Time (dt)    : {:.6f} s", overall_time_dt);
    }

    HSB_LOG_INFO("** Complete report: **");
    HSB_LOG_INFO("{:<30}{:<15}{:<15}{:<15}", "Metric", "Min", "Max", "Avg");
    detail::report_metric("Frame Time (in sec):", "Frame Time", frame_time);
    detail::report_metric("FGPA frame transfer latency (in sec):", "Frame Transfer Latency", transfer);
    detail::report_metric(
        "FGPA to Operator after network operator latency (in sec):", "Operator Latency", operator_latency);
    detail::report_metric(
        "Processing of frame latency until display hand-off (in sec):", "Processing Latency", processing);
    detail::report_metric("Frame start till display hand-off latency (in sec):", "SW Pipeline Latency", overall);
    return EXIT_SUCCESS;
}

// ---- The run ------------------------------------------------------------------------------------

// Runs the measurement and returns the exit status. \p add_receiver(graph, channel, camera,
// cu_context, frame_size) adds the receiver operator, named "receiver", and returns its handle.
template <typename AddReceiver>
int run(const Options& options, const char* graph_name, AddReceiver&& add_receiver,
    holoscan::ManagedShutdownSignals& signals)
{
    HSB_LOG_INFO("Initializing.");
    // Get a handle to the GPU
    CudaCheck(cuInit(0));
    const int cu_device_ordinal = 0;
    CUdevice cu_device;
    CudaCheck(cuDeviceGet(&cu_device, cu_device_ordinal));
    CUcontext cu_context;
    CudaCheck(cuDevicePrimaryCtxRetain(&cu_context, cu_device));
    cuCtxSetCurrent(cu_context);
    // Get a handle to the Hololink device
    hololink::Metadata channel_metadata = hololink::Enumerator::find_channel(options.hololink_ip);
    if (options.roce) {
        HSB_LOG_INFO("channel_metadata={}", channel_metadata);
    }
    auto hololink_channel = std::make_shared<hololink::DataChannel>(channel_metadata);
    // Get a handle to the camera
    auto camera = std::make_shared<hololink::sensors::NativeImx274Sensor>(
        *hololink_channel, options.expander_configuration);
    const auto camera_mode = static_cast<hololink::sensors::imx274_mode::Mode>(options.camera_mode);
    camera->set_mode(camera_mode);
    const auto width = static_cast<uint32_t>(camera->get_width());
    const auto height = static_cast<uint32_t>(camera->get_height());
    const auto bayer_format = camera->get_bayer_format();
    const auto pixel_format = camera->get_pixel_format();

    holoscan::Graph graph { graph_name };

    hololink::operators::CsiToBayerParams csi_params {
        .cuda_device_ordinal = cu_device_ordinal,
        .bayer_format = bayer_format,
    };
    camera->configure_converter(csi_params.geometry);
    const auto& geometry = csi_params.geometry;
    const size_t frame_size = geometry.start_byte
        + static_cast<size_t>(geometry.bytes_per_line) * geometry.pixel_height + geometry.trailing_bytes;
    // CsiToBayerOp stamps when it starts on a frame: the HSDK4 tool's "operator timestamp".
    auto csi = graph.op<hololink::operators::CsiToBayerOp>("csi_to_bayer");
    csi.set_params(csi_params);

    auto receiver = add_receiver(graph, hololink_channel, camera, cu_context, frame_size);

    auto ip = graph.op<hololink::operators::ImageProcessorOp>("image_processor");
    ip.set_params(hololink::operators::ImageProcessorParams {
        .pixel_format = int(pixel_format),
        .bayer_format = int(bayer_format),
        .optical_black = 50, // Optical black value for imx274 is 50
        .cuda_device_ordinal = cu_device_ordinal,
        .pixel_width = geometry.pixel_width,
        .pixel_height = geometry.pixel_height,
    });

    auto demosaic = graph.op<holoscan::vision::BayerDemosaicOp>("demosaic");
    demosaic.set_params(holoscan::vision::BayerDemosaicOpParams {
        .bayer_grid_pos = static_cast<holoscan::vision::BayerGridPosition>(bayer_format),
        .generate_alpha = true,
        .alpha_value = 65535U,
    });

    holoscan::CompileOptions compile_options;
    holoscan::visualization::LayerStack layers;
    layers.add_image_layer("video", rgba16_layer(width, height));
    auto visualizer = add_holoviz(graph, "holoviz", std::move(layers),
        DisplayConfig { .fullscreen = options.fullscreen, .headless = options.headless }, compile_options,
        cu_device_ordinal);

    auto recorder = std::make_shared<LatencyRecorder>();
    auto handoff = graph.op<DisplayHandoffOp>("display_handoff");
    handoff.set_params(DisplayHandoffParams {
        .recorder = recorder,
        .frame_limit = options.frame_limit,
        .log_interval = options.log_interval,
    });

    const holoscan::ConnectionOptions edge = pipeline_edge();
    graph.add_flow(receiver->tensor, csi->input_tensor, edge);
    graph.add_flow(receiver->metadata, csi->input_metadata, edge);
    graph.add_flow(csi->output_tensor, ip->input_tensor, edge);
    graph.add_flow(csi->output_metadata, ip->input_metadata, edge);
    graph.add_flow(ip->output_tensor, demosaic->input, edge);
    // Each image goes to HolovizOp and, at the same time, to the hand-off timestamp. The hand-off
    // runs on the demosaic's thread, so its image queue never fills: BayerDemosaicOp would fail
    // the run on a full queue, while a hand-off that falls behind now only holds back the
    // demosaic, and ImageProcessorOp drops (and counts) the frames it can't hand on. The hand-off
    // queues don't drop the oldest, so every image the demosaic makes is timed with its own
    // metadata; the metadata queue is deeper than ImageProcessorOp's queue to the demosaic, so a
    // frame is dropped before its metadata could be.
    const holoscan::ConnectionOptions handoff_edge { .queue_depth = 4U };
    graph.add_flow(demosaic->output, visualizer->image_layer_input(0), display_edge());
    graph.add_flow(demosaic->output, handoff->image, handoff_edge);
    graph.add_flow(ip->output_metadata, handoff->metadata, handoff_edge);
    graph.partition("display_handoff").add(demosaic, handoff);
    graph.auto_partition_remaining();

    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

    for (const auto& [operator_path, output_port] :
        { std::pair<std::string_view, std::string_view> { "receiver", "tensor" }, { "csi_to_bayer", "tensor" },
            { "image_processor", "tensor" }, { "demosaic", "output" } }) {
        compile_options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement {
            .operator_path = operator_path,
            .output_port = output_port,
            .device = holoscan::DeviceId { cu_device_ordinal },
        });
    }
    const auto plan = holoscan::compile(graph, std::move(compile_options));
    if (!plan.ok()) {
        print_compile_failure(plan);
        CudaCheck(cuDevicePrimaryCtxRelease(cu_device));
        return EXIT_FAILURE;
    }

    // Run it.
    bool failed = false;
    {
        std::shared_ptr<hololink::Hololink> hololink = hololink_channel->hololink();
        hololink->start();
        hololink::core::OnScopeExit stop_hololink([&hololink] { hololink->stop(); });
        if (!options.skip_reset) {
            hololink->reset();
        }
        HSB_LOG_DEBUG("Waiting for PTP sync.");
        if (!hololink->ptp_synchronize()) {
            HSB_LOG_ERROR("Failed to synchronize PTP.");
            return EXIT_FAILURE;
        }
        HSB_LOG_DEBUG("PTP synchronized.");
        if (!options.skip_reset) {
            camera->setup_clock();
        }
        camera->configure(camera_mode);
        camera->set_digital_gain_reg(0x4);
        if (options.pattern) {
            camera->enable_test_pattern(static_cast<uint8_t>(*options.pattern));
        }
        HSB_LOG_INFO("Calling run");
        auto session = holoscan::run_async(plan, signals);
        session.wait();
        failed = report_session(session, plan);
    }

    CudaCheck(cuDevicePrimaryCtxRelease(cu_device));

    // Ctrl+C and the window closing end the measurement too.
    return failed ? EXIT_FAILURE : report(recorder->frames());
}

// The receiver's device_stop: stops the camera, then lets the HSB's egress buffer drain, as the
// HSDK4 tools' Python camera driver did.
inline void stop_camera(hololink::sensors::NativeImx274Sensor& camera)
{
    camera.stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

} // namespace hololink::examples::imx274_latency

#endif /* HOLOLINK_EXAMPLES_IMX274_LATENCY_HPP */
