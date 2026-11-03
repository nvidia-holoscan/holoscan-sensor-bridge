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

// What the example players share to run like their Holoscan SDK 4 versions: Ctrl+C and SIGTERM
// stop a run cooperatively, a failed run prints its diagnostics and fails the process,
// --frame-limit ends the run, and --headless renders offscreen instead of into a window.
// Players without HoloViz define HOLOLINK_PLAYER_SUPPORT_NO_DISPLAY to leave the display part out.

#ifndef HOLOLINK_EXAMPLES_PLAYER_SUPPORT_HPP
#define HOLOLINK_EXAMPLES_PLAYER_SUPPORT_HPP

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/logger/logger.hpp>

#include <capture_metadata_traits.hpp>
#include <hololink/core/logging.hpp>
#include <hololink/operators/capture_metadata_utils.hpp>

#if !defined(HOLOLINK_PLAYER_SUPPORT_NO_DISPLAY)
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/visualization/holoviz_op.hpp>
#include <holoscan/visualization/sink_desc.hpp>
#endif

namespace hololink::examples {

// ---- Signals and the run's outcome --------------------------------------------------------------

// Runs \p player, an int(holoscan::ManagedShutdownSignals&) returning the exit status, with
// Holoscan managing Ctrl+C and SIGTERM. Call it from main() before anything starts a thread
// (cuInit(), enumeration, the board connection). While a session runs bound to the signals
// (holoscan::run_async(plan, signals)), the first Ctrl+C or SIGTERM stops it cooperatively, so
// the operators' stop() (camera stop, channel teardown) and the player's own teardown still run;
// before and after that, a signal takes its usual effect, as it did outside an HSDK4 player's
// Application::run(). An exception escaping \p player is printed and fails the process.
template <typename Player>
int run_with_managed_signals(Player&& player)
{
    auto created = holoscan::ManagedShutdownSignals::create();
    if (!created) {
        std::cerr << holoscan::runtime_error_message(created.error()) << std::endl;
        return EXIT_FAILURE;
    }
    auto signals = std::move(created).value();

    int result = EXIT_FAILURE;
    try {
        result = std::forward<Player>(player)(signals);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    if (auto closed = signals.close(); !closed) {
        std::cerr << holoscan::runtime_error_message(closed.error()) << std::endl;
        return EXIT_FAILURE;
    }
    return result;
}

inline void print_compile_failure(const holoscan::ExecutionPlan& plan)
{
    std::cerr << "Graph compilation failed:" << std::endl
              << holoscan::format_diagnostics(plan.diagnostics(), plan.graph()) << std::endl;
}

// Prints the session's diagnostics and returns whether the run failed. A run that completed
// (kCompleted; kRequestedCompleted at --frame-limit, with frame_limit_ends_run or where
// FrameLimitOp counts) or was stopped by Ctrl+C, SIGTERM or the window closing (kStopped) ended
// the way HSDK4 runs ended on the same events.
[[nodiscard]] inline bool report_session(const holoscan::RunSession& session, const holoscan::ExecutionPlan& plan)
{
    const auto graph = plan.graph();
    for (const auto& diagnostic : session.diagnostics()) {
        std::cerr << "Session diagnostic: " << holoscan::format_diagnostic(diagnostic, graph) << std::endl;
    }
    return session.termination() == holoscan::RunTermination::kFailed;
}

// ---- Edges --------------------------------------------------------------------------------------

// An edge between two operators of a camera chain. An HSDK4 edge queued one frame while its
// consumer worked on another; here the frame being worked on keeps its queue slot, so the same is
// two slots. A chain that runs on one thread never fills them. Where an edge leads to another
// thread, a hololink producer that finds the queue full drops its frame, but an SDK operator such
// as BayerDemosaicOp fails the run: keep its consumers on its thread, or behind display_edge().
[[nodiscard]] inline holoscan::ConnectionOptions pipeline_edge()
{
    return holoscan::ConnectionOptions { .queue_depth = 2U };
}

// Counts a frame that \p op dropped because receive_paired_tensor_metadata() found its payload and
// metadata stamped as different frames, and warns about such drops at most once a second.
inline void count_mismatched_frame(hololink::operators::DropCounter& mismatched, std::string_view op)
{
    if (const std::uint64_t dropped = mismatched.count()) {
        HOLOSCAN_LOG_WARN("{}: dropped {} frame(s) in the last second whose payload and metadata were "
                          "stamped as different frames",
            op, dropped);
    }
}

// ---- --frame-limit ------------------------------------------------------------------------------

// The FuSa players' FusaCoeCaptureParams::stall_timeout_ms for --frame-limit \p frame_limit and a
// capture request timeout of \p timeout_ms. HSDK4's capture operator counted each request that
// timed out toward its CountCondition, so a camera that sent nothing ended the run after
// frame_limit timeouts. HSDK5 counts frames only; this fails such a run after as long instead. 0
// (no stall limit) without a frame limit.
[[nodiscard]] inline uint32_t frame_limit_stall_timeout_ms(int64_t frame_limit, uint32_t timeout_ms)
{
    if (frame_limit <= 0) {
        return 0;
    }
    const uint64_t total_ms = static_cast<uint64_t>(frame_limit) * timeout_ms;
    return static_cast<uint32_t>(std::min<uint64_t>(total_ms, std::numeric_limits<uint32_t>::max()));
}

// The sub-frame players' frame limit, shared by their legs (one per camera chain). The other
// players limit frames in their receivers (frame_limit), as HSDK4 players did with a
// CountCondition; a sub-frame receiver delivers bands, so these players count whole frames at the
// end of the chain instead and complete the run once the last leg has seen that many.
class FrameLimitGroup {
public:
    FrameLimitGroup(int64_t limit, uint32_t legs)
        : limit_(limit)
        , legs_(legs ? legs : 1U)
    {
    }

    // 0 or less: no limit.
    [[nodiscard]] int64_t limit() const noexcept { return limit_; }

    // A leg reached the limit; true for the last one.
    [[nodiscard]] bool leg_done() noexcept { return legs_done_.fetch_add(1) + 1 == legs_; }

private:
    const int64_t limit_;
    const uint32_t legs_;
    std::atomic<uint32_t> legs_done_ { 0 };
};

struct FrameLimitParams {
    // The run's limit; add_frame_limit() adds the operator only when there is one.
    std::shared_ptr<FrameLimitGroup> group;
    // Sub-frame bands share their frame's frame_number: count a frame when a later one starts.
    bool count_frame_number_changes = false;
};

// Counts one leg's frames on its CaptureMetadataT stream and completes the run when the group's
// last leg reaches the limit. It drains its queue on a clock rather than activating per frame,
// which stalled a FuSa CoE capture chain; fed by add_frame_limit()'s drop-oldest edge, a stall
// here sheds frames from the count (so the run ends later) rather than holding the pipeline back.
class FrameLimitOp : public holoscan::Operator<FrameLimitParams> {
public:
    static constexpr uint32_t kQueueDepth = 8U;

    holoscan::Input<hololink::CaptureMetadataT> input_metadata { "metadata" };

    void setup(holoscan::OperatorSpec& spec) override { spec.input(input_metadata, "metadata").queue_depth(kQueueDepth); }

    [[nodiscard]] holoscan::Contract contract() const override
    {
        holoscan::Contract c;
        c.trigger(holoscan::OnClock { .period = std::chrono::milliseconds(16) })
            .input(input_metadata, holoscan::DrainAll {});
        return c;
    }

    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override
    {
        const auto& p = params();
        int64_t arrived = 0;
        auto drained = input_metadata.for_each_selected(
            [&](const holoscan::Sample<hololink::CaptureMetadataT>& sample) -> holoscan::expected<void, holoscan::Error> {
                if (!p.count_frame_number_changes) {
                    ++arrived;
                } else if (sample.data.frame_number > frame_number_) {
                    // A late band of an earlier frame changes nothing.
                    if (frame_number_ >= 0) {
                        ++arrived;
                    }
                    frame_number_ = sample.data.frame_number;
                }
                return {};
            });
        if (!drained) {
            return holoscan::make_unexpected(std::move(drained).error());
        }
        if (!p.group || p.group->limit() <= 0 || leg_done_) {
            return {};
        }
        frames_ += arrived;
        if (frames_ >= p.group->limit()) {
            leg_done_ = true;
            if (p.group->leg_done()) {
                context.request_completion();
            }
        }
        return {};
    }

private:
    int64_t frames_ = 0;
    int64_t frame_number_ = -1;
    bool leg_done_ = false;
};

// Counts the frames on \p metadata, a CaptureMetadataT output, as one leg of \p group. Without a
// limit it adds nothing, as HSDK4 players added no CountCondition: \p metadata stays unconnected,
// which hololink's operators allow.
template <typename MetadataOutput>
void add_frame_limit(holoscan::Graph& graph, const std::string& name, const MetadataOutput& metadata,
    std::shared_ptr<FrameLimitGroup> group, bool count_frame_number_changes = false)
{
    if (!group || group->limit() <= 0) {
        return;
    }
    auto op = graph.op<FrameLimitOp>(name);
    op.set_params(FrameLimitParams {
        .group = std::move(group),
        .count_frame_number_changes = count_frame_number_changes,
    });
    graph.add_flow(metadata, op->input_metadata,
        holoscan::ConnectionOptions {
            .queue_policy = holoscan::QueuePolicy::kDropOldest,
            .queue_depth = FrameLimitOp::kQueueDepth,
        });
}

// ---- Options ------------------------------------------------------------------------------------

// trace, debug, info, warn, error, critical or off, in lower or upper case, as HSDK4 players took
// --log-level.
[[nodiscard]] inline std::optional<holoscan::LogLevel> parse_log_level(std::string_view name)
{
    using holoscan::LogLevel;
    static constexpr std::pair<std::string_view, LogLevel> kLevels[] = {
        { "trace", LogLevel::TRACE },
        { "TRACE", LogLevel::TRACE },
        { "debug", LogLevel::DEBUG },
        { "DEBUG", LogLevel::DEBUG },
        { "info", LogLevel::INFO },
        { "INFO", LogLevel::INFO },
        { "warn", LogLevel::WARN },
        { "WARN", LogLevel::WARN },
        { "error", LogLevel::ERROR },
        { "ERROR", LogLevel::ERROR },
        { "critical", LogLevel::CRITICAL },
        { "CRITICAL", LogLevel::CRITICAL },
        { "off", LogLevel::OFF },
        { "OFF", LogLevel::OFF },
    };
    for (const auto& [level_name, level] : kLevels) {
        if (name == level_name) {
            return level;
        }
    }
    return std::nullopt;
}

// The level of Python logging level \p level, as HSDK4's Python tools took --log-level: 10 DEBUG,
// 20 INFO, 30 WARNING, 40 ERROR, 50 CRITICAL; below 10 is TRACE, above 50 OFF.
[[nodiscard]] inline holoscan::LogLevel log_level_from_python(int level)
{
    if (level < 10) {
        return holoscan::LogLevel::TRACE;
    }
    if (level < 20) {
        return holoscan::LogLevel::DEBUG;
    }
    if (level < 30) {
        return holoscan::LogLevel::INFO;
    }
    if (level < 40) {
        return holoscan::LogLevel::WARN;
    }
    if (level < 50) {
        return holoscan::LogLevel::ERROR;
    }
    return level == 50 ? holoscan::LogLevel::CRITICAL : holoscan::LogLevel::OFF;
}

// Makes hololink's own (HSB) logger show what \p level shows. It has no CRITICAL level: CRITICAL
// shows its errors, OFF nothing.
inline void set_hsb_log_level(holoscan::LogLevel level)
{
    using hololink::logging::HsbLogLevel;
    switch (level) {
    case holoscan::LogLevel::TRACE:
        hololink::logging::hsb_log_level = HsbLogLevel::HSB_LOG_LEVEL_TRACE;
        break;
    case holoscan::LogLevel::DEBUG:
        hololink::logging::hsb_log_level = HsbLogLevel::HSB_LOG_LEVEL_DEBUG;
        break;
    case holoscan::LogLevel::INFO:
        hololink::logging::hsb_log_level = HsbLogLevel::HSB_LOG_LEVEL_INFO;
        break;
    case holoscan::LogLevel::WARN:
        hololink::logging::hsb_log_level = HsbLogLevel::HSB_LOG_LEVEL_WARN;
        break;
    case holoscan::LogLevel::ERROR:
    case holoscan::LogLevel::CRITICAL:
        hololink::logging::hsb_log_level = HsbLogLevel::HSB_LOG_LEVEL_ERROR;
        break;
    default:
        hololink::logging::hsb_log_level = static_cast<HsbLogLevel>(HsbLogLevel::HSB_LOG_LEVEL_ERROR + 10);
        break;
    }
}

// HSDK4 players passed --configuration to Application::config(), which rejected a file that does
// not exist, and read nothing from it; HSDK5 applications have no YAML configuration. Returns
// false, after logging why, when \p path does not exist.
[[nodiscard]] inline bool check_configuration(const std::string& path)
{
    std::error_code error;
    if (!std::filesystem::exists(path, error)) {
        HOLOSCAN_LOG_ERROR("Configuration file {} does not exist", path);
        return false;
    }
    HOLOSCAN_LOG_WARN("Ignoring --configuration {}: Holoscan SDK 5 applications read no YAML configuration", path);
    return true;
}

#if !defined(HOLOLINK_PLAYER_SUPPORT_NO_DISPLAY)
// ---- Display ------------------------------------------------------------------------------------

struct DisplayConfig {
    std::string title = "Holoviz";
    uint32_t width = 1920;
    uint32_t height = 1080;
    bool fullscreen = false;
    // Render offscreen, into a CUDA image of the window's size, instead of into a window.
    bool headless = false;
    // HSDK4's HolovizOp (vsync off) preferred MAILBOX. HSDK5 kAuto picks a tear-free mode without
    // it (FIFO_LATEST_READY when available); an explicit kMailbox fails start() on a surface
    // without MAILBOX instead of falling back.
    holoscan::visualization::PresentMode present_mode = holoscan::visualization::PresentMode::kAuto;
    // HolovizOp's max_input_age: each layer shows its newest image no older than this, and a render
    // is skipped while a layer that does not trigger renders has none. Cameras triggered at a low
    // rate need max_input_age_for_rate().
    std::chrono::nanoseconds max_input_age = std::chrono::milliseconds(100);
};

// A max_input_age for a window of cameras triggered together at \p hz: two trigger periods, so that
// a layer can still show the previous trigger's image while this trigger's has not arrived; never
// less than the default.
[[nodiscard]] inline std::chrono::nanoseconds max_input_age_for_rate(unsigned hz)
{
    const std::chrono::nanoseconds fallback = DisplayConfig {}.max_input_age;
    return hz ? std::max(fallback, std::chrono::nanoseconds(std::chrono::seconds(2)) / hz) : fallback;
}

[[nodiscard]] inline holoscan::visualization::SinkDesc make_sink(const DisplayConfig& display)
{
    namespace viz = holoscan::visualization;
    if (display.headless) {
        return viz::CudaSinkDesc {
            .width = display.width,
            .height = display.height,
            .format = viz::ImageFormat::kR8G8B8A8Unorm,
        };
    }
    return viz::WindowSinkDesc {
        .title = display.title,
        .width = display.width,
        .height = display.height,
        .present_mode = display.present_mode,
        .fullscreen = display.fullscreen,
        .close_on_escape = true,
        // The demosaiced images are linear sensor data: encode to sRGB on write.
        .framebuffer_srgb = true,
    };
}

// For every edge into HolovizOp: a depth-1 mailbox, so a display that falls behind skips to the
// newest frame instead of holding the pipeline back. Across threads (a HolovizOp fed by several
// cameras runs on a thread of its own) this needs Holoscan SDK 5.0.0.7: earlier builds don't size
// the producer's tensor pool for a drop-oldest edge, and BayerDemosaicOp then runs out of slots.
[[nodiscard]] inline holoscan::ConnectionOptions display_edge()
{
    return holoscan::ConnectionOptions {
        .queue_policy = holoscan::QueuePolicy::kDropOldest,
        .queue_depth = 1U,
    };
}

// An image layer for BayerDemosaicOp's RGBA16 output. The maxima are the sensor's image size:
// HolovizOp rejects larger images, and the window size is independent of them.
[[nodiscard]] inline holoscan::visualization::ImageLayerDesc rgba16_layer(uint32_t width, uint32_t height,
    std::optional<holoscan::visualization::ViewTransform> view = std::nullopt)
{
    return holoscan::visualization::ImageLayerDesc {
        .format = holoscan::visualization::ImageFormat::kR16G16B16A16Unorm,
        .max_width = width,
        .max_height = height,
        .view = view,
    };
}

// Where layer \p index of \p count goes in a window showing them side by side.
[[nodiscard]] inline holoscan::visualization::ViewTransform side_by_side_view(std::size_t index, std::size_t count)
{
    const float width = 1.f / static_cast<float>(count);
    return holoscan::visualization::ViewTransform {
        .offset_x = width * static_cast<float>(index),
        .offset_y = 0.f,
        .scale_x = width,
        .scale_y = 1.f,
    };
}

// Takes the images a headless HolovizOp renders, and drops them.
class DiscardImageOp : public holoscan::Operator<> {
public:
    holoscan::Input<holoscan::schema::ImageT> input { "input" };

    void setup(holoscan::OperatorSpec& spec) override
    {
        (void)spec.input(input, "input").tensor_field(holoscan::schema::ImageTensorFields::data);
    }

    [[nodiscard]] holoscan::Contract contract() const override
    {
        holoscan::Contract c;
        c.trigger(holoscan::OnEach { input });
        return c;
    }

    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override
    {
        auto sample = input.receive();
        if (!sample) {
            return holoscan::make_unexpected(std::move(sample).error());
        }
        return {};
    }
};

// Adds the player's HolovizOp \p name: a window, or with display.headless an offscreen CUDA
// image that the graph drains (its "output" is bound to the CUDA device in \p compile_options).
// By default it renders once per image on layer 0; connect that edge with display_edge().
inline holoscan::OperatorHandle<holoscan::visualization::HolovizOp> add_holoviz(holoscan::Graph& graph,
    const std::string& name, holoscan::visualization::LayerStack layers, const DisplayConfig& display,
    holoscan::CompileOptions& compile_options, int cuda_device_ordinal,
    holoscan::visualization::FrameTrigger trigger = holoscan::visualization::LayerTrigger {})
{
    auto viz = graph.op<holoscan::visualization::HolovizOp>(
        name, std::move(layers), make_sink(display), std::move(trigger), display.max_input_age);
    if (display.headless) {
        auto discard = graph.op<DiscardImageOp>(name + "_discard");
        graph.add_flow(viz->output(), discard->input, display_edge());
        compile_options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement {
            .operator_path = name,
            .output_port = "output",
            .device = holoscan::DeviceId { cuda_device_ordinal },
        });
    }
    return viz;
}
#endif

} // namespace hololink::examples

#endif /* HOLOLINK_EXAMPLES_PLAYER_SUPPORT_HPP */
