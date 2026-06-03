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

#include "sub_frame_visualizer.hpp"

#include <yaml-cpp/parser.h>

namespace viz = holoscan::viz;

namespace {
/**
 * RAII type class to push a Holoviz instance. Previous instance will be restored when class
 * instance goes out of scope.
 */
class ScopedPushInstance {
public:
    /**
     * Push a Holoviz instance
     *
     * @param instance instance to push
     */
    explicit ScopedPushInstance(holoscan::viz::InstanceHandle instance)
        : prev_instance_(viz::GetCurrent())
    {
        viz::SetCurrent(instance);
    }

    /**
     * Destructor, restore the previous instance.
     */
    ~ScopedPushInstance() { viz::SetCurrent(prev_instance_); }

private:
    // hide default and copy constructors, copy assignment
    ScopedPushInstance() = delete;
    ScopedPushInstance(const ScopedPushInstance&) = delete;
    ScopedPushInstance& operator=(const ScopedPushInstance&) = delete;

    const holoscan::viz::InstanceHandle prev_instance_;
};

constexpr auto NS_PER_SEC = 1'000'000'000L;
constexpr auto MS_PER_SEC = 1'000L;
constexpr auto US_PER_SEC = 1'000'000L;
constexpr auto NS_PER_MS = NS_PER_SEC / MS_PER_SEC;
constexpr auto NS_PER_US = NS_PER_SEC / US_PER_SEC;

static bool less(const struct timespec& time1, const struct timespec& time0)
{
    return time1.tv_sec < time0.tv_sec || (time1.tv_sec == time0.tv_sec && time1.tv_nsec < time0.tv_nsec);
}

static bool greater(const struct timespec& time1, const struct timespec& time0)
{
    return time1.tv_sec > time0.tv_sec || (time1.tv_sec == time0.tv_sec && time1.tv_nsec > time0.tv_nsec);
}

static struct timespec sub(const struct timespec& time1, const struct timespec& time0)
{
    struct timespec diff = { .tv_sec = time1.tv_sec - time0.tv_sec, .tv_nsec = time1.tv_nsec - time0.tv_nsec };
    if (diff.tv_nsec < 0) {
        diff.tv_nsec += NS_PER_SEC;
        diff.tv_sec--;
    }
    return diff;
}

static struct timespec add(const struct timespec& time1, const struct timespec& time0)
{
    struct timespec sum = { .tv_sec = time1.tv_sec + time0.tv_sec, .tv_nsec = time1.tv_nsec + time0.tv_nsec };
    if (sum.tv_nsec >= NS_PER_SEC) {
        sum.tv_nsec -= NS_PER_SEC;
        sum.tv_sec++;
    }
    return sum;
}

static struct timespec max(const struct timespec& time1, const struct timespec& time0)
{
    if (less(time1, time0)) {
        return time0;
    }
    return time1;
}

/** Remainder of two non-negative durations in nanoseconds; O(1). Use for phase within display_period. */
static struct timespec timespec_mod_period_ns(const struct timespec& dividend, const struct timespec& period)
{
    const uint64_t div_ns = static_cast<uint64_t>(dividend.tv_sec) * static_cast<uint64_t>(NS_PER_SEC)
        + static_cast<uint64_t>(dividend.tv_nsec);
    const uint64_t period_ns = static_cast<uint64_t>(period.tv_sec) * static_cast<uint64_t>(NS_PER_SEC)
        + static_cast<uint64_t>(period.tv_nsec);
    if (period_ns == 0) {
        return { 0, 0 };
    }
    const uint64_t rem = div_ns % period_ns;
    return { 0, static_cast<long>(rem) };
}

static double to_seconds(const struct timespec& time)
{
    return static_cast<double>(time.tv_sec) + static_cast<double>(time.tv_nsec) / NS_PER_SEC;
}

static double to_ms(const struct timespec& time)
{
    return static_cast<double>(time.tv_sec) * MS_PER_SEC + static_cast<double>(time.tv_nsec) / NS_PER_MS;
}

/**
 * Nanosecond phase written to FPGA VSYNC_DELAY from absolute capture_time (wall clock).
 * Single-pulse PPS: hardware wraps each second — use cap_ns % 1s.
 * Multi-pulse / vsync train: phase within one display period — use cap_ns % display_period.
 */
static uint32_t fpga_vsync_delay_ns(const struct timespec& capture_time, const struct timespec& display_period,
    bool ptp_single_pulse_pps)
{
    const auto cap_ns = static_cast<uint64_t>(capture_time.tv_sec) * static_cast<uint64_t>(NS_PER_SEC)
        + static_cast<uint64_t>(capture_time.tv_nsec);
    if (ptp_single_pulse_pps) {
        return static_cast<uint32_t>(cap_ns % static_cast<uint64_t>(NS_PER_SEC));
    }
    const auto period_ns = static_cast<uint64_t>(display_period.tv_sec) * static_cast<uint64_t>(NS_PER_SEC)
        + static_cast<uint64_t>(display_period.tv_nsec);
    if (period_ns == 0) {
        return static_cast<uint32_t>(capture_time.tv_nsec);
    }
    return static_cast<uint32_t>(cap_ns % period_ns);
}

} // namespace

template <>
struct YAML::convert<hololink::Hololink::PtpSynchronizer*> {
    static Node encode(hololink::Hololink::PtpSynchronizer*&) { throw std::runtime_error("Unsupported"); }

    static bool decode(const Node&, hololink::Hololink::PtpSynchronizer*&) { throw std::runtime_error("Unsupported"); }
};

namespace hololink::operators {

void SubFrameVisualizerOp::thread_func()
{
    ScopedPushInstance scoped_instance(instance_);

    // this is a (arbitrary selected) delay to make sure the delay is not expired before the register is written
    constexpr struct timespec MIN_CAPTURE_DELAY_FROM_NOW = { 0, 2 * NS_PER_MS };
    // this is a delay added to the capture to render time to make sure the frame is rendered before the display driver reads
    // (the processing time might vary depending on the system load)
    constexpr struct timespec PROCESSING_MARGIN = { 0, 2 * NS_PER_MS };

    struct timespec capture_time = { 0, 0 };
    struct timespec last_fpo_render_start_time = { 0, 0 };
    struct timespec cached_capture_to_render_delay = { 0, 0 };

    // Initialize display period from parameter (will be refined by FPO measurements)
    double framerate_hz = display_framerate_.get() > 0 ? (display_framerate_.get() / 1000.0) : 59.95;
    struct timespec display_period = { 0, static_cast<long>(NS_PER_SEC / framerate_hz) };
    HOLOSCAN_LOG_INFO("Initial display refresh rate: {:.2f} Hz (will be measured from FPO events)", framerate_hz);

    // Variables for FPO-based refresh rate measurement
    struct timespec previous_fpo_time = { 0, 0 };
    uint32_t fpo_sample_count = 0;
    static constexpr uint32_t FPO_SAMPLES_NEEDED = 10; // Number of samples to average for stable measurement

    // Fold pipeline delay phase into one PTP/PPS second (not one display period) before subtracting from next frame tick.
    static constexpr struct timespec kPtpPpsModulus = { 1, 0 };

    // Match FPGA: single-pulse PPS arms once per second — set_delay is phase mod 1s. Use false with ptp_pps_output(60).
    static constexpr bool kPtpSinglePulsePpsFpgaDelay = true;

    while (!stop_requested_) {
        // choose the timeout to be longer than the display period (assume 10Hz)
        const bool is_ready = viz::WaitForDisplayEvent(viz::DisplayEventType::FIRST_PIXEL_OUT, NS_PER_SEC);
        fpo_available_.store(is_ready);

        if (is_ready) {
            struct timespec now;
            if (clock_gettime(CLOCK_REALTIME, &now) != 0) {
                throw std::runtime_error(fmt::format("clock_gettime failed: {}", errno));
            }

            // Measure display refresh rate dynamically from FPO events
            // Measure the time between two FPO callbacks for the first 10 frames
            if ((previous_fpo_time.tv_sec != 0 || previous_fpo_time.tv_nsec != 0) && fpo_sample_count < FPO_SAMPLES_NEEDED) {
                // Calculate period from time between FPO events
                struct timespec measured_period = sub(now, previous_fpo_time);

                if (fpo_sample_count == 0) {
                    // First measurement, use it directly
                    display_period = measured_period;
                } else {
                    // Average with previous measurements for stability using full 64-bit nanoseconds
                    // to avoid dropping tv_sec (which corrupts the average for periods >= 1s or near it).
                    const uint64_t prev_ns = static_cast<uint64_t>(display_period.tv_sec) * static_cast<uint64_t>(NS_PER_SEC)
                        + static_cast<uint64_t>(display_period.tv_nsec);
                    const uint64_t meas_ns = static_cast<uint64_t>(measured_period.tv_sec) * static_cast<uint64_t>(NS_PER_SEC)
                        + static_cast<uint64_t>(measured_period.tv_nsec);
                    const uint64_t avg_ns = (prev_ns * fpo_sample_count + meas_ns) / (fpo_sample_count + 1);
                    display_period = { static_cast<time_t>(avg_ns / NS_PER_SEC), static_cast<long>(avg_ns % NS_PER_SEC) };
                }

                fpo_sample_count++;
                if (fpo_sample_count == FPO_SAMPLES_NEEDED) {
                    const uint64_t period_ns = static_cast<uint64_t>(display_period.tv_sec) * static_cast<uint64_t>(NS_PER_SEC)
                        + static_cast<uint64_t>(display_period.tv_nsec);
                    double measured_fps = period_ns > 0 ? static_cast<double>(NS_PER_SEC) / static_cast<double>(period_ns) : 0.0;
                    HOLOSCAN_LOG_INFO("Measured display refresh rate from {} FPO events: {:.2f} Hz",
                        FPO_SAMPLES_NEEDED, measured_fps);
                }
            }
            previous_fpo_time = now;

            // get the time the previous render started
            struct timespec render_start_time;
            {
                std::lock_guard<std::mutex> lock(render_start_time_mutex_);
                render_start_time = render_start_time_;
            }

            uint32_t delay_ns_for_fpga = 0;

            if ((render_start_time.tv_sec == 0) && (render_start_time.tv_nsec == 0)) {
                // if this is the first frame, since we don't know the delays yet, trigger the camera capture just now
                capture_time = add(now, MIN_CAPTURE_DELAY_FROM_NOW);
                delay_ns_for_fpga = fpga_vsync_delay_ns(capture_time, display_period, kPtpSinglePulsePpsFpgaDelay);
            } else {
                // we want that the next camera frame is rendered just before the next display refresh

                // If render has started but capture_time was never set (e.g. FPO before first schedule path),
                // sub(render_start, {0,0}) is ~wall-clock epoch and poisons schedule math; initialize like first frame.
                if ((capture_time.tv_sec == 0) && (capture_time.tv_nsec == 0)) {
                    HOLOSCAN_LOG_WARN(
                        "capture_time unset while render_start is valid; initializing schedule (was using epoch in delay math)");
                    capture_time = add(now, MIN_CAPTURE_DELAY_FROM_NOW);
                    delay_ns_for_fpga = fpga_vsync_delay_ns(capture_time, display_period, kPtpSinglePulsePpsFpgaDelay);
                } else {
                    // Detect slow camera: render_start_time unchanged since last FPO means
                    // no new frame arrived this display period — camera rate < display rate.
                    // Warn but do not skip: always schedule a capture to keep the pipeline full.
                    if (render_start_time.tv_sec == last_fpo_render_start_time.tv_sec
                        && render_start_time.tv_nsec == last_fpo_render_start_time.tv_nsec
                        && (render_start_time.tv_sec != 0 || render_start_time.tv_nsec != 0)) {
                        HOLOSCAN_LOG_WARN("No new frame since last FPO — camera rate may be less than display rate");
                    }
                    last_fpo_render_start_time = render_start_time;

                    // Calculate pipeline delay from the last completed frame.
                    // If the previous frame is still in-flight (no new render yet), reuse the
                    // cached delay from the last known good measurement. Using {0,0} here instead
                    // would cause delay_ns_for_fpga to ramp by display_period every FPO tick,
                    // which causes the FPGA to restart its capture burst at a new phase each tick
                    // and the camera stops producing frames.
                    struct timespec capture_to_render_delay;
                    if (less(render_start_time, capture_time)) {
                        capture_to_render_delay = cached_capture_to_render_delay;
                    } else {
                        cached_capture_to_render_delay = sub(render_start_time, capture_time);
                        capture_to_render_delay = cached_capture_to_render_delay;
                    }
                    // calculate the time we need to wait between the capture time and the vblank time
                    // add a delta to make sure the frame is rendered before the display driver reads
                    // out the frame buffer
                    struct timespec capture_to_vblank_delay = add(capture_to_render_delay, PROCESSING_MARGIN);

                    // Next capture tick: advance one display period, then pull back by pipeline delay phase
                    // folded into one PTP/PPS second (aligns with FPGA delay relative to second boundary).
                    capture_time = add(now, display_period);
                    struct timespec capture_to_vblank_phase = timespec_mod_period_ns(capture_to_vblank_delay, kPtpPpsModulus);
                    capture_time = sub(capture_time, capture_to_vblank_phase);

                    // make sure there is enough time for the FPGA to trigger the capture
                    capture_time = max(capture_time, add(now, MIN_CAPTURE_DELAY_FROM_NOW));
                    delay_ns_for_fpga = fpga_vsync_delay_ns(capture_time, display_period, kPtpSinglePulsePpsFpgaDelay);
                }
            }
            HOLOSCAN_LOG_DEBUG("vsync delay(ms) for fpga = {:.3f}", delay_ns_for_fpga / 1000000.0);
            ptp_synchronizer_->set_delay(delay_ns_for_fpga);
        }
    }
}

void SubFrameVisualizerOp::setup(holoscan::OperatorSpec& spec)
{
    register_converter<Hololink::PtpSynchronizer*>();

    spec.input<holoscan::gxf::Entity>("input");

    spec.param(fullscreen_, "fullscreen", "Fullscreen",
        "Fullscreen mode", false);
    spec.param(use_exclusive_display_, "use_exclusive_display", "UseExclusiveDisplay",
        "Exclusive display mode", false);
    spec.param(display_name_, "display_name", "DisplayName",
        "Display name", std::string(""));
    spec.param(display_width_, "display_width", "DisplayWidth",
        "Width of the display", 1920U);
    spec.param(display_height_, "display_height", "DisplayHeight",
        "Height of the display", 1080U);
    spec.param(display_framerate_, "display_framerate", "DisplayFramerate",
        "Framerate of the display in Hz/1000 (e.g. 60000 for 60Hz, 59950 for 59.95Hz)", 59'950U);
    spec.param(window_title_, "window_title", "WindowTitle",
        "Window title", std::string("Sub-Frame Visualizer"));
    spec.param(ptp_synchronizer_, "ptp_synchronizer", "PtpSynchronizer",
        "Pointer to PtpSynchronizer object. If provided, the operator will synchronize the display with the camera capture. "
        "If not provided the display refresh and the camera capture will run independently resulting in tearing.",
        (Hololink::PtpSynchronizer*)(nullptr));

    spec.param(full_frame_height_, "full_frame_height", "FullFrameHeight",
        "Height of the full frame");

    window_close_condition_ = fragment()->make_condition<holoscan::BooleanCondition>("window_close_condition");
    add_arg(window_close_condition_);
}

void SubFrameVisualizerOp::start()
{
    // create Holoviz instance
    instance_ = viz::Create();
    // make the instance current
    ScopedPushInstance scoped_instance(instance_);

    // initialize Holoviz
    if (use_exclusive_display_) {
        // shared continuous refresh means that both the application and the presentation engine can access the same image
        viz::SetPresentMode(viz::PresentMode::SHARED_CONTINUOUS_REFRESH);

        viz::Init(
            display_name_.get().c_str(), display_width_.get(), display_height_.get(),
            uint32_t(display_framerate_.get()), viz::InitFlags::NONE);

        // need to swap once when using shared continuous refresh mode
        viz::Begin(viz::RenderFlags::DONT_CLEAR_COLOR | viz::RenderFlags::DONT_CLEAR_DEPTH);
        viz::End();
    } else {
        viz::InitFlags init_flags = viz::InitFlags::NONE;
        if (fullscreen_) {
            init_flags = viz::InitFlags::FULLSCREEN;
        }

        viz::Init(display_width_.get(),
            display_height_.get(),
            window_title_.get().c_str(),
            init_flags,
            display_name_.get().empty() ? nullptr : display_name_.get().c_str());
    }

    // pick a SRGB surface format
    uint32_t surface_format_count = 0;
    viz::GetSurfaceFormats(&surface_format_count, nullptr);
    std::vector<viz::SurfaceFormat> surface_formats(surface_format_count);
    viz::GetSurfaceFormats(&surface_format_count, surface_formats.data());
    bool found_surface_format = false;
    for (const auto& format : surface_formats) {
        if ((format.image_format_ == viz::ImageFormat::R8G8B8A8_SRGB) || (format.image_format_ == viz::ImageFormat::B8G8R8A8_SRGB) || (format.image_format_ == viz::ImageFormat::A8B8G8R8_SRGB_PACK32)) {
            viz::SetSurfaceFormat(format);
            found_surface_format = true;
            break;
        }
    }
    if (!found_surface_format) {
        throw std::runtime_error("No supported SRGB surface format found");
    }

    if (ptp_synchronizer_) {
        // start the thread to wait for the first pixel out and trigger camera capture
        thread_ = std::thread(&SubFrameVisualizerOp::thread_func, this);
    }

    window_close_condition_->enable_tick();
}

void SubFrameVisualizerOp::stop()
{
    stop_requested_ = true;
    if (thread_.joinable()) {
        thread_.join();
    }

    viz::Shutdown(instance_);
}

void SubFrameVisualizerOp::compute(holoscan::InputContext& op_input,
    holoscan::OutputContext& op_output,
    holoscan::ExecutionContext& context)
{
    auto maybe_entity = op_input.receive<holoscan::gxf::Entity>("input");
    if (!maybe_entity) {
        throw std::runtime_error("Failed to receive input");
    }
    auto& entity = static_cast<nvidia::gxf::Entity&>(maybe_entity.value());

    const auto maybe_tensor = entity.get<nvidia::gxf::Tensor>();
    if (!maybe_tensor) {
        throw std::runtime_error("Tensor not found in message");
    }

    const auto input_tensor = maybe_tensor.value();
    if (input_tensor->rank() != 3) {
        throw std::runtime_error("Tensor must be two dimensional");
    }
    if (input_tensor->element_type() != nvidia::gxf::PrimitiveType::kUnsigned16) {
        throw std::runtime_error("Tensor must be unsigned 16-bit");
    }
    if (input_tensor->storage_type() != nvidia::gxf::MemoryStorageType::kDevice) {
        throw std::runtime_error("Tensor must be stored on the device");
    }

    const auto& input_shape = input_tensor->shape();
    const auto tensor_height = input_shape.dimension(0);
    const auto tensor_width = input_shape.dimension(1);

    if (input_shape.dimension(2) != 4) {
        throw std::runtime_error("Tensor must have 4 elements");
    }

    const auto sub_frame_offset = metadata()->get<int64_t>("sub_frame_offset");

    viz::RenderFlags render_flags = viz::RenderFlags::NONE;
    if (use_exclusive_display_) {
        if (fpo_available_.load()) {
            render_flags = viz::RenderFlags::DONT_CLEAR_COLOR | viz::RenderFlags::DONT_CLEAR_DEPTH | viz::RenderFlags::DONT_SWAP_BUFFERS;
        } else {
            // FPO failed: swap on last sub-frame so the window updates
            render_flags = viz::RenderFlags::DONT_CLEAR_COLOR | viz::RenderFlags::DONT_CLEAR_DEPTH;
        }
    } else {
        if (sub_frame_offset + tensor_height == full_frame_height_.get()) {
            // last sub frame, don't clear color or depth, but swap buffers
            render_flags = render_flags | viz::RenderFlags::DONT_CLEAR_COLOR | viz::RenderFlags::DONT_CLEAR_DEPTH;
        } else if (sub_frame_offset == 0) {
            // first sub frame, don't swap buffers
            render_flags = render_flags | viz::RenderFlags::DONT_SWAP_BUFFERS;
        } else {
            // middle sub frames, don't clear color or depth, and don't swap buffers
            render_flags = render_flags | viz::RenderFlags::DONT_CLEAR_COLOR | viz::RenderFlags::DONT_CLEAR_DEPTH | viz::RenderFlags::DONT_SWAP_BUFFERS;
        }
    }

    // record the render start time when the first sub frame is rendered
    if (sub_frame_offset == 0) {
        struct timespec now;
        if (clock_gettime(CLOCK_REALTIME, &now) != 0) {
            throw std::runtime_error(fmt::format("clock_gettime failed: {}", errno));
        }
        std::lock_guard<std::mutex> lock(render_start_time_mutex_);
        render_start_time_ = now;
    }

    ScopedPushInstance scoped_instance(instance_);

    if (viz::WindowShouldClose()) {
        window_close_condition_->disable_tick();
    }

    auto cuda_stream = op_input.receive_cuda_stream();
    viz::SetCudaStream(cuda_stream);

    // Set stream on memory buffer for stream-aware deallocation (sink operators don't emit)
    input_tensor->memory_buffer().setStream(cuda_stream);

    viz::Begin(render_flags);
    viz::BeginImageLayer();
    viz::LayerAddView(0.f, static_cast<float>(sub_frame_offset) / static_cast<float>(full_frame_height_.get()),
        1.f, static_cast<float>(tensor_height) / static_cast<float>(full_frame_height_.get()));
    viz::ImageCudaDevice(tensor_width, tensor_height,
        viz::ImageFormat::R16G16B16A16_UNORM, reinterpret_cast<CUdeviceptr>(input_tensor->pointer()));
    viz::EndLayer();
    viz::End();
}

} // namespace hololink::operators
