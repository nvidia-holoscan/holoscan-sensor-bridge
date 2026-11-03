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

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <hololink/core/logging_internal.hpp>
#include <hololink/operators/capture_metadata_utils.hpp>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/sample.hpp>

#include <holoviz/adapter.hpp>

#include "sub_frame_visualizer.slang.hpp" // generated from shader.slang at build time

namespace {

constexpr auto NS_PER_SEC = 1'000'000'000L;
constexpr auto MS_PER_SEC = 1'000L;
constexpr auto NS_PER_MS = NS_PER_SEC / MS_PER_SEC;

static bool less(const struct timespec& time1, const struct timespec& time0)
{
    return time1.tv_sec < time0.tv_sec || (time1.tv_sec == time0.tv_sec && time1.tv_nsec < time0.tv_nsec);
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

/**
 * Nanosecond phase written to FPGA VSYNC_DELAY from absolute capture_time (wall clock).
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

// Must match PushConstants / kFlagGrayscaleReplicate in shader.slang (std430 layout: three
// 4-byte scalars, no padding).
constexpr uint32_t kShaderFlagGrayscaleReplicate = 1U;
struct ShaderPushConstants {
    uint32_t flags;
    // Vertical sampling bounds (texel centres of the rows this draw may read). Linear filtering
    // at a band edge would otherwise pull in the neighbouring row, which may still hold the
    // previous frame when the display is scaled relative to the sensor.
    float uv_min_y;
    float uv_max_y;
};
static_assert(sizeof(ShaderPushConstants) == 12);

[[noreturn]] static void throw_rhi(const holoviz::Error& error, const char* what)
{
    throw std::runtime_error(fmt::format("SubFrameVisualizerOp: {} failed: {}", what, error.message));
}

static const char* present_mode_name(holoviz::PresentMode mode)
{
    switch (mode) {
    case holoviz::PresentMode::kAuto:
        return "auto";
    case holoviz::PresentMode::kFIFO:
        return "fifo";
    case holoviz::PresentMode::kFIFOLatestReady:
        return "fifo-latest-ready";
    case holoviz::PresentMode::kMailbox:
        return "mailbox";
    case holoviz::PresentMode::kImmediate:
        return "immediate";
    case holoviz::PresentMode::kFIFORelaxed:
        return "fifo-relaxed";
    case holoviz::PresentMode::kSharedDemandRefresh:
        return "shared-demand-refresh";
    case holoviz::PresentMode::kSharedContinuousRefresh:
        return "shared-continuous-refresh";
    }
    return "unknown";
}

static bool is_shared_present_mode(holoviz::PresentMode mode)
{
    return mode == holoviz::PresentMode::kSharedDemandRefresh
        || mode == holoviz::PresentMode::kSharedContinuousRefresh;
}

/**
 * Pick the concrete present mode for the swapchain: the requested one when the surface offers it,
 * otherwise the kAuto choice (kFIFOLatestReady if available, else kFIFO) with a warning.
 */
static holoviz::PresentMode resolve_present_mode(holoviz::PresentMode requested,
    const holoviz::PresentModeSet& supported)
{
    const holoviz::PresentMode automatic = supported.contains(holoviz::PresentMode::kFIFOLatestReady)
        ? holoviz::PresentMode::kFIFOLatestReady
        : holoviz::PresentMode::kFIFO;
    if (requested == holoviz::PresentMode::kAuto) {
        return automatic;
    }
    if (supported.contains(requested)) {
        return requested;
    }
    std::string offered;
    for (uint8_t i = 0; i <= static_cast<uint8_t>(holoviz::PresentMode::kSharedContinuousRefresh); ++i) {
        const auto mode = static_cast<holoviz::PresentMode>(i);
        if (supported.contains(mode)) {
            offered += fmt::format(" {}", present_mode_name(mode));
        }
    }
    HSB_LOG_WARN("SubFrameVisualizerOp: present mode '{}' is not supported on this surface "
                 "(offered:{}); falling back to '{}'",
        present_mode_name(requested), offered, present_mode_name(automatic));
    return automatic;
}

/**
 * Select the display for exclusive (direct-to-display) scanout, following HolovizOp's rule: a
 * non-empty name is a case-sensitive substring match in enumeration order; an empty name takes
 * the first display no compositor owns, then one whose ownership could not be classified.
 */
static holoviz::Display resolve_exclusive_display(const std::string& display_name)
{
    auto adapters = holoviz::enumerate_adapters();
    if (!adapters) {
        throw_rhi(adapters.error(), "enumerate_adapters");
    }
    std::vector<holoviz::Display> candidates;
    for (const auto& adapter : *adapters) {
        auto displays = holoviz::enumerate_displays(adapter);
        if (!displays) {
            HSB_LOG_WARN("SubFrameVisualizerOp: display enumeration on adapter '{}' failed: {}",
                adapter.name.data(), displays.error().message);
            continue;
        }
        candidates.insert(candidates.end(), displays->begin(), displays->end());
    }
    const auto listing = [&candidates] {
        std::string text;
        for (const auto& display : candidates) {
            const char* ownership = "unknown";
            if (display.ownership == holoviz::DisplayOwnership::kAvailable) {
                ownership = "available";
            } else if (display.ownership == holoviz::DisplayOwnership::kCompositor) {
                ownership = "compositor-owned";
            }
            text += fmt::format(" \"{}\" ({}x{}@{}mHz, {});", display.name.data(),
                display.current_mode.resolution.width, display.current_mode.resolution.height,
                display.current_mode.refresh_rate_mhz, ownership);
        }
        return text.empty() ? std::string(" none") : text;
    };

    if (!display_name.empty()) {
        for (const auto& display : candidates) {
            if (std::string(display.name.data()).find(display_name) != std::string::npos) {
                return display;
            }
        }
        throw std::runtime_error(fmt::format(
            "SubFrameVisualizerOp: no display matches \"{}\"; connected displays:{}",
            display_name, listing()));
    }
    const holoviz::Display* unclassified = nullptr;
    for (const auto& display : candidates) {
        if (display.ownership == holoviz::DisplayOwnership::kAvailable) {
            return display;
        }
        if (display.ownership == holoviz::DisplayOwnership::kUnknown && unclassified == nullptr) {
            unclassified = &display;
        }
    }
    if (unclassified != nullptr) {
        HSB_LOG_WARN("SubFrameVisualizerOp: ownership of display \"{}\" could not be classified; "
                     "attempting exclusive acquisition anyway (name a display explicitly to override)",
            unclassified->name.data());
        return *unclassified;
    }
    throw std::runtime_error(fmt::format(
        "SubFrameVisualizerOp: no display is free for exclusive scanout (stop the desktop session "
        "driving the GPU or name a display explicitly); connected displays:{}",
        listing()));
}

static const char* format_name(holoviz::Format format)
{
    switch (format) {
    case holoviz::Format::kB8G8R8A8Srgb:
        return "B8G8R8A8Srgb";
    case holoviz::Format::kR8G8B8A8Srgb:
        return "R8G8B8A8Srgb";
    case holoviz::Format::kB8G8R8A8Unorm:
        return "B8G8R8A8Unorm";
    case holoviz::Format::kR8G8B8A8Unorm:
        return "R8G8B8A8Unorm";
    default:
        return "other";
    }
}

/**
 * Swapchain format / sRGB-encoding choice. The bands are linear sensor data, so the framebuffer
 * must encode to sRGB on write (as HSDK4's SubFrameVisualizerOp did by picking an *_SRGB surface
 * format): prefer a native sRGB scanout format, else an sRGB attachment view over UNORM scanout
 * when the device supports mutable swapchain formats, else fall back to UNORM (linear values
 * shown as if encoded: visibly dark) with a warning.
 */
struct SwapchainFormatChoice {
    holoviz::Format format { holoviz::Format::kB8G8R8A8Unorm };
    bool srgb_attachment { false };
    bool srgb_encoded { false };
};

static SwapchainFormatChoice choose_swapchain_format(const holoviz::Device& device,
    const holoviz::Surface& surface)
{
    SwapchainFormatChoice choice {};
    auto formats_r = device.surface_formats(surface);
    if (!formats_r) {
        HSB_LOG_WARN("SubFrameVisualizerOp: surface_formats failed ({}); using the default UNORM "
                     "swapchain, output will look dark",
            formats_r.error().message);
        return choice;
    }
    const auto offers = [&](holoviz::Format f) {
        for (const auto& pair : *formats_r) {
            if (pair.format == f && pair.color_space == holoviz::ColorSpace::kSrgbNonlinear) {
                return true;
            }
        }
        return false;
    };
    for (const auto f : { holoviz::Format::kB8G8R8A8Srgb, holoviz::Format::kR8G8B8A8Srgb }) {
        if (offers(f)) {
            choice.format = f;
            choice.srgb_encoded = true;
            return choice;
        }
    }
    const bool mutable_format = device.capabilities()
                                    .transform([](const holoviz::CapabilityReport& c) { return c.mutable_swapchain_format; })
                                    .value_or(false);
    for (const auto f : { holoviz::Format::kB8G8R8A8Unorm, holoviz::Format::kR8G8B8A8Unorm }) {
        if (offers(f)) {
            choice.format = f;
            if (mutable_format) {
                choice.srgb_attachment = true;
                choice.srgb_encoded = true;
            }
            break;
        }
    }
    if (!choice.srgb_encoded) {
        HSB_LOG_WARN("SubFrameVisualizerOp: the surface offers no sRGB framebuffer (no *Srgb format "
                     "and no mutable swapchain format); linear sensor data will be shown "
                     "without sRGB encoding and look dark");
    }
    return choice;
}

static holoviz::Pipeline make_pipeline(holoviz::Device& device)
{
    // Compiled to SPIR-V at build time from shader.slang (see CMakeLists.txt).
    const holoviz::ShaderBlob blob = hololink::operators::sub_frame_visualizer_shader::make_shader_blob();
    auto pipeline = device.create_pipeline({
        .shader_blob = blob,
        .topology = holoviz::PrimitiveTopology::kTriangleList,
        .raster = { .cull = holoviz::CullMode::kNone },
        .depth = { .test = false, .write = false, .compare = holoviz::CompareOp::kAlways },
    });
    if (!pipeline) {
        throw_rhi(pipeline.error(), "create_pipeline");
    }
    return std::move(*pipeline);
}

} // namespace

namespace hololink::operators {

SubFrameVisualizerOp::SubFrameVisualizerOp(uint32_t full_frame_height,
    bool fullscreen,
    bool use_exclusive_display,
    std::string display_name,
    uint32_t display_width,
    uint32_t display_height,
    uint32_t display_framerate,
    std::string window_title,
    std::shared_ptr<Hololink::PtpSynchronizer> ptp_synchronizer,
    holoviz::PresentMode present_mode)
    : full_frame_height_(full_frame_height)
    , fullscreen_(fullscreen)
    , use_exclusive_display_(use_exclusive_display)
    , display_name_(std::move(display_name))
    , display_width_(display_width)
    , display_height_(display_height)
    , display_framerate_(display_framerate)
    , window_title_(std::move(window_title))
    , ptp_synchronizer_(std::move(ptp_synchronizer))
    , present_mode_(present_mode)
{
    if (display_width_ == 0 || display_height_ == 0 || full_frame_height_ == 0) {
        throw std::invalid_argument("SubFrameVisualizerOp: dimensions must be > 0");
    }
}

void SubFrameVisualizerOp::thread_func()
{
    constexpr struct timespec MIN_CAPTURE_DELAY_FROM_NOW = { 0, 2 * NS_PER_MS };
    constexpr struct timespec PROCESSING_MARGIN = { 0, 2 * NS_PER_MS };

    struct timespec capture_time = { 0, 0 };
    struct timespec last_fpo_render_start_time = { 0, 0 };
    struct timespec cached_capture_to_render_delay = { 0, 0 };

    double framerate_hz = display_framerate_ > 0 ? (display_framerate_ / 1000.0) : 59.95;
    struct timespec display_period = { 0, static_cast<long>(NS_PER_SEC / framerate_hz) };
    HSB_LOG_INFO("Initial display refresh rate: {:.2f} Hz (will be measured from FPO events)", framerate_hz);

    struct timespec previous_fpo_time = { 0, 0 };
    uint32_t fpo_sample_count = 0;
    static constexpr uint32_t FPO_SAMPLES_NEEDED = 10;

    static constexpr struct timespec kPtpPpsModulus = { 1, 0 };
    static constexpr bool kPtpSinglePulsePpsFpgaDelay = true;

    if (!display_for_fpo_) {
        // start() could not resolve the display behind the window (e.g. a windowed Wayland
        // session): there is nothing to pace on, so the capture schedule stays untouched.
        HSB_LOG_WARN("SubFrameVisualizerOp: no display for first-pixel-out events; PTP capture "
                     "scheduling disabled");
        fpo_available_.store(false);
        return;
    }
    while (!stop_requested_) {
        auto r = device_.wait_for_display_event(*display_for_fpo_,
            holoviz::DisplayEventType::kFirstPixelOut, std::chrono::nanoseconds(NS_PER_SEC));
        const bool is_ready = static_cast<bool>(r);
        fpo_available_.store(is_ready);
        if (!is_ready) {
            if (r.error().code == holoviz::ErrorCode::kNotImplemented) {
                // Permanent: the driver rejects display events on this GPU (e.g. AGX Thor,
                // bug 6194895) although it advertises the extension. Say so once and stop
                // instead of polling for the operator's lifetime.
                HSB_LOG_WARN("SubFrameVisualizerOp: display events unavailable on this GPU ({}); "
                             "PTP capture scheduling disabled",
                    r.error().message);
                return;
            }
            // Transient (timeout or a failed registration): wait_for_display_event() returns
            // without blocking on failure, so back off instead of spinning a core.
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }

        struct timespec now;
        if (clock_gettime(CLOCK_REALTIME, &now) != 0) {
            throw std::runtime_error(fmt::format("clock_gettime failed: {}", errno));
        }

        if ((previous_fpo_time.tv_sec != 0 || previous_fpo_time.tv_nsec != 0) && fpo_sample_count < FPO_SAMPLES_NEEDED) {
            struct timespec measured_period = sub(now, previous_fpo_time);

            if (fpo_sample_count == 0) {
                display_period = measured_period;
            } else {
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
                HSB_LOG_INFO("Measured display refresh rate from {} FPO events: {:.2f} Hz",
                    FPO_SAMPLES_NEEDED, measured_fps);
            }
        }
        previous_fpo_time = now;

        struct timespec render_start_time;
        {
            std::lock_guard<std::mutex> lock(render_start_time_mutex_);
            render_start_time = render_start_time_;
        }

        uint32_t delay_ns_for_fpga = 0;

        if ((render_start_time.tv_sec == 0) && (render_start_time.tv_nsec == 0)) {
            capture_time = add(now, MIN_CAPTURE_DELAY_FROM_NOW);
            delay_ns_for_fpga = fpga_vsync_delay_ns(capture_time, display_period, kPtpSinglePulsePpsFpgaDelay);
        } else {
            if ((capture_time.tv_sec == 0) && (capture_time.tv_nsec == 0)) {
                HSB_LOG_WARN(
                    "capture_time unset while render_start is valid; initializing schedule (was using epoch in delay math)");
                capture_time = add(now, MIN_CAPTURE_DELAY_FROM_NOW);
                delay_ns_for_fpga = fpga_vsync_delay_ns(capture_time, display_period, kPtpSinglePulsePpsFpgaDelay);
            } else {
                if (render_start_time.tv_sec == last_fpo_render_start_time.tv_sec
                    && render_start_time.tv_nsec == last_fpo_render_start_time.tv_nsec
                    && (render_start_time.tv_sec != 0 || render_start_time.tv_nsec != 0)) {
                    HSB_LOG_WARN("No new frame since last FPO -- camera rate may be less than display rate");
                }
                last_fpo_render_start_time = render_start_time;

                struct timespec capture_to_render_delay;
                if (less(render_start_time, capture_time)) {
                    capture_to_render_delay = cached_capture_to_render_delay;
                } else {
                    cached_capture_to_render_delay = sub(render_start_time, capture_time);
                    capture_to_render_delay = cached_capture_to_render_delay;
                }
                struct timespec capture_to_vblank_delay = add(capture_to_render_delay, PROCESSING_MARGIN);

                capture_time = add(now, display_period);
                struct timespec capture_to_vblank_phase = timespec_mod_period_ns(capture_to_vblank_delay, kPtpPpsModulus);
                capture_time = sub(capture_time, capture_to_vblank_phase);

                capture_time = max(capture_time, add(now, MIN_CAPTURE_DELAY_FROM_NOW));
                delay_ns_for_fpga = fpga_vsync_delay_ns(capture_time, display_period, kPtpSinglePulsePpsFpgaDelay);
            }
        }
        HSB_LOG_DEBUG("vsync delay(ms) for fpga = {:.3f}", delay_ns_for_fpga / 1000000.0);
        ptp_synchronizer_->set_delay(delay_ns_for_fpga);
    }
}

void SubFrameVisualizerOp::setup(holoscan::OperatorSpec& spec)
{
    (void)spec.input(input_tensor, "tensor")
        .tensor_field(holoscan::schema::ImageTensorFields::data);
    spec.input(input_metadata, "metadata");
}

holoscan::Contract SubFrameVisualizerOp::contract() const
{
    // Pair on exactly equal capture_time (see kMetadataPairingWindow). The band and its metadata
    // may reach this operator by different routes -- the band through a demosaic, the metadata
    // straight from ImageProcessorOp -- and each route sheds independently when the display
    // stalls; a head whose partner was shed is retired instead of being mispaired.
    holoscan::Contract result;
    result
        .trigger(holoscan::OnAligned {
            .ports = holoscan::ports(input_tensor, input_metadata),
            .window = hololink::operators::kMetadataPairingWindow,
        })
        .input(input_tensor, holoscan::Exact {})
        .input(input_metadata, holoscan::Exact {});
    return result;
}

void SubFrameVisualizerOp::start()
{
#ifndef _WIN32
    if (!use_exclusive_display_ && fullscreen_ && !display_name_.empty()) {
        // Ask the driver to sync to the monitor we render on (G-Sync / vblank source).
        constexpr const char* kEnv = "__GL_SYNC_DISPLAY_DEVICE";
        if (std::getenv(kEnv) == nullptr) {
            setenv(kEnv, display_name_.c_str(), 0);
            HSB_LOG_INFO("SubFrameVisualizerOp: set {}={}", kEnv, display_name_);
        }
    }
#endif

    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&cuda_device_, 0) != CUDA_SUCCESS) {
        throw std::runtime_error("SubFrameVisualizerOp: CUDA init failed");
    }
    if (cuDevicePrimaryCtxRetain(&cuda_context_, cuda_device_) != CUDA_SUCCESS) {
        throw std::runtime_error("SubFrameVisualizerOp: cuDevicePrimaryCtxRetain failed");
    }
    cuda_primary_ctx_retained_ = true; // released in stop(), also on a failed start()
    if (cuCtxSetCurrent(cuda_context_) != CUDA_SUCCESS) {
        stop();
        throw std::runtime_error("SubFrameVisualizerOp: cuCtxSetCurrent failed");
    }
    // Everything below creates GPU objects that stop() knows how to release; unwind through it
    // so a failed start() leaves neither the primary-context reference nor a half-built RHI.
    try {
        start_gpu();
    } catch (...) {
        stop();
        throw;
    }
}

void SubFrameVisualizerOp::start_gpu()
{
    if (cuStreamCreate(&cuda_stream_, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) {
        throw std::runtime_error("SubFrameVisualizerOp: cuStreamCreate failed");
    }
    if (cuEventCreate(&producer_ready_event_, CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS
        || cuEventCreate(&copy_done_event_, CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS) {
        throw std::runtime_error("SubFrameVisualizerOp: cuEventCreate failed");
    }
    frames_with_missing_bands_ = 0;
    missing_bands_union_ = 0;
    max_acquire_wait_ = std::chrono::microseconds { 0 };
    missing_bands_report_time_ = std::chrono::steady_clock::now();

    if (use_exclusive_display_) {
        // Direct-to-display: acquire the DRM connector and scan out without a compositor.
        // The surface is fixed-size (the chosen mode); the display is released when the
        // surface is destroyed in stop().
        holoviz::Display display = resolve_exclusive_display(display_name_);
        auto surface_r = display.create_surface({
            .mode = {
                .resolution = { display_width_, display_height_ },
                .refresh_rate_mhz = display_framerate_,
            },
        });
        if (!surface_r) {
            throw_rhi(surface_r.error(), "Display::create_surface");
        }
        surface_ = std::move(*surface_r);
        const holoviz::Extent2D extent = surface_.extent().value_or(holoviz::Extent2D { display_width_, display_height_ });
        HSB_LOG_INFO("SubFrameVisualizerOp: exclusive display \"{}\" acquired at {}x{} "
                     "(requested {}x{}@{}mHz)",
            display.name.data(), extent.width, extent.height, display_width_, display_height_,
            display_framerate_);
        display_for_fpo_ = std::move(display);
    } else {
        auto win_r = holoviz::create_window({
            .title = window_title_,
            .size = { display_width_, display_height_ },
            .resizable = !fullscreen_,
            .fullscreen = fullscreen_,
            .display_name = display_name_,
            .close_on_escape = fullscreen_,
        });
        if (!win_r) {
            throw_rhi(win_r.error(), "create_window");
        }
        window_ = std::move(*win_r);

        auto surface_r = window_.create_surface();
        if (!surface_r) {
            throw_rhi(surface_r.error(), "Window::create_surface");
        }
        surface_ = std::move(*surface_r);
    }

    auto dev_r = holoviz::create_device_for_surface(
        { .cuda_device = cuda_device_, .cuda_context = cuda_context_ }, surface_);
    if (!dev_r) {
        throw_rhi(dev_r.error(), "create_device_for_surface");
    }
    device_ = std::move(*dev_r);
    HSB_LOG_INFO("SubFrameVisualizerOp: device {}", device_.adapter().value().name.data());

    if (!use_exclusive_display_) {
        if (auto d = holoviz::display_for_window(window_, device_.adapter().value())) {
            display_for_fpo_ = std::move(*d);
        } else {
            HSB_LOG_WARN("SubFrameVisualizerOp: display for FPO events not found ({}); "
                         "capture scheduling disabled",
                d.error().message);
        }
    }

    // Present mode: honor the request when the surface offers it, else the best tear-free mode.
    auto caps_r = device_.surface_capabilities(surface_);
    if (!caps_r) {
        throw_rhi(caps_r.error(), "surface_capabilities");
    }
    const holoviz::PresentMode present_mode = resolve_present_mode(present_mode_, caps_r->present_modes);
    shared_mode_ = is_shared_present_mode(present_mode);
    continuous_mode_ = present_mode == holoviz::PresentMode::kSharedContinuousRefresh;
    presentation_started_ = false;

    const SwapchainFormatChoice sc_format_choice = choose_swapchain_format(device_, surface_);
    auto sc_r = device_.create_swapchain({
        .surface = surface_,
        .format = sc_format_choice.format,
        .color_space = holoviz::ColorSpace::kSrgbNonlinear,
        .present_mode = present_mode,
        // Shared modes use the single shared image. Otherwise triple-buffer: with only two
        // images acquire() blocks until the previous frame flips (up to a full refresh) while
        // the next frame's bands keep arriving, and the backlog costs bands under backpressure.
        .image_count = shared_mode_ ? 1U : 3U,
        .srgb_attachment = sc_format_choice.srgb_attachment,
    });
    if (!sc_r) {
        throw_rhi(sc_r.error(), "create_swapchain");
    }
    swapchain_ = std::move(*sc_r);

    const uint32_t image_count = swapchain_.image_count().value();
    const holoviz::Format sc_format = swapchain_.format().value();
    swapchain_generation_ = swapchain_.generation().value();
    command_lists_.resize(image_count);
    HSB_LOG_INFO("SubFrameVisualizerOp: {} sink, present mode '{}', {} swapchain image(s), "
                 "format {}{}",
        use_exclusive_display_ ? "exclusive-display" : "window",
        present_mode_name(swapchain_.present_mode().value_or(present_mode)), image_count,
        format_name(sc_format), sc_format_choice.srgb_attachment ? " (sRGB attachment view)" : "");

    // Full-clear pass for each swapchain image's first use: it moves the image into its
    // presentation home layout, which the kLoad band pass loads from.
    auto init_pass_r = device_.create_render_pass({
        .render_targets = { { .format = sc_format, .shared_present = shared_mode_ } },
    });
    if (!init_pass_r) {
        throw_rhi(init_pass_r.error(), "create_render_pass(init)");
    }
    init_pass_ = std::move(*init_pass_r);
    // The band pass keeps the presentable image's existing contents (HSDK4's
    // DONT_CLEAR_COLOR), so each pass adds one band and the bands accumulate. In shared modes
    // it renders in place in the shared-present layout the presentation engine reads from.
    auto band_pass_r = device_.create_render_pass({
        .render_targets = { { .load = holoviz::LoadOp::kLoad, .format = sc_format, .shared_present = shared_mode_ } },
    });
    if (!band_pass_r) {
        throw_rhi(band_pass_r.error(), "create_render_pass(band)");
    }
    band_pass_ = std::move(*band_pass_r);

    pipeline_ = make_pipeline(device_);
    // frame_image_ and descriptor_ are created on the first band (ensure_frame_image), once the
    // tensor's channel count is known.

    auto sem_r = device_.create_external_semaphore(cuda_stream_);
    if (!sem_r) {
        throw_rhi(sem_r.error(), "create_external_semaphore");
    }
    external_semaphore_ = std::move(*sem_r);

    if (ptp_synchronizer_) {
        stop_requested_ = false;
        thread_ = std::thread(&SubFrameVisualizerOp::thread_func, this);
    }
}

void SubFrameVisualizerOp::ensure_frame_image(uint32_t channels)
{
    if (channels_ == channels) {
        return;
    }
    if (channels_ != 0) {
        throw std::runtime_error(fmt::format(
            "SubFrameVisualizerOp: band channel count changed from {} to {}", channels_, channels));
    }
    holoviz::Format format {};
    switch (channels) {
    case 1:
        format = holoviz::Format::kR16Unorm;
        break;
    case 4:
        format = holoviz::Format::kR16G16B16A16Unorm;
        break;
    default:
        throw std::runtime_error(fmt::format(
            "SubFrameVisualizerOp: unsupported channel count {} (expected 1 or 4)", channels));
    }

    // Exported image in the upstream tensor format; one band is copied into it per compute().
    auto img_r = device_.create_image({
        .extent = { display_width_, full_frame_height_ },
        .format = format,
        .usage = holoviz::ImageUsage::kSampled,
        .share = holoviz::ShareMode::kExported,
    });
    if (!img_r) {
        throw_rhi(img_r.error(), "create_image");
    }
    frame_image_ = std::move(*img_r);

    auto arr_r = frame_image_.cuda_mipmapped_array();
    if (!arr_r) {
        throw_rhi(arr_r.error(), "cuda_mipmapped_array");
    }
    if (cuMipmappedArrayGetLevel(&frame_image_level0_, *arr_r, 0) != CUDA_SUCCESS) {
        throw std::runtime_error("SubFrameVisualizerOp: cuMipmappedArrayGetLevel failed");
    }

    auto desc_r = pipeline_.create_descriptor();
    if (!desc_r) {
        throw_rhi(desc_r.error(), "create_descriptor");
    }
    descriptor_ = std::move(*desc_r);
    if (auto r = descriptor_.bind_image(frame_image_, "frameTexture"); !r) {
        throw_rhi(r.error(), "bind_image");
    }
    if (auto r = descriptor_.bind_sampler(device_, {}, "frameSampler"); !r) {
        throw_rhi(r.error(), "bind_sampler");
    }
    channels_ = channels;
    HSB_LOG_INFO("SubFrameVisualizerOp: frame image {}x{} with {} uint16 channel(s)",
        display_width_, full_frame_height_, channels);
}

void SubFrameVisualizerOp::stop()
{
    stop_requested_ = true;
    if (thread_.joinable()) {
        thread_.join();
    }

    if (device_) {
        (void)device_.wait_idle();
    }

    if (cuda_stream_) {
        cuStreamSynchronize(cuda_stream_);
        cuStreamDestroy(cuda_stream_);
        cuda_stream_ = {};
    }
    if (producer_ready_event_) {
        cuEventDestroy(producer_ready_event_);
        producer_ready_event_ = {};
    }
    if (copy_done_event_) {
        cuEventDestroy(copy_done_event_);
        copy_done_event_ = {};
    }

    pending_frame_.reset();
    last_frame_token_ = {};
    swapchain_generation_ = 0;
    command_lists_.clear();
    frame_image_level0_ = {};
    channels_ = 0;
    render_start_time_ = { 0, 0 };
    presentation_started_ = false;

    // Release every object created from device_ before device_ itself (Device::destroy()
    // requires its children gone first), so a framework-driven restart recreates everything
    // from a clean slate. Reverse creation order: swapchain before surface (the surface must
    // outlive it), surface before window; destroying an exclusive-display surface also
    // releases the display for the next user.
    descriptor_ = holoviz::Descriptor {};
    frame_image_ = holoviz::Image {};
    external_semaphore_ = holoviz::ExternalSemaphore {};
    pipeline_ = holoviz::Pipeline {};
    band_pass_ = holoviz::RenderPass {};
    init_pass_ = holoviz::RenderPass {};
    swapchain_ = holoviz::Swapchain {};
    display_for_fpo_.reset();
    device_ = holoviz::Device {};
    surface_ = holoviz::Surface {};
    window_ = holoviz::Window {};

    // Last: the primary-context reference taken in start(), after every CUDA object is gone.
    if (cuda_primary_ctx_retained_) {
        cuDevicePrimaryCtxRelease(cuda_device_);
        cuda_primary_ctx_retained_ = false;
        cuda_context_ = {};
    }
}

void SubFrameVisualizerOp::abandon_pending_frame(const char* why, bool present_partial)
{
    if (!pending_frame_) {
        return;
    }
    PendingFrame frame = std::move(*pending_frame_);
    pending_frame_.reset();
    // Whatever happens below, the next first band must wait on this frame's last submit before
    // overwriting frame_image_.
    last_frame_token_ = frame.previous_band_token;
    if (!frame.acquire) {
        return; // continuous shared mode: nothing outstanding in the swapchain
    }

    HSB_LOG_WARN("SubFrameVisualizerOp: releasing the in-flight frame early ({})", why);
    if (present_partial) {
        // Preferred release: present what has been drawn so far. Both gates unconsumed (no band
        // was submitted) may present directly; with the acquire gate consumed the present gate
        // must be armed first, which signal() does from the CPU with an empty submit.
        if (frame.use_gates && frame.acquire->acquire_gate.consumed().value_or(false)
            && !frame.acquire->present_gate.consumed().value_or(false)) {
            if (auto r = frame.acquire->present_gate.signal(); !r) {
                HSB_LOG_WARN("SubFrameVisualizerOp: present_gate.signal() failed: {}",
                    r.error().message);
            }
        }
        auto present_r = swapchain_.present(std::move(*frame.acquire), last_frame_token_);
        if (present_r) {
            presentation_started_ = true;
            return;
        }
        if (present_r.error().code == holoviz::ErrorCode::kFrameDropped) {
            return; // consumed by a recreation; nothing outstanding
        }
        HSB_LOG_WARN("SubFrameVisualizerOp: early present failed ({}); recreating the swapchain",
            present_r.error().message);
    } else {
        // Nothing of this frame reached the GPU (the first band's submit failed): presenting the
        // acquired image would show whatever it held last. Drop it through a recreation instead.
    }

    // Last resort: resize() with the surface's own extent is the documented way to drop an
    // acquired frame without presenting it; it also invalidates the per-image bookkeeping.
    if (auto r = swapchain_.resize(); !r) {
        HSB_LOG_ERROR("SubFrameVisualizerOp: swapchain resize after abandoned frame failed: {}",
            r.error().message);
        return;
    }
    const auto n = static_cast<size_t>(swapchain_.image_count().value_or(0));
    swapchain_generation_ = swapchain_.generation().value_or(swapchain_generation_);
    command_lists_.clear();
    command_lists_.resize(n);
    last_frame_token_ = {};
    presentation_started_ = false;
}

holoscan::expected<void, holoscan::Error> SubFrameVisualizerOp::compute(holoscan::ExecutionContext& context)
{
    if (window_) {
        if (auto closed = window_.close_requested(); !closed || *closed) {
            context.request_stop();
            return {};
        }
        holoviz::poll_window_events();
    }
    if (context.cancellation().stop_requested()) {
        return {};
    }

    auto maybe_paired = hololink::operators::receive_paired_tensor_metadata(input_tensor, input_metadata);
    if (!maybe_paired) {
        return holoscan::make_unexpected(std::move(maybe_paired).error());
    }
    if (!*maybe_paired) {
        if (const uint64_t dropped = mismatched_frames_.count()) {
            HSB_LOG_WARN("SubFrameVisualizerOp: dropped {} frame(s) in the last second whose payload and metadata "
                         "were stamped as different frames",
                dropped);
        }
        return {};
    }
    auto& paired = **maybe_paired;
    const holoscan::schema::ImageT& in_image = paired.tensor.data;
    if (!in_image.data) {
        return holoscan::make_unexpected(holoscan::Error { holoscan::ErrorCode::kInvalidArgument,
            "Image carries no tensor" });
    }
    holoscan::Tensor& in_tensor = *in_image.data;
    const hololink::CaptureMetadataT& in_frame = paired.metadata.data;

    if (in_tensor.shape_span().size() != 3) {
        return holoscan::make_unexpected(holoscan::Error { holoscan::ErrorCode::kInvalidArgument,
            "Tensor must be rank 3" });
    }

    const DLDataType dtype = in_tensor.dtype();
    if (dtype.code != kDLUInt || dtype.bits != 16) {
        return holoscan::make_unexpected(holoscan::Error { holoscan::ErrorCode::kInvalidArgument,
            "Tensor must be unsigned 16-bit" });
    }

    const DLDeviceType device_type = in_tensor.device().device_type;
    if (device_type != kDLCUDA) {
        return holoscan::make_unexpected(holoscan::Error { holoscan::ErrorCode::kInvalidArgument,
            "Tensor must be stored on the device" });
    }

    const int64_t tensor_height = in_tensor.shape_span()[0];
    const int64_t tensor_width = in_tensor.shape_span()[1];

    const int64_t tensor_channels = in_tensor.shape_span()[2];
    if (tensor_channels != 1 && tensor_channels != 4) {
        return holoscan::make_unexpected(holoscan::Error { holoscan::ErrorCode::kInvalidArgument,
            "Tensor must have 1 or 4 elements per pixel" });
    }
    // The pixels of a row must be packed; the rows themselves may be padded (the band copy below
    // uses the row stride as its pitch).
    const std::vector<int64_t> strides = in_tensor.strides();
    const int64_t pixel_bytes = int64_t(sizeof(uint16_t)) * tensor_channels;
    if (strides.size() != 3 || strides[2] != int64_t(sizeof(uint16_t)) || strides[1] != pixel_bytes
        || strides[0] < tensor_width * pixel_bytes) {
        return holoscan::make_unexpected(holoscan::Error { holoscan::ErrorCode::kInvalidArgument,
            fmt::format("Unsupported image layout: byte strides [{}]", fmt::join(strides, ", ")) });
    }

    const int64_t sub_frame_offset_val = in_frame.sub_frame_offset;
    if (sub_frame_offset_val < 0 || sub_frame_offset_val + tensor_height > full_frame_height_) {
        HSB_LOG_ERROR("SubFrameVisualizerOp: band [{}, {}) outside the {}-row frame; dropping",
            sub_frame_offset_val, sub_frame_offset_val + tensor_height, full_frame_height_);
        return {};
    }
    if (tensor_width != display_width_) {
        HSB_LOG_ERROR("SubFrameVisualizerOp: band width {} != display_width {}; dropping",
            tensor_width, display_width_);
        return {};
    }

    const uint32_t y0 = static_cast<uint32_t>(sub_frame_offset_val);
    const uint32_t band_height = static_cast<uint32_t>(tensor_height);
    const bool is_first_band = (y0 == 0);
    const bool is_last_band = (y0 + band_height >= full_frame_height_);

    // Record the host arrival time of the first sub frame. The FPO controller subtracts the
    // scheduled capture time from this to estimate capture-to-render latency, so it has to be
    // measured here, after transport and processing; the sensor's own frame-start timestamp
    // (in_frame.timestamp_*) would hide that whole interval from the feedback loop.
    if (is_first_band) {
        struct timespec now = { 0, 0 };
        if (clock_gettime(CLOCK_REALTIME, &now) != 0) {
            return holoscan::make_unexpected(holoscan::Error { holoscan::ErrorCode::kFailure,
                fmt::format("clock_gettime failed: {}", errno) });
        }
        std::lock_guard<std::mutex> lock(render_start_time_mutex_);
        render_start_time_ = now;
    }

    // First band of a frame: wait for the previous frame's GPU work (it reads the exported image
    // we are about to overwrite), then acquire a swapchain image.
    if (is_first_band) {
        if (pending_frame_) {
            abandon_pending_frame("a new frame started before the previous one's last band arrived");
        }
        if (auto r = external_semaphore_.wait(last_frame_token_); !r) {
            HSB_LOG_ERROR("SubFrameVisualizerOp: ExternalSemaphore::wait failed: {}", r.error().message);
            return {};
        }

        if (const uint64_t gen = swapchain_.generation().value_or(swapchain_generation_);
            gen != swapchain_generation_) {
            swapchain_generation_ = gen;
        }

        if (continuous_mode_ && presentation_started_) {
            // Continuous shared refresh: the presentation engine reads the single shared image
            // on its own once started, so there is no per-frame acquire/present. Bands reach
            // the display as their submits execute.
            pending_frame_ = PendingFrame {
                .acquire = std::nullopt,
                .previous_band_token = last_frame_token_,
                .band_idx = 0,
                .use_gates = false,
            };
        } else {
            const auto acquire_begin = std::chrono::steady_clock::now();
            auto acq_r = swapchain_.acquire(std::chrono::milliseconds(100));
            max_acquire_wait_ = std::max(max_acquire_wait_,
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - acquire_begin));
            if (!acq_r) {
                const auto code = acq_r.error().code;
                if (code == holoviz::ErrorCode::kBackpressured || code == holoviz::ErrorCode::kOutOfDate) {
                    return {}; // transient; drop this frame
                }
                if (code == holoviz::ErrorCode::kImageCountChanged) {
                    // A recreation changed image_count(): rebuild the per-image bookkeeping and
                    // pick the frame up on the next first band.
                    const auto n = static_cast<size_t>(swapchain_.image_count().value_or(0));
                    command_lists_.clear();
                    command_lists_.resize(n);
                    return {};
                }
                HSB_LOG_ERROR("SubFrameVisualizerOp: acquire failed: {}", acq_r.error().message);
                return {};
            }
            // Shared-mode acquires after the first mint pre-consumed gates (there is no
            // presentation-engine handshake to synchronize with); they must not go to submit().
            const bool use_gates = !acq_r->acquire_gate.consumed().value_or(false);
            pending_frame_ = PendingFrame {
                .acquire = std::move(*acq_r),
                .previous_band_token = last_frame_token_,
                .band_idx = 0,
                .use_gates = use_gates,
            };
        }
    }

    if (!pending_frame_) {
        // No frame in flight (acquire failed or bands arrived before the first one); wait for the
        // next frame start.
        return {};
    }

    // Shared swapchains have exactly one image, so index 0 is the shared image when no acquire
    // token is outstanding (continuous mode after the first present).
    const uint32_t image_index = pending_frame_->acquire ? pending_frame_->acquire->image_index : 0;

    ensure_frame_image(static_cast<uint32_t>(tensor_channels));

    // The previous band's draw may still be sampling frame_image_ (the first band's full-frame
    // pass reads every row, and filtering reads a neighbour row at each band edge). Order this
    // band's upload after it on the CUDA stream before touching the image. The first band
    // already waited on the previous frame's last token above.
    if (pending_frame_->band_idx != 0) {
        if (auto r = external_semaphore_.wait(pending_frame_->previous_band_token); !r) {
            HSB_LOG_ERROR("SubFrameVisualizerOp: image reuse wait failed: {}", r.error().message);
            return {};
        }
    }

    // Producer: copy this band from the device tensor into the exported image. The copy runs on
    // the operator's own stream (the external semaphore is bound to it), so it must be ordered
    // explicitly against the runtime's activation stream: the runtime queues the producer's
    // readiness there, and releases the input tensor by that stream's progress. Without this the
    // copy could read the band before ImageProcessorOp finished writing it, or after its pool
    // slot was recycled for a newer band -- a stale band and frame-to-frame flicker.
    const CUstream activation_stream = context.cuda_stream().get();
    if (cuEventRecord(producer_ready_event_, activation_stream) != CUDA_SUCCESS
        || cuStreamWaitEvent(cuda_stream_, producer_ready_event_, 0) != CUDA_SUCCESS) {
        HSB_LOG_ERROR("SubFrameVisualizerOp: failed to order the band copy after the producer");
        return {};
    }
    const auto src_ptr = static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(in_tensor.data()));
    CUDA_MEMCPY2D copy {};
    copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.srcDevice = src_ptr;
    copy.srcPitch = static_cast<size_t>(strides[0]);
    copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.dstArray = frame_image_level0_;
    copy.dstY = y0;
    copy.WidthInBytes = static_cast<size_t>(tensor_width * pixel_bytes);
    copy.Height = band_height;
    if (cuMemcpy2DAsync(&copy, cuda_stream_) != CUDA_SUCCESS) {
        HSB_LOG_ERROR("SubFrameVisualizerOp: cuMemcpy2DAsync failed at y={}", y0);
        return {};
    }
    if (cuEventRecord(copy_done_event_, cuda_stream_) != CUDA_SUCCESS
        || cuStreamWaitEvent(activation_stream, copy_done_event_, 0) != CUDA_SUCCESS) {
        HSB_LOG_ERROR("SubFrameVisualizerOp: failed to order the input release after the band copy");
        return {};
    }
    // Stream-aware pools additionally defer slot reuse until cuda_stream_ has passed the copy.
    (void)in_tensor.set_deallocation_stream(cuda_stream_);
    auto ext_token_r = external_semaphore_.signal();
    if (!ext_token_r) {
        HSB_LOG_ERROR("SubFrameVisualizerOp: ExternalSemaphore::signal failed: {}", ext_token_r.error().message);
        return {};
    }

    // Consumer: record and submit the draw for this band.
    auto& lists = command_lists_[image_index];
    const uint32_t band_idx = pending_frame_->band_idx;
    if (band_idx >= lists.size()) {
        auto cl_r = device_.create_command_list({ .type = holoviz::CommandListType::kTransient });
        if (!cl_r) {
            HSB_LOG_ERROR("SubFrameVisualizerOp: create_command_list failed: {}", cl_r.error().message);
            return {};
        }
        lists.push_back(std::move(*cl_r));
    }
    auto& cl = lists[band_idx];

    if (auto r = cl.reset(std::chrono::milliseconds(100)); !r) {
        HSB_LOG_ERROR("SubFrameVisualizerOp: command list reset failed: {}", r.error().message);
        return {};
    }
    (void)cl.begin();

    const holoviz::Extent2D sc_extent = swapchain_.extent().value_or(
        holoviz::Extent2D { display_width_, display_height_ });
    const holoviz::Rect2D full_area { 0, 0, sc_extent.width, sc_extent.height };
    const holoviz::Image& current_image = swapchain_.image(image_index).value();

    // Single-channel (kR16Unorm) samples land in .r only; the shader replicates it to gray.
    const uint32_t push_flags = channels_ == 1 ? kShaderFlagGrayscaleReplicate : 0U;
    const auto frame_rows = static_cast<float>(full_frame_height_);
    // Full-frame pass: every row is valid, clamp only to the frame's own texel centres.
    const ShaderPushConstants full_push {
        .flags = push_flags,
        .uv_min_y = 0.5F / frame_rows,
        .uv_max_y = (frame_rows - 0.5F) / frame_rows,
    };
    // Band pass: only this band's rows are guaranteed current.
    const ShaderPushConstants band_push {
        .flags = push_flags,
        .uv_min_y = (static_cast<float>(y0) + 0.5F) / frame_rows,
        .uv_max_y = (static_cast<float>(y0 + band_height) - 0.5F) / frame_rows,
    };
    const std::span<const std::byte> full_push_bytes(
        reinterpret_cast<const std::byte*>(&full_push), sizeof(full_push));
    const std::span<const std::byte> band_push_bytes(
        reinterpret_cast<const std::byte*>(&band_push), sizeof(band_push));

    // First band of a frame: redraw the whole frame image (newest content of every band, i.e.
    // the previous frame everywhere but band 0) into this swapchain image. With 2-3 swapchain
    // images and band-only updates, a band that never arrives would otherwise leave content
    // from 2-3 frames ago on that image and the alternating images flicker between them; after
    // this pass nothing on screen is older than one frame. The clear-load pass also moves the
    // image into its presentation layout on first use.
    if (is_first_band) {
        (void)init_pass_.set_render_image(current_image, 0, { 0.F, 0.F, 0.F, 1.F });
        (void)cl.begin_render(init_pass_, full_area);
        (void)cl.set_viewport(full_area);
        (void)cl.set_scissor(full_area);
        (void)cl.bind_pipeline(pipeline_);
        (void)cl.bind_descriptor(descriptor_);
        (void)cl.set_push_data(pipeline_, full_push_bytes);
        (void)cl.draw(3);
        (void)cl.end_render();
    }

    // Band rect: sensor rows mapped proportionally onto display rows, like HSDK4's LayerAddView.
    const uint32_t disp_y0 = static_cast<uint32_t>(static_cast<uint64_t>(y0) * sc_extent.height / full_frame_height_);
    const uint32_t disp_y1 = static_cast<uint32_t>(
        static_cast<uint64_t>(y0 + band_height) * sc_extent.height / full_frame_height_);
    const holoviz::Rect2D band_rect { 0, static_cast<int32_t>(disp_y0), sc_extent.width, disp_y1 - disp_y0 };
    (void)band_pass_.set_render_image(current_image, 0, { 0.F, 0.F, 0.F, 1.F });
    (void)cl.begin_render(band_pass_, band_rect);
    (void)cl.set_viewport(full_area);
    (void)cl.set_scissor(band_rect);
    (void)cl.bind_pipeline(pipeline_);
    (void)cl.bind_descriptor(descriptor_);
    (void)cl.set_push_data(pipeline_, band_push_bytes);
    (void)cl.draw(3);
    (void)cl.end_render();
    (void)cl.end();

    // HSDK4 flag choreography: the first band consumes the acquire gate, the last band arms the
    // present gate; middle bands chain on the previous band's submit token only. The decision
    // is per frame (PendingFrame::use_gates): once the first band's submit has consumed the
    // acquire gate, consumed() reads true, so re-checking it here would skip the present gate
    // on the last band and present() would reject the frame.
    const bool use_gates = pending_frame_->use_gates;
    holoviz::SubmitDesc submit {
        .wait_submit_sync_tokens = { &pending_frame_->previous_band_token, 1 },
        .wait_external_sync_tokens = { &*ext_token_r, 1 },
        .command_lists = { &cl, 1 },
    };
    if (use_gates && is_first_band) {
        submit.acquire_gates = { &pending_frame_->acquire->acquire_gate, 1 };
    }
    if (use_gates && is_last_band) {
        submit.present_gates = { &pending_frame_->acquire->present_gate, 1 };
    }
    auto submit_r = device_.submit(submit);
    if (!submit_r) {
        HSB_LOG_ERROR("SubFrameVisualizerOp: submit failed: {}", submit_r.error().message);
        // The acquired image may hold a stale frame (nothing, or only part of this one, was
        // drawn): never present it. Drop the frame without presenting.
        abandon_pending_frame("band submit failed", /*present_partial=*/false);
        return {};
    }
    pending_frame_->previous_band_token = *submit_r;
    ++pending_frame_->band_idx;
    pending_frame_->rows_seen += band_height;
    if (const uint32_t band_index = y0 / band_height; band_index < 64U) {
        pending_frame_->bands_mask |= (uint64_t { 1 } << band_index);
    }

    if (is_last_band) {
        if (pending_frame_->rows_seen < full_frame_height_) {
            // Bands were dropped upstream (backpressure); they show last frame's content.
            ++frames_with_missing_bands_;
            const uint32_t band_count = std::min(64U, (full_frame_height_ + band_height - 1) / band_height);
            const uint64_t expected = band_count >= 64U ? ~uint64_t { 0 } : ((uint64_t { 1 } << band_count) - 1);
            missing_bands_union_ |= expected & ~pending_frame_->bands_mask;
        }
        const auto now = std::chrono::steady_clock::now();
        if (frames_with_missing_bands_ > 0 && now - missing_bands_report_time_ >= std::chrono::seconds(1)) {
            std::string missing;
            for (uint32_t i = 0; i < 64U; ++i) {
                if (missing_bands_union_ & (uint64_t { 1 } << i)) {
                    missing += fmt::format("{}{}", missing.empty() ? "" : ",", i);
                }
            }
            HSB_LOG_WARN("SubFrameVisualizerOp: {} frame(s) closed with missing bands in the last second "
                         "(band indices {}; dropped upstream, those bands show the previous frame); "
                         "longest acquire() wait {:.1f} ms",
                frames_with_missing_bands_, missing, max_acquire_wait_.count() / 1000.0);
            frames_with_missing_bands_ = 0;
            missing_bands_union_ = 0;
            max_acquire_wait_ = std::chrono::microseconds { 0 };
            missing_bands_report_time_ = now;
        }
        last_frame_token_ = pending_frame_->previous_band_token;
        if (pending_frame_->acquire) {
            if (shared_mode_) {
                // A shared-mode present is a refresh request showing whatever is in the image at
                // that moment and (after the first frame) waits on no semaphore: wait for the
                // last band's render to finish so the refresh shows the completed frame.
                if (auto r = device_.wait_on_submit(last_frame_token_, std::chrono::seconds(1)); !r) {
                    HSB_LOG_WARN("SubFrameVisualizerOp: wait_on_submit before shared present: {}",
                        r.error().message);
                }
            }
            auto present_r = swapchain_.present(std::move(*pending_frame_->acquire), last_frame_token_);
            if (present_r) {
                presentation_started_ = true;
            } else if (present_r.error().code != holoviz::ErrorCode::kFrameDropped) {
                // A pre-queue rejection leaves the acquire outstanding inside the swapchain and
                // every later acquire() would fail; release it so the next frame can start.
                HSB_LOG_ERROR("SubFrameVisualizerOp: present failed: {}", present_r.error().message);
                abandon_pending_frame("present rejected the frame");
                return {};
            }
        }
        pending_frame_.reset();
    }

    return {};
}

} // namespace hololink::operators
