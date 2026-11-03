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

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <cuda.h>
#include <time.h>

#include <capture_metadata_traits.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/operators/capture_metadata_utils.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include <holoviz/device.hpp>
#include <holoviz/display.hpp>
#include <holoviz/external_semaphore.hpp>
#include <holoviz/holoviz.hpp>
#include <holoviz/image.hpp>
#include <holoviz/pipeline.hpp>
#include <holoviz/present_mode.hpp>
#include <holoviz/render_pass.hpp>
#include <holoviz/surface.hpp>
#include <holoviz/swapchain.hpp>
#include <holoviz/window.hpp>

namespace hololink::operators {

/**
 * HSDK5 sub-frame visualizer.
 *
 * Each compute() activation receives one horizontal band (sub-frame) of the sensor frame as a
 * schema::ImageT sample (uint16, 1 or 4 components per pixel), paired with its CaptureMetadataT.
 * Bands may arrive out of order; `sub_frame_offset` identifies each band's row position. The band is copied into a CUDA-exported RHI image and drawn into the acquired
 * swapchain image with a content-preserving (LoadOp::kLoad) render pass, so earlier bands survive.
 * The frame is presented once the last band (offset + rows >= full_frame_height) has been drawn.
 *
 * HSDK4 -> HSDK5 mapping:
 *   viz::Begin(DONT_CLEAR_COLOR|DONT_SWAP_BUFFERS)  -> LoadOp::kLoad band pass + per-band submit
 *   viz::LayerAddView(0, off/H, 1, rows/H)          -> begin_render(band_rect) + set_scissor(band_rect)
 *   viz::ImageCudaDevice(ptr)                        -> cuMemcpy2DAsync into kExported image
 *   viz::WaitForDisplayEvent(FIRST_PIXEL_OUT)        -> Device::wait_for_display_event(kFirstPixelOut)
 *   viz::End() on the last band                      -> Swapchain::present()
 *
 * Sinks:
 *   - windowed (default): a window on `display_name` (fullscreen when `fullscreen`), presented
 *     through the compositor.
 *   - exclusive (`use_exclusive_display`): direct-to-display scanout on the DRM connector whose
 *     name contains `display_name` (empty = the first display no compositor owns), at the mode
 *     closest to display_width x display_height @ display_framerate. Linux only; the process must
 *     run where no X11/Wayland session holds the GPU.
 *
 * `present_mode` selects how presented frames reach the display. kAuto picks the best tear-free
 * mode the surface offers (kFIFOLatestReady, else kFIFO). The shared modes
 * (kSharedDemandRefresh / kSharedContinuousRefresh, VK_KHR_shared_presentable_image) render every
 * band in place into the single presentable image; in continuous mode the presentation engine
 * then refreshes from that image on its own, so after the first present bands stream straight to
 * the display with no per-frame acquire/present at all (the HSDK4 "never swap buffers" path). A
 * mode the surface does not support falls back to the kAuto choice with a warning.
 */
class SubFrameVisualizerOp : public holoscan::Operator<> {
public:
    SubFrameVisualizerOp(uint32_t full_frame_height,
        bool fullscreen = false,
        bool use_exclusive_display = false,
        std::string display_name = {},
        uint32_t display_width = 1920,
        uint32_t display_height = 1080,
        uint32_t display_framerate = 59'950,
        std::string window_title = "Sub-Frame Visualizer",
        std::shared_ptr<Hololink::PtpSynchronizer> ptp_synchronizer = nullptr,
        holoviz::PresentMode present_mode = holoviz::PresentMode::kFIFO);

    void setup(holoscan::OperatorSpec& spec) override;
    holoscan::Contract contract() const override;
    void start() override;
    /// Second half of start(): everything that stop() unwinds on failure.
    void start_gpu();
    void stop() override;
    holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

    holoscan::Input<holoscan::schema::ImageT> input_tensor { "tensor" };
    holoscan::Input<hololink::CaptureMetadataT> input_metadata { "metadata" };

private:
    // Frames whose payload and metadata were stamped as different frames, reported at most once
    // per second.
    hololink::operators::DropCounter mismatched_frames_;
    /**
     * Background thread that waits for the display First Pixel Out (FPO) event and
     * schedules the next FPGA camera capture via PTP/PPS set_delay(), synchronizing
     * capture timing to the display refresh rate.
     */
    void thread_func();

    /**
     * Create the CUDA-exported frame image (and bind it to the descriptor) for a band tensor with
     * `channels` uint16 components per pixel (1 -> kR16Unorm, 4 -> kR16G16B16A16Unorm). Called on
     * the first band; later bands must match.
     */
    void ensure_frame_image(uint32_t channels);

    uint32_t full_frame_height_;
    bool fullscreen_;
    bool use_exclusive_display_;
    std::string display_name_;
    uint32_t display_width_;
    uint32_t display_height_;
    uint32_t display_framerate_;
    std::string window_title_;
    // Shared rather than raw: graph.op<>() rejects raw object pointers among its captured
    // constructor arguments, and the operator outlives the caller's own reference.
    std::shared_ptr<Hololink::PtpSynchronizer> ptp_synchronizer_;
    holoviz::PresentMode present_mode_; // requested; the resolved mode lives in the swapchain
    bool shared_mode_ { false }; // swapchain runs a shared present mode
    bool continuous_mode_ { false }; // kSharedContinuousRefresh: no acquire/present after the first frame
    bool presentation_started_ { false }; // first present() succeeded

    // RHI handles, valid only between start() and stop(). Declaration order matters for
    // teardown: the surface must outlive the swapchain, and the window the surface.
    holoviz::Window window_; // empty in exclusive-display mode
    holoviz::Surface surface_; // from the window, or from the exclusively acquired display
    holoviz::Device device_;
    holoviz::Swapchain swapchain_;
    holoviz::Image frame_image_; // kExported, display_width_ x full_frame_height_
    CUarray frame_image_level0_ {};
    uint32_t channels_ { 0 }; // uint16 components per pixel of frame_image_; 0 until the first band
    holoviz::Pipeline pipeline_;
    holoviz::Descriptor descriptor_;
    holoviz::RenderPass init_pass_; // full clear on first use of a swapchain image
    holoviz::RenderPass band_pass_; // LoadOp::kLoad, preserves the other bands
    holoviz::ExternalSemaphore external_semaphore_;
    CUdevice cuda_device_ {};
    CUcontext cuda_context_ {}; // the device's primary context, retained for start()..stop()
    bool cuda_primary_ctx_retained_ { false };
    CUstream cuda_stream_ {};
    // Orders the band copy on cuda_stream_ after the producer's writes (queued by the runtime on
    // the activation stream) and makes the activation stream wait for the copy, so the runtime
    // releases the input tensor only once the copy has read it.
    CUevent producer_ready_event_ {};
    CUevent copy_done_event_ {};
    // command_lists_[image_index][band_idx], grown lazily
    std::vector<std::vector<holoviz::CommandList>> command_lists_;
    std::optional<holoviz::Display> display_for_fpo_;

    // State of the frame currently being assembled (between first and last band)
    struct PendingFrame {
        // nullopt in continuous shared mode once presentation has started: bands then draw
        // straight into the shared image and no present() closes the frame.
        std::optional<holoviz::AcquireSyncToken> acquire;
        holoviz::SubmitSyncToken previous_band_token;
        uint32_t band_idx { 0 };
        // Decided once per frame: the first band's submit consumes the acquire gate and the
        // last band's arms the present gate. Shared-mode acquires after the first arrive
        // pre-consumed and take no gates at all.
        bool use_gates { false };
        uint32_t rows_seen { 0 }; // sensor rows drawn so far; < full_frame_height_ at close = dropped bands
        uint64_t bands_mask { 0 }; // bit i set once band i (y0 / band height) has been drawn
    };
    // Diagnostics, reported at most once per second from the frame-closing band.
    uint32_t frames_with_missing_bands_ { 0 };
    uint64_t missing_bands_union_ { 0 }; // which band indices went missing since the last report
    std::chrono::microseconds max_acquire_wait_ { 0 }; // longest swapchain acquire() block
    std::chrono::steady_clock::time_point missing_bands_report_time_ {};
    std::optional<PendingFrame> pending_frame_;

    /**
     * Release the frame being assembled (present() rejected it, its last band never arrived, or
     * a band's submit failed) so the next acquire() succeeds. With \p present_partial the bands
     * drawn so far are presented; without it (nothing of the frame reached the GPU) the acquired
     * image is dropped through a swapchain recreation, the only RHI release path, so a stale
     * image is never shown.
     */
    void abandon_pending_frame(const char* why, bool present_partial = true);
    holoviz::SubmitSyncToken last_frame_token_;
    uint64_t swapchain_generation_ { 0 };

    std::thread thread_;
    std::atomic<bool> stop_requested_ = false;
    std::mutex render_start_time_mutex_;
    /**
     * Pipeline time anchor at first sub-frame: sensor frame start from timestamp_s/ns when present
     * (PTP-aligned with FPGA in typical stacks), otherwise CLOCK_REALTIME when compute runs. The FPO
     * thread uses this for the capture phase calculation.
     */
    timespec render_start_time_ = { 0, 0 };
    std::atomic<bool> fpo_available_ { true };
};

} // namespace hololink::operators
