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
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <hololink/core/csi_controller.hpp>
#include <hololink/core/hololink.hpp>
#include <holoscan/holoscan.hpp>

#include <INvSIPLISPStatCustomInterface.hpp>
#include <NvSIPLCamera.hpp>
#include <NvSIPLCameraQuery.hpp>

#include "sipl_compat.hpp"

namespace hololink::operators {

/**
 * @brief Manages a single INvSIPLCamera instance and per-camera SIPL/NvSci resources.
 *
 * SIPLCameraOutputOp instances (one per camera) use this service to acquire frames.
 */
class SIPLCaptureService : public std::enable_shared_from_this<SIPLCaptureService> {
public:
    struct CameraInfo {
        std::string output_name;
        uint32_t offset;
        uint32_t width;
        uint32_t height;
        uint32_t bytes_per_line;
        hololink::csi::PixelFormat pixel_format;
        hololink::csi::BayerFormat bayer_format;
    };

    SIPLCaptureService(const std::string& camera_config,
        const std::string& json_config,
        bool raw_output,
        uint32_t capture_queue_depth = 4,
        const std::string& nito_base_path = "/var/nvidia/nvcam/settings/sipl",
        uint32_t timeout_us = 1000000);

    ~SIPLCaptureService();

    static void list_available_configs(const std::string& json_config = "");

    const std::vector<CameraInfo>& get_camera_info();

    uint32_t camera_count();

    /// Called by each SIPLCameraOutputOp::start(); allocates buffers on first call.
    void add_operator_ref();

    /// Called by each SIPLCameraOutputOp::stop(); tears down when the last ref is released.
    void remove_operator_ref();

    enum class AcquireStatus {
        Ok,
        Timeout,
        Error,
    };

    /// Frame borrowed from SIPL until release_acquired_frame() or register_pending_output().
    /// plane_* pointers refer to NvSci attribute storage valid for the lifetime of this struct.
    struct AcquiredFrame {
        AcquiredFrame() = default;
        AcquiredFrame(const AcquiredFrame&) = delete;
        AcquiredFrame& operator=(const AcquiredFrame&) = delete;
        AcquiredFrame(AcquiredFrame&&) = default;
        AcquiredFrame& operator=(AcquiredFrame&&) = default;

        AcquireStatus status = AcquireStatus::Error;
        nvsipl::INvSIPLClient::INvSIPLBuffer* buffer = nullptr;
        nvsipl::INvSIPLClient::INvSIPLBuffer* buffer_raw = nullptr;
        nvsipl::INvSIPLClient::INvSIPLBuffer* buffer_isp = nullptr;
        NvSciBufObj buf_obj = nullptr;
        /// HSB frame record copied out of the RAW buffer by the acquire thread (after its EOF
        /// wait) and, in ISP mode, matched to this frame by capture-start TSC. Absent when no
        /// record was found. buffer_raw is nullptr in ISP mode: the RAW capture slot was
        /// released as soon as its record was copied.
        std::optional<Hololink::FrameMetadata> hsb_metadata;
        void* cuda_ptr = nullptr;
        uint64_t buffer_size = 0;
        uint32_t plane_count = 0;
        const uint32_t* plane_pitch = nullptr;
        const uint32_t* plane_width = nullptr;
        const uint32_t* plane_height = nullptr;
        const uint64_t* plane_offset = nullptr;
        const NvSciBufAttrValColorFmt* plane_color_format = nullptr;
    };
    AcquiredFrame acquire_frame(uint32_t camera_index, uint32_t timeout_us);

    /// Release SIPL buffers when processing fails before register_pending_output().
    void release_acquired_frame(AcquiredFrame& frame);

    const std::string& output_name(uint32_t camera_index) const;
    nvsipl::INvSIPLISPStatCustomInterface* isp_stats(uint32_t camera_index) const;
    uint32_t default_timeout_us() const { return timeout_us_; }
    bool raw_output() const { return raw_output_; }

    void release_raw_buffer_if_unused(nvsipl::INvSIPLClient::INvSIPLBuffer* buffer,
        nvsipl::INvSIPLClient::INvSIPLBuffer* buffer_raw);
    void register_pending_output(void* cuda_ptr, nvsipl::INvSIPLClient::INvSIPLBuffer* buffer);

    static nvidia::gxf::Expected<void> buffer_release_callback(void* pointer);

private:
    struct PerCameraState {
        PerCameraState();

        std::string output_name_;
        nvsipl::INvSIPLISPStatCustomInterface* isp_stats_;
        nvsipl::NvSIPLPipelineQueues queues_;
        std::vector<NvSciBufObj> sci_bufs_icp_;
        std::vector<NvSciBufObj> sci_bufs_isp0_;
        NvSciSyncObj sci_sync_isp0_;
        // Used only by this camera's acquire thread; NvSciSyncCpuWaitContext is a
        // single-owner wait primitive and must not be shared across threads.
        NvSciSyncCpuWaitContext cpu_wait_context_ = nullptr;

        std::thread acquire_thread_;
        std::unique_ptr<std::atomic<bool>> stop_thread_;
        std::unique_ptr<std::mutex> buffer_mutex_;
        std::unique_ptr<std::condition_variable> buffer_available_;
        nvsipl::INvSIPLClient::INvSIPLBuffer* buffer_raw_;
        nvsipl::INvSIPLClient::INvSIPLBuffer* buffer_isp_;
        // HSB record for the published frame; handed to acquire_frame() with the buffers.
        std::optional<Hololink::FrameMetadata> hsb_metadata_;
        // Acquire-thread only: NvSci API failures while reading the HSB trailer (rate-limited WARN).
        uint64_t hsb_metadata_read_failures_ = 0;
    };

    struct CudaBufferMapping {
        cudaExternalMemory_t mem_;
        void* ptr_;
    };

    void init_cameras();
    void init_nvsipl();
    void init_nvsci();
    void fill_camera_info();
    void allocate_buffers(uint32_t camera_index, nvsipl::INvSIPLClient::ConsumerDesc::OutputType output_type, std::vector<NvSciBufObj>& bufs);
    /// Check that the image rows CameraInfo advertises for this camera lie inside the RAW
    /// plane described by the reconciled attributes its capture buffers are allocated from.
    /// SIPLCameraOutputOp wraps that plane as-is and its consumer indexes it by CameraInfo
    /// without checking, so this is verified once here and nowhere per frame.
    void validate_raw_geometry(uint32_t camera_index, NvSciBufAttrList reconciled_attr_list) const;
    void register_buffers(uint32_t camera_index, nvsipl::INvSIPLClient::ConsumerDesc::OutputType output_type, const std::vector<NvSciBufObj>& bufs);
    void free_buffers(std::vector<NvSciBufObj>& bufs);
    void allocate_sync(uint32_t camera_index, nvsipl::INvSIPLClient::ConsumerDesc::OutputType output_type, NvSciSyncObj& sync);
    void register_autocontrol(uint32_t camera_index);
    bool load_nito_file(std::string name, std::vector<uint8_t>& nito);
    void start_buffers();
    void stop_buffers();
    void teardown_initialized_state();
    void teardown_buffer_allocations();
    void forfeit_pending_outputs();
    void ensure_streaming_started();
    nvidia::gxf::Expected<void> on_pending_output_released(void* pointer);
    void acquire_buffer_thread_func(PerCameraState* camera_state);
    /// Wait for the buffer's EOF fence on this camera's wait context. A missing fence is
    /// treated as "already complete" unless fence_required (ISP0 has a registered EOF sync).
    bool wait_for_eof(nvsipl::INvSIPLClient::INvSIPLBuffer* buffer, PerCameraState& state,
        bool fence_required = false);
    /// Read the HSB metadata trailer out of a RAW (ICP) buffer; nullopt when absent/invalid.
    /// NvSci API failures are counted on the camera state and reported at WARN, rate-limited.
    std::optional<Hololink::FrameMetadata> read_hsb_metadata(nvsipl::INvSIPLClient::INvSIPLBuffer* buffer,
        PerCameraState& state);
    void* map_buffer_to_cuda(NvSciBufObj buf_obj, uint64_t size);

    std::string camera_config_;
    std::string json_config_;
    bool raw_output_;
    uint32_t capture_queue_depth_;
    std::string nito_base_path_;
    uint32_t timeout_us_;

    std::unique_ptr<nvsipl::INvSIPLCameraQuery> sipl_query_;
    sipl_compat::SystemConfig sipl_config_;
    std::unique_ptr<nvsipl::INvSIPLCamera> sipl_camera_;

    NvSciBufModule sci_buf_module_ = nullptr;
    NvSciSyncModule sci_sync_module_ = nullptr;

    std::vector<PerCameraState> per_camera_state_;
    std::vector<CameraInfo> camera_info_;
    std::map<NvSciBufObj, CudaBufferMapping> cuda_mappings_;

    bool initialized_ = false;
    bool buffers_started_ = false;
    bool streaming_ = false;

    std::mutex init_mutex_;
    std::mutex ref_mutex_;
    uint32_t operator_ref_count_ = 0;
    std::mutex streaming_mutex_;
    std::mutex cuda_mappings_mutex_;

    std::map<void*, nvsipl::INvSIPLClient::INvSIPLBuffer*> pending_outputs_;
    std::mutex pending_outputs_mutex_;
    std::condition_variable pending_output_released_;

    static std::map<void*, std::weak_ptr<SIPLCaptureService>> pending_release_targets_;
    static std::mutex pending_release_targets_mutex_;
};

} // namespace hololink::operators
