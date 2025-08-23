/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "sipl_capture.hpp"
#include "sipl_fmt.hpp"

#include <cuda.h>

namespace hololink::operators {

std::map<void*, nvsipl::INvSIPLClient::INvSIPLBuffer*> SIPLCaptureOp::pending_buffers_;
std::mutex SIPLCaptureOp::pending_buffers_mutex_;

void SIPLCaptureOp::setup(holoscan::OperatorSpec& spec)
{
    spec.output<holoscan::gxf::Entity>("output");

    spec.param(capture_queue_depth_, "capture_queue_depth", "Capture Queue Depth",
        "Depth of the NvSIPL capture queue", 4U);
    spec.param(nito_base_path_, "nito_base_path", "NITO Base Path",
        "Base path for NITO autocontrol files", std::string("/var/nvidia/nvcam/settings/sipl"));
    spec.param(timeout_, "timeout", "Timeout",
        "Timeout for capture requests, in microseconds", 1000000U);
}

nvidia::gxf::Expected<void> SIPLCaptureOp::buffer_release_callback(void* pointer)
{
    std::lock_guard<std::mutex> lock(pending_buffers_mutex_);
    auto status = pending_buffers_[pointer]->Release();
    if (status != nvsipl::NVSIPL_STATUS_OK) {
        HSB_LOG_ERROR("Failed to release buffer {}", (void*)pending_buffers_[pointer]);
    } else {
        HSB_LOG_TRACE("Released buffer {}", (void*)pending_buffers_[pointer]);
    }
    pending_buffers_.erase(pointer);
    return nvidia::gxf::Expected<void>();
}

void SIPLCaptureOp::list_available_configs(const std::string& json_config)
{
    auto sipl_query = nvsipl::INvSIPLCameraQuery::GetInstance();
    if (!sipl_query) {
        throw std::runtime_error("Failed to get NvSIPLCameraQuery instance");
    }

    if (json_config != "") {
        // Parse database with provided JSON configuration.
        if (sipl_query->ParseJsonFile(json_config) != nvsipl::NVSIPL_STATUS_OK) {
            throw std::runtime_error("Failed to parse NvSIPLCameraQuery database");
        }
    } else {
        // Parse database with default configuration.
        if (sipl_query->ParseDatabase() != nvsipl::NVSIPL_STATUS_OK) {
            throw std::runtime_error("Failed to parse NvSIPLCameraQuery database");
        }
    }

    // Get available configuration names.
    const auto& config_names = sipl_query->GetCameraConfigNames();
    if (config_names.empty()) {
        throw std::runtime_error("No camera configurations available");
    }

    // Detect and warn about duplicate config names (e.g. if multiple
    // driver libraries use the same config names).
    std::set<std::string> unique_configs(config_names.begin(), config_names.end());
    if (config_names.size() != unique_configs.size()) {
        HSB_LOG_WARN("Duplicate camera configs found: {}", config_names);
    }

    // Print out the config details.
    std::cout << unique_configs.size() << " Available Camera Configurations:\n\n";
    for (const auto& config_name : unique_configs) {
        nvsipl::CameraSystemConfig camera_config;
        auto status = sipl_query->GetCameraSystemConfig(config_name, camera_config);
        if (status != nvsipl::NVSIPL_STATUS_OK) {
            throw std::runtime_error("Failed to get camera system config");
        }
        std::cout << fmt::format("{}:\n{}\n", config_name, camera_config);
    }
}

void SIPLCaptureOp::init_cameras()
{
    if (!initialized_) {
        init_nvsipl();
        init_nvsci();
        fill_camera_info();
        initialized_ = true;
    }
}

void SIPLCaptureOp::init_nvsipl()
{
    sipl_query_ = nvsipl::INvSIPLCameraQuery::GetInstance();
    if (!sipl_query_) {
        throw std::runtime_error("Failed to get NvSIPLCameraQuery instance");
    }

    if (json_config_ != "") {
        // Parse database with provided JSON configuration.
        HSB_LOG_DEBUG("Using JSON config: {}", json_config_);
        if (sipl_query_->ParseJsonFile(json_config_) != nvsipl::NVSIPL_STATUS_OK) {
            throw std::runtime_error("Failed to parse NvSIPLCameraQuery database");
        }
    } else {
        // Parse database with default configuration.
        if (sipl_query_->ParseDatabase() != nvsipl::NVSIPL_STATUS_OK) {
            throw std::runtime_error("Failed to parse NvSIPLCameraQuery database");
        }
    }

    // If camera config is not specified, select the first available CoE configuration.
    if (camera_config_ == "") {
        const auto& config_names = sipl_query_->GetCameraConfigNames();
        if (config_names.empty()) {
            throw std::runtime_error("No camera configurations available");
        }
        for (const auto& name : config_names) {
            nvsipl::CameraSystemConfig camera_config;
            auto status = sipl_query_->GetCameraSystemConfig(name, camera_config);
            if (status != nvsipl::NVSIPL_STATUS_OK) {
                throw std::runtime_error("Failed to get camera system config");
            }
            if (std::holds_alternative<nvsipl::CoECamera>(camera_config.cameras[0].cameratype)) {
                camera_config_ = name;
                break;
            }
        }
        if (camera_config_ == "") {
            throw std::runtime_error("Failed to find available CoE camera configuration");
        }
    }

    HSB_LOG_DEBUG("Using camera config: {}", camera_config_);
    auto status = sipl_query_->GetCameraSystemConfig(camera_config_, sipl_config_);
    if (status != nvsipl::NVSIPL_STATUS_OK) {
        throw std::runtime_error("Failed to get camera system config");
    }
    HSB_LOG_TRACE("Config Details:\n{}\n", sipl_config_);

    // Open the SIPL instance.
    sipl_camera_ = nvsipl::INvSIPLCamera::GetInstance();
    if (!sipl_camera_) {
        throw std::runtime_error("Failed to get NvSIPLCamera instance");
    }

    // Set the platform config.
    status = sipl_camera_->SetPlatformCfg(sipl_config_);
    if (status != nvsipl::NVSIPL_STATUS_OK) {
        throw std::runtime_error("Failed to set NvSIPLCamera platform config");
    }

    // Allocate the per-camera state.
    per_camera_state_.resize(sipl_config_.cameras.size());

    // Set the pipeline configs to capture either RAW or ISP-processed images.
    nvsipl::NvSIPLPipelineConfiguration sipl_pipeline_config = {
        .captureOutputRequested = raw_output_,
        .isp0OutputRequested = !raw_output_,
        .isp1OutputRequested = false,
        .isp2OutputRequested = false,
        .disableSubframe = true
    };
    for (uint32_t camera_index = 0; camera_index < per_camera_state_.size(); ++camera_index) {
        auto& camera_state = per_camera_state_[camera_index];
        status = sipl_camera_->SetPipelineCfg(camera_index, sipl_pipeline_config, camera_state.queues_);
        if (status != nvsipl::NVSIPL_STATUS_OK) {
            throw std::runtime_error("Failed to set NvSIPLCamera pipeline config");
        }

        camera_state.output_name_ = fmt::format("{}_{}", sipl_config_.cameras[camera_index].name, camera_index);
    }

    // Initialize SIPL
    status = sipl_camera_->Init();
    if (status != nvsipl::NVSIPL_STATUS_OK) {
        throw std::runtime_error("Failed to initialize NvSIPLCamera");
    }
}

void SIPLCaptureOp::init_nvsci()
{
    NvSciError err;

    err = NvSciSyncModuleOpen(&sci_sync_module_);
    if (err != NvSciError_Success) {
        throw std::runtime_error("Failed to initialize NvSciSyncModule");
    }

    err = NvSciBufModuleOpen(&sci_buf_module_);
    if (err != NvSciError_Success) {
        throw std::runtime_error("Failed to initialize NvSciBufModule");
    }

    err = NvSciSyncCpuWaitContextAlloc(sci_sync_module_, &cpu_wait_context_);
    if (err != NvSciError_Success) {
        throw std::runtime_error("Failed to allocate NvSciCpuWaitContext");
    }
}

void SIPLCaptureOp::fill_camera_info()
{
    camera_info_.resize(sipl_config_.cameras.size());
    for (uint32_t camera_index = 0; camera_index < camera_info_.size(); ++camera_index) {
        // Get and reconcile the buffer attributes for the given camera in order to
        // determine the buffer pitch.
        std::unique_ptr<NvSciBufAttrList> attr_list(new NvSciBufAttrList());
        auto err = NvSciBufAttrListCreate(sci_buf_module_, attr_list.get());
        if (err != NvSciError_Success) {
            throw std::runtime_error("Failed to create NvSciBufAttrList");
        }

        const auto& camera = sipl_config_.cameras[camera_index];
        auto status = sipl_camera_->GetImageAttributes(camera.sensorInfo.id,
            nvsipl::INvSIPLClient::ConsumerDesc::OutputType::ICP, *(attr_list.get()));
        if (status != nvsipl::NVSIPL_STATUS_OK) {
            throw std::runtime_error(fmt::format("Failed to get image attributes ({})", static_cast<int>(status)));
        }

        std::unique_ptr<NvSciBufAttrList> reconciled_attr_list(new NvSciBufAttrList());
        std::unique_ptr<NvSciBufAttrList> conflict_attr_list(new NvSciBufAttrList());
        err = NvSciBufAttrListReconcile(attr_list.get(),
            1U,
            reconciled_attr_list.get(),
            conflict_attr_list.get());
        if (err != NvSciError_Success) {
            throw std::runtime_error("Failed to reconcile NvSciBuf attributes");
        }

        NvSciBufAttrKeyValuePair img_attrs[] = { { NvSciBufImageAttrKey_PlanePitch, NULL, 0 } };
        err = NvSciBufAttrListGetAttrs(*reconciled_attr_list, img_attrs, 1);
        if (err != NvSciError_Success) {
            throw std::runtime_error("Failed to get buffer attributes");
        }
        const uint32_t plane_pitch = *(static_cast<const uint32_t*>(img_attrs[0].value));

        // Fill in the camera information.
        auto& info = camera_info_[camera_index];
        info.output_name = per_camera_state_[camera_index].output_name_;
        info.offset = 0;
        info.width = camera.sensorInfo.vcInfo.resolution.width;
        info.height = camera.sensorInfo.vcInfo.resolution.height;
        info.bytes_per_line = plane_pitch;

        // Pixel format
        switch (camera.sensorInfo.vcInfo.inputFormat) {
        case NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW10:
        case NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW10TP:
            info.pixel_format = hololink::csi::PixelFormat::RAW_10;
            info.offset = ((info.width * 10) / 8) * camera.sensorInfo.vcInfo.embeddedTopLines;
            break;
        case NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12:
            info.pixel_format = hololink::csi::PixelFormat::RAW_12;
            info.offset = ((info.width * 12) / 8) * camera.sensorInfo.vcInfo.embeddedTopLines;
            break;
        default:
            throw std::runtime_error("Unsupported input format");
        }

        // Bayer order
        switch (camera.sensorInfo.vcInfo.cfa) {
        case NVSIPL_PIXEL_ORDER_RGGB:
            info.bayer_format = hololink::csi::BayerFormat::RGGB;
            break;
        case NVSIPL_PIXEL_ORDER_BGGR:
            info.bayer_format = hololink::csi::BayerFormat::BGGR;
            break;
        case NVSIPL_PIXEL_ORDER_GRBG:
            info.bayer_format = hololink::csi::BayerFormat::GRBG;
            break;
        case NVSIPL_PIXEL_ORDER_GBRG:
            info.bayer_format = hololink::csi::BayerFormat::GBRG;
            break;
        default:
            throw std::runtime_error("Unsupported color filter array");
        }
    }
}

void SIPLCaptureOp::allocate_buffers(uint32_t camera_index, nvsipl::INvSIPLClient::ConsumerDesc::OutputType output_type, std::vector<NvSciBufObj>& bufs)
{
    // Helper for cleaning up attribute lists
    struct CloseNvSciBufAttrList {
        void operator()(NvSciBufAttrList* attr_list) const
        {
            if (attr_list != nullptr) {
                if ((*attr_list) != nullptr) {
                    NvSciBufAttrListFree(*attr_list);
                }
                delete attr_list;
            }
        }
    };

    // Create the requested attribute list.
    std::unique_ptr<NvSciBufAttrList, CloseNvSciBufAttrList> attr_list;
    attr_list.reset(new NvSciBufAttrList());
    NvSciError err = NvSciBufAttrListCreate(sci_buf_module_, attr_list.get());
    if (err != NvSciError_Success) {
        throw std::runtime_error("Failed to create NvSciBufAttrList");
    }

    // Common attributes (to ICP and ISP outputs).
    constexpr NvSciBufType buf_type = NvSciBufType_Image;
    constexpr NvSciBufAttrValAccessPerm access_perm = NvSciBufAccessPerm_Readonly;

    // Allow buffers to be mapped into the GPU for both ICP and ISP0 outputs.
    CUuuid uuid;
    cuDeviceGetUuid_v2(&uuid, 0);
    NvSciRmGpuId gpu_id = { 0 };
    memcpy(&gpu_id.bytes, uuid.bytes, sizeof(uuid.bytes));

    // Require CPU read access for ISP output.
    constexpr bool is_cpu_access_req = true;
    constexpr bool is_cpu_cache_enabled = true;

    // Require pitch-linear NV12 for ISP output.
    constexpr NvSciBufAttrValImageLayoutType layout = NvSciBufImage_PitchLinearType;
    constexpr NvSciBufSurfType surf_type = NvSciSurfType_YUV;
    constexpr NvSciBufSurfSampleType surf_sample_type = NvSciSurfSampleType_420;
    constexpr NvSciBufSurfBPC surf_bpc = NvSciSurfBPC_8;
    constexpr NvSciBufSurfMemLayout surf_mem_layout = NvSciSurfMemLayout_SemiPlanar;
    constexpr NvSciBufSurfComponentOrder surf_comp_order = NvSciSurfComponentOrder_YUV;
    constexpr NvSciBufAttrValColorStd surf_color_std[] = { NvSciColorStd_REC709_ER };

    NvSciBufAttrKeyValuePair attr_kvp[] = {
        { NvSciBufGeneralAttrKey_Types, &buf_type, sizeof(buf_type) },
        { NvSciBufGeneralAttrKey_RequiredPerm, &access_perm, sizeof(access_perm) },
        { NvSciBufGeneralAttrKey_GpuId, &gpu_id, sizeof(gpu_id) },
        // ISP-specific attributes:
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &is_cpu_access_req, sizeof(is_cpu_access_req) },
        { NvSciBufGeneralAttrKey_EnableCpuCache, &is_cpu_cache_enabled, sizeof(is_cpu_cache_enabled) },
        { NvSciBufImageAttrKey_Layout, &layout, sizeof(layout) },
        { NvSciBufImageAttrKey_SurfType, &surf_type, sizeof(surf_type) },
        { NvSciBufImageAttrKey_SurfBPC, &surf_bpc, sizeof(surf_bpc) },
        { NvSciBufImageAttrKey_SurfMemLayout, &surf_mem_layout, sizeof(surf_mem_layout) },
        { NvSciBufImageAttrKey_SurfSampleType, &surf_sample_type, sizeof(surf_sample_type) },
        { NvSciBufImageAttrKey_SurfComponentOrder, &surf_comp_order, sizeof(surf_comp_order) },
        { NvSciBufImageAttrKey_SurfColorStd, &surf_color_std, sizeof(surf_color_std) },
    };

    const size_t num_attrs = (output_type == nvsipl::INvSIPLClient::ConsumerDesc::OutputType::ICP)
        ? 3U
        : sizeof(attr_kvp) / sizeof(attr_kvp[0]);
    err = NvSciBufAttrListSetAttrs(*(attr_list.get()), attr_kvp, num_attrs);
    if (err != NvSciError_Success) {
        throw std::runtime_error("Failed to set NvSciBufAttrList values");
    }

    // Get the attributes provided by the camera.
    const auto sensor_id = sipl_config_.cameras[camera_index].sensorInfo.id;
    auto status = sipl_camera_->GetImageAttributes(sensor_id, output_type, *(attr_list.get()));
    if (err != NvSciError_Success) {
        throw std::runtime_error("Failed to get image attributes");
    }

    // Reconcile the attributes.
    std::unique_ptr<NvSciBufAttrList, CloseNvSciBufAttrList> reconciled_attr_list;
    std::unique_ptr<NvSciBufAttrList, CloseNvSciBufAttrList> conflict_attr_list;
    reconciled_attr_list.reset(new NvSciBufAttrList());
    conflict_attr_list.reset(new NvSciBufAttrList());
    err = NvSciBufAttrListReconcile(attr_list.get(),
        1U,
        reconciled_attr_list.get(),
        conflict_attr_list.get());
    if (err != NvSciError_Success) {
        throw std::runtime_error("Failed to reconcile NvSciBuf attributes");
    }

    // Allocate the buffers.
    for (size_t i = 0; i < capture_queue_depth_.get(); i++) {
        NvSciBufObj buf_obj;
        err = NvSciBufObjAlloc(*(reconciled_attr_list.get()), &buf_obj);
        if (err != NvSciError_Success || !buf_obj) {
            throw std::runtime_error("Failed to allocate NvSciBufObj");
        }
        bufs.push_back(buf_obj);
    }

    // Register the buffers with SIPL.
    status = sipl_camera_->RegisterImages(sensor_id, output_type, bufs);
    if (status != nvsipl::NVSIPL_STATUS_OK) {
        throw std::runtime_error("Failed to register images with NvSIPLCamera");
    }

    HSB_LOG_DEBUG("Allocated and registered {} buffers for output type {}",
        bufs.size(), static_cast<uint32_t>(output_type));
}

void SIPLCaptureOp::free_buffers(std::vector<NvSciBufObj>& bufs)
{
    for (auto buf : bufs) {
        NvSciBufObjFree(buf);
    }
    bufs.clear();
}

void SIPLCaptureOp::allocate_sync(uint32_t camera_index, nvsipl::INvSIPLClient::ConsumerDesc::OutputType output_type, NvSciSyncObj& sync)
{
    // Helper for cleaning up attribute lists
    struct CloseNvSciSyncAttrList {
        void operator()(NvSciSyncAttrList* attr_list) const
        {
            if (attr_list != nullptr) {
                if ((*attr_list) != nullptr) {
                    NvSciSyncAttrListFree(*attr_list);
                }
                delete attr_list;
            }
        }
    };

    // Create the CPU waiter attribute list.
    std::unique_ptr<NvSciSyncAttrList, CloseNvSciSyncAttrList> waiter_attr_list;
    waiter_attr_list.reset(new NvSciSyncAttrList());
    auto err = NvSciSyncAttrListCreate(sci_sync_module_, waiter_attr_list.get());
    if (err != NvSciError_Success) {
        throw std::runtime_error("Failed to create waiter NvSciSyncAttrList");
    }

    NvSciSyncAttrKeyValuePair kv[2];
    memset(kv, 0, sizeof(kv));

    bool cpu_signaler_waiter = true;
    kv[0].attrKey = NvSciSyncAttrKey_NeedCpuAccess;
    kv[0].value = (void*)&cpu_signaler_waiter;
    kv[0].len = sizeof(cpu_signaler_waiter);

    NvSciSyncAccessPerm cpu_perm = NvSciSyncAccessPerm_WaitOnly;
    kv[1].attrKey = NvSciSyncAttrKey_RequiredPerm;
    kv[1].value = (void*)&cpu_perm;
    kv[1].len = sizeof(cpu_perm);

    err = NvSciSyncAttrListSetAttrs(*(waiter_attr_list.get()), kv, 2);
    if (err != NvSciError_Success) {
        throw std::runtime_error("Failed to set NvSciSyncAttrList values");
    }

    // Get the camera signaler attribute list.
    std::unique_ptr<NvSciSyncAttrList, CloseNvSciSyncAttrList> signaler_attr_list;
    signaler_attr_list.reset(new NvSciSyncAttrList());
    err = NvSciSyncAttrListCreate(sci_sync_module_, signaler_attr_list.get());
    if (err != NvSciError_Success) {
        throw std::runtime_error("Failed to create signaler NvSciSyncAttrList");
    }

    const auto sensor_id = sipl_config_.cameras[camera_index].sensorInfo.id;
    auto status = sipl_camera_->FillNvSciSyncAttrList(sensor_id, output_type, *(signaler_attr_list.get()), nvsipl::SIPL_SIGNALER);
    if (status != nvsipl::NVSIPL_STATUS_OK) {
        throw std::runtime_error("Failure in FillNvSciSyncAttrList");
    }

    // Reconcile the attributes.
    NvSciSyncAttrList unreconciled_list[2];
    unreconciled_list[0] = *(waiter_attr_list.get());
    unreconciled_list[1] = *(signaler_attr_list.get());

    std::unique_ptr<NvSciSyncAttrList, CloseNvSciSyncAttrList> reconciled_attr_list;
    std::unique_ptr<NvSciSyncAttrList, CloseNvSciSyncAttrList> conflict_attr_list;
    reconciled_attr_list.reset(new NvSciSyncAttrList());
    conflict_attr_list.reset(new NvSciSyncAttrList());

    err = NvSciSyncAttrListReconcile(unreconciled_list, 2, reconciled_attr_list.get(), conflict_attr_list.get());
    if (err != NvSciError_Success) {
        throw std::runtime_error("Failed to reconcile NvSciSync attributes");
    }

    // Allocate the sync object.
    err = NvSciSyncObjAlloc(*(reconciled_attr_list.get()), &sync);
    if (err != NvSciError_Success) {
        throw std::runtime_error("Failed to allocate NvSciSyncObj");
    }

    // Register the sync object.
    status = sipl_camera_->RegisterNvSciSyncObj(sensor_id, output_type, nvsipl::NVSIPL_EOFSYNCOBJ, sync);
    if (status != nvsipl::NVSIPL_STATUS_OK) {
        throw std::runtime_error("Failed to register NvSciSyncObj");
    }

    HSB_LOG_DEBUG("Created and registered sync object for output type {}", static_cast<uint32_t>(output_type));
}

void SIPLCaptureOp::register_autocontrol(uint32_t camera_index)
{
    std::vector<uint8_t> blob;
    if (!load_nito_file(sipl_config_.cameras[camera_index].name, blob)) {
        throw std::runtime_error("Failed to load NITO file for autocontrol");
    }

    const auto sensor_id = sipl_config_.cameras[camera_index].sensorInfo.id;
    auto status = sipl_camera_->RegisterAutoControlPlugin(sensor_id, nvsipl::NV_PLUGIN, nullptr, blob);
    if (status != nvsipl::NVSIPL_STATUS_OK) {
        throw std::runtime_error("Failed to register autocontrol plugin");
    }
}

void SIPLCaptureOp::start()
{
    init_cameras();

    for (uint32_t camera_index = 0; camera_index < per_camera_state_.size(); ++camera_index) {
        auto& camera_state = per_camera_state_[camera_index];

        // Allocate RAW capture buffers.
        allocate_buffers(camera_index, nvsipl::INvSIPLClient::ConsumerDesc::OutputType::ICP, camera_state.sci_bufs_icp_);

        // Allocate ISP0 output buffers, sync object, and register autocontrol.
        if (!raw_output_) {
            allocate_buffers(camera_index, nvsipl::INvSIPLClient::ConsumerDesc::OutputType::ISP0, camera_state.sci_bufs_isp0_);
            allocate_sync(camera_index, nvsipl::INvSIPLClient::ConsumerDesc::OutputType::ISP0, camera_state.sci_sync_isp0_);
            register_autocontrol(camera_index);
        }
    }
}

void SIPLCaptureOp::stop()
{
    // Stop streaming.
    sipl_camera_->Stop();

    // Clear CUDA mappings.
    for (auto mapping : cuda_mappings_) {
        cudaFree(mapping.second.ptr_);
        cudaDestroyExternalMemory(mapping.second.mem_);
    }
    cuda_mappings_.clear();

    // Cleanup per-camera state.
    for (uint32_t camera_index = 0; camera_index < per_camera_state_.size(); ++camera_index) {
        auto& camera_state = per_camera_state_[camera_index];

        // Free sync objects.
        if (camera_state.sci_sync_isp0_) {
            NvSciSyncObjFree(camera_state.sci_sync_isp0_);
        }

        // Free buffers.
        free_buffers(camera_state.sci_bufs_icp_);
        free_buffers(camera_state.sci_bufs_isp0_);
    }
    per_camera_state_.clear();

    // Cleanup NvSciSync/Buf modules.
    if (cpu_wait_context_) {
        NvSciSyncCpuWaitContextFree(cpu_wait_context_);
        cpu_wait_context_ = nullptr;
    }
    if (sci_sync_module_) {
        NvSciSyncModuleClose(sci_sync_module_);
        sci_sync_module_ = nullptr;
    }
    if (sci_buf_module_) {
        NvSciBufModuleClose(sci_buf_module_);
        sci_buf_module_ = nullptr;
    }

    // Cleanup NvSIPL state.
    sipl_camera_.reset();
    sipl_query_.reset();
}

void SIPLCaptureOp::compute(holoscan::InputContext& op_input,
    holoscan::OutputContext& op_output,
    holoscan::ExecutionContext& context)
{
    // Streaming is started when the first frame is requested.
    // This is deferred instead of being done in start() to minimize the capture queue
    // from filling due to the long initialization of some other operator.
    if (!streaming_) {
        HSB_LOG_DEBUG("Starting streaming");
        auto status = sipl_camera_->Start();
        if (status != nvsipl::NVSIPL_STATUS_OK) {
            throw std::runtime_error("Failed to start streaming");
        }
        streaming_ = true;
    }

    auto entity = holoscan::gxf::Entity::New(&context);
    for (auto& camera_state : per_camera_state_) {
        // Get completion queue based on output type.
        auto completion_queue = raw_output_
            ? camera_state.queues_.captureCompletionQueue
            : camera_state.queues_.isp0CompletionQueue;

        // Wait for a new frame.
        HSB_LOG_TRACE("Waiting for buffer for {} (timeout = {}us)...", camera_state.output_name_, timeout_.get());
        nvsipl::INvSIPLClient::INvSIPLBuffer* buffer = nullptr;
        auto status = completion_queue->Get(buffer, timeout_.get());
        if (status != nvsipl::NVSIPL_STATUS_OK || buffer == nullptr) {
            throw std::runtime_error(fmt::format("Failed to get buffer for {} (status = {})",
                camera_state.output_name_, static_cast<int>(status)));
        }

        auto nvm_buffer = dynamic_cast<nvsipl::INvSIPLClient::INvSIPLNvMBuffer*>(buffer);
        if (nvm_buffer == nullptr) {
            throw std::runtime_error("Failed to get INvSIPLNvMBuffer");
        }

        // Get and wait for the EOF fence (if there is one).
        NvSciSyncFence fence = NvSciSyncFenceInitializer;
        status = nvm_buffer->GetEOFNvSciSyncFence(&fence);
        if (status == nvsipl::NVSIPL_STATUS_OK) {
            auto err = NvSciSyncFenceWait(&fence, cpu_wait_context_, timeout_.get());
            if (err != NvSciError_Success) {
                NvSciSyncFenceClear(&fence);
                throw std::runtime_error("Failed to wait for EOF fence");
            }
        }
        NvSciSyncFenceClear(&fence);

        // Get the NvSci buffer object.
        NvSciBufObj buf_obj = nvm_buffer->GetNvSciBufImage();
        if (buf_obj == nullptr) {
            throw std::runtime_error("Failed to get NvSciBufObj");
        }

        // Get the attributes of the buffer.
        NvSciBufAttrList buf_attr_list;
        auto err = NvSciBufObjGetAttrList(buf_obj, &buf_attr_list);
        if (err != NvSciError_Success) {
            throw std::runtime_error("Failed to get buffer attribute list");
        }

        // Get the specific attributes that are needed to determine how to
        // allocate and read pixels to an output buffer.
        NvSciBufAttrKeyValuePair img_attrs[] = {
            { NvSciBufImageAttrKey_Size, NULL, 0 }, // 0
            { NvSciBufImageAttrKey_Layout, NULL, 0 }, // 1
            { NvSciBufImageAttrKey_PlaneCount, NULL, 0 }, // 2
            { NvSciBufImageAttrKey_PlanePitch, NULL, 0 }, // 3
            { NvSciBufImageAttrKey_PlaneWidth, NULL, 0 }, // 4
            { NvSciBufImageAttrKey_PlaneHeight, NULL, 0 }, // 5
            { NvSciBufImageAttrKey_PlaneBitsPerPixel, NULL, 0 }, // 6
            { NvSciBufImageAttrKey_PlaneAlignedHeight, NULL, 0 }, // 7
            { NvSciBufImageAttrKey_PlaneAlignedSize, NULL, 0 }, // 8
            { NvSciBufImageAttrKey_PlaneChannelCount, NULL, 0 }, // 9
            { NvSciBufImageAttrKey_PlaneOffset, NULL, 0 }, // 10
            { NvSciBufImageAttrKey_PlaneColorFormat, NULL, 0 }, // 11
            { NvSciBufGeneralAttrKey_CpuNeedSwCacheCoherency, NULL, 0 }, // 12
        };
        size_t num_attrs = sizeof(img_attrs) / sizeof(img_attrs[0]);

        err = NvSciBufAttrListGetAttrs(buf_attr_list, img_attrs, num_attrs);
        if (err != NvSciError_Success) {
            throw std::runtime_error("Failed to get buffer attributes");
        }

        const uint64_t size = *(static_cast<const uint64_t*>(img_attrs[0].value));
        const uint32_t layout = *(static_cast<const uint32_t*>(img_attrs[1].value));
        const uint32_t plane_count = *(static_cast<const uint32_t*>(img_attrs[2].value));
        const uint32_t* plane_pitch = static_cast<const uint32_t*>(img_attrs[3].value);
        const uint32_t* plane_width = static_cast<const uint32_t*>(img_attrs[4].value);
        const uint32_t* plane_height = static_cast<const uint32_t*>(img_attrs[5].value);
        const uint32_t* plane_bits_per_pixel = static_cast<const uint32_t*>(img_attrs[6].value);
        const uint32_t* plane_aligned_height = static_cast<const uint32_t*>(img_attrs[7].value);
        const uint64_t* plane_aligned_size = static_cast<const uint64_t*>(img_attrs[8].value);
        const uint8_t* plane_channel_count = static_cast<const uint8_t*>(img_attrs[9].value);
        const uint64_t* plane_offset = static_cast<const uint64_t*>(img_attrs[10].value);
        const NvSciBufAttrValColorFmt* plane_color_format = static_cast<const NvSciBufAttrValColorFmt*>(img_attrs[11].value);

        if (hololink::logging::HSB_LOG_LEVEL_TRACE >= hololink::logging::hsb_log_level) {
            std::stringstream ss;
            ss << "Got buffer; output = " << camera_state.output_name_;
            ss << ", planes = " << plane_count;
            ss << ", size = " << size;
            ss << ", layout = " << layout << "\n";
            for (uint32_t i = 0; i < plane_count; ++i) {
                ss << "  " << i << ": pitch = " << plane_pitch[i];
                ss << ", width = " << plane_width[i];
                ss << ", height = " << plane_height[i];
                ss << ", bpp = " << plane_bits_per_pixel[i];
                ss << ", aligned_height = " << plane_aligned_height[i];
                ss << ", aligned_size = " << plane_aligned_size[i];
                ss << ", channel_count = " << static_cast<int>(plane_channel_count[i]);
                ss << ", offset = " << plane_offset[i];
                ss << ", format = " << fmt::format("{}", plane_color_format[i]) << "\n";
            }
            HSB_LOG_TRACE(ss.str());
        }

        // Map the buffer into CUDA (if it hasn't already been mapped before).
        if (cuda_mappings_.find(buf_obj) == cuda_mappings_.end()) {
            CudaBufferMapping mapping;

            // Register NvSciBuf with CUDA
            cudaExternalMemoryHandleDesc ext_mem_handle_desc;
            memset(&ext_mem_handle_desc, 0, sizeof(ext_mem_handle_desc));
            ext_mem_handle_desc.type = cudaExternalMemoryHandleTypeNvSciBuf;
            ext_mem_handle_desc.handle.nvSciBufObject = buf_obj;
            ext_mem_handle_desc.size = size;
            auto cuda_err = cudaImportExternalMemory(&mapping.mem_, &ext_mem_handle_desc);
            if (cuda_err != cudaSuccess) {
                throw std::runtime_error("Failed to import NvSciBuf into CUDA");
            }

            // Map buffer into CUDA
            cudaExternalMemoryBufferDesc buffer_desc;
            memset(&buffer_desc, 0, sizeof(buffer_desc));
            buffer_desc.offset = 0;
            buffer_desc.size = size;
            cuda_err = cudaExternalMemoryGetMappedBuffer(&mapping.ptr_, mapping.mem_, &buffer_desc);
            if (cuda_err != cudaSuccess) {
                throw std::runtime_error("Failed to map NvSciBuf into CUDA");
            }

            // Add the mapping to the list.
            cuda_mappings_[buf_obj] = mapping;

            HSB_LOG_DEBUG("Mapped buffer ({}) into CUDA (mem={} / ptr={})",
                (void*)buf_obj, (void*)mapping.mem_, (void*)mapping.ptr_);
        }

        const auto name = camera_state.output_name_.c_str();

        const bool is_nv12 = plane_color_format[0] == NvSciColor_Y8 && plane_color_format[1] == NvSciColor_V8U8;
        const bool is_raw10 = plane_color_format[0] == NvSciColor_X2Rc10Rb10Ra10_Bayer10RGGB
            || plane_color_format[0] == NvSciColor_X2Rc10Rb10Ra10_Bayer10BGGR
            || plane_color_format[0] == NvSciColor_X2Rc10Rb10Ra10_Bayer10GRBG
            || plane_color_format[0] == NvSciColor_X2Rc10Rb10Ra10_Bayer10GBRG;

        if (is_nv12) {
            // Create the output VideoBuffer to wrap the buffer.
            auto video_buffer = static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::VideoBuffer>(name);
            if (!video_buffer) {
                throw std::runtime_error("Failed to add GXF VideoBuffer");
            }
            nvidia::gxf::VideoTypeTraits<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12> video_type;
            nvidia::gxf::VideoFormatSize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12> color_format;
            auto color_planes = color_format.getDefaultColorPlanes(plane_width[0], plane_height[0]);
            color_planes[1].offset = plane_offset[1];
            nvidia::gxf::VideoBufferInfo info {
                plane_width[0],
                plane_height[0],
                video_type.value,
                std::move(color_planes),
                nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR
            };
            if (!video_buffer.value()->wrapMemory(info, size, nvidia::gxf::MemoryStorageType::kDevice,
                    cuda_mappings_[buf_obj].ptr_, buffer_release_callback)) {
                throw std::runtime_error("Failed to add wrapped VideoBuffer memory");
            }
        } else if (is_raw10) {
            // Create the output Tensor to wrap the buffer.
            auto tensor = static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::Tensor>(name);
            if (!tensor) {
                throw std::runtime_error("Failed to add GXF Tensor");
            }

            nvidia::gxf::Shape shape { static_cast<int>(size) };
            const auto element_type = nvidia::gxf::PrimitiveType::kUnsigned8;
            const auto element_size = nvidia::gxf::PrimitiveTypeSize(element_type);

            if (!tensor.value()->wrapMemory(shape, element_type, element_size,
                    nvidia::gxf::ComputeTrivialStrides(shape, element_size),
                    nvidia::gxf::MemoryStorageType::kDevice,
                    cuda_mappings_[buf_obj].ptr_, buffer_release_callback)) {
                throw std::runtime_error("Failed to add wrapped memory");
            }
        } else {
            throw std::runtime_error(fmt::format("Buffer has unsupported color format: {}",
                plane_color_format[0]));
        }

        std::lock_guard<std::mutex> lock(pending_buffers_mutex_);
        pending_buffers_[cuda_mappings_[buf_obj].ptr_] = buffer;
        HSB_LOG_TRACE("Output buffer {} ({} pending)", static_cast<void*>(buffer), pending_buffers_.size());
    }

    op_output.emit(entity, "output");
}

const std::vector<SIPLCaptureOp::CameraInfo>& SIPLCaptureOp::get_camera_info()
{
    init_cameras();
    return camera_info_;
}

bool SIPLCaptureOp::load_nito_file(std::string module_name, std::vector<uint8_t>& nito)
{
    std::string module_name_lower;
    for (auto& c : module_name) {
        module_name_lower.push_back(std::tolower(c));
    }
    std::string nito_file[] = { nito_base_path_.get() + "/" + module_name + ".nito",
        nito_base_path_.get() + "/" + module_name_lower + ".nito" };

    FILE* fp = nullptr;
    HSB_LOG_DEBUG("Opening NITO file for module \"{}\"", module_name);
    for (uint32_t i = 0; i < sizeof(nito_file) / sizeof(nito_file[0]); ++i) {
        fp = fopen(nito_file[i].c_str(), "rb");
        if (fp == NULL) {
            HSB_LOG_DEBUG("  File not found: \"{}\"", nito_file[i]);
        } else {
            HSB_LOG_DEBUG("  Opened file: \"{}\"", nito_file[i]);
            break;
        }
    }

    if (fp == NULL) {
        HSB_LOG_ERROR("Unable to open NITO file for module \"{}\"", module_name);
        return false;
    }

    // Check file size.
    fseek(fp, 0, SEEK_END);
    auto fsize = ftell(fp);
    rewind(fp);
    if (fsize <= 0U) {
        HSB_LOG_ERROR("NITO file for module \"{}\" is of invalid size", module_name);
        fclose(fp);
        return false;
    }

    // Read the file.
    nito.resize(fsize);
    auto result = (long int)fread(nito.data(), 1, fsize, fp);
    if (result != fsize) {
        HSB_LOG_ERROR("Fail to read data from NITO file for module \"{}\"", module_name);
        nito.resize(0);
        fclose(fp);
        return false;
    }

    fclose(fp);

    HSB_LOG_DEBUG("Data from NITO file loaded for module \"{}\"", module_name);

    return true;
}

} // namespace holoscan::ops
