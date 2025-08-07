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

#pragma once

#include <vector>

#include <holoscan/holoscan.hpp>

#include <NvSIPLCamera.hpp>
#include <NvSIPLCameraQuery.hpp>

namespace hololink::operators {

/**
 * @brief Operator class to capture images using SIPL
 */
class SIPLCaptureOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(SIPLCaptureOp)

    // Constructor with camera and JSON config file parameters, since these are
    // required by methods called before holoscan::Parameter parsing.
    template <typename... ArgsT>
    explicit SIPLCaptureOp(const std::string& camera_config, const std::string& json_config, ArgsT&&... args)
        : holoscan::Operator(std::forward<ArgsT>(args)...)
        , camera_config_(camera_config)
        , json_config_(json_config)
    {
    }

    void setup(holoscan::OperatorSpec& spec) override;
    void start() override;
    void stop() override;
    void compute(holoscan::InputContext& op_input,
        holoscan::OutputContext& op_output,
        holoscan::ExecutionContext& context) override;
    std::vector<std::string> get_output_names();

    static void list_available_configs(const std::string& json_config = "");

    static nvidia::gxf::Expected<void> buffer_release_callback(void* pointer);

private:
    void start_nvsipl();
    void start_nvsci();
    void allocate_buffers(uint32_t camera_index, nvsipl::INvSIPLClient::ConsumerDesc::OutputType output_type, std::vector<NvSciBufObj>& bufs);
    void free_buffers(std::vector<NvSciBufObj>& bufs);
    void allocate_sync(uint32_t camera_index, nvsipl::INvSIPLClient::ConsumerDesc::OutputType output_type, NvSciSyncObj& sync);
    void register_autocontrol(uint32_t camera_index);
    bool load_nito_file(std::string name, std::vector<uint8_t>& nito);

    std::string camera_config_;
    std::string json_config_;

    holoscan::Parameter<uint32_t> capture_queue_depth_;
    holoscan::Parameter<std::string> nito_base_path_;
    holoscan::Parameter<uint32_t> timeout_;

    std::unique_ptr<nvsipl::INvSIPLCameraQuery> sipl_query_;
    nvsipl::CameraSystemConfig sipl_config_;
    std::unique_ptr<nvsipl::INvSIPLCamera> sipl_camera_;

    NvSciBufModule sci_buf_module_;
    NvSciSyncModule sci_sync_module_;
    NvSciSyncCpuWaitContext cpu_wait_context_;

    struct PerCameraState {
        std::string output_name_;
        nvsipl::NvSIPLPipelineQueues queues_;
        std::vector<NvSciBufObj> sci_bufs_icp_;
        std::vector<NvSciBufObj> sci_bufs_isp0_;
        NvSciSyncObj sci_sync_isp0_;
    };
    std::vector<PerCameraState> per_camera_state_;

    struct CudaBufferMapping {
        cudaExternalMemory_t mem_;
        void* ptr_;
    };
    std::map<NvSciBufObj, CudaBufferMapping> cuda_mappings_;

    bool streaming_ = false;

    // Pending buffer map is static since the callback only provides the buffer pointer.
    static std::map<void*, nvsipl::INvSIPLClient::INvSIPLBuffer*> pending_buffers_;
    static std::mutex pending_buffers_mutex_;
};

} // namespace holoscan::ops
