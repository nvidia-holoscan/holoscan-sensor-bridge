/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef SRC_HOLOLINK_OPERATORS_ARGUS_ISP_ARGUS_ISP
#define SRC_HOLOLINK_OPERATORS_ARGUS_ISP_ARGUS_ISP

#include <holoscan/core/operator.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/utils/cuda_stream_handler.hpp>

#include <hololink/native/cuda_helper.hpp>

#include "cuda.h"
#include "cudaEGL.h"

namespace hololink::operators {

// Forward declaration
class CameraProvider;
class ArgusImpl;

// Maximum number of frames for EGLFrame to rotate
constexpr size_t MaxNumberOfFrames = 8;

class ArgusIspOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(ArgusIspOp)

    enum class BayerFormat {
        INVALID = -1,
        // NOTE THAT THESE GUYS LINE UP WITH THE VALUES USED BY NPP; see
        // https://docs.nvidia.com/cuda/npp/nppdefs.html#c.NppiBayerGridPosition
        RGGB = 1,
        GBRG = 2,
    };

    void setup(holoscan::OperatorSpec& spec) override;
    void start() override;
    void stop() override;
    void compute(holoscan::InputContext&,
        holoscan::OutputContext& op_output,
        holoscan::ExecutionContext&) override;

private:
    holoscan::Parameter<int> bayer_format_;
    holoscan::Parameter<float> exposure_time_ms_;
    holoscan::Parameter<float> analog_gain_;
    holoscan::Parameter<int> pixel_bit_depth_;
    holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> pool_;

    std::unique_ptr<ArgusImpl> argus_impl_;
    std::unique_ptr<CameraProvider> camera_provider_;

    CUcontext cuda_context_ = nullptr;
    CUdevice cuda_device_ = 0;
    bool is_integrated_ = true;
    bool host_memory_warning_ = false;

    CUeglFrame producerEglFrame[MaxNumberOfFrames];
    std::shared_ptr<nvidia::gxf::DLManagedTensorContext> tensor_pointers_[MaxNumberOfFrames];
    CUdeviceptr device_ptr_nv12_;

    holoscan::CudaStreamHandler cuda_stream_handler_;
    uint32_t frame_number_ = 0;
};

} // namespace hololink::operators

#endif // SRC_HOLOLINK_OPERATORS_ARGUS_ISP_ARGUS_ISP
