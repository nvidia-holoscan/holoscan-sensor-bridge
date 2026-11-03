/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <optional>
#include <string>

#include <capture_metadata_traits.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/memory/allocator.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include <hololink/common/cuda_helper.hpp>
#include <hololink/operators/capture_metadata_utils.hpp>

#include "cuda.h"
#include "cudaEGL.h"

namespace hololink::operators {

// Forward declaration
class CameraProvider;
class ArgusImpl;

// Maximum number of frames for EGLFrame to rotate
constexpr size_t MaxNumberOfFrames = 8;

class ArgusIspOp : public holoscan::Operator<> {
public:
    enum class BayerFormat {
        INVALID = -1,
        // NOTE THAT THESE GUYS LINE UP WITH THE VALUES USED BY NPP; see
        // https://docs.nvidia.com/cuda/npp/nppdefs.html#c.NppiBayerGridPosition
        RGGB = 1,
        GBRG = 2,
    };

    ArgusIspOp(int bayer_format,
        float exposure_time_ms,
        float analog_gain,
        int pixel_bit_depth,
        std::shared_ptr<holoscan::Allocator> pool = nullptr,
        uint32_t camera_index = 0,
        uint32_t sensor_mode_index = 0);
    // Declared here (not defaulted inline) and defined in argus_isp.cpp: argus_impl_/
    // camera_provider_ below are unique_ptrs to types this header only forward-declares, so an
    // implicit inline destructor would need their complete definitions at every call site that
    // instantiates it (any TU that destroys an ArgusIspOp by value/unique_ptr, e.g.
    // operator_catalog.cpp's graph.op<ArgusIspOp>() registration row) -- only argus_isp.cpp
    // itself includes argus_impl.hpp/camera_provider.hpp.
    ~ArgusIspOp() override;

    void setup(holoscan::OperatorSpec& spec) override;
    [[nodiscard]] holoscan::Contract contract() const override;
    void start() override;
    void stop() override;
    holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

    holoscan::Input<holoscan::schema::ImageT> input_tensor { "in_tensor" };
    holoscan::Input<hololink::CaptureMetadataT> input_metadata { "in_metadata" };
    holoscan::Output<holoscan::schema::ImageT> output_tensor { "tensor" };
    holoscan::Output<hololink::CaptureMetadataT> output_metadata { "metadata" };

private:
    holoscan::expected<void, holoscan::Error> compute_impl(holoscan::ExecutionContext& context);

    int bayer_format_;
    float exposure_time_ms_;
    float analog_gain_;
    int pixel_bit_depth_;
    std::shared_ptr<holoscan::Allocator> pool_;
    uint32_t camera_index_;
    uint32_t sensor_mode_index_;

    std::unique_ptr<ArgusImpl> argus_impl_;
    std::unique_ptr<CameraProvider> camera_provider_;

    CUcontext cuda_context_ = nullptr;
    CUdevice cuda_device_ = 0;
    bool is_integrated_ = true;
    bool host_memory_warning_ = false;

    CUeglFrame producerEglFrame[MaxNumberOfFrames];
    // Keep input tensors alive while the EGL producer holds a reference to their data.
    std::optional<holoscan::Tensor> tensor_holders_[MaxNumberOfFrames];
    CUdeviceptr device_ptr_nv12_ = 0;

    uint32_t frame_number_ = 0;
    hololink::operators::DropCounter dropped_frames_;
    // Frames whose payload and metadata were stamped as different frames, reported at most once
    // per second.
    hololink::operators::DropCounter mismatched_frames_;
};

} // namespace hololink::operators

#endif // SRC_HOLOLINK_OPERATORS_ARGUS_ISP_ARGUS_ISP
