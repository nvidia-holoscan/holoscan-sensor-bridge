/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_OPERATORS_IMAGE_PROCESSOR_OP_HPP
#define HOLOLINK_MODULE_OPERATORS_IMAGE_PROCESSOR_OP_HPP

#include <cstdint>
#include <memory>

#include <cuda.h>

#include <holoscan/core/operator.hpp>
#include <holoscan/core/parameter.hpp>

#include "hololink/module/cuda_unique.hpp" // UniqueCUdeviceptr

namespace hololink::module::operators {

class CudaFunctionLauncher;

/* Native module image-processing operator.
 *
 * A plain holoscan::Operator (unlike the dual-role CsiToBayerOp it is not an
 * module converter — it implements no V1 interface). It applies optical-black
 * correction and a Grey-World auto white-balance to the 16-bit Bayer image
 * produced upstream, in place, and re-emits the same entity.
 *
 * Self-contained: it carries its own copy of the engine (the NVRTC
 * applyBlackLevel / histogram / calcWBGains / applyOperations kernels, the
 * histogram sizing in start(), and the sub-frame-aware white-balance
 * accumulation in compute()). The pixel_format / bayer_format parameters are
 * plain integer enumerator values, interpreted against the module
 * hololink::module::csi::PixelFormat / BayerFormat. */
class ImageProcessorOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(ImageProcessorOp);

    void setup(holoscan::OperatorSpec& spec) override;
    void start() override;
    void stop() override;
    void compute(holoscan::InputContext&, holoscan::OutputContext& op_output,
        holoscan::ExecutionContext&) override;

private:
    holoscan::Parameter<int> pixel_format_;
    holoscan::Parameter<int> bayer_format_;
    holoscan::Parameter<int32_t> optical_black_;
    holoscan::Parameter<int> cuda_device_ordinal_;

    CUcontext cuda_context_ = nullptr;
    CUdevice cuda_device_ = 0;
    bool is_integrated_ = false;
    bool host_memory_warning_ = false;

    std::shared_ptr<CudaFunctionLauncher> cuda_function_launcher_;

    hololink::module::UniqueCUdeviceptr histogram_memory_;
    hololink::module::UniqueCUdeviceptr white_balance_gains_memory_;

    uint32_t histogram_threadblock_size_;
    int64_t expected_frame_number_ = 0;
};

} // namespace hololink::module::operators

#endif // HOLOLINK_MODULE_OPERATORS_IMAGE_PROCESSOR_OP_HPP
