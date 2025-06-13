// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <holoscan/holoscan.hpp>
#include <hololink/native/cuda_helper.hpp>  // For CudaFunctionLauncher
#include <memory>
#include "colorizer.hpp"

namespace holoscan::operators {

class D457RealSenseDepthSourceOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(D457RealSenseDepthSourceOp)

  void setup(OperatorSpec& spec) override;
  void start() override;
  void stop() override;
  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;
 private:
  Parameter<std::shared_ptr<Allocator>> allocator_;
  std::unique_ptr<rs2::utils::Colorizer> colorizer_;
  CUdevice cuda_device_ = 0;
  CUcontext cuda_context_ = nullptr;
  std::unique_ptr<hololink::native::CudaContextScopedPush> context_guard_;
  std::shared_ptr<hololink::native::CudaFunctionLauncher> cuda_function_launcher_;
  uint32_t width_ = 0;
  uint32_t height_ = 0;
};

}  // namespace holoscan::operators
