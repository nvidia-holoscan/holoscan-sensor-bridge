#pragma once

#include <holoscan/holoscan.hpp>
#include <hololink/native/cuda_helper.hpp>
#include <librealsense2/rs.hpp>
#include "colorizer.hpp"

namespace holoscan::operators {

class D455RealSenseDualSourceOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(D455RealSenseDualSourceOp)

  void setup(OperatorSpec& spec) override;
  void start() override;
  void stop() override;
  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;

 private:
  Parameter<std::shared_ptr<Allocator>> allocator_;

  rs2::pipeline pipe_;
  std::unique_ptr<rs2::utils::Colorizer> colorizer_;
  CUdevice cuda_device_ = 0;
  CUcontext cuda_context_ = nullptr;
  std::unique_ptr<hololink::native::CudaContextScopedPush> context_guard_;
};

}  // namespace holoscan::operators
