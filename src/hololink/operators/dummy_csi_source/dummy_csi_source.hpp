#pragma once

#include <holoscan/holoscan.hpp>
#include <hololink/native/cuda_helper.hpp>  // For CudaFunctionLauncher

namespace holoscan::operators {

class DummyCSISourceOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DummyCSISourceOp)

  void setup(OperatorSpec& spec) override {
    spec.output<holoscan::gxf::Entity>("output");
    spec.param(allocator_, "allocator", "Allocator",
      "Allocator used to allocate the output Bayer image, defaults to BlockMemoryPool");
    spec.param(width_, "width", "Image Width", "Width of the dummy image", 640);
    spec.param(height_, "height", "Image Height", "Height of the dummy image", 480);
  }

  void start() override;
  void stop() override;
  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;

 private:
  Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
  Parameter<int> width_;
  Parameter<int> height_;

  // Needed for CUDA context handling
  CUdevice cuda_device_ = 0;
  CUcontext cuda_context_ = nullptr;

  // Runtime CUDA kernel launcher
  std::unique_ptr<hololink::native::CudaFunctionLauncher> cuda_function_launcher_;

  // Dummy frame counter
  int frame_count_ = 0;
};

}  // namespace holoscan::operators
