#include "dummy_csi_source.hpp"

#include <holoscan/core/gxf/entity.hpp>
#include <holoscan/core/fragment.hpp>
#include <gxf/std/tensor.hpp>
#include <hololink/native/cuda_helper.hpp>

namespace {

const char* grayscale_kernel_source = R"(
extern "C" {

__global__ void fillGrayscale(unsigned char* buffer, int width, int height, int frame_count) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int index = y * width + x;
    buffer[index] = (x + y + frame_count) % 256;
  }
}

}
)";

} // anonymous namespace

namespace holoscan::operators {

void DummyCSISourceOp::start() {
  CudaCheck(cuInit(0));
  CudaCheck(cuDeviceGet(&cuda_device_, 0));
  CudaCheck(cuDevicePrimaryCtxRetain(&cuda_context_, cuda_device_));

  hololink::native::CudaContextScopedPush cur_cuda_context(cuda_context_);

  cuda_function_launcher_ = std::make_unique<hololink::native::CudaFunctionLauncher>(
      grayscale_kernel_source,
      std::vector<std::string>{ "fillGrayscale" });
}

void DummyCSISourceOp::stop() {
  hololink::native::CudaContextScopedPush cur_cuda_context(cuda_context_);

  cuda_function_launcher_.reset();

  CudaCheck(cuDevicePrimaryCtxRelease(cuda_device_));
  cuda_context_ = nullptr;
}

void DummyCSISourceOp::compute(InputContext&, OutputContext& output, ExecutionContext& context) {
  const int width = width_.get();
  const int height = height_.get();
  const size_t num_elements = static_cast<size_t>(width * height);

  nvidia::gxf::Shape shape{static_cast<int32_t>(num_elements)};
  const auto type = nvidia::gxf::PrimitiveType::kUnsigned8;
  const auto bytes_per_element = nvidia::gxf::PrimitiveTypeSize(type);
  const auto strides = nvidia::gxf::ComputeTrivialStrides(shape, bytes_per_element);

  auto allocator_handle = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
      fragment()->executor().context(), allocator_->gxf_cid());
  if (!allocator_handle) {
    throw std::runtime_error("Failed to get GXF allocator handle");
  }

  nvidia::gxf::Expected<nvidia::gxf::Entity> maybe_entity =
      CreateTensorMap(context.context(), allocator_handle.value(),
          { { "dummy_tensor", nvidia::gxf::MemoryStorageType::kDevice,
              shape, type, 0, strides } },
          false);
  if (!maybe_entity) {
    throw std::runtime_error("Failed to create output entity");
  }

  auto entity = std::move(maybe_entity.value());
  auto maybe_tensor = entity.get<nvidia::gxf::Tensor>("dummy_tensor");
  if (!maybe_tensor) {
    throw std::runtime_error("Failed to retrieve dummy_tensor");
  }

  auto tensor = maybe_tensor.value();

  hololink::native::CudaContextScopedPush cur_cuda_context(cuda_context_);

  dim3 grid((width + 15) / 16, (height + 15) / 16, 1);
  dim3 block(16, 16, 1);

  cuda_function_launcher_->launch("fillGrayscale",
      {grid.x, grid.y, grid.z},
      nullptr, // CUDA stream
      reinterpret_cast<unsigned char*>(tensor->pointer()),
      width, height, frame_count_);

  cudaDeviceSynchronize();
  frame_count_++;

  output.emit(entity);
}

} // namespace holoscan::operators
