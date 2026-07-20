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

#include "adcam_unpack_op.hpp"

#include <hololink/common/cuda_helper.hpp>
#include <hololink/core/logging_internal.hpp>
#include <hololink/core/networking.hpp>
#include <holoscan/holoscan.hpp>

#include <gxf/core/entity.hpp>
#include <gxf/std/tensor.hpp>

#include <fstream>
#include <cuda_runtime.h>

//------------------------------------------------------------------------------
// CUDA error checking helper
//------------------------------------------------------------------------------
#define CudaCheckRuntime(FUNC)                                      \
  {                                                                 \
    cudaError_t err = FUNC;                                         \
    if (err != cudaSuccess) {                                       \
      throw std::runtime_error(cudaGetErrorString(err));            \
    }                                                               \
  }

  //------------------------------------------------------------------------------
// Save raw packed frame (device → host → file)
//------------------------------------------------------------------------------
bool should_save_raw_packed(const std::string& filename,
                     uint8_t* device_ptr,
                     size_t bytes,
                     cudaStream_t stream) {
    std::vector<uint8_t> host_buffer(bytes);

    cudaMemcpyAsync(host_buffer.data(),
                    device_ptr,
                    bytes,
                    cudaMemcpyDeviceToHost,
                    stream);
    cudaStreamSynchronize(stream);

    if (host_buffer[0xF0] != 0x0)
    {
        return true;
    }
    //std::ofstream ofs(filename, std::ios::binary);
    //ofs.write(reinterpret_cast<char*>(host_buffer.data()), bytes);
    HOLOSCAN_LOG_INFO("Recieved invalid data");
    return false;
}


//------------------------------------------------------------------------------
// Save raw packed frame (device → host → file)
//------------------------------------------------------------------------------
void save_raw_packed(const std::string& filename,
                     uint8_t* device_ptr,
                     size_t bytes,
                     cudaStream_t stream) {
    std::vector<uint8_t> host_buffer(bytes);

    cudaMemcpyAsync(host_buffer.data(),
                    device_ptr,
                    bytes,
                    cudaMemcpyDeviceToHost,
                    stream);
    cudaStreamSynchronize(stream);

    std::ofstream ofs(filename, std::ios::binary);
    ofs.write(reinterpret_cast<char*>(host_buffer.data()), bytes);
}

//==============================================================================
// ADTFUnpackOp Implementation
//==============================================================================
namespace hololink::operators {

//------------------------------------------------------------------------------
// Setup: declare inputs, outputs, parameters
//------------------------------------------------------------------------------
void ADTFUnpackOp::setup(holoscan::OperatorSpec& spec) {
     HOLOSCAN_LOG_DEBUG("ADTFUnpackOp setup start");
    spec.input<holoscan::gxf::Entity>("input");
    spec.output<holoscan::gxf::Entity>("output");

    spec.param(width_, "width");
    spec.param(height_, "height");
    spec.param(num_planes_, "num_planes");

    spec.param(allocator_, "allocator", "Allocator",
               "Device allocator for output tensors");

    // Python equivalent: in_message.get("")
    spec.param(in_tensor_name_, "in_tensor_name", "",
               "Name of the input tensor ('' = unnamed)", std::string(""));

    spec.param(out_tensor_name_, "out_tensor_name", "output",
               "Name of the output port", std::string("output"));

    cuda_stream_handler_.define_params(spec);

    // For mode 0 &1, pixel size is 4, otherwise 5
    //

    HOLOSCAN_LOG_DEBUG("ADTFUnpackOp setup complete");
}

//------------------------------------------------------------------------------
// Start: compute frame size
//------------------------------------------------------------------------------
void ADTFUnpackOp::start() {
    frame_size_ = width_.get() * height_.get();
    pixel_size_ = num_planes_ == 2? 4 : 5;
    HOLOSCAN_LOG_DEBUG("ADTFUnpackOp start complete");
}

//------------------------------------------------------------------------------
void ADTFUnpackOp::stop() {
    HOLOSCAN_LOG_DEBUG("ADTFUnpackOp stop complete");
}

//------------------------------------------------------------------------------
// Main compute function
//------------------------------------------------------------------------------
void ADTFUnpackOp::compute(holoscan::InputContext& op_input,
                           holoscan::OutputContext& op_output,
                           holoscan::ExecutionContext& context) {

    //--------------------------------------------------------------------------
    // 1. Receive input entity
    //--------------------------------------------------------------------------
    auto maybe_entity = op_input.receive<holoscan::gxf::Entity>("input");
    if (!maybe_entity) {
        throw std::runtime_error("Failed to receive input entity");
    }
    auto& entity = static_cast<nvidia::gxf::Entity&>(maybe_entity.value());

    //--------------------------------------------------------------------------
    // 2. Extract CUDA stream from message
    //--------------------------------------------------------------------------
    gxf_result_t stream_handler_result =
        cuda_stream_handler_.from_message(context.context(), entity);
    if (stream_handler_result != GXF_SUCCESS) {
        throw std::runtime_error(fmt::format(
            "Failed to get CUDA stream: {}", GxfResultStr(stream_handler_result)));
    }
    cudaStream_t stream = cuda_stream_handler_.get_cuda_stream(context.context());

    //--------------------------------------------------------------------------
    // 3. Get input tensor (Python: msg = in_message.get(""))
    //--------------------------------------------------------------------------
    auto maybe_tensor =
        entity.get<nvidia::gxf::Tensor>(in_tensor_name_.get().c_str());
    if (!maybe_tensor) {
        throw std::runtime_error(fmt::format(
            "Input tensor '{}' not found", in_tensor_name_.get()));
    }
    auto input_tensor = maybe_tensor.value();

    //--------------------------------------------------------------------------
    // 4. Validate storage type
    //--------------------------------------------------------------------------
    if (input_tensor->storage_type() == nvidia::gxf::MemoryStorageType::kHost) {
        HOLOSCAN_LOG_WARN("Input tensor is in host memory — slower performance.");
    } else if (input_tensor->storage_type() != nvidia::gxf::MemoryStorageType::kDevice) {
        throw std::runtime_error("Unsupported tensor storage type");
    }

    const int width  = width_.get();
    const int height = height_.get();
    const int size   = width * height;

    // Expect 5 bytes per pixel (ADI ToF packed format)
    const size_t expected_bytes = static_cast<size_t>(size * pixel_size_);
    if (input_tensor->size() < expected_bytes) {
        throw std::runtime_error(fmt::format(
            "Input tensor too small: {} bytes, expected {}", input_tensor->size(), expected_bytes));
    }

    HOLOSCAN_LOG_DEBUG("Input tensor: {} bytes, {} elements",
                       input_tensor->size(), input_tensor->element_count());

    //--------------------------------------------------------------------------
    // 5. Get allocator handle
    //--------------------------------------------------------------------------
    auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
        fragment()->executor().context(), allocator_->gxf_cid());
    if (!allocator) {
        throw std::runtime_error("Failed to get allocator handle");
    }

    //--------------------------------------------------------------------------
    // 6. Create output entity and RGB output tensors
    //--------------------------------------------------------------------------
    auto out_message = nvidia::gxf::Entity::New(context.context()).value();

    auto depth_tensor = out_message.add<nvidia::gxf::Tensor>("Depth").value();
    auto ab_tensor    = out_message.add<nvidia::gxf::Tensor>("ActiveBrightness").value();
    nvidia::gxf::Handle<nvidia::gxf::Tensor> conf_tensor;
    uint8_t* conf_rgb_ptr = nullptr;
    if (num_planes_ != 2) 
    {
        conf_tensor  = out_message.add<nvidia::gxf::Tensor>("Conf").value();
        conf_tensor->reshape<uint8_t>({height, width, 3},
            nvidia::gxf::MemoryStorageType::kDevice, allocator.value());
        conf_rgb_ptr  = conf_tensor->data<uint8_t>().value();
    }

    depth_tensor->reshape<uint8_t>({height, width, 3},
        nvidia::gxf::MemoryStorageType::kDevice, allocator.value());
    ab_tensor->reshape<uint8_t>({height, width, 3},
        nvidia::gxf::MemoryStorageType::kDevice, allocator.value());

    uint8_t* depth_rgb_ptr = depth_tensor->data<uint8_t>().value();
    uint8_t* ab_rgb_ptr    = ab_tensor->data<uint8_t>().value();


    //--------------------------------------------------------------------------
    // 7. Interpret input as uint16 (Python: cp.asarray(msg))
    //--------------------------------------------------------------------------
    uint16_t* raw_u16 = input_tensor->data<uint16_t>().value();

    //raw_u16 += 128; //remove metadata frame

    //--------------------------------------------------------------------------
    // 8. Allocate internal unpack buffers (uint16)
    //--------------------------------------------------------------------------
    
    auto scratch_entity = nvidia::gxf::Entity::New(context.context()).value();

    auto depthraw_tensor = scratch_entity.add<nvidia::gxf::Tensor>("depthraw").value();
    auto abraw_tensor    = scratch_entity.add<nvidia::gxf::Tensor>("abraw").value();
    nvidia::gxf::Handle<nvidia::gxf::Tensor> confraw_tensor; //  = scratch_entity.add<nvidia::gxf::Tensor>("confraw").value();    
    uint16_t* conf  = nullptr;
    if (num_planes_ != 2)
    {
        confraw_tensor    = scratch_entity.add<nvidia::gxf::Tensor>("abraw").value();
        confraw_tensor->reshape<uint16_t>({height, width},
            nvidia::gxf::MemoryStorageType::kDevice, allocator.value());        
        conf  = confraw_tensor->data<uint16_t>().value();
    }    

    //nvidia::gxf::Tensor depthraw_tensor, confraw_tensor, abraw_tensor;

    depthraw_tensor->reshape<uint16_t>({height, width},
        nvidia::gxf::MemoryStorageType::kDevice, allocator.value());
    abraw_tensor->reshape<uint16_t>({height, width},
        nvidia::gxf::MemoryStorageType::kDevice, allocator.value());

    uint16_t* depth = depthraw_tensor->data<uint16_t>().value();
    uint16_t* ab    = abraw_tensor->data<uint16_t>().value();

    //--------------------------------------------------------------------------
    // 9. Convert 16-bit → 8-bit (Python: (cp_frame >> 8).astype(uint8))
    //--------------------------------------------------------------------------
    //nvidia::gxf::Tensor frame_u8_tensor;
    auto frame_u8_tensor = scratch_entity.add<nvidia::gxf::Tensor>("rawraw").value();
    frame_u8_tensor->reshape<uint8_t>({height, width*pixel_size_},
        nvidia::gxf::MemoryStorageType::kDevice, allocator.value());

    uint8_t* raw = frame_u8_tensor->data<uint8_t>().value();

    shift_and_cast_kernel(raw_u16, raw, size * pixel_size_, stream);

    // Save raw packed frame once
    static bool saved_once = false, valid_data = false;

    //valid_data = should_save_raw_packed("packed_frame.bin", raw, expected_bytes, stream);
    if (!saved_once){ // && (valid_data == true)) {
        save_raw_packed("packed_frame.bin", raw, expected_bytes, stream);
        saved_once = true;
    }

    //--------------------------------------------------------------------------
    // 10. Unpack 5-byte/pixel → depth/conf/ab (uint16) or 
    // Unpack 4-byte/pixel → depth/ab (uint16)
    //--------------------------------------------------------------------------
    if (num_planes_ == 2)
    {
        unpack_kernel_launch(raw, depth, nullptr, ab, width, height, stream);
    }
    else
    {
        unpack_kernel_launch(raw, depth, conf, ab, width, height, stream);
    }

    //--------------------------------------------------------------------------
    // 11. Convert to RGB (Jet + grayscale)
    //--------------------------------------------------------------------------
    jet_kernel_launch(depth, depth_rgb_ptr, size, stream);
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            HOLOSCAN_LOG_ERROR("[ADTFUnpackOp] jet_kernel launch error: {}",
                               cudaGetErrorString(err));
        }
    }
    grayscale_kernel_launch(conf, conf_rgb_ptr, size, stream,
                            255.0f); // conf: 8-bit range 0-255
    grayscale_kernel_launch(ab, ab_rgb_ptr, size, stream,
                            4096.0f); // AB:   12-bit range 0-4096
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            HOLOSCAN_LOG_ERROR(
                "[ADTFUnpackOp] grayscale_kernel launch error: {}",
                cudaGetErrorString(err));
        }
    }

    //--------------------------------------------------------------------------
    // 12. Emit output entity
    //--------------------------------------------------------------------------
    op_output.emit(out_message);
}

} // namespace hololink::operators
