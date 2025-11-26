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

#include <deque>
#include <map>
#include <queue>
#include <vector>

#include <hololink/core/csi_controller.hpp>
#include <holoscan/holoscan.hpp>

#include <NvFusaCaptureExternal.hpp>
#include <nvscibuf.h>

#include <cuda.h>

namespace hololink {
class DataChannel;
} // namespace hololink

namespace hololink::operators {

/**
 * @brief Operator class to capture images using NvFusaCaptureCoe
 */
class FusaCoeCaptureOp : public holoscan::Operator, public csi::CsiConverter {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(FusaCoeCaptureOp)

    // holoscan::Operator methods.
    void setup(holoscan::OperatorSpec& spec) override;
    void start() override;
    void stop() override;
    void compute(holoscan::InputContext& op_input,
        holoscan::OutputContext& op_output,
        holoscan::ExecutionContext& context) override;

    // csi::CsiConverter methods.
    uint32_t receiver_start_byte() override;
    uint32_t received_line_bytes(uint32_t line_bytes) override;
    uint32_t transmitted_line_bytes(csi::PixelFormat pixel_format,
        uint32_t pixel_width) override;
    void configure(uint32_t start_byte, uint32_t received_bytes_per_line,
        uint32_t pixel_width, uint32_t pixel_height,
        csi::PixelFormat pixel_format,
        uint32_t trailing_bytes) override;

    // Provides a configure_converter implementation in order to configure a downstream operator
    // that is responsible for converting the packetized pixel formats that this outputs.
    void configure_converter(csi::CsiConverter& converter);

    // Callback for releasing the wrapped buffers that were passed to downstream operators.
    static nvidia::gxf::Expected<void> buffer_release_callback(void* pointer);

private:
    bool alloc_buffers();
    bool alloc_sci_buf(NvSciBufObj& buf_obj, size_t& size);
    void free_buffers();
    bool register_buffers();
    void unregister_buffers();
    bool issue_capture();
    void acquire_buffer_thread_func();

    const uint32_t capture_queue_depth_ = 4;

    holoscan::Parameter<std::string> interface_;
    holoscan::Parameter<std::vector<uint8_t>> mac_addr_;
    holoscan::Parameter<uint32_t> timeout_;
    holoscan::Parameter<std::string> out_tensor_name_;
    holoscan::Parameter<DataChannel*> hololink_channel_;
    holoscan::Parameter<std::function<void()>> device_start_;
    holoscan::Parameter<std::function<void()>> device_stop_;

    uint32_t start_byte_;
    uint32_t bytes_per_line_;
    uint32_t pixel_width_;
    uint32_t pixel_height_;
    uint32_t trailing_bytes_;
    csi::PixelFormat pixel_format_;
    bool configured_ = false;

    CUcontext cuda_context_ = nullptr;
    CUdevice cuda_device_ = 0;

    NvSciBufModule sci_buf_module_ = nullptr;

    NvFusaCaptureExternal::CoeSettings coe_settings_ = { 0 };
    NvFusaCaptureExternal::INvFusaCaptureCoeHandler* coe_handler_ = nullptr;

    std::thread acquire_thread_;
    std::atomic<bool> stop_thread_;
    std::mutex buffer_mutex_;
    std::condition_variable buffer_acquired_;

    struct BufferInfo {
        BufferInfo(FusaCoeCaptureOp* parent)
            : parent_(parent)
        {
        }

        NvSciBufObj sci_buf_ = nullptr;
        size_t size_ = 0;
        void* cpu_ptr_ = nullptr;
        cudaExternalMemory_t cuda_ext_mem_ = nullptr;
        void* cuda_device_ptr_ = 0;

        // Holds reference to parent for the sake of the release callback.
        FusaCoeCaptureOp* parent_ = nullptr;
    };

    std::deque<BufferInfo*> available_buffers_; // Buffers available for new capture requests.
    std::queue<BufferInfo*> in_flight_captures_; // Buffers in use for in-flight captures.
    BufferInfo* acquired_buffer_ = nullptr; // Last buffer acquired from capture.

    // Pending buffers that have been wrapped and passed to downstream operators and have not yet
    // been released. This map is static since the callback provides only the CUDA device pointer.
    static std::map<void*, BufferInfo*> pending_buffers_;
    static std::mutex pending_buffers_mutex_;
};

} // namespace hololink::operators
