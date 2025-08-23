/**
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
 *
 * See README.md for detailed information.
 */

#ifndef SRC_HOLOLINK_OPERATORS_BASE_RECEIVER_OP
#define SRC_HOLOLINK_OPERATORS_BASE_RECEIVER_OP

#include <functional>
#include <memory>
#include <tuple>

#include <cuda.h>

#include <holoscan/holoscan.hpp>

#include <hololink/common/cuda_helper.hpp>
#include <hololink/core/metadata.hpp>
#include <hololink/core/networking.hpp>

namespace hololink {
class DataChannel;
} // namespace hololink

namespace hololink::operators {

class ReceiverMemoryDescriptor {
public:
    /**
     * Allocate a region of GPU memory which will be page
     * aligned and freed on destruction.
     */
    explicit ReceiverMemoryDescriptor(CUcontext context, size_t size);
    ReceiverMemoryDescriptor() = delete;
    ~ReceiverMemoryDescriptor();

    CUdeviceptr get() { return mem_; };

protected:
    common::UniqueCUdeviceptr deviceptr_;
    common::UniqueCUhostptr host_deviceptr_;
    CUdeviceptr mem_;
};

class BaseReceiverOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(BaseReceiverOp);

    virtual ~BaseReceiverOp() = default;

    void setup(holoscan::OperatorSpec& spec) override;
    void start() override;
    void stop() override;
    void compute(holoscan::InputContext&, holoscan::OutputContext& op_output,
        holoscan::ExecutionContext&) override;

protected:
    holoscan::Parameter<DataChannel*> hololink_channel_;
    holoscan::Parameter<std::function<void()>> device_start_;
    holoscan::Parameter<std::function<void()>> device_stop_;
    holoscan::Parameter<CUcontext> frame_context_;
    holoscan::Parameter<size_t> frame_size_;
    holoscan::Parameter<bool> trim_;
    std::shared_ptr<holoscan::AsynchronousCondition> frame_ready_condition_;
    uint64_t frame_count_;

    core::UniqueFileDescriptor data_socket_;

    virtual void start_receiver() = 0;
    virtual void stop_receiver() = 0;
    virtual std::tuple<CUdeviceptr, std::shared_ptr<Metadata>> get_next_frame(double timeout_ms) = 0;
    virtual std::tuple<std::string, uint32_t> local_ip_and_port();
    virtual void timeout(holoscan::InputContext& input, holoscan::OutputContext& output,
        holoscan::ExecutionContext& context);

    // Subclasses call this in order to queue up a call to compute.
    void frame_ready();

private:
    bool ok_ = false;
};

} // namespace hololink::operators

#endif /* SRC_HOLOLINK_OPERATORS_BASE_RECEIVER_OP */
