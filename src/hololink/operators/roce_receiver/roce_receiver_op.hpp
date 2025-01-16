/**
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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
#ifndef SRC_HOLOLINK_OPERATORS_ROCE_RECEIVER_ROCE_RECEIVER_OP
#define SRC_HOLOLINK_OPERATORS_ROCE_RECEIVER_ROCE_RECEIVER_OP

#include <memory>
#include <string>
#include <thread>

#include "hololink/operators/base_receiver_op.hpp"

namespace hololink::operators {

class RoceReceiver;

class RoceReceiverOp : public BaseReceiverOp {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(RoceReceiverOp, BaseReceiverOp);

    ~RoceReceiverOp() = default;

    // holoscan::Operator virtual functions
    void setup(holoscan::OperatorSpec& spec) override;

    // BaseReceiverOp virtual functions
    void start_receiver() override;
    void stop_receiver() override;
    std::tuple<CUdeviceptr, std::shared_ptr<hololink::Metadata>> get_next_frame(double timeout_ms) override;
    std::tuple<std::string, uint32_t> local_ip_and_port() override;

private:
    holoscan::Parameter<std::string> ibv_name_;
    holoscan::Parameter<uint32_t> ibv_port_;

    std::shared_ptr<RoceReceiver> receiver_;
    std::unique_ptr<std::thread> receiver_thread_;
    static constexpr unsigned PAGES = 2;
    std::unique_ptr<ReceiverMemoryDescriptor> frame_memory_;

    void run();
};

} // namespace hololink::operators

#endif /* SRC_HOLOLINK_OPERATORS_ROCE_RECEIVER_ROCE_RECEIVER_OP */
