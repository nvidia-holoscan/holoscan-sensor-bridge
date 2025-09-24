/**
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

#ifndef SRC_HOLOLINK_OPERATORS_LINUX_RECEIVER_LINUX_RECEIVER_OP
#define SRC_HOLOLINK_OPERATORS_LINUX_RECEIVER_LINUX_RECEIVER_OP

#include <functional>
#include <memory>
#include <set>
#include <string>
#include <thread>

#include "hololink/operators/base_receiver_op.hpp"

namespace hololink::operators {

class LinuxReceiver;

class LinuxReceiverOp : public BaseReceiverOp {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(LinuxReceiverOp, BaseReceiverOp);

    ~LinuxReceiverOp() = default;

    // holoscan::Operator virtual functions
    void initialize() override;
    void setup(holoscan::OperatorSpec& spec) override;

    // BaseReceiverOp virtual functions
    void start_receiver() override;
    void stop_receiver() override;
    std::tuple<CUdeviceptr, std::shared_ptr<hololink::Metadata>> get_next_frame(double timeout_ms) override;

    // Setter for rename_metadata function and receiver_affinity
    void set_rename_metadata(std::function<std::string(const std::string&)> rename_fn);
    void set_receiver_affinity(const std::vector<int>& affinity);

private:
    holoscan::Parameter<std::vector<int>> receiver_affinity_;
    std::function<std::string(const std::string&)> rename_metadata_;

    // Cached metadata key names
    std::string frame_packets_received_metadata_;
    std::string frame_number_metadata_;
    std::string received_frame_number_metadata_;
    std::string frame_bytes_received_metadata_;
    std::string received_s_metadata_;
    std::string received_ns_metadata_;
    std::string timestamp_s_metadata_;
    std::string timestamp_ns_metadata_;
    std::string metadata_s_metadata_;
    std::string metadata_ns_metadata_;
    std::string packets_dropped_metadata_;
    std::string crc_metadata_;
    std::string psn_metadata_;
    std::string bytes_written_metadata_;

    std::shared_ptr<LinuxReceiver> receiver_;
    std::unique_ptr<std::thread> receiver_thread_;
    std::unique_ptr<ReceiverMemoryDescriptor> frame_memory_;

    void run();
    void check_buffer_size(size_t data_memory_size);
    uint64_t received_address_offset();
};

} // namespace hololink::operators

#endif /* SRC_HOLOLINK_OPERATORS_LINUX_RECEIVER_LINUX_RECEIVER_OP */
