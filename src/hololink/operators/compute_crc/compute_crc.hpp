/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef SRC_OPERATORS_COMPUTE_CRC
#define SRC_OPERATORS_COMPUTE_CRC

#include <memory>

#include <holoscan/core/operator.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/utils/cuda_stream_handler.hpp>

#include <cuda.h>
#include <nvcomp/crc32.h>

namespace hololink::operators {

class ComputeCrcOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(ComputeCrcOp);

    void start() override;
    void stop() override;
    void setup(holoscan::OperatorSpec& spec) override;
    void compute(holoscan::InputContext&, holoscan::OutputContext& op_output,
        holoscan::ExecutionContext&) override;

    /**
     * Synchronizes on the stream watching the CRC computation,
     * then returns the CRC value computed by that kernel.
     */
    uint32_t get_computed_crc();

protected:
    holoscan::Parameter<int> cuda_device_ordinal_;
    /** Note that frame_size is the upper bound on the size
     * of the data we'll compute over; this is used to tune
     * the CRC calculation algorithm.
     */
    holoscan::Parameter<uint64_t> frame_size_;

    bool is_integrated_ = false;
    bool host_memory_warning_ = false;
    holoscan::CudaStreamHandler cuda_stream_handler_;
    cudaStream_t stream_ = 0;
    size_t* message_size_dptr_ = nullptr;
    void** message_dptr_ = nullptr;
    uint32_t* result_dptr_ = nullptr;
    uint32_t computed_crc_ = 0;
    nvcompBatchedCRC32Opts_t crc_opts_;
};

class CheckCrcOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(CheckCrcOp);

    // All ABCs have virtual destructors.
    virtual ~CheckCrcOp() = default;

    void setup(holoscan::OperatorSpec& spec) override;
    void compute(holoscan::InputContext&, holoscan::OutputContext& op_output,
        holoscan::ExecutionContext&) override;

    virtual void check_crc(uint32_t computed_crc) = 0;

protected:
    holoscan::Parameter<std::shared_ptr<ComputeCrcOp>> compute_crc_op_;

    holoscan::CudaStreamHandler cuda_stream_handler_;
};

} // namespace hololink::operators

#endif /* SRC_OPERATORS_COMPUTE_CRC */
