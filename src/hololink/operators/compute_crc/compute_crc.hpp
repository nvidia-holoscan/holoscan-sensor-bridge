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

/**
 * @class ComputeCrcOp
 * @brief Operator that computes CRC32 checksums on GPU tensors using nvCOMP.
 *
 * This operator receives a tensor on the "input" port, computes a CRC32 checksum
 * using the nvcomp library on the GPU, and emits the tensor unchanged on the "output"
 * port. The computed CRC can be retrieved via get_computed_crc() for validation;
 * normally this is done by making a CheckCrcOp instance and adding it into the
 * application pipeline at the point where the CRC value is needed.
 *
 * Uses the CRC-32/JAMCRC algorithm (bit-inverted CRC32) which matches the format
 * recorded from HSB.
 *
 * @param cuda_device_ordinal, the CUDA device to use (default: 0)
 * @param frame_size, Upper bound for data size; used to tune CRC algorithm.  The
 *      actual CRC block size is always taken from the received tensor.
 *
 * @code{.py}
 * # In your Application.compose() method:
 * compute_crc = hololink.operators.ComputeCrcOp(
 *     self,
 *     name="compute_crc",
 *     frame_size=width * height * bytes_per_pixel,
 * )
 *
 * # Connect in pipeline
 * self.add_flow(source_op, compute_crc, {("output", "input")})
 * self.add_flow(compute_crc, check_crc_op, {("output", "input")})
 * @endcode
 */
class ComputeCrcOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(ComputeCrcOp);

    void start() override;
    void stop() override;
    void setup(holoscan::OperatorSpec& spec) override;
    void compute(holoscan::InputContext&, holoscan::OutputContext& op_output,
        holoscan::ExecutionContext&) override;

    /**
     * @brief Retrieves the computed CRC value.
     *
     * Synchronizes on the CUDA stream used for CRC computation,
     * then returns the CRC value computed by the nvcomp kernel.
     * Normally this is done by CheckCrcOp, but you can call this
     * directly if desired.
     *
     * @return uint32_t The computed CRC32 value in JAMCRC format.
     * @note This call blocks until the GPU CRC computation completes.
     */
    uint32_t get_computed_crc();

protected:
    holoscan::Parameter<int> cuda_device_ordinal_; ///< CUDA device ordinal for computation
    /**
     * @brief Upper bound for the size of data to compute CRC over.
     * @note Used to tune the CRC calculation algorithm performance.
     */
    holoscan::Parameter<uint64_t> frame_size_;

    bool is_integrated_ = false; ///< True if running on integrated GPU
    bool host_memory_warning_ = false; ///< Tracks if host memory warning was issued
    holoscan::CudaStreamHandler cuda_stream_handler_;
    cudaStream_t stream_ = 0; ///< CUDA stream for CRC computation
    size_t* message_size_dptr_ = nullptr; ///< Device pointer to message size
    void** message_dptr_ = nullptr; ///< Device pointer to message pointer
    uint32_t* result_dptr_ = nullptr; ///< Device pointer to CRC result
    uint32_t computed_crc_ = 0; ///< Most recently computed CRC value
    nvcompBatchedCRC32Opts_t crc_opts_; ///< nvcomp CRC configuration
};

/**
 * @class CheckCrcOp
 * @brief Operator which 1) copies it's input parameter to it's output,
 * so it can be applied to a pipeline wherever necessary; and 2) fetches the result from
 * ComputeCrcOp, stalling the pipeline if necessary until the CRC value is computed.  This
 * second function is the primary job of this operator.
 *
 * This operator works in conjunction with ComputeCrcOp,
 * retrieving the computed CRC from a linked ComputeCrcOp instance and calls
 * the virtual check_crc() method for validation. Subclass this operator to
 * implement custom CRC validation logic.  By default, check_crc() will save
 * the computed value in the pipeline metadata under the name "computed_crc"
 * (or by any name you pass in with the `computed_crc_metadata_name' parameter).
 *
 * @param compute_crc_op points to the ComputeCrcOp that computed the CRC
 * @param computed_crc_metadata_name provides the metadata key for storing computed CRC
 *   (default: "computed_crc").  This value is ignored when a subclass overrides
 *  the check_crc method.
 *
 * @code{.py}
 * # Subclass CheckCrcOp to implement custom validation
 * class CrcValidationOp(hololink.operators.CheckCrcOp):
 *     def __init__(self, *args, compute_crc_op=None, crc_metadata_name="crc", **kwargs):
 *         hololink.operators.CheckCrcOp.__init__(
 *             self, *args, compute_crc_op=compute_crc_op, **kwargs
 *         )
 *         self._crc_metadata_name = crc_metadata_name
 *         self._error_count = 0
 *
 *     def check_crc(self, computed_crc):
 *         # Called by base class with the GPU-computed CRC
 *         received_crc = self.metadata.get(self._crc_metadata_name, 0)
 *         if computed_crc != received_crc:
 *             self._error_count += 1
 *             print(f"CRC error: computed={computed_crc:#x}, received={received_crc:#x}")
 *
 * # In compose():
 * compute_crc = hololink.operators.ComputeCrcOp(self, name="compute_crc", ...)
 * validator = CrcValidationOp(self, name="validator", compute_crc_op=compute_crc)
 * self.add_flow(compute_crc, validator, {("output", "input")})
 * @endcode
 *
 * or
 *
 * @code{.py}
 * # In compose():
 * compute_crc = hololink.operators.ComputeCrcOp(self, name="compute_crc", ...)
 * validator = hololink.operators.CheckCrcOp(self, name="validator", compute_crc_op=compute_crc)
 * self.add_flow(compute_crc, validator, {("output", "input")})
 * @endcode
 *
 * Then, at any point after `validator` is run, the computed CRC value can be found
 * at `metadata["computed_crc"]`.
 */
class CheckCrcOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(CheckCrcOp);

    /// Virtual destructor for proper polymorphic cleanup.
    virtual ~CheckCrcOp() = default;

    void setup(holoscan::OperatorSpec& spec) override;

    /**
     * Passes the "input" tensor to "output", and fetches the CRC
     * results computed by the ComputeCrcOp we were constructed with,
     * and passing that value to the `check_crc` method.  Note that this
     * operator can possibly stall the pipeline (via a CUDA stream synchronization)
     * if the CRC value is not ready yet.
     */
    void compute(holoscan::InputContext&, holoscan::OutputContext& op_output,
        holoscan::ExecutionContext&) override;

    /**
     * @brief Virtual method called with the computed CRC for validation.
     *
     * Override this method in derived classes to implement custom CRC validation
     * logic. The default implementation stores the computed CRC in metadata
     * with the metadata name stored in `computed_crc_metadata_name_`.
     *
     * @param computed_crc The CRC32 value computed by the linked ComputeCrcOp.
     */
    virtual void check_crc(uint32_t computed_crc);

protected:
    holoscan::Parameter<std::shared_ptr<ComputeCrcOp>> compute_crc_op_; ///< Linked CRC compute operator
    holoscan::Parameter<std::string> computed_crc_metadata_name_; ///< Metadata key for CRC

    holoscan::CudaStreamHandler cuda_stream_handler_;
    nvidia::gxf::Handle<holoscan::MetadataDictionary> meta_ = nullptr; ///< Entity metadata handle
};

} // namespace hololink::operators

#endif /* SRC_OPERATORS_COMPUTE_CRC */
