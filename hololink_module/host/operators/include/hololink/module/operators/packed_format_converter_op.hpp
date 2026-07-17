/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_OPERATORS_PACKED_FORMAT_CONVERTER_OP_HPP
#define HOLOLINK_MODULE_OPERATORS_PACKED_FORMAT_CONVERTER_OP_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <cuda.h>

#include <holoscan/core/operator.hpp>
#include <holoscan/core/parameter.hpp>

#include "hololink/module/csi_converter.hpp"

namespace hololink::module::operators {

class CudaFunctionLauncher;

/* Native module packed-CSI -> 16-bit converter operator.
 *
 * Dual role (the same pattern CsiToBayerOp / FusaCoeCaptureOp use): a
 * holoscan::Operator that unpacks the received packed-CSI buffer (RAW8/10/12)
 * into a 16-bit single-channel image in compute(), AND a
 * hololink::module::csi::CsiConverterV1 the receiver (FusaCoeCaptureOp) trains
 * through configure_converter() to learn the received byte layout.
 *
 * Self-contained: it carries its own copy of the unpack engine (the NVRTC
 * packed8bitTo16bit / packed10bitTo16bit / packed12bitTo16bit kernels),
 * expressed against the module hololink::module::csi::PixelFormat, and uses the
 * module-owned CUDA helpers. */
class PackedFormatConverterOp : public holoscan::Operator,
                                public hololink::module::csi::CsiConverterV1 {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(PackedFormatConverterOp);

    void start() override;
    void stop() override;
    void setup(holoscan::OperatorSpec& spec) override;
    void compute(holoscan::InputContext&, holoscan::OutputContext& op_output,
        holoscan::ExecutionContext&) override;

    // hololink::module::csi::CsiConverterV1
    void configure(uint32_t start_byte, uint32_t received_bytes_per_line,
        uint32_t pixel_width, uint32_t pixel_height,
        hololink::module::csi::PixelFormat pixel_format,
        uint32_t trailing_bytes) override;
    uint32_t receiver_start_byte() override;
    uint32_t received_line_bytes(uint32_t line_bytes) override;
    uint32_t transmitted_line_bytes(
        hololink::module::csi::PixelFormat pixel_format, uint32_t pixel_width) override;

    /** Size of the received CSI frame in bytes (start_byte + bytes_per_line *
     * height + trailing). Used by the receiver to size its frame buffer. */
    size_t get_frame_size();

private:
    holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
    holoscan::Parameter<int> cuda_device_ordinal_;
    holoscan::Parameter<std::string> in_tensor_name_;
    holoscan::Parameter<std::string> out_tensor_name_;

    CUcontext cuda_context_ = nullptr;
    CUdevice cuda_device_ = 0;
    bool is_integrated_ = false;
    bool host_memory_warning_ = false;

    std::shared_ptr<CudaFunctionLauncher> cuda_function_launcher_;

    uint32_t start_byte_ = 0;
    uint32_t bytes_per_line_ = 0;
    uint32_t pixel_width_ = 0;
    uint32_t pixel_height_ = 0;
    hololink::module::csi::PixelFormat pixel_format_
        = hololink::module::csi::PixelFormat::RAW_8;
    uint32_t trailing_bytes_ = 0;
    bool configured_ = false;
};

} // namespace hololink::module::operators

#endif // HOLOLINK_MODULE_OPERATORS_PACKED_FORMAT_CONVERTER_OP_HPP
