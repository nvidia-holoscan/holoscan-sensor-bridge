/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_OPERATORS_CSI_TO_BAYER_OP_HPP
#define HOLOLINK_MODULE_OPERATORS_CSI_TO_BAYER_OP_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <cuda.h>

#include <holoscan/core/operator.hpp>
#include <holoscan/core/parameter.hpp>

#include "hololink/module/csi_converter.hpp"
#include "hololink/module/cuda_unique.hpp" // UniqueCUdeviceptr

namespace hololink::module::operators {

class CudaFunctionLauncher;

/* Native module CSI -> Bayer converter operator.
 *
 * Dual role (the same pattern FusaCoeCaptureOp uses): a holoscan::Operator
 * that decodes the received CSI buffer into a 16-bit Bayer image in
 * compute(), AND a hololink::module::csi::CsiConverterV1 the module sensor
 * drivers (Imx274Cam / Vb1940Cam) train through configure_converter(). The
 * operator therefore plugs straight into the pipeline and is handed to the
 * sensor directly — no application-layer shim.
 *
 * Self-contained: it carries its own copy of the CSI -> Bayer engine (the
 * NVRTC frameReconstruction8/10/12 kernels, the sub-frame accumulation, and
 * the four geometry helpers), expressed against the module
 * hololink::module::csi::PixelFormat. It needs no module service resolution
 * (no enumeration_metadata). */
class CsiToBayerOp : public holoscan::Operator,
                     public hololink::module::csi::CsiConverterV1 {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(CsiToBayerOp);

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

    /** Length of the CSI data in bytes (start_byte + bytes_per_line * height +
     * trailing). Used by the receiver operator to size its frame buffer. */
    size_t get_csi_length();

    /** Size of a sub-frame in bytes when sub_frame_rows is set, otherwise the
     * full frame size. */
    size_t get_sub_frame_size();

private:
    holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
    holoscan::Parameter<int> cuda_device_ordinal_;
    holoscan::Parameter<std::string> out_tensor_name_;
    holoscan::Parameter<uint32_t> sub_frame_rows_;

    CUcontext cuda_context_ = nullptr;
    CUdevice cuda_device_ = 0;
    bool is_integrated_ = false;
    bool host_memory_warning_ = false;

    std::shared_ptr<CudaFunctionLauncher> cuda_function_launcher_;

    size_t sub_frame_memory_size_ = 0;
    hololink::module::UniqueCUdeviceptr sub_frame_memory_;

    uint32_t pixel_width_ = 0;
    uint32_t pixel_height_ = 0;
    hololink::module::csi::PixelFormat pixel_format_
        = hololink::module::csi::PixelFormat::RAW_8;
    uint32_t start_byte_ = 0;
    uint32_t bytes_per_line_ = 0;
    size_t csi_length_ = 0;

    /// Frame size in bytes; differs from csi_length_ when sub_frame_rows_ is set.
    size_t frame_size_ = 0;

    bool configured_ = false;
};

} // namespace hololink::module::operators

#endif // HOLOLINK_MODULE_OPERATORS_CSI_TO_BAYER_OP_HPP
