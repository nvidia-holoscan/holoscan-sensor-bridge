/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <array>
#include <chrono>
#include <thread>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <npp.h>

#include "argus_isp.hpp"

// Complete type inclusion
// ensure this includes the full definition of ArgusImpl
#include <hololink/operators/argus_isp/argus_impl.hpp>
// ensure this includes the full definition of CameraProvider
#include <hololink/operators/argus_isp/camera_provider.hpp>

#include <hololink/common/cuda_helper.hpp>
#include <hololink/core/logging_internal.hpp>
#include <hololink/operators/capture_metadata_utils.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>

namespace hololink::operators {

ArgusIspOp::ArgusIspOp(int bayer_format, float exposure_time_ms, float analog_gain,
    int pixel_bit_depth,
    std::shared_ptr<holoscan::Allocator> pool,
    uint32_t camera_index, uint32_t sensor_mode_index)
    : bayer_format_(bayer_format)
    , exposure_time_ms_(exposure_time_ms)
    , analog_gain_(analog_gain)
    , pixel_bit_depth_(pixel_bit_depth)
    , pool_(std::move(pool))
    , camera_index_(camera_index)
    , sensor_mode_index_(sensor_mode_index)
{
    if (!pool_)
        pool_ = std::make_shared<holoscan::UnboundedAllocator>();
}

ArgusIspOp::~ArgusIspOp() = default;

void ArgusIspOp::setup(holoscan::OperatorSpec& spec)
{
    // Input and output declaration names share one namespace per operator (Holoscan rejects
    // reusing a name across inputs/outputs/readiness sources), so the input ports here use an
    // "input_" prefix while the outputs keep the bare "tensor"/"metadata" names that downstream
    // consumers (add_flow, bind_tensor_output_device) wire against.
    (void)spec.input(input_tensor, "in_tensor")
        .tensor_field(holoscan::schema::ImageTensorFields::data);
    spec.input(input_metadata, "in_metadata");
    // The RGB output is loaned from the runtime pool (like ImageProcessorOp), so that
    // downstream operators such as HolovizOp can bind it. A plain make_tensor() allocation
    // gives RUNTIME_PORT_BINDING_UNAVAILABLE downstream.
    static constexpr DLDataType kUInt8 { static_cast<uint8_t>(kDLUInt), 8, 1 };
    static constexpr std::size_t kMaxOutputBytes = 64ULL * 1024 * 1024;
    (void)spec.output(output_tensor, "tensor")
        .max_emits_per_compute(1U)
        .tensor_field(holoscan::schema::ImageTensorFields::data)
        .produces_tensor(holoscan::TensorOutputSpec {
            .representation = holoscan::TensorRepresentation {
                .memory_kind = holoscan::MemoryKind::kCudaDevice,
                .dtype = kUInt8,
            },
            .bounds = holoscan::TensorBounds {
                .max_rank = 3U,
                .max_byte_span = kMaxOutputBytes,
            },
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
        });
    spec.output(output_metadata, "metadata").max_emits_per_compute(1U);
}

holoscan::Contract ArgusIspOp::contract() const
{
    // Upstream stamps both ports with an identical capture_time, so OnAligned pairs them on an
    // exact match (see kMetadataPairingWindow).
    holoscan::Contract result;
    result
        .trigger(holoscan::OnAligned {
            .ports = holoscan::ports(input_tensor, input_metadata),
            .window = hololink::operators::kMetadataPairingWindow,
        })
        .input(input_tensor, holoscan::Exact {})
        .input(input_metadata, holoscan::Exact {});
    return result;
}

void ArgusIspOp::start()
{
    camera_provider_ = std::make_unique<CameraProvider>();
    if (!camera_provider_) {
        throw std::runtime_error("Failed to initialize CameraProvider");
    }

    argus_impl_ = std::make_unique<ArgusImpl>(camera_provider_->get_camera_provider());
    if (!argus_impl_) {
        throw std::runtime_error("Failed to initialize ArgusImpl");
    }

    argus_impl_->output_pixel_format_ = OUTPUT_PIXEL_FORMAT_YUV420;

    // Get the camera devices.
    argus_impl_->setup_camera_devices(camera_index_);

    // set sensormode.
    argus_impl_->set_sensor_mode_info(sensor_mode_index_);

    // set reprocessing info
    argus_impl_->set_reprocess_info(bayer_format_, pixel_bit_depth_);

    // set output stream
    argus_impl_->setup_output_streams(sensor_mode_index_);

    // Pipe the eglStreams for camera images to CUDA.
    CudaCheck(cuInit(0));
    CudaCheck(cuDeviceGet(&cuda_device_, 0));
    CudaCheck(cuDevicePrimaryCtxRetain(&cuda_context_, cuda_device_));

    int integrated = 0;
    CudaCheck(cuDeviceGetAttribute(&integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, cuda_device_));
    is_integrated_ = (integrated != 0);

    hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);

    CudaCheck(cuEGLStreamConsumerConnect(&argus_impl_->cuda_egl_o_connection_,
        argus_impl_->i_egl_output_stream_->getEGLStream()));

    argus_impl_->setup_capture_request(analog_gain_, exposure_time_ms_);
    argus_impl_->setup_input_streams();

    CudaCheck(cuEGLStreamProducerConnect(&argus_impl_->cuda_egl_i_connection_,
        argus_impl_->i_egl_input_stream_->getEGLStream(),
        argus_impl_->i_sensor_mode_->getResolution().width(),
        argus_impl_->i_sensor_mode_->getResolution().height()));

    const uint32_t height = argus_impl_->i_sensor_mode_->getResolution().width();
    const uint32_t width = argus_impl_->i_sensor_mode_->getResolution().height();
    size_t nv12Size = (width * height) + (width * (height / 2));
    if (cuMemAlloc(&device_ptr_nv12_, nv12Size) != CUDA_SUCCESS) {
        throw std::runtime_error("Memory allocation to the device_ptr_nv12_ failed\n");
    }
}

void ArgusIspOp::stop()
{
    hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);

    Argus::ICaptureSession* i_capture_session = Argus::interface_cast<Argus::ICaptureSession>(argus_impl_->capture_session_);
    i_capture_session->stopRepeat();

    argus_impl_->stop();
    argus_impl_.reset();

    camera_provider_.reset();

    // Release held tensor references so EGL-backed buffers can be freed
    for (int i = 0; i < MaxNumberOfFrames; i++) {
        tensor_holders_[i].reset();
    }

    cuMemFree(device_ptr_nv12_);
    CudaCheck(cuDevicePrimaryCtxRelease(cuda_device_));
    cuda_context_ = nullptr;
}

holoscan::expected<void, holoscan::Error> ArgusIspOp::compute(holoscan::ExecutionContext& context)
{
    // kNotReady is a normal, expected outcome from any call in compute_impl() below (a
    // clock-driven downstream consumer, e.g. HolovizOp, can trigger this operator's activation
    // -- or contend for its output pool -- before new data/capacity has actually arrived) --
    // drop this activation instead of failing the whole pipeline over it.
    auto result = compute_impl(context);
    if (!result && result.error().code == holoscan::ErrorCode::kNotReady) {
        return {};
    }
    return result;
}

holoscan::expected<void, holoscan::Error> ArgusIspOp::compute_impl(holoscan::ExecutionContext& context)
{
    auto maybe_paired = hololink::operators::receive_paired_tensor_metadata(input_tensor, input_metadata);
    if (!maybe_paired) {
        return holoscan::make_unexpected(std::move(maybe_paired).error());
    }
    if (!*maybe_paired) {
        if (const uint64_t dropped = mismatched_frames_.count()) {
            HSB_LOG_WARN("ArgusIspOp: dropped {} frame(s) in the last second whose payload and metadata "
                         "were stamped as different frames",
                dropped);
        }
        return {};
    }
    auto& paired = **maybe_paired;
    const holoscan::schema::ImageT& in_image = paired.tensor.data;
    const holoscan::Tensor& in_tensor = *in_image.data;
    const hololink::CaptureMetadataT& in_frame = paired.metadata.data;

    DLDevice input_device = in_tensor.device();
    if (input_device.device_type == kDLCUDAHost) {
        return holoscan::make_unexpected(holoscan::Error {
            holoscan::ErrorCode::kInvalidArgument, "The tensor is on the host" });
    } else if (!is_integrated_) {
        return holoscan::make_unexpected(holoscan::Error {
            holoscan::ErrorCode::kInvalidArgument, "The tensor is not on the iGPU" });
    }

    if (in_tensor.ndim() != 3) {
        return holoscan::make_unexpected(holoscan::Error {
            holoscan::ErrorCode::kInvalidArgument, "Tensor must be an image" });
    }

    DLDataType dtype = in_tensor.dtype();
    if (dtype.code != kDLUInt || dtype.bits != 16) {
        return holoscan::make_unexpected(holoscan::Error {
            holoscan::ErrorCode::kInvalidArgument,
            fmt::format("Unexpected image data type '(code: {}, bits: {})',"
                        "expected '(code: {}, bits: {})'",
                static_cast<int>(dtype.code), dtype.bits, static_cast<int>(kDLUInt), 16) });
    }

    const auto input_shape = in_tensor.shape_span();
    const uint32_t height = input_shape[0];
    const uint32_t width = input_shape[1];
    const uint32_t components = input_shape[2];
    if (components != 1) {
        return holoscan::make_unexpected(holoscan::Error {
            holoscan::ErrorCode::kInvalidArgument,
            fmt::format("Unexpected component count {}, expected '1'", components) });
    }
    // The EGL frame below is handed the tensor's data with no row pitch (0): its rows must be
    // dense.
    const std::vector<int64_t> strides = in_tensor.strides();
    if (strides.size() != 3 || strides[2] != int64_t(sizeof(uint16_t))
        || strides[1] != int64_t(sizeof(uint16_t)) || strides[0] != int64_t(width) * int64_t(sizeof(uint16_t))) {
        return holoscan::make_unexpected(holoscan::Error {
            holoscan::ErrorCode::kInvalidArgument,
            fmt::format("Unsupported image layout: byte strides [{}]; ArgusIspOp needs dense rows",
                fmt::join(strides, ", ")) });
    }

    hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);
    // Get the CUDA stream to use for this frame.
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(context.cuda_stream().get());

    // Map the cuda pointer to the EGL frame
    size_t frame_index = frame_number_ % MaxNumberOfFrames;
    producerEglFrame[frame_index].width = argus_impl_->i_sensor_mode_->getResolution().width();
    producerEglFrame[frame_index].height = argus_impl_->i_sensor_mode_->getResolution().height();
    producerEglFrame[frame_index].pitch = 0;
    producerEglFrame[frame_index].planeCount = 1; // Bayer image
    producerEglFrame[frame_index].depth = 1;
    producerEglFrame[frame_index].numChannels = 1;
    producerEglFrame[frame_index].frameType = CU_EGL_FRAME_TYPE_PITCH;
    producerEglFrame[frame_index].eglColorFormat = CU_EGL_COLOR_FORMAT_BAYER_RGGB;
    producerEglFrame[frame_index].cuFormat = CU_AD_FORMAT_UNSIGNED_INT16;

    // Keep the input tensor alive while the EGL producer holds a reference to its data
    tensor_holders_[frame_index] = in_tensor;
    producerEglFrame[frame_index].frame.pPitch[0] = const_cast<void*>(in_tensor.data());

    CUresult r = cuEGLStreamProducerPresentFrame(&argus_impl_->cuda_egl_i_connection_,
        producerEglFrame[frame_index],
        reinterpret_cast<CUstream*>(&cuda_stream));
    if (r != CUDA_SUCCESS) {
        return holoscan::make_unexpected(holoscan::Error {
            holoscan::ErrorCode::kFailure, "Failed to present the frame to the cuEGLStream" });
    }

    if (frame_number_ > 0) {
        CUeglFrame frame;
        CUresult res;
        constexpr int kMaxRetries = 100;
        for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
            res = cuEGLStreamProducerReturnFrame(&argus_impl_->cuda_egl_i_connection_,
                &frame,
                reinterpret_cast<CUstream*>(&cuda_stream));
            if (res == CUDA_SUCCESS)
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (res != CUDA_SUCCESS) {
            return holoscan::make_unexpected(holoscan::Error {
                holoscan::ErrorCode::kFailure, "Failed to return the previous input frame" });
        }

        // Release the previous frame's tensor now that the EGL producer returned it
        tensor_holders_[(frame_number_ - 1) % MaxNumberOfFrames].reset();
    }

    Argus::ICaptureSession* i_capture_session = Argus::interface_cast<Argus::ICaptureSession>(argus_impl_->capture_session_);
    if (!i_capture_session) {
        return holoscan::make_unexpected(holoscan::Error {
            holoscan::ErrorCode::kFailure, "Failed to create CaptureSession in setup capture request" });
    }

    if (i_capture_session->capture(argus_impl_->request_.get()) == 0) {
        return holoscan::make_unexpected(holoscan::Error {
            holoscan::ErrorCode::kFailure, "Failed to submit capture request" });
    }

    CUgraphicsResource cuda_resource = { 0 };
    if (cuEGLStreamConsumerAcquireFrame(&argus_impl_->cuda_egl_o_connection_,
            &cuda_resource, reinterpret_cast<CUstream*>(&cuda_stream), CUDA_EGL_INFINITE_TIMEOUT)
        != CUDA_SUCCESS) {
        return holoscan::make_unexpected(holoscan::Error {
            holoscan::ErrorCode::kFailure, "Failed to acquire frame from EGLStream" });
    }

    // Release the consumer frame on every exit path after this point.
    auto release_consumer_frame = [&]() {
        cuEGLStreamConsumerReleaseFrame(&argus_impl_->cuda_egl_o_connection_,
            cuda_resource, reinterpret_cast<CUstream*>(&cuda_stream));
    };

    CUeglFrame cuda_egl_frame;
    if (cuGraphicsResourceGetMappedEglFrame(&cuda_egl_frame, cuda_resource, 0, 0) != CUDA_SUCCESS) {
        release_consumer_frame();
        return holoscan::make_unexpected(holoscan::Error {
            holoscan::ErrorCode::kFailure, "Failed to map output eglframe" });
    }

    // Loan the output tensor from the output port's runtime pool: (height, width, 3) uint8 RGB
    // on device.
    static constexpr DLDataType kUInt8 { static_cast<uint8_t>(kDLUInt), 8, 1 };
    const std::array<int64_t, 3> shape_arr { int64_t(height), int64_t(width), 3 };
    auto loan = output_tensor.allocate_tensor(holoscan::TensorLoanRequest { .shape = shape_arr, .dtype = kUInt8 });
    if (!loan) {
        release_consumer_frame();
        if (loan.error().code == holoscan::ErrorCode::kResourceExhausted) {
            // Every pool slot is still held downstream: drop this frame.
            if (const uint64_t dropped = dropped_frames_.count()) {
                HSB_LOG_WARN("ArgusIspOp: dropped {} frame(s) in the last second because downstream "
                             "was behind",
                    dropped);
            }
            // The input frame was presented above, so advance as after an emitted frame: the next
            // frame then takes the next slot, and the next return releases this frame's tensor.
            frame_number_++;
            return {};
        }
        return holoscan::make_unexpected(std::move(loan).error());
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
        release_consumer_frame();
        return holoscan::make_unexpected(std::move(writer).error());
    }
    auto* const out_rgb_ptr = static_cast<Npp8u*>(writer->data());

    CUDA_MEMCPY2D pCopyParams;
    pCopyParams.srcY = 0u;
    pCopyParams.srcXInBytes = 0u;
    pCopyParams.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    pCopyParams.srcArray = cuda_egl_frame.frame.pArray[0];
    pCopyParams.dstY = 0u;
    pCopyParams.dstXInBytes = 0u;
    pCopyParams.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    pCopyParams.dstDevice = device_ptr_nv12_;
    pCopyParams.dstPitch = static_cast<unsigned int>(cuda_egl_frame.width);
    pCopyParams.WidthInBytes = static_cast<unsigned int>(cuda_egl_frame.width);
    pCopyParams.Height = static_cast<unsigned int>(cuda_egl_frame.height);
    if (cuMemcpy2DAsync(&pCopyParams, cuda_stream) != CUDA_SUCCESS) {
        release_consumer_frame();
        return holoscan::make_unexpected(holoscan::Error {
            holoscan::ErrorCode::kFailure, "Could not copy from Y array to device" });
    }
    pCopyParams.srcArray = cuda_egl_frame.frame.pArray[1];
    pCopyParams.dstY = static_cast<unsigned int>(cuda_egl_frame.height);
    pCopyParams.Height = static_cast<unsigned int>(cuda_egl_frame.height) / 2;
    if (cuMemcpy2DAsync(&pCopyParams, cuda_stream) != CUDA_SUCCESS) {
        release_consumer_frame();
        return holoscan::make_unexpected(holoscan::Error {
            holoscan::ErrorCode::kFailure, "Could not copy from UV array to device" });
    }

    const auto in_y_ptr = reinterpret_cast<const uint8_t*>(device_ptr_nv12_);
    const auto in_uv_ptr = in_y_ptr + width * height;
    const uint8_t* in_y_uv_ptrs[2] = { in_y_ptr, in_uv_ptr };

    // add the npp function
    NppStreamContext npp_stream_ctx_ {};
    npp_stream_ctx_.hStream = cuda_stream;

    NppiSize oSizeROI;
    oSizeROI.width = width;
    oSizeROI.height = height;
    NppStatus status = nppiNV12ToRGB_709HDTV_8u_P2C3R_Ctx(
        in_y_uv_ptrs,
        width * sizeof(Npp8u),
        out_rgb_ptr,
        width * sizeof(Npp8u) * 3,
        oSizeROI,
        npp_stream_ctx_);
    if (status != NPP_SUCCESS) {
        release_consumer_frame();
        return holoscan::make_unexpected(holoscan::Error {
            holoscan::ErrorCode::kFailure,
            fmt::format("Failed with \"{}\" to convert NV12 to RGB\n", static_cast<int>(status)) });
    }

    if (cuEGLStreamConsumerReleaseFrame(&argus_impl_->cuda_egl_o_connection_,
            cuda_resource, reinterpret_cast<CUstream*>(&cuda_stream))
        != CUDA_SUCCESS) {
        return holoscan::make_unexpected(holoscan::Error {
            holoscan::ErrorCode::kFailure, "Failed to release the frame" });
    }

    frame_number_++;

    // Pass every metadata field through unchanged (this operator only transforms the
    // tensor payload from Bayer to RGB, not the frame's hardware metadata).
    hololink::CaptureMetadataT out_metadata = hololink::operators::forward_metadata(in_frame);

    // Both emits below share the ORIGINAL capture_time this frame arrived with (propagated,
    // not re-minted -- see propagate_emit_options()), so downstream consumers' own bounded
    // OnAligned pairing windows stay meaningful and capture_time keeps reflecting the real
    // capture instant all the way through the pipeline.
    const holoscan::EmitOptions emit_opts = hololink::operators::propagate_emit_options(
        paired.tensor.metadata, static_cast<uint64_t>(in_frame.frame_number));
    auto committed = std::move(*writer).commit();
    if (!committed) {
        return holoscan::make_unexpected(std::move(committed).error());
    }
    holoscan::schema::ImageT out_descriptor;
    out_descriptor.height = static_cast<int32_t>(height);
    out_descriptor.width = static_cast<int32_t>(width);
    out_descriptor.encoding = holoscan::schema::ImageEncoding_RGB8;
    // color_space stays UNSPECIFIED. The BT.709 matrix in nppiNV12ToRGB_709HDTV_8u_P2C3R_Ctx
    // above was used to convert from YUV; the RGB result carries no BT.709 claim, and HolovizOp
    // accepts REC709 only on YUV layers, so labelling it would make the display reject it.
    auto emitted = hololink::operators::emit_tensor_and_metadata(
        [&] { return output_tensor.emit_tensor(std::move(*loan), out_descriptor, emit_opts); },
        output_metadata, out_metadata, emit_opts);
    if (!emitted) {
        return holoscan::make_unexpected(std::move(emitted).error());
    }
    if (*emitted == hololink::operators::PairedEmit::kDropped) {
        // A downstream queue is full: the frame is dropped rather than failing the run.
        if (const uint64_t dropped = dropped_frames_.count()) {
            HSB_LOG_WARN("ArgusIspOp: dropped {} frame(s) in the last second because downstream "
                         "was behind",
                dropped);
        }
    }
    return {};
}

} // namespace hololink::operators
