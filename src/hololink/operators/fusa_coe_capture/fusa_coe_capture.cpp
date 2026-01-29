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

#include "fusa_coe_capture.hpp"

#include <hololink/common/cuda_error.hpp>
#include <hololink/common/cuda_helper.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/logging.hpp>

/**
 * @brief This macro defining a YAML converter which throws for unsupported types.
 *
 * Background: Holoscan supports setting parameters through YAML files. But for some parameters
 * accepted by the receiver operators like `DataChannel` class of functions it makes no sense
 * to specify them in YAML files. Therefore use a converter which throws for these types.
 *
 * @tparam TYPE
 */
#define YAML_CONVERTER(TYPE)                                                                \
    template <>                                                                             \
    struct YAML::convert<TYPE> {                                                            \
        static Node encode(TYPE&) { throw std::runtime_error("Unsupported"); }              \
        static bool decode(const Node&, TYPE&) { throw std::runtime_error("Unsupported"); } \
    };

YAML_CONVERTER(hololink::DataChannel*);
YAML_CONVERTER(std::function<void()>);

using namespace NvFusaCaptureExternal;

namespace hololink::operators {

std::map<void*, FusaCoeCaptureOp::BufferInfo*> FusaCoeCaptureOp::pending_buffers_;
std::mutex FusaCoeCaptureOp::pending_buffers_mutex_;

void FusaCoeCaptureOp::setup(holoscan::OperatorSpec& spec)
{
    spec.output<holoscan::gxf::Entity>("output");

    register_converter<hololink::DataChannel*>();
    register_converter<std::function<void()>>();

    spec.param(interface_, "interface", "Interface", "Interface for the CoE device.");
    spec.param(mac_addr_, "mac_addr", "MACAddr", "MAC Address for the CoE device.");
    spec.param(timeout_, "timeout", "Timeout", "Timeout for capture requests, in milliseconds");
    spec.param(out_tensor_name_, "out_tensor_name", "OutputTensorName",
        "Name of the output tensor", std::string(""));
    spec.param(hololink_channel_, "hololink_channel", "HololinkChannel",
        "Pointer to Hololink Datachannel object");
    spec.param(device_start_, "device_start", "DeviceStart",
        "Function to be called to start the device");
    spec.param(device_stop_, "device_stop", "DeviceStop",
        "Function to be called to stop the device");
}

void FusaCoeCaptureOp::start()
{
    if (!configured_) {
        throw std::runtime_error("FusaCoeCaptureOp is not configured.");
    }

    // Initialize CUDA.
    CudaCheck(cuInit(0));
    CudaCheck(cuDeviceGet(&cuda_device_, 0));
    CudaCheck(cuDevicePrimaryCtxRetain(&cuda_context_, cuda_device_));

    // Initialize NvSciBuf module.
    if (NvSciBufModuleOpen(&sci_buf_module_) != NvSciError_Success) {
        throw std::runtime_error("Failed to open NvSciBuf module");
    }

    // Allocate buffers.
    if (!alloc_buffers()) {
        throw std::runtime_error("Failed to allocate buffers");
    }

    // Open NvFusa CoE capture handler.
    coe_settings_.if_name = interface_.get().c_str();
    memcpy(coe_settings_.sensor_mac_addr,
        mac_addr_.get().data(),
        sizeof(coe_settings_.sensor_mac_addr));

    auto status = INvFusaCapture::openCoeHandler(coe_settings_, &coe_handler_);
    if (status != NvFusaCaptureStatus::OK || !coe_handler_) {
        throw std::runtime_error(
            fmt::format("Failed to open NvFusa CoE Handler (error = {}, handle = {})",
                static_cast<int>(status), static_cast<void*>(coe_handler_)));
    }

    HSB_LOG_INFO("Opened NvFusaCapture channel {}", coe_handler_->getChannelNumber());

    // Register output buffers with NvFusa.
    if (!register_buffers()) {
        throw std::runtime_error("Failed to register output buffers");
    }

    // Configure HSB packetizer.
    auto packetizer_program = csi::get_packetizer_program(pixel_format_);
    hololink_channel_.get()->set_packetizer_program(packetizer_program);

    // Configure HSB for CoE captures.
    uint32_t frame_size = start_byte_ + (pixel_height_ * bytes_per_line_) + trailing_bytes_;
    hololink_channel_.get()->configure_coe(coe_handler_->getChannelNumber(),
        frame_size, bytes_per_line_);

    // Issue initial capture requests.
    while (available_buffers_.size() > 0) {
        if (!issue_capture()) {
            throw std::runtime_error("Failed to issue capture request");
        }
    }

    // Start the capture acquisition thread.
    stop_thread_.store(false);
    acquire_thread_ = std::thread(&FusaCoeCaptureOp::acquire_buffer_thread_func, this);

    // Start streaming.
    device_start_.get()();
}

void FusaCoeCaptureOp::stop()
{
    // Stop the buffer acquisition thread. This will drain the in-flight captures before
    // terminating and needs to be done before stopping the device stream below.
    if (acquire_thread_.joinable()) {
        stop_thread_.store(true);
        acquire_thread_.join();
    }

    // Stop streaming.
    device_stop_.get()();

    // Unregister buffers and close NvFusa channel.
    if (coe_handler_ != nullptr) {
        unregister_buffers();
        INvFusaCapture::closeHandler(coe_handler_);
        coe_handler_ = nullptr;
    }

    // Free buffers.
    free_buffers();

    // Cleanup NvSciBuf module.
    if (sci_buf_module_ != nullptr) {
        NvSciBufModuleClose(sci_buf_module_);
        sci_buf_module_ = nullptr;
    }

    // Cleanup Hololink.
    hololink_channel_.get()->unconfigure();
}

void FusaCoeCaptureOp::acquire_buffer_thread_func()
{
    while (in_flight_captures_.size() > 0) {
        // Acquire the next capture.
        auto buffer = in_flight_captures_.front();
        auto status = coe_handler_->getCaptureStatus(buffer->sci_buf_, timeout_.get());
        if (status != NvFusaCaptureStatus::OK) {
            throw std::runtime_error(fmt::format("CoE capture failed (error = {})", static_cast<int>(status)));
        }
        in_flight_captures_.pop();

        std::lock_guard<std::mutex> lock(buffer_mutex_);

        // Notify compute thread of the new buffer (replacing any unused buffer).
        if (acquired_buffer_) {
            available_buffers_.push_back(acquired_buffer_);
        }
        acquired_buffer_ = buffer;
        buffer_acquired_.notify_all();

        // Issue new capture requests (unless stop has been requested).
        if (!stop_thread_.load()) {
            while (available_buffers_.size() > 0) {
                if (!issue_capture()) {
                    throw std::runtime_error("Failed to issue capture request");
                }
            }
        }
    }

    // Return any unused buffers.
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    if (acquired_buffer_) {
        available_buffers_.push_back(acquired_buffer_);
        acquired_buffer_ = nullptr;
    }
}

void FusaCoeCaptureOp::compute(holoscan::InputContext& op_input,
    holoscan::OutputContext& op_output,
    holoscan::ExecutionContext& context)
{
    // Wait for and take ownership of the next acquired buffer.
    std::unique_lock<std::mutex> lock(buffer_mutex_);
    BufferInfo* buffer = acquired_buffer_;
    while (buffer == nullptr) {
        HSB_LOG_TRACE("Waiting for buffer for output \"{}\" (timeout = {}ms)...",
            out_tensor_name_.get(), timeout_.get());
        auto status = buffer_acquired_.wait_for(lock, std::chrono::milliseconds(timeout_.get()));
        if (status == std::cv_status::timeout) {
            throw std::runtime_error(fmt::format("Failed to get buffer for output \"{}\"", out_tensor_name_.get()));
        }
        buffer = acquired_buffer_;
    }
    acquired_buffer_ = nullptr;
    HSB_LOG_TRACE("Got buffer for output \"{}\" ({})",
        out_tensor_name_.get(), static_cast<void*>(buffer));
    lock.unlock();

    // Read the HSB metadata
    if (is_metadata_enabled()) {
        auto* metadata_ptr = static_cast<uint8_t*>(buffer->cpu_ptr_) + start_byte_ + (pixel_height_ * bytes_per_line_) + trailing_bytes_;
        auto frame_metadata = Hololink::deserialize_metadata(metadata_ptr, hololink::METADATA_SIZE);
        auto const& meta = metadata();
        meta->set("timestamp_s", int64_t(frame_metadata.timestamp_s));
        meta->set("timestamp_ns", int64_t(frame_metadata.timestamp_ns));
        meta->set("metadata_s", int64_t(frame_metadata.metadata_s));
        meta->set("metadata_ns", int64_t(frame_metadata.metadata_ns));
        meta->set("crc", int64_t(frame_metadata.crc));
    }

    // Create the output Tensor to wrap the buffer.
    auto entity = holoscan::gxf::Entity::New(&context);
    auto name = out_tensor_name_.get().c_str();
    auto tensor = static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::Tensor>(name);
    if (!tensor) {
        throw std::runtime_error("Failed to add GXF Tensor");
    }

    nvidia::gxf::Shape shape { static_cast<int>(bytes_per_line_ * pixel_height_) };
    const auto element_type = nvidia::gxf::PrimitiveType::kUnsigned8;
    const auto element_size = nvidia::gxf::PrimitiveTypeSize(element_type);

    void* data_ptr = static_cast<uint8_t*>(buffer->cuda_device_ptr_) + start_byte_;
    if (!tensor.value()->wrapMemory(shape, element_type, element_size,
            nvidia::gxf::ComputeTrivialStrides(shape, element_size),
            nvidia::gxf::MemoryStorageType::kDevice,
            data_ptr, buffer_release_callback)) {
        throw std::runtime_error("Failed to add wrapped memory");
    }

    // Add the buffer to the pending list for the release callback.
    std::lock_guard<std::mutex> pending_lock(pending_buffers_mutex_);
    pending_buffers_[data_ptr] = buffer;

    op_output.emit(entity, "output");
}

uint32_t FusaCoeCaptureOp::receiver_start_byte()
{
    return 0;
}

uint32_t FusaCoeCaptureOp::received_line_bytes(uint32_t line_bytes)
{
    return hololink::core::round_up(line_bytes, 64);
}

uint32_t FusaCoeCaptureOp::transmitted_line_bytes(csi::PixelFormat pixel_format,
    uint32_t pixel_width)
{
    switch (pixel_format) {
    case csi::PixelFormat::RAW_8:
        return pixel_width;
    case csi::PixelFormat::RAW_10:
        // 3 pixels per 4 bytes
        return ((pixel_width + 2) / 3) * 4;
    case csi::PixelFormat::RAW_12:
        // 2 pixels per 3 bytes
        return ((pixel_width + 1) / 2) * 3;
    default:
        throw std::runtime_error("Invalid bit depth");
    }
}

void FusaCoeCaptureOp::configure(uint32_t start_byte, uint32_t received_bytes_per_line,
    uint32_t pixel_width, uint32_t pixel_height,
    csi::PixelFormat pixel_format,
    uint32_t trailing_bytes)
{
    HSB_LOG_INFO("start={}, bytes_per_line={}, width={}, height={}, format={}, trailing_bytes={}",
        start_byte, received_bytes_per_line, pixel_width, pixel_height,
        static_cast<int>(pixel_format), trailing_bytes);
    start_byte_ = start_byte;
    bytes_per_line_ = received_bytes_per_line;
    pixel_width_ = pixel_width;
    pixel_height_ = pixel_height;
    pixel_format_ = pixel_format;
    trailing_bytes_ = trailing_bytes;

    configured_ = true;
}

void FusaCoeCaptureOp::configure_converter(csi::CsiConverter& converter)
{
    converter.configure(0, bytes_per_line_, pixel_width_, pixel_height_, pixel_format_, 0);
}

bool FusaCoeCaptureOp::alloc_sci_buf(NvSciBufObj& buf_obj, size_t& size)
{
    // Add padding lines to the buffer for sensor and HSB metadata.
    uint32_t width = pixel_width_;
    uint32_t height = pixel_height_;
    height += hololink::core::round_up(start_byte_, bytes_per_line_) / bytes_per_line_;
    height += hololink::core::round_up(trailing_bytes_, bytes_per_line_) / bytes_per_line_;
    height += hololink::core::round_up(hololink::METADATA_SIZE, bytes_per_line_) / bytes_per_line_;

    // MGBE requires 4k alignment
    const uint32_t mgbe_coe_alignment = 4 * 1024;

    // Convert pixel format to NvSciBuf color format
    NvSciBufAttrValColorFmt color_format;
    switch (pixel_format_) {
    case csi::PixelFormat::RAW_8:
        color_format = NvSciColor_Bayer8RGGB;
        break;
    case csi::PixelFormat::RAW_10:
        color_format = NvSciColor_X2Rc10Rb10Ra10_Bayer10GBRG;
        break;
    case csi::PixelFormat::RAW_12:
        color_format = NvSciColor_Bayer16RGGB; // Using RAW16 as RAW12 doesn't exist in NvSci
        break;
    default:
        HSB_LOG_ERROR("Unsupported pixel format");
        return false;
    }

    // Setup NvSciBuf attributes
    uint32_t plane_count = 1;
    NvSciBufAttrValColorStd color_std = NvSciColorStd_REC709_ER;
    NvSciBufType buf_type = NvSciBufType_Image;
    NvSciBufAttrValImageScanType scan_type = NvSciBufScan_ProgressiveType;
    NvSciBufAttrValAccessPerm access_perm = NvSciBufAccessPerm_ReadWrite;
    NvSciBufAttrValImageLayoutType layout = NvSciBufImage_PitchLinearType;
    bool need_cpu_access = true;
    NvSciBufPeerHwEngine engine[3] = {
        { NvSciBufHwEngName_COE, NvSciBufPlatformName_Thor },
        { NvSciBufHwEngName_Display, NvSciBufPlatformName_Thor },
        { NvSciBufHwEngName_Gpu, NvSciBufPlatformName_Thor }
    };

    // Get GPU UUID for CUDA interop
    CUuuid gpu_uuid;
    CUresult cuda_result = cuDeviceGetUuid(&gpu_uuid, cuda_device_);
    if (cuda_result != CUDA_SUCCESS) {
        HSB_LOG_ERROR("Failed to get CUDA device UUID: {}", cuda_result);
        return false;
    }

    NvSciBufAttrKeyValuePair attrs[] = {
        { NvSciBufGeneralAttrKey_Types, &buf_type, sizeof(buf_type) },
        { NvSciBufImageAttrKey_Layout, &layout, sizeof(layout) },
        { NvSciBufImageAttrKey_PlaneCount, &plane_count, sizeof(plane_count) },
        { NvSciBufImageAttrKey_PlaneWidth, &width, sizeof(width) },
        { NvSciBufImageAttrKey_PlaneHeight, &height, sizeof(height) },
        { NvSciBufImageAttrKey_PlaneColorFormat, &color_format, sizeof(color_format) },
        { NvSciBufImageAttrKey_PlaneColorStd, &color_std, sizeof(color_std) },
        { NvSciBufImageAttrKey_PlaneBaseAddrAlign, &mgbe_coe_alignment, sizeof(mgbe_coe_alignment) },
        { NvSciBufImageAttrKey_PlaneScanType, &scan_type, sizeof(scan_type) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &need_cpu_access, sizeof(need_cpu_access) },
        { NvSciBufGeneralAttrKey_RequiredPerm, &access_perm, sizeof(access_perm) },
        { NvSciBufGeneralAttrKey_PeerHwEngineArray, &engine, sizeof(engine) },
        { NvSciBufGeneralAttrKey_GpuId, &gpu_uuid, sizeof(gpu_uuid) },
    };

    NvSciBufAttrList attr_list = nullptr;
    NvSciBufAttrList reconciled_attr_list = nullptr;
    NvSciBufAttrList conflict_list = nullptr;

    NvSciError err = NvSciBufAttrListCreate(sci_buf_module_, &attr_list);
    if (err != NvSciError_Success) {
        HSB_LOG_ERROR("Failed to create SciBuf attribute list");
        return false;
    }

    err = NvSciBufAttrListSetAttrs(attr_list, attrs, sizeof(attrs) / sizeof(attrs[0]));
    if (err != NvSciError_Success) {
        HSB_LOG_ERROR("Failed to set SciBuf attributes");
        NvSciBufAttrListFree(attr_list);
        return false;
    }

    err = NvSciBufAttrListReconcile(&attr_list, 1, &reconciled_attr_list, &conflict_list);
    if (err != NvSciError_Success) {
        HSB_LOG_ERROR("Failed to reconcile SciBuf attributes");
        NvSciBufAttrListFree(attr_list);
        return false;
    }

    // Allocate the buffer.
    err = NvSciBufObjAlloc(reconciled_attr_list, &buf_obj);

    NvSciBufAttrListFree(attr_list);
    NvSciBufAttrListFree(reconciled_attr_list);
    NvSciBufAttrListFree(conflict_list);

    if (err != NvSciError_Success) {
        HSB_LOG_ERROR("Failed to allocate SciBuf object");
        return false;
    }

    // Get attributes from the allocated buffer.
    NvSciBufAttrList alloc_attr_list;
    err = NvSciBufObjGetAttrList(buf_obj, &alloc_attr_list);
    if (err != NvSciError_Success) {
        HSB_LOG_ERROR("Failed to get buffer attribute list");
        return false;
    }

    NvSciBufAttrKeyValuePair alloc_attrs[] = {
        { NvSciBufImageAttrKey_Size, NULL, 0 },
    };
    size_t num_alloc_attrs = sizeof(alloc_attrs) / sizeof(alloc_attrs[0]);

    err = NvSciBufAttrListGetAttrs(alloc_attr_list, alloc_attrs, num_alloc_attrs);
    if (err != NvSciError_Success) {
        HSB_LOG_ERROR("Failed to get buffer attributes");
        return false;
    }

    size = *(static_cast<const uint64_t*>(alloc_attrs[0].value));

    return true;
}

bool FusaCoeCaptureOp::alloc_buffers()
{
    for (uint32_t i = 0; i < capture_queue_depth_; i++) {
        available_buffers_.push_back(new BufferInfo(this));

        auto buffer = available_buffers_.back();
        if (!alloc_sci_buf(buffer->sci_buf_, buffer->size_)) {
            return false;
        }

        // Map the buffer for CPU access
        NvSciError err = NvSciBufObjGetCpuPtr(buffer->sci_buf_, &buffer->cpu_ptr_);
        if (err != NvSciError_Success) {
            HSB_LOG_ERROR("Failed to map image buffer for CPU");
            return false;
        }

        // Push CUDA context before importing external memory
        CUcontext prev_context;
        CUresult cu_err = cuCtxGetCurrent(&prev_context);
        if (cu_err != CUDA_SUCCESS) {
            HSB_LOG_ERROR("Failed to get current CUDA context: {}", cu_err);
            return false;
        }

        if (prev_context != cuda_context_) {
            cu_err = cuCtxPushCurrent(cuda_context_);
            if (cu_err != CUDA_SUCCESS) {
                HSB_LOG_ERROR("Failed to push CUDA context: {}", cu_err);
                return false;
            }
        }

        // Import NvSciBuf as CUDA external memory.
        cudaExternalMemoryHandleDesc mem_handle_desc;
        memset(&mem_handle_desc, 0, sizeof(mem_handle_desc));
        mem_handle_desc.type = cudaExternalMemoryHandleTypeNvSciBuf;
        mem_handle_desc.handle.nvSciBufObject = buffer->sci_buf_;
        mem_handle_desc.size = buffer->size_;
        auto cuda_err = cudaImportExternalMemory(&buffer->cuda_ext_mem_, &mem_handle_desc);
        if (cuda_err != cudaSuccess) {
            const char* error_name = cudaGetErrorName(cuda_err);
            const char* error_string = cudaGetErrorString(cuda_err);
            HSB_LOG_ERROR("Failed to import NvSciBuf as CUDA external memory. Error {}: {} - {}",
                static_cast<int>(cuda_err),
                error_name ? error_name : "unknown",
                error_string ? error_string : "unknown");
            if (prev_context != cuda_context_) {
                cuCtxPopCurrent(&prev_context);
            }
            return false;
        }

        // Map the buffer to get CUDA device pointer
        cudaExternalMemoryBufferDesc buffer_desc = {};
        buffer_desc.offset = 0;
        buffer_desc.size = buffer->size_;
        cuda_err = cudaExternalMemoryGetMappedBuffer(
            &buffer->cuda_device_ptr_, buffer->cuda_ext_mem_, &buffer_desc);
        if (cuda_err != cudaSuccess) {
            HSB_LOG_ERROR("Failed to map CUDA device pointer: {}", static_cast<int>(cuda_err));
            if (prev_context != cuda_context_) {
                cuCtxPopCurrent(&prev_context);
            }
            return false;
        }

        // Restore previous CUDA context
        if (prev_context != cuda_context_) {
            cuCtxPopCurrent(&prev_context);
        }
    }

    return true;
}

void FusaCoeCaptureOp::free_buffers()
{
    while (available_buffers_.size() > 0) {
        BufferInfo* buffer = available_buffers_.front();
        available_buffers_.pop_front();

        cudaFree(buffer->cuda_device_ptr_);
        cudaDestroyExternalMemory(buffer->cuda_ext_mem_);

        NvSciBufObjFree(buffer->sci_buf_);

        delete buffer;
    }
}

bool FusaCoeCaptureOp::register_buffers()
{
    for (auto buffer : available_buffers_) {
        NvFusaCaptureStatus status = coe_handler_->registerBuffer(buffer->sci_buf_);
        if (status != NvFusaCaptureStatus::OK) {
            HSB_LOG_ERROR("Failed to register output buffer: {}", static_cast<int>(status));
            return false;
        }
    }

    return true;
}

void FusaCoeCaptureOp::unregister_buffers()
{
    for (auto buffer : available_buffers_) {
        coe_handler_->unregisterBuffer(buffer->sci_buf_);
    }
}

bool FusaCoeCaptureOp::issue_capture()
{
    if (available_buffers_.empty()) {
        HSB_LOG_ERROR("No available buffers to issue capture");
        return false;
    }

    // Issue capture.
    auto buffer = available_buffers_.front();
    NvFusaCaptureStatus status = coe_handler_->startCapture(buffer->sci_buf_);
    if (status != NvFusaCaptureStatus::OK) {
        HSB_LOG_ERROR("Failed to start capture: {}", static_cast<int>(status));
        return false;
    }
    available_buffers_.pop_front();

    // Add to in-flight queue.
    in_flight_captures_.push(buffer);

    return true;
}

nvidia::gxf::Expected<void> FusaCoeCaptureOp::buffer_release_callback(void* pointer)
{
    // Remove from the pending list.
    std::unique_lock<std::mutex> pending_lock(pending_buffers_mutex_);
    auto buffer = pending_buffers_[pointer];
    if (!buffer) {
        HSB_LOG_ERROR("Buffer not found in pending list ({})", pointer);
        return nvidia::gxf::ExpectedOrCode(GXF_FAILURE);
    }
    pending_buffers_.erase(pointer);
    pending_lock.unlock();

    // Add to the available queue.
    std::lock_guard<std::mutex> buffer_lock(buffer->parent_->buffer_mutex_);
    buffer->parent_->available_buffers_.push_back(buffer);

    return nvidia::gxf::Expected<void>();
}

} // namespace hololink::operators
