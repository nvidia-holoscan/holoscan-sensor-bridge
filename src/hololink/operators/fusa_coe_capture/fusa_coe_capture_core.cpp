/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "fusa_coe_capture_core.hpp"

#include <algorithm>
#include <chrono>
#include <unordered_set>
#include <utility>

#include <hololink/common/cuda_error.hpp>
#include <hololink/common/cuda_helper.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/core/logging.hpp>
#include <hololink/core/networking.hpp>

using namespace NvFusaCaptureExternal;

namespace hololink::operators::fusa_coe_capture {

namespace {

    // Runs a cleanup action on scope exit, including during stack unwinding, so an
    // exception from a step deliberately left outside it still triggers teardown.
    template <typename F>
    class OnScopeExit {
    public:
        explicit OnScopeExit(F fn)
            : fn_(std::move(fn))
        {
        }
        ~OnScopeExit() { fn_(); }
        OnScopeExit(const OnScopeExit&) = delete;
        OnScopeExit& operator=(const OnScopeExit&) = delete;

    private:
        F fn_;
    };

} // namespace

std::map<void*, FusaCoeCaptureCore::BufferInfo*> FusaCoeCaptureCore::pending_buffers_;
std::mutex FusaCoeCaptureCore::pending_buffers_mutex_;

void FusaCoeCaptureCore::SciBufModuleDeleter::operator()(NvSciBufModule module) const
{
    NvSciBufModuleClose(module);
}

void FusaCoeCaptureCore::CoeChannel::reset()
{
    if (handler_) {
        parent_->unregister_buffers(handler_);
        INvFusaCapture::closeHandler(handler_);
        handler_ = nullptr;
    }
}

FusaCoeCaptureCore::BufferInfo::~BufferInfo()
{
    if (cuda_device_ptr_) {
        cudaFree(cuda_device_ptr_);
    }
    if (cuda_ext_mem_) {
        cudaDestroyExternalMemory(cuda_ext_mem_);
    }
    if (sci_buf_) {
        NvSciBufObjFree(sci_buf_);
    }
}

uint32_t FusaCoeCaptureCore::receiver_start_byte()
{
    return 0;
}

uint32_t FusaCoeCaptureCore::received_line_bytes(uint32_t line_bytes)
{
    return hololink::core::round_up(line_bytes, 64);
}

uint32_t FusaCoeCaptureCore::transmitted_line_bytes(
    hololink::csi::PixelFormat pixel_format, uint32_t pixel_width)
{
    switch (pixel_format) {
    case hololink::csi::PixelFormat::RAW_8:
        return pixel_width;
    case hololink::csi::PixelFormat::RAW_10:
        return ((pixel_width + 2) / 3) * 4;
    case hololink::csi::PixelFormat::RAW_12:
        return ((pixel_width + 1) / 2) * 3;
    default:
        throw std::runtime_error("Invalid bit depth");
    }
}

void FusaCoeCaptureCore::configure(
    uint32_t start_byte, uint32_t received_bytes_per_line,
    uint32_t pixel_width, uint32_t pixel_height,
    hololink::csi::PixelFormat pixel_format,
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
    csi_pixel_format_ = true;
    configured_ = true;
}

void FusaCoeCaptureCore::configure_frame_size(uint32_t frame_size_bytes)
{
    if (frame_size_bytes == 0) {
        throw std::runtime_error("frame_size_bytes must be > 0");
    }
    HSB_LOG_INFO("configure: frame_size_bytes={}", frame_size_bytes);
    start_byte_ = 0;
    bytes_per_line_ = frame_size_bytes;
    pixel_width_ = frame_size_bytes;
    pixel_height_ = 1;
    pixel_format_ = hololink::csi::PixelFormat::RAW_8;
    trailing_bytes_ = 0;
    csi_pixel_format_ = false;
    configured_ = true;
}

void FusaCoeCaptureCore::start(
    const std::string& interface,
    const std::vector<uint8_t>& mac_addr,
    uint32_t capture_timeout_ms,
    CoeChannelConfig& channel,
    std::function<void()> device_start,
    std::function<void()> device_stop)
{
    if (!configured_) {
        throw std::runtime_error("FusaCoeCaptureCore is not configured.");
    }

    capture_timeout_ms_ = capture_timeout_ms;
    capture_failed_.store(false);
    received_frame_count_ = 0;
    {
        std::lock_guard<std::mutex> error_lock(capture_error_mutex_);
        capture_error_message_.clear();
    }

    bool device_started = false;

    try {
        CudaCheck(cuInit(0));
        CudaCheck(cuDeviceGet(&cuda_device_, 0));
        CudaCheck(cuDevicePrimaryCtxRetain(&cuda_context_, cuda_device_));

        NvSciBufModule sci_buf_module = nullptr;
        if (NvSciBufModuleOpen(&sci_buf_module) != NvSciError_Success) {
            throw std::runtime_error("Failed to open NvSciBuf module");
        }
        sci_buf_module_ = UniqueSciBufModule(sci_buf_module);

        if (!alloc_buffers()) {
            throw std::runtime_error("Failed to allocate buffers");
        }

        coe_settings_.if_name = interface.c_str();
        memcpy(coe_settings_.sensor_mac_addr,
            mac_addr.data(),
            sizeof(coe_settings_.sensor_mac_addr));

        INvFusaCaptureCoeHandler* coe_handler = nullptr;
        auto status = INvFusaCapture::openCoeHandler(coe_settings_, &coe_handler);
        if (status != NvFusaCaptureStatus::OK || !coe_handler) {
            throw std::runtime_error(
                fmt::format("Failed to open NvFusa CoE Handler (error = {}, handle = {})",
                    static_cast<int>(status), static_cast<void*>(coe_handler)));
        }
        coe_handler_ = CoeChannel(this, coe_handler);

        HSB_LOG_INFO("Opened NvFusaCapture channel {}", coe_handler_->getChannelNumber());

        if (!register_buffers()) {
            throw std::runtime_error("Failed to register output buffers");
        }

        const uint32_t frame_size = start_byte_ + (pixel_height_ * bytes_per_line_) + trailing_bytes_;
        if (csi_pixel_format_) {
            channel.set_packetizer_if_needed(csi_pixel_format_, pixel_format_);
        }
        channel.configure_coe(
            coe_handler_->getChannelNumber(), frame_size, bytes_per_line_);

        device_start();
        device_started = true;

        while (available_buffers_.size() > 0) {
            if (!issue_capture()) {
                throw std::runtime_error("Failed to issue capture request");
            }
        }

        stop_thread_.store(false);
        acquire_thread_ = std::thread(&FusaCoeCaptureCore::acquire_buffer_thread_func, this);
    } catch (...) {
        rollback_failed_start(channel, device_stop, device_started);
        throw;
    }
}

void FusaCoeCaptureCore::stop(
    CoeChannelConfig& channel,
    std::function<void()> device_stop)
{
    if (acquire_thread_.joinable()) {
        stop_thread_.store(true);
        buffer_available_.notify_all();
        acquire_thread_.join();
    }

    // device_stop() is control-plane teardown (a sendto to the board) and can
    // throw -- e.g. EBADF when the control socket was already torn down during a
    // stereo/failed shutdown. Release owned resources on scope exit so that throw
    // still tears everything down (and then propagates). Order matters: reset the
    // CoE handle first -- it unregisters every buffer, needing both the handle and
    // the buffers alive -- then free the buffers, then close the SciBuf module.
    OnScopeExit release([&] {
        coe_handler_.reset();
        {
            std::lock_guard<std::mutex> pending_lock(pending_buffers_mutex_);
            clear_pending_buffers_locked();
        }
        free_buffers();
        sci_buf_module_.reset();
        // start() retains the primary context on every call, so release it here
        // (after free_buffers has dropped the CUDA resources that need it alive)
        // or the retain count leaks across repeated start/stop cycles.
        if (cuda_context_ != nullptr) {
            cuDevicePrimaryCtxRelease(cuda_device_);
            cuda_context_ = nullptr;
        }
        channel.unconfigure();
    });

    device_stop();
}

void FusaCoeCaptureCore::acquire_buffer_thread_func()
{
    while (!stop_thread_.load()) {
        if (in_flight_captures_.empty()) {
            std::unique_lock<std::mutex> lock(buffer_mutex_);
            buffer_available_.wait(lock, [this] {
                return stop_thread_.load() || issue_capture();
            });
            continue;
        }

        auto buffer = in_flight_captures_.front();
        auto capture_status = coe_handler_->getCaptureStatus(
            buffer->sci_buf_, capture_timeout_ms_);
        if (capture_status != NvFusaCaptureStatus::OK) {
            HSB_LOG_DEBUG(
                "acquire_buffer_thread_func: getCaptureStatus returned {} after "
                "{} received frames",
                static_cast<int>(capture_status), received_frame_count_);
            {
                std::lock_guard<std::mutex> error_lock(capture_error_mutex_);
                capture_error_message_ = fmt::format(
                    "CoE capture failed (error = {})",
                    static_cast<int>(capture_status));
            }
            capture_failed_.store(true);
            stop_thread_.store(true);
            buffer_available_.notify_all();
            buffer_acquired_.notify_all();
            return;
        }
        in_flight_captures_.pop();
        ++received_frame_count_;

        std::lock_guard<std::mutex> lock(buffer_mutex_);

        if (acquired_buffer_) {
            available_buffers_.push_back(acquired_buffer_);
        }
        acquired_buffer_ = buffer;
        buffer_acquired_.notify_all();

        while (issue_capture()) { }
    }

    std::lock_guard<std::mutex> lock(buffer_mutex_);
    if (acquired_buffer_) {
        available_buffers_.push_back(acquired_buffer_);
        acquired_buffer_ = nullptr;
    }
}

bool FusaCoeCaptureCore::wait_for_acquired_buffer(
    uint32_t timeout_ms, const char* out_tensor_name, BufferView& out)
{
    // Throws on a genuine capture failure (any state that is neither OK nor a
    // plain timeout). A plain timeout returns false; only an acquired frame
    // returns true (with `out` populated).
    auto throw_if_capture_failed = [this]() {
        if (capture_failed_.load()) {
            std::string error_message;
            {
                std::lock_guard<std::mutex> error_lock(capture_error_mutex_);
                error_message = capture_error_message_;
            }
            throw std::runtime_error(error_message.empty()
                    ? "CoE capture failed"
                    : error_message);
        }
    };

    std::unique_lock<std::mutex> lock(buffer_mutex_);
    throw_if_capture_failed();
    BufferInfo* buffer = acquired_buffer_;
    while (buffer == nullptr) {
        throw_if_capture_failed();
        HSB_LOG_TRACE("Waiting for buffer for output \"{}\" (timeout = {}ms)...",
            out_tensor_name, timeout_ms);
        auto wait_status = buffer_acquired_.wait_for(
            lock, std::chrono::milliseconds(timeout_ms));
        // Re-check under the lock: a frame may have landed right at the timeout
        // boundary, and the wait can also wake spuriously.
        buffer = acquired_buffer_;
        if (buffer == nullptr && wait_status == std::cv_status::timeout) {
            // No frame within the window; not an error, the caller may retry.
            throw_if_capture_failed();
            HSB_LOG_TRACE("Timed out waiting for buffer for output \"{}\"",
                out_tensor_name);
            return false;
        }
    }
    acquired_buffer_ = nullptr;
    buffer_in_compute_ = buffer;
    HSB_LOG_TRACE("Got buffer for output \"{}\" ({})",
        out_tensor_name, static_cast<void*>(buffer));
    lock.unlock();

    out = BufferView { buffer->cpu_ptr_, buffer->cuda_device_ptr_ };
    return true;
}

void FusaCoeCaptureCore::register_pending_output(void* tensor_device_ptr)
{
    std::lock_guard<std::mutex> pending_lock(pending_buffers_mutex_);
    pending_buffers_[tensor_device_ptr] = buffer_in_compute_;
    buffer_in_compute_ = nullptr;
}

size_t FusaCoeCaptureCore::metadata_offset() const
{
    size_t offset = start_byte_ + (pixel_height_ * bytes_per_line_) + trailing_bytes_;
    return hololink::core::round_up(offset, hololink::METADATA_SIZE);
}

bool FusaCoeCaptureCore::decode_metadata(
    const BufferView& buffer,
    CoeMetadataDecoder& decoder,
    CoeFrameMetadata& out) const
{
    const auto* metadata_ptr = static_cast<const uint8_t*>(buffer.cpu_ptr) + metadata_offset();
    return decoder.decode(metadata_ptr, hololink::METADATA_SIZE, out);
}

bool FusaCoeCaptureCore::alloc_sci_buf(NvSciBufObj& buf_obj, size_t& size)
{
    uint32_t width = pixel_width_;
    uint32_t height = pixel_height_;
    height += hololink::core::round_up(start_byte_, bytes_per_line_) / bytes_per_line_;
    height += hololink::core::round_up(trailing_bytes_, bytes_per_line_) / bytes_per_line_;
    height += hololink::core::round_up(hololink::METADATA_SIZE, bytes_per_line_) / bytes_per_line_;

    const uint32_t mgbe_coe_alignment = 4 * 1024;

    NvSciBufAttrValColorFmt color_format;
    if (csi_pixel_format_) {
        switch (pixel_format_) {
        case hololink::csi::PixelFormat::RAW_8:
            color_format = NvSciColor_Bayer8RGGB;
            break;
        case hololink::csi::PixelFormat::RAW_10:
            color_format = NvSciColor_X2Rc10Rb10Ra10_Bayer10GBRG;
            break;
        case hololink::csi::PixelFormat::RAW_12:
            color_format = NvSciColor_Bayer16RGGB;
            break;
        default:
            HSB_LOG_ERROR("Unsupported pixel format");
            return false;
        }
    } else {
        color_format = NvSciColor_R8;
    }

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

    NvSciError err = NvSciBufAttrListCreate(sci_buf_module_.get(), &attr_list);
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

    err = NvSciBufObjAlloc(reconciled_attr_list, &buf_obj);

    NvSciBufAttrListFree(attr_list);
    NvSciBufAttrListFree(reconciled_attr_list);
    NvSciBufAttrListFree(conflict_list);

    if (err != NvSciError_Success) {
        HSB_LOG_ERROR("Failed to allocate SciBuf object");
        return false;
    }

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

bool FusaCoeCaptureCore::alloc_buffers()
{
    for (uint32_t i = 0; i < capture_queue_depth_; i++) {
        buffer_storage_.push_back(std::make_unique<BufferInfo>(this, i));
        BufferInfo* buffer = buffer_storage_.back().get();
        available_buffers_.push_back(buffer);
        if (!alloc_sci_buf(buffer->sci_buf_, buffer->size_)) {
            return false;
        }

        NvSciError err = NvSciBufObjGetCpuPtr(buffer->sci_buf_, &buffer->cpu_ptr_);
        if (err != NvSciError_Success) {
            HSB_LOG_ERROR("Failed to map image buffer for CPU");
            return false;
        }

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

        if (prev_context != cuda_context_) {
            cuCtxPopCurrent(&prev_context);
        }
    }

    return true;
}

void FusaCoeCaptureCore::free_buffers()
{
    // Drop the non-owning views first, then the storage that owns them; each
    // BufferInfo frees its CUDA and SciBuf resources in its destructor. Clearing
    // storage outside the lock keeps the CUDA/SciBuf frees off the buffer mutex.
    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        available_buffers_.clear();
        in_flight_captures_ = std::queue<BufferInfo*>();
        acquired_buffer_ = nullptr;
        buffer_in_compute_ = nullptr;
    }
    buffer_storage_.clear();
}

void FusaCoeCaptureCore::clear_pending_buffers_locked()
{
    for (auto it = pending_buffers_.begin(); it != pending_buffers_.end();) {
        if (it->second && it->second->parent_ == this) {
            it = pending_buffers_.erase(it);
        } else {
            ++it;
        }
    }
}

std::unordered_set<FusaCoeCaptureCore::BufferInfo*> FusaCoeCaptureCore::collect_all_buffer_infos()
{
    std::unordered_set<BufferInfo*> buffers;

    {
        std::lock_guard<std::mutex> pending_lock(pending_buffers_mutex_);
        for (const auto& entry : pending_buffers_) {
            // Only this instance's pending buffers; the map is shared with
            // sibling cores in stereo mode.
            if (entry.second && entry.second->parent_ == this) {
                buffers.insert(entry.second);
            }
        }
    }

    std::lock_guard<std::mutex> lock(buffer_mutex_);
    for (auto* buffer : available_buffers_) {
        buffers.insert(buffer);
    }
    auto in_flight = in_flight_captures_;
    while (!in_flight.empty()) {
        buffers.insert(in_flight.front());
        in_flight.pop();
    }
    if (acquired_buffer_) {
        buffers.insert(acquired_buffer_);
    }
    if (buffer_in_compute_) {
        buffers.insert(buffer_in_compute_);
    }

    return buffers;
}

void FusaCoeCaptureCore::rollback_failed_start(
    CoeChannelConfig& channel,
    const std::function<void()>& device_stop,
    bool device_started)
{
    if (acquire_thread_.joinable()) {
        stop_thread_.store(true);
        buffer_available_.notify_all();
        buffer_acquired_.notify_all();
        acquire_thread_.join();
    }

    // Same ordered teardown as stop(), plus the primary CUDA context. Runs on
    // scope exit so a throw from device_stop() still releases everything.
    OnScopeExit release([&] {
        coe_handler_.reset();
        {
            std::lock_guard<std::mutex> pending_lock(pending_buffers_mutex_);
            clear_pending_buffers_locked();
        }
        free_buffers();
        sci_buf_module_.reset();
        if (cuda_context_ != nullptr) {
            cuDevicePrimaryCtxRelease(cuda_device_);
            cuda_context_ = nullptr;
        }
        channel.unconfigure();
    });

    if (device_started && device_stop) {
        device_stop();
    }
}

bool FusaCoeCaptureCore::register_buffers()
{
    // NvFuSa requires startCapture to be called in the same FIFO order as
    // registerBuffer. Iterate available_buffers_ directly (allocation order
    // 0..N-1) so the registration order matches issue_capture's index order.
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    for (auto* buffer : available_buffers_) {
        NvFusaCaptureStatus reg_status = coe_handler_->registerBuffer(buffer->sci_buf_);
        if (reg_status != NvFusaCaptureStatus::OK) {
            HSB_LOG_ERROR("Failed to register output buffer: {}", static_cast<int>(reg_status));
            return false;
        }
    }

    return true;
}

void FusaCoeCaptureCore::unregister_buffers(INvFusaCaptureCoeHandler* handler)
{
    for (auto* buffer : collect_all_buffer_infos()) {
        handler->unregisterBuffer(buffer->sci_buf_);
    }
}

bool FusaCoeCaptureCore::issue_capture()
{
    auto it = std::find_if(available_buffers_.begin(), available_buffers_.end(),
        [this](const BufferInfo* buf) { return buf->index_ == next_reissue_index_; });

    if (it == available_buffers_.end()) {
        return false;
    }

    auto buffer = *it;
    available_buffers_.erase(it);
    NvFusaCaptureStatus capture_status = coe_handler_->startCapture(buffer->sci_buf_);
    if (capture_status != NvFusaCaptureStatus::OK) {
        available_buffers_.push_back(buffer);
        HSB_LOG_ERROR(
            "Failed to start capture: {}", static_cast<int>(capture_status));
        return false;
    }

    in_flight_captures_.push(buffer);
    next_reissue_index_ = (next_reissue_index_ + 1) % capture_queue_depth_;

    return true;
}

nvidia::gxf::Expected<void> FusaCoeCaptureCore::buffer_release_callback(void* pointer)
{
    // Hold pending_buffers_mutex_ across the whole handoff. Teardown clears
    // pending_buffers_ under this same lock before free_buffers() destroys
    // buffer_storage_, so keeping it held makes the pending-clear a barrier: a
    // callback with a live buffer completes before teardown can proceed, and one
    // arriving after the clear finds nothing and bails -- neither derefs a freed
    // BufferInfo. Lock order is pending -> buffer everywhere, so no deadlock.
    std::lock_guard<std::mutex> pending_lock(pending_buffers_mutex_);
    auto buffer = pending_buffers_[pointer];
    if (!buffer) {
        HSB_LOG_ERROR("Buffer not found in pending list ({})", pointer);
        return nvidia::gxf::ExpectedOrCode(GXF_FAILURE);
    }
    pending_buffers_.erase(pointer);

    {
        std::lock_guard<std::mutex> buffer_lock(buffer->parent_->buffer_mutex_);
        buffer->parent_->available_buffers_.push_back(buffer);
    }
    buffer->parent_->buffer_available_.notify_one();

    return nvidia::gxf::Expected<void>();
}

} // namespace hololink::operators::fusa_coe_capture
