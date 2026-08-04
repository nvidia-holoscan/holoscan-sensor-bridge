/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <hololink/core/csi_controller.hpp>

#include <NvFusaCaptureExternal.hpp>
#include <nvscibuf.h>

#include <cuda.h>
#include <holoscan/holoscan.hpp>

namespace hololink::operators::fusa_coe_capture {

struct CoeFrameMetadata {
    int64_t timestamp_s = 0;
    int64_t timestamp_ns = 0;
    int64_t metadata_s = 0;
    int64_t metadata_ns = 0;
    int64_t crc = 0;
    int64_t frame_number = 0;
};

/**
 * Resolve the (bit depth, CFA, packetizer) triple to the one NvSciBuf color format that
 * correctly describes the captured buffer.
 *
 * The packetizer is an independent control: when it is enabled HSB repacks the CSI-2 stream
 * into the right-justified layouts NvSciBuf calls "packed", and those exist only at 10 and 12
 * bits. When it is disabled the stream stays CSI-2 and the buffer is described by the
 * corresponding unpacked container format.
 *
 *   bit depth | packetizer | color format
 *   ----------+------------+----------------------------------------
 *   RAW_12    | enabled    | NvSciColor_Bayer12<CFA>
 *   RAW_12    | disabled   | NvSciColor_X4Bayer12<CFA>
 *   RAW_10    | enabled    | NvSciColor_X2Rc10Rb10Ra10_Bayer10<CFA>
 *   RAW_10    | disabled   | NvSciColor_X6Bayer10<CFA>
 *   RAW_8     | enabled    | rejected -- no packed 8-bit format exists
 *   RAW_8     | disabled   | NvSciColor_Bayer8<CFA>
 *
 * Throws std::runtime_error for any triple with no correct representation.
 */
NvSciBufAttrValColorFmt bayer_color_format(
    hololink::csi::PixelFormat pixel_format,
    hololink::csi::BayerFormat bayer_format,
    bool packetizer_enabled);

class CoeChannelConfig {
public:
    virtual ~CoeChannelConfig() = default;

    virtual void set_packetizer(
        bool enabled, hololink::csi::PixelFormat pixel_format)
        = 0;
    virtual void configure_coe(
        uint8_t channel, size_t frame_size, uint32_t bytes_per_line)
        = 0;
    virtual void unconfigure() = 0;
};

class CoeMetadataDecoder {
public:
    virtual ~CoeMetadataDecoder() = default;

    virtual bool decode(
        const void* host_memory, size_t host_memory_size, CoeFrameMetadata& out) const = 0;
};

class FusaCoeCaptureCore {
public:
    static uint32_t receiver_start_byte();

    /**
     * Bytes from the start of one received line to the start of the next. HSB pads lines to
     * 64 bytes when the packetizer is repacking them, and to 8 bytes when it is passing
     * CSI-2 through untouched, so this depends on the packetizer just as
     * transmitted_line_bytes() does.
     */
    uint32_t received_line_bytes(uint32_t line_bytes) const;

    /**
     * State the capture format as a whole: bit depth, sensor CFA, and whether the HSB
     * packetizer is enabled. These three are independent controls -- the packetizer is not
     * implied by the bit depth -- and together they determine exactly one NvSciBuf color
     * format. An unrepresentable combination throws here, at request time, rather than
     * failing later during buffer allocation.
     *
     * Must be called before transmitted_line_bytes(), configure() or start(): the line
     * geometry depends on the packetizer state.
     */
    void request_format(hololink::csi::PixelFormat pixel_format,
        hololink::csi::BayerFormat bayer_format,
        bool packetizer_enabled);

    /**
     * Bytes on the wire for one line. Depends on the packetizer: enabled, HSB emits the
     * NvSciBuf packed layouts; disabled, the stream stays CSI-2 MIPI packed.
     */
    uint32_t transmitted_line_bytes(
        hololink::csi::PixelFormat pixel_format, uint32_t pixel_width) const;

    void configure(uint32_t start_byte, uint32_t received_bytes_per_line,
        uint32_t pixel_width, uint32_t pixel_height,
        hololink::csi::PixelFormat pixel_format,
        uint32_t trailing_bytes);
    void configure_frame_size(uint32_t frame_size_bytes);

    bool is_configured() const { return configured_; }

    uint32_t start_byte() const { return start_byte_; }
    uint32_t bytes_per_line() const { return bytes_per_line_; }
    uint32_t pixel_width() const { return pixel_width_; }
    uint32_t pixel_height() const { return pixel_height_; }
    hololink::csi::PixelFormat pixel_format() const { return pixel_format_; }
    hololink::csi::BayerFormat bayer_format() const { return bayer_format_; }
    bool packetizer_enabled() const { return packetizer_enabled_; }

    void start(
        const std::string& interface,
        const std::vector<uint8_t>& mac_addr,
        uint32_t capture_timeout_ms,
        CoeChannelConfig& channel,
        std::function<void()> device_start,
        std::function<void()> device_stop);

    void stop(
        CoeChannelConfig& channel,
        std::function<void()> device_stop);

    struct BufferView {
        void* cpu_ptr = nullptr;
        void* cuda_device_ptr = nullptr;
    };

    // Blocks until a frame is acquired or the timeout elapses.
    //   returns true  -> a buffer was acquired; `out` is populated.
    //   returns false -> timed out with no frame; not an error, the caller may
    //                    simply retry (e.g. skip a tick, or loop in a monitor).
    //   throws         -> the capture path failed, i.e. any state that is neither
    //                    OK nor a plain timeout. This is deliberately fatal: a
    //                    caller that lets it escape terminates the process,
    //                    surfacing the failure loudly (and landing on the fault
    //                    under a debugger) rather than stalling silently.
    bool wait_for_acquired_buffer(
        uint32_t timeout_ms, const char* out_tensor_name, BufferView& out);

    size_t metadata_offset() const;

    bool decode_metadata(
        const BufferView& buffer,
        CoeMetadataDecoder& decoder,
        CoeFrameMetadata& out) const;

    void register_pending_output(void* tensor_device_ptr);

    static nvidia::gxf::Expected<void> buffer_release_callback(void* pointer);

private:
    bool alloc_buffers();
    bool alloc_sci_buf(NvSciBufObj& buf_obj, size_t& size);
    void free_buffers();
    bool register_buffers();
    void unregister_buffers(NvFusaCaptureExternal::INvFusaCaptureCoeHandler* handler);
    bool issue_capture();
    void acquire_buffer_thread_func();
    void rollback_failed_start(
        CoeChannelConfig& channel,
        const std::function<void()>& device_stop,
        bool device_started);

    // NvSciBuf module handle: closed on destruction. Declared before the buffers
    // it allocates so it outlives them.
    struct SciBufModuleDeleter {
        using pointer = NvSciBufModule;
        void operator()(NvSciBufModule module) const;
    };
    using UniqueSciBufModule = std::unique_ptr<NvSciBufModule, SciBufModuleDeleter>;

    // RAII owner for the CoE channel handle. On reset/destruction it unregisters
    // every buffer (which needs both the handle and the buffers alive) and then
    // closes. The buffer storage it depends on must therefore be declared before
    // coe_handler_ so it is destroyed after it.
    class CoeChannel {
    public:
        CoeChannel() = default;
        CoeChannel(FusaCoeCaptureCore* parent,
            NvFusaCaptureExternal::INvFusaCaptureCoeHandler* handler)
            : parent_(parent)
            , handler_(handler)
        {
        }
        ~CoeChannel() { reset(); }

        CoeChannel(CoeChannel&& other) noexcept
            : parent_(other.parent_)
            , handler_(other.handler_)
        {
            other.handler_ = nullptr;
        }
        CoeChannel& operator=(CoeChannel&& other) noexcept
        {
            if (this != &other) {
                reset();
                parent_ = other.parent_;
                handler_ = other.handler_;
                other.handler_ = nullptr;
            }
            return *this;
        }
        CoeChannel(const CoeChannel&) = delete;
        CoeChannel& operator=(const CoeChannel&) = delete;

        void reset();
        NvFusaCaptureExternal::INvFusaCaptureCoeHandler* operator->() const { return handler_; }

    private:
        FusaCoeCaptureCore* parent_ = nullptr;
        NvFusaCaptureExternal::INvFusaCaptureCoeHandler* handler_ = nullptr;
    };

    const uint32_t capture_queue_depth_ = 4;

    uint32_t start_byte_ = 0;
    uint32_t bytes_per_line_ = 0;
    uint32_t pixel_width_ = 0;
    uint32_t pixel_height_ = 0;
    uint32_t trailing_bytes_ = 0;
    hololink::csi::PixelFormat pixel_format_ = hololink::csi::PixelFormat::RAW_8;
    hololink::csi::BayerFormat bayer_format_ = hololink::csi::BayerFormat::RGGB;
    // Whether the HSB packetizer repacks the CSI-2 stream. Independent of bit depth and of
    // image_mode_; set only by request_format().
    bool packetizer_enabled_ = false;
    hololink::csi::PixelFormat requested_pixel_format_ = hololink::csi::PixelFormat::RAW_8;
    bool format_requested_ = false;
    // True for image capture (configure()), false for the opaque byte-blob path
    // (configure_frame_size()). Says nothing about the packetizer.
    bool image_mode_ = false;
    bool configured_ = false;

    CUcontext cuda_context_ = nullptr;
    CUdevice cuda_device_ = 0;

    UniqueSciBufModule sci_buf_module_;

    NvFusaCaptureExternal::CoeSettings coe_settings_ = { 0 };

    std::thread acquire_thread_;
    std::atomic<bool> stop_thread_ { false };
    std::atomic<bool> capture_failed_ { false };
    std::mutex capture_error_mutex_;
    std::string capture_error_message_;
    std::mutex buffer_mutex_;
    std::condition_variable buffer_acquired_;
    std::condition_variable buffer_available_;

    struct BufferInfo {
        BufferInfo(FusaCoeCaptureCore* parent, uint32_t index)
            : parent_(parent)
            , index_(index)
        {
        }
        ~BufferInfo();
        BufferInfo(const BufferInfo&) = delete;
        BufferInfo& operator=(const BufferInfo&) = delete;

        NvSciBufObj sci_buf_ = nullptr;
        size_t size_ = 0;
        void* cpu_ptr_ = nullptr;
        cudaExternalMemory_t cuda_ext_mem_ = nullptr;
        void* cuda_device_ptr_ = nullptr;

        FusaCoeCaptureCore* parent_ = nullptr;
        uint32_t index_ = 0;
    };

    // Sole owner of the BufferInfo objects; the containers below hold non-owning
    // views. Declared after sci_buf_module_ (so the module outlives the buffers)
    // and before coe_handler_ (so the buffers outlive the handle that unregisters
    // them on close).
    std::vector<std::unique_ptr<BufferInfo>> buffer_storage_;

    std::deque<BufferInfo*> available_buffers_;
    std::queue<BufferInfo*> in_flight_captures_;
    BufferInfo* acquired_buffer_ = nullptr;
    BufferInfo* buffer_in_compute_ = nullptr;

    std::unordered_set<BufferInfo*> collect_all_buffer_infos();

    // Erase only this instance's entries from the shared pending_buffers_ map;
    // in stereo mode siblings share it, so a blanket clear() would drop their
    // pending buffers. Caller must hold pending_buffers_mutex_.
    void clear_pending_buffers_locked();

    uint32_t next_reissue_index_ = 0;
    uint32_t capture_timeout_ms_ = 0;

    // Count of frames successfully received (acquired) by the acquire thread.
    // Touched only from that thread, so no synchronization is needed.
    uint64_t received_frame_count_ = 0;

    // Declared last so it is destroyed first: on reset it unregisters every buffer
    // (needs the handle and buffer_storage_ both alive) before closing the channel.
    CoeChannel coe_handler_;

    static std::map<void*, BufferInfo*> pending_buffers_;
    static std::mutex pending_buffers_mutex_;
};

} // namespace hololink::operators::fusa_coe_capture
