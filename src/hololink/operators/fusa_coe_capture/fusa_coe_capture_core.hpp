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
#include <hololink/core/ingress_stall.hpp>
#include <hololink/core/newest_frame_slot.hpp>

#include <NvFusaCaptureExternal.hpp>
#include <nvscibuf.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace hololink::operators::fusa_coe_capture {

struct CoeFrameMetadata {
    int64_t timestamp_s = 0;
    int64_t timestamp_ns = 0;
    int64_t metadata_s = 0;
    int64_t metadata_ns = 0;
    int64_t crc = 0;
    int64_t frame_number = 0;
    int64_t bytes_written = 0;
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
 * Built against the SIPL API v1 (JetPack 7.1 / L4T R38) headers, whose nvscibuf.h has no
 * NvSciColor_Bayer12<CFA>, packetized RAW_12 is named NvSciColor_Bayer16<CFA> instead and
 * that name means the packed 12-bit layout throughout this core (NVSIPL_API_MAJOR_VERSION).
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
     * 64 bytes when the packetizer is enabled (whether or not it repacked the line: the
     * embedded-data lines it bypasses are padded too), and to 8 bytes when it is passing
     * CSI-2 through untouched, so this depends on the packetizer just as
     * transmitted_line_bytes() does. Static (and packetizer_enabled taken explicitly rather
     * than read from request_format() state) so a geometry-capture CsiConverter adapter can
     * call it before any FusaCoeCaptureCore instance exists -- see
     * hololink::operators::FusaCoeCaptureOp::make_geometry_adapter().
     */
    static uint32_t received_line_bytes(uint32_t line_bytes, bool packetizer_enabled);

    /**
     * State the capture format as a whole: bit depth, sensor CFA, and whether the HSB
     * packetizer is enabled. These three are independent controls -- the packetizer is not
     * implied by the bit depth -- and together they determine exactly one NvSciBuf color
     * format. An unrepresentable combination throws here, at request time, rather than
     * failing later during buffer allocation.
     *
     * Must be called before configure() or start().
     */
    void request_format(hololink::csi::PixelFormat pixel_format,
        hololink::csi::BayerFormat bayer_format,
        bool packetizer_enabled);

    /**
     * Bytes on the wire for one line, before received_line_bytes()'s padding. Depends on
     * the packetizer: disabled, the stream stays CSI-2 MIPI packed; enabled, HSB runs the
     * pixel format's packetizer program (hololink::csi::get_packetizer_program()) on the
     * CSI-2 line, and the program's schedule sizes the output -- a whole number of its
     * loops, so a line may carry padding pixels before the 64-byte pad. Static for the
     * same reason as received_line_bytes() above.
     */
    static uint32_t transmitted_line_bytes(
        hololink::csi::PixelFormat pixel_format, uint32_t pixel_width, bool packetizer_enabled);

    /**
     * Image capture mode: state the CSI geometry (start byte, line stride, dimensions,
     * pixel format, trailing bytes) for a real image. Mutually exclusive with
     * configure_frame_size() below -- whichever is called last wins (sets image_mode_
     * accordingly). Requires request_format() to have been called first (throws
     * otherwise): the capture format it states there feeds this call's own validation
     * and the eventual NvSciBuf color-format resolution.
     */
    void configure(uint32_t start_byte, uint32_t received_bytes_per_line,
        uint32_t pixel_width, uint32_t pixel_height,
        hololink::csi::PixelFormat pixel_format,
        uint32_t trailing_bytes);

    /**
     * Opaque byte-blob capture mode: state a flat frame_size_bytes payload with no CSI
     * image structure -- no CFA, no packetizer, allocated as plain NvSciColor_R8.
     * Mutually exclusive with configure() above. Does not require request_format()
     * first (the blob carries no (bit depth, CFA, packetizer) triple to validate).
     * See FusaCoeCaptureParams::frame_size_bytes / examples/fusa_coe_test_sensor_receiver.cpp
     * for the sensor-side use case this exists for.
     */
    void configure_frame_size(uint32_t frame_size_bytes);

    bool is_configured() const { return configured_; }

    uint32_t start_byte() const { return start_byte_; }
    uint32_t bytes_per_line() const { return bytes_per_line_; }
    uint32_t pixel_width() const { return pixel_width_; }
    uint32_t pixel_height() const { return pixel_height_; }
    hololink::csi::PixelFormat pixel_format() const { return pixel_format_; }
    hololink::csi::BayerFormat bayer_format() const { return bayer_format_; }
    bool packetizer_enabled() const { return packetizer_enabled_; }

    // frame_ready_callback: called from the acquire thread to have compute() run once more:
    // when it stores a frame and compute() has taken the one before (see
    // hololink::core::NewestFrameSlot), so that compute() runs once per frame it can take,
    // and once to wake compute() after a stall or a capture failure. Returns true once the
    // notification is durably handled (posted, or failed for a reason no retry would fix);
    // false means "not accepted yet, call again shortly" -- the acquire thread retries in
    // that case, until stop(), since nothing else will re-signal that frame.
    //
    // label: names this capture (e.g. the operator's name, capture_0) in its log
    // messages, so a stall in a stereo run says which camera stopped.
    void start(
        const std::string& interface,
        const std::vector<uint8_t>& mac_addr,
        uint32_t capture_timeout_ms,
        CoeChannelConfig& channel,
        std::function<void()> device_start,
        std::function<void()> device_stop,
        std::function<bool()> frame_ready_callback = {},
        uint32_t stall_timeout_ms = 0,
        std::string label = {});

    void stop(
        CoeChannelConfig& channel,
        std::function<void()> device_stop);

    struct BufferView {
        void* cpu_ptr = nullptr;
        void* cuda_device_ptr = nullptr;
    };

    // Takes the newest acquired frame, without waiting: compute() runs once per frame it
    // can take (see frame_ready_callback above), so a frame is there unless the capture
    // has stalled.
    //   returns true  -> a buffer was taken; `out` is populated.
    //   returns false -> no frame: the acquire thread woke compute() to report a stall
    //                    (stalled(), below).
    //   throws         -> the capture path failed, i.e. any state that is neither
    //                    OK nor a plain timeout. This is deliberately fatal: a
    //                    caller that lets it escape terminates the process,
    //                    surfacing the failure loudly (and landing on the fault
    //                    under a debugger) rather than stalling silently.
    bool take_acquired_buffer(BufferView& out);

    // Frames the acquire thread replaced with a newer one before compute() took them,
    // since the last call: frames lost because compute() did not run before the next
    // frame arrived. Those that arrived before compute() first ran, while the run was
    // starting, are not counted.
    uint64_t take_replaced_frame_count();

    // True once the acquire thread has declared a hard stall (stall_timeout_ms
    // of consecutive empty capture-status polls, if that was set nonzero in
    // start()). Set at most once per start()/stop() cycle; the caller should
    // check this whenever take_acquired_buffer() returns false, and fail
    // its compute() if it is true -- only compute() can end the run, so the
    // acquire thread cannot do it directly.
    bool stalled() const { return stalled_.load(); }
    uint64_t stall_duration_ms() const { return stall_duration_ms_.load(); }

    size_t metadata_offset() const;

    bool decode_metadata(
        const BufferView& buffer,
        CoeMetadataDecoder& decoder,
        CoeFrameMetadata& out) const;

    void register_pending_output(void* tensor_device_ptr);

    // Called (via cudaLaunchHostFunc) after the device copy of the buffer data
    // completes. Returns the buffer to the available pool and re-arms the capture.
    static void buffer_release_callback(void* pointer);

private:
    bool alloc_buffers();
    bool alloc_sci_buf(NvSciBufObj& buf_obj, size_t& size);
    void free_buffers();
    bool register_buffers();
    void unregister_buffers(NvFusaCaptureExternal::INvFusaCaptureCoeHandler* handler);
    bool issue_capture();
    void acquire_buffer_thread_func();
    // Has compute() run once more (frame_ready_callback_), retrying until the run takes the
    // notification or stop() is called.
    void notify_compute();
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
    // The newest acquired frame, waiting for compute(); guarded by buffer_mutex_.
    hololink::core::NewestFrameSlot<BufferInfo> acquired_buffer_;
    BufferInfo* buffer_in_compute_ = nullptr;
    // Whether compute() has taken a frame since start(); guarded by buffer_mutex_.
    bool frame_taken_ = false;

    std::unordered_set<BufferInfo*> collect_all_buffer_infos();

    // Erase only this instance's entries from the shared pending_buffers_ map;
    // in stereo mode siblings share it, so a blanket clear() would drop their
    // pending buffers. Caller must hold pending_buffers_mutex_.
    void clear_pending_buffers_locked();

    uint32_t next_reissue_index_ = 0;
    uint32_t capture_timeout_ms_ = 0;
    std::function<bool()> frame_ready_callback_;
    // Set by start() before the acquire thread is spawned; read-only afterwards.
    std::string label_;

    // Count of frames successfully received (acquired) by the acquire thread.
    // Touched only from that thread, so no synchronization is needed.
    uint64_t received_frame_count_ = 0;

    // Ingress-stall detection (see stalled()/stall_duration_ms() above). Both
    // ingress_stall_ and stall_timeout_ms_ are touched only from the acquire
    // thread; stalled_/stall_duration_ms_ are written once from that thread
    // and read from the caller of take_acquired_buffer(), hence atomic.
    hololink::core::IngressStall ingress_stall_;
    uint32_t stall_timeout_ms_ = 0;
    std::atomic<bool> stalled_ { false };
    std::atomic<uint64_t> stall_duration_ms_ { 0 };

    // Declared last so it is destroyed first: on reset it unregisters every buffer
    // (needs the handle and buffer_storage_ both alive) before closing the channel.
    CoeChannel coe_handler_;

    static std::map<void*, BufferInfo*> pending_buffers_;
    static std::mutex pending_buffers_mutex_;
};

} // namespace hololink::operators::fusa_coe_capture
