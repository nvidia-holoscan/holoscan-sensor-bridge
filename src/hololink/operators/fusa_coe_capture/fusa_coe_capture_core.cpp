/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "fusa_coe_capture_core.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <string>
#include <unordered_set>
#include <utility>

#include <hololink/common/cuda_error.hpp>
#include <hololink/common/cuda_helper.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/core/logging.hpp>
#include <hololink/core/networking.hpp>
#include <hololink/core/on_scope_exit.hpp>
#include <hololink/core/packetizer_program.hpp>

using namespace NvFusaCaptureExternal;

namespace hololink::operators::fusa_coe_capture {

using hololink::core::OnScopeExit;

namespace {

    // A control-plane teardown op (channel.unconfigure() / device_stop()) issues
    // a sendto() on the board's control socket, which can throw -- e.g. EBADF when
    // that (shared, stereo) control socket was closed out from under us during
    // shutdown (NvFusa's closeHandler closes an fd the control socket may have
    // reused). stop() is invoked from GXF's noexcept codelet-stop path, so letting
    // such a throw escape calls std::terminate(). Run these ops through here so a
    // teardown-time control-plane failure is logged and swallowed instead.
    template <typename F>
    void guard_teardown(const char* what, F&& fn)
    {
        try {
            fn();
        } catch (const std::exception& e) {
            HSB_LOG_ERROR("{} during teardown failed (ignored): {}", what, e.what());
        } catch (...) {
            HSB_LOG_ERROR("{} during teardown failed (ignored, unknown exception)", what);
        }
    }

    // WORKAROUND for an nvethernet defect, not a fix. nvether_coe_config() calls
    // macsec_open() during CoE bring-up, but the CoE release path never calls
    // macsec_close(), so <iface>.macsec_ns stays armed after this operator stops,
    // firing once per ~4 received packets. Left on CPU0 (the Tegra default: L4T
    // runs no irqbalance) it starves the NET_RX softirq and overflows the MAC RX
    // FIFO, corrupting the frames of whatever Linux-socket application runs next
    // on that interface until a reboot. Move it off CPU0. Match by name: the IRQ
    // number is not stable across boots.
    void relocate_orphaned_macsec_irqs(const char* cpu_list)
    {
        std::ifstream interrupts("/proc/interrupts");
        if (!interrupts) {
            HSB_LOG_WARN("Cannot read /proc/interrupts to find a macsec_ns IRQ left armed "
                         "after CoE teardown; reboot before running Linux-socket "
                         "applications on this interface");
            return;
        }
        std::string line;
        while (std::getline(interrupts, line)) {
            if (line.find(".macsec_ns") == std::string::npos) {
                continue;
            }
            const auto colon = line.find(':');
            if (colon == std::string::npos) {
                continue;
            }
            std::string irq = line.substr(0, colon);
            irq.erase(0, irq.find_first_not_of(" \t"));

            const std::string path = "/proc/irq/" + irq + "/smp_affinity_list";
            std::ofstream affinity(path);
            if (!affinity || !(affinity << cpu_list << std::flush)) {
                HSB_LOG_WARN("macsec_ns IRQ {} stays armed after CoE teardown and {} is not "
                             "writable (needs CAP_SYS_ADMIN); reboot before running "
                             "Linux-socket applications on this interface",
                    irq, path);
                continue;
            }
            HSB_LOG_WARN("macsec_ns IRQ {} stays armed after CoE teardown; moved it to CPU {}",
                irq, cpu_list);
        }
    }

} // namespace

NvSciBufAttrValColorFmt bayer_color_format(
    hololink::csi::PixelFormat pixel_format,
    hololink::csi::BayerFormat bayer_format,
    bool packetizer_enabled)
{
    using hololink::csi::BayerFormat;
    using hololink::csi::PixelFormat;

    // The request is a triple, so every rejection reports the whole triple rather than just
    // the offending member.
    const auto triple = [&] {
        return fmt::format("(bit depth {}, CFA {}, packetizer {})",
            static_cast<int>(pixel_format), static_cast<int>(bayer_format),
            packetizer_enabled ? "enabled" : "disabled");
    };

    // Range-check both enums up front. They can arrive as an arbitrary integer via the
    // Python bindings, which static_cast a plain uint32; checking here means an out-of-range
    // value is reported as such instead of being blamed on the packetizer branch it happens
    // to fall through.
    switch (pixel_format) {
    case PixelFormat::RAW_8:
    case PixelFormat::RAW_10:
    case PixelFormat::RAW_12:
        break;
    default:
        throw std::runtime_error(fmt::format(
            "Unsupported capture format {}: bit depth is not one of RAW_8/RAW_10/RAW_12",
            triple()));
    }
    switch (bayer_format) {
    case BayerFormat::RGGB:
    case BayerFormat::BGGR:
    case BayerFormat::GRBG:
    case BayerFormat::GBRG:
        break;
    default:
        throw std::runtime_error(fmt::format(
            "Unsupported capture format {}: CFA is not one of RGGB/BGGR/GRBG/GBRG", triple()));
    }

    // Selects one of the four CFA orderings within a format family. Every family below
    // provides all four, so this is total.
    const auto by_cfa = [&](NvSciBufAttrValColorFmt rggb, NvSciBufAttrValColorFmt bggr,
                            NvSciBufAttrValColorFmt grbg, NvSciBufAttrValColorFmt gbrg) {
        switch (bayer_format) {
        case BayerFormat::RGGB:
            return rggb;
        case BayerFormat::BGGR:
            return bggr;
        case BayerFormat::GRBG:
            return grbg;
        case BayerFormat::GBRG:
            return gbrg;
        }
        throw std::runtime_error(fmt::format(
            "Unsupported capture format {}: CFA is not one of RGGB/BGGR/GRBG/GBRG", triple()));
    };

    if (packetizer_enabled) {
        // Only RAW_10 and RAW_12 have packed representations, and correspondingly they are
        // the only bit depths get_packetizer_program() has a program for.
        switch (pixel_format) {
        case PixelFormat::RAW_12:
#if !defined(NVSIPL_API_MAJOR_VERSION)
#error "NVSIPL_API_MAJOR_VERSION must be defined (see cmake/sipl_api.cmake)"
#elif NVSIPL_API_MAJOR_VERSION == 1
            // SIPL API v1 (JetPack 7.1 / L4T R38): nvscibuf.h has no packed 12-bit
            // NvSciColor_Bayer12<CFA> family, so the packed data is allocated in the
            // 16-bit container it used before the 12-bit formats existed.
            return by_cfa(NvSciColor_Bayer16RGGB, NvSciColor_Bayer16BGGR,
                NvSciColor_Bayer16GRBG, NvSciColor_Bayer16GBRG);
#else
            // SIPL API v2 (JetPack 7.2 / L4T R39): proper packed 12-bit container.
            return by_cfa(NvSciColor_Bayer12RGGB, NvSciColor_Bayer12BGGR,
                NvSciColor_Bayer12GRBG, NvSciColor_Bayer12GBRG);
#endif
        case PixelFormat::RAW_10:
            return by_cfa(NvSciColor_X2Rc10Rb10Ra10_Bayer10RGGB,
                NvSciColor_X2Rc10Rb10Ra10_Bayer10BGGR,
                NvSciColor_X2Rc10Rb10Ra10_Bayer10GRBG,
                NvSciColor_X2Rc10Rb10Ra10_Bayer10GBRG);
        default:
            throw std::runtime_error(fmt::format(
                "Unsupported capture format {}: the packetizer can only be enabled for "
                "RAW_10 and RAW_12, which are the only bit depths with packed NvSciBuf "
                "formats",
                triple()));
        }
    }

    switch (pixel_format) {
    case PixelFormat::RAW_12:
        return by_cfa(NvSciColor_X4Bayer12RGGB, NvSciColor_X4Bayer12BGGR,
            NvSciColor_X4Bayer12GRBG, NvSciColor_X4Bayer12GBRG);
    case PixelFormat::RAW_10:
        return by_cfa(NvSciColor_X6Bayer10RGGB, NvSciColor_X6Bayer10BGGR,
            NvSciColor_X6Bayer10GRBG, NvSciColor_X6Bayer10GBRG);
    case PixelFormat::RAW_8:
        return by_cfa(NvSciColor_Bayer8RGGB, NvSciColor_Bayer8BGGR,
            NvSciColor_Bayer8GRBG, NvSciColor_Bayer8GBRG);
    }
    throw std::runtime_error(fmt::format(
        "Unsupported capture format {}: bit depth is not one of RAW_8/RAW_10/RAW_12",
        triple()));
}

namespace {

    // How NvSciBuf lays out a pitch-linear line of a color format bayer_color_format() can
    // return: pixels per container and bytes per container. The pitch it derives from
    // NvSciBufImageAttrKey_PlaneWidth is a whole number of containers, then aligned to the
    // HW engines' constraint (nvscibuf.h, NvSciBufImageAttrKey_PlanePitch).
    struct PixelContainer {
        uint32_t pixels;
        uint32_t bytes;
    };

    PixelContainer pixel_container(NvSciBufAttrValColorFmt color_format)
    {
        switch (color_format) {
        // 10-bit packed, 3 pixels per 32-bit container (2 padding bits).
        case NvSciColor_X2Rc10Rb10Ra10_Bayer10RGGB:
        case NvSciColor_X2Rc10Rb10Ra10_Bayer10BGGR:
        case NvSciColor_X2Rc10Rb10Ra10_Bayer10GRBG:
        case NvSciColor_X2Rc10Rb10Ra10_Bayer10GBRG:
            return { 3, 4 };
#if NVSIPL_API_MAJOR_VERSION == 1
        case NvSciColor_Bayer16RGGB:
        case NvSciColor_Bayer16BGGR:
        case NvSciColor_Bayer16GRBG:
        case NvSciColor_Bayer16GBRG:
#else
        case NvSciColor_Bayer12RGGB:
        case NvSciColor_Bayer12BGGR:
        case NvSciColor_Bayer12GRBG:
        case NvSciColor_Bayer12GBRG:
#endif
            // 12-bit packed, 2 pixels per 3 bytes. The SIPL API v1 (L4T R38) nvscibuf.h
            // has no NvSciColor_Bayer12<CFA>, so bayer_color_format() names the
            // packetized RAW_12 buffer NvSciColor_Bayer16<CFA> there, and that name
            // stands for the packed layout wherever this core deals with it: the line
            // holds 2 pixels per 3 bytes whatever the format is called. (NvSciBuf lays
            // Bayer16 out at 2 bytes per pixel, so the pitch it derives from a width
            // counted this way is wider than the stride, never narrower; the 16-bit
            // container is not supported as a capture format.)
            return { 2, 3 };
        // One pixel per 16-bit container.
        case NvSciColor_X4Bayer12RGGB:
        case NvSciColor_X4Bayer12BGGR:
        case NvSciColor_X4Bayer12GRBG:
        case NvSciColor_X4Bayer12GBRG:
        case NvSciColor_X6Bayer10RGGB:
        case NvSciColor_X6Bayer10BGGR:
        case NvSciColor_X6Bayer10GRBG:
        case NvSciColor_X6Bayer10GBRG:
            return { 1, 2 };
        // One pixel per byte.
        case NvSciColor_Bayer8RGGB:
        case NvSciColor_Bayer8BGGR:
        case NvSciColor_Bayer8GRBG:
        case NvSciColor_Bayer8GBRG:
        case NvSciColor_R8:
            return { 1, 1 };
        default:
            throw std::runtime_error(fmt::format(
                "No pixel container known for NvSciBufAttrValColorFmt {}",
                static_cast<int>(color_format)));
        }
    }

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

uint32_t FusaCoeCaptureCore::received_line_bytes(uint32_t line_bytes, bool packetizer_enabled)
{
    // With the packetizer enabled, HSB pads every line to 64 bytes once its program has
    // run (packetizer_top.sv PACK_PAD; the embedded-data lines the program bypasses are
    // padded too, which is why this takes the transmitted size rather than the pixel
    // width). With it disabled the CSI-2 stream is passed through and lines are padded
    // to 8, matching CsiToBayerOp. Using 64 unconditionally over-states both the line
    // stride and the embedded-data offset whenever the true value is not already
    // 64-aligned.
    return hololink::core::round_up(
        line_bytes, packetizer_enabled ? hololink::PacketizerProgram::LINE_PAD_BYTES : 8);
}

uint32_t FusaCoeCaptureCore::transmitted_line_bytes(
    hololink::csi::PixelFormat pixel_format, uint32_t pixel_width, bool packetizer_enabled)
{
    // What the sensor puts on CSI-2: MIPI packed.
    uint32_t csi_line_bytes = 0;
    switch (pixel_format) {
    case hololink::csi::PixelFormat::RAW_8:
        csi_line_bytes = pixel_width;
        break;
    case hololink::csi::PixelFormat::RAW_10:
        // 4 pixels per 5 bytes
        csi_line_bytes = pixel_width * 5 / 4;
        break;
    case hololink::csi::PixelFormat::RAW_12:
        // 2 pixels per 3 bytes
        csi_line_bytes = pixel_width * 3 / 2;
        break;
    default:
        throw std::runtime_error("Invalid bit depth");
    }
    if (!packetizer_enabled) {
        return csi_line_bytes;
    }

    // What HSB's packetizer makes of that line. The packetizer knows nothing of pixel
    // formats: it runs the program the channel loads for this one (the same program
    // DataChannel::configure() enables), and that program's schedule -- input beats
    // consumed and output beats emitted per loop, with a line always completing a whole
    // loop -- is what sizes the output. So the size is the program's to compute; deriving
    // it here from the pixel count instead (3 RAW_10 pixels per 4 bytes) under-states a
    // line whose CSI-2 size is not a multiple of the program's 120-byte loop, e.g. 1280
    // RAW_10 pixels: 1600 CSI-2 bytes are 14 loops, 1792 bytes, not 1708 rounded to 1728.
    const auto program = hololink::csi::get_packetizer_program(pixel_format);
    if (!program->packetizes()) {
        throw std::runtime_error(fmt::format(
            "The packetizer has no program for bit depth {}; it can only be enabled for "
            "RAW_10 and RAW_12",
            static_cast<int>(pixel_format)));
    }
    return program->get_output_size(csi_line_bytes);
}

void FusaCoeCaptureCore::request_format(
    hololink::csi::PixelFormat pixel_format,
    hololink::csi::BayerFormat bayer_format,
    bool packetizer_enabled)
{
    // Resolve now so an unrepresentable triple is rejected here rather than at allocation.
    const NvSciBufAttrValColorFmt color_format
        = bayer_color_format(pixel_format, bayer_format, packetizer_enabled);

    HSB_LOG_INFO("request_format: bit_depth={}, bayer_format={}, packetizer_enabled={} "
                 "-> NvSciBufAttrValColorFmt {}",
        static_cast<int>(pixel_format), static_cast<int>(bayer_format), packetizer_enabled,
        static_cast<int>(color_format));

    requested_pixel_format_ = pixel_format;
    pixel_format_ = pixel_format;
    bayer_format_ = bayer_format;
    packetizer_enabled_ = packetizer_enabled;
    format_requested_ = true;
}

void FusaCoeCaptureCore::configure(
    uint32_t start_byte, uint32_t received_bytes_per_line,
    uint32_t pixel_width, uint32_t pixel_height,
    hololink::csi::PixelFormat pixel_format,
    uint32_t trailing_bytes)
{
    if (!format_requested_) {
        throw std::runtime_error(
            "request_format() must be called before configure(): the capture format is "
            "(bit depth, CFA, packetizer enabled) and none of it can be inferred here");
    }
    if (pixel_format != requested_pixel_format_) {
        throw std::runtime_error(fmt::format(
            "Pixel format mismatch: request_format() asked for bit depth {} but configure() "
            "was given {}",
            static_cast<int>(requested_pixel_format_), static_cast<int>(pixel_format)));
    }

    HSB_LOG_INFO("start={}, bytes_per_line={}, width={}, height={}, format={}, trailing_bytes={}",
        start_byte, received_bytes_per_line, pixel_width, pixel_height,
        static_cast<int>(pixel_format), trailing_bytes);
    start_byte_ = start_byte;
    bytes_per_line_ = received_bytes_per_line;
    pixel_width_ = pixel_width;
    pixel_height_ = pixel_height;
    pixel_format_ = pixel_format;
    trailing_bytes_ = trailing_bytes;
    image_mode_ = true;
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
    // Opaque byte blob: no CFA, no packetizer, allocated as NvSciColor_R8.
    image_mode_ = false;
    configured_ = true;
}

void FusaCoeCaptureCore::start(
    const std::string& interface,
    const std::vector<uint8_t>& mac_addr,
    uint32_t capture_timeout_ms,
    CoeChannelConfig& channel,
    std::function<void()> device_start,
    std::function<void()> device_stop,
    std::function<bool()> frame_ready_callback,
    uint32_t stall_timeout_ms,
    std::string label)
{
    if (!configured_) {
        throw std::runtime_error("FusaCoeCaptureCore is not configured.");
    }
    if (capture_timeout_ms == 0) {
        // 0 is not just an unset sentinel -- NvFusaCaptureExternal's own
        // getCaptureStatus() contract explicitly disallows a 0 timeout, so this
        // also doubles as the "timeout must be set" check FusaCoeCaptureParams's
        // aggregate-struct defaults can't otherwise enforce.
        throw std::runtime_error("FusaCoeCaptureCore::start: capture_timeout_ms must be > 0");
    }
    frame_ready_callback_ = std::move(frame_ready_callback);
    label_ = std::move(label);

    capture_timeout_ms_ = capture_timeout_ms;
    stall_timeout_ms_ = stall_timeout_ms;
    ingress_stall_.reset();
    stalled_.store(false);
    stall_duration_ms_.store(0);
    capture_failed_.store(false);
    received_frame_count_ = 0;
    {
        std::lock_guard<std::mutex> error_lock(capture_error_mutex_);
        capture_error_message_.clear();
    }
    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        frame_taken_ = false;
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
        if (image_mode_) {
            // Always drive the packetizer explicitly -- disabled installs the null program
            // rather than relying on whatever the channel was last left in.
            channel.set_packetizer(packetizer_enabled_, pixel_format_);
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
    // stereo/failed shutdown -- so it is run through guard_teardown() (which logs
    // and swallows) rather than allowed to escape into GXF's noexcept stop().
    // Release owned resources on scope exit so a throw still tears everything down.
    // Order matters: reset the CoE handle first -- it unregisters every buffer,
    // needing both the handle and the buffers alive -- then free the buffers, then
    // close the SciBuf module.
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
        guard_teardown("channel.unconfigure()", [&] { channel.unconfigure(); });
        // The capture-device close above drives the kernel's CoE release path,
        // so the orphaned macsec_ns IRQ exists from this point on.
        relocate_orphaned_macsec_irqs("6");
    });

    guard_teardown("device_stop()", [&] { device_stop(); });
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
        if (capture_status == NvFusaCaptureStatus::TIMEOUT) {
            // A timeout is ordinary: NvFusaCapture's guidance is to pick one
            // short enough to discard a late capture rather than stall. Treating
            // it as fatal (it used to set capture_failed_) terminated the process
            // from the caller's monitor thread on any sensor hiccup. The driver
            // drops a timed-out request -- re-polling the buffer answers
            // BAD_PARAMETER -- so recycle it and let the round-robin re-issue it.

            // Ingress-stall reporting, as HSDK4's module FusaCoeCaptureOp::monitor()
            // did. An occasional empty poll is routine; a run of them means the
            // sensor has gone quiet, which downstream sees only as a pipeline that
            // stopped producing. Always warn once, early (IngressStall's own fixed
            // threshold); optionally fail the run once total silence exceeds
            // stall_timeout_ms_ (0 = disabled).
            if (ingress_stall_.on_timeout(capture_timeout_ms_)) {
                HSB_LOG_WARN("CoE capture stalled on \"{}\": no frame for {} consecutive {}ms "
                             "capture requests after {} received frames",
                    label_, ingress_stall_.consecutive_timeouts(), capture_timeout_ms_,
                    received_frame_count_);
            }
            if (stall_timeout_ms_ != 0
                && ingress_stall_.consecutive_timeout_ms() >= stall_timeout_ms_) {
                const uint64_t stalled_for_ms = ingress_stall_.consecutive_timeout_ms();
                HSB_LOG_ERROR("CoE capture stalled on \"{}\": failing after {}ms without a "
                              "frame ({} capture requests, {} received frames)",
                    label_, stalled_for_ms, ingress_stall_.consecutive_timeouts(),
                    received_frame_count_);
                stall_duration_ms_.store(stalled_for_ms);
                stalled_.store(true);
                // Only compute() can end the run; wake it (as the capture-failure
                // path below does), so that it finds no frame and fails it.
                notify_compute();
                return;
            }

            in_flight_captures_.pop();

            std::lock_guard<std::mutex> lock(buffer_mutex_);
            available_buffers_.push_back(buffer);
            while (issue_capture()) { }
            continue;
        }
        if (capture_status != NvFusaCaptureStatus::OK) {
            HSB_LOG_ERROR(
                "acquire_buffer_thread_func: getCaptureStatus returned {} on \"{}\" after "
                "{} received frames -- capture has failed",
                static_cast<int>(capture_status), label_, received_frame_count_);
            {
                std::lock_guard<std::mutex> error_lock(capture_error_mutex_);
                capture_error_message_ = fmt::format(
                    "CoE capture failed (error = {})",
                    static_cast<int>(capture_status));
            }
            capture_failed_.store(true);
            // take_acquired_buffer() only converts capture_failed_ into a thrown
            // error the next time it runs, and compute() is notification-driven (see
            // FusaCoeCaptureOp::contract()), so nothing calls it again unless we wake
            // it ourselves here. Post one notification so compute() runs once more,
            // reaches take_acquired_buffer(), and surfaces the failure instead of
            // the graph hanging silently forever.
            notify_compute();
            return;
        }
        in_flight_captures_.pop();
        ++received_frame_count_;
        if (const uint64_t stalled_requests = ingress_stall_.on_frame()) {
            HSB_LOG_WARN("CoE capture on \"{}\" resumed after {} timed-out requests",
                label_, stalled_requests);
        }

        bool notify = false;
        {
            std::lock_guard<std::mutex> lock(buffer_mutex_);

            // Keep only the newest frame: one that compute() has not taken yet is replaced
            // (and counted, see take_replaced_frame_count()), and the notification already
            // pending for it has compute() take this one instead.
            const auto stored = acquired_buffer_.store(buffer);
            if (stored.replaced) {
                available_buffers_.push_back(stored.replaced);
            }
            notify = stored.notify;

            while (issue_capture()) { }
        }

        // Only a frame stored in an empty slot needs a notification: compute() then runs
        // exactly once per frame it can take, never for nothing -- a run without a frame
        // would still count toward frame_limit.
        if (notify) {
            notify_compute();
        }
    }

    std::lock_guard<std::mutex> lock(buffer_mutex_);
    if (BufferInfo* pending = acquired_buffer_.reset()) {
        available_buffers_.push_back(pending);
    }
}

void FusaCoeCaptureCore::notify_compute()
{
    if (!frame_ready_callback_) {
        return;
    }
    // Safe unsynchronized: start()/stop()'s thread-spawn/join provide the happens-before
    // edge for frame_ready_callback_.
    //
    // This thread starts in FusaCoeCaptureOp::start(), and the run takes no notification
    // until every operator has started, nor while its capacity is full: then
    // frame_ready_callback_() returns false. Nothing else would have compute() run for this
    // frame (or stall, or failure), so retry here, off the thread running start(), until the
    // run takes it.
    while (!frame_ready_callback_() && !stop_thread_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

bool FusaCoeCaptureCore::take_acquired_buffer(BufferView& out)
{
    // Throws on a genuine capture failure (any state that is neither OK nor a
    // plain timeout).
    std::unique_lock<std::mutex> lock(buffer_mutex_);
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
    BufferInfo* buffer = acquired_buffer_.take();
    if (buffer == nullptr) {
        return false;
    }
    if (!frame_taken_) {
        // The frames this one replaced arrived while the run was starting, before it took
        // notifications: not frames lost to a pipeline that is behind the camera.
        frame_taken_ = true;
        if (const uint64_t replaced = acquired_buffer_.take_replaced()) {
            HSB_LOG_INFO("CoE capture on \"{}\": first frame taken; it replaced {} frame(s) "
                         "received while the run was starting",
                label_, replaced);
        }
    }
    buffer_in_compute_ = buffer;
    HSB_LOG_TRACE("Got buffer for \"{}\" ({})", label_, static_cast<void*>(buffer));
    lock.unlock();

    out = BufferView { buffer->cpu_ptr_, buffer->cuda_device_ptr_ };
    return true;
}

uint64_t FusaCoeCaptureCore::take_replaced_frame_count()
{
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    return frame_taken_ ? acquired_buffer_.take_replaced() : 0;
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
    // The (bit depth, CFA, packetizer) triple determines the color format outright; the blob
    // path carries no pixel structure at all.
    const NvSciBufAttrValColorFmt color_format = image_mode_
        ? bayer_color_format(pixel_format_, bayer_format_, packetizer_enabled_)
        : NvSciColor_R8;

    // The CoE engine writes the frame as one byte stream, bytes_per_line_ per row, and
    // bytes_per_line_ is what HSB sends for a line (received_line_bytes()), not what
    // pixel_width_ pixels of the color format occupy: a packetized line is a whole number of
    // the packetizer program's loops, so 1280 RAW_10 pixels arrive as 1792 bytes where
    // NvSciBuf derives 1708 from the pixel width, and whether the pitch alignment it then
    // applies happens to cover the difference is up to the HW engines on the attribute list.
    // NvSciBuf sizes the pitch from PlaneWidth alone -- the padding keys are passed through
    // to consumers and change neither the pitch nor the size (measured on Thor) -- so ask
    // for the width that spans the received line, and report the excess as right padding so
    // a consumer of the attribute list knows the active width. For the unpacketized 16-bit
    // container formats the pixel width already spans the (MIPI packed) line and there is
    // no padding.
    const PixelContainer container = pixel_container(color_format);
    const uint32_t line_width = std::max(pixel_width_,
        static_cast<uint32_t>(
            hololink::core::round_up(bytes_per_line_, container.bytes) / container.bytes)
            * container.pixels);
    uint32_t width = line_width;
    uint64_t right_padding = line_width - pixel_width_;
    uint32_t height = pixel_height_;
    height += hololink::core::round_up(start_byte_, bytes_per_line_) / bytes_per_line_;
    height += hololink::core::round_up(trailing_bytes_, bytes_per_line_) / bytes_per_line_;
    height += hololink::core::round_up(hololink::METADATA_SIZE, bytes_per_line_) / bytes_per_line_;

    const uint32_t mgbe_coe_alignment = 4 * 1024;

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
        { NvSciBufImageAttrKey_RightPadding, &right_padding, sizeof(right_padding) },
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
        { NvSciBufImageAttrKey_PlanePitch, NULL, 0 },
    };
    size_t num_alloc_attrs = sizeof(alloc_attrs) / sizeof(alloc_attrs[0]);

    err = NvSciBufAttrListGetAttrs(alloc_attr_list, alloc_attrs, num_alloc_attrs);
    if (err != NvSciError_Success) {
        HSB_LOG_ERROR("Failed to get buffer attributes");
        return false;
    }

    size = *(static_cast<const uint64_t*>(alloc_attrs[0].value));

    // The CoE engine writes rows at bytes_per_line_ stride into this buffer, but the
    // allocation size comes from the pitch NvSciBuf derives from the width and color format.
    // The width above was chosen so that pitch covers the stride; should NvSciBuf lay a
    // format out differently than pixel_container() says, the pitch would be narrower than
    // the stride and we would run off the end of the buffer, so refuse to proceed rather
    // than corrupt memory.
    const uint32_t plane_pitch = *(static_cast<const uint32_t*>(alloc_attrs[1].value));
    HSB_LOG_INFO("NvSciBuf: color_format={}, width={} (pixel_width={}, right_padding={}), "
                 "height={}, plane_pitch={}, bytes_per_line={}, size={}",
        static_cast<int>(color_format), width, pixel_width_, right_padding, height,
        plane_pitch, bytes_per_line_, size);
    if (plane_pitch < bytes_per_line_) {
        HSB_LOG_ERROR("NvSciBuf plane pitch {} is narrower than the CoE line stride {} for "
                      "color format {}; refusing to allocate an undersized buffer",
            plane_pitch, bytes_per_line_, static_cast<int>(color_format));
        return false;
    }

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
        acquired_buffer_.reset();
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
    if (BufferInfo* pending = acquired_buffer_.peek()) {
        buffers.insert(pending);
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
        guard_teardown("channel.unconfigure()", [&] { channel.unconfigure(); });
        // The capture-device close above drives the kernel's CoE release path,
        // so the orphaned macsec_ns IRQ exists from this point on.
        relocate_orphaned_macsec_irqs("6");
    });

    if (device_started && device_stop) {
        guard_teardown("device_stop()", [&] { device_stop(); });
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

void FusaCoeCaptureCore::buffer_release_callback(void* pointer)
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
        return;
    }
    pending_buffers_.erase(pointer);

    {
        std::lock_guard<std::mutex> buffer_lock(buffer->parent_->buffer_mutex_);
        buffer->parent_->available_buffers_.push_back(buffer);
    }
    buffer->parent_->buffer_available_.notify_one();
}

} // namespace hololink::operators::fusa_coe_capture
