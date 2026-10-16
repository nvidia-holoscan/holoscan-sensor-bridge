/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#ifndef SRC_HOLOLINK_OPERATORS_CPU_ROCE_TRANSCEIVER_CPU_ROCE_FRAME_RECEIVER_HPP
#define SRC_HOLOLINK_OPERATORS_CPU_ROCE_TRANSCEIVER_CPU_ROCE_FRAME_RECEIVER_HPP

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include <hololink/core/hololink.hpp>
#include <hololink/transport/roce/cpu_roce_transceiver.hpp>

namespace hololink::operators {

class CpuRoceFrameReceiver;

/**
 * One claimed RX slot.
 *
 * The payload pointer addresses the slot. For host memory it is a CPU
 * pointer. For DMA-BUF it is the caller's application-visible address:
 * it may be a host pointer or a device address, and the CPU may use it
 * only when the backing and the exporter's synchronization allow it.
 * For GPU-pointer memory it is a GPU pointer and must not be
 * dereferenced by the CPU. Destroying the frame releases the slot with
 * no synchronize function. release() runs at most once. Do not destroy
 * the receiver while get_frame() is running on it.
 *
 * payload_size() is the configured slot capacity, not a count of bytes the
 * completion wrote. dropped(), overwritten(), invalid_immediates(), and
 * completion_count() are cumulative since start(), snapshotted when this
 * frame was published. dropped() counts unread frames replaced by a newer
 * one. overwritten() counts completions for a slot that was still leased.
 * It is a lower bound: a write that completes after release() is not counted.
 * invalid_immediates() counts completions whose 12-bit slot was outside the
 * ring. completion_count()
 * counts every accepted completion, published or not, so a gap between
 * frames is a completion that was not handed out. psn() is the low 20 bits
 * of the HSB packet sequence number from the immediate, or empty when that
 * completion had none.
 *
 * The lease does not stop the NIC. A later write can change held bytes
 * before that completion is seen. A completion for a leased slot counts an
 * overwrite, does not publish, and leaves the receiver running. The next
 * published frame carries that count. dropped(), overwritten(), and
 * invalid_immediates() saturate at UINT32_MAX. Size slot_count so a held
 * frame does not span a full lap of the ring.
 */
class HsbFrame {
public:
    HsbFrame(const HsbFrame&) = delete;
    HsbFrame& operator=(const HsbFrame&) = delete;
    HsbFrame(HsbFrame&& other) noexcept;
    HsbFrame& operator=(HsbFrame&& other) noexcept;
    ~HsbFrame();

    std::uint8_t* payload() const
    {
        return payload_;
    }
    std::size_t payload_size() const
    {
        return payload_size_;
    }
    std::uint32_t slot() const
    {
        return slot_;
    }
    const std::optional<std::uint32_t>& immediate_data() const
    {
        return immediate_data_;
    }
    std::uint32_t dropped() const
    {
        return dropped_;
    }
    std::uint32_t overwritten() const
    {
        return overwritten_;
    }
    std::uint32_t invalid_immediates() const
    {
        return invalid_immediates_;
    }
    std::uint64_t completion_count() const
    {
        return completion_count_;
    }
    // Wire frame number rebased onto this receiver when the frame has a
    // trailer. The first such frame is 1. Later frames add the 16-bit wire
    // delta, including frames get_frame never returned. Zero when
    // metadata_size is 0 or host_trailer is false.
    std::uint32_t session_frame_number() const
    {
        return session_frame_number_;
    }
    std::optional<std::uint32_t> psn() const
    {
        if (!immediate_data_) {
            return std::nullopt;
        }
        return (*immediate_data_ >> 12) & 0xFFFFF;
    }
    const std::optional<hololink::Hololink::FrameMetadata>& metadata() const
    {
        return metadata_;
    }
    std::chrono::system_clock::time_point received_at() const
    {
        return received_at_;
    }

    /**
     * Run synchronize on this thread with no lock held, at most once.
     * An empty function means the caller already synchronized. Success
     * clears the lease. A throw stores that error, wakes get_frame(), closes
     * the transceiver, leaves the slot leased, and rethrows. A failing
     * release blocks until the poller finishes its current
     * batch and the queue pair and memory regions are torn down. On failure
     * the payload owner is kept for the life of the process; a caller who
     * owns the memory directly must not free it until the GPU is done.
     */
    void release(std::function<void()> synchronize = {});

private:
    friend class CpuRoceFrameReceiver;
    struct SlotSync;

    // arm_wake bumps the futex word. wake_waiters wakes a parked get_frame().
    static void arm_wake(SlotSync& slots);
    static void wake_waiters(SlotSync& slots);
    static void begin_stats(SlotSync& slots);
    static void end_stats(SlotSync& slots);
    static void discard_mailbox(SlotSync& slots);

    HsbFrame(std::uint8_t* payload, std::size_t payload_size, std::uint32_t slot,
        std::optional<std::uint32_t> immediate_data, std::uint32_t dropped,
        std::uint32_t overwritten, std::uint32_t invalid_immediates,
        std::uint64_t completion_count, std::uint32_t session_frame_number,
        std::optional<hololink::Hololink::FrameMetadata> metadata,
        std::chrono::system_clock::time_point received_at,
        std::shared_ptr<SlotSync> slots);

    std::uint8_t* payload_ = nullptr;
    std::size_t payload_size_ = 0;
    std::uint32_t slot_ = 0;
    std::optional<std::uint32_t> immediate_data_;
    std::uint32_t dropped_ = 0;
    std::uint32_t overwritten_ = 0;
    std::uint32_t invalid_immediates_ = 0;
    std::uint64_t completion_count_ = 0;
    std::uint32_t session_frame_number_ = 0;
    std::optional<hololink::Hololink::FrameMetadata> metadata_;
    std::chrono::system_clock::time_point received_at_ {};
    std::shared_ptr<SlotSync> slots_;
};

/**
 * Latest-frame receiver on CpuRoceTransceiver.
 * Experimental interface; it may change between releases. The stable
 * contract is hololink/transport/roce/cpu_roce_transceiver.hpp.
 *
 * The caller keeps the payload memory alive for the life of the receiver
 * and of every HsbFrame taken from it, unless payload_owner is set. That
 * owner is held with the slot state, so a held frame keeps those bytes
 * alive after the receiver is destroyed. After a failed release that owner
 * is kept for the life of the process, because GPU completion is unknown.
 * The receiver does not allocate
 * the payload. metadata_size is 0 or hololink::METADATA_SIZE. The sensor
 * still writes that tail. The host reads it only when host_trailer is set,
 * which is the default. The HSB metadata packet must then be enabled or
 * that trailer is stale.
 * Host memory is read in place. DMA-BUF and GPU-pointer trailers are written
 * to host memory when umr_metadata is set and the NIC accepts an indirect
 * key, and copied to the host on a nonblocking stream otherwise.
 * cuda_context is a CUcontext the poller makes current, and it must stay
 * valid until stop() returns.
 * With host_trailer on a device payload, that address must be a CUDA device
 * pointer in that context, because the copy remains the fallback. The
 * immediate PSN and the metadata PSN are both left on the frame.
 *
 * stride() is the page size passed to configure_roce(). Immediate data uses
 * the 12-bit slot field, so the caller must use HSB IP 0x2602 or newer.
 * receive_credit_count is the number of zero-SGE receive WQEs. It is
 * independent of slot_count. The default is 256.
 *
 * connect() brings the queue pair up. start() asynchronously launches the
 * poller and calls connect() first when that has not been done. Both are
 * control-thread only. A second start() while the poller is running returns
 * quietly.
 * After quiesce(), stop(), or a stored failure(), connect() and start()
 * throw. qp_number(), rkey(), and path_mtu() are valid after connect(),
 * including after quiesce().
 * quiesce() stops the poller, discards a queued frame, and wakes get_frame()
 * to return empty. The queue pair and the memory registration stay up until
 * stop() or destruction. stop() quiesces and then closes.
 * get_frame() has one caller at a time. frames_ready() is true when a frame
 * is waiting, failure() is set, or quiesce() or stop() has been called.
 * nullopt from get_frame() means timeout, quiesce, stop, or a stored
 * failure(). An immediate whose
 * 12-bit slot is outside the ring increments invalid_immediates() and the
 * credit is reposted. The receiver does not log. It does not set
 * failure() for that immediate. A completion with no immediate does set
 * failure(), as does release()'s synchronize function when it throws. An
 * overwrite does not set it. dropped(), overwritten(), and
 * invalid_immediates() saturate at UINT32_MAX. qp_number(), rkey(), path_mtu(), and
 * external_frame_memory() reject a receiver whose failure() is set, and
 * after stop().
 */
class CpuRoceFrameReceiver {
public:
    enum class ReceiveError { Overwrite,
        InvalidImmediate };
    // Called on the poller, outside the slot lock, only when overwritten()
    // or invalid_immediates() actually incremented. total is that saturating
    // uint32 widened to uint64. The function must not throw or call this
    // receiver's quiesce(), stop(), or destructor. It may block, and a
    // blocking call delays error-path polling. A call that won the
    // lock just before quiesce() can still run while quiesce() joins the
    // poller. No call happens after quiesce() returns, so any context must
    // outlive that return. This hook is not part of the CUDA-Q transport.
    using ReceiveErrorHook = void (*)(void* context, ReceiveError error,
        std::uint64_t total) noexcept;

    struct Config {
        std::string ibv_name;
        std::uint32_t ibv_port = 1;
        std::string local_ip;
        hololink::transport::roce::CpuRoceTransceiver::Peer peer;
        // Caller-registered memory. Ignored when cuda_payload is set.
        hololink::transport::roce::PayloadMemory payload_memory;
        // A CUDA allocation the receiver registers. host_mapped uses host
        // memory. Otherwise the receiver exports a DMA-BUF and owns that
        // file descriptor, and falls back to the raw GPU pointer. The
        // transport itself stays CUDA-free. cuda_context is required for
        // the export and for a device trailer.
        struct CudaPayload {
            void* address = nullptr;
            std::size_t size = 0;
            bool host_mapped = false;
        };
        std::optional<CudaPayload> cuda_payload;
        std::size_t payload_size = 0;
        std::uint32_t slot_count = 0;
        // Zero-SGE receive WQEs. Independent of slot_count. Range 1..4096.
        std::uint32_t receive_credit_count = 256;
        std::size_t metadata_size = 0;
        std::optional<int> poll_cpu;
        // When set, the transceiver tries poll_cpu on the poller thread and
        // keeps the current mask if the affinity read or the pin fails.
        // Otherwise that failure is fatal, which is what transmit and
        // CUDA-Q keep.
        bool poll_cpu_best_effort = false;
        // Program a 4096-byte receive path MTU. Off uses the port active MTU.
        bool rx_path_mtu_4096 = false;
        // Keeps payload_memory alive with the slot state when set.
        std::shared_ptr<void> payload_owner;
        // CUcontext. Required when host_trailer is set and the payload is
        // DMA-BUF or a GPU pointer. Unused for host memory. The caller keeps
        // it valid until stop() returns. With host_trailer on, the payload
        // address must be a CUDA device pointer in this context.
        void* cuda_context = nullptr;
        ReceiveErrorHook on_receive_error = nullptr;
        void* on_receive_error_context = nullptr;
        // Ask an mlx5 NIC to write DMA-BUF and GPU-pointer trailers to host
        // memory through an indirect key. Used only when host_trailer is set.
        // False keeps the device copy. umr_active() reports which path is in
        // use.
        bool umr_metadata = true;
        // Read the trailer on the host. The default keeps the current read
        // when metadata_size is 128. False leaves the tail in the slot: no
        // indirect key and no copy. True uses umr_metadata to choose the key
        // or the device copy. metadata_size 0 has nothing to read. Appended
        // last so a positional Config initializer keeps its existing fields.
        bool host_trailer = true;
        // Called once on the poller thread before it polls. Null does nothing.
        // The receiver does not log and does not interpret the hook.
        // Appended after host_trailer so a positional Config initializer keeps
        // its existing fields.
        using PollThreadStartHook = void (*)(void* context) noexcept;
        PollThreadStartHook on_poll_thread_start = nullptr;
        void* on_poll_thread_start_context = nullptr;
    };

    explicit CpuRoceFrameReceiver(Config config);
    ~CpuRoceFrameReceiver();

    CpuRoceFrameReceiver(const CpuRoceFrameReceiver&) = delete;
    CpuRoceFrameReceiver& operator=(const CpuRoceFrameReceiver&) = delete;

    void connect();
    void start();
    // Join the poller and stop handing out frames. The queue pair and
    // memory registration stay up. This is not VerbsResources::quiesce(),
    // which releases the queue pair.
    void quiesce();
    void stop();

    std::uint32_t qp_number() const;
    std::uint32_t rkey() const;
    // Payload bytes of the path MTU programmed on the queue pair. Valid
    // after connect(), including after quiesce(), until stop(). With
    // rx_path_mtu_4096 this is 4096 even when the interface is smaller.
    // The HSB packet size must still be capped by the interface MTU.
    std::uint32_t path_mtu() const;
    std::uint64_t external_frame_memory() const;
    std::size_t stride() const
    {
        return stride_;
    }

    /**
     * True while the indirect key is in use and the NIC writes device-ring
     * trailers to host memory. False on the device-copy path, on a host
     * ring, and when metadata_size is 0. Throws unless the receiver is
     * live, same as qp_number().
     */
    bool umr_active() const;

    /**
     * Return the newest completed frame.
     *
     * The receiver keeps at most one frame waiting. A newer frame replaces
     * any older waiting frame and increases dropped(). If a completion
     * arrives for a slot whose previously returned HsbFrame has not been
     * released or destroyed, that slot's bytes have already been overwritten.
     * The completion is not returned and overwritten() increases.
     *
     * A returned frame holds its slot until release or destruction. A zero
     * timeout checks once. A positive timeout waits for a frame, failure,
     * quiesce, stop, or timeout. nullopt means no frame is available.
     */
    std::optional<HsbFrame> get_frame(std::chrono::microseconds timeout);
    bool frames_ready() const;
    std::exception_ptr failure() const;
    // One coherent snapshot of frames the poller accepted, including frames
    // get_frame never returned. frame_number, completions, rx_bytes, and psn
    // are updated together after the trailer is parsed. With metadata_size 0,
    // or with host_trailer false, the host does not read a trailer, so
    // frame_number, rx_bytes, and psn stay 0.
    // completions is the full count. dropped, overwritten, and
    // invalid_immediates saturate at UINT32_MAX. Narrowing
    // belongs to the caller.
    struct Stats {
        std::uint64_t rx_bytes = 0;
        std::uint64_t completions = 0;
        std::uint32_t frame_number = 0;
        std::uint32_t psn = 0;
        std::uint32_t dropped = 0;
        std::uint32_t overwritten = 0;
        std::uint32_t invalid_immediates = 0;
    };
    Stats stats() const;

    std::uint32_t dropped() const;
    std::uint32_t overwritten() const;
    std::uint32_t invalid_immediates() const;
    std::uint64_t completion_count() const;

private:
    friend struct CpuRoceFrameReceiverTestAccess;
    struct DeviceTrailerCopy;

    void on_rx_completion(std::optional<std::uint32_t> immediate_data);
    void publish(std::uint32_t slot, std::optional<std::uint32_t> immediate_data);
    void note_trailer_locked(const hololink::Hololink::FrameMetadata& wire);
    void monitor();
    void store_failure(std::exception_ptr failure);
    [[noreturn]] void fail(std::exception_ptr failure);
    std::uint8_t* slot_payload(std::uint32_t slot) const;
    bool facts_ready() const;

    Config config_;
    std::size_t stride_ = 0;
    std::size_t metadata_offset_ = 0;
    bool device_trailer_ = false;
    // Cached after start(), before the monitor thread. Null means copy.
    const std::uint8_t* host_trailers_ = nullptr;
    std::unique_ptr<DeviceTrailerCopy> device_trailer_copy_;
    // Declared before the transceiver so the descriptor outlives close().
    struct OwnedDmaBufFd {
        int fd = -1;
        ~OwnedDmaBufFd();
        OwnedDmaBufFd() = default;
        OwnedDmaBufFd(const OwnedDmaBufFd&) = delete;
        OwnedDmaBufFd& operator=(const OwnedDmaBufFd&) = delete;
    };
    OwnedDmaBufFd dmabuf_;
    std::shared_ptr<hololink::transport::roce::CpuRoceTransceiver> transceiver_;
    std::shared_ptr<HsbFrame::SlotSync> slots_;
    std::thread monitor_;
    bool connected_ = false;
    bool polling_ = false;
    bool quiesced_ = false;

    void report_receive_error(ReceiveError error, std::uint32_t total) const;
};

} // namespace hololink::operators

#endif /* SRC_HOLOLINK_OPERATORS_CPU_ROCE_TRANSCEIVER_CPU_ROCE_FRAME_RECEIVER_HPP */
