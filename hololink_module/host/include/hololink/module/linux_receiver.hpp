/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_LINUX_RECEIVER_HPP
#define HOLOLINK_MODULE_LINUX_RECEIVER_HPP

#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>

#include "enumeration_metadata.hpp"
#include "module.hpp"
#include "service.hpp"
#include "status.h"

namespace hololink::module {

/* Per-frame info populated by LinuxReceiverInterfaceV1::get_next_frame.
 * The V1 Linux operator reads these fields after each frame arrives.
 *
 * Unlike the hardware RoCE path — where the FPGA writes an end-of-frame
 * metadata block into device memory that the operator copies back and
 * decodes via FrameMetadataInterfaceV1::decode — the software receiver
 * reassembles HSB's RoCEv2 UDP stream in user space and decodes the
 * per-frame metadata as it goes. So the decoded fields are returned
 * here directly and the operator stamps them without a device-side
 * read or a separate decode step.
 *
 * It is an output struct the receiver fills, so — like
 * RoceReceiverFrameInfoV1 — the struct form is acceptable: a missing
 * field shows up at the consumer the same way a missing tuple element
 * would. */
class LinuxReceiverFrameInfoV1 {
public:
    uint64_t frame_memory = 0; // CUdeviceptr to the frame's pixel data

    // Receiver-side per-frame accounting (the software receiver
    // surfaces packet / byte counts the kernel-bypass hardware path
    // doesn't).
    uint32_t frame_packets_received = 0;
    uint32_t frame_bytes_received = 0;
    uint32_t received_frame_number = 0;
    uint32_t frame_number = 0; // 32-bit extended frame counter
    uint32_t imm_data = 0;
    int64_t received_s = 0;
    int64_t received_ns = 0;
    uint64_t packets_dropped = 0;

    // Frame metadata the receiver decoded from the incoming stream
    // (the fields of the legacy Hololink::FrameMetadata the operator
    // stamps onto its outbound map).
    uint32_t flags = 0;
    uint32_t psn = 0;
    uint32_t crc = 0;
    uint32_t timestamp_ns = 0;
    uint64_t timestamp_s = 0;
    uint64_t bytes_written = 0;
    uint32_t metadata_ns = 0;
    uint64_t metadata_s = 0;
};

/* Per-channel software receiver. Where RoceReceiverInterfaceV1 fronts a
 * hardware ibverbs QP, this reassembles HSB's RoCEv2 UDP packets in user
 * space over an ordinary datagram socket — so it needs no infiniband
 * device and links no ibverbs. It still produces a software-emulated
 * qp_number / rkey and still drives the FPGA through configure_roce, so
 * it sits behind its own transport view of the channel
 * (LinuxDataChannelInterfaceV1) exactly as RoCE does.
 *
 * Every supplement publishes a LinuxReceiverInterfaceV1 per (serial,
 * data_channel) under the same "serial=<serial_number>;data_channel=<n>"
 * instance_id it uses for the matching LinuxDataChannelInterfaceV1 (and
 * the RoCE pair). Per-board / per-FPGA-revision behavior diffs ship as
 * module overrides at the service-locator level.
 *
 * Lifecycle is operator-driven:
 *   - The supplement publishes the receiver on first lookup.
 *   - The operator creates a datagram socket and binds it to the data
 *     plane via LinuxDataChannelInterfaceV1::configure_socket(fd).
 *   - The operator calls start(fd, ...) which constructs the underlying
 *     receiver over that socket + the CUDA buffer and starts its
 *     run() thread.
 *   - The operator hands the receiver to
 *     LinuxDataChannelInterfaceV1::attach_receiver(receiver), which
 *     reads get_qp_number / get_rkey / local_port / frame_size /
 *     page_size / pages off it and programs the FPGA.
 *   - The operator's worker thread runs blocking_monitor() until
 *     close() fires; compute() calls get_next_frame() to pull frames.
 */
class LinuxReceiverInterfaceV1
    : public ConfigurableService<LinuxReceiverInterfaceV1> {
public:
    static constexpr const char* type_id = "linux_receiver.v1";

    /* Per-channel instance_id derivation used by the metadata-form
     * get_service. Throws if the metadata is missing 'serial_number'
     * or 'data_channel'. Same shape as the RoCE receiver / channel. */
    static std::string locator_id(const EnumerationMetadata& metadata)
    {
        return "serial=" + metadata.get<std::string>("serial_number")
            + ";data_channel=" + std::to_string(metadata.get<int64_t>("data_channel"));
    }

    virtual ~LinuxReceiverInterfaceV1() = default;

    /* Construct the underlying software receiver over `data_socket`
     * (already bound by LinuxDataChannelInterfaceV1::configure_socket)
     * and the CUDA buffer, then start its run() thread. Parameters are
     * passed individually (rather than bundled into a config struct) so
     * the compiler forces the caller to supply every one. There is no
     * ibv_name / ibv_port / peer_ip — the socket is the transport, and
     * HSB targets the socket's bound local_port; the receiver's
     * received_address_offset is cu_buffer (HSB writes from address 0
     * and the software receiver adds the local offset). */
    virtual hololink_module_status_t start(
        int data_socket,
        uint64_t cu_buffer,
        size_t cu_buffer_size,
        size_t cu_frame_size,
        size_t cu_page_size,
        unsigned pages,
        unsigned queue_size)
        = 0;

    /* Signal the run() thread to exit; safe to call from any thread
     * (typically the operator's stop()). Pairs with start(); a fresh
     * start(...) after close() rebuilds the underlying receiver. */
    virtual void close() = 0;

    /* Receiver worker-thread body (legacy LinuxReceiver::run()). Blocks
     * until close() is called. Run on a dedicated thread (the
     * operator's monitor thread) that also carries the CUDA context the
     * receiver was started with. */
    virtual void blocking_monitor() = 0;

    /* Block up to `timeout_ms` for the next complete frame. On success
     * fills `info` with the frame's device pointer + per-frame
     * statistics + decoded metadata; returns false on timeout.
     *
     * `cuda_stream` is a CUstream passed as an opaque pointer so this
     * module header carries no <cuda.h> dependency (mirroring the way
     * frame_memory is a uint64_t rather than a CUdeviceptr). The
     * software receiver issues the host→device copy of the reassembled
     * frame on this stream so it overlaps with downstream pipeline
     * work; the operator obtains it from the Holoscan execution
     * context. */
    virtual bool get_next_frame(unsigned timeout_ms,
        LinuxReceiverFrameInfoV1& info, void* cuda_stream)
        = 0;

    /* Returns false if get_next_frame may block. */
    virtual bool frames_ready() = 0;

    /* Register a callback the monitor thread invokes each time a new
     * frame becomes available. The operator uses this to wake an
     * AsynchronousCondition instead of blocking in get_next_frame, so a
     * stalled leg can't starve the scheduler (required when multiple
     * receivers feed a single downstream join). Pass an empty
     * std::function to clear. Set before blocking_monitor() runs. */
    virtual void set_frame_ready(std::function<void()> callback) = 0;

    /* QP wire-up + frame-layout values the channel reads in
     * LinuxDataChannelInterfaceV1::attach_receiver(receiver) to program
     * the FPGA. Populated after start(...) returns. local_port is the
     * bound socket's UDP port (where HSB sends); there is no
     * external_frame_memory() — the Linux channel always passes 0 for
     * the device frame-memory address. */
    virtual uint32_t get_qp_number() = 0;
    virtual uint32_t get_rkey() = 0;
    virtual uint32_t local_port() = 0;
    virtual size_t frame_size() = 0;
    virtual size_t page_size() = 0;
    virtual unsigned pages() = 0;
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_LINUX_RECEIVER_HPP
