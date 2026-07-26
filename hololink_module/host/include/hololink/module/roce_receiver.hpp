/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_ROCE_RECEIVER_HPP
#define HOLOLINK_MODULE_ROCE_RECEIVER_HPP

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

/* Per-frame info populated by RoceReceiverInterfaceV1::get_next_frame.
 * The V1 RoCE operator reads these fields after each frame arrives. */
class RoceReceiverFrameInfoV1 {
public:
    uint64_t frame_memory = 0; // CUdeviceptr to the frame's pixel data
    uint64_t metadata_memory = 0; // CUdeviceptr to the 48-byte EOF block
    uint64_t received_frame_number = 0;
    uint32_t frame_number = 0;
    uint32_t imm_data = 0;
    uint64_t received_s = 0;
    uint64_t received_ns = 0;
    uint64_t rx_write_requests = 0;
    uint32_t dropped = 0;
};

/* Per-channel ibverbs receiver. Every supplement publishes a
 * RoceReceiverInterfaceV1 per (serial, data_channel) under the same
 * "serial=<serial_number>;data_channel=<n>" instance_id it uses for
 * the matching RoceDataChannelInterfaceV1. Per-board /
 * per-FPGA-revision behavior diffs ship as module overrides at the
 * service-locator level.
 *
 * Lifecycle is operator-driven:
 *   - The supplement publishes the receiver on first lookup.
 *   - The operator calls start(...) with the runtime parameters,
 *     bringing the QP up.
 *   - The operator hands the receiver to
 *     RoceDataChannelInterfaceV1::attach_receiver(receiver), which
 *     reads get_qp_number / get_rkey / external_frame_memory /
 *     frame_size / page_size / pages off it and programs the FPGA.
 *   - The operator's worker thread runs blocking_monitor() until
 *     close() fires; compute() calls get_next_frame() to pull frames.
 */
class RoceReceiverInterfaceV1
    : public ConfigurableService<RoceReceiverInterfaceV1> {
public:
    static constexpr const char* type_id = "roce_receiver.v1";

    /* Per-channel instance_id derivation used by the metadata-form
     * get_service. Throws if the metadata is missing 'serial_number'
     * or 'data_channel'. */
    static std::string locator_id(const EnumerationMetadata& metadata)
    {
        return "serial=" + metadata.get<std::string>("serial_number")
            + ";data_channel=" + std::to_string(metadata.get<int64_t>("data_channel"));
    }

    virtual ~RoceReceiverInterfaceV1() = default;

    /* Bring up the underlying ibverbs receiver. Parameters are passed
     * individually (rather than bundled into a config struct) so the
     * compiler forces the caller to supply every one — a missing
     * struct field would silently zero-initialize and surface only at
     * runtime. */
    virtual hololink_module_status_t start(
        const std::string& ibv_name,
        unsigned ibv_port,
        uint64_t cu_buffer,
        size_t cu_buffer_size,
        size_t cu_frame_size,
        size_t cu_page_size,
        unsigned pages,
        size_t metadata_offset,
        const std::string& peer_ip,
        unsigned queue_size)
        = 0;

    /* Signal the monitor thread to exit; safe to call from any thread
     * (typically the operator's stop()). Pairs with start(); a fresh
     * start(...) after close() rebuilds the underlying receiver. */
    virtual void close() = 0;

    /* Receiver worker-thread body. Blocks until close() is called.
     * Run on a dedicated thread (the operator's monitor thread) that
     * also carries the CUDA context the receiver was started with. */
    virtual void blocking_monitor() = 0;

    /* Block up to `timeout_ms` for the next complete frame. On success
     * fills `info` with the frame's device pointers + per-frame
     * statistics; returns false on timeout. */
    virtual bool get_next_frame(unsigned timeout_ms,
        RoceReceiverFrameInfoV1& info)
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
     * RoceDataChannelInterfaceV1::attach_receiver(receiver) to
     * program the FPGA. Populated after start(...) returns. */
    virtual uint32_t get_qp_number() = 0;
    virtual uint32_t get_rkey() = 0;
    virtual uint64_t external_frame_memory() = 0;
    virtual size_t frame_size() = 0;
    virtual size_t page_size() = 0;
    virtual unsigned pages() = 0;
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_ROCE_RECEIVER_HPP
