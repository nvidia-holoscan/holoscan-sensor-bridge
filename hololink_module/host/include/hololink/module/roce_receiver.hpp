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

/**
 * @brief Describes one received frame: where it is in memory, and
 * when and how it arrived.
 *
 * Filled in by RoceReceiverInterfaceV1::get_next_frame(). All fields are
 * set on a successful call; their values are unspecified when
 * get_next_frame() returns `false`.
 */
class RoceReceiverFrameInfoV1 {
public:
    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    static constexpr const char* release_version = "2.7.0";
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief Device memory holding this frame's data, as a CUdeviceptr.
     *
     * This is within the memory the receiver was started with. The
     * receiver reuses its pages in turn, so the frame must be consumed
     * or copied before the device wraps around to the same page.
     */
    uint64_t frame_memory = 0;

    /**
     * @brief Device memory holding the metadata block that follows this
     * frame's data, as a CUdeviceptr.
     *
     * Decode it with FrameMetadataInterfaceV1. It is valid for the same
     * period as `frame_memory`.
     */
    uint64_t metadata_memory = 0;

    /**
     * @brief Frames the receiver has taken in since it started, this one
     * included.
     *
     * Counts frames that were received but later discarded (e.g. the
     * application isn't able to keep up), so a jump between consecutive frames
     * you receive indicates frames were lost.
     */
    uint64_t received_frame_number = 0;

    /// Frame counter the device reports, extended to 32 bits.
    uint32_t frame_number = 0;

    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    /**
     * @brief Immediate value the device sent with the frame's last write.
     *
     * Identifies which of the receiver's pages holds the frame, and
     * carries the packet sequence number. Its layout is defined by the
     * device.
     */
    uint32_t imm_data = 0;
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief Host time when the receiver was notified that the frame had
     * arrived, in seconds.
     *
     * Wall-clock (CLOCK_REALTIME) time; see `received_ns` for the
     * fractional part.
     */
    uint64_t received_s = 0;

    /// Nanoseconds part of `received_s`.
    uint64_t received_ns = 0;

    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    /// Reserved. Currently always 0.
    uint64_t rx_write_requests = 0;
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief Frames discarded since the receiver started, as of this
     * frame's arrival.
     *
     * A frame is discarded when it arrives while every page the receiver
     * holds is still waiting to be collected with get_next_frame();
     * the oldest waiting frame is dropped. Seeing this value increase
     * means the application is not keeping up with the device.
     */
    uint32_t dropped = 0;
};

/**
 * @brief Controller that manages data reception via RoCE packets.
 *
 * This is usually driven by e.g. RoceReceiverOp and serves to
 * set up the host for RDMA data reception, configure the HSB device for
 * compatible data transmission, and provides a routine that observes
 * received data indications and queues those up for application reception.
 */
class RoceReceiverInterfaceV1
    : public ConfigurableService<RoceReceiverInterfaceV1> {
public:
    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    static constexpr const char* type_id = "roce_receiver.v1";
    static constexpr const char* release_version = "2.7.0";

    /* Per-channel instance_id derivation used by the metadata-form
     * get_service. Throws if the metadata is missing 'serial_number'
     * or 'data_channel'. */
    static std::string locator_id(const EnumerationMetadata& metadata)
    {
        return "serial=" + metadata.get<std::string>("serial_number")
            + ";data_channel=" + std::to_string(metadata.get<int64_t>("data_channel"));
    }

    virtual ~RoceReceiverInterfaceV1() = default;
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief Configure the receiver and network adapter for RDMA.
     *
     * Frames are delivered into `cu_buffer`, which is a single
     * contiguous region with enough space to accommodate the total
     * number of frames indicated by the `pages` parameter; each frame
     * is `cu_page_size` bytes apart; `cu_page_size` needs to include
     * space for received frame metadata (and its alignment).
     *
     * @param ibv_name Name of the InfiniBand verbs device of the network
     *      adapter that receives the frames, e.g. "roceP5p3s0f0".
     * @param ibv_port Port of that device to use.
     * @param cu_buffer CUDA memory frames are delivered into, as a
     *      CUdeviceptr.
     * @param cu_buffer_size Bytes available at `cu_buffer`.
     * @param cu_frame_size Bytes of frame data expected for each
     *      received data frame: the HSB device is programmed to trigger
     *      an end-of-frame when this many bytes are received; but
     *      an earlier TLAST signal will trigger end-of-frame at an
     *      earlier point.
     * @param cu_page_size Bytes from one page to the next.
     * @param pages Pages the device cycles through.
     * @param metadata_offset Offset, in bytes from the start of a page,
     *      of the metadata block the device writes after the frame data.
     * @param peer_ip IPv4 address of the device, in dotted-decimal
     *      form. The receiver accepts data only from this address.
     * @param queue_size Completed frames the receiver holds while the
     *      application is busy with an earlier one. When a frame
     *      completes with this many already waiting, the oldest waiting
     *      frame is discarded (see RoceReceiverFrameInfoV1::dropped).
     * @return HOLOLINK_MODULE_OK or an error status.
     * @throws std::runtime_error if the receiver cannot be brought up,
     *      for example the named device is not found or `peer_ip` is not
     *      a valid address.
     */
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

    /**
     * @brief End the run, returning blocking_monitor() to its caller.
     *
     * Callable from any thread, including while blocking_monitor() runs.
     * Does nothing if the receiver was never started.
     */
    virtual void close() = 0;

    /**
     * @brief Run the receiver until close() is called.
     *
     * Frames arrive only while this runs, so it gets a thread of its
     * own, and that thread carries the CUDA context the receiver was
     * started with.  Clients typically execute this in its own
     * thread.
     *
     * @throws std::runtime_error if start() has not been called.
     */
    virtual void blocking_monitor() = 0;

    /**
     * @brief Take the next completed frame.
     *
     * Waits up to `timeout_ms` for a frame.  On successful reception of a
     * frame, `info` is updated with various metadata about the data, which is
     * present at the indicated address within the `cu_buffer` passed to
     * `start`.  Frames are returned in the order they are received.
     *
     * @param timeout_ms How long to wait for a frame, in milliseconds.
     * @param[out] info Describes the frame that was taken.
     * @return True when a frame was taken; false when the timeout
     *      elapsed first.
     * @throws std::runtime_error if start() has not been called.
     */
    virtual bool get_next_frame(unsigned timeout_ms,
        RoceReceiverFrameInfoV1& info)
        = 0;

    /**
     * @return False when get_next_frame() will block waiting for
     *      a new frame to arrive.
     */
    virtual bool frames_ready() = 0;

    /**
     * @brief Register a callback run each time a frame becomes available.
     *
     * An application that cannot afford to wait in get_next_frame() is
     * told from here that a frame is waiting, and takes it when it is
     * ready to. The callback runs on the thread running
     * blocking_monitor(). Register it before that starts; pass an empty
     * std::function to stop being called. It stays registered across
     * calls to start().
     *
     * @param callback Run once per frame that becomes available.
     */
    virtual void set_frame_ready(std::function<void()> callback) = 0;

    /**
     * @brief The number of the queue pair that receives this channel's
     *      frames.
     *
     * The channel programs this into the device so it sends to the
     * right queue pair.
     *
     * @return The number. Only meaningful once start() has run.
     * @throws std::runtime_error if start() has not been called.
     */
    virtual uint32_t get_qp_number() = 0;

    /**
     * @brief The key the device presents when it writes a frame into the
     *      receiver's memory.
     *
     * @return The key. Only meaningful once start() has run.
     * @throws std::runtime_error if start() has not been called.
     */
    virtual uint32_t get_rkey() = 0;

    /**
     * @brief The address, in the device's view of the receiver's memory,
     *      where the first page begins.
     *
     * The device addresses page `n` at this value plus `n` times
     * page_size(). This is not a CUDA address.
     *
     * @return The address. Only meaningful once start() has run.
     * @throws std::runtime_error if start() has not been called.
     */
    virtual uint64_t external_frame_memory() = 0;

    /**
     * @brief Bytes of frame data the device sends per frame.
     *
     * @return The `cu_frame_size` start() was given.
     */
    virtual size_t frame_size() = 0;

    /**
     * @brief Bytes from one page to the next.
     *
     * @return The `cu_page_size` start() was given.
     */
    virtual size_t page_size() = 0;

    /**
     * @brief Pages the device cycles through.
     *
     * @return The `pages` start() was given.
     */
    virtual unsigned pages() = 0;
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_ROCE_RECEIVER_HPP
