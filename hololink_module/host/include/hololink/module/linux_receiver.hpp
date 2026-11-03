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

/**
 * @brief Received frame metadata.
 *
 * LinuxReceiverInterfaceV1::get_next_frame() fills one of these in for
 * every frame it hands back, including information about which buffer
 * is updated, timestamps associated with the received data, and other
 * diagnostics.
 */
class LinuxReceiverFrameInfoV1 {
public:
    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    static constexpr const char* release_version = "2.7.0";
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /// Device memory holding this frame's data, as a CUdeviceptr.
    uint64_t frame_memory = 0;

    /// Count of packets received for this specific data frame.
    uint32_t frame_packets_received = 0;

    /// Number of bytes from all packets received for this frame;
    /// note that this doesn't specify which bytes were written
    /// to the buffer.
    uint32_t frame_bytes_received = 0;

    /// Data frames the receiver has handled since it started, this one included.
    uint32_t received_frame_number = 0;

    /// Frame counter the device reports, extended to 32 bits.
    uint32_t frame_number = 0;

    /// Immediate value the device sent with the frame's last write.
    uint32_t imm_data = 0;

    /// Host time when the frame's last packet arrived.
    int64_t received_s = 0;

    /// Nanoseconds part of `received_s`.
    int64_t received_ns = 0;

    /**
     * @brief Packets missing from the stream since the receiver started.
     *
     * Counted from gaps in the packet sequence numbers, so packets that
     * arrive out of order count as missing.
     */
    uint64_t packets_dropped = 0;

    /// Device-reported status of the frame.
    uint32_t flags = 0;

    /// Packet sequence number the frame ended on.
    uint32_t psn = 0;

    /// CRC the device computed over the frame data.
    uint32_t crc = 0;

    /// Nanoseconds part of `timestamp_s`.
    uint32_t timestamp_ns = 0;

    /// Device time when the frame's first sample data was received.
    uint64_t timestamp_s = 0;

    /// Frame data the device wrote, in bytes.
    uint64_t bytes_written = 0;

    /// Nanoseconds part of `metadata_s`.
    uint32_t metadata_ns = 0;

    /// Device time when the metadata block itself was sent; this
    /// timestamp follows the last data included in the data frame.
    uint64_t metadata_s = 0;
};

/**
 * @brief Provide device-side support for LinuxReceiver.
 *
 * The device transmits a data channel's frames to a host socket as a
 * stream of UDP datagrams. This receiver reassembles that stream into
 * whole frames and delivers each one into CUDA memory the application
 * supplies, along with a LinuxReceiverFrameInfoV1 describing it. One
 * receiver serves one data channel; RoceReceiverInterfaceV1 is the
 * counterpart for channels the NIC receives without the host reading
 * packets.
 *
 * Fetch it with the EnumerationMetadata describing the channel, after
 * fetching that channel's LinuxDataChannelInterfaceV1.
 *
 * The application drives the receiver:
 *
 * - start() prepares the receiver over a socket the channel has
 *   already prepared, and over the CUDA buffer frames are delivered
 *   into. The values the channel programs the device with are readable
 *   once it returns, so this precedes
 *   LinuxDataChannelInterfaceV1::attach_receiver().
 * - blocking_monitor() runs the receiver and returns when close() is
 *   called, so it gets a thread of its own.
 * - get_next_frame() hands over frames as they complete;
 *   frames_ready() and set_frame_ready() tell an application when one
 *   is waiting.
 * - close() ends the run, and start() may then run the receiver again.
 */
class LinuxReceiverInterfaceV1
    : public ConfigurableService<LinuxReceiverInterfaceV1> {
public:
    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    static constexpr const char* type_id = "linux_receiver.v1";
    static constexpr const char* release_version = "2.7.0";

    /* Per-channel instance_id derivation used by the metadata-form
     * get_service. Throws if the metadata is missing 'serial_number'
     * or 'data_channel'. */
    static std::string locator_id(const EnumerationMetadata& metadata)
    {
        return "serial=" + metadata.get<std::string>("serial_number")
            + ";data_channel=" + std::to_string(metadata.get<int64_t>("data_channel"));
    }

    virtual ~LinuxReceiverInterfaceV1() = default;
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief Prepare the receiver to run over a socket and a CUDA buffer.
     *
     * Frames are delivered into `cu_buffer`, one frame at a time. The
     * device writes each frame to one of `pages` pages, `cu_page_size`
     * bytes apart, so a page holds a frame's data, the padding that
     * aligns it, and the metadata block that follows it.
     *
     * get_qp_number(), get_rkey(), local_port(), frame_size(),
     * page_size() and pages() report what the device is programmed with
     * once this returns, so an application calls this before
     * LinuxDataChannelInterfaceV1::attach_receiver(). Calling it again
     * after close() runs the receiver a second time.
     *
     * @param data_socket Socket prepared by
     * LinuxDataChannelInterfaceV1::configure_socket(). The device
     * transmits to the port it is bound to.
     * @param cu_buffer CUDA memory frames are delivered into, as a
     * CUdeviceptr.
     * @param cu_buffer_size Bytes available at `cu_buffer`.
     * @param cu_frame_size Bytes of frame data the device sends per
     * frame.
     * @param cu_page_size Bytes from one page to the next.
     * @param pages Pages the device cycles through.
     * @param queue_size Completed frames the receiver holds while the
     * application is busy with an earlier one.
     * @return HOLOLINK_MODULE_OK or an error status.
     */
    virtual hololink_module_status_t start(
        int data_socket,
        uint64_t cu_buffer,
        size_t cu_buffer_size,
        size_t cu_frame_size,
        size_t cu_page_size,
        unsigned pages,
        unsigned queue_size)
        = 0;

    /**
     * @brief End the run, returning blocking_monitor() to its caller.
     *
     * Callable from any thread, including while blocking_monitor() runs.
     */
    virtual void close() = 0;

    /**
     * @brief Run the receiver until close() is called.
     *
     * Frames arrive only while this runs, so it gets a thread of its
     * own, and that thread carries the CUDA context the receiver was
     * started with.
     */
    virtual void blocking_monitor() = 0;

    /**
     * @brief Take the next completed frame.
     *
     * Waits up to `timeout_ms` for a frame, then delivers it into the
     * CUDA buffer start() was given and describes it in `info`. The copy
     * runs on `cuda_stream`, so work already queued on that stream
     * precedes it and work queued afterwards sees the frame.
     *
     * @param timeout_ms How long to wait for a frame, in milliseconds.
     * @param[out] info Describes the frame that was delivered.
     * @param cuda_stream Stream the frame is delivered on, as a CUstream.
     * @return True when a frame was delivered; false when the timeout
     * elapsed first.
     */
    virtual bool get_next_frame(unsigned timeout_ms,
        LinuxReceiverFrameInfoV1& info, void* cuda_stream)
        = 0;

    /**
     * @brief Whether a frame is waiting to be taken.
     *
     * @return True when get_next_frame() has a frame to hand over; false
     * when it would wait for one.
     */
    virtual bool frames_ready() = 0;

    /**
     * @brief Register a callback run each time a frame becomes available.
     *
     * An application that cannot afford to wait in get_next_frame() is
     * told from here that a frame is waiting, and takes it when it is
     * ready to. The callback runs on the thread running
     * blocking_monitor(). Register it before that starts; pass an empty
     * std::function to stop being called.
     *
     * @param callback Run once per frame that becomes available.
     */
    virtual void set_frame_ready(std::function<void()> callback) = 0;

    /**
     * @brief The connection number the device stamps into this channel's
     *      packets.
     *
     * @return The number, once start() has run.
     */
    virtual uint32_t get_qp_number() = 0;

    /**
     * @brief The key the device presents when it writes a frame.
     *
     * @return The key, once start() has run.
     */
    virtual uint32_t get_rkey() = 0;

    /**
     * @brief The UDP port the device transmits this channel's frames to.
     *
     * The port the socket given to start() is bound to.
     *
     * @return The port, once start() has run.
     */
    virtual uint32_t local_port() = 0;

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

#endif // HOLOLINK_MODULE_LINUX_RECEIVER_HPP
