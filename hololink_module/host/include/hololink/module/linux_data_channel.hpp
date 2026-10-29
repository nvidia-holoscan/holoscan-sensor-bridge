/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_LINUX_DATA_CHANNEL_HPP
#define HOLOLINK_MODULE_LINUX_DATA_CHANNEL_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "enumeration_metadata.hpp"
#include "module.hpp"
#include "service.hpp"
#include "status.h"

namespace hololink::module {

class HololinkInterfaceV1;
class SequencerInterfaceV1;
class LinuxReceiverInterfaceV1;

/**
 * @brief Provide support for Linux data receivers.
 *
 * For systems without hardware accelerated RDMA for delivery
 * of received data, a Linux specific emulator for this is available,
 * which uses the CPU to receive HSB data (via UDP) into a local buffer,
 * and on the frame-end condition, that buffer is copied into the
 * target (usually GPU) memory.
 *
 * Application code would normally interact with a Linux receiver object,
 * which uses this object to support configuration of the local data
 * receiver socket.
 */
class LinuxDataChannelInterfaceV1
    : public ConfigurableService<LinuxDataChannelInterfaceV1> {
public:
    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    static constexpr const char* type_id = "linux_data_channel.v1";
    static constexpr const char* release_version = "2.7.0";

    /* Per-channel instance_id derivation used by the metadata-form
     * get_service. Throws if the metadata is missing 'serial_number'
     * or 'data_channel'. */
    static std::string locator_id(const EnumerationMetadata& metadata)
    {
        return "serial=" + metadata.get<std::string>("serial_number")
            + ";data_channel=" + std::to_string(metadata.get<int64_t>("data_channel"));
    }

    virtual ~LinuxDataChannelInterfaceV1() = default;
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief Prepare a datagram socket to receive this channel's traffic.
     *
     * Binds the socket to the host interface that reaches the device.
     * When the channel's enumeration metadata selects multicast or
     * broadcast delivery, the socket joins that group or accepts that
     * broadcast; otherwise the host assigns it a port, and
     * attach_receiver() programs the device to transmit there.
     *
     * The application creates the socket and calls this before anything
     * else runs on it.
     *
     * @param data_socket Datagram socket to prepare.
     * @return HOLOLINK_MODULE_OK or an error status.
     */
    virtual hololink_module_status_t configure_socket(int data_socket) = 0;

    /**
     * @brief Deliver this channel's frames to a running receiver.
     *
     * Programs the device to transmit frames to the receiver; the frame
     * layout it transmits -- frame size, page size and page count -- and
     * the destination port come from the receiver, so the receiver is
     * started, on a socket configure_socket() prepared, before this call.
     * Frames arrive from here on. Pair it with detach_receiver() before
     * tearing the channel down.
     *
     * @param receiver The started receiver frames are delivered to.
     * @return HOLOLINK_MODULE_OK or an error status.
     */
    virtual hololink_module_status_t attach_receiver(
        std::shared_ptr<LinuxReceiverInterfaceV1> receiver)
        = 0;

    /**
     * @brief Stop transmitting, clearing the state attach_receiver() set
     *      up.
     *
     * The channel delivers nothing until another attach_receiver(). This
     * writes to the device, so it fails when the device is unreachable.
     *
     * @return HOLOLINK_MODULE_OK or an error status.
     */
    virtual hololink_module_status_t detach_receiver() = 0;

    /**
     * @brief The sequencer this channel triggers at the end of each frame.
     *
     * The device runs the sequence once per completed frame on this
     * channel. Receivers add the device reads they want taken with every
     * frame.
     *
     * @tparam T The interface version to fetch.
     * @param allow_null Return an empty pointer, rather than throwing, when
     * this channel publishes no such sequencer.
     * @return The sequencer.
     */
    template <typename T = SequencerInterfaceV1>
    std::shared_ptr<T> frame_end_sequencer(bool allow_null = false)
    {
        const std::string instance_id = frame_end_sequencer_instance_id();
        return T::get_service(module(), instance_id.c_str(), allow_null);
    }

    /**
     * @brief The device this channel runs on.
     *
     * The same device DataChannelInterfaceV1::hololink() returns for this
     * channel, for callers holding the software transport.
     *
     * @tparam T The interface version to fetch.
     * @param allow_null Return an empty pointer, rather than throwing, when
     * this device is not published.
     * @return The device.
     */
    template <typename T = HololinkInterfaceV1>
    std::shared_ptr<T> get_hololink(bool allow_null = false)
    {
        const std::string instance_id = parent_hololink_instance_id();
        return T::get_service(module(), instance_id.c_str(), allow_null);
    }

    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
protected:
    virtual std::string frame_end_sequencer_instance_id() = 0;
    virtual std::string parent_hololink_instance_id() = 0;
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_LINUX_DATA_CHANNEL_HPP
