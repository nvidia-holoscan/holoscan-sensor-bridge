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

/* Software (Linux) transport view of a per-channel DataChannelInterfaceV1.
 * A separate ConfigurableService — has-a, not is-a — whose impl holds a
 * shared_ptr to its DataChannelInterfaceV1 anchor and adds the
 * software-transport surface (configure_socket / attach_receiver /
 * detach_receiver / frame_end_sequencer). Sibling of
 * RoceDataChannelInterfaceV1; both compose the same anchor.
 *
 * The wire protocol is still RoCEv2 (HSB always speaks RoCEv2), so the
 * channel still drives the legacy DataChannel::configure_roce. Two
 * things differ from the hardware path, both because the receiver lives
 * in user space rather than on a NIC:
 *   - the device frame-memory address programmed into the FPGA is 0
 *     (HSB writes from address 0; the software receiver adds the local
 *     buffer offset itself), not the receiver's external_frame_memory();
 *   - the destination UDP port is the host socket's bound local_port,
 *     not the fixed RoCEv2 port the hardware channel uses.
 *
 * Cached under instance_id "serial=<serial_number>;data_channel=<n>"
 * (the same key as the anchor it wraps). The supplement's
 * construct_service for this type_id fetches the anchor through the
 * Publisher and passes it to the impl's constructor. The impl's
 * configure(metadata) drives the anchor's ensure_configured(metadata)
 * before building its own legacy backing — application code calls
 * LinuxDataChannelInterfaceV1::get_service(metadata) without first
 * constructing the DataChannelInterfaceV1 itself. */
class LinuxDataChannelInterfaceV1
    : public ConfigurableService<LinuxDataChannelInterfaceV1> {
public:
    static constexpr const char* type_id = "linux_data_channel.v1";

    /* Per-channel instance_id derivation used by the metadata-form
     * get_service. Same shape as DataChannelInterfaceV1::locator_id
     * — both services key by (serial, data_channel). */
    static std::string locator_id(const EnumerationMetadata& metadata)
    {
        return "serial=" + metadata.get<std::string>("serial_number")
            + ";data_channel=" + std::to_string(metadata.get<int64_t>("data_channel"));
    }

    virtual ~LinuxDataChannelInterfaceV1() = default;

    /* Bind the operator-created datagram socket to this channel's data
     * plane (delegates to the backing legacy
     * DataChannel::configure_socket). Must be called before
     * LinuxReceiverInterfaceV1::start(socket, …), so the receiver runs
     * on a socket bound to the right local interface and HSB has a
     * local_port to target. */
    virtual hololink_module_status_t configure_socket(int data_socket) = 0;

    /* Wire up the on-device packetizer to deliver frames to the bound
     * socket. Issues authenticate(get_qp_number, get_rkey) then
     * configure_roce(0, frame_size, page_size, pages, local_port) — the
     * device frame-memory address is 0 (the receiver adds the local
     * offset) and the destination port is the receiver's local_port.
     * The caller must have already brought the receiver up via
     * LinuxReceiverInterfaceV1::start(...) (and configure_socket(...)
     * before that). Call once after enumeration; pair with
     * detach_receiver() before tearing the channel down. Per-channel
     * enumeration fields are already cached on the anchor
     * DataChannelInterfaceV1, so no metadata parameter is needed. */
    virtual hololink_module_status_t attach_receiver(
        std::shared_ptr<LinuxReceiverInterfaceV1> receiver)
        = 0;

    /* Detach the on-device packetizer from any host memory. */
    virtual hololink_module_status_t detach_receiver() = 0;

    /* The frame-end Sequencer attached to this channel. Triggered by
     * the packetizer when a frame completes; receivers register
     * read-back operations here. */
    template <typename T = SequencerInterfaceV1>
    std::shared_ptr<T> frame_end_sequencer(bool allow_null = false)
    {
        const std::string instance_id = frame_end_sequencer_instance_id();
        return T::get_service(module(), instance_id.c_str(), allow_null);
    }

    /* The HololinkInterface this channel belongs to — looked up via the
     * per-board instance id helper. The corresponding
     * DataChannelInterfaceV1 anchor's hololink() returns the same
     * instance; kept here for callers that have a Linux-typed
     * shared_ptr in hand. */
    template <typename T = HololinkInterfaceV1>
    std::shared_ptr<T> get_hololink(bool allow_null = false)
    {
        const std::string instance_id = parent_hololink_instance_id();
        return T::get_service(module(), instance_id.c_str(), allow_null);
    }

protected:
    virtual std::string frame_end_sequencer_instance_id() = 0;
    virtual std::string parent_hololink_instance_id() = 0;
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_LINUX_DATA_CHANNEL_HPP
