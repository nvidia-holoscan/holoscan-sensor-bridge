/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_LEOPARD_VB1940_2510_LEOPARD_VB1940_2510_PUBLISHER_HPP
#define HOLOLINK_MODULE_LEOPARD_VB1940_2510_LEOPARD_VB1940_2510_PUBLISHER_HPP

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "hololink/module/data_channel.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/logging.hpp"
#include "hololink/module/publisher.hpp"
#include "hololink/module/status.h"

#include "hsb_ip_2510_data_channel.hpp"
#include "leopard_vb1940_publisher.hpp"

namespace hololink::module::leopard_vb1940 {

/* Publisher for Leopard VB1940 carriers built on the pre-0x2602 FPGA.
 * The board itself is unchanged — D-PHY shape, Bajoran TS2 oscillator,
 * sensor layout and the LeopardVb1940InterfaceV1 supplement all come
 * from LeopardVb1940Publisher — so this subclass carries only the four
 * transport branches whose memory map and RDMA immediate-data layout
 * differ, plus the version gate.
 *
 * These boards usually report compat_id 0, so the loader can't select
 * this module by filename; it ships as the bare hololink_<UUID>.so and
 * is therefore the catch-all for every Leopard VB1940 board without a
 * dedicated compat-suffixed .so. The transports it publishes handle
 * both revisions on their own: each one's make_backing / make_receiver
 * compares the reported hsb_ip_version and picks the modern class at or
 * above the cutover. */
class LeopardVb1940_2510_Publisher : public LeopardVb1940Publisher {
public:
    /* This module understands FPGA IP versions up to 0x2603. Because it
     * ships as the bare hololink_<UUID>.so, the Adapter loader routes every
     * Leopard VB1940 board without a dedicated compat-suffixed .so here —
     * including newer silicon this build predates. Decline anything newer
     * than 0x2603 (returning HOLOLINK_MODULE_ENUMERATION_SKIPPED rather than
     * erroring) so an unsupported board is announced to no one instead of
     * being mis-driven.
     *
     * There is deliberately no lower bound, unlike the HSB-Lite 2510 module:
     * a 2510 VB1940 is known only to report some version below 0x2603, and
     * this module is the board's catch-all, so a lower bound guessed too high
     * would decline the very hardware it exists for. Add one once the exact
     * reported version is known. */
    hololink_module_status_t update_metadata(
        EnumerationMetadata& metadata,
        const uint8_t* raw_packet,
        size_t raw_packet_len) override
    {
        constexpr int64_t MAX_SUPPORTED_VERSION = 0x2603;
        // -1 means the bootp payload carried no IP version; make no
        // judgment in that case and delegate to the base.
        const int64_t version_id = metadata.get<int64_t>("hsb_ip_version", -1);
        if (version_id > MAX_SUPPORTED_VERSION) {
            const std::string serial_number
                = metadata.get<std::string>("serial_number", std::string());
            // Cap the notice per serial number so a device that
            // re-announces on every bootp broadcast logs at most 4 times.
            // No locking: update_metadata only runs on the reactor thread.
            constexpr unsigned MAX_UNSUPPORTED_REPORTS = 4;
            unsigned& reports = unsupported_reports_[serial_number];
            if (reports < MAX_UNSUPPORTED_REPORTS) {
                HSB_LOG_INFO(
                    "Discovered Leopard VB1940 device (serial_number='{}') "
                    "reporting IP version 0x{:04x}, which is newer than the "
                    "leopard_vb1940_2510 module supports (up to 0x{:04x}); "
                    "ignoring this device.",
                    serial_number, version_id, MAX_SUPPORTED_VERSION);
                if (++reports == MAX_UNSUPPORTED_REPORTS) {
                    HSB_LOG_INFO(
                        "Further unsupported-IP-version notices for Leopard "
                        "VB1940 device (serial_number='{}') will be "
                        "suppressed.",
                        serial_number);
                }
            }
            return HOLOLINK_MODULE_ENUMERATION_SKIPPED;
        }
        return LeopardVb1940Publisher::update_metadata(
            metadata, raw_packet, raw_packet_len);
    }

protected:
    std::string module_name() const override { return "leopard_vb1940_2510"; }

    bool construct_roce_data_channel(
        const std::string& instance_id,
        const std::string& type_id) override
    {
        using hololink::module::module_core::HsbIp2510RoceDataChannelV1;
        if (!Publisher::has_type_id<HsbIp2510RoceDataChannelV1>(type_id)) {
            return false;
        }
        auto anchor = DataChannelInterfaceV1::get_service(
            this->self_module(), instance_id.c_str());
        auto impl = std::make_shared<HsbIp2510RoceDataChannelV1>(
            std::move(anchor));
        ServicePublisher<HsbIp2510RoceDataChannelV1>(shared_from_this())
            .publish(instance_id, impl);
        return true;
    }

    bool construct_linux_data_channel(
        const std::string& instance_id,
        const std::string& type_id) override
    {
        using hololink::module::module_core::HsbIp2510LinuxDataChannelV1;
        if (!Publisher::has_type_id<HsbIp2510LinuxDataChannelV1>(type_id)) {
            return false;
        }
        auto anchor = DataChannelInterfaceV1::get_service(
            this->self_module(), instance_id.c_str());
        auto impl = std::make_shared<HsbIp2510LinuxDataChannelV1>(
            std::move(anchor));
        ServicePublisher<HsbIp2510LinuxDataChannelV1>(shared_from_this())
            .publish(instance_id, impl);
        return true;
    }

    /* Always declared; defined out-of-line in
     * roce_receiver_construct_2510.cpp. Publishes the 2510 receiver when the
     * build has RoCE, and is empty (publishes nothing) otherwise. */
    bool construct_roce_receiver(
        const std::string& instance_id,
        const std::string& type_id) override;

    /* Software-transport sibling of construct_roce_receiver; defined
     * out-of-line in linux_receiver_construct_2510.cpp. Unconditional (the
     * software receiver needs no ibverbs), so it always publishes the 2510
     * receiver. */
    bool construct_linux_receiver(
        const std::string& instance_id,
        const std::string& type_id) override;

private:
    /* Per-serial-number count of "unsupported version" notices already
     * emitted, so update_metadata can cap the log at 4 per device.
     * Unsynchronized: update_metadata is only ever called on the
     * reactor thread. */
    std::map<std::string, unsigned> unsupported_reports_;
};

} // namespace hololink::module::leopard_vb1940

#endif // HOLOLINK_MODULE_LEOPARD_VB1940_2510_LEOPARD_VB1940_2510_PUBLISHER_HPP
