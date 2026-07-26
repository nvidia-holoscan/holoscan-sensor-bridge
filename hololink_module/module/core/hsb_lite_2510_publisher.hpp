/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_CORE_HSB_LITE_2510_PUBLISHER_HPP
#define HOLOLINK_MODULE_CORE_HSB_LITE_2510_PUBLISHER_HPP

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "hololink/module/data_channel.hpp"
#include "hololink/module/logging.hpp"
#include "hololink/module/publisher.hpp"

#include "hsb_lite_2510_data_channel.hpp"
#include "hsb_lite_publisher.hpp"

namespace hololink::module::module_core {

/* Concrete Publisher for compat-id 2510 carriers. Inherits the
 * canonical HSB-Lite branch chain + enumeration body from
 * HsbLitePublisher and overrides only the two RoCE branches that
 * need a 2510-specific impl subclass. Future partner supplements
 * targeting pre-0x2602 / pre-0x2603 FPGA revisions instantiate this
 * Publisher directly (or subclass it further to override
 * module_name). Every other canonical service uses the base's
 * default behavior unchanged. */
class HsbLite2510Publisher : public HsbLitePublisher {
public:
    /* This module drives HSB-Lite FPGAs with IP version 0x2510 through
     * 0x2603 (inclusive). Because it ships as the bare hololink_<UUID>.so,
     * the Adapter loader routes every HSB-Lite board without a dedicated
     * compat-suffixed .so here — including newer silicon this build
     * predates. Gate on the reported IP version and decline anything
     * newer than 0x2603 or older than 0x2510 (returning
     * HOLOLINK_MODULE_ENUMERATION_SKIPPED rather than erroring) so an
     * unsupported board is announced to no one instead of being
     * mis-driven. Supported versions delegate to the canonical
     * HsbLitePublisher enrichment. */
    hololink_module_status_t update_metadata(
        EnumerationMetadata& metadata,
        const uint8_t* raw_packet,
        size_t raw_packet_len) override
    {
        constexpr int64_t MIN_SUPPORTED_VERSION = 0x2510;
        constexpr int64_t MAX_SUPPORTED_VERSION = 0x2603;
        // -1 means the bootp payload carried no IP version; make no
        // judgment in that case and delegate to the base.
        const int64_t version_id = metadata.get<int64_t>("hsb_ip_version", -1);
        if (version_id >= 0
            && (version_id < MIN_SUPPORTED_VERSION
                || version_id > MAX_SUPPORTED_VERSION)) {
            const std::string serial_number
                = metadata.get<std::string>("serial_number", std::string());
            // Cap the notice per serial number so a device that
            // re-announces on every bootp broadcast logs at most 4 times.
            // No locking: update_metadata only runs on the reactor thread.
            constexpr unsigned MAX_UNSUPPORTED_REPORTS = 4;
            unsigned& reports = unsupported_reports_[serial_number];
            if (reports < MAX_UNSUPPORTED_REPORTS) {
                HSB_LOG_INFO(
                    "Discovered HSB-Lite device (serial_number='{}') reporting "
                    "IP version 0x{:04x}, which is outside the range the "
                    "hsb_lite_2510 module supports (0x{:04x} through 0x{:04x}); "
                    "ignoring this device.",
                    serial_number, version_id, MIN_SUPPORTED_VERSION,
                    MAX_SUPPORTED_VERSION);
                if (++reports == MAX_UNSUPPORTED_REPORTS) {
                    HSB_LOG_INFO(
                        "Further unsupported-IP-version notices for HSB-Lite "
                        "device (serial_number='{}') will be suppressed.",
                        serial_number);
                }
            }
            return HOLOLINK_MODULE_ENUMERATION_SKIPPED;
        }
        return HsbLitePublisher::update_metadata(
            metadata, raw_packet, raw_packet_len);
    }

protected:
    std::string module_name() const override { return "hsb_lite_2510"; }

    bool construct_roce_data_channel(
        const std::string& instance_id,
        const std::string& type_id) override
    {
        if (!Publisher::has_type_id<HsbLite2510RoceDataChannelV1>(type_id)) {
            return false;
        }
        auto anchor = DataChannelInterfaceV1::get_service(
            this->self_module(), instance_id.c_str());
        auto impl = std::make_shared<HsbLite2510RoceDataChannelV1>(
            std::move(anchor));
        ServicePublisher<HsbLite2510RoceDataChannelV1>(shared_from_this())
            .publish(instance_id, impl);
        return true;
    }

    bool construct_linux_data_channel(
        const std::string& instance_id,
        const std::string& type_id) override
    {
        if (!Publisher::has_type_id<HsbLite2510LinuxDataChannelV1>(type_id)) {
            return false;
        }
        auto anchor = DataChannelInterfaceV1::get_service(
            this->self_module(), instance_id.c_str());
        auto impl = std::make_shared<HsbLite2510LinuxDataChannelV1>(
            std::move(anchor));
        ServicePublisher<HsbLite2510LinuxDataChannelV1>(shared_from_this())
            .publish(instance_id, impl);
        return true;
    }

    /* Always declared; defined out-of-line in roce_receiver_construct.cpp.
     * Publishes the 2510-specific receiver subclass when the build has
     * RoCE, and is empty (publishes nothing) otherwise. */
    bool construct_roce_receiver(
        const std::string& instance_id,
        const std::string& type_id) override;

    /* Software-transport sibling of construct_roce_receiver; defined
     * out-of-line in linux_receiver_construct_2510.cpp. Unconditional (the
     * software receiver needs no ibverbs), so it always publishes the
     * 2510-specific receiver subclass. */
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

} // namespace hololink::module::module_core

#endif // HOLOLINK_MODULE_CORE_HSB_LITE_2510_PUBLISHER_HPP
