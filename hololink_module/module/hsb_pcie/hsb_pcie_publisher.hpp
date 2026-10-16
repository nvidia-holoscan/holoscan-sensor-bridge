/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#ifndef HOLOLINK_MODULE_HSB_PCIE_HSB_PCIE_PUBLISHER_HPP
#define HOLOLINK_MODULE_HSB_PCIE_HSB_PCIE_PUBLISHER_HPP

/**
 * @file
 * @brief Publisher for the hsb_pcie module: HSB IP over a PCIe endpoint.
 *
 * Shape of module/core/hsb_lite_publisher.hpp (HsbLiteChannelConfigurationV1 /
 * HsbLitePublisher) with the PCIe specifics: two sensors, one SIF each, on data plane 0
 * with a per-sensor MSI-X vector; PcieHololinkV1 as the per-board Hololink; the PCIe data
 * channel (PcieDataChannelV1, the transport view of the per-channel anchor) published from
 * construct_board_services; the Ethernet data-channel services and the oscillator declined;
 * and the PCIe-specific completion of the kernel driver's bootp announcement (the BDF from
 * the htype 0x50 hardware address, the vendor tail) done here, in update_metadata, so the
 * host's bootp parser stays transport-agnostic.
 */

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/logging.hpp"
#include "hololink/module/oscillator.hpp"
#include "hololink/module/pcie/hololink_pcie_bootp.h"
#include "hololink/module/pcie/pcie_sysfs.hpp"
#include "hololink/module/publisher.hpp"
#include "hololink/module/status.h"

#include <hololink/core/deserializer.hpp>

#include "hsb_ip_publisher.hpp"

#include "pcie_data_channel_v1.hpp"
#include "pcie_hololink_v1.hpp"

namespace hololink::module::hsb_pcie {

/* ChannelConfigurationInterfaceV1 for the PCIe carrier. Two sensors, each spanning one SIF,
 * both on physical data plane 0 (there is no HIF on this transport; the module-native
 * receiver keys on pcie_vector instead). The address fields come from the shared
 * hsb_ip_sensor_metadata formula so VP/SIF addressing is identical to the Ethernet boards:
 * sensor n -> vp_address 0x1000 + 0x40 * n, i.e. the FPGA's PCIE_DP block n. */
class PcieChannelConfigurationV1 : public module_core::HsbIpChannelConfigurationV1 {
public:
    static constexpr int64_t TOTAL_SENSORS = 2;
    static constexpr int64_t DATA_PLANE = 0;
    static constexpr uint32_t SIFS_PER_SENSOR = 1;

    PcieChannelConfigurationV1() = default;

    bool is_sensor_valid(int64_t sensor_number) const override
    {
        return sensor_number >= 0 && sensor_number < TOTAL_SENSORS;
    }

    void use_sensor(EnumerationMetadata& metadata, int64_t sensor_number) override
    {
        if (!is_sensor_valid(sensor_number)) {
            throw std::runtime_error(
                "While selecting an hsb_pcie sensor: sensor_number "
                + std::to_string(sensor_number)
                + " is out of range (hsb_pcie supports 0.."
                + std::to_string(TOTAL_SENSORS - 1) + ")");
        }
        module_core::hsb_ip_sensor_metadata(metadata, sensor_number,
            /*data_plane=*/DATA_PLANE,
            /*sifs_per_sensor=*/SIFS_PER_SENSOR);
        // MSI-X vector the module-native receiver waits on for this sensor.
        metadata["pcie_vector"] = sensor_number;
        HSB_LOG_DEBUG(
            "use_sensor sensor_number={} vp_address={:#x} sif_address={:#x} pcie_vector={} data_channel={}",
            sensor_number,
            metadata.get<int64_t>("vp_address", int64_t { 0 }),
            metadata.get<int64_t>("sif_address", int64_t { 0 }),
            sensor_number,
            metadata.get<int64_t>("data_channel", int64_t { 0 }));
    }
};

/* Publisher for the PCIe carrier. Everything HSB-shaped (I2C, sequencer, PTP/PPS, null
 * vsync, the per-channel DataChannel anchor, frame metadata) comes from HsbIpPublisher; this
 * class substitutes the PCIe Hololink, adds the PCIe data channel, declines the Ethernet
 * data-plane services and gates enumeration on PCIe-shaped metadata.
 *
 * Data-channel arrangement, mirroring the RoCE path: the transport-agnostic anchor stays the
 * module_core DataChannelV1 that construct_data_channel publishes under "data_channel.v1"
 * (the one HololinkV1::configure builds as default_data_channel() is the same class, held on
 * the Hololink rather than published). PcieDataChannelV1 is has-a with that anchor -- it is
 * published only under PcieDataChannelInterfaceV1's ids ("pcie_data_channel.v1" and its impl
 * id), keyed by the same "serial=<n>;data_channel=<k>", exactly as RoceDataChannelV1 is
 * published next to (not instead of) the anchor. Publishing the PCIe impl under
 * "data_channel.v1" as well would make it is-a DataChannelInterfaceV1, which the framework's
 * anchor design deliberately avoids, and would race construct_data_channel for the key. */
class HsbPciePublisher : public module_core::HsbIpPublisher {
public:
    static constexpr const char* MODULE_NAME = "hsb_pcie";

protected:
    std::string module_name() const override { return MODULE_NAME; }

    /* EnumerationInterfaceV1 entry point. HsbIpPublisher's boards leave this to the base and
     * customize update_board_metadata; this module overrides it because the base hook does not
     * see the raw packet, and two things about the hololink_pcie kernel driver's announcement
     * are only in the packet or only meaningful for this transport. The host's bootp parser is
     * transport-agnostic: for the driver's htype 0x50 record it delivers hardware_type = 0x50,
     * the 4-byte chaddr in hardware_address (trimmed to hlen by the Adapter), mac_id as the
     * colon-hex of those bytes, peer_ip as the datagram's IPv4 source, and it stops decoding
     * the vendor area at fpga_crc. Complete that here, before the shared path (module_name,
     * use_sensor, update_board_metadata) runs and before the Adapter matches register_ip /
     * wait_for_channel subscribers on peer_ip:
     *
     *  - The hardware address (render_pcie_hardware_address). For htype 0x50 / hlen 4 the
     *    chaddr is the PCI function address, big-endian (domain, bus, devfn); it is rendered
     *    as the textual BDF "dddd:bb:dd.f" into peer_ip -- the channel key applications wait
     *    on -- and, as the hardware address in its htype's convention, into mac_id and
     *    hardware_address_text. The IPv4 source it replaces is not kept separately: the
     *    driver sources from the BDF packed as a dotted quad, the value client_ip_address
     *    already carries.
     *  - The vendor tail (decode_vendor_tail): vendor offsets 41..57 -- four reserved bytes,
     *    the endpoint's PTP time (BE48 seconds, BE32 nanoseconds), a BE16 announcement
     *    sequence number and a status byte -- the positions the Ethernet firmware uses
     *    (docs/user_guide/ip_integration.mdx, Table 2), decoded only when the vendor length
     *    byte says they are present (56; older drivers said 39) into ptp_timestamp_s,
     *    ptp_timestamp_ns, bootp_psn and bootp_status (the raw status byte: bit0 PTP
     *    timestamp live, bit1 serial is the Device Serial Number). Only from a packet:
     *    hand-built metadata (raw_packet null) keeps whatever bootp_status it carries for
     *    the serial rule below. */
    hololink_module_status_t update_metadata(
        EnumerationMetadata& metadata,
        const uint8_t* raw_packet,
        size_t raw_packet_len) override
    {
        render_pcie_hardware_address(metadata);
        decode_vendor_tail(metadata, raw_packet, raw_packet_len);
        return module_core::HsbIpPublisher::update_metadata(metadata, raw_packet, raw_packet_len);
    }

    /* Four shapes of metadata reach update_board_metadata (selected by the board UUID):
     *
     *  1a. The hololink_pcie kernel driver's bootp announcement (htype 0x50), after
     *      update_metadata above completed it: peer_ip and mac_id are the textual BDF
     *      "dddd:bb:dd.f" (from the 4-byte chaddr), hardware_type = 0x50, and the vendor tail
     *      is decoded (bootp_status, ptp_timestamp_*, bootp_psn). The BDF is stamped out into
     *      pcie_bdf / pcie_domain / pcie_bus / pcie_device / pcie_function, and
     *      client_ip_address (the same BDF packed as a dotted quad, which is also what the
     *      driver sources the datagram from) is cross-checked.
     *  1b. An older driver's announcement (htype 1): peer_ip is the IPv4 source -- the packed
     *      BDF "0.5.1.0", or 127.0.0.1 from older drivers still and from userspace senders --
     *      and the BDF has to be restored from client_ip_address. Same stamps; peer_ip is
     *      rewritten to the BDF here, before the Adapter matches register_ip /
     *      wait_for_channel subscribers, so applications address and wait for the endpoint
     *      by BDF in both cases.
     *  2.  Hand-built metadata with peer_ip already a PCI function address (no hardware_type).
     *  3.  Hand-built metadata spelling the endpoint as /dev/hololink_pcieN or a
     *      /dev/hololink/by-bdf/<bdf> alias, with an explicit transport = "pcie".
     *
     * serial_number, for the bootp shapes (1a, 1b): the announced 7-byte serial is kept when
     * bootp_status says it is the endpoint's PCIe Device Serial Number (14 hex digits, unique
     * per board); otherwise it is the driver's placeholder ("PCIE" + BDF bytes) and is replaced
     * by "pcie-<bdf>", unique per slot. Both are filesystem-safe for NamedLock and are the
     * "serial=<...>" instance key every service is published under. pcie_serial_source says
     * which was used. Hand-built metadata keeps whatever serial it carries.
     *
     * Anything else -- an IPv4 peer with no client address, i.e. an Ethernet-shaped record
     * that somehow carries this UUID -- is declined. mac_id is informational (for shape 1a it
     * is the BDF text as well). Stamps transport=pcie when absent. */
    hololink_module_status_t update_board_metadata(
        EnumerationMetadata& metadata) override
    {
        const auto peer_it = metadata.find(PcieHololinkV1::KEY_PEER_IP);
        if (peer_it == metadata.end() || !std::holds_alternative<std::string>(peer_it->second)) {
            HSB_LOG_DEBUG("hsb_pcie: declining enumeration without a peer_ip string");
            return HOLOLINK_MODULE_ENUMERATION_SKIPPED;
        }
        std::string peer_ip = std::get<std::string>(peer_it->second);
        const int64_t hardware_type = metadata.get<int64_t>("hardware_type", int64_t { -1 });
        std::optional<uint32_t> bdf32; // set once the record resolves to a PCI function
        bool from_bootp = false; // shapes 1a / 1b: apply the serial rule

        if (PcieHololinkV1::is_ipv4_peer(peer_ip)) {
            // Shape 1b: restore the BDF from the bootp client address.
            const auto client_it = metadata.find("client_ip_address");
            uint32_t value = 0;
            if (client_it == metadata.end() || !std::holds_alternative<std::string>(client_it->second)
                || !pcie::dotted_quad_to_u32(std::get<std::string>(client_it->second), value)) {
                HSB_LOG_DEBUG("hsb_pcie: declining enumeration for peer_ip='{}' (an IPv4 address with "
                              "no bootp client address to carry a PCI function)",
                    peer_ip);
                return HOLOLINK_MODULE_ENUMERATION_SKIPPED;
            }
            const std::string bdf = pcie::bdf32_to_string(value);
            uint32_t source32 = 0;
            const bool source_is_bdf
                = pcie::dotted_quad_to_u32(peer_ip, source32) && source32 == value;
            if (!source_is_bdf && peer_ip != "127.0.0.1") {
                HSB_LOG_WARN("hsb_pcie: bootp announcement for {} arrived from {}, which is neither "
                             "that BDF packed as an IPv4 address ({}.{}.{}.{}) nor the loopback "
                             "address; trusting the bootp client address",
                    bdf, peer_ip, (value >> 24) & 0xFFu, (value >> 16) & 0xFFu,
                    (value >> 8) & 0xFFu, value & 0xFFu);
            }
            metadata["pcie_source_ip"] = peer_ip;
            HSB_LOG_DEBUG("hsb_pcie: bootp announcement from {} restored to peer_ip={}", peer_ip, bdf);
            peer_ip = bdf;
            bdf32 = value;
            from_bootp = true;
        } else if (PcieHololinkV1::is_bdf_peer(peer_ip)) {
            // Shape 1a (the parser already rendered chaddr as the BDF) or shape 2 (hand-built).
            peer_ip = pcie::normalize_bdf(peer_ip);
            uint32_t value = 0;
            if (pcie::bdf_string_to_u32(peer_ip, value)) {
                bdf32 = value;
            }
            if (hardware_type == static_cast<int64_t>(HL_PCIE_BOOTP_HTYPE_PCIE)) {
                from_bootp = true;
                // ciaddr carries the same BDF packed as a dotted quad; disagreement means a
                // corrupt or hand-assembled record. chaddr (peer_ip) wins.
                const std::string client_ip = metadata.get<std::string>("client_ip_address", std::string());
                uint32_t client32 = 0;
                if (bdf32 && !client_ip.empty() && pcie::dotted_quad_to_u32(client_ip, client32)
                    && client32 != *bdf32) {
                    HSB_LOG_WARN("hsb_pcie: announcement for {} carries client_ip_address {} ({}); "
                                 "trusting the hardware address",
                        peer_ip, client_ip, pcie::bdf32_to_string(client32));
                }
            }
        }

        if (bdf32) {
            metadata[PcieHololinkV1::KEY_PEER_IP] = peer_ip;
            metadata[PcieHololinkV1::KEY_BDF] = peer_ip;
            metadata["pcie_domain"] = static_cast<int64_t>((*bdf32 >> 16) & 0xFFFFu);
            metadata["pcie_bus"] = static_cast<int64_t>((*bdf32 >> 8) & 0xFFu);
            metadata["pcie_device"] = static_cast<int64_t>((*bdf32 >> 3) & 0x1Fu);
            metadata["pcie_function"] = static_cast<int64_t>(*bdf32 & 0x7u);
            metadata[PcieHololinkV1::KEY_TRANSPORT] = std::string(PcieHololinkV1::TRANSPORT_PCIE);
        }
        if (from_bootp && bdf32) {
            const int64_t status = metadata.get<int64_t>("bootp_status", int64_t { 0 });
            const std::string announced = metadata.get<std::string>(PcieHololinkV1::KEY_SERIAL_NUMBER, std::string());
            const bool hardware_serial = (status & HL_PCIE_BOOTP_STATUS_SERIAL_DSN) != 0
                && announced.size() == 2 * HL_PCIE_BOOTP_SERIAL_LEN
                && std::all_of(announced.begin(), announced.end(),
                    [](unsigned char c) { return std::isxdigit(c) != 0; });
            if (hardware_serial) {
                metadata["pcie_serial_source"] = std::string("dsn");
            } else {
                metadata[PcieHololinkV1::KEY_SERIAL_NUMBER] = std::string("pcie-") + peer_ip;
                metadata["pcie_serial_source"] = std::string("bdf");
            }
        }

        std::string transport;
        const auto transport_it = metadata.find(PcieHololinkV1::KEY_TRANSPORT);
        if (transport_it != metadata.end() && std::holds_alternative<std::string>(transport_it->second)) {
            transport = std::get<std::string>(transport_it->second);
        }
        if (!PcieHololinkV1::is_bdf_peer(peer_ip) && transport != PcieHololinkV1::TRANSPORT_PCIE) {
            HSB_LOG_DEBUG("hsb_pcie: declining enumeration for peer_ip='{}' (not a PCI address "
                          "DDDD:BB:DD.F and transport is not '{}')",
                peer_ip, PcieHololinkV1::TRANSPORT_PCIE);
            return HOLOLINK_MODULE_ENUMERATION_SKIPPED;
        }
        const auto valid_it = metadata.find("pcie_identity_valid");
        if (valid_it != metadata.end() && std::holds_alternative<int64_t>(valid_it->second)
            && std::get<int64_t>(valid_it->second) == 0) {
            HSB_LOG_WARN("hsb_pcie: declining {}: endpoint is not responding", peer_ip);
            return HOLOLINK_MODULE_ENUMERATION_SKIPPED;
        }
        if (transport.empty()) {
            metadata[PcieHololinkV1::KEY_TRANSPORT] = std::string(PcieHololinkV1::TRANSPORT_PCIE);
        }
        return HOLOLINK_MODULE_OK;
    }

    void publish_channel_configuration() override
    {
        auto impl = std::make_shared<PcieChannelConfigurationV1>();
        ServicePublisher<module_core::HsbIpChannelConfigurationV1>(shared_from_this())
            .publish("", impl);
    }

    /* Same ids the base publishes HololinkV1 under ("hololink.v1" and
     * "hololink.module_core.v1"); the impl is a PcieHololinkV1, which IS-A HololinkV1, so
     * every sibling that fetches HololinkV1::get_service(...) gets the PCIe-backed board. */
    bool construct_hololink(
        const std::string& instance_id,
        const std::string& type_id) override
    {
        if (!Publisher::has_type_id<module_core::HololinkV1>(type_id)) {
            return false;
        }
        std::shared_ptr<module_core::HololinkV1> impl = std::make_shared<PcieHololinkV1>();
        ServicePublisher<module_core::HololinkV1>(shared_from_this())
            .publish(instance_id, impl);
        return true;
    }

    /* instance_id is "serial=<n>;data_channel=<k>". The PCIe transport view of that
     * channel: fetch the anchor through the Publisher (constructing it via
     * construct_data_channel if this is the first lookup) and hand it to the impl, exactly
     * as HsbIpPublisher::construct_roce_data_channel does for RoceDataChannelV1. Published
     * from the tail hook because the canonical chain knows nothing about this type. */
    bool construct_board_services(
        const std::string& instance_id,
        const std::string& type_id) override
    {
        if (!Publisher::has_type_id<PcieDataChannelV1>(type_id)) {
            return false;
        }
        auto anchor = DataChannelInterfaceV1::get_service(
            this->self_module(), instance_id.c_str());
        auto impl = std::make_shared<PcieDataChannelV1>(std::move(anchor));
        ServicePublisher<PcieDataChannelV1>(shared_from_this())
            .publish(instance_id, impl);
        return true;
    }

    /* Ethernet data-plane configuration has no meaning on this transport (the SIF/HIF
     * blocks are shadowed by the address policy); fail the lookup so an Ethernet-style
     * configure fails fast instead of programming registers that do not exist. The PCIe
     * transport view is PcieDataChannelInterfaceV1 (construct_board_services). */
    bool construct_roce_data_channel(
        const std::string& instance_id,
        const std::string& type_id) override
    {
        if (!Publisher::has_type_id<module_core::RoceDataChannelV1>(type_id)) {
            return false;
        }
        HSB_LOG_WARN("hsb_pcie: RoceDataChannelInterfaceV1 ('{}') is not supported on the PCIe transport", instance_id);
        return false;
    }

    bool construct_linux_data_channel(
        const std::string& instance_id,
        const std::string& type_id) override
    {
        if (!Publisher::has_type_id<module_core::LinuxDataChannelV1>(type_id)) {
            return false;
        }
        HSB_LOG_WARN("hsb_pcie: LinuxDataChannelInterfaceV1 ('{}') is not supported on the PCIe transport", instance_id);
        return false;
    }

    bool construct_coe_data_channel(
        const std::string& instance_id,
        const std::string& type_id) override
    {
        if (!Publisher::has_type_id<module_core::CoeDataChannelV1>(type_id)) {
            return false;
        }
        HSB_LOG_WARN("hsb_pcie: CoeDataChannelInterfaceV1 ('{}') is not supported on the PCIe transport", instance_id);
        return false;
    }

    /* No clock generator is reachable on this carrier. */
    bool construct_oscillator(
        const std::string& instance_id,
        const std::string& type_id) override
    {
        if (!Publisher::has_type_id<OscillatorInterfaceV1>(type_id)) {
            return false;
        }
        HSB_LOG_WARN("hsb_pcie: OscillatorInterfaceV1 ('{}') is not supported on the PCIe transport", instance_id);
        return false;
    }

private:
    /* The bootp header's hardware address in its htype's convention, for the one htype the
     * host parser does not know: 0x50 with a 4-byte chaddr is the PCI function address
     * (domain BE16, bus, devfn), rendered as the textual BDF into peer_ip (the channel key),
     * mac_id and hardware_address_text. Any other htype, length or shape of the entry is left
     * exactly as the parser delivered it (an Ethernet record, or hand-built metadata that
     * names the endpoint some other way). */
    static void render_pcie_hardware_address(EnumerationMetadata& metadata)
    {
        if (metadata.get<int64_t>("hardware_type", int64_t { -1 })
            != static_cast<int64_t>(HL_PCIE_BOOTP_HTYPE_PCIE)) {
            return;
        }
        const auto it = metadata.find("hardware_address");
        if (it == metadata.end() || !std::holds_alternative<std::vector<uint8_t>>(it->second)) {
            return;
        }
        const std::vector<uint8_t>& chaddr = std::get<std::vector<uint8_t>>(it->second);
        // The Adapter trims the parser's fixed 16-byte blob to hlen; a hand-built entry may
        // carry the length key alone or the four bytes alone. Either way exactly four bytes
        // must be meaningful.
        const int64_t hlen = metadata.get<int64_t>("hardware_address_length",
            static_cast<int64_t>(chaddr.size()));
        if (hlen != static_cast<int64_t>(HL_PCIE_BOOTP_HLEN) || chaddr.size() < HL_PCIE_BOOTP_HLEN) {
            return;
        }
        const uint32_t bdf32 = (uint32_t(chaddr[0]) << 24) | (uint32_t(chaddr[1]) << 16)
            | (uint32_t(chaddr[2]) << 8) | uint32_t(chaddr[3]);
        const std::string bdf = pcie::bdf32_to_string(bdf32);
        metadata[PcieHololinkV1::KEY_PEER_IP] = bdf;
        metadata["mac_id"] = bdf;
        metadata["hardware_address_text"] = bdf;
    }

    /* The tail of the enum-version-2 vendor area, past fpga_crc, read straight from the
     * packet (RFC 951: the vendor area starts at byte 236 with the tag and its length byte;
     * offsets below are relative to that tag, as hololink_pcie_bootp.h numbers them). Present
     * when the length byte says so; a shorter record simply lacks these keys. */
    static void decode_vendor_tail(EnumerationMetadata& metadata,
        const uint8_t* packet, size_t packet_len)
    {
        constexpr size_t VENDOR_OFFSET = 236; // 28-byte header + 16 chaddr + 64 sname + 128 file
        constexpr size_t TAIL_END = VENDOR_OFFSET + HL_PCIE_BOOTP_VEND_STATUS + 1;
        if (!packet || packet_len < TAIL_END) {
            return;
        }
        const uint8_t* vend = packet + VENDOR_OFFSET;
        if (vend[0] != HL_PCIE_BOOTP_VENDOR_TAG || vend[1] < HL_PCIE_BOOTP_VEND_TAG_LEN) {
            return;
        }
        // PTP seconds, PTP nanoseconds, PSN and status are contiguous from VEND_PTP_SEC.
        hololink::core::Deserializer tail(vend + HL_PCIE_BOOTP_VEND_PTP_SEC, packet_len - VENDOR_OFFSET - HL_PCIE_BOOTP_VEND_PTP_SEC);
        uint64_t ptp_timestamp_s = 0;
        uint32_t ptp_timestamp_ns = 0;
        uint16_t psn = 0;
        uint8_t status = 0;
        if (!(tail.next_uint48_be(ptp_timestamp_s)
                && tail.next_uint32_be(ptp_timestamp_ns)
                && tail.next_uint16_be(psn)
                && tail.next_uint8(status))) {
            return;
        }
        metadata["ptp_timestamp_s"] = static_cast<int64_t>(ptp_timestamp_s);
        metadata["ptp_timestamp_ns"] = static_cast<int64_t>(ptp_timestamp_ns);
        metadata["bootp_psn"] = static_cast<int64_t>(psn);
        metadata["bootp_status"] = static_cast<int64_t>(status);
    }
};

} // namespace hololink::module::hsb_pcie

#endif /* HOLOLINK_MODULE_HSB_PCIE_HSB_PCIE_PUBLISHER_HPP */
