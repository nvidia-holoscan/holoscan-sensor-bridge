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

// PcieHololinkV1::create_backing(): the one module_core extension point the hsb_pcie module
// overrides (see module/core/hololink_default.cpp for the Ethernet default it replaces).
// Builds AddressPolicy + endpoint (PcieEndpoint or LoopbackPcieEndpoint) + PcieHololink from
// the enumeration metadata. The endpoint is named by its PCI function (peer_ip = BDF); the
// /dev/hololink_pcieN minor the driver assigned is looked up in sysfs here, through the
// header-only resolvers the host enumerator shares (a module .so cannot link hololink::module).

#include "pcie_hololink_v1.hpp"

#include <cctype>
#include <cstdlib>
#include <optional>
#include <stdexcept>

#include <fmt/format.h>

#include "pcie_regs.hpp"
#include "pcie_sysfs.hpp"
#include <hololink/module/logging.hpp>

#include "loopback_pcie_endpoint.hpp"
#include "pcie_endpoint.hpp"
#include "pcie_hololink.hpp"

namespace hololink::module::hsb_pcie {

namespace {
    /// Removed engine-selection key; tolerated (and ignored) so old metadata still enumerates.
    constexpr const char* LEGACY_KEY_APB_MODE = "pcie_apb_mode";
} // namespace

bool PcieHololinkV1::is_bdf_peer(const std::string& peer_ip)
{
    return pcie::is_bdf(peer_ip);
}

bool PcieHololinkV1::is_ipv4_peer(const std::string& peer_ip)
{
    // Four dot-separated decimal groups of 1..3 digits: what bootp puts in peer_ip.
    unsigned groups = 0;
    size_t digits = 0;
    for (size_t i = 0; i <= peer_ip.size(); i++) {
        if (i == peer_ip.size() || peer_ip[i] == '.') {
            if (digits == 0 || digits > 3) {
                return false;
            }
            groups++;
            digits = 0;
        } else if (std::isdigit(static_cast<unsigned char>(peer_ip[i]))) {
            digits++;
        } else {
            return false;
        }
    }
    return groups == 4;
}

uint16_t PcieHololinkV1::resolve_minor(const std::string& peer_ip)
{
    // An explicit node names its minor outright; no sysfs needed.
    if (const auto minor = pcie::minor_from_device_node(peer_ip)) {
        return *minor;
    }
    const std::optional<std::string> bdf = pcie::bdf_for_peer(peer_ip);
    if (!bdf) {
        throw std::runtime_error(fmt::format(
            "hsb_pcie: peer_ip '{}' is not a PCIe channel key; expected the endpoint's PCI "
            "address 'DDDD:BB:DD.F' (e.g. '0005:06:00.0'), a /dev/{}N node or a /dev/{}/<bdf> alias",
            peer_ip, pcie::PCIE_DEVICE_NAME, pcie::BY_BDF_ALIAS_SUBDIR));
    }
    const std::optional<uint16_t> minor = pcie::minor_for_bdf(*bdf);
    if (!minor) {
        throw std::runtime_error(fmt::format(
            "hsb_pcie: no hololink_pcie device is bound to PCI function {} (peer_ip '{}'): "
            "neither {}/{}/{} nor {} lists it; is the endpoint present and bound to the "
            "hololink_pcie driver?",
            *bdf, peer_ip, pcie::sysfs_pci_devices_dir(pcie::DEFAULT_SYSFS_CLASS).string(), *bdf,
            pcie::PCIE_DEVICE_NAME, pcie::DEFAULT_SYSFS_CLASS));
    }
    return *minor;
}

std::shared_ptr<module_core::LegacyHololinkAccess> PcieHololinkV1::create_backing(
    const EnumerationMetadata& metadata)
{
    const std::string peer_ip = metadata.get<std::string>(KEY_PEER_IP);
    const std::string serial_number = metadata.get<std::string>(KEY_SERIAL_NUMBER);
    const std::string backend = metadata.get<std::string>(KEY_BACKEND, std::string(BACKEND_DEVICE));
    if (metadata.contains(LEGACY_KEY_APB_MODE)) {
        // Older metadata selected an APB engine; the in-process software sequencer is the
        // only engine now (the endpoint bitfile does not execute CTRL_BUS_EVT/EVT_RAM programs).
        HSB_LOG_WARN("hsb_pcie: ignoring metadata key '{}' (the software APB sequencer is the only engine)",
            LEGACY_KEY_APB_MODE);
    }

    // The policy pins the I2C CLK_CNT from the CertusPro-NX Versa's 60 MHz APB clock, the
    // only board this module is built for (its UUID names the .so).
    auto policy = std::make_shared<AddressPolicy>();

    std::shared_ptr<PcieEndpointInterface> endpoint;
    uint16_t minor = 0;
    if (backend == BACKEND_DEVICE) {
        // The BDF is authoritative; the minor is whatever the driver bound to that function
        // right now (the enumerator's pcie_minor may be stale after a reload).
        minor = resolve_minor(peer_ip);
        endpoint = std::make_shared<PcieEndpoint>(minor);
    } else if (backend == BACKEND_LOOPBACK) {
        // No sysfs behind the loopback: the minor is what the metadata claims and the BDF is
        // the peer_ip when it is one (else pcie_bdf, else the default), so identity checks
        // line up with what the enumerator would have announced.
        const int64_t claimed = metadata.get<int64_t>(KEY_MINOR, int64_t { 0 });
        if (claimed < 0 || claimed > 0xFFFF) {
            throw std::runtime_error(fmt::format("hsb_pcie: {} {} is out of range", KEY_MINOR, claimed));
        }
        minor = static_cast<uint16_t>(claimed);
        const std::string bdf = is_bdf_peer(peer_ip)
            ? pcie::normalize_bdf(peer_ip)
            : metadata.get<std::string>(KEY_BDF, std::string(LOOPBACK_DEFAULT_BDF));
        endpoint = std::make_shared<LoopbackPcieEndpoint>(minor, bdf);
    } else {
        throw std::runtime_error(fmt::format(
            "hsb_pcie: unknown {} '{}' (expected '{}' or '{}')", KEY_BACKEND, backend, BACKEND_DEVICE, BACKEND_LOOPBACK));
    }

    HSB_LOG_INFO("hsb_pcie backing: peer={} minor={} backend={} serial={} apb_clk={} Hz (I2C CLK_CNT {})",
        peer_ip, minor, backend, serial_number, policy->apb_clk_hz(), policy->i2c_clk_cnt());
    return std::make_shared<PcieHololink>(std::move(endpoint), std::move(policy), peer_ip, serial_number);
}

} // namespace hololink::module::hsb_pcie
