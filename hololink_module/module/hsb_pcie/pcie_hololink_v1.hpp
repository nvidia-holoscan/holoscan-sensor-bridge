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

#ifndef HOLOLINK_MODULE_HSB_PCIE_PCIE_HOLOLINK_V1_HPP
#define HOLOLINK_MODULE_HSB_PCIE_PCIE_HOLOLINK_V1_HPP

/**
 * @file
 * @brief module_core::HololinkV1 whose backing legacy Hololink is a PcieHololink.
 *
 * Overrides only HololinkV1::create_backing() (module/core/hololink_default.hpp): the
 * enumeration metadata names the endpoint by its PCI function address (peer_ip =
 * "DDDD:BB:DD.F", the BDF -- the FPGA function whose BAR0 the control plane lands on, the
 * PCIe counterpart of an Ethernet board's IP) and the carrier (pcie_backend = device |
 * loopback) -- the only knob. The /dev/hololink_pcieN minor is derived from the BDF through
 * sysfs at construction (pcie_sysfs.hpp, header-only because a module
 * .so must not link the host library); minors follow probe order and are never the key.
 * For convenience a peer_ip that is already "/dev/hololink_pcieN" or a
 * "/dev/hololink/by-bdf/<bdf>" udev alias is accepted too (the publisher then requires
 * transport = "pcie"). The loopback backend has no sysfs behind it and takes the minor from
 * the metadata's pcie_minor (default 0).
 *
 * APB sequencer programs are always run by the in-process software sequencer (the endpoint
 * bitfile does not execute them), so there is no engine selection; the removed
 * engine-selection key from earlier builds is ignored with a warning when old metadata still
 * carries it. Everything else -- start/stop/reset/ptp_synchronize delegation, batched
 * write_uint32 / block read_uint32, the reset-callback registry -- is inherited, and the
 * object is published under HololinkV1's own type ids (it IS-A HololinkV1), so I2cV1,
 * SequencerV1 and the data-channel wrappers find it through HololinkV1::get_service unchanged.
 */

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "hololink/module/enumeration_metadata.hpp"

#include "hololink_default.hpp"

#include "address_policy.hpp"

namespace hololink::module::hsb_pcie {

class PcieHololinkV1 : public module_core::HololinkV1 {
public:
    /// Metadata keys read by create_backing() / the publisher's enumeration gate.
    static constexpr const char* KEY_PEER_IP = "peer_ip";
    static constexpr const char* KEY_TRANSPORT = "transport";
    static constexpr const char* KEY_BACKEND = "pcie_backend";
    static constexpr const char* KEY_MINOR = "pcie_minor";
    static constexpr const char* KEY_BDF = "pcie_bdf";
    static constexpr const char* KEY_SERIAL_NUMBER = "serial_number";
    static constexpr const char* TRANSPORT_PCIE = "pcie";
    static constexpr const char* BACKEND_DEVICE = "device";
    static constexpr const char* BACKEND_LOOPBACK = "loopback";
    /// BDF the loopback endpoint reports when the metadata supplies none.
    static constexpr const char* LOOPBACK_DEFAULT_BDF = "0000:00:00.0";

    PcieHololinkV1() = default;

    /// True when `peer_ip` is a PCI function address "DDDD:BB:DD.F" (the PCIe channel key).
    static bool is_bdf_peer(const std::string& peer_ip);
    /// True when `peer_ip` is a dotted-quad IPv4 address (a bootp-shaped Ethernet peer).
    static bool is_ipv4_peer(const std::string& peer_ip);

    /**
     * @brief The hololink_pcieN minor bound to `peer_ip` (BDF, "/dev/hololink_pcieN" or a
     * "/dev/hololink/by-bdf/<bdf>" alias), resolved through sysfs.
     * @throw std::runtime_error naming the peer when it has none of those forms, or naming
     *        the BDF when no hololink_pcie device is bound to it.
     */
    static uint16_t resolve_minor(const std::string& peer_ip);

protected:
    std::shared_ptr<module_core::LegacyHololinkAccess> create_backing(
        const EnumerationMetadata& metadata) override;
};

} // namespace hololink::module::hsb_pcie

#endif /* HOLOLINK_MODULE_HSB_PCIE_PCIE_HOLOLINK_V1_HPP */
