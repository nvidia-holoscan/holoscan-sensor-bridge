/*
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

#pragma once

#include "hsb_flasher_context.hpp"

#include <cstdint>
#include <functional>
#include <vector>

namespace hsb_flasher {

/**
 * @brief Interface for device enumeration providers.
 *
 * Abstracts the mechanism used to discover HSB devices on the network.
 * Production code uses BootpEnumerationProvider; tests can supply mock
 * implementations that deliver synthetic packets and metadata.
 */
class IEnumerationProvider {
public:
    virtual ~IEnumerationProvider() = default;

    /**
     * Callback invoked for each discovered device.
     * @param packet  Raw BOOTP packet data.
     * @param metadata  Parsed metadata for the device.
     * @return true to continue enumeration, false to stop.
     */
    using Callback = std::function<bool(const std::vector<uint8_t>&, hololink::Metadata&)>;

    /**
     * @brief Run enumeration, invoking @p callback for each device found.
     * @param callback  Called once per discovered device.
     * @param timeout   Maximum enumeration duration in seconds.
     */
    virtual void enumerate(Callback callback, float timeout) = 0;
};

/**
 * @brief Production enumeration provider that uses BOOTP via hololink::Enumerator.
 */
class BootpEnumerationProvider : public IEnumerationProvider {
public:
    void enumerate(Callback callback, float timeout) override;
};

/**
 * @brief Discover the target Hololink HSB device on the network.
 *
 * Uses the supplied @p provider to listen for device announcements and
 * matches responses whose IP address equals @p context.hololink_ip.
 * On a match the device metadata is saved to @p context.enumeration_metadata.
 *
 * The function collects responses for the entire timeout window and uses
 * MAC addresses to distinguish unique devices. If multiple distinct devices
 * (different MACs) are discovered at the same IP address, the function
 * returns false — this is treated as an unrecoverable conflict. Repeated
 * packets from the same device are safely ignored.
 *
 * When vendor data is present in the BOOTP response, the metadata is
 * populated directly from it. When vendor data is absent (legacy devices),
 * the function enriches the metadata with fallback values:
 *   - @c "mac_id"         – extracted from the raw BOOTP packet header.
 *   - @c "hsb_ip_version" – probed via a direct UDP register read.
 *   - @c "fpga_uuid"      – set to "N/A".
 *
 * After a successful call, consumers can access device information via
 * @c context.enumeration_metadata using keys such as @c "peer_ip",
 * @c "mac_id", @c "hsb_ip_version", and @c "fpga_uuid".
 *
 * @param[in,out] context   Manager context. @c hololink_ip and @c timeout
 *                          must be set before calling. On success the
 *                          @c enumeration_metadata field is filled in.
 * @param[in]     provider  Enumeration provider to use for device discovery.
 * @return true if exactly one device was found, false on timeout, conflict,
 *         or error.
 */
bool discover_device(hsb_flasher_context& context, IEnumerationProvider& provider);

/**
 * @brief Convenience overload that uses BOOTP enumeration.
 * @see discover_device(hsb_flasher_context&, IEnumerationProvider&)
 */
bool discover_device(hsb_flasher_context& context);

} // namespace hsb_flasher
