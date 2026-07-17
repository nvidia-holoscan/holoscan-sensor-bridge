/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_BOOTP_HPP
#define HOLOLINK_MODULE_BOOTP_HPP

#include <cstdint>
#include <utility>
#include <vector>

#include "hololink/module/enumeration_metadata.hpp"

/* Host-private bootp-v2 listener. Adapter-owned so the host framework
 * needs no dependency on src/hololink/core/enumerator: it parses the
 * bootp payload directly into EnumerationMetadata. Device-specific
 * enrichment (sensor/data-plane register layout) is intentionally NOT
 * done here — that is the loaded module's EnumerationInterfaceV1::
 * update_metadata, keyed on the data_plane this parser records. */

namespace hololink::module {

/* Enable IP_PKTINFO (so the receive path learns the arrival interface)
 * and bind the UDP socket to `port` on INADDR_ANY. SO_REUSEADDR /
 * SO_REUSEPORT are expected to be set by the caller before binding.
 * Returns true on success; throws on a setsockopt/bind failure. */
bool configure_bootp_socket(int fd, uint32_t port);

/* Receive one datagram from `fd` and parse it into EnumerationMetadata
 * (bootp header, NVDA vendor section, and the v2 board body) along with
 * the raw packet bytes. Mirrors the field set the legacy enumerator
 * produced, minus the legacy strategy / data-channel enrichment. */
std::pair<EnumerationMetadata, std::vector<uint8_t>> receive_bootp(int fd);

} // namespace hololink::module

#endif // HOLOLINK_MODULE_BOOTP_HPP
