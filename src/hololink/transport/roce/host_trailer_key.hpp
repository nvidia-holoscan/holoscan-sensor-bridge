/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#ifndef SRC_HOLOLINK_TRANSPORT_ROCE_HOST_TRAILER_KEY_HPP
#define SRC_HOLOLINK_TRANSPORT_ROCE_HOST_TRAILER_KEY_HPP

#include <cstdint>
#include <memory>

#include "cpu_roce_transceiver_internal.hpp"

struct mlx5dv_mkey;

namespace hololink::transport::roce::detail {

/**
 * mlx5 indirect memory key that leaves each slot's payload in the RX region
 * and writes its trailer to host memory. Held by unique_ptr so release runs
 * the destructor: the key goes before both regions. Member-wise assignment
 * would release them in the opposite order.
 */
struct HostTrailerKey {
    HostAllocation trailers;
    ibv::MemoryRegion region;
    std::unique_ptr<::mlx5dv_mkey, int (*)(::mlx5dv_mkey*)> mkey { nullptr, nullptr };
    std::uint32_t rkey = 0;
};

/**
 * Slot i maps [i * stride, (i + 1) * stride - trailer_bytes) to `payload` at
 * the same offset, and its last trailer_bytes to trailers + i * trailer_bytes.
 * Null when libmlx5 or the NIC declines.
 */
std::unique_ptr<HostTrailerKey> try_host_trailer_key(ibv::Context& context,
    ibv::ProtectionDomain& protection_domain, std::uint8_t port_num,
    ::ibv_mtu path_mtu, const LocalRoceV2Ipv4Gid& gid,
    const ibv::MemoryRegion& payload, std::uint64_t stride,
    std::uint32_t trailer_bytes, std::uint32_t slots) noexcept;

} // namespace hololink::transport::roce::detail

#endif
