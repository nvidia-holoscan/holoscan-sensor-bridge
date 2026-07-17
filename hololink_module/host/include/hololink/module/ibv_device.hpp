/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_IBV_DEVICE_HPP
#define HOLOLINK_MODULE_IBV_DEVICE_HPP

#include <cstdint>
#include <string>
#include <utility>

namespace hololink::module {

/* Choose the IB device + port whose underlying kernel netdev is the
 * route-resolved local interface for `peer_ip`. Used to bind each
 * RoCE QP to the NIC that can ARP its peer when multiple data planes
 * span more than one network interface.
 *
 * Resolution: the peer IP routes through some local kernel interface
 * (resolved by the host's route lookup); the IB device on
 * top of that interface has a GID entry under sysfs at
 * /sys/class/infiniband/<dev>/ports/<port>/gid_attrs/ndevs/<gid_idx>
 * whose contents are the netdev name. This function walks that tree
 * and returns the first (device, port) whose ndevs file matches the
 * local interface.
 *
 * Throws std::runtime_error when no matching device is found, naming
 * the peer and the local interface in the message. */
std::pair<std::string, uint32_t> ibv_device_for_peer(const std::string& peer_ip);

} // namespace hololink::module

#endif // HOLOLINK_MODULE_IBV_DEVICE_HPP
