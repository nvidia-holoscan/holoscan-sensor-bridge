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

/**
 * @brief Find the InfiniBand verbs device and port through which this
 * host reaches a device.
 *
 * A RoCE receiver must use the network adapter that is actually
 * connected to the device it receives from. When the host has more than
 * one adapter, this identifies the right one: the adapter on the
 * network interface used to reach `peer_ip`. Pass the result as the
 * `ibv_name` and `ibv_port` of RoceReceiverInterfaceV1::start().
 *
 * Only RoCE builds provide this function, since it uses the ibverbs
 * library.
 *
 * @param peer_ip IP address of the device. It is used to identify the
 *      network interface only when `interface` is not given; it is
 *      also named in error messages.
 * @param interface Name of the host's network interface that is
 *      connected to the device, as recorded in the enumeration metadata
 *      under "interface" (the interface on which the device's
 *      enumeration message arrived). Pass it whenever it is available.
 *      If empty, the interface is chosen by the host's routing table
 *      for `peer_ip`, which gives the wrong answer when two devices
 *      share the same default data-plane subnet (192.168.0.x): both
 *      resolve to the same adapter, and frames from the other device
 *      are never received.
 * @returns The device name (for example `"roceP5p3s0f0"`) and the
 *      port number on that device. If more than one device and port
 *      match, the first one the system lists is returned.
 * @throws std::runtime_error if the list of devices cannot be read, or
 *      no device and port is bound to the network interface. The message
 *      names `peer_ip` and, in the second case, the interface.
 */
std::pair<std::string, uint32_t> ibv_device_for_peer(
    const std::string& peer_ip, const std::string& interface = std::string());

} // namespace hololink::module

#endif // HOLOLINK_MODULE_IBV_DEVICE_HPP
