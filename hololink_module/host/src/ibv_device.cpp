/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hololink/module/ibv_device.hpp"

#include <cerrno>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include <infiniband/verbs.h>

#include "hololink/module/logging.hpp" // HSB_LOG_DEBUG
#include "networking.hpp"

namespace hololink::module {

namespace fs = std::filesystem;

static std::string read_first_line(const fs::path& path)
{
    std::ifstream f(path);
    std::string line;
    std::getline(f, line);
    return line;
}

std::pair<std::string, uint32_t> ibv_device_for_peer(const std::string& peer_ip)
{
    const auto [local_ip, local_interface, local_mac]
        = local_ip_and_mac(peer_ip);

    int num_devices = 0;
    ibv_device** devices = ibv_get_device_list(&num_devices);
    if (!devices) {
        throw std::runtime_error(
            "While selecting IB device for peer '" + peer_ip
            + "': ibv_get_device_list failed with errno="
            + std::to_string(errno));
    }

    std::string matched_device;
    uint32_t matched_port = 0;

    for (int i = 0; i < num_devices && matched_device.empty(); ++i) {
        const char* device_name = ibv_get_device_name(devices[i]);
        if (!device_name) {
            continue;
        }
        const fs::path ports_dir
            = fs::path("/sys/class/infiniband").append(device_name).append("ports");
        std::error_code ec;
        if (!fs::is_directory(ports_dir, ec)) {
            continue;
        }
        for (const auto& port_entry : fs::directory_iterator(ports_dir, ec)) {
            if (!port_entry.is_directory()) {
                continue;
            }
            fs::path ndevs_dir = port_entry.path();
            ndevs_dir.append("gid_attrs").append("ndevs");
            if (!fs::is_directory(ndevs_dir, ec)) {
                continue;
            }
            for (const auto& ndev_entry : fs::directory_iterator(ndevs_dir, ec)) {
                if (read_first_line(ndev_entry.path()) == local_interface) {
                    matched_device = device_name;
                    matched_port = static_cast<uint32_t>(
                        std::stoul(port_entry.path().filename().string()));
                    break;
                }
            }
            if (!matched_device.empty()) {
                break;
            }
        }
    }

    ibv_free_device_list(devices);

    if (matched_device.empty()) {
        throw std::runtime_error(
            "While selecting IB device for peer '" + peer_ip
            + "': no IB device found bound to kernel interface '"
            + local_interface + "'");
    }
    // Debug diagnostic: the peer this caller passed in, the local interface
    // local_ip_and_mac resolved for it, and the IB device the QP will bind to.
    // Compare against the data channel's DP_HOST_IP (programmed from the same
    // local_ip_and_mac on the data channel's peer): a mismatch in peer here vs
    // there is the divergence that loses data.
    HSB_LOG_DEBUG(
        "ibv_device_for_peer peer_ip={} -> local_interface={} ib_device={} port={}",
        peer_ip, local_interface, matched_device, matched_port);
    return { matched_device, matched_port };
}

} // namespace hololink::module
