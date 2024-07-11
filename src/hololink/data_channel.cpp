/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "data_channel.hpp"

#include <arpa/inet.h>

#include <holoscan/logger/logger.hpp>

namespace hololink {

namespace {

    // This memory map used by the Enumeraotr is only supported on CPNX FPGAs that are this
    // version or newer.
    constexpr int64_t MINIMUM_CPNX_VERSION = 0x2402;

    // Camera Receiver interfaces
    constexpr uint32_t VP_START[] { 0x00, 0x80 };

} // anonymous namespace

DataChannel::DataChannel(const Metadata& metadata, const std::function<std::shared_ptr<Hololink>(const Metadata& metadata)>& create_hololink)
{
    auto cpnx_version = metadata.get<int64_t>("cpnx_version"); // or None
    if (!cpnx_version) {
        throw UnsupportedVersion("No 'cpnx_version' field found.");
    }
    if (cpnx_version.value() < MINIMUM_CPNX_VERSION) {
        throw UnsupportedVersion(fmt::format("cpnx_version={:#X}; minimum supported version={:#X}.",
            cpnx_version.value(), MINIMUM_CPNX_VERSION));
    }
    hololink_ = create_hololink(metadata);
    address_ = metadata.get<int64_t>("configuration_address").value();
    peer_ip_ = metadata.get<std::string>("peer_ip").value();
    vip_mask_ = metadata.get<int64_t>("vip_mask").value();
    qp_number_ = 0;
    rkey_ = 0;
}

/*static*/ bool DataChannel::enumerated(const Metadata& metadata)
{
    if (!metadata.get<int64_t>("configuration_address")) {
        return false;
    }
    if (!metadata.get<std::string>("peer_ip")) {
        return false;
    }
    return Hololink::enumerated(metadata);
}

std::shared_ptr<Hololink> DataChannel::hololink() const { return hololink_; }

const std::string& DataChannel::peer_ip() const { return peer_ip_; }

void DataChannel::authenticate(uint32_t qp_number, uint32_t rkey)
{
    qp_number_ = qp_number;
    rkey_ = rkey;
}

void DataChannel::configure(uint64_t frame_address, uint64_t frame_size, uint32_t local_data_port)
{
    const uint32_t header_size = 78;
    const uint32_t cache_size = 128;
    const uint32_t mtu = 1472; // TCP/IP illustrated vol 1 (1994), section 11.6, page 151
    const uint32_t payload_size = ((mtu - header_size + cache_size - 1) / cache_size) * cache_size;
    const uint64_t packets = (frame_size + payload_size - 1) / payload_size; // round up
    HOLOSCAN_LOG_INFO(
        "header_size={} payload_size={} packets={}", header_size, payload_size, packets);
    const std::string& peer_ip = this->peer_ip();
    auto [local_ip, local_device, local_mac] = native::local_ip_and_mac(peer_ip);
    configure_internal(frame_size, payload_size, header_size, local_mac, local_ip, local_data_port,
        qp_number_, rkey_, frame_address, frame_size);
}

bool DataChannel::write_uint32(uint32_t reg, uint32_t value)
{
    return hololink_->write_uint32(address_ + reg, value);
}

void DataChannel::configure_internal(uint64_t frame_size, uint32_t payload_size,
    uint32_t header_size, const native::MacAddress& local_mac, const std::string& local_ip,
    uint32_t local_data_port, uint32_t qp_number, uint32_t rkey, uint64_t address, uint64_t size)
{
    // This is for FPGA 0116 in classic data plane mode
    const uint32_t mac_high = (local_mac[0] << 8) | (local_mac[1] << 0);
    const uint32_t mac_low
        = ((local_mac[2] << 24) | (local_mac[3] << 16) | (local_mac[4] << 8) | (local_mac[5] << 0));

    const in_addr_t ip = inet_network(local_ip.c_str());

    // Clearing DP_VIP_MASK should be unnecessary-- we should only
    // be here following a reset, but be defensive and make
    // sure we're not transmitting anything while we update.
    write_uint32(DP_VIP_MASK, 0);
    write_uint32(DP_PACKET_SIZE, header_size + payload_size);
    write_uint32(DP_HOST_MAC_LOW, mac_low);
    write_uint32(DP_HOST_MAC_HIGH, mac_high);
    write_uint32(DP_HOST_IP, ip);
    write_uint32(DP_HOST_UDP_PORT, local_data_port);
    //
    // "31:28 = end buf
    //  27:24 = start buf
    //  23: 0 = qp"
    // Only use DMA descriptor ("buf") 0.
    // We write the same addressing information into both VPs for
    // this ethernet port; DP_VIP_MASK from the map above selects
    // which one of these is actually used in the hardware.
    for (auto&& vp : VP_START) {
        write_uint32(DP_ROCE_CFG + vp, qp_number & 0x00FF'FFFF);
        write_uint32(DP_ROCE_RKEY_0 + vp, rkey);
        write_uint32(DP_ROCE_VADDR_MSB_0 + vp, (address >> 32));
        write_uint32(DP_ROCE_VADDR_LSB_0 + vp, (address & 0xFFFF'FFFF));
        write_uint32(DP_ROCE_BUF_END_MSB_0 + vp, ((address + size) >> 32));
        write_uint32(DP_ROCE_BUF_END_LSB_0 + vp, ((address + size) & 0xFFFF'FFFF));
    }
    // 0x1 meaning to connect sensor 1 to the current ethernet port
    write_uint32(DP_VIP_MASK, vip_mask_);
}

} // namespace hololink
