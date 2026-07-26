/**
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 *
 * See README.md for detailed information.
 */

#include <assert.h>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "base_transmitter.hpp"
#include "data_plane.hpp"
#include "hsb_config.hpp"
#include "hsb_emulator.hpp"
#include "net.hpp"
#include "utils.hpp"

#define BOOTP_PACKET_SIZE 342u

namespace hololink::emulation {

static inline uint32_t get_delta_msecs(const struct timespec& start_time, const struct timespec& end_time)
{
    long long diff_sec = end_time.tv_sec - start_time.tv_sec;
    long long diff_nsec = end_time.tv_nsec - start_time.tv_nsec;
    return static_cast<uint32_t>((diff_sec * 1000) + (diff_nsec / 1000000));
}

int vp_data_readback_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count)
{
    DPSensorRegisters* dp_sensor_registers = (DPSensorRegisters*)ctxt;
    int i = 0;
    while (i < max_count) {
        uint32_t address = AVP_GET_ADDRESS(addr_val);
        if (address >= dp_sensor_registers->vp_address && address <= dp_sensor_registers->vp_address + DP_HOST_UDP_PORT) {
            AVP_SET_VALUE(addr_val, dp_sensor_registers->vp_data[(address - dp_sensor_registers->vp_address) / REGISTER_SIZE]);
        } else {
            return i;
        }
        addr_val++;
        i++;
    }
    return i;
}
int vp_data_configure_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count)
{
    DPSensorRegisters* dp_sensor_registers = (DPSensorRegisters*)ctxt;

    int i = 0;
    while (i < max_count) {
        uint32_t address = AVP_GET_ADDRESS(addr_val);
        if (address >= dp_sensor_registers->vp_address && address <= dp_sensor_registers->vp_address + DP_HOST_UDP_PORT) {
            dp_sensor_registers->vp_data[(address - dp_sensor_registers->vp_address) / REGISTER_SIZE] = AVP_GET_VALUE(addr_val);
        } else {
            return i;
        }
        addr_val++;
        i++;
    }
    return i;
}

int hif_data_readback_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count)
{
    DPRegisters* dp_registers = (DPRegisters*)ctxt;

    int i = 0;
    while (i < max_count) {
        uint32_t address = AVP_GET_ADDRESS(addr_val);
        if (address >= dp_registers->hif_address && address <= dp_registers->hif_address + DP_VP_MASK) {
            AVP_SET_VALUE(addr_val, dp_registers->hif_data[(address - dp_registers->hif_address) / REGISTER_SIZE]);
        } else {
            return i;
        }
        addr_val++;
        i++;
    }
    return i;
}
int hif_data_configure_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count)
{
    DPRegisters* dp_registers = (DPRegisters*)ctxt;

    int i = 0;
    while (i < max_count) {
        uint32_t address = AVP_GET_ADDRESS(addr_val);
        if (address >= dp_registers->hif_address && address <= dp_registers->hif_address + DP_VP_MASK) {
            dp_registers->hif_data[(address - dp_registers->hif_address) / REGISTER_SIZE] = AVP_GET_VALUE(addr_val);
        } else {
            return i;
        }
        addr_val++;
        i++;
    }
    return i;
}

// this is a subset of the HSBConfiguration struct that is part of the BootpPacket
// but it is kept separate because it must be packed.
struct VendorInfo {
    uint8_t tag;
    uint8_t tag_length;
    uint8_t vendor_id[VENDOR_ID_SIZE];
    uint8_t data_plane;
    uint8_t enum_version; // must be v2
    uint8_t board_id_lo;
    uint8_t board_id_hi;
    uint8_t uuid[BOARD_VERSION_SIZE];
    uint8_t serial_num[BOARD_SERIAL_NUM_SIZE];
    // have to break up the next 2 fields because they are not WORD-aligned
    uint8_t hsb_ip_version_lo;
    uint8_t hsb_ip_version_hi;
    uint8_t fpga_crc_lo;
    uint8_t fpga_crc_hi;
};

static_assert(sizeof(VendorInfo) == 41, "VendorInfo must be 41 bytes to ensure it is packed");

void init_bootp_packet(uint8_t* bootp_buffer, IPAddress& ip_address, HSBConfiguration& configuration)
{
    struct ether_header* ether_header = (struct ether_header*)bootp_buffer;
    {
        uint8_t broadcast_addr[] = ETHER_BROADCAST_ADDR;
        memcpy(ether_header->ether_dhost, broadcast_addr, ETHER_ADDR_LEN);
    }
    memcpy(ether_header->ether_shost, ip_address.mac, ETHER_ADDR_LEN);
    ether_header->ether_type = htons(ETHERTYPE_IP);

    struct iphdr* ip_header = (struct iphdr*)(bootp_buffer + ETHER_HDR_LEN);
    IPHDR_SET_IHL_VERSION(ip_header, IP_HDR_LEN, IPVERSION);
    ip_header->tot_len = htons(sizeof(BootpPacket) + IP_HDR_LEN + UDP_HDR_LEN);
    ip_header->id = 0, // not used without fragmentation
        ip_header->frag_off = htons(IP_DF), // 0x4000 not used without fragmentation
        ip_header->ttl = IPDEFTTL, // not used without fragmentation
        ip_header->protocol = IPPROTO_UDP, // UDP
        ip_header->check = 0;
    ip_header->saddr = ip_address.ip_address;
    ip_header->daddr = ip_address.broadcast_address;

    struct udphdr* udp_header = (struct udphdr*)(bootp_buffer + ETHER_HDR_LEN + IP_HDR_LEN);
    udp_header->source = htons(BOOTP_REPLY_PORT);
    udp_header->dest = htons(BOOTP_REQUEST_PORT);
    udp_header->len = htons(sizeof(BootpPacket) + UDP_HDR_LEN);
    udp_header->check = 0;

    struct BootpPacket* bootp_packet = (struct BootpPacket*)(bootp_buffer + ETHER_HDR_LEN + IP_HDR_LEN + UDP_HDR_LEN);

    bootp_packet->op = 1;
    bootp_packet->htype = 1;
    bootp_packet->hlen = 6;
    bootp_packet->xid = htonl(configuration.data_plane);
    bootp_packet->ciaddr = ip_address.ip_address;
    memcpy(bootp_packet->chaddr, ip_address.mac, ETHER_ADDR_LEN);

    VendorInfo vendor_info = {
        .tag = configuration.tag,
        .tag_length = configuration.tag_length,
        .data_plane = configuration.data_plane,
        .enum_version = configuration.enum_version,
        .board_id_lo = configuration.board_id_lo,
        .board_id_hi = configuration.board_id_hi,
        .hsb_ip_version_lo = (uint8_t)(configuration.hsb_ip_version & 0xFF),
        .hsb_ip_version_hi = (uint8_t)(configuration.hsb_ip_version >> 8),
        .fpga_crc_lo = (uint8_t)(configuration.fpga_crc & 0xFF),
        .fpga_crc_hi = (uint8_t)(configuration.fpga_crc >> 8),
    };
    memcpy(vendor_info.vendor_id, configuration.vendor_id, sizeof(configuration.vendor_id));
    memcpy(vendor_info.uuid, configuration.uuid, sizeof(configuration.uuid));
    memcpy(vendor_info.serial_num, configuration.serial_num, sizeof(configuration.serial_num));

    memcpy(bootp_packet->vend, &vendor_info, sizeof(vendor_info));
}

void update_bootp_packet(uint8_t* bootp_buffer, const struct timespec& start_time)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts)) {
        Error_Handler("Failed to get timestamp in update_bootp_packet");
    }
    struct BootpPacket* bootp_packet = (struct BootpPacket*)(bootp_buffer + ETHER_HDR_LEN + IP_HDR_LEN + UDP_HDR_LEN);
    bootp_packet->secs = (uint16_t)(get_delta_msecs(start_time, ts) / 1000);
}

int packetizer_readback_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count)
{
    DPSensorRegisters* dpsr = (DPSensorRegisters*)ctxt;
    int n = 0;
    while (n < max_count) {
        uint32_t address = AVP_GET_ADDRESS(addr_val);
        if (address == dpsr->sif_address + PACKETIZER_MODE) {
            AVP_SET_VALUE(addr_val, dpsr->packetizer_mode);
        } else if (address == dpsr->sif_address + PACKETIZER_RAM) {
            AVP_SET_VALUE(addr_val, dpsr->packetizer_ram_pointer);
        } else if (address == dpsr->sif_address + PACKETIZER_DATA) {
            uint32_t v = (dpsr->packetizer_ram_pointer < PACKETIZER_RAM_DEPTH)
                ? dpsr->packetizer_ram[dpsr->packetizer_ram_pointer]
                : 0u;
            AVP_SET_VALUE(addr_val, v);
        } else {
            return n;
        }
        addr_val++;
        n++;
    }
    return n;
}
int packetizer_configure_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count)
{
    DPSensorRegisters* dpsr = (DPSensorRegisters*)ctxt;
    int n = 0;
    while (n < max_count) {
        uint32_t address = AVP_GET_ADDRESS(addr_val);
        uint32_t value = AVP_GET_VALUE(addr_val);
        if (address == dpsr->sif_address + PACKETIZER_MODE) {
            dpsr->packetizer_mode = value;
        } else if (address == dpsr->sif_address + PACKETIZER_RAM) {
            dpsr->packetizer_ram_pointer = value;
        } else if (address == dpsr->sif_address + PACKETIZER_DATA) {
            // out-of-range LUT writes silently absorbed (matches prior hashmap behavior)
            if (dpsr->packetizer_ram_pointer < PACKETIZER_RAM_DEPTH) {
                dpsr->packetizer_ram[dpsr->packetizer_ram_pointer] = value;
            }
        } else {
            return n;
        }
        addr_val++;
        n++;
    }
    return n;
}

// common initialization code for all data planes
void DataPlane::init(HSBEmulator& hsb_emulator)
{
    // validate against configuration
    if (data_plane_id_ >= configuration_.data_plane_count) {
        Error_Handler("data_plane_id exceeds configured data_plane_count");
    }
    if (sensor_id_ >= configuration_.sensor_count) {
        Error_Handler("sensor_id exceeds configured sensor count");
    }
    if (!(ip_address_.flags & IPADDRESS_HAS_BROADCAST)) {
        Error_Handler("Broadcast address not provided in DataPlane initialization");
    }
    // strictly speaking, this is just a warning.
    if (!(ip_address_.flags & IPADDRESS_HAS_MAC)) {
        fprintf(stderr, "MAC address not found/recognized for ip address %s. For non-COE applications, this is non-fatal\n", IPAddress_to_string(ip_address_).c_str());
    }

    // DataPlaneConfiguration
    configuration_.data_plane = data_plane_id_;

    // SensorConfiguration. See DataChannel::use_sensor() for more details
    uint8_t sif0_index = sensor_id_ * configuration_.sifs_per_sensor;

    // Both slices were wired by the platform-specific DataPlane constructor. dp_registers
    // (hif slice) is shared by every DataPlane bound to the same data_plane_id;
    // dp_sensor_registers (vp slice) is shared by every DataPlane bound to the same sensor_id.
    // The hif/sif/vp register-base addresses are derived from data_plane_id_/sensor_id_ here
    // and stored on the shared structs (DataPlane has no duplicate copies).
    DPRegisters* dp_registers = data_plane_ctxt_->dp_registers;
    DPSensorRegisters* dp_sensor_registers = data_plane_ctxt_->dp_sensor_registers;
    dp_registers->hif_address = 0x02000300 + 0x10000 * data_plane_id_;
    dp_sensor_registers->sif_address = 0x01000000 + 0x10000 * sensor_id_;
    dp_sensor_registers->vp_address = 0x1000 + 0x40 * sif0_index;

    // register callbacks for sif packetizer registers (MODE/RAM/DATA). Multiple DataPlanes
    // sharing a sensor_id all pass the same dp_sensor_registers pointer here, so
    // AddressMap::set silently replaces the existing entry with an equivalent one.
    CHECK_STATUS(hsb_emulator.register_read_callback(dp_sensor_registers->sif_address + PACKETIZER_RAM, dp_sensor_registers->sif_address + PACKETIZER_MODE + REGISTER_SIZE, packetizer_readback_cb, dp_sensor_registers),
        "Failed to register read callback for packetizer registers");
    CHECK_STATUS(hsb_emulator.register_write_callback(dp_sensor_registers->sif_address + PACKETIZER_RAM, dp_sensor_registers->sif_address + PACKETIZER_MODE + REGISTER_SIZE, packetizer_configure_cb, dp_sensor_registers),
        "Failed to register write callback for packetizer registers");

    // register callbacks for data plane vp data. Multiple DataPlanes sharing a sensor_id all
    // pass the same dp_sensor_registers pointer here, so AddressMap::set silently replaces
    // the existing entry with an equivalent one.
    CHECK_STATUS(hsb_emulator.register_read_callback(dp_sensor_registers->vp_address + DP_QP, dp_sensor_registers->vp_address + DP_HOST_UDP_PORT + REGISTER_SIZE, vp_data_readback_cb, dp_sensor_registers),
        "Failed to register read callback for data plane vp data");
    CHECK_STATUS(hsb_emulator.register_write_callback(dp_sensor_registers->vp_address + DP_QP, dp_sensor_registers->vp_address + DP_HOST_UDP_PORT + REGISTER_SIZE, vp_data_configure_cb, dp_sensor_registers),
        "Failed to register write callback for data plane vp data");

    // register callbacks for data plane hif data. Same pattern as vp: multiple DataPlanes
    // sharing a data_plane_id all pass the same dp_registers pointer here.
    CHECK_STATUS(hsb_emulator.register_read_callback(dp_registers->hif_address, dp_registers->hif_address + DP_VP_MASK + REGISTER_SIZE, hif_data_readback_cb, dp_registers),
        "Failed to register read callback for data plane hif data");
    CHECK_STATUS(hsb_emulator.register_write_callback(dp_registers->hif_address, dp_registers->hif_address + DP_VP_MASK + REGISTER_SIZE, hif_data_configure_cb, dp_registers),
        "Failed to register write callback for data plane hif data");

    // register the DataPlane with the HSBEmulator so that it can be started/stopped
    hsb_emulator.add_data_plane(*this);
}

DataPlane::~DataPlane()
{
    stop();
}

// DataPlane::start / DataPlane::stop / DataPlane::is_running live in the platform-specific
// data_plane.cpp files: Linux's variants acquire LinuxDataPlaneCtxt::running_mutex around
// every read/write of base.running; STM32's are unsynchronized (single-threaded main loop).

bool DataPlane::packetizer_enabled() const
{
    return (data_plane_ctxt_->dp_sensor_registers->packetizer_mode >> 28u) & 0x1u;
}

void DataPlane::update_metadata()
{
}

struct timespec DataPlane::get_start_time() const
{
    return data_plane_ctxt_->start_time;
}

} // namespace hololink::emulation
