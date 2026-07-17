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

#include <chrono>
#include <climits>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include "core/deserializer.hpp"
#include "core/serializer.hpp"

#include "address_memory.hpp"
#include "apb_events.hpp" // apb_ram_{read,write}, async_event_*_cb, handle_apb_sw_event
#include "data_plane.hpp"
#include "hsb_emulator.hpp"
#include "i2c.hpp" // platform i2c.hpp (resolved via include path) for i2c_{configure,readback}_cb
#include "i2c_interface.hpp"
#include "net.hpp"
#include "utils.hpp"

// forward declarations
int GPIO_init(void* ctxt);
int net_init(void* ctxt);
int tim_init(void* ctxt);
int spi_init(void* ctxt);

namespace hololink::emulation {

// forward declaration in namespace
// control_plane_reply is platform specific
void control_plane_reply(HSBEmulatorCtxt* ctxt, ETH_BufferTypeDef* buffer);
// platform-invariant reply prep — defined below
void prepare_control_plane_reply(HSBEmulatorCtxt* ctxt, ETH_BufferTypeDef* buffer, struct ControlMessage& message);

int read_hsb_data(void* ctxt, struct AddressValuePair* addr_val, int max_count)
{
    if (max_count < 1) {
        return 0;
    }
    HSBEmulator* hsb_emulator = ((HSBEmulator*)ctxt);
    const HSBConfiguration& configuration = hsb_emulator->get_config();
    uint32_t address = AVP_GET_ADDRESS(addr_val);
    if (HSB_IP_VERSION == address) {
        AVP_SET_VALUE(addr_val, configuration.hsb_ip_version);
    } else if (FPGA_DATE == address) {
        AVP_SET_VALUE(addr_val, configuration.fpga_crc);
    } else {
        return 0;
    }
    return 1;
}

int reset_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count)
{
    int n = 0;
    while (n < max_count) {
        uint32_t address = AVP_GET_ADDRESS(addr_val);
        if (address != RESET_REG_CTRL && address != SOFT_RESET_REG_CTRL) {
            break;
        }
        addr_val++;
        n++;
    }
    if (n > 0) {
        HSBEmulator* hsb_emulator = ((HSBEmulator*)ctxt);
        hsb_emulator->reset();
    }
    return n;
}

// VSYNC peripheral stub (registers at 0x70000000..0x70000014). Real Eagle
// hardware exposes a VSYNC pulse generator and the hololink client side
// drives it correctly via Hololink::PtpSynchronizer::setup() (writes the
// full CONTROL/FREQUENCY/DELAY/START/EXPOSURE/GPIO block during init).
// What's missing is the emulator-side counterpart: no class under
// src/hololink/emulation/ models the VSYNC peripheral, so writes in the
// 0x70000000+ range have no registered callback. After commit 32726728
// ("HSB Emulator HAL layer for MCUs") tightened RegisterMemory::write()
// to reject unhandled addresses, those previously-tolerated writes now
// return RESPONSE_INVALID_ADDR and abort the client's setup path. This
// stub accepts all writes/reads in the VSYNC range so PtpSynchronizer::
// setup() completes; no state is tracked because the staging clients
// that hit this path don't actually need VSYNC pulses for sim-hsb. Drop
// once a real Vsync*Emulator class is added to src/hololink/emulation/.
constexpr uint32_t kVsyncBase = 0x70000000u;
constexpr uint32_t kVsyncBlockBytes = 6u * REGISTER_SIZE; // CONTROL..GPIO
int vsync_stub_cb(void* /*ctxt*/, struct AddressValuePair* addr_val, int max_count)
{
    int n = 0;
    while (n < max_count) {
        uint32_t address = AVP_GET_ADDRESS(addr_val);
        if (address < kVsyncBase || address >= kVsyncBase + kVsyncBlockBytes) {
            break;
        }
        // No state to track for a stub. Explicitly zero the value so the
        // callback is safe to register on cp_read_map too: read callers can
        // arrive with an uninitialized value field, and "reads return 0" is
        // the documented stub contract.
        AVP_SET_VALUE(addr_val, 0);
        addr_val++;
        n++;
    }
    return n;
}

// callbacks for common registers
int ptp_readback_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count)
{
    struct PTPConfig* ptp_config = (struct PTPConfig*)ctxt;
    struct AddressValuePair* cur = addr_val;
    int n = 0;
    while (n < max_count) {
        uint32_t address = AVP_GET_ADDRESS(cur);
        switch (address) {
        case FPGA_PTP_CTRL:
            AVP_SET_VALUE(cur, ptp_config->ctrl);
            break;
        case FPGA_PTP_DELAY_ASYMMETRY:
            AVP_SET_VALUE(cur, ptp_config->delay_asymetry);
            break;
        case FPGA_PTP_CTRL_DPLL_CFG1:
            AVP_SET_VALUE(cur, ptp_config->dpll_cfg1);
            break;
        case FPGA_PTP_CTRL_DPLL_CFG2:
            AVP_SET_VALUE(cur, ptp_config->dpll_cfg2);
            break;
        case FPGA_PTP_CTRL_DELAY_AVG_FACTOR:
            AVP_SET_VALUE(cur, ptp_config->delay_avg_factor);
            break;
        case FPGA_PTP_SYNC_TS_0:
            AVP_SET_VALUE(cur, ptp_config->sync_ts_0);
            break;
        case FPGA_PTP_SYNC_STAT:
            AVP_SET_VALUE(cur, ptp_config->sync_stat);
            break;
        case FPGA_PTP_OFM:
            AVP_SET_VALUE(cur, ptp_config->ofm);
            break;
        default:
            return (n == 0) ? 0 : n;
        }
        cur++;
        n++;
    }
    return n;
}
int ptp_configure_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count)
{
    struct PTPConfig* ptp_config = (struct PTPConfig*)ctxt;
    struct AddressValuePair* cur = addr_val;
    int n = 0;
    while (n < max_count) {
        uint32_t address = AVP_GET_ADDRESS(cur);
        switch (address) {
        case FPGA_PTP_CTRL:
            ptp_config->ctrl = AVP_GET_VALUE(cur);
            break;
        case FPGA_PTP_DELAY_ASYMMETRY:
            ptp_config->delay_asymetry = AVP_GET_VALUE(cur);
            break;
        case FPGA_PTP_CTRL_DPLL_CFG1:
            ptp_config->dpll_cfg1 = AVP_GET_VALUE(cur);
            break;
        case FPGA_PTP_CTRL_DPLL_CFG2:
            ptp_config->dpll_cfg2 = AVP_GET_VALUE(cur);
            break;
        case FPGA_PTP_CTRL_DELAY_AVG_FACTOR:
            ptp_config->delay_avg_factor = AVP_GET_VALUE(cur);
            break;
        case FPGA_PTP_SYNC_TS_0:
            ptp_config->sync_ts_0 = AVP_GET_VALUE(cur);
            break;
        case FPGA_PTP_SYNC_STAT:
            ptp_config->sync_stat = AVP_GET_VALUE(cur);
            break;
        case FPGA_PTP_OFM:
            ptp_config->ofm = AVP_GET_VALUE(cur);
            break;
        default:
            return (n == 0) ? 0 : n;
        }
        cur++;
        n++;
    }
    return n;
}

I2CController::~I2CController()
{
    stop();
}

uint8_t handle_read_message(HSBEmulatorCtxt* ctxt, struct ControlMessage& message)
{
    uint16_t i = 0;
    auto& cp_read_map = ctxt->cp_read_map;
    while (i < message.num_addresses) {
        uint32_t address = AVP_GET_ADDRESS(message.addr_vals + i);
        const ControlPlaneCallback* handler = cp_read_map.get(address);
        if (!handler) {
            return ECB_ADDRESS_ERROR;
        }
        int count = message.num_addresses - i;
        int ret = handler->callback(handler->ctxt, message.addr_vals + i, count);
        if (ret <= 0) {
            return ECB_ADDRESS_ERROR;
        }
        i += ret;
    }
    return ECB_SUCCESS;
}

uint8_t handle_write_message(HSBEmulatorCtxt* ctxt, struct ControlMessage& message)
{
    uint16_t i = 0;
    auto& cp_write_map = ctxt->cp_write_map;
    while (i < message.num_addresses) {
        uint32_t address = AVP_GET_ADDRESS(message.addr_vals + i);
        const ControlPlaneCallback* handler = cp_write_map.get(address);
        if (!handler) {
            return ECB_ADDRESS_ERROR;
        }
        int count = message.num_addresses - i;
        int ret = handler->callback(handler->ctxt, message.addr_vals + i, count);
        if (ret <= 0) {
            return ECB_ADDRESS_ERROR;
        }
        i += ret;
    }
    return ECB_SUCCESS;
}

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define AVP_NTOHL(u8p) (*(uint32_t*)(u8p))
#define AVP_HTONL(u8p, u32) *(uint32_t*)(u8p) = u32
#else
#define AVP_NTOHL(u8p) (uint32_t)((u8p)[0] << 24 | (u8p)[1] << 16 | (u8p)[2] << 8 | (u8p)[3])
// the order of the u8p accesses are very important here to ensure the data is preserved
// this is not a generic HTONL, instead it is specific to the bit shift that needs to happen
// shifts right 2 bits left 4 and swaps the inner bits. requires 5 bytes in u8p
#define AVP_HTONL(u8p, u32) \
    (u8p[0]) = (u32) >> 24; \
    (u8p[1]) = (u32) >> 16; \
    (u8p)[4] = (u32) >> 8;  \
    (u8p)[3] = (u32);       \
    (u8p[2]) = (u8p)[4];
#endif

// this assumes a buffer size that can accommodate shifting addr_vals_unaligned by 2 bytes
struct AddressValuePair* align_address_value_pairs(uint8_t* addr_vals_unaligned, int num_addresses)
{
    struct AddressValuePair* addr_vals_aligned = (struct AddressValuePair*)(addr_vals_unaligned + 2) + num_addresses;
    addr_vals_unaligned += (num_addresses - 1) * sizeof(struct AddressValuePair) + sizeof(uint32_t);
    while (num_addresses--) {
        addr_vals_aligned--;
        addr_vals_aligned->value = AVP_NTOHL(addr_vals_unaligned);
        addr_vals_unaligned -= sizeof(uint32_t);
        addr_vals_aligned->address = AVP_NTOHL(addr_vals_unaligned);
        addr_vals_unaligned -= sizeof(uint32_t);
    }
    return addr_vals_aligned;
}

// this assumes the underlying buffer in addr_vals_aligned is already aligned and can be shifted left by 2 bytes
void unalign_address_value_pairs(struct AddressValuePair* addr_vals_aligned, int num_addresses)
{
    uint8_t* addr_vals_unaligned = (uint8_t*)(addr_vals_aligned)-2;
    while (num_addresses--) {
        AVP_HTONL(addr_vals_unaligned, addr_vals_aligned->address);
        addr_vals_unaligned += sizeof(uint32_t);
        AVP_HTONL(addr_vals_unaligned, addr_vals_aligned->value);
        addr_vals_unaligned += sizeof(uint32_t);
        addr_vals_aligned++;
    }
}

static inline uint16_t control_message_n_addresses(size_t payload_length)
{
    // if reading or writing one word, host will only send one word, but if reading and writing in blocks, it will send 2 words per address.
    // so we add one word and divide by 2-word pairs to get the number of addresses in message
    return ((payload_length - CONTROL_MESSAGE_HEADER_LENGTH) + sizeof(uint32_t)) / (sizeof(uint32_t) * 2);
}

void handle_control_packet(HSBEmulatorCtxt* ctxt, ETH_BufferTypeDef* buffer)
{
    struct udphdr* udp_hdr = NET_GET_UDP_HDR(buffer);
    uint8_t* buffer_start = NET_GET_UDP_PAYLOAD(udp_hdr);

    struct ControlMessage message = {
        .ctrl_hdr = (struct ControlMessageHeader*)buffer_start,
        .num_addresses = control_message_n_addresses(ntohs(udp_hdr->len) - UDP_HDR_LEN),
    };
    message.addr_vals = align_address_value_pairs(buffer_start + CONTROL_MESSAGE_HEADER_LENGTH, message.num_addresses);

    // TODO: make this a lookup table for better performance
    switch (message.ctrl_hdr->cmd_code) {
    case RD_BYTE:
    case RD_WORD:
    case RD_DWORD:
    case RD_BLOCK:
        // handle_read_message(message, control_socket, &host_addr, host_addr_len);
        message.ctrl_hdr->status = handle_read_message(ctxt, message);
        break;
    case WR_BYTE:
    case WR_WORD:
    case WR_DWORD:
    case WR_BLOCK: {
        message.ctrl_hdr->status = handle_write_message(ctxt, message);
        break;
    }
    default: {
        message.ctrl_hdr->status = ECB_COMMAND_ERROR;
        break;
    }
    }

    if (message.ctrl_hdr->flags & REQUEST_FLAGS_ACK_REQUEST) {
        unalign_address_value_pairs(message.addr_vals, message.num_addresses);
        // Common prep mutates the buffer in place (reply bit, latched sequence trailer,
        // L2/L3/L4 endpoint swaps) and updates buffer->len to the new outgoing frame
        // length. Platform control_plane_reply only handles the wire emit.
        prepare_control_plane_reply(ctxt, buffer, message);
        control_plane_reply(ctxt, buffer);
    }
}

// Platform-invariant control-plane reply prep. Flips the reply bit on cmd_code, appends
// the 4-byte latched-sequence trailer, swaps the L2/L3/L4 endpoint fields, and updates
// buffer->len to the outgoing Ethernet frame size so the same buffer can be handed
// straight to the platform's wire emit. Called from handle_control_packet() — platform
// control_plane_reply() implementations only do the transport call.
//
// Linux's transport (BSD sendto) rebuilds the L2/L3 headers from socket state, so the
// UDP/IP/Eth swaps here are dead on Linux but harmless — the prepared buffer's headers
// aren't read out by the kernel. STM32 transmits the prepared frame raw via
// HAL_ETH_Transmit and so does consume the prepared headers.
void prepare_control_plane_reply(HSBEmulatorCtxt* ctxt, ETH_BufferTypeDef* buffer, struct ControlMessage& message)
{
    (void)ctxt;
    struct ether_header* eth_hdr = NET_GET_ETHER_HDR(buffer);
    struct iphdr* ip_hdr = NET_GET_IP_HDR(buffer);
    struct udphdr* udp_hdr = NET_GET_UDP_HDR(buffer);

    // mark cmd_code as a reply
    message.ctrl_hdr->cmd_code = (uint8_t)0x80U | message.ctrl_hdr->cmd_code;

    // append the 4-byte latched-sequence trailer after the (address, value) pairs
    uint32_t control_payload_length = CONTROL_MESSAGE_HEADER_LENGTH + 2 * message.num_addresses * sizeof(uint32_t);
    uint8_t* message_end = ((uint8_t*)(message.ctrl_hdr)) + control_payload_length;
    memset(message_end, 0, sizeof(uint32_t));
    control_payload_length += sizeof(uint32_t);

    // udp header: swap ports, set length, clear checksum
    udp_hdr->len = htons(control_payload_length + UDP_HDR_LEN);
    udp_hdr->check = 0;
    {
        uint16_t port = udp_hdr->source;
        udp_hdr->source = udp_hdr->dest;
        udp_hdr->dest = port;
    }

    // ip header: swap addrs, reset ttl, set total length, clear checksum
    ip_hdr->ttl = IPDEFTTL;
    ip_hdr->check = 0;
    {
        uint32_t ip = ip_hdr->saddr;
        ip_hdr->saddr = ip_hdr->daddr;
        ip_hdr->daddr = ip;
    }
    ip_hdr->tot_len = htons(control_payload_length + UDP_HDR_LEN + IP_HDR_LEN);

    // ether header: swap source/destination MACs from the inbound frame. For unicast
    // control traffic the incoming ether_dhost is the local MAC, so swapping is the
    // right reply source without consulting platform state.
    {
        uint8_t mac[ETH_ALEN];
        memcpy(mac, eth_hdr->ether_dhost, ETH_ALEN);
        memcpy(eth_hdr->ether_dhost, eth_hdr->ether_shost, ETH_ALEN);
        memcpy(eth_hdr->ether_shost, mac, ETH_ALEN);
    }

    // refresh the buffer's own length so calculate_buffer_length(buffer) (or buffer->len
    // for single-segment buffers) gives the outgoing Ethernet frame size.
    buffer->len = control_payload_length + UDP_HDR_LEN + IP_HDR_LEN + ETHER_HDR_LEN;
}

HSBEmulator::HSBEmulator()
    : HSBEmulator(HSB_EMULATOR_CONFIG)
{
}

AddressMemory& HSBEmulator::get_memory()
{
    return ctxt_->register_memory;
}

// Platform-invariant control-plane callback registrations. The ranges and callback functions
// here are byte-identical between Linux and STM32 — only the platform-specific extras (GPIO/
// SPI on STM32, peripheral attach on Linux) remain in the platform constructor.
//
// Preconditions: ctxt_ is allocated and i2c_controller_ is constructed. reset() wires
// the ctxt_ back-pointer and the register_memory dispatch ctxt, then registers the
// platform-invariant callbacks.
void HSBEmulator::reset()
{
    if (is_running()) {
        stop();
        start();
    } else {
        // need to clear out memory
        // Wire the ctxt back-pointer to `this` and tell RegisterMemory which ctxt to
        // dispatch reads/writes against. Done here (rather than in each platform
        // constructor) so the assignments can never be skipped by a new port.
        ctxt_->hsb_emulator = this;
        ctxt_->register_memory.set_ctxt(ctxt_.get());

        // basic data read
        CHECK_STATUS(ctxt_->cp_read_map.set({ HSB_IP_VERSION, HSB_IP_VERSION + 2 * REGISTER_SIZE }, { read_hsb_data, (void*)this }),
            "Failed to register read callback for hsb ip version");

        // resets
        CHECK_STATUS(ctxt_->cp_write_map.set({ RESET_REG_CTRL, SOFT_RESET_REG_CTRL + REGISTER_SIZE }, { reset_cb, (void*)this }),
            "Failed to register write callback for resets");

        // ptp configuration
        CHECK_STATUS(ctxt_->cp_write_map.set({ FPGA_PTP_CTRL, FPGA_PTP_CTRL_DELAY_AVG_FACTOR + REGISTER_SIZE }, { ptp_configure_cb, &ctxt_->ptp_config }),
            "Failed to register write callback for ptp configuration");
        CHECK_STATUS(ctxt_->cp_write_map.set({ FPGA_PTP_SYNC_TS_0, FPGA_PTP_OFM + REGISTER_SIZE }, { ptp_configure_cb, &ctxt_->ptp_config }),
            "Failed to register write callback for ptp sync ts 0");
        CHECK_STATUS(ctxt_->cp_read_map.set({ FPGA_PTP_CTRL, FPGA_PTP_CTRL_DELAY_AVG_FACTOR + REGISTER_SIZE }, { ptp_readback_cb, &ctxt_->ptp_config }),
            "Failed to register read callback for ptp ctrl");
        CHECK_STATUS(ctxt_->cp_read_map.set({ FPGA_PTP_SYNC_TS_0, FPGA_PTP_OFM + REGISTER_SIZE }, { ptp_readback_cb, &ctxt_->ptp_config }),
            "Failed to register read callback for ptp sync ts 0");

        // apb ram read/write
        CHECK_STATUS(ctxt_->cp_read_map.set({ APB_RAM, APB_RAM + APB_RAM_DATA_SIZE }, { apb_ram_read, &ctxt_->apb_ram_data[0] }),
            "Failed to register read callback for apb ram");
        CHECK_STATUS(ctxt_->cp_write_map.set({ APB_RAM, APB_RAM + APB_RAM_DATA_SIZE }, { apb_ram_write, &ctxt_->apb_ram_data[0] }),
            "Failed to register write callback for apb ram");

        // async events
        CHECK_STATUS(ctxt_->cp_read_map.set({ CTRL_EVENT, CTRL_EVT_SW_EVENT }, { async_event_readback_cb, &ctxt_->async_event_ctxt }),
            "Failed to register read callback for async events");
        CHECK_STATUS(ctxt_->cp_read_map.set({ CTRL_EVT_STAT, CTRL_EVT_STAT + REGISTER_SIZE }, { async_event_readback_cb, &ctxt_->async_event_ctxt }),
            "Failed to register read callback for async event stat");
        CHECK_STATUS(ctxt_->cp_write_map.set({ CTRL_EVENT, CTRL_EVT_SW_EVENT }, { async_event_configure_cb, &ctxt_->async_event_ctxt }),
            "Failed to register write callback for async event configure");
        CHECK_STATUS(ctxt_->cp_write_map.set({ CTRL_EVT_SW_EVENT, CTRL_EVT_SW_EVENT + REGISTER_SIZE }, { handle_apb_sw_event, ctxt_.get() }),
            "Failed to register write callback for async event sw event");

        // VSYNC peripheral stub — accept writes/reads from Hololink::PtpSynchronizer's
        // setup path (CONTROL/FREQUENCY/DELAY/START/EXPOSURE/GPIO at 0x70000000+).
        // Real hardware models these; the emulator stubs them so client setup completes.
        CHECK_STATUS(ctxt_->cp_write_map.set({ kVsyncBase, kVsyncBase + kVsyncBlockBytes }, { vsync_stub_cb, nullptr }),
            "Failed to register VSYNC stub write callback");
        CHECK_STATUS(ctxt_->cp_read_map.set({ kVsyncBase, kVsyncBase + kVsyncBlockBytes }, { vsync_stub_cb, nullptr }),
            "Failed to register VSYNC stub read callback");

        // i2c register block
        auto i2c_ctxt = i2c_controller_.ctxt_.get();
        CHECK_STATUS(ctxt_->cp_read_map.set({ I2C_CTRL + I2C_REG_DATA_BUFFER, I2C_CTRL + I2C_REG_DATA_BUFFER + I2C_DATA_BUFFER_SIZE }, { i2c_readback_cb, i2c_ctxt }),
            "Failed to register read callback for i2c data buffer");
        CHECK_STATUS(ctxt_->cp_read_map.set({ I2C_CTRL, I2C_CTRL + I2C_REG_CLK_CNT + REGISTER_SIZE }, { i2c_readback_cb, i2c_ctxt }),
            "Failed to register read callback for i2c clk cnt");
        CHECK_STATUS(ctxt_->cp_read_map.set({ I2C_CTRL + I2C_REG_STATUS, I2C_CTRL + I2C_REG_STATUS + REGISTER_SIZE }, { i2c_readback_cb, i2c_ctxt }),
            "Failed to register read callback for i2c status");
        CHECK_STATUS(ctxt_->cp_write_map.set({ I2C_CTRL + I2C_REG_DATA_BUFFER, I2C_CTRL + I2C_REG_DATA_BUFFER + I2C_DATA_BUFFER_SIZE }, { i2c_configure_cb, i2c_ctxt }),
            "Failed to register write callback for i2c data buffer");
        CHECK_STATUS(ctxt_->cp_write_map.set({ I2C_CTRL, I2C_CTRL + I2C_REG_CLK_CNT + REGISTER_SIZE }, { i2c_configure_cb, i2c_ctxt }),
            "Failed to register write callback for i2c clk cnt");
        CHECK_STATUS(ctxt_->cp_write_map.set({ I2C_CTRL + I2C_REG_STATUS, I2C_CTRL + I2C_REG_STATUS + REGISTER_SIZE }, { i2c_configure_cb, i2c_ctxt }),
            "Failed to register write callback for i2c status");
    }
}

int HSBEmulator::write(uint32_t address, uint32_t value)
{
    struct AddressValuePair address_value = { address, value };
    // Propagate the dispatch result so callers can detect RESPONSE_INVALID_ADDR-
    // class failures (e.g. writes to addresses with no registered callback).
    // Previously this always returned 0, silently masking failed writes.
    return get_memory().write(address_value);
}

int HSBEmulator::read(uint32_t address, uint32_t& value)
{
    struct AddressValuePair address_value = { address, 0 };
    int rc = get_memory().read(address_value);
    value = address_value.value;
    return rc;
}

int HSBEmulator::register_read_callback(uint32_t start_address, uint32_t end_address, ControlPlaneCallback_f callback, void* ctxt)
{
    return ctxt_->cp_read_map.set({ start_address, end_address }, { callback, ctxt });
}

int HSBEmulator::register_write_callback(uint32_t start_address, uint32_t end_address, ControlPlaneCallback_f callback, void* ctxt)
{
    return ctxt_->cp_write_map.set({ start_address, end_address }, { callback, ctxt });
}

}
