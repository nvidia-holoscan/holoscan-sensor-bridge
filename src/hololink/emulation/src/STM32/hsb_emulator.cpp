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
 *
 * See README.md for detailed information.
 */

#include <functional>

#include "STM32/gpio.hpp"
#include "STM32/hsb_config.hpp"
#include "STM32/hsb_emulator.hpp"
#include "STM32/i2c.hpp"
#include "STM32/spi.hpp"
#include "STM32/stm32_system.h"
#include "STM32/tim.h"
#include "data_plane.hpp"
#include <climits>
#include <cstring>

namespace hololink::emulation {

static struct HSBEmulatorCtxt HSBEMULATORCTXT;

void HSBEmulator_deleter(__attribute__((unused)) struct HSBEmulatorCtxt* ctxt)
{
    *ctxt = HSBEmulatorCtxt();
}

int read_hsb_data(void* ctxt, struct AddressValuePair* addr_val, int max_count);
int ptp_readback_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count);
int ptp_configure_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count);
int reset_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count);

HSBEmulator::HSBEmulator(const HSBConfiguration& config)
    : ctxt_(&HSBEMULATORCTXT)
    , configuration_(config)
    , i2c_controller_(*this, hololink::I2C_CTRL)
{
    const char* config_error = validate_configuration(&configuration_);
    if (config_error) {
        Error_Handler();
    }

    // initialize the system, but leave individual modules uninitialized
    MPU_Config();
    HAL_Init();
    SystemClock_Config();
    // requires HAL_init. Here before net_init because the MAC address is used by DataPlane before the network is initialized.
    generate_mac_address();

    this->ctxt_.get_deleter() = HSBEmulator_deleter;
    ctxt_->hsb_emulator = this;
    ctxt_->register_memory.set_ctxt(ctxt_.get());
    ctxt_->spi_ctxt = &SPI_CONTROLLER_CTXT;
    spi_constructor(ctxt_->spi_ctxt, SPI_CTRL);
    this->registers_.reset(&ctxt_->register_memory);
    this->registers_.get_deleter() = [](AddressMemory* p) { (void)p; };

    // set basic read/write handlers
    // register callbacks for basic data read
    CHECK_CP_MAP_SET(ctxt_->cp_read_map.set({ HSB_IP_VERSION, HSB_IP_VERSION + 2 * REGISTER_SIZE }, { read_hsb_data, (void*)this }));

    // register callbacks for resets
    CHECK_CP_MAP_SET(ctxt_->cp_write_map.set({ RESET_REG_CTRL, SOFT_RESET_REG_CTRL + REGISTER_SIZE }, { reset_cb, (void*)this }));

    // register callbacks for ptp configuration
    CHECK_CP_MAP_SET(ctxt_->cp_write_map.set({ FPGA_PTP_CTRL, FPGA_PTP_CTRL_DELAY_AVG_FACTOR + REGISTER_SIZE }, { ptp_configure_cb, &ctxt_->ptp_config }));
    CHECK_CP_MAP_SET(ctxt_->cp_write_map.set({ FPGA_PTP_SYNC_TS_0, FPGA_PTP_OFM + REGISTER_SIZE }, { ptp_configure_cb, &ctxt_->ptp_config }));
    CHECK_CP_MAP_SET(ctxt_->cp_read_map.set({ FPGA_PTP_CTRL, FPGA_PTP_CTRL_DELAY_AVG_FACTOR + REGISTER_SIZE }, { ptp_readback_cb, &ctxt_->ptp_config }));
    CHECK_CP_MAP_SET(ctxt_->cp_read_map.set({ FPGA_PTP_SYNC_TS_0, FPGA_PTP_OFM + REGISTER_SIZE }, { ptp_readback_cb, &ctxt_->ptp_config }));

    // register callbacks for apb ram read/write
    CHECK_CP_MAP_SET(ctxt_->cp_read_map.set({ APB_RAM, APB_RAM + APB_RAM_DATA_SIZE }, { apb_ram_read, &ctxt_->apb_ram_data[0] }));
    CHECK_CP_MAP_SET(ctxt_->cp_write_map.set({ APB_RAM, APB_RAM + APB_RAM_DATA_SIZE }, { apb_ram_write, &ctxt_->apb_ram_data[0] }));

    // register callbacks for async events
    CHECK_CP_MAP_SET(ctxt_->cp_read_map.set({ CTRL_EVENT, CTRL_EVT_SW_EVENT }, { async_event_readback_cb, &ctxt_->async_event_ctxt }));
    CHECK_CP_MAP_SET(ctxt_->cp_read_map.set({ CTRL_EVT_STAT, CTRL_EVT_STAT + REGISTER_SIZE }, { async_event_readback_cb, &ctxt_->async_event_ctxt }));
    CHECK_CP_MAP_SET(ctxt_->cp_write_map.set({ CTRL_EVENT, CTRL_EVT_SW_EVENT }, { async_event_configure_cb, &ctxt_->async_event_ctxt }));
    CHECK_CP_MAP_SET(ctxt_->cp_write_map.set({ CTRL_EVT_SW_EVENT, CTRL_EVT_SW_EVENT + REGISTER_SIZE }, { handle_apb_sw_event, ctxt_.get() }));

    // register callbacks for gpio
    CHECK_CP_MAP_SET(ctxt_->cp_write_map.set({ GPIO_OUTPUT_BASE_REGISTER, GPIO_OUTPUT_BASE_REGISTER + GPIO_REGISTER_RANGE }, { GPIO_set_value, nullptr }));
    CHECK_CP_MAP_SET(ctxt_->cp_write_map.set({ GPIO_DIRECTION_BASE_REGISTER, GPIO_DIRECTION_BASE_REGISTER + GPIO_REGISTER_RANGE }, { GPIO_set_direction, nullptr }));
    CHECK_CP_MAP_SET(ctxt_->cp_read_map.set({ GPIO_STATUS_BASE_REGISTER, GPIO_STATUS_BASE_REGISTER + GPIO_REGISTER_RANGE }, { GPIO_get_value, nullptr }));
    CHECK_CP_MAP_SET(ctxt_->cp_read_map.set({ GPIO_DIRECTION_BASE_REGISTER, GPIO_DIRECTION_BASE_REGISTER + GPIO_REGISTER_RANGE }, { GPIO_get_direction, nullptr }));

    // register callbacks for i2c
    auto i2c_ctxt = i2c_controller_.ctxt_.get();
    CHECK_CP_MAP_SET(ctxt_->cp_read_map.set({ I2C_CTRL + I2C_REG_DATA_BUFFER, I2C_CTRL + I2C_REG_DATA_BUFFER + I2C_DATA_BUFFER_SIZE }, { i2c_readback_cb, i2c_ctxt }));
    CHECK_CP_MAP_SET(ctxt_->cp_read_map.set({ I2C_CTRL, I2C_CTRL + I2C_REG_CLK_CNT + REGISTER_SIZE }, { i2c_readback_cb, i2c_ctxt }));
    CHECK_CP_MAP_SET(ctxt_->cp_read_map.set({ I2C_CTRL + I2C_REG_STATUS, I2C_CTRL + I2C_REG_STATUS + REGISTER_SIZE }, { i2c_readback_cb, i2c_ctxt }));
    CHECK_CP_MAP_SET(ctxt_->cp_write_map.set({ I2C_CTRL + I2C_REG_DATA_BUFFER, I2C_CTRL + I2C_REG_DATA_BUFFER + I2C_DATA_BUFFER_SIZE }, { i2c_configure_cb, i2c_ctxt }));
    CHECK_CP_MAP_SET(ctxt_->cp_write_map.set({ I2C_CTRL, I2C_CTRL + I2C_REG_CLK_CNT + REGISTER_SIZE }, { i2c_configure_cb, i2c_ctxt }));

    // register callbacks for spi
    auto spi_ctxt = ctxt_->spi_ctxt;
    CHECK_CP_MAP_SET(ctxt_->cp_read_map.set({ SPI_CTRL + SPI_REG_DATA_BUFFER, SPI_CTRL + SPI_REG_DATA_BUFFER + SPI_DATA_BUFFER_SIZE }, { spi_readback_cb, spi_ctxt }));
    CHECK_CP_MAP_SET(ctxt_->cp_read_map.set({ SPI_CTRL, SPI_CTRL + SPI_REG_NUM_CMD_BYTES + REGISTER_SIZE }, { spi_readback_cb, spi_ctxt }));
    CHECK_CP_MAP_SET(ctxt_->cp_read_map.set({ SPI_CTRL + SPI_REG_STATUS, SPI_CTRL + SPI_REG_STATUS + REGISTER_SIZE }, { spi_readback_cb, spi_ctxt }));
    CHECK_CP_MAP_SET(ctxt_->cp_write_map.set({ SPI_CTRL + SPI_REG_DATA_BUFFER, SPI_CTRL + SPI_REG_DATA_BUFFER + SPI_DATA_BUFFER_SIZE }, { spi_configure_cb, spi_ctxt }));
    CHECK_CP_MAP_SET(ctxt_->cp_write_map.set({ SPI_CTRL, SPI_CTRL + SPI_REG_NUM_CMD_BYTES + REGISTER_SIZE }, { spi_configure_cb, spi_ctxt }));
}

int HSBEmulator::register_read_callback(uint32_t start_address, uint32_t end_address, ControlPlaneCallback_f callback, void* ctxt)
{
    return ctxt_->cp_read_map.set({ start_address, end_address }, { callback, ctxt });
}

int HSBEmulator::register_write_callback(uint32_t start_address, uint32_t end_address, ControlPlaneCallback_f callback, void* ctxt)
{
    return ctxt_->cp_write_map.set({ start_address, end_address }, { callback, ctxt });
}

int reset_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count)
{
    (void)ctxt;
    int n = 0;
    while (n < max_count) {
        uint32_t address = AVP_GET_ADDRESS(addr_val);
        if (address != RESET_REG_CTRL && address != SOFT_RESET_REG_CTRL) {
            break;
        }
        addr_val++;
        n++;
    }
    return n;
}
struct ControlMessageHeader {
    uint8_t cmd_code;
    uint8_t flags;
    uint16_t sequence;
    uint8_t status;
    uint8_t reserved;
};

static_assert(sizeof(struct ControlMessageHeader) == 6, "ControlMessageHeader must be 6 bytes");

struct ControlMessage {
    struct ControlMessageHeader* ctrl_hdr;
    // pointers to pairs of addresses/values in 16-bit words
    // this is to ensure 4-byte access alignments
    // the address at index i is ntohl(((uint32_t)(addr_val_wd[2*i].address << 16)) | addr_val_wd[2*i+1].address)
    struct AddressValuePair* addr_vals;
    uint16_t num_addresses; // addr_vals is of length num_addresses
};

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

// size of header: cmd_code, flags, sequence, status, reserved
#define CONTROL_MESSAGE_HEADER_LENGTH (sizeof(struct ControlMessageHeader))
static inline uint16_t control_message_n_addresses(size_t payload_length)
{
    // if reading or writing one word, host will only send one word, but if reading and writing in blocks, it will send 2 words per address.
    // so we add one word and divide by 2-word pairs to get the number of addresses in message
    return ((payload_length - CONTROL_MESSAGE_HEADER_LENGTH) + sizeof(uint32_t)) / (sizeof(uint32_t) * 2);
}

void control_plane_reply(HSBEmulatorCtxt* ctxt, ETH_BufferTypeDef* buffer, struct ControlMessage& message)
{
    ETH_HandleTypeDef* eth_handle = &ctxt->eth_handle;
    struct ether_header* eth_hdr = NET_GET_ETHER_HDR(buffer);
    struct iphdr* ip_hdr = NET_GET_IP_HDR(buffer);
    struct udphdr* udp_hdr = NET_GET_UDP_HDR(buffer);

    message.ctrl_hdr->cmd_code = (uint8_t)0x80U | message.ctrl_hdr->cmd_code;

    // build the payload first so we can set lengths
    // add the latched sequence
    uint32_t control_payload_length = CONTROL_MESSAGE_HEADER_LENGTH + 2 * message.num_addresses * sizeof(uint32_t);
    uint8_t* message_end = ((uint8_t*)(message.ctrl_hdr)) + control_payload_length;
    memset(message_end, 0, sizeof(uint32_t));
    control_payload_length += sizeof(uint32_t);

    // set up the udp_header
    {
        udp_hdr->len = htons(control_payload_length + UDP_HDR_LEN);
        udp_hdr->check = 0;
        // swap the ports
        uint16_t port = udp_hdr->source;
        udp_hdr->source = udp_hdr->dest;
        udp_hdr->dest = port;
    }

    // set up ip_header
    {
        // swap the ip addresses
        ip_hdr->ttl = IPDEFTTL;
        ip_hdr->check = 0;
        // swap the source and destination ip addresses
        uint32_t ip = ip_hdr->saddr;
        ip_hdr->saddr = ip_hdr->daddr;
        ip_hdr->daddr = ip;
        ip_hdr->tot_len = htons(control_payload_length + UDP_HDR_LEN + IP_HDR_LEN);
    }

    // set up ether_header
    {
        memmove(eth_hdr->ether_dhost, eth_hdr->ether_shost, ETH_ALEN);
        memcpy(eth_hdr->ether_shost, eth_handle->Init.MACAddr, ETH_ALEN);
    }

    // build buffers
    ETH_BufferTypeDef tx_buffers = {
        .buffer = (uint8_t*)(message.ctrl_hdr) - ETHER_HDR_LEN - IP_HDR_LEN - UDP_HDR_LEN,
        .len = control_payload_length + ETHER_HDR_LEN + IP_HDR_LEN + UDP_HDR_LEN,
        .next = NULL,
    };

    // TODO: this could just be statically allocated
    ETH_TxPacketConfigTypeDef tx_config {
        .Attributes = ETH_TX_PACKETS_FEATURES_CSUM | ETH_TX_PACKETS_FEATURES_CRCPAD,
        .Length = tx_buffers.len,
        .TxBuffer = &tx_buffers,
        .CRCPadCtrl = ETH_CRC_PAD_INSERT,
        .ChecksumCtrl = ETH_CHECKSUM_IPHDR_PAYLOAD_INSERT_PHDR_CALC,
    };

    // blocking reply here to avoid loss of stack allocated buffer
    HAL_StatusTypeDef status = HAL_ETH_Transmit(eth_handle, &tx_config, HSB_DEFAULT_TIMEOUT_MSEC);
    if (status != HAL_OK) {
        return;
    }
    return;
}

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
            return (n == 0) ? 0 : -1;
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
            return (n == 0) ? 0 : -1;
        }
        cur++;
        n++;
    }
    return n;
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
        control_plane_reply(ctxt, buffer, message);
    }
}

int HSBEmulator::add_data_plane(DataPlane& data_plane)
{
    if (ctxt_->data_plane_count >= MAX_DATA_PLANES) {
        return 1;
    }
    ctxt_->data_plane_list[ctxt_->data_plane_count++] = &data_plane;
    return 0;
}

int HSBEmulator::handle_msgs()
{
    ETH_BufferTypeDef* buffer;
    int status = eth_receive(&buffer);
    if (0 > status) {
        Error_Handler();
    }

    // RISK: potential infinite loop if under extreme load. TODO: need strategy to tradeoff packet loss against infinite loop.
    while (buffer) {
        handle_control_packet(ctxt_.get(), buffer);
        eth_release(buffer);
        status = eth_receive(&buffer);
        if (0 > status) {
            Error_Handler();
        }
    }

    // trigger events-based messages
    ctxt_->up_time_msec = HAL_GetTick();
    if (ctxt_->up_time_msec >= ctxt_->next_bootp_time_msec) {
        for (unsigned short i = 0; i < ctxt_->data_plane_count; i++) {
            DataPlane* data_plane = ctxt_->data_plane_list[i];
            if (data_plane->is_running()) {
                data_plane->broadcast_bootp();
            }
        }
        // schedule next bootp
        ctxt_->next_bootp_time_msec = ctxt_->up_time_msec + BOOTP_INTERVAL_SEC * 1000;
    }
    return 0;
}

bool HSBEmulator::is_running()
{
    return ctxt_->running;
}

/* this will block until all registered DataPlanes have stopped */
void HSBEmulator::stop()
{
    // if not running, do nothing...idempotent operation
    if (!is_running()) {
        return;
    }

    for (unsigned short i = 0; i < ctxt_->data_plane_count; i++) {
        DataPlane* data_plane = ctxt_->data_plane_list[i];
        if (data_plane->is_running()) {
            data_plane->stop();
        }
    }
    ctxt_->running = false;
}

void HSBEmulator::start()
{
    if (is_running()) {
        return;
    }

    // GPIO should be first because it will initialize all the clocks
    if (GPIO_init(nullptr)) {
        return;
    }

    // initialize networking module. This is not in the DataPlane because it is global to the HSBEmulator instance since we only have one interface
    if (net_init(&ctxt_->eth_handle)) {
        return;
    }

    // initialize timer module
    if (tim_init(nullptr)) {
        return;
    }

    // i2c_controller_ handles initializing the i2c module
    i2c_controller_.start();
    if (!i2c_controller_.is_running()) {
        return;
    }

    // do not have spi_controller yet, so initialize spi module here
    if (spi_init(&ctxt_->spi_ctxt->hspi)) {
        return;
    }

    for (unsigned short i = 0; i < ctxt_->data_plane_count; i++) {
        DataPlane* data_plane = ctxt_->data_plane_list[i];
        if (data_plane->is_running()) {
            continue;
        }
        data_plane->start();
    }

    ctxt_->cp_write_map.build();
    ctxt_->cp_read_map.build();

    // start the control plane thread
    ctxt_->running = true;

    ctxt_->up_time_msec = 0;
    ctxt_->next_bootp_time_msec = 0;

    if (!is_running()) {
        Error_Handler();
    }
}

int RegisterMemory::write(AddressValuePair& address_value)
{
    auto callback = ctxt_->cp_write_map.get(address_value.address);
    int ret = 1;
    if (callback && (1 == callback->callback(callback->ctxt, &address_value, 1))) {
        ret = 0;
    }
    return ret;
}

int RegisterMemory::read(AddressValuePair& address_value)
{
    auto callback = ctxt_->cp_read_map.get(address_value.address);
    int ret = 1;
    if (callback && (1 == callback->callback(callback->ctxt, &address_value, 1))) {
        ret = 0;
    }
    return ret;
}

int RegisterMemory::write_many(AddressValuePair* address_values, int num_addresses)
{
    if (!address_values || num_addresses < 0) {
        return 1;
    }
    int ret = 0;
    int i = 0;
    while (i < num_addresses) {
        auto callback = ctxt_->cp_write_map.get(address_values[i].address);
        if (!callback) {
            ret = 1;
            break;
        }
        int ncomsumed = callback->callback(callback->ctxt, address_values + i, num_addresses - i);
        if (0 >= ncomsumed) {
            ret = 1;
            break;
        }
        i += ncomsumed;
    }
    return ret;
}

int RegisterMemory::read_many(AddressValuePair* address_values, int num_addresses)
{
    if (!address_values || num_addresses < 0) {
        return 1;
    }
    int ret = 0;
    int i = 0;
    while (i < num_addresses) {
        auto callback = ctxt_->cp_read_map.get(address_values[i].address);
        if (!callback) {
            ret = 1;
            break;
        }
        int ncomsumed = callback->callback(callback->ctxt, address_values + i, num_addresses - i);
        if (0 >= ncomsumed) {
            ret = 1;
            break;
        }
        i += ncomsumed;
    }
    return ret;
}

int RegisterMemory::write_range(uint32_t start_address, uint32_t* values, int num_addresses, int stride)
{
    if (!values || num_addresses < 0) {
        return 1;
    }
    int ret = 0;
    while (num_addresses) {
        struct AddressValuePair address_value = { start_address, *values };
        auto callback = ctxt_->cp_write_map.get(start_address);
        if (!callback) {
            ret = 1;
            break;
        }
        int consumed = callback->callback(callback->ctxt, &address_value, 1);
        if (0 >= consumed) {
            ret = 1;
            break;
        }
        num_addresses -= consumed;
        values += consumed * stride;
        start_address += consumed * REGISTER_SIZE;
    }
    return ret;
}

int RegisterMemory::read_range(uint32_t start_address, uint32_t* values, int num_addresses, int stride)
{
    if (!values || num_addresses < 0) {
        return 1;
    }
    int ret = 0;
    while (num_addresses) {
        struct AddressValuePair address_value = { start_address, 0 };
        auto callback = ctxt_->cp_read_map.get(start_address);
        if (!callback) {
            ret = 1;
            break;
        }
        int consumed = callback->callback(callback->ctxt, &address_value, 1);
        if (0 >= consumed) {
            ret = 1;
            break;
        }
        *values = address_value.value;
        num_addresses -= consumed;
        values += consumed * stride;
        start_address += consumed * REGISTER_SIZE;
    }
    return ret;
}

} // namespace hololink::emulation
