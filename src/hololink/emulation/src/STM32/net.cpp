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

#include "STM32/net.hpp"
#include "STM32/stm32_system.h"
#include "inttypes.h"
#include "string.h"
#include <cstdlib>

#ifndef N_RX_BUFFER_DESC
#define N_RX_BUFFER_DESC ETH_RXBUFNB
#endif

// module level variables

in_addr_t IP_ADDRESS = INADDR_ANY;

uint8_t MAC_ADDRESS[6] = { 0x00, 0x80, 0xE1, 0x01, 0x02, 0x03 };

struct rx_buffer_list_t {
    ETH_BufferTypeDef buffer;
    // We want buffer_data to be aligned to 2 bytes, but not 4 bytes and still
    // be able to use offsetof from buffer_data to get the descriptor.
    // So we add 2 bytes on a 4-byte boundary to satisfy DMA and the buffer_data immediately follows.
    // This also prevents any possible DMA transfers starting on 4-byte boundaries from overwriting data
    alignas(4) uint8_t reserved[2];
    uint8_t buffer_data[RX_BUFFER_SIZE];
    bool in_use;
} rx_buffer_list[N_RX_BUFFER_DESC];

// static assert our intent described in the descriptor comment above
static_assert(offsetof(struct rx_buffer_list_t, buffer_data) - offsetof(struct rx_buffer_list_t, reserved) == 2);

ETH_DMADescTypeDef DMARxDscrTab[ETH_RX_DESC_CNT] __attribute__((section(".RxDecripSection"))); /* Ethernet Rx DMA Descriptors */
ETH_DMADescTypeDef DMATxDscrTab[ETH_TX_DESC_CNT] __attribute__((section(".TxDecripSection"))); /* Ethernet Tx DMA Descriptors */

ETH_HandleTypeDef* ETH_HANDLE = nullptr;

// Functions

void set_mac_address(const uint8_t* mac_address)
{
    memcpy(MAC_ADDRESS, mac_address, 6);
}

const uint8_t* get_mac_address(void)
{
    return &MAC_ADDRESS[0];
}

void generate_mac_address(void)
{
    char* uid = get_uid();

    // hash the uid
    uint32_t hash = 0;
    for (int i = 0; i < 12; i++) {
        hash = (hash << 5) + hash + uid[i];
    }
    hash = hash & 0xFFFFFFu; // only keep the lower 24 bits
    MAC_ADDRESS[3] = (hash >> 16) & 0xFF;
    MAC_ADDRESS[4] = (hash >> 8) & 0xFF;
    MAC_ADDRESS[5] = hash & 0xFF;
}

// overrides the HAL_ETH_MspInit function to initialize the ETH and GPIO in STM32 HAL Driver
extern "C" void HAL_ETH_MspInit(ETH_HandleTypeDef* ethHandle)
{

    GPIO_InitTypeDef gpio_cfg = { 0 };
    if (ethHandle->Instance == ETH) {
        // enable ETH clock
        __HAL_RCC_ETH_CLK_ENABLE();

        // RMII MDC, RX Data 0, RX Data 1 pins
        gpio_cfg.Pin = GPIO_PIN_1 | GPIO_PIN_4 | GPIO_PIN_5;
        gpio_cfg.Mode = GPIO_MODE_AF_PP;
        gpio_cfg.Pull = GPIO_NOPULL;
        gpio_cfg.Speed = GPIO_SPEED_FREQ_HIGH;
        gpio_cfg.Alternate = GPIO_AF11_ETH;
        HAL_GPIO_Init(GPIOC, &gpio_cfg);

        // RMII REF CLK, MDIO, CRS DV pins
        gpio_cfg.Pin = GPIO_PIN_1 | GPIO_PIN_2 | GPIO_PIN_7;
        gpio_cfg.Mode = GPIO_MODE_AF_PP;
        gpio_cfg.Pull = GPIO_NOPULL;
        gpio_cfg.Speed = GPIO_SPEED_FREQ_HIGH;
        gpio_cfg.Alternate = GPIO_AF11_ETH;
        HAL_GPIO_Init(GPIOA, &gpio_cfg);

        // RMII TX Data 1 pin
        gpio_cfg.Pin = GPIO_PIN_13;
        gpio_cfg.Mode = GPIO_MODE_AF_PP;
        gpio_cfg.Pull = GPIO_NOPULL;
        gpio_cfg.Speed = GPIO_SPEED_FREQ_HIGH;
        gpio_cfg.Alternate = GPIO_AF11_ETH;
        HAL_GPIO_Init(GPIOB, &gpio_cfg);

        // RMII TX Enable, TX Data 0 pins
        gpio_cfg.Pin = GPIO_PIN_11 | GPIO_PIN_13;
        gpio_cfg.Mode = GPIO_MODE_AF_PP;
        gpio_cfg.Pull = GPIO_NOPULL;
        gpio_cfg.Speed = GPIO_SPEED_FREQ_HIGH;
        gpio_cfg.Alternate = GPIO_AF11_ETH;
        HAL_GPIO_Init(GPIOG, &gpio_cfg);
    }
}

// overrides the HAL_ETH_MspDeInit function to de-initialize the ETH and GPIO in STM32 HAL Driver
extern "C" void HAL_ETH_MspDeInit(ETH_HandleTypeDef* ethHandle)
{

    if (ethHandle->Instance == ETH) {
        ETH_HANDLE = nullptr;

        // disable ETH clock
        __HAL_RCC_ETH_CLK_DISABLE();

        // de-init RMII
        // RMII MDC, RX Data 0, RX Data 1 pins
        HAL_GPIO_DeInit(GPIOC, GPIO_PIN_1 | GPIO_PIN_4 | GPIO_PIN_5);
        // RMII REF CLK, MDIO, CRS DV pins
        HAL_GPIO_DeInit(GPIOA, GPIO_PIN_1 | GPIO_PIN_2 | GPIO_PIN_7);
        // RMII TX Data 1 pin
        HAL_GPIO_DeInit(GPIOB, GPIO_PIN_13);
        // RMII TX Enable
        HAL_GPIO_DeInit(GPIOG, GPIO_PIN_11 | GPIO_PIN_13);
    }
}

char INET_NTOP_ERROR[] = "-1.-1.-1.-1";

char* inet_ntoa(struct in_addr addr)
{
    static char buf[16];
    int len = snprintf(buf, sizeof(buf), PRIu32 "." PRIu32 "." PRIu32 "." PRIu32,
        (addr.s_addr >> 0) & 0xff,
        (addr.s_addr >> 8) & 0xff,
        (addr.s_addr >> 16) & 0xff,
        (addr.s_addr >> 24) & 0xff);
    if (len < 0 || (unsigned)len >= sizeof(buf)) {
        return &INET_NTOP_ERROR[0]; // return of inet_ntoa on error is unspecified
    }
    buf[len] = '\0';
    return &buf[0];
}

extern "C" void rx_allocate_buffer(uint8_t** buffer)
{
    for (unsigned int i = 0; i < N_RX_BUFFER_DESC; i++) {
        if (!rx_buffer_list[i].in_use) {
            rx_buffer_list[i].in_use = true;
            *buffer = rx_buffer_list[i].buffer.buffer;
            return;
        }
    }
    *buffer = NULL;
}

extern "C" void rx_link_buffer_desc(void** pStart, void** pEnd, uint8_t* buff, uint16_t length)
{
    ETH_BufferTypeDef* buffer_desc = (ETH_BufferTypeDef*)(buff - offsetof(struct rx_buffer_list_t, buffer_data));
    buffer_desc->len = length;
    buffer_desc->next = NULL;
    if (*pStart == NULL) {
        *pStart = buffer_desc;
        *pEnd = *pStart;
    } else {
        ((ETH_BufferTypeDef*)(*pEnd))->next = buffer_desc;
        *pEnd = buffer_desc;
    }
}

volatile uint16_t n_rx_packets = 0;

extern "C" void rx_cplt_callback(__attribute__((unused)) ETH_HandleTypeDef* heth)
{
    n_rx_packets++;
}

/* Tx complete callback so the HAL leaves BUSY state and can accept the next transmit.
 * Without this, after a transmit the driver may stay in BUSY and HAL_ETH_Transmit returns HAL_BUSY. */
extern "C" void tx_cplt_callback(ETH_HandleTypeDef* heth)
{
    HAL_ETH_ReleaseTxPacket(heth);
}

void init_rx_buffer_pool(void)
{
    for (unsigned int i = 0; i < ETH_RXBUFNB; i++) {
        rx_buffer_list[i].buffer.buffer = &rx_buffer_list[i].buffer_data[0];
        rx_buffer_list[i].buffer.next = NULL;
        rx_buffer_list[i].buffer.len = RX_BUFFER_SIZE;
        rx_buffer_list[i].in_use = false;
    }
}

extern "C" void ETH_IRQHandler(void)
{
    HAL_ETH_IRQHandler(ETH_HANDLE);
}

void send_arp_reply(struct ether_header* eth_hdr, uint8_t* buffer_start)
{

    struct arphdr* arp_hdr = (struct arphdr*)buffer_start;
    buffer_start += ARP_HDR_LEN;
    if (ETHERTYPE_IP != ntohs(arp_hdr->ar_pro)) {
        // only respond to IPv4 ARP requests
        return;
    }
    {
        // move the source mac to the destination mac
        memmove(eth_hdr->ether_dhost, eth_hdr->ether_shost, ETH_ALEN);
        // write the source mac to the source mac
        memcpy(eth_hdr->ether_shost, MAC_ADDRESS, ETH_ALEN);
    }
    arp_hdr->ar_op = htons(ARPOP_REPLY);

    // just reuse the buffer provided with the request packet
    uint8_t* buffer = buffer_start - ETHER_HDR_LEN - ARP_HDR_LEN;

    // move the sender data to the target data, do not advance the buffer pointer
    memmove(buffer_start + arp_hdr->ar_hln + arp_hdr->ar_pln, buffer_start, arp_hdr->ar_hln + arp_hdr->ar_pln);

    // These next copies are specific to MAC & IPv4 protocol ARP replies
    memcpy(buffer_start, MAC_ADDRESS, ETH_ALEN);
    buffer_start += ETH_ALEN;
    memcpy(buffer_start, &IP_ADDRESS, IP_ADDR_LEN);
    buffer_start += IP_ADDR_LEN;

    // build buffers
    ETH_BufferTypeDef tx_buffers = {
        .buffer = buffer,
        .len = 2 * (arp_hdr->ar_hln + arp_hdr->ar_pln) + ETHER_HDR_LEN + ARP_HDR_LEN,
        .next = NULL,
    };

    ETH_TxPacketConfigTypeDef tx_config {
        .Attributes = ETH_TX_PACKETS_FEATURES_CSUM | ETH_TX_PACKETS_FEATURES_CRCPAD,
        .Length = tx_buffers.len,
        .TxBuffer = &tx_buffers,
        .CRCPadCtrl = ETH_CRC_PAD_INSERT,
        .ChecksumCtrl = ETH_CHECKSUM_IPHDR_PAYLOAD_INSERT_PHDR_CALC,
    };

    HAL_StatusTypeDef status = HAL_ETH_Transmit(ETH_HANDLE, &tx_config, HSB_DEFAULT_TIMEOUT_MSEC);
    if (status != HAL_OK) {
        return;
    }
    return;
}

// ICMP request must already be filtered for targeting a valid IP address of this device
void send_icmp_echo_reply(struct ether_header* eth_hdr, struct iphdr* ip_hdr, uint8_t* buffer_start)
{
    struct icmphdr* icmp_hdr = (struct icmphdr*)buffer_start;
    buffer_start += ICMP_HDR_LEN;
    if (ICMP_ECHO != icmp_hdr->type) {
        // only respond to echo requests
        return;
    }

    {
        // move the source mac to the destination mac
        memmove(eth_hdr->ether_dhost, eth_hdr->ether_shost, ETH_ALEN);
        // write the destination mac to the source mac
        memcpy(eth_hdr->ether_shost, MAC_ADDRESS, ETH_ALEN);
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
    }

    // set up the icmp_header
    {
        icmp_hdr->type = ICMP_ECHOREPLY;
        icmp_hdr->code = 0;
        icmp_hdr->checksum = 0;
    }

    // reuse the buffer provided with the request packet
    uint8_t* buffer = buffer_start - ICMP_HDR_LEN - IP_HDR_LEN - ETHER_HDR_LEN;

    // build buffers
    ETH_BufferTypeDef tx_buffers = {
        .buffer = buffer,
        .len = ntohs(ip_hdr->tot_len) + ETHER_HDR_LEN,
        .next = NULL,
    };

    ETH_TxPacketConfigTypeDef tx_config {
        .Attributes = ETH_TX_PACKETS_FEATURES_CSUM | ETH_TX_PACKETS_FEATURES_CRCPAD,
        .Length = tx_buffers.len,
        .TxBuffer = &tx_buffers,
        .CRCPadCtrl = ETH_CRC_PAD_INSERT,
        .ChecksumCtrl = ETH_CHECKSUM_IPHDR_PAYLOAD_INSERT_PHDR_CALC,
    };

    HAL_StatusTypeDef status = HAL_ETH_Transmit(ETH_HANDLE, &tx_config, HSB_DEFAULT_TIMEOUT_MSEC);

    if (status != HAL_OK) {
        return;
    }
    return;
}

// return 0 is success, HAL status on error (> 0). Valid buffer only if buffer returned is not NULL.
int get_next_packet(ETH_BufferTypeDef** buffer)
{
    *buffer = NULL;
    bool ready = false;
    HAL_NVIC_DisableIRQ(ETH_IRQn);
    if (n_rx_packets) {
        n_rx_packets--;
        ready = true;
    }
    HAL_NVIC_EnableIRQ(ETH_IRQn);
    if (ready) {
        HAL_StatusTypeDef status = HAL_ETH_ReadData(ETH_HANDLE, (void**)buffer);
        if (status != HAL_OK) {
            return (int)status;
        }
        return 0;
    }
    return 0;
}

void net_set_ip_address(in_addr_t ip_address)
{
    IP_ADDRESS = ip_address;
}

void eth_release(ETH_BufferTypeDef* buffer)
{
    if (NULL == buffer) {
        return;
    }
    while (buffer) {
        struct rx_buffer_list_t* buffer_desc = (struct rx_buffer_list_t*)buffer;
        buffer = buffer->next;
        buffer_desc->in_use = false;
    }
}

// returns 0 on success, < 0 on error. returns > 0 are reserved for future use.Valid buffer only if buffer returned is not NULL.
int eth_receive(ETH_BufferTypeDef** buffer_rtn)
{
    *buffer_rtn = NULL;
    ETH_BufferTypeDef* buffer;
    int status = get_next_packet(&buffer);
    if (status) {
        return -status;
    }
    if (!buffer) {
        return 0;
    }

    struct ether_header* eth_hdr = NET_GET_ETHER_HDR(buffer);

    // filter ip
    if (eth_hdr->ether_type == htons(ETHERTYPE_IP)) {
        struct iphdr* ip_hdr = NET_GET_IP_HDR(buffer);
        if (IP_ADDRESS != ip_hdr->daddr && INADDR_ANY != ip_hdr->daddr) {
            // drop packet
            goto cleanup;
        }

        // filter udp
        if (ip_hdr->protocol == IPPROTO_UDP) {
            struct udphdr* udp_hdr = NET_GET_UDP_HDR(buffer);

            // return control message for handling by the calling emulator
            if (CONTROL_UDP_PORT == ntohs(udp_hdr->dest)) {
                *buffer_rtn = buffer;
                return 0;
            }

            // filter ICMPv4
        } else if (IPPROTO_ICMP == ip_hdr->protocol && INADDR_ANY != ip_hdr->daddr) {
            send_icmp_echo_reply(eth_hdr, ip_hdr, NET_GET_IP_PAYLOAD(ip_hdr));
        }

        // filter ARPv4
    } else if (eth_hdr->ether_type == htons(ETHERTYPE_ARP)) {
        send_arp_reply(eth_hdr, NET_GET_ETHER_PAYLOAD(eth_hdr));
    }

cleanup:
    eth_release(buffer);
    return 0;
}

/* ETH init function */
int net_init(ETH_HandleTypeDef* heth)
{
    // already initialized
    if (ETH_HANDLE && ETH_HANDLE == heth) {
        return 0;
    }
    if (!heth) {
        return -1;
    }

    heth->Instance = ETH;
    // TODO: replace this with a calculation from UID
    heth->Init.MACAddr = &MAC_ADDRESS[0];
    heth->Init.MediaInterface = HAL_ETH_RMII_MODE;
    heth->Init.TxDesc = DMATxDscrTab;
    heth->Init.RxDesc = DMARxDscrTab;
    heth->Init.RxBuffLen = RX_BUFFER_SIZE;

    if (HAL_ETH_Init(heth) != HAL_OK) {
        Error_Handler();
    }

    int return_status = 0;

    init_rx_buffer_pool();
    HAL_StatusTypeDef status = HAL_ETH_RegisterRxAllocateCallback(heth, rx_allocate_buffer);
    if (status != HAL_OK) {
        return_status = -2;
        goto post_init_fail;
    }
    status = HAL_ETH_RegisterRxLinkCallback(heth, rx_link_buffer_desc);
    if (status != HAL_OK) {
        return_status = -3;
        goto post_init_fail;
    }
    status = HAL_ETH_RegisterCallback(heth, HAL_ETH_RX_COMPLETE_CB_ID, rx_cplt_callback);
    if (status != HAL_OK) {
        return_status = -4;
        goto post_init_fail;
    }
    status = HAL_ETH_RegisterCallback(heth, HAL_ETH_TX_COMPLETE_CB_ID, tx_cplt_callback);
    if (status != HAL_OK) {
        return_status = -5;
        goto post_init_fail;
    }
    // this should not be necessary as HAL_ETH_Init calls it
    HAL_ETH_MspInit(heth);
    // should happen after MspInit so that GPIOs are configured
    HAL_NVIC_SetPriority(ETH_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(ETH_IRQn);
    HAL_ETH_Start_IT(heth);

    // mark as initialized
    ETH_HANDLE = heth;
    return 0;

post_init_fail:
    HAL_ETH_DeInit(heth);
    return return_status;
}

int ptp_configure(struct PTPConfig* ptp_config)
{
    (void)ptp_config;
    return 0;
}

// basic implementation of posix inet_aton to parse an IP address string into a in_addr_t
int inet_aton(const char* cp, in_addr* inp)
{
    static const unsigned long long MAX_INTERPRETED_VALUE = 0xFFFFFFFFu;
    in_addr_t ip_address = 0;
    uint8_t nbits = 0;
    char* endptr = NULL;

    while (nbits < 32u && isdigit(*cp)) {
        // convert next string value to unsigned long long. Use it to guarantee that
        unsigned long long value = strtoull(cp, &endptr, 10);
        if (endptr == cp || (!value && (*cp != '0')) || value > (MAX_INTERPRETED_VALUE >> nbits)) { // no data interpreted or invalid data
            break;
        }

        if (value <= 0xFF) { // one octet is provided
            ip_address |= (value << nbits);
            nbits += 8u;
        } else if (value <= 0xFFFFu) { // two octets are provided
            ip_address |= ((in_addr_t)htons((uint16_t)value)) << nbits;
            nbits += 16u;
        } else if (value <= 0xFFFFFFu) { // three octets are provided
            ip_address |= ((in_addr_t)htonl((in_addr_t)value)) << nbits;
            nbits += 24u;
        } else { // four octets or more are provided, but more than 4 is blocked by check above
            ip_address = htonl((in_addr_t)value);
            nbits += 32u;
        }
        cp = endptr;
        if (*cp == '.') {
            cp++;
        }
    }
    if (*cp && *cp != '/') { // invalid format if there is data left that is not potentially in CIDR notation
        inp->s_addr = INADDR_NONE;
        return 0;
    }
    // specifying < 32 bits of IP address is allowed, shift right the remaining bits
    if (nbits < 32u) {
        ip_address >>= (32u - nbits);
    }
    inp->s_addr = ip_address;
    return 1; // success
}

// public interface implementations

namespace hololink::emulation {

uint32_t get_broadcast_address(const IPAddress& ip_address)
{
    return (ip_address.ip_address & ip_address.subnet_mask) | ~ip_address.subnet_mask;
}

IPAddress IPAddress_from_string(const std::string& ip_address)
{
    in_addr inp = {
        .s_addr = INADDR_NONE,
    };
    int status = inet_aton(ip_address.c_str(), &inp);
    IPAddress addr; // default-init: if_name is empty string, scalars zeroed
    addr.ip_address = inp.s_addr;
    if (status) {
        addr.flags = IPADDRESS_HAS_ADDR | IPADDRESS_HAS_BROADCAST | IPADDRESS_HAS_NETMASK;
        addr.subnet_mask = htonl(0xFFFFFF00);
        addr.broadcast_address = get_broadcast_address(addr);
    }
    memcpy(addr.mac, MAC_ADDRESS, sizeof(MAC_ADDRESS));
    addr.flags |= IPADDRESS_HAS_MAC;
    return addr;
}

std::string IPAddress_to_string(const IPAddress& ip_address)
{
    if (!(ip_address.flags & IPADDRESS_HAS_ADDR)) {
        return std::string();
    }
    struct in_addr addr = {
        .s_addr = ip_address.ip_address,
    };
    return std::string(inet_ntoa(addr));
}

}