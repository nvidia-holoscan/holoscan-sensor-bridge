/*
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
 */

#include "hsb_lite_flash_2407.hpp"
#include "../firmware_utils.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <iostream>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace hololink {

// Control protocol commands
constexpr uint8_t WR_DWORD = 0x04;
constexpr uint8_t REQUEST_FLAGS_ACK_REQUEST = 0x01;

// SPI controller addresses
constexpr uint32_t CLNX_SPI_CTRL = 0x03000000;
constexpr uint32_t CPNX_SPI_CTRL = 0x03000200;

// CLNX bridge commands
constexpr uint8_t CLNX_WRITE = 0x1;
constexpr uint8_t FLASH_FORWARD_EN = 6;

// Flash chip commands
constexpr uint8_t WRITE_ENABLE = 0x06;
constexpr uint8_t BLOCK_ERASE = 0xD8;
constexpr uint8_t PAGE_PROGRAM = 0x02;

// Flash parameters
constexpr size_t BLOCK_SIZE = 128;
constexpr size_t ERASE_SIZE = 64 * 1024;

namespace {

    /**
     * Raw UDP control for blind flashing.
     */
    class BlindControl {
    public:
        explicit BlindControl(const std::string& ip)
            : seq_(0x100)
        {
            sock_ = socket(AF_INET, SOCK_DGRAM, 0);
            if (sock_ < 0) {
                throw std::runtime_error("Failed to create socket");
            }

            memset(&addr_, 0, sizeof(addr_));
            addr_.sin_family = AF_INET;
            addr_.sin_port = htons(8192);
            if (inet_pton(AF_INET, ip.c_str(), &addr_.sin_addr) <= 0) {
                throw std::runtime_error("Invalid IP address: " + ip);
            }

            struct timeval tv = { 0, 500000 }; // 500ms timeout
            if (setsockopt(sock_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0) {
                int err = errno;
                close(sock_);
                sock_ = -1;
                throw std::runtime_error(
                    std::string("Failed to set SO_RCVTIMEO on socket: ") + strerror(err));
            }
        }

        ~BlindControl()
        {
            if (sock_ >= 0) {
                close(sock_);
            }
        }

        BlindControl(const BlindControl&) = delete;
        BlindControl& operator=(const BlindControl&) = delete;
        BlindControl(BlindControl&&) = delete;
        BlindControl& operator=(BlindControl&&) = delete;

        void write_uint32(uint32_t address, uint32_t value)
        {
            uint8_t req[14] = {
                WR_DWORD, REQUEST_FLAGS_ACK_REQUEST,
                uint8_t(seq_ >> 8), uint8_t(seq_), 0, 0,
                uint8_t(address >> 24), uint8_t(address >> 16), uint8_t(address >> 8), uint8_t(address),
                uint8_t(value >> 24), uint8_t(value >> 16), uint8_t(value >> 8), uint8_t(value)
            };

            ssize_t ret = sendto(sock_, req, 14, 0, (struct sockaddr*)&addr_, sizeof(addr_));
            if (ret < 0) {
                throw std::runtime_error(
                    std::string("write_uint32: sendto failed: ") + strerror(errno));
            }
            if (ret != 14) {
                throw std::runtime_error(
                    "write_uint32: sendto sent " + std::to_string(ret) + " bytes, expected 14");
            }

            uint8_t resp[256];
            ssize_t r = recvfrom(sock_, resp, 256, 0, nullptr, nullptr);
            if (r < 0) {
                throw std::runtime_error(
                    std::string("write_uint32: recvfrom failed: ") + strerror(errno));
            }

            seq_++;
        }

    private:
        int sock_;
        struct sockaddr_in addr_;
        uint16_t seq_;
    };

    /**
     * SPI operations via raw UDP.
     */
    class BlindSpi {
    public:
        BlindSpi(BlindControl& ctrl, uint32_t base, uint32_t cs, uint32_t div,
            uint32_t cpol, uint32_t cpha, uint32_t width)
            : ctrl_(ctrl)
            , base_(base)
        {

            // Width encoding: 1->0, 2->(2<<8), 4->(3<<8)
            uint32_t width_bits = 0;
            if (width == 2) {
                width_bits = (2 << 8);
            } else if (width == 4) {
                width_bits = (3 << 8);
            }

            config_ = div | (cs << 12) | width_bits;
            if (cpol)
                config_ |= (1 << 4);
            if (cpha)
                config_ |= (1 << 5);
        }

        void transaction(const std::vector<uint8_t>& cmd, const std::vector<uint8_t>& write_data)
        {
            std::vector<uint8_t> all_data = cmd;
            all_data.insert(all_data.end(), write_data.begin(), write_data.end());

            // Write SPI config
            ctrl_.write_uint32(base_ + 8, config_);

            // Write data buffer (4 bytes at a time)
            for (size_t i = 0; i < all_data.size(); i += 4) {
                uint32_t value = all_data[i];
                if (i + 1 < all_data.size())
                    value |= (all_data[i + 1] << 8);
                if (i + 2 < all_data.size())
                    value |= (all_data[i + 2] << 16);
                if (i + 3 < all_data.size())
                    value |= (all_data[i + 3] << 24);
                ctrl_.write_uint32(base_ + 16 + i, value);
            }

            // Set byte counts: write_count | (read_count << 16)
            ctrl_.write_uint32(base_ + 4, all_data.size());

            // Set command length
            ctrl_.write_uint32(base_ + 12, cmd.size() << 8);

            // Start transaction
            ctrl_.write_uint32(base_ + 0, 1);

            // Fixed delay instead of status polling (blind mode)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

    private:
        BlindControl& ctrl_;
        uint32_t base_;
        uint32_t config_;
    };

    /**
     * Flash a chip using blind mode (no verification).
     */
    void blind_flash_chip(const std::string& name, BlindSpi* slow_spi, BlindSpi& fast_spi,
        const std::vector<uint8_t>& firmware)
    {
        std::cout << "[hsb_lite_flash_2407] Programming " << name << " (" << firmware.size() << " bytes)" << std::endl;

        for (size_t erase_addr = 0; erase_addr < firmware.size(); erase_addr += ERASE_SIZE) {
            // Enable CLNX bridge if needed
            if (slow_spi) {
                slow_spi->transaction({}, { CLNX_WRITE, FLASH_FORWARD_EN, 0x11 });
            }

            // Write enable
            fast_spi.transaction({ WRITE_ENABLE }, {});
            std::this_thread::sleep_for(std::chrono::milliseconds(50));

            // Block erase
            if (slow_spi) {
                slow_spi->transaction({}, { CLNX_WRITE, FLASH_FORWARD_EN, 0x11 });
            }
            fast_spi.transaction(
                { BLOCK_ERASE,
                    uint8_t(erase_addr >> 16),
                    uint8_t(erase_addr >> 8),
                    uint8_t(erase_addr) },
                {});

            // Wait for erase to complete (blind timing)
            std::this_thread::sleep_for(std::chrono::seconds(2));

            // Program pages in this block
            for (size_t addr = erase_addr;
                 addr < std::min(firmware.size(), erase_addr + ERASE_SIZE);
                 addr += BLOCK_SIZE) {

                if ((addr & 0xFFFF) == 0) {
                    std::cout << "  Address: 0x" << std::hex << addr << std::dec << std::endl;
                }

                // Write enable
                if (slow_spi) {
                    slow_spi->transaction({}, { CLNX_WRITE, FLASH_FORWARD_EN, 0x11 });
                }
                fast_spi.transaction({ WRITE_ENABLE }, {});

                // Prepare page data
                size_t end = std::min(addr + BLOCK_SIZE, firmware.size());
                std::vector<uint8_t> page_data(firmware.begin() + addr, firmware.begin() + end);

                // Program page
                if (slow_spi) {
                    slow_spi->transaction({}, { CLNX_WRITE, FLASH_FORWARD_EN, 0x11 });
                }
                fast_spi.transaction(
                    { PAGE_PROGRAM,
                        uint8_t(addr >> 16),
                        uint8_t(addr >> 8),
                        uint8_t(addr) },
                    page_data);

                // Wait for write to complete (blind timing)
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }

        std::cout << "[hsb_lite_flash_2407] " << name << " programming complete (blind mode - NOT verified)" << std::endl;
    }

} // anonymous namespace

bool hsb_lite_flash_2407(const std::string& ip_address,
    const std::string& clnx_path,
    const std::string& cpnx_path)
{
    std::cout << "[hsb_lite_flash_2407] Blind UDP flash" << std::endl;
    std::cout << "  IP: " << ip_address << std::endl;
    std::cout << "  CLNX: " << clnx_path << std::endl;
    std::cout << "  CPNX: " << cpnx_path << std::endl;
    std::cout << "  WARNING: NO verification - writes firmware blindly!" << std::endl;

    try {
        std::cout << "[hsb_lite_flash_2407] Loading firmware files..." << std::endl;
        auto clnx_content = load_firmware_file(clnx_path);
        auto cpnx_content = load_firmware_file(cpnx_path);
        std::cout << "  CLNX size: " << clnx_content.size() << " bytes" << std::endl;
        std::cout << "  CPNX size: " << cpnx_content.size() << " bytes" << std::endl;

        std::cout << "[hsb_lite_flash_2407] Connecting via raw UDP..." << std::endl;
        BlindControl control(ip_address);

        BlindSpi clnx_slow(control, CLNX_SPI_CTRL, 0, 0xF, 0, 1, 1);
        BlindSpi clnx_fast(control, CLNX_SPI_CTRL, 0, 0x4, 1, 1, 4);
        BlindSpi cpnx_spi(control, CPNX_SPI_CTRL, 0, 0, 1, 1, 1);

        std::cout << "[hsb_lite_flash_2407] Programming CLNX..." << std::endl;
        blind_flash_chip("CLNX", &clnx_slow, clnx_fast, clnx_content);

        std::cout << "[hsb_lite_flash_2407] Programming CPNX..." << std::endl;
        blind_flash_chip("CPNX", nullptr, cpnx_spi, cpnx_content);

        std::cout << "[hsb_lite_flash_2407] Flash completed (blind mode - NOT verified)!" << std::endl;
        std::cout << "[hsb_lite_flash_2407] Power cycle the device to apply changes." << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[hsb_lite_flash_2407] Error: " << e.what() << std::endl;
        return false;
    }
}

} // namespace hololink
