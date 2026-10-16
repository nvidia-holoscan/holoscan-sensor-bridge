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
constexpr uint8_t RD_DWORD = 0x14;
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
constexpr uint8_t READ = 0x03;
constexpr uint8_t STATUS = 0x05;

// SPI transaction-engine busy bit
constexpr uint32_t SPI_BUSY = 0b0000'0001'0000'0000;

// Flash parameters
constexpr size_t BLOCK_SIZE = 128;
constexpr size_t ERASE_SIZE = 64 * 1024;

namespace {

    /**
     * UDP register access
     */
    class UdpControl {
    public:
        explicit UdpControl(const std::string& ip)
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

        ~UdpControl()
        {
            if (sock_ >= 0) {
                close(sock_);
            }
        }

        UdpControl(const UdpControl&) = delete;
        UdpControl& operator=(const UdpControl&) = delete;
        UdpControl(UdpControl&&) = delete;
        UdpControl& operator=(UdpControl&&) = delete;

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
            recv_validated(resp, sizeof(resp), "write_uint32");

            seq_++;
        }

        uint32_t read_uint32(uint32_t address)
        {
            uint8_t req[10] = {
                RD_DWORD, REQUEST_FLAGS_ACK_REQUEST,
                uint8_t(seq_ >> 8), uint8_t(seq_), 0, 0,
                uint8_t(address >> 24), uint8_t(address >> 16), uint8_t(address >> 8), uint8_t(address)
            };

            ssize_t ret = sendto(sock_, req, 10, 0, (struct sockaddr*)&addr_, sizeof(addr_));
            if (ret < 0) {
                throw std::runtime_error(
                    std::string("read_uint32: sendto failed: ") + strerror(errno));
            }
            if (ret != 10) {
                throw std::runtime_error(
                    "read_uint32: sendto sent " + std::to_string(ret) + " bytes, expected 10");
            }

            uint8_t resp[256];
            ssize_t r = recv_validated(resp, sizeof(resp), "read_uint32");
            if (r < 14) {
                throw std::runtime_error(
                    "read_uint32: response too short (" + std::to_string(r) + " bytes)");
            }

            seq_++;
            return (uint32_t(resp[10]) << 24) | (uint32_t(resp[11]) << 16)
                | (uint32_t(resp[12]) << 8) | uint32_t(resp[13]);
        }

    private:
        ssize_t recv_validated(uint8_t* resp, size_t resp_size, const char* who)
        {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
            for (;;) {
                struct sockaddr_in from;
                socklen_t from_len = sizeof(from);
                ssize_t r = recvfrom(sock_, resp, resp_size, 0, (struct sockaddr*)&from, &from_len);
                if (r < 0) {
                    throw std::runtime_error(
                        std::string(who) + ": recvfrom failed: " + strerror(errno));
                }
                bool from_device = (from_len >= sizeof(from))
                    && (from.sin_addr.s_addr == addr_.sin_addr.s_addr)
                    && (from.sin_port == addr_.sin_port);
                bool seq_matches = (r >= 4)
                    && (uint16_t((resp[2] << 8) | resp[3]) == seq_);
                if (from_device && seq_matches) {
                    return r;
                }
                if (std::chrono::steady_clock::now() >= deadline) {
                    throw std::runtime_error(
                        std::string(who) + ": timed out waiting for a valid reply");
                }
            }
        }

        int sock_;
        struct sockaddr_in addr_;
        uint16_t seq_;
    };

    /**
     * SPI operations via raw UDP.
     */
    class UdpSpi {
    public:
        UdpSpi(UdpControl& ctrl, uint32_t base, uint32_t cs, uint32_t div,
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

        std::vector<uint8_t> transaction(const std::vector<uint8_t>& cmd,
            const std::vector<uint8_t>& write_data, size_t read_count = 0)
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
            ctrl_.write_uint32(base_ + 4, all_data.size() | (uint32_t(read_count) << 16));

            // Set command length
            ctrl_.write_uint32(base_ + 12, cmd.size() << 8);

            // Start transaction
            ctrl_.write_uint32(base_ + 0, 1);

            // Poll until the SPI engine reports done
            for (int attempt = 0; attempt < 500; ++attempt) {
                uint32_t status = ctrl_.read_uint32(base_ + 0);
                if ((status & SPI_BUSY) == 0) {
                    break;
                }
                if (attempt == 499) {
                    throw std::runtime_error("UdpSpi::transaction: timed out waiting for busy to clear");
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            if (read_count == 0) {
                return {};
            }

            // Read back the response buffer: command bytes, then read data.
            const size_t read_byte_total = cmd.size() + read_count;
            std::vector<uint8_t> response(read_byte_total + 3);
            for (size_t i = 0; i < read_byte_total; i += 4) {
                uint32_t value = ctrl_.read_uint32(base_ + 16 + i);
                response[i + 0] = (value >> 0) & 0xFF;
                response[i + 1] = (value >> 8) & 0xFF;
                response[i + 2] = (value >> 16) & 0xFF;
                response[i + 3] = (value >> 24) & 0xFF;
            }
            response.resize(read_byte_total);
            return { response.begin() + static_cast<std::ptrdiff_t>(cmd.size()), response.end() };
        }

    private:
        UdpControl& ctrl_;
        uint32_t base_;
        uint32_t config_;
    };

    /**
     * Issue a flash-chip SPI command
     */
    std::vector<uint8_t> udp_spi_command(UdpSpi* slow_spi, UdpSpi& fast_spi,
        const std::vector<uint8_t>& cmd, const std::vector<uint8_t>& write_data = {},
        size_t read_count = 0)
    {
        if (slow_spi) {
            slow_spi->transaction({}, { CLNX_WRITE, FLASH_FORWARD_EN, 0x11 });
        }
        return fast_spi.transaction(cmd, write_data, read_count);
    }

    /**
     * Poll the flash chip's STATUS register
     */
    void udp_wait_for_spi_ready(UdpSpi* slow_spi, UdpSpi& fast_spi)
    {
        // 1/2 second, same budget as WinbondW25q128jw::wait_for_spi_ready
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);

        while (std::chrono::steady_clock::now() < deadline) {
            auto result = udp_spi_command(slow_spi, fast_spi, { STATUS }, {}, 1);
            if ((result[0] & 1) == 0) {
                return;
            }
        }
        throw std::runtime_error("udp_wait_for_spi_ready: SPI busy timeout");
    }

    /**
     * Flash a chip over UDP, using the flash chip's STATUS register for coordination
     */
    void udp_flash_chip(const std::string& name, UdpSpi* slow_spi, UdpSpi& fast_spi,
        const std::vector<uint8_t>& firmware)
    {
        std::cout << "[hsb_lite_flash_2407] Programming " << name << " (" << firmware.size() << " bytes)" << std::endl;

        for (size_t erase_addr = 0; erase_addr < firmware.size(); erase_addr += ERASE_SIZE) {
            udp_wait_for_spi_ready(slow_spi, fast_spi);
            udp_spi_command(slow_spi, fast_spi, { WRITE_ENABLE });

            udp_spi_command(slow_spi, fast_spi,
                { BLOCK_ERASE,
                    uint8_t(erase_addr >> 16),
                    uint8_t(erase_addr >> 8),
                    uint8_t(erase_addr) });

            // Program pages in this block
            for (size_t addr = erase_addr;
                 addr < std::min(firmware.size(), erase_addr + ERASE_SIZE);
                 addr += BLOCK_SIZE) {

                if ((addr & 0xFFFF) == 0) {
                    std::cout << "  Address: 0x" << std::hex << addr << std::dec << std::endl;
                }

                udp_wait_for_spi_ready(slow_spi, fast_spi);
                udp_spi_command(slow_spi, fast_spi, { WRITE_ENABLE });

                // Prepare page data
                size_t end = std::min(addr + BLOCK_SIZE, firmware.size());
                std::vector<uint8_t> page_data(firmware.begin() + addr, firmware.begin() + end);

                udp_spi_command(slow_spi, fast_spi,
                    { PAGE_PROGRAM,
                        uint8_t(addr >> 16),
                        uint8_t(addr >> 8),
                        uint8_t(addr) },
                    page_data);
            }
        }

        udp_wait_for_spi_ready(slow_spi, fast_spi);
        std::cout << "[hsb_lite_flash_2407] " << name << " programming complete" << std::endl;
    }

    /**
     * Verify a chip over UDP by reading back every page and comparing it against what
     * was written.
     */
    bool udp_verify_chip(const std::string& name, UdpSpi* slow_spi, UdpSpi& fast_spi,
        const std::vector<uint8_t>& firmware)
    {
        bool ok = true;

        for (size_t addr = 0; addr < firmware.size(); addr += BLOCK_SIZE) {
            if ((addr & 0xFFFF) == 0) {
                std::cout << "  Verify address: 0x" << std::hex << addr << std::dec << std::endl;
            }

            size_t end = std::min(addr + BLOCK_SIZE, firmware.size());
            std::vector<uint8_t> original(firmware.begin() + addr, firmware.begin() + end);

            udp_wait_for_spi_ready(slow_spi, fast_spi);
            auto flash_content = udp_spi_command(slow_spi, fast_spi,
                { READ,
                    uint8_t(addr >> 16),
                    uint8_t(addr >> 8),
                    uint8_t(addr) },
                {}, original.size());

            if (flash_content != original) {
                std::cout << "[hsb_lite_flash_2407] " << name << " verify failed, address=0x"
                          << std::hex << addr << std::dec << std::endl;
                ok = false;
            }
        }

        return ok;
    }

} // anonymous namespace

bool hsb_lite_flash_2407(const std::string& ip_address,
    const std::string& clnx_path,
    const std::string& cpnx_path)
{
    std::cout << "[hsb_lite_flash_2407] UDP flash" << std::endl;
    std::cout << "  IP: " << ip_address << std::endl;
    std::cout << "  CLNX: " << clnx_path << std::endl;
    std::cout << "  CPNX: " << cpnx_path << std::endl;

    try {
        std::cout << "[hsb_lite_flash_2407] Loading firmware files..." << std::endl;
        auto clnx_content = load_firmware_file(clnx_path);
        auto cpnx_content = load_firmware_file(cpnx_path);
        std::cout << "  CLNX size: " << clnx_content.size() << " bytes" << std::endl;
        std::cout << "  CPNX size: " << cpnx_content.size() << " bytes" << std::endl;

        std::cout << "[hsb_lite_flash_2407] Connecting via UDP..." << std::endl;
        UdpControl control(ip_address);

        UdpSpi clnx_slow(control, CLNX_SPI_CTRL, 0, 0xF, 0, 1, 1);
        UdpSpi clnx_fast(control, CLNX_SPI_CTRL, 0, 0x4, 1, 1, 4);
        UdpSpi cpnx_spi(control, CPNX_SPI_CTRL, 0, 0, 1, 1, 1);

        std::cout << "[hsb_lite_flash_2407] Programming CLNX..." << std::endl;
        udp_flash_chip("CLNX", &clnx_slow, clnx_fast, clnx_content);

        std::cout << "[hsb_lite_flash_2407] Verifying CLNX..." << std::endl;
        if (!udp_verify_chip("CLNX", &clnx_slow, clnx_fast, clnx_content)) {
            std::cerr << "[hsb_lite_flash_2407] CLNX verification failed!" << std::endl;
            return false;
        }

        std::cout << "[hsb_lite_flash_2407] Programming CPNX..." << std::endl;
        udp_flash_chip("CPNX", nullptr, cpnx_spi, cpnx_content);

        std::cout << "[hsb_lite_flash_2407] Verifying CPNX..." << std::endl;
        if (!udp_verify_chip("CPNX", nullptr, cpnx_spi, cpnx_content)) {
            std::cerr << "[hsb_lite_flash_2407] CPNX verification failed!" << std::endl;
            return false;
        }

        std::cout << "[hsb_lite_flash_2407] Flash completed successfully!" << std::endl;
        std::cout << "[hsb_lite_flash_2407] Power cycle the device to apply changes." << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[hsb_lite_flash_2407] Error: " << e.what() << std::endl;
        return false;
    }
}

} // namespace hololink
