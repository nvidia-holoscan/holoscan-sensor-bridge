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
 */

#include "jesd.hpp"
#include "logging_internal.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

namespace hololink {

SpiDaemonThread::SpiDaemonThread(Hololink& hololink, JESDConfig& jesd_config)
    : hololink_(hololink)
    , jesd_config_(jesd_config)
    , running_(false)
{
}

void SpiDaemonThread::run()
{
    // Power on the HSB before starting the thread or configuring other components.
    jesd_config_.power_on();

    thread_ = std::thread(&SpiDaemonThread::thread_func, this);
    while (!running_.load(std::memory_order_relaxed)) {
        HSB_LOG_INFO("Waiting for SPI Daemon connection...");
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    HSB_LOG_INFO("SPI Daemon Connected");
}

void SpiDaemonThread::stop()
{
    running_.store(false, std::memory_order_release);
    thread_.join();
}

void SpiDaemonThread::thread_func()
{
    constexpr int SPI_DAEMON_SERVER_PORT = 8400;
    constexpr size_t MAX_SPI_DEVICES = 2;
    uint8_t buffer[300];

    struct sockaddr_in address;
    socklen_t addrlen = sizeof(address);
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(SPI_DAEMON_SERVER_PORT);

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        HSB_LOG_ERROR("Failed to create server socket");
        return;
    }

    // Set REUSEADDR/PORT to ensure that a server can be launched again immediately after closing.
    int enable = 1;
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable)) < 0) {
        HSB_LOG_ERROR("Failed to set SO_REUSEADDR on server socket");
        return;
    }
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEPORT, &enable, sizeof(enable)) < 0) {
        HSB_LOG_ERROR("Failed to set SO_REUSEPORT on server socket");
        return;
    }

    if (bind(server_fd, (struct sockaddr*)&address, addrlen) < 0) {
        HSB_LOG_ERROR("Failed to bind server socket");
        return;
    }

    if (listen(server_fd, 1) < 0) {
        HSB_LOG_ERROR("Failed to listen for client connection");
        return;
    }

    HSB_LOG_INFO("Listening on port {} for SPI Daemon connection...", SPI_DAEMON_SERVER_PORT);

    int sock_fd = accept(server_fd, (struct sockaddr*)&address, &addrlen);
    if (sock_fd < 0) {
        HSB_LOG_ERROR("Failed to accept client connection");
        return;
    }

    std::shared_ptr<Hololink::Spi> spi[MAX_SPI_DEVICES];
    for (size_t i = 0; i < MAX_SPI_DEVICES; ++i) {
        spi[i] = hololink_.get_spi(/*bus_number*/ i,
            /*chip_select*/ 0, /*clock_divisor*/ 15,
            /*cpol*/ 1, /*cpha*/ 1, /*width*/ 1,
            /*spi_address*/ 0x03000000);
    }

    running_.store(true, std::memory_order_release);
    while (running_.load(std::memory_order_relaxed)) {
        // Read message from the daemon.
        int bytes_read = recv(sock_fd, buffer, sizeof(buffer), 0);
        if (bytes_read == 0) {
            break;
        }

        // Parse/execute the message.
        struct hsb_spi_message* msg = (struct hsb_spi_message*)buffer;
        size_t data_size = bytes_read - sizeof(*msg);
        std::vector<uint8_t> read_bytes;
        if (msg->type == HSB_SPI_MSG_TYPE_SPI) {
            // Gather the SPI parameters/bytes.
            uint8_t id = (msg->u.spi.id_cs >> 4) & 0xF;
            uint8_t cs = msg->u.spi.id_cs & 0xF;
            uint8_t* cmd_bytes_base = ((uint8_t*)buffer) + sizeof(*msg);
            std::vector<uint8_t> cmd_bytes(cmd_bytes_base, cmd_bytes_base + msg->u.spi.cmd_bytes);
            uint8_t* wr_bytes_base = cmd_bytes_base + msg->u.spi.cmd_bytes;
            size_t wr_byte_count = msg->u.spi.wr_bytes - msg->u.spi.cmd_bytes;
            std::vector<uint8_t> wr_bytes(wr_bytes_base, wr_bytes_base + wr_byte_count);

            // Dispatch the SPI command.
            read_bytes = spi[id]->spi_transaction(cmd_bytes, wr_bytes, msg->u.spi.rd_bytes);
            HSB_LOG_DEBUG("id={} cs={} cmd=[{:02x}] wr=[{:02x}] rd=[{:02x}]", id, cs,
                fmt::join(cmd_bytes, " "),
                fmt::join(wr_bytes, " "),
                fmt::join(read_bytes, " "));
        } else if (msg->type == HSB_SPI_MSG_TYPE_JESD) {
            if (data_size > 0) {
                HSB_LOG_ERROR("Extra data received with JESD command ({} bytes)", data_size);
            }

            // Execute the JESD transition.
            int result = execute_jesd(msg->u.jesd.id);
            read_bytes.push_back(result);
        } else {
            throw std::runtime_error(fmt::format("Invalid message type: {}", msg->type));
        }

        // Write the read bytes back to the buffer.
        *((uint16_t*)buffer) = htons(read_bytes.size());
        memcpy(buffer + sizeof(uint16_t), read_bytes.data(), read_bytes.size());

        // Send the response.
        size_t response_size = read_bytes.size() + sizeof(uint16_t);
        send(sock_fd, buffer, response_size, 0);
    }

    close(sock_fd);
    close(server_fd);
}

int SpiDaemonThread::execute_jesd(int jesd_state)
{
    HSB_LOG_INFO("JESD Transitioning to state {}", jesd_state);

    switch (jesd_state) {
    case JESD204_OP_LINK_PRE_SETUP:
        jesd_config_.setup_clocks();
        break;
    case JESD204_OP_LINK_SETUP:
        jesd_config_.configure();
        break;
    case JESD204_OP_LINK_ENABLE:
        jesd_config_.run();
        break;
    default:
        // Currently ignore other state transitions.
        break;
    }

    // Return JESD204_STATE_CHANGE_DONE
    return 1;
}

AD9986Config::AD9986Config(Hololink& hololink)
    : hololink_(hololink)
    , jesd_configured_(false)
{
}

void AD9986Config::apply(void)
{
    spi_daemon_thread_.reset(new SpiDaemonThread(hololink_, *this));
    spi_daemon_thread_->run();

    while (!jesd_configured_.load(std::memory_order_relaxed)) {
        HSB_LOG_INFO("Waiting for JESD configuration to complete...");
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    HSB_LOG_INFO("JESD configuration complete");
}

void AD9986Config::host_pause_mapping(uint32_t mask)
{
    host_pause_mapping_mask_ = mask;
}

void AD9986Config::power_on()
{
    HSB_LOG_INFO("JESD::power_on");

    // First check the XCVR's refclk. Switch back if not on refclk 1 before starting config.
    // Only check the first XCVR b/c that should indicate which clock all XCVRs are using.
    // This is a generic function that is associated with the Stratix 10 FPGA E-Tile Transceiver.
    auto refclk = hololink_.read_uint32(0x051003B0) & 0xF;
    if (refclk != 1) {
        HSB_LOG_INFO("Switching XCVRs back to refclk 1");
        for (size_t i = 0; i < 8; ++i) {
            task_refclk_sw(i, 1, 0);
        }
    }

    // Cycle the MxFE power signals
    //  - These signals are connected to circuits/chips on the ADI board.
    //  - The time delays are here to cover a worst-case power ramp down/up
    HSB_LOG_INFO("Cycling MxFE Power Signals");

    hololink_.write_uint32(0x0000000C, 0x0);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    hololink_.write_uint32(0x0000000C, 0xF);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
}

void AD9986Config::setup_clocks()
{
    HSB_LOG_INFO("JESD::setup_clocks");

    // Switch refeclks over to GBTCLK0
    //  - This switches the JESD XCVR reference clock input to the clock from the HMC7044
    //    This is a generic function that is associated with the Stratix 10 FPGA E-Tile Transceiver.
    for (int i = 0; i < 8; i++) {
        if (!task_refclk_sw(i, 2, 0)) {
            HSB_LOG_INFO("RefClk Switch FAILED for channel:{}", i);
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    // Reset the JESD
    //- This is to ensure the nvidia JESD IP is reset after clocks have been stabilized
    hololink_.write_uint32(0x05300000, 0x1);
    hololink_.write_uint32(0x05300000, 0x0);
}

void AD9986Config::configure()
{
    HSB_LOG_INFO("JESD::configure");

    // Packetizer Programming
    hololink_.write_uint32(0x0100000C, 0x10000001);
    hololink_.write_uint32(0x01000004, 0x00000000);
    hololink_.write_uint32(0x01000008, 0xFFFFFFFF);
    hololink_.write_uint32(0x01000004, 0x00000070);
    hololink_.write_uint32(0x01000008, 0x00000001);

    // Map Pause to ethernet interface
    hololink_.write_uint32(0x0120000C, host_pause_mapping_mask_);

    // Configure the JESD IP
    //- This configures the nvidia JESD IP in the specific mode that the original MxFE demo was configured for.
    //- I would imagine these functions would be made more generic to support different JESD modes in the future.
    configure_nvda_jesd_tx();
    configure_nvda_jesd_rx();
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // Enable the TX data (input to the JESD IP) and the RX data (output from the JESD IP)
    hololink_.write_uint32(0x05300000, 0x10);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
}

void AD9986Config::run()
{
    HSB_LOG_INFO("JESD::run");

    // Cycle the RX Link
    //  - This cycles the RX link on the nvidia JESD IP. It's kinda like a reset but not completely.
    //  - TODO: Determine if sleeps are needed here.
    hololink_.write_uint32(0x05039000, 0x0);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    hololink_.write_uint32(0x05039000, 0x1);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // Write the rx lane status to clear errors
    //  - Clears some RX lane status that we are interested in.
    for (int i = 0x0; i < 0x800; i += 0x100) {
        uint32_t address = 0x0503C008 + i;
        hololink_.write_uint32(address, 0xff);
    }

    // Read the status back
    //  - Reads back lane 64B66B status.
    //  - Used to check SH/EMB/User lock status of JESD lanes.
    //  - Read the gearbox status to check any overflow conditions.
    //  - TODO: Intelligently assess the status based on what we read back...
    HSB_LOG_DEBUG("LANE 64B66B Status:");
    for (int i = 0x0; i < 0x800; i += 0x100) {
        uint32_t address = 0x0503C008 + i;
        HSB_LOG_DEBUG("address:{:#x}, value:{:#x}", address, hololink_.read_uint32(address));
    }
    HSB_LOG_DEBUG("UPHY OVERFLOW Status:");
    HSB_LOG_DEBUG("value:{:#x}", hololink_.read_uint32(0x0502102C));
    HSB_LOG_DEBUG("value:{:#x}", hololink_.read_uint32(0x05021448));
    HSB_LOG_DEBUG("value:{:#x}", hololink_.read_uint32(0x05021864));
    HSB_LOG_DEBUG("value:{:#x}", hololink_.read_uint32(0x05021C80));
    HSB_LOG_DEBUG("value:{:#x}", hololink_.read_uint32(0x0502209C));
    HSB_LOG_DEBUG("value:{:#x}", hololink_.read_uint32(0x050224B8));
    HSB_LOG_DEBUG("value:{:#x}", hololink_.read_uint32(0x050228D4));
    HSB_LOG_DEBUG("value:{:#x}", hololink_.read_uint32(0x05022CF0));

    hololink_.write_uint32(0x05300000, 0x30); // Enable RX
    hololink_.write_uint32(0x01200000, 0x3); // Enable TX

    jesd_configured_.store(true, std::memory_order_release);
}

bool AD9986Config::task_refclk_sw(uint32_t channel, uint32_t refclk, uint32_t hwseq)
{
    HSB_LOG_DEBUG("SWITCHING TO REFCLK: {},ON CHANNEL: {}", refclk, channel);

    bool loop_status = false;
    uint32_t retry_cnt = 0;
    uint32_t ch_offset = channel * 0x00010000 + 0x05100000;
    uint32_t data = refclk;

    // Put a retry around this for when the refclk sw fails.
    // Typically a PMA analog reset will fix it.
    while (!loop_status && retry_cnt < 2) {
        if (!task_set_pma_attribute(channel, 0x0030, 0x0003)) { // Switch to refclkB
            HSB_LOG_INFO("Set PMA Attribute Switch to RefClk B FAILED in RefClk Switch");
            HSB_LOG_INFO("Trying PMA Analog Reset");
            task_pma_analog_reset(channel);
            retry_cnt++;
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            continue;
        }

        uint32_t addr = ch_offset + 0xec * 4;
        hololink_.write_uint32(addr, data);

        if (!task_set_pma_attribute(channel, 0x0030, 0x0000)) { // Switch to refclkA
            HSB_LOG_INFO("Set PMA Attribute Switch to RefClk A FAILED in RefClk Switch");
            HSB_LOG_INFO("Trying PMA Analog Reset");
            task_pma_analog_reset(channel);
            retry_cnt++;
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            continue;
        }

        task_pma_analog_reset(channel);
        loop_status = true;
    }

    return loop_status;
}

bool AD9986Config::task_set_pma_attribute(uint32_t channel, uint32_t code, uint32_t data)
{
    bool failed = false;
    uint32_t tries = 0;
    uint32_t rd_data;

    // Added retry method based on PMA_functions_ETILE.c from:
    //  https://community.intel.com/t5/FPGA-Wiki/High-Speed-Transceiver-Demo-Designs-Intel-Stratix-10-TX-Series/ta-p/735133
    //  See PMA_functions_ETILE.zip
    while (tries < 5) {
        HSB_LOG_DEBUG("Setting PMA Attribute:{:#} to:{:#} for channel{}", code, data, channel);

        failed = false;
        uint32_t ch_offset = channel * 0x00010000 + 0x05100000;

        uint32_t low_data = data & 0xFF; // Get the lower byte of data to program and left-pad to 8 characters
        uint32_t high_data = data >> 8 & 0xFF; // Get the upper byte of data to program and left-pad to 8 characters
        uint32_t low_code = code & 0xFF; // Get the lower byte of the attribute code to program and left-pad to 8 characters
        uint32_t high_code = code >> 8 & 0xFF; // Get the upper byte of the attribute code to program and left-pad to 8 characters

        uint32_t addr = ch_offset + 0x8A * 4; // Calculate the address for offset 0x8A and left-pad to 8 characters
        hololink_.write_uint32(addr, 0x00000080); // Write offset 0x8A to 0x80 to clear the bit indicating successful PMA attribute transmission

        addr = ch_offset + 0x84 * 4; // Calculate the address for offset 0x84 and left-pad to 8 characters
        hololink_.write_uint32(addr, low_data); // Write the lower byte of data to 0x84

        addr += 4; // Calculate the address for offset 0x85 and left-pad to 8 characters
        hololink_.write_uint32(addr, high_data); // Write the upper byte of data to 0x85

        addr += 4; // Calculate the address for offset 0x86 and left-pad to 8 characters
        hololink_.write_uint32(addr, low_code); // Write the lower byte of the attribute code to 0x8600

        addr += 4; // Calculate the address for offset 0x87 and left-pad to 8 characters
        hololink_.write_uint32(addr, high_code); // Write the upper byte of the attribute code to 0x87

        addr = ch_offset + 0x90 * 4; // Calculate the address for offset 0x90 and left-pad to 8 characters
        hololink_.write_uint32(addr, 0x00000001); // Write offset 0x90 to issue the PMA attribute

        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        addr = ch_offset + 0x8A * 4; // Calculate the address for offset 0x8A
        rd_data = hololink_.read_uint32(addr) & 0x80; // Read the value of offset 0x8A and mask bit 7
        if (rd_data == 128) { // Check bit 7 is set
            HSB_LOG_DEBUG("Bit 7 of 0x8A is 1. Continuing..."); // If bit 7 is set, then continue
        } else {
            HSB_LOG_DEBUG("Bit 7 of 0x8A is not 1."); // If bit 7 is not set, then error out
            failed = true;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        addr = ch_offset + 0x8B * 4;
        rd_data = hololink_.read_uint32(addr) & 0x1; // Read offset 0x8B and mask bit 0
        if (rd_data) { // If bit 0 is = 1, error out
            HSB_LOG_DEBUG("Bit 0 of 0x8B is not 0.");
            failed = true;
        } else {
            HSB_LOG_DEBUG("Bit 0 of 0x8B is 0. Continuing..."); // If bit 0 is = 0, continue
        }

        if (failed) {
            HSB_LOG_DEBUG("Retrying PMA Attribute due to error.");
            tries++;
        } else {
            HSB_LOG_DEBUG("Retry Count for PMA code:{:#} with data:{:#} :{}", code, data, tries);
            break;
        }
    }

    if (failed) {
        HSB_LOG_ERROR("\tSet PMA attribute failed after{} attempts.", tries);
        return false;
    } else {
        return true;
    }
}

void AD9986Config::task_pma_analog_reset(uint32_t channel)
{
    uint32_t rd_data = 0;
    uint32_t addr = 0;

    HSB_LOG_INFO("Issuing PMA Analog Reset for Channel:{}", channel);
    uint32_t ch_offset = channel * 0x00010000 + 0x05100000;

    for (uint32_t i = 0; i < 3; i++) {
        addr = ch_offset + (0x200 + i) * 4;
        hololink_.write_uint32(addr, 0x00000000);
    }

    addr = ch_offset + (0x200 + 0x3) * 4;
    hololink_.write_uint32(addr, 0x00000081);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    addr = ch_offset + (0x200 + 0x7) * 4;
    rd_data = hololink_.read_uint32(addr) & 0xFF;

    // Check 0x207 is 0x80; if so, return
    if (rd_data != 0x80)
        return;

    addr = ch_offset + 0x95 * 4;
    rd_data = hololink_.read_uint32(addr) | 0x20;
    hololink_.write_uint32(addr, rd_data);

    addr = ch_offset + 0x91 * 4;
    hololink_.write_uint32(addr, 0x00000001);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

void AD9986Config::configure_nvda_jesd_tx(void)
{
    uint32_t tx_base = 0x05040000;

    // Configure TX link parameters
    hololink_.write_uint32(tx_base + 0x39028, 0x0000A802); // CNTRL3

    // Configure 64B66B
    hololink_.write_uint32(tx_base + 0x39030, 0x00000022);

    // Configure sysref control
    hololink_.write_uint32(tx_base + 0x11014, 0x0003C032);

    // Configure TX lane active
    hololink_.write_uint32(tx_base + 0x10018, 0x000000FF); // All lanes active

    // Enable the link
    hololink_.write_uint32(tx_base + 0x39000, 0x00000001);
}

void AD9986Config::configure_nvda_jesd_rx(void)
{
    uint32_t rx_base = 0x05000000;

    // Configure RBD per-lane
    hololink_.write_uint32(rx_base + 0x13008, 0x0000003f); // Calculated from sequence

    // Configure RX lane active
    hololink_.write_uint32(rx_base + 0x10018, 0x000000FF); // All lanes active

    // Configure sysref control
    hololink_.write_uint32(rx_base + 0x11014, 0x0003C032);

    // Configure RX link parameters
    hololink_.write_uint32(rx_base + 0x39028, 0x0000A802); // CNTRL3

    // Configure 64B66B
    hololink_.write_uint32(rx_base + 0x39030, 0x00000022);

    // Enable the link
    hololink_.write_uint32(rx_base + 0x39000, 0x00000001);
}

} // namespace hololink
