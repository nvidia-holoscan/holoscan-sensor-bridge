/*
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

#include "adsd3500_flash.hpp"
#include "adcam_lib.hpp"
#include "compute_crc.hpp"
#include <cstring>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

#define FLASH_PAGE_SIZE 256
#define WRITE_MASTER_FIRMWARE_COMMAND 0x04
#define WRITE_SLAVE_FIRMWARE_COMMAND 0x2A
#define GET_MASTER_FIRMWARE_COMMAND 0x01
#define GET_SLAVE_FIRMWARE_COMMAND 0x04
#define ADI_STATUS_FIRMWARE_UPDATE 0x000E
#define SET_SWITCH_TO_BURST_MODE 0x0019
#define GET_IMAGER_STATUS_CMD 0x0020
#define RESET_ADSD3500_CMD 0x0024
#define ADI_STATUS_SECOND_FIRMWARE_FLASH_UPDATE 0x0027
#define GET_MASTER_CHIP_ID_CMD 0x0112
#define GET_SLAVE_CHIP_ID_CMD 0x0116
#define GET_DUAL_ADSD3500_ENABLED_CMD                                          \
    0x005A // Read: 0x0001=Dual Enabled, 0x0000=Dual Disabled

/* Seed value for CRC computation */
#define ADI_ROM_CFG_CRC_SEED_VALUE (0xFFFFFFFFu)

/* CRC32 Polynomial to be used for CRC computation */
#define ADI_ROM_CFG_CRC_POLYNOMIAL (0x04C11DB7u)

typedef union {
    uint8_t cmd_header_byte[16];
    struct __attribute__((__packed__)) {
        uint8_t id8;                // 0xAD
        uint16_t chunk_size16;      // 256 is flash page size
        uint8_t cmd8;               // 0x04 is the CMD for fw upgrade
        uint32_t total_size_fw32;   // 4 bytes (total size of firmware)
        uint32_t header_checksum32; // 4 bytes header checksum
        uint32_t crc_of_fw32;       // 4 bytes CRC of the Firmware Binary
    };
} cmd_header_t;

bool Adsd3500::adsd3500_flash(const std::vector<uint8_t> &file_data,
                              std::shared_ptr<hololink::sensors::Adcam> adcam,
                              bool force) {
    this->force = force;
    adcam_ = adcam;

    if (file_data.size() < 2 * ADI_DUAL_FW_SLOT_SIZE) {
        std::cerr << "Firmware file too small to contain both firmware slots"
                  << std::endl;
        return false;
    }

    // Slot 0: master firmware (chunkId=0xAD, chunkType=0x54) at offset 0
    if (file_data[0] != 0xAD || file_data[1] != 0x54) {
        std::cerr
            << "Invalid Slot 0 header (expected chunkId=0xAD, chunkType=0x54)"
            << std::endl;
        return false;
    }
    uint32_t master_len =
        (uint32_t)file_data[8] | ((uint32_t)file_data[9] << 8) |
        ((uint32_t)file_data[10] << 16) | ((uint32_t)file_data[11] << 24);
    if (master_len == 0 ||
        master_len > ADI_DUAL_FW_SLOT_SIZE - ADI_CHUNK_HEADER_SIZE) {
        std::cerr << "Invalid master firmware size in Slot 0 header: "
                  << master_len << " bytes" << std::endl;
        return false;
    }

    // Slot 1: slave firmware (chunkId=0xAD, chunkType=0x60) at offset ADI_DUAL_FW_SLOT_SIZE
    if (file_data[ADI_DUAL_FW_SLOT_SIZE] != 0xAD ||
        file_data[ADI_DUAL_FW_SLOT_SIZE + 1] != 0x60) {
        std::cerr
            << "Invalid Slot 1 header (expected chunkId=0xAD, chunkType=0x60)"
            << std::endl;
        return false;
    }
    uint32_t slave_len =
        (uint32_t)file_data[ADI_DUAL_FW_SLOT_SIZE + 8] |
        ((uint32_t)file_data[ADI_DUAL_FW_SLOT_SIZE + 9] << 8) |
        ((uint32_t)file_data[ADI_DUAL_FW_SLOT_SIZE + 10] << 16) |
        ((uint32_t)file_data[ADI_DUAL_FW_SLOT_SIZE + 11] << 24);
    if (slave_len == 0 ||
        slave_len > ADI_DUAL_FW_SLOT_SIZE - ADI_CHUNK_HEADER_SIZE) {
        std::cerr << "Invalid slave firmware size in Slot 1 header: "
                  << slave_len << " bytes" << std::endl;
        return false;
    }

    // Extract master firmware payload from Slot 0 (after 20-byte chunk header)
    std::vector<uint8_t> master_fw(file_data.begin() + ADI_CHUNK_HEADER_SIZE,
                                   file_data.begin() + ADI_CHUNK_HEADER_SIZE +
                                       master_len);
    if (master_fw.size() != master_len) {
        std::cerr << "Master firmware buffer size mismatch: expected "
                  << master_len << " bytes, got " << master_fw.size()
                  << " bytes" << std::endl;
        return false;
    }

    // Guard against null/zero-filled master firmware payload
    bool master_all_zero = true;
    for (uint32_t i = 0; i < master_len && master_all_zero; i++)
        if (master_fw[i] != 0x00)
            master_all_zero = false;
    if (master_all_zero) {
        std::cerr << "[ERR] Slot 0 master firmware payload is all zeros. "
                     "The .bin file may have been generated from a null "
                     "stream. Aborting."
                  << std::endl;
        return false;
    }

    // Extract master CRC trailer (4 bytes immediately after the page-padded payload)
    uint32_t master_expected_crc =
        (uint32_t)file_data[ADI_CHUNK_HEADER_SIZE + master_len] |
        ((uint32_t)file_data[ADI_CHUNK_HEADER_SIZE + master_len + 1] << 8) |
        ((uint32_t)file_data[ADI_CHUNK_HEADER_SIZE + master_len + 2] << 16) |
        ((uint32_t)file_data[ADI_CHUNK_HEADER_SIZE + master_len + 3] << 24);
    std::cout << "[INFO] Header Master CRC : 0x" << std::hex << std::uppercase
              << std::setw(8) << std::setfill('0') << master_expected_crc
              << std::dec << std::endl;

    // Extract slave firmware payload from Slot 1 (after 20-byte chunk header)
    std::vector<uint8_t> slave_fw(file_data.begin() + ADI_DUAL_FW_SLOT_SIZE +
                                      ADI_CHUNK_HEADER_SIZE,
                                  file_data.begin() + ADI_DUAL_FW_SLOT_SIZE +
                                      ADI_CHUNK_HEADER_SIZE + slave_len);
    if (slave_fw.size() != slave_len) {
        std::cerr << "Slave firmware buffer size mismatch: expected "
                  << slave_len << " bytes, got " << slave_fw.size() << " bytes"
                  << std::endl;
        return false;
    }

    // Guard against null/zero-filled slave firmware payload
    bool slave_all_zero = true;
    for (uint32_t i = 0; i < slave_len && slave_all_zero; i++)
        if (slave_fw[i] != 0x00)
            slave_all_zero = false;
    if (slave_all_zero) {
        std::cerr << "[ERR] Slot 1 slave firmware payload is all zeros. "
                     "The .bin file may have been generated from a null "
                     "stream. Aborting."
                  << std::endl;
        return false;
    }

    // Extract slave CRC trailer (4 bytes immediately after the raw slave payload)
    uint32_t slave_expected_crc =
        (uint32_t)file_data[ADI_DUAL_FW_SLOT_SIZE + ADI_CHUNK_HEADER_SIZE +
                            slave_len] |
        ((uint32_t)file_data[ADI_DUAL_FW_SLOT_SIZE + ADI_CHUNK_HEADER_SIZE +
                             slave_len + 1]
         << 8) |
        ((uint32_t)file_data[ADI_DUAL_FW_SLOT_SIZE + ADI_CHUNK_HEADER_SIZE +
                             slave_len + 2]
         << 16) |
        ((uint32_t)file_data[ADI_DUAL_FW_SLOT_SIZE + ADI_CHUNK_HEADER_SIZE +
                             slave_len + 3]
         << 24);
    std::cout << "[INFO] Header Slave CRC  : 0x" << std::hex << std::uppercase
              << std::setw(8) << std::setfill('0') << slave_expected_crc
              << std::dec << std::endl;

    // Version check: first 4 bytes of both firmware payloads must match
    if (master_len < 4 || slave_len < 4) {
        std::cerr << "Firmware payload too small to contain a version number"
                  << std::endl;
        return false;
    }
    char master_ver_str[32], slave_ver_str[32];
    snprintf(master_ver_str, sizeof(master_ver_str), "%d.%d.%d.%d",
             master_fw[0], master_fw[1], master_fw[2], master_fw[3]);
    snprintf(slave_ver_str, sizeof(slave_ver_str), "%d.%d.%d.%d", slave_fw[0],
             slave_fw[1], slave_fw[2], slave_fw[3]);
    std::cout << "[INFO] Master firmware version : " << master_ver_str
              << std::endl;
    std::cout << "[INFO] Slave  firmware version : " << slave_ver_str
              << std::endl;
    if (master_fw[0] != slave_fw[0] || master_fw[1] != slave_fw[1] ||
        master_fw[2] != slave_fw[2] || master_fw[3] != slave_fw[3]) {
        std::cerr << "[ERR] Version mismatch between master (" << master_ver_str
                  << ") and slave (" << slave_ver_str
                  << ") firmware payloads. Aborting." << std::endl;
        return false;
    }
    std::cout << "[INFO] Firmware version match confirmed: " << master_ver_str
              << std::endl;

    // Probe master device — mandatory
    uint16_t master_chip_id = 0;
    bool master_found = read_cmd(GET_MASTER_CHIP_ID_CMD, &master_chip_id);
    if (!master_found) {
        std::cerr
            << "No ADSD3500 master device detected. Aborting firmware update."
            << std::endl;
        return false;
    }
    std::cout << "[INFO] Master Chip ID is: 0x" << std::hex << std::uppercase
              << std::setw(4) << std::setfill('0') << master_chip_id << std::dec
              << std::endl;

    // Silent probe for slave — absence is expected in single-device configuration
    uint16_t slave_chip_id = 0;
    /* for debugging */
    //bool slave_found = read_cmd(GET_SLAVE_CHIP_ID_CMD, &slave_chip_id);
    bool slave_found = false;
    if (slave_found) {
        std::cout << "[INFO] Slave Chip ID is: 0x" << std::hex << std::uppercase
                  << std::setw(4) << std::setfill('0') << slave_chip_id
                  << std::dec << std::endl;
    } else {
        // Slave chip ID read failed — slave may not be booted yet.
        // Query master to confirm whether dual ADSD3500 is enabled.
        uint16_t dual_enabled = 0;
        bool dual_query_ok =
            read_cmd(GET_DUAL_ADSD3500_ENABLED_CMD, &dual_enabled);
        if (dual_query_ok) {
            std::cout << "[INFO] Get Is Dual ADSD3500 Enabled (0x005A): 0x"
                      << std::hex << std::uppercase << std::setw(4)
                      << std::setfill('0') << dual_enabled << std::dec
                      << std::endl;
            if (dual_enabled == 0x0001) {
                std::cout << "[INFO] Dual ADSD3500 is enabled. Slave device "
                             "confirmed via master query."
                          << std::endl;
                slave_found = true;
            } else {
                std::cout << "[INFO] Dual ADSD3500 is disabled. Single-device "
                             "configuration confirmed."
                          << std::endl;
            }
        } else {
            std::cout << "[INFO] Slave chip ID read failed and dual-enable "
                         "query also failed."
                      << std::endl;
            std::cout << "[INFO] Assuming single-device configuration."
                      << std::endl;
        }
    }

    if (master_found && slave_found) {
        std::cout << std::endl;
        std::cout << "Both ADSD3500 devices detected. Updating master and "
                     "slave firmware."
                  << std::endl;
        if (!this->updateAdsd3500MasterFirmware(master_fw.data(), master_len,
                                                force, master_expected_crc)) {
            std::cerr << "Master firmware update failed." << std::endl;
            return false;
        }

        if (!this->updateAdsd3500SlaveFirmware(slave_fw.data(), slave_len,
                                               force, slave_expected_crc)) {
            std::cerr << "Slave firmware update failed." << std::endl;
            return false;
        }
    } else {
        std::cout << std::endl;
        std::cout
            << "Single ADSD3500 device detected. Updating master firmware only."
            << std::endl;
        if (!this->updateAdsd3500MasterFirmware(master_fw.data(), master_len,
                                                force, master_expected_crc)) {
            std::cerr << "Master firmware update failed." << std::endl;
            return false;
        }
    }

    return true;
}

bool Adsd3500::updateAdsd3500MasterFirmware(uint8_t *fw_data, uint32_t fw_len,
                                            bool force, uint32_t expected_crc) {
    bool status = true;
    uint8_t Wait_Time = 0;
    uint16_t Status_Command = 0;
    uint32_t nResidualCRC = ADI_ROM_CFG_CRC_SEED_VALUE;

    std::cout << std::endl;
    std::cout << "===== updateAdsd3500MasterFirmware: Starting Master Firmware "
                 "Update ====="
              << std::endl;
    if (!adcam_->get_ChipID(GET_MASTER_CHIP_ID_CMD)) {
        std::cerr << "[MASTER] Failed to read Chip ID" << std::endl;
    }
    sleep(1);

    std::cout << std::dec;

    std::cout << "[MASTER] Switching to burst mode" << std::endl;
    if (!adcam_->switch_from_standard_to_burst()) {
        std::cerr << "[MASTER] Failed to switch to burst mode" << std::endl;
    }
    sleep(1);

    std::cout << "[MASTER] Before upgrading new firmware " << std::endl;
    uint8_t current_ver[44] = {0};
    {
        auto ver =
            adcam_->get_fw_version_burst_mode(GET_MASTER_FIRMWARE_COMMAND);
        if (ver.size() != 44) {
            std::cerr
                << "[MASTER] Failed to read current firmware version (got "
                << ver.size() << " bytes, expected 44)" << std::endl;
            return false;
        } else {
            std::memcpy(current_ver, ver.data(), 44);
            std::cout << "[MASTER] Current firmware version  : "
                      << (int)current_ver[0] << "." << (int)current_ver[1]
                      << "." << (int)current_ver[2] << "."
                      << (int)current_ver[3] << std::endl;
        }
    }

    // Version downgrade check
    if (fw_len >= 4) {
        char update_ver_str[32], current_ver_str[32];
        snprintf(update_ver_str, sizeof(update_ver_str), "%d.%d.%d.%d",
                 fw_data[0], fw_data[1], fw_data[2], fw_data[3]);
        snprintf(current_ver_str, sizeof(current_ver_str), "%d.%d.%d.%d",
                 current_ver[0], current_ver[1], current_ver[2],
                 current_ver[3]);
        std::cout << "[MASTER] Update firmware version  : " << update_ver_str
                  << std::endl;

        // Minimum version check: firmware must be >= 8.1.0.0
        const uint8_t min_ver[4] = {8, 1, 0, 0};
        bool below_minimum = false;
        for (int i = 0; i < 4; i++) {
            if (fw_data[i] < min_ver[i]) {
                below_minimum = true;
                break;
            }
            if (fw_data[i] > min_ver[i]) {
                below_minimum = false;
                break;
            }
        }
        if (below_minimum) {
            std::cerr
                << "[MASTER] ERROR: Firmware version " << update_ver_str
                << " is below the minimum required version 8.1.0.0. Aborting."
                << std::endl;
            if (!adcam_->switch_from_burst_to_standard()) {
                std::cerr << "[MASTER] Failed to switch to standard mode"
                          << std::endl;
            }
            return false;
        }

        bool is_downgrade = false;
        for (int i = 0; i < 4; i++) {
            if (fw_data[i] < current_ver[i]) {
                is_downgrade = true;
                break;
            }
            if (fw_data[i] > current_ver[i]) {
                is_downgrade = false;
                break;
            }
        }

        if (is_downgrade) {
            std::cerr << std::endl;
            std::cerr
                << "[MASTER] WARNING: Downgrade detected for master firmware!"
                << std::endl;
            std::cerr << "  Current version : " << current_ver_str << std::endl;
            std::cerr << "  Update version  : " << update_ver_str << std::endl;
            if (!force) {
                std::cerr << "Downgrade requires explicit confirmation."
                          << std::endl;
                std::cerr << "Re-run with --force to proceed." << std::endl;
                if (!adcam_->switch_from_burst_to_standard()) {
                    std::cerr << "[MASTER] Failed to switch to standard mode"
                              << std::endl;
                }
                return false;
            }
            std::cerr
                << "[MASTER] Proceeding with downgrade (--force specified)."
                << std::endl;
        }
    }

    cmd_header_t fw_upgrade_header;
    fw_upgrade_header.id8 = 0xAD;
    fw_upgrade_header.chunk_size16 = 0x0100; // 256=0x100
    fw_upgrade_header.cmd8 = WRITE_MASTER_FIRMWARE_COMMAND;
    fw_upgrade_header.total_size_fw32 = fw_len;
    fw_upgrade_header.header_checksum32 = 0;

    for (int i = 1; i < 8; i++) {
        fw_upgrade_header.header_checksum32 +=
            fw_upgrade_header.cmd_header_byte[i];
    }

    crc_parameters_t crc_params;
    crc_params.type = CRC_32bit;
    crc_params.polynomial.polynomial_crc32_bit = ADI_ROM_CFG_CRC_POLYNOMIAL;
    crc_params.initial_crc.crc_32bit = nResidualCRC;
    crc_params.crc_compute_flags = IS_CRC_MIRROR;

    crc_output_t res = compute_crc(&crc_params, fw_data, fw_len);
    nResidualCRC = ~res.crc_32bit;
    std::cout << "[MASTER] CRC raw result : 0x" << std::hex << std::uppercase
              << std::setw(8) << std::setfill('0') << res.crc_32bit << std::dec
              << std::endl;
    std::cout << "[MASTER] nResidualCRC   : 0x" << std::hex << std::uppercase
              << std::setw(8) << std::setfill('0') << nResidualCRC << std::dec
              << std::endl;
    std::cout << "[MASTER] Expected CRC   : 0x" << std::hex << std::uppercase
              << std::setw(8) << std::setfill('0') << expected_crc << std::dec
              << std::endl;
    if (nResidualCRC != expected_crc) {
        std::cerr << "[MASTER] CRC MISMATCH: computed 0x" << std::hex
                  << std::uppercase << std::setw(8) << std::setfill('0')
                  << nResidualCRC << " != expected 0x" << std::setw(8)
                  << std::setfill('0') << expected_crc << std::dec << std::endl;
        return false;
    }
    std::cout << "[MASTER] CRC OK: computed CRC matches expected CRC."
              << std::endl;

    fw_upgrade_header.crc_of_fw32 = (uint32_t)nResidualCRC;

    {
        const uint8_t *p = fw_upgrade_header.cmd_header_byte;
        uint16_t reg[9];
        reg[0] = 8;
        for (int i = 0; i < 8; ++i)
            reg[i + 1] = (uint16_t(p[2 * i]) << 8) | uint16_t(p[2 * i + 1]);
        status = adcam_->set_register16_no_response(reg);
    }
    if (!status) {
        std::cout << std::endl;
        std::cerr << "[MASTER] Failed to send fw upgrade header" << std::endl;
        return status;
    }

    int packetsToSend;
    if ((fw_len % FLASH_PAGE_SIZE) != 0) {
        packetsToSend = (fw_len / FLASH_PAGE_SIZE + 1);
    } else {
        packetsToSend = (fw_len / FLASH_PAGE_SIZE);
    }

    uint8_t data_out[FLASH_PAGE_SIZE];

    std::cout << std::endl;
    std::cout << "[MASTER] Writing Firmware packets..." << std::endl;
    for (int i = 0; i < packetsToSend; i++) {
        int start = FLASH_PAGE_SIZE * i;
        int end = FLASH_PAGE_SIZE * (i + 1);

        for (int j = start; j < end; j++) {
            if (j < (int)fw_len) {
                data_out[j - start] = fw_data[j];
            } else {
                data_out[j - start] = 0x00;
            }
        }
        status = write_payload(data_out, FLASH_PAGE_SIZE);

        if (!status) {
            std::cerr << "[MASTER] Failed to send packet number " << i
                      << " out of " << packetsToSend << " packets!"
                      << std::endl;
            return status;
        }

        std::cout << "[MASTER] Packet number: " << i + 1 << " / "
                  << packetsToSend << '\r';
        fflush(stdout);
    }
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "[MASTER] Adsd3500 master firmware packets sent successfully!"
              << std::endl;

    std::cout << std::endl;
    for (int i = 20; i >= 0; i--) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::cout << "[MASTER] Waiting for " << i << " seconds" << '\r';
        fflush(stdout);
    }
    std::cout << std::endl;

    status = read_cmd(GET_IMAGER_STATUS_CMD, &Status_Command);
    std::cout << "[MASTER] Get status Command 0x" << std::hex << std::uppercase
              << std::setw(2) << std::setfill('0') << Status_Command << std::dec
              << std::endl;

    if (Status_Command != ADI_STATUS_FIRMWARE_UPDATE) {
        std::cout << "[MASTER] Firmware update failed" << std::endl;
        return false;
    }

    sleep(2);

    /*Soft Reset the ADSD3500*/
    status = write_cmd(RESET_ADSD3500_CMD, 0x0000);
    if (!status) {
        std::cout << std::endl;
        std::cerr << "Failed to Soft Reset the ADSD3500!" << std::endl;
        return status;
    } else {
        std::cout << std::endl;
        std::cout << "[MASTER] Firmware soft resetting...";
    }

    std::cout << std::endl;
    for (int i = 9; i >= 0; i--) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::cout << "[MASTER] Waiting for " << i << " seconds" << '\r';
        fflush(stdout);
    }
    std::cout << std::endl;

    std::cout << "[MASTER] ";
    if (!adcam_->get_ChipID(GET_MASTER_CHIP_ID_CMD)) {
        std::cerr << "[MASTER] Failed to read Chip ID" << std::endl;
    }
    sleep(1);

    if (!adcam_->switch_from_standard_to_burst()) {
        std::cerr << "[MASTER] Failed to switch to burst mode" << std::endl;
    }
    sleep(1);

    std::cout << std::endl;
    std::cout << "[MASTER] After upgrading new firmware " << std::endl;
    {
        auto ver =
            adcam_->get_fw_version_burst_mode(GET_MASTER_FIRMWARE_COMMAND);
        if (ver.size() != 44) {
            std::cerr
                << "[MASTER] Failed to read updated firmware version (got "
                << ver.size() << " bytes, expected 44)" << std::endl;
        } else {
            std::cout << "[MASTER] Updated firmware version   : " << (int)ver[0]
                      << "." << (int)ver[1] << "." << (int)ver[2] << "."
                      << (int)ver[3] << std::endl;
        }
    }
    sleep(1);

    if (!adcam_->switch_from_burst_to_standard()) {
        std::cerr << "[MASTER] Failed to switch to standard mode" << std::endl;
    }
    sleep(1);

    std::cout << std::endl << "[MASTER] ";
    if (!adcam_->get_ChipID(GET_MASTER_CHIP_ID_CMD)) {
        std::cerr << "[MASTER] Failed to read Chip ID" << std::endl;
    }

    return true;
}

bool Adsd3500::updateAdsd3500SlaveFirmware(uint8_t *fw_data, uint32_t fw_len,
                                           bool force, uint32_t expected_crc) {

    bool status = true;
    uint8_t Wait_Time = 0;
    uint16_t Status_Command = 0;
    uint32_t nResidualCRC = ADI_ROM_CFG_CRC_SEED_VALUE;

    std::cout << std::endl;
    std::cout << "===== updateAdsd3500SlaveFirmware: Starting Slave Firmware "
                 "Update ====="
              << std::endl;
    /* Commented out slave chip ID read for now since the master fails to read the slave chip ID.
    std::cout << "[SLAVE] ";
    if (!adcam_->get_ChipID(GET_SLAVE_CHIP_ID_CMD)) {
        std::cerr << "[SLAVE] Failed to read Chip ID" << std::endl;
    }
    */
    sleep(1);

    std::cout << std::dec;

    if (!adcam_->switch_from_standard_to_burst()) {
        std::cerr << "[SLAVE] Failed to switch to burst mode" << std::endl;
    }
    sleep(1);

    std::cout << std::endl;
    std::cout << "[SLAVE] Before upgrading new firmware " << std::endl;
    uint8_t current_ver[44] = {0};
    {
        auto ver =
            adcam_->get_fw_version_burst_mode(GET_SLAVE_FIRMWARE_COMMAND);
        if (ver.size() != 44) {
            std::cerr << "[SLAVE] Failed to read current firmware version (got "
                      << ver.size() << " bytes, expected 44)" << std::endl;
        } else {
            std::memcpy(current_ver, ver.data(), 44);
            std::cout << "[SLAVE] Current firmware version   : "
                      << (int)current_ver[0] << "." << (int)current_ver[1]
                      << "." << (int)current_ver[2] << "."
                      << (int)current_ver[3] << std::endl;
        }
    }

    // Version downgrade check
    if (fw_len >= 4) {
        char update_ver_str[32], current_ver_str[32];
        snprintf(update_ver_str, sizeof(update_ver_str), "%d.%d.%d.%d",
                 fw_data[0], fw_data[1], fw_data[2], fw_data[3]);
        snprintf(current_ver_str, sizeof(current_ver_str), "%d.%d.%d.%d",
                 current_ver[0], current_ver[1], current_ver[2],
                 current_ver[3]);
        std::cout << "[SLAVE] Update firmware version   : " << update_ver_str
                  << std::endl;

        // Minimum version check: firmware must be >= 8.1.0.0
        const uint8_t min_ver[4] = {8, 1, 0, 0};
        bool below_minimum = false;
        for (int i = 0; i < 4; i++) {
            if (fw_data[i] < min_ver[i]) {
                below_minimum = true;
                break;
            }
            if (fw_data[i] > min_ver[i]) {
                below_minimum = false;
                break;
            }
        }
        if (below_minimum) {
            std::cerr
                << "[SLAVE] ERROR: Firmware version " << update_ver_str
                << " is below the minimum required version 8.1.0.0. Aborting."
                << std::endl;
            if (!adcam_->switch_from_burst_to_standard()) {
                std::cerr << "[SLAVE] Failed to switch to standard mode"
                          << std::endl;
            }
            return false;
        }

        bool is_downgrade = false;
        for (int i = 0; i < 4; i++) {
            if (fw_data[i] < current_ver[i]) {
                is_downgrade = true;
                break;
            }
            if (fw_data[i] > current_ver[i]) {
                is_downgrade = false;
                break;
            }
        }

        if (is_downgrade) {
            std::cerr << std::endl;
            std::cerr
                << "[SLAVE] WARNING: Downgrade detected for slave firmware!"
                << std::endl;
            std::cerr << "  Current version : " << current_ver_str << std::endl;
            std::cerr << "  Update version  : " << update_ver_str << std::endl;
            if (!force) {
                std::cerr << "Downgrade requires explicit confirmation."
                          << std::endl;
                std::cerr << "Re-run with --force to proceed." << std::endl;
                if (!adcam_->switch_from_burst_to_standard()) {
                    std::cerr << "[SLAVE] Failed to switch to standard mode"
                              << std::endl;
                }
                return false;
            }
            std::cerr
                << "[SLAVE] Proceeding with downgrade (--force specified)."
                << std::endl;
        }
    }

    cmd_header_t fw_upgrade_header;
    fw_upgrade_header.id8 = 0xAD;
    fw_upgrade_header.chunk_size16 = 0x0100; // 256=0x100
    fw_upgrade_header.cmd8 = WRITE_SLAVE_FIRMWARE_COMMAND;
    fw_upgrade_header.total_size_fw32 = fw_len;
    fw_upgrade_header.header_checksum32 = 0;

    for (int i = 1; i < 8; i++) {
        fw_upgrade_header.header_checksum32 +=
            fw_upgrade_header.cmd_header_byte[i];
    }

    crc_parameters_t crc_params;
    crc_params.type = CRC_32bit;
    crc_params.polynomial.polynomial_crc32_bit = ADI_ROM_CFG_CRC_POLYNOMIAL;
    crc_params.initial_crc.crc_32bit = nResidualCRC;
    crc_params.crc_compute_flags = IS_CRC_MIRROR;

    crc_output_t res = compute_crc(&crc_params, fw_data, fw_len);
    nResidualCRC = ~res.crc_32bit;
    std::cout << "[SLAVE] CRC raw result : 0x" << std::hex << std::uppercase
              << std::setw(8) << std::setfill('0') << res.crc_32bit << std::dec
              << std::endl;
    std::cout << "[SLAVE] nResidualCRC   : 0x" << std::hex << std::uppercase
              << std::setw(8) << std::setfill('0') << nResidualCRC << std::dec
              << std::endl;
    std::cout << "[SLAVE] Expected CRC   : 0x" << std::hex << std::uppercase
              << std::setw(8) << std::setfill('0') << expected_crc << std::dec
              << std::endl;
    if (nResidualCRC != expected_crc) {
        std::cerr << "[SLAVE] CRC MISMATCH: computed 0x" << std::hex
                  << std::uppercase << std::setw(8) << std::setfill('0')
                  << nResidualCRC << " != expected 0x" << std::setw(8)
                  << std::setfill('0') << expected_crc << std::dec << std::endl;
        return false;
    }
    std::cout << "[SLAVE] CRC OK: computed CRC matches expected CRC."
              << std::endl;

    fw_upgrade_header.crc_of_fw32 = (uint32_t)nResidualCRC;

    {
        const uint8_t *p = fw_upgrade_header.cmd_header_byte;
        uint16_t reg[9];
        reg[0] = 8;
        for (int i = 0; i < 8; ++i)
            reg[i + 1] = (uint16_t(p[2 * i]) << 8) | uint16_t(p[2 * i + 1]);
        status = adcam_->set_register16_no_response(reg);
    }
    if (!status) {
        std::cout << std::endl;
        std::cerr << "[SLAVE] Failed to send fw upgrade header" << std::endl;
        return status;
    }

    int packetsToSend;
    if ((fw_len % FLASH_PAGE_SIZE) != 0) {
        packetsToSend = (fw_len / FLASH_PAGE_SIZE + 1);
    } else {
        packetsToSend = (fw_len / FLASH_PAGE_SIZE);
    }

    uint8_t data_out[FLASH_PAGE_SIZE];

    std::cout << std::endl;
    std::cout << "[SLAVE] Writing Firmware packets..." << std::endl;
    for (int i = 0; i < packetsToSend; i++) {
        int start = FLASH_PAGE_SIZE * i;
        int end = FLASH_PAGE_SIZE * (i + 1);

        for (int j = start; j < end; j++) {
            if (j < (int)fw_len) {
                data_out[j - start] = fw_data[j];
            } else {
                data_out[j - start] = 0x00;
            }
        }
        status = write_payload(data_out, FLASH_PAGE_SIZE);

        if (!status) {
            std::cerr << "[SLAVE] Failed to send packet number " << i
                      << " out of " << packetsToSend << " packets!"
                      << std::endl;
            return status;
        }

        std::cout << "[SLAVE] Packet number: " << i + 1 << " / "
                  << packetsToSend << '\r';
        fflush(stdout);
    }
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "[SLAVE] Adsd3500 slave firmware packets sent successfully!"
              << std::endl;

    std::cout << std::endl;
    for (int i = 20; i >= 0; i--) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::cout << "[SLAVE] Waiting for " << i << " seconds" << '\r';
        fflush(stdout);
    }

    sleep(2);

    if (!adcam_->switch_from_burst_to_standard()) {
        std::cerr << "[SLAVE] Failed to switch to standard mode" << std::endl;
    }
    sleep(1);

    status = read_cmd(GET_IMAGER_STATUS_CMD, &Status_Command);
    std::cout << "[SLAVE] Get status Command 0x" << std::hex << std::uppercase
              << std::setw(2) << std::setfill('0') << Status_Command << std::dec
              << std::endl;

    if (Status_Command != ADI_STATUS_SECOND_FIRMWARE_FLASH_UPDATE) {
        std::cout << "Slave Firmware write failed" << std::endl;
        return false;
    } else {
        std::cout << "Slave Firmware Flash write completed and is successful."
                  << std::endl;
    }

    /*Soft Reset the ADSD3500*/
    status = write_cmd(RESET_ADSD3500_CMD, 0x0000);
    if (!status) {
        std::cout << std::endl;
        std::cerr << "Failed to Soft Reset the ADSD3500!" << std::endl;
        return status;
    } else {
        std::cout << std::endl;
        std::cout << "[SLAVE] Firmware soft resetting..." << std::endl;
    }

    std::cout << std::endl;
    for (int i = 9; i >= 0; i--) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::cout << "[SLAVE] Waiting for " << i << " seconds" << '\r';
        fflush(stdout);
    }
    std::cout << std::endl;

    /* Commented out slave chip ID read for now since the master fails to read the slave chip ID.
    std::cout << "[SLAVE] ";
    if (!adcam_->get_ChipID(GET_SLAVE_CHIP_ID_CMD)) {
        std::cerr << "[SLAVE] Failed to read Chip ID" << std::endl;
    }
    */
    sleep(1);

    if (!adcam_->switch_from_standard_to_burst()) {
        std::cerr << "[SLAVE] Failed to switch to burst mode" << std::endl;
    }
    sleep(1);

    std::cout << std::endl;
    std::cout << "[SLAVE] After upgrading new firmware " << std::endl;
    {
        auto ver =
            adcam_->get_fw_version_burst_mode(GET_SLAVE_FIRMWARE_COMMAND);
        if (ver.size() != 44) {
            std::cerr << "[SLAVE] Failed to read updated firmware version (got "
                      << ver.size() << " bytes, expected 44)" << std::endl;
        } else {
            std::cout << "[SLAVE] Updated firmware version    : " << (int)ver[0]
                      << "." << (int)ver[1] << "." << (int)ver[2] << "."
                      << (int)ver[3] << std::endl;
        }
    }
    sleep(1);

    if (!adcam_->switch_from_burst_to_standard()) {
        std::cerr << "[SLAVE] Failed to switch to standard mode" << std::endl;
    }
    sleep(1);

    /* Commented out slave chip ID read for now since the master fails to read the slave chip ID.
    std::cout << std::endl << "[SLAVE] ";
    if (!adcam_->get_ChipID(GET_SLAVE_CHIP_ID_CMD)) {
        std::cerr << "[SLAVE] Failed to read Chip ID" << std::endl;
    }
    */
    return true;
}

// =============================================================================
// I2C command helpers (delegate to Adcam)
// =============================================================================

bool Adsd3500::write_payload(uint8_t *payload, uint16_t payload_len) {
    if (payload_len == 0 || payload_len % 2 != 0) {
        std::cerr
            << "write_payload: payload_len must be a non-zero even number, got "
            << payload_len << std::endl;
        return false;
    }
    uint16_t word_count = payload_len / 2;
    std::vector<uint16_t> reg(1 + word_count);
    reg[0] = word_count;
    for (uint16_t i = 0; i < word_count; i++) {
        reg[i + 1] =
            (uint16_t(payload[2 * i]) << 8) | uint16_t(payload[2 * i + 1]);
    }
    return adcam_->set_register16_no_response(reg.data());
}

bool Adsd3500::write_cmd(uint16_t cmd, uint16_t value) {
    uint16_t reg[] = {2, cmd, value};
    return adcam_->set_register16_no_response(reg);
}

bool Adsd3500::read_cmd(uint16_t cmd, uint16_t *data) {
    if (!data) {
        std::cerr << "read_cmd: null data pointer" << std::endl;
        return false;
    }
    uint16_t reg[] = {1, cmd};
    auto resp = adcam_->set_register16_response(reg, 2);
    if (resp.size() < 2) {
        return false;
    }
    *data = (uint16_t(resp[0]) << 8) | uint16_t(resp[1]);
    return true;
}

bool Adsd3500::read_burst_cmd(uint8_t *payload, uint16_t payload_len,
                              uint8_t *data) {
    if (!data) {
        std::cerr << "read_burst_cmd: null data pointer" << std::endl;
        return false;
    }
    if (payload_len == 0 || payload_len % 2 != 0) {
        std::cerr << "read_burst_cmd: payload_len must be a non-zero even "
                     "number, got "
                  << payload_len << std::endl;
        return false;
    }
    uint16_t word_count = payload_len / 2;
    std::vector<uint16_t> reg(1 + word_count);
    reg[0] = word_count;
    for (uint16_t i = 0; i < word_count; i++) {
        reg[i + 1] =
            (uint16_t(payload[2 * i]) << 8) | uint16_t(payload[2 * i + 1]);
    }
    auto resp = adcam_->set_register16_response(reg.data(), 44);
    if (resp.size() < 44) {
        return false;
    }
    memcpy(data, resp.data(), 44);
    return true;
}