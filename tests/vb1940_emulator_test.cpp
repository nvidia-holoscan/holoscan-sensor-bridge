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

#include "gtest/gtest.h"

#include <array>
#include <cstdint>
#include <numeric>
#include <vector>

#include <hololink/emulation/i2c_interface.hpp>
#include <hololink/emulation/sensors/vb1940_emulator.hpp>

namespace hololink::emulation::sensors::tests {

namespace {

    // Peripheral address the Vb1940 rig calibration EEPROM lives at. Mirrors
    // EEPROM_I2C_ADDRESS in vb1940_emulator.cpp (kept as a literal here so the
    // test doesn't depend on the constant being exposed in the public header).
    constexpr uint16_t kEepromAddress = 0x51;

    // Encode the 2-byte big-endian EEPROM register pointer used by 24LCxx-style
    // I²C writes: write_bytes[0] = addr_hi, write_bytes[1] = addr_lo.
    std::vector<uint8_t> make_addr_prefix(uint16_t addr, size_t reserve = 0)
    {
        // added the reserve to silence GCC-11 false-positive overread warnings on static analysis of calling insert() after initializing the 2-element vector with no additional capacity.
        std::vector<uint8_t> buf;
        buf.reserve(2 + reserve);
        buf.push_back(static_cast<uint8_t>(addr >> 8));
        buf.push_back(static_cast<uint8_t>(addr & 0xFF));
        return buf;
    }

    // Build a write buffer: 2-byte address pointer + payload bytes.
    std::vector<uint8_t> make_write(uint16_t addr, const std::vector<uint8_t>& payload)
    {
        auto buf = make_addr_prefix(addr, payload.size());
        buf.insert(buf.end(), payload.begin(), payload.end());
        return buf;
    }

} // namespace

// Without set_eeprom_data() the backing store is zero-initialized, so a read at
// the EEPROM peripheral must succeed and return all-zero bytes. This is the
// pre-patch behavior; the test guards that backwards compatibility is preserved.
TEST(Vb1940EmulatorEeprom, DefaultsToZero)
{
    Vb1940Emulator emu;

    const auto write = make_addr_prefix(0);
    std::array<uint8_t, 32> read_buf {};
    read_buf.fill(0xAA);

    const auto status = emu.i2c_transaction(
        kEepromAddress, write.data(), write.size(), read_buf.data(), read_buf.size());

    EXPECT_EQ(status, I2CStatus::I2C_STATUS_SUCCESS);
    for (uint8_t b : read_buf) {
        EXPECT_EQ(b, 0u);
    }
}

// set_eeprom_data() -> i2c read round-trip returns exactly the bytes written.
TEST(Vb1940EmulatorEeprom, RoundTripViaSetEepromData)
{
    Vb1940Emulator emu;

    std::vector<uint8_t> blob(Vb1940Emulator::EEPROM_REGION_BYTES);
    std::iota(blob.begin(), blob.end(), uint8_t { 1 }); // 1, 2, 3, ...
    emu.set_eeprom_data(blob.data(), blob.size());

    const auto write = make_addr_prefix(0);
    std::vector<uint8_t> read_buf(blob.size(), 0);
    const auto status = emu.i2c_transaction(
        kEepromAddress, write.data(), write.size(), read_buf.data(), read_buf.size());

    EXPECT_EQ(status, I2CStatus::I2C_STATUS_SUCCESS);
    EXPECT_EQ(read_buf, blob);
}

// Reads with a non-zero start address (i.e. the auto-incrementing internal
// address pointer the 24LCxx wire protocol exposes) return bytes starting at
// that address.
TEST(Vb1940EmulatorEeprom, OffsetReadHonoursAddressPointer)
{
    Vb1940Emulator emu;

    std::vector<uint8_t> blob(Vb1940Emulator::EEPROM_REGION_BYTES);
    std::iota(blob.begin(), blob.end(), uint8_t { 0 }); // 0, 1, 2, ...
    emu.set_eeprom_data(blob.data(), blob.size());

    constexpr uint16_t kOffset = 64;
    constexpr size_t kReadLen = 16;
    const auto write = make_addr_prefix(kOffset);
    std::array<uint8_t, kReadLen> read_buf {};
    const auto status = emu.i2c_transaction(
        kEepromAddress, write.data(), write.size(), read_buf.data(), read_buf.size());

    EXPECT_EQ(status, I2CStatus::I2C_STATUS_SUCCESS);
    for (size_t i = 0; i < kReadLen; ++i) {
        EXPECT_EQ(read_buf[i], static_cast<uint8_t>(kOffset + i));
    }
}

// Writes via the 24LCxx wire protocol (write_size > 2) populate the EEPROM, and
// a subsequent read returns the same bytes. This validates the host-can-also-
// program-via-i2c path, independent of set_eeprom_data().
TEST(Vb1940EmulatorEeprom, I2cWriteThenRead)
{
    Vb1940Emulator emu;

    const std::vector<uint8_t> payload { 0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04 };
    const auto write = make_write(0, payload);
    const auto write_status = emu.i2c_transaction(
        kEepromAddress, write.data(), write.size(), nullptr, 0);
    ASSERT_EQ(write_status, I2CStatus::I2C_STATUS_SUCCESS);

    const auto read_addr = make_addr_prefix(0);
    std::vector<uint8_t> read_buf(payload.size(), 0);
    const auto read_status = emu.i2c_transaction(
        kEepromAddress, read_addr.data(), read_addr.size(),
        read_buf.data(), read_buf.size());

    EXPECT_EQ(read_status, I2CStatus::I2C_STATUS_SUCCESS);
    EXPECT_EQ(read_buf, payload);
}

// A read that runs past the end of the 256-byte region returns
// I2C_STATUS_READ_FAILED instead of accessing OOB memory.
TEST(Vb1940EmulatorEeprom, OutOfBoundsReadFails)
{
    Vb1940Emulator emu;

    // Start at the last byte and ask for 2 bytes -- 1 byte over the edge.
    const uint16_t addr = static_cast<uint16_t>(Vb1940Emulator::EEPROM_REGION_BYTES - 1);
    const auto write = make_addr_prefix(addr);
    std::array<uint8_t, 2> read_buf {};
    const auto status = emu.i2c_transaction(
        kEepromAddress, write.data(), write.size(), read_buf.data(), read_buf.size());

    EXPECT_EQ(status, I2CStatus::I2C_STATUS_READ_FAILED);
}

// A write that runs past the end of the 256-byte region returns
// I2C_STATUS_WRITE_FAILED instead of overflowing the backing store.
TEST(Vb1940EmulatorEeprom, OutOfBoundsWriteFails)
{
    Vb1940Emulator emu;

    // Address points to byte 250; payload is 10 bytes -> last 4 bytes overflow.
    const std::vector<uint8_t> payload(10, 0x5A);
    const uint16_t addr = static_cast<uint16_t>(Vb1940Emulator::EEPROM_REGION_BYTES - 6);
    const auto write = make_write(addr, payload);
    const auto status = emu.i2c_transaction(
        kEepromAddress, write.data(), write.size(), nullptr, 0);

    EXPECT_EQ(status, I2CStatus::I2C_STATUS_WRITE_FAILED);
}

// A short set_eeprom_data() payload populates only the leading bytes and
// zero-fills the rest. Repeated calls with shrinking sizes must not leave
// stale bytes from the previous call.
TEST(Vb1940EmulatorEeprom, ShortPayloadZeroPadsTail)
{
    Vb1940Emulator emu;

    // First, populate the whole region with 0xCC so we can detect stale bytes.
    std::vector<uint8_t> full(Vb1940Emulator::EEPROM_REGION_BYTES, 0xCC);
    emu.set_eeprom_data(full.data(), full.size());

    // Now overwrite with a short 8-byte blob. Bytes 8..255 must come back zero.
    const std::vector<uint8_t> short_blob { 1, 2, 3, 4, 5, 6, 7, 8 };
    emu.set_eeprom_data(short_blob.data(), short_blob.size());

    const auto write = make_addr_prefix(0);
    std::vector<uint8_t> read_buf(Vb1940Emulator::EEPROM_REGION_BYTES, 0xFF);
    const auto status = emu.i2c_transaction(
        kEepromAddress, write.data(), write.size(), read_buf.data(), read_buf.size());

    ASSERT_EQ(status, I2CStatus::I2C_STATUS_SUCCESS);
    for (size_t i = 0; i < short_blob.size(); ++i) {
        EXPECT_EQ(read_buf[i], short_blob[i]);
    }
    for (size_t i = short_blob.size(); i < read_buf.size(); ++i) {
        EXPECT_EQ(read_buf[i], 0u) << "stale byte at offset " << i;
    }
}

// set_eeprom_data() must tolerate a null data pointer (treated as a 0-byte
// payload that zero-fills the region) rather than crash inside memcpy.
// Guards against caller bugs that would otherwise produce a segfault.
TEST(Vb1940EmulatorEeprom, NullPointerIsTreatedAsZeroFill)
{
    Vb1940Emulator emu;

    // Pre-populate with a sentinel so we can detect that the call zero-fills.
    std::vector<uint8_t> sentinel(Vb1940Emulator::EEPROM_REGION_BYTES, 0x77);
    emu.set_eeprom_data(sentinel.data(), sentinel.size());

    // nullptr with non-zero size and with zero size -- both must be safe.
    emu.set_eeprom_data(nullptr, 100);
    emu.set_eeprom_data(nullptr, 0);

    const auto write = make_addr_prefix(0);
    std::vector<uint8_t> read_buf(Vb1940Emulator::EEPROM_REGION_BYTES, 0xFF);
    const auto status = emu.i2c_transaction(
        kEepromAddress, write.data(), write.size(), read_buf.data(), read_buf.size());

    ASSERT_EQ(status, I2CStatus::I2C_STATUS_SUCCESS);
    for (uint8_t b : read_buf) {
        EXPECT_EQ(b, 0u) << "nullptr input should leave region zero-filled";
    }
}

// Oversized set_eeprom_data() payloads are clamped to EEPROM_REGION_BYTES; the
// implementation must not write past the backing store.
TEST(Vb1940EmulatorEeprom, OversizedPayloadIsClamped)
{
    Vb1940Emulator emu;

    std::vector<uint8_t> oversized(Vb1940Emulator::EEPROM_REGION_BYTES * 2, 0xAB);
    // Run; if the impl over-writes the buffer, we'd corrupt adjacent memory or
    // ASan would catch it. At minimum the EEPROM region must read back as 0xAB.
    emu.set_eeprom_data(oversized.data(), oversized.size());

    const auto write = make_addr_prefix(0);
    std::vector<uint8_t> read_buf(Vb1940Emulator::EEPROM_REGION_BYTES, 0);
    const auto status = emu.i2c_transaction(
        kEepromAddress, write.data(), write.size(), read_buf.data(), read_buf.size());

    EXPECT_EQ(status, I2CStatus::I2C_STATUS_SUCCESS);
    for (uint8_t b : read_buf) {
        EXPECT_EQ(b, 0xABu);
    }
}

} // namespace hololink::emulation::sensors::tests
