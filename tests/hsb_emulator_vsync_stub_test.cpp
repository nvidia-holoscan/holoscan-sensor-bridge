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

#include <cstdint>

#include <hololink/emulation/hsb_config.hpp> // for HSB_IP_VERSION
#include <hololink/emulation/hsb_emulator.hpp>

namespace hololink::emulation::tests {

namespace {

    // VSYNC peripheral register addresses (mirrors src/hololink/core/hololink.hpp).
    // Kept as literals here so the test doesn't reach into core/ headers; the
    // purpose of the stub is precisely to accept these addresses from any client.
    constexpr uint32_t kVsyncControl = 0x70000000;
    constexpr uint32_t kVsyncFrequency = 0x70000004;
    constexpr uint32_t kVsyncDelay = 0x70000008;
    constexpr uint32_t kVsyncStart = 0x7000000C;
    constexpr uint32_t kVsyncExposure = 0x70000010;
    constexpr uint32_t kVsyncGpio = 0x70000014;

    // First byte past the registered range — must NOT be accepted by the stub.
    constexpr uint32_t kJustAboveVsync = 0x70000018;
    // Last byte before the registered range — must NOT be accepted by the stub
    // either, since the registration starts exactly at kVsyncControl.
    constexpr uint32_t kJustBelowVsync = 0x6FFFFFFC;

    // An address far from any registered peripheral. Used to prove the new
    // HSBEmulator::write return-code propagation (was previously always 0) and
    // to verify that adding the VSYNC stub didn't widen the accepted address
    // space anywhere else.
    constexpr uint32_t kUnrelatedUnregisteredAddress = 0xDEAD0000;

} // namespace

// Fixture: a fresh HSBEmulator per test, started so its cp_{read,write}_map
// callback tables are built() and queryable. Without start() the map dispatch
// short-circuits to "not found" because the table is unsorted; start() runs
// build() which sorts and validates the registered ranges.
class HsbEmulatorVsyncStubTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        emu_.start();
    }

    HSBEmulator emu_;
};

// Without the stub, RegisterMemory::write() returns non-zero for any address
// not present in cp_write_map and the control-plane handler turns that into
// RESPONSE_INVALID_ADDR on the wire. Writes to VSYNC_CONTROL during
// Hololink::PtpSynchronizer::setup() then abort the client's startup. The
// stub registers a callback for the full VSYNC register block so the write
// succeeds; this test guards that the registration covers the canonical
// 0x70000000 address.
TEST_F(HsbEmulatorVsyncStubTest, VsyncControlWriteSucceeds)
{
    EXPECT_EQ(emu_.write(kVsyncControl, /*value=*/0), 0);
}

// All six registers in the VSYNC peripheral family must be accepted, matching
// the sequence PtpSynchronizer::setup() programs them in: CONTROL, FREQUENCY,
// START, DELAY, EXPOSURE, GPIO. The stub must not be narrower than the full
// block or the client's setup will fail on whichever register isn't covered.
TEST_F(HsbEmulatorVsyncStubTest, AllVsyncRegistersAcceptWrites)
{
    for (uint32_t addr :
        { kVsyncControl, kVsyncFrequency, kVsyncDelay, kVsyncStart, kVsyncExposure, kVsyncGpio }) {
        EXPECT_EQ(emu_.write(addr, /*value=*/0xCAFEBABE), 0)
            << "VSYNC write must succeed at address 0x" << std::hex << addr;
    }
}

// Reads in the VSYNC range must also succeed AND must return 0. The stub
// doesn't track state, so "reads return 0" is the documented contract; this
// test pins that contract so a future contributor doesn't silently add
// stateful behavior that breaks the no-state invariant.
TEST_F(HsbEmulatorVsyncStubTest, AllVsyncRegistersReadAsZero)
{
    for (uint32_t addr :
        { kVsyncControl, kVsyncFrequency, kVsyncDelay, kVsyncStart, kVsyncExposure, kVsyncGpio }) {
        uint32_t value = 0xDEADBEEF;
        ASSERT_EQ(emu_.read(addr, value), 0)
            << "VSYNC read must succeed at address 0x" << std::hex << addr;
        EXPECT_EQ(value, 0u)
            << "stub does not track state; reads must return 0 at 0x" << std::hex << addr;
    }
}

// The stub registration must be range-limited: an address one word past the
// last VSYNC register has no registered callback and must still fail. Catches
// regressions where the registration range is accidentally widened past the
// VSYNC peripheral, which would mask future bugs in unrelated registers.
TEST_F(HsbEmulatorVsyncStubTest, AddressJustAboveRangeStillFails)
{
    EXPECT_NE(emu_.write(kJustAboveVsync, /*value=*/0), 0);
}

// Symmetric to the above: addresses immediately before VSYNC_BASE must also
// remain unhandled. Guards against accidental left-extension of the range.
TEST_F(HsbEmulatorVsyncStubTest, AddressJustBelowRangeStillFails)
{
    EXPECT_NE(emu_.write(kJustBelowVsync, /*value=*/0), 0);
}

// Reproduce the exact sequence PtpSynchronizer::setup() executes, end to end:
// CONTROL=0 -> FREQUENCY -> START -> DELAY -> EXPOSURE -> GPIO. Each must
// succeed. This is the regression check that mirrors the original failure
// signature ("bad_write_response write_uint32((0x70000000,0x0))").
TEST_F(HsbEmulatorVsyncStubTest, PtpSynchronizerSetupSequence)
{
    ASSERT_EQ(emu_.write(kVsyncControl, 0), 0) << "CONTROL=0 (disable) failed";
    ASSERT_EQ(emu_.write(kVsyncFrequency, 30), 0) << "FREQUENCY failed";
    ASSERT_EQ(emu_.write(kVsyncStart, 0), 0) << "START failed";
    ASSERT_EQ(emu_.write(kVsyncDelay, 1000), 0) << "DELAY failed";
    ASSERT_EQ(emu_.write(kVsyncExposure, 0xF4240), 0) << "EXPOSURE failed";
    ASSERT_EQ(emu_.write(kVsyncGpio, 0xF), 0) << "GPIO failed";
}

// Pin the contract for the HSBEmulator::write/read return-code fix bundled
// with this change. Previously these always returned 0, silently masking
// failed writes. After this MR, writes to addresses with no registered
// callback (anywhere outside the VSYNC range or other previously-handled
// peripherals) must propagate the dispatch failure to the caller.
TEST_F(HsbEmulatorVsyncStubTest, UnregisteredWriteReturnsNonZero)
{
    EXPECT_NE(emu_.write(kUnrelatedUnregisteredAddress, /*value=*/0xDEAD), 0)
        << "HSBEmulator::write must propagate the no-callback failure";
}

TEST_F(HsbEmulatorVsyncStubTest, UnregisteredReadReturnsNonZero)
{
    uint32_t value = 0xDEADBEEF;
    EXPECT_NE(emu_.read(kUnrelatedUnregisteredAddress, value), 0)
        << "HSBEmulator::read must propagate the no-callback failure";
}

// HSB_IP_VERSION is registered by HSBEmulator::reset() as part of the
// platform-invariant callback block. Adding our VSYNC stub must not disturb
// that registration. This is a positive smoke test for an unrelated, known-
// good address; combined with the JustAbove/JustBelow Range tests above, it
// proves the stub neither bleeds outside its range nor poisons existing
// registrations.
TEST_F(HsbEmulatorVsyncStubTest, KnownRegisterStillReadable)
{
    uint32_t value = 0;
    EXPECT_EQ(emu_.read(HSB_IP_VERSION, value), 0);
}

} // namespace hololink::emulation::tests
