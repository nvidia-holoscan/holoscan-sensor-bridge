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
 */

// PcieHololink: hololink::Hololink whose control plane is an in-process ECB bridge onto a
// PCIe endpoint's BAR0. Replaces the UDP transport of src/hololink/core/hololink.cpp
// (start/stop/reset/send_control/receive_control) for the hsb_pcie module; the rest of the
// legacy class (WriteData batching, block reads, I2C/SPI/sequencer helpers) is inherited.

#include "pcie_hololink.hpp"

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <thread>

#include <fmt/format.h>

#include <hololink/core/hololink.hpp>
#include <hololink/module/logging.hpp>
#include <hololink/module/pcie/pcie_regs.hpp>

// UUID_REGS / uuid_from_word live next to the loopback endpoint.
#include "loopback_pcie_endpoint.hpp"

namespace hololink::module::hsb_pcie {

namespace {

    std::string uuid_text(const uint8_t* uuid)
    {
        return fmt::format(
            "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
            uuid[0], uuid[1], uuid[2], uuid[3], uuid[4], uuid[5], uuid[6], uuid[7],
            uuid[8], uuid[9], uuid[10], uuid[11], uuid[12], uuid[13], uuid[14], uuid[15]);
    }

    /// A register word that reads 0 or all-ones is what a design mid-reset (or an
    /// unassigned BAR / downed link) returns; neither is a live value.
    bool live_word(uint32_t value)
    {
        return value != 0u && value != 0xFFFFFFFFu;
    }

    /// The UUID must contain at least one byte that is neither 0x00 nor 0xFF.
    bool live_uuid(const uint8_t* uuid)
    {
        for (size_t i = 0; i < UUID_BYTES; i++) {
            if (uuid[i] != 0x00 && uuid[i] != 0xFF) {
                return true;
            }
        }
        return false;
    }

} // namespace

PcieHololink::PcieHololink(std::shared_ptr<PcieEndpointInterface> endpoint,
    std::shared_ptr<AddressPolicy> policy,
    const std::string& peer_key,
    const std::string& serial_number)
    : module_core::LegacyHololinkAccess(peer_key, /*control_port=*/0, serial_number,
        /*sequence_number_checking=*/false)
    // require() throws before bridge_ (below) binds references to *endpoint_ / *policy_.
    , endpoint_(require(std::move(endpoint), "endpoint"))
    , policy_(require(std::move(policy), "address policy"))
    , sequencer_()
    , bridge_(*policy_, *endpoint_)
{
    // The software sequencer is the only engine for APB programs (the endpoint bitfile
    // does not execute CTRL_BUS_EVT/EVT_RAM programs); install it as the policy's SW_EVENT
    // trigger for the lifetime of this object.
    sequencer_ = std::make_unique<ApbSequencer>(*policy_, *endpoint_);
    sequencer_->install();

    // A loopback endpoint (or an endpoint the caller already opened) reports its identity
    // before start(); capture it now so reset() has a reference even without start().
    if (endpoint_->identity().flags & IDENTITY_VERSION_VALID) {
        remember_identity();
    }
}

PcieHololink::~PcieHololink()
{
    // Hololink::~Hololink calls the base stop(), which assumes the UDP sockets and the
    // async-event thread exist when started_ is set. Clear it here first.
    stop();
    // Uninstall the SW_EVENT trigger (it captured `this`) while policy_ is still alive.
    // ~ApbSequencer does this itself; being explicit here pins the order relative to the
    // policy_ shared_ptr, which callers (tests, diagnostics) may keep beyond this object.
    sequencer_.reset();
}

void PcieHololink::remember_identity()
{
    const PcieIdentity id = endpoint_->identity();
    expected_bdf_ = id.bdf;
    std::memcpy(expected_uuid_, id.uuid, UUID_BYTES);
    identity_known_ = true;
    HSB_LOG_DEBUG("hsb_pcie identity: bdf={} uuid={} hsb_version={:#x} hsb_date={} minor={}",
        expected_bdf_, uuid_text(expected_uuid_), id.hsb_version, id.hsb_date, id.minor);
}

void PcieHololink::program_i2c_timing()
{
    using namespace hololink::pcie;
    const uint32_t clk_cnt_address = I2C_CTRL_BASE + I2C_CLK_CNT;
    const uint32_t timeout_address = I2C_CTRL_BASE + I2C_TIMEOUT;
    const uint32_t clk_cnt = policy_->i2c_clk_cnt();
    auto status = endpoint_->bar_write32(clk_cnt_address, clk_cnt);
    if (status != PcieEndpointInterface::Status::SUCCESS) {
        throw std::runtime_error(fmt::format(
            "PcieHololink: programming I2C CLK_CNT at {:#x} failed: {}", clk_cnt_address, to_string(status)));
    }
    status = endpoint_->bar_write32(timeout_address, I2C_TIMEOUT_VALUE);
    if (status != PcieEndpointInterface::Status::SUCCESS) {
        throw std::runtime_error(fmt::format(
            "PcieHololink: programming I2C TIMEOUT at {:#x} failed: {}", timeout_address, to_string(status)));
    }
    HSB_LOG_INFO("hsb_pcie I2C timing: APB {} Hz, CLK_CNT={} (SCL ~{} Hz), TIMEOUT={}",
        policy_->apb_clk_hz(), clk_cnt, policy_->apb_clk_hz() / (2u * clk_cnt), I2C_TIMEOUT_VALUE);
}

bool PcieHololink::release_sensor_resets()
{
    using namespace hololink::pcie;
    // SW_SEN_RST is active low with reset value 0. On the CertusPro-NX Versa bit 0 is wired
    // to the IMX258's reset pin on J37, so a design fresh from configuration holds the
    // camera in reset and its very first I2C transaction NAKs; the Versa bringup tooling
    // (tools/hololink_pcie_bringup/examples/cpnx_versa_imx258/board.py, set_camera_reset())
    // drives exactly this bit before touching the sensor. Releasing on every start() is a
    // no-op where the bits are already 1 and the required power-on release where they are
    // 0. It is also the end state the Ethernet
    // trigger_reset() intends -- it writes 0x3 *before* pulsing SYS_RST, which puts the word
    // back to 0, so reset() restores it once the design is back. Release, never pulse, so a
    // sibling another process may be using is not restarted; a sensor that has been held in
    // reset since power-up has had its reset pulse. Returns true when the word was changed,
    // so the caller can allow the sensor its release settle time.
    uint32_t value = 0;
    auto status = endpoint_->bar_read32(HLNK_CTRL_1, value);
    if (status != PcieEndpointInterface::Status::SUCCESS) {
        throw std::runtime_error(fmt::format(
            "PcieHololink: reading HLNK_CTRL_1 at {:#x} failed: {}", HLNK_CTRL_1, to_string(status)));
    }
    if ((value & HLNK_CTRL_1_SW_SEN_RST_MASK) == HLNK_CTRL_1_SW_SEN_RST_MASK) {
        HSB_LOG_DEBUG("hsb_pcie: sensor resets already released (HLNK_CTRL_1={:#x})", value);
        return false;
    }
    const uint32_t released = value | HLNK_CTRL_1_SW_SEN_RST_MASK;
    status = endpoint_->bar_write32(HLNK_CTRL_1, released);
    if (status != PcieEndpointInterface::Status::SUCCESS) {
        throw std::runtime_error(fmt::format(
            "PcieHololink: releasing SW_SEN_RST at {:#x} failed: {}", HLNK_CTRL_1, to_string(status)));
    }
    HSB_LOG_INFO("hsb_pcie: SW_SEN_RST released (HLNK_CTRL_1 {:#x} -> {:#x})", value, released);
    return true;
}

void PcieHololink::start()
{
    // Calling start is idempotent (same latch as Hololink::start()).
    bool expected = false;
    if (!started_.compare_exchange_strong(expected, true)) {
        return;
    }
    bool opened_here = false;
    try {
        if (!endpoint_->is_open()) {
            if (!endpoint_->open()) {
                throw std::runtime_error(fmt::format(
                    "PcieHololink: could not open PCIe endpoint minor={}", endpoint_->minor()));
            }
            opened_here = true;
        }
        if (!identity_known_) {
            remember_identity();
        }
        program_i2c_timing();
        // Take the sensors out of reset before anything talks to them over I2C. On the
        // Versa this is the camera's hardware reset (see release_sensor_resets()); give a
        // sensor that was actually held in reset its release settle time, the same
        // SENSOR_RESET_SETTLE_MS the Versa bringup tooling waits after releasing it.
        if (release_sensor_resets()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(SENSOR_RESET_SETTLE_MS));
        }
        // Reads HSB_VER/HSB_DATE, primes PTP (shadowed), clears CTRL_EVT and installs the
        // null sequence -- all through the bridge.
        configure_hsb();
    } catch (...) {
        // Leave the object exactly as it was: not started, and without holding the device
        // fd / BAR0 mapping this call acquired (stop() would be a no-op with started_ clear).
        started_ = false;
        if (opened_here) {
            endpoint_->close();
        }
        throw;
    }
    started_ = true;
    HSB_LOG_INFO("hsb_pcie started: minor={} bdf={}", endpoint_->minor(), endpoint_->bdf());
}

void PcieHololink::stop()
{
    if (!started_.exchange(false)) {
        return;
    }
    endpoint_->close();
    HSB_LOG_DEBUG("hsb_pcie stopped: minor={}", endpoint_->minor());
}

bool PcieHololink::probe_live_identity(uint32_t& hsb_version, uint8_t uuid[UUID_BYTES])
{
    using Status = PcieEndpointInterface::Status;
    if (!endpoint_->is_open() && !endpoint_->open()) {
        return false;
    }
    hsb_version = 0;
    if (endpoint_->bar_read32(hololink::pcie::HSB_VER, hsb_version) != Status::SUCCESS
        || !live_word(hsb_version)) {
        return false;
    }
    for (size_t i = 0; i < 4; i++) {
        uint32_t word = 0;
        if (endpoint_->bar_read32(UUID_REGS[i], word) != Status::SUCCESS) {
            return false;
        }
        uuid_from_word(word, uuid, i * 4);
    }
    return live_uuid(uuid);
}

void PcieHololink::reset()
{
    // 0x08 <- 0, 0x08 <- 3, then 0x04 <- 0x8 (SYS_RST). Every one of these is forwarded to
    // the FPGA; the last one genuinely resets the design, so we never trust the pre-reset
    // identity afterwards. On the bitfiles seen so far the sensor receive path does not come
    // back from SYS_RST until the FPGA is reconfigured (AGX Orin, 0x5c694fdc: one reset left
    // every later run -- this module's and the reference player's -- capturing nothing while
    // the sensor kept emitting), so on this transport reset() means "reload the design
    // afterwards", not "drive it into a known state". Say so every time it is called.
    HSB_LOG_WARN("hsb_pcie reset: pulsing SYS_RST on {}; on current bitfiles the sensor receive "
                 "path stays dead afterwards until the FPGA is reconfigured",
        endpoint_->bdf());
    trigger_reset();

    // Wait for the design to answer again. This must be a *live* BAR0 probe: on the
    // driver-backed endpoint identity() is served by HL_PCIE_IOC_IDENTITY, which reports
    // hsb_version / VERSION_VALID from the driver's probe-time cache and only re-reads the
    // UUID, so it says "valid" while the FPGA is still mid-reset. Reading HSB_VER and the
    // UUID registers directly returns 0 / all-ones until the design is back. The loopback
    // answers immediately, so the loop exits on its first pass there.
    const auto deadline = std::chrono::steady_clock::now()
        + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(reset_identity_timeout_s_));
    uint32_t hsb_version = 0;
    uint8_t uuid[UUID_BYTES] = { 0 };
    while (true) {
        const bool live = probe_live_identity(hsb_version, uuid);
        if (live && (!identity_known_ || std::memcmp(uuid, expected_uuid_, UUID_BYTES) == 0)) {
            break;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            if (live) {
                // The design is answering, but it is not the board we started with.
                throw std::runtime_error(fmt::format(
                    "PcieHololink: after reset the endpoint reports UUID {} but {} was expected",
                    uuid_text(uuid), uuid_text(expected_uuid_)));
            }
            throw std::runtime_error(fmt::format(
                "PcieHololink: PCIe endpoint minor={} did not come back within {} s after reset "
                "(HSB_VER reads {:#010x})",
                endpoint_->minor(), reset_identity_timeout_s_, hsb_version));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(RESET_POLL_INTERVAL_MS));
    }

    // The BDF is bound to the PCI device the fd was opened on and survives SYS_RST; the
    // driver (or sysfs) is authoritative for it.
    const PcieIdentity id = endpoint_->identity();
    if (!identity_known_) {
        // Nothing to compare against (never started); adopt what the endpoint reports.
        remember_identity();
    } else if (id.bdf != expected_bdf_) {
        throw std::runtime_error(fmt::format(
            "PcieHololink: after reset the endpoint reports BDF '{}' but '{}' was expected",
            id.bdf, expected_bdf_));
    }
    HSB_LOG_INFO("hsb_pcie reset: minor={} bdf={} re-identified, hsb_version={:#x}",
        id.minor, id.bdf, hsb_version);

    // Re-program the I2C timing the reset wiped and take the sensor receive paths out of the
    // reset SYS_RST put them back into, then let the legacy class reconfigure HSB and notify
    // the registered ResetControllers.
    program_i2c_timing();
    if (release_sensor_resets()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(SENSOR_RESET_SETTLE_MS));
    }
    post_reset_configuration();
}

void PcieHololink::send_control(const std::vector<uint8_t>& request)
{
    pending_reply_ = bridge_.process(request);
}

std::vector<uint8_t> PcieHololink::receive_control(const std::shared_ptr<Timeout>& /*timeout*/)
{
    std::vector<uint8_t> reply = std::move(pending_reply_);
    pending_reply_.clear();
    return reply;
}

} // namespace hololink::module::hsb_pcie
