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

#ifndef HOLOLINK_MODULE_HSB_PCIE_PCIE_HOLOLINK_HPP
#define HOLOLINK_MODULE_HSB_PCIE_PCIE_HOLOLINK_HPP

/**
 * @file
 * @brief hololink::Hololink carried over a PCIe endpoint instead of UDP.
 *
 * The legacy Hololink funnels every control-plane access through execute() ->
 * send_control() / receive_control() (src/hololink/core/hololink.cpp). This subclass
 * overrides that seam: a request frame is decoded by EcbBridge, each (address[, value])
 * pair is dispatched through the AddressPolicy onto the endpoint's BAR0, and the reply is
 * synthesized in-process. It is the in-process replacement for the UDP round trip to the
 * hsb_pcie emulator daemon (hsb_pcie/driver/src/hsb_pcie/hsb_pcie_emulator.cpp): no control
 * socket, no async-event socket, no listener thread.
 *
 * It rides under module_core::LegacyHololinkAccess so the module_core siblings (I2cV1,
 * SequencerV1, DataChannelV1, PtpPpsOutputV1) reach it through HololinkV1::legacy_access()
 * unchanged.
 */

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "hololink_default.hpp"

#include "address_policy.hpp"
#include "apb_sequencer.hpp"
#include "ecb_bridge.hpp"
#include "pcie_endpoint_interface.hpp"

namespace hololink::module::hsb_pcie {

class PcieHololink : public module_core::LegacyHololinkAccess {
public:
    /// I2C timeout programmed by start() (BAR0 I2C_CTRL_BASE + TIMEOUT), in APB clocks. The
    /// SCL divider written next to it is the policy's i2c_clk_cnt(): AddressPolicy::I2C_SCL_HZ
    /// at the Versa's APB clock (AddressPolicy::DEFAULT_APB_CLK_HZ).
    static constexpr uint32_t I2C_TIMEOUT_VALUE = 5000000u;
    /// Settle time allowed after start()/reset() actually release a held SW_SEN_RST bit,
    /// before the first I2C transaction: on the CertusPro-NX Versa that bit is the camera's
    /// hardware reset, and its bringup tooling waits this long after releasing it.
    static constexpr unsigned SENSOR_RESET_SETTLE_MS = 10;
    /// Default for how long reset() waits for the design to answer live BAR0 identity reads
    /// again after SYS_RST (see set_reset_identity_timeout()).
    static constexpr double RESET_IDENTITY_TIMEOUT_S = 5.0;
    /// Interval between the live identity probes reset() issues while waiting.
    static constexpr unsigned RESET_POLL_INTERVAL_MS = 50;

    /**
     * @param endpoint the BAR0 carrier (PcieEndpoint over the driver, or LoopbackPcieEndpoint)
     * @param policy the HSB-address -> BAR0 policy every ECB pair is dispatched through
     * @param peer_key the channel key (the endpoint's BDF "DDDD:BB:DD.F"); stored as Hololink::peer_ip_
     * @param serial_number per-board serial ("pcie-<BDF>"), used for the NamedLock files
     * @throws std::runtime_error when endpoint or policy is null (checked before any member
     *         that dereferences them is constructed)
     */
    PcieHololink(std::shared_ptr<PcieEndpointInterface> endpoint,
        std::shared_ptr<AddressPolicy> policy,
        const std::string& peer_key,
        const std::string& serial_number);
    ~PcieHololink() override;

    /**
     * Idempotent. Opens the endpoint, programs the I2C block's timing (CLK_CNT=125 for
     * 400 kHz, TIMEOUT=5000000), runs configure_hsb() and marks the device started. Opens
     * no sockets and starts no threads. On failure the device is left not-started and an
     * endpoint this call opened is closed again before the exception propagates.
     */
    void start() override;

    /// Marks the device stopped and closes the endpoint.
    void stop() override;

    /**
     * trigger_reset() (0x08 <- 0, 0x08 <- 3, 0x04 <- 8: SYS_RST is forwarded, so the endpoint
     * really resets), then waits for the design to come back by issuing *live* BAR0 reads of
     * HSB_VER and HSB_UUID_3..0 (re-opening the endpoint if the reset dropped it) until
     * HSB_VER reads as a design (not 0 / all-ones) and the UUID equals the one captured at
     * construction/start(); the driver's IDENTITY ioctl is deliberately not used for this
     * because it serves the version and VERSION_VALID from its probe-time cache. Once live,
     * the BDF is checked, the I2C timing is re-programmed and post_reset_configuration()
     * runs. Throws std::runtime_error when the design does not come back within
     * reset_identity_timeout() (a UUID that is live but different is reported as such), or
     * when the BDF changed.
     */
    void reset() override;

    /// Override the wait bound reset() applies (seconds); tests use this to keep the
    /// never-recovers case short.
    void set_reset_identity_timeout(double seconds) { reset_identity_timeout_s_ = seconds; }
    double reset_identity_timeout() const { return reset_identity_timeout_s_; }

    /// Decode the request, dispatch it, and park the reply for receive_control().
    void send_control(const std::vector<uint8_t>& request) override;
    /// Return (and clear) the reply parked by send_control(); empty when none is pending.
    std::vector<uint8_t> receive_control(const std::shared_ptr<Timeout>& timeout) override;

    // ---- accessors (tests, diagnostics) ---------------------------------------------------
    EcbBridge& bridge() { return bridge_; }
    AddressPolicy& policy() { return *policy_; }
    /// CLK_CNT start()/reset() program: AddressPolicy::I2C_SCL_HZ at the policy's APB clock.
    uint32_t i2c_clk_cnt() const { return policy_->i2c_clk_cnt(); }
    PcieEndpointInterface& endpoint() { return *endpoint_; }
    const std::shared_ptr<AddressPolicy>& policy_ptr() const { return policy_; }
    const std::shared_ptr<PcieEndpointInterface>& endpoint_ptr() const { return endpoint_; }
    ApbSequencer& sequencer() { return *sequencer_; }
    /// True once BDF/UUID have been captured for reset() to compare against.
    bool identity_known() const { return identity_known_; }
    const std::string& expected_bdf() const { return expected_bdf_; }
    const uint8_t* expected_uuid() const { return expected_uuid_; }

private:
    /// Throw std::runtime_error when `p` is null; used in the member-initializer list so
    /// the check runs before EcbBridge/ApbSequencer bind references to *p.
    template <typename T>
    static std::shared_ptr<T> require(std::shared_ptr<T> p, const char* what)
    {
        if (!p) {
            throw std::runtime_error(std::string("PcieHololink: ") + what + " must be non-null");
        }
        return p;
    }

    /// Snapshot the endpoint's BDF and UUID as the reference reset() compares against.
    void remember_identity();
    /// Program the I2C block's timing directly on BAR0 (no HSB equivalent for TIMEOUT).
    void program_i2c_timing();
    /// Take every sensor out of reset (HLNK_CTRL_1 SW_SEN_RST bits set): on a design fresh
    /// from configuration, or after SYS_RST returned the word to 0. A read-modify-write, so
    /// nothing already running is disturbed. Called from start() before the first I2C
    /// transaction and from reset() after the design has re-identified. Returns true when a
    /// held bit was actually released (the caller then waits SENSOR_RESET_SETTLE_MS).
    bool release_sensor_resets();
    /**
     * One live identity probe: (re)open the endpoint if needed, read HSB_VER and the four
     * UUID registers straight off BAR0. Returns true only when every read completed and
     * HSB_VER/UUID look like a running design (not 0, not all-ones).
     */
    bool probe_live_identity(uint32_t& hsb_version, uint8_t uuid[UUID_BYTES]);

    std::shared_ptr<PcieEndpointInterface> endpoint_;
    std::shared_ptr<AddressPolicy> policy_;
    std::unique_ptr<ApbSequencer> sequencer_;
    EcbBridge bridge_;
    std::vector<uint8_t> pending_reply_;

    bool identity_known_ { false };
    std::string expected_bdf_;
    uint8_t expected_uuid_[UUID_BYTES] { 0 };
    double reset_identity_timeout_s_ { RESET_IDENTITY_TIMEOUT_S };
};

} // namespace hololink::module::hsb_pcie

#endif /* HOLOLINK_MODULE_HSB_PCIE_PCIE_HOLOLINK_HPP */
