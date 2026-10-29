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
 * The legacy Hololink funnels every register access through three virtual seams --
 * write_uint32(WriteData&, ...), the scalar read_uint32(address, ...) and the block
 * read_uint32(address, count, ...) (src/hololink/core/hololink.hpp). The UDP base bodies turn
 * those into ECB frames; this subclass overrides them instead and dispatches each
 * (address[, value]) pair through the AddressPolicy straight onto the endpoint's BAR0 (or
 * the policy's shadow words). No frame is built, no control socket, no async-event socket,
 * no listener thread. send_control()/receive_control() are overridden to throw, so a core
 * path that ever bypassed the seam fails loudly instead of trying UDP on a BDF.
 *
 * It rides under module_core::LegacyHololinkAccess so the module_core siblings (I2cV1,
 * SequencerV1, DataChannelV1, PtpPpsOutputV1) reach it through HololinkV1::legacy_access()
 * unchanged.
 */

#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "hololink_default.hpp"

#include "address_policy.hpp"
#include "apb_sequencer.hpp"
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
     * @param policy the HSB-address -> BAR0 policy every register access is dispatched through
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
     * Idempotent. Opens the endpoint, programs the I2C block's timing (CLK_CNT per the
     * policy's APB clock for 400 kHz, TIMEOUT=5000000), releases the sensor resets, runs
     * configure_hsb() and marks the device started. Opens no sockets and starts no threads.
     * On failure the device is left not-started and an endpoint this call opened is closed
     * again before the exception propagates.
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

    // ---- the register seam -----------------------------------------------------------------
    // The overrides below hide the same-named inline overloads of the base; re-expose them so
    // write_uint32(address, value) / read_uint32(address) keep resolving on a PcieHololink.
    using Hololink::read_uint32;
    using Hololink::write_uint32;

    /**
     * Apply a WriteData batch to the design: the pairs are written in order, as one unit with
     * respect to every other register access made through this object. On the first pair that
     * fails, the earlier pairs stay applied, that pair and the later ones are not applied, and
     * the failure is reported through bad_write_response() (TransactionError by default;
     * RESPONSE_INVALID_ADDR for an address the design does not accept, RESPONSE_ERROR_GENERAL
     * when the access itself failed). Any deferred effect a write in the batch arms takes
     * place once the whole batch has been applied, also when a later pair failed. A
     * misaligned address throws std::runtime_error before anything is applied. An empty batch
     * is a no-op returning true.
     *
     * in_timeout, retry and sequence_check are ignored: every access completes synchronously
     * with a status.
     */
    bool write_uint32(WriteData& data, std::shared_ptr<Timeout> in_timeout, bool retry,
        bool sequence_check) override;
    /// Read one register; a failure is reported through bad_read_response() (TransactionError
    /// by default). A misaligned address throws std::runtime_error.
    uint32_t read_uint32(uint32_t address, const std::shared_ptr<Timeout>& in_timeout,
        bool sequence_check) override;
    /// Read `count` consecutive words from `address`; on the first failure the result is what
    /// bad_read_response() returns (it throws TransactionError by default).
    std::tuple<bool, std::vector<uint32_t>> read_uint32(uint32_t address, uint32_t count,
        const std::shared_ptr<Timeout>& in_timeout, bool sequence_check = true) override;

    /// Not a transport on PCIe: throws std::logic_error. Every register access must arrive
    /// through the read_uint32 / write_uint32 overrides above.
    void send_control(const std::vector<uint8_t>& request) override;
    /// Not a transport on PCIe: throws std::logic_error (see send_control()).
    std::vector<uint8_t> receive_control(const std::shared_ptr<Timeout>& timeout) override;

    // ---- accessors (tests, diagnostics) ---------------------------------------------------
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
    /// the check runs before ApbSequencer binds references to *p.
    template <typename T>
    static std::shared_ptr<T> require(std::shared_ptr<T> p, const char* what)
    {
        if (!p) {
            throw std::runtime_error(std::string("PcieHololink: ") + what + " must be non-null");
        }
        return p;
    }

    /// The base throws std::runtime_error for an address that is not a multiple of four
    /// before any transport call; keep that contract on this transport.
    static void check_aligned(uint32_t address);
    /// Text for the RESPONSE_* codes the policy produces (mirrors core's private table).
    static const char* describe(uint32_t response_code);

    /// Snapshot the endpoint's BDF and UUID as the reference reset() compares against.
    void remember_identity();
    /// Program the I2C block's timing on BAR0 through the seam (no HSB equivalent for TIMEOUT).
    void program_i2c_timing();
    /// Take every sensor out of reset (HLNK_CTRL_1 SW_SEN_RST bits set): on a design fresh
    /// from configuration, or after SYS_RST returned the word to 0. A read-modify-write, so
    /// nothing already running is disturbed. Called from start() before the first I2C
    /// transaction and from reset() after the design has re-identified. Returns true when a
    /// held bit was actually released (the caller then waits SENSOR_RESET_SETTLE_MS).
    bool release_sensor_resets();
    /**
     * One live identity probe: (re)open the endpoint if needed, read HSB_VER and the four
     * UUID registers straight off BAR0 (not through the seam: a failed read must report
     * false, not throw, while reset() polls). Returns true only when every read completed
     * and HSB_VER/UUID look like a running design (not 0, not all-ones).
     */
    bool probe_live_identity(uint32_t& hsb_version, uint8_t uuid[UUID_BYTES]);

    std::shared_ptr<PcieEndpointInterface> endpoint_;
    std::shared_ptr<AddressPolicy> policy_;
    std::unique_ptr<ApbSequencer> sequencer_;
    /**
     * Serializes every BAR0 / shadow access and the sequencer run a batch arms. Takes over
     * the role Hololink::execute_mutex_ plays for the UDP transport: AddressPolicy and its
     * pending-trigger flag are not internally synchronized. Held for a whole WriteData batch
     * (trigger included) and for stop()'s close; never held across configure_hsb(),
     * post_reset_configuration() or trigger_reset(), which call back into the overrides.
     * The ApbSequencer reaches the endpoint through the policy, never through these
     * overrides, so a plain (non-recursive) mutex is correct.
     */
    std::mutex access_mutex_;

    bool identity_known_ { false };
    std::string expected_bdf_;
    uint8_t expected_uuid_[UUID_BYTES] { 0 };
    double reset_identity_timeout_s_ { RESET_IDENTITY_TIMEOUT_S };
};

} // namespace hololink::module::hsb_pcie

#endif /* HOLOLINK_MODULE_HSB_PCIE_PCIE_HOLOLINK_HPP */
