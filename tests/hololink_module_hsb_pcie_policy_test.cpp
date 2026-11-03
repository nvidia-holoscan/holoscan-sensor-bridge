/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Hardware-free tests for the hsb_pcie module core below PcieHololink: register accesses
 * are dispatched through AddressPolicy onto a LoopbackPcieEndpoint and the BAR0 / shadow
 * side effects are asserted (forwarding, alias rebase, shadowing, rejects, the CLK_CNT
 * hook). The software APB sequencer -- the only engine for SW_EVENT programs; CTRL_EVT and
 * APB RAM are always shadowed -- is exercised with hand-encoded programs in the
 * Hololink::Sequencer word layout. Batch-level semantics (first failure stops the batch,
 * the trigger fires once after the whole batch) are covered through the public Hololink
 * API in hololink_module_hsb_pcie_hololink_test.cpp.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <ios>
#include <stdexcept>
#include <utility>
#include <vector>

#include <hololink/core/hololink.hpp>

#include "address_policy.hpp"
#include "apb_sequencer.hpp"
#include "hololink_module_hsb_pcie_test_support.hpp"
#include "loopback_pcie_endpoint.hpp"

namespace {

using namespace hololink::module::hsb_pcie;
using hsb_pcie_test::Pairs;
using hsb_pcie_test::Program;

class HsbPciePolicy : public ::testing::Test {
protected:
    HsbPciePolicy()
        : endpoint(0, "0000:05:00.0")
        , policy()
    {
        EXPECT_TRUE(endpoint.open());
    }

    /// Dispatch a batch the way PcieHololink::write_uint32(WriteData&) does: pairs in order,
    /// stop at the first failure, fire a pending sequencer trigger once afterwards. Returns
    /// the first failing Result, or success.
    Result apply(const Pairs& pairs)
    {
        Result result { true, hololink::RESPONSE_SUCCESS };
        for (const auto& [address, value] : pairs) {
            result = policy.dispatch_write(address, value, endpoint);
            if (!result.ok) {
                break;
            }
        }
        policy.fire_pending_sequencer_trigger(endpoint);
        return result;
    }

    Result write(uint32_t address, uint32_t value) { return apply({ { address, value } }); }

    /// One dispatched read; the Result is reported through `ok` when the caller cares.
    uint32_t read(uint32_t address, Result* ok = nullptr)
    {
        uint32_t value = 0;
        Result result = policy.dispatch_read(address, endpoint, value);
        if (ok) {
            *ok = result;
        } else {
            EXPECT_TRUE(result.ok) << std::hex << address;
        }
        return value;
    }

    LoopbackPcieEndpoint endpoint;
    AddressPolicy policy;
};

} // namespace

TEST_F(HsbPciePolicy, LoopbackIdentity)
{
    const auto& id = endpoint.identity();
    EXPECT_EQ(id.hsb_version, 0x2606u);
    EXPECT_EQ(id.hsb_date, 20260219u);
    EXPECT_EQ(id.hsb_status, 0x484C5043u);
    EXPECT_EQ(id.flags, IDENTITY_VERSION_VALID | IDENTITY_MAGIC_VALID | IDENTITY_UUID_VALID);
    EXPECT_EQ(id.bdf, "0000:05:00.0");
    EXPECT_EQ(id.requester_id, 0x0500u);
    EXPECT_EQ(id.bar0_len, 0x100000u);
    for (size_t i = 0; i < UUID_BYTES; i++) {
        EXPECT_EQ(id.uuid[i], HSB_PCIE_MODULE_UUID[i]) << "byte " << i;
    }
    // MSW-first register order per msix_pcie.hpp UUID_REGS.
    EXPECT_EQ(endpoint.peek(0xB8), 0x58c1419bu);
    EXPECT_EQ(endpoint.peek(0xB4), 0x1e7a4120u);
    EXPECT_EQ(endpoint.peek(0xB0), 0x95fd4efcu);
    EXPECT_EQ(endpoint.peek(0xAC), 0x156187c1u);
}

TEST_F(HsbPciePolicy, LoopbackRejectsBadOffsetsAndGatesOnOpen)
{
    uint32_t value = 0;
    EXPECT_EQ(endpoint.bar_read32(0x2, value), PcieEndpointInterface::Status::INVALID_ADDRESS);
    EXPECT_EQ(endpoint.bar_read32(0x100000, value), PcieEndpointInterface::Status::INVALID_ADDRESS);
    EXPECT_EQ(endpoint.bar_write32(0xFFFFC, 1), PcieEndpointInterface::Status::SUCCESS);
    // I2C START -> DONE model.
    EXPECT_EQ(endpoint.bar_write32(0x40200, 1), PcieEndpointInterface::Status::SUCCESS);
    EXPECT_EQ(endpoint.peek(0x40280), 0x10u);
    EXPECT_EQ(endpoint.bar_write32(0x40200, 0), PcieEndpointInterface::Status::SUCCESS);
    EXPECT_EQ(endpoint.peek(0x40280), 0u);
    endpoint.close();
    EXPECT_EQ(endpoint.bar_read32(0x80, value), PcieEndpointInterface::Status::NOT_ATTACHED);
    // A closed endpoint surfaces as a general error, not an address error.
    Result r = policy.dispatch_read(0x80, endpoint, value);
    EXPECT_FALSE(r.ok);
    EXPECT_EQ(r.response_code, hololink::RESPONSE_ERROR_GENERAL);
}

TEST_F(HsbPciePolicy, IdentityForward)
{
    EXPECT_EQ(read(0x80), 0x2606u);
    EXPECT_TRUE(write(0x1000, 0xDEADBEEF).ok);
    EXPECT_EQ(endpoint.peek(0x1000), 0xDEADBEEFu);
    EXPECT_EQ(read(0x1000), 0xDEADBEEFu);
}

TEST_F(HsbPciePolicy, SysRstAtOffset4IsForwarded)
{
    EXPECT_TRUE(write(0x0004, 0x8).ok);
    EXPECT_EQ(endpoint.peek(0x4), 0x8u);
}

TEST_F(HsbPciePolicy, I2cAliasRebase)
{
    EXPECT_TRUE(write(0x03000204, 0x1).ok);
    EXPECT_EQ(endpoint.peek(0x40204), 0x1u);
    EXPECT_FALSE(endpoint.has_register(0x03000204)); // never reached the BAR unrebased
    // Read back through the alias too.
    EXPECT_EQ(read(0x03000204), 0x1u);
}

TEST_F(HsbPciePolicy, SpiAliasRebase)
{
    EXPECT_TRUE(write(0x03000008, 0x55).ok);
    EXPECT_EQ(endpoint.peek(0x40008), 0x55u);
}

// The divider follows the APB clock the policy is built for; the default is the CertusPro-NX
// Versa's 60 MHz, which pins 75 (a 100 MHz clock would pin 125).
TEST_F(HsbPciePolicy, ClkCntFollowsApbClock)
{
    EXPECT_EQ(AddressPolicy::clk_cnt_for(400000, 60000000), 75u);
    EXPECT_EQ(AddressPolicy::clk_cnt_for(400000, 100000000), 125u);
    AddressPolicy versa(60000000);
    EXPECT_EQ(versa.apb_clk_hz(), 60000000u);
    EXPECT_EQ(versa.i2c_clk_cnt(), 75u);
    EXPECT_TRUE(versa.dispatch_write(0x0300020C, 0x19, endpoint).ok);
    EXPECT_EQ(endpoint.peek(0x4020C), 75u);
    EXPECT_EQ(AddressPolicy::DEFAULT_APB_CLK_HZ, 60000000u) << "the Versa's clock is the default";
    EXPECT_EQ(AddressPolicy().i2c_clk_cnt(), 75u);
    EXPECT_THROW(AddressPolicy(0), std::invalid_argument);
}

TEST_F(HsbPciePolicy, ClkCntHookPinsScl)
{
    EXPECT_EQ(AddressPolicy::clk_cnt_for(400000), 75u);
    EXPECT_TRUE(write(0x0300020C, 0x19).ok);
    EXPECT_EQ(endpoint.peek(0x4020C), 75u);
    // A direct BAR0 write to CLK_CNT is not rewritten.
    EXPECT_TRUE(write(0x4020C, 0x19).ok);
    EXPECT_EQ(endpoint.peek(0x4020C), 0x19u);
}

TEST_F(HsbPciePolicy, RejectRangesAndDefaultAction)
{
    EXPECT_EQ(policy.action_for(0x40000000), Action::Reject); // JESD
    EXPECT_EQ(policy.action_for(0x80000000), Action::Reject); // I2S APB
    EXPECT_EQ(policy.action_for(0xFFFFFFFC), Action::Reject);
    EXPECT_EQ(policy.action_for(0x70000000), Action::Shadow); // VSYNC
    EXPECT_EQ(policy.action_for(0x01000000), Action::Shadow); // SIF
    EXPECT_EQ(policy.action_for(0x01200000), Action::Shadow); // I2S FIFO
    EXPECT_EQ(policy.action_for(0x02000000), Action::Shadow); // HIF
    EXPECT_EQ(policy.action_for(0x00100000), Action::Shadow); // unmatched -> default
    Result r;
    read(0x00100000, &r);
    EXPECT_TRUE(r.ok);
    EXPECT_EQ(r.response_code, hololink::RESPONSE_SUCCESS);
    policy.set_default_action(Action::Reject);
    read(0x00100000, &r);
    EXPECT_FALSE(r.ok);
    EXPECT_EQ(r.response_code, hololink::RESPONSE_INVALID_ADDR);
    r = write(0x40000000, 1);
    EXPECT_FALSE(r.ok);
    EXPECT_EQ(r.response_code, hololink::RESPONSE_INVALID_ADDR);
    EXPECT_FALSE(endpoint.has_register(0x40000000));
    EXPECT_THROW(policy.set_default_action(Action::ForwardIdentity), std::invalid_argument);
}

TEST_F(HsbPciePolicy, MisalignedAddressIsInvalid)
{
    Result r;
    read(0x82, &r);
    EXPECT_FALSE(r.ok);
    EXPECT_EQ(r.response_code, hololink::RESPONSE_INVALID_ADDR);
    r = write(0x82, 0);
    EXPECT_FALSE(r.ok);
    EXPECT_EQ(r.response_code, hololink::RESPONSE_INVALID_ADDR);
}

TEST_F(HsbPciePolicy, PtpSyncStatReadsLocked)
{
    EXPECT_EQ(read(0x188), 0xFu);
    // Writes are accepted and dropped; other PTP words are shadowed.
    EXPECT_TRUE(write(0x188, 0).ok);
    EXPECT_EQ(read(0x188), 0xFu);
    EXPECT_TRUE(write(0x104, 0x77).ok);
    EXPECT_EQ(read(0x104), 0x77u);
    EXPECT_FALSE(endpoint.has_register(0x104));
    EXPECT_FALSE(endpoint.has_register(0x188));
}

TEST_F(HsbPciePolicy, HostCtrlEvtWordsAreShadowed)
{
    EXPECT_TRUE(write(0x210, 0xAABBCCDD).ok);
    EXPECT_FALSE(endpoint.has_register(0x210));
    EXPECT_EQ(read(0x210), 0xAABBCCDDu);
    EXPECT_EQ(read(0x214), 0u); // never written -> 0
    // 0x230 (EVT VLAN) is shadowed as well.
    EXPECT_TRUE(write(0x230, 0x10001).ok);
    EXPECT_FALSE(endpoint.has_register(0x230));
}

// The endpoint bitfile does not execute CTRL_BUS_EVT/EVT_RAM programs, so the CTRL_EVT APB
// words and APB RAM never reach BAR0: they are shadowed for the software sequencer and a
// write of 1 to SW_EVENT arms its (deferred) trigger.
TEST_F(HsbPciePolicy, CtrlEvtAndApbRamAreShadowed)
{
    for (uint32_t address : { 0x200u, 0x204u, 0x20Cu, 0x224u, 0x228u, 0x280u, 0x284u, 0x2000u, 0x2FFCu }) {
        EXPECT_EQ(policy.action_for(address), Action::Shadow) << std::hex << address;
    }
    EXPECT_EQ(policy.action_for(0x22C), Action::ApbSequencer);
    EXPECT_EQ(policy.action_for(0x1000), Action::ForwardIdentity);
    EXPECT_EQ(policy.action_for(0x210), Action::Shadow);

    // APB_INTERRUPT_EN / APB_TIMEOUT: shadowed, readable back, never on the endpoint.
    EXPECT_TRUE(apply({ { 0x224, 0x4 }, { 0x228, 0x3937 } }).ok);
    EXPECT_FALSE(endpoint.has_register(0x224));
    EXPECT_FALSE(endpoint.has_register(0x228));
    EXPECT_EQ(read(0x224), 0x4u);
    EXPECT_EQ(read(0x228), 0x3937u);

    // Program words land in shadow RAM, not on the loopback endpoint; unwritten words read 0.
    EXPECT_TRUE(write(0x2800, 0x29).ok);
    EXPECT_FALSE(endpoint.has_register(0x2800));
    EXPECT_EQ(policy.shadow_read(0x2800), 0x29u);
    EXPECT_EQ(read(0x2800), 0x29u);
    EXPECT_EQ(read(0x2FFC), 0u);
    EXPECT_EQ(read(0x284), 0u);

    // SW_EVENT=1 arms the trigger instead of being forwarded (no sequencer installed in
    // this fixture, so the dispatch is checked directly rather than through apply()).
    EXPECT_TRUE(policy.dispatch_write(0x22C, 1, endpoint).ok);
    EXPECT_TRUE(policy.sequencer_trigger_pending());
    EXPECT_FALSE(endpoint.has_register(0x22C));
    EXPECT_EQ(policy.shadow_read(0x22C), 1u);
    policy.clear_pending_sequencer_trigger();
    EXPECT_FALSE(policy.sequencer_trigger_pending());
}

namespace {

class HsbPcieSoftwareSequencer : public HsbPciePolicy {
protected:
    HsbPcieSoftwareSequencer()
        : sequencer(policy, endpoint)
    {
        sequencer.install();
    }

    void load(const Program& program, uint32_t location = hsb_pcie_test::PROGRAM_LOCATION)
    {
        ASSERT_TRUE(apply(program.load_pairs(location)).ok);
    }

    void trigger()
    {
        ASSERT_TRUE(apply({ { 0x22C, 1 }, { 0x22C, 0 } }).ok);
    }

    uint32_t shadow(uint32_t address) { return read(address); }

    ApbSequencer sequencer;
};

} // namespace

TEST_F(HsbPcieSoftwareSequencer, ProgramExecutesOnSwEvent)
{
    Program p;
    p.wr(0x0, 0xA5A5A5A5);
    unsigned slot = p.rd(0x80);
    p.done();
    EXPECT_EQ(p.words[0], 0x29u); // WR, RD, RD(done)
    load(p);
    EXPECT_FALSE(endpoint.has_register(0x0));
    EXPECT_FALSE(endpoint.has_register(0x2800)); // APB RAM is shadowed (the software sequencer is the only engine)

    trigger();
    EXPECT_TRUE(sequencer.last_run().ran);
    EXPECT_FALSE(sequencer.last_run().error) << sequencer.last_run().message;
    EXPECT_EQ(sequencer.last_run().commands, 2u);
    EXPECT_EQ(endpoint.peek(0x0), 0xA5A5A5A5u);
    EXPECT_EQ(shadow(0x2800 + slot * 4), 0x2606u);
    EXPECT_EQ(shadow(0x284) & ApbSequencer::FSM_ERROR_MASK, 0u);
    EXPECT_EQ(shadow(0x280) & ApbSequencer::STAT_SW_EVENT_MASK, ApbSequencer::STAT_SW_EVENT_MASK);
}

TEST_F(HsbPcieSoftwareSequencer, EmbeddedAliasReachesBar0)
{
    Program p;
    p.wr(0x03000204, 0x1); // I2C BUS_EN via the HSB alias
    p.wr(0x0300020C, 0x19); // CLK_CNT via the alias -> hook -> 75 (60 MHz APB)
    p.wr(0x03000200, 0x1); // START -> loopback sets DONE
    unsigned status = p.rd(0x03000280);
    p.done();
    load(p);
    trigger();
    EXPECT_FALSE(sequencer.last_run().error) << sequencer.last_run().message;
    EXPECT_EQ(endpoint.peek(0x40204), 0x1u);
    EXPECT_EQ(endpoint.peek(0x4020C), 75u);
    EXPECT_EQ(endpoint.peek(0x40200), 0x1u);
    EXPECT_EQ(shadow(0x2800 + status * 4), 0x10u); // I2C_STATUS_DONE
}

TEST_F(HsbPcieSoftwareSequencer, PollMatchesAndCrossesCommandWordBoundary)
{
    endpoint.poke(0x1010, 0x00000030);
    Program p;
    for (unsigned i = 0; i < 17; i++) { // > 16 ops forces a second command word
        p.wr(0x1100 + i * 4, i);
    }
    p.poll(0x1010, 0xF0, 0x30);
    unsigned slot = p.rd(0x1110);
    p.done();
    EXPECT_EQ(p.words.size(), 1u + 17 * 2 + 1 + 3 + 2 + 1u);
    load(p);
    trigger();
    EXPECT_FALSE(sequencer.last_run().error) << sequencer.last_run().message;
    EXPECT_EQ(sequencer.last_run().commands, 19u);
    EXPECT_EQ(endpoint.peek(0x1140), 16u);
    EXPECT_EQ(shadow(0x2800 + slot * 4), 4u);
}

TEST_F(HsbPcieSoftwareSequencer, PollTimesOutPerApbTimeoutAndSetsFsmError)
{
    Program p;
    p.poll(0x1010, 0x1, 0x1); // never matches: 0x1010 reads 0
    p.done();
    load(p);
    // 0x1000 units * 51.2 ns = ~210 us.
    ASSERT_TRUE(write(0x228, 0x1000).ok);
    EXPECT_NEAR(sequencer.poll_timeout_s(), 0x1000 * 51.2e-9, 1e-12);

    auto t0 = std::chrono::steady_clock::now();
    trigger();
    double elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    EXPECT_TRUE(sequencer.last_run().error);
    EXPECT_GE(elapsed_s, 0x1000 * 51.2e-9);
    EXPECT_LT(elapsed_s, 0.5);
    EXPECT_NE(shadow(0x284) & ApbSequencer::FSM_ERROR_MASK, 0u);
    EXPECT_NE(shadow(0x280) & ApbSequencer::STAT_APB_ERROR_MASK, 0u);

    // Default bound when APB_TIMEOUT was never written.
    AddressPolicy fresh;
    ApbSequencer fresh_sequencer(fresh, endpoint);
    EXPECT_DOUBLE_EQ(fresh_sequencer.poll_timeout_s(), ApbSequencer::DEFAULT_TIMEOUT_S);
}

TEST_F(HsbPcieSoftwareSequencer, DirectDispatchWriteArmsAndFireDrains)
{
    Program p;
    p.wr(0x1000, 0x55);
    p.done();
    load(p);
    // A caller dispatching outside PcieHololink's batch only arms the trigger ...
    Result w = policy.dispatch_write(0x22C, 1, endpoint);
    EXPECT_TRUE(w.ok);
    EXPECT_TRUE(policy.sequencer_trigger_pending());
    EXPECT_FALSE(sequencer.last_run().ran);
    EXPECT_FALSE(endpoint.has_register(0x1000));
    // ... and must drain it itself.
    EXPECT_TRUE(policy.fire_pending_sequencer_trigger(endpoint));
    EXPECT_FALSE(policy.sequencer_trigger_pending());
    EXPECT_TRUE(sequencer.last_run().ran);
    EXPECT_EQ(endpoint.peek(0x1000), 0x55u);
    // Nothing pending: fire is a no-op.
    EXPECT_FALSE(policy.fire_pending_sequencer_trigger(endpoint));
    // A write of 0 does not arm; clear_pending_sequencer_trigger() drops an armed trigger.
    EXPECT_TRUE(policy.dispatch_write(0x22C, 0, endpoint).ok);
    EXPECT_FALSE(policy.sequencer_trigger_pending());
    EXPECT_TRUE(policy.dispatch_write(0x22C, 1, endpoint).ok);
    EXPECT_TRUE(policy.sequencer_trigger_pending());
    policy.clear_pending_sequencer_trigger();
    EXPECT_FALSE(policy.sequencer_trigger_pending());
    EXPECT_FALSE(policy.fire_pending_sequencer_trigger(endpoint));
}

TEST_F(HsbPcieSoftwareSequencer, ProgramReArmingItselfIsNotReentered)
{
    Program p;
    p.wr(0x22C, 1); // the program pokes SW_EVENT while running
    p.wr(0x1000, 0x99);
    p.done();
    load(p);
    trigger();
    EXPECT_FALSE(sequencer.last_run().error) << sequencer.last_run().message;
    EXPECT_EQ(sequencer.last_run().commands, 2u);
    EXPECT_EQ(endpoint.peek(0x1000), 0x99u);
    EXPECT_FALSE(policy.sequencer_trigger_pending());
}

TEST_F(HsbPcieSoftwareSequencer, BadOpcodeAndUnprogrammedVectorSetFsmError)
{
    Program p;
    p.op(ApbSequencer::ERROR);
    p.words.push_back(0x1000);
    p.done();
    load(p);
    trigger();
    EXPECT_TRUE(sequencer.last_run().error);
    EXPECT_NE(shadow(0x284) & ApbSequencer::FSM_ERROR_MASK, 0u);

    // A successful run clears the error again.
    Program ok;
    ok.wr(0x1000, 1);
    ok.done();
    load(ok);
    trigger();
    EXPECT_FALSE(sequencer.last_run().error);
    EXPECT_EQ(shadow(0x284) & ApbSequencer::FSM_ERROR_MASK, 0u);

    // Unprogrammed vector.
    ASSERT_TRUE(write(ApbSequencer::SW_EVENT_VECTOR, 0).ok);
    trigger();
    EXPECT_TRUE(sequencer.last_run().error);
}
