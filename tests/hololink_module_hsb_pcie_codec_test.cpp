/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Hardware-free tests for the hsb_pcie module core: raw ECB frames, built
 * exactly as hololink::Hololink does in src/hololink/core/hololink.cpp, are
 * pushed through EcbBridge -> AddressPolicy -> LoopbackPcieEndpoint and the
 * reply bytes plus the BAR0 side effects are asserted. The software APB
 * sequencer -- the only engine for SW_EVENT programs; CTRL_EVT and APB RAM
 * are always shadowed -- is exercised with hand-encoded programs in the
 * Hololink::Sequencer word layout.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <ios>
#include <stdexcept>
#include <utility>
#include <vector>

#include <hololink/core/deserializer.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/core/serializer.hpp>

#include "address_policy.hpp"
#include "apb_sequencer.hpp"
#include "ecb_bridge.hpp"
#include "loopback_pcie_endpoint.hpp"

namespace {

using namespace hololink::module::hsb_pcie;

constexpr size_t CONTROL_PACKET_SIZE = 1472; // hololink.cpp

// ---- request builders (mirror write_uint32_/write_uint32_block_/read_uint32_/read_uint32_block_)

std::vector<uint8_t> header(uint8_t command, uint16_t sequence, hololink::core::Serializer& s)
{
    uint8_t flags = hololink::REQUEST_FLAGS_ACK_REQUEST | hololink::REQUEST_FLAGS_SEQUENCE_CHECK;
    EXPECT_TRUE(s.append_uint8(command) && s.append_uint8(flags) && s.append_uint16_be(sequence)
        && s.append_uint8(0) && s.append_uint8(0));
    return {};
}

std::vector<uint8_t> wr_dword(uint16_t sequence, uint32_t address, uint32_t value)
{
    std::vector<uint8_t> request(CONTROL_PACKET_SIZE);
    hololink::core::Serializer s(request.data(), request.size());
    header(hololink::WR_DWORD, sequence, s);
    EXPECT_TRUE(s.append_uint32_be(address) && s.append_uint32_be(value));
    request.resize(s.length());
    return request;
}

std::vector<uint8_t> wr_block(uint16_t sequence, const std::vector<std::pair<uint32_t, uint32_t>>& pairs)
{
    std::vector<uint8_t> request(CONTROL_PACKET_SIZE);
    hololink::core::Serializer s(request.data(), request.size());
    header(hololink::WR_BLOCK, sequence, s);
    for (const auto& [address, value] : pairs) {
        EXPECT_TRUE(s.append_uint32_be(address) && s.append_uint32_be(value));
    }
    request.resize(s.length());
    return request;
}

std::vector<uint8_t> rd_dword(uint16_t sequence, uint32_t address)
{
    std::vector<uint8_t> request(CONTROL_PACKET_SIZE);
    hololink::core::Serializer s(request.data(), request.size());
    header(hololink::RD_DWORD, sequence, s);
    EXPECT_TRUE(s.append_uint32_be(address));
    request.resize(s.length());
    return request;
}

std::vector<uint8_t> rd_block(uint16_t sequence, uint32_t address, uint32_t count)
{
    std::vector<uint8_t> request(CONTROL_PACKET_SIZE);
    hololink::core::Serializer s(request.data(), request.size());
    header(hololink::RD_BLOCK, sequence, s);
    for (uint32_t i = 0; i < count; i++) {
        EXPECT_TRUE(s.append_uint32_be(address + i * 4) && s.append_uint32_be(0));
    }
    request.resize(s.length());
    return request;
}

// ---- reply parser (mirrors Hololink::execute + the readers)

struct Reply {
    uint8_t cmd { 0 };
    uint8_t flags { 0 };
    uint16_t sequence { 0 };
    uint8_t response_code { 0xFF };
    uint8_t reserved { 0xFF };
    std::vector<std::pair<uint32_t, uint32_t>> pairs;
    uint16_t latched_sequence { 0 };
    size_t remaining { 0 };
};

Reply parse(const std::vector<uint8_t>& bytes, size_t expected_pairs)
{
    Reply r;
    hololink::core::Deserializer d(bytes.data(), bytes.size());
    EXPECT_TRUE(d.next_uint8(r.cmd) && d.next_uint8(r.flags) && d.next_uint16_be(r.sequence)
        && d.next_uint8(r.response_code) && d.next_uint8(r.reserved));
    for (size_t i = 0; i < expected_pairs; i++) {
        uint32_t address = 0, value = 0;
        EXPECT_TRUE(d.next_uint32_be(address) && d.next_uint32_be(value)) << "pair " << i;
        r.pairs.emplace_back(address, value);
    }
    EXPECT_TRUE(d.next_uint16_be(r.latched_sequence));
    r.remaining = d.remaining();
    return r;
}

class HsbPcieCodec : public ::testing::Test {
protected:
    HsbPcieCodec()
        : endpoint(0, "0000:05:00.0")
        , policy()
        , bridge(policy, endpoint)
    {
        EXPECT_TRUE(endpoint.open());
    }

    Reply run(const std::vector<uint8_t>& request, size_t expected_pairs)
    {
        auto reply = bridge.process(request);
        return parse(reply, expected_pairs);
    }

    LoopbackPcieEndpoint endpoint;
    AddressPolicy policy;
    EcbBridge bridge;
};

} // namespace

TEST_F(HsbPcieCodec, LoopbackIdentity)
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

TEST_F(HsbPcieCodec, LoopbackRejectsBadOffsetsAndGatesOnOpen)
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
}

TEST_F(HsbPcieCodec, RdDwordHsbVersion)
{
    Reply r = run(rd_dword(0x1234, 0x80), 1);
    EXPECT_EQ(r.cmd, hololink::RD_DWORD | 0x80);
    EXPECT_EQ(r.flags, hololink::REQUEST_FLAGS_ACK_REQUEST | hololink::REQUEST_FLAGS_SEQUENCE_CHECK);
    EXPECT_EQ(r.sequence, 0x1234);
    EXPECT_EQ(r.response_code, hololink::RESPONSE_SUCCESS);
    EXPECT_EQ(r.reserved, 0);
    ASSERT_EQ(r.pairs.size(), 1u);
    EXPECT_EQ(r.pairs[0].first, 0x80u);
    EXPECT_EQ(r.pairs[0].second, 0x2606u);
    EXPECT_EQ(r.latched_sequence, 0x1234);
    EXPECT_EQ(r.remaining, 2u); // two pad bytes
    EXPECT_EQ(bridge.latched_sequence(), 0x1234);
}

TEST_F(HsbPcieCodec, WrDwordIdentityForward)
{
    Reply r = run(wr_dword(7, 0x1000, 0xDEADBEEF), 1);
    EXPECT_EQ(r.cmd, hololink::WR_DWORD | 0x80);
    EXPECT_EQ(r.response_code, hololink::RESPONSE_SUCCESS);
    ASSERT_EQ(r.pairs.size(), 1u);
    EXPECT_EQ(r.pairs[0].first, 0x1000u);
    EXPECT_EQ(r.pairs[0].second, 0xDEADBEEFu);
    EXPECT_EQ(endpoint.peek(0x1000), 0xDEADBEEFu);
    EXPECT_EQ(r.latched_sequence, 7);
}

TEST_F(HsbPcieCodec, SysRstAtOffset4IsForwarded)
{
    Reply r = run(wr_dword(8, 0x0004, 0x8), 1);
    EXPECT_EQ(r.response_code, hololink::RESPONSE_SUCCESS);
    EXPECT_EQ(endpoint.peek(0x4), 0x8u);
}

TEST_F(HsbPcieCodec, I2cAliasRebase)
{
    Reply r = run(wr_dword(9, 0x03000204, 0x1), 1);
    EXPECT_EQ(r.response_code, hololink::RESPONSE_SUCCESS);
    EXPECT_EQ(r.pairs[0].first, 0x03000204u);
    EXPECT_EQ(endpoint.peek(0x40204), 0x1u);
    EXPECT_FALSE(endpoint.has_register(0x03000204)); // never reached the BAR unrebased

    // Read back through the alias too.
    Reply rr = run(rd_dword(10, 0x03000204), 1);
    EXPECT_EQ(rr.response_code, hololink::RESPONSE_SUCCESS);
    EXPECT_EQ(rr.pairs[0].first, 0x03000204u);
    EXPECT_EQ(rr.pairs[0].second, 0x1u);
}

TEST_F(HsbPcieCodec, SpiAliasRebase)
{
    Reply r = run(wr_dword(11, 0x03000008, 0x55), 1);
    EXPECT_EQ(r.response_code, hololink::RESPONSE_SUCCESS);
    EXPECT_EQ(endpoint.peek(0x40008), 0x55u);
}

// The divider follows the APB clock the policy is built for; the default is the CertusPro-NX
// Versa's 60 MHz, which pins 75 (a 100 MHz clock would pin 125).
TEST_F(HsbPcieCodec, ClkCntFollowsApbClock)
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

TEST_F(HsbPcieCodec, ClkCntHookPinsScl)
{
    EXPECT_EQ(AddressPolicy::clk_cnt_for(400000), 75u);
    Reply r = run(wr_dword(12, 0x0300020C, 0x19), 1);
    EXPECT_EQ(r.response_code, hololink::RESPONSE_SUCCESS);
    EXPECT_EQ(endpoint.peek(0x4020C), 75u);
    // A direct BAR0 write to CLK_CNT is not rewritten.
    run(wr_dword(13, 0x4020C, 0x19), 1);
    EXPECT_EQ(endpoint.peek(0x4020C), 0x19u);
}

TEST_F(HsbPcieCodec, RdBlockEchoesAddressesInOrder)
{
    // A forwarded BAR0 range (0x2000+ is shadow APB RAM and never reaches the endpoint).
    endpoint.poke(0x1000, 0x11);
    endpoint.poke(0x1004, 0x22);
    endpoint.poke(0x1008, 0x33);
    endpoint.poke(0x100C, 0x44);
    Reply r = run(rd_block(0x55AA, 0x1000, 4), 4);
    EXPECT_EQ(r.cmd, hololink::RD_BLOCK | 0x80);
    EXPECT_EQ(r.response_code, hololink::RESPONSE_SUCCESS);
    ASSERT_EQ(r.pairs.size(), 4u);
    for (uint32_t i = 0; i < 4; i++) {
        EXPECT_EQ(r.pairs[i].first, 0x1000u + i * 4);
    }
    EXPECT_EQ(r.pairs[0].second, 0x11u);
    EXPECT_EQ(r.pairs[1].second, 0x22u);
    EXPECT_EQ(r.pairs[2].second, 0x33u);
    EXPECT_EQ(r.pairs[3].second, 0x44u);
    EXPECT_EQ(r.latched_sequence, 0x55AA);
}

TEST_F(HsbPcieCodec, WrBlockStopsAtRejectKeepingEarlierWrites)
{
    Reply r = run(wr_block(21, { { 0x1000, 1 }, { 0x1004, 2 }, { 0x40000000, 3 }, { 0x1008, 4 } }), 4);
    EXPECT_EQ(r.cmd, hololink::WR_BLOCK | 0x80);
    EXPECT_EQ(r.response_code, hololink::RESPONSE_INVALID_ADDR);
    ASSERT_EQ(r.pairs.size(), 4u);
    EXPECT_EQ(r.pairs[2].first, 0x40000000u);
    EXPECT_EQ(endpoint.peek(0x1000), 1u);
    EXPECT_EQ(endpoint.peek(0x1004), 2u);
    EXPECT_FALSE(endpoint.has_register(0x1008)); // not applied after the failure
    EXPECT_EQ(r.latched_sequence, 21);
}

TEST_F(HsbPcieCodec, RejectRangesAndDefaultAction)
{
    EXPECT_EQ(policy.action_for(0x40000000), Action::Reject); // JESD
    EXPECT_EQ(policy.action_for(0x80000000), Action::Reject); // I2S APB
    EXPECT_EQ(policy.action_for(0xFFFFFFFC), Action::Reject);
    EXPECT_EQ(policy.action_for(0x70000000), Action::Shadow); // VSYNC
    EXPECT_EQ(policy.action_for(0x01000000), Action::Shadow); // SIF
    EXPECT_EQ(policy.action_for(0x01200000), Action::Shadow); // I2S FIFO
    EXPECT_EQ(policy.action_for(0x02000000), Action::Shadow); // HIF
    EXPECT_EQ(policy.action_for(0x00100000), Action::Shadow); // unmatched -> default
    Reply r = run(rd_dword(30, 0x00100000), 1);
    EXPECT_EQ(r.response_code, hololink::RESPONSE_SUCCESS);
    policy.set_default_action(Action::Reject);
    r = run(rd_dword(31, 0x00100000), 1);
    EXPECT_EQ(r.response_code, hololink::RESPONSE_INVALID_ADDR);
    EXPECT_THROW(policy.set_default_action(Action::ForwardIdentity), std::invalid_argument);
}

TEST_F(HsbPcieCodec, MisalignedAddressIsInvalid)
{
    Reply r = run(rd_dword(32, 0x82), 1);
    EXPECT_EQ(r.response_code, hololink::RESPONSE_INVALID_ADDR);
    EXPECT_EQ(r.pairs[0].first, 0x82u);
}

TEST_F(HsbPcieCodec, UnknownCommand)
{
    std::vector<uint8_t> request(CONTROL_PACKET_SIZE);
    hololink::core::Serializer s(request.data(), request.size());
    header(0x42, 33, s);
    EXPECT_TRUE(s.append_uint32_be(0x80));
    request.resize(s.length());
    Reply r = run(request, 0);
    EXPECT_EQ(r.cmd, 0x42 | 0x80);
    EXPECT_EQ(r.sequence, 33);
    EXPECT_EQ(r.response_code, hololink::RESPONSE_INVALID_CMD);
    EXPECT_EQ(r.latched_sequence, 33);
    EXPECT_TRUE(bridge.process({}).empty());
}

TEST_F(HsbPcieCodec, PtpSyncStatReadsLocked)
{
    Reply r = run(rd_dword(40, 0x188), 1);
    EXPECT_EQ(r.response_code, hololink::RESPONSE_SUCCESS);
    EXPECT_EQ(r.pairs[0].second, 0xFu);
    // Writes are accepted and dropped; other PTP words are shadowed.
    run(wr_dword(41, 0x188, 0), 1);
    EXPECT_EQ(run(rd_dword(42, 0x188), 1).pairs[0].second, 0xFu);
    run(wr_dword(43, 0x104, 0x77), 1);
    EXPECT_EQ(run(rd_dword(44, 0x104), 1).pairs[0].second, 0x77u);
    EXPECT_FALSE(endpoint.has_register(0x104));
    EXPECT_FALSE(endpoint.has_register(0x188));
}

TEST_F(HsbPcieCodec, HostCtrlEvtWordsAreShadowed)
{
    Reply r = run(wr_dword(50, 0x210, 0xAABBCCDD), 1);
    EXPECT_EQ(r.response_code, hololink::RESPONSE_SUCCESS);
    EXPECT_FALSE(endpoint.has_register(0x210));
    EXPECT_EQ(run(rd_dword(51, 0x210), 1).pairs[0].second, 0xAABBCCDDu);
    EXPECT_EQ(run(rd_dword(52, 0x214), 1).pairs[0].second, 0u); // never written -> 0
    // 0x230 (EVT VLAN) is shadowed as well.
    run(wr_dword(53, 0x230, 0x10001), 1);
    EXPECT_FALSE(endpoint.has_register(0x230));
}

// The endpoint bitfile does not execute CTRL_BUS_EVT/EVT_RAM programs, so the CTRL_EVT APB
// words and APB RAM never reach BAR0: they are shadowed for the software sequencer and a
// write of 1 to SW_EVENT arms its (deferred) trigger.
TEST_F(HsbPcieCodec, CtrlEvtAndApbRamAreShadowed)
{
    for (uint32_t address : { 0x200u, 0x204u, 0x20Cu, 0x224u, 0x228u, 0x280u, 0x284u, 0x2000u, 0x2FFCu }) {
        EXPECT_EQ(policy.action_for(address), Action::Shadow) << std::hex << address;
    }
    EXPECT_EQ(policy.action_for(0x22C), Action::ApbSequencer);
    EXPECT_EQ(policy.action_for(0x1000), Action::ForwardIdentity);
    EXPECT_EQ(policy.action_for(0x210), Action::Shadow);

    // APB_INTERRUPT_EN / APB_TIMEOUT: shadowed, readable back, never on the endpoint.
    Reply r = run(wr_block(60, { { 0x224, 0x4 }, { 0x228, 0x3937 } }), 2);
    EXPECT_EQ(r.response_code, hololink::RESPONSE_SUCCESS);
    EXPECT_FALSE(endpoint.has_register(0x224));
    EXPECT_FALSE(endpoint.has_register(0x228));
    EXPECT_EQ(run(rd_dword(61, 0x224), 1).pairs[0].second, 0x4u);
    EXPECT_EQ(run(rd_dword(62, 0x228), 1).pairs[0].second, 0x3937u);

    // Program words land in shadow RAM, not on the loopback endpoint; unwritten words read 0.
    run(wr_dword(63, 0x2800, 0x29), 1);
    EXPECT_FALSE(endpoint.has_register(0x2800));
    EXPECT_EQ(policy.shadow_read(0x2800), 0x29u);
    EXPECT_EQ(run(rd_dword(64, 0x2800), 1).pairs[0].second, 0x29u);
    EXPECT_EQ(run(rd_dword(65, 0x2FFC), 1).pairs[0].second, 0u);
    EXPECT_EQ(run(rd_dword(66, 0x284), 1).pairs[0].second, 0u);

    // SW_EVENT=1 arms the trigger instead of being forwarded (no sequencer installed in
    // this fixture, so the dispatch is checked directly rather than through the bridge).
    EXPECT_TRUE(policy.dispatch_write(0x22C, 1, endpoint).ok);
    EXPECT_TRUE(policy.sequencer_trigger_pending());
    EXPECT_FALSE(endpoint.has_register(0x22C));
    EXPECT_EQ(policy.shadow_read(0x22C), 1u);
    policy.clear_pending_sequencer_trigger();
    EXPECT_FALSE(policy.sequencer_trigger_pending());
}

namespace {

// Hand-encode a program in the Hololink::Sequencer layout: command word first, then per
// command ADDR [DATA [DATA]]; 2 bits per op, LSB first.
struct Program {
    std::vector<uint32_t> words { 0 };
    unsigned command_index { 0 };
    unsigned command_bit { 0 };

    void op(uint32_t code)
    {
        if (command_bit == 32) {
            command_index = words.size();
            words.push_back(0);
            command_bit = 0;
        }
        words[command_index] |= (code << command_bit);
        command_bit += 2;
    }
    void wr(uint32_t address, uint32_t value)
    {
        op(ApbSequencer::WR);
        words.push_back(address);
        words.push_back(value);
    }
    unsigned rd(uint32_t address, uint32_t initial = 0)
    {
        op(ApbSequencer::RD);
        words.push_back(address);
        words.push_back(initial);
        return words.size() - 1;
    }
    void poll(uint32_t address, uint32_t mask, uint32_t match)
    {
        op(ApbSequencer::POLL);
        words.push_back(address);
        words.push_back(match);
        words.push_back(mask);
    }
    void done()
    {
        op(ApbSequencer::RD);
        words.push_back(0xFFFFFFFF);
    }
};

class HsbPcieSoftwareSequencer : public HsbPcieCodec {
protected:
    HsbPcieSoftwareSequencer()
        : sequencer(policy, endpoint)
    {
        sequencer.install();
    }

    void load(const Program& program, uint32_t location = 0x2800)
    {
        std::vector<std::pair<uint32_t, uint32_t>> pairs;
        uint32_t address = location;
        for (uint32_t word : program.words) {
            pairs.emplace_back(address, word);
            address += 4;
        }
        // Vector table slot for SW_EVENT (configure_apb_event) + interrupt enable.
        pairs.emplace_back(ApbSequencer::SW_EVENT_VECTOR, location);
        pairs.emplace_back(0x224, 1u << ApbSequencer::SW_EVENT);
        Reply r = run(wr_block(100, pairs), pairs.size());
        ASSERT_EQ(r.response_code, hololink::RESPONSE_SUCCESS);
    }

    void trigger()
    {
        Reply r = run(wr_block(101, { { 0x22C, 1 }, { 0x22C, 0 } }), 2);
        ASSERT_EQ(r.response_code, hololink::RESPONSE_SUCCESS);
    }

    uint32_t shadow(uint32_t address) { return run(rd_dword(102, address), 1).pairs[0].second; }

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
    p.wr(0x0300020C, 0x19); // CLK_CNT via the alias -> hook -> 125
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
    ASSERT_EQ(run(wr_dword(103, 0x228, 0x1000), 1).response_code, hololink::RESPONSE_SUCCESS);
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

// I2c::i2c_transaction (hololink.cpp) queues SW_EVENT=1, SW_EVENT=0 and only then
// CTRL_EVT_APB_TIMEOUT in one WR_BLOCK. The run must be bounded by that later value, not by
// whatever APB_TIMEOUT held before the request -- i.e. the sequencer runs after the whole
// pair loop, not synchronously on the SW_EVENT pair.
TEST_F(HsbPcieSoftwareSequencer, ApbTimeoutQueuedAfterSwEventBoundsThisRun)
{
    Program p;
    p.poll(0x1010, 0x1, 0x1); // never matches: 0x1010 reads 0
    p.done();
    load(p);
    // Stale bound from an earlier transaction: 5,000,000 units = 256 ms.
    constexpr uint32_t stale_units = 5000000u;
    constexpr uint32_t this_units = 0x1000u; // ~210 us
    ASSERT_EQ(run(wr_dword(110, 0x228, stale_units), 1).response_code, hololink::RESPONSE_SUCCESS);
    ASSERT_NEAR(sequencer.poll_timeout_s(), stale_units * 51.2e-9, 1e-12);
    EXPECT_FALSE(policy.sequencer_trigger_pending());

    auto t0 = std::chrono::steady_clock::now();
    Reply r = run(wr_block(111, { { 0x22C, 1 }, { 0x22C, 0 }, { 0x228, this_units } }), 3);
    double elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    ASSERT_EQ(r.response_code, hololink::RESPONSE_SUCCESS);
    ASSERT_EQ(r.pairs.size(), 3u);
    EXPECT_EQ(r.pairs[2].first, 0x228u);
    EXPECT_EQ(r.pairs[2].second, this_units);

    // The trigger was consumed by the request and the run used the timeout from the same block.
    EXPECT_FALSE(policy.sequencer_trigger_pending());
    EXPECT_TRUE(sequencer.last_run().ran);
    EXPECT_TRUE(sequencer.last_run().error) << "POLL on a zero register must time out";
    EXPECT_EQ(sequencer.last_run().polls, 1u);
    EXPECT_NEAR(sequencer.last_run().poll_timeout_s, this_units * 51.2e-9, 1e-12);
    EXPECT_NEAR(sequencer.poll_timeout_s(), this_units * 51.2e-9, 1e-12);
    EXPECT_GE(elapsed_s, this_units * 51.2e-9);
    EXPECT_LT(elapsed_s, stale_units * 51.2e-9) << "ran with the stale bound";
    // SW_EVENT=0 landed before the run, as on the FPGA.
    EXPECT_EQ(shadow(0x22C), 0u);
    EXPECT_NE(shadow(0x284) & ApbSequencer::FSM_ERROR_MASK, 0u);
}

TEST_F(HsbPcieSoftwareSequencer, TriggerRunsAfterPairLoopEvenWhenALaterPairFails)
{
    Program p;
    p.wr(0x1000, 0x77);
    p.done();
    load(p);
    // The FPGA would already have latched SW_EVENT=1 when the rejected pair arrives.
    Reply r = run(wr_block(112, { { 0x22C, 1 }, { 0x22C, 0 }, { 0x40000000, 3 }, { 0x1004, 4 } }), 4);
    EXPECT_EQ(r.response_code, hololink::RESPONSE_INVALID_ADDR);
    EXPECT_FALSE(policy.sequencer_trigger_pending());
    EXPECT_TRUE(sequencer.last_run().ran);
    EXPECT_FALSE(sequencer.last_run().error) << sequencer.last_run().message;
    EXPECT_EQ(endpoint.peek(0x1000), 0x77u);
    EXPECT_FALSE(endpoint.has_register(0x1004));
}

// Two SW_EVENT=1 writes in one WR_BLOCK (the FPGA would see a single latched pulse) must
// run the program exactly once. The program first captures 0x1000 into a shadow slot and
// then overwrites 0x1000: a second run would capture the overwritten value instead.
TEST_F(HsbPcieSoftwareSequencer, RepeatedSwEventInOneBlockRunsOnce)
{
    endpoint.poke(0x1000, 0x11);
    Program p;
    unsigned slot = p.rd(0x1000);
    p.wr(0x1000, 0xC0FFEE);
    p.done();
    load(p);

    Reply r = run(wr_block(113, { { 0x22C, 1 }, { 0x22C, 1 }, { 0x22C, 0 } }), 3);
    EXPECT_EQ(r.response_code, hololink::RESPONSE_SUCCESS);
    EXPECT_FALSE(policy.sequencer_trigger_pending());
    EXPECT_TRUE(sequencer.last_run().ran);
    EXPECT_FALSE(sequencer.last_run().error) << sequencer.last_run().message;
    EXPECT_EQ(sequencer.last_run().commands, 2u);
    EXPECT_EQ(endpoint.peek(0x1000), 0xC0FFEEu);
    // Captured before the program's own write: proof the program ran only once.
    EXPECT_EQ(shadow(0x2800 + slot * 4), 0x11u);
    EXPECT_EQ(shadow(0x22C), 0u);
}

TEST_F(HsbPcieSoftwareSequencer, DirectDispatchWriteArmsAndFireDrains)
{
    Program p;
    p.wr(0x1000, 0x55);
    p.done();
    load(p);
    // A caller bypassing EcbBridge only arms the trigger ...
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
    ASSERT_EQ(run(wr_dword(104, ApbSequencer::SW_EVENT_VECTOR, 0), 1).response_code, hololink::RESPONSE_SUCCESS);
    trigger();
    EXPECT_TRUE(sequencer.last_run().error);
}
