/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Hardware-free tests for PcieHololink (hololink_module/module/hsb_pcie/pcie_hololink.*):
 * the legacy hololink::Hololink carried over an in-memory LoopbackPcieEndpoint through the
 * ECB bridge + address policy. Verifies that start() opens no sockets, that configure_hsb()
 * lands where the policy says (CTRL_EVT / APB RAM shadowed for the software sequencer, PTP
 * and host-address words shadowed, I2C timing forwarded), that scalar / batched / block
 * register access and and_uint32 / or_uint32 work
 * unchanged, and that reset() really waits for live BAR0 identity reads (HSB_VER + UUID)
 * to come back after SYS_RST before reconfiguring -- including the case where the design is
 * temporarily unavailable and the case where it never returns.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <hololink/core/hololink.hpp>
#include <hololink/module/pcie/pcie_regs.hpp>

#include "address_policy.hpp"
#include "loopback_pcie_endpoint.hpp"
#include "pcie_hololink.hpp"
#include "pcie_hololink_v1.hpp"

namespace {

using namespace hololink::module::hsb_pcie;

constexpr const char* PEER_KEY = "/dev/hololink_pcie0";
constexpr const char* BDF = "0000:05:00.0";
constexpr const char* SERIAL = "pcie-0000:05:00.0";

// Register addresses from src/hololink/core/hololink.hpp / the address policy table.
constexpr uint32_t SW_SEN_RST = 0x08;
constexpr uint32_t SYS_RST = 0x04;
constexpr uint32_t PTP_CTRL = hololink::FPGA_PTP_CTRL; // 0x104
constexpr uint32_t PTP_DPLL_CFG1 = hololink::FPGA_PTP_CTRL_DPLL_CFG1; // 0x110
constexpr uint32_t CTRL_EVT_HOST_MAC_LO = hololink::CTRL_EVT_HOST_MAC_ADDR_LO; // 0x210
constexpr uint32_t CTRL_EVT_APB_INT_EN = hololink::CTRL_EVT_APB_INTERRUPT_EN; // 0x224
constexpr uint32_t CTRL_EVT_APB_TIMEOUT = hololink::CTRL_EVT_APB_TIMEOUT; // 0x228
constexpr uint32_t CTRL_EVT_SW_EVENT = hololink::CTRL_EVT_SW_EVENT; // 0x22C
constexpr uint32_t NULL_SEQUENCE = hololink::APB_RAM + 0x80; // 0x2080
constexpr uint32_t APB_TIMEOUT_750US = 0x3937;
constexpr uint32_t I2C_CLK_CNT_BAR0 = 0x4020C;
constexpr uint32_t I2C_TIMEOUT_BAR0 = 0x40210;
constexpr uint32_t HSB_VER = 0x80;
constexpr uint32_t DEAD_WORD = 0xFFFFFFFFu; // what a BAR read returns while the design is in reset

size_t open_fd_count()
{
    size_t count = 0;
    for ([[maybe_unused]] const auto& entry : std::filesystem::directory_iterator("/proc/self/fd")) {
        count++;
    }
    return count;
}

class CountingResetController : public hololink::Hololink::ResetController {
public:
    void reset() override { resets++; }
    unsigned resets { 0 };
};

// Delegates to a LoopbackPcieEndpoint but fails every write to one BAR0 offset, so start()
// can be made to throw part-way through (after it opened the endpoint).
class FailingWriteEndpoint : public PcieEndpointInterface {
public:
    FailingWriteEndpoint(std::shared_ptr<LoopbackPcieEndpoint> inner, uint32_t failing_offset)
        : inner_(std::move(inner))
        , failing_offset_(failing_offset)
    {
    }
    bool open() override { return inner_->open(); }
    void close() override { inner_->close(); }
    bool is_open() const override { return inner_->is_open(); }
    uint32_t minor() const override { return inner_->minor(); }
    const std::string& bdf() const override { return inner_->bdf(); }
    uint16_t requester_id() const override { return inner_->requester_id(); }
    PcieIdentity identity() const override { return inner_->identity(); }
    Status bar_read32(uint32_t offset, uint32_t& value) override { return inner_->bar_read32(offset, value); }
    Status bar_write32(uint32_t offset, uint32_t value) override
    {
        if (offset == failing_offset_) {
            return Status::WRITE_FAILED;
        }
        return inner_->bar_write32(offset, value);
    }

private:
    std::shared_ptr<LoopbackPcieEndpoint> inner_;
    uint32_t failing_offset_;
};

// Make the loopback look like a design that is mid-reset: HSB_VER and the UUID words read
// all-ones. Returns the values needed to bring it back.
struct SavedIdentity {
    uint32_t hsb_ver;
    uint32_t uuid[4];
};

SavedIdentity knock_out_identity(LoopbackPcieEndpoint& endpoint)
{
    SavedIdentity saved { endpoint.peek(HSB_VER), {} };
    endpoint.poke(HSB_VER, DEAD_WORD);
    for (size_t i = 0; i < 4; i++) {
        saved.uuid[i] = endpoint.peek(UUID_REGS[i]);
        endpoint.poke(UUID_REGS[i], DEAD_WORD);
    }
    return saved;
}

void restore_identity(LoopbackPcieEndpoint& endpoint, const SavedIdentity& saved)
{
    for (size_t i = 0; i < 4; i++) {
        endpoint.poke(UUID_REGS[i], saved.uuid[i]);
    }
    endpoint.poke(HSB_VER, saved.hsb_ver);
}

class HsbPcieHololink : public ::testing::Test {
protected:
    void SetUp() override
    {
        make();
    }

    void TearDown() override
    {
        if (hololink) {
            hololink->stop();
        }
    }

    void make()
    {
        endpoint = std::make_shared<LoopbackPcieEndpoint>(0, BDF);
        policy = std::make_shared<AddressPolicy>();
        hololink = std::make_shared<PcieHololink>(endpoint, policy, PEER_KEY, SERIAL);
    }

    std::shared_ptr<LoopbackPcieEndpoint> endpoint;
    std::shared_ptr<AddressPolicy> policy;
    std::shared_ptr<PcieHololink> hololink;
};

} // namespace

TEST_F(HsbPcieHololink, ConstructionCapturesIdentityWithoutIo)
{
    EXPECT_FALSE(endpoint->is_open());
    EXPECT_TRUE(hololink->identity_known());
    EXPECT_EQ(hololink->expected_bdf(), BDF);
    for (size_t i = 0; i < UUID_BYTES; i++) {
        EXPECT_EQ(hololink->expected_uuid()[i], HSB_PCIE_MODULE_UUID[i]) << "byte " << i;
    }
    // Only the seeded identity block exists so far.
    EXPECT_FALSE(endpoint->has_register(CTRL_EVT_APB_TIMEOUT));
}

TEST_F(HsbPcieHololink, StartOpensNoSocketsAndConfiguresHsb)
{
    const size_t fds_before = open_fd_count();
    ASSERT_NO_THROW(hololink->start());
    EXPECT_EQ(open_fd_count(), fds_before) << "start() must not open sockets or spawn listeners";
    EXPECT_TRUE(endpoint->is_open());

    // I2C timing programmed straight onto BAR0.
    EXPECT_EQ(endpoint->peek(I2C_CLK_CNT_BAR0), 75u);
    EXPECT_EQ(endpoint->peek(I2C_TIMEOUT_BAR0), 5000000u);

    // configure_hsb(): CTRL_EVT APB words are shadowed for the software sequencer, never
    // forwarded (the endpoint bitfile does not execute sequencer programs) ...
    EXPECT_TRUE(policy->shadow_contains(CTRL_EVT_APB_INT_EN));
    EXPECT_FALSE(endpoint->has_register(CTRL_EVT_APB_INT_EN));
    EXPECT_EQ(policy->shadow_read(CTRL_EVT_APB_TIMEOUT), APB_TIMEOUT_750US);
    EXPECT_FALSE(endpoint->has_register(CTRL_EVT_APB_TIMEOUT));
    EXPECT_TRUE(policy->shadow_contains(CTRL_EVT_SW_EVENT));
    EXPECT_EQ(policy->shadow_read(CTRL_EVT_SW_EVENT), 0u);
    EXPECT_FALSE(endpoint->has_register(CTRL_EVT_SW_EVENT));
    EXPECT_FALSE(policy->sequencer_trigger_pending());
    // ... the null sequence lands in shadow APB RAM ...
    EXPECT_TRUE(policy->shadow_contains(NULL_SEQUENCE));
    EXPECT_FALSE(endpoint->has_register(NULL_SEQUENCE));
    // ... PTP and the hololink-only host-address words are shadowed, never forwarded.
    EXPECT_TRUE(policy->shadow_contains(PTP_CTRL));
    EXPECT_TRUE(policy->shadow_contains(PTP_DPLL_CFG1));
    EXPECT_FALSE(endpoint->has_register(PTP_CTRL));
    EXPECT_FALSE(endpoint->has_register(PTP_DPLL_CFG1));
    EXPECT_TRUE(policy->shadow_contains(CTRL_EVT_HOST_MAC_LO));
    EXPECT_FALSE(endpoint->has_register(CTRL_EVT_HOST_MAC_LO));

    // Idempotent.
    const size_t registers = endpoint->register_count();
    ASSERT_NO_THROW(hololink->start());
    EXPECT_EQ(endpoint->register_count(), registers);
}

TEST_F(HsbPcieHololink, ScalarBatchedAndBlockAccess)
{
    hololink->start();

    // Scalar write/read (WR_DWORD / RD_DWORD) onto a forwarded BAR0 offset.
    EXPECT_TRUE(hololink->write_uint32(0x1000, 0xA5A5A5A5u));
    EXPECT_EQ(endpoint->peek(0x1000), 0xA5A5A5A5u);
    EXPECT_EQ(hololink->read_uint32(0x1000), 0xA5A5A5A5u);

    // Batched writes (WR_BLOCK) -- the path HololinkV1::write_uint32 uses.
    hololink::Hololink::WriteData batch;
    batch.queue_write_uint32(0x1040, 0x11111111u);
    batch.queue_write_uint32(0x1044, 0x22222222u);
    batch.queue_write_uint32(0x03000204, 0x1u); // I2C alias -> BAR0 0x40204
    EXPECT_TRUE(hololink->write_uint32(batch));
    EXPECT_EQ(endpoint->peek(0x1040), 0x11111111u);
    EXPECT_EQ(endpoint->peek(0x1044), 0x22222222u);
    EXPECT_EQ(endpoint->peek(0x40204), 0x1u);
    EXPECT_FALSE(endpoint->has_register(0x03000204));

    // Block read (RD_BLOCK) -- the path HololinkV1::read_uint32 uses for consecutive words.
    auto [ok, values] = hololink->read_uint32(0x80, 3, std::shared_ptr<hololink::Timeout>());
    ASSERT_TRUE(ok);
    ASSERT_EQ(values.size(), 3u);
    EXPECT_EQ(values[0], 0x2606u);
    EXPECT_EQ(values[1], 20260219u);
    EXPECT_EQ(values[2], 0x484C5043u);

    // A rejected address (JESD block, absent on PCIe) comes back as RESPONSE_INVALID_ADDR,
    // which the legacy class reports through bad_write_response -> TransactionError.
    EXPECT_THROW(hololink->write_uint32(0x40000000u, 1u, nullptr, /*retry=*/false),
        hololink::TransactionError);
    EXPECT_FALSE(endpoint->has_register(0x40000000u));
}

TEST_F(HsbPcieHololink, AndOrUint32)
{
    hololink->start();
    ASSERT_TRUE(hololink->write_uint32(0x1000, 0xFFFF0000u));
    EXPECT_TRUE(hololink->and_uint32(0x1000, 0x0000FFFFu));
    EXPECT_EQ(endpoint->peek(0x1000), 0u);
    EXPECT_TRUE(hololink->or_uint32(0x1000, 0x1234u));
    EXPECT_EQ(endpoint->peek(0x1000), 0x1234u);
    EXPECT_TRUE(hololink->or_uint32(0x1000, 0x8000u));
    EXPECT_EQ(endpoint->peek(0x1000), 0x9234u);
}

TEST_F(HsbPcieHololink, ResetReidentifiesAndReconfigures)
{
    hololink->start();
    auto controller = std::make_shared<CountingResetController>();
    hololink->on_reset(controller);

    // Knock out state configure_hsb() re-establishes (shadowed APB timeout, forwarded I2C
    // timing), so we can see it run again.
    policy->shadow_write(CTRL_EVT_APB_TIMEOUT, 0);
    endpoint->poke(I2C_CLK_CNT_BAR0, 0);

    ASSERT_NO_THROW(hololink->reset());

    // trigger_reset(): 0x08 <- 0, 0x08 <- 3, 0x04 <- 8 all forwarded.
    EXPECT_EQ(endpoint->peek(SW_SEN_RST), 0x3u);
    EXPECT_EQ(endpoint->peek(SYS_RST), 0x8u);
    // Re-identified against the ctor-time reference and reconfigured.
    EXPECT_TRUE(endpoint->is_open());
    EXPECT_EQ(endpoint->peek(I2C_CLK_CNT_BAR0), 75u);
    EXPECT_EQ(policy->shadow_read(CTRL_EVT_APB_TIMEOUT), APB_TIMEOUT_750US);
    EXPECT_FALSE(endpoint->has_register(CTRL_EVT_APB_TIMEOUT));
    EXPECT_EQ(controller->resets, 1u);
}

TEST_F(HsbPcieHololink, ResetRejectsIdentityMismatch)
{
    hololink->start();
    hololink->set_reset_identity_timeout(0.3);
    // A different board answering after SYS_RST: MSW of the UUID changes. The design is
    // live (HSB_VER answers), so this must be reported as a UUID mismatch, not a timeout.
    endpoint->poke(UUID_REGS[0], 0xDEADBEEFu);
    try {
        hololink->reset();
        FAIL() << "reset() accepted a different UUID";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("UUID"), std::string::npos) << e.what();
        EXPECT_NE(std::string(e.what()).find("deadbeef"), std::string::npos) << e.what();
    }
}

TEST_F(HsbPcieHololink, ResetWaitsForDesignToComeBack)
{
    hololink->start();
    auto controller = std::make_shared<CountingResetController>();
    hololink->on_reset(controller);
    policy->shadow_write(CTRL_EVT_APB_TIMEOUT, 0);

    // The design "drops off the bus" (all-ones identity reads) and returns 200 ms later.
    constexpr auto outage = std::chrono::milliseconds(200);
    const SavedIdentity saved = knock_out_identity(*endpoint);
    std::thread restorer([&] {
        std::this_thread::sleep_for(outage);
        restore_identity(*endpoint, saved);
    });
    // A failed ASSERT below returns before the explicit join; a joinable std::thread
    // terminates in its destructor, so join on every exit path (the thread is bounded).
    struct JoinOnExit {
        std::thread& thread;
        ~JoinOnExit()
        {
            if (thread.joinable()) {
                thread.join();
            }
        }
    } join_restorer { restorer };

    const auto t0 = std::chrono::steady_clock::now();
    ASSERT_NO_THROW(hololink->reset());
    const auto elapsed = std::chrono::steady_clock::now() - t0;
    restorer.join();

    // reset() must have blocked until the identity came back (not exited on a cached flag).
    EXPECT_GE(elapsed, outage - std::chrono::milliseconds(50));
    EXPECT_LT(elapsed, std::chrono::seconds(3));
    // ... and then re-identified (same UUID) and reconfigured.
    EXPECT_TRUE(hololink->identity_known());
    for (size_t i = 0; i < UUID_BYTES; i++) {
        EXPECT_EQ(hololink->expected_uuid()[i], HSB_PCIE_MODULE_UUID[i]) << "byte " << i;
    }
    EXPECT_EQ(endpoint->peek(SYS_RST), 0x8u);
    EXPECT_EQ(endpoint->peek(I2C_CLK_CNT_BAR0), 75u);
    EXPECT_EQ(policy->shadow_read(CTRL_EVT_APB_TIMEOUT), APB_TIMEOUT_750US);
    EXPECT_FALSE(endpoint->has_register(CTRL_EVT_APB_TIMEOUT));
    EXPECT_EQ(controller->resets, 1u);
}

TEST_F(HsbPcieHololink, ResetTimesOutWhenDesignNeverReturns)
{
    hololink->start();
    auto controller = std::make_shared<CountingResetController>();
    hololink->on_reset(controller);
    constexpr double timeout_s = 0.3;
    hololink->set_reset_identity_timeout(timeout_s);
    EXPECT_DOUBLE_EQ(hololink->reset_identity_timeout(), timeout_s);

    knock_out_identity(*endpoint);

    const auto t0 = std::chrono::steady_clock::now();
    try {
        hololink->reset();
        FAIL() << "reset() returned although the design never came back";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("did not come back"), std::string::npos) << e.what();
    }
    const auto elapsed = std::chrono::steady_clock::now() - t0;
    EXPECT_GE(elapsed, std::chrono::milliseconds(250));
    EXPECT_LT(elapsed, std::chrono::seconds(3));
    // Nothing downstream ran.
    EXPECT_EQ(controller->resets, 0u);
}

TEST_F(HsbPcieHololink, ConstructorRejectsNullArguments)
{
    EXPECT_THROW(PcieHololink(nullptr, policy, PEER_KEY, SERIAL), std::runtime_error);
    EXPECT_THROW(PcieHololink(endpoint, nullptr, PEER_KEY, SERIAL), std::runtime_error);
}

TEST_F(HsbPcieHololink, FailedStartClosesEndpointItOpenedAndIsRetryable)
{
    // Fail the second write start() issues (I2C TIMEOUT), i.e. after the endpoint is open.
    auto failing = std::make_shared<FailingWriteEndpoint>(endpoint, I2C_TIMEOUT_BAR0);
    auto broken = std::make_shared<PcieHololink>(failing, std::make_shared<AddressPolicy>(), PEER_KEY, SERIAL);
    ASSERT_FALSE(endpoint->is_open());
    EXPECT_THROW(broken->start(), std::runtime_error);
    // Not started, and the fd start() opened is released again.
    EXPECT_FALSE(endpoint->is_open());
    // stop() on a never-started object stays a no-op; a later start() runs the sequence again.
    EXPECT_NO_THROW(broken->stop());
    EXPECT_THROW(broken->start(), std::runtime_error);
    EXPECT_FALSE(endpoint->is_open());

    // An endpoint the caller opened beforehand is left open.
    ASSERT_TRUE(endpoint->open());
    EXPECT_THROW(broken->start(), std::runtime_error);
    EXPECT_TRUE(endpoint->is_open());
}

TEST_F(HsbPcieHololink, DestructionUninstallsSequencerTrigger)
{
    hololink->start();
    EXPECT_TRUE(policy->has_sequencer_trigger());
    hololink->stop();
    hololink.reset();
    // The policy outlives the PcieHololink here (as it may for any caller holding its own
    // handle); a SW_EVENT write through it must not call into the destroyed sequencer.
    EXPECT_FALSE(policy->has_sequencer_trigger());
    ASSERT_TRUE(endpoint->open());
    Result r = policy->dispatch_write(CTRL_EVT_SW_EVENT, 1, *endpoint);
    EXPECT_TRUE(r.ok);
}

TEST_F(HsbPcieHololink, StopClosesEndpointAndRestartWorks)
{
    hololink->start();
    EXPECT_TRUE(endpoint->is_open());
    hololink->stop();
    EXPECT_FALSE(endpoint->is_open());
    // stop() is idempotent and start() can be called again.
    hololink->stop();
    ASSERT_NO_THROW(hololink->start());
    EXPECT_TRUE(endpoint->is_open());
}

TEST_F(HsbPcieHololink, DestructionWithoutStopIsSafe)
{
    hololink->start();
    // ~PcieHololink must clear started_ before ~Hololink runs the UDP-shaped base stop().
    EXPECT_NO_THROW(hololink.reset());
}

TEST_F(HsbPcieHololink, CtrlEvtAndApbRamStayHostSide)
{
    ASSERT_NO_THROW(hololink->start());
    EXPECT_TRUE(policy->has_sequencer_trigger()); // the software sequencer is always installed
    // The null sequence and CTRL_EVT programming stay host-side ...
    EXPECT_TRUE(policy->shadow_contains(NULL_SEQUENCE));
    EXPECT_FALSE(endpoint->has_register(NULL_SEQUENCE));
    EXPECT_TRUE(policy->shadow_contains(CTRL_EVT_APB_TIMEOUT));
    EXPECT_FALSE(endpoint->has_register(CTRL_EVT_APB_TIMEOUT));
    // ... while BAR0 traffic still reaches the endpoint.
    EXPECT_EQ(endpoint->peek(I2C_CLK_CNT_BAR0), 75u);
    EXPECT_EQ(hololink->read_uint32(0x80), 0x2606u);
}

// Delegates to a LoopbackPcieEndpoint and models the one thing the register file does not:
// SYS_RST returns HLNK_CTRL_1 (SW_SEN_RST) to its reset value 0, undoing the 0x3
// trigger_reset() wrote a moment earlier. Unrelated bits of the word are modelled as
// surviving so the read-modify-write can be checked.
class DesignResetEndpoint : public PcieEndpointInterface {
public:
    explicit DesignResetEndpoint(std::shared_ptr<LoopbackPcieEndpoint> inner)
        : inner_(std::move(inner))
    {
    }
    bool open() override { return inner_->open(); }
    void close() override { inner_->close(); }
    bool is_open() const override { return inner_->is_open(); }
    uint32_t minor() const override { return inner_->minor(); }
    const std::string& bdf() const override { return inner_->bdf(); }
    uint16_t requester_id() const override { return inner_->requester_id(); }
    PcieIdentity identity() const override { return inner_->identity(); }
    Status bar_read32(uint32_t offset, uint32_t& value) override { return inner_->bar_read32(offset, value); }
    Status bar_write32(uint32_t offset, uint32_t value) override
    {
        const Status status = inner_->bar_write32(offset, value);
        if (offset == SYS_RST && (value & ::hololink::pcie::HLNK_CTRL_0_SYS_RST_MASK)) {
            sys_resets++;
            inner_->poke(::hololink::pcie::HLNK_CTRL_1,
                inner_->peek(::hololink::pcie::HLNK_CTRL_1) & ~::hololink::pcie::HLNK_CTRL_1_SW_SEN_RST_MASK);
        }
        return status;
    }
    unsigned sys_resets { 0 };

private:
    std::shared_ptr<LoopbackPcieEndpoint> inner_;
};

TEST_F(HsbPcieHololink, StartAndResetReleaseSensorResets)
{
    // A design fresh from configuration reads HLNK_CTRL_1 = 0: both SW_SEN_RST bits held.
    // On the CertusPro-NX Versa bit 0 is the IMX258's hardware reset, so start() must
    // release the bits (read-modify-write, unrelated bits kept) before the first I2C
    // transaction. trigger_reset() writes 0x3 before SYS_RST and the design reset puts it
    // straight back to 0, so reset() has to release again after the design re-identifies.
    auto design = std::make_shared<DesignResetEndpoint>(endpoint);
    auto hl = std::make_shared<PcieHololink>(design, std::make_shared<AddressPolicy>(), PEER_KEY, SERIAL);
    endpoint->poke(::hololink::pcie::HLNK_CTRL_1, 0x00000030u); // held (bits 1:0 clear), unrelated 0x30 set
    hl->start();
    EXPECT_EQ(endpoint->peek(::hololink::pcie::HLNK_CTRL_1), 0x00000033u)
        << "start() releases the sensor resets and keeps the unrelated bits";

    ASSERT_NO_THROW(hl->reset());
    EXPECT_EQ(design->sys_resets, 1u);
    // Released after the reset. (The unrelated 0x30 is gone: trigger_reset() writes the
    // whole word, 0 then 0x3, before SYS_RST -- the inherited Ethernet behaviour. The
    // release itself is a read-modify-write of the bits the design zeroed.)
    EXPECT_EQ(endpoint->peek(::hololink::pcie::HLNK_CTRL_1), 0x00000003u);

    // A second reset: the design zeroes it again, reset() releases it again.
    ASSERT_NO_THROW(hl->reset());
    EXPECT_EQ(design->sys_resets, 2u);
    EXPECT_EQ(endpoint->peek(::hololink::pcie::HLNK_CTRL_1), 0x00000003u);
    hl->stop();
}

// The APB clock the I2C divider is computed against is the CertusPro-NX Versa's 60 MHz, the
// policy's default; a PcieHololink built on an explicit policy with that clock programs the
// same CLK_CNT (75) as one built on the default.
TEST_F(HsbPcieHololink, ApbClockIsTheVersaClock)
{
    EXPECT_EQ(AddressPolicy::DEFAULT_APB_CLK_HZ, 60000000u);
    EXPECT_EQ(policy->apb_clk_hz(), 60000000u);
    EXPECT_EQ(hololink->i2c_clk_cnt(), 75u);

    auto versa = std::make_shared<PcieHololink>(endpoint, std::make_shared<AddressPolicy>(60000000u), PEER_KEY, SERIAL);
    EXPECT_EQ(versa->i2c_clk_cnt(), 75u);
    versa->start();
    EXPECT_EQ(endpoint->peek(I2C_CLK_CNT_BAR0), 75u);
    versa->stop();
}
