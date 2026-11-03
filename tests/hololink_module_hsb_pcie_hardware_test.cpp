/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * PcieHololink against a real, bound Hololink PCIe endpoint. Opt-in: every case skips
 * unless HOLOLINK names the board the way the players' --hololink does -- its BDF
 * (HOLOLINK=0004:06:00.0 ctest -R hsb_pcie_hardware -V), or a /dev/hololink_pcieN /
 * /dev/hololink/by-bdf/<bdf> spelling. A HOLOLINK that is not a PCIe endpoint (an Ethernet
 * board's IP, say) also skips, so the variable can be shared with Ethernet tests. Needs the
 * hololink_pcie driver bound to that function and read/write access to its
 * /dev/hololink_pcieN node (group `hololink`, or root).
 *
 * What it proves that the loopback tests cannot: the read_uint32 / write_uint32 overrides
 * really reach BAR0 through the driver (start() identifies the design and its I2C timing
 * reads back), a scratch register round-trips through scalar, batched, block and
 * and/or access, concurrent callers are serialized on the real endpoint, and an I2C
 * transaction -- a WriteData batch carrying SW_EVENT plus the deferred software-sequencer
 * run against the real I2C block -- answers (the IMX258 model ID when a sensor is attached;
 * a NAK still means the whole path ran). Nothing here resets the design: SYS_RST kills the
 * sensor receive path on current bitfiles. The scratch register is restored on exit.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "pcie_regs.hpp"
#include <hololink/core/hololink.hpp>
#include <hololink/module/sensors/imx258/imx258_mode.hpp>

#include "address_policy.hpp"
#include "loopback_pcie_endpoint.hpp" // UUID_REGS / uuid_from_word
#include "pcie_endpoint.hpp"
#include "pcie_hololink.hpp"
#include "pcie_sysfs.hpp"

namespace {

using namespace hololink::module::hsb_pcie;
namespace regs = hololink::pcie;
namespace imx258 = hololink::module::sensors::imx258;

constexpr const char* PEER_ENV = "HOLOLINK";

bool live_word(uint32_t value)
{
    return value != 0u && value != 0xFFFFFFFFu;
}

class HsbPcieHardware : public ::testing::Test {
protected:
    void SetUp() override
    {
        const char* env = std::getenv(PEER_ENV);
        if (!env || !*env) {
            GTEST_SKIP() << PEER_ENV << " is not set; set it to the endpoint's PCI address "
                                        "(DDDD:BB:DD.F) to run against hardware";
        }
        const std::string peer = env;
        const auto resolved = hololink::module::pcie::bdf_for_peer(peer);
        if (!resolved) {
            GTEST_SKIP() << PEER_ENV << "=" << peer << " is not a PCIe endpoint (expected a BDF "
                                                       "DDDD:BB:DD.F or a /dev/hololink_pcieN node)";
        }
        bdf = *resolved;
        const auto minor = hololink::module::pcie::minor_for_bdf(bdf);
        ASSERT_TRUE(minor.has_value())
            << "no /sys/class/hololink_pcie entry for " << bdf
            << ": is the hololink_pcie driver bound to that function?";
        endpoint = std::make_shared<PcieEndpoint>(*minor);
        policy = std::make_shared<AddressPolicy>();
        hololink = std::make_shared<PcieHololink>(endpoint, policy, bdf, "pcie-" + bdf);
    }

    void TearDown() override
    {
        if (hololink) {
            hololink->stop();
        }
    }

    /// Save GLB_SCRATCH and put it back when the test leaves, pass or fail.
    struct ScratchGuard {
        PcieHololink& hololink;
        uint32_t saved;
        explicit ScratchGuard(PcieHololink& h)
            : hololink(h)
            , saved(h.read_uint32(regs::GLB_SCRATCH))
        {
        }
        ~ScratchGuard()
        {
            try {
                hololink.write_uint32(regs::GLB_SCRATCH, saved);
            } catch (...) {
            }
        }
    };

    std::string bdf;
    std::shared_ptr<PcieEndpoint> endpoint;
    std::shared_ptr<AddressPolicy> policy;
    std::shared_ptr<PcieHololink> hololink;
};

} // namespace

TEST_F(HsbPcieHardware, StartIdentifiesTheDesignThroughTheSeam)
{
    ASSERT_NO_THROW(hololink->start());
    ASSERT_TRUE(endpoint->is_open());
    EXPECT_TRUE(hololink->identity_known());
    EXPECT_EQ(hololink->expected_bdf(), bdf);

    // Scalar reads land on BAR0: HSB_VER agrees with what the driver reported at open.
    const PcieIdentity id = endpoint->identity();
    const uint32_t hsb_version = hololink->read_uint32(regs::HSB_VER);
    EXPECT_TRUE(live_word(hsb_version)) << std::hex << hsb_version;
    EXPECT_EQ(hsb_version, id.hsb_version);

    // Block read: the same words, in order.
    auto [ok, words] = hololink->read_uint32(regs::HSB_VER, 3, nullptr);
    ASSERT_TRUE(ok);
    ASSERT_EQ(words.size(), 3u);
    EXPECT_EQ(words[0], hsb_version);
    EXPECT_EQ(words[1], id.hsb_date);
    EXPECT_EQ(words[2], id.hsb_status);

    // The UUID registers read through the seam match the identity the module is keyed on.
    uint8_t uuid[UUID_BYTES] = { 0 };
    for (size_t i = 0; i < 4; i++) {
        uuid_from_word(hololink->read_uint32(UUID_REGS[i]), uuid, i * 4);
    }
    EXPECT_EQ(std::memcmp(uuid, id.uuid, UUID_BYTES), 0);
    EXPECT_EQ(std::memcmp(uuid, hololink->expected_uuid(), UUID_BYTES), 0);

    // start() wrote the I2C timing through the seam; it reads back from the real block.
    EXPECT_EQ(hololink->read_uint32(regs::I2C_CTRL_BASE + regs::I2C_CLK_CNT), hololink->i2c_clk_cnt());
    EXPECT_EQ(hololink->read_uint32(regs::I2C_CTRL_BASE + regs::I2C_TIMEOUT), PcieHololink::I2C_TIMEOUT_VALUE);

    // configure_hsb()'s Ethernet-only words stayed host-side.
    EXPECT_TRUE(policy->shadow_contains(hololink::CTRL_EVT_APB_TIMEOUT));
    EXPECT_FALSE(policy->sequencer_trigger_pending());

    // Cost of one scalar read on this transport (ctest -V shows it).
    constexpr unsigned samples = 1000;
    const auto t0 = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < samples; i++) {
        (void)hololink->read_uint32(regs::HSB_VER);
    }
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "[   INFO   ] read_uint32(HSB_VER) on " << bdf << ": mean " << us / samples
              << " us over " << samples << " reads" << std::endl;
}

TEST_F(HsbPcieHardware, ScratchRegisterRoundTripsEveryAccessShape)
{
    ASSERT_NO_THROW(hololink->start());
    ScratchGuard guard(*hololink);

    // Scalar.
    for (uint32_t pattern : { 0xA5A5A5A5u, 0x5A5A5A5Au, 0x00000000u, 0xFFFFFFFFu, 0x12345678u }) {
        ASSERT_TRUE(hololink->write_uint32(regs::GLB_SCRATCH, pattern));
        EXPECT_EQ(hololink->read_uint32(regs::GLB_SCRATCH), pattern) << std::hex << pattern;
    }

    // Batched: the last pair for the same address wins, in order.
    hololink::Hololink::WriteData batch;
    batch.queue_write_uint32(regs::GLB_SCRATCH, 0x11111111u);
    batch.queue_write_uint32(regs::GLB_SCRATCH, 0x22222222u);
    ASSERT_TRUE(hololink->write_uint32(batch));
    EXPECT_EQ(hololink->read_uint32(regs::GLB_SCRATCH), 0x22222222u);

    // Read-modify-write helpers (NamedLock around two seam calls).
    EXPECT_TRUE(hololink->and_uint32(regs::GLB_SCRATCH, 0x0000FFFFu));
    EXPECT_EQ(hololink->read_uint32(regs::GLB_SCRATCH), 0x00002222u);
    EXPECT_TRUE(hololink->or_uint32(regs::GLB_SCRATCH, 0x33000000u));
    EXPECT_EQ(hololink->read_uint32(regs::GLB_SCRATCH), 0x33002222u);

    // Block read spanning the scratch word and its neighbours.
    auto [ok, words] = hololink->read_uint32(regs::GLB_SCRATCH, 3, nullptr);
    ASSERT_TRUE(ok);
    ASSERT_EQ(words.size(), 3u);
    EXPECT_EQ(words[0], 0x33002222u);
    EXPECT_EQ(words[2], hololink->read_uint32(regs::HLNK_CTRL_1));

    // A rejected HSB range never reaches the board and reports like the loopback does.
    EXPECT_THROW(hololink->write_uint32(0x40000000u, 1u), hololink::TransactionError);
}

TEST_F(HsbPcieHardware, ConcurrentAccessIsSerializedOnTheEndpoint)
{
    ASSERT_NO_THROW(hololink->start());
    const uint32_t hsb_version = hololink->read_uint32(regs::HSB_VER);
    constexpr unsigned threads = 4;
    constexpr unsigned iterations = 250;
    std::atomic<unsigned> mismatches { 0 };
    std::vector<std::thread> workers;
    for (unsigned t = 0; t < threads; t++) {
        workers.emplace_back([&] {
            for (unsigned i = 0; i < iterations; i++) {
                if (hololink->read_uint32(regs::HSB_VER) != hsb_version) {
                    mismatches++;
                }
                auto [ok, words] = hololink->read_uint32(regs::HSB_VER, 2, nullptr);
                if (!ok || words.size() != 2 || words[0] != hsb_version) {
                    mismatches++;
                }
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    EXPECT_EQ(mismatches.load(), 0u);
}

// One I2C register read through the legacy Hololink::I2c: encode_i2c_request writes the
// program into shadow APB RAM, then one WriteData batch carries SW_EVENT=1, SW_EVENT=0 and
// APB_TIMEOUT, and the override runs the software sequencer after that batch against the
// board's real I2C block (START, DONE poll, status, data -- all through the alias onto BAR0).
TEST_F(HsbPcieHardware, I2cModelIdRunsTheSequencerPathOnTheBoard)
{
    ASSERT_NO_THROW(hololink->start());
    auto i2c = hololink->get_i2c(imx258::DEFAULT_CAM_I2C_BUS);
    ASSERT_TRUE(i2c);

    auto read_register = [&](uint16_t reg) -> uint8_t {
        std::vector<uint8_t> write_bytes {
            static_cast<uint8_t>((reg >> 8) & 0xFF),
            static_cast<uint8_t>(reg & 0xFF),
        };
        std::lock_guard<hololink::NamedLock> lock(hololink->i2c_lock());
        const std::vector<uint8_t> bytes = i2c->i2c_transaction(imx258::IMX258_I2C_ADDRESS, write_bytes, 1);
        EXPECT_EQ(bytes.size(), 1u);
        return bytes.empty() ? 0 : bytes[0];
    };

    uint16_t model_id = 0;
    try {
        const uint8_t hi = read_register(imx258::REG_MODEL_ID_HI);
        const uint8_t lo = read_register(imx258::REG_MODEL_ID_LO);
        model_id = static_cast<uint16_t>((hi << 8) | lo);
    } catch (const hololink::TransactionError& e) {
        // The transaction reached the sensor bus and the sequencer reported the outcome;
        // nothing answering at the IMX258 address is a bench condition, not a seam bug.
        if (std::string(e.what()).find("NAK") != std::string::npos) {
            EXPECT_TRUE(hololink->sequencer().last_run().ran);
            GTEST_SKIP() << "the I2C transaction ran through the sequencer path but no sensor "
                            "answered at "
                         << std::hex << imx258::IMX258_I2C_ADDRESS << " on bus "
                         << imx258::DEFAULT_CAM_I2C_BUS << ": " << e.what();
        }
        throw;
    }
    // The last transaction's run completed without a sequencer error and drained the trigger.
    EXPECT_TRUE(hololink->sequencer().last_run().ran);
    EXPECT_FALSE(hololink->sequencer().last_run().error) << hololink->sequencer().last_run().message;
    EXPECT_FALSE(policy->sequencer_trigger_pending());
    EXPECT_EQ(model_id, imx258::MODEL_ID) << std::hex << model_id;
    std::cout << "[   INFO   ] IMX258 model ID 0x" << std::hex << model_id << " read over I2C on " << bdf
              << std::endl;
}
