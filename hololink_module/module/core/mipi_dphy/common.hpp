/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_CORE_MIPI_DPHY_COMMON_HPP
#define HOLOLINK_MODULE_CORE_MIPI_DPHY_COMMON_HPP

#include <cstdint>
#include <limits>
#include <memory>
#include <utility>

#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/mipi_dphy.hpp"
#include "hololink/module/service.hpp"
#include "hololink/module/status.h"

#include "lmmi_access.hpp"

namespace hololink::module::module_core {

/* The D-PHY's shape is a property of the FPGA, and the FPGA is already
 * identified by UUID and IP version at enumeration. So the numbers arrive the
 * way sif_address / vp_address / hif_address do: the module stamps them into
 * the enumeration metadata and the implementation reads them back. A later IP
 * revision that moves a register stamps a different number and no code here
 * changes; a design with no D-PHY stamps nothing. */
namespace mipi_dphy_key {
    constexpr const char* BASE_ADDRESS = "mipi_dphy_base_address";
    constexpr const char* RECEIVER_STRIDE = "mipi_dphy_receiver_stride";
    constexpr const char* RECEIVER_COUNT = "mipi_dphy_receiver_count";
    /* LMMI register numbers, not byte offsets: a bus that addresses the
     * register file differently -- APB scales by 4 -- accounts for that in its
     * LmmiAccess, so these mean the same thing on every design. */
    constexpr const char* LANE_CONFIG_REGISTER = "mipi_dphy_lane_config_register";
    constexpr const char* DATA_SETTLE_REGISTER = "mipi_dphy_data_settle_register";
    constexpr const char* CLOCK_SETTLE_REGISTER = "mipi_dphy_clock_settle_register";
    constexpr const char* CLOCK_SETTLE_CYCLES = "mipi_dphy_clock_settle_cycles";
    constexpr const char* MAX_LINE_RATE_MBPS = "mipi_dphy_max_line_rate_mbps";
    /* Frequency of the clock the IP's settle counter runs on: sync_clk_i on
     * mipi_rx_ip, mipi_sync_clk on mipi_bridge. */
    constexpr const char* SETTLE_CLK_MHZ = "mipi_dphy_settle_clk_mhz";
} // namespace mipi_dphy_key

constexpr unsigned MIPI_DPHY_MAX_LANE_COUNT = 4;
/* LMMI addresses its register file with 8 bits, so a stamped number above
 * this truncates in program()'s cast and writes to a different register. */
constexpr unsigned MIPI_DPHY_MAX_REGISTER = 0xFF;

/* T_HS-SETTLE must land inside (85 + 6*UI, 145 + 10*UI) ns, UI = 1000 / line
 * rate as per spec, HS-RX timing. The IP counts the interval in cycles of its settle clock,
 * so: convert both edges to cycles, take the low edge rounded up plus one cycle of margin,
 * clamp to the high edge, and return 0 when no count fits between them. */
constexpr unsigned data_lane_settle_cycles(unsigned line_rate_mbps, unsigned settle_clk_mhz)
{
    if ((line_rate_mbps == 0) || (settle_clk_mhz == 0)) {
        return 0;
    }
    const uint64_t unit_interval_ps = 1'000'000u / line_rate_mbps;
    const uint64_t minimum_ps = 85'000u + 6u * unit_interval_ps;
    const uint64_t maximum_ps = 145'000u + 10u * unit_interval_ps;
    const uint64_t lowest = (minimum_ps * settle_clk_mhz + 999'999u) / 1'000'000u + 1u;
    const uint64_t highest = (maximum_ps * settle_clk_mhz) / 1'000'000u;
    if (highest == 0) {
        return 0;
    }
    return static_cast<unsigned>((lowest > highest) ? highest : lowest);
}

/* Everything a D-PHY implementation needs that is neither the register layout
 * nor the bus it sits on. Not a service itself: the layouts are the service
 * types, so a module publishes the one matching its FPGA. */
class MipiDphyBase {
protected:
    /* Reads the stamped shape. A design that stamped nothing leaves this
     * unconfigured and every program() call reports NOT_FOUND. */
    void configure_shape(const EnumerationMetadata& metadata)
    {
        receiver_count_ = value_of(metadata, mipi_dphy_key::RECEIVER_COUNT);
        max_line_rate_mbps_ = value_of(metadata, mipi_dphy_key::MAX_LINE_RATE_MBPS);
        lane_config_register_ = value_of(metadata, mipi_dphy_key::LANE_CONFIG_REGISTER);
        data_settle_register_ = value_of(metadata, mipi_dphy_key::DATA_SETTLE_REGISTER);
        clock_settle_register_ = value_of(metadata, mipi_dphy_key::CLOCK_SETTLE_REGISTER);
        clock_settle_cycles_ = value_of(metadata, mipi_dphy_key::CLOCK_SETTLE_CYCLES);
        settle_clk_mhz_ = value_of(metadata, mipi_dphy_key::SETTLE_CLK_MHZ);
    }

    /* Settle cycles for this design's clock. */
    unsigned data_settle_cycles(unsigned line_rate_mbps) const
    {
        return data_lane_settle_cycles(line_rate_mbps, settle_clk_mhz_);
    }

    /* The keys every layout needs. A layout additionally requires whatever is
     * specific to its transport or packing, and refuses to configure without
     * it -- so a design stamping one shape while inheriting another layout is
     * detected here rather than writing to the wrong registers. */
    bool configured() const
    {
        return (receiver_count_ > 0) && (settle_clk_mhz_ > 0)
            && in_register_range(lane_config_register_)
            && in_register_range(data_settle_register_);
    }

    /* Valid interfaces are [0, receiver_count); the count is stamped by the design. */
    hololink_module_status_t validate_interface(
        unsigned physical_mipi_interface) const
    {
        return (physical_mipi_interface < receiver_count_)
            ? HOLOLINK_MODULE_OK
            : HOLOLINK_MODULE_INVALID_PARAMETER;
    }

    /* Resolves MipiDphyInterfaceV1::LINE_RATE_DESIGN_MAXIMUM to the stamped
     * maximum, and rejects a rate the design cannot carry. */
    hololink_module_status_t resolve_rate(unsigned lane_count,
        unsigned line_rate_mbps, unsigned& out_line_rate_mbps) const
    {
        if ((lane_count < 1) || (lane_count > MIPI_DPHY_MAX_LANE_COUNT)) {
            return HOLOLINK_MODULE_INVALID_PARAMETER;
        }
        const unsigned rate = (line_rate_mbps == 0) ? max_line_rate_mbps_ : line_rate_mbps;
        if ((rate == 0) || ((max_line_rate_mbps_ > 0) && (rate > max_line_rate_mbps_))) {
            return HOLOLINK_MODULE_INVALID_PARAMETER;
        }
        out_line_rate_mbps = rate;
        return HOLOLINK_MODULE_OK;
    }

    /* Read a register to establish the block is really there before writing
     * anything. Any failure to read means the design cannot be driven here;
     * the accessor has already turned a throwing control plane into a
     * status. */
    hololink_module_status_t query(unsigned receiver, uint8_t reg)
    {
        if (!lmmi_) {
            return HOLOLINK_MODULE_NOT_FOUND;
        }
        uint8_t value = 0;
        return (lmmi_->read(receiver, reg, value) == HOLOLINK_MODULE_OK)
            ? HOLOLINK_MODULE_OK
            : HOLOLINK_MODULE_NOT_FOUND;
    }

    /* Stamped, non-zero, and small enough to survive the cast to uint8_t. */
    static bool in_register_range(unsigned reg)
    {
        return (reg > 0) && (reg <= MIPI_DPHY_MAX_REGISTER);
    }

    /* Reads a stamped number, or 0 for anything this class cannot hold. A
     * value too large to fit would otherwise truncate into a plausible-looking
     * one -- a base address of 0x1'5000'0000 becoming 0x5000'0000 -- so it is
     * reported as absent instead, which every caller already rejects. */
    static unsigned value_of(const EnumerationMetadata& metadata, const char* key)
    {
        const int64_t value = metadata.get<int64_t>(key, int64_t { 0 });
        if ((value <= 0)
            || (value > static_cast<int64_t>(std::numeric_limits<unsigned>::max()))) {
            return 0u;
        }
        return static_cast<unsigned>(value);
    }

    std::shared_ptr<LmmiAccess> lmmi_;
    unsigned receiver_count_ = 0;
    unsigned max_line_rate_mbps_ = 0;
    unsigned lane_config_register_ = 0;
    unsigned data_settle_register_ = 0;
    unsigned clock_settle_register_ = 0;
    unsigned clock_settle_cycles_ = 0;
    unsigned settle_clk_mhz_ = 0;
};

} // namespace hololink::module::module_core

#endif // HOLOLINK_MODULE_CORE_MIPI_DPHY_COMMON_HPP
