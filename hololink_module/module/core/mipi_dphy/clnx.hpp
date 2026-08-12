/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_CORE_MIPI_DPHY_CLNX_HPP
#define HOLOLINK_MODULE_CORE_MIPI_DPHY_CLNX_HPP

#include <cstdint>
#include <memory>
#include <utility>

#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/mipi_dphy.hpp"
#include "hololink/module/service.hpp"
#include "hololink/module/status.h"

#include "clnx_spi_lmmi_access.hpp"
#include "common.hpp"

namespace hololink::module::module_core {

/* The shape of the CrossLink-NX half of the two-FPGA designs. mipi_bridge
 * takes mipi_sync_clk from aclk_60 and mipi_csi_rx_rcfg is generated for
 * 2500 Mbps per lane.
 *   fpga/lattice/cpnx100-eth-sensor-bridge/clnx17/rtl/top/FPGA_clnx_top.sv
 *   fpga/lattice/cpnx100-eth-sensor-bridge/clnx17/build/ip/mipi_csi_rx_rcfg/mipi_csi_rx_rcfg.cfg
 * There is no base address: the register file is reached over SPI, not a
 * window, so only the register numbers are stamped. */
inline void stamp_clnx_mipi_dphy_metadata(EnumerationMetadata& metadata)
{
    metadata[mipi_dphy_key::RECEIVER_COUNT] = int64_t { 2 };
    metadata[mipi_dphy_key::SETTLE_CLK_MHZ] = int64_t { 60 };
    metadata[mipi_dphy_key::LANE_CONFIG_REGISTER] = int64_t { 0x0A };
    metadata[mipi_dphy_key::CLOCK_SETTLE_REGISTER] = int64_t { 0x0B };
    metadata[mipi_dphy_key::DATA_SETTLE_REGISTER] = int64_t { 0x0F };
    metadata[mipi_dphy_key::CLOCK_SETTLE_CYCLES] = int64_t { 9 };
    metadata[mipi_dphy_key::MAX_LINE_RATE_MBPS] = int64_t { 2500 };
}

/* Lattice mipi_csi_rx_rcfg: both settle values straddle register boundaries,
 * so a write touches five registers. */
class ClnxMipiDphyV1 : public MipiDphyInterfaceV1,
                       public Service<ClnxMipiDphyV1>,
                       private MipiDphyBase {
public:
    static constexpr const char* type_id = "mipi_dphy.clnx.v1";
    using Service<ClnxMipiDphyV1>::get_service;
    using Service<ClnxMipiDphyV1>::for_each_type_id;
    using ServiceAlias = MipiDphyInterfaceV1;

    /* Split across DATA_SETTLE[3:2] and DATA_SETTLE+1[3:0]. */
    static constexpr unsigned DATA_LANE_SETTLE_MAX = 0x3F;

    explicit ClnxMipiDphyV1(std::shared_ptr<LegacyHololinkAccess> hololink)
        : hololink_(std::move(hololink))
    {
    }

    void configure(const EnumerationMetadata& metadata) override
    {
        configure_shape(metadata);
        /* This layout splits both settle values across registers, so a design
         * that stamps no clock-settle field is not ours -- leave it
         * unconfigured rather than writing whole registers. */
        if (!configured() || (clock_settle_register_ == 0)
            || (clock_settle_cycles_ == 0)) {
            return;
        }
        lmmi_ = std::make_shared<ClnxSpiLmmiAccess>(hololink_, receiver_count_);
    }

    hololink_module_status_t program(unsigned physical_mipi_interface,
        unsigned lane_count, unsigned line_rate_mbps) override
    {
        if (!configured() || !lmmi_) {
            return HOLOLINK_MODULE_NOT_FOUND;
        }
        unsigned rate = 0;
        auto status = resolve_rate(lane_count, line_rate_mbps, rate);
        if (status != HOLOLINK_MODULE_OK) {
            return status;
        }
        const unsigned data_settle = data_settle_cycles(rate);
        if ((data_settle == 0) || (data_settle > DATA_LANE_SETTLE_MAX)) {
            return HOLOLINK_MODULE_INVALID_PARAMETER;
        }
        status = validate_interface(physical_mipi_interface);
        if (status != HOLOLINK_MODULE_OK) {
            return status;
        }
        const unsigned receiver = physical_mipi_interface;
        const uint8_t lane_reg = static_cast<uint8_t>(lane_config_register_);
        const uint8_t clock_reg = static_cast<uint8_t>(clock_settle_register_);
        const uint8_t data_reg = static_cast<uint8_t>(data_settle_register_);
        status = query(receiver, lane_reg);
        if (status != HOLOLINK_MODULE_OK) {
            return status;
        }
        const unsigned clock_settle = clock_settle_cycles_;
        const uint8_t writes[][2] = {
            { lane_reg,
                static_cast<uint8_t>(
                    ((clock_settle & 0x1u) << 3) | ((lane_count - 1u) << 1)) },
            { clock_reg, static_cast<uint8_t>(clock_settle >> 1) },
            { static_cast<uint8_t>(clock_reg + 1),
                static_cast<uint8_t>((clock_settle >> 2) & 0x80u) },
            { data_reg, static_cast<uint8_t>((data_settle & 0x3u) << 2) },
            { static_cast<uint8_t>(data_reg + 1),
                static_cast<uint8_t>((data_settle >> 2) & 0xFu) },
        };
        for (const auto& write : writes) {
            status = lmmi_->write(receiver, write[0], write[1]);
            if (status != HOLOLINK_MODULE_OK) {
                return status;
            }
        }
        return HOLOLINK_MODULE_OK;
    }

private:
    std::shared_ptr<LegacyHololinkAccess> hololink_;
};

} // namespace hololink::module::module_core

#endif // HOLOLINK_MODULE_CORE_MIPI_DPHY_CLNX_HPP
