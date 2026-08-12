/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_CORE_MIPI_DPHY_CPNX_HPP
#define HOLOLINK_MODULE_CORE_MIPI_DPHY_CPNX_HPP

#include <cstdint>
#include <memory>
#include <utility>

#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/hololink.hpp"
#include "hololink/module/mipi_dphy.hpp"
#include "hololink/module/service.hpp"
#include "hololink/module/status.h"

#include "common.hpp"
#include "lmmi_access.hpp"

namespace hololink::module::module_core {

/* The shape of mipi_cpnx_ref_design, stamped by the modules that
 * serve single-FPGA CertusPro-NX boards. Registers are stamped as LMMI
 * numbers; ApbLmmiAccess applies the x4.
 *
 * Under fpga/nv_mipi_ref_design/mipi_cpnx_ref_design/:
 *   rtl/top/FPGA_top.sv                 apb_psel[i+4] -> (index + 1) << 28;
 *                                       pcs_clk drives i_mipi_sync_clk
 *   rtl/mipi_cam_rcvr/mipi_cam_rcvr.sv  lmmi_offset_i = paddr[9:2]; the
 *                                       "// 60MHz" on sync_clk_i is stale
 *   build/ip/mipi_rx_ip/mipi_rx_ip.cfg  RX_LINE_RATE 1500
 *   build/ip/osc_clk/osc_clk.cfg        HF_CLK_FREQ 150 -> settle_clk_mhz */

inline void stamp_cpnx_mipi_dphy_metadata(
    EnumerationMetadata& metadata, unsigned settle_clk_mhz)
{
    metadata[mipi_dphy_key::SETTLE_CLK_MHZ] = int64_t { settle_clk_mhz };
    metadata[mipi_dphy_key::BASE_ADDRESS] = int64_t { 0x5000'0000 };
    metadata[mipi_dphy_key::RECEIVER_STRIDE] = int64_t { 0x1000'0000 };
    metadata[mipi_dphy_key::RECEIVER_COUNT] = int64_t { 2 };
    metadata[mipi_dphy_key::LANE_CONFIG_REGISTER] = int64_t { 0x0A };
    metadata[mipi_dphy_key::DATA_SETTLE_REGISTER] = int64_t { 0x36 };
    metadata[mipi_dphy_key::MAX_LINE_RATE_MBPS] = int64_t { 1500 };
}

/* Lattice mipi_rx_ip: lane count and data-lane settle are each a whole
 * register, reached over an APB window per receiver. */
class CpnxMipiDphyV1 : public MipiDphyInterfaceV1,
                       public Service<CpnxMipiDphyV1>,
                       private MipiDphyBase {
public:
    static constexpr const char* type_id = "mipi_dphy.cpnx.v1";
    using Service<CpnxMipiDphyV1>::get_service;
    using Service<CpnxMipiDphyV1>::for_each_type_id;
    using ServiceAlias = MipiDphyInterfaceV1;

    static constexpr unsigned DATA_LANE_SETTLE_MAX = 0xFF;

    explicit CpnxMipiDphyV1(std::shared_ptr<HololinkInterfaceV1> hololink)
        : hololink_(std::move(hololink))
    {
    }

    void configure(const EnumerationMetadata& metadata) override
    {
        configure_shape(metadata);
        const auto base_address = value_of(metadata, mipi_dphy_key::BASE_ADDRESS);
        const auto receiver_stride
            = value_of(metadata, mipi_dphy_key::RECEIVER_STRIDE);
        /* This layout reaches the register file through an APB window, so a
         * design without one is not ours -- leave it unconfigured. */
        if (!configured() || (base_address == 0) || (receiver_stride == 0)) {
            return;
        }
        lmmi_ = std::make_shared<ApbLmmiAccess>(hololink_,
            static_cast<uint32_t>(base_address),
            static_cast<uint32_t>(receiver_stride), receiver_count_);
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
        const uint8_t settle_reg = static_cast<uint8_t>(data_settle_register_);
        status = query(receiver, lane_reg);
        if (status != HOLOLINK_MODULE_OK) {
            return status;
        }
        status = lmmi_->write(
            receiver, lane_reg, static_cast<uint8_t>((lane_count - 1u) << 1));
        if (status != HOLOLINK_MODULE_OK) {
            return status;
        }
        return lmmi_->write(
            receiver, settle_reg, static_cast<uint8_t>(data_settle));
    }

private:
    std::shared_ptr<HololinkInterfaceV1> hololink_;
};

} // namespace hololink::module::module_core

#endif // HOLOLINK_MODULE_CORE_MIPI_DPHY_CPNX_HPP
