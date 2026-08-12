/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_CORE_MIPI_DPHY_LMMI_ACCESS_HPP
#define HOLOLINK_MODULE_CORE_MIPI_DPHY_LMMI_ACCESS_HPP

#include <cstdint>
#include <exception>
#include <memory>
#include <utility>
#include <vector>

#include "hololink/module/hololink.hpp"
#include "hololink/module/status.h"

namespace hololink::module::module_core {

/* The 8-bit register file the MIPI IPs expose. Designs differ in how a
 * host reaches it, but the register space is the same shape, so the D-PHY
 * layouts are written against this rather than against a bus. */
class LmmiAccess {
public:
    virtual ~LmmiAccess() = default;
    virtual unsigned receiver_count() const = 0;
    virtual hololink_module_status_t read(unsigned receiver, uint8_t reg, uint8_t& out_value) = 0;
    virtual hololink_module_status_t write(unsigned receiver, uint8_t reg, uint8_t value) = 0;
};

/* One APB window per receiver, LMMI offset = paddr[9:2] and data = pwdata[7:0].
 * fpga/nv_mipi_ref_design/mipi_cpnx_ref_design/rtl/mipi_cam_rcvr/mipi_cam_rcvr.sv */
class ApbLmmiAccess : public LmmiAccess {
public:
    ApbLmmiAccess(std::shared_ptr<HololinkInterfaceV1> hololink,
        uint32_t base_address, uint32_t receiver_stride, unsigned receiver_count)
        : hololink_(std::move(hololink))
        , base_address_(base_address)
        , receiver_stride_(receiver_stride)
        , receiver_count_(receiver_count)
    {
    }

    unsigned receiver_count() const override { return receiver_count_; }

    hololink_module_status_t read(
        unsigned receiver, uint8_t reg, uint8_t& out_value) override
    {
        if (receiver >= receiver_count_) {
            return HOLOLINK_MODULE_INVALID_PARAMETER;
        }
        if (!hololink_) {
            return HOLOLINK_MODULE_NOT_FOUND;
        }
        std::vector<uint32_t> values;
        try {
            const auto status
                = hololink_->read_uint32({ address(receiver, reg) }, values);
            if (status != HOLOLINK_MODULE_OK) {
                return status;
            }
        } catch (const std::exception&) {
            /* The control plane reports a bad address or a timeout by throwing. This is the
             * boundary between it and the module API, so callers above see a status and never an
             * exception. */
            return HOLOLINK_MODULE_NETWORK_ERROR;
        }
        if (values.empty()) {
            return HOLOLINK_MODULE_NOT_FOUND;
        }
        out_value = static_cast<uint8_t>(values.at(0) & 0xFFu);
        return HOLOLINK_MODULE_OK;
    }

    hololink_module_status_t write(
        unsigned receiver, uint8_t reg, uint8_t value) override
    {
        if (receiver >= receiver_count_) {
            return HOLOLINK_MODULE_INVALID_PARAMETER;
        }
        if (!hololink_) {
            return HOLOLINK_MODULE_NOT_FOUND;
        }
        try {
            return hololink_->write_uint32({ address(receiver, reg) }, { value });
        } catch (const std::exception&) {
            return HOLOLINK_MODULE_NETWORK_ERROR;
        }
    }

private:
    uint32_t address(unsigned receiver, uint8_t reg) const
    {
        return base_address_ + (receiver * receiver_stride_) + (uint32_t(reg) * 4u);
    }

    std::shared_ptr<HololinkInterfaceV1> hololink_;
    uint32_t base_address_;
    uint32_t receiver_stride_;
    unsigned receiver_count_;
};

} // namespace hololink::module::module_core

#endif // HOLOLINK_MODULE_CORE_MIPI_DPHY_LMMI_ACCESS_HPP
