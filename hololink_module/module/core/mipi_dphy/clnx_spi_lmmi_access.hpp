/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_CORE_MIPI_DPHY_CLNX_SPI_LMMI_ACCESS_HPP
#define HOLOLINK_MODULE_CORE_MIPI_DPHY_CLNX_SPI_LMMI_ACCESS_HPP

#include <cstdint>
#include <exception>
#include <memory>
#include <utility>

#include "hololink/module/status.h"

#include "hololink/core/hololink.hpp"

#include "hololink_default.hpp"
#include "lmmi_access.hpp"

namespace hololink::module::module_core {

/* LMMI in the CLNX, reached through the SPI peripheral register table in
 * fpga/lattice/cpnx100-eth-sensor-bridge/clnx17/rtl/spi/FPGA_spi_peri_ctrl_fsm.v:
 *
 *   regtbl[0x0C] bit0 = lmmi_wr_rdn, bits[5:4] = lmmi_request per receiver,
 *                bits[3:2] = lmmi_ready per receiver
 *   regtbl[0x0D] = lmmi_wdata   regtbl[0x0E] = lmmi_rdata   regtbl[0x0F] = lmmi_offset
 */
class ClnxSpiLmmiAccess : public LmmiAccess {
public:
    ClnxSpiLmmiAccess(
        std::shared_ptr<LegacyHololinkAccess> hololink, unsigned receiver_count)
        : hololink_(std::move(hololink))
        , receiver_count_(receiver_count)
    {
    }

    unsigned receiver_count() const override { return receiver_count_; }

    hololink_module_status_t write(
        unsigned receiver, uint8_t reg, uint8_t value) override
    {
        if ((receiver >= receiver_count_) || (receiver >= RECEIVER_LIMIT)) {
            return HOLOLINK_MODULE_INVALID_PARAMETER;
        }
        try {
            /* Inside the guard: get_spi() validates its parameters in the Spi
             * constructor's initializer list and throws on a bad one. */
            auto spi = open();
            if (!spi) {
                return HOLOLINK_MODULE_NOT_FOUND;
            }
            write_regtbl(spi, REGTBL_LMMI_WDATA, value);
            write_regtbl(spi, REGTBL_LMMI_OFFSET, reg);
            write_regtbl(spi, REGTBL_LMMI_CONTROL,
                static_cast<uint8_t>(request_bit(receiver) | LMMI_CONTROL_WRITE));
            return wait_ready(spi, receiver);
        } catch (const std::exception&) {
            /* The SPI controller reports failure by throwing; convert it here
             * so callers above see a status and never an exception. */
            return HOLOLINK_MODULE_NETWORK_ERROR;
        }
    }

    hololink_module_status_t read(
        unsigned receiver, uint8_t reg, uint8_t& out_value) override
    {
        if ((receiver >= receiver_count_) || (receiver >= RECEIVER_LIMIT)) {
            return HOLOLINK_MODULE_INVALID_PARAMETER;
        }
        try {
            auto spi = open();
            if (!spi) {
                return HOLOLINK_MODULE_NOT_FOUND;
            }
            write_regtbl(spi, REGTBL_LMMI_OFFSET, reg);
            write_regtbl(spi, REGTBL_LMMI_CONTROL, request_bit(receiver));
            const auto status = wait_ready(spi, receiver);
            if (status != HOLOLINK_MODULE_OK) {
                return status;
            }
            return read_regtbl(spi, REGTBL_LMMI_RDATA, out_value);
        } catch (const std::exception&) {
            return HOLOLINK_MODULE_NETWORK_ERROR;
        }
    }

private:
    static constexpr uint8_t SPI_WRITE_COMMAND = 0x01;
    static constexpr uint8_t SPI_READ_COMMAND = 0x11;
    static constexpr uint8_t REGTBL_LMMI_CONTROL = 0x0C;
    static constexpr uint8_t REGTBL_LMMI_WDATA = 0x0D;
    static constexpr uint8_t REGTBL_LMMI_RDATA = 0x0E;
    static constexpr uint8_t REGTBL_LMMI_OFFSET = 0x0F;
    static constexpr uint8_t LMMI_CONTROL_WRITE = 0x01;
    static constexpr unsigned LMMI_REQUEST_SHIFT = 4;
    static constexpr unsigned LMMI_READY_SHIFT = 2;
    /* regtbl[0x0C] gives lmmi_request bits [5:4] and lmmi_ready bits [3:2], so
     * this bridge reaches exactly two receivers however many the design says
     * it has. Beyond that the shifts would land on unrelated fields. */
    static constexpr unsigned RECEIVER_LIMIT = 2;
    static constexpr unsigned READY_POLL_LIMIT = 100;

    /* Empty when the controller can't be reached; callers report that rather
     * than dereferencing it. */
    std::shared_ptr<hololink::Hololink::Spi> open() const
    {
        if (!hololink_) {
            return {};
        }
        return hololink_->get_spi(hololink::CLNX_SPI_BUS, /*chip_select=*/0,
            /*clock_divisor=*/0x0F, /*cpol=*/0, /*cpha=*/1, /*width=*/1);
    }

    static uint8_t request_bit(unsigned receiver)
    {
        return static_cast<uint8_t>(1u << (LMMI_REQUEST_SHIFT + receiver));
    }

    static void write_regtbl(
        const std::shared_ptr<hololink::Hololink::Spi>& spi, uint8_t reg, uint8_t value)
    {
        spi->spi_transaction({}, { SPI_WRITE_COMMAND, reg, value }, 0);
    }

    /* An empty result means the transaction did not deliver a byte; reporting
     * that as a zero would be indistinguishable from a genuine zero. */
    static hololink_module_status_t read_regtbl(
        const std::shared_ptr<hololink::Hololink::Spi>& spi, uint8_t reg,
        uint8_t& out_value)
    {
        const auto values = spi->spi_transaction({}, { SPI_READ_COMMAND, reg }, 1);
        if (values.empty()) {
            return HOLOLINK_MODULE_NETWORK_ERROR;
        }
        out_value = values.at(0);
        return HOLOLINK_MODULE_OK;
    }

    /* Each receiver has its own ready bit; ecam0m30tof_player.py polls
     * receiver 0's unconditionally, which is wrong for receiver 1. */
    static hololink_module_status_t wait_ready(
        const std::shared_ptr<hololink::Hololink::Spi>& spi, unsigned receiver)
    {
        const uint8_t ready = static_cast<uint8_t>(1u << (LMMI_READY_SHIFT + receiver));
        for (unsigned i = 0; i < READY_POLL_LIMIT; ++i) {
            uint8_t control = 0;
            const auto status = read_regtbl(spi, REGTBL_LMMI_CONTROL, control);
            if (status != HOLOLINK_MODULE_OK) {
                return status;
            }
            if (control & ready) {
                return HOLOLINK_MODULE_OK;
            }
        }
        return HOLOLINK_MODULE_TIMEOUT;
    }

    std::shared_ptr<LegacyHololinkAccess> hololink_;
    unsigned receiver_count_;
};

} // namespace hololink::module::module_core

#endif // HOLOLINK_MODULE_CORE_MIPI_DPHY_CLNX_SPI_LMMI_ACCESS_HPP
