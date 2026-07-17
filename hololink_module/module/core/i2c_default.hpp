/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_CORE_I2C_DEFAULT_HPP
#define HOLOLINK_MODULE_CORE_I2C_DEFAULT_HPP

#include <memory>
#include <stdexcept>
#include <utility>

#include "hololink/module/i2c.hpp"
#include "hololink/module/sequencer.hpp"
#include "hololink/module/status.h"

#include "hololink/core/hololink.hpp"

#include "hololink_default.hpp"

namespace hololink::module::module_core {

/* I2cInterfaceV1 backed by a hololink::Hololink::I2c. The legacy
 * I2c is reached via Hololink::get_i2c(bus, address); this wrapper
 * holds the shared_ptr<I2c> so the legacy controller stays alive.
 * It also holds its owning HololinkV1 (keeping the legacy Hololink
 * the I2c is parented to alive) and registers with it, so the
 * board's device_lost() invalidates this per-serial I2c too. */
class I2cV1 : public I2cInterfaceV1 {
public:
    I2cV1(std::shared_ptr<HololinkV1> hololink,
        std::shared_ptr<hololink::Hololink::I2c> backing)
        : hololink_(std::move(hololink))
        , backing_(std::move(backing))
    {
        if (!hololink_) {
            throw std::runtime_error(
                "While constructing I2cV1: owning HololinkV1 must be non-null");
        }
        if (!backing_) {
            throw std::runtime_error(
                "While constructing I2cV1: backing I2c must be non-null");
        }
        hololink_->register_associated(this);
    }

    hololink_module_status_t i2c_transaction(
        uint32_t peripheral_address,
        const std::vector<uint8_t>& write_bytes,
        std::vector<uint8_t>& read_bytes) override
    {
        const uint32_t want = static_cast<uint32_t>(read_bytes.size());
        std::vector<uint8_t> reply = backing_->i2c_transaction(
            peripheral_address, write_bytes, want);
        if (reply.size() != read_bytes.size()) {
            return HOLOLINK_MODULE_NETWORK_ERROR;
        }
        read_bytes = std::move(reply);
        return HOLOLINK_MODULE_OK;
    }

    /* Sequencer-encoded I2C requests are not yet bridged through
     * this wrapper. The legacy Hololink::I2c::encode_i2c_request
     * expects a hololink::Hololink::Sequencer&; recovering that from
     * an arbitrary SequencerInterfaceV1 needs either RTTI or a new
     * virtual hook on V1. Callers driving through the V1 path return
     * INVALID_PARAMETER and use synchronous i2c_transaction instead;
     * this method gains a real implementation when the bridging hook
     * lands alongside HsbLiteV1 integration. */
    hololink_module_status_t encode_i2c_request(
        SequencerInterfaceV1& /*sequencer*/,
        uint32_t /*peripheral_i2c_address*/,
        const std::vector<uint8_t>& /*write_bytes*/,
        uint32_t /*read_byte_count*/,
        std::vector<unsigned>& /*out_write_indexes*/,
        std::vector<unsigned>& /*out_read_indexes*/,
        unsigned& /*out_status_index*/) override
    {
        return HOLOLINK_MODULE_INVALID_PARAMETER;
    }

private:
    std::shared_ptr<HololinkV1> hololink_;
    std::shared_ptr<hololink::Hololink::I2c> backing_;
};

} // namespace hololink::module::module_core

#endif // HOLOLINK_MODULE_CORE_I2C_DEFAULT_HPP
