/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_I2C_HPP
#define HOLOLINK_MODULE_I2C_HPP

#include <cstdint>
#include <memory>
#include <vector>

#include "module.hpp"
#include "sequencer.hpp"
#include "service.hpp"
#include "status.h"

namespace hololink::module {

/* A single I2C bus + peripheral pair on the board. Reached through
 * HololinkInterfaceV1::get_i2c(bus, address) — the factory builds the
 * "serial=<n>;bus=<n>;address=<addr>" instance_id from the caller's
 * arguments and the per-board serial number. */
class I2cInterfaceV1 : public Service<I2cInterfaceV1> {
public:
    static constexpr const char* type_id = "i2c.v1";

    virtual ~I2cInterfaceV1() = default;

    /* Run a synchronous I2C transaction. write_bytes is sent over the
     * bus to peripheral_address; the read phase fills read_bytes
     * (resized to the caller-set size before the call). */
    virtual hololink_module_status_t i2c_transaction(
        uint32_t peripheral_address,
        const std::vector<uint8_t>& write_bytes,
        std::vector<uint8_t>& read_bytes)
        = 0;

    /* Encode an I2C transaction into the supplied Sequencer for
     * deferred / event-triggered execution. The output index vectors
     * identify locations in the sequencer buffer where the host can
     * later read back per-byte results and the final transaction
     * status. */
    virtual hololink_module_status_t encode_i2c_request(
        SequencerInterfaceV1& sequencer,
        uint32_t peripheral_i2c_address,
        const std::vector<uint8_t>& write_bytes,
        uint32_t read_byte_count,
        std::vector<unsigned>& out_write_indexes,
        std::vector<unsigned>& out_read_indexes,
        unsigned& out_status_index)
        = 0;
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_I2C_HPP
