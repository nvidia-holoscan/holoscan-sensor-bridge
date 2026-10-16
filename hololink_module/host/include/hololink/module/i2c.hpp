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

/**
 * @brief Provides access to I2C controllers instantiated within
 *      an HSB device.
 *
 * You can request an immediate I2C transaction (with i2c_transaction)
 * or one that is downloaded and triggered by a specific event
 * (using encode_i2c_request).
 *
 * In some cases (specifically the default I2C configuration within
 * the HSB IP core) a single I2C controller is muxed between different
 * output pins, providing hardware with different I2C buses, but
 * requiring that the host serialize access to each.  When this
 * serialization is required, it's managed automatically within the
 * module-- so application code doesn't have to know that.
 */
class I2cInterfaceV1 : public Service<I2cInterfaceV1> {
public:
    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    static constexpr const char* type_id = "i2c.v1";
    static constexpr const char* release_version = "2.7.0";

    virtual ~I2cInterfaceV1() = default;
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief Execute an I2C transaction.
     *
     * Writes `write_bytes` to the peripheral, then reads back as many bytes
     * as `read_bytes` is configured to hold; size it to zero for a
     * write-only transaction.
     *
     * @code
     *  // Read the four bytes the peripheral holds at register 0x0010.
     *  std::vector<uint8_t> write_bytes { 0x00, 0x10 };
     *  // read_bytes carries the byte count in: size it to what you want to
     *  // read, and the call fills it in place.
     *  std::vector<uint8_t> read_bytes(4);
     *  if (i2c->i2c_transaction(peripheral_address, write_bytes, read_bytes)
     *      != HOLOLINK_MODULE_OK) {
     *      // The transaction didn't run; read_bytes holds nothing to use.
     *  } else {
     *      // read_bytes now holds the four bytes the peripheral returned.
     *  }
     * @endcode
     *
     * @param peripheral_address Address of the peripheral on this bus.
     * @param write_bytes Bytes sent to the peripheral.
     * @param[in,out] read_bytes Sized by the caller to the number of bytes to
     * read; receives them.
     * @return HOLOLINK_MODULE_OK or an error status.
     */
    virtual hololink_module_status_t i2c_transaction(
        uint32_t peripheral_address,
        const std::vector<uint8_t>& write_bytes,
        std::vector<uint8_t>& read_bytes)
        = 0;

    /**
     * @brief Encode an I2C transaction into a sequencer.
     *
     * Sequencers cache a list of local bus transactions, after
     * that cache is written to an HSB device, and the trigger activates,
     * the HSB IP will execute that transaction list.  This API appends
     * an I2C transaction to the sequencer cache.  The bytes to be
     * written to I2C are included in the call (in write_bytes); when
     * reading from I2C, the list of HSB IP local bus addresses where
     * the cached data can be found are put in `out_read_indexes`.  Each
     * location has 4 bytes; the last bytes (when read_byte_count
     * isn't an even 32-bits) are undefined.
     *
     * @param sequencer Sequencer to encode into.
     * @param peripheral_i2c_address Address of the peripheral on this bus.
     * @param write_bytes Bytes to send to the peripheral.
     * @param read_byte_count Bytes to read back from the peripheral.
     * @param[out] out_write_indexes Where the bytes to send are encoded.
     * @param[out] out_read_indexes Where the bytes read back are deposited.
     * @param[out] out_status_index Where the transaction's status is
     * deposited.
     * @return HOLOLINK_MODULE_OK or an error status.
     */
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
