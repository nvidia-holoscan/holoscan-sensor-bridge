/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef HOLOLINK_MODULE_HSB_PCIE_ECB_BRIDGE_HPP
#define HOLOLINK_MODULE_HSB_PCIE_ECB_BRIDGE_HPP

/**
 * @file
 * @brief ECB control-packet codec: one request frame in, one reply frame out.
 *
 * Decodes the frames hololink::Hololink builds in src/hololink/core/hololink.cpp
 * (write_uint32_ / write_uint32_block_ / read_uint32_ / read_uint32_block_, lines
 * ~582-660, 737-790, 863-951), dispatches every (address[, value]) pair through the
 * AddressPolicy, and synthesizes the reply exactly as Hololink::execute() and the readers
 * parse it. It is the in-process stand-in for the UDP round trip to the hsb_pcie emulator
 * daemon (hsb_pcie/driver/src/hsb_pcie/hsb_pcie_emulator.cpp).
 */

#include <cstdint>
#include <vector>

#include "address_policy.hpp"
#include "pcie_endpoint_interface.hpp"

namespace hololink::module::hsb_pcie {

class EcbBridge {
public:
    /// Request/reply header: cmd u8, flags u8, sequence u16 BE, response_code/reserved u8, reserved u8.
    static constexpr size_t HEADER_BYTES = 6;
    /// Trailer: latched_sequence u16 BE + two zero pad bytes.
    static constexpr size_t TRAILER_BYTES = 4;
    static constexpr uint8_t REPLY_BIT = 0x80;

    EcbBridge(AddressPolicy& policy, PcieEndpointInterface& endpoint);

    /**
     * Execute one request and return the reply bytes.
     *
     * Pairs are dispatched in order; the first failure stops dispatch, keeps the earlier
     * writes, and sets the response code (RESPONSE_INVALID_ADDR for rejected/unmapped or
     * misaligned addresses). The reply always carries one (address, value) pair per request
     * pair: the value written for writes, the value read for reads, and the request value
     * (or 0 for reads) for pairs that were not reached.
     *
     * An unknown command yields RESPONSE_INVALID_CMD; a frame that cannot be decoded yields
     * RESPONSE_INVALID_PKT_LENGTH; both carry no pairs. An empty request yields an empty reply.
     *
     * A software-sequencer trigger armed by any pair of the request (a write of 1 to
     * CTRL_EVT_SW_EVENT) runs after the whole pair loop -- so a
     * CTRL_EVT_APB_TIMEOUT written later in the same WR_BLOCK, as I2c::i2c_transaction
     * does, bounds that run -- and before the reply is built. It runs even when a later pair
     * failed, matching an FPGA that already latched the earlier writes.
     */
    std::vector<uint8_t> process(const std::vector<uint8_t>& request);

    /// Sequence number of the most recent request (echoed in every reply's trailer).
    uint16_t latched_sequence() const { return latched_sequence_; }

    /// Number of (address[, value]) items encoded in a frame of `length` bytes.
    static size_t pair_count(size_t length)
    {
        return length < HEADER_BYTES ? 0 : ((length - HEADER_BYTES) + 4) / 8;
    }

private:
    AddressPolicy& policy_;
    PcieEndpointInterface& endpoint_;
    uint16_t latched_sequence_ { 0 };
};

} // namespace hololink::module::hsb_pcie

#endif /* HOLOLINK_MODULE_HSB_PCIE_ECB_BRIDGE_HPP */
