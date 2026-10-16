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

// ECB request -> reply codec over AddressPolicy; layouts per src/hololink/core/hololink.cpp.

#include "ecb_bridge.hpp"

#include <stdexcept>

#include <hololink/core/deserializer.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/core/serializer.hpp>
#include <hololink/module/logging.hpp>

namespace hololink::module::hsb_pcie {

namespace {
    struct Item {
        uint32_t address;
        uint32_t value; ///< request value (writes) / value read or 0 (reads)
    };

    std::vector<uint8_t> encode_reply(uint8_t cmd, uint8_t flags, uint16_t sequence,
        uint8_t response_code, const std::vector<Item>& items, uint16_t latched_sequence)
    {
        std::vector<uint8_t> reply(EcbBridge::HEADER_BYTES + items.size() * 8 + EcbBridge::TRAILER_BYTES);
        core::Serializer serializer(reply.data(), reply.size());
        bool ok = serializer.append_uint8(static_cast<uint8_t>(cmd | EcbBridge::REPLY_BIT))
            && serializer.append_uint8(flags)
            && serializer.append_uint16_be(sequence)
            && serializer.append_uint8(response_code)
            && serializer.append_uint8(0); // reserved
        for (const Item& item : items) {
            ok = ok && serializer.append_uint32_be(item.address) && serializer.append_uint32_be(item.value);
        }
        ok = ok && serializer.append_uint16_be(latched_sequence)
            && serializer.append_uint8(0) && serializer.append_uint8(0); // pad
        if (!ok) {
            throw std::runtime_error("Unable to serialize ECB reply");
        }
        reply.resize(serializer.length());
        return reply;
    }
} // namespace

EcbBridge::EcbBridge(AddressPolicy& policy, PcieEndpointInterface& endpoint)
    : policy_(policy)
    , endpoint_(endpoint)
{
}

std::vector<uint8_t> EcbBridge::process(const std::vector<uint8_t>& request)
{
    if (request.empty()) {
        return {};
    }
    core::Deserializer deserializer(request.data(), request.size());
    uint8_t cmd = 0;
    uint8_t flags = 0;
    uint16_t sequence = 0;
    uint8_t reserved = 0;
    if (!(deserializer.next_uint8(cmd) && deserializer.next_uint8(flags)
            && deserializer.next_uint16_be(sequence) && deserializer.next_uint8(reserved)
            && deserializer.next_uint8(reserved))) {
        HSB_LOG_WARN("ECB request too short ({} bytes)", request.size());
        latched_sequence_ = 0;
        return encode_reply(cmd, flags, sequence, hololink::RESPONSE_INVALID_PKT_LENGTH, {}, latched_sequence_);
    }
    latched_sequence_ = sequence;
    // REQUEST_FLAGS_SEQUENCE_CHECK is ignored: there is no lossy carrier to detect
    // duplicates on, and the reader only compares the echoed sequence.

    bool is_write = false;
    bool single_address = false;
    switch (cmd) {
    case hololink::WR_DWORD:
    case hololink::WR_BLOCK:
        is_write = true;
        break;
    case hololink::RD_DWORD:
        single_address = true;
        break;
    case hololink::RD_BLOCK:
        break;
    default:
        HSB_LOG_WARN("ECB request with unknown cmd={:#x}", cmd);
        return encode_reply(cmd, flags, sequence, hololink::RESPONSE_INVALID_CMD, {}, latched_sequence_);
    }

    // n = ((len - 6) + 4) / 8: RD_DWORD carries one address, everything else 8-byte pairs.
    const size_t n = pair_count(request.size());
    std::vector<Item> items;
    items.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Item item { 0, 0 };
        bool ok = deserializer.next_uint32_be(item.address);
        if (ok && !single_address) {
            ok = deserializer.next_uint32_be(item.value);
        }
        if (!ok) {
            HSB_LOG_WARN("ECB request cmd={:#x} malformed at item {} of {}", cmd, i, n);
            return encode_reply(cmd, flags, sequence, hololink::RESPONSE_INVALID_PKT_LENGTH, {}, latched_sequence_);
        }
        if (!is_write) {
            item.value = 0; // RD_BLOCK carries a placeholder value we must not echo
        }
        items.push_back(item);
    }

    uint8_t response_code = hololink::RESPONSE_SUCCESS;
    for (Item& item : items) {
        Result result;
        if (is_write) {
            result = policy_.dispatch_write(item.address, item.value, endpoint_);
        } else {
            uint32_t value = 0;
            result = policy_.dispatch_read(item.address, endpoint_, value);
            if (result.ok) {
                item.value = value;
            }
        }
        if (!result.ok) {
            response_code = static_cast<uint8_t>(result.response_code);
            HSB_LOG_DEBUG("ECB cmd={:#x} address={:#x} failed response_code={}", cmd, item.address, response_code);
            break;
        }
    }
    // Every pair of this request has been applied (including a CTRL_EVT_APB_TIMEOUT queued
    // after SW_EVENT, per I2c::i2c_transaction); now run the sequencer it may have armed.
    if (policy_.sequencer_trigger_pending()) {
        policy_.fire_pending_sequencer_trigger(endpoint_);
    }
    return encode_reply(cmd, flags, sequence, response_code, items, latched_sequence_);
}

} // namespace hololink::module::hsb_pcie
