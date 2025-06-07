/**
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 *
 * See README.md for detailed information.
 */

#ifndef SRC_HOLOLINK_NATIVE_DESERIALIZER
#define SRC_HOLOLINK_NATIVE_DESERIALIZER

#include <stddef.h>
#include <stdint.h>

#include <vector>

namespace hololink::core {

class Deserializer {
public:
    Deserializer(const uint8_t* buffer, size_t size)
        : buffer_(buffer)
        , limit_(size)
        , position_(0)
    {
    }

    // Returns true if result is set;
    // false on buffer overflow.
    bool next_uint32_le(uint32_t& result)
    {
        if ((position_ + 4) > limit_) {
            return false;
        }

        result = (uint32_t(buffer_[position_ + 0]) << 0) | (uint32_t(buffer_[position_ + 1]) << 8)
            | (uint32_t(buffer_[position_ + 2]) << 16) | (uint32_t(buffer_[position_ + 3]) << 24);

        position_ += 4;
        return true;
    }

    // Returns true if result is set;
    // false on buffer overflow.
    bool next_uint8(uint8_t& result)
    {
        if ((position_ + 1) > limit_) {
            return false;
        }

        result = buffer_[position_];
        position_ += 1;
        return true;
    }

    // Returns true if result is set;
    // false on buffer overflow.
    bool next_uint16_be(uint16_t& result)
    {
        if ((position_ + 2) > limit_) {
            return false;
        }
        result = buffer_[position_ + 0];
        result = (result << 8) | buffer_[position_ + 1];
        position_ += 2;
        return true;
    }

    // Returns true if result is set;
    // false on buffer overflow.
    bool next_uint16_le(uint16_t& result)
    {
        if ((position_ + 2) > limit_) {
            return false;
        }
        result = buffer_[position_ + 0];
        result |= (uint16_t(buffer_[position_ + 1]) << 8);
        position_ += 2;
        return true;
    }

    // Returns true if result is set;
    // false on buffer overflow.
    bool next_uint24_be(uint32_t& result)
    {
        if ((position_ + 3) > limit_) {
            return false;
        }
        result = buffer_[position_ + 0];
        result = (result << 8) | buffer_[position_ + 1];
        result = (result << 8) | buffer_[position_ + 2];
        position_ += 3;
        return true;
    }

    // Returns true if result is set;
    // false on buffer overflow.
    bool next_uint32_be(uint32_t& result)
    {
        if ((position_ + 4) > limit_) {
            return false;
        }
        result = buffer_[position_ + 0];
        result = (result << 8) | buffer_[position_ + 1];
        result = (result << 8) | buffer_[position_ + 2];
        result = (result << 8) | buffer_[position_ + 3];
        position_ += 4;
        return true;
    }

    // Returns true if result is set;
    // false on buffer overflow.
    bool next_uint48_be(uint64_t& result)
    {
        if ((position_ + 6) > limit_) {
            return false;
        }
        result = buffer_[position_ + 0];
        result = (result << 8) | buffer_[position_ + 1];
        result = (result << 8) | buffer_[position_ + 2];
        result = (result << 8) | buffer_[position_ + 3];
        result = (result << 8) | buffer_[position_ + 4];
        result = (result << 8) | buffer_[position_ + 5];
        position_ += 6;
        return true;
    }

    // Returns true if result is set;
    // false on buffer overflow.
    bool next_uint64_be(uint64_t& result)
    {
        if ((position_ + 8) > limit_) {
            return false;
        }
        result = buffer_[position_ + 0];
        result = (result << 8) | buffer_[position_ + 1];
        result = (result << 8) | buffer_[position_ + 2];
        result = (result << 8) | buffer_[position_ + 3];
        result = (result << 8) | buffer_[position_ + 4];
        result = (result << 8) | buffer_[position_ + 5];
        result = (result << 8) | buffer_[position_ + 6];
        result = (result << 8) | buffer_[position_ + 7];
        position_ += 8;
        return true;
    }

    // Returns true if result is set;
    // false on buffer overflow.
    bool next_uint64_le(uint64_t& result)
    {
        if ((position_ + 8) > limit_) {
            return false;
        }
        result = (uint64_t(buffer_[position_ + 0]) << 0);
        result |= (uint64_t(buffer_[position_ + 1]) << 8);
        result |= (uint64_t(buffer_[position_ + 2]) << 16);
        result |= (uint64_t(buffer_[position_ + 3]) << 24);
        result |= (uint64_t(buffer_[position_ + 4]) << 32);
        result |= (uint64_t(buffer_[position_ + 5]) << 40);
        result |= (uint64_t(buffer_[position_ + 6]) << 48);
        result |= (uint64_t(buffer_[position_ + 7]) << 56);
        position_ += 8;
        return true;
    }

    // Fetch a pointer to the current offset in the buffer; returns
    // false on buffer overflow.
    bool pointer(const uint8_t*& pointer, unsigned n)
    {
        if ((position_ + n) > limit_) {
            return false;
        }
        pointer = &(buffer_[position_]);
        position_ += n;
        return true;
    }

    // Returns true if all the elements in the buffer
    // are updated; or false on underflow.
    bool next_buffer(std::vector<uint8_t>& buffer)
    {
        const uint8_t* p = nullptr;
        unsigned n = buffer.size();
        if (pointer(p, n)) {
            buffer = std::vector<uint8_t>(p, &p[n]);
            return true;
        }
        return false;
    }

    size_t position() { return position_; }

protected:
    const uint8_t* buffer_;
    size_t limit_;
    size_t position_;
};

} // namespace hololink::core

#endif /* SRC_HOLOLINK_NATIVE_DESERIALIZER */
