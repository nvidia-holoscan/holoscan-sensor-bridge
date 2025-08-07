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

#ifndef SRC_HOLOLINK_CORE_SERIALIZER
#define SRC_HOLOLINK_CORE_SERIALIZER

#include <cstring>
#include <stddef.h>
#include <stdint.h>

namespace hololink::core {

class Serializer {
public:
    Serializer(uint8_t* buffer, size_t size)
        : buffer_(buffer)
        , limit_(size)
        , position_(0)
    {
    }

    unsigned length()
    {
        return position_;
    }

    bool append_uint32_le(uint32_t value)
    {
        if ((position_ + 4) > limit_) {
            return false;
        }
        buffer_[position_] = (value >> 0) & 0xFF;
        buffer_[position_ + 1] = (value >> 8) & 0xFF;
        buffer_[position_ + 2] = (value >> 16) & 0xFF;
        buffer_[position_ + 3] = (value >> 24) & 0xFF;
        position_ += 4;
        return true;
    }

    bool append_uint32_be(uint32_t value)
    {
        if ((position_ + 4) > limit_) {
            return false;
        }
        buffer_[position_] = (value >> 24) & 0xFF;
        buffer_[position_ + 1] = (value >> 16) & 0xFF;
        buffer_[position_ + 2] = (value >> 8) & 0xFF;
        buffer_[position_ + 3] = (value >> 0) & 0xFF;
        position_ += 4;
        return true;
    }

    bool append_uint16_le(uint16_t value)
    {
        if ((position_ + 2) > limit_) {
            return false;
        }
        buffer_[position_] = (value >> 0) & 0xFF;
        buffer_[position_ + 1] = (value >> 8) & 0xFF;
        position_ += 2;
        return true;
    }

    bool append_uint16_be(uint16_t value)
    {
        if ((position_ + 2) > limit_) {
            return false;
        }
        buffer_[position_] = (value >> 8) & 0xFF;
        buffer_[position_ + 1] = (value >> 0) & 0xFF;
        position_ += 2;
        return true;
    }

    bool append_uint8(uint8_t value)
    {
        if ((position_ + 1) > limit_) {
            return false;
        }
        buffer_[position_] = value & 0xFF;
        position_ += 1;
        return true;
    }

    bool append_buffer(uint8_t* b, unsigned length)
    {
        if ((position_ + length) > limit_) {
            return false;
        }
        memcpy(&buffer_[position_], b, length);
        position_ += length;
        return true;
    }

    bool append_uint64_be(uint64_t value)
    {
        if ((position_ + 8) > limit_) {
            return false;
        }
        buffer_[position_] = (value >> 56) & 0xFF;
        buffer_[position_ + 1] = (value >> 48) & 0xFF;
        buffer_[position_ + 2] = (value >> 40) & 0xFF;
        buffer_[position_ + 3] = (value >> 32) & 0xFF;
        buffer_[position_ + 4] = (value >> 24) & 0xFF;
        buffer_[position_ + 5] = (value >> 16) & 0xFF;
        buffer_[position_ + 6] = (value >> 8) & 0xFF;
        buffer_[position_ + 7] = (value >> 0) & 0xFF;
        position_ += 8;
        return true;
    }

protected:
    uint8_t* buffer_;
    size_t limit_;
    size_t position_;
};

} // namespace hololink::core

#endif /* SRC_HOLOLINK_CORE_SERIALIZER */
