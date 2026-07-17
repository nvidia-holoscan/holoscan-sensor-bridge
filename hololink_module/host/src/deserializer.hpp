/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_DESERIALIZER_HPP
#define HOLOLINK_MODULE_DESERIALIZER_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

/* Host-private big/little-endian byte-stream reader used by the bootp
 * parser. Adapter-owned so the host framework needs no dependency on
 * src/hololink/core/deserializer. Each accessor advances the cursor and
 * returns false on buffer overflow rather than throwing. */

namespace hololink::module {

class Deserializer {
public:
    Deserializer(const uint8_t* buffer, size_t size)
        : buffer_(buffer)
        , limit_(size)
        , position_(0)
    {
    }

    bool next_uint8(uint8_t& result)
    {
        if (!has_bytes(1)) {
            return false;
        }
        result = buffer_[position_];
        position_ += 1;
        return true;
    }

    bool next_uint16_be(uint16_t& result)
    {
        if (!has_bytes(2)) {
            return false;
        }
        result = buffer_[position_ + 0];
        result = (result << 8) | buffer_[position_ + 1];
        position_ += 2;
        return true;
    }

    bool next_uint16_le(uint16_t& result)
    {
        if (!has_bytes(2)) {
            return false;
        }
        result = buffer_[position_ + 0];
        result |= (uint16_t(buffer_[position_ + 1]) << 8);
        position_ += 2;
        return true;
    }

    bool next_uint24_be(uint32_t& result)
    {
        if (!has_bytes(3)) {
            return false;
        }
        result = buffer_[position_ + 0];
        result = (result << 8) | buffer_[position_ + 1];
        result = (result << 8) | buffer_[position_ + 2];
        position_ += 3;
        return true;
    }

    bool next_uint32_be(uint32_t& result)
    {
        if (!has_bytes(4)) {
            return false;
        }
        result = buffer_[position_ + 0];
        result = (result << 8) | buffer_[position_ + 1];
        result = (result << 8) | buffer_[position_ + 2];
        result = (result << 8) | buffer_[position_ + 3];
        position_ += 4;
        return true;
    }

    // Fetch a pointer to n bytes at the cursor; false on overflow.
    bool pointer(const uint8_t*& pointer, size_t n)
    {
        if (!has_bytes(n)) {
            return false;
        }
        pointer = &(buffer_[position_]);
        position_ += n;
        return true;
    }

    // Fill buffer (using its current size) from the cursor; false on underflow.
    bool next_buffer(std::vector<uint8_t>& buffer)
    {
        const uint8_t* p = nullptr;
        size_t n = buffer.size();
        if (pointer(p, n)) {
            buffer = std::vector<uint8_t>(p, &p[n]);
            return true;
        }
        return false;
    }

    size_t position() const { return position_; }
    size_t remaining() const { return (limit_ >= position_) ? (limit_ - position_) : 0; }

private:
    // True if at least n bytes remain at the cursor. Subtracting rather than
    // adding keeps the check overflow-safe for any n.
    bool has_bytes(size_t n) const
    {
        return (position_ <= limit_) && (n <= (limit_ - position_));
    }

protected:
    const uint8_t* buffer_;
    size_t limit_;
    size_t position_;
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_DESERIALIZER_HPP
