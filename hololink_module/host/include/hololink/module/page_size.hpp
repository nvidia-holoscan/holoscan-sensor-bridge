/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_PAGE_SIZE_HPP
#define HOLOLINK_MODULE_PAGE_SIZE_HPP

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace hololink::module {

/*
 * Matches the corresponding constant in src/hololink/core/networking.hpp
 * so module-side alignment math agrees with the underlying HSB hardware.
 */

/**
 * @brief Alignment, in bytes, that the device requires of the frame data
 * it transfers.
 *
 * Each frame's data, and the metadata block that follows it, begin at an
 * offset that is a multiple of this value, and the sizes of buffers that
 * hold them are rounded up to a multiple of it; use `round_up(size,
 * PAGE_SIZE)` to do so. It is a power of two.
 *
 * This is the device's own granularity. It has no relation to the host
 * operating system's memory page size.
 */
constexpr uint32_t PAGE_SIZE = 128;

/**
 * @brief Round a value up to a multiple of an alignment.
 *
 * Used to size and place buffers so they start and end on the
 * boundaries the hardware or an API requires, for example
 * `round_up(size, PAGE_SIZE)`.
 *
 * The alignment may be any positive value; it need not be a power of
 * two. A `value` that is already a multiple of `alignment` (including
 * zero) is returned unchanged.
 *
 * @param value The size or offset to round.
 * @param alignment The multiple to round up to. Must be greater than
 *      zero.
 * @returns The smallest multiple of `alignment` that is not less than
 *      `value`.
 * @throws std::runtime_error if `alignment` is zero.
 * @throws std::overflow_error if the rounded result is too large to be
 *      represented in a `size_t`.
 */
inline size_t round_up(size_t value, size_t alignment)
{
    if (alignment == 0) {
        throw std::runtime_error(
            "While rounding up: alignment must be greater than zero");
    }
    const size_t remainder = value % alignment;
    if (remainder == 0) {
        return value;
    }
    const size_t padding = alignment - remainder;
    if (value > std::numeric_limits<size_t>::max() - padding) {
        throw std::overflow_error(
            "While rounding up: result does not fit in size_t");
    }
    return value + padding;
}

} // namespace hololink::module

#endif // HOLOLINK_MODULE_PAGE_SIZE_HPP
