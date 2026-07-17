/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_PAGE_SIZE_HPP
#define HOLOLINK_MODULE_PAGE_SIZE_HPP

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace hololink::module {

// All module I/O addresses (data-channel buffers, EOF metadata
// blocks) are aligned to this page size. Matches the corresponding
// constant in src/hololink/core/networking.hpp so module-side
// alignment math agrees with the underlying HSB hardware.
constexpr uint32_t PAGE_SIZE = 128;

// Round value up to the next multiple of alignment. Throws
// std::runtime_error if alignment is zero. Inline so loaded modules
// (which include this header but don't link the module host
// library) get the symbol without an extra link dep.
inline size_t round_up(size_t value, size_t alignment)
{
    if (alignment == 0) {
        throw std::runtime_error(
            "While rounding up: alignment must be greater than zero");
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

} // namespace hololink::module

#endif // HOLOLINK_MODULE_PAGE_SIZE_HPP
