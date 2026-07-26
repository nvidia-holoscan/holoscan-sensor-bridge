/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_CUDA_UNIQUE_HPP
#define HOLOLINK_MODULE_CUDA_UNIQUE_HPP

#include <memory>
#include <sstream>
#include <stdexcept>

#include <cuda.h>

#include "hololink/module/logging.hpp"

/* Adapter-owned CUDA RAII helpers and an error-check macro. The host
 * framework deliberately does not depend on src/hololink/common, so the
 * pieces a public module header needs (here, the unique_ptr aliases that
 * back ReceiverMemoryDescriptor) are defined locally rather than pulled
 * from the legacy cuda_helper.hpp. */

namespace hololink::module {

/* Wraps a trivially-copyable CUDA handle so it satisfies the
 * NullablePointer requirement and can be stored in a std::unique_ptr.
 * Mirrors the legacy Nullable<T> used by cuda_helper.hpp. */
template <typename T>
class Nullable {
public:
    Nullable(T value = 0)
        : value_(value)
    {
    }
    Nullable(std::nullptr_t)
        : value_(0)
    {
    }
    operator T() const { return value_; }
    explicit operator bool() const { return value_ != 0; }

    friend bool operator==(Nullable l, Nullable r) { return l.value_ == r.value_; }
    friend bool operator!=(Nullable l, Nullable r) { return !(l == r); }

    /* Calls `func` on the held handle when the unique_ptr is reset. */
    template <typename RESULT, RESULT func(T)>
    struct Deleter {
        typedef Nullable<T> pointer;
        void operator()(T value) const { func(value); }
    };

private:
    T value_;
};

/* Deleter for plain-pointer CUDA handles (e.g. host allocations). */
template <typename T, CUresult func(T)>
struct CuDeleter {
    typedef T pointer;
    void operator()(T value) const { func(value); }
};

/* RAII handles for CUDA device / host allocations, usable like
 * std::unique_ptr. */
using UniqueCUdeviceptr
    = std::unique_ptr<Nullable<CUdeviceptr>,
        Nullable<CUdeviceptr>::Deleter<CUresult, &cuMemFree>>;
using UniqueCUhostptr
    = std::unique_ptr<void, CuDeleter<void*, &cuMemFreeHost>>;

/* Throws std::runtime_error with the decoded CUDA error if `result` is
 * not CUDA_SUCCESS. Use through HOLOLINK_MODULE_CUDA_CHECK so the call
 * site's file/line are captured. */
inline void cuda_check(CUresult result, const char* file, int line)
{
    if (result != CUDA_SUCCESS) {
        const char* error_name = "";
        cuGetErrorName(result, &error_name);
        const char* error_string = "";
        cuGetErrorString(result, &error_string);
        std::stringstream buf;
        buf << "[" << file << ":" << line << "] CUDA driver error " << result
            << " (" << error_name << "): " << error_string;
        throw std::runtime_error(buf.str());
    }
}

} // namespace hololink::module

#define HOLOLINK_MODULE_CUDA_CHECK(FUNC) \
    ::hololink::module::cuda_check((FUNC), HOLOLINK_MODULE_FILE, __LINE__)

#endif // HOLOLINK_MODULE_CUDA_UNIQUE_HPP
