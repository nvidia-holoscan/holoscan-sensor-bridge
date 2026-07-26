/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_RECEIVER_MEMORY_DESCRIPTOR_HPP
#define HOLOLINK_MODULE_RECEIVER_MEMORY_DESCRIPTOR_HPP

#include <cstddef>

#include <cuda.h>

#include "hololink/module/cuda_unique.hpp"

namespace hololink::module {

/* Allocates a page-aligned region of GPU memory used as the
 * destination buffer for an incoming data stream (frame payload plus
 * the EOF metadata block). On a discrete GPU the storage is
 * cuMemAlloc-backed device memory; on an integrated (Tegra) GPU it's
 * cuMemHostAlloc-backed pinned memory mapped as a device pointer. The
 * pointer returned by get() is page-aligned; the underlying
 * allocation is freed when the descriptor goes out of scope. */
class ReceiverMemoryDescriptor {
public:
    explicit ReceiverMemoryDescriptor(CUcontext context, size_t size);
    ReceiverMemoryDescriptor() = delete;
    ~ReceiverMemoryDescriptor() = default;

    CUdeviceptr get() const { return mem_; }

private:
    UniqueCUdeviceptr deviceptr_;
    UniqueCUhostptr host_deviceptr_;
    CUdeviceptr mem_;
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_RECEIVER_MEMORY_DESCRIPTOR_HPP
