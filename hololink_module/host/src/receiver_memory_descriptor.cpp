/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hololink/module/receiver_memory_descriptor.hpp"

#include <limits>
#include <stdexcept>
#include <unistd.h> // getpagesize

#include <fmt/format.h>

#include "hololink/module/logging.hpp" // HSB_LOG_TRACE

namespace hololink::module {

ReceiverMemoryDescriptor::ReceiverMemoryDescriptor(CUcontext cu_context, size_t size)
{
    HOLOLINK_MODULE_CUDA_CHECK(cuInit(0));
    HOLOLINK_MODULE_CUDA_CHECK(cuCtxSetCurrent(cu_context));
    CUdevice device;
    HOLOLINK_MODULE_CUDA_CHECK(cuCtxGetDevice(&device));
    int integrated = 0;

    // Pad the allocation so the returned pointer can be aligned up to
    // a page boundary regardless of what cuMem*Alloc handed us.
    const size_t page_size = getpagesize();
    const size_t page_mask = page_size - 1;
    if (size > std::numeric_limits<size_t>::max() - page_size) {
        throw std::overflow_error(fmt::format(
            "While allocating receiver memory: requested size={:#x} is too "
            "large for page-aligned allocation",
            size));
    }
    const size_t allocation_size = size + page_size;

    HOLOLINK_MODULE_CUDA_CHECK(cuDeviceGetAttribute(&integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, device));
    HSB_LOG_TRACE("integrated={}", integrated);

    if (integrated == 0) {
        // Discrete GPU — cuMemAlloc / cuMemFree.
        deviceptr_.reset([allocation_size] {
            CUdeviceptr device_deviceptr;
            HOLOLINK_MODULE_CUDA_CHECK(cuMemAlloc(&device_deviceptr, allocation_size));
            return device_deviceptr;
        }());
        CUdeviceptr mem = deviceptr_.get();
        const size_t rem = mem & page_mask;
        if (rem) {
            mem += (page_size - rem);
        }
        mem_ = mem;
    } else {
        // Integrated GPU (e.g. Tegra) — cuMemHostAlloc + map.
        host_deviceptr_.reset([allocation_size] {
            void* host_deviceptr;
            const unsigned int flags = CU_MEMHOSTALLOC_DEVICEMAP;
            HOLOLINK_MODULE_CUDA_CHECK(cuMemHostAlloc(&host_deviceptr, allocation_size, flags));
            return host_deviceptr;
        }());
        CUdeviceptr device_deviceptr;
        HOLOLINK_MODULE_CUDA_CHECK(cuMemHostGetDevicePointer(&device_deviceptr, host_deviceptr_.get(), 0));
        CUdeviceptr mem = device_deviceptr;
        const size_t rem = mem & page_mask;
        if (rem) {
            mem += (page_size - rem);
        }
        mem_ = mem;
    }
}

} // namespace hololink::module
