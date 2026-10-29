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

/**
 * @brief A page-aligned region of GPU-accessible memory that a receiver
 * writes incoming data into, usually on GPU directly accessible memory.
 *
 * The memory is not initialized. It is released when the descriptor is
 * destroyed; the CUDA context it was created with must still be valid
 * at that point. Because the memory is released with the descriptor,
 * anything still using the address returned by get() must be finished
 * first.
 */
class ReceiverMemoryDescriptor {
public:
    /**
     * @brief Where the memory comes from.
     */
    enum class Placement {
        /// Device memory (cuMemAlloc) on a discrete GPU; on an integrated
        /// GPU, device-mapped host memory (cuMemHostAlloc), which the
        /// CPU can also read. The default.
        Automatic,
        /// Device memory (cuMemAlloc) on every GPU. Required when the
        /// buffer is dma-buf exported to another device
        /// (cuMemGetHandleForAddressRange), which host allocations
        /// cannot be; on an integrated GPU it is still system memory.
        Device,
    };

    /**
     * @brief Allocate the memory.
     *
     * Initializes the CUDA driver if necessary and makes `context` the
     * current CUDA context of the calling thread.
     *
     * @param context CUDA context in which to allocate. The address
     *      returned by get() is valid in this context.
     * @param size Number of usable bytes required, starting at get().
     *      It is not rounded up, so callers that need whole pages must
     *      do so themselves.
     * @param placement Which allocator to use; see Placement.
     * @throws std::overflow_error if `size` is too large to allocate
     *      with page alignment.
     * @throws std::runtime_error if a CUDA call fails, for example if
     *      the allocation cannot be satisfied.
     */
    explicit ReceiverMemoryDescriptor(CUcontext context, size_t size,
        Placement placement = Placement::Automatic);

    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    ReceiverMemoryDescriptor() = delete;
    ~ReceiverMemoryDescriptor() = default;
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief The start of the usable memory, as a GPU device address.
     *
     * The address is aligned to the system page size, and at least
     * `size` bytes (as given to the constructor) are available from it.
     * It is constant for the life of the descriptor.
     *
     * @returns The device pointer.
     */
    CUdeviceptr get() const { return mem_; }

    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
private:
    UniqueCUdeviceptr deviceptr_;
    UniqueCUhostptr host_deviceptr_;
    CUdeviceptr mem_;
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_RECEIVER_MEMORY_DESCRIPTOR_HPP
