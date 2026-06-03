/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PVA_CRC_HPP
#define PVA_CRC_HPP

#include <cstdint>

// Forward declaration for async API (opaque handle)
struct PvaCrcFence;

/**
 * @brief PVA-accelerated CRC32 computation - Asynchronous 1D API with Parallel Support
 *
 * This is a pre-compiled library - users only need this header and libpva_2.9_crc.a
 *
 * Pure asynchronous API for maximum pipeline parallelism:
 *   1. launchCompute() - submits to PVA and returns immediately
 *   2. checkResults()  - polls for completion (non-blocking or blocking)
 *   3. freeFence()     - cleans up fence handle
 *
 * For blocking behavior, use checkResults() with timeoutUs = -1
 *
 * - 1D API with parallel computation support
 * - Compute CRC on any contiguous buffer
 * - Support for parallel chunks (results can be XORed)
 */
class PvaCrc {
public:
    PvaCrc();
    ~PvaCrc();

    PvaCrc(const PvaCrc&) = delete;
    PvaCrc& operator=(const PvaCrc&) = delete;

    /**
     * @brief Initialize with buffer size
     * @param dataSize Size of buffer in bytes (must be multiple of 4, min 32KB, max 12MB)
     * @param remainingSize Bytes remaining after this chunk (for parallel computation), default 0
     * @param isFirstChunk True to apply preconditioning (for first chunk), default true
     * @param useDualVpu Enable dual VPU parallel processing (auto-splits data across VPU0 and VPU1), default false
     * @return 0 on success, negative on error
     *
     * @note For manual parallel computation: split data into chunks, set remainingSize appropriately,
     *       compute each chunk's CRC, then XOR all results for final CRC
     * @note For automatic dual VPU: set useDualVpu=true, data is automatically split across both VPUs
     */
    int32_t init(uint32_t dataSize, uint32_t remainingSize = 0, bool isFirstChunk = true, bool useDualVpu = false);

    /**
     * @brief Launch CRC computation on PVA (NON-BLOCKING)
     *
     * Copies input from GPU to PVA memory, submits to PVA hardware, and returns immediately.
     * PVA executes in background while CPU/GPU can do other work.
     *
     * @param inputBuffer Input data buffer (GPU device memory)
     * @param outFence Output fence handle for polling completion (caller must free with freeFence)
     * @return 0 on success, negative on error
     *
     * @note This function returns before PVA completes. Use checkResults() to poll.
     */
    int32_t launchCompute(const void* inputBuffer, PvaCrcFence** outFence);

    /**
     * @brief Check if PVA computation is complete and retrieve results
     *
     * Polls the fence with specified timeout. If complete, copies results from PVA to GPU memory.
     *
     * @param fence Fence handle from launchCompute()
     * @param outputCrc Output CRC32 value (single uint32_t, GPU device memory)
     * @param timeoutUs Timeout in microseconds:
     *                  - 0 = non-blocking poll (return immediately if not ready)
     *                  - -1 = wait forever (blocking mode)
     *                  - >0 = wait up to N microseconds
     * @return 0 if complete and results copied,
     *         1 if still computing (timeout occurred),
     *         negative on error
     *
     * @note Multiple calls with timeout=0 enable polling without blocking
     */
    int32_t checkResults(PvaCrcFence* fence, uint32_t* outputCrc, int64_t timeoutUs = 0);

    /**
     * @brief Free fence handle
     *
     * Call this after checkResults() returns 0 to clean up the fence.
     * Safe to call on nullptr.
     *
     * @param fence Fence handle to free
     */
    void freeFence(PvaCrcFence* fence);

    /**
     * @brief Check if initialized
     * @return true if ready to process
     */
    bool isInitialized() const;

private:
    void* pImpl;
    bool m_initialized;
};

#endif // PVA_CRC_HPP
