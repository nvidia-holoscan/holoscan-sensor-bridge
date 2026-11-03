/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 */

#ifndef SRC_HOLOLINK_OPERATORS_COMPUTE_CRC_CRC_CHANNEL
#define SRC_HOLOLINK_OPERATORS_COMPUTE_CRC_CRC_CHANNEL

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

namespace hololink::operators {

/**
 * Hands each frame's CRC from a ComputeCrcOp to the CheckCrcOp that validates it (native or
 * hololink_module operators alike).
 *
 * The two operators are split so the GPU CRC overlaps the pipeline work between them, so the
 * checker must find the CRC of the frame it is looking at while later frames' CRCs are already
 * being computed. graph.op<>() constructs operators from replayable factories, so one operator
 * can't hand the other a pointer to itself; instead the application creates one CrcChannel and
 * passes it to both graph.op<>() calls:
 *
 *     auto crc_channel = std::make_shared<hololink::operators::CrcChannel>();
 *     graph.op<ComputeCrcTensorOp>("compute_crc", crc_channel, cuda_device_ordinal, frame_size);
 *     graph.op<MyCheckCrcOp>("check_crc", crc_channel);
 *
 * Each computed CRC sits in a slot of its own, keyed by frame number, until the checker takes it.
 * Slots for frames dropped between the two operators are recycled when a later frame is taken;
 * beyond \p max_pending slots the oldest is recycled.
 */
class CrcChannel {
public:
    /// One frame's CRC in flight: nvcomp's device arguments and result, the pinned host copy of
    /// the result, and the event that completes it.
    struct Slot {
        void** message_dptr = nullptr;
        size_t* message_size_dptr = nullptr;
        uint32_t* result_dptr = nullptr;
        uint32_t* result_hptr = nullptr;
        cudaEvent_t done = nullptr;
        int64_t frame_number = -1;
    };

    explicit CrcChannel(size_t max_pending = 16)
        : max_pending_(max_pending ? max_pending : 1)
    {
    }

    CrcChannel(const CrcChannel&) = delete;
    CrcChannel& operator=(const CrcChannel&) = delete;

    ~CrcChannel()
    {
        for (auto& slot : slots_) {
            // Best effort: the process may already be tearing CUDA down.
            if (slot->done) {
                cudaEventSynchronize(slot->done);
                cudaEventDestroy(slot->done);
            }
            cudaFree(slot->message_dptr);
            cudaFree(slot->message_size_dptr);
            cudaFree(slot->result_dptr);
            cudaFreeHost(slot->result_hptr);
        }
    }

    /// ComputeCrcOp: a slot for frame \p frame_number, allocated on the current device when
    /// none is free. Enqueue the CRC into it, then publish() it (or release() it on failure).
    Slot* acquire(int64_t frame_number)
    {
        Slot* slot = nullptr;
        {
            std::lock_guard lock(mutex_);
            if (!free_.empty()) {
                slot = free_.back();
                free_.pop_back();
            }
        }
        if (!slot) {
            slot = allocate();
        } else {
            // A recycled slot may still belong to a CRC nobody took.
            check(cudaEventSynchronize(slot->done), "cudaEventSynchronize");
        }
        slot->frame_number = frame_number;
        return slot;
    }

    /// ComputeCrcOp: after the CRC is enqueued on \p stream, copies its result to the host on the
    /// same stream, records the slot's event and makes the CRC available to take().
    void publish(Slot* slot, cudaStream_t stream)
    {
        check(cudaMemcpyAsync(slot->result_hptr, slot->result_dptr, sizeof(*slot->result_hptr),
                  cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync");
        check(cudaEventRecord(slot->done, stream), "cudaEventRecord");
        std::lock_guard lock(mutex_);
        pending_.push_back(slot);
        while (pending_.size() > max_pending_) {
            // The checker fell behind (or isn't connected): forget the oldest CRC.
            free_.push_back(pending_.front());
            pending_.pop_front();
        }
    }

    /// ComputeCrcOp: gives back a slot whose CRC couldn't be enqueued.
    void release(Slot* slot)
    {
        std::lock_guard lock(mutex_);
        free_.push_back(slot);
    }

    /// CheckCrcOp: the CRC of frame \p frame_number, once computed (JAMCRC: bit-inverted CRC32,
    /// as the HSB reports it); nullopt when none is pending for that frame (it was dropped before
    /// its CRC was computed). Pending CRCs of earlier frames are discarded: those frames were
    /// dropped between the two operators.
    std::optional<uint32_t> take(int64_t frame_number)
    {
        Slot* slot = nullptr;
        {
            std::lock_guard lock(mutex_);
            for (auto it = pending_.begin(); it != pending_.end(); ++it) {
                if ((*it)->frame_number == frame_number) {
                    slot = *it;
                    free_.insert(free_.end(), pending_.begin(), it);
                    pending_.erase(pending_.begin(), it + 1);
                    break;
                }
            }
        }
        if (!slot) {
            return std::nullopt;
        }
        const cudaError_t status = cudaEventSynchronize(slot->done);
        const uint32_t crc = *slot->result_hptr;
        release(slot);
        check(status, "cudaEventSynchronize");
        return crc;
    }

    /// Forgets every pending CRC (ComputeCrcOp::start(): a restarted run starts afresh).
    void clear()
    {
        std::lock_guard lock(mutex_);
        free_.insert(free_.end(), pending_.begin(), pending_.end());
        pending_.clear();
    }

private:
    static void check(cudaError_t status, const char* what)
    {
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("CrcChannel: ") + what + " failed: " + cudaGetErrorString(status));
        }
    }

    Slot* allocate()
    {
        auto slot = std::make_unique<Slot>();
        check(cudaMalloc(&slot->message_dptr, sizeof(*slot->message_dptr)), "cudaMalloc");
        check(cudaMalloc(&slot->message_size_dptr, sizeof(*slot->message_size_dptr)), "cudaMalloc");
        check(cudaMalloc(&slot->result_dptr, sizeof(*slot->result_dptr)), "cudaMalloc");
        check(cudaMallocHost(&slot->result_hptr, sizeof(*slot->result_hptr)), "cudaMallocHost");
        check(cudaEventCreateWithFlags(&slot->done, cudaEventDisableTiming), "cudaEventCreateWithFlags");
        std::lock_guard lock(mutex_);
        slots_.push_back(std::move(slot));
        return slots_.back().get();
    }

    const size_t max_pending_;
    std::mutex mutex_;
    std::vector<std::unique_ptr<Slot>> slots_; ///< every slot, for the destructor
    std::vector<Slot*> free_;
    std::deque<Slot*> pending_; ///< published, oldest first
};

} // namespace hololink::operators

#endif /* SRC_HOLOLINK_OPERATORS_COMPUTE_CRC_CRC_CHANNEL */
