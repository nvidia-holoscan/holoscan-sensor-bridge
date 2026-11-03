/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
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

#ifndef SRC_HOLOLINK_OPERATORS_CPU_ROCE_TRANSCEIVER_CPU_ROCE_FRAME_TRANSMITTER_HPP
#define SRC_HOLOLINK_OPERATORS_CPU_ROCE_TRANSCEIVER_CPU_ROCE_FRAME_TRANSMITTER_HPP

#include <chrono>
#include <cstdint>
#include <memory>

#include <hololink/transport/roce/cpu_roce_transceiver.hpp>

struct CUevent_st;

namespace hololink::operators {

using CudaEvent = ::CUevent_st*;

/**
 * CUDA-aware UC frame transmitter above the caller-posted send path.
 * Experimental interface; it may change between releases. The stable
 * contract is hololink/transport/roce/cpu_roce_transceiver.hpp.
 *
 * The caller owns the payload allocation. A host-mapped pool is registered
 * as host memory. A device pool is exported as a DMA-BUF, and registered as
 * a GPU pointer if that export fails. The trailer, when present, is a
 * separate host region owned here. Slot::data() is a host pointer when
 * CudaPayload::host_mapped is set, and a device pointer otherwise.
 *
 * claim() lends an empty slot. send() posts it. copy_in() copies a foreign
 * buffer into a slot and leaves the slot loaned; the caller then send()s it.
 * Each loan is consumed once, by send() or release(). One late release of an
 * outstanding unsent loan remains valid after this object is destroyed. The
 * caller must not use a consumed handle again: after that release the pointer
 * can be invalid, and a later claim may return the same address for a
 * different loan. A second send of a consumed loan returns Failed and does
 * not post. Neither calls the error hook.
 * The ready event passed to send() must stay valid, and must not be recorded
 * again, until that slot is retired. In worker mode that wait happens later,
 * on the worker thread.
 *
 * send() and copy_in() do not throw. The constructor throws if setup or
 * connect fails. Synchronous send returns Completed, Failed, Timeout, or
 * Rejected. Worker send returns Accepted or Rejected. copy_in() returns null
 * when the pool stays empty for capacity_timeout, when stop() has begun,
 * when the sender has failed, or when the copy fails. Each of those nulls
 * counts one error, because the frame is lost; the hook message names which
 * one. Rejected counts one
 * error in both modes. capacity_timeout is per call because it bounds only
 * the wait for a free slot.
 *
 * start(), stop(), and close() are not called on top of an active send or
 * copy_in. stop() does wake a copy_in that is waiting for a free slot.
 * Waking that call does not mean it has returned. Every operation except a
 * late release of an unsent loan must finish before close().
 */
class CpuRoceFrameTransmitter {
    struct Impl;

public:
    enum class SenderMode { Synchronous,
        Worker };

    enum class SendStatus { Completed,
        Accepted,
        Rejected,
        Failed,
        Timeout };

    class Slot {
    public:
        void* data() const
        {
            return data_;
        }
        std::uint32_t capacity() const
        {
            return capacity_;
        }

    private:
        friend class CpuRoceFrameTransmitter;
        friend struct CpuRoceFrameTransmitter::Impl;
        void* block_ = nullptr;
        std::uint32_t index_ = 0;
        void* data_ = nullptr;
        std::uint32_t capacity_ = 0;
    };

    struct CudaPayload {
        void* address = nullptr;
        std::size_t size = 0;
        bool host_mapped = false;
    };

    using ErrorHook = void (*)(void* context, const char* message) noexcept;

    struct Config {
        std::string ibv_name;
        std::uint32_t ibv_port = 1;
        std::string local_ip;
        hololink::transport::roce::CpuRoceTransceiver::Peer peer;
        CudaPayload cuda_payload;
        std::shared_ptr<void> payload_owner;
        void* cuda_context = nullptr;
        std::size_t stride = 0;
        std::size_t payload_size = 0;
        std::uint32_t slot_count = 0;
        std::uint32_t trailer_capacity = 0;
        SenderMode mode = SenderMode::Synchronous;
        bool clamp_path_mtu = true;
        std::chrono::milliseconds send_timeout { 5000 };
        ErrorHook on_error = nullptr;
        void* on_error_context = nullptr;
    };

    explicit CpuRoceFrameTransmitter(Config config);
    ~CpuRoceFrameTransmitter();

    CpuRoceFrameTransmitter(const CpuRoceFrameTransmitter&) = delete;
    CpuRoceFrameTransmitter& operator=(const CpuRoceFrameTransmitter&) = delete;

    Slot* claim();
    // C ABI release callback. Consumes one outstanding loan. The wait result
    // is ignored, and nothing here allocates or calls the hook.
    static void release(Slot* slot, CudaEvent done = nullptr) noexcept;

    Slot* copy_in(const void* source, std::uint32_t bytes, CudaEvent ready,
        std::chrono::milliseconds capacity_timeout);
    SendStatus send(Slot* slot, std::uint32_t bytes, CudaEvent ready = nullptr,
        const void* trailer = nullptr, std::uint32_t trailer_bytes = 0);

    void start();
    void stop();
    void close();

    struct Stats {
        std::uint64_t bytes = 0;
        std::uint64_t frames = 0;
        std::uint64_t errors = 0;
        std::uint32_t last_psn = 0;
    };
    Stats stats() const;
    std::uint32_t qp_number() const;
    std::uint32_t path_mtu() const;
    bool failed() const;

private:
    std::unique_ptr<Impl> impl_;
};

} // namespace hololink::operators

#endif /* SRC_HOLOLINK_OPERATORS_CPU_ROCE_TRANSCEIVER_CPU_ROCE_FRAME_TRANSMITTER_HPP */
