/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#ifndef SRC_HOLOLINK_OPERATORS_CPU_ROCE_TRANSCEIVER_CPU_ROCE_FRAME_RECEIVER_INTERNAL_HPP
#define SRC_HOLOLINK_OPERATORS_CPU_ROCE_TRANSCEIVER_CPU_ROCE_FRAME_RECEIVER_INTERNAL_HPP

#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include <hololink/operators/cpu_roce_transceiver/cpu_roce_frame_receiver.hpp>

namespace hololink::operators {

// One producer publishes. One consumer claims. A record is written only in
// Writing, read only after Ready -> Leased, and freed from Leased or from
// Ready when nobody consumed it.
//   Free -> Writing -> Ready -> Leased -> Free
// The mailbox token is (generation << 32) | slot. Zero is empty. Generation
// starts at 1, so slot 0 is a real token. wake_seq is the futex word.
struct HsbFrame::SlotSync {
    enum State : std::uint8_t { Free = 0,
        Writing = 1,
        Ready = 2,
        Leased = 3 };
    static constexpr std::uint32_t kFlagStopping = 1u;
    static constexpr std::uint32_t kFlagFailure = 2u;
    static constexpr std::uint32_t kFlagQuiesced = 4u;

    struct Arrival {
        std::uint32_t slot = 0;
        std::optional<std::uint32_t> immediate_data;
        std::uint32_t dropped = 0;
        std::uint32_t overwritten = 0;
        std::uint32_t invalid_immediates = 0;
        std::uint64_t completion_count = 0;
        std::uint32_t session_frame_number = 0;
        std::optional<hololink::Hololink::FrameMetadata> metadata;
        std::chrono::system_clock::time_point received_at {};
    };
    struct alignas(128) Record {
        std::atomic<std::uint8_t> state { Free };
        Arrival snapshot;
    };
    // Arrival is 112 bytes at offset 8, leaving 8 bytes. Growing
    // FrameMetadata by 16 bytes makes this record 256 bytes, four cache
    // lines, and adds one line to every publish and claim.
    static_assert(sizeof(Record) == 128);

    std::mutex mutex;
    std::vector<Record> records;
    alignas(64) std::atomic<std::uint64_t> mailbox { 0 };
    alignas(64) std::atomic<std::uint32_t> wake_seq { 0 };
    alignas(64) std::atomic<std::uint32_t> flags { 0 };
    alignas(64) std::atomic<std::uint32_t> stats_seq { 0 };
    std::atomic<std::uint32_t> dropped { 0 };
    std::atomic<std::uint32_t> overwritten { 0 };
    std::atomic<std::uint32_t> invalid_immediates { 0 };
    std::atomic<std::uint64_t> completion_count { 0 };
    struct Reception {
        std::atomic<std::uint64_t> rx_bytes { 0 };
        std::atomic<std::uint32_t> frame_number { 0 };
        std::atomic<std::uint32_t> psn { 0 };
        std::uint16_t last_wire = 0;
        bool have_wire = false;
    };
    Reception reception;
    // Producer only. The first published token uses generation 1.
    std::uint32_t generation = 0;
    std::weak_ptr<hololink::transport::roce::CpuRoceTransceiver> transceiver;
    std::shared_ptr<void> payload_owner;
    // Set only after a failed release. Holds this object, and payload_owner,
    // for the life of the process. Copying the pointer does not allocate.
    std::shared_ptr<SlotSync> keep_alive;
    bool stopping = false;
    std::exception_ptr failure;
    std::atomic<bool> get_frame_active { false };
};

} // namespace hololink::operators

#endif /* SRC_HOLOLINK_OPERATORS_CPU_ROCE_TRANSCEIVER_CPU_ROCE_FRAME_RECEIVER_INTERNAL_HPP */
