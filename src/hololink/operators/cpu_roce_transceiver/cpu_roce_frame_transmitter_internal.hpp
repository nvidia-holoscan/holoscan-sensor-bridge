/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Slot table for CpuRoceFrameTransmitter. No verbs and no CUDA, so the
 * transitions can be tested without a device. The transmitter is the only
 * production caller.
 */

#ifndef SRC_HOLOLINK_OPERATORS_CPU_ROCE_TRANSCEIVER_CPU_ROCE_FRAME_TRANSMITTER_INTERNAL_HPP
#define SRC_HOLOLINK_OPERATORS_CPU_ROCE_TRANSCEIVER_CPU_ROCE_FRAME_TRANSMITTER_INTERNAL_HPP

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hololink::operators::frame_tx {

enum class State : std::uint8_t {
    Free,
    Loaned,
    Queued,
    InFlight,
};

struct Stats {
    std::uint64_t bytes = 0;
    std::uint64_t frames = 0;
    std::uint64_t errors = 0;
    std::uint32_t last_psn = 0;
};

// A validation error leaves the queue pair up. invalid_argument and
// out_of_range derive from logic_error, so they have to be recognized first.
inline bool failure_is_terminal(const std::exception& error)
{
    if (dynamic_cast<const std::invalid_argument*>(&error) != nullptr
        || dynamic_cast<const std::out_of_range*>(&error) != nullptr) {
        return false;
    }
    return true;
}

class Pool : public std::enable_shared_from_this<Pool> {
public:
    static std::shared_ptr<Pool> create(
        std::uint32_t slot_count, void* base, std::size_t stride, std::uint32_t capacity)
    {
        return std::shared_ptr<Pool>(new Pool(slot_count, base, stride, capacity));
    }

    std::uint32_t slot_count() const
    {
        return slot_count_;
    }

    void* data(std::uint32_t index) const
    {
        return base_ + static_cast<std::size_t>(index) * stride_;
    }

    std::uint32_t capacity() const
    {
        return capacity_;
    }

    std::optional<std::uint32_t> claim()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return claim_locked();
    }

    void release_loaned(std::uint32_t index)
    {
        std::shared_ptr<Pool> dropped;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (index >= slot_count_ || records_[index].state != State::Loaned) {
                return;
            }
            dropped = recycle_locked(index, false);
        }
    }

    bool is_loaned(std::uint32_t index) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return index < slot_count_ && records_[index].state == State::Loaned;
    }

    bool enqueue(std::uint32_t index, std::uint32_t bytes, std::uint32_t trailer_bytes, void* ready)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index >= slot_count_ || records_[index].state != State::Loaned) {
            return false;
        }
        records_[index].state = State::Queued;
        records_[index].bytes = bytes;
        records_[index].trailer_bytes = trailer_bytes;
        records_[index].ready = ready;
        push(queued_ring_, queued_head_, queued_count_, index);
        wake_.notify_all();
        return true;
    }

    bool pop_queued(std::uint32_t* index, std::uint32_t* bytes, std::uint32_t* trailer_bytes, void** ready)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queued_count_ == 0) {
            return false;
        }
        const std::uint32_t value = take(queued_ring_, queued_head_, queued_count_);
        *index = value;
        *bytes = records_[value].bytes;
        *trailer_bytes = records_[value].trailer_bytes;
        *ready = records_[value].ready;
        return true;
    }

    void mark_inflight(std::uint32_t index, std::uint32_t psn, std::uint32_t bytes)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& record = records_[index];
        if (record.state != State::Loaned && record.state != State::Queued) {
            return;
        }
        record.state = State::InFlight;
        record.bytes = bytes;
        record.psn = psn;
        ++inflight_;
    }

    // The work request was not accepted, or the frame was refused before post.
    // The slot is Loaned, or Queued and already removed from the send ring.
    void abandon(std::uint32_t index, bool count_error)
    {
        std::shared_ptr<Pool> dropped;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (index >= slot_count_) {
                return;
            }
            const State state = records_[index].state;
            if (state != State::Loaned && state != State::Queued) {
                return;
            }
            dropped = recycle_locked(index, count_error);
        }
    }

    // True when this completion retired an in-flight slot. A late success
    // after a timeout still counts bytes and frames.
    bool retire(std::uint64_t wr_id, bool success, bool* success_out)
    {
        std::shared_ptr<Pool> dropped;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (wr_id >= slot_count_) {
                return false;
            }
            const auto index = static_cast<std::uint32_t>(wr_id);
            auto& record = records_[index];
            if (record.state != State::InFlight) {
                return false;
            }
            if (success) {
                stats_.bytes += record.bytes;
                ++stats_.frames;
                stats_.last_psn = record.psn;
            } else {
                ++stats_.errors;
            }
            if (inflight_ > 0) {
                --inflight_;
            }
            dropped = recycle_locked(index, false);
        }
        if (success_out != nullptr) {
            *success_out = success;
        }
        return true;
    }

    std::uint32_t drop_queued()
    {
        std::uint32_t dropped_count = 0;
        for (;;) {
            std::uint32_t index = 0;
            std::uint32_t bytes = 0;
            std::uint32_t trailer_bytes = 0;
            void* ready = nullptr;
            if (!pop_queued(&index, &bytes, &trailer_bytes, &ready)) {
                break;
            }
            abandon(index, true);
            ++dropped_count;
        }
        return dropped_count;
    }

    void retire_inflight()
    {
        for (;;) {
            std::shared_ptr<Pool> dropped;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                std::uint32_t index = 0;
                for (; index < slot_count_; ++index) {
                    if (records_[index].state == State::InFlight) {
                        break;
                    }
                }
                if (index == slot_count_) {
                    return;
                }
                if (inflight_ > 0) {
                    --inflight_;
                }
                dropped = recycle_locked(index, false);
            }
        }
    }

    void count_inflight_errors()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.errors += inflight_;
    }

    std::uint32_t inflight() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return inflight_;
    }

    Stats stats() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_;
    }

    void add_error()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++stats_.errors;
    }

    std::mutex& mutex()
    {
        return mutex_;
    }

    std::condition_variable& wake()
    {
        return wake_;
    }

    // Caller holds mutex().
    bool has_free() const
    {
        return free_count_ != 0;
    }

    // Caller holds mutex().
    bool has_queued() const
    {
        return queued_count_ != 0;
    }

    std::uint32_t queued() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return queued_count_;
    }

private:
    Pool(std::uint32_t slot_count, void* base, std::size_t stride, std::uint32_t capacity)
        : slot_count_(slot_count)
        , base_(static_cast<std::uint8_t*>(base))
        , stride_(stride)
        , capacity_(capacity)
        , records_(slot_count)
        , free_ring_(slot_count)
        , queued_ring_(slot_count)
    {
        for (std::uint32_t index = 0; index < slot_count_; ++index) {
            free_ring_[index] = index;
        }
        free_count_ = slot_count_;
    }

    std::optional<std::uint32_t> claim_locked()
    {
        if (free_count_ == 0) {
            return std::nullopt;
        }
        const std::uint32_t index = take(free_ring_, free_head_, free_count_);
        records_[index].state = State::Loaned;
        records_[index].bytes = 0;
        records_[index].ready = nullptr;
        records_[index].loan = shared_from_this();
        return index;
    }

    std::shared_ptr<Pool> recycle_locked(std::uint32_t index, bool count_error)
    {
        auto& record = records_[index];
        record.state = State::Free;
        record.bytes = 0;
        record.ready = nullptr;
        if (count_error) {
            ++stats_.errors;
        }
        push(free_ring_, free_head_, free_count_, index);
        wake_.notify_all();
        return std::move(record.loan);
    }

    void push(std::vector<std::uint32_t>& ring, std::uint32_t head, std::uint32_t& count,
        std::uint32_t value)
    {
        ring[(head + count) % slot_count_] = value;
        ++count;
    }

    std::uint32_t take(std::vector<std::uint32_t>& ring, std::uint32_t& head, std::uint32_t& count)
    {
        const std::uint32_t value = ring[head];
        head = (head + 1) % slot_count_;
        --count;
        return value;
    }

    struct Record {
        State state = State::Free;
        std::uint32_t bytes = 0;
        std::uint32_t trailer_bytes = 0;
        std::uint32_t psn = 0;
        void* ready = nullptr;
        std::shared_ptr<Pool> loan;
    };

public:
    // Keeps the caller's allocation alive while any loan still holds this pool.
    std::shared_ptr<void> owner;

private:
    std::uint32_t slot_count_ = 0;
    std::uint8_t* base_ = nullptr;
    std::size_t stride_ = 0;
    std::uint32_t capacity_ = 0;
    std::vector<Record> records_;
    std::vector<std::uint32_t> free_ring_;
    std::vector<std::uint32_t> queued_ring_;
    std::uint32_t free_head_ = 0;
    std::uint32_t free_count_ = 0;
    std::uint32_t queued_head_ = 0;
    std::uint32_t queued_count_ = 0;
    std::uint32_t inflight_ = 0;
    Stats stats_;
    mutable std::mutex mutex_;
    std::condition_variable wake_;
};

} // namespace hololink::operators::frame_tx

#endif /* SRC_HOLOLINK_OPERATORS_CPU_ROCE_TRANSCEIVER_CPU_ROCE_FRAME_TRANSMITTER_INTERNAL_HPP */
