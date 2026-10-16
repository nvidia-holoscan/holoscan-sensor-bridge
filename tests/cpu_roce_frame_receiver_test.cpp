/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Publication and claim tests. FailedStartIsOneShot calls start() with an
 * empty device name, which fails before a RoCE device is opened.
 */

#include "gtest/gtest.h"

#include <sched.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include <hololink/core/frame_metadata_parser.hpp>
#include <hololink/operators/cpu_roce_transceiver/cpu_roce_frame_receiver_internal.hpp>

namespace hololink::operators {

struct CpuRoceFrameReceiverTestAccess {
    static void publish(CpuRoceFrameReceiver& receiver, std::uint32_t slot,
        std::optional<std::uint32_t> immediate_data)
    {
        receiver.publish(slot, immediate_data);
    }

    static void on_rx_completion(CpuRoceFrameReceiver& receiver,
        std::optional<std::uint32_t> immediate_data)
    {
        receiver.on_rx_completion(immediate_data);
    }

    static bool slot_in_use(const CpuRoceFrameReceiver& receiver, std::uint32_t slot)
    {
        using Sync = std::remove_reference_t<decltype(*receiver.slots_)>;
        return slot < receiver.slots_->records.size()
            && receiver.slots_->records[slot].state.load(std::memory_order_acquire)
            == Sync::Leased;
    }

    // True while get_frame() is inside its wait.
    static bool get_frame_blocked(const CpuRoceFrameReceiver& receiver)
    {
        return receiver.slots_->get_frame_active.load(std::memory_order_acquire);
    }

    static std::optional<int> poll_cpu(const CpuRoceFrameReceiver& receiver)
    {
        return receiver.config_.poll_cpu;
    }

    static bool is_stopped(const CpuRoceFrameReceiver& receiver)
    {
        return receiver.slots_->stopping;
    }

    static void mark_connected(CpuRoceFrameReceiver& receiver)
    {
        receiver.connected_ = true;
    }

    static void mark_polling(CpuRoceFrameReceiver& receiver)
    {
        receiver.polling_ = true;
    }

    static void take_mailbox(CpuRoceFrameReceiver& receiver)
    {
        receiver.slots_->mailbox.exchange(0, std::memory_order_acq_rel);
    }

    static void store_failure(CpuRoceFrameReceiver& receiver)
    {
        using Sync = std::remove_reference_t<decltype(*receiver.slots_)>;
        receiver.slots_->failure = std::make_exception_ptr(
            std::runtime_error("stored failure"));
        receiver.slots_->flags.fetch_or(
            Sync::kFlagFailure, std::memory_order_release);
    }

    static void set_overwritten(CpuRoceFrameReceiver& receiver, std::uint32_t value)
    {
        receiver.slots_->overwritten.store(value, std::memory_order_relaxed);
    }

    static void set_invalid_immediates(CpuRoceFrameReceiver& receiver, std::uint32_t value)
    {
        receiver.slots_->invalid_immediates.store(value, std::memory_order_relaxed);
    }

    static void monitor(CpuRoceFrameReceiver& receiver)
    {
        receiver.monitor();
    }

    // True when close() has already run. setup() checks that flag before it
    // opens a device, so this stays off the NIC.
    static bool transceiver_was_closed(CpuRoceFrameReceiver& receiver)
    {
        try {
            receiver.transceiver_->setup();
        } catch (const std::logic_error& error) {
            return std::string(error.what()).find("after close()") != std::string::npos;
        }
        return false;
    }
};

} // namespace hololink::operators

namespace {

using hololink::operators::CpuRoceFrameReceiver;
using hololink::operators::CpuRoceFrameReceiverTestAccess;
using hololink::transport::roce::DmaBufMemory;
using hololink::transport::roce::GpuPointerMemory;
using hololink::transport::roce::HostMemory;

CpuRoceFrameReceiver::Config host_config(std::vector<std::uint8_t>& storage,
    std::size_t metadata_size)
{
    constexpr std::size_t kPayload = 32;
    constexpr std::uint32_t kSlots = 4;
    const std::size_t stride = 128 + (metadata_size == 0 ? 0 : 128);
    storage.assign(stride * kSlots, 0);
    CpuRoceFrameReceiver::Config config;
    config.ibv_name = "unused";
    config.payload_memory = HostMemory { storage.data(), storage.size() };
    config.payload_size = kPayload;
    config.slot_count = kSlots;
    config.metadata_size = metadata_size;
    return config;
}

void store_be16(std::uint8_t* out, std::uint16_t value)
{
    out[0] = static_cast<std::uint8_t>(value >> 8);
    out[1] = static_cast<std::uint8_t>(value);
}

void store_be32(std::uint8_t* out, std::uint32_t value)
{
    out[0] = static_cast<std::uint8_t>(value >> 24);
    out[1] = static_cast<std::uint8_t>(value >> 16);
    out[2] = static_cast<std::uint8_t>(value >> 8);
    out[3] = static_cast<std::uint8_t>(value);
}

void store_be64(std::uint8_t* out, std::uint64_t value)
{
    for (int shift = 56; shift >= 0; shift -= 8) {
        *out++ = static_cast<std::uint8_t>(value >> shift);
    }
}

void write_metadata(std::uint8_t* trailer, std::uint16_t frame_number)
{
    std::memset(trailer, 0, 128);
    store_be16(trailer + 34, frame_number);
}

} // namespace

TEST(CpuRoceFrameReceiver, RejectsACreditCountOutsideTheRange)
{
    std::vector<std::uint8_t> storage;
    auto zero = host_config(storage, 0);
    zero.receive_credit_count = 0;
    EXPECT_THROW(
        CpuRoceFrameReceiver { std::move(zero) }, std::invalid_argument);
    auto huge = host_config(storage, 0);
    huge.receive_credit_count = 4097;
    EXPECT_THROW(
        CpuRoceFrameReceiver { std::move(huge) }, std::invalid_argument);
}

TEST(CpuRoceFrameReceiver, QuiescedMonitorFailureIsStored)
{
    std::vector<std::uint8_t> storage;
    CpuRoceFrameReceiver receiver(host_config(storage, 0));
    receiver.quiesce();
    CpuRoceFrameReceiverTestAccess::monitor(receiver);
    EXPECT_TRUE(static_cast<bool>(receiver.failure()));
}

TEST(CpuRoceFrameReceiver, DropsAnImmediateOutsideTheRing)
{
    std::vector<std::uint8_t> storage;
    CpuRoceFrameReceiver receiver(host_config(storage, 0));
    CpuRoceFrameReceiverTestAccess::on_rx_completion(receiver, 4u);
    EXPECT_FALSE(receiver.failure());
    EXPECT_FALSE(receiver.get_frame(std::chrono::microseconds(0)).has_value());
    EXPECT_FALSE(receiver.get_frame(std::chrono::microseconds(-1)).has_value());
    EXPECT_EQ(receiver.completion_count(), 0u);
    EXPECT_EQ(receiver.invalid_immediates(), 1u);

    CpuRoceFrameReceiverTestAccess::on_rx_completion(receiver, (0x22u << 12) | 1u);
    auto frame = receiver.get_frame(std::chrono::milliseconds(0));
    ASSERT_TRUE(frame.has_value());
    EXPECT_EQ(frame->slot(), 1u);
    EXPECT_EQ(frame->psn(), 0x22u);
    EXPECT_EQ(frame->invalid_immediates(), 1u);
}

TEST(CpuRoceFrameReceiver, MissingImmediateIsTerminal)
{
    std::vector<std::uint8_t> storage;
    CpuRoceFrameReceiver receiver(host_config(storage, 0));
    EXPECT_THROW(
        CpuRoceFrameReceiverTestAccess::on_rx_completion(receiver, std::nullopt),
        std::runtime_error);
    EXPECT_TRUE(static_cast<bool>(receiver.failure()));
    EXPECT_FALSE(receiver.get_frame(std::chrono::milliseconds(0)).has_value());
}

TEST(CpuRoceFrameReceiver, RejectsUnsupportedMetadata)
{
    std::vector<std::uint8_t> storage;
    auto config = host_config(storage, 0);
    config.metadata_size = 64;
    EXPECT_THROW(CpuRoceFrameReceiver { std::move(config) }, std::invalid_argument);

    auto dma = host_config(storage, 0);
    dma.metadata_size = 128;
    dma.payload_memory = DmaBufMemory { storage.data(), 3, 0, storage.size() };
    EXPECT_THROW(CpuRoceFrameReceiver { std::move(dma) }, std::invalid_argument);

    auto gpu = host_config(storage, 0);
    gpu.metadata_size = 128;
    gpu.payload_memory = GpuPointerMemory { storage.data(), storage.size() };
    EXPECT_THROW(CpuRoceFrameReceiver { std::move(gpu) }, std::invalid_argument);
}

TEST(CpuRoceFrameReceiver, AcceptsDeviceMetadataWhenGivenACudaContext)
{
    // Construction only. Publishing would copy from the device.
    int context_sentinel = 0;
    std::vector<std::uint8_t> storage;
    auto dma = host_config(storage, 128);
    dma.cuda_context = &context_sentinel;
    dma.payload_memory = DmaBufMemory { storage.data(), 3, 0, storage.size() };
    EXPECT_NO_THROW(CpuRoceFrameReceiver { std::move(dma) });

    auto gpu = host_config(storage, 128);
    gpu.cuda_context = &context_sentinel;
    gpu.payload_memory = GpuPointerMemory { storage.data(), storage.size() };
    EXPECT_NO_THROW(CpuRoceFrameReceiver { std::move(gpu) });
}

TEST(CpuRoceFrameReceiver, AcceptsAGpuPointerWithMetadataOff)
{
    std::vector<std::uint8_t> storage;
    auto config = host_config(storage, 0);
    config.payload_memory = GpuPointerMemory { storage.data(), storage.size() };
    CpuRoceFrameReceiver receiver(std::move(config));
    CpuRoceFrameReceiverTestAccess::publish(receiver, 1, 0x11);
    auto frame = receiver.get_frame(std::chrono::milliseconds(0));
    ASSERT_TRUE(frame.has_value());
    EXPECT_EQ(frame->payload(), storage.data() + receiver.stride());
}

TEST(CpuRoceFrameReceiver, DoesNotReadADeviceTrailerUnlessAsked)
{
    std::vector<std::uint8_t> storage;
    auto config = host_config(storage, 128);
    config.host_trailer = false;
    config.payload_memory = GpuPointerMemory { storage.data(), storage.size() };
    write_metadata(storage.data() + 128, 0x1234);
    CpuRoceFrameReceiver receiver(std::move(config));
    CpuRoceFrameReceiverTestAccess::publish(receiver, 0, std::nullopt);
    auto frame = receiver.get_frame(std::chrono::milliseconds(0));
    ASSERT_TRUE(frame.has_value());
    EXPECT_FALSE(frame->metadata().has_value());
    EXPECT_EQ(receiver.stats().rx_bytes, 0u);
}

TEST(CpuRoceFrameReceiver, KeepsTheLatestFrameAndCountsDrops)
{
    std::vector<std::uint8_t> storage;
    CpuRoceFrameReceiver receiver(host_config(storage, 0));
    CpuRoceFrameReceiverTestAccess::publish(receiver, 1, 0x11);
    CpuRoceFrameReceiverTestAccess::publish(receiver, 2, 0x22);

    auto frame = receiver.get_frame(std::chrono::milliseconds(0));
    ASSERT_TRUE(frame.has_value());
    EXPECT_EQ(frame->slot(), 2u);
    EXPECT_EQ(frame->immediate_data(), 0x22u);
    EXPECT_EQ(frame->dropped(), 1u);
    EXPECT_EQ(receiver.dropped(), 1u);
    EXPECT_EQ(frame->dropped(), receiver.dropped());
    EXPECT_FALSE(frame->metadata().has_value());
    EXPECT_TRUE(CpuRoceFrameReceiverTestAccess::slot_in_use(receiver, 2));

    frame.reset();
    EXPECT_FALSE(CpuRoceFrameReceiverTestAccess::slot_in_use(receiver, 2));
    EXPECT_FALSE(receiver.get_frame(std::chrono::milliseconds(0)).has_value());
    EXPECT_FALSE(receiver.failure());
}

TEST(CpuRoceFrameReceiver, LeavesTheTrailerUnreadWhenHostTrailerIsOff)
{
    std::vector<std::uint8_t> storage;
    auto config = host_config(storage, 128);
    config.host_trailer = false;
    CpuRoceFrameReceiver receiver(std::move(config));
    write_metadata(storage.data() + 128, 0x1234);
    store_be64(storage.data() + 128 + 24, 100);
    CpuRoceFrameReceiverTestAccess::publish(receiver, 0, std::nullopt);

    auto frame = receiver.get_frame(std::chrono::milliseconds(0));
    ASSERT_TRUE(frame.has_value());
    EXPECT_FALSE(frame->metadata().has_value());
    EXPECT_EQ(receiver.stats().frame_number, 0u);
    EXPECT_EQ(receiver.stats().rx_bytes, 0u);
    EXPECT_EQ(receiver.completion_count(), 1u);
}

TEST(CpuRoceFrameReceiver, BindsHostMetadataAtPublication)
{
    std::vector<std::uint8_t> storage;
    CpuRoceFrameReceiver receiver(host_config(storage, 128));
    write_metadata(storage.data() + 128, 0x1234);
    CpuRoceFrameReceiverTestAccess::publish(receiver, 0, std::nullopt);
    write_metadata(storage.data() + 128, 0x5678);

    auto frame = receiver.get_frame(std::chrono::milliseconds(0));
    ASSERT_TRUE(frame.has_value());
    ASSERT_TRUE(frame->metadata().has_value());
    EXPECT_EQ(frame->metadata()->frame_number, 0x1234);
}

TEST(CpuRoceFrameReceiver, LeasedOverwriteIsCounted)
{
    std::vector<std::uint8_t> storage;
    CpuRoceFrameReceiver receiver(host_config(storage, 128));
    constexpr std::size_t kStride = 256;
    auto* first = storage.data() + kStride + 128;
    write_metadata(first, 10);
    store_be64(first + 24, 100);
    store_be32(first + 4, 0x111);
    CpuRoceFrameReceiverTestAccess::publish(receiver, 1, std::nullopt);
    auto held = receiver.get_frame(std::chrono::milliseconds(0));
    ASSERT_TRUE(held.has_value());
    EXPECT_EQ(held->session_frame_number(), 1u);

    auto* again = storage.data() + kStride + 128;
    write_metadata(again, 12);
    store_be64(again + 24, 40);
    store_be32(again + 4, 0x222);
    CpuRoceFrameReceiverTestAccess::publish(receiver, 1, std::nullopt);
    EXPECT_FALSE(receiver.failure());
    EXPECT_EQ(receiver.overwritten(), 1u);
    EXPECT_EQ(receiver.completion_count(), 2u);
    const auto after = receiver.stats();
    EXPECT_EQ(after.completions, 2u);
    EXPECT_EQ(after.frame_number, 3u);
    EXPECT_EQ(after.rx_bytes, 140u);
    EXPECT_EQ(after.psn, 0x222u);
    EXPECT_EQ(after.dropped, 0u);
    EXPECT_EQ(held->session_frame_number(), 1u);
    EXPECT_FALSE(receiver.get_frame(std::chrono::milliseconds(0)).has_value());
    EXPECT_TRUE(CpuRoceFrameReceiverTestAccess::slot_in_use(receiver, 1));

    CpuRoceFrameReceiverTestAccess::publish(receiver, 2, std::nullopt);
    auto next = receiver.get_frame(std::chrono::milliseconds(0));
    ASSERT_TRUE(next.has_value());
    EXPECT_EQ(next->slot(), 2u);
    EXPECT_EQ(next->overwritten(), 1u);
    EXPECT_EQ(next->completion_count(), 3u);
    EXPECT_FALSE(receiver.failure());
}

struct WaiterGuard {
    CpuRoceFrameReceiver& receiver;
    std::thread& waiter;
    ~WaiterGuard()
    {
        try {
            receiver.stop();
        } catch (...) {
        }
        if (waiter.joinable()) {
            waiter.join();
        }
    }
};

bool wait_until_get_frame_blocked(const CpuRoceFrameReceiver& receiver)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!CpuRoceFrameReceiverTestAccess::get_frame_blocked(receiver)) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

TEST(CpuRoceFrameReceiver, RejectsASecondGetFrame)
{
    std::vector<std::uint8_t> storage;
    CpuRoceFrameReceiver receiver(host_config(storage, 0));
    std::exception_ptr error;
    std::thread waiter([&] {
        try {
            receiver.get_frame(std::chrono::seconds(2));
        } catch (...) {
            error = std::current_exception();
        }
    });
    [[maybe_unused]] WaiterGuard guard { receiver, waiter };
    ASSERT_TRUE(wait_until_get_frame_blocked(receiver))
        << "waiter did not block in get_frame";
    EXPECT_THROW(receiver.get_frame(std::chrono::milliseconds(0)), std::logic_error);
    CpuRoceFrameReceiverTestAccess::publish(receiver, 0, std::nullopt);
    waiter.join();
    EXPECT_FALSE(error);
}

TEST(CpuRoceFrameReceiver, StopWakesGetFrame)
{
    std::vector<std::uint8_t> storage;
    CpuRoceFrameReceiver receiver(host_config(storage, 0));
    std::optional<bool> got_frame;
    std::thread waiter([&] {
        got_frame = receiver.get_frame(std::chrono::seconds(2)).has_value();
    });
    [[maybe_unused]] WaiterGuard guard { receiver, waiter };
    ASSERT_TRUE(wait_until_get_frame_blocked(receiver))
        << "waiter did not block in get_frame";
    EXPECT_FALSE(receiver.frames_ready());
    receiver.stop();
    waiter.join();
    EXPECT_EQ(got_frame, false);
    EXPECT_TRUE(receiver.frames_ready());
    receiver.stop();
}

TEST(CpuRoceFrameReceiver, SameSlotReplacementCountsOneDrop)
{
    std::vector<std::uint8_t> storage;
    CpuRoceFrameReceiver receiver(host_config(storage, 0));
    CpuRoceFrameReceiverTestAccess::publish(receiver, 0, 0x11);
    CpuRoceFrameReceiverTestAccess::publish(receiver, 0, 0x22);

    auto frame = receiver.get_frame(std::chrono::milliseconds(0));
    ASSERT_TRUE(frame.has_value());
    EXPECT_EQ(frame->slot(), 0u);
    EXPECT_EQ(frame->immediate_data(), 0x22u);
    EXPECT_EQ(frame->dropped(), 1u);
    EXPECT_FALSE(receiver.failure());
}

TEST(CpuRoceFrameReceiver, OrphanedReadyAndMailboxEachCountADrop)
{
    std::vector<std::uint8_t> storage;
    CpuRoceFrameReceiver receiver(host_config(storage, 0));
    CpuRoceFrameReceiverTestAccess::publish(receiver, 0, 0x11);
    CpuRoceFrameReceiverTestAccess::take_mailbox(receiver);
    CpuRoceFrameReceiverTestAccess::publish(receiver, 1, 0x22);
    CpuRoceFrameReceiverTestAccess::publish(receiver, 0, 0x33);

    EXPECT_EQ(receiver.dropped(), 2u);
    auto frame = receiver.get_frame(std::chrono::milliseconds(0));
    ASSERT_TRUE(frame.has_value());
    EXPECT_EQ(frame->slot(), 0u);
    EXPECT_EQ(frame->immediate_data(), 0x33u);
    EXPECT_EQ(frame->dropped(), 2u);
}

TEST(CpuRoceFrameReceiver, PublishAfterFailureRethrowsWithoutDeadlock)
{
    std::vector<std::uint8_t> storage;
    CpuRoceFrameReceiver receiver(host_config(storage, 0));
    CpuRoceFrameReceiverTestAccess::store_failure(receiver);
    EXPECT_THROW(CpuRoceFrameReceiverTestAccess::publish(receiver, 0, std::nullopt),
        std::runtime_error);
    EXPECT_TRUE(static_cast<bool>(receiver.failure()));
    EXPECT_EQ(receiver.completion_count(), 0u);
}

TEST(CpuRoceFrameReceiver, OutOfRangeSlotIsTerminal)
{
    std::vector<std::uint8_t> storage;
    CpuRoceFrameReceiver receiver(host_config(storage, 0));
    EXPECT_THROW(CpuRoceFrameReceiverTestAccess::publish(receiver, 4, std::nullopt),
        std::out_of_range);
    EXPECT_TRUE(receiver.failure());
    EXPECT_EQ(receiver.completion_count(), 0u);
    EXPECT_FALSE(receiver.get_frame(std::chrono::milliseconds(0)).has_value());
}

TEST(CpuRoceFrameReceiver, FrameOutlivesTheReceiver)
{
    std::vector<std::uint8_t> storage;
    std::optional<hololink::operators::HsbFrame> frame;
    {
        CpuRoceFrameReceiver receiver(host_config(storage, 0));
        CpuRoceFrameReceiverTestAccess::publish(receiver, 1, 0x11);
        frame = receiver.get_frame(std::chrono::milliseconds(0));
        ASSERT_TRUE(frame.has_value());
    }
    EXPECT_EQ(frame->slot(), 1u);
    EXPECT_EQ(frame->immediate_data(), 0x11u);
    frame.reset();
}

TEST(CpuRoceFrameReceiver, FailedStartIsOneShot)
{
    std::vector<std::uint8_t> storage;
    auto config = host_config(storage, 0);
    config.ibv_name.clear();
    CpuRoceFrameReceiver receiver(std::move(config));
    EXPECT_THROW(receiver.start(), std::invalid_argument);
    EXPECT_THROW(receiver.start(), std::logic_error);
}

TEST(CpuRoceFrameReceiver, ConnectAndStartRejectADeadReceiver)
{
    std::vector<std::uint8_t> storage;
    CpuRoceFrameReceiver receiver(host_config(storage, 0));
    CpuRoceFrameReceiverTestAccess::mark_connected(receiver);
    CpuRoceFrameReceiverTestAccess::mark_polling(receiver);
    EXPECT_NO_THROW(receiver.connect());
    EXPECT_NO_THROW(receiver.start());

    CpuRoceFrameReceiverTestAccess::store_failure(receiver);
    EXPECT_THROW(receiver.connect(), std::logic_error);
    EXPECT_THROW(receiver.start(), std::logic_error);
}

TEST(CpuRoceFrameReceiver, ReportsTheConfigureRoceStride)
{
    std::vector<std::uint8_t> storage(128 * 4, 0);
    CpuRoceFrameReceiver::Config config;
    config.ibv_name = "unused";
    config.payload_memory = HostMemory { storage.data(), storage.size() };
    config.payload_size = 32;
    config.slot_count = 4;
    {
        CpuRoceFrameReceiver receiver(config);
        EXPECT_EQ(receiver.stride(), 128u);
    }

    storage.assign(384 * 2, 0);
    config.payload_memory = HostMemory { storage.data(), storage.size() };
    config.payload_size = 129;
    config.slot_count = 2;
    CpuRoceFrameReceiver receiver(config);
    EXPECT_EQ(receiver.stride(), 384u);
}

TEST(CpuRoceFrameReceiver, ReportsPsnAndKeepsAMetadataMismatch)
{
    std::vector<std::uint8_t> storage;
    CpuRoceFrameReceiver receiver(host_config(storage, 128));
    // Slot 0's trailer begins at metadata_offset, which is one page for this payload.
    write_metadata(storage.data() + 128, 7);
    store_be32(storage.data() + 128 + 4, 0x11111);

    CpuRoceFrameReceiverTestAccess::publish(receiver, 0, std::nullopt);
    auto without_immediate = receiver.get_frame(std::chrono::milliseconds(0));
    ASSERT_TRUE(without_immediate.has_value());
    EXPECT_FALSE(without_immediate->psn().has_value());
    without_immediate.reset();

    const std::uint32_t immediate = 0x22u << 12;
    CpuRoceFrameReceiverTestAccess::publish(receiver, 0, immediate);
    auto frame = receiver.get_frame(std::chrono::milliseconds(0));
    ASSERT_TRUE(frame.has_value());
    EXPECT_EQ(frame->psn(), 0x22u);
    ASSERT_TRUE(frame->metadata().has_value());
    EXPECT_EQ(frame->metadata()->psn, 0x11111u);
    EXPECT_FALSE(receiver.failure());
}

TEST(CpuRoceFrameReceiver, ReleaseWithoutSynchronizeClearsTheLeaseOnce)
{
    std::vector<std::uint8_t> storage;
    CpuRoceFrameReceiver receiver(host_config(storage, 0));
    CpuRoceFrameReceiverTestAccess::publish(receiver, 2, std::nullopt);
    auto frame = receiver.get_frame(std::chrono::milliseconds(0));
    ASSERT_TRUE(frame.has_value());

    frame->release();
    EXPECT_EQ(frame->payload(), nullptr);
    EXPECT_FALSE(CpuRoceFrameReceiverTestAccess::slot_in_use(receiver, 2));
    EXPECT_FALSE(receiver.failure());

    CpuRoceFrameReceiverTestAccess::publish(receiver, 3, std::nullopt);
    auto again = receiver.get_frame(std::chrono::milliseconds(0));
    ASSERT_TRUE(again.has_value());
    int calls = 0;
    again->release([&] { ++calls; });
    EXPECT_EQ(calls, 1);
    EXPECT_FALSE(CpuRoceFrameReceiverTestAccess::slot_in_use(receiver, 3));
    again->release([&] { ++calls; });
    EXPECT_EQ(calls, 1);
}

TEST(CpuRoceFrameReceiver, FailedReleaseClosesAndWakes)
{
    std::vector<std::uint8_t> storage;
    CpuRoceFrameReceiver receiver(host_config(storage, 0));
    CpuRoceFrameReceiverTestAccess::publish(receiver, 1, std::nullopt);
    auto frame = receiver.get_frame(std::chrono::milliseconds(0));
    ASSERT_TRUE(frame.has_value());
    CpuRoceFrameReceiverTestAccess::mark_connected(receiver);
    EXPECT_EQ(receiver.qp_number(), 0u);
    EXPECT_THROW(receiver.path_mtu(), std::logic_error);

    std::optional<bool> got_frame;
    std::thread waiter([&] {
        got_frame = receiver.get_frame(std::chrono::seconds(2)).has_value();
    });
    [[maybe_unused]] WaiterGuard guard { receiver, waiter };
    ASSERT_TRUE(wait_until_get_frame_blocked(receiver))
        << "waiter did not block in get_frame";

    int calls = 0;
    EXPECT_THROW(frame->release([&] {
        ++calls;
        throw std::runtime_error("sync failed");
    }),
        std::runtime_error);
    EXPECT_EQ(calls, 1);
    frame->release([&] { ++calls; });
    EXPECT_EQ(calls, 1);

    waiter.join();
    EXPECT_EQ(got_frame, false);
    EXPECT_TRUE(receiver.failure());
    EXPECT_TRUE(receiver.frames_ready());
    EXPECT_TRUE(CpuRoceFrameReceiverTestAccess::slot_in_use(receiver, 1));
    EXPECT_TRUE(CpuRoceFrameReceiverTestAccess::transceiver_was_closed(receiver));
    EXPECT_THROW(receiver.qp_number(), std::logic_error);
    EXPECT_THROW(receiver.path_mtu(), std::logic_error);
    EXPECT_FALSE(receiver.get_frame(std::chrono::milliseconds(0)).has_value());
}

TEST(CpuRoceFrameReceiver, FailedReleaseAfterDestroyIsHarmless)
{
    std::vector<std::uint8_t> storage;
    std::optional<hololink::operators::HsbFrame> frame;
    {
        CpuRoceFrameReceiver receiver(host_config(storage, 0));
        CpuRoceFrameReceiverTestAccess::publish(receiver, 1, std::nullopt);
        frame = receiver.get_frame(std::chrono::milliseconds(0));
        ASSERT_TRUE(frame.has_value());
    }
    EXPECT_THROW(frame->release([] { throw std::runtime_error("sync failed"); }),
        std::runtime_error);
    frame.reset();
}

TEST(CpuRoceFrameReceiver, PayloadOwnerOutlivesTheReceiver)
{
    auto storage = std::make_shared<std::vector<std::uint8_t>>();
    auto config = host_config(*storage, 0);
    config.payload_owner = storage;
    std::optional<hololink::operators::HsbFrame> frame;
    {
        CpuRoceFrameReceiver receiver(std::move(config));
        CpuRoceFrameReceiverTestAccess::publish(receiver, 0, std::nullopt);
        frame = receiver.get_frame(std::chrono::milliseconds(0));
        ASSERT_TRUE(frame.has_value());
    }
    storage.reset();
    ASSERT_NE(frame->payload(), nullptr);
    frame->payload()[0] = 1;
    frame.reset();
}

TEST(CpuRoceFrameReceiver, FailedReleaseKeepsThePayloadOwner)
{
    auto freed = std::make_shared<bool>(false);
    auto storage = std::shared_ptr<std::vector<std::uint8_t>>(
        new std::vector<std::uint8_t>(),
        [freed](std::vector<std::uint8_t>* memory) {
            *freed = true;
            delete memory;
        });
    auto config = host_config(*storage, 0);
    config.payload_owner = storage;
    std::optional<hololink::operators::HsbFrame> frame;
    {
        CpuRoceFrameReceiver receiver(std::move(config));
        CpuRoceFrameReceiverTestAccess::publish(receiver, 0, std::nullopt);
        frame = receiver.get_frame(std::chrono::milliseconds(0));
        ASSERT_TRUE(frame.has_value());
        EXPECT_THROW(frame->release([] { throw std::runtime_error("sync failed"); }),
            std::runtime_error);
        EXPECT_EQ(frame->payload(), nullptr);
    }
    storage.reset();
    frame.reset();
    EXPECT_FALSE(*freed);
}

TEST(CpuRoceFrameReceiver, CountsAcceptedFramesBeforeTheyAreFetched)
{
    std::vector<std::uint8_t> storage;
    CpuRoceFrameReceiver receiver(host_config(storage, 128));
    constexpr std::size_t kStride = 256;
    auto* first = storage.data() + 128;
    write_metadata(first, 10);
    store_be64(first + 24, 100);
    store_be32(first + 4, 0xabc);
    CpuRoceFrameReceiverTestAccess::publish(receiver, 0, std::nullopt);

    const auto early = receiver.stats();
    EXPECT_EQ(early.frame_number, 1u);
    EXPECT_EQ(early.completions, 1u);
    EXPECT_EQ(early.rx_bytes, 100u);
    EXPECT_EQ(early.psn, 0xabcu);
    EXPECT_EQ(early.overwritten, 0u);

    auto* second = storage.data() + kStride + 128;
    write_metadata(second, 12);
    store_be64(second + 24, 40);
    CpuRoceFrameReceiverTestAccess::publish(receiver, 1, std::nullopt);
    const auto both = receiver.stats();
    EXPECT_EQ(both.frame_number, 3u);
    EXPECT_EQ(both.completions, 2u);
    EXPECT_EQ(both.rx_bytes, 140u);

    auto frame = receiver.get_frame(std::chrono::milliseconds(0));
    ASSERT_TRUE(frame.has_value());
    EXPECT_EQ(frame->session_frame_number(), 3u);
    EXPECT_EQ(frame->metadata()->frame_number, 12u);

    auto* third = storage.data() + kStride + 128;
    write_metadata(third, 13);
    store_be64(third + 24, 7);
    store_be32(third + 4, 0xdef);
    CpuRoceFrameReceiverTestAccess::publish(receiver, 1, std::nullopt);
    const auto overwritten = receiver.stats();
    EXPECT_EQ(overwritten.frame_number, 4u);
    EXPECT_EQ(overwritten.completions, 3u);
    EXPECT_EQ(overwritten.rx_bytes, 147u);
    EXPECT_EQ(overwritten.overwritten, 1u);
    EXPECT_EQ(overwritten.dropped, 1u);
    EXPECT_EQ(overwritten.psn, 0xdefu);
    EXPECT_FALSE(receiver.get_frame(std::chrono::milliseconds(0)).has_value());
}

TEST(CpuRoceFrameReceiver, QuiesceDropsAQueuedFrameAndLeavesTheQueuePair)
{
    std::vector<std::uint8_t> storage;
    CpuRoceFrameReceiver receiver(host_config(storage, 0));
    CpuRoceFrameReceiverTestAccess::publish(receiver, 1, 0x11);
    receiver.quiesce();
    receiver.quiesce();
    EXPECT_TRUE(receiver.frames_ready());
    EXPECT_FALSE(receiver.get_frame(std::chrono::milliseconds(0)).has_value());
    EXPECT_FALSE(CpuRoceFrameReceiverTestAccess::is_stopped(receiver));
    CpuRoceFrameReceiverTestAccess::publish(receiver, 2, 0x22);
    EXPECT_FALSE(receiver.get_frame(std::chrono::milliseconds(0)).has_value());
    receiver.stop();
    EXPECT_TRUE(CpuRoceFrameReceiverTestAccess::is_stopped(receiver));
    EXPECT_TRUE(CpuRoceFrameReceiverTestAccess::transceiver_was_closed(receiver));
}

TEST(CpuRoceFrameReceiver, CudaPayloadHostMappedIsHostMemory)
{
    std::vector<std::uint8_t> storage;
    auto config = host_config(storage, 0);
    config.payload_memory = hololink::transport::roce::TransceiverAllocation {};
    config.cuda_payload = CpuRoceFrameReceiver::Config::CudaPayload {
        storage.data(), storage.size(), true
    };
    CpuRoceFrameReceiver receiver(std::move(config));
    CpuRoceFrameReceiverTestAccess::publish(receiver, 0, 0x5);
    auto frame = receiver.get_frame(std::chrono::milliseconds(0));
    ASSERT_TRUE(frame.has_value());
    EXPECT_EQ(frame->payload(), storage.data());
}

TEST(CpuRoceFrameReceiver, ParserMatchesDeserializeMetadata)
{
    std::uint8_t trailer[128] = {};
    store_be32(trailer + 0, 0x11u);
    store_be32(trailer + 4, 0x22u);
    store_be32(trailer + 8, 0x33u);
    store_be64(trailer + 12, 0x44u);
    store_be32(trailer + 20, 0x55u);
    store_be64(trailer + 24, 0x66u);
    store_be16(trailer + 32, 0x77u);
    store_be16(trailer + 34, 0x1234u);
    store_be64(trailer + 36, 0x88u);
    store_be32(trailer + 44, 0x99u);

    const auto parsed = hololink::core::parse_frame_metadata(trailer, 128);
    const auto legacy = hololink::Hololink::deserialize_metadata(trailer, 128);
    EXPECT_EQ(parsed.flags, legacy.flags);
    EXPECT_EQ(parsed.psn, legacy.psn);
    EXPECT_EQ(parsed.crc, legacy.crc);
    EXPECT_EQ(parsed.frame_number, legacy.frame_number);
    EXPECT_EQ(parsed.timestamp_ns, legacy.timestamp_ns);
    EXPECT_EQ(parsed.timestamp_s, legacy.timestamp_s);
    EXPECT_EQ(parsed.bytes_written, legacy.bytes_written);
    EXPECT_EQ(parsed.metadata_ns, legacy.metadata_ns);
    EXPECT_EQ(parsed.metadata_s, legacy.metadata_s);
    EXPECT_EQ(parsed.flags, 0x11u);
    EXPECT_EQ(parsed.psn, 0x22u);
    EXPECT_EQ(parsed.crc, 0x33u);
    EXPECT_EQ(parsed.timestamp_s, 0x44u);
    EXPECT_EQ(parsed.timestamp_ns, 0x55u);
    EXPECT_EQ(parsed.bytes_written, 0x66u);
    EXPECT_EQ(parsed.frame_number, 0x1234u);
    EXPECT_EQ(parsed.metadata_s, 0x88u);
    EXPECT_EQ(parsed.metadata_ns, 0x99u);

    try {
        hololink::core::parse_frame_metadata(trailer, 4);
        FAIL() << "short buffer must throw";
    } catch (const std::runtime_error& error) {
        EXPECT_STREQ(error.what(), "Buffer underflow in metadata");
    }
    try {
        hololink::Hololink::deserialize_metadata(trailer, 4);
        FAIL() << "short buffer must throw";
    } catch (const std::runtime_error& error) {
        EXPECT_STREQ(error.what(), "Buffer underflow in metadata");
    }
}

struct ReceiveErrorRecord {
    int calls = 0;
    hololink::operators::CpuRoceFrameReceiver::ReceiveError error {};
    std::uint64_t total = 0;
};

void record_receive_error(void* context,
    hololink::operators::CpuRoceFrameReceiver::ReceiveError error,
    std::uint64_t total) noexcept
{
    auto* record = static_cast<ReceiveErrorRecord*>(context);
    ++record->calls;
    record->error = error;
    record->total = total;
}

TEST(CpuRoceFrameReceiver, ErrorHookSeesUnpublishedCountsOnce)
{
    std::vector<std::uint8_t> storage;
    auto config = host_config(storage, 0);
    ReceiveErrorRecord record;
    config.on_receive_error = record_receive_error;
    config.on_receive_error_context = &record;
    CpuRoceFrameReceiver receiver(std::move(config));

    CpuRoceFrameReceiverTestAccess::publish(receiver, 0, std::nullopt);
    EXPECT_EQ(record.calls, 0);
    auto held = receiver.get_frame(std::chrono::milliseconds(0));
    ASSERT_TRUE(held.has_value());

    CpuRoceFrameReceiverTestAccess::publish(receiver, 0, std::nullopt);
    EXPECT_EQ(record.calls, 1);
    EXPECT_EQ(record.error, CpuRoceFrameReceiver::ReceiveError::Overwrite);
    EXPECT_EQ(record.total, 1u);

    record = {};
    CpuRoceFrameReceiverTestAccess::on_rx_completion(receiver, 4u);
    EXPECT_EQ(record.calls, 1);
    EXPECT_EQ(record.error, CpuRoceFrameReceiver::ReceiveError::InvalidImmediate);
    EXPECT_EQ(record.total, 1u);
}

TEST(CpuRoceFrameReceiver, ErrorHookStopsAtTheCounterLimitAndAfterQuiesce)
{
    std::vector<std::uint8_t> storage;
    auto config = host_config(storage, 0);
    ReceiveErrorRecord record;
    config.on_receive_error = record_receive_error;
    config.on_receive_error_context = &record;
    CpuRoceFrameReceiver receiver(std::move(config));
    CpuRoceFrameReceiverTestAccess::set_overwritten(
        receiver, std::numeric_limits<std::uint32_t>::max());
    CpuRoceFrameReceiverTestAccess::set_invalid_immediates(
        receiver, std::numeric_limits<std::uint32_t>::max());

    CpuRoceFrameReceiverTestAccess::publish(receiver, 0, std::nullopt);
    auto held = receiver.get_frame(std::chrono::milliseconds(0));
    ASSERT_TRUE(held.has_value());
    CpuRoceFrameReceiverTestAccess::publish(receiver, 0, std::nullopt);
    CpuRoceFrameReceiverTestAccess::on_rx_completion(receiver, 4u);
    EXPECT_EQ(record.calls, 0);
    EXPECT_EQ(receiver.overwritten(), std::numeric_limits<std::uint32_t>::max());
    EXPECT_EQ(receiver.invalid_immediates(), std::numeric_limits<std::uint32_t>::max());

    receiver.quiesce();
    CpuRoceFrameReceiverTestAccess::set_overwritten(receiver, 0);
    CpuRoceFrameReceiverTestAccess::set_invalid_immediates(receiver, 0);
    CpuRoceFrameReceiverTestAccess::publish(receiver, 0, std::nullopt);
    CpuRoceFrameReceiverTestAccess::on_rx_completion(receiver, 4u);
    EXPECT_EQ(record.calls, 0);
}

TEST(CpuRoceFrameReceiver, BestEffortPollCpuStaysSetUntilThePoller)
{
    std::vector<std::uint8_t> storage;
    auto config = host_config(storage, 0);
    config.poll_cpu = CPU_SETSIZE;
    config.poll_cpu_best_effort = true;
    CpuRoceFrameReceiver receiver(std::move(config));
    const auto poll_cpu = CpuRoceFrameReceiverTestAccess::poll_cpu(receiver);
    ASSERT_TRUE(poll_cpu.has_value());
    EXPECT_EQ(*poll_cpu, CPU_SETSIZE);
}
