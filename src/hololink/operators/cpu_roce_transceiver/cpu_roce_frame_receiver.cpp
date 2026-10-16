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

#include "cpu_roce_frame_receiver.hpp"
#include "cpu_roce_frame_receiver_internal.hpp"

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <variant>

#include <climits>
#include <unistd.h>

#include <linux/futex.h>
#include <sys/syscall.h>

#include <cuda.h>

#include <hololink/core/frame_metadata_parser.hpp>
#include <hololink/transport/roce/cpu_roce_transceiver_internal.hpp>

namespace hololink::operators {

namespace {

    using hololink::transport::roce::CpuRoceTransceiver;
    using hololink::transport::roce::DmaBufMemory;
    using hololink::transport::roce::GpuPointerMemory;
    using hololink::transport::roce::HostMemory;

    constexpr int kMetadataEventSpinCount = 64;

    void check_cuda(CUresult result)
    {
        if (result == CUDA_SUCCESS) {
            return;
        }
        const char* error_name = "";
        cuGetErrorName(result, &error_name);
        const char* error_string = "";
        cuGetErrorString(result, &error_string);
        throw std::runtime_error(std::string("CUDA driver error ")
            + std::to_string(static_cast<int>(result)) + " ("
            + (error_name != nullptr ? error_name : "") + "): "
            + (error_string != nullptr ? error_string : ""));
    }

    std::size_t ring_bytes(std::size_t stride, std::uint32_t slot_count)
    {
        if (stride == 0 || slot_count == 0
            || stride > std::numeric_limits<std::size_t>::max() / slot_count) {
            throw std::invalid_argument(
                "CpuRoceFrameReceiver ring size is out of range");
        }
        return stride * slot_count;
    }

    struct CudaContextPush {
        explicit CudaContextPush(CUcontext context)
        {
            check_cuda(cuCtxPushCurrent(context));
        }
        ~CudaContextPush()
        {
            CUcontext popped = nullptr;
            cuCtxPopCurrent(&popped);
        }
        CudaContextPush(const CudaContextPush&) = delete;
        CudaContextPush& operator=(const CudaContextPush&) = delete;
    };

} // namespace

template <typename T, CUresult func(T)>
struct CudaDeleter {
    using pointer = T;
    void operator()(T value) const
    {
        func(value);
    }
};

using UniqueHost = std::unique_ptr<void, CudaDeleter<void*, &cuMemFreeHost>>;
using UniqueEvent = std::unique_ptr<CUevent, CudaDeleter<CUevent, &cuEventDestroy>>;
using UniqueStream = std::unique_ptr<CUstream, CudaDeleter<CUstream, &cuStreamDestroy>>;

constexpr std::size_t kProtocolPage = 128;

void HsbFrame::arm_wake(SlotSync& slots)
{
    slots.wake_seq.fetch_add(1, std::memory_order_release);
}

void HsbFrame::wake_waiters(SlotSync& slots)
{
    ::syscall(SYS_futex, reinterpret_cast<std::uint32_t*>(&slots.wake_seq),
        FUTEX_WAKE_PRIVATE, INT_MAX, nullptr, nullptr, 0);
}

void HsbFrame::begin_stats(SlotSync& slots)
{
    // A throw before end_stats leaves stats_seq odd and stats() spins.
    const auto sequence = slots.stats_seq.load(std::memory_order_relaxed);
    slots.stats_seq.store(sequence | 1u, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
}

void HsbFrame::end_stats(SlotSync& slots)
{
    const auto sequence = slots.stats_seq.load(std::memory_order_relaxed);
    slots.stats_seq.store(sequence + 1u, std::memory_order_release);
}

std::uint64_t pack_token(std::uint32_t generation, std::uint32_t slot)
{
    return (static_cast<std::uint64_t>(generation) << 32) | slot;
}

std::uint32_t token_slot(std::uint64_t token)
{
    return static_cast<std::uint32_t>(token);
}

void HsbFrame::discard_mailbox(SlotSync& slots)
{
    const auto token = slots.mailbox.exchange(0, std::memory_order_acq_rel);
    if (token == 0) {
        return;
    }
    auto& record = slots.records[token_slot(token)];
    std::uint8_t ready = SlotSync::Ready;
    record.state.compare_exchange_strong(
        ready, SlotSync::Free, std::memory_order_acq_rel);
}

// One producer. Returns the new value, or 0 when the counter is already saturated.
std::uint32_t saturating_increment(std::atomic<std::uint32_t>& counter)
{
    const auto current = counter.load(std::memory_order_relaxed);
    if (current == std::numeric_limits<std::uint32_t>::max()) {
        return 0;
    }
    counter.store(current + 1u, std::memory_order_relaxed);
    return current + 1u;
}

void saturating_add(std::atomic<std::uint32_t>& counter, std::uint32_t amount)
{
    if (amount == 0) {
        return;
    }
    const auto current = counter.load(std::memory_order_relaxed);
    const auto room = std::numeric_limits<std::uint32_t>::max() - current;
    const auto added = amount < room ? amount : room;
    if (added == 0) {
        return;
    }
    counter.store(current + added, std::memory_order_relaxed);
}

CpuRoceFrameReceiver::OwnedDmaBufFd::~OwnedDmaBufFd()
{
    if (fd >= 0) {
        ::close(fd);
    }
}

struct CpuRoceFrameReceiver::DeviceTrailerCopy {
    ~DeviceTrailerCopy()
    {
        if (stream) {
            cuStreamSynchronize(stream.get());
        }
    }

    void initialize()
    {
        void* host_pointer = nullptr;
        check_cuda(cuMemHostAlloc(
            &host_pointer, hololink::METADATA_SIZE, 0));
        host.reset(host_pointer);
        CUevent event_handle = nullptr;
        check_cuda(cuEventCreate(&event_handle, CU_EVENT_DISABLE_TIMING));
        event.reset(event_handle);
        CUstream stream_handle = nullptr;
        check_cuda(cuStreamCreate(
            &stream_handle, CU_STREAM_NON_BLOCKING));
        stream.reset(stream_handle);
    }

    const std::uint8_t* copy(CUdeviceptr source)
    {
        check_cuda(cuMemcpyDtoHAsync(
            host.get(), source, hololink::METADATA_SIZE, stream.get()));
        check_cuda(cuEventRecord(event.get(), stream.get()));
        for (int i = 0; i < kMetadataEventSpinCount; ++i) {
            const CUresult result = cuEventQuery(event.get());
            if (result == CUDA_SUCCESS) {
                return static_cast<const std::uint8_t*>(host.get());
            }
            if (result != CUDA_ERROR_NOT_READY) {
                check_cuda(result);
            }
        }
        check_cuda(cuEventSynchronize(event.get()));
        return static_cast<const std::uint8_t*>(host.get());
    }

    UniqueHost host;
    UniqueEvent event;
    UniqueStream stream;
};

HsbFrame::HsbFrame(std::uint8_t* payload, std::size_t payload_size, std::uint32_t slot,
    std::optional<std::uint32_t> immediate_data, std::uint32_t dropped,
    std::uint32_t overwritten, std::uint32_t invalid_immediates,
    std::uint64_t completion_count, std::uint32_t session_frame_number,
    std::optional<hololink::Hololink::FrameMetadata> metadata,
    std::chrono::system_clock::time_point received_at, std::shared_ptr<SlotSync> slots)
    : payload_(payload)
    , payload_size_(payload_size)
    , slot_(slot)
    , immediate_data_(immediate_data)
    , dropped_(dropped)
    , overwritten_(overwritten)
    , invalid_immediates_(invalid_immediates)
    , completion_count_(completion_count)
    , session_frame_number_(session_frame_number)
    , metadata_(std::move(metadata))
    , received_at_(received_at)
    , slots_(std::move(slots))
{
}

HsbFrame::HsbFrame(HsbFrame&& other) noexcept
    : payload_(other.payload_)
    , payload_size_(other.payload_size_)
    , slot_(other.slot_)
    , immediate_data_(other.immediate_data_)
    , dropped_(other.dropped_)
    , overwritten_(other.overwritten_)
    , invalid_immediates_(other.invalid_immediates_)
    , completion_count_(other.completion_count_)
    , session_frame_number_(other.session_frame_number_)
    , metadata_(std::move(other.metadata_))
    , received_at_(other.received_at_)
    , slots_(std::move(other.slots_))
{
    other.payload_ = nullptr;
}

HsbFrame& HsbFrame::operator=(HsbFrame&& other) noexcept
{
    if (this != &other) {
        try {
            release();
        } catch (...) {
        }
        payload_ = other.payload_;
        payload_size_ = other.payload_size_;
        slot_ = other.slot_;
        immediate_data_ = other.immediate_data_;
        dropped_ = other.dropped_;
        overwritten_ = other.overwritten_;
        invalid_immediates_ = other.invalid_immediates_;
        completion_count_ = other.completion_count_;
        session_frame_number_ = other.session_frame_number_;
        metadata_ = std::move(other.metadata_);
        received_at_ = other.received_at_;
        slots_ = std::move(other.slots_);
        other.payload_ = nullptr;
    }
    return *this;
}

HsbFrame::~HsbFrame()
{
    try {
        release();
    } catch (...) {
    }
}

void HsbFrame::release(std::function<void()> synchronize)
{
    if (!slots_) {
        return;
    }
    // Drop our reference first so a second release, including the destructor, is a no-op.
    const auto slots = std::move(slots_);
    payload_ = nullptr;
    std::exception_ptr error;
    if (synchronize) {
        try {
            synchronize();
        } catch (...) {
            error = std::current_exception();
        }
    }
    if (!error) {
        std::uint8_t expected = HsbFrame::SlotSync::Leased;
        slots->records[slot_].state.compare_exchange_strong(
            expected, HsbFrame::SlotSync::Free, std::memory_order_acq_rel);
        return;
    }

    {
        std::lock_guard<std::mutex> lock(slots->mutex);
        // Copying the existing control block does not allocate. Do this before
        // notify or close, so a later exception cannot drop the payload owner.
        static_assert(noexcept(slots->keep_alive = slots),
            "failed-release retain must not throw");
        slots->keep_alive = slots;
        if (!slots->failure) {
            slots->failure = error;
        }
        slots->flags.fetch_or(HsbFrame::SlotSync::kFlagFailure, std::memory_order_release);
        HsbFrame::arm_wake(*slots);
    }
    HsbFrame::discard_mailbox(*slots);
    HsbFrame::wake_waiters(*slots);
    if (const auto transceiver = slots->transceiver.lock()) {
        try {
            transceiver->close();
        } catch (...) {
        }
    }
    std::rethrow_exception(error);
}

CpuRoceFrameReceiver::CpuRoceFrameReceiver(Config config)
    : config_(std::move(config))
    , slots_(std::make_shared<HsbFrame::SlotSync>())
{
    if (config_.metadata_size != 0 && config_.metadata_size != hololink::METADATA_SIZE) {
        throw std::invalid_argument(
            "CpuRoceFrameReceiver metadata_size must be 0 or 128");
    }
    if (config_.cuda_payload) {
        const auto& cuda = *config_.cuda_payload;
        if (cuda.address == nullptr || cuda.size == 0) {
            throw std::invalid_argument(
                "CpuRoceFrameReceiver CUDA payload needs an address and size");
        }
        if (cuda.host_mapped) {
            config_.payload_memory = HostMemory { cuda.address, cuda.size };
        } else {
            if (config_.cuda_context == nullptr) {
                throw std::invalid_argument(
                    "CpuRoceFrameReceiver CUDA payload export requires a "
                    "CUDA context");
            }
            const CudaContextPush current(
                static_cast<CUcontext>(config_.cuda_context));
            int fd = -1;
            const CUresult exported = cuMemGetHandleForAddressRange(
                &fd, reinterpret_cast<CUdeviceptr>(cuda.address), cuda.size,
                CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
            if (exported == CUDA_SUCCESS && fd >= 0) {
                dmabuf_.fd = fd;
                config_.payload_memory = DmaBufMemory {
                    cuda.address, fd, 0, cuda.size
                };
            } else {
                config_.payload_memory = GpuPointerMemory {
                    cuda.address, cuda.size
                };
            }
        }
    }
    const bool device_payload
        = std::holds_alternative<DmaBufMemory>(config_.payload_memory)
        || std::holds_alternative<GpuPointerMemory>(config_.payload_memory);
    if (device_payload && config_.host_trailer && config_.metadata_size != 0) {
        if (config_.cuda_context == nullptr) {
            throw std::invalid_argument(
                "CpuRoceFrameReceiver device metadata requires a CUDA context");
        }
        device_trailer_ = true;
    }
    if (config_.payload_size == 0 || config_.slot_count == 0) {
        throw std::invalid_argument(
            "CpuRoceFrameReceiver payload_size and slot_count must be non-zero");
    }
    if (config_.receive_credit_count == 0
        || config_.receive_credit_count
            > hololink::transport::roce::detail::kMaxHandledReceiveCredits) {
        throw std::invalid_argument(
            "CpuRoceFrameReceiver receive_credit_count must be from 1 to "
            + std::to_string(
                hololink::transport::roce::detail::kMaxHandledReceiveCredits));
    }

    static_assert(hololink::METADATA_SIZE == kProtocolPage);
    if (config_.payload_size
        > std::numeric_limits<std::size_t>::max() - (kProtocolPage - 1)) {
        throw std::invalid_argument(
            "CpuRoceFrameReceiver payload_size cannot be aligned");
    }
    metadata_offset_ = ((config_.payload_size + (kProtocolPage - 1)) / kProtocolPage)
        * kProtocolPage;
    stride_ = metadata_offset_;
    // DataChannel::configure_roce reserves one HSB metadata page when the
    // payload exceeds one protocol page even when metadata is disabled.
    // With metadata, this tail is the trailer region.
    if (config_.metadata_size != 0 || config_.payload_size > kProtocolPage) {
        if (stride_ > std::numeric_limits<std::size_t>::max() - hololink::METADATA_SIZE) {
            throw std::invalid_argument(
                "CpuRoceFrameReceiver stride is out of range");
        }
        stride_ += hololink::METADATA_SIZE;
    }
    const std::size_t bytes = ring_bytes(stride_, config_.slot_count);
    std::size_t available = 0;
    if (const auto* host = std::get_if<HostMemory>(&config_.payload_memory)) {
        if (host->address == nullptr) {
            throw std::invalid_argument(
                "CpuRoceFrameReceiver host payload address is null");
        }
        available = host->size;
    } else if (const auto* dma = std::get_if<DmaBufMemory>(&config_.payload_memory)) {
        if (dma->address == nullptr || dma->fd < 0) {
            throw std::invalid_argument(
                "CpuRoceFrameReceiver DMA-BUF payload needs an address and fd");
        }
        available = dma->size;
    } else if (const auto* gpu = std::get_if<GpuPointerMemory>(&config_.payload_memory)) {
        if (gpu->address == nullptr) {
            throw std::invalid_argument(
                "CpuRoceFrameReceiver GPU-pointer payload address is null");
        }
        available = gpu->size;
    } else {
        throw std::invalid_argument(
            "CpuRoceFrameReceiver requires caller-owned host, DMA-BUF, or "
            "GPU-pointer memory");
    }
    if (available < bytes) {
        throw std::invalid_argument(
            "CpuRoceFrameReceiver payload memory is smaller than the ring");
    }

    slots_->records = std::vector<HsbFrame::SlotSync::Record>(config_.slot_count);
    slots_->payload_owner = config_.payload_owner;

    CpuRoceTransceiver::Config transport;
    transport.ibv_name = config_.ibv_name;
    transport.ibv_port = config_.ibv_port;
    transport.local_ip = config_.local_ip;
    transport.peer = config_.peer;
    transport.mode = CpuRoceTransceiver::Mode::Rx;
    transport.poll_cpu = config_.poll_cpu;
    transport.poll_cpu_best_effort = config_.poll_cpu_best_effort;
    transport.rx_path_mtu_4096 = config_.rx_path_mtu_4096;
    transport.rx.payload_memory = config_.payload_memory;
    transport.rx.stride = stride_;
    transport.rx.slot_count = config_.slot_count;
    transport.rx.frame_layout.payload_size = config_.payload_size;
    transport.rx.frame_layout.metadata_offset = metadata_offset_;
    transport.rx.frame_layout.metadata_size = config_.metadata_size;
    transceiver_ = std::make_shared<CpuRoceTransceiver>(std::move(transport));
    slots_->transceiver = transceiver_;
}

CpuRoceFrameReceiver::~CpuRoceFrameReceiver()
{
    try {
        stop();
    } catch (...) {
    }
}

void CpuRoceFrameReceiver::connect()
{
    {
        std::lock_guard<std::mutex> lock(slots_->mutex);
        if (slots_->stopping || quiesced_ || slots_->failure) {
            throw std::logic_error("CpuRoceFrameReceiver::connect is one-shot");
        }
        if (connected_) {
            return;
        }
    }
    try {
        hololink::transport::roce::detail::CpuRoceRxCompletionHandler handler(*transceiver_);
        if (device_trailer_ && config_.umr_metadata) {
            handler.request_host_metadata();
        }
        handler.enable_handled_receive(
            [](void* context, hololink::transport::roce::detail::RxCompletion completion) {
                static_cast<CpuRoceFrameReceiver*>(context)->on_rx_completion(
                    completion.immediate_data);
            },
            this, config_.receive_credit_count);
        transceiver_->start();
        if (device_trailer_) {
            host_trailers_ = handler.host_metadata();
        }
        std::lock_guard<std::mutex> lock(slots_->mutex);
        connected_ = true;
    } catch (...) {
        try {
            transceiver_->close();
        } catch (...) {
        }
        {
            std::lock_guard<std::mutex> lock(slots_->mutex);
            slots_->stopping = true;
            slots_->flags.fetch_or(
                HsbFrame::SlotSync::kFlagStopping, std::memory_order_release);
            HsbFrame::arm_wake(*slots_);
        }
        HsbFrame::wake_waiters(*slots_);
        throw;
    }
}

void CpuRoceFrameReceiver::start()
{
    connect();
    {
        std::lock_guard<std::mutex> lock(slots_->mutex);
        if (slots_->stopping || quiesced_ || slots_->failure) {
            throw std::logic_error("CpuRoceFrameReceiver::start is one-shot");
        }
        if (polling_) {
            return;
        }
    }
    try {
        monitor_ = std::thread([this] { monitor(); });
        std::lock_guard<std::mutex> lock(slots_->mutex);
        polling_ = true;
    } catch (...) {
        try {
            transceiver_->close();
        } catch (...) {
        }
        {
            std::lock_guard<std::mutex> lock(slots_->mutex);
            slots_->stopping = true;
            slots_->flags.fetch_or(
                HsbFrame::SlotSync::kFlagStopping, std::memory_order_release);
            HsbFrame::arm_wake(*slots_);
        }
        HsbFrame::wake_waiters(*slots_);
        if (monitor_.joinable()) {
            monitor_.join();
        }
        throw;
    }
}

void CpuRoceFrameReceiver::quiesce()
{
    {
        std::lock_guard<std::mutex> lock(slots_->mutex);
        if (quiesced_) {
            return;
        }
        quiesced_ = true;
        slots_->flags.fetch_or(
            HsbFrame::SlotSync::kFlagQuiesced, std::memory_order_release);
        HsbFrame::discard_mailbox(*slots_);
        HsbFrame::arm_wake(*slots_);
    }
    // Writes may continue until the FPGA is disarmed. Nothing further is
    // published, and a waiter returns empty instead of holding its timeout.
    transceiver_->request_exit();
    HsbFrame::wake_waiters(*slots_);
    if (monitor_.joinable()) {
        monitor_.join();
    }
}

void CpuRoceFrameReceiver::stop()
{
    quiesce();
    bool close_transceiver = false;
    {
        std::lock_guard<std::mutex> lock(slots_->mutex);
        if (!slots_->stopping) {
            slots_->stopping = true;
            close_transceiver = true;
            slots_->flags.fetch_or(
                HsbFrame::SlotSync::kFlagStopping, std::memory_order_release);
            HsbFrame::discard_mailbox(*slots_);
            HsbFrame::arm_wake(*slots_);
        }
    }
    if (close_transceiver) {
        HsbFrame::wake_waiters(*slots_);
    }
    std::exception_ptr close_error;
    if (close_transceiver) {
        try {
            transceiver_->close();
        } catch (...) {
            close_error = std::current_exception();
        }
    }
    if (monitor_.joinable()) {
        monitor_.join();
    }
    if (close_error) {
        std::rethrow_exception(close_error);
    }
}

void CpuRoceFrameReceiver::monitor()
{
    const auto terminalize = [this](std::exception_ptr failure) {
        // This thread has already left blocking_monitor(), so close() is
        // allowed and the queue pair does not stay connected.
        try {
            transceiver_->close();
        } catch (...) {
        }
        store_failure(std::move(failure));
    };
    try {
        if (config_.on_poll_thread_start != nullptr) {
            config_.on_poll_thread_start(config_.on_poll_thread_start_context);
        }
        // The completion handler runs on this thread. A cached host base
        // already holds each trailer. Otherwise the stream and pinned buffer
        // are created once here and reused for every trailer.
        if (device_trailer_ && host_trailers_ == nullptr) {
            check_cuda(cuCtxSetCurrent(
                static_cast<CUcontext>(config_.cuda_context)));
            device_trailer_copy_ = std::make_unique<DeviceTrailerCopy>();
            device_trailer_copy_->initialize();
        }
        transceiver_->blocking_monitor();
    } catch (const hololink::transport::roce::detail::ThreadAffinityRestoreError&) {
        bool quiesced = false;
        {
            std::lock_guard<std::mutex> lock(slots_->mutex);
            quiesced = quiesced_;
        }
        // A restore error after request_exit is not a datapath failure and
        // must not break quiesce()'s live-QP guarantee.
        if (!quiesced) {
            terminalize(std::current_exception());
        }
    } catch (...) {
        terminalize(std::current_exception());
    }
    device_trailer_copy_.reset();
}

void CpuRoceFrameReceiver::store_failure(std::exception_ptr failure)
{
    {
        std::lock_guard<std::mutex> lock(slots_->mutex);
        if (!slots_->failure) {
            slots_->failure = failure;
        }
        slots_->flags.fetch_or(HsbFrame::SlotSync::kFlagFailure, std::memory_order_release);
        HsbFrame::discard_mailbox(*slots_);
        HsbFrame::arm_wake(*slots_);
    }
    HsbFrame::wake_waiters(*slots_);
}

void CpuRoceFrameReceiver::report_receive_error(
    ReceiveError error, std::uint32_t total) const
{
    if (config_.on_receive_error != nullptr) {
        config_.on_receive_error(config_.on_receive_error_context, error, total);
    }
}

void CpuRoceFrameReceiver::fail(std::exception_ptr failure)
{
    store_failure(failure);
    std::rethrow_exception(std::move(failure));
}

bool CpuRoceFrameReceiver::facts_ready() const
{
    std::lock_guard<std::mutex> lock(slots_->mutex);
    return connected_ && !slots_->stopping && !slots_->failure;
}

std::uint32_t CpuRoceFrameReceiver::qp_number() const
{
    if (!facts_ready()) {
        throw std::logic_error("CpuRoceFrameReceiver QP is not live");
    }
    return transceiver_->get_qp_number();
}

std::uint32_t CpuRoceFrameReceiver::rkey() const
{
    if (!facts_ready()) {
        throw std::logic_error("CpuRoceFrameReceiver rkey is not live");
    }
    return transceiver_->get_rkey();
}

std::uint32_t CpuRoceFrameReceiver::path_mtu() const
{
    if (!facts_ready()) {
        throw std::logic_error("CpuRoceFrameReceiver path MTU is not live");
    }
    const auto payload = transceiver_->get_path_mtu();
    if (payload == 0) {
        throw std::logic_error("CpuRoceFrameReceiver path MTU is not programmed");
    }
    return payload;
}

std::uint64_t CpuRoceFrameReceiver::external_frame_memory() const
{
    if (!facts_ready()) {
        throw std::logic_error("CpuRoceFrameReceiver frame memory is not live");
    }
    return transceiver_->external_frame_memory();
}

bool CpuRoceFrameReceiver::umr_active() const
{
    if (!facts_ready()) {
        throw std::logic_error("CpuRoceFrameReceiver UMR state is not live");
    }
    return host_trailers_ != nullptr;
}

bool CpuRoceFrameReceiver::frames_ready() const
{
    return slots_->mailbox.load(std::memory_order_acquire) != 0
        || slots_->flags.load(std::memory_order_acquire) != 0;
}

std::exception_ptr CpuRoceFrameReceiver::failure() const
{
    std::lock_guard<std::mutex> lock(slots_->mutex);
    return slots_->failure;
}

std::uint32_t CpuRoceFrameReceiver::dropped() const
{
    return slots_->dropped.load(std::memory_order_relaxed);
}

std::uint32_t CpuRoceFrameReceiver::overwritten() const
{
    return slots_->overwritten.load(std::memory_order_relaxed);
}

std::uint32_t CpuRoceFrameReceiver::invalid_immediates() const
{
    return slots_->invalid_immediates.load(std::memory_order_relaxed);
}

std::uint64_t CpuRoceFrameReceiver::completion_count() const
{
    return slots_->completion_count.load(std::memory_order_relaxed);
}

std::uint8_t* CpuRoceFrameReceiver::slot_payload(std::uint32_t slot) const
{
    void* base = nullptr;
    if (const auto* host = std::get_if<HostMemory>(&config_.payload_memory)) {
        base = host->address;
    } else if (const auto* dma = std::get_if<DmaBufMemory>(&config_.payload_memory)) {
        base = dma->address;
    } else if (const auto* gpu = std::get_if<GpuPointerMemory>(&config_.payload_memory)) {
        base = gpu->address;
    }
    return static_cast<std::uint8_t*>(base) + slot * stride_;
}

void CpuRoceFrameReceiver::on_rx_completion(
    std::optional<std::uint32_t> immediate_data)
{
    if (!immediate_data) {
        fail(std::make_exception_ptr(std::runtime_error(
            "CpuRoceFrameReceiver completion has no immediate data")));
    }
    constexpr std::uint32_t kImmediateSlotMask = 0xFFF;
    const std::uint32_t slot = *immediate_data & kImmediateSlotMask;
    // Returning leaves the credit to be reposted. A throw would take the QP down.
    if (slot >= config_.slot_count) {
        std::uint32_t total = 0;
        bool report = false;
        if ((slots_->flags.load(std::memory_order_acquire)
                & (HsbFrame::SlotSync::kFlagStopping | HsbFrame::SlotSync::kFlagQuiesced
                    | HsbFrame::SlotSync::kFlagFailure))
            == 0) {
            HsbFrame::begin_stats(*slots_);
            total = saturating_increment(slots_->invalid_immediates);
            HsbFrame::end_stats(*slots_);
            report = total != 0;
        }
        if (report) {
            report_receive_error(ReceiveError::InvalidImmediate, total);
        }
        return;
    }
    publish(slot, immediate_data);
}

void CpuRoceFrameReceiver::note_trailer_locked(
    const hololink::Hololink::FrameMetadata& wire)
{
    // Caller holds the stats seqlock. have_wire and last_wire are producer-only.
    auto& reception = slots_->reception;
    if (!reception.have_wire) {
        reception.frame_number.store(1, std::memory_order_relaxed);
        reception.have_wire = true;
    } else {
        const auto next = reception.frame_number.load(std::memory_order_relaxed)
            + static_cast<std::uint16_t>(wire.frame_number - reception.last_wire);
        reception.frame_number.store(next, std::memory_order_relaxed);
    }
    reception.last_wire = wire.frame_number;
    reception.rx_bytes.fetch_add(wire.bytes_written, std::memory_order_relaxed);
    reception.psn.store(wire.psn, std::memory_order_relaxed);
}

void CpuRoceFrameReceiver::publish(
    std::uint32_t slot, std::optional<std::uint32_t> immediate_data)
{
    const auto received_at = std::chrono::system_clock::now();
    const auto terminal = slots_->flags.load(std::memory_order_acquire);
    if ((terminal & (HsbFrame::SlotSync::kFlagStopping | HsbFrame::SlotSync::kFlagQuiesced))
        != 0) {
        return;
    }
    if ((terminal & HsbFrame::SlotSync::kFlagFailure) != 0) {
        std::exception_ptr failure;
        {
            std::lock_guard<std::mutex> lock(slots_->mutex);
            failure = slots_->failure;
        }
        fail(failure ? std::move(failure)
                     : std::make_exception_ptr(std::runtime_error(
                         "CpuRoceFrameReceiver has failed")));
    }
    if (slot >= slots_->records.size()) {
        fail(std::make_exception_ptr(std::out_of_range(
            "CpuRoceFrameReceiver completion slot is outside the ring")));
    }

    auto& record = slots_->records[slot];
    std::uint8_t observed = HsbFrame::SlotSync::Free;
    bool leased = false;
    std::uint32_t drops = 0;
    for (;;) {
        observed = record.state.load(std::memory_order_acquire);
        if (observed == HsbFrame::SlotSync::Writing) {
            fail(std::make_exception_ptr(std::logic_error(
                "CpuRoceFrameReceiver completion arrived while its slot was being published")));
        }
        if (observed == HsbFrame::SlotSync::Leased) {
            leased = true;
            break;
        }
        std::uint8_t expected = observed;
        if (record.state.compare_exchange_weak(expected, HsbFrame::SlotSync::Writing,
                std::memory_order_acq_rel, std::memory_order_acquire)) {
            // Replacing an unread Ready frame is one drop. A later claim of
            // the same token fails because the record is now Writing.
            drops += observed == HsbFrame::SlotSync::Ready ? 1u : 0u;
            break;
        }
        observed = expected;
    }

    struct WritingGuard {
        HsbFrame::SlotSync::Record& record;
        bool armed = true;
        ~WritingGuard()
        {
            if (!armed) {
                return;
            }
            std::uint8_t expected = HsbFrame::SlotSync::Writing;
            record.state.compare_exchange_strong(expected, HsbFrame::SlotSync::Free,
                std::memory_order_acq_rel);
        }
    };
    WritingGuard guard { record };
    if (leased) {
        guard.armed = false;
    }

    std::optional<hololink::Hololink::FrameMetadata> metadata;
    if (config_.host_trailer && config_.metadata_size != 0) {
        if (device_trailer_) {
            const auto* trailer = host_trailers_ != nullptr
                ? host_trailers_ + static_cast<std::size_t>(slot) * config_.metadata_size
                : device_trailer_copy_->copy(reinterpret_cast<CUdeviceptr>(
                    slot_payload(slot) + metadata_offset_));
            metadata = hololink::core::parse_frame_metadata(
                trailer, static_cast<unsigned>(config_.metadata_size));
        } else {
            metadata = hololink::core::parse_frame_metadata(
                slot_payload(slot) + metadata_offset_,
                static_cast<unsigned>(config_.metadata_size));
        }
    }

    if ((slots_->flags.load(std::memory_order_acquire)
            & (HsbFrame::SlotSync::kFlagStopping | HsbFrame::SlotSync::kFlagQuiesced
                | HsbFrame::SlotSync::kFlagFailure))
        != 0) {
        return;
    }

    std::uint32_t overwrite_total = 0;
    std::uint64_t previous = 0;
    if (leased) {
        HsbFrame::begin_stats(*slots_);
        slots_->completion_count.fetch_add(1, std::memory_order_relaxed);
        overwrite_total = saturating_increment(slots_->overwritten);
        if (metadata.has_value()) {
            note_trailer_locked(metadata.value());
        }
        HsbFrame::end_stats(*slots_);
        if (overwrite_total != 0) {
            report_receive_error(ReceiveError::Overwrite, overwrite_total);
        }
        return;
    }

    previous = slots_->mailbox.exchange(0, std::memory_order_acq_rel);
    if (previous != 0) {
        auto& previous_record = slots_->records[token_slot(previous)];
        std::uint8_t ready = HsbFrame::SlotSync::Ready;
        if (previous_record.state.compare_exchange_strong(
                ready, HsbFrame::SlotSync::Free, std::memory_order_acq_rel)) {
            ++drops;
        }
    }

    HsbFrame::begin_stats(*slots_);
    slots_->completion_count.fetch_add(1, std::memory_order_relaxed);
    saturating_add(slots_->dropped, drops);
    if (metadata.has_value()) {
        note_trailer_locked(metadata.value());
    }
    HsbFrame::SlotSync::Arrival arrival;
    arrival.slot = slot;
    arrival.immediate_data = immediate_data;
    arrival.dropped = slots_->dropped.load(std::memory_order_relaxed);
    arrival.overwritten = slots_->overwritten.load(std::memory_order_relaxed);
    arrival.invalid_immediates = slots_->invalid_immediates.load(std::memory_order_relaxed);
    arrival.completion_count = slots_->completion_count.load(std::memory_order_relaxed);
    arrival.session_frame_number = slots_->reception.frame_number.load(std::memory_order_relaxed);
    arrival.metadata = std::move(metadata);
    arrival.received_at = received_at;
    HsbFrame::end_stats(*slots_);

    record.snapshot = std::move(arrival);
    guard.armed = false;
    record.state.store(HsbFrame::SlotSync::Ready, std::memory_order_release);
    // Generation makes a republished slot's token nonzero and distinct.
    // Claim checks the slot state, so a stale token leases the newer frame.
    auto generation = ++slots_->generation;
    if (generation == 0) {
        generation = ++slots_->generation;
    }
    const auto token = pack_token(generation, slot);
    slots_->mailbox.store(token, std::memory_order_release);
    if (slots_->flags.load(std::memory_order_acquire) != 0) {
        HsbFrame::discard_mailbox(*slots_);
    }
    HsbFrame::arm_wake(*slots_);
    HsbFrame::wake_waiters(*slots_);
}

CpuRoceFrameReceiver::Stats CpuRoceFrameReceiver::stats() const
{
    Stats out;
    auto& slots = *slots_;
    unsigned misses = 0;
    for (;;) {
        const auto start = slots.stats_seq.load(std::memory_order_acquire);
        bool consistent = (start & 1u) == 0;
        if (consistent) {
            out.rx_bytes = slots.reception.rx_bytes.load(std::memory_order_relaxed);
            out.completions = slots.completion_count.load(std::memory_order_relaxed);
            out.frame_number = slots.reception.frame_number.load(std::memory_order_relaxed);
            out.psn = slots.reception.psn.load(std::memory_order_relaxed);
            out.dropped = slots.dropped.load(std::memory_order_relaxed);
            out.overwritten = slots.overwritten.load(std::memory_order_relaxed);
            out.invalid_immediates = slots.invalid_immediates.load(std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_acquire);
            consistent = slots.stats_seq.load(std::memory_order_relaxed) == start;
        }
        if (consistent) {
            return out;
        }
        // A higher-priority caller on the poller's CPU must sleep, or the
        // preempted writer never reaches end_stats.
        if (++misses == 64u) {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
            misses = 0;
        }
    }
}

std::optional<HsbFrame> CpuRoceFrameReceiver::get_frame(std::chrono::microseconds timeout)
{
    bool expected = false;
    if (!slots_->get_frame_active.compare_exchange_strong(
            expected, true, std::memory_order_acq_rel)) {
        throw std::logic_error("CpuRoceFrameReceiver::get_frame is already running");
    }
    struct ActiveGuard {
        std::atomic<bool>& active;
        ~ActiveGuard()
        {
            active.store(false, std::memory_order_release);
        }
    } active { slots_->get_frame_active };

    const auto terminal = [&] {
        return slots_->flags.load(std::memory_order_acquire) != 0;
    };
    const auto take = [&]() -> std::optional<HsbFrame::SlotSync::Arrival> {
        if (terminal()) {
            return std::nullopt;
        }
        const auto token = slots_->mailbox.exchange(0, std::memory_order_acq_rel);
        if (token == 0) {
            return std::nullopt;
        }
        const auto index = token_slot(token);
        auto& record = slots_->records[index];
        std::uint8_t ready = HsbFrame::SlotSync::Ready;
        if (!record.state.compare_exchange_strong(ready, HsbFrame::SlotSync::Leased,
                std::memory_order_acq_rel)) {
            return std::nullopt;
        }
        return record.snapshot;
    };

    std::optional<HsbFrame::SlotSync::Arrival> arrival;
    if (timeout <= std::chrono::microseconds::zero()) {
        arrival = take();
    } else {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        for (;;) {
            arrival = take();
            if (arrival.has_value() || terminal()) {
                break;
            }
            const auto observed = slots_->wake_seq.load(std::memory_order_acquire);
            arrival = take();
            if (arrival.has_value() || terminal()) {
                break;
            }
            const auto remaining = std::chrono::duration_cast<std::chrono::microseconds>(
                deadline - std::chrono::steady_clock::now());
            if (remaining <= std::chrono::microseconds::zero()) {
                break;
            }
            timespec timeout_spec { static_cast<time_t>(remaining.count() / 1000000),
                static_cast<long>((remaining.count() % 1000000) * 1000) };
            const auto waited = ::syscall(SYS_futex,
                reinterpret_cast<std::uint32_t*>(&slots_->wake_seq), FUTEX_WAIT_PRIVATE,
                observed, &timeout_spec, nullptr, 0);
            if (waited < 0 && errno != EAGAIN && errno != ETIMEDOUT && errno != EINTR) {
                throw std::system_error(errno, std::generic_category(), "futex wait");
            }
        }
    }
    if (!arrival.has_value()) {
        return std::nullopt;
    }
    auto* const payload = slot_payload(arrival->slot);
    return HsbFrame(payload, config_.payload_size, arrival->slot, arrival->immediate_data,
        arrival->dropped, arrival->overwritten, arrival->invalid_immediates,
        arrival->completion_count, arrival->session_frame_number,
        std::move(arrival->metadata), arrival->received_at, slots_);
}
} // namespace hololink::operators
