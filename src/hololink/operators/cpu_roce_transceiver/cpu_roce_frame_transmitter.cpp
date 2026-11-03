/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cpu_roce_frame_transmitter.hpp"

#include "cpu_roce_frame_transmitter_internal.hpp"

#include <unistd.h>

#include <cuda.h>

#include <hololink/transport/roce/cpu_roce_transceiver_internal.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace hololink::operators {

namespace {

    using Clock = std::chrono::steady_clock;
    namespace roce = hololink::transport::roce;
    using frame_tx::Pool;

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
            throw std::invalid_argument("CpuRoceFrameTransmitter ring size is out of range");
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

    struct OwnedFd {
        int fd = -1;
        ~OwnedFd()
        {
            if (fd >= 0) {
                ::close(fd);
            }
        }
        OwnedFd() = default;
        OwnedFd(const OwnedFd&) = delete;
        OwnedFd& operator=(const OwnedFd&) = delete;
    };

    struct AlignedBytes {
        void* address = nullptr;
        explicit AlignedBytes(std::size_t bytes)
        {
            if (bytes == 0) {
                return;
            }
            const std::size_t page = static_cast<std::size_t>(::getpagesize());
            if (::posix_memalign(&address, page, bytes) != 0) {
                throw std::bad_alloc();
            }
            std::memset(address, 0, bytes);
        }
        ~AlignedBytes()
        {
            std::free(address);
        }
        AlignedBytes(const AlignedBytes&) = delete;
        AlignedBytes& operator=(const AlignedBytes&) = delete;
    };

    struct Kept {
        std::shared_ptr<void> payload_owner;
        std::vector<CpuRoceFrameTransmitter::Slot> slots;
    };

} // namespace

struct CpuRoceFrameTransmitter::Impl {
    explicit Impl(Config config)
        : config_(std::move(config))
    {
        validate();
        const std::size_t bytes = ring_bytes(config_.stride, config_.slot_count);
        if (config_.cuda_payload.size < bytes) {
            throw std::invalid_argument(
                "CpuRoceFrameTransmitter CUDA allocation is smaller than the ring");
        }
        const std::uint64_t trailer_region
            = static_cast<std::uint64_t>(config_.trailer_capacity) * config_.slot_count;
        if (trailer_region > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument(
                "CpuRoceFrameTransmitter trailer region does not fit in uint32_t");
        }
        trailer_bytes_ = static_cast<std::uint32_t>(trailer_region);
        context_ = static_cast<CUcontext>(config_.cuda_context);
        export_payload();
        if (trailer_bytes_ != 0) {
            trailer_ = std::make_unique<AlignedBytes>(trailer_bytes_);
        }
        pool_ = Pool::create(config_.slot_count, config_.cuda_payload.address, config_.stride,
            static_cast<std::uint32_t>(config_.payload_size));
        kept_ = std::make_shared<Kept>();
        kept_->payload_owner = config_.payload_owner;
        pool_->owner = kept_;
        kept_->slots.resize(config_.slot_count);
        for (std::uint32_t index = 0; index < config_.slot_count; ++index) {
            auto& slot = kept_->slots[index];
            slot.block_ = pool_.get();
            slot.index_ = index;
            slot.data_ = pool_->data(index);
            slot.capacity_ = static_cast<std::uint32_t>(config_.payload_size);
        }
        bring_up(false);
        CudaContextPush current(context_);
        CUstream stream = nullptr;
        check_cuda(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        stream_ = stream;
    }

    ~Impl()
    {
        close();
    }

    void validate() const
    {
        if (config_.ibv_name.empty() || config_.ibv_port == 0 || config_.cuda_context == nullptr
            || config_.peer.ip.empty() || config_.peer.qp == 0) {
            throw std::invalid_argument("CpuRoceFrameTransmitter config is incomplete");
        }
        if (config_.slot_count == 0 || config_.slot_count > 4096 || config_.payload_size == 0
            || config_.payload_size > config_.stride
            || config_.payload_size > std::numeric_limits<std::uint32_t>::max()
            || config_.cuda_payload.address == nullptr) {
            throw std::invalid_argument("CpuRoceFrameTransmitter geometry is invalid");
        }
        const auto message = config_.payload_size + config_.trailer_capacity;
        if (message < config_.payload_size || message > config_.peer.rx_stride) {
            throw std::invalid_argument(
                "CpuRoceFrameTransmitter payload plus trailer exceeds the peer stride");
        }
    }

    void export_payload()
    {
        if (config_.cuda_payload.host_mapped) {
            return;
        }
        CudaContextPush current(context_);
        int fd = -1;
        const CUresult exported = cuMemGetHandleForAddressRange(&fd,
            reinterpret_cast<CUdeviceptr>(config_.cuda_payload.address), config_.cuda_payload.size,
            CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
        if (exported == CUDA_SUCCESS && fd >= 0) {
            dmabuf_.fd = fd;
        }
    }

    roce::PayloadMemory payload_memory(bool gpu_pointer) const
    {
        const auto& cuda = config_.cuda_payload;
        if (cuda.host_mapped) {
            return roce::HostMemory { cuda.address, cuda.size };
        }
        if (!gpu_pointer && dmabuf_.fd >= 0) {
            return roce::DmaBufMemory { cuda.address, dmabuf_.fd, 0, cuda.size };
        }
        return roce::GpuPointerMemory { cuda.address, cuda.size };
    }

    void bring_up(bool gpu_pointer)
    {
        roce::CpuRoceTransceiver::Config transport;
        transport.ibv_name = config_.ibv_name;
        transport.ibv_port = config_.ibv_port;
        transport.local_ip = config_.local_ip;
        transport.mode = roce::CpuRoceTransceiver::Mode::Tx;
        transport.tx_operation = roce::CpuRoceTransceiver::TxOperation::Send;
        transport.tx.payload_memory = payload_memory(gpu_pointer);
        transport.tx.stride = config_.stride;
        transport.tx.slot_count = config_.slot_count;
        transport.tx.frame_layout.payload_size = config_.payload_size;
        auto transceiver = std::make_unique<roce::CpuRoceTransceiver>(std::move(transport));
        auto caller = std::make_unique<roce::detail::CpuRoceCallerTx>(*transceiver);
        roce::detail::CpuRoceCallerTx::Config extra;
        extra.extra_host = trailer_ ? trailer_->address : nullptr;
        extra.extra_host_bytes = trailer_bytes_;
        extra.clamp_path_mtu = config_.clamp_path_mtu;
        caller->configure(extra);
        const bool retry = !gpu_pointer && !config_.cuda_payload.host_mapped && dmabuf_.fd >= 0;
        try {
            transceiver->setup();
        } catch (const roce::RegistrationError&) {
            if (!retry) {
                throw;
            }
            caller.reset();
            transceiver.reset();
            if (dmabuf_.fd >= 0) {
                ::close(dmabuf_.fd);
                dmabuf_.fd = -1;
            }
            bring_up(true);
            return;
        }
        transceiver->connect(config_.peer);
        transceiver_ = std::move(transceiver);
        caller_ = std::move(caller);
    }

    void report(const char* message) const
    {
        if (config_.on_error != nullptr && message != nullptr) {
            config_.on_error(config_.on_error_context, message);
        }
    }

    CpuRoceFrameTransmitter::Slot* slot_at(std::uint32_t index) const
    {
        return &kept_->slots[index];
    }

    CpuRoceFrameTransmitter::Slot* miss(const char* why)
    {
        if (pool_ != nullptr) {
            pool_->add_error();
        }
        report(why);
        return nullptr;
    }

    bool loaned(const CpuRoceFrameTransmitter::Slot* slot) const
    {
        if (slot == nullptr || pool_ == nullptr || slot->block_ != pool_.get()) {
            return false;
        }
        return pool_->is_loaned(slot->index_);
    }

    int poll_once()
    {
        if (caller_ == nullptr) {
            return 0;
        }
        int count = 0;
        try {
            count = caller_->poll(completions_, 16);
        } catch (const std::invalid_argument&) {
            return 0;
        } catch (const std::out_of_range&) {
            return 0;
        } catch (const std::exception& error) {
            failed_.store(true, std::memory_order_release);
            report(error.what());
            return 0;
        }
        for (int index = 0; index < count; ++index) {
            const auto& completion = completions_[index];
            if (completion.status == roce::detail::CpuRoceCallerTx::Status::Error) {
                failed_.store(true, std::memory_order_release);
                report("send completion failed");
            }
            const bool success = completion.status == roce::detail::CpuRoceCallerTx::Status::Ok;
            pool_->retire(completion.wr_id, success, nullptr);
        }
        return count;
    }

    bool stage_trailer(
        std::uint32_t index, const void* trailer, std::uint32_t trailer_bytes) const
    {
        if (trailer_bytes > config_.trailer_capacity) {
            return false;
        }
        if (trailer_bytes == 0) {
            return true;
        }
        if (trailer == nullptr || trailer_ == nullptr) {
            return false;
        }
        auto* destination = static_cast<std::uint8_t*>(trailer_->address)
            + static_cast<std::size_t>(index) * config_.trailer_capacity;
        std::memcpy(destination, trailer, trailer_bytes);
        return true;
    }

    CpuRoceFrameTransmitter::SendStatus post_staged(
        std::uint32_t index, std::uint32_t bytes, std::uint32_t trailer_bytes)
    {
        const std::uint8_t* extra = nullptr;
        if (trailer_bytes != 0) {
            extra = static_cast<std::uint8_t*>(trailer_->address)
                + static_cast<std::size_t>(index) * config_.trailer_capacity;
        }
        std::uint32_t psn = 0;
        try {
            psn = caller_->post(index, bytes, index, extra, trailer_bytes);
        } catch (const std::invalid_argument& error) {
            pool_->abandon(index, true);
            report(error.what());
            return SendStatus::Failed;
        } catch (const std::out_of_range& error) {
            pool_->abandon(index, true);
            report(error.what());
            return SendStatus::Failed;
        } catch (const std::exception& error) {
            failed_.store(true, std::memory_order_release);
            pool_->abandon(index, true);
            report(error.what());
            return SendStatus::Failed;
        }
        pool_->mark_inflight(index, psn, bytes);
        return SendStatus::Accepted;
    }

    CpuRoceFrameTransmitter::SendStatus wait_completion(std::uint32_t index)
    {
        const auto deadline = Clock::now() + config_.send_timeout;
        while (Clock::now() < deadline) {
            int count = 0;
            try {
                count = caller_->poll(completions_, 16);
            } catch (const std::exception& error) {
                if (frame_tx::failure_is_terminal(error)) {
                    failed_.store(true, std::memory_order_release);
                }
                report(error.what());
                if (pool_ != nullptr) {
                    pool_->add_error();
                }
                return SendStatus::Failed;
            }
            for (int entry = 0; entry < count; ++entry) {
                const auto& completion = completions_[entry];
                if (completion.status == roce::detail::CpuRoceCallerTx::Status::Error) {
                    failed_.store(true, std::memory_order_release);
                    report("send completion failed");
                }
                const bool success = completion.status == roce::detail::CpuRoceCallerTx::Status::Ok;
                bool retired_success = false;
                if (!pool_->retire(completion.wr_id, success, &retired_success)) {
                    continue;
                }
                if (completion.wr_id == index) {
                    return retired_success ? SendStatus::Completed : SendStatus::Failed;
                }
            }
        }
        report("send timed out");
        if (pool_ != nullptr) {
            pool_->add_error();
        }
        return SendStatus::Timeout;
    }

    std::optional<std::uint32_t> wait_sync(std::chrono::milliseconds capacity_timeout)
    {
        const auto deadline = Clock::now() + capacity_timeout;
        for (;;) {
            if (!open_.load(std::memory_order_acquire)
                || failed_.load(std::memory_order_acquire)) {
                return std::nullopt;
            }
            if (auto index = pool_->claim()) {
                return index;
            }
            if (owner_.try_lock()) {
                poll_once();
                owner_.unlock();
                if (auto index = pool_->claim()) {
                    return index;
                }
            }
            if (Clock::now() >= deadline) {
                return std::nullopt;
            }
            const auto slice = Clock::now() + std::chrono::milliseconds(1);
            std::unique_lock<std::mutex> lock(pool_->mutex());
            pool_->wake().wait_until(lock, slice < deadline ? slice : deadline, [this] {
                return !open_.load(std::memory_order_acquire)
                    || failed_.load(std::memory_order_acquire) || pool_->has_free();
            });
        }
    }

    std::optional<std::uint32_t> wait_worker(std::chrono::milliseconds capacity_timeout)
    {
        const auto deadline = Clock::now() + capacity_timeout;
        while (open_.load(std::memory_order_acquire)
            && !failed_.load(std::memory_order_acquire)
            && Clock::now() < deadline) {
            if (auto index = pool_->claim()) {
                return index;
            }
            std::unique_lock<std::mutex> lock(pool_->mutex());
            pool_->wake().wait_until(lock, deadline, [this] {
                return !open_.load(std::memory_order_acquire)
                    || failed_.load(std::memory_order_acquire) || pool_->has_free();
            });
        }
        if (open_.load(std::memory_order_acquire)
            && !failed_.load(std::memory_order_acquire)) {
            return pool_->claim();
        }
        return std::nullopt;
    }

    void run_worker()
    {
        while (open_.load(std::memory_order_acquire)) {
            std::uint32_t index = 0;
            std::uint32_t bytes = 0;
            std::uint32_t trailer_bytes = 0;
            void* ready = nullptr;
            if (pool_->pop_queued(&index, &bytes, &trailer_bytes, &ready)) {
                if (ready != nullptr
                    && cuEventSynchronize(static_cast<CUevent>(ready)) != CUDA_SUCCESS) {
                    pool_->abandon(index, true);
                    report("ready event wait failed");
                    continue;
                }
                if (!open_.load(std::memory_order_acquire)
                    || failed_.load(std::memory_order_acquire)) {
                    pool_->abandon(index, true);
                    report("send rejected");
                    continue;
                }
                post_staged(index, bytes, trailer_bytes);
                poll_once();
                continue;
            }
            if (failed_.load(std::memory_order_acquire)) {
                poll_once();
                std::unique_lock<std::mutex> lock(pool_->mutex());
                pool_->wake().wait_for(lock, std::chrono::milliseconds(100), [this] {
                    return !open_.load(std::memory_order_acquire);
                });
                continue;
            }
            if (pool_->inflight() != 0) {
                poll_once();
                continue;
            }
            std::unique_lock<std::mutex> lock(pool_->mutex());
            pool_->wake().wait_for(lock, std::chrono::milliseconds(100), [this] {
                return !open_.load(std::memory_order_acquire) || pool_->has_queued();
            });
        }
    }

    void start()
    {
        if (stopped_ || started_) {
            return;
        }
        open_.store(true, std::memory_order_release);
        if (config_.mode == SenderMode::Worker) {
            try {
                worker_ = std::thread([this] {
                    try {
                        run_worker();
                    } catch (const std::exception& error) {
                        failed_.store(true, std::memory_order_release);
                        open_.store(false, std::memory_order_release);
                        report(error.what());
                    } catch (...) {
                        failed_.store(true, std::memory_order_release);
                        open_.store(false, std::memory_order_release);
                        report("worker failed");
                    }
                });
            } catch (...) {
                open_.store(false, std::memory_order_release);
                throw;
            }
        }
        started_ = true;
    }

    void stop()
    {
        if (stopped_) {
            return;
        }
        stopped_ = true;
        open_.store(false, std::memory_order_release);
        if (pool_ != nullptr) {
            std::lock_guard<std::mutex> lock(pool_->mutex());
            pool_->wake().notify_all();
        }
        std::unique_lock<std::mutex> owner;
        if (worker_.joinable()) {
            worker_.join();
        } else {
            owner = std::unique_lock<std::mutex>(owner_);
        }
        if (pool_ != nullptr) {
            pool_->drop_queued();
        }
        const auto deadline = Clock::now() + std::chrono::seconds(1);
        while (pool_ != nullptr && pool_->inflight() != 0 && Clock::now() < deadline) {
            poll_once();
        }
        if (pool_ != nullptr && pool_->inflight() != 0 && caller_ != nullptr) {
            try {
                caller_->fail();
            } catch (const std::exception& error) {
                report(error.what());
            }
            while (poll_once() > 0) {
            }
        }
        if (pool_ != nullptr) {
            pool_->count_inflight_errors();
        }
    }

    void close()
    {
        if (closed_) {
            return;
        }
        closed_ = true;
        stop();
        caller_.reset();
        transceiver_.reset();
        if (pool_ != nullptr) {
            pool_->retire_inflight();
        }
        if (stream_ != nullptr) {
            if (cuCtxPushCurrent(context_) == CUDA_SUCCESS) {
                cuStreamSynchronize(stream_);
                cuStreamDestroy(stream_);
                CUcontext popped = nullptr;
                cuCtxPopCurrent(&popped);
            }
            stream_ = nullptr;
        }
        if (dmabuf_.fd >= 0) {
            ::close(dmabuf_.fd);
            dmabuf_.fd = -1;
        }
        pool_.reset();
        kept_.reset();
    }

    CpuRoceFrameTransmitter::Slot* claim()
    {
        if (pool_ == nullptr) {
            return nullptr;
        }
        if (auto index = pool_->claim()) {
            return slot_at(*index);
        }
        if (config_.mode == SenderMode::Worker || !owner_.try_lock()) {
            return nullptr;
        }
        poll_once();
        auto index = pool_->claim();
        owner_.unlock();
        return index ? slot_at(*index) : nullptr;
    }

    CpuRoceFrameTransmitter::Slot* copy_in(const void* source, std::uint32_t bytes, CudaEvent ready,
        std::chrono::milliseconds capacity_timeout)
    {
        if (pool_ == nullptr) {
            return nullptr;
        }
        if (source == nullptr || bytes == 0 || bytes > config_.payload_size) {
            return miss("copy_in rejected");
        }
        const auto index = config_.mode == SenderMode::Worker
            ? wait_worker(capacity_timeout)
            : wait_sync(capacity_timeout);
        if (!index) {
            if (!open_.load(std::memory_order_acquire)) {
                return miss("stopped while waiting for a pool buffer");
            }
            if (failed_.load(std::memory_order_acquire)) {
                return miss("sender failed while waiting for a pool buffer");
            }
            return miss("no free slot");
        }
        try {
            CudaContextPush current(context_);
            CUresult result = CUDA_SUCCESS;
            if (ready != nullptr) {
                result = cuStreamWaitEvent(stream_, static_cast<CUevent>(ready), 0);
            }
            if (result == CUDA_SUCCESS) {
                result = cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(pool_->data(*index)),
                    reinterpret_cast<CUdeviceptr>(source), bytes, stream_);
            }
            if (result == CUDA_SUCCESS) {
                result = cuStreamSynchronize(stream_);
            }
            check_cuda(result);
        } catch (const std::exception& error) {
            pool_->abandon(*index, true);
            report(error.what());
            return nullptr;
        }
        return slot_at(*index);
    }

    CpuRoceFrameTransmitter::SendStatus send(CpuRoceFrameTransmitter::Slot* slot,
        std::uint32_t bytes, CudaEvent ready, const void* trailer, std::uint32_t trailer_bytes)
    {
        if (!loaned(slot)) {
            return SendStatus::Failed;
        }
        const std::uint32_t index = slot->index_;
        if (bytes == 0 || bytes > slot->capacity_) {
            pool_->abandon(index, true);
            report("send length is outside the slot");
            return SendStatus::Failed;
        }
        if (config_.mode == SenderMode::Synchronous && ready != nullptr
            && cuEventSynchronize(static_cast<CUevent>(ready)) != CUDA_SUCCESS) {
            pool_->abandon(index, true);
            report("ready event wait failed");
            return SendStatus::Failed;
        }
        std::unique_lock<std::mutex> owner;
        if (config_.mode == SenderMode::Synchronous) {
            owner = std::unique_lock<std::mutex>(owner_);
        }
        if (!open_.load(std::memory_order_acquire) || failed_.load(std::memory_order_acquire)) {
            pool_->abandon(index, true);
            report("send rejected");
            return SendStatus::Rejected;
        }
        if (!stage_trailer(index, trailer, trailer_bytes)) {
            pool_->abandon(index, true);
            report("trailer exceeds capacity");
            return SendStatus::Failed;
        }
        if (config_.mode == SenderMode::Worker) {
            if (!pool_->enqueue(index, bytes, trailer_bytes, ready)) {
                return SendStatus::Failed;
            }
            return SendStatus::Accepted;
        }
        if (post_staged(index, bytes, trailer_bytes) != SendStatus::Accepted) {
            return SendStatus::Failed;
        }
        return wait_completion(index);
    }

    Config config_;
    CUcontext context_ = nullptr;
    OwnedFd dmabuf_;
    std::unique_ptr<AlignedBytes> trailer_;
    std::uint32_t trailer_bytes_ = 0;
    std::shared_ptr<Pool> pool_;
    std::shared_ptr<Kept> kept_;
    std::unique_ptr<roce::CpuRoceTransceiver> transceiver_;
    std::unique_ptr<roce::detail::CpuRoceCallerTx> caller_;
    roce::detail::CpuRoceCallerTx::Completion completions_[16];
    CUstream stream_ = nullptr;
    std::mutex owner_;
    std::thread worker_;
    std::atomic<bool> open_ { false };
    std::atomic<bool> failed_ { false };
    bool started_ = false;
    bool stopped_ = false;
    bool closed_ = false;
};

CpuRoceFrameTransmitter::CpuRoceFrameTransmitter(Config config)
    : impl_(std::make_unique<Impl>(std::move(config)))
{
}

CpuRoceFrameTransmitter::~CpuRoceFrameTransmitter() = default;

CpuRoceFrameTransmitter::Slot* CpuRoceFrameTransmitter::claim()
{
    return impl_->claim();
}

void CpuRoceFrameTransmitter::release(Slot* slot, CudaEvent done) noexcept
{
    if (slot == nullptr || slot->block_ == nullptr) {
        return;
    }
    if (done != nullptr) {
        cuEventSynchronize(static_cast<CUevent>(done));
    }
    static_cast<frame_tx::Pool*>(slot->block_)->release_loaned(slot->index_);
}

CpuRoceFrameTransmitter::Slot* CpuRoceFrameTransmitter::copy_in(const void* source,
    std::uint32_t bytes, CudaEvent ready, std::chrono::milliseconds capacity_timeout)
{
    return impl_->copy_in(source, bytes, ready, capacity_timeout);
}

CpuRoceFrameTransmitter::SendStatus CpuRoceFrameTransmitter::send(
    Slot* slot, std::uint32_t bytes, CudaEvent ready, const void* trailer,
    std::uint32_t trailer_bytes)
{
    return impl_->send(slot, bytes, ready, trailer, trailer_bytes);
}

void CpuRoceFrameTransmitter::start()
{
    impl_->start();
}

void CpuRoceFrameTransmitter::stop()
{
    impl_->stop();
}

void CpuRoceFrameTransmitter::close()
{
    impl_->close();
}

CpuRoceFrameTransmitter::Stats CpuRoceFrameTransmitter::stats() const
{
    Stats out;
    if (impl_->pool_ == nullptr) {
        return out;
    }
    const auto stats = impl_->pool_->stats();
    out.bytes = stats.bytes;
    out.frames = stats.frames;
    out.errors = stats.errors;
    out.last_psn = stats.last_psn;
    return out;
}

std::uint32_t CpuRoceFrameTransmitter::qp_number() const
{
    return impl_->transceiver_ != nullptr ? impl_->transceiver_->get_qp_number() : 0;
}

std::uint32_t CpuRoceFrameTransmitter::path_mtu() const
{
    return impl_->transceiver_ != nullptr ? impl_->transceiver_->get_path_mtu() : 0;
}

bool CpuRoceFrameTransmitter::failed() const
{
    return impl_->failed_.load(std::memory_order_acquire);
}

} // namespace hololink::operators
