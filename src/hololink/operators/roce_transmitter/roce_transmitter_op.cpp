/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "roce_transmitter_op.hpp"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

#include <cuda_runtime.h>
#include <infiniband/mlx5dv.h>
#include <unistd.h>

#include <hololink/common/cuda_error.hpp>
#include <hololink/common/cuda_helper.hpp>
#include <hololink/core/logging_internal.hpp>
#include <hololink/core/networking.hpp>
#include <hololink/transport/roce/ibv.hpp>

#include "roce_transmitter_buffer.hpp"

#define THROW_ERROR(streamable)                   \
    do {                                          \
        using namespace hololink::operators;      \
        std::stringstream ss;                     \
        ss << streamable;                         \
        HSB_LOG_ERROR("{}", ss.str());            \
        throw RoceTransmitterOp::Error(ss.str()); \
    } while (0)

namespace hololink::operators {
namespace ibv = hololink::transport::roce::ibv;
namespace {

    enum cudaAllocateType {
        cudaAllocateTypeHost,
        cudaAllocateTypeDevice,
        cudaAllocateTypeManaged,
    };

    struct CudaDeleter {
        CudaDeleter(enum cudaAllocateType mem_type = cudaAllocateType::cudaAllocateTypeDevice)
            : mem_type_(mem_type)
        {
        }
        void operator()(char* ptr) const
        {
            switch (mem_type_) {
            case cudaAllocateTypeDevice:
            case cudaAllocateTypeManaged:
                cudaFree(ptr);
                break;
            case cudaAllocateTypeHost:
                cudaFreeHost(ptr);
                break;
            default:
                throw std::runtime_error("Invalid memory type");
            }
        }

    private:
        enum cudaAllocateType mem_type_;
    };

    template <typename T>
    using CudaUniquePtr = std::unique_ptr<T, CudaDeleter>;

    template <typename T>
    CudaUniquePtr<T> CudaRDMAAllocate(size_t length = 1)
    {
        T* ptr = nullptr;
        int device_id = 0;
        CUDA_CHECK(cudaGetDevice(&device_id));
        int integrated = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, device_id));
        if (integrated) {
            int can_map_host_mem = 0;
            CUDA_CHECK(cudaDeviceGetAttribute(&can_map_host_mem, cudaDevAttrCanMapHostMemory, device_id));
            if (can_map_host_mem) {
                CUDA_CHECK(cudaMallocHost(&ptr, sizeof(T) * length));
                return CudaUniquePtr<T>(ptr, CudaDeleter(cudaAllocateType::cudaAllocateTypeHost));
            } else {
                // Not guaranteed to work for RDMA, but cudaMallocManaged is the
                // best available option when the device reports no mappable host
                // memory; if RDMA fails, this operator will not function.
                CUDA_CHECK(cudaMallocManaged(&ptr, sizeof(T) * length));
                return CudaUniquePtr<T>(ptr, CudaDeleter(cudaAllocateType::cudaAllocateTypeManaged));
            }
        }
        CUDA_CHECK(cudaMalloc(&ptr, sizeof(T) * length));
        return CudaUniquePtr<T>(ptr, CudaDeleter(cudaAllocateType::cudaAllocateTypeDevice));
    }

    class CudaRDMABuffer {
    public:
        CudaRDMABuffer() = default;

        explicit CudaRDMABuffer(size_t payload_size)
            : page_size_(static_cast<size_t>(getpagesize()))
            , layout_(detail::dma_buf_layout(payload_size, page_size_))
            , allocation_(CudaRDMAAllocate<char>(layout_.allocation_size))
            , data_(detail::page_aligned_pointer(allocation_.get(), page_size_))
        {
        }

        char* data() const { return data_; }
        size_t size() const { return layout_.registered_size; }

    private:
        size_t page_size_ {};
        detail::DmaBufLayout layout_ {};
        CudaUniquePtr<char> allocation_;
        char* data_ {};
    };

    struct CudaStream {
        CudaStream()
        {
            CUDA_CHECK(cudaStreamCreate(&stream_));
        }
        CudaStream(const CudaStream&) = delete;
        CudaStream(CudaStream&& other)
            : stream_(other.stream_)
        {
            other.stream_ = nullptr;
        }
        CudaStream& operator=(const CudaStream&) = delete;
        CudaStream& operator=(CudaStream&& other)
        {
            swap(*this, other);
            return *this;
        }
        ~CudaStream()
        try {
            if (stream_ != nullptr)
                CUDA_CHECK(cudaStreamDestroy(stream_));
        } catch (const CudaError&) {
            return;
        }

        friend void swap(CudaStream& lhs, CudaStream& rhs) noexcept
        {
            using std::swap;
            swap(lhs.stream_, rhs.stream_);
        }

        operator cudaStream_t() const
        {
            return stream_;
        }

        cudaStream_t stream_;
    };

    /// A CUDA event without timing, for ordering one stream after another.
    struct CudaEvent {
        CudaEvent()
        {
            CUDA_CHECK(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
        }
        CudaEvent(const CudaEvent&) = delete;
        CudaEvent& operator=(const CudaEvent&) = delete;
        ~CudaEvent()
        try {
            CUDA_CHECK(cudaEventDestroy(event_));
        } catch (const CudaError&) {
            return;
        }

        operator cudaEvent_t() const
        {
            return event_;
        }

        cudaEvent_t event_ {};
    };

} // unnamed namespace

class RoceTransmitterOp::Buffer {
public:
    Buffer(ibv::ProtectionDomain& protection_domain, size_t size)
        : output_buffer_size_(size)
        , output_buffer_(output_buffer_size_)
        , memory_region_(register_dmabuf_memory(
              protection_domain, output_buffer_.data(), output_buffer_.size()))
    {
    }

    ~Buffer()
    {
        if (dmabuf_fd_ >= 0) {
            ::close(dmabuf_fd_);
            dmabuf_fd_ = -1;
        }
    }

    ibv::ScatterGatherElement get_scatter_gather_element()
    {
        if (input_tensor_) {
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            input_tensor_.reset();
        }

        return ibv::ScatterGatherElement(
            reinterpret_cast<uint64_t>(output_buffer_.data()),
            payload_size_,
            memory_region_.ptr_->lkey);
    }

    /// Copies tensor into the staging buffer. producer_stream is the activation stream of
    /// the compute() that received the tensor: the runtime queues the producer's readiness
    /// there, so the copy, which runs on this buffer's own stream, is ordered after it.
    /// Without that the copy can read the tensor's pool memory before the producer's write
    /// to it has landed (seen on Thor's iGPU: small payloads arrived as zeros).
    void Store(const holoscan::Tensor& tensor, cudaStream_t producer_stream)
    {
        if (static_cast<size_t>(tensor.nbytes()) > output_buffer_size_)
            THROW_ERROR("Received payload size (" << tensor.nbytes() << ") is larger than the allowed size (" << output_buffer_size_ << ")");

        // Copy the Tensor value — increments the internal refcount so the upstream
        // HSDK5 pool cannot recycle the tensor's GPU memory until input_tensor_ is reset().
        input_tensor_ = tensor;
        payload_size_ = input_tensor_->nbytes();
        CUDA_CHECK(cudaEventRecord(producer_ready_, producer_stream));
        CUDA_CHECK(cudaStreamWaitEvent(stream_, producer_ready_, 0));
        CUDA_CHECK(cudaMemcpyAsync(output_buffer_.data(), input_tensor_->data(), payload_size_, cudaMemcpyDeviceToDevice, stream_));
    }

private:
    ibv::MemoryRegion register_dmabuf_memory(ibv::ProtectionDomain& protection_domain, void* addr, size_t size)
    {
        int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;

        // Try to get dmabuf fd from CUDA memory
        using namespace hololink::common;
        CUresult cu_result = cuMemGetHandleForAddressRange(
            (void*)&dmabuf_fd_, reinterpret_cast<CUdeviceptr>(addr),
            size, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);

        if (cu_result == CUDA_SUCCESS) {
            // Try to register memory region using dmabuf
            try {
                uint64_t iova = reinterpret_cast<uint64_t>(addr);
                auto mr = protection_domain.register_dmabuf_memory_region(0, size, iova, dmabuf_fd_, access);
                HSB_LOG_DEBUG("Successfully registered memory using dmabuf for p={:#x}, size={}", iova, size);
                return mr;
            } catch (const std::exception& e) {
                HSB_LOG_WARN("Failed to register dmabuf memory region ({}), falling back to ibv_reg_mr", e.what());
                ::close(dmabuf_fd_);
                dmabuf_fd_ = -1;
            }
        } else {
            HSB_LOG_WARN("Cannot get dmabuf fd for memory region p={} size={}, cu_result={}, falling back to ibv_reg_mr",
                addr, size, cu_result);
            dmabuf_fd_ = -1;
        }

        // Fallback to traditional memory registration
        HSB_LOG_DEBUG("Using traditional ibv_reg_mr for p={}, size={}", addr, size);
        return protection_domain.register_memory_region(addr, size, access);
    }

    CudaStream stream_;
    CudaEvent producer_ready_; // the producer's progress at Store(), waited for on stream_
    size_t output_buffer_size_;
    CudaRDMABuffer output_buffer_;

    // std::optional<Tensor> keeps the GPU buffer alive via Tensor's internal
    // DLManagedTensorContext refcount until reset() is called after transmission.
    std::optional<holoscan::Tensor> input_tensor_;
    size_t payload_size_ {};
    // Declared before memory_region_: register_dmabuf_memory() stores the fd while memory_region_
    // is initialized, which would otherwise be overwritten by this initializer and never closed.
    int dmabuf_fd_ { -1 };
    ibv::MemoryRegion memory_region_;
};

struct RoceTransmitterOp::Resource {
    Resource(const std::string& ibv_name, unsigned ibv_port, size_t buffer_size,
        size_t queue_size, const std::string& hololink_ip, uint32_t ibv_qp);

    using BufferQueue = std::queue<std::shared_ptr<RoceTransmitterOp::Buffer>>;

    ibv::Context context_;
    ::ibv_device_attr device_attr_;
    std::string host_if_name_;
    unsigned ibv_port_;
    ::ibv_port_attr port_attr_;
    int gid_index_;
    ::ibv_gid local_gid_;
    ibv::ProtectionDomain protection_domain_;
    ibv::CompletionQueue completion_queue_;
    size_t buffer_size_;
    BufferQueue available_buffers_; // Buffers that are available to be used
    BufferQueue pending_buffers_; // Buffers that are ready to be transmitted
    // Buffers are waiting for the transmission to be completed including their wr_id
    std::queue<std::pair<uint64_t, std::shared_ptr<RoceTransmitterOp::Buffer>>> sent_buffers_;
    ibv::QueuePair queue_pair_;

    ::ibv_gid remote_gid_;
    uint32_t remote_qp_num_;
};

RoceTransmitterOp::Resource::Resource(const std::string& ibv_name, unsigned ibv_port, size_t buffer_size,
    size_t queue_size, const std::string& hololink_ip, uint32_t ibv_qp)
    : context_(ibv_name)
    , device_attr_(context_.query_device())
    , host_if_name_(context_.get_host_if_name())
    , ibv_port_(ibv_port)
    , port_attr_(context_.query_port(ibv_port))
    , gid_index_(context_.query_gid_roce_v2_ip(static_cast<int>(ibv_port)))
    , local_gid_(context_.query_gid(ibv_port, gid_index_))
    // Allocate Protection Domain
    , protection_domain_(context_.allocate_protection_domain())
    // Create Completion Queue
    , completion_queue_(context_.create_completion_queue(queue_size))
    // Allocate and register the memory buffers that will hold the data
    , buffer_size_(buffer_size)
    , available_buffers_([this, queue_size] {
        BufferQueue queue;
        for (size_t i = 0; i < queue_size; ++i)
            queue.emplace(std::make_shared<Buffer>(protection_domain_, buffer_size_));
        return queue;
    }())
    , queue_pair_(
          protection_domain_.create_queue_pair(
              ibv::QueuePair::InitAttr().query_pair_type(IBV_QPT_UC).send_completion_queue(completion_queue_.ptr_).receive_completion_queue(completion_queue_.ptr_).capacity(ibv::QueuePair::Capacity().max_send_write_read(512).max_receive_write_read(512).max_send_scatter_gather_entry(1).max_receive_scatter_gather_entry(1))))
    , remote_gid_(ibv::ipv4_to_gid(hololink_ip))
    , remote_qp_num_(ibv_qp)
{
    HSB_LOG_INFO("IB resource allocated successfully");
}

void RoceTransmitterOp::setup(holoscan::OperatorSpec& spec)
{
    spec.input(input, "input")
        .expects_tensor(holoscan::TensorInputSpec {
            .representation = holoscan::TensorRepresentation {
                .memory_kind = holoscan::MemoryKind::kCudaDevice,
            },
        });
}

holoscan::Contract RoceTransmitterOp::contract() const
{
    holoscan::Contract result;
    result.trigger(holoscan::OnEach { input });
    return result;
}

void RoceTransmitterOp::start()
{
    const auto& p = params();

    if (p.queue_size == 0)
        THROW_ERROR("queue_size must not be zero");

    // ibv calls seem to have trouble with
    // reentrancy.  No problem; since we're only
    // at startup, run just one at a time.  This
    // only affects multithreaded schedulers like
    // EventBasedScheduler.
    std::lock_guard lock(get_lock());

    // ibverbs has some reentrancy problems with programs
    // that use fork, which may happen in applications we're
    // used in.
    static bool ibv_fork_init_done = false;
    if (!ibv_fork_init_done) {
        int r = ibv_fork_init();
        if (r != 0)
            THROW_ERROR("ibv_fork_init failed, r=" << r);
        ibv_fork_init_done = true;
    }

    // Set the CUDA device before constructing Resource: Buffer::CudaRDMAAllocate()
    // calls cudaMalloc* and cudaStreamCreate() during construction, so the device
    // must be current before those calls.
    CUDA_CHECK(cudaSetDevice(p.cuda_device_id));

    // Create resource. Context requires a device name, so an empty name is
    // resolved here: the device on the network interface that reaches the board.
    std::string ibv_name = p.ibv_name;
    uint32_t ibv_port = p.ibv_port;
    try {
        if (ibv_name.empty()) {
            std::tie(ibv_name, ibv_port) = hololink::core::infiniband_device_for_peer(p.hololink_ip);
            HSB_LOG_INFO("Using IB device {} port {} for the board at {}", ibv_name, ibv_port, p.hololink_ip);
        }
        resource_ = std::make_unique<Resource>(
            ibv_name, ibv_port, p.buffer_size, p.queue_size, p.hololink_ip, p.ibv_qp);

        // Connecting the QP by transitioning the state to Ready-to-Send.
        // We are doing one sided bring up for TX
        if (!connect())
            THROW_ERROR("Failed to connect QPs");
    } catch (const ibv::Error& e) {
        THROW_ERROR(e.what());
    }

    if (p.on_start)
        p.on_start(ConnectionInfo { resource_->queue_pair_.ptr_->qp_num });

    running_ = true;
    completion_thread_ = std::thread(&RoceTransmitterOp::completion_thread, this);
    transmission_thread_ = std::thread(&RoceTransmitterOp::transmission_thread, this);
}

void RoceTransmitterOp::stop()
{
    const auto& p = params();
    if (p.on_stop)
        p.on_stop(ConnectionInfo { resource_->queue_pair_.ptr_->qp_num });
    // transmission_thread() re-checks running_ inside cv_.wait's predicate, which it
    // evaluates holding mutex_.  Clearing the flag here without that mutex leaves a
    // window where the predicate has already read running_ as true but the thread has
    // not yet parked on cv_: the notify_all() below lands on an empty wait queue, is
    // lost, and the thread sleeps until the next notify -- which never comes, so the
    // join() below hangs forever.  Take mutex_ so the store cannot fall in that window.
    {
        std::lock_guard<std::mutex> lock(mutex_);
        running_ = false;
    }
    cv_.notify_all();
    transmission_thread_.join();
    completion_thread_.join();
    resource_.reset();
}

holoscan::expected<void, holoscan::Error> RoceTransmitterOp::compute(
    holoscan::ExecutionContext& context)
{
    auto maybe_sample = input.receive();
    if (!maybe_sample) {
        HSB_LOG_ERROR("Failed to receive tensor on port 'input'");
        return holoscan::make_unexpected(std::move(maybe_sample).error());
    }
    queue_buffer(std::move(*maybe_sample), context.cancellation(), context.cuda_stream().get());
    return {};
}

bool RoceTransmitterOp::flush(std::chrono::milliseconds timeout)
{
    std::unique_lock<std::mutex> lock(mutex_);
    if (!resource_)
        return true;
    auto condition = [this] {
        return !running_ || (resource_->pending_buffers_.empty() && resource_->sent_buffers_.empty());
    };
    if (timeout.count() > 0)
        cv_.wait_for(lock, timeout, condition);
    else
        cv_.wait(lock, condition);
    return resource_->pending_buffers_.empty() && resource_->sent_buffers_.empty();
}

void RoceTransmitterOp::queue_buffer(holoscan::Sample<holoscan::Tensor> sample,
    const holoscan::CancellationView& cancellation, cudaStream_t producer_stream)
{
    std::unique_lock<std::mutex> lock(mutex_);
    // Also exit on shutdown so stop() cannot deadlock a compute thread waiting
    // for a buffer slot while the transmission thread has already exited -- and on a stop
    // request, since stop() only runs once this compute() returns: with the HSB holding
    // transmission paused, no slot may ever free up.
    while (!cv_.wait_for(lock, std::chrono::milliseconds(100),
        [this] { return !running_ || resource_->available_buffers_.size(); })) {
        if (cancellation.stop_requested()) {
            HSB_LOG_WARN("Stop requested while waiting for a free transmit buffer; dropping the input");
            return;
        }
    }
    if (!running_)
        return;
    auto pending_buffer = resource_->available_buffers_.front();
    resource_->available_buffers_.pop();
    // Store copies the Tensor value (increments internal refcount) so the GPU buffer
    // stays alive until the transmission thread calls input_tensor_.reset().
    pending_buffer->Store(sample.data, producer_stream);
    resource_->pending_buffers_.push(std::move(pending_buffer));
    cv_.notify_all();
}

bool RoceTransmitterOp::connect()
{
    // Connecting the QP by transitioning the state to Ready-to-Send.
    // We are doing one sided bring up for TX
    if (int r = resource_->queue_pair_.reset_to_init(resource_->ibv_port_)) {
        HSB_LOG_ERROR("Failed to modify QP state to INIT, {}", strerror(r));
        return false;
    }
    if (int r = resource_->queue_pair_.init_to_rtr(resource_->ibv_port_, resource_->port_attr_,
            resource_->host_if_name_, resource_->remote_gid_, resource_->remote_qp_num_,
            resource_->gid_index_)) {
        HSB_LOG_ERROR("Cannot modify queue pair to IBV_QPS_RTR, errno={} ({}).", r, strerror(r));
        return false;
    }
    if (int r = resource_->queue_pair_.rtr_to_rts()) {
        HSB_LOG_ERROR("Failed to modify QP state to RTS, {}", strerror(r));
        return false;
    }
    return true;
}

bool RoceTransmitterOp::send(uint64_t id, std::shared_ptr<Buffer> buffer)
try {
    ibv::ScatterGatherElement sge(buffer->get_scatter_gather_element());
    ibv::SendWorkRequest wr(id, &sge);

    if (int err = resource_->queue_pair_.post_send(wr)) {
        HSB_LOG_ERROR("Failed to post SEND Request, {}", strerror(err));
        return false;
    }
    return true;
} catch (const std::exception&) {
    return false;
}

void RoceTransmitterOp::completion_thread()
{
    try {
        while (running_) {
            // poll the completion
            std::optional<::ibv_wc> work_completion;
            while (!(work_completion = resource_->completion_queue_.poll())) {
                if (!running_)
                    return;
                std::this_thread::yield();
            }

            // CQE found
            if (work_completion->status != IBV_WC_SUCCESS) {
                HSB_LOG_ERROR("Failed to complete Work Request: {}", ibv_wc_status_str(work_completion->status));
                // The buffer is stranded in sent_buffers_ and will never return to
                // available_buffers_.  Signal shutdown so queue_buffer() exits its
                // wait instead of deadlocking (especially with queue_size == 1).
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    running_ = false;
                }
                cv_.notify_all();
                return;
            }

            switch (work_completion->opcode) {
            case IBV_WC_SEND: {
                HSB_LOG_DEBUG("Send Request {} completed", work_completion->wr_id);
                std::unique_lock<std::mutex> lock(mutex_);
                auto id_buffer_pair = std::move(resource_->sent_buffers_.front());
                resource_->sent_buffers_.pop();
                if (id_buffer_pair.first != work_completion->wr_id)
                    HSB_LOG_ERROR("Send Request {} received while expected {}", work_completion->wr_id, id_buffer_pair.first);
                // Return the buffer to the available buffers queue
                resource_->available_buffers_.push(std::move(id_buffer_pair.second));
                cv_.notify_all();
                break;
            }
            default:
                HSB_LOG_ERROR("Unexpected completion opcode {}", static_cast<int>(work_completion->opcode));
                continue;
            }
        }
    } catch (const std::exception& e) {
        HSB_LOG_ERROR("RoCE completion thread error: {}", e.what());
        {
            std::lock_guard<std::mutex> lock(mutex_);
            running_ = false;
        }
        cv_.notify_all();
    }
}

void RoceTransmitterOp::transmission_thread()
{
    try {
        uint64_t wr_id = 0;
        while (true) {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return !running_ || resource_->pending_buffers_.size(); });
            if (!running_)
                break;

            auto buffer = std::move(resource_->pending_buffers_.front());
            resource_->pending_buffers_.pop();
            auto id = wr_id++;
            resource_->sent_buffers_.emplace(std::make_pair(id, buffer));
            lock.unlock();

            if (!send(id, std::move(buffer))) {
                HSB_LOG_ERROR("Failed to send buffer");
                // Buffer is stranded in sent_buffers_ — no WC will arrive for
                // a failed post_send.  Signal shutdown so queue_buffer() exits.
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    running_ = false;
                }
                cv_.notify_all();
                break;
            }
            HSB_LOG_DEBUG("Buffer sent");
        }
    } catch (const std::exception& e) {
        HSB_LOG_ERROR("RoCE transmission thread error: {}", e.what());
        {
            std::lock_guard<std::mutex> lock(mutex_);
            running_ = false;
        }
        cv_.notify_all();
    }
}

std::mutex& RoceTransmitterOp::get_lock()
{
    // We want all RoceTransmitter instances in this process
    // to share the same lock.
    static std::mutex lock;
    return lock;
}

} // namespace hololink::operators
