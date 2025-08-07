/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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
#include <memory>
#include <queue>
#include <sstream>
#include <string>

#include <infiniband/mlx5dv.h>

#include <hololink/common/cuda_error.hpp>
#include <hololink/core/logging_internal.hpp>

#include "ibv.hpp"

#define THROW_ERROR(streamable)                   \
    do {                                          \
        using namespace hololink::operators;      \
        std::stringstream ss;                     \
        ss << streamable;                         \
        HSB_LOG_ERROR(ss.str());                  \
        throw RoceTransmitterOp::Error(ss.str()); \
    } while (0)

/**
 * @brief This macro defining a YAML converter which throws for unsupported types.
 *
 * Background: Holoscan supports setting parameters through YAML files. But for some parameters
 * accepted by the receiver operators like `DataChannel` class of functions it makes no sense
 * to specify them in YAML files. Therefore use a converter which throws for these types.
 *
 * @tparam TYPE
 */
#define YAML_CONVERTER(TYPE)                                                                \
    template <>                                                                             \
    struct YAML::convert<TYPE> {                                                            \
        static Node encode(TYPE&) { throw std::runtime_error("Unsupported"); }              \
                                                                                            \
        static bool decode(const Node&, TYPE&) { throw std::runtime_error("Unsupported"); } \
    };

YAML_CONVERTER(hololink::operators::RoceTransmitterOp::OnStartCallback);

namespace hololink::operators {
namespace {

    static const std::string input_name("input");
    struct CudaDeleter {
        void operator()(char* ptr) const
        {
            cudaFree(ptr);
        }
    };

    template <typename T>
    using CudaUniquePtr = std::unique_ptr<T, CudaDeleter>;

    template <typename T>
    CudaUniquePtr<T> CudaAllocate(size_t length = 1)
    {
        T* ptr;
        CUDA_CHECK(cudaMalloc(&ptr, sizeof(T) * length));
        return CudaUniquePtr<T>(ptr, CudaDeleter());
    }

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

} // unnamed namespace

class RoceTransmitterOp::Buffer {
public:
    Buffer(ibv::ProtectionDomain& protection_domain, size_t size)
        : output_buffer_size_(size)
        , output_buffer_(CudaAllocate<char>(output_buffer_size_))
        , memory_region_(protection_domain.register_memory_region(
              output_buffer_.get(),
              output_buffer_size_,
              IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE))
    {
    }

    Buffer(ibv::ProtectionDomain& protection_domain, const Tensor& buffer)
        : output_buffer_size_(0)
        , input_tensor_(buffer)
        , payload_size_(input_tensor_->nbytes())
        , memory_region_(protection_domain.register_memory_region(
              input_tensor_->data(),
              payload_size_,
              IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE))
    {
    }

    ibv::ScatterGatherElement get_scatter_gather_element()
    {
        if (output_buffer_size_ == 0) {
            return ibv::ScatterGatherElement(
                reinterpret_cast<uint64_t>(input_tensor_->data()),
                payload_size_,
                memory_region_.ptr_->lkey);
        }

        if (input_tensor_) {
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            input_tensor_.reset();
        }

        return ibv::ScatterGatherElement(
            reinterpret_cast<uint64_t>(output_buffer_.get()),
            payload_size_,
            memory_region_.ptr_->lkey);
    }

    void Store(const Tensor& tensor)
    {
        // No need to Store the data, the data is already stored in the input_tensor_
        if (tensor == input_tensor_)
            return;

        if (static_cast<size_t>(tensor->nbytes()) > output_buffer_size_)
            THROW_ERROR("Received payload size (" << tensor->nbytes() << ") is larger than the allowed size (" << output_buffer_size_ << ")");

        input_tensor_ = tensor;
        payload_size_ = input_tensor_->nbytes();
        CUDA_CHECK(cudaMemcpyAsync(output_buffer_.get(), input_tensor_->data(), payload_size_, cudaMemcpyDeviceToDevice, stream_));
    }

private:
    CudaStream stream_;
    size_t output_buffer_size_;
    CudaUniquePtr<char> output_buffer_;

    Tensor input_tensor_;
    size_t payload_size_ {};
    ibv::MemoryRegion memory_region_;
};

struct RoceTransmitterOp::Resource {
    Resource(const std::string& ibv_name, unsigned ibv_port, size_t buffer_size,
        size_t queue_size, const std::string& hololink_ip, uint32_t ibv_qp);

    using BufferQueue = std::queue<std::shared_ptr<RoceTransmitterOp::Buffer>>;

    ibv::Context context_;
    ::ibv_device_attr device_attr_;
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
    , port_attr_(context_.query_port(ibv_port))
    , gid_index_(context_.query_gid_roce_v2_ip())
    , local_gid_(context_.query_gid(ibv_port, gid_index_))
    ,
    // Allocate Protection Domain
    protection_domain_(context_.allocate_protection_domain())
    ,
    // Create Completion Queue
    completion_queue_(context_.create_completion_queue(queue_size))
    ,
    // Allocate and register the memory buffers that will hold the data
    buffer_size_(buffer_size)
    , available_buffers_([this, queue_size] {
        BufferQueue queue;
        // The queue_size is increased by one because while a buffer is transmitted, queue_size can wait in the queue
        // (The buffer that is currently transmitted is stored in the queue as well)
        for (size_t i = 0; i < queue_size; ++i)
            queue.emplace(std::make_shared<Buffer>(protection_domain_, buffer_size_));
        return queue;
    }())
    , queue_pair_(
          protection_domain_.create_queue_pair(
              ibv::QueuePair::InitAttr().query_pair_type(IBV_QPT_UC).send_completion_queue(completion_queue_.ptr_).receive_completion_queue(completion_queue_.ptr_).capacity(ibv::QueuePair::Capacity().max_send_write_read(512).max_receive_write_read(512).max_send_scatter_gather_entry(1).max_receive_scallter_gather_entry(1))))
    , remote_gid_(ibv::ipv4_to_gid(hololink_ip))
    , remote_qp_num_(ibv_qp)
{
    HSB_LOG_INFO("IB resource allocated successfully");
}

void RoceTransmitterOp::start()
{
    if (queue_size_ == 0)
        THROW_ERROR("queue_size must not be zero");

    // ibv calls seem to have trouble with
    // reentrancy.  No problem; since we're only
    // at startup, run just one at a time.  This
    // only affects multithreaded schedulers like
    // EventBasedScheduler.
    std::lock_guard lock(get_lock());

    // Create resource
    resource_ = std::make_unique<Resource>(ibv_name_.get(), ibv_port_.get(), buffer_size_.get(), queue_size_.get(), hololink_ip_.get(), ibv_qp_.get());

    // Connecting the QP by transitioning the state to Ready-to-Send.
    // We are doing one sided bring up for TX
    if (!connect())
        THROW_ERROR("Failed to connect QPs");

    if (on_start_.get())
        on_start_.get()(ConnectionInfo {
            resource_->queue_pair_.ptr_->qp_num });
    running_ = true;
    completion_thread_ = std::thread(std::bind(&RoceTransmitterOp::completion_thread, this));
    transmission_thread_ = std::thread(std::bind(&RoceTransmitterOp::transmission_thread, this));
}

void RoceTransmitterOp::stop()
{
    if (on_stop_.get())
        on_stop_.get()(ConnectionInfo { resource_->queue_pair_.ptr_->qp_num });
    running_ = false;
    cv_.notify_all();
    transmission_thread_.join();
    completion_thread_.join();
    resource_.reset();
}

void RoceTransmitterOp::setup(holoscan::OperatorSpec& spec)
{
    spec.input<Tensor>(input_name);

    /// Register converters for arguments not defined by Holoscan
    register_converter<RoceTransmitterOp::OnStartCallback>();
    register_converter<RoceTransmitterOp::OnStopCallback>();

    spec.param(ibv_name_, "ibv_name", "IBVName", "IBV device to use", std::string("roceP5p3s0f0"));
    spec.param(ibv_port_, "ibv_port", "IBVPort", "Port number of IBV device", 1u);
    spec.param(hololink_ip_, "hololink_ip", "HololinkIp", "IP address of Hololink board");
    spec.param(ibv_qp_, "ibv_qp", "IBVQp", "QP number for the IBV stream", 2u);
    spec.param(buffer_size_, "buffer_size", "BUFFER SIZE", "Max buffer size", static_cast<uint64_t>(1024 * 1024 * 1024));
    spec.param(queue_size_, "queue_size", "QUEUE SIZE", "The number of buffers that can wait to be transmitted", static_cast<uint64_t>(1));
    spec.param(on_start_, "on_start", "OnStart", "Callback function to be called when the connection is created", OnStartCallback());
    spec.param(on_stop_, "on_stop", "OnStop", "Callback function to be called when the connection is destroyed", OnStopCallback());
}

void RoceTransmitterOp::compute(holoscan::InputContext& op_input, [[maybe_unused]] holoscan::OutputContext& op_output,
    [[maybe_unused]] holoscan::ExecutionContext& context)
{
    auto maybe_tensor = op_input.receive<Tensor>(input_name.c_str());
    if (!maybe_tensor) {
        HSB_LOG_ERROR("Failed to receive message from port '{}'", input_name.c_str());
        throw std::exception();
    }
    auto tensor = maybe_tensor.value();
    queue_buffer(tensor);
}

bool RoceTransmitterOp::flush(std::chrono::milliseconds timeout)
{
    std::unique_lock<std::mutex> lock(mutex_);
    flushing_ = true;
    auto condition = [this] { return resource_->pending_buffers_.empty() && resource_->sent_buffers_.empty(); };
    if (timeout.count() > 0)
        cv_.wait_for(lock, timeout, condition);
    else
        cv_.wait(lock, condition);
    flushing_ = false;
    cv_.notify_all();
    return condition();
}

void RoceTransmitterOp::queue_buffer(const Tensor& tensor)
{
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return resource_->available_buffers_.size(); });
    // Pop out the first available buffer
    auto pending_buffer = resource_->available_buffers_.front();
    resource_->available_buffers_.pop();
    // Store the new Tensor buffer in it
    pending_buffer->Store(tensor);
    // Push in the buffer to the pending buffers queue
    resource_->pending_buffers_.push(std::move(pending_buffer));
    cv_.notify_all();
}

bool RoceTransmitterOp::connect()
{
    // Connecting the QP by transitioning the state to Ready-to-Send.
    // We are doing one sided bring up for TX
    return resource_->queue_pair_.reset_to_init(ibv_port_.get()) && resource_->queue_pair_.init_to_rtr(ibv_port_.get(), resource_->remote_gid_, resource_->remote_qp_num_, resource_->gid_index_) && resource_->queue_pair_.rtr_to_rts();
}

bool RoceTransmitterOp::send(uint64_t id, std::shared_ptr<Buffer> buffer)
try {
    ibv::ScatterGatherElement sge(buffer->get_scatter_gather_element());
    ibv::SendWorkRequest wr(id, &sge);

    if (!resource_->queue_pair_.post_send(wr)) {
        HSB_LOG_ERROR("Failed to post SEND Request");
        return false;
    }
    return true;
} catch (const std::exception&) {
    return false;
}

void RoceTransmitterOp::completion_thread()
{
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
            continue;
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
}

void RoceTransmitterOp::transmission_thread()
{
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

        if (!send(id, std::move(buffer)))
            HSB_LOG_ERROR("Failed to send buffer");
        HSB_LOG_DEBUG("Buffer sent");
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
