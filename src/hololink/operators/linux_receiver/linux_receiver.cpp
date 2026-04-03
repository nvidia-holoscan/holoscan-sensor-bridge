/**
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES.
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
 *
 * See README.md for detailed information.
 */

#include "linux_receiver.hpp"

#include <alloca.h>
#include <atomic>
#include <errno.h>
#include <poll.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <hololink/common/cuda_helper.hpp>
#include <hololink/core/deserializer.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/core/logging_internal.hpp>
#include <hololink/core/networking.hpp>
#include <hololink/core/nvtx_trace.hpp>

#if defined(HOLOLINK_HAVE_IBV_OPCODE)
#include <infiniband/opcode.h>
#endif

#define OPCODE_UC_RDMA_WRITE_ONLY (0x2A)
#define OPCODE_UC_RDMA_WRITE_ONLY_WITH_IMMEDIATE (0x2B)

#if defined(HOLOLINK_HAVE_IBV_OPCODE)
static_assert(OPCODE_UC_RDMA_WRITE_ONLY == IBV_OPCODE_UC_RDMA_WRITE_ONLY,
    "OPCODE_UC_RDMA_WRITE_ONLY must match libibverbs definition");
static_assert(OPCODE_UC_RDMA_WRITE_ONLY_WITH_IMMEDIATE == IBV_OPCODE_UC_RDMA_WRITE_ONLY_WITH_IMMEDIATE,
    "OPCODE_UC_RDMA_WRITE_ONLY_WITH_IMMEDIATE must match libibverbs definition");
#endif

#define NUM_OF(x) (sizeof(x) / sizeof(x[0]))

namespace hololink::operators {

static std::atomic<uint32_t> next_qp_number_(0xCAF0);

static inline struct timespec add_ms(struct timespec& ts, unsigned long ms)
{
    unsigned long us = ms * 1000;
    unsigned long ns = us * 1000;
    ns += ts.tv_nsec;
#ifndef NS_PER_S
#define MS_PER_S (1000ULL)
#define US_PER_S (1000ULL * MS_PER_S)
#define NS_PER_S (1000ULL * US_PER_S)
#endif /* NS_PER_S */
    lldiv_t d = lldiv(ns, NS_PER_S);
    struct timespec r;
    r.tv_nsec = d.rem;
    r.tv_sec = ts.tv_sec + d.quot;
    return r;
}

class LinuxReceiverDescriptor {
public:
    LinuxReceiverDescriptor(uint8_t* memory, CUevent event)
        : memory_(memory)
        , event_(event)
    {
    }

    uint8_t* memory_;
    CUevent event_;
    LinuxReceiverMetadata metadata_;
};

LinuxReceiver::LinuxReceiver(CUdeviceptr cu_buffer,
    size_t cu_buffer_size,
    size_t cu_page_size,
    unsigned pages,
    int socket,
    uint64_t received_address_offset,
    unsigned queue_size)
    : cu_buffer_(cu_buffer)
    , cu_buffer_size_(cu_buffer_size)
    , cu_page_size_(cu_page_size)
    , pages_(pages)
    , socket_(socket)
    , received_address_offset_(received_address_offset)
    , queue_size_(queue_size)
    , exit_(false)
    , ready_mutex_(PTHREAD_MUTEX_INITIALIZER)
    , ready_condition_(PTHREAD_COND_INITIALIZER)
    , consumer_ready_(false)
    , qp_number_(next_qp_number_++)
    , rkey_(0xBEEF)
    , local_(NULL)
    , frame_ready_([](const LinuxReceiver&) {})
    , frame_number_()
{
    int r = pthread_mutex_init(&ready_mutex_, NULL);
    if (r != 0) {
        throw std::runtime_error("pthread_mutex_init failed.");
    }
    pthread_condattr_t pthread_condattr;
    pthread_condattr_init(&pthread_condattr);
    pthread_condattr_setclock(&pthread_condattr, CLOCK_MONOTONIC);
    r = pthread_cond_init(&ready_condition_, &pthread_condattr);
    if (r != 0) {
        throw std::runtime_error("pthread_cond_init failed.");
    }

    // set the receive timeout, we use that to check to periodically return from the
    // recv() call to check if the receiver thread should exit
    struct timeval timeout;
    timeout.tv_sec = 0;
    timeout.tv_usec = 100000;
    if (setsockopt(socket_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) < 0) {
        throw std::runtime_error(fmt::format("setsockopt failed errno={}", errno));
    }
}

LinuxReceiver::~LinuxReceiver()
{
    pthread_cond_destroy(&ready_condition_);
    pthread_mutex_destroy(&ready_mutex_);
}

void LinuxReceiver::run()
{
    HSB_LOG_DEBUG("Starting, qp={:#x}.", qp_number_);
    core::NvtxTrace::setThreadName("linux_receiver");

    // Round the buffer size up to 64k
#define BUFFER_ALIGNMENT (0x10000)
    uint64_t buffer_size = (cu_page_size_ + BUFFER_ALIGNMENT - 1) & ~(BUFFER_ALIGNMENT - 1);
    // +1 for the buffer we're currently receiving into
    const uint32_t buffer_count = queue_size_ + 1;
    // Allocate pages, details below
    CudaCheck(cuMemHostAlloc((void**)(&local_), buffer_size * buffer_count, CU_MEMHOSTALLOC_WRITECOMBINED));
    CUevent events[buffer_count];
    for (int i = 0; i < buffer_count; i++) {
        CudaCheck(cuEventCreate(&events[i], CU_EVENT_DISABLE_TIMING));
    }
    // Construct a descriptor for each page
    std::unique_ptr<LinuxReceiverDescriptor> descriptors[buffer_count];
    for (int i = 0; i < buffer_count; i++) {
        descriptors[i] = std::make_unique<LinuxReceiverDescriptor>(&local_[buffer_size * i], events[i]);
    }
    // receiving points to the buffer we're currently receiving into.
    // ready_ has the the buffers for the application to process.
    // available_ has the buffers which are not used.
    LinuxReceiverDescriptor* receiving = descriptors[0].get();
    for (int i = 1; i < buffer_count; i++) {
        available_.push(descriptors[i].get());
    }

    // Received UDP message goes here.
    uint8_t received[hololink::core::UDP_PACKET_SIZE];

    unsigned frame_count = 0;
    [[maybe_unused]] unsigned packet_count = 0;
    unsigned frame_packets_received = 0, frame_bytes_received = 0;
    struct timespec now = { 0 }, frame_start = { 0 };
    uint64_t packets_dropped = 0;
    uint32_t last_psn = 0;
    bool first = true;

    while (true) {
        int recv_flags = 0;
        ssize_t received_bytes = recv(socket_, received, sizeof(received), recv_flags);
        int recv_errno = errno;

        // Get the clock as close to the packet receipt as possible.
        if (clock_gettime(CLOCK_REALTIME, &now) != 0) {
            HSB_LOG_ERROR("clock_gettime failed, errno={}", errno);
            break;
        }

        if (received_bytes <= 0) {
            // check if there is a timeout
            if ((recv_errno == EAGAIN) || (recv_errno == EWOULDBLOCK) || (recv_errno == EINTR)) {
                // should we exit?
                if (exit_) {
                    break;
                }
                // if not, continue
                continue;
            }
            HSB_LOG_ERROR("recv returned received_bytes={}, recv_errno={}", received_bytes, recv_errno);
            break;
        }

        packet_count++;
        frame_packets_received++;
        if (!frame_bytes_received) {
            frame_start = now;
        }

        do {
            core::Deserializer deserializer(received, received_bytes);
            uint8_t opcode = 0, flags = 0;
            uint16_t pkey = 0;
            uint8_t becn = 0, ack_request = 0;
            uint32_t qp = 0, psn = 0;
            if (!(deserializer.next_uint8(opcode)
                    && deserializer.next_uint8(flags)
                    && deserializer.next_uint16_be(pkey)
                    && deserializer.next_uint8(becn)
                    && deserializer.next_uint24_be(qp)
                    && deserializer.next_uint8(ack_request)
                    && deserializer.next_uint24_be(psn))) {
                HSB_LOG_ERROR("Unable to decode runt IB request, received_bytes={}", received_bytes);
                break;
            }

            // Note that 'psn' is only 24 bits.  Use that to determine
            // how many packets were dropped.  Note that this doesn't
            // account for out-of-order delivery.
            if (!first) {
                uint32_t next_psn = (last_psn + 1) & 0xFFFFFF;
                uint32_t diff = (psn - next_psn) & 0xFFFFFF;
                packets_dropped += diff;
            }
            last_psn = psn;
            first = false;

            uint64_t address = 0;
            uint32_t rkey = 0;
            uint32_t size = 0;
            const uint8_t* content = NULL;
            if ((opcode == OPCODE_UC_RDMA_WRITE_ONLY)
                && deserializer.next_uint64_be(address)
                && deserializer.next_uint32_be(rkey)
                && deserializer.next_uint32_be(size)
                && deserializer.pointer(content, size)) {
                uint64_t target_address = address + received_address_offset_;
                if ((target_address >= cu_buffer_) && (target_address + size <= (cu_buffer_ + cu_buffer_size_))) {
                    uint64_t offset = (target_address - cu_buffer_) % cu_page_size_;
                    if (offset + size > cu_page_size_) {
                        HSB_LOG_ERROR("Packet spans page boundary; address={:`#x`}, size={:`#x`}, offset={:`#x`}, cu_page_size_={:`#x`}",
                            target_address, size, offset, cu_page_size_);
                        break;
                    }
                    memcpy(&receiving->memory_[offset], content, size);
                    frame_bytes_received += size;
                } else {
                    HSB_LOG_ERROR("Ignoring contents for a packet with address={:#x} and size={:#x}, cu_buffer_={:#x}, cu_buffer_size_={:#x}.", target_address, size, cu_buffer_, cu_buffer_size_);
                }
                break;
            }

            uint32_t imm_data = 0;
            if ((opcode == OPCODE_UC_RDMA_WRITE_ONLY_WITH_IMMEDIATE)
                && deserializer.next_uint64_be(address)
                && deserializer.next_uint32_be(rkey)
                && deserializer.next_uint32_be(size)
                && deserializer.next_uint32_be(imm_data)
                && deserializer.pointer(content, size)) {
                frame_count++;
                core::NvtxTrace::event_u64("frame_count", frame_count);

                if (size != METADATA_SIZE) {
                    HSB_LOG_ERROR("Unexpected size for metadata, size={}, expected={}", size, METADATA_SIZE);
                    break;
                }
                Hololink::FrameMetadata frame_metadata = Hololink::deserialize_metadata(content, size);
                LinuxReceiverMetadata& metadata = receiving->metadata_;
                metadata.frame_packets_received = frame_packets_received;
                metadata.frame_bytes_received = frame_bytes_received;
                metadata.received_frame_number = frame_count;
                metadata.frame_start_s = frame_start.tv_sec;
                metadata.frame_start_ns = frame_start.tv_nsec;
                metadata.frame_end_s = now.tv_sec;
                metadata.frame_end_ns = now.tv_nsec;
                metadata.imm_data = imm_data;
                metadata.packets_dropped = packets_dropped;
                metadata.received_s = now.tv_sec;
                metadata.received_ns = now.tv_nsec;
                metadata.frame_metadata = frame_metadata;
                metadata.frame_number = frame_number_.update(frame_metadata.frame_number);

                unsigned page = imm_data & 0xFFF;
                if (page >= pages_) {
                    throw std::runtime_error(fmt::format("Invalid page={}; ignoring.", page));
                }

                // Send it
                // - signal the pipeline so it wakes up if necessary.
                receiving = signal(receiving);
                // Make it easy to identify missing packets.
                memset(receiving->memory_, 0xFF, cu_page_size_);
                // Reset metadata.
                frame_packets_received = 0;
                frame_bytes_received = 0;
                break;
            }

            HSB_LOG_ERROR("Unable to decode IB request with opcode={:x}", opcode);
        } while (false);
    }

    while (!ready_.empty()) {
        ready_.pop();
    }
    while (!available_.empty()) {
        available_.pop();
    }

    for (int i = 0; i < buffer_count; i++) {
        CudaCheck(cuEventDestroy(events[i]));
    }
    CudaCheck(cuMemFreeHost((void*)(local_)));
    local_ = NULL;
    HSB_LOG_DEBUG("Done.");
}

LinuxReceiverDescriptor* LinuxReceiver::signal(LinuxReceiverDescriptor* received)
{
    int r = pthread_mutex_lock(&ready_mutex_);
    if (r != 0) {
        throw std::runtime_error(fmt::format("pthread_mutex_lock returned r={}.", r));
    }
    LinuxReceiverDescriptor* available;
    if (!consumer_ready_ && !ready_.empty()) {
        // if the consumer is not ready yet, avoid queueing up more frames
        available = ready_.front();
        ready_.pop();
    } else {
        if (available_.empty()) {
            // if there is no available descriptor use the oldest ready one and drop the frame
            available = ready_.front();
            ready_.pop();
            if (consumer_ready_) {
                HSB_LOG_DEBUG("No available descriptors, dropping oldest ready frame {}.", available->metadata_.frame_number);
            }
        } else {
            available = available_.front();
            available_.pop();
        }
    }
    // make sure the copy is complete before we overwrite the buffer
    CudaCheck(cuEventSynchronize(available->event_));

    // place in the ready queue for the application to process
    ready_.push(received);

    r = pthread_cond_signal(&ready_condition_);
    if (r != 0) {
        throw std::runtime_error(fmt::format("pthread_cond_signal returned r={}.", r));
    }
    r = pthread_mutex_unlock(&ready_mutex_);
    if (r != 0) {
        throw std::runtime_error(fmt::format("pthread_mutex_unlock returned r={}.", r));
    }

    // Provide the local callback, letting the application know
    // that get_next_frame won't block.
    frame_ready_(this[0]);

    return available;
}

bool LinuxReceiver::get_next_frame(unsigned timeout_ms, LinuxReceiverMetadata& metadata, CUstream cuda_stream)
{
    int status = pthread_mutex_lock(&ready_mutex_);
    if (status != 0) {
        throw std::runtime_error(fmt::format("pthread_mutex_lock returned status={}.", status));
    }

    // signal the background thread that the consumer is ready to receive frames
    consumer_ready_ = true;

    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        HSB_LOG_ERROR("clock_gettime failed, errno={}", errno);
    }
    struct timespec timeout = add_ms(now, timeout_ms);

    bool result = true;
    while (ready_.empty()) {
        status = pthread_cond_timedwait(&ready_condition_, &ready_mutex_, &timeout);
        if (status == ETIMEDOUT) {
            result = false;
            break;
        }
        if (status != 0) {
            HSB_LOG_ERROR("pthread_cond_wait returned status={}", status);
            result = false;
            break;
        }
    }

    if (result) {
        LinuxReceiverDescriptor* ready_descriptor = ready_.front();
        ready_.pop();

        metadata = ready_descriptor->metadata_;
        metadata.frame_memory = cu_buffer_;

        // Because we're setting up the next frame of data for pipeline processing,
        // we can allow this memcpy to overlap with other GPU work. We just make sure
        // that this copy is happening on the same stream as the pipeline.
        CudaCheck(cuMemcpyHtoDAsync(cu_buffer_, ready_descriptor->memory_, metadata.frame_metadata.bytes_written, cuda_stream));
        CudaCheck(cuEventRecord(ready_descriptor->event_, cuda_stream));

        available_.push(ready_descriptor);
    }

    status = pthread_mutex_unlock(&ready_mutex_);
    if (status != 0) {
        throw std::runtime_error(fmt::format("pthread_mutex_unlock returned status={}.", status));
    }
    return result;
}

bool LinuxReceiver::frames_ready()
{
    int status = pthread_mutex_lock(&ready_mutex_);
    if (status != 0) {
        throw std::runtime_error(fmt::format("pthread_mutex_lock returned status={}.", status));
    }

    bool result = ready_.empty();

    status = pthread_mutex_unlock(&ready_mutex_);
    if (status != 0) {
        throw std::runtime_error(fmt::format("pthread_mutex_unlock returned status={}.", status));
    }
    return !result;
}

void LinuxReceiver::close()
{
    exit_ = true;
}

void LinuxReceiver::set_frame_ready(std::function<void(const LinuxReceiver&)> frame_ready)
{
    frame_ready_ = frame_ready;
}

} // namespace hololink::operators
