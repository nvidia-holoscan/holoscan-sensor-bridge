/**
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
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
#include <chrono>
#include <errno.h>
#include <poll.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <infiniband/opcode.h>

#include <hololink/common/cuda_helper.hpp>
#include <hololink/core/deserializer.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/core/logging_internal.hpp>
#include <hololink/core/networking.hpp>
#include <hololink/core/nvtx_trace.hpp>

#define NUM_OF(x) (sizeof(x) / sizeof(x[0]))

namespace hololink::operators {

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
    LinuxReceiverDescriptor(uint8_t* memory)
        : memory_(memory)
    {
    }

    uint8_t* memory_;
    LinuxReceiverMetadata metadata_;
};

LinuxReceiver::LinuxReceiver(CUdeviceptr cu_buffer,
    size_t cu_buffer_size,
    int socket,
    uint64_t received_address_offset)
    : cu_buffer_(cu_buffer)
    , cu_buffer_size_(cu_buffer_size)
    , socket_(socket)
    , received_address_offset_(received_address_offset)
    , ready_(false)
    , exit_(false)
    , ready_mutex_(PTHREAD_MUTEX_INITIALIZER)
    , ready_condition_(PTHREAD_COND_INITIALIZER)
    , qp_number_(0xCAFE)
    , rkey_(0xBEEF)
    , local_(NULL)
    , available_(NULL)
    , busy_(NULL)
    , cu_stream_(0)
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

    // See get_next_frame.
    CUresult cu_result = cuStreamCreate(&cu_stream_, CU_STREAM_NON_BLOCKING);
    if (cu_result != CUDA_SUCCESS) {
        throw std::runtime_error(fmt::format("cuStreamCreate failed, cu_result={}.", cu_result));
    }
}

LinuxReceiver::~LinuxReceiver()
{
    pthread_cond_destroy(&ready_condition_);
    pthread_mutex_destroy(&ready_mutex_);
    cuStreamDestroy(cu_stream_);
}

void LinuxReceiver::run()
{
    HSB_LOG_DEBUG("Starting.");
    core::NvtxTrace::setThreadName("linux_receiver");

    // Round the buffer size up to 64k
#define BUFFER_ALIGNMENT (0x10000)
    uint64_t buffer_size = (cu_buffer_size_ + BUFFER_ALIGNMENT - 1) & ~(BUFFER_ALIGNMENT - 1);
    // Allocate three pages, details below
    CUresult cu_result = cuMemHostAlloc((void**)(&local_), buffer_size * 3, CU_MEMHOSTALLOC_WRITECOMBINED);
    if (cu_result != CUDA_SUCCESS) {
        throw std::runtime_error(fmt::format("cuMemHostAlloc failed, cu_result={}.", cu_result));
    }
    // Construct a descriptor for each page
    LinuxReceiverDescriptor d0(&local_[buffer_size * 0]);
    LinuxReceiverDescriptor d1(&local_[buffer_size * 1]);
    LinuxReceiverDescriptor d2(&local_[buffer_size * 2]);
    // receiving points to the section we're currently receiving into
    // busy_ points to the buffer that the application is using.
    // available_ points to the last completed frame
    LinuxReceiverDescriptor* receiving = &d0;
    busy_ = &d1;
    available_.store(&d2);

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

        HSB_LOG_TRACE("received_bytes={} recv_errno={}.", received_bytes, recv_errno);

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
            core::NvtxTrace::event_u64("psn", psn);
            core::NvtxTrace::event_u64("frame_packets_received", frame_packets_received);
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
            if ((opcode == IBV_OPCODE_UC_RDMA_WRITE_ONLY)
                && deserializer.next_uint64_be(address)
                && deserializer.next_uint32_be(rkey)
                && deserializer.next_uint32_be(size)
                && deserializer.pointer(content, size)) {
                HSB_LOG_TRACE("opcode=2A address={:x} size={:x}", address, size);
                uint64_t target_address = address + received_address_offset_;
                if ((target_address >= cu_buffer_) && (target_address + size <= (cu_buffer_ + cu_buffer_size_))) {
                    uint64_t offset = target_address - cu_buffer_;
                    memcpy(&receiving->memory_[offset], content, size);
                    frame_bytes_received += size;
                }
                break;
            }

            uint32_t imm_data = 0;
            if ((opcode == IBV_OPCODE_UC_RDMA_WRITE_ONLY_WITH_IMMEDIATE)
                && deserializer.next_uint64_be(address)
                && deserializer.next_uint32_be(rkey)
                && deserializer.next_uint32_be(size)
                && deserializer.next_uint32_be(imm_data)
                && deserializer.pointer(content, size)) {
                frame_count++;
                core::NvtxTrace::event_u64("frame_count", frame_count);

                HSB_LOG_TRACE("opcode=2B address={:#x} size={:x}", address, size);
                uint64_t target_address = address + received_address_offset_;
                if ((target_address >= cu_buffer_) && (target_address + size <= (cu_buffer_ + cu_buffer_size_))) {
                    uint64_t offset = target_address - cu_buffer_;
                    memcpy(&receiving->memory_[offset], content, size);
                    frame_bytes_received += size;
                }
                // Send it
                // - receiving now has legit data;
                // - swap it with available_, now
                //  available_ points to received data
                //  and we'll continue to receive into what
                //  was in available_ (but not consumed by
                //  the application)
                // - signal the pipeline so it wakes up if necessary.
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

                receiving = available_.exchange(receiving);
                signal();
                // Make it easy to identify missing packets.
                memset(receiving->memory_, 0xFF, buffer_size);
                // Reset metadata.
                frame_packets_received = 0;
                frame_bytes_received = 0;
                break;
            }

            HSB_LOG_ERROR("Unable to decode IB request with opcode={:x}", opcode);
        } while (false);
    }

    busy_ = NULL;
    available_.store(NULL);

    cu_result = cuMemFreeHost((void*)(local_));
    if (cu_result != CUDA_SUCCESS) {
        HSB_LOG_ERROR("cuMemFreeHost failed, cu_result={}", cu_result);
        return;
    }
    local_ = NULL;
    HSB_LOG_DEBUG("Done.");
}

void LinuxReceiver::signal()
{
    int r = pthread_mutex_lock(&ready_mutex_);
    if (r != 0) {
        throw std::runtime_error(fmt::format("pthread_mutex_lock returned r={}.", r));
    }
    ready_ = true;
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
}

bool LinuxReceiver::get_next_frame(unsigned timeout_ms, LinuxReceiverMetadata& metadata)
{
    bool r = wait(timeout_ms);
    if (r) {
        busy_ = available_.exchange(busy_);
        if (busy_) {
            // Because we're setting up the next frame of data for
            // pipeline processing, we can allow this memcpy to overlap
            // with other GPU work-- we just make sure that this copy is done
            // (with the cuStreamSynchronize below) to ensure that this memcpy
            // finishes before the pipeline uses the destination buffer.
            CUresult cu_result = cuMemcpyHtoDAsync(cu_buffer_, busy_->memory_, cu_buffer_size_, cu_stream_);
            if (cu_result != CUDA_SUCCESS) {
                HSB_LOG_ERROR("cuMemcpyHtoDAsync failed, cu_result={}", cu_result);
                r = false;
            } else {
                cu_result = cuStreamSynchronize(cu_stream_);
                if (cu_result != CUDA_SUCCESS) {
                    HSB_LOG_ERROR("cuStreamSynchronize failed, cu_result={}", cu_result);
                    r = false;
                }
            }
            metadata = busy_->metadata_;
        } else {
            // run() exited.
            HSB_LOG_ERROR("get_next_frame failed, receiver has terminated.");
            r = false;
        }
    }
    return r;
}

bool LinuxReceiver::wait(unsigned timeout_ms)
{
    int status = pthread_mutex_lock(&ready_mutex_);
    if (status != 0) {
        throw std::runtime_error(fmt::format("pthread_mutex_lock returned status={}.", status));
    }
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        HSB_LOG_ERROR("clock_gettime failed, errno={}", errno);
    }
    struct timespec timeout = add_ms(now, timeout_ms);

    while (!ready_) {
        status = pthread_cond_timedwait(&ready_condition_, &ready_mutex_, &timeout);
        if (status == ETIMEDOUT) {
            break;
        }
        if (status != 0) {
            HSB_LOG_ERROR("pthread_cond_wait returned status={}", status);
            break;
        }
    }
    bool r = ready_;
    ready_ = false;
    status = pthread_mutex_unlock(&ready_mutex_);
    if (status != 0) {
        throw std::runtime_error(fmt::format("pthread_mutex_unlock returned status={}.", status));
    }
    return r;
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
