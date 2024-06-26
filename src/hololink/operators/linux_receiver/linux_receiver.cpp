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
#include <errno.h>
#include <poll.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <infiniband/opcode.h>

#include <hololink/native/deserializer.hpp>
#include <hololink/native/nvtx_trace.hpp>

#define TRACE(fmt, ...) /* ignored */
#define DEBUG(fmt...) fprintf(stderr, "DEBUG -- " fmt)
#define ERROR(fmt...) fprintf(stderr, "ERROR -- " fmt)

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
    int socket)
    : cu_buffer_(cu_buffer)
    , cu_buffer_size_(cu_buffer_size)
    , socket_(socket)
    , ready_(false)
    , exit_(false)
    , ready_mutex_(PTHREAD_MUTEX_INITIALIZER)
    , ready_condition_(PTHREAD_COND_INITIALIZER)
    , qp_number_(0xCAFE)
    , rkey_(0xBEEF)
    , local_(NULL)
    , available_(NULL)
    , busy_(NULL)
{
    int r = pthread_mutex_init(&ready_mutex_, NULL);
    if (r != 0) {
        ERROR("pthread_mutex_init failed, r=%d.\n", r);
    }
    pthread_condattr_t pthread_condattr;
    pthread_condattr_init(&pthread_condattr);
    pthread_condattr_setclock(&pthread_condattr, CLOCK_MONOTONIC);
    r = pthread_cond_init(&ready_condition_, &pthread_condattr);
    if (r != 0) {
        ERROR("pthread_cond_init failed, r=%d.\n", r);
    }

    // set the receive timeout, we use that to check to periodically return from the
    // recv() call to check if the receiver thread should exit
    struct timeval timeout;
    timeout.tv_sec = 0;
    timeout.tv_usec = 100000;
    if (setsockopt(socket_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) < 0) {
        ERROR("setsockopt failed errno=%d.\n", (int)errno);
    }
}

LinuxReceiver::~LinuxReceiver()
{
    pthread_cond_destroy(&ready_condition_);
    pthread_mutex_destroy(&ready_mutex_);
}

void LinuxReceiver::run()
{
    DEBUG("Starting.\n");
    native::NvtxTrace::setThreadName("linux_receiver");

    // Round the buffer size up to 64k
#define BUFFER_ALIGNMENT (0x10000)
    uint64_t buffer_size = (cu_buffer_size_ + BUFFER_ALIGNMENT - 1) & ~(BUFFER_ALIGNMENT - 1);
    // Allocate three pages, details below
    CUresult cu_result = cuMemHostAlloc((void**)(&local_), buffer_size * 3, CU_MEMHOSTALLOC_WRITECOMBINED);
    if (cu_result != CUDA_SUCCESS) {
        ERROR("cuMemHostAlloc failed, cu_result=%d.\n", (int)cu_result);
        return;
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
    uint8_t received[8192];

    unsigned frame_count = 0, packet_count = 0;
    unsigned frame_packets_received = 0, frame_bytes_received = 0;
    struct timespec frame_start = { 0 }, frame_end = { 0 };

    while (true) {
        int recv_flags = 0;
        ssize_t received_bytes = recv(socket_, received, sizeof(received), recv_flags);
        if (received_bytes <= 0) {
            // check if there is a timeout
            if ((errno == EAGAIN) || (errno == EWOULDBLOCK)) {
                // should we exit?
                if (exit_) {
                    break;
                }
                // if not, continue
                continue;
            }
            ERROR("recv returned received_bytes=%d, errno=%d.\n", (int)received_bytes, (int)errno);
            break;
        }

        packet_count++;
        frame_packets_received++;
        if (!frame_bytes_received) {
            if (clock_gettime(CLOCK_MONOTONIC, &frame_start) != 0) {
                ERROR("clock_gettime failed, errno=%d.\n", (int)errno);
                break;
            }
        }

        do {
            native::Deserializer deserializer(received, received_bytes);
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
                ERROR("Unable to decode runt IB request, received_bytes=%d.\n", (int)received_bytes);
                break;
            }

            uint64_t address = 0;
            uint32_t rkey = 0;
            uint32_t size = 0;
            uint8_t* content = NULL;
            if ((opcode == IBV_OPCODE_UC_RDMA_WRITE_ONLY)
                && deserializer.next_uint64_be(address)
                && deserializer.next_uint32_be(rkey)
                && deserializer.next_uint32_be(size)
                && deserializer.pointer(content, size)) {
                TRACE("opcode=2A address=0x%llX size=0x%X\n", (unsigned long long)address, (unsigned)size);
                if ((address >= cu_buffer_) && (address + size <= (cu_buffer_ + cu_buffer_size_))) {
                    uint64_t offset = address - cu_buffer_;
                    memcpy(&receiving->memory_[offset], content, size);
                    frame_bytes_received += size;
                }
                break;
            }

            uint32_t imm = 0;
            if ((opcode == IBV_OPCODE_UC_RDMA_WRITE_ONLY_WITH_IMMEDIATE)
                && deserializer.next_uint64_be(address)
                && deserializer.next_uint32_be(rkey)
                && deserializer.next_uint32_be(size)
                && deserializer.next_uint32_be(imm)
                && deserializer.pointer(content, size)) {
                frame_count++;
                TRACE("opcode=2B address=0x%llX size=0x%X\n", (unsigned long long)address, (unsigned)size);
                if ((address >= cu_buffer_) && (address + size <= (cu_buffer_ + cu_buffer_size_))) {
                    uint64_t offset = address - cu_buffer_;
                    memcpy(&receiving->memory_[offset], content, size);
                    frame_bytes_received += size;
                }
                struct timespec frame_end;
                if (clock_gettime(CLOCK_MONOTONIC, &frame_end) != 0) {
                    ERROR("clock_gettime failed, errno=%d.\n", (int)errno);
                    break;
                }
                // Send it
                // - receiving now has legit data;
                // - swap it with available_, now
                //  available_ points to received data
                //  and we'll continue to receive into what
                //  was in available_ (but not consumed by
                //  the application)
                // - signal the pipeline so it wakes up if necessary.
                LinuxReceiverMetadata& metadata = receiving->metadata_;
                metadata.frame_packets_received = frame_packets_received;
                metadata.frame_bytes_received = frame_bytes_received;
                metadata.frame_number = frame_count;
                metadata.frame_start_s = frame_start.tv_sec;
                metadata.frame_start_ns = frame_start.tv_nsec;
                metadata.frame_end_s = frame_end.tv_sec;
                metadata.frame_end_ns = frame_end.tv_nsec;

                receiving = available_.exchange(receiving);
                signal();
                // Make it easy to identify missing packets.
                memset(receiving->memory_, 0xFF, buffer_size);
                // Reset metadata.
                frame_packets_received = 0;
                frame_bytes_received = 0;
                break;
            }

            ERROR("Unable to decode IB request with opcode=0x%X.\n", (unsigned)opcode);
        } while (false);
    }

    busy_ = NULL;
    available_.store(NULL);

    cu_result = cuMemFreeHost((void*)(local_));
    if (cu_result != CUDA_SUCCESS) {
        ERROR("cuMemFreeHost failed, cu_result=%d.\n", (int)cu_result);
        return;
    }
    local_ = NULL;
    DEBUG("Done.\n");
}

void LinuxReceiver::signal()
{
    int r = pthread_mutex_lock(&ready_mutex_);
    if (r != 0) {
        ERROR("pthread_mutex_lock returned r=%d.\n", r);
    }
    ready_ = true;
    r = pthread_cond_signal(&ready_condition_);
    if (r != 0) {
        ERROR("pthread_cond_signal returned r=%d.\n", r);
    }
    r = pthread_mutex_unlock(&ready_mutex_);
    if (r != 0) {
        ERROR("pthread_mutex_unlock returned r=%d.\n", r);
    }
}

bool LinuxReceiver::get_next_frame(unsigned timeout_ms, LinuxReceiverMetadata& metadata)
{
    bool r = wait(timeout_ms);
    if (r) {
        busy_ = available_.exchange(busy_);
        if (busy_) {
            CUstream stream = CU_STREAM_PER_THREAD;
            CUresult cu_result = cuMemcpyHtoDAsync(cu_buffer_, busy_->memory_, cu_buffer_size_, stream);
            if (cu_result != CUDA_SUCCESS) {
                ERROR("cuMemcpyHtoD failed, cu_result=%d.\n", (int)cu_result);
                r = false;
            }
            metadata = busy_->metadata_;
        } else {
            // run() exited.
            ERROR("get_next_frame failed, recevier has terminated.\n");
            r = false;
        }
    }
    return r;
}

bool LinuxReceiver::wait(unsigned timeout_ms)
{
    int status = pthread_mutex_lock(&ready_mutex_);
    if (status != 0) {
        ERROR("pthread_mutex_lock returned status=%d.\n", status);
        return false;
    }
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        ERROR("clock_gettime failed, errno=%d.\n", (int)errno);
    }
    struct timespec timeout = add_ms(now, timeout_ms);

    while (!ready_) {
        status = pthread_cond_timedwait(&ready_condition_, &ready_mutex_, &timeout);
        if (status == ETIMEDOUT) {
            break;
        }
        if (status != 0) {
            ERROR("pthread_cond_wait returned status=%d.\n", status);
            break;
        }
    }
    bool r = ready_;
    ready_ = false;
    status = pthread_mutex_unlock(&ready_mutex_);
    if (status != 0) {
        ERROR("pthread_mutex_unlock returned status=%d.\n", status);
    }
    return r;
}

void LinuxReceiver::close()
{
    exit_ = true;
}

} // namespace hololink::operators
