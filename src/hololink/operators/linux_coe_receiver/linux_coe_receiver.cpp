/**
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

#include "linux_coe_receiver.hpp"

#include <errno.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <hololink/common/cuda_helper.hpp>
#include <hololink/core/deserializer.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/core/logging_internal.hpp>
#include <hololink/core/networking.hpp>
#include <hololink/core/nvtx_trace.hpp>

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

class LinuxCoeReceiverDescriptor {
public:
    LinuxCoeReceiverDescriptor(uint8_t* memory)
        : memory_(memory)
    {
    }

    uint8_t* memory_;
    LinuxCoeReceiverMetadata metadata_;
};

LinuxCoeReceiver::LinuxCoeReceiver(CUdeviceptr cu_buffer,
    size_t cu_buffer_size,
    int socket,
    uint16_t channel)
    : cu_buffer_(cu_buffer)
    , cu_buffer_size_(cu_buffer_size)
    , socket_(socket)
    , channel_(channel)
    , ready_(false)
    , exit_(false)
    , ready_mutex_(PTHREAD_MUTEX_INITIALIZER)
    , ready_condition_(PTHREAD_COND_INITIALIZER)
    , local_(NULL)
    , available_(NULL)
    , busy_(NULL)
    , cu_stream_(0)
    , frame_ready_([](const LinuxCoeReceiver&) {})
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

LinuxCoeReceiver::~LinuxCoeReceiver()
{
    pthread_cond_destroy(&ready_condition_);
    pthread_mutex_destroy(&ready_mutex_);
    cuStreamDestroy(cu_stream_);
}

void LinuxCoeReceiver::run()
{
    HSB_LOG_DEBUG("Starting.");
    core::NvtxTrace::setThreadName("linux_coe_receiver");

    // Round the buffer size up so that the next buffer starts on a 128-byte boundary.
    size_t buffer_size = hololink::core::round_up(cu_buffer_size_, 128);
    // Allocate three pages, details below
    CUresult cu_result = cuMemHostAlloc((void**)(&local_), buffer_size * 3, CU_MEMHOSTALLOC_WRITECOMBINED);
    if (cu_result != CUDA_SUCCESS) {
        throw std::runtime_error(fmt::format("cuMemHostAlloc failed, cu_result={}.", cu_result));
    }
    // Construct a descriptor for each page
    LinuxCoeReceiverDescriptor d0(&local_[buffer_size * 0]);
    LinuxCoeReceiverDescriptor d1(&local_[buffer_size * 1]);
    LinuxCoeReceiverDescriptor d2(&local_[buffer_size * 2]);
    // receiving points to the section we're currently receiving into
    // busy_ points to the buffer that the application is using.
    // available_ points to the last completed frame
    LinuxCoeReceiverDescriptor* receiving = &d0;
    busy_ = &d1;
    available_.store(&d2);

    // Received L2 network message goes here.
    constexpr uint32_t ETHERNET_PACKET_SIZE = 10240;
    uint8_t received[ETHERNET_PACKET_SIZE];

    unsigned frame_count = 0;
    [[maybe_unused]] unsigned packet_count = 0;
    unsigned frame_packets_received = 0, frame_bytes_received = 0;
    struct timespec now = { 0 }, frame_start = { 0 };

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
                // Go try again
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
            std::vector<uint8_t> destination_mac(6);
            std::vector<uint8_t> source_mac(6);
            uint16_t ethertype = 0;
            constexpr uint16_t AVTP_ETHERTYPE = 0x22F0;
            uint8_t subtype = 0;
            constexpr uint8_t NTSCF = 0x82;
            uint16_t ntscf_data_length = 0;
            uint8_t sequence = 0;
            std::vector<uint8_t> stream_id(8);
            uint8_t acf_message_type = 0;
            constexpr uint8_t ACF_MESSAGE_TYPE = 0x0C; // No idea what this should be called.
            uint8_t acf_message_length = 0;
            uint8_t reserved_1 = 0, reserved_2 = 0;
            uint32_t timestamp_ns = 0, timestamp_sec = 0;
            uint8_t sequence_number_c = 0;
            uint16_t channel = 0;
            uint8_t flags = 0;
            [[maybe_unused]] constexpr uint32_t FRAME_START = 0x01;
            [[maybe_unused]] constexpr uint32_t FRAME_END = 0x02;
            [[maybe_unused]] constexpr uint32_t LINE_END = 0x04;
            uint32_t address = 0;
            const uint8_t* payload = 0;

            // The kernel feeds us everything received here, so don't
            // pay any mind to packets that aren't specifically a part
            // of our stack.
            if (!(deserializer.next_buffer(destination_mac)
                    && deserializer.next_buffer(source_mac)
                    && deserializer.next_uint16_be(ethertype)
                    && (ethertype == AVTP_ETHERTYPE)
                    && deserializer.next_uint8(subtype)
                    && (subtype == NTSCF)
                    && deserializer.next_uint16_be(ntscf_data_length)
                    && deserializer.next_uint8(sequence)
                    && deserializer.next_buffer(stream_id)
                    && deserializer.next_uint8(acf_message_type)
                    && (acf_message_type == ACF_MESSAGE_TYPE)
                    && deserializer.next_uint8(acf_message_length)
                    && deserializer.next_uint8(reserved_1)
                    && deserializer.next_uint8(reserved_2)
                    && deserializer.next_uint32_be(timestamp_sec)
                    && deserializer.next_uint32_be(timestamp_ns)
                    && deserializer.next_uint8(sequence_number_c)
                    && deserializer.next_uint16_be(channel) // note that we check this below
                    && deserializer.next_uint8(flags)
                    && deserializer.next_uint32_be(address)
                    && deserializer.pointer(payload, 0)
                    && (deserializer.position() <= received_bytes))) {
                // Ignore this guy.
                break;
            }

            // NOTE that ntscv_data_length is really
            // (sv, version (3 bits), r, and length (11 bits))
            // per 1722-2016.pdf page 78.
            [[maybe_unused]] uint8_t sv = (ntscf_data_length & 0x8000) >> 15;
            [[maybe_unused]] uint8_t version = (ntscf_data_length >> 12) & 0x7;
            [[maybe_unused]] uint8_t r = (ntscf_data_length & 0x800) >> 11;
            ntscf_data_length &= 0x7FF;
            // NOTE that channel is really
            // (e, se, fcv, ver (2 bits), exposure (2 bits), reserved (3 bits), channel number (6 bits)
            [[maybe_unused]] uint8_t e = (channel & 0x8000) >> 15; // Not clear how this is used
            [[maybe_unused]] uint8_t se = (channel & 0x4000) >> 14; // Not clear how this is used
            [[maybe_unused]] uint8_t fcv = (channel & 0x2000) >> 13; // Not clear how this is used
            [[maybe_unused]] uint8_t acf_version = (channel & 0x1800) >> 11; // Not clear how this is used
            [[maybe_unused]] uint8_t exposure = (channel & 0x600) >> 9; // Not clear how this is used
            channel &= 0x3F;
            // Skip traffic for other channels.
            if (channel != channel_) {
                continue;
            }
            // This has to be >= 0 due to the test above.
            uint32_t payload_bytes = received_bytes - deserializer.position();
            // NOTE that address is really
            // ( frame number (4 bits), byte offset (28 bits) )
            [[maybe_unused]] uint8_t frame_number = (address & 0xF000'0000) >> 28;
            address &= 0xFFF'FFFF;

            core::NvtxTrace::event_u64(fmt::format("address={:#x} flags={:#x} sequence_number_c={} payload_bytes={}",
                                           address, flags, sequence_number_c, payload_bytes)
                                           .c_str(),
                0);

            // Cache the payload data.
            if ((address + payload_bytes) <= (cu_buffer_size_)) {
                HSB_LOG_TRACE("address={:#x} payload_bytes={:#x} flags={:#x}", address, payload_bytes, flags);
                memcpy(&receiving->memory_[address], payload, payload_bytes);
                frame_bytes_received += payload_bytes;
            } else {
                HSB_LOG_ERROR("Ignoring contents for a packet with address={:#x} and payload_bytes={:#x}, cu_buffer_size={:#x}, flags={:#x}.", address, payload_bytes, cu_buffer_size_, flags);
            }

            if (flags & FRAME_END) {
                frame_count++;
                core::NvtxTrace::event_u64("FRAME_END", frame_count);

                // Send it
                // - receiving now has legit data;
                // - swap it with available_, now
                //  available_ points to received data
                //  and we'll continue to receive into what
                //  was in available_ (but not consumed by
                //  the application)
                // - signal the pipeline so it wakes up if necessary.
                Hololink::FrameMetadata frame_metadata = Hololink::deserialize_metadata(payload, payload_bytes);
                LinuxCoeReceiverMetadata& metadata = receiving->metadata_;
                metadata.frame_packets_received = frame_packets_received;
                metadata.frame_bytes_received = frame_bytes_received;
                metadata.received_frame_number = frame_count;
                metadata.frame_start_s = frame_start.tv_sec;
                metadata.frame_start_ns = frame_start.tv_nsec;
                metadata.frame_end_s = now.tv_sec;
                metadata.frame_end_ns = now.tv_nsec;
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
            }
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

void LinuxCoeReceiver::signal()
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

bool LinuxCoeReceiver::get_next_frame(unsigned timeout_ms, LinuxCoeReceiverMetadata& metadata)
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
            // Without this, it'd use the device stream instance which would
            // wait until the device was completely idle.
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

bool LinuxCoeReceiver::wait(unsigned timeout_ms)
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

void LinuxCoeReceiver::close()
{
    exit_ = true;
}

void LinuxCoeReceiver::set_frame_ready(std::function<void(const LinuxCoeReceiver&)> frame_ready)
{
    frame_ready_ = frame_ready;
}

} // namespace hololink::operators
