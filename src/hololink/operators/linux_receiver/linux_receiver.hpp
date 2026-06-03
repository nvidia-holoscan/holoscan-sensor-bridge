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

#ifndef SRC_HOLOLINK_OPERATORS_LINUX_RECEIVER_LINUX_RECEIVER
#define SRC_HOLOLINK_OPERATORS_LINUX_RECEIVER_LINUX_RECEIVER

#include <atomic>
#include <queue>
#include <semaphore.h>
#include <stdint.h>

#include <cuda.h>

#include <hololink/core/hololink.hpp>

namespace hololink::operators {

class LinuxReceiverMetadata {
public:
    // Data accumulated just in this frame
    unsigned frame_packets_received = 0;
    unsigned frame_bytes_received = 0;
    unsigned received_frame_number = 0;
    uint64_t frame_start_s = 0;
    uint64_t frame_start_ns = 0;
    uint64_t frame_end_s = 0;
    uint64_t frame_end_ns = 0;
    uint32_t imm_data = 0;
    int64_t received_s = 0;
    int64_t received_ns = 0;
    CUdeviceptr frame_memory = 0;
    // Data accumulated over the life of the application
    uint64_t packets_dropped = 0;
    // Data received directly from HSB.
    Hololink::FrameMetadata frame_metadata;
    uint32_t frame_number = 0; // 32-bit extended version of the 16-bit frame_metadata.frame_number
};

class LinuxReceiverDescriptor;

class LinuxReceiver {
public:
    LinuxReceiver(CUdeviceptr cu_buffer,
        size_t cu_buffer_size,
        size_t cu_page_size,
        unsigned pages,
        int socket,
        uint64_t received_address_offset,
        unsigned queue_size);

    ~LinuxReceiver();

    /**
     * Runs perpetually, looking for packets received
     * via socket (given in the constructor).  Call
     * close() to inspire this method to return.
     */
    void run();

    /**
     * Set a flag that will encourage run() to return.
     */
    void close();

    /**
     * Block until the next complete frame arrives.
     * @returns false if timeout_ms elapses before
     * the complete frame is observed.
     * @param metadata is updated with statistics
     * collected with the video frame.
     */
    bool get_next_frame(unsigned timeout_ms, LinuxReceiverMetadata& metadata, CUstream cuda_stream);

    /** Returns false if get_next_frame() may block. */
    bool frames_ready();

    uint32_t get_qp_number() { return qp_number_; };

    uint32_t get_rkey() { return rkey_; };

    /**
     * If the application schedules the call to get_next_frame after this
     * callback occurs, then get_next_frame won't block.
     */
    void set_frame_ready(std::function<void(const LinuxReceiver&)> frame_ready);

protected:
    // Signal that a new frame is available and return the next available descriptor.
    LinuxReceiverDescriptor* signal(LinuxReceiverDescriptor* descriptor);

protected:
    const CUdeviceptr cu_buffer_;
    const size_t cu_buffer_size_;
    const size_t cu_page_size_;
    const unsigned pages_;
    const int socket_;
    const uint64_t received_address_offset_;
    const unsigned queue_size_;
    bool volatile exit_;
    pthread_mutex_t ready_mutex_;
    pthread_cond_t ready_condition_;
    std::atomic<bool> consumer_ready_;
    uint32_t qp_number_;
    uint32_t rkey_;
    uint8_t* local_;
    std::queue<LinuxReceiverDescriptor*> available_;
    std::queue<LinuxReceiverDescriptor*> ready_;
    std::function<void(const LinuxReceiver&)> frame_ready_;
    /** Sign-extended frame_number value. */
    ExtendedCounter<uint32_t, uint16_t> frame_number_;
};

} // namespace hololink::operators

#endif /* SRC_HOLOLINK_OPERATORS_LINUX_RECEIVER_LINUX_RECEIVER */
