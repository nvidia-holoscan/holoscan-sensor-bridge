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

#ifndef SRC_HOLOLINK_OPERATORS_LINUX_RECEIVER_LINUX_RECEIVER
#define SRC_HOLOLINK_OPERATORS_LINUX_RECEIVER_LINUX_RECEIVER

#include <atomic>
#include <semaphore.h>
#include <stdint.h>

#include <cuda.h>

namespace hololink::operators {

class LinuxReceiverMetadata {
public:
    // Data accumulated just in this frame
    unsigned frame_packets_received;
    unsigned frame_bytes_received;
    unsigned frame_number;
    uint64_t frame_start_s;
    uint64_t frame_start_ns;
    uint64_t frame_end_s;
    uint64_t frame_end_ns;
    // Data accumulated over the life of the application
    //  uint64_t packets_received;
    //  uint64_t frames_received;
    //  uint64_t frames_dropped;
    //  uint64_t frames_timed_out;
    //  uint64_t frames_seen;
    //  uint64_t checksum_errors;
    //  uint64_t unexpected_byte_counters;
    //  uint64_t unexpected_packet_counts;
    //  uint64_t data_overflow;
    //  uint64_t packets_dropped;
    //  uint64_t rejected;
};

class LinuxReceiverDescriptor;

class LinuxReceiver {
public:
    LinuxReceiver(CUdeviceptr cu_buffer,
        size_t cu_buffer_size,
        int socket);

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
    bool get_next_frame(unsigned timeout_ms, LinuxReceiverMetadata& metadata);

    uint32_t get_qp_number() { return qp_number_; };

    uint32_t get_rkey() { return rkey_; };

protected:
    // Blocks execution until signal() is called;
    // @returns false if timeout_ms elapses before
    // signal is observed.
    bool wait(unsigned timeout_ms);

    // Pass a message to wait() telling it to wake up.
    void signal();

protected:
    CUdeviceptr cu_buffer_;
    size_t cu_buffer_size_;
    int socket_;
    bool volatile ready_;
    bool volatile exit_;
    pthread_mutex_t ready_mutex_;
    pthread_cond_t ready_condition_;
    uint32_t qp_number_;
    uint32_t rkey_;
    uint8_t* local_;
    std::atomic<LinuxReceiverDescriptor*> available_;
    LinuxReceiverDescriptor* busy_;
};

} // namespace hololink::operators

#endif /* SRC_HOLOLINK_OPERATORS_LINUX_RECEIVER_LINUX_RECEIVER */
