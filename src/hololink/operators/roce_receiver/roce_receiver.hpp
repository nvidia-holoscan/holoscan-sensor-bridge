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

#ifndef SRC_HOLOLINK_OPERATORS_ROCE_RECEIVER_ROCE_RECEIVER
#define SRC_HOLOLINK_OPERATORS_ROCE_RECEIVER_ROCE_RECEIVER

#include <atomic>
#include <mutex>
#include <stddef.h>
#include <stdint.h>

#include <cuda.h>

#include <infiniband/verbs.h>

#include <hololink/core/deserializer.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/core/nvtx_trace.hpp>

namespace hololink::operators {

class RoceReceiverMetadata {
public:
    uint64_t rx_write_requests = 0; // over all of time
    uint64_t received_frame_number = 0;
    uint32_t imm_data = 0;
    uint64_t received_s = 0;
    uint64_t received_ns = 0;
    CUdeviceptr frame_memory = 0;
    CUdeviceptr metadata_memory = 0;
    uint32_t dropped = 0;
    // Data received directly from HSB.
    Hololink::FrameMetadata frame_metadata;
    uint32_t frame_number = 0; // 32-bit extended version of the 16-bit frame_metadata.frame_number
};

/**
 *
 */
class RoceReceiver {
public:
    /**
     * @param ibv_name Name of infiniband verbs device name, e.g. "roceP5p3s0f0"
     */
    RoceReceiver(
        const char* ibv_name,
        unsigned ibv_port,
        CUdeviceptr cu_buffer,
        size_t cu_buffer_size,
        size_t cu_frame_size,
        size_t cu_page_size,
        unsigned pages,
        size_t metadata_offset,
        const char* peer_ip);

    ~RoceReceiver();

    void blocking_monitor();

    bool start();

    void close(); // causes the run method to terminate

    /**
     * Block until the next complete frame arrives.
     * @returns false if timeout_ms elapses before
     * the complete frame is observed.
     * @param metadata is updated with statistics
     * collected with the video frame.
     */
    bool get_next_frame(unsigned timeout_ms, RoceReceiverMetadata& metadata);

    uint32_t get_qp_number() { return qp_number_; };

    uint32_t get_rkey() { return rkey_; };

    // What target address do we write into HSB?
    uint64_t external_frame_memory();

    /**
     * If the application schedules the call to get_next_frame after this
     * callback occurs, then get_next_frame won't block.
     */
    void set_frame_ready(std::function<void(const RoceReceiver&)> frame_ready);

protected:
    void free_ib_resources();

    bool check_async_events();

protected:
    char* ibv_name_;
    unsigned ibv_port_;
    CUdeviceptr cu_buffer_;
    size_t cu_buffer_size_;
    size_t cu_frame_size_;
    size_t cu_page_size_;
    unsigned pages_;
    size_t metadata_offset_;
    char* peer_ip_;
    struct ibv_qp* ib_qp_;
    struct ibv_mr* ib_mr_;
    struct ibv_cq* ib_cq_;
    struct ibv_pd* ib_pd_;
    struct ibv_context* ib_context_;
    struct ibv_comp_channel* ib_completion_channel_;
    uint32_t qp_number_;
    uint32_t rkey_;
    bool volatile ready_;
    pthread_mutex_t ready_mutex_;
    pthread_cond_t ready_condition_;
    bool volatile done_;
    int control_r_, control_w_;
    uint64_t received_frame_number_;
    int rx_write_requests_fd_;
    uint64_t volatile rx_write_requests_; // over all of time
    uint32_t volatile imm_data_;
    struct timespec volatile event_time_;
    struct timespec volatile received_;
    CUdeviceptr volatile current_buffer_;
    CUstream metadata_stream_;
    uint32_t volatile dropped_;
    uint8_t* metadata_buffer_;
    uint32_t volatile received_psn_;
    unsigned volatile received_page_;
    std::function<void(const RoceReceiver&)> frame_ready_;
    /** Sign-extended frame_number value. */
    ExtendedCounter<uint32_t, uint16_t> frame_number_;

    std::mutex& get_lock(); // Ensures reentrency protection for ibv calls.
};

} // namespace hololink::operators

#endif /* SRC_HOLOLINK_OPERATORS_ROCE_RECEIVER_ROCE_RECEIVER */
