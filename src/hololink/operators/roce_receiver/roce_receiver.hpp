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

#include <hololink/native/deserializer.hpp>
#include <hololink/native/nvtx_trace.hpp>

namespace hololink::operators {

class RoceReceiverMetadata {
public:
    uint64_t rx_write_requests; // over all of time
    uint64_t frame_number;
    uint64_t frame_end_s;
    uint64_t frame_end_ns;
    uint32_t imm_data;
    int64_t received_ns;
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

protected:
    void signal();

    bool wait(unsigned timeout_ms);

    void free_ib_resources();

    bool check_async_events();

protected:
    char* ibv_name_;
    unsigned ibv_port_;
    CUdeviceptr cu_buffer_;
    size_t cu_buffer_size_;
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
    uint64_t frame_number_;
    int rx_write_requests_fd_;
    uint64_t volatile rx_write_requests_; // over all of time
    struct timespec frame_end_;
    uint32_t volatile imm_data_;
    struct timespec event_time_;
    int64_t volatile received_ns_;

    std::mutex& get_lock(); // Ensures reentrency protection for ibv calls.
};

} // namespace hololink::operators

#endif /* SRC_HOLOLINK_OPERATORS_ROCE_RECEIVER_ROCE_RECEIVER */
