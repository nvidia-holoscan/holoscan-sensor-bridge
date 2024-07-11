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

#ifndef SRC_HOLOLINK_OPERATORS_ROCE_TRANSMITTER_ROCE_TRANSMITTER
#define SRC_HOLOLINK_OPERATORS_ROCE_TRANSMITTER_ROCE_TRANSMITTER

#include <atomic>
#include <stddef.h>
#include <stdint.h>

#include <infiniband/verbs.h>

namespace hololink::operators {

class RoceTransmitter {
public:
    /**
     * @param ibv_name Name of infiniband verbs device name, e.g. "roceP5p3s0f0"
     */
    RoceTransmitter(
        const char* ibv_name,
        unsigned ibv_port,
        void* buffer,
        size_t buffer_size,
        char const* peer_ip);

    ~RoceTransmitter();

    void start(uint32_t destination_qp);

    void stop();

    bool check_cq();

    bool write_request(uint64_t remote_address, uint32_t rkey, uint64_t buffer, uint32_t bytes);
    bool write_immediate_request(uint64_t remote_address, uint32_t rkey, uint64_t buffer, uint32_t bytes, uint32_t imm_data);

protected:
    char* ibv_name_;
    unsigned ibv_port_;
    void* buffer_;
    size_t buffer_size_;
    char* peer_ip_;
    struct ibv_qp* ib_qp_;
    struct ibv_mr* ib_mr_;
    struct ibv_cq* ib_cq_;
    struct ibv_pd* ib_pd_;
    struct ibv_context* ib_context_;
};

} // namespace hololink::operators

#endif /* SRC_HOLOLINK_OPERATORS_ROCE_TRANSMITTER_ROCE_TRANSMITTER */
