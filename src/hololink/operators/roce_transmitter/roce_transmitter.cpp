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

#include "roce_transmitter.hpp"

#include <arpa/inet.h>
#include <stdio.h>

#include <cuda.h>

#include <hololink/native/deserializer.hpp>
#include <hololink/native/nvtx_trace.hpp>

#define S2(s) #s
#define S(s) S2(s)

#if 0
// For reproducible builds
#define TRACE(fmt, ...) /* ignored */
#define DEBUG(fmt...) fprintf(stderr, "DEBUG -- " fmt)
#define ERROR(fmt...) fprintf(stderr, "ERROR -- " fmt)
#else
// For better debugging
#define TRACE(fmt, ...) /* ignored */
#define DEBUG(fmt...) fprintf(stderr, "DEBUG " __FILE__ ":" S(__LINE__) " -- " fmt)
#define ERROR(fmt...) fprintf(stderr, "ERROR " __FILE__ ":" S(__LINE__) " -- " fmt)
#endif

namespace hololink::operators {

RoceTransmitter::RoceTransmitter(
    const char* ibv_name,
    unsigned ibv_port,
    void* buffer,
    size_t buffer_size,
    char const* peer_ip)
    : ibv_name_(strdup(ibv_name))
    , ibv_port_(ibv_port)
    , buffer_(buffer)
    , buffer_size_(buffer_size)
    , peer_ip_(strdup(peer_ip))
    , ib_qp_(NULL)
    , ib_mr_(NULL)
    , ib_cq_(NULL)
    , ib_pd_(NULL)
    , ib_context_(NULL)
{
    DEBUG("buffer=%p buffer_size=%u\n",
        buffer, (unsigned)buffer_size);
}

RoceTransmitter::~RoceTransmitter()
{
    free(ibv_name_);
    free(peer_ip_);
}

void RoceTransmitter::start(uint32_t destination_qp)
{
    DEBUG("Starting.\n");
    native::NvtxTrace::setThreadName("socket_receiver");

    // Find the IB controller
    bool ok = false;
    unsigned completion_queue_size = 10;
    int access = 0;
    struct ibv_qp_init_attr ib_qp_init_attr = { 0 }; // C fills the rest with 0s
    struct ibv_qp_attr ib_qp_attr = { .port_num = 0 }; // C fills in the rest with 0s
    int flags = 0;
    //  struct ibv_recv_wr ib_wr = { 0 }; // C fills the rest with 0s
    //  struct ibv_recv_wr * bad_wr = NULL;
    int gid_index = 0;
    union ibv_gid remote_gid = { 0 }; // C fills the rest with 0s
    unsigned long client_ip = 0;
    uint64_t client_interface_id = 0;

    int num_devices = 0;
    struct ibv_device** ib_devices = ibv_get_device_list(&num_devices);
    if (!ib_devices) {
        ERROR("ibv_get_device_list failed; errno=%d.\n", (int)errno);
        return;
    }
    if (num_devices < 0) {
        ERROR("ibv_get_device_list set unexpected value for num_devices=%d.\n", num_devices);
        return;
    }
    struct ibv_device* ib_device = NULL;
    for (unsigned i = 0; i < (unsigned)num_devices; i++) {
        const char* device_name = ibv_get_device_name(ib_devices[i]);
        DEBUG("ibv_get_device_list[%d]=%s.\n", i, device_name);
        if (strcmp(device_name, ibv_name_) != 0) {
            continue;
        }
        ib_device = ib_devices[i];
        break;
    }
    if (ib_device == NULL) {
        ERROR("ibv_get_device_list didnt find a device named \"%s\".\n", ibv_name_);
        ibv_free_device_list(ib_devices);
        return;
    }

    // Open the IB device
    ib_context_ = ibv_open_device(ib_device);
    if (!ib_context_) {
        ERROR("ibv_open_device failed, errno=%d.\n", (int)errno);
        return;
    }
    ibv_free_device_list(ib_devices); // Note that "ib_device" is invalid after this.
    ib_device = NULL;
    ib_devices = NULL;

    struct ibv_device_attr ib_device_attr = { 0 }; // C fills the rest with 0s
    if (ibv_query_device(ib_context_, &ib_device_attr)) {
        ERROR("ibv_query_device failed, errno=%d.\n", (int)errno);
        stop();
        return;
    }

    struct ibv_port_attr ib_port_attr = { .flags = 0 }; // C fills the rest with 0s
    if (ibv_query_port(ib_context_, ibv_port_, &ib_port_attr)) {
        ERROR("ibv_query_port failed, errno=%d.\n", (int)errno);
        stop();
        return;
    }

    // Fetch the GID
    ok = false;
    struct ibv_gid_entry ib_gid_entry = { 0 }; // C fills the rest with 0s
    for (gid_index = 0; 1; gid_index++) {
        uint32_t flags = 0;
        int r = ibv_query_gid_ex(ib_context_, ibv_port_, gid_index, &ib_gid_entry, flags);
        if (r && (errno != ENODATA)) {
            break;
        }

        struct ibv_gid_entry* u = &(ib_gid_entry);
        DEBUG("gid_index=%u gid_entry(gid_index=%u port_num=%u gid_type=%u ndev_ifindex=%d subnet_prefix=%d interface_id=0x%X)\n", (unsigned)gid_index, (unsigned)u->gid_index, (unsigned)u->port_num, (unsigned)u->gid_type, (unsigned)u->ndev_ifindex, (unsigned)u->gid.global.subnet_prefix, (unsigned)u->gid.global.interface_id);

        if (ib_gid_entry.gid_type != IBV_GID_TYPE_ROCE_V2) {
            continue;
        }
        if (ib_gid_entry.gid.global.subnet_prefix != 0) {
            continue;
        }
        if ((ib_gid_entry.gid.global.interface_id & 0xFFFFFFFF) != 0xFFFF0000) {
            continue;
        }
        ok = true;
        break;
    }
    if (!ok) {
        ERROR("Cannot find GID for IBV_GID_TYPE_ROCE_V2.\n");
        stop();
        return;
    }

    // Create a protection domain
    ib_pd_ = ibv_alloc_pd(ib_context_);
    if (ib_pd_ == NULL) {
        ERROR("Cannot allocate a protection domain, errno=%d.\n", (int)errno);
        stop();
        return;
    }

    // Create a completion queue
    ib_cq_ = ibv_create_cq(ib_context_, completion_queue_size, NULL, NULL, 0);
    if (ib_cq_ == NULL) {
        ERROR("Cannot create a completion queue, errno=%d.\n", (int)errno);
        stop();
        return;
    }

    // Provide access to the frame buffer
    access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
    ib_mr_ = ibv_reg_mr(ib_pd_, buffer_, buffer_size_, access);
    if (ib_mr_ == NULL) {
        ERROR("Cannot register memory region p=%p size=%u, errno=%d.\n", buffer_, (unsigned)buffer_size_, (int)errno);
        stop();
        return;
    }

    // Set up the queue pair
    ib_qp_init_attr = {
        .send_cq = ib_cq_,
        .recv_cq = ib_cq_,
        .cap = {
            .max_send_wr = 4096,
            .max_recv_wr = 4096,
            .max_send_sge = 1,
            .max_recv_sge = 1,
        },
        .qp_type = IBV_QPT_UC,
        .sq_sig_all = 0,
    }; // C++ sets the rest of the values to 0
    ib_qp_ = ibv_create_qp(ib_pd_, &ib_qp_init_attr);
    if (ib_qp_ == NULL) {
        ERROR("Cannot create queue pair, errno=%d.\n", (int)errno);
        stop();
        return;
    }

    // qp is currently RST; go to INIT
    ib_qp_attr = {
        .qp_state = IBV_QPS_INIT,
        .qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE,
        .pkey_index = 0,
        .port_num = (uint8_t)ibv_port_,
    }; // C sets the rest to 0s
    flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    if (ibv_modify_qp(ib_qp_, &ib_qp_attr, flags)) {
        ERROR("Cannot modify queue pair to IBV_QPS_INIT, errno=%d.\n", (int)errno);
        stop();
        return;
    }
#if 0
    // Set up the list of work requests
    ib_wr = {
        .wr_id = 1,
        .next = NULL,
    };
    if (ibv_post_recv(ib_qp_, &ib_wr, &bad_wr)) {
        ERROR("Cannot post the receiver work list, errno=%d.\n", (int) errno);
        stop();
        return;
    }
#endif
    // qp is currently INIT; go to RTR
    if (inet_pton(AF_INET, peer_ip_, &client_ip) != 1) {
        ERROR("Unable to convert \"%s\" to an IP address.\n", peer_ip_);
        stop();
        return;
    }
    // client_ip is in network-byte-order
    client_interface_id = client_ip;
    client_interface_id <<= 32;
    client_interface_id |= 0xFFFF0000;
    remote_gid = {
        .global = {
            .subnet_prefix = 0,
            .interface_id = client_interface_id,
        },
    };

    ib_qp_attr = {
        .qp_state = IBV_QPS_RTR,
        .path_mtu = IBV_MTU_4096,
        .rq_psn = 0,
        .dest_qp_num = destination_qp,
        .ah_attr = {
            .grh = {
                .dgid = remote_gid,
                .sgid_index = (uint8_t)gid_index,
                .hop_limit = 0xFF,
            },
            .dlid = 0,
            .sl = 0,
            .src_path_bits = 0,
            .is_global = 1,
            .port_num = (uint8_t)ibv_port_,
        },
    }; // C sets the rest to 0s
    flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN;
    if (ibv_modify_qp(ib_qp_, &ib_qp_attr, flags)) {
        ERROR("Cannot modify queue pair to IBV_QPS_RTR, errno=%d.\n", (int)errno);
        stop();
        return;
    }

    // RTR to RTS
    ib_qp_attr = {
        .qp_state = IBV_QPS_RTS,
        .sq_psn = 0,
    }; // C sets the rest to 0s
    flags = IBV_QP_STATE | IBV_QP_SQ_PSN;
    if (ibv_modify_qp(ib_qp_, &ib_qp_attr, flags)) {
        ERROR("Cannot modify queue pair to IBV_QPS_RTS, errno=%d.\n", (int)errno);
        stop();
        return;
    }
}

void RoceTransmitter::stop()
{
    if (ib_qp_ != NULL) {
        if (ibv_destroy_qp(ib_qp_)) {
            ERROR("ibv_destroy_qp failed, errno=%d.\n", (int)errno);
        }
        ib_qp_ = NULL;
    }

    if (ib_mr_ != NULL) {
        if (ibv_dereg_mr(ib_mr_)) {
            ERROR("ibv_dereg_mr failed, errno=%d.\n", (int)errno);
        }
        ib_mr_ = NULL;
    }

    if (ib_cq_ != NULL) {
        if (ibv_destroy_cq(ib_cq_)) {
            ERROR("ibv_destroy_cq failed, errno=%d.\n", (int)errno);
        }
        ib_cq_ = NULL;
    }

    if (ib_pd_ != NULL) {
        if (ibv_dealloc_pd(ib_pd_)) {
            ERROR("ibv_dealloc_pd failed, errno=%d.\n", (int)errno);
        }
        ib_pd_ = NULL;
    }

    if (ib_context_ != NULL) {
        if (ibv_close_device(ib_context_)) {
            ERROR("ibv_close_device failed, errno=%d.\n", (int)errno);
        }
        ib_context_ = NULL;
    }

    TRACE("Done.\n");
}

static uint64_t write_id = 0;

bool RoceTransmitter::write_request(uint64_t remote_address, uint32_t rkey, uint64_t buffer, uint32_t bytes)
{
    check_cq();
    struct ibv_sge ib_sge = {
        .addr = buffer,
        .length = bytes,
        .lkey = ib_mr_->lkey,
    };

    struct ibv_send_wr ib_send_wr = {
        .wr_id = write_id,
        .next = NULL,
        .sg_list = &ib_sge,
        .num_sge = 1,
        .opcode = IBV_WR_RDMA_WRITE,
        .send_flags = IBV_SEND_SIGNALED,
        .wr = {
            .rdma = {
                .remote_addr = remote_address,
                .rkey = rkey,
            },
        },
    };

    struct ibv_send_wr* bad_wr = NULL;
    bool r = true;
    int v = ibv_post_send(ib_qp_, &ib_send_wr, &bad_wr);

    if (v) {
        ERROR("ibv_post_send failed, v=%d errno=%d.\n", v, (int)errno);
        r = false;
    }

    write_id++;
    return r;
}

bool RoceTransmitter::write_immediate_request(uint64_t remote_address, uint32_t rkey, uint64_t buffer, uint32_t bytes, uint32_t imm_data)
{
    check_cq();

    struct ibv_sge ib_sge = {
        .addr = buffer,
        .length = bytes,
        .lkey = ib_mr_->lkey,
    };

    struct ibv_send_wr ib_send_wr = {
        .wr_id = write_id,
        .next = NULL,
        .sg_list = &ib_sge,
        .num_sge = 1,
        .opcode = IBV_WR_RDMA_WRITE_WITH_IMM,
        .send_flags = IBV_SEND_SIGNALED,
        .imm_data = htonl(imm_data),
        .wr = {
            .rdma = {
                .remote_addr = remote_address,
                .rkey = rkey,
            },
        },
    };

    struct ibv_send_wr* bad_wr = NULL;
    bool r = true;
    if (ibv_post_send(ib_qp_, &ib_send_wr, &bad_wr)) {
        ERROR("ibv_post_send failed, errno=%d.\n", (int)errno);
        r = false;
    }

    write_id = (write_id & ~0xFFFF) + 0x10000;
    return r;
}

bool RoceTransmitter::check_cq()
{
    if (!ib_cq_) {
        return true;
    }
    unsigned i;
    struct ibv_wc ib_wc = { 0 };
    for (i = 0; i < 1000; i++) {
        int r = ibv_poll_cq(ib_cq_, 1, &ib_wc);
        if (r < 0) {
            ERROR("ibv_poll_cq failed, errno=%d.\n", (int)errno);
            return false;
        }
        if (r == 0) {
            break;
        }
    }
    return true;
}

} // namespace hololink::operators
