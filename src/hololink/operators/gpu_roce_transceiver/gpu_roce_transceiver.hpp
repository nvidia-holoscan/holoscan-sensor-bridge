
/**
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

#ifndef SRC_HOLOLINK_OPERATORS_GPU_ROCE_TRANSCEIVER_HPP
#define SRC_HOLOLINK_OPERATORS_GPU_ROCE_TRANSCEIVER_HPP

#include <arpa/inet.h>
#include <atomic>
#include <fcntl.h>
#include <functional>
#include <infiniband/mlx5dv.h>
#include <infiniband/verbs.h>
#include <memory>
#include <mutex>
#include <poll.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <unistd.h>

#include <chrono>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <doca_dev.h>
#include <doca_error.h>
#include <doca_gpunetio.h>
#include <doca_log.h>
#include <doca_rdma_bridge.h>
#include <doca_uar.h>
#include <doca_umem.h>
#include <doca_verbs.h>
#include <doca_verbs_bridge.h>

#include "hololink/operators/roce_receiver/roce_receiver.hpp"
#include <hololink/core/logging_internal.hpp>
#include <hololink/core/nvtx_trace.hpp>

#define WQE_NUM 64
#define DOCA_UC_QP_RST2INIT_REQ_ATTR_MASK \
    (DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE | DOCA_VERBS_QP_ATTR_PKEY_INDEX | DOCA_VERBS_QP_ATTR_PORT_NUM)
#define DOCA_UC_QP_INIT2RTR_REQ_ATTR_MASK \
    (DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_RQ_PSN | DOCA_VERBS_QP_ATTR_DEST_QP_NUM | DOCA_VERBS_QP_ATTR_PATH_MTU | DOCA_VERBS_QP_ATTR_AH_ATTR)
#define DOCA_UC_QP_RTR2RTS_REQ_ATTR_MASK (DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_SQ_PSN)

#define NUM_OF(x) (sizeof(x) / sizeof(x[0]))
#define VERBS_TEST_DBR_SIZE (8)
#define ROUND_UP(unaligned_mapping_size, align_val) ((unaligned_mapping_size) + (align_val)-1) & (~((align_val)-1))
#define ALIGN_SIZE(size, align) size = ((size + (align)-1) / (align)) * (align);

#define GPU_ROCE_MAX_RECV_INLINE_WQE 32
#define DOCA_VERBS_ENABLE_RECV_INLINE 0
#define DOCA_ENABLE_NON_RDMA 0

enum gpu_roce_max_frame_size {
    GPU_ROCE_MAX_FRAME_SIZE_0B = 0,
    GPU_ROCE_MAX_FRAME_SIZE_4B = 4,
    GPU_ROCE_MAX_FRAME_SIZE_8B = 8,
    GPU_ROCE_MAX_FRAME_SIZE_12B = 12,
    GPU_ROCE_MAX_FRAME_SIZE_16B = 16,
    GPU_ROCE_MAX_FRAME_SIZE_20B = 20,
    GPU_ROCE_MAX_FRAME_SIZE_24B = 24,
    GPU_ROCE_MAX_FRAME_SIZE_28B = 28,
    GPU_ROCE_MAX_FRAME_SIZE_32B = 32,
    GPU_ROCE_MAX_FRAME_SIZE_36B = 36,
    GPU_ROCE_MAX_FRAME_SIZE_40B = 40,
    GPU_ROCE_MAX_FRAME_SIZE_44B = 44
};

struct gpu_roce_ring_buffer {
    void* addr;
    size_t stride_sz;
    uint32_t stride_num;
    struct ibv_mr* addr_mr;
    int dmabuf_fd;
    uint64_t* flag;
};

// Static GPU datapath utils functions
inline size_t get_page_size(void)
{
    long ret = sysconf(_SC_PAGESIZE);
    if (ret == -1)
        return 4096; // 4KB, default Linux page size

    return (size_t)ret;
}

namespace hololink::operators {

class DocaCq {
public:
    DocaCq(uint32_t cqe_num_,
        struct doca_gpu* gdev_,
        struct doca_dev* ndev_,
        struct doca_uar* uar_,
        struct doca_verbs_context* vctx_,
        bool umem_cpu_)
        : cqe_num(cqe_num_)
        , gdev(gdev_)
        , ndev(ndev_)
        , uar(uar_)
        , vctx(vctx_)
        , umem_cpu(umem_cpu) {};

    ~DocaCq()
    {
        if (umem)
            doca_umem_destroy(umem);
        if (umem_dev_ptr)
            doca_gpu_mem_free(gdev, umem_dev_ptr);
        if (cq != nullptr)
            doca_verbs_cq_destroy(cq);
    }

    doca_error_t create();
    struct doca_verbs_cq* get() const
    {
        return cq;
    };

    struct doca_gpu* gdev;
    struct doca_dev* ndev;
    struct doca_uar* uar;
    struct doca_verbs_context* vctx;
    uint32_t cqe_num;
    struct doca_verbs_cq* cq = nullptr;
    void* umem_dev_ptr;
    struct doca_umem* umem;
    bool umem_cpu;
};

class DocaQp {
public:
    DocaQp(uint32_t wqe_num_,
        struct doca_gpu* gdev_,
        struct doca_dev* ndev_,
        struct doca_uar* uar_,
        struct doca_verbs_context* vctx_,
        struct doca_verbs_pd* vpd_,
        struct doca_verbs_cq* cq_rq_,
        struct doca_verbs_cq* cq_sq_,
        bool umem_cpu_)
        : wqe_num(wqe_num_)
        , gdev(gdev_)
        , ndev(ndev_)
        , uar(uar_)
        , vctx(vctx_)
        , vpd(vpd_)
        , cq_rq(cq_rq_)
        , cq_sq(cq_sq_)
        , umem_cpu(umem_cpu) {};

    ~DocaQp();

    doca_error_t create(struct doca_verbs_context* verbs_ctx, const size_t frame_size);
    doca_error_t create_ring(size_t stride_sz, unsigned stride_num, struct ibv_pd* ibv_pd);
    doca_error_t connect(struct doca_verbs_gid& doca_rgid, uint32_t gid_index, uint32_t dest_qp_num);

    struct doca_verbs_qp* get() const
    {
        return qp;
    };

    struct doca_gpu_verbs_qp* get_gpu() const
    {
        return gpu_qp;
    };
    struct doca_gpu_dev_verbs_qp* get_gpu_dev() const
    {
        return gpu_dev_qp;
    };

    struct doca_gpu* gdev;
    struct doca_dev* ndev;
    struct doca_uar* uar;
    struct doca_verbs_context* vctx;
    struct doca_verbs_pd* vpd;
    uint32_t wqe_num;
    struct doca_verbs_qp* qp = nullptr;
    void* umem_dev_ptr;
    struct doca_umem* umem;
    struct doca_umem* umem_dbr;
    void* umem_dbr_dev_ptr;
    struct doca_gpu_verbs_qp* gpu_qp;
    struct doca_gpu_dev_verbs_qp* gpu_dev_qp;
    struct doca_verbs_cq* cq_rq;
    struct doca_verbs_cq* cq_sq;

    struct gpu_roce_ring_buffer gpu_rx_ring;
    struct gpu_roce_ring_buffer gpu_tx_ring;

    bool umem_cpu;
};

struct doca_verbs_context* open_ib_device(char* name);

struct cpu_proxy_args {
    struct doca_gpu_verbs_qp* qp_cpu;
    uint64_t* exit_flag;
};

class GpuRoceTransceiver {
public:
    GpuRoceTransceiver(const char* ibv_name, unsigned ibv_port,
        uint32_t gpu_id,
        size_t cu_frame_size, size_t cu_page_size,
        unsigned pages, const char* peer_ip,
        const bool forward, const bool rx_only, const bool tx_only);

    ~GpuRoceTransceiver();

    void blocking_monitor();
    bool start();
    void close();

    uint32_t get_qp_number();
    uint32_t get_rkey();
    uint64_t external_frame_memory();

    void set_frame_ready(std::function<void()> frame_ready)
    {
        frame_ready_ = frame_ready;
    }

    void* get_rx_ring_data_addr();
    uint64_t* get_rx_ring_flag_addr();
    void* get_tx_ring_data_addr();
    uint64_t* get_tx_ring_flag_addr();

    bool get_next_frame(unsigned timeout_ms, RoceReceiverMetadata& metadata);

private:
    bool check_async_events();

    char* ibv_name_;
    unsigned ibv_port_;
    uint32_t gpu_id_;
    size_t cu_frame_size_;
    size_t cu_page_size_;
    unsigned pages_;
    char* peer_ip_;

    struct ibv_pd* ibv_pd = nullptr;
    struct doca_dev* doca_device_ = nullptr;
    struct doca_gpu* doca_gpu_device_ = nullptr;
    struct doca_verbs_context* doca_verbs_ctx_ = nullptr;
    struct doca_verbs_pd* doca_pd_ = nullptr;
    DocaCq* doca_cq_rq;
    DocaCq* doca_cq_sq;
    DocaQp* doca_qp;
    struct doca_uar* uar_ = nullptr;

    uint32_t qp_number_ = 0;
    uint32_t rkey_ = 0;

    bool volatile ready_;
    pthread_mutex_t ready_mutex_;
    pthread_cond_t ready_condition_;
    bool volatile done_;
    int control_r_, control_w_;
    uint64_t frame_number_;
    int rx_write_requests_fd_;
    uint64_t volatile rx_write_requests_; // over all of time
    uint32_t volatile imm_data_;
    struct timespec volatile event_time_;
    struct timespec volatile received_;
    uint32_t volatile dropped_;
    uint32_t volatile received_psn_;
    unsigned volatile received_page_;
    std::function<void()> frame_ready_;
    bool gpu_datapath_;
    uint32_t* cpu_exit_flag;
    uint32_t* gpu_exit_flag;
    bool doca_kernel_launched = false;
    cudaStream_t forward_stream;
    cudaStream_t rx_only_stream;
    cudaStream_t tx_only_stream;

    bool forward_;
    bool rx_only_;
    bool tx_only_;

    pthread_t cpu_proxy_thread_id;
    struct cpu_proxy_args targs;
    std::mutex& get_lock();

    bool umem_cpu;
};

extern "C" {

doca_error_t GpuRoceTransceiverPrepareKernel(cudaStream_t stream,
    struct doca_gpu_dev_verbs_qp* qp,
    size_t frame_size,
    uint32_t cu_page_mkey,
    uint32_t cuda_blocks,
    uint32_t cuda_threads);

doca_error_t GpuRoceTransceiverForwardKernel(cudaStream_t stream,
    struct doca_gpu_dev_verbs_qp* qp,
    uint32_t* exit_flag,
    uint8_t* cu_buffer,
    size_t cu_buffer_size,
    uint32_t cu_buffer_mkey,
    unsigned pages,
    size_t frame_size,
    uint32_t cuda_blocks,
    uint32_t cuda_threads);

doca_error_t GpuRoceTransceiverRxOnlyKernel(cudaStream_t stream,
    struct doca_gpu_dev_verbs_qp* qp, uint32_t* exit_flag,
    uint8_t* ring_buf, const size_t ring_buf_stride_sz,
    const uint32_t ring_buf_stride_num, uint64_t* ring_flag,
    size_t frame_size, uint32_t cuda_blocks, uint32_t cuda_threads);

doca_error_t GpuRoceTransceiverTxOnlyKernel(cudaStream_t stream,
    struct doca_gpu_dev_verbs_qp* qp, uint32_t* exit_flag,
    uint8_t* ring_buf, const size_t ring_buf_stride_sz,
    const uint32_t ring_buf_stride_num, uint64_t* ring_flag,
    size_t frame_size, uint32_t cuda_blocks, uint32_t cuda_threads);
}

} // namespace hololink::operators

#endif // SRC_HOLOLINK_OPERATORS_GPU_ROCE_TRANSCEIVER_HPP
