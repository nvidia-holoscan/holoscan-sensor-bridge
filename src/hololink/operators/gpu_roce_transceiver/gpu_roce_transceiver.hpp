/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
#include <sstream>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <unistd.h>
#include <vector>

#include <chrono>
#include <iostream>

#include <doca_dev.h>
#include <doca_gpunetio.h>
#include <doca_log.h>
#include <doca_rdma_bridge.h>
#include <doca_uar.h>
#include <doca_umem.h>
#include <doca_verbs.h>
#include <doca_verbs_bridge.h>

#include <hololink/core/logging_internal.hpp>
#include <hololink/core/nvtx_trace.hpp>

#include "gpu_roce_transceiver_common.hpp"

struct gpu_roce_ring_buffer {
    uint8_t* addr = nullptr;
    size_t stride_sz = 0;
    uint32_t stride_num = 0;
    struct ibv_mr* addr_mr = nullptr;
    int dmabuf_fd = -1;
    uint64_t* flag = nullptr;
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
        , umem_cpu(umem_cpu_) {};

    ~DocaCq()
    {
        // The CQ's DEVX object holds a reference to the umem, so the umem
        // cannot be deregistered until the CQ itself is gone.
        if (cq != nullptr)
            doca_verbs_cq_destroy(cq);
        if (umem)
            doca_umem_destroy(umem);
        if (umem_dev_ptr)
            doca_gpu_mem_free(gdev, umem_dev_ptr);
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
    void* umem_dev_ptr = nullptr;
    struct doca_umem* umem = nullptr;
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
        , umem_cpu(umem_cpu_) {};

    ~DocaQp();

    doca_error_t create(struct doca_verbs_context* verbs_ctx, const size_t frame_size, enum doca_gpu_dev_verbs_nic_handler nic_handler);
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
    void* umem_dev_ptr = nullptr;
    struct doca_umem* umem = nullptr;
    struct doca_umem* umem_dbr = nullptr;
    void* umem_dbr_dev_ptr = nullptr;
    struct doca_gpu_verbs_qp* gpu_qp = nullptr;
    struct doca_gpu_dev_verbs_qp* gpu_dev_qp = nullptr;
    struct doca_verbs_cq* cq_rq;
    struct doca_verbs_cq* cq_sq;

    struct gpu_roce_ring_buffer gpu_rx_ring;
    struct gpu_roce_ring_buffer gpu_tx_ring;

    bool umem_cpu;
};

struct doca_verbs_context* open_ib_device(char* name);

struct cpu_proxy_args {
    struct doca_gpu_verbs_qp* qp_cpu[GPU_ROCE_TRANSCEIVER_MAX_QP] = {};
    uint64_t* exit_flag = nullptr;
    uint16_t num_qp = 0;
};

class GpuRoceTransceiver {
public:
    GpuRoceTransceiver(const char* ibv_name, unsigned ibv_port,
        std::string tx_ibv_qp, uint32_t gpu_id,
        size_t cu_frame_size, size_t cu_page_size,
        unsigned pages, const char* peer_ip,
        const bool forward, const bool rx_only, const bool tx_only);

    ~GpuRoceTransceiver();

    void blocking_monitor();
    bool start();
    void close();

    /// Number of QPs, one per remote QP listed in tx_ibv_qp
    size_t get_num_qps() const { return tx_ibv_qp_num_.size(); }
    uint32_t get_qp_number(unsigned sensor_id);
    uint32_t get_rkey(unsigned sensor_id);
    /// Address HSB writes the frames of any QP to: each RX ring is registered at IOVA 0
    uint64_t external_frame_memory();

    void set_frame_ready(std::function<void()> frame_ready)
    {
        frame_ready_ = frame_ready;
    }

    uint8_t* get_rx_ring_data_addr(unsigned sensor_id);
    size_t get_rx_ring_stride_sz(unsigned sensor_id);
    uint32_t get_rx_ring_stride_num(unsigned sensor_id);
    uint64_t* get_rx_ring_flag_addr(unsigned sensor_id);
    uint8_t* get_tx_ring_data_addr(unsigned sensor_id);
    size_t get_tx_ring_stride_sz(unsigned sensor_id);
    uint32_t get_tx_ring_stride_num(unsigned sensor_id);
    uint64_t* get_tx_ring_flag_addr(unsigned sensor_id);

    /**
     * @brief Local key of the TX ring's memory region.
     *
     * For callers that post their own sends from the TX ring.  Valid after
     * start().
     *
     * @return the TX ring lkey, in host byte order like get_rkey().
     */
    uint32_t get_tx_ring_lkey(unsigned sensor_id);

    /// Force CPU+GPU accessible allocation for ring flags and data only
    /// (DOCA_GPU_MEM_TYPE_CPU_GPU) even on dGPU systems.  Must be called
    /// before start().  Does NOT affect CQ/QP UMEMs or TX kernel handler.
    /// Required when a CPU thread needs to read ring flags/data directly
    /// (e.g. HOST_LOOP dispatcher on Grace-Blackwell).
    void set_cpu_ring_buffers(bool enable) { cpu_ring_buffers_ = enable; }

    /** Blocks until close(); returns false (no CPU frame stream; kernel owns datapath). */
    bool get_next_frame(unsigned timeout_ms, CUstream cuda_stream);

    /** false: get_next_frame may block until teardown. */
    bool frames_ready() { return false; }

    void* get_doca_gpu_dev_qp(unsigned sensor_id)
    {
        if (!qp_created(sensor_id))
            return nullptr;
        return doca_qp_[sensor_id]->get_gpu_dev();
    }

private:
    bool check_async_events();
    bool launch_kernels();
    /// True if the QP of sensor_id is configured and created
    bool qp_created(unsigned sensor_id) const
    {
        return sensor_id < tx_ibv_qp_num_.size() && doca_qp_[sensor_id] != nullptr;
    }

    char* ibv_name_;
    unsigned ibv_port_;
    std::string tx_ibv_qp_;
    std::vector<uint32_t> tx_ibv_qp_num_;
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
    DocaCq* doca_cq_rq_[GPU_ROCE_TRANSCEIVER_MAX_QP] = {};
    DocaCq* doca_cq_sq_[GPU_ROCE_TRANSCEIVER_MAX_QP] = {};
    DocaQp* doca_qp_[GPU_ROCE_TRANSCEIVER_MAX_QP] = {};
    struct doca_uar* uar_[GPU_ROCE_TRANSCEIVER_MAX_QP] = {};

    uint32_t qp_number_ = 0;
    uint32_t rkey_ = 0;

    pthread_mutex_t ready_mutex_;
    pthread_cond_t ready_condition_;
    std::atomic<bool> done_;
    int control_r_ = -1, control_w_ = -1;
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
    uint32_t* cpu_exit_flag = nullptr;
    uint32_t* gpu_exit_flag = nullptr;
    bool doca_kernel_launched = false;
    cudaStream_t forward_stream = nullptr;
    cudaStream_t rx_only_stream = nullptr;
    cudaStream_t tx_only_stream = nullptr;
    cudaStream_t compute_stream = nullptr;

    bool forward_;
    bool rx_only_;
    bool tx_only_;

    pthread_t cpu_proxy_thread_id;
    bool cpu_proxy_started_ = false;
    struct cpu_proxy_args targs;
    std::mutex& get_lock();

    bool umem_cpu = false;
    bool cpu_ring_buffers_ = false;

    CUdevice cuDevice = 0;
    CUcontext cuContext = nullptr;
    enum doca_gpu_dev_verbs_nic_handler nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF;
};

extern "C" {

doca_error_t GpuRoceTransceiverPrepareKernel(cudaStream_t stream,
    struct doca_gpu_dev_verbs_qp* qp,
    size_t frame_size,
    uint32_t cu_page_mkey,
    uint32_t cuda_blocks,
    uint32_t cuda_threads);

doca_error_t GpuRoceTransceiverForwardKernel(cudaStream_t stream,
    uint32_t* exit_flag,
    struct doca_gpu_dev_verbs_qp** qp,
    uint8_t** ring_buf,
    size_t* ring_buf_stride_sz,
    uint32_t* ring_buf_mkey,
    uint32_t* ring_buf_stride_num,
    size_t frame_size,
    const enum doca_gpu_dev_verbs_nic_handler nic_handler,
    uint32_t cuda_blocks,
    uint32_t cuda_threads);

doca_error_t GpuRoceTransceiverRxOnlyKernel(cudaStream_t stream,
    uint32_t* exit_flag, struct doca_gpu_dev_verbs_qp** qp,
    uint8_t** ring_buf, const size_t* ring_buf_stride_sz,
    const uint32_t* ring_buf_stride_num, uint64_t** ring_flag,
    size_t frame_size, uint32_t cuda_blocks, uint32_t cuda_threads);

doca_error_t GpuRoceTransceiverTxOnlyKernel(cudaStream_t stream,
    uint32_t* exit_flag, struct doca_gpu_dev_verbs_qp** qp,
    uint8_t** ring_buf, const size_t* ring_buf_stride_sz,
    const uint32_t* ring_buf_stride_num, uint32_t* ring_buf_mkey, uint64_t** ring_flag,
    size_t frame_size, const enum doca_gpu_dev_verbs_nic_handler nic_handler,
    uint32_t cuda_blocks, uint32_t cuda_threads);

doca_error_t GpuRoceTransceiverComputeKernel(cudaStream_t stream, uint32_t* exit_flag,
    uint64_t** ring_rx_flag, const uint32_t* ring_rx_stride_num,
    uint64_t** ring_tx_flag, uint8_t** ring_tx_addr, const size_t* ring_tx_stride_sz, const uint32_t* ring_tx_stride_num,
    size_t frame_size, uint32_t cuda_blocks, uint32_t cuda_threads);

} // extern C

} // namespace hololink::operators

#endif // SRC_HOLOLINK_OPERATORS_GPU_ROCE_TRANSCEIVER_HPP
