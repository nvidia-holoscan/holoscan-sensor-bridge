/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

#include "gpu_roce_transceiver.hpp"

#define ACCESS_VOLATILE(x) (*(volatile typeof(x)*)&(x))

#ifndef ACCESS_ONCE_64b
#define ACCESS_ONCE_64b(x) (*(volatile uint64_t*)&(x))
#endif

#ifndef WRITE_ONCE_64b
#define WRITE_ONCE_64b(x, v) (ACCESS_ONCE_64b(x) = (v))
#endif

/* Period of the monitor thread checks while the GPU kernels run */
#define GPU_ROCE_MONITOR_POLL_MS 100

namespace hololink::operators {

GpuRoceTransceiver::GpuRoceTransceiver(const char* ibv_name, unsigned ibv_port,
    std::string tx_ibv_qp, uint32_t gpu_id,
    size_t cu_frame_size, size_t cu_page_size,
    unsigned pages, const char* peer_ip,
    const bool forward, const bool rx_only, const bool tx_only)
    : ibv_name_(strdup(ibv_name))
    , ibv_port_(ibv_port)
    , tx_ibv_qp_(tx_ibv_qp)
    , gpu_id_(gpu_id)
    , cu_frame_size_(cu_frame_size)
    , cu_page_size_(cu_page_size)
    , pages_(pages)
    , peer_ip_(strdup(peer_ip))
    , forward_(forward)
    , rx_only_(rx_only)
    , tx_only_(tx_only)
    , ready_mutex_(PTHREAD_MUTEX_INITIALIZER)
    , ready_condition_(PTHREAD_COND_INITIALIZER)
    , done_(false)
    , frame_number_(0)
    , rx_write_requests_fd_(-1)
    , rx_write_requests_(0)
    , received_ {}
    , imm_data_(0)
    , event_time_ {}
    , dropped_(0)
    , received_psn_(0)
    , received_page_(0)
{
    if (forward_ == true && (rx_only == true || tx_only == true))
        HSB_LOG_WARN("forward is set to true, so rx_only and tx_only will be ignored");

    int r = pthread_mutex_init(&ready_mutex_, nullptr);
    if (r != 0) {
        throw std::runtime_error("pthread_mutex_init failed.");
    }
    pthread_condattr_t pthread_condattr;
    pthread_condattr_init(&pthread_condattr);
    pthread_condattr_setclock(&pthread_condattr, CLOCK_MONOTONIC);
    r = pthread_cond_init(&ready_condition_, &pthread_condattr);
    pthread_condattr_destroy(&pthread_condattr);
    if (r != 0) {
        throw std::runtime_error("pthread_cond_init failed.");
    }
    int pipe_fds[2] = { -1, -1 };

    // If these aren't updated, we'll get an error when we try to read, which is good.
    r = pipe(pipe_fds);
    if (r != 0) {
        throw std::runtime_error("pipe call failed.");
    }
    control_r_ = pipe_fds[0];
    control_w_ = pipe_fds[1];

    char rx_write_requests_filename[8192];
    int written = snprintf(
        rx_write_requests_filename, sizeof(rx_write_requests_filename),
        "/sys/class/infiniband/%s/ports/%d/hw_counters/rx_write_requests",
        ibv_name_, ibv_port_);
    if (written < 0) {
        throw std::runtime_error("Error writing to rx_write_requests_filename.");
    } else if (((size_t)written) >= sizeof(rx_write_requests_filename)) {
        throw std::runtime_error(
            "Buffer isn't large enough to compute rx_write_requests filename.");
    } else {
        rx_write_requests_fd_ = open(rx_write_requests_filename, O_RDONLY);
    }
    if (rx_write_requests_fd_ < 0) {
        // Note that the rest of the code is OK if this occurs.
        HSB_LOG_ERROR("Unable to fetch rx_write_requests; ignoring.");
    }

    std::stringstream ss(tx_ibv_qp_);
    std::string item;

    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            size_t parsed = 0;
            unsigned long qp_num = 0;
            try {
                qp_num = std::stoul(item, &parsed);
            } catch (const std::exception&) {
                parsed = 0;
            }
            // QP numbers are 24 bits
            if (parsed == 0 || item.find_first_not_of(" \t", parsed) != std::string::npos || qp_num > 0xFFFFFF)
                throw std::runtime_error("tx_ibv_qp=\"" + tx_ibv_qp_ + "\": \"" + item + "\" is not a valid QP number");
            tx_ibv_qp_num_.push_back((uint32_t)qp_num);
        }
    }

    if (tx_ibv_qp_num_.empty())
        throw std::runtime_error("tx_ibv_qp must include at least one QP");

    if (tx_ibv_qp_num_.size() > GPU_ROCE_TRANSCEIVER_MAX_QP)
        throw std::runtime_error("Number of QPs (" + std::to_string(tx_ibv_qp_num_.size()) + ") is bigger than max allowed (" + std::to_string(GPU_ROCE_TRANSCEIVER_MAX_QP) + ")");
}

GpuRoceTransceiver::~GpuRoceTransceiver()
{
    // Stop the persistent kernels: they check the exit flag also while waiting for packets
    if (cpu_exit_flag)
        __atomic_store_n(&cpu_exit_flag[0], 1, __ATOMIC_RELEASE);

    if (forward_) {
        if (forward_stream) {
            cudaStreamSynchronize(forward_stream);
            cudaStreamDestroy(forward_stream);
        }
    } else {
        if (rx_only_ && rx_only_stream) {
            cudaStreamSynchronize(rx_only_stream);
            cudaStreamDestroy(rx_only_stream);
        }

        if (tx_only_ && tx_only_stream) {
            cudaStreamSynchronize(tx_only_stream);
            cudaStreamDestroy(tx_only_stream);
        }

#if DOCA_SIMULATE_RX_COMPUTE_TX == 1
        if (compute_stream) {
            cudaStreamSynchronize(compute_stream);
            cudaStreamDestroy(compute_stream);
        }
#endif
    }

    // The CPU proxy rings the doorbells for the kernels: stop it once they are done
    if (cpu_proxy_started_) {
        WRITE_ONCE_64b(*targs.exit_flag, 1);
        pthread_join(cpu_proxy_thread_id, NULL);
        cpu_proxy_started_ = false;
    }
    free(targs.exit_flag);
    targs.exit_flag = nullptr;

    if (gpu_exit_flag)
        doca_gpu_mem_free(doca_gpu_device_, gpu_exit_flag);

    // A QP references its CQs: destroy the QPs first
    for (const auto& qp : doca_qp_)
        delete qp;
    for (const auto& cq : doca_cq_rq_)
        delete cq;
    for (const auto& cq : doca_cq_sq_)
        delete cq;

    for (size_t x = 0; x < tx_ibv_qp_num_.size(); x++) {
        if (uar_[x] != nullptr) {
            doca_error_t result = doca_uar_destroy(uar_[x]);
            if (result != DOCA_SUCCESS) {
                HSB_LOG_ERROR("Failed to destroy UAR: {}", doca_error_get_descr(result));
            }
            uar_[x] = nullptr;
        }
    }

    if (doca_pd_ != nullptr) {
        doca_error_t result = doca_verbs_pd_destroy(doca_pd_);
        if (result != DOCA_SUCCESS) {
            HSB_LOG_ERROR("Failed to destroy PD: {}", doca_error_get_descr(result));
        }
        doca_pd_ = nullptr;
    }

    if (doca_verbs_ctx_ != nullptr) {
        doca_error_t result = doca_verbs_context_destroy(doca_verbs_ctx_);
        if (result != DOCA_SUCCESS) {
            HSB_LOG_ERROR("Failed to destroy verbs context: {}",
                doca_error_get_descr(result));
        }
        doca_verbs_ctx_ = nullptr;
    }

    if (doca_device_ != nullptr) {
        doca_error_t result = doca_dev_close(doca_device_);
        if (result != DOCA_SUCCESS) {
            HSB_LOG_ERROR("Failed to close DOCA device: {}",
                doca_error_get_descr(result));
        }
        doca_device_ = nullptr;
    }

    if (doca_gpu_device_ != nullptr)
        doca_gpu_destroy(doca_gpu_device_);

    HSB_LOG_TRACE("DOCA resources cleaned up");

    pthread_cond_destroy(&ready_condition_);
    pthread_mutex_destroy(&ready_mutex_);
    if (control_r_ >= 0)
        ::close(control_r_);
    if (control_w_ >= 0)
        ::close(control_w_);
    if (rx_write_requests_fd_ >= 0)
        ::close(rx_write_requests_fd_);
    free(ibv_name_);
    free(peer_ip_);
}

bool GpuRoceTransceiver::start()
{
    union ibv_gid rgid;
    uint32_t gid_index = 0;
    bool gid_found = false;
    struct ibv_gid_entry ib_gid_entry = { 0 };
    cudaError_t result_cuda;
    doca_error_t result;
    char gpu_bus_id[256];
    int num_devices = 0;
    int dmabuf_fd;
    struct cudaDeviceProp prop;

    HSB_LOG_DEBUG("Starting DOCA Verbs RoCE Receiver.");

    // Lock to ensure thread safety
    std::lock_guard lock(get_lock());

    // DOCA keeps every log backend created: create the SDK one only once per process
    static std::once_flag sdk_log_once;
    static doca_error_t sdk_log_result = DOCA_SUCCESS;
    std::call_once(sdk_log_once, []() {
        struct doca_log_backend* sdk_backend;
        sdk_log_result = doca_log_backend_create_with_file_sdk(stderr, &sdk_backend);
    });
    if (sdk_log_result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to create SDK log backend: {}", doca_error_get_descr(sdk_log_result));
        return false;
    }

    doca_log_level_set_global_sdk_limit(DOCA_LOG_LEVEL_ERROR);

    HSB_LOG_DEBUG("Opening DOCA device with name: {}", ibv_name_);
    doca_verbs_ctx_ = open_ib_device(ibv_name_);
    if (doca_verbs_ctx_ == nullptr) {
        HSB_LOG_ERROR("DOCA ctx NULL");
        return false;
    }

    HSB_LOG_INFO("DOCA device opened successfully");

    cudaFree(0);
    cudaSetDevice(gpu_id_);
    cuDeviceGet(&cuDevice, gpu_id_);
    cuCtxGetCurrent(&cuContext);

    result_cuda = cudaDeviceGetPCIBusId(gpu_bus_id, 256, gpu_id_);
    if (result_cuda != cudaSuccess) {
        HSB_LOG_ERROR("cudaDeviceGetPCIBusId returned error {}", (int)result_cuda);
        return false;
    }

    result_cuda = cudaGetDeviceProperties(&prop, gpu_id_);
    if (result_cuda != cudaSuccess) {
        HSB_LOG_ERROR("cudaGetDeviceProperties returned error {}", (int)result_cuda);
        return false;
    }

    umem_cpu = false;
    nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF;
    // DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB
    if (prop.integrated) {
        umem_cpu = true;
        nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY;
    }

    HSB_LOG_INFO("Device {} GPU type {}", gpu_id_, prop.integrated ? "iGPU" : "dGPU");

    result = doca_gpu_create(gpu_bus_id, &doca_gpu_device_);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to create GPU device: {}",
            doca_error_get_descr(result));
        return false;
    }

    HSB_LOG_INFO("Created GPU device successfully");

    result = doca_verbs_pd_create(doca_verbs_ctx_, &doca_pd_);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to create doca verbs pd: {}", doca_error_get_descr(result));
        return false;
    }

    ibv_pd = doca_verbs_bridge_verbs_pd_get_ibv_pd(doca_pd_);
    if (ibv_pd == NULL) {
        HSB_LOG_ERROR("Failed to get ibv_pd");
        return false;
    }

    result = doca_rdma_bridge_open_dev_from_pd(ibv_pd, &doca_device_);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to create doca verbs pd: {}", doca_error_get_descr(result));
        return false;
    }

    // Search for RoCE v2 GID (same logic as IB implementation)
    for (gid_index = 0;; gid_index++) { // Typical max GID index
        uint32_t flags = 0;
        // ibv_port_ - is given to constructor from the receiver operator parameter
        int ret = ibv_query_gid_ex(ibv_pd->context, ibv_port_, gid_index, &ib_gid_entry, flags);
        if (ret != 0 && errno != ENODATA) {
            break;
        }

        HSB_LOG_DEBUG(
            "gid_index={} gid_entry(gid_index={} port_num={} gid_type={} "
            "ndev_ifindex={} subnet_prefix={} interface_id={:#x})",
            gid_index, ib_gid_entry.gid_index, ib_gid_entry.port_num,
            ib_gid_entry.gid_type, ib_gid_entry.ndev_ifindex,
            ib_gid_entry.gid.global.subnet_prefix,
            ib_gid_entry.gid.global.interface_id);

        // Check for RoCE v2 GID characteristics
        if (ib_gid_entry.gid_type == IBV_GID_TYPE_ROCE_V2 && ib_gid_entry.gid.global.subnet_prefix == 0 && (ib_gid_entry.gid.global.interface_id & 0xFFFFFFFF) == 0xFFFF0000) {
            gid_found = true;
            HSB_LOG_INFO("Found RoCE v2 GID at index {}", gid_index);
            break;
        }
    }

    if (!gid_found) {
        HSB_LOG_ERROR("Cannot find GID for RoCE v2");
        return false;
    }

    // Configure remote address for RTR state
    unsigned long client_ip = 0;
    if (inet_pton(AF_INET, peer_ip_, &client_ip) != 1) {
        HSB_LOG_ERROR("Unable to convert \"{}\" to an IP address.", peer_ip_);
        // doca_verbs_qp_attr_destroy(qp_attr);
        return false;
    }

    struct doca_verbs_gid doca_rgid; /* remote gid address */
    if (client_ip != 0) {
        uint64_t client_interface_id = client_ip;
        client_interface_id <<= 32;
        client_interface_id |= 0xFFFF0000;
        union ibv_gid remote_gid = {
            .global = {
                .subnet_prefix = 0,
                .interface_id = client_interface_id,
            },
        };
        memcpy(doca_rgid.raw, remote_gid.raw, 16);
    } else {
        HSB_LOG_ERROR("Failed to get client_ip");
        return false;
    }

    /* **************** multi-QP creation **************** */
    for (int x = 0; x < tx_ibv_qp_num_.size(); x++) {
        if (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF)
            result = doca_uar_create(doca_device_, DOCA_UAR_ALLOCATION_TYPE_BLUEFLAME, &uar_[x]);
        else
            result = doca_uar_create(doca_device_, DOCA_UAR_ALLOCATION_TYPE_NONCACHE, &uar_[x]);

        if (result != DOCA_SUCCESS) {
            HSB_LOG_ERROR("Failed to create UAR: {}", doca_error_get_descr(result));
            return false;
        }

        doca_cq_rq_[x] = new DocaCq(WQE_NUM, doca_gpu_device_, doca_device_, uar_[x], doca_verbs_ctx_, umem_cpu);
        result = doca_cq_rq_[x]->create();
        if (result != DOCA_SUCCESS) {
            HSB_LOG_ERROR("Failed to create CQ RQ: {}", doca_error_get_descr(result));
            return false;
        }

        doca_cq_sq_[x] = new DocaCq(WQE_NUM, doca_gpu_device_, doca_device_, uar_[x], doca_verbs_ctx_, umem_cpu);
        result = doca_cq_sq_[x]->create();
        if (result != DOCA_SUCCESS) {
            HSB_LOG_ERROR("Failed to create CQ SQ: {}", doca_error_get_descr(result));
            return false;
        }

        HSB_LOG_INFO("Created CQ successfully");

        doca_qp_[x] = new DocaQp(WQE_NUM, doca_gpu_device_, doca_device_, uar_[x], doca_verbs_ctx_, doca_pd_, doca_cq_rq_[x]->get(), doca_cq_sq_[x]->get(), umem_cpu);

        result = doca_qp_[x]->create(doca_verbs_ctx_, cu_frame_size_, nic_handler);
        if (result != DOCA_SUCCESS) {
            HSB_LOG_ERROR("Failed to create QP: {}", doca_error_get_descr(result));
            return false;
        }

        // cpu_ring_buffers_: force CPU_GPU allocation for ring flags and data only,
        // without changing CQ/QP UMEMs or TX kernel handler mode.
        if (cpu_ring_buffers_ && !umem_cpu)
            doca_qp_[x]->umem_cpu = true;

        result = doca_qp_[x]->create_ring(cu_page_size_, pages_, ibv_pd);
        if (result != DOCA_SUCCESS) {
            HSB_LOG_ERROR("Failed to create ring buffers: {}", doca_error_get_descr(result));
            return false;
        }

        // Restore original umem_cpu so TX kernel handler uses BlueFlame on dGPU.
        if (cpu_ring_buffers_ && !umem_cpu)
            doca_qp_[x]->umem_cpu = false;

        HSB_LOG_INFO("Created QP with number: {:#x} Rx ring buffer Rkey {}", doca_verbs_qp_get_qpn(doca_qp_[x]->get()), doca_qp_[x]->gpu_rx_ring.addr_mr->rkey);

        // dest_qp_num = 0x2, FPGA specific port
        result = doca_qp_[x]->connect(doca_rgid, gid_index, tx_ibv_qp_num_[x]);
        if (result != DOCA_SUCCESS) {
            HSB_LOG_ERROR("Failed to connect Tx QP: {}", doca_error_get_descr(result));
            return false;
        }

        HSB_LOG_INFO("Connected QP {} {}", x, tx_ibv_qp_num_[x]);

        // Kernel to pre-post all recv
        if (forward_)
            GpuRoceTransceiverPrepareKernel(0, doca_qp_[x]->get_gpu_dev(), cu_frame_size_, htobe32(doca_qp_[x]->gpu_rx_ring.addr_mr->rkey), 1, WQE_NUM);
        else if (rx_only_ == true || tx_only_ == true)
            GpuRoceTransceiverPrepareKernel(0, doca_qp_[x]->get_gpu_dev(), cu_frame_size_, htobe32(doca_qp_[x]->gpu_tx_ring.addr_mr->rkey), 1, WQE_NUM);

        HSB_LOG_INFO("GpuRoceTransceiverPrepareKernel done {} {}", x, tx_ibv_qp_num_[x]);

        result_cuda = cudaStreamSynchronize(0);
        if (result_cuda != cudaSuccess) {
            HSB_LOG_ERROR("Failed to cudaStreamSynchronize for QP {} : {}", x, (int)result_cuda);
            return false;
        }
    }
    /* ********************************************* */

    /* **************** CUDA kernel warmup **************** */
    struct doca_gpu_dev_verbs_qp* qp[GPU_ROCE_TRANSCEIVER_MAX_QP] = { 0 };
    uint8_t* ring_rx_buf[GPU_ROCE_TRANSCEIVER_MAX_QP] = { 0 };
    size_t ring_rx_buf_stride_sz[GPU_ROCE_TRANSCEIVER_MAX_QP] = { 0 };
    uint32_t ring_rx_buf_mkey[GPU_ROCE_TRANSCEIVER_MAX_QP] = { 0 };
    uint32_t ring_rx_buf_stride_num[GPU_ROCE_TRANSCEIVER_MAX_QP] = { 0 };
    uint64_t* ring_rx_flag[GPU_ROCE_TRANSCEIVER_MAX_QP] = { 0 };

    uint8_t* ring_tx_buf[GPU_ROCE_TRANSCEIVER_MAX_QP] = { 0 };
    size_t ring_tx_buf_stride_sz[GPU_ROCE_TRANSCEIVER_MAX_QP] = { 0 };
    uint32_t ring_tx_buf_mkey[GPU_ROCE_TRANSCEIVER_MAX_QP] = { 0 };
    uint32_t ring_tx_buf_stride_num[GPU_ROCE_TRANSCEIVER_MAX_QP] = { 0 };
    uint64_t* ring_tx_flag[GPU_ROCE_TRANSCEIVER_MAX_QP] = { 0 };

    if (umem_cpu) {
        result = doca_gpu_mem_alloc(doca_gpu_device_,
            sizeof(uint32_t),
            get_page_size(),
            DOCA_GPU_MEM_TYPE_CPU_GPU,
            (void**)&gpu_exit_flag,
            (void**)&cpu_exit_flag);
    } else {
        result = doca_gpu_mem_alloc(doca_gpu_device_,
            sizeof(uint32_t),
            get_page_size(),
            DOCA_GPU_MEM_TYPE_GPU_CPU,
            (void**)&gpu_exit_flag,
            (void**)&cpu_exit_flag);
    }

    if (result != DOCA_SUCCESS || gpu_exit_flag == nullptr || cpu_exit_flag == nullptr) {
        HSB_LOG_ERROR("Failed to alloc flag exit: {}", doca_error_get_descr(result));
        return false;
    }

    if (forward_) {
        result_cuda = cudaStreamCreateWithFlags(&forward_stream, cudaStreamNonBlocking);
        if (result_cuda != cudaSuccess) {
            HSB_LOG_ERROR("Failed to create new CUDA stream: {}", (int)result_cuda);
            return false;
        }

        GpuRoceTransceiverForwardKernel(forward_stream, gpu_exit_flag,
            qp, ring_rx_buf, ring_rx_buf_stride_sz, ring_rx_buf_mkey, ring_rx_buf_stride_num,
            cu_frame_size_, nic_handler, (uint32_t)tx_ibv_qp_num_.size(), WQE_NUM);
    } else {
        int leastPriority, greatestPriority;
        result_cuda = cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
        if (result_cuda != cudaSuccess) {
            HSB_LOG_ERROR("Failed to get CUDA stream priorities: {}", (int)result_cuda);
            return false;
        }

        if (rx_only_) {
            result_cuda = cudaStreamCreateWithPriority(&rx_only_stream, cudaStreamNonBlocking, greatestPriority);
            if (result_cuda != cudaSuccess) {
                HSB_LOG_ERROR("Failed to create new CUDA stream: {}", (int)result_cuda);
                return false;
            }

            // Warmup
            GpuRoceTransceiverRxOnlyKernel(rx_only_stream, gpu_exit_flag,
                qp, ring_rx_buf, ring_rx_buf_stride_sz, ring_rx_buf_stride_num, ring_rx_flag,
                cu_frame_size_, (uint32_t)tx_ibv_qp_num_.size(), WQE_NUM);
        }

        if (tx_only_) {
            result_cuda = cudaStreamCreateWithPriority(&tx_only_stream, cudaStreamNonBlocking, greatestPriority);
            if (result_cuda != cudaSuccess) {
                HSB_LOG_ERROR("Failed to create new CUDA stream: {}", (int)result_cuda);
                return false;
            }
            // Warmup
            GpuRoceTransceiverTxOnlyKernel(tx_only_stream, gpu_exit_flag,
                qp, ring_tx_buf, ring_tx_buf_stride_sz, ring_tx_buf_stride_num, ring_tx_buf_mkey, ring_tx_flag,
                cu_frame_size_, nic_handler, (uint32_t)tx_ibv_qp_num_.size(), WQE_NUM);
        }

#if DOCA_SIMULATE_RX_COMPUTE_TX == 1
        result_cuda = cudaStreamCreateWithPriority(&compute_stream, cudaStreamNonBlocking, greatestPriority);
        if (result_cuda != cudaSuccess) {
            HSB_LOG_ERROR("Failed to create new CUDA stream: {}", (int)result_cuda);
            return false;
        }
        // Warmup
        GpuRoceTransceiverComputeKernel(compute_stream, gpu_exit_flag,
            ring_rx_flag, ring_rx_buf_stride_num,
            ring_tx_flag, ring_tx_buf, ring_tx_buf_stride_sz, ring_tx_buf_stride_num,
            cu_frame_size_, (uint32_t)tx_ibv_qp_num_.size(), WQE_NUM);
#endif
    }
    /* ********************************************* */

    HSB_LOG_INFO("DOCA Verbs RoCE Receiver started successfully");

    return true;
}

void GpuRoceTransceiver::close()
{
    done_ = true;
    // Tell GPU kernel to exit. Release: the monitor thread seeing the exit flag also sees done_.
    if (cpu_exit_flag) {
        __atomic_store_n(&cpu_exit_flag[0], 1, __ATOMIC_RELEASE);
    }
    // Wake any thread blocked in get_next_frame(wait-until-close)
    int status = pthread_mutex_lock(&ready_mutex_);
    if (status == 0) {
        pthread_cond_broadcast(&ready_condition_);
        pthread_mutex_unlock(&ready_mutex_);
    }
    // Wake poll by writing to control pipe
    if (control_w_ >= 0) {
        const uint8_t byte = 0xFF;
        ssize_t written = write(control_w_, &byte, 1);
        if (written != 1) {
            HSB_LOG_WARN("write() to control pipe returned {}", written);
        }
        ::close(control_w_);
        control_w_ = -1;
    }
}

void* progress_cpu_proxy(void* args_)
{
    struct cpu_proxy_args* args = (struct cpu_proxy_args*)args_;

    printf("Thread CPU proxy progress is running... %ld\n", ACCESS_ONCE_64b(*args->exit_flag));

    while (ACCESS_ONCE_64b(*args->exit_flag) == 0) {
        for (int x = 0; x < args->num_qp; x++)
            doca_gpu_verbs_cpu_proxy_progress(args->qp_cpu[x]);
    }

    return NULL;
}

bool GpuRoceTransceiver::launch_kernels()
{
    struct doca_gpu_dev_verbs_qp* qp[GPU_ROCE_TRANSCEIVER_MAX_QP];
    uint8_t* ring_rx_buf[GPU_ROCE_TRANSCEIVER_MAX_QP];
    size_t ring_rx_buf_stride_sz[GPU_ROCE_TRANSCEIVER_MAX_QP];
    uint32_t ring_rx_buf_mkey[GPU_ROCE_TRANSCEIVER_MAX_QP];
    uint32_t ring_rx_buf_stride_num[GPU_ROCE_TRANSCEIVER_MAX_QP];
    uint64_t* ring_rx_flag[GPU_ROCE_TRANSCEIVER_MAX_QP];

    uint8_t* ring_tx_buf[GPU_ROCE_TRANSCEIVER_MAX_QP];
    size_t ring_tx_buf_stride_sz[GPU_ROCE_TRANSCEIVER_MAX_QP];
    uint32_t ring_tx_buf_mkey[GPU_ROCE_TRANSCEIVER_MAX_QP];
    uint32_t ring_tx_buf_stride_num[GPU_ROCE_TRANSCEIVER_MAX_QP];
    uint64_t* ring_tx_flag[GPU_ROCE_TRANSCEIVER_MAX_QP];
    doca_error_t result = DOCA_SUCCESS;

    __atomic_store_n(&cpu_exit_flag[0], 0, __ATOMIC_RELEASE);
    for (unsigned x = 0; x < GPU_ROCE_TRANSCEIVER_MAX_QP; x++) {
        if (x < tx_ibv_qp_num_.size()) {
            HSB_LOG_INFO("QP {} preparation", x);
            qp[x] = doca_qp_[x]->get_gpu_dev();

            ring_rx_buf[x] = (uint8_t*)doca_qp_[x]->gpu_rx_ring.addr;
            ring_rx_buf_stride_sz[x] = doca_qp_[x]->gpu_rx_ring.stride_sz;
            ring_rx_buf_mkey[x] = htobe32(doca_qp_[x]->gpu_rx_ring.addr_mr->rkey);
            ring_rx_buf_stride_num[x] = doca_qp_[x]->gpu_rx_ring.stride_num;
            ring_rx_flag[x] = doca_qp_[x]->gpu_rx_ring.flag;

            ring_tx_buf[x] = (uint8_t*)doca_qp_[x]->gpu_tx_ring.addr;
            ring_tx_buf_stride_sz[x] = doca_qp_[x]->gpu_tx_ring.stride_sz;
            ring_tx_buf_mkey[x] = htobe32(doca_qp_[x]->gpu_tx_ring.addr_mr->rkey);
            ring_tx_buf_stride_num[x] = doca_qp_[x]->gpu_tx_ring.stride_num;
            ring_tx_flag[x] = doca_qp_[x]->gpu_tx_ring.flag;
        } else {
            qp[x] = nullptr;

            ring_rx_buf[x] = nullptr;
            ring_rx_buf_stride_sz[x] = 0;
            ring_rx_buf_mkey[x] = 0;
            ring_rx_buf_stride_num[x] = 0;
            ring_rx_flag[x] = nullptr;

            ring_tx_buf[x] = nullptr;
            ring_tx_buf_stride_sz[x] = 0;
            ring_tx_buf_mkey[x] = 0;
            ring_tx_buf_stride_num[x] = 0;
            ring_tx_flag[x] = nullptr;
        }
    }

    if (forward_) {
        result = GpuRoceTransceiverForwardKernel(forward_stream, gpu_exit_flag,
            qp, ring_rx_buf, ring_rx_buf_stride_sz, ring_rx_buf_mkey, ring_rx_buf_stride_num,
            cu_frame_size_, nic_handler, (uint32_t)tx_ibv_qp_num_.size(), WQE_NUM);
    } else {
        if (rx_only_) {
            result = GpuRoceTransceiverRxOnlyKernel(rx_only_stream, gpu_exit_flag,
                qp, ring_rx_buf, ring_rx_buf_stride_sz, ring_rx_buf_stride_num, ring_rx_flag,
                cu_frame_size_, (uint32_t)tx_ibv_qp_num_.size(), WQE_NUM);
        }

        if (result == DOCA_SUCCESS && tx_only_) {
            result = GpuRoceTransceiverTxOnlyKernel(tx_only_stream, gpu_exit_flag,
                qp, ring_tx_buf, ring_tx_buf_stride_sz, ring_tx_buf_stride_num, ring_tx_buf_mkey, ring_tx_flag,
                cu_frame_size_, nic_handler, (uint32_t)tx_ibv_qp_num_.size(), WQE_NUM);
        }

#if DOCA_SIMULATE_RX_COMPUTE_TX == 1
        if (result == DOCA_SUCCESS) {
            result = GpuRoceTransceiverComputeKernel(compute_stream, gpu_exit_flag,
                ring_rx_flag, ring_rx_buf_stride_num,
                ring_tx_flag, ring_tx_buf, ring_tx_buf_stride_sz, ring_tx_buf_stride_num,
                cu_frame_size_, (uint32_t)tx_ibv_qp_num_.size(), WQE_NUM);
        }
#endif
    }

    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to launch the GPU kernels: {}", doca_error_get_descr(result));
        // Stop the kernels already launched
        __atomic_store_n(&cpu_exit_flag[0], 1, __ATOMIC_RELEASE);
        return false;
    }

    // Should it spawns 1 CPU thread per QP?
    if (umem_cpu) {
        for (unsigned x = 0; x < GPU_ROCE_TRANSCEIVER_MAX_QP; x++) {
            if (x < tx_ibv_qp_num_.size())
                targs.qp_cpu[x] = doca_qp_[x]->get_gpu();
            else
                targs.qp_cpu[x] = nullptr;
        }

        targs.num_qp = (uint16_t)tx_ibv_qp_num_.size();
        targs.exit_flag = (uint64_t*)calloc(1, sizeof(uint64_t));
        if (targs.exit_flag == nullptr) {
            HSB_LOG_ERROR("Failed to allocate the CPU proxy exit flag");
            __atomic_store_n(&cpu_exit_flag[0], 1, __ATOMIC_RELEASE);
            return false;
        }

        int ret = pthread_create(&cpu_proxy_thread_id, NULL, progress_cpu_proxy, (void*)&targs);
        if (ret != 0) {
            HSB_LOG_ERROR("Failed to create the CPU proxy thread: {}", ret);
            // Without the CPU proxy the kernels can't send: stop them
            __atomic_store_n(&cpu_exit_flag[0], 1, __ATOMIC_RELEASE);
            return false;
        }
        cpu_proxy_started_ = true;
    }

    doca_kernel_launched = true;
    return true;
}

void GpuRoceTransceiver::blocking_monitor()
{
    struct pollfd poll_fds[1];
    bool gpu_error_reported = false;
    int ret;

    core::NvtxTrace::setThreadName("GpuRoceTransceiver::run");
    HSB_LOG_INFO("DOCA Blocking Monitor.");

    poll_fds[0] = {
        .fd = control_r_,
        .events = POLLIN | POLLHUP | POLLERR,
    };

    cudaFree(0);
    cudaSetDevice(gpu_id_);
    cuCtxPushCurrent(cuContext);

    if (!doca_kernel_launched && !done_ && !launch_kernels()) {
        cuCtxPopCurrent(&cuContext);
        return;
    }

    // The kernels own the datapath: wait for close(), checking periodically if the kernels stopped on error
    while (!done_) {
        ret = poll(poll_fds, 1, GPU_ROCE_MONITOR_POLL_MS);
        if (ret == -1) {
            if (errno == EINTR)
                continue;
            throw std::runtime_error(fmt::format("poll returned ret={}, errno={}.", ret, errno));
        }

        if (ret > 0 && poll_fds[0].revents) {
            // Currently, the only activity that we see on control_r_ is a flag
            // telling us that someone closed the control_w_ side (which we do
            // in close()).  That specific event is an indication
            // that this loop is instructed to terminate.
            HSB_LOG_DEBUG("Closing.");
            break;
        }

        // Unless close() set it, the exit flag is set by the kernels after an error completion
        if (!gpu_error_reported && __atomic_load_n(&cpu_exit_flag[0], __ATOMIC_ACQUIRE) != 0 && !done_) {
            HSB_LOG_ERROR("GPU RoCE kernels stopped after an error completion");
            gpu_error_reported = true;
        }
    }

    cuCtxPopCurrent(&cuContext);
    HSB_LOG_DEBUG("Closed.");
}

bool GpuRoceTransceiver::check_async_events()
{
    HSB_LOG_INFO("Checking async events");
    unsigned i;
    struct doca_verbs_async_event async_event;
    for (i = 0; i < 100; i++) {
        HSB_LOG_INFO("Checking async event {}", i);
        doca_error_t result = doca_verbs_get_async_event(doca_verbs_ctx_, &async_event);
        HSB_LOG_INFO("Got async event {}", i);
        if (result != DOCA_SUCCESS) {
            HSB_LOG_ERROR("Failed to get async event: {}",
                doca_error_get_descr(result));
            break;
        }

        auto event_type = async_event.event_type;
        // log the event
        HSB_LOG_INFO("DOCA got async event: {}", (int)async_event.event_type);

        // acknowledge the event
        doca_verbs_ack_async_event(&async_event);
    }
    HSB_LOG_INFO("Checked async events FINISHED");
    return true;
}

uint32_t GpuRoceTransceiver::get_qp_number(unsigned sensor_id)
{
    if (qp_created(sensor_id))
        return doca_verbs_qp_get_qpn(doca_qp_[sensor_id]->get());
    return 0;
}

uint32_t GpuRoceTransceiver::get_rkey(unsigned sensor_id)
{
    if (qp_created(sensor_id) && doca_qp_[sensor_id]->gpu_rx_ring.addr_mr)
        return doca_qp_[sensor_id]->gpu_rx_ring.addr_mr->rkey;
    return 0;
}

uint64_t GpuRoceTransceiver::external_frame_memory()
{
    // Using iova option with dmabuf, so frame memory starts at 0
    return 0;
}

uint8_t* GpuRoceTransceiver::get_rx_ring_data_addr(unsigned sensor_id)
{
    if (!qp_created(sensor_id))
        return nullptr;

    return doca_qp_[sensor_id]->gpu_rx_ring.addr;
}

size_t GpuRoceTransceiver::get_rx_ring_stride_sz(unsigned sensor_id)
{
    if (!qp_created(sensor_id))
        return 0;

    return doca_qp_[sensor_id]->gpu_rx_ring.stride_sz;
}

uint32_t GpuRoceTransceiver::get_rx_ring_stride_num(unsigned sensor_id)
{
    if (!qp_created(sensor_id))
        return 0;

    return doca_qp_[sensor_id]->gpu_rx_ring.stride_num;
}

uint64_t* GpuRoceTransceiver::get_rx_ring_flag_addr(unsigned sensor_id)
{
    if (!qp_created(sensor_id))
        return nullptr;

    return doca_qp_[sensor_id]->gpu_rx_ring.flag;
}

uint8_t* GpuRoceTransceiver::get_tx_ring_data_addr(unsigned sensor_id)
{
    if (!qp_created(sensor_id))
        return nullptr;

    return doca_qp_[sensor_id]->gpu_tx_ring.addr;
}

size_t GpuRoceTransceiver::get_tx_ring_stride_sz(unsigned sensor_id)
{
    if (!qp_created(sensor_id))
        return 0;

    return doca_qp_[sensor_id]->gpu_tx_ring.stride_sz;
}

uint32_t GpuRoceTransceiver::get_tx_ring_stride_num(unsigned sensor_id)
{
    if (!qp_created(sensor_id))
        return 0;

    return doca_qp_[sensor_id]->gpu_tx_ring.stride_num;
}

uint64_t* GpuRoceTransceiver::get_tx_ring_flag_addr(unsigned sensor_id)
{
    if (!qp_created(sensor_id))
        return nullptr;

    return doca_qp_[sensor_id]->gpu_tx_ring.flag;
}

uint32_t GpuRoceTransceiver::get_tx_ring_lkey(unsigned sensor_id)
{
    return doca_qp_[sensor_id]->gpu_tx_ring.addr_mr->lkey;
}

std::mutex& GpuRoceTransceiver::get_lock()
{
    static std::mutex instance;
    return instance;
}

bool GpuRoceTransceiver::get_next_frame(unsigned timeout_ms, CUstream cuda_stream)
{
    (void)timeout_ms;
    (void)cuda_stream;
    int status = pthread_mutex_lock(&ready_mutex_);
    if (status != 0) {
        HSB_LOG_ERROR("pthread_mutex_lock returned status={}.", status);
        return false;
    }
    while (!done_) {
        status = pthread_cond_wait(&ready_condition_, &ready_mutex_);
        if (status != 0) {
            HSB_LOG_ERROR("pthread_cond_wait returned status={}.", status);
            break;
        }
    }
    pthread_mutex_unlock(&ready_mutex_);
    return false;
}

} // namespace hololink::operators
