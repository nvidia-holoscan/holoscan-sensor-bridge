
#include "gpu_roce_transceiver.hpp"

#define ACCESS_VOLATILE(x) (*(volatile typeof(x)*)&(x))

#ifndef ACCESS_ONCE_64b
#define ACCESS_ONCE_64b(x) (*(volatile uint64_t*)&(x))
#endif

#ifndef WRITE_ONCE_64b
#define WRITE_ONCE_64b(x, v) (ACCESS_ONCE_64b(x) = (v))
#endif

namespace hololink::operators {

GpuRoceTransceiver::GpuRoceTransceiver(const char* ibv_name, unsigned ibv_port,
    uint32_t gpu_id,
    size_t cu_frame_size, size_t cu_page_size,
    unsigned pages, const char* peer_ip,
    const bool forward, const bool rx_only, const bool tx_only)
    : ibv_name_(strdup(ibv_name))
    , ibv_port_(ibv_port)
    , gpu_id_(gpu_id)
    , cu_frame_size_(cu_frame_size)
    , cu_page_size_(cu_page_size)
    , pages_(pages)
    , peer_ip_(strdup(peer_ip))
    , forward_(forward)
    , rx_only_(rx_only)
    , tx_only_(tx_only)
    , ready_(false)
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
}

GpuRoceTransceiver::~GpuRoceTransceiver()
{
    if (cpu_exit_flag)
        cpu_exit_flag[0] = 1;

    if (forward_ && forward_stream) {
        cudaStreamSynchronize(forward_stream);
        cudaStreamDestroy(forward_stream);
    } else {
        if (rx_only_ && rx_only_stream) {
            cudaStreamSynchronize(rx_only_stream);
            cudaStreamDestroy(rx_only_stream);
        }

        if (tx_only_ && tx_only_stream) {
            cudaStreamSynchronize(tx_only_stream);
            cudaStreamDestroy(tx_only_stream);
        }
    }

    if (gpu_exit_flag)
        doca_gpu_mem_free(doca_gpu_device_, gpu_exit_flag);

#if DOCA_ENABLE_NON_RDMA == 1
    if (targs.exit_flag) {
        WRITE_ONCE_64b(*targs.exit_flag, 1);
        pthread_join(cpu_proxy_thread_id, NULL);
        free(targs.exit_flag);
    }
#endif

    delete doca_cq_rq;
    delete doca_cq_sq;
    delete doca_qp;

    doca_gpu_destroy(doca_gpu_device_);

    if (uar_ != nullptr) {
        doca_error_t result = doca_uar_destroy(uar_);
        if (result != DOCA_SUCCESS) {
            HSB_LOG_ERROR("Failed to destroy UAR: {}", doca_error_get_descr(result));
        }
        uar_ = nullptr;
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

    HSB_LOG_TRACE("DOCA resources cleaned up");

    pthread_cond_destroy(&ready_condition_);
    pthread_mutex_destroy(&ready_mutex_);
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
    struct doca_log_backend* sdk_backend;
    char gpu_bus_id[256];
    int num_devices = 0;
    int dmabuf_fd;
    struct cudaDeviceProp prop;

    HSB_LOG_DEBUG("Starting DOCA Verbs RoCE Receiver.");

    // Lock to ensure thread safety
    std::lock_guard lock(get_lock());

    struct ibv_device** ib_devices = ibv_get_device_list(&num_devices);
    if (!ib_devices) {
        HSB_LOG_ERROR("ibv_get_device_list failed; errno={}.", errno);
        return false;
    }

    if (num_devices < 0) {
        HSB_LOG_ERROR(
            "ibv_get_device_list set unexpected value for num_devices={}.",
            num_devices);
        return false;
    }

    result = doca_log_backend_create_with_file_sdk(stderr, &sdk_backend);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to create SDK log backend: {}", doca_error_get_descr(result));
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
    if (prop.integrated)
        umem_cpu = true;
    HSB_LOG_INFO("Device {} GPU type {}", gpu_id_, prop.integrated ? "iGPU" : "dGPU");

#if DOCA_ENABLE_NON_RDMA == 1
    umem_cpu = true;
#endif

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

    result = doca_uar_create(doca_device_, DOCA_UAR_ALLOCATION_TYPE_NONCACHE, &uar_);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to create UAR: {}", doca_error_get_descr(result));
        return false;
    }

    doca_cq_rq = new DocaCq(WQE_NUM, doca_gpu_device_, doca_device_, uar_, doca_verbs_ctx_, umem_cpu);
    result = doca_cq_rq->create();
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to create CQ RQ: {}", doca_error_get_descr(result));
        return false;
    }

    doca_cq_sq = new DocaCq(WQE_NUM, doca_gpu_device_, doca_device_, uar_, doca_verbs_ctx_, umem_cpu);
    result = doca_cq_sq->create();
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to create CQ SQ: {}", doca_error_get_descr(result));
        return false;
    }

    HSB_LOG_INFO("Created CQ successfully");

    doca_qp = new DocaQp(WQE_NUM, doca_gpu_device_, doca_device_, uar_, doca_verbs_ctx_, doca_pd_, doca_cq_rq->get(), doca_cq_sq->get(), umem_cpu);

    result = doca_qp->create(doca_verbs_ctx_, cu_frame_size_);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to create CQ RQ: {}", doca_error_get_descr(result));
        return false;
    }

    result = doca_qp->create_ring(cu_page_size_, pages_, ibv_pd);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to create CQ RQ: {}", doca_error_get_descr(result));
        return false;
    }

    // Ger RX buffer rkey
    rkey_ = doca_qp->gpu_rx_ring.addr_mr->rkey;
    // Get QP number
    qp_number_ = doca_verbs_qp_get_qpn(doca_qp->get());

    HSB_LOG_INFO("Created QP with number: {:#x} Rx ring buffer Rkey {}", qp_number_, rkey_);

    // Configure remote address for RTR state
    unsigned long client_ip = 0;
    if (inet_pton(AF_INET, peer_ip_, &client_ip) != 1) {
        HSB_LOG_ERROR("Unable to convert \"{}\" to an IP address.", peer_ip_);
        // doca_verbs_qp_attr_destroy(qp_attr);
        return false;
    }

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
        struct doca_verbs_gid doca_rgid; /* remote gid address */
        memcpy(doca_rgid.raw, remote_gid.raw, 16);

        // dest_qp_num = 0x2, FPGA specific port
        result = doca_qp->connect(doca_rgid, gid_index, 0x2);
        if (result != DOCA_SUCCESS) {
            HSB_LOG_ERROR("Failed to create CQ RQ: {}", doca_error_get_descr(result));
            return false;
        }
    } else {
        HSB_LOG_ERROR("Failed to get client_ip");
        return false;
    }

    // Kernel to pre-post all recv
    if (forward_)
        GpuRoceTransceiverPrepareKernel(0, doca_qp->get_gpu_dev(), cu_frame_size_, htobe32(doca_qp->gpu_rx_ring.addr_mr->rkey), 1, WQE_NUM);
    else if (rx_only_ == true || tx_only_ == true)
        GpuRoceTransceiverPrepareKernel(0, doca_qp->get_gpu_dev(), cu_frame_size_, htobe32(doca_qp->gpu_tx_ring.addr_mr->rkey), 1, WQE_NUM);
    cudaStreamSynchronize(0);

    if (forward_) {
        result_cuda = cudaStreamCreateWithFlags(&forward_stream, cudaStreamNonBlocking);
        if (result_cuda != cudaSuccess) {
            HSB_LOG_ERROR("Failed to create new CUDA stream: {}", (int)result_cuda);
            return false;
        }
    } else {
        if (rx_only_) {
            result_cuda = cudaStreamCreateWithFlags(&rx_only_stream, cudaStreamNonBlocking);
            if (result_cuda != cudaSuccess) {
                HSB_LOG_ERROR("Failed to create new CUDA stream: {}", (int)result_cuda);
                return false;
            }
        }

        if (tx_only_) {
            result_cuda = cudaStreamCreateWithFlags(&tx_only_stream, cudaStreamNonBlocking);
            if (result_cuda != cudaSuccess) {
                HSB_LOG_ERROR("Failed to create new CUDA stream: {}", (int)result_cuda);
                return false;
            }
        }
    }

    result = doca_gpu_mem_alloc(doca_gpu_device_,
        sizeof(uint32_t),
        get_page_size(),
        DOCA_GPU_MEM_TYPE_GPU_CPU,
        (void**)&gpu_exit_flag,
        (void**)&cpu_exit_flag);
    if (result != DOCA_SUCCESS || gpu_exit_flag == nullptr || cpu_exit_flag == nullptr) {
        HSB_LOG_ERROR("Failed to alloc flag exit: {}", doca_error_get_descr(result));
        return false;
    }

    HSB_LOG_INFO("DOCA Verbs RoCE Receiver started successfully");
    return true;
}

void GpuRoceTransceiver::close()
{
    done_ = true;
    // Tell GPU kernel to exit
    if (cpu_exit_flag) {
        cpu_exit_flag[0] = 1;
    }
    // Wake any waiter on ready_condition_
    int status = pthread_mutex_lock(&ready_mutex_);
    if (status == 0) {
        // Mark ready_ so any waiter can proceed and higher layers can transition
        ready_ = true;
        pthread_cond_signal(&ready_condition_);
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

    while (ACCESS_ONCE_64b(*args->exit_flag) == 0)
        doca_gpu_verbs_cpu_proxy_progress(args->qp_cpu);

    return NULL;
}

void GpuRoceTransceiver::blocking_monitor()
{
    struct pollfd poll_fds[1];
    int ret, timeout = 0; // no blocking poll

    core::NvtxTrace::setThreadName("GpuRoceTransceiver::run");
    HSB_LOG_INFO("DOCA Blocking Monitor.");

    poll_fds[0] = {
        .fd = control_r_,
        .events = POLLIN | POLLHUP | POLLERR,
    };

    while (!done_) {

        // HSB_LOG_INFO("Polling for CQ events");
        timeout = 0;
        ret = poll(poll_fds, 1, timeout);
        if (ret == -1)
            throw std::runtime_error(fmt::format("poll returned ret={}, errno={}.", ret, errno));

        /* Capture timestamp as close as possible to completion arrival */
        clock_gettime(CLOCK_REALTIME, const_cast<struct timespec*>(&event_time_));

        if (poll_fds[0].revents) {
            // Currently, the only activity that we see on control_r_ is a flag
            // telling us that someone closed the control_w_ side (which we do
            // in LinuxReceiver::close).  That specific event is an indication
            // that this loop is instructed to terminate.
            HSB_LOG_DEBUG("Closing.");
            break;
        }

        if (!doca_kernel_launched) {
            cpu_exit_flag[0] = 0;
            if (forward_) {
                GpuRoceTransceiverForwardKernel(forward_stream, doca_qp->get_gpu_dev(), gpu_exit_flag,
                    (uint8_t*)doca_qp->gpu_rx_ring.addr, doca_qp->gpu_rx_ring.stride_sz,
                    htobe32(doca_qp->gpu_rx_ring.addr_mr->rkey), doca_qp->gpu_rx_ring.stride_num,
                    cu_frame_size_, 1, WQE_NUM);
            } else {
                if (rx_only_) {
                    GpuRoceTransceiverRxOnlyKernel(rx_only_stream, doca_qp->get_gpu_dev(), gpu_exit_flag,
                        (uint8_t*)doca_qp->gpu_rx_ring.addr, doca_qp->gpu_rx_ring.stride_sz,
                        doca_qp->gpu_rx_ring.stride_num, doca_qp->gpu_rx_ring.flag,
                        cu_frame_size_, 1, WQE_NUM);
                }

                if (tx_only_) {
                    GpuRoceTransceiverTxOnlyKernel(rx_only_stream, doca_qp->get_gpu_dev(), gpu_exit_flag,
                        (uint8_t*)doca_qp->gpu_tx_ring.addr, doca_qp->gpu_tx_ring.stride_sz,
                        doca_qp->gpu_tx_ring.stride_num, doca_qp->gpu_tx_ring.flag,
                        cu_frame_size_, 1, WQE_NUM);
                }
            }

#if DOCA_ENABLE_NON_RDMA == 1
            targs.qp_cpu = doca_qp->get_gpu();
            targs.exit_flag = (uint64_t*)calloc(1, sizeof(uint64_t));
            *(targs.exit_flag) = 0;

            int result = pthread_create(&cpu_proxy_thread_id, NULL, progress_cpu_proxy, (void*)&targs);
            if (result != 0) {
                HSB_LOG_INFO("Failed to create thread");
                break;
            }
#endif

            doca_kernel_launched = true;
        }
    }

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

uint32_t GpuRoceTransceiver::get_qp_number() { return qp_number_; }

uint32_t GpuRoceTransceiver::get_rkey() { return rkey_; }

uint64_t GpuRoceTransceiver::external_frame_memory()
{
    // Using iova option with dmabuf, so frame memory starts at 0
    return 0;
}

void* GpuRoceTransceiver::get_rx_ring_data_addr()
{
    return doca_qp->gpu_rx_ring.addr;
}

uint64_t* GpuRoceTransceiver::get_rx_ring_flag_addr()
{
    return doca_qp->gpu_rx_ring.flag;
}

void* GpuRoceTransceiver::get_tx_ring_data_addr()
{
    return doca_qp->gpu_tx_ring.addr;
}

uint64_t* GpuRoceTransceiver::get_tx_ring_flag_addr()
{
    return doca_qp->gpu_tx_ring.flag;
}

std::mutex& GpuRoceTransceiver::get_lock()
{
    static std::mutex instance;
    return instance;
}

static inline struct timespec add_ms_local(struct timespec& ts,
    unsigned long ms)
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

bool GpuRoceTransceiver::get_next_frame(unsigned timeout_ms,
    RoceReceiverMetadata& metadata)
{
    HSB_LOG_INFO("DOCA get_next_frame");
    int status = pthread_mutex_lock(&ready_mutex_);
    if (status != 0) {
        HSB_LOG_ERROR("pthread_mutex_lock returned status={}.", status);
        return false;
    }
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        HSB_LOG_ERROR("clock_gettime failed, errno={}.", errno);
    }
    struct timespec timeout = add_ms_local(now, timeout_ms);

    HSB_LOG_INFO("DOCA waiting for frame to be ready");
    while (!ready_) {
        status = pthread_cond_timedwait(&ready_condition_, &ready_mutex_, &timeout);
        if (status == ETIMEDOUT) {
            break;
        }
        if (status != 0) {
            HSB_LOG_ERROR("pthread_cond_wait returned status={}.", status);
            break;
        }
    }
    HSB_LOG_INFO("DOCA waiting for frame done: ready_={}", ready_);
    bool r = ready_;
    ready_ = false;
    metadata.frame_number = frame_number_;
    metadata.dropped = dropped_;
    if (r) {
        Hololink::FrameMetadata frame_metadata;
        // memset(&frame_metadata, 0, sizeof(frame_metadata));
        metadata.rx_write_requests = rx_write_requests_;
        metadata.imm_data = imm_data_;
        metadata.received_s = received_.tv_sec;
        metadata.received_ns = received_.tv_nsec;
        // metadata.frame_memory = frame_memory_;
        metadata.metadata_memory = 0;
        metadata.frame_metadata = {};
        // Print received frame number on device
        HSB_LOG_INFO("received frame number={}", frame_metadata.frame_number);
    } else {
        metadata.rx_write_requests = 0;
        metadata.imm_data = 0;
        metadata.received_s = 0;
        metadata.received_ns = 0;
        metadata.frame_memory = 0;
        metadata.metadata_memory = 0;
        metadata.frame_metadata = {}; // All 0s
    }
    status = pthread_mutex_unlock(&ready_mutex_);
    if (status != 0) {
        HSB_LOG_ERROR("pthread_mutex_unlock returned status={}.", status);
    }
    return r;
}

} // namespace hololink::operators
