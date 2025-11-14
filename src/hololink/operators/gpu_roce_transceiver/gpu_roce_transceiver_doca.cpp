
#include "gpu_roce_transceiver.hpp"

static uint32_t align_up_uint32(uint32_t value, uint32_t alignment)
{
    uint64_t remainder = (value % alignment);
    if (remainder == 0)
        return value;
    return (uint32_t)(value + (alignment - remainder));
}

static uint32_t calc_cq_external_umem_size(uint32_t queue_size)
{
    uint32_t cqe_buf_size = 0;

    if (queue_size != 0)
        cqe_buf_size = (uint32_t)(queue_size * sizeof(struct mlx5_cqe64));

    return align_up_uint32(cqe_buf_size + VERBS_TEST_DBR_SIZE, get_page_size());
}

static uint32_t calc_qp_external_umem_size(uint32_t rq_nwqes, uint32_t sq_nwqes)
{
    uint32_t rq_ring_size = 0;
    uint32_t sq_ring_size = 0;

    if (rq_nwqes != 0)
        rq_ring_size = (uint32_t)(rq_nwqes * sizeof(struct mlx5_wqe_data_seg));
    if (sq_nwqes != 0)
        sq_ring_size = (uint32_t)(sq_nwqes * sizeof(struct doca_gpu_dev_verbs_wqe));

    return align_up_uint32(rq_ring_size + sq_ring_size, get_page_size());
}

static void mlx5_init_cqes(struct mlx5_cqe64* cqes, uint32_t nb_cqes)
{
    for (uint32_t cqe_idx = 0; cqe_idx < nb_cqes; cqe_idx++)
        cqes[cqe_idx].op_own = (MLX5_CQE_INVALID << DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT) | MLX5_CQE_OWNER_MASK;
}

namespace hololink::operators {

struct doca_verbs_context* open_ib_device(char* name)
{
    int nb_ibdevs = 0;
    struct ibv_device** ibdev_list = ibv_get_device_list(&nb_ibdevs);
    struct doca_verbs_context* context;

    if ((ibdev_list == NULL) || (nb_ibdevs == 0)) {
        HSB_LOG_ERROR("Failed to get RDMA devices list, ibdev_list null");
        return NULL;
    }

    for (int i = 0; i < nb_ibdevs; i++) {
        if (strncmp(ibv_get_device_name(ibdev_list[i]), name, strlen(name)) == 0) {
            struct ibv_device* dev_handle = ibdev_list[i];
            ibv_free_device_list(ibdev_list);

            if (doca_verbs_bridge_verbs_context_create(dev_handle,
                    DOCA_VERBS_CONTEXT_CREATE_FLAGS_NONE,
                    &context)
                != DOCA_SUCCESS)
                return NULL;

            return context;
        }
    }

    ibv_free_device_list(ibdev_list);

    return NULL;
}

doca_error_t DocaCq::create()
{
    struct doca_verbs_cq_attr* cq_attr = nullptr;
    uint32_t external_umem_size;
    struct mlx5_cqe64* cq_ring_haddr = nullptr;
    doca_error_t result = DOCA_SUCCESS;
    cudaError_t result_cuda;

    result = doca_verbs_cq_attr_create(&cq_attr);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to create CQ attributes: {}",
            doca_error_get_descr(result));
        return result;
    }

    // Set CQ attributes
    result = doca_verbs_cq_attr_set_external_uar(cq_attr, uar);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to set external UAR: {}",
            doca_error_get_descr(result));
        goto exit;
    }

    result = doca_verbs_cq_attr_set_cq_size(cq_attr, cqe_num);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to set CQ size: {}", doca_error_get_descr(result));
        goto exit;
    }

    result = doca_verbs_cq_attr_set_external_datapath_en(cq_attr, 1);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to set external datapath enable: {}",
            doca_error_get_descr(result));
        goto exit;
    }

    result = doca_verbs_cq_attr_set_entry_size(cq_attr, DOCA_VERBS_CQ_ENTRY_SIZE_64);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to set CQ entry size: {}",
            doca_error_get_descr(result));
        goto exit;
    }

    result = doca_verbs_cq_attr_set_cq_overrun(cq_attr, 1);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to set doca verbs cq size");
        goto exit;
    }

    external_umem_size = calc_cq_external_umem_size(cqe_num);

    cq_ring_haddr = (struct mlx5_cqe64*)(calloc(external_umem_size, sizeof(uint8_t)));
    if (cq_ring_haddr == nullptr) {
        HSB_LOG_ERROR("Failed to allocate cq host ring buffer memory for initialization");
        goto exit;
    }

    mlx5_init_cqes(cq_ring_haddr, cqe_num);

    if (umem_cpu) {
        result = doca_gpu_mem_alloc(gdev,
            external_umem_size,
            get_page_size(),
            DOCA_GPU_MEM_TYPE_CPU_GPU,
            (void**)&umem_dev_ptr,
            (void**)&umem_dev_ptr);
        if (result != DOCA_SUCCESS) {
            HSB_LOG_ERROR("Failed to alloc gpu memory for external umem qp");
            goto exit;
        }

        result = doca_umem_create(ndev,
            umem_dev_ptr,
            external_umem_size,
            DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_ATOMIC,
            &umem);
        if (result != DOCA_SUCCESS) {
            HSB_LOG_ERROR("Failed to create gpu umem: {}", doca_error_get_descr(result));
            goto exit;
        }
    } else {
        // Assume CPU and GPU mem pointers are the same
        result = doca_gpu_mem_alloc(gdev,
            external_umem_size,
            get_page_size(),
            DOCA_GPU_MEM_TYPE_GPU,
            (void**)&umem_dev_ptr,
            nullptr);

        if (result != DOCA_SUCCESS) {
            HSB_LOG_ERROR("Failed to alloc gpu memory for external umem qp");
            goto exit;
        }

        result = doca_umem_gpu_create(gdev,
            ndev,
            umem_dev_ptr,
            external_umem_size,
            DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_ATOMIC,
            &umem);
        if (result != DOCA_SUCCESS) {
            HSB_LOG_ERROR("Failed to create gpu umem: {}", doca_error_get_descr(result));
            goto exit;
        }
    }

    if (umem_cpu)
        memcpy((void*)umem_dev_ptr, (void*)(cq_ring_haddr), external_umem_size);
    else {
        result_cuda = cudaMemcpy(umem_dev_ptr, (void*)(cq_ring_haddr), external_umem_size, cudaMemcpyDefault);
        if (result_cuda != cudaSuccess) {
            HSB_LOG_ERROR("Failed to cudaMempy gpu cq cq ring buffer");
            goto exit;
        }
    }

    result = doca_verbs_cq_attr_set_external_umem(cq_attr, umem, 0);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to set doca verbs cq external umem");
        goto exit;
    }

    // Create the CQ
    result = doca_verbs_cq_create(vctx, cq_attr, &cq);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to create CQ: {}", doca_error_get_descr(result));
        goto exit;
    }

exit:
    doca_verbs_cq_attr_destroy(cq_attr);
    if (cq_ring_haddr)
        free(cq_ring_haddr);
    return result;
}

doca_error_t DocaQp::create(struct doca_verbs_context* verbs_ctx, const size_t frame_size)
{
    size_t dbr_umem_align_sz;
    struct doca_verbs_qp_init_attr* qp_init_attr = nullptr;
    doca_error_t result;
    cudaError_t result_cuda;
    uint32_t external_umem_size;

    result = doca_verbs_qp_init_attr_create(&qp_init_attr);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to create QP init attributes: {}",
            doca_error_get_descr(result));
        return result;
    }

#if DOCA_VERBS_ENABLE_RECV_INLINE == 1
    if ((int)frame_size <= GPU_ROCE_MAX_RECV_INLINE_WQE) {
        struct doca_verbs_device_attr* verbs_device_attr;

        result = doca_verbs_query_device(verbs_ctx, &verbs_device_attr);
        if (result != DOCA_SUCCESS) {
            HSB_LOG_ERROR("Failed to query device attributes: {}", doca_error_get_descr(result));
            return result;
        }

        result = doca_verbs_device_attr_get_is_cqe_inline_supported(verbs_device_attr);
        if (result != DOCA_SUCCESS) {
            HSB_LOG_ERROR("CQE inline resp not supported by this device: {}", doca_error_get_descr(result));
            return result;
        }
    }
#endif

    // Set QP attributes
    result = doca_verbs_qp_init_attr_set_external_uar(qp_init_attr, uar);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to set external UAR for QP: {}",
            doca_error_get_descr(result));
        return result;
    }

    result = doca_verbs_qp_init_attr_set_pd(qp_init_attr, vpd);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to set PD for QP: {}", doca_error_get_descr(result));
        goto exit;
    }

    result = doca_verbs_qp_init_attr_set_external_datapath_en(qp_init_attr, 1);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to set external datapath for QP: {}",
            doca_error_get_descr(result));
        goto exit;
    }

    // Set send/receive CQs
    result = doca_verbs_qp_init_attr_set_receive_cq(qp_init_attr, cq_rq);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to set receive CQ: {}", doca_error_get_descr(result));
        goto exit;
    }

    result = doca_verbs_qp_init_attr_set_send_cq(qp_init_attr, cq_sq);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to set send CQ: {}", doca_error_get_descr(result));
        goto exit;
    }

    // Set WR capacities
    result = doca_verbs_qp_init_attr_set_sq_wr(qp_init_attr, wqe_num);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to set SQ WR: {}", doca_error_get_descr(result));
        goto exit;
    }

    result = doca_verbs_qp_init_attr_set_rq_wr(qp_init_attr, wqe_num);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to set RQ WR: {}", doca_error_get_descr(result));
        goto exit;
    }

    // send sge and recv sge
    result = doca_verbs_qp_init_attr_set_send_max_sges(qp_init_attr, 1);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to set send max sges: {}",
            doca_error_get_descr(result));
        goto exit;
    }

    result = doca_verbs_qp_init_attr_set_receive_max_sges(qp_init_attr, 1);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to set receive max sges: {}",
            doca_error_get_descr(result));
        goto exit;
    }

    // Set QP type to UC (Unreliable Connected) - same as IB implementation
    result = doca_verbs_qp_init_attr_set_qp_type(qp_init_attr, DOCA_VERBS_QP_TYPE_UC);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to set QP type: {}", doca_error_get_descr(result));
        goto exit;
    }

    // set sq_sig_all to 0
    result = doca_verbs_qp_init_attr_set_sq_sig_all(qp_init_attr, 0);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to set sq sig all: {}", doca_error_get_descr(result));
        goto exit;
    }

#if DOCA_VERBS_ENABLE_RECV_INLINE == 1
    if ((int)frame_size <= GPU_ROCE_MAX_RECV_INLINE_WQE) {
        result = doca_verbs_qp_init_attr_set_cqe_inline(qp_init_attr, DOCA_VERBS_QP_SCATTER_TO_CQE_32B);
        if (result != DOCA_SUCCESS) {
            HSB_LOG_ERROR("Failed to set cqe_inline");
            goto exit;
        }
    }
#endif

    external_umem_size = calc_qp_external_umem_size(wqe_num, wqe_num);

    if (umem_cpu) {
        result = doca_gpu_mem_alloc(gdev,
            external_umem_size,
            get_page_size(),
            DOCA_GPU_MEM_TYPE_CPU_GPU,
            (void**)&umem_dev_ptr,
            (void**)&umem_dev_ptr);
        if (result != DOCA_SUCCESS) {
            HSB_LOG_ERROR("Failed to alloc gpu memory for external umem qp");
            goto exit;
        }

        result = doca_umem_create(ndev,
            umem_dev_ptr,
            external_umem_size,
            DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_ATOMIC,
            &umem);
        if (result != DOCA_SUCCESS) {
            HSB_LOG_ERROR("Failed to create gpu umem: {}", doca_error_get_descr(result));
            goto exit;
        }
    } else {
        // Assume CPU and GPU mem pointers are the same
        result = doca_gpu_mem_alloc(gdev,
            external_umem_size,
            get_page_size(),
            DOCA_GPU_MEM_TYPE_GPU,
            (void**)&umem_dev_ptr,
            nullptr);

        if (result != DOCA_SUCCESS) {
            HSB_LOG_ERROR("Failed to alloc gpu memory for external umem qp");
            goto exit;
        }

        result = doca_umem_gpu_create(gdev,
            ndev,
            umem_dev_ptr,
            external_umem_size,
            DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_ATOMIC,
            &umem);
        if (result != DOCA_SUCCESS) {
            HSB_LOG_ERROR("Failed to create gpu umem: {}", doca_error_get_descr(result));
            goto exit;
        }
    }

    result = doca_verbs_qp_init_attr_set_external_umem(qp_init_attr, umem, 0);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to set doca verbs qp external umem");
        goto exit;
    }

    dbr_umem_align_sz = ROUND_UP(VERBS_TEST_DBR_SIZE, get_page_size());

    if (umem_cpu) {
        result = doca_gpu_mem_alloc(gdev,
            dbr_umem_align_sz,
            get_page_size(),
            DOCA_GPU_MEM_TYPE_CPU_GPU,
            (void**)&umem_dbr_dev_ptr,
            (void**)&umem_dbr_dev_ptr);
        if (result != DOCA_SUCCESS) {
            HSB_LOG_ERROR("Failed to alloc gpu memory for external umem qp");
            goto exit;
        }

        result = doca_umem_create(ndev,
            umem_dbr_dev_ptr,
            dbr_umem_align_sz,
            DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_ATOMIC,
            &umem_dbr);
        if (result != DOCA_SUCCESS) {
            HSB_LOG_ERROR("Failed to create gpu umem: {}", doca_error_get_descr(result));
            goto exit;
        }
    } else {
        result = doca_gpu_mem_alloc(gdev,
            dbr_umem_align_sz,
            get_page_size(),
            DOCA_GPU_MEM_TYPE_GPU,
            (void**)&umem_dbr_dev_ptr,
            nullptr);
        if (result != DOCA_SUCCESS) {
            HSB_LOG_ERROR("Failed to alloc gpu memory for external umem qp");
            goto exit;
        }

        result = doca_umem_gpu_create(gdev,
            ndev,
            umem_dbr_dev_ptr,
            dbr_umem_align_sz,
            DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE | DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_ATOMIC,
            &umem_dbr);
        if (result != DOCA_SUCCESS) {
            HSB_LOG_ERROR("Failed to create gpu umem: {}", doca_error_get_descr(result));
            goto exit;
        }
    }

    result = doca_verbs_qp_init_attr_set_external_dbr_umem(qp_init_attr, umem_dbr, 0);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to set doca verbs qp external dbr umem");
        goto exit;
    }

    // Create the QP
    result = doca_verbs_qp_create(vctx, qp_init_attr, &qp);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to create QP: {}", doca_error_get_descr(result));
        goto exit;
    }

    if (umem_cpu) {
        result = doca_gpu_verbs_export_qp(gdev,
            ndev,
            qp,
            DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY,
            umem_dbr_dev_ptr,
            cq_sq,
            cq_rq,
            &gpu_qp);
    } else {
        result = doca_gpu_verbs_export_qp(gdev,
            ndev,
            qp,
            DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB,
            umem_dbr_dev_ptr,
            cq_sq,
            cq_rq,
            &gpu_qp);
    }

    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to create GPU verbs QP");
        goto exit;
    }

    result = doca_gpu_verbs_get_qp_dev(gpu_qp, &gpu_dev_qp);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("doca_gpu_verbs_get_qp_dev failed");
        goto exit;
    }

exit:
    doca_verbs_qp_init_attr_destroy(qp_init_attr);

    return result;
}

doca_error_t DocaQp::create_ring(size_t stride_sz, unsigned stride_num, struct ibv_pd* ibv_pd)
{
    doca_error_t result;
    cudaError_t result_cuda;

    gpu_rx_ring.stride_sz = stride_sz;
    gpu_rx_ring.stride_num = stride_num;

    if (umem_cpu) {
        result = doca_gpu_mem_alloc(gdev,
            gpu_rx_ring.stride_sz * gpu_rx_ring.stride_num,
            get_page_size(),
            DOCA_GPU_MEM_TYPE_CPU_GPU,
            (void**)&(gpu_rx_ring.addr),
            (void**)&(gpu_rx_ring.addr));
    } else {
        result = doca_gpu_mem_alloc(gdev,
            gpu_rx_ring.stride_sz * gpu_rx_ring.stride_num,
            get_page_size(),
            DOCA_GPU_MEM_TYPE_GPU,
            (void**)&(gpu_rx_ring.addr),
            nullptr);
    }

    if (result != DOCA_SUCCESS || gpu_rx_ring.addr == nullptr) {
        HSB_LOG_ERROR("Failed to alloc rx ring buffer: {}", doca_error_get_descr(result));
        goto exit;
    }

    if (umem_cpu)
        memset(gpu_rx_ring.addr, 0, gpu_rx_ring.stride_sz * gpu_rx_ring.stride_num);
    else
        cudaMemset(gpu_rx_ring.addr, 0, gpu_rx_ring.stride_sz * gpu_rx_ring.stride_num);

    gpu_rx_ring.addr_mr = nullptr;
    if (umem_cpu == false) {
        result = doca_gpu_dmabuf_fd(gdev, (void*)gpu_rx_ring.addr, gpu_rx_ring.stride_sz * gpu_rx_ring.stride_num, &gpu_rx_ring.dmabuf_fd);
        if (result == DOCA_SUCCESS) {
            // To use ibv_reg_dmabuf_mr() we must ensure application provides address and size aligned with the system page size.
            gpu_rx_ring.addr_mr = ibv_reg_dmabuf_mr(ibv_pd,
                0,
                gpu_rx_ring.stride_sz * gpu_rx_ring.stride_num,
                0, // (uint64_t)(void*)cu_buffer_,
                gpu_rx_ring.dmabuf_fd,
                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_RELAXED_ORDERING);
        }
    }

    if (gpu_rx_ring.addr_mr == nullptr)
        gpu_rx_ring.addr_mr = ibv_reg_mr_iova(ibv_pd, (void*)gpu_rx_ring.addr, gpu_rx_ring.stride_sz * gpu_rx_ring.stride_num, 0, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_RELAXED_ORDERING);
    if (gpu_rx_ring.addr_mr == nullptr) {
        HSB_LOG_ERROR("Cannot register gpu_rx_ring memory region p={} size={}, errno={}.",
            (uint64_t)gpu_rx_ring.addr, (int)gpu_rx_ring.stride_sz * gpu_rx_ring.stride_num, errno);
        return DOCA_ERROR_NOT_SUPPORTED;
    }

    result = doca_gpu_mem_alloc(gdev,
        sizeof(uint64_t) * gpu_rx_ring.stride_num,
        get_page_size(),
        DOCA_GPU_MEM_TYPE_GPU,
        (void**)&(gpu_rx_ring.flag),
        nullptr);
    if (result != DOCA_SUCCESS || gpu_rx_ring.flag == nullptr) {
        HSB_LOG_ERROR("Failed to alloc rx flag ring buffer: {}", doca_error_get_descr(result));
        goto exit;
    }

    result_cuda = cudaMemset(gpu_rx_ring.flag, 0, sizeof(uint64_t) * gpu_rx_ring.stride_num);
    if (result_cuda != cudaSuccess) {
        HSB_LOG_ERROR("cudaMemset returned error {}", (int)result_cuda);
        goto exit;
    }

    gpu_tx_ring.stride_sz = stride_sz;
    gpu_tx_ring.stride_num = stride_num;

    if (umem_cpu) {
        result = doca_gpu_mem_alloc(gdev,
            gpu_tx_ring.stride_sz * gpu_tx_ring.stride_num,
            get_page_size(),
            DOCA_GPU_MEM_TYPE_CPU_GPU,
            (void**)&(gpu_tx_ring.addr),
            (void**)&(gpu_tx_ring.addr));
    } else {
        result = doca_gpu_mem_alloc(gdev,
            gpu_tx_ring.stride_sz * gpu_tx_ring.stride_num,
            get_page_size(),
            DOCA_GPU_MEM_TYPE_GPU,
            (void**)&(gpu_tx_ring.addr),
            nullptr);
    }

    if (result != DOCA_SUCCESS || gpu_tx_ring.addr == nullptr) {
        HSB_LOG_ERROR("Failed to alloc tx ring buffer: {}", doca_error_get_descr(result));
        goto exit;
    }

    if (umem_cpu)
        memset(gpu_tx_ring.addr, 0, gpu_tx_ring.stride_sz * gpu_tx_ring.stride_num);
    else
        cudaMemset(gpu_tx_ring.addr, 0, gpu_tx_ring.stride_sz * gpu_tx_ring.stride_num);

    // Tx ring MR is not needed in case of send inline but it can be used in case of tx_only and non-inline send mode
    gpu_tx_ring.addr_mr = nullptr;
    if (umem_cpu == false) {
        result = doca_gpu_dmabuf_fd(gdev, (void*)gpu_tx_ring.addr, gpu_tx_ring.stride_sz * gpu_tx_ring.stride_num, &gpu_tx_ring.dmabuf_fd);
        if (result == DOCA_SUCCESS) {
            // To use ibv_reg_dmabuf_mr() we must ensure application provides address and size aligned with the system page size.
            gpu_tx_ring.addr_mr = ibv_reg_dmabuf_mr(ibv_pd,
                0,
                gpu_tx_ring.stride_sz * gpu_tx_ring.stride_num,
                0,
                gpu_tx_ring.dmabuf_fd,
                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_RELAXED_ORDERING);
        }
    }

    if (gpu_tx_ring.addr_mr == nullptr)
        gpu_tx_ring.addr_mr = ibv_reg_mr_iova(ibv_pd, (void*)gpu_tx_ring.addr, gpu_tx_ring.stride_sz * gpu_tx_ring.stride_num, 0, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_RELAXED_ORDERING);
    if (gpu_tx_ring.addr_mr == nullptr) {
        HSB_LOG_ERROR("Cannot register gpu_tx_ring memory region p={} size={}, errno={}.",
            (uint64_t)gpu_tx_ring.addr, (int)gpu_tx_ring.stride_sz * gpu_tx_ring.stride_num, errno);
        return DOCA_ERROR_NOT_SUPPORTED;
    }

    result = doca_gpu_mem_alloc(gdev,
        sizeof(uint64_t) * gpu_tx_ring.stride_num,
        get_page_size(),
        DOCA_GPU_MEM_TYPE_GPU,
        (void**)&(gpu_tx_ring.flag),
        nullptr);
    if (result != DOCA_SUCCESS || gpu_tx_ring.flag == nullptr) {
        HSB_LOG_ERROR("Failed to alloc tx flag ring buffer: {}", doca_error_get_descr(result));
        goto exit;
    }

    cudaMemset(gpu_tx_ring.flag, 0, sizeof(uint64_t) * gpu_tx_ring.stride_num);

    return DOCA_SUCCESS;

exit:
    if (gpu_rx_ring.addr_mr)
        ibv_dereg_mr(gpu_rx_ring.addr_mr);
    if (gpu_rx_ring.addr)
        doca_gpu_mem_free(gdev, gpu_rx_ring.addr);
    if (gpu_rx_ring.flag)
        doca_gpu_mem_free(gdev, gpu_rx_ring.flag);

    if (gpu_tx_ring.addr_mr)
        ibv_dereg_mr(gpu_tx_ring.addr_mr);
    if (gpu_tx_ring.addr)
        doca_gpu_mem_free(gdev, gpu_tx_ring.addr);
    if (gpu_tx_ring.flag)
        doca_gpu_mem_free(gdev, gpu_tx_ring.flag);

    return result;
}

doca_error_t DocaQp::connect(struct doca_verbs_gid& doca_rgid, uint32_t gid_index, uint32_t dest_qp_num)
{
    doca_error_t result;
    struct doca_verbs_qp_attr* verbs_qp_attr {};
    struct doca_verbs_ah_attr* ah_attr = nullptr;
    uint32_t rst2init_mask = DOCA_UC_QP_RST2INIT_REQ_ATTR_MASK;
    uint32_t init2rtr_mask = DOCA_UC_QP_INIT2RTR_REQ_ATTR_MASK;
    uint32_t rtr2rts_mask = DOCA_UC_QP_RTR2RTS_REQ_ATTR_MASK;

    result = doca_verbs_ah_attr_create(vctx, &ah_attr);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to create AH attributes: {}",
            doca_error_get_descr(result));
        return result;
    }

    result = doca_verbs_qp_attr_create(&verbs_qp_attr);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to create AH attributes: {}",
            doca_error_get_descr(result));
        return result;
    }

    result = doca_verbs_ah_attr_set_gid(ah_attr, doca_rgid);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to set remote gid: {}", doca_error_get_descr(result));
        return result;
    }

    /* IB only parameter */
    result = doca_verbs_ah_attr_set_dlid(ah_attr, 0);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to set dlid {}", doca_error_get_descr(result));
        goto exit;
    }

    result = doca_verbs_ah_attr_set_addr_type(ah_attr, DOCA_VERBS_ADDR_TYPE_IPv4);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to set address type: {}", doca_error_get_descr(result));
        goto exit;
    }

    result = doca_verbs_ah_attr_set_sgid_index(ah_attr, gid_index);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to set sgid index: {}", doca_error_get_descr(result));
        goto exit;
    }

    result = doca_verbs_ah_attr_set_hop_limit(ah_attr, 0xFF);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("Failed to set hop limit: {}", doca_error_get_descr(result));
        goto exit;
    }

    result = doca_verbs_qp_attr_set_path_mtu(verbs_qp_attr, DOCA_MTU_SIZE_4K_BYTES);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("doca_verbs_qp_attr_set_path_mtu {}", doca_error_get_descr(result));
        goto exit;
    }
    result = doca_verbs_qp_attr_set_rq_psn(verbs_qp_attr, 0);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("doca_verbs_qp_attr_set_rq_psn {}", doca_error_get_descr(result));
        goto exit;
    }
    result = doca_verbs_qp_attr_set_sq_psn(verbs_qp_attr, 0);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("doca_verbs_qp_attr_set_sq_psn {}", doca_error_get_descr(result));
        goto exit;
    }
    result = doca_verbs_qp_attr_set_ah_attr(verbs_qp_attr, ah_attr);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("doca_verbs_qp_attr_set_ah_attr {}", doca_error_get_descr(result));
        goto exit;
    }
    result = doca_verbs_qp_attr_set_port_num(verbs_qp_attr, 1);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("doca_verbs_qp_attr_set_port_num {}", doca_error_get_descr(result));
        goto exit;
    }
    result = doca_verbs_qp_attr_set_ack_timeout(verbs_qp_attr, 14);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("doca_verbs_qp_attr_set_ack_timeout {}", doca_error_get_descr(result));
        goto exit;
    }

    result = doca_verbs_qp_attr_set_retry_cnt(verbs_qp_attr, 7);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("doca_verbs_qp_attr_set_retry_cnt {}", doca_error_get_descr(result));
        goto exit;
    }

    result = doca_verbs_qp_attr_set_rnr_retry(verbs_qp_attr, 6);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("doca_verbs_qp_attr_set_rnr_retry {}", doca_error_get_descr(result));
        goto exit;
    }

    result = doca_verbs_qp_attr_set_min_rnr_timer(verbs_qp_attr, 12);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("doca_verbs_qp_attr_set_min_rnr_timer {}", doca_error_get_descr(result));
        goto exit;
    }

    result = doca_verbs_qp_attr_set_allow_remote_write(verbs_qp_attr, 1);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("doca_verbs_qp_attr_set_allow_remote_write {}", doca_error_get_descr(result));
        goto exit;
    }

    result = doca_verbs_qp_attr_set_allow_remote_read(verbs_qp_attr, 1);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("doca_verbs_qp_attr_set_allow_remote_read {}", doca_error_get_descr(result));
        goto exit;
    }

    result = doca_verbs_qp_attr_set_dest_qp_num(verbs_qp_attr, dest_qp_num);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("doca_verbs_qp_attr_set_dest_qp_num {}", doca_error_get_descr(result));
        goto exit;
    }

    /* Modify QP - RST2INIT */
    result = doca_verbs_qp_attr_set_next_state(verbs_qp_attr, DOCA_VERBS_QP_STATE_INIT);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("doca_verbs_qp_attr_set_next_state {}", doca_error_get_descr(result));
        goto exit;
    }

    result = doca_verbs_qp_modify(qp, verbs_qp_attr, rst2init_mask);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("doca_verbs_qp_modify {}", doca_error_get_descr(result));
        goto exit;
    }

    /* Modify QP - INIT2RTR */
    result = doca_verbs_qp_attr_set_next_state(verbs_qp_attr, DOCA_VERBS_QP_STATE_RTR);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("doca_verbs_qp_attr_set_next_state {}", doca_error_get_descr(result));
        goto exit;
    }

    result = doca_verbs_qp_modify(qp, verbs_qp_attr, init2rtr_mask);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("doca_verbs_qp_modify {}", doca_error_get_descr(result));
        goto exit;
    }

    /* Modify QP - RTR2RTS */
    result = doca_verbs_qp_attr_set_next_state(verbs_qp_attr, DOCA_VERBS_QP_STATE_RTS);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("doca_verbs_qp_attr_set_next_state {}", doca_error_get_descr(result));
        goto exit;
    }

    result = doca_verbs_qp_modify(qp, verbs_qp_attr, rtr2rts_mask);
    if (result != DOCA_SUCCESS) {
        HSB_LOG_ERROR("doca_verbs_qp_modify RTR2RTS {}", doca_error_get_descr(result));
        goto exit;
    }

exit:
    doca_verbs_qp_attr_destroy(verbs_qp_attr);
    doca_verbs_ah_attr_destroy(ah_attr);
    return result;
}

DocaQp::~DocaQp()
{
    if (umem)
        doca_umem_destroy(umem);
    if (umem_dbr)
        doca_umem_destroy(umem_dbr);
    if (umem_dev_ptr)
        doca_gpu_mem_free(gdev, umem_dev_ptr);
    if (umem_dbr_dev_ptr)
        doca_gpu_mem_free(gdev, umem_dbr_dev_ptr);
    if (qp)
        doca_verbs_qp_destroy(qp);

    if (gpu_rx_ring.addr_mr)
        ibv_dereg_mr(gpu_rx_ring.addr_mr);
    if (gpu_rx_ring.addr)
        doca_gpu_mem_free(gdev, gpu_rx_ring.addr);
    if (gpu_rx_ring.flag)
        doca_gpu_mem_free(gdev, gpu_rx_ring.flag);

    if (gpu_tx_ring.addr_mr)
        ibv_dereg_mr(gpu_tx_ring.addr_mr);
    if (gpu_tx_ring.addr)
        doca_gpu_mem_free(gdev, gpu_tx_ring.addr);
    if (gpu_tx_ring.flag)
        doca_gpu_mem_free(gdev, gpu_tx_ring.flag);
}

} // namespace hololink::operators
