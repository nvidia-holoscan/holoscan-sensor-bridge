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

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "gpu_roce_transceiver.cuh"
#include "gpu_roce_transceiver.hpp"

#define RWQE_TERMINATE_KEY 0x100
#define DSEG_SIZE_2 12
#define DSEG_SIZE_3 28
#define MAX_SEND_INLINE_WQE 44
#define GPU_ROCE_TRANSCEIVER_DEBUG 0

__global__ void kernel_prepare_receive_send(struct doca_gpu_dev_verbs_qp* qp, const size_t frame_size, const uint32_t mkey)
{
    prepare_receive_send(qp, frame_size, mkey);
}

template <enum gpu_roce_max_frame_size max_frame_size = GPU_ROCE_MAX_FRAME_SIZE_0B>
__global__ void forward_bf(struct doca_gpu_dev_verbs_qp* qp, uint32_t* exit_flag,
    uint8_t* ring_buf, const size_t ring_buf_stride_sz, const uint32_t ring_buf_mkey,
    const uint32_t ring_buf_stride_num, const size_t frame_size)
{
    uint32_t stride;
    struct doca_gpu_dev_verbs_cq* cq_rq = doca_gpu_dev_verbs_qp_get_cq_rq(qp);
    uint8_t* cqe = (uint8_t*)__ldg((uintptr_t*)&cq_rq->cqe_daddr);
    const uint32_t cqe_mask = (__ldg(&cq_rq->cqe_num) - 1);
    doca_gpu_dev_verbs_ticket_t out_ticket = threadIdx.x;
    uint64_t wqe_idx = threadIdx.x;
    struct mlx5_cqe64* cqe64;
    __shared__ struct doca_gpu_dev_verbs_wqe wqe_ptr_sh[WQE_NUM];

    // Warmup
    if (qp == nullptr)
        return;

    prepare_send_shared(qp, wqe_ptr_sh, frame_size, ring_buf_mkey);

#if GPU_ROCE_TRANSCEIVER_DEBUG == 1
    printf("forward kernel: thread %d frame size %d GPU_ROCE_MAX_RECV_INLINE_WQE %d\n",
        threadIdx.x, (int)frame_size, GPU_ROCE_MAX_RECV_INLINE_WQE);
#endif

    while (DOCA_GPUNETIO_VOLATILE(*exit_flag) == 0) {
        stride = receive(cq_rq, cqe, cqe_mask, out_ticket, &cqe64);

        if (stride < ring_buf_stride_num) {
            // No send inline, just send the address value
            if (max_frame_size == GPU_ROCE_MAX_FRAME_SIZE_0B)
                send_bf<max_frame_size>(qp, wqe_ptr_sh, wqe_idx, (((uint64_t)ring_buf_stride_sz) * stride));
            // Send inline, need buffer address at right offset
            else
                send_bf<max_frame_size>(qp, wqe_ptr_sh, wqe_idx, ((uint64_t)ring_buf + (((uint64_t)ring_buf_stride_sz) * stride)));
        }

#if GPU_ROCE_TRANSCEIVER_DEBUG == 1
        else
            printf("Invalid stride=%d; ignoring.", stride);
#endif

        // First WQE_NUM CQE already pre-posted by another kernel
        wqe_idx += WQE_NUM;
        out_ticket = repost_receive(qp, wqe_idx);
    }
}

template <enum gpu_roce_max_frame_size max_frame_size = GPU_ROCE_MAX_FRAME_SIZE_0B>
__global__ void forward(struct doca_gpu_dev_verbs_qp* qp, uint32_t* exit_flag,
    uint8_t* ring_buf, const size_t ring_buf_stride_sz, const uint32_t ring_buf_mkey,
    const uint32_t ring_buf_stride_num, const size_t frame_size)
{
    uint32_t stride;
    struct doca_gpu_dev_verbs_cq* cq_rq = doca_gpu_dev_verbs_qp_get_cq_rq(qp);
    uint8_t* cqe = (uint8_t*)__ldg((uintptr_t*)&cq_rq->cqe_daddr);
    const uint32_t cqe_mask = (__ldg(&cq_rq->cqe_num) - 1);
    doca_gpu_dev_verbs_ticket_t out_ticket = threadIdx.x;
    uint64_t wqe_idx = threadIdx.x;
    struct mlx5_cqe64* cqe64;

    // Warmup
    if (qp == nullptr)
        return;

#if GPU_ROCE_TRANSCEIVER_DEBUG == 1
    printf("forward kernel: thread %d frame size %d GPU_ROCE_MAX_RECV_INLINE_WQE %d\n",
        threadIdx.x, (int)frame_size, GPU_ROCE_MAX_RECV_INLINE_WQE);
#endif

    while (DOCA_GPUNETIO_VOLATILE(*exit_flag) == 0) {
        stride = receive(cq_rq, cqe, cqe_mask, out_ticket, &cqe64);

        if (stride < ring_buf_stride_num) {
            // No send inline, just send the address value
            if (max_frame_size == GPU_ROCE_MAX_FRAME_SIZE_0B)
                send<max_frame_size>(qp, wqe_idx, (((uint64_t)ring_buf_stride_sz) * stride));
            // Send inline, need buffer address at right offset
            else
                send<max_frame_size>(qp, wqe_idx, ((uint64_t)ring_buf + (((uint64_t)ring_buf_stride_sz) * stride)));
        }

#if GPU_ROCE_TRANSCEIVER_DEBUG == 1
        else
            printf("Invalid stride=%d; ignoring.", stride);
#endif

        // First WQE_NUM CQE already pre-posted by another kernel
        wqe_idx += WQE_NUM;
        out_ticket = repost_receive(qp, wqe_idx);
    }
}

__global__ void rx_only(struct doca_gpu_dev_verbs_qp* qp, uint32_t* exit_flag,
    uint8_t* ring_buf, const size_t ring_buf_stride_sz, const uint32_t ring_buf_stride_num,
    uint64_t* ring_flag)
{
    uint32_t stride;
    struct doca_gpu_dev_verbs_cq* cq_rq = doca_gpu_dev_verbs_qp_get_cq_rq(qp);
    uint8_t* cqe = (uint8_t*)__ldg((uintptr_t*)&cq_rq->cqe_daddr);
    const uint32_t cqe_mask = (__ldg(&cq_rq->cqe_num) - 1);
    doca_gpu_dev_verbs_ticket_t out_ticket = threadIdx.x;
    uint64_t wqe_idx = threadIdx.x;
    struct mlx5_cqe64* cqe64;

    // Warmup
    if (qp == nullptr)
        return;

#if GPU_ROCE_TRANSCEIVER_DEBUG == 1
    printf("rx_only kernel: thread %d\n", threadIdx.x);
#endif

    while (DOCA_GPUNETIO_VOLATILE(*exit_flag) == 0) {
        stride = receive(cq_rq, cqe, cqe_mask, out_ticket, &cqe64);

        // Do we need to check stride (page) id in production?
        if (stride < ring_buf_stride_num) {
            while (DOCA_GPUNETIO_VOLATILE(ring_flag[stride]) != 0)
                ;
            doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
            DOCA_GPUNETIO_VOLATILE(ring_flag[stride]) = ((uint64_t)ring_buf + (((uint64_t)ring_buf_stride_sz) * stride));
        }
#if GPU_ROCE_TRANSCEIVER_DEBUG == 1
        else
            printf("Invalid stride=%d/%d; ignoring.", stride, ring_buf_stride_num);
#endif

        // First WQE_NUM CQE already pre-posted by another kernel
        wqe_idx += WQE_NUM;
        out_ticket = repost_receive(qp, wqe_idx);
    }
}

template <enum gpu_roce_max_frame_size max_frame_size = GPU_ROCE_MAX_FRAME_SIZE_0B>
__global__ void tx_only_bf(struct doca_gpu_dev_verbs_qp* qp, uint32_t* exit_flag,
    uint8_t* ring_buf, const size_t ring_buf_stride_sz,
    const uint32_t ring_buf_stride_num, const uint32_t ring_buf_mkey, uint64_t* ring_flag, const size_t frame_size)
{
    uint64_t tx_buf_address;
    uint64_t wqe_idx = threadIdx.x;
    __shared__ struct doca_gpu_dev_verbs_wqe wqe_ptr_sh[WQE_NUM];
    // Assuming ring_buf_stride_num is pow(2)
    const uint32_t ring_buf_stride_mask = ring_buf_stride_num - 1;

    // Warmup
    if (qp == nullptr)
        return;

    prepare_send_shared(qp, wqe_ptr_sh, frame_size, ring_buf_mkey);

    while (DOCA_GPUNETIO_VOLATILE(*exit_flag) == 0) {
        // Assumption is tx_buf_address holds some address from tx_ring data buffer (same mkey in the pre-prepare)
        tx_buf_address = DOCA_GPUNETIO_VERBS_VOLATILE(ring_flag[wqe_idx & ring_buf_stride_mask]);
        if (tx_buf_address == 0)
            continue;

        // doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
        DOCA_GPUNETIO_VERBS_VOLATILE(ring_flag[wqe_idx & ring_buf_stride_mask]) = 0;

        // No send inline, just send the address value
        if (max_frame_size == GPU_ROCE_MAX_FRAME_SIZE_0B)
            send_bf<max_frame_size>(qp, wqe_ptr_sh, wqe_idx, tx_buf_address - (uint64_t)ring_buf);
        // Send inline, need buffer address at right offset
        else
            send_bf<max_frame_size>(qp, wqe_ptr_sh, wqe_idx, tx_buf_address);

        wqe_idx += WQE_NUM;
    }
}

template <enum gpu_roce_max_frame_size max_frame_size = GPU_ROCE_MAX_FRAME_SIZE_0B>
__global__ void tx_only(struct doca_gpu_dev_verbs_qp* qp, uint32_t* exit_flag,
    uint8_t* ring_buf, const size_t ring_buf_stride_sz,
    const uint32_t ring_buf_stride_num, const uint32_t ring_buf_mkey, uint64_t* ring_flag, const size_t frame_size)
{
    uint64_t tx_buf_address;
    uint64_t wqe_idx = threadIdx.x;
    // Assuming ring_buf_stride_num is pow(2)
    const uint32_t ring_buf_stride_mask = ring_buf_stride_num - 1;

    // Warmup
    if (qp == nullptr)
        return;

    while (DOCA_GPUNETIO_VOLATILE(*exit_flag) == 0) {
        // Assumption is tx_buf_address holds some address from tx_ring data buffer (same mkey in the pre-prepare)
        tx_buf_address = DOCA_GPUNETIO_VERBS_VOLATILE(ring_flag[wqe_idx & ring_buf_stride_mask]);
        if (tx_buf_address == 0)
            continue;

        // doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
        DOCA_GPUNETIO_VERBS_VOLATILE(ring_flag[wqe_idx & ring_buf_stride_mask]) = 0;

        // No send inline, just send the address value
        if (max_frame_size == GPU_ROCE_MAX_FRAME_SIZE_0B)
            send<max_frame_size>(qp, wqe_idx, tx_buf_address - (uint64_t)ring_buf);
        // Send inline, need buffer address at right offset
        else
            send<max_frame_size>(qp, wqe_idx, tx_buf_address);

        wqe_idx += WQE_NUM;
    }
}

__global__ void compute(uint32_t* exit_flag,
    uint64_t* ring_rx_flag, const size_t ring_rx_stride_sz, const uint32_t ring_rx_stride_num,
    uint64_t* ring_tx_flag, uint8_t* ring_tx_addr, const size_t ring_tx_stride_sz, const uint32_t ring_tx_stride_num,
    size_t frame_size)
{
    uint64_t rx_buf_address;
    uint64_t tx_buf_address;
    uint64_t stride = threadIdx.x;
    // Assuming ring_buf_stride_num is pow(2)
    const uint32_t ring_rx_stride_mask = ring_rx_stride_num - 1;
    const uint32_t ring_tx_stride_mask = ring_tx_stride_num - 1;

    // Warmup
    if (exit_flag == nullptr)
        return;

    tx_buf_address = ((uint64_t)ring_tx_addr) + ((stride & ring_tx_stride_mask) * (uint64_t)ring_tx_stride_sz);

    while (DOCA_GPUNETIO_VOLATILE(*exit_flag) == 0) {

        rx_buf_address = DOCA_GPUNETIO_VERBS_VOLATILE(ring_rx_flag[stride & ring_rx_stride_mask]);
        if (rx_buf_address == 0)
            continue;

        // doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
        DOCA_GPUNETIO_VERBS_VOLATILE(ring_rx_flag[stride & ring_rx_stride_mask]) = 0;

        // Assuming frame size 32B
        DOCA_GPUNETIO_VOLATILE(((uint64_t*)tx_buf_address)[0]) = DOCA_GPUNETIO_VOLATILE(((uint64_t*)rx_buf_address)[0]);
        DOCA_GPUNETIO_VOLATILE(((uint64_t*)tx_buf_address)[1]) = DOCA_GPUNETIO_VOLATILE(((uint64_t*)rx_buf_address)[1]);

        doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();

        DOCA_GPUNETIO_VERBS_VOLATILE(ring_tx_flag[stride & ring_tx_stride_mask]) = (uint64_t)tx_buf_address;

        stride += WQE_NUM;
        tx_buf_address = ((uint64_t)ring_tx_addr) + ((stride & ring_tx_stride_mask) * ring_tx_stride_sz);
    }
}

extern "C" {

doca_error_t GpuRoceTransceiverPrepareKernel(cudaStream_t stream,
    struct doca_gpu_dev_verbs_qp* qp,
    size_t frame_size,
    uint32_t mkey,
    uint32_t cuda_blocks,
    uint32_t cuda_threads)
{
    cudaError_t result = cudaSuccess;

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (cudaSuccess != result) {
        fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    kernel_prepare_receive_send<<<cuda_blocks, cuda_threads, 0, stream>>>(qp, frame_size, mkey);

    result = cudaGetLastError();
    if (cudaSuccess != result) {
        fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}

doca_error_t GpuRoceTransceiverForwardKernel(cudaStream_t stream,
    struct doca_gpu_dev_verbs_qp* qp,
    uint32_t* exit_flag,
    uint8_t* ring_buf,
    size_t ring_buf_stride_sz,
    uint32_t ring_buf_mkey,
    uint32_t ring_buf_stride_num,
    size_t frame_size,
    const enum doca_gpu_dev_verbs_nic_handler nic_handler,
    uint32_t cuda_blocks,
    uint32_t cuda_threads)
{
    cudaError_t result = cudaSuccess;

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (cudaSuccess != result) {
        fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    if (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF) {
        if (frame_size > MAX_SEND_INLINE_WQE)
            forward_bf<GPU_ROCE_MAX_FRAME_SIZE_0B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
        else {
            if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_4B)
                forward_bf<GPU_ROCE_MAX_FRAME_SIZE_4B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_8B)
                forward_bf<GPU_ROCE_MAX_FRAME_SIZE_8B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_12B)
                forward_bf<GPU_ROCE_MAX_FRAME_SIZE_12B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_16B)
                forward_bf<GPU_ROCE_MAX_FRAME_SIZE_16B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_20B)
                forward_bf<GPU_ROCE_MAX_FRAME_SIZE_20B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_24B)
                forward_bf<GPU_ROCE_MAX_FRAME_SIZE_24B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_28B)
                forward_bf<GPU_ROCE_MAX_FRAME_SIZE_28B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_32B)
                forward_bf<GPU_ROCE_MAX_FRAME_SIZE_32B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_36B)
                forward_bf<GPU_ROCE_MAX_FRAME_SIZE_36B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_40B)
                forward_bf<GPU_ROCE_MAX_FRAME_SIZE_40B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_44B)
                forward_bf<GPU_ROCE_MAX_FRAME_SIZE_44B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
        }
    } else {
        if (frame_size > MAX_SEND_INLINE_WQE)
            forward<GPU_ROCE_MAX_FRAME_SIZE_0B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
        else {
            if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_4B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_4B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_8B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_8B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_12B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_12B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_16B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_16B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_20B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_20B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_24B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_24B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_28B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_28B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_32B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_32B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_36B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_36B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_40B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_40B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_44B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_44B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
        }
    }

    result = cudaGetLastError();
    if (cudaSuccess != result) {
        fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}

doca_error_t GpuRoceTransceiverRxOnlyKernel(cudaStream_t stream,
    struct doca_gpu_dev_verbs_qp* qp, uint32_t* exit_flag,
    uint8_t* ring_buf, const size_t ring_buf_stride_sz,
    const uint32_t ring_buf_stride_num, uint64_t* ring_flag,
    size_t frame_size, uint32_t cuda_blocks, uint32_t cuda_threads)
{
    cudaError_t result = cudaSuccess;

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (cudaSuccess != result) {
        fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    rx_only<<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_flag);

    result = cudaGetLastError();
    if (cudaSuccess != result) {
        fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}

doca_error_t GpuRoceTransceiverTxOnlyKernel(cudaStream_t stream,
    struct doca_gpu_dev_verbs_qp* qp, uint32_t* exit_flag,
    uint8_t* ring_buf, const size_t ring_buf_stride_sz,
    const uint32_t ring_buf_stride_num, uint32_t ring_buf_mkey, uint64_t* ring_flag,
    size_t frame_size, const enum doca_gpu_dev_verbs_nic_handler nic_handler,
    uint32_t cuda_blocks, uint32_t cuda_threads)
{
    cudaError_t result = cudaSuccess;

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (cudaSuccess != result) {
        fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    if (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF) {
        if (frame_size > MAX_SEND_INLINE_WQE)
            tx_only_bf<GPU_ROCE_MAX_FRAME_SIZE_0B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_buf_mkey, ring_flag, frame_size);
        else {
            if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_4B)
                tx_only_bf<GPU_ROCE_MAX_FRAME_SIZE_4B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_buf_mkey, ring_flag, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_8B)
                tx_only_bf<GPU_ROCE_MAX_FRAME_SIZE_8B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_buf_mkey, ring_flag, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_12B)
                tx_only_bf<GPU_ROCE_MAX_FRAME_SIZE_12B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_buf_mkey, ring_flag, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_16B)
                tx_only_bf<GPU_ROCE_MAX_FRAME_SIZE_16B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_buf_mkey, ring_flag, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_20B)
                tx_only_bf<GPU_ROCE_MAX_FRAME_SIZE_20B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_buf_mkey, ring_flag, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_24B)
                tx_only_bf<GPU_ROCE_MAX_FRAME_SIZE_24B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_buf_mkey, ring_flag, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_28B)
                tx_only_bf<GPU_ROCE_MAX_FRAME_SIZE_28B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_buf_mkey, ring_flag, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_32B)
                tx_only_bf<GPU_ROCE_MAX_FRAME_SIZE_32B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_buf_mkey, ring_flag, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_36B)
                tx_only_bf<GPU_ROCE_MAX_FRAME_SIZE_36B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_buf_mkey, ring_flag, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_40B)
                tx_only_bf<GPU_ROCE_MAX_FRAME_SIZE_40B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_buf_mkey, ring_flag, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_44B)
                tx_only_bf<GPU_ROCE_MAX_FRAME_SIZE_44B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_buf_mkey, ring_flag, frame_size);
        }
    } else {
        if (frame_size > MAX_SEND_INLINE_WQE)
            tx_only<GPU_ROCE_MAX_FRAME_SIZE_0B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_buf_mkey, ring_flag, frame_size);
        else {
            if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_4B)
                tx_only<GPU_ROCE_MAX_FRAME_SIZE_4B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_buf_mkey, ring_flag, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_8B)
                tx_only<GPU_ROCE_MAX_FRAME_SIZE_8B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_buf_mkey, ring_flag, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_12B)
                tx_only<GPU_ROCE_MAX_FRAME_SIZE_12B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_buf_mkey, ring_flag, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_16B)
                tx_only<GPU_ROCE_MAX_FRAME_SIZE_16B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_buf_mkey, ring_flag, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_20B)
                tx_only<GPU_ROCE_MAX_FRAME_SIZE_20B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_buf_mkey, ring_flag, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_24B)
                tx_only<GPU_ROCE_MAX_FRAME_SIZE_24B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_buf_mkey, ring_flag, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_28B)
                tx_only<GPU_ROCE_MAX_FRAME_SIZE_28B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_buf_mkey, ring_flag, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_32B)
                tx_only<GPU_ROCE_MAX_FRAME_SIZE_32B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_buf_mkey, ring_flag, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_36B)
                tx_only<GPU_ROCE_MAX_FRAME_SIZE_36B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_buf_mkey, ring_flag, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_40B)
                tx_only<GPU_ROCE_MAX_FRAME_SIZE_40B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_buf_mkey, ring_flag, frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_44B)
                tx_only<GPU_ROCE_MAX_FRAME_SIZE_44B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_buf_mkey, ring_flag, frame_size);
        }
    }

    result = cudaGetLastError();
    if (cudaSuccess != result) {
        fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}

doca_error_t GpuRoceTransceiverComputeKernel(cudaStream_t stream, uint32_t* exit_flag,
    uint64_t* ring_rx_flag, const size_t ring_rx_stride_sz, const uint32_t ring_rx_stride_num,
    uint64_t* ring_tx_flag, uint8_t* ring_tx_addr, const size_t ring_tx_stride_sz, const uint32_t ring_tx_stride_num,
    size_t frame_size, uint32_t cuda_blocks, uint32_t cuda_threads)
{
    cudaError_t result = cudaSuccess;

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (cudaSuccess != result) {
        fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    compute<<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
        ring_rx_flag, ring_rx_stride_sz, ring_rx_stride_num,
        ring_tx_flag, ring_tx_addr, ring_tx_stride_sz, ring_tx_stride_num,
        frame_size);

    result = cudaGetLastError();
    if (cudaSuccess != result) {
        fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}

cudaError_t GpuRoceTransceiverQueryOccupancy(int* out_prepare, int* out_rx, int* out_tx)
{
    cudaError_t err;
    int num_blocks = 0;

    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks, prepare_receive_send, 1, 0);
    if (err != cudaSuccess)
        return err;
    if (out_prepare)
        *out_prepare = num_blocks;

    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks, rx_only, 1, 0);
    if (err != cudaSuccess)
        return err;
    if (out_rx)
        *out_rx = num_blocks;

    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks, tx_only<GPU_ROCE_MAX_FRAME_SIZE_0B>, 1, 0);
    if (err != cudaSuccess)
        return err;
    if (out_tx)
        *out_tx = num_blocks;

    return cudaSuccess;
}

} // extern C
