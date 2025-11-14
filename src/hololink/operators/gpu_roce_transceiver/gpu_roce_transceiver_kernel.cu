/*
 * Copyright (c) 2025 NVIDIA CORPORATION AND AFFILIATES.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <doca_error.h>
#include <doca_gpunetio_dev_verbs_twosided.cuh>

#include "gpu_roce_transceiver.hpp"

#define RWQE_TERMINATE_KEY 0x100
#define DSEG_SIZE_2 12
#define DSEG_SIZE_3 28
#define MAX_SEND_INLINE_WQE 44
#define GPU_ROCE_TRANSCEIVER_DEBUG 0

__global__ void prepare_receive_send(struct doca_gpu_dev_verbs_qp* qp, const size_t frame_size, const uint32_t mkey)
{
    doca_gpu_dev_verbs_ticket_t out_ticket;
    struct doca_gpu_dev_verbs_wqe* wqe_ptr;
    struct doca_gpu_dev_verbs_wqe_ctrl_seg cseg;

    doca_gpu_dev_verbs_recv<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU, DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB>(qp,
        doca_gpu_dev_verbs_addr { .addr = 0, .key = doca_gpu_dev_verbs_bswap32(RWQE_TERMINATE_KEY) },
        0,
        &out_ticket);

    wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, threadIdx.x);
    cseg.opmod_idx_opcode = doca_gpu_dev_verbs_bswap32(((uint32_t)threadIdx.x << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) | DOCA_GPUNETIO_MLX5_OPCODE_SEND);
    cseg.fm_ce_se = DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_ERROR_UPDATE;

    if (frame_size > MAX_SEND_INLINE_WQE) {
        cseg.qpn_ds = doca_gpu_dev_verbs_bswap32(__ldg(&qp->sq_num_shift8) | 2);
        wqe_ptr->dseg1.byte_count = doca_gpu_dev_verbs_bswap32((uint32_t)frame_size);
        wqe_ptr->dseg1.lkey = mkey;
    } else {
        if (frame_size <= DSEG_SIZE_2)
            cseg.qpn_ds = doca_gpu_dev_verbs_bswap32(__ldg(&qp->sq_num_shift8) | 2);
        else if (frame_size <= DSEG_SIZE_3)
            cseg.qpn_ds = doca_gpu_dev_verbs_bswap32(__ldg(&qp->sq_num_shift8) | 3);
        else
            cseg.qpn_ds = doca_gpu_dev_verbs_bswap32(__ldg(&qp->sq_num_shift8) | 4);
        wqe_ptr->dseg1.byte_count = doca_gpu_dev_verbs_bswap32((uint32_t)frame_size | MLX5_INLINE_SEG);
    }

    doca_gpu_dev_verbs_store_wqe_seg((uint64_t*)&(wqe_ptr->dseg0), (uint64_t*)&(cseg));
}

__device__ inline uint32_t receive(struct doca_gpu_dev_verbs_cq* cq_rq, uint8_t* cqe, const uint32_t cqe_mask, const doca_gpu_dev_verbs_ticket_t out_ticket, struct mlx5_cqe64** cqe64)
{
    doca_gpu_dev_verbs_poll_cq_at<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU, DOCA_GPUNETIO_VERBS_QP_RQ>(cq_rq, out_ticket);
    (*cqe64) = (struct mlx5_cqe64*)(cqe + ((out_ticket & cqe_mask) * DOCA_GPUNETIO_VERBS_CQE_SIZE));
    return doca_gpu_dev_verbs_bswap32((*cqe64)->imm_inval_pkey) & 0xFFF;
}

template <enum gpu_roce_max_frame_size max_frame_size = GPU_ROCE_MAX_FRAME_SIZE_0B>
__device__ inline void send(struct doca_gpu_dev_verbs_qp* qp, uint64_t wqe_idx, uint64_t buffer)
{
    uint16_t wqe_idx_ = wqe_idx;
    struct doca_gpu_dev_verbs_wqe* wqe_ptr;

    // Get WQE pointer
    wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);

    // Update WQE index
    wqe_ptr->snd_cseg.opmod_idx_opcode = doca_gpu_dev_verbs_bswap32(((uint32_t)wqe_idx_ << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) | DOCA_GPUNETIO_MLX5_OPCODE_SEND);

    // Update WQE content
    if (max_frame_size == GPU_ROCE_MAX_FRAME_SIZE_0B)
        wqe_ptr->dseg1.addr = doca_gpu_dev_verbs_bswap64(buffer);
    else {
        if (max_frame_size >= GPU_ROCE_MAX_FRAME_SIZE_4B)
            wqe_ptr->dseg1.lkey = ((uint32_t*)buffer)[0];
        if (max_frame_size >= GPU_ROCE_MAX_FRAME_SIZE_8B)
            ((uint32_t*)&wqe_ptr->dseg1)[2] = ((uint32_t*)buffer)[1];
        if (max_frame_size >= GPU_ROCE_MAX_FRAME_SIZE_12B)
            ((uint32_t*)&wqe_ptr->dseg1)[3] = ((uint32_t*)buffer)[2];
        if (max_frame_size >= GPU_ROCE_MAX_FRAME_SIZE_16B)
            wqe_ptr->dseg2.byte_count = ((uint32_t*)buffer)[3];
        if (max_frame_size >= GPU_ROCE_MAX_FRAME_SIZE_20B)
            wqe_ptr->dseg2.lkey = ((uint32_t*)buffer)[4];
        if (max_frame_size >= GPU_ROCE_MAX_FRAME_SIZE_24B)
            ((uint32_t*)&wqe_ptr->dseg2)[2] = ((uint32_t*)buffer)[5];
        if (max_frame_size >= GPU_ROCE_MAX_FRAME_SIZE_28B)
            ((uint32_t*)&wqe_ptr->dseg2)[3] = ((uint32_t*)buffer)[6];
        if (max_frame_size >= GPU_ROCE_MAX_FRAME_SIZE_32B)
            wqe_ptr->dseg3.byte_count = ((uint32_t*)buffer)[7];
        if (max_frame_size >= GPU_ROCE_MAX_FRAME_SIZE_36B)
            wqe_ptr->dseg3.lkey = ((uint32_t*)buffer)[8];
        if (max_frame_size >= GPU_ROCE_MAX_FRAME_SIZE_40B)
            ((uint32_t*)&wqe_ptr->dseg3)[2] = ((uint32_t*)buffer)[9];
        if (max_frame_size == GPU_ROCE_MAX_FRAME_SIZE_44B)
            ((uint32_t*)&wqe_ptr->dseg3)[3] = ((uint32_t*)buffer)[10];
    }

    // Ensure in-order send
    doca_gpu_dev_verbs_mark_wqes_ready<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(qp, wqe_idx, wqe_idx);

    // Trigger send
    doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
        DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(qp, wqe_idx + 1);
}

__device__ inline uint64_t repost_receive(struct doca_gpu_dev_verbs_qp* qp, uint64_t rwqe_idx)
{
    doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
        DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB,
        DOCA_GPUNETIO_VERBS_QP_RQ>(qp, rwqe_idx + 1);

    return rwqe_idx;
}

template <enum gpu_roce_max_frame_size max_frame_size = GPU_ROCE_MAX_FRAME_SIZE_0B, bool recv_inline = false>
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
    if (frame_size == 0)
        return;

#if GPU_ROCE_TRANSCEIVER_DEBUG == 1
    printf("forward kernel: thread %d recv_inline %d frame size %d GPU_ROCE_MAX_RECV_INLINE_WQE %d\n",
        threadIdx.x, recv_inline, (int)frame_size, GPU_ROCE_MAX_RECV_INLINE_WQE);
#endif
    while (DOCA_GPUNETIO_VOLATILE(*exit_flag) == 0) {
        stride = receive(cq_rq, cqe, cqe_mask, out_ticket, &cqe64);
#if DOCA_VERBS_ENABLE_RECV_INLINE == 1
        if (recv_inline) {
#if GPU_ROCE_TRANSCEIVER_DEBUG == 1
            if (doca_gpu_dev_verbs_cqe_is_inline(cqe64) == 0)
                printf("Error: CQE %ld at %p has not inline data as expected (%d Bytes)\n",
                    out_ticket, (void*)cqe64, doca_gpu_dev_verbs_cqe_get_bytes(cqe64));
#endif
            if (stride < ring_buf_stride_num)
                send<max_frame_size>(qp, wqe_idx, (uint64_t)doca_gpu_dev_verbs_cqe_get_inl_data(cqe64));
#if GPU_ROCE_TRANSCEIVER_DEBUG == 1
            else
                printf("Invalid stride=%d; ignoring.", stride);
#endif
        } else 
#endif
        {
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
        }

        // First WQE_NUM CQE already pre-posted by another kernel
        wqe_idx += WQE_NUM;
        out_ticket = repost_receive(qp, wqe_idx);
    }
}

template <bool recv_inline = false>
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
    printf("rx_only kernel: thread %d recv_inline %d frame size %d GPU_ROCE_MAX_RECV_INLINE_WQE %d\n",
        threadIdx.x, recv_inline, (int)frame_size, GPU_ROCE_MAX_RECV_INLINE_WQE);
#endif

    while (DOCA_GPUNETIO_VOLATILE(*exit_flag) == 0) {
        stride = receive(cq_rq, cqe, cqe_mask, out_ticket, &cqe64);
        if (recv_inline) {
#if GPU_ROCE_TRANSCEIVER_DEBUG == 1
            if (doca_gpu_dev_verbs_cqe_is_inline(cqe64) == 0)
                printf("Error: CQE %ld at %p has not inline data as expected (%d Bytes)\n",
                    out_ticket, (void*)cqe64, doca_gpu_dev_verbs_cqe_get_bytes(cqe64));
#endif
            // Do we need to check stride (page) id in production?
            if (stride < ring_buf_stride_num)
                ring_flag[stride] = (uint64_t)doca_gpu_dev_verbs_cqe_get_inl_data(cqe64);
#if GPU_ROCE_TRANSCEIVER_DEBUG == 1
            else
                printf("Invalid stride=%d; ignoring.", stride);
#endif
        } else {
            // Do we need to check stride (page) id in production?
            if (stride < ring_buf_stride_num)
                ring_flag[stride] = ((uint64_t)ring_buf + (((uint64_t)ring_buf_stride_sz) * stride));
#if GPU_ROCE_TRANSCEIVER_DEBUG == 1
            else
                printf("Invalid stride=%d; ignoring.", stride);
#endif
        }

        // First WQE_NUM CQE already pre-posted by another kernel
        wqe_idx += WQE_NUM;
        out_ticket = repost_receive(qp, wqe_idx);
    }
}

template <enum gpu_roce_max_frame_size max_frame_size = GPU_ROCE_MAX_FRAME_SIZE_0B>
__global__ void tx_only(struct doca_gpu_dev_verbs_qp* qp, uint32_t* exit_flag,
    uint8_t* ring_buf, const size_t ring_buf_stride_sz,
    const uint32_t ring_buf_stride_num, uint64_t* ring_flag)
{
    uint64_t tx_buf_address;
    uint64_t wqe_idx = threadIdx.x;

    // Warmup
    if (qp == nullptr)
        return;

    while (DOCA_GPUNETIO_VOLATILE(*exit_flag) == 0) {
        // Assumption is tx_buf_address holds some address from tx_ring data buffer (same mkey in the pre-prepare)
        tx_buf_address = DOCA_GPUNETIO_VERBS_VOLATILE(ring_flag[wqe_idx % ring_buf_stride_num]);
        if (tx_buf_address == 0)
            continue;

        // Send no-inline, need buffer offset as registered with iova
        if (max_frame_size == GPU_ROCE_MAX_FRAME_SIZE_0B)
            send<max_frame_size>(qp, wqe_idx, tx_buf_address - (uint64_t)ring_buf);
        // Send inline, need buffer address at right offset
        else
            send<max_frame_size>(qp, wqe_idx, tx_buf_address);

        DOCA_GPUNETIO_VERBS_VOLATILE(ring_flag[wqe_idx % ring_buf_stride_num]) = 0;

        wqe_idx += WQE_NUM;
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

    prepare_receive_send<<<cuda_blocks, cuda_threads, 0, stream>>>(qp, frame_size, mkey);

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

    forward<<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, 0);
    cudaStreamSynchronize(stream);

#if DOCA_VERBS_ENABLE_RECV_INLINE == 1
    if (frame_size <= GPU_ROCE_MAX_RECV_INLINE_WQE) {
        if (frame_size > MAX_SEND_INLINE_WQE)
            forward<GPU_ROCE_MAX_FRAME_SIZE_0B, true><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
        else {
            if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_4B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_4B, true><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_8B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_8B, true><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_12B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_12B, true><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_16B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_16B, true><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_20B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_20B, true><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_24B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_24B, true><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_28B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_28B, true><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_32B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_32B, true><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_36B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_36B, true><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_40B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_40B, true><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_44B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_44B, true><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
        }
    } else {
#endif
        if (frame_size > MAX_SEND_INLINE_WQE)
            forward<GPU_ROCE_MAX_FRAME_SIZE_0B, false><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
        else {
            if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_4B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_4B, false><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_8B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_8B, false><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_12B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_12B, false><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_16B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_16B, false><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_20B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_20B, false><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_24B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_24B, false><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_28B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_28B, false><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_32B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_32B, false><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_36B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_36B, false><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_40B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_40B, false><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
            if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_44B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_44B, false><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_mkey, ring_buf_stride_num, frame_size);
        }
#if DOCA_VERBS_ENABLE_RECV_INLINE == 1
    }
#endif

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

    rx_only<<<cuda_blocks, cuda_threads, 0, stream>>>(nullptr, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_flag);
    cudaStreamSynchronize(stream);

#if DOCA_VERBS_ENABLE_RECV_INLINE == 1
    if (frame_size <= GPU_ROCE_MAX_RECV_INLINE_WQE)
        rx_only<true><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_flag);
    else
#endif
        rx_only<false><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_flag);

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

    tx_only<GPU_ROCE_MAX_FRAME_SIZE_0B><<<cuda_blocks, cuda_threads, 0, stream>>>(nullptr, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_flag);
    cudaStreamSynchronize(stream);

    if (frame_size > MAX_SEND_INLINE_WQE)
        tx_only<GPU_ROCE_MAX_FRAME_SIZE_0B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_flag);
    else {
        if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_4B)
            tx_only<GPU_ROCE_MAX_FRAME_SIZE_4B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_flag);
        if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_8B)
            tx_only<GPU_ROCE_MAX_FRAME_SIZE_8B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_flag);
        if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_12B)
            tx_only<GPU_ROCE_MAX_FRAME_SIZE_12B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_flag);
        if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_16B)
            tx_only<GPU_ROCE_MAX_FRAME_SIZE_16B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_flag);
        if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_20B)
            tx_only<GPU_ROCE_MAX_FRAME_SIZE_20B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_flag);
        if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_24B)
            tx_only<GPU_ROCE_MAX_FRAME_SIZE_24B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_flag);
        if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_28B)
            tx_only<GPU_ROCE_MAX_FRAME_SIZE_28B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_flag);
        if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_32B)
            tx_only<GPU_ROCE_MAX_FRAME_SIZE_32B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_flag);
        if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_36B)
            tx_only<GPU_ROCE_MAX_FRAME_SIZE_36B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_flag);
        if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_40B)
            tx_only<GPU_ROCE_MAX_FRAME_SIZE_40B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_flag);
        if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_44B)
            tx_only<GPU_ROCE_MAX_FRAME_SIZE_44B><<<cuda_blocks, cuda_threads, 0, stream>>>(qp, exit_flag, ring_buf, ring_buf_stride_sz, ring_buf_stride_num, ring_flag);
    }

    result = cudaGetLastError();
    if (cudaSuccess != result) {
        fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}

} // extern "C"
