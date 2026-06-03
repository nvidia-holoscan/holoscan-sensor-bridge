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

#ifndef DEVICE_HOLOLINK_OPERATORS_GPU_ROCE_TRANSCEIVER_CUH
#define DEVICE_HOLOLINK_OPERATORS_GPU_ROCE_TRANSCEIVER_CUH

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <doca_gpunetio_dev_verbs_twosided.cuh>

#include "gpu_roce_transceiver_common.hpp"

__device__ void prepare_receive_send(struct doca_gpu_dev_verbs_qp* qp, const size_t frame_size, const uint32_t mkey)
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

__device__ void prepare_send_shared(struct doca_gpu_dev_verbs_qp* qp, struct doca_gpu_dev_verbs_wqe* wqe_sh, const size_t frame_size, const uint32_t mkey)
{
    struct doca_gpu_dev_verbs_wqe* wqe_ptr;
    struct doca_gpu_dev_verbs_wqe_ctrl_seg cseg;

    wqe_ptr = &(wqe_sh[threadIdx.x]);
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
    doca_gpu_dev_verbs_mark_wqes_ready<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA>(qp, wqe_idx, wqe_idx);

    doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA,
        DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(qp, wqe_idx + 1);
}

template <enum gpu_roce_max_frame_size max_frame_size = GPU_ROCE_MAX_FRAME_SIZE_0B>
__device__ inline void send_bf(struct doca_gpu_dev_verbs_qp* qp, struct doca_gpu_dev_verbs_wqe* wqe_ptr_sh, uint64_t wqe_idx, uint64_t buffer)
{
    uint16_t wqe_idx_ = wqe_idx;

    // Update WQE index
    wqe_ptr_sh[threadIdx.x].snd_cseg.opmod_idx_opcode = doca_gpu_dev_verbs_bswap32(((uint32_t)wqe_idx_ << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) | DOCA_GPUNETIO_MLX5_OPCODE_SEND);

    // Update WQE content
    if (max_frame_size == GPU_ROCE_MAX_FRAME_SIZE_0B)
        wqe_ptr_sh[threadIdx.x].dseg1.addr = doca_gpu_dev_verbs_bswap64(buffer);
    else {
        if (max_frame_size >= GPU_ROCE_MAX_FRAME_SIZE_4B)
            wqe_ptr_sh[threadIdx.x].dseg1.lkey = ((uint32_t*)buffer)[0];
        if (max_frame_size >= GPU_ROCE_MAX_FRAME_SIZE_8B)
            ((uint32_t*)&wqe_ptr_sh[threadIdx.x].dseg1)[2] = ((uint32_t*)buffer)[1];
        if (max_frame_size >= GPU_ROCE_MAX_FRAME_SIZE_12B)
            ((uint32_t*)&wqe_ptr_sh[threadIdx.x].dseg1)[3] = ((uint32_t*)buffer)[2];
        if (max_frame_size >= GPU_ROCE_MAX_FRAME_SIZE_16B)
            wqe_ptr_sh[threadIdx.x].dseg2.byte_count = ((uint32_t*)buffer)[3];
        if (max_frame_size >= GPU_ROCE_MAX_FRAME_SIZE_20B)
            wqe_ptr_sh[threadIdx.x].dseg2.lkey = ((uint32_t*)buffer)[4];
        if (max_frame_size >= GPU_ROCE_MAX_FRAME_SIZE_24B)
            ((uint32_t*)&wqe_ptr_sh[threadIdx.x].dseg2)[2] = ((uint32_t*)buffer)[5];
        if (max_frame_size >= GPU_ROCE_MAX_FRAME_SIZE_28B)
            ((uint32_t*)&wqe_ptr_sh[threadIdx.x].dseg2)[3] = ((uint32_t*)buffer)[6];
        if (max_frame_size >= GPU_ROCE_MAX_FRAME_SIZE_32B)
            wqe_ptr_sh[threadIdx.x].dseg3.byte_count = ((uint32_t*)buffer)[7];
        if (max_frame_size >= GPU_ROCE_MAX_FRAME_SIZE_36B)
            wqe_ptr_sh[threadIdx.x].dseg3.lkey = ((uint32_t*)buffer)[8];
        if (max_frame_size >= GPU_ROCE_MAX_FRAME_SIZE_40B)
            ((uint32_t*)&wqe_ptr_sh[threadIdx.x].dseg3)[2] = ((uint32_t*)buffer)[9];
        if (max_frame_size == GPU_ROCE_MAX_FRAME_SIZE_44B)
            ((uint32_t*)&wqe_ptr_sh[threadIdx.x].dseg3)[3] = ((uint32_t*)buffer)[10];
    }

    // Ensure in-order send
    doca_gpu_dev_verbs_mark_wqes_ready<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA>(qp, wqe_idx, wqe_idx);

    // Trigger send
    doca_gpu_dev_verbs_submit_bf<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA,
        DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>(qp, wqe_idx + 1, &(wqe_ptr_sh[threadIdx.x]));
}

__device__ inline uint64_t repost_receive(struct doca_gpu_dev_verbs_qp* qp, uint64_t rwqe_idx)
{
    doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA,
        DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB,
        DOCA_GPUNETIO_VERBS_QP_RQ>(qp, rwqe_idx + 1);

    return rwqe_idx;
}

#endif // DEVICE_HOLOLINK_OPERATORS_GPU_ROCE_TRANSCEIVER_CUH
