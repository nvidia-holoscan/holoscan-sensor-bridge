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
__global__ void forward_bf(uint32_t* exit_flag,
    struct doca_gpu_dev_verbs_qp* qp0, struct doca_gpu_dev_verbs_qp* qp1, struct doca_gpu_dev_verbs_qp* qp2, struct doca_gpu_dev_verbs_qp* qp3,
    uint8_t* ring_buf0, uint8_t* ring_buf1, uint8_t* ring_buf2, uint8_t* ring_buf3,
    const size_t ring_buf_stride_sz0, const size_t ring_buf_stride_sz1, const size_t ring_buf_stride_sz2, const size_t ring_buf_stride_sz3,
    const uint32_t ring_buf_mkey0, const uint32_t ring_buf_mkey1, const uint32_t ring_buf_mkey2, const uint32_t ring_buf_mkey3,
    const uint32_t ring_buf_stride_num0, const uint32_t ring_buf_stride_num1, const uint32_t ring_buf_stride_num2, const uint32_t ring_buf_stride_num3,
    const size_t frame_size)
{
    uint32_t stride;
    int status;
    struct doca_gpu_dev_verbs_cq* cq_rq;
    uint8_t* cqe;
    doca_gpu_dev_verbs_ticket_t out_ticket = threadIdx.x;
    uint64_t wqe_idx = threadIdx.x;
    __shared__ struct doca_gpu_dev_verbs_wqe wqe_ptr_sh[WQE_NUM];
    struct mlx5_cqe64* cqe64;

    struct doca_gpu_dev_verbs_qp* qp = nullptr;
    uint8_t* ring_buf;
    size_t ring_buf_stride_sz;
    uint32_t ring_buf_mkey;
    uint32_t ring_buf_stride_num;

    if (blockIdx.x == 0) {
        qp = qp0;
        ring_buf = ring_buf0;
        ring_buf_stride_sz = ring_buf_stride_sz0;
        ring_buf_mkey = ring_buf_mkey0;
        ring_buf_stride_num = ring_buf_stride_num0;
    } else if (blockIdx.x == 1) {
        qp = qp1;
        ring_buf = ring_buf1;
        ring_buf_stride_sz = ring_buf_stride_sz1;
        ring_buf_mkey = ring_buf_mkey1;
        ring_buf_stride_num = ring_buf_stride_num1;
    } else if (blockIdx.x == 2) {
        qp = qp2;
        ring_buf = ring_buf2;
        ring_buf_stride_sz = ring_buf_stride_sz2;
        ring_buf_mkey = ring_buf_mkey2;
        ring_buf_stride_num = ring_buf_stride_num2;
    } else if (blockIdx.x == 3) {
        qp = qp3;
        ring_buf = ring_buf3;
        ring_buf_stride_sz = ring_buf_stride_sz3;
        ring_buf_mkey = ring_buf_mkey3;
        ring_buf_stride_num = ring_buf_stride_num3;
    }

    // Warmup
    if (qp == nullptr)
        return;

    cq_rq = doca_gpu_dev_verbs_qp_get_cq_rq(qp);
    cqe = (uint8_t*)__ldg((uintptr_t*)&cq_rq->cqe_daddr);
    const uint32_t cqe_mask = (__ldg(&cq_rq->cqe_num) - 1);

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
        } else {
            // No frame to send: a NOP lets the next WQEs be posted
            send_bf_nop(qp, wqe_ptr_sh, wqe_idx);
#if GPU_ROCE_TRANSCEIVER_DEBUG == 1
            printf("Invalid stride=%d status=%d; ignoring.", stride, status);
#endif
            // Error CQE: the QP is in error state, stop all the kernel threads
            if (status != 0) {
                DOCA_GPUNETIO_VOLATILE(*exit_flag) = 1;
                break;
            }
        }

        // First WQE_NUM CQE already pre-posted by another kernel
        wqe_idx += WQE_NUM;
        out_ticket = repost_receive(qp, wqe_idx);
    }
}

template <enum gpu_roce_max_frame_size max_frame_size = GPU_ROCE_MAX_FRAME_SIZE_0B>
__global__ void forward(uint32_t* exit_flag,
    struct doca_gpu_dev_verbs_qp* qp0, struct doca_gpu_dev_verbs_qp* qp1, struct doca_gpu_dev_verbs_qp* qp2, struct doca_gpu_dev_verbs_qp* qp3,
    uint8_t* ring_buf0, uint8_t* ring_buf1, uint8_t* ring_buf2, uint8_t* ring_buf3,
    const size_t ring_buf_stride_sz0, const size_t ring_buf_stride_sz1, const size_t ring_buf_stride_sz2, const size_t ring_buf_stride_sz3,
    const uint32_t ring_buf_mkey0, const uint32_t ring_buf_mkey1, const uint32_t ring_buf_mkey2, const uint32_t ring_buf_mkey3,
    const uint32_t ring_buf_stride_num0, const uint32_t ring_buf_stride_num1, const uint32_t ring_buf_stride_num2, const uint32_t ring_buf_stride_num3,
    const size_t frame_size)
{
    uint32_t stride;
    int status;
    struct doca_gpu_dev_verbs_cq* cq_rq;
    uint8_t* cqe;
    doca_gpu_dev_verbs_ticket_t out_ticket = threadIdx.x;
    uint64_t wqe_idx = threadIdx.x;
    struct mlx5_cqe64* cqe64;

    struct doca_gpu_dev_verbs_qp* qp = nullptr;
    uint8_t* ring_buf;
    size_t ring_buf_stride_sz;
    uint32_t ring_buf_mkey;
    uint32_t ring_buf_stride_num;

    if (blockIdx.x == 0) {
        qp = qp0;
        ring_buf = ring_buf0;
        ring_buf_stride_sz = ring_buf_stride_sz0;
        ring_buf_mkey = ring_buf_mkey0;
        ring_buf_stride_num = ring_buf_stride_num0;
    } else if (blockIdx.x == 1) {
        qp = qp1;
        ring_buf = ring_buf1;
        ring_buf_stride_sz = ring_buf_stride_sz1;
        ring_buf_mkey = ring_buf_mkey1;
        ring_buf_stride_num = ring_buf_stride_num1;
    } else if (blockIdx.x == 2) {
        qp = qp2;
        ring_buf = ring_buf2;
        ring_buf_stride_sz = ring_buf_stride_sz2;
        ring_buf_mkey = ring_buf_mkey2;
        ring_buf_stride_num = ring_buf_stride_num2;
    } else if (blockIdx.x == 3) {
        qp = qp3;
        ring_buf = ring_buf3;
        ring_buf_stride_sz = ring_buf_stride_sz3;
        ring_buf_mkey = ring_buf_mkey3;
        ring_buf_stride_num = ring_buf_stride_num3;
    }

    // Warmup
    if (qp == nullptr)
        return;

    cq_rq = doca_gpu_dev_verbs_qp_get_cq_rq(qp);
    cqe = (uint8_t*)__ldg((uintptr_t*)&cq_rq->cqe_daddr);
    const uint32_t cqe_mask = (__ldg(&cq_rq->cqe_num) - 1);

#if GPU_ROCE_TRANSCEIVER_DEBUG == 1
    printf("forward kernel: thread %d frame size %d GPU_ROCE_MAX_RECV_INLINE_WQE %d\n",
        threadIdx.x, (int)frame_size, GPU_ROCE_MAX_RECV_INLINE_WQE);
#endif

    while (DOCA_GPUNETIO_VOLATILE(*exit_flag) == 0) {
        stride = receive(cq_rq, cqe, cqe_mask, out_ticket, &cqe64);

        // Sends are posted in WQE index order: wait for the previous ones, unless exiting
        if (!wait_sq_turn_or_exit(qp, wqe_idx, exit_flag))
            break;

        if (stride < ring_buf_stride_num) {
            // No send inline, just send the address value
            if (max_frame_size == GPU_ROCE_MAX_FRAME_SIZE_0B)
                send<max_frame_size>(qp, wqe_idx, (((uint64_t)ring_buf_stride_sz) * stride));
            // Send inline, need buffer address at right offset
            else
                send<max_frame_size>(qp, wqe_idx, ((uint64_t)ring_buf + (((uint64_t)ring_buf_stride_sz) * stride)));
        } else {
            // No frame to send: a NOP lets the next WQEs be posted
            send_nop(qp, wqe_idx);
#if GPU_ROCE_TRANSCEIVER_DEBUG == 1
            printf("Invalid stride=%d status=%d; ignoring.", stride, status);
#endif
            // Error CQE: the QP is in error state, stop all the kernel threads
            if (status != 0) {
                DOCA_GPUNETIO_VOLATILE(*exit_flag) = 1;
                break;
            }
        }

        // First WQE_NUM CQE already pre-posted by another kernel
        wqe_idx += WQE_NUM;
        out_ticket = repost_receive(qp, wqe_idx);
    }
}

__global__ void rx_only(uint32_t* exit_flag,
    struct doca_gpu_dev_verbs_qp* qp0, struct doca_gpu_dev_verbs_qp* qp1, struct doca_gpu_dev_verbs_qp* qp2, struct doca_gpu_dev_verbs_qp* qp3,
    uint8_t* ring_buf0, uint8_t* ring_buf1, uint8_t* ring_buf2, uint8_t* ring_buf3,
    const size_t ring_buf_stride_sz0, const size_t ring_buf_stride_sz1, const size_t ring_buf_stride_sz2, const size_t ring_buf_stride_sz3,
    const uint32_t ring_buf_stride_num0, const uint32_t ring_buf_stride_num1, const uint32_t ring_buf_stride_num2, const uint32_t ring_buf_stride_num3,
    uint64_t* ring_flag0, uint64_t* ring_flag1, uint64_t* ring_flag2, uint64_t* ring_flag3)
{
    uint32_t stride;
    int status;
    struct doca_gpu_dev_verbs_cq* cq_rq;
    uint8_t* cqe;
    doca_gpu_dev_verbs_ticket_t out_ticket = threadIdx.x;
    uint64_t wqe_idx = threadIdx.x;

    struct doca_gpu_dev_verbs_qp* qp = nullptr;
    uint8_t* ring_buf;
    size_t ring_buf_stride_sz;
    uint64_t* ring_flag;
    uint32_t ring_buf_stride_num;

    if (blockIdx.x == 0) {
        qp = qp0;
        ring_buf = ring_buf0;
        ring_buf_stride_sz = ring_buf_stride_sz0;
        ring_buf_stride_num = ring_buf_stride_num0;
        ring_flag = ring_flag0;
    } else if (blockIdx.x == 1) {
        qp = qp1;
        ring_buf = ring_buf1;
        ring_buf_stride_sz = ring_buf_stride_sz1;
        ring_buf_stride_num = ring_buf_stride_num1;
        ring_flag = ring_flag1;
    } else if (blockIdx.x == 2) {
        qp = qp2;
        ring_buf = ring_buf2;
        ring_buf_stride_sz = ring_buf_stride_sz2;
        ring_buf_stride_num = ring_buf_stride_num2;
        ring_flag = ring_flag2;
    } else if (blockIdx.x == 3) {
        qp = qp3;
        ring_buf = ring_buf3;
        ring_buf_stride_sz = ring_buf_stride_sz3;
        ring_buf_stride_num = ring_buf_stride_num3;
        ring_flag = ring_flag3;
    }

    // Warmup
    if (qp == nullptr)
        return;

    cq_rq = doca_gpu_dev_verbs_qp_get_cq_rq(qp);
    cqe = (uint8_t*)__ldg((uintptr_t*)&cq_rq->cqe_daddr);
    const uint32_t cqe_mask = (__ldg(&cq_rq->cqe_num) - 1);

#if GPU_ROCE_TRANSCEIVER_DEBUG == 1
    printf("rx_only kernel: thread %d\n", threadIdx.x);
#endif

    while (DOCA_GPUNETIO_VOLATILE(*exit_flag) == 0) {
        status = receive_or_exit(cq_rq, cqe, cqe_mask, out_ticket, exit_flag, &stride);
        if (status == GPU_ROCE_RECV_EXIT)
            break;

        // Error CQE: the QP is in error state, stop all the kernel threads
        if (status != 0) {
            DOCA_GPUNETIO_VOLATILE(*exit_flag) = 1;
            break;
        }

        // Do we need to check stride (page) id in production?
        if (stride < ring_buf_stride_num) {
            // Wait for the application to release the previous frame of this page, unless exiting
            uint32_t polls = 0;
            bool exiting = false;
            while (DOCA_GPUNETIO_VOLATILE(ring_flag[stride]) != 0) {
                if ((++polls % GPU_ROCE_EXIT_POLL_INTERVAL) == 0 && DOCA_GPUNETIO_VOLATILE(*exit_flag) != 0) {
                    exiting = true;
                    break;
                }
            }
            if (exiting)
                break;

            // Publish the frame: who sees the flag also sees the frame data written by the NIC
            ring_flag_store_release(&ring_flag[stride], ((uint64_t)ring_buf + (((uint64_t)ring_buf_stride_sz) * stride)));
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
__global__ void tx_only_bf(uint32_t* exit_flag,
    struct doca_gpu_dev_verbs_qp* qp0, struct doca_gpu_dev_verbs_qp* qp1, struct doca_gpu_dev_verbs_qp* qp2, struct doca_gpu_dev_verbs_qp* qp3,
    uint8_t* ring_buf0, uint8_t* ring_buf1, uint8_t* ring_buf2, uint8_t* ring_buf3,
    const size_t ring_buf_stride_sz0, const size_t ring_buf_stride_sz1, const size_t ring_buf_stride_sz2, const size_t ring_buf_stride_sz3,
    const uint32_t ring_buf_stride_num0, const uint32_t ring_buf_stride_num1, const uint32_t ring_buf_stride_num2, const uint32_t ring_buf_stride_num3,
    const uint32_t ring_buf_mkey0, const uint32_t ring_buf_mkey1, const uint32_t ring_buf_mkey2, const uint32_t ring_buf_mkey3,
    uint64_t* ring_flag0, uint64_t* ring_flag1, uint64_t* ring_flag2, uint64_t* ring_flag3,
    const size_t frame_size)
{
    uint64_t tx_buf_address;
    uint64_t wqe_idx = threadIdx.x;
    __shared__ struct doca_gpu_dev_verbs_wqe wqe_ptr_sh[WQE_NUM];

    struct doca_gpu_dev_verbs_qp* qp = nullptr;
    uint8_t* ring_buf;
    size_t ring_buf_stride_sz;
    uint32_t ring_buf_mkey;
    uint64_t* ring_flag;
    uint32_t ring_buf_stride_num;

    if (blockIdx.x == 0) {
        qp = qp0;
        ring_buf = ring_buf0;
        ring_buf_stride_sz = ring_buf_stride_sz0;
        ring_buf_stride_num = ring_buf_stride_num0;
        ring_flag = ring_flag0;
        ring_buf_mkey = ring_buf_mkey0;
    } else if (blockIdx.x == 1) {
        qp = qp1;
        ring_buf = ring_buf1;
        ring_buf_stride_sz = ring_buf_stride_sz1;
        ring_buf_stride_num = ring_buf_stride_num1;
        ring_flag = ring_flag1;
        ring_buf_mkey = ring_buf_mkey1;
    } else if (blockIdx.x == 2) {
        qp = qp2;
        ring_buf = ring_buf2;
        ring_buf_stride_sz = ring_buf_stride_sz2;
        ring_buf_stride_num = ring_buf_stride_num2;
        ring_flag = ring_flag2;
        ring_buf_mkey = ring_buf_mkey2;
    } else if (blockIdx.x == 3) {
        qp = qp3;
        ring_buf = ring_buf3;
        ring_buf_stride_sz = ring_buf_stride_sz3;
        ring_buf_stride_num = ring_buf_stride_num3;
        ring_flag = ring_flag3;
        ring_buf_mkey = ring_buf_mkey3;
    }

    // Assuming ring_buf_stride_num is pow(2)
    const uint32_t ring_buf_stride_mask = ring_buf_stride_num - 1;

    // Warmup
    if (qp == nullptr)
        return;

    prepare_send_shared(qp, wqe_ptr_sh, frame_size, ring_buf_mkey);

    while (DOCA_GPUNETIO_VOLATILE(*exit_flag) == 0) {
        // Assumption is tx_buf_address holds some address from tx_ring data buffer (same mkey in the pre-prepare)
        // If not 0, the frame data written by the application before the flag are visible
        tx_buf_address = ring_flag_load_acquire(&ring_flag[wqe_idx & ring_buf_stride_mask]);
        if (tx_buf_address == 0)
            continue;

        // Sends are posted in WQE index order: wait for the previous ones, unless exiting
        if (!wait_sq_turn_or_exit(qp, wqe_idx, exit_flag))
            break;

        // No send inline, just send the address value
        if (max_frame_size == GPU_ROCE_MAX_FRAME_SIZE_0B)
            send_bf<max_frame_size>(qp, wqe_ptr_sh, wqe_idx, tx_buf_address - (uint64_t)ring_buf);
        // Send inline, need buffer address at right offset
        else
            send_bf<max_frame_size>(qp, wqe_ptr_sh, wqe_idx, tx_buf_address);

        // Release the TX ring slot only now: an inline send has copied the frame into the WQE
        ring_flag_store_release(&ring_flag[wqe_idx & ring_buf_stride_mask], 0);

        wqe_idx += WQE_NUM;
    }
}

template <enum gpu_roce_max_frame_size max_frame_size = GPU_ROCE_MAX_FRAME_SIZE_0B>
__global__ void tx_only(uint32_t* exit_flag,
    struct doca_gpu_dev_verbs_qp* qp0, struct doca_gpu_dev_verbs_qp* qp1, struct doca_gpu_dev_verbs_qp* qp2, struct doca_gpu_dev_verbs_qp* qp3,
    uint8_t* ring_buf0, uint8_t* ring_buf1, uint8_t* ring_buf2, uint8_t* ring_buf3,
    const size_t ring_buf_stride_sz0, const size_t ring_buf_stride_sz1, const size_t ring_buf_stride_sz2, const size_t ring_buf_stride_sz3,
    const uint32_t ring_buf_stride_num0, const uint32_t ring_buf_stride_num1, const uint32_t ring_buf_stride_num2, const uint32_t ring_buf_stride_num3,
    const uint32_t ring_buf_mkey0, const uint32_t ring_buf_mkey1, const uint32_t ring_buf_mkey2, const uint32_t ring_buf_mkey3,
    uint64_t* ring_flag0, uint64_t* ring_flag1, uint64_t* ring_flag2, uint64_t* ring_flag3,
    const size_t frame_size)
{
    uint64_t tx_buf_address;
    uint64_t wqe_idx = threadIdx.x;

    struct doca_gpu_dev_verbs_qp* qp = nullptr;
    uint8_t* ring_buf;
    size_t ring_buf_stride_sz;
    uint32_t ring_buf_mkey;
    uint64_t* ring_flag;
    uint32_t ring_buf_stride_num;

    if (blockIdx.x == 0) {
        qp = qp0;
        ring_buf = ring_buf0;
        ring_buf_stride_sz = ring_buf_stride_sz0;
        ring_buf_stride_num = ring_buf_stride_num0;
        ring_flag = ring_flag0;
        ring_buf_mkey = ring_buf_mkey0;
    } else if (blockIdx.x == 1) {
        qp = qp1;
        ring_buf = ring_buf1;
        ring_buf_stride_sz = ring_buf_stride_sz1;
        ring_buf_stride_num = ring_buf_stride_num1;
        ring_flag = ring_flag1;
        ring_buf_mkey = ring_buf_mkey1;
    } else if (blockIdx.x == 2) {
        qp = qp2;
        ring_buf = ring_buf2;
        ring_buf_stride_sz = ring_buf_stride_sz2;
        ring_buf_stride_num = ring_buf_stride_num2;
        ring_flag = ring_flag2;
        ring_buf_mkey = ring_buf_mkey2;
    } else if (blockIdx.x == 3) {
        qp = qp3;
        ring_buf = ring_buf3;
        ring_buf_stride_sz = ring_buf_stride_sz3;
        ring_buf_stride_num = ring_buf_stride_num3;
        ring_flag = ring_flag3;
        ring_buf_mkey = ring_buf_mkey3;
    }

    // Assuming ring_buf_stride_num is pow(2)
    const uint32_t ring_buf_stride_mask = ring_buf_stride_num - 1;

    // Warmup
    if (qp == nullptr)
        return;

    while (DOCA_GPUNETIO_VOLATILE(*exit_flag) == 0) {
        // Assumption is tx_buf_address holds some address from tx_ring data buffer (same mkey in the pre-prepare)
        // If not 0, the frame data written by the application before the flag are visible
        tx_buf_address = ring_flag_load_acquire(&ring_flag[wqe_idx & ring_buf_stride_mask]);
        if (tx_buf_address == 0)
            continue;

        // Sends are posted in WQE index order: wait for the previous ones, unless exiting
        if (!wait_sq_turn_or_exit(qp, wqe_idx, exit_flag))
            break;

        // No send inline, just send the address value
        if (max_frame_size == GPU_ROCE_MAX_FRAME_SIZE_0B)
            send<max_frame_size>(qp, wqe_idx, tx_buf_address - (uint64_t)ring_buf);
        // Send inline, need buffer address at right offset
        else
            send<max_frame_size>(qp, wqe_idx, tx_buf_address);

        // Release the TX ring slot only now: an inline send has copied the frame into the WQE
        ring_flag_store_release(&ring_flag[wqe_idx & ring_buf_stride_mask], 0);

        wqe_idx += WQE_NUM;
    }
}

__global__ void compute(uint32_t* exit_flag,
    uint64_t* ring_rx_flag0, uint64_t* ring_rx_flag1, uint64_t* ring_rx_flag2, uint64_t* ring_rx_flag3,
    const uint32_t ring_rx_stride_num0, const uint32_t ring_rx_stride_num1, const uint32_t ring_rx_stride_num2, const uint32_t ring_rx_stride_num3,
    uint64_t* ring_tx_flag0, uint64_t* ring_tx_flag1, uint64_t* ring_tx_flag2, uint64_t* ring_tx_flag3,
    uint8_t* ring_tx_addr0, uint8_t* ring_tx_addr1, uint8_t* ring_tx_addr2, uint8_t* ring_tx_addr3,
    const size_t ring_tx_stride_sz0, const size_t ring_tx_stride_sz1, const size_t ring_tx_stride_sz2, const size_t ring_tx_stride_sz3,
    const uint32_t ring_tx_stride_num0, const uint32_t ring_tx_stride_num1, const uint32_t ring_tx_stride_num2, const uint32_t ring_tx_stride_num3,
    size_t frame_size)
{
    uint64_t rx_buf_address;
    uint64_t tx_buf_address;
    uint64_t stride = threadIdx.x;

    uint64_t* ring_rx_flag;
    uint32_t ring_rx_stride_num;
    uint64_t* ring_tx_flag;
    uint8_t* ring_tx_addr;
    size_t ring_tx_stride_sz;
    uint32_t ring_tx_stride_num;

    if (blockIdx.x == 0) {
        ring_rx_flag = ring_rx_flag0;
        ring_rx_stride_num = ring_rx_stride_num0;
        ring_tx_flag = ring_tx_flag0;
        ring_tx_addr = ring_tx_addr0;
        ring_tx_stride_sz = ring_tx_stride_sz0;
        ring_tx_stride_num = ring_tx_stride_num0;
    } else if (blockIdx.x == 1) {
        ring_rx_flag = ring_rx_flag1;
        ring_rx_stride_num = ring_rx_stride_num1;
        ring_tx_flag = ring_tx_flag1;
        ring_tx_addr = ring_tx_addr1;
        ring_tx_stride_sz = ring_tx_stride_sz1;
        ring_tx_stride_num = ring_tx_stride_num1;
    } else if (blockIdx.x == 2) {
        ring_rx_flag = ring_rx_flag2;
        ring_rx_stride_num = ring_rx_stride_num2;
        ring_tx_flag = ring_tx_flag2;
        ring_tx_addr = ring_tx_addr2;
        ring_tx_stride_sz = ring_tx_stride_sz2;
        ring_tx_stride_num = ring_tx_stride_num2;
    } else if (blockIdx.x == 3) {
        ring_rx_flag = ring_rx_flag3;
        ring_rx_stride_num = ring_rx_stride_num3;
        ring_tx_flag = ring_tx_flag3;
        ring_tx_addr = ring_tx_addr3;
        ring_tx_stride_sz = ring_tx_stride_sz3;
        ring_tx_stride_num = ring_tx_stride_num3;
    }

    // Assuming ring_buf_stride_num is pow(2)
    const uint32_t ring_rx_stride_mask = ring_rx_stride_num - 1;
    const uint32_t ring_tx_stride_mask = ring_tx_stride_num - 1;

    // Warmup
    if (ring_rx_flag == nullptr || ring_tx_flag == nullptr)
        return;

    tx_buf_address = ((uint64_t)ring_tx_addr) + ((stride & ring_tx_stride_mask) * (uint64_t)ring_tx_stride_sz);

    while (DOCA_GPUNETIO_VOLATILE(*exit_flag) == 0) {

        // If not 0, the frame data written by the NIC before the flag are visible
        rx_buf_address = ring_flag_load_acquire(&ring_rx_flag[stride & ring_rx_stride_mask]);
        if (rx_buf_address == 0)
            continue;

        // The TX ring slot is free once the TX kernel has sent its previous frame
        while (DOCA_GPUNETIO_VERBS_VOLATILE(ring_tx_flag[stride & ring_tx_stride_mask]) != 0) {
            if (DOCA_GPUNETIO_VOLATILE(*exit_flag) != 0)
                return;
        }
        doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();

        // Copy the first 16 bytes of the frame
        DOCA_GPUNETIO_VOLATILE(((uint64_t*)tx_buf_address)[0]) = DOCA_GPUNETIO_VOLATILE(((uint64_t*)rx_buf_address)[0]);
        DOCA_GPUNETIO_VOLATILE(((uint64_t*)tx_buf_address)[1]) = DOCA_GPUNETIO_VOLATILE(((uint64_t*)rx_buf_address)[1]);

        // Release the RX ring slot after reading the frame, then publish the TX frame
        ring_flag_store_release(&ring_rx_flag[stride & ring_rx_stride_mask], 0);
        ring_flag_store_release(&ring_tx_flag[stride & ring_tx_stride_mask], (uint64_t)tx_buf_address);

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
    uint32_t* exit_flag,
    struct doca_gpu_dev_verbs_qp** qp,
    uint8_t** ring_buf,
    size_t* ring_buf_stride_sz,
    uint32_t* ring_buf_mkey,
    uint32_t* ring_buf_stride_num,
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
            forward_bf<GPU_ROCE_MAX_FRAME_SIZE_0B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                qp[0], qp[1], qp[2], qp[3],
                ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                frame_size);
        else {
            if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_4B)
                forward_bf<GPU_ROCE_MAX_FRAME_SIZE_4B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_8B)
                forward_bf<GPU_ROCE_MAX_FRAME_SIZE_8B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_12B)
                forward_bf<GPU_ROCE_MAX_FRAME_SIZE_12B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_16B)
                forward_bf<GPU_ROCE_MAX_FRAME_SIZE_16B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_20B)
                forward_bf<GPU_ROCE_MAX_FRAME_SIZE_20B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_24B)
                forward_bf<GPU_ROCE_MAX_FRAME_SIZE_24B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_28B)
                forward_bf<GPU_ROCE_MAX_FRAME_SIZE_28B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_32B)
                forward_bf<GPU_ROCE_MAX_FRAME_SIZE_32B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_36B)
                forward_bf<GPU_ROCE_MAX_FRAME_SIZE_36B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_40B)
                forward_bf<GPU_ROCE_MAX_FRAME_SIZE_40B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_44B)
                forward_bf<GPU_ROCE_MAX_FRAME_SIZE_44B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    frame_size);
        }
    } else {
        if (frame_size > MAX_SEND_INLINE_WQE)
            forward<GPU_ROCE_MAX_FRAME_SIZE_0B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                qp[0], qp[1], qp[2], qp[3],
                ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                frame_size);
        else {
            if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_4B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_4B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_8B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_8B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_12B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_12B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_16B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_16B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_20B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_20B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_24B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_24B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_28B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_28B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_32B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_32B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_36B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_36B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_40B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_40B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_44B)
                forward<GPU_ROCE_MAX_FRAME_SIZE_44B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    frame_size);
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
    uint32_t* exit_flag, struct doca_gpu_dev_verbs_qp** qp,
    uint8_t** ring_buf, const size_t* ring_buf_stride_sz,
    const uint32_t* ring_buf_stride_num, uint64_t** ring_flag,
    size_t frame_size, uint32_t cuda_blocks, uint32_t cuda_threads)
{
    cudaError_t result = cudaSuccess;

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (cudaSuccess != result) {
        fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    rx_only<<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag, qp[0], qp[1], qp[2], qp[3],
        ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
        ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
        ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
        ring_flag[0], ring_flag[1], ring_flag[2], ring_flag[3]);

    result = cudaGetLastError();
    if (cudaSuccess != result) {
        fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}

doca_error_t GpuRoceTransceiverTxOnlyKernel(cudaStream_t stream,
    uint32_t* exit_flag, struct doca_gpu_dev_verbs_qp** qp,
    uint8_t** ring_buf, const size_t* ring_buf_stride_sz,
    const uint32_t* ring_buf_stride_num, uint32_t* ring_buf_mkey, uint64_t** ring_flag,
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
            tx_only_bf<GPU_ROCE_MAX_FRAME_SIZE_0B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                qp[0], qp[1], qp[2], qp[3],
                ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                ring_flag[0], ring_flag[1], ring_flag[2], ring_flag[3],
                frame_size);
        else {
            if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_4B)
                tx_only_bf<GPU_ROCE_MAX_FRAME_SIZE_4B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_flag[0], ring_flag[1], ring_flag[2], ring_flag[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_8B)
                tx_only_bf<GPU_ROCE_MAX_FRAME_SIZE_8B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_flag[0], ring_flag[1], ring_flag[2], ring_flag[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_12B)
                tx_only_bf<GPU_ROCE_MAX_FRAME_SIZE_12B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_flag[0], ring_flag[1], ring_flag[2], ring_flag[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_16B)
                tx_only_bf<GPU_ROCE_MAX_FRAME_SIZE_16B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_flag[0], ring_flag[1], ring_flag[2], ring_flag[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_20B)
                tx_only_bf<GPU_ROCE_MAX_FRAME_SIZE_20B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_flag[0], ring_flag[1], ring_flag[2], ring_flag[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_24B)
                tx_only_bf<GPU_ROCE_MAX_FRAME_SIZE_24B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_flag[0], ring_flag[1], ring_flag[2], ring_flag[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_28B)
                tx_only_bf<GPU_ROCE_MAX_FRAME_SIZE_28B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_flag[0], ring_flag[1], ring_flag[2], ring_flag[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_32B)
                tx_only_bf<GPU_ROCE_MAX_FRAME_SIZE_32B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_flag[0], ring_flag[1], ring_flag[2], ring_flag[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_36B)
                tx_only_bf<GPU_ROCE_MAX_FRAME_SIZE_36B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_flag[0], ring_flag[1], ring_flag[2], ring_flag[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_40B)
                tx_only_bf<GPU_ROCE_MAX_FRAME_SIZE_40B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_flag[0], ring_flag[1], ring_flag[2], ring_flag[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_44B)
                tx_only_bf<GPU_ROCE_MAX_FRAME_SIZE_44B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_flag[0], ring_flag[1], ring_flag[2], ring_flag[3],
                    frame_size);
        }
    } else {
        if (frame_size > MAX_SEND_INLINE_WQE)
            tx_only<GPU_ROCE_MAX_FRAME_SIZE_0B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                qp[0], qp[1], qp[2], qp[3],
                ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                ring_flag[0], ring_flag[1], ring_flag[2], ring_flag[3],
                frame_size);
        else {
            if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_4B)
                tx_only<GPU_ROCE_MAX_FRAME_SIZE_4B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_flag[0], ring_flag[1], ring_flag[2], ring_flag[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_8B)
                tx_only<GPU_ROCE_MAX_FRAME_SIZE_8B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_flag[0], ring_flag[1], ring_flag[2], ring_flag[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_12B)
                tx_only<GPU_ROCE_MAX_FRAME_SIZE_12B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_flag[0], ring_flag[1], ring_flag[2], ring_flag[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_16B)
                tx_only<GPU_ROCE_MAX_FRAME_SIZE_16B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_flag[0], ring_flag[1], ring_flag[2], ring_flag[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_20B)
                tx_only<GPU_ROCE_MAX_FRAME_SIZE_20B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_flag[0], ring_flag[1], ring_flag[2], ring_flag[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_24B)
                tx_only<GPU_ROCE_MAX_FRAME_SIZE_24B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_flag[0], ring_flag[1], ring_flag[2], ring_flag[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_28B)
                tx_only<GPU_ROCE_MAX_FRAME_SIZE_28B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_flag[0], ring_flag[1], ring_flag[2], ring_flag[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_32B)
                tx_only<GPU_ROCE_MAX_FRAME_SIZE_32B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_flag[0], ring_flag[1], ring_flag[2], ring_flag[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_36B)
                tx_only<GPU_ROCE_MAX_FRAME_SIZE_36B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_flag[0], ring_flag[1], ring_flag[2], ring_flag[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_40B)
                tx_only<GPU_ROCE_MAX_FRAME_SIZE_40B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_flag[0], ring_flag[1], ring_flag[2], ring_flag[3],
                    frame_size);
            else if (frame_size <= GPU_ROCE_MAX_FRAME_SIZE_44B)
                tx_only<GPU_ROCE_MAX_FRAME_SIZE_44B><<<cuda_blocks, cuda_threads, 0, stream>>>(exit_flag,
                    qp[0], qp[1], qp[2], qp[3],
                    ring_buf[0], ring_buf[1], ring_buf[2], ring_buf[3],
                    ring_buf_stride_sz[0], ring_buf_stride_sz[1], ring_buf_stride_sz[2], ring_buf_stride_sz[3],
                    ring_buf_stride_num[0], ring_buf_stride_num[1], ring_buf_stride_num[2], ring_buf_stride_num[3],
                    ring_buf_mkey[0], ring_buf_mkey[1], ring_buf_mkey[2], ring_buf_mkey[3],
                    ring_flag[0], ring_flag[1], ring_flag[2], ring_flag[3],
                    frame_size);
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
    uint64_t** ring_rx_flag, const uint32_t* ring_rx_stride_num,
    uint64_t** ring_tx_flag, uint8_t** ring_tx_addr, const size_t* ring_tx_stride_sz, const uint32_t* ring_tx_stride_num,
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
        ring_rx_flag[0], ring_rx_flag[1], ring_rx_flag[2], ring_rx_flag[3],
        ring_rx_stride_num[0], ring_rx_stride_num[1], ring_rx_stride_num[2], ring_rx_stride_num[3],
        ring_tx_flag[0], ring_tx_flag[1], ring_tx_flag[2], ring_tx_flag[3],
        ring_tx_addr[0], ring_tx_addr[1], ring_tx_addr[2], ring_tx_addr[3],
        ring_tx_stride_sz[0], ring_tx_stride_sz[1], ring_tx_stride_sz[2], ring_tx_stride_sz[3],
        ring_tx_stride_num[0], ring_tx_stride_num[1], ring_tx_stride_num[2], ring_tx_stride_num[3],
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
