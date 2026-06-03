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

#ifndef SRC_HOLOLINK_OPERATORS_GPU_ROCE_TRANSCEIVER_COMMON_HPP
#define SRC_HOLOLINK_OPERATORS_GPU_ROCE_TRANSCEIVER_COMMON_HPP

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <doca_error.h>

#define RWQE_TERMINATE_KEY 0x100
#define DSEG_SIZE_2 12
#define DSEG_SIZE_3 28
#define MAX_SEND_INLINE_WQE 44
#define GPU_ROCE_TRANSCEIVER_DEBUG 0

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
#define DOCA_SEND_BLUE_FLAME 1
#define DOCA_SIMULATE_RX_COMPUTE_TX 0

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

#endif // SRC_HOLOLINK_OPERATORS_GPU_ROCE_TRANSCEIVER_COMMON_HPP
