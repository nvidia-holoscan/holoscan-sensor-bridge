/**
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "coe_data_plane.hpp"

namespace hololink::emulation {

void COEDataPlane::update_metadata()
{

    COECtxt* coe_metadata = reinterpret_cast<COECtxt*>(data_plane_ctxt_.get());
    if (coe_metadata->in_use) {
        return;
    }
    struct DPRegisters* dp_reg = coe_metadata->base.base.dp_registers;
    struct DPSensorRegisters* dp_sensor_reg = coe_metadata->base.base.dp_sensor_registers;

    // derived metadata
    coe_metadata->payload_size = static_cast<uint16_t>(0xFFFF & (dp_reg->hif_data[DP_PACKET_SIZE / REGISTER_SIZE] * HSB_PAGE_SIZE));

    coe_metadata->frame_size = dp_sensor_reg->vp_data[DP_BUFFER_LENGTH / REGISTER_SIZE];
    coe_metadata->metadata_offset = (coe_metadata->frame_size + HSB_PAGE_SIZE - 1) & ~(HSB_PAGE_SIZE - 1);
    uint32_t qp = dp_sensor_reg->vp_data[DP_QP / REGISTER_SIZE];
    uint8_t lsh = (qp >> 25u) & 0x7F;
    if (lsh > 31) {
        Error_Handler("Line threshold log 2 is too large");
    }
    coe_metadata->line_threshold = 1u << lsh;
    coe_metadata->enable_1722b = (qp >> 24u) & 0x1;
    coe_metadata->channel = qp & 0x3Fu;
    coe_metadata->line_offset = 0;
    coe_metadata->address = 0;
}

} // namespace hololink::emulation
