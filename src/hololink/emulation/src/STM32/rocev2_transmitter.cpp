/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>

#include "dlpack/dlpack.h"

#include "../common/rocev2_transmitter.hpp"
#include "STM32/net.hpp"
#include "base_transmitter.hpp"
#include "rocev2_data_plane.hpp"

namespace hololink::emulation {

// DLTensor handling has moved up to DataPlane::send (platform-specific). STM32 has no
// CUDA so its DataPlane::send just forwards (uint8_t*)tensor.data straight through.
// eth_hal_send() is declared in STM32/net.hpp (included transitively above).
int16_t send_rocev2_packet(RoCEv2Ctxt* rocev2_ctxt, ETH_BufferTypeDef* tx_buffers)
{
    return eth_hal_send(rocev2_ctxt->base.eth_handle, tx_buffers);
}

} // namespace hololink::emulation
