/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef STM32_HSB_CONFIG_HPP
#define STM32_HSB_CONFIG_HPP

#include "../../hsb_config.hpp"
#include "STM32/net.hpp"

// control message addr_vals is just the location in memory of the addresses/values, but should never be accessed directly
// as they are not aligned accesses. Must explicitly set the address or value using the macros below.
#define AVP_GET_ADDRESS(avp) ((avp)->address)
#define AVP_SET_ADDRESS(avp, address_) ((avp)->address = address_)
#define AVP_GET_VALUE(avp) ((avp)->value)
#define AVP_SET_VALUE(avp, value_) ((avp)->value = value_)

#endif
