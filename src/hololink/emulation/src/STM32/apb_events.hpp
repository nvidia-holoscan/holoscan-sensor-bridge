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

#ifndef STM32_APB_EVENTS_H
#define STM32_APB_EVENTS_H

#include "hsb_config.hpp"

namespace hololink::emulation {

/**
 * @brief Callback function for asynchronous event readback.
 *
 * @param ctxt The context pointer.
 * @param addr_val The address-value pair.
 * @param max_count The maximum number of address-value pairs to read.
 *
 * @return The number of address-value pairs read.
 */
int async_event_readback_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count);

/**
 * @brief Callback function for asynchronous event configuration.
 *
 * @param ctxt The context pointer.
 * @param addr_val The address-value pair.
 * @param max_count The maximum number of address-value pairs to configure.
 *
 * @return The number of address-value pairs configured.
 */
int async_event_configure_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count);

/**
 * @brief Read from APB RAM.
 *
 * @param ctxt The context pointer.
 * @param addr_val The address-value pair.
 * @param max_count The maximum number of address-value pairs to read.
 *
 * @return The number of address-value pairs read.
 */
int apb_ram_read(void* ctxt, struct AddressValuePair* addr_val, int max_count);

/**
 * @brief Write to APB RAM.
 *
 * @param ctxt The context pointer.
 * @param addr_val The address-value pair.
 * @param max_count The maximum number of address-value pairs to write.
 *
 * @return The number of address-value pairs written.
 */
int apb_ram_write(void* ctxt, struct AddressValuePair* addr_val, int max_count);

/**
 * @brief Handle APB SW event triggered by the host.
 *
 * @param ctxt The context pointer.
 * @param addr_val The address-value pair.
 * @param max_count The maximum number of address-value pairs to handle.
 *
 * @return The number of address-value pairs handled.
 */
int handle_apb_sw_event(void* ctxt, struct AddressValuePair* addr_val, int max_count);

}

#endif