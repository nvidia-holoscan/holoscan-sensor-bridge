/*
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
 */

#pragma once

#include <string>

namespace hololink {

/**
 * Flash an HSB_Leopard device with CPNX firmware.
 * Uses the modern Hololink library connection with full verification.
 *
 * @param ip_address IP address of the device
 * @param cpnx_path Path to CPNX firmware file
 * @return true if flash succeeded, false otherwise
 */
bool hsb_leopard_flash_2507(const std::string& ip_address,
    const std::string& cpnx_path);

} // namespace hololink
