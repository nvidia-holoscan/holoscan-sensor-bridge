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

#include <algorithm>
#include <infiniband/verbs.h>
#include <string>
#include <vector>

namespace hololink {

std::vector<std::string> infiniband_devices()
{
    int num_devices = 0;
    struct ibv_device** devices = ibv_get_device_list(&num_devices);
    if (!devices) {
        return {};
    }
    std::vector<std::string> device_names;
    for (int i = 0; i < num_devices; i++) {
        device_names.push_back(ibv_get_device_name(devices[i]));
    }
    ibv_free_device_list(devices);
    std::sort(device_names.begin(), device_names.end());
    return device_names;
}

} // namespace hololink
