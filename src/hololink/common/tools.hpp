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

#ifndef SRC_HOLOLINK_TOOLS
#define SRC_HOLOLINK_TOOLS

#include <cstddef>
#include <string>
#include <vector>

namespace hololink {

// Return a sorted vector of Infiniband devices.
std::vector<std::string> infiniband_devices();

// Returns the i-th entry of the comma-separated HOLOLINK_IPS env
// var, or `fallback` when the env var is unset, empty, or has too
// few entries. Lets `ctest` pass per-test board IPs through without
// each test CMakeLists threading its own --hololink flag — matches
// the `HOLOLINK_IPS=...` shape that the pytest `--channel-ips`
// option also accepts via PYTEST_ADDOPTS.
std::string env_hololink_ip(std::size_t index, const std::string& fallback);

} // namespace hololink

#endif /* SRC_HOLOLINK_TOOLS */
