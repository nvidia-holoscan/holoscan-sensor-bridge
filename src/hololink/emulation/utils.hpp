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

#ifndef EMULATION_UTILS_HPP
#define EMULATION_UTILS_HPP

#include <cstdint>

#include "dlpack/dlpack.h"

namespace hololink::emulation {

#define UUID_SIZE 16

// convenience functions/wrappings for DLPack.

// Determine the total number of bytes in the tensor
int64_t DLTensor_n_bytes(const DLTensor& tensor);

typedef uint8_t uuid_t[UUID_SIZE];

// interface and semantics to match standard uuid library
// maps input null-terminated string in human-readable format to byte string
// uuid_s must be 37 characters long (32 hex characters + 4 hyphens + null terminator)
// returns 0 on success, 1 on failure. Failure puts uuid in indeterminate state with possibly partial writes.
int uuid_parse(const char* uuid_s, uuid_t uuid);

// out must be at least 37 characters long (32 hex characters + 4 hyphens + null terminator)
void uuid_unparse_lower(uuid_t uu, char* out);

} // namespace hololink::emulation

#endif
