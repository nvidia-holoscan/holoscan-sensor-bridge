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

#ifndef SRC_HOLOLINK_OPERATORS_OUTPUT_PRESSURE_HPP
#define SRC_HOLOLINK_OPERATORS_OUTPUT_PRESSURE_HPP

#include <cstdint>

#include <holoscan/core/errors.hpp>

#include <hololink/core/logging_internal.hpp>

namespace hololink::operators {

/**
 * HSDK5 never blocks an emit: when an output's runtime pool or a downstream queue is full,
 * allocate_tensor() fails with kResourceExhausted and emit() with kBackpressured, and returning
 * either from compute() fails the run. True for those outcomes -- and for a fan-out publication
 * some recipients may already have observed, which must never be retried -- so the operator can
 * drop the output and carry on instead.
 */
[[nodiscard]] inline bool is_output_pressure(const holoscan::Error& error) noexcept
{
    return (error.code == holoscan::ErrorCode::kResourceExhausted)
        || (error.code == holoscan::ErrorCode::kBackpressured)
        || holoscan::publication_error_consumes_authority(error.code);
}

/// Counts a dropped output in \p dropped, logging the first and then every 1000th.
inline void record_dropped_output(const char* name, uint64_t& dropped, const holoscan::Error& error)
{
    if ((dropped++ % 1000) == 0) {
        HSB_LOG_WARN("{}: dropped an output under backpressure ({} so far): {}", name, dropped,
            holoscan::runtime_error_message(error));
    }
}

} // namespace hololink::operators

#endif /* SRC_HOLOLINK_OPERATORS_OUTPUT_PRESSURE_HPP */
