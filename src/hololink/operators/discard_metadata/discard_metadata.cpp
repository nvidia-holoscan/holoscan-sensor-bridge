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

#include "discard_metadata.hpp"

#include <chrono>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator_spec.hpp>

namespace hololink::operators {

void DiscardMetadataOp::setup(holoscan::OperatorSpec& spec)
{
    spec.input(input_metadata, "metadata").queue_depth(4U);
}

holoscan::Contract DiscardMetadataOp::contract() const
{
    holoscan::Contract c;
    c.trigger(holoscan::OnClock { .period = std::chrono::milliseconds(16) })
        .input(input_metadata, holoscan::DrainAll {});
    return c;
}

holoscan::expected<void, holoscan::Error> DiscardMetadataOp::compute(holoscan::ExecutionContext&)
{
    return {};
}

} // namespace hololink::operators
