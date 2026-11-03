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

#include <capture_metadata_traits.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>

namespace hololink::operators {

// Drains a CaptureMetadataT port that nothing downstream consumes. A connected port
// backpressures once its queue is full, so something must keep draining it or the producer's
// emit() eventually fails outright on every frame -- e.g. ImageProcessorOp's metadata output in
// a pipeline that only cares about its tensor output (see examples/fusa_coe_vb1940_player.cpp).
// Originally local to that example; promoted here since the same need recurs in any
// tensor-only-consumer pipeline built from hololink's Tensor+CaptureMetadataT-port operators.
//
// Clock-triggered + DrainAll, not OnEach + receive(): this matches the SDK's own drain-sink
// pattern (see LateBindingTensorSink in holoscan-sdk's rate_aware_ports example). An
// OnEach-triggered sink ties its own activation directly to the producer's per-sample rate,
// which -- on real FUSA CoE hardware -- caused the upstream chain to stall after a handful of
// frames; OnClock decouples this sink from that rate entirely, matching how HolovizOp itself
// stays decoupled from its own producers via ClockTrigger.
class DiscardMetadataOp : public holoscan::Operator<> {
public:
    holoscan::Input<hololink::CaptureMetadataT> input_metadata { "metadata" };

    void setup(holoscan::OperatorSpec& spec) override;
    [[nodiscard]] holoscan::Contract contract() const override;
    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
        holoscan::ExecutionContext& context) override;
};

} // namespace hololink::operators
