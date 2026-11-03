/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_EXAMPLES_VSYNC_START_OP_HPP
#define HOLOLINK_EXAMPLES_VSYNC_START_OP_HPP

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "hololink/module/status.h"
#include "hololink/module/vsync.hpp"

struct VsyncStartParams {
    // Required: the Parameter is declared without a default in the HSDK4 original, so
    // omitting it there failed operator initialization. Params has no such enforcement --
    // compute() checks for null explicitly instead.
    std::shared_ptr<hololink::module::VsyncInterfaceV1> vsync;
};

/* One-shot holoscan operator whose compute() fires vsync->start(), activated by a short OnClock
 * tick (there is nothing to trigger on otherwise). Like the HSDK4 original's CountCondition(1),
 * Contract::complete_after(ComputeCalls{1}) completes it after that one call, so it doesn't keep
 * the run from completing at --frame-limit. HSDK5 accepts complete_after() only for an operator
 * with at least one output, hence the "done" output, which is never connected or emitted.
 *
 * Holoscan invokes compute() only after every operator's start() has run -- including the
 * RoceReceiverOp::device_start lambdas that arm each camera. By the time this first fires,
 * the cameras are in external-sync waiting state, so the first VSYNC pulse lands on all of
 * them simultaneously and frame N on each receiver tracks the same trigger edge. */
class VsyncStartOp : public holoscan::Operator<VsyncStartParams> {
public:
    holoscan::Output<std::uint32_t> done { "done" };

    void setup(holoscan::OperatorSpec& spec) override { spec.output(done, "done"); }

    [[nodiscard]] holoscan::Contract contract() const override
    {
        holoscan::Contract c;
        c.trigger(holoscan::OnClock { .period = std::chrono::milliseconds(1) })
            .complete_after(holoscan::ComputeCalls { .limit = 1U });
        return c;
    }

    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
        holoscan::ExecutionContext& /*context*/) override
    {
        const auto& p = params();
        if (!p.vsync) {
            return holoscan::make_unexpected(holoscan::Error {
                holoscan::ErrorCode::kInvalidArgument, "VsyncStartOp: vsync was not set" });
        }
        const hololink_module_status_t status = p.vsync->start();
        if (status != HOLOLINK_MODULE_OK) {
            return holoscan::make_unexpected(holoscan::Error { holoscan::ErrorCode::kFailure,
                "While starting VSYNC pulses from VsyncStartOp: status "
                    + std::to_string(static_cast<int>(status)) });
        }
        return {};
    }
};

#endif // HOLOLINK_EXAMPLES_VSYNC_START_OP_HPP
