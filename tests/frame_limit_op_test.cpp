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

// FrameLimitOp (examples/player_support.hpp), the sub-frame players' --frame-limit. Counting
// frame_number changes, it counts a frame when a later frame starts: the bands of one frame count
// once, and a late band of an earlier frame, which arrives after the next frame has started,
// counts nothing. Needs no GPU.

#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include <capture_metadata_traits.hpp>

#define HOLOLINK_PLAYER_SUPPORT_NO_DISPLAY
#include "player_support.hpp"

namespace hololink::tests {

namespace {

    namespace hx = hololink::examples;

    // Emits one band's CaptureMetadataT per activation, with the given frame_numbers in order, then
    // completes.
    class BandMetadataSourceOp : public holoscan::Operator<> {
    public:
        explicit BandMetadataSourceOp(std::vector<int64_t> frame_numbers)
            : frame_numbers_(std::move(frame_numbers))
        {
        }

        holoscan::Output<hololink::CaptureMetadataT> metadata { "metadata" };

        void setup(holoscan::OperatorSpec& spec) override { spec.output(metadata, "metadata").max_emits_per_compute(1U); }

        [[nodiscard]] holoscan::Contract contract() const override
        {
            holoscan::Contract contract;
            contract.trigger(holoscan::OnClock { .period = std::chrono::milliseconds(1) });
            contract.complete_after(holoscan::ComputeCalls { .limit = frame_numbers_.size() });
            return contract;
        }

        holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override
        {
            hololink::CaptureMetadataT band;
            band.frame_number = frame_numbers_.at(next_++);
            const holoscan::EmitOptions options {
                .capture_time = context.activation_time(),
                .frame_id = static_cast<uint64_t>(band.frame_number),
            };
            return metadata.emit(std::move(band), options);
        }

    private:
        std::vector<int64_t> frame_numbers_;
        std::size_t next_ = 0;
    };

    // Runs bands with \p frame_numbers into a FrameLimitOp counting frame_number changes, limited to
    // \p limit frames, and returns the termination: kRequestedCompleted once the limit is reached;
    // otherwise the run goes on until it is stopped, well after the source has finished (FrameLimitOp
    // drains its queue every 16 ms).
    holoscan::RunTermination run_frame_limit(std::vector<int64_t> frame_numbers, int64_t limit)
    {
        holoscan::Graph graph { "frame_limit_op_test" };
        auto source = graph.op<BandMetadataSourceOp>("source", std::move(frame_numbers));
        hx::add_frame_limit(graph, "limit", source->metadata, std::make_shared<hx::FrameLimitGroup>(limit, 1),
            /*count_frame_number_changes=*/true);
        graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
        auto plan = holoscan::compile(graph);
        if (!plan.ok()) {
            holoscan::log_diagnostics(plan);
            ADD_FAILURE() << "graph compilation failed";
            return holoscan::RunTermination::kFailed;
        }
        auto session = holoscan::run_async(plan);
        (void)session.wait_for(std::chrono::milliseconds(500));
        session.request_stop();
        session.wait();
        for (const auto& diagnostic : session.diagnostics()) {
            if (diagnostic.severity == holoscan::DiagnosticSeverity::kError) {
                ADD_FAILURE() << holoscan::format_diagnostic(diagnostic, plan.graph());
            }
        }
        return session.termination();
    }

} // namespace

// Frames 0, 1 and 2 count once frame 3 starts; the late band of frame 0 between frame 1's bands
// counts nothing.
TEST(FrameLimitOpTest, CountsAFrameWhenALaterFrameStarts)
{
    EXPECT_EQ(run_frame_limit({ 0, 0, 1, 0, 1, 2, 2, 3 }, 3), holoscan::RunTermination::kRequestedCompleted);
}

// Until frame 3 starts only frames 0 and 1 are complete, so the limit must not end the run, though
// the frame_number changed four times (counting the late band of frame 0 both ways).
TEST(FrameLimitOpTest, DoesNotCountALateBandOfAnEarlierFrame)
{
    const auto termination = run_frame_limit({ 0, 0, 1, 0, 1, 2, 2 }, 3);
    EXPECT_NE(termination, holoscan::RunTermination::kRequestedCompleted) << "the late band counted as a frame";
    EXPECT_NE(termination, holoscan::RunTermination::kFailed);
}

} // namespace hololink::tests
