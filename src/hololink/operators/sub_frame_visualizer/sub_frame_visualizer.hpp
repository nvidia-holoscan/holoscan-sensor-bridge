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

#include <mutex>
#include <thread>

#include <hololink/core/hololink.hpp>
#include <holoscan/holoscan.hpp>
#include <holoviz/holoviz.hpp>

namespace hololink::operators {

/**
 * @brief Operator class to accumulate and visualize sub-frames as a complete frame
 */
class SubFrameVisualizerOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(SubFrameVisualizerOp)

    void setup(holoscan::OperatorSpec& spec) override;
    void start() override;
    void stop() override;
    void compute(holoscan::InputContext& op_input,
        holoscan::OutputContext& op_output,
        holoscan::ExecutionContext& context) override;

private:
    /**
     * Background thread that waits for the display First Pixel Out (FPO) event and
     * schedules the next FPGA camera capture via PTP/PPS set_delay(), synchronizing
     * capture timing to the display refresh rate.
     */
    void thread_func();

    holoscan::Parameter<bool> fullscreen_;
    holoscan::Parameter<bool> use_exclusive_display_;
    holoscan::Parameter<std::string> display_name_;
    holoscan::Parameter<uint32_t> display_width_;
    holoscan::Parameter<uint32_t> display_height_;
    holoscan::Parameter<uint32_t> display_framerate_;
    holoscan::Parameter<std::string> window_title_;
    holoscan::Parameter<uint32_t> full_frame_height_;
    holoscan::Parameter<Hololink::PtpSynchronizer*> ptp_synchronizer_;

    std::shared_ptr<holoscan::BooleanCondition> window_close_condition_;

    holoscan::viz::InstanceHandle instance_;

    std::thread thread_;
    std::atomic<bool> stop_requested_ = false;

    std::mutex render_start_time_mutex_;
    /**
     * Pipeline time anchor at first sub-frame: sensor frame start from timestamp_s/ns when present (PTP-aligned
     * with FPGA in typical stacks), otherwise CLOCK_REALTIME when compute runs. FPO thread uses this for latency math.
     */
    timespec render_start_time_ = { 0, 0 };

    std::atomic<bool> fpo_available_ { true };
};

} // namespace hololink::operators
