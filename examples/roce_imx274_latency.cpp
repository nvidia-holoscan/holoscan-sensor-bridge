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

// C++ port of the HSDK4 tool imx274_latency.py: IMX274 latency over the RoCE receiver (see
// imx274_latency.hpp).

#include <cstddef>
#include <memory>
#include <utility>

#include <cuda.h>

#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>

#include <hololink/core/data_channel.hpp>
#include <hololink/core/logging_internal.hpp>
#include <hololink/operators/roce_receiver/roce_receiver_op.hpp>
#include <hololink/sensors/camera/imx274/native_imx274_sensor.hpp>

#include "imx274_latency.hpp"
#include "player_support.hpp"

namespace latency = hololink::examples::imx274_latency;

int main(int argc, char** argv)
{
    const latency::Options options = latency::parse_options(argc, argv, /*roce=*/true);
    return hololink::examples::run_with_managed_signals([&](holoscan::ManagedShutdownSignals& signals) {
        return latency::run(
            options, "roce_imx274_latency",
            [&options](holoscan::Graph& graph, std::shared_ptr<hololink::DataChannel> channel,
                std::shared_ptr<hololink::sensors::NativeImx274Sensor> camera, CUcontext cu_context,
                std::size_t frame_size) {
                HSB_LOG_INFO("frame_size={}", frame_size);
                auto receiver = graph.op<hololink::operators::RoceReceiverOp>(
                    "receiver", std::move(channel), [camera] { camera->start(); },
                    [camera] { latency::stop_camera(*camera); });
                receiver.set_params(hololink::operators::RoceReceiverParams {
                    .frame_context = cu_context,
                    .frame_size = frame_size,
                    .ibv_name = options.ibv_name,
                    .ibv_port = options.ibv_port,
                });
                return receiver;
            },
            signals);
    });
}
