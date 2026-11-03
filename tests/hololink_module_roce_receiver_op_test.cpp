/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Construction smoke test for the RoceReceiverOp adapter operator, ported to Holoscan SDK
 * 5.0's Graph / Operator<Params> shape. RoceReceiverOp resolves its
 * RoceDataChannelInterfaceV1 + receiver + FrameMetadataInterfaceV1 services from the
 * enumeration metadata at start() time via the Adapter singleton; this test only verifies the
 * operator can be constructed with the minimal Params set and that the graph compiles. It
 * does not drive start() / stop() — that path requires a loaded supplement module, real
 * ibverbs, and an HSB peer.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <iostream>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/operators/roce_receiver_op.hpp"

TEST(HololinkAdapterRoceReceiverOp, ConstructsAndComposes)
{
    holoscan::Graph graph { "hololink_module_roce_receiver_op_test" };

    auto rx = graph.op<hololink::module::operators::RoceReceiverOp>(
        "roce_receiver", hololink::module::EnumerationMetadata {}, [] {}, [] {});
    rx.set_params(hololink::module::operators::RoceReceiverParams {
        .frame_size = std::size_t { 4 * 1024 * 1024 },
        .page_size = std::size_t { 4 * 1024 * 1024 },
        .pages = std::uint32_t { 2 },
    });

    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

    holoscan::CompileOptions options;
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement {
        .operator_path = "roce_receiver",
        .output_port = "tensor",
        .device = holoscan::DeviceId { 0 },
    });

    auto plan = holoscan::compile(graph, std::move(options));
    if (!plan.ok()) {
        for (const auto& diag : plan.diagnostics()) {
            std::cerr << "Diagnostic: token=" << diag.token
                      << " required=" << diag.required
                      << " provided=" << diag.provided << "\n";
        }
    }
    EXPECT_TRUE(plan.ok()) << "Graph compilation failed";
}
