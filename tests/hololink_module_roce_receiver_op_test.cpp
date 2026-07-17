/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Construction smoke test for the RoceReceiverOp adapter operator.
 * RoceReceiverOp resolves its RoceDataChannelInterfaceV1 + receiver +
 * FrameMetadataInterfaceV1 services from the enumeration metadata at
 * start() time via the Adapter singleton; this test only verifies the
 * operator can be constructed with the minimal Arg set, that
 * compose_graph() succeeds, and that the C++ symbol is reachable.
 * It does not drive start() / stop() — that path requires a loaded
 * supplement module, real ibverbs, and an HSB peer.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>

#include <holoscan/holoscan.hpp>

#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/operators/roce_receiver_op.hpp"

namespace {

class RoceReceiverOpTestApp : public holoscan::Application {
public:
    std::shared_ptr<hololink::module::operators::RoceReceiverOp> op() const { return op_; }

    void compose() override
    {
        op_ = make_operator<hololink::module::operators::RoceReceiverOp>(
            "roce_receiver",
            holoscan::Arg("enumeration_metadata",
                hololink::module::EnumerationMetadata {}),
            holoscan::Arg("frame_size", std::size_t { 4 * 1024 * 1024 }),
            holoscan::Arg("page_size", std::size_t { 4 * 1024 * 1024 }),
            holoscan::Arg("pages", std::uint32_t { 2 }));
        add_operator(op_);
    }

private:
    std::shared_ptr<hololink::module::operators::RoceReceiverOp> op_;
};

} // namespace

TEST(HololinkAdapterRoceReceiverOp, ConstructsAndComposes)
{
    auto app = holoscan::make_application<RoceReceiverOpTestApp>();
    app->compose_graph();

    EXPECT_NE(app->op(), nullptr);
}
