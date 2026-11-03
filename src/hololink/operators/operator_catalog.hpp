// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <holoscan/authoring/constructor_handle.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

#include <memory>
#include <vector>

#include <hololink/core/data_channel.hpp>

namespace hololink::operators {

/// @brief Install the hololink Operator rows into the process-local authoring registry.
///
/// Process-idempotent and safe to call from every executable or language module that links
/// this library. Each row is generated from the operator's canonical `Params` schema and
/// construction thunk (automatic for schema-only operators, hand-written for operators with
/// non-trivial constructors).
/// @return Success, or the first registration failure retained across repeated calls.
holoscan::expected<void, holoscan::Error> register_hololink_operator_catalog();

/// @brief The constructor-handle type of a `hololink::DataChannel`, stable name
/// `hololink.DataChannel`.
///
/// The `data_channel` input of the LinuxReceiverOp, LinuxCoeReceiverOp, RoceReceiverOp,
/// GpuRoceTransceiverOp and FusaCoeCaptureOp rows has this type. Python's
/// `hololink.DataChannel` exports its handle through it, so an application passes the channel
/// itself: `graph.op(hololink.operators.RoceReceiverOp, "rx", data_channel=channel, ...)`. The
/// descriptor is defined once, in this library.
[[nodiscard]] const holoscan::authoring::ConstructorHandleType<DataChannel>&
data_channel_handle_type();

/// @brief The DataChannels of a GpuRoceTransceiverOp's sensors after its `data_channel`'s, one
/// per QP of its `tx_ibv_qp` after the first, in the same order (Python: `hololink.DataChannels`).
struct DataChannels : std::enable_shared_from_this<DataChannels> {
    std::vector<std::shared_ptr<DataChannel>> channels;
};

/// @brief The constructor-handle type of DataChannels, stable name `hololink.DataChannels`: the
/// GpuRoceTransceiverOp row's optional `extra_data_channels` input.
[[nodiscard]] const holoscan::authoring::ConstructorHandleType<DataChannels>&
data_channels_handle_type();

struct CrcCheckChannel;

// Handle type of hololink.CrcChannel, shared by ComputeCrcOp and CheckCrcOp (crc_channel=...).
[[nodiscard]] const holoscan::authoring::ConstructorHandleType<CrcCheckChannel>&
crc_channel_handle_type();

} // namespace hololink::operators
