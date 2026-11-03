/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "compute_crc.hpp"

#include <optional>

#include "compute_crc.inl"

namespace hololink::operators {

// Explicit instantiation for both supported payload types -- compute_crc is a STATIC library
// (see CMakeLists.txt), so these instantiations, plus the method bodies pulled in via
// compute_crc.inl above, are what actually emits ComputeCrcOp<T> object code into
// libcompute_crc.a. Any PayloadT beyond these two needs its own instantiation here.
template class ComputeCrcOp<holoscan::Tensor>;
template class ComputeCrcOp<holoscan::schema::ImageT>;

CheckCrcOp::CheckCrcOp(std::shared_ptr<CrcChannel> crc_channel)
    : crc_channel_(std::move(crc_channel))
{
}

void CheckCrcOp::setup(holoscan::OperatorSpec& spec)
{
    spec.input(input_metadata, "in_metadata");
    spec.output(output_metadata, "metadata");
}

holoscan::Contract CheckCrcOp::contract() const
{
    holoscan::Contract result;
    result.trigger(holoscan::OnEach { input_metadata });
    return result;
}

holoscan::expected<void, holoscan::Error> CheckCrcOp::compute(holoscan::ExecutionContext& /*context*/)
{
    auto sample = input_metadata.receive();
    if (!sample) {
        // kNotReady: activated without new metadata.
        if (sample.error().code == holoscan::ErrorCode::kNotReady) {
            return {};
        }
        return holoscan::make_unexpected(std::move(sample).error());
    }
    if (!crc_channel_) {
        return holoscan::make_unexpected(holoscan::Error {
            holoscan::ErrorCode::kInvalidArgument,
            "CheckCrcOp needs the CrcChannel its ComputeCrcOp was given" });
    }

    const int64_t frame_number = sample->data.frame_number;
    if (const std::optional<uint32_t> computed_crc = crc_channel_->take(frame_number)) {
        check_crc(*computed_crc, sample->data);
    } else if (const uint64_t missing = missing_crcs_.count()) {
        // The frame was dropped before ComputeCrcOp saw it.
        HSB_LOG_WARN("CheckCrcOp: no CRC for {} frame(s) in the last second (latest frame {})",
            missing, frame_number);
    }

    auto emitted = hololink::operators::emit_metadata(output_metadata, sample->data,
        hololink::operators::propagate_emit_options(sample->metadata, static_cast<uint64_t>(frame_number)));
    if (!emitted) {
        return holoscan::make_unexpected(std::move(emitted).error());
    }
    if (*emitted == hololink::operators::PairedEmit::kDropped) {
        if (const uint64_t dropped = dropped_frames_.count()) {
            HSB_LOG_WARN("CheckCrcOp: dropped {} frame(s) in the last second because downstream "
                         "was behind",
                dropped);
        }
    }
    return {};
}

void CheckCrcOp::check_crc(uint32_t computed_crc)
{
    last_computed_crc_ = computed_crc;
}

void CheckCrcOp::check_crc(uint32_t computed_crc, const hololink::CaptureMetadataT& /*metadata*/)
{
    check_crc(computed_crc);
}

CallbackCheckCrcOp::CallbackCheckCrcOp(std::shared_ptr<CrcCheckChannel> crc_channel)
    : CheckCrcOp(crc_channel->channel)
    , crc_check_(std::move(crc_channel))
{
}

void CallbackCheckCrcOp::check_crc(uint32_t computed_crc, const hololink::CaptureMetadataT& metadata)
{
    CheckCrcOp::check_crc(computed_crc, metadata);
    if (crc_check_->check) {
        crc_check_->check(computed_crc, metadata.crc, metadata.frame_number);
    }
}

} // namespace hololink::operators
