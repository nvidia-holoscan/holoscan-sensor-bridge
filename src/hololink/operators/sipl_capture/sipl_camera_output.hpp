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

#include <cstdint>
#include <memory>

#include <holoscan/holoscan.hpp>

#include "sipl_capture_service.hpp"

namespace hololink::operators {

/**
 * @brief Holoscan operator that outputs frames for one camera managed by SIPLCaptureService.
 */
class SIPLCameraOutputOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(SIPLCameraOutputOp)

    template <typename... ArgsT>
    explicit SIPLCameraOutputOp(std::shared_ptr<SIPLCaptureService> service, uint32_t camera_index, ArgsT&&... args)
        : holoscan::Operator(std::forward<ArgsT>(args)...)
        , service_(std::move(service))
        , camera_index_(camera_index)
    {
    }

    void setup(holoscan::OperatorSpec& spec) override;
    void start() override;
    void stop() override;
    void compute(holoscan::InputContext& op_input,
        holoscan::OutputContext& op_output,
        holoscan::ExecutionContext& context) override;

private:
    std::shared_ptr<SIPLCaptureService> service_;
    uint32_t camera_index_;

    holoscan::Parameter<uint32_t> timeout_;
};

} // namespace hololink::operators
