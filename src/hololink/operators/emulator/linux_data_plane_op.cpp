/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

// need to include this before holoscan components otherwise compiler complains that ucx.h uses undefined sockaddr_storage
#include <sys/socket.h>

#include "hololink/core/logging_internal.hpp"
#include "hololink/emulation/hsb_emulator.hpp"
#include "hololink/emulation/linux_data_plane.hpp"
#include "holoscan/core/domain/tensor.hpp"
#include "linux_data_plane_op.hpp"

namespace hololink::operators {

#define LINUX_DATA_PLANE_INPUT_NAME "input"

void LinuxDataPlaneOp::setup(holoscan::OperatorSpec& spec)
{
    spec.input<holoscan::Tensor>(LINUX_DATA_PLANE_INPUT_NAME);
    spec.param(hsb_emulator_, "hsb_emulator", "HSBEmulator", "HSBEmulator", std::make_shared<hololink::emulation::HSBEmulator>());
    spec.param(source_ip_address_, "source_ip_address", "IPAddress", "IPAddress", std::string("192.168.0.2"));
    spec.param(subnet_bits_, "subnet_bits", "subnet ibts", "Number of Subnet bits", 24u);
    spec.param(source_port_, "source_port", "source port", "Source port", 12888u);
    spec.param(sensor_id_, "sensor_id", "SensorID", "SensorID", hololink::emulation::SensorID::SENSOR_0);
    spec.param(data_plane_id_,
        "data_plane_id",
        "DataPlaneID",
        "DataPlaneID",
        hololink::emulation::DataPlaneID::DATA_PLANE_0);
}

void LinuxDataPlaneOp::initialize()
{
    Operator::initialize(); // let holoscan bind parameters first
    data_plane_ = std::make_unique<hololink::emulation::LinuxDataPlane>(
        *hsb_emulator_.get(),
        hololink::emulation::IPAddress_from_string(source_ip_address_.get(), subnet_bits_.get()),
        source_port_.get(),
        data_plane_id_.get(),
        sensor_id_.get());
}

void LinuxDataPlaneOp::compute(holoscan::InputContext& op_input, [[maybe_unused]] holoscan::OutputContext& op_output,
    [[maybe_unused]] holoscan::ExecutionContext& context)
{
    auto maybe_tensor = op_input.receive<holoscan::Tensor>(LINUX_DATA_PLANE_INPUT_NAME);
    if (!maybe_tensor) {
        HSB_LOG_ERROR("Failed to receive message from port '{}'", LINUX_DATA_PLANE_INPUT_NAME);
        throw std::exception();
    }

    auto tensor = maybe_tensor.value();
    DLManagedTensor* dl_managed_tensor = tensor.to_dlpack();
    data_plane_->send(dl_managed_tensor->dl_tensor);
    dl_managed_tensor->deleter(dl_managed_tensor);
}

} // namespace hololink::operators
