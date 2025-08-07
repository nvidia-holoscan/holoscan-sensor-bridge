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

#ifndef OPERATORS_EMULATOR_HSB_EMULATOR_OP_HPP
#define OPERATORS_EMULATOR_HSB_EMULATOR_OP_HPP

#include <memory>
#include <string>

#include "holoscan/holoscan.hpp"

#include "hololink/emulation/linux_data_plane.hpp"

namespace hololink::operators {

/***
 * The LinuxDataPlaneOp operator emulates a Holoscan Sensor Bridge with a single LinuxDataPlane
 *
 * Parameters:
 *  'enum_version' - The enum version of the HSB sensor. Default is "ENUM_VERSION_2".
 *  'ip_address'   - The IP address of the HSB sensor. Default is "192.168.0.2".
 *  'data_plane_id' - The data plane ID of the HSB sensor (DATA_PLANE_0 or DATA_PLANE_1). Default is "DATA_PLANE_0".
 *  'sensor_id'     - The sensor ID of the HSB sensor (SENSOR_0 or SENSOR_1). Default is "SENSOR_0".
 *  'source_port'   - The source port of the HSB sensor. Default is "12888".
 *  'subnet_bits'   - The subnet bits of the HSB sensor. Default is "24".
 */
class LinuxDataPlaneOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(LinuxDataPlaneOp)

    LinuxDataPlaneOp() = default;

    void setup(holoscan::OperatorSpec& spec) override;

    void initialize() override;

    void compute(holoscan::InputContext& op_input, [[maybe_unused]] holoscan::OutputContext& op_output,
        [[maybe_unused]] holoscan::ExecutionContext& context) override;

private:
    std::unique_ptr<hololink::emulation::DataPlane> data_plane_;

    holoscan::Parameter<std::shared_ptr<hololink::emulation::HSBEmulator>> hsb_emulator_;
    holoscan::Parameter<std::string> source_ip_address_;
    holoscan::Parameter<uint32_t> subnet_bits_;
    holoscan::Parameter<uint32_t> source_port_;
    holoscan::Parameter<hololink::emulation::DataPlaneID> data_plane_id_;
    holoscan::Parameter<hololink::emulation::SensorID> sensor_id_;
};

} // namespace hololink::operators

#endif /* OPERATORS_EMULATOR_HSB_EMULATOR_OP_HPP */
