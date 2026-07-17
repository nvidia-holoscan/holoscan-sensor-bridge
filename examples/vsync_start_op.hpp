/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_EXAMPLES_VSYNC_START_OP_HPP
#define HOLOLINK_EXAMPLES_VSYNC_START_OP_HPP

#include <memory>
#include <stdexcept>
#include <string>

#include <yaml-cpp/yaml.h>

#include <holoscan/holoscan.hpp>

#include "hololink/module/status.h"
#include "hololink/module/vsync.hpp"

// Holoscan binds operator parameters through YAML converters. The
// VsyncStartOp `vsync` parameter is only ever set with a C++
// holoscan::Arg("vsync", vsync) from application code — never from a
// YAML config — so this converter exists solely to satisfy
// register_converter<T>()'s requirement and throws if ever invoked.
// (Mirrors HOLOLINK_MODULE_YAML_CONVERTER_UNSUPPORTED in the adapter
// operator .cpp files.)
template <>
struct YAML::convert<std::shared_ptr<hololink::module::VsyncInterfaceV1>> {
    static Node encode(std::shared_ptr<hololink::module::VsyncInterfaceV1>&)
    {
        throw std::runtime_error("Unsupported");
    }
    static bool decode(const Node&, std::shared_ptr<hololink::module::VsyncInterfaceV1>&)
    {
        throw std::runtime_error("Unsupported");
    }
};

/* One-shot holoscan operator whose first compute() fires
 * vsync->start(). Wired into an application's compose() with a
 * CountCondition(1) and no input/output ports. Holoscan invokes
 * compute() only after every operator's start() has run — including
 * the RoceReceiverOp::device_start lambdas that arm each camera.
 * By the time this fires, the cameras are in external-sync waiting
 * state, so the first VSYNC pulse lands on all of them simultaneously
 * and frame N on each receiver tracks the same trigger edge.
 *
 * The vsync source is taken via a holoscan::Parameter — applications
 * pass holoscan::Arg("vsync", vsync) to make_operator. Required: the
 * Parameter is declared without a default, so omitting the Arg
 * fails operator initialization. */
class VsyncStartOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(VsyncStartOp)

    VsyncStartOp() = default;

    void setup(holoscan::OperatorSpec& spec) override
    {
        register_converter<std::shared_ptr<hololink::module::VsyncInterfaceV1>>();

        spec.param(vsync_, "vsync",
            "Vsync source",
            "Vsync source whose start() is fired once when this "
            "operator's compute() runs.");
    }

    void compute(holoscan::InputContext& /*op_input*/,
        holoscan::OutputContext& /*op_output*/,
        holoscan::ExecutionContext& /*context*/) override
    {
        const hololink_module_status_t status = vsync_.get()->start();
        if (status != HOLOLINK_MODULE_OK) {
            throw std::runtime_error(
                "While starting VSYNC pulses from VsyncStartOp: status "
                + std::to_string(static_cast<int>(status)));
        }
    }

private:
    holoscan::Parameter<std::shared_ptr<hololink::module::VsyncInterfaceV1>> vsync_;
};

#endif // HOLOLINK_EXAMPLES_VSYNC_START_OP_HPP
