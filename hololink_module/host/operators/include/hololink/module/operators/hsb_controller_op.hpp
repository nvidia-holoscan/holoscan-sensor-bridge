/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_OPERATORS_HSB_CONTROLLER_OP_HPP
#define HOLOLINK_MODULE_OPERATORS_HSB_CONTROLLER_OP_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include <cuda.h>
#include <holoscan/holoscan.hpp>

#include "hololink/module/enumeration_metadata.hpp"

#include "hololink/module/operators/hsb_controller.hpp"
#include "hololink/module/operators/network_receiver.hpp"
#include "hololink/module/operators/sensor_factory.hpp"

namespace hololink::module::operators {

/* Thin HSDK adapter over HsbController: it owns the frame-ready condition
 * and the GXF tensor plumbing, and forwards start/stop/compute to the
 * controller — no reconnection, watchdog, or transport logic lives here.
 * compute() emits the controller's next frame while connected, or the
 * controller's fallback frame while disconnected, over "output" (ahead of
 * CsiToBayerOp). */
class HsbControllerOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(HsbControllerOp);

    ~HsbControllerOp() override;

    void setup(holoscan::OperatorSpec& spec) override;
    void start() override;
    void stop() override;
    void compute(holoscan::InputContext&,
        holoscan::OutputContext& op_output,
        holoscan::ExecutionContext&) override;

private:
    holoscan::Parameter<hololink::module::EnumerationMetadata> metadata_;
    holoscan::Parameter<CUcontext> frame_context_;
    holoscan::Parameter<size_t> frame_size_;
    holoscan::Parameter<size_t> page_size_;
    holoscan::Parameter<uint32_t> pages_;
    holoscan::Parameter<uint32_t> queue_size_;
    holoscan::Parameter<size_t> metadata_offset_;
    holoscan::Parameter<std::string> out_tensor_name_;
    holoscan::Parameter<std::function<std::string(const std::string&)>>
        rename_metadata_;
    holoscan::Parameter<std::shared_ptr<NetworkReceiverFactory>>
        network_receiver_factory_;
    holoscan::Parameter<std::shared_ptr<SensorFactory>> sensor_factory_;
    holoscan::Parameter<double> watchdog_timeout_s_;

    std::unique_ptr<HsbController> controller_;

    std::shared_ptr<holoscan::AsynchronousCondition> frame_ready_condition_;
    std::atomic<bool> running_ { false };
    void frame_ready();

    void emit_tensor(holoscan::OutputContext& op_output,
        holoscan::ExecutionContext& context, CUdeviceptr device_ptr,
        size_t size, std::shared_ptr<void> owner, CUstream cuda_stream);
};

} // namespace hololink::module::operators

#endif // HOLOLINK_MODULE_OPERATORS_HSB_CONTROLLER_OP_HPP
