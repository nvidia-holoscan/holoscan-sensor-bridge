/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hololink/module/operators/hsb_controller_op.hpp"

#include <climits>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include <cuda.h>
#include <gxf/std/tensor.hpp>
#include <yaml-cpp/yaml.h>

#include "hololink/module/cuda_unique.hpp" // HOLOLINK_MODULE_CUDA_CHECK
#include "hololink/module/status.h"

#define HOLOLINK_MODULE_YAML_CONVERTER_UNSUPPORTED(TYPE)                                    \
    template <>                                                                             \
    struct YAML::convert<TYPE> {                                                            \
        static Node encode(TYPE&) { throw std::runtime_error("Unsupported"); }              \
        static bool decode(const Node&, TYPE&) { throw std::runtime_error("Unsupported"); } \
    };

HOLOLINK_MODULE_YAML_CONVERTER_UNSUPPORTED(hololink::module::EnumerationMetadata);
HOLOLINK_MODULE_YAML_CONVERTER_UNSUPPORTED(CUcontext);
HOLOLINK_MODULE_YAML_CONVERTER_UNSUPPORTED(
    std::function<std::string(const std::string&)>);
HOLOLINK_MODULE_YAML_CONVERTER_UNSUPPORTED(
    std::shared_ptr<hololink::module::operators::NetworkReceiverFactory>);
HOLOLINK_MODULE_YAML_CONVERTER_UNSUPPORTED(
    std::shared_ptr<hololink::module::operators::SensorFactory>);

namespace hololink::module::operators {

static constexpr unsigned GET_NEXT_FRAME_TIMEOUT_MS = 1000;

HsbControllerOp::~HsbControllerOp() = default;

void HsbControllerOp::setup(holoscan::OperatorSpec& spec)
{
    spec.output<holoscan::gxf::Entity>("output");

    register_converter<EnumerationMetadata>();
    register_converter<CUcontext>();
    register_converter<size_t>();
    register_converter<std::function<std::string(const std::string&)>>();
    register_converter<std::shared_ptr<NetworkReceiverFactory>>();
    register_converter<std::shared_ptr<SensorFactory>>();

    spec.param(metadata_, "enumeration_metadata", "EnumerationMetadata",
        "Channel enumeration metadata (peer_ip used to register for the "
        "device's announcements)");
    spec.param(frame_context_, "frame_context", "FrameContext",
        "CUDA context for the frame memory");
    spec.param(frame_size_, "frame_size", "FrameSize",
        "Per-frame payload size in bytes");
    spec.param(page_size_, "page_size", "PageSize", "", size_t { 0 });
    spec.param(pages_, "pages", "Pages", "", uint32_t { 2 });
    spec.param(queue_size_, "queue_size", "QueueSize", "", uint32_t { 1 });
    spec.param(metadata_offset_, "metadata_offset", "MetadataOffset", "", size_t { 0 });
    spec.param(out_tensor_name_, "out_tensor_name", "OutTensorName",
        "Name for the emitted frame tensor", std::string {});
    spec.param(rename_metadata_, "rename_metadata", "RenameMetadata",
        "Optional per-key renaming for the frame metadata",
        std::function<std::string(const std::string&)> {});
    spec.param(network_receiver_factory_, "network_receiver_factory",
        "NetworkReceiverFactory", "Builds the NetworkReceiver (RoCE / Linux / …)");
    spec.param(sensor_factory_, "sensor_factory", "SensorFactory",
        "Owns the sensor, watchdog, and reconnection policy");
    spec.param(watchdog_timeout_s_, "watchdog_timeout_s", "WatchdogTimeoutS",
        "Seconds without a frame before the device is declared lost", double { 0.5 });

    auto* frag = fragment();
    frame_ready_condition_ = frag->make_condition<holoscan::AsynchronousCondition>(
        "frame_ready_condition");
    add_arg(frame_ready_condition_);
    frame_ready_condition_->event_state(holoscan::AsynchronousEventState::WAIT);
}

void HsbControllerOp::frame_ready()
{
    if (frame_ready_condition_ && running_.load()) {
        frame_ready_condition_->event_state(holoscan::AsynchronousEventState::EVENT_DONE);
    }
}

void HsbControllerOp::start()
{
    running_ = true;

    NetworkReceiver::Config config;
    config.frame_context = frame_context_.get();
    config.frame_size = frame_size_.get();
    config.page_size = page_size_.get();
    config.pages = pages_.get();
    config.queue_size = queue_size_.get();
    config.metadata_offset = metadata_offset_.get();
    config.rename_metadata = rename_metadata_.get();
    // config.frame_ready is set by HsbController::start from our wake.

    controller_ = std::make_unique<HsbController>(
        sensor_factory_.get(), network_receiver_factory_.get(), std::move(config),
        metadata_.get(), watchdog_timeout_s_.get());
    controller_->start([this]() { frame_ready(); });

    // Tick compute() once at startup (before any device has connected) so it
    // emits the fallback test image; the receiver's frame-ready signals then
    // drive it, replacing the test image with live frames.
    frame_ready();
}

void HsbControllerOp::stop()
{
    running_ = false;
    if (frame_ready_condition_) {
        frame_ready_condition_->event_state(holoscan::AsynchronousEventState::EVENT_NEVER);
    }
    if (controller_) {
        controller_->stop();
        controller_.reset();
    }
}

void HsbControllerOp::compute(holoscan::InputContext& /*op_input*/,
    holoscan::OutputContext& op_output,
    holoscan::ExecutionContext& context)
{
    HOLOLINK_MODULE_CUDA_CHECK(cuCtxSetCurrent(frame_context_.get()));

    // Allocate the pipeline stream for this compute and hand it to the
    // receiver: a software transport places its host->device frame copy on it,
    // and the emitted tensor carries it so downstream work overlaps the copy. A
    // hardware transport ignores it. Also used for the fallback emit.
    auto maybe_cuda_stream = context.allocate_cuda_stream("hsb_controller_stream");
    if (!maybe_cuda_stream) {
        throw std::runtime_error(
            "While computing HsbControllerOp: failed to allocate CUDA stream");
    }
    const auto cuda_stream = maybe_cuda_stream.value();

    CUdeviceptr device_ptr = 0;
    size_t size = 0;
    std::shared_ptr<void> owner;
    if (controller_
        && controller_->get_next_frame(GET_NEXT_FRAME_TIMEOUT_MS, *metadata(),
            cuda_stream, device_ptr, size, owner)) {
        if (frame_ready_condition_) {
            frame_ready_condition_->event_state(
                holoscan::AsynchronousEventState::EVENT_WAITING);
            if (controller_->frames_ready()) {
                frame_ready_condition_->event_state(
                    holoscan::AsynchronousEventState::EVENT_DONE);
            }
        }
        emit_tensor(op_output, context, device_ptr, size, owner, cuda_stream);
        return;
    }

    if (frame_ready_condition_) {
        frame_ready_condition_->event_state(holoscan::AsynchronousEventState::EVENT_WAITING);
    }
    if (!controller_ || !controller_->connected()) {
        CUdeviceptr fallback_ptr = 0;
        size_t fallback_size = 0;
        if (controller_) {
            controller_->fallback_frame(fallback_ptr, fallback_size);
        }
        if (fallback_ptr != 0 && fallback_size != 0) {
            emit_tensor(op_output, context, fallback_ptr, fallback_size,
                /*owner=*/nullptr, cuda_stream);
        }
    }
}

void HsbControllerOp::emit_tensor(holoscan::OutputContext& op_output,
    holoscan::ExecutionContext& context, CUdeviceptr device_ptr, size_t size,
    std::shared_ptr<void> owner, CUstream cuda_stream)
{
    nvidia::gxf::Expected<nvidia::gxf::Entity> out_entity
        = nvidia::gxf::Entity::New(context.context());
    if (!out_entity) {
        throw std::runtime_error(
            "While building output entity: nvidia::gxf::Entity::New failed");
    }
    nvidia::gxf::Expected<nvidia::gxf::Handle<nvidia::gxf::Tensor>> tensor
        = out_entity.value().add<nvidia::gxf::Tensor>(out_tensor_name_.get().c_str());
    if (!tensor) {
        throw std::runtime_error(
            "While building output entity: failed to add GXF tensor");
    }
    if (size > static_cast<size_t>(INT_MAX)) {
        throw std::runtime_error(
            "While building output entity: frame_size exceeds INT_MAX");
    }
    const nvidia::gxf::Shape shape { static_cast<int>(size) };
    constexpr nvidia::gxf::PrimitiveType element_type
        = nvidia::gxf::PrimitiveType::kUnsigned8;
    const uint64_t element_size = nvidia::gxf::PrimitiveTypeSize(element_type);
    if (!tensor.value()->wrapMemory(shape, element_type, element_size,
            nvidia::gxf::ComputeTrivialStrides(shape, element_size),
            nvidia::gxf::MemoryStorageType::kDevice,
            reinterpret_cast<void*>(device_ptr),
            [owner](void*) { return nvidia::gxf::Success; })) {
        throw std::runtime_error(
            "While building output entity: Tensor::wrapMemory failed");
    }
    // Carry the stream a software transport copied the frame on, so downstream
    // work is ordered after the copy completes.
    op_output.set_cuda_stream(cuda_stream, "output");
    op_output.emit(out_entity.value(), "output");
}

} // namespace hololink::module::operators
