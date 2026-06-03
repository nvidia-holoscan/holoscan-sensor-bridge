/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "base_receiver_op.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <holoscan/holoscan.hpp>
#include <yaml-cpp/parser.h>

#include <hololink/core/data_channel.hpp>
#include <hololink/core/logging_internal.hpp>

/**
 * @brief This macro defining a YAML converter which throws for unsupported types.
 *
 * Background: Holoscan supports setting parameters through YAML files. But for some parameters
 * accepted by the receiver operators like `DataChannel` class of functions it makes no sense
 * to specify them in YAML files. Therefore use a converter which throws for these types.
 *
 * @tparam TYPE
 */
#define YAML_CONVERTER(TYPE)                                                                \
    template <>                                                                             \
    struct YAML::convert<TYPE> {                                                            \
        static Node encode(TYPE&) { throw std::runtime_error("Unsupported"); }              \
                                                                                            \
        static bool decode(const Node&, TYPE&) { throw std::runtime_error("Unsupported"); } \
    };

YAML_CONVERTER(hololink::DataChannel*);
YAML_CONVERTER(std::function<void()>);
YAML_CONVERTER(CUcontext);

namespace hololink::operators {

void BaseReceiverOp::setup(holoscan::OperatorSpec& spec)
{
    spec.output<holoscan::gxf::Entity>("output");

    /// Register converters for arguments not defined by Holoscan
    register_converter<hololink::DataChannel*>();
    register_converter<std::function<void()>>();
    register_converter<CUcontext>();
    register_converter<size_t>();
    register_converter<CUdeviceptr>();

    spec.param(hololink_channel_, "hololink_channel", "HololinkChannel",
        "Pointer to Hololink Datachannel object");
    spec.param(
        device_start_, "device_start", "DeviceStart", "Function to be called to start the device");
    spec.param(
        device_stop_, "device_stop", "DeviceStop", "Function to be called to stop the device");
    spec.param(frame_context_, "frame_context", "FrameContext", "CUDA context");
    spec.param(frame_size_, "frame_size", "FrameSize", "Size of one frame in bytes");
    spec.param(trim_, "trim", "Trim", "Set output length to bytes_written (else frame_size)", true);
    spec.param(use_frame_ready_condition_, "use_frame_ready_condition", "UseFrameReadyCondition",
        "Use an AsynchronousCondition to a waiting state until a new frame is ready. If this is enabled "
        "then the compute method will not block and the scheduler can schedule other operators at the cost "
        "of a slight increase in latency. If this is disabled then the compute method will block until a "
        "new frame is ready, reducing latency.",
        true);

    // The condition needs to be added before the start method is called, but the parameter values
    // are not yet set to the arguments. So we need to check the arguments for the value.
    bool use_frame_ready_condition = use_frame_ready_condition_.default_value();
    for (auto& arg : args()) {
        if (arg.name() == "use_frame_ready_condition") {
            use_frame_ready_condition = std::any_cast<bool>(arg.value());
        }
    }

    if (use_frame_ready_condition) {
        auto frag = fragment();
        frame_ready_condition_ = frag->make_condition<holoscan::AsynchronousCondition>("frame_ready_condition");
        add_arg(frame_ready_condition_);
        frame_ready_condition_->event_state(holoscan::AsynchronousEventState::WAIT);
    }

    frame_count_ = 0;
}

void BaseReceiverOp::start()
{
    //
    data_socket_.reset(socket(AF_INET, SOCK_DGRAM, 0));
    if (!data_socket_) {
        throw std::runtime_error("Failed to create socket");
    }

    hololink_channel_->configure_socket(data_socket_.get());
    running_ = true;
    start_receiver();
    device_start_.get()();
    if (frame_ready_condition_ && frames_ready()) {
        // Don't wait if we have data ready.
        frame_ready_condition_->event_state(holoscan::AsynchronousEventState::EVENT_DONE);
    }
}

void BaseReceiverOp::stop()
{
    running_ = false;
    device_stop_.get()();
    stop_receiver();
    if (frame_ready_condition_) {
        frame_ready_condition_->event_state(holoscan::AsynchronousEventState::EVENT_NEVER);
    }
}

void BaseReceiverOp::compute(holoscan::InputContext& input, holoscan::OutputContext& output,
    holoscan::ExecutionContext& context)
{
    // get the CUDA stream for this operator
    auto maybe_cuda_stream = context.allocate_cuda_stream();
    if (!maybe_cuda_stream) {
        throw std::runtime_error("Failed to allocate CUDA stream");
    }
    const auto cuda_stream = maybe_cuda_stream.value();

    const double timeout_ms = 1000.f;
    auto [frame_memory, frame_metadata] = get_next_frame(timeout_ms, cuda_stream);
    if (!frame_metadata) {
        timeout(input, output, context);
        // In this case, we have no frame data to write to the application,
        // so we'll not produce any output.  The rest of the objects in the pipeline
        // will be skipped (due to no input) and execution will come back to us.
        return;
    }

    ok_ = true;
    frame_count_ += 1;
    if (frame_ready_condition_) {
        // Clear our asynchronous event
        frame_ready_condition_->event_state(holoscan::AsynchronousEventState::EVENT_WAITING);
        // Undo EVENT_WAITING if there is data available.  This is written
        // this way to avoid a race condition-- we always test frames_ready()
        // after setting EVENT_WAITING.
        if (frames_ready()) {
            frame_ready_condition_->event_state(holoscan::AsynchronousEventState::EVENT_DONE);
        }
    }

    // Create an Entity and use GXF tensor to wrap the CUDA memory.
    nvidia::gxf::Expected<nvidia::gxf::Entity> out_message
        = nvidia::gxf::Entity::New(context.context());
    if (!out_message) {
        throw std::runtime_error("Failed to create GXF entity");
    }
    nvidia::gxf::Expected<nvidia::gxf::Handle<nvidia::gxf::Tensor>> gxf_tensor
        = out_message.value().add<nvidia::gxf::Tensor>("");
    if (!out_message) {
        throw std::runtime_error("Failed to add GXF tensor");
    }
    // frame_size_ is size_t; protect the following static_cast<int>.
    if (frame_size_.get() > INT_MAX) {
        throw std::runtime_error(fmt::format("frame_size={} is above the maximum value of {}.", frame_size_.get(), INT_MAX));
    }
    // How many bytes should be included in our output tensor?
    // If self._trim is true then use metadata["bytes_written"],
    // otherwise pass all the receiver buffer.
    int tensor_size = static_cast<int>(frame_size_.get());
    do { // use "break" to get out
        if (!trim_) {
            break;
        }
        auto bytes_written_opt = frame_metadata->get<int64_t>("bytes_written");
        if (!bytes_written_opt) {
            break;
        }
        int bytes_written = static_cast<int>(*bytes_written_opt);
        if (bytes_written > tensor_size) {
            static unsigned flooding_control = 0;
            if (flooding_control < 5) {
                HSB_LOG_ERROR("Unexpected bytes_written={:#x} is larger than the buffer_size={:#x}, ignoring.",
                    bytes_written, tensor_size);
                flooding_control++;
            }
            break;
        }
        tensor_size = bytes_written;
    } while (false);
    const nvidia::gxf::Shape shape { tensor_size };
    const nvidia::gxf::PrimitiveType element_type = nvidia::gxf::PrimitiveType::kUnsigned8;
    const uint64_t element_size = nvidia::gxf::PrimitiveTypeSize(element_type);
    if (!gxf_tensor.value()->wrapMemory(shape, element_type, element_size,
            nvidia::gxf::ComputeTrivialStrides(shape, element_size),
            nvidia::gxf::MemoryStorageType::kDevice, reinterpret_cast<void*>(frame_memory),
            [](void*) {
                // release function, nothing to do
                return nvidia::gxf::Success;
            })) {
        throw std::runtime_error("Failed to add wrap memory");
    }
    // Publish the received metadata to the pipeline.
    auto const& meta = metadata();
    for (auto const& x : *frame_metadata) {
        // x.second is hololink::Metadata's map content type,
        // e.g. std::variant<int64_t, std::string, std::vector<uint8_t>>.
        // Poll though our various types in order to figure out what to
        // add to meta.
        if (std::holds_alternative<int64_t>(x.second)) {
            auto value = std::get<int64_t>(x.second);
            meta->set(x.first, value);
            continue;
        }
        if (std::holds_alternative<std::string>(x.second)) {
            auto value = std::get<std::string>(x.second);
            meta->set(x.first, value);
            continue;
        }
        if (std::holds_alternative<std::vector<uint8_t>>(x.second)) {
            auto value = std::get<std::vector<uint8_t>>(x.second);
            meta->set(x.first, value);
            continue;
        }
        throw std::runtime_error(fmt::format("Unable to copy metadata \"{}\".", x.first));
    }

    // Emit the stream to the output
    output.set_cuda_stream(cuda_stream, "output");
    // Emit the tensor.
    output.emit(out_message.value(), "output");
}

void BaseReceiverOp::timeout(holoscan::InputContext& input, holoscan::OutputContext& output,
    holoscan::ExecutionContext& context)
{
    if (ok_) {
        ok_ = false;
        HSB_LOG_ERROR("Ingress frame timeout; ignoring.");
    }
}

void BaseReceiverOp::frame_ready()
{
    if (frame_ready_condition_ && running_.load()) {
        frame_ready_condition_->event_state(holoscan::AsynchronousEventState::EVENT_DONE);
    }
}

std::tuple<std::string, uint32_t> BaseReceiverOp::local_ip_and_port()
{
    sockaddr_in ip {};
    ip.sin_family = AF_UNSPEC;
    socklen_t ip_len = sizeof(ip);
    if (getsockname(data_socket_.get(), (sockaddr*)&ip, &ip_len) < 0) {
        throw std::runtime_error(
            fmt::format("getsockname failed with errno={}: \"{}\"", errno, strerror(errno)));
    }

    const std::string local_ip = inet_ntoa(ip.sin_addr);
    const in_port_t local_port = ntohs(ip.sin_port);

    return { local_ip, local_port };
}

} // namespace hololink::operators
