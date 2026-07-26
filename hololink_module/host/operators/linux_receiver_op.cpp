/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hololink/module/operators/linux_receiver_op.hpp"

#include <netinet/in.h>
#include <sched.h>
#include <sys/socket.h>
#include <unistd.h> // close, getpagesize

#include <cerrno> // errno
#include <climits>
#include <cstdint>
#include <cstdlib> // getenv
#include <cstring> // strlen
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <cuda.h>

#include <fmt/format.h>
#include <gxf/std/tensor.hpp>
#include <yaml-cpp/yaml.h>

#include "hololink/module/adapter.hpp" // Adapter::get_adapter / get_module
#include "hololink/module/logging.hpp" // HSB_LOG_INFO
#include "hololink/module/module.hpp"
#include "hololink/module/page_size.hpp" // PAGE_SIZE, round_up
#include "hololink/module/receiver_memory_descriptor.hpp"
#include "hololink/module/status.h"

// See roce_receiver_op.cpp: these YAML converters exist only to satisfy
// Holoscan's register_converter<T>() requirement; the actual binding
// happens through the C++ holoscan::Arg path and never invokes them.
#define HOLOLINK_MODULE_YAML_CONVERTER_UNSUPPORTED(TYPE)                                    \
    template <>                                                                             \
    struct YAML::convert<TYPE> {                                                            \
        static Node encode(TYPE&) { throw std::runtime_error("Unsupported"); }              \
        static bool decode(const Node&, TYPE&) { throw std::runtime_error("Unsupported"); } \
    };

HOLOLINK_MODULE_YAML_CONVERTER_UNSUPPORTED(hololink::module::EnumerationMetadata);
HOLOLINK_MODULE_YAML_CONVERTER_UNSUPPORTED(CUcontext);
HOLOLINK_MODULE_YAML_CONVERTER_UNSUPPORTED(std::function<void()>);
HOLOLINK_MODULE_YAML_CONVERTER_UNSUPPORTED(
    std::function<std::string(const std::string&)>);

namespace hololink::module::operators {

static constexpr unsigned GET_NEXT_FRAME_TIMEOUT_MS = 1000;
static constexpr int DEFAULT_RECEIVER_AFFINITY = 2;

void LinuxReceiverOp::setup(holoscan::OperatorSpec& spec)
{
    spec.output<holoscan::gxf::Entity>("output");

    register_converter<EnumerationMetadata>();
    register_converter<CUcontext>();
    register_converter<size_t>();
    register_converter<std::function<void()>>();
    register_converter<std::function<std::string(const std::string&)>>();

    spec.param(metadata_, "enumeration_metadata", "EnumerationMetadata",
        "Enumeration metadata for the channel — the operator uses this to "
        "resolve the supplement module and look up the data channel + "
        "receiver services");
    spec.param(frame_context_, "frame_context", "FrameContext",
        "CUDA context for the frame memory");
    spec.param(frame_size_, "frame_size", "FrameSize",
        "Per-frame payload size in bytes");
    spec.param(page_size_, "page_size", "PageSize",
        "Per-page buffer size in bytes", size_t { 0 });
    spec.param(pages_, "pages", "Pages",
        "Number of frame pages to post to the receiver", uint32_t { 2 });
    spec.param(queue_size_, "queue_size", "QueueSize",
        "Receive-queue depth (number of buffers that can be queued)", uint32_t { 1 });
    spec.param(receiver_affinity_, "receiver_affinity", "ReceiverAffinity",
        "CPU affinity set for the receiver worker thread", std::vector<int> {});
    spec.param(device_start_, "device_start", "DeviceStart",
        "Callback fired after the receiver is up — the sensor's start hook lives here",
        std::function<void()> {});
    spec.param(device_stop_, "device_stop", "DeviceStop",
        "Callback fired before the receiver is torn down — the sensor's stop hook lives here",
        std::function<void()> {});
    spec.param(rename_metadata_, "rename_metadata", "RenameMetadata",
        "Optional function mapping each emitted metadata key to a new "
        "name (e.g. add a per-leg prefix). Defaults to identity.",
        std::function<std::string(const std::string&)> {});
    spec.param(out_tensor_name_, "out_tensor_name", "OutTensorName",
        "Name to give the emitted frame tensor (e.g. a per-leg name so a "
        "downstream consumer can tell legs apart). Defaults to \"\".",
        std::string {});
    // compute() is always driven by an AsynchronousCondition the monitor
    // thread signals on each new frame, so it never blocks waiting for one
    // (the scheduler stays free for other operators — required for joins).
    // The condition must be created and added as an arg during setup(),
    // before start() runs.
    auto* frag = fragment();
    frame_ready_condition_ = frag->make_condition<holoscan::AsynchronousCondition>(
        "frame_ready_condition");
    add_arg(frame_ready_condition_);
    frame_ready_condition_->event_state(holoscan::AsynchronousEventState::WAIT);
}

LinuxReceiverOp::~LinuxReceiverOp() = default;

void LinuxReceiverOp::frame_ready()
{
    // Invoked from the monitor thread when a frame lands. running_ guards
    // against a late signal re-arming a condition stop() has retired.
    if (frame_ready_condition_ && running_.load()) {
        frame_ready_condition_->event_state(holoscan::AsynchronousEventState::EVENT_DONE);
    }
}

void LinuxReceiverOp::start()
{
    if (queue_size_.get() == 0) {
        throw std::runtime_error("Queue size cannot be 0");
    }
    if (queue_size_.get() > pages_.get()) {
        throw std::runtime_error(fmt::format("Queue size {} cannot be greater than the number of pages {}", queue_size_.get(), pages_.get()));
    }

    // Cache the (possibly renamed) metadata key names once, so compute()
    // does no per-frame string work. Default to identity when no
    // rename_metadata function was supplied.
    auto rename = rename_metadata_.get();
    if (!rename) {
        rename = [](const std::string& name) { return name; };
    }
    frame_packets_received_key_ = rename("frame_packets_received");
    frame_bytes_received_key_ = rename("frame_bytes_received");
    frame_number_key_ = rename("frame_number");
    received_frame_number_key_ = rename("received_frame_number");
    received_s_key_ = rename("received_s");
    received_ns_key_ = rename("received_ns");
    timestamp_s_key_ = rename("timestamp_s");
    timestamp_ns_key_ = rename("timestamp_ns");
    metadata_s_key_ = rename("metadata_s");
    metadata_ns_key_ = rename("metadata_ns");
    packets_dropped_key_ = rename("packets_dropped");
    crc_key_ = rename("crc");
    psn_key_ = rename("psn");
    imm_data_key_ = rename("imm_data");
    page_number_key_ = rename("page_number");
    bytes_written_key_ = rename("bytes_written");

    // Default the receiver affinity from HOLOLINK_AFFINITY (or a fixed
    // default) when the application didn't set one.
    if (receiver_affinity_.get().empty()) {
        const char* affinity_env = std::getenv("HOLOLINK_AFFINITY");
        if (affinity_env && std::strlen(affinity_env) > 0) {
            receiver_affinity_ = std::vector<int> { std::atoi(affinity_env) };
        } else {
            receiver_affinity_ = std::vector<int> { DEFAULT_RECEIVER_AFFINITY };
        }
    }

    // Resolve the supplement module that handled this metadata's
    // enumeration, then fetch the V1 services it published:
    //   - LinuxDataChannelInterfaceV1 (per (serial, data_channel))
    //   - LinuxReceiverInterfaceV1 (per (serial, data_channel))
    //   - FrameMetadataInterfaceV1 (per-module singleton; block_size only)
    auto& adapter = hololink::module::Adapter::get_adapter();
    auto module = adapter.get_module(metadata_.get());
    channel_ = hololink::module::LinuxDataChannelInterfaceV1::get_service(
        metadata_.get());
    frame_metadata_ = hololink::module::FrameMetadataInterfaceV1::get_service(module);

    size_t metadata_address = hololink::module::round_up(
        frame_size_.get(), hololink::module::PAGE_SIZE);
    // received_frame_size wants to be page aligned; the metadata block is
    // sized by the module's FrameMetadataInterfaceV1, rounded up to
    // PAGE_SIZE so the per-frame stride stays page-aligned.
    static_assert(
        (hololink::module::PAGE_SIZE & (hololink::module::PAGE_SIZE - 1)) == 0);
    const size_t metadata_size = hololink::module::round_up(
        frame_metadata_->block_size(), hololink::module::PAGE_SIZE);
    size_t received_frame_size = metadata_address + metadata_size;
    size_t buffer_size = hololink::module::round_up(
        received_frame_size * pages_.get(), getpagesize());
    frame_buffer_ = std::make_shared<hololink::module::ReceiverMemoryDescriptor>(
        frame_context_.get(), buffer_size);
    HSB_LOG_INFO("frame_size={:#x} frame={:#x} buffer_size={:#x}", frame_size_.get(), frame_buffer_->get(), buffer_size);

    size_t page_size = page_size_.get();
    if (!page_size) {
        // Per-frame stride the FPGA cycles through. Setting this to
        // buffer_size would land every page > 0 frame outside the
        // buffer, so received_frame_size is used.
        page_size = received_frame_size;
    }

    // Create the datagram socket and bind it to the data plane via the
    // channel before the receiver runs on it. The receiver's local_port
    // (read in attach_receiver) comes from this bound socket.
    data_socket_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (data_socket_ < 0) {
        throw std::runtime_error("While starting LinuxReceiverOp: failed to create data socket");
    }
    const hololink_module_status_t socket_status = channel_->configure_socket(data_socket_);
    if (socket_status != HOLOLINK_MODULE_OK) {
        throw std::runtime_error(std::string(
                                     "While starting LinuxReceiverOp: channel->configure_socket returned status ")
            + std::to_string(socket_status));
    }

    receiver_ = hololink::module::LinuxReceiverInterfaceV1::get_service(
        metadata_.get());

    const hololink_module_status_t start_status = receiver_->start(
        data_socket_,
        static_cast<uint64_t>(frame_buffer_->get()),
        buffer_size,
        frame_size_.get(),
        page_size,
        pages_.get(),
        queue_size_.get());
    if (start_status != HOLOLINK_MODULE_OK) {
        throw std::runtime_error(std::string(
                                     "While starting LinuxReceiverOp: LinuxReceiverV1::start returned status ")
            + std::to_string(start_status));
    }

    const hololink_module_status_t status = channel_->attach_receiver(receiver_);
    if (status != HOLOLINK_MODULE_OK) {
        throw std::runtime_error(std::string(
                                     "While starting LinuxReceiverOp: channel->attach_receiver returned status ")
            + std::to_string(status));
    }
    configured_ = true;

    running_ = true;
    // Register the frame-ready callback before the monitor thread starts
    // so no frame slips through before the condition is armed.
    if (frame_ready_condition_) {
        receiver_->set_frame_ready([this]() { frame_ready(); });
    }

    monitor_thread_ = std::make_shared<std::thread>(
        [self = receiver_, frame_context = frame_context_,
            affinity = receiver_affinity_.get()]() {
            HOLOLINK_MODULE_CUDA_CHECK(cuCtxSetCurrent(frame_context));
            if (!affinity.empty()) {
                cpu_set_t cpu_set;
                CPU_ZERO(&cpu_set);
                for (int cpu : affinity) {
                    CPU_SET(cpu, &cpu_set);
                }
                if (sched_setaffinity(0, sizeof(cpu_set), &cpu_set) != 0) {
                    HSB_LOG_WARN("Failed to set receiver CPU affinity, errno={}", errno);
                }
            }
            self->blocking_monitor();
        });

    if (device_start_.get()) {
        device_start_.get()();
    }

    // Don't wait for the next signal if a frame is already queued.
    if (frame_ready_condition_ && receiver_->frames_ready()) {
        frame_ready_condition_->event_state(holoscan::AsynchronousEventState::EVENT_DONE);
    }
}

void LinuxReceiverOp::stop()
{
    // Cleanup runs from this destructor so it executes even if other
    // calls throw below.
    class Cleanup {
    public:
        explicit Cleanup(LinuxReceiverOp& self)
            : self_(self)
        {
        }
        ~Cleanup()
        {
            self_.configured_ = false;
            if (self_.receiver_) {
                self_.receiver_->close();
            }
            if (self_.monitor_thread_ && self_.monitor_thread_->joinable()) {
                self_.monitor_thread_->join();
            }
            self_.monitor_thread_.reset();
            self_.receiver_.reset();
            self_.channel_.reset();
            self_.frame_metadata_.reset();
            self_.frame_buffer_.reset();
            if (self_.data_socket_ >= 0) {
                ::close(self_.data_socket_);
                self_.data_socket_ = -1;
            }
        }

    private:
        LinuxReceiverOp& self_;
    };
    Cleanup cleanup(*this);

    // Stop signaling and retire the condition before tearing the receiver
    // down, so a frame arriving mid-teardown can't re-arm it.
    running_ = false;
    if (frame_ready_condition_) {
        frame_ready_condition_->event_state(holoscan::AsynchronousEventState::EVENT_NEVER);
    }

    if (device_stop_.get()) {
        device_stop_.get()();
    }

    if (configured_ && channel_) {
        const hololink_module_status_t status = channel_->detach_receiver();
        if (status != HOLOLINK_MODULE_OK) {
            throw std::runtime_error(std::string(
                                         "While stopping LinuxReceiverOp: channel->detach_receiver returned status ")
                + std::to_string(status));
        }
    }
}

void LinuxReceiverOp::compute(holoscan::InputContext& /*op_input*/,
    holoscan::OutputContext& op_output,
    holoscan::ExecutionContext& context)
{
    // The software receiver issues a context-sensitive cuMemcpyHtoDAsync
    // inside get_next_frame; bind the frame context on this worker
    // thread (the monitor thread already does the same).
    HOLOLINK_MODULE_CUDA_CHECK(cuCtxSetCurrent(frame_context_.get()));

    // Allocate the stream the receiver copies the reassembled frame on,
    // so the host→device copy overlaps with downstream pipeline work.
    auto maybe_cuda_stream = context.allocate_cuda_stream("linux_receiver_stream");
    if (!maybe_cuda_stream) {
        throw std::runtime_error(
            "While computing LinuxReceiverOp: failed to allocate CUDA stream");
    }
    const auto cuda_stream = maybe_cuda_stream.value();

    hololink::module::LinuxReceiverFrameInfoV1 frame_info {};
    if (!receiver_->get_next_frame(GET_NEXT_FRAME_TIMEOUT_MS, frame_info,
            reinterpret_cast<void*>(cuda_stream))) {
        // Timeout: produce no output; the scheduler will revisit us.
        return;
    }

    if (frame_ready_condition_) {
        // Re-arm: wait for the next monitor signal unless another frame is
        // already queued (then run again immediately). Set WAIT before the
        // frames_ready() test to avoid a lost wakeup.
        frame_ready_condition_->event_state(holoscan::AsynchronousEventState::EVENT_WAITING);
        if (receiver_->frames_ready()) {
            frame_ready_condition_->event_state(holoscan::AsynchronousEventState::EVENT_DONE);
        }
    }

    // Wrap the just-received frame buffer in a GXF tensor.
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
    if (frame_size_.get() > static_cast<size_t>(INT_MAX)) {
        throw std::runtime_error(
            "While building output entity: frame_size exceeds INT_MAX");
    }
    const nvidia::gxf::Shape shape { static_cast<int>(frame_size_.get()) };
    constexpr nvidia::gxf::PrimitiveType element_type
        = nvidia::gxf::PrimitiveType::kUnsigned8;
    const uint64_t element_size
        = nvidia::gxf::PrimitiveTypeSize(element_type);
    // Keep the receiver's frame buffer alive for as long as this wrapped
    // tensor is referenced anywhere downstream. Otherwise stop() frees the
    // buffer (frame_buffer_.reset()) while in-flight frames still point into
    // it, and the GPU pipeline reads freed memory (CUDA_ERROR_ILLEGAL_ADDRESS).
    // Capturing the shared_ptr in the release callback means the buffer is
    // freed only after the last referencing tensor is released.
    auto buffer_owner = frame_buffer_;
    if (!tensor.value()->wrapMemory(shape, element_type, element_size,
            nvidia::gxf::ComputeTrivialStrides(shape, element_size),
            nvidia::gxf::MemoryStorageType::kDevice,
            reinterpret_cast<void*>(frame_info.frame_memory),
            [buffer_owner](void*) {
                // Holds the owning reference until the tensor's memory is released.
                return nvidia::gxf::Success;
            })) {
        throw std::runtime_error(
            "While building output entity: Tensor::wrapMemory failed");
    }

    // Stamp the operator's metadata map. Unlike RoceReceiverOp these
    // fields come straight off the software receiver's frame-info struct
    // (it decoded them during reassembly) rather than from a decoded EOF
    // block.
    auto meta_map = metadata();
    meta_map->set(frame_packets_received_key_, static_cast<int64_t>(frame_info.frame_packets_received));
    meta_map->set(frame_bytes_received_key_, static_cast<int64_t>(frame_info.frame_bytes_received));
    meta_map->set(frame_number_key_, static_cast<int64_t>(frame_info.frame_number));
    meta_map->set(received_frame_number_key_, static_cast<int64_t>(frame_info.received_frame_number));
    meta_map->set(received_s_key_, static_cast<int64_t>(frame_info.received_s));
    meta_map->set(received_ns_key_, static_cast<int64_t>(frame_info.received_ns));
    meta_map->set(timestamp_s_key_, static_cast<int64_t>(frame_info.timestamp_s));
    meta_map->set(timestamp_ns_key_, static_cast<int64_t>(frame_info.timestamp_ns));
    meta_map->set(metadata_s_key_, static_cast<int64_t>(frame_info.metadata_s));
    meta_map->set(metadata_ns_key_, static_cast<int64_t>(frame_info.metadata_ns));
    meta_map->set(packets_dropped_key_, static_cast<int64_t>(frame_info.packets_dropped));
    meta_map->set(crc_key_, static_cast<int64_t>(frame_info.crc));
    meta_map->set(psn_key_, static_cast<int64_t>(frame_info.psn));
    meta_map->set(imm_data_key_, static_cast<int64_t>(frame_info.imm_data));
    meta_map->set(page_number_key_, static_cast<int64_t>(frame_info.imm_data & 0xFFF));
    meta_map->set(bytes_written_key_, static_cast<int64_t>(frame_info.bytes_written));

    op_output.set_cuda_stream(cuda_stream, "output");
    op_output.emit(out_entity.value(), "output");
}

} // namespace hololink::module::operators
