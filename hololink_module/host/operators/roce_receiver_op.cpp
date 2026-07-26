/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hololink/module/operators/roce_receiver_op.hpp"

#include <unistd.h> // getpagesize

#include <climits>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <cuda.h>

#include <fmt/format.h>
#include <gxf/std/tensor.hpp>
#include <yaml-cpp/yaml.h>

#include "hololink/module/adapter.hpp" // Adapter::get_adapter / get_module
#include "hololink/module/ibv_device.hpp" // ibv_device_for_peer
#include "hololink/module/logging.hpp" // HSB_LOG_INFO
#include "hololink/module/module.hpp"
#include "hololink/module/page_size.hpp" // PAGE_SIZE, round_up
#include "hololink/module/receiver_memory_descriptor.hpp"
#include "hololink/module/status.h"

// Holoscan deserializes operator parameters through YAML. The V1
// service handles and `EnumerationMetadata` are passed by direct
// `holoscan::Arg(..., value)` from application code — never via YAML
// configs — so the YAML converters below just throw on use. They
// exist solely to satisfy Holoscan's `register_converter<T>()`
// requirement that every custom parameter type be representable in
// YAML; the actual argument binding happens through the C++
// `holoscan::Arg` path and never invokes these.
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

// The backing receiver decodes the EOF block internally for its own
// bookkeeping; we still copy it back through the V1
// FrameMetadataInterfaceV1::decode contract so modules own the
// metadata layout (and its size, via block_size()).

void RoceReceiverOp::setup(holoscan::OperatorSpec& spec)
{
    spec.output<holoscan::gxf::Entity>("output");

    // Register converters for parameter types Holoscan doesn't know
    // about out of the box. Without these, `make_operator<...>(...,
    // Arg("enumeration_metadata", md), ...)` from application code
    // can't bind because Holoscan looks up a YAML converter by type.
    register_converter<EnumerationMetadata>();
    register_converter<CUcontext>();
    register_converter<size_t>();
    register_converter<std::function<void()>>();
    register_converter<std::function<std::string(const std::string&)>>();

    spec.param(metadata_, "enumeration_metadata", "EnumerationMetadata",
        "Enumeration metadata for the channel — the operator uses this to "
        "resolve the supplement module and look up the data channel + "
        "frame-metadata services");
    spec.param(frame_context_, "frame_context", "FrameContext",
        "CUDA context for the frame memory");
    spec.param(frame_size_, "frame_size", "FrameSize",
        "Per-frame payload size in bytes");
    spec.param(page_size_, "page_size", "PageSize",
        "Per-page buffer size in bytes", size_t { 0 });
    spec.param(pages_, "pages", "Pages",
        "Number of frame pages to post to the receiver", uint32_t { 2 });
    spec.param(queue_size_, "queue_size", "QueueSize",
        "Receive-queue depth (number of in-flight WRs)", uint32_t { 1 });
    spec.param(metadata_offset_, "metadata_offset", "MetadataOffset",
        "In-page offset of the 48-byte EOF metadata block", size_t { 0 });
    spec.param(device_start_, "device_start", "DeviceStart",
        "Callback fired after the QP is up — the sensor's start hook lives here",
        std::function<void()> {});
    spec.param(device_stop_, "device_stop", "DeviceStop",
        "Callback fired before the QP is torn down — the sensor's stop hook lives here",
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

RoceReceiverOp::~RoceReceiverOp() = default;

void RoceReceiverOp::frame_ready()
{
    // Invoked from the monitor thread when a frame lands. running_ guards
    // against a late signal re-arming a condition stop() has retired.
    if (frame_ready_condition_ && running_.load()) {
        frame_ready_condition_->event_state(holoscan::AsynchronousEventState::EVENT_DONE);
    }
}

void RoceReceiverOp::start()
{
    // Cache the (possibly renamed) metadata key names once, so compute()
    // does no per-frame string work. Default to identity when no
    // rename_metadata function was supplied.
    auto rename = rename_metadata_.get();
    if (!rename) {
        rename = [](const std::string& name) { return name; };
    }
    flags_key_ = rename("flags");
    psn_key_ = rename("psn");
    crc_key_ = rename("crc");
    frame_number_key_ = rename("frame_number");
    timestamp_s_key_ = rename("timestamp_s");
    timestamp_ns_key_ = rename("timestamp_ns");
    bytes_written_key_ = rename("bytes_written");
    metadata_s_key_ = rename("metadata_s");
    metadata_ns_key_ = rename("metadata_ns");
    received_s_key_ = rename("received_s");
    received_ns_key_ = rename("received_ns");
    imm_data_key_ = rename("imm_data");
    page_number_key_ = rename("page_number");

    if (queue_size_.get() == 0) {
        throw std::runtime_error("Queue size cannot be 0");
    }
    if (queue_size_.get() > pages_.get()) {
        throw std::runtime_error(fmt::format("Queue size {} cannot be greater than the number of pages {}", queue_size_.get(), pages_.get()));
    }

    // Resolve the supplement module that handled this metadata's
    // enumeration, then fetch the V1 services it published:
    //   - RoceDataChannelInterfaceV1 (per (serial, data_plane))
    //   - FrameMetadataInterfaceV1 (per-module singleton)
    //   - RoceReceiverInterfaceV1 (per (serial, data_plane))
    // All three sit behind the same module Publisher; on hsb_lite_2510
    // boards the channel + receiver are subclass overrides, picked up
    // automatically by the locator.
    auto& adapter = hololink::module::Adapter::get_adapter();
    auto module = adapter.get_module(metadata_.get());
    // Metadata-form get_service: caches the per-(serial, data_plane)
    // RoceDataChannelImpl and runs its configure(metadata) so the backing
    // DataChannel is materialized against the per-board LegacyHololinkAccess.
    channel_ = hololink::module::RoceDataChannelInterfaceV1::get_service(
        metadata_.get());
    frame_metadata_ = hololink::module::FrameMetadataInterfaceV1::get_service(module);
    host_metadata_.assign(frame_metadata_->block_size(), 0);

    size_t metadata_address = hololink::module::round_up(
        frame_size_.get(), hololink::module::PAGE_SIZE);
    if (metadata_offset_ == 0) {
        metadata_offset_ = metadata_address;
    }
    if (metadata_offset_ > metadata_address) {
        // This only occurs if the user passed in a metadata_offset that is
        // past the end of the buffer we'll allocate.
        throw std::runtime_error(fmt::format("metadata_offset={:#x} is beyond the receiver buffer limit={:#x}.", metadata_offset_, metadata_address));
    }
    // received_frame_size wants to be page aligned; the EOF metadata
    // block is sized by the module's FrameMetadataInterfaceV1, rounded
    // up to PAGE_SIZE so the per-frame stride stays page-aligned.
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
        // Per-frame stride the FPGA cycles through:
        // frame_memory[page] = cu_buffer + page_size * page. Setting this to
        // buffer_size would land every page > 0 frame outside the registered
        // MR, so the received frame size is used.
        page_size = received_frame_size;
    }

    receiver_ = hololink::module::RoceReceiverInterfaceV1::get_service(
        metadata_.get());

    // peer_ip travels in the enumeration metadata; ibv_name / ibv_port
    // are derived from it via ibv_device_for_peer (deterministic for a
    // given (peer_ip, host networking) pair).
    const std::string peer_ip = metadata_.get().get<std::string>("peer_ip");
    // Debug diagnostic: the peer_ip this receiver pulls from its enumeration
    // metadata, plus the data_channel that identifies the leg. If this peer
    // differs run-to-run for the same leg (e.g. .2 vs .3), the variable is the
    // metadata fed in, not local_ip_and_mac.
    HSB_LOG_DEBUG("RoceReceiverOp::start peer_ip(from metadata)={} data_channel={} serial={}",
        peer_ip,
        metadata_.get().get<int64_t>("data_channel", int64_t { -1 }),
        metadata_.get().get<std::string>("serial_number", std::string {}));
    const auto [ibv_name, ibv_port] = hololink::module::ibv_device_for_peer(peer_ip);

    const hololink_module_status_t start_status = receiver_->start(
        ibv_name,
        ibv_port,
        static_cast<uint64_t>(frame_buffer_->get()),
        buffer_size,
        frame_size_.get(),
        page_size,
        pages_.get(),
        metadata_offset_.get(),
        peer_ip,
        queue_size_.get());
    if (start_status != HOLOLINK_MODULE_OK) {
        throw std::runtime_error(std::string(
                                     "While starting RoceReceiverOp: RoceReceiverV1::start returned status ")
            + std::to_string(start_status));
    }

    const hololink_module_status_t status = channel_->attach_receiver(receiver_);
    if (status != HOLOLINK_MODULE_OK) {
        throw std::runtime_error(std::string(
                                     "While starting RoceReceiverOp: channel->attach_receiver returned status ")
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
        [self = receiver_, frame_context = frame_context_]() {
            HOLOLINK_MODULE_CUDA_CHECK(cuCtxSetCurrent(frame_context));
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

void RoceReceiverOp::stop()
{
    // Cleanup runs from this destructor so it executes even if
    // other calls throw exceptions below.
    class Cleanup {
    public:
        explicit Cleanup(RoceReceiverOp& self)
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
        }

    private:
        RoceReceiverOp& self_;
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
                                         "While stopping RoceReceiverOp: channel->detach_receiver returned status ")
                + std::to_string(status));
        }
    }
}

void RoceReceiverOp::compute(holoscan::InputContext& /*op_input*/,
    holoscan::OutputContext& op_output,
    holoscan::ExecutionContext& context)
{
    // cuMemcpyDtoH below is a context-sensitive driver call. Holoscan
    // worker threads don't carry frame_context_, so bind it here every
    // compute — the monitor thread already does the same in its lambda.
    HOLOLINK_MODULE_CUDA_CHECK(cuCtxSetCurrent(frame_context_.get()));

    hololink::module::RoceReceiverFrameInfoV1 frame_info {};
    if (!receiver_->get_next_frame(GET_NEXT_FRAME_TIMEOUT_MS, frame_info)) {
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

    // Pull the EOF block from device memory and let the V1
    // FrameMetadataInterfaceV1::decode own the layout. The backing
    // receiver also decodes internally, but routing through the V1
    // contract keeps modules in charge of their own metadata format
    // and its size. host_metadata_ was sized in start() from
    // frame_metadata_->block_size().
    const CUresult copy_result = cuMemcpyDtoH(
        host_metadata_.data(),
        static_cast<CUdeviceptr>(frame_info.metadata_memory),
        host_metadata_.size());
    if (copy_result != CUDA_SUCCESS) {
        throw std::runtime_error(
            "While reading frame metadata: cuMemcpyDtoH failed");
    }

    hololink::module::FrameMetadataInterfaceV1::FrameMetadata v1_metadata {};
    const hololink_module_status_t decode_status = frame_metadata_->decode(
        host_metadata_.data(), host_metadata_.size(), v1_metadata);
    if (decode_status != HOLOLINK_MODULE_OK) {
        throw std::runtime_error(std::string(
                                     "While decoding frame metadata: status ")
            + std::to_string(decode_status));
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

    // Stamp the operator's metadata map with the V1 FrameMetadata
    // fields so downstream operators can read them by name.
    auto meta_map = metadata();
    meta_map->set(flags_key_, static_cast<int64_t>(v1_metadata.flags));
    meta_map->set(psn_key_, static_cast<int64_t>(v1_metadata.psn));
    meta_map->set(crc_key_, static_cast<int64_t>(v1_metadata.crc));
    meta_map->set(frame_number_key_, static_cast<int64_t>(v1_metadata.frame_number));
    meta_map->set(timestamp_s_key_, static_cast<int64_t>(v1_metadata.timestamp_s));
    meta_map->set(timestamp_ns_key_, static_cast<int64_t>(v1_metadata.timestamp_ns));
    meta_map->set(bytes_written_key_, static_cast<int64_t>(v1_metadata.bytes_written));
    meta_map->set(metadata_s_key_, static_cast<int64_t>(v1_metadata.metadata_s));
    meta_map->set(metadata_ns_key_, static_cast<int64_t>(v1_metadata.metadata_ns));
    // received_s/ns is the host-side reception time (when the last frame
    // data landed); the RoCE receiver records it on frame_info rather than
    // in the decoded EOF block. Emit it so timestamp diagnostics can
    // compute FPGA-to-host latency, matching LinuxReceiverOp.
    meta_map->set(received_s_key_, static_cast<int64_t>(frame_info.received_s));
    meta_map->set(received_ns_key_, static_cast<int64_t>(frame_info.received_ns));
    // imm_data isn't carried in the decoded EOF block (FrameMetadata),
    // but the RoCE receiver surfaces it on frame_info; emit it (and the
    // low-12-bit page_number) so the RoCE path matches LinuxReceiverOp.
    meta_map->set(imm_data_key_, static_cast<int64_t>(frame_info.imm_data));
    meta_map->set(page_number_key_, static_cast<int64_t>(frame_info.imm_data & 0xFFF));

    op_output.emit(out_entity.value(), "output");
}

} // namespace hololink::module::operators
