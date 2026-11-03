/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// PcieReceiverOp: PCIe receiver operator, the shape of roce_receiver_op.cpp: V1 services
// from the enumeration metadata, AsynchronousCondition driven compute, RAII stop. Both
// halves of the data path are hsb_pcie module services reached through get_service --
// PcieReceiverInterfaceV1 (the ring, the dma-buf mapping, the MSI-X completion wait) and
// PcieDataChannelInterfaceV1 (the endpoint's registers: attach_receiver / detach_receiver,
// like the RoCE channel). This operator owns the frame buffer, the monitor thread and the
// Holoscan plumbing, and nothing PCIe-specific below the service interfaces. No legacy core.

#include "hololink/module/operators/pcie_receiver_op.hpp"

#include <climits>
#include <cstdint>
#include <exception>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <unistd.h> // getpagesize

#include <cuda.h>

#include <fmt/format.h>
#include <gxf/std/tensor.hpp>
#include <yaml-cpp/yaml.h>

#include "hololink/module/adapter.hpp" // Adapter::get_adapter / get_module
#include "hololink/module/cuda_unique.hpp" // HOLOLINK_MODULE_CUDA_CHECK
#include "hololink/module/hsb_pcie/pcie_data_channel.hpp"
#include "hololink/module/hsb_pcie/pcie_receiver.hpp"
#include "hololink/module/logging.hpp" // HSB_LOG_INFO
#include "hololink/module/module.hpp"
#include "hololink/module/page_size.hpp" // PAGE_SIZE, round_up
#include "hololink/module/receiver_memory_descriptor.hpp"
#include "hololink/module/status.h"

// See roce_receiver_op.cpp: these YAML converters exist only to satisfy
// Holoscan's register_converter<T>() requirement; the actual binding
// happens through the C++ holoscan::Arg path and never invokes these.
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

namespace {

    constexpr unsigned GET_NEXT_FRAME_TIMEOUT_MS = 1000;
    constexpr const char* PEER_IP_KEY = "peer_ip";
    constexpr const char* SENSOR_KEY = "sensor";

    using hololink::module::hsb_pcie::PCIE_RX_METADATA_BYTES;
    using hololink::module::hsb_pcie::PcieDataChannelInterfaceV1;
    using hololink::module::hsb_pcie::PcieReceiverFrameInfoV1;
    using hololink::module::hsb_pcie::PcieReceiverInterfaceV1;

} // namespace

void PcieReceiverOp::setup(holoscan::OperatorSpec& spec)
{
    spec.output<holoscan::gxf::Entity>("output");

    register_converter<EnumerationMetadata>();
    register_converter<CUcontext>();
    register_converter<size_t>();
    register_converter<std::function<void()>>();
    register_converter<std::function<std::string(const std::string&)>>();

    spec.param(metadata_, "enumeration_metadata", "EnumerationMetadata",
        "Enumeration metadata for the channel; peer_ip is the endpoint's PCI address "
        "(DDDD:BB:DD.F). The receiver and data-channel services are resolved from it "
        "(PcieReceiverInterfaceV1 / PcieDataChannelInterfaceV1::get_service)");
    spec.param(frame_context_, "frame_context", "FrameContext",
        "CUDA context for the frame memory");
    spec.param(frame_size_, "frame_size", "FrameSize",
        "Bytes the endpoint writes per frame: the CSI converter's length "
        "(CsiToBayerOp::get_csi_length), as for RoceReceiverOp");
    spec.param(device_start_, "device_start", "DeviceStart",
        "Callback fired after the channel is attached and armed and the monitor is running — "
        "the sensor's start hook lives here",
        std::function<void()> {});
    spec.param(device_stop_, "device_stop", "DeviceStop",
        "Callback fired before the channel is detached (disarmed) — the sensor's stop hook "
        "lives here",
        std::function<void()> {});
    spec.param(out_tensor_name_, "out_tensor_name", "OutTensorName",
        "Name to give the emitted frame tensor. Defaults to \"\".",
        std::string {});
    spec.param(rename_metadata_, "rename_metadata", "RenameMetadata",
        "Optional function mapping each emitted metadata key to a new name (e.g. a "
        "per-leg prefix). Defaults to identity.",
        std::function<std::string(const std::string&)> {});

    spec.param(sensor_, "sensor", "Sensor",
        "Sensor interface index; also selects the PCIE_DP block and MSI-X vector. "
        "-1 reads the metadata's 'sensor' key; an explicit value must match the channel "
        "the metadata resolves to (Adapter::use_sensor).",
        int32_t { -1 });

    spec.param(pages_, "pages", "Pages",
        "Number of frame pages the endpoint cycles through (the receive ring)", uint32_t { 4 });
    spec.param(chunk_pages_, "chunk_pages", "ChunkPages",
        "Data TLP size in 128-byte pages (DP_PKT_LEN, shared by every sensor on the "
        "endpoint); 1 gives maximum-size TLPs. Passed to the channel's attach_receiver.",
        uint32_t { 1 });
    // compute() is driven by an AsynchronousCondition the monitor thread
    // signals on each new frame, so it never blocks waiting for one.
    auto* frag = fragment();
    frame_ready_condition_ = frag->make_condition<holoscan::AsynchronousCondition>(
        "frame_ready_condition");
    add_arg(frame_ready_condition_);
    frame_ready_condition_->event_state(holoscan::AsynchronousEventState::WAIT);
}

PcieReceiverOp::~PcieReceiverOp() = default;

void PcieReceiverOp::frame_ready()
{
    // Invoked from the monitor thread when a frame lands. running_ guards
    // against a late signal re-arming a condition stop() has retired.
    if (frame_ready_condition_ && running_.load()) {
        frame_ready_condition_->event_state(holoscan::AsynchronousEventState::EVENT_DONE);
    }
}

void PcieReceiverOp::start()
{
    // Cache the (possibly renamed) metadata key names once, so compute()
    // does no per-frame string work.
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
    frame_memory_key_ = rename("frame_memory");
    page_number_key_ = rename("page_number");
    dropped_key_ = rename("dropped");
    frames_key_ = rename("frames");

    // peer_ip comes first: without it nothing else here can be reached, and the error
    // should name the missing key rather than a downstream symptom. Resolving it to a
    // device node (and finding nothing bound to the BDF) is the receiver service's job.
    const EnumerationMetadata& metadata = metadata_.get();
    if (!metadata.contains(PEER_IP_KEY)
        || metadata.get<std::string>(PEER_IP_KEY, std::string {}).empty()) {
        throw std::runtime_error(fmt::format(
            "PcieReceiverOp: enumeration metadata has no '{}' key; on the PCIe transport it "
            "carries the endpoint's PCI address (DDDD:BB:DD.F) and is required to open the "
            "endpoint",
            PEER_IP_KEY));
    }

    // Sensor index: explicit parameter, else the supplement's per-sensor stamping.
    uint32_t sensor = 0;
    if (sensor_.get() >= 0) {
        sensor = static_cast<uint32_t>(sensor_.get());
    } else if (metadata.contains(SENSOR_KEY)) {
        const int64_t value = metadata.get<int64_t>(SENSOR_KEY);
        if (value < 0) {
            throw std::runtime_error(fmt::format(
                "PcieReceiverOp: metadata '{}' = {} is not a valid sensor index", SENSOR_KEY, value));
        }
        sensor = static_cast<uint32_t>(value);
    } else {
        throw std::runtime_error(fmt::format(
            "PcieReceiverOp: 'sensor' parameter is -1 and the metadata has no '{}' key; run "
            "Adapter::use_sensor(metadata, n) or pass the sensor explicitly",
            SENSOR_KEY));
    }

    // Ring layout, laid out as RoceReceiverOp lays out its pages: the frame, then the
    // endpoint's 128-byte metadata block at the next 128-byte boundary, is one page; the
    // endpoint cycles through `pages` of them. The frame size is the CSI converter's length;
    // what the bytes hold is not this operator's concern. The receiver validates the rest
    // (page count, 32-bit frame size, alignment) in start().
    const size_t frame_size = frame_size_.get();
    if (frame_size == 0) {
        throw std::runtime_error("PcieReceiverOp: frame_size must be positive (pass the CSI "
                                 "converter's length, CsiToBayerOp::get_csi_length)");
    }
    const size_t metadata_address = hololink::module::round_up(frame_size, hololink::module::PAGE_SIZE);
    const size_t page_size = metadata_address
        + hololink::module::round_up(PCIE_RX_METADATA_BYTES, hololink::module::PAGE_SIZE);
    const uint32_t pages = pages_.get();
    const size_t buffer_size = hololink::module::round_up(page_size * pages, getpagesize());

    // Resolve the supplement module that handled this metadata's enumeration, then the V1
    // services it published:
    //   - PcieDataChannelInterfaceV1 (per (serial, data_channel)): programs the endpoint's
    //     PCIE_DP block and MSI-X entry for the receiver and arms / disarms this sensor.
    //     Metadata-form get_service caches it and runs its configure(metadata), which also
    //     configures the DataChannelInterfaceV1 anchor and the per-board Hololink.
    //   - PcieReceiverInterfaceV1 (same key): the ring, the dma-buf mapping and the MSI-X
    //     completion wait; also the PcieReceiverAttachmentV1 the channel programs from.
    //     Fetched after the channel, which its configure() requires (as on RoCE).
    //   - FrameMetadataInterfaceV1 (per-module singleton): decodes the HSB EOF layout the
    //     endpoint writes; optional because a carrier module without one just falls back to
    //     the receiver's own parser.
    auto& adapter = hololink::module::Adapter::get_adapter();
    auto module = adapter.get_module(metadata);
    channel_ = PcieDataChannelInterfaceV1::get_service(metadata);
    receiver_ = PcieReceiverInterfaceV1::get_service(metadata);
    frame_metadata_ = hololink::module::FrameMetadataInterfaceV1::get_service(module, true);
    if (channel_->sensor() != sensor) {
        throw std::runtime_error(fmt::format(
            "PcieReceiverOp: 'sensor' = {} but the metadata's data channel is sensor {} "
            "(vector {}); select the sensor with Adapter::use_sensor(metadata, {}) instead of "
            "overriding it here",
            sensor, channel_->sensor(), channel_->vector(), sensor));
    }
    if (receiver_->sensor() != sensor) {
        throw std::runtime_error(fmt::format(
            "PcieReceiverOp: 'sensor' = {} but the metadata's receiver is sensor {}; select the "
            "sensor with Adapter::use_sensor(metadata, {}) instead of overriding it here",
            sensor, receiver_->sensor(), sensor));
    }

    emitted_frame_bytes_ = frame_size;
    HSB_LOG_INFO("PcieReceiverOp: {} sensor {} frame_size={:#x} page_size={:#x} pages={} "
                 "buffer_size={:#x}",
        receiver_->description(), sensor, frame_size, page_size, pages, buffer_size);

    // Device memory on every GPU: the receiver dma-buf exports this buffer to the endpoint
    // (cuMemGetHandleForAddressRange), which only device allocations support. The
    // descriptor's automatic placement would pick cuMemHostAlloc on an integrated GPU and
    // the export would fail before the driver is reached.
    frame_buffer_ = std::make_shared<hololink::module::ReceiverMemoryDescriptor>(
        frame_context_.get(), buffer_size,
        hololink::module::ReceiverMemoryDescriptor::Placement::Device);
    const uint64_t buffer = frame_buffer_->get();

    // Drop any timeout/frame counters left over from a prior run.
    ingress_stall_.reset();

    running_ = true;
    // Register the frame-ready callback before the monitor thread starts so no frame slips
    // through before the condition is armed. The service keeps the callback until it is
    // replaced and outlives this operator, and the callback captures `this`: every exit
    // from here on -- the failures below and stop()'s Cleanup -- clears it again, so a
    // later user of the same per-channel service cannot be called back into a destroyed
    // operator.
    if (frame_ready_condition_) {
        receiver_->set_frame_ready([this]() { frame_ready(); });
    }
    const auto fail_start = [this](const std::string& message) {
        running_ = false;
        receiver_->set_frame_ready({});
        receiver_->close();
        throw std::runtime_error(message);
    };

    // Bring the receiver up (device node opened, mapping, completion-counter baseline, vector
    // enabled; no register writes), then hand it to the channel, which programs the endpoint
    // from it and arms. start() zeroes every page's metadata block with cuMemsetD8 and needs
    // frame_context current on this thread; bind it explicitly rather than rely on the
    // allocation above having left it current.
    HOLOLINK_MODULE_CUDA_CHECK(cuCtxSetCurrent(frame_context_.get()));
    const hololink_module_status_t start_status
        = receiver_->start(buffer, buffer_size, frame_size, page_size, pages);
    if (start_status != HOLOLINK_MODULE_OK) {
        fail_start(fmt::format(
            "While starting PcieReceiverOp: receiver->start on {} sensor {} returned status {}",
            receiver_->description(), sensor, start_status));
    }
    const hololink_module_status_t attach_status
        = channel_->attach_receiver(receiver_, chunk_pages_.get());
    if (attach_status != HOLOLINK_MODULE_OK) {
        // The receiver is started (mapping made, vector enabled); fail_start() closes it.
        fail_start(fmt::format(
            "While starting PcieReceiverOp: channel->attach_receiver returned status {}",
            attach_status));
    }
    configured_ = true;
    HSB_LOG_INFO("PcieReceiverOp: channel attached and armed on {} sensor {} iova={:#x}",
        receiver_->description(), sensor, receiver_->iova());

    monitor_thread_ = std::make_shared<std::thread>(
        [self = receiver_, frame_context = frame_context_.get()]() {
            HOLOLINK_MODULE_CUDA_CHECK(cuCtxSetCurrent(frame_context));
            self->blocking_monitor();
        });
    // pthread_setname_np rejects names longer than 15 bytes (ERANGE), so truncate rather
    // than fail on a long operator name.
    const std::string thread_name = name().substr(0, 15);
    if (pthread_setname_np(monitor_thread_->native_handle(), thread_name.c_str()) != 0) {
        HSB_LOG_WARN("PcieReceiverOp: could not name the monitor thread '{}'", thread_name);
    }

    if (device_start_.get()) {
        try {
            device_start_.get()();
        } catch (...) {
            // By now the channel is armed and the monitor thread is running. Unwind both
            // through stop(): nothing guarantees stop() is called after a failed start(),
            // and a joinable std::thread terminates the process in its destructor.
            try {
                stop();
            } catch (const std::exception& e) {
                HSB_LOG_WARN("PcieReceiverOp: teardown after a failed device_start also failed: {}",
                    e.what());
            }
            throw;
        }
    }

    // Don't wait for the next signal if a frame is already queued.
    if (frame_ready_condition_ && receiver_->frames_ready()) {
        frame_ready_condition_->event_state(holoscan::AsynchronousEventState::EVENT_DONE);
    }
}

void PcieReceiverOp::stop()
{
    // Cleanup runs from this destructor so it executes even if other calls throw below.
    // By then the channel has been detached (the endpoint disarmed, this vector's MSI-X
    // entry re-masked); close() waits for the monitor to leave blocking_monitor() and only
    // then releases the mapping, so a completion that landed during the disarm is still
    // decoded on live memory; the join afterwards is then immediate. The mapping is released
    // before the buffer.
    class Cleanup {
    public:
        explicit Cleanup(PcieReceiverOp& self)
            : self_(self)
        {
        }
        ~Cleanup()
        {
            self_.configured_ = false;
            if (self_.receiver_) {
                // Unregister the callback that captures this operator before the service
                // is let go of: it persists across the service's start() calls.
                self_.receiver_->set_frame_ready({});
                self_.receiver_->close();
            }
            if (self_.monitor_thread_ && self_.monitor_thread_->joinable()) {
                self_.monitor_thread_->join();
            }
            self_.monitor_thread_.reset();
            self_.receiver_.reset();
            self_.frame_metadata_.reset();
            self_.channel_.reset();
            self_.frame_buffer_.reset();
        }

    private:
        PcieReceiverOp& self_;
    };
    Cleanup cleanup(*this);

    // Stop signaling and retire the condition before tearing the receiver down, so a frame
    // arriving mid-teardown can't re-arm it.
    running_ = false;
    if (frame_ready_condition_) {
        frame_ready_condition_->event_state(holoscan::AsynchronousEventState::EVENT_NEVER);
    }

    // Stand the monitor down before the sensor and the channel go: stopping the sensor
    // (device_stop) cuts the stream mid-frame, and the endpoint raises the completion for
    // that truncated transfer without ever writing its end-of-frame metadata block; the
    // disarm in detach_receiver() can do the same to a frame in flight. A monitor still
    // running would wait its settle time on that completion and log a spurious "no new
    // complete metadata block" miss. stop() touches no register and leaves the mapping in
    // place; close() in Cleanup still waits for the monitor to leave blocking_monitor()
    // before releasing memory.
    if (receiver_) {
        receiver_->stop();
    }

    // A device_stop failure must not skip the disarm below: Cleanup releases the mapping,
    // and the endpoint has to be quiet before that happens. Hold the exception, detach,
    // then rethrow it.
    std::exception_ptr device_stop_error;
    if (device_stop_.get()) {
        try {
            device_stop_.get()();
        } catch (const std::exception& e) {
            HSB_LOG_ERROR("PcieReceiverOp: device_stop failed ({}); disarming the channel "
                          "before rethrowing",
                e.what());
            device_stop_error = std::current_exception();
        } catch (...) {
            device_stop_error = std::current_exception();
        }
    }

    // Disarm before the receiver's memory goes away (same order as RoceReceiverOp).
    if (configured_ && channel_) {
        const hololink_module_status_t status = channel_->detach_receiver();
        if (status != HOLOLINK_MODULE_OK) {
            throw std::runtime_error(fmt::format(
                "While stopping PcieReceiverOp: channel->detach_receiver returned status {}",
                status));
        }
    }
    if (device_stop_error) {
        std::rethrow_exception(device_stop_error);
    }
}

void PcieReceiverOp::log_stall_diagnostics()
{
    if (!channel_) {
        return;
    }
    hololink::module::hsb_pcie::PcieChannelDiagnostics d;
    try {
        if (channel_->diagnostics(d) != HOLOLINK_MODULE_OK) {
            HSB_LOG_WARN("PcieReceiverOp: channel diagnostics unavailable");
            return;
        }
    } catch (const std::exception& e) {
        HSB_LOG_WARN("PcieReceiverOp: channel diagnostics unavailable ({})", e.what());
        return;
    }
    HSB_LOG_WARN("PcieReceiverOp stall diagnostics, sensor {}: pack_tvalid={:#x} pack_psn={:#x} "
                 "pack_ctrl={:#x} dp_pkt_stat={:#x} vip_mask={:#x} dp_pkt_len={:#x} "
                 "buf_start={:#x}:{:#x} buf_len={:#x} buf_inc={:#x} ptr_range={:#x} "
                 "requester_id={:#x} metadata_ctrl={:#x} "
                 "msix_addr={:#x}:{:#x} msix_data={:#x} msix_vector_control={:#x} "
                 "msix_pba={:#x}:{:#x} scratch={:#x}",
        channel_->sensor(), d.pack_tvalid, d.pack_psn, d.pack_ctrl, d.dp_pkt_stat, d.vip_mask,
        d.dp_pkt_len, d.fixed_msb, d.buf_start, d.buf_len, d.buf_inc, d.ptr_range,
        d.requester_id, d.metadata_ctrl, d.msix_addr_hi, d.msix_addr_lo,
        d.msix_data, d.msix_vector_control, d.msix_pba_hi, d.msix_pba_lo, d.scratch);
    HSB_LOG_WARN("  reading it: pack_tvalid not advancing between two stalls = the CSI receive "
                 "block sees no sensor data (D-PHY lanes/rate, lane order, RX mux); advancing "
                 "with no completion = DMA or MSI-X path (msix_vector_control 1 = masked, a "
                 "set msix_pba bit = pending but undelivered; check dmesg for SMMU/IOMMU faults)");
}

void PcieReceiverOp::compute(holoscan::InputContext& /*op_input*/,
    holoscan::OutputContext& op_output,
    holoscan::ExecutionContext& context)
{
    // Holoscan worker threads don't carry frame_context_; bind it here every compute so any
    // context-sensitive driver call downstream of the tensor sees the frame's context.
    HOLOLINK_MODULE_CUDA_CHECK(cuCtxSetCurrent(frame_context_.get()));

    PcieReceiverFrameInfoV1 frame_info;
    if (!receiver_->get_next_frame(GET_NEXT_FRAME_TIMEOUT_MS, frame_info)) {
        // Timeout: produce no output; the scheduler will revisit us.
        if (ingress_stall_.on_timeout(GET_NEXT_FRAME_TIMEOUT_MS)) {
            HSB_LOG_WARN("Ingress stalled: no PCIe frame for {} consecutive {}ms polls "
                         "after {} received frames",
                ingress_stall_.consecutive_timeouts(), GET_NEXT_FRAME_TIMEOUT_MS,
                ingress_stall_.frames_received());
            log_stall_diagnostics();
        }
        return;
    }
    if (const uint64_t stalled_polls = ingress_stall_.on_frame()) {
        HSB_LOG_WARN("Ingress resumed after {} empty polls", stalled_polls);
    }

    if (frame_ready_condition_) {
        // Re-arm: wait for the next monitor signal unless another frame is already queued.
        // Set WAIT before the frames_ready() test to avoid a lost wakeup.
        frame_ready_condition_->event_state(holoscan::AsynchronousEventState::EVENT_WAITING);
        if (receiver_->frames_ready()) {
            frame_ready_condition_->event_state(holoscan::AsynchronousEventState::EVENT_DONE);
        }
    }

    // The slot's metadata block is the 48-byte HSB end-of-frame block followed by zero
    // padding (the receiver delivers no frame without one), so when the module publishes a
    // FrameMetadataInterfaceV1 let it own the decode (psn, crc, timestamps included); the
    // receiver's own reading of flags / frame_number / bytes_written covers the no-service
    // case. The receiver already copied the block out, so no second device read is needed.
    hololink::module::FrameMetadataInterfaceV1::FrameMetadata v1_metadata {};
    bool decoded = false;
    if (frame_metadata_ && frame_metadata_->block_size() <= frame_info.raw_metadata.size()) {
        decoded = frame_metadata_->decode(frame_info.raw_metadata.data(),
                      frame_info.raw_metadata.size(), v1_metadata)
            == HOLOLINK_MODULE_OK;
    }
    if (!decoded) {
        v1_metadata.flags = frame_info.flags;
        v1_metadata.frame_number = static_cast<uint16_t>(frame_info.frame_number & 0xFFFFu);
        v1_metadata.bytes_written = frame_info.bytes_written;
    }

    // Wrap the just-received frame slot in a GXF tensor.
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
    if (emitted_frame_bytes_ > static_cast<size_t>(INT_MAX)) {
        throw std::runtime_error(
            "While building output entity: frame size exceeds INT_MAX");
    }
    const nvidia::gxf::Shape shape { static_cast<int>(emitted_frame_bytes_) };
    constexpr nvidia::gxf::PrimitiveType element_type
        = nvidia::gxf::PrimitiveType::kUnsigned8;
    const uint64_t element_size
        = nvidia::gxf::PrimitiveTypeSize(element_type);
    // Keep the frame buffer alive for as long as this wrapped tensor is referenced anywhere
    // downstream, so stop() freeing frame_buffer_ cannot pull memory out from under an
    // in-flight frame.
    std::shared_ptr<void> buffer_owner = frame_buffer_;
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

    // Stamp the operator's metadata map so downstream operators can read the fields by name.
    auto meta_map = metadata();
    meta_map->set(flags_key_, static_cast<int64_t>(v1_metadata.flags));
    meta_map->set(psn_key_, static_cast<int64_t>(v1_metadata.psn));
    meta_map->set(crc_key_, static_cast<int64_t>(v1_metadata.crc));
    // The monotonic, wraparound-extended counter rather than the 16-bit wire value.
    meta_map->set(frame_number_key_, static_cast<int64_t>(frame_info.frame_number));
    meta_map->set(timestamp_s_key_, static_cast<int64_t>(v1_metadata.timestamp_s));
    meta_map->set(timestamp_ns_key_, static_cast<int64_t>(v1_metadata.timestamp_ns));
    meta_map->set(bytes_written_key_, static_cast<int64_t>(v1_metadata.bytes_written));
    meta_map->set(metadata_s_key_, static_cast<int64_t>(v1_metadata.metadata_s));
    meta_map->set(metadata_ns_key_, static_cast<int64_t>(v1_metadata.metadata_ns));
    meta_map->set(received_s_key_, static_cast<int64_t>(frame_info.received_s));
    meta_map->set(received_ns_key_, static_cast<int64_t>(frame_info.received_ns));
    meta_map->set(frame_memory_key_, static_cast<int64_t>(frame_info.frame_memory));
    meta_map->set(page_number_key_, static_cast<int64_t>(frame_info.page_number));
    meta_map->set(dropped_key_, static_cast<int64_t>(frame_info.dropped));
    meta_map->set(frames_key_, static_cast<int64_t>(frame_info.frames));

    op_output.emit(out_entity.value(), "output");
}

} // namespace hololink::module::operators
