/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// PcieReceiverOp: module-native PCIe receiver operator. Port of
// the reference bring-up tree's receiver operator onto the shape of
// roce_receiver_op.cpp (V1 services from the enumeration metadata, AsynchronousCondition
// driven compute, RAII stop). Register programming is the hsb_pcie module's
// PcieDataChannelInterfaceV1 (attach_receiver / detach_receiver, like the RoCE channel);
// memory and completion waits live in hololink::module::pcie::PcieReceiver. No legacy core.

#include "hololink/module/operators/pcie_receiver_op.hpp"

#include <climits>
#include <cstdint>
#include <exception>
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
#include "hololink/module/cuda_unique.hpp" // HOLOLINK_MODULE_CUDA_CHECK
#include "hololink/module/hsb_pcie/pcie_data_channel.hpp"
#include "hololink/module/logging.hpp" // HSB_LOG_INFO
#include "hololink/module/module.hpp"
#include "hololink/module/pcie/pcie_device_access.hpp"
#include "hololink/module/pcie/pcie_receiver.hpp"
#include "hololink/module/pcie/pcie_rx_geometry.hpp"
#include "hololink/module/pcie/pcie_sysfs.hpp" // BDF -> /dev/hololink_pcieN
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

    using hololink::module::hsb_pcie::PcieDataChannelInterfaceV1;
    using hololink::module::pcie::FrameBufferSource;
    using hololink::module::pcie::PcieDeviceAccess;
    using hololink::module::pcie::PcieFrameInfo;
    using hololink::module::pcie::PcieReceiver;
    using hololink::module::pcie::RxFrameGeometry;
    using hololink::module::pcie::RxMetadataLayout;
    using hololink::module::pcie::RxRingConfig;
    using hololink::module::pcie::RxRingLayout;

} // namespace

std::string PcieReceiverOp::device_node_from_metadata(const EnumerationMetadata& metadata)
{
    if (!metadata.contains(PEER_IP_KEY)) {
        throw std::runtime_error(fmt::format(
            "PcieReceiverOp: enumeration metadata has no '{}' key; on the PCIe transport it "
            "carries the endpoint's PCI address (DDDD:BB:DD.F), from which the device node is "
            "derived, and is required to open the endpoint",
            PEER_IP_KEY));
    }
    const std::string peer = metadata.get<std::string>(PEER_IP_KEY, std::string {});
    if (peer.empty()) {
        throw std::runtime_error(fmt::format(
            "PcieReceiverOp: enumeration metadata '{}' must be a non-empty string: the "
            "endpoint's PCI address DDDD:BB:DD.F",
            PEER_IP_KEY));
    }
    // Convenience spelling: an explicit node names itself, no sysfs needed.
    if (hololink::module::pcie::minor_from_device_node(peer)) {
        return peer;
    }
    // The BDF is authoritative: the node the driver bound to that PCI function right now.
    // (The metadata's pcie_device_node is informational and may be stale after a reload.)
    const auto bdf = hololink::module::pcie::bdf_for_peer(peer);
    if (!bdf) {
        throw std::runtime_error(fmt::format(
            "PcieReceiverOp: '{}' = '{}' is neither a PCI address DDDD:BB:DD.F nor a "
            "/dev/hololink_pcieN node",
            PEER_IP_KEY, peer));
    }
    const auto node = hololink::module::pcie::device_node_for_bdf(*bdf);
    if (!node) {
        throw std::runtime_error(fmt::format(
            "PcieReceiverOp: no hololink_pcie device is bound to PCI function {} ('{}' = '{}'); "
            "is the endpoint present and bound to the hololink_pcie driver?",
            *bdf, PEER_IP_KEY, peer));
    }
    return *node;
}

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
        "(DDDD:BB:DD.F), from which the device node is derived through sysfs, and the "
        "per-board HololinkInterfaceV1 is resolved from it");
    spec.param(frame_context_, "frame_context", "FrameContext",
        "CUDA context for the frame memory");
    spec.param(frame_size_, "frame_size", "FrameSize",
        "Per-frame payload size in bytes; 0 derives it from the geometry, a non-zero "
        "value must equal the geometry's DMA frame size",
        size_t { 0 });
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

    spec.param(num_buffers_, "num_buffers", "NumBuffers",
        "Number of frame slots in the receive ring", uint32_t { 4 });
    spec.param(chunk_pages_, "chunk_pages", "ChunkPages",
        "Data TLP size in 128-byte pages (DP_PKT_LEN, shared by every sensor on the "
        "endpoint); 1 gives maximum-size TLPs. Passed to the channel's attach_receiver.",
        uint32_t { 1 });
    spec.param(window_bytes_, "window_bytes", "WindowBytes",
        "Payload window per slot; 0 means exactly the frame size", size_t { 0 });

    spec.param(width_, "width", "Width", "Image width in pixels", uint32_t { 1920 });
    spec.param(height_, "height", "Height", "Image height in pixels", uint32_t { 1080 });
    spec.param(stride_pixels_, "stride_pixels", "StridePixels",
        "Pixels per transmitted line, which may exceed width; 0 means width", uint32_t { 0 });
    spec.param(crop_x_, "crop_x", "CropX", "Horizontal crop offset in pixels", uint32_t { 0 });
    spec.param(line_alignment_, "line_alignment", "LineAlignment",
        "RAW10 line padding; use 64 only with FPGA packetizer padding enabled", uint32_t { 8 });
    spec.param(raw_start_byte_, "raw_start_byte", "RawStartByte",
        "Leading prefix the sensor emits before the image", uint32_t { 0 });
    spec.param(frame_buffer_source_, "frame_buffer_source", "FrameBufferSource",
        "\"caller\": dma-buf import a ReceiverMemoryDescriptor in frame_context (VRAM); "
        "\"device\": host-staged memory allocated through the device (opt-in)",
        std::string("caller"));

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

    // The device node comes first: without it nothing else here can be reached, and the
    // error should name the missing key / unbound BDF rather than a downstream symptom.
    const EnumerationMetadata& metadata = metadata_.get();
    const std::string device_node = device_node_from_metadata(metadata);

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

    // Geometry (packed RAW10, the only format the Sensor RX path delivers).
    RxFrameGeometry geometry;
    geometry.width = width_.get();
    geometry.height = height_.get();
    geometry.stride_pixels = stride_pixels_.get() ? stride_pixels_.get() : width_.get();
    geometry.crop_x = crop_x_.get();
    geometry.line_alignment = line_alignment_.get();
    geometry.raw_start_byte = raw_start_byte_.get();
    if (const char* error = geometry.validate()) {
        throw std::runtime_error(fmt::format("PcieReceiverOp: invalid frame geometry: {}", error));
    }
    const uint64_t frame_dma_bytes = geometry.frame_dma_bytes();
    if (frame_size_.get() != 0 && frame_size_.get() != frame_dma_bytes) {
        throw std::runtime_error(fmt::format(
            "PcieReceiverOp: frame_size={} does not match the geometry's DMA frame size {} "
            "({}x{} stride {} RAW10 start {}); pass 0 to derive it",
            frame_size_.get(), frame_dma_bytes, geometry.width, geometry.height,
            geometry.stride_pixels, geometry.raw_start_byte));
    }
    emitted_frame_bytes_ = frame_dma_bytes;

    FrameBufferSource source;
    if (frame_buffer_source_.get() == "caller") {
        source = FrameBufferSource::Caller;
    } else if (frame_buffer_source_.get() == "device") {
        source = FrameBufferSource::DeviceAccess;
    } else {
        throw std::runtime_error(fmt::format(
            "PcieReceiverOp: frame_buffer_source must be caller or device, got '{}'",
            frame_buffer_source_.get()));
    }

    RxRingConfig config;
    config.sensor = sensor;
    config.num_buffers = num_buffers_.get();
    config.window_bytes = window_bytes_.get();

    // Size the allocation from the ring layout rather than from frame_size: slots are
    // padded out to a 4 KiB boundary and each carries a metadata block past its window.
    const RxRingLayout layout = RxRingLayout::compute(geometry, config.num_buffers, config.window_bytes);
    const size_t buffer_size = layout.required_bytes();

    // Resolve the supplement module that handled this metadata's enumeration, then the V1
    // services it published:
    //   - PcieDataChannelInterfaceV1 (per (serial, data_channel)): programs the endpoint's
    //     PCIE_DP block and MSI-X entry for the receiver and arms / disarms this sensor.
    //     Metadata-form get_service caches it and runs its configure(metadata), which also
    //     configures the DataChannelInterfaceV1 anchor and the per-board Hololink.
    //   - FrameMetadataInterfaceV1 (per-module singleton): decodes the HSB EOF layout the
    //     endpoint writes; optional because a carrier module without one just falls back to
    //     the receiver's own parser.
    auto& adapter = hololink::module::Adapter::get_adapter();
    auto module = adapter.get_module(metadata);
    channel_ = PcieDataChannelInterfaceV1::get_service(metadata);
    frame_metadata_ = hololink::module::FrameMetadataInterfaceV1::get_service(module, true);
    if (channel_->sensor() != sensor) {
        throw std::runtime_error(fmt::format(
            "PcieReceiverOp: 'sensor' = {} but the metadata's data channel is sensor {} "
            "(vector {}); select the sensor with Adapter::use_sensor(metadata, {}) instead of "
            "overriding it here",
            sensor, channel_->sensor(), channel_->vector(), sensor));
    }

    // The receiver's own handle on the endpoint. Requester id from the driver identity.
    auto access = std::make_shared<PcieDeviceAccess>(device_node,
        source == FrameBufferSource::DeviceAccess);
    device_access_ = access;
    const auto identity = access->identity();
    HSB_LOG_INFO("PcieReceiverOp: {} bdf={} requester_id={:#x} msix_vectors={} sensor={} "
                 "frame_dma_bytes={:#x} buffer_size={:#x} slot_stride={:#x} slots={}",
        device_node, identity.bdf, identity.requester_id, identity.num_msix_vectors, sensor,
        frame_dma_bytes, buffer_size, layout.slot_stride, config.num_buffers);

    CUdeviceptr buffer = 0;
    if (source == FrameBufferSource::Caller) {
        frame_buffer_ = std::make_shared<hololink::module::ReceiverMemoryDescriptor>(
            frame_context_.get(), buffer_size);
        buffer = frame_buffer_->get();
    }

    receiver_ = std::make_shared<PcieReceiver>(device_access_, geometry, config, buffer,
        buffer_size, source);

    // Drop any timeout/frame counters left over from a prior run.
    ingress_stall_.reset();

    running_ = true;
    // Register the frame-ready callback before the monitor thread starts so no frame slips
    // through before the condition is armed.
    if (frame_ready_condition_) {
        receiver_->set_frame_ready([this]() { frame_ready(); });
    }

    // Bring the receiver up (mapping, completion-counter baseline, vector enabled; no register
    // writes), then hand it to the channel, which programs the endpoint from it and arms.
    // start() zeroes every slot's metadata block with cuMemsetD8 and needs frame_context
    // current on this thread; bind it explicitly rather than rely on the allocation above
    // having left it current (it does today, but the DeviceAccess source skips that path).
    HOLOLINK_MODULE_CUDA_CHECK(cuCtxSetCurrent(frame_context_.get()));
    if (!receiver_->start()) {
        running_ = false;
        throw std::runtime_error(fmt::format(
            "PcieReceiverOp: failed to start the PCIe receiver on {} sensor {}", device_node, sensor));
    }
    const hololink_module_status_t attach_status
        = channel_->attach_receiver(receiver_, chunk_pages_.get());
    if (attach_status != HOLOLINK_MODULE_OK) {
        running_ = false;
        throw std::runtime_error(fmt::format(
            "While starting PcieReceiverOp: channel->attach_receiver returned status {}",
            attach_status));
    }
    configured_ = true;
    HSB_LOG_INFO("PcieReceiverOp: channel attached and armed on {} sensor {} iova={:#x}",
        device_node, sensor, receiver_->iova());

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
                self_.receiver_->close();
            }
            if (self_.monitor_thread_ && self_.monitor_thread_->joinable()) {
                self_.monitor_thread_->join();
            }
            self_.monitor_thread_.reset();
            self_.receiver_.reset();
            self_.device_access_.reset();
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

    PcieFrameInfo frame_info;
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

    // The endpoint's current metadata layout is the 48-byte HSB end-of-frame block followed
    // by zero padding, so when the module publishes a FrameMetadataInterfaceV1 let it own the
    // decode (psn, crc, timestamps included). The receiver's local parser covers the legacy
    // WR-IMM layout and the no-service case; the receiver already copied the block out, so
    // no second device read is needed here.
    hololink::module::FrameMetadataInterfaceV1::FrameMetadata v1_metadata {};
    bool decoded = false;
    if (frame_metadata_ && frame_info.metadata_layout == RxMetadataLayout::CURRENT
        && frame_metadata_->block_size() <= frame_info.raw_metadata.size()) {
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
    // Keep the receiver's frame buffer alive for as long as this wrapped tensor is referenced
    // anywhere downstream, so stop() freeing frame_buffer_ cannot pull memory out from under
    // an in-flight frame. For the host-staged source the device access owns the memory.
    std::shared_ptr<void> buffer_owner = frame_buffer_;
    if (!buffer_owner) {
        buffer_owner = device_access_;
    }
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
