/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hololink/module/operators/fusa_coe_capture_op.hpp"

#include <ctime> // clock_gettime, CLOCK_REALTIME, struct timespec

#include <gxf/std/tensor.hpp>
#include <yaml-cpp/yaml.h>

#include "hololink/module/adapter.hpp"
#include "hololink/module/logging.hpp" // HSB_LOG_*
#include "hololink/module/status.h"

#define HOLOLINK_MODULE_YAML_CONVERTER_UNSUPPORTED(TYPE)                                    \
    template <>                                                                             \
    struct YAML::convert<TYPE> {                                                            \
        static Node encode(TYPE&) { throw std::runtime_error("Unsupported"); }              \
        static bool decode(const Node&, TYPE&) { throw std::runtime_error("Unsupported"); } \
    };

HOLOLINK_MODULE_YAML_CONVERTER_UNSUPPORTED(hololink::module::EnumerationMetadata);
HOLOLINK_MODULE_YAML_CONVERTER_UNSUPPORTED(std::function<void()>);
HOLOLINK_MODULE_YAML_CONVERTER_UNSUPPORTED(
    std::function<std::string(const std::string&)>);

namespace hololink::module::operators {

// File-local helper adapters, declared at namespace scope (the project style
// avoids anonymous namespaces in .cpp files).
class AdapterCoeChannelConfig : public hololink::operators::fusa_coe_capture::CoeChannelConfig {
public:
    explicit AdapterCoeChannelConfig(
        const std::shared_ptr<CoeDataChannelInterfaceV1>& channel)
        : channel_(channel)
    {
    }

    void set_packetizer_if_needed(
        bool csi_pixel_format, hololink::csi::PixelFormat pixel_format) override
    {
        if (!csi_pixel_format) {
            return;
        }
        const hololink_module_status_t status = channel_->set_packetizer_for_pixel_format(
            static_cast<uint32_t>(pixel_format));
        if (status != HOLOLINK_MODULE_OK) {
            throw std::runtime_error(
                "While starting FusaCoeCaptureOp: set_packetizer_for_pixel_format "
                "returned status "
                + std::to_string(status));
        }
    }

    void configure_coe(
        uint8_t coe_channel, size_t frame_size, uint32_t bytes_per_line) override
    {
        const hololink_module_status_t status = channel_->configure_coe(
            coe_channel, frame_size, bytes_per_line);
        if (status != HOLOLINK_MODULE_OK) {
            throw std::runtime_error(
                std::string("While starting FusaCoeCaptureOp: configure_coe returned status ")
                + std::to_string(status));
        }
    }

    void unconfigure() override
    {
        const hololink_module_status_t status = channel_->unconfigure();
        if (status != HOLOLINK_MODULE_OK) {
            throw std::runtime_error(
                std::string("While stopping FusaCoeCaptureOp: channel->unconfigure "
                            "returned status ")
                + std::to_string(status));
        }
    }

private:
    std::shared_ptr<CoeDataChannelInterfaceV1> channel_;
};

class AdapterCoeMetadataDecoder
    : public hololink::operators::fusa_coe_capture::CoeMetadataDecoder {
public:
    explicit AdapterCoeMetadataDecoder(
        const std::shared_ptr<FrameMetadataInterfaceV1>& frame_metadata)
        : frame_metadata_(frame_metadata)
    {
    }

    bool decode(
        const void* host_memory, size_t host_memory_size,
        hololink::operators::fusa_coe_capture::CoeFrameMetadata& out) const override
    {
        const size_t block_size = frame_metadata_->block_size();
        if (host_memory_size < block_size) {
            return false;
        }

        FrameMetadataInterfaceV1::FrameMetadata v1_metadata {};
        const hololink_module_status_t decode_status = frame_metadata_->decode(
            host_memory, block_size, v1_metadata);
        if (decode_status != HOLOLINK_MODULE_OK) {
            throw std::runtime_error(
                std::string("While decoding frame metadata: status ")
                + std::to_string(decode_status));
        }

        out.timestamp_s = static_cast<int64_t>(v1_metadata.timestamp_s);
        out.timestamp_ns = static_cast<int64_t>(v1_metadata.timestamp_ns);
        out.metadata_s = static_cast<int64_t>(v1_metadata.metadata_s);
        out.metadata_ns = static_cast<int64_t>(v1_metadata.metadata_ns);
        out.crc = static_cast<int64_t>(v1_metadata.crc);
        out.frame_number = static_cast<int64_t>(v1_metadata.frame_number);
        return true;
    }

private:
    std::shared_ptr<FrameMetadataInterfaceV1> frame_metadata_;
};

void FusaCoeCaptureOp::setup(holoscan::OperatorSpec& spec)
{
    spec.output<holoscan::gxf::Entity>("output");

    register_converter<EnumerationMetadata>();
    register_converter<std::function<void()>>();
    register_converter<std::function<std::string(const std::string&)>>();

    spec.param(interface_, "interface", "Interface", "Interface for the CoE device.");
    spec.param(mac_addr_, "mac_addr", "MACAddr", "MAC Address for the CoE device.");
    spec.param(timeout_, "timeout", "Timeout", "Timeout for capture requests, in milliseconds");
    spec.param(out_tensor_name_, "out_tensor_name", "OutputTensorName",
        "Name of the output tensor", std::string(""));
    spec.param(metadata_, "enumeration_metadata", "EnumerationMetadata",
        "Enumeration metadata for the channel — used to resolve the CoeDataChannel "
        "and frame-metadata services");
    spec.param(device_start_, "device_start", "DeviceStart",
        "Function to be called to start the device");
    spec.param(device_stop_, "device_stop", "DeviceStop",
        "Function to be called to stop the device");
    spec.param(rename_metadata_, "rename_metadata", "RenameMetadata",
        "Optional function mapping each emitted metadata key to a new "
        "name (e.g. add a per-leg prefix). Defaults to identity.",
        std::function<std::string(const std::string&)> {});

    // compute() is driven by an AsynchronousCondition that the monitor thread
    // signals on each acquired frame, so it never blocks the scheduler waiting
    // for a frame (the blocking wait_for_acquired_buffer lives on the monitor
    // thread). The condition must be created and added as an arg during
    // setup(), before start() runs.
    auto* frag = fragment();
    frame_ready_condition_ = frag->make_condition<holoscan::AsynchronousCondition>(
        "frame_ready_condition");
    add_arg(frame_ready_condition_);
    frame_ready_condition_->event_state(holoscan::AsynchronousEventState::WAIT);
}

void FusaCoeCaptureOp::start()
{
    // Cache the (possibly renamed) metadata key names once, so compute() does
    // no per-frame string work. Default to identity when no rename_metadata
    // function was supplied.
    auto rename = rename_metadata_.get();
    if (!rename) {
        rename = [](const std::string& name) { return name; };
    }
    timestamp_s_key_ = rename("timestamp_s");
    timestamp_ns_key_ = rename("timestamp_ns");
    metadata_s_key_ = rename("metadata_s");
    metadata_ns_key_ = rename("metadata_ns");
    crc_key_ = rename("crc");
    frame_number_key_ = rename("frame_number");
    received_s_key_ = rename("received_s");
    received_ns_key_ = rename("received_ns");

    auto& adapter = hololink::module::Adapter::get_adapter();
    auto module = adapter.get_module(metadata_.get());
    channel_ = CoeDataChannelInterfaceV1::get_service(metadata_.get());
    frame_metadata_ = FrameMetadataInterfaceV1::get_service(module);

    AdapterCoeChannelConfig channel_config(channel_);
    core_.start(
        interface_.get(),
        mac_addr_.get(),
        timeout_.get(),
        channel_config,
        device_start_.get(),
        device_stop_.get());

    // Start the monitor thread that owns the blocking wait_for_acquired_buffer.
    // It hands each frame to compute() via the AsynchronousCondition, so the
    // scheduler thread is never blocked on capture. core_.start() has already
    // fired device_start (the sensor is streaming), so buffers can arrive now.
    {
        std::lock_guard<std::mutex> lock(handoff_mutex_);
        frame_ready_ = false;
        frame_consumed_ = true;
    }
    running_ = true;
    monitor_thread_ = std::thread([this]() { monitor(); });

    // Instrumentation: correlate this leg (out_tensor_name) with its monitor
    // thread so the shutdown ordering in the log is unambiguous.
    HSB_LOG_DEBUG("FusaCoeCaptureOp::start: leg \"{}\" started; monitor thread launched",
        out_tensor_name_.get());
}

FusaCoeCaptureOp::~FusaCoeCaptureOp()
{
    // Defensive: if the graph tore down without stop() (e.g. an early error),
    // make sure the monitor thread is joined so ~thread doesn't terminate().
    running_ = false;
    {
        std::lock_guard<std::mutex> lock(handoff_mutex_);
        frame_consumed_ = true;
    }
    handoff_cv_.notify_all();
    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }
}

void FusaCoeCaptureOp::monitor()
{
    while (running_.load()) {
        hololink::operators::fusa_coe_capture::FusaCoeCaptureCore::BufferView view {};
        // wait_for_acquired_buffer returns false on a plain per-request timeout
        // (no frame within the window) and throws only on a genuine capture
        // failure. A timeout is normal at startup and between frames, so retry.
        // A failure is deliberately left uncaught: it propagates out of this
        // monitor thread and terminates the process, surfacing the fault loudly
        // (and landing directly on it under a debugger) instead of being
        // swallowed into a silently stalled pipeline.
        if (!core_.wait_for_acquired_buffer(
                timeout_.get(), out_tensor_name_.get().c_str(), view)) {
            if (!running_.load()) {
                break;
            }
            continue;
        }

        // received_s/ns is the host-side reception time: the CLOCK_REALTIME
        // instant the CPU woke up with the frame-ready notification — i.e. the
        // moment wait_for_acquired_buffer returned. Captured here (not in
        // compute) so it stays close to that wakeup, matching the RoCE/Linux
        // receivers' received_* semantics.
        struct timespec received = {};
        clock_gettime(CLOCK_REALTIME, &received);

        {
            std::lock_guard<std::mutex> lock(handoff_mutex_);
            pending_buffer_ = view;
            pending_received_ = received;
            frame_ready_ = true;
            frame_consumed_ = false;
        }
        if (running_.load() && frame_ready_condition_) {
            frame_ready_condition_->event_state(
                holoscan::AsynchronousEventState::EVENT_DONE);
        }

        // Wait until compute() has emitted this frame before acquiring the
        // next, preserving the core's single acquire -> register_pending_output
        // -> release cycle (buffer_in_compute_ is a one-deep slot).
        std::unique_lock<std::mutex> lock(handoff_mutex_);
        handoff_cv_.wait(
            lock, [this]() { return frame_consumed_ || !running_.load(); });
        if (!running_.load()) {
            break;
        }
    }
}

void FusaCoeCaptureOp::stop()
{
    // Instrumentation: mark when the pipeline asks this leg to stop, to test the
    // theory that a stop() (e.g. on the peer leg during graph teardown) is what
    // provokes the "CoE capture failed (error = 5)" in the still-running leg.
    HSB_LOG_DEBUG("FusaCoeCaptureOp::stop: entry, leg \"{}\"", out_tensor_name_.get());

    // Stop the monitor first so no wait_for_acquired_buffer is in flight when
    // the core tears down its buffers and acquire thread below. (The monitor
    // may take up to one capture timeout to notice, if it is mid-wait.)
    running_ = false;
    if (frame_ready_condition_) {
        frame_ready_condition_->event_state(
            holoscan::AsynchronousEventState::EVENT_NEVER);
    }
    {
        std::lock_guard<std::mutex> lock(handoff_mutex_);
        frame_consumed_ = true;
    }
    handoff_cv_.notify_all();
    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }

    HSB_LOG_DEBUG("FusaCoeCaptureOp::stop: leg \"{}\" monitor joined", out_tensor_name_.get());

    if (!channel_) {
        return;
    }

    AdapterCoeChannelConfig channel_config(channel_);
    HSB_LOG_DEBUG("FusaCoeCaptureOp::stop: leg \"{}\" calling core_.stop()",
        out_tensor_name_.get());
    core_.stop(channel_config, device_stop_.get());
    channel_.reset();
    frame_metadata_.reset();
    HSB_LOG_DEBUG("FusaCoeCaptureOp::stop: leg \"{}\" complete", out_tensor_name_.get());
}

void FusaCoeCaptureOp::compute(holoscan::InputContext& /*op_input*/,
    holoscan::OutputContext& op_output,
    holoscan::ExecutionContext& context)
{
    // Non-blocking: the monitor thread already did the blocking wait and handed
    // us an acquired buffer (that's why the AsynchronousCondition fired). Take
    // it; if there's nothing (a spurious tick), do nothing.
    hololink::operators::fusa_coe_capture::FusaCoeCaptureCore::BufferView buffer {};
    struct timespec received = {};
    {
        std::unique_lock<std::mutex> lock(handoff_mutex_);
        if (!frame_ready_) {
            return;
        }
        buffer = pending_buffer_;
        received = pending_received_;
        frame_ready_ = false;
    }

    // Re-arm before releasing the monitor so a fast next frame can't be lost:
    // the monitor won't acquire the next buffer until frame_consumed_ is set
    // below, by which point the condition is back in WAITING.
    if (frame_ready_condition_) {
        frame_ready_condition_->event_state(
            holoscan::AsynchronousEventState::EVENT_WAITING);
    }

    if (is_metadata_enabled()) {
        // received_s/ns is the host-side reception time captured by the monitor
        // thread right after wait_for_acquired_buffer returned; CLOCK_REALTIME
        // shares the PTP-disciplined domain of the FPGA timestamps, so a
        // downstream "received - timestamp" latency is meaningful.
        hololink::operators::fusa_coe_capture::CoeFrameMetadata frame_metadata {};
        AdapterCoeMetadataDecoder decoder(frame_metadata_);
        if (core_.decode_metadata(buffer, decoder, frame_metadata)) {
            auto const& meta = metadata();
            meta->set(timestamp_s_key_, frame_metadata.timestamp_s);
            meta->set(timestamp_ns_key_, frame_metadata.timestamp_ns);
            meta->set(metadata_s_key_, frame_metadata.metadata_s);
            meta->set(metadata_ns_key_, frame_metadata.metadata_ns);
            meta->set(crc_key_, frame_metadata.crc);
            meta->set(frame_number_key_, frame_metadata.frame_number);
            meta->set(received_s_key_, static_cast<int64_t>(received.tv_sec));
            meta->set(received_ns_key_, static_cast<int64_t>(received.tv_nsec));
        }
    }

    auto entity = holoscan::gxf::Entity::New(&context);
    auto name = out_tensor_name_.get().c_str();
    auto tensor = static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::Tensor>(name);
    if (!tensor) {
        throw std::runtime_error("Failed to add GXF Tensor");
    }

    nvidia::gxf::Shape shape {
        static_cast<int>(core_.bytes_per_line() * core_.pixel_height())
    };
    const auto element_type = nvidia::gxf::PrimitiveType::kUnsigned8;
    const auto element_size = nvidia::gxf::PrimitiveTypeSize(element_type);

    void* data_ptr = static_cast<uint8_t*>(buffer.cuda_device_ptr) + core_.start_byte();
    if (!tensor.value()->wrapMemory(shape, element_type, element_size,
            nvidia::gxf::ComputeTrivialStrides(shape, element_size),
            nvidia::gxf::MemoryStorageType::kDevice,
            data_ptr, buffer_release_callback)) {
        throw std::runtime_error("Failed to add wrapped memory");
    }

    core_.register_pending_output(data_ptr);

    op_output.emit(entity, "output");

    // Frame emitted: let the monitor issue the next wait_for_acquired_buffer.
    {
        std::lock_guard<std::mutex> lock(handoff_mutex_);
        frame_consumed_ = true;
    }
    handoff_cv_.notify_one();
}

uint32_t FusaCoeCaptureOp::receiver_start_byte()
{
    return hololink::operators::fusa_coe_capture::FusaCoeCaptureCore::receiver_start_byte();
}

uint32_t FusaCoeCaptureOp::received_line_bytes(uint32_t line_bytes)
{
    return hololink::operators::fusa_coe_capture::FusaCoeCaptureCore::received_line_bytes(
        line_bytes);
}

uint32_t FusaCoeCaptureOp::transmitted_line_bytes(
    hololink::module::csi::PixelFormat pixel_format, uint32_t pixel_width)
{
    return hololink::operators::fusa_coe_capture::FusaCoeCaptureCore::transmitted_line_bytes(
        static_cast<hololink::csi::PixelFormat>(pixel_format), pixel_width);
}

void FusaCoeCaptureOp::configure(
    uint32_t start_byte, uint32_t received_bytes_per_line,
    uint32_t pixel_width, uint32_t pixel_height,
    hololink::module::csi::PixelFormat pixel_format,
    uint32_t trailing_bytes)
{
    core_.configure(
        start_byte, received_bytes_per_line, pixel_width, pixel_height,
        static_cast<hololink::csi::PixelFormat>(pixel_format), trailing_bytes);
}

void FusaCoeCaptureOp::configure_converter(hololink::module::csi::CsiConverterV1& converter)
{
    // core_.pixel_format() returns a hololink::csi::PixelFormat; the module csi
    // enum carries identical integer values, so cast across the boundary.
    converter.configure(
        0, core_.bytes_per_line(), core_.pixel_width(), core_.pixel_height(),
        static_cast<hololink::module::csi::PixelFormat>(core_.pixel_format()), 0);
}

void FusaCoeCaptureOp::configure_frame_size(uint32_t frame_size_bytes)
{
    core_.configure_frame_size(frame_size_bytes);
}

nvidia::gxf::Expected<void> FusaCoeCaptureOp::buffer_release_callback(void* pointer)
{
    return hololink::operators::fusa_coe_capture::FusaCoeCaptureCore::buffer_release_callback(
        pointer);
}

} // namespace hololink::module::operators
