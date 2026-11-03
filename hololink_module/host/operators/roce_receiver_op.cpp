/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hololink/module/operators/roce_receiver_op.hpp"
#include "hololink/module/operators/base_receiver_op.inl"

#include <unistd.h> // getpagesize

#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <cuda.h>

#include <fmt/format.h>

#include "hololink/module/adapter.hpp" // Adapter::get_adapter / get_module
#include "hololink/module/ibv_device.hpp" // ibv_device_for_peer
#include "hololink/module/logging.hpp" // HSB_LOG_INFO
#include "hololink/module/module.hpp"
#include "hololink/module/page_size.hpp" // PAGE_SIZE, round_up
#include "hololink/module/receiver_memory_descriptor.hpp"
#include "hololink/module/status.h"

namespace hololink::module::operators {

void RoceReceiverOp::start_receiver()
{
    const auto& p = this->params();

    if (p.queue_size == 0) {
        throw std::runtime_error("Queue size cannot be 0");
    }
    if (p.queue_size > p.pages) {
        throw std::runtime_error(fmt::format("Queue size {} cannot be greater than the number of pages {}", p.queue_size, p.pages));
    }
    frame_number_ = {};

    // Resolve the supplement module that handled this metadata's
    // enumeration, then fetch the V1 services it published:
    //   - RoceDataChannelInterfaceV1 (per (serial, data_plane))
    //   - FrameMetadataInterfaceV1 (per-module singleton)
    //   - RoceReceiverInterfaceV1 (per (serial, data_plane))
    // All three sit behind the same module Publisher; on hsb_lite_2510
    // boards the channel + receiver are subclass overrides, picked up
    // automatically by the locator.
    auto& adapter = hololink::module::Adapter::get_adapter();
    auto module = adapter.get_module(metadata_);
    // Metadata-form get_service: caches the per-(serial, data_plane)
    // RoceDataChannelImpl and runs its configure(metadata) so the backing
    // DataChannel is materialized against the per-board LegacyHololinkAccess.
    channel_ = hololink::module::RoceDataChannelInterfaceV1::get_service(metadata_);
    frame_metadata_ = hololink::module::FrameMetadataInterfaceV1::get_service(module);
    host_metadata_.assign(frame_metadata_->block_size(), 0);

    size_t metadata_address = hololink::module::round_up(
        p.frame_size, hololink::module::PAGE_SIZE);
    size_t metadata_offset = p.metadata_offset;
    if (metadata_offset == 0) {
        metadata_offset = metadata_address;
    }
    if (metadata_offset > metadata_address) {
        // This only occurs if the user passed in a metadata_offset that is
        // past the end of the buffer we'll allocate.
        throw std::runtime_error(fmt::format("metadata_offset={:#x} is beyond the receiver buffer limit={:#x}.", metadata_offset, metadata_address));
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
        received_frame_size * p.pages, getpagesize());
    frame_buffer_ = std::make_shared<hololink::module::ReceiverMemoryDescriptor>(
        p.frame_context, buffer_size);
    HSB_LOG_INFO("frame_size={:#x} frame={:#x} buffer_size={:#x}", p.frame_size, frame_buffer_->get(), buffer_size);

    size_t page_size = p.page_size;
    if (!page_size) {
        // Per-frame stride the FPGA cycles through:
        // frame_memory[page] = cu_buffer + page_size * page. Setting this to
        // buffer_size would land every page > 0 frame outside the registered
        // MR, so the received frame size is used.
        page_size = received_frame_size;
    }

    // V2 also reports a failed receive path; with a module that publishes only V1 the run
    // goes on without that report.
    receiver_v2_ = hololink::module::RoceReceiverInterfaceV2::get_service(metadata_, /*allow_null=*/true);
    receiver_ = receiver_v2_;
    if (!receiver_) {
        receiver_ = hololink::module::RoceReceiverInterfaceV1::get_service(metadata_);
    }

    // peer_ip travels in the enumeration metadata; ibv_name / ibv_port
    // are derived from it via ibv_device_for_peer (deterministic for a
    // given (peer_ip, host networking) pair).
    const std::string peer_ip = metadata_.get<std::string>("peer_ip");
    // Debug diagnostic: the peer_ip this receiver pulls from its enumeration
    // metadata, plus the data_channel that identifies the leg. If this peer
    // differs run-to-run for the same leg (e.g. .2 vs .3), the variable is the
    // metadata fed in, not local_ip_and_mac.
    HSB_LOG_DEBUG("RoceReceiverOp::start_receiver peer_ip(from metadata)={} data_channel={} serial={}",
        peer_ip,
        metadata_.get<int64_t>("data_channel", int64_t { -1 }),
        metadata_.get<std::string>("serial_number", std::string {}));
    const auto [ibv_name, ibv_port] = hololink::module::ibv_device_for_peer(
        peer_ip, metadata_.get<std::string>("interface", std::string {}));

    const hololink_module_status_t start_status = receiver_->start(
        ibv_name,
        ibv_port,
        static_cast<uint64_t>(frame_buffer_->get()),
        buffer_size,
        p.frame_size,
        page_size,
        p.pages,
        metadata_offset,
        peer_ip,
        p.queue_size);
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

    // Register the frame-ready callback before the monitor thread starts
    // so no frame slips through before the condition is armed.
    receiver_->set_frame_ready([this]() { this->frame_ready(); });

    monitor_thread_ = std::make_shared<std::thread>(
        [self = receiver_, frame_context = p.frame_context]() {
            HOLOLINK_MODULE_CUDA_CHECK(cuCtxSetCurrent(frame_context));
            self->blocking_monitor();
        });
}

void RoceReceiverOp::stop_receiver()
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
            self_.receiver_v2_.reset();
            self_.channel_.reset();
            self_.frame_metadata_.reset();
            self_.frame_buffer_.reset();
        }

    private:
        RoceReceiverOp& self_;
    };
    Cleanup cleanup(*this);

    if (configured_ && channel_) {
        const hololink_module_status_t status = channel_->detach_receiver();
        if (status != HOLOLINK_MODULE_OK) {
            throw std::runtime_error(std::string(
                                         "While stopping RoceReceiverOp: channel->detach_receiver returned status ")
                + std::to_string(status));
        }
    }
}

bool RoceReceiverOp::frames_ready()
{
    return receiver_->frames_ready();
}

bool RoceReceiverOp::receive_path_failed()
{
    return receiver_v2_ && receiver_v2_->receive_path_failed();
}

std::tuple<CUdeviceptr, std::shared_ptr<hololink::Metadata>>
RoceReceiverOp::get_next_frame(double timeout_ms, CUstream /*cuda_stream*/)
{
    hololink::module::RoceReceiverFrameInfoV1 frame_info {};
    if (!receiver_->get_next_frame(static_cast<unsigned>(timeout_ms), frame_info)) {
        return {};
    }

    // Pull the EOF block from device memory and let the V1
    // FrameMetadataInterfaceV1::decode own the layout. The backing
    // receiver also decodes internally, but routing through the V1
    // contract keeps modules in charge of their own metadata format
    // and its size. host_metadata_ was sized in start_receiver() from
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

    // v1_metadata.flags has no corresponding CaptureMetadataT schema field
    // (see src/hololink/operators/capture_metadata.fbs) -- dropped here,
    // matching native RoceReceiverOp's identical omission.
    auto metadata = std::make_shared<hololink::Metadata>();
    (*metadata)["psn"] = int64_t(v1_metadata.psn);
    (*metadata)["crc"] = int64_t(v1_metadata.crc);
    // The block's frame_number is 16 bits; report it extended to 32, as the other receivers do.
    (*metadata)["frame_number"] = int64_t(frame_number_.update(v1_metadata.frame_number));
    (*metadata)["timestamp_s"] = int64_t(v1_metadata.timestamp_s);
    (*metadata)["timestamp_ns"] = int64_t(v1_metadata.timestamp_ns);
    (*metadata)["bytes_written"] = int64_t(v1_metadata.bytes_written);
    (*metadata)["metadata_s"] = int64_t(v1_metadata.metadata_s);
    (*metadata)["metadata_ns"] = int64_t(v1_metadata.metadata_ns);
    // received_s/ns is the host-side reception time (when the last frame
    // data landed); the RoCE receiver records it on frame_info rather than
    // in the decoded EOF block. Emit it so timestamp diagnostics can
    // compute FPGA-to-host latency, matching LinuxReceiverOp.
    (*metadata)["received_s"] = int64_t(frame_info.received_s);
    (*metadata)["received_ns"] = int64_t(frame_info.received_ns);
    // imm_data isn't carried in the decoded EOF block (FrameMetadata),
    // but the RoCE receiver surfaces it on frame_info; emit it (and the
    // low-12-bit page_number) so the RoCE path matches LinuxReceiverOp.
    (*metadata)["imm_data"] = int64_t(frame_info.imm_data);
    (*metadata)["page_number"] = int64_t(frame_info.imm_data & 0xFFF);
    // The receiver's own frame count and the frames it dropped (ring overflow), as the
    // native RoceReceiverOp reports them.
    (*metadata)["received_frame_number"] = int64_t(frame_info.received_frame_number);
    (*metadata)["dropped"] = int64_t(frame_info.dropped);

    return { static_cast<CUdeviceptr>(frame_info.frame_memory), metadata };
}

template class BaseReceiverOp<RoceReceiverOp, RoceReceiverParams>;

} // namespace hololink::module::operators
