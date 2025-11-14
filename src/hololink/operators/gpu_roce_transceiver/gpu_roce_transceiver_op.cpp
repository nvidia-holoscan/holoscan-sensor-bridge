/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include "gpu_roce_transceiver_op.hpp"

#include <hololink/common/cuda_helper.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/logging_internal.hpp>

namespace hololink::operators {

void GpuRoceTransceiverOp::setup(holoscan::OperatorSpec& spec)
{
    HSB_LOG_INFO("Setting up DOCA-based RoCE receiver");

    BaseReceiverOp::setup(spec);

    // and add our own parameters
    spec.param(ibv_name_, "ibv_name", "IBVName", "IBV device to use",
        std::string("roceP5p3s0f0"));
    spec.param(ibv_port_, "ibv_port", "IBVPort", "Port number of IBV device", 1u);
    spec.param(gpu_id_, "gpu_id", "GPU Device ID", "CUDA device ID of the GPU", 0u);
}

void GpuRoceTransceiverOp::start()
{
    frame_ready_condition_->event_state(holoscan::AsynchronousEventState::EVENT_WAITING);
    BaseReceiverOp::start();
}

void GpuRoceTransceiverOp::stop()
{
    // Signal the async condition to EVENT_DONE to wake up the scheduler
    // This tells the scheduler the event has occurred and it can proceed
    if (frame_ready_condition_) {
        frame_ready_condition_->event_state(holoscan::AsynchronousEventState::EVENT_DONE);
    }
    // Now call base class stop which will call stop_receiver()
    BaseReceiverOp::stop();
}

void GpuRoceTransceiverOp::start_receiver()
{
    HSB_LOG_INFO("DOCA-based RoCE receiver is being implemented!");

    // constexpr uint32_t DOCA_PAGE_SIZE = 4096;
    size_t page_size = hololink::core::round_up(
        frame_size_.get(), hololink::core::PAGE_SIZE);
    // size_t buffer_size = hololink::core::round_up(
    //     page_size * PAGES, DOCA_PAGE_SIZE);
    // frame_memory_.reset(
    //     new ReceiverMemoryDescriptor(frame_context_, buffer_size));
    // HSB_LOG_INFO(
    //     "frame_size={:#x} frame={:#x} buffer_size={:#x} page_size={:#x}",
    //     frame_size_.get(), frame_memory_->get(), buffer_size, page_size);

    const std::string& peer_ip = hololink_channel_->peer_ip();
    HSB_LOG_INFO("ibv_name_={} ibv_port_={} peer_ip={} gpu_id_={}", ibv_name_.get(),
        ibv_port_.get(), peer_ip, gpu_id_);

    gpu_roce_transceiver_.reset(new GpuRoceTransceiver(
        ibv_name_.get().c_str(), ibv_port_.get(), gpu_id_.get(),
        frame_size_.get(), page_size, PAGES,
        peer_ip.c_str(), true, false, false));

    gpu_roce_transceiver_->set_frame_ready([this]() { this->frame_ready(); });

    if (!gpu_roce_transceiver_->start()) {
        throw std::runtime_error("Failed to start GpuRoceTransceiver");
    }

    hololink_channel_->authenticate(gpu_roce_transceiver_->get_qp_number(),
        gpu_roce_transceiver_->get_rkey());

    HSB_LOG_INFO(
        "HoloLink channel authenticated with DOCA-based QP number and RKEY");
    // we need to start the blocking_monitor thread but let's do it later

    gpu_monitor_thread_.reset(new std::thread(
        &hololink::operators::GpuRoceTransceiverOp::gpu_monitor_loop, this));

#if 0
      //Fails on IGX
      const int error =
        pthread_setname_np(gpu_monitor_thread_->native_handle(), name().c_str());
    if (error != 0) {
      throw std::runtime_error("Failed to set thread name");
    }
#endif

    HSB_LOG_INFO("DOCA Receiver thread started");

    auto [local_ip, local_port] = RoceReceiverOp::local_ip_and_port();
    HSB_LOG_INFO("local_ip={} local_port={}", local_ip, local_port);

    hololink_channel_->configure_roce(gpu_roce_transceiver_->external_frame_memory(),
        frame_size_, page_size, PAGES, local_port);
    HSB_LOG_INFO("HoloLink configured successfully");

    // throw std::runtime_error("DOCA is not supported from here");
}

void GpuRoceTransceiverOp::stop_receiver()
{
    HSB_LOG_INFO("Stopping DOCA-based RoCE receiver");

    hololink_channel_->unconfigure();
    data_socket_.reset();
    gpu_roce_transceiver_->close();

    // Detach the thread instead of joining to avoid hanging on GPU kernel cleanup
    // The thread will exit naturally after processing the control pipe signal
    if (gpu_monitor_thread_ && gpu_monitor_thread_->joinable()) {
        gpu_monitor_thread_->detach();
    }
    gpu_monitor_thread_.reset();
    // frame_memory_.reset();
}

void GpuRoceTransceiverOp::gpu_monitor_loop()
{
    HSB_LOG_INFO("Starting GPU monitoring loop for DOCA CQ");

    CudaCheck(cuCtxSetCurrent(frame_context_));
    CUcontext current_context;
    CudaCheck(cuCtxGetCurrent(&current_context));
    std::cout << "gpu_roce_transceiver context: " << current_context << std::endl;
    gpu_roce_transceiver_->blocking_monitor();

    HSB_LOG_INFO("GPU monitoring loop ended");
}

std::tuple<CUdeviceptr, std::shared_ptr<hololink::Metadata>>
GpuRoceTransceiverOp::get_next_frame(double timeout_ms)
{
    static CUdeviceptr frame_memory_location = 0;
    static std::shared_ptr<hololink::Metadata> frame_metadata_location;
    // HSB_LOG_INFO(fmt::format("DOCA start: frame_memory_location: 0x{:X}",
    //                          frame_memory_location));
    if (!frame_memory_location) {
        RoceReceiverMetadata roce_receiver_metadata;
        if (gpu_roce_transceiver_->get_next_frame(timeout_ms, roce_receiver_metadata)) {
            frame_memory_location = roce_receiver_metadata.frame_memory;
            auto metadata = std::make_shared<Metadata>();
            (*metadata)["frame_number"] = int64_t(roce_receiver_metadata.frame_number);
            (*metadata)["rx_write_requests"] = int64_t(roce_receiver_metadata.rx_write_requests);
            (*metadata)["received_s"] = int64_t(roce_receiver_metadata.received_s);
            (*metadata)["received_ns"] = int64_t(roce_receiver_metadata.received_ns);
            (*metadata)["imm_data"] = int64_t(roce_receiver_metadata.imm_data);
            CUdeviceptr frame_memory = roce_receiver_metadata.frame_memory;
            (*metadata)["frame_memory"] = int64_t(frame_memory);
            (*metadata)["dropped"] = int64_t(roce_receiver_metadata.dropped);
            // No embedded metadata in buffer; omit metadata_* and crc fields
            frame_metadata_location = metadata;
        }
    } else {
    }
    // HSB_LOG_INFO(fmt::format("DOCA end: frame_memory_location: 0x{:X}",
    //                          frame_memory_location));
    return { frame_memory_location, frame_metadata_location };
}

} // namespace hololink::operators
