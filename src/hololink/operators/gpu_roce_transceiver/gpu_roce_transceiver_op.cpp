/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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
 *
 * See README.md for detailed information.
 */

#include "gpu_roce_transceiver_op.hpp"

#include <future>
#include <netinet/in.h>
#include <unistd.h>

#include <hololink/common/cuda_helper.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/logging_internal.hpp>

namespace hololink::operators {

CUdeviceptr GpuRoceTransceiverOp::frame_memory()
{
    return (CUdeviceptr)(gpu_roce_transceiver_ ? gpu_roce_transceiver_->get_rx_ring_data_addr() : 0);
}

void GpuRoceTransceiverOp::setup(holoscan::OperatorSpec& spec)
{
    HSB_LOG_INFO("Setting up DOCA-based RoCE receiver");

    BaseReceiverOp::setup(spec);

    spec.param(ibv_name_, "ibv_name", "IBVName", "IBV device to use",
        std::string("roceP5p3s0f0"));
    spec.param(tx_ibv_qp_, "tx_ibv_qp", "Tx remote QP", "Remote QP number for Tx", 2u);
    spec.param(ibv_port_, "ibv_port", "IBVPort", "Port number of IBV device", 1u);
    spec.param(gpu_id_, "gpu_id", "GPU Device ID", "CUDA device ID of the GPU", 0u);
    spec.param(forward_, "forward", "Forward mode", "Enable forward mode", 0u);
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

    size_t page_size = hololink::core::round_up(
        frame_size_.get(), hololink::core::PAGE_SIZE);

    const std::string& peer_ip = hololink_channel_->peer_ip();
    HSB_LOG_INFO("ibv_name_={} ibv_port_={} peer_ip={} tx_ibv_qp_={} gpu_id_={} forward_={}", ibv_name_.get(),
        ibv_port_.get(), peer_ip, tx_ibv_qp_, gpu_id_, forward_);

    if (forward_ == 1)
        gpu_roce_transceiver_.reset(new GpuRoceTransceiver(
            ibv_name_.get().c_str(), ibv_port_.get(), tx_ibv_qp_.get(), gpu_id_.get(),
            frame_size_.get(), page_size, PAGES,
            peer_ip.c_str(), true, false, false));
    else
        gpu_roce_transceiver_.reset(new GpuRoceTransceiver(
            ibv_name_.get().c_str(), ibv_port_.get(), tx_ibv_qp_.get(), gpu_id_.get(),
            frame_size_.get(), page_size, PAGES,
            peer_ip.c_str(), false, true, true));

    gpu_roce_transceiver_->set_frame_ready([this]() { this->frame_ready(); });

    if (!gpu_roce_transceiver_->start()) {
        throw std::runtime_error("Failed to start GpuRoceTransceiver");
    }

    hololink_channel_->authenticate(gpu_roce_transceiver_->get_qp_number(),
        gpu_roce_transceiver_->get_rkey());

    HSB_LOG_INFO(
        "HoloLink channel authenticated with DOCA-based QP number and RKEY");

    gpu_monitor_thread_.reset(new std::thread(
        &hololink::operators::GpuRoceTransceiverOp::gpu_monitor_loop, this));

    HSB_LOG_INFO("DOCA Receiver thread started");

    auto [local_ip, local_port] = local_ip_and_port();
    HSB_LOG_INFO("local_ip={} local_port={}", local_ip, local_port);

    hololink_channel_->configure_roce(gpu_roce_transceiver_->external_frame_memory(),
        frame_size_, page_size, PAGES, local_port);
    HSB_LOG_INFO("HoloLink configured successfully");
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
        // Give the thread time to exit gracefully after close() signaled it
        auto future = std::async(std::launch::async, [this]() {
            gpu_monitor_thread_->join();
        });

        if (future.wait_for(std::chrono::seconds(2)) == std::future_status::timeout) {
            HSB_LOG_WARN("GPU monitor thread did not exit in time, detaching");
            gpu_monitor_thread_->detach();
        }
    }
    gpu_monitor_thread_.reset();
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

bool GpuRoceTransceiverOp::frames_ready()
{
    // Persistent kernel: no CPU frame stream; get_next_frame blocks until close.
    return false;
}

std::tuple<CUdeviceptr, std::shared_ptr<hololink::Metadata>> GpuRoceTransceiverOp::get_next_frame(
    double timeout_ms, CUstream cuda_stream)
{
    (void)timeout_ms;
    (void)cuda_stream;
    if (!gpu_roce_transceiver_) {
        return {};
    }
    // Blocks until close(); returns false → compute emits nothing (kernel owns datapath).
    gpu_roce_transceiver_->get_next_frame(0, cuda_stream);
    return {};
}

std::tuple<std::string, uint32_t> GpuRoceTransceiverOp::local_ip_and_port()
{
    sockaddr_in ip {};
    ip.sin_family = AF_UNSPEC;
    socklen_t ip_len = sizeof(ip);
    if (getsockname(data_socket_.get(), (sockaddr*)&ip, &ip_len) < 0) {
        throw std::runtime_error(
            fmt::format("getsockname failed with errno={}: \"{}\"", errno, strerror(errno)));
    }

    const std::string local_ip = inet_ntoa(ip.sin_addr);
    // Match RoceReceiverOp: advertise ROCE receiver port 4791
    const in_port_t local_port = 4791;
    return { local_ip, local_port };
}

uint8_t* GpuRoceTransceiverOp::get_ring_addr(bool is_rx)
{
    if (is_rx)
        return gpu_roce_transceiver_->get_rx_ring_data_addr();
    else
        return gpu_roce_transceiver_->get_tx_ring_data_addr();
}

size_t GpuRoceTransceiverOp::get_ring_stride_sz(bool is_rx)
{
    if (is_rx)
        return gpu_roce_transceiver_->get_rx_ring_stride_sz();
    else
        return gpu_roce_transceiver_->get_tx_ring_stride_sz();
}

uint32_t GpuRoceTransceiverOp::get_ring_stride_num(bool is_rx)
{
    if (is_rx)
        return gpu_roce_transceiver_->get_rx_ring_stride_num();
    else
        return gpu_roce_transceiver_->get_tx_ring_stride_num();
}

uint64_t* GpuRoceTransceiverOp::get_ring_flag_addr(bool is_rx)
{
    if (is_rx)
        return gpu_roce_transceiver_->get_rx_ring_flag_addr();
    else
        return gpu_roce_transceiver_->get_tx_ring_flag_addr();
}

} // namespace hololink::operators
