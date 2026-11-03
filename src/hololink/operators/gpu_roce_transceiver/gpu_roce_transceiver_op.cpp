/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 *
 * See README.md for detailed information.
 */

#include "gpu_roce_transceiver_op.hpp"
#include "../base_receiver_op.inl"

// Full GpuRoceTransceiver definition — needed here but kept out of the header
// to avoid exposing DOCA/ibverbs headers to consumers of the operator.
#include "gpu_roce_transceiver.hpp"

#include <exception>
#include <future>
#include <netinet/in.h>
#include <unistd.h>

#include <hololink/common/cuda_helper.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/logging_internal.hpp>

namespace hololink::operators {

CUdeviceptr GpuRoceTransceiverOp::frame_memory(unsigned sensor_id)
{
    return (CUdeviceptr)(gpu_roce_transceiver_
            ? gpu_roce_transceiver_->get_rx_ring_data_addr(sensor_id)
            : 0);
}

std::vector<std::shared_ptr<DataChannel>> GpuRoceTransceiverOp::data_channels()
{
    std::vector<std::shared_ptr<DataChannel>> channels = this->params().hololink_channels;
    if (channels.empty()) {
        channels.push_back(hololink_channel_);
    }

    for (size_t x = 0; x < channels.size(); x++) {
        if (!channels[x]) {
            throw std::runtime_error(fmt::format("hololink_channels[{}] is null", x));
        }
    }

    // One local QP per channel, all connected to the same board
    const std::string& peer_ip = channels[0]->peer_ip();
    for (size_t x = 1; x < channels.size(); x++) {
        if (channels[x]->peer_ip() != peer_ip) {
            throw std::runtime_error(fmt::format(
                "hololink_channels[{}] peer_ip={} differs from hololink_channels[0] peer_ip={}: all "
                "the channels must belong to the same board",
                x, channels[x]->peer_ip(), peer_ip));
        }
    }
    return channels;
}

void GpuRoceTransceiverOp::start_receiver()
{
    const auto& p = this->params();
    HSB_LOG_INFO("Starting DOCA-based RoCE receiver");

    size_t page_size = hololink::core::round_up(p.frame_size, hololink::core::PAGE_SIZE);

    const std::vector<std::shared_ptr<DataChannel>> channels = data_channels();
    const std::string& peer_ip = channels[0]->peer_ip();
    // Without an ibv_name, the device that reaches the board.
    const auto [ibv_name, ibv_port] = p.ibv_name.empty()
        ? channels[0]->infiniband_device()
        : std::make_pair(p.ibv_name, p.ibv_port);
    HSB_LOG_INFO("ibv_name={} ibv_port={} peer_ip={} tx_ibv_qp={} channels={} gpu_id={} forward={}",
        ibv_name, ibv_port, peer_ip, p.tx_ibv_qp, channels.size(), p.gpu_id, p.forward);

    if (p.forward == 1) {
        gpu_roce_transceiver_.reset(new GpuRoceTransceiver(
            ibv_name.c_str(), ibv_port, p.tx_ibv_qp, p.gpu_id,
            p.frame_size, page_size, PAGES,
            peer_ip.c_str(), true, false, false));
    } else {
        gpu_roce_transceiver_.reset(new GpuRoceTransceiver(
            ibv_name.c_str(), ibv_port, p.tx_ibv_qp, p.gpu_id,
            p.frame_size, page_size, PAGES,
            peer_ip.c_str(), false, true, true));
    }

    // QP x carries the traffic of channels[x]
    if (gpu_roce_transceiver_->get_num_qps() != channels.size()) {
        throw std::runtime_error(fmt::format(
            "tx_ibv_qp=\"{}\" lists {} QPs for {} Hololink DataChannels: there must be one QP per "
            "channel",
            p.tx_ibv_qp, gpu_roce_transceiver_->get_num_qps(), channels.size()));
    }

    gpu_roce_transceiver_->set_frame_ready([this]() { this->frame_ready(); });

    if (!gpu_roce_transceiver_->start()) {
        throw std::runtime_error("Failed to start GpuRoceTransceiver");
    }

    try {
        for (size_t x = 0; x < channels.size(); x++) {
            channels[x]->authenticate(gpu_roce_transceiver_->get_qp_number(x),
                gpu_roce_transceiver_->get_rkey(x));
            HSB_LOG_INFO("HoloLink channel {} authenticated with DOCA-based QP {:#x} and RKEY", x,
                gpu_roce_transceiver_->get_qp_number(x));
        }

        gpu_monitor_thread_.reset(new std::thread(
            &GpuRoceTransceiverOp::gpu_monitor_loop, this));

        HSB_LOG_INFO("DOCA Receiver thread started");

        auto [local_ip, local_port] = local_ip_and_port();
        HSB_LOG_INFO("local_ip={} local_port={}", local_ip, local_port);

        for (const auto& channel : channels) {
            // Recorded first so that a partially written configuration is also undone on stop
            configured_channels_.push_back(channel);
            channel->configure_roce(gpu_roce_transceiver_->external_frame_memory(),
                p.frame_size, page_size, PAGES, local_port);
        }
        HSB_LOG_INFO("HoloLink configured successfully");
    } catch (...) {
        // Don't leave the monitor thread and the GPU kernels running, nor HSB sending, after a
        // failed start (BaseReceiverOp::start() expects this to clean up after itself). The
        // failure that got here is the one reported.
        const bool kernels_may_run = static_cast<bool>(gpu_monitor_thread_);
        try {
            stop_receiver();
        } catch (const std::exception& e) {
            HSB_LOG_ERROR("Stopping the receiver after a failed start failed: {}", e.what());
        }
        // Also free the DOCA resources and, on an iGPU, the CPU proxy thread, which runs until the
        // transceiver is destroyed. Not if forward-mode kernels may be running: they don't exit on
        // close(), so the destructor would wait for them forever.
        if (!p.forward || !kernels_may_run) {
            gpu_roce_transceiver_.reset();
        }
        throw;
    }
}

void GpuRoceTransceiverOp::stop_receiver()
{
    HSB_LOG_INFO("Stopping DOCA-based RoCE receiver");

    // Stop HSB sending to every channel first, even if one of them fails; tear everything down
    // before reporting the first failure (destroying a joinable thread terminates the process).
    std::exception_ptr unconfigure_error;
    for (const auto& channel : configured_channels_) {
        try {
            channel->unconfigure();
        } catch (...) {
            if (!unconfigure_error) {
                unconfigure_error = std::current_exception();
            }
        }
    }
    configured_channels_.clear();

    data_socket_.reset();
    // Guard against being called after a start_receiver() that failed before the transceiver
    // existed.
    if (gpu_roce_transceiver_) {
        gpu_roce_transceiver_->close();
    }

    // close() wakes up the monitor thread through the control pipe and sets the kernels' exit
    // flag. The receive-only and transmit-only kernels check it also while waiting for packets;
    // the forward-mode kernels only between packets, so they keep waiting once HSB stops sending.
    if (gpu_monitor_thread_ && gpu_monitor_thread_->joinable()) {
        gpu_monitor_thread_->join();
    }
    gpu_monitor_thread_.reset();
    if (unconfigure_error) {
        std::rethrow_exception(unconfigure_error);
    }
}

void GpuRoceTransceiverOp::gpu_monitor_loop()
{
    HSB_LOG_INFO("Starting GPU monitoring loop for DOCA CQ");

    CudaCheck(cuCtxSetCurrent(this->params().frame_context));
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
    // GPU kernel owns the data path; always returns {} so compute() emits nothing.
    gpu_roce_transceiver_->get_next_frame(0, cuda_stream);
    return {};
}

std::tuple<std::string, uint32_t> GpuRoceTransceiverOp::local_ip_and_port()
{
    sockaddr_in ip {};
    ip.sin_family = AF_UNSPEC;
    socklen_t ip_len = sizeof(ip);
    if (getsockname(data_socket_.get(), reinterpret_cast<sockaddr*>(&ip), &ip_len) < 0) {
        throw std::runtime_error(
            fmt::format("getsockname failed with errno={}: \"{}\"", errno, strerror(errno)));
    }

    const std::string local_ip = inet_ntoa(ip.sin_addr);
    // Match RoceReceiverOp: advertise ROCE receiver port 4791
    const in_port_t local_port = 4791;
    return { local_ip, local_port };
}

uint8_t* GpuRoceTransceiverOp::get_ring_addr(bool is_rx, unsigned sensor_id)
{
    return is_rx ? gpu_roce_transceiver_->get_rx_ring_data_addr(sensor_id)
                 : gpu_roce_transceiver_->get_tx_ring_data_addr(sensor_id);
}

size_t GpuRoceTransceiverOp::get_ring_stride_sz(bool is_rx, unsigned sensor_id)
{
    return is_rx ? gpu_roce_transceiver_->get_rx_ring_stride_sz(sensor_id)
                 : gpu_roce_transceiver_->get_tx_ring_stride_sz(sensor_id);
}

uint32_t GpuRoceTransceiverOp::get_ring_stride_num(bool is_rx, unsigned sensor_id)
{
    return is_rx ? gpu_roce_transceiver_->get_rx_ring_stride_num(sensor_id)
                 : gpu_roce_transceiver_->get_tx_ring_stride_num(sensor_id);
}

uint64_t* GpuRoceTransceiverOp::get_ring_flag_addr(bool is_rx, unsigned sensor_id)
{
    return is_rx ? gpu_roce_transceiver_->get_rx_ring_flag_addr(sensor_id)
                 : gpu_roce_transceiver_->get_tx_ring_flag_addr(sensor_id);
}

template class BaseReceiverOp<GpuRoceTransceiverOp, GpuRoceTransceiverParams>;

} // namespace hololink::operators
