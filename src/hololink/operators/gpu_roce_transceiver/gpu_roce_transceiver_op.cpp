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

#include <any>
#include <future>
#include <mutex>
#include <netinet/in.h>
#include <typeindex>
#include <unistd.h>

#include <holoscan/core/argument_setter.hpp>

#include <hololink/common/cuda_helper.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/logging_internal.hpp>

namespace hololink::operators {

namespace {

    /**
     * Holoscan's default argument setter rejects a vector of a custom type, like the DataChannel
     * pointers of hololink_channels: register one that takes the vector as is. It must be in place
     * before OperatorSpec::param() registers the default setter for the same type.
     */
    void register_data_channels_setter()
    {
        static std::once_flag once;
        std::call_once(once, []() {
            holoscan::ArgumentSetter::get_instance().add_argument_setter(
                std::type_index(typeid(std::vector<DataChannel*>)),
                [](holoscan::ParameterWrapper& param_wrap, holoscan::Arg& arg) -> bool {
                    auto& param = *std::any_cast<holoscan::Parameter<std::vector<DataChannel*>>*>(param_wrap.value());
                    // No name and no value: use the default value
                    if (arg.name().empty() && !arg.has_value()) {
                        param.set_default_value();
                        return true;
                    }
                    try {
                        param = std::any_cast<std::vector<DataChannel*>>(arg.value());
                    } catch (const std::bad_any_cast&) {
                        HSB_LOG_ERROR("Argument '{}' must be a std::vector<hololink::DataChannel*>", arg.name());
                        return false;
                    }
                    return true;
                });
        });
    }

} // namespace

CUdeviceptr GpuRoceTransceiverOp::frame_memory(unsigned sensor_id)
{
    return (CUdeviceptr)(gpu_roce_transceiver_ ? gpu_roce_transceiver_->get_rx_ring_data_addr(sensor_id) : 0);
}

void GpuRoceTransceiverOp::setup(holoscan::OperatorSpec& spec)
{
    HSB_LOG_INFO("Setting up DOCA-based RoCE receiver");

    BaseReceiverOp::setup(spec);

    register_data_channels_setter();
    spec.param(hololink_channels_, "hololink_channels", "HololinkChannels",
        "Hololink DataChannels, one per QP of tx_ibv_qp in the same order, all on the same board. "
        "Empty: hololink_channel is the only one",
        std::vector<DataChannel*> {});
    spec.param(ibv_name_, "ibv_name", "IBVName", "IBV device to use",
        std::string("roceP5p3s0f0"));
    spec.param(tx_ibv_qp_, "tx_ibv_qp", "Tx remote QPs",
        "Comma-separated remote QP numbers, one per Hololink DataChannel (e.g. \"2,4,6,8\")", std::string("2"));
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

std::vector<DataChannel*> GpuRoceTransceiverOp::data_channels()
{
    std::vector<DataChannel*> channels = hololink_channels_.get();
    if (channels.empty())
        channels.push_back(hololink_channel_.get());

    for (size_t x = 0; x < channels.size(); x++) {
        if (channels[x] == nullptr)
            throw std::runtime_error(fmt::format("hololink_channels[{}] is null", x));
    }

    // One local QP per channel, all connected to the same board
    const std::string& peer_ip = channels[0]->peer_ip();
    for (size_t x = 1; x < channels.size(); x++) {
        if (channels[x]->peer_ip() != peer_ip)
            throw std::runtime_error(fmt::format(
                "hololink_channels[{}] peer_ip={} differs from hololink_channel peer_ip={}: all the channels must "
                "belong to the same board",
                x, channels[x]->peer_ip(), peer_ip));
    }
    return channels;
}

void GpuRoceTransceiverOp::start_receiver()
{
    HSB_LOG_INFO("DOCA-based RoCE receiver is being implemented!");

    size_t page_size = hololink::core::round_up(
        frame_size_.get(), hololink::core::PAGE_SIZE);

    const std::vector<DataChannel*> channels = data_channels();
    const std::string& peer_ip = channels[0]->peer_ip();
    HSB_LOG_INFO("ibv_name_={} ibv_port_={} peer_ip={} tx_ibv_qp_={} channels={} gpu_id_={} forward_={}",
        ibv_name_.get(), ibv_port_.get(), peer_ip, tx_ibv_qp_.get(), channels.size(), gpu_id_, forward_);

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

    // QP x carries the traffic of channels[x]
    if (gpu_roce_transceiver_->get_num_qps() != channels.size()) {
        throw std::runtime_error(fmt::format(
            "tx_ibv_qp=\"{}\" lists {} QPs for {} Hololink DataChannels: there must be one QP per channel",
            tx_ibv_qp_.get(), gpu_roce_transceiver_->get_num_qps(), channels.size()));
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
            &hololink::operators::GpuRoceTransceiverOp::gpu_monitor_loop, this));

        HSB_LOG_INFO("DOCA Receiver thread started");

        auto [local_ip, local_port] = local_ip_and_port();
        HSB_LOG_INFO("local_ip={} local_port={}", local_ip, local_port);

        for (auto* channel : channels) {
            // Recorded first so that a partially written configuration is also undone on stop
            configured_channels_.push_back(channel);
            channel->configure_roce(gpu_roce_transceiver_->external_frame_memory(),
                frame_size_, page_size, PAGES, local_port);
        }
        HSB_LOG_INFO("HoloLink configured successfully");
    } catch (...) {
        // Don't leave the monitor thread and the GPU kernels running, nor HSB sending, after a failed start
        stop_receiver();
        throw;
    }
}

void GpuRoceTransceiverOp::stop_receiver()
{
    HSB_LOG_INFO("Stopping DOCA-based RoCE receiver");

    // Stop HSB sending to every channel first, even if one of them fails
    for (auto* channel : configured_channels_) {
        try {
            channel->unconfigure();
        } catch (const std::exception& e) {
            HSB_LOG_ERROR("Failed to unconfigure a Hololink channel: {}", e.what());
        }
    }
    configured_channels_.clear();

    data_socket_.reset();
    if (gpu_roce_transceiver_)
        gpu_roce_transceiver_->close();

    // close() wakes up the monitor thread through the control pipe, and tells the GPU kernels to exit:
    // they check the exit flag also while waiting for packets.
    if (gpu_monitor_thread_ && gpu_monitor_thread_->joinable())
        gpu_monitor_thread_->join();
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

uint8_t* GpuRoceTransceiverOp::get_ring_addr(bool is_rx, unsigned sensor_id)
{
    if (is_rx)
        return gpu_roce_transceiver_->get_rx_ring_data_addr(sensor_id);
    else
        return gpu_roce_transceiver_->get_tx_ring_data_addr(sensor_id);
}

size_t GpuRoceTransceiverOp::get_ring_stride_sz(bool is_rx, unsigned sensor_id)
{
    if (is_rx)
        return gpu_roce_transceiver_->get_rx_ring_stride_sz(sensor_id);
    else
        return gpu_roce_transceiver_->get_tx_ring_stride_sz(sensor_id);
}

uint32_t GpuRoceTransceiverOp::get_ring_stride_num(bool is_rx, unsigned sensor_id)
{
    if (is_rx)
        return gpu_roce_transceiver_->get_rx_ring_stride_num(sensor_id);
    else
        return gpu_roce_transceiver_->get_tx_ring_stride_num(sensor_id);
}

uint64_t* GpuRoceTransceiverOp::get_ring_flag_addr(bool is_rx, unsigned sensor_id)
{
    if (is_rx)
        return gpu_roce_transceiver_->get_rx_ring_flag_addr(sensor_id);
    else
        return gpu_roce_transceiver_->get_tx_ring_flag_addr(sensor_id);
}

} // namespace hololink::operators
