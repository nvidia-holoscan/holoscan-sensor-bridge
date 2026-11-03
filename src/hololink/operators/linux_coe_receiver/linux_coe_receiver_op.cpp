/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 */

#include "linux_coe_receiver_op.hpp"
#include "../base_receiver_op.inl"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <sched.h>
#include <sys/socket.h>
#include <unistd.h>

#include <linux/if_packet.h>
#include <net/if.h>

#include <hololink/common/cuda_helper.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/logging_internal.hpp>
#include <hololink/core/networking.hpp>

#include "linux_coe_receiver.hpp"

namespace hololink::operators {

namespace {
    constexpr int DEFAULT_RECEIVER_AFFINITY = 2;
    // IEEE 1722/AVTP ethertype. No standard <linux/if_ether.h> macro for this one (unlike
    // ETH_P_IP etc.) -- matches the old HSDK4-era Python LinuxCoeReceiverOp's own hardcoded
    // ETH_P_AVTP constant.
    constexpr uint16_t ETH_P_AVTP = 0x22F0;
    // Every COE socket on an interface receives the AVTP traffic of all sensors on that
    // interface and drops the other channels in user space. The kernel default receive
    // buffer (net.core.rmem_default, ~208 KB) holds well under a millisecond of a 3 Gb/s
    // stereo stream, so short receiver-thread stalls drop packets silently. This matches the
    // net.core.rmem_max value in docs/user_guide/setup.mdx, so SO_RCVBUF grants it in full on a
    // host set up per the docs.
    constexpr int COE_SOCKET_BUFFER_SIZE = 31326208;
} // unnamed namespace

LinuxCoeReceiverOp::~LinuxCoeReceiverOp()
{
    // stop() has already joined the stall monitor; this covers an operator destroyed without
    // one.
    stop_stall_monitor();
}

void LinuxCoeReceiverOp::start()
{
    BaseReceiverOp<LinuxCoeReceiverOp, LinuxCoeReceiverParams>::start();
    // The stall timeout counts from here, after the device started, as it did in HSDK4: no
    // frame comes before. A start() that fails gets no stop(), so undo the start when the
    // monitor can't run.
    try {
        start_stall_monitor();
    } catch (...) {
        try {
            BaseReceiverOp<LinuxCoeReceiverOp, LinuxCoeReceiverParams>::stop();
        } catch (const std::exception& e) {
            HSB_LOG_ERROR("Stopping the receiver after the stall monitor failed to start failed: {}", e.what());
        }
        throw;
    }
}

void LinuxCoeReceiverOp::stop()
{
    stop_stall_monitor();
    BaseReceiverOp<LinuxCoeReceiverOp, LinuxCoeReceiverParams>::stop();
}

holoscan::expected<void, holoscan::Error> LinuxCoeReceiverOp::compute(
    holoscan::ExecutionContext& context)
{
    compute_calls_ += 1;
    // watch_for_stall() logged the stall at ERROR and woke compute() to end the run. Fail it, as
    // FusaCoeCaptureOp does: the run ends kFailed and its diagnostic records the stall.
    if (stalled_.load()) {
        return holoscan::make_unexpected(stall_error());
    }
    auto result = BaseReceiverOp<LinuxCoeReceiverOp, LinuxCoeReceiverParams>::compute(context);
    if (!result) {
        return result;
    }
    // This activation may have been waiting for a frame when the stall was declared.
    if (stalled_.load()) {
        return holoscan::make_unexpected(stall_error());
    }
    // At the frame limit, this operator has completed and no compute() follows: a frame that
    // doesn't come now is no stall. The limit counts compute() calls, including the ones that
    // timed out without a frame, so frame_count_ can still be short of it.
    const int64_t frame_limit = this->params().frame_limit;
    if (frame_limit > 0 && compute_calls_ >= static_cast<uint64_t>(frame_limit)) {
        {
            std::lock_guard<std::mutex> lock(stall_mutex_);
            stall_monitor_running_ = false;
        }
        stall_cv_.notify_all();
    }
    return result;
}

void LinuxCoeReceiverOp::timeout(holoscan::ExecutionContext& context)
{
    if (!stalled_.load()) {
        BaseReceiverOp<LinuxCoeReceiverOp, LinuxCoeReceiverParams>::timeout(context);
    }
}

holoscan::Error LinuxCoeReceiverOp::stall_error() const
{
    const auto& p = this->params();
    return holoscan::Error {
        holoscan::ErrorCode::kTimeout,
        fmt::format("CoE capture on \"{}\" channel {} stalled for {}ms without a frame",
            p.coe_interface, p.coe_channel, stall_duration_ms_.load())
    };
}

void LinuxCoeReceiverOp::start_stall_monitor()
{
    // A reused operator starts without a stall.
    stalled_ = false;
    stall_duration_ms_ = 0;
    compute_calls_ = 0;
    const std::chrono::milliseconds stall_timeout { this->params().stall_timeout_ms };
    if (stall_timeout.count() == 0) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(stall_mutex_);
        stall_timeout_ = stall_timeout;
        last_frame_time_ = std::chrono::steady_clock::now();
        frames_received_ = 0;
        stall_monitor_running_ = true;
    }
    try {
        stall_monitor_thread_ = std::thread(&LinuxCoeReceiverOp::watch_for_stall, this);
    } catch (...) {
        std::lock_guard<std::mutex> lock(stall_mutex_);
        stall_monitor_running_ = false;
        throw;
    }
}

void LinuxCoeReceiverOp::stop_stall_monitor()
{
    {
        std::lock_guard<std::mutex> lock(stall_mutex_);
        stall_monitor_running_ = false;
    }
    stall_cv_.notify_all();
    if (stall_monitor_thread_.joinable()) {
        stall_monitor_thread_.join();
    }
}

void LinuxCoeReceiverOp::frame_arrived()
{
    std::lock_guard<std::mutex> lock(stall_mutex_);
    if (stall_monitor_running_) {
        last_frame_time_ = std::chrono::steady_clock::now();
        frames_received_ += 1;
    }
}

void LinuxCoeReceiverOp::watch_for_stall()
{
    // pthread_setname_np silently truncates names longer than 15 characters.
    pthread_setname_np(pthread_self(), "coe_stall_watch");
    std::unique_lock<std::mutex> lock(stall_mutex_);
    while (stall_monitor_running_) {
        // Every frame moves the deadline. Wake up at it, at stop_stall_monitor(), or spuriously.
        const auto deadline = last_frame_time_ + stall_timeout_;
        if (std::chrono::steady_clock::now() < deadline) {
            stall_cv_.wait_until(lock, deadline);
            continue;
        }
        const auto stalled_for = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - last_frame_time_);
        const uint64_t frames = frames_received_;
        stall_monitor_running_ = false;
        stall_duration_ms_ = stalled_for.count();
        stalled_ = true;
        lock.unlock();
        HSB_LOG_ERROR("CoE capture stalled: failing after {}ms without a frame ({} received frames)",
            stalled_for.count(), frames);
        // Only compute() can end the run.
        frame_ready();
        return;
    }
}

void LinuxCoeReceiverOp::set_receiver_affinity(const std::vector<int>& affinity)
{
    affinity_override_ = affinity;
}

void LinuxCoeReceiverOp::create_data_socket()
{
    const auto& p = this->params();

    data_socket_.reset(socket(AF_PACKET, SOCK_RAW, htons(ETH_P_AVTP)));
    if (!data_socket_) {
        throw std::runtime_error(
            fmt::format("Failed to create COE socket, errno={}: \"{}\"", errno, strerror(errno)));
    }

    const unsigned int ifindex = if_nametoindex(p.coe_interface.c_str());
    if (ifindex == 0) {
        throw std::runtime_error(fmt::format(
            "Unknown network interface \"{}\", errno={}: \"{}\"", p.coe_interface, errno, strerror(errno)));
    }

    sockaddr_ll address {};
    address.sll_family = AF_PACKET;
    address.sll_protocol = htons(ETH_P_AVTP);
    address.sll_ifindex = static_cast<int>(ifindex);

    if (bind(data_socket_.get(), reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) {
        throw std::runtime_error(fmt::format(
            "Failed to bind COE socket to \"{}\", errno={}: \"{}\"", p.coe_interface, errno, strerror(errno)));
    }

    // SO_RCVBUFFORCE ignores net.core.rmem_max but needs CAP_NET_ADMIN; without it,
    // fall back to SO_RCVBUF, which the kernel caps at net.core.rmem_max.
    const int request_size = COE_SOCKET_BUFFER_SIZE;
    if (setsockopt(data_socket_.get(), SOL_SOCKET, SO_RCVBUFFORCE, &request_size, sizeof(request_size)) < 0
        && setsockopt(data_socket_.get(), SOL_SOCKET, SO_RCVBUF, &request_size, sizeof(request_size)) < 0) {
        HSB_LOG_WARN("Failed to set COE socket buffer size, errno={}: \"{}\"", errno, strerror(errno));
    }

    int receiver_buffer_size = 0;
    socklen_t opt_len = sizeof(receiver_buffer_size);
    if (getsockopt(data_socket_.get(), SOL_SOCKET, SO_RCVBUF, &receiver_buffer_size, &opt_len) < 0) {
        HSB_LOG_WARN("Failed to get COE socket buffer size, errno={}: \"{}\"", errno, strerror(errno));
    }

    HSB_LOG_DEBUG("COE receiver buffer size={}", receiver_buffer_size);
    // The kernel reports twice the usable size it granted.
    if (receiver_buffer_size < 2 * request_size) {
        HSB_LOG_WARN("Kernel COE receiver buffer size is too small; packets will be dropped at high data rates.");
        HSB_LOG_WARN("Resolve this with \"echo {} | sudo tee /proc/sys/net/core/rmem_max\"", request_size);
    }
}

void LinuxCoeReceiverOp::start_receiver()
{
    const auto& p = this->params();

    // Resolve effective CPU affinity (override > params > env-var > default); same precedence
    // as LinuxReceiverOp's own.
    {
        std::vector<int> base = affinity_override_ ? *affinity_override_ : p.receiver_affinity;
        if (base.empty()) {
            const char* env = std::getenv("HOLOLINK_AFFINITY");
            if (env && std::strlen(env) > 0) {
                base = { std::atoi(env) };
            } else {
                base = { DEFAULT_RECEIVER_AFFINITY };
            }
        }
        effective_affinity_ = std::move(base);
    }

    // Unlike LinuxReceiverOp's multi-page ring buffer, LinuxCoeReceiver uses a single buffer
    // (see its own constructor doc comment): frame_size rounded up to a page, plus one
    // metadata block.
    const size_t buffer_size
        = hololink::core::round_up(p.frame_size, hololink::core::PAGE_SIZE) + hololink::METADATA_SIZE;

    frame_memory_ = std::make_shared<hololink::ReceiverMemoryDescriptor>(p.frame_context, buffer_size);

    receiver_.reset(new LinuxCoeReceiver(
        frame_memory_->get(),
        buffer_size,
        data_socket_.get(),
        p.coe_channel));

    receiver_->set_frame_ready([this](const LinuxCoeReceiver&) {
        this->frame_arrived();
        this->frame_ready();
    });

    receiver_thread_.reset(new std::thread(&LinuxCoeReceiverOp::run, this));
    // pthread_setname_np silently truncates names longer than 15 characters.
    const int rc = pthread_setname_np(receiver_thread_->native_handle(), "linux_coe_recv");
    if (rc != 0) {
        HSB_LOG_WARN("Failed to set receiver thread name, error={}", rc);
    }

    try {
        // vlan_enabled is not exposed as a LinuxCoeReceiverParams field: the old HSDK4-era
        // Python LinuxCoeReceiverOp always hardcoded False here too (never a constructor
        // kwarg there either).
        hololink_channel_->configure_coe(p.coe_channel, p.frame_size, p.pixel_width, /*vlan_enabled=*/false);
    } catch (const std::exception& e) {
        HSB_LOG_ERROR("start_receiver failed after receiver thread started: {}", e.what());
        receiver_->close();
        receiver_thread_->join();
        receiver_thread_.reset();
        receiver_.reset();
        throw;
    }
}

void LinuxCoeReceiverOp::run()
{
    CudaCheck(cuCtxSetCurrent(this->params().frame_context));

    if (!effective_affinity_.empty()) {
        cpu_set_t cpu_set;
        CPU_ZERO(&cpu_set);
        for (int cpu : effective_affinity_) {
            if (cpu < 0 || cpu >= CPU_SETSIZE) {
                throw std::runtime_error(
                    fmt::format("CPU affinity {} is out of range [0, {})", cpu, CPU_SETSIZE));
            }
            CPU_SET(cpu, &cpu_set);
        }
        if (sched_setaffinity(0, sizeof(cpu_set), &cpu_set) != 0) {
            HSB_LOG_WARN("Failed to set CPU affinity, errno={}", errno);
        }
    }

    receiver_->run();
}

void LinuxCoeReceiverOp::stop_receiver()
{
    // Unlike LinuxReceiverOp::stop_receiver(), this does NOT call
    // hololink_channel_->unconfigure(): that method clears configure_roce()'s own RoCE/UDP
    // registers (queue pair, rkey, host MAC/IP/port) -- none of which configure_coe() ever
    // wrote -- confirmed by reading DataChannel::unconfigure() directly. The old HSDK4-era
    // Python LinuxCoeReceiverOp never called an unconfigure equivalent either.
    if (receiver_) {
        receiver_->close();
    }
    data_socket_.reset();
    if (receiver_thread_) {
        receiver_thread_->join();
        receiver_thread_.reset();
    }
    frame_memory_.reset();
}

std::tuple<CUdeviceptr, std::shared_ptr<hololink::Metadata>>
LinuxCoeReceiverOp::get_next_frame(double timeout_ms, CUstream cuda_stream)
{
    LinuxCoeReceiverMetadata linux_coe_receiver_metadata;
    if (!receiver_->get_next_frame(static_cast<unsigned>(timeout_ms), linux_coe_receiver_metadata, cuda_stream)) {
        return {};
    }

    auto metadata = std::make_shared<Metadata>();
    (*metadata)["frame_packets_received"] = int64_t(linux_coe_receiver_metadata.frame_packets_received);
    (*metadata)["frame_bytes_received"] = int64_t(linux_coe_receiver_metadata.frame_bytes_received);
    (*metadata)["frame_number"] = int64_t(linux_coe_receiver_metadata.frame_number);
    (*metadata)["received_frame_number"] = int64_t(linux_coe_receiver_metadata.received_frame_number);
    (*metadata)["received_s"] = int64_t(linux_coe_receiver_metadata.received_s);
    (*metadata)["received_ns"] = int64_t(linux_coe_receiver_metadata.received_ns);
    (*metadata)["timestamp_s"] = int64_t(linux_coe_receiver_metadata.frame_metadata.timestamp_s);
    (*metadata)["timestamp_ns"] = int64_t(linux_coe_receiver_metadata.frame_metadata.timestamp_ns);
    (*metadata)["metadata_s"] = int64_t(linux_coe_receiver_metadata.frame_metadata.metadata_s);
    (*metadata)["metadata_ns"] = int64_t(linux_coe_receiver_metadata.frame_metadata.metadata_ns);
    (*metadata)["crc"] = int64_t(linux_coe_receiver_metadata.frame_metadata.crc);
    (*metadata)["psn"] = int64_t(linux_coe_receiver_metadata.frame_metadata.psn);
    (*metadata)["bytes_written"] = int64_t(linux_coe_receiver_metadata.frame_metadata.bytes_written);

    return { frame_memory_->get(), metadata };
}

bool LinuxCoeReceiverOp::frames_ready()
{
    return receiver_->frames_ready();
}

template class BaseReceiverOp<LinuxCoeReceiverOp, LinuxCoeReceiverParams>;

} // namespace hololink::operators
