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
 */

#include "cpu_roce_transceiver_op.hpp"
#include "../base_receiver_op.inl"

#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <exception>
#include <limits>
#include <netinet/in.h>
#include <optional>
#include <pthread.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <utility>

#include <hololink/common/cuda_helper.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/core/logging_internal.hpp>
#include <hololink/core/networking.hpp>

namespace hololink::operators {

namespace {
    constexpr int UNSET_CPU = -1;
    constexpr uint32_t ROCE_DATA_PORT = 4791;
    constexpr int64_t MINIMUM_IMM_HSB_IP_VERSION = 0x2602;

    bool cpu_is_set(int cpu) { return cpu != UNSET_CPU; }

    std::string optional_cpu_text(const std::optional<int>& cpu)
    {
        return cpu ? std::to_string(*cpu) : std::string("unset");
    }

    void verify_imm_layout(hololink::DataChannel& channel)
    {
        const auto version
            = channel.enumeration_metadata().get<int64_t>("hsb_ip_version");
        if (!version) {
            throw hololink::UnsupportedVersion(
                "CpuRoceTransceiverOp requires hsb_ip_version >= 0x2602; "
                "the field is missing");
        }
        if (*version < MINIMUM_IMM_HSB_IP_VERSION) {
            throw hololink::UnsupportedVersion(
                "CpuRoceTransceiverOp requires hsb_ip_version >= 0x2602 "
                "for the 12-bit immediate-data slot layout");
        }
    }
}

CpuRoceTransceiverOp::~CpuRoceTransceiverOp()
{
    // The run stops a started operator (BaseReceiverOp::stop() always calls stop_receiver()),
    // and a failed start() cleans up after itself. This is the last resort for the paths that
    // bypass that anyway - a join() that threw during an earlier teardown - so that a joinable
    // monitor_thread_ cannot std::terminate in unique_ptr's destructor. stop_receiver() is
    // idempotent.
    try {
        if (monitor_thread_ || transceiver_ || hololink_configured_
            || rx_gpu_memory_) {
            stop_receiver();
        }
    } catch (...) {
    }
}

void CpuRoceTransceiverOp::start_receiver()
{
    using hololink::transport::roce::FrameLayout;
    using hololink::transport::roce::RingConfig;

    const auto& p = params();

    // Revision-specific DataChannel subclasses may accept older register maps,
    // but CpuRoceTransceiver requires the 0x2602+ immediate-data layout.
    verify_imm_layout(*hololink_channel_);

    const size_t frame_size = p.frame_size;
    const size_t metadata_offset
        = hololink::core::round_up(frame_size, hololink::core::PAGE_SIZE);
    const bool frame_metadata = p.frame_metadata;
    static_assert(hololink::METADATA_SIZE >= hololink::core::PAGE_SIZE);
    static_assert(
        (hololink::METADATA_SIZE % hololink::core::PAGE_SIZE) == 0);
    size_t stride = metadata_offset;
    if (frame_metadata || frame_size > hololink::core::PAGE_SIZE) {
        // DataChannel::configure_roce reserves one HSB metadata page when
        // payload exceeds one protocol page even when metadata is disabled.
        // With frame_metadata, this tail is the actual metadata region.
        stride += hololink::METADATA_SIZE;
    }

    auto [local_ip, local_port] = local_ip_and_port();
    const std::string peer_ip = hololink_channel_->peer_ip();

    Transceiver::Config config;
    // Without an ibv_name, the device that reaches the board.
    std::tie(config.ibv_name, config.ibv_port) = p.ibv_name.empty()
        ? hololink_channel_->infiniband_device()
        : std::make_pair(p.ibv_name, p.ibv_port);
    config.local_ip = local_ip;
    if (p.forward > 1) {
        throw std::invalid_argument(
            "CpuRoceTransceiverOp forward must be 0 (Duplex) or 1 (Forward)");
    }
    const bool forward = p.forward == 1;
    const bool gpu_memory = p.gpu_memory;
    if (gpu_memory && !forward) {
        throw std::invalid_argument(
            "CpuRoceTransceiverOp gpu_memory is currently Forward-only");
    }
    const uint32_t slots = p.slots;
    if (slots == 0) {
        throw std::invalid_argument(
            "CpuRoceTransceiverOp requires at least one payload-ring slot");
    }
    if (stride > std::numeric_limits<size_t>::max() / slots) {
        throw std::invalid_argument(
            "CpuRoceTransceiverOp payload ring size overflows size_t");
    }
    if (stride > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(
            "CpuRoceTransceiverOp stride exceeds ibv_sge length");
    }
    // DataChannel and the modern HSB IMM slot field both cap the ring at
    // 4096. Reject before allocating/exporting a GPU payload ring.
    constexpr uint32_t kMaxSlotCount = 4096;
    if (frame_size == 0) {
        throw std::invalid_argument(
            "CpuRoceTransceiverOp requires a non-zero frame_size");
    }
    if (slots > kMaxSlotCount) {
        throw std::invalid_argument(
            "CpuRoceTransceiverOp requires slots <= "
            + std::to_string(kMaxSlotCount));
    }

    config.mode = forward ? Transceiver::Mode::Forward : Transceiver::Mode::Duplex;
    config.tx_operation = Transceiver::TxOperation::Send;
    const int poll_cpu = p.poll_cpu;
    const int rx_cpu = p.rx_cpu;
    const int tx_cpu = p.tx_cpu;
    if (forward) {
        if (cpu_is_set(rx_cpu) || cpu_is_set(tx_cpu)) {
            throw std::invalid_argument(
                "CpuRoceTransceiverOp rx_cpu and tx_cpu are Duplex-only");
        }
        if (cpu_is_set(poll_cpu)) {
            config.poll_cpu = poll_cpu;
        }
    } else {
        if (cpu_is_set(poll_cpu)) {
            throw std::invalid_argument(
                "CpuRoceTransceiverOp poll_cpu is not valid in Duplex; "
                "use rx_cpu and tx_cpu");
        }
        if (cpu_is_set(rx_cpu) != cpu_is_set(tx_cpu)) {
            throw std::invalid_argument(
                "CpuRoceTransceiverOp Duplex requires both rx_cpu and "
                "tx_cpu, or neither");
        }
        if (cpu_is_set(rx_cpu)) {
            config.rx_cpu = rx_cpu;
            config.tx_cpu = tx_cpu;
        }
    }

    const size_t ring_bytes = stride * slots;

    const auto make_gpu_memory =
        [&](std::shared_ptr<hololink::ReceiverMemoryDescriptor>& owner,
            int& dmabuf_fd) -> hololink::transport::roce::PayloadMemory {
        owner = std::make_shared<hololink::ReceiverMemoryDescriptor>(
            p.frame_context, ring_bytes);
        if (owner->get_host_ptr() != nullptr) {
            throw std::runtime_error(
                "CpuRoceTransceiverOp gpu_memory requested, but CUDA device "
                "memory allocation/export fell back to pinned host memory");
        }

        const CUdeviceptr address = owner->get();
        int sync_memops = 1;
        CudaCheck(cuPointerSetAttribute(
            &sync_memops, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, address));
        // cuMemGetHandleForAddressRange requires a page-aligned base and
        // length. address is already page-aligned (ReceiverMemoryDescriptor
        // guarantees it), but ring_bytes is only a multiple of hololink's
        // 128-byte PAGE_SIZE, not the OS page. Export the OS-page-rounded
        // span; ReceiverMemoryDescriptor over-allocates by one OS page
        // precisely so this rounded span stays in-bounds.
        const size_t export_bytes
            = hololink::core::round_up(ring_bytes, getpagesize());
        CudaCheck(cuMemGetHandleForAddressRange(
            &dmabuf_fd, address, export_bytes,
            CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0));
        HSB_LOG_INFO(
            "CpuRoceTransceiverOp exported GPU payload ring address={:#x} "
            "bytes={:#x} fd={}",
            address, export_bytes, dmabuf_fd);
        return hololink::transport::roce::DmaBufMemory {
            reinterpret_cast<void*>(address), dmabuf_fd, 0, export_bytes
        };
    };

    hololink::transport::roce::PayloadMemory rx_memory
        = hololink::transport::roce::TransceiverAllocation {};
    if (gpu_memory) {
        try {
            rx_memory = make_gpu_memory(rx_gpu_memory_, rx_dmabuf_fd_);
        } catch (...) {
            // stop_receiver() frees the ring and the descriptor; nothing else is set up yet.
            auto first = std::current_exception();
            try {
                stop_receiver();
            } catch (...) {
            }
            std::rethrow_exception(first);
        }
    }

    config.rx = RingConfig {
        std::move(rx_memory),
        stride,
        slots,
        FrameLayout {
            frame_size,
            frame_metadata ? metadata_offset : 0,
            frame_metadata ? hololink::METADATA_SIZE : 0,
        },
    };
    if (!forward) {
        config.tx = RingConfig {
            hololink::transport::roce::TransceiverAllocation {},
            stride,
            slots,
            FrameLayout { frame_size, 0, 0 },
        };
    }

    HSB_LOG_INFO(
        "Starting CpuRoceTransceiver {}: ibv_name={} ibv_port={} "
        "local_ip={} peer_ip={} peer_qp={} frame_size={} stride={} slots={} "
        "memory={} frame_metadata={} poll_cpu={} rx_cpu={} tx_cpu={}",
        forward ? "Forward" : "Duplex", config.ibv_name, config.ibv_port,
        local_ip, peer_ip, p.tx_ibv_qp, frame_size, stride,
        config.rx.slot_count, gpu_memory ? "GPU DMA-BUF" : "host",
        frame_metadata ? "enabled" : "disabled",
        optional_cpu_text(config.poll_cpu),
        optional_cpu_text(config.rx_cpu), optional_cpu_text(config.tx_cpu));

    {
        std::lock_guard<std::mutex> lock(monitor_mutex_);
        stopped_ = false;
        receive_path_failed_ = false;
        receive_path_failure_.clear();
    }

    try {
        transceiver_ = std::make_unique<Transceiver>(std::move(config));
        transceiver_->setup();
        HSB_LOG_INFO("CpuRoceTransceiver created QP={:#x} rkey={:#x}",
            transceiver_->get_qp_number(), transceiver_->get_rkey());

        hololink_channel_->authenticate(
            transceiver_->get_qp_number(), transceiver_->get_rkey());
        transceiver_->connect(Transceiver::Peer {
            p.tx_ibv_qp, peer_ip, 0, 0, stride, 0 });
        HSB_LOG_INFO("CpuRoceTransceiver connected to FPGA QP={:#x}",
            p.tx_ibv_qp);

        monitor_thread_
            = std::make_unique<std::thread>(&CpuRoceTransceiverOp::monitor_loop, this);
        const int name_error = pthread_setname_np(monitor_thread_->native_handle(), "cpu_roce_mon");
        if (name_error != 0) {
            HSB_LOG_WARN("Failed to set CpuRoce monitor thread name, error={}", name_error);
        }

        hololink_configured_ = true;
        try {
            hololink_channel_->configure_roce(
                transceiver_->external_frame_memory(), frame_size, stride,
                slots, local_port);
        } catch (const hololink::UnsupportedVersion&) {
            // configure_roce validates the version before any register write.
            // Do not make stop_receiver() call unconfigure() on a register map
            // that configure_roce rejected as unsupported.
            hololink_configured_ = false;
            throw;
        }
        HSB_LOG_INFO("HoloLink configured for CpuRoceTransceiver {}",
            forward ? "Forward" : "Duplex");

        // A monitor that failed at once (for example, on a CPU it can't be pinned to) fails the
        // start with its own error, so that the device isn't started for a dead datapath.
        // compute() reports a later failure.
        std::unique_lock<std::mutex> lock(monitor_mutex_);
        if (receive_path_failed_) {
            const std::string failure = receive_path_failure_;
            lock.unlock();
            throw std::runtime_error(
                "CpuRoceTransceiver CQ monitor failed: " + failure);
        }
    } catch (...) {
        // A start_receiver() that throws gets no stop_receiver() from BaseReceiverOp, so undo
        // what was set up here. The failure that got here is the one reported.
        auto first = std::current_exception();
        try {
            stop_receiver();
        } catch (...) {
        }
        std::rethrow_exception(first);
    }
}

void CpuRoceTransceiverOp::stop_receiver()
{
    HSB_LOG_INFO("Stopping CpuRoceTransceiver");

    if (monitor_thread_ && monitor_thread_->get_id() == std::this_thread::get_id()) {
        // Teardown always runs on the operator's control thread; joining the
        // monitor from itself would deadlock.
        throw std::logic_error(
            "CpuRoceTransceiverOp::stop_receiver called from the monitor thread");
    }

    {
        std::lock_guard<std::mutex> lock(monitor_mutex_);
        stopped_ = true;
    }
    monitor_cv_.notify_all();

    std::exception_ptr first;
    const auto capture = [&](auto&& step) {
        try {
            step();
        } catch (...) {
            if (!first) {
                first = std::current_exception();
            }
        }
    };

    // Capture instead of propagating: an unconfigure() failure must not keep
    // close() from releasing blocking_monitor(), or the join below would hang.
    if (hololink_configured_) {
        hololink_configured_ = false;
        capture([&] { hololink_channel_->unconfigure(); });
    }
    capture([&] {
        if (transceiver_) {
            transceiver_->close();
        }
    });

    bool joined = true;
    if (monitor_thread_ && monitor_thread_->joinable()) {
        joined = false;
        capture([&] {
            monitor_thread_->join();
            joined = true;
        });
    }
    if (joined) {
        // A successful join is the only proof that the monitor has stopped
        // touching transceiver_ and this operator; close() releasing its waiter
        // does not mean monitor_loop() has returned. So a failed join never
        // falls back to detach: keep the thread and the transceiver alive and
        // let a later stop_receiver() retry the join.
        monitor_thread_.reset();
        transceiver_.reset();
        if (rx_dmabuf_fd_ >= 0) {
            ::close(rx_dmabuf_fd_);
            rx_dmabuf_fd_ = -1;
        }
        rx_gpu_memory_.reset();
    }
    data_socket_.reset();
    if (first) {
        std::rethrow_exception(first);
    }
}

void CpuRoceTransceiverOp::monitor_loop() noexcept
{
    try {
        HSB_LOG_INFO("Starting CpuRoceTransceiver CQ monitor");
        transceiver_->blocking_monitor();
        HSB_LOG_INFO("CpuRoceTransceiver CQ monitor ended");
    } catch (const std::exception& e) {
        // Catch at the std::thread boundary to avoid std::terminate.
        on_monitor_failure(e.what());
    } catch (...) {
        on_monitor_failure("unknown exception");
    }
}

void CpuRoceTransceiverOp::on_monitor_failure(const char* what) noexcept
{
    // A failure outside teardown ends the run, as a fatal receive-path event does in
    // RoceReceiverOp.
    try {
        bool during_teardown = false;
        {
            std::lock_guard<std::mutex> lock(monitor_mutex_);
            during_teardown = stopped_;
            if (!during_teardown) {
                receive_path_failed_ = true;
                receive_path_failure_ = what;
            }
        }
        if (during_teardown) {
            HSB_LOG_INFO("CpuRoceTransceiver CQ monitor stopped during teardown: {}", what);
            return;
        }
        HSB_LOG_ERROR("CpuRoceTransceiver CQ monitor failed; no further frames will be "
                      "forwarded or received: {}",
            what);
        monitor_cv_.notify_all();
        frame_ready();
    } catch (...) {
    }
}

std::tuple<CUdeviceptr, std::shared_ptr<hololink::Metadata>>
CpuRoceTransceiverOp::get_next_frame(double timeout_ms, CUstream cuda_stream)
{
    (void)cuda_stream;
    // No CPU frame stream, so this returns empty metadata. compute() runs only when the monitor
    // fails (it then fails the run through receive_path_failed()), but honor timeout_ms anyway
    // so that a call can't park a scheduler worker.
    std::unique_lock<std::mutex> lock(monitor_mutex_);
    monitor_cv_.wait_for(lock, std::chrono::duration<double, std::milli>(timeout_ms),
        [this] { return stopped_ || receive_path_failed_; });
    return {};
}

std::tuple<std::string, uint32_t> CpuRoceTransceiverOp::local_ip_and_port()
{
    sockaddr_in ip {};
    socklen_t ip_len = sizeof(ip);
    if (getsockname(data_socket_.get(), reinterpret_cast<sockaddr*>(&ip), &ip_len)
        < 0) {
        throw std::runtime_error(fmt::format(
            "getsockname failed with errno={}: \"{}\"", errno,
            std::strerror(errno)));
    }
    return { inet_ntoa(ip.sin_addr), ROCE_DATA_PORT };
}

bool CpuRoceTransceiverOp::frames_ready()
{
    // Persistent CQ monitor: no CPU frame stream, as in GpuRoceTransceiverOp. Only a failed
    // receive path has something for compute() to do: fail the run.
    return receive_path_failed();
}

bool CpuRoceTransceiverOp::receive_path_failed()
{
    std::lock_guard<std::mutex> lock(monitor_mutex_);
    return receive_path_failed_;
}

uint8_t* CpuRoceTransceiverOp::get_ring_addr(bool is_rx) const
{
    if (!transceiver_) {
        return nullptr;
    }
    return is_rx ? transceiver_->get_rx_ring_data_addr()
                 : transceiver_->get_tx_ring_data_addr();
}

size_t CpuRoceTransceiverOp::get_ring_stride_sz(bool is_rx) const
{
    if (!transceiver_) {
        return 0;
    }
    return is_rx ? transceiver_->get_rx_ring_stride()
                 : transceiver_->get_tx_ring_stride();
}

uint32_t CpuRoceTransceiverOp::get_ring_stride_num(bool is_rx) const
{
    if (!transceiver_) {
        return 0;
    }
    return is_rx ? transceiver_->get_rx_ring_slot_count()
                 : transceiver_->get_tx_ring_slot_count();
}

uint64_t* CpuRoceTransceiverOp::get_ring_flag_addr(bool is_rx) const
{
    if (!transceiver_) {
        return nullptr;
    }
    return is_rx ? transceiver_->get_rx_ring_flag_addr()
                 : transceiver_->get_tx_ring_flag_addr();
}

template class BaseReceiverOp<CpuRoceTransceiverOp, CpuRoceTransceiverParams>;

} // namespace hololink::operators
