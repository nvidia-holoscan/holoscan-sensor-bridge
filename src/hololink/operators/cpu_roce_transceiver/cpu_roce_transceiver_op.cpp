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

void CpuRoceTransceiverOp::setup(holoscan::OperatorSpec& spec)
{
    BaseReceiverOp::setup(spec);

    spec.param(ibv_name_, "ibv_name", "IBVName", "IBV device to use",
        std::string("roceP5p3s0f0"));
    spec.param(ibv_port_, "ibv_port", "IBVPort", "Port number of IBV device", 1u);
    spec.param(tx_ibv_qp_, "tx_ibv_qp", "Tx remote QP",
        "Remote QP number for transmit operations", 2u);
    spec.param(slots_, "slots", "Slots",
        "Payload ring slots (max 4096). Duplex uses this count for RX and TX; "
        "TX must be a power of two. Each slot is HSB-page-rounded frame_size, "
        "plus one HSB page for frame metadata or when frame_size exceeds one "
        "HSB page; RX is registered twice against RLIMIT_MEMLOCK for host memory",
        1024u);
    spec.param(forward_, "forward", "Forward mode", "Enable forward mode", 0u);
    spec.param(gpu_memory_, "gpu_memory", "GPU memory",
        "Use CUDA device memory exported as DMA-BUF for the Forward RX ring. "
        "Fail instead of falling back to host memory. Forward-only",
        false);
    spec.param(frame_metadata_, "frame_metadata", "Frame metadata",
        "Reserve and receive the fixed HSB metadata trailer after each RX "
        "payload. The transport validates its location but does not "
        "deserialize or forward the trailer. The application must configure "
        "the HSB metadata-packet setting to match",
        false);
    spec.param(poll_cpu_, "poll_cpu", "PollCpu",
        "Forward Config::poll_cpu. -1 unset (inherit the monitor thread "
        "mask). Duplex rejects this field",
        UNSET_CPU);
    spec.param(rx_cpu_, "rx_cpu", "RxCpu",
        "Duplex Config::rx_cpu. Set with tx_cpu, or leave both -1",
        UNSET_CPU);
    spec.param(tx_cpu_, "tx_cpu", "TxCpu",
        "Duplex Config::tx_cpu. Set with rx_cpu, or leave both -1",
        UNSET_CPU);
}

void CpuRoceTransceiverOp::start()
{
    if (frame_ready_condition_) {
        frame_ready_condition_->event_state(
            holoscan::AsynchronousEventState::EVENT_WAITING);
    }
    // BaseReceiverOp::start runs device_start_ after start_receiver() has
    // already spawned the monitor. A throw there leaves the GXF entity in
    // StartPending, and GXF only stops entities that reached Started, so
    // stop() never runs. Quiesce here instead of deferring to the destructor.
    std::exception_ptr first;
    try {
        BaseReceiverOp::start();
    } catch (...) {
        first = std::current_exception();
        try {
            stop_receiver();
        } catch (...) {
        }
    }
    if (first) {
        std::rethrow_exception(first);
    }
}

void CpuRoceTransceiverOp::stop()
{
    if (frame_ready_condition_) {
        frame_ready_condition_->event_state(
            holoscan::AsynchronousEventState::EVENT_DONE);
    }
    // BaseReceiverOp::stop runs device_stop_ then stop_receiver(). If
    // device_stop_ throws, still quiesce the transceiver so a joinable
    // monitor thread cannot std::terminate in the destructor.
    std::exception_ptr first;
    try {
        BaseReceiverOp::stop();
    } catch (...) {
        first = std::current_exception();
        try {
            stop_receiver();
        } catch (...) {
        }
    }
    if (first) {
        std::rethrow_exception(first);
    }
}

CpuRoceTransceiverOp::~CpuRoceTransceiverOp()
{
    // Holoscan does reach stop() after a throwing compute(): GXFWrapper::tick
    // stores the exception and returns GXF_FAILURE, and graph deactivation
    // stops every entity that reached Started. This is the last resort for the
    // paths that bypass stop() anyway - a join() that threw during an earlier
    // teardown - so that a joinable monitor_thread_ cannot std::terminate in
    // unique_ptr's destructor. stop_receiver() is idempotent.
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

    // Revision-specific DataChannel subclasses may accept older register maps,
    // but CpuRoceTransceiver requires the 0x2602+ immediate-data layout.
    verify_imm_layout(*hololink_channel_);

    const size_t frame_size = frame_size_.get();
    const size_t metadata_offset
        = hololink::core::round_up(frame_size, hololink::core::PAGE_SIZE);
    const bool frame_metadata = frame_metadata_.get();
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
    config.ibv_name = ibv_name_.get();
    config.ibv_port = ibv_port_.get();
    config.local_ip = local_ip;
    const uint32_t forward_value = forward_.get();
    if (forward_value > 1) {
        throw std::invalid_argument(
            "CpuRoceTransceiverOp forward must be 0 (Duplex) or 1 (Forward)");
    }
    const bool forward = forward_value == 1;
    const bool gpu_memory = gpu_memory_.get();
    if (gpu_memory && !forward) {
        throw std::invalid_argument(
            "CpuRoceTransceiverOp gpu_memory is currently Forward-only");
    }
    const uint32_t slots = slots_.get();
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
    const int poll_cpu = poll_cpu_.get();
    const int rx_cpu = rx_cpu_.get();
    const int tx_cpu = tx_cpu_.get();
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
            frame_context_.get(), ring_bytes);
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
        rx_memory = make_gpu_memory(rx_gpu_memory_, rx_dmabuf_fd_);
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
        local_ip, peer_ip, tx_ibv_qp_.get(), frame_size, stride,
        config.rx.slot_count, gpu_memory ? "GPU DMA-BUF" : "host",
        frame_metadata ? "enabled" : "disabled",
        optional_cpu_text(config.poll_cpu),
        optional_cpu_text(config.rx_cpu), optional_cpu_text(config.tx_cpu));

    {
        std::lock_guard<std::mutex> lock(monitor_mutex_);
        stopped_ = false;
    }

    try {
        transceiver_ = std::make_unique<Transceiver>(std::move(config));
        transceiver_->setup();
        HSB_LOG_INFO("CpuRoceTransceiver created QP={:#x} rkey={:#x}",
            transceiver_->get_qp_number(), transceiver_->get_rkey());

        hololink_channel_->authenticate(
            transceiver_->get_qp_number(), transceiver_->get_rkey());
        transceiver_->connect(Transceiver::Peer {
            tx_ibv_qp_.get(), peer_ip, 0, 0, stride, 0 });
        HSB_LOG_INFO("CpuRoceTransceiver connected to FPGA QP={:#x}",
            tx_ibv_qp_.get());

        monitor_thread_
            = std::make_unique<std::thread>(&CpuRoceTransceiverOp::monitor_loop, this);
        // pthread_setname_np: 16-byte buffer including NUL (TASK_COMM_LEN).
        std::string thread_name = name();
        if (thread_name.size() > 15) {
            thread_name.resize(15);
        }
        const int name_error = pthread_setname_np(
            monitor_thread_->native_handle(), thread_name.c_str());
        if (name_error != 0) {
            HSB_LOG_WARN("Failed to set CpuRoce monitor thread name, error={}", name_error);
        }

        hololink_configured_ = true;
        try {
            hololink_channel_->configure_roce(
                transceiver_->external_frame_memory(), frame_size, stride,
                slots_.get(), local_port);
        } catch (const hololink::UnsupportedVersion&) {
            // configure_roce validates the version before any register write.
            // Do not make stop_receiver() call unconfigure() on a register map
            // that configure_roce rejected as unsupported.
            hololink_configured_ = false;
            throw;
        }
        HSB_LOG_INFO("HoloLink configured for CpuRoceTransceiver {}",
            forward ? "Forward" : "Duplex");
    } catch (...) {
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
    const auto stop_requested = [this] {
        std::lock_guard<std::mutex> lock(monitor_mutex_);
        return stopped_;
    };
    try {
        HSB_LOG_INFO("Starting CpuRoceTransceiver CQ monitor");
        transceiver_->blocking_monitor();
        HSB_LOG_INFO("CpuRoceTransceiver CQ monitor ended");
    } catch (const std::exception& e) {
        // Catch at the std::thread boundary to avoid std::terminate. HSB receiver
        // operators do not propagate receive-path termination through compute();
        // log it and leave graph lifetime to the application.
        try {
            if (stop_requested()) {
                HSB_LOG_INFO(
                    "CpuRoceTransceiver CQ monitor stopped during teardown: {}",
                    e.what());
            } else {
                HSB_LOG_ERROR(
                    "CpuRoceTransceiver CQ monitor failed; the datapath is inactive until "
                    "the operator is restarted: {}",
                    e.what());
            }
        } catch (...) {
        }
    } catch (...) {
        try {
            if (stop_requested()) {
                HSB_LOG_INFO(
                    "CpuRoceTransceiver CQ monitor stopped during teardown: "
                    "unknown exception");
            } else {
                HSB_LOG_ERROR(
                    "CpuRoceTransceiver CQ monitor failed; the datapath is inactive until "
                    "the operator is restarted: unknown exception");
            }
        } catch (...) {
        }
    }
}

std::tuple<CUdeviceptr, std::shared_ptr<hololink::Metadata>>
CpuRoceTransceiverOp::get_next_frame(double timeout_ms, CUstream cuda_stream)
{
    (void)cuda_stream;
    // No CPU frame stream, so this returns empty metadata. Honor timeout_ms
    // so a tick cannot park a scheduler worker. A monitor failure does not
    // wake this wait. With the default AsynchronousCondition, compute() is
    // not scheduled until stop(); if that condition is disabled, Holoscan
    // may call this repeatedly and each call returns after the timeout.
    std::unique_lock<std::mutex> lock(monitor_mutex_);
    monitor_cv_.wait_for(lock, std::chrono::duration<double, std::milli>(timeout_ms),
        [this] { return stopped_; });
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
    // Persistent CQ monitor: no CPU frame stream. Same as GpuRoceTransceiverOp.
    return false;
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

} // namespace hololink::operators
