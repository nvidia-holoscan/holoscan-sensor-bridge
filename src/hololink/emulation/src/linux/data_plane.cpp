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

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <time.h>
#include <unistd.h>

#include "dlpack/dlpack.h"
#include <cuda_runtime.h>

#include "linux/data_plane.hpp"
#include "linux/hsb_emulator.hpp"
#include "linux/net.hpp"
#include "utils.hpp"

namespace hololink::emulation {

// Helper: downcast the base-class DataPlaneCtxt* to the linux extension. Standard-layout
// guarantees zero offset between LinuxDataPlaneCtxt and its first `base` member.
// (Free function, not a DataPlane member: data_plane_ctxt_ is protected, so DataPlane
// methods call this via `linux_ctxt(data_plane_ctxt_.get())`.)
static inline LinuxDataPlaneCtxt* linux_ctxt(DataPlaneCtxt* base)
{
    return reinterpret_cast<LinuxDataPlaneCtxt*>(base);
}

// bootp thread manager
class BootpThread {
public:
    BootpThread(DataPlane& data_plane, IPAddress& ip_address, HSBConfiguration& configuration)
        : data_plane_(data_plane)
    {
        socket_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (socket_fd_ < 0) {
            fprintf(stderr, "Failed to create UDP socket: %d - %s\n", errno, strerror(errno));
            throw std::runtime_error("Failed to create data channel");
        }
        // Enable address reuse
        int reuse = 1;
        if (setsockopt(socket_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
            fprintf(stderr, "Failed to set reuse address on bootp socket...socket will be set up with this option disabled: %d - %s\n", errno, strerror(errno));
        }

        struct sockaddr_in bind_addr {
            0
        };
        bind_addr.sin_family = AF_INET;
        bind_addr.sin_port = htons(BOOTP_REPLY_PORT);
        bind_addr.sin_addr.s_addr = ip_address.ip_address;

        if (bind(socket_fd_, (struct sockaddr*)&bind_addr, sizeof(bind_addr)) < 0) {
            fprintf(stderr, "Failed to bind bootp socket: %d - %s\n", errno, strerror(errno));
            close(socket_fd_);
            socket_fd_ = -1;
            throw std::runtime_error("Failed to create data channel");
        }

        // Enable broadcast
        int broadcast = 1;
        if (setsockopt(socket_fd_, SOL_SOCKET, SO_BROADCAST, &broadcast, sizeof(broadcast)) < 0) {
            fprintf(stderr, "Failed to set broadcast option. invalid bootp socket: %d - %s\n", errno, strerror(errno));
            close(socket_fd_);
            socket_fd_ = -1;
            throw std::runtime_error("Failed to create data channel");
        }

        init_bootp_packet(bootp_buffer_, ip_address, configuration);
        running = true;
        thread_ = std::thread(&BootpThread::loop, this, get_broadcast_address(ip_address));
    }

    ~BootpThread()
    {
        running = false;
        if (thread_.joinable()) {
            thread_.join();
        }
        if (socket_fd_ >= 0) {
            close(socket_fd_);
        }
    }
    int broadcast_bootp()
    {
        if (data_plane_.is_running()) {
            update_bootp_packet(bootp_buffer_, data_plane_.get_start_time());
            ssize_t sent = sendmsg(socket_fd_, &msg_, 0);
            if (sent < 0) {
                fprintf(stderr, "Failed to send bootp packet. error %d - %s\n", errno, strerror(errno));
                return -1;
            }
        }
        return 0;
    }

private:
    void loop(uint32_t broadcast_address)
    {
        struct iovec iov {
            0
        };
        struct sockaddr_in dest_addr {
            0
        };

        // we reuse the code from init_bootp_packet() to build the bootp packet, but we skip the headers because this implementation
        // will use socket buffers directly.
        iov.iov_base = &bootp_buffer_[ETHER_HDR_LEN + IP_HDR_LEN + UDP_HDR_LEN];
        iov.iov_len = BOOTP_SIZE;

        // initialize the socket message header
        // set up broadcast address on ipv4 port 8192
        dest_addr.sin_family = AF_INET;
        dest_addr.sin_port = htons(BOOTP_REQUEST_PORT);
        dest_addr.sin_addr.s_addr = broadcast_address;

        msg_.msg_name = &dest_addr;
        msg_.msg_namelen = sizeof(dest_addr);
        msg_.msg_iov = &iov;
        msg_.msg_iovlen = 1;

        while (true) {
            if (!running) {
                break;
            }
            broadcast_bootp();
            std::this_thread::sleep_for(std::chrono::seconds(BOOTP_INTERVAL_SEC));
        }
    }
    DataPlane& data_plane_; // don't want to store a copy in thread so store it here and access it by reference
    std::thread thread_;
    std::atomic<bool> running { false };
    uint8_t bootp_buffer_[sizeof(BootpPacket) + ETHER_HDR_LEN + IP_HDR_LEN + UDP_HDR_LEN];
    struct msghdr msg_ {
        0
    };
    int socket_fd_ { -1 };
};

int DataPlane::broadcast_bootp()
{
    return linux_ctxt(data_plane_ctxt_.get())->bootp_thread->broadcast_bootp();
}

// Out-of-line so `delete bootp_thread` sees the complete BootpThread type defined
// above. Also frees the realloc-grown double_buffer.
LinuxDataPlaneCtxt::~LinuxDataPlaneCtxt()
{
    delete bootp_thread;
    ::free(double_buffer);
}

// Build the platform-default context for the public ctor: heap-allocate a LinuxDataPlaneCtxt
// and wrap its `base` in a unique_ptr whose deleter knows to downcast back to the extension
// type before calling delete. Standard-layout guarantees &ext->base == ext.
static std::unique_ptr<DataPlaneCtxt, std::function<void(DataPlaneCtxt*)>>
make_default_linux_ctxt()
{
    LinuxDataPlaneCtxt* owned = new LinuxDataPlaneCtxt();
    if (!owned) {
        Error_Handler("Failed to allocate LinuxDataPlaneCtxt");
    }
    return {
        &owned->base,
        [](DataPlaneCtxt* p) { delete reinterpret_cast<LinuxDataPlaneCtxt*>(p); }
    };
}

// Public ctor: application code path. Delegate to the protected overload with a
// heap-allocated default context whose deleter performs the matching `delete`.
DataPlane::DataPlane(HSBEmulator& hsb_emulator, const IPAddress& ip_address, uint8_t data_plane_id, uint8_t sensor_id)
    : DataPlane(hsb_emulator, ip_address, data_plane_id, sensor_id, make_default_linux_ctxt())
{
}

// HSBEmulator must remain alive for as long as the longest-lived DataPlane object it was used to construct.
// IPAddress is both the source IP address and subnet mask to be able to set the appropriate broadcast address (!cannot use INADDR_BROADCAST!).
// `ctxt` is the caller-supplied owning unique_ptr; the caller chose the deleter (heap delete,
// no-op for static, ...). This overload is `protected`: only subclasses can reach it.
DataPlane::DataPlane(HSBEmulator& hsb_emulator, const IPAddress& ip_address, uint8_t data_plane_id, uint8_t sensor_id,
    std::unique_ptr<DataPlaneCtxt, std::function<void(DataPlaneCtxt*)>> ctxt)
    : registers_(hsb_emulator.get_memory())
    , ip_address_(ip_address)
    , configuration_(hsb_emulator.get_config())
    , sensor_id_(sensor_id)
    , data_plane_id_(data_plane_id)
    , data_plane_ctxt_(std::move(ctxt))
{
    if (!data_plane_ctxt_) {
        Error_Handler("DataPlane: null DataPlaneCtxt passed to protected ctor");
    }

    // Look up (or lazily create) the per-data-plane and per-sensor register slices in the
    // HSBEmulator's vector caches, indexed by data_plane_id / sensor_id. DataPlane instances
    // sharing a data_plane_id share the hif slice; instances sharing a sensor_id share the vp
    // slice. The caches live on the linux-side extension (LinuxHSBEmulatorCtxt) since STM32
    // uses file-scope arrays instead. DataPlane is friend of HSBEmulator and reaches into
    // HSBEmulator::ctxt_, then downcasts to the linux extension to reach the caches.
    LinuxHSBEmulatorCtxt* lhsb = reinterpret_cast<LinuxHSBEmulatorCtxt*>(hsb_emulator.ctxt_.get());
    data_plane_ctxt_->dp_registers
        = get_or_create_dp_registers(*lhsb, data_plane_id_);
    data_plane_ctxt_->dp_sensor_registers
        = get_or_create_dp_sensor_registers(*lhsb, sensor_id_);

    linux_ctxt(data_plane_ctxt_.get())->metadata_mutex = get_metadata_mutex(*lhsb);

    // common board initialization
    init(hsb_emulator);

    // create the BootpThread instance after init() because it requires the final DataPlane::configuration_.
    // Raw pointer ownership: ~LinuxDataPlaneCtxt deletes it.
    linux_ctxt(data_plane_ctxt_.get())->bootp_thread
        = new BootpThread(*this, ip_address_, configuration_);
    if (!linux_ctxt(data_plane_ctxt_.get())->bootp_thread) {
        Error_Handler("Failed to create bootp thread");
    }
}

void DataPlane::start()
{
    LinuxDataPlaneCtxt* lctxt = linux_ctxt(data_plane_ctxt_.get());
    std::scoped_lock<std::mutex> lock(lctxt->running_mutex);
    if (lctxt->base.running) {
        return;
    }
    lctxt->base.running = true;
    clock_gettime(CLOCK_REALTIME, &lctxt->base.start_time);
}

void DataPlane::stop()
{
    if (!data_plane_ctxt_) {
        return;
    }
    LinuxDataPlaneCtxt* lctxt = linux_ctxt(data_plane_ctxt_.get());
    std::scoped_lock<std::mutex> lock(lctxt->running_mutex);
    lctxt->base.running = false;
}

bool DataPlane::is_running()
{
    if (!data_plane_ctxt_) {
        return false;
    }
    LinuxDataPlaneCtxt* lctxt = linux_ctxt(data_plane_ctxt_.get());
    std::scoped_lock<std::mutex> lock(lctxt->running_mutex);
    return lctxt->base.running;
}

// Linux-only: copy a CUDA-device DLTensor into a host bounce buffer (owned by
// LinuxDataPlaneCtxt) before forwarding to the transmitter. Returns the byte count
// copied (>0) for device memory, 0 if the tensor is already host-accessible, or -1 on
// failure. Host-accessible cases (kDLCPU / kDLCUDAHost) early-return 0 so the caller
// uses tensor.data directly.
static int64_t double_buffer_gpu_memory(uint8_t** double_buffer_, int64_t* double_buffer_size_, const DLTensor& tensor)
{
    switch (tensor.device.device_type) {
    case kDLCPU:
    case kDLCUDAHost:
        return 0;
    // Treat managed memory as device-resident and bounce; cudaMemcpyDefault could avoid
    // the copy when Unified Addressing is on, but the perf win isn't guaranteed.
    case kDLCUDAManaged:
    case kDLCUDA: {
        const int64_t n_bytes = DLTensor_n_bytes(tensor);
        if (*double_buffer_size_ < n_bytes) {
            uint8_t* buffer = (uint8_t*)realloc(*double_buffer_, n_bytes);
            if (!buffer) {
                auto err_str = std::string("failed to reallocate double buffer from ") + std::to_string(*double_buffer_size_) + " to " + std::to_string(n_bytes);
                Error_Handler(err_str.c_str());
                return -1;
            }
            *double_buffer_ = buffer;
            *double_buffer_size_ = n_bytes;
        }
        cudaError_t error = cudaMemcpy(*double_buffer_, tensor.data, (size_t)n_bytes, cudaMemcpyDeviceToHost);
        if (cudaSuccess != error) {
            auto err_str = std::string("cudaMemcpy Device->Host failed: ") + std::string(cudaGetErrorString(error));
            Error_Handler(err_str.c_str());
            return -1;
        }
        return n_bytes;
    }
    default:
        auto err_str = std::string("Unsupported device memory type: ") + std::to_string(tensor.device.device_type);
        Error_Handler(err_str.c_str());
        return -1;
    }
}

int64_t DataPlane::send(const DLTensor& tensor, FrameMetadata* frame_metadata)
{
    if (!transmitter_) {
        fprintf(stderr, "DataPlane::send() no transmitter\n");
        return -1;
    }

    if (frame_metadata == nullptr) {
        frame_metadata = DEFAULT_FRAME_METADATA;
    }

    // metadata_mutex serializes update_metadata + transmitter->send. NOT running_mutex —
    // send paths do not read base.running.
    LinuxDataPlaneCtxt* lctxt = linux_ctxt(data_plane_ctxt_.get());
    {
        std::scoped_lock<std::mutex> lock(*lctxt->metadata_mutex);
        update_metadata();
    }

    // Bounce CUDA-device data into the per-DataPlane host buffer; host-memory tensors
    // (CPU / pinned / managed) get returned with size==0 and the original pointer is
    // used directly.
    uint8_t* content = (uint8_t*)tensor.data;
    int64_t size = double_buffer_gpu_memory(&lctxt->double_buffer, &lctxt->double_buffer_size, tensor);
    if (size < 0) {
        Error_Handler("Failed to copy tensor to host");
        return -1;
    } else if (size > 0) {
        content = lctxt->double_buffer;
    } else {
        size = DLTensor_n_bytes(tensor);
    }

    int64_t n_bytes = transmitter_->send(data_plane_ctxt_.get(), content, size, frame_metadata);
    if (n_bytes < 0) {
        fprintf(stderr, "DataPlane::send() error sending tensor\n");
    }
    return n_bytes;
}

int64_t DataPlane::send(const uint8_t* buffer, size_t buffer_size, FrameMetadata* frame_metadata)
{
    if (!transmitter_) {
        fprintf(stderr, "DataPlane::send() no transmitter\n");
        return -1;
    }
    LinuxDataPlaneCtxt* lctxt = linux_ctxt(data_plane_ctxt_.get());
    {
        std::scoped_lock<std::mutex> lock(*lctxt->metadata_mutex);
        update_metadata();
    }
    int64_t n_bytes = transmitter_->send(data_plane_ctxt_.get(), buffer, buffer_size, frame_metadata);
    if (n_bytes < 0) {
        fprintf(stderr, "DataPlane::send() error sending buffer\n");
    }
    return n_bytes;
}

}
