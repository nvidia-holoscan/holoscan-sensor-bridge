/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Compile the receiver into this TU so the private device-trailer copy is
 * visible. Do not also link hololink::operators::cpu_roce_frame_receiver.
 */

#include "gtest/gtest.h"

#include <cuda.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <optional>
#include <unistd.h>
#include <variant>
#include <vector>

#include "../src/hololink/operators/cpu_roce_transceiver/cpu_roce_frame_receiver.cpp"

namespace hololink::operators {

struct CpuRoceFrameReceiverTestAccess {
    static void install_device_copy(CpuRoceFrameReceiver& receiver)
    {
        auto copy = std::make_unique<CpuRoceFrameReceiver::DeviceTrailerCopy>();
        copy->initialize();
        receiver.device_trailer_copy_ = std::move(copy);
    }

    static void publish(CpuRoceFrameReceiver& receiver, std::uint32_t slot,
        std::optional<std::uint32_t> immediate_data)
    {
        receiver.publish(slot, immediate_data);
    }

    static const hololink::transport::roce::PayloadMemory& payload_memory(
        const CpuRoceFrameReceiver& receiver)
    {
        return receiver.config_.payload_memory;
    }

    static int dmabuf_fd(const CpuRoceFrameReceiver& receiver)
    {
        return receiver.dmabuf_.fd;
    }
};

} // namespace hololink::operators

namespace {

using hololink::operators::CpuRoceFrameReceiver;
using hololink::operators::CpuRoceFrameReceiverTestAccess;
using hololink::transport::roce::DmaBufMemory;
using hololink::transport::roce::GpuPointerMemory;

void store_be16(std::uint8_t* out, std::uint16_t value)
{
    out[0] = static_cast<std::uint8_t>(value >> 8);
    out[1] = static_cast<std::uint8_t>(value);
}

void write_metadata(std::uint8_t* trailer, std::uint16_t frame_number)
{
    std::memset(trailer, 0, 128);
    store_be16(trailer + 34, frame_number);
}

} // namespace

TEST(CpuRoceFrameReceiverGpu, CopiesADeviceTrailerIntoFrameMetadata)
{
    const CUresult init = cuInit(0);
    if (init == CUDA_ERROR_NO_DEVICE) {
        GTEST_SKIP() << "no CUDA device";
    }
    ASSERT_EQ(init, CUDA_SUCCESS);
    int device_count = 0;
    ASSERT_EQ(cuDeviceGetCount(&device_count), CUDA_SUCCESS);
    if (device_count == 0) {
        GTEST_SKIP() << "no CUDA device";
    }
    CUdevice device = 0;
    ASSERT_EQ(cuDeviceGet(&device, 0), CUDA_SUCCESS);
    CUcontext context = nullptr;
    ASSERT_EQ(cuDevicePrimaryCtxRetain(&context, device), CUDA_SUCCESS);
    CUcontext previous = nullptr;
    ASSERT_EQ(cuCtxGetCurrent(&previous), CUDA_SUCCESS);
    struct ReleaseContext {
        CUdevice device;
        ~ReleaseContext() { cuDevicePrimaryCtxRelease(device); }
    } release_context { device };
    ASSERT_EQ(cuCtxSetCurrent(context), CUDA_SUCCESS);
    // Declared before the allocation and the receiver, so those still run
    // with this context current. Restoring on the way out keeps the next
    // test in this process off a context this test released.
    struct RestoreCurrent {
        CUcontext previous;
        ~RestoreCurrent() { cuCtxSetCurrent(previous); }
    } restore_current { previous };

    constexpr std::size_t kPayload = 32;
    constexpr std::uint32_t kSlots = 4;
    constexpr std::size_t kStride = 256;
    CUdeviceptr buffer = 0;
    ASSERT_EQ(cuMemAlloc(&buffer, kStride * kSlots), CUDA_SUCCESS);
    struct FreeBuffer {
        CUdeviceptr buffer;
        ~FreeBuffer() { cuMemFree(buffer); }
    } free_buffer { buffer };

    std::vector<std::uint8_t> trailer(128);
    write_metadata(trailer.data(), 0x1234);
    ASSERT_EQ(
        cuMemcpyHtoD(buffer + 128, trailer.data(), trailer.size()), CUDA_SUCCESS);

    CpuRoceFrameReceiver::Config config;
    config.ibv_name = "unused";
    config.payload_memory = GpuPointerMemory {
        reinterpret_cast<void*>(buffer), kStride * kSlots
    };
    config.payload_size = kPayload;
    config.slot_count = kSlots;
    config.metadata_size = 128;
    config.cuda_context = context;
    // Destroy the receiver, then the GPU buffer, while this context is current.
    // The trailer copy synchronizes its stream in its destructor.
    CpuRoceFrameReceiver receiver(std::move(config));
    ASSERT_EQ(receiver.stride(), kStride);
    CpuRoceFrameReceiverTestAccess::install_device_copy(receiver);
    CpuRoceFrameReceiverTestAccess::publish(receiver, 0, 0x22u << 12);
    auto frame = receiver.get_frame(std::chrono::milliseconds(0));
    ASSERT_TRUE(frame.has_value());
    ASSERT_TRUE(frame->metadata().has_value());
    EXPECT_EQ(frame->metadata()->frame_number, 0x1234);
    EXPECT_EQ(frame->psn(), 0x22u);
}

TEST(CpuRoceFrameReceiverGpu, ExportsCudaPayloadAsDmaBuf)
{
    const CUresult init = cuInit(0);
    if (init == CUDA_ERROR_NO_DEVICE) {
        GTEST_SKIP() << "no CUDA device";
    }
    ASSERT_EQ(init, CUDA_SUCCESS);
    int device_count = 0;
    ASSERT_EQ(cuDeviceGetCount(&device_count), CUDA_SUCCESS);
    if (device_count == 0) {
        GTEST_SKIP() << "no CUDA device";
    }
    CUdevice device = 0;
    ASSERT_EQ(cuDeviceGet(&device, 0), CUDA_SUCCESS);
    CUcontext context = nullptr;
    ASSERT_EQ(cuDevicePrimaryCtxRetain(&context, device), CUDA_SUCCESS);
    CUcontext previous = nullptr;
    ASSERT_EQ(cuCtxGetCurrent(&previous), CUDA_SUCCESS);
    struct ReleaseContext {
        CUdevice device;
        ~ReleaseContext() { cuDevicePrimaryCtxRelease(device); }
    } release_context { device };
    ASSERT_EQ(cuCtxSetCurrent(context), CUDA_SUCCESS);
    struct RestoreCurrent {
        CUcontext previous;
        ~RestoreCurrent() { cuCtxSetCurrent(previous); }
    } restore_current { previous };

    constexpr std::size_t kBytes = 65536;
    CUdeviceptr buffer = 0;
    ASSERT_EQ(cuMemAlloc(&buffer, kBytes), CUDA_SUCCESS);
    struct FreeBuffer {
        CUdeviceptr buffer;
        ~FreeBuffer() { cuMemFree(buffer); }
    } free_buffer { buffer };

    int probe = -1;
    const CUresult exported = cuMemGetHandleForAddressRange(
        &probe, buffer, kBytes, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
    if (exported != CUDA_SUCCESS || probe < 0) {
        GTEST_SKIP() << "DMA-BUF export is not available";
    }
    ::close(probe);

    CpuRoceFrameReceiver::Config config;
    config.ibv_name = "unused";
    config.cuda_context = context;
    config.cuda_payload = CpuRoceFrameReceiver::Config::CudaPayload {
        reinterpret_cast<void*>(buffer), kBytes, false
    };
    config.payload_size = 128;
    config.slot_count = 4;
    config.metadata_size = 0;
    // Destroy the receiver before the allocation so it closes the owned fd.
    CpuRoceFrameReceiver receiver(std::move(config));
    const auto* dma = std::get_if<DmaBufMemory>(
        &CpuRoceFrameReceiverTestAccess::payload_memory(receiver));
    ASSERT_NE(dma, nullptr);
    EXPECT_EQ(dma->address, reinterpret_cast<void*>(buffer));
    EXPECT_EQ(dma->size, kBytes);
    EXPECT_GE(dma->fd, 0);
    EXPECT_EQ(CpuRoceFrameReceiverTestAccess::dmabuf_fd(receiver), dma->fd);
}
