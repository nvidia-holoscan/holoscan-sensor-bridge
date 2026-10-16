/*
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
 * Hardware-free tests for the module-native PCIe receive path
 * (hololink_module/host/pcie: geometry, ring layout, PcieReceiver) and, when the Holoscan
 * operators are built, the PcieReceiverOp smoke test. The receiver
 * is driven against a fake PcieDeviceAccessInterface whose wait_frame() is released by the
 * test, so the mapping / MSI-X routing / baseline sequence of start(), the
 * PcieReceiverAttachmentV1 values the hsb_pcie module's data channel programs the endpoint
 * from, the slot accounting and the frame-number extension are asserted with no endpoint, no
 * driver and no CUDA context. The receiver touches no register, so no HololinkInterfaceV1
 * appears here; the channel's register sequence is asserted module-side in
 * tests/hololink_module_hsb_pcie_test.cpp over the loopback endpoint.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <hololink/module/hsb_pcie/pcie_data_channel.hpp>
#include <hololink/module/pcie/pcie_device_access.hpp>
#include <hololink/module/pcie/pcie_receiver.hpp>
#include <hololink/module/pcie/pcie_rx_geometry.hpp>

#ifdef HOLOLINK_MODULE_PCIE_RECEIVER_OP_TEST
#include <holoscan/holoscan.hpp>

#include <hololink/module/enumeration_metadata.hpp>
#include <hololink/module/operators/pcie_receiver_op.hpp>
#endif

namespace {

using namespace hololink::module::pcie;
using hololink::module::EnumerationMetadata;
using hololink::module::hsb_pcie::PcieReceiverAttachmentV1;

constexpr uint64_t TEST_IOVA = 0x2800e00000ull;
constexpr uint32_t TEST_REQUESTER_ID = 0x0600;
constexpr uint64_t TEST_MSIX_ADDR = 0xFEE00000ull;
constexpr uint32_t TEST_MSIX_DATA = 0x4021;
constexpr uint32_t TEST_MSIX_VECTORS = 8;

// ---------------------------------------------------------------------------
// Fakes
// ---------------------------------------------------------------------------

/// Device access whose completion counter the test advances with complete(); msix_info
/// answers the vector count start() checks and msix_ctrl records the enable / disable the
/// receiver issues (the kernel-owned MSI-X path). Each instance names a distinct device node.
class FakePcieDeviceAccess : public PcieDeviceAccessInterface {
public:
    FakePcieDeviceAccess()
    {
        static std::atomic<unsigned> next_minor { 100 };
        description_ = "/dev/hololink_pcie" + std::to_string(next_minor++);
    }

    void complete(uint32_t frames = 1)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            irq_count_ += frames;
        }
        irq_available_.notify_all();
    }

    PcieDeviceIdentity identity() override
    {
        PcieDeviceIdentity id;
        id.requester_id = TEST_REQUESTER_ID;
        id.num_msix_vectors = TEST_MSIX_VECTORS;
        id.bdf = "0000:06:00.0";
        id.from_ioctl = true;
        identity_calls++;
        return id;
    }
    PcieStatus import_dmabuf(CUdeviceptr buffer, size_t size, DmaMapping& mapping) override
    {
        imported_buffer = buffer;
        imported_size = size;
        mapping.iova = iova;
        mapping.handle = 42;
        mapping.size = size;
        imports++;
        return PcieStatus::SUCCESS;
    }
    PcieStatus release(uint64_t handle) override
    {
        released_handles.push_back(handle);
        return PcieStatus::SUCCESS;
    }
    PcieStatus msix_info(uint32_t vector, bool reset_count, MsixVectorInfo& info) override
    {
        msix_info_vector = vector;
        msix_info_reset = reset_count;
        info.msg_addr = TEST_MSIX_ADDR;
        info.msg_data = TEST_MSIX_DATA;
        info.num_vectors = TEST_MSIX_VECTORS;
        info.enabled = msix_enabled;
        std::lock_guard<std::mutex> lock(mutex_);
        if (reset_count) {
            irq_count_ = 0;
        }
        info.irq_count = irq_count_;
        return PcieStatus::SUCCESS;
    }
    PcieStatus msix_ctrl(uint32_t vector, bool enable, uint32_t& refs) override
    {
        ctrl_calls.emplace_back(vector, enable);
        if (msix_ctrl_status != PcieStatus::SUCCESS) {
            return msix_ctrl_status;
        }
        if (enable) {
            enabled_vectors.insert(vector);
        } else {
            enabled_vectors.erase(vector);
        }
        refs = enabled_vectors.count(vector) ? 1u + extra_holders : 0u;
        return PcieStatus::SUCCESS;
    }
    PcieStatus wait_frame(uint32_t vector, uint32_t timeout_ms, uint32_t last_count,
        uint32_t& count) override
    {
        waited_vector = vector;
        if (wait_frame_status != PcieStatus::SUCCESS) {
            // The MSIX_WAIT ioctl itself failing (the node went away, a driver without it).
            count = last_count;
            return wait_frame_status;
        }
        std::unique_lock<std::mutex> lock(mutex_);
        irq_available_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
            [&] { return irq_count_ != last_count; });
        count = irq_count_;
        return count != last_count ? PcieStatus::SUCCESS : PcieStatus::TIMEOUT;
    }
    std::string description() const override { return description_; }

    uint64_t iova { TEST_IOVA };
    bool msix_enabled { true };
    CUdeviceptr imported_buffer { 0 };
    size_t imported_size { 0 };
    unsigned imports { 0 };
    unsigned identity_calls { 0 };
    std::vector<uint64_t> released_handles;
    uint32_t msix_info_vector { 0xFFFFFFFFu };
    bool msix_info_reset { false };
    uint32_t waited_vector { 0xFFFFFFFFu };
    /// What wait_frame() returns instead of waiting when not SUCCESS. Set before the thread
    /// that will call it starts.
    PcieStatus wait_frame_status { PcieStatus::SUCCESS };
    /// (vector, enable) in call order.
    std::vector<std::pair<uint32_t, bool>> ctrl_calls;
    std::set<uint32_t> enabled_vectors;
    /// Forced result of msix_ctrl, e.g. UNSUPPORTED for a driver without the ioctl.
    PcieStatus msix_ctrl_status { PcieStatus::SUCCESS };
    /// Other fds the fake reports as holding a vector, added to refs.
    uint32_t extra_holders { 0 };

private:
    std::string description_;
    std::mutex mutex_;
    std::condition_variable irq_available_;
    uint32_t irq_count_ { 0 };
};

/// A tiny RAW10 frame so the ring fits in a small buffer: 64x4, 320 bytes.
RxFrameGeometry small_geometry()
{
    RxFrameGeometry geometry;
    geometry.width = 64;
    geometry.height = 4;
    geometry.stride_pixels = 64;
    return geometry;
}

RxRingConfig ring_config(uint32_t sensor, uint32_t num_buffers)
{
    RxRingConfig config;
    config.sensor = sensor;
    config.num_buffers = num_buffers;
    return config;
}

} // namespace

// ---------------------------------------------------------------------------
// Geometry parity with the legacy path (tests/hololink_dma_rx_ring_test.cpp).
// ---------------------------------------------------------------------------

TEST(PcieRxGeometry, Raw10ParityWithLegacyPath)
{
    // The validated IGX RAW10 configuration: 1920x1080, 8-byte line alignment, and the
    // 640-byte leading prefix left once IMX258 PDAF auxiliary output is disabled.
    RxFrameGeometry geometry;
    geometry.width = 1920;
    geometry.height = 1080;
    geometry.stride_pixels = 1920;
    geometry.line_alignment = 8;
    geometry.raw_start_byte = 640;
    EXPECT_EQ(geometry.validate(), nullptr);
    EXPECT_EQ(geometry.dma_line_bytes(), 2400u);
    EXPECT_EQ(geometry.frame_dma_bytes(), 640u + 2400u * 1080u);
    EXPECT_EQ(geometry.metadata_offset(), align_up(640u + 2400u * 1080u, 128));
}

TEST(PcieRxGeometry, RingLayoutFollowsLegacyArithmetic)
{
    const RxRingLayout layout = RxRingLayout::compute(small_geometry(), 3);
    EXPECT_EQ(layout.window_bytes, 320u);
    EXPECT_EQ(layout.metadata_offset, 384u); // align_up(320, 128)
    EXPECT_EQ(layout.slot_stride, 4096u); // align_up(384 + 128, 4096)
    EXPECT_EQ(layout.buf_inc_pages, 32u); // 4096 / 128
    EXPECT_EQ(layout.required_bytes(), 3u * 4096u);
    EXPECT_EQ(layout.slot_offset(2), 2u * 4096u);

    // A single-buffer ring rewrites the same slot: no increment.
    EXPECT_EQ(RxRingLayout::compute(small_geometry(), 1).buf_inc_pages, 0u);

    // A window larger than the frame is honoured.
    const RxRingLayout wide = RxRingLayout::compute(small_geometry(), 2, 5000);
    EXPECT_EQ(wide.window_bytes, 5000u);
    EXPECT_EQ(wide.metadata_offset, 5120u);
    EXPECT_EQ(wide.slot_stride, 8192u);
}

TEST(PcieRxGeometry, RejectsBadGeometry)
{
    RxFrameGeometry geometry = small_geometry();
    geometry.width = 66;
    EXPECT_NE(geometry.validate(), nullptr);
    geometry = small_geometry();
    geometry.raw_start_byte = 4;
    EXPECT_NE(geometry.validate(), nullptr);
    geometry = small_geometry();
    geometry.line_alignment = 12;
    EXPECT_NE(geometry.validate(), nullptr);
}

TEST(PcieRxGeometry, RingLayoutLimitsFollowTheEndpointRegisters)
{
    const RxFrameGeometry geometry = small_geometry();
    EXPECT_EQ(RxRingLayout::compute(geometry, 3).validate(geometry), nullptr);
    // BUF_PTR_RANGE is 12 bits: [1, 4096].
    EXPECT_NE(RxRingLayout::compute(geometry, 0).validate(geometry), nullptr);
    EXPECT_EQ(RxRingLayout::compute(geometry, RX_MAX_BUFFERS).validate(geometry), nullptr);
    EXPECT_NE(RxRingLayout::compute(geometry, RX_MAX_BUFFERS + 1).validate(geometry), nullptr);
    // A window smaller than the frame cannot hold it.
    EXPECT_NE(RxRingLayout::compute(geometry, 2, 64).validate(geometry), nullptr);
    // A bad geometry is reported through the layout too.
    RxFrameGeometry bad = geometry;
    bad.width = 66;
    EXPECT_NE(RxRingLayout::compute(bad, 2).validate(bad), nullptr);
}

TEST(PcieRxGeometry, RejectsBuffersTheAddressGeneratorCannotUse)
{
    const RxRingLayout layout = RxRingLayout::compute(small_geometry(), 3);
    EXPECT_NE(layout.validate_buffer(TEST_IOVA, layout.required_bytes() - 1), nullptr);
    EXPECT_NE(layout.validate_buffer(TEST_IOVA + 64, layout.required_bytes()), nullptr);
    EXPECT_NE(layout.validate_buffer(0x1'0000'0000ull - 4096, layout.required_bytes()), nullptr);
    EXPECT_EQ(layout.validate_buffer(TEST_IOVA, layout.required_bytes()), nullptr);
}

// ---------------------------------------------------------------------------
// Metadata decoding: both in-field layouts.
// ---------------------------------------------------------------------------

TEST(PcieRxMetadata, DecodesCurrentLayout)
{
    std::vector<uint8_t> raw(RX_METADATA_BYTES, 0);
    raw[0x03] = 0x01; // flags = 1 (unsync)
    raw[0x1E] = 0x12; // valid_bytes = 0x1234
    raw[0x1F] = 0x34;
    raw[0x22] = 0xAB; // frame_number = 0xABCD
    raw[0x23] = 0xCD;
    const RxMetadata metadata = parse_rx_metadata(raw.data(), raw.size(), 2, 0xA5);
    EXPECT_TRUE(metadata.parsed);
    EXPECT_EQ(metadata.layout, RxMetadataLayout::CURRENT);
    EXPECT_EQ(metadata.slot, 2u);
    EXPECT_EQ(metadata.flags, 1u);
    EXPECT_TRUE(metadata.unsync());
    EXPECT_EQ(metadata.valid_bytes, 0x1234u);
    EXPECT_EQ(metadata.frame_number, 0xABCDu);
}

TEST(PcieRxMetadata, DecodesLegacyWrImmLayout)
{
    std::vector<uint8_t> raw(RX_METADATA_BYTES, 0);
    raw[0x53] = 0x02; // flags = 2 (bpw_only), and non-zero past 0x30 selects the legacy layout
    raw[0x6E] = 0x56; // valid_bytes = 0x5678
    raw[0x6F] = 0x78;
    raw[0x72] = 0x00; // frame_number = 0x0042
    raw[0x73] = 0x42;
    const RxMetadata metadata = parse_rx_metadata(raw.data(), raw.size(), 1, 0);
    EXPECT_TRUE(metadata.parsed);
    EXPECT_EQ(metadata.layout, RxMetadataLayout::LEGACY);
    EXPECT_EQ(metadata.flags, 2u);
    EXPECT_TRUE(metadata.bpw_only());
    EXPECT_EQ(metadata.valid_bytes, 0x5678u);
    EXPECT_EQ(metadata.frame_number, 0x42u);
}

TEST(PcieRxMetadata, UntouchedBlockIsNotParsed)
{
    std::vector<uint8_t> raw(RX_METADATA_BYTES, 0xA5);
    const RxMetadata metadata = parse_rx_metadata(raw.data(), raw.size(), 0, 0xA5);
    EXPECT_FALSE(metadata.parsed);
    EXPECT_EQ(metadata.layout, RxMetadataLayout::NONE);
    EXPECT_FALSE(parse_rx_metadata(raw.data(), 16, 0, 0).parsed);
}

// ---------------------------------------------------------------------------
// PcieReceiver start / stop / close and the attachment values.
// ---------------------------------------------------------------------------

TEST(PcieReceiver, StartMapsReadsMsixRoutingAndBaselines)
{
    auto access = std::make_shared<FakePcieDeviceAccess>();
    const CUdeviceptr buffer = 0x7f0000000000ull;
    PcieReceiver receiver(access, small_geometry(), ring_config(1, 3), buffer, 3 * 4096);
    // The requester id is read from the device identity at construction.
    EXPECT_EQ(access->identity_calls, 1u);
    EXPECT_EQ(receiver.requester_id(), TEST_REQUESTER_ID);
    EXPECT_EQ(receiver.iova(), 0u) << "no mapping before start()";
    EXPECT_FALSE(receiver.vector_enabled());
    EXPECT_TRUE(access->ctrl_calls.empty());

    ASSERT_TRUE(receiver.start());
    EXPECT_TRUE(receiver.running());
    EXPECT_EQ(access->imports, 1u);
    EXPECT_EQ(access->imported_buffer, buffer);
    EXPECT_EQ(access->imported_size, 3u * 4096u);
    EXPECT_EQ(receiver.frame_buffer(), buffer);
    // MSI-X routing was read for this sensor's vector with the counter reset, and the
    // baseline wait_frame() polled the same vector.
    EXPECT_EQ(access->msix_info_vector, 1u);
    EXPECT_TRUE(access->msix_info_reset);
    EXPECT_EQ(access->waited_vector, 1u);
    EXPECT_EQ(receiver.msix_vectors(), TEST_MSIX_VECTORS);
    // The receiver, not the channel, turned the vector on -- once, after the baseline.
    EXPECT_TRUE(receiver.vector_enabled());
    ASSERT_EQ(access->ctrl_calls.size(), 1u);
    EXPECT_EQ(access->ctrl_calls[0], std::make_pair(1u, true));
    EXPECT_EQ(access->enabled_vectors, (std::set<uint32_t> { 1 }));

    // What the hsb_pcie channel programs the endpoint from (PcieReceiverAttachmentV1).
    PcieReceiverAttachmentV1& attachment = receiver;
    EXPECT_EQ(attachment.iova(), TEST_IOVA);
    EXPECT_EQ(attachment.window_bytes(), 320u);
    EXPECT_EQ(attachment.buf_inc_pages(), 32u);
    EXPECT_EQ(attachment.num_buffers(), 3u);
    EXPECT_EQ(attachment.requester_id(), TEST_REQUESTER_ID);
    EXPECT_EQ(attachment.vector(), 1u);
    EXPECT_FALSE(attachment.disable_metadata());

    receiver.close();
    EXPECT_FALSE(receiver.running());
    // Vector off before the mapping went, and exactly once.
    EXPECT_FALSE(receiver.vector_enabled());
    ASSERT_EQ(access->ctrl_calls.size(), 2u);
    EXPECT_EQ(access->ctrl_calls[1], std::make_pair(1u, false));
    EXPECT_TRUE(access->enabled_vectors.empty());
    EXPECT_EQ(access->released_handles, (std::vector<uint64_t> { 42 }));
    receiver.close(); // idempotent
    EXPECT_EQ(access->released_handles.size(), 1u);
    EXPECT_EQ(access->ctrl_calls.size(), 2u) << "close() twice disables once";
}

TEST(PcieReceiver, StartFailsCleanlyWhenTheDriverLacksMsixCtrl)
{
    // A driver predating HL_PCIE_IOC_MSIX_CTRL leaves every vector masked forever, and this
    // receiver no longer programs the table itself, so it must refuse rather than wait on a
    // vector that can never fire. The mapping it made is released again.
    auto access = std::make_shared<FakePcieDeviceAccess>();
    access->msix_ctrl_status = PcieStatus::UNSUPPORTED;
    PcieReceiver receiver(access, small_geometry(), ring_config(1, 3), 0x7f0000000000ull, 3 * 4096);
    EXPECT_FALSE(receiver.start());
    EXPECT_FALSE(receiver.running());
    EXPECT_FALSE(receiver.vector_enabled());
    EXPECT_EQ(access->imports, 1u);
    EXPECT_EQ(access->released_handles, (std::vector<uint64_t> { 42 }));
    ASSERT_EQ(access->ctrl_calls.size(), 1u) << "no disable for an enable that failed";
    EXPECT_EQ(access->ctrl_calls[0], std::make_pair(1u, true));
}

TEST(PcieReceiver, VectorSharedWithAnotherFdIsStillReleasedOnClose)
{
    // The driver refcounts holders across fds; the receiver only ever drops its own hold.
    auto access = std::make_shared<FakePcieDeviceAccess>();
    access->extra_holders = 1;
    PcieReceiver receiver(access, small_geometry(), ring_config(0, 3), 0x7f0000000000ull, 3 * 4096);
    ASSERT_TRUE(receiver.start());
    EXPECT_TRUE(receiver.vector_enabled());
    receiver.close();
    ASSERT_EQ(access->ctrl_calls.size(), 2u);
    EXPECT_EQ(access->ctrl_calls[0], std::make_pair(0u, true));
    EXPECT_EQ(access->ctrl_calls[1], std::make_pair(0u, false));
}

TEST(PcieReceiver, DisableMetadataIsRejectedAtStart)
{
    // The monitor finds frames through the metadata blocks, so a ring without them could
    // never deliver one: start() refuses rather than attach a receiver that only misses.
    auto access = std::make_shared<FakePcieDeviceAccess>();
    RxRingConfig config = ring_config(0, 1);
    config.disable_metadata = true;
    PcieReceiver receiver(access, small_geometry(), config, 0x1000, 4096);
    EXPECT_FALSE(receiver.start());
    EXPECT_TRUE(access->released_handles.empty()) << "nothing was mapped";
    EXPECT_EQ(receiver.buf_inc_pages(), 0u) << "a single-slot ring has no increment";
    EXPECT_EQ(receiver.num_buffers(), 1u);
}

TEST(PcieReceiver, StartFailsCleanlyWhenTheBufferOrLayoutIsRejected)
{
    // Too small for three slots: the mapping is made, rejected and released again.
    auto access = std::make_shared<FakePcieDeviceAccess>();
    PcieReceiver receiver(access, small_geometry(), ring_config(0, 3), 0x1000, 4096);
    EXPECT_FALSE(receiver.start());
    EXPECT_FALSE(receiver.running());
    EXPECT_EQ(access->released_handles, (std::vector<uint64_t> { 42 }));

    // A misaligned IOVA is rejected the same way.
    auto misaligned = std::make_shared<FakePcieDeviceAccess>();
    misaligned->iova = TEST_IOVA + 64;
    PcieReceiver bad_iova(misaligned, small_geometry(), ring_config(0, 2), 0x1000, 1 << 20);
    EXPECT_FALSE(bad_iova.start());
    EXPECT_EQ(misaligned->released_handles, (std::vector<uint64_t> { 42 }));

    // Zero buffers never reaches the device.
    auto untouched = std::make_shared<FakePcieDeviceAccess>();
    PcieReceiver zero(untouched, small_geometry(), ring_config(0, 0), 0x1000, 1 << 20);
    EXPECT_FALSE(zero.start());
    EXPECT_EQ(untouched->imports, 0u);
}

TEST(PcieReceiver, StartRejectsAVectorTheDeviceCannotServe)
{
    // Sensor index beyond the allocated vectors.
    auto access = std::make_shared<FakePcieDeviceAccess>();
    PcieReceiver too_many(access, small_geometry(), ring_config(TEST_MSIX_VECTORS, 2), 0x1000, 1 << 20);
    EXPECT_FALSE(too_many.start());
    EXPECT_FALSE(too_many.running());
    EXPECT_EQ(access->released_handles, (std::vector<uint64_t> { 42 }));

    // MSI-X disabled on the device.
    auto disabled = std::make_shared<FakePcieDeviceAccess>();
    disabled->msix_enabled = false;
    PcieReceiver receiver(disabled, small_geometry(), ring_config(0, 2), 0x1000, 1 << 20);
    EXPECT_FALSE(receiver.start());
    EXPECT_EQ(disabled->released_handles, (std::vector<uint64_t> { 42 }));
}

TEST(PcieReceiver, GetNextFrameTimesOutWhenNothingCompletes)
{
    auto access = std::make_shared<FakePcieDeviceAccess>();
    PcieReceiver receiver(access, small_geometry(), ring_config(0, 2), 0x1000, 1 << 20);
    ASSERT_TRUE(receiver.start());
    PcieFrameInfo info;
    EXPECT_FALSE(receiver.get_next_frame(10, info));
    EXPECT_FALSE(receiver.frames_ready());
}

TEST(PcieReceiver, CloseWakesAParkedGetNextFrame)
{
    auto access = std::make_shared<FakePcieDeviceAccess>();
    PcieReceiver receiver(access, small_geometry(), ring_config(0, 2), 0x1000, 1 << 20);
    ASSERT_TRUE(receiver.start());

    std::atomic<bool> got_frame { true };
    std::chrono::steady_clock::duration waited {};
    std::thread waiter([&] {
        PcieFrameInfo info;
        const auto started = std::chrono::steady_clock::now();
        got_frame = receiver.get_next_frame(5000, info);
        waited = std::chrono::steady_clock::now() - started;
    });
    // Let the waiter park on the condition variable, then close underneath it.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    receiver.close();
    waiter.join();

    EXPECT_FALSE(got_frame);
    EXPECT_LT(waited, std::chrono::milliseconds(2000)) << "close() must wake the waiter, "
                                                          "not leave it to the 5 s timeout";
}

// ---------------------------------------------------------------------------
// PcieReceiver frame accounting in blocking_monitor().
//
// Finding the written slot from its metadata, dropped-frame counting, and extension of the
// 16-bit wire frame number are the receiver's only real logic. Two seams make this testable with no endpoint and no
// CUDA context: the fake device access for the completion signal, and read_slot_metadata()
// for what the endpoint "wrote".
// ---------------------------------------------------------------------------

namespace {

/// Models what the endpoint leaves in each slot's metadata block: the monitor discovers
/// the written slot by scanning, so the test decides which slot holds which frame.
class ScriptedReceiver : public PcieReceiver {
public:
    using PcieReceiver::PcieReceiver;

    std::vector<uint32_t> slots_read;
    bool legacy_layout { false };
    /// Runs on the monitor thread inside read_slot_metadata(), before the block is built.
    std::function<void(uint32_t)> on_read;

    /// "The endpoint wrote frame `frame_number` into `slot`."
    void write_slot(uint32_t slot, uint16_t frame_number)
    {
        std::lock_guard<std::mutex> lock(slots_mutex_);
        slot_frames_[slot] = frame_number;
    }

protected:
    void clear_metadata_blocks() override
    {
        std::lock_guard<std::mutex> lock(slots_mutex_);
        slot_frames_.clear();
    }

    void read_slot_metadata(uint32_t slot, uint8_t* block) override
    {
        slots_read.push_back(slot);
        if (on_read) {
            on_read(slot);
        }
        std::memset(block, 0, RX_METADATA_BYTES);
        std::optional<uint16_t> frame_number;
        {
            std::lock_guard<std::mutex> lock(slots_mutex_);
            auto it = slot_frames_.find(slot);
            if (it != slot_frames_.end()) {
                frame_number = it->second;
            }
        }
        if (!frame_number) {
            return; // untouched slot: the block stays all-zero and does not parse
        }
        // Build the block the endpoint would have written; the receiver decodes it through
        // the real parser so the test exercises that too.
        if (legacy_layout) {
            block[0x72] = static_cast<uint8_t>(*frame_number >> 8);
            block[0x73] = static_cast<uint8_t>(*frame_number & 0xFF);
            block[0x6F] = 0x40; // non-zero valid_bytes
        } else {
            block[0x22] = static_cast<uint8_t>(*frame_number >> 8);
            block[0x23] = static_cast<uint8_t>(*frame_number & 0xFF);
            block[0x1F] = 0x40; // non-zero valid_bytes
        }
    }

private:
    std::mutex slots_mutex_;
    std::map<uint32_t, uint16_t> slot_frames_;
};

/// One completion batch: the slots the endpoint wrote (slot, frame_number), then that many
/// interrupts.
using Batch = std::vector<std::pair<uint32_t, uint16_t>>;

/// Ends the monitor on every exit path. A fatal ASSERT_* returns from the test body before
/// its explicit close()/join(), and a joinable std::thread terminates the process in its
/// destructor. close() is idempotent, so the explicit calls that follow stay as they are.
class MonitorJoiner {
public:
    MonitorJoiner(PcieReceiver& receiver, std::thread& monitor)
        : receiver_(receiver)
        , monitor_(monitor)
    {
    }
    ~MonitorJoiner()
    {
        receiver_.close();
        if (monitor_.joinable()) {
            monitor_.join();
        }
    }

private:
    PcieReceiver& receiver_;
    std::thread& monitor_;
};

/// Drive batches through the monitor and collect what came out.
std::vector<PcieFrameInfo> run_monitor(ScriptedReceiver& receiver,
    FakePcieDeviceAccess& access, const std::vector<Batch>& batches)
{
    std::vector<PcieFrameInfo> received;
    std::thread monitor([&receiver] { receiver.blocking_monitor(); });
    for (const Batch& batch : batches) {
        for (const auto& [slot, frame_number] : batch) {
            receiver.write_slot(slot, frame_number);
        }
        access.complete(static_cast<uint32_t>(batch.size()));
        PcieFrameInfo info;
        if (receiver.get_next_frame(1000, info)) {
            received.push_back(info);
        }
    }
    receiver.close();
    monitor.join();
    return received;
}

} // namespace

TEST(PcieReceiver, StopKeepsTheMappingAndCloseWaitsForTheMonitor)
{
    auto access = std::make_shared<FakePcieDeviceAccess>();
    ScriptedReceiver receiver(access, small_geometry(), ring_config(0, 2), 0x1000, 1 << 20);
    ASSERT_TRUE(receiver.start());

    // The monitor reports when it is inside read_slot_metadata() and then lingers there, so
    // close() below provably races a frame that is still being decoded -- the host-staged
    // path reads that metadata straight out of the mapping close() releases.
    std::mutex mutex;
    std::condition_variable reading_cv;
    bool reading = false;
    std::atomic<bool> released_during_read { false };
    receiver.on_read = [&](uint32_t) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            reading = true;
        }
        reading_cv.notify_all();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        released_during_read = !access->released_handles.empty();
    };
    std::thread monitor([&receiver] { receiver.blocking_monitor(); });
    MonitorJoiner joiner(receiver, monitor);
    receiver.write_slot(0, 1);
    access->complete(1);
    {
        std::unique_lock<std::mutex> lock(mutex);
        ASSERT_TRUE(reading_cv.wait_for(lock, std::chrono::seconds(2), [&] { return reading; }));
    }

    // stop(): no longer running, mapping still held.
    receiver.stop();
    EXPECT_FALSE(receiver.running());
    EXPECT_TRUE(access->released_handles.empty());

    // close(): returns only after the monitor has left blocking_monitor(), then releases.
    receiver.close();
    EXPECT_EQ(access->released_handles, (std::vector<uint64_t> { 42 }));
    monitor.join();
    EXPECT_FALSE(released_during_read) << "the mapping was released while the monitor was "
                                          "still reading slot metadata from it";
    ASSERT_FALSE(receiver.slots_read.empty());
    EXPECT_EQ(receiver.slots_read.front(), 0u);
}

TEST(PcieReceiverAccounting, ReportsTheSlotTheEndpointWroteAndWrapsTheRing)
{
    auto access = std::make_shared<FakePcieDeviceAccess>();
    ScriptedReceiver receiver(access, small_geometry(), ring_config(0, 3), 0x10000, 1 << 20);
    ASSERT_TRUE(receiver.start());

    // Ring of 3, endpoint pointer starting at 0: it fills 0,1,2 and wraps to 0. Each
    // completion reports the slot whose metadata block is new.
    auto received = run_monitor(receiver, *access,
        { { { 0, 10 } }, { { 1, 11 } }, { { 2, 12 } }, { { 0, 13 } } });
    ASSERT_EQ(received.size(), 4u);
    EXPECT_EQ(received[0].page_number, 0u);
    EXPECT_EQ(received[1].page_number, 1u);
    EXPECT_EQ(received[2].page_number, 2u);
    EXPECT_EQ(received[3].page_number, 0u);

    // Slot addresses follow the layout, and the metadata block sits inside the slot.
    EXPECT_EQ(received[1].frame_memory, 0x10000u + 4096u);
    EXPECT_EQ(received[1].metadata_memory, 0x10000u + 4096u + 384u);
    EXPECT_EQ(received[2].frame_memory, 0x10000u + 2u * 4096u);

    // Frame numbers, counters, layout and payload all published.
    EXPECT_EQ(received[0].frame_number, 10u);
    EXPECT_EQ(received[3].frame_number, 13u);
    EXPECT_EQ(received[3].frames, 4u);
    EXPECT_EQ(received[3].dropped, 0u);
    EXPECT_EQ(received[3].bytes_written, 0x40u);
    EXPECT_EQ(received[3].metadata_layout, RxMetadataLayout::CURRENT);
    EXPECT_EQ(received[3].raw_metadata[0x23], 13u);
    EXPECT_NE(received[3].received_s, 0u);
}

TEST(PcieReceiverAccounting, FindsTheSlotRegardlessOfTheEndpointsStartingPointer)
{
    // The endpoint's write pointer survives re-programming: on hardware a run after a
    // 30-frame player run started writing at slot 3 while a pointer-tracking receiver read
    // slot 0 and reported empty frames. The scan must follow the data, not a counter.
    auto access = std::make_shared<FakePcieDeviceAccess>();
    ScriptedReceiver receiver(access, small_geometry(), ring_config(0, 4), 0, 1 << 20);
    ASSERT_TRUE(receiver.start());

    auto received = run_monitor(receiver, *access,
        { { { 3, 100 } }, { { 0, 101 } }, { { 1, 102 } }, { { 2, 103 } }, { { 3, 104 } } });
    ASSERT_EQ(received.size(), 5u);
    EXPECT_EQ(received[0].page_number, 3u);
    EXPECT_EQ(received[1].page_number, 0u);
    EXPECT_EQ(received[2].page_number, 1u);
    EXPECT_EQ(received[3].page_number, 2u);
    EXPECT_EQ(received[4].page_number, 3u); // same slot as frame 100, new frame number
    EXPECT_EQ(received[4].frame_number, 104u);
    EXPECT_EQ(received[4].dropped, 0u);
}

TEST(PcieReceiverAccounting, StaleBlocksFromAPreviousRunAreIgnored)
{
    // A slot whose block was already delivered is not reported again on a later interrupt
    // unless the endpoint rewrote it with a newer frame.
    auto access = std::make_shared<FakePcieDeviceAccess>();
    ScriptedReceiver receiver(access, small_geometry(), ring_config(0, 2), 0, 1 << 20);
    ASSERT_TRUE(receiver.start());

    std::thread monitor([&receiver] { receiver.blocking_monitor(); });
    MonitorJoiner joiner(receiver, monitor);
    PcieFrameInfo info;
    receiver.write_slot(0, 5);
    access->complete(1);
    ASSERT_TRUE(receiver.get_next_frame(1000, info));
    EXPECT_EQ(info.page_number, 0u);
    // Interrupt with nothing new written: no frame is delivered.
    access->complete(1);
    EXPECT_FALSE(receiver.get_next_frame(300, info));
    // The endpoint writes slot 1: delivered.
    receiver.write_slot(1, 6);
    access->complete(1);
    ASSERT_TRUE(receiver.get_next_frame(1000, info));
    EXPECT_EQ(info.page_number, 1u);
    EXPECT_EQ(info.frame_number, 6u);
    receiver.close();
    monitor.join();
}

TEST(PcieReceiverAccounting, ResynchronizesWhenTheFrameCounterRestarts)
{
    // On hardware a windowed IMX258 run delivered one frame and then reported completions
    // with "no new complete metadata block" indefinitely: every completed frame parsed fine
    // but was rejected as older than the last delivered number. Whatever put that number
    // ahead (a counter restart, or a first pick from a block that was not this run's), the
    // receiver must not stay wedged behind it. After RESYNC_AFTER_MISSES such completions it
    // drops the reference and takes the newest frame present; blocks already delivered stay
    // delivered.
    auto access = std::make_shared<FakePcieDeviceAccess>();
    ScriptedReceiver receiver(access, small_geometry(), ring_config(0, 2), 0, 1 << 20);
    ASSERT_TRUE(receiver.start());

    std::thread monitor([&receiver] { receiver.blocking_monitor(); });
    MonitorJoiner joiner(receiver, monitor);
    PcieFrameInfo info;
    receiver.write_slot(0, 30000);
    access->complete(1);
    ASSERT_TRUE(receiver.get_next_frame(1000, info));
    EXPECT_EQ(info.frame_number, 30000u);
    EXPECT_EQ(receiver.resyncs(), 0u);

    // The endpoint now writes frame 5 into slot 1: 29995 behind 30000 in 16-bit arithmetic (a
    // distance >= 0x8000, i.e. older), so it is rejected --
    // until the miss streak reaches the resync threshold, at which point it is delivered
    // from the same completion.
    receiver.write_slot(1, 5);
    unsigned completions = 0;
    bool delivered = false;
    for (; completions < 12 && !delivered; completions++) {
        access->complete(1);
        delivered = receiver.get_next_frame(400, info);
    }
    ASSERT_TRUE(delivered) << "never resynchronized";
    EXPECT_EQ(receiver.resyncs(), 1u);
    EXPECT_GE(completions, 8u) << "resynchronized too eagerly";
    EXPECT_EQ(info.page_number, 1u);
    EXPECT_EQ(info.frame_number, 5u) << "the extended counter restarts from the wire value";

    // Slot 0 still holds the already-delivered frame 30000: not handed out again, even
    // though the reference was dropped.
    access->complete(1);
    EXPECT_FALSE(receiver.get_next_frame(300, info));
    // Normal progress resumes from the new reference.
    receiver.write_slot(0, 6);
    access->complete(1);
    ASSERT_TRUE(receiver.get_next_frame(1000, info));
    EXPECT_EQ(info.frame_number, 6u);
    EXPECT_EQ(receiver.resyncs(), 1u);
    receiver.close();
    monitor.join();
}

TEST(PcieReceiverAccounting, CountsDroppedFramesFromABatchedCompletion)
{
    auto access = std::make_shared<FakePcieDeviceAccess>();
    ScriptedReceiver receiver(access, small_geometry(), ring_config(0, 4), 0, 1 << 20);
    ASSERT_TRUE(receiver.start());

    // One frame, then three at once, then one: the batch of three reports its newest frame
    // (slot 3, frame 4) and two dropped frames; the next frame lands in slot 0.
    auto received = run_monitor(receiver, *access,
        { { { 0, 1 } }, { { 1, 2 }, { 2, 3 }, { 3, 4 } }, { { 0, 5 } } });
    ASSERT_EQ(received.size(), 3u);
    EXPECT_EQ(received[0].page_number, 0u);
    EXPECT_EQ(received[0].dropped, 0u);
    EXPECT_EQ(received[1].page_number, 3u);
    EXPECT_EQ(received[1].frame_number, 4u);
    EXPECT_EQ(received[1].dropped, 2u);
    EXPECT_EQ(received[1].frames, 4u);
    EXPECT_EQ(received[2].page_number, 0u);
    EXPECT_EQ(received[2].dropped, 2u);
    EXPECT_EQ(received[2].frames, 5u);
}

TEST(PcieReceiverAccounting, ExtendsTheFrameNumberAcrossWraparound)
{
    auto access = std::make_shared<FakePcieDeviceAccess>();
    ScriptedReceiver receiver(access, small_geometry(), ring_config(0, 2), 0, 1 << 20);
    ASSERT_TRUE(receiver.start());

    auto received = run_monitor(receiver, *access,
        { { { 0, 0xFFFE } }, { { 1, 0xFFFF } }, { { 0, 0x0000 } }, { { 1, 0x0001 } } });
    ASSERT_EQ(received.size(), 4u);
    EXPECT_EQ(received[0].frame_number, 0xFFFEu);
    EXPECT_EQ(received[1].frame_number, 0xFFFFu);
    EXPECT_EQ(received[2].frame_number, 0x10000u);
    EXPECT_EQ(received[3].frame_number, 0x10001u);
}

TEST(PcieReceiverAccounting, DecodesTheLegacyMetadataLayoutToo)
{
    auto access = std::make_shared<FakePcieDeviceAccess>();
    ScriptedReceiver receiver(access, small_geometry(), ring_config(0, 2), 0, 1 << 20);
    receiver.legacy_layout = true;
    ASSERT_TRUE(receiver.start());

    auto received = run_monitor(receiver, *access, { { { 0, 7 } }, { { 1, 8 } } });
    ASSERT_EQ(received.size(), 2u);
    EXPECT_EQ(received[0].metadata_layout, RxMetadataLayout::LEGACY);
    EXPECT_EQ(received[0].frame_number, 7u);
    EXPECT_EQ(received[1].frame_number, 8u);
    EXPECT_EQ(received[1].bytes_written, 0x40u);
}

TEST(PcieReceiverAccounting, FrameReadyCallbackFiresPerFrame)
{
    auto access = std::make_shared<FakePcieDeviceAccess>();
    ScriptedReceiver receiver(access, small_geometry(), ring_config(0, 2), 0, 1 << 20);
    std::atomic<unsigned> callbacks { 0 };
    receiver.set_frame_ready([&] { callbacks++; });
    ASSERT_TRUE(receiver.start());
    auto received = run_monitor(receiver, *access, { { { 0, 1 } }, { { 1, 2 } }, { { 0, 3 } } });
    EXPECT_EQ(received.size(), 3u);
    EXPECT_EQ(callbacks.load(), 3u);
}

TEST(PcieReceiverAccounting, AFrameNotCollectedBeforeTheNextCountsAsDropped)
{
    // pending_ holds one frame. When the consumer has not collected it by the time the next
    // completion is processed, the newer frame replaces it and the evicted one is counted
    // as dropped, the same as a frame skipped inside a batched completion.
    auto access = std::make_shared<FakePcieDeviceAccess>();
    ScriptedReceiver receiver(access, small_geometry(), ring_config(0, 2), 0, 1 << 20);
    std::atomic<unsigned> queued { 0 };
    receiver.set_frame_ready([&queued] { queued++; });
    ASSERT_TRUE(receiver.start());

    std::thread monitor([&receiver] { receiver.blocking_monitor(); });
    MonitorJoiner joiner(receiver, monitor);
    receiver.write_slot(0, 1);
    access->complete(1);
    for (int i = 0; i < 200 && queued.load() < 1; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_EQ(queued.load(), 1u) << "first frame never queued";
    receiver.write_slot(1, 2);
    access->complete(1);
    for (int i = 0; i < 200 && queued.load() < 2; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_EQ(queued.load(), 2u) << "second frame never queued";

    PcieFrameInfo info;
    ASSERT_TRUE(receiver.get_next_frame(1000, info));
    EXPECT_EQ(info.frame_number, 2u) << "the newer frame is what the consumer gets";
    EXPECT_EQ(info.page_number, 1u);
    EXPECT_EQ(info.dropped, 1u) << "the evicted frame 1";
    EXPECT_FALSE(receiver.get_next_frame(200, info)) << "frame 1 is gone, not queued behind";
    receiver.close();
    monitor.join();
}

TEST(PcieReceiver, AFailedBaselineWaitFailsStart)
{
    // TIMEOUT just means no completion yet; FAILED is the MSIX_WAIT ioctl itself failing,
    // which every later wait would hit too. start() reports it rather than deferring it to
    // the monitor.
    auto access = std::make_shared<FakePcieDeviceAccess>();
    access->wait_frame_status = PcieStatus::FAILED;
    ScriptedReceiver receiver(access, small_geometry(), ring_config(0, 2), 0, 1 << 20);
    EXPECT_FALSE(receiver.start());
    EXPECT_FALSE(receiver.running());
    EXPECT_FALSE(access->released_handles.empty()) << "the mapping made before the wait is released";
}

TEST(PcieReceiver, MonitorExitOnWaitFailureStopsTheReceiver)
{
    // A hard wait_frame() failure mid-run ends the monitor. No frame can arrive after that,
    // so the receiver reports itself stopped and a consumer blocked in get_next_frame()
    // returns at once instead of running out its timeout.
    auto access = std::make_shared<FakePcieDeviceAccess>();
    ScriptedReceiver receiver(access, small_geometry(), ring_config(0, 2), 0, 1 << 20);
    ASSERT_TRUE(receiver.start());
    EXPECT_TRUE(receiver.running());

    access->wait_frame_status = PcieStatus::FAILED;
    std::thread monitor([&receiver] { receiver.blocking_monitor(); });
    monitor.join(); // the very first wait fails, so this returns immediately
    EXPECT_FALSE(receiver.running()) << "no frame can arrive once the monitor has left";

    const auto t0 = std::chrono::steady_clock::now();
    PcieFrameInfo info;
    EXPECT_FALSE(receiver.get_next_frame(5000, info));
    EXPECT_LT(std::chrono::steady_clock::now() - t0, std::chrono::seconds(1))
        << "returns on the stopped state, not on the timeout";
    receiver.close();
}

// ---------------------------------------------------------------------------
// PcieReceiverOp (Holoscan-coupled; only when HOLOLINK_MODULE_BUILD_OPERATORS).
// ---------------------------------------------------------------------------

#ifdef HOLOLINK_MODULE_PCIE_RECEIVER_OP_TEST

namespace {

class PcieReceiverOpTestApp : public holoscan::Application {
public:
    explicit PcieReceiverOpTestApp(EnumerationMetadata metadata)
        : metadata_(std::move(metadata))
    {
    }

    std::shared_ptr<hololink::module::operators::PcieReceiverOp> op() const { return op_; }

    void compose() override
    {
        op_ = make_operator<hololink::module::operators::PcieReceiverOp>(
            "pcie_receiver",
            holoscan::Arg("enumeration_metadata", metadata_),
            holoscan::Arg("frame_context", CUcontext { nullptr }),
            holoscan::Arg("sensor", int32_t { 0 }),
            holoscan::Arg("num_buffers", uint32_t { 3 }),
            holoscan::Arg("width", uint32_t { 1920 }),
            holoscan::Arg("height", uint32_t { 1080 }),
            holoscan::Arg("raw_start_byte", uint32_t { 640 }),
            make_condition<holoscan::CountCondition>(1));
        add_operator(op_);
    }

private:
    EnumerationMetadata metadata_;
    std::shared_ptr<hololink::module::operators::PcieReceiverOp> op_;
};

} // namespace

TEST(PcieReceiverOp, ConstructsAndComposes)
{
    auto app = holoscan::make_application<PcieReceiverOpTestApp>(EnumerationMetadata {});
    app->compose_graph();
    EXPECT_NE(app->op(), nullptr);
}

TEST(PcieReceiverOp, MetadataWithoutPeerIpIsRejectedNamingTheKey)
{
    using hololink::module::operators::PcieReceiverOp;

    // The static check start() runs first.
    EnumerationMetadata metadata;
    metadata["serial_number"] = std::string("pcie-0000:06:00.0");
    metadata["sensor"] = int64_t { 0 };
    try {
        PcieReceiverOp::device_node_from_metadata(metadata);
        FAIL() << "missing peer_ip must be rejected";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("peer_ip"), std::string::npos) << e.what();
    }
    // A literal node is accepted as a convenience spelling and needs no sysfs.
    metadata["peer_ip"] = std::string("/dev/hololink_pcie0");
    EXPECT_EQ(PcieReceiverOp::device_node_from_metadata(metadata), "/dev/hololink_pcie0");

    // The channel key proper is the BDF; one nothing is bound to is rejected naming it.
    metadata["peer_ip"] = std::string("ffff:ff:1f.7");
    try {
        PcieReceiverOp::device_node_from_metadata(metadata);
        FAIL() << "an unbound BDF must be rejected";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("ffff:ff:1f.7"), std::string::npos) << e.what();
    }
    // Neither a BDF nor a node.
    metadata["peer_ip"] = std::string("192.0.2.9");
    try {
        PcieReceiverOp::device_node_from_metadata(metadata);
        FAIL() << "an IPv4 peer must be rejected";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("peer_ip"), std::string::npos) << e.what();
    }

    // And the same failure through the operator's start(): Holoscan rethrows the
    // exception start() raised once the (aborted) run returns.
    EnumerationMetadata without_peer;
    without_peer["serial_number"] = std::string("pcie-0000:06:00.0");
    without_peer["sensor"] = int64_t { 0 };
    auto app = holoscan::make_application<PcieReceiverOpTestApp>(without_peer);
    try {
        app->run();
        FAIL() << "start() without peer_ip must fail the run";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("peer_ip"), std::string::npos) << e.what();
    }
}

#endif // HOLOLINK_MODULE_PCIE_RECEIVER_OP_TEST
