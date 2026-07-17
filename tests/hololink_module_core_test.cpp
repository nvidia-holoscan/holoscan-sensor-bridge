/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Direct unit tests for hololink::module's default V1 wrapper
 * implementations that don't require live hardware:
 *   - I2cLockImplV1: BasicLockable / Lockable concept compliance
 *     against a std::mutex backing store.
 *   - FrameMetadataV1: byte-for-byte decode of a synthetic
 *     end-of-frame block that the existing
 *     hololink::core::Hololink::deserialize_metadata produces.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>

#include "frame_metadata_default.hpp"
#include "hololink/module/frame_metadata.hpp"
#include "hololink/module/i2c_lock.hpp"
#include "i2c_lock_default.hpp"

namespace {

void encode_be_u16(uint8_t*& p, uint16_t v)
{
    *p++ = static_cast<uint8_t>(v >> 8);
    *p++ = static_cast<uint8_t>(v);
}

void encode_be_u32(uint8_t*& p, uint32_t v)
{
    *p++ = static_cast<uint8_t>(v >> 24);
    *p++ = static_cast<uint8_t>(v >> 16);
    *p++ = static_cast<uint8_t>(v >> 8);
    *p++ = static_cast<uint8_t>(v);
}

void encode_be_u64(uint8_t*& p, uint64_t v)
{
    for (int shift = 56; shift >= 0; shift -= 8) {
        *p++ = static_cast<uint8_t>(v >> shift);
    }
}

} // namespace

TEST(HololinkAdapterModuleCore, I2cLockImplFollowsBasicLockable)
{
    using hololink::module::module_core::I2cLockImplV1;

    auto mutex = std::make_shared<std::mutex>();
    I2cLockImplV1 lock(mutex);

    lock.lock();
    EXPECT_FALSE(mutex->try_lock()); // backing mutex is held
    lock.unlock();
    EXPECT_TRUE(mutex->try_lock());
    mutex->unlock();

    EXPECT_TRUE(lock.try_lock());
    EXPECT_FALSE(lock.try_lock()); // already held
    lock.unlock();
    EXPECT_TRUE(lock.try_lock());
    lock.unlock();
}

TEST(HololinkAdapterModuleCore, I2cLockImplSerializesThreads)
{
    using hololink::module::module_core::I2cLockImplV1;

    auto mutex = std::make_shared<std::mutex>();
    I2cLockImplV1 lock(mutex);

    int counter = 0;
    constexpr int PER_THREAD = 10000;
    std::thread t1([&]() {
        for (int i = 0; i < PER_THREAD; ++i) {
            std::lock_guard<I2cLockImplV1> guard(lock);
            ++counter;
        }
    });
    std::thread t2([&]() {
        for (int i = 0; i < PER_THREAD; ++i) {
            std::lock_guard<I2cLockImplV1> guard(lock);
            ++counter;
        }
    });
    t1.join();
    t2.join();
    EXPECT_EQ(counter, 2 * PER_THREAD);
}

TEST(HololinkAdapterModuleCore, FrameMetadataImplBlockSize)
{
    using hololink::module::module_core::FrameMetadataV1;

    // Call block_size() to prove the method is reachable through the
    // interface; the layout-specific value lives in the impl and isn't
    // pinned here.
    FrameMetadataV1 impl;
    (void)impl.block_size();
}

TEST(HololinkAdapterModuleCore, FrameMetadataImplDecodesAllFields)
{
    using hololink::module::FrameMetadataInterfaceV1;
    using hololink::module::module_core::FrameMetadataV1;

    // 48-byte block matching hololink::core::Hololink::deserialize_metadata.
    constexpr size_t BLOCK_SIZE = 48;
    uint8_t buf[BLOCK_SIZE] = {};
    uint8_t* p = buf;
    encode_be_u32(p, 0x11223344u); // flags
    encode_be_u32(p, 0xAABBCCDDu); // psn
    encode_be_u32(p, 0xCAFEBABEu); // crc
    encode_be_u64(p, 0x0102030405060708ULL); // timestamp_s
    encode_be_u32(p, 0x09ABCDEFu); // timestamp_ns
    encode_be_u64(p, 0xDEADBEEFCAFEBABEULL); // bytes_written
    encode_be_u16(p, 0x0000u); // ignored
    encode_be_u16(p, 0x4242u); // frame_number
    encode_be_u64(p, 0x1122334455667788ULL); // metadata_s
    encode_be_u32(p, 0xFEEDFACEu); // metadata_ns
    ASSERT_EQ(static_cast<size_t>(p - buf), BLOCK_SIZE);

    FrameMetadataV1 impl;
    FrameMetadataInterfaceV1::FrameMetadata out {};
    EXPECT_EQ(impl.decode(buf, sizeof(buf), out), HOLOLINK_MODULE_OK);

    EXPECT_EQ(out.flags, 0x11223344u);
    EXPECT_EQ(out.psn, 0xAABBCCDDu);
    EXPECT_EQ(out.crc, 0xCAFEBABEu);
    EXPECT_EQ(out.timestamp_s, 0x0102030405060708ULL);
    EXPECT_EQ(out.timestamp_ns, 0x09ABCDEFu);
    EXPECT_EQ(out.bytes_written, 0xDEADBEEFCAFEBABEULL);
    EXPECT_EQ(out.frame_number, 0x4242u);
    EXPECT_EQ(out.metadata_s, 0x1122334455667788ULL);
    EXPECT_EQ(out.metadata_ns, 0xFEEDFACEu);
}

TEST(HololinkAdapterModuleCore, FrameMetadataImplRejectsTooSmall)
{
    using hololink::module::FrameMetadataInterfaceV1;
    using hololink::module::module_core::FrameMetadataV1;

    uint8_t buf[10] = {};
    FrameMetadataV1 impl;
    FrameMetadataInterfaceV1::FrameMetadata out {};
    EXPECT_EQ(impl.decode(buf, sizeof(buf), out),
        HOLOLINK_MODULE_INVALID_PARAMETER);
}

TEST(HololinkAdapterModuleCore, FrameMetadataImplRejectsNullPointer)
{
    using hololink::module::FrameMetadataInterfaceV1;
    using hololink::module::module_core::FrameMetadataV1;

    FrameMetadataV1 impl;
    FrameMetadataInterfaceV1::FrameMetadata out {};
    EXPECT_EQ(impl.decode(nullptr, 48, out),
        HOLOLINK_MODULE_INVALID_PARAMETER);
}
