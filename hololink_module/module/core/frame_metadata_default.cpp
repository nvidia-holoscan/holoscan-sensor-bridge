/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "frame_metadata_default.hpp"

#include <cstdint>

namespace hololink::module::module_core {

// End-of-frame metadata block layout in big-endian network order.
// Mirrors the layout the device emits — see
// hololink::Hololink::deserialize_metadata in src/hololink/core/.
//
//   uint32 flags
//   uint32 psn
//   uint32 crc
//   uint64 timestamp_s
//   uint32 timestamp_ns
//   uint64 bytes_written
//   uint16 reserved
//   uint16 frame_number
//   uint64 metadata_s
//   uint32 metadata_ns
//
// Total: 48 bytes.
static constexpr size_t BLOCK_SIZE = 48;

static uint16_t read_be_u16(const uint8_t* p)
{
    return static_cast<uint16_t>(
        (static_cast<uint16_t>(p[0]) << 8) | static_cast<uint16_t>(p[1]));
}

static uint32_t read_be_u32(const uint8_t* p)
{
    return (static_cast<uint32_t>(p[0]) << 24)
        | (static_cast<uint32_t>(p[1]) << 16)
        | (static_cast<uint32_t>(p[2]) << 8)
        | static_cast<uint32_t>(p[3]);
}

static uint64_t read_be_u64(const uint8_t* p)
{
    return (static_cast<uint64_t>(read_be_u32(p)) << 32)
        | static_cast<uint64_t>(read_be_u32(p + 4));
}

size_t FrameMetadataV1::block_size() const
{
    return BLOCK_SIZE;
}

hololink_module_status_t FrameMetadataV1::decode(
    const void* host_memory,
    size_t host_memory_size_bytes,
    FrameMetadata& out_metadata) const
{
    if (!host_memory || host_memory_size_bytes < BLOCK_SIZE) {
        return HOLOLINK_MODULE_INVALID_PARAMETER;
    }

    const uint8_t* p = static_cast<const uint8_t*>(host_memory);
    out_metadata.flags = read_be_u32(p);
    p += 4;
    out_metadata.psn = read_be_u32(p);
    p += 4;
    out_metadata.crc = read_be_u32(p);
    p += 4;
    out_metadata.timestamp_s = read_be_u64(p);
    p += 8;
    out_metadata.timestamp_ns = read_be_u32(p);
    p += 4;
    out_metadata.bytes_written = read_be_u64(p);
    p += 8;
    p += 2; // reserved
    out_metadata.frame_number = read_be_u16(p);
    p += 2;
    out_metadata.metadata_s = read_be_u64(p);
    p += 8;
    out_metadata.metadata_ns = read_be_u32(p);
    return HOLOLINK_MODULE_OK;
}

} // namespace hololink::module::module_core
