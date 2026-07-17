/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_FRAME_METADATA_HPP
#define HOLOLINK_MODULE_FRAME_METADATA_HPP

#include <cstddef>
#include <cstdint>
#include <memory>

#include "module.hpp"
#include "service.hpp"
#include "status.h"

namespace hololink::module {

/* Singleton service that decodes the fixed-size end-of-frame metadata
 * block the device deposits into the receiver's host buffer. The
 * implementation in module/core/ owns the layout; receivers pass a
 * pointer + size and consume the structured fields out_metadata. */
class FrameMetadataInterfaceV1 : public Service<FrameMetadataInterfaceV1> {
public:
    static constexpr const char* type_id = "frame_metadata.v1";

    // Singleton: hides the inherited three-arg form, passes "" instance_id.
    static std::shared_ptr<FrameMetadataInterfaceV1> get_service(
        std::shared_ptr<Module> module, bool allow_null = false)
    {
        return Service<FrameMetadataInterfaceV1>::get_service(
            std::move(module), "", allow_null);
    }

    virtual ~FrameMetadataInterfaceV1() = default;

    struct FrameMetadata {
        uint32_t flags;
        uint32_t psn;
        uint32_t crc;
        uint16_t frame_number;
        // Time when the first sample data for the frame was received.
        uint64_t timestamp_s;
        uint32_t timestamp_ns;
        uint64_t bytes_written;
        // Time the metadata block itself was emitted.
        uint64_t metadata_s;
        uint32_t metadata_ns;
    };

    /* Size in bytes of the end-of-frame metadata block this
     * implementation decodes. Callers use it to size the host buffer
     * they hand to decode(...). Constant for the lifetime of the
     * implementation. */
    virtual size_t block_size() const = 0;

    /* Decode the end-of-frame metadata block at host_memory into
     * out_metadata. host_memory_size_bytes is the size of the block
     * available at host_memory; HOLOLINK_MODULE_INVALID_PARAMETER is
     * returned when the buffer is too small to contain a valid block.
     *
     * Stateless and const so the singleton is safe to call
     * concurrently from receiver threads. */
    virtual hololink_module_status_t decode(
        const void* host_memory,
        size_t host_memory_size_bytes,
        FrameMetadata& out_metadata) const = 0;
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_FRAME_METADATA_HPP
