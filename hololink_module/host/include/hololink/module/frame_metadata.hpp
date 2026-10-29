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

/**
 * @brief Decode the block of metadata the device sends with each frame.
 *
 * Along with a frame's data, the device transmits a block of metadata
 * associated with that frame.  A receiver reserves block_size() bytes for the
 * block when it sizes the buffer it receives frames into, then passes the
 * received block to decode() to read the values out of it.
 *
 * The device writes the block in its own layout; use this service to
 * deserialize it.
 */
class FrameMetadataInterfaceV1 : public Service<FrameMetadataInterfaceV1> {
public:
    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    static constexpr const char* type_id = "frame_metadata.v1";
    static constexpr const char* release_version = "2.7.0";
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief Fetch a module's FrameMetadataInterfaceV1 instance.
     *
     * @param module The module that claims the device, from
     * @ref Adapter::get_module.
     * @param allow_null Return an empty pointer, rather than throwing, when
     * the module publishes no frame metadata service.
     * @return The module's frame metadata service.
     * @throws std::runtime_error if the module publishes no frame metadata
     * service and allow_null is false.
     */
    static std::shared_ptr<FrameMetadataInterfaceV1> get_service(
        std::shared_ptr<Module> module, bool allow_null = false)
    {
        return Service<FrameMetadataInterfaceV1>::get_service(
            std::move(module), "", allow_null);
    }

    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    virtual ~FrameMetadataInterfaceV1() = default;
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief What the device reported about one received frame.
     */
    struct FrameMetadata {
        /// Device-reported status of the frame.
        uint32_t flags;

        /// Packet sequence number the frame ended on.
        uint32_t psn;

        /// CRC the device computed over the frame data.
        uint32_t crc;

        /// Frame counter, as the device numbers frames.
        uint16_t frame_number;

        /// Device time when the frame's first sample data was received.
        uint64_t timestamp_s;

        /// Nanoseconds part of `timestamp_s`.
        uint32_t timestamp_ns;

        /// Frame data the device wrote, in bytes.
        uint64_t bytes_written;

        /// Device time when the metadata block itself was sent.
        uint64_t metadata_s;

        /// Nanoseconds part of `metadata_s`.
        uint32_t metadata_ns;
    };

    /**
     * @brief The size of a frame's metadata block, in bytes.
     *
     * A receiver adds this to the space it reserves for each frame, and
     * hands decode() a block this size. The value doesn't change while the
     * service is in use.
     *
     * @return Size of the metadata block, in bytes.
     */
    virtual size_t block_size() const = 0;

    /**
     * @brief Deserialize received frame metadata into `out_metadata`.
     *
     * @param host_memory The metadata block received with a frame, in host
     * memory.
     * @param host_memory_size_bytes Bytes available at `host_memory`.
     * @param[out] out_metadata What the device reported about the frame.
     * @return HOLOLINK_MODULE_OK, or HOLOLINK_MODULE_INVALID_PARAMETER when
     * fewer than block_size() bytes are available at `host_memory`.
     */
    virtual hololink_module_status_t decode(
        const void* host_memory,
        size_t host_memory_size_bytes,
        FrameMetadata& out_metadata) const = 0;
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_FRAME_METADATA_HPP
