/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_CORE_FRAME_METADATA_DEFAULT_HPP
#define HOLOLINK_MODULE_CORE_FRAME_METADATA_DEFAULT_HPP

#include <cstddef>

#include "hololink/module/frame_metadata.hpp"
#include "hololink/module/status.h"

namespace hololink::module::module_core {

/* Default FrameMetadataInterfaceV1 — delegates byte-decoding to the
 * existing src/hololink/core/ Hololink::deserialize_metadata. Stateless
 * and const so the singleton is safe to call concurrently. */
class FrameMetadataV1 : public FrameMetadataInterfaceV1 {
public:
    size_t block_size() const override;

    hololink_module_status_t decode(
        const void* host_memory,
        size_t host_memory_size_bytes,
        FrameMetadata& out_metadata) const override;
};

} // namespace hololink::module::module_core

#endif // HOLOLINK_MODULE_CORE_FRAME_METADATA_DEFAULT_HPP
