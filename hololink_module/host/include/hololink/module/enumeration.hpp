/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_ENUMERATION_HPP
#define HOLOLINK_MODULE_ENUMERATION_HPP

#include <cstddef>
#include <cstdint>
#include <memory>

#include "enumeration_metadata.hpp"
#include "module.hpp"
#include "service.hpp"
#include "status.h"

namespace hololink::module {

/* Per-module discovery hook. The Adapter calls update_metadata
 * after a bootp packet arrives (or after a manual enumerate() call)
 * with the bootp-decoded metadata; the module enriches it with
 * version-specific fields it knows about (VP/SIF/HIF addresses,
 * sensor counts, board-specific descriptors). raw_packet may be
 * null when invoked from the manual path.
 *
 * Modules publish a single EnumerationInterfaceV1 instance under
 * instance_id "" (singleton).
 *
 * update_metadata returns HOLOLINK_MODULE_OK after enriching the
 * metadata, HOLOLINK_MODULE_ENUMERATION_SKIPPED when the module
 * recognizes the device but declines to drive it (the Adapter then
 * suppresses the announcement without treating it as an error), or any
 * other status to signal a hard enrichment failure (the Adapter throws). */
class EnumerationInterfaceV1 : public Service<EnumerationInterfaceV1> {
public:
    static constexpr const char* type_id = "enumeration.v1";

    // Singleton: hides the inherited three-arg form, passes "" instance_id.
    static std::shared_ptr<EnumerationInterfaceV1> get_service(
        std::shared_ptr<Module> module, bool allow_null = false)
    {
        return Service<EnumerationInterfaceV1>::get_service(
            std::move(module), "", allow_null);
    }

    virtual ~EnumerationInterfaceV1() = default;

    virtual hololink_module_status_t update_metadata(
        EnumerationMetadata& metadata,
        const uint8_t* raw_packet, size_t raw_packet_len)
        = 0;
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_ENUMERATION_HPP
