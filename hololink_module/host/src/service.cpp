/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hololink/module/service.hpp"

#include <memory>
#include <string>

#include "hololink/module/adapter.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/module.hpp"

namespace hololink::module {

std::shared_ptr<const void> get_or_construct_service_via_metadata(
    const EnumerationMetadata& metadata,
    const std::string& instance_id,
    const char* type_id,
    bool allow_null)
{
    std::shared_ptr<Module> module = Adapter::get_adapter().get_module(metadata);
    return module->get_service(instance_id.c_str(), type_id, allow_null);
}

} // namespace hololink::module
