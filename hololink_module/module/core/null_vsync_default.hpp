/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_CORE_NULL_VSYNC_DEFAULT_HPP
#define HOLOLINK_MODULE_CORE_NULL_VSYNC_DEFAULT_HPP

#include <memory>
#include <utility>

#include "hololink/module/status.h"
#include "hololink/module/vsync.hpp"

#include "hololink_default.hpp"

namespace hololink::module::module_core {

/* Module-side no-op VsyncInterfaceV1. Applications fetch it via
 * HololinkInterfaceV1::null_vsync() to obtain a concrete Vsync
 * handle backed by the standard locator alongside PtpPpsOutputV1. */
class NullVsyncV1 : public VsyncInterfaceV1 {
public:
    explicit NullVsyncV1(std::shared_ptr<HololinkV1> owner)
        : owner_(std::move(owner))
    {
        owner_->register_associated(this);
    }

    bool is_enabled() const override { return false; }
    hololink_module_status_t start() override { return HOLOLINK_MODULE_OK; }
    hololink_module_status_t stop() override { return HOLOLINK_MODULE_OK; }

private:
    std::shared_ptr<HololinkV1> owner_;
};

} // namespace hololink::module::module_core

#endif // HOLOLINK_MODULE_CORE_NULL_VSYNC_DEFAULT_HPP
