/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_CORE_DATA_CHANNEL_DEFAULT_HPP
#define HOLOLINK_MODULE_CORE_DATA_CHANNEL_DEFAULT_HPP

#include <memory>
#include <stdexcept>
#include <string>

#include "hololink/module/data_channel.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/hololink.hpp"

#include "hololink_default.hpp"

namespace hololink::module::module_core {

/* DataChannelInterfaceV1 — the per-channel anchor impl. Stashes the
 * channel's enumeration metadata and looks up the per-board
 * HololinkInterfaceV1, ensuring it's configured (via the framework's
 * ensure_configured latch). Sibling per-channel services read these
 * two values off the anchor.
 *
 * Transport-agnostic — the legacy hololink::DataChannel is built by a
 * separate transport service (RoceDataChannelV1 et al) that holds
 * a shared_ptr to this anchor. */
class DataChannelV1 : public DataChannelInterfaceV1,
                      public Service<DataChannelV1> {
public:
    static constexpr const char* type_id = "data_channel.module_core.v1";
    using Service<DataChannelV1>::get_service;
    using Service<DataChannelV1>::for_each_type_id;
    using ServiceAlias = DataChannelInterfaceV1;

    DataChannelV1() = default;

    /* Eager constructor used by HololinkV1::configure to build the
     * default per-board DataChannel alongside the Hololink itself. The
     * parent Hololink is injected here rather than looked up later, so
     * the two share state by construction — they cannot drift onto
     * inconsistent metadata. The framework's configure(metadata) latch
     * is not flipped (this object is held on the Hololink, not
     * published into the Publisher's cache); the corresponding configure
     * call is not expected to run, but if it does it just re-stashes
     * the same fields. */
    DataChannelV1(std::shared_ptr<HololinkInterfaceV1> hololink,
        EnumerationMetadata metadata)
        : metadata_(std::move(metadata))
        , hololink_(std::move(hololink))
    {
        if (!hololink_) {
            throw std::runtime_error(
                "While constructing default DataChannel: parent "
                "HololinkInterface must be non-null");
        }
    }

    const EnumerationMetadata& enumeration_metadata() const override
    {
        return metadata_;
    }

    std::shared_ptr<HololinkInterfaceV1> hololink() const override
    {
        return hololink_;
    }

    hololink_module_status_t device_lost() override
    {
        // The board owns the invalidation; this channel registered itself
        // with the Hololink at configure, so the cascade covers it.
        return hololink_->device_lost();
    }

    void configure(const EnumerationMetadata& metadata) override
    {
        // configure runs exactly once per instance — framework's
        // ConfigurableService<T>::get_service wraps this in
        // std::call_once on the instance's configure_once_ latch.
        metadata_ = metadata;

        // Reach the per-board HololinkInterfaceV1 via the instance-id
        // form (the only form available inside a module .so), then
        // ensure_configured(metadata) so the legacy backing
        // materializes via the framework's latch. Subsequent
        // get_service(metadata) calls on the same Hololink instance
        // observe the latch already set and return immediately.
        const std::string hololink_id
            = HololinkInterfaceV1::locator_id(metadata);
        auto hololink_impl = HololinkV1::get_service(
            this->module(), hololink_id.c_str());
        hololink_impl->ensure_configured(metadata);
        hololink_impl->register_associated(this);
        hololink_ = hololink_impl;
    }

private:
    EnumerationMetadata metadata_;
    std::shared_ptr<HololinkInterfaceV1> hololink_;
};

} // namespace hololink::module::module_core

#endif // HOLOLINK_MODULE_CORE_DATA_CHANNEL_DEFAULT_HPP
