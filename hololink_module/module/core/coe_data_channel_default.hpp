/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_CORE_COE_DATA_CHANNEL_DEFAULT_HPP
#define HOLOLINK_MODULE_CORE_COE_DATA_CHANNEL_DEFAULT_HPP

#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include "hololink/module/coe_data_channel.hpp"
#include "hololink/module/data_channel.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/hololink.hpp"
#include "hololink/module/status.h"

#include "hololink/core/csi_formats.hpp"
#include "hololink/core/data_channel.hpp"
#include "hololink/core/hololink.hpp"
#include "hololink/core/metadata.hpp"

#include "hololink_default.hpp"

namespace hololink::module::module_core {

/* CoeDataChannelInterfaceV1 backed by a hololink::DataChannel.
 *
 * Has-a relationship with the per-channel anchor: holds a
 * shared_ptr<DataChannelInterfaceV1> handed in at construction by the
 * supplement's construct_service. The anchor owns the per-channel
 * metadata + Hololink lookup; this class owns the CoE-specific
 * transport state (legacy hololink::DataChannel, configure_coe /
 * unconfigure behaviour). */
class CoeDataChannelV1 : public CoeDataChannelInterfaceV1,
                         public Service<CoeDataChannelV1> {
public:
    static constexpr const char* type_id = "coe_data_channel.module_core.v1";
    using Service<CoeDataChannelV1>::get_service;
    using Service<CoeDataChannelV1>::for_each_type_id;
    using ServiceAlias = CoeDataChannelInterfaceV1;

    explicit CoeDataChannelV1(std::shared_ptr<DataChannelInterfaceV1> anchor)
        : anchor_(std::move(anchor))
    {
        if (!anchor_) {
            throw std::runtime_error(
                "While constructing CoeDataChannelImpl: anchor "
                "(DataChannelInterfaceV1) must be non-null");
        }
    }

    // Reconfigure when the application resolves the channel with metadata
    // that differs from what backing_ was last built from, instead of
    // freezing the first resolution (see RoceDataChannelV1 for rationale).
    void ensure_configured(const EnumerationMetadata& metadata) override
    {
        // get_service(metadata) invokes this without holding any framework
        // lock, so serialize concurrent resolutions of this per-(serial,
        // data_channel) singleton — the base's std::call_once (which this
        // override replaces) is what previously provided that safety.
        std::lock_guard<std::mutex> guard(configure_mutex_);
        if (backing_ && metadata == applied_metadata_) {
            return;
        }
        configure(metadata);
        applied_metadata_ = metadata;
    }

    void configure(const EnumerationMetadata& metadata) override
    {
        anchor_->ensure_configured(metadata);

        auto hololink_impl = HololinkV1::get_service(
            this->module(),
            HololinkInterfaceV1::locator_id(metadata).c_str());
        hololink_impl->register_associated(this);
        auto legacy = hololink_impl->legacy_access();

        hololink::Metadata legacy_metadata;
        for (const auto& kv : metadata) {
            legacy_metadata[kv.first] = kv.second;
        }
        auto create_hololink
            = [legacy](const hololink::Metadata&) -> std::shared_ptr<hololink::Hololink> {
            return legacy;
        };
        backing_ = make_backing(legacy_metadata, create_hololink);
    }

    hololink_module_status_t set_packetizer_for_pixel_format(
        uint32_t pixel_format) override
    {
        if (!backing_) {
            return HOLOLINK_MODULE_INVALID_PARAMETER;
        }
        auto program = hololink::csi::get_packetizer_program(
            static_cast<hololink::csi::PixelFormat>(pixel_format));
        backing_->set_packetizer_program(program);
        return HOLOLINK_MODULE_OK;
    }

    hololink_module_status_t configure_coe(
        uint8_t channel, size_t frame_size, uint32_t pixel_width,
        bool vlan_enabled) override
    {
        if (!backing_) {
            return HOLOLINK_MODULE_INVALID_PARAMETER;
        }
        backing_->configure_coe(channel, frame_size, pixel_width, vlan_enabled);
        return HOLOLINK_MODULE_OK;
    }

    hololink_module_status_t unconfigure() override
    {
        if (!backing_) {
            return HOLOLINK_MODULE_INVALID_PARAMETER;
        }
        backing_->unconfigure();
        return HOLOLINK_MODULE_OK;
    }

    std::shared_ptr<hololink::DataChannel> legacy_data_channel() const { return backing_; }

    std::shared_ptr<DataChannelInterfaceV1> data_channel() const { return anchor_; }

protected:
    virtual std::shared_ptr<hololink::DataChannel> make_backing(
        const hololink::Metadata& legacy_metadata,
        std::function<std::shared_ptr<hololink::Hololink>(const hololink::Metadata&)>
            create_hololink)
    {
        return std::make_shared<hololink::DataChannel>(
            legacy_metadata, std::move(create_hololink));
    }

private:
    std::shared_ptr<DataChannelInterfaceV1> anchor_;
    std::shared_ptr<hololink::DataChannel> backing_;
    // The metadata backing_ was last built from; ensure_configured() rebuilds
    // only when the application resolves the channel with different metadata.
    EnumerationMetadata applied_metadata_;
    // Serializes ensure_configured() so concurrent resolutions can't race on
    // backing_ / applied_metadata_.
    std::mutex configure_mutex_;
};

} // namespace hololink::module::module_core

#endif // HOLOLINK_MODULE_CORE_COE_DATA_CHANNEL_DEFAULT_HPP
