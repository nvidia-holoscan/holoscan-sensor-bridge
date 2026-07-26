/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_CORE_LINUX_DATA_CHANNEL_DEFAULT_HPP
#define HOLOLINK_MODULE_CORE_LINUX_DATA_CHANNEL_DEFAULT_HPP

#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include "hololink/module/data_channel.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/hololink.hpp"
#include "hololink/module/linux_data_channel.hpp"
#include "hololink/module/linux_receiver.hpp"
#include "hololink/module/status.h"

#include "hololink/core/data_channel.hpp"
#include "hololink/core/hololink.hpp"
#include "hololink/core/metadata.hpp"

#include "hololink_default.hpp"

namespace hololink::module::module_core {

/* LinuxDataChannelInterfaceV1 backed by a hololink::DataChannel — the
 * software-transport sibling of RoceDataChannelV1. Same has-a
 * relationship with the per-channel anchor; same configure(metadata)
 * that drives the anchor's ensure_configured(metadata) and then builds
 * the legacy DataChannel against the per-board HololinkImpl's shared
 * legacy access.
 *
 * The wire protocol is still RoCEv2, so the channel still calls the
 * legacy DataChannel::configure_roce. The two software-path differences
 * (device frame-memory address 0 and the host socket's local_port) live
 * in attach_receiver below. */
class LinuxDataChannelV1 : public LinuxDataChannelInterfaceV1,
                           public Service<LinuxDataChannelV1> {
public:
    static constexpr const char* type_id = "linux_data_channel.module_core.v1";
    using Service<LinuxDataChannelV1>::get_service;
    using Service<LinuxDataChannelV1>::for_each_type_id;
    using ServiceAlias = LinuxDataChannelInterfaceV1;

    explicit LinuxDataChannelV1(std::shared_ptr<DataChannelInterfaceV1> anchor)
        : anchor_(std::move(anchor))
    {
        if (!anchor_) {
            throw std::runtime_error(
                "While constructing LinuxDataChannel: anchor "
                "(DataChannelInterfaceV1) must be non-null");
        }
    }

    // Reconfigure when the application resolves the channel with metadata
    // that differs from what backing_ was last built from, instead of
    // freezing the first resolution (see RoceDataChannelV1 for the
    // rationale — process-lifetime cached singleton + per-application host
    // / data-plane addressing).
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

    hololink_module_status_t configure_socket(int data_socket) override
    {
        if (!backing_) {
            throw std::runtime_error(
                "While configuring the Linux data channel socket: channel "
                "has not been configured");
        }
        backing_->configure_socket(data_socket);
        return HOLOLINK_MODULE_OK;
    }

    hololink_module_status_t attach_receiver(
        std::shared_ptr<LinuxReceiverInterfaceV1> receiver) override
    {
        if (!receiver) {
            throw std::runtime_error(
                "While attaching a receiver to the Linux data channel: "
                "receiver is null (start the receiver before calling "
                "attach_receiver)");
        }
        backing_->authenticate(receiver->get_qp_number(), receiver->get_rkey());
        // Software path: HSB is configured to write from address 0 (the
        // user-space receiver adds the local buffer offset itself), and
        // it targets the host socket's bound UDP port. Contrast the
        // hardware RoCE channel, which programs the receiver's device
        // frame-memory address and the fixed RoCEv2 port.
        constexpr uint64_t DISTAL_MEMORY_ADDRESS_START = 0;
        backing_->configure_roce(
            DISTAL_MEMORY_ADDRESS_START,
            receiver->frame_size(),
            receiver->page_size(),
            receiver->pages(),
            receiver->local_port());
        return HOLOLINK_MODULE_OK;
    }

    hololink_module_status_t detach_receiver() override
    {
        backing_->unconfigure();
        return HOLOLINK_MODULE_OK;
    }

    /* The materialized legacy DataChannel, or null if configure() has
     * not yet run. */
    std::shared_ptr<hololink::DataChannel> legacy_data_channel() const { return backing_; }

    /* The per-channel anchor this software wrapper composes with. */
    std::shared_ptr<DataChannelInterfaceV1> data_channel() const { return anchor_; }

protected:
    std::string frame_end_sequencer_instance_id() override
    {
        return "serial=" + serial() + ";data_channel=" + data_channel_id() + ";kind=frame_end";
    }

    std::string parent_hololink_instance_id() override
    {
        return "serial=" + serial();
    }

    /* Hook for FPGA-revision-specific DataChannel subclasses. Default
     * builds a plain hololink::DataChannel; per-revision supplements
     * override to return their subclassed DataChannel so
     * backing_->configure_roce(...) dispatches through the
     * revision-specific overrides. Mirrors RoceDataChannelV1. */
    virtual std::shared_ptr<hololink::DataChannel> make_backing(
        const hololink::Metadata& legacy_metadata,
        std::function<std::shared_ptr<hololink::Hololink>(const hololink::Metadata&)>
            create_hololink)
    {
        return std::make_shared<hololink::DataChannel>(
            legacy_metadata, std::move(create_hololink));
    }

private:
    std::string serial() const
    {
        return anchor_->enumeration_metadata().get<std::string>(
            "serial_number", std::string {});
    }

    std::string data_channel_id() const
    {
        return std::to_string(
            anchor_->enumeration_metadata().get<int64_t>(
                "data_channel", int64_t { 0 }));
    }

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

#endif // HOLOLINK_MODULE_CORE_LINUX_DATA_CHANNEL_DEFAULT_HPP
