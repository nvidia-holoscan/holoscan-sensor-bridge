/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_CORE_ROCE_DATA_CHANNEL_DEFAULT_HPP
#define HOLOLINK_MODULE_CORE_ROCE_DATA_CHANNEL_DEFAULT_HPP

#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include "hololink/module/data_channel.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/hololink.hpp"
#include "hololink/module/roce_data_channel.hpp"
#include "hololink/module/roce_receiver.hpp"
#include "hololink/module/status.h"

#include "hololink/core/data_channel.hpp"
#include "hololink/core/hololink.hpp"
#include "hololink/core/metadata.hpp"

#include "hololink_default.hpp"

namespace hololink::module::module_core {

/* RoceDataChannelInterfaceV1 backed by a hololink::DataChannel.
 *
 * Has-a relationship with the per-channel anchor: holds a
 * shared_ptr<DataChannelInterfaceV1> handed in at construction by the
 * supplement's construct_service. The anchor owns the per-channel
 * metadata + Hololink lookup; this class owns the RoCE-specific
 * transport state (legacy hololink::DataChannel, attach_receiver /
 * detach_receiver behaviour).
 *
 * configure(metadata) drives the anchor's ensure_configured(metadata)
 * — the anchor's framework latch makes this a no-op if the application
 * already configured the anchor directly. That's the contract: an
 * application that wants only the RoCE view can call
 * RoceDataChannelInterfaceV1::get_service(metadata) without first
 * touching the anchor; this code makes sure the anchor is ready before
 * touching its hololink(). */
class RoceDataChannelV1 : public RoceDataChannelInterfaceV1,
                          public Service<RoceDataChannelV1> {
public:
    static constexpr const char* type_id = "roce_data_channel.module_core.v1";
    using Service<RoceDataChannelV1>::get_service;
    using Service<RoceDataChannelV1>::for_each_type_id;
    using ServiceAlias = RoceDataChannelInterfaceV1;

    explicit RoceDataChannelV1(std::shared_ptr<DataChannelInterfaceV1> anchor)
        : anchor_(std::move(anchor))
    {
        if (!anchor_) {
            throw std::runtime_error(
                "While constructing RoceDataChannelImpl: anchor "
                "(DataChannelInterfaceV1) must be non-null");
        }
    }

    // DataChannel config tracks the application's enumeration metadata:
    // reconfigure when the metadata supplied to get_service differs from
    // what backing_ was last built from, instead of freezing the first
    // resolution (the base ensure_configured's call_once). The
    // per-(serial,data_channel) service is a process-lifetime cached
    // singleton, so without this a value an earlier resolution cached
    // (e.g. another test's channel for the same key) would silently
    // override what the application asks for now — e.g. peer_ip, which
    // drives DP_HOST, and the hif/vp data-plane addressing.
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
        // Drive the anchor's configure (idempotent via its own framework
        // latch). After this call the anchor's enumeration_metadata() /
        // hololink() are usable.
        anchor_->ensure_configured(metadata);

        auto hololink_impl = HololinkV1::get_service(
            this->module(),
            HololinkInterfaceV1::locator_id(metadata).c_str());
        hololink_impl->register_associated(this);
        auto legacy = hololink_impl->legacy_access();

        // Build the legacy DataChannel against the shared legacy
        // access. The create_hololink lambda hands the same legacy
        // back to the DataChannel's internal lookups so the two share
        // state.
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

    hololink_module_status_t attach_receiver(
        std::shared_ptr<RoceReceiverInterfaceV1> receiver) override
    {
        if (!receiver) {
            throw std::runtime_error(
                "While attaching a receiver to the RoCE data channel: receiver "
                "is null (start the receiver before calling attach_receiver)");
        }
        backing_->authenticate(receiver->get_qp_number(), receiver->get_rkey());
        // RoCEv2 destination UDP port — what ibverbs binds the QP to, and
        // what we tell the FPGA to target. Hardcoded in the legacy receiver
        // operator (roce_receiver_op.cpp local_port = 4791) for the same
        // reason: the legacy stack writes this verbatim to DP_HOST_UDP_PORT.
        constexpr uint32_t ROCE_V2_UDP_PORT = 4791;
        backing_->configure_roce(
            receiver->external_frame_memory(),
            receiver->frame_size(),
            receiver->page_size(),
            receiver->pages(),
            ROCE_V2_UDP_PORT);
        return HOLOLINK_MODULE_OK;
    }

    hololink_module_status_t detach_receiver() override
    {
        backing_->unconfigure();
        return HOLOLINK_MODULE_OK;
    }

    /* The materialized legacy DataChannel, or null if configure() has
     * not yet run. Sibling impls in the same module — currently
     * SequencerV1, reached through this module's Publisher — borrow
     * it via the construct_service path; abstract callers go through
     * the V1 surface and never touch this. */
    std::shared_ptr<hololink::DataChannel> legacy_data_channel() const { return backing_; }

    /* The per-channel anchor this RoCE wrapper composes with. Module
     * code that already has a RoCE-typed shared_ptr in hand can reach
     * the anchor's metadata + hololink without re-resolving them
     * through the Publisher. */
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
     * (e.g. the hsb_lite_2510 variant) override to return their
     * subclassed DataChannel so backing_->configure_roce(...) and
     * friends dispatch through the revision-specific overrides. */
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

#endif // HOLOLINK_MODULE_CORE_ROCE_DATA_CHANNEL_DEFAULT_HPP
