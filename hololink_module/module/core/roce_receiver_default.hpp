/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_CORE_ROCE_RECEIVER_DEFAULT_HPP
#define HOLOLINK_MODULE_CORE_ROCE_RECEIVER_DEFAULT_HPP

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "hololink/module/data_channel.hpp"
#include "hololink/module/roce_receiver.hpp"
#include "hololink/module/status.h"

#include <hololink/operators/roce_receiver/roce_receiver.hpp>

#include "hololink_default.hpp"

namespace hololink::module::module_core {

/* RoceReceiverInterfaceV1 backed by hololink::operators::RoceReceiver.
 * Constructed by the per-board supplement as a shell at enumerate
 * time; start(...) is when the underlying legacy receiver actually
 * comes up.
 *
 * Per-board overrides plug in by deriving from this class and
 * overriding make_receiver() to return a subclass of the
 * legacy receiver — `module/hsb_lite_2510/` uses this seam to attach
 * the 2510-specific behavior the FPGA needs (the legacy class already
 * has virtual hooks for metadata copy / decode). */
class RoceReceiverV1 : public RoceReceiverInterfaceV1,
                       public Service<RoceReceiverV1> {
public:
    static constexpr const char* type_id = "roce_receiver.module_core.v1";
    using Service<RoceReceiverV1>::get_service;
    using Service<RoceReceiverV1>::for_each_type_id;
    using ServiceAlias = RoceReceiverInterfaceV1;

    RoceReceiverV1() = default;

    /* Anchor-fetch: per-channel RoceReceiver routes through the
     * per-channel DataChannelInterfaceV1 anchor. Cache-only — the
     * application is required to construct the DataChannel for this
     * (serial, data_plane) first via
     * RoceDataChannelInterfaceV1::get_service(metadata). If the
     * cached channel exists but hasn't been configured (no backing
     * Hololink yet), the receiver fails fast with a clear message
     * so the misordered call site is obvious.
     *
     * Runs exactly once per instance — framework's
     * ConfigurableService<T>::get_service wraps this in
     * std::call_once on the instance's configure_once_ latch. */
    void configure(const EnumerationMetadata& metadata) override
    {
        const std::string channel_id
            = DataChannelInterfaceV1::locator_id(metadata);
        auto channel = DataChannelInterfaceV1::get_service(
            this->module(), channel_id.c_str());
        if (!channel->hololink()) {
            throw std::runtime_error(
                "While configuring RoceReceiverInterface: parent "
                "DataChannelInterface ("
                + channel_id + ") has not "
                               "been configured — call "
                               "RoceDataChannelInterfaceV1::get_service(metadata) for "
                               "this channel first");
        }
        HololinkV1::get_service(this->module(),
            HololinkInterfaceV1::locator_id(metadata).c_str())
            ->register_associated(this);
        const int64_t hsb_ip_version = metadata.get<int64_t>("hsb_ip_version");
        if (hsb_ip_version < 0 || hsb_ip_version > UINT16_MAX) {
            throw std::runtime_error(
                "While configuring RoceReceiverInterface: hsb_ip_version="
                + std::to_string(hsb_ip_version)
                + " is out of the valid 16-bit range");
        }
        hsb_ip_version_ = static_cast<uint16_t>(hsb_ip_version);
    }

    hololink_module_status_t start(
        const std::string& ibv_name,
        unsigned ibv_port,
        uint64_t cu_buffer,
        size_t cu_buffer_size,
        size_t cu_frame_size,
        size_t cu_page_size,
        unsigned pages,
        size_t metadata_offset,
        const std::string& peer_ip,
        unsigned queue_size) override
    {
        // Release any previous backing receiver — the operator's
        // stop() pairs close() with a monitor-thread join, so by the
        // time we see a fresh start() the prior monitor has exited
        // and dropping the old shared_ptr is safe.
        backing_.reset();
        backing_ = make_receiver(
            ibv_name, ibv_port, cu_buffer, cu_buffer_size, cu_frame_size,
            cu_page_size, pages, metadata_offset, peer_ip, queue_size);
        // Re-apply any registered frame-ready callback to the fresh
        // backing receiver before it starts streaming.
        apply_frame_ready();
        frame_size_ = cu_frame_size;
        page_size_ = cu_page_size;
        pages_ = pages;
        if (!backing_->start()) {
            throw std::runtime_error(
                "While starting the RoCE receiver: legacy "
                "RoceReceiver::start() failed");
        }
        return HOLOLINK_MODULE_OK;
    }

    void close() override
    {
        if (backing_) {
            backing_->close();
        }
    }

    void blocking_monitor() override
    {
        if (!backing_) {
            throw std::runtime_error(
                "While running the RoCE receiver monitor thread: "
                "receiver has not been started");
        }
        backing_->blocking_monitor();
    }

    bool get_next_frame(unsigned timeout_ms,
        RoceReceiverFrameInfoV1& info) override
    {
        if (!backing_) {
            throw std::runtime_error(
                "While reading the next frame: receiver has not been started");
        }
        hololink::operators::RoceReceiverMetadata legacy {};
        if (!backing_->get_next_frame(timeout_ms, legacy)) {
            return false;
        }
        info.frame_memory = static_cast<uint64_t>(legacy.frame_memory);
        info.metadata_memory = static_cast<uint64_t>(legacy.metadata_memory);
        info.received_frame_number = legacy.received_frame_number;
        info.frame_number = legacy.frame_number;
        info.imm_data = legacy.imm_data;
        info.received_s = legacy.received_s;
        info.received_ns = legacy.received_ns;
        info.rx_write_requests = legacy.rx_write_requests;
        info.dropped = legacy.dropped;
        return true;
    }

    bool frames_ready() override
    {
        return backing_ ? backing_->frames_ready() : false;
    }

    void set_frame_ready(std::function<void()> callback) override
    {
        frame_ready_callback_ = std::move(callback);
        apply_frame_ready();
    }

    uint32_t get_qp_number() override { return require_backing().get_qp_number(); }
    uint32_t get_rkey() override { return require_backing().get_rkey(); }
    uint64_t external_frame_memory() override { return require_backing().external_frame_memory(); }
    size_t frame_size() override { return frame_size_; }
    size_t page_size() override { return page_size_; }
    unsigned pages() override { return pages_; }
    uint16_t hsb_ip_version() { return hsb_ip_version_; }

protected:
    /* Override hook for per-board legacy-receiver subclasses. Default
     * constructs the legacy class verbatim; overrides return a
     * subclass that changes behavior the board requires. */
    virtual std::shared_ptr<hololink::operators::RoceReceiver> make_receiver(
        const std::string& ibv_name,
        unsigned ibv_port,
        uint64_t cu_buffer,
        size_t cu_buffer_size,
        size_t cu_frame_size,
        size_t cu_page_size,
        unsigned pages,
        size_t metadata_offset,
        const std::string& peer_ip,
        unsigned queue_size)
    {
        return std::make_shared<hololink::operators::RoceReceiver>(
            ibv_name.c_str(), ibv_port,
            static_cast<CUdeviceptr>(cu_buffer), cu_buffer_size, cu_frame_size,
            cu_page_size, pages, metadata_offset, peer_ip.c_str(), queue_size);
    }

private:
    hololink::operators::RoceReceiver& require_backing()
    {
        if (!backing_) {
            throw std::runtime_error(
                "While reading RoCE receiver state: receiver has not been started");
        }
        return *backing_;
    }

    // Forward the stored frame-ready callback to the backing receiver,
    // adapting its (const RoceReceiver&) signature. Clears the backing
    // hook when no callback is registered (calling an empty
    // std::function would throw, so we never install an empty wrapper).
    void apply_frame_ready()
    {
        if (!backing_) {
            return;
        }
        if (frame_ready_callback_) {
            backing_->set_frame_ready(
                [cb = frame_ready_callback_](const hololink::operators::RoceReceiver&) { cb(); });
        } else {
            backing_->set_frame_ready({});
        }
    }

    std::shared_ptr<hololink::operators::RoceReceiver> backing_;
    std::function<void()> frame_ready_callback_;
    size_t frame_size_ = 0;
    size_t page_size_ = 0;
    unsigned pages_ = 0;
    uint16_t hsb_ip_version_ = 0;
};

} // namespace hololink::module::module_core

#endif // HOLOLINK_MODULE_CORE_ROCE_RECEIVER_DEFAULT_HPP
