/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_CORE_LINUX_RECEIVER_DEFAULT_HPP
#define HOLOLINK_MODULE_CORE_LINUX_RECEIVER_DEFAULT_HPP

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include <netinet/in.h>
#include <sys/socket.h>

#include <cuda.h>

#include "hololink/module/data_channel.hpp"
#include "hololink/module/linux_receiver.hpp"
#include "hololink/module/status.h"

#include <hololink/operators/linux_receiver/linux_receiver.hpp>

#include "hololink_default.hpp"

namespace hololink::module::module_core {

/* LinuxReceiverInterfaceV1 backed by hololink::operators::LinuxReceiver
 * — the user-space software receiver that reassembles HSB's RoCEv2 UDP
 * stream over a datagram socket. Constructed by the per-board supplement
 * as a shell at enumerate time; start(...) is when the underlying legacy
 * receiver is actually constructed (which is when its qp_number / rkey
 * become valid) and its run() thread becomes runnable.
 *
 * Per-board overrides plug in by deriving from this class and overriding
 * make_receiver() to return a subclass of the legacy receiver. No
 * first-party override exists today (the 2510 metadata-layout diff
 * routes through FrameMetadataInterfaceV1 / the channel's configure_roce
 * override, not the receiver), but the seam matches the RoCE pattern. */
class LinuxReceiverV1 : public LinuxReceiverInterfaceV1,
                        public Service<LinuxReceiverV1> {
public:
    static constexpr const char* type_id = "linux_receiver.module_core.v1";
    using Service<LinuxReceiverV1>::get_service;
    using Service<LinuxReceiverV1>::for_each_type_id;
    using ServiceAlias = LinuxReceiverInterfaceV1;

    LinuxReceiverV1() = default;

    /* Anchor-fetch guard, identical in shape to RoceReceiverV1: the
     * per-channel receiver routes through the per-channel
     * DataChannelInterfaceV1 anchor, which the application must have
     * constructed first (via LinuxDataChannelInterfaceV1::get_service).
     * Runs exactly once per instance — framework's
     * ConfigurableService<T>::get_service wraps this in std::call_once. */
    void configure(const EnumerationMetadata& metadata) override
    {
        const std::string channel_id
            = DataChannelInterfaceV1::locator_id(metadata);
        auto channel = DataChannelInterfaceV1::get_service(
            this->module(), channel_id.c_str());
        if (!channel->hololink()) {
            throw std::runtime_error(
                "While configuring LinuxReceiverInterface: parent "
                "DataChannelInterface ("
                + channel_id
                + ") has not been configured — call "
                  "LinuxDataChannelInterfaceV1::get_service(metadata) for "
                  "this channel first");
        }
        HololinkV1::get_service(this->module(),
            HololinkInterfaceV1::locator_id(metadata).c_str())
            ->register_associated(this);
        const int64_t hsb_ip_version = metadata.get<int64_t>("hsb_ip_version");
        if (hsb_ip_version < 0 || hsb_ip_version > UINT16_MAX) {
            throw std::runtime_error(
                "While configuring LinuxReceiverInterface: hsb_ip_version="
                + std::to_string(hsb_ip_version)
                + " is out of the valid 16-bit range");
        }
        hsb_ip_version_ = static_cast<uint16_t>(hsb_ip_version);
    }

    hololink_module_status_t start(
        int data_socket,
        uint64_t cu_buffer,
        size_t cu_buffer_size,
        size_t cu_frame_size,
        size_t cu_page_size,
        unsigned pages,
        unsigned queue_size) override
    {
        // Release any previous backing receiver — the operator's stop()
        // pairs close() with a monitor-thread join, so by the time we
        // see a fresh start() the prior run() has exited and dropping
        // the old shared_ptr is safe.
        backing_.reset();
        // HSB is configured with distal_memory_address_start = 0; the
        // software receiver adds the local buffer address itself, so
        // received_address_offset == cu_buffer.
        backing_ = make_receiver(
            cu_buffer, cu_buffer_size, cu_page_size, pages, data_socket,
            cu_buffer, queue_size);
        // Re-apply any registered frame-ready callback to the fresh
        // backing receiver before it starts streaming.
        apply_frame_ready();
        socket_ = data_socket;
        frame_size_ = cu_frame_size;
        page_size_ = cu_page_size;
        pages_ = pages;
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
        require_backing().run();
    }

    bool get_next_frame(unsigned timeout_ms,
        LinuxReceiverFrameInfoV1& info, void* cuda_stream) override
    {
        hololink::operators::LinuxReceiverMetadata legacy {};
        if (!require_backing().get_next_frame(
                timeout_ms, legacy, static_cast<CUstream>(cuda_stream))) {
            return false;
        }
        info.frame_memory = static_cast<uint64_t>(legacy.frame_memory);
        info.frame_packets_received = legacy.frame_packets_received;
        info.frame_bytes_received = legacy.frame_bytes_received;
        info.received_frame_number = legacy.received_frame_number;
        info.frame_number = legacy.frame_number;
        info.imm_data = legacy.imm_data;
        info.received_s = legacy.received_s;
        info.received_ns = legacy.received_ns;
        info.packets_dropped = legacy.packets_dropped;
        // Metadata the software receiver already decoded from the stream.
        info.flags = legacy.frame_metadata.flags;
        info.psn = legacy.frame_metadata.psn;
        info.crc = legacy.frame_metadata.crc;
        info.timestamp_ns = legacy.frame_metadata.timestamp_ns;
        info.timestamp_s = legacy.frame_metadata.timestamp_s;
        info.bytes_written = legacy.frame_metadata.bytes_written;
        info.metadata_ns = legacy.frame_metadata.metadata_ns;
        info.metadata_s = legacy.frame_metadata.metadata_s;
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

    /* The bound socket's UDP port — the destination HSB sends to. Read
     * from the socket the operator handed to start() (and that
     * LinuxDataChannelInterfaceV1::configure_socket bound). */
    uint32_t local_port() override
    {
        if (socket_ < 0) {
            throw std::runtime_error(
                "While reading the Linux receiver local port: receiver has "
                "not been started");
        }
        sockaddr_in addr {};
        socklen_t addr_len = sizeof(addr);
        if (getsockname(socket_, reinterpret_cast<sockaddr*>(&addr), &addr_len)
            < 0) {
            throw std::runtime_error(
                "While reading the Linux receiver local port: getsockname "
                "failed");
        }
        return ntohs(addr.sin_port);
    }

    size_t frame_size() override { return frame_size_; }
    size_t page_size() override { return page_size_; }
    unsigned pages() override { return pages_; }

protected:
    uint16_t hsb_ip_version() { return hsb_ip_version_; }

    /* Override hook for per-board legacy-receiver subclasses. Default
     * constructs the legacy class verbatim. */
    virtual std::shared_ptr<hololink::operators::LinuxReceiver> make_receiver(
        uint64_t cu_buffer,
        size_t cu_buffer_size,
        size_t cu_page_size,
        unsigned pages,
        int socket,
        uint64_t received_address_offset,
        unsigned queue_size)
    {
        return std::make_shared<hololink::operators::LinuxReceiver>(
            static_cast<CUdeviceptr>(cu_buffer), cu_buffer_size, cu_page_size,
            pages, socket, received_address_offset, queue_size);
    }

private:
    hololink::operators::LinuxReceiver& require_backing()
    {
        if (!backing_) {
            throw std::runtime_error(
                "While accessing the Linux receiver: receiver has not been "
                "started");
        }
        return *backing_;
    }

    // Forward the stored frame-ready callback to the backing receiver,
    // adapting its (const LinuxReceiver&) signature. Clears the backing
    // hook when no callback is registered (calling an empty
    // std::function would throw, so we never install an empty wrapper).
    void apply_frame_ready()
    {
        if (!backing_) {
            return;
        }
        if (frame_ready_callback_) {
            backing_->set_frame_ready(
                [cb = frame_ready_callback_](const hololink::operators::LinuxReceiver&) { cb(); });
        } else {
            backing_->set_frame_ready({});
        }
    }

    std::shared_ptr<hololink::operators::LinuxReceiver> backing_;
    std::function<void()> frame_ready_callback_;
    int socket_ = -1;
    size_t frame_size_ = 0;
    size_t page_size_ = 0;
    unsigned pages_ = 0;
    uint16_t hsb_ip_version_ = 0;
};

} // namespace hololink::module::module_core

#endif // HOLOLINK_MODULE_CORE_LINUX_RECEIVER_DEFAULT_HPP
