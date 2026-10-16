/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_CORE_HSB_IP_2510_ROCE_RECEIVER_HPP
#define HOLOLINK_MODULE_CORE_HSB_IP_2510_ROCE_RECEIVER_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <hololink/operators/roce_receiver/roce_receiver.hpp>

#include "hololink/module/service.hpp"

#include "roce_receiver_default.hpp"

namespace hololink::module::module_core {

/* Subclass of the legacy hololink::operators::RoceReceiver — the hook
 * point for behaviors that differ on the pre-0x2603 receiver IP from
 * the modern one. The legacy class already declares the PSN
 * helpers virtual for exactly this purpose. */
class HsbIp2510RoceReceiver
    : public hololink::operators::RoceReceiver {
public:
    using hololink::operators::RoceReceiver::RoceReceiver;

    uint32_t page_from_imm(uint32_t imm_data) override
    {
        return imm_data & 0xFF;
    }

    uint32_t psn_from_imm(uint32_t imm_data) override
    {
        return (imm_data >> 8) & 0xFFFFFF;
    }

    bool same_psn(uint32_t frame_metadata_psn, uint32_t received_psn) override
    {
        return (frame_metadata_psn == received_psn);
    }
};

/* V1 receiver wrapper for boards whose FPGA predates the modern
 * receiver IP. Identical to the default RoceReceiverV1 except that
 * make_receiver() picks HsbIp2510RoceReceiver when the FPGA's
 * hsb_ip_version predates it. */
class HsbIp2510RoceReceiverV1
    : public RoceReceiverV1,
      public Service<HsbIp2510RoceReceiverV1> {
public:
    static constexpr const char* type_id = "roce_receiver.hsb_ip_2510.v1";
    using Service<HsbIp2510RoceReceiverV1>::get_service;
    using Service<HsbIp2510RoceReceiverV1>::for_each_type_id;
    /* Chain walks through module_core::RoceReceiverV1 (which itself
     * chains to RoceReceiverInterfaceV1), then emits this subclass's
     * own type_id. Three keys total. */
    using ServiceAlias = RoceReceiverV1;

protected:
    std::shared_ptr<hololink::operators::RoceReceiver> make_receiver(
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
        if (hsb_ip_version() >= 0x2603) {
            return std::make_shared<hololink::operators::RoceReceiver>(
                ibv_name.c_str(), ibv_port,
                static_cast<CUdeviceptr>(cu_buffer), cu_buffer_size, cu_frame_size,
                cu_page_size, pages, metadata_offset, peer_ip.c_str(), queue_size);
        }

        // Use the older style
        return std::make_shared<HsbIp2510RoceReceiver>(
            ibv_name.c_str(), ibv_port,
            static_cast<CUdeviceptr>(cu_buffer), cu_buffer_size, cu_frame_size,
            cu_page_size, pages, metadata_offset, peer_ip.c_str(), queue_size);
    }
};

} // namespace hololink::module::module_core

#endif // HOLOLINK_MODULE_CORE_HSB_IP_2510_ROCE_RECEIVER_HPP
