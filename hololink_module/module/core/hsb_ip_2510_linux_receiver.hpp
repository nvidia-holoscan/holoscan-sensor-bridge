/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_CORE_HSB_IP_2510_LINUX_RECEIVER_HPP
#define HOLOLINK_MODULE_CORE_HSB_IP_2510_LINUX_RECEIVER_HPP

#include <cstddef>
#include <cstdint>
#include <memory>

#include <hololink/operators/linux_receiver/linux_receiver.hpp>

#include "hololink/module/service.hpp"

#include "linux_receiver_default.hpp"

namespace hololink::module::module_core {

/* Subclass of the legacy hololink::operators::LinuxReceiver — the hook
 * point for behaviors that differ on the pre-0x2603 receiver IP from the
 * modern one. The software receiver derives the frame's page index
 * from the RDMA immediate data, whose bit layout changed at 0x2603; the
 * base masks 0xFFF, the 2510 layout masks 0xFF. Mirrors
 * HsbIp2510RoceReceiver (which the ibverbs path uses for the same
 * reason). */
class HsbIp2510LinuxReceiver
    : public hololink::operators::LinuxReceiver {
public:
    using hololink::operators::LinuxReceiver::LinuxReceiver;

    uint32_t page_from_imm(uint32_t imm_data) override
    {
        return imm_data & 0xFF;
    }
};

/* V1 receiver wrapper for boards whose FPGA predates the modern
 * receiver IP. Identical to the default LinuxReceiverV1 except that
 * make_receiver() picks HsbIp2510LinuxReceiver when the FPGA's
 * hsb_ip_version predates it. */
class HsbIp2510LinuxReceiverV1
    : public LinuxReceiverV1,
      public Service<HsbIp2510LinuxReceiverV1> {
public:
    static constexpr const char* type_id = "linux_receiver.hsb_ip_2510.v1";
    using Service<HsbIp2510LinuxReceiverV1>::get_service;
    using Service<HsbIp2510LinuxReceiverV1>::for_each_type_id;
    /* Chain walks through module_core::LinuxReceiverV1 (which itself
     * chains to LinuxReceiverInterfaceV1), then emits this subclass's
     * own type_id. Three keys total. */
    using ServiceAlias = LinuxReceiverV1;

protected:
    std::shared_ptr<hololink::operators::LinuxReceiver> make_receiver(
        uint64_t cu_buffer,
        size_t cu_buffer_size,
        size_t cu_page_size,
        unsigned pages,
        int socket,
        uint64_t received_address_offset,
        unsigned queue_size) override
    {
        if (hsb_ip_version() >= 0x2603) {
            return std::make_shared<hololink::operators::LinuxReceiver>(
                static_cast<CUdeviceptr>(cu_buffer), cu_buffer_size,
                cu_page_size, pages, socket, received_address_offset,
                queue_size);
        }

        // Use the older style
        return std::make_shared<HsbIp2510LinuxReceiver>(
            static_cast<CUdeviceptr>(cu_buffer), cu_buffer_size,
            cu_page_size, pages, socket, received_address_offset,
            queue_size);
    }
};

} // namespace hololink::module::module_core

#endif // HOLOLINK_MODULE_CORE_HSB_IP_2510_LINUX_RECEIVER_HPP
