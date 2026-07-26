/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_CORE_HSB_LITE_2510_LINUX_RECEIVER_HPP
#define HOLOLINK_MODULE_CORE_HSB_LITE_2510_LINUX_RECEIVER_HPP

#include <cstddef>
#include <cstdint>
#include <memory>

#include <hololink/operators/linux_receiver/linux_receiver.hpp>

#include "hololink/module/service.hpp"

#include "linux_receiver_default.hpp"

namespace hololink::module::module_core {

/* Subclass of the legacy hololink::operators::LinuxReceiver — the hook
 * point for behaviors that differ on the pre-0x2603 receiver IP from the
 * modern HSB-Lite. The software receiver derives the frame's page index
 * from the RDMA immediate data, whose bit layout changed at 0x2603; the
 * base masks 0xFFF, the 2510 layout masks 0xFF. Mirrors
 * HsbLite2510RoceReceiver (which the ibverbs path uses for the same
 * reason). */
class HsbLite2510LinuxReceiver
    : public hololink::operators::LinuxReceiver {
public:
    using hololink::operators::LinuxReceiver::LinuxReceiver;

    uint32_t page_from_imm(uint32_t imm_data) override
    {
        return imm_data & 0xFF;
    }
};

/* V1 receiver wrapper for compat-id 2510 boards. Identical to the
 * default LinuxReceiverV1 except that make_receiver() picks
 * HsbLite2510LinuxReceiver when the FPGA's hsb_ip_version predates the
 * modern receiver IP. */
class HsbLite2510LinuxReceiverV1
    : public LinuxReceiverV1,
      public Service<HsbLite2510LinuxReceiverV1> {
public:
    static constexpr const char* type_id = "linux_receiver.hsb_lite_2510.v1";
    using Service<HsbLite2510LinuxReceiverV1>::get_service;
    using Service<HsbLite2510LinuxReceiverV1>::for_each_type_id;
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
        return std::make_shared<HsbLite2510LinuxReceiver>(
            static_cast<CUdeviceptr>(cu_buffer), cu_buffer_size,
            cu_page_size, pages, socket, received_address_offset,
            queue_size);
    }
};

} // namespace hololink::module::module_core

#endif // HOLOLINK_MODULE_CORE_HSB_LITE_2510_LINUX_RECEIVER_HPP
