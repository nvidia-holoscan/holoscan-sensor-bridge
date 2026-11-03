/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Out-of-line definition of LeopardVb1940_2510_Publisher::construct_linux_receiver.
// Same build shape as module/core/linux_receiver_construct.cpp: unlike the
// RoCE receiver, the software receiver needs no ibverbs, so the body is
// unconditional and always publishes the 2510 receiver.

#include "leopard_vb1940_2510_publisher.hpp"

#include <memory>

#include "hsb_ip_2510_linux_receiver.hpp"

namespace hololink::module::leopard_vb1940 {

bool LeopardVb1940_2510_Publisher::construct_linux_receiver(
    const std::string& instance_id, const std::string& type_id)
{
    using hololink::module::module_core::HsbIp2510LinuxReceiverV1;
    if (!Publisher::has_type_id<HsbIp2510LinuxReceiverV1>(type_id)) {
        return false;
    }
    auto impl = std::make_shared<HsbIp2510LinuxReceiverV1>();
    ServicePublisher<HsbIp2510LinuxReceiverV1>(shared_from_this())
        .publish(instance_id, impl);
    return true;
}

} // namespace hololink::module::leopard_vb1940
