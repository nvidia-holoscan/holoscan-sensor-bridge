/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Out-of-line definition of LeopardVb1940_2510_Publisher::construct_roce_receiver.
// Same build shape as module/core/roce_receiver_construct.cpp: the body
// publishes the 2510 receiver with RoCE, and is empty otherwise.

#include "leopard_vb1940_2510_publisher.hpp"

#ifdef HOLOLINK_BUILD_ROCE
#include <memory>

#include "hsb_ip_2510_roce_receiver.hpp"
#endif

namespace hololink::module::leopard_vb1940 {

bool LeopardVb1940_2510_Publisher::construct_roce_receiver(
    const std::string& instance_id, const std::string& type_id)
{
#ifdef HOLOLINK_BUILD_ROCE
    using hololink::module::module_core::HsbIp2510RoceReceiverV1;
    if (!Publisher::has_type_id<HsbIp2510RoceReceiverV1>(type_id)) {
        return false;
    }
    auto impl = std::make_shared<HsbIp2510RoceReceiverV1>();
    ServicePublisher<HsbIp2510RoceReceiverV1>(shared_from_this())
        .publish(instance_id, impl);
    return true;
#else
    (void)instance_id;
    (void)type_id;
    return false;
#endif
}

} // namespace hololink::module::leopard_vb1940
