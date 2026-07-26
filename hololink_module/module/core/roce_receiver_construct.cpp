/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Out-of-line definition of HsbLitePublisher::construct_roce_receiver.
// The publisher header declares it (and the construct_service chain
// always calls it); only the body here varies with the build's RoCE
// capability. Defining it requires the full HsbLitePublisher class,
// which transitively includes a board supplement header, so this TU is
// listed in each per-board module's SOURCES (which carries that include
// path) rather than compiled into hololink::module. HOLOLINK_BUILD_ROCE
// reaches it as a PUBLIC define inherited from hololink::module.
//
// With HOLOLINK_BUILD_ROCE the body publishes the functional
// RoceReceiverV1 (this TU is then the place that names the ibverbs
// receiver); without it the method is a no-op that returns false and
// publishes nothing, so a consumer asking for the RoCE receiver gets a
// clean get_service miss rather than a do-nothing stub.

#include "hsb_lite_publisher.hpp"

#ifdef HOLOLINK_BUILD_ROCE
#include <memory>

#include "roce_receiver_default.hpp"
#endif

namespace hololink::module::module_core {

bool HsbLitePublisher::construct_roce_receiver(
    const std::string& instance_id, const std::string& type_id)
{
#ifdef HOLOLINK_BUILD_ROCE
    if (!Publisher::has_type_id<RoceReceiverV1>(type_id)) {
        return false;
    }
    auto impl = std::make_shared<RoceReceiverV1>();
    ServicePublisher<RoceReceiverV1>(shared_from_this())
        .publish(instance_id, impl);
    return true;
#else
    (void)instance_id;
    (void)type_id;
    return false;
#endif
}

} // namespace hololink::module::module_core
