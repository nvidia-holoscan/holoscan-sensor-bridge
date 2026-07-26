/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Out-of-line definition of HsbLite2510Publisher::construct_roce_receiver
// (the compat-0x2510 override). Compiled only into module/hsb_lite_2510,
// the sole module that instantiates HsbLite2510Publisher. Same build
// shape as roce_receiver_construct.cpp: the body publishes the
// 2510-specific receiver subclass with RoCE, and is empty otherwise.

#include "hsb_lite_2510_publisher.hpp"

#ifdef HOLOLINK_BUILD_ROCE
#include <memory>

#include "hsb_lite_2510_roce_receiver.hpp"
#endif

namespace hololink::module::module_core {

bool HsbLite2510Publisher::construct_roce_receiver(
    const std::string& instance_id, const std::string& type_id)
{
#ifdef HOLOLINK_BUILD_ROCE
    if (!Publisher::has_type_id<HsbLite2510RoceReceiverV1>(type_id)) {
        return false;
    }
    auto impl = std::make_shared<HsbLite2510RoceReceiverV1>();
    ServicePublisher<HsbLite2510RoceReceiverV1>(shared_from_this())
        .publish(instance_id, impl);
    return true;
#else
    (void)instance_id;
    (void)type_id;
    return false;
#endif
}

} // namespace hololink::module::module_core
