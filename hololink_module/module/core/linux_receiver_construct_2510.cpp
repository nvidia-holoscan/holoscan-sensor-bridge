/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Out-of-line definition of HsbLite2510Publisher::construct_linux_receiver
// (the compat-0x2510 override). Compiled only into module/hsb_lite_2510,
// the sole module that instantiates HsbLite2510Publisher. Same build
// shape as linux_receiver_construct.cpp: unlike the RoCE receiver, the
// software receiver needs no ibverbs, so the body is unconditional and
// always publishes the 2510-specific receiver subclass.

#include "hsb_lite_2510_publisher.hpp"

#include <memory>

#include "hsb_lite_2510_linux_receiver.hpp"

namespace hololink::module::module_core {

bool HsbLite2510Publisher::construct_linux_receiver(
    const std::string& instance_id, const std::string& type_id)
{
    if (!Publisher::has_type_id<HsbLite2510LinuxReceiverV1>(type_id)) {
        return false;
    }
    auto impl = std::make_shared<HsbLite2510LinuxReceiverV1>();
    ServicePublisher<HsbLite2510LinuxReceiverV1>(shared_from_this())
        .publish(instance_id, impl);
    return true;
}

} // namespace hololink::module::module_core
