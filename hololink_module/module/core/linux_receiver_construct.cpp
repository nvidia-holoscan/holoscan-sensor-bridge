/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Out-of-line definition of HsbIpPublisher::construct_linux_receiver.
// The publisher header declares it (and the construct_service chain
// always calls it); the definition lives here so the header stays free
// of the legacy LinuxReceiver's <cuda.h> dependency
// (linux_receiver_default.hpp pulls it in). This TU is listed in each
// per-board module's SOURCES rather than compiled into
// hololink::module_core — same placement as roce_receiver_construct.cpp.
//
// Unlike the RoCE receiver, the software receiver needs no ibverbs and
// is not gated on HOLOLINK_BUILD_ROCE: the body always publishes a
// functional LinuxReceiverV1.

#include "hsb_ip_publisher.hpp"

#include <memory>

#include "linux_receiver_default.hpp"

namespace hololink::module::module_core {

bool HsbIpPublisher::construct_linux_receiver(
    const std::string& instance_id, const std::string& type_id)
{
    if (!Publisher::has_type_id<LinuxReceiverV1>(type_id)) {
        return false;
    }
    auto impl = std::make_shared<LinuxReceiverV1>();
    ServicePublisher<LinuxReceiverV1>(shared_from_this())
        .publish(instance_id, impl);
    return true;
}

} // namespace hololink::module::module_core
