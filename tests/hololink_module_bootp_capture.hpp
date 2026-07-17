/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Test-only V1 service the bootp stub module publishes so the host
 * gtest can observe what update_metadata received. Lives in tests/ —
 * not part of the framework or any production module.
 */

#ifndef TESTS_HOLOLINK_MODULE_BOOTP_CAPTURE_HPP
#define TESTS_HOLOLINK_MODULE_BOOTP_CAPTURE_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "hololink/module/service.hpp"

namespace test {

class BootpCaptureV1
    : public hololink::module::Service<BootpCaptureV1> {
public:
    static constexpr const char* type_id = "test.bootp_capture.v1";

    virtual ~BootpCaptureV1() = default;

    /* Wait until the stub's EnumerationInterfaceV1::update_metadata
     * has been invoked, or `timeout_ms` elapses. Returns true if the
     * call was observed. */
    virtual bool wait_for_metadata(unsigned timeout_ms) = 0;

    /* Snapshot of the raw bootp packet bytes the host passed through
     * `Adapter::enumerate(metadata, raw_packet, raw_packet_len)`. */
    virtual std::vector<std::uint8_t> raw_packet() const = 0;

    /* Snapshot fields the legacy parser stamps into the metadata
     * before the supplement override sees it. */
    virtual std::string fpga_uuid() const = 0;
    virtual std::string serial_number() const = 0;
    virtual std::int64_t compat_id() const = 0;
    virtual std::int64_t data_plane() const = 0;
    virtual std::string peer_ip() const = 0;
};

} // namespace test

#endif // TESTS_HOLOLINK_MODULE_BOOTP_CAPTURE_HPP
