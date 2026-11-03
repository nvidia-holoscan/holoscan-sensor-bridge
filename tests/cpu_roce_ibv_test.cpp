/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "gtest/gtest.h"

#include <memory>
#include <stdexcept>
#include <string>

// Compile ibv.cpp into this TU so anonymous-namespace validate_mtu is visible.
// Do not also link hololink::transport::roce (duplicate symbols).
// Complements cpu_roce_ibv_link_test.cpp, which links the archive and can
// only see exported ibv.hpp symbols.
#include "../src/hololink/transport/roce/ibv.cpp"

namespace {

constexpr int kHifMtus[] = { 0, 1024, 1500, 2048, 2200, 4096, 4200, 9000 };

} // namespace

// Tests live in the same namespace as the anonymous helpers from ibv.cpp.
namespace hololink::transport::roce::ibv {

namespace {

    // nullopt when the host has no device or cannot enumerate one. An empty
    // list stays distinct in FirstNameMatchesEntryZero.
    std::optional<std::string> first_device_name()
    {
        try {
            return DeviceList().first_name();
        } catch (const Error&) {
            return std::nullopt;
        }
    }

} // namespace

TEST(ReceiveWorkRequest, NullSgeCreatesCreditOnlyWorkRequest)
{
    ReceiveWorkRequest request(7, nullptr);

    EXPECT_EQ(request.wr_id, 7u);
    EXPECT_EQ(request.sg_list, nullptr);
    EXPECT_EQ(request.num_sge, 0);
}

TEST(VerbsResourceErrors, EnomemIncludesMemlockLimits)
{
    try {
        throw_verbs_resource_error("test registration", ENOMEM);
        FAIL() << "expected memory registration error";
    } catch (const Error& error) {
        const std::string message { error.what() };
        EXPECT_NE(message.find("test registration"), std::string::npos);
        EXPECT_NE(message.find("RLIMIT_MEMLOCK"), std::string::npos);
        EXPECT_NE(message.find("possible pinned-memory constraint"),
            std::string::npos);
        EXPECT_NE(message.find("soft_bytes="), std::string::npos);
    }
}

// Annotation is diagnostic context, not proof of the errno's cause.
TEST(VerbsResourceErrors, EpermIncludesMemlockLimits)
{
    try {
        throw_verbs_resource_error("test queue creation", EPERM);
        FAIL() << "expected verbs resource error";
    } catch (const Error& error) {
        const std::string message { error.what() };
        EXPECT_NE(message.find("test queue creation"), std::string::npos);
        EXPECT_NE(message.find("RLIMIT_MEMLOCK"), std::string::npos);
        EXPECT_NE(message.find("possible pinned-memory constraint"),
            std::string::npos);
    }
}

TEST(VerbsResourceErrors, UnrelatedErrnoDoesNotNameMemlockLimits)
{
    for (const int err : { EINVAL, EIO }) {
        try {
            throw_verbs_resource_error("test other error", err);
            FAIL() << "expected verbs resource error for errno " << err;
        } catch (const Error& error) {
            const std::string message { error.what() };
            EXPECT_NE(message.find("test other error"), std::string::npos)
                << err;
            EXPECT_EQ(message.find("RLIMIT_MEMLOCK"), std::string::npos)
                << message;
        }
    }
}

TEST(ValidateMtu, NegativeHifMtuUsesVerbsActiveMtu)
{
    EXPECT_NO_THROW(validate_mtu(-1, IBV_MTU_256));
}

TEST(ValidateMtu, UnknownActiveMtuThrows)
{
    for (int hif_mtu : kHifMtus) {
        try {
            validate_mtu(hif_mtu, static_cast<ibv_mtu>(0));
            FAIL() << "expected Unknown active MTU for hif_mtu=" << hif_mtu;
        } catch (const std::runtime_error& e) {
            EXPECT_NE(std::string(e.what()).find("Unknown active MTU"), std::string::npos)
                << "hif_mtu=" << hif_mtu << " what=" << e.what();
        }
    }
}

TEST(ValidateMtu, ActiveMtu256)
{
    for (int hif_mtu : kHifMtus) {
        if (hif_mtu == 0) {
            EXPECT_THROW(validate_mtu(hif_mtu, IBV_MTU_256), std::runtime_error)
                << "hif_mtu=" << hif_mtu;
        } else {
            EXPECT_NO_THROW(validate_mtu(hif_mtu, IBV_MTU_256)) << "hif_mtu=" << hif_mtu;
        }
    }
}

TEST(ValidateMtu, ActiveMtu512)
{
    for (int hif_mtu : kHifMtus) {
        if (hif_mtu == 0) {
            EXPECT_THROW(validate_mtu(hif_mtu, IBV_MTU_512), std::runtime_error)
                << "hif_mtu=" << hif_mtu;
        } else {
            EXPECT_NO_THROW(validate_mtu(hif_mtu, IBV_MTU_512)) << "hif_mtu=" << hif_mtu;
        }
    }
}

TEST(ValidateMtu, ActiveMtu1024)
{
    for (int hif_mtu : kHifMtus) {
        if (hif_mtu == 0 || hif_mtu == 1024) {
            EXPECT_THROW(validate_mtu(hif_mtu, IBV_MTU_1024), std::runtime_error)
                << "hif_mtu=" << hif_mtu;
        } else {
            EXPECT_NO_THROW(validate_mtu(hif_mtu, IBV_MTU_1024)) << "hif_mtu=" << hif_mtu;
        }
    }
}

TEST(ValidateMtu, ActiveMtu2048)
{
    for (int hif_mtu : kHifMtus) {
        if (hif_mtu == 0 || hif_mtu == 1024 || hif_mtu == 1500 || hif_mtu == 2048) {
            EXPECT_THROW(validate_mtu(hif_mtu, IBV_MTU_2048), std::runtime_error)
                << "hif_mtu=" << hif_mtu;
        } else {
            EXPECT_NO_THROW(validate_mtu(hif_mtu, IBV_MTU_2048)) << "hif_mtu=" << hif_mtu;
        }
    }
}

TEST(ValidateMtu, ActiveMtu4096)
{
    for (int hif_mtu : kHifMtus) {
        if (hif_mtu == 0 || hif_mtu == 1024 || hif_mtu == 1500 || hif_mtu == 2048
            || hif_mtu == 2200 || hif_mtu == 4096) {
            EXPECT_THROW(validate_mtu(hif_mtu, IBV_MTU_4096), std::runtime_error)
                << "hif_mtu=" << hif_mtu;
        } else {
            EXPECT_NO_THROW(validate_mtu(hif_mtu, IBV_MTU_4096)) << "hif_mtu=" << hif_mtu;
        }
    }
}

TEST(CompletionQueue, DefaultPollRejectsNullCq)
{
    CompletionQueue cq;
    ::ibv_wc wc {};
    EXPECT_THROW(cq.poll(&wc, 1), Error);
}

TEST(CompletionQueue, InvalidPollArgumentsThrow)
{
    // Default CQ: !ptr_ is checked first, so these are smoke coverage of
    // rejected inputs, not independent proofs of the buffer/count branches.
    CompletionQueue cq;
    ::ibv_wc wc {};
    EXPECT_THROW(cq.poll(nullptr, 1), Error);
    EXPECT_THROW(cq.poll(&wc, 0), Error);
    EXPECT_THROW(cq.poll(&wc, -1), Error);
}

TEST(CompletionQueue, DiscardPendingOnNullCqIsHarmless)
{
    CompletionQueue cq;
    EXPECT_NO_THROW(cq.discard_pending());
}

TEST(Context, DefaultPollAsyncEventThrows)
{
    EXPECT_THROW(Context {}.poll_async_event(), Error);
}

TEST(Gid, Ipv4RoundTrip)
{
    const ::ibv_gid gid = ipv4_to_gid("192.0.2.1");
    EXPECT_EQ(gid_to_ipv4(gid), "192.0.2.1");

    ::ibv_gid_entry entry {};
    entry.gid_type = IBV_GID_TYPE_ROCE_V2;
    entry.gid = gid;
    EXPECT_TRUE(is_roce_v2_ipv4_mapped(entry));
}

TEST(Gid, RejectsNonIpv4Text)
{
    for (const char* text : { "not-an-ip", "2001:db8::1", "" }) {
        try {
            ipv4_to_gid(text);
            FAIL() << text;
        } catch (const Error& error) {
            EXPECT_NE(std::string(error.what()).find("Unable to convert IPv4 address"),
                std::string::npos)
                << text << " " << error.what();
        }
    }
}

TEST(Gid, MappedCheckRejectsOtherGids)
{
    ::ibv_gid_entry entry {};
    entry.gid = ipv4_to_gid("192.0.2.1");
    entry.gid_type = IBV_GID_TYPE_IB;
    EXPECT_FALSE(is_roce_v2_ipv4_mapped(entry));

    entry.gid_type = IBV_GID_TYPE_ROCE_V2;
    entry.gid.raw[0] = 1;
    EXPECT_FALSE(is_roce_v2_ipv4_mapped(entry));
}

TEST(CompletionQueue, SinglePollRejectsNullCq)
{
    CompletionQueue cq;
    EXPECT_THROW(cq.poll(), Error);
}

TEST(Context, HostInterfaceRequiresADevice)
{
    try {
        Context {}.get_host_if_name();
        FAIL() << "null context returned an interface";
    } catch (const Error& error) {
        EXPECT_NE(std::string(error.what()).find("Device is null"), std::string::npos)
            << error.what();
    }
}

TEST(Context, QueryPortRejectsPortZeroWithoutADevice)
{
    try {
        Context {}.query_port(0);
        FAIL() << "port 0 was accepted";
    } catch (const Error& error) {
        EXPECT_NE(std::string(error.what()).find("does not fit a non-zero uint8_t"),
            std::string::npos)
            << error.what();
    }
}

TEST(Context, QueryGidExRejectsInvalidIdentifiersWithoutADevice)
{
    try {
        Context {}.query_gid_ex(0, 0);
        FAIL() << "invalid GID query was accepted";
    } catch (const Error& error) {
        EXPECT_NE(std::string(error.what()).find("Invalid GID query identifiers"),
            std::string::npos)
            << error.what();
    }
}

TEST(DeviceList, FirstNameMatchesEntryZero)
{
    // Skip only when the host cannot enumerate devices. Failures from
    // first_name() or name() are the behavior this test checks.
    std::unique_ptr<DeviceList> devices;
    try {
        devices = std::make_unique<DeviceList>();
    } catch (const Error& error) {
        GTEST_SKIP() << error.what();
    }

    if (devices->size() == 0) {
        EXPECT_FALSE(devices->first_name().has_value());
        EXPECT_THROW(devices->name(0), Error);
        return;
    }
    const std::optional<std::string> first = devices->first_name();
    ASSERT_TRUE(first.has_value());
    EXPECT_FALSE(first.value().empty());
    EXPECT_EQ(first.value(), devices->name(0));
    EXPECT_THROW(devices->name(devices->size()), Error);
}

TEST(Context, RejectsEmptyAndMissingDeviceNames)
{
    try {
        DeviceList devices;
        (void)devices;
    } catch (const Error& error) {
        GTEST_SKIP() << error.what();
    }

    try {
        Context context("");
        FAIL() << "empty device name opened a context";
    } catch (const Error& error) {
        EXPECT_NE(std::string(error.what()).find("must not be empty"), std::string::npos)
            << error.what();
    }

    try {
        Context context("cpu-roce-device-that-does-not-exist");
        FAIL() << "missing device opened a context";
    } catch (const Error& error) {
        EXPECT_NE(std::string(error.what()).find("Failed to find IB device"), std::string::npos)
            << error.what();
    }
}

TEST(Context, OpenedDeviceAnswersPortQueries)
{
    const std::optional<std::string> device_name = first_device_name();
    if (!device_name)
        GTEST_SKIP() << "no IB device";

    Context context(device_name.value());
    const ::ibv_device_attr attributes = context.query_device();
    ASSERT_GE(attributes.phys_port_cnt, 1);

    // Port 1 exists on every device, whatever the link state. Its GID table
    // can be empty.
    const ::ibv_port_attr port_attributes = context.query_port(1);
    EXPECT_NE(port_attributes.state, IBV_PORT_NOP);
    if (port_attributes.gid_tbl_len > 0)
        EXPECT_NO_THROW((void)context.query_gid(1, 0));
}

TEST(Context, RoceDeviceExposesGidIndexAndHostInterface)
{
    const std::optional<std::string> device_name = first_device_name();
    if (!device_name)
        GTEST_SKIP() << "no IB device";

    Context context(device_name.value());
    int gid_index = -1;
    try {
        gid_index = context.query_gid_roce_v2_ip();
    } catch (const Error& error) {
        // Not an IPv4 RoCE v2 device.
        GTEST_SKIP() << error.what();
    }
    EXPECT_GE(gid_index, 0);
    EXPECT_LE(gid_index, 255);
    // A RoCE device is backed by an ethernet interface.
    EXPECT_FALSE(context.get_host_if_name().empty());
}

} // namespace hololink::transport::roce::ibv
