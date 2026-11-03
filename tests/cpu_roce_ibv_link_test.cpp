/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Links hololink::transport::roce and includes only ibv.hpp, so this TU sees
 * what the built archive exports. Complements cpu_roce_ibv_test.cpp, which
 * compiles ibv.cpp into itself to reach internal-linkage helpers and therefore
 * cannot link the library. Keep the split: a symbol that stops being exported
 * still passes there and fails to link here.
 */

#include "gtest/gtest.h"

#include <cstdint>
#include <iterator>
#include <string>
#include <type_traits>
#include <vector>

#include <hololink/transport/roce/ibv.hpp>

namespace hololink::transport::roce {

static_assert(sizeof(ibv::QueuePair) == sizeof(::ibv_qp*));
static_assert(sizeof(ibv::MemoryRegion) == sizeof(::ibv_mr*));
static_assert(sizeof(ibv::CompletionQueue) == sizeof(::ibv_cq*));
static_assert(sizeof(ibv::ProtectionDomain) == sizeof(::ibv_pd*));
static_assert(sizeof(ibv::Context) == sizeof(::ibv_context*));
static_assert(std::is_nothrow_move_constructible_v<ibv::QueuePair>);
static_assert(std::is_nothrow_move_assignable_v<ibv::QueuePair>);
static_assert(std::is_nothrow_move_constructible_v<ibv::MemoryRegion>);
static_assert(std::is_nothrow_move_assignable_v<ibv::MemoryRegion>);
static_assert(std::is_nothrow_move_constructible_v<ibv::CompletionQueue>);
static_assert(std::is_nothrow_move_assignable_v<ibv::CompletionQueue>);
static_assert(std::is_nothrow_move_constructible_v<ibv::ProtectionDomain>);
static_assert(std::is_nothrow_move_assignable_v<ibv::ProtectionDomain>);
static_assert(std::is_nothrow_move_constructible_v<ibv::Context>);
static_assert(std::is_nothrow_move_assignable_v<ibv::Context>);

TEST(TransportRoceHelpers, Ipv4GidUsesNetworkOrderRawBytes)
{
    const ::ibv_gid gid = ibv::ipv4_to_gid("192.0.2.17");
    const std::vector<std::uint8_t> actual(
        std::begin(gid.raw), std::end(gid.raw));
    const std::vector<std::uint8_t> expected {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 192, 0, 2, 17
    };
    EXPECT_EQ(actual, expected);
}

TEST(TransportRoceHelpers, Ipv4ToGidThrowsWithUsefulContext)
{
    try {
        (void)ibv::ipv4_to_gid("not-an-address");
        FAIL() << "expected invalid IPv4 address to throw";
    } catch (const ibv::Error& error) {
        EXPECT_NE(
            std::string(error.what()).find("not-an-address"),
            std::string::npos);
    }
}

TEST(TransportRoceHelpers, IsRoceV2Ipv4MappedInspectsRawBytes)
{
    ::ibv_gid_entry entry {};
    entry.gid = ibv::ipv4_to_gid("192.0.2.17");
    entry.gid_type = IBV_GID_TYPE_ROCE_V2;
    EXPECT_TRUE(ibv::is_roce_v2_ipv4_mapped(entry));

    entry.gid_type = IBV_GID_TYPE_ROCE_V1;
    EXPECT_FALSE(ibv::is_roce_v2_ipv4_mapped(entry));

    entry.gid_type = IBV_GID_TYPE_ROCE_V2;
    entry.gid.raw[0] = 1;
    EXPECT_FALSE(ibv::is_roce_v2_ipv4_mapped(entry));
}

} // namespace hololink::transport::roce
