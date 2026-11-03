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

// infiniband_device_for_interface() and DataChannel::infiniband_device() against a fake
// /sys/class/infiniband tree; no NIC needed.

#include <unistd.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <gtest/gtest.h>

#include <hololink/core/data_channel.hpp>
#include <hololink/core/metadata.hpp>
#include <hololink/core/networking.hpp>

namespace {

namespace fs = std::filesystem;
using hololink::core::infiniband_device_for_interface;
using Device = std::optional<std::pair<std::string, uint32_t>>;

// The channel of a board on the loopback address, which the routing table reaches through "lo";
// nothing contacts it. interface: the one it enumerated on, if any.
hololink::DataChannel loopback_channel(const std::string& interface)
{
    hololink::Metadata metadata;
    metadata["peer_ip"] = std::string("127.0.0.1");
    metadata["vp_mask"] = int64_t(0);
    metadata["data_plane"] = int64_t(0);
    metadata["sensor"] = int64_t(0);
    metadata["sif_address"] = int64_t(0);
    metadata["vp_address"] = int64_t(0);
    metadata["hif_address"] = int64_t(0);
    metadata["fpga_uuid"] = std::string("00000000-0000-0000-0000-000000000000");
    if (!interface.empty()) {
        metadata["interface"] = interface;
    }
    return hololink::DataChannel(
        metadata, [](const hololink::Metadata&) { return std::shared_ptr<hololink::Hololink> {}; });
}

class FakeInfiniband : public ::testing::Test {
protected:
    void SetUp() override
    {
        root_ = fs::temp_directory_path() / ("hololink_networking_test_" + std::to_string(::getpid()));
        fs::remove_all(root_);
        fs::create_directories(root_);
    }

    void TearDown() override { fs::remove_all(root_); }

    // A RoCE device: port 1's GID table entries are bound to netdev (GIDs exist for the
    // link-local IPv6 address even when the interface has no IPv4 address).
    void add_roce_device(const std::string& device, const std::string& netdev, unsigned port = 1)
    {
        const fs::path ndevs = root_ / device / "ports" / std::to_string(port) / "gid_attrs" / "ndevs";
        fs::create_directories(ndevs);
        std::ofstream(ndevs / "0") << netdev << '\n';
        std::ofstream(ndevs / "1") << netdev << '\n';
        // An unused GID index; real sysfs fails the read.
        std::ofstream(ndevs / "2");
        fs::create_directories(root_ / device / "device" / "net" / netdev);
    }

    std::string root() const { return root_.string(); }

private:
    fs::path root_;
};

// The DGX Spark: each QSFP port is two PCI functions, each with its own netdev and device; the
// boards' IP addresses are on one netdev of each port.
TEST_F(FakeInfiniband, PicksTheDeviceOfEachInterface)
{
    add_roce_device("mlx5_0", "enp1s0f0np0");
    add_roce_device("mlx5_1", "enp1s0f1np1");
    add_roce_device("mlx5_2", "enP2p1s0f0np0");
    add_roce_device("mlx5_3", "enP2p1s0f1np1");

    EXPECT_EQ(infiniband_device_for_interface("enp1s0f0np0", root()), Device(std::make_pair("mlx5_0", 1U)));
    EXPECT_EQ(infiniband_device_for_interface("enP2p1s0f1np1", root()), Device(std::make_pair("mlx5_3", 1U)));
    // The twin netdevs, which carry no IP address, still map to their own devices.
    EXPECT_EQ(infiniband_device_for_interface("enp1s0f1np1", root()), Device(std::make_pair("mlx5_1", 1U)));
    EXPECT_EQ(infiniband_device_for_interface("enP2p1s0f0np0", root()), Device(std::make_pair("mlx5_2", 1U)));
}

TEST_F(FakeInfiniband, ReportsThePort)
{
    add_roce_device("mlx5_0", "eth0", 2);
    EXPECT_EQ(infiniband_device_for_interface("eth0", root()), Device(std::make_pair("mlx5_0", 2U)));
}

TEST_F(FakeInfiniband, FallsBackToThePciFunction)
{
    // No GID table (e.g. not a RoCE port): the device whose PCI function owns the netdev, port 1.
    fs::create_directories(fs::path(root()) / "mlx5_0" / "device" / "net" / "eth0");
    EXPECT_EQ(infiniband_device_for_interface("eth0", root()), Device(std::make_pair("mlx5_0", 1U)));
}

TEST_F(FakeInfiniband, NoMatch)
{
    add_roce_device("mlx5_0", "enp1s0f0np0");
    EXPECT_EQ(infiniband_device_for_interface("eth9", root()), std::nullopt);
    EXPECT_EQ(infiniband_device_for_interface("", root()), std::nullopt);
    EXPECT_EQ(infiniband_device_for_interface("enp1s0f0np0", root() + "/missing"), std::nullopt);
}

// What a RoCE operator without an ibv_name uses: the device on the interface the board
// enumerated on, or the routing table's when the board recorded none.
TEST_F(FakeInfiniband, ChannelPicksTheDeviceOnItsBoardsInterface)
{
    add_roce_device("mlx5_0", "enp1s0f0np0");
    add_roce_device("mlx5_1", "lo");
    const auto mlx5_1 = std::make_pair(std::string("mlx5_1"), 1U);
    EXPECT_EQ(loopback_channel("lo").infiniband_device(root()), mlx5_1);
    EXPECT_EQ(loopback_channel("").infiniband_device(root()), mlx5_1);
}

// An enumeration interface without an IPv4 address isn't the one DataChannel programs the board
// to send to; the routing table's is, and so is its device.
TEST_F(FakeInfiniband, ChannelFollowsTheRoutingTableOffAnInterfaceWithoutAddress)
{
    add_roce_device("mlx5_0", "hsbtest0"); // no such interface on the host, so no IPv4 address
    add_roce_device("mlx5_1", "lo");
    EXPECT_EQ(loopback_channel("hsbtest0").infiniband_device(root()), std::make_pair(std::string("mlx5_1"), 1U));
}

TEST_F(FakeInfiniband, ChannelNamesTheInterfaceNoDeviceIsBoundTo)
{
    add_roce_device("mlx5_0", "enp1s0f0np0");
    try {
        loopback_channel("lo").infiniband_device(root());
        FAIL() << "a device was found for interface \"lo\"";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("interface \"lo\""), std::string::npos) << e.what();
        EXPECT_NE(std::string(e.what()).find("board at 127.0.0.1"), std::string::npos) << e.what();
    }
}

} // namespace
