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

#include <gtest/gtest.h>

#include "hsb_flasher_args.hpp"

#include <getopt.h>
#include <string>
#include <vector>

using namespace hsb_flasher;

// ── Helpers ──────────────────────────────────────────────────────────

/// Builds an argc/argv pair from a list of strings.
/// The first element should be the program name.
class Args {
public:
    Args(std::initializer_list<std::string> args)
        : strings_(args)
    {
        for (auto& s : strings_)
            ptrs_.push_back(s.data());
        ptrs_.push_back(nullptr);
    }

    int argc() const { return static_cast<int>(strings_.size()); }
    char** argv() { return ptrs_.data(); }

private:
    std::vector<std::string> strings_;
    std::vector<char*> ptrs_;
};

/// Reset getopt global state before each parse_arguments() call.
class ParseArgumentsTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        optind = 0;
    }

    hsb_flasher_context ctx_ {};
};

// ── Standard mode ────────────────────────────────────────────────────

// Valid standard mode args with IP and target version
TEST_F(ParseArgumentsTest, StandardMode_ValidArgs)
{
    Args a { "hsb_flasher", "-H", "192.168.0.2", "-t", "2507" };
    ASSERT_TRUE(parse_arguments(a.argc(), a.argv(), ctx_));
    EXPECT_EQ(ctx_.hololink_ip, "192.168.0.2");
    EXPECT_EQ(ctx_.target_version, "2507");
    EXPECT_FALSE(ctx_.direct_flash.enabled());
    EXPECT_EQ(ctx_.log_level, hsb_flasher_log_level::INFO);
    EXPECT_FLOAT_EQ(ctx_.timeout, 3.0f);
}

// Verbose flag, UUID override, and custom timeout are parsed correctly
TEST_F(ParseArgumentsTest, StandardMode_VerboseAndUuidOverride)
{
    Args a { "hsb_flasher", "-H", "10.0.0.1", "-t", "2510", "-v",
        "-u", "my-uuid", "-e", "5" };
    ASSERT_TRUE(parse_arguments(a.argc(), a.argv(), ctx_));
    EXPECT_EQ(ctx_.log_level, hsb_flasher_log_level::DEBUG);
    EXPECT_EQ(ctx_.fpga_uuid_override, "my-uuid");
    EXPECT_FLOAT_EQ(ctx_.timeout, 5.0f);
}

// Long-form flags (--hololink, --target-version) work the same as short flags
TEST_F(ParseArgumentsTest, StandardMode_LongFlags)
{
    Args a { "hsb_flasher", "--hololink", "192.168.0.3",
        "--target-version", "2603" };
    ASSERT_TRUE(parse_arguments(a.argc(), a.argv(), ctx_));
    EXPECT_EQ(ctx_.hololink_ip, "192.168.0.3");
    EXPECT_EQ(ctx_.target_version, "2603");
}

// ── Direct flash mode ────────────────────────────────────────────────

// Direct mode with --cpnx and --flash-version (no --clnx)
TEST_F(ParseArgumentsTest, DirectMode_CpnxAndFlashVersion)
{
    Args a { "hsb_flasher", "-H", "192.168.0.2",
        "--cpnx", "cpnx.bin", "--flash-version", "2507" };
    ASSERT_TRUE(parse_arguments(a.argc(), a.argv(), ctx_));
    EXPECT_TRUE(ctx_.direct_flash.enabled());
    EXPECT_EQ(ctx_.direct_flash.cpnx_path, "cpnx.bin");
    EXPECT_EQ(ctx_.direct_flash.flash_version, "2507");
    EXPECT_TRUE(ctx_.direct_flash.clnx_path.empty());
}

// Direct mode with all three args: --clnx, --cpnx, --flash-version
TEST_F(ParseArgumentsTest, DirectMode_WithClnx)
{
    Args a { "hsb_flasher", "-H", "192.168.0.2",
        "--clnx", "clnx.bin", "--cpnx", "cpnx.bin",
        "--flash-version", "2504" };
    ASSERT_TRUE(parse_arguments(a.argc(), a.argv(), ctx_));
    EXPECT_TRUE(ctx_.direct_flash.enabled());
    EXPECT_EQ(ctx_.direct_flash.clnx_path, "clnx.bin");
    EXPECT_EQ(ctx_.direct_flash.cpnx_path, "cpnx.bin");
    EXPECT_EQ(ctx_.direct_flash.flash_version, "2504");
}

// ── Failure cases ────────────────────────────────────────────────────

// No -H/--hololink provided
TEST_F(ParseArgumentsTest, MissingIP_ExpectFailure)
{
    Args a { "hsb_flasher", "-t", "2507" };
    EXPECT_FALSE(parse_arguments(a.argc(), a.argv(), ctx_));
}

// IP address is not a valid IPv4 format
TEST_F(ParseArgumentsTest, InvalidIP_ExpectFailure)
{
    Args a { "hsb_flasher", "-H", "not.an.ip", "-t", "2507" };
    EXPECT_FALSE(parse_arguments(a.argc(), a.argv(), ctx_));
}

// IP address octets exceed 255
TEST_F(ParseArgumentsTest, IPOutOfRange_ExpectFailure)
{
    Args a { "hsb_flasher", "-H", "999.999.999.999", "-t", "2507" };
    EXPECT_FALSE(parse_arguments(a.argc(), a.argv(), ctx_));
}

// Standard mode with IP but no -t/--target-version
TEST_F(ParseArgumentsTest, MissingTargetVersion_ExpectFailure)
{
    Args a { "hsb_flasher", "-H", "192.168.0.2" };
    EXPECT_FALSE(parse_arguments(a.argc(), a.argv(), ctx_));
}

// Target version contains non-hex characters
TEST_F(ParseArgumentsTest, InvalidHexVersion_ExpectFailure)
{
    Args a { "hsb_flasher", "-H", "192.168.0.2", "-t", "xyz!" };
    EXPECT_FALSE(parse_arguments(a.argc(), a.argv(), ctx_));
}

// --cpnx provided without --flash-version (valid: uses device's discovered version)
TEST_F(ParseArgumentsTest, DirectMode_CpnxOnly)
{
    Args a { "hsb_flasher", "-H", "192.168.0.2", "--cpnx", "cpnx.bin" };
    ASSERT_TRUE(parse_arguments(a.argc(), a.argv(), ctx_));
    EXPECT_TRUE(ctx_.direct_flash.enabled());
    EXPECT_EQ(ctx_.direct_flash.cpnx_path, "cpnx.bin");
    EXPECT_TRUE(ctx_.direct_flash.flash_version.empty());
}

// --flash-version provided without --cpnx
TEST_F(ParseArgumentsTest, PartialDirectArgs_FlashVersionOnly_ExpectFailure)
{
    Args a { "hsb_flasher", "-H", "192.168.0.2", "--flash-version", "2507" };
    EXPECT_FALSE(parse_arguments(a.argc(), a.argv(), ctx_));
}

// --clnx provided without --cpnx
TEST_F(ParseArgumentsTest, PartialDirectArgs_ClnxOnly_ExpectFailure)
{
    Args a { "hsb_flasher", "-H", "192.168.0.2", "--clnx", "clnx.bin" };
    EXPECT_FALSE(parse_arguments(a.argc(), a.argv(), ctx_));
}

// Negative enumeration timeout
TEST_F(ParseArgumentsTest, NegativeTimeout_ExpectFailure)
{
    Args a { "hsb_flasher", "-H", "192.168.0.2", "-t", "2507", "-e", "-1" };
    EXPECT_FALSE(parse_arguments(a.argc(), a.argv(), ctx_));
}

// Zero enumeration timeout
TEST_F(ParseArgumentsTest, ZeroTimeout_ExpectFailure)
{
    Args a { "hsb_flasher", "-H", "192.168.0.2", "-t", "2507", "-e", "0" };
    EXPECT_FALSE(parse_arguments(a.argc(), a.argv(), ctx_));
}

// Direct mode --flash-version with non-hex value
TEST_F(ParseArgumentsTest, DirectMode_InvalidFlashVersion_ExpectFailure)
{
    Args a { "hsb_flasher", "-H", "192.168.0.2",
        "--cpnx", "cpnx.bin", "--flash-version", "zzzz" };
    EXPECT_FALSE(parse_arguments(a.argc(), a.argv(), ctx_));
}

// ── direct_flash_config unit tests ───────────────────────────────────

// enabled() requires cpnx_path; flash_version is optional
TEST(DirectFlashConfig, EnabledRequiresCpnx)
{
    direct_flash_config cfg;
    EXPECT_FALSE(cfg.enabled());

    cfg.cpnx_path = "cpnx.bin";
    EXPECT_TRUE(cfg.enabled());

    cfg.flash_version = "2507";
    EXPECT_TRUE(cfg.enabled());
}

// has_any() returns true if any direct flash field is set
TEST(DirectFlashConfig, HasAnyDetectsPartialArgs)
{
    direct_flash_config cfg;
    EXPECT_FALSE(cfg.has_any());

    cfg.clnx_path = "clnx.bin";
    EXPECT_TRUE(cfg.has_any());

    cfg = {};
    cfg.flash_version = "2507";
    EXPECT_TRUE(cfg.has_any());
}
