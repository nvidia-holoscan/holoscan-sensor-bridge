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

// The examples' command-line parser: values are parsed whole and strictly, and a bad value ends
// the program the way argp reports its own errors (a message and exit status EX_USAGE).

#include <hololink/common/holoargs.hpp>

#include <gtest/gtest.h>

#include <sysexits.h>

#include <cstdint>
#include <string>
#include <vector>

namespace {

using namespace hololink::args;

VariablesMap parse(std::vector<std::string> args)
{
    OptionsDescription options("holoargs test");
    // clang-format off
    options.add_options()
        ("count", value<uint32_t>()->default_value(7), "An unsigned count")
        ("offset", value<int>()->default_value(0), "A signed offset")
        ("scale", value<double>()->default_value(1.0), "A scale")
        ("title", value<std::string>()->default_value("default"), "A title")
        ("flag", bool_switch()->default_value(false), "A switch")
        ;
    // clang-format on
    args.insert(args.begin(), "holoargs_test");
    std::vector<char*> argv;
    for (auto& arg : args) {
        argv.push_back(arg.data());
    }
    return Parser().parse_command_line(static_cast<int>(argv.size()), argv.data(), options);
}

TEST(HoloargsTest, ParsesValidValues)
{
    auto variables = parse({ "--count=12", "--offset", "-3", "--scale=0.5", "--title", "two words", "--flag" });
    EXPECT_EQ(variables["count"].as<uint32_t>(), 12U);
    EXPECT_EQ(variables["offset"].as<int>(), -3);
    EXPECT_DOUBLE_EQ(variables["scale"].as<double>(), 0.5);
    EXPECT_EQ(variables["title"].as<std::string>(), "two words");
    EXPECT_TRUE(variables["flag"].as<bool>());
}

TEST(HoloargsTest, AcceptsSurroundingSpaces)
{
    auto variables = parse({ "--count", " 5 ", "--title", " padded " });
    EXPECT_EQ(variables["count"].as<uint32_t>(), 5U);
    EXPECT_EQ(variables["title"].as<std::string>(), " padded ");
}

TEST(HoloargsTest, AppliesDefaults)
{
    auto variables = parse({});
    EXPECT_EQ(variables["count"].as<uint32_t>(), 7U);
    EXPECT_EQ(variables["offset"].as<int>(), 0);
    EXPECT_EQ(variables["title"].as<std::string>(), "default");
    EXPECT_FALSE(variables["flag"].as<bool>());
}

TEST(HoloargsDeathTest, RejectsANonNumericValue)
{
    EXPECT_EXIT(parse({ "--count=abc" }), testing::ExitedWithCode(EX_USAGE),
        "invalid value 'abc' for option '--count'");
}

TEST(HoloargsDeathTest, RejectsTrailingCharacters)
{
    EXPECT_EXIT(parse({ "--offset=10x" }), testing::ExitedWithCode(EX_USAGE),
        "invalid value '10x' for option '--offset'");
}

TEST(HoloargsDeathTest, RejectsANegativeUnsignedValue)
{
    EXPECT_EXIT(parse({ "--count", "-1" }), testing::ExitedWithCode(EX_USAGE),
        "invalid value '-1' for option '--count'");
}

TEST(HoloargsDeathTest, RejectsAnEmptyNumber)
{
    EXPECT_EXIT(parse({ "--scale=" }), testing::ExitedWithCode(EX_USAGE),
        "invalid value '' for option '--scale'");
}

} // namespace
