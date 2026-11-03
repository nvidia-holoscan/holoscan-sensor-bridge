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

// ImGuiRenderer's window order, on a plain list; no display needed.

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hololink/common/gui_renderer.hpp>

namespace {

using hololink::ImGuiRenderer;

std::vector<std::string> names(const ImGuiRenderer::DrawFunctions& windows)
{
    std::vector<std::string> result;
    for (const auto& window : windows) {
        result.push_back(window.name);
    }
    return result;
}

const std::vector<std::string> kSignalGeneratorOrder {
    "Signal Generator", "IQ Encoder", "IQ Decoder", "Signal Viewer"
};

// The signal generator's operators register in start(); HSDK5 started them in this order.
TEST(ImGuiRendererOrder, InsertsInTheDeclaredOrder)
{
    ImGuiRenderer::DrawFunctions windows;
    for (const char* name : { "IQ Decoder", "IQ Encoder", "Signal Generator", "Signal Viewer" }) {
        ImGuiRenderer::insert_window(windows, kSignalGeneratorOrder, { name, [] {} });
    }
    EXPECT_EQ(names(windows), kSignalGeneratorOrder);
}

TEST(ImGuiRendererOrder, UnlistedWindowsComeLastInRegistrationOrder)
{
    ImGuiRenderer::DrawFunctions windows;
    for (const char* name : { "Other B", "Signal Viewer", "Other A", "Signal Generator" }) {
        ImGuiRenderer::insert_window(windows, kSignalGeneratorOrder, { name, [] {} });
    }
    EXPECT_EQ(names(windows),
        (std::vector<std::string> { "Signal Generator", "Signal Viewer", "Other B", "Other A" }));
}

TEST(ImGuiRendererOrder, NoOrderKeepsRegistrationOrder)
{
    ImGuiRenderer::DrawFunctions windows;
    for (const char* name : { "IQ Decoder", "IQ Encoder", "Signal Generator" }) {
        ImGuiRenderer::insert_window(windows, {}, { name, [] {} });
    }
    EXPECT_EQ(names(windows),
        (std::vector<std::string> { "IQ Decoder", "IQ Encoder", "Signal Generator" }));
}

// An order declared after some windows were added reorders them; the handles stay valid.
TEST(ImGuiRendererOrder, SortKeepsHandles)
{
    ImGuiRenderer::DrawFunctions windows;
    const auto decoder = ImGuiRenderer::insert_window(windows, {}, { "IQ Decoder", [] {} });
    ImGuiRenderer::insert_window(windows, {}, { "Other", [] {} });
    const auto generator = ImGuiRenderer::insert_window(windows, {}, { "Signal Generator", [] {} });

    ImGuiRenderer::sort_windows(windows, kSignalGeneratorOrder);
    EXPECT_EQ(names(windows),
        (std::vector<std::string> { "Signal Generator", "IQ Decoder", "Other" }));
    EXPECT_EQ(decoder->name, "IQ Decoder");
    EXPECT_EQ(generator->name, "Signal Generator");
    windows.erase(decoder);
    EXPECT_EQ(names(windows), (std::vector<std::string> { "Signal Generator", "Other" }));
}

} // namespace
