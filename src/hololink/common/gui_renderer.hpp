/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#ifndef SRC_HOLOLINK_GUI_RENDERER_HPP
#define SRC_HOLOLINK_GUI_RENDERER_HPP

#include <atomic>
#include <condition_variable>
#include <functional>
#include <list>
#include <mutex>
#include <stdexcept>
#include <thread>

#include <holoscan/holoscan.hpp> // Needed for the YAML::convert

struct ImGuiInputTextCallbackData;
namespace hololink {

/**
 * The Holoviz operator does NOT support Dear ImGui functionality.
 * Ref: https://docs.nvidia.com/holoscan/sdk-user-guide/visualization.html#imgui-layers
 * This class meant to workaround it by creating a dedicated Dear ImGui renderer outside
 * the Holoscan operators.
 * The class maintain its own rendering thread, that so, does not depends on the Holoscan
 * scheduler.
 * Operators can "inject" ImGui drawing commands to the renderer by calling the
 * AddDrawFunction function.
 */
class ImGuiRenderer {
public:
    using DrawFunction = std::function<void()>;
    using DrawFunctions = std::list<std::pair<std::string, DrawFunction>>;
    using Handle = DrawFunctions::reverse_iterator;

    ImGuiRenderer();
    ~ImGuiRenderer();

    Handle add_draw_function(const std::string& name, DrawFunction draw_func);
    void remove_draw_function(Handle);
    bool is_running() const; // The renderer stops running if the GUI windows is closed by the user

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    DrawFunctions draw_functions_;
    bool initialized_ = false;
    std::atomic<bool> quit_ { false };
    std::thread thread_;
};

class CutCopyPaste {
public:
    CutCopyPaste(std::string& str);
    CutCopyPaste(CutCopyPaste&) = delete;
    CutCopyPaste& operator=(CutCopyPaste&) = delete;
    void reset();
    void operator()(float width = 0);

    static int InputTextCallback(ImGuiInputTextCallbackData* data);
    static int InputTextCallback(CutCopyPaste* self, ImGuiInputTextCallbackData* data);

private:
    int cursor_position_ = 0;
    int selection_start_ = 0;
    int selection_end_ = 0;
    std::string& str_;
};

} // namespace hololink

// The ImGuiRenderer does not meant to be configured by a YAML file.
// The following implementation is just to satisfy the compiler but will throw
// an exception if it will be used.
template <>
struct YAML::convert<hololink::ImGuiRenderer*> {
    static Node encode(hololink::ImGuiRenderer*)
    {
        throw std::runtime_error("Unsupported");
        return Node {};
    }

    static bool decode(const Node&, hololink::ImGuiRenderer*)
    {
        throw std::runtime_error("Unsupported");
        return false;
    }
};

#endif /* SRC_HOLOLINK_GUI_RENDERER_HPP */
