/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef SRC_HOLOLINK_GUI_RENDERER_HPP
#define SRC_HOLOLINK_GUI_RENDERER_HPP

#include <atomic>
#include <condition_variable>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct ImGuiInputTextCallbackData;
namespace hololink {

/**
 * HolovizOp has no Dear ImGui layer, so this renderer opens its own window and runs its own
 * rendering thread (on HoloViz's ImGui backend), independent of the Holoscan scheduler.
 * Operators "inject" ImGui drawing commands by registering a draw function; each one gets its
 * own ImGui window, stacked vertically. Operators register in start(), whose order the runtime
 * picks, so an app that wants a fixed layout declares it with set_window_order().
 *
 * The window is created lazily, when the first draw function is registered. If it can't be
 * created (e.g. no display), the renderer logs a warning and stops running; operators then
 * skip their GUI work.
 *
 * Dear ImGui's current context is process-global, so a process must use one renderer: get it
 * through shared().
 */
class ImGuiRenderer {
public:
    using DrawFunction = std::function<void()>;
    struct DrawEntry {
        std::string name;
        DrawFunction draw;
        bool failed = false;
    };
    using DrawFunctions = std::list<DrawEntry>;
    using Handle = DrawFunctions::iterator;

    ImGuiRenderer();
    ~ImGuiRenderer();
    ImGuiRenderer(const ImGuiRenderer&) = delete;
    ImGuiRenderer& operator=(const ImGuiRenderer&) = delete;

    /// The process-wide renderer, created on first use and destroyed with its last owner.
    static std::shared_ptr<ImGuiRenderer> shared();

    Handle add_draw_function(const std::string& name, DrawFunction draw_func);
    /// Once this returns, the draw function is not running and will never run again.
    void remove_draw_function(Handle handle);
    /// Stack the windows named in `names` first, in that order, and the others below them in
    /// the order they were added.
    void set_window_order(std::vector<std::string> names);

    /// The ordering behind add_draw_function() and set_window_order(), on a plain list (so a
    /// test needs no window): insert_window() adds `entry` above the first window that `order`
    /// puts below it; sort_windows() reorders `windows` by `order`, keeping the order among
    /// windows it ranks alike.
    static Handle insert_window(
        DrawFunctions& windows, const std::vector<std::string>& order, DrawEntry entry);
    static void sort_windows(DrawFunctions& windows, const std::vector<std::string>& order);

    /// False once the window was closed or couldn't be created.
    bool is_running() const;
    /// True only when the user closed the window.
    bool window_closed() const;

private:
    void run();
    void draw_windows();

    std::mutex mutex_;
    std::condition_variable cv_;
    DrawFunctions draw_functions_;
    std::vector<std::string> window_order_;
    std::atomic<bool> quit_ { false };
    std::atomic<bool> running_ { true };
    std::atomic<bool> window_closed_ { false };
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

#endif /* SRC_HOLOLINK_GUI_RENDERER_HPP */
