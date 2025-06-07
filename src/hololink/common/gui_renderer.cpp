/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "gui_renderer.hpp"

#include <utility>

#include <imgui.h>

#include <holoviz/holoviz.hpp>

namespace hololink {

ImGuiRenderer::ImGuiRenderer()
    : thread_([this] {
        holoscan::viz::InstanceHandle instance {};
        // Keep running until the object is destructed or the windows is closed
        while (true) {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return quit_ || !draw_functions_.empty(); });

            if (quit_)
                break; // The object is being destructed

            // Lazy initialization
            // Only initialized when there is something to render.
            if (!initialized_) {
                instance = holoscan::viz::Create();
                holoscan::viz::SetCurrent(instance);
                holoscan::viz::Init(1024, 1280, "Visualization");
                initialized_ = true;
            }

            if (holoscan::viz::WindowShouldClose())
                break; // The window was closed

            // Each draw function gets its own sub-window
            holoscan::viz::Begin();
            holoscan::viz::BeginImGuiLayer();

            ImVec2 window_position(0, 0);
            for (auto& pair : draw_functions_) {
                ImGui::SetNextWindowPos(window_position);
                ImGui::Begin(pair.first.c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize);
                pair.second(); // Draw
                window_position.y = ImGui::GetWindowPos().y + ImGui::GetWindowHeight();
                ImGui::End();
            }

            holoscan::viz::EndLayer();
            holoscan::viz::End();

            std::this_thread::yield();
        }

        holoscan::viz::Shutdown(instance);
        quit_ = true;
    })
{
}

ImGuiRenderer::~ImGuiRenderer()
{
    std::unique_lock<std::mutex> lock(mutex_);
    quit_ = true;
    cv_.notify_all();
    lock.unlock();
    thread_.join();
}

ImGuiRenderer::Handle ImGuiRenderer::add_draw_function(const std::string& name,
    ImGuiRenderer::DrawFunction draw_func)
{
    std::lock_guard<std::mutex> lock(mutex_);
    draw_functions_.emplace_back(name, std::move(draw_func));
    cv_.notify_all();
    return draw_functions_.rbegin();
}

void ImGuiRenderer::remove_draw_function(ImGuiRenderer::Handle handle)
{
    std::lock_guard<std::mutex> lock(mutex_);
    // https://stackoverflow.com/questions/1830158/how-to-call-erase-with-a-reverse-iterator
    draw_functions_.erase(std::next(handle).base());
}

bool ImGuiRenderer::is_running() const
{
    return !quit_;
}

CutCopyPaste::CutCopyPaste(std::string& str)
    : str_(str)
{
}

void CutCopyPaste::reset()
{
    cursor_position_ = 0;
    selection_start_ = 0;
    selection_end_ = 0;
}

void CutCopyPaste::operator()(float width)
{
    auto selection_size = selection_end_ - selection_start_;
    ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0.0f, 0.5f));
    if (!selection_size)
        ImGui::BeginDisabled();
    if (ImGui::Button("Cut", ImVec2(width, 0))) {
        ImGui::SetClipboardText(str_.substr(selection_start_, selection_end_).c_str());
        str_ = str_.substr(0, selection_start_) + str_.substr(selection_end_);
        reset();
        ImGui::CloseCurrentPopup();
    }
    if (ImGui::Button("Copy", ImVec2(width, 0))) {
        ImGui::SetClipboardText(str_.substr(selection_start_, selection_end_).c_str());
        ImGui::CloseCurrentPopup();
    }
    if (!selection_size)
        ImGui::EndDisabled();

    std::string clipboard_text(ImGui::GetClipboardText() ? ImGui::GetClipboardText() : "");
    if (clipboard_text.empty())
        ImGui::BeginDisabled();
    if (ImGui::Button("Paste", ImVec2(width, 0))) {
        if (selection_size)
            str_ = str_.substr(0, selection_start_) + clipboard_text + (cursor_position_ < str_.size() ? str_.substr(selection_end_) : std::string());
        else
            str_ = str_.substr(0, cursor_position_) + clipboard_text + (cursor_position_ < str_.size() ? str_.substr(cursor_position_) : std::string());
        reset();
        ImGui::CloseCurrentPopup();
    }
    if (clipboard_text.empty())
        ImGui::EndDisabled();
    ImGui::PopStyleVar();
}

int CutCopyPaste::InputTextCallback(ImGuiInputTextCallbackData* data)
{
    return InputTextCallback(reinterpret_cast<CutCopyPaste*>(data->UserData), data);
}

int CutCopyPaste::InputTextCallback(CutCopyPaste* self, ImGuiInputTextCallbackData* data)
{
    self->cursor_position_ = data->CursorPos;
    self->selection_start_ = data->SelectionStart;
    self->selection_end_ = data->SelectionEnd;
    return 0;
}
} // namespace hololink
