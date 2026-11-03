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
#include "gui_renderer.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <utility>
#include <vector>

#include <cuda.h>

#include <holoviz/holoviz.hpp>
#include <holoviz/imgui/ui.hpp>

#include <hololink/core/logging_internal.hpp>

namespace hololink {

namespace {

    constexpr auto kFrameTimeout = std::chrono::milliseconds(100);
    constexpr auto kTeardownTimeout = std::chrono::seconds(5);
    constexpr auto kMinimizedSleep = std::chrono::milliseconds(20);
    // Per-image state is pre-sized (like HolovizOp's), so a swapchain image-count change never
    // reallocates it under an in-flight frame.
    constexpr uint32_t kMaxSwapchainImages = 8;
    // Give up on the window after this many consecutive frames fail.
    constexpr unsigned kMaxConsecutiveFailures = 100;

    // A window's place in an app's window order; windows the order doesn't name come last.
    size_t window_rank(const std::vector<std::string>& order, const std::string& name)
    {
        return static_cast<size_t>(std::find(order.begin(), order.end(), name) - order.begin());
    }

    void log_error(const holoviz::expected<void>& result, const char* what)
    {
        if (!result) {
            HSB_LOG_ERROR("ImGuiRenderer: {} failed: {}", what, result.error().message);
        }
    }

    enum class FrameResult {
        kRendered,
        kSkipped, // transient, retry with the next frame
        kFailed,
        kFatal,
    };

    // The window and everything rendering into it; created, used and destroyed on the renderer
    // thread.
    struct Display {
        holoviz::Window window;
        holoviz::Surface surface;
        holoviz::Device device;
        CUcontext cuda_context = nullptr;
        CUstream stream = nullptr;
        holoviz::Swapchain swapchain;
        holoviz::RenderPass render_pass;
        holoviz::imgui::Renderer ui;
        std::vector<holoviz::CommandList> command_lists;
        std::vector<holoviz::SubmitSyncToken> image_tokens;
        std::vector<holoviz::SubmitSyncToken> slot_tokens;
        uint64_t frame = 0;

        Display() = default;
        Display(const Display&) = delete;
        Display& operator=(const Display&) = delete;
        ~Display() { destroy(); }

        holoviz::expected<void> create(const char* title, holoviz::Extent2D size)
        {
            auto created_window = holoviz::create_window({ .title = title, .size = size, .resizable = true });
            if (!created_window) {
                return holoviz::forward_error(std::move(created_window));
            }
            window = std::move(*created_window);

            auto created_surface = window.create_surface();
            if (!created_surface) {
                return holoviz::forward_error(std::move(created_surface));
            }
            surface = std::move(*created_surface);

            // Device 0, the operators' device: the draw functions copy their data on it.
            auto created_device = holoviz::create_device_for_surface({ .cuda_device = 0 }, surface);
            if (!created_device) {
                return holoviz::forward_error(std::move(created_device));
            }
            device = std::move(*created_device);

            auto context = device.cuda_context();
            if (!context) {
                return holoviz::forward_error(std::move(context));
            }
            cuda_context = *context;
            if ((cuCtxSetCurrent(cuda_context) != CUDA_SUCCESS)
                || (cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS)) {
                stream = nullptr;
                return holoviz::make_unexpected(holoviz::Error { holoviz::ErrorCode::kCudaError,
                    "creating the GUI's CUDA stream failed" });
            }

            auto created_swapchain = device.create_swapchain({
                .surface = surface,
                .present_mode = holoviz::PresentMode::kFIFO,
                .image_count = 3,
            });
            if (!created_swapchain) {
                return holoviz::forward_error(std::move(created_swapchain));
            }
            swapchain = std::move(*created_swapchain);

            auto format = swapchain.format();
            if (!format) {
                return holoviz::forward_error(std::move(format));
            }
            auto created_render_pass = device.create_render_pass({ .render_targets = { { .format = *format } } });
            if (!created_render_pass) {
                return holoviz::forward_error(std::move(created_render_pass));
            }
            render_pass = std::move(*created_render_pass);

            for (uint32_t i = 0; i < kMaxSwapchainImages; ++i) {
                auto command_list = device.create_command_list({ .type = holoviz::CommandListType::kTransient });
                if (!command_list) {
                    return holoviz::forward_error(std::move(command_list));
                }
                command_lists.push_back(std::move(*command_list));
            }
            image_tokens.assign(kMaxSwapchainImages, holoviz::SubmitSyncToken {});
            // The ImGui renderer's buffer slots are keyed by frame count, not swapchain image.
            slot_tokens.assign(kMaxSwapchainImages + 1, holoviz::SubmitSyncToken {});

            auto created_ui = holoviz::imgui::create_renderer(device,
                { .stream = stream, .frames_in_flight = static_cast<uint32_t>(slot_tokens.size()) });
            if (!created_ui) {
                return holoviz::forward_error(std::move(created_ui));
            }
            ui = std::move(*created_ui);
            return ui.install_input_callbacks(window);
        }

        // Give an acquired image back to the swapchain without rendering it.
        void release(holoviz::AcquireSyncToken&& acquired)
        {
            log_error(ui.cancel_frame(), "imgui::Renderer::cancel_frame");
            log_error(swapchain.present(std::move(acquired), holoviz::SubmitSyncToken {}, 0),
                "Swapchain::present (unrendered)");
        }

        FrameResult render_frame(std::mutex& mutex, const std::function<void()>& build)
        {
            auto acquired = swapchain.acquire(kFrameTimeout);
            if (!acquired) {
                switch (acquired.error().code) {
                case holoviz::ErrorCode::kBackpressured: // timed out, or minimized
                case holoviz::ErrorCode::kImageCountChanged: // per-image state is pre-sized
                    return FrameResult::kSkipped;
                case holoviz::ErrorCode::kOutOfDate:
                    if (auto resized = swapchain.resize(); !resized) {
                        HSB_LOG_ERROR("ImGuiRenderer: Swapchain::resize failed: {}", resized.error().message);
                        return FrameResult::kFailed;
                    }
                    return FrameResult::kSkipped;
                default:
                    HSB_LOG_ERROR("ImGuiRenderer: Swapchain::acquire failed: {}", acquired.error().message);
                    return FrameResult::kFatal;
                }
            }
            const uint32_t image_index = acquired->image_index;
            if (image_index >= command_lists.size()) {
                HSB_LOG_ERROR("ImGuiRenderer: swapchain image {} exceeds the supported {} images", image_index,
                    command_lists.size());
                release(std::move(*acquired));
                return FrameResult::kFatal;
            }

            auto extent = swapchain.extent();
            auto& command_list = command_lists[image_index];
            const uint32_t slot = static_cast<uint32_t>(frame % slot_tokens.size());
            if (!extent || !command_list.reset(kFrameTimeout) || !ui.begin_frame(slot, slot_tokens[slot])) {
                release(std::move(*acquired));
                return FrameResult::kFailed;
            }

            holoviz::expected<holoviz::imgui::FrameSync> ui_sync;
            {
                std::lock_guard<std::mutex> lock(mutex);
                ui_sync = ui.render_ui(*extent, build);
            }
            if (!ui_sync) {
                HSB_LOG_ERROR("ImGuiRenderer: imgui::Renderer::render_ui failed: {}", ui_sync.error().message);
                release(std::move(*acquired));
                return FrameResult::kFailed;
            }

            auto image = swapchain.image(image_index);
            const holoviz::Rect2D render_area { 0, 0, extent->width, extent->height };
            if (!image || !render_pass.set_render_image(image->get(), 0, { 0.1F, 0.1F, 0.1F, 1.0F })
                || !command_list.begin() || !command_list.begin_render(render_pass, render_area)
                || !ui.record_draws(command_list, *extent) || !command_list.end_render() || !command_list.end()) {
                release(std::move(*acquired));
                return FrameResult::kFailed;
            }

            const holoviz::ExternalSyncToken upload_tokens[] = { ui_sync->upload_done };
            const holoviz::SubmitDesc submit_desc {
                .wait_submit_sync_tokens = { &image_tokens[image_index], 1 },
                .wait_external_sync_tokens = upload_tokens,
                .command_lists = { &command_list, 1 },
                .acquire_gates = { &acquired->acquire_gate, 1 },
                .present_gates = { &acquired->present_gate, 1 },
            };
            auto submitted = device.submit(submit_desc);
            if (!submitted) {
                HSB_LOG_ERROR("ImGuiRenderer: Device::submit failed: {}", submitted.error().message);
                release(std::move(*acquired));
                return FrameResult::kFatal;
            }
            image_tokens[image_index] = *submitted;
            slot_tokens[slot] = *submitted;
            ++frame;

            auto presented = swapchain.present(std::move(*acquired), *submitted);
            if (!presented && (presented.error().code != holoviz::ErrorCode::kFrameDropped)
                && (presented.error().code != holoviz::ErrorCode::kImageCountChanged)) {
                HSB_LOG_ERROR("ImGuiRenderer: Swapchain::present failed: {}", presented.error().message);
                return FrameResult::kFailed;
            }
            return FrameResult::kRendered;
        }

        // Children before the device, the device before the window.
        void destroy()
        {
            if (!device) {
                return;
            }
            if (cuda_context) {
                cuCtxSetCurrent(cuda_context);
            }
            if (ui) {
                log_error(ui.uninstall_input_callbacks(), "imgui::Renderer::uninstall_input_callbacks");
            }
            log_error(device.wait_idle(), "Device::wait_idle");
            if (ui) {
                log_error(ui.destroy(kTeardownTimeout), "imgui::Renderer::destroy");
            }
            if (stream) {
                cuStreamSynchronize(stream);
                cuStreamDestroy(stream);
                stream = nullptr;
            }
            for (auto& command_list : command_lists) {
                log_error(command_list.destroy(kTeardownTimeout), "CommandList::destroy");
            }
            command_lists.clear();
            log_error(render_pass.destroy(kTeardownTimeout), "RenderPass::destroy");
            log_error(swapchain.destroy(kTeardownTimeout), "Swapchain::destroy");
            log_error(surface.destroy(), "Surface::destroy");
            log_error(device.destroy(kTeardownTimeout), "Device::destroy");
            log_error(window.destroy(), "Window::destroy");
            cuda_context = nullptr;
        }
    };

} // namespace

ImGuiRenderer::ImGuiRenderer()
    : thread_([this] { run(); })
{
}

ImGuiRenderer::~ImGuiRenderer()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        quit_ = true;
    }
    cv_.notify_all();
    if (thread_.get_id() == std::this_thread::get_id()) {
        thread_.detach();
    } else {
        thread_.join();
    }
}

std::shared_ptr<ImGuiRenderer> ImGuiRenderer::shared()
{
    static std::mutex mutex;
    // weak, so the renderer (and its window) goes away with its last user, not at static
    // destruction time.
    static std::weak_ptr<ImGuiRenderer> instance;
    std::lock_guard<std::mutex> lock(mutex);
    auto renderer = instance.lock();
    if (!renderer) {
        renderer = std::make_shared<ImGuiRenderer>();
        instance = renderer;
    }
    return renderer;
}

void ImGuiRenderer::run()
{
    {
        // Only open a window once there is something to draw.
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return quit_ || !draw_functions_.empty(); });
        if (quit_) {
            running_ = false;
            return;
        }
    }

    Display display;
    if (auto created = display.create("Visualization", { 1024, 1280 }); !created) {
        HSB_LOG_WARN("ImGuiRenderer: can't open the GUI window ({}); continuing without the GUI",
            created.error().message);
        running_ = false;
        return;
    }
    HSB_LOG_INFO("ImGuiRenderer: GUI window created");

    unsigned consecutive_failures = 0;
    const std::function<void()> build = [this] { draw_windows(); };
    while (!quit_) {
        holoviz::poll_window_events();
        if (display.window.close_requested().value_or(false)) {
            HSB_LOG_INFO("ImGuiRenderer: GUI window closed");
            window_closed_ = true;
            break;
        }
        if (display.window.minimized().value_or(false)) {
            std::this_thread::sleep_for(kMinimizedSleep);
            continue;
        }
        // Draw functions may call the CUDA runtime; keep the device's context current for them.
        cuCtxSetCurrent(display.cuda_context);
        const FrameResult result = display.render_frame(mutex_, build);
        if (result == FrameResult::kFailed) {
            ++consecutive_failures;
        } else if (result != FrameResult::kSkipped) {
            consecutive_failures = 0;
        }
        if ((result == FrameResult::kFatal) || (consecutive_failures >= kMaxConsecutiveFailures)) {
            HSB_LOG_ERROR("ImGuiRenderer: giving up on the GUI window; continuing without the GUI");
            break;
        }
    }
    running_ = false;
    display.destroy();
}

void ImGuiRenderer::draw_windows()
{
    // Runs on the renderer thread, inside render_ui(), with mutex_ held.
    ImVec2 window_position(0, 0);
    for (auto& entry : draw_functions_) {
        ImGui::SetNextWindowPos(window_position);
        ImGui::Begin(entry.name.c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        if (!entry.failed) {
            try {
                entry.draw();
            } catch (const std::exception& e) {
                entry.failed = true;
                HSB_LOG_ERROR("ImGuiRenderer: disabling the '{}' GUI after an error: {}", entry.name, e.what());
            } catch (...) {
                entry.failed = true;
                HSB_LOG_ERROR("ImGuiRenderer: disabling the '{}' GUI after an unknown error", entry.name);
            }
        }
        window_position.y = ImGui::GetWindowPos().y + ImGui::GetWindowHeight();
        ImGui::End();
    }
}

ImGuiRenderer::Handle ImGuiRenderer::add_draw_function(const std::string& name,
    ImGuiRenderer::DrawFunction draw_func)
{
    std::lock_guard<std::mutex> lock(mutex_);
    const Handle handle = insert_window(draw_functions_, window_order_, DrawEntry { name, std::move(draw_func) });
    cv_.notify_all();
    return handle;
}

void ImGuiRenderer::set_window_order(std::vector<std::string> names)
{
    std::lock_guard<std::mutex> lock(mutex_);
    window_order_ = std::move(names);
    sort_windows(draw_functions_, window_order_);
}

ImGuiRenderer::Handle ImGuiRenderer::insert_window(
    DrawFunctions& windows, const std::vector<std::string>& order, DrawEntry entry)
{
    const size_t rank = window_rank(order, entry.name);
    const auto below = std::find_if(windows.begin(), windows.end(),
        [&order, rank](const DrawEntry& window) { return window_rank(order, window.name) > rank; });
    return windows.insert(below, std::move(entry));
}

void ImGuiRenderer::sort_windows(DrawFunctions& windows, const std::vector<std::string>& order)
{
    // std::list::sort is stable and keeps iterators (the operators' handles) valid.
    windows.sort([&order](const DrawEntry& a, const DrawEntry& b) {
        return window_rank(order, a.name) < window_rank(order, b.name);
    });
}

void ImGuiRenderer::remove_draw_function(ImGuiRenderer::Handle handle)
{
    std::lock_guard<std::mutex> lock(mutex_);
    draw_functions_.erase(handle);
}

bool ImGuiRenderer::is_running() const
{
    return running_;
}

bool ImGuiRenderer::window_closed() const
{
    return window_closed_;
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
    // ImGui reports a backward selection with start > end.
    const size_t size = str_.size();
    const size_t selection_start = std::min<size_t>(std::min(selection_start_, selection_end_), size);
    const size_t selection_end = std::min<size_t>(std::max(selection_start_, selection_end_), size);
    const size_t selection_size = selection_end - selection_start;
    const size_t cursor_position = std::min<size_t>(cursor_position_, size);

    ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0.0f, 0.5f));
    if (!selection_size)
        ImGui::BeginDisabled();
    if (ImGui::Button("Cut", ImVec2(width, 0))) {
        ImGui::SetClipboardText(str_.substr(selection_start, selection_size).c_str());
        str_ = str_.substr(0, selection_start) + str_.substr(selection_end);
        reset();
        ImGui::CloseCurrentPopup();
    }
    if (ImGui::Button("Copy", ImVec2(width, 0))) {
        ImGui::SetClipboardText(str_.substr(selection_start, selection_size).c_str());
        ImGui::CloseCurrentPopup();
    }
    if (!selection_size)
        ImGui::EndDisabled();

    std::string clipboard_text(ImGui::GetClipboardText() ? ImGui::GetClipboardText() : "");
    if (clipboard_text.empty())
        ImGui::BeginDisabled();
    if (ImGui::Button("Paste", ImVec2(width, 0))) {
        if (selection_size)
            str_ = str_.substr(0, selection_start) + clipboard_text + str_.substr(selection_end);
        else
            str_ = str_.substr(0, cursor_position) + clipboard_text + str_.substr(cursor_position);
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
