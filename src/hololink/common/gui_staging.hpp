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

#ifndef SRC_HOLOLINK_COMMON_GUI_STAGING_HPP
#define SRC_HOLOLINK_COMMON_GUI_STAGING_HPP

#include <cstddef>
#include <functional>
#include <mutex>
#include <vector>

#include <cuda_runtime_api.h>

namespace hololink {

/**
 * Hands device data from an operator's compute() to its GUI draw function without either side
 * blocking on the GPU and without the GUI holding on to runtime tensors (which would pin their
 * pool slots). compute() enqueues copies into a pinned host buffer on its own CUDA stream; the
 * draw function picks the batch up once the copies have landed.
 *
 * At most one batch is in flight: until the GUI has taken the last one, stage() skips, so a GUI
 * slower than the pipeline just shows fewer updates. Thread-safe.
 */
class GuiStaging {
public:
    GuiStaging();
    ~GuiStaging();
    GuiStaging(const GuiStaging&) = delete;
    GuiStaging& operator=(const GuiStaging&) = delete;

    /// compute() side: if the GUI took the previous batch, call enqueue(host_buffer) to enqueue
    /// asynchronous copies of up to \p count floats into the buffer on \p stream, and return true.
    /// Returns false (without calling enqueue) while the previous batch is still pending. \p tag
    /// travels with the batch, e.g. to describe how its floats are laid out.
    bool stage(size_t count, const std::function<void(float* host_buffer)>& enqueue, cudaStream_t stream,
        std::vector<int> tag = {});

    /// GUI side: if a staged batch has landed, copy it into \p batch (and its tag into \p tag)
    /// and return true.
    bool take(std::vector<float>& batch, std::vector<int>* tag = nullptr);

private:
    enum class State {
        kIdle,
        kFilling,
        kInFlight,
    };

    std::mutex mutex_;
    State state_ = State::kIdle;
    float* buffer_ = nullptr;
    size_t capacity_ = 0;
    size_t count_ = 0;
    std::vector<int> tag_;
    cudaEvent_t event_ = nullptr;
};

} // namespace hololink

#endif /* SRC_HOLOLINK_COMMON_GUI_STAGING_HPP */
