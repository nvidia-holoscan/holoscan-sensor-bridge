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
#include "gui_staging.hpp"

#include <exception>
#include <utility>

#include <hololink/common/cuda_error.hpp>
#include <hololink/core/logging_internal.hpp>

namespace hololink {

GuiStaging::GuiStaging()
{
    CUDA_CHECK(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
}

GuiStaging::~GuiStaging()
{
    // A batch may still be landing: wait for it before freeing its buffer.
    if (state_ == State::kInFlight) {
        cudaEventSynchronize(event_);
    }
    if (buffer_) {
        cudaFreeHost(buffer_);
    }
    cudaEventDestroy(event_);
}

bool GuiStaging::stage(size_t count, const std::function<void(float* host_buffer)>& enqueue, cudaStream_t stream,
    std::vector<int> tag)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ != State::kIdle) {
            return false;
        }
        state_ = State::kFilling;
    }
    // Idle means no copy targets the buffer, so it's safe to replace it here.
    try {
        if (count > capacity_) {
            if (buffer_) {
                CUDA_CHECK(cudaFreeHost(buffer_));
                buffer_ = nullptr;
                capacity_ = 0;
            }
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&buffer_), count * sizeof(float)));
            capacity_ = count;
        }
        enqueue(buffer_);
        CUDA_CHECK(cudaEventRecord(event_, stream));
    } catch (...) {
        // Copies enqueued before the failure may still be landing in the buffer.
        cudaStreamSynchronize(stream);
        std::lock_guard<std::mutex> lock(mutex_);
        state_ = State::kIdle;
        throw;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    count_ = count;
    tag_ = std::move(tag);
    state_ = State::kInFlight;
    return true;
}

bool GuiStaging::take(std::vector<float>& batch, std::vector<int>* tag)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != State::kInFlight) {
        return false;
    }
    const cudaError_t status = cudaEventQuery(event_);
    if (status == cudaErrorNotReady) {
        return false;
    }
    if (status != cudaSuccess) {
        HSB_LOG_ERROR("GUI staging copy failed: {}", cudaGetErrorString(status));
        state_ = State::kIdle;
        return false;
    }
    batch.assign(buffer_, buffer_ + count_);
    if (tag) {
        *tag = tag_;
    }
    state_ = State::kIdle;
    return true;
}

} // namespace hololink
