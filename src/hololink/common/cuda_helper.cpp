/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cuda_helper.hpp"

#include <memory>

#include <nvrtc.h>

#include <hololink/core/logging_internal.hpp>

/**
 * NvRTC API error check helper
 */
#define NvRTCCheck(FUNC)                                                                                                    \
    {                                                                                                                       \
        const nvrtcResult result = FUNC;                                                                                    \
        if (result != NVRTC_SUCCESS) {                                                                                      \
            std::stringstream buf;                                                                                          \
            buf << "[" << __FILE__ << ":" << __LINE__ << "] NvRTC error " << result << ": " << nvrtcGetErrorString(result); \
            throw std::runtime_error(buf.str().c_str());                                                                    \
        }                                                                                                                   \
    }

namespace hololink::common {

CudaFunctionLauncher::CudaFunctionLauncher(const char* source,
    const std::vector<std::string>& functions, const std::vector<std::string>& options)
{
    nvrtcProgram prog;
    NvRTCCheck(nvrtcCreateProgram(&prog, // prog
        source, // buffer
        "", // name
        0, // numHeaders
        NULL, // headers
        NULL)); // includeNames
    std::vector<const char*> compile_options;
    for (auto&& option : options) {
        compile_options.push_back(option.c_str());
    }
    compile_options.push_back("--include-path=/usr/local/cuda/include");
    compile_options.push_back("--include-path=/usr/local/cuda/include/cccl");
    if (nvrtcCompileProgram(prog, compile_options.size(), compile_options.data()) != NVRTC_SUCCESS) {
        // Obtain compilation log from the program.
        size_t logSize;
        NvRTCCheck(nvrtcGetProgramLogSize(prog, &logSize));
        std::unique_ptr<char[]> log(new char[logSize]);
        NvRTCCheck(nvrtcGetProgramLog(prog, log.get()));
        std::stringstream buf;
        buf << "Failed to compile: " << log.get();
        throw std::runtime_error(buf.str().c_str());
    }
    // Obtain PTX from the program.
    size_t ptxSize;
    NvRTCCheck(nvrtcGetPTXSize(prog, &ptxSize));
    std::unique_ptr<char[]> ptx(new char[ptxSize]);
    NvRTCCheck(nvrtcGetPTX(prog, ptx.get()));
    // Destroy the program.
    NvRTCCheck(nvrtcDestroyProgram(&prog));

    // Load the generated PTX and get a handle to the kernels
    CudaCheck(cuModuleLoadDataEx(&module_, ptx.get(), 0, 0, 0));
    for (auto&& function : functions) {
        LaunchParams launch_params;
        CudaCheck(cuModuleGetFunction(&launch_params.function_, module_, function.c_str()));

        // calculate the optimal block size for max occupancy
        int min_grid_size = 0;
        int optimal_block_size = 0;
        CudaCheck(cuOccupancyMaxPotentialBlockSize(&min_grid_size, &optimal_block_size, launch_params.function_, nullptr, 0, 0));

        // get a 2D block size from the optimal block size
        launch_params.block_dim_.x = 1;
        launch_params.block_dim_.y = 1;
        launch_params.block_dim_.z = 1;
        while (static_cast<int>(launch_params.block_dim_.x * launch_params.block_dim_.y * 2) <= optimal_block_size) {
            if (launch_params.block_dim_.x > launch_params.block_dim_.y) {
                launch_params.block_dim_.y *= 2;
            } else {
                launch_params.block_dim_.x *= 2;
            }
        }

        functions_[function] = launch_params;
    }
}

CudaFunctionLauncher::~CudaFunctionLauncher()
{
    try {
        CudaCheck(cuModuleUnload(module_));
    } catch (const std::exception& e) {
        HSB_LOG_ERROR("CudaFunctionLauncher destructor failed with {}", e.what());
    }
}

void CudaFunctionLauncher::launch_internal(const std::string& name, const dim3& grid, const dim3* block, CUstream stream, void** args) const
{
    const LaunchParams& launch_params = functions_.at(name);
    dim3 cur_block;
    if (block) {
        cur_block = *block;
    } else {
        cur_block = launch_params.block_dim_;
    }

    // calculate the launch grid size
    dim3 launch_grid;
    launch_grid.x = (grid.x + (cur_block.x - 1)) / cur_block.x;
    launch_grid.y = (grid.y + (cur_block.y - 1)) / cur_block.y;
    launch_grid.z = (grid.z + (cur_block.z - 1)) / cur_block.z;
    CudaCheck(cuLaunchKernel(launch_params.function_,
        launch_grid.x,
        launch_grid.y,
        launch_grid.z,
        cur_block.x,
        cur_block.y,
        cur_block.z,
        0,
        stream,
        args,
        nullptr));
}

CudaContextScopedPush::CudaContextScopedPush(CUcontext cuda_context)
    : cuda_context_(cuda_context)
{
    // might be called from a different thread than the thread
    // which constructed the context, therefore call cuInit()
    CudaCheck(cuInit(0));
    CudaCheck(cuCtxPushCurrent(cuda_context_));
}

CudaContextScopedPush::~CudaContextScopedPush()
{
    try {
        CUcontext popped_context;
        CudaCheck(cuCtxPopCurrent(&popped_context));
        if (popped_context != cuda_context_) {
            HSB_LOG_ERROR("Cuda: Unexpected context popped");
        }
    } catch (const std::exception& e) {
        HSB_LOG_ERROR("ScopedPush destructor failed with {}", e.what());
    }
}

} // namespace hololink
