/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef SRC_HOLOLINK_COMMON_CUDA_HELPER
#define SRC_HOLOLINK_COMMON_CUDA_HELPER

#include <cuda.h>
#include <vector_types.h>

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

#include "hololink/core/nullable_pointer.hpp"

template <>
struct fmt::formatter<CUresult> : fmt::formatter<int> {
    auto format(CUresult cu_result, format_context& ctx) const
    {
        return fmt::formatter<int>::format(static_cast<int>(cu_result), ctx);
    }
};

namespace hololink::common {

/**
 * Returns the compute-capability arch string for the device attached to the
 * current CUDA context, e.g. "sm_90", "sm_100", "sm_110". Used to target
 * NVRTC CUBIN emission at a specific SM so we bypass the driver's PTX JIT
 * (required on platforms like R580.00 where PTX from newer NVRTC toolchains
 * is rejected with CUDA_ERROR_UNSUPPORTED_PTX_VERSION).
 *
 * \warning The SM string reflects whichever device is bound to the active
 * context at the call site. Does not currently support multi-GPU scenarios.
 */
std::string current_device_sm_arch();

/**
 * CUDA driver API error check helper
 */
#define CudaCheck(FUNC)                                                         \
    {                                                                           \
        const CUresult result = FUNC;                                           \
        if (result != CUDA_SUCCESS) {                                           \
            const char* error_name = "";                                        \
            cuGetErrorName(result, &error_name);                                \
            const char* error_string = "";                                      \
            cuGetErrorString(result, &error_string);                            \
            std::stringstream buf;                                              \
            buf << "[" << __FILE__ << ":" << __LINE__ << "] CUDA driver error " \
                << result << " (" << error_name << "): " << error_string;       \
            throw std::runtime_error(buf.str().c_str());                        \
        }                                                                       \
    }

/**
 * This class is used to compile the provided CUDA source code and call kernel functions
 * defined by that code.
 */
class CudaFunctionLauncher {
public:
    /**
     * Construct a new CudaFunctionLauncher object
     *
     * @param source pointer to source code
     * @param functions list of functions defined by the source code.
     * @param options options to pass to the compiler
     */
    CudaFunctionLauncher(const char* source, const std::vector<std::string>& functions, const std::vector<std::string>& options = std::vector<std::string>());
    ~CudaFunctionLauncher();

    /**
     * Launch a kernel on a grid.
     *
     * @param grid [in] grid size
     * @param stream [in] stream
     * @param args [in] kernel arguments (optional)
     */
    template <class... TYPES>
    void launch(const std::string& name, const dim3& grid, CUstream stream, TYPES... args) const
    {
        void* args_array[] = { reinterpret_cast<void*>(&args)... };
        launch_internal(name, grid, nullptr, stream, args_array);
    }

    /**
     * Launch a kernel with block size on a grid.
     *
     * @param grid [in] grid size
     * @param block [in] block size
     * @param stream [in] stream
     * @param args [in] kernel arguments (optional)
     */
    template <class... TYPES>
    void launch(const std::string& name, const dim3& grid, const dim3& block, CUstream stream, TYPES... args) const
    {
        void* args_array[] = { reinterpret_cast<void*>(&args)... };
        launch_internal(name, grid, &block, stream, args_array);
    }

private:
    /**
     * Launch a kernel with block size on a grid.
     *
     * @param grid [in] grid size
     * @param block [in] block size (optional)
     * @param stream [in] stream (optional)
     * @param args [in] kernel arguments (optional)
     */
    void launch_internal(const std::string& name, const dim3& grid, const dim3* block, CUstream stream, void** args) const;

    CUmodule module_ = nullptr;
    struct LaunchParams {
        dim3 block_dim_;
        CUfunction function_;
    };
    std::unordered_map<std::string, LaunchParams> functions_;
};

/**
 * RAII type classes for CUDA allocations and objects, use like std::unique_ptr.
 */
using UniqueCUdeviceptr = std::unique_ptr<Nullable<CUdeviceptr>, Nullable<CUdeviceptr>::Deleter<CUresult, &cuMemFree>>;

/**
 * @brief Custom deleter for CUDA objects managed by unique_ptr
 *
 * This template provides a custom deleter for CUDA objects that need
 * to be properly cleaned up when managed by std::unique_ptr. It calls
 * the specified cleanup function when the unique_ptr goes out of scope.
 *
 * @tparam T The CUDA object type
 * @tparam func The cleanup function to call (e.g., cuMemFreeHost)
 */
template <typename T, CUresult func(T)>
struct Deleter {
    typedef T pointer;
    /**
     * @brief Operator to call the cleanup function
     *
     * @param value The CUDA object to clean up
     */
    void operator()(T value) const { func(value); }
};

using UniqueCUhostptr = std::unique_ptr<void, Deleter<void*, &cuMemFreeHost>>;
using UniqueCUevent = std::unique_ptr<CUevent, Deleter<CUevent, &cuEventDestroy>>;
using UniqueCUstream = std::unique_ptr<CUstream, Deleter<CUstream, &cuStreamDestroy>>;

/**
 * RAII type class to push a CUDA context.
 */
class CudaContextScopedPush {
public:
    /**
     * @brief Construct a new scoped cuda context object
     *
     * @param cuda_context context to push
     */
    explicit CudaContextScopedPush(CUcontext cuda_context);
    CudaContextScopedPush() = delete;

    /**
     * Destructor
     */
    ~CudaContextScopedPush();

private:
    const CUcontext cuda_context_;
};

} // namespace hololink::common

#endif /* SRC_HOLOLINK_COMMON_CUDA_HELPER */
