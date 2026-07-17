/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef HOLOLINK_MODULE_OPERATORS_CUDA_FUNCTION_LAUNCHER_HPP
#define HOLOLINK_MODULE_OPERATORS_CUDA_FUNCTION_LAUNCHER_HPP

#include <string>
#include <unordered_map>
#include <vector>

#include <cuda.h>
#include <vector_types.h>

/* Module-owned CUDA kernel helpers (CudaFunctionLauncher /
 * CudaContextScopedPush) used by CsiToBayerOp, vendored here so the operators
 * tree is self-contained. The unique_ptr aliases and the
 * HOLOLINK_MODULE_CUDA_CHECK macro come from cuda_unique.hpp. */

namespace hololink::module::operators {

/**
 * Compiles the provided CUDA source with NVRTC and launches the kernel
 * functions it defines.
 */
class CudaFunctionLauncher {
public:
    /**
     * @param source pointer to source code
     * @param functions list of functions defined by the source code
     * @param options options to pass to the compiler
     */
    CudaFunctionLauncher(const char* source, const std::vector<std::string>& functions,
        const std::vector<std::string>& options = std::vector<std::string>());
    ~CudaFunctionLauncher();

    /**
     * Launch a kernel on a grid.
     */
    template <class... TYPES>
    void launch(const std::string& name, const dim3& grid, CUstream stream, TYPES... args) const
    {
        void* args_array[] = { reinterpret_cast<void*>(&args)... };
        launch_internal(name, grid, nullptr, stream, args_array);
    }

    /**
     * Launch a kernel with an explicit block size on a grid.
     */
    template <class... TYPES>
    void launch(const std::string& name, const dim3& grid, const dim3& block, CUstream stream, TYPES... args) const
    {
        void* args_array[] = { reinterpret_cast<void*>(&args)... };
        launch_internal(name, grid, &block, stream, args_array);
    }

private:
    void launch_internal(const std::string& name, const dim3& grid, const dim3* block,
        CUstream stream, void** args) const;

    CUmodule module_ = nullptr;
    struct LaunchParams {
        dim3 block_dim_;
        CUfunction function_;
    };
    std::unordered_map<std::string, LaunchParams> functions_;
};

/**
 * RAII type that pushes a CUDA context on construction and pops it on
 * destruction.
 */
class CudaContextScopedPush {
public:
    /**
     * @param cuda_context context to push
     */
    explicit CudaContextScopedPush(CUcontext cuda_context);
    CudaContextScopedPush() = delete;
    ~CudaContextScopedPush();

private:
    const CUcontext cuda_context_;
};

} // namespace hololink::module::operators

#endif // HOLOLINK_MODULE_OPERATORS_CUDA_FUNCTION_LAUNCHER_HPP
