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

#ifndef SRC_HOLOLINK_CUDA_ERROR_HPP
#define SRC_HOLOLINK_CUDA_ERROR_HPP

#include <exception>

#include <cuda_runtime.h>
#include <cufft.h>
#include <nvrtc.h>

#include <hololink/core/logging_internal.hpp>

#define CUDA_CHECK(err)                                             \
    do {                                                            \
        auto e = err; /* err might be a function call */            \
        if (e != cudaSuccess) {                                     \
            HSB_LOG_ERROR("CUDA ERROR: {}", cudaGetErrorString(e)); \
            throw hololink::CudaError(e);                           \
        }                                                           \
    } while (0)

#define NVRTC_CHECK(err)                                              \
    do {                                                              \
        auto e = err; /* err might be a function call */              \
        if (e != NVRTC_SUCCESS) {                                     \
            HSB_LOG_ERROR("NVRTC ERROR: {}", nvrtcGetErrorString(e)); \
            throw hololink::NvrtcError(e);                            \
        }                                                             \
    } while (0)

#define CUFFT_CHECK(err)                                                        \
    do {                                                                        \
        auto e = err; /* err might be a function call */                        \
        if (e != CUFFT_SUCCESS) {                                               \
            HSB_LOG_ERROR("CUFFT ERROR: {}", hololink::cufftGetErrorString(e)); \
            throw hololink::CufftError(e);                                      \
        }                                                                       \
    } while (0)

namespace hololink {

inline const char* cufftGetErrorString(cufftResult result)
{
    switch (result) {
    case CUFFT_SUCCESS:
        return "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN:
        return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED:
        return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE:
        return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE:
        return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR:
        return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED:
        return "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED:
        return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE:
        return "CUFFT_INVALID_SIZE";
    case CUFFT_UNALIGNED_DATA:
        return "CUFFT_UNALIGNED_DATA";
#if CUFFT_VERSION < 12000
    case CUFFT_INCOMPLETE_PARAMETER_LIST:
        return "CUFFT_INCOMPLETE_PARAMETER_LIST";
#endif
    case CUFFT_INVALID_DEVICE:
        return "CUFFT_INVALID_DEVICE";
#if CUFFT_VERSION < 12000
    case CUFFT_PARSE_ERROR:
        return "CUFFT_PARSE_ERROR";
#endif
    case CUFFT_NO_WORKSPACE:
        return "CUFFT_NO_WORKSPACE";
    case CUFFT_NOT_IMPLEMENTED:
        return "CUFFT_NOT_IMPLEMENTED";
#if CUFFT_VERSION < 12000
    case CUFFT_LICENSE_ERROR:
        return "CUFFT_LICENSE_ERROR";
#endif
    case CUFFT_NOT_SUPPORTED:
        return "CUFFT_NOT_SUPPORTED";
    }
    return "Unknown cufftResult";
}

class CudaError : public std::exception {
public:
    CudaError(cudaError e)
        : e_(e)
    {
    }
    const char* what() const noexcept override
    {
        return cudaGetErrorString(e_);
    }

private:
    cudaError e_;
};

class NvrtcError : public std::exception {
public:
    NvrtcError(nvrtcResult e)
        : e_(e)
    {
    }
    const char* what() const noexcept override
    {
        return nvrtcGetErrorString(e_);
    }

private:
    nvrtcResult e_;
};

class CufftError : public std::exception {
public:
    CufftError(cufftResult e)
        : e_(e)
    {
    }
    const char* what() const noexcept override
    {
        return cufftGetErrorString(e_);
    }

private:
    cufftResult e_;
};

} // namespace hololink

#endif /* SRC_HOLOLINK_CUDA_ERROR_HPP */
