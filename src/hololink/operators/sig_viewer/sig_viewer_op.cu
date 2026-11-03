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

#include "sig_viewer_spectrum.hpp"

#include <hololink/common/cuda_error.hpp>

static const int thread_block_size = 1024;

__global__ void cuda_compute_complex_power_kernel(float* magnitude, const cufftComplex* complex, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= count)
        return;
    magnitude[idx] = complex[idx].x * complex[idx].x + complex[idx].y * complex[idx].y;
}

__global__ void cuda_compute_complex_amplitude_kernel(float* magnitude, const cufftComplex* complex, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size)
        return;
    magnitude[idx] = sqrtf(complex[idx].x * complex[idx].x + complex[idx].y * complex[idx].y);
}

namespace hololink::operators {

SignalSpectrum::SignalSpectrum(int count) :
    count_(count)
{
    // 1D FFT, Complex-to-Complex
    CUFFT_CHECK(cufftPlan1d(&plan_, count_, CUFFT_C2C, 1));
    try {
        CUDA_CHECK(cudaMalloc(&complex_, count_ * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMalloc(&magnitude_, count_ * sizeof(float)));
    } catch (...) {
        cudaFree(complex_);
        cufftDestroy(plan_);
        throw;
    }
}

SignalSpectrum::~SignalSpectrum() {
    cudaFree(magnitude_);
    cudaFree(complex_);
    cufftDestroy(plan_);
}

void SignalSpectrum::calculate(const float* device_samples_iq, SpectrumOutput output, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(complex_, device_samples_iq, count_ * sizeof(cufftComplex), cudaMemcpyDeviceToDevice,
        stream));
    CUFFT_CHECK(cufftSetStream(plan_, stream));
    // In place: the signal is overwritten with its frequency domain data.
    CUFFT_CHECK(cufftExecC2C(plan_, complex_, complex_, CUFFT_FORWARD));

    const int blocks = (count_ + thread_block_size - 1) / thread_block_size;
    switch (output) {
    case SpectrumOutput::kAmplitude:
        cuda_compute_complex_amplitude_kernel<<<blocks, thread_block_size, 0, stream>>>(magnitude_, complex_, count_);
        break;
    case SpectrumOutput::kPower:
        cuda_compute_complex_power_kernel<<<blocks, thread_block_size, 0, stream>>>(magnitude_, complex_, count_);
        break;
    }
}

}  // namespace hololink::operators
