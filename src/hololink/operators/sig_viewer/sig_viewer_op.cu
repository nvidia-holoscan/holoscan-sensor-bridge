/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include <stdexcept>

#include <cufft.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include <hololink/common/cuda_error.hpp>

#include "sig_viewer_op.hpp"

static const size_t thread_block_size = 1024;

// A class representing a cufftPlan1d
struct Plan {
    // size is the plan's size
    Plan(int size) :
        size_(size) {
        // 1D FFT, Complex-to-Complex
        CUFFT_CHECK(cufftPlan1d(&handle_, size_, CUFFT_C2C, 1));
    }
    ~Plan() try {
        CUFFT_CHECK(cufftDestroy(handle_));
    } catch(const std::exception&) {
        return;
    }
    Plan(Plan&) = delete;
    Plan& operator=(Plan&) = delete;

    void SetStream(cudaStream_t stream) {
        CUFFT_CHECK(cufftSetStream(handle_, stream));
    }

    int size_;
    cufftHandle handle_;
};

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

struct SignalViewerOp::SpectrumView::Spectrum::Impl {
    Impl(int64_t count) :
        count_(count),
        complex_(count_),
        plan_(count_),
        magnitude_(count_, 0)
    {
    }

    int64_t count_;
    thrust::device_vector<cufftComplex> complex_;
    Plan plan_;
    thrust::device_vector<float> magnitude_;

    void calculate(const float* device_samples_iq, OutputType output_type, cudaStream_t stream) {
        thrust::copy(
            reinterpret_cast<const cufftComplex*>(device_samples_iq),
            reinterpret_cast<const cufftComplex*>(device_samples_iq) + count_,
            complex_.begin());
    
        plan_.SetStream(stream);
    
        // Perform FFT (this will overwrite complex signal with frequency domain data)
        cufftExecC2C(plan_.handle_, complex_.data().get(), complex_.data().get(), CUFFT_FORWARD);

        switch (output_type) {
        case OutputType::Amplitude:
            cuda_compute_complex_amplitude_kernel<<<static_cast<size_t>(std::ceil(1.0 * count_ / thread_block_size)),
                            thread_block_size, 0, stream>>>(magnitude_.data().get(), complex_.data().get(), count_);
            break;
        case OutputType::Power:
            cuda_compute_complex_power_kernel<<<static_cast<size_t>(std::ceil(1.0 * count_ / thread_block_size)),
                            thread_block_size, 0, stream>>>(magnitude_.data().get(), complex_.data().get(), count_);
            break;
        }
    }

    const thrust::device_vector<float>& get_data() const {
        return magnitude_;
    }
};

SignalViewerOp::SpectrumView::Spectrum::Spectrum(int64_t size) :
    pimpl_(new Impl(size))
{
}

SignalViewerOp::SpectrumView::Spectrum::~Spectrum() = default;

int64_t SignalViewerOp::SpectrumView::Spectrum::get_count() const {
    return pimpl_->count_;
}

void SignalViewerOp::SpectrumView::Spectrum::calculate(const float* device_samples_iq, OutputType output_type, cudaStream_t stream) {
    pimpl_->calculate(device_samples_iq, output_type, stream);
}

const thrust::device_vector<float>& SignalViewerOp::SpectrumView::Spectrum::get_data() const {
    return pimpl_->get_data();
}

std::pair<float, float> SignalViewerOp::cuda_minmax(const float* data, size_t size, cudaStream_t stream) {
    thrust::device_ptr<const float> device_ptr = thrust::device_pointer_cast(data);
    auto result = thrust::minmax_element(thrust::cuda::par.on(stream), device_ptr, device_ptr + size);
    return std::pair<float, float>(*result.first, *result.second);
}
 

}  // namespace hololink::operators
