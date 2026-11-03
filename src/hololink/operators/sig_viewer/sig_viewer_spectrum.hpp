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

#ifndef SRC_HOLOLINK_OPERATORS_SIG_VIEWER_SIG_VIEWER_SPECTRUM_HPP
#define SRC_HOLOLINK_OPERATORS_SIG_VIEWER_SIG_VIEWER_SPECTRUM_HPP

#include <cuda_runtime_api.h>
#include <cufft.h>

namespace hololink::operators {

enum class SpectrumOutput {
    kPower = 0,
    kAmplitude = 1,
};

// The spectrum of the first count() samples of an interleaved IQ (complex) device buffer.
class SignalSpectrum {
public:
    explicit SignalSpectrum(int count);
    ~SignalSpectrum();
    SignalSpectrum(const SignalSpectrum&) = delete;
    SignalSpectrum& operator=(const SignalSpectrum&) = delete;

    int count() const { return count_; }

    // Enqueue the computation on stream.
    void calculate(const float* device_samples_iq, SpectrumOutput output, cudaStream_t stream);

    // Device buffer of count() floats, valid once calculate()'s work completes.
    const float* data() const { return magnitude_; }

private:
    int count_;
    cufftHandle plan_ {};
    cufftComplex* complex_ = nullptr;
    float* magnitude_ = nullptr;
};

} // namespace hololink::operators

#endif /* SRC_HOLOLINK_OPERATORS_SIG_VIEWER_SIG_VIEWER_SPECTRUM_HPP */
