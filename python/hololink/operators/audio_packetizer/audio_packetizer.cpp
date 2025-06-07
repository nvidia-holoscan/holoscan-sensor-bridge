/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// AudioPacketizerOp (Python bindings)
#include <hololink/operators/audio_packetizer/audio_packetizer.hpp>

#include <pybind11/pybind11.h>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace hololink::operators {

class PyAudioPacketizerOp : public AudioPacketizerOp {
public:
    /* Inherit the constructors */
    using AudioPacketizerOp::AudioPacketizerOp;

    // Define a constructor that fully initializes the object
    PyAudioPacketizerOp(holoscan::Fragment* fragment,
        const std::string& wav_file,
        uint32_t chunk_size,
        std::shared_ptr<holoscan::Allocator> pool,
        const std::string& name)
        : AudioPacketizerOp(holoscan::ArgList {
            holoscan::Arg { "wav_file", wav_file },
            holoscan::Arg { "chunk_size", chunk_size },
            holoscan::Arg { "pool", pool } })
    {
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<holoscan::OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

PYBIND11_MODULE(_audio_packetizer, m)
{
    m.doc() = "Audio Packetizer Operator for transmitting WAV file data"; // optional module docstring

    py::class_<AudioPacketizerOp, PyAudioPacketizerOp, holoscan::Operator,
        std::shared_ptr<AudioPacketizerOp>>(m, "AudioPacketizerOp")
        .def(py::init<holoscan::Fragment*,
                 const std::string&,
                 uint32_t,
                 std::shared_ptr<holoscan::Allocator>,
                 const std::string&>(),
            "fragment"_a,
            "wav_file"_a,
            "chunk_size"_a = 1024,
            "pool"_a,
            "name"_a = "audio_packetizer");
}

} // namespace hololink::operators
