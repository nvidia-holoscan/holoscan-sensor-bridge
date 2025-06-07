/**
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <hololink/core/arp_wrapper.hpp>
#include <hololink/core/deserializer.hpp>
#include <hololink/core/networking.hpp>
#include <hololink/core/serializer.hpp>
#include <hololink/core/tools.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace hololink::core {

// Provide access to the buffer passed in
class WrappedSerializer : public Serializer {
public:
    WrappedSerializer(uint8_t* buffer, unsigned size)
        : Serializer(buffer, size)
    {
    }

    uint8_t* buffer() { return buffer_; }
};

PYBIND11_MODULE(_hololink_core, m)
{
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    py::class_<ArpWrapper>(m, "ArpWrapper")
        .def(py::init<>())
        .def_static(
            "arp_set", &ArpWrapper::arp_set, "socket_fd"_a, "eth_device"_a, "ip"_a, "mac_id"_a);

    py::class_<Deserializer>(m, "Deserializer")
        .def(py::init([](py::buffer buffer) {
            py::buffer_info info = buffer.request();
            if (info.ndim != 1) {
                throw std::runtime_error("Only 1-dimensional buffers are acceptable.");
            }
            return Deserializer((uint8_t*)info.ptr, info.shape[0]);
        }),
            py::keep_alive<1, 2>())
        .def(py::init([](py::buffer buffer, unsigned length) {
            py::buffer_info info = buffer.request();
            if (info.ndim != 1) {
                throw std::runtime_error("Only 1-dimensional buffers are acceptable.");
            }
            if (length > info.shape[0]) {
                throw std::runtime_error("Excessive \"length\" value");
            }
            return Deserializer((uint8_t*)info.ptr, length);
        }),
            "buffer"_a, "length"_a = 0, py::keep_alive<1, 2>())
        .def("next_uint8",
            [](Deserializer& me) {
                uint8_t value = 0;
                bool r = me.next_uint8(value);
                if (!r) {
                    throw std::runtime_error("Buffer underflow");
                }
                return value;
            })
        .def("next_uint16_be",
            [](Deserializer& me) {
                uint16_t value = 0;
                bool r = me.next_uint16_be(value);
                if (!r) {
                    throw std::runtime_error("Buffer underflow");
                }
                return value;
            })
        .def("next_uint16_le",
            [](Deserializer& me) {
                uint16_t value = 0;
                bool r = me.next_uint16_le(value);
                if (!r) {
                    throw std::runtime_error("Buffer underflow");
                }
                return value;
            })
        .def("next_uint32_be",
            [](Deserializer& me) {
                uint32_t value = 0;
                bool r = me.next_uint32_be(value);
                if (!r) {
                    throw std::runtime_error("Buffer underflow");
                }
                return value;
            })
        .def("next_uint32_le",
            [](Deserializer& me) {
                uint32_t value = 0;
                bool r = me.next_uint32_le(value);
                if (!r) {
                    throw std::runtime_error("Buffer underflow");
                }
                return value;
            })
        .def("next_uint48_be",
            [](Deserializer& me) {
                uint64_t value = 0;
                bool r = me.next_uint48_be(value);
                if (!r) {
                    throw std::runtime_error("Buffer underflow");
                }
                return value;
            })
        .def("next_uint64_be",
            [](Deserializer& me) {
                uint64_t value = 0;
                bool r = me.next_uint64_be(value);
                if (!r) {
                    throw std::runtime_error("Buffer underflow");
                }
                return value;
            })
        .def("next_uint64_le",
            [](Deserializer& me) {
                uint64_t value = 0;
                bool r = me.next_uint64_le(value);
                if (!r) {
                    throw std::runtime_error("Buffer underflow");
                }
                return value;
            })
        // next_buffer returns a pointer to an internal buffer, Python should not take ownership but
        // couple the lifetime of the pointer to the Deserializer instance
        .def(
            "next_buffer",
            [](Deserializer& me, unsigned n) {
                const uint8_t* pointer = nullptr;
                bool r = me.pointer(pointer, n);
                if (!r) {
                    throw std::runtime_error("Buffer underflow");
                }
                return py::memoryview::from_memory(pointer, sizeof(uint8_t) * n);
            },
            "n"_a)
        .def("position", &Deserializer::position);

    py::class_<WrappedSerializer>(m, "Serializer")
        .def(py::init([](py::buffer buffer) {
            py::buffer_info info = buffer.request();
            if (info.ndim != 1) {
                throw std::runtime_error("Only 1-dimensional buffers are acceptable.");
            }
            return WrappedSerializer((uint8_t*)info.ptr, info.shape[0]);
        }),
            py::keep_alive<1, 2>())
        .def(py::init([](py::buffer buffer, unsigned length) {
            py::buffer_info info = buffer.request();
            if (info.ndim != 1) {
                throw std::runtime_error("Only 1-dimensional buffers are acceptable.");
            }
            if (length > info.shape[0]) {
                throw std::runtime_error("Excessive \"length\" value");
            }
            return WrappedSerializer((uint8_t*)info.ptr, length);
        }),
            "buffer"_a, "length"_a = 0, py::keep_alive<1, 2>())
        .def("append_uint32_be",
            [](WrappedSerializer& me, uint32_t value) {
                bool r = me.append_uint32_be(value);
                if (!r) {
                    throw std::runtime_error("Buffer overflow");
                }
                return r;
            })
        .def("append_uint32_le",
            [](WrappedSerializer& me, uint32_t value) {
                bool r = me.append_uint32_le(value);
                if (!r) {
                    throw std::runtime_error("Buffer overflow");
                }
                return r;
            })
        .def("append_uint16_be",
            [](WrappedSerializer& me, uint16_t value) {
                bool r = me.append_uint16_be(value);
                if (!r) {
                    throw std::runtime_error("Buffer overflow");
                }
                return r;
            })
        .def("append_uint16_le",
            [](WrappedSerializer& me, uint16_t value) {
                bool r = me.append_uint16_le(value);
                if (!r) {
                    throw std::runtime_error("Buffer overflow");
                }
                return r;
            })
        .def("append_uint8",
            [](WrappedSerializer& me, uint8_t value) {
                bool r = me.append_uint8(value);
                if (!r) {
                    throw std::runtime_error("Buffer overflow");
                }
                return r;
            })
        .def("append_buffer",
            [](WrappedSerializer& me, py::buffer buffer) {
                py::buffer_info info = buffer.request();
                if (info.ndim != 1) {
                    throw std::runtime_error("Only 1-dimensional buffers are acceptable.");
                }
                unsigned length = info.shape[0];
                bool r = me.append_buffer((uint8_t*)info.ptr, length);
                if (!r) {
                    throw std::runtime_error("Buffer overflow");
                }
                return r;
            })
        .def("append_buffer",
            [](WrappedSerializer& me, py::buffer buffer, unsigned length = 0) {
                py::buffer_info info = buffer.request();
                if (info.ndim != 1) {
                    throw std::runtime_error("Only 1-dimensional buffers are acceptable.");
                }
                if (length > info.shape[0]) {
                    throw std::runtime_error("Excessive \"length\" value");
                }
                bool r = me.append_buffer((uint8_t*)info.ptr, length);
                if (!r) {
                    throw std::runtime_error("Buffer overflow");
                }
                return r;
            })
        .def("append_buffer",
            [](WrappedSerializer& me, const std::string& buffer) {
                bool r = me.append_buffer((uint8_t*)buffer.data(), buffer.size());
                if (!r) {
                    throw std::runtime_error("Buffer overflow");
                }
                return r;
            })
        .def("data",
            [](WrappedSerializer& me) {
                std::string s((const char*)me.buffer(), me.length());
                return py::bytes(s);
            })
        .def("length", &WrappedSerializer::length);

    m.def("local_mac", &local_mac);

    m.def("local_ip_and_mac", &local_ip_and_mac, "destination_ip"_a, "port"_a);

    m.def("local_ip_and_mac_from_socket", &local_ip_and_mac_from_socket, "socket_fd"_a);

    m.attr("UDP_PACKET_SIZE") = UDP_PACKET_SIZE;
    m.attr("PAGE_SIZE") = PAGE_SIZE;

    m.def("round_up", &round_up, "value"_a, "alignment"_a);

    m.def("infiniband_devices", &infiniband_devices, "Return a sorted list of Infiniband devices.");
} // PYBIND11_MODULE

} // namespace hololink::core
