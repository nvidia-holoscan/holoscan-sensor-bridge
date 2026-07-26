/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "hololink/module/adapter.hpp"
#include "hololink/module/csi_converter.hpp"
#include "hololink/module/data_channel.hpp"
#include "hololink/module/enumeration.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/frame_metadata.hpp"
#include "hololink/module/hololink.hpp"
#include "hololink/module/i2c.hpp"
#include "hololink/module/i2c_lock.hpp"
#include "hololink/module/ibv_device.hpp"
#include "hololink/module/logging.hpp"
#include "hololink/module/module.hpp"
#include "hololink/module/networking.hpp"
#include "hololink/module/oscillator.hpp"
#include "hololink/module/ptp_pps_output.hpp"
#include "hololink/module/reactor.hpp"
#include "hololink/module/roce_data_channel.hpp"
#include "hololink/module/roce_receiver.hpp"
#include "hololink/module/sequencer.hpp"
#include "hololink/module/status.h"
#include "hololink/module/vsync.hpp"

#ifdef HOLOLINK_BUILD_ROCE
#include <infiniband/verbs.h> // ibv_get_device_list for infiniband_devices()
#endif

namespace py = pybind11;
namespace ad = hololink::module;

// Validates that `obj` exposes the buffer protocol and the returned
// buffer is a 1-D, itemsize-1, C-contiguous byte sequence — i.e. a
// `bytes` / `bytearray` / contiguous `memoryview` of bytes. Returns the
// populated buffer_info; the caller copies info.size bytes from
// info.ptr under the GIL (the caller holds it; buffer_info keeps the
// underlying Py_buffer alive until its destructor runs). Throws
// std::runtime_error with `context` woven into the message on any
// violation — context should read as a verb phrase, e.g.
// "invoking FrameMetadataInterfaceV1.decode host_memory".
static py::buffer_info require_1d_contiguous_byte_buffer(
    py::handle obj, const char* context)
{
    if (!PyObject_CheckBuffer(obj.ptr())) {
        throw std::runtime_error(
            std::string("While ") + context
            + ": value must support the buffer protocol "
              "(bytes, bytearray, memoryview, ...)");
    }
    py::buffer_info info = py::reinterpret_borrow<py::buffer>(obj).request();
    if (info.ndim != 1) {
        throw std::runtime_error(
            std::string("While ") + context
            + ": buffer must be 1-dimensional");
    }
    if (info.itemsize != 1) {
        throw std::runtime_error(
            std::string("While ") + context
            + ": buffer itemsize must be 1 byte");
    }
    // For a 1-D itemsize-1 buffer, C-contiguous means strides[0] == 1.
    // request() populates strides for ndim>=1.
    if (!info.strides.empty() && info.strides[0] != 1) {
        throw std::runtime_error(
            std::string("While ") + context
            + ": buffer must be C-contiguous");
    }
    return info;
}

// -------------------------------------------------------------------
// Trampoline classes — let Python subclass each abstract V1 interface.
// Each trampoline forwards every pure-virtual to a Python override via
// PYBIND11_OVERRIDE_PURE_NAME, exposing the C++ method under the same
// Python name. The .so is built with hidden visibility, so the
// trampolines stay internal to this translation unit despite living
// at namespace scope (no anonymous namespace).
// -------------------------------------------------------------------

class PyI2cLock : public ad::I2cLockV1 {
public:
    using ad::I2cLockV1::I2cLockV1;

    void lock() override
    {
        PYBIND11_OVERRIDE_PURE(void, ad::I2cLockV1, lock);
    }
    void unlock() override
    {
        PYBIND11_OVERRIDE_PURE(void, ad::I2cLockV1, unlock);
    }
    bool try_lock() override
    {
        PYBIND11_OVERRIDE_PURE(bool, ad::I2cLockV1, try_lock);
    }
};

// I2cInterfaceV1 trampoline. The Python signature for i2c_transaction
// is (peripheral_address, write_bytes, read_byte_count) -> bytes;
// the trampoline translates the C++ in/out vector into that shape.
class PyI2cInterface : public ad::I2cInterfaceV1 {
public:
    using ad::I2cInterfaceV1::I2cInterfaceV1;

    hololink_module_status_t i2c_transaction(
        uint32_t peripheral_address,
        const std::vector<uint8_t>& write_bytes,
        std::vector<uint8_t>& read_bytes) override
    {
        py::gil_scoped_acquire gil;
        py::function override_fn = py::get_override(
            static_cast<const ad::I2cInterfaceV1*>(this), "i2c_transaction");
        if (!override_fn) {
            throw std::runtime_error(
                "While invoking I2cInterfaceV1.i2c_transaction: subclass must "
                "override this method");
        }
        const size_t want = read_bytes.size();
        py::object result = override_fn(
            peripheral_address, py::bytes(reinterpret_cast<const char*>(write_bytes.data()), write_bytes.size()),
            want);
        // Accept bytes / bytearray / any buffer-protocol object.
        // py::cast<std::string> would also accept str and silently
        // UTF-8-encode it, which corrupts arbitrary binary payloads.
        py::buffer_info info = require_1d_contiguous_byte_buffer(
            result,
            "invoking I2cInterfaceV1.i2c_transaction override return value");
        if (static_cast<size_t>(info.size) != want) {
            throw std::runtime_error(
                "While invoking I2cInterfaceV1.i2c_transaction override: "
                "returned byte count does not match read_byte_count");
        }
        // GIL is held (acquired above), so the copy is safe; buffer_info
        // also keeps the Py_buffer alive for the lifetime of `info`.
        std::memcpy(read_bytes.data(), info.ptr, want);
        return HOLOLINK_MODULE_OK;
    }

    hololink_module_status_t encode_i2c_request(
        ad::SequencerInterfaceV1& sequencer,
        uint32_t peripheral_i2c_address,
        const std::vector<uint8_t>& write_bytes,
        uint32_t read_byte_count,
        std::vector<unsigned>& out_write_indexes,
        std::vector<unsigned>& out_read_indexes,
        unsigned& out_status_index) override
    {
        py::gil_scoped_acquire gil;
        py::function override_fn = py::get_override(
            static_cast<const ad::I2cInterfaceV1*>(this), "encode_i2c_request");
        if (!override_fn) {
            throw std::runtime_error(
                "While invoking I2cInterfaceV1.encode_i2c_request: subclass "
                "must override this method");
        }
        py::object result = override_fn(
            py::cast(&sequencer, py::return_value_policy::reference),
            peripheral_i2c_address,
            py::bytes(reinterpret_cast<const char*>(write_bytes.data()),
                write_bytes.size()),
            read_byte_count);
        py::tuple t = py::cast<py::tuple>(result);
        out_write_indexes = py::cast<std::vector<unsigned>>(t[0]);
        out_read_indexes = py::cast<std::vector<unsigned>>(t[1]);
        out_status_index = py::cast<unsigned>(t[2]);
        return HOLOLINK_MODULE_OK;
    }
};

class PySequencer : public ad::SequencerInterfaceV1 {
public:
    using ad::SequencerInterfaceV1::SequencerInterfaceV1;

    unsigned write_uint32(uint32_t address, uint32_t data) override
    {
        PYBIND11_OVERRIDE_PURE(unsigned, ad::SequencerInterfaceV1, write_uint32, address, data);
    }
    unsigned read_uint32(uint32_t address, uint32_t initial_value) override
    {
        PYBIND11_OVERRIDE_PURE(unsigned, ad::SequencerInterfaceV1, read_uint32, address, initial_value);
    }
    unsigned poll(uint32_t address, uint32_t mask, uint32_t match) override
    {
        PYBIND11_OVERRIDE_PURE(unsigned, ad::SequencerInterfaceV1, poll, address, mask, match);
    }
    hololink_module_status_t enable() override
    {
        PYBIND11_OVERRIDE_PURE(hololink_module_status_t, ad::SequencerInterfaceV1, enable);
    }
    uint32_t location() override
    {
        PYBIND11_OVERRIDE_PURE(uint32_t, ad::SequencerInterfaceV1, location);
    }
};

// DataChannel trampoline — per-channel anchor. Pure virtuals are the
// transport-agnostic enumeration_metadata / hololink accessors.
class PyDataChannel : public ad::DataChannelInterfaceV1 {
public:
    using ad::DataChannelInterfaceV1::DataChannelInterfaceV1;

    const ad::EnumerationMetadata& enumeration_metadata() const override
    {
        PYBIND11_OVERRIDE_PURE(
            const ad::EnumerationMetadata&, ad::DataChannelInterfaceV1,
            enumeration_metadata);
    }
    std::shared_ptr<ad::HololinkInterfaceV1> hololink() const override
    {
        PYBIND11_OVERRIDE_PURE(
            std::shared_ptr<ad::HololinkInterfaceV1>,
            ad::DataChannelInterfaceV1, hololink);
    }
    hololink_module_status_t device_lost() override
    {
        PYBIND11_OVERRIDE_PURE(
            hololink_module_status_t, ad::DataChannelInterfaceV1, device_lost);
    }
};

// RoceDataChannel trampoline. Standalone — does NOT inherit from
// DataChannelInterfaceV1. RoCE-typed callers reach the channel's
// metadata + hololink by holding a separate shared_ptr to the
// DataChannelInterfaceV1 anchor (the same instance the impl composes
// with internally).
class PyRoceDataChannel : public ad::RoceDataChannelInterfaceV1 {
public:
    using ad::RoceDataChannelInterfaceV1::RoceDataChannelInterfaceV1;

    hololink_module_status_t attach_receiver(
        std::shared_ptr<ad::RoceReceiverInterfaceV1> receiver) override
    {
        PYBIND11_OVERRIDE_PURE(
            hololink_module_status_t, ad::RoceDataChannelInterfaceV1, attach_receiver,
            receiver);
    }
    hololink_module_status_t detach_receiver() override
    {
        PYBIND11_OVERRIDE_PURE(
            hololink_module_status_t, ad::RoceDataChannelInterfaceV1, detach_receiver);
    }

    std::string frame_end_sequencer_instance_id() override
    {
        PYBIND11_OVERRIDE_PURE(
            std::string, ad::RoceDataChannelInterfaceV1, frame_end_sequencer_instance_id);
    }
    std::string parent_hololink_instance_id() override
    {
        PYBIND11_OVERRIDE_PURE(
            std::string, ad::RoceDataChannelInterfaceV1, parent_hololink_instance_id);
    }
};

class PyHololinkInterface : public ad::HololinkInterfaceV1 {
public:
    using ad::HololinkInterfaceV1::HololinkInterfaceV1;

    const ad::EnumerationMetadata& enumeration_metadata() const override
    {
        PYBIND11_OVERRIDE_PURE(
            const ad::EnumerationMetadata&, ad::HololinkInterfaceV1,
            enumeration_metadata);
    }

    std::shared_ptr<ad::DataChannelInterfaceV1> default_data_channel() const override
    {
        PYBIND11_OVERRIDE_PURE(
            std::shared_ptr<ad::DataChannelInterfaceV1>, ad::HololinkInterfaceV1,
            default_data_channel);
    }

    hololink_module_status_t start() override
    {
        PYBIND11_OVERRIDE_PURE(hololink_module_status_t, ad::HololinkInterfaceV1, start);
    }
    hololink_module_status_t stop() override
    {
        PYBIND11_OVERRIDE_PURE(hololink_module_status_t, ad::HololinkInterfaceV1, stop);
    }
    hololink_module_status_t reset() override
    {
        PYBIND11_OVERRIDE_PURE(hololink_module_status_t, ad::HololinkInterfaceV1, reset);
    }
    hololink_module_status_t configure_hsb() override
    {
        PYBIND11_OVERRIDE_PURE(hololink_module_status_t, ad::HololinkInterfaceV1, configure_hsb);
    }
    hololink_module_status_t device_lost() override
    {
        PYBIND11_OVERRIDE_PURE(hololink_module_status_t, ad::HololinkInterfaceV1, device_lost);
    }
    bool ptp_synchronize() override
    {
        PYBIND11_OVERRIDE_PURE(bool, ad::HololinkInterfaceV1, ptp_synchronize);
    }
    std::shared_ptr<ad::HololinkInterfaceV1::ResetRegistration> on_reset(
        std::function<void()> callback) override
    {
        PYBIND11_OVERRIDE_PURE(
            std::shared_ptr<ad::HololinkInterfaceV1::ResetRegistration>,
            ad::HololinkInterfaceV1, on_reset, callback);
    }
    hololink_module_status_t write_uint32(
        const std::vector<uint32_t>& addresses,
        const std::vector<uint32_t>& values) override
    {
        PYBIND11_OVERRIDE_PURE(
            hololink_module_status_t, ad::HololinkInterfaceV1, write_uint32,
            addresses, values);
    }
    hololink_module_status_t read_uint32(
        const std::vector<uint32_t>& addresses,
        std::vector<uint32_t>& out_values) override
    {
        py::gil_scoped_acquire gil;
        py::function override_fn = py::get_override(
            static_cast<const ad::HololinkInterfaceV1*>(this), "read_uint32");
        if (!override_fn) {
            throw std::runtime_error(
                "While invoking HololinkInterfaceV1.read_uint32: subclass must "
                "override this method");
        }
        py::object result = override_fn(addresses);
        out_values = py::cast<std::vector<uint32_t>>(result);
        return HOLOLINK_MODULE_OK;
    }
    hololink_module_status_t and_uint32(uint32_t address, uint32_t mask) override
    {
        PYBIND11_OVERRIDE_PURE(
            hololink_module_status_t, ad::HololinkInterfaceV1, and_uint32, address, mask);
    }
    hololink_module_status_t or_uint32(uint32_t address, uint32_t mask) override
    {
        PYBIND11_OVERRIDE_PURE(
            hololink_module_status_t, ad::HololinkInterfaceV1, or_uint32, address, mask);
    }
    hololink_module_status_t i2c_lock(std::unique_ptr<ad::I2cLockV1>&) override
    {
        // The V1 i2c_lock contract returns a unique_ptr<I2cLock> with the
        // default deleter; that ownership transfer cannot be bridged
        // cleanly from a Python-owned override return. Real Hololink
        // wrappers implement i2c_lock in C++.
        throw std::runtime_error(
            "While invoking HololinkInterfaceV1.i2c_lock: Python subclasses "
            "of HololinkInterfaceV1 cannot override i2c_lock; implement the "
            "wrapper in C++ instead");
    }

    std::string roce_data_channel_instance_id(
        const ad::EnumerationMetadata& md) override
    {
        PYBIND11_OVERRIDE_PURE(
            std::string, ad::HololinkInterfaceV1, roce_data_channel_instance_id, md);
    }
    std::string i2c_instance_id(uint32_t bus, uint32_t address) override
    {
        PYBIND11_OVERRIDE_PURE(
            std::string, ad::HololinkInterfaceV1, i2c_instance_id, bus, address);
    }
    std::string ptp_pps_output_instance_id() override
    {
        PYBIND11_OVERRIDE_PURE(
            std::string, ad::HololinkInterfaceV1, ptp_pps_output_instance_id);
    }
    std::string null_vsync_instance_id() override
    {
        PYBIND11_OVERRIDE_PURE(
            std::string, ad::HololinkInterfaceV1, null_vsync_instance_id);
    }
};

class PyEnumerationInterface : public ad::EnumerationInterfaceV1 {
public:
    using ad::EnumerationInterfaceV1::EnumerationInterfaceV1;

    hololink_module_status_t update_metadata(
        ad::EnumerationMetadata& metadata,
        const uint8_t* raw_packet, size_t raw_packet_len) override
    {
        py::gil_scoped_acquire gil;
        py::function override_fn = py::get_override(
            static_cast<const ad::EnumerationInterfaceV1*>(this), "update_metadata");
        if (!override_fn) {
            throw std::runtime_error(
                "While invoking EnumerationInterfaceV1.update_metadata: subclass "
                "must override this method");
        }
        py::object packet_arg;
        if (raw_packet == nullptr) {
            packet_arg = py::none();
        } else {
            packet_arg = py::bytes(reinterpret_cast<const char*>(raw_packet),
                raw_packet_len);
        }
        override_fn(py::cast(&metadata, py::return_value_policy::reference),
            packet_arg);
        return HOLOLINK_MODULE_OK;
    }
};

class PyLoggingInterface : public ad::LoggingInterfaceV1 {
public:
    using ad::LoggingInterfaceV1::LoggingInterfaceV1;

    ad::LogLevel level() const override
    {
        PYBIND11_OVERRIDE_PURE(ad::LogLevel, ad::LoggingInterfaceV1, level);
    }
    void log(ad::LogLevel level_,
        const char* file, unsigned line, const char* function,
        const char* message) override
    {
        PYBIND11_OVERRIDE_PURE_NAME(
            void, ad::LoggingInterfaceV1, "log", log,
            level_, std::string(file), line,
            std::string(function), std::string(message));
    }
};

// -------------------------------------------------------------------
// Per-interface bind helpers. Each function adds a single py::class_
// to the module; the PYBIND11_MODULE entry-point calls them in order.
// -------------------------------------------------------------------

// Convert an EnumerationMetadata variant value to a Python object.
// std::vector<uint8_t> must round-trip as bytes — pybind11's STL caster
// would otherwise turn it into list[int], breaking symmetry with the
// __setitem__ bytes path below.
static py::object enumeration_value_to_py(
    const ad::EnumerationMetadata::Value& v)
{
    return std::visit(
        [](auto&& x) -> py::object {
            using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<T, std::vector<uint8_t>>) {
                return py::bytes(
                    reinterpret_cast<const char*>(x.data()), x.size());
            } else {
                return py::cast(x);
            }
        },
        v);
}

static void bind_enumeration_metadata(py::module_& m)
{
    py::class_<ad::EnumerationMetadata>(m, "EnumerationMetadata")
        .def(py::init<>())
        // Copy ctor: needed by stereo flows that fork one base
        // metadata into per-sensor clones before calling
        // Adapter.use_sensor(clone, n). Mirrors how the C++ stereo
        // example does `left.metadata = base_metadata;`.
        .def(py::init<const ad::EnumerationMetadata&>(), py::arg("other"))
        .def("__setitem__",
            [](ad::EnumerationMetadata& self,
                const std::string& key, py::object value) {
                if (py::isinstance<py::int_>(value)) {
                    self[key] = static_cast<int64_t>(value.cast<py::int_>());
                } else if (py::isinstance<py::str>(value)) {
                    self[key] = value.cast<std::string>();
                } else if (py::isinstance<py::bytes>(value)) {
                    const std::string s = value.cast<std::string>();
                    self[key] = std::vector<uint8_t>(s.begin(), s.end());
                } else {
                    throw py::type_error(
                        "EnumerationMetadata values must be int, str, or bytes");
                }
            })
        .def("__getitem__",
            [](const ad::EnumerationMetadata& self,
                const std::string& key) -> py::object {
                const auto it = self.find(key);
                if (it == self.cend()) {
                    throw py::key_error(key);
                }
                return enumeration_value_to_py(it->second);
            })
        .def("__contains__",
            [](const ad::EnumerationMetadata& self, const std::string& key) {
                return self.find(key) != self.cend();
            })
        .def("__len__",
            [](const ad::EnumerationMetadata& self) { return self.size(); })
        .def(
            "get",
            [](const ad::EnumerationMetadata& self,
                const std::string& key, py::object default_value) -> py::object {
                const auto it = self.find(key);
                if (it == self.cend()) {
                    return default_value;
                }
                return enumeration_value_to_py(it->second);
            },
            py::arg("key"), py::arg("default") = py::none());
}

static void bind_module(py::module_& m)
{
    py::class_<ad::Module, std::shared_ptr<ad::Module>>(m, "Module");
}

// CsiConverterV1 trampoline so Python can implement the converter
// contract. The native in-tree implementation is the module
// CsiToBayerOp (hololink_module.operators.CsiToBayerOp), which
// subclasses this; the core publishes the abstract type so Python code
// can subclass it and the C++ sensor drivers (Vb1940Cam) accept it.
class PyCsiConverterV1 : public ad::csi::CsiConverterV1 {
public:
    using ad::csi::CsiConverterV1::CsiConverterV1;

    void configure(uint32_t start_byte, uint32_t received_bytes_per_line,
        uint32_t pixel_width, uint32_t pixel_height,
        ad::csi::PixelFormat pixel_format, uint32_t trailing_bytes) override
    {
        PYBIND11_OVERRIDE_PURE(void, ad::csi::CsiConverterV1, configure,
            start_byte, received_bytes_per_line, pixel_width, pixel_height,
            pixel_format, trailing_bytes);
    }
    uint32_t receiver_start_byte() override
    {
        PYBIND11_OVERRIDE_PURE(uint32_t, ad::csi::CsiConverterV1, receiver_start_byte);
    }
    uint32_t received_line_bytes(uint32_t line_bytes) override
    {
        PYBIND11_OVERRIDE_PURE(
            uint32_t, ad::csi::CsiConverterV1, received_line_bytes, line_bytes);
    }
    uint32_t transmitted_line_bytes(
        ad::csi::PixelFormat pixel_format, uint32_t pixel_width) override
    {
        PYBIND11_OVERRIDE_PURE(uint32_t, ad::csi::CsiConverterV1,
            transmitted_line_bytes, pixel_format, pixel_width);
    }
};

static void bind_csi_converter(py::module_& m)
{
    // Note: the ad::csi::PixelFormat enum is registered by the per-sensor
    // pybind modules (e.g. sensors/vb1940) — pybind keys types by C++
    // type, so registering it here too would fail with "type already
    // registered". The methods below reference it only in their
    // signatures; pybind resolves the type caster at call time, by which
    // point the sensor module that registered it is loaded.
    py::class_<ad::csi::CsiConverterV1, PyCsiConverterV1,
        std::shared_ptr<ad::csi::CsiConverterV1>>(m, "CsiConverterV1")
        .def(py::init<>())
        .def("configure", &ad::csi::CsiConverterV1::configure,
            py::arg("start_byte"), py::arg("received_bytes_per_line"),
            py::arg("pixel_width"), py::arg("pixel_height"),
            py::arg("pixel_format"), py::arg("trailing_bytes") = 0)
        .def("receiver_start_byte", &ad::csi::CsiConverterV1::receiver_start_byte)
        .def("received_line_bytes", &ad::csi::CsiConverterV1::received_line_bytes,
            py::arg("line_bytes"))
        .def("transmitted_line_bytes", &ad::csi::CsiConverterV1::transmitted_line_bytes,
            py::arg("pixel_format"), py::arg("pixel_width"));
}

static void bind_frame_metadata(py::module_& m)
{
    py::class_<ad::FrameMetadataInterfaceV1::FrameMetadata>(m, "FrameMetadata")
        .def_readonly("flags", &ad::FrameMetadataInterfaceV1::FrameMetadata::flags)
        .def_readonly("psn", &ad::FrameMetadataInterfaceV1::FrameMetadata::psn)
        .def_readonly("crc", &ad::FrameMetadataInterfaceV1::FrameMetadata::crc)
        .def_readonly("frame_number",
            &ad::FrameMetadataInterfaceV1::FrameMetadata::frame_number)
        .def_readonly("timestamp_s",
            &ad::FrameMetadataInterfaceV1::FrameMetadata::timestamp_s)
        .def_readonly("timestamp_ns",
            &ad::FrameMetadataInterfaceV1::FrameMetadata::timestamp_ns)
        .def_readonly("bytes_written",
            &ad::FrameMetadataInterfaceV1::FrameMetadata::bytes_written)
        .def_readonly("metadata_s",
            &ad::FrameMetadataInterfaceV1::FrameMetadata::metadata_s)
        .def_readonly("metadata_ns",
            &ad::FrameMetadataInterfaceV1::FrameMetadata::metadata_ns);

    py::class_<ad::FrameMetadataInterfaceV1,
        std::shared_ptr<ad::FrameMetadataInterfaceV1>>(m, "FrameMetadataInterfaceV1")
        .def_static(
            "get_service",
            [](std::shared_ptr<ad::Module> module, bool allow_null) {
                return ad::FrameMetadataInterfaceV1::get_service(
                    std::move(module), allow_null);
            },
            py::arg("module"), py::arg("allow_null") = false)
        .def("block_size", &ad::FrameMetadataInterfaceV1::block_size)
        .def(
            "decode",
            [](const ad::FrameMetadataInterfaceV1& self, py::buffer host_memory) {
                py::buffer_info info = require_1d_contiguous_byte_buffer(
                    host_memory,
                    "invoking FrameMetadataInterfaceV1.decode host_memory");
                ad::FrameMetadataInterfaceV1::FrameMetadata out {};
                const hololink_module_status_t s = self.decode(
                    info.ptr,
                    static_cast<size_t>(info.size),
                    out);
                if (s != HOLOLINK_MODULE_OK) {
                    throw std::runtime_error(
                        "While decoding frame metadata: status "
                        + std::to_string(s));
                }
                return out;
            },
            py::arg("host_memory"));
}

static void bind_i2c_lock(py::module_& m)
{
    py::class_<ad::I2cLockV1, PyI2cLock>(m, "I2cLockV1")
        .def(py::init<>())
        .def("lock", &ad::I2cLockV1::lock)
        .def("unlock", &ad::I2cLockV1::unlock)
        .def("try_lock", &ad::I2cLockV1::try_lock);
}

static void bind_i2c_interface(py::module_& m)
{
    py::class_<ad::I2cInterfaceV1, PyI2cInterface,
        std::shared_ptr<ad::I2cInterfaceV1>>(m, "I2cInterfaceV1")
        .def(py::init<>())
        .def_static(
            "get_service",
            [](std::shared_ptr<ad::Module> module, const std::string& instance_id,
                bool allow_null) {
                return ad::I2cInterfaceV1::get_service(
                    std::move(module), instance_id.c_str(), allow_null);
            },
            py::arg("module"), py::arg("instance_id"),
            py::arg("allow_null") = false)
        .def(
            "i2c_transaction",
            [](ad::I2cInterfaceV1& self, uint32_t peripheral_address,
                py::buffer write_bytes, size_t read_byte_count) {
                py::buffer_info info = write_bytes.request();
                std::vector<uint8_t> w(info.size * info.itemsize);
                std::memcpy(w.data(), info.ptr, w.size());
                std::vector<uint8_t> r(read_byte_count);
                const hololink_module_status_t s = self.i2c_transaction(
                    peripheral_address, w, r);
                if (s != HOLOLINK_MODULE_OK) {
                    throw std::runtime_error(
                        "While invoking I2cInterfaceV1.i2c_transaction: status "
                        + std::to_string(s));
                }
                return py::bytes(reinterpret_cast<const char*>(r.data()), r.size());
            },
            py::arg("peripheral_address"), py::arg("write_bytes"),
            py::arg("read_byte_count"))
        .def(
            "encode_i2c_request",
            [](ad::I2cInterfaceV1& self, ad::SequencerInterfaceV1& sequencer,
                uint32_t peripheral_i2c_address, py::buffer write_bytes,
                uint32_t read_byte_count) {
                py::buffer_info info = write_bytes.request();
                std::vector<uint8_t> w(info.size * info.itemsize);
                std::memcpy(w.data(), info.ptr, w.size());
                std::vector<unsigned> wi, ri;
                unsigned status_index = 0;
                const hololink_module_status_t s = self.encode_i2c_request(
                    sequencer, peripheral_i2c_address, w, read_byte_count,
                    wi, ri, status_index);
                if (s != HOLOLINK_MODULE_OK) {
                    throw std::runtime_error(
                        "While invoking I2cInterfaceV1.encode_i2c_request: status "
                        + std::to_string(s));
                }
                return py::make_tuple(wi, ri, status_index);
            },
            py::arg("sequencer"), py::arg("peripheral_i2c_address"),
            py::arg("write_bytes"), py::arg("read_byte_count"));
}

static void bind_sequencer(py::module_& m)
{
    py::class_<ad::SequencerInterfaceV1, PySequencer, std::shared_ptr<ad::SequencerInterfaceV1>>(
        m, "SequencerInterfaceV1")
        .def(py::init<>())
        .def_static(
            "get_service",
            [](std::shared_ptr<ad::Module> module, const std::string& instance_id,
                bool allow_null) {
                return ad::SequencerInterfaceV1::get_service(
                    std::move(module), instance_id.c_str(), allow_null);
            },
            py::arg("module"), py::arg("instance_id"),
            py::arg("allow_null") = false)
        .def("write_uint32", &ad::SequencerInterfaceV1::write_uint32,
            py::arg("address"), py::arg("data"))
        .def("read_uint32", &ad::SequencerInterfaceV1::read_uint32,
            py::arg("address"), py::arg("initial_value") = 0xFFFFFFFFu)
        .def("poll", &ad::SequencerInterfaceV1::poll,
            py::arg("address"), py::arg("mask"), py::arg("match"))
        .def("enable", &ad::SequencerInterfaceV1::enable)
        .def("location", &ad::SequencerInterfaceV1::location);
}

// Opaque RoceReceiverInterfaceV1 binding. No methods are exposed —
// applications drive the receiver indirectly by constructing
// RoceReceiverOp, which fetches the receiver via
// RoceReceiverV1::get_service and forwards into receiver->start(...)
// internally. The class_ is registered so pybind11 can resolve
// shared_ptr<RoceReceiverV1> at the Python boundary (e.g. for
// RoceDataChannelInterfaceV1.attach_receiver(receiver) when application
// code wants to hand a handle through).
static void bind_oscillator(py::module_& m)
{
    py::class_<ad::OscillatorInterfaceV1,
        std::shared_ptr<ad::OscillatorInterfaceV1>>(m, "OscillatorInterfaceV1")
        .def_static(
            "get_service",
            [](std::shared_ptr<ad::Module> module, const std::string& instance_id,
                bool allow_null) {
                return ad::OscillatorInterfaceV1::get_service(
                    std::move(module), instance_id.c_str(), allow_null);
            },
            py::arg("module"), py::arg("instance_id"),
            py::arg("allow_null") = false)
        .def_static(
            "get_service",
            [](const ad::EnumerationMetadata& metadata, bool allow_null) {
                return ad::OscillatorInterfaceV1::get_service(metadata, allow_null);
            },
            py::arg("metadata"), py::arg("allow_null") = false)
        .def("enable", &ad::OscillatorInterfaceV1::enable,
            py::arg("clocks_per_second"))
        .def("get_caps", &ad::OscillatorInterfaceV1::get_caps)
        .def("set_caps", &ad::OscillatorInterfaceV1::set_caps, py::arg("caps"));
}

static void bind_roce_receiver(py::module_& m)
{
    py::class_<ad::RoceReceiverInterfaceV1,
        std::shared_ptr<ad::RoceReceiverInterfaceV1>>(m, "RoceReceiverInterfaceV1")
        .def_static(
            "get_service",
            [](std::shared_ptr<ad::Module> module, const std::string& instance_id,
                bool allow_null) {
                return ad::RoceReceiverInterfaceV1::get_service(
                    std::move(module), instance_id.c_str(), allow_null);
            },
            py::arg("module"), py::arg("instance_id"),
            py::arg("allow_null") = false)
        .def_static(
            "get_service",
            [](const ad::EnumerationMetadata& metadata, bool allow_null) {
                return ad::RoceReceiverInterfaceV1::get_service(metadata, allow_null);
            },
            py::arg("metadata"), py::arg("allow_null") = false);
}

static void bind_data_channel(py::module_& m)
{
    // Per-channel anchor base. Owns the cache slot; Python sees the
    // anchor surface (enumeration_metadata, hololink) plus the
    // get_service entry points that route through it.
    py::class_<ad::DataChannelInterfaceV1, PyDataChannel,
        std::shared_ptr<ad::DataChannelInterfaceV1>>(m, "DataChannelInterfaceV1")
        .def(py::init<>())
        .def_static(
            "get_service",
            [](std::shared_ptr<ad::Module> module, const std::string& instance_id,
                bool allow_null) {
                return ad::DataChannelInterfaceV1::get_service(
                    std::move(module), instance_id.c_str(), allow_null);
            },
            py::arg("module"), py::arg("instance_id"),
            py::arg("allow_null") = false)
        .def_static(
            "get_service",
            [](const ad::EnumerationMetadata& metadata, bool allow_null) {
                return ad::DataChannelInterfaceV1::get_service(metadata, allow_null);
            },
            py::arg("metadata"), py::arg("allow_null") = false)
        .def("enumeration_metadata",
            &ad::DataChannelInterfaceV1::enumeration_metadata,
            py::return_value_policy::reference_internal)
        .def("hololink", &ad::DataChannelInterfaceV1::hololink)
        .def("device_lost", &ad::DataChannelInterfaceV1::device_lost,
            "Channel-level entry point for HololinkInterfaceV1.device_lost, to "
            "invalidate a lost device through a data channel already held.");
}

static void bind_roce_data_channel(py::module_& m)
{
    // RoCE transport — its own ConfigurableService, not derived from
    // DataChannelInterfaceV1. Composes with the per-channel anchor
    // internally; Python callers fetch the anchor separately via
    // DataChannelInterfaceV1.get_service when they need its metadata
    // or hololink.
    py::class_<ad::RoceDataChannelInterfaceV1, PyRoceDataChannel,
        std::shared_ptr<ad::RoceDataChannelInterfaceV1>>(m, "RoceDataChannelInterfaceV1")
        .def(py::init<>())
        .def_static(
            "get_service",
            [](std::shared_ptr<ad::Module> module, const std::string& instance_id,
                bool allow_null) {
                return ad::RoceDataChannelInterfaceV1::get_service(
                    std::move(module), instance_id.c_str(), allow_null);
            },
            py::arg("module"), py::arg("instance_id"),
            py::arg("allow_null") = false)
        .def_static(
            "get_service",
            [](const ad::EnumerationMetadata& metadata, bool allow_null) {
                return ad::RoceDataChannelInterfaceV1::get_service(metadata, allow_null);
            },
            py::arg("metadata"), py::arg("allow_null") = false)
        .def("attach_receiver", &ad::RoceDataChannelInterfaceV1::attach_receiver,
            py::arg("receiver"))
        .def("detach_receiver", &ad::RoceDataChannelInterfaceV1::detach_receiver)
        .def(
            "frame_end_sequencer",
            [](ad::RoceDataChannelInterfaceV1& self, bool allow_null) {
                return self.frame_end_sequencer<>(allow_null);
            },
            py::arg("allow_null") = false)
        .def(
            "get_hololink",
            [](ad::RoceDataChannelInterfaceV1& self, bool allow_null) {
                return self.get_hololink<>(allow_null);
            },
            py::arg("allow_null") = false);
}

// Vsync — abstract consumer-facing base. Bound as a polymorphic
// py::class_ so derived types (PtpPpsOutput) upcast cleanly when
// handed to Python sites that consume a Vsync (e.g. the module
// Vb1940Cam's `vsync=` keyword in its per-module pybind extension).
static void bind_vsync(py::module_& m)
{
    py::class_<ad::VsyncInterfaceV1, std::shared_ptr<ad::VsyncInterfaceV1>>(m, "VsyncInterfaceV1")
        .def_static(
            "get_service",
            [](std::shared_ptr<ad::Module> module, const std::string& instance_id,
                bool allow_null) {
                return ad::VsyncInterfaceV1::get_service(
                    std::move(module), instance_id.c_str(), allow_null);
            },
            py::arg("module"), py::arg("instance_id"),
            py::arg("allow_null") = false)
        .def("is_enabled", &ad::VsyncInterfaceV1::is_enabled)
        .def("start", &ad::VsyncInterfaceV1::start)
        .def("stop", &ad::VsyncInterfaceV1::stop);
}

// PtpPpsOutput — the PTP-PPS-driven concrete Vsync source. The
// py::class_ declares Vsync as a base so a Python shared_ptr<PtpPpsOutput>
// flows through any annotation expecting Vsync.
static void bind_ptp_pps_output(py::module_& m)
{
    py::class_<ad::PtpPpsOutputInterfaceV1, ad::VsyncInterfaceV1,
        std::shared_ptr<ad::PtpPpsOutputInterfaceV1>>(m, "PtpPpsOutputInterfaceV1")
        .def_static(
            "get_service",
            [](std::shared_ptr<ad::Module> module, const std::string& instance_id,
                bool allow_null) {
                return ad::PtpPpsOutputInterfaceV1::get_service(
                    std::move(module), instance_id.c_str(), allow_null);
            },
            py::arg("module"), py::arg("instance_id"),
            py::arg("allow_null") = false)
        .def_static(
            "get_service",
            [](const ad::EnumerationMetadata& metadata, bool allow_null) {
                return ad::PtpPpsOutputInterfaceV1::get_service(metadata, allow_null);
            },
            py::arg("metadata"), py::arg("allow_null") = false)
        .def("enable", &ad::PtpPpsOutputInterfaceV1::enable, py::arg("frequency_hz"))
        .def("disable", &ad::PtpPpsOutputInterfaceV1::disable);
}

static void bind_hololink_interface(py::module_& m)
{
    // Opaque RAII handle returned by on_reset(); holding it keeps the
    // callback registered, dropping it (e.g. when the owning camera is
    // garbage-collected) unregisters the callback. No methods are exposed.
    py::class_<ad::HololinkInterfaceV1::ResetRegistration,
        std::shared_ptr<ad::HololinkInterfaceV1::ResetRegistration>>(
        m, "ResetRegistration");

    py::class_<ad::HololinkInterfaceV1, PyHololinkInterface,
        std::shared_ptr<ad::HololinkInterfaceV1>>(m, "HololinkInterfaceV1")
        .def(py::init<>())
        .def_static(
            "get_service",
            [](std::shared_ptr<ad::Module> module, const std::string& instance_id,
                bool allow_null) {
                return ad::HololinkInterfaceV1::get_service(
                    std::move(module), instance_id.c_str(), allow_null);
            },
            py::arg("module"), py::arg("instance_id"),
            py::arg("allow_null") = false)
        .def_static(
            "get_service",
            [](const ad::EnumerationMetadata& metadata, bool allow_null) {
                return ad::HololinkInterfaceV1::get_service(metadata, allow_null);
            },
            py::arg("metadata"), py::arg("allow_null") = false)
        .def("enumeration_metadata",
            &ad::HololinkInterfaceV1::enumeration_metadata,
            py::return_value_policy::reference_internal)
        .def("default_data_channel", &ad::HololinkInterfaceV1::default_data_channel)
        .def("start", &ad::HololinkInterfaceV1::start)
        .def("stop", &ad::HololinkInterfaceV1::stop)
        .def("reset", &ad::HololinkInterfaceV1::reset)
        .def("configure_hsb", &ad::HololinkInterfaceV1::configure_hsb)
        .def("device_lost", &ad::HololinkInterfaceV1::device_lost,
            "Invalidate this device's cached services so re-resolving them "
            "after rediscovery yields fresh instances; the caller must "
            "re-fetch to resync. Called by the reconnection path on loss.")
        .def("ptp_synchronize", &ad::HololinkInterfaceV1::ptp_synchronize,
            // Blocks up to the backing default timeout (~20s) waiting for
            // PTP lock; release the GIL so other Python threads (e.g. the
            // receivers' monitor callbacks) keep running meanwhile.
            py::call_guard<py::gil_scoped_release>(),
            "Block until the board's clock is disciplined to the PTP "
            "grandmaster; returns True on success, False on timeout.")
        .def("on_reset", &ad::HololinkInterfaceV1::on_reset, py::arg("callback"),
            "Register a callback fired each time the board completes a "
            "reset (after the FPGA reboots). Multiple callbacks may be "
            "registered; each fires once per reset(). Returns a handle: "
            "keep it alive while the callback should fire, drop it (e.g. "
            "let the owning object be garbage-collected) to unregister.")
        .def("write_uint32", &ad::HololinkInterfaceV1::write_uint32,
            py::arg("addresses"), py::arg("values"))
        .def(
            "read_uint32",
            [](ad::HololinkInterfaceV1& self,
                const std::vector<uint32_t>& addresses) {
                std::vector<uint32_t> out;
                const hololink_module_status_t s = self.read_uint32(addresses, out);
                if (s != HOLOLINK_MODULE_OK) {
                    throw std::runtime_error(
                        "While invoking HololinkInterfaceV1.read_uint32: status "
                        + std::to_string(s));
                }
                return out;
            },
            py::arg("addresses"))
        .def("and_uint32", &ad::HololinkInterfaceV1::and_uint32,
            py::arg("address"), py::arg("mask"))
        .def("or_uint32", &ad::HololinkInterfaceV1::or_uint32,
            py::arg("address"), py::arg("mask"))
        .def(
            "i2c_lock",
            [](ad::HololinkInterfaceV1& self) -> ad::I2cLockV1* {
                std::unique_ptr<ad::I2cLockV1> lock;
                const hololink_module_status_t s = self.i2c_lock(lock);
                if (s != HOLOLINK_MODULE_OK) {
                    throw std::runtime_error(
                        "While invoking HololinkInterfaceV1.i2c_lock: status "
                        + std::to_string(s));
                }
                return lock.release();
            },
            py::return_value_policy::take_ownership)
        .def(
            "get_roce_data_channel",
            [](ad::HololinkInterfaceV1& self,
                const ad::EnumerationMetadata& md, bool allow_null) {
                return self.get_roce_data_channel<>(md, allow_null);
            },
            py::arg("metadata"), py::arg("allow_null") = false)
        .def(
            "get_i2c",
            [](ad::HololinkInterfaceV1& self, uint32_t bus, uint32_t address,
                bool allow_null) {
                return self.get_i2c<>(bus, address, allow_null);
            },
            py::arg("bus"), py::arg("address"), py::arg("allow_null") = false)
        .def(
            "software_sequencer",
            [](ad::HololinkInterfaceV1& self, bool allow_null) {
                return self.software_sequencer<>(allow_null);
            },
            py::arg("allow_null") = false)
        .def(
            "gpio0_sequencer",
            [](ad::HololinkInterfaceV1& self, bool allow_null) {
                return self.gpio0_sequencer<>(allow_null);
            },
            py::arg("allow_null") = false)
        .def(
            "gpio1_sequencer",
            [](ad::HololinkInterfaceV1& self, bool allow_null) {
                return self.gpio1_sequencer<>(allow_null);
            },
            py::arg("allow_null") = false)
        .def(
            "sif0_frame_start_sequencer",
            [](ad::HololinkInterfaceV1& self, bool allow_null) {
                return self.sif0_frame_start_sequencer<>(allow_null);
            },
            py::arg("allow_null") = false)
        .def(
            "sif1_frame_start_sequencer",
            [](ad::HololinkInterfaceV1& self, bool allow_null) {
                return self.sif1_frame_start_sequencer<>(allow_null);
            },
            py::arg("allow_null") = false)
        .def(
            "ptp_pps_output",
            [](ad::HololinkInterfaceV1& self, bool allow_null) {
                return self.ptp_pps_output<>(allow_null);
            },
            py::arg("allow_null") = false)
        .def(
            "null_vsync",
            [](ad::HololinkInterfaceV1& self, bool allow_null) {
                return self.null_vsync<>(allow_null);
            },
            py::arg("allow_null") = false);
}

static void bind_enumeration_interface(py::module_& m)
{
    py::class_<ad::EnumerationInterfaceV1, PyEnumerationInterface,
        std::shared_ptr<ad::EnumerationInterfaceV1>>(m, "EnumerationInterfaceV1")
        .def(py::init<>())
        .def_static(
            "get_service",
            [](std::shared_ptr<ad::Module> module, bool allow_null) {
                return ad::EnumerationInterfaceV1::get_service(
                    std::move(module), allow_null);
            },
            py::arg("module"), py::arg("allow_null") = false)
        .def(
            "update_metadata",
            [](ad::EnumerationInterfaceV1& self,
                ad::EnumerationMetadata& metadata, py::object raw_packet) {
                hololink_module_status_t st;
                if (raw_packet.is_none()) {
                    st = self.update_metadata(metadata, nullptr, 0);
                } else {
                    // Validated buffer_info stays alive (keeping the
                    // Py_buffer pinned) until self.update_metadata
                    // returns, so the underlying bytes can't be
                    // mutated or freed mid-call.
                    py::buffer_info info = require_1d_contiguous_byte_buffer(
                        raw_packet,
                        "invoking EnumerationInterfaceV1.update_metadata raw_packet");
                    st = self.update_metadata(
                        metadata,
                        static_cast<const uint8_t*>(info.ptr),
                        static_cast<size_t>(info.size));
                }
                if (st != HOLOLINK_MODULE_OK) {
                    throw std::runtime_error(
                        "While invoking EnumerationInterfaceV1.update_metadata: "
                        "status "
                        + std::to_string(st));
                }
            },
            py::arg("metadata"), py::arg("raw_packet") = py::none());
}

static void bind_logging_interface(py::module_& m)
{
    py::enum_<ad::LogLevel>(m, "LogLevel")
        .value("Trace", ad::LogLevel::Trace)
        .value("Debug", ad::LogLevel::Debug)
        .value("Info", ad::LogLevel::Info)
        .value("Warning", ad::LogLevel::Warning)
        .value("Error", ad::LogLevel::Error);

    py::class_<ad::LoggingInterfaceV1, PyLoggingInterface,
        std::shared_ptr<ad::LoggingInterfaceV1>>(m, "LoggingInterfaceV1")
        .def(py::init<>())
        .def_static(
            "get_service",
            [](std::shared_ptr<ad::Module> module, bool allow_null) {
                return ad::LoggingInterfaceV1::get_service(
                    std::move(module), allow_null);
            },
            py::arg("module"), py::arg("allow_null") = false)
        .def("level", &ad::LoggingInterfaceV1::level)
        .def(
            "log",
            [](ad::LoggingInterfaceV1& self, ad::LogLevel level_,
                const std::string& file, unsigned line,
                const std::string& function, const std::string& message) {
                self.log(level_, file.c_str(), line, function.c_str(),
                    message.c_str());
            },
            py::arg("level"), py::arg("file"), py::arg("line"),
            py::arg("function"), py::arg("message"));
}

// Wrap a Python callable in a Reactor::Callback whose held py::object
// is destroyed under the GIL. The core::Reactor dispatch thread (see
// reactor_impl.cpp:45) does not hold the GIL when it releases the
// shared_ptr<Callback>, so capturing a bare py::object in the lambda
// would Py_DECREF without the GIL — undefined behavior.
static std::shared_ptr<ad::ReactorV1::Callback> make_python_reactor_callback(
    py::object callback)
{
    std::shared_ptr<py::object> holder(
        new py::object(std::move(callback)),
        [](py::object* p) {
            py::gil_scoped_acquire gil;
            delete p;
        });
    return std::make_shared<ad::ReactorV1::Callback>([holder]() {
        py::gil_scoped_acquire gil;
        (*holder)();
    });
}

static void bind_reactor(py::module_& m)
{
    py::class_<ad::ReactorV1::AlarmEntry,
        std::shared_ptr<ad::ReactorV1::AlarmEntry>>(m, "AlarmEntry");

    // Reactor is host-published (no Python subclassing); the binding
    // exposes the consumer surface only.
    py::class_<ad::ReactorV1, std::shared_ptr<ad::ReactorV1>>(m, "ReactorV1")
        .def_static(
            "get_service",
            [](std::shared_ptr<ad::Module> module, bool allow_null) {
                return ad::ReactorV1::get_service(std::move(module), allow_null);
            },
            py::arg("module"), py::arg("allow_null") = false)
        .def(
            "now",
            [](const ad::ReactorV1& self) {
                struct timespec ts = self.now();
                return py::make_tuple(static_cast<int64_t>(ts.tv_sec),
                    static_cast<int64_t>(ts.tv_nsec));
            })
        .def(
            "add_callback",
            [](ad::ReactorV1& self, py::object callback) {
                self.add_callback(
                    make_python_reactor_callback(std::move(callback)));
            },
            py::arg("callback"))
        .def(
            "add_alarm_s",
            [](ad::ReactorV1& self, float seconds, py::object callback) {
                return self.add_alarm_s(seconds,
                    make_python_reactor_callback(std::move(callback)));
            },
            py::arg("seconds"), py::arg("callback"))
        .def("cancel_alarm", &ad::ReactorV1::cancel_alarm, py::arg("handle"))
        .def("is_current_thread", &ad::ReactorV1::is_current_thread);
}

static void bind_adapter(py::module_& m)
{
    // Opaque handle for register_ip / register_all. Python code
    // holds the returned object, passes it to unregister(), and
    // doesn't inspect anything inside.
    py::class_<ad::EnumerationCallback,
        std::shared_ptr<ad::EnumerationCallback>>(m, "EnumerationCallbackHandle");

    py::class_<ad::Adapter, std::unique_ptr<ad::Adapter, py::nodelete>>(m, "Adapter")
        .def_static(
            "get_adapter",
            []() -> ad::Adapter& { return ad::Adapter::get_adapter(); },
            py::return_value_policy::reference)
        .def(
            "set_module_directory",
            [](ad::Adapter& self, const std::string& dir) {
                self.set_module_directory(std::filesystem::path(dir));
            },
            py::arg("dir"))
        .def(
            "load_module",
            [](ad::Adapter& self, const std::string& so_path) {
                return self.load_module(std::filesystem::path(so_path));
            },
            py::arg("so_path"))
        .def(
            "enumerate",
            [](ad::Adapter& self, ad::EnumerationMetadata metadata) {
                self.enumerate(std::move(metadata));
            },
            py::arg("metadata"))
        .def("start_bootp_listener", &ad::Adapter::start_bootp_listener,
            py::arg("port") = uint32_t { 12267 })
        .def("stop_bootp_listener", &ad::Adapter::stop_bootp_listener)
        .def(
            "wait_for_channel",
            [](ad::Adapter& self, const std::string& peer_ip, double timeout_s) {
                const auto timeout
                    = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::duration<double>(timeout_s));
                return self.wait_for_channel(peer_ip, timeout);
            },
            py::arg("peer_ip"), py::arg("timeout_s"),
            // Release the GIL during the blocking wait so another
            // Python thread can call adapter.enumerate(...) — the
            // very thing this call is waiting for. Without this the
            // wait deadlocks on its own GIL hold and times out every
            // time the announcement is fired from a Python thread.
            py::call_guard<py::gil_scoped_release>(),
            "Block up to `timeout_s` seconds for an enumeration announcement "
            "for `peer_ip` that arrives after this call started waiting, and "
            "return its metadata; raises RuntimeError on timeout. Built on "
            "register_ip — earlier announcements don't count.")
        .def(
            "register_ip",
            [](ad::Adapter& self, const std::string& peer_ip,
                std::function<void(const ad::EnumerationMetadata&)> callback) {
                return self.register_ip(peer_ip, std::move(callback));
            },
            py::arg("peer_ip"), py::arg("callback"),
            "Register a callback that fires for every enumeration "
            "announcement matching `peer_ip`. Returns an opaque handle "
            "to pass to unregister().")
        .def(
            "register_all",
            [](ad::Adapter& self,
                std::function<void(const ad::EnumerationMetadata&)> callback) {
                return self.register_all(std::move(callback));
            },
            py::arg("callback"),
            "Register a callback that fires for every enumeration "
            "announcement, regardless of peer_ip. Returns an opaque "
            "handle to pass to unregister().")
        .def("unregister", &ad::Adapter::unregister, py::arg("handle"))
        .def(
            "use_sensor",
            [](ad::Adapter& self, ad::EnumerationMetadata& metadata,
                int64_t sensor_number) {
                self.use_sensor(metadata, sensor_number);
            },
            py::arg("metadata"), py::arg("sensor_number"),
            "Stamp per-sensor channel-configuration fields onto the "
            "metadata in place (data_channel locator key plus the "
            "supplement's per-sensor address fields). Stereo flows "
            "clone the base enumeration metadata and call this once "
            "per camera before constructing per-sensor handles.")
        .def(
            "use_mtu",
            [](ad::Adapter& self, ad::EnumerationMetadata& metadata,
                uint32_t mtu) {
                self.use_mtu(metadata, mtu);
            },
            py::arg("metadata"), py::arg("mtu"),
            "Stamp the requested MTU onto the metadata in place. The "
            "per-channel RoCE constructors read the stamped value when "
            "sizing packets; call this before constructing per-channel "
            "services to use a non-default MTU.")
        .def(
            "use_multicast",
            [](ad::Adapter& self, ad::EnumerationMetadata& metadata,
                const std::string& address, uint16_t port) {
                self.use_multicast(metadata, address, port);
            },
            py::arg("metadata"), py::arg("address"), py::arg("port"),
            "Stamp a multicast destination (group address + port) onto "
            "the metadata in place. The RoCE data plane copies these "
            "onto the legacy DataChannel, which programs the FPGA to "
            "send frames to the multicast group; call this before "
            "constructing per-channel services.")
        .def(
            "get_module",
            [](ad::Adapter& self, const ad::EnumerationMetadata& metadata) {
                return self.get_module(metadata);
            },
            py::arg("metadata"))
        .def(
            "reactor",
            [](ad::Adapter& self) {
                return ad::ReactorV1::get_service(
                    self.host_publisher()->self_module());
            },
            "The host ReactorV1 singleton (published on the host module, not a "
            "device module).");
}

PYBIND11_MODULE(_hololink_py_module, m)
{
    m.doc() = "hololink_module native bindings (core surface)";

    bind_enumeration_metadata(m);
    bind_module(m);
    bind_csi_converter(m);
    bind_frame_metadata(m);

    bind_i2c_lock(m);
    bind_i2c_interface(m);
    bind_sequencer(m);
    bind_oscillator(m);
    bind_roce_receiver(m);
    bind_data_channel(m);
    bind_roce_data_channel(m);
    bind_vsync(m);
    bind_ptp_pps_output(m);
    bind_hololink_interface(m);

    bind_enumeration_interface(m);
    bind_logging_interface(m);
    bind_reactor(m);

    bind_adapter(m);

    // Convenience logging functions routing through the module HSB_LOG_*
    // macros (which dispatch to the registered hsb_logger_cache and safely
    // no-op when none is set). Adapter equivalents of the legacy
    // hololink.hsb_log_* helpers so tests / applications can log through the
    // module without fetching a LoggingInterfaceV1 service by hand. The
    // message is passed as a format argument, not a format string, so braces
    // in the text are not interpreted.
    m.def(
        "hsb_log_trace", [](const std::string& message) { HSB_LOG_TRACE("{}", message); },
        py::arg("message"), "Log a message at TRACE level through the module logger.");
    m.def(
        "hsb_log_debug", [](const std::string& message) { HSB_LOG_DEBUG("{}", message); },
        py::arg("message"), "Log a message at DEBUG level through the module logger.");
    m.def(
        "hsb_log_info", [](const std::string& message) { HSB_LOG_INFO("{}", message); },
        py::arg("message"), "Log a message at INFO level through the module logger.");
    m.def(
        "hsb_log_warn", [](const std::string& message) { HSB_LOG_WARN("{}", message); },
        py::arg("message"), "Log a message at WARNING level through the module logger.");
    m.def(
        "hsb_log_error", [](const std::string& message) { HSB_LOG_ERROR("{}", message); },
        py::arg("message"), "Log a message at ERROR level through the module logger.");

    // Adapter-owned networking constants (mirrors of the legacy hololink::core
    // values) so Python callers name no legacy hololink constant. See
    // hololink/module/networking.hpp.
    m.attr("DEFAULT_MTU") = ad::DEFAULT_MTU;

    // Status codes (hololink/module/status.h) as module-level ints, so Python
    // overrides of V1 interface methods return e.g. HOLOLINK_MODULE_OK rather
    // than a bare integer. They cross the trampoline boundary as the underlying
    // uint32_t.
    m.attr("HOLOLINK_MODULE_OK") = HOLOLINK_MODULE_OK;
    m.attr("HOLOLINK_MODULE_INVALID_PARAMETER") = HOLOLINK_MODULE_INVALID_PARAMETER;
    m.attr("HOLOLINK_MODULE_NOT_FOUND") = HOLOLINK_MODULE_NOT_FOUND;
    m.attr("HOLOLINK_MODULE_NETWORK_ERROR") = HOLOLINK_MODULE_NETWORK_ERROR;
    m.attr("HOLOLINK_MODULE_TIMEOUT") = HOLOLINK_MODULE_TIMEOUT;
    m.attr("HOLOLINK_MODULE_ABI_MISMATCH") = HOLOLINK_MODULE_ABI_MISMATCH;
    m.attr("HOLOLINK_MODULE_INIT_FAILED") = HOLOLINK_MODULE_INIT_FAILED;
    m.attr("HOLOLINK_MODULE_ENUMERATION_SKIPPED") = HOLOLINK_MODULE_ENUMERATION_SKIPPED;

#ifdef HOLOLINK_BUILD_ROCE
    // ibv_device_for_peer links ibverbs and is built only with RoCE; its
    // only consumer is the RoCE receiver operator (which derives ibv_name
    // internally), so the binding is gated rather than stubbed — nothing
    // calls it in a non-RoCE build.
    m.def(
        "ibv_device_for_peer",
        [](const std::string& peer_ip) {
            return ad::ibv_device_for_peer(peer_ip);
        },
        py::arg("peer_ip"),
        "Return (ibv_name, ibv_port) for the IB device whose underlying "
        "kernel netdev is the route-resolved local interface for peer_ip. "
        "Raises RuntimeError if no matching device is found.");

    // infiniband_devices enumerates IB devices via ibverbs; like
    // ibv_device_for_peer it is gated on RoCE (ibverbs is only linked then).
    // Module equivalent of the legacy hololink.infiniband_devices().
    m.def(
        "infiniband_devices",
        []() {
            std::vector<std::string> device_names;
            int num_devices = 0;
            struct ibv_device** devices = ibv_get_device_list(&num_devices);
            if (!devices) {
                return device_names;
            }
            for (int i = 0; i < num_devices; i++) {
                device_names.push_back(ibv_get_device_name(devices[i]));
            }
            ibv_free_device_list(devices);
            std::sort(device_names.begin(), device_names.end());
            return device_names;
        },
        "Return a sorted list of Infiniband device names.");
#endif
}
