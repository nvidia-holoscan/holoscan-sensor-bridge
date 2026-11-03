/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>

// holoscan::RuntimeError, thrown below if operator-catalog registration fails.
#include <holoscan/core/errors.hpp>
#include <holoscan/python/constructor_handle_capsule.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <cuda.h>

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

#include "hololink/module/operators/csi_to_bayer_op.hpp"
#include "hololink/module/operators/linux_network_receiver.hpp"
#include "hololink/module/operators/module_operator_catalog.hpp"
#include "hololink/module/operators/network_receiver.hpp"
#include "hololink/module/operators/sensor_device.hpp"
#include "hololink/module/operators/sensor_factory.hpp"

#ifdef HOLOLINK_BUILD_FUSA
#include "hololink/module/operators/fusa_coe_capture_op.hpp"
#endif

#include "hololink/module/sensors/vb1940/vb1940_cam.hpp"

#include "hololink/module/taurotech_da326/taurotech_da326.hpp"

#ifdef HOLOLINK_BUILD_ROCE
#include "hololink/module/operators/roce_network_receiver.hpp"
#include <infiniband/verbs.h> // ibv_get_device_list for infiniband_devices()
#endif

namespace nb = nanobind;
namespace ad = hololink::module;
namespace v1940 = hololink::module::sensors::vb1940;

// RAII wrapper around a raw CPython Py_buffer. nanobind dropped pybind11's
// py::buffer / py::buffer_info wrapper entirely (see
// build/_deps/nanobind-src/docs/porting.rst's "Buffer protocol binding...
// removed" bullet, in favor of nb::ndarray<>, which targets typed numeric
// arrays rather than "accept any buffer-protocol object generically"). This
// plugs the same gap py::buffer_info filled: owns the Py_buffer for exactly
// the request's lifetime and releases it on destruction.
struct PyBufferRequest {
    Py_buffer view {};
    bool acquired = false;

    PyBufferRequest() = default;
    PyBufferRequest(const PyBufferRequest&) = delete;
    PyBufferRequest& operator=(const PyBufferRequest&) = delete;
    PyBufferRequest(PyBufferRequest&& other) noexcept { *this = std::move(other); }
    PyBufferRequest& operator=(PyBufferRequest&& other) noexcept
    {
        if (this != &other) {
            if (acquired) {
                PyBuffer_Release(&view);
            }
            view = other.view;
            acquired = other.acquired;
            other.acquired = false;
        }
        return *this;
    }
    ~PyBufferRequest()
    {
        if (acquired) {
            PyBuffer_Release(&view);
        }
    }
};

// Raw buffer-protocol request with no shape/stride/itemsize validation --
// the primitive pybind11's py::buffer::request() provided (PyObject_GetBuffer
// with PyBUF_STRIDES | PyBUF_FORMAT, matching pybind11's pytypes.h
// buffer::request() flags exactly, confirmed by reading that vendored source
// directly). require_1d_contiguous_byte_buffer() below layers validation on
// top of this. Used directly by call sites that, like the original pybind11
// bindings, only need a pointer + total byte count and perform no shape
// validation of their own (e.g. I2cInterfaceV1.i2c_transaction's
// write_bytes, which pybind11 bound as a bare `py::buffer` parameter --
// nanobind has no parameter type that auto-invokes the buffer protocol, so
// the request moved into the lambda body here). Throws nanobind's
// python_error (surfacing CPython's own "does not support the buffer
// protocol" TypeError) if `obj` doesn't support the protocol.
static PyBufferRequest request_py_buffer(nb::handle obj)
{
    PyBufferRequest info;
    if (PyObject_GetBuffer(obj.ptr(), &info.view, PyBUF_STRIDES | PyBUF_FORMAT) != 0) {
        throw nb::python_error();
    }
    info.acquired = true;
    return info;
}

// Validates that `obj` exposes the buffer protocol and the returned
// buffer is a 1-D, itemsize-1, C-contiguous byte sequence — i.e. a
// `bytes` / `bytearray` / contiguous `memoryview` of bytes. Returns the
// populated PyBufferRequest; the caller copies view.len bytes from
// view.buf under the GIL (the caller holds it; PyBufferRequest keeps the
// underlying Py_buffer alive until its destructor runs). Throws
// std::runtime_error with `context` woven into the message on any
// violation — context should read as a verb phrase, e.g.
// "invoking FrameMetadataInterfaceV1.decode host_memory".
static PyBufferRequest require_1d_contiguous_byte_buffer(
    nb::handle obj, const char* context)
{
    if (!PyObject_CheckBuffer(obj.ptr())) {
        throw std::runtime_error(
            std::string("While ") + context
            + ": value must support the buffer protocol "
              "(bytes, bytearray, memoryview, ...)");
    }
    PyBufferRequest info = request_py_buffer(obj);
    if (info.view.ndim != 1) {
        throw std::runtime_error(
            std::string("While ") + context
            + ": buffer must be 1-dimensional");
    }
    if (info.view.itemsize != 1) {
        throw std::runtime_error(
            std::string("While ") + context
            + ": buffer itemsize must be 1 byte");
    }
    // For a 1-D itemsize-1 buffer, C-contiguous means strides[0] == 1.
    // PyBUF_STRIDES normally guarantees non-null strides, but some
    // exporters (e.g. ctypes objects) ignore the flag and report NULL
    // strides anyway -- pybind11's buffer_info treated that as implicitly
    // contiguous (it synthesizes C-style strides when the exporter omits
    // them; see buffer_info.h's own comment to that effect), so this
    // mirrors that by skipping the check when strides is null rather than
    // rejecting the buffer.
    if (info.view.strides != nullptr && info.view.strides[0] != 1) {
        throw std::runtime_error(
            std::string("While ") + context
            + ": buffer must be C-contiguous");
    }
    return info;
}

// -------------------------------------------------------------------
// Trampoline classes — let Python subclass each abstract V1 interface.
// Each trampoline forwards every pure-virtual to a Python override via
// NB_OVERRIDE_PURE (or, for the handful of methods needing custom
// argument marshaling that NB_OVERRIDE_PURE's automatic arg-forwarding
// can't handle, the manual nanobind::detail::ticket pattern that
// NB_OVERRIDE_PURE_NAME's own macro expansion uses internally -- see each
// such method's own comment), exposing the C++ method under the same
// Python name. The .so is built with hidden visibility, so the
// trampolines stay internal to this translation unit despite living
// at namespace scope (no anonymous namespace).
// -------------------------------------------------------------------

class PyI2cLock : public ad::I2cLockV1 {
public:
    NB_TRAMPOLINE(ad::I2cLockV1, 3);

    void lock() override
    {
        NB_OVERRIDE_PURE(lock);
    }
    void unlock() override
    {
        NB_OVERRIDE_PURE(unlock);
    }
    bool try_lock() override
    {
        NB_OVERRIDE_PURE(try_lock);
    }
};

// I2cInterfaceV1 trampoline. The Python signature for i2c_transaction
// is (peripheral_address, write_bytes, read_byte_count) -> bytes;
// the trampoline translates the C++ in/out vector into that shape.
class PyI2cInterface : public ad::I2cInterfaceV1 {
public:
    NB_TRAMPOLINE(ad::I2cInterfaceV1, 2);

    hololink_module_status_t i2c_transaction(
        uint32_t peripheral_address,
        const std::vector<uint8_t>& write_bytes,
        std::vector<uint8_t>& read_bytes) override
    {
        // Manual override dispatch (not NB_OVERRIDE_PURE): the Python-facing
        // argument/return shape (bytes in, bytes out) differs from the C++
        // signature's in/out-vector shape, so this needs marshaling
        // NB_OVERRIDE_PURE's automatic arg-forwarding/return-casting can't
        // do. i2c_transaction is pure virtual in i2c.hpp, so constructing
        // `ticket` with pure=true already raises an appropriate Python error
        // if nothing overrides it -- no separate "if (!override) throw"
        // check needed (see trampoline.cpp's trampoline_enter()).
        nb::gil_scoped_acquire gil;
        constexpr uint64_t hash = nanobind::detail::str_hash("i2c_transaction");
        nanobind::detail::ticket ticket(nb_trampoline, "i2c_transaction", hash, /*pure=*/true);
        const size_t want = read_bytes.size();
        nb::object result = nb_trampoline.base().attr(ticket.key)(
            peripheral_address,
            nb::bytes(write_bytes.data(), write_bytes.size()),
            want);
        // Accept bytes / bytearray / any buffer-protocol object.
        // nb::cast<std::string> would also accept str and silently
        // UTF-8-encode it, which corrupts arbitrary binary payloads.
        PyBufferRequest info = require_1d_contiguous_byte_buffer(
            result,
            "invoking I2cInterfaceV1.i2c_transaction override return value");
        if (static_cast<size_t>(info.view.len) != want) {
            throw std::runtime_error(
                "While invoking I2cInterfaceV1.i2c_transaction override: "
                "returned byte count does not match read_byte_count");
        }
        // GIL is held (acquired above), so the copy is safe; PyBufferRequest
        // also keeps the Py_buffer alive for the lifetime of `info`.
        std::memcpy(read_bytes.data(), info.view.buf, want);
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
        // Manual override dispatch: same reasoning as i2c_transaction above
        // (bytes-in/tuple-out Python shape vs. the C++ in/out-vector-refs
        // shape); encode_i2c_request is also pure virtual.
        nb::gil_scoped_acquire gil;
        constexpr uint64_t hash = nanobind::detail::str_hash("encode_i2c_request");
        nanobind::detail::ticket ticket(nb_trampoline, "encode_i2c_request", hash, /*pure=*/true);
        nb::object result = nb_trampoline.base().attr(ticket.key)(
            nb::cast(&sequencer, nb::rv_policy::reference),
            peripheral_i2c_address,
            nb::bytes(write_bytes.data(), write_bytes.size()),
            read_byte_count);
        nb::tuple t = nb::cast<nb::tuple>(result);
        out_write_indexes = nb::cast<std::vector<unsigned>>(t[0]);
        out_read_indexes = nb::cast<std::vector<unsigned>>(t[1]);
        out_status_index = nb::cast<unsigned>(t[2]);
        return HOLOLINK_MODULE_OK;
    }
};

class PySequencer : public ad::SequencerInterfaceV1 {
public:
    NB_TRAMPOLINE(ad::SequencerInterfaceV1, 5);

    unsigned write_uint32(uint32_t address, uint32_t data) override
    {
        NB_OVERRIDE_PURE(write_uint32, address, data);
    }
    unsigned read_uint32(uint32_t address, uint32_t initial_value) override
    {
        NB_OVERRIDE_PURE(read_uint32, address, initial_value);
    }
    unsigned poll(uint32_t address, uint32_t mask, uint32_t match) override
    {
        NB_OVERRIDE_PURE(poll, address, mask, match);
    }
    hololink_module_status_t enable() override
    {
        NB_OVERRIDE_PURE(enable);
    }
    uint32_t location() override
    {
        NB_OVERRIDE_PURE(location);
    }
};

// DataChannel trampoline — per-channel anchor. Pure virtuals are the
// transport-agnostic enumeration_metadata / hololink accessors.
class PyDataChannel : public ad::DataChannelInterfaceV1 {
public:
    NB_TRAMPOLINE(ad::DataChannelInterfaceV1, 3);

    const ad::EnumerationMetadata& enumeration_metadata() const override
    {
        NB_OVERRIDE_PURE(enumeration_metadata);
    }
    std::shared_ptr<ad::HololinkInterfaceV1> hololink() const override
    {
        NB_OVERRIDE_PURE(hololink);
    }
    hololink_module_status_t device_lost() override
    {
        NB_OVERRIDE_PURE(device_lost);
    }
};

// RoceDataChannel trampoline. Standalone — does NOT inherit from
// DataChannelInterfaceV1. RoCE-typed callers reach the channel's
// metadata + hololink by holding a separate shared_ptr to the
// DataChannelInterfaceV1 anchor (the same instance the impl composes
// with internally).
class PyRoceDataChannel : public ad::RoceDataChannelInterfaceV1 {
public:
    NB_TRAMPOLINE(ad::RoceDataChannelInterfaceV1, 4);

    hololink_module_status_t attach_receiver(
        std::shared_ptr<ad::RoceReceiverInterfaceV1> receiver) override
    {
        NB_OVERRIDE_PURE(attach_receiver, receiver);
    }
    hololink_module_status_t detach_receiver() override
    {
        NB_OVERRIDE_PURE(detach_receiver);
    }

    std::string frame_end_sequencer_instance_id() override
    {
        NB_OVERRIDE_PURE(frame_end_sequencer_instance_id);
    }
    std::string parent_hololink_instance_id() override
    {
        NB_OVERRIDE_PURE(parent_hololink_instance_id);
    }
};

class PyHololinkInterface : public ad::HololinkInterfaceV1 {
public:
    NB_TRAMPOLINE(ad::HololinkInterfaceV1, 18);

    const ad::EnumerationMetadata& enumeration_metadata() const override
    {
        NB_OVERRIDE_PURE(enumeration_metadata);
    }

    std::shared_ptr<ad::DataChannelInterfaceV1> default_data_channel() const override
    {
        NB_OVERRIDE_PURE(default_data_channel);
    }

    hololink_module_status_t start() override
    {
        NB_OVERRIDE_PURE(start);
    }
    hololink_module_status_t stop() override
    {
        NB_OVERRIDE_PURE(stop);
    }
    hololink_module_status_t reset() override
    {
        NB_OVERRIDE_PURE(reset);
    }
    hololink_module_status_t configure_hsb() override
    {
        NB_OVERRIDE_PURE(configure_hsb);
    }
    hololink_module_status_t device_lost() override
    {
        NB_OVERRIDE_PURE(device_lost);
    }
    bool ptp_synchronize() override
    {
        NB_OVERRIDE_PURE(ptp_synchronize);
    }
    std::shared_ptr<ad::HololinkInterfaceV1::ResetRegistration> on_reset(
        std::function<void()> callback) override
    {
        NB_OVERRIDE_PURE(on_reset, callback);
    }
    hololink_module_status_t write_uint32(
        const std::vector<uint32_t>& addresses,
        const std::vector<uint32_t>& values) override
    {
        NB_OVERRIDE_PURE(write_uint32, addresses, values);
    }
    hololink_module_status_t read_uint32(
        const std::vector<uint32_t>& addresses,
        std::vector<uint32_t>& out_values) override
    {
        // Manual override dispatch: the Python override returns a plain
        // list, which has to be cast and written into the out-param here;
        // NB_OVERRIDE_PURE has no out-param support. Pure virtual, so the
        // ticket's own pure=true check covers "subclass didn't override".
        nb::gil_scoped_acquire gil;
        constexpr uint64_t hash = nanobind::detail::str_hash("read_uint32");
        nanobind::detail::ticket ticket(nb_trampoline, "read_uint32", hash, /*pure=*/true);
        nb::object result = nb_trampoline.base().attr(ticket.key)(addresses);
        out_values = nb::cast<std::vector<uint32_t>>(result);
        return HOLOLINK_MODULE_OK;
    }
    hololink_module_status_t and_uint32(uint32_t address, uint32_t mask) override
    {
        NB_OVERRIDE_PURE(and_uint32, address, mask);
    }
    hololink_module_status_t or_uint32(uint32_t address, uint32_t mask) override
    {
        NB_OVERRIDE_PURE(or_uint32, address, mask);
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
        NB_OVERRIDE_PURE(roce_data_channel_instance_id, md);
    }
    std::string i2c_instance_id(uint32_t bus, uint32_t address) override
    {
        NB_OVERRIDE_PURE(i2c_instance_id, bus, address);
    }
    std::string ptp_pps_output_instance_id() override
    {
        NB_OVERRIDE_PURE(ptp_pps_output_instance_id);
    }
    std::string null_vsync_instance_id() override
    {
        NB_OVERRIDE_PURE(null_vsync_instance_id);
    }
};

class PyEnumerationInterface : public ad::EnumerationInterfaceV1 {
public:
    NB_TRAMPOLINE(ad::EnumerationInterfaceV1, 1);

    hololink_module_status_t update_metadata(
        ad::EnumerationMetadata& metadata,
        const uint8_t* raw_packet, size_t raw_packet_len) override
    {
        // Manual override dispatch: raw_packet is a raw pointer + length
        // that has to become `bytes` or `None` for Python, which
        // NB_OVERRIDE_PURE's automatic forwarding can't do. Pure virtual,
        // so pure=true's own check covers the unoverridden case.
        nb::gil_scoped_acquire gil;
        constexpr uint64_t hash = nanobind::detail::str_hash("update_metadata");
        nanobind::detail::ticket ticket(nb_trampoline, "update_metadata", hash, /*pure=*/true);
        nb::object packet_arg;
        if (raw_packet == nullptr) {
            packet_arg = nb::none();
        } else {
            packet_arg = nb::bytes(raw_packet, raw_packet_len);
        }
        nb_trampoline.base().attr(ticket.key)(
            nb::cast(&metadata, nb::rv_policy::reference), packet_arg);
        return HOLOLINK_MODULE_OK;
    }
};

class PyLoggingInterface : public ad::LoggingInterfaceV1 {
public:
    NB_TRAMPOLINE(ad::LoggingInterfaceV1, 2);

    ad::LogLevel level() const override
    {
        NB_OVERRIDE_PURE(level);
    }
    void log(ad::LogLevel level_,
        const char* file, unsigned line, const char* function,
        const char* message) override
    {
        // TODO(nanobind-port): can't use NB_OVERRIDE_PURE_NAME literally
        // here. Its macro expansion deduces the Python-call return type via
        // `decltype(NBBase::log(level_, std::string(file), line,
        // std::string(function), std::string(message)))` -- i.e. it
        // type-checks the *marshaled* (std::string) arguments against the
        // real base method, whose parameters are `const char*`. std::string
        // has no implicit conversion to `const char*`, so that decltype
        // does not compile (confirmed by reading trampoline.h directly: the
        // non-_NAME, non-PURE-less macros all share this decltype-of-a-
        // real-call trick, unlike pybind11's PYBIND11_OVERRIDE_PURE_NAME,
        // which took an explicit ret_type instead of deducing one from a
        // real call to the base). Replicating the macro's own expansion
        // manually below, with an explicit `void` return instead of a
        // deduced one, sidesteps the issue while keeping the same
        // char*->std::string marshaling the original pybind11 version
        // performed for the Python call. log() is pure virtual, so
        // pure=true's own check covers the unoverridden case.
        nb::gil_scoped_acquire gil;
        constexpr uint64_t hash = nanobind::detail::str_hash("log");
        nanobind::detail::ticket ticket(nb_trampoline, "log", hash, /*pure=*/true);
        nb_trampoline.base().attr(ticket.key)(
            level_, std::string(file), line, std::string(function), std::string(message));
    }
};

// -------------------------------------------------------------------
// Per-interface bind helpers. Each function adds a single nb::class_
// to the module; the NB_MODULE entry-point calls them in order.
// -------------------------------------------------------------------

// Convert an EnumerationMetadata variant value to a Python object.
// std::vector<uint8_t> must round-trip as bytes — nanobind's STL caster
// would otherwise turn it into list[int], breaking symmetry with the
// __setitem__ bytes path below.
static nb::object enumeration_value_to_py(
    const ad::EnumerationMetadata::Value& v)
{
    return std::visit(
        [](auto&& x) -> nb::object {
            using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<T, std::vector<uint8_t>>) {
                return nb::bytes(
                    reinterpret_cast<const char*>(x.data()), x.size());
            } else {
                return nb::cast(x);
            }
        },
        v);
}

// The result of holoscan::python::export_constructor_handle(): a new reference to a capsule
// (or NotImplemented, for another type), or nullptr with a Python exception set.
static nb::object steal_constructor_handle(PyObject* exported)
{
    if (exported == nullptr) {
        throw nb::python_error();
    }
    return nb::steal(exported);
}

static void bind_enumeration_metadata(nb::module_& m)
{
    nb::class_<ad::EnumerationMetadata>(m, "EnumerationMetadata")
        .def(nb::init<>())
        // Passes a copy of the metadata to an Operator as the `metadata` constructor handle:
        // graph.op(hololink_module.operators.LinuxReceiverOp, "rx", metadata=channel_metadata,
        // ...). A fresh C++-owned copy, so the Operator never keeps this Python object alive and
        // later changes to it don't reach the Operator.
        .def(
            "__holoscan_constructor_handle__",
            [](const ad::EnumerationMetadata& self, const std::string& type_name) {
                return steal_constructor_handle(holoscan::python::export_constructor_handle(
                    type_name, hololink::module::operators::enumeration_metadata_handle_type(),
                    std::make_shared<ad::EnumerationMetadata>(self)));
            },
            nb::arg("type_name"),
            "Holoscan calls this to pass the metadata to an Operator as a constructor handle.")
        // Copy ctor: needed by stereo flows that fork one base
        // metadata into per-sensor clones before calling
        // Adapter.use_sensor(clone, n). Mirrors how the C++ stereo
        // example does `left.metadata = base_metadata;`.
        .def(nb::init<const ad::EnumerationMetadata&>(), nb::arg("other"))
        .def("__setitem__",
            [](ad::EnumerationMetadata& self,
                const std::string& key, nb::object value) {
                if (nb::isinstance<nb::int_>(value)) {
                    self[key] = nb::cast<int64_t>(value);
                } else if (nb::isinstance<nb::str>(value)) {
                    self[key] = nb::cast<std::string>(value);
                } else if (nb::isinstance<nb::bytes>(value)) {
                    nb::bytes b = nb::cast<nb::bytes>(value);
                    const uint8_t* data = reinterpret_cast<const uint8_t*>(b.c_str());
                    self[key] = std::vector<uint8_t>(data, data + b.size());
                } else {
                    throw nb::type_error("EnumerationMetadata values must be int, str, or bytes");
                }
            })
        .def("__getitem__",
            [](const ad::EnumerationMetadata& self,
                const std::string& key) -> nb::object {
                const auto it = self.find(key);
                if (it == self.cend()) {
                    throw nb::key_error(key.c_str());
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
                const std::string& key, nb::object default_value) -> nb::object {
                const auto it = self.find(key);
                if (it == self.cend()) {
                    return default_value;
                }
                return enumeration_value_to_py(it->second);
            },
            nb::arg("key"), nb::arg("default") = nb::none());
}

static void bind_module(nb::module_& m)
{
    nb::class_<ad::Module>(m, "Module");
}

// CsiConverterV1 trampoline so Python can implement the converter
// contract. The native in-tree implementation is the module
// CsiToBayerOp (hololink_module.operators.CsiToBayerOp), which
// subclasses this; the core publishes the abstract type so Python code
// can subclass it and the C++ sensor drivers (Vb1940Cam) accept it.
class PyCsiConverterV1 : public ad::csi::CsiConverterV1 {
public:
    NB_TRAMPOLINE(ad::csi::CsiConverterV1, 4);

    void configure(uint32_t start_byte, uint32_t received_bytes_per_line,
        uint32_t pixel_width, uint32_t pixel_height,
        ad::csi::PixelFormat pixel_format, uint32_t trailing_bytes) override
    {
        NB_OVERRIDE_PURE(configure,
            start_byte, received_bytes_per_line, pixel_width, pixel_height,
            pixel_format, trailing_bytes);
    }
    uint32_t receiver_start_byte() override
    {
        NB_OVERRIDE_PURE(receiver_start_byte);
    }
    uint32_t received_line_bytes(uint32_t line_bytes) override
    {
        NB_OVERRIDE_PURE(received_line_bytes, line_bytes);
    }
    uint32_t transmitted_line_bytes(
        ad::csi::PixelFormat pixel_format, uint32_t pixel_width) override
    {
        NB_OVERRIDE_PURE(transmitted_line_bytes, pixel_format, pixel_width);
    }
};

static void bind_csi_converter(nb::module_& m)
{
    // Note: the ad::csi::PixelFormat / BayerFormat enums are registered by
    // bind_vb1940_cam() below (this translation unit's counterpart of the
    // original's separate per-sensor vb1940 pybind module) -- nanobind keys
    // registered types by C++ std::type_info, so registering them here too
    // would fail with an "already registered" error. The methods below
    // reference PixelFormat only in their signatures; nanobind resolves the
    // type caster at call time, by which point bind_vb1940_cam() has
    // already run (see the NB_MODULE body's call order at the bottom of
    // this file) -- so the registration-order constraint that made the
    // original cross-module py::module_::import(...) dance necessary no
    // longer applies here at all (everything is one translation unit now).
    nb::class_<ad::csi::CsiConverterV1, PyCsiConverterV1>(m, "CsiConverterV1")
        .def(nb::init<>())
        .def("configure", &ad::csi::CsiConverterV1::configure,
            nb::arg("start_byte"), nb::arg("received_bytes_per_line"),
            nb::arg("pixel_width"), nb::arg("pixel_height"),
            nb::arg("pixel_format"), nb::arg("trailing_bytes") = 0)
        .def("receiver_start_byte", &ad::csi::CsiConverterV1::receiver_start_byte)
        .def("received_line_bytes", &ad::csi::CsiConverterV1::received_line_bytes,
            nb::arg("line_bytes"))
        .def("transmitted_line_bytes", &ad::csi::CsiConverterV1::transmitted_line_bytes,
            nb::arg("pixel_format"), nb::arg("pixel_width"));
    nb::object cls = m.attr("CsiConverterV1");
    cls.attr("release_version") = ad::csi::CsiConverterV1::release_version;
}

static void bind_frame_metadata(nb::module_& m)
{
    nb::class_<ad::FrameMetadataInterfaceV1::FrameMetadata>(m, "FrameMetadata")
        .def_ro("flags", &ad::FrameMetadataInterfaceV1::FrameMetadata::flags)
        .def_ro("psn", &ad::FrameMetadataInterfaceV1::FrameMetadata::psn)
        .def_ro("crc", &ad::FrameMetadataInterfaceV1::FrameMetadata::crc)
        .def_ro("frame_number",
            &ad::FrameMetadataInterfaceV1::FrameMetadata::frame_number)
        .def_ro("timestamp_s",
            &ad::FrameMetadataInterfaceV1::FrameMetadata::timestamp_s)
        .def_ro("timestamp_ns",
            &ad::FrameMetadataInterfaceV1::FrameMetadata::timestamp_ns)
        .def_ro("bytes_written",
            &ad::FrameMetadataInterfaceV1::FrameMetadata::bytes_written)
        .def_ro("metadata_s",
            &ad::FrameMetadataInterfaceV1::FrameMetadata::metadata_s)
        .def_ro("metadata_ns",
            &ad::FrameMetadataInterfaceV1::FrameMetadata::metadata_ns);

    nb::class_<ad::FrameMetadataInterfaceV1>(m, "FrameMetadataInterfaceV1")
        .def_static(
            "get_service",
            [](std::shared_ptr<ad::Module> module, bool allow_null) {
                return ad::FrameMetadataInterfaceV1::get_service(
                    std::move(module), allow_null);
            },
            nb::arg("module"), nb::arg("allow_null") = false)
        .def("block_size", &ad::FrameMetadataInterfaceV1::block_size)
        .def(
            "decode",
            [](const ad::FrameMetadataInterfaceV1& self, nb::object host_memory) {
                PyBufferRequest info = require_1d_contiguous_byte_buffer(
                    host_memory,
                    "invoking FrameMetadataInterfaceV1.decode host_memory");
                ad::FrameMetadataInterfaceV1::FrameMetadata out {};
                const hololink_module_status_t s = self.decode(
                    info.view.buf,
                    static_cast<size_t>(info.view.len),
                    out);
                if (s != HOLOLINK_MODULE_OK) {
                    throw std::runtime_error(
                        "While decoding frame metadata: status "
                        + std::to_string(s));
                }
                return out;
            },
            nb::arg("host_memory"));
    nb::object cls = m.attr("FrameMetadataInterfaceV1");
    cls.attr("type_id") = ad::FrameMetadataInterfaceV1::type_id;
    cls.attr("release_version") = ad::FrameMetadataInterfaceV1::release_version;
}

static void bind_i2c_lock(nb::module_& m)
{
    nb::class_<ad::I2cLockV1, PyI2cLock>(m, "I2cLockV1")
        .def(nb::init<>())
        .def("lock", &ad::I2cLockV1::lock)
        .def("unlock", &ad::I2cLockV1::unlock)
        .def("try_lock", &ad::I2cLockV1::try_lock);
    nb::object cls = m.attr("I2cLockV1");
    cls.attr("release_version") = ad::I2cLockV1::release_version;
}

static void bind_i2c_interface(nb::module_& m)
{
    nb::class_<ad::I2cInterfaceV1, PyI2cInterface>(m, "I2cInterfaceV1")
        .def(nb::init<>())
        .def_static(
            "get_service",
            [](std::shared_ptr<ad::Module> module, const std::string& instance_id,
                bool allow_null) {
                return ad::I2cInterfaceV1::get_service(
                    std::move(module), instance_id.c_str(), allow_null);
            },
            nb::arg("module"), nb::arg("instance_id"),
            nb::arg("allow_null") = false)
        .def(
            "i2c_transaction",
            [](ad::I2cInterfaceV1& self, uint32_t peripheral_address,
                nb::object write_bytes, size_t read_byte_count) {
                PyBufferRequest info = request_py_buffer(write_bytes);
                // info.view.len is the buffer's total byte count (shape
                // product * itemsize) regardless of itemsize/ndim -- same
                // value pybind11's `info.size * info.itemsize` computed.
                std::vector<uint8_t> w(static_cast<size_t>(info.view.len));
                std::memcpy(w.data(), info.view.buf, w.size());
                std::vector<uint8_t> r(read_byte_count);
                const hololink_module_status_t s = self.i2c_transaction(
                    peripheral_address, w, r);
                if (s != HOLOLINK_MODULE_OK) {
                    throw std::runtime_error(
                        "While invoking I2cInterfaceV1.i2c_transaction: status "
                        + std::to_string(s));
                }
                return nb::bytes(reinterpret_cast<const char*>(r.data()), r.size());
            },
            nb::arg("peripheral_address"), nb::arg("write_bytes"),
            nb::arg("read_byte_count"))
        .def(
            "encode_i2c_request",
            [](ad::I2cInterfaceV1& self, ad::SequencerInterfaceV1& sequencer,
                uint32_t peripheral_i2c_address, nb::object write_bytes,
                uint32_t read_byte_count) {
                PyBufferRequest info = request_py_buffer(write_bytes);
                std::vector<uint8_t> w(static_cast<size_t>(info.view.len));
                std::memcpy(w.data(), info.view.buf, w.size());
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
                return nb::make_tuple(wi, ri, status_index);
            },
            nb::arg("sequencer"), nb::arg("peripheral_i2c_address"),
            nb::arg("write_bytes"), nb::arg("read_byte_count"));
    nb::object cls = m.attr("I2cInterfaceV1");
    cls.attr("type_id") = ad::I2cInterfaceV1::type_id;
    cls.attr("release_version") = ad::I2cInterfaceV1::release_version;
}

static void bind_sequencer(nb::module_& m)
{
    nb::class_<ad::SequencerInterfaceV1, PySequencer>(
        m, "SequencerInterfaceV1")
        .def(nb::init<>())
        .def_static(
            "get_service",
            [](std::shared_ptr<ad::Module> module, const std::string& instance_id,
                bool allow_null) {
                return ad::SequencerInterfaceV1::get_service(
                    std::move(module), instance_id.c_str(), allow_null);
            },
            nb::arg("module"), nb::arg("instance_id"),
            nb::arg("allow_null") = false)
        .def("write_uint32", &ad::SequencerInterfaceV1::write_uint32,
            nb::arg("address"), nb::arg("data"))
        .def("read_uint32", &ad::SequencerInterfaceV1::read_uint32,
            nb::arg("address"), nb::arg("initial_value") = 0xFFFFFFFFu)
        .def("poll", &ad::SequencerInterfaceV1::poll,
            nb::arg("address"), nb::arg("mask"), nb::arg("match"))
        .def("enable", &ad::SequencerInterfaceV1::enable)
        .def("location", &ad::SequencerInterfaceV1::location);
    nb::object cls = m.attr("SequencerInterfaceV1");
    cls.attr("type_id") = ad::SequencerInterfaceV1::type_id;
    cls.attr("release_version") = ad::SequencerInterfaceV1::release_version;
}

// Opaque RoceReceiverInterfaceV1 binding. No methods are exposed —
// applications drive the receiver indirectly by constructing
// RoceReceiverOp, which fetches the receiver via
// RoceReceiverV1::get_service and forwards into receiver->start(...)
// internally. The class_ is registered so nanobind can resolve
// shared_ptr<RoceReceiverV1> at the Python boundary (e.g. for
// RoceDataChannelInterfaceV1.attach_receiver(receiver) when application
// code wants to hand a handle through).
static void bind_oscillator(nb::module_& m)
{
    nb::class_<ad::OscillatorInterfaceV1>(m, "OscillatorInterfaceV1")
        .def_static(
            "get_service",
            [](std::shared_ptr<ad::Module> module, const std::string& instance_id,
                bool allow_null) {
                return ad::OscillatorInterfaceV1::get_service(
                    std::move(module), instance_id.c_str(), allow_null);
            },
            nb::arg("module"), nb::arg("instance_id"),
            nb::arg("allow_null") = false)
        .def_static(
            "get_service",
            [](const ad::EnumerationMetadata& metadata, bool allow_null) {
                return ad::OscillatorInterfaceV1::get_service(metadata, allow_null);
            },
            nb::arg("metadata"), nb::arg("allow_null") = false)
        .def("enable", &ad::OscillatorInterfaceV1::enable,
            nb::arg("clocks_per_second"))
        .def("get_caps", &ad::OscillatorInterfaceV1::get_caps)
        .def("set_caps", &ad::OscillatorInterfaceV1::set_caps, nb::arg("caps"));
    nb::object cls = m.attr("OscillatorInterfaceV1");
    cls.attr("type_id") = ad::OscillatorInterfaceV1::type_id;
    cls.attr("release_version") = ad::OscillatorInterfaceV1::release_version;
}

static void bind_roce_receiver(nb::module_& m)
{
    nb::class_<ad::RoceReceiverInterfaceV1>(
        m, "RoceReceiverInterfaceV1")
        .def_static(
            "get_service",
            [](std::shared_ptr<ad::Module> module, const std::string& instance_id,
                bool allow_null) {
                return ad::RoceReceiverInterfaceV1::get_service(
                    std::move(module), instance_id.c_str(), allow_null);
            },
            nb::arg("module"), nb::arg("instance_id"),
            nb::arg("allow_null") = false)
        .def_static(
            "get_service",
            [](const ad::EnumerationMetadata& metadata, bool allow_null) {
                return ad::RoceReceiverInterfaceV1::get_service(metadata, allow_null);
            },
            nb::arg("metadata"), nb::arg("allow_null") = false);
    nb::object cls = m.attr("RoceReceiverInterfaceV1");
    cls.attr("type_id") = ad::RoceReceiverInterfaceV1::type_id;
    cls.attr("release_version") = ad::RoceReceiverInterfaceV1::release_version;
}

static void bind_data_channel(nb::module_& m)
{
    // Per-channel anchor base. Owns the cache slot; Python sees the
    // anchor surface (enumeration_metadata, hololink) plus the
    // get_service entry points that route through it.
    nb::class_<ad::DataChannelInterfaceV1, PyDataChannel>(m, "DataChannelInterfaceV1")
        .def(nb::init<>())
        .def_static(
            "get_service",
            [](std::shared_ptr<ad::Module> module, const std::string& instance_id,
                bool allow_null) {
                return ad::DataChannelInterfaceV1::get_service(
                    std::move(module), instance_id.c_str(), allow_null);
            },
            nb::arg("module"), nb::arg("instance_id"),
            nb::arg("allow_null") = false)
        .def_static(
            "get_service",
            [](const ad::EnumerationMetadata& metadata, bool allow_null) {
                return ad::DataChannelInterfaceV1::get_service(metadata, allow_null);
            },
            nb::arg("metadata"), nb::arg("allow_null") = false)
        .def("enumeration_metadata",
            &ad::DataChannelInterfaceV1::enumeration_metadata,
            nb::rv_policy::reference_internal)
        .def("hololink", &ad::DataChannelInterfaceV1::hololink)
        .def("device_lost", &ad::DataChannelInterfaceV1::device_lost,
            "Channel-level entry point for HololinkInterfaceV1.device_lost, to "
            "invalidate a lost device through a data channel already held.");
    nb::object cls = m.attr("DataChannelInterfaceV1");
    cls.attr("type_id") = ad::DataChannelInterfaceV1::type_id;
    cls.attr("release_version") = ad::DataChannelInterfaceV1::release_version;
}

static void bind_roce_data_channel(nb::module_& m)
{
    // RoCE transport — its own ConfigurableService, not derived from
    // DataChannelInterfaceV1. Composes with the per-channel anchor
    // internally; Python callers fetch the anchor separately via
    // DataChannelInterfaceV1.get_service when they need its metadata
    // or hololink.
    nb::class_<ad::RoceDataChannelInterfaceV1, PyRoceDataChannel>(
        m, "RoceDataChannelInterfaceV1")
        .def(nb::init<>())
        .def_static(
            "get_service",
            [](std::shared_ptr<ad::Module> module, const std::string& instance_id,
                bool allow_null) {
                return ad::RoceDataChannelInterfaceV1::get_service(
                    std::move(module), instance_id.c_str(), allow_null);
            },
            nb::arg("module"), nb::arg("instance_id"),
            nb::arg("allow_null") = false)
        .def_static(
            "get_service",
            [](const ad::EnumerationMetadata& metadata, bool allow_null) {
                return ad::RoceDataChannelInterfaceV1::get_service(metadata, allow_null);
            },
            nb::arg("metadata"), nb::arg("allow_null") = false)
        .def("attach_receiver", &ad::RoceDataChannelInterfaceV1::attach_receiver,
            nb::arg("receiver"))
        .def("detach_receiver", &ad::RoceDataChannelInterfaceV1::detach_receiver)
        .def(
            "frame_end_sequencer",
            [](ad::RoceDataChannelInterfaceV1& self, bool allow_null) {
                return self.frame_end_sequencer<>(allow_null);
            },
            nb::arg("allow_null") = false)
        .def(
            "get_hololink",
            [](ad::RoceDataChannelInterfaceV1& self, bool allow_null) {
                return self.get_hololink<>(allow_null);
            },
            nb::arg("allow_null") = false);
    nb::object cls = m.attr("RoceDataChannelInterfaceV1");
    cls.attr("type_id") = ad::RoceDataChannelInterfaceV1::type_id;
    cls.attr("release_version") = ad::RoceDataChannelInterfaceV1::release_version;
}

// Vsync — abstract consumer-facing base. Bound as a polymorphic
// nb::class_ so derived types (PtpPpsOutput) upcast cleanly when
// handed to Python sites that consume a Vsync (e.g. the module
// Vb1940Cam's `vsync=` keyword, bound by bind_vb1940_cam() below).
static void bind_vsync(nb::module_& m)
{
    nb::class_<ad::VsyncInterfaceV1>(
        m, "VsyncInterfaceV1")
        .def_static(
            "get_service",
            [](std::shared_ptr<ad::Module> module, const std::string& instance_id,
                bool allow_null) {
                return ad::VsyncInterfaceV1::get_service(
                    std::move(module), instance_id.c_str(), allow_null);
            },
            nb::arg("module"), nb::arg("instance_id"),
            nb::arg("allow_null") = false)
        .def("is_enabled", &ad::VsyncInterfaceV1::is_enabled)
        .def("start", &ad::VsyncInterfaceV1::start)
        .def("stop", &ad::VsyncInterfaceV1::stop);
    nb::object cls = m.attr("VsyncInterfaceV1");
    cls.attr("type_id") = ad::VsyncInterfaceV1::type_id;
    cls.attr("release_version") = ad::VsyncInterfaceV1::release_version;
}

// PtpPpsOutput — the PTP-PPS-driven concrete Vsync source. The
// nb::class_ declares Vsync as a base so a Python shared_ptr<PtpPpsOutput>
// flows through any annotation expecting Vsync.
static void bind_ptp_pps_output(nb::module_& m)
{
    nb::class_<ad::PtpPpsOutputInterfaceV1, ad::VsyncInterfaceV1>(
        m, "PtpPpsOutputInterfaceV1")
        .def_static(
            "get_service",
            [](std::shared_ptr<ad::Module> module, const std::string& instance_id,
                bool allow_null) {
                return ad::PtpPpsOutputInterfaceV1::get_service(
                    std::move(module), instance_id.c_str(), allow_null);
            },
            nb::arg("module"), nb::arg("instance_id"),
            nb::arg("allow_null") = false)
        .def_static(
            "get_service",
            [](const ad::EnumerationMetadata& metadata, bool allow_null) {
                return ad::PtpPpsOutputInterfaceV1::get_service(metadata, allow_null);
            },
            nb::arg("metadata"), nb::arg("allow_null") = false)
        .def("enable", &ad::PtpPpsOutputInterfaceV1::enable, nb::arg("frequency_hz"))
        .def("disable", &ad::PtpPpsOutputInterfaceV1::disable);
    nb::object cls = m.attr("PtpPpsOutputInterfaceV1");
    cls.attr("type_id") = ad::PtpPpsOutputInterfaceV1::type_id;
    cls.attr("release_version") = ad::PtpPpsOutputInterfaceV1::release_version;
}

static void bind_hololink_interface(nb::module_& m)
{
    // Opaque RAII handle returned by on_reset(); holding it keeps the
    // callback registered, dropping it (e.g. when the owning camera is
    // garbage-collected) unregisters the callback. No methods are exposed.
    nb::class_<ad::HololinkInterfaceV1::ResetRegistration>(
        m, "ResetRegistration");

    nb::class_<ad::HololinkInterfaceV1, PyHololinkInterface>(m, "HololinkInterfaceV1")
        .def(nb::init<>())
        .def_static(
            "get_service",
            [](std::shared_ptr<ad::Module> module, const std::string& instance_id,
                bool allow_null) {
                return ad::HololinkInterfaceV1::get_service(
                    std::move(module), instance_id.c_str(), allow_null);
            },
            nb::arg("module"), nb::arg("instance_id"),
            nb::arg("allow_null") = false)
        .def_static(
            "get_service",
            [](const ad::EnumerationMetadata& metadata, bool allow_null) {
                return ad::HololinkInterfaceV1::get_service(metadata, allow_null);
            },
            nb::arg("metadata"), nb::arg("allow_null") = false)
        .def("enumeration_metadata",
            &ad::HololinkInterfaceV1::enumeration_metadata,
            nb::rv_policy::reference_internal)
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
            nb::call_guard<nb::gil_scoped_release>(),
            "Block until the board's clock is disciplined to the PTP "
            "grandmaster; returns True on success, False on timeout.")
        .def("on_reset", &ad::HololinkInterfaceV1::on_reset, nb::arg("callback"),
            "Register a callback fired each time the board completes a "
            "reset (after the FPGA reboots). Multiple callbacks may be "
            "registered; each fires once per reset(). Returns a handle: "
            "keep it alive while the callback should fire, drop it (e.g. "
            "let the owning object be garbage-collected) to unregister.")
        .def("write_uint32", &ad::HololinkInterfaceV1::write_uint32,
            nb::arg("addresses"), nb::arg("values"))
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
            nb::arg("addresses"))
        .def("and_uint32", &ad::HololinkInterfaceV1::and_uint32,
            nb::arg("address"), nb::arg("mask"))
        .def("or_uint32", &ad::HololinkInterfaceV1::or_uint32,
            nb::arg("address"), nb::arg("mask"))
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
            nb::rv_policy::take_ownership)
        .def(
            "get_roce_data_channel",
            [](ad::HololinkInterfaceV1& self,
                const ad::EnumerationMetadata& md, bool allow_null) {
                return self.get_roce_data_channel<>(md, allow_null);
            },
            nb::arg("metadata"), nb::arg("allow_null") = false)
        .def(
            "get_i2c",
            [](ad::HololinkInterfaceV1& self, uint32_t bus, uint32_t address,
                bool allow_null) {
                return self.get_i2c<>(bus, address, allow_null);
            },
            nb::arg("bus"), nb::arg("address"), nb::arg("allow_null") = false)
        .def(
            "software_sequencer",
            [](ad::HololinkInterfaceV1& self, bool allow_null) {
                return self.software_sequencer<>(allow_null);
            },
            nb::arg("allow_null") = false)
        .def(
            "gpio0_sequencer",
            [](ad::HololinkInterfaceV1& self, bool allow_null) {
                return self.gpio0_sequencer<>(allow_null);
            },
            nb::arg("allow_null") = false)
        .def(
            "gpio1_sequencer",
            [](ad::HololinkInterfaceV1& self, bool allow_null) {
                return self.gpio1_sequencer<>(allow_null);
            },
            nb::arg("allow_null") = false)
        .def(
            "sif0_frame_start_sequencer",
            [](ad::HololinkInterfaceV1& self, bool allow_null) {
                return self.sif0_frame_start_sequencer<>(allow_null);
            },
            nb::arg("allow_null") = false)
        .def(
            "sif1_frame_start_sequencer",
            [](ad::HololinkInterfaceV1& self, bool allow_null) {
                return self.sif1_frame_start_sequencer<>(allow_null);
            },
            nb::arg("allow_null") = false)
        .def(
            "ptp_pps_output",
            [](ad::HololinkInterfaceV1& self, bool allow_null) {
                return self.ptp_pps_output<>(allow_null);
            },
            nb::arg("allow_null") = false)
        .def(
            "null_vsync",
            [](ad::HololinkInterfaceV1& self, bool allow_null) {
                return self.null_vsync<>(allow_null);
            },
            nb::arg("allow_null") = false);
    nb::object cls = m.attr("HololinkInterfaceV1");
    cls.attr("type_id") = ad::HololinkInterfaceV1::type_id;
    cls.attr("release_version") = ad::HololinkInterfaceV1::release_version;
}

static void bind_enumeration_interface(nb::module_& m)
{
    nb::class_<ad::EnumerationInterfaceV1, PyEnumerationInterface>(m, "EnumerationInterfaceV1")
        .def(nb::init<>())
        .def_static(
            "get_service",
            [](std::shared_ptr<ad::Module> module, bool allow_null) {
                return ad::EnumerationInterfaceV1::get_service(
                    std::move(module), allow_null);
            },
            nb::arg("module"), nb::arg("allow_null") = false)
        .def(
            "update_metadata",
            [](ad::EnumerationInterfaceV1& self,
                ad::EnumerationMetadata& metadata, nb::object raw_packet) {
                hololink_module_status_t st;
                if (raw_packet.is_none()) {
                    st = self.update_metadata(metadata, nullptr, 0);
                } else {
                    // Validated PyBufferRequest stays alive (keeping the
                    // Py_buffer pinned) until self.update_metadata
                    // returns, so the underlying bytes can't be
                    // mutated or freed mid-call.
                    PyBufferRequest info = require_1d_contiguous_byte_buffer(
                        raw_packet,
                        "invoking EnumerationInterfaceV1.update_metadata raw_packet");
                    st = self.update_metadata(
                        metadata,
                        static_cast<const uint8_t*>(info.view.buf),
                        static_cast<size_t>(info.view.len));
                }
                if (st != HOLOLINK_MODULE_OK) {
                    throw std::runtime_error(
                        "While invoking EnumerationInterfaceV1.update_metadata: "
                        "status "
                        + std::to_string(st));
                }
            },
            nb::arg("metadata"), nb::arg("raw_packet") = nb::none());
    nb::object cls = m.attr("EnumerationInterfaceV1");
    cls.attr("type_id") = ad::EnumerationInterfaceV1::type_id;
    cls.attr("release_version") = ad::EnumerationInterfaceV1::release_version;
}

static void bind_logging_interface(nb::module_& m)
{
    // nb::is_arithmetic() makes this a real IntEnum member (int(value) works directly,
    // matching the legacy pybind11 binding's implicit behavior) -- see
    // python/hololink/_hololink.cpp's own PixelFormat/BayerFormat comment. Without it,
    // LoggingInterfaceV1 overrides that do int(level) (the common case -- see
    // PyLoggingInterface::log's own ticket-based override above) raise TypeError.
    nb::enum_<ad::LogLevel>(m, "LogLevel", nb::is_arithmetic())
        .value("Trace", ad::LogLevel::Trace)
        .value("Debug", ad::LogLevel::Debug)
        .value("Info", ad::LogLevel::Info)
        .value("Warning", ad::LogLevel::Warning)
        .value("Error", ad::LogLevel::Error);

    nb::class_<ad::LoggingInterfaceV1, PyLoggingInterface>(m, "LoggingInterfaceV1")
        .def(nb::init<>())
        .def_static(
            "get_service",
            [](std::shared_ptr<ad::Module> module, bool allow_null) {
                return ad::LoggingInterfaceV1::get_service(
                    std::move(module), allow_null);
            },
            nb::arg("module"), nb::arg("allow_null") = false)
        .def("level", &ad::LoggingInterfaceV1::level)
        .def(
            "log",
            [](ad::LoggingInterfaceV1& self, ad::LogLevel level_,
                const std::string& file, unsigned line,
                const std::string& function, const std::string& message) {
                self.log(level_, file.c_str(), line, function.c_str(),
                    message.c_str());
            },
            nb::arg("level"), nb::arg("file"), nb::arg("line"),
            nb::arg("function"), nb::arg("message"));
    nb::object cls = m.attr("LoggingInterfaceV1");
    cls.attr("type_id") = ad::LoggingInterfaceV1::type_id;
    cls.attr("release_version") = ad::LoggingInterfaceV1::release_version;
}

// Wrap a Python callable in a Reactor::Callback whose held nb::object
// is destroyed under the GIL. The core::Reactor dispatch thread (see
// reactor_impl.cpp:45) does not hold the GIL when it releases the
// shared_ptr<Callback>, so capturing a bare nb::object in the lambda
// would Py_DECREF without the GIL — undefined behavior.
static std::shared_ptr<ad::ReactorV1::Callback> make_python_reactor_callback(
    nb::object callback)
{
    std::shared_ptr<nb::object> holder(
        new nb::object(std::move(callback)),
        [](nb::object* p) {
            nb::gil_scoped_acquire gil;
            delete p;
        });
    return std::make_shared<ad::ReactorV1::Callback>([holder]() {
        nb::gil_scoped_acquire gil;
        (*holder)();
    });
}

static void bind_reactor(nb::module_& m)
{
    nb::class_<ad::ReactorV1::AlarmEntry>(m, "AlarmEntry");

    // Reactor is host-published (no Python subclassing); the binding
    // exposes the consumer surface only.
    nb::class_<ad::ReactorV1>(m, "ReactorV1")
        .def_static(
            "get_service",
            [](std::shared_ptr<ad::Module> module, bool allow_null) {
                return ad::ReactorV1::get_service(std::move(module), allow_null);
            },
            nb::arg("module"), nb::arg("allow_null") = false)
        .def(
            "now",
            [](const ad::ReactorV1& self) {
                struct timespec ts = self.now();
                return nb::make_tuple(static_cast<int64_t>(ts.tv_sec),
                    static_cast<int64_t>(ts.tv_nsec));
            })
        .def(
            "add_callback",
            [](ad::ReactorV1& self, nb::object callback) {
                self.add_callback(
                    make_python_reactor_callback(std::move(callback)));
            },
            nb::arg("callback"))
        .def(
            "add_alarm_s",
            [](ad::ReactorV1& self, float seconds, nb::object callback) {
                return self.add_alarm_s(seconds,
                    make_python_reactor_callback(std::move(callback)));
            },
            nb::arg("seconds"), nb::arg("callback"))
        .def("cancel_alarm", &ad::ReactorV1::cancel_alarm, nb::arg("handle"))
        .def("is_current_thread", &ad::ReactorV1::is_current_thread);
    nb::object cls = m.attr("ReactorV1");
    cls.attr("type_id") = ad::ReactorV1::type_id;
    cls.attr("release_version") = ad::ReactorV1::release_version;
}

// Opaque handle for register_ip / register_all. Python code holds the returned object, passes it to
// unregister(), and doesn't inspect anything inside. The handle is wrapped in a distinct type because
// adapter.hpp keeps EnumerationCallback incomplete, and nanobind's shared_ptr caster needs a complete,
// bound pointee.
struct EnumerationRegistration {
    ad::EnumerationCallbackHandle handle;
};

static void bind_adapter(nb::module_& m)
{
    nb::class_<EnumerationRegistration>(m, "EnumerationCallbackHandle");

    // Adapter is a process-wide singleton never destructed from Python;
    // nb::never_destruct() is nanobind's direct equivalent of the original
    // pybind11 binding's std::unique_ptr<Adapter, py::nodelete> holder (see
    // docs/porting.rst's "py::nodelete was used with holders..." note, and
    // classes.rst's "Preventing object destruction" section, whose own
    // Singleton::get_instance() example is this exact pattern). Per that
    // section's warning, get_adapter() below must (and does) return via
    // nb::rv_policy::reference, not the default.
    nb::class_<ad::Adapter>(m, "Adapter", nb::never_destruct())
        .def_static(
            "get_adapter",
            []() -> ad::Adapter& { return ad::Adapter::get_adapter(); },
            nb::rv_policy::reference)
        .def(
            "set_module_directory",
            [](ad::Adapter& self, const std::string& dir) {
                self.set_module_directory(std::filesystem::path(dir));
            },
            nb::arg("dir"))
        .def(
            "load_module",
            [](ad::Adapter& self, const std::string& so_path) {
                return self.load_module(std::filesystem::path(so_path));
            },
            nb::arg("so_path"))
        .def(
            "enumerate",
            [](ad::Adapter& self, ad::EnumerationMetadata metadata) {
                self.enumerate(std::move(metadata));
            },
            nb::arg("metadata"))
        .def("start_bootp_listener", &ad::Adapter::start_bootp_listener,
            nb::arg("port") = uint32_t { 12267 })
        .def("stop_bootp_listener", &ad::Adapter::stop_bootp_listener)
        .def(
            "wait_for_channel",
            [](ad::Adapter& self, const std::string& peer_ip, double timeout_s) {
                const auto timeout
                    = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::duration<double>(timeout_s));
                return self.wait_for_channel(peer_ip, timeout);
            },
            nb::arg("peer_ip"), nb::arg("timeout_s"),
            // Release the GIL during the blocking wait so another
            // Python thread can call adapter.enumerate(...) — the
            // very thing this call is waiting for. Without this the
            // wait deadlocks on its own GIL hold and times out every
            // time the announcement is fired from a Python thread.
            nb::call_guard<nb::gil_scoped_release>(),
            "Block up to `timeout_s` seconds for an enumeration announcement "
            "for `peer_ip` that arrives after this call started waiting, and "
            "return its metadata; raises RuntimeError on timeout. Built on "
            "register_ip — earlier announcements don't count.")
        .def(
            "register_ip",
            [](ad::Adapter& self, const std::string& peer_ip,
                std::function<void(const ad::EnumerationMetadata&)> callback) {
                return EnumerationRegistration { self.register_ip(peer_ip, std::move(callback)) };
            },
            nb::arg("peer_ip"), nb::arg("callback"),
            "Register a callback that fires for every enumeration "
            "announcement matching `peer_ip`. Returns an opaque handle "
            "to pass to unregister().")
        .def(
            "register_all",
            [](ad::Adapter& self,
                std::function<void(const ad::EnumerationMetadata&)> callback) {
                return EnumerationRegistration { self.register_all(std::move(callback)) };
            },
            nb::arg("callback"),
            "Register a callback that fires for every enumeration "
            "announcement, regardless of peer_ip. Returns an opaque "
            "handle to pass to unregister().")
        .def(
            "unregister",
            [](ad::Adapter& self, const EnumerationRegistration& registration) {
                self.unregister(registration.handle);
            },
            nb::arg("handle"))
        .def(
            "use_sensor",
            [](ad::Adapter& self, ad::EnumerationMetadata& metadata,
                int64_t sensor_number) {
                self.use_sensor(metadata, sensor_number);
            },
            nb::arg("metadata"), nb::arg("sensor_number"),
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
            nb::arg("metadata"), nb::arg("mtu"),
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
            nb::arg("metadata"), nb::arg("address"), nb::arg("port"),
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
            nb::arg("metadata"))
        .def(
            "reactor",
            [](ad::Adapter& self) {
                return ad::ReactorV1::get_service(
                    self.host_publisher()->self_module());
            },
            "The host ReactorV1 singleton (published on the host module, not a "
            "device module).");
}

// -------------------------------------------------------------------
// Operators-file surface (hololink_module/host/operators/python/operators_py.cpp):
// SensorDevice / SensorFactory / NetworkReceiverFactory only. The 9
// Operator<Params>-shaped operators that file's pybind11 trampolines used to
// bind (RoceReceiverOp, LinuxReceiverOp, FusaCoeCaptureOp, CsiToBayerOp,
// ImageProcessorOp, PackedFormatConverterOp, HsbControllerOp, ComputeCrcOp,
// CheckCrcOp) are NOT translated here: they were ported to Holoscan SDK
// 5.0's Operator<Params> shape (typed tensor/metadata ports,
// Contract/NotificationSource, .set_params() configuration), which has no
// holoscan::ArgList / holoscan::Fragment* construction path for pybind11
// (or nanobind) trampolines to forward through. Re-adding Python bindings
// for the ported shape -- through HSDK5's holoscan::authoring_registry(),
// a different mechanism that needs no nb::class_ binding for the operators
// themselves -- is a separate, follow-up task, not part of this port.
// -------------------------------------------------------------------

namespace hololink::module::operators {

/* Trampoline for the sensor wrapper. The application subclasses SensorDevice
 * in Python (e.g. Imx274SensorDevice), overriding stop_sensor(). */
class PySensorDevice : public SensorDevice {
public:
    NB_TRAMPOLINE(SensorDevice, 1);

    hololink_module_status_t stop_sensor() override
    {
        NB_OVERRIDE_PURE(stop_sensor);
    }
};

/* Trampoline for the sensor-side controller. The application subclasses
 * SensorFactory in Python (e.g. Imx274SensorFactory), overriding new_sensor to
 * build + arm its camera and return a SensorDevice, and optionally
 * fallback_frame to supply the test image shown at startup / during an outage.
 * The base class owns the watchdog and reconnection policy; those methods run
 * their C++ implementations unless a subclass overrides them. */
class PySensorFactory : public SensorFactory {
public:
    NB_TRAMPOLINE(SensorFactory, 2);

    void fallback_frame(CUdeviceptr& out_ptr, size_t& out_size) override
    {
        // Pythonic out-params: the override returns a (ptr, size) tuple, or
        // None / nothing for no fallback (the base default).
        //
        // Manual override dispatch (not NB_OVERRIDE_PURE): fallback_frame is
        // a non-pure virtual with a default ({0, 0}) C++ implementation, and
        // the Python-facing (ptr, size)-tuple-or-None return shape needs
        // marshaling NB_OVERRIDE's automatic casting can't do. pure=false
        // here, so ticket.key is left invalid (rather than raising) when
        // nothing overrides the method -- matching the original pybind11
        // version's own manual "if (!override) return" fallback-to-default
        // behavior (out_ptr/out_size are already defaulted above).
        out_ptr = 0;
        out_size = 0;
        nb::gil_scoped_acquire gil;
        constexpr uint64_t hash = nanobind::detail::str_hash("fallback_frame");
        nanobind::detail::ticket ticket(nb_trampoline, "fallback_frame", hash, /*pure=*/false);
        if (!ticket.key.is_valid()) {
            return;
        }
        nb::object result = nb_trampoline.base().attr(ticket.key)();
        if (result.is_none()) {
            return;
        }
        auto frame = nb::cast<std::pair<uint64_t, size_t>>(result);
        out_ptr = static_cast<CUdeviceptr>(frame.first);
        out_size = frame.second;
    }

    std::shared_ptr<SensorDevice> new_sensor(
        const hololink::module::EnumerationMetadata& metadata) override
    {
        // Manual override dispatch, with pure=false even though new_sensor
        // is pure virtual in sensor_factory.hpp: the original pybind11
        // version deliberately raises its own worded error ("SensorFactory.
        // new_sensor is not overridden") instead of pybind11's generic
        // pure-virtual message, and wraps the call in a try/catch that
        // reports (rather than propagates) a Python exception raised during
        // (re)connect -- see the catch block below. Using pure=true here
        // would replace that custom message with nanobind's generic "tried
        // to call a pure virtual function" wording; pure=false preserves it
        // by letting this code do its own "not overridden" check instead.
        nb::gil_scoped_acquire gil;
        constexpr uint64_t hash = nanobind::detail::str_hash("new_sensor");
        nanobind::detail::ticket ticket(nb_trampoline, "new_sensor", hash, /*pure=*/false);
        if (!ticket.key.is_valid()) {
            throw std::runtime_error("SensorFactory.new_sensor is not overridden");
        }
        try {
            return nb::cast<std::shared_ptr<SensorDevice>>(
                nb_trampoline.base().attr(ticket.key)(metadata));
        } catch (nb::python_error& e) {
            // A (re)connect attempt raised. Report it and return null so the
            // factory stays disconnected and retries on the next announcement,
            // rather than letting the exception abort the reactor thread
            // new_sensor runs on.
            e.discard_as_unraisable("SensorFactory.new_sensor (retry on re-announce)");
            return nullptr;
        }
    }
};

} // namespace hololink::module::operators

// CsiGeometry -> the dict CsiToBayerOp's `geometry` parameter takes. Mirrors
// python/hololink/_hololink.cpp's own geometry_to_dict().
nb::dict module_geometry_to_dict(const ad::csi::CsiGeometry& geometry)
{
    nb::dict out;
    out["start_byte"] = geometry.start_byte;
    out["bytes_per_line"] = geometry.bytes_per_line;
    out["pixel_width"] = geometry.pixel_width;
    out["pixel_height"] = geometry.pixel_height;
    out["pixel_format"] = geometry.pixel_format == ad::csi::PixelFormat::RAW_8 ? "raw_8"
        : geometry.pixel_format == ad::csi::PixelFormat::RAW_10                ? "raw_10"
                                                                               : "raw_12";
    out["trailing_bytes"] = geometry.trailing_bytes;
    return out;
}

#ifdef HOLOLINK_BUILD_FUSA
nb::dict resolve_fusa_coe_geometry_from_dimensions(
    int pixel_format, uint32_t width, uint32_t height, bool packetizer_enabled)
{
    ad::csi::CsiGeometry geometry;
    auto adapter = hololink::module::operators::FusaCoeCaptureOp::make_geometry_adapter(
        geometry, packetizer_enabled);
    auto fmt = static_cast<ad::csi::PixelFormat>(pixel_format);
    const uint32_t start_byte = adapter->receiver_start_byte();
    const uint32_t transmitted = adapter->transmitted_line_bytes(fmt, width);
    const uint32_t received = adapter->received_line_bytes(transmitted);
    adapter->configure(start_byte, received, width, height, fmt);
    return module_geometry_to_dict(geometry);
}
#endif // HOLOLINK_BUILD_FUSA

static void bind_operators(nb::module_& m)
{
    // The converter a sensor driver trains in place of CsiToBayerOp (see its class comment);
    // CsiConverterV1's methods are inherited, so any driver's configure_converter() accepts it.
    nb::class_<hololink::module::operators::CsiToBayerConverter, ad::csi::CsiConverterV1>(
        m, "CsiToBayerConverter")
        .def(nb::init<>())
        .def("get_csi_length", &hololink::module::operators::CsiToBayerConverter::get_csi_length)
        .def("geometry", [](const hololink::module::operators::CsiToBayerConverter& self) {
            return module_geometry_to_dict(self.geometry());
        });

    // NetworkReceiverFactory is registered opaquely (no constructor / methods)
    // only so make_*_network_receiver_factory()'s return value can be held in
    // Python and handed back to the HsbControllerOp constructor. The
    // application never touches its contents.
    nb::class_<hololink::module::operators::NetworkReceiverFactory>(
        m, "NetworkReceiverFactory")
        .def(
            "__holoscan_constructor_handle__",
            [](hololink::module::operators::NetworkReceiverFactory& self,
                const std::string& type_name) {
                return steal_constructor_handle(holoscan::python::export_constructor_handle(
                    type_name,
                    hololink::module::operators::network_receiver_factory_handle_type(),
                    self.weak_from_this().lock()));
            },
            nb::arg("type_name"),
            "Holoscan calls this to pass the factory to HsbControllerOp as a constructor handle.");

    // The software (Linux) data plane is always built (it links no ibverbs), so
    // its factory is bound unconditionally — unlike the RoCE one below.
    m.def("make_linux_network_receiver_factory",
        &hololink::module::operators::make_linux_network_receiver_factory,
        "Build a NetworkReceiverFactory for the software (Linux) data plane, to "
        "pass as HsbControllerOp's network_receiver_factory.");

    // The sensor wrapper: subclassed in Python to build + arm one camera.
    // nanobind's shared_ptr<T> support (via the <nanobind/stl/shared_ptr.h>
    // include) is what lets SensorFactory::new_sensor's return value load
    // across the Python boundary and co-own the underlying Python object.
    nb::class_<hololink::module::operators::SensorDevice,
        hololink::module::operators::PySensorDevice>(
        m, "SensorDevice")
        .def(nb::init<>())
        .def("stop_sensor", &hololink::module::operators::SensorDevice::stop_sensor);

    // The sensor-side controller: subclassed in Python (new_sensor) to supply
    // the camera. shared_ptr<T> support again (see SensorDevice above) is
    // what lets HsbControllerOp hold it as a shared_ptr<SensorFactory>
    // parameter.
    nb::class_<hololink::module::operators::SensorFactory,
        hololink::module::operators::PySensorFactory>(
        m, "SensorFactory")
        .def(nb::init<>())
        .def("tap", &hololink::module::operators::SensorFactory::tap)
        // Passes the factory to HsbControllerOp as its `sensor_factory` constructor handle.
        // Defined on the base so Python subclasses inherit it. A subclass is implemented in
        // Python, so the shared_ptr nanobind converts it to keeps the Python object alive for
        // as long as the Operator needs it.
        .def(
            "__holoscan_constructor_handle__",
            [](std::shared_ptr<hololink::module::operators::SensorFactory> self,
                const std::string& type_name) {
                return steal_constructor_handle(holoscan::python::export_constructor_handle(
                    type_name, hololink::module::operators::sensor_factory_handle_type(),
                    std::move(self)));
            },
            nb::arg("type_name"),
            "Holoscan calls this to pass the factory to HsbControllerOp as a constructor handle.");

    // LinuxReceiverOp, RoceReceiverOp, FusaCoeCaptureOp and HsbControllerOp are registered
    // operators (module_operator_catalog.cpp) that take their live objects as constructor
    // handles: the EnumerationMetadata as `metadata=`, the camera's start/stop methods as
    // `device_start=`/`device_stop=`, and HsbControllerOp's `sensor_factory=` and
    // `network_receiver_factory=` (see the __holoscan_constructor_handle__ methods above).

#ifdef HOLOLINK_BUILD_ROCE
    m.def("make_roce_network_receiver_factory",
        &hololink::module::operators::make_roce_network_receiver_factory,
        "Build a NetworkReceiverFactory for the RoCE (ibverbs) data plane, to "
        "pass as HsbControllerOp's network_receiver_factory.");
    // RoceReceiverOp is no longer bound here -- see this section's
    // top-of-section comment above.
#endif // HOLOLINK_BUILD_ROCE

    // FusaCoeCaptureOp / ComputeCrcOp / CheckCrcOp classes themselves are no longer bound
    // here -- see this section's top-of-section comment above. FusaCoeCaptureOp still needs its
    // geometry resolved before graph.op(hololink_module.operators.FusaCoeCaptureOp, ...) --
    // mirrors native's own resolve_geometry() bindings in python/hololink/_hololink.cpp.
#ifdef HOLOLINK_BUILD_FUSA
    m.def(
        "resolve_fusa_coe_geometry_from_dimensions", &resolve_fusa_coe_geometry_from_dimensions,
        nb::arg("pixel_format"), nb::arg("width"), nb::arg("height"),
        nb::arg("packetizer_enabled"),
        "Resolve CSI geometry (64-byte HSB line alignment) for a FusaCoeCaptureOp-fed "
        "pipeline, for pure-Python sensor drivers with no nanobind-bound camera type -- the "
        "module-tree, FUSA-specific equivalent of hololink.resolve_generic_geometry_from_"
        "dimensions() (generic 8-byte alignment). pixel_format is module csi.PixelFormat's "
        "own .value.");
#endif // HOLOLINK_BUILD_FUSA
}

// -------------------------------------------------------------------
// Vb1940Cam (hololink_module/host/sensors/vb1940/python/vb1940_py.cpp):
// module-side VB1940 camera bindings, plus its PixelFormat / BayerFormat /
// Vb1940_Mode enums.
// -------------------------------------------------------------------

static void bind_vb1940_cam(nb::module_& m)
{
    // Per-sensor CSI format enums. Integer values match the legacy
    // hololink::csi::* enums byte-for-byte, so passing
    // camera.bayer_format().value into legacy operators
    // (BayerDemosaicOp, ImageProcessorOp) works without conversion.
    // nb::is_arithmetic() makes these real IntEnum members (int(value) works
    // directly), matching both the legacy pybind11 binding's implicit behavior and
    // native's own identically-valued PixelFormat/BayerFormat -- see
    // python/hololink/_hololink.cpp's own comment.
    nb::enum_<ad::csi::PixelFormat>(m, "PixelFormat", nb::is_arithmetic())
        .value("RAW_8", ad::csi::PixelFormat::RAW_8)
        .value("RAW_10", ad::csi::PixelFormat::RAW_10)
        .value("RAW_12", ad::csi::PixelFormat::RAW_12);

    nb::enum_<ad::csi::BayerFormat>(m, "BayerFormat", nb::is_arithmetic())
        .value("BGGR", ad::csi::BayerFormat::BGGR)
        .value("RGGB", ad::csi::BayerFormat::RGGB)
        .value("GBRG", ad::csi::BayerFormat::GBRG)
        .value("GRBG", ad::csi::BayerFormat::GRBG);

    nb::enum_<v1940::Vb1940_Mode>(m, "Vb1940_Mode")
        .value("VB1940_MODE_2560X1984_30FPS",
            v1940::Vb1940_Mode::VB1940_MODE_2560X1984_30FPS)
        .value("VB1940_MODE_1920X1080_30FPS",
            v1940::Vb1940_Mode::VB1940_MODE_1920X1080_30FPS)
        .value("VB1940_MODE_2560X1984_30FPS_8BIT",
            v1940::Vb1940_Mode::VB1940_MODE_2560X1984_30FPS_8BIT)
        .value("VB1940_MODE_2560X1984_60FPS",
            v1940::Vb1940_Mode::VB1940_MODE_2560X1984_60FPS);

    nb::class_<v1940::Vb1940Cam>(m, "Vb1940Cam")
        // nb::new_(...) (not nb::init<>()/placement-new "__init__"): this
        // factory must return a std::shared_ptr<Vb1940Cam> built via
        // std::make_shared, matching the original pybind11 binding's
        // std::shared_ptr<Vb1940Cam> holder type exactly -- see
        // docs/classes.rst's "Preventing object destruction" / constructor
        // sections: nb::new_ is nanobind's documented mechanism for a
        // constructor whose factory lambda returns a pointer or smart
        // pointer rather than a plain by-value instance.
        .def(nb::new_([](const ad::EnumerationMetadata& metadata,
                          std::shared_ptr<ad::VsyncInterfaceV1> vsync) {
            return std::make_shared<v1940::Vb1940Cam>(metadata, std::move(vsync));
        }),
            nb::arg("metadata"), nb::arg("vsync") = nullptr)
        .def("configure", &v1940::Vb1940Cam::configure, nb::arg("mode"))
        .def("start", &v1940::Vb1940Cam::start)
        .def("stop", &v1940::Vb1940Cam::stop)
        .def("set_exposure_reg", &v1940::Vb1940Cam::set_exposure_reg,
            nb::arg("value") = 0x0014)
        .def("set_analog_gain_reg", &v1940::Vb1940Cam::set_analog_gain_reg,
            nb::arg("value") = 0x00)
        .def("width", &v1940::Vb1940Cam::width)
        .def("height", &v1940::Vb1940Cam::height)
        .def("pixel_format", &v1940::Vb1940Cam::pixel_format)
        .def("bayer_format", &v1940::Vb1940Cam::bayer_format)
        // configure_converter takes the module-native CsiConverterV1.
        // No in-tree Python player drives the module VB1940 binding, so
        // this is exposed for completeness; a Python caller would pass a
        // CsiConverterV1 implementation (e.g. a bridge wrapping a legacy
        // CsiToBayerOp).
        .def("configure_converter", &v1940::Vb1940Cam::configure_converter,
            nb::arg("converter"));
}

// -------------------------------------------------------------------
// TauroTechDa326InterfaceV1
// (hololink_module/module/taurotech_da326/python/taurotech_da326_py.cpp).
// -------------------------------------------------------------------

static void bind_taurotech_da326(nb::module_& m)
{
    nb::class_<ad::taurotech_da326::TauroTechDa326InterfaceV1>(
        m, "TauroTechDa326InterfaceV1")
        .def_static(
            "get_service",
            [](std::shared_ptr<ad::Module> module, const std::string& instance_id,
                bool allow_null) {
                return ad::taurotech_da326::TauroTechDa326InterfaceV1::get_service(
                    std::move(module), instance_id.c_str(), allow_null);
            },
            nb::arg("module"), nb::arg("instance_id"),
            nb::arg("allow_null") = false)
        .def_static(
            "get_service",
            [](const ad::EnumerationMetadata& metadata, bool allow_null) {
                return ad::taurotech_da326::TauroTechDa326InterfaceV1::get_service(
                    metadata, allow_null);
            },
            nb::arg("metadata"), nb::arg("allow_null") = false)
        .def("release_reset",
            [](ad::taurotech_da326::TauroTechDa326InterfaceV1& self) {
                const auto s = self.release_reset();
                if (s != HOLOLINK_MODULE_OK)
                    throw std::runtime_error(
                        "TauroTechDa326Interface.release_reset failed: "
                        + std::to_string(s));
            })
        .def("power_cycle",
            [](ad::taurotech_da326::TauroTechDa326InterfaceV1& self) {
                const auto s = self.power_cycle();
                if (s != HOLOLINK_MODULE_OK)
                    throw std::runtime_error(
                        "TauroTechDa326Interface.power_cycle failed: "
                        + std::to_string(s));
            })
        .def("check_power",
            [](ad::taurotech_da326::TauroTechDa326InterfaceV1& self) {
                const auto s = self.check_power();
                if (s != HOLOLINK_MODULE_OK)
                    throw std::runtime_error(
                        "TauroTechDa326Interface.check_power failed: "
                        + std::to_string(s));
            })
        .def("setup_clock",
            [](ad::taurotech_da326::TauroTechDa326InterfaceV1& self) {
                const auto s = self.setup_clock();
                if (s != HOLOLINK_MODULE_OK)
                    throw std::runtime_error(
                        "TauroTechDa326Interface.setup_clock failed: "
                        + std::to_string(s));
            })
        .def("hololink",
            &ad::taurotech_da326::TauroTechDa326InterfaceV1::hololink);
    nb::object cls = m.attr("TauroTechDa326InterfaceV1");
    cls.attr("type_id") = ad::taurotech_da326::TauroTechDa326InterfaceV1::type_id;
    cls.attr("release_version") = ad::taurotech_da326::TauroTechDa326InterfaceV1::release_version;
}

NB_MODULE(_hololink_module, m)
{
    // Forces the linker to pull module_operator_catalog.cpp's translation unit out of the
    // otherwise-unreferenced hololink_module_operators_hsdk5 static archive, the same way
    // python/hololink/_hololink.cpp's own NB_MODULE entry point does for
    // hololink::operators::operator_catalog -- without a real call site, nothing in this
    // binary references a symbol from that TU and its HOLOSCAN_REGISTER_OPERATOR static
    // initializers (CsiToBayerOp, ImageProcessorOp, PackedFormatConverterOp, ComputeCrcOp)
    // never run, leaving hololink_module.operators empty at import time.
    auto registered = hololink::module::operators::register_hololink_module_operator_catalog();
    if (!registered) {
        throw holoscan::RuntimeError { registered.error() };
    }

    m.doc() = "hololink_module native bindings (core, operators, sensor, and "
              "board-supplement surface).";

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
        nb::arg("message"), "Log a message at TRACE level through the module logger.");
    m.def(
        "hsb_log_debug", [](const std::string& message) { HSB_LOG_DEBUG("{}", message); },
        nb::arg("message"), "Log a message at DEBUG level through the module logger.");
    m.def(
        "hsb_log_info", [](const std::string& message) { HSB_LOG_INFO("{}", message); },
        nb::arg("message"), "Log a message at INFO level through the module logger.");
    m.def(
        "hsb_log_warn", [](const std::string& message) { HSB_LOG_WARN("{}", message); },
        nb::arg("message"), "Log a message at WARNING level through the module logger.");
    m.def(
        "hsb_log_error", [](const std::string& message) { HSB_LOG_ERROR("{}", message); },
        nb::arg("message"), "Log a message at ERROR level through the module logger.");

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
        [](const std::string& peer_ip, const std::string& interface) {
            return ad::ibv_device_for_peer(peer_ip, interface);
        },
        nb::arg("peer_ip"), nb::arg("interface") = std::string(),
        "Return (ibv_name, ibv_port) for the IB device whose underlying "
        "kernel netdev is `interface` -- the NIC the board enumerated on, as "
        "carried in the enumeration metadata. Falls back to the route-resolved "
        "local interface for peer_ip when `interface` is empty, which is "
        "ambiguous if two boards share a data-plane subnet. "
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

    bind_operators(m);
    bind_vb1940_cam(m);
    bind_taurotech_da326(m);
}
