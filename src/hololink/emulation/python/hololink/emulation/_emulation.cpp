/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// hololink.emulation._emulation: the HSB emulator's Python module, on nanobind.
//
// One extension holds both the core emulator classes and, as the `sensors` submodule, the
// C++ sensor emulators. Keeping them in one module lets the sensors derive from the bound
// I2CPeripheral without sharing nanobind's type registry across two extensions, and it
// removes the lazy-import workaround the two-module layout needed in __init__.py.
//
// The Python-callable i2c_transaction()/set_eeprom_data() borrow the caller's buffer through
// the buffer protocol, and DataPlane.send() consumes the tensor's DLPack capsule directly. An
// I2CPeripheral.i2c_transaction override, on the other hand, gets copies: `bytes` for the
// write data and a `bytearray` for the reply, copied into the controller's buffer when the
// override returns. A view over the controller's own storage can't be revoked (a slice or
// cast of a released memoryview stays writable), so it could outlive the transaction and
// scribble over a later one; a transfer is at most I2C_DATA_BUFFER_SIZE (256) bytes, so the
// copies cost nothing worth that risk. All of this is plain CPython limited API, so the module
// builds with nanobind's STABLE_ABI like hololink's _hololink module.
//
// Python code runs on the emulator's control thread (peripheral overrides, control-plane
// callbacks). An exception there must not unwind into C++ -- std::terminate -- so every such
// entry point catches it, reports it as unraisable (sys.unraisablehook) and returns an error
// status instead.

#include "dlpack/dlpack.h"
#include "hololink/emulation/coe_data_plane.hpp"
#include "hololink/emulation/data_plane.hpp"
#include "hololink/emulation/hsb_config.hpp"
#include "hololink/emulation/hsb_emulator.hpp"
#include "hololink/emulation/i2c_interface.hpp"
#include "hololink/emulation/rocev2_data_plane.hpp"
#include "hololink/emulation/sensors/imx274_emulator.hpp"
#include "hololink/emulation/sensors/test_sensor.hpp"
#include "hololink/emulation/sensors/vb1940_emulator.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/trampoline.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace nb = nanobind;
using namespace nb::literals;

namespace hololink::emulation {

namespace {

    // ---------------------------------------------------------------------------------------
    // Buffer-protocol helpers (zero-copy views of caller-owned Python buffers)
    // ---------------------------------------------------------------------------------------

    // A Py_buffer that releases itself. Requested with PyBUF_STRIDES so the exporter hands
    // us ndim/itemsize/strides and we can give precise ValueErrors instead of letting a
    // non-contiguous numpy view fail deep inside the exporter.
    struct ByteBuffer {
        Py_buffer view {};
        bool held = false;

        ByteBuffer() = default;
        ByteBuffer(const ByteBuffer&) = delete;
        ByteBuffer& operator=(const ByteBuffer&) = delete;
        ~ByteBuffer()
        {
            if (held) {
                PyBuffer_Release(&view);
            }
        }

        // Borrow `obj` as a 1-D contiguous byte buffer; `writable` requests write access
        // (bytes then fails with BufferError, as the buffer protocol defines). None leaves
        // the buffer empty (ptr() == nullptr, size() == 0).
        void request(nb::handle obj, bool writable, const char* who, const char* name)
        {
            if (obj.is_none()) {
                return;
            }
            int flags = PyBUF_STRIDES | (writable ? PyBUF_WRITABLE : 0);
            if (PyObject_GetBuffer(obj.ptr(), &view, flags) != 0) {
                throw nb::python_error();
            }
            held = true;
            if (view.ndim != 1) {
                throw nb::value_error((std::string(who) + " expects a 1-D buffer for " + name).c_str());
            }
            if (view.itemsize != 1) {
                throw nb::value_error((std::string(who) + " expects a byte buffer (itemsize==1) for " + name).c_str());
            }
            // For a 1-D itemsize-1 buffer, C-contiguous means strides[0] == 1.
            if (view.strides != nullptr && view.len > 1 && view.strides[0] != 1) {
                throw nb::value_error((std::string(who) + " expects a contiguous buffer for " + name).c_str());
            }
        }

        uint8_t* ptr() const { return held ? static_cast<uint8_t*>(view.buf) : nullptr; }
        size_t size() const { return held ? static_cast<size_t>(view.len) : 0; }
    };

    // Resolve an optional explicit transfer size against the bytes available in `buffer`.
    uint16_t resolve_transfer_size(const ByteBuffer& buffer, nb::handle size_obj, const char* name)
    {
        size_t size = buffer.size();
        if (!size_obj.is_none()) {
            size = nb::cast<size_t>(size_obj);
            if (size > buffer.size()) {
                throw nb::value_error((std::string(name) + " exceeds the length of the buffer passed").c_str());
            }
        }
        if (size > 0xFFFF) {
            throw nb::value_error((std::string(name) + " exceeds the 16-bit I2C transfer limit").c_str());
        }
        return static_cast<uint16_t>(size);
    }

    // Keep `peripheral_obj` alive for as long as the emulator behind `controller_obj` (see
    // "Lifetime of attached peripherals" in bind_emulator): a keep_alive record on the
    // emulator's Python object. HSBEmulator.get_i2c() leaves the emulator on the controller
    // wrapper as `_hsb_emulator`; a controller handle that came from anywhere else falls back
    // to holding the peripheral itself.
    void keep_peripheral_alive(nb::handle controller_obj, nb::handle peripheral_obj)
    {
        nb::object nurse = nb::borrow(controller_obj);
        if (nb::hasattr(controller_obj, "_hsb_emulator")) {
            nurse = controller_obj.attr("_hsb_emulator");
        }
        NB_CALL(keep_alive_py)
        (NB_CTX, nurse.ptr(), peripheral_obj.ptr());
    }

    // attach_to_i2c for a bound C++ peripheral type: call the (virtual) C++ method with the
    // GIL released (it takes the controller's i2c_mutex, which the control thread may hold
    // while it waits for the GIL inside a Python peripheral), then keep the peripheral alive.
    // A Python subclass's own attach_to_i2c override reaches attach_i2c_peripheral() below,
    // which does the same.
    template <typename Peripheral>
    void def_attach_to_i2c(nb::class_<Peripheral, I2CPeripheral>& cls, const char* doc)
    {
        cls.def(
            "attach_to_i2c",
            [](nb::object self_obj, nb::object controller_obj, uint8_t bus_address) {
                Peripheral& peripheral = nb::cast<Peripheral&>(self_obj);
                I2CController& controller = nb::cast<I2CController&>(controller_obj);
                {
                    nb::gil_scoped_release nogil;
                    peripheral.attach_to_i2c(controller, bus_address);
                }
                keep_peripheral_alive(controller_obj, self_obj);
            },
            "i2c_controller"_a, "bus_address"_a, doc);
    }

    // Report a Python exception raised on the emulator's thread, where nobody can catch it,
    // through sys.unraisablehook. Must run with the GIL held.
    void report_unraisable(nb::python_error& e, const char* context)
    {
        e.discard_as_unraisable(context);
    }

    // The same for a C++ exception (a wrong return type is a nb::cast_error): surface its
    // message as a RuntimeError through sys.unraisablehook.
    void report_unraisable(const std::exception& e, const char* context)
    {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        nb::python_error error;
        error.discard_as_unraisable(context);
    }

    // ---------------------------------------------------------------------------------------
    // DataPlane.send(tensor): consume the tensor's DLPack capsule, no host copy here
    // ---------------------------------------------------------------------------------------

    int64_t py_send_dltensor(DataPlane& self, nb::object py_tensor)
    {
        nb::object capsule = py_tensor.attr("__dlpack__")();
        if (!PyCapsule_CheckExact(capsule.ptr())) {
            throw std::runtime_error("object does not satisfy Python array API standard - __dlpack__() returned non-capsule object");
        }
        DLTensor* tensor = static_cast<DLTensor*>(PyCapsule_GetPointer(capsule.ptr(), "dltensor"));
        if (tensor == nullptr) {
            PyErr_Clear();
            throw std::runtime_error("could not extract DLTensor from PyCapsule");
        }
        // The capsule keeps its name, so its destructor still frees the DLManagedTensor when
        // `capsule` goes out of scope; send() must therefore not retain the tensor.
        //
        // Release the GIL around the native send: for GPU tensors it does a synchronous
        // device->host cudaMemcpy plus a blocking socket send. Holding the GIL across those
        // would freeze every other Python thread -- including a timeout watchdog and the
        // emulator's own control-plane callbacks -- so a driver/socket stall would become an
        // unrecoverable hang. `capsule` stays alive for the whole call, keeping tensor->data
        // valid.
        int64_t sent;
        {
            nb::gil_scoped_release nogil;
            sent = self.send(*tensor);
        }
        return sent;
    }

    // ---------------------------------------------------------------------------------------
    // Trampolines
    // ---------------------------------------------------------------------------------------

    class PyDataPlane : public DataPlane {
    public:
        NB_TRAMPOLINE(DataPlane);

        void update_metadata() override
        {
            NB_OVERRIDE(update_metadata);
        }
    };

    // update_metadata() is protected in DataPlane; this exposes it for the binding below.
    struct DataPlanePublicist : public DataPlane {
        using DataPlane::update_metadata;
    };

    class PyI2CPeripheral : public I2CPeripheral {
    public:
        NB_TRAMPOLINE(I2CPeripheral);

        // NB_OVERRIDE_PURE cannot be used here: nanobind would cast the `uint8_t*` arguments
        // as pointers to single bytes, so the Python override would see only the first byte
        // of each buffer and could not hand read results back. Marshal by hand instead:
        //   - write_bytes -> an immutable `bytes` copy (the controller passes the SAME storage
        //     for write_bytes and read_bytes, so the write data must be copied out before the
        //     reply is copied in anyway);
        //   - read_bytes -> a `bytearray` of read_size bytes that the override fills
        //     (`read_bytes[0:n] = ...`) and that is copied into the controller's buffer when
        //     it returns. A reference the override keeps is then just a stale copy (see the
        //     file comment for why it is not a view over the controller's buffer).
        // The ticket acquires the GIL for its lifetime and raises if the Python subclass has
        // no override (the method is pure virtual). This runs on the control thread: an
        // exception from the override, a wrong return type or a resized read_bytes ends the
        // transaction with I2C_STATUS_FATAL_ERROR (the controller reports I2C_I2C_ERR) and is
        // reported as unraisable, rather than unwinding into the emulator.
        I2CStatus i2c_transaction(uint16_t peripheral_address, const uint8_t* write_bytes, uint16_t write_size, uint8_t* read_bytes, uint16_t read_size) override
        {
            static constexpr const char* context = "hololink.emulation.I2CPeripheral.i2c_transaction (Python override)";
            try {
                constexpr uint64_t hash = nb::detail::str_hash("i2c_transaction");
                nb::detail::ticket ticket(nb_trampoline, "i2c_transaction", hash, /*pure=*/true);

                nb::bytes write(write_size ? static_cast<const void*>(write_bytes) : static_cast<const void*>(""), write_size);
                nb::bytearray read(
                    (read_bytes != nullptr && read_size > 0) ? static_cast<const void*>(read_bytes) : static_cast<const void*>(""),
                    read_size);

                nb::object result = nb_trampoline.base().attr(ticket.key)(peripheral_address, write, write_size, read, read_size);
                const I2CStatus status = nb::cast<I2CStatus>(result);
                if (read.size() != read_size) {
                    throw std::runtime_error("the override resized read_bytes (" + std::to_string(read.size())
                        + " bytes); it must fill the " + std::to_string(read_size) + " bytes it was given");
                }
                if (read_size > 0) {
                    std::memcpy(read_bytes, read.data(), read_size);
                }
                return status;
            } catch (nb::python_error& e) {
                nb::gil_scoped_acquire gil;
                report_unraisable(e, context);
            } catch (const std::exception& e) {
                nb::gil_scoped_acquire gil;
                report_unraisable(e, context);
            }
            return I2CStatus::I2C_STATUS_FATAL_ERROR;
        }

        // start()/stop() are called by the controller, on whichever thread starts or stops the
        // emulator; an exception from the override is reported and otherwise ignored.
        void start() override
        {
            try {
                NB_OVERRIDE(start);
            } catch (nb::python_error& e) {
                nb::gil_scoped_acquire gil;
                report_unraisable(e, "hololink.emulation.I2CPeripheral.start (Python override)");
            }
        }

        void stop() override
        {
            try {
                NB_OVERRIDE(stop);
            } catch (nb::python_error& e) {
                nb::gil_scoped_acquire gil;
                report_unraisable(e, "hololink.emulation.I2CPeripheral.stop (Python override)");
            }
        }

        void attach_to_i2c(I2CController& i2c_controller, uint8_t i2c_bus_address) override
        {
            NB_OVERRIDE_PURE(attach_to_i2c, i2c_controller, i2c_bus_address);
        }
    };

    // ---------------------------------------------------------------------------------------
    // Bindings
    // ---------------------------------------------------------------------------------------

    void bind_config(nb::module_& m)
    {
        nb::class_<HSBConfiguration>(m, "HSBConfiguration")
            .def(nb::init<>())
            .def_rw("tag", &HSBConfiguration::tag)
            .def_rw("tag_length", &HSBConfiguration::tag_length)
            .def_prop_rw(
                "vendor_id",
                [](const HSBConfiguration& self) -> std::string {
                    return std::string(reinterpret_cast<const char*>(self.vendor_id), VENDOR_ID_SIZE);
                },
                [](HSBConfiguration& self, const std::string& vendor_id) {
                    if (vendor_id.length() != VENDOR_ID_SIZE) {
                        throw std::runtime_error("Vendor ID must be exactly 4 characters");
                    }
                    std::memcpy(self.vendor_id, vendor_id.c_str(), VENDOR_ID_SIZE);
                })
            .def_rw("data_plane", &HSBConfiguration::data_plane)
            .def_rw("enum_version", &HSBConfiguration::enum_version)
            .def_rw("board_id_lo", &HSBConfiguration::board_id_lo)
            .def_rw("board_id_hi", &HSBConfiguration::board_id_hi)
            .def_prop_rw(
                "uuid",
                [](const HSBConfiguration& self) -> std::string {
                    return std::string(reinterpret_cast<const char*>(self.uuid), BOARD_VERSION_SIZE);
                },
                [](HSBConfiguration& self, const std::string& uuid_str) {
                    if (hsb_config_set_uuid(&self, uuid_str.c_str()) != 0) {
                        throw std::runtime_error("Invalid UUID format");
                    }
                })
            .def_prop_rw(
                "serial_num",
                [](const HSBConfiguration& self) -> std::string {
                    return std::string(reinterpret_cast<const char*>(self.serial_num), BOARD_SERIAL_NUM_SIZE);
                },
                [](HSBConfiguration& self, const std::string& serial_num) {
                    if (serial_num.length() != BOARD_SERIAL_NUM_SIZE) {
                        throw std::runtime_error("Serial number must be exactly 7 characters");
                    }
                    std::memcpy(self.serial_num, serial_num.c_str(), BOARD_SERIAL_NUM_SIZE);
                })
            .def_rw("hsb_ip_version", &HSBConfiguration::hsb_ip_version)
            .def_rw("fpga_crc", &HSBConfiguration::fpga_crc)
            .def_rw("sensor_count", &HSBConfiguration::sensor_count)
            .def_rw("data_plane_count", &HSBConfiguration::data_plane_count)
            .def_rw("sifs_per_sensor", &HSBConfiguration::sifs_per_sensor)
            // Pickle support -- enables copy.copy() / copy.deepcopy() and pickle.dumps/loads on
            // HSBConfiguration, so applications can clone HSB_EMULATOR_CONFIG and tweak fields
            // without mutating the module-level constant in place. State is a tuple of all 14
            // fields; the fixed-size byte arrays travel as bytes so binary content (UUID,
            // serial_num, vendor_id) round-trips losslessly.
            .def("__getstate__", [](const HSBConfiguration& self) {
                return nb::make_tuple(
                    self.tag,
                    self.tag_length,
                    nb::bytes(self.vendor_id, VENDOR_ID_SIZE),
                    self.data_plane,
                    self.enum_version,
                    self.board_id_lo,
                    self.board_id_hi,
                    nb::bytes(self.uuid, BOARD_VERSION_SIZE),
                    nb::bytes(self.serial_num, BOARD_SERIAL_NUM_SIZE),
                    self.hsb_ip_version,
                    self.fpga_crc,
                    self.sensor_count,
                    self.data_plane_count,
                    self.sifs_per_sensor);
            })
            .def("__setstate__", [](HSBConfiguration& self, nb::tuple t) {
                if (t.size() != 14) {
                    throw std::runtime_error(
                        "Invalid HSBConfiguration pickle state: expected 14 fields, got "
                        + std::to_string(t.size()));
                }
                auto item = [&t](size_t i) -> nb::object { return t[i]; };
                auto fixed_bytes = [&item](size_t i, uint8_t* dest, size_t size, const char* name) {
                    nb::bytes v = nb::cast<nb::bytes>(item(i));
                    if (v.size() != size) {
                        throw std::runtime_error(
                            std::string("Invalid ") + name + " length in pickle state: expected "
                            + std::to_string(size) + ", got " + std::to_string(v.size()));
                    }
                    std::memcpy(dest, v.c_str(), size);
                };
                HSBConfiguration cfg {};
                cfg.tag = nb::cast<uint8_t>(item(0));
                cfg.tag_length = nb::cast<uint8_t>(item(1));
                fixed_bytes(2, cfg.vendor_id, VENDOR_ID_SIZE, "vendor_id");
                cfg.data_plane = nb::cast<uint8_t>(item(3));
                cfg.enum_version = nb::cast<uint8_t>(item(4));
                cfg.board_id_lo = nb::cast<uint8_t>(item(5));
                cfg.board_id_hi = nb::cast<uint8_t>(item(6));
                fixed_bytes(7, cfg.uuid, BOARD_VERSION_SIZE, "uuid");
                fixed_bytes(8, cfg.serial_num, BOARD_SERIAL_NUM_SIZE, "serial_num");
                cfg.hsb_ip_version = nb::cast<uint16_t>(item(9));
                cfg.fpga_crc = nb::cast<uint16_t>(item(10));
                cfg.sensor_count = nb::cast<uint8_t>(item(11));
                cfg.data_plane_count = nb::cast<uint8_t>(item(12));
                cfg.sifs_per_sensor = nb::cast<uint8_t>(item(13));
                new (&self) HSBConfiguration(cfg);
            });

        m.attr("HSB_EMULATOR_CONFIG") = HSB_EMULATOR_CONFIG;
        m.attr("HSB_LEOPARD_EAGLE_CONFIG") = HSB_LEOPARD_EAGLE_CONFIG;

        // Helper function for setting UUID in HSBConfiguration
        m.def("hsb_config_set_uuid", &hsb_config_set_uuid, "config"_a, "uuid_str"_a, "Set UUID in HSBConfiguration from string");

        nb::class_<IPAddress>(m, "IPAddress")
            .def(
                "__init__",
                [](IPAddress* self, const std::string& ip_address) {
                    new (self) IPAddress(IPAddress_from_string(ip_address));
                },
                "ip_address"_a)
            .def("__str__", &IPAddress_to_string);

        nb::class_<AddressValuePair>(m, "AddressValuePair",
            "Address-value pair used by control-plane read/write callbacks. Callbacks receive a list of these.")
            .def(nb::init<>())
            .def_rw("address", &AddressValuePair::address, "Register address")
            .def_rw("value", &AddressValuePair::value, "Register value");
    }

    // The HSBEmulator Python type wraps this subclass: its destructor stops the emulator
    // first, with the GIL released. nanobind destroys an instance in this order: clear its
    // __dict__, run the C++ destructor, release its keep_alive patients. The attached
    // peripherals are keep_alive patients (see "Lifetime of attached peripherals" below), so
    // they are still alive while stop() joins the control thread, which may be inside one of
    // them; and that thread needs the GIL to get out, which is why it is released here.
    // HSBEmulator::stop() is a no-op once stopped, so the base destructor finds nothing to do.
    class PyHSBEmulator : public HSBEmulator {
    public:
        using HSBEmulator::HSBEmulator;
        ~PyHSBEmulator()
        {
            nb::gil_scoped_release nogil;
            stop();
        }
    };

    void bind_emulator(nb::module_& m)
    {
        // Control-plane callbacks run on the emulator's control-plane thread, so the wrapper
        // takes the GIL before touching Python. The Python callback sees
        // (ctxt, [AddressValuePair, ...]) and returns an int: for reads, the number of
        // pairs whose .value it filled in. An exception, or a result that is not an int,
        // is reported as unraisable and the request fails (a result <= 0 makes the control
        // plane answer with an address error) instead of unwinding into the emulator.
        auto register_read_callback = [](HSBEmulator& self, uint32_t start_address, uint32_t end_address, nb::callable callback, nb::object ctxt) {
            auto wrapper = [callback, ctxt](void* /* ctxt_ptr */, AddressValuePair* addr_val, int count) -> int {
                nb::gil_scoped_acquire gil;
                try {
                    nb::list pairs;
                    for (int i = 0; i < count; ++i) {
                        pairs.append(nb::cast(addr_val[i]));
                    }
                    int result = nb::cast<int>(callback(ctxt, pairs));
                    for (int i = 0; i < result && i < count; ++i) {
                        addr_val[i].value = nb::cast<uint32_t>(pairs[i].attr("value"));
                    }
                    return result;
                } catch (nb::python_error& e) {
                    report_unraisable(e, "hololink.emulation.HSBEmulator read callback");
                } catch (const std::exception& e) {
                    report_unraisable(e, "hololink.emulation.HSBEmulator read callback");
                }
                return -1;
            };
            return self.register_read_callback(start_address, end_address, wrapper, nullptr);
        };

        auto register_write_callback = [](HSBEmulator& self, uint32_t start_address, uint32_t end_address, nb::callable callback, nb::object ctxt) {
            auto wrapper = [callback, ctxt](void* /* ctxt_ptr */, AddressValuePair* addr_val, int count) -> int {
                nb::gil_scoped_acquire gil;
                try {
                    nb::list pairs;
                    for (int i = 0; i < count; ++i) {
                        pairs.append(nb::cast(addr_val[i]));
                    }
                    return nb::cast<int>(callback(ctxt, pairs));
                } catch (nb::python_error& e) {
                    report_unraisable(e, "hololink.emulation.HSBEmulator write callback");
                } catch (const std::exception& e) {
                    report_unraisable(e, "hololink.emulation.HSBEmulator write callback");
                }
                return -1;
            };
            return self.register_write_callback(start_address, end_address, wrapper, nullptr);
        };

        // Lifetime of attached peripherals
        // --------------------------------
        // I2CController::attach_i2c_peripheral stores a raw I2CPeripheral*; in C++ the caller
        // owns the object and keeps it alive while attached. A Python peripheral is owned by
        // its Python object, so if the application dropped its last reference the controller
        // would be left with a dangling pointer. So every attach (attach_to_i2c on a C++
        // sensor, attach_i2c_peripheral on the controller) adds the peripheral as a
        // keep_alive patient of the emulator's Python object, which releases it only after
        // the C++ emulator is destroyed -- i.e. after the control thread that could still
        // call it has been joined (see PyHSBEmulator). Instance attributes would not do:
        // nanobind clears an instance's __dict__ before it runs the C++ destructor, so a
        // peripheral held that way died while the control thread could still reach it.
        //
        // get_i2c() returns the controller with reference_internal: the wrapper keeps the
        // emulator alive, and the emulator holds no reference to the wrapper, so dropping the
        // last reference to both destroys the emulator at once (a cycle through keep_alive
        // records would be invisible to the garbage collector). The wrapper carries the
        // emulator as `_hsb_emulator`, which is how the attach bindings find the nurse. One
        // C++ controller gets one wrapper for as long as any reference to it exists: on Linux
        // get_i2c() returns the same controller whatever the address, so attaching through a
        // wrapper obtained with another address keeps the peripheral alive just the same.
        auto get_i2c = [](nb::object self_obj, uint32_t controller_address) -> nb::object {
            HSBEmulator& emulator = nb::cast<HSBEmulator&>(self_obj);
            I2CController& controller = emulator.get_i2c(controller_address);
            nb::object controller_obj = nb::cast(&controller, nb::rv_policy::reference_internal, self_obj);
            controller_obj.attr("_hsb_emulator") = self_obj;
            return controller_obj;
        };

        // The C++ class is registered under a private name so that HSBEmulator& parameters
        // (the data planes' constructors) accept the wrapper type below.
        nb::class_<HSBEmulator>(m, "_HSBEmulatorBase");
        // start()/stop()/write()/read() release the GIL: they take the emulator's locks
        // (I2CController::start/stop and a write that triggers an I2C transaction take
        // i2c_mutex), which the control thread may hold while it waits for the GIL inside a
        // Python peripheral or callback. Holding the GIL across them would deadlock the two.
        nb::class_<PyHSBEmulator, HSBEmulator>(m, "HSBEmulator")
            .def(nb::init<const HSBConfiguration&>(), "config"_a = HSB_EMULATOR_CONFIG)
            .def("start", &HSBEmulator::start, nb::call_guard<nb::gil_scoped_release>(), "start HSB emulator")
            .def("stop", &HSBEmulator::stop, nb::call_guard<nb::gil_scoped_release>(), "stop HSB emulator")
            .def("is_running", &HSBEmulator::is_running, "check if HSB emulator is running")
            .def("register_read_callback", register_read_callback, "start_address"_a, "end_address"_a, "callback"_a, "ctxt"_a = nb::none(),
                "register a read callback for a range of addresses")
            .def("register_write_callback", register_write_callback, "start_address"_a, "end_address"_a, "callback"_a, "ctxt"_a = nb::none(),
                "register a write callback for a range of addresses")
            .def("get_i2c", get_i2c, "controller_address"_a = hololink::I2C_CTRL,
                "get the I2CController at controller_address (default: hololink.I2C_CTRL); the controller keeps the emulator alive")
            .def("write", &HSBEmulator::write, nb::call_guard<nb::gil_scoped_release>(), "address"_a, "value"_a,
                "write a 32-bit register in the HSB emulator; returns 0 on success, non-zero if no register is mapped at address")
            // HSBEmulator::read returns the value through a reference out-parameter, which a
            // binding cannot express; return the value and raise on failure instead.
            .def(
                "read",
                [](HSBEmulator& self, uint32_t address) -> uint32_t {
                    uint32_t value = 0;
                    int rc;
                    {
                        nb::gil_scoped_release nogil;
                        rc = self.read(address, value);
                    }
                    if (rc != 0) {
                        throw std::runtime_error("HSBEmulator.read: no register is mapped at address " + std::to_string(address));
                    }
                    return value;
                },
                "address"_a,
                "read a 32-bit register from the HSB emulator; raises RuntimeError if no register is mapped at address");

        nb::class_<DataPlane, PyDataPlane>(m, "DataPlane")
            .def("send", &py_send_dltensor, "tensor"_a,
                "send a frame; tensor is any object implementing __dlpack__ (host or CUDA memory)")
            // A data plane's thread may run a Python update_metadata override (PyDataPlane), so
            // start()/stop() release the GIL like the emulator's do.
            .def("start", &DataPlane::start, nb::call_guard<nb::gil_scoped_release>(), "start DataPlane")
            .def("stop", &DataPlane::stop, nb::call_guard<nb::gil_scoped_release>(), "stop DataPlane")
            .def("stop_bootp", &DataPlane::stop_bootp, "stop bootp on DataPlane")
            .def("is_running", &DataPlane::is_running, "check if DataPlane is running")
            .def("get_sensor_id", &DataPlane::get_sensor_id, "get sensor id from DataPlane")
            .def("update_metadata", &DataPlanePublicist::update_metadata, "update DataPlane metadata")
            .def("packetizer_enabled", &DataPlane::packetizer_enabled, "check if packetizer is enabled");

        // keep_alive<1, 2>: a data plane keeps its HSBEmulator alive.
        nb::class_<RoCEv2DataPlane, DataPlane>(m, "RoCEv2DataPlane")
            .def(nb::init<HSBEmulator&, const IPAddress&, uint8_t, uint8_t>(),
                "hsb_emulator"_a, "source_ip"_a, "data_plane_id"_a, "sensor_id"_a, nb::keep_alive<1, 2>());
        // update_metadata is not overridable from Python on the concrete data planes
        // (no trampoline): subclassing extensions of DataPlane subclasses are not supported yet.

        nb::class_<COEDataPlane, DataPlane>(m, "COEDataPlane")
            .def(nb::init<HSBEmulator&, const IPAddress&, uint8_t, uint8_t>(),
                "hsb_emulator"_a, "source_ip"_a, "data_plane_id"_a, "sensor_id"_a, nb::keep_alive<1, 2>());
    }

    void bind_i2c(nb::module_& m)
    {
        nb::enum_<I2CStatus>(m, "I2CStatus")
            .value("I2C_STATUS_SUCCESS", I2CStatus::I2C_STATUS_SUCCESS)
            .value("I2C_STATUS_BUSY", I2CStatus::I2C_STATUS_BUSY)
            .value("I2C_STATUS_WRITE_FAILED", I2CStatus::I2C_STATUS_WRITE_FAILED)
            .value("I2C_STATUS_READ_FAILED", I2CStatus::I2C_STATUS_READ_FAILED)
            .value("I2C_STATUS_MESSAGE_NOT_UNDERSTOOD", I2CStatus::I2C_STATUS_MESSAGE_NOT_UNDERSTOOD)
            .value("I2C_STATUS_TIMEOUT", I2CStatus::I2C_STATUS_TIMEOUT)
            .value("I2C_STATUS_NACK", I2CStatus::I2C_STATUS_NACK)
            .value("I2C_STATUS_BUFFER_SIZE_SMALL", I2CStatus::I2C_STATUS_BUFFER_SIZE_SMALL)
            .value("I2C_STATUS_BUFFER_SIZE_LARGE", I2CStatus::I2C_STATUS_BUFFER_SIZE_LARGE)
            .value("I2C_STATUS_INVALID_PERIPHERAL_ADDRESS", I2CStatus::I2C_STATUS_INVALID_PERIPHERAL_ADDRESS)
            .value("I2C_STATUS_INVALID_REGISTER_ADDRESS", I2CStatus::I2C_STATUS_INVALID_REGISTER_ADDRESS)
            .value("I2C_STATUS_FATAL_ERROR", I2CStatus::I2C_STATUS_FATAL_ERROR);

        // Python-callable form of i2c_transaction (e.g. driving a C++ sensor emulator such as
        // hemu.sensors.TestSensor directly from a test, or chaining to another peripheral).
        // Binding the member function pointer directly would expose the raw `uint8_t*`
        // parameters as single ints; borrow buffer-protocol objects instead, zero-copy.
        // write_bytes is any 1-D byte buffer (bytes, bytearray, memoryview, ...); read_bytes
        // must be a *writable* 1-D byte buffer (bytearray, memoryview over one, ...) that the
        // peripheral fills in place. write_size / read_size default to the buffer lengths and
        // may be given to use only a prefix of a larger buffer.
        auto i2c_transaction = [](I2CPeripheral& self, uint16_t peripheral_address, nb::handle write_bytes, nb::handle read_bytes, nb::handle write_size, nb::handle read_size) -> I2CStatus {
            ByteBuffer write;
            ByteBuffer read;
            write.request(write_bytes, false, "i2c_transaction", "write_bytes");
            read.request(read_bytes, true, "i2c_transaction", "read_bytes");
            uint16_t write_count = resolve_transfer_size(write, write_size, "write_size");
            uint16_t read_count = resolve_transfer_size(read, read_size, "read_size");
            // Without the GIL: a C++ sensor may take its own locks, and a Python peripheral's
            // override reacquires it through the trampoline.
            nb::gil_scoped_release nogil;
            return self.i2c_transaction(peripheral_address, write.ptr(), write_count, read.ptr(), read_count);
        };

        nb::class_<I2CPeripheral, PyI2CPeripheral>(m, "I2CPeripheral")
            .def(nb::init<>())
            .def("i2c_transaction", i2c_transaction,
                "peripheral_address"_a, "write_bytes"_a, "read_bytes"_a = nb::none(), "write_size"_a = nb::none(), "read_size"_a = nb::none(),
                "perform an I2C transaction: write_bytes is a byte buffer to send, read_bytes a writable byte buffer the peripheral fills in place")
            .def("start", &I2CPeripheral::start, "start I2CPeripheral")
            .def("stop", &I2CPeripheral::stop, "stop I2CPeripheral")
            // Covers C++ peripherals whose attach_to_i2c() calls attach_i2c_peripheral()
            // natively, bypassing the Python binding below; a Python override reaches
            // attach_i2c_peripheral() itself, which keeps the peripheral alive. The GIL stays
            // held here: a Python override needs it, and a C++ one gets it released inside
            // attach_i2c_peripheral() below only when called from Python.
            .def(
                "attach_to_i2c",
                [](nb::object self_obj, nb::object controller_obj, uint8_t bus_address) {
                    nb::cast<I2CPeripheral&>(self_obj).attach_to_i2c(nb::cast<I2CController&>(controller_obj), bus_address);
                    keep_peripheral_alive(controller_obj, self_obj);
                },
                "i2c_controller"_a, "bus_address"_a,
                "attach I2CPeripheral to I2CController; the emulator keeps the peripheral alive while attached");

        // dynamic_attr: get_i2c() leaves the emulator on the wrapper as `_hsb_emulator`.
        nb::class_<I2CController>(m, "I2CController", nb::dynamic_attr())
            .def(
                "attach_i2c_peripheral",
                [](nb::object self_obj, uint32_t bus_address, uint16_t peripheral_address, nb::object peripheral_obj) {
                    I2CController& controller = nb::cast<I2CController&>(self_obj);
                    I2CPeripheral* peripheral = nb::cast<I2CPeripheral*>(peripheral_obj);
                    {
                        // Takes i2c_mutex (see HSBEmulator.start above).
                        nb::gil_scoped_release nogil;
                        controller.attach_i2c_peripheral(bus_address, peripheral_address, peripheral);
                    }
                    keep_peripheral_alive(self_obj, peripheral_obj);
                },
                "bus_address"_a, "peripheral_address"_a, "peripheral"_a,
                "attach I2CPeripheral to I2CController; the emulator keeps the peripheral alive while attached");
    }

    // The C++ sensor emulators. attach_to_i2c (def_attach_to_i2c) makes the emulator keep the
    // sensor alive while it is attached; the C++ controller only stores a raw pointer.
    // i2c_transaction is deliberately NOT re-bound on the concrete sensors: they inherit the
    // buffer-based binding from hemu.I2CPeripheral.
    void bind_sensors(nb::module_& m)
    {
        nb::module_ sensors_module = m.def_submodule("sensors", "C++ sensor emulators attachable to the HSB emulator's I2C controller");

        nb::enum_<sensors::TestPatternMode>(sensors_module, "TestPatternMode")
            .value("CONSTANT", sensors::TestPatternMode::CONSTANT)
            .value("INCREMENTING", sensors::TestPatternMode::INCREMENTING);

        nb::class_<sensors::TestSensor, I2CPeripheral> test_sensor(sensors_module, "TestSensor");
        def_attach_to_i2c(test_sensor, "attach the TestSensor to an I2C controller; the emulator keeps it alive while attached");
        test_sensor
            .def(nb::init<>())
            .def("reset", &sensors::TestSensor::reset, "reset the TestSensor")
            .def("is_streaming", &sensors::TestSensor::is_streaming, "check if TestSensor is streaming")
            .def("get_frame_size", &sensors::TestSensor::get_frame_size, "get the frame size in bytes")
            .def("get_frame_rate_hz", &sensors::TestSensor::get_frame_rate_hz, "get the frame rate in Hz")
            .def("get_pattern_mode", &sensors::TestSensor::get_pattern_mode, "get the pattern mode")
            .def("get_constant_byte", &sensors::TestSensor::get_constant_byte, "get the constant pattern byte");
        sensors_module.attr("TEST_I2C_ADDRESS") = sensors::TEST_I2C_ADDRESS;

        nb::class_<sensors::Vb1940Emulator, I2CPeripheral> vb1940(sensors_module, "Vb1940Emulator");
        def_attach_to_i2c(vb1940, "attach the Vb1940Emulator to an I2C controller; the emulator keeps it alive while attached");
        vb1940
            .def(nb::init<>())
            .def("reset", &sensors::Vb1940Emulator::reset, "reset the Vb1940Emulator")
            .def("is_streaming", &sensors::Vb1940Emulator::is_streaming, "check if Vb1940Emulator is streaming")
            .def("get_pixel_width", &sensors::Vb1940Emulator::get_pixel_width, "get the pixel width")
            .def("get_pixel_height", &sensors::Vb1940Emulator::get_pixel_height, "get the pixel height")
            .def("get_bytes_per_line", &sensors::Vb1940Emulator::get_bytes_per_line, "get the bytes per line")
            .def("get_image_start_byte", &sensors::Vb1940Emulator::get_image_start_byte, "get the image start byte")
            .def("get_pixel_bits", &sensors::Vb1940Emulator::get_pixel_bits, "get the pixel bits")
            .def("get_csi_length", &sensors::Vb1940Emulator::get_csi_length, "get the CSI length")
            .def(
                "set_eeprom_data",
                [](sensors::Vb1940Emulator& self, nb::handle data) {
                    ByteBuffer buffer;
                    buffer.request(data, false, "set_eeprom_data", "data");
                    self.set_eeprom_data(buffer.ptr(), buffer.size());
                },
                "data"_a,
                "populate the rig calibration EEPROM region served at I2C 0x51 from a 1-D byte buffer");
        vb1940.attr("EEPROM_REGION_BYTES") = sensors::Vb1940Emulator::EEPROM_REGION_BYTES;

        nb::class_<sensors::IMX274Emulator, I2CPeripheral> imx274(sensors_module, "IMX274Emulator");
        def_attach_to_i2c(imx274, "attach the IMX274Emulator to an I2C controller; the emulator keeps it alive while attached");
        imx274
            .def(nb::init<>())
            .def("reset", &sensors::IMX274Emulator::reset, "reset the IMX274Emulator")
            .def("is_streaming", &sensors::IMX274Emulator::is_streaming, "check if IMX274Emulator is streaming")
            .def("get_pixel_width", &sensors::IMX274Emulator::get_pixel_width, "get the pixel width")
            .def("get_pixel_height", &sensors::IMX274Emulator::get_pixel_height, "get the pixel height")
            .def("get_bytes_per_line", &sensors::IMX274Emulator::get_bytes_per_line, "get the bytes per line")
            .def("get_image_start_byte", &sensors::IMX274Emulator::get_image_start_byte, "get the image start byte")
            .def("get_pixel_bits", &sensors::IMX274Emulator::get_pixel_bits, "get the pixel bits")
            .def("get_csi_length", &sensors::IMX274Emulator::get_csi_length, "get the CSI length");
    }

    // The HSB register constants an emulator client needs, straight from hsb_config.hpp.
    // In the full hololink build these also come from hololink._hololink; the emulator-only
    // package's minimal hololink/__init__.py re-exports them from here so the same client
    // and test code runs against both.
    void bind_constants(nb::module_& m)
    {
        m.attr("I2C_CTRL") = static_cast<uint32_t>(hololink::I2C_CTRL);
        m.attr("CAM_I2C_BUS") = static_cast<uint32_t>(hololink::CAM_I2C_BUS);
        m.attr("BL_I2C_BUS") = static_cast<uint32_t>(hololink::BL_I2C_BUS);
        m.attr("I2C_REG_CONTROL") = static_cast<uint32_t>(I2C_REG_CONTROL);
        m.attr("I2C_REG_BUS_EN") = static_cast<uint32_t>(I2C_REG_BUS_EN);
        m.attr("I2C_REG_NUM_BYTES") = static_cast<uint32_t>(I2C_REG_NUM_BYTES);
        m.attr("I2C_REG_CLK_CNT") = static_cast<uint32_t>(I2C_REG_CLK_CNT);
        m.attr("I2C_REG_STATUS") = static_cast<uint32_t>(I2C_REG_STATUS);
        m.attr("I2C_REG_DATA_BUFFER") = static_cast<uint32_t>(I2C_REG_DATA_BUFFER);
        m.attr("I2C_START") = static_cast<uint32_t>(I2C_START);
        m.attr("I2C_BUSY") = static_cast<uint32_t>(I2C_BUSY);
        m.attr("I2C_FSM_ERR") = static_cast<uint32_t>(I2C_FSM_ERR);
        m.attr("I2C_I2C_ERR") = static_cast<uint32_t>(I2C_I2C_ERR);
        m.attr("I2C_DONE") = static_cast<uint32_t>(I2C_DONE);
    }

} // namespace

NB_MODULE(_emulation, m)
{
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    bind_config(m);
    bind_emulator(m);
    bind_i2c(m);
    bind_sensors(m);
    bind_constants(m);
}

} // namespace hololink::emulation
