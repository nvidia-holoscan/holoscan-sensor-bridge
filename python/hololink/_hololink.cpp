// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <cuda.h>

#include <holoscan/python/constructor_handle_capsule.hpp>

#ifdef HOLOLINK_HAS_IMGUI
#include <hololink/common/gui_renderer.hpp>
#endif
#include <hololink/common/tools.hpp>
#include <hololink/core/csi_controller.hpp>
#include <hololink/core/csi_formats.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/deserializer.hpp>
#include <hololink/core/enumerator.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/core/jesd.hpp>
#include <hololink/core/logging.hpp>
#include <hololink/core/logging_internal.hpp>
#include <hololink/core/metadata.hpp>
#include <hololink/core/networking.hpp>
#include <hololink/core/serializer.hpp>
#include <hololink/operators/compute_crc/compute_crc.hpp>
#include <hololink/operators/discard_metadata/discard_metadata.hpp>
#include <hololink/operators/linux_coe_receiver/linux_coe_receiver.hpp>
#include <hololink/operators/linux_receiver/linux_receiver.hpp>
#ifdef HOLOLINK_BUILD_FUSA
#include <hololink/operators/fusa_coe_capture/fusa_coe_capture.hpp>
#endif
#include <hololink/operators/operator_catalog.hpp>
#include <hololink/operators/packed_format_converter/packed_format_converter.hpp>
#include <hololink/sensors/camera/camera_sensor.hpp>
#include <hololink/sensors/camera/imx274/native_imx274_sensor.hpp>
#include <hololink/sensors/camera/vb1940/native_vb1940_sensor.hpp>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

// The result of holoscan::python::export_constructor_handle(): a new reference to a capsule
// (or NotImplemented, for another type), or nullptr with a Python exception set.
nb::object steal_constructor_handle(PyObject* exported)
{
    if (exported == nullptr) {
        throw nb::python_error();
    }
    return nb::steal(exported);
}

// Sets one enumeration metadata value from Python: an int or a str, as C++ stores them.
void set_metadata_value(hololink::Metadata& metadata, const std::string& name, nb::handle value)
{
    if (nb::isinstance<nb::int_>(value)) {
        metadata[name] = nb::cast<int64_t>(value);
    } else if (nb::isinstance<nb::str>(value)) {
        metadata[name] = nb::cast<std::string>(value);
    } else {
        throw nb::type_error("Metadata values must be int or str");
    }
}

// Plain dict shaped to match CsiToBayerOp's `geometry` schema field and FusaCoeCaptureOp's own
// `geometry` construction parameter -- both parse a nested dict the same way (see
// operator_catalog.cpp's parse_geometry() and holoscan-sdk's AssignmentStorage::convert_value),
// so this sidesteps needing a full CsiGeometry class binding. Shared by both geometry-resolution
// paths below (FUSA-alignment-aware and generic).
nb::dict geometry_to_dict(const hololink::csi::CsiGeometry& geometry)
{
    nb::dict out;
    out["start_byte"] = geometry.start_byte;
    out["bytes_per_line"] = geometry.bytes_per_line;
    out["pixel_width"] = geometry.pixel_width;
    out["pixel_height"] = geometry.pixel_height;
    out["pixel_format"] = geometry.pixel_format == hololink::csi::PixelFormat::RAW_8 ? "raw_8"
        : geometry.pixel_format == hololink::csi::PixelFormat::RAW_10                ? "raw_10"
                                                                                     : "raw_12";
    out["trailing_bytes"] = geometry.trailing_bytes;
    return out;
}

#ifdef HOLOLINK_BUILD_FUSA
// Resolve CSI geometry once via FusaCoeCaptureOp's HSB-alignment-aware adapter (see its own
// make_geometry_adapter() doc comment: there's no live FusaCoeCaptureOp instance to query
// before the graph compiles, so this has to run before construction, not through it). Only
// correct for a FusaCoeCaptureOp-fed pipeline -- FUSA's 64-byte HSB line alignment differs from
// the generic 8-byte alignment resolve_generic_geometry_for() below uses, so a RoceReceiverOp/
// LinuxReceiverOp-fed pipeline (no FUSA capture stage) must use that one instead, not this one.
// Templated so both Python-bound camera types (NativeVb1940Sensor, NativeImx274Sensor) share
// the same dict-building logic -- the two nb::def() overloads below just pick the C++ overload
// nanobind resolves from the passed camera's bound type.
template <typename CameraT>
nb::dict resolve_geometry_for(std::shared_ptr<CameraT> camera, bool packetizer_enabled)
{
    hololink::csi::CsiGeometry geometry;
    auto adapter
        = hololink::operators::FusaCoeCaptureOp::make_geometry_adapter(geometry, packetizer_enabled);
    camera->configure_converter(adapter);
    return geometry_to_dict(geometry);
}

nb::dict resolve_geometry(
    std::shared_ptr<hololink::sensors::NativeVb1940Sensor> camera, bool packetizer_enabled)
{
    return resolve_geometry_for(std::move(camera), packetizer_enabled);
}

nb::dict resolve_geometry(
    std::shared_ptr<hololink::sensors::NativeImx274Sensor> camera, bool packetizer_enabled)
{
    return resolve_geometry_for(std::move(camera), packetizer_enabled);
}
#endif // HOLOLINK_BUILD_FUSA

// Resolve CSI geometry via CameraSensor's own plain, generic adapter (8-byte line alignment --
// see CameraSensor::configure_converter(CsiGeometry&) in camera_sensor.cpp) -- what every
// RoceReceiverOp/LinuxReceiverOp-fed pipeline needs (e.g. linux_imx274_player.cpp,
// imx274_player.cpp), as opposed to resolve_geometry() above (FusaCoeCaptureOp-fed pipelines
// only).
template <typename CameraT>
nb::dict resolve_generic_geometry_for(std::shared_ptr<CameraT> camera)
{
    hololink::csi::CsiGeometry geometry;
    camera->configure_converter(geometry);
    return geometry_to_dict(geometry);
}

nb::dict resolve_generic_geometry(std::shared_ptr<hololink::sensors::NativeVb1940Sensor> camera)
{
    return resolve_generic_geometry_for(std::move(camera));
}

nb::dict resolve_generic_geometry(std::shared_ptr<hololink::sensors::NativeImx274Sensor> camera)
{
    return resolve_generic_geometry_for(std::move(camera));
}

// The geometry of a camera's RAW_10 or RAW_12 frames when the HSB's packetizer packs them (a
// DataChannel with get_packetizer_program()'s program): no receiver header, and each line packed
// as PackedFormatConverterOp unpacks it and padded to 64 bytes. These are the HSB's rules, the
// same as FusaCoeCaptureOp::make_geometry_adapter(geometry, true)'s, so they hold for any
// receiver of packed frames (e.g. LinuxCoeReceiverOp), and in builds without FuSa.
class PackedGeometryConverter : public hololink::csi::CsiConverter {
public:
    explicit PackedGeometryConverter(hololink::csi::CsiGeometry& out)
        : out_(out)
    {
    }

    uint32_t receiver_start_byte() override { return 0; }

    uint32_t received_line_bytes(uint32_t line_bytes) override
    {
        return static_cast<uint32_t>(hololink::core::round_up(line_bytes, 64));
    }

    uint32_t transmitted_line_bytes(hololink::csi::PixelFormat pixel_format, uint32_t pixel_width) override
    {
        return hololink::operators::PackedFormatConverterOp::transmitted_line_bytes(
            pixel_format, pixel_width);
    }

    void configure(uint32_t start_byte, uint32_t bytes_per_line, uint32_t pixel_width,
        uint32_t pixel_height, hololink::csi::PixelFormat pixel_format,
        uint32_t trailing_bytes = 0) override
    {
        out_ = { start_byte, bytes_per_line, pixel_width, pixel_height, pixel_format, trailing_bytes };
    }

private:
    hololink::csi::CsiGeometry& out_;
};

template <typename CameraT>
nb::dict resolve_packed_geometry_for(std::shared_ptr<CameraT> camera)
{
    hololink::csi::CsiGeometry geometry;
    camera->configure_converter(std::make_shared<PackedGeometryConverter>(geometry));
    return geometry_to_dict(geometry);
}

nb::dict resolve_packed_geometry(std::shared_ptr<hololink::sensors::NativeVb1940Sensor> camera)
{
    return resolve_packed_geometry_for(std::move(camera));
}

nb::dict resolve_packed_geometry(std::shared_ptr<hololink::sensors::NativeImx274Sensor> camera)
{
    return resolve_packed_geometry_for(std::move(camera));
}

// Same generic (8-byte line alignment) geometry resolution as resolve_generic_geometry_for()
// above, but driven from plain (pixel_format, width, height) scalars instead of a
// nanobind-bound camera object -- for pure-Python sensor drivers with no C++-bound camera type
// (e.g. ECam0M30Tof) that can't be passed through either resolve_generic_geometry() overload.
nb::dict resolve_generic_geometry_from_dimensions(int pixel_format, uint32_t width, uint32_t height)
{
    hololink::csi::CsiGeometry geometry;
    auto adapter = hololink::sensors::CameraSensor::make_generic_geometry_adapter(geometry);
    auto fmt = static_cast<hololink::csi::PixelFormat>(pixel_format);
    const uint32_t start_byte = adapter->receiver_start_byte();
    const uint32_t transmitted = adapter->transmitted_line_bytes(fmt, width);
    const uint32_t received = adapter->received_line_bytes(transmitted);
    adapter->configure(start_byte, received, width, height, fmt);
    return geometry_to_dict(geometry);
}

// Wraps hololink::core::Serializer (a raw-pointer-into-a-caller-owned-buffer writer -- see
// src/hololink/core/serializer.hpp) with an owned nb::bytearray so the backing memory stays
// alive for exactly this wrapper's lifetime, and value-returning methods so Python call sites
// (e.g. python/hololink/sensors/ecam0m30tof/ecam0m30tof.py, li_i2c_expander.py) don't need to
// touch the underlying bool-return/out-param C++ shape directly. buffer_ is declared before
// serializer_ deliberately: serializer_'s constructor takes a raw pointer into buffer_'s own
// storage, so buffer_ must finish constructing first.
class PySerializer {
public:
    explicit PySerializer(nb::bytearray buffer)
        : buffer_(std::move(buffer))
        , serializer_(static_cast<uint8_t*>(buffer_.data()), buffer_.size())
    {
    }
    unsigned length() { return serializer_.length(); }
    bool append_uint8(uint8_t value) { return serializer_.append_uint8(value); }
    bool append_uint16_be(uint16_t value) { return serializer_.append_uint16_be(value); }
    bool append_uint16_le(uint16_t value) { return serializer_.append_uint16_le(value); }
    bool append_uint32_be(uint32_t value) { return serializer_.append_uint32_be(value); }
    bool append_uint32_le(uint32_t value) { return serializer_.append_uint32_le(value); }
    bool append_uint64_be(uint64_t value) { return serializer_.append_uint64_be(value); }
    // Python callers pass bytes (e.g. socket.inet_aton()) or bytearray.
    bool append_buffer(nb::bytes value)
    {
        return serializer_.append_buffer(
            reinterpret_cast<uint8_t*>(const_cast<char*>(value.c_str())), value.size());
    }
    bool append_bytearray(nb::bytearray value)
    {
        return serializer_.append_buffer(static_cast<uint8_t*>(value.data()), value.size());
    }
    // The bytes written so far (the first length() bytes of the buffer), as in the old
    // pybind11 binding; callers pass this to i2c_transaction() and similar.
    nb::bytes data()
    {
        return nb::bytes(static_cast<const char*>(buffer_.data()), serializer_.length());
    }

private:
    nb::bytearray buffer_;
    hololink::core::Serializer serializer_;
};

// Wraps hololink::core::Deserializer the same way PySerializer wraps Serializer above, but
// read-only (nb::bytes, not nb::bytearray) and with each next_*() raising on underflow instead
// of returning a bool, matching the Python call sites' plain value-returning usage.
class PyDeserializer {
public:
    explicit PyDeserializer(nb::bytes buffer)
        : buffer_(std::move(buffer))
        , deserializer_(reinterpret_cast<const uint8_t*>(buffer_.c_str()), buffer_.size())
    {
    }
    // A bytearray is copied into bytes, so later changes to it cannot affect the reader.
    explicit PyDeserializer(nb::bytearray buffer)
        : PyDeserializer(nb::bytes(static_cast<const char*>(buffer.data()), buffer.size()))
    {
    }
    // The old binding's Deserializer(buffer, length): read only the first `length` bytes.
    PyDeserializer(nb::bytes buffer, size_t length)
        : PyDeserializer(nb::bytes(buffer.c_str(), checked_length(buffer.size(), length)))
    {
    }
    PyDeserializer(nb::bytearray buffer, size_t length)
        : PyDeserializer(nb::bytes(
            static_cast<const char*>(buffer.data()), checked_length(buffer.size(), length)))
    {
    }
    static size_t checked_length(size_t size, size_t length)
    {
        if (length > size) {
            throw std::invalid_argument("Deserializer length is larger than the buffer");
        }
        return length;
    }
    uint8_t next_uint8()
    {
        uint8_t value = 0;
        if (!deserializer_.next_uint8(value)) {
            throw std::runtime_error("Buffer underflow");
        }
        return value;
    }
    uint16_t next_uint16_be()
    {
        uint16_t value = 0;
        if (!deserializer_.next_uint16_be(value)) {
            throw std::runtime_error("Buffer underflow");
        }
        return value;
    }
    uint16_t next_uint16_le()
    {
        uint16_t value = 0;
        if (!deserializer_.next_uint16_le(value)) {
            throw std::runtime_error("Buffer underflow");
        }
        return value;
    }
    uint32_t next_uint32_be()
    {
        uint32_t value = 0;
        if (!deserializer_.next_uint32_be(value)) {
            throw std::runtime_error("Buffer underflow");
        }
        return value;
    }
    uint32_t next_uint32_le()
    {
        uint32_t value = 0;
        if (!deserializer_.next_uint32_le(value)) {
            throw std::runtime_error("Buffer underflow");
        }
        return value;
    }
    uint64_t next_uint64_be()
    {
        uint64_t value = 0;
        if (!deserializer_.next_uint64_be(value)) {
            throw std::runtime_error("Buffer underflow");
        }
        return value;
    }
    uint64_t next_uint64_le()
    {
        uint64_t value = 0;
        if (!deserializer_.next_uint64_le(value)) {
            throw std::runtime_error("Buffer underflow");
        }
        return value;
    }
    uint64_t next_uint48_be()
    {
        uint64_t value = 0;
        if (!deserializer_.next_uint48_be(value)) {
            throw std::runtime_error("Buffer underflow");
        }
        return value;
    }
    // Returns a copy of the next n bytes (the old pybind11 binding returned a memoryview).
    nb::bytes next_buffer(unsigned n)
    {
        const uint8_t* pointer = nullptr;
        if (!deserializer_.pointer(pointer, n)) {
            throw std::runtime_error("Buffer underflow");
        }
        return nb::bytes(reinterpret_cast<const char*>(pointer), n);
    }
    size_t position() { return deserializer_.position(); }
    size_t remaining() { return deserializer_.remaining(); }
    bool reset(size_t position) { return deserializer_.reset(position); }

private:
    nb::bytes buffer_;
    hololink::core::Deserializer deserializer_;
};

// Trampoline: lets a Python subclass override AsyncEventListener's virtual event callbacks
// (see hololink::Hololink::AsyncEventListener's own doc comment). NB_TRAMPOLINE's slot count
// (6) must match the number of NB_OVERRIDE_* calls below, one per virtual method. These
// callbacks run on Hololink's internal async-event reader thread (not a thread Python created),
// so nanobind must attach that thread to the interpreter before calling back into Python --
// NB_OVERRIDE's call path does this automatically (falls back to PyGILState_Ensure() for a
// foreign thread), the same mechanism CPython itself sanctions for this exact situation.
struct PyAsyncEventListener : hololink::Hololink::AsyncEventListener {
    NB_TRAMPOLINE(hololink::Hololink::AsyncEventListener, 6);

    void on_gpio0_event(uint32_t state, uint64_t timestamp_s, uint32_t timestamp_ns) override
    {
        NB_OVERRIDE(on_gpio0_event, state, timestamp_s, timestamp_ns);
    }
    void on_gpio1_event(uint32_t state, uint64_t timestamp_s, uint32_t timestamp_ns) override
    {
        NB_OVERRIDE(on_gpio1_event, state, timestamp_s, timestamp_ns);
    }
    void on_sif0_frame_start_event(uint32_t state, uint64_t timestamp_s, uint32_t timestamp_ns) override
    {
        NB_OVERRIDE(on_sif0_frame_start_event, state, timestamp_s, timestamp_ns);
    }
    void on_sif0_frame_end_event(uint32_t state, uint64_t timestamp_s, uint32_t timestamp_ns) override
    {
        NB_OVERRIDE(on_sif0_frame_end_event, state, timestamp_s, timestamp_ns);
    }
    void on_sif1_frame_start_event(uint32_t state, uint64_t timestamp_s, uint32_t timestamp_ns) override
    {
        NB_OVERRIDE(on_sif1_frame_start_event, state, timestamp_s, timestamp_ns);
    }
    void on_sif1_frame_end_event(uint32_t state, uint64_t timestamp_s, uint32_t timestamp_ns) override
    {
        NB_OVERRIDE(on_sif1_frame_end_event, state, timestamp_s, timestamp_ns);
    }
};

// Trampoline: lets a Python class subclass Synchronizable (e.g. Vb1940Cam, so a Synchronizer
// can treat a camera driver as one of its attachable peers -- see Synchronizer::attach()).
// Synchronizable itself declares no virtual methods beyond its destructor (it's a marker/tag
// base, not an interface with behavior to override), so there are no NB_OVERRIDE calls to add
// here -- unlike PyAsyncEventListener above, this trampoline exists purely so nanobind allows
// Python subclasses of this C++ type at all, matching internal's own old pybind11
// PySynchronizable trampoline (python/hololink/hololink.cpp), which had the same empty body for
// the same reason.
struct PySynchronizable : hololink::Synchronizable {
    NB_TRAMPOLINE(hololink::Synchronizable, 0);
};

// The bytes of a bytes argument, for the overloads that take write_bytes as bytes: the Python
// sensor drivers pass Serializer.data(), and nanobind's std::vector<uint8_t> caster takes a list,
// tuple or bytearray of int but not bytes. Anything else goes on to the next overload, so that a
// sequence of ints that needs converting (e.g. numpy.uint8) still reaches the vector one.
std::vector<uint8_t> byte_vector(nb::handle buffer)
{
    if (!nb::isinstance<nb::bytes>(buffer)) {
        throw nb::next_overload();
    }
    auto bytes = nb::borrow<nb::bytes>(buffer);
    const auto* begin = reinterpret_cast<const uint8_t*>(bytes.c_str());
    return std::vector<uint8_t>(begin, begin + bytes.size());
}

// HololinkDevice.on_reset(method): the Hololink keeps a ResetController for its lifetime and
// calls each one after reset(). As v2.7's pybind11 binding did, it takes a bound method and
// keeps a weak reference to the method's object, so a camera that registers its own method can
// still be collected; then nothing is called. Unlike v2.7, it keeps the method's function weakly
// too (the object's class keeps it alive): the Hololink outlives Python's exit, and a function
// would keep its module, and whatever that refers to, alive past it.
class BoundResetController : public hololink::Hololink::ResetController {
public:
    explicit BoundResetController(nb::handle method)
    {
        if (!nb::hasattr(method, "__self__") || !nb::hasattr(method, "__func__")) {
            throw nb::type_error("on_reset() takes a bound method");
        }
        const nb::object self = method.attr("__self__");
        const nb::object function = method.attr("__func__");
        self_ = nb::weakref(self);
        function_ = nb::weakref(function);
    }

    ~BoundResetController() override
    {
        // Dropping the references needs the interpreter, and the Hololink can go on any thread,
        // also while Python finalizes; then they are left alone.
        if (nb::detail::cleanup_guard guard {}) {
            self_.reset();
            function_.reset();
        } else {
            self_.release();
            function_.release();
        }
    }

    void reset() override
    {
        nb::gil_scoped_acquire gil;
        const nb::object self = self_();
        const nb::object function = function_();
        if (!self.is_none() && !function.is_none()) {
            function(self);
        }
    }

private:
    // Weakrefs to the method's object and function.
    nb::object self_;
    nb::object function_;
};

} // namespace

NB_MODULE(_hololink, module)
{
    auto registered = hololink::operators::register_hololink_operator_catalog();
    if (!registered) {
        throw holoscan::RuntimeError { registered.error() };
    }
    module.doc() = "hololink core, sensor, and operator-catalog bindings.";

    nb::class_<hololink::Metadata>(module, "Metadata")
        // Empty-construction wasn't previously exposed -- every existing app only ever receives
        // a Metadata as find_channel()'s return value. BasicEnumerationStrategy's constructor
        // below needs a real (possibly empty) Metadata to pass as additional_metadata, matching
        // internal's old pybind11 binding, which bound the same no-arg py::init().
        .def(nb::init<>())
        // A copy, e.g. to select a different sensor of the same board with use_sensor().
        .def(nb::init<const hololink::Metadata&>(), "source"_a)
        // From a dict of int and str values, as internal's binding did: the board tools' metadata
        // for a board that doesn't enumerate (--force).
        .def(
            "__init__",
            [](hololink::Metadata* self, nb::dict values) {
                hololink::Metadata metadata;
                for (auto [name, value] : values) {
                    set_metadata_value(metadata, nb::cast<std::string>(name), value);
                }
                new (self) hololink::Metadata(std::move(metadata));
            },
            "values"_a)
        .def("__repr__",
            [](const hololink::Metadata& self) {
                nb::dict values;
                for (const auto& [name, element] : self) {
                    values[name.c_str()] = nb::cast(element);
                }
                return "Metadata(" + std::string(nb::repr(values).c_str()) + ")";
            })
        // An int, str or list of byte values; KeyError when absent, as in internal's binding.
        .def(
            "__getitem__",
            [](const hololink::Metadata& self, const std::string& name) {
                const auto element = self.get<hololink::Metadata::Element>(name);
                if (!element) {
                    throw nb::key_error(name.c_str());
                }
                return hololink::Metadata::Element { *element };
            },
            "name"_a)
        // `name in metadata` and metadata.get(name, default), as in internal's binding. Without
        // __contains__, `in` would fall back to __getitem__(0) and raise a TypeError.
        .def(
            "__contains__",
            [](const hololink::Metadata& self, const std::string& name) {
                return self.get<hololink::Metadata::Element>(name).has_value();
            },
            "name"_a)
        .def(
            "get",
            [](const hololink::Metadata& self, const std::string& name, nb::object default_value) {
                const auto element = self.get<hololink::Metadata::Element>(name);
                if (!element) {
                    return default_value;
                }
                return nb::cast(hololink::Metadata::Element { *element });
            },
            "name"_a, "default"_a = nb::none())
        // Builds enumeration metadata by hand, e.g. for a DataChannel in a hardware-free test;
        // mirrors hololink_module.EnumerationMetadata's own __setitem__.
        .def(
            "__setitem__",
            [](hololink::Metadata& self, const std::string& name, nb::object value) {
                set_metadata_value(self, name, value);
            },
            "name"_a, "value"_a)
        .def(
            "get_string", [](const hololink::Metadata& self, const std::string& name) -> std::optional<std::string> {
                // Metadata::get<T>() returns std::optional<const T> -- nanobind's optional caster
                // can't convert a const-qualified value type, so strip the const here.
                auto value = self.get<std::string>(name);
                if (!value) {
                    return std::nullopt;
                }
                return std::string { *value };
            },
            "name"_a)
        .def(
            "get_int", [](const hololink::Metadata& self, const std::string& name) -> std::optional<int64_t> {
                auto value = self.get<int64_t>(name);
                if (!value) {
                    return std::nullopt;
                }
                return *value;
            },
            "name"_a);

    // Timeout(timeout_s, retry_s=0.0), for the calls below that take a timeout; None there means
    // the C++ default.
    nb::class_<hololink::Timeout>(module, "Timeout")
        .def(nb::init<float, float>(), "timeout_s"_a, "retry_s"_a = 0.f);

    // Waits for the board's announcement with the GIL released; RuntimeError when it doesn't
    // come within the timeout (10 s by default).
    const auto find_channel = [](const std::string& channel_ip,
                                  std::shared_ptr<hololink::Timeout> timeout) {
        nb::gil_scoped_release release;
        return timeout ? hololink::Enumerator::find_channel(channel_ip, timeout)
                       : hololink::Enumerator::find_channel(channel_ip);
    };
    module.def("find_channel", find_channel, "channel_ip"_a, "timeout"_a.none() = nb::none(),
        "Enumerate the network for a Hololink board at the given IP address.");

    // internal's Enumerator statics, which the board tools (python/tools) use. enumerated() calls
    // call_back(metadata) for each board announcement until it returns False or the timeout
    // passes (None: never); it waits with the GIL released and takes it for each call.
    nb::class_<hololink::Enumerator>(module, "Enumerator")
        .def_static("find_channel", find_channel, "channel_ip"_a, "timeout"_a.none() = nb::none())
        .def_static(
            "enumerated",
            [](nb::callable call_back, std::shared_ptr<hololink::Timeout> timeout) {
                nb::gil_scoped_release release;
                hololink::Enumerator::enumerated(
                    [&call_back](hololink::Metadata& metadata) {
                        nb::gil_scoped_acquire acquire;
                        return nb::cast<bool>(call_back(metadata));
                    },
                    timeout);
            },
            "call_back"_a, "timeout"_a.none() = nb::none());

    // EnumerationStrategy/BasicEnumerationStrategy: per-UUID override for the sensor/dataplane/
    // SIF topology find_channel() assumes while enumerating. Needed only by boards whose
    // enumeration metadata doesn't match any strategy Enumerator already knows by UUID (e.g. a
    // non-camera data-generator board used for GpuRoceTransceiverOp loopback testing). Bound as a
    // real class (not flattened to a module function like find_channel/use_sensor above) because
    // set_uuid_strategy needs a live object to register, not a scalar/string argument. Only the
    // single real C++ constructor is bound (additional_metadata required, not defaulted) --
    // internal's old pybind11 binding additionally exposed a no-metadata convenience overload
    // that just forward to this one with an empty Metadata{}; callers get the identical result by
    // passing Metadata() explicitly instead.
    nb::class_<hololink::EnumerationStrategy>(module, "EnumerationStrategy");
    nb::class_<hololink::BasicEnumerationStrategy, hololink::EnumerationStrategy>(
        module, "BasicEnumerationStrategy")
        .def(nb::init<const hololink::Metadata&, unsigned, unsigned, unsigned>(),
            "additional_metadata"_a, "total_sensors"_a = 2, "total_dataplanes"_a = 2,
            "sifs_per_sensor"_a = 2);

    // Enumerator's registry is never destroyed: holding the Python object itself (nanobind's
    // shared_ptr keeps a reference) would keep it past the interpreter, and nanobind would report
    // it leaked at every exit. So a BasicEnumerationStrategy, the only kind Python can create, is
    // registered as a C++ copy, and a strategy of another type, which only an earlier call can
    // have returned, as is.
    module.def(
        "set_uuid_strategy",
        [](std::string uuid, const hololink::BasicEnumerationStrategy& enumeration_strategy) {
            return hololink::Enumerator::set_uuid_strategy(std::move(uuid),
                std::make_shared<hololink::BasicEnumerationStrategy>(enumeration_strategy));
        },
        "uuid"_a, "enumeration_strategy"_a,
        "Register enumeration_strategy as the topology override for boards enumerating with the "
        "given UUID; returns the strategy it replaces, if any.");
    module.def("set_uuid_strategy", &hololink::Enumerator::set_uuid_strategy, "uuid"_a,
        "enumeration_strategy"_a);

    module.def("use_sensor", &hololink::DataChannel::use_sensor, "metadata"_a, "sensor_number"_a);

    module.def("use_mtu", &hololink::DataChannel::use_mtu, "metadata"_a, "mtu"_a,
        "Stamp the data-plane MTU into enumeration metadata so it's applied during "
        "configure_roce()/configure_coe(); use 9000+ for jumbo frames.");

    module.def("use_vlan", &hololink::DataChannel::use_vlan, "metadata"_a, "vlan_id"_a,
        "sensor_pcp"_a, "evt_pcp"_a,
        "Stamp 802.1Q VLAN parameters into enumeration metadata so they're applied "
        "automatically during configure_roce()/configure_coe(). vlan_id is a 12-bit VLAN ID "
        "in [1, 4094]; sensor_pcp/evt_pcp are 3-bit priority code points for the sensor data "
        "virtual port and the EVT channel respectively.");

    module.def("infiniband_devices", &hololink::infiniband_devices,
        "Sorted list of available Infiniband (RoCE) device names, e.g. for RoceReceiverOp's "
        "ibv_name parameter. Empty if this build has no ibverbs support.");

    module.def(
        "infiniband_device_for_peer",
        [](const std::string& peer_ip, const std::string& interface) {
            return hololink::core::infiniband_device_for_peer(peer_ip, interface);
        },
        "peer_ip"_a, "interface"_a = std::string(),
        "(ibv_name, ibv_port) of the Infiniband (RoCE) device that reaches the board at peer_ip: "
        "the device on the network interface its DataChannel uses -- the one it enumerated on "
        "(its enumeration metadata's \"interface\"), or the routing table's when that one has no "
        "IPv4 address. RuntimeError if no device is bound to that interface.");

    // The SDK's own (C++) log messages don't go through Python logging. get/set_hsb_log_level()
    // read and set the level they need to be shown, one of HSB_LOG_LEVEL_*: 10 trace, 20 debug,
    // 30 info, 40 warn, 50 error. The first message applies HOLOLINK_LOG_LEVEL, or else
    // HOLOSCAN_LOG_LEVEL, over a level set before it; an unrecognized value makes logging throw.
    module.attr("HSB_LOG_LEVEL_INVALID") = static_cast<int>(hololink::logging::HSB_LOG_LEVEL_INVALID);
    module.attr("HSB_LOG_LEVEL_TRACE") = static_cast<int>(hololink::logging::HSB_LOG_LEVEL_TRACE);
    module.attr("HSB_LOG_LEVEL_DEBUG") = static_cast<int>(hololink::logging::HSB_LOG_LEVEL_DEBUG);
    module.attr("HSB_LOG_LEVEL_INFO") = static_cast<int>(hololink::logging::HSB_LOG_LEVEL_INFO);
    module.attr("HSB_LOG_LEVEL_WARN") = static_cast<int>(hololink::logging::HSB_LOG_LEVEL_WARN);
    module.attr("HSB_LOG_LEVEL_ERROR") = static_cast<int>(hololink::logging::HSB_LOG_LEVEL_ERROR);
    module.def("get_hsb_log_level", [] { return static_cast<int>(hololink::logging::hsb_log_level); });
    module.def(
        "set_hsb_log_level",
        [](int level) {
            hololink::logging::hsb_log_level = static_cast<hololink::logging::HsbLogLevel>(level);
        },
        "level"_a);
    // Log through the SDK's logger, in its format and at its level, as internal's binding did.
    // hololink_module.hsb_log_*() use the module logger instead, which prints nothing until the
    // Adapter exists. The message is a format argument, so braces in it print as they are.
    module.def(
        "hsb_log_trace", [](const std::string& message) { HSB_LOG_TRACE("{}", message); }, "message"_a);
    module.def(
        "hsb_log_debug", [](const std::string& message) { HSB_LOG_DEBUG("{}", message); }, "message"_a);
    module.def(
        "hsb_log_info", [](const std::string& message) { HSB_LOG_INFO("{}", message); }, "message"_a);
    module.def(
        "hsb_log_warn", [](const std::string& message) { HSB_LOG_WARN("{}", message); }, "message"_a);
    module.def(
        "hsb_log_error", [](const std::string& message) { HSB_LOG_ERROR("{}", message); }, "message"_a);
    // The clock hololink's (C++ core) log messages are stamped with, in seconds; Python log lines
    // that use it (tests/conftest.py, pytest.ini's log_format) line up with those.
    // hololink_module's logger keeps its own clock.
    module.def("log_timestamp_s", &hololink::logging::log_timestamp_s);

    // Bus-selector/address constants for HololinkDevice.get_i2c()/get_spi() below -- e.g.
    // hololink.CAM_I2C_BUS for a camera's own I2C bus. hololink namespace-scope constexprs
    // (hololink.hpp), not nested HololinkDevice members, so these bind as plain module
    // attributes rather than class-level ones.
    module.attr("SPI_CTRL") = hololink::SPI_CTRL;
    module.attr("RESET_SPI_BUS") = hololink::RESET_SPI_BUS;
    module.attr("CLNX_SPI_BUS") = hololink::CLNX_SPI_BUS;
    module.attr("CPNX_SPI_BUS") = hololink::CPNX_SPI_BUS;
    module.attr("BL_I2C_BUS") = hololink::BL_I2C_BUS;
    module.attr("CAM_I2C_BUS") = hololink::CAM_I2C_BUS;
    module.attr("I2C_CTRL") = hololink::I2C_CTRL;
    // Registers and flags that the board tools (python/tools) read for their reports.
    module.attr("HSB_IP_VERSION") = hololink::HSB_IP_VERSION;
    module.attr("FPGA_DATE") = hololink::FPGA_DATE;
    module.attr("CTRL_EVENT") = hololink::CTRL_EVENT;
    module.attr("RD_DWORD") = hololink::RD_DWORD;
    module.attr("REQUEST_FLAGS_ACK_REQUEST") = hololink::REQUEST_FLAGS_ACK_REQUEST;
    module.attr("I2C_REG_STATUS") = hololink::I2C_REG_STATUS;
    module.attr("I2C_BUSY") = hololink::I2C_BUSY;
    module.attr("I2C_FSM_ERR") = hololink::I2C_FSM_ERR;
    module.attr("I2C_I2C_ERR") = hololink::I2C_I2C_ERR;
    module.attr("I2C_I2C_NAK") = hololink::I2C_I2C_NAK;
    module.attr("I2C_DONE") = hololink::I2C_DONE;
    // The I2C controller register map and START bit, for driving a controller by register
    // (the emulator tests do this against hololink.emulation.HSBEmulator.write/read).
    module.attr("I2C_REG_CONTROL") = hololink::I2C_REG_CONTROL;
    module.attr("I2C_REG_BUS_EN") = hololink::I2C_REG_BUS_EN;
    module.attr("I2C_REG_NUM_BYTES") = hololink::I2C_REG_NUM_BYTES;
    module.attr("I2C_REG_CLK_CNT") = hololink::I2C_REG_CLK_CNT;
    module.attr("I2C_REG_DATA_BUFFER") = hololink::I2C_REG_DATA_BUFFER;
    module.attr("I2C_START") = hololink::I2C_START;
    // Receive-buffer geometry: frames are page aligned and followed by METADATA_SIZE bytes
    // of per-frame metadata (see ReceiverMemoryDescriptor and the receivers below).
    module.attr("PAGE_SIZE") = hololink::core::PAGE_SIZE;
    module.attr("METADATA_SIZE") = hololink::METADATA_SIZE;

    // nb::is_arithmetic() makes these real IntEnum members (int(value) works directly, matching
    // how every pixel_format/bayer_format value elsewhere in this package is already a plain
    // int/string, not a bare enum.Enum that can't convert) -- without it, nanobind's default
    // enum factory is plain enum.Enum, which int() rejects.
    nb::enum_<hololink::csi::PixelFormat>(module, "PixelFormat", nb::is_arithmetic())
        .value("RAW_8", hololink::csi::PixelFormat::RAW_8)
        .value("RAW_10", hololink::csi::PixelFormat::RAW_10)
        .value("RAW_12", hololink::csi::PixelFormat::RAW_12);

    nb::enum_<hololink::csi::BayerFormat>(module, "BayerFormat", nb::is_arithmetic())
        .value("BGGR", hololink::csi::BayerFormat::BGGR)
        .value("RGGB", hololink::csi::BayerFormat::RGGB)
        .value("GBRG", hololink::csi::BayerFormat::GBRG)
        .value("GRBG", hololink::csi::BayerFormat::GRBG);

    // Opaque handle (no Python-visible methods): produced by Synchronizer::null_synchronizer()
    // or HololinkDevice.ptp_pps_output(), consumed by NativeVb1940Sensor's vsync constructor
    // argument. nanobind's shared_ptr<T> auto-detection (see the <nanobind/stl/shared_ptr.h>
    // include) is all this needs -- Python code never calls a method on it directly.
    nb::class_<hololink::Synchronizer>(module, "Synchronizer")
        .def_static("null_synchronizer", &hololink::Synchronizer::null_synchronizer);

    // Produced by HololinkDevice.{sif0,sif1}_frame_{start,end}_sequencer() and
    // HololinkDevice.{gpio0,gpio1}_sequencer() below: a small
    // program of register writes, loaded into HSB and armed by enable() so it fires on a
    // specific FPGA hardware event (e.g. "drive this GPIO pin high when a frame-start trigger
    // happens") without needing a host round-trip at trigger time. Only the two methods this
    // repo's ported apps actually call are bound (read_uint32()/poll() exist in C++ but have no
    // current caller) -- matches this effort's minimal-binding convention.
    nb::class_<hololink::Hololink::Sequencer>(module, "Sequencer")
        .def("write_uint32", &hololink::Hololink::Sequencer::write_uint32, "address"_a, "data"_a)
        .def("enable", &hololink::Hololink::Sequencer::enable);

    // Subclass in Python and override whichever on_*_event() methods you need (defaults are
    // no-ops); register an instance via HololinkDevice.on_async_event(). See
    // PyAsyncEventListener's own comment above for the cross-thread callback mechanism.
    // The virtual callbacks are bound so NB_OVERRIDE can fall back to the C++ no-op base when a
    // Python subclass does not override one; without them every non-overridden event raises
    // "trampoline_enter(...): lookup failed!".
    using AsyncEventListener = hololink::Hololink::AsyncEventListener;
    nb::class_<AsyncEventListener, PyAsyncEventListener>(module, "AsyncEventListener")
        .def(nb::init<>())
        .def("on_gpio0_event", &AsyncEventListener::on_gpio0_event, "state"_a, "timestamp_s"_a, "timestamp_ns"_a)
        .def("on_gpio1_event", &AsyncEventListener::on_gpio1_event, "state"_a, "timestamp_s"_a, "timestamp_ns"_a)
        .def("on_sif0_frame_start_event", &AsyncEventListener::on_sif0_frame_start_event, "state"_a, "timestamp_s"_a, "timestamp_ns"_a)
        .def("on_sif0_frame_end_event", &AsyncEventListener::on_sif0_frame_end_event, "state"_a, "timestamp_s"_a, "timestamp_ns"_a)
        .def("on_sif1_frame_start_event", &AsyncEventListener::on_sif1_frame_start_event, "state"_a, "timestamp_s"_a, "timestamp_ns"_a)
        .def("on_sif1_frame_end_event", &AsyncEventListener::on_sif1_frame_end_event, "state"_a, "timestamp_s"_a, "timestamp_ns"_a);

    // Synchronizable: subclass in Python (e.g. a camera driver) to become attachable to a
    // Synchronizer via Synchronizer::attach(). See PySynchronizable's own comment above.
    nb::class_<hololink::Synchronizable, PySynchronizable>(module, "Synchronizable")
        .def(nb::init<>());

    // Guards a critical section shared by every I2c/Spi transaction on a given bus (see
    // HololinkDevice.i2c_lock() below) -- wraps a non-copyable std::recursive_mutex, so this
    // binds lock()/unlock() only; Python code never constructs one directly, only receives one
    // by reference from i2c_lock()/spi_lock().
    nb::class_<hololink::NamedLock>(module, "NamedLock")
        .def("lock", &hololink::NamedLock::lock)
        .def("unlock", &hololink::NamedLock::unlock);

    // Pure-virtual ABC Python only ever *receives* (from HololinkDevice.get_i2c() below), never
    // subclasses -- no trampoline needed, unlike AsyncEventListener above.
    // A timeout defaults to None: an empty shared_ptr, which means the default timeout in C++.
    nb::class_<hololink::Hololink::I2c>(module, "I2c")
        .def(
            "i2c_transaction",
            [](hololink::Hololink::I2c& self, uint32_t peripheral_i2c_address,
                const std::vector<uint8_t>& write_bytes, uint32_t read_byte_count,
                std::shared_ptr<hololink::Timeout> timeout, bool ignore_nak) {
                std::vector<uint8_t> result;
                {
                    // Scoped to just the blocking hardware I/O call -- nb::bytes(...) below
                    // needs the GIL back.
                    nb::gil_scoped_release release;
                    result = self.i2c_transaction(
                        peripheral_i2c_address, write_bytes, read_byte_count, timeout, ignore_nak);
                }
                return nb::bytes(reinterpret_cast<const char*>(result.data()), result.size());
            },
            "peripheral_i2c_address"_a, "write_bytes"_a, "read_byte_count"_a,
            "timeout"_a.none() = nb::none(), "ignore_nak"_a = false)
        // Same, with write_bytes as bytes (see byte_vector()).
        .def(
            "i2c_transaction",
            [](hololink::Hololink::I2c& self, uint32_t peripheral_i2c_address,
                nb::handle write_bytes, uint32_t read_byte_count,
                std::shared_ptr<hololink::Timeout> timeout, bool ignore_nak) {
                const std::vector<uint8_t> write_vector = byte_vector(write_bytes);
                std::vector<uint8_t> result;
                {
                    nb::gil_scoped_release release;
                    result = self.i2c_transaction(
                        peripheral_i2c_address, write_vector, read_byte_count, timeout, ignore_nak);
                }
                return nb::bytes(reinterpret_cast<const char*>(result.data()), result.size());
            },
            "peripheral_i2c_address"_a, "write_bytes"_a, "read_byte_count"_a,
            "timeout"_a.none() = nb::none(), "ignore_nak"_a = false)
        // Encodes the transaction into sequencer, which runs it when its event fires (for
        // example DataChannel.frame_end_sequencer()), rather than running it now. Returns
        // (write_indexes, read_indexes, status_index): the sequencer's slots for the bytes
        // written, the bytes read and the I2C status.
        .def("encode_i2c_request", &hololink::Hololink::I2c::encode_i2c_request, "sequencer"_a,
            "peripheral_i2c_address"_a, "write_bytes"_a, "read_byte_count"_a)
        // Same, with write_bytes as bytes (see byte_vector()).
        .def(
            "encode_i2c_request",
            [](hololink::Hololink::I2c& self, hololink::Hololink::Sequencer& sequencer,
                uint32_t peripheral_i2c_address, nb::handle write_bytes, uint32_t read_byte_count) {
                return self.encode_i2c_request(
                    sequencer, peripheral_i2c_address, byte_vector(write_bytes), read_byte_count);
            },
            "sequencer"_a, "peripheral_i2c_address"_a, "write_bytes"_a, "read_byte_count"_a);

    // Same reasoning as I2c above. spi_transaction() takes two separate byte buffers (a command
    // buffer and a data buffer), not a peripheral address -- confirmed against the current C++
    // signature directly, not assumed from the old pybind11 binding's kwarg names (which used
    // "peripheral_i2c_address" as a cosmetic label for the first buffer argument, something
    // SPI transactions don't actually have).
    nb::class_<hololink::Hololink::Spi>(module, "Spi")
        .def(
            "spi_transaction",
            [](hololink::Hololink::Spi& self, const std::vector<uint8_t>& write_command_bytes,
                const std::vector<uint8_t>& write_data_bytes, uint32_t read_byte_count,
                std::shared_ptr<hololink::Timeout> timeout) {
                std::vector<uint8_t> result;
                {
                    nb::gil_scoped_release release;
                    result = self.spi_transaction(
                        write_command_bytes, write_data_bytes, read_byte_count, timeout);
                }
                return nb::bytes(reinterpret_cast<const char*>(result.data()), result.size());
            },
            "write_command_bytes"_a, "write_data_bytes"_a, "read_byte_count"_a,
            "timeout"_a.none() = nb::none());

    nb::class_<PySerializer>(module, "Serializer")
        .def(nb::init<nb::bytearray>())
        .def("length", &PySerializer::length)
        .def("append_uint8", &PySerializer::append_uint8, "value"_a)
        .def("append_uint16_be", &PySerializer::append_uint16_be, "value"_a)
        .def("append_uint16_le", &PySerializer::append_uint16_le, "value"_a)
        .def("append_uint32_be", &PySerializer::append_uint32_be, "value"_a)
        .def("append_uint32_le", &PySerializer::append_uint32_le, "value"_a)
        .def("append_uint64_be", &PySerializer::append_uint64_be, "value"_a)
        .def("append_buffer", &PySerializer::append_buffer, "value"_a)
        .def("append_buffer", &PySerializer::append_bytearray, "value"_a)
        .def("data", &PySerializer::data);

    nb::class_<PyDeserializer>(module, "Deserializer")
        .def(nb::init<nb::bytes>())
        .def(nb::init<nb::bytearray>())
        .def(nb::init<nb::bytes, size_t>(), "buffer"_a, "length"_a)
        .def(nb::init<nb::bytearray, size_t>(), "buffer"_a, "length"_a)
        .def("next_uint8", &PyDeserializer::next_uint8)
        .def("next_uint16_be", &PyDeserializer::next_uint16_be)
        .def("next_uint16_le", &PyDeserializer::next_uint16_le)
        .def("next_uint32_be", &PyDeserializer::next_uint32_be)
        .def("next_uint32_le", &PyDeserializer::next_uint32_le)
        .def("next_uint64_be", &PyDeserializer::next_uint64_be)
        .def("next_uint64_le", &PyDeserializer::next_uint64_le)
        .def("next_uint48_be", &PyDeserializer::next_uint48_be)
        .def("next_buffer", &PyDeserializer::next_buffer, "n"_a)
        .def("position", &PyDeserializer::position)
        .def("remaining", &PyDeserializer::remaining)
        .def("reset", &PyDeserializer::reset, "position"_a = 0);

    nb::class_<hololink::Hololink>(module, "HololinkDevice")
        // These release the GIL, as internal's binding did: they can wait on the async event
        // thread, which takes the GIL to call a Python AsyncEventListener.
        .def("start", &hololink::Hololink::start, nb::call_guard<nb::gil_scoped_release>())
        .def("stop", &hololink::Hololink::stop, nb::call_guard<nb::gil_scoped_release>())
        .def("reset", &hololink::Hololink::reset, nb::call_guard<nb::gil_scoped_release>())
        // Drops the boards this module cached (from_enumeration_metadata() keeps one per serial
        // number) and throws if hololink's core reactor still has an alarm or fd callback. Board
        // modules that hololink_module loads keep their own cache and reactor. tests/conftest.py
        // calls it after each test, as internal's binding allowed.
        .def_static("reset_framework", &hololink::Hololink::reset_framework,
            nb::call_guard<nb::gil_scoped_release>())
        // Program the board's clock synthesizer from a register profile such as
        // hololink.renesas_bajoran_lite_ts1.device_configuration(); the pure-Python sensor
        // drivers (imx715, ...) call this from their setup_clock().
        .def(
            "setup_clock",
            [](hololink::Hololink& self, nb::sequence clock_profile) {
                // Each entry is a register write: bytes/bytearray (what
                // renesas_bajoran_lite_ts1.device_configuration() returns) or a sequence of
                // ints. nanobind's Sequence[int] caster deliberately rejects bytes, so
                // convert by hand.
                std::vector<std::vector<uint8_t>> profile;
                for (nb::handle entry : clock_profile) {
                    if (PyBytes_Check(entry.ptr()) || PyByteArray_Check(entry.ptr())) {
                        nb::bytes as_bytes(PyBytes_Check(entry.ptr()) ? nb::borrow<nb::object>(entry) : nb::steal<nb::object>(PyBytes_FromObject(entry.ptr())));
                        const auto* data = reinterpret_cast<const uint8_t*>(as_bytes.c_str());
                        profile.emplace_back(data, data + as_bytes.size());
                    } else {
                        profile.push_back(nb::cast<std::vector<uint8_t>>(entry));
                    }
                }
                nb::gil_scoped_release nogil;
                self.setup_clock(profile);
            },
            "clock_profile"_a,
            "write a clock synthesizer register profile: a sequence of byte strings (or int sequences), one per register write")
        .def("ptp_synchronize", nb::overload_cast<>(&hololink::Hololink::ptp_synchronize),
            "Wait (default 30s timeout) for PTP time to synchronize. Returns false on timeout.")
        .def("ptp_synchronize",
            nb::overload_cast<const std::shared_ptr<hololink::Timeout>&>(
                &hololink::Hololink::ptp_synchronize),
            "timeout"_a, nb::call_guard<nb::gil_scoped_release>(),
            "Wait up to timeout for PTP time to synchronize. Returns false on timeout.")
        .def("configure_ptp", &hololink::Hololink::configure_ptp, "profile"_a, "domain"_a = 0,
            "Configure this board's PTP profile (0=IEEE 1588 E2E, 1=gPTP/IEEE 802.1AS, "
            "2=IEEE 1588 P2P) and domain number before ptp_synchronize().")
        // Register access waits for the board's reply: release the GIL meanwhile, as internal's
        // pybind11 binding did.
        .def("write_uint32",
            nb::overload_cast<uint32_t, uint32_t>(&hololink::Hololink::write_uint32),
            "address"_a, "value"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("read_uint32", nb::overload_cast<uint32_t>(&hololink::Hololink::read_uint32),
            "address"_a, nb::call_guard<nb::gil_scoped_release>(),
            "Read a 32-bit register, using the internal default timeout.")
        .def("ptp_pps_output", &hololink::Hololink::ptp_pps_output, "frequency"_a = 0,
            "Build a Synchronizer driven by this board's PTP PPS output at the given "
            "frequency (Hz), for camera VSYNC triggering.")
        .def("sif0_frame_start_sequencer", &hololink::Hololink::sif0_frame_start_sequencer,
            "Sequencer that fires on the SIF0 frame-start hardware event.")
        .def("sif1_frame_start_sequencer", &hololink::Hololink::sif1_frame_start_sequencer,
            "Sequencer that fires on the SIF1 frame-start hardware event.")
        .def("sif0_frame_end_sequencer", &hololink::Hololink::sif0_frame_end_sequencer,
            "Sequencer that fires on the SIF0 frame-end hardware event.")
        .def("sif1_frame_end_sequencer", &hololink::Hololink::sif1_frame_end_sequencer,
            "Sequencer that fires on the SIF1 frame-end hardware event.")
        .def("gpio0_sequencer", &hololink::Hololink::gpio0_sequencer,
            "Sequencer that fires on a GPIO0 input transition.")
        .def("gpio1_sequencer", &hololink::Hololink::gpio1_sequencer,
            "Sequencer that fires on a GPIO1 input transition.")
        .def("on_async_event", &hololink::Hololink::on_async_event, "listener"_a,
            "Register an AsyncEventListener; its overridden on_*_event() methods are called "
            "from this board's internal async-event reader thread as matching event packets "
            "arrive. Listeners stay registered until remove_async_event() is called.")
        .def("remove_async_event", &hololink::Hololink::remove_async_event, "listener"_a,
            "Unregister a listener added with on_async_event(), so it can be freed before the "
            "process exits. A no-op if it isn't registered.")
        .def(
            "on_reset",
            [](hololink::Hololink& self, nb::handle method) {
                self.on_reset(std::make_shared<BoundResetController>(method));
            },
            "method"_a,
            "Call the bound method after each reset(), unless its object was collected. "
            "Methods accumulate for this object's lifetime -- there's no unregister.")
        .def("get_i2c", &hololink::Hololink::get_i2c, "i2c_bus"_a,
            "i2c_address"_a = hololink::I2C_CTRL)
        .def("get_spi", &hololink::Hololink::get_spi, "bus_number"_a, "chip_select"_a,
            "clock_divisor"_a = 0x0Fu, "cpol"_a = 1u, "cpha"_a = 1u, "width"_a = 1u,
            "spi_address"_a = hololink::SPI_CTRL)
        .def("i2c_lock", &hololink::Hololink::i2c_lock, nb::rv_policy::reference_internal)
        .def(
            "get_i2s",
            [](hololink::Hololink& self, hololink::Metadata& enumeration_metadata,
                uint32_t mclk_div, uint32_t bclk_div_sel, bool lrclk_polarity, bool enable_tx,
                bool enable_rx, bool enable_loopback, uint32_t tx_almost_full,
                uint32_t tx_almost_empty, bool pause_after, uint32_t slot_mode) {
                // The returned I2s handle isn't bound to Python -- every caller today uses
                // get_i2s() purely for its configuration side effect, never the handle itself.
                // bit_depth/format/channel_mode are hardcoded to their real C++ defaults rather
                // than exposed -- no caller overrides them.
                self.get_i2s(enumeration_metadata, mclk_div, bclk_div_sel,
                    hololink::Hololink::I2s::BitDepth::BD_24,
                    hololink::Hololink::I2s::DataFormat::I2S,
                    hololink::Hololink::I2s::ChannelMode::Stereo, lrclk_polarity, enable_tx,
                    enable_rx, enable_loopback, tx_almost_full, tx_almost_empty, pause_after,
                    slot_mode);
            },
            "enumeration_metadata"_a, "mclk_div"_a = 16, "bclk_div_sel"_a = 3,
            "lrclk_polarity"_a = false, "enable_tx"_a = true, "enable_rx"_a = true,
            "enable_loopback"_a = false, "tx_almost_full"_a = 0x800,
            "tx_almost_empty"_a = 0x400, "pause_after"_a = true, "slot_mode"_a = 1,
            "Configure the I2S peripheral (TX/RX enable, clock dividers, almost-full/empty "
            "watermarks). Matches Hololink::get_i2s()'s own defaults.")
        .def(
            "get_hsb_ip_version",
            [](hololink::Hololink& self, std::shared_ptr<hololink::Timeout> timeout,
                bool check_sequence) { return self.get_hsb_ip_version(timeout, check_sequence); },
            "timeout"_a.none() = nb::none(), "check_sequence"_a = true,
            nb::call_guard<nb::gil_scoped_release>(),
            "HSB IP version register. check_sequence=False doesn't ask the board to check the "
            "request's sequence number, e.g. right after a reset.")
        .def("get_fpga_date", &hololink::Hololink::get_fpga_date)
        .def("get_gpio", &hololink::Hololink::get_gpio, "metadata"_a)
        .def("get_uart", &hololink::Hololink::get_uart, "port_number"_a = 0,
            "baud_rate"_a = 115200, "data_bits"_a = 8, "parity"_a = 0, "stop_bits"_a = 1,
            "flow_control"_a = 0);

    module.def("round_up", &hololink::core::round_up, "value"_a, "alignment"_a,
        "Round value up to the next multiple of alignment.");

    // GPIO: register-level pin control, reached only via HololinkDevice.get_gpio() above -- no
    // public constructor (matches the C++ class itself, which has none either). Mirrors
    // internal's own old pybind11 binding exactly (python/hololink/hololink.cpp, method-for-
    // method and constant-for-constant), just in nanobind's own class-attribute idiom.
    auto gpio_cls = nb::class_<hololink::Hololink::GPIO>(module, "GPIO")
                        .def("set_direction", &hololink::Hololink::GPIO::set_direction, "pin"_a, "direction"_a)
                        .def("get_direction", &hololink::Hololink::GPIO::get_direction, "pin"_a)
                        .def("set_value", &hololink::Hololink::GPIO::set_value, "pin"_a, "value"_a)
                        .def("get_value", &hololink::Hololink::GPIO::get_value, "pin"_a)
                        .def("get_supported_pin_num", &hololink::Hololink::GPIO::get_supported_pin_num);
    gpio_cls.attr("IN") = hololink::Hololink::GPIO::IN;
    gpio_cls.attr("OUT") = hololink::Hololink::GPIO::OUT;
    gpio_cls.attr("LOW") = hololink::Hololink::GPIO::LOW;
    gpio_cls.attr("HIGH") = hololink::Hololink::GPIO::HIGH;
    gpio_cls.attr("GPIO_PIN_RANGE") = hololink::Hololink::GPIO::GPIO_PIN_RANGE;

    // Uart: reached only via HololinkDevice.get_uart() above. uart_read()'s out-param
    // (bytes_read) becomes a (data, bytes_read) tuple return, matching internal's own lambda
    // wrapper -- nanobind has no out-param support any more than pybind11 did.
    nb::class_<hololink::Hololink::Uart>(module, "Uart")
        .def("uart_write", &hololink::Hololink::Uart::uart_write, "data"_a)
        .def(
            "uart_read",
            [](hololink::Hololink::Uart& self, size_t bytes_num) {
                uint32_t bytes_read = 0;
                std::vector<uint8_t> result = self.uart_read(bytes_num, bytes_read);
                return std::make_tuple(std::move(result), bytes_read);
            },
            "bytes_num"_a)
        .def("uart_disable", &hololink::Hololink::Uart::uart_disable)
        .def("uart_enable", &hololink::Hololink::Uart::uart_enable)
        .def("uart_tx_disable", &hololink::Hololink::Uart::uart_tx_disable)
        .def("uart_tx_enable", &hololink::Hololink::Uart::uart_tx_enable)
        .def("uart_rx_disable", &hololink::Hololink::Uart::uart_rx_disable)
        .def("uart_rx_enable", &hololink::Hololink::Uart::uart_rx_enable)
        .def("uart_configure", &hololink::Hololink::Uart::uart_configure, "baud_rate"_a,
            "data_bits"_a, "parity"_a, "stop_bits"_a, "flow_control"_a = 0,
            "tx_almost_empty_threshold"_a = 4, "tx_almost_full_threshold"_a = 252,
            "rx_almost_empty_threshold"_a = 4, "rx_almost_full_threshold"_a = 252)
        .def("uart_set_baud_rate", &hololink::Hololink::Uart::uart_set_baud_rate, "baud_rate"_a)
        .def("uart_set_data_bits", &hololink::Hololink::Uart::uart_set_data_bits, "data_bits"_a)
        .def("uart_set_parity", &hololink::Hololink::Uart::uart_set_parity, "parity"_a)
        .def("uart_set_stop_bits", &hololink::Hololink::Uart::uart_set_stop_bits, "stop_bits"_a)
        .def("uart_set_flow_control", &hololink::Hololink::Uart::uart_set_flow_control,
            "flow_control"_a)
        .def("uart_set_tx_fifo_threshold", &hololink::Hololink::Uart::uart_set_tx_fifo_threshold,
            "tx_almost_empty_threshold"_a, "tx_almost_full_threshold"_a)
        .def("uart_set_rx_fifo_threshold", &hololink::Hololink::Uart::uart_set_rx_fifo_threshold,
            "rx_almost_empty_threshold"_a, "rx_almost_full_threshold"_a)
        .def("uart_dump_registers", &hololink::Hololink::Uart::uart_dump_registers)
        .def("uart_enable_internal_loopback",
            &hololink::Hololink::Uart::uart_enable_internal_loopback)
        .def("uart_disable_internal_loopback",
            &hololink::Hololink::Uart::uart_disable_internal_loopback)
        .def("uart_clear_sticky_bits", &hololink::Hololink::Uart::uart_clear_sticky_bits,
            "bitmask"_a)
        .def("baud_rate", &hololink::Hololink::Uart::baud_rate)
        .def("data_bits", &hololink::Hololink::Uart::data_bits)
        .def("parity", &hololink::Hololink::Uart::parity)
        .def("stop_bits", &hololink::Hololink::Uart::stop_bits)
        .def("flow_control", &hololink::Hololink::Uart::flow_control)
        .def("tx_almost_empty_threshold", &hololink::Hololink::Uart::tx_almost_empty_threshold)
        .def("tx_almost_full_threshold", &hololink::Hololink::Uart::tx_almost_full_threshold)
        .def("rx_almost_empty_threshold", &hololink::Hololink::Uart::rx_almost_empty_threshold)
        .def("rx_almost_full_threshold", &hololink::Hololink::Uart::rx_almost_full_threshold);

    // How the HSB's packetizer repacks a sensor's CSI data before sending it; a DataChannel
    // enables its program when the receiver configures it (the default disables the
    // packetizer). FusaCoeCaptureOp sets its own; set one for another receiver with
    // DataChannel.set_packetizer_program().
    nb::class_<hololink::PacketizerProgram>(module, "PacketizerProgram")
        .def("packetizes", &hololink::PacketizerProgram::packetizes,
            "Whether enabling this program turns the packetizer on (False for "
            "NullPacketizerProgram).")
        .def("get_output_size", &hololink::PacketizerProgram::get_output_size, "input_size"_a,
            "The size of the line the packetizer's program makes of input_size bytes of CSI "
            "data, before the HSB pads it to 64 bytes.")
        .def("padded_output_size", &hololink::PacketizerProgram::padded_output_size,
            "input_size"_a,
            "The size of the line that the HSB sends for input_size bytes of CSI data: "
            "get_output_size() padded to 64 bytes.");
    // The programs get_packetizer_program() returns, so that one shows which it is.
    nb::class_<hololink::NullPacketizerProgram, hololink::PacketizerProgram>(
        module, "NullPacketizerProgram");
    nb::class_<hololink::Csi10ToPacked10, hololink::PacketizerProgram>(module, "Csi10ToPacked10");
    nb::class_<hololink::Csi12ToPacked12, hololink::PacketizerProgram>(module, "Csi12ToPacked12");
    module.def(
        "get_packetizer_program",
        [](int pixel_format) {
            const auto format = static_cast<hololink::csi::PixelFormat>(pixel_format);
            switch (format) {
            case hololink::csi::PixelFormat::RAW_8:
            case hololink::csi::PixelFormat::RAW_10:
            case hololink::csi::PixelFormat::RAW_12:
                return hololink::csi::get_packetizer_program(format);
            }
            throw nb::value_error("pixel_format must be RAW_8, RAW_10 or RAW_12");
        },
        "pixel_format"_a,
        "The packetizer program that packs a sensor's RAW_10 or RAW_12 pixels; for RAW_8, the "
        "one that disables the packetizer. v2.7 had this as hololink.csi.get_packetizer_program().");

    // Created shared-owned (nb::new_ + make_shared, not nb::init), so the channel can pass
    // itself to an Operator as the `data_channel` constructor handle without the Operator
    // keeping this Python wrapper alive: graph.op(hololink.operators.RoceReceiverOp, "rx",
    // data_channel=channel, ...).
    nb::class_<hololink::DataChannel>(module, "DataChannel")
        .def(nb::new_([](const hololink::Metadata& metadata) {
            return std::make_shared<hololink::DataChannel>(metadata);
        }),
            "metadata"_a)
        .def("hololink", &hololink::DataChannel::hololink)
        .def("frame_end_sequencer", &hololink::DataChannel::frame_end_sequencer,
            "A Sequencer that runs at each frame end of this channel's sensor, e.g. for "
            "register writes synchronized with frames (I2c.encode_i2c_request()).")
        .def("set_packetizer_program", &hololink::DataChannel::set_packetizer_program,
            "program"_a,
            "Use program (get_packetizer_program()) when the receiver configures the channel.")
        // Direct data-plane setup, for hosts that run a raw LinuxReceiver/LinuxCoeReceiver
        // outside an operator graph (tests/test_emulator.py drives the emulator this way).
        // The calls that write the board's registers release the GIL, as v2.7 did: an emulator
        // in the same process needs it to answer them.
        .def("configure_socket", &hololink::DataChannel::configure_socket, "socket_fd"_a,
            nb::call_guard<nb::gil_scoped_release>(),
            "bind the UDP data socket to this channel's local interface and address")
        .def("authenticate", &hololink::DataChannel::authenticate, "qp_number"_a, "rkey"_a)
        .def("configure_roce", &hololink::DataChannel::configure_roce,
            "frame_memory"_a, "frame_size"_a, "page_size"_a, "pages"_a, "local_data_port"_a,
            nb::call_guard<nb::gil_scoped_release>(),
            "program the board to send RoCEv2 frames to local_data_port; start of transmission")
        .def("configure_coe", &hololink::DataChannel::configure_coe,
            "channel"_a, "frame_size"_a, "pixel_width"_a, "vlan_enabled"_a = false,
            nb::call_guard<nb::gil_scoped_release>(),
            "program the board to send IEEE 1722 CoE frames on channel")
        .def("unconfigure", &hololink::DataChannel::unconfigure,
            nb::call_guard<nb::gil_scoped_release>(), "stop transmission")
        // internal's DataChannel statics, which the board tools use on hand-built metadata;
        // use_sensor() is also the module function hololink.use_sensor().
        .def_static("use_sensor", &hololink::DataChannel::use_sensor, "metadata"_a,
            "sensor_number"_a)
        .def_static("use_data_plane_configuration",
            &hololink::DataChannel::use_data_plane_configuration, "metadata"_a, "data_plane"_a)
        .def(
            "__holoscan_constructor_handle__",
            [](hololink::DataChannel& self, const std::string& type_name) {
                return steal_constructor_handle(holoscan::python::export_constructor_handle(
                    type_name, hololink::operators::data_channel_handle_type(),
                    self.weak_from_this().lock()));
            },
            "type_name"_a,
            "Holoscan calls this to pass the channel to an Operator as a constructor handle.");

    // hololink.DataChannels: the DataChannels of a GpuRoceTransceiverOp's sensors after its
    // data_channel's (extra_data_channels=...), one per QP of its tx_ibv_qp after the first.
    nb::class_<hololink::operators::DataChannels>(module, "DataChannels")
        .def(nb::new_([](std::vector<std::shared_ptr<hololink::DataChannel>> channels) {
            auto data_channels = std::make_shared<hololink::operators::DataChannels>();
            data_channels->channels = std::move(channels);
            return data_channels;
        }),
            "channels"_a)
        .def(
            "__holoscan_constructor_handle__",
            [](hololink::operators::DataChannels& self, const std::string& type_name) {
                return steal_constructor_handle(holoscan::python::export_constructor_handle(
                    type_name, hololink::operators::data_channels_handle_type(),
                    self.shared_from_this()));
            },
            "type_name"_a,
            "Holoscan calls this to pass the channels to an Operator as a constructor handle.");

    // JESD configuration of an HSB100G's AD9986 + HMC7044 MxFE (signal_generator.py). It holds a
    // reference to its HololinkDevice, which keep_alive pins. apply() blocks until an SPI daemon
    // in the board's ADI Petalinux VM connects and drives the JESD state machine;
    // verify_link_status() does register I/O; stop() stops the SPI daemon thread. All three release
    // the GIL. stop() must not run concurrently with apply(): pass both to the receiver as its
    // device_start/device_stop callbacks, whose lifecycle calls them in turn.
    nb::class_<hololink::AD9986Config>(module, "AD9986Config")
        .def(nb::init<hololink::Hololink&>(), "hololink"_a, nb::keep_alive<1, 2>())
        .def("host_pause_mapping", &hololink::AD9986Config::host_pause_mapping, "mask"_a,
            "Select the host-pause input: 0x01 for interface 0, 0x02 for interface 1.")
        .def("apply", &hololink::AD9986Config::apply, nb::call_guard<nb::gil_scoped_release>())
        .def("verify_link_status", &hololink::AD9986Config::verify_link_status,
            nb::call_guard<nb::gil_scoped_release>(),
            "True if the JESD link is up with no lane errors and no UPHY overflow.")
        .def("stop", &hololink::AD9986Config::stop, nb::call_guard<nb::gil_scoped_release>(),
            "Stop the SPI daemon thread apply() started.");

#ifdef HOLOLINK_HAS_IMGUI
    // The process-wide window the signal operators draw their GUIs into (their gui=True
    // parameter). Bound so apps can poll it: window_closed() becomes true when the user closes
    // the window.
    nb::class_<hololink::ImGuiRenderer>(module, "ImGuiRenderer")
        .def_static("shared", &hololink::ImGuiRenderer::shared,
            "The process-wide renderer, created on first use and destroyed with its last owner.")
        .def("is_running", &hololink::ImGuiRenderer::is_running,
            "False once the window was closed or couldn't be created.")
        .def("window_closed", &hololink::ImGuiRenderer::window_closed,
            "True only when the user closed the window.")
        .def("set_window_order", &hololink::ImGuiRenderer::set_window_order, "names"_a,
            "Stack the windows with these names first, in this order, and the others below "
            "them in the order the operators added them (in their start(), whose order the "
            "runtime picks).");
#else
    struct ImGuiRendererNotBuilt { };
    nb::class_<ImGuiRendererNotBuilt>(module, "ImGuiRenderer")
        .def_static("shared", []() -> nb::object {
            throw holoscan::RuntimeError { holoscan::Error { holoscan::ErrorCode::kNotSupported,
                "HOLOLINK_IMGUI_NOT_BUILT detail=built without holoscan::visualization::holoviz" } };
        });
#endif

    // hololink.CrcChannel: the CrcChannel a ComputeCrcOp and its CheckCrcOp share
    // (crc_channel=...), with an optional check_crc(computed_crc, received_crc, frame_number)
    // callback that the CheckCrcOp calls for each frame it checks. The callback runs on
    // CheckCrcOp's thread; it takes the GIL itself, and a Python exception in it fails that
    // activation.
    nb::class_<hololink::operators::CrcCheckChannel>(module, "CrcChannel")
        .def(nb::new_([](nb::object check_crc, size_t depth) {
            hololink::operators::CrcCheckChannel::Check callback;
            if (!check_crc.is_none()) {
                if (!PyCallable_Check(check_crc.ptr())) {
                    throw nb::type_error("CrcChannel: check_crc must be callable or None");
                }
                // The last copy of the callback can be destroyed on a runtime thread: release
                // the Python reference with the GIL held.
                auto callable = std::shared_ptr<nb::object>(
                    new nb::object(std::move(check_crc)), [](nb::object* p) {
                        nb::gil_scoped_acquire gil;
                        delete p;
                    });
                callback = [callable](uint32_t computed_crc, int64_t received_crc, int64_t frame_number) {
                    nb::gil_scoped_acquire gil;
                    try {
                        (*callable)(computed_crc, received_crc, frame_number);
                    } catch (nb::python_error& e) {
                        std::string message = e.what();
                        throw std::runtime_error("CrcChannel check_crc callback failed: " + message);
                    }
                };
            }
            return std::make_shared<hololink::operators::CrcCheckChannel>(std::move(callback), depth);
        }),
            "check_crc"_a = nb::none(), "depth"_a = 16,
            "check_crc(computed_crc, received_crc, frame_number) is called for each frame the "
            "CheckCrcOp checks. depth is how many frames' CRCs can wait for their CheckCrcOp.")
        .def(
            "__holoscan_constructor_handle__",
            [](hololink::operators::CrcCheckChannel& self, const std::string& type_name) {
                return steal_constructor_handle(holoscan::python::export_constructor_handle(
                    type_name, hololink::operators::crc_channel_handle_type(), self.shared_from_this()));
            },
            "type_name"_a,
            "Holoscan calls this to pass the channel to ComputeCrcOp and CheckCrcOp as a "
            "constructor handle.");

    // Created shared-owned for the same reason as DataChannel: start() calls shared_from_this(),
    // and an Operator calls start() through a device_start=camera.start callback, which no
    // shared_ptr conversion ever reaches.
    nb::class_<hololink::sensors::NativeVb1940Sensor>(module, "NativeVb1940Sensor")
        .def(nb::new_([](hololink::DataChannel& data_channel) {
            return std::make_shared<hololink::sensors::NativeVb1940Sensor>(data_channel);
        }),
            "data_channel"_a)
        .def(nb::new_([](hololink::DataChannel& data_channel,
                          std::shared_ptr<hololink::Synchronizer> synchronizer) {
            return std::make_shared<hololink::sensors::NativeVb1940Sensor>(
                data_channel, std::move(synchronizer));
        }),
            "data_channel"_a, "synchronizer"_a)
        .def("setup_clock", &hololink::sensors::NativeVb1940Sensor::setup_clock)
        .def(
            "configure", [](hololink::sensors::NativeVb1940Sensor& self, int mode) {
                self.configure(static_cast<hololink::sensors::vb1940_mode::Mode>(mode));
            },
            "mode"_a)
        .def("get_register_32", &hololink::sensors::NativeVb1940Sensor::get_register_32, "reg"_a)
        .def("set_analog_gain_reg", &hololink::sensors::NativeVb1940Sensor::set_analog_gain_reg, "value"_a = 0x00)
        .def("set_exposure_reg", &hololink::sensors::NativeVb1940Sensor::set_exposure_reg, "value"_a = 0x0014)
        .def(
            "get_rgb_calibration_data_string", [](hololink::sensors::NativeVb1940Sensor& self) {
                return self.get_rgb_calibration_data().to_string();
            },
            "Formatted RGB calibration/EEPROM data. Raises if the camera has none or it "
            "can't be read -- matches the native C++ getter, which can legitimately throw "
            "(the app should catch this, not treat it as fatal).")
        .def(
            "get_ir_calibration_data_string", [](hololink::sensors::NativeVb1940Sensor& self) {
                return self.get_ir_calibration_data().to_string();
            },
            "Formatted IR calibration/EEPROM data, like get_rgb_calibration_data_string().")
        .def("get_pixel_format", [](const hololink::sensors::NativeVb1940Sensor& self) {
            return static_cast<int>(self.get_pixel_format());
        })
        .def("get_bayer_format", [](const hololink::sensors::NativeVb1940Sensor& self) {
            switch (self.get_bayer_format()) {
            case hololink::csi::BayerFormat::BGGR:
                return "bggr";
            case hololink::csi::BayerFormat::RGGB:
                return "rggb";
            case hololink::csi::BayerFormat::GBRG:
                return "gbrg";
            case hololink::csi::BayerFormat::GRBG:
                return "grbg";
            }
            return "gbrg";
        })
        .def("get_width", &hololink::sensors::NativeVb1940Sensor::get_width)
        .def("get_height", &hololink::sensors::NativeVb1940Sensor::get_height)
        .def("start", &hololink::sensors::NativeVb1940Sensor::start)
        .def("stop", &hololink::sensors::NativeVb1940Sensor::stop);

    nb::class_<hololink::sensors::NativeImx274Sensor>(module, "NativeImx274Sensor")
        .def(nb::init<hololink::DataChannel&, int>(), "data_channel"_a, "expander_configuration"_a = 0)
        .def("setup_clock", &hololink::sensors::NativeImx274Sensor::setup_clock)
        // Raw register access, for python/tools/imx274.py.
        .def("get_register", &hololink::sensors::NativeImx274Sensor::get_register, "address"_a)
        .def(
            "set_register",
            [](hololink::sensors::NativeImx274Sensor& self, uint16_t address, uint8_t value) {
                self.set_register(address, value);
            },
            "address"_a, "value"_a)
        .def(
            "configure", [](hololink::sensors::NativeImx274Sensor& self, int mode) {
                self.configure(static_cast<hololink::sensors::imx274_mode::Mode>(mode));
            },
            "mode"_a)
        // Selects the mode's frame geometry (get_width()/get_height()/...) without touching the
        // sensor, so an app can check its arguments against it before configuring the board.
        .def(
            "set_mode", [](hololink::sensors::NativeImx274Sensor& self, int mode) {
                self.set_mode(static_cast<hololink::sensors::imx274_mode::Mode>(mode));
            },
            "mode"_a)
        .def("set_digital_gain_reg", &hololink::sensors::NativeImx274Sensor::set_digital_gain_reg, "value"_a = 0x0000)
        .def("test_pattern", &hololink::sensors::NativeImx274Sensor::test_pattern, "pattern"_a = 0)
        .def("get_pixel_format", [](const hololink::sensors::NativeImx274Sensor& self) {
            return static_cast<int>(self.get_pixel_format());
        })
        .def("get_bayer_format", [](const hololink::sensors::NativeImx274Sensor& self) {
            switch (self.get_bayer_format()) {
            case hololink::csi::BayerFormat::BGGR:
                return "bggr";
            case hololink::csi::BayerFormat::RGGB:
                return "rggb";
            case hololink::csi::BayerFormat::GBRG:
                return "gbrg";
            case hololink::csi::BayerFormat::GRBG:
                return "grbg";
            }
            return "gbrg";
        })
        .def("get_width", &hololink::sensors::NativeImx274Sensor::get_width)
        .def("get_height", &hololink::sensors::NativeImx274Sensor::get_height)
        .def("start", &hololink::sensors::NativeImx274Sensor::start)
        .def("stop", &hololink::sensors::NativeImx274Sensor::stop);

#ifdef HOLOLINK_BUILD_FUSA
    module.def("resolve_geometry",
        nb::overload_cast<std::shared_ptr<hololink::sensors::NativeVb1940Sensor>, bool>(&resolve_geometry),
        "camera"_a, "packetizer_enabled"_a);
    module.def("resolve_geometry",
        nb::overload_cast<std::shared_ptr<hololink::sensors::NativeImx274Sensor>, bool>(&resolve_geometry),
        "camera"_a, "packetizer_enabled"_a);
#endif // HOLOLINK_BUILD_FUSA

    module.def("resolve_packed_geometry",
        nb::overload_cast<std::shared_ptr<hololink::sensors::NativeVb1940Sensor>>(&resolve_packed_geometry),
        "camera"_a,
        "Resolve the CSI geometry of the camera's frames when the HSB's packetizer packs them "
        "(DataChannel.set_packetizer_program()), for PackedFormatConverterOp after any "
        "receiver.");
    module.def("resolve_packed_geometry",
        nb::overload_cast<std::shared_ptr<hololink::sensors::NativeImx274Sensor>>(&resolve_packed_geometry),
        "camera"_a);
    module.def("resolve_generic_geometry",
        nb::overload_cast<std::shared_ptr<hololink::sensors::NativeVb1940Sensor>>(&resolve_generic_geometry),
        "camera"_a,
        "Resolve CSI geometry for a RoceReceiverOp/LinuxReceiverOp-fed pipeline (generic "
        "8-byte line alignment) -- use resolve_geometry() instead for a FusaCoeCaptureOp-fed "
        "pipeline (64-byte HSB alignment).");
    module.def("resolve_generic_geometry",
        nb::overload_cast<std::shared_ptr<hololink::sensors::NativeImx274Sensor>>(&resolve_generic_geometry),
        "camera"_a, "NativeImx274Sensor overload of resolve_generic_geometry() above.");

    module.def("resolve_generic_geometry_from_dimensions", &resolve_generic_geometry_from_dimensions,
        "pixel_format"_a, "width"_a, "height"_a,
        "Same generic (8-byte line alignment) geometry resolution as resolve_generic_geometry(), "
        "for pure-Python sensor drivers with no C++-bound camera type to pass to that function "
        "instead (e.g. hololink.sensors.ecam0m30tof.ECam0M30Tof).");

    // ---------------------------------------------------------------------------------------
    // Raw receivers, usable without an operator graph (tests/test_emulator.py,
    // tests/test_examples_emulator.py). Each blocking call releases the GIL: run() loops
    // until close(), get_next_frame() waits up to timeout_ms, and both are normally called
    // from a Python thread other than the one that owns the emulator or the data channel.
    // CUdeviceptr / CUcontext / CUstream cross as plain integers, as cuda-python exposes them.
    // ---------------------------------------------------------------------------------------
    nb::class_<hololink::ReceiverMemoryDescriptor>(module, "ReceiverMemoryDescriptor",
        "A receive buffer in device (or, failing that, pinned host) memory of the given size, "
        "allocated in the given CUDA context; released with the object")
        .def(
            "__init__",
            [](hololink::ReceiverMemoryDescriptor* self, int64_t context, size_t size) {
                new (self) hololink::ReceiverMemoryDescriptor(reinterpret_cast<CUcontext>(context), size);
            },
            "context"_a, "size"_a)
        .def(
            "get", [](hololink::ReceiverMemoryDescriptor& self) -> uint64_t { return static_cast<uint64_t>(self.get()); },
            "the CUdeviceptr of the allocated memory");

    nb::class_<hololink::operators::LinuxReceiverMetadata>(module, "LinuxReceiverMetadata")
        .def_ro("frame_packets_received", &hololink::operators::LinuxReceiverMetadata::frame_packets_received)
        .def_ro("frame_bytes_received", &hololink::operators::LinuxReceiverMetadata::frame_bytes_received)
        .def_ro("received_frame_number", &hololink::operators::LinuxReceiverMetadata::received_frame_number)
        .def_ro("frame_start_s", &hololink::operators::LinuxReceiverMetadata::frame_start_s)
        .def_ro("frame_start_ns", &hololink::operators::LinuxReceiverMetadata::frame_start_ns)
        .def_ro("frame_end_s", &hololink::operators::LinuxReceiverMetadata::frame_end_s)
        .def_ro("frame_end_ns", &hololink::operators::LinuxReceiverMetadata::frame_end_ns)
        .def_ro("imm_data", &hololink::operators::LinuxReceiverMetadata::imm_data)
        .def_ro("packets_dropped", &hololink::operators::LinuxReceiverMetadata::packets_dropped)
        .def_ro("received_s", &hololink::operators::LinuxReceiverMetadata::received_s)
        .def_ro("received_ns", &hololink::operators::LinuxReceiverMetadata::received_ns)
        .def_ro("frame_number", &hololink::operators::LinuxReceiverMetadata::frame_number)
        .def_prop_ro("frame_memory", [](const hololink::operators::LinuxReceiverMetadata& me) -> uint64_t { return static_cast<uint64_t>(me.frame_memory); })
        .def_prop_ro("timestamp_s", [](const hololink::operators::LinuxReceiverMetadata& me) { return me.frame_metadata.timestamp_s; })
        .def_prop_ro("timestamp_ns", [](const hololink::operators::LinuxReceiverMetadata& me) { return me.frame_metadata.timestamp_ns; })
        .def_prop_ro("metadata_s", [](const hololink::operators::LinuxReceiverMetadata& me) { return me.frame_metadata.metadata_s; })
        .def_prop_ro("metadata_ns", [](const hololink::operators::LinuxReceiverMetadata& me) { return me.frame_metadata.metadata_ns; })
        .def_prop_ro("crc", [](const hololink::operators::LinuxReceiverMetadata& me) { return me.frame_metadata.crc; })
        .def_prop_ro("psn", [](const hololink::operators::LinuxReceiverMetadata& me) { return me.frame_metadata.psn; })
        .def_prop_ro("bytes_written", [](const hololink::operators::LinuxReceiverMetadata& me) { return me.frame_metadata.bytes_written; });

    nb::class_<hololink::operators::LinuxReceiver>(module, "LinuxReceiver",
        "UDP receiver of the board's RoCEv2 data stream into a ReceiverMemoryDescriptor; "
        "run() blocks until close()")
        .def(
            "__init__",
            [](hololink::operators::LinuxReceiver* self, uint64_t cu_buffer, size_t cu_buffer_size, size_t cu_page_size,
                unsigned pages, int socket, uint64_t received_address_offset, unsigned queue_size) {
                new (self) hololink::operators::LinuxReceiver(static_cast<CUdeviceptr>(cu_buffer), cu_buffer_size, cu_page_size,
                    pages, socket, received_address_offset, queue_size);
            },
            "cu_buffer"_a, "cu_buffer_size"_a, "cu_page_size"_a, "pages"_a, "socket"_a, "received_address_offset"_a,
            "queue_size"_a = 1)
        .def("run", &hololink::operators::LinuxReceiver::run, nb::call_guard<nb::gil_scoped_release>())
        .def("close", &hololink::operators::LinuxReceiver::close)
        .def(
            "get_next_frame",
            [](hololink::operators::LinuxReceiver& self, unsigned timeout_ms, int64_t cuda_stream) {
                hololink::operators::LinuxReceiverMetadata metadata;
                bool ok;
                {
                    nb::gil_scoped_release nogil;
                    ok = self.get_next_frame(timeout_ms, metadata, reinterpret_cast<CUstream>(cuda_stream));
                }
                return std::make_tuple(ok, metadata);
            },
            "timeout_ms"_a, "cuda_stream"_a = 0,
            "wait up to timeout_ms for the next frame; returns (ok, LinuxReceiverMetadata)")
        .def("frames_ready", &hololink::operators::LinuxReceiver::frames_ready, nb::call_guard<nb::gil_scoped_release>())
        .def("get_qp_number", &hololink::operators::LinuxReceiver::get_qp_number)
        .def("get_rkey", &hololink::operators::LinuxReceiver::get_rkey);

    nb::class_<hololink::operators::LinuxCoeReceiverMetadata>(module, "LinuxCoeReceiverMetadata")
        .def_ro("frame_packets_received", &hololink::operators::LinuxCoeReceiverMetadata::frame_packets_received)
        .def_ro("frame_bytes_received", &hololink::operators::LinuxCoeReceiverMetadata::frame_bytes_received)
        .def_ro("received_frame_number", &hololink::operators::LinuxCoeReceiverMetadata::received_frame_number)
        .def_ro("frame_start_s", &hololink::operators::LinuxCoeReceiverMetadata::frame_start_s)
        .def_ro("frame_start_ns", &hololink::operators::LinuxCoeReceiverMetadata::frame_start_ns)
        .def_ro("frame_end_s", &hololink::operators::LinuxCoeReceiverMetadata::frame_end_s)
        .def_ro("frame_end_ns", &hololink::operators::LinuxCoeReceiverMetadata::frame_end_ns)
        .def_ro("received_s", &hololink::operators::LinuxCoeReceiverMetadata::received_s)
        .def_ro("received_ns", &hololink::operators::LinuxCoeReceiverMetadata::received_ns)
        .def_ro("frame_number", &hololink::operators::LinuxCoeReceiverMetadata::frame_number)
        .def_prop_ro("timestamp_s", [](const hololink::operators::LinuxCoeReceiverMetadata& me) { return me.frame_metadata.timestamp_s; })
        .def_prop_ro("timestamp_ns", [](const hololink::operators::LinuxCoeReceiverMetadata& me) { return me.frame_metadata.timestamp_ns; })
        .def_prop_ro("metadata_s", [](const hololink::operators::LinuxCoeReceiverMetadata& me) { return me.frame_metadata.metadata_s; })
        .def_prop_ro("metadata_ns", [](const hololink::operators::LinuxCoeReceiverMetadata& me) { return me.frame_metadata.metadata_ns; })
        .def_prop_ro("crc", [](const hololink::operators::LinuxCoeReceiverMetadata& me) { return me.frame_metadata.crc; })
        .def_prop_ro("psn", [](const hololink::operators::LinuxCoeReceiverMetadata& me) { return me.frame_metadata.psn; })
        .def_prop_ro("bytes_written", [](const hololink::operators::LinuxCoeReceiverMetadata& me) { return me.frame_metadata.bytes_written; });

    nb::class_<hololink::operators::LinuxCoeReceiver>(module, "LinuxCoeReceiver",
        "AF_PACKET receiver of the board's IEEE 1722 CoE data stream into a ReceiverMemoryDescriptor; "
        "run() blocks until close()")
        .def(
            "__init__",
            [](hololink::operators::LinuxCoeReceiver* self, uint64_t cu_buffer, size_t cu_buffer_size, int socket, uint16_t channel) {
                new (self) hololink::operators::LinuxCoeReceiver(static_cast<CUdeviceptr>(cu_buffer), cu_buffer_size, socket, channel);
            },
            "cu_buffer"_a, "cu_buffer_size"_a, "socket"_a, "channel"_a)
        .def("run", &hololink::operators::LinuxCoeReceiver::run, nb::call_guard<nb::gil_scoped_release>())
        .def("close", &hololink::operators::LinuxCoeReceiver::close)
        .def(
            "get_next_frame",
            [](hololink::operators::LinuxCoeReceiver& self, unsigned timeout_ms, int64_t cuda_stream) {
                hololink::operators::LinuxCoeReceiverMetadata metadata;
                bool ok;
                {
                    nb::gil_scoped_release nogil;
                    ok = self.get_next_frame(timeout_ms, metadata, reinterpret_cast<CUstream>(cuda_stream));
                }
                return std::make_tuple(ok, metadata);
            },
            "timeout_ms"_a, "cuda_stream"_a = 0,
            "wait up to timeout_ms for the next frame; returns (ok, LinuxCoeReceiverMetadata)")
        .def("frames_ready", &hololink::operators::LinuxCoeReceiver::frames_ready, nb::call_guard<nb::gil_scoped_release>());

    // Keep the real hololink operators DSO in the dependency closure, matching
    // holoscan-sdk's _vision.cpp _check_native_linkage() pattern.
    module.def("_check_native_linkage",
        [] { const hololink::operators::DiscardMetadataOp operation; (void)operation; });
}
