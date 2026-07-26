/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pybind11/pybind11.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cuda.h>

#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <holoscan/core/condition.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/resource.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>

#include "hololink/module/csi_converter.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/frame_metadata.hpp"
#include "hololink/module/linux_data_channel.hpp"
#include "hololink/module/operators/csi_to_bayer_op.hpp"
#include "hololink/module/operators/hsb_controller_op.hpp"
#include "hololink/module/operators/image_processor_op.hpp"
#include "hololink/module/operators/linux_network_receiver.hpp"
#include "hololink/module/operators/linux_receiver_op.hpp"
#include "hololink/module/operators/network_receiver.hpp"
#include "hololink/module/operators/packed_format_converter_op.hpp"
#include "hololink/module/operators/sensor_device.hpp"
#include "hololink/module/operators/sensor_factory.hpp"

#ifdef HOLOLINK_BUILD_ROCE
#include "hololink/module/operators/roce_network_receiver.hpp"
#include "hololink/module/operators/roce_receiver_op.hpp"
#include "hololink/module/roce_data_channel.hpp"
#endif

#ifdef HOLOLINK_BUILD_FUSA
#include "hololink/module/operators/fusa_coe_capture_op.hpp"
#endif

#ifdef HOLOLINK_MODULE_BUILD_CRC
#include "hololink/module/operators/compute_crc_op.hpp"
#endif

namespace py = pybind11;

using std::string_literals::operator""s;
using py::literals::operator""_a;

namespace hololink::module::operators {

static void forward_positional_condition_and_resource_args(
    holoscan::Operator* op, const py::args& args)
{
    for (const auto& a : args) {
        if (py::isinstance<holoscan::Condition>(a)) {
            op->add_arg(a.cast<std::shared_ptr<holoscan::Condition>>());
        } else if (py::isinstance<holoscan::Resource>(a)) {
            op->add_arg(a.cast<std::shared_ptr<holoscan::Resource>>());
        }
    }
}

/* Trampoline for the software receiver operator. Mirrors
 * PyRoceReceiverOp but with no metadata_offset (the software receiver
 * sizes its metadata block from the module's FrameMetadataInterface)
 * and an added receiver_affinity argument. Not gated on RoCE — the
 * Linux receiver needs no ibverbs and is always available when the
 * operators tree is built. */
class PyLinuxReceiverOp : public LinuxReceiverOp {
public:
    using LinuxReceiverOp::LinuxReceiverOp;

    PyLinuxReceiverOp(holoscan::Fragment* fragment, const py::args& args,
        hololink::module::EnumerationMetadata enumeration_metadata,
        py::object frame_context,
        size_t frame_size,
        size_t page_size,
        uint32_t pages,
        uint32_t queue_size,
        std::vector<int> receiver_affinity,
        std::function<void()> device_start,
        std::function<void()> device_stop,
        std::function<std::string(const std::string&)> rename_metadata,
        const std::string& out_tensor_name,
        const std::string& name)
        : LinuxReceiverOp(holoscan::ArgList {
            holoscan::Arg { "enumeration_metadata", std::move(enumeration_metadata) },
            holoscan::Arg { "frame_context",
                reinterpret_cast<CUcontext>(frame_context.cast<int64_t>()) },
            holoscan::Arg { "frame_size", frame_size },
            holoscan::Arg { "page_size", page_size },
            holoscan::Arg { "pages", pages },
            holoscan::Arg { "queue_size", queue_size },
            holoscan::Arg { "receiver_affinity", std::move(receiver_affinity) },
            holoscan::Arg { "device_start", std::move(device_start) },
            holoscan::Arg { "device_stop", std::move(device_stop) },
            holoscan::Arg { "rename_metadata", std::move(rename_metadata) },
            holoscan::Arg { "out_tensor_name", out_tensor_name },
        })
    {
        forward_positional_condition_and_resource_args(this, args);
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<holoscan::OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

/* Trampoline for the CSI -> Bayer converter operator. A Pythonic kwarg
 * constructor forwarding to a holoscan::ArgList then running the
 * make_operator<> setup steps. Not
 * gated on any capability — the converter needs no ibverbs / FUSA and is
 * always built when the operators tree is. */
class PyCsiToBayerOp : public CsiToBayerOp {
public:
    using CsiToBayerOp::CsiToBayerOp;

    PyCsiToBayerOp(holoscan::Fragment* fragment, const py::args& args,
        const std::shared_ptr<holoscan::Allocator>& allocator,
        int cuda_device_ordinal,
        const std::string& out_tensor_name,
        uint32_t sub_frame_rows,
        const std::string& name)
        : CsiToBayerOp(holoscan::ArgList {
            holoscan::Arg { "allocator", allocator },
            holoscan::Arg { "cuda_device_ordinal", cuda_device_ordinal },
            holoscan::Arg { "out_tensor_name", out_tensor_name },
            holoscan::Arg { "sub_frame_rows", sub_frame_rows },
        })
    {
        forward_positional_condition_and_resource_args(this, args);
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<holoscan::OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

/* Trampoline for the image-processing operator. A Pythonic kwarg constructor
 * forwarding to a holoscan::ArgList then running the make_operator<> setup
 * steps. Not gated
 * on any capability — it needs no ibverbs / FUSA and is always built when
 * the operators tree is. */
class PyImageProcessorOp : public ImageProcessorOp {
public:
    using ImageProcessorOp::ImageProcessorOp;

    PyImageProcessorOp(holoscan::Fragment* fragment, const py::args& args,
        int pixel_format,
        int bayer_format,
        int32_t optical_black,
        int cuda_device_ordinal,
        const std::string& name)
        : ImageProcessorOp(holoscan::ArgList {
            holoscan::Arg { "pixel_format", pixel_format },
            holoscan::Arg { "bayer_format", bayer_format },
            holoscan::Arg { "optical_black", optical_black },
            holoscan::Arg { "cuda_device_ordinal", cuda_device_ordinal },
        })
    {
        forward_positional_condition_and_resource_args(this, args);
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<holoscan::OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

/* Trampoline for the packed-format converter operator. A Pythonic kwarg
 * constructor forwarding to a holoscan::ArgList then running the
 * make_operator<> setup steps. Not gated on
 * any capability — it needs no ibverbs / FUSA and is always built when the
 * operators tree is. */
class PyPackedFormatConverterOp : public PackedFormatConverterOp {
public:
    using PackedFormatConverterOp::PackedFormatConverterOp;

    PyPackedFormatConverterOp(holoscan::Fragment* fragment, const py::args& args,
        const std::shared_ptr<holoscan::Allocator>& allocator,
        int cuda_device_ordinal,
        const std::string& in_tensor_name,
        const std::string& out_tensor_name,
        const std::string& name)
        : PackedFormatConverterOp(holoscan::ArgList {
            holoscan::Arg { "allocator", allocator },
            holoscan::Arg { "cuda_device_ordinal", cuda_device_ordinal },
            holoscan::Arg { "in_tensor_name", in_tensor_name },
            holoscan::Arg { "out_tensor_name", out_tensor_name },
        })
    {
        forward_positional_condition_and_resource_args(this, args);
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<holoscan::OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

/* Trampoline for the sensor wrapper. The application subclasses SensorDevice
 * in Python (e.g. Imx274SensorDevice), overriding stop_sensor(). */
class PySensorDevice : public SensorDevice {
public:
    using SensorDevice::SensorDevice;

    hololink_module_status_t stop_sensor() override
    {
        PYBIND11_OVERRIDE_PURE(hololink_module_status_t, SensorDevice, stop_sensor);
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
    using SensorFactory::SensorFactory;

    void fallback_frame(CUdeviceptr& out_ptr, size_t& out_size) override
    {
        // Pythonic out-params: the override returns a (ptr, size) tuple, or
        // None / nothing for no fallback (the base default).
        out_ptr = 0;
        out_size = 0;
        py::gil_scoped_acquire gil;
        py::function override
            = py::get_override(static_cast<const SensorFactory*>(this), "fallback_frame");
        if (!override) {
            return;
        }
        py::object result = override();
        if (result.is_none()) {
            return;
        }
        auto frame = result.cast<std::pair<uint64_t, size_t>>();
        out_ptr = static_cast<CUdeviceptr>(frame.first);
        out_size = frame.second;
    }

    std::shared_ptr<SensorDevice> new_sensor(
        const hololink::module::EnumerationMetadata& metadata) override
    {
        py::gil_scoped_acquire gil;
        py::function override
            = py::get_override(static_cast<const SensorFactory*>(this), "new_sensor");
        if (!override) {
            throw std::runtime_error("SensorFactory.new_sensor is not overridden");
        }
        try {
            return override(metadata).cast<std::shared_ptr<SensorDevice>>();
        } catch (py::error_already_set& e) {
            // A (re)connect attempt raised. Report it and return null so the
            // factory stays disconnected and retries on the next announcement,
            // rather than letting the exception abort the reactor thread
            // new_sensor runs on.
            e.discard_as_unraisable("SensorFactory.new_sensor (retry on re-announce)");
            return nullptr;
        }
    }
};

#ifdef HOLOLINK_BUILD_ROCE
/* Trampoline class for handling Python kwargs.
 *
 * Adds a Pythonic constructor that takes a Fragment plus the
 * module-flavored parameters (typed shared_ptr<...> for the V1
 * service handles), forwards them to RoceReceiverOp via a
 * holoscan::ArgList, then runs the same Fragment::make_operator<>
 * setup steps the existing tree's operator pybinds use. */
class PyRoceReceiverOp : public RoceReceiverOp {
public:
    using RoceReceiverOp::RoceReceiverOp;

    PyRoceReceiverOp(holoscan::Fragment* fragment, const py::args& args,
        hololink::module::EnumerationMetadata enumeration_metadata,
        py::object frame_context,
        size_t frame_size,
        size_t page_size,
        uint32_t pages,
        uint32_t queue_size,
        size_t metadata_offset,
        std::function<void()> device_start,
        std::function<void()> device_stop,
        std::function<std::string(const std::string&)> rename_metadata,
        const std::string& out_tensor_name,
        const std::string& name)
        : RoceReceiverOp(holoscan::ArgList {
            holoscan::Arg { "enumeration_metadata", std::move(enumeration_metadata) },
            holoscan::Arg { "frame_context",
                reinterpret_cast<CUcontext>(frame_context.cast<int64_t>()) },
            holoscan::Arg { "frame_size", frame_size },
            holoscan::Arg { "page_size", page_size },
            holoscan::Arg { "pages", pages },
            holoscan::Arg { "queue_size", queue_size },
            holoscan::Arg { "metadata_offset", metadata_offset },
            holoscan::Arg { "device_start", std::move(device_start) },
            holoscan::Arg { "device_stop", std::move(device_stop) },
            holoscan::Arg { "rename_metadata", std::move(rename_metadata) },
            holoscan::Arg { "out_tensor_name", out_tensor_name },
        })
    {
        forward_positional_condition_and_resource_args(this, args);
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<holoscan::OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};
#endif // HOLOLINK_BUILD_ROCE

/* Trampoline for the reconnection controller operator. Pythonic kwarg
 * constructor forwarding to a holoscan::ArgList then running the
 * make_operator<> setup steps. Transport-agnostic (always built): the
 * network_receiver_factory selects RoCE / Linux, and the sensor_factory
 * supplies the camera. */
class PyHsbControllerOp : public HsbControllerOp {
public:
    using HsbControllerOp::HsbControllerOp;

    PyHsbControllerOp(holoscan::Fragment* fragment, const py::args& args,
        hololink::module::EnumerationMetadata enumeration_metadata,
        py::object frame_context,
        size_t frame_size,
        std::shared_ptr<NetworkReceiverFactory> network_receiver_factory,
        std::shared_ptr<SensorFactory> sensor_factory,
        size_t page_size,
        uint32_t pages,
        uint32_t queue_size,
        size_t metadata_offset,
        double watchdog_timeout_s,
        std::function<std::string(const std::string&)> rename_metadata,
        const std::string& out_tensor_name,
        const std::string& name)
        : HsbControllerOp(holoscan::ArgList {
            holoscan::Arg { "enumeration_metadata", std::move(enumeration_metadata) },
            holoscan::Arg { "frame_context",
                reinterpret_cast<CUcontext>(frame_context.cast<int64_t>()) },
            holoscan::Arg { "frame_size", frame_size },
            holoscan::Arg { "network_receiver_factory",
                std::move(network_receiver_factory) },
            holoscan::Arg { "sensor_factory", std::move(sensor_factory) },
            holoscan::Arg { "page_size", page_size },
            holoscan::Arg { "pages", pages },
            holoscan::Arg { "queue_size", queue_size },
            holoscan::Arg { "metadata_offset", metadata_offset },
            holoscan::Arg { "watchdog_timeout_s", watchdog_timeout_s },
            holoscan::Arg { "rename_metadata", std::move(rename_metadata) },
            holoscan::Arg { "out_tensor_name", out_tensor_name },
        })
    {
        forward_positional_condition_and_resource_args(this, args);
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<holoscan::OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

#ifdef HOLOLINK_BUILD_FUSA

class PyFusaCoeCaptureOp : public FusaCoeCaptureOp {
public:
    using FusaCoeCaptureOp::FusaCoeCaptureOp;

    PyFusaCoeCaptureOp(holoscan::Fragment* fragment, const py::args& args,
        hololink::module::EnumerationMetadata enumeration_metadata,
        const std::string& interface,
        const std::vector<uint8_t>& mac_addr,
        uint32_t timeout,
        py::object device,
        std::function<std::string(const std::string&)> rename_metadata,
        const std::string& out_tensor_name,
        const std::string& name)
        : FusaCoeCaptureOp(holoscan::ArgList {
            holoscan::Arg { "enumeration_metadata", std::move(enumeration_metadata) },
            holoscan::Arg { "interface", interface },
            holoscan::Arg { "mac_addr", mac_addr },
            holoscan::Arg { "timeout", timeout },
            holoscan::Arg { "out_tensor_name", out_tensor_name },
            holoscan::Arg { "rename_metadata", std::move(rename_metadata) },
            holoscan::Arg { "device_start", std::function<void()>([this]() {
                               py::gil_scoped_acquire guard;
                               device_.attr("start")();
                           }) },
            holoscan::Arg { "device_stop", std::function<void()>([this]() {
                               py::gil_scoped_acquire guard;
                               device_.attr("stop")();
                           }) },
        })
        , device_(device)
    {
        forward_positional_condition_and_resource_args(this, args);
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<holoscan::OperatorSpec>(fragment);
        setup(*spec_.get());
    }

private:
    py::object device_;
};

#endif // HOLOLINK_BUILD_FUSA

#ifdef HOLOLINK_MODULE_BUILD_CRC

/* Trampoline for the CRC compute operator. A Pythonic kwarg constructor
 * forwarding to a holoscan::ArgList then running the make_operator<> setup
 * steps. Built only when nvcomp is available. */
class PyComputeCrcOp : public ComputeCrcOp {
public:
    using ComputeCrcOp::ComputeCrcOp;

    PyComputeCrcOp(holoscan::Fragment* fragment, const py::args& args,
        int cuda_device_ordinal,
        uint64_t frame_size,
        const std::string& name)
        : ComputeCrcOp(holoscan::ArgList {
            holoscan::Arg { "cuda_device_ordinal", cuda_device_ordinal },
            holoscan::Arg { "frame_size", frame_size },
        })
    {
        forward_positional_condition_and_resource_args(this, args);
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<holoscan::OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

/* Trampoline for the CRC check operator, including the PYBIND11_OVERRIDE so
 * Python subclasses can override check_crc(). */
class PyCheckCrcOp : public CheckCrcOp {
public:
    using CheckCrcOp::CheckCrcOp;

    PyCheckCrcOp(holoscan::Fragment* fragment, const py::args& args,
        std::shared_ptr<ComputeCrcOp> compute_crc_op,
        const std::string& computed_crc_metadata_name,
        const std::string& name)
        : CheckCrcOp(holoscan::ArgList {
            holoscan::Arg { "compute_crc_op", compute_crc_op },
            holoscan::Arg { "computed_crc_metadata_name", computed_crc_metadata_name },
        })
    {
        forward_positional_condition_and_resource_args(this, args);
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<holoscan::OperatorSpec>(fragment);
        setup(*spec_.get());
    }

    void check_crc(uint32_t computed_crc) override
    {
        PYBIND11_OVERRIDE(void, CheckCrcOp, check_crc, computed_crc);
    }
};

#endif // HOLOLINK_MODULE_BUILD_CRC

} // namespace hololink::module::operators

PYBIND11_MODULE(_hololink_module_operators, m)
{
    m.doc() = "hololink_module Holoscan-coupled operator bindings";

    // Pull in the core extension so the V1 service types
    // (LinuxDataChannelInterface, RoceDataChannelInterface,
    // FrameMetadataInterface, EnumerationMetadata) are registered before
    // any operator's parameter types are introspected from Python.
    // Always imported so `import hololink_module.operators` works
    // regardless of which operators this build enabled.
    py::module_::import("hololink_module._hololink_py_module");

    // The software (Linux) receiver operator needs no ibverbs and is
    // always built when the operators tree is.
    py::class_<hololink::module::operators::LinuxReceiverOp,
        hololink::module::operators::PyLinuxReceiverOp,
        holoscan::Operator,
        std::shared_ptr<hololink::module::operators::LinuxReceiverOp>>(
        m, "LinuxReceiverOp")
        .def(py::init<holoscan::Fragment*, const py::args&,
                 hololink::module::EnumerationMetadata,
                 py::object,
                 size_t,
                 size_t,
                 uint32_t,
                 uint32_t,
                 std::vector<int>,
                 std::function<void()>,
                 std::function<void()>,
                 std::function<std::string(const std::string&)>,
                 const std::string&,
                 const std::string&>(),
            "fragment"_a,
            "enumeration_metadata"_a,
            "frame_context"_a,
            "frame_size"_a,
            "page_size"_a = size_t { 0 },
            "pages"_a = uint32_t { 2 },
            "queue_size"_a = uint32_t { 1 },
            "receiver_affinity"_a = std::vector<int> {},
            "device_start"_a = std::function<void()> {},
            "device_stop"_a = std::function<void()> {},
            "rename_metadata"_a = std::function<std::string(const std::string&)> {},
            "out_tensor_name"_a = ""s,
            "name"_a = "linux_receiver"s)
        .def("setup", &hololink::module::operators::LinuxReceiverOp::setup, "spec"_a)
        .def("start", &hololink::module::operators::LinuxReceiverOp::start)
        .def("stop", &hololink::module::operators::LinuxReceiverOp::stop);

    // The CSI -> Bayer converter operator is a holoscan::Operator AND a
    // CsiConverterV1, so it lists both as bases: it plugs into the pipeline
    // and is handed straight to a sensor's configure_converter(). Always
    // built (no ibverbs / FUSA dependency). The pixel-format arguments are
    // taken as plain integers (the enumerator value) rather than the bound
    // hololink_module.sensors.vb1940.PixelFormat enum, so the pure-Python
    // Imx274Cam driver can pass its csi.PixelFormat.value at the boundary
    // without importing the C++ enum.
    py::class_<hololink::module::operators::CsiToBayerOp,
        hololink::module::operators::PyCsiToBayerOp,
        holoscan::Operator,
        hololink::module::csi::CsiConverterV1,
        std::shared_ptr<hololink::module::operators::CsiToBayerOp>>(
        m, "CsiToBayerOp")
        .def(py::init<holoscan::Fragment*, const py::args&,
                 const std::shared_ptr<holoscan::Allocator>&,
                 int,
                 const std::string&,
                 uint32_t,
                 const std::string&>(),
            "fragment"_a,
            "allocator"_a,
            "cuda_device_ordinal"_a = 0,
            "out_tensor_name"_a = ""s,
            "sub_frame_rows"_a = uint32_t { 0 },
            "name"_a = "csi_to_bayer"s)
        .def("setup", &hololink::module::operators::CsiToBayerOp::setup, "spec"_a)
        .def("start", &hololink::module::operators::CsiToBayerOp::start)
        .def("stop", &hololink::module::operators::CsiToBayerOp::stop)
        .def(
            "configure",
            [](hololink::module::operators::CsiToBayerOp& self, uint32_t start_byte,
                uint32_t received_bytes_per_line, uint32_t pixel_width, uint32_t pixel_height,
                uint32_t pixel_format, uint32_t trailing_bytes) {
                self.configure(start_byte, received_bytes_per_line, pixel_width, pixel_height,
                    static_cast<hololink::module::csi::PixelFormat>(pixel_format), trailing_bytes);
            },
            "start_byte"_a, "received_bytes_per_line"_a, "pixel_width"_a, "pixel_height"_a,
            "pixel_format"_a, "trailing_bytes"_a = 0)
        .def("receiver_start_byte",
            &hololink::module::operators::CsiToBayerOp::receiver_start_byte)
        .def("received_line_bytes",
            &hololink::module::operators::CsiToBayerOp::received_line_bytes, "line_bytes"_a)
        .def(
            "transmitted_line_bytes",
            [](hololink::module::operators::CsiToBayerOp& self, uint32_t pixel_format,
                uint32_t pixel_width) {
                return self.transmitted_line_bytes(
                    static_cast<hololink::module::csi::PixelFormat>(pixel_format), pixel_width);
            },
            "pixel_format"_a, "pixel_width"_a)
        .def("get_csi_length", &hololink::module::operators::CsiToBayerOp::get_csi_length)
        .def("get_sub_frame_size",
            &hololink::module::operators::CsiToBayerOp::get_sub_frame_size);

    // The image-processing operator is a plain holoscan::Operator (not a
    // converter, so it lists no module interface base). Always built (no
    // ibverbs / FUSA dependency). pixel_format / bayer_format are plain
    // integer enumerator values, so the
    // pure-Python sensor drivers pass csi.PixelFormat.value / .BayerFormat at
    // the boundary with no enum coercion needed.
    py::class_<hololink::module::operators::ImageProcessorOp,
        hololink::module::operators::PyImageProcessorOp,
        holoscan::Operator,
        std::shared_ptr<hololink::module::operators::ImageProcessorOp>>(
        m, "ImageProcessorOp")
        .def(py::init<holoscan::Fragment*, const py::args&,
                 int,
                 int,
                 int32_t,
                 int,
                 const std::string&>(),
            "fragment"_a,
            "pixel_format"_a,
            "bayer_format"_a,
            "optical_black"_a = 0,
            "cuda_device_ordinal"_a = 0,
            "name"_a = "image_processor"s)
        .def("setup", &hololink::module::operators::ImageProcessorOp::setup, "spec"_a)
        .def("start", &hololink::module::operators::ImageProcessorOp::start)
        .def("stop", &hololink::module::operators::ImageProcessorOp::stop);

    // The packed-format converter is a holoscan::Operator AND a CsiConverterV1,
    // so it lists both as bases: it plugs into the pipeline and is handed to a
    // receiver's configure_converter(). Always built (no ibverbs / FUSA
    // dependency). The pixel-format arguments are taken as plain integers (the
    // enumerator value), matching CsiToBayerOp, so Python callers pass
    // csi.PixelFormat.value at the boundary without importing the C++ enum.
    py::class_<hololink::module::operators::PackedFormatConverterOp,
        hololink::module::operators::PyPackedFormatConverterOp,
        holoscan::Operator,
        hololink::module::csi::CsiConverterV1,
        std::shared_ptr<hololink::module::operators::PackedFormatConverterOp>>(
        m, "PackedFormatConverterOp")
        .def(py::init<holoscan::Fragment*, const py::args&,
                 const std::shared_ptr<holoscan::Allocator>&,
                 int,
                 const std::string&,
                 const std::string&,
                 const std::string&>(),
            "fragment"_a,
            "allocator"_a,
            "cuda_device_ordinal"_a = 0,
            "in_tensor_name"_a = ""s,
            "out_tensor_name"_a = ""s,
            "name"_a = "packed_format_converter"s)
        .def("setup", &hololink::module::operators::PackedFormatConverterOp::setup, "spec"_a)
        .def("start", &hololink::module::operators::PackedFormatConverterOp::start)
        .def("stop", &hololink::module::operators::PackedFormatConverterOp::stop)
        .def(
            "configure",
            [](hololink::module::operators::PackedFormatConverterOp& self, uint32_t start_byte,
                uint32_t received_bytes_per_line, uint32_t pixel_width, uint32_t pixel_height,
                uint32_t pixel_format, uint32_t trailing_bytes) {
                self.configure(start_byte, received_bytes_per_line, pixel_width, pixel_height,
                    static_cast<hololink::module::csi::PixelFormat>(pixel_format), trailing_bytes);
            },
            "start_byte"_a, "received_bytes_per_line"_a, "pixel_width"_a, "pixel_height"_a,
            "pixel_format"_a, "trailing_bytes"_a = 0)
        .def("receiver_start_byte",
            &hololink::module::operators::PackedFormatConverterOp::receiver_start_byte)
        .def("received_line_bytes",
            &hololink::module::operators::PackedFormatConverterOp::received_line_bytes, "line_bytes"_a)
        .def(
            "transmitted_line_bytes",
            [](hololink::module::operators::PackedFormatConverterOp& self, uint32_t pixel_format,
                uint32_t pixel_width) {
                return self.transmitted_line_bytes(
                    static_cast<hololink::module::csi::PixelFormat>(pixel_format), pixel_width);
            },
            "pixel_format"_a, "pixel_width"_a)
        .def("get_frame_size",
            &hololink::module::operators::PackedFormatConverterOp::get_frame_size);

    // NetworkReceiverFactory is registered opaquely (no constructor / methods)
    // only so make_*_network_receiver_factory()'s return value can be held in
    // Python and handed back to the HsbControllerOp constructor. The
    // application never touches its contents.
    py::class_<hololink::module::operators::NetworkReceiverFactory,
        std::shared_ptr<hololink::module::operators::NetworkReceiverFactory>>(
        m, "NetworkReceiverFactory");

    // The software (Linux) data plane is always built (it links no ibverbs), so
    // its factory is bound unconditionally — unlike the RoCE one below.
    m.def("make_linux_network_receiver_factory",
        &hololink::module::operators::make_linux_network_receiver_factory,
        "Build a NetworkReceiverFactory for the software (Linux) data plane, to "
        "pass as HsbControllerOp's network_receiver_factory.");

    // The sensor wrapper: subclassed in Python to build + arm one camera.
    // shared_ptr holder so SensorFactory::new_sensor's return loads across the
    // pybind boundary and co-owns the Python object.
    py::class_<hololink::module::operators::SensorDevice,
        hololink::module::operators::PySensorDevice,
        std::shared_ptr<hololink::module::operators::SensorDevice>>(
        m, "SensorDevice")
        .def(py::init<>())
        .def("stop_sensor", &hololink::module::operators::SensorDevice::stop_sensor);

    // The sensor-side controller: subclassed in Python (new_sensor) to supply
    // the camera. shared_ptr holder — HsbControllerOp holds it as a
    // shared_ptr<SensorFactory> parameter.
    py::class_<hololink::module::operators::SensorFactory,
        hololink::module::operators::PySensorFactory,
        std::shared_ptr<hololink::module::operators::SensorFactory>>(
        m, "SensorFactory")
        .def(py::init<>())
        .def("tap", &hololink::module::operators::SensorFactory::tap);

    // The reconnection controller operator. Transport-agnostic (always built);
    // the network_receiver_factory selects the data plane.
    py::class_<hololink::module::operators::HsbControllerOp,
        hololink::module::operators::PyHsbControllerOp,
        holoscan::Operator,
        std::shared_ptr<hololink::module::operators::HsbControllerOp>>(
        m, "HsbControllerOp")
        .def(py::init<holoscan::Fragment*, const py::args&,
                 hololink::module::EnumerationMetadata,
                 py::object,
                 size_t,
                 std::shared_ptr<hololink::module::operators::NetworkReceiverFactory>,
                 std::shared_ptr<hololink::module::operators::SensorFactory>,
                 size_t,
                 uint32_t,
                 uint32_t,
                 size_t,
                 double,
                 std::function<std::string(const std::string&)>,
                 const std::string&,
                 const std::string&>(),
            "fragment"_a,
            "enumeration_metadata"_a,
            "frame_context"_a,
            "frame_size"_a,
            "network_receiver_factory"_a,
            "sensor_factory"_a,
            "page_size"_a = size_t { 0 },
            "pages"_a = uint32_t { 2 },
            "queue_size"_a = uint32_t { 1 },
            "metadata_offset"_a = size_t { 0 },
            "watchdog_timeout_s"_a = double { 0.5 },
            "rename_metadata"_a = std::function<std::string(const std::string&)> {},
            "out_tensor_name"_a = ""s,
            "name"_a = "hsb_controller"s)
        .def("setup", &hololink::module::operators::HsbControllerOp::setup, "spec"_a)
        .def("start", &hololink::module::operators::HsbControllerOp::start)
        .def("stop", &hololink::module::operators::HsbControllerOp::stop);

#ifdef HOLOLINK_BUILD_ROCE
    m.def("make_roce_network_receiver_factory",
        &hololink::module::operators::make_roce_network_receiver_factory,
        "Build a NetworkReceiverFactory for the RoCE (ibverbs) data plane, to "
        "pass as HsbControllerOp's network_receiver_factory.");

    py::class_<hololink::module::operators::RoceReceiverOp,
        hololink::module::operators::PyRoceReceiverOp,
        holoscan::Operator,
        std::shared_ptr<hololink::module::operators::RoceReceiverOp>>(
        m, "RoceReceiverOp")
        .def(py::init<holoscan::Fragment*, const py::args&,
                 hololink::module::EnumerationMetadata,
                 py::object,
                 size_t,
                 size_t,
                 uint32_t,
                 uint32_t,
                 size_t,
                 std::function<void()>,
                 std::function<void()>,
                 std::function<std::string(const std::string&)>,
                 const std::string&,
                 const std::string&>(),
            "fragment"_a,
            "enumeration_metadata"_a,
            "frame_context"_a,
            "frame_size"_a,
            "page_size"_a = size_t { 0 },
            "pages"_a = uint32_t { 2 },
            "queue_size"_a = uint32_t { 1 },
            "metadata_offset"_a = size_t { 0 },
            "device_start"_a = std::function<void()> {},
            "device_stop"_a = std::function<void()> {},
            "rename_metadata"_a = std::function<std::string(const std::string&)> {},
            "out_tensor_name"_a = ""s,
            "name"_a = "roce_receiver"s)
        .def("setup", &hololink::module::operators::RoceReceiverOp::setup, "spec"_a)
        .def("start", &hololink::module::operators::RoceReceiverOp::start)
        .def("stop", &hololink::module::operators::RoceReceiverOp::stop);
#endif // HOLOLINK_BUILD_ROCE

#ifdef HOLOLINK_BUILD_FUSA
    // FusaCoeCaptureOp is a holoscan::Operator AND a CsiConverterV1 (a sensor
    // trains it via camera.configure_converter(fusa)), so it lists both bases
    // — matching CsiToBayerOp / PackedFormatConverterOp. Without the
    // CsiConverterV1 base pybind can't upcast it, and passing it to a
    // configure_converter(CsiConverterV1) parameter raises a TypeError.
    py::class_<hololink::module::operators::FusaCoeCaptureOp,
        hololink::module::operators::PyFusaCoeCaptureOp,
        holoscan::Operator,
        hololink::module::csi::CsiConverterV1,
        std::shared_ptr<hololink::module::operators::FusaCoeCaptureOp>>(
        m, "FusaCoeCaptureOp")
        .def(py::init<holoscan::Fragment*,
                 const py::args&,
                 hololink::module::EnumerationMetadata,
                 const std::string&,
                 const std::vector<uint8_t>&,
                 uint32_t,
                 py::object,
                 std::function<std::string(const std::string&)>,
                 const std::string&,
                 const std::string&>(),
            "fragment"_a,
            "enumeration_metadata"_a,
            "interface"_a,
            "mac_addr"_a,
            "timeout"_a,
            "device"_a,
            "rename_metadata"_a = std::function<std::string(const std::string&)> {},
            "out_tensor_name"_a = ""s,
            "name"_a = "fusa_coe_capture"s)
        .def("setup", &hololink::module::operators::FusaCoeCaptureOp::setup, "spec"_a)
        .def("start", &hololink::module::operators::FusaCoeCaptureOp::start)
        .def("stop", &hololink::module::operators::FusaCoeCaptureOp::stop)
        // pixel_format is taken as a plain integer (the enumerator value)
        // rather than the bound C++ enum, matching CsiToBayerOp /
        // PackedFormatConverterOp, so the pure-Python sensor drivers can pass
        // csi.PixelFormat.value at the boundary without importing the C++ enum.
        .def(
            "configure",
            [](hololink::module::operators::FusaCoeCaptureOp& self, uint32_t start_byte,
                uint32_t received_bytes_per_line, uint32_t pixel_width, uint32_t pixel_height,
                uint32_t pixel_format, uint32_t trailing_bytes) {
                self.configure(start_byte, received_bytes_per_line, pixel_width, pixel_height,
                    static_cast<hololink::module::csi::PixelFormat>(pixel_format), trailing_bytes);
            },
            "start_byte"_a, "received_bytes_per_line"_a, "pixel_width"_a, "pixel_height"_a,
            "pixel_format"_a, "trailing_bytes"_a = 0)
        .def("configure_frame_size",
            &hololink::module::operators::FusaCoeCaptureOp::configure_frame_size,
            "frame_size_bytes"_a)
        .def("receiver_start_byte",
            &hololink::module::operators::FusaCoeCaptureOp::receiver_start_byte)
        .def("received_line_bytes",
            &hololink::module::operators::FusaCoeCaptureOp::received_line_bytes,
            "line_bytes"_a)
        .def(
            "transmitted_line_bytes",
            [](hololink::module::operators::FusaCoeCaptureOp& self, uint32_t pixel_format,
                uint32_t pixel_width) {
                return self.transmitted_line_bytes(
                    static_cast<hololink::module::csi::PixelFormat>(pixel_format), pixel_width);
            },
            "pixel_format"_a, "pixel_width"_a)
        .def("configure_converter",
            &hololink::module::operators::FusaCoeCaptureOp::configure_converter,
            "converter"_a);
#endif // HOLOLINK_BUILD_FUSA

#ifdef HOLOLINK_MODULE_BUILD_CRC
    py::class_<hololink::module::operators::ComputeCrcOp,
        hololink::module::operators::PyComputeCrcOp,
        holoscan::Operator,
        std::shared_ptr<hololink::module::operators::ComputeCrcOp>>(
        m, "ComputeCrcOp")
        .def(py::init<holoscan::Fragment*, const py::args&,
                 int,
                 uint64_t,
                 const std::string&>(),
            "fragment"_a,
            "cuda_device_ordinal"_a = 0,
            "frame_size"_a = uint64_t { 0 },
            "name"_a = "compute_crc"s)
        .def("setup", &hololink::module::operators::ComputeCrcOp::setup, "spec"_a);

    py::class_<hololink::module::operators::CheckCrcOp,
        hololink::module::operators::PyCheckCrcOp,
        holoscan::Operator,
        std::shared_ptr<hololink::module::operators::CheckCrcOp>>(
        m, "CheckCrcOp")
        .def(py::init<holoscan::Fragment*, const py::args&,
                 std::shared_ptr<hololink::module::operators::ComputeCrcOp>,
                 const std::string&,
                 const std::string&>(),
            "fragment"_a,
            "compute_crc_op"_a,
            "computed_crc_metadata_name"_a = "computed_crc"s,
            "name"_a = "check_crc"s,
            py::keep_alive<0, 3>()) // keep compute_crc_op alive while CheckCrcOp lives
        .def("setup", &hololink::module::operators::CheckCrcOp::setup, "spec"_a);
#endif // HOLOLINK_MODULE_BUILD_CRC
}
