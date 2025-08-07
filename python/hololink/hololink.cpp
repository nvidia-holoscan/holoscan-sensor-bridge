/**
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

#include <hololink/common/gui_renderer.hpp>
#include <hololink/core/csi_controller.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/enumerator.hpp>
#include <hololink/core/jesd.hpp>
#include <hololink/core/logging_internal.hpp>
#include <hololink/core/nvtx_trace.hpp>
#include <hololink/core/timeout.hpp>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

/**
 * Make the data type we use to pass binary data between C++ and Python opaque. With that pybind11
 * is not copying when converting between Python types and this C++ type. See
 * https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html#making-opaque-types for more
 * information.
 */
PYBIND11_MAKE_OPAQUE(std::vector<uint8_t>);

namespace pybind11 {

using vector_uint8_class = py::class_<std::vector<uint8_t>, std::unique_ptr<std::vector<uint8_t>>>;

/*
 * Pybind11 is using the ostream << operator for printing the uint8_t vector. This expects utf-8
 * encoded data but we are using this for binary data. Create a specialication of the template
 * adding the `__repr__` operator to print hex binary data.
 */
template <>
auto detail::vector_if_insertion_operator<std::vector<uint8_t>, vector_uint8_class>(
    vector_uint8_class& cl, std::string const& name) -> decltype(std::declval<std::ostream&>()
        << std::declval<uint8_t>(),
    void())
{
    cl.def(
        "__repr__",
        [name](std::vector<uint8_t>& v) {
            return fmt::format("{}[{:02x}]", name, fmt::join(v, ", "));
        },
        "Return the canonical string representation of this list.");
}

} // namespace pybind11

namespace hololink {

// Pybind11 creates default values on startup, this is bad for timeouts since they start ticking
// when they are created. Therefore use this object to check if the default timeout for the function
// needs to be created. This will be done when the function is called so that the timeout starts
// only then.
static const std::shared_ptr<Timeout> default_timeout = std::make_shared<Timeout>(1.f);

/**
 * The trampoline class of Hololink allows overriding virtual functions in Python.
 *
 * See
 * https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python.
 */
class PyHololink : public Hololink {
public:
    /* Inherit the constructors */
    using Hololink::Hololink;

    /* Trampoline (need one for each virtual function) */
    void executed(double request_time, const std::vector<uint8_t>& request, double reply_time,
        const std::vector<uint8_t>& reply) override
    {
        PYBIND11_OVERRIDE(void, /* Return type */
            Hololink, /* Parent class */
            executed, /* Name of function in C++ (must match Python name) */
            request_time, request, reply_time, reply /* Argument(s) */
        );
    }

    void send_control(const std::vector<uint8_t>& request) override
    {
        PYBIND11_OVERRIDE(void, /* Return type */
            Hololink, /* Parent class */
            send_control, /* Name of function in C++ (must match Python name) */
            request /* Argument(s) */
        );
    }

    std::vector<uint8_t> receive_control(const std::shared_ptr<Timeout>& timeout) override
    {
        PYBIND11_OVERRIDE(std::vector<uint8_t>, /* Return type */
            Hololink, /* Parent class */
            receive_control, /* Name of function in C++ (must match Python name) */
            timeout /* Argument(s) */
        );
    }
};

// This is only included here for flash memory programming on
// older (2502 and older) FPGAs; so this prototype isn't included
// in a public header file anywhere.
std::shared_ptr<Hololink::Spi> get_traditional_spi(Hololink& hololink, uint32_t spi_address, uint32_t chip_select,
    uint32_t clock_divisor, uint32_t cpol, uint32_t cpha, uint32_t width);

// This is only included here for FPGAs with older I2C interfaces
// (like the MPF200 from Microchip); because this interface is deprecated
// we don't have it in a public header file anywhere.
std::shared_ptr<Hololink::I2c> get_traditional_i2c(Hololink& hololink, uint32_t i2c_address);

PYBIND11_MODULE(_hololink, m)
{
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    // create a opaque type for the type we use to pass binary data
    py::bind_vector<std::vector<uint8_t>>(m, "VectorUInt8", py::buffer_protocol());
    // enable implicit conversion for the vector type
    py::implicitly_convertible<py::list, std::vector<uint8_t>>();
    py::implicitly_convertible<py::bytes, std::vector<uint8_t>>();
    py::implicitly_convertible<py::bytearray, std::vector<uint8_t>>();

    // This class handle timeouts and retries for bus transactions.
    py::class_<Timeout, std::shared_ptr<Timeout>>(m, "Timeout")
        .def(py::init<float, float>(), "timeout_s"_a, "retry_s"_a = 0.f)
        .def_static(
            "default_timeout", &Timeout::default_timeout, "timeout"_a = std::shared_ptr<Timeout>())
        .def_static("i2c_timeout", &Timeout::i2c_timeout, "timeout"_a = std::shared_ptr<Timeout>())
        .def_static("spi_timeout", &Timeout::spi_timeout, "timeout"_a = std::shared_ptr<Timeout>())
        .def_static("now_s", &Timeout::now_s)
        .def_static("now_ns", &Timeout::now_ns)
        .def("expired", &Timeout::expired)
        .def("trigger_s", &Timeout::trigger_s)
        .def("retry", &Timeout::retry);

    // Present Metadata as an unmutable dictionary to Python
    py::class_<Metadata, std::shared_ptr<Metadata>>(m, "Metadata")
        .def(py::init())
        /**
         * @brief Create from Python dict
         *
         * @param dict
         */
        .def(py::init([](py::dict dict) {
            auto metadata = std::make_shared<Metadata>();
            for (auto it = dict.begin(); it != dict.end(); ++it) {
                if (py::isinstance<py::str>(it->second)) {
                    (*metadata)[it->first.cast<std::string>()] = it->second.cast<std::string>();
                } else if (py::isinstance<py::int_>(it->second)) {
                    (*metadata)[it->first.cast<std::string>()] = it->second.cast<int64_t>();
                }
            }
            return metadata;
        }))
        .def(py::init([](const Metadata& source) {
            auto metadata = std::make_shared<Metadata>(source);
            return metadata;
        }))
        /**
         * @returns an iterator object that can iterate over all objects in Metadata
         */
        .def(
            "__iter__", [](const Metadata& me) { return py::make_iterator(me.begin(), me.end()); },
            py::keep_alive<0, 1>())
        .def("__getitem__", [](Metadata& me, const std::string& name) {
            std::optional<const Metadata::Element> val = me.get<Metadata::Element>(name);
            if (!val.has_value()) {
                throw py::key_error(name);
            }
            return val;
        })
        .def(
            "get", [](Metadata& me, const std::string& name) {
                std::optional<const Metadata::Element> val = me.get<Metadata::Element>(name);
                // pybind11 maps empty optional to None in Python
                if (!val.has_value()) {
                    return std::optional<const Metadata::Element>();
                }
                return val;
            },
            "name"_a)
        /**
         * @brief Get with default
         *
         * @param name
         * @param value
         */
        .def(
            "get", [](Metadata& me, const std::string& name, const Metadata::Element& value) {
                std::optional<const Metadata::Element> element = me.get<Metadata::Element>(name);
                if (!element.has_value()) {
                    // return new optional value created from default 'value'. not inserted into Metadata
                    return std::optional<const Metadata::Element>(value);
                }
                return element;
            },
            "name"_a, "value"_a)
        .def("__repr__", [](const Metadata& metadata) { return fmt::format("{}", metadata); })
        .def("update", &Metadata::update, "other"_a);

    py::class_<Enumerator, std::shared_ptr<Enumerator>>(m, "Enumerator")
        .def(py::init<const std::string&, uint32_t, uint32_t>(),
            "local_interface"_a = std::string(),
            "bootp_request_port"_a = 12267u, "bootp_reply_port"_a = 12268u)
        .def_static("enumerated", &Enumerator::enumerated, "call_back"_a,
            "timeout"_a = std::shared_ptr<Timeout>())
        .def_static(
            "find_channel",
            [](const std::string& channel_ip, const std::shared_ptr<Timeout>& timeout) {
                // check if the default timeout should be used and then create a new timeout
                // starting now
                return Enumerator::find_channel(channel_ip,
                    timeout == default_timeout ? std::make_shared<Timeout>(20.f) : timeout);
            },
            "channel_ip"_a, "timeout"_a = default_timeout)
        .def("enumeration_packets", &Enumerator::enumeration_packets, "call_back"_a,
            "timeout"_a = std::shared_ptr<Timeout>())
        .def("send_bootp_reply", &Enumerator::send_bootp_reply, "peer_address"_a, "reply_packet"_a,
            "metadata"_a)
        .def_static("set_uuid_strategy", &Enumerator::set_uuid_strategy,
            "uuid"_a, "enumeration_strategy"_a);

    m.attr("APB_RAM") = APB_RAM;
    m.attr("BL_I2C_BUS") = BL_I2C_BUS;
    m.attr("CAM_I2C_BUS") = CAM_I2C_BUS;
    m.attr("CLNX_SPI_BUS") = CLNX_SPI_BUS;
    m.attr("CTRL_EVENT") = CTRL_EVENT;
    m.attr("CTRL_EVT_SW_EVENT") = CTRL_EVT_SW_EVENT;
    m.attr("CPNX_SPI_BUS") = CPNX_SPI_BUS;
    m.attr("DP_ADDRESS_0") = DP_ADDRESS_0;
    m.attr("DP_ADDRESS_1") = DP_ADDRESS_1;
    m.attr("DP_ADDRESS_2") = DP_ADDRESS_2;
    m.attr("DP_ADDRESS_3") = DP_ADDRESS_3;
    m.attr("DP_BUFFER_LENGTH") = DP_BUFFER_LENGTH;
    m.attr("DP_BUFFER_MASK") = DP_BUFFER_MASK;
    m.attr("DP_HOST_IP") = DP_HOST_IP;
    m.attr("DP_HOST_MAC_HIGH") = DP_HOST_MAC_HIGH;
    m.attr("DP_HOST_MAC_LOW") = DP_HOST_MAC_LOW;
    m.attr("DP_HOST_UDP_PORT") = DP_HOST_UDP_PORT;
    m.attr("DP_PACKET_SIZE") = DP_PACKET_SIZE;
    m.attr("DP_QP") = DP_QP;
    m.attr("DP_RKEY") = DP_RKEY;
    m.attr("DP_VP_MASK") = DP_VP_MASK;
    m.attr("FPGA_DATE") = FPGA_DATE;
    m.attr("HOLOLINK_100G_BOARD_ID") = HOLOLINK_100G_BOARD_ID;
    m.attr("HOLOLINK_LITE_BOARD_ID") = HOLOLINK_LITE_BOARD_ID;
    m.attr("HOLOLINK_NANO_BOARD_ID") = HOLOLINK_NANO_BOARD_ID;
    m.attr("HSB_IP_VERSION") = HSB_IP_VERSION;
    m.attr("I2C_10B_ADDRESS") = I2C_10B_ADDRESS;
    m.attr("I2C_CTRL") = I2C_CTRL;
    m.attr("I2C_BUSY") = I2C_BUSY;
    m.attr("I2C_DONE") = I2C_DONE;
    m.attr("I2C_FSM_ERR") = I2C_FSM_ERR;
    m.attr("I2C_I2C_ERR") = I2C_I2C_ERR;
    m.attr("I2C_I2C_NAK") = I2C_I2C_NAK;
    m.attr("I2C_REG_BUS_EN") = I2C_REG_BUS_EN;
    m.attr("I2C_REG_CLK_CNT") = I2C_REG_CLK_CNT;
    m.attr("I2C_REG_CONTROL") = I2C_REG_CONTROL;
    m.attr("I2C_REG_DATA_BUFFER") = I2C_REG_DATA_BUFFER;
    m.attr("I2C_REG_NUM_BYTES") = I2C_REG_NUM_BYTES;
    m.attr("I2C_REG_STATUS") = I2C_REG_STATUS;
    m.attr("I2C_START") = I2C_START;
    m.attr("LEOPARD_EAGLE_BOARD_ID") = LEOPARD_EAGLE_BOARD_ID;
    m.attr("METADATA_SIZE") = METADATA_SIZE;
    m.attr("MICROCHIP_POLARFIRE_BOARD_ID") = MICROCHIP_POLARFIRE_BOARD_ID;
    m.attr("RD_DWORD") = RD_DWORD;
    m.attr("REQUEST_FLAGS_ACK_REQUEST") = REQUEST_FLAGS_ACK_REQUEST;
    m.attr("RESPONSE_INVALID_CMD") = RESPONSE_INVALID_CMD;
    m.attr("RESPONSE_SUCCESS") = RESPONSE_SUCCESS;
    m.attr("SPI_CTRL") = SPI_CTRL;
    m.attr("WR_DWORD") = WR_DWORD;

    // Bind PixelFormat enum
    py::enum_<hololink::csi::PixelFormat>(m, "PixelFormat")
        .value("RAW_8", hololink::csi::PixelFormat::RAW_8, R"pbdoc(RAW 8-bit)pbdoc")
        .value("RAW_10", hololink::csi::PixelFormat::RAW_10, R"pbdoc(RAW 10-bit)pbdoc")
        .value("RAW_12", hololink::csi::PixelFormat::RAW_12, R"pbdoc(RAW 12-bit)pbdoc");

    // Bind BayerFormat enum
    py::enum_<hololink::csi::BayerFormat>(m, "BayerFormat")
        .value("RGGB", csi::BayerFormat::RGGB)
        .value("BGGR", csi::BayerFormat::BGGR)
        .value("GBRG", csi::BayerFormat::GBRG)
        .value("GRBG", csi::BayerFormat::GRBG);

    py::class_<DataChannel, std::shared_ptr<DataChannel>>(m, "DataChannel")
        .def(py::init<const Metadata&>(), "metadata"_a)
        .def(py::init<const Metadata&,
                 const std::function<std::shared_ptr<Hololink>(const Metadata&)>&>(),
            "metadata"_a, "create_hololink"_a)
        .def_static("enumerated", &DataChannel::enumerated, "metadata"_a)
        .def("hololink", &DataChannel::hololink)
        .def("peer_ip", &DataChannel::peer_ip)
        .def("authenticate", &DataChannel::authenticate, "qp_number"_a, "rkey"_a)
        .def("configure_roce", &DataChannel::configure_roce, "frame_memory"_a, "frame_size"_a, "page_size"_a, "pages"_a, "local_data_port"_a)
        .def("configure_coe", &DataChannel::configure_coe, "channel"_a, "frame_size"_a, "pixel_width"_a, "vlan_enabled"_a = false)
        .def("disable_packetizer", &DataChannel::disable_packetizer)
        .def("enable_packetizer_10", &DataChannel::enable_packetizer_10)
        .def("unconfigure", &DataChannel::unconfigure)
        .def_static("use_multicast", &DataChannel::use_multicast, "metadata"_a, "address"_a, "port"_a)
        .def_static("use_broadcast", &DataChannel::use_broadcast, "metadata"_a, "port"_a)
        .def("configure_socket", &DataChannel::configure_socket, "socket_fd"_a)
        .def_static("use_sensor", &DataChannel::use_sensor, "metadata"_a, "sensor_number"_a)
        .def("frame_end_sequencer", &DataChannel::frame_end_sequencer)
        .def_static("use_data_plane_configuration", &DataChannel::use_data_plane_configuration, "metadata"_a, "data_plane"_a)
        .def("enumeration_metadata", &DataChannel::enumeration_metadata);

    py::register_exception<TimeoutError>(m, "TimeoutError");
    py::register_exception<UnsupportedVersion>(m, "UnsupportedVersion");

    auto hololink_module = py::class_<Hololink, PyHololink, std::shared_ptr<Hololink>>(m, "Hololink")
                               .def(py::init<const std::string&, uint32_t, const std::string&, bool>(), "peer_ip"_a,
                                   "control_port"_a, "serial_number"_a, "sequence_number_checking"_a)
                               .def_static("from_enumeration_metadata", &Hololink::from_enumeration_metadata, "metadata"_a)
                               .def_static("reset_framework", &Hololink::reset_framework)
                               .def_static("enumerated", &Hololink::enumerated, "metadata"_a)
                               .def("start", &Hololink::start)
                               .def("stop", &Hololink::stop)
                               .def("reset", &Hololink::reset)
                               .def("get_hsb_ip_version", &Hololink::get_hsb_ip_version,
                                   "timeout"_a = std::shared_ptr<Timeout>(), "check_sequence"_a = true)
                               .def("get_fpga_date", &Hololink::get_fpga_date)
                               .def(
                                   "write_uint32",
                                   [](Hololink& me, uint32_t address, uint32_t value, const std::shared_ptr<Timeout>& timeout, bool retry) {
                                       return me.write_uint32(address, value, timeout, retry);
                                   },
                                   "address"_a, "value"_a, "timeout"_a = std::shared_ptr<Timeout>(), "retry"_a = true)
                               .def(
                                   "read_uint32",
                                   [](Hololink& me, uint32_t address, const std::shared_ptr<Timeout>& timeout) {
                                       return me.read_uint32(address, timeout);
                                   },
                                   "address"_a, "timeout"_a = std::shared_ptr<Timeout>())
                               .def("setup_clock", &Hololink::setup_clock, "clock_profile"_a)
                               .def("get_i2c", &Hololink::get_i2c, "i2c_bus"_a, "i2c_address"_a = I2C_CTRL)
                               .def("get_spi", &Hololink::get_spi, "bus_address"_a, "chip_select"_a,
                                   "prescaler"_a = 0x0F, "cpol"_a = 1, "cpha"_a = 1, "width"_a = 1, "spi_address"_a = SPI_CTRL)
                               .def("get_gpio", &Hololink::get_gpio, "metadata"_a)
                               .def("send_control", &Hololink::send_control)
                               .def("receive_control", &Hololink::receive_control, "timeout"_a)
                               .def(
                                   "on_reset",
                                   [](Hololink& me, py::object py_reset_callback) {
                                       // Handling unbound py_reset_callback instances
                                       // is deferred to a later time.
                                       if (!py::hasattr(py_reset_callback, "__self__")) {
                                           throw std::runtime_error("on_reset only works with bound methods.\n");
                                       }
                                       // Given
                                       //  - py_reset_callback is a bound method
                                       //  - Hololink::ResetController is an object that
                                       //      will be owned by the Hololink instance--
                                       //      this object's lifetime is tied to the lifetime of
                                       //      the hololink that holds it
                                       //  - we don't want to leak the python object that our
                                       //      method is bound to--we want python to collect
                                       //      that object when it goes out of scope.
                                       //
                                       // BoundResetController adapts these two interfaces,
                                       //  providing a C++ ResetController instance whose lifetime
                                       //  goes with the Hololink instance, and does the right thing
                                       //  if the object to which the bound method has been collected.
                                       //
                                       // Note that python's bound methods are a bit tricky:
                                       //  if we just inc_ref py_reset_callback, then we'll leak the
                                       //  object to which the method is bound (breaking the third
                                       //  requirement).  We can't just create a weak reference
                                       //  to the bound method instance, as python always immediately
                                       //  collects these things (when there's no outstanding references).
                                       //
                                       // We work around this by taking the bound method apart, fetching
                                       //   the "__self__" field (which is the object to which we're
                                       //   bound) and the "__func__" field (which is an unbound function).
                                       //   We'll then get a weak reference to __self__, and whenever
                                       //   our reset() call occurs, and the weak __self__ reference still
                                       //   finds a legit __self__ instance, we'll call __func__(__self__).
                                       //
                                       // At this time, we don't remove ResetController instances from the
                                       //  hololink instance; so calls to hololink.reset() will poll all
                                       //  known sensors even after they're known to be gone.  (It knows
                                       //  not to call reset on those dead references.)  Changing this
                                       //  doesn't seem worthwhile: reset() is a very rare call (typically
                                       //  only once at app startup) and it's not expected that sensor
                                       //  instances are collected either (typically they live the
                                       //  entire lifetime of the application).
                                       class BoundResetController : public Hololink::ResetController {
                                       public:
                                           BoundResetController(py::object py_reset_callback)
                                           {
                                               // Get a weak reference to the object to which
                                               // the callback is bound.
                                               PyObject* self = py_reset_callback.attr("__self__").ptr();
                                               weak_self_ = PyWeakref_NewRef(self, nullptr);
                                               // At reset time, call this guy with the self object from above.
                                               func_ = py_reset_callback.attr("__func__").ptr();
                                               Py_XINCREF(func_);
                                           }
                                           ~BoundResetController()
                                           {
                                               py::gil_scoped_acquire gil;
                                               Py_XDECREF(func_);
                                               Py_XDECREF(weak_self_);
                                           }
                                           void reset() override
                                           {
                                               py::gil_scoped_acquire gil;
                                               PyObject* self = PyWeakref_GetObject(weak_self_);
                                               if (self == Py_None) {
                                                   // "self" was gc'd, so don't call it.
                                                   return;
                                               }
                                               PyObject* r = PyObject_CallOneArg(func_, self);
                                               Py_DECREF(r);
                                           }

                                       public:
                                           PyObject* weak_self_;
                                           PyObject* func_;
                                       };
                                       std::shared_ptr reset_controller = std::make_shared<BoundResetController>(py_reset_callback);
                                       me.on_reset(reset_controller);
                                   },
                                   "reset_controller"_a)
                               .def(
                                   "ptp_synchronize", [](Hololink& me, const std::shared_ptr<Timeout>& timeout) { return me.ptp_synchronize(timeout); }, "timeout_s"_a)
                               .def("ptp_synchronize", [](Hololink& me) { return me.ptp_synchronize(); })
                               .def("ptp_pps_output", &Hololink::ptp_pps_output, "frequency"_a = 0)
                               .def("configure_apb_event", &Hololink::configure_apb_event, "event"_a, "handler"_a, "rising_edge"_a = true)
                               .def("clear_apb_event", &Hololink::clear_apb_event, "event"_a);

    py::class_<Hololink::I2c, std::shared_ptr<Hololink::I2c>>(m, "I2c")
        .def("i2c_transaction",
            &Hololink::I2c::i2c_transaction, "peripheral_i2c_address"_a, "write_bytes"_a,
            "read_byte_count"_a, "timeout"_a = std::shared_ptr<Timeout>(),
            "ignore_nak"_a = false)
        .def("encode_i2c_request", &Hololink::I2c::encode_i2c_request,
            "sequencer"_a,
            "peripheral_i2c_address"_a,
            "write_bytes"_a,
            "read_byte_count"_a);

    py::class_<Hololink::Spi, std::shared_ptr<Hololink::Spi>>(m, "Spi").def("spi_transaction",
        &Hololink::Spi::spi_transaction, "peripheral_i2c_address"_a, "write_bytes"_a,
        "read_byte_count"_a, "timeout"_a = std::shared_ptr<Timeout>());

    py::enum_<Hololink::Event>(hololink_module, "Event")
        .value("SW_EVENT", Hololink::Event::SW_EVENT)
        .value("SIF_0_FRAME_END", Hololink::Event::SIF_0_FRAME_END)
        .value("SIF_1_FRAME_END", Hololink::Event::SIF_1_FRAME_END);

    auto gpio = py::class_<Hololink::GPIO, std::shared_ptr<Hololink::GPIO>>(m, "GPIO")
                    .def("set_direction", &Hololink::GPIO::set_direction, "pin"_a, "direction"_a)
                    .def("get_direction", &Hololink::GPIO::get_direction, "pin"_a)
                    .def("set_value", &Hololink::GPIO::set_value, "pin"_a, "value"_a)
                    .def("get_value", &Hololink::GPIO::get_value, "pin"_a)
                    .def("get_supported_pin_num", &Hololink::GPIO::get_supported_pin_num);
    gpio.attr("IN") = Hololink::GPIO::IN;
    gpio.attr("OUT") = Hololink::GPIO::OUT;
    gpio.attr("LOW") = Hololink::GPIO::LOW;
    gpio.attr("HIGH") = Hololink::GPIO::HIGH;
    gpio.attr("GPIO_PIN_RANGE") = Hololink::GPIO::GPIO_PIN_RANGE;

    py::class_<AD9986Config, std::shared_ptr<AD9986Config>>(m, "AD9986Config")
        .def(py::init<Hololink&>(), "hololink"_a)
        .def("host_pause_mapping", &AD9986Config::host_pause_mapping, "mask"_a)
        .def("apply", &AD9986Config::apply);

    py::class_<core::NvtxTrace, std::shared_ptr<core::NvtxTrace>>(m, "NvtxTrace")
        .def_static("setThreadName", &core::NvtxTrace::setThreadName, "threadName"_a)
        .def_static("event_u64", &core::NvtxTrace::event_u64, "message"_a, "datum"_a);

    // The ImGuiRenderer is used by the SignalGeneratorOp and the SignalViewerOp operators.
    py::class_<ImGuiRenderer, std::shared_ptr<ImGuiRenderer>>(m, "ImGuiRenderer")
        .def(py::init<>());

    //
    auto csi_converter = py::class_<csi::CsiConverter, std::shared_ptr<csi::CsiConverter>>(m, "CsiConverter")
                             .def("configure", &csi::CsiConverter::configure, "start_byte"_a, "recieved_bytes_per_line"_a, "pixel_width"_a, "pixel_height"_a, "pixel_format"_a, "trailing_bytes"_a = 0)
                             .def("receiver_start_byte", &csi::CsiConverter::receiver_start_byte)
                             .def("received_line_bytes", &csi::CsiConverter::received_line_bytes, "line_bytes"_a)
                             .def("transmitted_line_bytes", &csi::CsiConverter::transmitted_line_bytes, "pixel_format"_a, "pixel_width"_a);

    // Support for legacy I2C interfaces; we only use this on FPGAs
    // that aren't updated (e.g. MPF200)
    m.def("get_traditional_i2c", &get_traditional_i2c, "hololink"_a, "i2c_address"_a);

    // Support for legacy SPI interfaces; we only use this to write
    // the flash memory of older HSB FPGAs (0x2502 and earlier)
    m.def("get_traditional_spi", &get_traditional_spi,
        "hololink"_a, "spi_address"_a, "chip_select"_a, "clock_divisor"_a,
        "cpol"_a, "cpha"_a, "width"_a);

    auto sequencer = py::class_<Hololink::Sequencer, std::shared_ptr<Hololink::Sequencer>>(m, "Sequencer")
                         .def("write_uint32", &Hololink::Sequencer::write_uint32, "address"_a, "data"_a)
                         .def("read_uint32", &Hololink::Sequencer::read_uint32, "address"_a, "initial_value"_a = 0xFFFFFFFF)
                         .def("poll", &Hololink::Sequencer::poll, "address"_a, "mask"_a, "match"_a)
                         .def("enable", &Hololink::Sequencer::enable);

    py::enum_<Hololink::Sequencer::Op>(sequencer, "Op")
        .value("POLL", Hololink::Sequencer::Op::POLL)
        .value("RD", Hololink::Sequencer::Op::RD)
        .value("WR", Hololink::Sequencer::Op::WR);

    py::class_<EnumerationStrategy, std::shared_ptr<EnumerationStrategy>>(m, "EnumerationStrategy");

    py::class_<BasicEnumerationStrategy, std::shared_ptr<BasicEnumerationStrategy>, EnumerationStrategy>(m, "BasicEnumerationStrategy")
        .def(py::init<const Metadata&, unsigned, unsigned, unsigned>(),
            "additional_metadata"_a, "total_sensors"_a = 2, "total_dataplanes"_a = 2, "sifs_per_sensor"_a = 2)
        .def(py::init([](unsigned total_sensors, unsigned total_dataplanes, unsigned sifs_per_sensor) {
            Metadata empty_metadata;
            return std::make_shared<BasicEnumerationStrategy>(empty_metadata, total_sensors, total_dataplanes, sifs_per_sensor);
        }),
            "total_sensors"_a = 2, "total_dataplanes"_a = 2, "sifs_per_sensor"_a = 2);

    // Trampoline class for Synchronizable to allow Python subclasses
    class PySynchronizable : public Synchronizable {
    public:
        using Synchronizable::Synchronizable;
    };

    // Bind Synchronizable class
    py::class_<Synchronizable, PySynchronizable, std::shared_ptr<Synchronizable>>(m, "Synchronizable")
        .def(py::init<>());

    // Bind Synchronizer class
    py::class_<Synchronizer, std::shared_ptr<Synchronizer>>(m, "Synchronizer")
        .def_static("null_synchronizer", &Synchronizer::null_synchronizer)
        .def("attach", &Synchronizer::attach, "peer"_a)
        .def("detach", &Synchronizer::detach, "peer"_a)
        .def("setup", &Synchronizer::setup)
        .def("shutdown", &Synchronizer::shutdown)
        .def("is_enabled", &Synchronizer::is_enabled);

} // PYBIND11_MODULE

} // namespace hololink
