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

#include <hololink/data_channel.hpp>
#include <hololink/enumerator.hpp>
#include <hololink/timeout.hpp>

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
};

PYBIND11_MODULE(_hololink, m)
{
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    // create a opaque type for the type we use to pass binary data
    py::bind_vector<std::vector<uint8_t>>(m, "VectorUInt8", py::buffer_protocol());
    // enable implict conversion for the vector type
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

    // Present Metadata as an unmutable dictonary to Python
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
        /**
         * @returns an iterator object that can iterate over all objects in Metadata
         */
        .def(
            "__iter__", [](const Metadata& me) { return py::make_iterator(me.begin(), me.end()); },
            py::keep_alive<0, 1>())
        .def("__getitem__", [](Metadata& me, const char* name) { return me[name]; })
        .def("get", &Metadata::get<Metadata::Element>, "name"_a)
        /**
         * @brief Get with default
         *
         * @param name
         * @param value
         */
        .def(
            "get",
            [](Metadata& me, const std::string& name, const std::string& value) {
                std::optional<Metadata::Element> element = me.get<Metadata::Element>(name);
                if (!element.has_value()) {
                    element.value() = value;
                }
                return element;
            },
            "name"_a, "value"_a)
        .def("__repr__", [](const Metadata& metadata) { return fmt::format("{}", metadata); });

    py::class_<Enumerator, std::shared_ptr<Enumerator>>(m, "Enumerator")
        .def(py::init<const std::string&, uint32_t, uint32_t, uint32_t>(),
            "local_interface"_a = std::string(), "enumeration_port"_a = 10001u,
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
            "metadata"_a);

    m.attr("CLNX_SPI_CTRL") = CLNX_SPI_CTRL;
    m.attr("CPNX_SPI_CTRL") = CPNX_SPI_CTRL;
    m.attr("BL_I2C_CTRL") = BL_I2C_CTRL;
    m.attr("CAM_I2C_CTRL") = CAM_I2C_CTRL;
    m.attr("WR_DWORD") = WR_DWORD;
    m.attr("RD_DWORD") = RD_DWORD;
    m.attr("REQUEST_FLAGS_ACK_REQUEST") = REQUEST_FLAGS_ACK_REQUEST;
    m.attr("RESPONSE_SUCCESS") = RESPONSE_SUCCESS;
    m.attr("RESPONSE_INVALID_CMD") = RESPONSE_INVALID_CMD;
    m.attr("I2C_START") = I2C_START;
    m.attr("I2C_CORE_EN") = I2C_CORE_EN;
    m.attr("I2C_DONE_CLEAR") = I2C_DONE_CLEAR;
    m.attr("I2C_BUSY") = I2C_BUSY;
    m.attr("I2C_DONE") = I2C_DONE;
    m.attr("FPGA_VERSION") = FPGA_VERSION;
    m.attr("FPGA_DATE") = FPGA_DATE;
    m.attr("DP_PACKET_SIZE") = DP_PACKET_SIZE;
    m.attr("DP_HOST_MAC_LOW") = DP_HOST_MAC_LOW;
    m.attr("DP_HOST_MAC_HIGH") = DP_HOST_MAC_HIGH;
    m.attr("DP_HOST_IP") = DP_HOST_IP;
    m.attr("DP_HOST_UDP_PORT") = DP_HOST_UDP_PORT;
    m.attr("DP_VIP_MASK") = DP_VIP_MASK;
    m.attr("DP_ROCE_CFG") = DP_ROCE_CFG;
    m.attr("DP_ROCE_RKEY_0") = DP_ROCE_RKEY_0;
    m.attr("DP_ROCE_VADDR_MSB_0") = DP_ROCE_VADDR_MSB_0;
    m.attr("DP_ROCE_VADDR_LSB_0") = DP_ROCE_VADDR_LSB_0;
    m.attr("DP_ROCE_BUF_END_MSB_0") = DP_ROCE_BUF_END_MSB_0;
    m.attr("DP_ROCE_BUF_END_LSB_0") = DP_ROCE_BUF_END_LSB_0;
    m.attr("HOLOLINK_LITE_BOARD_ID") = HOLOLINK_LITE_BOARD_ID;
    m.attr("HOLOLINK_BOARD_ID") = HOLOLINK_BOARD_ID;
    m.attr("HOLOLINK_100G_BOARD_ID") = HOLOLINK_100G_BOARD_ID;

    py::class_<DataChannel, std::shared_ptr<DataChannel>>(m, "DataChannel")
        .def(py::init<const Metadata&>(), "metadata"_a)
        .def(py::init<const Metadata&,
                 const std::function<std::shared_ptr<Hololink>(const Metadata&)>&>(),
            "metadata"_a, "create_hololink"_a)
        .def_static("enumerated", &DataChannel::enumerated, "metadata"_a)
        .def("hololink", &DataChannel::hololink)
        .def("peer_ip", &DataChannel::peer_ip)
        .def("authenticate", &DataChannel::authenticate, "qp_number"_a, "rkey"_a)
        .def("configure", &DataChannel::configure, "frame_address"_a, "frame_size"_a,
            "local_data_port"_a)
        .def("write_uint32", &DataChannel::write_uint32, "address"_a, "value"_a);

    py::register_exception<TimeoutError>(m, "TimeoutError");
    py::register_exception<UnsupportedVersion>(m, "UnsupportedVersion");

    py::class_<Hololink, PyHololink, std::shared_ptr<Hololink>>(m, "Hololink")
        .def(py::init<const std::string&, uint32_t, const std::string&>(), "peer_ip"_a,
            "control_port"_a, "serial_number"_a)
        .def_static("from_enumeration_metadata", &Hololink::from_enumeration_metadata, "metadata"_a)
        .def_static("reset_framework", &Hololink::reset_framework)
        .def_static("enumerated", &Hololink::enumerated, "metadata"_a)
        .def("csi_size", &Hololink::csi_size)
        .def("start", &Hololink::start)
        .def("stop", &Hololink::stop)
        .def("reset", &Hololink::reset)
        .def("get_fpga_version", &Hololink::get_fpga_version,
            "timeout"_a = std::shared_ptr<Timeout>())
        .def("get_fpga_date", &Hololink::get_fpga_date)
        .def("write_uint32", &Hololink::write_uint32, "address"_a, "value"_a,
            "timeout"_a = std::shared_ptr<Timeout>(), "retry"_a = true)
        .def("read_uint32", &Hololink::read_uint32, "address"_a,
            "timeout"_a = std::shared_ptr<Timeout>())
        .def("setup_clock", &Hololink::setup_clock, "clock_profile"_a)
        .def("get_i2c", &Hololink::get_i2c, "i2c_address"_a)
        .def("get_spi", &Hololink::get_spi, "spi_address"_a, "chip_select"_a,
            "clock_divisor"_a = 0x0F, "cpol"_a = 1, "cpha"_a = 1, "width"_a = 1)
        .def("get_gpio", &Hololink::get_gpio)
        .def("send_control", &Hololink::send_control);

    py::class_<Hololink::I2c, std::shared_ptr<Hololink::I2c>>(m, "I2c").def("i2c_transaction",
        &Hololink::I2c::i2c_transaction, "peripheral_i2c_address"_a, "write_bytes"_a,
        "read_byte_count"_a, "timeout"_a = std::shared_ptr<Timeout>());

    py::class_<Hololink::Spi, std::shared_ptr<Hololink::Spi>>(m, "Spi").def("spi_transaction",
        &Hololink::Spi::spi_transaction, "peripheral_i2c_address"_a, "write_bytes"_a,
        "read_byte_count"_a, "timeout"_a = std::shared_ptr<Timeout>());

    auto gpio = py::class_<Hololink::GPIO, std::shared_ptr<Hololink::GPIO>>(m, "GPIO")
                    .def("set_direction", &Hololink::GPIO::set_direction, "pin"_a, "direction"_a)
                    .def("get_direction", &Hololink::GPIO::get_direction, "pin"_a)
                    .def("set_value", &Hololink::GPIO::set_value, "pin"_a, "value"_a)
                    .def("get_value", &Hololink::GPIO::get_value, "pin"_a);
    gpio.attr("IN") = Hololink::GPIO::IN;
    gpio.attr("OUT") = Hololink::GPIO::OUT;
    gpio.attr("LOW") = Hololink::GPIO::LOW;
    gpio.attr("HIGH") = Hololink::GPIO::HIGH;
    gpio.attr("GPIO_PIN_RANGE") = Hololink::GPIO::GPIO_PIN_RANGE;

} // PYBIND11_MODULE

} // namespace hololink
