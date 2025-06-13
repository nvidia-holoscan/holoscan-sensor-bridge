// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>

#include <hololink/operators/d455_realsense_depth_source/realsense_depth_source.hpp>

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;
namespace hsops = holoscan::operators;

class PyRealSenseDepthSourceOp : public hsops::D455RealSenseDepthSourceOp {
 public:
  using hsops::D455RealSenseDepthSourceOp::D455RealSenseDepthSourceOp;

  PyRealSenseDepthSourceOp(holoscan::Fragment* fragment,
                           const std::shared_ptr<holoscan::Allocator>& allocator,
                           const std::string& name = "d455_realsense_depth_source")
      : hsops::D455RealSenseDepthSourceOp(holoscan::ArgList{
            holoscan::Arg("allocator", allocator)
        })
  {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<holoscan::OperatorSpec>(fragment);
    setup(*spec_);
  }
};

PYBIND11_MODULE(_d455_realsense_depth_source, m) {
#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<hsops::D455RealSenseDepthSourceOp, PyRealSenseDepthSourceOp, holoscan::Operator,
             std::shared_ptr<hsops::D455RealSenseDepthSourceOp>>(m, "D455RealSenseDepthSourceOp")
      .def(py::init<holoscan::Fragment*, const std::shared_ptr<holoscan::Allocator>&, const std::string&>(),
           "fragment"_a,
           "allocator"_a,
           "name"_a = "d455_realsense_depth_source"s)
      .def("setup", &hsops::D455RealSenseDepthSourceOp::setup, "spec"_a);
}
