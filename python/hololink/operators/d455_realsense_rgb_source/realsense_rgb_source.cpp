#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>

#include <hololink/operators/d455_realsense_rgb_source/realsense_rgb_source.hpp>

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;
namespace hsops = holoscan::operators;

class PyRealSenseRGBSourceOp : public hsops::D455RealSenseRGBSourceOp {
 public:
  using hsops::D455RealSenseRGBSourceOp::D455RealSenseRGBSourceOp;

  PyRealSenseRGBSourceOp(holoscan::Fragment* fragment,
                         const std::shared_ptr<holoscan::Allocator>& allocator,
                         const std::string& name = "d455_realsense_rgb_source")
      : hsops::D455RealSenseRGBSourceOp(holoscan::ArgList{
            holoscan::Arg("allocator", allocator)
        })
  {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<holoscan::OperatorSpec>(fragment);
    setup(*spec_);
  }
};

PYBIND11_MODULE(_d455_realsense_rgb_source, m) {
#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<hsops::D455RealSenseRGBSourceOp, PyRealSenseRGBSourceOp, holoscan::Operator,
             std::shared_ptr<hsops::D455RealSenseRGBSourceOp>>(m, "D455RealSenseRGBSourceOp")
      .def(py::init<holoscan::Fragment*, const std::shared_ptr<holoscan::Allocator>&, const std::string&>(),
           "fragment"_a,
           "allocator"_a,
           "name"_a = "d455_realsense_rgb_source"s)
      .def("setup", &hsops::D455RealSenseRGBSourceOp::setup, "spec"_a);
}
