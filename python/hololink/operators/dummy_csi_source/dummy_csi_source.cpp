#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>  // <-- Needed for Allocator

#include <hololink/operators/dummy_csi_source/dummy_csi_source.hpp>

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;
namespace hsops = holoscan::operators;

class PyDummyCSISourceOp : public hsops::DummyCSISourceOp {
 public:
  using hsops::DummyCSISourceOp::DummyCSISourceOp;

  PyDummyCSISourceOp(holoscan::Fragment* fragment,
                       const std::shared_ptr<holoscan::Allocator>& allocator,
                       int width,
                       int height,
                       const std::string& name = "dummy_csi_source")
      : hsops::DummyCSISourceOp(holoscan::ArgList{
            holoscan::Arg("allocator", allocator),
            holoscan::Arg("width", width),
            holoscan::Arg("height", height)
        })
  {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<holoscan::OperatorSpec>(fragment);
    setup(*spec_);
  }
};

PYBIND11_MODULE(_dummy_csi_source, m) {
#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<hsops::DummyCSISourceOp, PyDummyCSISourceOp, holoscan::Operator,
             std::shared_ptr<hsops::DummyCSISourceOp>>(m, "DummyCSISourceOp")
      .def(py::init<holoscan::Fragment*, const std::shared_ptr<holoscan::Allocator>&, int, int, const std::string&>(),
           "fragment"_a,
           "allocator"_a,
           "width"_a = 640,
           "height"_a = 480,
           "name"_a = "dummy_csi_source"s)
      .def("setup", &hsops::DummyCSISourceOp::setup, "spec"_a);
}
