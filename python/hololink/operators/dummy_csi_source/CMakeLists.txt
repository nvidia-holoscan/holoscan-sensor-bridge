include(pybind11_add_hololink_module)

pybind11_add_hololink_module(
    CPP_CMAKE_TARGET dummy_csi_source
    CLASS_NAME "DummyCSISourceOp"
    IMPORT "import holoscan.core"
    SOURCES dummy_csi_source.cpp
)
