include(pybind11_add_hololink_module)

pybind11_add_hololink_module(
    CPP_CMAKE_TARGET d455_realsense_dual_source
    CLASS_NAME "D455RealSenseDualSourceOp"
    IMPORT "import holoscan.core"
    SOURCES realsense_dual_source.cpp
)
