include(pybind11_add_hololink_module)

pybind11_add_hololink_module(
    CPP_CMAKE_TARGET d455_realsense_rgb_source
    CLASS_NAME "D455RealSenseRGBSourceOp"
    IMPORT "import holoscan.core"
    SOURCES realsense_rgb_source.cpp
)
