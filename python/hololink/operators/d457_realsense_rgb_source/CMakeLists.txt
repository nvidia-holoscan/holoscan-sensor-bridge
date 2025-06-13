include(pybind11_add_hololink_module)

pybind11_add_hololink_module(
    CPP_CMAKE_TARGET d457_realsense_rgb_source
    CLASS_NAME "D457RealSenseRGBSourceOp"
    IMPORT "import holoscan.core"
    SOURCES realsense_rgb_source.cpp
)
