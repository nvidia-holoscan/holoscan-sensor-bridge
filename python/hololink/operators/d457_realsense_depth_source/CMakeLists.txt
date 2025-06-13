include(pybind11_add_hololink_module)

pybind11_add_hololink_module(
    CPP_CMAKE_TARGET d457_realsense_depth_source
    CLASS_NAME "D457RealSenseDepthSourceOp"
    IMPORT "import holoscan.core"
    SOURCES realsense_depth_source.cpp
)
