include(pybind11_add_hololink_module)

pybind11_add_hololink_module(
    CPP_CMAKE_TARGET image_decoder
    CLASS_NAME "ImageDecoderOp"
    IMPORT "import holoscan.core"
    SOURCES image_decoder.cpp
)
