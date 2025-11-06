include(pybind11_add_hololink_module)

set(EMULATION_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src/hololink/emulation/python/hololink/emulation)

pybind11_add_hololink_module(
    CPP_CMAKE_TARGET emulation
    CLASS_NAME "_emulation"
    SOURCES ${EMULATION_SOURCE_DIR}/_emulation.cpp
)
# manually add some additional links - pybind11_add_hololink_module only allows 1
target_link_libraries(emulation_python
    PRIVATE
        hololink::emulation::coe
        hololink::emulation::roce
)

add_custom_command(TARGET emulation_python PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${EMULATION_SOURCE_DIR}/__init__.py ${CMAKE_CURRENT_SOURCE_DIR}/
    COMMENT "Copying __init__.py to hololink/emulation/"
)

add_subdirectory(sensors)