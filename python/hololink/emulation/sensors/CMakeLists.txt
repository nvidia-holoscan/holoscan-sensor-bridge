include(pybind11_add_hololink_module)

set(EMULATION_SENSORS_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src/hololink/emulation/python/hololink/emulation/sensors)
        
pybind11_add_hololink_module(
    CPP_CMAKE_TARGET emulation_sensors
    CLASS_NAME "_emulation_sensors"
    SOURCES ${EMULATION_SENSORS_SOURCE_DIR}/_sensors.cpp
)
add_custom_command(TARGET emulation_sensors_python PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${EMULATION_SENSORS_SOURCE_DIR}/__init__.py ${CMAKE_CURRENT_SOURCE_DIR}/
    COMMENT "Copying __init__.py to hololink/emulation/sensors/"
)