#!/bin/bash

if [ $# -lt 3 ]; then
    echo "Usage: $0 <EMULATION_LIB_FILE> <SENSORS_LIB_FILE> <PACKAGE_PATH> <PY_VENV> <CUDA_MAJOR_VERSION> <CUPY>"
    echo "  EMULATION_LIB_FILE: (relative) Path to the library directory"
    echo "  SENSORS_LIB_FILE: (relative) Path to the sensors library directory"
    echo "  PACKAGE_PATH: (relative) Path to the python package directory"
    echo "  PY_VENV: (relative) Path to Python virtual environment"
    echo "  CUDA_MAJOR_VERSION: CUDA major version (only 12 and 13 are supported)"
    exit 1
fi

EMULATION_LIB_FILE=$1
SENSORS_LIB_FILE=$2
PACKAGE_PATH=$3
REQUIREMENTS_FILE=$PACKAGE_PATH/requirements.txt
PY_VENV=$4
CUDA_MAJOR_VERSION=$5
CUPY=$6

# Validate CUDA_MAJOR_VERSION
if [ "$CUDA_MAJOR_VERSION" -lt 12 ] || [ "$CUDA_MAJOR_VERSION" -gt 13 ]; then
    echo "Error: CUDA_MAJOR_VERSION must be 12 or 13, got: $CUDA_MAJOR_VERSION"
    exit 1
fi

ENV_PYTHON=$PY_VENV/bin/python3
ENV_PIP="$ENV_PYTHON -m pip"

cp $EMULATION_LIB_FILE $PACKAGE_PATH/hololink/emulation/
cp $SENSORS_LIB_FILE $PACKAGE_PATH/hololink/emulation/sensors/

echo "Creating Python virtual environment in $PY_VENV"

python3 -m venv "$PY_VENV" && \
. $PY_VENV/bin/activate  && \
$ENV_PIP install -r "$REQUIREMENTS_FILE" && \
$ENV_PIP install cupy-cuda${CUDA_MAJOR_VERSION}x ; \
$ENV_PIP install $PACKAGE_PATH && \
rm -f $PACKAGE_PATH/hololink/emulation/$EMULATION_LIB_FILE $PACKAGE_PATH/hololink/emulation/sensors/$SENSORS_LIB_FILE
deactivate