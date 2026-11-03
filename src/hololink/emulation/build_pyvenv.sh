#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Build the Emulator-only build's Python virtual environment: copy the freshly built
# `_emulation` extension into the package tree, create the venv, install the package's
# requirements, cupy for the host's CUDA major version, and the package itself, then take
# the extension copy out of the source tree again.

if [ $# -lt 4 ]; then
    echo "Usage: $0 <EMULATION_LIB_FILE> <PACKAGE_PATH> <PY_VENV> <CUDA_MAJOR_VERSION>"
    echo "  EMULATION_LIB_FILE: Path to the built _emulation extension module"
    echo "  PACKAGE_PATH: Path to the python package directory (contains setup.py)"
    echo "  PY_VENV: Path to the Python virtual environment to create"
    echo "  CUDA_MAJOR_VERSION: CUDA major version (only 12 and 13 are supported)"
    exit 1
fi

EMULATION_LIB_FILE=$1
PACKAGE_PATH=$2
REQUIREMENTS_FILE=$PACKAGE_PATH/requirements.txt
PY_VENV=$3
CUDA_MAJOR_VERSION=$4

if [ "$CUDA_MAJOR_VERSION" -lt 12 ] || [ "$CUDA_MAJOR_VERSION" -gt 13 ]; then
    echo "Error: CUDA_MAJOR_VERSION must be 12 or 13, got: $CUDA_MAJOR_VERSION"
    exit 1
fi

ENV_PYTHON=$PY_VENV/bin/python3
ENV_PIP="$ENV_PYTHON -m pip"
LIB_COPY=$PACKAGE_PATH/hololink/emulation/$(basename "$EMULATION_LIB_FILE")

# Start from a clean package tree: setuptools reuses a leftover build/lib from an earlier
# run, and a stale extension (e.g. a pybind11-era _emulation.cpython-*.so, which Python
# prefers over the stable-ABI _emulation.abi3.so) would then be packaged instead of the
# one just built.
rm -rf "$PACKAGE_PATH/build" "$PACKAGE_PATH/hololink.egg-info"
find "$PACKAGE_PATH/hololink" -name '*.so' -delete

cp "$EMULATION_LIB_FILE" "$LIB_COPY"

# Any failure below fails the build step; before, a failed pip install still left this
# script reporting success and the venv unusable.
set -o errexit
set -o pipefail
echo "Creating Python virtual environment in $PY_VENV"
python3 -m venv "$PY_VENV"
. "$PY_VENV/bin/activate"
$ENV_PIP install -r "$REQUIREMENTS_FILE"
# cupy is only needed by the GPU examples and has no wheel for every platform: warn, don't fail.
if ! $ENV_PIP install cupy-cuda${CUDA_MAJOR_VERSION}x; then
    echo "Warning: cupy-cuda${CUDA_MAJOR_VERSION}x did not install; the GPU examples need it" >&2
fi
$ENV_PIP install "$PACKAGE_PATH"
rm -f "$LIB_COPY"
rm -rf "$PACKAGE_PATH/hololink.egg-info"
deactivate
