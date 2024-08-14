#!/bin/sh

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# See README.md for detailed information.

# Usage:
#   imx274_ptp_test.sh
#
# Invokes the CI container, then runs the PTP test inside it.

set -o errexit
set -o xtrace

SCRIPT=`realpath "$0"`
HERE=`dirname "$SCRIPT"`

if [ "$1" = "go" ]; then
    shift  # get rid of "go"
    umask 0
    # Python tests
    pytest tests/test_imx274_timestamps.py --imx274 --ptp --headless -vv $@
    exit 0
fi

BUILD=$1

sh $HERE/ci.sh $BUILD $SCRIPT go $@
