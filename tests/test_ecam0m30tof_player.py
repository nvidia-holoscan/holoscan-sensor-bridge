# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# See README.md for detailed information.

import sys
from unittest import mock

import pytest

import hololink as hololink_module
from examples import ecam0m30tof_player


@pytest.mark.skip_unless_ecam0m30tof
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [
        hololink_module.sensors.ecam0m30tof.ECam0M30Tof_Mode.EDEPTH_MODE_DEPTH_IR,
        hololink_module.sensors.ecam0m30tof.ECam0M30Tof_Mode.EDEPTH_MODE_DEPTH,
        hololink_module.sensors.ecam0m30tof.ECam0M30Tof_Mode.EDEPTH_MODE_IR,
    ],
)
def test_ecam0m30tof_player(
    camera_mode, headless, hololink_address, ibv_name, ibv_port, capsys
):
    arguments = [
        sys.argv[0],
        "--camera-mode",
        str(camera_mode),
        "--hololink",
        hololink_address,
        "--ibv-name",
        ibv_name,
        "--ibv-port",
        str(ibv_port),
    ]
    if headless:
        arguments.extend(["--headless"])

    with mock.patch("sys.argv", arguments):
        ecam0m30tof_player.main()

        # check for errors
        captured = capsys.readouterr()
        assert captured.err == ""
