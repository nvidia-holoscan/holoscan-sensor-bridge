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
from examples import (
    linux_single_network_stereo_vb1940_player,
    linux_vb1940_player,
    single_network_stereo_vb1940_player,
    vb1940_player,
)

all_vb1940_camera_modes = [
    hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS,
]


# Unaccelerated, single camera tests
@pytest.mark.skip_unless_vb1940
@pytest.mark.parametrize(
    "camera_mode",
    all_vb1940_camera_modes,
)
def test_vb1940_linux_player(
    camera_mode, headless, frame_limit, hololink_address, capsys
):
    arguments = [
        sys.argv[0],
        "--camera-mode",
        str(camera_mode.value),
        "--frame-limit",
        str(frame_limit),
        "--hololink",
        hololink_address,
    ]
    if headless:
        arguments.extend(["--headless"])

    with mock.patch("sys.argv", arguments):
        linux_vb1940_player.main()

    # check for errors
    captured = capsys.readouterr()
    assert captured.err == ""


# Unaccelerated, stereo, over one network port
@pytest.mark.skip_unless_vb1940
@pytest.mark.parametrize(
    "camera_mode",
    all_vb1940_camera_modes,
)
def test_vb1940_linux_single_network_stereo_player(
    camera_mode, headless, frame_limit, hololink_address, capsys
):
    arguments = [
        sys.argv[0],
        "--camera-mode",
        str(camera_mode.value),
        "--frame-limit",
        str(frame_limit),
        "--hololink",
        hololink_address,
    ]
    if headless:
        arguments.extend(["--headless"])

    with mock.patch("sys.argv", arguments):
        linux_single_network_stereo_vb1940_player.main()

    # check for errors
    captured = capsys.readouterr()
    assert captured.err == ""


# Accelerated, single camera tests
@pytest.mark.skip_unless_vb1940
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode",
    all_vb1940_camera_modes,
)
def test_vb1940_player(
    camera_mode, headless, frame_limit, hololink_address, ibv_name, ibv_port, capsys
):
    arguments = [
        sys.argv[0],
        "--camera-mode",
        str(camera_mode.value),
        "--frame-limit",
        str(frame_limit),
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
        vb1940_player.main()

        # check for errors
        captured = capsys.readouterr()
        assert captured.err == ""


# Accelerated, stereo, over one network port
@pytest.mark.skip_unless_vb1940
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode",
    all_vb1940_camera_modes,
)
def test_vb1940_single_network_stereo_player(
    camera_mode, headless, frame_limit, hololink_address, ibv_name, ibv_port, capsys
):
    arguments = [
        sys.argv[0],
        "--camera-mode",
        str(camera_mode.value),
        "--frame-limit",
        str(frame_limit),
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
        single_network_stereo_vb1940_player.main()

        # check for errors
        captured = capsys.readouterr()
        assert captured.err == ""
