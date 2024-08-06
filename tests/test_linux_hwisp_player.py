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

# See README.md for detailed information.

import sys
from unittest import mock

import pytest

import hololink as hololink_module
from examples import linux_hwisp_player


@pytest.mark.skip_unless_igpu
@pytest.mark.skip_unless_imx274
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
    ],
)
def test_linux_hwisp_player(
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
        linux_hwisp_player.main()

        # check for errors
        captured = capsys.readouterr()
        assert captured.err == ""
