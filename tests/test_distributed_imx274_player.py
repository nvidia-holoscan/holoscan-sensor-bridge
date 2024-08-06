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
from examples import distributed_imx274_player


@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_3840X2160_60FPS,
    ],
)
def test_distributed_imx274_player(
    camera_mode, headless, hololink_address, ibv_name, ibv_port, capfd
):
    frame_limit = 10
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
        arguments.append("--headless")

    with mock.patch("sys.argv", arguments):
        distributed_imx274_player.main()

        # check for errors
        captured = capfd.readouterr()
        assert "[error]" not in captured.err
