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

from examples import (
    linux_single_network_stereo_imx274_player,
    single_network_stereo_imx274_player,
)


@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
def test_single_network_stereo_imx274_player(
    headless, frame_limit, ibv_name, ibv_port, capsys
):
    arguments = [
        sys.argv[0],
        "--frame-limit",
        str(frame_limit),
        "--ibv-name",
        ibv_name,
        "--ibv-port",
        str(ibv_port),
    ]
    if headless:
        arguments.extend(["--headless"])

    with mock.patch("sys.argv", arguments):
        single_network_stereo_imx274_player.main()

        # check for errors
        captured = capsys.readouterr()
        assert captured.err == ""


@pytest.mark.skip_unless_imx274
def test_single_network_linux_stereo_imx274_player(headless, frame_limit, capsys):
    arguments = [
        sys.argv[0],
        "--frame-limit",
        str(frame_limit),
    ]
    if headless:
        arguments.extend(["--headless"])

    with mock.patch("sys.argv", arguments):
        linux_single_network_stereo_imx274_player.main()

        # check for errors
        captured = capsys.readouterr()
        assert captured.err == ""
