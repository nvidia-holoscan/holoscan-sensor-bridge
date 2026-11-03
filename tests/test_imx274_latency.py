# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import re
import sys
from unittest import mock

import pytest
import utils
from hololink_module.sensors.imx274.imx274_mode import Imx274_Mode

from examples import imx274_latency

# The app's default --frame-limit, which the test doesn't change.
APP_FRAME_LIMIT = 300


@utils.requires_roce
@pytest.mark.skip_unless_ptp
@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode",
    [
        Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
    ],
)
def test_imx274_latency(camera_mode, headless, hololink_address, capsys):
    arguments = [
        sys.argv[0],
        "--camera-mode",
        str(camera_mode.value),
        "--hololink",
        hololink_address,
    ]
    if headless:
        arguments.extend(["--headless"])

    with utils.RunMonitor() as monitor, mock.patch("sys.argv", arguments):
        result = imx274_latency.main()

    assert result == 0
    # The app ends the run with request_stop() at its frame limit, which its
    # receiver doesn't know, so RunMonitor can't relate the floor to it.
    monitor.assert_ok(min_frames=utils.frame_floor(APP_FRAME_LIMIT), allow_stopped=True)

    # check for errors
    captured = capsys.readouterr()
    assert captured.err == ""
    # The app measured the latency of the frames it received.
    report = re.search(r"=== Latency Report \((\d+) frames\) ===", captured.out)
    assert report, "The latency app printed no latency report"
    assert int(report.group(1)) > 10
