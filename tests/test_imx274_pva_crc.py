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

import logging
import re
import sys
from unittest import mock

import pytest
import utils
from hololink_module.sensors.imx274.imx274_mode import Imx274_Mode

from examples import imx274_pva_crc_validation


@utils.requires_roce
@pytest.mark.skip_unless_pva
@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode",
    [
        Imx274_Mode.IMX274_MODE_3840X2160_60FPS,
    ],
)
def test_imx274_pva_crc_validation(
    camera_mode,
    headless,
    frame_limit,
    hololink_address,
    ibv_name,
    ibv_port,
    caplog,
    capsys,
):
    """
    Test IMX274 frame validation with PVA CRC hardware.

    This test verifies:
    - PVA CRC computation on IMX274 frames
    - CRC comparison between FPGA-embedded and PVA-computed values
    - Frame validation and timing statistics
    """
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

    with caplog.at_level(logging.ERROR):
        with utils.RunMonitor() as monitor, mock.patch("sys.argv", arguments):
            result = imx274_pva_crc_validation.main()

    # Fail if app logged any ERROR or CRITICAL
    error_logs = [rec for rec in caplog.records if rec.levelno >= logging.ERROR]
    assert not error_logs, (
        f"PVA CRC validation logged {len(error_logs)} error(s); "
        f"first: {error_logs[0].getMessage()}"
    )
    assert result == 0
    monitor.assert_ok()
    # The app logs each mismatch (caught above) and prints a summary; v2.7 checked
    # only the log. The summary also shows that the PVA checked frames at all.
    output = capsys.readouterr().out
    validated = re.search(r"Validated frames: (\d+)", output)
    mismatches = re.search(r"Mismatches: +(\d+)", output)
    assert validated and mismatches, "The app printed no PVA CRC summary"
    checked, errors = int(validated.group(1)), int(mismatches.group(1))
    assert errors == 0, f"PVA CRC validation found {errors} errors in {checked} frames"
    assert checked > 10, f"PVA CRC validation checked only {checked} frames"
