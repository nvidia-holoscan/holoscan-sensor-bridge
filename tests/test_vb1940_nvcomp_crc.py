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
import sys
from unittest import mock

import pytest

import hololink as hololink_module
from examples import vb1940_fusa_nvcomp_crc_validation


@pytest.mark.skip_unless_fusa
@pytest.mark.skip_unless_vb1940
@pytest.mark.parametrize(
    "camera_mode",
    [
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS,
    ],
)
def test_vb1940_nvcomp_crc_validation(
    camera_mode, headless, frame_limit, hololink_address, caplog
):
    """
    Test VB1940 frame validation with nvCOMP CRC.

    This test verifies:
    - nvCOMP CRC computation on VB1940 frames using FUSA CoE capture
    - CRC comparison between FPGA-embedded and nvCOMP-computed values
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
    ]
    if headless:
        arguments.extend(["--headless"])

    with caplog.at_level(logging.ERROR):
        with mock.patch("sys.argv", arguments):
            vb1940_fusa_nvcomp_crc_validation.main()

    # Fail if app logged any ERROR or CRITICAL
    error_logs = [rec for rec in caplog.records if rec.levelno >= logging.ERROR]
    assert not error_logs, (
        f"nvCOMP CRC validation logged {len(error_logs)} error(s); "
        f"first: {error_logs[0].getMessage()}"
    )
