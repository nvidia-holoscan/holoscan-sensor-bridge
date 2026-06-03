# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# This test relies on FUSA which may not be
# configured on this host.  If that's the case,
# then skip the test.
pytest.importorskip("hololink.operators.fusa_coe_capture")

from examples import fusa_coe_vb1940_latency  # noqa: E402

vb1940_camera_mode = [
    hololink_module.sensors.vb1940.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS,
]


@pytest.mark.skip_unless_ptp
@pytest.mark.skip_unless_vb1940
@pytest.mark.skip_unless_coe
@pytest.mark.parametrize("camera_mode", vb1940_camera_mode)
def test_fusa_coe_vb1940_latency(
    camera_mode, headless, frame_limit, hololink_address, coe_interfaces, capsys
):
    """Test VB1940 latency measurement using FuSa CoE capture path."""
    # Use first available CoE interface
    coe_interface = coe_interfaces[0] if coe_interfaces else None

    arguments = [
        sys.argv[0],
        "--hololink",
        hololink_address,
        "--camera-mode",
        str(camera_mode.value),
        "--frame-limit",
        str(frame_limit),
    ]
    if coe_interface:
        arguments.extend(["--interface", coe_interface])
    if headless:
        arguments.extend(["--headless"])

    with mock.patch("sys.argv", arguments):
        result = fusa_coe_vb1940_latency.main()

        # check for errors
        captured = capsys.readouterr()
        assert captured.err == ""
        assert result == 0, "Latency measurement should complete successfully"
