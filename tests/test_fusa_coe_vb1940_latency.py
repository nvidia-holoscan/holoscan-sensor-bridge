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
from hololink.sensors.vb1940.vb1940_mode import Vb1940_Mode

import hololink as hololink_module

# This test relies on FUSA which may not be
# configured on this host.  If that's the case,
# then skip the test.
if not hasattr(hololink_module.operators, "FusaCoeCaptureOp"):
    pytest.skip(
        "FusaCoeCaptureOp isn't in this build (HOLOLINK_BUILD_FUSA=OFF)",
        allow_module_level=True,
    )

from examples import fusa_coe_vb1940_latency  # noqa: E402

vb1940_camera_mode = [
    Vb1940_Mode.VB1940_MODE_2560X1984_30FPS,
]


@pytest.mark.skip_unless_ptp
@pytest.mark.skip_unless_vb1940
@pytest.mark.skip_unless_coe
@pytest.mark.parametrize("camera_mode", vb1940_camera_mode)
def test_fusa_coe_vb1940_latency(
    camera_mode, headless, frame_limit, hololink_address, capsys
):
    """Test VB1940 latency measurement using FuSa CoE capture path."""
    # No --interface: the app then uses the NIC the board enumerated on, for both the board's
    # CoE destination and the capture. A route lookup (the coe_interfaces fixture) can pick
    # another NIC on the same subnet, which would never see a frame.
    arguments = [
        sys.argv[0],
        "--hololink",
        hololink_address,
        "--camera-mode",
        str(camera_mode.value),
        "--frame-limit",
        str(frame_limit),
    ]
    if headless:
        arguments.extend(["--headless"])

    with mock.patch("sys.argv", arguments):
        result = fusa_coe_vb1940_latency.main()

        # check for errors
        captured = capsys.readouterr()
        assert captured.err == ""
        assert result == 0, "Latency measurement should complete successfully"
