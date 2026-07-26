# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# See README.md for detailed information.

import sys
from unittest import mock

import pytest
from hololink_module.sensors.imx274 import imx274_mode as adapter_imx274_mode

from examples import module_linux_imx274_player

# Software-receiver end-to-end run against real HSB-Lite hardware with an
# IMX274 camera attached. Gated on skip_unless_imx274 (the --imx274
# switch) only — deliberately NOT accelerated_networking: the Linux
# software receiver needs no infiniband device, so this runs in
# HOLOLINK_BUILD_ROCE=OFF environments where the RoCE player tests skip.


@pytest.mark.skip_unless_imx274
@pytest.mark.parametrize(
    "camera_mode",
    [
        adapter_imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
    ],
)
def test_module_linux_imx274_player(
    camera_mode, headless, frame_limit, hololink_address
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
        arguments.append("--headless")

    with mock.patch("sys.argv", arguments):
        module_linux_imx274_player.main()
