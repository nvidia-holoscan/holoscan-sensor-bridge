# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from examples.hawk_single_link_player import HoloscanApplication
from examples.linux_single_network_stereo_hawk_player import (
    HoloscanApplication as LinuxHoloscanApplication,
)


@pytest.mark.parametrize(
    "application_type", [HoloscanApplication, LinuxHoloscanApplication]
)
def test_shared_hawk_stops_after_both_receivers(application_type):
    camera = MagicMock()
    application = SimpleNamespace(
        _camera=camera,
        _camera_lifecycle_lock=threading.Lock(),
        _camera_users=0,
    )

    application_type._start_camera(application)
    application_type._start_camera(application)
    camera.start.assert_called_once_with()

    application_type._stop_camera(application)
    camera.stop.assert_not_called()

    application_type._stop_camera(application)
    camera.stop.assert_called_once_with()

    # Holoscan can call stop during rollback even when start did not finish.
    application_type._stop_camera(application)
    camera.stop.assert_called_once_with()
