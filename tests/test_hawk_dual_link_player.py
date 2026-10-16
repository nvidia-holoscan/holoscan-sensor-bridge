# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from examples.hawk_dual_link_player import HoloscanApplication


def test_hawks_stop_after_both_receivers():
    cameras = [MagicMock(), MagicMock()]
    application = SimpleNamespace(
        _cameras=cameras,
        _camera_lifecycle_lock=threading.Lock(),
        _camera_users=0,
    )

    HoloscanApplication._start_cameras(application)
    HoloscanApplication._start_cameras(application)
    assert application._camera_users == 2
    for camera in cameras:
        camera.start.assert_called_once_with()

    HoloscanApplication._stop_cameras(application)
    assert application._camera_users == 1
    for camera in cameras:
        camera.stop.assert_not_called()

    HoloscanApplication._stop_cameras(application)
    assert application._camera_users == 0
    for camera in cameras:
        camera.stop.assert_called_once_with()

    HoloscanApplication._stop_cameras(application)
    assert application._camera_users == 0
    for camera in cameras:
        camera.stop.assert_called_once_with()


def test_start_failure_rolls_back_without_adding_user():
    cameras = [MagicMock(), MagicMock()]
    cameras[1].start.side_effect = RuntimeError("start failed")
    application = SimpleNamespace(
        _cameras=cameras,
        _camera_lifecycle_lock=threading.Lock(),
        _camera_users=0,
    )

    with pytest.raises(RuntimeError, match="start failed"):
        HoloscanApplication._start_cameras(application)

    assert application._camera_users == 0
    cameras[0].stop.assert_called_once_with()
    cameras[1].stop.assert_not_called()


def test_stop_failure_keeps_final_user():
    cameras = [MagicMock(), MagicMock()]
    cameras[1].stop.side_effect = RuntimeError("stop failed")
    application = SimpleNamespace(
        _cameras=cameras,
        _camera_lifecycle_lock=threading.Lock(),
        _camera_users=1,
    )

    with pytest.raises(RuntimeError, match="stop failed"):
        HoloscanApplication._stop_cameras(application)

    assert application._camera_users == 1
    for camera in cameras:
        camera.stop.assert_called_once_with()
