# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import applications
import pytest

import hololink as hololink_module
from examples import (
    linux_single_network_stereo_vb1940_player,
    linux_vb1940_player,
    single_network_stereo_vb1940_player,
    vb1940_imu_player,
    vb1940_player,
    vb1940_stereo_imu_player,
)

MS_PER_SEC = 1000.0
US_PER_SEC = 1000.0 * MS_PER_SEC
NS_PER_SEC = 1000.0 * US_PER_SEC
SEC_PER_NS = 1.0 / NS_PER_SEC

all_vb1940_camera_modes = [
    hololink_module.sensors.vb1940.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS,
]


# Unaccelerated, single camera tests
@pytest.mark.skip_unless_vb1940
@pytest.mark.parametrize(
    "camera_mode",
    all_vb1940_camera_modes,
)
def test_vb1940_linux_player(
    camera_mode, headless, frame_limit, hololink_address, capsys
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
        arguments.extend(["--headless"])

    with mock.patch("sys.argv", arguments):
        linux_vb1940_player.main()

    # check for errors
    captured = capsys.readouterr()
    assert captured.err == ""


# Unaccelerated, stereo, over one network port
@pytest.mark.skip_unless_vb1940
@pytest.mark.parametrize(
    "camera_mode",
    all_vb1940_camera_modes,
)
def test_vb1940_linux_single_network_stereo_player(
    camera_mode, headless, frame_limit, hololink_address, capsys
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
        arguments.extend(["--headless"])

    with mock.patch("sys.argv", arguments):
        linux_single_network_stereo_vb1940_player.main()

    # check for errors
    captured = capsys.readouterr()
    assert captured.err == ""


# Accelerated, single camera tests
@pytest.mark.skip_unless_vb1940
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode",
    all_vb1940_camera_modes,
)
def test_vb1940_player(
    camera_mode, headless, frame_limit, hololink_address, ibv_name, ibv_port, capsys
):
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

    with mock.patch("sys.argv", arguments):
        vb1940_player.main()

        # check for errors
        captured = capsys.readouterr()
        assert captured.err == ""


# Accelerated, stereo, over one network port
@pytest.mark.skip_unless_vb1940
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode",
    all_vb1940_camera_modes,
)
def test_vb1940_single_network_stereo_player(
    camera_mode, headless, frame_limit, hololink_address, ibv_name, ibv_port, capsys
):
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

    with mock.patch("sys.argv", arguments):
        single_network_stereo_vb1940_player.main()

        # check for errors
        captured = capsys.readouterr()
        assert captured.err == ""


@pytest.mark.skip_unless_vb1940
def test_vb1940_configuration_eeprom(
    hololink_address,
):
    # Get a handle to the Hololink device
    channel_metadata = hololink_module.Enumerator.find_channel(
        channel_ip=hololink_address
    )
    hololink_channel = hololink_module.DataChannel(channel_metadata)
    hololink = hololink_channel.hololink()
    hololink.start()
    # Get a handle to the camera
    camera = hololink_module.sensors.vb1940.Vb1940Cam(hololink_channel)
    calibration_data = camera.get_calibration_data(2)
    logging.info(f"{calibration_data=}")


@pytest.mark.skip_unless_ptp
@pytest.mark.accelerated_networking
@pytest.mark.skip_unless_vb1940
@pytest.mark.parametrize(
    "camera_mode, frames_per_second",  # noqa: E501
    [
        (
            hololink_module.sensors.vb1940.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS,
            30,
        ),
    ],
)
def test_vb1940_stereo_roce_naive(
    camera_mode,
    frames_per_second,
    headless,
    frame_limit,
    hololink_address,
    ptp_enable,
    ibv_name,
    ibv_port,
):
    frame_count = 0

    class StereoTest(applications.StereoTest):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def monitor_callback(self, operator, metadata):
            left_times = operator.get_times(
                metadata, rename_metadata=lambda name: f"left_{name}"
            )
            right_times = operator.get_times(
                metadata, rename_metadata=lambda name: f"right_{name}"
            )
            left_frame_start = left_times["left_frame_start"]
            right_frame_start = right_times["right_frame_start"]
            dt = abs(right_frame_start - left_frame_start)
            logging.info(f"{left_frame_start=} {right_frame_start=} {dt=:0.6f}")
            limit_us = 100
            limit_s = limit_us / US_PER_SEC
            nonlocal frame_count
            frame_count += 1
            assert dt <= limit_s, f"{frame_count=}"

    def left_camera_factory(
        channel, instance, camera_mode=camera_mode, frames_per_second=frames_per_second
    ):
        hololink = channel.hololink()
        synchronizer = hololink.ptp_pps_output(frequency=frames_per_second)
        camera = applications.vb1940_camera_factory(
            channel,
            instance,
            camera_mode,
            vsync=synchronizer,
        )
        return camera

    def right_camera_factory(
        channel, instance, camera_mode=camera_mode, frames_per_second=frames_per_second
    ):
        hololink = channel.hololink()
        synchronizer = hololink.ptp_pps_output()
        camera = applications.vb1940_camera_factory(
            channel,
            instance,
            camera_mode,
            vsync=synchronizer,
        )
        return camera

    stereo_test = StereoTest(
        camera_mode,
        headless,
        frame_limit,
        hololink_address,
        left_receiver_factory=lambda *args, **kwargs: applications.roce_receiver_factory(
            ibv_name, ibv_port, *args, **kwargs
        ),
        right_receiver_factory=lambda *args, **kwargs: applications.roce_receiver_factory(
            ibv_name, ibv_port, *args, **kwargs
        ),
        left_isp_factory=applications.naive_isp,
        right_isp_factory=applications.naive_isp,
        left_camera_factory=left_camera_factory,
        right_camera_factory=right_camera_factory,
        ptp_enable=ptp_enable,
    )
    stereo_test.execute()


@pytest.mark.skip_unless_vb1940
@pytest.mark.accelerated_networking
@pytest.mark.skip_unless_ptp
def test_vb1940_imu(frame_limit, hololink_address, ibv_name, ibv_port, capsys):
    arguments = [
        sys.argv[0],
        "--frame-limit",
        str(frame_limit),
        "--hololink",
        hololink_address,
        "--ibv-name",
        ibv_name,
        "--ibv-port",
        str(ibv_port),
    ]

    with mock.patch("sys.argv", arguments):
        vb1940_imu_player.main()

        # check for errors
        captured = capsys.readouterr()
        assert captured.err == ""


@pytest.mark.skip_unless_vb1940
@pytest.mark.accelerated_networking
@pytest.mark.skip_unless_ptp
def test_vb1940_stereo_imu(
    frame_limit, hololink_address, ibv_name, ibv_port, capsys, headless
):
    arguments = [
        sys.argv[0],
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

    with mock.patch("sys.argv", arguments):
        vb1940_stereo_imu_player.main()

        # check for errors
        captured = capsys.readouterr()
        assert captured.err == ""
