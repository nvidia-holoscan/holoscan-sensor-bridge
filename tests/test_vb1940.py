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
import threading
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
from hololink import AsyncEventListener, Hololink

MS_PER_SEC = 1000.0
US_PER_SEC = 1000.0 * MS_PER_SEC
NS_PER_SEC = 1000.0 * US_PER_SEC
SEC_PER_NS = 1.0 / NS_PER_SEC

vb1940_camera_modes_30fps = [
    hololink_module.sensors.vb1940.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS,
    hololink_module.sensors.vb1940.Vb1940_Mode.VB1940_MODE_1920X1080_30FPS,
    hololink_module.sensors.vb1940.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS_8BIT,
]

all_vb1940_camera_modes = [
    *vb1940_camera_modes_30fps,
    hololink_module.sensors.vb1940.Vb1940_Mode.VB1940_MODE_2560X1984_60FPS,
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
    vb1940_camera_modes_30fps,
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
    vb1940_camera_modes_30fps,
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
@pytest.mark.flaky(reruns=2)
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
    skew_samples = []
    max_skew_us = 100
    limit_s = max_skew_us / US_PER_SEC

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
            nonlocal frame_count
            frame_count += 1
            skew_samples.append(
                (
                    frame_count,
                    left_times["left_frame_number"],
                    right_times["right_frame_number"],
                    left_frame_start,
                    right_frame_start,
                    dt,
                )
            )

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
        aligner_allowable_dt=max_skew_us / US_PER_SEC,
    )
    stereo_test.execute()
    assert frame_count > 0, "monitor_callback did not observe any aligned stereo pairs"
    # The receiver conditions stop after frame_limit frames, but the aligner
    # can discard startup races and the opposite leg of a dropped frame-end
    # packet to resume with synchronized pairs. Require sustained delivery
    # without expecting every received frame to reach the monitor.
    assert (
        frame_count >= frame_limit * 0.8
    ), f"monitor_callback observed only {frame_count}/{frame_limit} aligned stereo pairs"
    for tap, left_frame, right_frame, left, right, dt in skew_samples:
        assert dt <= limit_s, (
            f"tap={tap}, left_frame={left_frame}, right_frame={right_frame}, "
            f"left={left:.9f}, right={right:.9f}, skew_us={dt * US_PER_SEC:.1f} "
            f"exceeded the {max_skew_us}us stereo skew limit"
        )


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


class FrameStartEndListener(AsyncEventListener):
    """Logs active frame start and end events."""

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._sif0_starts = []
        self._sif0_ends = []
        self._sif1_starts = []
        self._sif1_ends = []

    @staticmethod
    def get_timestamp(timestamp_s, timestamp_ns):
        return float(timestamp_s) + float(timestamp_ns) * 1e-9

    def on_sif0_frame_start_event(self, state, timestamp_s, timestamp_ns):
        if not (state & (1 << int(Hololink.Event.SIF_0_FRAME_START))):
            return
        ts = self.get_timestamp(timestamp_s, timestamp_ns)
        with self._lock:
            self._sif0_starts.append(ts)

    def on_sif0_frame_end_event(self, state, timestamp_s, timestamp_ns):
        if not (state & (1 << int(Hololink.Event.SIF_0_FRAME_END))):
            return
        ts = self.get_timestamp(timestamp_s, timestamp_ns)
        with self._lock:
            self._sif0_ends.append(ts)

    def on_sif1_frame_start_event(self, state, timestamp_s, timestamp_ns):
        if not (state & (1 << int(Hololink.Event.SIF_1_FRAME_START))):
            return
        ts = self.get_timestamp(timestamp_s, timestamp_ns)
        with self._lock:
            self._sif1_starts.append(ts)

    def on_sif1_frame_end_event(self, state, timestamp_s, timestamp_ns):
        if not (state & (1 << int(Hololink.Event.SIF_1_FRAME_END))):
            return
        ts = self.get_timestamp(timestamp_s, timestamp_ns)
        with self._lock:
            self._sif1_ends.append(ts)

    def deltas_s(self, sensor):
        with self._lock:
            if sensor == 0:
                assert len(self._sif0_starts) == len(
                    self._sif0_ends
                ), "expected paired frame start/end events"
                return [
                    end - start
                    for end, start in zip(self._sif0_ends, self._sif0_starts)
                ]
            elif sensor == 1:
                assert len(self._sif1_starts) == len(
                    self._sif1_ends
                ), "expected paired frame start/end events"
                return [
                    end - start
                    for end, start in zip(self._sif1_ends, self._sif1_starts)
                ]
            else:
                raise AssertionError("invalid sensor")


def _test_vb1940_frame_events(headless, frame_limit, hololink_address, sensor):
    """Test frame start/end sequencers, ensuring that both events trigger and are reasonably spaced."""

    camera_mode = hololink_module.sensors.vb1940.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS
    frames_per_second = 30

    # Expect at least 80% of the total frame interval between events.
    min_frame_s = 0.8 / float(frames_per_second)

    channel_metadata = hololink_module.Enumerator.find_channel(
        channel_ip=hololink_address
    )
    hololink_module.DataChannel.use_sensor(channel_metadata, sensor)
    hololink_channel = hololink_module.DataChannel(channel_metadata)
    hololink = hololink_channel.hololink()
    listener = FrameStartEndListener()
    hololink.on_async_event(listener)

    camera = hololink_module.sensors.vb1940.Vb1940Cam(hololink_channel)

    with applications.CudaContext() as (cu_context, cu_device_ordinal):
        application = linux_vb1940_player.HoloscanApplication(
            headless,
            False,
            cu_context,
            cu_device_ordinal,
            hololink_channel,
            camera,
            camera_mode,
            frame_limit,
        )

        hololink.start()
        try:
            hololink.reset()
            # reset() restarts the HSB's clock at boot-relative time; wait for
            # PTP to re-lock before capturing
            if not hololink.ptp_synchronize():
                raise ValueError("Failed to synchronize PTP.")
            camera.setup_clock()
            camera.configure(camera_mode)

            # Enable the sequencers
            if sensor == 0:
                fs_seq = hololink.sif0_frame_start_sequencer()
                fs_seq.write_uint32(0x0000020C, 0xFFFFFFFF)
                fs_seq.write_uint32(0x0000020C, 0x0)
                fs_seq.enable()

                fe_seq = hololink.sif0_frame_end_sequencer()
                fe_seq.write_uint32(0x0000020C, 0xFFFFFFFF)
                fe_seq.write_uint32(0x0000020C, 0x0)
                fe_seq.enable()
            else:
                fs_seq = hololink.sif1_frame_start_sequencer()
                fs_seq.write_uint32(0x0000020C, 0xFFFFFFFF)
                fs_seq.write_uint32(0x0000020C, 0x0)
                fs_seq.enable()

                fe_seq = hololink.sif1_frame_end_sequencer()
                fe_seq.write_uint32(0x0000020C, 0xFFFFFFFF)
                fe_seq.write_uint32(0x0000020C, 0x0)
                fe_seq.enable()

            application.run()
        finally:
            hololink.stop()

    deltas = listener.deltas_s(sensor)
    assert len(deltas) > 0, "no events received"
    min_s = min(deltas)
    max_s = max(deltas)
    logging.info("Frame start->end deltas (s): min=%s max=%s", min_s, max_s)
    # All deltas should be nearly equal to each other (allow 5% difference)
    assert max_s <= min_s * 1.05, f"inconsistent deltas: min={min_s} max={max_s}"
    # All deltas must be greater than the expected threshold.
    for i, dt in enumerate(deltas):
        assert dt >= min_frame_s, f"pair {i}: delta {dt}s < {min_frame_s}s"


@pytest.mark.skip_unless_vb1940
def test_vb1940_sif0_frame_events(headless, frame_limit, hololink_address):
    _test_vb1940_frame_events(headless, frame_limit, hololink_address, 0)


@pytest.mark.skip_unless_vb1940
def test_vb1940_sif1_frame_events(headless, frame_limit, hololink_address):
    _test_vb1940_frame_events(headless, frame_limit, hololink_address, 1)
