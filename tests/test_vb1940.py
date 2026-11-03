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

# HSDK5 differences from v2.7:
# - The player tests also check the example's return value and the frames that
#   its receivers emitted (utils.RunMonitor).
# - The stereo RoCE test builds an hs.Graph instead of applications.StereoTest's
#   HSDK4 application, which failed on any frame pair that HolovizOp took with a
#   skew above 100 us. Here each leg has its own watchdog, and the legs'
#   frame-start times are paired after the run within half a frame period: 80%
#   of frame_limit must pair, with a pair at least every 0.5 s
#   (operators.assert_aligned()), and each pair within 100 us.
# - The frame-event tests build the linux_vb1940_player pipeline as an hs.Graph:
#   the HSDK5 example has no HoloscanApplication class.
# - HSDK5 doesn't bind hololink.Hololink.Event; its SIF event bit numbers are
#   defined below.

import logging
import math
import sys
import threading
from unittest import mock

import applications
import operators
import pytest
import utils
from hololink.sensors.vb1940.vb1940_mode import Vb1940_Mode

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

# Hololink::Event bits in an async event's state (src/hololink/core/hololink.hpp).
SIF_0_FRAME_END = 16
SIF_1_FRAME_END = 17
SIF_0_FRAME_START = 20
SIF_1_FRAME_START = 21

vb1940_camera_modes_30fps = [
    Vb1940_Mode.VB1940_MODE_2560X1984_30FPS,
    Vb1940_Mode.VB1940_MODE_1920X1080_30FPS,
    Vb1940_Mode.VB1940_MODE_2560X1984_30FPS_8BIT,
]

all_vb1940_camera_modes = [
    *vb1940_camera_modes_30fps,
    Vb1940_Mode.VB1940_MODE_2560X1984_60FPS,
]


def _run_example(
    example, arguments, capsys, receivers=1, allow_stopped=False, min_frames=None
):
    with utils.RunMonitor() as monitor, mock.patch("sys.argv", arguments):
        # Some examples (the IMU players) end main() with sys.exit(run(...))
        # instead of returning run()'s result.
        try:
            result = example.main()
        except SystemExit as e:
            result = e.code

    assert result == 0
    assert len(monitor.frames) >= receivers, monitor.frames
    monitor.assert_ok(min_frames=min_frames, allow_stopped=allow_stopped)

    # check for errors
    captured = capsys.readouterr()
    assert captured.err == ""


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
    _run_example(linux_vb1940_player, arguments, capsys)


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
    # The app ends the run with request_stop() at its frame limit, which its
    # receivers don't know, so RunMonitor can't relate their floor to it.
    _run_example(
        linux_single_network_stereo_vb1940_player,
        arguments,
        capsys,
        receivers=2,
        allow_stopped=True,
        min_frames=utils.frame_floor(frame_limit),
    )


# Accelerated, single camera tests
@utils.requires_roce
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
    _run_example(vb1940_player, arguments, capsys)


# Accelerated, stereo, over one network port
@utils.requires_roce
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
    _run_example(single_network_stereo_vb1940_player, arguments, capsys, receivers=2)


@pytest.mark.skip_unless_vb1940
def test_vb1940_configuration_eeprom(
    hololink_address,
):
    # Get a handle to the Hololink device
    channel_metadata = hololink_module.find_channel(hololink_address)
    hololink_channel = hololink_module.DataChannel(channel_metadata)
    hololink = hololink_channel.hololink()
    hololink.start()
    try:
        # Get a handle to the camera
        camera = hololink_module.NativeVb1940Sensor(hololink_channel)
        # Both parts, as v2.7's get_calibration_data(2) read them.
        rgb_calibration_data = camera.get_rgb_calibration_data_string()
        logging.info(f"{rgb_calibration_data=}")
        ir_calibration_data = camera.get_ir_calibration_data_string()
        logging.info(f"{ir_calibration_data=}")
    finally:
        hololink.stop()


@utils.requires_roce
@pytest.mark.skip_unless_ptp
@pytest.mark.accelerated_networking
@pytest.mark.skip_unless_vb1940
@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize(
    "camera_mode, frames_per_second",  # noqa: E501
    [
        (
            Vb1940_Mode.VB1940_MODE_2560X1984_30FPS,
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
    max_skew_us = 100
    limit_s = max_skew_us / US_PER_SEC
    frame_starts = {"left": [], "right": []}

    with applications.CudaContext() as cuda_context:
        # The board's PTP PPS output, at the frame rate, drives both cameras'
        # vsync, as in v2.7's left_camera_factory and right_camera_factory.
        left = applications.open_vb1940(
            hololink_address,
            camera_mode,
            sensor=0,
            vsync=frames_per_second,
            ptp=ptp_enable,
        )
        device = left.device
        try:
            right = applications.open_vb1940(
                hololink_address, camera_mode, sensor=1, vsync=0, device=device
            )
            with utils.leg_watchdogs(
                ("left", "right"),
                initial_timeout=operators.color_profiler_initial_timeout(frame_limit),
                min_taps=utils.frame_floor(frame_limit),
            ) as watchdogs, utils.RunMonitor() as monitor:
                test_graph = applications.TestGraph("vb1940_stereo")
                images = []
                for name, camera in (("left", left), ("right", right)):
                    rx = applications.add_receiver(
                        test_graph,
                        f"{name}_receiver",
                        hololink_module.operators.RoceReceiverOp,
                        camera,
                        cuda_context.cu_context,
                        frame_limit,
                        ibv_name=ibv_name,
                        ibv_port=ibv_port,
                    )
                    isp = applications.add_isp(test_graph, f"{name}_", rx, camera)
                    images.append(
                        applications.Image(
                            isp["demosaic"].output, camera.width, camera.height
                        )
                    )

                    def time_profile(
                        image_timestamp_s, *args, starts=frame_starts[name]
                    ):
                        starts.append((image_timestamp_s, args[-1]))

                    profiler = test_graph.op(
                        operators.TimeProfiler,
                        f"{name}_profiler",
                        callback_token=operators.register(time_profile),
                    )
                    operators.metadata_flow(
                        test_graph, isp["image_processor"].metadata, profiler.metadata
                    )
                    watchdog_operator = test_graph.op(
                        operators.WatchdogOp,
                        f"{name}_watchdog",
                        watchdog_token=operators.register(watchdogs[name]),
                    )
                    operators.metadata_flow(
                        test_graph,
                        isp["image_processor"].metadata,
                        watchdog_operator.metadata,
                    )
                applications.show_images(
                    test_graph, "holoviz", images, headless, "VB1940 stereo test"
                )
                test_graph.run()
        finally:
            device.stop()

    monitor.assert_ok()
    # A dropped frame-end packet loses one leg of a pair; require sustained
    # delivery without expecting every received frame to pair.
    # The legs are paired within half a frame period, so that each frame finds the
    # other leg's frame of the same VSYNC; then every pair must be within 100 us,
    # as each pair that v2.7's HolovizOp took had to be.
    pairs = operators.assert_aligned(
        frame_starts["left"],
        frame_starts["right"],
        0.5 / frames_per_second,
        min_pairs=math.ceil(frame_limit * 0.8),
        skip_first=operators.COLOR_PROFILER_START_FRAME + 2,
    )
    for tap, ((left_start, left_frame), (right_start, right_frame)) in enumerate(pairs):
        dt = abs(right_start - left_start)
        assert dt <= limit_s, (
            f"tap={tap}, left_frame={left_frame}, right_frame={right_frame}, "
            f"left={left_start:.9f}, right={right_start:.9f}, "
            f"skew_us={dt * US_PER_SEC:.1f} "
            f"exceeded the {max_skew_us}us stereo skew limit"
        )


# On IGX Thor with a VB1940 board, the BMI088 IMU answers over I2C and its data
# registers change, and the HSB IMU streaming engine is started, but the IMU's
# data-ready interrupt never triggers the engine, so no IMU sample reaches the
# host (v2.7 gets none either; its test didn't check for data). With no data,
# test_vb1940_imu waits in session.wait() forever and test_vb1940_stereo_imu
# fails with imu_receiver at 0 frames. Remove the skip when the IMU interrupt
# path to the HSB is fixed.
IMU_SKIP_REASON = (
    "VB1940 IMU data-ready interrupt doesn't trigger the HSB IMU stream, so no "
    "IMU data arrives (also on v2.7)"
)


@pytest.mark.skip(reason=IMU_SKIP_REASON)
@utils.requires_roce
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
    _run_example(vb1940_imu_player, arguments, capsys)


@pytest.mark.skip(reason=IMU_SKIP_REASON)
@utils.requires_roce
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
    # Left, right and IMU receivers.
    _run_example(vb1940_stereo_imu_player, arguments, capsys, receivers=3)


class FrameStartEndListener(hololink_module.AsyncEventListener):
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
        if not (state & (1 << SIF_0_FRAME_START)):
            return
        ts = self.get_timestamp(timestamp_s, timestamp_ns)
        with self._lock:
            self._sif0_starts.append(ts)

    def on_sif0_frame_end_event(self, state, timestamp_s, timestamp_ns):
        if not (state & (1 << SIF_0_FRAME_END)):
            return
        ts = self.get_timestamp(timestamp_s, timestamp_ns)
        with self._lock:
            self._sif0_ends.append(ts)

    def on_sif1_frame_start_event(self, state, timestamp_s, timestamp_ns):
        if not (state & (1 << SIF_1_FRAME_START)):
            return
        ts = self.get_timestamp(timestamp_s, timestamp_ns)
        with self._lock:
            self._sif1_starts.append(ts)

    def on_sif1_frame_end_event(self, state, timestamp_s, timestamp_ns):
        if not (state & (1 << SIF_1_FRAME_END)):
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


def _enable_sequencer(sequencer):
    sequencer.write_uint32(0x0000020C, 0xFFFFFFFF)
    sequencer.write_uint32(0x0000020C, 0x0)
    sequencer.enable()


def _test_vb1940_frame_events(frame_limit, hololink_address, sensor, headless):
    """Test frame start/end sequencers, ensuring that both events trigger and are reasonably spaced."""

    camera_mode = Vb1940_Mode.VB1940_MODE_2560X1984_30FPS
    frames_per_second = 30

    # Expect at least 80% of the total frame interval between events.
    min_frame_s = 0.8 / float(frames_per_second)

    listener = FrameStartEndListener()
    with applications.CudaContext() as cuda_context:
        # reset() restarts the HSB's clock at boot-relative time; wait for
        # PTP to re-lock before capturing
        camera = applications.open_vb1940(
            hololink_address, camera_mode, sensor=sensor, ptp=True
        )
        hololink = camera.device
        try:
            hololink.on_async_event(listener)
            # Enable the sequencers
            if sensor == 0:
                _enable_sequencer(hololink.sif0_frame_start_sequencer())
                _enable_sequencer(hololink.sif0_frame_end_sequencer())
            else:
                _enable_sequencer(hololink.sif1_frame_start_sequencer())
                _enable_sequencer(hololink.sif1_frame_end_sequencer())
            with utils.RunMonitor() as monitor:
                test_graph = applications.TestGraph("vb1940_frame_events")
                rx = applications.add_receiver(
                    test_graph,
                    "receiver",
                    hololink_module.operators.LinuxReceiverOp,
                    camera,
                    cuda_context.cu_context,
                    frame_limit,
                )
                isp = applications.add_isp(test_graph, "", rx, camera)
                applications.show_images(
                    test_graph,
                    "holoviz",
                    [
                        applications.Image(
                            isp["demosaic"].output, camera.width, camera.height
                        )
                    ],
                    headless,
                    "VB1940 frame events",
                )
                test_graph.run()
        finally:
            hololink.stop()
    monitor.assert_ok()

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
    _test_vb1940_frame_events(frame_limit, hololink_address, 0, headless)


@pytest.mark.skip_unless_vb1940
def test_vb1940_sif1_frame_events(headless, frame_limit, hololink_address):
    _test_vb1940_frame_events(frame_limit, hololink_address, 1, headless)
