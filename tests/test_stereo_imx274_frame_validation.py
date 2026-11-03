# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Stereo IMX274 frame validation: the FPGA CRC of each frame of both cameras must
# match the CRC computed on the GPU. Frame order, size and timing problems are
# logged as warnings only, as in v2.7.
#
# HSDK5 differences from v2.7:
# - CRCs are checked by ComputeCrcOp and CheckCrcOp through an hl.CrcChannel
#   (operators.add_crc_check), as in examples/imx274_frame_validation.py. HSDK5
#   operators can't be subclassed from Python.
# - HSDK5 has no selectable schedulers, so the test no longer runs once per
#   scheduler.
# - v2.7's cameras each registered a reset callback; here the test registers one
#   for each camera on the board's HololinkDevice, which open_imx274() resets.

import dataclasses
import logging
import math

import applications
import operators
import pytest
import utils
from hololink_module.sensors.imx274.imx274_mode import Imx274_Mode

import hololink as hololink_module

# Frames at the start whose CRC isn't checked.
SKIP_INITIAL_FRAMES = operators.COLOR_PROFILER_START_FRAME

# Frame buffers of each receiver. With 2 (the default), the FPGA overwrites a page
# within ~16.6 ms; under stereo 4K60 load the GPU CRC kernel can't always read
# the page before then, and the CRCs mismatch (as in v2.7).
MAX_PAGES = 5


class ResetCounter:
    """Counts the device resets, as v2.7's cameras did with their reset callbacks."""

    def __init__(self):
        self.count = 0

    def reset(self):
        self.count += 1


@dataclasses.dataclass
class FrameRecord:
    frame_number: int
    timestamp_s: int
    timestamp_ns: int
    bytes_written: int


def log_frame_validation(name, record, frame_size, expected_frame_time_s):
    """Log frame order, size and timing problems (informational, as in v2.7)."""
    order_errors = sum(
        1
        for previous, current in zip(record, record[1:])
        if current.frame_number != previous.frame_number + 1
    )
    size_errors = sum(1 for frame in record if frame.bytes_written != frame_size)
    timestamps = [frame.timestamp_s + frame.timestamp_ns * 1e-9 for frame in record]
    timestamp_errors = sum(
        1
        for previous, current in zip(timestamps, timestamps[1:])
        if current - previous > 2 * expected_frame_time_s
    )
    logging.info(
        f"{name}: {len(record)} frames, {order_errors} frame order errors, "
        f"{size_errors} frame size errors, {timestamp_errors} timestamp errors"
    )
    if order_errors or size_errors or timestamp_errors:
        logging.warning(f"{name} had frame validation errors (informational only)")


def run_stereo_frame_validation_test(
    channel_ip_left,
    ibv_name_left,
    ibv_port_left,
    channel_ip_right,
    ibv_name_right,
    ibv_port_right,
    camera_mode,
    frame_limit,
    headless,
):
    logging.info("Initializing stereo frame validation test")
    records = {"left": [], "right": []}
    checkers = {}
    # One HololinkDevice per board: this is the one that open_imx274() resets.
    device = hololink_module.DataChannel(
        hololink_module.find_channel(channel_ip_left)
    ).hololink()
    resets = {"left": ResetCounter(), "right": ResetCounter()}
    for counter in resets.values():
        device.on_reset(counter.reset)
    with applications.CudaContext() as cuda_context, applications.imx274_camera(
        channel_ip_left, camera_mode, expander_configuration=0, ptp=True
    ) as camera_left, utils.leg_watchdogs(
        ("left", "right"),
        initial_timeout=operators.color_profiler_initial_timeout(frame_limit),
        min_taps=utils.frame_floor(frame_limit),
    ) as watchdogs, utils.RunMonitor() as monitor:
        # The left camera's setup_clock() also sets the right camera's clock.
        camera_right = applications.open_imx274(
            channel_ip_right,
            camera_mode,
            expander_configuration=1,
            reset=False,
            device=camera_left.device,
        )
        assert camera_left.device is camera_right.channel.hololink()
        assert camera_left.device is device
        assert {name: counter.count for name, counter in resets.items()} == {
            "left": 1,
            "right": 1,
        }, "The reset didn't call each camera's reset callback once"
        assert camera_left.frame_size == camera_right.frame_size
        assert camera_left.bayer_format == camera_right.bayer_format
        assert camera_left.pixel_format == camera_right.pixel_format

        test_graph = applications.TestGraph("stereo_frame_validation")
        images = []
        for name, camera, ibv_name, ibv_port in (
            ("left", camera_left, ibv_name_left, ibv_port_left),
            ("right", camera_right, ibv_name_right, ibv_port_right),
        ):
            rx = applications.add_receiver(
                test_graph,
                f"receiver_{name}",
                hololink_module.operators.RoceReceiverOp,
                camera,
                cuda_context.cu_context,
                frame_limit,
                ibv_name=ibv_name,
                ibv_port=ibv_port,
                pages=MAX_PAGES,
                queue_size=MAX_PAGES,
            )
            checkers[name] = operators.add_crc_check(
                test_graph, name, rx, camera.frame_size
            )
            isp = applications.add_isp(test_graph, f"{name}_", rx, camera)
            record_operator = test_graph.op(
                operators.RecordMetadataOp,
                f"{name}_record",
                record_token=operators.register((FrameRecord, records[name])),
            )
            operators.metadata_flow(test_graph, rx.metadata, record_operator.metadata)
            watchdog_operator = test_graph.op(
                operators.WatchdogOp,
                f"{name}_watchdog",
                watchdog_token=operators.register(watchdogs[name]),
            )
            operators.metadata_flow(
                test_graph, isp["image_processor"].metadata, watchdog_operator.metadata
            )
            images.append(
                applications.Image(isp["demosaic"].output, camera.width, camera.height)
            )
        applications.show_images(
            test_graph, "holoviz", images, headless, "IMX274 frame validation"
        )
        logging.info("Running application")
        test_graph.run()

    monitor.assert_ok()
    expected_frame_time_s = 1 / 60
    for name in ("left", "right"):
        log_frame_validation(
            name, records[name], camera_left.frame_size, expected_frame_time_s
        )

    for name in ("left", "right"):
        result = checkers[name].result(skip_first=SKIP_INITIAL_FRAMES)
        settled = len(records[name]) - SKIP_INITIAL_FRAMES
        logging.info(
            f"{name} CRC: {result.checked} of {settled} frames checked, "
            f"{len(result.bad)} errors"
        )
        minimum = max(11, math.ceil(settled * operators.CRC_COVERAGE))
        assert (
            result.checked >= minimum
        ), f"{name}: only {result.checked} of {settled} frames' CRCs checked"
        assert not result.bad, f"{name}: {len(result.bad)} CRC errors: {result.bad[:5]}"


@utils.requires_roce
@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode",
    [
        Imx274_Mode.IMX274_MODE_3840X2160_60FPS,
        Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
    ],
)
@pytest.mark.parametrize(
    "frame_limit",
    [100],
)
def test_stereo_imx274_frame_validation(
    camera_mode,
    frame_limit,
    headless,
    channel_ips,
):
    """
    Test stereo IMX274 frame validation.

    FAIL CONDITIONS (test fails if any occur):
    - CRC mismatches on left camera
    - CRC mismatches on right camera
    - too few frames received or CRC-checked on either camera

    INFORMATIONAL (logged as warnings, does not fail test):
    - Frame ordering issues (dropped or out-of-order frames)
    - Frame size inconsistencies
    - Timestamp gaps or jitter
    """
    if len(channel_ips) < 2:
        pytest.skip("--channel-ips needs at least two IPs for this test")
    # Each leg's receiver uses the IBV device on its own network interface.
    ibv_name_left, ibv_name_right = "", ""
    ibv_port_left, ibv_port_right = 1, 1
    run_stereo_frame_validation_test(
        channel_ips[0],
        ibv_name_left,
        ibv_port_left,
        channel_ips[1],
        ibv_name_right,
        ibv_port_right,
        camera_mode,
        frame_limit,
        headless,
    )
