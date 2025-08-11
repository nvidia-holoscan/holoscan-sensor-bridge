# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import ctypes
import datetime
import logging
import sys
import threading
from unittest import mock

import holoscan
import operators
import pytest
import utils

import hololink as hololink_module
from examples import imx477_player

timestamps = None
network_mode = None
roce_network_mode = "ROCE"
linux_network_mode = "Linux"


class TimestampTestApplication(holoscan.core.Application):
    def __init__(
        self,
        headless,
        fullscreen,
        cuda_context,
        cuda_device_ordinal,
        hololink_channel,
        ibv_name,
        ibv_port,
        camera,
        frame_limit,
    ):
        logging.info("__init__")
        super().__init__()
        self._headless = headless
        self._fullscreen = fullscreen
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._hololink_channel = hololink_channel
        self._ibv_name = ibv_name
        self._ibv_port = ibv_port
        self._camera = camera
        self._frame_limit = frame_limit
        self._lock = threading.Lock()
        self._timestamps = []
        self.enable_metadata(True)

    def compose(self):
        logging.info("compose")
        self._condition = holoscan.conditions.BooleanCondition(
            self, name="ok", enable_tick=True
        )

        csi_to_bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="pool",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=self._camera._width
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera._height,
            num_blocks=2,
        )
        csi_to_bayer_operator = hololink_module.operators.CsiToBayerOp(
            self,
            name="csi_to_bayer",
            allocator=csi_to_bayer_pool,
            cuda_device_ordinal=self._cuda_device_ordinal,
        )
        self._camera.configure_converter(csi_to_bayer_operator)

        frame_size = csi_to_bayer_operator.get_csi_length()
        logging.info(f"{frame_size=}")
        frame_context = self._cuda_context

        global network_mode, roce_network_mode, linux_network_mode
        if network_mode == roce_network_mode:
            receiver_operator = hololink_module.operators.RoceReceiverOp(
                self,
                self._condition,
                name="receiver",
                frame_size=frame_size,
                frame_context=frame_context,
                ibv_name=self._ibv_name,
                ibv_port=self._ibv_port,
                hololink_channel=self._hololink_channel,
                device=self._camera,
            )
        elif network_mode == linux_network_mode:
            receiver_operator = hololink_module.operators.LinuxReceiverOperator(
                self,
                self._condition,
                name="receiver",
                frame_size=frame_size,
                frame_context=frame_context,
                hololink_channel=self._hololink_channel,
                device=self._camera,
            )
        else:
            assert False and f"Invalid {network_mode=}"

        rgba_components_per_pixel = 4
        bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="pool",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=self._camera._width
            * rgba_components_per_pixel
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera._height,
            num_blocks=2,
        )
        bayer_format = self._camera.bayer_format()
        demosaic = holoscan.operators.BayerDemosaicOp(
            self,
            name="demosaic",
            pool=bayer_pool,
            generate_alpha=True,
            alpha_value=65535,
            bayer_grid_pos=bayer_format.value,
            interpolation_mode=0,
        )
        profiler = operators.TimeProfiler(
            self,
            name="profiler",
            callback=self.time_profile,
        )
        visualizer = holoscan.operators.HolovizOp(
            self,
            name="holoviz",
            fullscreen=self._fullscreen,
            headless=self._headless,
            enable_camera_pose_output=True,
            camera_pose_output_type="extrinsics_model",
        )
        initial_timeout = utils.timeout_sequence(
            [(30, 20), (0.5, self._frame_limit - 40), (30, 1)]
        )
        self._watchdog = utils.Watchdog(
            name="watchdog",
            initial_timeout=initial_timeout,
        )
        watchdog_operator = operators.WatchdogOp(
            self,
            name="watchdog_operator",
            watchdog=self._watchdog,
        )
        #
        self.add_flow(receiver_operator, csi_to_bayer_operator, {("output", "input")})
        self.add_flow(csi_to_bayer_operator, demosaic, {("output", "receiver")})
        self.add_flow(demosaic, profiler, {("transmitter", "input")})
        self.add_flow(profiler, visualizer, {("output", "receivers")})
        self.add_flow(visualizer, watchdog_operator, {("camera_pose_output", "input")})
        #
        self._watchdog.start()

    def time_profile(
        self,
        image_timestamp_s,
        metadata_timestamp_s,
        received_timestamp_s,
        pipeline_timestamp_s,
        frame_number,
    ):
        with self._lock:
            self._timestamps.append(
                (
                    image_timestamp_s,
                    metadata_timestamp_s,
                    received_timestamp_s,
                    pipeline_timestamp_s,
                    frame_number,
                )
            )
            if len(self._timestamps) >= self._frame_limit:
                self._watchdog.stop()
                self._condition.disable_tick()
                global timestamps
                timestamps = self._timestamps


# frame_time represents the constant time difference between when the
#   frame-start and frame-end messages arrive at the FPGA.
# time_limit, the acceptable amount of time between when the frame was sent and
#   when we got around to looking at it, is much smaller in the RDMA
#   configuration.
@pytest.mark.skip_unless_ptp
@pytest.mark.skip_unless_imx477
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "roce_mode, frame_time, time_limit",  # noqa: E501
    [
        (
            True,
            0.015,
            0.004,
        ),
        (
            False,
            0.015,
            0.012,
        ),
    ],
)
def test_imx477_timestamps(
    roce_mode,
    frame_time,
    time_limit,
    headless,
    hololink_address,
    ibv_name,
    ibv_port,
    frame_limit,
):
    arguments = [
        sys.argv[0],
        "--hololink",
        hololink_address,
        "--ibv-name",
        ibv_name,
        "--ibv-port",
        str(ibv_port),
        "--pattern",
        "--ptp-sync",
        f"--frame-limit={frame_limit}",
    ]
    if headless:
        arguments.extend(["--headless"])

    global network_mode, roce_network_mode, linux_network_mode
    if roce_mode:
        network_mode = roce_network_mode
    else:
        network_mode = linux_network_mode

    with mock.patch("sys.argv", arguments):
        with mock.patch(
            "examples.imx477_player.HoloscanApplication", TimestampTestApplication
        ):
            with utils.PriorityScheduler():
                imx477_player.main()

    # check for errors
    global timestamps
    pipeline_dts, receiver_dts = [], []
    metadata_receiver_dts = []
    # Allow for startup times to be a bit longer
    settled_timestamps = timestamps[5:-5]
    assert len(settled_timestamps) >= 100
    for (
        image_timestamp_s,
        metadata_timestamp_s,
        received_timestamp_s,
        pipeline_timestamp_s,
        frame_number,
    ) in settled_timestamps:
        image_timestamp = datetime.datetime.fromtimestamp(image_timestamp_s).isoformat()
        metadata_timestamp = datetime.datetime.fromtimestamp(
            metadata_timestamp_s
        ).isoformat()
        received_timestamp = datetime.datetime.fromtimestamp(
            received_timestamp_s
        ).isoformat()
        pipeline_timestamp = datetime.datetime.fromtimestamp(
            pipeline_timestamp_s
        ).isoformat()
        pipeline_dt = pipeline_timestamp_s - image_timestamp_s
        logging.debug(
            f"{image_timestamp=} {pipeline_timestamp=} {pipeline_dt=:0.6f} {frame_number=}"
        )
        pipeline_dts.append(round(pipeline_dt, 4))
        receiver_dt = received_timestamp_s - image_timestamp_s
        logging.debug(
            f"{image_timestamp=} {received_timestamp=} {receiver_dt=:0.6f} {frame_number=}"
        )
        receiver_dts.append(round(receiver_dt, 4))
        metadata_receiver_dt = received_timestamp_s - metadata_timestamp_s
        logging.debug(
            f"{metadata_timestamp=} {received_timestamp=} {metadata_receiver_dt=:0.6f} {frame_number=}"
        )
        metadata_receiver_dts.append(round(metadata_receiver_dt, 4))

    smallest_time_difference = min(pipeline_dts)
    largest_time_difference = max(pipeline_dts)
    logging.info(f"pipeline {smallest_time_difference=} {largest_time_difference=}")
    #
    smallest_time_difference = min(receiver_dts)
    largest_time_difference = max(receiver_dts)
    logging.info(f"receiver {smallest_time_difference=} {largest_time_difference=}")
    # frame_time is passed in from above and represents the constant time
    # difference between when the frame-start and frame-end messages arrive at
    # the FPGA.  The time we get with the frame data is captured at frame-start
    # time but isn't delivered to us until the frame-end is sent.  For us to
    # check the validity of the timestamp, we check that the timestamp received
    # with the frame, plus this constant offset, is within (time_limit) of the
    # reception time recorded by the host.  Reception time is recorded when the
    # last frame data is transmitted to us.
    assert (frame_time + 0) <= smallest_time_difference
    assert smallest_time_difference < largest_time_difference
    assert largest_time_difference < (frame_time + time_limit)
    #
    smallest_time_difference = min(metadata_receiver_dts)
    largest_time_difference = max(metadata_receiver_dts)
    logging.info(
        f"FPGA to full frame received {smallest_time_difference=} {largest_time_difference=}"
    )
    assert smallest_time_difference < largest_time_difference
