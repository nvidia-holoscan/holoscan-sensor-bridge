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

import applications
import holoscan
import operators
import pytest
import utils

import hololink as hololink_module
from examples import imx274_player

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
        camera_mode,
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
        self._camera_mode = camera_mode
        self._frame_limit = frame_limit
        self._lock = threading.Lock()
        self._timestamps = []
        self.enable_metadata(True)

    def compose(self):
        logging.info("compose")
        self._condition = holoscan.conditions.BooleanCondition(
            self, name="ok", enable_tick=True
        )
        self._camera.set_mode(self._camera_mode)

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
#   frame-start and frame-end messages arrive at the FPGA; for IMX274
#   it takes just under 8ms for a 1080p or almost 16ms for a 4k image.
# time_limit, the acceptable amount of time between when the frame was sent and
#   when we got around to looking at it, is much smaller in the RDMA
#   configuration.
@pytest.mark.skip_unless_ptp
@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode, roce_mode, frame_time, time_limit, max_recv_time",  # noqa: E501
    [
        (
            hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_3840X2160_60FPS,  # noqa: E501
            True,
            0.015,
            0.004,
            0.0015,
        ),
        (
            hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,  # noqa: E501
            True,
            0.0075,
            0.0040,
            0.0015,
        ),
        (
            hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_3840X2160_60FPS,  # noqa: E501
            False,
            0.015,
            0.012,
            0.0015,
        ),
        (
            hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
            False,
            0.0075,
            0.0120,
            0.0035,
        ),
    ],
)
def test_imx274_timestamps(
    camera_mode,
    roce_mode,
    frame_time,
    time_limit,
    max_recv_time,
    headless,
    hololink_address,
    ibv_name,
    ibv_port,
    frame_limit,
):
    pattern = 10
    arguments = [
        sys.argv[0],
        "--camera-mode",
        str(camera_mode.value),
        "--hololink",
        hololink_address,
        "--ibv-name",
        ibv_name,
        "--ibv-port",
        str(ibv_port),
        f"--pattern={pattern}",
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
            "examples.imx274_player.HoloscanApplication", TimestampTestApplication
        ):
            with utils.PriorityScheduler():
                imx274_player.main()

    # check for errors
    global timestamps
    pipeline_dts, receiver_dts = [], []
    metadata_receiver_dts = []
    # Allow for startup times to be a bit longer
    settled_timestamps = timestamps[5:-5]
    assert len(settled_timestamps) >= 100
    last_image_timestamp_s = None
    last_received_timestamp_s = None
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
        receiver_dts.append(round(receiver_dt, 4))
        metadata_receiver_dt = received_timestamp_s - metadata_timestamp_s
        logging.debug(
            f"{image_timestamp=} {metadata_timestamp=} {received_timestamp=} {receiver_dt=:0.6f} {metadata_receiver_dt=:0.6f} {frame_number=}"
        )
        metadata_receiver_dts.append(round(metadata_receiver_dt, 4))
        if last_image_timestamp_s is not None:
            time_from_last_image_s = image_timestamp_s - last_image_timestamp_s
            time_from_last_received_s = received_timestamp_s - last_received_timestamp_s
            logging.debug(
                f"{time_from_last_image_s=:.4f} {time_from_last_received_s=:.4f}"
            )
        last_image_timestamp_s = image_timestamp_s
        last_received_timestamp_s = received_timestamp_s

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
    average_time_difference = sum(metadata_receiver_dts) / len(metadata_receiver_dts)
    logging.info(
        f"FPGA to full frame received {smallest_time_difference=} {largest_time_difference=}"
    )
    assert smallest_time_difference < largest_time_difference
    # The time taken from the end of image frame received at HSB fpga to full frame
    # received on IGX should be less than max_recv_time on average.
    assert average_time_difference < max_recv_time


class TimestampTestApplicationWithRenamedMetadata(holoscan.core.Application):
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
        camera_mode,
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
        self._camera_mode = camera_mode
        self._frame_limit = frame_limit
        self._lock = threading.Lock()
        self._timestamps = []
        self.enable_metadata(True)

    def compose(self):
        logging.info("compose")
        self._condition = holoscan.conditions.BooleanCondition(
            self, name="ok", enable_tick=True
        )
        self._camera.set_mode(self._camera_mode)

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

        # Test the rename_metadata functionality by adding a prefix
        def rename_metadata(name):
            return f"roce_{name}"

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
                rename_metadata=rename_metadata,
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
                rename_metadata=rename_metadata,
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
            rename_metadata=rename_metadata,
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


@pytest.mark.skip_unless_ptp
@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode, roce_mode, frame_time, time_limit, max_recv_time",  # noqa: E501
    [
        (
            hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
            True,
            0.0075,
            0.0040,
            0.0015,
        ),
    ],
)
def test_imx274_timestamps_with_renamed_metadata(
    camera_mode,
    roce_mode,
    frame_time,
    time_limit,
    max_recv_time,
    headless,
    hololink_address,
    ibv_name,
    ibv_port,
    frame_limit,
):
    """Test that the rename_metadata functionality works correctly."""
    pattern = 10
    arguments = [
        sys.argv[0],
        "--camera-mode",
        str(camera_mode.value),
        "--hololink",
        hololink_address,
        "--ibv-name",
        ibv_name,
        "--ibv-port",
        str(ibv_port),
        f"--pattern={pattern}",
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
            "examples.imx274_player.HoloscanApplication",
            TimestampTestApplicationWithRenamedMetadata,
        ):
            with utils.PriorityScheduler():
                imx274_player.main()

    # check for errors
    global timestamps
    pipeline_dts, receiver_dts = [], []
    metadata_receiver_dts = []
    # Allow for startup times to be a bit longer
    settled_timestamps = timestamps[5:-5]
    assert len(settled_timestamps) >= 100
    last_image_timestamp_s = None
    last_received_timestamp_s = None
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
        if last_image_timestamp_s is not None:
            time_from_last_image_s = image_timestamp_s - last_image_timestamp_s
            time_from_last_received_s = received_timestamp_s - last_received_timestamp_s
            logging.debug(
                f"{time_from_last_image_s=:.4f} {time_from_last_received_s=:.4f}"
            )
        last_image_timestamp_s = image_timestamp_s
        last_received_timestamp_s = received_timestamp_s

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
    assert smallest_time_difference <= largest_time_difference
    assert largest_time_difference < (frame_time + time_limit)
    #
    smallest_time_difference = min(metadata_receiver_dts)
    largest_time_difference = max(metadata_receiver_dts)
    average_time_difference = sum(metadata_receiver_dts) / len(metadata_receiver_dts)
    logging.info(
        f"FPGA to full frame received {smallest_time_difference=} {largest_time_difference=}"
    )
    assert smallest_time_difference < largest_time_difference
    # The time taken from the end of image frame received at HSB fpga to full frame
    # received on IGX should be less than max_recv_time on average.
    assert average_time_difference < max_recv_time


@pytest.mark.skip_unless_ptp
@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
    ],
)
def test_imx274_stereo_roce_naive(
    camera_mode,
    headless,
    frame_limit,
    hololink_address,
    ptp_enable,
    ibv_name,
    ibv_port,
):
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
            dt = right_frame_start - left_frame_start
            logging.info(f"{left_frame_start=} {right_frame_start=} {dt=}")

    left_pattern = 10
    right_pattern = 11
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
        left_camera_factory=lambda channel, instance: applications.imx274_camera_factory(
            channel,
            instance,
            camera_mode,
            left_pattern,
        ),
        right_camera_factory=lambda channel, instance: applications.imx274_camera_factory(
            channel,
            instance,
            camera_mode,
            right_pattern,
        ),
        ptp_enable=ptp_enable,
    )
    stereo_test.execute()
