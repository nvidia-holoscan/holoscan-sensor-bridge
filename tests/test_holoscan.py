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

# HSDK5 differences from v2.7: the pipeline is an hs.Graph. The receiver ends
# the run at frame_limit frames; v2.7 used a CountCondition.
# The test also checks that the receiver emitted frames and that the run ended
# well (utils.RunMonitor).

import logging

import applications
import mock_camera
import mock_server
import operators
import pytest
import utils

import hololink as hololink_module


class CameraWatchdog:
    """Taps the mock camera's watchdog for each frame, as v2.7's WatchdogOperator
    did; the mock camera stops sending when it isn't tapped."""

    def __init__(self, camera):
        self._camera = camera

    def tap(self):
        self._camera.tap_watchdog()


def run_holoscan_test(
    server, camera_mode, frame_limit, receiver, receiver_kwargs, headless
):
    with applications.CudaContext() as cuda_context:
        # Get the data plane controller.
        channel_metadata = server.channel_metadata()
        hololink_channel = hololink_module.DataChannel(channel_metadata)
        hololink = hololink_channel.hololink()
        # Camera
        camera = mock_camera.MockCamera(hololink_channel)
        hololink.start()
        try:
            camera.configure(*camera_mode)
            geometry = camera.geometry()
            sensor = applications.SensorCamera(
                metadata=channel_metadata,
                channel=hololink_channel,
                device=hololink,
                camera=camera,
                geometry=geometry,
                frame_size=geometry["start_byte"]
                + geometry["bytes_per_line"] * geometry["pixel_height"]
                + geometry["trailing_bytes"],
                pixel_format=camera.get_pixel_format().value,
                bayer_format=camera.get_bayer_format().name.lower(),
                width=camera.get_width(),
                height=camera.get_height(),
            )
            # Here's our holoscan application
            logging.info("Initializing.")
            with utils.RunMonitor() as monitor:
                test_graph = applications.TestGraph("holoscan")
                rx = applications.add_receiver(
                    test_graph,
                    "receiver",
                    receiver,
                    sensor,
                    cuda_context.cu_context,
                    frame_limit,
                    **receiver_kwargs,
                )
                isp = applications.add_isp(test_graph, "", rx, sensor)
                applications.show_images(
                    test_graph,
                    "holoviz",
                    [
                        applications.Image(
                            isp["demosaic"].output, sensor.width, sensor.height
                        )
                    ],
                    headless,
                    "Mock camera",
                )
                watchdog_operator = test_graph.op(
                    operators.WatchdogOp,
                    "watchdog",
                    watchdog_token=operators.register(CameraWatchdog(camera)),
                )
                operators.metadata_flow(
                    test_graph,
                    isp["image_processor"].metadata,
                    watchdog_operator.metadata,
                )
                test_graph.run()
        finally:
            hololink.stop()
    # The run is frame_limit frames long; the last one isn't counted.
    monitor.assert_ok(min_frames=min(11, frame_limit // 2))


frame_rate_s = 0.1
formats = [
    (
        720,
        1280,
        hololink_module.sensors.csi.BayerFormat.RGGB,
        hololink_module.sensors.csi.PixelFormat.RAW_8,
        frame_rate_s,
    ),
    (
        720,
        1280,
        hololink_module.sensors.csi.BayerFormat.GBRG,
        hololink_module.sensors.csi.PixelFormat.RAW_8,
        frame_rate_s,
    ),
    (
        720,
        1280,
        hololink_module.sensors.csi.BayerFormat.RGGB,
        hololink_module.sensors.csi.PixelFormat.RAW_12,
        frame_rate_s,
    ),
    (
        720,
        1280,
        hololink_module.sensors.csi.BayerFormat.GBRG,
        hololink_module.sensors.csi.PixelFormat.RAW_12,
        frame_rate_s,
    ),
    (
        720,
        1280,
        hololink_module.sensors.csi.BayerFormat.RGGB,
        hololink_module.sensors.csi.PixelFormat.RAW_10,
        frame_rate_s,
    ),
    (
        720,
        1280,
        hololink_module.sensors.csi.BayerFormat.GBRG,
        hololink_module.sensors.csi.PixelFormat.RAW_10,
        frame_rate_s,
    ),
]


@pytest.mark.parametrize(
    "height, width, bayer_format, pixel_format, frame_rate_s",
    formats,
)
def test_holoscan(
    height,
    width,
    bayer_format,
    pixel_format,
    frame_rate_s,
    headless,
    mock_camera_ip,
    frame_limit=10,
):
    #
    image, _ = utils.make_image(height, width, bayer_format, pixel_format)

    with mock_server.TestServer(mock_camera_ip) as server:
        camera_mode = (height, width, bayer_format, pixel_format, frame_rate_s)
        run_holoscan_test(
            server,
            camera_mode,
            frame_limit,
            hololink_module.operators.LinuxReceiverOp,
            {},
            headless,
        )


@utils.requires_roce
@pytest.mark.skip_unless_mock_camera
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "height, width, bayer_format, pixel_format, frame_rate_s",
    formats,
)
def test_network_accelerated_holoscan(
    height,
    width,
    bayer_format,
    pixel_format,
    frame_rate_s,
    headless,
    mock_camera_ip,
    ibv_name,
    ibv_port,
    frame_limit=10,
):
    #
    image, _ = utils.make_image(height, width, bayer_format, pixel_format)

    with mock_server.TestServer(mock_camera_ip) as server:
        camera_mode = (height, width, bayer_format, pixel_format, frame_rate_s)
        run_holoscan_test(
            server,
            camera_mode,
            frame_limit,
            hololink_module.operators.RoceReceiverOp,
            {"ibv_name": ibv_name, "ibv_port": ibv_port},
            headless,
        )
