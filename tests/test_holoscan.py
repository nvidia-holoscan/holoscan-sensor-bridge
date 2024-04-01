# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import logging

import pytest
import udp_server
import utils
from cuda import cuda

import hololink as hololink_module

holoscan = pytest.importorskip(
    "holoscan", reason="Use 'pip3 install holoscan' to enable holoscan tests."
)


class WatchdogOperator(holoscan.core.Operator):
    def __init__(self, *args, camera=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert camera is not None
        self._camera = camera

    def setup(self, spec):
        logging.info("setup")
        spec.input("input")

    def start(self):
        pass

    def stop(self):
        pass

    def compute(self, op_input, op_output, context):
        # Get input message; we have to do this
        # otherwise Holoscan is unhappy
        op_input.receive("input")
        self._camera.tap_watchdog()


class HoloscanApplication(holoscan.core.Application):
    def __init__(
        self,
        cuda_context,
        cuda_device_ordinal,
        hololink_channel,
        camera,
        camera_mode,
        frame_limit=10,
        headless=False,
    ):
        logging.info("__init__")
        super().__init__()
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._hololink_channel = hololink_channel
        self._camera = camera
        self._camera_mode = camera_mode
        self._frame_limit = frame_limit
        self._headless = headless

    def compose(self):
        logging.info("compose")
        cuda.cuCtxSetCurrent(self._cuda_context)

        self._camera.configure(*self._camera_mode)

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
        frame_context = self._cuda_context
        receiver_operator = hololink_module.operators.LinuxReceiverOperator(
            self,
            holoscan.conditions.CountCondition(self, count=self._frame_limit),
            name="receiver",
            frame_size=frame_size,
            frame_context=frame_context,
            hololink_channel=self._hololink_channel,
            device=self._camera,
        )

        bayer_format = self._camera.bayer_format()
        pixel_format = self._camera.pixel_format()
        image_processor_operator = hololink_module.operators.ImageProcessorOp(
            self,
            name="image_processor",
            bayer_format=bayer_format.value,
            pixel_format=pixel_format.value,
        )

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
        demosaic = holoscan.operators.BayerDemosaicOp(
            self,
            name="demosaic",
            pool=bayer_pool,
            generate_alpha=True,
            alpha_value=65535,
            bayer_grid_pos=bayer_format.value,
            interpolation_mode=0,
        )

        gamma_correction = hololink_module.operators.GammaCorrectionOp(
            self,
            name="gamma_correction",
            cuda_device_ordinal=self._cuda_device_ordinal,
        )

        visualizer = holoscan.operators.HolovizOp(
            self,
            name="holoviz",
            headless=self._headless,
        )

        watchdog_operator = WatchdogOperator(
            self,
            name="watchdog",
            camera=self._camera,
        )

        #
        self.add_flow(receiver_operator, csi_to_bayer_operator, {("output", "input")})
        self.add_flow(
            csi_to_bayer_operator, image_processor_operator, {("output", "input")}
        )
        self.add_flow(image_processor_operator, demosaic, {("output", "receiver")})
        self.add_flow(demosaic, gamma_correction, {("transmitter", "input")})
        self.add_flow(gamma_correction, visualizer, {("output", "receivers")})
        self.add_flow(gamma_correction, watchdog_operator, {("output", "input")})


frame_rate_s = 0.1


@pytest.mark.parametrize(
    "height, width, bayer_format, pixel_format, frame_rate_s",
    [
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
    ],
)
def test_holoscan(
    height,
    width,
    bayer_format,
    pixel_format,
    frame_rate_s,
    headless,
    udpcam,
    frame_limit=10,
):
    #
    image, _ = utils.make_image(height, width, bayer_format, pixel_format)

    with udp_server.TestServer(udpcam) as server:
        # Get a handle to the GPU
        (cu_result,) = cuda.cuInit(0)
        assert cu_result == cuda.CUresult.CUDA_SUCCESS
        cu_device_ordinal = 0
        cu_result, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
        assert cu_result == cuda.CUresult.CUDA_SUCCESS
        cu_result, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
        assert cu_result == cuda.CUresult.CUDA_SUCCESS

        # Get the data plane controller.
        channel_metadata = server.channel_metadata()
        hololink_channel = hololink_module.HololinkDataChannel(channel_metadata)

        # Camera
        camera = hololink_module.sensors.udp_cam.UdpCam(hololink_channel)
        camera_mode = (height, width, bayer_format, pixel_format, frame_rate_s)

        # Here's our holoscan application
        logging.info("Initializing.")
        application = HoloscanApplication(
            cu_context,
            cu_device_ordinal,
            hololink_channel,
            camera,
            camera_mode,
            headless=headless,
            frame_limit=frame_limit,
        )
        hololink = hololink_channel.hololink()
        hololink.start()
        application.run()
        hololink.stop()
