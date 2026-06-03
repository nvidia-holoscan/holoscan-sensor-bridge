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

import ctypes
import logging
import time

import cuda.bindings.driver as cuda
import holoscan
import operators
import pytest
import utils

import hololink as hololink_module


class DroppedFrameTestApplication(holoscan.core.Application):
    def __init__(
        self,
        headless,
        cuda_context,
        cuda_device_ordinal,
        hololink_channel,
        camera,
        camera_mode,
        frame_limit,
        watchdog,
        network_receiver_factory,
    ):
        logging.info("__init__")
        super().__init__()
        self._headless = headless
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._hololink_channel = hololink_channel
        self._camera = camera
        self._camera_mode = camera_mode
        self._frame_limit = frame_limit
        self._watchdog = watchdog
        self._network_receiver_factory = network_receiver_factory

    def compose(self):
        logging.info("compose")
        self._count = holoscan.conditions.CountCondition(
            self,
            name="count",
            count=self._frame_limit,
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
            num_blocks=4,
        )
        csi_to_bayer_operator = hololink_module.operators.CsiToBayerOp(
            self,
            name="csi_to_bayer",
            allocator=csi_to_bayer_pool,
            cuda_device_ordinal=self._cuda_device_ordinal,
        )
        self._camera.configure_converter(csi_to_bayer_operator)

        frame_size = csi_to_bayer_operator.get_csi_length()
        receiver_operator = self._network_receiver_factory(
            self,
            self._count,
            frame_size,
            self._cuda_context,
        )

        pixel_format = self._camera.pixel_format()
        bayer_format = self._camera.bayer_format()
        image_processor_operator = hololink_module.operators.ImageProcessorOp(
            self,
            name="image_processor",
            # Optical black value for imx274 is 50
            optical_black=50,
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
            num_blocks=4,
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

        visualizer = holoscan.operators.HolovizOp(
            self,
            name="holoviz",
            headless=self._headless,
            framebuffer_srgb=True,
            enable_camera_pose_output=True,
            camera_pose_output_type="extrinsics_model",
        )

        watchdog_operator = operators.WatchdogOp(
            self,
            name="watchdog_operator",
            watchdog=self._watchdog,
        )

        def stall():
            logging.info("STALLING")
            time.sleep(0.1)

        stall_operator = operators.OnFrameNOperator(
            self,
            name="stall_operator",
            trigger_frame=30,
            callback=stall,
        )

        #
        self.add_flow(receiver_operator, csi_to_bayer_operator, {("output", "input")})
        self.add_flow(
            csi_to_bayer_operator, image_processor_operator, {("output", "input")}
        )
        self.add_flow(image_processor_operator, demosaic, {("output", "receiver")})
        self.add_flow(demosaic, visualizer, {("transmitter", "receivers")})
        self.add_flow(visualizer, watchdog_operator, {("camera_pose_output", "input")})
        self.add_flow(visualizer, stall_operator, {("camera_pose_output", "input")})


def dropped_frame_test(
    camera_mode,
    pattern,
    headless,
    hololink,
    frame_limit,
    network_receiver_factory,
):
    # Get a handle to data sources
    channel_metadata = hololink_module.Enumerator.find_channel(channel_ip=hololink)
    logging.info("Initializing.")
    # Get a handle to the GPU
    (cu_result,) = cuda.cuInit(0)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_device_ordinal = 0
    cu_result, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_result, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    #
    hololink_channel = hololink_module.DataChannel(channel_metadata)
    # Get a handle to the camera
    camera = hololink_module.sensors.imx274.dual_imx274.Imx274Cam(
        hololink_channel, expander_configuration=0
    )
    with utils.Watchdog(
        "frame-reception",
        initial_timeout=operators.color_profiler_initial_timeout(frame_limit),
    ) as watchdog:
        # Set up the application
        def receiver_factory(fragment, condition, frame_size, frame_context):
            receiver_operator = network_receiver_factory(
                fragment,
                condition,
                name="receiver",
                frame_size=frame_size,
                frame_context=frame_context,
                hololink_channel=hololink_channel,
                device=camera,
            )
            return receiver_operator

        application = DroppedFrameTestApplication(
            headless,
            cu_context,
            cu_device_ordinal,
            hololink_channel,
            camera,
            camera_mode,
            frame_limit,
            watchdog,
            network_receiver_factory=receiver_factory,
        )
        # Run it.
        hololink = hololink_channel.hololink()
        hololink.start()
        try:
            hololink.reset()
            camera.setup_clock()
            camera.configure(camera_mode)
            camera.test_pattern(pattern)
            #
            application.run()
        finally:
            hololink.stop()

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS


# This may execute on unaccelerated configurations, where
# there may be any number of infiniband interfaces (but
# most likely zero).  Make sure we can address elements
# [0] and [1]; if those come back as None then the test
# will fail-- as expected.
sys_ibv_name = (hololink_module.infiniband_devices() + [None, None])[:2]


@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode, pattern",
    [
        (
            hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
            10,
        ),
    ],
)
@pytest.mark.parametrize(
    "hololink, ibv_name",
    [
        ("192.168.0.2", sys_ibv_name[0]),
    ],
)
def test_roce_imx274_dropped_frame(
    camera_mode,
    pattern,
    headless,
    hololink,
    ibv_name,
    frame_limit,
):
    def network_receiver_factory(
        fragment,
        condition,
        name,
        frame_size,
        frame_context,
        hololink_channel,
        device,
    ):
        ibv_port = 1
        receiver_operator = hololink_module.operators.RoceReceiverOp(
            fragment,
            condition,
            name=name,
            frame_size=frame_size,
            frame_context=frame_context,
            hololink_channel=hololink_channel,
            device=device,
            ibv_name=ibv_name,
            ibv_port=ibv_port,
        )
        return receiver_operator

    dropped_frame_test(
        camera_mode,
        pattern,
        headless,
        hololink,
        frame_limit,
        network_receiver_factory,
    )
