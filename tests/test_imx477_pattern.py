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
import logging
import os

import holoscan
import operators
import pytest
from cuda import cuda

import hololink as hololink_module

actual = None


def reset_globals():
    global actual
    actual = None


class PatternTestApplication(holoscan.core.Application):
    def __init__(
        self,
        headless,
        cuda_context,
        cuda_device_ordinal,
        hololink_channel,
        camera,
    ):
        logging.info("__init__")
        super().__init__()
        self._headless = headless
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._hololink_channel = hololink_channel
        self._camera = camera

    def compose(self):
        logging.info("compose")
        self._ok = holoscan.conditions.BooleanCondition(
            self, name="ok", enable_tick=True
        )
        condition = self._ok

        self._camera.set_mode()

        #
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

        #
        frame_size = csi_to_bayer_operator.get_csi_length()
        logging.info(f"{frame_size=}")
        frame_context = self._cuda_context
        receiver_operator = hololink_module.operators.LinuxReceiverOperator(
            self,
            condition,
            name="receiver",
            frame_size=frame_size,
            frame_context=frame_context,
            hololink_channel=self._hololink_channel,
            device=self._camera,
        )
        #
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

        #
        profiler = operators.ColorProfiler(
            self,
            name="profiler",
            callback=lambda buckets: self.buckets(buckets),
            out_tensor_name="1",
        )

        visualizer = holoscan.operators.HolovizOp(
            self,
            name="holoviz",
            fullscreen=not self._headless,
            headless=self._headless,
        )

        self.add_flow(receiver_operator, csi_to_bayer_operator, {("output", "input")})
        self.add_flow(csi_to_bayer_operator, demosaic, {("output", "receiver")})
        self.add_flow(demosaic, profiler, {("transmitter", "input")})
        self.add_flow(profiler, visualizer, {("output", "receivers")})

    def _check_done(self):
        global actual
        logging.trace(f"{actual=}")
        if actual is None:
            return
        logging.info("DONE")
        self._ok.disable_tick()

    def buckets(self, buckets):
        global actual
        if actual is None:
            actual = buckets
        self._check_done()


@pytest.mark.skip_unless_imx477
@pytest.mark.parametrize(
    "camera_mode,expected",
    [
        (
            [3840, 2160],
            # fmt: on
            [
                641520,
                4320,
                1090800,
                2160,
                0,
                1088640,
                4320,
                1090800,
                0,
                1088640,
                4320,
                1090800,
                4320,
                0,
                1088640,
                1095120,
            ],  #
        ),
    ],
)
def test_imx477_pattern(
    camera_mode,
    expected,
    headless,
    hololink_address,
):

    logging.info("Initializing.")
    #
    reset_globals()
    # Get a handle to the GPU
    (cu_result,) = cuda.cuInit(0)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_device_ordinal = 0
    cu_result, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_result, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    # Get a handle to data sources

    channel_metadata = hololink_module.Enumerator.find_channel(
        channel_ip=hololink_address
    )
    hololink_channel = hololink_module.DataChannel(channel_metadata)

    # Get a handle to the camera
    camera_id = 0
    camera = hololink_module.sensors.imx477.Imx477(hololink_channel, camera_id)
    #
    # Set up the application
    application = PatternTestApplication(
        headless,
        cu_context,
        cu_device_ordinal,
        hololink_channel,
        camera,
    )
    default_configuration = os.path.join(
        os.path.dirname(__file__), "example_configuration.yaml"
    )

    application.config(default_configuration)
    # Run it.
    hololink = hololink_channel.hololink()
    hololink.start()

    hololink.reset()

    camera.configure()
    camera.set_pattern()

    # Prove that get_register works as expected.
    pattern_register = camera.get_register(0x601)
    assert pattern_register == 0x2

    application.run()
    hololink.stop()

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS

    # Now check the buckets.
    global actual

    logging.info(f"{expected=}")
    logging.info(f"{actual=}")
    diffs = [abs(a - e) for e, a in zip(expected, actual, strict=True)]
    logging.info(f"{diffs=}")
    diff = sum(diffs)
    logging.info(f"{diff=}")

    assert 0 <= diff < 4
