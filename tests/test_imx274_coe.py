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
import os
import sys
from unittest import mock

import holoscan
import operators
import pytest
import utils
from cuda import cuda

import hololink as hololink_module
from examples import linux_coe_imx274_player


# Unaccelerated, single camera test
@pytest.mark.skip_unless_coe
@pytest.mark.skip_unless_imx274
def test_coe_linux_player(headless, frame_limit, coe_interfaces, capsys):
    arguments = [
        sys.argv[0],
        "--frame-limit",
        str(frame_limit),
        "--coe-interface",
        coe_interfaces[0],
    ]
    if headless:
        arguments.extend(["--headless"])

    with mock.patch("sys.argv", arguments):
        linux_coe_imx274_player.main()

    # check for errors
    captured = capsys.readouterr()
    assert captured.err == ""


class DualCoeTestApplication(holoscan.core.Application):
    def __init__(
        self,
        frame_limit,
        headless,
        cuda_context,
        cuda_device_ordinal,
        hololink_channel_left,
        camera_left,
        camera_mode_left,
        hololink_channel_right,
        camera_right,
        camera_mode_right,
        coe_interface_left,
        coe_interface_right,
        watchdog,
    ):
        logging.info("__init__")
        super().__init__()
        self._frame_limit = frame_limit
        self._headless = headless
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._hololink_channel_left = hololink_channel_left
        self._hololink_left = hololink_channel_left.hololink()
        self._camera_left = camera_left
        self._camera_mode_left = camera_mode_left
        self._coe_interface_left = coe_interface_left
        self._hololink_channel_right = hololink_channel_right
        self._hololink_right = hololink_channel_right.hololink()
        self._camera_right = camera_right
        self._camera_mode_right = camera_mode_right
        self._coe_interface_right = coe_interface_right
        self._watchdog = watchdog
        # Each camera sharing a network connection must use
        # a unique channel number from 0..63.
        self._coe_channel_left = 1
        self._coe_channel_right = 2
        # These are HSDK controls-- because we have both the
        # image data and postprocessor output
        # going into the same visualizer, don't
        # raise an error when each path present metadata
        # with the same names.  Because we don't use that metadata,
        # it's easiest to just ignore new items with the same
        # names as existing items.
        self.enable_metadata(True)
        self.metadata_policy = holoscan.core.MetadataPolicy.REJECT

    def compose(self):
        logging.info("compose")
        self._condition_left = holoscan.conditions.CountCondition(
            self,
            name="condition_left",
            count=self._frame_limit,
        )
        self._condition_right = holoscan.conditions.CountCondition(
            self,
            name="condition_right",
            count=self._frame_limit,
        )

        self._camera_left.set_mode(self._camera_mode_left)
        self._camera_right.set_mode(self._camera_mode_right)

        converter_pool_left = holoscan.resources.BlockMemoryPool(
            self,
            name="pool_left",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=self._camera_left._width
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera_left._height,
            num_blocks=4,
        )
        converter_left = hololink_module.operators.PackedFormatConverterOp(
            self,
            name="converter_left",
            allocator=converter_pool_left,
            cuda_device_ordinal=self._cuda_device_ordinal,
            hololink_channel=self._hololink_channel_left,
        )
        self._camera_left.configure_converter(converter_left)

        converter_pool_right = holoscan.resources.BlockMemoryPool(
            self,
            name="pool_right",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=self._camera_right._width
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera_right._height,
            num_blocks=4,
        )
        converter_right = hololink_module.operators.PackedFormatConverterOp(
            self,
            name="converter_right",
            allocator=converter_pool_right,
            cuda_device_ordinal=self._cuda_device_ordinal,
            hololink_channel=self._hololink_channel_right,
        )
        self._camera_right.configure_converter(converter_right)

        receiver_left = hololink_module.operators.LinuxCoeReceiverOp(
            self,
            self._condition_left,
            name="receiver_left",
            frame_size=converter_left.get_frame_size(),
            frame_context=self._cuda_context,
            hololink_channel=self._hololink_channel_left,
            device=self._camera_left,
            coe_interface=self._coe_interface_left,
            pixel_width=self._camera_left._width,
            coe_channel=self._coe_channel_left,
        )
        receiver_right = hololink_module.operators.LinuxCoeReceiverOp(
            self,
            self._condition_right,
            name="receiver_right",
            frame_size=converter_right.get_frame_size(),
            frame_context=self._cuda_context,
            hololink_channel=self._hololink_channel_right,
            device=self._camera_right,
            coe_interface=self._coe_interface_right,
            pixel_width=self._camera_right._width,
            coe_channel=self._coe_channel_right,
        )

        isp_left = hololink_module.operators.ImageProcessorOp(
            self,
            name="isp_left",
            # Optical black value for imx274 is 50
            optical_black=50,
            bayer_format=self._camera_left.bayer_format().value,
            pixel_format=self._camera_left.pixel_format().value,
        )
        isp_right = hololink_module.operators.ImageProcessorOp(
            self,
            name="isp_right",
            # Optical black value for imx274 is 50
            optical_black=50,
            bayer_format=self._camera_right.bayer_format().value,
            pixel_format=self._camera_right.pixel_format().value,
        )

        rgba_components_per_pixel = 4
        bayer_pool_left = holoscan.resources.BlockMemoryPool(
            self,
            name="bayer_pool_left",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=self._camera_left._width
            * rgba_components_per_pixel
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera_left._height,
            num_blocks=2,
        )
        demosaic_left = holoscan.operators.BayerDemosaicOp(
            self,
            name="demosaic_left",
            pool=bayer_pool_left,
            generate_alpha=True,
            alpha_value=65535,
            bayer_grid_pos=self._camera_left.bayer_format().value,
            interpolation_mode=0,
            out_tensor_name="left",
        )
        bayer_pool_right = holoscan.resources.BlockMemoryPool(
            self,
            name="bayer_pool_right",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=self._camera_right._width
            * rgba_components_per_pixel
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera_right._height,
            num_blocks=2,
        )
        demosaic_right = holoscan.operators.BayerDemosaicOp(
            self,
            name="demosaic_right",
            pool=bayer_pool_right,
            generate_alpha=True,
            alpha_value=65535,
            bayer_grid_pos=self._camera_right.bayer_format().value,
            interpolation_mode=0,
            out_tensor_name="right",
        )

        #
        left_spec = holoscan.operators.HolovizOp.InputSpec(
            "left", holoscan.operators.HolovizOp.InputType.COLOR
        )
        left_spec_view = holoscan.operators.HolovizOp.InputSpec.View()
        left_spec_view.offset_x = 0
        left_spec_view.offset_y = 0
        left_spec_view.width = 0.5
        left_spec_view.height = 1
        left_spec.views = [left_spec_view]

        right_spec = holoscan.operators.HolovizOp.InputSpec(
            "right", holoscan.operators.HolovizOp.InputType.COLOR
        )
        right_spec_view = holoscan.operators.HolovizOp.InputSpec.View()
        right_spec_view.offset_x = 0.5
        right_spec_view.offset_y = 0
        right_spec_view.width = 0.5
        right_spec_view.height = 1
        right_spec.views = [right_spec_view]

        window_height = 200
        window_width = 600  # for the pair
        visualizer = holoscan.operators.HolovizOp(
            self,
            name="visualizer",
            headless=self._headless,
            tensors=[left_spec, right_spec],
            height=window_height,
            width=window_width,
            window_title="IMX274 pattern test",
            enable_camera_pose_output=True,
            camera_pose_output_type="extrinsics_model",
        )
        watchdog_operator = operators.WatchdogOp(
            self,
            name="watchdog_operator",
            watchdog=self._watchdog,
        )

        #
        self.add_flow(receiver_left, converter_left, {("output", "input")})
        self.add_flow(converter_left, isp_left, {("output", "input")})
        self.add_flow(isp_left, demosaic_left, {("output", "receiver")})
        self.add_flow(demosaic_left, visualizer, {("transmitter", "receivers")})

        self.add_flow(receiver_right, converter_right, {("output", "input")})
        self.add_flow(converter_right, isp_right, {("output", "input")})
        self.add_flow(isp_right, demosaic_right, {("output", "receiver")})
        self.add_flow(demosaic_right, visualizer, {("transmitter", "receivers")})
        self.add_flow(visualizer, watchdog_operator, {("camera_pose_output", "input")})


def run_dual_coe_test(
    frame_limit,
    headless,
    channel_metadata_left,
    camera_mode_left,
    pattern_left,
    channel_metadata_right,
    camera_mode_right,
    pattern_right,
    coe_interface_left,
    coe_interface_right,
):
    #
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
    hololink_channel_left = hololink_module.DataChannel(channel_metadata_left)
    hololink_channel_right = hololink_module.DataChannel(channel_metadata_right)
    # Get a handle to the camera
    camera_left = hololink_module.sensors.imx274.dual_imx274.Imx274Cam(
        hololink_channel_left, expander_configuration=0
    )
    camera_right = hololink_module.sensors.imx274.dual_imx274.Imx274Cam(
        hololink_channel_right, expander_configuration=1
    )
    # Note that ColorProfiler takes longer on the COLOR_PROFILER_START_FRAMEth frame, where it
    # starts running (and builds CUDA code).
    with utils.Watchdog(
        "frame-reception",
        initial_timeout=operators.color_profiler_initial_timeout(frame_limit),
    ) as watchdog:
        # Set up the application
        application = DualCoeTestApplication(
            frame_limit,
            headless,
            cu_context,
            cu_device_ordinal,
            hololink_channel_left,
            camera_left,
            camera_mode_left,
            hololink_channel_right,
            camera_right,
            camera_mode_right,
            coe_interface_left,
            coe_interface_right,
            watchdog,
        )
        default_configuration = os.path.join(
            os.path.dirname(__file__), "example_configuration.yaml"
        )
        application.config(default_configuration)
        # Run it.
        hololink = hololink_channel_left.hololink()
        assert hololink is hololink_channel_right.hololink()
        hololink.start()
        hololink.reset()
        camera_left.setup_clock()  # this also sets camera_right's clock
        camera_left.configure(camera_mode_left)
        camera_left.test_pattern(pattern_left)
        camera_right.configure(camera_mode_right)
        camera_right.test_pattern(pattern_right)
        #
        application.run()
        hololink.stop()

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS


# Test COE modes.
@pytest.mark.skip_unless_imx274
@pytest.mark.skip_unless_coe
@pytest.mark.parametrize(
    "camera_mode_left, pattern_left, camera_mode_right, pattern_right",  # noqa: E501
    [
        (  # left
            hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
            10,
            # right
            hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
            11,
        ),
    ],
)
def test_dual_imx274_pattern_coe(
    frame_limit,
    camera_mode_left,
    pattern_left,
    camera_mode_right,
    pattern_right,
    headless,
    channel_ips,
    coe_interfaces,
):
    # This only works if there are two interfaces listed in channel_ips.
    if len(channel_ips) < 2:
        pytest.skip(
            "Test is valid only when --channel-ips has more than one interface."
        )
    hololink_left = channel_ips[0]
    hololink_right = channel_ips[1]
    # This only works if there are two interfaces listed in coe_interfaces.
    if len(coe_interfaces) < 2:
        pytest.skip(
            "Test is valid only with --coe-interface is given with more than one interface."
        )

    # Get a handle to data sources
    channel_metadata_left = hololink_module.Enumerator.find_channel(
        channel_ip=hololink_left
    )
    channel_metadata_right = hololink_module.Enumerator.find_channel(
        channel_ip=hololink_right
    )
    # We're unaccelerated; don't test the results.
    run_dual_coe_test(
        frame_limit,
        headless,
        channel_metadata_left,
        camera_mode_left,
        pattern_left,
        channel_metadata_right,
        camera_mode_right,
        pattern_right,
        coe_interface_left=coe_interfaces[0],
        coe_interface_right=coe_interfaces[1],
    )


class CoeTestApplication(holoscan.core.Application):
    def __init__(
        self,
        frame_limit,
        headless,
        cuda_context,
        cuda_device_ordinal,
        hololink_channel,
        camera,
        camera_mode,
        watchdog,
        coe_interface,
    ):
        logging.info("__init__")
        super().__init__()
        self._frame_limit = frame_limit
        self._headless = headless
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._hololink_channel = hololink_channel
        self._hololink = hololink_channel.hololink()
        self._camera = camera
        self._camera_mode = camera_mode
        self._watchdog = watchdog
        self._coe_interface = coe_interface
        # Each camera sharing a network connection must use
        # a unique channel number from 0..63.
        self._coe_channel = 1

    def compose(self):
        logging.info("compose")
        self._condition = holoscan.conditions.CountCondition(
            self,
            name="condition",
            count=self._frame_limit,
        )

        self._camera.set_mode(self._camera_mode)

        converter_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="pool",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=self._camera._width
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera._height,
            num_blocks=4,
        )
        data_in = hololink_module.operators.PackedFormatConverterOp(
            self,
            name="data_in",
            allocator=converter_pool,
            cuda_device_ordinal=self._cuda_device_ordinal,
            hololink_channel=self._hololink_channel,
        )
        self._camera.configure_converter(data_in)

        receiver = hololink_module.operators.LinuxCoeReceiverOp(
            self,
            self._condition,
            name="receiver",
            frame_size=data_in.get_frame_size(),
            frame_context=self._cuda_context,
            hololink_channel=self._hololink_channel,
            device=self._camera,
            coe_interface=self._coe_interface,
            pixel_width=self._camera._width,
            coe_channel=self._coe_channel,
        )

        isp = hololink_module.operators.ImageProcessorOp(
            self,
            name="isp",
            # Optical black value for imx274 is 50
            optical_black=50,
            bayer_format=self._camera.bayer_format().value,
            pixel_format=self._camera.pixel_format().value,
        )

        rgba_components_per_pixel = 4
        bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="bayer_pool",
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
            bayer_grid_pos=self._camera.bayer_format().value,
            interpolation_mode=0,
        )

        visualizer = holoscan.operators.HolovizOp(
            self,
            name="visualizer",
            headless=self._headless,
            window_title="Visualizer",
            enable_camera_pose_output=True,
            camera_pose_output_type="extrinsics_model",
        )
        watchdog_operator = operators.WatchdogOp(
            self,
            name="watchdog_operator",
            watchdog=self._watchdog,
        )

        #
        self.add_flow(receiver, data_in, {("output", "input")})
        self.add_flow(data_in, isp, {("output", "input")})
        self.add_flow(isp, demosaic, {("output", "receiver")})
        self.add_flow(demosaic, visualizer, {("transmitter", "receivers")})
        self.add_flow(visualizer, watchdog_operator, {("camera_pose_output", "input")})


@pytest.mark.skip_unless_imx274
@pytest.mark.skip_unless_coe
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
    "hololink",
    [
        "192.168.0.2",
    ],
)
def test_imx274_pattern_coe(
    camera_mode,
    pattern,
    headless,
    hololink,
    coe_interfaces,
    frame_limit,
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
    # Note that ColorProfiler takes longer on the COLOR_PROFILER_START_FRAMEth frame, where it
    # starts running (and builds CUDA code).
    with utils.Watchdog(
        "frame-reception",
        initial_timeout=operators.color_profiler_initial_timeout(frame_limit),
    ) as watchdog:
        # Set up the application
        application = CoeTestApplication(
            frame_limit,
            headless,
            cu_context,
            cu_device_ordinal,
            hololink_channel,
            camera,
            camera_mode,
            watchdog,
            coe_interface=coe_interfaces[0],
        )
        default_configuration = os.path.join(
            os.path.dirname(__file__), "example_configuration.yaml"
        )
        application.config(default_configuration)
        # Run it.
        hololink = hololink_channel.hololink()
        hololink.start()
        hololink.reset()
        camera.setup_clock()
        camera.configure(camera_mode)
        camera.test_pattern(pattern)
        #
        application.run()
        hololink.stop()

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
