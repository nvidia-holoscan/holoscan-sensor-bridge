# SPDX-FileCopyrightText: Copyright (c) NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import time

import holoscan
import operators
import pytest
from cuda import cuda

import hololink as hololink_module


class StereoApplication(holoscan.core.Application):
    def __init__(
        self,
        headless,
        cuda_context,
        cuda_device_ordinal,
        hololink_channel_left,
        camera_left,
        hololink_channel_right,
        camera_right,
        camera_mode,
        frame_limit,
        window_height,
        window_width,
        window_title,
        left_receiver_factory,
        right_receiver_factory,
        left_isp_factory,
        right_isp_factory,
        ptp_enable,
    ):
        logging.info("__init__")
        super().__init__()
        self._headless = headless
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._hololink_channel_left = hololink_channel_left
        self._camera_left = camera_left
        self._hololink_channel_right = hololink_channel_right
        self._camera_right = camera_right
        self._camera_mode = camera_mode
        self._frame_limit = frame_limit
        self._window_height = window_height
        self._window_width = window_width
        self._window_title = window_title
        self._left_receiver_factory = left_receiver_factory
        self._right_receiver_factory = right_receiver_factory
        self._left_isp_factory = left_isp_factory
        self._right_isp_factory = right_isp_factory
        self._ptp_enable = ptp_enable
        # This is a control for HSDK
        self.is_metadata_enabled = True

    def compose(self):
        logging.info("compose")
        self._count_left = holoscan.conditions.CountCondition(
            self,
            name="count_left",
            count=self._frame_limit,
        )
        self._count_right = holoscan.conditions.CountCondition(
            self,
            name="count_right",
            count=self._frame_limit,
        )
        self._camera_left.set_mode(self._camera_mode)
        self._camera_right.set_mode(self._camera_mode)

        csi_to_bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="pool",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=self._camera_left._width
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera_left._height,
            num_blocks=6,
        )
        csi_to_bayer_operator_left = hololink_module.operators.CsiToBayerOp(
            self,
            name="csi_to_bayer_left",
            allocator=csi_to_bayer_pool,
            cuda_device_ordinal=self._cuda_device_ordinal,
        )
        self._camera_left.configure_converter(csi_to_bayer_operator_left)
        csi_to_bayer_operator_right = hololink_module.operators.CsiToBayerOp(
            self,
            name="csi_to_bayer_right",
            allocator=csi_to_bayer_pool,
            cuda_device_ordinal=self._cuda_device_ordinal,
        )
        self._camera_right.configure_converter(csi_to_bayer_operator_right)

        receiver_operator_left = self._left_receiver_factory(
            self,
            self._cuda_context,
            self._count_left,
            name="receiver_left",
            frame_size=csi_to_bayer_operator_left.get_csi_length(),
            hololink_channel=self._hololink_channel_left,
            device=self._camera_left,
            rename_metadata=lambda name: f"left_{name}",
        )
        receiver_operator_right = self._right_receiver_factory(
            self,
            self._cuda_context,
            self._count_right,
            name="receiver_right",
            frame_size=csi_to_bayer_operator_right.get_csi_length(),
            hololink_channel=self._hololink_channel_right,
            device=self._camera_right,
            rename_metadata=lambda name: f"right_{name}",
        )

        isp_left_in, isp_left_in_name, isp_left_out, isp_left_out_name = (
            self._left_isp_factory(
                self,
                name="isp_left",
                camera_index=0,
                optical_black=50,
                bayer_format=self._camera_left.bayer_format().value,
                pixel_format=self._camera_left.pixel_format().value,
                pixel_width=self._camera_left._width,
                pixel_height=self._camera_left._height,
                out_tensor_name="left",
            )
        )
        isp_right_in, isp_right_in_name, isp_right_out, isp_right_out_name = (
            self._right_isp_factory(
                self,
                name="isp_right",
                camera_index=1,
                optical_black=50,
                bayer_format=self._camera_right.bayer_format().value,
                pixel_format=self._camera_right.pixel_format().value,
                pixel_width=self._camera_right._width,
                pixel_height=self._camera_right._height,
                out_tensor_name="right",
            )
        )

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

        visualizer = holoscan.operators.HolovizOp(
            self,
            name="holoviz",
            headless=self._headless,
            framebuffer_srgb=True,
            tensors=[left_spec, right_spec],
            height=self._window_height,
            width=self._window_width,
            window_title=self._window_title,
            enable_camera_pose_output=True,
            camera_pose_output_type="extrinsics_model",
        )

        monitor = operators.MonitorOperator(
            self,
            name="monitor",
            callback=(
                self._monitor_callback
                if self._ptp_enable
                else self._ignore_monitor_callback
            ),
        )
        #
        self.add_flow(
            receiver_operator_left, csi_to_bayer_operator_left, {("output", "input")}
        )
        self.add_flow(
            receiver_operator_right, csi_to_bayer_operator_right, {("output", "input")}
        )
        self.add_flow(
            csi_to_bayer_operator_left, isp_left_in, {("output", isp_left_in_name)}
        )
        self.add_flow(
            csi_to_bayer_operator_right, isp_right_in, {("output", isp_right_in_name)}
        )
        self.add_flow(isp_left_out, visualizer, {(isp_left_out_name, "receivers")})
        self.add_flow(isp_right_out, visualizer, {(isp_right_out_name, "receivers")})
        #
        self.add_flow(visualizer, monitor, {("camera_pose_output", "input")})

    def _ignore_monitor_callback(self, operator, metadata):
        pass

    def _monitor_callback(self, operator, metadata):
        left_times = operator.get_times(
            metadata, rename_metadata=lambda name: f"left_{name}"
        )
        right_times = operator.get_times(
            metadata, rename_metadata=lambda name: f"right_{name}"
        )
        left_frame_number = left_times["left_frame_number"]
        left_frame_time = left_times["left_frame_end"] - left_times["left_frame_start"]
        left_overall_time = left_times["complete"] - left_times["left_frame_start"]
        left_processing_time = left_times["complete"] - left_times["left_frame_end"]
        right_frame_number = right_times["right_frame_number"]
        right_frame_time = (
            right_times["right_frame_end"] - right_times["right_frame_start"]
        )
        right_overall_time = right_times["complete"] - right_times["right_frame_start"]
        right_processing_time = right_times["complete"] - right_times["right_frame_end"]
        logging.info(
            f"{left_frame_number=} {left_frame_time=:.4f} {left_overall_time=:.4f} {left_processing_time=:.4f}"
        )
        logging.info(
            f"{right_frame_number=} {right_frame_time=:.4f} {right_overall_time=:.4f} {right_processing_time=:.4f}"
        )


def imx274_camera_factory(hololink_channel, instance, camera_mode, test_pattern):
    camera = hololink_module.sensors.imx274.dual_imx274.Imx274Cam(
        hololink_channel, expander_configuration=instance
    )
    if instance == 0:
        camera.setup_clock()  # also configures other cameras' clocks on this specific board
    camera.configure(camera_mode)
    camera.test_pattern(test_pattern)
    camera.set_digital_gain_reg(0x4)
    return camera


def vb1940_camera_factory(hololink_channel, instance, camera_mode):
    camera = hololink_module.sensors.vb1940.vb1940.Vb1940Cam(
        hololink_channel, expander_configuration=instance
    )
    if instance == 0:
        camera.setup_clock()  # also configures other cameras' clocks on this specific board
    hololink = hololink_channel.hololink()
    hololink.write_uint32(0x8, 0x1)  # Release the sensor RESET to high
    time.sleep(100 / 1000)
    camera.get_register_32(0x0000)  # DEVICE_MODEL_ID:"S940"(ASCII code:0x53393430)
    camera.get_register_32(0x0734)  # EXT_CLOCK(25MHz = 0x017d7840)
    camera.configure(camera_mode)
    return camera


def run_stereo_test(
    camera_mode,
    headless,
    frame_limit,
    hololink_address,
    left_receiver_factory,
    right_receiver_factory,
    left_isp_factory,
    right_isp_factory,
    left_camera_factory,
    right_camera_factory,
    ptp_enable,
):
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
    # Now make separate ones for left and right; and set them to
    # use sensor 0 and 1 respectively.
    channel_metadata_left = hololink_module.Metadata(channel_metadata)
    hololink_module.DataChannel.use_sensor(channel_metadata_left, 0)
    channel_metadata_right = hololink_module.Metadata(channel_metadata)
    hololink_module.DataChannel.use_sensor(channel_metadata_right, 1)
    #
    hololink_channel_left = hololink_module.DataChannel(channel_metadata_left)
    hololink_channel_right = hololink_module.DataChannel(channel_metadata_right)
    #
    hololink = hololink_channel_left.hololink()
    assert hololink == hololink_channel_right.hololink()
    hololink.start()
    hololink.reset()
    #
    if ptp_enable:
        ptp_sync_timeout_s = 10
        ptp_sync_timeout = hololink_module.Timeout(ptp_sync_timeout_s)
        logging.debug("Waiting for PTP sync.")
        if not hololink.ptp_synchronize(ptp_sync_timeout):
            assert (
                False
            ), f"Failed to synchronize PTP after {ptp_sync_timeout_s} seconds."
    # Get a handle to the camera
    camera_left = left_camera_factory(hololink_channel_left, 0)
    camera_right = right_camera_factory(hololink_channel_left, 1)
    # Set up the application
    window_height = 200
    window_width = 600  # for the pair
    window_title = "Pattern test"
    application = StereoApplication(
        headless,
        cu_context,
        cu_device_ordinal,
        hololink_channel_left,
        camera_left,
        hololink_channel_right,
        camera_right,
        camera_mode,
        frame_limit,
        window_height,
        window_width,
        window_title,
        left_receiver_factory=left_receiver_factory,
        right_receiver_factory=right_receiver_factory,
        left_isp_factory=left_isp_factory,
        right_isp_factory=right_isp_factory,
        ptp_enable=ptp_enable,
    )

    default_configuration = os.path.join(
        os.path.dirname(__file__), "example_configuration.yaml"
    )
    application.config(default_configuration)
    # Run it.
    application.run()
    hololink.stop()

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS


def naive_isp(
    app,
    name,
    camera_index,
    optical_black,
    bayer_format,
    pixel_format,
    pixel_width,
    pixel_height,
    out_tensor_name,
):
    isp = hololink_module.operators.ImageProcessorOp(
        app,
        name=name,
        optical_black=optical_black,
        bayer_format=bayer_format,
        pixel_format=pixel_format,
    )
    rgba_components_per_pixel = 4
    bayer_pool = holoscan.resources.BlockMemoryPool(
        app,
        name=f"{name}_bayer_pool",
        # storage_type of 1 is device memory
        storage_type=1,
        block_size=pixel_width
        * rgba_components_per_pixel
        * ctypes.sizeof(ctypes.c_uint16)
        * pixel_height,
        num_blocks=6,
    )
    demosaic = holoscan.operators.BayerDemosaicOp(
        app,
        name=f"{name}_demosaic",
        pool=bayer_pool,
        generate_alpha=True,
        alpha_value=65535,
        bayer_grid_pos=bayer_format,
        interpolation_mode=0,
        out_tensor_name=out_tensor_name,
    )
    app.add_flow(isp, demosaic, {("output", "receiver")})
    return isp, "input", demosaic, "transmitter"


def argus_isp(
    app,
    name,
    camera_index,
    optical_black,
    bayer_format,
    pixel_format,
    pixel_width,
    pixel_height,
    out_tensor_name,
):
    rgba_components_per_pixel = 3
    pool = holoscan.resources.BlockMemoryPool(
        app,
        name=f"{name}pool",
        # storage_type of 1 is device memory
        storage_type=1,
        block_size=pixel_width
        * rgba_components_per_pixel
        * ctypes.sizeof(ctypes.c_uint16)
        * pixel_height,
        num_blocks=2,
    )
    # 60fps is 16.67ms
    exposure_time_ms = 16.67
    analog_gain = 10.0
    pixel_bit_depth = 10
    isp = hololink_module.operators.ArgusIspOp(
        app,
        name=name,
        bayer_format=bayer_format,
        exposure_time_ms=exposure_time_ms,
        analog_gain=analog_gain,
        pixel_bit_depth=pixel_bit_depth,
        pool=pool,
        out_tensor_name=out_tensor_name,
        camera_index=camera_index,
    )
    return isp, "input", isp, "output"


def linux_receiver_factory(
    app,
    cu_context,
    condition,
    name,
    frame_size,
    hololink_channel,
    device,
    rename_metadata=lambda name: name,
):
    r = hololink_module.operators.LinuxReceiverOperator(
        app,
        condition,
        name=name,
        frame_size=frame_size,
        frame_context=cu_context,
        hololink_channel=hololink_channel,
        device=device,
        rename_metadata=rename_metadata,
    )
    return r


def linux_coe_receiver_factory(
    coe_interface,
    coe_channel,
    pixel_width,
    app,
    cu_context,
    condition,
    name,
    frame_size,
    hololink_channel,
    device,
    rename_metadata=lambda name: name,
):
    r = hololink_module.operators.LinuxCoeReceiverOp(
        app,
        condition,
        name=name,
        frame_size=frame_size,
        frame_context=cu_context,
        hololink_channel=hololink_channel,
        device=device,
        coe_interface=coe_interface,
        pixel_width=pixel_width,
        coe_channel=coe_channel,
        rename_metadata=rename_metadata,
    )
    return r


@pytest.mark.skip_unless_imx274
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
    ],
)
def test_imx274_mono_linux_naive(
    camera_mode, headless, frame_limit, hololink_address, ptp_enable
):
    run_mono_test(
        camera_mode,
        headless,
        frame_limit,
        hololink_address,
        receiver_factory=linux_receiver_factory,
        isp_factory=naive_isp,
        camera_factory=lambda channel, instance: imx274_camera_factory(
            channel, instance, camera_mode, 10
        ),
        ptp_enable=ptp_enable,
    )


@pytest.mark.skip_unless_imx274
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
    ],
)
def test_imx274_stereo_linux_naive(
    camera_mode,
    headless,
    frame_limit,
    hololink_address,
    ptp_enable,
):
    run_stereo_test(
        camera_mode,
        headless,
        frame_limit,
        hololink_address,
        left_receiver_factory=linux_receiver_factory,
        right_receiver_factory=linux_receiver_factory,
        left_isp_factory=naive_isp,
        right_isp_factory=naive_isp,
        left_camera_factory=lambda channel, instance: imx274_camera_factory(
            channel, instance, camera_mode, 10
        ),
        right_camera_factory=lambda channel, instance: imx274_camera_factory(
            channel, instance, camera_mode, 11
        ),
        ptp_enable=ptp_enable,
    )


@pytest.mark.skip_unless_coe
@pytest.mark.skip_unless_imx274
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
    ],
)
def test_imx274_stereo_linux_naive_coe(
    camera_mode,
    headless,
    frame_limit,
    hololink_address,
    ptp_enable,
    coe_interfaces,
):
    pixel_width = 1920
    run_stereo_test(
        camera_mode,
        headless,
        frame_limit,
        hololink_address,
        left_receiver_factory=lambda *args, **kwargs: linux_coe_receiver_factory(
            coe_interfaces[0], 1, pixel_width, *args, **kwargs
        ),
        right_receiver_factory=lambda *args, **kwargs: linux_coe_receiver_factory(
            coe_interfaces[0], 2, pixel_width, *args, **kwargs
        ),
        left_isp_factory=naive_isp,
        right_isp_factory=naive_isp,
        left_camera_factory=lambda channel, instance: imx274_camera_factory(
            channel, instance, camera_mode, 10
        ),
        right_camera_factory=lambda channel, instance: imx274_camera_factory(
            channel, instance, camera_mode, 11
        ),
        ptp_enable=ptp_enable,
    )


# NOTE THAT ARGUS ONLY WORKS FOR ONE CHANNEL.
@pytest.mark.skip_unless_igpu
@pytest.mark.skip_unless_imx274
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
    ],
)
def test_imx274_stereo_linux_argus(
    camera_mode,
    headless,
    frame_limit,
    hololink_address,
    ptp_enable,
):
    run_stereo_test(
        camera_mode,
        headless,
        frame_limit,
        hololink_address,
        left_receiver_factory=linux_receiver_factory,
        right_receiver_factory=linux_receiver_factory,
        left_isp_factory=argus_isp,
        right_isp_factory=argus_isp,
        left_camera_factory=lambda channel, instance: imx274_camera_factory(
            channel, instance, camera_mode, 10
        ),
        right_camera_factory=lambda channel, instance: imx274_camera_factory(
            channel, instance, camera_mode, 11
        ),
        ptp_enable=ptp_enable,
    )


# NOTE THAT ARGUS ONLY WORKS FOR ONE CHANNEL.
@pytest.mark.skip_unless_igpu
@pytest.mark.skip_unless_imx274
@pytest.mark.skip_unless_coe
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
    ],
)
def test_imx274_stereo_linux_argus_coe(
    camera_mode,
    headless,
    frame_limit,
    hololink_address,
    ptp_enable,
    coe_interfaces,
):
    pixel_width = 1920
    run_stereo_test(
        camera_mode,
        headless,
        frame_limit,
        hololink_address,
        left_receiver_factory=lambda *args, **kwargs: linux_coe_receiver_factory(
            coe_interfaces[0], 1, pixel_width, *args, **kwargs
        ),
        right_receiver_factory=lambda *args, **kwargs: linux_coe_receiver_factory(
            coe_interfaces[0], 2, pixel_width, *args, **kwargs
        ),
        left_isp_factory=argus_isp,
        right_isp_factory=argus_isp,
        left_camera_factory=lambda channel, instance: imx274_camera_factory(
            channel, instance, camera_mode, 10
        ),
        right_camera_factory=lambda channel, instance: imx274_camera_factory(
            channel, instance, camera_mode, 11
        ),
        ptp_enable=ptp_enable,
    )


class MonoApplication(holoscan.core.Application):
    def __init__(
        self,
        headless,
        cuda_context,
        cuda_device_ordinal,
        hololink_channel,
        camera,
        camera_mode,
        frame_limit,
        window_height,
        window_width,
        window_title,
        receiver_factory,
        isp_factory,
        ptp_enable,
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
        self._window_height = window_height
        self._window_width = window_width
        self._window_title = window_title
        self._receiver_factory = receiver_factory
        self._isp_factory = isp_factory
        self._ptp_enable = ptp_enable
        # This is a control for HSDK
        self.is_metadata_enabled = True

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
            num_blocks=6,
        )
        csi_to_bayer_operator = hololink_module.operators.CsiToBayerOp(
            self,
            name="csi_to_bayer",
            allocator=csi_to_bayer_pool,
            cuda_device_ordinal=self._cuda_device_ordinal,
        )
        self._camera.configure_converter(csi_to_bayer_operator)

        receiver_operator = self._receiver_factory(
            self,
            self._cuda_context,
            self._count,
            name="receiver",
            frame_size=csi_to_bayer_operator.get_csi_length(),
            hololink_channel=self._hololink_channel,
            device=self._camera,
        )
        isp_in, isp_in_name, isp_out, isp_out_name = self._isp_factory(
            self,
            name="isp",
            camera_index=0,
            optical_black=50,
            bayer_format=self._camera.bayer_format().value,
            pixel_format=self._camera.pixel_format().value,
            pixel_width=self._camera._width,
            pixel_height=self._camera._height,
            out_tensor_name="output",
        )

        visualizer = holoscan.operators.HolovizOp(
            self,
            name="holoviz",
            headless=self._headless,
            framebuffer_srgb=True,
            enable_camera_pose_output=True,
            camera_pose_output_type="extrinsics_model",
        )

        monitor = operators.MonitorOperator(
            self,
            name="monitor",
            callback=(
                self._monitor_callback
                if self._ptp_enable
                else self._ignore_monitor_callback
            ),
        )

        #
        self.add_flow(receiver_operator, csi_to_bayer_operator, {("output", "input")})
        self.add_flow(csi_to_bayer_operator, isp_in, {("output", isp_in_name)})
        self.add_flow(isp_out, visualizer, {(isp_out_name, "receivers")})
        self.add_flow(visualizer, monitor, {("camera_pose_output", "input")})

    def _ignore_monitor_callback(self, operator, metadata):
        pass

    def _monitor_callback(self, operator, metadata):
        times = operator.get_times(metadata)
        frame_number = times["frame_number"]
        frame_time = times["frame_end"] - times["frame_start"]
        overall_time = times["complete"] - times["frame_start"]
        processing_time = times["complete"] - times["frame_end"]
        logging.info(
            f"{frame_number=} {frame_time=:.4f} {overall_time=:.4f} {processing_time=:.4f}"
        )


def run_mono_test(
    camera_mode,
    headless,
    frame_limit,
    hololink_address,
    receiver_factory,
    isp_factory,
    camera_factory,
    ptp_enable,
):
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
    #
    hololink_channel = hololink_module.DataChannel(channel_metadata)
    # Set up hololink
    hololink = hololink_channel.hololink()
    hololink.start()
    hololink.reset()
    #
    if ptp_enable:
        ptp_sync_timeout_s = 10
        ptp_sync_timeout = hololink_module.Timeout(ptp_sync_timeout_s)
        logging.debug("Waiting for PTP sync.")
        if not hololink.ptp_synchronize(ptp_sync_timeout):
            assert (
                False
            ), f"Failed to synchronize PTP after {ptp_sync_timeout_s} seconds."
    # Get a handle to the camera
    camera = camera_factory(hololink_channel, 0)
    # Set up the application
    window_height = 200
    window_width = 600  # for the pair
    window_title = "Pattern test"
    application = MonoApplication(
        headless,
        cu_context,
        cu_device_ordinal,
        hololink_channel,
        camera,
        camera_mode,
        frame_limit,
        window_height,
        window_width,
        window_title,
        receiver_factory=receiver_factory,
        isp_factory=isp_factory,
        ptp_enable=ptp_enable,
    )

    default_configuration = os.path.join(
        os.path.dirname(__file__), "example_configuration.yaml"
    )
    application.config(default_configuration)
    # Run it.
    application.run()
    hololink.stop()

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS


@pytest.mark.skip_unless_igpu
@pytest.mark.skip_unless_imx274
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
    ],
)
def test_imx274_mono_linux_argus(
    camera_mode, headless, frame_limit, hololink_address, ptp_enable
):
    run_mono_test(
        camera_mode,
        headless,
        frame_limit,
        hololink_address,
        receiver_factory=linux_receiver_factory,
        isp_factory=argus_isp,
        camera_factory=lambda channel, instance: imx274_camera_factory(
            channel, instance, camera_mode, 10
        ),
        ptp_enable=ptp_enable,
    )


@pytest.mark.skip_unless_igpu
@pytest.mark.skip_unless_imx274
@pytest.mark.skip_unless_coe
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
    ],
)
def test_imx274_mono_linux_argus_coe(
    camera_mode,
    headless,
    frame_limit,
    hololink_address,
    ptp_enable,
    coe_interfaces,
):
    pixel_width = 1920
    run_mono_test(
        camera_mode,
        headless,
        frame_limit,
        hololink_address,
        receiver_factory=lambda *args, **kwargs: linux_coe_receiver_factory(
            coe_interfaces[0], 1, pixel_width, *args, **kwargs
        ),
        isp_factory=argus_isp,
        camera_factory=lambda channel, instance: imx274_camera_factory(
            channel, instance, camera_mode, 10
        ),
        ptp_enable=ptp_enable,
    )


@pytest.mark.skip_unless_vb1940
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS,
    ],
)
def test_vb1940_mono_linux_naive(
    camera_mode, headless, frame_limit, hololink_address, ptp_enable
):
    run_mono_test(
        camera_mode,
        headless,
        frame_limit,
        hololink_address,
        receiver_factory=linux_receiver_factory,
        isp_factory=naive_isp,
        camera_factory=lambda channel, instance: vb1940_camera_factory(
            channel, instance, camera_mode
        ),
        ptp_enable=ptp_enable,
    )


@pytest.mark.skip_unless_igpu
@pytest.mark.skip_unless_vb1940
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS,
    ],
)
def test_vb1940_mono_linux_argus(
    camera_mode, headless, frame_limit, hololink_address, ptp_enable
):
    run_mono_test(
        camera_mode,
        headless,
        frame_limit,
        hololink_address,
        receiver_factory=linux_receiver_factory,
        isp_factory=argus_isp,
        camera_factory=lambda channel, instance: vb1940_camera_factory(
            channel, instance, camera_mode
        ),
        ptp_enable=ptp_enable,
    )


@pytest.mark.skip_unless_coe
@pytest.mark.skip_unless_vb1940
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS,
    ],
)
def test_vb1940_mono_linux_naive_coe(
    camera_mode,
    headless,
    frame_limit,
    hololink_address,
    ptp_enable,
    coe_interfaces,
):
    pixel_width = 2560
    run_mono_test(
        camera_mode,
        headless,
        frame_limit,
        hololink_address,
        receiver_factory=lambda *args, **kwargs: linux_coe_receiver_factory(
            coe_interfaces[0], 1, pixel_width, *args, **kwargs
        ),
        isp_factory=naive_isp,
        camera_factory=lambda channel, instance: vb1940_camera_factory(
            channel, instance, camera_mode
        ),
        ptp_enable=ptp_enable,
    )


@pytest.mark.skip_unless_vb1940
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS,
    ],
)
def test_vb1940_stereo_linux_naive(
    camera_mode,
    headless,
    frame_limit,
    hololink_address,
    ptp_enable,
):
    run_stereo_test(
        camera_mode,
        headless,
        frame_limit,
        hololink_address,
        left_receiver_factory=linux_receiver_factory,
        right_receiver_factory=linux_receiver_factory,
        left_isp_factory=naive_isp,
        right_isp_factory=naive_isp,
        left_camera_factory=lambda channel, instance: vb1940_camera_factory(
            channel, instance, camera_mode
        ),
        right_camera_factory=lambda channel, instance: vb1940_camera_factory(
            channel, instance, camera_mode
        ),
        ptp_enable=ptp_enable,
    )


@pytest.mark.skip_unless_coe
@pytest.mark.skip_unless_vb1940
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS,
    ],
)
def test_vb1940_stereo_linux_naive_coe(
    camera_mode,
    headless,
    frame_limit,
    hololink_address,
    ptp_enable,
    coe_interfaces,
):
    pixel_width = 2560
    run_stereo_test(
        camera_mode,
        headless,
        frame_limit,
        hololink_address,
        left_receiver_factory=lambda *args, **kwargs: linux_coe_receiver_factory(
            coe_interfaces[0], 1, pixel_width, *args, **kwargs
        ),
        right_receiver_factory=lambda *args, **kwargs: linux_coe_receiver_factory(
            coe_interfaces[0], 2, pixel_width, *args, **kwargs
        ),
        left_isp_factory=naive_isp,
        right_isp_factory=naive_isp,
        left_camera_factory=lambda channel, instance: vb1940_camera_factory(
            channel, instance, camera_mode
        ),
        right_camera_factory=lambda channel, instance: vb1940_camera_factory(
            channel, instance, camera_mode
        ),
        ptp_enable=ptp_enable,
    )


# NOTE THAT ARGUS ONLY WORKS FOR ONE CHANNEL.
@pytest.mark.skip_unless_coe
@pytest.mark.skip_unless_igpu
@pytest.mark.skip_unless_vb1940
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS,
    ],
)
def test_vb1940_stereo_linux_argus_coe(
    camera_mode,
    headless,
    frame_limit,
    hololink_address,
    ptp_enable,
    coe_interfaces,
):
    pixel_width = 2560
    run_stereo_test(
        camera_mode,
        headless,
        frame_limit,
        hololink_address,
        left_receiver_factory=lambda *args, **kwargs: linux_coe_receiver_factory(
            coe_interfaces[0], 1, pixel_width, *args, **kwargs
        ),
        right_receiver_factory=lambda *args, **kwargs: linux_coe_receiver_factory(
            coe_interfaces[0], 2, pixel_width, *args, **kwargs
        ),
        left_isp_factory=argus_isp,
        right_isp_factory=argus_isp,
        left_camera_factory=lambda channel, instance: vb1940_camera_factory(
            channel, instance, camera_mode
        ),
        right_camera_factory=lambda channel, instance: vb1940_camera_factory(
            channel, instance, camera_mode
        ),
        ptp_enable=ptp_enable,
    )
