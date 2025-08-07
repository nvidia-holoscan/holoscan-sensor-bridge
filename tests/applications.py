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
import time

import holoscan
import operators
import utils
from cuda import cuda

import hololink as hololink_module


class CudaContext:
    """Context manager for CUDA initialization and cleanup."""

    def __init__(self, device_ordinal=0):
        self.device_ordinal = device_ordinal
        self.cu_device = None
        self.cu_context = None

    def __enter__(self):
        # Initialize CUDA
        (cu_result,) = cuda.cuInit(0)
        assert cu_result == cuda.CUresult.CUDA_SUCCESS

        # Get device
        cu_result, self.cu_device = cuda.cuDeviceGet(self.device_ordinal)
        assert cu_result == cuda.CUresult.CUDA_SUCCESS

        # Get context
        cu_result, self.cu_context = cuda.cuDevicePrimaryCtxRetain(self.cu_device)
        assert cu_result == cuda.CUresult.CUDA_SUCCESS

        return self.cu_context, self.device_ordinal

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Release context
        if self.cu_device is not None:
            (cu_result,) = cuda.cuDevicePrimaryCtxRelease(self.cu_device)
            assert cu_result == cuda.CUresult.CUDA_SUCCESS


class StereoApplication(holoscan.core.Application):
    def __init__(
        self,
        stereo_test,
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
        watchdog,
    ):
        logging.info("__init__")
        super().__init__()
        self._stereo_test = stereo_test
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
        self._watchdog = watchdog
        #
        self.enable_metadata(True)
        # always use the most recently added frame.
        self.metadata_policy = holoscan.core.MetadataPolicy.INPLACE_UPDATE

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

        check_left = self._stereo_test.check_left(self)
        check_right = self._stereo_test.check_right(self)
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

        check_metadata = self._stereo_test.check_metadata(self)

        watchdog = operators.WatchdogOp(
            self,
            name="watchdog",
            watchdog=self._watchdog,
        )
        #
        self.add_flow(receiver_operator_left, check_left, {("output", "input")})
        self.add_flow(receiver_operator_right, check_right, {("output", "input")})
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
        self.add_flow(visualizer, check_metadata, {("camera_pose_output", "input")})
        self.add_flow(visualizer, watchdog, {("camera_pose_output", "input")})


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


def vb1940_camera_factory(
    hololink_channel,
    instance,
    camera_mode,
    vsync=hololink_module.Synchronizer.null_synchronizer(),
):
    camera = hololink_module.sensors.vb1940.Vb1940Cam(
        hololink_channel,
        vsync=vsync,
    )
    if instance == 0:
        camera.setup_clock()  # also configures other cameras' clocks on this specific board
    hololink = hololink_channel.hololink()
    hololink.write_uint32(0x8, 0x3)  # Release the sensor RESET to high
    time.sleep(100 / 1000)
    camera.get_register_32(0x0000)  # DEVICE_MODEL_ID:"S940"(ASCII code:0x53393430)
    camera.get_register_32(0x0734)  # EXT_CLOCK(25MHz = 0x017d7840)
    camera.configure(camera_mode)
    return camera


class StereoTest:
    def __init__(
        self,
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
        self._camera_mode = camera_mode
        self._headless = headless
        self._frame_limit = frame_limit
        self._hololink_address = hololink_address
        self._left_receiver_factory = left_receiver_factory
        self._right_receiver_factory = right_receiver_factory
        self._left_isp_factory = left_isp_factory
        self._right_isp_factory = right_isp_factory
        self._left_camera_factory = left_camera_factory
        self._right_camera_factory = right_camera_factory
        self._ptp_enable = ptp_enable

    def execute(self):
        # Get a handle to the GPU using context manager
        with CudaContext() as (cu_context, cu_device_ordinal):
            # Get a handle to data sources
            channel_metadata = hololink_module.Enumerator.find_channel(
                channel_ip=self._hololink_address
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
            try:
                hololink.reset()
                #
                if self._ptp_enable:
                    logging.debug("Waiting for PTP sync.")
                    if not hololink.ptp_synchronize():
                        raise ValueError("Failed to synchronize PTP.")
                    logging.info("PTP synchronized.")
                # Get a handle to the camera
                camera_left = self._left_camera_factory(hololink_channel_left, 0)
                camera_right = self._right_camera_factory(hololink_channel_right, 1)
                # Set up the application
                window_height = 200
                window_width = 600  # for the pair
                window_title = "Pattern test"
                with utils.Watchdog(
                    "watchdog",
                    initial_timeout=operators.color_profiler_initial_timeout(
                        self._frame_limit
                    ),
                ) as watchdog:
                    self._application = StereoApplication(
                        self,
                        self._headless,
                        cu_context,
                        cu_device_ordinal,
                        hololink_channel_left,
                        camera_left,
                        hololink_channel_right,
                        camera_right,
                        self._camera_mode,
                        self._frame_limit,
                        window_height,
                        window_width,
                        window_title,
                        left_receiver_factory=self._left_receiver_factory,
                        right_receiver_factory=self._right_receiver_factory,
                        left_isp_factory=self._left_isp_factory,
                        right_isp_factory=self._right_isp_factory,
                        watchdog=watchdog,
                    )

                    default_configuration = os.path.join(
                        os.path.dirname(__file__), "example_configuration.yaml"
                    )
                    self._application.config(default_configuration)
                    #
                    # Run it.
                    self._application.run()
            finally:
                hololink.stop()

    def check_left(self, application, out_tensor_name="left"):
        """Override to catch the pipeline just after network receiver."""
        return operators.PassThroughOperator(
            application,
            name="check_left",
            out_tensor_name=out_tensor_name,
        )

    def check_right(self, application, out_tensor_name="right"):
        """Override to catch the pipeline just after network receiver."""
        return operators.PassThroughOperator(
            application,
            name="check_right",
            out_tensor_name=out_tensor_name,
        )

    def check_metadata(self, application):
        monitor = operators.MonitorOperator(
            application,
            name="monitor",
            callback=self.monitor_callback,
        )
        return monitor

    def monitor_callback(self, operator, metadata):
        if not self._ptp_enable:
            return
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


def roce_receiver_factory(
    ibv_name,
    ibv_port,
    app,
    cu_context,
    condition,
    name,
    frame_size,
    hololink_channel,
    device,
    rename_metadata=lambda name: name,
):
    r = hololink_module.operators.RoceReceiverOp(
        app,
        condition,
        name=name,
        frame_size=frame_size,
        frame_context=cu_context,
        ibv_name=ibv_name,
        ibv_port=ibv_port,
        hololink_channel=hololink_channel,
        device=device,
        rename_metadata=rename_metadata,
    )
    return r


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
        watchdog,
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
        self._watchdog = watchdog
        self.enable_metadata(True)

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

        watchdog = operators.WatchdogOp(
            self,
            name="watchdog",
            watchdog=self._watchdog,
        )

        monitor = operators.MonitorOperator(
            self,
            name="monitor",
            callback=self.monitor_callback,
        )

        #
        self.add_flow(receiver_operator, csi_to_bayer_operator, {("output", "input")})
        self.add_flow(csi_to_bayer_operator, isp_in, {("output", isp_in_name)})
        self.add_flow(isp_out, visualizer, {(isp_out_name, "receivers")})
        self.add_flow(visualizer, monitor, {("camera_pose_output", "input")})
        self.add_flow(visualizer, watchdog, {("camera_pose_output", "input")})

    def monitor_callback(self, operator, metadata):
        if not self._ptp_enable:
            return
        times = operator.get_times(metadata)
        frame_number = times["frame_number"]
        frame_time = times["frame_end"] - times["frame_start"]
        overall_time = times["complete"] - times["frame_start"]
        processing_time = times["complete"] - times["frame_end"]
        logging.info(
            f"{frame_number=} {frame_time=:.4f} {overall_time=:.4f} {processing_time=:.4f}"
        )


class MonoTest:
    def __init__(
        self,
        camera_mode,
        headless,
        frame_limit,
        hololink_address,
        receiver_factory,
        isp_factory,
        camera_factory,
        ptp_enable,
    ):
        self._camera_mode = camera_mode
        self._headless = headless
        self._frame_limit = frame_limit
        self._hololink_address = hololink_address
        self._receiver_factory = receiver_factory
        self._isp_factory = isp_factory
        self._camera_factory = camera_factory
        self._ptp_enable = ptp_enable

    def execute(self):
        # Get a handle to the GPU using context manager
        with CudaContext() as (cu_context, cu_device_ordinal):
            # Get a handle to data sources
            channel_metadata = hololink_module.Enumerator.find_channel(
                channel_ip=self._hololink_address
            )
            #
            hololink_channel = hololink_module.DataChannel(channel_metadata)
            # Set up hololink
            hololink = hololink_channel.hololink()
            hololink.start()
            try:
                hololink.reset()
                #
                if self._ptp_enable:
                    logging.debug("Waiting for PTP sync.")
                    if not hololink.ptp_synchronize():
                        raise ValueError("Failed to synchronize PTP.")
                # Get a handle to the camera
                camera = self._camera_factory(hololink_channel, 0)
                # Set up the application
                window_height = 200
                window_width = 600  # for the pair
                window_title = "Pattern test"
                with utils.Watchdog(
                    "watchdog",
                    initial_timeout=operators.color_profiler_initial_timeout(
                        self._frame_limit
                    ),
                ) as watchdog:
                    application = MonoApplication(
                        self._headless,
                        cu_context,
                        cu_device_ordinal,
                        hololink_channel,
                        camera,
                        self._camera_mode,
                        self._frame_limit,
                        window_height,
                        window_width,
                        window_title,
                        receiver_factory=self._receiver_factory,
                        isp_factory=self._isp_factory,
                        ptp_enable=self._ptp_enable,
                        watchdog=watchdog,
                    )

                    default_configuration = os.path.join(
                        os.path.dirname(__file__), "example_configuration.yaml"
                    )
                    application.config(default_configuration)
                    # Run it.
                    application.run()
            finally:
                hololink.stop()
