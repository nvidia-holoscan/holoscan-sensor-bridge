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
import dataclasses
import logging
import time

import cuda.bindings.driver as cuda
import cupy as cp
import holoscan
import operators
import pytest
import utils

import hololink as hololink_module


class UpdateCameraOp(holoscan.core.Operator):
    def __init__(
        self,
        *args,
        callback=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._callback = callback
        self._last_time = None
        self._frame_count = 0

    def setup(self, spec):
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("input")
        cp_frame = cp.asarray(in_message.get(""))
        op_output.emit({"": cp_frame}, "output")
        #
        self._frame_count += 1
        self._callback(self)
        #
        now = time.monotonic()
        interval = 1  # second
        if self._last_time is None:
            self._last_time = now
        elif (now - self._last_time) > interval:
            self._last_time += interval
            logging.info(f"{self._frame_count=}")


@dataclasses.dataclass
class MetadataRecord:
    crc: int  # CRC accumulated by HSB FPGA
    check_crc: int  # CRC calculated over received buffer; should be same as above
    image_crc: int  # CRC calculated over demosaic'd data; for image validation
    imm_data: int
    bytes_written: int


# test IMX274 settings synchronized with frame-end.
class Imx274I2cTestApplication(holoscan.core.Application):
    def __init__(
        self,
        headless,
        cuda_context,
        cuda_device_ordinal,
        hololink_channel,
        ibv_name,
        ibv_port,
        camera,
        camera_mode,
        frame_limit,
        watchdog,
        patterns,
    ):
        logging.info("__init__")
        super().__init__()
        self._headless = headless
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._hololink_channel = hololink_channel
        self._ibv_name = ibv_name
        self._ibv_port = ibv_port
        self._camera = camera
        self._camera_mode = camera_mode
        self._frame_limit = frame_limit
        self._watchdog = watchdog
        self._patterns = patterns

    def compose(self):
        logging.info("compose")
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

        self._condition = holoscan.conditions.CountCondition(
            self,
            name="count_condition",
            count=self._frame_limit,
        )

        frame_size = csi_to_bayer_operator.get_csi_length()
        frame_context = self._cuda_context
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
        compute_crc = hololink_module.operators.ComputeCrcOp(
            self,
            name="compute_crc",
            frame_size=frame_size,
        )
        check_crc = hololink_module.operators.CheckCrcOp(
            self,
            name="check_crc",
            compute_crc_op=compute_crc,
            computed_crc_metadata_name="check_crc",
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
        rgba_image_size = (
            self._camera._width
            * rgba_components_per_pixel
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera._height
        )
        bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="pool",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=rgba_image_size,
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

        compute_image_crc = hololink_module.operators.ComputeCrcOp(
            self,
            name="compute_image_crc",
            frame_size=rgba_image_size,
        )
        check_image_crc = hololink_module.operators.CheckCrcOp(
            self,
            name="check_image_crc",
            compute_crc_op=compute_image_crc,
            computed_crc_metadata_name="image_crc",
        )

        # The current received frame was configured two frames ago;
        # self._pattern_queue keeps track of this.
        def update_camera_callback(operator):
            n = operator._frame_count % len(self._patterns)
            next_pattern = self._patterns[n]
            self._camera.synchronized_test_pattern_update(next_pattern)
            self._pattern_queue.append(next_pattern)

        update_camera = UpdateCameraOp(
            self,
            name="update_camera",
            callback=update_camera_callback,
        )

        pattern = self._patterns[-1]
        self._camera.test_pattern(pattern)
        self._pattern_queue = [pattern, pattern]

        #
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

        self._record_metadata = operators.RecordMetadataOp(
            self,
            name="record_metadata",
            metadata_class=MetadataRecord,
        )

        #
        self.add_flow(receiver_operator, update_camera, {("output", "input")})
        self.add_flow(update_camera, compute_crc, {("output", "input")})
        self.add_flow(compute_crc, check_crc, {("output", "input")})
        self.add_flow(check_crc, csi_to_bayer_operator, {("output", "input")})
        self.add_flow(
            csi_to_bayer_operator, image_processor_operator, {("output", "input")}
        )
        self.add_flow(image_processor_operator, demosaic, {("output", "receiver")})
        self.add_flow(demosaic, compute_image_crc, {("transmitter", "input")})
        self.add_flow(compute_image_crc, check_image_crc, {("output", "input")})
        self.add_flow(check_image_crc, self._record_metadata, {("output", "input")})
        self.add_flow(self._record_metadata, visualizer, {("output", "receivers")})
        self.add_flow(visualizer, watchdog, {("camera_pose_output", "input")})


# fmt: off
imx274_expected_color_profile = {
    10: 0xA9072F9B,
    11: 0xD5A4409B,
}
# fmt: on


@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
    ],
)
@pytest.mark.parametrize(
    "patterns",
    [
        list(imx274_expected_color_profile.keys()),
    ],
)
def test_imx274_synchronized_i2c_settings(
    camera_mode,
    patterns,
    headless,
    frame_limit,
    hololink_address,
    ibv_name,
    ibv_port,
):
    # Get a handle to the GPU
    (cu_result,) = cuda.cuInit(0)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_device_ordinal = 0
    cu_result, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_result, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    # Get a handle to the Hololink device
    channel_metadata = hololink_module.Enumerator.find_channel(
        channel_ip=hololink_address
    )
    hololink_channel = hololink_module.DataChannel(channel_metadata)
    # Get a handle to the camera
    camera = hololink_module.sensors.imx274.dual_imx274.Imx274Cam(
        hololink_channel,
        expander_configuration=0,
    )
    ready_frame = 10  # ignore this many initial frames while the pipeline initializes
    initial_timeout = utils.timeout_sequence(
        [(30, ready_frame), (0.5, frame_limit - ready_frame - 2), (30, 1)]
    )
    with utils.Watchdog(
        "frame-reception",
        initial_timeout=initial_timeout,
    ) as watchdog:
        # Set up the application
        application = Imx274I2cTestApplication(
            headless,
            cu_context,
            cu_device_ordinal,
            hololink_channel,
            ibv_name,
            ibv_port,
            camera,
            camera_mode,
            frame_limit,
            watchdog,
            patterns,
        )
        # Run it.
        hololink = hololink_channel.hololink()
        hololink.start()
        try:
            hololink.reset()
            camera.setup_clock()
            camera.configure(camera_mode)
            application.run()
        finally:
            hololink.stop()
    start_frames = 10
    # the pipeline sees frames that we don't due
    # to longer startup for some pipeline elements
    shutdown_frames = frame_limit - 10
    bad = False
    records = application._record_metadata.get_record()
    pattern_queue = application._pattern_queue
    # records is a list of MetadataRecord
    for n, record in enumerate(records):
        pattern = pattern_queue.pop(0)
        received_crc = record.crc
        check_crc = record.check_crc  # should be same as received_crc
        image_crc = record.image_crc
        expected_image_crc = imx274_expected_color_profile[pattern]
        message = f"{n=} {received_crc=:#x} {check_crc=:#x} {pattern=} {expected_image_crc=:#x} {image_crc=:#x}"
        ok = (received_crc == check_crc) and (image_crc == expected_image_crc)
        if ok:
            logging.debug(message)
            continue
        if n < start_frames:
            logging.debug(message)
            logging.info(f"Frame {n} didn't match but is during startup; ignoring.")
            continue
        if n > shutdown_frames:
            logging.debug(message)
            logging.info(f"Frame {n} didn't match but is during shutdown; ignoring.")
            continue
        logging.error(message)
        bad = True
    assert not bad

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS


# test BOTH CHANNELS on Stereo IMX274 settings synchronized with frame-end
class StereoImx274I2cTestApplication(holoscan.core.Application):
    def __init__(
        self,
        headless,
        cuda_context,
        cuda_device_ordinal,
        hololink_channel_left,
        ibv_name_left,
        ibv_port_left,
        camera_left,
        camera_mode_left,
        hololink_channel_right,
        ibv_name_right,
        ibv_port_right,
        camera_right,
        camera_mode_right,
        watchdog,
        frame_limit,
        patterns,
    ):
        logging.info("__init__")
        super().__init__()
        self._headless = headless
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._hololink_channel_left = hololink_channel_left
        self._ibv_name_left = ibv_name_left
        self._ibv_port_left = ibv_port_left
        self._camera_left = camera_left
        self._camera_mode_left = camera_mode_left
        self._hololink_channel_right = hololink_channel_right
        self._ibv_name_right = ibv_name_right
        self._ibv_port_right = ibv_port_right
        self._camera_right = camera_right
        self._camera_mode_right = camera_mode_right
        self._frame_limit = frame_limit
        self._watchdog = watchdog
        self._patterns = patterns
        # These are HSDK controls-- because we have stereo
        # camera paths going into the same visualizer, don't
        # raise an error when each path present metadata
        # with the same names.  Because we don't use that metadata,
        # it's easiest to just ignore new items with the same
        # names as existing items.
        self.is_metadata_enabled = True
        self.metadata_policy = holoscan.core.MetadataPolicy.REJECT

    def compose(self):
        logging.info("compose")

        def compose_camera(
            context,
            camera,
            camera_mode,
            ibv_name,
            ibv_port,
            hololink_channel,
            update_camera_callback,
            out_tensor_name,
            metadata_class,
        ):
            camera.set_mode(camera_mode)

            csi_to_bayer_pool = holoscan.resources.BlockMemoryPool(
                self,
                name=f"csi_to_bayer_pool_{context}",
                # storage_type of 1 is device memory
                storage_type=1,
                block_size=camera._width
                * ctypes.sizeof(ctypes.c_uint16)
                * camera._height,
                num_blocks=4,
            )
            csi_to_bayer_operator = hololink_module.operators.CsiToBayerOp(
                self,
                name=f"csi_to_bayer_{context}",
                allocator=csi_to_bayer_pool,
                cuda_device_ordinal=self._cuda_device_ordinal,
            )
            camera.configure_converter(csi_to_bayer_operator)

            condition = holoscan.conditions.CountCondition(
                self,
                name=f"count_condition_{context}",
                count=self._frame_limit,
            )

            frame_size = csi_to_bayer_operator.get_csi_length()
            frame_context = self._cuda_context
            receiver_operator = hololink_module.operators.RoceReceiverOp(
                self,
                condition,
                name=f"receiver_{context}",
                frame_size=frame_size,
                frame_context=frame_context,
                ibv_name=ibv_name,
                ibv_port=ibv_port,
                hololink_channel=hololink_channel,
                device=camera,
            )

            compute_crc = hololink_module.operators.ComputeCrcOp(
                self,
                name=f"compute_crc_{context}",
                frame_size=frame_size,
            )
            check_crc = hololink_module.operators.CheckCrcOp(
                self,
                name=f"check_crc_{context}",
                compute_crc_op=compute_crc,
                computed_crc_metadata_name="check_crc",
            )

            pixel_format = camera.pixel_format()
            bayer_format = camera.bayer_format()
            image_processor_operator = hololink_module.operators.ImageProcessorOp(
                self,
                name=f"image_processor_{context}",
                # Optical black value for imx274 is 50
                optical_black=50,
                bayer_format=bayer_format.value,
                pixel_format=pixel_format.value,
            )

            rgba_components_per_pixel = 4
            rgba_image_size = (
                camera._width
                * rgba_components_per_pixel
                * ctypes.sizeof(ctypes.c_uint16)
                * camera._height
            )
            bayer_pool = holoscan.resources.BlockMemoryPool(
                self,
                name=f"bayer_pool_{context}",
                # storage_type of 1 is device memory
                storage_type=1,
                block_size=rgba_image_size,
                num_blocks=4,
            )
            demosaic = holoscan.operators.BayerDemosaicOp(
                self,
                name=f"demosaic_{context}",
                pool=bayer_pool,
                generate_alpha=True,
                alpha_value=65535,
                bayer_grid_pos=bayer_format.value,
                interpolation_mode=0,
            )

            compute_image_crc = hololink_module.operators.ComputeCrcOp(
                self,
                name=f"compute_image_crc_{context}",
                frame_size=rgba_image_size,
            )
            check_image_crc = hololink_module.operators.CheckCrcOp(
                self,
                name=f"check_image_crc_{context}",
                compute_crc_op=compute_image_crc,
                computed_crc_metadata_name="image_crc",
            )

            update_camera = UpdateCameraOp(
                self,
                name=f"update_camera_{context}",
                callback=update_camera_callback,
            )

            record_metadata = operators.RecordMetadataOp(
                self,
                name=f"record_metadata_{context}",
                metadata_class=metadata_class,
                out_tensor_name=out_tensor_name,
            )
            #
            self.add_flow(receiver_operator, update_camera, {("output", "input")})
            self.add_flow(update_camera, compute_crc, {("output", "input")})
            self.add_flow(compute_crc, check_crc, {("output", "input")})
            self.add_flow(check_crc, csi_to_bayer_operator, {("output", "input")})
            self.add_flow(
                csi_to_bayer_operator, image_processor_operator, {("output", "input")}
            )
            self.add_flow(image_processor_operator, demosaic, {("output", "receiver")})
            self.add_flow(demosaic, compute_image_crc, {("transmitter", "input")})
            self.add_flow(compute_image_crc, check_image_crc, {("output", "input")})
            self.add_flow(check_image_crc, record_metadata, {("output", "input")})

            return condition, record_metadata

        # LEFT
        def update_camera_callback_left(operator):
            n = operator._frame_count % len(self._patterns)
            next_pattern = self._patterns[n]
            self._camera_left.synchronized_test_pattern_update(next_pattern)
            self._pattern_queue_left.append(next_pattern)

        pattern = self._patterns[-1]
        self._camera_left.test_pattern(pattern)
        self._pattern_queue_left = [pattern, pattern]

        condition_left, self._metadata_recorder_left = compose_camera(
            "left",
            self._camera_left,
            self._camera_mode_left,
            self._ibv_name_left,
            self._ibv_port_left,
            self._hololink_channel_left,
            update_camera_callback_left,
            out_tensor_name="left",
            metadata_class=MetadataRecord,
        )

        # RIGHT
        def update_camera_callback_right(operator):
            n = operator._frame_count % len(self._patterns)
            next_pattern = self._patterns[n]
            self._camera_right.synchronized_test_pattern_update(next_pattern)
            self._pattern_queue_right.append(next_pattern)

        self._camera_right.test_pattern(pattern)
        self._pattern_queue_right = [pattern, pattern]

        condition_right, self._metadata_recorder_right = compose_camera(
            "right",
            self._camera_right,
            self._camera_mode_right,
            self._ibv_name_right,
            self._ibv_port_right,
            self._hololink_channel_right,
            update_camera_callback_right,
            out_tensor_name="right",
            metadata_class=MetadataRecord,
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
            name="holoviz",
            headless=self._headless,
            framebuffer_srgb=True,
            tensors=[left_spec, right_spec],
            height=window_height,
            width=window_width,
            window_title="IMX274 I2C test",
            enable_camera_pose_output=True,
            camera_pose_output_type="extrinsics_model",
        )

        watchdog = operators.WatchdogOp(
            self,
            name="watchdog",
            watchdog=self._watchdog,
        )
        #
        self.add_flow(
            self._metadata_recorder_left, visualizer, {("output", "receivers")}
        )
        self.add_flow(
            self._metadata_recorder_right, visualizer, {("output", "receivers")}
        )
        self.add_flow(visualizer, watchdog, {("camera_pose_output", "input")})


# This may execute on unaccelerated configurations, where
# there may be any number of infiniband interfaces (but
# most likely zero).  In this case, placate parametrize
# by providing dummy None values in these columns.
sys_ibv_name_left, sys_ibv_name_right = (
    hololink_module.infiniband_devices() + [None, None]
)[:2]


@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode_left, camera_mode_right",  # noqa: E501
    [
        (
            hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
            hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
        ),
    ],
)
@pytest.mark.parametrize(
    "patterns",
    [
        list(imx274_expected_color_profile.keys()),
    ],
)
@pytest.mark.parametrize(
    "ibv_name_left, ibv_name_right",
    [
        (sys_ibv_name_left, sys_ibv_name_right),
    ],
)
def test_stereo_imx274_synchronized_i2c_settings(
    camera_mode_left,
    camera_mode_right,
    patterns,
    headless,
    frame_limit,
    channel_ips,
    ibv_name_left,
    ibv_name_right,
):
    # Get a handle to the GPU
    (cu_result,) = cuda.cuInit(0)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_device_ordinal = 0
    cu_result, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_result, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    # Get handles to Hololink devices
    channel_metadata_left = hololink_module.Enumerator.find_channel(
        channel_ip=channel_ips[0],
    )
    channel_metadata_right = hololink_module.Enumerator.find_channel(
        channel_ip=channel_ips[1],
    )
    hololink_channel_left = hololink_module.DataChannel(channel_metadata_left)
    hololink_channel_right = hololink_module.DataChannel(channel_metadata_right)
    # Get handles to cameras
    camera_left = hololink_module.sensors.imx274.dual_imx274.Imx274Cam(
        hololink_channel_left,
        expander_configuration=0,
    )
    camera_right = hololink_module.sensors.imx274.dual_imx274.Imx274Cam(
        hololink_channel_right,
        expander_configuration=1,
    )
    hololink = hololink_channel_left.hololink()
    assert hololink is hololink_channel_right.hololink()
    ready_frame = 10  # ignore this many initial frames while the pipeline initializes
    initial_timeout = utils.timeout_sequence(
        [(30, ready_frame), (0.5, frame_limit - ready_frame - 2), (30, 1)]
    )
    with utils.Watchdog(
        "frame-reception",
        initial_timeout=initial_timeout,
    ) as watchdog:
        ibv_port_left, ibv_port_right = 1, 1
        # Set up the application
        application = StereoImx274I2cTestApplication(
            headless,
            cu_context,
            cu_device_ordinal,
            hololink_channel_left,
            ibv_name_left,
            ibv_port_left,
            camera_left,
            camera_mode_left,
            hololink_channel_right,
            ibv_name_right,
            ibv_port_right,
            camera_right,
            camera_mode_right,
            watchdog,
            frame_limit,
            patterns,
        )
        # Run it.
        hololink.start()
        try:
            hololink.reset()
            camera_left.setup_clock()
            camera_left.configure(camera_mode_left)
            camera_right.configure(camera_mode_right)
            application.run()
        finally:
            hololink.stop()

    def check_records(context, records, pattern_queue):
        # the pipeline sees frames that we don't due
        # to longer startup for some pipeline elements
        shutdown_frames = frame_limit - 10
        bad = False
        for n, record in enumerate(records):
            pattern = pattern_queue.pop(0)
            received_crc = record.crc
            check_crc = record.check_crc  # should be same as received_crc
            image_crc = record.image_crc
            expected_image_crc = imx274_expected_color_profile[pattern]
            message = f"{context=} {n=} {received_crc=:#x} {check_crc=:#x} {pattern=} {expected_image_crc=:#x} {image_crc=:#x}"
            ok = (received_crc == check_crc) and (image_crc == expected_image_crc)
            if ok:
                logging.debug(message)
                continue
            if n < ready_frame:
                logging.debug(message)
                logging.info(f"Frame {n} didn't match but is during startup; ignoring.")
                continue
            if n > shutdown_frames:
                logging.debug(message)
                logging.info(
                    f"Frame {n} didn't match but is during shutdown; ignoring."
                )
                continue
            logging.error(message)
            bad = True
        return bad

    bad_left = check_records(
        "left",
        application._metadata_recorder_left.get_record(),
        application._pattern_queue_left,
    )
    bad_right = check_records(
        "right",
        application._metadata_recorder_right.get_record(),
        application._pattern_queue_right,
    )

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS

    assert not bad_left and not bad_right
