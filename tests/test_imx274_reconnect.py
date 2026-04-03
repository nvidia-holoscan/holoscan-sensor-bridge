# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import threading

import applications
import holoscan
import numpy as np
import operators
import pytest
import utils

import hololink as hololink_module

hololink_state = {}
UNKNOWN = "UNKNOWN"
CLOCK_UP = "CLOCK_UP"
hololink_state_lock = threading.Lock()


@dataclasses.dataclass
class ComputedCrcRecord:
    computed_crc: int


class Imx274SensorFactory(hololink_module.hsb_controller.SensorFactory):
    def __init__(self, camera_factory, channel_ip, instance, camera_mode, pattern):
        super().__init__(channel_ip)
        self._camera_factory = camera_factory
        self._instance = instance  # instance is 0 or 1 for left or right
        self._camera_mode = camera_mode
        self._pattern = pattern
        #
        self._camera = None
        #
        if (
            self._camera_mode
            == hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS
        ):
            self._height, self._width = 1080, 1920
            self._pixel_format, self._bayer_format = (
                hololink_module.sensors.csi.PixelFormat.RAW_10,
                hololink_module.sensors.csi.BayerFormat.RGGB,
            )
        else:
            raise RuntimeError(f"Unsupported {camera_mode=}")

    def image_size(self):
        return self._height, self._width

    def image_format(self):
        return self._pixel_format, self._bayer_format

    def configure_converter(self, converter):
        # where do we find the first received byte?
        start_byte = converter.receiver_start_byte()
        transmitted_line_bytes = converter.transmitted_line_bytes(
            self._pixel_format, self._width
        )
        received_line_bytes = converter.received_line_bytes(transmitted_line_bytes)
        # We get 175 bytes of metadata preceding the image data.
        start_byte += converter.received_line_bytes(175)
        if self._pixel_format == hololink_module.sensors.csi.PixelFormat.RAW_10:
            # sensor has 8 lines of optical black before the real image data starts
            start_byte += received_line_bytes * 8
        elif self._pixel_format == hololink_module.sensors.csi.PixelFormat.RAW_12:
            # sensor has 16 lines of optical black before the real image data starts
            start_byte += received_line_bytes * 16
        else:
            raise Exception(f"Incorrect pixel format={self._pixel_format} for IMX274.")
        converter.configure(
            start_byte,
            received_line_bytes,
            self._width,
            self._height,
            self._pixel_format,
        )

    def new_sensor(self):
        self._camera = self._camera_factory(
            self._hololink_channel,
            expander_configuration=self._instance,
        )
        # Configuration for shared resources on HSB
        # which executes only before the first device
        # is initialized.  This is important, e.g. a single
        # clock device drives all sensors on the board.
        with hololink_state_lock:
            if hololink_state[self._hololink] is UNKNOWN:
                self._camera.setup_clock()
                hololink_state[self._hololink] = CLOCK_UP
        self._camera.configure(self._camera_mode)
        self._camera.set_digital_gain_reg(0x4)
        self._camera.test_pattern(self._pattern)
        return self._camera

    def new_data_channel(self):
        with hololink_state_lock:
            super().new_data_channel()
            hololink_state[self._hololink] = UNKNOWN
            self._hololink.on_reset(self._reset)

    def _reset(self):
        logging.info("device is reset.")
        with hololink_state_lock:
            hololink_state[self._hololink] = UNKNOWN


class StatusOp(holoscan.core.Operator):
    def __init__(
        self,
        *args,
        in_tensor_name="",
        out_tensor_name="",
        status_name="status",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._in_tensor_name = in_tensor_name
        self._out_tensor_name = out_tensor_name
        self._status_name = status_name
        self._status = np.asarray([(0.5, 0.5)])

    def setup(self, spec):
        logging.info("setup")
        spec.input("input")
        spec.output("output")
        spec.output("output_specs")

    def start(self):
        self._frame_count = 0

    def compute(self, op_input, op_output, context):
        #
        self._frame_count += 1
        #
        in_message = op_input.receive("input")
        in_tensor = in_message.get(self._in_tensor_name)
        op_output.emit(
            {
                self._status_name: self._status,
                self._out_tensor_name: in_tensor,
            },
            "output",
        )
        #
        spec = holoscan.operators.HolovizOp.InputSpec(self._status_name, "text")
        spec.text = [f"{self._frame_count=}"]
        spec.priority = 1
        op_output.emit([spec], "output_specs")


class CsiImage:
    def __init__(self, sensor_factory, filename):
        self._sensor_factory = sensor_factory
        self._filename = filename

    def receiver_start_byte(self):
        return 0

    def received_line_bytes(self, transmitted_line_bytes):
        return hololink_module.round_up(transmitted_line_bytes, 8)

    def transmitted_line_bytes(self, pixel_format, pixel_width):
        if pixel_format == hololink_module.sensors.csi.PixelFormat.RAW_8:
            return pixel_width
        if pixel_format == hololink_module.sensors.csi.PixelFormat.RAW_10:
            return pixel_width * 5 // 4
        if pixel_format == hololink_module.sensors.csi.PixelFormat.RAW_12:
            return pixel_width * 3 // 2
        raise Exception(f"Unexpected {pixel_format=}")

    def configure(self, start_byte, received_line_bytes, width, height, pixel_format):
        _, bayer_format = self._sensor_factory.image_format()
        self._csi_image = utils.make_csi_from_image_file(
            height,
            width,
            pixel_format,
            bayer_format,
            start_byte,
            received_line_bytes,
            filename=self._filename,
        )
        logging.info(f"{self._filename=} {len(self._csi_image)=}")

    def csi_image(self):
        return self._csi_image


def check_crcs(
    context,
    expected_crc,
    computed_metadata,
    get_computed_crc,
    frame_limit,
    first_frame=1,
    end_frame=2,
):
    # Don't count the last couple of frames while the pipeline shuts down
    last_frame = frame_limit - end_frame
    assert (last_frame - first_frame) > 50
    assert len(computed_metadata) > last_frame
    checked = 0
    passed = 0
    for frame, metadata in enumerate(computed_metadata):
        computed_crc = get_computed_crc(metadata)
        if frame < first_frame:
            continue
        if frame >= last_frame:
            break
        checked += 1
        if computed_crc == expected_crc:
            passed += 1
        else:
            logging.info(f"{context} {frame=} {computed_crc=:#x} {expected_crc=:#x}")
    return checked, passed


class ReconnectTestApplication(holoscan.core.Application):
    def __init__(
        self,
        headless,
        cuda_context,
        cuda_device_ordinal,
        frame_limit,
        watchdog,
        sensor_factory,
        receiver_factory,
        reset_after,
    ):
        logging.info("__init__")
        super().__init__()
        self._headless = headless
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._frame_limit = frame_limit
        self._watchdog = watchdog
        self._sensor_factory = sensor_factory
        self._receiver_factory = receiver_factory
        self._reset_after = reset_after
        # These are HSDK controls-- because we have multiple
        # data paths going into the same visualizer--one for
        # settings that brings up text display, and the other
        # is the image data--don't
        # raise an error when each path present metadata
        # with the same names.  Because we don't use that metadata,
        # it's easiest to just ignore new items with the same
        # names as existing items.
        self.is_metadata_enabled = True
        self.metadata_policy = holoscan.core.MetadataPolicy.REJECT

    def compose(self):
        # Stop execution after this many frames are received.
        self._condition = holoscan.conditions.CountCondition(
            self,
            name="condition",
            count=self._frame_limit,
        )

        # csi_to_bayer_operator is the first operator that actually
        # knows the format of the received sensor data.
        height, width = self._sensor_factory.image_size()
        csi_pool_block_size = width * height * ctypes.sizeof(ctypes.c_uint16)
        csi_to_bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="csi_to_bayer_pool",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=csi_pool_block_size,
            num_blocks=4,
        )
        csi_to_bayer_operator = hololink_module.operators.CsiToBayerOp(
            self,
            name="csi_to_bayer",
            allocator=csi_to_bayer_pool,
            cuda_device_ordinal=self._cuda_device_ordinal,
        )
        self._sensor_factory.configure_converter(csi_to_bayer_operator)

        # Now we know how large a buffer to allocate for each
        # received data frame.
        frame_size = csi_to_bayer_operator.get_csi_length()

        # What do we publish when no video is available?
        here = os.path.dirname(__file__)
        fallback_image = CsiImage(
            self._sensor_factory, filename=os.path.join(here, "SMPTE_Color_Bars.png")
        )
        self._sensor_factory.configure_converter(fallback_image)

        receiver = self._receiver_factory(
            self._cuda_context, self._cuda_device_ordinal, frame_size
        )
        hsb_controller_operator = hololink_module.operators.HsbControllerOp(
            self,
            self._condition,
            "controller",
            sensor_factory=self._sensor_factory,
            network_receiver=receiver,
            fallback_image=fallback_image,
        )

        pixel_format, bayer_format = self._sensor_factory.image_format()
        rgba_components_per_pixel = 4
        bayer_size = (
            width * rgba_components_per_pixel * ctypes.sizeof(ctypes.c_uint16) * height
        )
        bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="pool",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=bayer_size,
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

        # Compute the CRC of the bayer image and use that to
        # determine if we're receiving the expected image.
        bayer_crc = hololink_module.operators.ComputeCrcOp(
            self,
            name="bayer_crc",
            frame_size=bayer_size,
        )
        record_bayer_crc = hololink_module.operators.CheckCrcOp(
            self,
            name="record_bayer_crc",
            compute_crc_op=bayer_crc,
        )

        status = StatusOp(
            self,
            name="status",
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

        self._metadata_recorder_operator = operators.RecordMetadataOp(
            self,
            name="metadata_recorder_operator",
            metadata_class=ComputedCrcRecord,
        )

        def reset():
            logging.info("Triggering reset.")
            hololink = self._sensor_factory._hololink
            hololink.trigger_reset()

        reactor = hololink_module.Reactor.get_reactor()
        reset_operators = []
        for trigger_frame in self._reset_after:
            reset_operator = operators.OnFrameNOperator(
                self,
                name=f"reset_operator_{trigger_frame}",
                trigger_frame=trigger_frame,
                callback=lambda reactor=reactor: reactor.add_callback(reset),
            )
            reset_operators.append(reset_operator)

        #
        self.add_flow(
            hsb_controller_operator, csi_to_bayer_operator, {("output", "input")}
        )
        self.add_flow(csi_to_bayer_operator, demosaic, {("output", "receiver")})
        self.add_flow(demosaic, bayer_crc, {("transmitter", "input")})
        self.add_flow(bayer_crc, record_bayer_crc, {("output", "input")})
        self.add_flow(
            record_bayer_crc, self._metadata_recorder_operator, {("output", "input")}
        )
        self.add_flow(self._metadata_recorder_operator, status, {("output", "input")})
        self.add_flow(status, visualizer, {("output", "receivers")})
        self.add_flow(status, visualizer, {("output_specs", "input_specs")})
        self.add_flow(visualizer, watchdog_operator, {("camera_pose_output", "input")})

        for reset_operator in reset_operators:
            self.add_flow(visualizer, reset_operator, {("camera_pose_output", "input")})


def reconnect_test(
    sensor_factory,
    receiver_factory,
    headless,
    frame_limit,
    reset_after,
):
    logging.info("Initializing.")
    hololink_module.hsb_controller.reset_device_map()
    hololink_module.NvtxTrace.setThreadName("CUDA")
    with applications.CudaContext() as (cu_context, cu_device_ordinal):
        hololink_module.NvtxTrace.setThreadName("MAIN")
        with utils.Watchdog(
            "frame-reception",
            timeout=30,
        ) as watchdog:
            application = ReconnectTestApplication(
                headless,
                cu_context,
                cu_device_ordinal,
                frame_limit,
                watchdog,
                sensor_factory=sensor_factory,
                receiver_factory=receiver_factory,
                reset_after=reset_after,
            )
            application.run()
    computed_metadata = application._metadata_recorder_operator.get_record()
    return computed_metadata


stereo_modes_and_patterns = [
    (
        # left
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
        10,
        0xB718A38C,
        # right
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
        11,
        0x8F4D79DE,
    ),
]

mono_modes_and_patterns = [
    (
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
        10,
        0xB718A38C,
    ),
]


@pytest.mark.skip_unless_imx274
@pytest.mark.parametrize(
    "camera_mode, pattern, expected",
    mono_modes_and_patterns,
)
@pytest.mark.parametrize(
    "hololink_channel_ip, hololink_sensor_instance",
    [
        ("192.168.0.2", 0),
        ("192.168.0.3", 1),
    ],
)
def test_linux_imx274_reconnect(
    camera_mode,
    pattern,
    expected,
    headless,
    hololink_channel_ip,
    hololink_sensor_instance,
):
    # Normally this can be set on the command line,
    # but we're triggering after specific frame numbers here.
    frame_limit = 300
    reset_after = [150]

    sensor_factory = Imx274SensorFactory(
        hololink_module.sensors.imx274.dual_imx274.Imx274Cam,
        hololink_channel_ip,
        hololink_sensor_instance,
        camera_mode,
        pattern,
    )
    receiver_factory = (
        hololink_module.operators.linux_controller_receiver.LinuxControllerReceiver
    )

    reconnect_test(
        sensor_factory,
        receiver_factory,
        headless,
        frame_limit,
        reset_after,
    )
    # We don't check the CRCs because packet loss is so
    # common in this mode.


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
    "camera_mode, pattern, expected",
    mono_modes_and_patterns,
)
@pytest.mark.parametrize(
    "hololink_channel_ip, hololink_sensor_instance, ibv_name, ibv_port",
    [
        ("192.168.0.2", 0, sys_ibv_name_left, 1),
        ("192.168.0.3", 1, sys_ibv_name_right, 1),
    ],
)
def test_roce_imx274_reconnect(
    camera_mode,
    pattern,
    expected,
    headless,
    hololink_channel_ip,
    hololink_sensor_instance,
    ibv_name,
    ibv_port,
):
    # Normally this can be set on the command line,
    # but we're triggering after specific frame numbers here.
    frame_limit = 300
    reset_after = [150]

    sensor_factory = Imx274SensorFactory(
        hololink_module.sensors.imx274.dual_imx274.Imx274Cam,
        hololink_channel_ip,
        hololink_sensor_instance,
        camera_mode,
        pattern,
    )
    receiver_factory = lambda *args: hololink_module.operators.roce_controller_receiver.RoceControllerReceiver(  # noqa: E731
        *args, ibv_name, ibv_port
    )

    computed_metadata = reconnect_test(
        sensor_factory,
        receiver_factory,
        headless,
        frame_limit,
        reset_after,
    )

    checked, passed = check_crcs(
        "reconnect-test",
        expected,
        computed_metadata,
        lambda metadata: metadata.computed_crc,
        frame_limit,
    )
    assert checked > (frame_limit - 10)
    assert passed > (frame_limit - 10)


class StereoReconnectTestApplication(holoscan.core.Application):
    def __init__(
        self,
        headless,
        cuda_context,
        cuda_device_ordinal,
        frame_limit,
        left_watchdog,
        left_sensor_factory,
        left_receiver_factory,
        right_watchdog,
        right_sensor_factory,
        right_receiver_factory,
        reset_after,
    ):
        logging.info("__init__")
        super().__init__()
        self._headless = headless
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._frame_limit = frame_limit
        self._left_watchdog = left_watchdog
        self._left_sensor_factory = left_sensor_factory
        self._left_receiver_factory = left_receiver_factory
        self._right_watchdog = right_watchdog
        self._right_sensor_factory = right_sensor_factory
        self._right_receiver_factory = right_receiver_factory
        self._reset_after = reset_after
        # These are HSDK controls-- because we have multiple
        # data paths going into the same visualizer--one for
        # settings that brings up text display, and the other
        # is the image data--don't
        # raise an error when each path present metadata
        # with the same names.  Because we don't use that metadata,
        # it's easiest to just ignore new items with the same
        # names as existing items.
        self.is_metadata_enabled = True
        self.metadata_policy = holoscan.core.MetadataPolicy.REJECT

    def compose(self):
        def per_channel(context, sensor_factory, receiver_factory, watchdog):
            # Stop execution after this many frames are received.
            condition = holoscan.conditions.CountCondition(
                self,
                name=f"{context}_condition",
                count=self._frame_limit,
            )

            # csi_to_bayer_operator is the first operator that actually
            # knows the format of the received sensor data.
            height, width = sensor_factory.image_size()
            csi_pool_block_size = width * height * ctypes.sizeof(ctypes.c_uint16)
            csi_to_bayer_pool = holoscan.resources.BlockMemoryPool(
                self,
                name=f"{context}_csi_to_bayer_pool",
                # storage_type of 1 is device memory
                storage_type=1,
                block_size=csi_pool_block_size,
                num_blocks=4,
            )
            csi_to_bayer_operator = hololink_module.operators.CsiToBayerOp(
                self,
                name=f"{context}_csi_to_bayer",
                allocator=csi_to_bayer_pool,
                cuda_device_ordinal=self._cuda_device_ordinal,
            )
            sensor_factory.configure_converter(csi_to_bayer_operator)

            # Now we know how large a buffer to allocate for each
            # received data frame.
            frame_size = csi_to_bayer_operator.get_csi_length()

            # What do we publish when no video is available?
            here = os.path.dirname(__file__)
            fallback_image = CsiImage(
                sensor_factory, filename=os.path.join(here, "SMPTE_Color_Bars.png")
            )
            sensor_factory.configure_converter(fallback_image)

            receiver = receiver_factory(
                self._cuda_context, self._cuda_device_ordinal, frame_size
            )
            hsb_controller_operator = hololink_module.operators.HsbControllerOp(
                self,
                condition,
                name=f"{context}_controller",
                sensor_factory=sensor_factory,
                network_receiver=receiver,
                fallback_image=fallback_image,
            )

            pixel_format, bayer_format = sensor_factory.image_format()
            rgba_components_per_pixel = 4
            bayer_size = (
                width
                * rgba_components_per_pixel
                * ctypes.sizeof(ctypes.c_uint16)
                * height
            )
            bayer_pool = holoscan.resources.BlockMemoryPool(
                self,
                name=f"{context}_pool",
                # storage_type of 1 is device memory
                storage_type=1,
                block_size=bayer_size,
                num_blocks=4,
            )
            demosaic = holoscan.operators.BayerDemosaicOp(
                self,
                name=f"{context}_demosaic",
                pool=bayer_pool,
                generate_alpha=True,
                alpha_value=65535,
                bayer_grid_pos=bayer_format.value,
                interpolation_mode=0,
            )

            # Compute the CRC of the bayer image and use that to
            # determine if we're receiving the expected image.
            bayer_crc = hololink_module.operators.ComputeCrcOp(
                self,
                name=f"{context}_bayer_crc",
                frame_size=bayer_size,
            )
            record_bayer_crc = hololink_module.operators.CheckCrcOp(
                self,
                name=f"{context}_record_bayer_crc",
                compute_crc_op=bayer_crc,
            )

            status = StatusOp(
                self,
                name=f"{context}_status",
            )

            window_height = 400
            window_width = 600
            visualizer = holoscan.operators.HolovizOp(
                self,
                name=f"{context}_holoviz",
                height=window_height,
                width=window_width,
                headless=self._headless,
                framebuffer_srgb=True,
                enable_camera_pose_output=True,
                camera_pose_output_type="extrinsics_model",
            )

            watchdog_operator = operators.WatchdogOp(
                self,
                name=f"{context}_watchdog_operator",
                watchdog=watchdog,
            )

            metadata_recorder_operator = operators.RecordMetadataOp(
                self,
                name=f"{context}_metadata_recorder_operator",
                metadata_class=ComputedCrcRecord,
            )
            #
            self.add_flow(
                hsb_controller_operator, csi_to_bayer_operator, {("output", "input")}
            )
            self.add_flow(csi_to_bayer_operator, demosaic, {("output", "receiver")})
            self.add_flow(demosaic, bayer_crc, {("transmitter", "input")})
            self.add_flow(bayer_crc, record_bayer_crc, {("output", "input")})
            self.add_flow(
                record_bayer_crc, metadata_recorder_operator, {("output", "input")}
            )
            self.add_flow(metadata_recorder_operator, status, {("output", "input")})
            self.add_flow(status, visualizer, {("output", "receivers")})
            self.add_flow(status, visualizer, {("output_specs", "input_specs")})
            self.add_flow(
                visualizer, watchdog_operator, {("camera_pose_output", "input")}
            )
            #
            return visualizer, metadata_recorder_operator

        left_visualizer, self._left_metadata_recorder_operator = per_channel(
            "left",
            self._left_sensor_factory,
            self._left_receiver_factory,
            self._left_watchdog,
        )
        right_visualizer, self._right_metadata_recorder_operator = per_channel(
            "right",
            self._right_sensor_factory,
            self._right_receiver_factory,
            self._right_watchdog,
        )

        def reset():
            logging.info("Triggering reset.")
            hololink = self._left_sensor_factory._hololink
            hololink.trigger_reset()

        reactor = hololink_module.Reactor.get_reactor()
        reset_operators = []
        for trigger_frame in self._reset_after:
            reset_operator = operators.OnFrameNOperator(
                self,
                name=f"reset_operator_{trigger_frame}",
                trigger_frame=trigger_frame,
                callback=lambda reactor=reactor: reactor.add_callback(reset),
            )
            reset_operators.append(reset_operator)

        #
        for reset_operator in reset_operators:
            self.add_flow(
                left_visualizer, reset_operator, {("camera_pose_output", "input")}
            )


def stereo_reconnect_test(
    left_sensor_factory,
    left_receiver_factory,
    right_sensor_factory,
    right_receiver_factory,
    headless,
    frame_limit,
    reset_after,
):
    logging.info("Initializing.")
    hololink_module.hsb_controller.reset_device_map()
    hololink_module.NvtxTrace.setThreadName("CUDA")
    with applications.CudaContext() as (cu_context, cu_device_ordinal):
        hololink_module.NvtxTrace.setThreadName("MAIN")
        with utils.Watchdog(
            "left_frame-reception",
            timeout=30,
        ) as left_watchdog:
            with utils.Watchdog(
                "right_frame-reception",
                timeout=30,
            ) as right_watchdog:
                application = StereoReconnectTestApplication(
                    headless,
                    cu_context,
                    cu_device_ordinal,
                    frame_limit,
                    left_watchdog,
                    left_sensor_factory,
                    left_receiver_factory,
                    right_watchdog,
                    right_sensor_factory,
                    right_receiver_factory,
                    reset_after=reset_after,
                )
                application.run()

    left_computed_metadata = application._left_metadata_recorder_operator.get_record()
    right_computed_metadata = application._right_metadata_recorder_operator.get_record()
    return left_computed_metadata, right_computed_metadata


@pytest.mark.skip_unless_imx274
@pytest.mark.parametrize(
    "camera_mode_left, pattern_left, expected_crc_left, camera_mode_right, pattern_right, expected_crc_right",
    stereo_modes_and_patterns,
)
def test_stereo_linux_imx274_reconnect(
    camera_mode_left,
    pattern_left,
    expected_crc_left,
    camera_mode_right,
    pattern_right,
    expected_crc_right,
    headless,
    channel_ips,
):
    channel_ip_left, channel_ip_right = channel_ips

    # Normally this can be set on the command line,
    # but we're triggering after specific frame numbers here.
    frame_limit = 300
    reset_after = [150]

    sensor_instance_left = 0
    sensor_factory_left = Imx274SensorFactory(
        hololink_module.sensors.imx274.dual_imx274.Imx274Cam,
        channel_ip_left,
        sensor_instance_left,
        camera_mode_left,
        pattern_left,
    )
    sensor_instance_right = 1
    sensor_factory_right = Imx274SensorFactory(
        hololink_module.sensors.imx274.dual_imx274.Imx274Cam,
        channel_ip_right,
        sensor_instance_right,
        camera_mode_right,
        pattern_right,
    )
    receiver_factory_left = (
        hololink_module.operators.linux_controller_receiver.LinuxControllerReceiver
    )
    receiver_factory_right = (
        hololink_module.operators.linux_controller_receiver.LinuxControllerReceiver
    )

    stereo_reconnect_test(
        sensor_factory_left,
        receiver_factory_left,
        sensor_factory_right,
        receiver_factory_right,
        headless,
        frame_limit,
        reset_after,
    )
    # We don't check CRCs in nonaccelerated mode due
    # to excessive packet losses.


@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode_left, pattern_left, expected_crc_left, camera_mode_right, pattern_right, expected_crc_right",
    stereo_modes_and_patterns,
)
@pytest.mark.parametrize(
    "ibv_name_left, ibv_port_left, ibv_name_right, ibv_port_right",  # noqa: E501
    [
        (sys_ibv_name_left, 1, sys_ibv_name_right, 1),
    ],
)
@pytest.mark.parametrize(
    "channel_ip_left, sensor_instance_left, channel_ip_right, sensor_instance_right",
    [
        ("192.168.0.2", 0, "192.168.0.3", 1),
    ],
)
def test_stereo_roce_imx274_reconnect(
    camera_mode_left,
    pattern_left,
    expected_crc_left,
    ibv_name_left,
    ibv_port_left,
    channel_ip_left,
    sensor_instance_left,
    camera_mode_right,
    pattern_right,
    expected_crc_right,
    ibv_name_right,
    ibv_port_right,
    channel_ip_right,
    sensor_instance_right,
    headless,
):
    # Normally this can be set on the command line,
    # but we're triggering after specific frame numbers here.
    frame_limit = 300
    reset_after = [150]

    sensor_factory_left = Imx274SensorFactory(
        hololink_module.sensors.imx274.dual_imx274.Imx274Cam,
        channel_ip_left,
        sensor_instance_left,
        camera_mode_left,
        pattern_left,
    )
    sensor_factory_right = Imx274SensorFactory(
        hololink_module.sensors.imx274.dual_imx274.Imx274Cam,
        channel_ip_right,
        sensor_instance_right,
        camera_mode_right,
        pattern_right,
    )
    receiver_factory_left = lambda *args: hololink_module.operators.roce_controller_receiver.RoceControllerReceiver(  # noqa: E731
        *args, ibv_name_left, ibv_port_left
    )
    receiver_factory_right = lambda *args: hololink_module.operators.roce_controller_receiver.RoceControllerReceiver(  # noqa: E731
        *args, ibv_name_right, ibv_port_right
    )

    computed_metadata_left, computed_metadata_right = stereo_reconnect_test(
        sensor_factory_left,
        receiver_factory_left,
        sensor_factory_right,
        receiver_factory_right,
        headless,
        frame_limit,
        reset_after,
    )
    # Check left side CRCs
    checked, passed = check_crcs(
        "stereo_reconnect_test-left",
        expected_crc_left,
        computed_metadata_left,
        lambda metadata: metadata.computed_crc,
        frame_limit,
    )
    assert checked > (frame_limit - 10)
    assert passed > (frame_limit - 10)
    # Check right side CRCs
    checked, passed = check_crcs(
        "stereo_reconnect_test-right",
        expected_crc_right,
        computed_metadata_right,
        lambda metadata: metadata.computed_crc,
        frame_limit,
    )
    assert checked > (frame_limit - 10)
    assert passed > (frame_limit - 10)


class InstrumentedImx274CamContext:
    def __init__(self, set_registers_trigger=None):
        self._set_registers_trigger = set_registers_trigger
        self._reactor = hololink_module.Reactor.get_reactor()
        self._set_register_calls = 0

    def set_register(self, camera, register, value, timeout=None):
        self._set_register_calls += 1
        if self._set_register_calls == self._set_registers_trigger:
            self.trigger_reset(camera._hololink)

    def trigger_reset(self, hololink):
        logging.info("Triggering reset.")
        hololink.trigger_reset()


class InstrumentedImx274Cam(hololink_module.sensors.imx274.dual_imx274.Imx274Cam):
    def __init__(self, *args, context=None, **kwargs):
        super().__init__(*args, **kwargs)
        #
        self._context = context

    def configure(self, mode):
        logging.info(f"Configuring, {mode=}.")
        super().configure(mode)

    def set_register(self, register, value, timeout=None):
        self._context.set_register(self, register, value, timeout)
        super().set_register(register, value, timeout)


@pytest.mark.skip_unless_imx274
@pytest.mark.parametrize(
    "camera_mode, pattern, expected",
    mono_modes_and_patterns,
)
@pytest.mark.parametrize(
    "hololink_channel_ip, hololink_sensor_instance",
    [
        ("192.168.0.2", 0),
    ],
)
def test_linux_imx274_reconnect_during_configuration(
    camera_mode,
    pattern,
    expected,
    headless,
    hololink_channel_ip,
    hololink_sensor_instance,
):
    # Normally this can be set on the command line,
    # but we're triggering after specific frame numbers here.
    frame_limit = 100
    reset_after = []

    camera_context = InstrumentedImx274CamContext(set_registers_trigger=20)
    camera_sensor_factory = lambda *args, **kwargs: InstrumentedImx274Cam(  # noqa: E731
        context=camera_context, *args, **kwargs
    )

    sensor_factory = Imx274SensorFactory(
        camera_sensor_factory,
        hololink_channel_ip,
        hololink_sensor_instance,
        camera_mode,
        pattern,
    )
    receiver_factory = (
        hololink_module.operators.linux_controller_receiver.LinuxControllerReceiver
    )

    reconnect_test(
        sensor_factory,
        receiver_factory,
        headless,
        frame_limit,
        reset_after,
    )
    #
    logging.info(f"{camera_context._set_register_calls=}")
    # We don't check the CRCs because packet loss is so
    # common in this mode.


@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode, pattern, expected",
    mono_modes_and_patterns,
)
@pytest.mark.parametrize(
    "hololink_channel_ip, hololink_sensor_instance, ibv_name, ibv_port",
    [
        ("192.168.0.2", 0, sys_ibv_name_left, 1),
        ("192.168.0.3", 1, sys_ibv_name_right, 1),
    ],
)
def test_roce_imx274_reconnect_during_configuration(
    camera_mode,
    pattern,
    expected,
    headless,
    hololink_channel_ip,
    hololink_sensor_instance,
    ibv_name,
    ibv_port,
):
    # Normally this can be set on the command line,
    # but we're triggering after specific frame numbers here.
    frame_limit = 100
    reset_after = []

    camera_context = InstrumentedImx274CamContext(set_registers_trigger=20)
    camera_sensor_factory = lambda *args, **kwargs: InstrumentedImx274Cam(  # noqa: E731
        context=camera_context, *args, **kwargs
    )

    sensor_factory = Imx274SensorFactory(
        camera_sensor_factory,
        hololink_channel_ip,
        hololink_sensor_instance,
        camera_mode,
        pattern,
    )
    receiver_factory = lambda *args: hololink_module.operators.roce_controller_receiver.RoceControllerReceiver(  # noqa: E731
        *args, ibv_name, ibv_port
    )

    computed_metadata = reconnect_test(
        sensor_factory,
        receiver_factory,
        headless,
        frame_limit,
        reset_after,
    )

    checked, passed = check_crcs(
        "reconnect-test",
        expected,
        computed_metadata,
        lambda metadata: metadata.computed_crc,
        frame_limit,
    )
    assert checked > (frame_limit - 10)
    assert passed > (frame_limit - 10)


@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode_left, pattern_left, expected_crc_left, camera_mode_right, pattern_right, expected_crc_right",
    stereo_modes_and_patterns,
)
@pytest.mark.parametrize(
    "ibv_name_left, ibv_port_left, ibv_name_right, ibv_port_right",  # noqa: E501
    [
        (sys_ibv_name_left, 1, sys_ibv_name_right, 1),
    ],
)
@pytest.mark.parametrize(
    "channel_ip_left, sensor_instance_left, channel_ip_right, sensor_instance_right",
    [
        ("192.168.0.2", 0, "192.168.0.3", 1),
    ],
)
def test_stereo_roce_imx274_reconnect_during_configuration(
    camera_mode_left,
    pattern_left,
    expected_crc_left,
    ibv_name_left,
    ibv_port_left,
    channel_ip_left,
    sensor_instance_left,
    camera_mode_right,
    pattern_right,
    expected_crc_right,
    ibv_name_right,
    ibv_port_right,
    channel_ip_right,
    sensor_instance_right,
    headless,
):
    # Normally this can be set on the command line,
    # but we're triggering after specific frame numbers here.
    frame_limit = 150
    reset_after = []

    camera_context_left = InstrumentedImx274CamContext(set_registers_trigger=20)
    camera_sensor_factory_left = (
        lambda *args, **kwargs: InstrumentedImx274Cam(  # noqa: E731
            context=camera_context_left, *args, **kwargs
        )
    )

    camera_sensor_factory_right = hololink_module.sensors.imx274.dual_imx274.Imx274Cam

    sensor_factory_left = Imx274SensorFactory(
        camera_sensor_factory_left,
        channel_ip_left,
        sensor_instance_left,
        camera_mode_left,
        pattern_left,
    )
    sensor_factory_right = Imx274SensorFactory(
        camera_sensor_factory_right,
        channel_ip_right,
        sensor_instance_right,
        camera_mode_right,
        pattern_right,
    )
    receiver_factory_left = lambda *args: hololink_module.operators.roce_controller_receiver.RoceControllerReceiver(  # noqa: E731
        *args, ibv_name_left, ibv_port_left
    )
    receiver_factory_right = lambda *args: hololink_module.operators.roce_controller_receiver.RoceControllerReceiver(  # noqa: E731
        *args, ibv_name_right, ibv_port_right
    )

    computed_metadata_left, computed_metadata_right = stereo_reconnect_test(
        sensor_factory_left,
        receiver_factory_left,
        sensor_factory_right,
        receiver_factory_right,
        headless,
        frame_limit,
        reset_after,
    )
    # Check left side CRCs
    checked, passed = check_crcs(
        "stereo_reconnect_test-left",
        expected_crc_left,
        computed_metadata_left,
        lambda metadata: metadata.computed_crc,
        frame_limit,
    )
    assert checked > (frame_limit - 10)
    assert passed > (frame_limit - 10)
    # Check right side CRCs
    checked, passed = check_crcs(
        "stereo_reconnect_test-right",
        expected_crc_right,
        computed_metadata_right,
        lambda metadata: metadata.computed_crc,
        frame_limit,
    )
    assert checked > (frame_limit - 10)
    assert passed > (frame_limit - 10)
