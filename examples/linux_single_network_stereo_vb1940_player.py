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

import argparse
import ctypes
import logging
import os
import time

import holoscan
from cuda import cuda

import hololink as hololink_module


class HoloscanApplication(holoscan.core.Application):
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
        if self._frame_limit:
            self._count_left = holoscan.conditions.CountCondition(
                self,
                name="count_left",
                count=self._frame_limit,
            )
            condition_left = self._count_left
            self._count_right = holoscan.conditions.CountCondition(
                self,
                name="count_right",
                count=self._frame_limit,
            )
            condition_right = self._count_right
        else:
            self._ok_left = holoscan.conditions.BooleanCondition(
                self, name="ok_left", enable_tick=True
            )
            condition_left = self._ok_left
            self._ok_right = holoscan.conditions.BooleanCondition(
                self, name="ok_right", enable_tick=True
            )
            condition_right = self._ok_right
        self._camera_left.set_mode(self._camera_mode)
        self._camera_right.set_mode(self._camera_mode)

        csi_to_bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="pool",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=self._camera_left.width()
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera_left.height(),
            num_blocks=6,
        )
        csi_to_bayer_operator_left = hololink_module.operators.CsiToBayerOp(
            self,
            name="csi_to_bayer_left",
            allocator=csi_to_bayer_pool,
            cuda_device_ordinal=self._cuda_device_ordinal,
            out_tensor_name="left",
        )
        self._camera_left.configure_converter(csi_to_bayer_operator_left)
        csi_to_bayer_operator_right = hololink_module.operators.CsiToBayerOp(
            self,
            name="csi_to_bayer_right",
            allocator=csi_to_bayer_pool,
            cuda_device_ordinal=self._cuda_device_ordinal,
            out_tensor_name="right",
        )
        self._camera_right.configure_converter(csi_to_bayer_operator_right)

        frame_size = csi_to_bayer_operator_left.get_csi_length()
        assert frame_size == csi_to_bayer_operator_right.get_csi_length()

        frame_context = self._cuda_context
        receiver_operator_left = hololink_module.operators.LinuxReceiverOperator(
            self,
            condition_left,
            name="receiver_left",
            frame_size=frame_size,
            frame_context=frame_context,
            hololink_channel=self._hololink_channel_left,
            device=self._camera_left,
        )

        #
        receiver_operator_right = hololink_module.operators.LinuxReceiverOperator(
            self,
            condition_right,
            frame_size=frame_size,
            frame_context=frame_context,
            hololink_channel=self._hololink_channel_right,
            device=self._camera_right,
        )

        bayer_format = self._camera_left.bayer_format()
        assert bayer_format == self._camera_right.bayer_format()
        pixel_format = self._camera_left.pixel_format()
        assert pixel_format == self._camera_right.pixel_format()
        image_processor_left = hololink_module.operators.ImageProcessorOp(
            self,
            name="image_processor_left",
            # Optical black value for vb1940 raw10 format is 8
            optical_black=8,
            bayer_format=bayer_format.value,
            pixel_format=pixel_format.value,
        )
        image_processor_right = hololink_module.operators.ImageProcessorOp(
            self,
            name="image_processor_right",
            # Optical black value for vb1940 raw10 format is 8
            optical_black=8,
            bayer_format=bayer_format.value,
            pixel_format=pixel_format.value,
        )

        rgba_components_per_pixel = 4
        bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="pool",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=self._camera_left.width()
            * rgba_components_per_pixel
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera_left.height(),
            num_blocks=6,
        )
        demosaic_left = holoscan.operators.BayerDemosaicOp(
            self,
            name="demosaic_left",
            pool=bayer_pool,
            generate_alpha=True,
            alpha_value=65535,
            bayer_grid_pos=bayer_format.value,
            interpolation_mode=0,
            in_tensor_name="left",
            out_tensor_name="left",
        )
        demosaic_right = holoscan.operators.BayerDemosaicOp(
            self,
            name="demosaic_right",
            pool=bayer_pool,
            generate_alpha=True,
            alpha_value=65535,
            bayer_grid_pos=bayer_format.value,
            interpolation_mode=0,
            in_tensor_name="right",
            out_tensor_name="right",
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
        )
        #
        self.add_flow(
            receiver_operator_left, csi_to_bayer_operator_left, {("output", "input")}
        )
        self.add_flow(
            receiver_operator_right, csi_to_bayer_operator_right, {("output", "input")}
        )
        self.add_flow(
            csi_to_bayer_operator_left, image_processor_left, {("output", "input")}
        )
        self.add_flow(
            csi_to_bayer_operator_right, image_processor_right, {("output", "input")}
        )
        self.add_flow(image_processor_left, demosaic_left, {("output", "receiver")})
        self.add_flow(image_processor_right, demosaic_right, {("output", "receiver")})
        self.add_flow(demosaic_left, visualizer, {("transmitter", "receivers")})
        self.add_flow(demosaic_right, visualizer, {("transmitter", "receivers")})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera-mode",
        type=int,
        default=hololink_module.sensors.vb1940.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS.value,
        help="VB1940 mode",
    )

    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Exit after receiving this many frames",
    )
    default_configuration = os.path.join(
        os.path.dirname(__file__), "example_configuration.yaml"
    )
    parser.add_argument(
        "--configuration",
        default=default_configuration,
        help="Configuration file",
    )
    parser.add_argument(
        "--hololink",
        default="192.168.0.2",
        help="IP address of Hololink board",
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level to display",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=2160 // 4,  # arbitrary default
        help="Set the height of the displayed window",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=3840 // 3,  # arbitrary default
        help="Set the width of the displayed window",
    )
    parser.add_argument(
        "--title",
        help="Set the window title",
    )
    parser.add_argument(
        "--trigger",
        action="store_true",
        help="Run in trigger mode",
    )
    parser.add_argument(
        "--exp",
        type=int,
        default=256,
        help="set EXPOSURE duration in lines, RANGE(4 to 65535). Default line value is 29.70usec",
    )
    parser.add_argument(
        "--gain",
        type=int,
        default=0,
        help="Set Analog Gain, RANGE(0 to 12). Default is 0. Equation is (16/(16-gain)",
    )
    parser.add_argument(
        "--frequency",
        type=int,
        default=30,
        help="VSYNC frequency in Hz (10, 30, 60, 90, 120). Default is 30Hz",
    )
    args = parser.parse_args()
    hololink_module.logging_level(args.log_level)
    logging.info("Initializing.")
    # Get a handle to the GPU
    (cu_result,) = cuda.cuInit(0)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_device_ordinal = 0
    cu_result, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_result, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS

    # Get a handle to data sources.  First, find an enumeration packet
    # from the IP address we want to use.
    channel_metadata = hololink_module.Enumerator.find_channel(channel_ip=args.hololink)
    logging.info(f"{channel_metadata=}")
    # Now make separate connection metadata for left and right; and set them to
    # use sensor 0 and 1 respectively.  This will borrow the data plane
    # configuration we found on that interface.
    channel_metadata_left = hololink_module.Metadata(channel_metadata)
    hololink_module.DataChannel.use_sensor(channel_metadata_left, 0)
    channel_metadata_right = hololink_module.Metadata(channel_metadata)
    hololink_module.DataChannel.use_sensor(channel_metadata_right, 1)
    #
    hololink_channel_left = hololink_module.DataChannel(channel_metadata_left)
    hololink_channel_right = hololink_module.DataChannel(channel_metadata_right)
    hololink = hololink_channel_left.hololink()
    assert hololink is hololink_channel_right.hololink()
    # Get a handle to the camera
    vsync = hololink_module.Synchronizer.null_synchronizer()
    if args.trigger:
        vsync = hololink.ptp_pps_output(args.frequency)
    camera_left = hololink_module.sensors.vb1940.Vb1940Cam(
        hololink_channel_left, vsync=vsync
    )
    camera_right = hololink_module.sensors.vb1940.Vb1940Cam(
        hololink_channel_right, vsync=vsync
    )
    camera_mode = hololink_module.sensors.vb1940.Vb1940_Mode(args.camera_mode)
    # What title should we use?
    window_title = f"Holoviz - {args.hololink}"
    if args.title is not None:
        window_title = args.title
    # Set up the application
    application = HoloscanApplication(
        args.headless,
        cu_context,
        cu_device_ordinal,
        hololink_channel_left,
        camera_left,
        hololink_channel_right,
        camera_right,
        camera_mode,
        args.frame_limit,
        args.window_height,
        args.window_width,
        window_title,
    )
    application.config(args.configuration)
    # Run it.
    hololink.start()
    hololink.reset()
    hololink.write_uint32(0x8, 0x0)
    camera_left.setup_clock()  # this also sets camera_right's clock
    hololink.write_uint32(0x8, 0x3)
    time.sleep(100 / 1000)
    camera_left.get_register_32(0x0000)  # DEVICE_MODEL_ID:"S940"(ASCII code:0x53393430)
    camera_left.get_register_32(0x0734)  # EXT_CLOCK(25MHz = 0x017d7840)
    camera_left.configure(camera_mode)
    camera_left.set_analog_gain_reg(args.gain)  # Gain value has to be int
    camera_left.set_exposure_reg(args.exp)  # Exposure value has to be int
    camera_right.get_register_32(
        0x0000
    )  # DEVICE_MODEL_ID:"S940"(ASCII code:0x53393430)
    camera_right.get_register_32(0x0734)  # EXT_CLOCK(25MHz = 0x017d7840)
    camera_right.configure(camera_mode)
    camera_right.set_analog_gain_reg(args.gain)  # Gain value has to be int
    camera_right.set_exposure_reg(args.exp)  # Exposure value has to be int

    # READ CAMERA EEPROM
    cal_eeprom = camera_left.get_calibration_data(0)
    print(cal_eeprom)

    application.run()
    hololink.stop()

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS


if __name__ == "__main__":
    main()
