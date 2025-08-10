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

import argparse
import ctypes
import logging

import holoscan
from cuda import cuda

import hololink as hololink_module


class MicroApplication(holoscan.core.Application):
    def __init__(
        self,
        headless,
        fullscreen,
        cuda_context,
        cuda_device_ordinal,
        hololink_channel_left,
        camera_left,
        hololink_channel_right,
        camera_right,
        frame_limit,
    ):
        logging.info("__init__")
        super().__init__()
        self._headless = headless
        self._fullscreen = fullscreen
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._hololink_channel_left = hololink_channel_left
        self._camera_left = camera_left
        self._hololink_channel_right = hololink_channel_right
        self._camera_right = camera_right
        self._frame_limit = frame_limit
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
        receiver_operator_right = hololink_module.operators.LinuxReceiverOperator(
            self,
            condition_right,
            name="receiver_right",
            frame_size=frame_size,
            frame_context=frame_context,
            hololink_channel=self._hololink_channel_right,
            device=self._camera_right,
        )

        pixel_format = self._camera_left.pixel_format()
        bayer_format = self._camera_left.bayer_format()

        image_processor_operator_left = hololink_module.operators.ImageProcessorOp(
            self,
            name="image_processor_left",
            optical_black=100,
            bayer_format=bayer_format.value,
            pixel_format=pixel_format.value,
        )
        image_processor_operator_right = hololink_module.operators.ImageProcessorOp(
            self,
            name="image_processor_right",
            optical_black=100,
            bayer_format=bayer_format.value,
            pixel_format=pixel_format.value,
        )

        rgba_components_per_pixel = 4

        bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="pool",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=self._camera_left._width
            * rgba_components_per_pixel
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera_left._height,
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
            fullscreen=self._fullscreen,
            headless=self._headless,
            framebuffer_srgb=True,
            tensors=[left_spec, right_spec],
            height=1920,
            width=1080,
        )

        self.add_flow(
            receiver_operator_left, csi_to_bayer_operator_left, {("output", "input")}
        )
        self.add_flow(
            receiver_operator_right, csi_to_bayer_operator_right, {("output", "input")}
        )
        self.add_flow(
            csi_to_bayer_operator_left,
            image_processor_operator_left,
            {("output", "input")},
        )
        self.add_flow(
            csi_to_bayer_operator_right,
            image_processor_operator_right,
            {("output", "input")},
        )
        self.add_flow(
            image_processor_operator_left, demosaic_left, {("output", "receiver")}
        )
        self.add_flow(
            image_processor_operator_right, demosaic_right, {("output", "receiver")}
        )
        self.add_flow(demosaic_left, visualizer, {("transmitter", "receivers")})
        self.add_flow(demosaic_right, visualizer, {("transmitter", "receivers")})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument(
        "--fullscreen", action="store_true", help="Run in fullscreen mode"
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Exit after receiving this many frames",
    )

    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level to display",
    )
    parser.add_argument(
        "--cam",
        type=int,
        default=0,
        choices=(0, 1),
        help="which camera to stream: 0 to stream camera connected to j14 or 1 to stream camera connected to j17 (default is 0)",
    )
    parser.add_argument(
        "--pattern",
        action="store_true",
        help="Configure to display a test pattern.",
    )
    parser.add_argument(
        "--exposure",
        type=int,
        default=0x05,
        help="Configure exposure.",
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
    #    cu_result, cu_context = cuda.cuCtxCreate(0, cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS

    # Get a handle to the Hololink device
    if args.cam == 0:
        channel_metadata = hololink_module.Enumerator.find_channel(
            channel_ip="192.168.0.2"
        )
    elif args.cam == 1:
        channel_metadata = hololink_module.Enumerator.find_channel(
            channel_ip="192.168.0.3"
        )
    else:
        raise Exception(f"Unexpected camera={args.cam}")

    channel_metadata_left = hololink_module.Metadata(channel_metadata)
    hololink_module.DataChannel.use_sensor(channel_metadata_left, 0)
    hololink_channel_left = hololink_module.DataChannel(channel_metadata_left)
    # Get a handle to the camera
    camera_left = hololink_module.sensors.imx477.Imx477(
        hololink_channel_left, camera_id=0, resolution="1080p"
    )

    channel_metadata_right = hololink_module.Metadata(channel_metadata)
    hololink_module.DataChannel.use_sensor(channel_metadata_right, 1)
    hololink_channel_right = hololink_module.DataChannel(channel_metadata_right)
    # Get a handle to the camera
    camera_right = hololink_module.sensors.imx477.Imx477(
        hololink_channel_right, camera_id=1, resolution="1080p"
    )

    # Set up the application
    application = MicroApplication(
        args.headless,
        args.fullscreen,
        cu_context,
        cu_device_ordinal,
        hololink_channel_left,
        camera_left,
        hololink_channel_right,
        camera_right,
        args.frame_limit,
    )

    # Run it.
    hololink = hololink_channel_left.hololink()
    hololink.start()
    hololink.reset()
    # Configures the camera for 3840x2160, 60fps
    camera_left.configure()
    camera_right.configure()

    # IMX477 Analog gain settings function. Analog gain value range is 0-1023 in decimal (10 bits). Users are free to experiment with the register values.
    camera_left.set_analog_gain(0x2FF)
    camera_left.set_exposure_reg(args.exposure)

    camera_right.set_analog_gain(0x2FF)
    camera_right.set_exposure_reg(args.exposure)
    if args.pattern:
        camera_left.set_pattern()
        camera_right.set_pattern()
    application.run()
    hololink.stop()


if __name__ == "__main__":
    main()
