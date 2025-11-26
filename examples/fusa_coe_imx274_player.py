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

import argparse
import ctypes
import logging

import holoscan

import hololink as hololink_module


class App(holoscan.core.Application):
    def __init__(
        self,
        hololink_channels,
        cameras,
        camera_mode,
        timeout,
        headless,
        width,
        height,
        frame_limit,
        interface,
    ):
        super().__init__()
        self._hololink_channels = hololink_channels
        self._cameras = cameras
        self._camera_mode = camera_mode
        self._timeout = timeout
        self._headless = headless
        self._width = width
        self._height = height
        self._frame_limit = frame_limit
        self._interface = interface

        # Because we have stereo camera paths going into the same visualizer, don't
        # raise an error when each path presents metadata with the same names.
        self.enable_metadata(True)
        self.metadata_policy = holoscan.core.MetadataPolicy.REJECT

    def compose(self):
        metadata = self._hololink_channels[0].enumeration_metadata()
        interface = self._interface
        if interface is None:
            interface = metadata.get("interface")
        local_ip = metadata.get("interface_address")
        hsb_ip = metadata.get("client_ip_address")
        hsb_mac = metadata.get("mac_id")
        hsb_mac_bytes = list(bytes.fromhex(hsb_mac.replace(":", "")))
        logging.info("Using IMX274:")
        logging.info(f"  Interface: {interface}")
        logging.info(f"  Local IP:  {local_ip}")
        logging.info(f"  HSB IP:    {hsb_ip}")
        logging.info(f"  HSB MAC:   {hsb_mac}")

        # Holoviz is used to render the image(s).
        tensors = []
        view_width = 1.0 / float(len(self._cameras))
        for i in range(len(self._cameras)):
            view = holoscan.operators.HolovizOp.InputSpec.View()
            view.offset_x = view_width * i
            view.offset_y = 0
            view.width = view_width
            view.height = 1
            spec = holoscan.operators.HolovizOp.InputSpec(
                f"{i}", holoscan.operators.HolovizOp.InputType.COLOR
            )
            spec.views = [view]
            tensors.append(spec)

        width = self._width * len(self._cameras)
        visualizer = holoscan.operators.HolovizOp(
            self,
            name="holoviz",
            headless=self._headless,
            framebuffer_srgb=True,
            tensors=tensors,
            height=self._height,
            width=width,
        )

        # Create the capture/conversion pipelines for each sensor.
        for i in range(len(self._cameras)):
            if self._frame_limit:
                condition = holoscan.conditions.CountCondition(
                    self, name="count", count=self._frame_limit
                )
            else:
                condition = holoscan.conditions.BooleanCondition(
                    self, name="ok", enable_tick=True
                )

            self._cameras[i].set_mode(self._camera_mode)
            pixel_format = self._cameras[i].pixel_format()
            bayer_format = self._cameras[i].bayer_format()
            name = f"{i}"

            # Capture from Fusa.
            fusa_coe_capture = hololink_module.operators.FusaCoeCaptureOp(
                self,
                condition,
                name=f"fusa_coe_capture_{name}",
                interface=interface,
                mac_addr=hsb_mac_bytes,
                hololink_channel=self._hololink_channels[i],
                timeout=self._timeout,
                device=self._cameras[i],
            )
            self._cameras[i].configure_converter(fusa_coe_capture)

            # Convert packed RAW to 16-bit Bayer.
            packed_format_converter_pool = holoscan.resources.BlockMemoryPool(
                self,
                name=f"packed_format_converter_pool_{name}",
                # storage_type of 1 is device memory
                storage_type=1,
                block_size=self._cameras[i]._width
                * ctypes.sizeof(ctypes.c_uint16)
                * self._cameras[i]._height,
                num_blocks=4,
            )
            packed_format_converter = hololink_module.operators.PackedFormatConverterOp(
                self,
                name=f"packed_format_converter_{name}",
                allocator=packed_format_converter_pool,
            )
            fusa_coe_capture.configure_converter(packed_format_converter)

            # Perform basic ISP operations.
            image_processor = hololink_module.operators.ImageProcessorOp(
                self,
                name=f"image_processor_{name}",
                optical_black=0,
                bayer_format=bayer_format.value,
                pixel_format=pixel_format.value,
            )

            # Bayer demosaic to RGBA buffer.
            rgba_components_per_pixel = 4
            bayer_demosaic_pool = holoscan.resources.BlockMemoryPool(
                self,
                name=f"bayer_demosaic_pool_{name}",
                # storage_type of 1 is device memory
                storage_type=1,
                block_size=self._cameras[i]._width
                * rgba_components_per_pixel
                * ctypes.sizeof(ctypes.c_uint16)
                * self._cameras[i]._height,
                num_blocks=4,
            )
            bayer_demosaic = holoscan.operators.BayerDemosaicOp(
                self,
                name=f"bayer_demosaic_{name}",
                pool=bayer_demosaic_pool,
                generate_alpha=True,
                alpha_value=65535,
                bayer_grid_pos=bayer_format.value,
                interpolation_mode=0,
                out_tensor_name=name,
            )

            self.add_flow(
                fusa_coe_capture, packed_format_converter, {("output", "input")}
            )
            self.add_flow(
                packed_format_converter, image_processor, {("output", "input")}
            )
            self.add_flow(image_processor, bayer_demosaic, {("output", "receiver")})
            self.add_flow(bayer_demosaic, visualizer, {("transmitter", "receivers")})


def main():
    parser = argparse.ArgumentParser()
    modes = hololink_module.sensors.imx274.imx274_mode.Imx274_Mode
    mode_choices = [mode.value for mode in modes]
    parser.add_argument(
        "--camera-mode",
        type=int,
        choices=mode_choices,
        default=mode_choices[1],
        help=" ".join([f"{mode.value}:{mode.name}" for mode in modes]),
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Number of frames to capture",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1500,
        help="Capture request timeout",
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
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument(
        "--width",
        type=int,
        default=960,
        help="Window width (per sensor)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=540,
        help="Window height",
    )
    parser.add_argument(
        "--sensor",
        type=int,
        choices=[-1, 0, 1],
        default=-1,
        help="Sensor to use (-1 for stereo)",
    )
    parser.add_argument(
        "--skip-reset",
        action="store_true",
        help="Skip reset (only valid when using one sensor)",
    )
    parser.add_argument(
        "--interface",
        default=None,
        help="Ethernet interface",
    )
    args = parser.parse_args()

    hololink_module.logging_level(args.log_level)

    # Get a handle to the Hololink device
    channel_metadata = hololink_module.Enumerator.find_channel(channel_ip=args.hololink)

    channel_metadatas = []
    hololink_channels = []
    if args.sensor == -1:
        # Create separate channels for each sensor
        channel_metadatas.append(hololink_module.Metadata(channel_metadata))
        hololink_module.DataChannel.use_sensor(channel_metadatas[0], 0)
        hololink_channels.append(hololink_module.DataChannel(channel_metadatas[0]))

        channel_metadatas.append(hololink_module.Metadata(channel_metadata))
        hololink_module.DataChannel.use_sensor(channel_metadatas[1], 1)
        hololink_channels.append(hololink_module.DataChannel(channel_metadatas[1]))
    else:
        channel_metadatas.append(hololink_module.Metadata(channel_metadata))
        hololink_module.DataChannel.use_sensor(channel_metadatas[0], args.sensor)
        hololink_channels.append(hololink_module.DataChannel(channel_metadatas[0]))

    # Start the HSB
    hololink = hololink_channels[0].hololink()
    hololink.start()
    if args.sensor == -1 or not args.skip_reset:
        hololink.reset()

    # Get handles to the cameras
    camera_mode = hololink_module.sensors.imx274.imx274_mode.Imx274_Mode(
        args.camera_mode
    )
    cameras = []
    for i in range(len(hololink_channels)):
        camera = hololink_module.sensors.imx274.dual_imx274.Imx274Cam(
            hololink_channels[i], expander_configuration=i
        )
        if i == 0:
            camera.setup_clock()
        camera.configure(camera_mode)
        camera.set_digital_gain_reg(4)
        cameras.append(camera)

    # Set up the application
    application = App(
        hololink_channels,
        cameras,
        camera_mode,
        args.timeout,
        args.headless,
        args.width,
        args.height,
        args.frame_limit,
        args.interface,
    )

    # Run the application
    application.run()
    hololink.stop()


if __name__ == "__main__":
    main()
