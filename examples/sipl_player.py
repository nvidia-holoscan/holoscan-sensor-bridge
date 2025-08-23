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
import logging

import holoscan

import hololink as hololink_module


class HoloscanApplication(holoscan.core.Application):
    def __init__(
        self,
        camera_config,
        json_config,
        raw_output,
        headless,
        fullscreen,
        frame_limit,
    ):
        logging.info("__init__")
        super().__init__()
        self._camera_config = camera_config
        self._json_config = json_config
        self._raw_output = raw_output
        self._headless = headless
        self._fullscreen = fullscreen
        self._frame_limit = frame_limit

    def compose(self):
        if self._frame_limit:
            self._count = holoscan.conditions.CountCondition(
                self,
                name="count",
                count=self._frame_limit,
            )
            condition = self._count
        else:
            self._ok = holoscan.conditions.BooleanCondition(
                self, name="ok", enable_tick=True
            )
            condition = self._ok

        # SIPL capture operator captures either the RAW or ISP-Processed image via NvSIPL.
        sipl_capture = hololink_module.operators.SIPLCaptureOp(
            self,
            condition,
            name="sipl_capture",
            camera_config=self._camera_config,
            json_config=self._json_config,
            raw_output=self._raw_output,
        )
        camera_info = sipl_capture.get_camera_info()

        # Holoviz is used to render the image(s).
        view_width = 1.0 / len(camera_info)
        specs = []
        output_index = 0
        for info in camera_info:
            view = holoscan.operators.HolovizOp.InputSpec.View()
            view.offset_x = view_width * output_index
            view.offset_y = 0
            view.width = view_width
            view.height = 1
            output_index = output_index + 1

            spec = holoscan.operators.HolovizOp.InputSpec(
                info.output_name, holoscan.operators.HolovizOp.InputType.COLOR
            )
            spec.views = [view]
            specs = specs + [spec]

        visualizer = holoscan.operators.HolovizOp(
            self,
            name="holoviz",
            fullscreen=self._fullscreen,
            headless=self._headless,
            tensors=specs,
        )

        if not self._raw_output:
            # When capturing ISP-processed images from SIPL, output directly
            # from SIPL to the vizualizer.
            self.add_flow(sipl_capture, visualizer, {("output", "receivers")})
        else:
            # If capturing RAW, we need to do the following between the SIPL capture and Holoviz:
            #   1) Convert the packed RAW to 16-bit Bayer (PackedFormatConverterOp)
            #   2) Perform basic ISP operations (ImageProcessorOp)
            #   3) Demosaic the Bayer to RGB.
            # These operators only process one buffer each, so each camera needs separate instances.
            for info in camera_info:
                # Convert packed RAW to 16-bit Bayer.
                converter_pool = holoscan.resources.BlockMemoryPool(
                    self,
                    name=f"converter_pool_{info.output_name}",
                    storage_type=1,  # Device memory
                    block_size=(info.width * info.height * 2),  # 16-bit Bayer
                    num_blocks=2,
                )
                format_converter = hololink_module.operators.PackedFormatConverterOp(
                    self,
                    name=f"packed_format_converter_{info.output_name}",
                    allocator=converter_pool,
                    in_tensor_name=info.output_name,
                    out_tensor_name=info.output_name,
                )
                format_converter.configure(
                    info.offset,
                    info.bytes_per_line,
                    info.width,
                    info.height,
                    info.pixel_format,
                )

                # Perform basic ISP operations.
                image_processor = hololink_module.operators.ImageProcessorOp(
                    self,
                    name=f"image_processor_{info.output_name}",
                    optical_black=0,
                    bayer_format=info.bayer_format,
                    pixel_format=info.pixel_format,
                )

                # Bayer demosaic to RGBA buffer.
                demosaic_pool = holoscan.resources.BlockMemoryPool(
                    self,
                    name=f"demosaic_pool_{info.output_name}",
                    storage_type=1,  # Device memory
                    block_size=(info.width * info.height * 2 * 4),  # 16-bit RGBA
                    num_blocks=2,
                )
                demosaic = holoscan.operators.BayerDemosaicOp(
                    self,
                    name=f"bayer_demosaic_{info.output_name}",
                    pool=demosaic_pool,
                    generate_alpha=True,
                    alpha_value=65535,
                    bayer_grid_pos=info.bayer_format,
                    in_tensor_name=info.output_name,
                    out_tensor_name=info.output_name,
                )

                # Define the application flow.
                self.add_flow(sipl_capture, format_converter, {("output", "input")})
                self.add_flow(format_converter, image_processor, {("output", "input")})
                self.add_flow(image_processor, demosaic, {("output", "receiver")})
                self.add_flow(demosaic, visualizer, {("transmitter", "receivers")})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list-configs", action="store_true", help="List available configs then exit"
    )
    parser.add_argument("--camera-config", default="", help="Camera config to use")
    parser.add_argument(
        "--json-config", default="", help="JSON configuration file to use"
    )
    parser.add_argument("--raw", action="store_true", help="Capture RAW buffers")
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
    args = parser.parse_args()
    hololink_module.logging_level(args.log_level)

    if args.list_configs:
        hololink_module.operators.SIPLCaptureOp.list_available_configs(args.json_config)
    else:
        application = HoloscanApplication(
            args.camera_config,
            args.json_config,
            args.raw,
            args.headless,
            args.fullscreen,
            args.frame_limit,
        )
        application.run()


if __name__ == "__main__":
    main()
