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
import os

import holoscan
from tao_peoplenet import FormatInferenceInputOp, PostprocessorOp

import hololink as hololink_module


class HoloscanApplication(holoscan.core.Application):
    def __init__(
        self,
        camera_config,
        json_config,
        headless,
        fullscreen,
        frame_limit,
        engine,
    ):
        super().__init__()
        self._camera_config = camera_config
        self._json_config = json_config
        self._headless = headless
        self._fullscreen = fullscreen
        self._frame_limit = frame_limit
        self._engine = engine

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

        # Capture ISP-processed NV12 using SIPL.
        sipl_capture = hololink_module.operators.SIPLCaptureOp(
            self,
            condition,
            name="sipl_capture",
            camera_config=self._camera_config,
            json_config=self._json_config,
        )
        camera_info = sipl_capture.get_camera_info()
        assert len(camera_info) == 1, "Only single-camera configs are supported"

        # Holoviz will render both the NV12 image and the PeopleNet overlay.
        holoviz_args = self.kwargs("holoviz")
        holoviz_args["tensors"][0]["name"] = camera_info[0].output_name
        visualizer = holoscan.operators.HolovizOp(
            self,
            name="holoviz",
            fullscreen=self._fullscreen,
            headless=self._headless,
            framebuffer_srgb=True,
            **holoviz_args,
        )

        # Convert NV12 to RGB for inference.
        pool = holoscan.resources.UnboundedAllocator(self)
        yuv_to_rgba = holoscan.operators.FormatConverterOp(
            self,
            name="yuv_to_rgba",
            pool=pool,
            in_dtype="nv12",
            out_dtype="rgb888",
        )

        # Perform inference.
        preprocessor_args = self.kwargs("preprocessor")
        preprocessor = holoscan.operators.FormatConverterOp(
            self,
            name="preprocessor",
            pool=pool,
            **preprocessor_args,
        )
        format_input = FormatInferenceInputOp(
            self,
            name="transpose",
            pool=pool,
        )
        inference = holoscan.operators.InferenceOp(
            self,
            name="inference",
            allocator=pool,
            model_path_map={
                "face_detect": self._engine,
            },
            **self.kwargs("inference"),
        )
        postprocessor_args = self.kwargs("postprocessor")
        postprocessor_args["image_width"] = preprocessor_args["resize_width"]
        postprocessor_args["image_height"] = preprocessor_args["resize_height"]
        postprocessor = PostprocessorOp(
            self,
            name="postprocessor",
            allocator=pool,
            **postprocessor_args,
        )

        # Render the NV12 image.
        self.add_flow(sipl_capture, visualizer, {("output", "receivers")})

        # Process and render the PeopleNet overlay.
        self.add_flow(sipl_capture, yuv_to_rgba, {("output", "")})
        self.add_flow(yuv_to_rgba, preprocessor)
        self.add_flow(preprocessor, format_input)
        self.add_flow(format_input, inference, {("", "receivers")})
        self.add_flow(inference, postprocessor, {("transmitter", "in")})
        self.add_flow(postprocessor, visualizer, {("out", "receivers")})

        # Not using metadata
        self.enable_metadata(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list-configs", action="store_true", help="List available configs then exit"
    )
    parser.add_argument("--camera-config", default="", help="Camera config to use")
    parser.add_argument(
        "--json-config", default="", help="JSON configuration file to use"
    )
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
    default_configuration = os.path.join(
        os.path.dirname(__file__), "tao_peoplenet.yaml"
    )
    parser.add_argument(
        "--configuration", default=default_configuration, help="Configuration file"
    )
    default_engine = os.path.join(
        os.path.dirname(__file__), "resnet34_peoplenet_int8.onnx"
    )
    parser.add_argument(
        "--engine",
        default=default_engine,
        help="TRT engine model",
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
            args.headless,
            args.fullscreen,
            args.frame_limit,
            args.engine,
        )
        application.config(args.configuration)
        application.run()


if __name__ == "__main__":
    main()
