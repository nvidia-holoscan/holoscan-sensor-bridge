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

import argparse
import ctypes
import logging
import os

import cuda.bindings.driver as cuda
import cupy as cp
import holoscan
import holoscan as hs
from body_pose_estimation import FormatInferenceInputOp, PostprocessorOp
from holoscan.core import Operator, OperatorSpec
from holoscan.gxf import Entity

import hololink as hololink_module

POSE_TENSOR_NAMES = [
    "boxes",
    "noses",
    "left_eyes",
    "right_eyes",
    "left_ears",
    "right_ears",
    "left_shoulders",
    "right_shoulders",
    "left_elbows",
    "right_elbows",
    "left_wrists",
    "right_wrists",
    "left_hips",
    "right_hips",
    "left_knees",
    "right_knees",
    "left_ankles",
    "right_ankles",
    "segments",
]


class RenameTensorsOp(Operator):
    """Rebroadcasts an incoming message with each tensor renamed per a source→destination map."""

    def __init__(self, *args, rename_map, **kwargs):
        self._rename_map = rename_map
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("in")
        out_message = Entity(context)
        for src, dst in self._rename_map.items():
            tensor = cp.asarray(in_message.get(src))
            out_message.add(hs.as_tensor(tensor), dst)
        op_output.emit(out_message, "out")


class HoloscanApplication(holoscan.core.Application):
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
        engine,
        window_height,
        window_width,
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
        self._engine = engine
        self._window_height = window_height
        self._window_width = window_width

    def _build_camera_pipeline(
        self,
        side,
        condition,
        hololink_channel,
        camera,
        csi_to_bayer_pool,
        bayer_pool,
        pool,
    ):
        csi_to_bayer_operator = hololink_module.operators.CsiToBayerOp(
            self,
            name=f"csi_to_bayer_{side}",
            allocator=csi_to_bayer_pool,
            cuda_device_ordinal=self._cuda_device_ordinal,
        )
        camera.configure_converter(csi_to_bayer_operator)

        frame_size = csi_to_bayer_operator.get_csi_length()
        receiver_operator = hololink_module.operators.LinuxReceiverOperator(
            self,
            condition,
            name=f"receiver_{side}",
            frame_size=frame_size,
            frame_context=self._cuda_context,
            hololink_channel=hololink_channel,
            device=camera,
        )

        bayer_format = camera.bayer_format()
        pixel_format = camera.pixel_format()
        image_processor_operator = hololink_module.operators.ImageProcessorOp(
            self,
            name=f"image_processor_{side}",
            optical_black=100,
            bayer_format=bayer_format.value,
            pixel_format=pixel_format.value,
        )

        demosaic = holoscan.operators.BayerDemosaicOp(
            self,
            name=f"demosaic_{side}",
            pool=bayer_pool,
            generate_alpha=False,
            bayer_grid_pos=bayer_format.value,
            interpolation_mode=0,
        )

        image_shift = hololink_module.operators.ImageShiftToUint8Operator(
            self, name=f"image_shift_{side}", shift=8
        )

        image_renamer = RenameTensorsOp(
            self,
            name=f"image_renamer_{side}",
            rename_map={"": f"{side}_image"},
        )

        preprocessor_args = self.kwargs("preprocessor")
        preprocessor = holoscan.operators.FormatConverterOp(
            self,
            name=f"preprocessor_{side}",
            pool=pool,
            **preprocessor_args,
        )
        format_input = FormatInferenceInputOp(
            self,
            name=f"transpose_{side}",
            pool=pool,
        )
        inference_kwargs = self.kwargs("inference")

        model_name = f"yolo_pose_{side}"
        inference_kwargs["pre_processor_map"] = {model_name: ["preprocessed"]}
        inference_kwargs["inference_map"] = {model_name: ["inference_output"]}
        inference = holoscan.operators.InferenceOp(
            self,
            name=f"inference_{side}",
            allocator=pool,
            model_path_map={model_name: self._engine},
            **inference_kwargs,
        )
        postprocessor_args = self.kwargs("postprocessor")
        postprocessor_args["image_width"] = preprocessor_args["resize_width"]
        postprocessor_args["image_height"] = preprocessor_args["resize_height"]
        postprocessor = PostprocessorOp(
            self,
            name=f"postprocessor_{side}",
            allocator=pool,
            **postprocessor_args,
        )
        pose_renamer = RenameTensorsOp(
            self,
            name=f"pose_renamer_{side}",
            rename_map={t: f"{side}_{t}" for t in POSE_TENSOR_NAMES},
        )

        self.add_flow(receiver_operator, csi_to_bayer_operator, {("output", "input")})
        self.add_flow(
            csi_to_bayer_operator, image_processor_operator, {("output", "input")}
        )
        self.add_flow(image_processor_operator, demosaic, {("output", "receiver")})
        self.add_flow(demosaic, image_shift, {("transmitter", "input")})
        self.add_flow(image_shift, image_renamer, {("output", "in")})
        self.add_flow(image_shift, preprocessor, {("output", "")})
        self.add_flow(preprocessor, format_input)
        self.add_flow(format_input, inference, {("", "receivers")})
        self.add_flow(inference, postprocessor, {("transmitter", "in")})
        self.add_flow(postprocessor, pose_renamer, {("out", "in")})

        return image_renamer, pose_renamer

    def compose(self):
        logging.info("compose")
        if self._frame_limit:
            condition_left = holoscan.conditions.CountCondition(
                self, name="count_left", count=self._frame_limit
            )
            condition_right = holoscan.conditions.CountCondition(
                self, name="count_right", count=self._frame_limit
            )
        else:
            condition_left = holoscan.conditions.BooleanCondition(
                self, name="ok_left", enable_tick=True
            )
            condition_right = holoscan.conditions.BooleanCondition(
                self, name="ok_right", enable_tick=True
            )

        csi_to_bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="csi_to_bayer_pool",
            storage_type=1,
            block_size=self._camera_left._width
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera_left._height,
            num_blocks=8,
        )
        rgb_components_per_pixel = 3
        bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="bayer_pool",
            storage_type=1,
            block_size=self._camera_left._width
            * rgb_components_per_pixel
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera_left._height,
            num_blocks=8,
        )
        pool = holoscan.resources.UnboundedAllocator(self)

        image_renamer_left, pose_renamer_left = self._build_camera_pipeline(
            "left",
            condition_left,
            self._hololink_channel_left,
            self._camera_left,
            csi_to_bayer_pool,
            bayer_pool,
            pool,
        )
        image_renamer_right, pose_renamer_right = self._build_camera_pipeline(
            "right",
            condition_right,
            self._hololink_channel_right,
            self._camera_right,
            csi_to_bayer_pool,
            bayer_pool,
            pool,
        )

        visualizer = holoscan.operators.HolovizOp(
            self,
            name="holoviz",
            fullscreen=self._fullscreen,
            headless=self._headless,
            framebuffer_srgb=True,
            tensors=self._build_holoviz_specs(),
            height=self._window_height,
            width=self._window_width,
        )

        self.add_flow(image_renamer_left, visualizer, {("out", "receivers")})
        self.add_flow(image_renamer_right, visualizer, {("out", "receivers")})
        self.add_flow(pose_renamer_left, visualizer, {("out", "receivers")})
        self.add_flow(pose_renamer_right, visualizer, {("out", "receivers")})

        self.enable_metadata(False)

    def _build_holoviz_specs(self):
        # Pull the overlay specs (colors, line widths, point sizes) from the shared
        # body_pose_estimation.yaml config and duplicate them per side with prefixed
        # tensor names. The yaml entry with name "" (the image) is replaced with
        # explicit left_image / right_image specs.
        yaml_specs = self.kwargs("holoviz").get("tensors", [])

        def make_view(offset_x):
            view = holoscan.operators.HolovizOp.InputSpec.View()
            view.offset_x = offset_x
            view.offset_y = 0
            view.width = 0.5
            view.height = 1.0
            return view

        specs = []
        for side, offset_x in (("left", 0.0), ("right", 0.5)):
            view = make_view(offset_x)

            image_spec = holoscan.operators.HolovizOp.InputSpec(
                f"{side}_image", holoscan.operators.HolovizOp.InputType.COLOR
            )
            image_spec.views = [view]
            specs.append(image_spec)

            for entry in yaml_specs:
                name = entry.get("name", "")
                if name == "":
                    continue
                type_str = entry.get("type", "color")
                input_type = {
                    "color": holoscan.operators.HolovizOp.InputType.COLOR,
                    "rectangles": holoscan.operators.HolovizOp.InputType.RECTANGLES,
                    "points": holoscan.operators.HolovizOp.InputType.POINTS,
                    "lines": holoscan.operators.HolovizOp.InputType.LINES,
                }[type_str]
                spec = holoscan.operators.HolovizOp.InputSpec(
                    f"{side}_{name}", input_type
                )
                if "color" in entry:
                    spec.color = entry["color"]
                if "line_width" in entry:
                    spec.line_width = entry["line_width"]
                if "point_size" in entry:
                    spec.point_size = entry["point_size"]
                if "opacity" in entry:
                    spec.opacity = entry["opacity"]
                spec.views = [view]
                specs.append(spec)

        return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument(
        "--fullscreen", action="store_true", help="Run in fullscreen mode"
    )
    parser.add_argument(
        "--hololink",
        default="192.168.0.2",
        help="IP address of Hololink board",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Exit after receiving this many frames",
    )
    default_configuration = os.path.join(
        os.path.dirname(__file__), "body_pose_estimation.yaml"
    )
    parser.add_argument(
        "--configuration", default=default_configuration, help="Configuration file"
    )
    default_engine = os.path.join(os.path.dirname(__file__), "yolov8n-pose.engine.fp32")
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
    parser.add_argument(
        "--resolution",
        default="1080p",
        help="4k or 1080p",
    )
    parser.add_argument(
        "--exposure",
        type=int,
        default=0x05,
        help="Configure exposure.",
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
    args = parser.parse_args()
    hololink_module.logging_level(args.log_level)
    logging.info("Initializing.")

    (cu_result,) = cuda.cuInit(0)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_device_ordinal = 0
    cu_result, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_result, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS

    channel_metadata = hololink_module.Enumerator.find_channel(channel_ip=args.hololink)

    channel_metadata_left = hololink_module.Metadata(channel_metadata)
    hololink_module.DataChannel.use_sensor(channel_metadata_left, 0)
    hololink_channel_left = hololink_module.DataChannel(channel_metadata_left)
    camera_left = hololink_module.sensors.imx477.Imx477(
        hololink_channel_left, camera_id=0, resolution=args.resolution
    )

    channel_metadata_right = hololink_module.Metadata(channel_metadata)
    hololink_module.DataChannel.use_sensor(channel_metadata_right, 1)
    hololink_channel_right = hololink_module.DataChannel(channel_metadata_right)
    camera_right = hololink_module.sensors.imx477.Imx477(
        hololink_channel_right, camera_id=1, resolution=args.resolution
    )

    application = HoloscanApplication(
        args.headless,
        args.fullscreen,
        cu_context,
        cu_device_ordinal,
        hololink_channel_left,
        camera_left,
        hololink_channel_right,
        camera_right,
        args.frame_limit,
        args.engine,
        args.window_height,
        args.window_width,
    )
    application.config(args.configuration)

    hololink = hololink_channel_left.hololink()
    hololink.start()
    try:
        hololink.reset()
        camera_left.configure()
        camera_right.configure()

        camera_left.set_analog_gain(0x2FF)
        camera_left.set_exposure_reg(args.exposure)
        camera_right.set_analog_gain(0x2FF)
        camera_right.set_exposure_reg(args.exposure)

        application.run()
    finally:
        hololink.stop()

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS


if __name__ == "__main__":
    main()
