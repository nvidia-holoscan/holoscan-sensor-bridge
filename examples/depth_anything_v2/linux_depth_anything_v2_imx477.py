# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Depth Anything V2 running on an IMX477 attached to a Holoscan Sensor Bridge,
# using the Linux socket receiver (no ConnectX required).
#
# This script is adapted from https://github.com/nvidia-holoscan/holohub/tree/main/applications/depth_anything_v2/
# repository.
#
# Source model for this instance is obtained from https://aihub.qualcomm.com/models/depth_anything_v2?searchTerm=depth+anything
#
# *NOTE: The user is responsible for checking if the model license is suitable for the intended purpose.*
#
# See README.md for detailed information.
#
# Python port of this repo's own previous (HSDK4-era) version of this app to the HSDK5 Python
# bindings; there is no C++ sibling. The camera chain follows examples/linux_imx477_player.py,
# the inference side-chain examples/linux_tao_peoplenet_imx477.py:
#
#   LinuxReceiverOp -> CsiToBayerOp -> ImageProcessorOp -> BayerDemosaicOp -> ConvertToOp (>> 8)
#     -> ResizeOp -> ConvertToOp (float32) -> FormatInferenceInputOp
#        -> (a) "output" -> InferenceOp -> PostprocessorOp "input_depthmap"
#        -> (b) "image" -> PostprocessorOp "input_image"
#   PostprocessorOp -> HolovizOp "image" layer + "display_mode" text layer
#   ImageProcessorOp metadata -> FrameLimitOp
#
# Preprocessing: HSDK4's hololink ImageShiftToUint8Operator (x >> 8, to uint8) and
# FormatConverterOp are gone. ConvertToOp(uint8, alpha=1/256, rounding=truncate) is the shift,
# exactly: the scale is a power of two, so the product is exact and truncating it drops the low
# byte. ResizeOp and ConvertToOp(float32, alpha=1/255) are FormatConverterOp with depth.yaml's
# preprocessor section (518x518, float32, scale_min 0, scale_max 1). ResizeOp averages areas:
# FormatConverterOp's default resize was NPP's cubic, but the image shrinks several-fold, where
# cubic interpolation aliases.
#
# InferenceOp (holoscan_inference) replaces holoscan.operators.InferenceOp. Its ports are named
# from input_map/output_map: "preprocessed" in, "inference_output" out (depth.yaml's
# pre_processor_map/inference_map). `import holoscan_inference` is deferred to run(), as in
# examples/linux_tao_peoplenet_imx477.py: it needs TensorRT, and --help must work without it;
# OpenCV (PostprocessorOp) is likewise only needed to run. is_engine_path, which HSDK4 took from
# depth.yaml, follows --model-path instead: an .onnx model is built into a TensorRT engine on the
# first run (and cached), anything else is loaded as an engine.
#
# --configuration is gone: HSDK5 has no Application.config()/kwargs(), and its YAML path
# (holoscan.config + holoscan.params) only reaches schema-registered operator parameters, which
# the manually registered InferenceOp and HolovizOp rows are not (InferenceOp's own config_path
# takes a different schema from depth.yaml's). The values this app read from depth.yaml's
# preprocessor, inference and holoviz sections are set in code, unchanged.
#
# Pairing: HSDK4 ran the graph on one thread in lockstep, so PostprocessorOp always got a depth
# map together with the image it came from. HSDK5 runs the operators concurrently, and
# InferenceOp only queues GPU work, so one frame at a time goes to the inference, as in HSDK4's
# lockstep: FormatInferenceInputOp skips the frames that come while one is in flight, until
# PostprocessorOp has used its depth map (depth.InferencePacing). FormatInferenceInputOp also
# passes the frame it sends on to PostprocessorOp, which pairs it with its depth map by capture
# time (OnAligned): the Python operators pass each frame's capture time on, and InferenceOp
# stamps its output with its input's. PostprocessorOp gets no other frames: fed the camera's, it
# lost the one a depth map came from whenever newer frames got there first, as after a slow
# first inference. The edges into the inference branch and PostprocessorOp drop the oldest
# sample when full, so a stage that falls behind skips frames instead of backpressuring an SDK
# operator, which would fail the run.
#
# The Python operators write their CUDA outputs into runtime-pool tensors (output loans, filled
# through a CuPy view as in the SDK's holoviz_cupy_pattern example) and read their inputs with
# cupy.from_dlpack() on the activation's stream. Their loaned outputs are bound to the CUDA device
# like the SDK operators' outputs, as examples/max96716a_player.py binds RgbReshapeOp's.
#
# Display: HSDK4's HolovizOp mouse_button/cursor_pos/framebuffer_size callbacks, which cycled the
# display mode on a left click and moved the "interactive" split with the middle or right button,
# have no HSDK5 counterpart (see depth.py). --display-mode selects the mode instead (default
# "depth", HSDK4's initial one) and --split-ratio the "interactive" split (default 0.5, HSDK4's
# initial ratio). The mode's name still shows, on a text layer. Since the mode is fixed for the
# run, the window is sized to PostprocessorOp's image (518x518, or 1036x518 side by side) instead
# of depth.yaml's 518x518, into which HSDK4 squeezed the side-by-side image. The window keeps
# HSDK4's sRGB framebuffer. HolovizOp renders when the text arrives (layer trigger on the text
# layer): PostprocessorOp sends the image first and the text second, with one capture time, so
# every render draws an image with its own text. HolovizOp has its own partition, so a render
# waiting for the display does not hold up the inference. --headless renders into a CUDA image
# (HolovizOp's CUDA sink) that DiscardImageOp drops, as examples/linux_sub_frame_imx274_player.py
# does.
#
# --frame-limit: HSDK4's CountCondition on the receiver becomes FrameLimitOp, which counts the
# frames ImageProcessorOp processed and stops the session at the limit. It also drains
# ImageProcessorOp's metadata output, which must be consumed (HSDK4 disabled metadata).
#
# --log-level sets Python logging only: HSDK5's hololink bindings have no logging_level().
#
# Usage:
#   linux_depth_anything_v2_imx477.py [OPTIONS]
#
# Options:
#   --hololink IP         IP address of Hololink board (default: 192.168.0.2)
#   --cam {0,1}           Which camera to stream: 0 for j14, 1 for j17 (default: 0)
#   --resolution          4k or 1080p (default: 4k)
#   --exposure N          Configure exposure (default: 0x05)
#   --model-path PATH     TensorRT engine, or ONNX model (default: depth.engine.fp32)
#   --display-mode MODE   original, depth, side-by-side or interactive (default: depth)
#   --split-ratio R       Share of the image shown as original in interactive mode (default: 0.5)
#   --frame-limit N       Number of frames to capture (default: unlimited)
#   --headless            Render into a CUDA image instead of a window
#   --fullscreen          Run in fullscreen mode
#   --log-level N         Python logging level (default: 20 = INFO)

"""
Download model (outside docker) inside the directory of: holoscan-sensor-bridge/

    wget https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/depth_anything_v2/releases/v0.59.0/depth_anything_v2-onnx-float.zip
    unzip depth_anything_v2-onnx-float.zip -d ./examples/depth_anything_v2

Start holoscan docker and install opencv(this step would need to be done every time the docker is restarted)
    pip install opencv-python-headless

Navigate to examples/depth_anything_v2/depth_anything_v2-onnx-float/ and run the following(inside docker demo):
    trtexec --onnx=depth_anything_v2.onnx   --saveEngine=depth.engine.fp32
    cp depth.engine.fp32 ../../../
    cd ../../../

Run the demo itself:
    python3 examples/depth_anything_v2/linux_depth_anything_v2_imx477.py

Or skip trtexec and pass the ONNX model; InferenceOp then builds the TensorRT engine on the first
run (and caches it next to the model):
    python3 examples/depth_anything_v2/linux_depth_anything_v2_imx477.py \\
        --model-path examples/depth_anything_v2/depth_anything_v2-onnx-float/depth_anything_v2.onnx

Running needs TensorRT (holoscan_inference) and OpenCV; --help needs neither.
"""

import argparse
import logging
import signal
import sys

import capture_metadata
import cupy as cp
import hololink.sensors.imx477  # noqa: F401 -- makes hl.sensors.imx477.X resolve below
import holoscan as hs
import holoscan_vision
import holoscan_visualization as hz  # noqa: F401 -- registers "holoscan.visualization.HolovizOp"
from cuda.bindings import driver as cuda_driver
from depth import (
    DEFAULT_DISPLAY_MODE,
    DISPLAY_MODES,
    MAX_TEXT_GLYPHS,
    MODEL_INPUT_SIZE,
    InferencePacing,
    PostprocessorOp,
    derive_from,
    writable_view,
)

import hololink as hl

# (hololink/__init__.py doesn't import the sensors subpackage itself, and Python doesn't expose
# an unimported submodule as a parent-package attribute).

_FLOAT32 = hs.tensor.DLDataType(hs.tensor.DLDataTypeCode.DLFLOAT, 32, 1)

# One model input: a 1 x 3 x 518 x 518 float32 tensor.
_MODEL_INPUT_BYTES = 3 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 4


def _cuda_check(result):
    err = result[0]
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        _, description = cuda_driver.cuGetErrorString(err)
        raise RuntimeError(f"CUDA error: {description}")
    if len(result) == 2:
        return result[1]
    return result[1:]


class FormatInferenceInputOp(hs.Operator):
    """Operator to format input image for inference: the HWC float32 image -> the NCHW tensor
    InferenceOp takes. HSDK4 transposed on the host ("Numpy conversion and moveaxis is needed to
    avoid strange strides issue when doing inference"); here the transposed image is copied on
    the GPU into a contiguous runtime-pool tensor. A copy of each image sent goes to
    PostprocessorOp, on "image", to be paired with its depth map."""

    input = hs.Input("input", hs.schema.Image)
    output = hs.Output("output", hs.payload.Tensor)
    image = hs.Output("image", hs.schema.Image)

    def setup(self, spec):
        spec.input(self.input, queue_depth=1)
        spec.output(self.output, max_emits_per_compute=1).produces_tensor(
            hs.tensor.OutputSpec(
                storage="runtime_pool",
                memory_kind=hs.tensor.MemoryKind.CUDA_DEVICE,
                dtype=_FLOAT32,
                rank=4,
                strides=hs.tensor.TensorStridePolicy.CONTIGUOUS_ONLY,
            ),
            hs.tensor.byte_span(_MODEL_INPUT_BYTES),
        )
        spec.output(self.image, max_emits_per_compute=1).produces_tensor(
            hs.tensor.OutputSpec(
                storage="runtime_pool",
                memory_kind=hs.tensor.MemoryKind.CUDA_DEVICE,
                dtype=_FLOAT32,
                rank=3,
                strides=hs.tensor.TensorStridePolicy.CONTIGUOUS_ONLY,
            ),
            hs.tensor.byte_span(_MODEL_INPUT_BYTES),
            field="data",
        )

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, context):
        sample = self.input.receive()
        capture_time = sample.metadata.capture_timestamp_ns
        # One inference at a time; skip the frames that come meanwhile (see InferencePacing).
        if not InferencePacing.try_send(capture_time):
            return
        image = sample.data
        options = derive_from(sample.metadata)
        try:
            loan = self.output.allocate_tensor(
                shape=(1, 3, image.height, image.width),
                dtype=_FLOAT32,
                options=options,
            )
            image_loan = self.image.allocate_tensors(
                hs.schema.Image(
                    height=image.height,
                    width=image.width,
                    encoding=hs.schema.image.ImageEncoding.RGB32F,
                ),
                {
                    "data": hs.tensor.LoanRequest(
                        shape=(image.height, image.width, 3), dtype=_FLOAT32
                    )
                },
                options=options,
            )
        except hs.errors.ResourceExhaustedError:
            # Every buffer is still in use downstream: skip this frame. A loan already taken is
            # abandoned unpublished.
            InferencePacing.done(capture_time)
            return
        with cp.cuda.Device(image.data.device.device_id), context.stream():
            source = cp.from_dlpack(image.data)
            with loan:
                writable_view(loan.tensor, cp.float32)[0] = cp.moveaxis(source, 2, 0)
            with image_loan:
                writable_view(image_loan.tensors["data"], cp.float32)[...] = source


class FrameLimitOp(hs.Operator):
    """Drains ImageProcessorOp's metadata output, and stops the session once --frame-limit frames
    have gone through it (0: no limit)."""

    metadata = hs.Input("metadata", capture_metadata.CaptureMetadata)

    frame_limit = hs.Param(0)

    def setup(self, spec):
        spec.input(self.metadata, queue_depth=4)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.metadata))

    def start(self):
        self._frames_seen = 0

    def compute(self, context):
        self.metadata.receive_data()
        self._frames_seen += 1
        if 0 < self.frame_limit <= self._frames_seen:
            context.request_stop()


class DiscardImageOp(hs.Operator):
    """With --headless, HolovizOp renders into a CUDA image on its output instead of a window;
    the output must be consumed, and this drops each rendered frame."""

    input = hs.Input("input", hs.schema.Image)

    def setup(self, spec):
        spec.input(self.input, queue_depth=1)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, context):
        self.input.receive_data()


def run(
    hololink_ip,
    cam,
    resolution,
    exposure,
    model_path,
    display_mode,
    split_ratio,
    frame_limit,
    headless,
    fullscreen,
):
    # Deferred; see this file's header comment.
    import cv2  # noqa: F401 -- PostprocessorOp's colormap
    import holoscan_inference  # noqa: F401 -- registers "holoscan.inference.InferenceOp"

    cuda_device_ordinal = 0

    print("Initializing CUDA...")
    _cuda_check(cuda_driver.cuInit(0))
    cu_device = _cuda_check(cuda_driver.cuDeviceGet(cuda_device_ordinal))
    cu_context = _cuda_check(cuda_driver.cuDevicePrimaryCtxRetain(cu_device))
    try:
        cuda_driver.cuCtxSetCurrent(cu_context)

        print(f"Enumerating HSB board at {hololink_ip}...")
        metadata = hl.find_channel(hololink_ip)
        hl.use_sensor(metadata, cam)
        channel = hl.DataChannel(metadata)

        hololink_dev = channel.hololink()
        hololink_dev.start()
        try:
            hololink_dev.reset()

            camera = hl.sensors.imx477.Imx477(
                channel, camera_id=cam, resolution=resolution
            )
            # Configures the camera for the requested resolution, 60fps.
            camera.configure()
            # IMX477 analog gain settings function. Analog gain value range is 0-1023 in
            # decimal (10 bits). Users are free to experiment with the register values.
            camera.set_analog_gain(0x2FF)
            camera.set_exposure_reg(exposure)

            pixel_format = camera.pixel_format()
            bayer_format = camera.bayer_format()
            frame_width = camera._width
            frame_height = camera._height
            geometry = hl.resolve_generic_geometry_from_dimensions(
                int(pixel_format), frame_width, frame_height
            )

            # PostprocessorOp's image: the model-size image, or two of them side by side.
            image_width = MODEL_INPUT_SIZE
            if display_mode == "side-by-side":
                image_width *= 2
            image_height = MODEL_INPUT_SIZE

            # Build the graph.
            graph = hs.Graph("linux_depth_anything_v2_imx477")

            rx = graph.op(
                hl.operators.LinuxReceiverOp,
                "rx",
                data_channel=channel,
                frame_context=int(cu_context),
                device_start=camera.start,
                device_stop=camera.stop,
                frame_size=(
                    geometry["start_byte"]
                    + geometry["bytes_per_line"] * geometry["pixel_height"]
                    + geometry["trailing_bytes"]
                ),
            )

            csi = graph.op(
                hl.operators.CsiToBayerOp,
                "csi_to_bayer",
                cuda_device_ordinal=cuda_device_ordinal,
                geometry=geometry,
                bayer_format=bayer_format.name.lower(),
            )
            graph.add_flow(rx.tensor, csi.in_tensor, queue_depth=4)
            graph.add_flow(rx.metadata, csi.in_metadata, queue_depth=4)

            image_processor = graph.op(
                hl.operators.ImageProcessorOp,
                "image_processor",
                pixel_format=int(pixel_format),
                bayer_format=int(bayer_format),
                optical_black=100,
                cuda_device_ordinal=cuda_device_ordinal,
            )
            graph.add_flow(csi.tensor, image_processor.in_tensor, queue_depth=4)
            graph.add_flow(csi.metadata, image_processor.in_metadata, queue_depth=4)

            frame_limiter = graph.op(
                FrameLimitOp, "frame_limit", frame_limit=frame_limit or 0
            )
            # The counter must never hold ImageProcessorOp back: shed rather than fill up.
            graph.add_flow(
                image_processor.metadata,
                frame_limiter.metadata,
                queue_policy=hs.QueuePolicy.DROP_OLDEST,
                queue_depth=4,
            )

            demosaic = graph.op(
                holoscan_vision.operators.BayerDemosaicOp,
                "bayer_demosaic",
                bayer_grid_pos=bayer_format.name.lower(),
                generate_alpha=False,
            )
            graph.add_flow(image_processor.tensor, demosaic.input, queue_depth=4)

            # HSDK4's ImageShiftToUint8Operator(shift=8): x >> 8, to uint8.
            image_shift = graph.op(
                holoscan_vision.operators.ConvertToOp,
                "image_shift",
                output_type="uint8",
                alpha=1.0 / 256.0,
                beta=0.0,
                rounding="truncate",
            )
            graph.add_flow(demosaic.output, image_shift.input, queue_depth=4)

            # HSDK4's FormatConverterOp with depth.yaml's preprocessor section: resize to the
            # model's 518x518, then uint8 -> float32 in [0, 1].
            resize = graph.op(
                holoscan_vision.operators.ResizeOp,
                "resize",
                width=MODEL_INPUT_SIZE,
                height=MODEL_INPUT_SIZE,
                max_batch_size=1,
                # The image shrinks several-fold, where cubic interpolation aliases.
                interpolation="area",
            )
            graph.add_flow(image_shift.output, resize.input, queue_depth=4)

            convert = graph.op(
                holoscan_vision.operators.ConvertToOp,
                "convert",
                output_type="float32",
                alpha=1.0 / 255.0,
                beta=0.0,
            )
            graph.add_flow(resize.output, convert.input, queue_depth=4)

            # Inference branch: a frame that arrives while the inference is busy replaces the one
            # waiting.
            format_input = graph.op(FormatInferenceInputOp, "transpose")
            graph.add_flow(
                convert.output,
                format_input.input,
                queue_policy=hs.QueuePolicy.DROP_OLDEST,
                queue_depth=1,
            )

            inference = graph.op(
                "holoscan.inference.InferenceOp",
                "inference",
                model_path_map={"depth": model_path},
                input_map={"depth": ["preprocessed"]},
                output_map={"depth": ["inference_output"]},
                backend="trt",
                input_on_cuda=True,
                is_engine_path=not model_path.lower().endswith(".onnx"),
            )
            graph.add_flow(
                format_input.output,
                inference.preprocessed,
                queue_policy=hs.QueuePolicy.DROP_OLDEST,
                queue_depth=1,
            )

            postprocessor = graph.op(
                PostprocessorOp,
                "postprocessor",
                display_mode=display_mode,
                split_ratio=split_ratio,
            )
            graph.add_flow(
                inference.inference_output,
                postprocessor.input_depthmap,
                queue_policy=hs.QueuePolicy.DROP_OLDEST,
                queue_depth=1,
            )
            # The frame sent to the inference also goes to PostprocessorOp, which pairs it with
            # its depth map (two queued after InferencePacing released a frame).
            graph.add_flow(
                format_input.image,
                postprocessor.input_image,
                queue_policy=hs.QueuePolicy.DROP_OLDEST,
                queue_depth=2,
            )

            if headless:
                # Renders into a CUDA image on HolovizOp's output (consumed by DiscardImageOp). The
                # CUDA sink has no framebuffer_srgb: its sRGB format encodes on write instead.
                sink = {
                    "kind": "cuda",
                    "width": image_width,
                    "height": image_height,
                    "format": "rgba8_srgb",
                }
            else:
                sink = {
                    "kind": "window",
                    "title": "DepthAnything v2",
                    "width": image_width,
                    "height": image_height,
                    "fullscreen": fullscreen,
                    "close_on_escape": True,
                    # As HSDK4's HolovizOp(framebuffer_srgb=True): encode to sRGB on write.
                    "framebuffer_srgb": True,
                }
            viz = graph.op(
                "holoscan.visualization.HolovizOp",
                "holoviz",
                layers=[
                    {
                        "port_name": "image",
                        "kind": "image",
                        "format": "rgb8_unorm",
                        "max_width": image_width,
                        "max_height": image_height,
                    },
                    {
                        "port_name": "display_mode",
                        "kind": "text",
                        "max_glyphs": MAX_TEXT_GLYPHS,
                        "normalized_coordinates": True,
                    },
                ],
                sink=sink,
                # One render per PostprocessorOp output: the text, sent after its image.
                trigger={"kind": "layer", "layer": 1},
            )
            # Equal drop-oldest queues keep each render's image and text together, even when
            # HolovizOp falls behind.
            graph.add_flow(
                postprocessor.output_image,
                viz.image,
                queue_policy=hs.QueuePolicy.DROP_OLDEST,
                queue_depth=2,
            )
            graph.add_flow(
                postprocessor.output_text,
                viz.display_mode,
                queue_policy=hs.QueuePolicy.DROP_OLDEST,
                queue_depth=2,
            )
            if headless:
                discard = graph.op(DiscardImageOp, "discard")
                graph.add_flow(
                    viz.output,
                    discard.input,
                    queue_policy=hs.QueuePolicy.DROP_OLDEST,
                    queue_depth=1,
                )

            graph.partition("holoviz").add(viz)
            graph.auto_partition_remaining()

            clock_key = graph.add_clock(hs.RealtimeClock, "clock")
            graph.set_default_clock(clock_key)

            options = hs.CompileOptions()
            tensor_outputs = [
                ("rx", "tensor"),
                ("csi_to_bayer", "tensor"),
                ("image_processor", "tensor"),
                ("bayer_demosaic", "output"),
                ("image_shift", "output"),
                ("resize", "output"),
                ("convert", "output"),
                ("transpose", "output"),
                ("transpose", "image"),
                ("inference", "inference_output"),
                ("postprocessor", "output_image"),
            ]
            if headless:
                tensor_outputs.append(("holoviz", "output"))
            for operator_path, output_port in tensor_outputs:
                options.deployment.bind_tensor_output_device(
                    operator_path=operator_path,
                    output_port=output_port,
                    device=hs.execution.DeviceId(cuda_device_ordinal),
                )

            plan = hs.compile(graph, options)
            if not plan.ok:
                print("Graph compilation failed:", file=sys.stderr)
                for diag in plan.diagnostics:
                    print(f"  {diag}", file=sys.stderr)
                return 1

            print(
                f"Capturing {resolution} {frame_width}x{frame_height}, display mode "
                f"{display_mode} -- running (Ctrl+C to stop)..."
            )

            with hs.run_async(plan) as session:
                try:
                    session.wait()
                except KeyboardInterrupt:
                    # Ctrl+C or SIGTERM: leaving the block stops the session cooperatively.
                    pass
            if session.diagnostics:
                print(session.summary(), file=sys.stderr)
            failed = session.termination == hs.RunTermination.FAILED
        finally:
            hololink_dev.stop()
    finally:
        _cuda_check(cuda_driver.cuDevicePrimaryCtxRelease(cu_device))

    if failed:
        return 1
    print("Done.")
    return 0


def main():
    # Like Ctrl+C, SIGTERM (kill, docker stop) raises KeyboardInterrupt, so the session stops
    # cooperatively and the board is stopped on the way out.
    signal.signal(signal.SIGTERM, signal.default_int_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Render into a CUDA image instead of a window",
    )
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
    parser.add_argument(
        "--model-path",
        default="depth.engine.fp32",
        help="Depth Anything V2 TensorRT engine, or ONNX model (built into an engine on the "
        "first run)",
    )
    parser.add_argument(
        "--display-mode",
        choices=DISPLAY_MODES,
        default=DEFAULT_DISPLAY_MODE,
        help="What to display: the camera image, its depth map, both side by side, or the "
        "image left of --split-ratio and the depth map right of it (interactive)",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.5,
        help="In interactive mode, how much of the original video to show (0 to 1)",
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
        "--resolution",
        default="4k",
        help="4k or 1080p",
    )
    parser.add_argument(
        "--exposure",
        type=int,
        default=0x05,
        help="Configure exposure.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.split_ratio <= 1.0:
        parser.error(f"--split-ratio must be between 0 and 1, got {args.split_ratio}")

    logging.basicConfig(level=args.log_level)

    return run(
        args.hololink,
        args.cam,
        args.resolution,
        args.exposure,
        args.model_path,
        args.display_mode,
        args.split_ratio,
        args.frame_limit,
        args.headless,
        args.fullscreen,
    )


if __name__ == "__main__":
    sys.exit(main())
