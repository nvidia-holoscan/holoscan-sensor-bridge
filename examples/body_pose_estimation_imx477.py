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

# Single IMX477 sensor via RoceReceiverOp, running a YOLOv8-pose body-pose-estimation model and
# overlaying keypoints/skeleton on the live video in Holoviz. Python port of this repo's own
# previous (HSDK4-era) Python version of this app, via the HSDK5 Python bindings -- there is no
# examples/body_pose_estimation_imx477.cpp sibling. This is the RoCE twin of
# examples/linux_body_pose_estimation_imx477.py (see that file's own header comment for the full
# rationale behind the inference side-chain and the keypoint-consolidation design -- InferenceOp/
# ResizeOp/ConvertToOp, the deferred holoscan_inference import, the geometry-layer color/opacity
# loss, and why 19 old overlay tensors became 2); only the receiver and bring-up differ here,
# matching every other RoCE/Linux IMX477 pair ported this session.
#
# Bring-up here is the simple pattern also used by tao_peoplenet_imx477.py (RoCE): the old app
# calls Enumerator.find_channel(channel_ip=args.hololink) directly with no hl.use_sensor() demux
# -- preserved exactly, not force-fit to the use_sensor pattern the Linux twin uses.
#
#   RoceReceiverOp -> CsiToBayerOp -> ImageProcessorOp -> BayerDemosaicOp -> ImageShiftToUint8Op
#     -> (a) HolovizOp "video" image layer
#     -> (b) ResizeOp -> ConvertToOp -> FormatInferenceInputOp -> InferenceOp -> PostprocessorOp
#          -> HolovizOp "keypoints"/"segments" geometry layers
#
# PostprocessorOp's math (keypoint extraction, skeleton-segment construction, NMS) is ported
# unchanged from this repo's own HSDK4-era body_pose_estimation.py PostprocessorOp, but its
# OUTPUT shape is not: the old operator emitted 19 separate named tensors (one Holoviz InputSpec
# each -- boxes, 17 individual keypoint types, segments). HolovizOp's new Graph-API operator caps
# layer count at kMaxLayerPorts=8 total (confirmed directly from its own C++ header/source), so
# 19 separate overlay layers is not just impractical but outright rejected at compile time. This
# port consolidates all 12 of the old app's actually-*visible* keypoint types (shoulders through
# ankles; confirmed from the old app's own YAML, which sets opacity=0 -- fully invisible -- for
# "boxes" and the 5 face-area keypoints: noses/eyes/ears) into one combined "keypoints" geometry
# layer (cross_list topology, one cross marker per point). "boxes" is dropped as an output
# entirely, for the same opacity=0 reason -- but its *computation* is kept, since it still drives
# NMS's box-overlap math, independent of whether a box is ever rendered. Face-area keypoints are
# likewise still computed (needed as skeleton-segment endpoints -- see the segments list below,
# which does connect to ears/nose) even though they're never emitted as their own points. Total
# layers: video + keypoints + segments = 3, comfortably under the cap.
#
# Real, flagged behavior reduction (same root cause as linux_tao_peoplenet_imx477.py, compounded
# here): HolovizOp's new "geometry" layer has no per-primitive OR per-layer opacity/color field
# at all. The old app relied on exactly this (opacity=0 on boxes/face-keypoints, 0.5 on
# everything else) to selectively hide overlays -- consolidation above preserves the old app's
# real *visible* output by omitting what was already invisible, but any future attempt to make
# individual keypoint types independently visible/hidden again would need per-layer opacity this
# SDK version's HolovizOp doesn't expose.
#
# Usage:
#   body_pose_estimation_imx477.py [OPTIONS]
#
# Options:
#   --cam {0,1}       Which camera to stream: 0 for j14, 1 for j17 (default: 0)
#   --resolution      4k or 1080p (default: 4k)
#   --exposure N      Configure exposure (default: 0x05)
#   --ibv-name NAME   IBV device to use (default: the device on the board's network interface)
#   --ibv-port N      Port number of IBV device (default: 1)
#   --engine PATH     YOLOv8-pose TRT engine path (default: examples/yolov8n-pose.engine.fp32)
#   --frame-limit N   Number of frames to capture (default: unlimited)
#   --headless        Run without opening a window (still renders, presents nowhere)
#   --fullscreen      Run in fullscreen mode
#   --log-level N     Python logging level (default: 20 = INFO)

import argparse
import logging
import os
import signal
import sys

import cupy as cp
import hololink.sensors.imx477  # noqa: F401 -- makes hl.sensors.imx477.X resolve below
import holoscan as hs
import holoscan_vision
import holoscan_visualization as hz  # noqa: F401 -- registers "holoscan.visualization.HolovizOp"
import numpy as np
from cuda.bindings import driver as cuda_driver

import hololink as hl

# (hololink/__init__.py doesn't import the sensors subpackage itself, and Python doesn't expose
# an unimported submodule as a parent-package attribute).

# YOLOv8-pose's raw output channel layout: cx, cy, w, h, score, then 17 keypoints x (x, y,
# visibility) -- only x, y are used below, visibility is ignored (matches the old app exactly).
_KEYPOINT_SLICES = {
    "nose": slice(5, 7),
    "left_eye": slice(8, 10),
    "right_eye": slice(11, 13),
    "left_ear": slice(14, 16),
    "right_ear": slice(17, 19),
    "left_shoulder": slice(20, 22),
    "right_shoulder": slice(23, 25),
    "left_elbow": slice(26, 28),
    "right_elbow": slice(29, 31),
    "left_wrist": slice(32, 34),
    "right_wrist": slice(35, 37),
    "left_hip": slice(38, 40),
    "right_hip": slice(41, 43),
    "left_knee": slice(44, 46),
    "right_knee": slice(47, 49),
    "left_ankle": slice(50, 52),
    "right_ankle": slice(53, 55),
}

# The 12 keypoint types the old app's own YAML actually renders (opacity 0.5); face-area
# keypoints (nose/eyes/ears, opacity 0 in the old YAML) are computed for segment endpoints below
# but never emitted as their own points -- see this file's header comment.
_VISIBLE_KEYPOINT_NAMES = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# Skeleton segments, ported unchanged from the old app's own list (18 pairs; "neck" is
# synthesized as the left/right shoulder midpoint, matching the old app exactly).
_SEGMENT_PAIRS = [
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_shoulder", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_hip", "right_hip"),
    ("left_ear", "neck"),
    ("right_ear", "neck"),
]

# Normalized-coordinate cross-marker size for keypoints (HolovizOp's cross_list topology wants
# cx, cy, size per primitive). The old app's point_size=10 was in a pixel-ish unit with no
# documented HSDK5 equivalent coordinate space; this is a reasonable visible-marker judgment
# call, not a value derived from the model or the old app.
_KEYPOINT_MARKER_SIZE = 0.01

_MAX_OVERLAY_PERSONS = (
    16  # conservative headroom for simultaneous detections in one frame.
)


def _cuda_check(result):
    err = result[0]
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        _, description = cuda_driver.cuGetErrorString(err)
        raise RuntimeError(f"CUDA error: {description}")
    if len(result) == 2:
        return result[1]
    return result[1:]


def import_tensor(tensor, stream):
    """Borrow a read-only GPU view; callers finish reads before compute returns."""
    return cp.from_dlpack(
        tensor.__dlpack__(stream=stream.ptr, max_version=(1, 0), copy=False)
    )


def emit_host_tensor(port, values, options):
    values = np.ascontiguousarray(values, dtype=np.float32).ravel()
    port.emit(hs.Tensor.as_tensor(values), options=options)


def wait_for_completion(session):
    """Wait for receiver completion or runtime failure, and handle Ctrl+C."""
    try:
        session.wait()
    except KeyboardInterrupt:
        session.request_stop()
        session.wait()
    return any(
        diag.severity == hs.diagnostics.DiagnosticSeverity.ERROR
        for diag in session.diagnostics
    )


class ImageShiftToUint8Op(hs.Operator):
    """Right-shifts a 16-bit image down to 8 bits per channel (shift=8 keeps the high byte).
    Uses an SDK-owned output buffer so native CUDA consumers can track its lifetime."""

    input = hs.Input("input", hs.schema.Image)
    output = hs.Output("output", hs.schema.Image)

    shift = hs.Param(0)

    def setup(self, spec):
        spec.input(self.input, queue_depth=4)
        spec.output(self.output).produces_tensor(
            hs.tensor.OutputSpec(
                storage="runtime_pool",
                memory_kind=hs.tensor.MemoryKind.CUDA_DEVICE,
                dtype=hs.tensor.DLDataType(1, 8, 1),
                rank=3,
            ),
            hs.tensor.byte_span(3840 * 2160 * 3),
            field="data",
        )

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, context):
        # TensorRT may capture concurrently; avoid implicit legacy-stream dependencies.
        with cp.cuda.Stream(non_blocking=True) as stream:
            try:
                self._compute(context, stream)
            finally:
                stream.synchronize()

    def _compute(self, context, stream):
        sample = self.input.receive()
        image = sample.data
        # Import versioned DLPack without copying, using an explicit consumer stream.
        # Complete GPU reads before returning from this callback.
        capsule = image.data.__dlpack__(
            stream=stream.ptr, max_version=(1, 0), copy=False
        )
        cp_frame = cp.from_dlpack(capsule)
        descriptor = hs.schema.Image(
            height=image.height,
            width=image.width,
            encoding=hs.schema.image.ImageEncoding.RGB8,
        )
        request = hs.tensor.LoanRequest(
            shape=cp_frame.shape, dtype=hs.tensor.DLDataType(1, 8, 1)
        )
        # Keep the camera's capture time, which BayerDemosaicOp passes on: HolovizOp draws
        # the overlays captured no later than this frame with it.
        provenance = hs.sensor_io.ProvenanceAccumulator()
        provenance.observe(sample.metadata)
        options = provenance.emit_options(hs.sensor_io.CaptureTimePolicy.EARLIEST)
        with self.output.allocate_tensors(
            descriptor, {"data": request}, options=options
        ) as loan:
            tensor = loan.tensors["data"]
            memory = cp.cuda.UnownedMemory(tensor.data, cp_frame.size, tensor)
            destination = cp.ndarray(
                cp_frame.shape,
                dtype=cp.uint8,
                memptr=cp.cuda.MemoryPointer(memory, 0),
            )
            # Finish the CuPy write before the loan publishes to native consumers.
            cp.right_shift(cp_frame, self.shift, out=destination, casting="unsafe")
            stream.synchronize()


class FormatInferenceInputOp(hs.Operator):
    """Transposes HWC->CHW and adds a batch dimension, matching InferenceOp's expected input
    layout. Keeps the original CPU roundtrip and publishes contiguous data
    through an SDK-owned CUDA output buffer."""

    input = hs.Input("input", hs.schema.Image)
    output = hs.Output("output", hs.payload.Tensor)

    width = hs.Param(640)
    height = hs.Param(640)

    def setup(self, spec):
        spec.input(self.input, queue_depth=4)
        spec.output(self.output).produces_tensor(
            hs.tensor.OutputSpec(
                storage="runtime_pool",
                memory_kind=hs.tensor.MemoryKind.CUDA_DEVICE,
                dtype=hs.tensor.DLDataType(2, 32, 1),
                rank=4,
            ),
            hs.tensor.byte_span(1 * 3 * self.height * self.width * 4),
        )

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, context):
        # TensorRT may capture concurrently; avoid implicit legacy-stream dependencies.
        with cp.cuda.Stream(non_blocking=True) as stream:
            try:
                self._compute(context, stream)
            finally:
                stream.synchronize()

    def _compute(self, context, stream):
        sample = self.input.receive()
        image = sample.data
        provenance = hs.sensor_io.ProvenanceAccumulator()
        provenance.observe(sample.metadata)
        options = provenance.emit_options(hs.sensor_io.CaptureTimePolicy.EARLIEST)
        capsule = image.data.__dlpack__(
            stream=stream.ptr, max_version=(1, 0), copy=False
        )
        frame = cp.from_dlpack(capsule)
        tensor = frame.get()
        tensor = np.moveaxis(tensor, 2, 0)[None]
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)
        with self.output.allocate_tensor(
            shape=tensor.shape, dtype="float32", options=options
        ) as loan:
            memory = cp.cuda.UnownedMemory(loan.tensor.data, tensor.nbytes, loan.tensor)
            destination = cp.ndarray(
                tensor.shape, dtype=cp.float32, memptr=cp.cuda.MemoryPointer(memory, 0)
            )
            destination.set(tensor)
            stream.synchronize()


class PostprocessorOp(hs.Operator):
    """Extracts keypoints, builds skeleton segments, runs NMS. Box/keypoint/segment math ported
    unchanged from this repo's own HSDK4-era body_pose_estimation.py PostprocessorOp; output
    shape changed to HSDK5's typed ports and consolidated geometry-layer tensors -- see this
    file's header comment for why (HolovizOp's 8-layer cap)."""

    in_inference = hs.Input("in_inference", hs.payload.Tensor)
    keypoints = hs.Output("keypoints", hs.payload.Tensor)
    segments = hs.Output("segments", hs.payload.Tensor)

    iou_threshold = hs.Param(0.5)
    score_threshold = hs.Param(0.5)
    image_dim = hs.Param(640)

    def setup(self, spec):
        spec.input(self.in_inference, queue_depth=4)
        spec.output(self.keypoints)
        spec.output(self.segments)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.in_inference))

    def compute(self, context):
        with cp.cuda.Stream(non_blocking=True) as stream:
            try:
                self._compute(context, stream)
            finally:
                stream.synchronize()

    def _compute(self, context, stream):
        # ONNX models produce a singleton batch dimension that we skip.
        sample = self.in_inference.receive()
        # Overlay geometry describes the inference input, not this activation.
        provenance = hs.sensor_io.ProvenanceAccumulator()
        provenance.observe(sample.metadata)
        self._overlay_options = provenance.emit_options(
            hs.sensor_io.CaptureTimePolicy.EARLIEST
        )
        results = import_tensor(sample.data, stream)[0]

        results = results[:, results[4, :] > self.score_threshold]
        scores = results[4, :]

        if results.shape[1] == 0:
            empty = np.zeros((0,), dtype=np.float32)
            emit_host_tensor(self.keypoints, empty, self._overlay_options)
            emit_host_tensor(self.segments, empty, self._overlay_options)
            return

        results = results.transpose([1, 0])  # (n, channels)

        cx, cy, w, h = results[:, 0], results[:, 1], results[:, 2], results[:, 3]
        x1, x2 = cx - w / 2, cx + w / 2
        y1, y2 = cy - h / 2, cy + h / 2
        boxes = cp.stack([x1, y1, x2, y2], axis=-1).transpose(
            [1, 0]
        )  # (4, n) -- NMS math only

        kp = {name: results[:, sl] for name, sl in _KEYPOINT_SLICES.items()}
        kp["neck"] = (kp["left_shoulder"] + kp["right_shoulder"]) / 2

        visible_keypoints = cp.stack(
            [kp[name] for name in _VISIBLE_KEYPOINT_NAMES], axis=1
        )
        segment_points = cp.stack(
            [kp[name] for pair in _SEGMENT_PAIRS for name in pair], axis=1
        )  # (n, 2*len(_SEGMENT_PAIRS), 2)

        selected_indices = self._nms(boxes, scores)[:_MAX_OVERLAY_PERSONS]
        if len(selected_indices) == 0:
            empty = np.zeros((0,), dtype=np.float32)
            emit_host_tensor(self.keypoints, empty, self._overlay_options)
            emit_host_tensor(self.segments, empty, self._overlay_options)
            return

        visible_keypoints = visible_keypoints[selected_indices] / self.image_dim
        segment_points = segment_points[selected_indices] / self.image_dim

        sizes = cp.full(
            visible_keypoints.shape[:2] + (1,), _KEYPOINT_MARKER_SIZE, dtype=cp.float32
        )
        keypoints_out = cp.concatenate([visible_keypoints, sizes], axis=-1)

        emit_host_tensor(
            self.keypoints,
            cp.asnumpy(keypoints_out).astype(np.float32).ravel(),
            self._overlay_options,
        )
        emit_host_tensor(
            self.segments,
            cp.asnumpy(segment_points).astype(np.float32).ravel(),
            self._overlay_options,
        )

    def _nms(self, boxes, scores):
        """Non-max suppression (NMS). boxes: array (4, n); scores: array (n,).
        Returns the surviving detections' original indices, highest score first."""
        if boxes.shape[1] == 0:
            return cp.asarray([], dtype=cp.int64)

        x0, y0, x1, y1 = boxes[0, :], boxes[1, :], boxes[2, :], boxes[3, :]
        area = (x1 - x0 + 1) * (y1 - y0 + 1)
        indices = cp.argsort(scores)

        selected = []
        while len(indices) > 0:
            index = indices[-1]
            selected.append(index)

            x00 = cp.maximum(x0[index], x0[indices[:-1]])
            x11 = cp.minimum(x1[index], x1[indices[:-1]])
            y00 = cp.maximum(y0[index], y0[indices[:-1]])
            y11 = cp.minimum(y1[index], y1[indices[:-1]])

            width = cp.maximum(0, x11 - x00 + 1)
            height = cp.maximum(0, y11 - y00 + 1)
            overlap = width * height
            union = area[index] + area[indices[:-1]] - overlap
            iou = overlap / union

            left = cp.where(iou < self.iou_threshold)
            indices = indices[left]

        return cp.asarray(selected)


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
    ibv_name,
    ibv_port,
    engine,
    frame_limit,
    headless,
    fullscreen,
):
    import holoscan_inference  # noqa: F401 -- deferred; see linux_tao_peoplenet_imx477.py's

    # header comment for why.

    cuda_device_ordinal = 0

    print("Initializing CUDA...")
    _cuda_check(cuda_driver.cuInit(0))
    cu_device = _cuda_check(cuda_driver.cuDeviceGet(cuda_device_ordinal))
    cu_context = _cuda_check(cuda_driver.cuDevicePrimaryCtxRetain(cu_device))
    hololink_dev = None
    try:
        cuda_driver.cuCtxSetCurrent(cu_context)

        print(f"Enumerating HSB board at {hololink_ip}...")
        metadata = hl.find_channel(hololink_ip)
        channel = hl.DataChannel(metadata)

        hololink_dev = channel.hololink()
        hololink_dev.start()
        hololink_dev.reset()

        camera = hl.sensors.imx477.Imx477(channel, camera_id=cam, resolution=resolution)
        # Configures the camera for the requested resolution, 60fps.
        camera.configure()
        # IMX477 analog gain settings function. Analog gain value range is 0-1023 in decimal
        # (10 bits). Users are free to experiment with the register values.
        camera.set_analog_gain(0x2FF)
        camera.set_exposure_reg(exposure)

        pixel_format = camera.pixel_format()
        bayer_format = camera.bayer_format()
        frame_width = camera._width
        frame_height = camera._height
        geometry = hl.resolve_generic_geometry_from_dimensions(
            int(pixel_format), frame_width, frame_height
        )

        # Build the graph.
        graph = hs.Graph("body_pose_estimation_imx477")

        rx = graph.op(
            hl.operators.RoceReceiverOp,
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
            ibv_name=ibv_name,
            ibv_port=ibv_port,
            frame_limit=frame_limit or 0,
            frame_limit_ends_run=True,
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
            pixel_width=geometry["pixel_width"],
            pixel_height=geometry["pixel_height"],
            bayer_format=int(bayer_format),
            optical_black=50,
            cuda_device_ordinal=cuda_device_ordinal,
        )
        graph.add_flow(csi.tensor, image_processor.in_tensor, queue_depth=4)
        graph.add_flow(csi.metadata, image_processor.in_metadata, queue_depth=4)

        demosaic = graph.op(
            holoscan_vision.operators.BayerDemosaicOp,
            "bayer_demosaic",
            bayer_grid_pos=bayer_format.name.lower(),
            generate_alpha=False,
        )
        graph.add_flow(image_processor.tensor, demosaic.input, queue_depth=4)

        image_shift = graph.op(ImageShiftToUint8Op, "image_shift", shift=8)
        graph.add_flow(
            demosaic.output,
            image_shift.input,
            queue_depth=4,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
        )

        # Inference side-chain: resize -> float32 cast+scale -> HWC->CHW+batch -> InferenceOp ->
        # postprocess.
        resize = graph.op(
            holoscan_vision.operators.ResizeOp,
            "resize",
            width=640,
            height=640,
            max_batch_size=1,
        )
        graph.add_flow(
            image_shift.output,
            resize.input,
            queue_depth=4,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
        )

        convert = graph.op(
            holoscan_vision.operators.ConvertToOp,
            "convert",
            output_type="float32",
            alpha=1.0 / 255.0,
            beta=0.0,
        )
        graph.add_flow(
            resize.output,
            convert.input,
            queue_depth=4,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
        )

        format_input = graph.op(FormatInferenceInputOp, "transpose")
        graph.add_flow(
            convert.output,
            format_input.input,
            queue_depth=4,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
        )

        inference = graph.op(
            "holoscan.inference.InferenceOp",
            "inference",
            model_path_map={"yolo_pose": engine},
            input_map={"yolo_pose": ["preprocessed"]},
            output_map={"yolo_pose": ["inference_output"]},
            backend="trt",
            input_on_cuda=True,
            is_engine_path=True,
        )
        graph.add_flow(
            format_input.output,
            inference.preprocessed,
            queue_depth=4,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
        )

        postprocessor = graph.op(
            PostprocessorOp,
            "postprocessor",
            iou_threshold=0.5,
            score_threshold=0.5,
            image_dim=640,
        )
        graph.add_flow(
            inference.inference_output,
            postprocessor.in_inference,
            queue_depth=4,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
        )

        if headless:
            # Renders into a CUDA image on HolovizOp's output (consumed by DiscardImageOp). The
            # CUDA sink has no framebuffer_srgb: its sRGB format encodes on write instead.
            sink = {
                "kind": "cuda",
                "width": frame_width,
                "height": frame_height,
                "format": "rgba8_srgb",
            }
        else:
            sink = {
                "kind": "window",
                "framebuffer_srgb": True,
                "title": f"Holoviz - {hololink_ip}",
                "width": frame_width,
                "height": frame_height,
                "fullscreen": fullscreen,
                "close_on_escape": True,
            }
        viz = graph.op(
            "holoscan.visualization.HolovizOp",
            "holoviz",
            layers=[
                {
                    "port_name": "video",
                    "kind": "image",
                    "format": "rgb8_unorm",
                    "max_width": frame_width,
                    "max_height": frame_height,
                },
                {
                    "port_name": "keypoints",
                    "kind": "geometry",
                    "topologies": ["cross_list"],
                    "max_primitives": _MAX_OVERLAY_PERSONS
                    * len(_VISIBLE_KEYPOINT_NAMES),
                    "default_line_width": 4,
                },
                {
                    "port_name": "segments",
                    "kind": "geometry",
                    "topologies": ["line_list"],
                    "max_primitives": _MAX_OVERLAY_PERSONS * len(_SEGMENT_PAIRS),
                    "default_line_width": 4,
                },
            ],
            sink=sink,
            trigger={"kind": "layer", "layer": 0},
        )
        graph.add_flow(
            image_shift.output,
            viz.video,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
            queue_depth=1,
        )
        graph.add_flow(
            postprocessor.keypoints,
            viz.keypoints,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
            queue_depth=1,
        )
        graph.add_flow(
            postprocessor.segments,
            viz.segments,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
            queue_depth=1,
        )
        if headless:
            discard = graph.op(DiscardImageOp, "discard")
            graph.add_flow(
                viz.output,
                discard.input,
                queue_policy=hs.QueuePolicy.DROP_OLDEST,
                queue_depth=1,
            )

        clock_key = graph.add_clock(hs.RealtimeClock, "clock")
        graph.set_default_clock(clock_key)

        options = hs.CompileOptions()
        for operator_path in ("rx", "csi_to_bayer", "image_processor"):
            options.deployment.bind_tensor_output_device(
                operator_path=operator_path,
                output_port="tensor",
                device=hs.execution.DeviceId(cuda_device_ordinal),
            )
        for operator_path in (
            "bayer_demosaic",
            "image_shift",
            "resize",
            "convert",
            "transpose",
        ):
            options.deployment.bind_tensor_output_device(
                operator_path=operator_path,
                output_port="output",
                device=hs.execution.DeviceId(cuda_device_ordinal),
            )

        options.deployment.bind_tensor_output_device(
            operator_path="inference",
            output_port="inference_output",
            device=hs.execution.DeviceId(cuda_device_ordinal),
        )

        if headless:
            options.deployment.bind_tensor_output_device(
                operator_path="holoviz",
                output_port="output",
                device=hs.execution.DeviceId(cuda_device_ordinal),
            )

        plan = hs.compile(graph, options)
        if not plan.ok:
            print("Graph compilation failed:", file=sys.stderr)
            for diag in plan.diagnostics:
                print(f"  {diag}", file=sys.stderr)
            return 1

        print(
            f"Capturing {resolution} {frame_width}x{frame_height} -- running (Ctrl+C to stop)..."
        )

        with hs.run_async(plan) as session:
            failed = wait_for_completion(session)
        if failed:
            print("Run failed; see runtime diagnostics above.", file=sys.stderr)
            return 1

        print("Done.")
        return 0

    finally:
        try:
            if hololink_dev is not None:
                hololink_dev.stop()
        finally:
            _cuda_check(cuda_driver.cuDevicePrimaryCtxRelease(cu_device))


def main():
    # Like Ctrl+C, SIGTERM (kill, docker stop) raises KeyboardInterrupt, so the session stops
    # cooperatively and the camera and the board are stopped on the way out.
    signal.signal(signal.SIGTERM, signal.default_int_handler)

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
        "--ibv-name",
        default="",
        help="IBV device to use (default: the device on the board's network interface)",
    )
    parser.add_argument(
        "--ibv-port",
        type=int,
        default=1,
        help="Port number of IBV device (with --ibv-name)",
    )
    # Dead flag, preserved for CLI compatibility: IMX477 takes no expander_configuration
    # argument at all (unlike IMX274's I2C-expander-driven boards) -- the old app parses this
    # but never passes it anywhere, matching its own real (if confusing) behavior.
    parser.add_argument(
        "--expander-configuration",
        type=int,
        default=0,
        choices=(0, 1),
        help="I2C Expander configuration",
    )
    parser.add_argument(
        "--exposure",
        type=int,
        default=0x05,
        help="Configure exposure.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    return run(
        args.hololink,
        args.cam,
        args.resolution,
        args.exposure,
        args.ibv_name,
        args.ibv_port,
        args.engine,
        args.frame_limit,
        args.headless,
        args.fullscreen,
    )


if __name__ == "__main__":
    sys.exit(main())
