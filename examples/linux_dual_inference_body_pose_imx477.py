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

# Stereo IMX477 sensors (always both) sharing one board/IP, each running its own independent
# YOLOv8-pose body-pose-estimation inference pass, tiled side by side in one Holoviz window.
# Python port of this repo's own previous (HSDK4-era) Python version of this app, via the HSDK5
# Python bindings -- there is no examples/linux_dual_inference_body_pose_imx477.cpp sibling. The
# per-sensor pipeline is the stereo form of examples/linux_body_pose_estimation_imx477.py (see
# that file's own header comment for the InferenceOp/ResizeOp/ConvertToOp/deferred-import/
# keypoint-consolidation/geometry-layer-color-loss rationale, all of which applies unchanged per
# side here); bring-up follows the single-network-stereo pattern already established by
# examples/linux_single_network_stereo_imx477_player.py (one find_channel() + hl.use_sensor()
# call per sensor, same physical board).
#
# "Dual" means two sensors each independently running the SAME model, not a two-stage cascade --
# confirmed from the old app's own _build_camera_pipeline() being called once per side with no
# cross-side data dependency.
#
#   [per side] LinuxReceiverOp -> CsiToBayerOp -> ImageProcessorOp -> BayerDemosaicOp ->
#     ImageShiftToUint8Op -> (a) HolovizOp "video_{side}" image layer
#                             -> (b) ResizeOp -> ConvertToOp -> FormatInferenceInputOp ->
#                                InferenceOp -> PostprocessorOp ->
#                                HolovizOp "keypoints_{side}"/"segments_{side}" geometry layers
#
# The old app's RenameTensorsOp (a custom operator that relabeled tensor keys with a left_/right_
# prefix, needed because the old API's add_flow() targets were fixed-name ports that would
# otherwise collide once both sides fed the same HolovizOp) is dropped entirely here, not
# ported: HSDK5's graph.op() already gives every operator instance (and therefore every
# HolovizOp layer port) its own explicit unique name per side (video_left/video_right,
# keypoints_left/keypoints_right, ...), so there is nothing left for a rename shim to do --
# confirmed by this exact no-rename-needed pattern already working for every other stereo app
# ported this session (stereo_imx477_player.py and friends).
#
# Real, flagged behavior change: the old app's per-side tiling used HolovizOp InputSpec's own
# `view` transform (offset_x/width), applied uniformly to the image layer AND every geometry
# layer for that side. HSDK5's new GeometryLayerDesc has no view field at all (confirmed
# directly from its C++ struct -- only ImageLayerDesc has one) -- geometry layers can't be
# independently positioned/scaled by the renderer. Image layers ("video_{side}") still use the
# declarative "view" dict (offset_x/scale_x, the new API's rename of the old offset_x/width);
# for the keypoints/segments geometry layers, the SAME half-and-half tiling is instead baked
# directly into the coordinates PostprocessorOp emits (view_offset_x/view_scale_x params) --
# same visual result, computed instead of declared.
#
# Usage:
#   linux_dual_inference_body_pose_imx477.py [OPTIONS]
#
# Options:
#   --hololink IP        IP address of Hololink board (default: 192.168.0.2)
#   --resolution         4k or 1080p -- shared by both cameras (default: 1080p)
#   --exposure N         Configure exposure -- shared by both cameras (default: 0x05)
#   --engine PATH        YOLOv8-pose TRT engine path -- shared by both cameras
#                         (default: examples/yolov8n-pose.engine.fp32)
#   --window-height N    Displayed window height (default: 2160 // 4)
#   --window-width N     Displayed window width (default: 3840 // 3)
#   --frame-limit N      Number of frames to capture per sensor (default: unlimited)
#   --headless           Run without opening a window (still renders, presents nowhere)
#   --fullscreen         Run in fullscreen mode
#   --log-level N        Python logging level (default: 20 = INFO)

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

_KEYPOINT_MARKER_SIZE = (
    0.01  # see linux_body_pose_estimation_imx477.py's own comment; scaled
)
# by view_scale_x below to stay proportional once squeezed into a half-width tile.

_MAX_OVERLAY_PERSONS = 16


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


def wait_for_completion(session, observer=None, frame_limit=0):
    """Notice runtime failure as well as capture completion, including multi-leg observers."""
    if observer is None:
        try:
            session.wait()
        except KeyboardInterrupt:
            session.request_stop()
            session.wait()
    while observer is not None and not session.wait(timeout=0.1):
        if frame_limit:
            with observer.condition:
                done = (
                    observer._done_locked()
                    if hasattr(observer, "_done_locked")
                    else observer.count >= frame_limit
                )
            if done:
                session.request_stop()
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
    shape changed to HSDK5's typed ports and consolidated geometry-layer tensors (see this
    file's header comment for why), with view_offset_x/view_scale_x baking this app's
    side-by-side tiling directly into the emitted coordinates (geometry layers have no view
    transform of their own -- also explained in the header comment)."""

    in_inference = hs.Input("in_inference", hs.payload.Tensor)
    keypoints = hs.Output("keypoints", hs.payload.Tensor)
    segments = hs.Output("segments", hs.payload.Tensor)

    iou_threshold = hs.Param(0.5)
    score_threshold = hs.Param(0.5)
    image_dim = hs.Param(640)
    view_offset_x = hs.Param(0.0)
    view_scale_x = hs.Param(1.0)

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

        # Bake this side's half-screen tiling directly into the coordinates (x only -- the full
        # frame height still maps to the full window height on both sides).
        visible_keypoints_x = (
            visible_keypoints[..., 0] * self.view_scale_x + self.view_offset_x
        )
        visible_keypoints = cp.stack(
            [visible_keypoints_x, visible_keypoints[..., 1]], axis=-1
        )
        segment_points_x = (
            segment_points[..., 0] * self.view_scale_x + self.view_offset_x
        )
        segment_points = cp.stack([segment_points_x, segment_points[..., 1]], axis=-1)

        sizes = cp.full(
            visible_keypoints.shape[:2] + (1,),
            _KEYPOINT_MARKER_SIZE * self.view_scale_x,
            dtype=cp.float32,
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


def _build_leg(
    graph,
    side,
    view_offset_x,
    cuda_device_ordinal,
    cu_context,
    channel,
    camera,
    engine,
    frame_limit,
):
    pixel_format = camera.pixel_format()
    bayer_format = camera.bayer_format()
    geometry = hl.resolve_generic_geometry_from_dimensions(
        int(pixel_format), camera._width, camera._height
    )

    rx = graph.op(
        hl.operators.LinuxReceiverOp,
        f"rx_{side}",
        data_channel=channel,
        frame_context=int(cu_context),
        device_start=camera.start,
        device_stop=camera.stop,
        frame_size=(
            geometry["start_byte"]
            + geometry["bytes_per_line"] * geometry["pixel_height"]
            + geometry["trailing_bytes"]
        ),
        frame_limit=frame_limit or 0,
        frame_limit_ends_run=True,
    )

    csi = graph.op(
        hl.operators.CsiToBayerOp,
        f"csi_to_bayer_{side}",
        cuda_device_ordinal=cuda_device_ordinal,
        geometry=geometry,
        bayer_format=bayer_format.name.lower(),
    )
    graph.add_flow(rx.tensor, csi.in_tensor, queue_depth=4)
    graph.add_flow(rx.metadata, csi.in_metadata, queue_depth=4)

    image_processor = graph.op(
        hl.operators.ImageProcessorOp,
        f"image_processor_{side}",
        pixel_format=int(pixel_format),
        pixel_width=geometry["pixel_width"],
        pixel_height=geometry["pixel_height"],
        bayer_format=int(bayer_format),
        optical_black=100,
        cuda_device_ordinal=cuda_device_ordinal,
    )
    graph.add_flow(csi.tensor, image_processor.in_tensor, queue_depth=4)
    graph.add_flow(csi.metadata, image_processor.in_metadata, queue_depth=4)

    demosaic = graph.op(
        holoscan_vision.operators.BayerDemosaicOp,
        f"bayer_demosaic_{side}",
        bayer_grid_pos=bayer_format.name.lower(),
        generate_alpha=False,
    )
    graph.add_flow(image_processor.tensor, demosaic.input, queue_depth=4)

    image_shift = graph.op(ImageShiftToUint8Op, f"image_shift_{side}", shift=8)
    graph.add_flow(
        demosaic.output,
        image_shift.input,
        queue_depth=4,
        queue_policy=hs.QueuePolicy.DROP_OLDEST,
    )

    resize = graph.op(
        holoscan_vision.operators.ResizeOp,
        f"resize_{side}",
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
        f"convert_{side}",
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

    format_input = graph.op(FormatInferenceInputOp, f"transpose_{side}")
    graph.add_flow(
        convert.output,
        format_input.input,
        queue_depth=4,
        queue_policy=hs.QueuePolicy.DROP_OLDEST,
    )

    model_name = f"yolo_pose_{side}"
    inference = graph.op(
        "holoscan.inference.InferenceOp",
        f"inference_{side}",
        model_path_map={model_name: engine},
        input_map={model_name: ["preprocessed"]},
        output_map={model_name: ["inference_output"]},
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
        f"postprocessor_{side}",
        iou_threshold=0.5,
        score_threshold=0.5,
        image_dim=640,
        view_offset_x=view_offset_x,
        view_scale_x=0.5,
    )
    graph.add_flow(
        inference.inference_output,
        postprocessor.in_inference,
        queue_depth=4,
        queue_policy=hs.QueuePolicy.DROP_OLDEST,
    )

    tensor_output_paths = [
        f"rx_{side}",
        f"csi_to_bayer_{side}",
        f"image_processor_{side}",
    ]
    image_output_paths = [
        f"bayer_demosaic_{side}",
        f"image_shift_{side}",
        f"resize_{side}",
        f"convert_{side}",
        f"transpose_{side}",
    ]
    return image_shift, postprocessor, tensor_output_paths, image_output_paths


def run(
    hololink_ip,
    resolution,
    exposure,
    engine,
    window_height,
    window_width,
    frame_limit,
    headless,
    fullscreen,
):
    import holoscan_inference  # noqa: F401 -- deferred; see linux_tao_peoplenet_imx477.py's

    # header comment for why.

    cuda_device_ordinal = 0
    sides = ["left", "right"]

    print("Initializing CUDA...")
    _cuda_check(cuda_driver.cuInit(0))
    cu_device = _cuda_check(cuda_driver.cuDeviceGet(cuda_device_ordinal))
    cu_context = _cuda_check(cuda_driver.cuDevicePrimaryCtxRetain(cu_device))
    cuda_driver.cuCtxSetCurrent(cu_context)

    print(f"Enumerating HSB board at {hololink_ip}...")
    channels = []
    for sensor_id in (0, 1):
        metadata = hl.find_channel(hololink_ip)
        hl.use_sensor(metadata, sensor_id)
        channels.append(hl.DataChannel(metadata))

    hololink_dev = channels[0].hololink()
    try:
        hololink_dev.start()
        hololink_dev.reset()

        cameras = [
            hl.sensors.imx477.Imx477(channels[0], camera_id=0, resolution=resolution),
            hl.sensors.imx477.Imx477(channels[1], camera_id=1, resolution=resolution),
        ]
        for camera in cameras:
            # Configures the camera for the requested resolution, 60fps.
            camera.configure()
            # IMX477 analog gain settings function. Analog gain value range is 0-1023 in decimal
            # (10 bits). Users are free to experiment with the register values.
            camera.set_analog_gain(0x2FF)
            camera.set_exposure_reg(exposure)

        # Build the graph.
        graph = hs.Graph("linux_dual_inference_body_pose_imx477")

        image_shifts = {}
        postprocessors = {}
        tensor_port_operators = []
        output_port_operators = []
        for i, (side, view_offset_x) in enumerate(zip(sides, (0.0, 0.5))):
            image_shift, postprocessor, tensor_paths, image_paths = _build_leg(
                graph,
                side,
                view_offset_x,
                cuda_device_ordinal,
                cu_context,
                channels[i],
                cameras[i],
                engine,
                frame_limit,
            )
            image_shifts[side] = image_shift
            postprocessors[side] = postprocessor
            tensor_port_operators += tensor_paths
            output_port_operators += image_paths

        layers = []
        for side, offset_x in zip(sides, (0.0, 0.5)):
            layers.append(
                {
                    "port_name": f"video_{side}",
                    "kind": "image",
                    "format": "rgb8_unorm",
                    "max_width": cameras[0]._width,
                    "max_height": cameras[0]._height,
                    "view": {
                        "offset_x": offset_x,
                        "offset_y": 0.0,
                        "scale_x": 0.5,
                        "scale_y": 1.0,
                    },
                }
            )
            layers.append(
                {
                    "port_name": f"keypoints_{side}",
                    "kind": "geometry",
                    "topologies": ["cross_list"],
                    "max_primitives": _MAX_OVERLAY_PERSONS
                    * len(_VISIBLE_KEYPOINT_NAMES),
                    "default_line_width": 4,
                }
            )
            layers.append(
                {
                    "port_name": f"segments_{side}",
                    "kind": "geometry",
                    "topologies": ["line_list"],
                    "max_primitives": _MAX_OVERLAY_PERSONS * len(_SEGMENT_PAIRS),
                    "default_line_width": 4,
                }
            )

        if headless:
            # Renders into a CUDA image on HolovizOp's output (consumed by DiscardImageOp). The
            # CUDA sink has no framebuffer_srgb: its sRGB format encodes on write instead.
            sink = {
                "kind": "cuda",
                "width": window_width,
                "height": window_height,
                "format": "rgba8_srgb",
            }
        else:
            sink = {
                "kind": "window",
                "framebuffer_srgb": True,
                "title": f"Holoviz - {hololink_ip}",
                "width": window_width,
                "height": window_height,
                "fullscreen": fullscreen,
                "close_on_escape": True,
            }
        viz = graph.op(
            "holoscan.visualization.HolovizOp",
            "holoviz",
            layers=layers,
            sink=sink,
            trigger={"kind": "layer", "layer": 0},
        )
        for side in sides:
            graph.add_flow(
                image_shifts[side].output,
                getattr(viz, f"video_{side}"),
                queue_policy=hs.QueuePolicy.DROP_OLDEST,
                queue_depth=1,
            )
            graph.add_flow(
                postprocessors[side].keypoints,
                getattr(viz, f"keypoints_{side}"),
                queue_policy=hs.QueuePolicy.DROP_OLDEST,
                queue_depth=1,
            )
            graph.add_flow(
                postprocessors[side].segments,
                getattr(viz, f"segments_{side}"),
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
        for operator_path in tensor_port_operators:
            options.deployment.bind_tensor_output_device(
                operator_path=operator_path,
                output_port="tensor",
                device=hs.execution.DeviceId(cuda_device_ordinal),
            )
        for operator_path in output_port_operators:
            options.deployment.bind_tensor_output_device(
                operator_path=operator_path,
                output_port="output",
                device=hs.execution.DeviceId(cuda_device_ordinal),
            )

        for side in sides:
            options.deployment.bind_tensor_output_device(
                operator_path=f"inference_{side}",
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

        print("Capturing 2 sensors -- running (Ctrl+C to stop)...")

        with hs.run_async(plan) as session:
            failed = wait_for_completion(session)

        if failed:
            print("Run failed; see runtime diagnostics above.", file=sys.stderr)
            return 1
        print("Done.")
        return 0

    finally:
        try:
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

    logging.basicConfig(level=args.log_level)

    return run(
        args.hololink,
        args.resolution,
        args.exposure,
        args.engine,
        args.window_height,
        args.window_width,
        args.frame_limit,
        args.headless,
        args.fullscreen,
    )


if __name__ == "__main__":
    sys.exit(main())
