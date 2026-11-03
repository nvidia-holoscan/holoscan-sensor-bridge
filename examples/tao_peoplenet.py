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

# Single IMX274 sensor via RoceReceiverOp, running TAO PeopleNet (person + face detection) and
# overlaying the boxes on the live video in Holoviz. Mechanical sensor swap of the already-ported
# examples/tao_peoplenet_imx477.py -- same inference side-chain (InferenceOp/ResizeOp/ConvertToOp/
# FormatInferenceInputOp/PostprocessorOp, including the box_scale=35.0/box_offset=0.5/grid 34x60
# postprocessor params and 960x544 resize, confirmed matching exactly between both apps' own
# YAML/code), same HolovizOp "person"/"faces" rectangle_list layers, same geometry-layer
# color-loss gap and deferred holoscan_inference import -- see that file's own header comment for
# the full rationale this round doesn't repeat. Only the camera/bring-up differs: NativeImx274Sensor
# + hl.resolve_generic_geometry(camera), matching examples/imx274_player.py's own already-proven
# pattern, and optical_black=50 (not IMX477's own 100 -- different sensor characteristics,
# matching this app's own old value, same as body_pose_estimation.py's own IMX274 optical_black).
#
#   RoceReceiverOp -> CsiToBayerOp -> ImageProcessorOp -> BayerDemosaicOp -> ImageShiftToUint8Op
#     -> (a) HolovizOp "video" image layer
#     -> (b) ResizeOp -> ConvertToOp -> FormatInferenceInputOp -> InferenceOp -> PostprocessorOp
#          -> HolovizOp "person"/"faces" geometry layers
#
# Usage:
#   tao_peoplenet.py [OPTIONS]
#
# Options:
#   --camera-mode N   IMX274 mode (default: 0 = 3840x2160@60fps)
#   --expander-configuration I2C expander configuration (default: 0)
#   --ibv-name NAME   IBV device to use (default: the device on the board's network interface)
#   --ibv-port N      Port number of IBV device (default: 1)
#   --engine PATH     TAO PeopleNet ONNX model path (default: examples/resnet34_peoplenet_int8.onnx)
#   --frame-limit N   Number of frames to capture (default: unlimited)
#   --headless        Run without opening a window (still renders, presents nowhere)
#   --fullscreen      Run in fullscreen mode
#   --log-level N     Python logging level (default: 20 = INFO)

import argparse
import logging
import os
import signal
import sys
from datetime import timedelta

import cupy as cp
import holoscan as hs
import holoscan_vision
import holoscan_visualization as hz  # noqa: F401 -- registers "holoscan.visualization.HolovizOp"
import numpy as np
from cuda.bindings import driver as cuda_driver

import hololink as hl

# PeopleNet has three output classes: 0 person, 1 bag, 2 face. Only person/face are shown,
# matching the old app's own selection (bag is decoded but discarded).
_PERSON_FACE_BOX_CHANNELS = [0, 1, 2, 3, 8, 9, 10, 11]
_PERSON_FACE_SCORE_CHANNELS = [0, 2]

# Bounded-window pairing for InferenceOp's two sibling output ports (scores, boxes), which it
# always emits together from the same compute() call -- same OnAligned+Exact{} idiom this
# session's own tensor/metadata pairings already use, just applied to two inference outputs
# instead of a tensor+metadata pair.
_INFERENCE_OUTPUT_PAIRING_WINDOW = timedelta(microseconds=500)

# Conservative overlay capacity: PeopleNet's 34x60 grid yields far more candidate boxes per
# class than real people/faces visible in one frame after NMS; 64 is generous headroom, not a
# value derived from the model itself.
_MAX_OVERLAY_BOXES = 64

# How long HolovizOp keeps an overlay sample for the video frames that follow it. The default
# (100 ms) is too short in 4K mode: inference runs at ~15 Hz against 60 Hz video, and its boxes
# reach HolovizOp more than 100 ms after their video frame, so most are dropped and the boxes
# flicker. 500 ms keeps each box until the next inference result arrives.
_OVERLAY_MAX_AGE = timedelta(milliseconds=500)


_BAYER_FORMAT_VALUES = {
    name: index for index, name in enumerate(["bggr", "rggb", "gbrg", "grbg"])
}


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
        # Run the CuPy work on the SDK's activation stream, not CuPy's legacy default stream
        # (which TensorRT's concurrent graph capture forbids). The runtime orders this
        # callback's inputs on it and records its outputs' readiness on it.
        with context.stream():
            self._compute(context, cp.cuda.get_current_stream())

    def _compute(self, context, stream):
        sample = self.input.receive()
        image = sample.data
        # Import versioned DLPack without copying, on the activation stream.
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
            cp.right_shift(cp_frame, self.shift, out=destination, casting="unsafe")


class FormatInferenceInputOp(hs.Operator):
    """Transposes HWC->CHW and adds a batch dimension, matching InferenceOp's expected input
    layout. Keeps the original CPU roundtrip and publishes contiguous data
    through an SDK-owned CUDA output buffer."""

    input = hs.Input("input", hs.schema.Image)
    output = hs.Output("output", hs.payload.Tensor)

    width = hs.Param(960)
    height = hs.Param(544)

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
        # Run the CuPy work on the SDK's activation stream, not CuPy's legacy default stream
        # (which TensorRT's concurrent graph capture forbids). The runtime orders this
        # callback's inputs on it and records its outputs' readiness on it.
        with context.stream():
            self._compute(context, cp.cuda.get_current_stream())

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


class PostprocessorOp(hs.Operator):
    """Reparameterizes PeopleNet's grid-encoded boxes, runs NMS, emits flat float32
    [x0,y0,x1,y1, ...] box arrays for HolovizOp's rectangle_list geometry layers (4 floats per
    box, confirmed against holoscan_visualization's own kGeometryRectFloats constant). Box math
    (reparameterize_boxes/nms) ported unchanged from this repo's own HSDK4-era
    tao_peoplenet.py PostprocessorOp; only I/O changed to HSDK5's typed ports and the new flat
    rank-1-tensor geometry format (the old code's final (1, N, 2) point-pair reshape, for the
    old HolovizOp InputSpec convention, becomes a flat (4*N,) ravel here instead)."""

    in_scores = hs.Input("in_scores", hs.payload.Tensor)
    in_boxes = hs.Input("in_boxes", hs.payload.Tensor)
    person = hs.Output("person", hs.payload.Tensor)
    faces = hs.Output("faces", hs.payload.Tensor)

    iou_threshold = hs.Param(0.15)
    score_threshold = hs.Param(0.5)
    image_width = hs.Param(0)
    image_height = hs.Param(0)
    box_scale = hs.Param(0.0)
    box_offset = hs.Param(0.0)
    grid_height = hs.Param(0)
    grid_width = hs.Param(0)

    def setup(self, spec):
        spec.input(self.in_scores, queue_depth=4)
        spec.input(self.in_boxes, queue_depth=4)
        spec.output(self.person)
        spec.output(self.faces)

    def contract(self):
        return (
            hs.Contract()
            .trigger(
                hs.OnAligned(
                    ports=[self.in_scores, self.in_boxes],
                    window=_INFERENCE_OUTPUT_PAIRING_WINDOW,
                )
            )
            .input(self.in_scores, hs.Exact())
            .input(self.in_boxes, hs.Exact())
        )

    def compute(self, context):
        # Run the CuPy work on the SDK's activation stream, not CuPy's legacy default stream
        # (which TensorRT's concurrent graph capture forbids). The runtime orders this
        # callback's inputs on it and records its outputs' readiness on it.
        with context.stream():
            self._compute(context, cp.cuda.get_current_stream())

    def _compute(self, context, stream):
        # ONNX models produce a singleton batch dimension that we skip.
        boxes_sample = self.in_boxes.receive()
        scores_sample = self.in_scores.receive()
        # Overlay geometry describes the aligned inference inputs, not this activation.
        provenance = hs.sensor_io.ProvenanceAccumulator()
        provenance.observe(boxes_sample.metadata)
        provenance.observe(scores_sample.metadata)
        self._overlay_options = provenance.emit_options(
            hs.sensor_io.CaptureTimePolicy.EARLIEST
        )
        boxes = import_tensor(boxes_sample.data, stream)[0, ...]
        scores = import_tensor(scores_sample.data, stream)[0, ...]

        boxes = boxes[_PERSON_FACE_BOX_CHANNELS, ...][None]
        scores = scores[_PERSON_FACE_SCORE_CHANNELS, ...][None]

        outputs = {}
        for i, label in enumerate(("person", "faces")):
            label_boxes, label_scores = self._reparameterize_boxes(
                boxes[:, 0 + i * 4 : 4 + i * 4, ...],
                scores[:, i, ...][None],
            )
            label_boxes, _ = self._nms(label_boxes, label_scores)
            label_boxes = label_boxes[:_MAX_OVERLAY_BOXES]
            if len(label_boxes) == 0:
                outputs[label] = np.zeros((0,), dtype=np.float32)
            else:
                label_boxes[:, [0, 2]] /= self.image_width
                label_boxes[:, [1, 3]] /= self.image_height
                outputs[label] = cp.asnumpy(label_boxes).astype(np.float32).ravel()

        emit_host_tensor(self.person, outputs["person"], self._overlay_options)
        emit_host_tensor(self.faces, outputs["faces"], self._overlay_options)

    def _nms(self, boxes, scores):
        """Non-max suppression (NMS). boxes: array (4, n); scores: array (n,).
        Returns boxes: array (m, 4), scores: array (m,)."""
        if len(boxes) == 0:
            return cp.asarray([]), cp.asarray([])

        x0, y0, x1, y1 = boxes[0, :], boxes[1, :], boxes[2, :], boxes[3, :]
        area = (x1 - x0 + 1) * (y1 - y0 + 1)
        indices = cp.argsort(scores)

        boxes_out, scores_out = [], []
        while len(indices) > 0:
            index = indices[-1]
            boxes_out.append(boxes[:, index])
            scores_out.append(scores[index])

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

        return cp.asarray(boxes_out), cp.asarray(scores_out)

    def _reparameterize_boxes(self, boxes, scores):
        """Reparameterize boxes from corner+width+height to corner+corner.
        boxes: array (1, 4, grid_height, grid_width); scores: array (1, 1, grid_height, grid_width).
        Returns boxes: array (4, n), scores: array (n,)."""
        cell_height = self.image_height / self.grid_height
        cell_width = self.image_width / self.grid_width

        mx, my = cp.meshgrid(cp.arange(self.grid_width), cp.arange(self.grid_height))
        mx = mx.astype(np.float32).reshape((1, 1, self.grid_height, self.grid_width))
        my = my.astype(np.float32).reshape((1, 1, self.grid_height, self.grid_width))

        xmin = -(boxes[0, 0, ...] + self.box_offset) * self.box_scale + mx * cell_width
        ymin = -(boxes[0, 1, ...] + self.box_offset) * self.box_scale + my * cell_height
        xmax = (boxes[0, 2, ...] + self.box_offset) * self.box_scale + mx * cell_width
        ymax = (boxes[0, 3, ...] + self.box_offset) * self.box_scale + my * cell_height
        boxes = cp.concatenate([xmin, ymin, xmax, ymax], axis=1)

        scores_mask = scores > self.score_threshold
        scores = scores[scores_mask]
        scores_mask = cp.repeat(scores_mask, 4, axis=1)
        boxes = boxes[scores_mask]

        n = int(boxes.size / 4)
        boxes = boxes.reshape(4, n)

        return boxes, scores


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
    camera_mode,
    expander_configuration,
    ibv_name,
    ibv_port,
    engine,
    frame_limit,
    headless,
    fullscreen,
):
    import holoscan_inference  # noqa: F401 -- deferred; see tao_peoplenet_imx477.py's header.

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

        camera = hl.NativeImx274Sensor(
            channel, expander_configuration=expander_configuration
        )
        camera.setup_clock()
        camera.configure(camera_mode)
        camera.set_digital_gain_reg(0x4)

        pixel_format = camera.get_pixel_format()
        bayer_format = camera.get_bayer_format()
        frame_width = camera.get_width()
        frame_height = camera.get_height()
        geometry = hl.resolve_generic_geometry(camera)

        # Build the graph.
        graph = hs.Graph("tao_peoplenet")

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
            bayer_format=bayer_format,
        )
        graph.add_flow(
            rx.tensor,
            csi.in_tensor,
            queue_depth=4,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
        )
        graph.add_flow(
            rx.metadata,
            csi.in_metadata,
            queue_depth=4,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
        )

        image_processor = graph.op(
            hl.operators.ImageProcessorOp,
            "image_processor",
            pixel_format=int(pixel_format),
            pixel_width=geometry["pixel_width"],
            pixel_height=geometry["pixel_height"],
            bayer_format=_BAYER_FORMAT_VALUES[bayer_format],
            optical_black=50,
            cuda_device_ordinal=cuda_device_ordinal,
        )
        graph.add_flow(
            csi.tensor,
            image_processor.in_tensor,
            queue_depth=4,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
        )
        graph.add_flow(
            csi.metadata,
            image_processor.in_metadata,
            queue_depth=4,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
        )

        demosaic = graph.op(
            holoscan_vision.operators.BayerDemosaicOp,
            "bayer_demosaic",
            bayer_grid_pos=bayer_format,
            generate_alpha=False,
        )
        graph.add_flow(
            image_processor.tensor,
            demosaic.input,
            queue_depth=4,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
        )

        image_shift = graph.op(ImageShiftToUint8Op, "image_shift", shift=8)
        # From here on every edge is a depth-1 drop-oldest mailbox: the inference chain works on
        # the newest frame and skips the others, instead of queueing stale frames ahead of it.
        graph.add_flow(
            demosaic.output,
            image_shift.input,
            queue_depth=1,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
        )

        # Inference side-chain: resize -> float32 cast+scale -> HWC->CHW+batch -> InferenceOp ->
        # postprocess.
        resize = graph.op(
            holoscan_vision.operators.ResizeOp,
            "resize",
            width=960,
            height=544,
            max_batch_size=1,
        )
        graph.add_flow(
            image_shift.output,
            resize.input,
            queue_depth=1,
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
            queue_depth=1,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
        )

        format_input = graph.op(FormatInferenceInputOp, "transpose")
        graph.add_flow(
            convert.output,
            format_input.input,
            queue_depth=1,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
        )

        inference = graph.op(
            "holoscan.inference.InferenceOp",
            "inference",
            model_path_map={"face_detect": engine},
            input_map={"face_detect": ["preprocessed"]},
            output_map={"face_detect": ["scores", "boxes"]},
            backend="trt",
            input_on_cuda=True,
            is_engine_path=False,
            enable_fp16=True,
        )
        graph.add_flow(
            format_input.output,
            inference.preprocessed,
            queue_depth=1,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
        )

        postprocessor = graph.op(
            PostprocessorOp,
            "postprocessor",
            iou_threshold=0.15,
            score_threshold=0.5,
            box_scale=35.0,
            box_offset=0.5,
            grid_height=34,
            grid_width=60,
            image_width=960,
            image_height=544,
        )
        graph.add_flow(
            inference.scores,
            postprocessor.in_scores,
            queue_depth=1,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
        )
        graph.add_flow(
            inference.boxes,
            postprocessor.in_boxes,
            queue_depth=1,
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
                    "port_name": "person",
                    "kind": "geometry",
                    "topologies": ["rectangle_list"],
                    "max_primitives": _MAX_OVERLAY_BOXES,
                    "default_line_width": 4,
                },
                {
                    "port_name": "faces",
                    "kind": "geometry",
                    "topologies": ["rectangle_list"],
                    "max_primitives": _MAX_OVERLAY_BOXES,
                    "default_line_width": 4,
                },
            ],
            sink=sink,
            trigger={"kind": "layer", "layer": 0},
            max_input_age_ns=int(_OVERLAY_MAX_AGE.total_seconds() * 1e9),
        )
        graph.add_flow(
            image_shift.output,
            viz.video,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
            queue_depth=1,
        )
        graph.add_flow(
            postprocessor.person,
            viz.person,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
            queue_depth=1,
        )
        graph.add_flow(
            postprocessor.faces,
            viz.faces,
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
            output_port="scores",
            device=hs.execution.DeviceId(cuda_device_ordinal),
        )
        options.deployment.bind_tensor_output_device(
            operator_path="inference",
            output_port="boxes",
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
            f"Capturing mode={camera_mode} {frame_width}x{frame_height} -- running (Ctrl+C to stop)..."
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
    parser.add_argument(
        "--camera-mode",
        type=int,
        default=0,
        help="IMX274 mode (default: 0 = 3840x2160@60fps)",
    )
    parser.add_argument(
        "--expander-configuration",
        type=int,
        default=0,
        choices=(0, 1),
        help="I2C Expander configuration",
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
    parser.add_argument(
        "--hololink",
        default="192.168.0.2",
        help="IP address of Hololink board",
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
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    return run(
        args.hololink,
        args.camera_mode,
        args.expander_configuration,
        args.ibv_name,
        args.ibv_port,
        args.engine,
        args.frame_limit,
        args.headless,
        args.fullscreen,
    )


if __name__ == "__main__":
    sys.exit(main())
