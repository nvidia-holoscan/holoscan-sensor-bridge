# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Depth Anything V2 post-processing (PostprocessorOp), used by linux_depth_anything_v2_imx477.py,
# and the standalone holohub application this file was adapted from
# (https://github.com/nvidia-holoscan/holohub/tree/main/applications/depth_anything_v2/).
#
# Python port of this file's previous (HSDK4-era) version to the HSDK5 Python bindings.
# PostprocessorOp keeps its HSDK4 depth-map normalization, OpenCV colormap and per-mode image
# composition. What changed:
#   - HSDK4 cycled the display mode on a left click and dragged the "interactive" split with the
#     middle or right mouse button, through HolovizOp's mouse_button/cursor_pos/framebuffer_size
#     callbacks. HSDK5's HolovizOp (holoscan_visualization) has no such callbacks: its catalog row
#     takes only layers/sink/trigger/max_input_age_ns, and it never registers the HoloViz window's
#     input callbacks. The display mode and the split ratio are therefore PostprocessorOp
#     parameters, fixed for a run (the IMX477 app's --display-mode and --split-ratio), defaulting
#     to HSDK4's initial "depth" mode and 0.5 ratio.
#   - The display-mode text still shows, on a HolovizOp text layer: PostprocessorOp emits one text
#     record, white, with its top-left corner at (0.025, 0.9) of the window, as HSDK4's text
#     InputSpec placed it.
#   - The depth map and the image it was computed from are paired by capture time (OnAligned).
#     HSDK4 ran the graph in lockstep on one thread. In HSDK5 the IMX477 app's
#     FormatInferenceInputOp passes each image it sends to the inference on to PostprocessorOp,
#     where it waits for its depth map.
#   - The output image is written into a runtime-pool CUDA tensor (an output loan) instead of
#     being emitted as a CuPy array; see linux_depth_anything_v2_imx477.py's header comment.
#
# OpenCV (pip install opencv-python-headless) is imported where the colormap is applied, not at
# module level, so that importing this module (as linux_depth_anything_v2_imx477.py does, also
# for --help) does not need it.
#
# The standalone holohub application (main() below) is not ported: it reads a V4L2 camera
# (V4L2VideoCaptureOp) or a recorded video (VideoStreamReplayerOp), and Holoscan SDK 5 provides
# neither (V4L2 capture moved to the separate Holoscan Camera project; there is no replayer).
# main() says so and exits. depth.yaml, that application's configuration, is kept.

import struct
import sys
import threading
import time
from datetime import timedelta

import cupy as cp
import holoscan as hs
import numpy as np

# HSDK4's display modes, in its click order; it started in "depth".
DISPLAY_MODES = ("original", "depth", "side-by-side", "interactive")
DEFAULT_DISPLAY_MODE = "depth"

# Depth Anything V2 takes 518x518 images and returns a 518x518 depth map.
MODEL_INPUT_SIZE = 518

# The text layer's glyph budget: the longest display-mode name.
MAX_TEXT_GLYPHS = max(len(mode) for mode in DISPLAY_MODES)

# A depth map has exactly the capture time of its image: FormatInferenceInputOp emits the image
# and the model input with the frame's capture time, and InferenceOp stamps its output with its
# input's.
_SAME_FRAME = timedelta(0)

_UINT8 = hs.tensor.DLDataType(hs.tensor.DLDataTypeCode.DLUINT, 8, 1)

# The largest output image: side-by-side, two model-size RGB images next to each other.
_MAX_OUTPUT_BYTES = MODEL_INPUT_SIZE * 2 * MODEL_INPUT_SIZE * 3

# HolovizOp's TextRunHeader: float32 x, y, size_px, then uint32 color (0xAABBGGRR), 16 bytes; the
# run's text follows it, NUL-padded.
_TEXT_HEADER = struct.Struct("<fffI")
# HSDK4's display-mode text: top-left corner at (0.025, 0.9) of the window, opaque white, and
# HoloViz's default text height, 5% of the window height (the window is MODEL_INPUT_SIZE tall;
# fullscreen keeps the size in pixels).
_TEXT_POSITION = (0.025, 0.9)
_TEXT_SIZE_PX = 0.05 * MODEL_INPUT_SIZE
_TEXT_COLOR = 0xFFFFFFFF


def derive_from(metadata):
    """Emit options that carry a received sample's capture time, frame ID, trace and flags over
    to a value computed from it, like the C++ holoscan::sensor_io::derive_from(). PostprocessorOp
    and HolovizOp match samples by capture time."""
    provenance = hs.sensor_io.ProvenanceAccumulator()
    provenance.observe(metadata)
    return provenance.emit_options(hs.sensor_io.CaptureTimePolicy.EARLIEST)


def writable_view(tensor, dtype):
    """A CuPy array over a CUDA output loan's tensor, to fill it on the activation's stream (the
    way the SDK's holoviz_cupy_pattern example fills its loans). Use it only inside the loan's
    `with` block."""
    memory = cp.cuda.UnownedMemory(
        tensor.data, tensor.nbytes, tensor, tensor.device.device_id
    )
    return cp.ndarray(tensor.shape, dtype, cp.cuda.MemoryPointer(memory, 0))


def _text_record(text):
    """One HolovizOp text-layer row showing `text` the way HSDK4 drew the display mode: a host
    uint8 {1, 16 + MAX_TEXT_GLYPHS} array, TextRunHeader then the NUL-padded text."""
    encoded = text.encode("ascii")
    record = np.zeros((1, _TEXT_HEADER.size + MAX_TEXT_GLYPHS), dtype=np.uint8)
    record[0, : _TEXT_HEADER.size] = np.frombuffer(
        _TEXT_HEADER.pack(*_TEXT_POSITION, _TEXT_SIZE_PX, _TEXT_COLOR), dtype=np.uint8
    )
    record[0, _TEXT_HEADER.size : _TEXT_HEADER.size + len(encoded)] = np.frombuffer(
        encoded, dtype=np.uint8
    )
    return record


def _apply_colormap(depth_normalized):
    """OpenCV's JET colormap of a [0, 255] depth map (on the device); returns a host BGR image,
    as HSDK4 computed it."""
    import cv2  # here, not at module level: see this file's header comment

    return cv2.applyColorMap(depth_normalized.get().astype("uint8"), cv2.COLORMAP_JET)


class InferencePacing:
    """One inference at a time, as HSDK4 ran them in lockstep: FormatInferenceInputOp sends a frame
    to InferenceOp only once PostprocessorOp has used the previous frame's depth map, and skips
    the camera frames that come meanwhile. InferenceOp only queues GPU work, so without this the
    inferences would queue up at camera rate: the display would lag several inferences behind,
    and the output pools would run out. Frames are identified by capture time. A frame whose depth
    map hasn't been used after RELEASE_AFTER_S frees its place: a slow first inference, or a
    depth map a full queue dropped. That is longer than an inference should take: a frame released
    while its inference still runs lets the next one overlap it."""

    RELEASE_AFTER_S = 2.0

    _lock = threading.Lock()
    # The capture time (ns) of each frame sent to InferenceOp -> when it was sent.
    _in_flight = {}

    @classmethod
    def try_send(cls, capture_time):
        """True, counting the frame in flight, if the frame with this capture time may go to
        InferenceOp now."""
        now = time.monotonic()
        with cls._lock:
            for frame, sent in list(cls._in_flight.items()):
                if now - sent > cls.RELEASE_AFTER_S:
                    del cls._in_flight[frame]
            if cls._in_flight:
                return False
            cls._in_flight[capture_time] = now
            return True

    @classmethod
    def done(cls, capture_time):
        """The frame with this capture time is done with: its depth map was used, or it wasn't
        sent. A frame released earlier, whose depth map was used late, leaves the frame sent
        after it in flight."""
        with cls._lock:
            for frame in [frame for frame in cls._in_flight if frame <= capture_time]:
                del cls._in_flight[frame]


class PostprocessorOp(hs.Operator):
    """Operator that does postprocessing before sending resulting image to Holoviz

    input:  "input_depthmap" - the depth map InferenceOp computed
    input:  "input_image"    - the RGB image (float32 in [0, 1]) the depth map was computed from
    output: "output_image"   - the RGB8 image for Holoviz to display
    output: "output_text"    - Holoviz text-layer record showing the current display mode

    This operator's output image depends on the display mode, if set to

        * "original": output the original image from input source
        * "depth": output the color depthmap based on the depthmap returned from
                   Depth Anything V2 model
        * "side-by-side": output a side-by-side view of the original image next to
                          the color depthmap
        * "interactive": show the original image left of the split_ratio point and the color
                         depthmap right of it (HSDK4 moved the split with the mouse; here it is
                         fixed for the run)
    """

    input_depthmap = hs.Input("input_depthmap", hs.payload.Tensor)
    input_image = hs.Input("input_image", hs.schema.Image)
    output_image = hs.Output("output_image", hs.schema.Image)
    output_text = hs.Output("output_text", hs.payload.Tensor)

    display_mode = hs.Param(DEFAULT_DISPLAY_MODE, allowed=DISPLAY_MODES)
    # In interactive mode, how much of the original video to show
    split_ratio = hs.Param(0.5, range=(0.0, 1.0))

    def setup(self, spec):
        spec.input(self.input_depthmap, queue_depth=1)
        # Only the images sent to the inference come: one at a time, two after InferencePacing
        # released a frame.
        spec.input(self.input_image, queue_depth=2)
        spec.output(self.output_image, max_emits_per_compute=1).produces_tensor(
            hs.tensor.OutputSpec(
                storage="runtime_pool",
                memory_kind=hs.tensor.MemoryKind.CUDA_DEVICE,
                dtype=_UINT8,
                rank=3,
                strides=hs.tensor.TensorStridePolicy.CONTIGUOUS_ONLY,
            ),
            hs.tensor.byte_span(_MAX_OUTPUT_BYTES),
            field="data",
        )
        spec.output(self.output_text)

    def contract(self):
        return (
            hs.Contract()
            .trigger(
                hs.OnAligned(
                    ports=[self.input_depthmap, self.input_image], window=_SAME_FRAME
                )
            )
            .input(self.input_depthmap, hs.Exact())
            .input(self.input_image, hs.Exact())
        )

    def normalize(self, depth_map):
        min_value = cp.min(depth_map)
        max_value = cp.max(depth_map)
        normalized = (depth_map - min_value) / (max_value - min_value)
        return 255 - (normalized * 255)

    def compute(self, context):
        depthmap_sample = self.input_depthmap.receive()
        image_sample = self.input_image.receive()
        image = image_sample.data
        capture_time = image_sample.metadata.capture_timestamp_ns
        output_width = image.width * (2 if self.display_mode == "side-by-side" else 1)
        # Both outputs carry the frame's capture time: HolovizOp draws the image and the text it
        # gets with one capture time together.
        options = derive_from(image_sample.metadata)
        try:
            output = self.output_image.allocate_tensors(
                hs.schema.Image(
                    height=image.height,
                    width=output_width,
                    encoding=hs.schema.image.ImageEncoding.RGB8,
                ),
                {
                    "data": hs.tensor.LoanRequest(
                        shape=(image.height, output_width, 3), dtype=_UINT8
                    )
                },
                options=options,
            )
        except hs.errors.ResourceExhaustedError:
            # HolovizOp still holds every output image: skip this frame, and let the next one go
            # to InferenceOp.
            InferencePacing.done(capture_time)
            return

        with cp.cuda.Device(image.data.device.device_id), context.stream():
            depth_map = cp.from_dlpack(depthmap_sample.data).squeeze()
            composed = self._compose(cp.from_dlpack(image.data), depth_map)
            with output:
                writable_view(output.tensors["data"], cp.uint8)[...] = composed
            # This stream waited for the depth map: once it is done, so is the inference, and the
            # next frame may go (see InferencePacing).
            cp.cuda.get_current_stream().synchronize()
        InferencePacing.done(capture_time)

        # Sent after the image: the text drives HolovizOp's rendering.
        self.output_text.emit(
            hs.Tensor.as_tensor(_text_record(self.display_mode)), options=options
        )

    def _compose(self, image, depth_map):
        """The output image for the display mode, from the [0, 1] RGB image and the depth map."""
        if self.display_mode == "original":
            # Display the original image
            return (image * 255).astype(cp.uint8)

        # cv2's colormap is BGR and is displayed as RGB, as in HSDK4 (and holohub); swapping the
        # channels would change the displayed color scale.
        depth_colormap = cp.asarray(_apply_colormap(self.normalize(depth_map)))
        if self.display_mode == "depth":
            # Display the color depthmap
            return depth_colormap

        image = (image * 255).astype(cp.uint8)
        if self.display_mode == "side-by-side":
            # Display both original and color depthmap images side-by-side
            return cp.hstack((image, depth_colormap))

        # Interactive mode
        pos = int(image.shape[1] * self.split_ratio)
        return cp.hstack((image[:, :pos, :], depth_colormap[:, pos:]))


def main():
    """The standalone holohub application: Depth Anything V2 on a V4L2 camera or a
    recorded video, configured by depth.yaml. It cannot run on Holoscan SDK 5; see this
    file's header comment."""
    print(
        "depth.py: the standalone Depth Anything V2 application needs a V4L2 camera "
        "(V4L2VideoCaptureOp) or video replayer (VideoStreamReplayerOp) source operator, which "
        "Holoscan SDK 5 does not provide. Run linux_depth_anything_v2_imx477.py instead.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
