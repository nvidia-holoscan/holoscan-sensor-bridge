# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Minimal stereo IMX274 player on the hololink_module V1 surface, using the reconnection
# framework (HsbControllerOp/SensorFactory/SensorDevice). Python port of this repo's own
# previous (pre-HSDK5) Python version of this app, via the HSDK5 Python bindings -- there is
# no .cpp sibling. HsbControllerOp's constructor takes live objects, which graph.op() passes as
# constructor handles: metadata=, sensor_factory= and network_receiver_factory=.
#
# Each IMX274 leg runs a self-healing capture pipeline (HsbControllerOp -> CsiToBayerOp ->
# ImageProcessorOp -> BayerDemosaicOp); both legs free-run independently (IMX274 cannot
# hardware-sync) and are shown side by side in one Holoviz window. The player starts
# immediately without waiting for any device: it shows the SMPTE test pattern and connects
# each leg when its device announces itself. If a leg's device is later lost, its device state
# is invalidated, the test pattern returns, and it reconnects on re-announcement -- the two
# legs recover independently.
#
#   HsbControllerOp -> FrameAlignerOp -> CsiToBayerOp(module) -> ImageProcessorOp(module) ->
#       BayerDemosaicOp -> HolovizOp
#
# The two legs are the two data channels of one stereo (dual-sensor) board -- each data
# channel enumerates under its own peer IP. Run it with those two IPs:
#
#   python reconnecting_imx274_player.py --channel-ips 192.168.0.2,192.168.0.3
#
# Add --frame-limit N to exit after N frames per leg (otherwise it runs until Ctrl-C), and
# --transport linux to use the software receiver instead of RoCE.
#
# FrameAlignerOp pairs the two legs' frames using hs.OnAligned, not a hand-rolled timestamp
# cache like the pre-port original: BaseReceiverOp(module)'s own compute() (shared by every
# receiver, including HsbControllerOp) already stamps each emitted tensor/metadata pair's
# capture_time from the graph's own clock (context.activation_time(), confirmed by reading
# base_receiver_op.inl directly) -- the same mechanism that already makes every other port's
# in_tensor/in_metadata OnAligned pairing work. Two independently free-running legs sharing one
# graph clock are therefore already comparable on that same basis, with no FPGA metadata
# involved: aligning on it is a strictly more general version of the old per-leg
# left_timestamp_s/right_timestamp_s dt-check (real-valued even for a fallback frame, which
# carries no FPGA timestamp at all -- removing that check's own dedicated zero-timestamp
# special case). A 4-port OnAligned (tensor+metadata for each leg, not just the two tensors) is
# used so a skew-rejected group drops all four consistently -- otherwise an orphaned metadata
# sample could pair with the wrong tensor downstream, at each leg's own CsiToBayerOp.
#
# The old app's PassThroughOperator ("rename", solely there to key BayerDemosaicOp's unnamed
# output tensor by "left"/"right" for HolovizOp's old dict-based InputSpec lookup) is dropped
# entirely, not ported -- HolovizOp's new per-layer named ports (video_left/video_right) make
# it unnecessary, the same simplification already applied to every other stereo app ported
# this session (e.g. linux_dual_inference_body_pose_imx477.py's dropped RenameTensorsOp).
#
# Unlike module_linux_imx274_player.py (this repo's own single-camera module IMX274
# reference), sensor geometry here is resolved from fixed per-mode constants, not a live
# camera query: the whole point of the reconnection framework is that the graph is built
# before any device exists to query -- matching the pre-port original's own
# Imx274SensorFactory.__init__, which hardcodes IMX274_MODE_1920X1080_60FPS's own
# width/height/pixel_format/bayer_format rather than reading them off a camera object, and
# raises for any other mode.
#
# --frame-limit has no HSDK5 CountCondition equivalent, so it's implemented with a small native
# Python operator, FrameLimitOp, tapping each leg's controller tensor output directly (before
# the aligner, counting every served frame -- live or fallback -- matching the old app's own
# CountCondition placement directly on HsbControllerOp); the one that brings the last leg to the
# limit stops the run from the graph, so a run that ends first, for example by failing, doesn't
# leave the app waiting.
#
# Usage:
#   reconnecting_imx274_player.py [OPTIONS]
#
# Options:
#   --channel-ips IPS       Comma-separated peer IPs of the two data channels (default:
#                           192.168.0.2,192.168.0.3)
#   --camera-mode N         IMX274 mode (default: 1 = 1920x1080 60fps; no other mode supported)
#   --frame-limit N         Exit after this many frames per leg (default: run until Ctrl-C)
#   --transport {roce,linux}  Data-plane transport (default: roce)
#   --headless              Run without opening a window (still renders, presents nowhere)
#   --module-dir PATH       Override the directory containing hololink_<UUID>.so files
#   --log-level N           Python logging level (default: 20 = INFO)

import argparse
import logging
import os
import signal
import sys
import threading
from datetime import timedelta

import capture_metadata
import cupy as cp
import hololink_module.operators
import hololink_module.sensors.imx274
import holoscan as hs
import holoscan_vision as hv
import holoscan_visualization as hz  # noqa: F401 -- registers "holoscan.visualization.HolovizOp"
import numpy as np
import PIL.Image
from cuda.bindings import driver as cuda_driver
from hololink_module.sensors import csi
from hololink_module.sensors.imx274 import imx274_mode

import hololink_module

# Largest skew FrameAlignerOp tolerates between the two legs. IMX274 cannot hardware-sync, so
# the acceptance window is one full frame period.
ALLOWABLE_DT_S = 1.0 / 60

# The two legs are two data channels of one board, so they share a control plane: board
# bring-up (start()/reset()) must run once per board, not once per leg, or the second leg's
# reset would disrupt the first mid-stream. The first leg to reach new_sensor brings the board
# up; the second sees it already up. A reconnect yields a fresh HololinkInterfaceV1
# (device_lost() invalidated the old one), so identity membership resets on its own. A plain
# set, not a WeakSet: the nanobind class is not weak-referenceable (weakref.ref() raises), so
# invalidated handles stay in the set; one small object per reconnect.
_board_lock = threading.Lock()
_boards_up = set()


def _cuda_check(result):
    err = result[0]
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        _, description = cuda_driver.cuGetErrorString(err)
        raise RuntimeError(f"CUDA error: {description}")
    if len(result) == 2:
        return result[1]
    return result[1:]


class Imx274SensorDevice(hololink_module.SensorDevice):
    """Wraps one armed Imx274Cam; replaced by a fresh one on reconnect."""

    def __init__(self, camera):
        super().__init__()
        self._camera = camera

    def stop_sensor(self):
        try:
            self._camera.stop()
        except Exception:
            logging.info("stop_sensor: camera.stop() failed (device likely gone).")
        return hololink_module.HOLOLINK_MODULE_OK


class Imx274SensorFactory(hololink_module.SensorFactory):
    """Sensor-side reconnection policy for one IMX274 leg. new_sensor brings the board up and
    arms a fresh camera on every (re)connect."""

    def __init__(self, instance, camera_mode):
        super().__init__()
        self._instance = instance
        self._camera_mode = camera_mode
        self._sensor = None  # keeps the current SensorDevice's Python side alive
        self._sensor_count = 0
        self._fallback_ptr = 0
        self._fallback_size = 0
        if camera_mode == imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS:
            self._height, self._width = 1080, 1920
            self._pixel_format = csi.PixelFormat.RAW_10
            self._bayer_format = csi.BayerFormat.RGGB
        else:
            raise RuntimeError(f"Unsupported {camera_mode=}")

    def image_size(self):
        return self._height, self._width

    def image_format(self):
        return self._pixel_format, self._bayer_format

    def set_fallback_frame(self, ptr, size):
        self._fallback_ptr = ptr
        self._fallback_size = size

    def fallback_frame(self):
        # The image shown when no live frame is available -- the SMPTE test bars at startup
        # (before the first connect) and during an outage. A device pointer built up front;
        # touches no live device handles.
        return (self._fallback_ptr, self._fallback_size)

    def configure_converter(self, converter):
        # Train the converter to this sensor's CSI layout without a live camera (none exists at
        # compose time; the board comes up in new_sensor), as Imx274Cam.configure_converter()
        # does: 175 bytes of metadata, then 8 (RAW_10) or 16 (RAW_12) optical-black lines.
        pixel_format = self._pixel_format.value
        start_byte = converter.receiver_start_byte()
        transmitted_line_bytes = converter.transmitted_line_bytes(
            pixel_format, self._width
        )
        received_line_bytes = converter.received_line_bytes(transmitted_line_bytes)
        start_byte += converter.received_line_bytes(175)
        optical_black_lines = 16 if self._pixel_format == csi.PixelFormat.RAW_12 else 8
        start_byte += received_line_bytes * optical_black_lines
        converter.configure(
            start_byte, received_line_bytes, self._width, self._height, pixel_format
        )

    def new_sensor(self, metadata):
        # Reactor thread. A fresh Imx274Cam(metadata) re-get_service()s the board's V1
        # handles, so it resyncs with post-reset module state. Report a failed bring-up (e.g.
        # the board is gone) by returning None so the framework invalidates the board and
        # retries on re-announce.
        try:
            hololink_dev = hololink_module.HololinkInterfaceV1.get_service(metadata)
            with _board_lock:
                if hololink_dev not in _boards_up:
                    hololink_dev.start()
                    hololink_dev.reset()
                    _boards_up.add(hololink_dev)
            hololink_module.sensors.imx274.Imx274Cam.use_expander_configuration(
                metadata, self._instance
            )
            camera = hololink_module.sensors.imx274.Imx274Cam(metadata)
            camera.configure(self._camera_mode)
            camera.set_digital_gain_reg(0x4)
            camera.start()
        except Exception as e:
            logging.info(f"new_sensor: bring-up failed ({e}); awaiting re-announce.")
            return None
        self._sensor_count += 1
        self._sensor = Imx274SensorDevice(camera)
        return self._sensor


# CSI-frame synthesis for the fallback image: render an image file into a sensor-format CSI
# frame (a device-side cupy array). Only the IMX274 RAW_10 / RGGB path is implemented.


def _encode_raw_10_bayer_image(bayer_image):
    # RAW10 (IMX274 spec): for each four 10-bit values c0..c3, emit their high 8 bits, then a
    # fifth byte packing the four low-2-bit remainders.
    bayer_height, bayer_width = bayer_image.shape
    ri = bayer_image.ravel()
    upper_byte = (ri >> 2).astype(np.uint8)
    lower_byte = (ri & 0x3).astype(np.uint8)
    combined_lower_bytes = (
        lower_byte[::4]
        | (lower_byte[1::4] << 2)
        | (lower_byte[2::4] << 4)
        | (lower_byte[3::4] << 6)
    )
    raw_10 = np.stack(
        [
            upper_byte[::4],
            upper_byte[1::4],
            upper_byte[2::4],
            upper_byte[3::4],
            combined_lower_bytes,
        ],
        axis=1,
    )
    return raw_10.reshape(bayer_height, bayer_width * 5 // 4)


def _encode_bayer_image(rgb_image, bayer_format):
    sr = rgb_image[:, :, 0]
    sg = rgb_image[:, :, 1]
    sb = rgb_image[:, :, 2]
    dtype = cp.uint16
    height, width, _ = rgb_image.shape
    bayer_height, bayer_width = height, width
    if bayer_format != csi.BayerFormat.RGGB:
        raise Exception(f"Unexpected bayer format={bayer_format}")
    # upper_line is red0, green0, red1, green1, ...
    r, g = sr.ravel(), sg.ravel()
    if r.size != g.size:
        raise RuntimeError("Red and green planes hold different numbers of pixels.")
    c = cp.empty(((r.size + g.size) // 2,), dtype=dtype)
    c[0::2] = r[::2]
    c[1::2] = g[::2]
    upper_line = c.reshape(height, bayer_width)[::2]  # every other line
    # lower_line is green0, blue0, green1, blue1, ...
    g, b = sg.ravel(), sb.ravel()
    c = cp.empty(((g.size + b.size) // 2,), dtype=dtype)
    c[0::2] = g[::2]
    c[1::2] = b[::2]
    lower_line = c.reshape(height, bayer_width)[::2]  # every other line
    return cp.stack([upper_line, lower_line], axis=1).reshape(bayer_height, bayer_width)


def _bayer_image_to_csi(pixel_format, bayer_image, start_byte, received_line_bytes):
    if pixel_format != csi.PixelFormat.RAW_10:
        raise Exception(f"Unexpected {pixel_format=}")
    csi_image = _encode_raw_10_bayer_image(bayer_image)
    rows, _ = csi_image.shape
    expected_lines = cp.resize(csi_image, (rows, received_line_bytes))
    start_metadata = cp.empty((start_byte,), dtype=cp.uint8)
    return cp.concatenate((start_metadata, expected_lines.ravel()))


def _make_csi_from_image_file(
    height, width, pixel_format, bayer_format, start_byte, received_line_bytes, filename
):
    logging.getLogger("PIL").setLevel(logging.WARN)
    image = (
        PIL.Image.open(filename)
        .convert("RGB")
        .resize((width, height), PIL.Image.LANCZOS)
    )
    uint16_image = cp.array(image, dtype=np.uint16)  # only 8 bits are set
    if pixel_format != csi.PixelFormat.RAW_10:
        raise Exception(f"Unsupported {pixel_format=} for the fallback image")
    uint16_image[:] <<= 2  # use 10 bits
    rgb_image = uint16_image.reshape(height, width, 3)
    bayer_image = _encode_bayer_image(rgb_image, bayer_format)
    return _bayer_image_to_csi(
        pixel_format, bayer_image, start_byte, received_line_bytes
    )


class _FrameLimit:
    """Shared state for --frame-limit: every leg must reach the limit before FrameLimitOp stops
    the run."""

    lock = threading.Lock()
    counts = {}
    limit = 0

    @classmethod
    def reset(cls, leg_count, limit):
        with cls.lock:
            cls.counts = {i: 0 for i in range(leg_count)}
            cls.limit = limit

    @classmethod
    def record(cls, leg):
        """Counts a frame; True once every counter has reached the limit."""
        with cls.lock:
            cls.counts[leg] += 1
            return cls._done_locked()

    @classmethod
    def _done_locked(cls):
        return cls.limit > 0 and all(
            count >= cls.limit for count in cls.counts.values()
        )


class FrameLimitOp(hs.Operator):
    """Count one leg's served frames toward --frame-limit; see _FrameLimit."""

    input = hs.Input("input", hs.payload.Tensor)
    leg = hs.Param(0)

    def setup(self, spec):
        spec.input(self.input, queue_depth=4)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, context):
        self.input.receive_data()
        if _FrameLimit.record(self.leg):
            context.request_stop()


class FrameAlignerOp(hs.Operator):
    """Pair the two legs' tensor+metadata within ALLOWABLE_DT_S; see module docstring for why
    this uses hs.OnAligned rather than a hand-rolled timestamp cache."""

    tensor_left = hs.Input("tensor_left", hs.payload.Tensor)
    metadata_left = hs.Input("metadata_left", capture_metadata.CaptureMetadata)
    tensor_right = hs.Input("tensor_right", hs.payload.Tensor)
    metadata_right = hs.Input("metadata_right", capture_metadata.CaptureMetadata)
    output_tensor_left = hs.Output("output_tensor_left", hs.payload.Tensor)
    output_metadata_left = hs.Output(
        "output_metadata_left", capture_metadata.CaptureMetadata
    )
    output_tensor_right = hs.Output("output_tensor_right", hs.payload.Tensor)
    output_metadata_right = hs.Output(
        "output_metadata_right", capture_metadata.CaptureMetadata
    )

    def setup(self, spec):
        spec.input(self.tensor_left, queue_depth=4)
        spec.input(self.metadata_left, queue_depth=4)
        spec.input(self.tensor_right, queue_depth=4)
        spec.input(self.metadata_right, queue_depth=4)
        spec.output(self.output_tensor_left)
        spec.output(self.output_metadata_left)
        spec.output(self.output_tensor_right)
        spec.output(self.output_metadata_right)

    def contract(self):
        window = timedelta(seconds=ALLOWABLE_DT_S)
        return (
            hs.Contract()
            .trigger(
                hs.OnAligned(
                    ports=[
                        self.tensor_left,
                        self.metadata_left,
                        self.tensor_right,
                        self.metadata_right,
                    ],
                    window=window,
                )
            )
            .input(self.tensor_left, hs.Exact())
            .input(self.metadata_left, hs.Exact())
            .input(self.tensor_right, hs.Exact())
            .input(self.metadata_right, hs.Exact())
        )

    def compute(self, context):
        self.output_tensor_left.forward(self.tensor_left.receive())
        self.output_metadata_left.forward(self.metadata_left.receive())
        self.output_tensor_right.forward(self.tensor_right.receive())
        self.output_metadata_right.forward(self.metadata_right.receive())


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


def _network_receiver_factory(transport):
    if transport == "roce":
        return hololink_module.make_roce_network_receiver_factory()
    if transport == "linux":
        return hololink_module.make_linux_network_receiver_factory()
    raise ValueError(f"Unknown transport {transport!r} (expected 'roce' or 'linux')")


def _build_leg(graph, name, index, cu_context, channel, aligner, fallback_buffers):
    """Build one leg's SensorFactory + fallback image + HsbControllerOp, wiring its tensor/
    metadata into the shared aligner's per-leg input ports. Returns (view_dict_inputs, the
    sensor factory, the controller's own operator path for --frame-limit tapping)."""
    metadata = channel["metadata"]
    factory = channel["sensor_factory"]

    height, width = factory.image_size()
    pixel_format, bayer_format = factory.image_format()
    bayer_name = bayer_format.name.lower()
    # The sensor's CSI layout (metadata preamble and optical-black lines), not just its image
    # size: the receiver's frame size and CsiToBayerOp's geometry come from it.
    converter = hololink_module.CsiToBayerConverter()
    factory.configure_converter(converter)
    geometry = converter.geometry()
    frame_size = converter.get_csi_length()

    # Fallback SMPTE bars shown while disconnected (at startup before the first connect and
    # during an outage): render the CSI frame once, size it into a frame_size device buffer,
    # and hand its pointer to the factory. HsbControllerOp serves it whenever no live frame is
    # available. Kept alive on fallback_buffers for the run.
    here = os.path.dirname(__file__)
    csi_image = _make_csi_from_image_file(
        height,
        width,
        pixel_format,
        bayer_format,
        geometry["start_byte"],
        geometry["bytes_per_line"],
        filename=os.path.join(here, "SMPTE_Color_Bars.png"),
    )
    fallback_buffer = cp.zeros((frame_size,), dtype=cp.uint8)
    flat = csi_image.astype(cp.uint8).ravel()
    n = min(int(flat.size), frame_size)
    fallback_buffer[:n] = flat[:n]
    fallback_buffers.append(fallback_buffer)
    factory.set_fallback_frame(int(fallback_buffer.data.ptr), frame_size)

    controller = graph.op(
        hololink_module.operators.HsbControllerOp,
        f"controller_{name}",
        metadata=metadata,
        frame_context=int(cu_context),
        sensor_factory=factory,
        network_receiver_factory=channel["network_receiver_factory"],
        frame_size=frame_size,
    )
    graph.add_flow(controller.tensor, getattr(aligner, f"tensor_{name}"), queue_depth=4)
    graph.add_flow(
        controller.metadata, getattr(aligner, f"metadata_{name}"), queue_depth=4
    )

    csi_to_bayer = graph.op(
        hololink_module.operators.CsiToBayerOp,
        f"csi_to_bayer_{name}",
        cuda_device_ordinal=0,
        geometry=geometry,
        bayer_format=bayer_name,
    )
    graph.add_flow(
        getattr(aligner, f"output_tensor_{name}"), csi_to_bayer.in_tensor, queue_depth=4
    )
    graph.add_flow(
        getattr(aligner, f"output_metadata_{name}"),
        csi_to_bayer.in_metadata,
        queue_depth=4,
    )

    image_processor = graph.op(
        hololink_module.operators.ImageProcessorOp,
        f"image_processor_{name}",
        pixel_format=pixel_format.value,
        pixel_width=geometry["pixel_width"],
        pixel_height=geometry["pixel_height"],
        bayer_format=bayer_format.value,
        optical_black=50,  # IMX274 optical black
        cuda_device_ordinal=0,
    )
    graph.add_flow(csi_to_bayer.tensor, image_processor.in_tensor, queue_depth=4)
    graph.add_flow(csi_to_bayer.metadata, image_processor.in_metadata, queue_depth=4)

    demosaic = graph.op(
        hv.operators.BayerDemosaicOp,
        f"bayer_demosaic_{name}",
        bayer_grid_pos=bayer_name,
        generate_alpha=True,
        alpha_value=65535,
    )
    graph.add_flow(image_processor.tensor, demosaic.input, queue_depth=4)

    leg_tensor_operators = [
        f"controller_{name}",
        f"csi_to_bayer_{name}",
        f"image_processor_{name}",
    ]
    return controller, demosaic, leg_tensor_operators


def run(
    channel_ips,
    camera_mode=imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
    frame_limit=None,
    headless=False,
    transport="roce",
    module_dir=None,
):
    """Build and run the stereo reconnection player."""
    if len(channel_ips) < 2:
        raise ValueError("stereo needs two channel IPs")

    cuda_device_ordinal = 0
    tensor_names = ["left", "right"]

    print("Initializing CUDA...")
    _cuda_check(cuda_driver.cuInit(0))
    cu_device = _cuda_check(cuda_driver.cuDeviceGet(cuda_device_ordinal))
    cu_context = _cuda_check(cuda_driver.cuDevicePrimaryCtxRetain(cu_device))
    cuda_driver.cuCtxSetCurrent(cu_context)

    adapter = hololink_module.Adapter.get_adapter()
    if module_dir:
        adapter.set_module_directory(module_dir)

    # Don't block on discovery. The reconnection framework registers the peer IP and connects
    # whenever the device announces itself, so the player starts immediately -- showing the
    # SMPTE test pattern until a device appears (and again during any later outage). The
    # startup metadata only needs the peer IP; every other field comes from the device's
    # announcement.
    channels = []
    for index, channel_ip in enumerate(channel_ips[:2]):
        metadata = hololink_module.EnumerationMetadata()
        metadata["peer_ip"] = channel_ip
        channels.append(
            {
                "metadata": metadata,
                "sensor_factory": Imx274SensorFactory(index, camera_mode),
                "network_receiver_factory": _network_receiver_factory(transport),
            }
        )

    if frame_limit:
        _FrameLimit.reset(len(channels), frame_limit)

    # Build the graph.
    graph = hs.Graph("reconnecting_imx274_player")

    frame_aligner = graph.op(FrameAlignerOp, "frame_aligner")

    layers = []
    demosaics = []
    tensor_output_operators = []
    fallback_buffers = (
        []
    )  # keeps every leg's fallback GPU buffer alive for the whole run
    for index, (name, channel) in enumerate(zip(tensor_names, channels)):
        controller, demosaic, leg_tensor_operators = _build_leg(
            graph, name, index, cu_context, channel, frame_aligner, fallback_buffers
        )
        demosaics.append(demosaic)
        tensor_output_operators.extend(leg_tensor_operators)

        if frame_limit:
            frame_limit_op = graph.op(FrameLimitOp, f"frame_limit_{name}", leg=index)
            graph.add_flow(controller.tensor, frame_limit_op.input, queue_depth=4)

        layers.append(
            {
                "port_name": f"video_{name}",
                "kind": "image",
                "format": "rgba16_unorm",
                "max_width": channel["sensor_factory"].image_size()[1],
                "max_height": channel["sensor_factory"].image_size()[0],
                "view": {
                    "offset_x": 0.5 * index,
                    "offset_y": 0.0,
                    "scale_x": 0.5,
                    "scale_y": 1.0,
                },
            }
        )

    if headless:
        # Renders into a CUDA image on HolovizOp's output (consumed by DiscardImageOp). The
        # CUDA sink has no framebuffer_srgb: its sRGB format encodes on write instead.
        sink = {
            "kind": "cuda",
            "width": 1280,
            "height": 400,
            "format": "rgba8_srgb",
        }
    else:
        sink = {
            "kind": "window",
            "framebuffer_srgb": True,
            "title": "Stereo IMX274 reconnection (module)",
            "width": 1280,
            "height": 400,
            "fullscreen": False,
            "close_on_escape": True,
        }
    viz = graph.op(
        "holoscan.visualization.HolovizOp",
        "holoviz",
        layers=layers,
        sink=sink,
        trigger={
            "kind": "clock",
            "period_ns": round(1e9 / 60.0),
        },
    )
    for name, demosaic in zip(tensor_names, demosaics):
        graph.add_flow(
            demosaic.output,
            getattr(viz, f"video_{name}"),
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
    for operator_path in tensor_output_operators:
        options.deployment.bind_tensor_output_device(
            operator_path=operator_path,
            output_port="tensor",
            device=hs.execution.DeviceId(cuda_device_ordinal),
        )
    for name in tensor_names:
        options.deployment.bind_tensor_output_device(
            operator_path=f"bayer_demosaic_{name}",
            output_port="output",
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
        _cuda_check(cuda_driver.cuDevicePrimaryCtxRelease(cu_device))
        return 1

    print("Starting immediately (no device required) -- running (Ctrl+C to stop)...")

    with hs.run_async(plan) as session:
        try:
            session.wait()
        except KeyboardInterrupt:
            # Ctrl+C or SIGTERM: leaving the block stops the session cooperatively, which
            # stops the camera.
            pass
    if session.diagnostics:
        print(session.summary(), file=sys.stderr)
    failed = session.termination == hs.RunTermination.FAILED

    _cuda_check(cuda_driver.cuDevicePrimaryCtxRelease(cu_device))

    if failed:
        return 1
    print("Done.")
    return 0


def main():
    # Like Ctrl+C, SIGTERM (kill, docker stop) raises KeyboardInterrupt, so the session stops
    # cooperatively and the camera and the board are stopped on the way out.
    signal.signal(signal.SIGTERM, signal.default_int_handler)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--channel-ips",
        default="192.168.0.2,192.168.0.3",
        help="Comma-separated peer IPs of the two data channels (typically the "
        "two channels of one stereo board)",
    )
    parser.add_argument(
        "--camera-mode",
        type=int,
        default=imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS.value,
        help="IMX274 mode (1 = 1920x1080 60fps)",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Exit after this many frames per leg (default: run until Ctrl-C)",
    )
    parser.add_argument(
        "--transport",
        choices=("roce", "linux"),
        default="roce",
        help="Data-plane transport (default: roce)",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--module-dir",
        default=None,
        help="Override the directory containing hololink_<UUID>.so files",
    )
    parser.add_argument("--log-level", type=int, default=20)
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)

    return run(
        channel_ips=[ip.strip() for ip in args.channel_ips.split(",") if ip.strip()],
        camera_mode=imx274_mode.Imx274_Mode(args.camera_mode),
        frame_limit=args.frame_limit,
        headless=args.headless,
        transport=args.transport,
        module_dir=args.module_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
