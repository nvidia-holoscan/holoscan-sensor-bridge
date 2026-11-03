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

# OX08D10 camera player for the Taurotech HSB DA326 board (GMSL2 via MAX96716A/MAX96719B), via
# RoceReceiverOp(module) (RDMA/InfiniBand). Python port of this repo's own previous (HSDK4-era)
# Python version of this app, via the HSDK5 Python bindings -- there is no
# examples/ox08d10_player.cpp sibling, so the pipeline/CLI shape below follows the HSDK5 pattern
# already established by examples/imx274_player.py, while the board/GMSL bring-up and receiver
# construction follow examples/hawk_single_link_player.py and
# examples/linux_max96716a_player.py's own already-ported "module receiver + native downstream"
# convention exactly (both already use this same hololink_module.Adapter/
# TauroTechDa326InterfaceV1/hololink_module.operators.RoceReceiverOp sequence).
#
# Prerequisites:
#   - Taurotech HSB DA326 board with MAX96716A deserializer
#   - Aumovio SAT301DP17 camera module connected to GMSL link A or B
#   - Board init (release_reset, power_cycle, setup_clock) runs via
#     TauroTechDa326InterfaceV1 before GMSL bring-up.
#
#   RoceReceiverOp(module) -> FrameSizeFilterOp -> CsiToBayerOp -> ImageProcessorOp ->
#       BayerDemosaicOp -> HolovizOp
#
# Ox08d10Cam is a pure-Python driver (python/hololink/sensors/ox08d10/ox08d10_cam.py). Its
# configure_converter() adds a real start-byte offset (2 front-embedded-data lines) and trailing
# bytes (14 non-image rows) -- hl.resolve_generic_geometry_from_dimensions() cannot reproduce
# this (no offset/trailing-bytes parameters, confirmed by reading
# src/hololink/sensors/camera/camera_sensor.cpp's GeometryCaptureConverter directly), so the
# geometry dict below is built by hand, replicating that same adapter's
# received_line_bytes/transmitted_line_bytes formulas plus ox08d10_cam.py's own offset math
# (reading FRONT_EMBEDDED_LINES/_received_height off the live camera object, not hardcoding
# them).
#
# The old app's compose()-time `self._camera.set_mode(self._camera_mode)` call is kept
# unconditional here too (not folded into the --skip-setup-gated bring-up block): it's the only
# thing that populates _width/_height/_pixel_format/_received_height when --skip-setup is given
# (configure_vendor_tunnel_path(), the only other set_mode() caller, is itself skipped in that
# path) -- confirmed by reading the old app's own compose()/main() call order directly.
#
# FrameSizeFilterOp is kept (not dropped just because geometry is now computed correctly): it's
# defensive insurance against any residual frame-size mismatch this port's own geometry fix isn't
# guaranteed to fully eliminate, ported to HSDK5's two-port (tensor+metadata) shape using the
# real Sample/OperatorPort.forward() passthrough API (see examples/e2e_imx274_latency.py's
# MultiEventAcknowledgmentOp for the established single-port precedent this extends to two
# paired ports).
#
# --frame-limit has no HSDK5 CountCondition equivalent, so it's implemented with a small native
# Python operator, FrameLimitOp, tapping the receiver's tensor output directly (matching the old
# app's own CountCondition-on-receiver placement -- it counts every received frame, not just
# ones FrameSizeFilterOp passes through).
#
# Usage:
#   ox08d10_player.py [OPTIONS]
#
# Options:
#   --camera-mode N        OX08D10 mode (only 1:OX08D10_MODE_3840X2160_30FPS_RAW12 is exposed,
#                           matching the old app's own restriction)
#   --hololink IP           IP address of Hololink board (default: 192.168.0.2)
#   --channel {A,B}         GMSL link the camera is connected to (default: A)
#   --window-height N       Displayed window height (default: 1080)
#   --window-width N        Displayed window width (default: 1920)
#   --title TITLE            Window title (default: "OX08D10")
#   --skip-reset             Skip the initial board reset (only meaningful without --skip-setup)
#   --skip-setup             Skip board/GMSL/sensor bring-up (assume already streaming)
#   --exposure N             Sensor exposure register value (default: 0x0200)
#   --pattern N              Test-pattern register value (0-3)
#   --check-frame-sizes N    Log receiver bytes_written/tensor sizes for the first N frames
#   --frame-limit N          Number of frames to capture (default: unlimited)
#   --headless               Run without opening a window (still renders, presents nowhere)
#   --fullscreen             Run in fullscreen mode
#   --log-level N            Python logging level (default: 20 = INFO)

import argparse
import logging
import signal
import sys
import threading
from datetime import timedelta

import capture_metadata
import hololink_module.operators
import hololink_module.taurotech_da326 as taurotech_da326_module
import holoscan as hs
import holoscan_vision as hv
import holoscan_visualization as hz  # noqa: F401 -- registers "holoscan.visualization.HolovizOp"
from cuda.bindings import driver as cuda_driver
from hololink.sensors.ox08d10 import ox08d10_mode
from hololink.sensors.ox08d10.ox08d10_cam import Ox08d10Cam

import hololink as hl  # native downstream operators (CsiToBayerOp/ImageProcessorOp)

# The module receiver + native downstream convention is established by
# examples/hawk_single_link_player.py.
import hololink_module

CHANNEL_TO_SIF = {"A": 0, "B": 1}

# Matches hololink::operators::kMetadataPairingWindow (capture_metadata_utils.hpp): the
# receiver/CsiToBayerOp/ImageProcessorOp chain stamps every port pair with an identical
# capture_time, so this bounded-window OnAligned+Exact{} pairs them precisely.
_METADATA_PAIRING_WINDOW = timedelta(microseconds=500)


def _cuda_check(result):
    err = result[0]
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        _, description = cuda_driver.cuGetErrorString(err)
        raise RuntimeError(f"CUDA error: {description}")
    if len(result) == 2:
        return result[1]
    return result[1:]


class _FrameLimit:
    """Shared state for --frame-limit: a single counter (this app is single-sensor)."""

    lock = threading.Lock()
    count = 0
    limit = 0

    @classmethod
    def reset(cls, limit):
        with cls.lock:
            cls.count = 0
            cls.limit = limit

    @classmethod
    def record(cls):
        """Counts a frame; True once the limit is reached."""
        with cls.lock:
            cls.count += 1
            return cls.limit > 0 and cls.count >= cls.limit


class FrameLimitOp(hs.Operator):
    """Count captured frames toward --frame-limit; see _FrameLimit."""

    input = hs.Input("input", hs.payload.Tensor)

    def setup(self, spec):
        spec.input(self.input, queue_depth=4)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, context):
        self.input.receive_data()
        if _FrameLimit.record():
            context.request_stop()


class FrameSizeFilterOp(hs.Operator):
    """Drop frames where bytes_written doesn't match expected_frame_size.

    Incomplete frames passed to CsiToBayerOp produce stitched/ghost artifacts. Logs frame sizes
    for the first max_log_frames frames to diagnose sizing issues. Both the tensor and metadata
    ports are forwarded together or dropped together, so CsiToBayerOp's own in_tensor/in_metadata
    pairing downstream stays intact.
    """

    in_tensor = hs.Input("in_tensor", hs.payload.Tensor)
    in_metadata = hs.Input("in_metadata", capture_metadata.CaptureMetadata)
    tensor = hs.Output("tensor", hs.payload.Tensor)
    metadata = hs.Output("metadata", capture_metadata.CaptureMetadata)

    expected_frame_size = hs.Param(0)
    max_log_frames = hs.Param(0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._frame_count = 0
        self._dropped_count = 0
        self._sizes = {}

    def setup(self, spec):
        spec.input(self.in_tensor, queue_depth=4)
        spec.input(self.in_metadata, queue_depth=4)
        spec.output(self.tensor)
        spec.output(self.metadata)

    def contract(self):
        return (
            hs.Contract()
            .trigger(
                hs.OnAligned(
                    ports=[self.in_tensor, self.in_metadata],
                    window=_METADATA_PAIRING_WINDOW,
                )
            )
            .input(self.in_tensor, hs.Exact())
            .input(self.in_metadata, hs.Exact())
        )

    def compute(self, context):
        tensor_sample = self.in_tensor.receive()
        metadata_sample = self.in_metadata.receive()

        bytes_written = metadata_sample.data.bytes_written
        self._sizes[bytes_written] = self._sizes.get(bytes_written, 0) + 1
        self._frame_count += 1

        if self.max_log_frames and self._frame_count <= self.max_log_frames:
            logging.info(
                "frame-size-check frame=%d bytes_written=%d expected=%d unique_sizes=%s",
                self._frame_count,
                bytes_written,
                self.expected_frame_size,
                dict(sorted(self._sizes.items())),
            )

        if bytes_written != self.expected_frame_size:
            self._dropped_count += 1
            if self._dropped_count == 1 or (
                self.max_log_frames
                and (self._dropped_count <= 10 or self._dropped_count % 60 == 0)
            ):
                logging.warning(
                    "Dropping incomplete OX08D10 frame: frame=%d dropped=%d "
                    "bytes_written=%d expected=%d",
                    self._frame_count,
                    self._dropped_count,
                    bytes_written,
                    self.expected_frame_size,
                )
            return

        self.tensor.forward(tensor_sample)
        self.metadata.forward(metadata_sample)


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


# Shared geometry helper -- replicates src/hololink/sensors/camera/camera_sensor.cpp's
# GeometryCaptureConverter formula (receiver_start_byte()=0, 8-byte-aligned
# received_line_bytes, RAW_8/10/12 transmitted_line_bytes) plus ox08d10_cam.py's own
# configure_converter() offset math (ox08d10_cam.py:480-503). No Python API exposes this
# directly for pure-Python camera drivers (only hl.resolve_generic_geometry(camera), which has
# C++ overloads for the nanobind-bound NativeImx274Sensor/NativeVb1940Sensor only).
def _received_line_bytes(n):
    return (n + 7) & ~7


def _transmitted_line_bytes(pixel_format, width):
    if pixel_format == hl.PixelFormat.RAW_8:
        return width
    if pixel_format == hl.PixelFormat.RAW_10:
        return width * 5 // 4
    if pixel_format == hl.PixelFormat.RAW_12:
        return width * 3 // 2
    raise ValueError(f"Unsupported pixel_format {pixel_format!r}")


def _geometry_dict(
    start_byte, bytes_per_line, width, height, pixel_format, trailing_bytes
):
    return {
        "start_byte": start_byte,
        "bytes_per_line": bytes_per_line,
        "pixel_width": width,
        "pixel_height": height,
        "pixel_format": pixel_format.name.lower(),
        "trailing_bytes": trailing_bytes,
    }


def _ox08d10_geometry(camera):
    pixel_format, width, height = camera.pixel_format(), camera._width, camera._height
    received = _received_line_bytes(_transmitted_line_bytes(pixel_format, width))
    front_embedded_bytes = received * camera.FRONT_EMBEDDED_LINES
    trailing_lines = camera._received_height - camera.FRONT_EMBEDDED_LINES - height
    return _geometry_dict(
        front_embedded_bytes,
        received,
        width,
        height,
        pixel_format,
        received * trailing_lines,
    )


def run(
    hololink_ip,
    camera_mode,
    channel,
    window_height,
    window_width,
    title,
    skip_reset,
    skip_setup,
    exposure,
    pattern,
    check_frame_sizes,
    frame_limit,
    headless,
    fullscreen,
):
    cuda_device_ordinal = 0

    print("Initializing CUDA...")
    _cuda_check(cuda_driver.cuInit(0))
    cu_device = _cuda_check(cuda_driver.cuDeviceGet(cuda_device_ordinal))
    cu_context = _cuda_check(cuda_driver.cuDevicePrimaryCtxRetain(cu_device))
    cuda_driver.cuCtxSetCurrent(cu_context)

    print(f"Enumerating HSB board at {hololink_ip}...")
    adapter = hololink_module.Adapter.get_adapter()
    metadata = adapter.wait_for_channel(hololink_ip, 30.0)

    holo_iface = hololink_module.HololinkInterfaceV1.get_service(metadata)
    holo_iface.start()
    try:
        board = taurotech_da326_module.TauroTechDa326InterfaceV1.get_service(metadata)

        sif = CHANNEL_TO_SIF[channel]
        sif_metadata = hololink_module.EnumerationMetadata(metadata)
        adapter.use_sensor(sif_metadata, sif)

        camera = Ox08d10Cam(board.hololink(), skip_i2c=skip_setup)

        if not skip_setup:
            if not skip_reset:
                holo_iface.reset()

            # DA326 board init: releases camera reset, power-cycles the module, verifies
            # power, and sets up board clocks. Must run before GMSL bring-up.
            board.release_reset()
            board.power_cycle()
            board.check_power()
            board.setup_clock()

            # GMSL bring-up: configures MAX96716A deserializer, MAX96719B serializer,
            # releases sensor reset, waits for boot, validates sensor ID, writes mode
            # registers, and sets virtual channel mapping.
            str_to_link = {
                "A": ox08d10_mode.GmslLink.LINK_A,
                "B": ox08d10_mode.GmslLink.LINK_B,
            }
            camera.configure_vendor_tunnel_path(camera_mode, str_to_link[channel])
            if pattern is not None:
                camera.test_pattern(pattern)
            camera.set_exposure_reg(exposure)

        # The only thing that populates _width/_height/_pixel_format/_received_height when
        # --skip-setup is given (configure_vendor_tunnel_path() above, the only other
        # set_mode() caller, is itself skipped in that path) -- keep this unconditional.
        camera.set_mode(camera_mode)

        pixel_format = camera.pixel_format()
        bayer_format = camera.bayer_format()
        frame_width = camera._width
        frame_height = camera._height
        geometry = _ox08d10_geometry(camera)
        frame_size = (
            geometry["start_byte"]
            + geometry["bytes_per_line"] * geometry["pixel_height"]
            + geometry["trailing_bytes"]
        )

        if frame_limit:
            _FrameLimit.reset(frame_limit)

        # Build the graph.
        graph = hs.Graph("ox08d10_player")

        rx = graph.op(
            hololink_module.operators.RoceReceiverOp,
            "rx",
            metadata=sif_metadata,
            frame_context=int(cu_context),
            device_start=camera.start,
            device_stop=camera.stop,
            frame_size=frame_size,
        )

        if frame_limit:
            frame_limit_op = graph.op(FrameLimitOp, "frame_limit")
            graph.add_flow(rx.tensor, frame_limit_op.input, queue_depth=4)

        frame_size_filter = graph.op(
            FrameSizeFilterOp,
            "frame_size_filter",
            expected_frame_size=frame_size,
            max_log_frames=check_frame_sizes,
        )
        graph.add_flow(rx.tensor, frame_size_filter.in_tensor, queue_depth=4)
        graph.add_flow(rx.metadata, frame_size_filter.in_metadata, queue_depth=4)

        csi = graph.op(
            hl.operators.CsiToBayerOp,
            "csi_to_bayer",
            cuda_device_ordinal=cuda_device_ordinal,
            geometry=geometry,
            bayer_format=bayer_format.name.lower(),
        )
        graph.add_flow(frame_size_filter.tensor, csi.in_tensor, queue_depth=4)
        graph.add_flow(frame_size_filter.metadata, csi.in_metadata, queue_depth=4)

        image_processor = graph.op(
            hl.operators.ImageProcessorOp,
            "image_processor",
            pixel_format=int(pixel_format),
            pixel_width=geometry["pixel_width"],
            pixel_height=geometry["pixel_height"],
            bayer_format=int(bayer_format),
            # OX08D10 optical black: datasheet BLKD_SEL default is 0x40 (64 DN at 10-bit). For
            # RAW12 HDR mode this is unconfirmed; adjust from vendor calibration.
            optical_black=64,
            cuda_device_ordinal=cuda_device_ordinal,
        )
        graph.add_flow(csi.tensor, image_processor.in_tensor, queue_depth=4)
        graph.add_flow(csi.metadata, image_processor.in_metadata, queue_depth=4)

        demosaic = graph.op(
            hv.operators.BayerDemosaicOp,
            "bayer_demosaic",
            bayer_grid_pos=bayer_format.name.lower(),
            generate_alpha=True,
            alpha_value=65535,
        )
        graph.add_flow(image_processor.tensor, demosaic.input, queue_depth=4)

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
                "title": title,
                "width": window_width,
                "height": window_height,
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
                    "format": "rgba16_unorm",
                    "max_width": frame_width,
                    "max_height": frame_height,
                }
            ],
            sink=sink,
            trigger={
                "kind": "clock",
                "period_ns": round(1e9 / 60.0),
            },
        )
        graph.add_flow(
            demosaic.output,
            viz.video,
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
        options.deployment.bind_tensor_output_device(
            operator_path="bayer_demosaic",
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
            return 1

        print(
            f"Capturing mode={camera_mode} {frame_width}x{frame_height} -- running (Ctrl+C to stop)..."
        )

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
    finally:
        holo_iface.stop()

    _cuda_check(cuda_driver.cuDevicePrimaryCtxRelease(cu_device))

    if failed:
        return 1
    print("Done.")
    return 0


def main():
    # Like Ctrl+C, SIGTERM (kill, docker stop) raises KeyboardInterrupt, so the session stops
    # cooperatively and the camera and the board are stopped on the way out.
    signal.signal(signal.SIGTERM, signal.default_int_handler)

    parser = argparse.ArgumentParser(
        description="OX08D10 player (GMSL2 via Taurotech HSB DA326 board)"
    )
    modes = ox08d10_mode.Ox08d10_Mode
    mode_choices = [modes.OX08D10_MODE_3840X2160_30FPS_RAW12.value]
    parser.add_argument(
        "--camera-mode",
        type=int,
        choices=mode_choices,
        default=modes.OX08D10_MODE_3840X2160_30FPS_RAW12.value,
        help="1:OX08D10_MODE_3840X2160_30FPS_RAW12",
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
        "--hololink", default="192.168.0.2", help="IP address of Hololink board"
    )
    parser.add_argument(
        "--log-level", type=int, default=20, help="Python logging level to display"
    )
    parser.add_argument(
        "--channel",
        default="A",
        choices=("A", "B"),
        help="GMSL link the camera is connected to on the Taurotech board",
    )
    parser.add_argument("--window-height", type=int, default=1080)
    parser.add_argument("--window-width", type=int, default=1920)
    parser.add_argument("--title", default="OX08D10")
    parser.add_argument("--skip-reset", action="store_true")
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip board init, GMSL bring-up and sensor init (use if already streaming)",
    )
    parser.add_argument("--exposure", type=lambda x: int(x, 0), default=0x0200)
    parser.add_argument("--pattern", type=int, choices=range(4))
    parser.add_argument(
        "--check-frame-sizes",
        type=int,
        default=0,
        metavar="N",
        help="Log receiver bytes_written/tensor sizes for the first N frames",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    camera_mode = ox08d10_mode.Ox08d10_Mode(args.camera_mode)

    return run(
        args.hololink,
        camera_mode,
        args.channel,
        args.window_height,
        args.window_width,
        args.title,
        args.skip_reset,
        args.skip_setup,
        args.exposure,
        args.pattern,
        args.check_frame_sizes,
        args.frame_limit,
        args.headless,
        args.fullscreen,
    )


if __name__ == "__main__":
    sys.exit(main())
