# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Single IMX715 sensor via LinuxReceiverOp (Linux UDP socket, no RoCE/ibverbs needed), displayed
# live with Holoviz. Python port of this repo's own previous (HSDK4-era) Python version of this
# app, via the HSDK5 Python bindings -- there is no examples/linux_imx715_player.cpp sibling to
# mirror, so this follows the same HSDK5 shape already established by
# examples/linux_imx274_player.py instead.
#
#   LinuxReceiverOp -> CsiToBayerOp -> ImageProcessorOp -> BayerDemosaicOp -> HolovizOp
#
# Imx715Cam is a pure-Python driver (python/hololink/sensors/imx715/imx715.py). Unlike IMX477,
# its configure_converter() adds a real per-mode start-byte offset and (for the 1080p mode)
# trailing bytes and extra inter-line bytes -- hl.resolve_generic_geometry_from_dimensions()
# cannot reproduce this (it has no offset/trailing-bytes parameters, confirmed by reading
# src/hololink/sensors/camera/camera_sensor.cpp's GeometryCaptureConverter directly), so the
# geometry dict below is built by hand, replicating that same adapter's
# received_line_bytes/transmitted_line_bytes formulas plus imx715.py's own per-mode offsets.
#
# The old app's compose()-time `self._camera.set_mode(self._camera_mode)` call is redundant and
# dropped here: camera.configure(...) already calls set_mode() internally via
# configure_camera() (confirmed by reading imx715.py directly), and there's no separate compose()
# phase in HSDK5 anyway.
#
# --frame-limit has no HSDK5 CountCondition equivalent, so it's implemented with a small native
# Python operator, FrameLimitOp, tapping the receiver's tensor output and counting activations.
#
# Usage:
#   linux_imx715_player.py [OPTIONS]
#
# Options:
#   --camera-mode N              IMX715 mode: 0=3840x2160@30fps 12bpp, 1=3840x2160@60fps 10bpp,
#                                 2=1920x1080@60fps 12bpp (default: 2)
#   --gain N                     Gain, 0-300 (default: 10)
#   --frame-rate N                Frame rate (default: 60)
#   --exp N                      Exposure in ms, 0-1000 (default: 16)
#   --trigger {0,1,2}            Trigger mode (default: 0)
#   --hololink IP                 IP address of Hololink board (default: 192.168.0.2)
#   --expander-configuration N   I2C expander configuration (0 or 2, default: 0)
#   --pattern N                  Configure a test pattern (0-11) instead of live video
#   --ptp-sync                   After reset, wait for PTP time to synchronize
#   --window-height N            Displayed window height (default: 270)
#   --window-width N             Displayed window width (default: 640)
#   --title TITLE                 Window title (default: "Holoviz - <hololink IP>")
#   --frame-limit N               Number of frames to capture (default: unlimited)
#   --headless                    Run without opening a window (still renders, presents nowhere)
#   --fullscreen                  Run in fullscreen mode
#   --log-level N                 Python logging level (default: 20 = INFO)

import argparse
import logging
import signal
import sys
import threading

import hololink.sensors.imx715  # noqa: F401 -- makes hl.sensors.imx715.X resolve below
import holoscan as hs
import holoscan_vision as hv
import holoscan_visualization as hz  # noqa: F401 -- registers "holoscan.visualization.HolovizOp"
from cuda.bindings import driver as cuda_driver

import hololink as hl

# (hololink/__init__.py doesn't import the sensors subpackage itself, and Python doesn't expose
# an unimported submodule as a parent-package attribute).


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
# received_line_bytes, RAW_8/10/12 transmitted_line_bytes) plus imx715.py's own per-mode
# start_byte/trailing_bytes offsets (imx715.py:531-561). No Python API exposes this directly for
# pure-Python camera drivers (only hl.resolve_generic_geometry(camera), which has C++ overloads
# for the nanobind-bound NativeImx274Sensor/NativeVb1940Sensor only).
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


def _imx715_geometry(camera):
    imx715 = hl.sensors.imx715
    pixel_format, width, height, mode = (
        camera.pixel_format(),
        camera._width,
        camera._height,
        camera._mode,
    )
    transmitted = _transmitted_line_bytes(pixel_format, width)
    if mode == imx715.IMX715_MODE_1920X1080_60FPS_12BPP:
        transmitted += (
            38  # extra inter-line bytes (imx715.py's own 1080p-mode adjustment)
        )
    received = _received_line_bytes(transmitted)
    if mode == imx715.IMX715_MODE_3840X2160_30FPS_12BPP:
        start_byte, trailing = 37 * received, 0
    elif mode == imx715.IMX715_MODE_3840X2160_60FPS_10BPP:
        start_byte, trailing = 41 * received, 0
    elif mode == imx715.IMX715_MODE_1920X1080_60FPS_12BPP:
        start_byte, trailing = 21 * received, 14 * received
    else:
        raise ValueError(f"Unsupported IMX715 mode {mode}")
    return _geometry_dict(start_byte, received, width, height, pixel_format, trailing)


def run(
    hololink_ip,
    camera_mode,
    gain,
    frame_rate,
    exp,
    trigger,
    expander_configuration,
    pattern,
    ptp_sync,
    window_height,
    window_width,
    title,
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
    metadata = hl.find_channel(hololink_ip)
    channel = hl.DataChannel(metadata)

    hololink_dev = channel.hololink()
    hololink_dev.start()
    hololink_dev.reset()
    if ptp_sync:
        logging.debug("Waiting for PTP sync.")
        if not hololink_dev.ptp_synchronize():
            logging.error("Failed to synchronize PTP; ignoring.")
        else:
            logging.debug("PTP synchronized.")

    camera = hl.sensors.imx715.Imx715Cam(
        channel, expander_configuration=expander_configuration
    )
    camera.cam_reset()
    camera.setup_clock()
    camera.configure(camera_mode, gain, frame_rate, exp, trigger)
    if pattern is not None:
        camera.test_pattern(pattern)

    pixel_format = camera.pixel_format()
    bayer_format = camera.bayer_format()
    frame_width = camera._width
    frame_height = camera._height
    geometry = _imx715_geometry(camera)

    if frame_limit:
        _FrameLimit.reset(frame_limit)

    # Build the graph.
    graph = hs.Graph("linux_imx715_player")

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

    if frame_limit:
        frame_limit_op = graph.op(FrameLimitOp, "frame_limit")
        graph.add_flow(rx.tensor, frame_limit_op.input, queue_depth=4)

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
        # Optical black value for imx715 is 50 (matches this app's own HSDK4-era default).
        optical_black=50,
        cuda_device_ordinal=cuda_device_ordinal,
        # Sizes the output tensor pool's slots to the image, not to the 64 MiB ceiling.
        pixel_width=frame_width,
        pixel_height=frame_height,
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
            "title": title or f"Holoviz - {hololink_ip}",
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
        hololink_dev.stop()
        _cuda_check(cuda_driver.cuDevicePrimaryCtxRelease(cu_device))
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

    hololink_dev.stop()
    _cuda_check(cuda_driver.cuDevicePrimaryCtxRelease(cu_device))

    if failed:
        return 1
    print("Done.")
    return 0


def main():
    # Like Ctrl+C, SIGTERM (kill, docker stop) raises KeyboardInterrupt, so the session stops
    # cooperatively and the camera and the board are stopped on the way out.
    signal.signal(signal.SIGTERM, signal.default_int_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera-mode",
        type=int,
        default=2,
        help="IMX715 supported mode:~ Index0: 3840x2160 30fps 12bpp Index1: 3840x2160 60fps 10bpp Index2: 1920x1080 60fps 12bpp\n",
    )
    parser.add_argument(
        "--gain",
        type=int,
        default=10,
        help="set GAIN values, RANGE(0 to 300)",
    )
    parser.add_argument(
        "--frame-rate",
        type=int,
        default=60,
        help="set FRAME RATE values",
    )
    parser.add_argument(
        "--exp",
        type=int,
        default=16,
        help="set EXPOSURE values in millisec, RANGE(0 to 1000)ms",
    )
    parser.add_argument(
        "--trigger",
        type=int,
        default=0,
        choices=range(3),
        help="set Trigger mode for IMX715, RANGE(0 to 2)",
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
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level to display",
    )
    parser.add_argument(
        "--expander-configuration",
        type=int,
        default=0,
        choices=(0, 2),
        help="I2C Expander configuration",
    )
    parser.add_argument(
        "--pattern",
        type=int,
        choices=range(12),
        help="Configure to display a test pattern.",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=270,
        help="Set the height of the displayed window",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=640,
        help="Set the width of the displayed window",
    )
    parser.add_argument(
        "--title",
        help="Set the window title",
    )
    parser.add_argument(
        "--ptp-sync",
        action="store_true",
        help="After reset, wait for PTP time to synchronize.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    return run(
        args.hololink,
        args.camera_mode,
        args.gain,
        args.frame_rate,
        args.exp,
        args.trigger,
        args.expander_configuration,
        args.pattern,
        args.ptp_sync,
        args.window_height,
        args.window_width,
        args.title,
        args.frame_limit,
        args.headless,
        args.fullscreen,
    )


if __name__ == "__main__":
    sys.exit(main())
