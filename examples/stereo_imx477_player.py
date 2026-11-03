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

# Stereo IMX477 sensors (always both) via two INDEPENDENT RoceReceiverOp legs, each with its own
# IB device/port and its own separately-enumerated DataChannel -- the old app enumerates each leg
# from its own board IP (--hololink-left/--hololink-right) and then asserts both channels resolve
# to the same physical Hololink device (two NIC ports on one board, not two independent boards).
# That assumption is preserved here rather than dropped silently. Python port of this repo's own
# previous (HSDK4-era) Python version of this app, via the HSDK5 Python bindings; there is no
# examples/stereo_imx477_player.cpp sibling, so this follows the same two-NIC-stereo shape
# already established by examples/stereo_imx274_player.py instead.
#
#   RoceReceiverOp -> CsiToBayerOp -> ImageProcessorOp -> BayerDemosaicOp -> HolovizOp
#
# Fixes two real bugs found in the old app while porting (not reproduced here):
#  1. Its frame-size sanity check compared the right camera's CSI length against itself
#     (`frame_size = csi_to_bayer_operator_right.get_csi_length(); assert frame_size ==
#     csi_to_bayer_operator_right.get_csi_length()`) -- always true, never actually checked
#     left vs. right. This port compares the two legs' independently-resolved frame sizes for
#     real, matching stereo_imx274_player.py's own correct left-vs-right check.
#  2. Its right-leg receiver operator had no `name=` kwarg at all (left had "receiver_left").
#     The per-leg loop below names every operator f"..._{name}" for both legs, so this can't
#     recur.
#
# --frame-limit has no HSDK5 CountCondition equivalent, so it's implemented with a small native
# Python operator, FrameLimitOp, tapping each leg's receiver tensor output and counting
# activations; main() waits for both legs to reach the limit before stopping.
#
# Usage:
#   stereo_imx477_player.py [OPTIONS]
#
# Options:
#   --hololink-left IP      IP address of the left-leg Hololink board/NIC (default: 192.168.0.2)
#   --hololink-right IP     IP address of the right-leg Hololink board/NIC (default: 192.168.0.3)
#   --ibv-name-left NAME    Left-leg IBV device (default: the device on the left leg's network interface)
#   --ibv-port-left N       Left-leg IBV device port (default: 1)
#   --ibv-name-right NAME   Right-leg IBV device (default: the device on the right leg's network interface)
#   --ibv-port-right N      Right-leg IBV device port (default: 1)
#   --window-height N       Displayed window height (default: 2160 // 8)
#   --window-width N        Displayed window width (default: 3840 // 6)
#   --title TITLE           Window title (default: "Holoviz - <hololink-left IP>")
#   --resolution            4k, 1080p, or 720p -- shared by both cameras (default: 4k)
#   --exposure N            Configure exposure -- shared by both cameras (default: 0x05)
#   --frame-limit N         Number of frames to capture per sensor (default: unlimited)
#   --headless              Run without opening a window (still renders, presents nowhere)
#   --log-level N           Python logging level (default: 20 = INFO)

import argparse
import logging
import sys
import threading

import hololink.sensors.imx477  # noqa: F401 -- makes hl.sensors.imx477.X resolve below
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


class _FrameLimitObserver:
    """Shared state for --frame-limit: every leg must reach the limit before main() stops."""

    condition = threading.Condition()
    counts = {}
    limit = 0

    @classmethod
    def reset(cls, leg_count, limit):
        with cls.condition:
            cls.counts = {i: 0 for i in range(leg_count)}
            cls.limit = limit

    @classmethod
    def record(cls, leg):
        with cls.condition:
            cls.counts[leg] += 1
            cls.condition.notify_all()

    @classmethod
    def _done_locked(cls):
        return cls.limit > 0 and all(
            count >= cls.limit for count in cls.counts.values()
        )

    @classmethod
    def wait_until_done(cls, session):
        with cls.condition:
            while not cls._done_locked():
                if not session.is_running:
                    return False
                cls.condition.wait(timeout=0.1)
            return True


class FrameLimitOp(hs.Operator):
    """Count one leg's captured frames toward --frame-limit; see _FrameLimitObserver."""

    input = hs.Input("input", hs.payload.Tensor)
    leg = hs.Param(0)

    def setup(self, spec):
        spec.input(self.input, queue_depth=4)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, context):
        self.input.receive_data()
        _FrameLimitObserver.record(self.leg)


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
    hololink_ip_left,
    hololink_ip_right,
    ibv_name_left,
    ibv_port_left,
    ibv_name_right,
    ibv_port_right,
    window_height,
    window_width,
    title,
    resolution,
    exposure,
    frame_limit,
    headless,
):
    cuda_device_ordinal = 0
    tensor_names = ["left", "right"]
    ibv_names = [ibv_name_left, ibv_name_right]
    ibv_ports = [ibv_port_left, ibv_port_right]

    print("Initializing CUDA...")
    _cuda_check(cuda_driver.cuInit(0))
    cu_device = _cuda_check(cuda_driver.cuDeviceGet(cuda_device_ordinal))
    cu_context = _cuda_check(cuda_driver.cuDevicePrimaryCtxRetain(cu_device))
    hololink_dev = None
    try:
        cuda_driver.cuCtxSetCurrent(cu_context)

        print(
            f"Enumerating HSB board at {hololink_ip_left} (left) and {hololink_ip_right} (right)..."
        )
        channels = [
            hl.DataChannel(hl.find_channel(hololink_ip_left)),
            hl.DataChannel(hl.find_channel(hololink_ip_right)),
        ]

        hololink_dev = channels[0].hololink()
        if hololink_dev is not channels[1].hololink():
            raise RuntimeError(
                "Both --hololink-left/--hololink-right IPs must resolve to the same physical "
                "Hololink device (two NIC ports on one board), not two independent boards."
            )
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

        pixel_formats = [camera.pixel_format() for camera in cameras]
        bayer_formats = [camera.bayer_format() for camera in cameras]
        geometries = [
            hl.resolve_generic_geometry_from_dimensions(
                int(camera.pixel_format()), camera._width, camera._height
            )
            for camera in cameras
        ]
        frame_sizes = [
            geometry["start_byte"]
            + geometry["bytes_per_line"] * geometry["pixel_height"]
            + geometry["trailing_bytes"]
            for geometry in geometries
        ]
        if frame_sizes[0] != frame_sizes[1]:
            raise RuntimeError("Both cameras must have the same frame size")

        if frame_limit:
            _FrameLimitObserver.reset(len(cameras), frame_limit)

        # Build the graph.
        graph = hs.Graph("stereo_imx477_player")

        # Pass 1: declare one image layer per camera, tiled side by side -- the layer list must be
        # complete before HolovizOp is constructed.
        layers = []
        for i, camera in enumerate(cameras):
            layers.append(
                {
                    "port_name": f"video_{tensor_names[i]}",
                    "kind": "image",
                    "format": "rgba16_unorm",
                    "max_width": camera._width,
                    "max_height": camera._height,
                    "view": {
                        "offset_x": 0.5 * i,
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
                "width": window_width,
                "height": window_height,
                "format": "rgba8_srgb",
            }
        else:
            sink = {
                "kind": "window",
                "framebuffer_srgb": True,
                "title": title or f"Holoviz - {hololink_ip_left}",
                "width": window_width,
                "height": window_height,
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
        if headless:
            discard = graph.op(DiscardImageOp, "discard")
            graph.add_flow(
                viz.output,
                discard.input,
                queue_policy=hs.QueuePolicy.DROP_OLDEST,
                queue_depth=1,
            )

        tensor_output_operators = []

        for i, camera in enumerate(cameras):
            name = tensor_names[i]
            geometry = geometries[i]
            pixel_format = pixel_formats[i]
            bayer_format = bayer_formats[i]

            rx = graph.op(
                hl.operators.RoceReceiverOp,
                f"rx_{name}",
                data_channel=channels[i],
                frame_context=int(cu_context),
                device_start=camera.start,
                device_stop=camera.stop,
                frame_size=frame_sizes[i],
                ibv_name=ibv_names[i],
                ibv_port=ibv_ports[i],
            )

            if frame_limit:
                frame_limit_op = graph.op(FrameLimitOp, f"frame_limit_{name}", leg=i)
                graph.add_flow(rx.tensor, frame_limit_op.input, queue_depth=4)

            csi = graph.op(
                hl.operators.CsiToBayerOp,
                f"csi_to_bayer_{name}",
                cuda_device_ordinal=cuda_device_ordinal,
                geometry=geometry,
                bayer_format=bayer_format.name.lower(),
            )
            graph.add_flow(rx.tensor, csi.in_tensor, queue_depth=4)
            graph.add_flow(rx.metadata, csi.in_metadata, queue_depth=4)

            image_processor = graph.op(
                hl.operators.ImageProcessorOp,
                f"image_processor_{name}",
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
                hv.operators.BayerDemosaicOp,
                f"bayer_demosaic_{name}",
                bayer_grid_pos=bayer_format.name.lower(),
                generate_alpha=True,
                alpha_value=65535,
            )
            graph.add_flow(image_processor.tensor, demosaic.input, queue_depth=4)
            graph.add_flow(
                demosaic.output,
                getattr(viz, f"video_{name}"),
                queue_policy=hs.QueuePolicy.DROP_OLDEST,
                queue_depth=1,
            )

            tensor_output_operators.append(rx.name)
            tensor_output_operators.append(csi.name)
            tensor_output_operators.append(image_processor.name)

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
            return 1

        print("Capturing 2 sensors -- running (Ctrl+C to stop)...")

        with hs.run_async(plan) as session:
            try:
                if frame_limit and _FrameLimitObserver.wait_until_done(session):
                    session.request_stop()
                session.wait()
            except KeyboardInterrupt:
                pass
        if session.diagnostics:
            print(session.summary(), file=sys.stderr)
        if session.termination == hs.RunTermination.FAILED:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Exit after receiving this many frames",
    )
    parser.add_argument(
        "--hololink-left",
        default="192.168.0.2",
        help="IP address of Hololink board",
    )
    parser.add_argument(
        "--hololink-right",
        default="192.168.0.3",
        help="IP address of Hololink board",
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level to display",
    )
    parser.add_argument(
        "--ibv-name-left",
        default="",
        help="IBV device to use (default: the device on the left leg's network interface)",
    )
    parser.add_argument(
        "--ibv-port-left",
        type=int,
        default=1,
        help="Port number of IBV device (with --ibv-name-left)",
    )
    parser.add_argument(
        "--ibv-name-right",
        default="",
        help="IBV device to use (default: the device on the right leg's network interface)",
    )
    parser.add_argument(
        "--ibv-port-right",
        type=int,
        default=1,
        help="Port number of IBV device (with --ibv-name-right)",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=2160 // 8,  # arbitrary default
        help="Set the height of the displayed window",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=3840 // 6,  # arbitrary default
        help="Set the width of the displayed window",
    )
    parser.add_argument(
        "--title",
        help="Set the window title",
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

    logging.basicConfig(level=args.log_level)

    return run(
        args.hololink_left,
        args.hololink_right,
        args.ibv_name_left,
        args.ibv_port_left,
        args.ibv_name_right,
        args.ibv_port_right,
        args.window_height,
        args.window_width,
        args.title,
        args.resolution,
        args.exposure,
        args.frame_limit,
        args.headless,
    )


if __name__ == "__main__":
    sys.exit(main())
