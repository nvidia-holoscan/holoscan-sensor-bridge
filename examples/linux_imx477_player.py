# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Single IMX477 sensor via LinuxReceiverOp (Linux UDP socket, no RoCE/ibverbs needed), displayed
# live with Holoviz. Python port of this repo's own previous (HSDK4-era) Python version of this
# app, via the HSDK5 Python bindings -- there is no examples/linux_imx477_player.cpp sibling to
# mirror (unlike most apps ported this session), so this follows the same HSDK5 shape already
# established by examples/linux_imx274_player.py instead.
#
#   LinuxReceiverOp -> CsiToBayerOp -> ImageProcessorOp -> BayerDemosaicOp -> HolovizOp
#
# Imx477 is a pure-Python driver (python/hololink/sensors/imx477.py); its configure_converter()
# adds no start-byte offset or trailing bytes, so hl.resolve_generic_geometry_from_dimensions()
# reproduces it exactly (same approach already used by examples/ecam0m30tof_player.py) -- no
# manual geometry math needed here, unlike the IMX715/OX08D10 ports.
#
# Usage:
#   linux_imx477_player.py [OPTIONS]
#
# Options:
#   --hololink IP      IP address of Hololink board (default: 192.168.0.2)
#   --cam {0,1}        Which camera to stream: 0 for j14, 1 for j17 (default: 0)
#   --pattern          Configure to display a test pattern
#   --resolution       4k, 1080p, or 720p (default: 4k)
#   --exposure N       Configure exposure (default: 0x05)
#   --flip             none, h, v, or hv (default: none)
#   --frame-limit N    Number of frames to capture (default: unlimited)
#   --headless         Run without opening a window (still renders, presents nowhere)
#   --fullscreen       Run in fullscreen mode
#   --log-level N      Python logging level (default: 20 = INFO)

import argparse
import logging
import sys

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
    pattern,
    resolution,
    exposure,
    flip,
    frame_limit,
    headless,
    fullscreen,
):
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
        hl.use_sensor(metadata, cam)
        channel = hl.DataChannel(metadata)

        hololink_dev = channel.hololink()
        hololink_dev.start()
        hololink_dev.reset()

        camera = hl.sensors.imx477.Imx477(
            channel, camera_id=cam, resolution=resolution, img_flip=flip
        )
        # Configures the camera for the requested resolution, 60fps.
        camera.configure()
        # IMX477 analog gain settings function. Analog gain value range is 0-1023 in decimal
        # (10 bits). Users are free to experiment with the register values.
        camera.set_analog_gain(0x2FF)
        camera.set_exposure_reg(exposure)
        if pattern:
            camera.set_pattern()

        pixel_format = camera.pixel_format()
        bayer_format = camera.bayer_format()
        frame_width = camera._width
        frame_height = camera._height
        geometry = hl.resolve_generic_geometry_from_dimensions(
            int(pixel_format), frame_width, frame_height
        )

        # Build the graph.
        graph = hs.Graph("linux_imx477_player")

        rx = graph.op(
            hl.operators.LinuxReceiverOp,
            "rx",
            data_channel=channel,
            frame_context=int(cu_context),
            frame_limit=frame_limit or 0,
            frame_limit_ends_run=True,
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
                "width": frame_width,
                "height": frame_height,
                "format": "rgba8_srgb",
            }
        else:
            sink = {
                "kind": "window",
                "framebuffer_srgb": True,
                "title": "Linux IMX477 Player",
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
                    "format": "rgba16_unorm",
                    "max_width": frame_width,
                    "max_height": frame_height,
                }
            ],
            sink=sink,
            trigger={"kind": "layer", "layer": 0},
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

        # Left to automatic partitioning alone, all five operators share one serial lane (one
        # thread), so the receiver's compute() -- the only thing that takes frames out of
        # LinuxReceiver's queue -- runs no more often than HolovizOp presents a frame. At 4k that
        # is slower than the camera on some machines, and LinuxReceiver then drops every frame
        # that lands while the display is busy ("no available descriptors"). Give the receiver
        # and the capture chain their own lanes; the depth-1 DROP_OLDEST mailbox into HolovizOp
        # then lets a slow display skip frames without holding the capture chain back.
        graph.partition("receiver").add(rx)
        graph.partition("capture").add(csi, image_processor, demosaic)
        graph.auto_partition_remaining()

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
            f"Capturing {resolution} {frame_width}x{frame_height} -- running (Ctrl+C to stop)..."
        )

        with hs.run_async(plan) as session:
            try:
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
        "--fullscreen", action="store_true", help="Run in fullscreen mode"
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Exit after receiving this many frames",
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
        "--pattern",
        action="store_true",
        help="Configure to display a test pattern.",
    )
    parser.add_argument(
        "--exposure",
        type=int,
        default=0x05,
        help="Configure exposure.",
    )
    parser.add_argument(
        "--resolution",
        default="4k",
        help="4k or 1080p",
    )
    parser.add_argument(
        "--hololink",
        default="192.168.0.2",
        help="IP address of Hololink board",
    )
    parser.add_argument(
        "--flip",
        default="none",
        choices=["none", "h", "v", "hv"],
        help="Choose between none, h, v or hv (for horizontal, vertical, or both)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    return run(
        args.hololink,
        args.cam,
        args.pattern,
        args.resolution,
        args.exposure,
        args.flip,
        args.frame_limit,
        args.headless,
        args.fullscreen,
    )


if __name__ == "__main__":
    sys.exit(main())
