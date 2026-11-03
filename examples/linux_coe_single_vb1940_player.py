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

# Displays live VB1940 video received over IEEE 1722/AVTP Class-Oriented-Encapsulation (COE)
# packets via LinuxCoeReceiverOp, read from a raw AF_PACKET socket bound to the given network
# interface (not UDP/RoCE like examples/vb1940_player.py). Python port of this repo's own
# previous (HSDK4-era) Python version of this app, via the HSDK5 Python bindings -- there is no
# examples/linux_coe_single_vb1940_player.cpp sibling. See examples/linux_coe_imx274_player.py's
# own header comment for the full rationale behind hololink.operators.LinuxCoeReceiverOp itself
# (new in this branch: a proper C++ Operator<Params> implementation replacing the old pure-Python
# BaseReceiverOp subclass dropped, unported, during this session's earlier nanobind rewrite).
# Sensor bring-up (the manual RESET-pin sequence, register sanity reads) matches this repo's own
# already-ported examples/vb1940_player.py exactly -- COE only changes the receiver and the
# network bring-up below it.
#
# To receive COE packets, create a VLAN interface tagged for VETH 0xCCC (3276) on top of your
# real network interface, then point --coe-interface at it -- see
# examples/linux_coe_imx274_player.py's header comment for the exact `ip link add` recipe.
#
#   LinuxCoeReceiverOp -> CsiToBayerOp -> ImageProcessorOp -> BayerDemosaicOp -> HolovizOp
#
# Usage:
#   linux_coe_single_vb1940_player.py [OPTIONS]
#
# Options:
#   --camera-mode N        VB1940 mode (default: VB1940_MODE_2560X1984_30FPS)
#   --coe-interface NAME   Network interface receiving VETH 3276 (0xCCC) COE packets (required)
#   --hololink IP          IP address of Hololink board (default: 192.168.0.2)
#   --frame-limit N        Number of frames to capture (default: unlimited)
#   --headless             Run without opening a window (still renders, presents nowhere)
#   --fullscreen           Run in fullscreen mode
#   --log-level N          Python logging level (default: 20 = INFO)

import argparse
import logging
import sys
import time

import holoscan as hs
import holoscan_vision as hv
import holoscan_visualization as hz  # noqa: F401 -- registers "holoscan.visualization.HolovizOp"
from cuda.bindings import driver as cuda_driver

import hololink as hl

_BAYER_FORMAT_NAMES = ["bggr", "rggb", "gbrg", "grbg"]
_BAYER_FORMAT_VALUES = {name: index for index, name in enumerate(_BAYER_FORMAT_NAMES)}

# Each camera sharing one COE network connection must use a unique channel number (0..63);
# matches the old app's own hardcoded value.
_COE_CHANNEL = 1


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
    camera_mode,
    coe_interface,
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
        channel = hl.DataChannel(metadata)

        hololink_dev = channel.hololink()
        hololink_dev.start()

        camera = hl.NativeVb1940Sensor(channel)

        # Manual sensor-reset sequence -- matches examples/vb1940_player.py exactly; VB1940's reset
        # pin is driven directly via a board register.
        hololink_dev.write_uint32(0x8, 0x0)  # Keep the sensor RESET at low
        camera.setup_clock()
        hololink_dev.write_uint32(0x8, 0x1)  # Release the sensor RESET to high
        time.sleep(100 / 1000)
        camera.get_register_32(0x0000)  # DEVICE_MODEL_ID:"S940"(ASCII code:0x53393430)
        camera.get_register_32(0x0734)  # EXT_CLOCK(25MHz = 0x017d7840)
        camera.configure(camera_mode)
        # No exposure/gain override: the mode table's 500-line integration time and unity analog
        # gain are what examples/vb1940_player.py shows. The HSDK4-era app's
        # set_exposure_reg(0x0000)/set_analog_gain_reg(0x0C) clamped exposure to the 4-line
        # minimum at 4x gain, ~30x darker than that player.

        pixel_format = camera.get_pixel_format()
        bayer_format = camera.get_bayer_format()
        frame_width = camera.get_width()
        frame_height = camera.get_height()

        geometry = hl.resolve_generic_geometry(camera)
        frame_size = (
            geometry["start_byte"]
            + geometry["bytes_per_line"] * geometry["pixel_height"]
            + geometry["trailing_bytes"]
        )

        # Build the graph.
        graph = hs.Graph("linux_coe_single_vb1940_player")

        rx = graph.op(
            hl.operators.LinuxCoeReceiverOp,
            "rx",
            data_channel=channel,
            frame_context=int(cu_context),
            device_start=camera.start,
            device_stop=camera.stop,
            frame_size=frame_size,
            coe_interface=coe_interface,
            pixel_width=frame_width,
            coe_channel=_COE_CHANNEL,
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
        graph.add_flow(rx.tensor, csi.in_tensor, queue_depth=4)
        graph.add_flow(rx.metadata, csi.in_metadata, queue_depth=4)

        image_processor = graph.op(
            hl.operators.ImageProcessorOp,
            "image_processor",
            pixel_format=pixel_format,
            pixel_width=geometry["pixel_width"],
            pixel_height=geometry["pixel_height"],
            bayer_format=_BAYER_FORMAT_VALUES[bayer_format],
            # Optical black value for vb1940 is 8 for RAW10 (matches this app's own HSDK4-era
            # default).
            optical_black=8,
            cuda_device_ordinal=cuda_device_ordinal,
        )
        graph.add_flow(csi.tensor, image_processor.in_tensor, queue_depth=4)
        graph.add_flow(csi.metadata, image_processor.in_metadata, queue_depth=4)

        demosaic = graph.op(
            hv.operators.BayerDemosaicOp,
            "bayer_demosaic",
            bayer_grid_pos=bayer_format,
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
                    "format": "rgba16_unorm",
                    "max_width": frame_width,
                    "max_height": frame_height,
                }
            ],
            sink=sink,
            trigger={"kind": "layer"},
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
                pass
        if session.diagnostics:
            print(session.summary(), file=sys.stderr)
        failed = session.termination == hs.RunTermination.FAILED

        if failed:
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
    parser.add_argument(
        "--camera-mode",
        type=int,
        default=0,
        help="VB1940 mode (default: 0 = VB1940_MODE_2560X1984_30FPS)",
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
        "--coe-interface",
        required=True,
        help="Name of interface connected to VETH 3276 (0xCCC).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    return run(
        args.hololink,
        args.camera_mode,
        args.coe_interface,
        args.frame_limit,
        args.headless,
        args.fullscreen,
    )


if __name__ == "__main__":
    sys.exit(main())
