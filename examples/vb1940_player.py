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

# Single VB1940 sensor via RoceReceiverOp (RDMA/InfiniBand), displayed live with Holoviz.
# Python port of this repo's own previous (HSDK4-era) Python version of this app, via the
# HSDK5 Python bindings; the receiver construction pattern and the manual sensor-reset sequence
# below also match examples/vb1940_player.cpp. Unlike that C++ file (which still carries a
# local BayerImageExtractorOp 16-bit->8-bit narrowing shim), this skips that stage -- same
# simplification already applied to this repo's other ported Python examples: BayerDemosaicOp
# accepts ImageProcessorOp's 16-bit schema::ImageT output directly.
#
#   RoceReceiverOp -> CsiToBayerOp -> ImageProcessorOp -> BayerDemosaicOp -> HolovizOp
#
# --frame-limit is the receiver's frame_limit parameter: the receiver ends the run after that
# many frames. HSDK5 has no CountCondition; this replaces the CountCondition of the old app.
#
# Usage:
#   vb1940_player.py [OPTIONS]
#
# Options:
#   --hololink IP        IP address of Hololink board (default: 192.168.0.2)
#   --camera-mode N       VB1940 mode (default: 0 = 2560x1984@30fps)
#   --ibv-name NAME       IBV device to use (default: the device on the board's network interface)
#   --ibv-port N          Port number of IBV device (default: 1)
#   --use-sensor N        Use sensor 0 or 1 (default: 0)
#   --frame-limit N       Number of frames to capture (default: unlimited)
#   --headless            Run without opening a window (still renders, presents nowhere)
#   --fullscreen          Run in fullscreen mode
#   --log-level N         Python logging level (default: 20 = INFO)

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
    ibv_name,
    ibv_port,
    use_sensor,
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
    hl.use_sensor(metadata, use_sensor)
    channel = hl.DataChannel(metadata)

    hololink_dev = channel.hololink()
    hololink_dev.start()
    hololink_dev.reset()

    camera = hl.NativeVb1940Sensor(channel)

    # Manual sensor-reset sequence -- matches vb1940_player.cpp exactly; VB1940's reset pin is
    # driven directly via a board register, unlike FUSA CoE boards where the FPGA handles this
    # internally (see fusa_coe_vb1940_player.cpp, which has no equivalent of this block).
    hololink_dev.write_uint32(0x8, 0x0)  # Keep the sensor RESET at low
    camera.setup_clock()
    # Release the sensor RESET to high
    hololink_dev.write_uint32(0x8, 0x1 if use_sensor == 0 else 0x2)
    time.sleep(100 / 1000)
    camera.get_register_32(0x0000)  # DEVICE_MODEL_ID:"S940"(ASCII code:0x53393430)
    camera.get_register_32(0x0734)  # EXT_CLOCK(25MHz = 0x017d7840)
    camera.configure(camera_mode)

    pixel_format = camera.get_pixel_format()
    bayer_format = camera.get_bayer_format()
    frame_width = camera.get_width()
    frame_height = camera.get_height()

    geometry = hl.resolve_generic_geometry(camera)

    # Build the graph.
    graph = hs.Graph("vb1940_player")

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
        # default; imx274_player.py's own 50 is sensor-specific, not a shared constant).
        optical_black=8,
        cuda_device_ordinal=cuda_device_ordinal,
    )
    graph.add_flow(csi.tensor, image_processor.in_tensor, queue_depth=4)
    graph.add_flow(csi.metadata, image_processor.in_metadata, queue_depth=4)

    # image_processor.metadata is not connected: nothing uses it, and an
    # output that is not connected has no queue that can fill.

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
            "title": f"Holoviz - {hololink_ip}",
            "width": frame_width,
            "height": frame_height,
            "fullscreen": fullscreen,
            "close_on_escape": True,
            "framebuffer_srgb": True,
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
        # One frame per received frame.
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
        hololink_dev.stop()
        _cuda_check(cuda_driver.cuDevicePrimaryCtxRelease(cu_device))
        return 1

    print(
        f"Capturing mode={camera_mode} {frame_width}x{frame_height} -- running (Ctrl+C to stop)..."
    )

    with hs.run_async(plan) as session:
        # With --frame-limit, the receiver ends the run after frame_limit frames.
        session.wait()

    hololink_dev.stop()
    _cuda_check(cuda_driver.cuDevicePrimaryCtxRelease(cu_device))

    print("Done.")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera-mode",
        type=int,
        default=0,
        help="VB1940 mode (default: 0 = 2560x1984@30fps)",
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
    parser.add_argument(
        "--use-sensor",
        type=int,
        default=0,
        choices=(0, 1),
        help="Use sensor 0 or 1",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    return run(
        args.hololink,
        args.camera_mode,
        args.ibv_name,
        args.ibv_port,
        args.use_sensor,
        args.frame_limit,
        args.headless,
        args.fullscreen,
    )


if __name__ == "__main__":
    sys.exit(main())
