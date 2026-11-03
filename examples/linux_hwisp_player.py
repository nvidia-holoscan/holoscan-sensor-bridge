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

# Single IMX274 sensor via LinuxReceiverOp, ISP'd on-device by ArgusIspOp (hardware Argus
# reprocessing + NPP NV12->RGB, BT.709) instead of the ImageProcessorOp/BayerDemosaicOp software
# path every other *_player.py port in this repo uses. Python port of this repo's own previous
# (HSDK4-era) Python version of this app, via the HSDK5 Python bindings and ArgusIspOp's new
# manual catalog row (src/hololink/operators/operator_catalog.cpp).
#
#   LinuxReceiverOp -> CsiToBayerOp -> ArgusIspOp -> HolovizOp
#
# ArgusIspOp emits (height, width, 3) uint8 RGB (schema::ImageEncoding_RGB8) directly -- unlike
# every BayerDemosaicOp-fed port in this repo (RGBA16), so the HolovizOp layer format below is
# "rgb8_unorm", not "rgba16_unorm". No ImageProcessorOp/BayerDemosaicOp stage: ArgusIspOp
# consumes CsiToBayerOp's raw Bayer tensor directly and does ISP + demosaic + color conversion
# in one native step. Its metadata output is not connected: nothing uses it, and an output
# that is not connected has no queue that can fill.
#
# --frame-limit is the receiver's frame_limit parameter: the receiver ends the run after that
# many frames. HSDK5 has no CountCondition; this replaces the CountCondition of the old app.
#
# Usage:
#   linux_hwisp_player.py [OPTIONS]
#
# Options:
#   --hololink IP                 IP address of Hololink board (default: 192.168.0.2)
#   --camera-mode N                IMX274 mode (default: platform-detected, falls back to 0)
#   --expander-configuration N    I2C expander configuration (0 or 1, default: 0)
#   --frame-limit N               Number of frames to capture (default: unlimited)
#   --headless                    Run without opening a window (still renders, presents nowhere)
#   --fullscreen                  Run in fullscreen mode
#   --log-level N                 Python logging level (default: 20 = INFO)

import argparse
import logging
import sys

import holoscan as hs
import holoscan_visualization as hz  # noqa: F401 -- registers "holoscan.visualization.HolovizOp"
from cuda.bindings import driver as cuda_driver

import hololink as hl


def _get_platform_default_camera_mode():
    """Mirrors the old app's get_platform_default_mode(): L4T>=39 Jetson hosts get the
    3840x2160@60fps mode (index 0); older hosts get 1920x1080@60fps (index 1). Off-Jetson (e.g.
    this port's own dev container), /sys/class/dmi/id/bios_version doesn't exist -- the old app
    would raise FileNotFoundError; this falls back to index 0 (matching every other IMX274 port
    in this repo's own hardcoded --camera-mode default) instead of carrying that crash over.
    """
    try:
        bios_version = open("/sys/class/dmi/id/bios_version").read().strip()
        l4t_version_major = int(bios_version.split(".")[0])
    except (OSError, ValueError, IndexError):
        return 0
    return 0 if l4t_version_major >= 39 else 1


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
    expander_configuration,
    frame_limit,
    headless,
    fullscreen,
):
    cuda_device_ordinal = 0

    if not hasattr(hl.operators, "ArgusIspOp"):
        print(
            "linux_hwisp_player.py needs ArgusIspOp, which this hololink build does not include: "
            "it is built only on L4T hosts with the Jetson Multimedia API installed "
            "(HOLOLINK_BUILD_ARGUS_ISP).",
            file=sys.stderr,
        )
        return 1

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

    camera = hl.NativeImx274Sensor(
        channel, expander_configuration=expander_configuration
    )
    camera.setup_clock()
    camera.configure(camera_mode)
    camera.set_digital_gain_reg(0x4)

    bayer_format = camera.get_bayer_format()
    frame_width = camera.get_width()
    frame_height = camera.get_height()

    geometry = hl.resolve_generic_geometry(camera)

    # Build the graph.
    graph = hs.Graph("linux_hwisp_player")

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

    argus_isp = graph.op(
        hl.operators.ArgusIspOp,
        "argus_isp",
        bayer_format=bayer_format,
        # 60fps is 16.67ms
        exposure_time_ms=16.67,
        analog_gain=10.0,
        pixel_bit_depth=10,
        sensor_mode_index=camera_mode,
    )
    graph.add_flow(csi.tensor, argus_isp.in_tensor, queue_depth=4)
    graph.add_flow(csi.metadata, argus_isp.in_metadata, queue_depth=4)

    # argus_isp.metadata is not connected: nothing uses it, and an
    # output that is not connected has no queue that can fill.

    if headless:
        # Renders into a CUDA image on HolovizOp's output (consumed by DiscardImageOp).
        sink = {
            "kind": "cuda",
            "width": frame_width,
            "height": frame_height,
            "format": "rgba8_unorm",
        }
    else:
        sink = {
            "kind": "window",
            "title": "Linux HWISP Player",
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
            }
        ],
        sink=sink,
        # One frame per received frame.
        trigger={"kind": "layer"},
    )
    graph.add_flow(
        argus_isp.tensor,
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

    # ArgusIspOp's compute() waits for the Argus ISP (about 10-18 ms for a 4K frame). In one
    # serial partition with the receiver, that wait plus the receiver's wait for the next frame
    # is longer than a frame period, and the receiver fails with kBackpressured. Separate
    # partitions run on separate threads, so the two waits overlap.
    graph.partition("capture").add(rx).add(csi)
    graph.partition("isp").add(argus_isp)
    graph.auto_partition_remaining()

    clock_key = graph.add_clock(hs.RealtimeClock, "clock")
    graph.set_default_clock(clock_key)

    options = hs.CompileOptions()
    for operator_path in ("rx", "csi_to_bayer"):
        options.deployment.bind_tensor_output_device(
            operator_path=operator_path,
            output_port="tensor",
            device=hs.execution.DeviceId(cuda_device_ordinal),
        )
    options.deployment.bind_tensor_output_device(
        operator_path="argus_isp",
        output_port="tensor",
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
        choices=[0, 1, 2],
        default=_get_platform_default_camera_mode(),
        help="IMX274 mode (default: platform-detected, falls back to 0 = 3840x2160@60fps)",
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
        "--expander-configuration",
        type=int,
        default=0,
        choices=(0, 1),
        help="I2C Expander configuration",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    return run(
        args.hololink,
        args.camera_mode,
        args.expander_configuration,
        args.frame_limit,
        args.headless,
        args.fullscreen,
    )


if __name__ == "__main__":
    sys.exit(main())
