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

# FuSa CoE live viewer -- single VB1940 sensor via FusaCoeCaptureOp, displayed live with
# Holoviz. Python port of examples/fusa_coe_vb1940_player.cpp -- see that file's own
# comments for the pipeline shape; this file mirrors its logic exactly, just via the HSDK5
# Python bindings instead of C++.
#
#   FusaCoeCaptureOp -> CsiToBayerOp -> ImageProcessorOp -> BayerDemosaicOp -> HolovizOp
#
# ImageProcessorOp already emits a schema::ImageT (16-bit single-channel Bayer mosaic, tagged
# with its Bayer encoding by CsiToBayerOp upstream and forwarded through unchanged).
# BayerDemosaicOp accepts 16-bit ImageT input directly and produces RGBA16 output -- no
# intermediate 16-bit->8-bit narrowing operator needed.
#
# Usage:
#   fusa_coe_vb1940_player.py [OPTIONS]
#
# Options:
#   --hololink IP       IP address of Hololink board (default: 192.168.0.2)
#   --sensor N          Sensor index: 0 or 1 (default: 0)
#   --camera-mode N     VB1940 mode (default: 0 = 2560x1984@30fps)
#   --timeout N         Per-frame capture timeout in milliseconds (default: 1500)
#   --bayer-grid-pos N  Override the demosaic CFA registration: 0=BGGR, 1=RGGB, 2=GBRG,
#                       3=GRBG (default: -1, use the sensor-reported value)
#   --headless          Run without opening a window (still renders, presents nowhere)
#   --fullscreen        Run in fullscreen mode
#   --frame-limit N     Number of frames to capture (default: unlimited)

import argparse
import sys

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
    sensor_id,
    camera_mode,
    timeout_ms,
    bayer_grid_pos_override,
    headless,
    fullscreen,
    frame_limit=None,
):
    cuda_device_ordinal = 0

    print("Initializing CUDA...")
    _cuda_check(cuda_driver.cuInit(0))
    cu_device = _cuda_check(cuda_driver.cuDeviceGet(cuda_device_ordinal))
    cu_context = _cuda_check(cuda_driver.cuDevicePrimaryCtxRetain(cu_device))
    cuda_driver.cuCtxSetCurrent(cu_context)

    hololink_dev = None
    try:
        print(f"Enumerating HSB board at {hololink_ip}...")
        metadata = hl.find_channel(hololink_ip)
        hl.use_sensor(metadata, sensor_id)
        channel = hl.DataChannel(metadata)

        interface = metadata.get_string("interface") or ""
        mac_str = metadata.get_string("mac_id") or ""
        mac_addr = [int(byte, 16) for byte in mac_str.split(":")]
        if len(mac_addr) != 6:
            raise ValueError("MAC address must have exactly 6 bytes")

        hololink_dev = channel.hololink()
        hololink_dev.start()
        hololink_dev.reset()

        camera = hl.NativeVb1940Sensor(channel)
        camera.setup_clock()
        camera.configure(camera_mode)

        pixel_format = camera.get_pixel_format()
        bayer_format = camera.get_bayer_format()
        frame_width = camera.get_width()
        frame_height = camera.get_height()

        # --bayer-grid-pos overrides the CFA phase used by every *software* interpretation of the
        # mosaic -- ImageProcessorOp's white-balance histogram bucketing and BayerDemosaicOp's
        # interpolation both have to agree on the same phase, or the two stages disagree about
        # which samples are R/G/B and no single demosaic-side override can produce correct colors
        # on its own. FusaCoeCaptureOp's bayer_format below is intentionally NOT overridden: it
        # programs the actual capture hardware/NvSciBuf format and must stay the camera-reported
        # value regardless of this override.
        bayer_grid_pos_value = (
            _BAYER_FORMAT_NAMES[bayer_grid_pos_override]
            if bayer_grid_pos_override >= 0
            else bayer_format
        )

        # Always capture unpacketized CSI-2 -- see the file-level comment for why.
        packetizer_enabled = False

        # Resolve geometry ONCE using FusaCoeCaptureOp's HSB-alignment-aware adapter, and reuse
        # that same geometry dict for both FusaCoeCaptureOp and CsiToBayerOp below. FUSA's 64-byte
        # HSB line alignment differs from the generic adapter CameraSensor otherwise uses (8-byte
        # alignment) -- resolving twice would give CsiToBayerOp a geometry that doesn't match the
        # byte layout FusaCoeCaptureOp actually wrote, causing misparsing. geometry["pixel_format"]
        # (a string) is also reused directly as FusaCoeCaptureOp's own pixel_format= kwarg below:
        # both represent the same camera.get_pixel_format() value, just already resolved to the
        # schema's string spelling.
        geometry = hl.resolve_geometry(camera, packetizer_enabled)

        # Build the graph.
        graph = hs.Graph("fusa_coe_vb1940_player")

        capture = graph.op(
            hl.operators.FusaCoeCaptureOp,
            "capture",
            data_channel=channel,
            device_start=camera.start,
            device_stop=camera.stop,
            interface=interface,
            mac_addr=mac_addr,
            timeout=timeout_ms,
            pixel_format=geometry["pixel_format"],
            bayer_format=bayer_format,
            packetizer_enabled=packetizer_enabled,
            geometry=geometry,
            frame_limit=frame_limit or 0,
            frame_limit_ends_run=True,
        )

        csi = graph.op(
            hl.operators.CsiToBayerOp,
            "csi_to_bayer",
            cuda_device_ordinal=cuda_device_ordinal,
            geometry=geometry,
            bayer_format=bayer_format,
            # Fed by FusaCoeCaptureOp, which already strips the CSI header before emitting.
            header_included=False,
        )

        image_processor = graph.op(
            hl.operators.ImageProcessorOp,
            "image_processor",
            pixel_format=pixel_format,
            pixel_width=geometry["pixel_width"],
            pixel_height=geometry["pixel_height"],
            bayer_format=_BAYER_FORMAT_VALUES[bayer_grid_pos_value],
            optical_black=0,
            cuda_device_ordinal=cuda_device_ordinal,
        )

        # Same CFA phase as image_processor's bayer_format above (see bayer_grid_pos_value) -- the
        # two stages must agree, or the white-balance gains and the demosaic interpolation disagree
        # about which raw samples are R/G/B.
        demosaic = graph.op(
            hv.operators.BayerDemosaicOp,
            "bayer_demosaic",
            bayer_grid_pos=bayer_grid_pos_value,
            # HolovizOp's kR16G16B16A16Unorm layer is RGBA -- generate an (unused, fully opaque)
            # alpha channel to match. image_processor's output is 16-bit, so demosaic's output is
            # RGBA16 too (output bit depth matches input); alpha_value must be a full-scale 16-bit
            # value on that path, not the 8-bit-path default of 255.
            generate_alpha=True,
            alpha_value=65535,
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
                "title": "FUSA CoE VB1940 Player",
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
            # max_input_age_ns intentionally omitted -- its 100ms default matches the C++ example.
        )
        if headless:
            discard = graph.op(DiscardImageOp, "discard")
            graph.add_flow(
                viz.output,
                discard.input,
                queue_policy=hs.QueuePolicy.DROP_OLDEST,
                queue_depth=1,
            )

        graph.add_flow(capture.tensor, csi.in_tensor, queue_depth=4)
        graph.add_flow(capture.metadata, csi.in_metadata, queue_depth=4)
        graph.add_flow(csi.tensor, image_processor.in_tensor, queue_depth=4)
        graph.add_flow(csi.metadata, image_processor.in_metadata, queue_depth=4)
        graph.add_flow(image_processor.tensor, demosaic.input, queue_depth=4)
        graph.add_flow(
            demosaic.output,
            viz.video,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
            queue_depth=1,
        )

        clock_key = graph.add_clock(hs.RealtimeClock, "clock")
        graph.set_default_clock(clock_key)

        graph.partition("capture").add(capture)
        graph.auto_partition_remaining()

        options = hs.CompileOptions()
        # capture/csi_to_bayer/image_processor emit their tensor on a port named "tensor";
        # bayer_demosaic emits a schema::ImageT-wrapped tensor on "output".
        for operator_path in ("capture", "csi_to_bayer", "image_processor"):
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
            f"Capturing sensor {sensor_id} mode={camera_mode} {frame_width}x{frame_height} "
            f"bayer_grid_pos={_BAYER_FORMAT_VALUES[bayer_grid_pos_value]} "
            "-- running (Ctrl+C to stop)..."
        )

        with hs.run_async(plan) as session:
            try:
                session.wait()
            except KeyboardInterrupt:
                session.request_stop()
                session.wait()
            failed = any(
                diag.severity == hs.diagnostics.DiagnosticSeverity.ERROR
                for diag in session.diagnostics
            )
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hololink", default="192.168.0.2", help="IP address of Hololink board"
    )
    parser.add_argument(
        "--sensor", type=int, choices=[0, 1], default=0, help="Sensor index (0 or 1)"
    )
    parser.add_argument("--camera-mode", type=int, default=0, help="VB1940 mode")
    parser.add_argument(
        "--timeout",
        type=int,
        default=1500,
        help="Per-frame capture timeout in milliseconds",
    )
    parser.add_argument(
        "--bayer-grid-pos",
        type=int,
        choices=[-1, 0, 1, 2, 3],
        default=-1,
        help="Override the demosaic CFA registration: 0=BGGR, 1=RGGB, 2=GBRG, 3=GRBG "
        "(default: -1, use the sensor-reported value)",
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run without opening a window"
    )
    parser.add_argument(
        "--fullscreen", action="store_true", help="Run in fullscreen mode"
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Number of frames to capture",
    )
    args = parser.parse_args()

    return run(
        args.hololink,
        args.sensor,
        args.camera_mode,
        args.timeout,
        args.bayer_grid_pos,
        args.headless,
        args.fullscreen,
        args.frame_limit,
    )


if __name__ == "__main__":
    sys.exit(main())
