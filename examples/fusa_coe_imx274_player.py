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

# FuSa CoE live viewer -- one or two IMX274 sensors via FusaCoeCaptureOp, displayed live with
# Holoviz. Python port of examples/fusa_coe_imx274_player.cpp -- see that file's own comments
# for the pipeline shape; this file mirrors its logic, via the HSDK5 Python bindings instead of
# C++, but keeps the CLI surface of this repo's own previous (HSDK4-era) Python version of this
# app: --frame-limit, --skip-reset, --interface, --width/--height and a camera-mode default of
# 1 (1920x1080@60fps) all come from that file, not from fusa_coe_imx274_player.cpp (whose CLI
# defaults differ slightly -- e.g. --camera-mode default 0 -- and which additionally supports
# --metadata; --metadata has no equivalent here, since MetadataPrinterOp is a local C++-only
# helper operator with no Python binding, and this repo's previous Python version of this app
# didn't have the flag either).
#
#   FusaCoeCaptureOp -> [PackedFormatConverterOp | CsiToBayerOp] -> ImageProcessorOp ->
#     BayerDemosaicOp -> HolovizOp
#
# ImageProcessorOp already emits a schema::ImageT (16-bit single-channel Bayer mosaic, tagged
# with its Bayer encoding by the converter stage upstream and forwarded through unchanged).
# BayerDemosaicOp accepts 16-bit ImageT input directly and produces RGBA16 output -- no
# intermediate 16-bit->8-bit narrowing operator needed (fusa_coe_imx274_player.cpp still
# carries its own local BayerImageExtractorOp for this, a narrowing step the sibling
# fusa_coe_vb1940_player.py port already found obsolete; skipped here for the same reason, and
# also because BayerImageExtractorOp has no Python binding of its own to port).
#
# The unpack stage depends on the sensor mode's pixel format: RAW_8 modes are always captured
# unpacketized (the HSB packetizer only has programs for RAW_10/RAW_12), so those use
# CsiToBayerOp; RAW_10/RAW_12 modes default to packetized capture (override with
# --no-packetizer), which uses PackedFormatConverterOp instead.
#
# --frame-limit is the frame_limit of each capture operator, which also sets frame_limit_ends_run:
# the run ends once every capture has delivered that many frames.
#
# Usage:
#   fusa_coe_imx274_player.py [OPTIONS]
#
# Options:
#   --hololink IP       IP address of Hololink board (default: 192.168.0.2)
#   --sensor N          Sensor index: 0 or 1, or -1 (default) for both sensors side by side
#   --camera-mode N     IMX274 mode (default: 1 = 1920x1080@60fps); the 4k modes (0, 2)
#                       require a single sensor, since two 4k streams overrun the network
#   --timeout N         Per-frame capture timeout in milliseconds (default: 1500)
#   --frame-limit N     Number of frames to capture per sensor (default: unlimited)
#   --skip-reset        Skip reset (only valid when using one sensor)
#   --interface NAME    Override the Ethernet interface (default: read from board metadata)
#   --no-packetizer     Capture unpacketized CSI-2 data instead of HSB-packed data
#   --headless          Run without opening a window (still renders, presents nowhere)
#   --width N           Window width, per sensor (default: 960)
#   --height N          Window height (default: 540)
#   --log-level N       Python logging level (default: 20 = INFO)

import argparse
import logging
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
    sensor,
    camera_mode,
    timeout_ms,
    frame_limit,
    skip_reset,
    interface_override,
    no_packetizer,
    headless,
    width,
    height,
):
    cuda_device_ordinal = 0

    print("Initializing CUDA...")
    _cuda_check(cuda_driver.cuInit(0))
    cu_device = _cuda_check(cuda_driver.cuDeviceGet(cuda_device_ordinal))
    cu_context = _cuda_check(cuda_driver.cuDevicePrimaryCtxRetain(cu_device))
    hololink_dev = None
    try:
        _cuda_check(cuda_driver.cuCtxSetCurrent(cu_context))

        # sensor == -1 selects both sensors, side by side; sensor == 0 or 1 selects just one.
        sensor_ids = [0, 1] if sensor == -1 else [sensor]

        print(f"Enumerating HSB board at {hololink_ip}...")
        # hl.Metadata has no Python-exposed copy constructor, so each sensor's DataChannel is
        # built from its own fresh enumeration instead of copying one shared Metadata object (the
        # approach examples/fusa_coe_imx274_player.cpp uses) -- functionally equivalent, just one
        # extra network round trip in stereo mode.
        metadatas = []
        channels = []
        for sensor_id in sensor_ids:
            metadata = hl.find_channel(hololink_ip)
            hl.use_sensor(metadata, sensor_id)
            metadatas.append(metadata)
            channels.append(hl.DataChannel(metadata))

        interface = interface_override or metadatas[0].get_string("interface") or ""
        mac_str = metadatas[0].get_string("mac_id") or ""
        mac_addr = [int(byte, 16) for byte in mac_str.split(":")]
        if len(mac_addr) != 6:
            raise ValueError("MAC address must have exactly 6 bytes")

        hololink_dev = channels[0].hololink()
        hololink_dev.start()
        if sensor == -1 or not skip_reset:
            hololink_dev.reset()

        cameras = []
        for i, sensor_id in enumerate(sensor_ids):
            # expander_configuration selects the I2C expander output (OUTPUT_1 for sensor 0,
            # OUTPUT_2 for sensor 1) and must match the data channel's sensor id, not the loop
            # index -- otherwise --sensor 1 drives sensor 0's I2C path.
            camera = hl.NativeImx274Sensor(
                channels[i], expander_configuration=sensor_id
            )
            if i == 0:
                camera.setup_clock()
            camera.configure(camera_mode)
            camera.set_digital_gain_reg(4)
            cameras.append(camera)

        # Build the graph.
        graph = hs.Graph("fusa_coe_imx274_player")

        view_width = 1.0 / float(len(cameras))
        layers = []
        for i, camera in enumerate(cameras):
            layers.append(
                {
                    "port_name": f"video_{i}",
                    "kind": "image",
                    "format": "rgba16_unorm",
                    "max_width": camera.get_width(),
                    "max_height": camera.get_height(),
                    "view": {
                        "offset_x": view_width * i,
                        "offset_y": 0.0,
                        "scale_x": view_width,
                        "scale_y": 1.0,
                    },
                }
            )

        if headless:
            # Renders into a CUDA image on HolovizOp's output (consumed by DiscardImageOp). The
            # CUDA sink has no framebuffer_srgb: its sRGB format encodes on write instead.
            sink = {
                "kind": "cuda",
                "width": width * len(cameras),
                "height": height,
                "format": "rgba8_srgb",
            }
        else:
            sink = {
                "kind": "window",
                "framebuffer_srgb": True,
                "title": "FUSA CoE IMX274 Player",
                "width": width * len(cameras),
                "height": height,
                "fullscreen": False,
                "close_on_escape": True,
            }
        viz = graph.op(
            "holoscan.visualization.HolovizOp",
            "holoviz",
            layers=layers,
            sink=sink,
            trigger={"kind": "layer", "layer": 0},
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
        image_output_operators = []

        for i, camera in enumerate(cameras):
            name = str(i)

            pixel_format = camera.get_pixel_format()
            bayer_format = camera.get_bayer_format()

            # The packetizer only has programs for RAW_10 and RAW_12, so an 8-bit mode is always
            # captured unpacketized. `hl` (the new nanobind module) exposes pixel formats as plain
            # ints with no bound PixelFormat enum to compare against -- RAW_8 is 0 in
            # hololink::csi::PixelFormat (see src/hololink/core/csi_formats.hpp), matched directly.
            packetizer_enabled = not no_packetizer and pixel_format != 0

            geometry = hl.resolve_geometry(camera, packetizer_enabled)

            capture = graph.op(
                hl.operators.FusaCoeCaptureOp,
                f"capture_{name}",
                data_channel=channels[i],
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

            if packetizer_enabled:
                converter = graph.op(
                    hl.operators.PackedFormatConverterOp,
                    f"packed_format_converter_{name}",
                    cuda_device_ordinal=cuda_device_ordinal,
                    bayer_format=bayer_format,
                    geometry=geometry,
                )
            else:
                converter = graph.op(
                    hl.operators.CsiToBayerOp,
                    f"csi_to_bayer_{name}",
                    cuda_device_ordinal=cuda_device_ordinal,
                    geometry=geometry,
                    bayer_format=bayer_format,
                    # Fed by FusaCoeCaptureOp, which already strips the CSI header before emitting.
                    header_included=False,
                )

            graph.add_flow(capture.tensor, converter.in_tensor, queue_depth=4)
            graph.add_flow(capture.metadata, converter.in_metadata, queue_depth=4)

            image_processor = graph.op(
                hl.operators.ImageProcessorOp,
                f"image_processor_{name}",
                pixel_format=pixel_format,
                pixel_width=geometry["pixel_width"],
                pixel_height=geometry["pixel_height"],
                bayer_format=_BAYER_FORMAT_VALUES[bayer_format],
                optical_black=0,
                cuda_device_ordinal=cuda_device_ordinal,
            )
            graph.add_flow(converter.tensor, image_processor.in_tensor, queue_depth=4)
            graph.add_flow(
                converter.metadata, image_processor.in_metadata, queue_depth=4
            )

            demosaic = graph.op(
                hv.operators.BayerDemosaicOp,
                f"bayer_demosaic_{name}",
                bayer_grid_pos=bayer_format,
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

            tensor_output_operators.append(capture.name)
            tensor_output_operators.append(converter.name)
            tensor_output_operators.append(image_processor.name)
            image_output_operators.append(demosaic.name)
            graph.partition(f"capture_{name}").add(capture)

        graph.auto_partition_remaining()
        clock_key = graph.add_clock(hs.RealtimeClock, "clock")
        graph.set_default_clock(clock_key)

        options = hs.CompileOptions()
        for operator_path in tensor_output_operators:
            options.deployment.bind_tensor_output_device(
                operator_path=operator_path,
                output_port="tensor",
                device=hs.execution.DeviceId(cuda_device_ordinal),
            )
        for operator_path in image_output_operators:
            options.deployment.bind_tensor_output_device(
                operator_path=operator_path,
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
            f"Capturing {len(cameras)} sensor(s), mode={camera_mode} "
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
        "--camera-mode",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="IMX274 mode (default: 1 = 1920x1080@60fps); 0 and 2 are 3840x2160 and "
        "require --sensor 0 or 1",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Number of frames to capture per sensor",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1500,
        help="Per-frame capture timeout in milliseconds",
    )
    parser.add_argument(
        "--hololink", default="192.168.0.2", help="IP address of Hololink board"
    )
    parser.add_argument(
        "--log-level", type=int, default=20, help="Python logging level to display"
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run without opening a window"
    )
    parser.add_argument(
        "--width", type=int, default=960, help="Window width (per sensor)"
    )
    parser.add_argument("--height", type=int, default=540, help="Window height")
    parser.add_argument(
        "--sensor",
        type=int,
        choices=[-1, 0, 1],
        default=-1,
        help="Sensor to use (-1 for stereo, the default)",
    )
    parser.add_argument(
        "--skip-reset",
        action="store_true",
        help="Skip reset (only valid when using one sensor)",
    )
    parser.add_argument("--interface", default=None, help="Ethernet interface")
    parser.add_argument(
        "--no-packetizer",
        action="store_true",
        help="Capture unpacketized CSI-2 data instead of HSB-packed data",
    )
    args = parser.parse_args()

    # Two 4k streams exceed the network bandwidth and take the interface down.
    if args.sensor == -1 and args.camera_mode in (0, 2):
        parser.error(
            "--camera-mode 0 and 2 (3840x2160) require a single camera; "
            "select one with --sensor 0 or --sensor 1"
        )

    logging.basicConfig(level=args.log_level)

    return run(
        args.hololink,
        args.sensor,
        args.camera_mode,
        args.timeout,
        args.frame_limit,
        args.skip_reset,
        args.interface,
        args.no_packetizer,
        args.headless,
        args.width,
        args.height,
    )


if __name__ == "__main__":
    sys.exit(main())
