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

# 1 to 4 IMX477 sensors (--num-cameras) sharing ONE Microchip PolarFire rev 2 board/IP, demuxed
# via hl.use_sensor(0..3) into independent LinuxReceiverOp (Linux UDP socket, no RoCE/ibverbs
# needed) legs, tiled in one Holoviz window -- the multi-camera variant of
# linux_single_network_stereo_imx477_player.py. Python port of this repo's own previous
# (HSDK4-era) Python version of this app, via the HSDK5 Python bindings; there is no C++ sibling,
# so each leg follows the single-network shape already established by
# examples/linux_single_network_stereo_imx477_player.py.
# single_network_multi_imx477_player.py is the RoCE variant of this app.
#
#   LinuxReceiverOp -> CsiToBayerOp -> ImageProcessorOp -> BayerDemosaicOp -> HolovizOp
#
# --frame-limit has no HSDK5 CountCondition equivalent, so it's implemented with a small native
# Python operator, FrameLimitOp, counting each leg's frames on its image processor's metadata
# output; the one that brings the last leg to the limit stops the run. It stops it from the
# graph, so a run that ends first, for example by failing, doesn't leave the app waiting.
#
# Usage:
#   linux_single_network_multi_imx477_player.py [OPTIONS]
#
# Options:
#   --hololink IP            IP address of Hololink board (default: 192.168.0.2)
#   --num-cameras {1,2,3,4}  Number of cameras to stream; 3 or 4 need a rev 2 board (default: 2)
#   --resolution RES         4k, 1080p, or 720p, shared by every camera (default: 1080p);
#                            3 or 4 cameras always use 720p, 2 cameras use 1080p instead of 4k
#   --pattern                Configure every camera to display a test pattern
#   --exposure N             Configure exposure -- shared by every camera (default: 0x05)
#   --window-height N        Window height per grid row; 2 rows for 2-4 cameras (default: 1080)
#   --window-width N         Displayed window width (default: 1920)
#   --frame-limit N          Number of frames to capture per sensor (default: unlimited)
#   --headless               Render into a CUDA image instead of a window
#   --fullscreen             Run in fullscreen mode
#   --log-level N            Python logging level (default: 20 = INFO)

import argparse
import logging
import signal
import sys
import threading

import capture_metadata
import hololink.sensors.imx477  # noqa: F401 -- makes hl.sensors.imx477.X resolve below
import holoscan as hs
import holoscan_vision as hv
import holoscan_visualization as hz  # noqa: F401 -- registers "holoscan.visualization.HolovizOp"
from cuda.bindings import driver as cuda_driver

import hololink as hl

# (hololink/__init__.py doesn't import the sensors subpackage itself, and Python doesn't expose
# an unimported submodule as a parent-package attribute).

_GRID_OFFSETS = [
    # (offset_x, offset_y, width, height) as fractions of the window, indexed by
    # [num_cameras - 1][camera]; HolovizOp's layer "view" takes width/height as scale_x/scale_y.
    [(0.0, 0.0, 1.0, 1.0)],
    [
        (0.0, 0.0, 1.0, 0.5),  # cam 0: top
        (0.0, 0.5, 1.0, 0.5),  # cam 1: bottom
    ],
    [
        (0.0, 0.0, 0.5, 0.5),  # cam 0: top-left
        (0.5, 0.0, 0.5, 0.5),  # cam 1: top-right
        (0.0, 0.5, 0.5, 0.5),  # cam 2: bottom-left
    ],
    [
        (0.0, 0.0, 0.5, 0.5),  # cam 0: top-left
        (0.5, 0.0, 0.5, 0.5),  # cam 1: top-right
        (0.0, 0.5, 0.5, 0.5),  # cam 2: bottom-left
        (0.5, 0.5, 0.5, 0.5),  # cam 3: bottom-right
    ],
]


def _cuda_check(result):
    err = result[0]
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        _, description = cuda_driver.cuGetErrorString(err)
        raise RuntimeError(f"CUDA error: {description}")
    if len(result) == 2:
        return result[1]
    return result[1:]


class _FrameLimit:
    """Shared state for --frame-limit: the frames each leg has captured."""

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
        """Counts a frame of leg; True once every leg has reached the limit."""
        with cls.lock:
            cls.counts[leg] += 1
            return cls.limit > 0 and all(
                count >= cls.limit for count in cls.counts.values()
            )


class FrameLimitOp(hs.Operator):
    """Count one leg's frames toward --frame-limit, on its image processor's metadata, and stop
    the run once every leg has reached it; see _FrameLimit."""

    metadata = hs.Input("metadata", capture_metadata.CaptureMetadata)
    leg = hs.Param(0)

    def setup(self, spec):
        spec.input(self.metadata, queue_depth=4)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.metadata))

    def compute(self, context):
        self.metadata.receive_data()
        if _FrameLimit.record(self.leg):
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


def run(
    hololink_ip,
    num_cameras,
    resolution,
    pattern,
    exposure,
    window_height,
    window_width,
    frame_limit,
    headless,
    fullscreen,
):
    cuda_device_ordinal = 0
    tensor_names = [f"cam{i}" for i in range(num_cameras)]

    # Force resolution adjustment for multiple cameras.
    if num_cameras > 2:
        resolution = "720p"
    elif num_cameras == 2 and resolution == "4k":
        resolution = "1080p"

    print("Initializing CUDA...")
    _cuda_check(cuda_driver.cuInit(0))
    cu_device = _cuda_check(cuda_driver.cuDeviceGet(cuda_device_ordinal))
    cu_context = _cuda_check(cuda_driver.cuDevicePrimaryCtxRetain(cu_device))
    hololink_dev = None
    try:
        cuda_driver.cuCtxSetCurrent(cu_context)

        # The rev 2 board carries 4 sensors (one SIF each) on 2 data planes. Register that topology
        # for its UUID before enumerating: find_channel() applies the strategy registered for the
        # board's UUID, and use_sensor() range-checks the sensor number against it. Kept as the old
        # app had it; it replaces enumerator.cpp's built-in strategy for this UUID
        # (MICROCHIP_POLARFIRE_R2_UUID), which has the same topology plus board_description and
        # gpio_pin_count metadata. The old pybind11 binding defaulted additional_metadata to an
        # empty Metadata; this one requires it.
        uuid_strategy = hl.BasicEnumerationStrategy(
            hl.Metadata(), total_sensors=4, total_dataplanes=2, sifs_per_sensor=1
        )
        rev2_uuid = "48484948-5045-4848-4853-564845484850"
        hl.set_uuid_strategy(rev2_uuid, uuid_strategy)

        print(f"Enumerating HSB board at {hololink_ip}...")
        channel_metadata = hl.find_channel(hololink_ip)
        # use_sensor() edits the metadata in place, so every sensor gets its own copy.
        channels = []
        cameras = []
        for sensor_id in range(num_cameras):
            metadata = hl.Metadata(channel_metadata)
            hl.use_sensor(metadata, sensor_id)
            channel = hl.DataChannel(metadata)
            channels.append(channel)
            cameras.append(
                hl.sensors.imx477.Imx477(
                    channel, camera_id=sensor_id, resolution=resolution
                )
            )

        # Every channel shares this one Hololink device (one board).
        hololink_dev = channels[0].hololink()
        hololink_dev.start()
        hololink_dev.reset()
        for camera in cameras:
            # Configures the camera for the requested resolution, 60fps.
            camera.configure()
            # IMX477 analog gain settings function. Analog gain value range is 0-1023 in
            # decimal (10 bits). Users are free to experiment with the register values.
            camera.set_analog_gain(0x2FF)
            camera.set_exposure_reg(exposure)
            if pattern:
                camera.set_pattern()

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

        if frame_limit:
            _FrameLimit.reset(len(cameras), frame_limit)

        # Build the graph.
        graph = hs.Graph("linux_single_network_multi_imx477_player")

        # Pass 1: declare one image layer per camera, placed by _GRID_OFFSETS -- the layer list
        # must be complete before HolovizOp is constructed.
        layers = []
        for i, camera in enumerate(cameras):
            offset_x, offset_y, width, height = _GRID_OFFSETS[num_cameras - 1][i]
            layers.append(
                {
                    "port_name": f"video_{tensor_names[i]}",
                    "kind": "image",
                    "format": "rgba16_unorm",
                    "max_width": camera._width,
                    "max_height": camera._height,
                    "view": {
                        "offset_x": offset_x,
                        "offset_y": offset_y,
                        "scale_x": width,
                        "scale_y": height,
                    },
                }
            )

        # --window-height per grid row: 2 rows for 2-4 cameras.
        height = window_height * min(num_cameras, 2)
        if headless:
            # Renders into a CUDA image on HolovizOp's output (consumed by DiscardImageOp). The
            # CUDA sink has no framebuffer_srgb: its sRGB format encodes on write instead.
            sink = {
                "kind": "cuda",
                "width": window_width,
                "height": height,
                "format": "rgba8_srgb",
            }
        else:
            sink = {
                "kind": "window",
                "title": "Holoviz",
                "width": window_width,
                "height": height,
                "fullscreen": fullscreen,
                "close_on_escape": True,
                "framebuffer_srgb": True,
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
                hl.operators.LinuxReceiverOp,
                f"rx_{name}",
                data_channel=channels[i],
                frame_context=int(cu_context),
                device_start=camera.start,
                device_stop=camera.stop,
                frame_size=frame_sizes[i],
            )

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
                optical_black=100,
                cuda_device_ordinal=cuda_device_ordinal,
            )
            graph.add_flow(csi.tensor, image_processor.in_tensor, queue_depth=4)
            graph.add_flow(csi.metadata, image_processor.in_metadata, queue_depth=4)

            # Consume metadata only when counting frames toward --frame-limit.
            if frame_limit:
                metadata_sink = graph.op(FrameLimitOp, f"frame_limit_{name}", leg=i)
                graph.add_flow(
                    image_processor.metadata,
                    metadata_sink.metadata,
                    queue_policy=hs.QueuePolicy.DROP_OLDEST,
                    queue_depth=4,
                )

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

        print(
            f"Capturing {num_cameras} sensors at {resolution} -- running (Ctrl+C to stop)..."
        )

        with hs.run_async(plan) as session:
            try:
                session.wait()
            except KeyboardInterrupt:
                # Ctrl+C or SIGTERM: leaving the block stops the session cooperatively, which
                # stops the cameras.
                pass
        if session.diagnostics:
            print(session.summary(), file=sys.stderr)
        failed = session.termination == hs.RunTermination.FAILED
    finally:
        # Like the old app, stop the board even when bring-up or the run fails.
        try:
            if hololink_dev is not None:
                hololink_dev.stop()
        finally:
            _cuda_check(cuda_driver.cuDevicePrimaryCtxRelease(cu_device))

    if failed:
        return 1
    print("Done.")
    return 0


def main():
    # Like Ctrl+C, SIGTERM (kill, docker stop) raises KeyboardInterrupt, so the session stops
    # cooperatively and the cameras and the board are stopped on the way out.
    signal.signal(signal.SIGTERM, signal.default_int_handler)

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
        default="1080p",
        help="4k, 1080p, or 720p",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=1080,  # arbitrary default
        help="Set the height of the displayed window",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=1920,  # arbitrary default
        help="Set the width of the displayed window",
    )
    parser.add_argument(
        "--num-cameras",
        type=int,
        default=2,
        choices=range(1, 5),
        help="Number of cameras to showcase (up to 4 if using Rev 2 board)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    return run(
        args.hololink,
        args.num_cameras,
        args.resolution,
        args.pattern,
        args.exposure,
        args.window_height,
        args.window_width,
        args.frame_limit,
        args.headless,
        args.fullscreen,
    )


if __name__ == "__main__":
    sys.exit(main())
