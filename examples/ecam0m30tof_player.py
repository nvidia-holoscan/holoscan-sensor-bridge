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

# NOTE: This example isn't supported on Holoscan SDK 5 yet; it will be supported in an upcoming update to release 3.0.
#
# e-CAM0M30_TOF depth/IR sensor via RoceReceiverOp (RDMA/InfiniBand), displayed live with
# Holoviz. Python port of this repo's own previous (HSDK4-era) Python version of this app, via
# the HSDK5 Python bindings. Needed new low-level I2C/SPI/Serializer/Deserializer bindings (the
# sensor driver, python/hololink/sensors/ecam0m30tof/ecam0m30tof.py, is pure Python and talks to
# hardware directly through them, unlike IMX274/VB1940's C++-bound sensors).
#
#   RoceReceiverOp -> CsiToBayerOp -> ImageShiftAndProcessingOperator -> HolovizOp
#
# No ImageProcessorOp/BayerDemosaicOp stage -- matches the old app's own pipeline shape exactly;
# ImageShiftAndProcessingOperator consumes CsiToBayerOp's raw tensor directly (this sensor's
# "image" is itself a depth/IR map, not a Bayer-mosaic color image).
#
# In 2-plane mode (EDEPTH_MODE_DEPTH_IR), ImageShiftAndProcessingOperator splits the frame into
# a depth half and an active-IR half, shown side by side. The old app rendered the depth half
# with HolovizOp's InputType.DEPTH_MAP (a pseudo-colored depth ramp); this HSDK5 SDK version's
# HolovizOp has no equivalent layer kind at all (confirmed against the SDK's own accepted-"kind"
# list: image/geometry/text/lut only) -- the depth half renders as plain grayscale instead. A
# real, visible behavior reduction versus the old app, not an oversight.
#
# --frame-limit has no HSDK5 CountCondition equivalent, so it's implemented with a small native
# Python operator, FrameLimitOp, tapping the receiver's tensor output and counting activations
# (same pattern as every other *_player.py port in this repo).
#
# Dropped, matching established precedent: --configuration (application.config(), no HSDK5
# equivalent, every port in this repo already drops this); --pattern (camera.test_pattern(...)
# -- ECam0M30Tof has no such method at all, so this was already dead/broken in the old app);
# --depth-range (parsed by the old app but never actually passed to camera.configure() anywhere
# in its own main() -- also already dead, always used configure()'s own depth_range=1 default
# regardless of what was passed on the CLI).
#
# Usage:
#   ecam0m30tof_player.py [OPTIONS]
#
# Options:
#   --hololink IP                 IP address of Hololink board (default: 192.168.0.2)
#   --camera-mode N                e-CAM0M30_TOF mode (default: 0 = depth+IR)
#   --ibv-name NAME                IBV device to use (default: the device on the board's network interface)
#   --ibv-port N                   Port number of IBV device (default: 1)
#   --expander-configuration N    I2C expander configuration (0 or 1, default: 0)
#   --ptp-sync                    After reset, wait for PTP time to synchronize
#   --skip-reset                  Don't call reset on the hololink device
#   --frame-limit N               Number of frames to capture (default: unlimited)
#   --headless                    Run without opening a window (still renders, presents nowhere)
#   --fullscreen                  Run in fullscreen mode
#   --log-level N                 Python logging level (default: 20 = INFO)

import argparse
import logging
import signal
import sys
import threading
import time

import cupy as cp
import hololink.sensors.ecam0m30tof  # noqa: F401 -- makes hl.sensors.ecam0m30tof.X resolve
import holoscan as hs
import holoscan_visualization as hz  # noqa: F401 -- registers "holoscan.visualization.HolovizOp"
from cuda.bindings import driver as cuda_driver

import hololink as hl

# below (hololink/__init__.py doesn't import the sensors subpackage itself, and Python doesn't
# expose an unimported submodule as a parent-package attribute).


def _cuda_check(result):
    err = result[0]
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        _, description = cuda_driver.cuGetErrorString(err)
        raise RuntimeError(f"CUDA error: {description}")
    if len(result) == 2:
        return result[1]
    return result[1:]


def write_lmmi_register(camera_idx, data, addr, hololink_dev):
    slow_spi = hololink_dev.get_spi(
        hl.CLNX_SPI_BUS,
        chip_select=0,
        cpol=0,
        cpha=1,
        width=1,
        clock_divisor=0xF,
    )
    request = [0x11, 0x0C]
    while 1:  # Check Ready
        rdval = slow_spi.spi_transaction(bytearray(), bytearray(request), 1)
        rdval_int = int.from_bytes(bytes(rdval), byteorder="little")
        if rdval_int & 0x4:
            break
    LMMI_WRITE = 1
    request = [LMMI_WRITE, 0x0D, data]
    slow_spi.spi_transaction(bytearray(), bytearray(request), 0)
    request = [LMMI_WRITE, 0x0F, addr]
    slow_spi.spi_transaction(bytearray(), bytearray(request), 0)
    request = [LMMI_WRITE, 0x0C, (camera_idx << 4) + 0x1]
    slow_spi.spi_transaction(bytearray(), bytearray(request), 0)


def reconfigurable_mipi(
    camera_idx, lane_prog, clock_lane_settle_cycle, data_lane_settle_cycle, hololink_dev
):
    logging.debug(lane_prog, clock_lane_settle_cycle, data_lane_settle_cycle)

    reg0A = ((clock_lane_settle_cycle & 0x1) << 3) + (lane_prog << 1)
    reg0B = clock_lane_settle_cycle >> 1
    reg0C = (clock_lane_settle_cycle >> 2) & 0x80
    reg0F = (data_lane_settle_cycle & 0x3) << 2
    reg10 = (data_lane_settle_cycle >> 2) & 0xF

    logging.debug(reg0A, reg0B, reg0C, reg0F, reg10)

    write_lmmi_register(camera_idx, reg0A, 0x0A, hololink_dev)
    write_lmmi_register(camera_idx, reg0B, 0x0B, hololink_dev)
    write_lmmi_register(camera_idx, reg0C, 0x0C, hololink_dev)
    write_lmmi_register(camera_idx, reg0F, 0x0F, hololink_dev)
    write_lmmi_register(camera_idx, reg10, 0x10, hololink_dev)


def program_mipi_phy(camera_idx, num_lanes, clock_freq, hololink_dev):
    lane_prog = num_lanes - 1
    if clock_freq >= 1000:
        data_lane_settle_cycle = 6
    elif clock_freq >= 350:
        data_lane_settle_cycle = 7
    elif clock_freq >= 200:
        data_lane_settle_cycle = 8
    elif clock_freq >= 150:
        data_lane_settle_cycle = 9
    else:
        data_lane_settle_cycle = 11
    clock_lane_settle_cycle = 9
    reconfigurable_mipi(
        camera_idx,
        lane_prog,
        clock_lane_settle_cycle,
        data_lane_settle_cycle,
        hololink_dev,
    )


def convert_depth_to_grayscale(depth_image):
    # Normalize the depth image to the range 0 to 255
    # 1. First, normalize the values from 0 to 65535 to 0 to 1
    depth_normalized = depth_image.astype(cp.float32) / 65535.0
    # 2. Scale it to the range 0 to 255 (for 8-bit grayscale)
    depth_grayscale = cp.clip(depth_normalized * 255, 0, 255).astype(cp.uint8)
    return depth_grayscale


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


class ImageShiftAndProcessingOperator(hs.Operator):
    """Converts the raw uint16 depth/IR tensor to uint8 grayscale; in 2-plane (depth+IR) mode,
    splits it into a depth half and a doubled active-IR half (replicated across 3 channels,
    matching the old app's own RGB color-map construction), emitted on separate ports for
    HolovizOp's side-by-side dual-view layout. See the file-level comment for why the depth
    half is plain grayscale here, not a pseudo-colored depth ramp."""

    # CsiToBayerOp's own "tensor" output (this operator's producer) and HolovizOp's "image"-kind
    # layer inputs are both schema-typed (holoscan.schema.Image), not a bare hs.payload.Tensor --
    # confirmed via GRAPH_PORT_TYPE_MISMATCH when this was first written with the latter.
    input = hs.Input("input", hs.schema.Image)
    image = hs.Output("image", hs.schema.Image)
    depth = hs.Output("depth", hs.schema.Image)

    no_of_planes = hs.Param(1)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._color_map = cp.zeros((480, 640, 3), cp.uint8)

    def setup(self, spec):
        spec.input(self.input, queue_depth=4)
        spec.output(self.image)
        if self.no_of_planes == 2:
            spec.output(self.depth)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, context):
        # self.input is schema-typed (holoscan.schema.Image, not a bare Tensor -- see its own
        # comment above): receive_data() returns the materialized Image struct, whose own
        # `data` field (not the Image object itself) carries the actual DLPack-compatible
        # Tensor (see image.fbs's own `data: Tensor` field doc).
        image_value = self.input.receive_data()
        cp_frame = cp.from_dlpack(image_value.data)
        cp_frame_uint8 = convert_depth_to_grayscale(cp_frame)
        if self.no_of_planes == 1:
            self.image.emit(
                hs.schema.Image(
                    height=480,
                    width=640,
                    # hs.schema.ImageEncoding does not resolve -- the schema module's
                    # __getattr__ only lazily re-exports a fixed set of top-level names
                    # ("Image", "Imu"; see holoscan/schema/__init__.py's own
                    # _OPTIONAL_TYPES), not the Image submodule's own ImageEncoding enum.
                    # The real location is the submodule itself (confirmed by reading it).
                    encoding=hs.schema.image.ImageEncoding.MONO8,
                    data=cp_frame_uint8,
                )
            )
        else:
            depth = cp_frame_uint8[-1::-2, :, :]
            active_ir = cp_frame_uint8[1::2, :, 0] * 2
            self._color_map[:, :, 0] = cp.squeeze(active_ir)
            self._color_map[:, :, 1] = cp.squeeze(active_ir)
            self._color_map[:, :, 2] = cp.squeeze(active_ir)
            self.depth.emit(
                hs.schema.Image(
                    height=480,
                    width=640,
                    encoding=hs.schema.image.ImageEncoding.MONO8,
                    data=depth,
                )
            )
            self.image.emit(
                hs.schema.Image(
                    height=480,
                    width=640,
                    encoding=hs.schema.image.ImageEncoding.RGB8,
                    data=self._color_map,
                )
            )


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
    expander_configuration,
    ptp_sync,
    skip_reset,
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
    if not skip_reset:
        hololink_dev.reset()
    if ptp_sync:
        logging.debug("Waiting for PTP sync.")
        if not hololink_dev.ptp_synchronize():
            logging.error("Failed to synchronize PTP; ignoring.")
        else:
            logging.debug("PTP synchronized.")

    program_mipi_phy(0, 2, 270, hololink_dev)
    program_mipi_phy(1, 2, 270, hololink_dev)

    camera = hl.sensors.ecam0m30tof.ECam0M30Tof(
        channel, expander_configuration=expander_configuration
    )
    camera.setup_clock()
    time.sleep(1)
    camera.configure(camera_mode)

    no_of_planes = (
        2
        if camera_mode == hl.sensors.ecam0m30tof.ECam0M30Tof_Mode.EDEPTH_MODE_DEPTH_IR
        else 1
    )
    frame_width = camera._width
    frame_height = camera._height
    geometry = hl.resolve_generic_geometry_from_dimensions(
        int(camera.pixel_format()), frame_width, frame_height
    )

    if frame_limit:
        _FrameLimit.reset(frame_limit)

    # Build the graph.
    graph = hs.Graph("ecam0m30tof_player")

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
    )

    if frame_limit:
        frame_limit_op = graph.op(FrameLimitOp, "frame_limit")
        graph.add_flow(rx.tensor, frame_limit_op.input, queue_depth=4)

    csi = graph.op(
        hl.operators.CsiToBayerOp,
        "csi_to_bayer",
        cuda_device_ordinal=cuda_device_ordinal,
        geometry=geometry,
        # TODO(ecam0m30tof driver): bayer_format needs to be depth- or IR-specific; the sensor
        # driver itself hardcodes GBRG as a placeholder (see its own bayer_format() comment) --
        # matched here rather than inventing a different placeholder.
        bayer_format="gbrg",
    )
    graph.add_flow(rx.tensor, csi.in_tensor, queue_depth=4)
    graph.add_flow(rx.metadata, csi.in_metadata, queue_depth=4)

    image_shift = graph.op(
        ImageShiftAndProcessingOperator, "image_shift", no_of_planes=no_of_planes
    )
    graph.add_flow(csi.tensor, image_shift.input, queue_depth=4)

    if no_of_planes == 2:
        layers = [
            {
                "port_name": "depth",
                "kind": "image",
                "format": "r8_unorm",
                "max_width": frame_width,
                "max_height": frame_height // 2,
                "view": {
                    "offset_x": 0.0,
                    "offset_y": 0.0,
                    "scale_x": 0.5,
                    "scale_y": 1.0,
                },
            },
            {
                "port_name": "image",
                "kind": "image",
                "format": "rgb8_unorm",
                "max_width": frame_width,
                "max_height": frame_height // 2,
                "view": {
                    "offset_x": 0.5,
                    "offset_y": 0.0,
                    "scale_x": 0.5,
                    "scale_y": 1.0,
                },
            },
        ]
    else:
        layers = [
            {
                "port_name": "image",
                "kind": "image",
                "format": "r8_unorm",
                "max_width": frame_width,
                "max_height": frame_height,
            }
        ]

    if headless:
        # Renders into a CUDA image on HolovizOp's output (consumed by DiscardImageOp).
        sink = {
            "kind": "cuda",
            "width": frame_width,
            "height": frame_height if no_of_planes == 1 else frame_height // 2,
            "format": "rgba8_unorm",
        }
    else:
        sink = {
            "kind": "window",
            "title": f"Holoviz - {hololink_ip}",
            "width": frame_width,
            "height": frame_height if no_of_planes == 1 else frame_height // 2,
            "fullscreen": fullscreen,
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
    graph.add_flow(
        image_shift.image,
        viz.image,
        queue_policy=hs.QueuePolicy.DROP_OLDEST,
        queue_depth=1,
    )
    if no_of_planes == 2:
        graph.add_flow(
            image_shift.depth,
            viz.depth,
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
    options.deployment.bind_tensor_output_device(
        operator_path="csi_to_bayer",
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
    modes = hl.sensors.ecam0m30tof.ECam0M30Tof_Mode
    mode_choices = [mode.value for mode in modes]
    mode_help = " ".join([f"{mode.value}:{mode.name}" for mode in modes])
    parser.add_argument(
        "--camera-mode",
        type=int,
        choices=mode_choices,
        default=mode_choices[0],
        help=mode_help,
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
        "--expander-configuration",
        type=int,
        default=0,
        choices=(0, 1),
        help="I2C Expander configuration",
    )
    parser.add_argument(
        "--ptp-sync",
        action="store_true",
        help="After reset, wait for PTP time to synchronize.",
    )
    parser.add_argument(
        "--skip-reset",
        action="store_true",
        help="Don't call reset on the hololink device.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    camera_mode = hl.sensors.ecam0m30tof.ECam0M30Tof_Mode(args.camera_mode)

    return run(
        args.hololink,
        camera_mode,
        args.ibv_name,
        args.ibv_port,
        args.expander_configuration,
        args.ptp_sync,
        args.skip_reset,
        args.frame_limit,
        args.headless,
        args.fullscreen,
    )


if __name__ == "__main__":
    sys.exit(main())
