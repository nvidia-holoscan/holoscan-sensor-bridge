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

# Displays live IMX274 video received over IEEE 1722/AVTP Class-Oriented-Encapsulation (COE)
# packets via LinuxCoeReceiverOp, read from a raw AF_PACKET socket bound to the given network
# interface (not UDP/RoCE like every other already-ported IMX274 player). Python port of this
# repo's own previous (HSDK4-era) Python version of this app, via the HSDK5 Python bindings --
# there is no examples/linux_coe_imx274_player.cpp sibling. hololink.operators.LinuxCoeReceiverOp
# itself is new in this branch: the old app's own pybind11-bound LinuxCoeReceiverOp (a pure-Python
# BaseReceiverOp subclass wrapping the low-level LinuxCoeReceiver helper) was dropped, unported,
# in this session's earlier nanobind rewrite -- this effort instead gave it a proper C++
# Operator<Params> implementation (hololink::operators::LinuxCoeReceiverOp, in
# src/hololink/operators/linux_coe_receiver/linux_coe_receiver_op.{hpp,cpp}), mirroring
# LinuxReceiverOp's own existing BaseReceiverOp-derived shape exactly, plus one small shared-base
# addition: BaseReceiverOp::start() used to hardcode a UDP socket, which doesn't work for COE's
# raw AF_PACKET/AVTP transport, so socket creation is now its own overridable
# create_data_socket() hook (default behavior unchanged for every other receiver).
#
# To receive COE packets, create a VLAN interface tagged for VETH 0xCCC (3276) on top of your
# real network interface, then point --coe-interface at it:
#
#   N0=enP5p3s0f0np0  # whatever network you're attached to
#   E0=coe.0
#   sudo ip link add link $N0 name $E0 type vlan id 3276
#   sudo ifconfig $E0 up
#   python3 examples/linux_coe_imx274_player.py --coe-interface=$E0
#
#   LinuxCoeReceiverOp -> CsiToBayerOp -> ImageProcessorOp -> BayerDemosaicOp -> HolovizOp
#                                                           -> TimingReportOp
#
# The old app's TimingReportOperator read its "overall_time" (capture-to-visualized-frame
# latency) off HolovizOp's own camera_pose_output port (enable_camera_pose_output=True); this
# SDK version's HolovizOp has no such port at all (confirmed against its own real parameter set
# -- layers/sink/trigger only, no camera-pose output). Ported here as a plain metadata tap
# instead, reading image_processor's CaptureMetadata output directly (the same "timestamp"/
# "metadata" FPGA fields the old app's own frame_time calculation already used) and comparing
# against host wall-clock time at receipt -- same two metrics, frame_time and overall_time, just
# measured through image processing rather than all the way through Holoviz rendering (the new
# architecture doesn't propagate CaptureMetadata that far either). PTP sync stays mandatory, as
# in the old app: both timing metrics are meaningless without it.
#
# --stall-timeout-ms sets LinuxCoeReceiverOp's stall_timeout_ms: once that long passes without
# a frame, the receiver fails the run, and the app exits with status 1 after printing the run's
# diagnostics. The old app checked its receiver's capture_stalled() after the run and raised.
# --frame-limit stops the run from FrameLimitOp, in the graph, so a run that fails first ends
# too.
#
# Usage:
#   linux_coe_imx274_player.py [OPTIONS]
#
# Options:
#   --camera-mode N          IMX274 mode (default: 0 = 3840x2160@60fps)
#   --coe-interface NAME     Network interface receiving VETH 3276 (0xCCC) COE packets (required)
#   --expander-configuration N  I2C expander configuration (0 or 1, default: 0)
#   --pattern N              Configure a test pattern (0-11) instead of live video
#   --hololink IP            IP address of Hololink board (default: 192.168.0.2)
#   --frame-limit N          Number of frames to capture (default: unlimited)
#   --stall-timeout-ms N     Fail after this many milliseconds without a CoE frame
#                            (default: 0 = never)
#   --headless               Render into a CUDA image instead of a window
#   --fullscreen             Run in fullscreen mode
#   --log-level N            Python logging level (default: 20 = INFO)

import argparse
import logging
import signal
import sys
import time

import capture_metadata
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


class FrameLimitOp(hs.Operator):
    """Stops the run once the receiver has captured --frame-limit frames."""

    input = hs.Input("input", hs.payload.Tensor)

    frame_limit = hs.Param(0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._frames_seen = 0

    def setup(self, spec):
        spec.input(self.input, queue_depth=4)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, context):
        self.input.receive_data()
        self._frames_seen += 1
        if self.frame_limit > 0 and self._frames_seen >= self.frame_limit:
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


class TimingReportOp(hs.Operator):
    """Logs frame_time (sensor-to-FPGA-metadata-packet time) and overall_time (that same start
    point to this operator's own wall-clock receipt) for a window of frames, skipping an initial
    settling period. Ported from this repo's own HSDK4-era TimingReportOperator; see this file's
    header comment for why it now taps image_processor's CaptureMetadata directly instead of
    HolovizOp's old camera_pose_output port."""

    metadata = hs.Input("metadata", capture_metadata.CaptureMetadata)

    skip_frames = hs.Param(10)
    report_frames = hs.Param(10)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._frames = 0

    def setup(self, spec):
        spec.input(self.metadata, queue_depth=4)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.metadata))

    def compute(self, context):
        complete_s = time.time()

        frame = self.metadata.receive_data()
        self._frames += 1
        if self._frames < self.skip_frames:
            return
        if self._frames >= (self.skip_frames + self.report_frames):
            return

        # frame_start_s is the time the first data for this frame arrived at the FPGA.
        frame_start_s = frame.timestamp_s + frame.timestamp_ns * 1e-9
        # frame_end_s is the time the FPGA sent the trailing metadata packet for this frame.
        frame_end_s = frame.metadata_s + frame.metadata_ns * 1e-9
        # frame_time: how long it took the sensor to produce one complete frame.
        frame_time = frame_end_s - frame_start_s
        # overall_time: capture start through this operator's own receipt (image-processed, not
        # full Holoviz-rendered -- see this file's header comment).
        overall_time = complete_s - frame_start_s
        logging.info(
            f"frame_number={frame.frame_number} {frame_time=:.4f} {overall_time=:.4f}"
        )


def run(
    hololink_ip,
    camera_mode,
    coe_interface,
    expander_configuration,
    pattern,
    frame_limit,
    stall_timeout_ms,
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
    try:
        hololink_dev.reset()

        # Sync PTP; otherwise the timestamps we get from HSB aren't synchronized with us, and both
        # of TimingReportOp's metrics below are meaningless (matches the old app's own mandatory,
        # hard-failing PTP sync).
        print("Waiting for PTP synchronization...")
        if not hololink_dev.ptp_synchronize():
            print("PTP synchronization failed.", file=sys.stderr)
            return 1
        print("PTP synchronized.")

        camera = hl.NativeImx274Sensor(
            channel, expander_configuration=expander_configuration
        )
        camera.setup_clock()
        camera.configure(camera_mode)
        camera.set_digital_gain_reg(0x4)
        if pattern is not None:
            camera.test_pattern(pattern)

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
        graph = hs.Graph("linux_coe_imx274_player")

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
            stall_timeout_ms=stall_timeout_ms,
        )

        if frame_limit:
            frame_limit_op = graph.op(
                FrameLimitOp, "frame_limit", frame_limit=frame_limit
            )
            graph.add_flow(rx.tensor, frame_limit_op.input, queue_depth=4)

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
            # Optical black value for imx274 is 50 (matches this app's own HSDK4-era default).
            optical_black=50,
            cuda_device_ordinal=cuda_device_ordinal,
        )
        graph.add_flow(csi.tensor, image_processor.in_tensor, queue_depth=4)
        graph.add_flow(csi.metadata, image_processor.in_metadata, queue_depth=4)

        timing_report = graph.op(TimingReportOp, "timing_report")
        # A consumer of the metadata alone must never hold the converter back: shed rather than fill.
        graph.add_flow(
            image_processor.metadata,
            timing_report.metadata,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
            queue_depth=4,
        )

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
            return 1

        print(
            f"Capturing mode={camera_mode} {frame_width}x{frame_height} -- running (Ctrl+C to stop)..."
        )

        with hs.run_async(plan) as session:
            try:
                session.wait()
            except KeyboardInterrupt:
                # Ctrl+C or SIGTERM: leaving the block stops the session cooperatively, which stops
                # the camera.
                pass
        if session.diagnostics:
            print(session.summary(), file=sys.stderr)
        failed = session.termination == hs.RunTermination.FAILED
    finally:
        # As the old app did, stop the board and release the context even when bring-up or the
        # run fails.
        try:
            hololink_dev.stop()
        finally:
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
        choices=[0, 1, 2],
        default=0,
        help="IMX274 mode (default: 0 = 3840x2160@60fps)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Render into a CUDA image instead of a window",
    )
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
        "--stall-timeout-ms",
        type=int,
        default=0,
        help="Fail after this many milliseconds without a CoE frame (0 disables)",
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
        choices=(0, 1),
        help="I2C Expander configuration",
    )
    parser.add_argument(
        "--pattern",
        type=int,
        choices=range(12),
        help="Configure to display a test pattern.",
    )
    parser.add_argument(
        "--coe-interface",
        required=True,
        help="Name of interface connected to VETH 3276 (0xCCC).",
    )
    args = parser.parse_args()
    if not 0 <= args.stall_timeout_ms <= 0xFFFFFFFF:
        parser.error(
            f"--stall-timeout-ms must be from 0 to 4294967295, got {args.stall_timeout_ms}"
        )

    logging.basicConfig(level=args.log_level)

    return run(
        args.hololink,
        args.camera_mode,
        args.coe_interface,
        args.expander_configuration,
        args.pattern,
        args.frame_limit,
        args.stall_timeout_ms,
        args.headless,
        args.fullscreen,
    )


if __name__ == "__main__":
    sys.exit(main())
