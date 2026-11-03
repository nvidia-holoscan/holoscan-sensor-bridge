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

# Single IMX274 sensor via LinuxReceiverOp, displayed live with Holoviz, with per-frame
# validation: CRC32 (checked every --crc-frame-check frames against the FPGA-stamped CRC),
# frame-number ordering, received-vs-calculated frame size, inter-frame timing, and a final
# latency report (sensor frame time / DMA+CPU wakeup / operator dispatch, matching
# linux_imx274_latency.py's own three preservable metrics). Python port of this repo's own
# previous (HSDK4-era) Python version of this app, via the HSDK5 Python bindings. Unlike every
# other CRC-needing app this porting effort has found, this one needs NO new C++ bindings at
# all: the FPGA's own CRC is already available as hololink.CaptureMetadataT's `crc` field (see
# src/hololink/operators/capture_metadata.fbs), so the check is done with Python's stdlib
# zlib.crc32() against that field directly -- the same approach the old app took, just reading
# the value from the new typed CaptureMetadataT payload instead of an untyped per-pipeline
# metadata dict.
#
#   LinuxReceiverOp -> FrameValidationOp -> CsiToBayerOp -> ImageProcessorOp ->
#     BayerDemosaicOp -> HolovizOp
#
# The old app split work across two operators connected by HSDK4's implicit per-pipeline
# metadata dict: InstrumentedTimeProfiler (pre-CsiToBayer, computed the CRC and stashed
# frame/CRC/size bookkeeping into that dict) and MonitorOperator (post-Holoviz, via
# camera_pose_output, read the same dict back out to do frame-to-frame consistency checks, then
# a final performance report once main() returns from application.run()). HSDK5 has no
# equivalent implicit metadata carry-through across the whole pipeline (the same limitation
# every latency-app port in this effort already worked around) -- so this port consolidates
# everything into one native Python pass-through operator, FrameValidationOp, sitting right
# after the receiver where hololink.CaptureMetadataT is still directly available, instead of
# splitting work across the start and end of the pipeline. This is an architectural adaptation,
# not a feature cut: all four validation checks (CRC, frame size, frame number ordering,
# inter-frame timing) and three of the old final report's five latency metrics are preserved,
# just consolidated into one operator instead of two. Two real, flagged (not silent) scope
# changes from that move:
# - The old app's inter-frame timing check compared Holoviz-complete timestamps (after
#   display); this port's version runs at FrameValidationOp instead (right after capture,
#   before display), since there's no post-Holoviz hook to read CaptureMetadataT from anymore.
#   HolovizOp's enable_camera_pose_output/camera_pose_output_type options (which only existed to
#   drive that now-removed hook) have no equivalent in the new HolovizOp binding at all either.
# - The old final report's "processing time" and "overall time" metrics both needed that same
#   post-Holoviz "complete" timestamp -- dropped for the same reason, not approximated. The
#   remaining three metrics (sensor frame time, DMA+CPU wakeup, operator dispatch) don't depend
#   on it and are preserved via the same accumulate-then-report-at-the-end pattern
#   linux_imx274_latency.py's LatencyReportSinkOp already established, reusing FrameValidationOp's
#   own compute-entry timestamp as the "operator dispatch" point (the same role
#   InstrumentedTimeProfiler's self-recorded "operator_timestamp" played in the old app -- both
#   are the first pipeline stage downstream of the receiver). FrameValidationOp has no natural
#   "done" trigger of its own (unlike LatencyReportSinkOp, which self-terminates at a fixed
#   frame_limit) since this app's --frame-limit defaults to unbounded (Ctrl+C to stop, matching
#   the old app's own default) -- so the report is read back and printed from run() after
#   session.wait() returns, via a plain Python-side instance stash (_ValidationReportHolder,
#   same shape as this effort's other pure-Python stashes) rather than a self-triggered print.
#
# --frame-limit is the receiver's frame_limit parameter: the receiver ends the run after that
# many frames. HSDK5 has no CountCondition; this replaces the CountCondition of the old app.
#
# Usage:
#   linux_imx274_frame_validation.py [OPTIONS]
#
# Options:
#   --hololink IP                 IP address of Hololink board (default: 192.168.0.2)
#   --camera-mode N                IMX274 mode (default: 0 = 3840x2160@60fps)
#   --expander-configuration N    I2C expander configuration (0 or 1, default: 0)
#   --pattern N                   Configure a test pattern (0-11) instead of live video
#   --crc-frame-check N           If > 0, check the CRC of every Nth frame (default: 0 = off)
#   --skip-reset                  Don't call reset on the hololink device
#   --frame-limit N               Number of frames to capture (default: unlimited)
#   --headless                    Run without opening a window (still renders, presents nowhere)
#   --fullscreen                  Run in fullscreen mode
#   --log-level N                 Python logging level (default: 20 = INFO)

import argparse
import logging
import sys
import time
import zlib
from datetime import timedelta

import capture_metadata
import cupy as cp
import holoscan as hs
import holoscan_vision as hv
import holoscan_visualization as hz  # noqa: F401 -- registers "holoscan.visualization.HolovizOp"
from cuda.bindings import driver as cuda_driver

import hololink as hl

_BAYER_FORMAT_NAMES = ["bggr", "rggb", "gbrg", "grbg"]
_BAYER_FORMAT_VALUES = {name: index for index, name in enumerate(_BAYER_FORMAT_NAMES)}

# Matches hololink::operators::kMetadataPairingWindow (capture_metadata_utils.hpp): the
# receiver stamps both ports with an identical capture_time, so this zero-width
# OnAligned+Exact{} pairs them precisely.
_METADATA_PAIRING_WINDOW = timedelta(0)

NS_PER_S = 1_000_000_000


def _cuda_check(result):
    err = result[0]
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        _, description = cuda_driver.cuGetErrorString(err)
        raise RuntimeError(f"CUDA error: {description}")
    if len(result) == 2:
        return result[1]
    return result[1:]


class _ValidationReportHolder:
    """Plain Python-side stash so run() can read FrameValidationOp's accumulated state back
    after session.wait() returns -- there's no frame-limit-triggered self-report here (unlike
    LatencyReportSinkOp) since --frame-limit defaults to unbounded, matching the old app.
    """

    instance = None


def _ns_to_ms(ns):
    return ns / 1.0e6


def _print_stat(label, values):
    mn = _ns_to_ms(min(values))
    mx = _ns_to_ms(max(values))
    avg = _ns_to_ms(sum(values)) / len(values)
    print(f"{label:<32}{mn:>10.3f} ms{mx:>10.3f} ms{avg:>10.3f} ms")


class FrameValidationOp(hs.Operator):
    """Pass-through: validates each received frame, then forwards tensor+metadata unchanged.

    Checks (matching the old app's validate_frame() exactly):
    - CRC32 (every crc_frame_check frames, if > 0): CPU zlib.crc32() over the raw received
      tensor bytes vs. the FPGA-stamped CaptureMetadataT.crc field. The FPGA's CRC is
      CRC-32/JAMCRC (bit-inverted, no final XOR); zlib.crc32() is standard CRC-32 (with the
      final XOR) -- the `^ 0xFFFFFFFF` below converts JAMCRC to the standard form for
      comparison, matching the old app exactly.
    - received-vs-calculated frame size.
    - frame_number ordering (consecutive, no gaps).
    - inter-frame timing (flagged if more than 2x the expected frame period).

    Also accumulates a final latency report (sensor frame time / DMA+CPU wakeup / operator
    dispatch -- see the file-level comment for why only these three of the old app's five
    metrics survive), printed via print_report() once run() calls it after the session ends.
    """

    in_tensor = hs.Input("in_tensor", hs.payload.Tensor)
    in_metadata = hs.Input("in_metadata", capture_metadata.CaptureMetadata)
    tensor = hs.Output("tensor", hs.payload.Tensor)
    metadata = hs.Output("metadata", capture_metadata.CaptureMetadata)

    calculated_frame_size = hs.Param(0)
    crc_frame_check = hs.Param(0)
    # Arbitrary number of milliseconds that aligns to IMX274 default mode (4K@60FPS ==
    # 16.6ms/frame); allow 2x frame time before flagging a timing mismatch -- matches the old
    # app's own hardcoded constant exactly.
    expected_frame_period_ms = hs.Param(16.6)
    # 0 = unbounded (matches this app's own --frame-limit default); only used to decide when
    # the "skip the last warmup_frames" trim window starts -- an unbounded run never trims from
    # the end, since there's no known end to trim against.
    frame_limit = hs.Param(0)
    warmup_frames = hs.Param(5)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._crc_errors = 0
        self._crc_checked = 0
        self._prev_frame_number = None
        self._prev_entry_ns = None
        self._frame_count = 0
        self._latency_samples = []
        _ValidationReportHolder.instance = self

    def setup(self, spec):
        spec.input(self.in_tensor, queue_depth=4)
        spec.input(self.in_metadata, queue_depth=4)
        spec.output(self.tensor)
        spec.output(self.metadata)

    def contract(self):
        return (
            hs.Contract()
            .trigger(
                hs.OnAligned(
                    ports=[self.in_tensor, self.in_metadata],
                    window=_METADATA_PAIRING_WINDOW,
                )
            )
            .input(self.in_tensor, hs.Exact())
            .input(self.in_metadata, hs.Exact())
        )

    def compute(self, context):
        entry_ns = time.time_ns()

        tensor_sample = self.in_tensor.receive()
        metadata_sample = self.in_metadata.receive()
        if tensor_sample.is_missing or metadata_sample.is_missing:
            return

        frame = metadata_sample.data
        frame_number = frame.frame_number
        if tensor_sample.metadata.frame_id != frame_number:
            logging.warning("Dropping mismatched tensor/metadata frame identities")
            return
        self._frame_count += 1

        # check if crc frame check is enabled, if so, check the CRC of every crc_frame_check
        # frames
        if self.crc_frame_check > 0 and frame_number % self.crc_frame_check == 0:
            # Finish the CUDA read before forwarding the received sample.
            with cp.cuda.Stream(non_blocking=True) as stream:
                capsule = tensor_sample.data.__dlpack__(
                    stream=stream.ptr, max_version=(1, 0), copy=False
                )
                cp_frame = cp.from_dlpack(capsule)
                np_frame = cp_frame.get(stream=stream)
                stream.synchronize()
            crc32_calculated = zlib.crc32(np_frame.tobytes())
            crc32_expected = frame.crc ^ 0xFFFFFFFF
            self._crc_checked += 1
            if crc32_calculated != crc32_expected:
                logging.debug(
                    f"frame number {frame_number}: CRC32 mismatch: "
                    f"calculated={crc32_calculated}, expected={crc32_expected}"
                )
                self._crc_errors += 1

        # The received tensor knows its size; only CRC-checked frames are copied.
        received_frame_size = tensor_sample.data.nbytes
        if received_frame_size != self.calculated_frame_size:
            logging.info(
                f"Frame size is not the same as the calculated frame size, "
                f"received frame size={received_frame_size}, "
                f"calculated frame size={self.calculated_frame_size}"
            )

        if self._prev_frame_number is not None:
            if frame_number != self._prev_frame_number + 1:
                logging.info(
                    f"Frame number is not in order, last frame number="
                    f"{self._prev_frame_number}, current frame number={frame_number}"
                )
            time_difference_ms = (entry_ns - self._prev_entry_ns) / 1.0e6
            if time_difference_ms > (2 * self.expected_frame_period_ms):
                logging.info(
                    f"Frame timestamp mismatch, Diff[ms]={time_difference_ms:.3f}"
                )
        self._prev_frame_number = frame_number
        self._prev_entry_ns = entry_ns

        # Collect settled latency samples (skip warmup at both ends, same spirit as
        # LatencyReportSinkOp -- only trim from the end if frame_limit is actually set).
        in_warmup = self._frame_count <= self.warmup_frames or (
            self.frame_limit
            and self._frame_count > self.frame_limit - self.warmup_frames
        )
        if not in_warmup:
            frame_start_ns = frame.timestamp_s * NS_PER_S + frame.timestamp_ns
            frame_end_ns = frame.metadata_s * NS_PER_S + frame.metadata_ns
            received_ns = frame.received_s * NS_PER_S + frame.received_ns
            self._latency_samples.append(
                (
                    frame_end_ns - frame_start_ns,  # sensor frame time
                    received_ns - frame_end_ns,  # DMA + CPU wakeup
                    entry_ns - received_ns,  # operator dispatch
                )
            )

        self.tensor.forward(tensor_sample)
        self.metadata.forward(metadata_sample)

    def print_report(self):
        if self._crc_frame_check_enabled():
            total = self._crc_checked
            errors = self._crc_errors
            if total > 0:
                success_rate = ((total - errors) / total) * 100
                print(
                    f"CRC Validation: {total} frames checked, {errors} errors, "
                    f"{success_rate:.2f}% success"
                )
            else:
                print("CRC Validation: enabled, but no frames were checked.")
        else:
            print("CRC validation was disabled (--crc-frame-check=0).")

        if not self._latency_samples:
            print("No settled frames collected -- nothing to report.")
            return

        labels = ["Sensor frame time", "DMA + CPU wakeup", "Operator dispatch"]
        print(f"\n=== Latency Report ({len(self._latency_samples)} frames) ===")
        print(f"{'Metric':<32}{'Min':>12}{'Max':>12}{'Avg':>12}")
        print("-" * 68)
        for index, label in enumerate(labels):
            _print_stat(label, [sample[index] for sample in self._latency_samples])
        print("-" * 68 + "\n")

    def _crc_frame_check_enabled(self):
        return self.crc_frame_check > 0


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
    pattern,
    crc_frame_check,
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
    logging.debug("Waiting for PTP sync.")
    if not hololink_dev.ptp_synchronize():
        raise ValueError("Failed to synchronize PTP.")
    logging.debug("PTP synchronized.")

    camera = hl.NativeImx274Sensor(
        channel, expander_configuration=expander_configuration
    )
    if not skip_reset:
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
    graph = hs.Graph("linux_imx274_frame_validation")

    rx = graph.op(
        hl.operators.LinuxReceiverOp,
        "rx",
        data_channel=channel,
        frame_context=int(cu_context),
        device_start=camera.start,
        device_stop=camera.stop,
        frame_size=frame_size,
        frame_limit=frame_limit or 0,
        frame_limit_ends_run=True,
    )

    validation = graph.op(
        FrameValidationOp,
        "frame_validation",
        calculated_frame_size=frame_size,
        crc_frame_check=crc_frame_check,
        frame_limit=frame_limit or 0,
    )
    graph.add_flow(rx.tensor, validation.in_tensor, queue_depth=4)
    graph.add_flow(rx.metadata, validation.in_metadata, queue_depth=4)

    csi = graph.op(
        hl.operators.CsiToBayerOp,
        "csi_to_bayer",
        cuda_device_ordinal=cuda_device_ordinal,
        geometry=geometry,
        bayer_format=bayer_format,
    )
    graph.add_flow(validation.tensor, csi.in_tensor, queue_depth=4)
    graph.add_flow(validation.metadata, csi.in_metadata, queue_depth=4)

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

    # image_processor.metadata is not connected: nothing after ImageProcessorOp uses the
    # metadata, and an output that is not connected has no queue that can fill.

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
            "title": "Linux IMX274 Frame Validation",
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
        failed = session.termination == hs.RunTermination.FAILED
        if failed:
            print(session.summary(), file=sys.stderr)

    hololink_dev.stop()
    _cuda_check(cuda_driver.cuDevicePrimaryCtxRelease(cu_device))

    if _ValidationReportHolder.instance is not None:
        _ValidationReportHolder.instance.print_report()

    if failed:
        return 1
    print("Done.")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera-mode",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="IMX274 mode (default: 0 = 3840x2160@60fps)",
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
    parser.add_argument(
        "--pattern",
        type=int,
        choices=range(12),
        help="Configure to display a test pattern.",
    )
    parser.add_argument(
        "--skip-reset",
        action="store_true",
        help="Don't call reset on the hololink device.",
    )
    parser.add_argument(
        "--crc-frame-check",
        type=int,
        default=0,
        help="if > 0, check the CRC of n'th frame. this might cause a performance hit",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    return run(
        args.hololink,
        args.camera_mode,
        args.expander_configuration,
        args.pattern,
        args.crc_frame_check,
        args.skip_reset,
        args.frame_limit,
        args.headless,
        args.fullscreen,
    )


if __name__ == "__main__":
    sys.exit(main())
