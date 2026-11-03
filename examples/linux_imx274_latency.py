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

# Measures and reports end-to-end pipeline latency for a single IMX274 sensor via
# LinuxReceiverOp. Python port of this repo's own previous (HSDK4-era) Python version of this
# app, via the HSDK5 Python bindings; the pipeline shape and metrics below also match
# examples/linux_imx274_latency.cpp.
#
#   LinuxReceiverOp -> CsiToBayerOp -> ImageProcessorOp -> LatencyReportSinkOp
#                                                  \-> BayerDemosaicOp -> HolovizOp
#
# The old Python app's InstrumentedTimeProfiler/MonitorOperator pair measured latency by
# stamping wall-clock times into each frame's Python-side metadata dict as it flowed through a
# full visualization pipeline (ending at HolovizOp's camera_pose_output). HSDK5 has no
# equivalent implicit per-pipeline metadata-dict propagation mechanism -- this port instead
# reads the same FPGA/receiver timing fields the old app used, but off of the typed
# hololink.CaptureMetadataT payload each receiver/operator already stamps (see
# src/hololink/operators/capture_metadata.fbs), in a single native Python sink operator at the
# end of the pipeline. The sink is the measurement point. An optional display branch
# (BayerDemosaicOp -> HolovizOp) takes the same ImageProcessorOp output with DROP_OLDEST
# queues, so a slow display cannot slow down the sink. Use --headless to leave it out.
#
# Usage:
#   linux_imx274_latency.py [OPTIONS]
#
# Options:
#   --hololink IP                 IP address of Hololink board (default: 192.168.0.2)
#   --camera-mode N                IMX274 mode (default: 0 = 3840x2160@60fps)
#   --expander-configuration N    I2C expander configuration (0 or 1, default: 0)
#   --frame-limit N               Number of frames to collect before reporting (default: 300)
#   --log-interval N              Progress log every N frames (default: 30)
#   --log-level N                 Python logging level (default: 20 = INFO)
#   --headless                    Do not open a HoloViz window (latency measurement only)
#   --fullscreen                  Run the HoloViz window in fullscreen mode

import argparse
import logging
import sys
import time
from datetime import timedelta

import capture_metadata
import holoscan as hs
import holoscan_vision as hv
import holoscan_visualization as hz  # noqa: F401 -- registers "holoscan.visualization.HolovizOp"
from cuda.bindings import driver as cuda_driver

import hololink as hl

_BAYER_FORMAT_NAMES = ["bggr", "rggb", "gbrg", "grbg"]
_BAYER_FORMAT_VALUES = {name: index for index, name in enumerate(_BAYER_FORMAT_NAMES)}

# Matches hololink::operators::kMetadataPairingWindow (capture_metadata_utils.hpp): the
# receiver/CsiToBayerOp/ImageProcessorOp chain stamps every port pair with an identical
# capture_time, so this bounded-window OnAligned+Exact{} pairs them precisely.
_METADATA_PAIRING_WINDOW = timedelta(microseconds=500)

NS_PER_S = 1_000_000_000


class LatencyReportSinkOp(hs.Operator):
    """Terminal sink: reads hololink.CaptureMetadataT fields, prints a latency report."""

    tensor = hs.Input("tensor", hs.schema.Image)
    metadata = hs.Input("metadata", capture_metadata.CaptureMetadata)

    frame_limit = hs.Param(300)
    warmup_frames = hs.Param(5)
    log_interval = hs.Param(30)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._frame_count = 0
        self._samples = []

    def setup(self, spec):
        spec.input(self.tensor, queue_depth=4)
        spec.input(self.metadata, queue_depth=4)

    def contract(self):
        return (
            hs.Contract()
            .trigger(
                hs.OnAligned(
                    ports=[self.tensor, self.metadata],
                    window=_METADATA_PAIRING_WINDOW,
                )
            )
            .input(self.tensor, hs.Exact())
            .input(self.metadata, hs.Exact())
        )

    def compute(self, context):
        # Capture host wall-clock at the very start of this compute call -- this is our
        # "pipeline complete" timestamp.
        complete_ns = time.time_ns()

        self.tensor.receive_data()
        frame = self.metadata.receive_data()

        self._frame_count += 1

        if self._frame_count % self.log_interval == 0:
            logging.info(
                f"[latency] frames={self._frame_count} frame_number={frame.frame_number}"
            )

        # Collect settled frames (skip warmup at both ends).
        in_warmup = self._frame_count <= self.warmup_frames or (
            self._frame_count > self.frame_limit - self.warmup_frames
        )
        if not in_warmup:
            frame_start_ns = frame.timestamp_s * NS_PER_S + frame.timestamp_ns
            frame_end_ns = frame.metadata_s * NS_PER_S + frame.metadata_ns
            received_ns = frame.received_s * NS_PER_S + frame.received_ns
            entry_ns = frame.operator_entry_ns

            self._samples.append(
                (
                    frame_end_ns - frame_start_ns,  # sensor frame time
                    received_ns - frame_end_ns,  # DMA + CPU wakeup
                    (
                        (entry_ns - received_ns) if entry_ns > 0 else 0
                    ),  # operator dispatch
                    complete_ns - received_ns,  # SW pipeline latency
                    complete_ns - frame_start_ns,  # end-to-end latency
                )
            )

        if self._frame_count >= self.frame_limit:
            _print_report(self._samples)
            context.request_stop()


def _ns_to_ms(ns):
    return ns / 1.0e6


def _print_stat(label, values):
    mn = _ns_to_ms(min(values))
    mx = _ns_to_ms(max(values))
    avg = _ns_to_ms(sum(values)) / len(values)
    print(f"{label:<32}{mn:>10.3f} ms{mx:>10.3f} ms{avg:>10.3f} ms")


def _print_report(samples):
    if not samples:
        print("No settled frames collected -- nothing to report.")
        return

    labels = [
        "Sensor frame time",
        "DMA + CPU wakeup",
        "Operator dispatch",
        "SW pipeline latency",
        "End-to-end latency",
    ]

    print(f"\n=== Latency Report ({len(samples)} frames) ===")
    print(f"{'Metric':<32}{'Min':>12}{'Max':>12}{'Avg':>12}")
    print("-" * 68)
    for index, label in enumerate(labels):
        _print_stat(label, [sample[index] for sample in samples])
    print("-" * 68 + "\n")


def _cuda_check(result):
    err = result[0]
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        _, description = cuda_driver.cuGetErrorString(err)
        raise RuntimeError(f"CUDA error: {description}")
    if len(result) == 2:
        return result[1]
    return result[1:]


def run(
    hololink_ip,
    camera_mode,
    expander_configuration,
    frame_limit,
    log_interval,
    headless=False,
    fullscreen=False,
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
    hololink_dev.reset()

    print("Waiting for PTP synchronization...")
    if not hololink_dev.ptp_synchronize():
        print(
            "PTP synchronization failed -- cannot measure latency without a synced clock.",
            file=sys.stderr,
        )
        _cuda_check(cuda_driver.cuDevicePrimaryCtxRelease(cu_device))
        return 1
    print("PTP synchronized.")

    camera = hl.NativeImx274Sensor(
        channel, expander_configuration=expander_configuration
    )
    camera.setup_clock()
    camera.configure(camera_mode)
    camera.set_digital_gain_reg(0x4)

    pixel_format = camera.get_pixel_format()
    bayer_format = camera.get_bayer_format()

    geometry = hl.resolve_generic_geometry(camera)
    frame_size = (
        geometry["start_byte"]
        + geometry["bytes_per_line"] * geometry["pixel_height"]
        + geometry["trailing_bytes"]
    )

    # Build the graph.
    graph = hs.Graph("linux_imx274_hsdk5_latency")

    rx = graph.op(
        hl.operators.LinuxReceiverOp,
        "rx",
        data_channel=channel,
        frame_context=int(cu_context),
        device_start=camera.start,
        device_stop=camera.stop,
        frame_size=frame_size,
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
        # Optical black value for imx274 is 50 (matches this app's own HSDK4-era default).
        optical_black=50,
        cuda_device_ordinal=cuda_device_ordinal,
    )
    graph.add_flow(csi.tensor, image_processor.in_tensor, queue_depth=4)
    graph.add_flow(csi.metadata, image_processor.in_metadata, queue_depth=4)

    sink = graph.op(
        LatencyReportSinkOp,
        "sink",
        frame_limit=frame_limit,
        warmup_frames=5,
        log_interval=log_interval,
    )
    graph.add_flow(image_processor.tensor, sink.tensor, queue_depth=4)
    graph.add_flow(image_processor.metadata, sink.metadata, queue_depth=4)

    # Optional display branch. The latency sink above stays the measurement point; this
    # branch drops old frames so a slow display cannot back-pressure the sink.
    if not headless:
        frame_width = camera.get_width()
        frame_height = camera.get_height()
        demosaic = graph.op(
            hv.operators.BayerDemosaicOp,
            "bayer_demosaic",
            bayer_grid_pos=bayer_format,
            generate_alpha=True,
            alpha_value=65535,
        )
        graph.add_flow(
            image_processor.tensor,
            demosaic.input,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
            queue_depth=1,
        )
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
            sink={
                "kind": "window",
                "title": f"Linux IMX274 Latency - {hololink_ip}",
                "width": frame_width,
                "height": frame_height,
                "fullscreen": fullscreen,
                "close_on_escape": True,
                "framebuffer_srgb": True,
            },
            # One frame per received frame.
            trigger={"kind": "layer"},
        )
        graph.add_flow(
            demosaic.output,
            viz.video,
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

    if not headless:
        options.deployment.bind_tensor_output_device(
            operator_path="bayer_demosaic",
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

    print(f"Collecting {frame_limit} frames (Ctrl+C to abort)...")

    with hs.run_async(plan) as session:
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
        default=0,
        help="IMX274 mode (default: 0 = 3840x2160@60fps)",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=300,
        help="Number of frames to collect before reporting (default: 300)",
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
        "--log-interval",
        type=int,
        default=30,
        help="Progress log every N frames (default: 30)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Do not open a HoloViz window (latency measurement only)",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Run the HoloViz window in fullscreen mode",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    return run(
        args.hololink,
        args.camera_mode,
        args.expander_configuration,
        args.frame_limit,
        args.log_interval,
        headless=args.headless,
        fullscreen=args.fullscreen,
    )


if __name__ == "__main__":
    sys.exit(main())
