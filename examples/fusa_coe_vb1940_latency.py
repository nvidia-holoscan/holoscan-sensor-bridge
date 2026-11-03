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

# Measures and reports end-to-end pipeline latency for a single VB1940 sensor via
# FusaCoeCaptureOp. Python port of this repo's own previous (HSDK4-era) Python version of this
# app, via the HSDK5 Python bindings; the capture/bring-up pattern below also matches the
# already-ported fusa_coe_vb1940_player.py/examples/fusa_coe_vb1940_player.cpp, and the
# latency-reporting sink matches linux_imx274_latency.py/imx274_latency.py.
#
#   FusaCoeCaptureOp -> CsiToBayerOp -> ImageProcessorOp -> LatencyReportSinkOp
#
# Always captures unpacketized CSI-2 (packetizer_enabled=False), like fusa_coe_vb1940_player.py
# (fusa_coe_vb1940_player.cpp packetizes RAW_10/RAW_12 modes by default) --
# PackedFormatConverterOp's graph.op<>() construction path is exercised by the IMX274 FUSA CoE
# apps instead, not this one.
#
# The old Python app's MonitorOperator measured latency by reading `self.metadata` (a
# per-pipeline Python dict HolovizOp's camera_pose_output triggered on) inside a full
# visualization pipeline. HSDK5 has no equivalent implicit per-pipeline metadata-dict
# propagation mechanism -- this port instead reads the same FPGA/receiver timing fields the old
# app used, but off of the typed hololink.CaptureMetadataT payload each
# receiver/converter/operator already stamps (see src/hololink/operators/capture_metadata.fbs),
# in a single native Python sink operator at the end of a minimal (no Holoviz) pipeline --
# adding a visualizer would add its own latency and distort the very thing being measured (same
# reasoning as linux_imx274_latency.py).
#
# Usage:
#   fusa_coe_vb1940_latency.py [OPTIONS]
#
# Options:
#   --hololink IP        IP address of Hololink board (default: 192.168.0.2)
#   --sensor N            Sensor index: 0 or 1 (default: 0)
#   --camera-mode N       VB1940 mode (default: 0 = 2560x1984@30fps)
#   --timeout N           Per-frame capture timeout in milliseconds (default: 1500)
#   --frame-limit N       Number of frames to collect before reporting (default: 300)
#   --log-interval N      Progress log every N frames (default: 30)
#   --log-level N         Python logging level (default: 20 = INFO)
#   --interface NAME      NIC the board streams to and the CoE capture listens on (default: the
#                         one the board enumerated on)
#   --headless, --width N, --height N
#                         Accepted for CLI compatibility; unused (this pipeline has no display)

import argparse
import logging
import sys
import time
from datetime import timedelta

import capture_metadata
import holoscan as hs
from cuda.bindings import driver as cuda_driver

import hololink as hl

_BAYER_FORMAT_NAMES = ["bggr", "rggb", "gbrg", "grbg"]
_BAYER_FORMAT_VALUES = {name: index for index, name in enumerate(_BAYER_FORMAT_NAMES)}

# Matches hololink::operators::kMetadataPairingWindow (capture_metadata_utils.hpp): the
# capture/CsiToBayerOp/ImageProcessorOp chain stamps every port pair with an identical
# capture_time, so this zero-width OnAligned+Exact{} pairs them precisely.
_METADATA_PAIRING_WINDOW = timedelta(0)

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

        tensor_sample = self.tensor.receive()
        metadata_sample = self.metadata.receive()
        if tensor_sample.is_missing or metadata_sample.is_missing:
            return
        frame = metadata_sample.data
        if tensor_sample.metadata.frame_id != frame.frame_number:
            logging.warning("Dropping mismatched tensor/metadata frame identities")
            return

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
    clock_alignment_valid = all(sample[1] >= 0 and sample[4] >= 0 for sample in samples)
    for index, label in enumerate(labels):
        if index in (1, 4) and not clock_alignment_valid:
            print(f"{label:<32} unavailable (negative cross-clock timing)")
        else:
            _print_stat(label, [sample[index] for sample in samples])
    print("-" * 68 + "\n")
    if not clock_alignment_valid:
        logging.warning(
            "Host/board clock alignment needs checking; cross-clock latencies are unavailable."
        )


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
    sensor_id,
    camera_mode,
    timeout_ms,
    frame_limit,
    log_interval,
    interface=None,
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
        if interface:
            # DataChannel.configure_coe() points the board at this NIC too, so the board streams
            # to the one the capture listens on.
            metadata["interface"] = interface
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

        # PTP sync is required for cross-domain (FPGA vs. host) latency measurement -- matches the
        # old Python app's own unconditional check, kept here as a hard failure rather than a
        # best-effort warning (same reasoning as linux_imx274_latency.py/imx274_latency.py).
        print("Waiting for PTP synchronization...")
        if not hololink_dev.ptp_synchronize():
            print(
                "PTP synchronization failed -- cannot measure latency without a synced clock.",
                file=sys.stderr,
            )
            return 1
        print("PTP synchronized.")

        camera = hl.NativeVb1940Sensor(channel)
        camera.setup_clock()
        camera.configure(camera_mode)

        pixel_format = camera.get_pixel_format()
        bayer_format = camera.get_bayer_format()

        # Always capture unpacketized CSI-2, like fusa_coe_vb1940_player.py;
        # PackedFormatConverterOp's graph.op<>() construction path is exercised by the IMX274 FUSA
        # CoE apps instead.
        packetizer_enabled = False

        # Resolve geometry ONCE using FusaCoeCaptureOp's HSB-alignment-aware adapter, and reuse that
        # same geometry dict for both FusaCoeCaptureOp and CsiToBayerOp below (see
        # fusa_coe_vb1940_player.py's identical comment for why resolving twice would misparse).
        geometry = hl.resolve_geometry(camera, packetizer_enabled)

        # Build the graph.
        graph = hs.Graph("fusa_coe_vb1940_hsdk5_latency")

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
        graph.add_flow(capture.tensor, csi.in_tensor, queue_depth=4)
        graph.add_flow(capture.metadata, csi.in_metadata, queue_depth=4)

        image_processor = graph.op(
            hl.operators.ImageProcessorOp,
            "image_processor",
            pixel_format=pixel_format,
            pixel_width=geometry["pixel_width"],
            pixel_height=geometry["pixel_height"],
            bayer_format=_BAYER_FORMAT_VALUES[bayer_format],
            optical_black=0,
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

        clock_key = graph.add_clock(hs.RealtimeClock, "clock")
        graph.set_default_clock(clock_key)

        graph.partition("capture").add(capture)
        graph.auto_partition_remaining()

        options = hs.CompileOptions()
        for operator_path in ("capture", "csi_to_bayer", "image_processor"):
            options.deployment.bind_tensor_output_device(
                operator_path=operator_path,
                output_port="tensor",
                device=hs.execution.DeviceId(cuda_device_ordinal),
            )

        plan = hs.compile(graph, options)
        if not plan.ok:
            print("Graph compilation failed:", file=sys.stderr)
            for diag in plan.diagnostics:
                print(f"  {diag}", file=sys.stderr)
            return 1

        print(
            f"Collecting {frame_limit} frames from sensor {sensor_id} (Ctrl+C to abort)..."
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
    parser.add_argument(
        "--camera-mode", type=int, choices=[0, 1, 2, 3], default=0, help="VB1940 mode"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1500,
        help="Per-frame capture timeout in milliseconds",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=300,
        help="Number of frames to collect before reporting (default: 300)",
    )
    parser.add_argument(
        "--log-level", type=int, default=20, help="Python logging level to display"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=30,
        help="Progress log every N frames (default: 30)",
    )
    parser.add_argument(
        "--interface",
        default=None,
        help="NIC the board streams to and the CoE capture listens on (default: the one the board enumerated on)",
    )
    # Accepted for CLI compatibility with the HSDK4 app.
    unused = "Accepted for CLI compatibility; unused (this pipeline has no display)"
    parser.add_argument("--headless", action="store_true", help=unused)
    parser.add_argument("--width", type=int, help=unused)
    parser.add_argument("--height", type=int, help=unused)
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    return run(
        args.hololink,
        args.sensor,
        args.camera_mode,
        args.timeout,
        args.frame_limit,
        args.log_interval,
        args.interface,
    )


if __name__ == "__main__":
    sys.exit(main())
