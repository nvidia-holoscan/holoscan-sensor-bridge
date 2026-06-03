#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# VB1940 Latency Measurement using FuSa CoE Capture
#
# PIPELINE:
#   FusaCoeCaptureOp -> PackedFormatConverterOp -> ImageProcessorOp ->
#   BayerDemosaicOp -> HolovizOp -> MonitorOp
#
# MEASUREMENTS:
# - Frame Time: Sensor readout time (FPGA timestamps)
# - Frame Interval: Frame-to-frame time (host timestamps)
# - Pipeline Latency (FSR->Monitor): End-to-end SW pipeline latency

import argparse
import ctypes
import datetime
import logging
import math

import holoscan

import hololink as hololink_module

NS_PER_SEC = 1e9
SEC_PER_NS = 1.0 / NS_PER_SEC


def get_timestamp(metadata, name):
    """Extract timestamp from metadata (seconds + nanoseconds)."""
    s = metadata[f"{name}_s"]
    ns = metadata[f"{name}_ns"]
    return s + (ns * SEC_PER_NS)


def save_timestamp(metadata, name, timestamp):
    """Save Python datetime to metadata as seconds + nanoseconds."""
    f, s = math.modf(timestamp.timestamp())
    metadata[f"{name}_s"] = int(s)
    metadata[f"{name}_ns"] = int(f * NS_PER_SEC)


class MonitorOperator(holoscan.core.Operator):
    """Operator to record final timestamp and collect statistics."""

    def __init__(self, *args, recorder_queue=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._recorder_queue = recorder_queue
        self._frame_count = 0

    def setup(self, spec):
        spec.input("input")

    def compute(self, op_input, op_output, context):
        # Record timestamp when visualization completes
        complete_timestamp = datetime.datetime.now(datetime.timezone.utc)

        # Receive input (camera pose from HolovizOp)
        _ = op_input.receive("input")
        self._frame_count += 1

        # Extract HSB metadata (FPGA timestamps)
        if self._recorder_queue is None:
            return

        try:
            frame_start_s = get_timestamp(self.metadata, "timestamp")
            frame_end_s = get_timestamp(self.metadata, "metadata")
            frame_number = self.metadata["frame_number"]
            self._recorder_queue.append(
                (
                    frame_start_s,
                    frame_end_s,
                    complete_timestamp.timestamp(),
                    self._frame_count,
                    frame_number,
                )
            )
        except KeyError as e:
            logging.warning(f"Frame {self._frame_count}: Missing metadata: {e}")


class App(holoscan.core.Application):
    """Holoscan application for VB1940 latency measurement."""

    def __init__(
        self,
        hololink_channel,
        camera,
        camera_mode,
        timeout,
        frame_limit,
        interface,
        recorder_queue,
        headless,
        width,
        height,
    ):
        super().__init__()
        self._hololink_channel = hololink_channel
        self._camera = camera
        self._camera_mode = camera_mode
        self._timeout = timeout
        self._frame_limit = frame_limit
        self._interface = interface
        self._recorder_queue = recorder_queue
        self._headless = headless
        self._width = width
        self._height = height
        self.is_metadata_enabled = True

    def compose(self):
        metadata = self._hololink_channel.enumeration_metadata()
        interface = self._interface or metadata.get("interface")
        hsb_mac = metadata.get("mac_id")
        hsb_mac_bytes = list(bytes.fromhex(hsb_mac.replace(":", "")))

        logging.info(f"Using interface: {interface}")
        logging.info(f"HSB MAC: {hsb_mac}")

        if self._frame_limit:
            condition = holoscan.conditions.CountCondition(
                self, name="count", count=self._frame_limit
            )
        else:
            condition = holoscan.conditions.BooleanCondition(
                self, name="ok", enable_tick=True
            )

        self._camera.set_mode(self._camera_mode)
        pixel_format = self._camera.pixel_format()
        bayer_format = self._camera.bayer_format()

        # Capture
        fusa_coe_capture = hololink_module.operators.FusaCoeCaptureOp(
            self,
            condition,
            name="fusa_coe_capture",
            interface=interface,
            mac_addr=hsb_mac_bytes,
            hololink_channel=self._hololink_channel,
            timeout=self._timeout,
            device=self._camera,
        )
        self._camera.configure_converter(fusa_coe_capture)

        # Format conversion
        packed_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="packed_pool",
            storage_type=1,
            block_size=self._camera._width
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera._height,
            num_blocks=4,
        )
        packed_converter = hololink_module.operators.PackedFormatConverterOp(
            self, name="packed_converter", allocator=packed_pool
        )
        fusa_coe_capture.configure_converter(packed_converter)

        # Image processing
        image_processor = hololink_module.operators.ImageProcessorOp(
            self,
            name="image_processor",
            optical_black=0,
            bayer_format=bayer_format.value,
            pixel_format=pixel_format.value,
        )

        # Demosaic
        demosaic_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="demosaic_pool",
            storage_type=1,
            block_size=self._camera._width
            * 4
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera._height,
            num_blocks=4,
        )
        demosaic = holoscan.operators.BayerDemosaicOp(
            self,
            name="demosaic",
            pool=demosaic_pool,
            generate_alpha=True,
            alpha_value=65535,
            bayer_grid_pos=bayer_format.value,
            interpolation_mode=0,
        )

        # Visualization
        visualizer = holoscan.operators.HolovizOp(
            self,
            name="holoviz",
            headless=self._headless,
            framebuffer_srgb=True,
            width=self._width,
            height=self._height,
            enable_camera_pose_output=True,
            camera_pose_output_type="extrinsics_model",
        )

        # Monitor
        monitor = MonitorOperator(
            self, name="monitor", recorder_queue=self._recorder_queue
        )

        # Connect pipeline
        self.add_flow(fusa_coe_capture, packed_converter, {("output", "input")})
        self.add_flow(packed_converter, image_processor, {("output", "input")})
        self.add_flow(image_processor, demosaic, {("output", "receiver")})
        self.add_flow(demosaic, visualizer, {("transmitter", "receivers")})
        self.add_flow(visualizer, monitor, {("camera_pose_output", "input")})


def analyze_latency(recorder_queue):
    """Analyze and report latency statistics."""
    # Arbitrary minimum of 20 frames since we drop the first and last 5 frames
    if len(recorder_queue) < 20:
        logging.error(f"Not enough frames: {len(recorder_queue)}")
        return

    # Drop first 5 and last 5 frames for settling
    settled_records = recorder_queue[5:-5]
    logging.info(f"\nAnalyzing {len(settled_records)} frames")

    frame_times = []
    frame_intervals = []
    latencies = []
    prev_ts = None
    prev_frame_number = None
    dropped_frames = 0

    for (
        frame_start_s,
        frame_end_s,
        complete_ts,
        frame_num,
        frame_number,
    ) in settled_records:
        # Time from first byte received by FPGA to last byte (sensor readout)
        frame_times.append((frame_end_s - frame_start_s) * 1000)

        if prev_ts is not None:
            frame_intervals.append((complete_ts - prev_ts) * 1000)
        prev_ts = complete_ts

        # Detect dropped frames via FPGA frame_number gaps
        if prev_frame_number is not None:
            expected = (prev_frame_number + 1) & 0xFFFF  # 16-bit wrap
            if frame_number != expected:
                gap = (frame_number - prev_frame_number - 1) & 0xFFFF
                dropped_frames += gap
        prev_frame_number = frame_number

        latency = (complete_ts - frame_start_s) * 1000
        if latency < 0:
            logging.warning(f"Frame {frame_num}: Negative latency ({latency:.1f}ms)")
        latencies.append(latency)

    def stats(name, values):
        if values:
            logging.info(
                f"{name:<35} {min(values):>10.2f} {max(values):>10.2f} {sum(values)/len(values):>10.2f}"
            )

    logging.info(f"\n{'Metric':<35} {'Min (ms)':>10} {'Max (ms)':>10} {'Avg (ms)':>10}")
    logging.info("-" * 70)
    stats("Frame Time (sensor readout)", frame_times)
    stats("Frame Interval", frame_intervals)
    stats("Pipeline Latency (FSR->Holoviz)", latencies)

    if frame_intervals:
        fps = 1000.0 / (sum(frame_intervals) / len(frame_intervals))
        logging.info(f"\nFrame Rate: {fps:.2f} FPS")

    logging.info(f"Dropped Frames: {dropped_frames}")


def main():
    """Main entry point for VB1940 latency measurement."""
    parser = argparse.ArgumentParser(
        description="VB1940 latency measurement (FuSa CoE)"
    )

    modes = hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode
    mode_choices = [m.value for m in modes]
    mode_help = ", ".join([f"{m.value}={m.name}" for m in modes])

    parser.add_argument(
        "--camera-mode",
        type=int,
        choices=mode_choices,
        default=mode_choices[0],
        help=f"Choices: {mode_help}",
    )
    parser.add_argument("--frame-limit", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=1500)
    parser.add_argument("--hololink", default="192.168.0.2")
    parser.add_argument("--log-level", type=int, default=20)
    parser.add_argument("--sensor", type=int, choices=[0, 1], default=0)
    parser.add_argument("--interface", default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=620)

    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    hololink_module.logging_level(args.log_level)

    logging.info(
        f"HSB: {args.hololink}, Sensor: {args.sensor}, Frames: {args.frame_limit}"
    )

    # Setup HSB
    try:
        channel_metadata = hololink_module.Enumerator.find_channel(
            channel_ip=args.hololink
        )
    except Exception as e:
        logging.error(f"HSB not found: {e}")
        return 1

    channel_metadata = hololink_module.Metadata(channel_metadata)
    hololink_module.DataChannel.use_sensor(channel_metadata, args.sensor)
    hololink_channel = hololink_module.DataChannel(channel_metadata)

    hololink = hololink_channel.hololink()
    hololink.start()
    hololink.reset()

    # PTP sync (required for cross-domain latency measurement)
    logging.info("Synchronizing PTP...")
    try:
        if not hololink.ptp_synchronize():
            logging.error("PTP synchronization failed")
            hololink.stop()
            return 1
        logging.info("PTP synchronized")
    except Exception as e:
        logging.error(f"PTP error: {e}")
        hololink.stop()
        return 1

    # Camera setup
    try:
        camera_mode = hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode(
            args.camera_mode
        )
        camera = hololink_module.sensors.vb1940.vb1940.Vb1940Cam(hololink_channel)
        camera.setup_clock()
        camera.configure(camera_mode)
        logging.info(f"Camera: {camera_mode.name}")
    except Exception as e:
        logging.error(f"Camera setup failed: {e}")
        hololink.stop()
        return 1

    # Verify PTP still synced after camera setup
    if hasattr(hololink, "ptp_synchronized") and not hololink.ptp_synchronized():
        logging.error("PTP sync lost after camera setup")
        hololink.stop()
        return 1

    recorder_queue = []

    try:
        app = App(
            hololink_channel,
            camera,
            camera_mode,
            args.timeout,
            args.frame_limit,
            args.interface,
            recorder_queue,
            args.headless,
            args.width,
            args.height,
        )
        app.run()
    except Exception as e:
        logging.error(f"Failed: {e}")
        return 1
    finally:
        hololink.stop()

    analyze_latency(recorder_queue)
    return 0


if __name__ == "__main__":
    exit(main())
