# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
IMX274 Frame Validation with PVA CRC Hardware

This script validates IMX274 camera frames using PVA (Programmable Vision Accelerator)
hardware for CRC computation, comparing PVA CRC against camera FPGA-embedded CRC values.

PVA computes line-based CRC with GF(2) polynomial merging, which is mathematically
equivalent to streaming CRC32/JAMCRC when processing the same data.
"""

import argparse
import datetime
import logging
import math
import queue
import threading

import cupy as cp
import holoscan
from cuda.bindings import driver as cuda

import hololink as hololink_module

MS_PER_SEC = 1000.0
US_PER_SEC = 1000.0 * MS_PER_SEC
NS_PER_SEC = 1000.0 * US_PER_SEC
SEC_PER_NS = 1.0 / NS_PER_SEC

# Shared lock for thread-safe access to recorder_queue internals
recorder_queue_lock = threading.Lock()


# Helper operators for metadata recording
class PassThroughOperator(holoscan.core.Operator):
    """Pass through operator that optionally renames tensors"""

    def __init__(self, *args, in_tensor_name="", out_tensor_name="", **kwargs):
        super().__init__(*args, **kwargs)
        self._in_tensor_name = in_tensor_name
        self._out_tensor_name = out_tensor_name

    def setup(self, spec):
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("input")
        tensor = in_message.get(self._in_tensor_name)
        op_output.emit({self._out_tensor_name: tensor}, "output")


class RecordMetadataOp(PassThroughOperator):
    """Records metadata values for post-run validation"""

    def __init__(self, *args, metadata_names=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._metadata_names = (
            list(metadata_names) if metadata_names is not None else []
        )
        self._record = []
        self._frame_count = 0

    def compute(self, op_input, op_output, context):
        super().compute(op_input, op_output, context)
        metadata = self.metadata
        value = []
        for name in self._metadata_names:
            value.append(metadata[name])
        self._record.append(value)

        # Log CRC comparison every 20th frame (and first 5 frames) - only if DEBUG level
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            if self._frame_count < 5 or self._frame_count % 20 == 0:
                camera_crc = metadata.get("crc", 0)
                pva_crc = metadata.get("pva_frame_crc", 0)
                match_str = "✓ MATCH" if camera_crc == pva_crc else "✗ MISMATCH"
                logging.info(
                    f"Frame {self._frame_count}: Camera=0x{camera_crc:08x}, PVA=0x{pva_crc:08x} {match_str}"
                )

        self._frame_count += 1

    def get_record(self):
        return self._record


# Custom CRC validator - compares PVA CRC with camera embedded CRC


# The purpose of this function is to parse the recorder queue values, extract the values that needed for:
# 1. checking frame number of current and last frame
# 2. checking frame actual size compared to calculated size
# 3. checking timestamp consistency between current and last frame
# 4. checking for recorded CRC errors
def validate_frame(recorder_queue, expected_fps=60):

    if recorder_queue.qsize() > 1:
        # slice out only the relevant data from the recorder queue for frame validation
        # use shared lock for thread safety when accessing internal deque
        with recorder_queue_lock:
            internal_q = list(recorder_queue.queue)  # Access internal deque (as list)
            recorder_queue_raw = internal_q[-5:]  # Get last 5 records (peek only)
            sliced_recorder_queue = [
                sublist[-5:] for sublist in recorder_queue_raw
            ]  # slice out the last 5 elements from each record

        # get the last 2 frames of data and unpack the list items to variables
        (
            prev_complete_timestamp_s,
            prev_frame_number,
            prev_crc32_errors,
            _,  # prev_received_frame_size (unused)
            _,  # prev_calculated_frame_size (unused)
        ) = sliced_recorder_queue[-2]

        (
            complete_timestamp_s,
            frame_number,
            crc32_errors,
            received_frame_size,
            calculated_frame_size,
        ) = sliced_recorder_queue[-1]

        # Variable naming convention: _s suffix = seconds, _ms suffix = milliseconds. Python date/time objects do not have suffixes.

        # make sure frame numbers of current and last frames are in order
        if frame_number != prev_frame_number + 1:
            logging.info(
                f"Frame number is not in order, last frame number={prev_frame_number}, current frame number={frame_number}"
            )

        # check if the frame size is the same as the calculated frame size
        if received_frame_size != calculated_frame_size:
            logging.info(
                f"Frame size is not the same as the calculated frame size, received frame size={received_frame_size}, calculated frame size={calculated_frame_size}"
            )

        # Compare the timestamps of the current and last frames
        complete_timestamp = datetime.datetime.fromtimestamp(complete_timestamp_s)
        prev_complete_timestamp = datetime.datetime.fromtimestamp(
            prev_complete_timestamp_s
        )
        time_difference = complete_timestamp - prev_complete_timestamp

        # convert time difference to a float in milliseconds
        time_difference_ms = time_difference.total_seconds() * 1000
        logging.debug(
            f"Time difference between current and last frame is {time_difference_ms} ms"
        )

        # Calculate frame time threshold dynamically based on expected FPS
        # Frame time = 1000ms / FPS, we allow 2x frame time before calling an error
        expected_frame_time_ms = 1000.0 / expected_fps
        frame_time_threshold_ms = 2 * expected_frame_time_ms

        if time_difference_ms > frame_time_threshold_ms:
            logging.info(
                f"Frame timestamp mismatch, last frame timestamp={prev_complete_timestamp_s}, current frame timestamp={complete_timestamp_s},Diff[ms]={time_difference_ms}"
            )

        # check for CRC errors
        if crc32_errors is not None and prev_crc32_errors is not None:
            if crc32_errors > prev_crc32_errors:  # print only on new error
                logging.info(f"CRC32 errors found so far: {crc32_errors}")


def get_timestamp(metadata, name):
    s = metadata[f"{name}_s"]
    f = metadata[f"{name}_ns"]
    f *= SEC_PER_NS
    return s + f


def record_times(recorder_queue, metadata, expected_fps=60):
    #
    now = datetime.datetime.now(datetime.timezone.utc)
    #
    frame_number = metadata.get("frame_number", 0)

    # get frame crc errors (PVA errors for this pipeline)
    crc32_errors = metadata.get("pva_errors", 0)

    # frame_start_s is the time that the first data arrived at the FPGA;
    # the network receiver calls this "timestamp".
    frame_start_s = get_timestamp(metadata, "timestamp")

    # After the FPGA sends the last sensor data packet for a frame, it follows
    # that with a 128-byte metadata packet.  This timestamp (which the network
    # receiver calls "metadata") is the time at which the FPGA sends that
    # packet; so it's the time immediately after the the last byte of sensor
    # data in this window.  The difference between frame_start_s and frame_end_s
    # is how long it took for the sensor to produce enough data for a complete
    # frame.
    frame_end_s = get_timestamp(metadata, "metadata")

    # received_timestamp_s is the host time after the background thread woke up
    # with the nofication that a frame of data was available.  This shows how long
    # it took for the CPU to actually run the backtground user-mode thread where it observes
    # the end-of-frame.  This background thread sets a flag that will wake up
    # the pipeline network receiver operator.
    received_timestamp_s = get_timestamp(metadata, "received")

    # operator_timestamp_s is the time when the next pipeline element woke up--
    # the next operator after the network receiver.  This is used to compute
    # how much time overhead is required for the pipeline to actually receive
    # sensor data.
    operator_timestamp_s = get_timestamp(metadata, "operator_timestamp")

    # complete_timestamp_s is the time when visualization finished.
    complete_timestamp_s = get_timestamp(metadata, "complete_timestamp")

    # get the frame size and calculated frame size from the metadata
    received_frame_size = metadata.get("received_frame_size")
    calculated_frame_size = metadata.get("calculated_frame_size")

    # Use thread-safe put() operation instead of append()
    recorder_queue.put(
        (
            now,
            frame_start_s,
            frame_end_s,
            received_timestamp_s,
            operator_timestamp_s,
            complete_timestamp_s,
            frame_number,
            crc32_errors,
            received_frame_size,
            calculated_frame_size,
        )
    )

    # validate the frame
    validate_frame(recorder_queue, expected_fps)


def save_timestamp(metadata, name, timestamp):
    f, s = math.modf(timestamp.timestamp())
    metadata[f"{name}_s"] = int(s)
    metadata[f"{name}_ns"] = int(f * NS_PER_SEC)


class InstrumentedTimeProfiler(holoscan.core.Operator):
    def __init__(
        self,
        *args,
        operator_name="operator",
        calculated_frame_size=0,
        crc_frame_check=1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._operator_name = operator_name
        self._calculated_frame_size = calculated_frame_size
        self._crc_frame_check = crc_frame_check

    def setup(self, spec):
        logging.info("setup")
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        # What time is it now?
        operator_timestamp = datetime.datetime.now(datetime.timezone.utc)

        in_message = op_input.receive("input")
        cp_frame = cp.asarray(in_message.get(""))

        # Save the number of bytes in the frame to compare to CSI calculated frame size
        self.metadata["received_frame_size"] = cp_frame.nbytes
        self.metadata["calculated_frame_size"] = self._calculated_frame_size

        save_timestamp(
            self.metadata, self._operator_name + "_timestamp", operator_timestamp
        )
        op_output.emit({"": cp_frame}, "output")


class MonitorOperator(holoscan.core.Operator):
    def __init__(
        self,
        *args,
        recorder_queue=None,
        expected_fps=60,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._recorder_queue = recorder_queue
        self._expected_fps = expected_fps

    def setup(self, spec):
        logging.info("setup")
        spec.input("input")

    def compute(self, op_input, op_output, context):
        # What time is it now?
        complete_timestamp = datetime.datetime.now(datetime.timezone.utc)
        _ = op_input.receive("input")

        # save the complete timestamp and record the times
        save_timestamp(self.metadata, "complete_timestamp", complete_timestamp)
        record_times(self._recorder_queue, self.metadata, self._expected_fps)


class HoloscanApplication(holoscan.core.Application):
    def __init__(
        self,
        headless,
        fullscreen,
        cuda_context,
        cuda_device_ordinal,
        hololink_channel,
        ibv_name,
        ibv_port,
        camera,
        camera_mode,
        frame_limit,
        recorder_queue,
    ):
        super().__init__()
        self._headless = headless
        self._fullscreen = fullscreen
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._hololink_channel = hololink_channel
        self._ibv_name = ibv_name
        self._ibv_port = ibv_port
        self._camera = camera
        self._camera_mode = camera_mode
        self._frame_limit = frame_limit
        self._recorder_queue = recorder_queue
        self.crc_validator = None

    def _validate_pva_frame_size(self, frame_size):
        """
        Validate frame size meets PVA 1D API constraints.

        PVA 1D API constraints:
        - data_size must be a multiple of 4 (hardware alignment)
        - data_size must be >= 32KB (minimum transfer size)
        - data_size must be <= 12MB (maximum single transfer)

        Args:
            frame_size: Total frame size in bytes (including CSI header)

        Raises:
            ValueError: If frame size doesn't meet PVA constraints
        """
        min_size = 32 * 1024  # 32KB minimum
        max_size = 12 * 1024 * 1024  # 12MB maximum

        if frame_size % 4 != 0:
            raise ValueError(
                f"Frame size {frame_size:,} bytes is not a multiple of 4.\n"
                f"PVA requires data_size to be 4-byte aligned."
            )

        if frame_size < min_size:
            raise ValueError(
                f"Frame size {frame_size:,} bytes is too small.\n"
                f"PVA requires data_size >= {min_size:,} bytes (32KB)."
            )

        if frame_size > max_size:
            raise ValueError(
                f"Frame size {frame_size:,} bytes is too large.\n"
                f"PVA supports data_size <= {max_size:,} bytes (12MB).\n"
                f"Consider processing in chunks using remaining_size parameter."
            )

        logging.debug(f"✓ Frame size {frame_size:,} bytes is valid for PVA 1D API")

    def compose(self):
        logging.info("Composing PVA CRC validation pipeline")

        if self._frame_limit:
            condition = holoscan.conditions.CountCondition(
                self, name="count", count=self._frame_limit
            )
        else:
            condition = holoscan.conditions.BooleanCondition(
                self, name="ok", enable_tick=True
            )

        self._camera.set_mode(self._camera_mode)

        csi_to_bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="pool",
            storage_type=1,
            block_size=self._camera._width
            * 2
            * self._camera._height,  # 2 bytes per uint16
            num_blocks=4,
        )

        csi_to_bayer_operator = hololink_module.operators.CsiToBayerOp(
            self,
            name="csi_to_bayer",
            allocator=csi_to_bayer_pool,
            cuda_device_ordinal=self._cuda_device_ordinal,
        )
        self._camera.configure_converter(csi_to_bayer_operator)

        frame_size = csi_to_bayer_operator.get_csi_length()
        logging.debug(f"Frame size: {frame_size} bytes")

        receiver_operator = hololink_module.operators.RoceReceiverOp(
            self,
            condition,
            name="receiver",
            frame_size=frame_size,
            frame_context=self._cuda_context,
            ibv_name=self._ibv_name,
            ibv_port=self._ibv_port,
            hololink_channel=self._hololink_channel,
            device=self._camera,
        )

        # PVA CRC Configuration
        #
        # PVA uses line-based CRC with GF(2) polynomial merging, which is mathematically
        # equivalent to streaming CRC32/JAMCRC when processing the same data.
        # Both use polynomial 0x04C11DB7 (or bit-reversed 0xEDB88320).
        #
        # The camera's embedded CRC is computed over the full CSI frame.
        # To match, PVA must process the entire frame with valid 2D grid dimensions:
        #   - width (bytes per row) must be multiple of 4
        #   - height (number of rows) must be >= 64 and <= 16,384
        #   - width × height must equal frame_size exactly

        # Calculate bytes per line for RAW10 format
        # RAW10: 10 bits per pixel = 5 bytes per 4 pixels
        pixel_width = self._camera._width
        pixel_height = self._camera._height
        bytes_per_line_pixel = ((pixel_width + 3) // 4) * 5
        pixel_data_size = bytes_per_line_pixel * pixel_height

        # The CSI header is: frame_size - pixel_data_size
        csi_header_size = frame_size - pixel_data_size

        logging.debug("Frame breakdown:")
        logging.debug(f"  Full CSI frame:  {frame_size:,} bytes")
        logging.debug(f"  CSI header:      {csi_header_size:,} bytes")
        logging.debug(
            f"  Pixel data:      {pixel_data_size:,} bytes ({bytes_per_line_pixel} × {pixel_height})"
        )

        # Validate frame size meets PVA 1D API constraints
        self._validate_pva_frame_size(frame_size)

        logging.debug("Configuring PVA CRC (1D API - full frame):")
        logging.debug(f"  Camera mode: {pixel_width}x{pixel_height} RAW10")
        logging.debug(f"  Full CSI frame: {frame_size:,} bytes")
        logging.debug(f"  PVA data_size: {frame_size:,} bytes")
        logging.debug("  ✓ 1D API: No 2D dimension constraints")

        # Use async two-operator pattern to hide PVA latency
        self.compute_pva_crc_op = hololink_module.operators.ComputePvaCrcOp(
            self,
            name="compute_pva_crc",
            data_size=frame_size,
            remaining_size=0,
            is_first_chunk=True,
            use_dual_vpu=True,
        )

        self.check_pva_crc_op = hololink_module.operators.CheckPvaCrcOp(
            self,
            name="check_pva_crc",
            compute_pva_crc_op=self.compute_pva_crc_op,
            computed_crc_metadata_name="pva_frame_crc",
        )

        # Record both camera CRC and PVA CRC for post-run validation
        self.record_metadata_op = RecordMetadataOp(
            self,
            name="record_metadata",
            metadata_names=["crc", "pva_frame_crc"],
        )

        logging.debug("PVA CRC validation enabled (async compute/check pattern)")

        profiler = InstrumentedTimeProfiler(
            self,
            name="profiler",
            calculated_frame_size=frame_size,
            crc_frame_check=1,
        )

        pixel_format = self._camera.pixel_format()
        bayer_format = self._camera.bayer_format()

        image_processor_operator = hololink_module.operators.ImageProcessorOp(
            self,
            name="image_processor",
            optical_black=50,
            bayer_format=bayer_format.value,
            pixel_format=pixel_format.value,
        )

        bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="bayer_pool",
            storage_type=1,
            block_size=self._camera._width
            * 4
            * 2
            * self._camera._height,  # 4 channels * 2 bytes per uint16
            num_blocks=4,
        )

        demosaic = holoscan.operators.BayerDemosaicOp(
            self,
            name="demosaic",
            pool=bayer_pool,
            generate_alpha=True,
            alpha_value=65535,
            bayer_grid_pos=bayer_format.value,
            interpolation_mode=0,
        )

        visualizer = holoscan.operators.HolovizOp(
            self,
            name="holoviz",
            fullscreen=self._fullscreen,
            headless=self._headless,
            framebuffer_srgb=True,
            enable_camera_pose_output=True,
            camera_pose_output_type="extrinsics_model",
        )

        camera_fps = hololink_module.sensors.imx274.imx274_mode.imx_frame_format[
            self._camera_mode.value
        ].framerate

        monitor = MonitorOperator(
            self,
            name="monitor",
            recorder_queue=self._recorder_queue,
            expected_fps=camera_fps,
        )

        # Pipeline with async PVA CRC validation
        # Launch PVA computation early, then do other work while PVA computes
        self.add_flow(receiver_operator, self.compute_pva_crc_op, {("output", "input")})
        self.add_flow(self.compute_pva_crc_op, profiler, {("output", "input")})
        self.add_flow(profiler, csi_to_bayer_operator, {("output", "input")})
        self.add_flow(
            csi_to_bayer_operator, image_processor_operator, {("output", "input")}
        )
        # Check PVA results after image processing (hides ~0.9ms PVA latency)
        self.add_flow(
            image_processor_operator, self.check_pva_crc_op, {("output", "input")}
        )
        self.add_flow(
            self.check_pva_crc_op, self.record_metadata_op, {("output", "input")}
        )
        self.add_flow(self.record_metadata_op, demosaic, {("output", "receiver")})
        self.add_flow(demosaic, visualizer, {("transmitter", "receivers")})
        self.add_flow(visualizer, monitor, {("camera_pose_output", "input")})


def main():
    parser = argparse.ArgumentParser(description="IMX274 Frame Validation with PVA CRC")

    modes = hololink_module.sensors.imx274.imx274_mode.Imx274_Mode
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
        "--frame-limit", type=int, default=None, help="Exit after this many frames"
    )
    parser.add_argument(
        "--hololink", default="192.168.0.2", help="IP address of Hololink board"
    )
    parser.add_argument("--log-level", type=int, default=20, help="Logging level")

    infiniband_devices = hololink_module.infiniband_devices()
    parser.add_argument(
        "--ibv-name", default=infiniband_devices[0] if infiniband_devices else None
    )
    parser.add_argument(
        "--ibv-port", type=int, default=1, help="Port number of IBV device"
    )
    parser.add_argument("--expander-configuration", type=int, default=0, choices=(0, 1))
    parser.add_argument("--pattern", type=int, choices=range(12), help="Test pattern")
    parser.add_argument("--skip-reset", action="store_true")

    args = parser.parse_args()

    hololink_module.logging_level(args.log_level)
    logging.info("IMX274 Frame Validation with PVA CRC Hardware")

    # Initialize CUDA
    (cu_result,) = cuda.cuInit(0)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_device_ordinal = 0
    cu_result, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_result, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS

    # Initialize Hololink
    channel_metadata = hololink_module.Enumerator.find_channel(channel_ip=args.hololink)
    logging.info(f"Hololink channel: {channel_metadata}")
    hololink_channel = hololink_module.DataChannel(channel_metadata)

    # Initialize camera
    camera = hololink_module.sensors.imx274.dual_imx274.Imx274Cam(
        hololink_channel, expander_configuration=args.expander_configuration
    )
    camera_mode = hololink_module.sensors.imx274.imx274_mode.Imx274_Mode(
        args.camera_mode
    )

    recorder_queue = queue.Queue(maxsize=30_000)

    # Create application
    application = HoloscanApplication(
        args.headless,
        args.fullscreen,
        cu_context,
        cu_device_ordinal,
        hololink_channel,
        args.ibv_name,
        args.ibv_port,
        camera,
        camera_mode,
        args.frame_limit,
        recorder_queue,
    )

    # Start Hololink
    hololink = hololink_channel.hololink()
    hololink.start()
    if not args.skip_reset:
        hololink.reset()

    logging.info("Waiting for PTP sync...")
    if not hololink.ptp_synchronize():
        raise ValueError("Failed to synchronize PTP")
    logging.info("PTP synchronized")

    if not args.skip_reset:
        camera.setup_clock()

    camera.configure(camera_mode)
    camera.set_digital_gain_reg(0x4)

    if args.pattern is not None:
        camera.test_pattern(args.pattern)

    logging.info("Starting application...")
    application.run()

    hololink.stop()

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS

    # Validate CRCs after pipeline completes
    SKIP_INITIAL_FRAMES = 10  # Skip first N frames during pipeline initialization
    SKIP_FINAL_FRAMES = 10  # Skip last N frames during pipeline shutdown

    if hasattr(application, "record_metadata_op"):
        crcs = application.record_metadata_op.get_record()
        total_frames = len(crcs)

        if total_frames == 0:
            logging.warning("No CRC frames recorded")
        else:
            logging.info(f"Captured {total_frames} frames")

            # Validate CRCs (skip initial and final frames to avoid artifacts)
            skip_frames = SKIP_INITIAL_FRAMES
            pva_errors = 0

            for frame_idx, (camera_crc, pva_crc) in enumerate(crcs):
                # Skip validation for first N frames (initialization artifacts)
                if frame_idx < skip_frames:
                    continue
                # Skip validation for last N frames (shutdown artifacts)
                if frame_idx >= (total_frames - SKIP_FINAL_FRAMES):
                    continue

                if camera_crc != pva_crc:
                    logging.error(
                        f"CRC mismatch at frame {frame_idx}: "
                        f"camera=0x{camera_crc:08x}, pva=0x{pva_crc:08x}"
                    )
                    pva_errors += 1

            validated_frames = max(total_frames - skip_frames - SKIP_FINAL_FRAMES, 0)
            pva_success = (
                ((validated_frames - pva_errors) / validated_frames) * 100
                if validated_frames > 0
                else 0
            )

            if pva_errors == 0:
                logging.info(f"✓ Validated {validated_frames} CRCs - all matched!")

            logging.info("\n" + "=" * 80)
            logging.info("PVA CRC VALIDATION SUMMARY")
            logging.info("=" * 80)
            logging.info(f"Total frames: {total_frames}")
            logging.info(
                f"Validated frames: {validated_frames} (skipped first {skip_frames}, last {SKIP_FINAL_FRAMES})"
            )
            logging.info("")

            # Frame dimensions
            logging.debug("FRAME DIMENSIONS:")
            if hasattr(application, "compute_pva_crc_op") and hasattr(
                application, "_camera"
            ):
                camera = application._camera
                pva_op = application.compute_pva_crc_op

                # Input dimensions from camera
                camera_width_pixels = camera._width
                camera_height_pixels = camera._height

                # PVA configuration (1D API)
                pva_data_size = pva_op._data_size
                pva_start_byte = pva_op._start_byte

                logging.debug("  Camera:")
                logging.debug(f"    Width:           {camera_width_pixels:,} pixels")
                logging.debug(f"    Height:          {camera_height_pixels:,} pixels")
                logging.debug("  PVA Configuration (1D API):")
                logging.debug(f"    Data size:       {pva_data_size:,} bytes")
                logging.debug(f"    Start byte:      {pva_start_byte:,}")
            logging.info("")

            # PVA vs Camera analysis
            logging.info("PVA Hardware CRC vs Camera FPGA CRC:")
            logging.info(f"  Matches:    {validated_frames - pva_errors}")
            logging.info(f"  Mismatches: {pva_errors}")
            logging.info(f"  Success:    {pva_success:.2f}%")
            logging.info("")

            logging.info("INTERPRETATION:")
            logging.info("-" * 50)
            if pva_errors == 0:
                logging.info("✓ SUCCESS! PVA matches camera perfectly (100%)")
                logging.info("   → PVA hardware CRC validation is working correctly")
            else:
                logging.info(
                    f"✗ PVA has {pva_errors} mismatches ({100-pva_success:.2f}% error rate)"
                )
                logging.info("   → Check PVA configuration and data alignment")
            logging.info("=" * 80)

    # Report stats at the end of the application
    frame_time_dts = []
    cpu_latency_dts = []
    operator_latency_dts = []
    processing_time_dts = []
    overall_time_dts = []

    # Extract all items from the thread-safe queue for processing
    # This creates a snapshot for safe iteration while preserving the queue contents
    with recorder_queue_lock:
        recorder_queue_items = list(recorder_queue.queue)

    if len(recorder_queue_items) >= 100:
        sliced_recorder_queue = [sublist[:7] for sublist in recorder_queue_items]
        settled_timestamps = sliced_recorder_queue[5:-5]

        for (
            now,
            frame_start_s,
            frame_end_s,
            received_timestamp_s,
            operator_timestamp_s,
            complete_timestamp_s,
            frame_number,
        ) in settled_timestamps:

            frame_start = datetime.datetime.fromtimestamp(frame_start_s).isoformat()
            frame_end = datetime.datetime.fromtimestamp(frame_end_s).isoformat()
            received_timestamp = datetime.datetime.fromtimestamp(
                received_timestamp_s
            ).isoformat()
            operator_timestamp = datetime.datetime.fromtimestamp(
                operator_timestamp_s
            ).isoformat()
            complete_timestamp = datetime.datetime.fromtimestamp(
                complete_timestamp_s
            ).isoformat()

            frame_time_dt = frame_end_s - frame_start_s
            frame_time_dts.append(round(frame_time_dt, 4))

            cpu_latency_dt = received_timestamp_s - frame_end_s
            cpu_latency_dts.append(round(cpu_latency_dt, 4))

            operator_latency_dt = operator_timestamp_s - received_timestamp_s
            operator_latency_dts.append(round(operator_latency_dt, 4))

            processing_time_dt = complete_timestamp_s - operator_timestamp_s
            processing_time_dts.append(round(processing_time_dt, 4))

            overall_time_dt = complete_timestamp_s - frame_start_s
            overall_time_dts.append(round(overall_time_dt, 4))
            logging.debug(f"** Frame Information  for Frame Number = {frame_number}**")
            logging.debug(f"Frame Start          : {frame_start}")
            logging.debug(f"Frame End            : {frame_end}")
            logging.debug(f"Received Timestamp   : {received_timestamp}")
            logging.debug(f"Operator Timestamp   : {operator_timestamp}")
            logging.debug(f"Complete Timestamp   : {complete_timestamp}")
            logging.debug(f"Frame Time (dt)      : {frame_time_dt:.6f} s")
            logging.debug(f"CPU Latency (dt)     : {cpu_latency_dt:.6f} s")
            logging.debug(f"Operator Latency (dt): {operator_latency_dt:.6f} s")
            logging.debug(f"Processing Time (dt) : {processing_time_dt:.6f} s")
            logging.debug(f"Overall Time (dt)    : {overall_time_dt:.6f} s")

        logging.info("\n** PERFORMANCE REPORT **")
        logging.info(f"{'Metric':<30}{'Min':<15}{'Max':<15}{'Avg':<15}")
        #
        ft_min_time_difference = min(frame_time_dts)
        ft_max_time_difference = max(frame_time_dts)
        ft_avg_time_difference = sum(frame_time_dts) / len(frame_time_dts)
        logging.info("Frame Time (in sec):")
        logging.info(
            f"{'Frame Time':<30}{ft_min_time_difference:<15}{ft_max_time_difference:<15}{ft_avg_time_difference:<15}"
        )
        #
        cl_min_time_difference = min(cpu_latency_dts)
        cl_max_time_difference = max(cpu_latency_dts)
        cl_avg_time_difference = sum(cpu_latency_dts) / len(cpu_latency_dts)
        logging.info("FPGA frame transfer latency (in sec):")
        logging.info(
            f"{'Frame Transfer Latency':<30}{cl_min_time_difference:<15}{cl_max_time_difference:<15}{cl_avg_time_difference:<15}"
        )
        #
        ol_min_time_difference = min(operator_latency_dts)
        ol_max_time_difference = max(operator_latency_dts)
        ol_avg_time_difference = sum(operator_latency_dts) / len(operator_latency_dts)
        logging.info("FPGA to Operator after network operator latency (in sec):")
        logging.info(
            f"{'Operator Latency':<30}{ol_min_time_difference:<15}{ol_max_time_difference:<15}{ol_avg_time_difference:<15}"
        )
        #
        pt_min_time_difference = min(processing_time_dts)
        pt_max_time_difference = max(processing_time_dts)
        pt_avg_time_difference = sum(processing_time_dts) / len(processing_time_dts)
        logging.info("Processing of frame latency (in sec):")
        logging.info(
            f"{'Processing Latency':<30}{pt_min_time_difference:<15}{pt_max_time_difference:<15}{pt_avg_time_difference:<15}"
        )
        #
        ot_min_time_difference = min(overall_time_dts)
        ot_max_time_difference = max(overall_time_dts)
        ot_avg_time_difference = sum(overall_time_dts) / len(overall_time_dts)
        logging.info("Frame start till end of SW pipeline latency (in sec):")
        logging.info(
            f"{'SW Pipeline Latency':<30}{ot_min_time_difference:<15}{ot_max_time_difference:<15}{ot_avg_time_difference:<15}"
        )
    else:
        logging.info(
            f"Not enough frames ({len(recorder_queue_items)}) for performance statistics"
        )


if __name__ == "__main__":
    main()
