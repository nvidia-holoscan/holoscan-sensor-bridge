# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import ctypes
import datetime
import logging
import math
import os
import queue
import threading

import cuda.bindings.driver as cuda
import cupy as cp
import holoscan

import hololink as hololink_module

MS_PER_SEC = 1000.0
US_PER_SEC = 1000.0 * MS_PER_SEC
NS_PER_SEC = 1000.0 * US_PER_SEC
SEC_PER_NS = 1.0 / NS_PER_SEC

# Shared lock for thread-safe access to recorder_queue.queue internal deque
recorder_queue_lock = threading.Lock()


# Custom CRC checker operator that inherits from CheckCrcOp
class CrcValidationOp(hololink_module.operators.CheckCrcOp):
    """
    Custom CRC checker that validates computed CRC against received CRC
    and tracks errors.
    """

    def __init__(
        self,
        *args,
        compute_crc_op=None,
        crc_metadata_name="crc",
        **kwargs,
    ):
        hololink_module.operators.CheckCrcOp.__init__(
            self, *args, compute_crc_op=compute_crc_op, **kwargs
        )
        self._crc_metadata_name = crc_metadata_name
        self._frame_count = 0
        self._crc_errors = 0
        self._crcs = []  # list of (received_crc, computed_crc)

    def check_crc(self, computed_crc):
        """
        This method is called by the C++ base class CheckCrcOp.
        It receives the computed CRC and can compare it with metadata.
        """
        self._frame_count += 1
        received_crc = self.metadata.get(self._crc_metadata_name, 0)

        if received_crc != computed_crc:
            self._crc_errors += 1
            logging.warning(
                f"Frame {self._frame_count}: CRC mismatch! "
                f"computed={computed_crc:#x}, received={received_crc:#x}"
            )
        else:
            logging.debug(
                f"Frame {self._frame_count}: CRC match " f"({computed_crc:#x})"
            )

        self._crcs.append((received_crc, computed_crc))

        # Store error count in metadata for validation operator
        self.metadata["crc_errors"] = self._crc_errors

    def get_crc_errors(self):
        return self._crc_errors

    def get_crcs(self):
        return self._crcs


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
            f"Time difference between current and last frame is {time_difference} ms"
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


def print_crc_results(crcs, crc_errors):
    """
    Print detailed CRC validation results for debugging.

    Args:
        crcs: List of tuples (received_crc, computed_crc) for each frame
        crc_errors: Total number of CRC errors
    """
    total_frames = len(crcs)

    # Print summary at info level
    if total_frames > 0:
        success_rate = ((total_frames - crc_errors) / total_frames) * 100
        logging.info(
            f"CRC Validation: {total_frames} frames, {crc_errors} errors, {success_rate:.2f}% success"
        )

    logging.debug("\n" + "=" * 60)
    logging.debug("CRC VALIDATION RESULTS (nvcomp 5.0)")
    logging.debug("=" * 60)
    logging.debug(f"Total frames processed: {total_frames}")
    logging.debug(f"CRC errors: {crc_errors}")
    if total_frames > 0:
        success_rate = ((total_frames - crc_errors) / total_frames) * 100
        logging.debug(f"Success rate: {success_rate:.2f}%")
    logging.debug("=" * 60)

    # List all CRCs for debugging
    logging.debug("\nDETAILED CRC LIST:")
    logging.debug("-" * 80)
    logging.debug(
        f"{'Frame':<8} {'Received CRC':<20} {'Computed CRC':<20} {'Status':<10}"
    )
    logging.debug("-" * 80)
    for frame_idx, (received_crc, computed_crc) in enumerate(crcs):
        status = "✓ MATCH" if received_crc == computed_crc else "✗ MISMATCH"
        logging.debug(
            f"{frame_idx:<8} {received_crc:#018x} {computed_crc:#018x} {status:<10}"
        )
    logging.debug("-" * 80)

    # Show only mismatches if there are many frames
    if total_frames > 20 and crc_errors > 0:
        logging.debug("\nCRC MISMATCHES ONLY:")
        logging.debug("-" * 80)
        logging.debug(
            f"{'Frame':<8} {'Received CRC':<20} {'Computed CRC':<20} {'Diff':<20}"
        )
        logging.debug("-" * 80)
        for frame_idx, (received_crc, computed_crc) in enumerate(crcs):
            if received_crc != computed_crc:
                diff = received_crc ^ computed_crc  # XOR to show bit differences
                logging.debug(
                    f"{frame_idx:<8} {received_crc:#018x} {computed_crc:#018x} {diff:#018x}"
                )
        logging.debug("-" * 80)


def get_timestamp(metadata, name):
    s = metadata[f"{name}_s"]
    f = metadata[f"{name}_ns"]
    f *= SEC_PER_NS
    return s + f


def record_times(recorder_queue, metadata, expected_fps=60):
    #
    now = datetime.datetime.now(datetime.UTC)
    #
    frame_number = metadata.get("frame_number", 0)

    # get frame crc errors
    crc32_errors = metadata.get("crc_errors", 0)

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
    # This method works around the fact that we can't store
    # datetime objects in metadata.
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
        operator_timestamp = datetime.datetime.now(datetime.UTC)

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
        complete_timestamp = datetime.datetime.now(datetime.UTC)
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
        crc_frame_check,
    ):
        logging.info("__init__")
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
        self._crc_frame_check = crc_frame_check

        # This is a control for HSDK
        self.is_metadata_enabled = True

        # CRC validator will be set in compose if enabled
        self.crc_validator = None

    def compose(self):
        logging.info("compose")
        if self._frame_limit:
            self._count = holoscan.conditions.CountCondition(
                self,
                name="count",
                count=self._frame_limit,
            )
            condition = self._count
        else:
            self._ok = holoscan.conditions.BooleanCondition(
                self, name="ok", enable_tick=True
            )
            condition = self._ok
        self._camera.set_mode(self._camera_mode)

        csi_to_bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="pool",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=self._camera._width
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera._height,
            num_blocks=2,
        )
        csi_to_bayer_operator = hololink_module.operators.CsiToBayerOp(
            self,
            name="csi_to_bayer",
            allocator=csi_to_bayer_pool,
            cuda_device_ordinal=self._cuda_device_ordinal,
        )
        self._camera.configure_converter(csi_to_bayer_operator)

        frame_size = csi_to_bayer_operator.get_csi_length()
        logging.info(f"{frame_size=}")

        frame_context = self._cuda_context
        receiver_operator = hololink_module.operators.RoceReceiverOp(
            self,
            condition,
            name="receiver",
            frame_size=frame_size,
            frame_context=frame_context,
            ibv_name=self._ibv_name,
            ibv_port=self._ibv_port,
            hololink_channel=self._hololink_channel,
            device=self._camera,
        )

        # CRC operators using nvcomp 5.0 (optional)
        # Note: Unlike Linux version which samples every Nth frame,
        # GPU-based CRC is fast enough to check every frame (when crc_frame_check=1)
        if self._crc_frame_check > 0:
            compute_crc = hololink_module.operators.ComputeCrcOp(
                self,
                name="compute_crc",
                frame_size=frame_size,
            )
            self.crc_validator = CrcValidationOp(
                self,
                name="crc_validator",
                compute_crc_op=compute_crc,
                crc_metadata_name="crc",
            )
            logging.info("CRC validation enabled (nvcomp 5.0) - checking every frame")
        else:
            logging.info("CRC validation disabled")

        profiler = InstrumentedTimeProfiler(
            self,
            name="profiler",
            calculated_frame_size=frame_size,
            crc_frame_check=self._crc_frame_check,
        )

        pixel_format = self._camera.pixel_format()
        bayer_format = self._camera.bayer_format()
        image_processor_operator = hololink_module.operators.ImageProcessorOp(
            self,
            name="image_processor",
            # Optical black value for imx274 is 50
            optical_black=50,
            bayer_format=bayer_format.value,
            pixel_format=pixel_format.value,
        )

        rgba_components_per_pixel = 4
        bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="pool",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=self._camera._width
            * rgba_components_per_pixel
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera._height,
            num_blocks=2,
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

        # Get FPS from camera mode for dynamic frame time validation
        camera_fps = hololink_module.sensors.imx274.imx274_mode.imx_frame_format[
            self._camera_mode.value
        ].framerate

        monitor = MonitorOperator(
            self,
            name="monitor",
            recorder_queue=self._recorder_queue,
            expected_fps=camera_fps,
        )

        # Pipeline flow - conditionally include CRC validation
        if self._crc_frame_check > 0:
            # Pipeline with CRC validation using nvcomp 5.0
            self.add_flow(receiver_operator, compute_crc, {("output", "input")})
            self.add_flow(compute_crc, self.crc_validator, {("output", "input")})
            self.add_flow(self.crc_validator, profiler, {("output", "input")})
        else:
            # Pipeline without CRC validation
            self.add_flow(receiver_operator, profiler, {("output", "input")})

        self.add_flow(profiler, csi_to_bayer_operator, {("output", "input")})
        self.add_flow(
            csi_to_bayer_operator, image_processor_operator, {("output", "input")}
        )
        self.add_flow(image_processor_operator, demosaic, {("output", "receiver")})
        self.add_flow(demosaic, visualizer, {("transmitter", "receivers")})
        self.add_flow(visualizer, monitor, {("camera_pose_output", "input")})

    def _terminate(self, recorded_timestamps):
        self._ok.disable_tick()
        global timestamps
        timestamps = recorded_timestamps


def main():
    parser = argparse.ArgumentParser()
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
        "--frame-limit",
        type=int,
        default=None,
        help="Exit after receiving this many frames",
    )
    default_configuration = os.path.join(
        os.path.dirname(__file__), "example_configuration.yaml"
    )
    parser.add_argument(
        "--configuration",
        default=default_configuration,
        help="Configuration file",
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
    infiniband_devices = hololink_module.infiniband_devices()
    parser.add_argument(
        "--ibv-name",
        default=infiniband_devices[0] if infiniband_devices else None,
        help="IBV device to use",
    )
    parser.add_argument(
        "--ibv-port",
        type=int,
        default=1,
        help="Port number of IBV device",
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
        default=1,
        help="GPU-based CRC validation using nvcomp 5.0: 0=disabled, 1=check every frame (default). Note: unlike CPU-based checking, GPU CRC is fast enough to always check every frame.",
    )

    args = parser.parse_args()
    hololink_module.logging_level(args.log_level)
    logging.info("Initializing.")
    # Get a handle to the GPU
    (cu_result,) = cuda.cuInit(0)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_device_ordinal = 0
    cu_result, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_result, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    # Get a handle to the Hololink device
    channel_metadata = hololink_module.Enumerator.find_channel(channel_ip=args.hololink)
    logging.info(f"{channel_metadata=}")
    hololink_channel = hololink_module.DataChannel(channel_metadata)
    # Get a handle to the camera
    camera = hololink_module.sensors.imx274.dual_imx274.Imx274Cam(
        hololink_channel, expander_configuration=args.expander_configuration
    )
    camera_mode = hololink_module.sensors.imx274.imx274_mode.Imx274_Mode(
        args.camera_mode
    )

    # Thread-safe recorder queue for timestamps - allows about 10 minutes of queuing for 60fps
    recorder_queue = queue.Queue(maxsize=30_000)

    # Set up the application
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
        args.crc_frame_check,
    )
    application.config(args.configuration)
    # Run it.
    hololink = hololink_channel.hololink()
    hololink.start()
    if not args.skip_reset:
        hololink.reset()
    logging.debug("Waiting for PTP sync.")
    if not hololink.ptp_synchronize():
        raise ValueError("Failed to synchronize PTP.")
    else:
        logging.debug("PTP synchronized.")
    if not args.skip_reset:
        camera.setup_clock()
    camera.configure(camera_mode)
    camera.set_digital_gain_reg(0x4)
    if args.pattern is not None:
        camera.test_pattern(args.pattern)
    logging.info("Calling run")
    application.run()
    hololink.stop()

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS

    # Report CRC validation results (if enabled)
    if args.crc_frame_check > 0 and application.crc_validator:
        crc_errors = application.crc_validator.get_crc_errors()
        crcs = application.crc_validator.get_crcs()
        print_crc_results(crcs, crc_errors)
    else:
        logging.info("CRC validation was disabled (--crc-frame-check=0)")

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
