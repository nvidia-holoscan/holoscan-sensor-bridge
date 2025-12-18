# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pytest for stereo IMX274 frame validation.
Tests CRC validation, frame timing, and metadata consistency for stereo camera setup.
"""

import ctypes
import datetime
import logging
import math
import queue
import threading

import cuda.bindings.driver as cuda
import holoscan
import operators
import pytest
import utils

import hololink as hololink_module

# Constants for timing
MS_PER_SEC = 1000.0
US_PER_SEC = 1000.0 * MS_PER_SEC
NS_PER_SEC = 1000.0 * US_PER_SEC
SEC_PER_NS = 1.0 / NS_PER_SEC

# Number of frames to skip during pipeline initialization to avoid artifacts
SKIP_INITIAL_FRAMES = 15


class CameraWrapper(hololink_module.sensors.imx274.dual_imx274.Imx274Cam):
    """Camera wrapper that tracks reset callbacks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reset_callbacks = 0
        self._hololink.on_reset(self._reset)

    def _reset(self):
        self._reset_callbacks += 1
        logging.info(f"Camera reset callback count: {self._reset_callbacks}")


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
        camera_name="",
        skip_initial_frames=SKIP_INITIAL_FRAMES,
        **kwargs,
    ):
        hololink_module.operators.CheckCrcOp.__init__(
            self, *args, compute_crc_op=compute_crc_op, **kwargs
        )
        self._crc_metadata_name = crc_metadata_name
        self._camera_name = camera_name
        self._frame_count = 0
        self._crc_errors = 0
        self._crcs = []  # list of (received_crc, computed_crc)
        self._skip_initial_frames = skip_initial_frames

    def check_crc(self, computed_crc):
        """
        This method is called by the C++ base class CheckCrcOp.
        It receives the computed CRC and can compare it with metadata.
        """
        self._frame_count += 1
        received_crc = self.metadata.get(self._crc_metadata_name, 0)

        # Skip validation for initial frames if configured (avoids initialization artifacts)
        if self._frame_count <= self._skip_initial_frames:
            logging.debug(
                f"[{self._camera_name}] Frame {self._frame_count}: "
                f"Skipping CRC validation (initialization phase)"
            )
            self._crcs.append((received_crc, computed_crc))
            return

        if received_crc != computed_crc:
            self._crc_errors += 1
            logging.error(
                f"[{self._camera_name}] Frame {self._frame_count}: CRC mismatch! "
                f"computed={computed_crc:#x}, received={received_crc:#x}"
            )
        else:
            logging.debug(
                f"[{self._camera_name}] Frame {self._frame_count}: CRC match ({computed_crc:#x})"
            )

        self._crcs.append((received_crc, computed_crc))

        # Store error count in metadata for validation operator
        self.metadata["crc_errors"] = self._crc_errors

    def get_crc_errors(self):
        return self._crc_errors

    def get_crcs(self):
        return self._crcs


def save_timestamp(metadata, name, timestamp):
    """Save timestamp to metadata (splits into seconds and nanoseconds)."""
    f, s = math.modf(timestamp.timestamp())
    metadata[f"{name}_s"] = int(s)
    metadata[f"{name}_ns"] = int(f * NS_PER_SEC)


def get_timestamp(metadata, name):
    """Get timestamp from metadata."""
    s = metadata[f"{name}_s"]
    f = metadata[f"{name}_ns"]
    f *= SEC_PER_NS
    return s + f


class FrameValidationData:
    """Store frame validation data for a camera."""

    def __init__(self, camera_name=""):
        self.camera_name = camera_name
        self.frame_count = 0
        self.frame_order_errors = 0
        self.frame_size_errors = 0
        self.timestamp_errors = 0
        self.frames = []  # Store frame metadata for analysis

    def add_frame(self, frame_data):
        self.frames.append(frame_data)
        self.frame_count += 1


validate_frame_lock = threading.Lock()


def validate_frame(recorder_queue, validation_data, expected_frame_time_ms=16.6):
    """
    Validate frame data from recorder queue.

    Args:
        recorder_queue: Queue containing frame metadata
        validation_data: FrameValidationData object to store results
        expected_frame_time_ms: Expected frame time in milliseconds
    """
    if recorder_queue.qsize() > 1:
        # Slice out only the relevant data from the recorder queue for frame validation
        with validate_frame_lock:
            internal_q = list(recorder_queue.queue)
            recorder_queue_raw = internal_q[-5:]
            sliced_recorder_queue = [sublist[-5:] for sublist in recorder_queue_raw]

        # Get the last 2 frames of data and unpack the list items to variables
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

        # Check frame numbers are in order
        if frame_number != prev_frame_number + 1:
            validation_data.frame_order_errors += 1
            logging.warning(
                f"[{validation_data.camera_name}] Frame number out of order: "
                f"previous={prev_frame_number}, current={frame_number}"
            )

        # Check if the frame size matches the calculated frame size
        if received_frame_size != calculated_frame_size:
            validation_data.frame_size_errors += 1
            logging.warning(
                f"[{validation_data.camera_name}] Frame size mismatch: "
                f"received={received_frame_size}, expected={calculated_frame_size}"
            )

        # Compare the timestamps of the current and last frames
        complete_timestamp = datetime.datetime.fromtimestamp(complete_timestamp_s)
        prev_complete_timestamp = datetime.datetime.fromtimestamp(
            prev_complete_timestamp_s
        )
        time_difference = complete_timestamp - prev_complete_timestamp
        time_difference_ms = time_difference.total_seconds() * MS_PER_SEC

        # Allow 2x frame time before calling an error
        if time_difference_ms > (2 * expected_frame_time_ms):
            validation_data.timestamp_errors += 1
            logging.warning(
                f"[{validation_data.camera_name}] Frame timestamp gap too large: "
                f"last={prev_complete_timestamp_s}, current={complete_timestamp_s}, "
                f"diff={time_difference_ms:.2f}ms"
            )


def record_times(recorder_queue, metadata, camera_name=""):
    """Record timing information from metadata to queue."""
    now = datetime.datetime.now(datetime.UTC)
    frame_number = metadata.get("frame_number", 0)
    crc32_errors = metadata.get("crc_errors", 0)

    frame_start_s = get_timestamp(metadata, "timestamp")
    frame_end_s = get_timestamp(metadata, "metadata")
    received_timestamp_s = get_timestamp(metadata, "received")
    operator_timestamp_s = get_timestamp(metadata, "operator_timestamp")
    complete_timestamp_s = get_timestamp(metadata, "complete_timestamp")

    received_frame_size = metadata.get("received_frame_size")
    calculated_frame_size = metadata.get("calculated_frame_size")

    # Use thread-safe put() operation
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


class InstrumentedTimeProfiler(holoscan.core.Operator):
    """Operator to profile timing and add frame size metadata."""

    def __init__(
        self,
        *args,
        operator_name="operator",
        calculated_frame_size=0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._operator_name = operator_name
        self._calculated_frame_size = calculated_frame_size

    def setup(self, spec):
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        operator_timestamp = datetime.datetime.now(datetime.UTC)

        in_message = op_input.receive("input")
        cp_frame = holoscan.as_tensor(in_message.get(""))

        # Save the number of bytes in the frame to compare to CSI calculated frame size
        self.metadata["received_frame_size"] = cp_frame.nbytes
        self.metadata["calculated_frame_size"] = self._calculated_frame_size

        save_timestamp(self.metadata, "operator_timestamp", operator_timestamp)
        op_output.emit({"": cp_frame}, "output")


class MonitorOperator(holoscan.core.Operator):
    """Operator to monitor frame data and record timing information."""

    def __init__(
        self,
        *args,
        recorder_queue=None,
        validation_data=None,
        camera_name="",
        expected_frame_time_ms=16.6,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._recorder_queue = recorder_queue
        self._validation_data = validation_data
        self._camera_name = camera_name
        self._expected_frame_time_ms = expected_frame_time_ms

    def setup(self, spec):
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        complete_timestamp = datetime.datetime.now(datetime.UTC)
        in_message = op_input.receive("input")

        # Save the complete timestamp and record the times
        save_timestamp(self.metadata, "complete_timestamp", complete_timestamp)
        record_times(self._recorder_queue, self.metadata, self._camera_name)

        # Validate the frame
        validate_frame(
            self._recorder_queue, self._validation_data, self._expected_frame_time_ms
        )

        # Pass through the data
        op_output.emit(in_message, "output")


class StereoFrameValidationApplication(holoscan.core.Application):
    """Application for testing stereo IMX274 frame validation."""

    def __init__(
        self,
        headless,
        cuda_context,
        cuda_device_ordinal,
        hololink_channel_left,
        ibv_name_left,
        ibv_port_left,
        camera_left,
        hololink_channel_right,
        ibv_name_right,
        ibv_port_right,
        camera_right,
        camera_mode,
        frame_limit,
        recorder_queue_left,
        recorder_queue_right,
        validation_data_left,
        validation_data_right,
        watchdog,
    ):
        super().__init__()
        self._headless = headless
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._hololink_channel_left = hololink_channel_left
        self._ibv_name_left = ibv_name_left
        self._ibv_port_left = ibv_port_left
        self._camera_left = camera_left
        self._hololink_channel_right = hololink_channel_right
        self._ibv_name_right = ibv_name_right
        self._ibv_port_right = ibv_port_right
        self._camera_right = camera_right
        self._camera_mode = camera_mode
        self._frame_limit = frame_limit
        self._recorder_queue_left = recorder_queue_left
        self._recorder_queue_right = recorder_queue_right
        self._validation_data_left = validation_data_left
        self._validation_data_right = validation_data_right
        self._watchdog = watchdog

        # HSDK controls for stereo paths going into the same visualizer
        self.is_metadata_enabled = True
        self.metadata_policy = holoscan.core.MetadataPolicy.REJECT

        # CRC validators will be set in compose
        self.crc_validator_left = None
        self.crc_validator_right = None

    def compose(self):
        logging.info("compose")

        # Set up conditions for frame limit
        self._condition_left = holoscan.conditions.CountCondition(
            self,
            name="count_left",
            count=self._frame_limit,
        )
        self._condition_right = holoscan.conditions.CountCondition(
            self,
            name="count_right",
            count=self._frame_limit,
        )

        self._camera_left.set_mode(self._camera_mode)
        self._camera_right.set_mode(self._camera_mode)

        # Separate memory pools for each camera to avoid contention
        csi_to_bayer_pool_left = holoscan.resources.BlockMemoryPool(
            self,
            name="csi_to_bayer_pool_left",
            storage_type=1,  # device memory
            block_size=self._camera_left._width
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera_left._height,
            num_blocks=4,
        )
        csi_to_bayer_pool_right = holoscan.resources.BlockMemoryPool(
            self,
            name="csi_to_bayer_pool_right",
            storage_type=1,  # device memory
            block_size=self._camera_right._width
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera_right._height,
            num_blocks=4,
        )

        csi_to_bayer_operator_left = hololink_module.operators.CsiToBayerOp(
            self,
            name="csi_to_bayer_left",
            allocator=csi_to_bayer_pool_left,
            cuda_device_ordinal=self._cuda_device_ordinal,
            out_tensor_name="left",
        )
        self._camera_left.configure_converter(csi_to_bayer_operator_left)

        csi_to_bayer_operator_right = hololink_module.operators.CsiToBayerOp(
            self,
            name="csi_to_bayer_right",
            allocator=csi_to_bayer_pool_right,
            cuda_device_ordinal=self._cuda_device_ordinal,
            out_tensor_name="right",
        )
        self._camera_right.configure_converter(csi_to_bayer_operator_right)

        frame_size = csi_to_bayer_operator_left.get_csi_length()
        assert frame_size == csi_to_bayer_operator_right.get_csi_length()
        logging.info(f"Frame size: {frame_size} bytes")

        frame_context = self._cuda_context
        receiver_operator_left = hololink_module.operators.RoceReceiverOp(
            self,
            self._condition_left,
            name="receiver_left",
            frame_size=frame_size,
            frame_context=frame_context,
            ibv_name=self._ibv_name_left,
            ibv_port=self._ibv_port_left,
            hololink_channel=self._hololink_channel_left,
            device=self._camera_left,
        )

        receiver_operator_right = hololink_module.operators.RoceReceiverOp(
            self,
            self._condition_right,
            name="receiver_right",
            frame_size=frame_size,
            frame_context=frame_context,
            ibv_name=self._ibv_name_right,
            ibv_port=self._ibv_port_right,
            hololink_channel=self._hololink_channel_right,
            device=self._camera_right,
        )

        # CRC operators
        compute_crc_left = hololink_module.operators.ComputeCrcOp(
            self,
            name="compute_crc_left",
            frame_size=frame_size,
        )
        self.crc_validator_left = CrcValidationOp(
            self,
            name="crc_validator_left",
            compute_crc_op=compute_crc_left,
            crc_metadata_name="crc",
            camera_name="LEFT",
        )

        compute_crc_right = hololink_module.operators.ComputeCrcOp(
            self,
            name="compute_crc_right",
            frame_size=frame_size,
        )
        self.crc_validator_right = CrcValidationOp(
            self,
            name="crc_validator_right",
            compute_crc_op=compute_crc_right,
            crc_metadata_name="crc",
            camera_name="RIGHT",
        )
        logging.info("CRC validation enabled - checking every frame")

        # Calculate expected frame time based on mode
        # Default mode is 4K@60FPS which correlates to frame time of 16.6ms
        expected_frame_time_ms = 16.6  # Will be adjusted based on actual mode if needed

        profiler_left = InstrumentedTimeProfiler(
            self,
            name="profiler_left",
            operator_name="operator_left",
            calculated_frame_size=frame_size,
        )

        profiler_right = InstrumentedTimeProfiler(
            self,
            name="profiler_right",
            operator_name="operator_right",
            calculated_frame_size=frame_size,
        )

        bayer_format = self._camera_left.bayer_format()
        assert bayer_format == self._camera_right.bayer_format()
        pixel_format = self._camera_left.pixel_format()
        assert pixel_format == self._camera_right.pixel_format()

        image_processor_left = hololink_module.operators.ImageProcessorOp(
            self,
            name="image_processor_left",
            optical_black=50,  # Optical black value for imx274
            bayer_format=bayer_format.value,
            pixel_format=pixel_format.value,
        )
        image_processor_right = hololink_module.operators.ImageProcessorOp(
            self,
            name="image_processor_right",
            optical_black=50,  # Optical black value for imx274
            bayer_format=bayer_format.value,
            pixel_format=pixel_format.value,
        )

        rgba_components_per_pixel = 4
        bayer_pool_left = holoscan.resources.BlockMemoryPool(
            self,
            name="bayer_pool_left",
            storage_type=1,  # device memory
            block_size=self._camera_left._width
            * rgba_components_per_pixel
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera_left._height,
            num_blocks=4,
        )
        bayer_pool_right = holoscan.resources.BlockMemoryPool(
            self,
            name="bayer_pool_right",
            storage_type=1,  # device memory
            block_size=self._camera_right._width
            * rgba_components_per_pixel
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera_right._height,
            num_blocks=4,
        )

        demosaic_left = holoscan.operators.BayerDemosaicOp(
            self,
            name="demosaic_left",
            pool=bayer_pool_left,
            generate_alpha=True,
            alpha_value=65535,
            bayer_grid_pos=bayer_format.value,
            interpolation_mode=0,
            in_tensor_name="left",
            out_tensor_name="left",
        )
        demosaic_right = holoscan.operators.BayerDemosaicOp(
            self,
            name="demosaic_right",
            pool=bayer_pool_right,
            generate_alpha=True,
            alpha_value=65535,
            bayer_grid_pos=bayer_format.value,
            interpolation_mode=0,
            in_tensor_name="right",
            out_tensor_name="right",
        )

        # Set up visualizer with side-by-side stereo view
        left_spec = holoscan.operators.HolovizOp.InputSpec(
            "left", holoscan.operators.HolovizOp.InputType.COLOR
        )
        left_spec_view = holoscan.operators.HolovizOp.InputSpec.View()
        left_spec_view.offset_x = 0
        left_spec_view.offset_y = 0
        left_spec_view.width = 0.5
        left_spec_view.height = 1
        left_spec.views = [left_spec_view]

        right_spec = holoscan.operators.HolovizOp.InputSpec(
            "right", holoscan.operators.HolovizOp.InputType.COLOR
        )
        right_spec_view = holoscan.operators.HolovizOp.InputSpec.View()
        right_spec_view.offset_x = 0.5
        right_spec_view.offset_y = 0
        right_spec_view.width = 0.5
        right_spec_view.height = 1
        right_spec.views = [right_spec_view]

        window_height = 200
        window_width = 600  # for the pair
        visualizer = holoscan.operators.HolovizOp(
            self,
            name="holoviz",
            fullscreen=False,
            headless=self._headless,
            framebuffer_srgb=True,
            tensors=[left_spec, right_spec],
            height=window_height,
            width=window_width,
            window_title="Stereo IMX274 Frame Validation Test",
            enable_camera_pose_output=True,
            camera_pose_output_type="extrinsics_model",
        )

        monitor_left = MonitorOperator(
            self,
            name="monitor_left",
            recorder_queue=self._recorder_queue_left,
            validation_data=self._validation_data_left,
            camera_name="LEFT",
            expected_frame_time_ms=expected_frame_time_ms,
        )

        monitor_right = MonitorOperator(
            self,
            name="monitor_right",
            recorder_queue=self._recorder_queue_right,
            validation_data=self._validation_data_right,
            camera_name="RIGHT",
            expected_frame_time_ms=expected_frame_time_ms,
        )

        watchdog_operator = operators.WatchdogOp(
            self,
            name="watchdog_operator",
            watchdog=self._watchdog,
        )

        # Pipeline flow with CRC validation
        # Left camera pipeline
        self.add_flow(receiver_operator_left, compute_crc_left, {("output", "input")})
        self.add_flow(compute_crc_left, self.crc_validator_left, {("output", "input")})
        self.add_flow(self.crc_validator_left, profiler_left, {("output", "input")})
        self.add_flow(profiler_left, csi_to_bayer_operator_left, {("output", "input")})
        self.add_flow(
            csi_to_bayer_operator_left, image_processor_left, {("output", "input")}
        )
        self.add_flow(image_processor_left, monitor_left, {("output", "input")})
        self.add_flow(monitor_left, demosaic_left, {("output", "receiver")})
        self.add_flow(demosaic_left, visualizer, {("transmitter", "receivers")})

        # Right camera pipeline
        self.add_flow(receiver_operator_right, compute_crc_right, {("output", "input")})
        self.add_flow(
            compute_crc_right, self.crc_validator_right, {("output", "input")}
        )
        self.add_flow(self.crc_validator_right, profiler_right, {("output", "input")})
        self.add_flow(
            profiler_right, csi_to_bayer_operator_right, {("output", "input")}
        )
        self.add_flow(
            csi_to_bayer_operator_right, image_processor_right, {("output", "input")}
        )
        self.add_flow(image_processor_right, monitor_right, {("output", "input")})
        self.add_flow(monitor_right, demosaic_right, {("output", "receiver")})
        self.add_flow(demosaic_right, visualizer, {("transmitter", "receivers")})

        # Add watchdog
        self.add_flow(visualizer, watchdog_operator, {("camera_pose_output", "input")})


# Get available InfiniBand devices
sys_ibv_name_left, sys_ibv_name_right = (
    hololink_module.infiniband_devices() + [None, None]
)[:2]


def run_stereo_frame_validation_test(
    headless,
    channel_metadata_left,
    ibv_name_left,
    ibv_port_left,
    channel_metadata_right,
    ibv_name_right,
    ibv_port_right,
    camera_mode,
    frame_limit,
    scheduler,
):
    """
    Run the stereo frame validation test.

    Args:
        headless: Run in headless mode
        channel_metadata_left: Left camera channel metadata
        ibv_name_left: Left InfiniBand device name
        ibv_port_left: Left InfiniBand port
        channel_metadata_right: Right camera channel metadata
        ibv_name_right: Right InfiniBand device name
        ibv_port_right: Right InfiniBand port
        camera_mode: Camera mode to use
        frame_limit: Number of frames to capture
        scheduler: Scheduler type to use
    """
    logging.info("Initializing stereo frame validation test")

    # Initialize CUDA
    (cu_result,) = cuda.cuInit(0)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_device_ordinal = 0
    cu_result, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_result, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS

    # Initialize hololink channels
    hololink_channel_left = hololink_module.DataChannel(channel_metadata_left)
    hololink_channel_right = hololink_module.DataChannel(channel_metadata_right)

    # Initialize cameras
    camera_left = CameraWrapper(hololink_channel_left, expander_configuration=0)
    camera_right = CameraWrapper(hololink_channel_right, expander_configuration=1)

    # Thread-safe recorder queues for timestamps
    recorder_queue_left = queue.Queue(maxsize=30_000)
    recorder_queue_right = queue.Queue(maxsize=30_000)

    # Validation data structures
    validation_data_left = FrameValidationData("LEFT")
    validation_data_right = FrameValidationData("RIGHT")

    # Set up watchdog
    with utils.Watchdog(
        "watchdog",
        initial_timeout=[30] * (operators.COLOR_PROFILER_START_FRAME + 2),
        timeout=0.5,
    ) as watchdog:
        # Set up the application
        application = StereoFrameValidationApplication(
            headless,
            cu_context,
            cu_device_ordinal,
            hololink_channel_left,
            ibv_name_left,
            ibv_port_left,
            camera_left,
            hololink_channel_right,
            ibv_name_right,
            ibv_port_right,
            camera_right,
            camera_mode,
            frame_limit,
            recorder_queue_left,
            recorder_queue_right,
            validation_data_left,
            validation_data_right,
            watchdog,
        )

        # Run it
        hololink = hololink_channel_left.hololink()
        assert hololink is hololink_channel_right.hololink()
        hololink.start()

        try:
            assert camera_left._reset_callbacks == 0
            assert camera_right._reset_callbacks == 0
            hololink.reset()
            assert camera_left._reset_callbacks == 1
            assert camera_right._reset_callbacks == 1

            logging.debug("Waiting for PTP sync.")
            if not hololink.ptp_synchronize():
                raise ValueError("Failed to synchronize PTP.")
            else:
                logging.debug("PTP synchronized.")

            camera_left.setup_clock()  # this also sets camera_right's clock
            camera_left.configure(camera_mode)
            camera_left.set_digital_gain_reg(0x4)
            camera_right.configure(camera_mode)
            camera_right.set_digital_gain_reg(0x4)

            # Configure scheduler
            if scheduler == "event":
                app_scheduler = holoscan.schedulers.EventBasedScheduler(
                    application,
                    worker_thread_number=4,
                    name="event_scheduler",
                )
                application.scheduler(app_scheduler)
            elif scheduler == "multithread":
                app_scheduler = holoscan.schedulers.MultiThreadScheduler(
                    application,
                    worker_thread_number=4,
                    name="multithread_scheduler",
                )
                application.scheduler(app_scheduler)
            elif scheduler == "greedy":
                app_scheduler = holoscan.schedulers.GreedyScheduler(
                    application,
                    name="greedy_scheduler",
                )
                application.scheduler(app_scheduler)
            elif scheduler == "default":
                # Use the default scheduler
                pass
            else:
                raise Exception(f"Unexpected scheduler type: {scheduler}")

            logging.info("Running application")
            application.run()
        finally:
            hololink.stop()

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS

    # Verify CRCs match
    logging.info("=" * 80)
    logging.info("VERIFYING CRC RESULTS")
    logging.info("=" * 80)

    crcs_left = application.crc_validator_left.get_crcs()
    crcs_right = application.crc_validator_right.get_crcs()
    crc_errors_left = application.crc_validator_left.get_crc_errors()
    crc_errors_right = application.crc_validator_right.get_crc_errors()
    skip_frames = SKIP_INITIAL_FRAMES

    logging.info(f"Left camera captured {len(crcs_left)} frames")
    logging.info(f"Right camera captured {len(crcs_right)} frames")

    # Validate left camera CRCs
    for frame_idx, (received_crc, computed_crc) in enumerate(crcs_left):
        if frame_idx < skip_frames:
            continue
        if received_crc != computed_crc:
            logging.error(
                f"Left CRC mismatch at frame {frame_idx}: "
                f"received={received_crc:#x}, computed={computed_crc:#x}"
            )
            raise AssertionError(
                f"Left CRC mismatch at frame {frame_idx}: "
                f"received={received_crc:#x}, computed={computed_crc:#x}"
            )

    # Validate right camera CRCs
    for frame_idx, (received_crc, computed_crc) in enumerate(crcs_right):
        if frame_idx < skip_frames:
            continue
        if received_crc != computed_crc:
            logging.error(
                f"Right CRC mismatch at frame {frame_idx}: "
                f"received={received_crc:#x}, computed={computed_crc:#x}"
            )
            raise AssertionError(
                f"Right CRC mismatch at frame {frame_idx}: "
                f"received={received_crc:#x}, computed={computed_crc:#x}"
            )

    logging.info(
        f"✓ All CRCs match! (validated {len(crcs_left) - skip_frames} left, "
        f"{len(crcs_right) - skip_frames} right frames)"
    )
    logging.info(f"Left CRC errors: {crc_errors_left}")
    logging.info(f"Right CRC errors: {crc_errors_right}")

    # Verify frame validation results
    logging.info("=" * 80)
    logging.info("FRAME VALIDATION RESULTS")
    logging.info("=" * 80)

    logging.info(f"Left camera frame count: {validation_data_left.frame_count}")
    logging.info(
        f"Left camera frame order errors: {validation_data_left.frame_order_errors}"
    )
    logging.info(
        f"Left camera frame size errors: {validation_data_left.frame_size_errors}"
    )
    logging.info(
        f"Left camera timestamp errors: {validation_data_left.timestamp_errors}"
    )

    logging.info(f"Right camera frame count: {validation_data_right.frame_count}")
    logging.info(
        f"Right camera frame order errors: {validation_data_right.frame_order_errors}"
    )
    logging.info(
        f"Right camera frame size errors: {validation_data_right.frame_size_errors}"
    )
    logging.info(
        f"Right camera timestamp errors: {validation_data_right.timestamp_errors}"
    )

    # Report frame validation issues as warnings (not failures)
    if validation_data_left.frame_order_errors > 0:
        logging.warning(
            f"Left camera had {validation_data_left.frame_order_errors} frame order errors "
            "(informational only, not failing test)"
        )

    if validation_data_right.frame_order_errors > 0:
        logging.warning(
            f"Right camera had {validation_data_right.frame_order_errors} frame order errors "
            "(informational only, not failing test)"
        )

    if validation_data_left.frame_size_errors > 0:
        logging.warning(
            f"Left camera had {validation_data_left.frame_size_errors} frame size errors "
            "(informational only, not failing test)"
        )

    if validation_data_right.frame_size_errors > 0:
        logging.warning(
            f"Right camera had {validation_data_right.frame_size_errors} frame size errors "
            "(informational only, not failing test)"
        )

    if validation_data_left.timestamp_errors > 0:
        logging.warning(
            f"Left camera had {validation_data_left.timestamp_errors} timestamp errors "
            "(informational only, not failing test)"
        )

    if validation_data_right.timestamp_errors > 0:
        logging.warning(
            f"Right camera had {validation_data_right.timestamp_errors} timestamp errors "
            "(informational only, not failing test)"
        )

    # ONLY assert on CRC errors - this is what causes test failure
    logging.info("=" * 80)
    logging.info("CRC VALIDATION CHECK (FAIL CONDITION)")
    logging.info("=" * 80)

    assert crc_errors_left == 0, f"Left camera had {crc_errors_left} CRC errors"
    assert crc_errors_right == 0, f"Right camera had {crc_errors_right} CRC errors"

    logging.info("=" * 80)
    logging.info("✓ CRC VALIDATION PASSED - No CRC errors detected")
    logging.info("=" * 80)


@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode",
    [
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_3840X2160_60FPS,
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
    ],
)
@pytest.mark.parametrize(
    "frame_limit",
    [100],
)
@pytest.mark.parametrize(
    "ibv_name_left, ibv_name_right",
    [
        (sys_ibv_name_left, sys_ibv_name_right),
    ],
)
@pytest.mark.parametrize(
    "hololink_left, hololink_right",
    [
        ("192.168.0.2", "192.168.0.3"),
    ],
)
def test_stereo_imx274_frame_validation(
    camera_mode,
    frame_limit,
    headless,
    hololink_left,
    hololink_right,
    scheduler,
    ibv_name_left,
    ibv_name_right,
):
    """
    Test stereo IMX274 frame validation.

    This test validates CRC integrity for both cameras and reports other metrics:

    FAIL CONDITIONS (test fails if any occur):
    - CRC mismatches on left camera
    - CRC mismatches on right camera

    INFORMATIONAL (logged as warnings, does not fail test):
    - Frame ordering issues (dropped or out-of-order frames)
    - Frame size inconsistencies
    - Timestamp gaps or jitter

    The test prioritizes CRC validation as the critical failure condition
    since frame drops and timing issues can occur under system load but
    CRC mismatches indicate data corruption.
    """
    # Get handles to data sources
    channel_metadata_left = hololink_module.Enumerator.find_channel(
        channel_ip=hololink_left
    )
    channel_metadata_right = hololink_module.Enumerator.find_channel(
        channel_ip=hololink_right
    )
    ibv_port_left, ibv_port_right = 1, 1

    run_stereo_frame_validation_test(
        headless,
        channel_metadata_left,
        ibv_name_left,
        ibv_port_left,
        channel_metadata_right,
        ibv_name_right,
        ibv_port_right,
        camera_mode,
        frame_limit,
        scheduler,
    )
