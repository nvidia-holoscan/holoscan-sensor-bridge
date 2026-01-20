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
        camera_name="",
        skip_initial_frames=15,
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
            logging.warning(
                f"[{self._camera_name}] Frame {self._frame_count}: CRC mismatch! "
                f"computed={computed_crc:#x}, received={received_crc:#x}"
            )
        else:
            logging.debug(
                f"[{self._camera_name}] Frame {self._frame_count}: CRC match "
                f"({computed_crc:#x})"
            )

        self._crcs.append((received_crc, computed_crc))

        # Store error count in metadata for validation operator
        self.metadata["crc_errors"] = self._crc_errors

    def get_crc_errors(self):
        return self._crc_errors

    def get_crcs(self):
        return self._crcs


validate_frame_lock = threading.Lock()


# The purpose of this function is to parse the recorder queue values, extract the values that needed for:
# 1. checking frame number of current and last frame
# 2. checking frame actual size compared to calculated size
# 3. checking timestamp consistency between current and last frame
# 4. checking for recorded CRC errors
def validate_frame(recorder_queue, camera_name=""):

    if recorder_queue.qsize() > 1:
        # slice out only the relevant data from the recorder queue for frame validation
        # put a lock for thread safety
        with validate_frame_lock:
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
                f"[{camera_name}] Frame number is not in order, last frame number={prev_frame_number}, current frame number={frame_number}"
            )

        # check if the frame size is the same as the calculated frame size
        if received_frame_size != calculated_frame_size:
            logging.info(
                f"[{camera_name}] Frame size is not the same as the calculated frame size, received frame size={received_frame_size}, calculated frame size={calculated_frame_size}"
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
            f"[{camera_name}] Time difference between current and last frame is {time_difference} ms"
        )

        # This is an arbitrary number of millisconds that aligns to IMX274 default mode set in the application.
        # Default mode is 4K@60FPS which correlates to frame time of 16.6ms.
        # In this example we allow 2 x frame time before calling an error.
        if time_difference_ms > (2 * 16.6):
            logging.info(
                f"[{camera_name}] Frame timestamp mismatch, last frame timestamp={prev_complete_timestamp_s}, current frame timestamp={complete_timestamp_s},Diff[ms]={time_difference_ms}"
            )

        # check for CRC errors
        if crc32_errors is not None and prev_crc32_errors is not None:
            if crc32_errors > prev_crc32_errors:  # print only on new error
                logging.info(
                    f"[{camera_name}] CRC32 errors found so far: {crc32_errors}"
                )


def print_crc_results(crcs, crc_errors, camera_name=""):
    """
    Print detailed CRC validation results for debugging.

    Args:
        crcs: List of tuples (received_crc, computed_crc) for each frame
        crc_errors: Total number of CRC errors
        camera_name: Name of the camera for labeling
    """
    total_frames = len(crcs)

    # Print summary at info level
    if total_frames > 0:
        success_rate = ((total_frames - crc_errors) / total_frames) * 100
        logging.info(
            f"[{camera_name}] CRC Validation: {total_frames} frames, {crc_errors} errors, {success_rate:.2f}% success"
        )

    logging.debug("\n" + "=" * 60)
    logging.debug(f"CRC VALIDATION RESULTS (nvCOMP 5.0) - {camera_name}")
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


def record_times(recorder_queue, metadata, camera_name=""):
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

    # get the frame aize and calculated frame size from the metadata
    received_frame_size = metadata.get("received_frame_size")
    calculated_frame_size = metadata.get("calculated_frame_size")
    # complete_timestamp_s is the time when visualization finished.

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
    validate_frame(recorder_queue, camera_name)


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

        save_timestamp(self.metadata, "operator_timestamp", operator_timestamp)
        op_output.emit({"": cp_frame}, "output")


class MonitorOperator(holoscan.core.Operator):
    def __init__(
        self,
        *args,
        recorder_queue=None,
        camera_name="",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._recorder_queue = recorder_queue
        self._camera_name = camera_name

    def setup(self, spec):
        logging.info("setup")
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        # What time is it now?
        complete_timestamp = datetime.datetime.now(datetime.UTC)
        in_message = op_input.receive("input")

        # save the complete timestamp and record the times
        save_timestamp(self.metadata, "complete_timestamp", complete_timestamp)
        record_times(self._recorder_queue, self.metadata, self._camera_name)

        # Pass through the data
        op_output.emit(in_message, "output")


class HoloscanApplication(holoscan.core.Application):
    def __init__(
        self,
        headless,
        fullscreen,
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
        crc_frame_check,
        window_height,
        window_width,
        window_title,
    ):
        logging.info("__init__")
        super().__init__()
        self._headless = headless
        self._fullscreen = fullscreen
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
        self._crc_frame_check = crc_frame_check
        self._window_height = window_height
        self._window_width = window_width
        self._window_title = window_title

        # These are HSDK controls-- because we have stereo
        # camera paths going into the same visualizer, don't
        # raise an error when each path present metadata
        # with the same names.  Because we don't use that metadata,
        # it's easiest to just ignore new items with the same
        # names as existing items.
        self.is_metadata_enabled = True
        self.metadata_policy = holoscan.core.MetadataPolicy.REJECT

        # CRC validators will be set in compose if enabled
        self.crc_validator_left = None
        self.crc_validator_right = None

    def compose(self):
        logging.info("compose")
        if self._frame_limit:
            self._count_left = holoscan.conditions.CountCondition(
                self,
                name="count_left",
                count=self._frame_limit,
            )
            condition_left = self._count_left
            self._count_right = holoscan.conditions.CountCondition(
                self,
                name="count_right",
                count=self._frame_limit,
            )
            condition_right = self._count_right
        else:
            self._ok_left = holoscan.conditions.BooleanCondition(
                self, name="ok_left", enable_tick=True
            )
            condition_left = self._ok_left
            self._ok_right = holoscan.conditions.BooleanCondition(
                self, name="ok_right", enable_tick=True
            )
            condition_right = self._ok_right
        self._camera_left.set_mode(self._camera_mode)
        self._camera_right.set_mode(self._camera_mode)

        # Separate memory pools for each camera to avoid contention with EventBasedScheduler
        csi_to_bayer_pool_left = holoscan.resources.BlockMemoryPool(
            self,
            name="csi_to_bayer_pool_left",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=self._camera_left._width
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera_left._height,
            num_blocks=4,
        )
        csi_to_bayer_pool_right = holoscan.resources.BlockMemoryPool(
            self,
            name="csi_to_bayer_pool_right",
            # storage_type of 1 is device memory
            storage_type=1,
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
        logging.info(f"{frame_size=}")

        frame_context = self._cuda_context
        receiver_operator_left = hololink_module.operators.RoceReceiverOp(
            self,
            condition_left,
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
            condition_right,
            name="receiver_right",
            frame_size=frame_size,
            frame_context=frame_context,
            ibv_name=self._ibv_name_right,
            ibv_port=self._ibv_port_right,
            hololink_channel=self._hololink_channel_right,
            device=self._camera_right,
        )

        # CRC operators using nvCOMP 5.0 (optional)
        # Note: Unlike Linux version which samples every Nth frame,
        # GPU-based CRC is fast enough to check every frame (when crc_frame_check=1)
        if self._crc_frame_check > 0:
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
            logging.info("CRC validation enabled (nvCOMP 5.0) - checking every frame")
        else:
            logging.info("CRC validation disabled")

        profiler_left = InstrumentedTimeProfiler(
            self,
            name="profiler_left",
            operator_name="operator_left",
            calculated_frame_size=frame_size,
            crc_frame_check=self._crc_frame_check,
        )

        profiler_right = InstrumentedTimeProfiler(
            self,
            name="profiler_right",
            operator_name="operator_right",
            calculated_frame_size=frame_size,
            crc_frame_check=self._crc_frame_check,
        )

        bayer_format = self._camera_left.bayer_format()
        assert bayer_format == self._camera_right.bayer_format()
        pixel_format = self._camera_left.pixel_format()
        assert pixel_format == self._camera_right.pixel_format()
        image_processor_left = hololink_module.operators.ImageProcessorOp(
            self,
            name="image_processor_left",
            # Optical black value for imx274 is 50
            optical_black=50,
            bayer_format=bayer_format.value,
            pixel_format=pixel_format.value,
        )
        image_processor_right = hololink_module.operators.ImageProcessorOp(
            self,
            name="image_processor_right",
            # Optical black value for imx274 is 50
            optical_black=50,
            bayer_format=bayer_format.value,
            pixel_format=pixel_format.value,
        )

        rgba_components_per_pixel = 4
        # Separate memory pools for each camera to avoid contention with EventBasedScheduler
        bayer_pool_left = holoscan.resources.BlockMemoryPool(
            self,
            name="bayer_pool_left",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=self._camera_left._width
            * rgba_components_per_pixel
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera_left._height,
            num_blocks=4,
        )
        bayer_pool_right = holoscan.resources.BlockMemoryPool(
            self,
            name="bayer_pool_right",
            # storage_type of 1 is device memory
            storage_type=1,
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

        visualizer = holoscan.operators.HolovizOp(
            self,
            name="holoviz",
            fullscreen=self._fullscreen,
            headless=self._headless,
            framebuffer_srgb=True,
            tensors=[left_spec, right_spec],
            height=self._window_height,
            width=self._window_width,
            window_title=self._window_title,
        )

        monitor_left = MonitorOperator(
            self,
            name="monitor_left",
            recorder_queue=self._recorder_queue_left,
            camera_name="LEFT",
        )

        monitor_right = MonitorOperator(
            self,
            name="monitor_right",
            recorder_queue=self._recorder_queue_right,
            camera_name="RIGHT",
        )

        # Pipeline flow - conditionally include CRC validation
        # Left camera pipeline
        if self._crc_frame_check > 0:
            # Pipeline with CRC validation using nvCOMP 5.0
            self.add_flow(
                receiver_operator_left, compute_crc_left, {("output", "input")}
            )
            self.add_flow(
                compute_crc_left, self.crc_validator_left, {("output", "input")}
            )
            self.add_flow(self.crc_validator_left, profiler_left, {("output", "input")})
        else:
            # Pipeline without CRC validation
            self.add_flow(receiver_operator_left, profiler_left, {("output", "input")})

        self.add_flow(profiler_left, csi_to_bayer_operator_left, {("output", "input")})
        self.add_flow(
            csi_to_bayer_operator_left, image_processor_left, {("output", "input")}
        )
        self.add_flow(image_processor_left, monitor_left, {("output", "input")})
        self.add_flow(monitor_left, demosaic_left, {("output", "receiver")})
        self.add_flow(demosaic_left, visualizer, {("transmitter", "receivers")})

        # Right camera pipeline
        if self._crc_frame_check > 0:
            # Pipeline with CRC validation using nvCOMP 5.0
            self.add_flow(
                receiver_operator_right, compute_crc_right, {("output", "input")}
            )
            self.add_flow(
                compute_crc_right, self.crc_validator_right, {("output", "input")}
            )
            self.add_flow(
                self.crc_validator_right, profiler_right, {("output", "input")}
            )
        else:
            # Pipeline without CRC validation
            self.add_flow(
                receiver_operator_right, profiler_right, {("output", "input")}
            )

        self.add_flow(
            profiler_right, csi_to_bayer_operator_right, {("output", "input")}
        )
        self.add_flow(
            csi_to_bayer_operator_right, image_processor_right, {("output", "input")}
        )
        self.add_flow(image_processor_right, monitor_right, {("output", "input")})
        self.add_flow(monitor_right, demosaic_right, {("output", "receiver")})
        self.add_flow(demosaic_right, visualizer, {("transmitter", "receivers")})

    def _terminate(self, recorded_timestamps_left, recorded_timestamps_right):
        if not self._frame_limit:
            self._ok_left.disable_tick()
            self._ok_right.disable_tick()


def print_performance_report(recorder_queue_items, camera_name=""):
    """Print performance statistics for a camera."""
    if len(recorder_queue_items) < 100:
        logging.info(
            f"[{camera_name}] Not enough frames ({len(recorder_queue_items)}) for performance statistics"
        )
        return

    frame_time_dts = []
    cpu_latency_dts = []
    operator_latency_dts = []
    processing_time_dts = []
    overall_time_dts = []

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
        logging.debug(
            f"[{camera_name}] ** Frame Information  for Frame Number = {frame_number}**"
        )
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

    logging.info(f"\n** PERFORMANCE REPORT [{camera_name}] **")
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
    logging.info("FGPA frame transfer latency (in sec):")
    logging.info(
        f"{'Frame Transfer Latency':<30}{cl_min_time_difference:<15}{cl_max_time_difference:<15}{cl_avg_time_difference:<15}"
    )
    #
    ol_min_time_difference = min(operator_latency_dts)
    ol_max_time_difference = max(operator_latency_dts)
    ol_avg_time_difference = sum(operator_latency_dts) / len(operator_latency_dts)
    logging.info("FGPA to Operator after network operator latency (in sec):")
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
        "--hololink-left",
        default="192.168.0.2",
        help="IP address of left Hololink board",
    )
    parser.add_argument(
        "--hololink-right",
        default="192.168.0.3",
        help="IP address of right Hololink board",
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level to display",
    )
    infiniband_devices = hololink_module.infiniband_devices()
    parser.add_argument(
        "--ibv-name-left",
        default=infiniband_devices[0] if len(infiniband_devices) > 0 else None,
        help="IBV device to use for left camera",
    )
    parser.add_argument(
        "--ibv-port-left",
        type=int,
        default=1,
        help="Port number of IBV device for left camera",
    )
    parser.add_argument(
        "--ibv-name-right",
        default=infiniband_devices[1] if len(infiniband_devices) > 1 else None,
        help="IBV device to use for right camera",
    )
    parser.add_argument(
        "--ibv-port-right",
        type=int,
        default=1,
        help="Port number of IBV device for right camera",
    )
    parser.add_argument(
        "--expander-configuration-left",
        type=int,
        default=0,
        choices=(0, 1),
        help="I2C Expander configuration for left camera",
    )
    parser.add_argument(
        "--expander-configuration-right",
        type=int,
        default=1,
        choices=(0, 1),
        help="I2C Expander configuration for right camera",
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
        help="GPU-based CRC validation using nvCOMP 5.0: 0=disabled, 1=check every frame (default). Note: unlike CPU-based checking, GPU CRC is fast enough to always check every frame.",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=2160 // 8,  # arbitrary default
        help="Set the height of the displayed window",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=3840 // 3,  # arbitrary default (wider for stereo)
        help="Set the width of the displayed window",
    )
    parser.add_argument(
        "--title",
        help="Set the window title",
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
    # Get a handle to data sources
    channel_metadata_left = hololink_module.Enumerator.find_channel(
        channel_ip=args.hololink_left
    )
    logging.info(f"Left channel: {channel_metadata_left=}")
    hololink_channel_left = hololink_module.DataChannel(channel_metadata_left)
    channel_metadata_right = hololink_module.Enumerator.find_channel(
        channel_ip=args.hololink_right
    )
    logging.info(f"Right channel: {channel_metadata_right=}")
    hololink_channel_right = hololink_module.DataChannel(channel_metadata_right)
    # Get a handle to the cameras
    camera_left = hololink_module.sensors.imx274.dual_imx274.Imx274Cam(
        hololink_channel_left, expander_configuration=args.expander_configuration_left
    )
    camera_right = hololink_module.sensors.imx274.dual_imx274.Imx274Cam(
        hololink_channel_right, expander_configuration=args.expander_configuration_right
    )
    camera_mode = hololink_module.sensors.imx274.imx274_mode.Imx274_Mode(
        args.camera_mode
    )

    # Thread-safe recorder queues for timestamps - allows about 10 minutes of queuing for 60fps
    recorder_queue_left = queue.Queue(maxsize=30_000)
    recorder_queue_right = queue.Queue(maxsize=30_000)

    # What title should we use?
    window_title = f"Holoviz Stereo - {args.hololink_left} / {args.hololink_right}"
    if args.title is not None:
        window_title = args.title

    # Set up the application
    application = HoloscanApplication(
        args.headless,
        args.fullscreen,
        cu_context,
        cu_device_ordinal,
        hololink_channel_left,
        args.ibv_name_left,
        args.ibv_port_left,
        camera_left,
        hololink_channel_right,
        args.ibv_name_right,
        args.ibv_port_right,
        camera_right,
        camera_mode,
        args.frame_limit,
        recorder_queue_left,
        recorder_queue_right,
        args.crc_frame_check,
        args.window_height,
        args.window_width,
        window_title,
    )
    application.config(args.configuration)
    # Run it.
    hololink = hololink_channel_left.hololink()
    assert hololink is hololink_channel_right.hololink()
    hololink.start()
    if not args.skip_reset:
        hololink.reset()
    logging.debug("Waiting for PTP sync.")
    if not hololink.ptp_synchronize():
        raise ValueError("Failed to synchronize PTP.")
    else:
        logging.debug("PTP synchronized.")
    if not args.skip_reset:
        camera_left.setup_clock()  # this also sets camera_right's clock
    camera_left.configure(camera_mode)
    camera_left.set_digital_gain_reg(0x4)
    camera_right.configure(camera_mode)
    camera_right.set_digital_gain_reg(0x4)
    if args.pattern is not None:
        camera_left.test_pattern(args.pattern)
        camera_right.test_pattern(args.pattern)

    # Using Greedy Scheduler (default) - may experience CRC mismatch issues with stereo
    scheduler = holoscan.schedulers.GreedyScheduler(
        application,
        name="greedy_scheduler",
    )
    application.scheduler(scheduler)

    logging.info("Calling run")
    application.run()
    hololink.stop()

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS

    # Report CRC validation results (if enabled)
    if args.crc_frame_check > 0:
        if application.crc_validator_left:
            crc_errors_left = application.crc_validator_left.get_crc_errors()
            crcs_left = application.crc_validator_left.get_crcs()
            print_crc_results(crcs_left, crc_errors_left, "LEFT")

        if application.crc_validator_right:
            crc_errors_right = application.crc_validator_right.get_crc_errors()
            crcs_right = application.crc_validator_right.get_crcs()
            print_crc_results(crcs_right, crc_errors_right, "RIGHT")
    else:
        logging.info("CRC validation was disabled (--crc-frame-check=0)")

    # Report stats at the end of the application for both cameras
    # Extract all items from the thread-safe queues for processing
    recorder_queue_left_items = list(recorder_queue_left.queue)
    recorder_queue_right_items = list(recorder_queue_right.queue)

    print_performance_report(recorder_queue_left_items, "LEFT")
    print_performance_report(recorder_queue_right_items, "RIGHT")


if __name__ == "__main__":
    main()
