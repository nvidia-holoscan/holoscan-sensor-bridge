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

# Stereo IMX274 sensors via two independent RoceReceiverOp legs (one QSFP port and one
# DataChannel for each leg), displayed side by side in one Holoviz window, with per-frame
# validation for each leg. HSDK5 port of the v2.7 (HSDK4) stereo_imx274_frame_validation.py;
# it keeps the same components, in each leg:
#
#   RoceReceiverOp -> InstrumentedTimeProfiler -> CsiToBayerOp -> ImageProcessorOp ->
#     BayerDemosaicOp -> HolovizOp (left or right half)
#   RoceReceiverOp -> ComputeCrcOp (GPU CRC, nvCOMP) -> CheckCrcOp (CRC check, metadata only)
#   ImageProcessorOp (metadata) -> MonitorOperator (record_times(), validate_frame())
#
# - ComputeCrcOp / CheckCrcOp are hololink's C++ nvCOMP 5 operators (CRC-32/JAMCRC on the GPU,
#   the FPGA's own format). Each leg has its own hl.CrcChannel: ComputeCrcOp puts each frame's
#   CRC in it under its frame number, and CheckCrcOp takes it. In v2.7 the chain was
#   receiver -> ComputeCrcOp -> CheckCrcOp -> profiler. In HSDK5 a tensor that one operator
#   forwards cannot be forwarded again by a C++ operator (RUNTIME_PORT_BINDING_UNAVAILABLE), so
#   ComputeCrcOp is a side branch from the receiver, CheckCrcOp takes ComputeCrcOp's metadata,
#   and the profiler stays in the main chain.
# - CrcValidationOp keeps the v2.7 check_crc(), get_crc_errors() and get_crcs() methods and the
#   15-frame start-up skip. In v2.7 it was a Python subclass of CheckCrcOp; HSDK5 operators
#   cannot be subclassed from Python, so it is a plain Python class and CheckCrcOp calls its
#   check_crc() through hl.CrcChannel.
# - InstrumentedTimeProfiler, MonitorOperator, record_times(), validate_frame(),
#   print_crc_results() and print_performance_report() are the v2.7 functions, with the v2.7
#   [LEFT]/[RIGHT] labels. HSDK5 metadata is a fixed CaptureMetadata struct, so the values that
#   v2.7 added to the metadata dictionary are kept per leg and frame number in _LegState.
# - HSDK5 HolovizOp has no camera_pose_output port; in v2.7 the monitor sat between the image
#   processor and the demosaic. A Python operator cannot pass the image on to BayerDemosaicOp
#   in HSDK5, so MonitorOperator reads the ImageProcessorOp metadata output instead. The
#   "complete" timestamp is the end of image processing, as in v2.7.
# - --frame-limit is the frame_limit parameter of each leg's receiver: the first receiver that
#   reaches the limit ends the run. HSDK5 has no CountCondition; this replaces the two
#   CountConditions of the old app.
# - --headless leaves out the BayerDemosaicOp and HolovizOp branch.
#
# Usage:
#   stereo_imx274_frame_validation.py [OPTIONS]
#
# Options:
#   --hololink-left IP      IP address of the left-leg Hololink board/NIC (default: 192.168.0.2)
#   --hololink-right IP     IP address of the right-leg Hololink board/NIC (default: 192.168.0.3)
#   --camera-mode N          IMX274 mode (default: 0 = 3840x2160@60fps)
#   --ibv-name-left NAME     Left-leg IBV device (default: the device on the left leg's network interface)
#   --ibv-port-left N        Left-leg IBV device port (default: 1)
#   --ibv-name-right NAME    Right-leg IBV device (default: the device on the right leg's network interface)
#   --ibv-port-right N       Right-leg IBV device port (default: 1)
#   --expander-configuration-left N   Left camera I2C expander configuration (default: 0)
#   --expander-configuration-right N  Right camera I2C expander configuration (default: 1)
#   --pattern N              Configure a test pattern (0-11) instead of live video
#   --crc-frame-check N      0 = CRC check disabled, 1 = check every frame (default)
#   --skip-reset             Don't call reset on the hololink device
#   --window-height N       Displayed window height (default: 2160 // 8)
#   --window-width N        Displayed window width (default: 3840 // 3)
#   --title TITLE            Window title (default: "Holoviz Stereo - <left IP> / <right IP>")
#   --frame-limit N         Number of frames to capture per sensor (default: unlimited)
#   --headless               Run without the display branch
#   --fullscreen             Run in fullscreen mode
#   --log-level N            Python logging level (default: 20 = INFO)

import argparse
import datetime
import logging
import math
import queue
import sys
import threading
from datetime import timedelta

import capture_metadata
import cupy as cp
import holoscan as hs
import holoscan_vision as hv
import holoscan_visualization as hz  # noqa: F401 -- registers "holoscan.visualization.HolovizOp"
from cuda.bindings import driver as cuda_driver

import hololink as hl

MS_PER_SEC = 1000.0
US_PER_SEC = 1000.0 * MS_PER_SEC
NS_PER_SEC = 1000.0 * US_PER_SEC
SEC_PER_NS = 1.0 / NS_PER_SEC

# Shared lock for thread-safe access to the recorder queues' internal deques
validate_frame_lock = threading.Lock()

_BAYER_FORMAT_NAMES = ["bggr", "rggb", "gbrg", "grbg"]
_BAYER_FORMAT_VALUES = {name: index for index, name in enumerate(_BAYER_FORMAT_NAMES)}

# Matches hololink::operators::kMetadataPairingWindow (capture_metadata_utils.hpp): the
# receiver stamps both ports with an identical capture_time, so this bounded-window
# OnAligned+Exact{} pairs them precisely.
_METADATA_PAIRING_WINDOW = timedelta(microseconds=500)


class CrcValidationOp:
    """
    Custom CRC checker that validates computed CRC against received CRC
    and tracks errors.

    hl.CrcChannel calls check_crc() from the C++ CheckCrcOp for every checked frame.
    """

    def __init__(self, camera_name="", skip_initial_frames=15):
        self._lock = threading.Lock()
        self._camera_name = camera_name
        self._frame_count = 0
        self._crc_errors = 0
        self._crcs = []  # list of (received_crc, computed_crc)
        self._skip_initial_frames = skip_initial_frames

    def check_crc(self, computed_crc, received_crc, frame_number):
        """
        This method is called by the C++ CheckCrcOp (through hl.CrcChannel).
        It receives the computed CRC and the CRC from the frame metadata.
        """
        with self._lock:
            self._frame_count += 1

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

    def get_crc_errors(self):
        with self._lock:
            return self._crc_errors

    def get_crcs(self):
        with self._lock:
            return list(self._crcs)


class _LegState:
    """Per-leg values that v2.7 kept in the metadata dictionary.

    InstrumentedTimeProfiler adds a record for each frame; MonitorOperator removes it.
    """

    legs = {}

    def __init__(
        self, camera_name, crc_validator, calculated_frame_size, recorder_queue, fps
    ):
        self.camera_name = camera_name
        self.crc_validator = crc_validator
        self.calculated_frame_size = calculated_frame_size
        self.recorder_queue = recorder_queue
        self.expected_fps = fps
        self.lock = threading.Lock()
        self.records = {}
        _LegState.legs[camera_name] = self

    def put(self, frame_number, record):
        with self.lock:
            self.records[frame_number] = record
            # Frames that do not get to the monitor must not stay here.
            if len(self.records) > 64:
                del self.records[min(self.records)]

    def pop(self, frame_number):
        with self.lock:
            return self.records.pop(frame_number, {})


# The purpose of this function is to parse the recorder queue values, extract the values that needed for:
# 1. checking frame number of current and last frame
# 2. checking frame actual size compared to calculated size
# 3. checking timestamp consistency between current and last frame
# 4. checking for recorded CRC errors
def validate_frame(recorder_queue, camera_name="", expected_fps=60):

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

        # Calculate frame time threshold dynamically based on expected FPS
        # Frame time = 1000ms / FPS, we allow 2x frame time before calling an error
        expected_frame_time_ms = 1000.0 / expected_fps
        if time_difference_ms > (2 * expected_frame_time_ms):
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
    # metadata is the CaptureMetadata struct (fields <name>_s and <name>_ns)
    # or a dictionary made by save_timestamp().
    if isinstance(metadata, dict):
        s = metadata[f"{name}_s"]
        f = metadata[f"{name}_ns"]
    else:
        s = getattr(metadata, f"{name}_s")
        f = getattr(metadata, f"{name}_ns")
    f *= SEC_PER_NS
    return s + f


def _put_dropping_oldest(recorder_queue, record):
    """Queue a record without blocking: when the queue is full, drop its oldest record."""
    try:
        recorder_queue.put_nowait(record)
    except queue.Full:
        recorder_queue.get_nowait()
        recorder_queue.put_nowait(record)


def record_times(recorder_queue, metadata, frame_record, camera_name="", fps=60):
    #
    now = datetime.datetime.now(datetime.timezone.utc)
    #
    frame_number = metadata.frame_number

    # get frame crc errors
    crc32_errors = frame_record.get("crc_errors", 0)

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
    operator_timestamp_s = get_timestamp(frame_record, "operator_timestamp")

    # complete_timestamp_s is the time when image processing finished.
    complete_timestamp_s = get_timestamp(frame_record, "complete_timestamp")

    # get the frame size and calculated frame size from the frame record
    received_frame_size = frame_record.get("received_frame_size")
    calculated_frame_size = frame_record.get("calculated_frame_size")

    # Thread-safe, and never blocks the monitor (see _put_dropping_oldest)
    _put_dropping_oldest(
        recorder_queue,
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
        ),
    )

    # validate the frame
    validate_frame(recorder_queue, camera_name, fps)


def save_timestamp(metadata, name, timestamp):
    # This method works around the fact that we can't store
    # datetime objects in metadata.
    f, s = math.modf(timestamp.timestamp())
    metadata[f"{name}_s"] = int(s)
    metadata[f"{name}_ns"] = int(f * NS_PER_SEC)


class InstrumentedTimeProfiler(hs.Operator):
    in_tensor = hs.Input("in_tensor", hs.payload.Tensor)
    in_metadata = hs.Input("in_metadata", capture_metadata.CaptureMetadata)
    tensor = hs.Output("tensor", hs.payload.Tensor)
    metadata = hs.Output("metadata", capture_metadata.CaptureMetadata)

    camera_name = hs.Param("")

    def setup(self, spec):
        logging.info("setup")
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
        # What time is it now?
        operator_timestamp = datetime.datetime.now(datetime.timezone.utc)

        tensor_sample = self.in_tensor.receive()
        metadata_sample = self.in_metadata.receive()
        if tensor_sample.is_missing or metadata_sample.is_missing:
            return

        # A received HSDK5 tensor is read-only, and HSDK5 exports it only as versioned DLPack
        # on a real CUDA stream (cupy's default legacy stream is refused with kNotSupported).
        # Import it without a copy on a private non-blocking stream, and synchronize before
        # the array is used.
        with cp.cuda.Stream(non_blocking=True) as stream:
            capsule = tensor_sample.data.__dlpack__(
                stream=stream.ptr,
                max_version=(1, 0),  # Versioned DLPack
                copy=False,  # No pixel-buffer copy
            )
            cp_frame = cp.from_dlpack(capsule)
            stream.synchronize()

        leg = _LegState.legs[self.camera_name]
        # Save the number of bytes in the frame to compare to CSI calculated frame size
        frame_record = {}
        frame_record["received_frame_size"] = cp_frame.nbytes
        frame_record["calculated_frame_size"] = leg.calculated_frame_size

        save_timestamp(frame_record, "operator_timestamp", operator_timestamp)
        leg.put(metadata_sample.data.frame_number, frame_record)

        self.tensor.forward(tensor_sample)
        self.metadata.forward(metadata_sample)


class MonitorOperator(hs.Operator):
    input = hs.Input("input", capture_metadata.CaptureMetadata)

    camera_name = hs.Param("")

    def setup(self, spec):
        logging.info("setup")
        spec.input(self.input, queue_depth=4)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, context):
        # What time is it now?
        complete_timestamp = datetime.datetime.now(datetime.timezone.utc)
        metadata = self.input.receive_data()

        # save the complete timestamp and record the times
        leg = _LegState.legs[self.camera_name]
        frame_record = leg.pop(metadata.frame_number)
        if "operator_timestamp_s" not in frame_record:
            return
        save_timestamp(frame_record, "complete_timestamp", complete_timestamp)
        if leg.crc_validator is not None:
            frame_record["crc_errors"] = leg.crc_validator.get_crc_errors()
        record_times(
            leg.recorder_queue,
            metadata,
            frame_record,
            self.camera_name,
            leg.expected_fps,
        )


class TensorSinkOp(hs.Operator):
    """Consume the image when --headless leaves out the display branch."""

    input = hs.Input("input", hs.schema.Image)

    def setup(self, spec):
        spec.input(self.input, queue_depth=1)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, context):
        self.input.receive_data()


def _cuda_check(result):
    err = result[0]
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        _, description = cuda_driver.cuGetErrorString(err)
        raise RuntimeError(f"CUDA error: {description}")
    if len(result) == 2:
        return result[1]
    return result[1:]


def _camera_fps(camera_mode):
    # Get FPS from camera mode for dynamic frame time validation
    try:
        from hololink.sensors.imx274 import imx274_mode

        return imx274_mode.imx_frame_format[camera_mode].framerate
    except (ImportError, AttributeError, IndexError):
        return 60


def print_performance_report(recorder_queue, camera_name=""):
    """Print performance statistics for a camera."""
    with validate_frame_lock:
        recorder_queue_items = list(recorder_queue.queue)

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


def run(
    hololink_ip_left,
    hololink_ip_right,
    camera_mode,
    ibv_name_left,
    ibv_port_left,
    ibv_name_right,
    ibv_port_right,
    expander_configuration_left,
    expander_configuration_right,
    pattern,
    crc_frame_check,
    skip_reset,
    window_height,
    window_width,
    title,
    frame_limit,
    headless,
    fullscreen,
):
    cuda_device_ordinal = 0
    tensor_names = ["left", "right"]
    ibv_names = [ibv_name_left, ibv_name_right]
    ibv_ports = [ibv_port_left, ibv_port_right]
    expander_configurations = [
        expander_configuration_left,
        expander_configuration_right,
    ]

    logging.info("Initializing.")
    # Get a handle to the GPU
    _cuda_check(cuda_driver.cuInit(0))
    cu_device = _cuda_check(cuda_driver.cuDeviceGet(cuda_device_ordinal))
    cu_context = _cuda_check(cuda_driver.cuDevicePrimaryCtxRetain(cu_device))
    cuda_driver.cuCtxSetCurrent(cu_context)

    # Get a handle to data sources
    logging.info(
        f"Enumerating HSB board at {hololink_ip_left} (left) and {hololink_ip_right} (right)"
    )
    channels = [
        hl.DataChannel(hl.find_channel(hololink_ip_left)),
        hl.DataChannel(hl.find_channel(hololink_ip_right)),
    ]

    hololink = channels[0].hololink()
    if hololink is not channels[1].hololink():
        raise RuntimeError(
            "Both --hololink-left/--hololink-right IPs must resolve to the same physical "
            "Hololink device (two NIC ports on one board), not two independent boards."
        )
    hololink.start()
    if not skip_reset:
        hololink.reset()
    logging.debug("Waiting for PTP sync.")
    if not hololink.ptp_synchronize():
        raise ValueError("Failed to synchronize PTP.")
    else:
        logging.debug("PTP synchronized.")

    # Get a handle to the cameras
    cameras = [
        hl.NativeImx274Sensor(
            channels[i], expander_configuration=expander_configurations[i]
        )
        for i in (0, 1)
    ]
    if not skip_reset:
        cameras[0].setup_clock()  # this also sets the other camera's clock
    for camera in cameras:
        camera.configure(camera_mode)
        camera.set_digital_gain_reg(0x4)
        if pattern is not None:
            camera.test_pattern(pattern)

    geometries = [hl.resolve_generic_geometry(camera) for camera in cameras]
    frame_sizes = [
        geometry["start_byte"]
        + geometry["bytes_per_line"] * geometry["pixel_height"]
        + geometry["trailing_bytes"]
        for geometry in geometries
    ]
    if frame_sizes[0] != frame_sizes[1]:
        raise RuntimeError("Both cameras must have the same frame size")
    logging.info(f"frame_size={frame_sizes[0]}")

    # Get FPS from camera mode for dynamic frame time validation
    camera_fps = _camera_fps(camera_mode)

    # Thread-safe recorder queues for timestamps: the last 30,000 records (about 8 minutes at
    # 60 fps); a longer run drops the oldest, so the final report covers the end of the run
    legs = []
    for name in tensor_names:
        camera_name = name.upper()
        crc_validator = (
            CrcValidationOp(camera_name=camera_name) if crc_frame_check > 0 else None
        )
        legs.append(
            _LegState(
                camera_name,
                crc_validator,
                frame_sizes[0],
                queue.Queue(maxsize=30_000),
                camera_fps,
            )
        )

    graph = hs.Graph("stereo_imx274_frame_validation")

    viz = None
    if not headless:
        # One image layer per camera, tiled side by side -- the layer list must be complete
        # before HolovizOp is constructed.
        layers = []
        for i, camera in enumerate(cameras):
            layers.append(
                {
                    "port_name": f"video_{tensor_names[i]}",
                    "kind": "image",
                    "format": "rgba16_unorm",
                    "max_width": camera.get_width(),
                    "max_height": camera.get_height(),
                    "view": {
                        "offset_x": 0.5 * i,
                        "offset_y": 0.0,
                        "scale_x": 0.5,
                        "scale_y": 1.0,
                    },
                }
            )
        viz = graph.op(
            "holoscan.visualization.HolovizOp",
            "holoviz",
            layers=layers,
            sink={
                "kind": "window",
                "framebuffer_srgb": True,
                "title": title
                or f"Holoviz Stereo - {hololink_ip_left} / {hololink_ip_right}",
                "width": window_width,
                "height": window_height,
                "fullscreen": fullscreen,
                "close_on_escape": True,
            },
            # One frame per received frame.
            trigger={"kind": "layer"},
        )

    # CRC operators using nvCOMP 5.0 (optional)
    # Note: Unlike Linux version which samples every Nth frame,
    # GPU-based CRC is fast enough to check every frame (when crc_frame_check=1)
    if crc_frame_check > 0:
        logging.info("CRC validation enabled (nvCOMP 5.0) - checking every frame")
    else:
        logging.info("CRC validation disabled")

    tensor_output_operators = []
    for i, camera in enumerate(cameras):
        name = tensor_names[i]
        leg = legs[i]
        bayer_format = camera.get_bayer_format()
        pixel_format = camera.get_pixel_format()

        receiver_operator = graph.op(
            hl.operators.RoceReceiverOp,
            f"receiver_{name}",
            data_channel=channels[i],
            frame_context=int(cu_context),
            device_start=camera.start,
            device_stop=camera.stop,
            frame_size=frame_sizes[i],
            ibv_name=ibv_names[i],
            ibv_port=ibv_ports[i],
            frame_limit=frame_limit or 0,
            frame_limit_ends_run=True,
        )

        # The profiler is the operator after the receiver, in the main chain, as in v2.7.
        # Thus MonitorOperator always gets a frame after the profiler has recorded it.
        profiler = graph.op(
            InstrumentedTimeProfiler, f"profiler_{name}", camera_name=leg.camera_name
        )
        graph.add_flow(receiver_operator.tensor, profiler.in_tensor, queue_depth=4)
        graph.add_flow(receiver_operator.metadata, profiler.in_metadata, queue_depth=4)

        if crc_frame_check > 0:
            # ComputeCrcOp is a branch from the receiver: in HSDK5 a tensor that one operator
            # forwards cannot be forwarded again by a C++ operator
            # (RUNTIME_PORT_BINDING_UNAVAILABLE), so its tensor output stays unconnected.
            # CheckCrcOp takes ComputeCrcOp's metadata, so each frame's CRC is in the
            # hl.CrcChannel (keyed by frame number) before CheckCrcOp looks it up.
            crc_channel = hl.CrcChannel(check_crc=leg.crc_validator.check_crc)
            compute_crc = graph.op(
                hl.operators.ComputeCrcOp,
                f"compute_crc_{name}",
                crc_channel=crc_channel,
                frame_size=frame_sizes[i],
                cuda_device_ordinal=cuda_device_ordinal,
            )
            check_crc = graph.op(
                hl.operators.CheckCrcOp,
                f"crc_validator_{name}",
                crc_channel=crc_channel,
            )
            graph.add_flow(
                receiver_operator.tensor, compute_crc.in_tensor, queue_depth=4
            )
            graph.add_flow(
                receiver_operator.metadata, compute_crc.in_metadata, queue_depth=4
            )
            graph.add_flow(
                compute_crc.metadata,
                check_crc.in_metadata,
                queue_depth=4,
                queue_policy=hs.QueuePolicy.DROP_OLDEST,
            )

        csi_to_bayer_operator = graph.op(
            hl.operators.CsiToBayerOp,
            f"csi_to_bayer_{name}",
            cuda_device_ordinal=cuda_device_ordinal,
            geometry=geometries[i],
            bayer_format=bayer_format,
        )
        graph.add_flow(profiler.tensor, csi_to_bayer_operator.in_tensor, queue_depth=4)
        graph.add_flow(
            profiler.metadata, csi_to_bayer_operator.in_metadata, queue_depth=4
        )

        image_processor_operator = graph.op(
            hl.operators.ImageProcessorOp,
            f"image_processor_{name}",
            pixel_format=pixel_format,
            pixel_width=geometries[i]["pixel_width"],
            pixel_height=geometries[i]["pixel_height"],
            bayer_format=_BAYER_FORMAT_VALUES[bayer_format],
            # Optical black value for imx274 is 50
            optical_black=50,
            cuda_device_ordinal=cuda_device_ordinal,
        )
        graph.add_flow(
            csi_to_bayer_operator.tensor,
            image_processor_operator.in_tensor,
            queue_depth=4,
        )
        graph.add_flow(
            csi_to_bayer_operator.metadata,
            image_processor_operator.in_metadata,
            queue_depth=4,
        )

        monitor = graph.op(
            MonitorOperator, f"monitor_{name}", camera_name=leg.camera_name
        )
        graph.add_flow(image_processor_operator.metadata, monitor.input, queue_depth=4)

        if viz is not None:
            demosaic = graph.op(
                hv.operators.BayerDemosaicOp,
                f"demosaic_{name}",
                bayer_grid_pos=bayer_format,
                generate_alpha=True,
                alpha_value=65535,
            )
            graph.add_flow(
                image_processor_operator.tensor, demosaic.input, queue_depth=4
            )
            # Drop old frames so a slow display cannot back-pressure the pipeline.
            graph.add_flow(
                demosaic.output,
                getattr(viz, f"video_{name}"),
                queue_policy=hs.QueuePolicy.DROP_OLDEST,
                queue_depth=1,
            )
        else:
            # ImageProcessorOp fails if its tensor output has no consumer.
            tensor_sink = graph.op(TensorSinkOp, f"tensor_sink_{name}")
            graph.add_flow(
                image_processor_operator.tensor,
                tensor_sink.input,
                queue_policy=hs.QueuePolicy.DROP_OLDEST,
                queue_depth=1,
            )

        tensor_output_operators += [
            receiver_operator.name,
            csi_to_bayer_operator.name,
            image_processor_operator.name,
        ]

    clock_key = graph.add_clock(hs.RealtimeClock, "clock")
    graph.set_default_clock(clock_key)

    options = hs.CompileOptions()
    for operator_path in tensor_output_operators:
        options.deployment.bind_tensor_output_device(
            operator_path=operator_path,
            output_port="tensor",
            device=hs.execution.DeviceId(cuda_device_ordinal),
        )
    if viz is not None:
        for name in tensor_names:
            options.deployment.bind_tensor_output_device(
                operator_path=f"demosaic_{name}",
                output_port="output",
                device=hs.execution.DeviceId(cuda_device_ordinal),
            )

    plan = hs.compile(graph, options)
    if not plan.ok:
        print("Graph compilation failed:", file=sys.stderr)
        for diag in plan.diagnostics:
            print(f"  {diag}", file=sys.stderr)
        hololink.stop()
        _cuda_check(cuda_driver.cuDevicePrimaryCtxRelease(cu_device))
        return 1

    # Run it.
    logging.info("Calling run")
    with hs.run_async(plan) as session:
        # With --frame-limit, the receivers end the run after frame_limit frames. The wait
        # runs on a thread, so that Ctrl+C can stop the run cleanly.
        waiter = threading.Thread(target=session.wait, daemon=True)
        waiter.start()
        try:
            while waiter.is_alive():
                waiter.join(timeout=0.5)
        except KeyboardInterrupt:
            session.request_stop()
            waiter.join()
    hololink.stop()

    _cuda_check(cuda_driver.cuDevicePrimaryCtxRelease(cu_device))

    # Report CRC validation results (if enabled)
    if crc_frame_check > 0:
        for leg in legs:
            print_crc_results(
                leg.crc_validator.get_crcs(),
                leg.crc_validator.get_crc_errors(),
                leg.camera_name,
            )
    else:
        logging.info("CRC validation was disabled (--crc-frame-check=0)")

    # Report stats at the end of the application for both cameras
    for leg in legs:
        print_performance_report(leg.recorder_queue, leg.camera_name)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera-mode",
        type=int,
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
        "--hololink-left",
        default="192.168.0.2",
        help="IP address of the left-leg Hololink board/NIC",
    )
    parser.add_argument(
        "--hololink-right",
        default="192.168.0.3",
        help="IP address of the right-leg Hololink board/NIC",
    )
    parser.add_argument(
        "--log-level", type=int, default=20, help="Python logging level to display"
    )
    parser.add_argument(
        "--ibv-name-left",
        default="",
        help="Left-leg IBV device to use "
        "(default: the device on the left leg's network interface)",
    )
    parser.add_argument(
        "--ibv-port-left",
        type=int,
        default=1,
        help="Left-leg IBV device port (with --ibv-name-left)",
    )
    parser.add_argument(
        "--ibv-name-right",
        default="",
        help="Right-leg IBV device to use "
        "(default: the device on the right leg's network interface)",
    )
    parser.add_argument(
        "--ibv-port-right",
        type=int,
        default=1,
        help="Right-leg IBV device port (with --ibv-name-right)",
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
        default=None,
        help="Set the window title",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    return run(
        args.hololink_left,
        args.hololink_right,
        args.camera_mode,
        args.ibv_name_left,
        args.ibv_port_left,
        args.ibv_name_right,
        args.ibv_port_right,
        args.expander_configuration_left,
        args.expander_configuration_right,
        args.pattern,
        args.crc_frame_check,
        args.skip_reset,
        args.window_height,
        args.window_width,
        args.title,
        args.frame_limit,
        args.headless,
        args.fullscreen,
    )


if __name__ == "__main__":
    sys.exit(main())
