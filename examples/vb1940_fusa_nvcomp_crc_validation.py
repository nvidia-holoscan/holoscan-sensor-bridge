# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
VB1940 Frame Validation with FUSA and nvCOMP CRC

This script validates VB1940 camera frames using FUSA capture and nvCOMP (NVIDIA Compression Library)
for CRC computation, comparing nvCOMP CRC against camera FPGA-embedded CRC values.

This version uses FUSA CoE (Camera over Ethernet) capture instead of ROCE for data transfer.

CSI layout for nvCOMP (start_byte, full_frame_size, etc.) is computed by
``_vb1940_csi_frame_layout``, using the same rules as ``Vb1940Cam.configure_converter`` in
``vb1940.py``, with line packing and 64-byte alignment taken from the capture converter
(``FusaCoeCaptureOp``) rather than duplicated RAW width formulas.

nvCOMP computes CRC32/JAMCRC using GPU acceleration, which is mathematically
equivalent to streaming CRC32/JAMCRC when processing the same data.

Pipeline: ``ComputeCrcOp`` / ``CheckCrcOp`` on the **full CSI** buffer run before unpack
and ISP so nvCOMP and the FPGA CRC use the same byte range. ``RecordMetadataOp`` (after
demosaic) snapshots ``crc`` and ``check_crc`` for the summary; metadata propagates through
the graph.
"""

import argparse
import ctypes
import datetime
import logging
import math
import queue
import threading
from types import SimpleNamespace

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


class PassThroughOperator(holoscan.core.Operator):
    """Tensor pass-through with optional input/output tensor names."""

    def __init__(self, *args, in_tensor_name="", out_tensor_name="", **kwargs):
        super().__init__(*args, **kwargs)
        self._in_tensor_name = in_tensor_name
        self._out_tensor_name = out_tensor_name

    def setup(self, spec):
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("input")
        if not in_message:
            return
        tensor = in_message.get(self._in_tensor_name)
        # Propagate entity metadata (e.g. crc) for downstream operators and RecordMetadataOp.
        if hasattr(in_message, "metadata") and in_message.metadata:
            for key, value in in_message.metadata.items():
                self.metadata[key] = value
        op_output.emit({self._out_tensor_name: tensor}, "output")


class RecordMetadataOp(PassThroughOperator):
    """Append selected metadata fields per frame for post-run validation."""

    def __init__(self, *args, metadata_names=None, **kwargs):
        super().__init__(*args, **kwargs)
        if metadata_names is None:
            metadata_names = []
        self._metadata_names = list(metadata_names)
        self._record = []
        self._frame_count = 0

    def compute(self, op_input, op_output, context):
        super().compute(op_input, op_output, context)
        metadata = self.metadata
        value = [metadata[name] for name in self._metadata_names]
        self._record.append(value)

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            if self._frame_count < 5 or self._frame_count % 20 == 0:
                camera_crc = metadata.get("crc", 0)
                nvcomp_crc = metadata.get("check_crc", 0)
                match_str = "MATCH" if camera_crc == nvcomp_crc else "MISMATCH"
                logging.debug(
                    f"Frame {self._frame_count}: Camera=0x{camera_crc:08x}, "
                    f"nvCOMP=0x{nvcomp_crc:08x} {match_str}"
                )

        self._frame_count += 1

    def get_record(self):
        return self._record


def _vb1940_csi_frame_layout(camera, converter):
    """Full CSI buffer layout — same steps as ``Vb1940Cam.configure_converter`` (vb1940.py).

    Uses ``converter.transmitted_line_bytes`` / ``received_line_bytes`` (e.g. FusaCoeCaptureOp)
    so packing and 64-byte alignment match the capture path.

    Returns:
        SimpleNamespace: start_byte, bytes_per_line, trailing_bytes, status_line_bytes,
        pixel_data_size, full_frame_size.
    """
    PixelFormat = hololink_module.sensors.csi.PixelFormat
    start_byte = converter.receiver_start_byte()
    pf = camera.pixel_format()
    w = camera.width()
    h = camera.height()
    transmitted_line_bytes = converter.transmitted_line_bytes(pf, w)
    received_line_bytes = converter.received_line_bytes(transmitted_line_bytes)
    status_line_bytes = 0
    if pf == PixelFormat.RAW_8:
        status_line_bytes = int(w)
    elif pf == PixelFormat.RAW_10:
        status_line_bytes = int(w * 10 / 8)
    elif pf == PixelFormat.RAW_12:
        status_line_bytes = int(w * 12 / 8)
    status_line_bytes = converter.received_line_bytes(status_line_bytes)
    start_byte += status_line_bytes
    trailing_bytes = status_line_bytes * 2
    pixel_data_size = received_line_bytes * h
    full_frame_size = start_byte + pixel_data_size + trailing_bytes
    return SimpleNamespace(
        start_byte=start_byte,
        bytes_per_line=received_line_bytes,
        trailing_bytes=trailing_bytes,
        status_line_bytes=status_line_bytes,
        pixel_data_size=pixel_data_size,
        full_frame_size=full_frame_size,
    )


class FullFrameAccessOp(holoscan.core.Operator):
    """Operator that provides access to full CSI frame buffer for nvCOMP CRC computation.

    FUSA outputs a tensor pointing to pixel data (offset by start_byte), but the
    underlying buffer contains the full frame. This operator creates a view of the
    full frame by adjusting the pointer back to the buffer start.

    The FUSA buffer layout:
    [CSI header (start_byte bytes)] [pixel data (pixel_data_size bytes)] [trailing bytes]

    The tensor we receive points to pixel data, so we subtract start_byte to get
    the full buffer start.
    """

    def __init__(self, *args, start_byte=0, full_frame_size=0, **kwargs):
        super().__init__(*args, **kwargs)
        self._start_byte = start_byte
        self._full_frame_size = full_frame_size

    def setup(self, spec):
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("input")
        if not in_message:
            return

        pixel_tensor = cp.asarray(in_message.get(""))

        # Full buffer base address = FUSA pixel tensor base minus header offset (see class docstring).
        pixel_ptr_int = ctypes.cast(pixel_tensor.data.ptr, ctypes.c_void_p).value
        full_buffer_ptr_int = pixel_ptr_int - self._start_byte

        full_buffer_mem = cp.cuda.UnownedMemory(
            full_buffer_ptr_int,
            self._full_frame_size,
            pixel_tensor,  # Pin lifetime of underlying allocation
        )
        full_buffer_memptr = cp.cuda.MemoryPointer(full_buffer_mem, 0)

        full_frame_array = cp.ndarray(
            (self._full_frame_size,), dtype=cp.uint8, memptr=full_buffer_memptr
        )

        op_output.emit({"": full_frame_array}, "output")


def get_timestamp(metadata, name):
    """Return UTC timestamp from ``{name}_s`` / ``{name}_ns`` (Hololink timestamp fields).

    If either component is missing, returns current UTC time (fallback for incomplete metadata).
    """
    s_key = f"{name}_s"
    ns_key = f"{name}_ns"

    if s_key not in metadata or ns_key not in metadata:
        return datetime.datetime.now(datetime.timezone.utc).timestamp()

    s = metadata[s_key]
    f = metadata[ns_key]
    f *= SEC_PER_NS
    return s + f


def validate_frame(recorder_queue, expected_fps=30):
    if recorder_queue.qsize() > 1:
        # slice out only the relevant data from the recorder queue for frame validation
        # use shared lock for thread safety when accessing internal deque
        with recorder_queue_lock:
            internal_q = list(recorder_queue.queue)
            recorder_queue_raw = internal_q[-5:]
            sliced_recorder_queue = [sublist[-5:] for sublist in recorder_queue_raw]
        (
            prev_complete_timestamp_s,
            prev_frame_number,
            prev_crc32_errors,
            _,
            _,
        ) = sliced_recorder_queue[-2]

        (
            complete_timestamp_s,
            frame_number,
            crc32_errors,
            received_frame_size,
            calculated_frame_size,
        ) = sliced_recorder_queue[-1]

        if frame_number != prev_frame_number + 1:
            logging.info(
                f"Frame number is not in order, last frame number={prev_frame_number}, current frame number={frame_number}"
            )

        if received_frame_size != calculated_frame_size:
            logging.info(
                f"Frame size is not the same as the calculated frame size, received frame size={received_frame_size}, calculated frame size={calculated_frame_size}"
            )

        complete_timestamp = datetime.datetime.fromtimestamp(complete_timestamp_s)
        prev_complete_timestamp = datetime.datetime.fromtimestamp(
            prev_complete_timestamp_s
        )
        time_difference = complete_timestamp - prev_complete_timestamp

        time_difference_ms = time_difference.total_seconds() * 1000
        logging.debug(
            f"Time difference between current and last frame is {time_difference_ms} ms"
        )

        # Allow up to 2× nominal frame interval before flagging timestamp gap
        expected_frame_time_ms = 1000.0 / expected_fps
        frame_time_threshold_ms = 2 * expected_frame_time_ms

        if time_difference_ms > frame_time_threshold_ms:
            logging.info(
                f"Frame timestamp mismatch, last frame timestamp={prev_complete_timestamp_s}, current frame timestamp={complete_timestamp_s},Diff[ms]={time_difference_ms}"
            )

        if crc32_errors is not None and prev_crc32_errors is not None:
            if crc32_errors > prev_crc32_errors:
                logging.info(f"CRC32 errors found so far: {crc32_errors}")


def record_times(recorder_queue, metadata, expected_fps=30):
    """Append one timing row to ``recorder_queue`` for latency / validation statistics."""
    now = datetime.datetime.now(datetime.timezone.utc)
    frame_number = metadata.get("frame_number", 0)

    # Optional; reserved if a future path sets cumulative crc_errors in metadata
    crc32_errors = metadata.get("crc_errors", 0)

    # First sensor packet (timestamp) and FPGA metadata packet (metadata) times
    frame_start_s = get_timestamp(metadata, "timestamp")
    frame_end_s = get_timestamp(metadata, "metadata")

    # Host receive time; FUSA often omits this — use metadata time
    if "received_s" in metadata:
        received_timestamp_s = get_timestamp(metadata, "received")
    else:
        received_timestamp_s = frame_end_s

    # Set by InstrumentedTimeProfiler
    if "operator_timestamp_s" in metadata:
        operator_timestamp_s = get_timestamp(metadata, "operator_timestamp")
    else:
        operator_timestamp_s = received_timestamp_s

    # Set by MonitorOperator when the frame leaves Holoviz
    if "complete_timestamp_s" in metadata:
        complete_timestamp_s = get_timestamp(metadata, "complete_timestamp")
    else:
        complete_timestamp_s = now.timestamp()

    received_frame_size = metadata.get("received_frame_size")
    calculated_frame_size = metadata.get("calculated_frame_size")

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
        operator_timestamp = datetime.datetime.now(datetime.timezone.utc)

        in_message = op_input.receive("input")
        cp_frame = cp.asarray(in_message.get(""))

        # Compared in validate_frame() against CSI layout (full-frame byte count)
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
        expected_fps=30,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._recorder_queue = recorder_queue
        self._expected_fps = expected_fps

    def setup(self, spec):
        logging.info("setup")
        spec.input("input")

    def compute(self, op_input, op_output, context):
        complete_timestamp = datetime.datetime.now(datetime.timezone.utc)
        _ = op_input.receive("input")

        save_timestamp(self.metadata, "complete_timestamp", complete_timestamp)
        # Enqueue row for end-of-run latency report and validate_frame()
        record_times(self._recorder_queue, self.metadata, self._expected_fps)


class HoloscanApplication(holoscan.core.Application):
    def __init__(
        self,
        headless,
        fullscreen,
        cuda_context,
        cuda_device_ordinal,
        hololink_channel,
        interface,
        camera,
        camera_mode,
        frame_limit,
        recorder_queue,
        timeout,
    ):
        super().__init__()
        # FUSA capture and CRC fields require pipeline metadata on entities
        self.enable_metadata(True)
        self.metadata_policy = holoscan.core.MetadataPolicy.REJECT

        self._headless = headless
        self._fullscreen = fullscreen
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._hololink_channel = hololink_channel
        self._interface = interface
        self._camera = camera
        self._camera_mode = camera_mode
        self._frame_limit = frame_limit
        self._recorder_queue = recorder_queue
        self._timeout = timeout
        self.record_metadata_op = None
        self.compute_crc_op = None

    def compose(self):
        logging.info("Composing FUSA + nvCOMP CRC validation pipeline")

        if self._frame_limit:
            condition = holoscan.conditions.CountCondition(
                self, name="count", count=self._frame_limit
            )
        else:
            condition = holoscan.conditions.BooleanCondition(
                self, name="ok", enable_tick=True
            )

        self._camera.set_mode(self._camera_mode)

        # Ethernet interface and HSB MAC for FUSA CoE
        metadata = self._hololink_channel.enumeration_metadata()
        interface = self._interface
        if interface is None:
            interface = metadata.get("interface")
        local_ip = metadata.get("interface_address")
        hsb_ip = metadata.get("client_ip_address")
        hsb_mac = metadata.get("mac_id")
        hsb_mac_bytes = list(bytes.fromhex(hsb_mac.replace(":", "")))

        logging.info("Using FUSA CoE Capture:")
        logging.info(f"  Interface: {interface}")
        logging.info(f"  Local IP:  {local_ip}")
        logging.info(f"  HSB IP:    {hsb_ip}")
        logging.info(f"  HSB MAC:   {hsb_mac}")

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

        layout = _vb1940_csi_frame_layout(self._camera, fusa_coe_capture)
        start_byte = layout.start_byte
        full_frame_size = layout.full_frame_size
        pixel_data_size = layout.pixel_data_size
        pixel_width = self._camera.width()
        pixel_height = self._camera.height()
        pixel_format = self._camera.pixel_format()

        logging.debug("CSI layout (_vb1940_csi_frame_layout, same rules as vb1940.py):")
        logging.debug(f"  Pixel dimensions: {pixel_width} × {pixel_height}")
        logging.debug(f"  Pixel format: {pixel_format}")
        logging.debug(f"  Bytes per line (from converter): {layout.bytes_per_line:,}")
        logging.debug(
            f"  Status line bytes (64-byte aligned): {layout.status_line_bytes:,}"
        )
        logging.debug(f"  Start byte (CSI header): {layout.start_byte:,}")
        logging.debug(f"  Trailing bytes: {layout.trailing_bytes:,}")
        logging.debug(f"  Pixel data size: {pixel_data_size:,} bytes")
        logging.debug(f"  Full CSI frame size: {full_frame_size:,} bytes")
        logging.debug(
            f"  FUSA tensor size (pixel data only): {pixel_data_size:,} bytes"
        )
        logging.debug(f"  nvCOMP frame_size (full frame): {full_frame_size:,} bytes")

        logging.debug("Configuring nvCOMP CRC (full CSI frame):")
        logging.debug(f"  Camera mode: {pixel_width}x{pixel_height} {pixel_format}")
        logging.debug(
            f"  Full CSI frame size: {full_frame_size:,} bytes (camera CRC computed on this)"
        )
        logging.debug(
            f"  Pixel data size: {pixel_data_size:,} bytes (FUSA outputs this)"
        )
        logging.debug(
            f"  nvCOMP frame_size: {full_frame_size:,} bytes (full frame via FullFrameAccessOp)"
        )
        logging.info(
            f"nvCOMP CRC will compute on full CSI frame ({full_frame_size:,} bytes) "
            f"to match camera CRC computation."
        )

        self._csi_full_frame_size = full_frame_size
        self._csi_start_byte = start_byte

        # View full CSI buffer for nvCOMP; FUSA tensor is pixel-only (see FullFrameAccessOp)
        full_frame_access = FullFrameAccessOp(
            self,
            name="full_frame_access",
            start_byte=start_byte,
            full_frame_size=full_frame_size,
        )

        # Full-frame CRC before unpack; CheckCrcOp writes check_crc into metadata
        compute_crc = hololink_module.operators.ComputeCrcOp(
            self,
            name="compute_crc",
            frame_size=full_frame_size,
        )
        self.compute_crc_op = compute_crc
        check_crc_raw = hololink_module.operators.CheckCrcOp(
            self,
            name="check_crc_raw",
            compute_crc_op=compute_crc,
            computed_crc_metadata_name="check_crc",
        )

        logging.debug(
            "nvCOMP: ComputeCrcOp -> CheckCrcOp -> profiler -> unpack -> ISP -> demosaic"
        )

        profiler = InstrumentedTimeProfiler(
            self,
            name="profiler",
            calculated_frame_size=full_frame_size,  # Full CSI frame size
            crc_frame_check=1,
        )

        # Convert packed RAW to 16-bit Bayer.
        packed_format_converter_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="packed_format_converter_pool",
            storage_type=1,  # device memory
            block_size=self._camera.width()
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera.height(),
            num_blocks=4,
        )
        packed_format_converter = hololink_module.operators.PackedFormatConverterOp(
            self,
            name="packed_format_converter",
            allocator=packed_format_converter_pool,
        )
        # Vb1940Cam path: full-frame sizing. (FusaCoeCaptureOp alone uses pixel-offset tensors.)
        self._camera.configure_converter(packed_format_converter)
        packed_sz = packed_format_converter.get_frame_size()
        if packed_sz != layout.full_frame_size:
            raise RuntimeError(
                f"CSI frame size mismatch: PackedFormatConverter.get_frame_size()={packed_sz} "
                f"vs layout.full_frame_size={layout.full_frame_size}"
            )

        pixel_format = self._camera.pixel_format()
        bayer_format = self._camera.bayer_format()

        image_processor_operator = hololink_module.operators.ImageProcessorOp(
            self,
            name="image_processor",
            optical_black=0,
            bayer_format=bayer_format.value,
            pixel_format=pixel_format.value,
        )

        rgba_components_per_pixel = 4
        bayer_rgba_size = (
            self._camera.width()
            * rgba_components_per_pixel
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera.height()
        )
        bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="bayer_pool",
            storage_type=1,
            block_size=bayer_rgba_size,
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

        self.record_metadata_op = RecordMetadataOp(
            self,
            name="record_metadata",
            metadata_names=["crc", "check_crc"],
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

        camera_fps = hololink_module.sensors.vb1940.vb1940_mode.vb1940_frame_format[
            self._camera_mode.value
        ].framerate

        monitor = MonitorOperator(
            self,
            name="monitor",
            recorder_queue=self._recorder_queue,
            expected_fps=camera_fps,
        )

        self.add_flow(fusa_coe_capture, full_frame_access, {("output", "input")})
        self.add_flow(full_frame_access, compute_crc, {("output", "input")})
        self.add_flow(compute_crc, check_crc_raw, {("output", "input")})
        self.add_flow(check_crc_raw, profiler, {("output", "input")})
        self.add_flow(profiler, packed_format_converter, {("output", "input")})
        self.add_flow(
            packed_format_converter, image_processor_operator, {("output", "input")}
        )
        self.add_flow(image_processor_operator, demosaic, {("output", "receiver")})
        self.add_flow(demosaic, self.record_metadata_op, {("transmitter", "input")})
        self.add_flow(self.record_metadata_op, visualizer, {("output", "receivers")})
        self.add_flow(visualizer, monitor, {("camera_pose_output", "input")})


def main():
    parser = argparse.ArgumentParser(
        description="VB1940 Frame Validation with FUSA and nvCOMP CRC"
    )

    modes = hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode
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
    parser.add_argument(
        "--interface",
        default=None,
        help="Ethernet interface (auto-detected if not specified)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1500,
        help="FUSA capture request timeout in milliseconds",
    )
    parser.add_argument(
        "--sensor",
        type=int,
        choices=[0, 1],
        default=0,
        help="Sensor to use (0 or 1)",
    )
    parser.add_argument("--skip-reset", action="store_true")

    args = parser.parse_args()

    hololink_module.logging_level(args.log_level)
    logging.info("VB1940 Frame Validation with FUSA CoE and nvCOMP CRC")

    # CUDA primary context for operators and Cupy
    (cu_result,) = cuda.cuInit(0)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_device_ordinal = 0
    cu_result, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_result, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS

    channel_metadata = hololink_module.Enumerator.find_channel(channel_ip=args.hololink)
    logging.info(f"Hololink channel: {channel_metadata}")

    # Data channel and VB1940 camera for selected sensor
    channel_metadata_obj = hololink_module.Metadata(channel_metadata)
    hololink_module.DataChannel.use_sensor(channel_metadata_obj, args.sensor)
    hololink_channel = hololink_module.DataChannel(channel_metadata_obj)

    camera = hololink_module.sensors.vb1940.vb1940.Vb1940Cam(hololink_channel)
    camera_mode = hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode(
        args.camera_mode
    )

    recorder_queue = queue.Queue(maxsize=30_000)

    application = HoloscanApplication(
        args.headless,
        args.fullscreen,
        cu_context,
        cu_device_ordinal,
        hololink_channel,
        args.interface,
        camera,
        camera_mode,
        args.frame_limit,
        recorder_queue,
        args.timeout,
    )

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

    logging.info("Starting application...")
    application.run()

    hololink.stop()

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS

    # Skip startup/shutdown frames when scoring CRC agreement (warmup / teardown artifacts)
    SKIP_INITIAL_FRAMES = 10
    SKIP_FINAL_FRAMES = 10

    if hasattr(application, "record_metadata_op") and application.record_metadata_op:
        crcs = application.record_metadata_op.get_record()
        total_frames = len(crcs)

        if total_frames == 0:
            logging.warning("No CRC frames recorded")
        else:
            logging.info(f"Captured {total_frames} frames")

            skip_frames = SKIP_INITIAL_FRAMES
            nvcomp_errors = 0

            for frame_idx, (camera_crc, nvcomp_crc) in enumerate(crcs):
                if frame_idx < skip_frames:
                    continue
                if frame_idx >= (total_frames - SKIP_FINAL_FRAMES):
                    continue

                if camera_crc != nvcomp_crc:
                    logging.error(
                        f"CRC mismatch at frame {frame_idx}: "
                        f"camera=0x{camera_crc:08x}, nvcomp=0x{nvcomp_crc:08x}"
                    )
                    nvcomp_errors += 1

            validated_frames = max(total_frames - skip_frames - SKIP_FINAL_FRAMES, 0)
            nvcomp_success = (
                ((validated_frames - nvcomp_errors) / validated_frames) * 100
                if validated_frames > 0
                else 0
            )

            if validated_frames == 0:
                logging.warning(
                    "Insufficient frames for CRC validation after skipping first "
                    f"{skip_frames} and last {SKIP_FINAL_FRAMES} (captured "
                    f"{total_frames}); need more than "
                    f"{skip_frames + SKIP_FINAL_FRAMES} frames total."
                )
            elif nvcomp_errors == 0:
                logging.info(f"OK: Validated {validated_frames} CRCs - all matched!")

            logging.info("\n" + "=" * 80)
            logging.info("nvCOMP CRC VALIDATION SUMMARY (FUSA CoE)")
            logging.info("=" * 80)
            logging.info(f"Total frames: {total_frames}")
            logging.info(
                f"Validated frames: {validated_frames} (skipped first {skip_frames}, last {SKIP_FINAL_FRAMES})"
            )
            logging.info("")

            logging.debug("FRAME DIMENSIONS:")
            if hasattr(application, "_camera"):
                camera = application._camera
                logging.debug("  Camera:")
                logging.debug(f"    Width:           {camera._width:,} pixels")
                logging.debug(f"    Height:          {camera._height:,} pixels")
            if hasattr(application, "compute_crc_op") and application.compute_crc_op:
                logging.debug("  nvCOMP (full CSI buffer):")
                fs = getattr(application, "_csi_full_frame_size", None)
                sb = getattr(application, "_csi_start_byte", None)
                if fs is not None:
                    logging.debug(f"    Frame size:      {fs:,} bytes")
                if sb is not None:
                    logging.debug(f"    Start byte:      {sb:,}")
            logging.info("")

            logging.info("nvCOMP CRC vs Camera FPGA CRC:")
            if validated_frames > 0:
                logging.info(f"  Matches:    {validated_frames - nvcomp_errors}")
                logging.info(f"  Mismatches: {nvcomp_errors}")
                logging.info(f"  Success:    {nvcomp_success:.2f}%")
            else:
                logging.info("  Matches:    (none scored)")
                logging.info("  Mismatches: (none scored)")
                logging.info("  Success:    N/A (no frames in validation window)")
            logging.info("")

            logging.info("INTERPRETATION:")
            logging.info("-" * 50)
            if validated_frames == 0:
                logging.info(
                    "SKIP: No CRC agreement was scored (insufficient frames after skip window)."
                )
                logging.info(
                    "   - Increase total capture length or reduce initial/final skip counts."
                )
            elif nvcomp_errors == 0:
                logging.info("SUCCESS: nvCOMP matches camera perfectly (100%)")
                logging.info("   - nvCOMP CRC validation is working correctly")
                logging.info("   - FUSA CoE capture is delivering data intact")
            else:
                logging.info(
                    f"FAIL: nvCOMP has {nvcomp_errors} mismatches ({100 - nvcomp_success:.2f}% error rate)"
                )
                logging.info(
                    "   - Check nvCOMP / full-frame alignment and buffer layout"
                )
                logging.info("   - Verify FUSA CoE capture configuration")
            logging.info("=" * 80)

    # Latency breakdown from queued samples (see record_times tuple layout)
    frame_time_dts = []
    cpu_latency_dts = []
    operator_latency_dts = []
    processing_time_dts = []
    overall_time_dts = []

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

        logging.info("\n** PERFORMANCE REPORT (FUSA CoE + nvCOMP CRC) **")
        logging.info(f"{'Metric':<30}{'Min':<15}{'Max':<15}{'Avg':<15}")

        ft_min = min(frame_time_dts)
        ft_max = max(frame_time_dts)
        ft_avg = sum(frame_time_dts) / len(frame_time_dts)
        logging.info("Frame Time (in sec):")
        logging.info(f"{'Frame Time':<30}{ft_min:<15}{ft_max:<15}{ft_avg:<15}")

        cl_min = min(cpu_latency_dts)
        cl_max = max(cpu_latency_dts)
        cl_avg = sum(cpu_latency_dts) / len(cpu_latency_dts)
        logging.info("FUSA capture latency (in sec):")
        logging.info(f"{'Capture Latency':<30}{cl_min:<15}{cl_max:<15}{cl_avg:<15}")

        ol_min = min(operator_latency_dts)
        ol_max = max(operator_latency_dts)
        ol_avg = sum(operator_latency_dts) / len(operator_latency_dts)
        logging.info("FUSA to Operator after network operator latency (in sec):")
        logging.info(f"{'Operator Latency':<30}{ol_min:<15}{ol_max:<15}{ol_avg:<15}")

        pt_min = min(processing_time_dts)
        pt_max = max(processing_time_dts)
        pt_avg = sum(processing_time_dts) / len(processing_time_dts)
        logging.info("Processing of frame latency (in sec):")
        logging.info(f"{'Processing Latency':<30}{pt_min:<15}{pt_max:<15}{pt_avg:<15}")

        ot_min = min(overall_time_dts)
        ot_max = max(overall_time_dts)
        ot_avg = sum(overall_time_dts) / len(overall_time_dts)
        logging.info("Frame start till end of SW pipeline latency (in sec):")
        logging.info(f"{'SW Pipeline Latency':<30}{ot_min:<15}{ot_max:<15}{ot_avg:<15}")
    else:
        logging.info(
            f"Not enough frames ({len(recorder_queue_items)}) for performance statistics"
        )


if __name__ == "__main__":
    main()
