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

"""VB1940 FuSa frame validation with asynchronous PVA hardware CRC.

Pipeline: capture (full CSI) -> launch PVA -> profiler -> RAW unpack -> ISP.
After ISP, collect the frame-keyed PVA result while demosaic/display proceed.
RecordMetadataOp pairs that result with the demosaiced image for the summary.

HSDK5 carries CaptureMetadata on a paired port instead of entity metadata. The
receiver copies the full CSI buffer into an SDK loan when include_csi_frame=True;
subtracting a header offset from its usual pixel-only loan would read outside it.
ComputePvaCrcOp launches PVA and retains the input until its fence completes.
CheckPvaCrcOp collects the matching result after image processing and emits
a JAMCRC scalar with the capture timestamp and frame identity.

The final CRC summary skips the first and last ten frames, matching internal.
Timing ends after demosaic: HSDK5 HoloViz has no per-frame camera-pose output for
an after-render monitor. --headless renders into a CUDA image instead of a window.
"""

import argparse
import logging
import os
import sys
import threading
import time
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
# capture/CsiToBayerOp/ImageProcessorOp chain stamps every port pair with an identical
# capture_time, so this zero-width OnAligned+Exact{} pairs them precisely.
_METADATA_PAIRING_WINDOW = timedelta(0)

NS_PER_S = 1_000_000_000
PVA_CHECK_TIMEOUT_US = 2_000


def _cuda_check(result):
    err = result[0]
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        _, description = cuda_driver.cuGetErrorString(err)
        raise RuntimeError(f"CUDA error: {description}")
    if len(result) == 2:
        return result[1]
    return result[1:]


class _PvaJobs:
    """Frame-keyed PVA fences, retained inputs and completed results for one run."""

    instances = {}

    def __init__(self, pva):
        self.pva = pva
        self.lock = threading.RLock()
        self.pending = {}
        self.completed = {}
        self.output = cp.zeros(2, dtype=cp.uint32)

    def _collect(self, frame_id, timeout_us):
        fence, sample, source = self.pending[frame_id]
        status = self.pva.check_results(
            fence, self.output.data.ptr, timeout_us=timeout_us
        )
        if status == 1:
            return False
        if status != 0:
            raise RuntimeError(f"PVA CRC failed for frame {frame_id}: status={status}")
        # check_results copies on a library-owned CUDA stream. Finish that copy
        # before reading the scalar or releasing the input/fence.
        cp.cuda.runtime.deviceSynchronize()
        self.completed[frame_id] = ~int(self.output[0].get()) & 0xFFFFFFFF
        self.pva.free_fence(fence)
        del self.pending[frame_id]
        while len(self.completed) > 64:
            del self.completed[next(iter(self.completed))]
        return True

    def launch(self, frame_id, sample, source):
        with self.lock:
            if frame_id in self.pending or frame_id in self.completed:
                raise RuntimeError(f"Duplicate PVA frame identity: {frame_id}")
            for key in list(self.pending):
                self._collect(key, 0)
            # Bound retained runtime-pool loans even when a downstream frame drops.
            # Usually the late checker has already retired the preceding frame.
            if len(self.pending) >= 2:
                key = next(iter(self.pending))
                if not self._collect(key, PVA_CHECK_TIMEOUT_US):
                    raise RuntimeError(f"PVA CRC timeout for frame {key}")
            fence = self.pva.launch_compute(source.data.ptr)
            self.pending[frame_id] = (fence, sample, source)

    def result(self, frame_id):
        with self.lock:
            if frame_id in self.pending:
                if not self._collect(frame_id, PVA_CHECK_TIMEOUT_US):
                    raise RuntimeError(f"PVA CRC timeout for frame {frame_id}")
            # Never substitute the next frame's result for a missing one.
            result = self.completed.pop(frame_id, None)
            if result is not None:
                # Compare 16-bit identities across wrap; pending fences keep their owners.
                for key in list(self.completed):
                    if 0 < ((frame_id - key) & 0xFFFF) < 0x8000:
                        del self.completed[key]
            return result

    def close(self):
        with self.lock:
            # Drain hardware before releasing loans, including after an interrupted run.
            for key in list(self.pending):
                if not self._collect(key, -1):
                    raise RuntimeError(f"Could not drain PVA frame {key}")
            self.completed.clear()


class ComputePvaCrcOp(hs.Operator):
    """Launch PVA early and forward immediately; the checker waits after image processing."""

    in_tensor = hs.Input("in_tensor", hs.payload.Tensor)
    in_metadata = hs.Input("in_metadata", capture_metadata.CaptureMetadata)
    tensor = hs.Output("tensor", hs.payload.Tensor)
    metadata = hs.Output("metadata", capture_metadata.CaptureMetadata)
    data_size = hs.Param(0)
    use_dual_vpu = hs.Param(True)
    job_key = hs.Param("compute_pva_crc")

    def setup(self, spec):
        spec.input(self.in_tensor, queue_depth=4)
        spec.input(self.in_metadata, queue_depth=4)
        spec.output(self.tensor).produces_tensor(
            hs.tensor.OutputSpec(
                storage="pass_through",
                memory_kind=hs.tensor.MemoryKind.CUDA_DEVICE,
                dtype="uint8",
                rank=1,
                bounds=hs.tensor.TensorBounds(max_byte_span=64 * 1024 * 1024),
            ),
        )
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

    def start(self):
        if not self.data_size:
            raise ValueError(
                "ComputePvaCrcOp.data_size is required and must be nonzero."
            )

        operator_dir = os.path.dirname(os.path.abspath(__file__))
        import hololink.operators.pva_crc as pva_crc_pkg

        cpp_dir = os.path.join(os.path.dirname(pva_crc_pkg.__file__), "cpp")
        if cpp_dir not in sys.path:
            sys.path.insert(0, cpp_dir)
        dev_build_path = os.environ.get(
            "PVA_CRC_DEV_BUILD",
            os.path.join(operator_dir, "..", "build", "pva_crc", "cpp"),
        )
        if os.path.exists(dev_build_path) and dev_build_path not in sys.path:
            sys.path.insert(0, dev_build_path)

        import pva_crc

        self._pva_crc = pva_crc.PvaCrc()
        self._pva_crc.initialize(
            data_size=self.data_size,
            remaining_size=0,
            is_first_chunk=True,
            use_dual_vpu=self.use_dual_vpu,
        )
        logging.info(
            f"ComputePvaCrcOp initialized: data_size={self.data_size:,} bytes, "
            f"use_dual_vpu={self.use_dual_vpu}"
        )

        self._jobs = _PvaJobs(self._pva_crc)
        _PvaJobs.instances[self.job_key] = self._jobs

    def compute(self, context):
        tensor_sample = self.in_tensor.receive()
        metadata_sample = self.in_metadata.receive()
        if tensor_sample.is_missing or metadata_sample.is_missing:
            return
        frame_id = tensor_sample.metadata.frame_id
        if frame_id != metadata_sample.data.frame_number:
            raise RuntimeError("PVA input tensor/metadata frame identity mismatch")
        if tensor_sample.data.nbytes != self.data_size:
            raise ValueError("PVA input size does not match the full CSI frame size")
        with cp.cuda.Stream(non_blocking=True) as stream:
            source = cp.from_dlpack(
                tensor_sample.data.__dlpack__(
                    stream=stream.ptr, max_version=(1, 0), copy=False
                )
            )
            # Make capture's CUDA writes visible to the PVA library's own stream.
            stream.synchronize()
            self._jobs.launch(frame_id, tensor_sample, source)
        self.tensor.forward(tensor_sample)
        self.metadata.forward(metadata_sample)

    def stop(self):
        jobs = getattr(self, "_jobs", None)
        if jobs is not None:
            jobs.close()
            if _PvaJobs.instances.get(self.job_key) is jobs:
                del _PvaJobs.instances[self.job_key]


class CheckPvaCrcOp(hs.Operator):
    """Collect the matching frame's PVA result after image processing."""

    in_metadata = hs.Input("in_metadata", capture_metadata.CaptureMetadata)
    crc = hs.Output("crc", hs.payload.uint32)
    compute_op_name = hs.Param("compute_pva_crc")

    def setup(self, spec):
        spec.input(self.in_metadata, queue_depth=4)
        spec.output(self.crc)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.in_metadata))

    def compute(self, context):
        sample = self.in_metadata.receive()
        if sample.is_missing:
            return
        frame_id = sample.metadata.frame_id
        if frame_id != sample.data.frame_number:
            raise RuntimeError("PVA checker metadata frame identity mismatch")
        result = _PvaJobs.instances[self.compute_op_name].result(frame_id)
        if result is None:
            logging.info("No retained PVA result for dropped/stale frame %d", frame_id)
            return
        self.crc.emit(
            result,
            options=hs.EmitOptions(
                capture_time=sample.metadata.capture_time, frame_id=frame_id
            ),
        )


def _ns_to_ms(ns):
    return ns / 1.0e6


def _print_stat(label, values):
    mn = _ns_to_ms(min(values))
    mx = _ns_to_ms(max(values))
    avg = _ns_to_ms(sum(values)) / len(values)
    print(f"{label:<32}{mn:>10.3f} ms{mx:>10.3f} ms{avg:>10.3f} ms")


class InstrumentedTimeProfiler(hs.Operator):
    """Validate full CSI sizing, ordering and timing before unpack/ISP."""

    in_tensor = hs.Input("in_tensor", hs.payload.Tensor)
    in_metadata = hs.Input("in_metadata", capture_metadata.CaptureMetadata)
    tensor = hs.Output("tensor", hs.payload.Tensor)
    metadata = hs.Output("metadata", capture_metadata.CaptureMetadata)

    entry_times = {}
    calculated_frame_size = hs.Param(0)
    # Allow twice the selected mode's nominal frame period, matching internal.
    expected_frame_period_ms = hs.Param(1000.0 / 30.0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prev_frame_number = None
        self._prev_entry_ns = None

    def setup(self, spec):
        spec.input(self.in_tensor, queue_depth=4)
        spec.input(self.in_metadata, queue_depth=4)
        spec.output(self.tensor).produces_tensor(
            hs.tensor.OutputSpec(
                storage="pass_through",
                memory_kind=hs.tensor.MemoryKind.CUDA_DEVICE,
                dtype="uint8",
                rank=1,
                bounds=hs.tensor.TensorBounds(max_byte_span=64 * 1024 * 1024),
            ),
        )
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
            logging.warning(
                "Dropping mismatched profiler tensor/metadata frame identities"
            )
            return
        type(self).entry_times[frame_number] = entry_ns
        while len(type(self).entry_times) > 64:
            del type(self).entry_times[next(iter(type(self).entry_times))]

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

        self.tensor.forward(tensor_sample)
        self.metadata.forward(metadata_sample)


class RecordMetadataOp(hs.Operator):
    """Record FPGA/PVA CRC pairs after demosaic, matching the nvCOMP app."""

    in_image = hs.Input("in_image", hs.schema.Image)
    in_metadata = hs.Input("in_metadata", capture_metadata.CaptureMetadata)
    in_crc = hs.Input("in_crc", hs.payload.uint32)
    warmup_frames = hs.Param(10)
    instance = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._record = []
        self._mismatched = 0
        self._times = []
        RecordMetadataOp.instance = self

    def setup(self, spec):
        spec.input(self.in_image, queue_depth=4)
        spec.input(self.in_metadata, queue_depth=4)
        spec.input(self.in_crc, queue_depth=4)

    def contract(self):
        return (
            hs.Contract()
            .trigger(
                hs.OnAligned(
                    ports=[self.in_image, self.in_metadata, self.in_crc],
                    window=_METADATA_PAIRING_WINDOW,
                )
            )
            .input(self.in_image, hs.Exact())
            .input(self.in_metadata, hs.Exact())
            .input(self.in_crc, hs.Exact())
        )

    def compute(self, context):
        complete_ns = time.time_ns()
        image_sample = self.in_image.receive()
        metadata_sample = self.in_metadata.receive()
        crc_sample = self.in_crc.receive()
        if (
            image_sample.is_missing
            or metadata_sample.is_missing
            or crc_sample.is_missing
        ):
            return
        if (
            crc_sample.metadata.frame_id != image_sample.metadata.frame_id
            or image_sample.metadata.frame_id != metadata_sample.data.frame_number
        ):
            logging.warning(
                "Dropping mismatched PVA image/metadata/CRC frame identities"
            )
            self._mismatched += 1
            return
        frame = metadata_sample.data
        self._record.append((frame.crc, crc_sample.data))
        start_ns = frame.timestamp_s * NS_PER_S + frame.timestamp_ns
        end_ns = frame.metadata_s * NS_PER_S + frame.metadata_ns
        received_ns = frame.received_s * NS_PER_S + frame.received_ns or end_ns
        operator_ns = InstrumentedTimeProfiler.entry_times.pop(
            frame.frame_number, frame.operator_entry_ns or received_ns
        )
        self._times.append(
            (
                end_ns - start_ns,
                received_ns - end_ns,
                operator_ns - received_ns,
                complete_ns - operator_ns,
                complete_ns - received_ns,
                complete_ns - start_ns,
            )
        )

    def get_record(self):
        return self._record

    def print_report(self):
        total = len(self._record)
        settled = self._record[self.warmup_frames : total - self.warmup_frames]
        errors = sum(camera_crc != pva_crc for camera_crc, pva_crc in settled)
        print("PVA CRC VALIDATION SUMMARY (FUSA CoE)")
        print(f"Total frames: {total}")
        print(
            f"Validated frames: {len(settled)} (skipped first {self.warmup_frames}, last {self.warmup_frames})"
        )
        if self._mismatched:
            print(f"Skipped frames with mismatched identities: {self._mismatched}")
        if not settled:
            if self._mismatched:
                # Frames arrived, but none could be checked: that's a failure, not a pass.
                print(
                    "No frames were validated: their image, metadata and CRC didn't match."
                )
                return 1
            print(
                "Insufficient frames for CRC validation after startup/shutdown trimming."
            )
            return 0
        success_rate = 100.0 * (len(settled) - errors) / len(settled)
        print(
            f"CRC Validation: {len(settled)} frames checked, {errors} errors, {success_rate:.2f}% success"
        )
        labels = [
            "Sensor frame time",
            "FuSa capture latency",
            "Operator dispatch",
            "Processing latency",
            "Overall latency",
            "SW pipeline latency",
        ]
        times = self._times[self.warmup_frames : total - self.warmup_frames]
        print(f"\n=== Latency Report ({len(times)} frames, through demosaic) ===")
        for index, label in enumerate(labels):
            values = [sample[index] for sample in times]
            if index in (1, 5) and min(values) < 0:
                print(f"{label:<32} unavailable (negative cross-clock timing)")
            else:
                _print_stat(label, values)
        if any(sample[1] < 0 or sample[5] < 0 for sample in times):
            print(
                "Host/board clock alignment needs checking; cross-clock latencies are unavailable."
            )
        return errors


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
    sensor_id,
    camera_mode,
    timeout_ms,
    interface_override,
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
    hl.use_sensor(metadata, sensor_id)
    channel = hl.DataChannel(metadata)

    interface = interface_override or metadata.get_string("interface") or ""
    mac_str = metadata.get_string("mac_id") or ""
    mac_addr = [int(byte, 16) for byte in mac_str.split(":")]
    if len(mac_addr) != 6:
        raise ValueError("MAC address must have exactly 6 bytes")

    hololink_dev = channel.hololink()
    hololink_dev.start()
    if not skip_reset:
        hololink_dev.reset()

    print("Waiting for PTP synchronization...")
    if not hololink_dev.ptp_synchronize():
        print("PTP synchronization failed.", file=sys.stderr)
        hololink_dev.stop()
        _cuda_check(cuda_driver.cuDevicePrimaryCtxRelease(cu_device))
        return 1
    print("PTP synchronized.")

    camera = hl.NativeVb1940Sensor(channel)
    if not skip_reset:
        camera.setup_clock()
    camera.configure(camera_mode)

    pixel_format = camera.get_pixel_format()
    bayer_format = camera.get_bayer_format()
    frame_width = camera.get_width()
    frame_height = camera.get_height()

    # Match internal: RAW8 uses CSI unpack; other formats use HSB packing.
    packetizer_enabled = pixel_format != int(hl.PixelFormat.RAW_8)

    # Resolve geometry ONCE using FusaCoeCaptureOp's HSB-alignment-aware adapter, and reuse that
    # same geometry dict for both FusaCoeCaptureOp and CsiToBayerOp below (see
    # fusa_coe_vb1940_player.py's identical comment for why resolving twice would misparse).
    geometry = hl.resolve_geometry(camera, packetizer_enabled)
    start_byte = geometry["start_byte"]
    full_frame_size = (
        start_byte
        + geometry["bytes_per_line"] * geometry["pixel_height"]
        + geometry["trailing_bytes"]
    )

    RecordMetadataOp.instance = None
    InstrumentedTimeProfiler.entry_times = {}

    # Build the graph.
    graph = hs.Graph("vb1940_fusa_pva_crc_validation")

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
        include_csi_frame=True,
        frame_limit=frame_limit or 0,
        frame_limit_ends_run=True,
    )

    compute_crc = graph.op(
        ComputePvaCrcOp,
        "compute_pva_crc",
        data_size=full_frame_size,
        use_dual_vpu=True,
    )
    graph.add_flow(capture.tensor, compute_crc.in_tensor, queue_depth=4)
    graph.add_flow(capture.metadata, compute_crc.in_metadata, queue_depth=4)

    profiler = graph.op(
        InstrumentedTimeProfiler,
        "profiler",
        calculated_frame_size=full_frame_size,
        expected_frame_period_ms=1000.0 / (60 if camera_mode == 3 else 30),
    )
    graph.add_flow(compute_crc.tensor, profiler.in_tensor, queue_depth=4)
    graph.add_flow(compute_crc.metadata, profiler.in_metadata, queue_depth=4)

    converter_type = (
        hl.operators.PackedFormatConverterOp
        if packetizer_enabled
        else hl.operators.CsiToBayerOp
    )
    converter_params = dict(
        cuda_device_ordinal=cuda_device_ordinal,
        geometry=geometry,
        bayer_format=bayer_format,
        header_included=True,
    )
    converter = graph.op(converter_type, "unpack", **converter_params)
    graph.add_flow(profiler.tensor, converter.in_tensor, queue_depth=4)
    graph.add_flow(profiler.metadata, converter.in_metadata, queue_depth=4)

    image_processor = graph.op(
        hl.operators.ImageProcessorOp,
        "image_processor",
        pixel_format=pixel_format,
        pixel_width=geometry["pixel_width"],
        pixel_height=geometry["pixel_height"],
        bayer_format=_BAYER_FORMAT_VALUES[bayer_format],
        # Matches this app's own HSDK4-era default (a VB1940 FUSA CoE-specific quirk).
        optical_black=0,
        cuda_device_ordinal=cuda_device_ordinal,
    )
    graph.add_flow(converter.tensor, image_processor.in_tensor, queue_depth=4)
    graph.add_flow(converter.metadata, image_processor.in_metadata, queue_depth=4)

    check_crc = graph.op(CheckPvaCrcOp, "check_crc_late")
    graph.add_flow(image_processor.metadata, check_crc.in_metadata, queue_depth=4)

    demosaic = graph.op(
        hv.operators.BayerDemosaicOp,
        "bayer_demosaic",
        bayer_grid_pos=bayer_format,
        generate_alpha=True,
        alpha_value=65535,
    )
    graph.add_flow(image_processor.tensor, demosaic.input, queue_depth=4)

    record_metadata = graph.op(
        RecordMetadataOp,
        "record_metadata",
    )
    graph.add_flow(demosaic.output, record_metadata.in_image, queue_depth=4)
    graph.add_flow(check_crc.crc, record_metadata.in_crc, queue_depth=4)
    graph.add_flow(image_processor.metadata, record_metadata.in_metadata, queue_depth=4)

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
            "title": "VB1940 FUSA CoE PVA CRC Validation",
            "width": frame_width,
            "height": frame_height,
            "fullscreen": fullscreen,
            "framebuffer_srgb": True,
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
        # Present each admitted camera image without a clock-trigger freshness
        # cutoff; the depth-one display queue below retains the newest image.
        trigger={"kind": "layer", "layer": 0},
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
    for operator_path in (
        "capture",
        "compute_pva_crc",
        "profiler",
        "image_processor",
    ):
        options.deployment.bind_tensor_output_device(
            operator_path=operator_path,
            output_port="tensor",
            device=hs.execution.DeviceId(cuda_device_ordinal),
        )
    options.deployment.bind_tensor_output_device(
        operator_path="unpack",
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
            print(
                f"  {diag}: required={diag.required}; provided={diag.provided}",
                file=sys.stderr,
            )
        hololink_dev.stop()
        _cuda_check(cuda_driver.cuDevicePrimaryCtxRelease(cu_device))
        return 1

    print(
        f"Capturing sensor {sensor_id} mode={camera_mode} {frame_width}x{frame_height} -- running (Ctrl+C to stop)..."
    )

    try:
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
    finally:
        hololink_dev.stop()
        _cuda_check(cuda_driver.cuDevicePrimaryCtxRelease(cu_device))

    if failed:
        print("Run failed; see runtime diagnostics above.", file=sys.stderr)
        return 1

    if RecordMetadataOp.instance is not None:
        errors = RecordMetadataOp.instance.print_report()
        if errors:
            return 1

    print("Done.")
    return 0


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
        "--interface",
        default=None,
        help="Ethernet interface (default: auto-detected from enumeration metadata)",
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
        "--skip-reset",
        action="store_true",
        help="Don't call reset on the hololink device.",
    )
    parser.add_argument(
        "--log-level", type=int, default=20, help="Python logging level to display"
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    return run(
        args.hololink,
        args.sensor,
        args.camera_mode,
        args.timeout,
        args.interface,
        args.skip_reset,
        args.frame_limit,
        args.headless,
        args.fullscreen,
    )


if __name__ == "__main__":
    sys.exit(main())
