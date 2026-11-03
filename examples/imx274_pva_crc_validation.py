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

# Single IMX274 sensor via RoceReceiverOp, validating each frame's CRC using NVIDIA PVA
# (Programmable Vision Accelerator) hardware against the camera's own FPGA-embedded CRC.
# Python port of this repo's own previous (pre-HSDK5) Python version of this app -- there is no
# .cpp sibling. See python/hololink/operators/pva_crc/ for the ComputePvaCrcOp/CheckPvaCrcOp
# binding this app drives (restored from this repo's own `internal` branch and ported from
# pybind11 to nanobind; see that commit for the full rationale -- these operators were always
# pure Python wrapping a thin C++ handle, not something needing new C++ design).
#
#   RoceReceiverOp -> ComputePvaCrcOp -> FrameValidationOp -> CsiToBayerOp -> ImageProcessorOp ->
#       CheckPvaCrcOp -> BayerDemosaicOp -> HolovizOp
#
# ComputePvaCrcOp launches the (non-blocking) PVA computation immediately after the receiver, on
# the raw CSI frame -- the same bytes the camera's own embedded CRC covers. CheckPvaCrcOp waits
# for the result after CsiToBayerOp/ImageProcessorOp have had a chance to run, hiding most of
# PVA's own ~1ms latency behind that other work (matches the old app's own placement and its own
# comment on why). FrameValidationOp (examples/linux_imx274_frame_validation.py) is reused
# unmodified for frame-size/frame-number/timing consistency checks, with crc_frame_check=0 --
# this app's whole point is validating the PVA *hardware* CRC path specifically, so a second,
# redundant CPU-side zlib CRC check would be pointless (PVA's purpose is to offload that exact
# computation from the CPU).
#
# Three real architectural differences from the old app, all forced by the same root cause --
# HSDK5 operators have no free-form per-frame metadata dict (confirmed directly: holoscan's own
# Sample.metadata is a fixed SampleMetadata struct about sample provenance, not an
# application-definable key/value store) -- so nothing here can do the old app's own
# `self.metadata["pva_fence"] = fence` trick:
#   - The fence handle (and the live PvaCrc instance itself -- check_results()/free_fence() must
#     run on the same instance that issued launch_compute(), a hard constraint of the precompiled
#     PVA library, not a design choice) travel through a small Python-side class-level stash,
#     _PvaCrcShared, instead -- the same kind of workaround this whole porting effort already
#     uses whenever two pure-Python operators need to share a live object graph.op() can't pass
#     as a kwarg. Unlike every other such stash in this effort (a single slot), fences are kept
#     per frame (ComputePvaCrcOp's async launch and CheckPvaCrcOp's later check can be mid-flight
#     for different frames at once, by design -- that's the whole point of hiding PVA's latency),
#     so this one is a dict keyed by frame number per operator name. A frame that only one of the
#     two operators sees (a dropped frame) costs only that frame's check.
#   - The old app's RecordMetadataOp/validate_frame()/recorder_queue machinery (a rolling
#     thread-safe queue polled for frame-number/size/timestamp/CRC consistency) is replaced by
#     reusing FrameValidationOp's own already-proven consolidation of exactly those same checks,
#     rather than re-deriving bespoke queue-polling logic.
#   - The old app's InstrumentedTimeProfiler/MonitorOperator pair (wall-clock timestamps stamped
#     into the per-frame metadata dict as it flowed through a full visualization pipeline) has no
#     HSDK5 equivalent either -- same reasoning as imx274_latency.py's own port. CheckPvaCrcOp
#     reports its own CRC-match statistics from the typed hololink.CaptureMetadataT fields each
#     receiver/operator already stamps, not wall-clock stamping of its own.
#
# --frame-limit is the receiver's frame_limit parameter: the receiver ends the run after that
# many frames. HSDK5 has no CountCondition; this replaces the CountCondition of the old app.
#
# Usage:
#   imx274_pva_crc_validation.py [OPTIONS]
#
# Options:
#   --hololink IP                 IP address of Hololink board (default: 192.168.0.2)
#   --camera-mode N                IMX274 mode (default: 0 = 3840x2160@60fps)
#   --ibv-name NAME                IBV device to use (default: the device on the board's network interface)
#   --ibv-port N                   Port number of IBV device (default: 1)
#   --expander-configuration N    I2C expander configuration (0 or 1, default: 0)
#   --pattern N                   Configure a test pattern (0-11) instead of live video
#   --skip-reset                  Don't call reset on the hololink device
#   --frame-limit N               Number of frames to capture (default: unlimited)
#   --headless                    Run without opening a window (still renders, presents nowhere)
#   --fullscreen                  Run in fullscreen mode
#   --log-level N                 Python logging level (default: 20 = INFO)

import argparse
import logging
import os
import sys
import threading
import time
import zlib
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
# receiver/CsiToBayerOp/ImageProcessorOp chain stamps every port pair with an identical
# capture_time, so this bounded-window OnAligned+Exact{} pairs them precisely. Same constant as
# examples/linux_imx274_frame_validation.py / imx274_latency.py.
_METADATA_PAIRING_WINDOW = timedelta(microseconds=500)

NS_PER_S = 1_000_000_000

# PVA CRC timing constants (microseconds), matching the old app's own values.
PVA_CHECK_TIMEOUT_US = 2_000  # 2ms -- margin over PVA's typical ~1ms execution time.

# Validation-summary trim, matching the old app's own constants exactly: skip frames at the very
# start/end of the run, where pipeline warmup/shutdown produces artifacts unrelated to PVA itself.
SKIP_INITIAL_FRAMES = 10
SKIP_FINAL_FRAMES = 10


def _cuda_check(result):
    err = result[0]
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        _, description = cuda_driver.cuGetErrorString(err)
        raise RuntimeError(f"CUDA error: {description}")
    if len(result) == 2:
        return result[1]
    return result[1:]


class _ValidationReportHolder:
    """Plain Python-side stash so run() can read FrameValidationOp's accumulated state back
    after session.wait() returns -- there's no frame-limit-triggered self-report here (unlike
    LatencyReportSinkOp) since --frame-limit defaults to unbounded, matching the old app. Same
    shape as examples/linux_imx274_frame_validation.py's own (duplicated here, not imported --
    every custom operator in this porting effort is self-contained per app file)."""

    instance = None


def _ns_to_ms(ns):
    return ns / 1.0e6


def _print_stat(label, values):
    mn = _ns_to_ms(min(values))
    mx = _ns_to_ms(max(values))
    avg = _ns_to_ms(sum(values)) / len(values)
    print(f"{label:<32}{mn:>10.3f} ms{mx:>10.3f} ms{avg:>10.3f} ms")


class FrameValidationOp(hs.Operator):
    """Pass-through: validates each received frame, then forwards tensor+metadata unchanged.
    Duplicated verbatim from examples/linux_imx274_frame_validation.py (see that file's own
    header comment for the full design rationale) -- crc_frame_check is disabled (0) by this
    app's own run() below, since PVA CRC validation is this app's whole point and a second,
    redundant CPU-side zlib check would be pointless.

    Checks: CRC32 (every crc_frame_check frames, if > 0) against the FPGA-stamped
    CaptureMetadataT.crc field; received-vs-calculated frame size; frame_number ordering;
    inter-frame timing. Also accumulates a latency report (sensor frame time / DMA+CPU wakeup /
    operator dispatch), printed via print_report() once run() calls it after the session ends.
    """

    in_tensor = hs.Input("in_tensor", hs.payload.Tensor)
    in_metadata = hs.Input("in_metadata", capture_metadata.CaptureMetadata)
    tensor = hs.Output("tensor", hs.payload.Tensor)
    metadata = hs.Output("metadata", capture_metadata.CaptureMetadata)

    calculated_frame_size = hs.Param(0)
    crc_frame_check = hs.Param(0)
    expected_frame_period_ms = hs.Param(16.6)
    frame_limit = hs.Param(0)
    warmup_frames = hs.Param(5)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._crc_errors = 0
        self._crc_checked = 0
        self._prev_frame_number = None
        self._prev_entry_ns = None
        self._frame_count = 0
        self._latency_samples = []
        _ValidationReportHolder.instance = self

    def setup(self, spec):
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
        entry_ns = time.time_ns()

        tensor_sample = self.in_tensor.receive()
        metadata_sample = self.in_metadata.receive()
        if tensor_sample.is_missing or metadata_sample.is_missing:
            return

        frame = metadata_sample.data
        frame_number = frame.frame_number
        self._frame_count += 1

        if self.crc_frame_check > 0 and frame_number % self.crc_frame_check == 0:
            # A received HSDK5 tensor is read-only, and HSDK5 exports it only as versioned
            # DLPack on a real CUDA stream (cupy's default legacy stream is refused with
            # kNotSupported). Import a copy of it on a private non-blocking stream, and
            # synchronize before the array is used.
            with cp.cuda.Stream(non_blocking=True) as stream:
                capsule = tensor_sample.data.__dlpack__(
                    stream=stream.ptr,
                    copy=True,
                )
                cp_frame = cp.from_dlpack(capsule)
                stream.synchronize()
            np_frame = cp.asnumpy(cp_frame, stream=stream)
            stream.synchronize()
            crc32_calculated = zlib.crc32(np_frame.tobytes())
            crc32_expected = frame.crc ^ 0xFFFFFFFF
            self._crc_checked += 1
            if crc32_calculated != crc32_expected:
                logging.debug(
                    f"frame number {frame_number}: CRC32 mismatch: "
                    f"calculated={crc32_calculated}, expected={crc32_expected}"
                )
                self._crc_errors += 1

        # The received tensor knows its size; only CRC-checked frames are copied.
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

        in_warmup = self._frame_count <= self.warmup_frames or (
            self.frame_limit
            and self._frame_count > self.frame_limit - self.warmup_frames
        )
        if not in_warmup:
            frame_start_ns = frame.timestamp_s * NS_PER_S + frame.timestamp_ns
            frame_end_ns = frame.metadata_s * NS_PER_S + frame.metadata_ns
            received_ns = frame.received_s * NS_PER_S + frame.received_ns
            self._latency_samples.append(
                (
                    frame_end_ns - frame_start_ns,  # sensor frame time
                    received_ns - frame_end_ns,  # DMA + CPU wakeup
                    entry_ns - received_ns,  # operator dispatch
                )
            )

        self.tensor.forward(tensor_sample)
        self.metadata.forward(metadata_sample)

    def print_report(self):
        if self._crc_frame_check_enabled():
            total = self._crc_checked
            errors = self._crc_errors
            if total > 0:
                success_rate = ((total - errors) / total) * 100
                print(
                    f"CRC Validation: {total} frames checked, {errors} errors, "
                    f"{success_rate:.2f}% success"
                )
            else:
                print("CRC Validation: enabled, but no frames were checked.")
        else:
            print("CRC validation was disabled (--crc-frame-check=0).")

        if not self._latency_samples:
            print("No settled frames collected -- nothing to report.")
            return

        labels = ["Sensor frame time", "DMA + CPU wakeup", "Operator dispatch"]
        print(f"\n=== Latency Report ({len(self._latency_samples)} frames) ===")
        print(f"{'Metric':<32}{'Min':>12}{'Max':>12}{'Avg':>12}")
        print("-" * 68)
        for index, label in enumerate(labels):
            _print_stat(label, [sample[index] for sample in self._latency_samples])
        print("-" * 68 + "\n")

    def _crc_frame_check_enabled(self):
        return self.crc_frame_check > 0


def _import_pva_crc():
    """Import the compiled pva_crc extension (ComputePvaCrcOp loads it in start()).

    hololink.operators.pva_crc is a real subpackage (houses the compiled pva_crc.*.so
    extension), but -- like any Python subpackage -- it isn't an attribute of the
    already-imported hololink.operators until something actually imports it;
    hololink.operators.__init__.py itself never does (confirmed empirically: a bare
    `hl.operators.pva_crc` raises AttributeError without this import first). The extension
    exists only in builds made with HOLOLINK_BUILD_PVA (PVA SDK present), so run() calls this
    first and reports a missing extension instead of failing inside the operator's start().
    """
    import hololink.operators.pva_crc as pva_crc_pkg

    cpp_dir = os.path.join(os.path.dirname(pva_crc_pkg.__file__), "cpp")
    if cpp_dir not in sys.path:
        sys.path.insert(0, cpp_dir)
    operator_dir = os.path.dirname(os.path.abspath(__file__))
    dev_build_path = os.environ.get(
        "PVA_CRC_DEV_BUILD",
        os.path.join(operator_dir, "..", "build", "pva_crc", "cpp"),
    )
    if os.path.exists(dev_build_path) and dev_build_path not in sys.path:
        sys.path.insert(0, dev_build_path)

    import pva_crc

    return pva_crc


class _PvaCrcShared:
    """Shares one PvaCrc instance (and its pending fences, keyed by frame number) between
    ComputePvaCrcOp and CheckPvaCrcOp. See this file's own header comment for why this exists --
    graph.op() can't pass a live Python object as a kwarg even between two pure-Python operators,
    and fences are kept per frame (not a single slot) because launch and check are deliberately
    overlapped across frames to hide PVA's own latency. Keying them by frame number means a frame
    one operator never sees cannot shift every later check onto the wrong frame's CRC.
    """

    _instances = {}
    _pending_fences = {}
    _conditions = {}

    @classmethod
    def register(cls, name, pva_crc):
        cls._instances[name] = pva_crc
        cls._pending_fences[name] = {}
        cls._conditions[name] = threading.Condition()

    @classmethod
    def instance(cls, name):
        return cls._instances[name]

    @classmethod
    def push_fence(cls, name, frame_number, fence):
        condition = cls._conditions[name]
        with condition:
            cls._pending_fences[name][frame_number] = fence
            condition.notify_all()

    @classmethod
    def take_fence(cls, name, frame_number, timeout_s=0.2):
        """Return (fence, stale): the fence launched for frame_number, or None if it did not
        arrive within timeout_s, and the fences of earlier frames, which nobody will check; the
        caller frees them. ComputePvaCrcOp is a side-tap on the receiver, so it is not
        guaranteed to have launched this frame's CRC before CheckPvaCrcOp (further down the
        chain) asks for it. It launches frames in order, so once a later frame's fence is there,
        this frame's will never come."""
        condition = cls._conditions[name]
        pending = cls._pending_fences[name]
        with condition:
            condition.wait_for(
                lambda: frame_number in pending
                or any(number > frame_number for number in pending),
                timeout=timeout_s,
            )
            fence = pending.pop(frame_number, None)
            stale = [
                pending.pop(number)
                for number in sorted(pending)
                if number < frame_number
            ]
        return fence, stale


class ComputePvaCrcOp(hs.Operator):
    """Launch PVA CRC computation (non-blocking) on the raw CSI frame. A pure sink (side-tap on
    the receiver's tensor output), not a pass-through: when this Python operator forwarded the
    frame to FrameValidationOp (another Python operator), the runtime failed with
    RUNTIME_PORT_BINDING_UNAVAILABLE before FrameValidationOp received its first frame. See
    CheckPvaCrcOp for retrieving the result."""

    in_tensor = hs.Input("in_tensor", hs.payload.Tensor)

    data_size = hs.Param(0)
    remaining_size = hs.Param(0)
    is_first_chunk = hs.Param(True)
    use_dual_vpu = hs.Param(True)
    start_byte = hs.Param(0)
    # Key for _PvaCrcShared; must equal CheckPvaCrcOp.compute_op_name. HSDK5 Python
    # operators have no name attribute.
    shared_name = hs.Param("compute_pva_crc")

    def setup(self, spec):
        spec.input(self.in_tensor, queue_depth=4)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.in_tensor))

    def start(self):
        if not self.data_size:
            raise ValueError(
                "ComputePvaCrcOp.data_size is required and must be nonzero."
            )

        pva_crc = _import_pva_crc()

        self._pva_crc = pva_crc.PvaCrc()
        self._pva_crc.initialize(
            data_size=self.data_size,
            remaining_size=self.remaining_size,
            is_first_chunk=self.is_first_chunk,
            use_dual_vpu=self.use_dual_vpu,
        )
        _PvaCrcShared.register(self.shared_name, self._pva_crc)
        logging.info(
            f"ComputePvaCrcOp initialized: data_size={self.data_size:,} bytes, "
            f"use_dual_vpu={self.use_dual_vpu}, start_byte={self.start_byte}"
        )

    def compute(self, context):
        sample = self.in_tensor.receive()
        if sample.is_missing:
            return

        # Import a copy, as in FrameValidationOp. The PVA engine reads the buffer
        # outside CUDA stream order, so wait until the frame is ready before launching.
        with cp.cuda.Stream(non_blocking=True) as stream:
            capsule = sample.data.__dlpack__(
                stream=stream.ptr,
                copy=True,
            )
            cp_frame = cp.from_dlpack(capsule)
            stream.synchronize()
        input_ptr = cp_frame.data.ptr + self.start_byte

        fence = self._pva_crc.launch_compute(input_ptr)
        _PvaCrcShared.push_fence(self.shared_name, sample.metadata.frame_id, fence)


class _PvaValidationReportHolder:
    """Plain Python-side stash so run() can read CheckPvaCrcOp's accumulated state back after
    session.wait() returns -- same shape as FrameValidationOp's own _ValidationReportHolder.
    """

    instance = None


class CheckPvaCrcOp(hs.Operator):
    """Retrieve the PVA CRC result (blocks briefly if not yet ready), compare it against the
    camera's own FPGA-embedded CRC, and accumulate match/mismatch statistics.

    A pure sink (side-tap on image_processor's output), not a pass-through, even though the old
    app's own version sat inline in the tensor chain: BayerDemosaicOp's input requires a
    byte-span bound declared by its *direct* producer, which only a real C++-registered operator
    can express (confirmed directly -- Python's own OperatorSpec.output() has no equivalent of
    C++'s produces_tensor(TensorOutputSpec), and inserting any custom Python pass-through here
    hits PHYSICAL_TENSOR_BYTE_SPAN_INPUT_UNBOUNDED at compile time, regardless of port type). So
    ImageProcessorOp feeds BayerDemosaicOp directly, same as every image_processor->demosaic
    wiring elsewhere in this whole porting effort, and this operator taps the same
    image_processor.tensor/metadata output in parallel instead, as a side branch that does
    not sit in the chain.
    """

    in_tensor = hs.Input("in_tensor", hs.schema.Image)
    in_metadata = hs.Input("in_metadata", capture_metadata.CaptureMetadata)

    compute_op_name = hs.Param("")
    frame_limit = hs.Param(
        0
    )  # 0 = unbounded; only trims the "skip last N" validation window.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._frame_count = 0
        self._matches = 0
        self._mismatches = 0
        self._not_checked = 0
        self._output_gpu = None
        _PvaValidationReportHolder.instance = self

    def setup(self, spec):
        spec.input(self.in_tensor, queue_depth=4)
        spec.input(self.in_metadata, queue_depth=4)

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
        # Pre-allocate once: 2x uint32 (dual-VPU output slots), matching the old app exactly.
        self._output_gpu = cp.zeros(2, dtype=cp.uint32)

    def compute(self, context):
        tensor_sample = self.in_tensor.receive()
        metadata_sample = self.in_metadata.receive()
        if tensor_sample.is_missing or metadata_sample.is_missing:
            return

        self._frame_count += 1
        frame = metadata_sample.data
        fence, stale = _PvaCrcShared.take_fence(
            self.compute_op_name, frame.frame_number
        )
        pva_crc = _PvaCrcShared.instance(self.compute_op_name)
        for stale_fence in stale:
            # Launched for a frame that never reached this operator (dropped on the way).
            self._free_fence(pva_crc, stale_fence)
        if fence is None:
            # ComputePvaCrcOp never got this frame: it cannot be checked.
            self._not_checked += 1
            return

        try:
            result = pva_crc.check_results(
                fence, self._output_gpu.data.ptr, timeout_us=PVA_CHECK_TIMEOUT_US
            )
            if result < 0:
                raise RuntimeError(f"PVA CRC check failed with error code {result}")
            if result == 1:
                raise RuntimeError(
                    f"PVA CRC timeout after {PVA_CHECK_TIMEOUT_US / 1000}ms "
                    "(pipeline not pacing frames slower than PVA can compute)"
                )

            # PVA returns standard CRC32; the camera's own embedded CRC is JAMCRC
            # (bit-inverted). Convert PVA's result to JAMCRC for a direct comparison --
            # matches the old app's own conversion exactly.
            pva_crc32 = int(self._output_gpu[0].get())
            pva_frame_crc = ~pva_crc32 & 0xFFFFFFFF

            in_window = self._frame_count > SKIP_INITIAL_FRAMES and (
                not self.frame_limit
                or self._frame_count <= self.frame_limit - SKIP_FINAL_FRAMES
            )
            if in_window:
                if pva_frame_crc == frame.crc:
                    self._matches += 1
                else:
                    self._mismatches += 1
                    logging.error(
                        f"CRC mismatch at frame {self._frame_count}: "
                        f"camera=0x{frame.crc:08x}, pva=0x{pva_frame_crc:08x}"
                    )
        finally:
            self._free_fence(pva_crc, fence)

    def _free_fence(self, pva_crc, fence):
        try:
            pva_crc.free_fence(fence)
        except Exception as e:
            logging.warning(
                f"Error freeing PVA fence on frame {self._frame_count}: {e}"
            )

    def print_report(self):
        validated = self._matches + self._mismatches
        print("\n" + "=" * 80)
        print("PVA CRC VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Total frames: {self._frame_count}")
        print(
            f"Validated frames: {validated} "
            f"(skipped first {SKIP_INITIAL_FRAMES}, last {SKIP_FINAL_FRAMES})"
        )
        print()
        print("PVA Hardware CRC vs Camera FPGA CRC:")
        print(f"  Matches:    {self._matches}")
        print(f"  Mismatches: {self._mismatches}")
        print(
            f"  Not checked (frame dropped before the PVA launch): {self._not_checked}"
        )
        if validated:
            print(f"  Success:    {(self._matches / validated) * 100:.2f}%")
        print()
        print("INTERPRETATION:")
        print("-" * 50)
        if validated and self._mismatches == 0:
            print("SUCCESS! PVA matches camera perfectly (100%)")
            print("   -> PVA hardware CRC validation is working correctly")
        elif validated:
            error_rate = (self._mismatches / validated) * 100
            print(
                f"PVA has {self._mismatches} mismatches ({error_rate:.2f}% error rate)"
            )
            print("   -> Check PVA configuration and data alignment")
        else:
            print("No frames were validated -- nothing to report.")
        print("=" * 80)


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
    camera_mode,
    ibv_name,
    ibv_port,
    expander_configuration,
    pattern,
    skip_reset,
    frame_limit,
    headless,
    fullscreen,
):
    cuda_device_ordinal = 0

    try:
        _import_pva_crc()
    except ImportError as e:
        print(
            "imx274_pva_crc_validation.py needs the pva_crc extension, which this hololink build "
            "does not include: it is built only with HOLOLINK_BUILD_PVA on hosts with the PVA SDK "
            f"and hardware ({e}).",
            file=sys.stderr,
        )
        return 1

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
    if not skip_reset:
        hololink_dev.reset()

    print("Waiting for PTP sync...")
    if not hololink_dev.ptp_synchronize():
        raise ValueError("Failed to synchronize PTP")
    print("PTP synchronized.")

    camera = hl.NativeImx274Sensor(
        channel, expander_configuration=expander_configuration
    )
    if not skip_reset:
        camera.setup_clock()
    camera.configure(camera_mode)
    camera.set_digital_gain_reg(0x4)
    if pattern is not None:
        camera.test_pattern(pattern)

    pixel_format = camera.get_pixel_format()
    bayer_format = camera.get_bayer_format()
    frame_width = camera.get_width()
    frame_height = camera.get_height()

    geometry = hl.resolve_generic_geometry(camera)
    frame_size = (
        geometry["start_byte"]
        + geometry["bytes_per_line"] * geometry["pixel_height"]
        + geometry["trailing_bytes"]
    )

    # Build the graph.
    graph = hs.Graph("imx274_pva_crc_validation")

    rx = graph.op(
        hl.operators.RoceReceiverOp,
        "rx",
        data_channel=channel,
        frame_context=int(cu_context),
        device_start=camera.start,
        device_stop=camera.stop,
        frame_size=frame_size,
        ibv_name=ibv_name,
        ibv_port=ibv_port,
        frame_limit=frame_limit or 0,
        frame_limit_ends_run=True,
    )

    compute_pva_crc = graph.op(
        ComputePvaCrcOp,
        "compute_pva_crc",
        shared_name="compute_pva_crc",
        data_size=frame_size,
        remaining_size=0,
        is_first_chunk=True,
        use_dual_vpu=True,
        start_byte=0,
    )
    graph.add_flow(rx.tensor, compute_pva_crc.in_tensor, queue_depth=4)

    validation = graph.op(
        FrameValidationOp,
        "validation",
        calculated_frame_size=frame_size,
        crc_frame_check=0,  # PVA covers CRC validation in this app; avoid a redundant CPU check.
        frame_limit=frame_limit or 0,
    )
    graph.add_flow(rx.tensor, validation.in_tensor, queue_depth=4)
    graph.add_flow(rx.metadata, validation.in_metadata, queue_depth=4)

    csi = graph.op(
        hl.operators.CsiToBayerOp,
        "csi_to_bayer",
        cuda_device_ordinal=cuda_device_ordinal,
        geometry=geometry,
        bayer_format=bayer_format,
    )
    graph.add_flow(validation.tensor, csi.in_tensor, queue_depth=4)
    graph.add_flow(validation.metadata, csi.in_metadata, queue_depth=4)

    image_processor = graph.op(
        hl.operators.ImageProcessorOp,
        "image_processor",
        pixel_format=pixel_format,
        pixel_width=geometry["pixel_width"],
        pixel_height=geometry["pixel_height"],
        bayer_format=_BAYER_FORMAT_VALUES[bayer_format],
        optical_black=50,  # Optical black value for imx274 (matches this app's own default).
        cuda_device_ordinal=cuda_device_ordinal,
    )
    graph.add_flow(csi.tensor, image_processor.in_tensor, queue_depth=4)
    graph.add_flow(csi.metadata, image_processor.in_metadata, queue_depth=4)

    # Check PVA results after image processing -- hides ~1ms PVA latency behind that other
    # work, matching the old app's own placement and its own comment on why. A side-tap on
    # image_processor's own output, not inline in the tensor chain -- see CheckPvaCrcOp's own
    # class comment for why (BayerDemosaicOp's input needs a byte-span bound only a real
    # C++-registered producer can declare).
    check_pva_crc = graph.op(
        CheckPvaCrcOp,
        "check_pva_crc",
        compute_op_name="compute_pva_crc",
        frame_limit=frame_limit or 0,
    )
    graph.add_flow(image_processor.tensor, check_pva_crc.in_tensor, queue_depth=4)
    graph.add_flow(image_processor.metadata, check_pva_crc.in_metadata, queue_depth=4)

    demosaic = graph.op(
        hv.operators.BayerDemosaicOp,
        "bayer_demosaic",
        bayer_grid_pos=bayer_format,
        generate_alpha=True,
        alpha_value=65535,
    )
    graph.add_flow(image_processor.tensor, demosaic.input, queue_depth=4)

    # image_processor.metadata is not connected: nothing uses it, and an
    # output that is not connected has no queue that can fill.

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
            "framebuffer_srgb": True,
            "title": f"Holoviz - {hololink_ip}",
            "width": frame_width,
            "height": frame_height,
            "fullscreen": fullscreen,
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
        # One frame per received frame.
        trigger={"kind": "layer"},
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
    for operator_path in ("rx", "csi_to_bayer", "image_processor"):
        options.deployment.bind_tensor_output_device(
            operator_path=operator_path,
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
            print(f"  {diag}", file=sys.stderr)
        hololink_dev.stop()
        _cuda_check(cuda_driver.cuDevicePrimaryCtxRelease(cu_device))
        return 1

    print(
        f"Capturing mode={camera_mode} {frame_width}x{frame_height} -- running (Ctrl+C to stop)..."
    )

    with hs.run_async(plan) as session:
        # With --frame-limit, the receiver ends the run after frame_limit frames.
        session.wait()

    hololink_dev.stop()
    _cuda_check(cuda_driver.cuDevicePrimaryCtxRelease(cu_device))

    if _ValidationReportHolder.instance is not None:
        _ValidationReportHolder.instance.print_report()
    if _PvaValidationReportHolder.instance is not None:
        _PvaValidationReportHolder.instance.print_report()

    print("Done.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="IMX274 Frame Validation with PVA CRC")
    parser.add_argument(
        "--camera-mode",
        type=int,
        choices=[0, 1, 2],
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
        "--hololink", default="192.168.0.2", help="IP address of Hololink board"
    )
    parser.add_argument(
        "--log-level", type=int, default=20, help="Python logging level to display"
    )
    parser.add_argument(
        "--ibv-name",
        default="",
        help="IBV device to use (default: the device on the board's network interface)",
    )
    parser.add_argument(
        "--ibv-port",
        type=int,
        default=1,
        help="Port number of IBV device (with --ibv-name)",
    )
    parser.add_argument("--expander-configuration", type=int, default=0, choices=(0, 1))
    parser.add_argument(
        "--pattern", type=int, choices=range(12), help="Configure a test pattern (0-11)"
    )
    parser.add_argument(
        "--skip-reset",
        action="store_true",
        help="Don't call reset on the hololink device",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    logging.info("IMX274 Frame Validation with PVA CRC Hardware")

    return run(
        args.hololink,
        args.camera_mode,
        args.ibv_name,
        args.ibv_port,
        args.expander_configuration,
        args.pattern,
        args.skip_reset,
        args.frame_limit,
        args.headless,
        args.fullscreen,
    )


if __name__ == "__main__":
    sys.exit(main())
