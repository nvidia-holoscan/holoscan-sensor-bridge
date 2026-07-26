# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter-based VB1940 stereo tests.

Mirrors the legacy ``test_vb1940_stereo_roce_naive`` in
``tests/test_vb1940.py:209`` but drives the new ``hololink_module``
surface end-to-end (per-board PtpPpsOutput, module-side Vb1940Cam,
module-side RoceReceiverOp). Asserts that the FPGA-stamped
per-frame ``timestamp_s + timestamp_ns`` from the two RoCE receivers
fall within 100us of each other, which is only achievable with
PTP-PPS-driven shutter sync wired through correctly.

The two free-running legs are paired by a FrameAlignerOp (as in
tests/test_module_coe.py): it drops any group whose legs' timestamps
differ by more than the skew bound, so an occasional 1-frame arrival
offset between channels can't shift the comparison.
"""

import ctypes
import logging
import threading

import hololink_module.operators
import hololink_module.sensors.vb1940
import holoscan
import holoscan.operators
import operators
import pytest
import utils

import hololink_module

NS_PER_SEC = 1_000_000_000

# Receiver frame buffers / outstanding RoCE receives (matches the sibling RoCE
# stereo tests).
MAX_PAGES = 5

# The frame aligner's acceptance window and the skew assertion share this one
# bound: a pair whose legs differ by more than 100us is dropped rather than
# measured, so a genuine loss of sync surfaces as "no aligned pairs".
MAX_SKEW_NS = 100 * 1000  # 100 us


class TimestampRecorderOp(operators.PassThroughOperator):
    """Inline tap: records one aligned leg's (frame_number, timestamp_ns) from
    its prefix-scoped metadata, then forwards the frame unchanged."""

    def __init__(self, *args, prefix, collected, lock, **kwargs):
        super().__init__(*args, **kwargs)
        self._prefix = prefix
        self._collected = collected
        self._lock = lock

    def compute(self, op_input, op_output, context):
        super().compute(op_input, op_output, context)
        metadata = self.metadata
        ts_ns = (
            metadata[f"{self._prefix}timestamp_s"] * NS_PER_SEC
            + metadata[f"{self._prefix}timestamp_ns"]
        )
        frame_number = metadata[f"{self._prefix}frame_number"]
        with self._lock:
            self._collected.append((frame_number, ts_ns))


class VsyncStartOp(holoscan.core.Operator):
    """One-shot operator that fires vsync.start() once after every
    receiver has finished its device_start callback (and therefore
    every Vb1940Cam is in external-sync waiting state). Holoscan
    invokes compute() only after all operators have entered the
    started state, so the first VSYNC pulse lands on every camera
    simultaneously and frame-number alignment across the stereo pair
    is preserved."""

    def __init__(self, *args, vsync, **kwargs):
        super().__init__(*args, **kwargs)
        self._vsync = vsync

    def setup(self, spec):
        pass

    def compute(self, op_input, op_output, context):
        self._vsync.start()


class StereoLatencyApplication(holoscan.core.Application):
    """Drives both VB1940 cameras through a FrameAlignerOp into a side-by-side
    HolovizOp, recording each aligned leg's timestamp inline for the latency
    assertion. ``headless`` only controls whether HolovizOp opens a window.

    receiver_{L,R} -> frame_aligner -> csi_to_bayer_{L,R} -> image_processor_{L,R}
        -> demosaic_{L,R} -> recorder_{L,R} -> holoviz -> watchdog (owns budget)"""

    def __init__(
        self,
        *,
        headless,
        cuda_context,
        cuda_device_ordinal,
        left_metadata,
        right_metadata,
        left_camera,
        right_camera,
        frame_context,
        frame_limit,
        allowable_dt,
        watchdog,
        left_collected,
        right_collected,
        lock,
        vsync,
    ):
        super().__init__()
        self._headless = headless
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._left_metadata = left_metadata
        self._right_metadata = right_metadata
        self._left_camera = left_camera
        self._right_camera = right_camera
        self._frame_context = frame_context
        self._frame_limit = frame_limit
        self._allowable_dt = allowable_dt
        self._watchdog = watchdog
        self._vsync = vsync
        self._left_collected = left_collected
        self._right_collected = right_collected
        self._lock = lock
        self.enable_metadata(True)
        # Each leg stamps per-leg-prefixed metadata (left_/right_), so the keys
        # never collide when the aligned pair converges at HolovizOp.
        self.metadata_policy = holoscan.core.MetadataPolicy.UPDATE

    def compose(self):
        # Both stereo cameras share width / height / formats.
        camera_width = self._left_camera.width()
        camera_height = self._left_camera.height()
        bayer_format = self._left_camera.bayer_format()
        pixel_format = self._left_camera.pixel_format()
        rgba_components_per_pixel = 4

        # The receivers free-run; the watchdog sink owns the frame budget and
        # disables these once frame_limit aligned frames have arrived (counting
        # on the receivers would starve the sink by the aligner's drop count).
        conditions = {
            side: holoscan.conditions.BooleanCondition(
                self, name=f"enable_{side}", enable_tick=True
            )
            for side in ("left", "right")
        }

        # Pairs the legs by tensor name / timestamp and drops any group skewed by
        # more than allowable_dt, then fans each aligned pair down the viz chains.
        frame_aligner = operators.FrameAlignerOp(
            self,
            name="frame_aligner",
            allowable_dt=self._allowable_dt,
            input_tensors=["left", "right"],
            rename_metadata=[
                lambda name: f"left_{name}",
                lambda name: f"right_{name}",
            ],
            outputs=["output_left", "output_right"],
        )

        sides = (
            ("left", self._left_metadata, self._left_camera, self._left_collected, 0.0),
            (
                "right",
                self._right_metadata,
                self._right_camera,
                self._right_collected,
                0.5,
            ),
        )

        tensor_specs = []
        recorders = []
        for side, metadata, camera, collected, offset_x in sides:
            csi_pool = holoscan.resources.BlockMemoryPool(
                self,
                name=f"csi_pool_{side}",
                storage_type=1,  # device memory
                block_size=camera_width
                * ctypes.sizeof(ctypes.c_uint16)
                * camera_height,
                num_blocks=8,
            )
            csi_op = hololink_module.operators.CsiToBayerOp(
                self,
                name=f"csi_to_bayer_{side}",
                allocator=csi_pool,
                cuda_device_ordinal=self._cuda_device_ordinal,
                out_tensor_name=side,
            )
            camera.configure_converter(csi_op)

            receiver = hololink_module.operators.RoceReceiverOp(
                self,
                conditions[side],
                name=f"receiver_{side}",
                enumeration_metadata=metadata,
                frame_context=self._frame_context,
                frame_size=csi_op.get_csi_length(),
                device_start=camera.start,
                device_stop=camera.stop,
                # Per-leg prefix + tensor name so the aligner can tell legs apart.
                rename_metadata=lambda name, s=side: f"{s}_{name}",
                out_tensor_name=side,
                pages=MAX_PAGES,
                queue_size=MAX_PAGES,
            )

            image_proc = hololink_module.operators.ImageProcessorOp(
                self,
                name=f"image_processor_{side}",
                # VB1940 RAW10 optical-black value (matches the C++
                # module_stereo_vb1940_player example).
                optical_black=8,
                bayer_format=bayer_format.value,
                pixel_format=pixel_format.value,
            )
            bayer_pool = holoscan.resources.BlockMemoryPool(
                self,
                name=f"bayer_pool_{side}",
                storage_type=1,
                block_size=camera_width
                * rgba_components_per_pixel
                * ctypes.sizeof(ctypes.c_uint16)
                * camera_height,
                num_blocks=8,
            )
            demosaic = holoscan.operators.BayerDemosaicOp(
                self,
                name=f"demosaic_{side}",
                pool=bayer_pool,
                generate_alpha=True,
                alpha_value=65535,
                bayer_grid_pos=bayer_format.value,
                interpolation_mode=0,
                in_tensor_name=side,
                out_tensor_name=side,
            )
            recorder = TimestampRecorderOp(
                self,
                name=f"recorder_{side}",
                prefix=f"{side}_",
                collected=collected,
                lock=self._lock,
                in_tensor_name=side,
                out_tensor_name=side,
            )
            recorders.append(recorder)

            spec = holoscan.operators.HolovizOp.InputSpec(
                side, holoscan.operators.HolovizOp.InputType.COLOR
            )
            view = holoscan.operators.HolovizOp.InputSpec.View()
            view.offset_x = offset_x
            view.offset_y = 0.0
            view.width = 0.5
            view.height = 1.0
            spec.views = [view]
            tensor_specs.append(spec)

            self.add_flow(receiver, frame_aligner, {("output", "input")})
            self.add_flow(frame_aligner, csi_op, {(f"output_{side}", "input")})
            self.add_flow(csi_op, image_proc, {("output", "input")})
            self.add_flow(image_proc, demosaic, {("output", "receiver")})
            self.add_flow(demosaic, recorder, {("transmitter", "input")})

        visualizer = holoscan.operators.HolovizOp(
            self,
            name="holoviz",
            headless=self._headless,
            framebuffer_srgb=True,
            tensors=tensor_specs,
            window_title="test_module_vb1940_stereo_roce_naive",
            enable_camera_pose_output=True,
            camera_pose_output_type="extrinsics_model",
        )
        for recorder in recorders:
            self.add_flow(recorder, visualizer, {("output", "receivers")})

        # The sink owns the frame budget: it ticks once per aligned frame
        # reaching the visualizer and, after frame_limit, disables the receivers
        # so the application drains and exits.
        watchdog_count = holoscan.conditions.CountCondition(
            self, name="watchdog_count", count=self._frame_limit
        )
        watchdog_operator = operators.WatchdogOp(
            self,
            watchdog_count,
            name="watchdog_operator",
            watchdog=self._watchdog,
            frame_limit=self._frame_limit,
            stop_conditions=list(conditions.values()),
        )
        self.add_flow(visualizer, watchdog_operator, {("camera_pose_output", "input")})

        # One-shot operator: fires vsync.start() after holoscan has finished
        # starting every receiver (which serially runs each camera's bring-up
        # via the device_start callback).
        vsync_starter = VsyncStartOp(
            self,
            holoscan.conditions.CountCondition(self, name="vsync_start_count", count=1),
            name="vsync_start",
            vsync=self._vsync,
        )
        self.add_operator(vsync_starter)


@pytest.mark.accelerated_networking
@pytest.mark.skip_unless_vb1940
@pytest.mark.skip_unless_ptp
@pytest.mark.parametrize(
    "camera_mode, frames_per_second",
    [
        (
            hololink_module.sensors.vb1940.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS,
            30,
        ),
    ],
)
def test_module_vb1940_stereo_roce_naive(
    camera_mode,
    frames_per_second,
    headless,
    frame_limit,
    hololink_address,
):
    """End-to-end stereo latency test for the module path.

    A FrameAlignerOp pairs the two free-running legs by timestamp and drops the
    occasional off-by-one-frame group. For each aligned pair the skew between
    the FPGA-stamped first-sample timestamps must stay inside 100us — the same
    bound the legacy ``test_vb1940_stereo_roce_naive`` asserts, and the aligner
    window, so a lost sync surfaces as "no aligned pairs" rather than a skew
    violation.
    """
    import cuda.bindings.driver as cuda

    # Bring up CUDA.
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS, err
    cuda_device_ordinal = 0
    err, cu_device = cuda.cuDeviceGet(cuda_device_ordinal)
    assert err == cuda.CUresult.CUDA_SUCCESS, err
    err, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
    assert err == cuda.CUresult.CUDA_SUCCESS, err

    adapter = hololink_module.Adapter.get_adapter()
    base_metadata = adapter.wait_for_channel(peer_ip=hololink_address, timeout_s=30.0)

    # Per-sensor metadata clones — Adapter::use_sensor stamps the
    # per-channel address fields and the data_channel locator key.
    left_metadata = hololink_module.EnumerationMetadata(base_metadata)
    adapter.use_sensor(left_metadata, 0)
    right_metadata = hololink_module.EnumerationMetadata(base_metadata)
    adapter.use_sensor(right_metadata, 1)

    # One HololinkInterface per board (both metadatas resolve to it).
    hololink = hololink_module.HololinkInterfaceV1.get_service(base_metadata)
    hololink.start()
    try:
        hololink.reset()

        # VSYNC derives from the FPGA PTP clock, so sync it before enabling.
        if not hololink.ptp_synchronize():
            logging.error("Failed to synchronize PTP; ignoring.")

        # PTP-PPS-driven VSYNC. Both Vb1940Cam ctors take this as the
        # `vsync=` argument; their configure() flips 0xAC6 to 0x01
        # (external sync) instead of the master-mode default.
        ptp_pps = hololink.ptp_pps_output()
        ptp_pps.enable(frames_per_second)
        assert ptp_pps.is_enabled()

        left_camera = hololink_module.sensors.vb1940.Vb1940Cam(
            left_metadata, vsync=ptp_pps
        )
        right_camera = hololink_module.sensors.vb1940.Vb1940Cam(
            right_metadata, vsync=ptp_pps
        )
        left_camera.configure(camera_mode)
        right_camera.configure(camera_mode)

        lock = threading.Lock()
        # (frame_number, timestamp_ns) per leg, in aligner-emission order.
        left_collected = []
        right_collected = []

        # Long timeout while the pipeline warms up, short in steady state, long
        # again for drain — the shape the sibling module ports use.
        ready_frame = 15
        initial_timeout = utils.timeout_sequence(
            [(30, ready_frame), (0.5, max(1, frame_limit - ready_frame - 2)), (30, 1)]
        )
        with utils.PriorityScheduler(), utils.Watchdog(
            "watchdog", initial_timeout=initial_timeout
        ) as watchdog:
            application = StereoLatencyApplication(
                headless=headless,
                cuda_context=cu_context,
                cuda_device_ordinal=cuda_device_ordinal,
                left_metadata=left_metadata,
                right_metadata=right_metadata,
                left_camera=left_camera,
                right_camera=right_camera,
                frame_context=cu_context,
                frame_limit=frame_limit,
                allowable_dt=MAX_SKEW_NS / NS_PER_SEC,  # aligner compares in seconds
                watchdog=watchdog,
                left_collected=left_collected,
                right_collected=right_collected,
                lock=lock,
                vsync=ptp_pps,
            )
            application.run()
    finally:
        hololink.stop()

    # The aligner emits synced pairs in lockstep, so the i-th entry of each leg
    # is one same-shutter pair — index-align them (no frame_number match). A
    # length mismatch means a recorder dropped a leg, which zip() would hide by
    # truncating the tail, so assert equal counts before pairing.
    assert len(left_collected) == len(right_collected), (
        f"recorder leg count mismatch: left={len(left_collected)} "
        f"right={len(right_collected)}"
    )
    pairs = list(zip(left_collected, right_collected))
    assert pairs, "stereo latency test produced no aligned frame pairs"

    limit_ns = MAX_SKEW_NS
    limit_ms = limit_ns / 1_000_000.0
    # Comparison stays in nanoseconds for precision; the display
    # values below convert to milliseconds at the boundary.
    over_limit = []
    for (left_frame, left_ts_ns), (right_frame, right_ts_ns) in pairs:
        dt_ns = abs(left_ts_ns - right_ts_ns)
        dt_ms = dt_ns / 1_000_000.0
        # Intermediate per-frame measurement — only visible at debug
        # log level.
        hololink_module.hsb_log_debug(
            f"left_frame={left_frame} right_frame={right_frame} "
            f"left_ts_ms={left_ts_ns / 1_000_000.0:.3f} "
            f"right_ts_ms={right_ts_ns / 1_000_000.0:.3f} "
            f"dt_ms={dt_ms:.3f}"
        )
        if dt_ns > limit_ns:
            # Failure-causing sample — visible at info level so the
            # user sees the offending frames without re-running.
            hololink_module.hsb_log_info(
                f"left_frame={left_frame} right_frame={right_frame} "
                f"dt_ms={dt_ms:.3f} exceeds limit {limit_ms:.3f} ms"
            )
            over_limit.append((left_frame, right_frame, dt_ms))
    assert not over_limit, (
        f"{len(over_limit)} frame(s) exceeded {limit_ms:.3f}ms skew: "
        f"{[(lf, rf, f'{d:.3f}') for lf, rf, d in over_limit[:5]]}"
    )
