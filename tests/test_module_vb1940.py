# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter-based VB1940 stereo tests.

Mirrors the legacy ``test_vb1940_stereo_roce_naive`` in
``tests/test_vb1940.py`` but drives the new ``hololink_module``
surface end-to-end (per-board PtpPpsOutput, module-side Vb1940Cam,
module-side RoceReceiverOp). Asserts that the FPGA-stamped
per-frame ``timestamp_s + timestamp_ns`` from the two RoCE receivers
fall within 100us of each other, which is only achievable with
PTP-PPS-driven shutter sync wired through correctly.

HSDK5 differences: v2.7 paired the two free-running legs in the graph with a
FrameAlignerOp; here each leg has its own watchdog, and the legs'
(frame_number, timestamp) records are paired after the run, with the same 100us
window. The vsync starts from the
second receiver's device_start callback, once both cameras are started, instead
of from a one-shot operator.
"""

import logging
import math
import threading

import applications
import hololink_module.operators
import hololink_module.sensors.vb1940
import holoscan_vision as hv
import operators
import pytest
import utils

import hololink_module

NS_PER_SEC = 1_000_000_000

# Receiver frame buffers / outstanding RoCE receives (matches the sibling RoCE
# stereo tests).
MAX_PAGES = 5

# The pairing window and the skew assertion share this one bound: a pair whose
# legs differ by more than 100us isn't paired, so a genuine loss of sync surfaces
# as too few pairs.
MAX_SKEW_NS = 100 * 1000  # 100 us


class VsyncStarter:
    """Starts vsync once every camera has started (from the receivers'
    device_start callbacks), so the first VSYNC pulse lands on every camera
    simultaneously and frame-number alignment across the stereo pair is
    preserved."""

    def __init__(self, vsync, cameras):
        self._vsync = vsync
        self._remaining = len(cameras)
        self._lock = threading.Lock()

    def device_start(self, camera):
        def start():
            camera.start()
            with self._lock:
                self._remaining -= 1
                last = self._remaining == 0
            if last:
                self._vsync.start()

        return start


def add_leg(
    test_graph, side, metadata, camera, cu_context, frame_limit, device_start, records
):
    bayer_format = camera.bayer_format()
    converter = hololink_module.CsiToBayerConverter()
    camera.configure_converter(converter)
    receiver = test_graph.op(
        hololink_module.operators.RoceReceiverOp,
        f"receiver_{side}",
        metadata=metadata,
        frame_context=int(cu_context),
        frame_size=converter.get_csi_length(),
        device_start=device_start,
        device_stop=camera.stop,
        pages=MAX_PAGES,
        queue_size=MAX_PAGES,
        frame_limit=frame_limit,
        frame_limit_ends_run=True,
    )
    csi_op = test_graph.op(
        hololink_module.operators.CsiToBayerOp,
        f"csi_to_bayer_{side}",
        cuda_device_ordinal=test_graph.cuda_device_ordinal,
        geometry=converter.geometry(),
        bayer_format=bayer_format.name.lower(),
    )
    test_graph.add_flow(receiver.tensor, csi_op.in_tensor, queue_depth=4)
    test_graph.add_flow(receiver.metadata, csi_op.in_metadata, queue_depth=4)
    test_graph.bind(f"receiver_{side}", "tensor")
    image_proc = test_graph.op(
        hololink_module.operators.ImageProcessorOp,
        f"image_processor_{side}",
        # VB1940 RAW10 optical-black value (matches the C++
        # module_stereo_vb1940_player example).
        optical_black=8,
        bayer_format=bayer_format.value,
        pixel_format=camera.pixel_format().value,
        cuda_device_ordinal=test_graph.cuda_device_ordinal,
    )
    test_graph.add_flow(csi_op.tensor, image_proc.in_tensor, queue_depth=4)
    test_graph.add_flow(csi_op.metadata, image_proc.in_metadata, queue_depth=4)
    test_graph.bind(f"csi_to_bayer_{side}", "tensor")
    demosaic = test_graph.op(
        hv.operators.BayerDemosaicOp,
        f"demosaic_{side}",
        bayer_grid_pos=bayer_format.name.lower(),
        generate_alpha=True,
        alpha_value=65535,
    )
    test_graph.add_flow(image_proc.tensor, demosaic.input, queue_depth=4)
    test_graph.bind(f"image_processor_{side}", "tensor")
    test_graph.bind(f"demosaic_{side}", "output")

    def time_profile(image_timestamp_s, *args):
        # Kept as (timestamp_s, (frame_number, timestamp_ns)) for pairing.
        records.append(
            (image_timestamp_s, (args[-1], round(image_timestamp_s * NS_PER_SEC)))
        )

    profiler = test_graph.op(
        operators.TimeProfiler,
        f"recorder_{side}",
        callback_token=operators.register(time_profile),
    )
    operators.metadata_flow(test_graph, image_proc.metadata, profiler.metadata)
    width, height = camera.width(), camera.height()
    return image_proc, applications.Image(demosaic.output, width, height)


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

    The two legs are paired by timestamp after the run; for each pair the skew
    between the FPGA-stamped first-sample timestamps must stay inside 100us —
    the same bound the legacy ``test_vb1940_stereo_roce_naive`` asserts.
    """
    if not hasattr(hololink_module.operators, "RoceReceiverOp"):
        pytest.skip("RoceReceiverOp isn't built (HOLOLINK_BUILD_ROCE=OFF)")

    adapter = hololink_module.Adapter.get_adapter()
    base_metadata = adapter.wait_for_channel(peer_ip=hololink_address, timeout_s=30.0)

    # Per-sensor metadata clones — Adapter::use_sensor stamps the
    # per-channel address fields and the data_channel locator key.
    left_metadata = hololink_module.EnumerationMetadata(base_metadata)
    adapter.use_sensor(left_metadata, 0)
    right_metadata = hololink_module.EnumerationMetadata(base_metadata)
    adapter.use_sensor(right_metadata, 1)

    left_collected = []
    right_collected = []

    # One HololinkInterface per board (both metadatas resolve to it).
    hololink = hololink_module.HololinkInterfaceV1.get_service(base_metadata)
    with applications.CudaContext() as cuda_context:
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
            vsync_starter = VsyncStarter(ptp_pps, [left_camera, right_camera])

            # Long timeout while the pipeline warms up, short in steady state, long
            # again for drain — the shape the sibling module ports use.
            ready_frame = 15
            initial_timeout = utils.timeout_sequence(
                [
                    (30, ready_frame),
                    # The last 10 frames have the long timeout: a leg can end a
                    # few frames short of frame_limit, and teardown follows.
                    (0.5, max(1, frame_limit - ready_frame - 10)),
                    (30, 1),
                ]
            )
            with utils.leg_watchdogs(
                ("left", "right"),
                initial_timeout=initial_timeout,
                min_taps=utils.frame_floor(frame_limit),
            ) as watchdogs, utils.RunMonitor() as monitor:
                test_graph = applications.TestGraph(
                    "module_vb1940_stereo",
                    cuda_device_ordinal=cuda_context.device_ordinal,
                )
                images = []
                for side, metadata, camera, collected in (
                    ("left", left_metadata, left_camera, left_collected),
                    ("right", right_metadata, right_camera, right_collected),
                ):
                    image_proc, image = add_leg(
                        test_graph,
                        side,
                        metadata,
                        camera,
                        cuda_context.cu_context,
                        frame_limit,
                        vsync_starter.device_start(camera),
                        collected,
                    )
                    images.append(image)
                    watchdog_operator = test_graph.op(
                        operators.WatchdogOp,
                        f"watchdog_{side}",
                        watchdog_token=operators.register(watchdogs[side]),
                    )
                    operators.metadata_flow(
                        test_graph, image_proc.metadata, watchdog_operator.metadata
                    )
                applications.show_images(
                    test_graph,
                    "holoviz",
                    images,
                    headless,
                    "test_module_vb1940_stereo_roce_naive",
                )
                test_graph.run()
        finally:
            hololink.stop()
    monitor.assert_ok()

    # v2.7 also asserted equal left and right counts: its FrameAlignerOp emitted
    # the legs in pairs. Here the legs run on their own and can differ by a frame.
    # The legs share VSYNC, so nearly every frame pairs: a leg can lose a frame,
    # but 80% of frame_limit must pair.
    pairs = [
        (left, right)
        for (_, left), (_, right) in operators.assert_aligned(
            left_collected,
            right_collected,
            MAX_SKEW_NS / NS_PER_SEC,
            min_pairs=math.ceil(frame_limit * 0.8),
            skip_first=ready_frame,
        )
    ]

    # Pairing already rejects a skew above MAX_SKEW_NS, so lost sync shows as too
    # few pairs above; this loop reports the skews (and fails if pairing changes).
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
