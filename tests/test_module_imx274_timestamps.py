# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# Adapter (hololink_module) port of tests/test_imx274_timestamps.py.
#
# The capture path is built entirely on the hololink_module V1 surface
# (Adapter discovery, Imx274Cam, RoceReceiverOp / LinuxReceiverOp,
# CsiToBayerOp), and validation reuses the shared operators.py helpers
# (TimeProfiler, WatchdogOp, pair_by_timestamp).
#
# Notable differences from the legacy test, all driven by the module API:
#   * The legacy test drove examples/imx274_player.main() via mock.patch;
#     there is no Python module_imx274_player, so this test self-composes
#     the pipeline and replicates main()'s bring-up (discovery, camera,
#     reset, PTP sync, configure) inline in run_test / run_stereo_test.
#   * Discovery is Adapter.wait_for_channel(peer_ip, timeout) rather than
#     Enumerator.find_channel; single-interface stereo clones the base
#     EnumerationMetadata and calls Adapter.use_sensor(clone, n).
#   * The module RoceReceiverOp selects its IB device from the peer, so
#     there is no ibv_name parameter; the RoCE-vs-Linux choice is simply
#     which receiver class the application instantiates.
#   * The module folds the sensor reference-clock setup into configure()
#     (via the per-data-plane oscillator), so there is no setup_clock()
#     call; hololink.ptp_synchronize() still runs so the FPGA timestamp
#     clock shares the host's PTP domain.
#
# HSDK5 differences:
#   * HSDK5 frame metadata is a fixed CaptureMetadata struct, and the
#     receivers have no rename_metadata parameter. The renamed-metadata
#     test instead reads the timestamps after CsiToBayerOp, so it checks that
#     the frame metadata goes through a pipeline stage unchanged.
#   * The stereo test gives each leg its own watchdog and pairs the two legs'
#     timestamps after the run, where v2.7 paired them in the graph with
#     FrameAlignerOp. As v2.7's watchdog after the aligner did,
#     operators.assert_aligned() needs a pair within allowable_dt at least
#     every 0.5 s.
#   * The receivers end the run at the frame limit.

import datetime
import logging
import math
import threading

import applications
import hololink_module.operators
import hololink_module.sensors.imx274 as imx274
import holoscan_vision as hv
import operators
import pytest
import utils

import hololink_module

# Seconds to wait for each bootp announcement before giving up.
DISCOVERY_TIMEOUT_S = 30.0

# Convert an integer nanosecond field to fractional seconds.
SEC_PER_NS = 1.0e-9


def _wait_for_channel(module_dir, peer_ip):
    adapter = hololink_module.Adapter.get_adapter()
    if module_dir:
        adapter.set_module_directory(module_dir)
    return adapter.wait_for_channel(peer_ip, DISCOVERY_TIMEOUT_S)


def add_receiver(
    test_graph, name, metadata, camera, camera_mode, cu_context, use_roce, frame_limit
):
    """Add a module receiver, CsiToBayerOp and BayerDemosaicOp for camera; return
    the receiver, the CsiToBayerOp and the demosaiced image to show."""
    camera.set_mode(camera_mode)
    converter = hololink_module.CsiToBayerConverter()
    camera.configure_converter(converter)
    frame_size = converter.get_csi_length()
    logging.info(f"{name} {frame_size=}")
    # The module RoceReceiverOp resolves its IB device from the peer in
    # the enumeration metadata, so (unlike the legacy op) there is no
    # ibv_name/ibv_port to pass; device_start/device_stop arm and disarm
    # the sensor at pipeline start/stop.
    MAX_PAGES = 5
    receiver_kwargs = {}
    if use_roce:
        receiver = hololink_module.operators.RoceReceiverOp
    else:
        receiver = hololink_module.operators.LinuxReceiverOp
        receiver_kwargs["receiver_affinity"] = [2]
    rx = test_graph.op(
        receiver,
        f"receiver{name}",
        metadata=metadata,
        frame_context=int(cu_context),
        frame_size=frame_size,
        device_start=camera.start,
        device_stop=camera.stop,
        pages=MAX_PAGES,
        queue_size=MAX_PAGES,
        frame_limit=frame_limit,
        frame_limit_ends_run=True,
        **receiver_kwargs,
    )
    csi = test_graph.op(
        hololink_module.operators.CsiToBayerOp,
        f"csi_to_bayer{name}",
        cuda_device_ordinal=test_graph.cuda_device_ordinal,
        geometry=converter.geometry(),
        bayer_format=camera.bayer_format().name.lower(),
    )
    test_graph.add_flow(rx.tensor, csi.in_tensor, queue_depth=4)
    test_graph.add_flow(rx.metadata, csi.in_metadata, queue_depth=4)
    test_graph.bind(f"receiver{name}", "tensor")
    demosaic = test_graph.op(
        hv.operators.BayerDemosaicOp,
        f"demosaic{name}",
        bayer_grid_pos=camera.bayer_format().name.lower(),
        generate_alpha=True,
        alpha_value=65535,
    )
    test_graph.add_flow(csi.tensor, demosaic.input, queue_depth=4)
    test_graph.bind(f"csi_to_bayer{name}", "tensor")
    test_graph.bind(f"demosaic{name}", "output")
    return rx, csi, applications.Image(demosaic.output, camera._width, camera._height)


def run_test(
    metadata,
    camera,
    camera_mode,
    use_roce,
    pattern,
    frame_limit,
    headless,
    profile_after_csi_to_bayer=False,
):
    """Bring the board up and run a single-camera timestamp capture.

    Replicates examples/imx274_player.main()'s bring-up on the module API:
    the module folds the sensor reference-clock setup into configure(), so
    (unlike the legacy path) there is no setup_clock() call — but PTP sync
    still runs so the FPGA timestamps share the host's PTP domain.
    """
    logging.info("Initializing.")
    lock = threading.Lock()
    timestamps = []

    def time_profile(
        image_timestamp_s,
        metadata_timestamp_s,
        received_timestamp_s,
        pipeline_timestamp_s,
        frame_number,
    ):
        with lock:
            timestamps.append(
                (
                    image_timestamp_s,
                    metadata_timestamp_s,
                    received_timestamp_s,
                    pipeline_timestamp_s,
                    frame_number,
                )
            )

    initial_timeout = utils.timeout_sequence(
        [(30, 20), (0.5, frame_limit - 40), (30, 1)]
    )
    # CudaContext retains the primary context and releases it on exit even if
    # the body raises, so an early error can't leak it.
    with applications.CudaContext() as cuda_context, utils.Watchdog(
        "watchdog",
        initial_timeout=initial_timeout,
        min_taps=utils.frame_floor(frame_limit),
    ) as watchdog, utils.RunMonitor() as monitor:
        hololink = hololink_module.HololinkInterfaceV1.get_service(metadata)
        hololink.start()
        # SIGABRT skips the finally block below.  Reset first so an aborted
        # capture cannot leave HSB streaming into subsequent stress iterations.
        with utils.AbortHook(hololink.reset):
            try:
                hololink.reset()
                logging.debug("Waiting for PTP sync.")
                if not hololink.ptp_synchronize():
                    logging.error("Failed to synchronize PTP; ignoring.")
                else:
                    logging.debug("PTP synchronized.")
                camera.configure(camera_mode)
                camera.set_digital_gain_reg(0x4)
                camera.test_pattern(pattern)
                test_graph = applications.TestGraph(
                    "imx274_timestamps", cuda_device_ordinal=cuda_context.device_ordinal
                )
                rx, csi, image = add_receiver(
                    test_graph,
                    "",
                    metadata,
                    camera,
                    camera_mode,
                    cuda_context.cu_context,
                    use_roce,
                    frame_limit,
                )
                profiler = test_graph.op(
                    operators.TimeProfiler,
                    "profiler",
                    callback_token=operators.register(time_profile),
                )
                source = csi if profile_after_csi_to_bayer else rx
                operators.metadata_flow(test_graph, source.metadata, profiler.metadata)
                watchdog_operator = test_graph.op(
                    operators.WatchdogOp,
                    "watchdog_operator",
                    watchdog_token=operators.register(watchdog),
                )
                operators.metadata_flow(
                    test_graph, csi.metadata, watchdog_operator.metadata
                )
                applications.show_images(
                    test_graph, "holoviz", [image], headless, "IMX274 timestamps"
                )
                logging.info("Calling run")
                test_graph.run()
            finally:
                hololink.stop()

    monitor.assert_ok()
    return timestamps


def _check_timestamps(timestamps, frame_time, time_limit, max_recv_time, use_roce):
    """Shared assertions for the single-camera timestamp tests.

    The timing-bound assertions are only enforced on the RoCE path. The
    Linux-socket path has no reliable timing (kernel scheduling, packet loss,
    userspace reassembly), so its timings are logged but not asserted — a slow
    Linux run must not fail the test. Frame collection (>= 100 settled frames)
    is still checked on both paths.
    """
    pipeline_dts, receiver_dts = [], []
    metadata_receiver_dts = []
    # Allow for startup times to be a bit longer
    settled_timestamps = timestamps[5:-5]
    assert len(settled_timestamps) >= 100
    last_image_timestamp_s = None
    last_received_timestamp_s = None
    for (
        image_timestamp_s,
        metadata_timestamp_s,
        received_timestamp_s,
        pipeline_timestamp_s,
        frame_number,
    ) in settled_timestamps:
        image_timestamp = datetime.datetime.fromtimestamp(image_timestamp_s).isoformat()
        metadata_timestamp = datetime.datetime.fromtimestamp(
            metadata_timestamp_s
        ).isoformat()
        received_timestamp = datetime.datetime.fromtimestamp(
            received_timestamp_s
        ).isoformat()
        pipeline_timestamp = datetime.datetime.fromtimestamp(
            pipeline_timestamp_s
        ).isoformat()
        pipeline_dt = pipeline_timestamp_s - image_timestamp_s
        logging.debug(
            f"{image_timestamp=} {pipeline_timestamp=} {pipeline_dt=:0.6f} {frame_number=}"
        )
        pipeline_dts.append(round(pipeline_dt, 4))
        receiver_dt = received_timestamp_s - image_timestamp_s
        receiver_dts.append(round(receiver_dt, 4))
        metadata_receiver_dt = received_timestamp_s - metadata_timestamp_s
        logging.debug(
            f"{image_timestamp=} {metadata_timestamp=} {received_timestamp=} {receiver_dt=:0.6f} {metadata_receiver_dt=:0.6f} {frame_number=}"
        )
        metadata_receiver_dts.append(round(metadata_receiver_dt, 4))
        if last_image_timestamp_s is not None:
            time_from_last_image_s = image_timestamp_s - last_image_timestamp_s
            time_from_last_received_s = received_timestamp_s - last_received_timestamp_s
            logging.debug(
                f"{time_from_last_image_s=:.4f} {time_from_last_received_s=:.4f}"
            )
        last_image_timestamp_s = image_timestamp_s
        last_received_timestamp_s = received_timestamp_s

    smallest_time_difference = min(pipeline_dts)
    largest_time_difference = max(pipeline_dts)
    logging.info(f"pipeline {smallest_time_difference=} {largest_time_difference=}")
    #
    smallest_time_difference = min(receiver_dts)
    largest_time_difference = max(receiver_dts)
    logging.info(f"receiver {smallest_time_difference=} {largest_time_difference=}")
    # frame_time is passed in from above and represents the constant time
    # difference between when the frame-start and frame-end messages arrive at
    # the FPGA.  The time we get with the frame data is captured at frame-start
    # time but isn't delivered to us until the frame-end is sent.  For us to
    # check the validity of the timestamp, we check that the timestamp received
    # with the frame, plus this constant offset, is within (time_limit) of the
    # reception time recorded by the host.  Reception time is recorded when the
    # last frame data is transmitted to us.
    if use_roce:
        assert (frame_time + 0) <= smallest_time_difference
        assert smallest_time_difference <= largest_time_difference
        assert largest_time_difference <= (frame_time + time_limit)
    else:
        logging.info("Linux mode: skipping receiver timing assertions.")
    #
    smallest_time_difference = min(metadata_receiver_dts)
    largest_time_difference = max(metadata_receiver_dts)
    average_time_difference = sum(metadata_receiver_dts) / len(metadata_receiver_dts)
    logging.info(
        f"FPGA to full frame received {smallest_time_difference=} {largest_time_difference=}"
    )
    # The time taken from the end of image frame received at HSB fpga to full frame
    # received on IGX should be less than max_recv_time on average.
    if use_roce:
        assert smallest_time_difference <= largest_time_difference
        assert average_time_difference < max_recv_time
    else:
        logging.info("Linux mode: skipping FPGA-to-received timing assertion.")


# frame_time represents the constant time difference between when the
#   frame-start and frame-end messages arrive at the FPGA; for IMX274
#   it takes just under 8ms for a 1080p or almost 16ms for a 4k image.
# time_limit, the acceptable amount of time between when the frame was sent and
#   when we got around to looking at it, is much smaller in the RDMA
#   configuration.
@utils.requires_module_roce
@pytest.mark.skip_unless_ptp
@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode, frame_time, time_limit, max_recv_time",  # noqa: E501
    [
        (
            imx274.Imx274_Mode.IMX274_MODE_3840X2160_60FPS,
            0.015,
            0.004,
            0.0015,
        ),
        (
            imx274.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
            0.0075,
            0.0040,
            0.0015,
        ),
    ],
)
def test_imx274_timestamps(
    camera_mode,
    frame_time,
    time_limit,
    max_recv_time,
    headless,
    hololink_address,
    frame_limit,
    module_dir,
):
    pattern = 10
    metadata = _wait_for_channel(module_dir, hololink_address)
    camera = imx274.Imx274Cam(metadata)
    use_roce = True

    timestamps = run_test(
        metadata,
        camera,
        camera_mode,
        use_roce,
        pattern,
        frame_limit,
        headless,
    )

    _check_timestamps(timestamps, frame_time, time_limit, max_recv_time, use_roce)


@pytest.mark.skip_unless_ptp
@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode, frame_time, time_limit, max_recv_time",  # noqa: E501
    [
        (
            imx274.Imx274_Mode.IMX274_MODE_3840X2160_60FPS,
            0.015,
            0.012,
            0.0015,
        ),
        (
            imx274.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
            0.0075,
            0.0120,
            0.0045,
        ),
    ],
)
def test_imx274_linux_timestamps(
    camera_mode,
    frame_time,
    time_limit,
    max_recv_time,
    headless,
    hololink_address,
    frame_limit,
    module_dir,
):
    pattern = 10
    metadata = _wait_for_channel(module_dir, hololink_address)
    camera = imx274.Imx274Cam(metadata)
    use_roce = False

    timestamps = run_test(
        metadata,
        camera,
        camera_mode,
        use_roce,
        pattern,
        frame_limit,
        headless,
    )

    _check_timestamps(timestamps, frame_time, time_limit, max_recv_time, use_roce)


@utils.requires_module_roce
@pytest.mark.skip_unless_ptp
@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode, frame_time, time_limit, max_recv_time",  # noqa: E501
    [
        (
            imx274.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
            0.0075,
            0.0040,
            0.0015,
        ),
    ],
)
def test_imx274_timestamps_with_renamed_metadata(
    camera_mode,
    frame_time,
    time_limit,
    max_recv_time,
    headless,
    hololink_address,
    frame_limit,
    module_dir,
):
    """Test that the frame metadata still validates after a pipeline stage.

    In v2.7 the receiver stamped every frame-metadata key with a "roce_" prefix
    and TimeProfiler read the prefixed keys. HSDK5 metadata is a fixed struct
    that the receivers can't rename, so TimeProfiler reads the metadata that
    CsiToBayerOp passes on instead of the receiver's.
    """
    pattern = 10
    metadata = _wait_for_channel(module_dir, hololink_address)
    camera = imx274.Imx274Cam(metadata)
    use_roce = True

    timestamps = run_test(
        metadata,
        camera,
        camera_mode,
        use_roce,
        pattern,
        frame_limit,
        headless,
        profile_after_csi_to_bayer=True,
    )

    _check_timestamps(timestamps, frame_time, time_limit, max_recv_time, use_roce)


# Round up so the acceptance window is never smaller than one frame period.
def round_up(f, decimal_places):
    factor = 10**decimal_places
    return math.ceil(f * factor) / factor


# IMX274 cannot hardware-sync, so the acceptance window for a left/right pair
# is one full frame period (1/60 s at 60 FPS).
ONE_FRAME_AT_60FPS = round_up(1.0 / 60, 3)


def run_stereo_test(
    metadata_left,
    camera_left,
    metadata_right,
    camera_right,
    camera_mode,
    allowable_dt,
    frame_limit,
    headless,
):
    logging.info("Initializing.")
    ready_frame = 15  # ignore this many initial frames while the pipeline warms up
    initial_timeout = utils.timeout_sequence(
        # The last 10 frames have the long timeout: a leg can end a few frames
        # short of frame_limit (frames dropped on the way), and teardown follows.
        [(30, ready_frame), (0.5, frame_limit - ready_frame - 10), (30, 1)]
    )
    frame_starts = {"_left": [], "_right": []}
    # CudaContext retains the primary context and releases it on exit even if
    # the body raises (e.g. the watchdog re-raising a timeout in __exit__), so
    # an early error can't leak it.
    with applications.CudaContext() as cuda_context, utils.leg_watchdogs(
        ("left", "right"),
        initial_timeout=initial_timeout,
        min_taps=utils.frame_floor(frame_limit),
    ) as watchdogs, utils.RunMonitor() as monitor:
        # Both sensors on a stereo board share the same control-plane
        # HololinkInterface, so bring the unit up just once.
        hololink = hololink_module.HololinkInterfaceV1.get_service(metadata_left)
        assert hololink is hololink_module.HololinkInterfaceV1.get_service(
            metadata_right
        )
        hololink.start()
        # See run_test: the watchdog's hard-abort path must quiesce HSB
        # before normal Python cleanup becomes impossible.
        with utils.AbortHook(hololink.reset):
            try:
                hololink.reset()
                logging.debug("Waiting for PTP sync.")
                if not hololink.ptp_synchronize():
                    logging.error("Failed to synchronize PTP; ignoring.")
                else:
                    logging.debug("PTP synchronized.")
                # Left = pattern 10, right = pattern 11 (as in the legacy test);
                # configure() also programs each leg's reference clock via its
                # oscillator, so there is no separate setup_clock() call.
                left_pattern, right_pattern = 10, 11
                camera_left.configure(camera_mode)
                camera_left.test_pattern(left_pattern)
                camera_left.set_digital_gain_reg(0x4)
                camera_right.configure(camera_mode)
                camera_right.test_pattern(right_pattern)
                camera_right.set_digital_gain_reg(0x4)
                test_graph = applications.TestGraph(
                    "imx274_stereo_timestamps",
                    cuda_device_ordinal=cuda_context.device_ordinal,
                )
                images = []
                for name, metadata, camera in (
                    ("_left", metadata_left, camera_left),
                    ("_right", metadata_right, camera_right),
                ):
                    rx, csi, image = add_receiver(
                        test_graph,
                        name,
                        metadata,
                        camera,
                        camera_mode,
                        cuda_context.cu_context,
                        True,  # use_roce
                        frame_limit,
                    )
                    images.append(image)

                    def time_profile(
                        image_timestamp_s, *args, starts=frame_starts[name]
                    ):
                        starts.append((image_timestamp_s, args[-1]))

                    profiler = test_graph.op(
                        operators.TimeProfiler,
                        f"profiler{name}",
                        callback_token=operators.register(time_profile),
                    )
                    operators.metadata_flow(test_graph, csi.metadata, profiler.metadata)
                    watchdog_operator = test_graph.op(
                        operators.WatchdogOp,
                        f"watchdog{name}",
                        watchdog_token=operators.register(watchdogs[name[1:]]),
                    )
                    operators.metadata_flow(
                        test_graph, csi.metadata, watchdog_operator.metadata
                    )
                applications.show_images(
                    test_graph,
                    "holoviz",
                    images,
                    headless,
                    "IMX274 stereo timestamps",
                )
                test_graph.run()
            finally:
                hololink.stop()

    monitor.assert_ok()
    # The two IMX274 sensors free-run (no hardware frame-sync), so their
    # frames drift relative to each other; pair them frame-for-frame where
    # the frame-start times differ by at most allowable_dt.
    pairs = operators.assert_aligned(
        frame_starts["_left"],
        frame_starts["_right"],
        allowable_dt,
        min_pairs=utils.frame_floor(frame_limit),
        skip_first=ready_frame,
    )
    for (left_start, _), (right_start, _) in pairs:
        logging.debug(f"{left_start=} {right_start=} dt={right_start - left_start}")


@utils.requires_module_roce
@pytest.mark.skip_unless_ptp
@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode, allowable_dt",  # noqa: E501
    [
        (imx274.Imx274_Mode.IMX274_MODE_1920X1080_60FPS, ONE_FRAME_AT_60FPS),
    ],
)
def test_imx274_stereo_roce_naive(
    camera_mode,
    allowable_dt,
    headless,
    frame_limit,
    hololink_address,
    module_dir,
):
    # A board that hosts a stereo pair exposes the two sensors as two data
    # planes that share one control-plane HololinkInterface; discover it once
    # and clone the metadata per sensor.
    channel_metadata = _wait_for_channel(module_dir, hololink_address)
    adapter = hololink_module.Adapter.get_adapter()

    channel_metadata_left = hololink_module.EnumerationMetadata(channel_metadata)
    adapter.use_sensor(channel_metadata_left, 0)
    channel_metadata_right = hololink_module.EnumerationMetadata(channel_metadata)
    adapter.use_sensor(channel_metadata_right, 1)

    # The two sensors share the board's expander; select output 0 for the
    # left sensor and output 1 for the right one.
    imx274.Imx274Cam.use_expander_configuration(channel_metadata_left, 0)
    imx274.Imx274Cam.use_expander_configuration(channel_metadata_right, 1)
    camera_left = imx274.Imx274Cam(channel_metadata_left)
    camera_right = imx274.Imx274Cam(channel_metadata_right)

    run_stereo_test(
        channel_metadata_left,
        camera_left,
        channel_metadata_right,
        camera_right,
        camera_mode,
        allowable_dt,
        frame_limit,
        headless,
    )
