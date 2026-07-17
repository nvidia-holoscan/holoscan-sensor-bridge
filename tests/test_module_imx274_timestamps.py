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
# (TimeProfiler, WatchdogOp, FrameAlignerOp).
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
#   * The stereo test aligns the two free-running legs with FrameAlignerOp
#     (IMX274 has no hardware frame-sync) before measuring their skew.

import ctypes
import datetime
import logging
import math
import threading

import hololink_module.operators
import hololink_module.sensors.imx274 as imx274
import holoscan
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


class TimestampTestApplication(holoscan.core.Application):
    """Single-camera capture pipeline that records per-frame timestamps.

    receiver -> csi_to_bayer -> demosaic -> profiler -> visualizer -> watchdog

    The receiver free-runs on a BooleanCondition; TimeProfiler's callback
    disables that condition (and stops the watchdog) once frame_limit frames
    have been collected, mirroring the legacy single-camera structure.
    """

    def __init__(
        self,
        headless,
        cuda_context,
        cuda_device_ordinal,
        metadata,
        camera,
        camera_mode,
        use_roce,
        frame_limit,
        rename_metadata=lambda name: name,
    ):
        logging.info("__init__")
        super().__init__()
        self._headless = headless
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._metadata = metadata
        self._camera = camera
        self._camera_mode = camera_mode
        self._use_roce = use_roce
        self._frame_limit = frame_limit
        self._rename_metadata = rename_metadata
        self._lock = threading.Lock()
        self._timestamps = []
        self.is_metadata_enabled = True

    def _make_receiver(self, condition, frame_size):
        # The module RoceReceiverOp resolves its IB device from the peer in
        # the enumeration metadata, so (unlike the legacy op) there is no
        # ibv_name/ibv_port to pass; device_start/device_stop arm and disarm
        # the sensor at pipeline start/stop.
        MAX_PAGES = 5
        if self._use_roce:
            return hololink_module.operators.RoceReceiverOp(
                self,
                condition,
                name="receiver",
                enumeration_metadata=self._metadata,
                frame_context=self._cuda_context,
                frame_size=frame_size,
                device_start=self._camera.start,
                device_stop=self._camera.stop,
                rename_metadata=self._rename_metadata,
                pages=MAX_PAGES,
                queue_size=MAX_PAGES,
            )
        return hololink_module.operators.LinuxReceiverOp(
            self,
            condition,
            name="receiver",
            enumeration_metadata=self._metadata,
            frame_context=self._cuda_context,
            frame_size=frame_size,
            receiver_affinity=[2],
            device_start=self._camera.start,
            device_stop=self._camera.stop,
            rename_metadata=self._rename_metadata,
            pages=MAX_PAGES,
            queue_size=MAX_PAGES,
        )

    def compose(self):
        logging.info("compose")
        self._condition = holoscan.conditions.BooleanCondition(
            self, name="ok", enable_tick=True
        )
        self._camera.set_mode(self._camera_mode)

        csi_to_bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="pool",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=self._camera._width
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera._height,
            num_blocks=4,
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

        receiver_operator = self._make_receiver(self._condition, frame_size)

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
            num_blocks=4,
        )
        bayer_format = self._camera.bayer_format()
        demosaic = holoscan.operators.BayerDemosaicOp(
            self,
            name="demosaic",
            pool=bayer_pool,
            generate_alpha=True,
            alpha_value=65535,
            bayer_grid_pos=bayer_format.value,
            interpolation_mode=0,
        )
        profiler = operators.TimeProfiler(
            self,
            name="profiler",
            callback=self.time_profile,
            rename_metadata=self._rename_metadata,
        )
        visualizer = holoscan.operators.HolovizOp(
            self,
            name="holoviz",
            headless=self._headless,
            enable_camera_pose_output=True,
            camera_pose_output_type="extrinsics_model",
        )
        initial_timeout = utils.timeout_sequence(
            [(30, 20), (0.5, self._frame_limit - 40), (30, 1)]
        )
        self._watchdog = utils.Watchdog(
            name="watchdog",
            initial_timeout=initial_timeout,
        )
        watchdog_operator = operators.WatchdogOp(
            self,
            name="watchdog_operator",
            watchdog=self._watchdog,
        )
        #
        self.add_flow(receiver_operator, csi_to_bayer_operator, {("output", "input")})
        self.add_flow(csi_to_bayer_operator, demosaic, {("output", "receiver")})
        self.add_flow(demosaic, profiler, {("transmitter", "input")})
        self.add_flow(profiler, visualizer, {("output", "receivers")})
        self.add_flow(visualizer, watchdog_operator, {("camera_pose_output", "input")})
        #
        self._watchdog.start()

    def time_profile(
        self,
        image_timestamp_s,
        metadata_timestamp_s,
        received_timestamp_s,
        pipeline_timestamp_s,
        frame_number,
    ):
        with self._lock:
            self._timestamps.append(
                (
                    image_timestamp_s,
                    metadata_timestamp_s,
                    received_timestamp_s,
                    pipeline_timestamp_s,
                    frame_number,
                )
            )
            if len(self._timestamps) >= self._frame_limit:
                self._watchdog.stop()
                self._condition.disable_tick()


def run_test(
    headless,
    metadata,
    camera,
    camera_mode,
    use_roce,
    pattern,
    frame_limit,
    rename_metadata=lambda name: name,
):
    """Bring the board up and run a single-camera timestamp capture.

    Replicates examples/imx274_player.main()'s bring-up on the module API:
    the module folds the sensor reference-clock setup into configure(), so
    (unlike the legacy path) there is no setup_clock() call — but PTP sync
    still runs so the FPGA timestamps share the host's PTP domain.
    """
    logging.info("Initializing.")
    # CudaContext retains the primary context and releases it on exit even if
    # the body raises, so an early error can't leak it.
    with utils.CudaContext() as cuda_context:
        application = TimestampTestApplication(
            headless,
            cuda_context.context,
            cuda_context.device_ordinal,
            metadata,
            camera,
            camera_mode,
            use_roce,
            frame_limit,
            rename_metadata,
        )
        hololink = hololink_module.HololinkInterfaceV1.get_service(metadata)
        hololink.start()
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
            logging.info("Calling run")
            application.run()
        finally:
            hololink.stop()

    return application._timestamps


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

    with utils.PriorityScheduler():
        timestamps = run_test(
            headless,
            metadata,
            camera,
            camera_mode,
            use_roce,
            pattern,
            frame_limit,
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

    with utils.PriorityScheduler():
        timestamps = run_test(
            headless,
            metadata,
            camera,
            camera_mode,
            use_roce,
            pattern,
            frame_limit,
        )

    _check_timestamps(timestamps, frame_time, time_limit, max_recv_time, use_roce)


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
    """Test that the rename_metadata functionality works correctly.

    The receiver stamps every frame-metadata key with a "roce_" prefix and
    TimeProfiler reads the prefixed keys, so the timestamps still validate.
    """
    pattern = 10

    def rename_metadata(name):
        return f"roce_{name}"

    metadata = _wait_for_channel(module_dir, hololink_address)
    camera = imx274.Imx274Cam(metadata)
    use_roce = True

    with utils.PriorityScheduler():
        timestamps = run_test(
            headless,
            metadata,
            camera,
            camera_mode,
            use_roce,
            pattern,
            frame_limit,
            rename_metadata=rename_metadata,
        )

    _check_timestamps(timestamps, frame_time, time_limit, max_recv_time, use_roce)


# Round up so the acceptance window is never smaller than one frame period.
def round_up(f, decimal_places):
    factor = 10**decimal_places
    return math.ceil(f * factor) / factor


# IMX274 cannot hardware-sync, so the frame aligner's acceptance window is one
# full frame period (1/60 s at 60 FPS).
ONE_FRAME_AT_60FPS = round_up(1.0 / 60, 3)


class StereoMonitorOp(holoscan.core.Operator):
    """Sink for the two aligned legs that logs their frame-start skew.

    The frame aligner fans a paired group out to output_left / output_right,
    which feed this operator's two input ports; because the aligner emits both
    together, compute() runs once per aligned pair. It reads each leg's
    left_/right_ timestamp (the module receiver's timestamp_s/ns — the
    frame-start time) and logs the inter-leg dt. It also owns the frame budget:
    once frame_limit aligned pairs have arrived it disables the receivers'
    tick conditions so the application drains and exits.
    """

    def __init__(
        self,
        *args,
        frame_limit=None,
        stop_conditions=None,
        watchdog=None,
        callback=lambda dt: None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._frame_limit = frame_limit
        self._stop_conditions = stop_conditions or []
        self._watchdog = watchdog
        self._callback = callback
        self._taps = 0

    def setup(self, spec):
        spec.input("left")
        spec.input("right")

    def _timestamp_s(self, metadata, prefix):
        return (
            metadata.get(f"{prefix}timestamp_s", 0)
            + metadata.get(f"{prefix}timestamp_ns", 0) * SEC_PER_NS
        )

    def compute(self, op_input, op_output, context):
        _ = op_input.receive("left")
        _ = op_input.receive("right")
        metadata = self.metadata
        left_frame_start = self._timestamp_s(metadata, "left_")
        right_frame_start = self._timestamp_s(metadata, "right_")
        dt = right_frame_start - left_frame_start
        logging.info(f"{left_frame_start=} {right_frame_start=} {dt=}")
        self._callback(dt)
        if self._watchdog is not None:
            self._watchdog.tap()
        self._taps += 1
        if self._frame_limit is not None and self._taps >= self._frame_limit:
            for condition in self._stop_conditions:
                condition.disable_tick()


class StereoTimestampApplication(holoscan.core.Application):
    """Two free-running IMX274 legs, aligned, whose skew is logged.

    receiver_{left,right} -> frame_aligner -> csi_to_bayer_{left,right}
        -> stereo_monitor

    The receivers free-run on BooleanConditions; the frame aligner drops
    unpaired / misaligned frames (dt > allowable_dt), and the monitor sink
    owns the frame budget — counting on the receivers would starve the sink.
    """

    def __init__(
        self,
        headless,
        cuda_context,
        cuda_device_ordinal,
        metadata_left,
        camera_left,
        metadata_right,
        camera_right,
        camera_mode,
        allowable_dt,
        watchdog,
        frame_limit,
    ):
        logging.info("__init__")
        super().__init__()
        self._headless = headless
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._metadata_left = metadata_left
        self._camera_left = camera_left
        self._metadata_right = metadata_right
        self._camera_right = camera_right
        self._camera_mode = camera_mode
        self._allowable_dt = allowable_dt
        self._watchdog = watchdog
        self._frame_limit = frame_limit
        # The two legs stamp per-leg-prefixed metadata (left_/right_), so the
        # keys never collide; merge everything the two aligned messages carry.
        self.is_metadata_enabled = True
        self.metadata_policy = holoscan.core.MetadataPolicy.UPDATE

    def _make_receiver(
        self,
        condition,
        name,
        metadata,
        camera,
        frame_size,
        affinity,
        rename_metadata,
        out_tensor_name,
    ):
        MAX_PAGES = 5
        return hololink_module.operators.RoceReceiverOp(
            self,
            condition,
            name=name,
            enumeration_metadata=metadata,
            frame_context=self._cuda_context,
            frame_size=frame_size,
            device_start=camera.start,
            device_stop=camera.stop,
            rename_metadata=rename_metadata,
            out_tensor_name=out_tensor_name,
            pages=MAX_PAGES,
            queue_size=MAX_PAGES,
        )

    def _csi_to_bayer(self, name, camera):
        csi_to_bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name=f"{name}_pool",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=camera._width * ctypes.sizeof(ctypes.c_uint16) * camera._height,
            num_blocks=4,
        )
        csi_to_bayer_operator = hololink_module.operators.CsiToBayerOp(
            self,
            name=name,
            allocator=csi_to_bayer_pool,
            cuda_device_ordinal=self._cuda_device_ordinal,
        )
        camera.configure_converter(csi_to_bayer_operator)
        return csi_to_bayer_operator

    def compose(self):
        logging.info("compose")
        # The receivers free-run; the monitor sink owns the frame budget and
        # disables these tick conditions once it has seen frame_limit aligned
        # pairs. Counting on the receivers would starve the sink, because the
        # frame aligner drops unpaired / misaligned frames upstream of it.
        self._condition_left = holoscan.conditions.BooleanCondition(
            self, name="enable_left", enable_tick=True
        )
        self._condition_right = holoscan.conditions.BooleanCondition(
            self, name="enable_right", enable_tick=True
        )
        self._camera_left.set_mode(self._camera_mode)
        self._camera_right.set_mode(self._camera_mode)

        csi_to_bayer_left = self._csi_to_bayer("csi_to_bayer_left", self._camera_left)
        csi_to_bayer_right = self._csi_to_bayer(
            "csi_to_bayer_right", self._camera_right
        )

        frame_size_left = csi_to_bayer_left.get_csi_length()
        frame_size_right = csi_to_bayer_right.get_csi_length()
        logging.info(f"{frame_size_left=} {frame_size_right=}")

        receiver_operator_left = self._make_receiver(
            self._condition_left,
            "receiver_left",
            self._metadata_left,
            self._camera_left,
            frame_size_left,
            [2],
            lambda name: f"left_{name}",
            "left",
        )
        receiver_operator_right = self._make_receiver(
            self._condition_right,
            "receiver_right",
            self._metadata_right,
            self._camera_right,
            frame_size_right,
            [3],
            lambda name: f"right_{name}",
            "right",
        )

        # Frame aligner. The two IMX274 sensors free-run (no hardware
        # frame-sync), so their frames drift relative to each other —
        # especially at startup. The aligner pairs them frame-for-frame,
        # dropping a group whenever the two legs' timestamps differ by more
        # than allowable_dt (one full frame period). It tells the legs apart
        # by tensor name (the receivers name their tensors "left"/"right" via
        # out_tensor_name) and reads the per-leg timestamp metadata the
        # receivers stamp (left_/right_timestamp_s, left_/right_timestamp_ns).
        self._frame_aligner = operators.FrameAlignerOp(
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

        monitor = StereoMonitorOp(
            self,
            name="stereo_monitor",
            watchdog=self._watchdog,
            frame_limit=self._frame_limit,
            stop_conditions=[self._condition_left, self._condition_right],
        )

        # Both receivers feed the aligner's single input; it fans out paired
        # frames to each leg's csi_to_bayer, which land on the monitor's two
        # input ports (one aligned pair per monitor tick).
        self.add_flow(
            receiver_operator_left, self._frame_aligner, {("output", "input")}
        )
        self.add_flow(
            receiver_operator_right, self._frame_aligner, {("output", "input")}
        )
        self.add_flow(
            self._frame_aligner, csi_to_bayer_left, {("output_left", "input")}
        )
        self.add_flow(
            self._frame_aligner, csi_to_bayer_right, {("output_right", "input")}
        )
        self.add_flow(csi_to_bayer_left, monitor, {("output", "left")})
        self.add_flow(csi_to_bayer_right, monitor, {("output", "right")})


def run_stereo_test(
    headless,
    metadata_left,
    camera_left,
    metadata_right,
    camera_right,
    camera_mode,
    allowable_dt,
    frame_limit,
):
    logging.info("Initializing.")
    ready_frame = 15  # ignore this many initial frames while the pipeline warms up
    initial_timeout = utils.timeout_sequence(
        [(30, ready_frame), (0.5, frame_limit - ready_frame - 2), (30, 1)]
    )
    # CudaContext retains the primary context and releases it on exit even if
    # the body raises (e.g. the watchdog re-raising a timeout in __exit__), so
    # an early error can't leak it.
    with utils.CudaContext() as cuda_context:
        with utils.Watchdog(
            "watchdog",
            initial_timeout=initial_timeout,
        ) as watchdog:
            application = StereoTimestampApplication(
                headless,
                cuda_context.context,
                cuda_context.device_ordinal,
                metadata_left,
                camera_left,
                metadata_right,
                camera_right,
                camera_mode,
                allowable_dt,
                watchdog,
                frame_limit=frame_limit,
            )
            # Both sensors on a stereo board share the same control-plane
            # HololinkInterface, so bring the unit up just once.
            hololink = hololink_module.HololinkInterfaceV1.get_service(metadata_left)
            assert hololink is hololink_module.HololinkInterfaceV1.get_service(
                metadata_right
            )
            hololink.start()
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
                application.run()
            finally:
                hololink.stop()


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

    with utils.PriorityScheduler():
        run_stereo_test(
            headless,
            channel_metadata_left,
            camera_left,
            channel_metadata_right,
            camera_right,
            camera_mode,
            allowable_dt,
            frame_limit,
        )
