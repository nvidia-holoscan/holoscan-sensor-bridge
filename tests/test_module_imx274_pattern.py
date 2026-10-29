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
# Adapter (hololink_module) port of tests/test_imx274_pattern.py.
#
# The capture path is built entirely on the hololink_module V1 surface
# (Adapter discovery, Imx274Cam, RoceReceiverOp / LinuxReceiverOp,
# CsiToBayerOp), and validation uses the module's own CRC operators
# (hololink_module.operators.ComputeCrcOp / CheckCrcOp), plus the shared
# test helpers in operators.py.
#
# Notable differences from the legacy test, all driven by the module API:
#   * Discovery is Adapter.wait_for_channel(peer_ip, timeout) rather than
#     Enumerator.find_channel; single-interface stereo clones the base
#     EnumerationMetadata and calls Adapter.use_sensor(clone, n).
#   * The module RoceReceiverOp selects its IB device from the peer, so
#     there is no ibv_name parameter; the RoCE-vs-Linux choice is simply
#     which receiver class the application instantiates.

import ctypes
import dataclasses
import gc
import logging
import math
import weakref

import cuda.bindings.driver as cuda
import hololink_module.operators
import hololink_module.sensors.imx274 as imx274
import holoscan
import operators
import pytest
import utils

import hololink_module

# Number of frames to skip during pipeline initialization to avoid artifacts
SKIP_INITIAL_FRAMES = 15

# Seconds to wait for each bootp announcement before giving up.
DISCOVERY_TIMEOUT_S = 30.0


# Each receiver renames its emitted frame metadata with a per-leg prefix
# (rename_metadata=lambda name: f"left_"/"right_" + name), so the received
# "crc", "imm_data", and "bytes_written" fields arrive as left_/right_
# variants. The bayer and received-CRC comparison fields are stamped by the
# CRC operators under per-side names (check_crc_left, bayer_crc_left, …).
@dataclasses.dataclass
class LeftMetadataRecord:
    left_crc: int
    check_crc_left: int
    bayer_crc_left: int
    left_imm_data: int
    left_bytes_written: int


@dataclasses.dataclass
class LeftRoceMetadataRecord(LeftMetadataRecord):
    left_flags: int
    left_frame_number: int


@dataclasses.dataclass
class RightMetadataRecord:
    right_crc: int
    check_crc_right: int
    bayer_crc_right: int
    right_imm_data: int
    right_bytes_written: int


@dataclasses.dataclass
class RightRoceMetadataRecord(RightMetadataRecord):
    right_flags: int
    right_frame_number: int


class CameraWrapper(imx274.Imx274Cam):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reset_callbacks = 0
        # Register a reset callback that does NOT strongly reference this
        # camera (a weak method) and keep the returned RAII handle. The
        # per-board HololinkInterface is a process-lifetime singleton, so a
        # strong registration would pin this camera (and keep firing its
        # callback) for the rest of the session. Holding the handle means
        # the registration is dropped when this camera is garbage-collected.
        weak_reset = weakref.WeakMethod(self._reset)

        def _on_reset():
            method = weak_reset()
            if method is not None:
                method()

        self._reset_registration = self._hololink.on_reset(_on_reset)

    def _reset(self):
        self._reset_callbacks += 1
        logging.info(f"{self._reset_callbacks=}")


class PatternTestApplication(holoscan.core.Application):
    def __init__(
        self,
        headless,
        cuda_context,
        cuda_device_ordinal,
        metadata_left,
        camera_left,
        camera_mode_left,
        metadata_right,
        camera_right,
        camera_mode_right,
        use_roce,
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
        self._camera_mode_left = camera_mode_left
        self._metadata_right = metadata_right
        self._camera_right = camera_right
        self._camera_mode_right = camera_mode_right
        self._use_roce = use_roce
        self._allowable_dt = allowable_dt
        self._watchdog = watchdog
        self._frame_limit = frame_limit
        # These are HSDK controls-- because we have stereo
        # camera paths going into the same visualizer, don't
        # raise an error when each path presents metadata
        # with the same names. Because we don't use that metadata,
        # it's easiest to just ignore new items with the same
        # names as existing items.
        self.is_metadata_enabled = True
        self.metadata_policy = holoscan.core.MetadataPolicy.REJECT

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
        # The module RoceReceiverOp resolves its IB device from the
        # peer in the enumeration metadata, so (unlike the legacy op)
        # there is no ibv_name/ibv_port to pass; device_start/device_stop
        # arm and disarm the sensor at pipeline start/stop. rename_metadata
        # adds a per-leg prefix to the emitted frame metadata so the two
        # legs don't collide when both feed the visualizer.
        MAX_PAGES = 5
        if self._use_roce:
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
        return hololink_module.operators.LinuxReceiverOp(
            self,
            condition,
            name=name,
            enumeration_metadata=metadata,
            frame_context=self._cuda_context,
            frame_size=frame_size,
            receiver_affinity=affinity,
            device_start=camera.start,
            device_stop=camera.stop,
            rename_metadata=rename_metadata,
            out_tensor_name=out_tensor_name,
            pages=MAX_PAGES,
            queue_size=MAX_PAGES,
        )

    def compose(self):
        logging.info("compose")
        # The receivers free-run; the sink (watchdog_operator) owns the frame
        # budget and disables these tick conditions once it has seen
        # frame_limit aligned frames. Counting on the receivers instead would
        # starve the sink, because the frame aligner drops unpaired/misaligned
        # frames between the receivers and the sink.
        self._condition_left = holoscan.conditions.BooleanCondition(
            self,
            name="enable_left",
            enable_tick=True,
        )
        self._condition_right = holoscan.conditions.BooleanCondition(
            self,
            name="enable_right",
            enable_tick=True,
        )
        self._camera_left.set_mode(self._camera_mode_left)
        self._camera_right.set_mode(self._camera_mode_right)

        #
        csi_to_bayer_pool_left = holoscan.resources.BlockMemoryPool(
            self,
            name="pool",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=self._camera_left._width
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera_left._height,
            num_blocks=4,
        )
        csi_to_bayer_operator_left = hololink_module.operators.CsiToBayerOp(
            self,
            name="csi_to_bayer_left",
            allocator=csi_to_bayer_pool_left,
            cuda_device_ordinal=self._cuda_device_ordinal,
        )
        self._camera_left.configure_converter(csi_to_bayer_operator_left)

        #
        csi_to_bayer_pool_right = holoscan.resources.BlockMemoryPool(
            self,
            name="pool",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=self._camera_right._width
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera_right._height,
            num_blocks=4,
        )
        csi_to_bayer_operator_right = hololink_module.operators.CsiToBayerOp(
            self,
            name="csi_to_bayer_right",
            allocator=csi_to_bayer_pool_right,
            cuda_device_ordinal=self._cuda_device_ordinal,
        )
        self._camera_right.configure_converter(csi_to_bayer_operator_right)

        #
        frame_size = csi_to_bayer_operator_left.get_csi_length()
        self._frame_size_left = frame_size
        logging.info(f"left {frame_size=}")
        receiver_operator_left = self._make_receiver(
            self._condition_left,
            "receiver_left",
            self._metadata_left,
            self._camera_left,
            frame_size,
            [2],
            lambda name: f"left_{name}",
            "left",
        )

        #
        frame_size = csi_to_bayer_operator_right.get_csi_length()
        self._frame_size_right = frame_size
        logging.info(f"right {frame_size=}")
        receiver_operator_right = self._make_receiver(
            self._condition_right,
            "receiver_right",
            self._metadata_right,
            self._camera_right,
            frame_size,
            [3],
            lambda name: f"right_{name}",
            "right",
        )

        # Frame aligner. The two IMX274 sensors free-run (no hardware
        # frame-sync), so their frames drift relative to each other. The
        # aligner pairs them frame-for-frame, dropping a group whenever the
        # two legs' timestamps differ by more than allowable_dt (one full
        # frame period). It sits first, before the CRC stage, so CRC and
        # everything downstream only run on aligned frames. It tells the
        # legs apart by tensor name (the receivers name their tensors
        # "left"/"right" via out_tensor_name) and reads the per-leg
        # timestamp metadata the receivers stamp (left_/right_timestamp_s,
        # left_/right_timestamp_ns).
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

        #
        # CRC operators (hololink_module.operators). They compute a CRC
        # over the frame bytes; the received CRC the receiver reports is
        # compared in Python below.
        compute_crc_left = hololink_module.operators.ComputeCrcOp(
            self,
            name="compute_crc_left",
            frame_size=csi_to_bayer_operator_left.get_csi_length(),
        )
        check_crc_left = hololink_module.operators.CheckCrcOp(
            self,
            compute_crc_op=compute_crc_left,
            name="check_crc_left",
            computed_crc_metadata_name="check_crc_left",
        )

        compute_crc_right = hololink_module.operators.ComputeCrcOp(
            self,
            name="compute_crc_right",
            frame_size=csi_to_bayer_operator_right.get_csi_length(),
        )
        check_crc_right = hololink_module.operators.CheckCrcOp(
            self,
            compute_crc_op=compute_crc_right,
            name="check_crc_right",
            computed_crc_metadata_name="check_crc_right",
        )

        #
        rgba_components_per_pixel = 4
        bayer_size_left = (
            self._camera_left._width
            * rgba_components_per_pixel
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera_left._height
        )
        bayer_pool_left = holoscan.resources.BlockMemoryPool(
            self,
            name="pool",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=bayer_size_left,
            num_blocks=8,
        )
        bayer_format = self._camera_left.bayer_format()
        demosaic_left = holoscan.operators.BayerDemosaicOp(
            self,
            name="demosaic_left",
            pool=bayer_pool_left,
            generate_alpha=True,
            alpha_value=65535,
            bayer_grid_pos=bayer_format.value,
            interpolation_mode=0,
        )
        bayer_size_right = (
            self._camera_right._width
            * rgba_components_per_pixel
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera_right._height
        )
        bayer_pool_right = holoscan.resources.BlockMemoryPool(
            self,
            name="pool",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=bayer_size_right,
            num_blocks=8,
        )
        bayer_format = self._camera_right.bayer_format()
        demosaic_right = holoscan.operators.BayerDemosaicOp(
            self,
            name="demosaic_right",
            pool=bayer_pool_right,
            generate_alpha=True,
            alpha_value=65535,
            bayer_grid_pos=bayer_format.value,
            interpolation_mode=0,
        )

        # Compute the CRC of the bayer image and use that to
        # determine if we're receiving the expected image.
        bayer_crc_left = hololink_module.operators.ComputeCrcOp(
            self,
            name="bayer_crc_left",
            frame_size=bayer_size_left,
        )
        bayer_check_crc_left = hololink_module.operators.CheckCrcOp(
            self,
            compute_crc_op=bayer_crc_left,
            name="bayer_check_crc_left",
            computed_crc_metadata_name="bayer_crc_left",
        )
        # the tensor name directs which pane on the visualizer is updated
        rename_left_tensor = operators.PassThroughOperator(
            self,
            name="rename_left_tensor",
            out_tensor_name="left",
        )

        bayer_crc_right = hololink_module.operators.ComputeCrcOp(
            self,
            name="bayer_crc_right",
            frame_size=bayer_size_right,
        )
        bayer_check_crc_right = hololink_module.operators.CheckCrcOp(
            self,
            compute_crc_op=bayer_crc_right,
            name="bayer_check_crc_right",
            computed_crc_metadata_name="bayer_crc_right",
        )

        self._record_metadata_left = operators.RecordMetadataOp(
            self,
            name="record_metadata_left",
            metadata_class=(
                LeftRoceMetadataRecord if self._use_roce else LeftMetadataRecord
            ),
        )
        self._record_metadata_right = operators.RecordMetadataOp(
            self,
            name="record_metadata_right",
            metadata_class=(
                RightRoceMetadataRecord if self._use_roce else RightMetadataRecord
            ),
        )

        # the tensor name directs which pane on the visualizer is updated
        rename_right_tensor = operators.PassThroughOperator(
            self,
            name="rename_right_tensor",
            out_tensor_name="right",
        )
        #
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
            headless=self._headless,
            tensors=[left_spec, right_spec],
            height=window_height,
            width=window_width,
            window_title="IMX274 pattern test (module)",
            enable_camera_pose_output=True,
            camera_pose_output_type="extrinsics_model",
        )
        #
        # The sink owns the frame budget: it ticks frame_limit times (once per
        # aligned frame reaching the visualizer) and then disables the
        # receivers' tick conditions so the application drains and exits.
        watchdog_count = holoscan.conditions.CountCondition(
            self,
            name="watchdog_count",
            count=self._frame_limit,
        )
        watchdog_operator = operators.WatchdogOp(
            self,
            watchdog_count,
            name="watchdog_operator",
            watchdog=self._watchdog,
            frame_limit=self._frame_limit,
            stop_conditions=[self._condition_left, self._condition_right],
        )

        # Compute and capture each raw-frame CRC before FrameAligner can retain
        # the receiver-backed tensor while its DMA page is reused.
        self.add_flow(receiver_operator_left, compute_crc_left, {("output", "input")})
        self.add_flow(compute_crc_left, check_crc_left, {("output", "input")})
        self.add_flow(check_crc_left, self._frame_aligner, {("output", "input")})

        self.add_flow(receiver_operator_right, compute_crc_right, {("output", "input")})
        self.add_flow(compute_crc_right, check_crc_right, {("output", "input")})
        self.add_flow(check_crc_right, self._frame_aligner, {("output", "input")})

        # The aligner still pairs the two legs by their original receiver
        # timestamps and forwards the CRC metadata added above.
        self.add_flow(
            self._frame_aligner,
            csi_to_bayer_operator_left,
            {("output_left", "input")},
        )
        self.add_flow(
            csi_to_bayer_operator_left, demosaic_left, {("output", "receiver")}
        )
        self.add_flow(demosaic_left, bayer_crc_left, {("transmitter", "input")})
        self.add_flow(bayer_crc_left, bayer_check_crc_left, {("output", "input")})
        self.add_flow(
            bayer_check_crc_left, self._record_metadata_left, {("output", "input")}
        )
        self.add_flow(
            self._record_metadata_left, rename_left_tensor, {("output", "input")}
        )
        self.add_flow(rename_left_tensor, visualizer, {("output", "receivers")})

        self.add_flow(
            self._frame_aligner,
            csi_to_bayer_operator_right,
            {("output_right", "input")},
        )
        self.add_flow(
            csi_to_bayer_operator_right, demosaic_right, {("output", "receiver")}
        )
        self.add_flow(demosaic_right, bayer_crc_right, {("transmitter", "input")})
        self.add_flow(bayer_crc_right, bayer_check_crc_right, {("output", "input")})
        self.add_flow(
            bayer_check_crc_right, self._record_metadata_right, {("output", "input")}
        )
        self.add_flow(
            self._record_metadata_right, rename_right_tensor, {("output", "input")}
        )
        self.add_flow(rename_right_tensor, visualizer, {("output", "receivers")})

        self.add_flow(visualizer, watchdog_operator, {("camera_pose_output", "input")})


# A board that hosts a stereo pair exposes the two sensors as two data
# planes that share one control-plane HololinkInterface; bring that unit
# up once.
@pytest.fixture(scope="module")
def cuda_primary_context():
    """Keep one primary-context retain across all parametrized cases."""
    (cu_result,) = cuda.cuInit(0)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_device_ordinal = 0
    cu_result, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_result, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    yield cu_device_ordinal, cu_context
    gc.collect()
    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS


def run_test(
    headless,
    cuda_primary_context,
    module_dir,
    metadata_left,
    metadata_right,
    use_roce,
    camera_mode_left,
    pattern_left,
    expected_left,
    camera_mode_right,
    pattern_right,
    expected_right,
    allowable_dt,
    scheduler,
):
    #
    logging.info("Initializing.")
    cu_device_ordinal, cu_context = cuda_primary_context

    application = None
    app_scheduler = None

    def cleanup_application():
        nonlocal application, app_scheduler
        app_scheduler = None
        application = None
        gc.collect()

    #
    # The two sensors share the board's expander; select output 0 for the
    # left sensor and output 1 for the right one (the metadata-based
    # Imx274Cam reads this back when it constructs).
    imx274.Imx274Cam.use_expander_configuration(metadata_left, 0)
    imx274.Imx274Cam.use_expander_configuration(metadata_right, 1)
    camera_left = CameraWrapper(metadata_left)
    camera_right = CameraWrapper(metadata_right)
    frame_limit = 100
    ready_frame = 15  # ignore this many initial frames while the pipeline initializes
    initial_timeout = utils.timeout_sequence(
        [(30, ready_frame), (0.5, frame_limit - ready_frame - 2), (30, 1)]
    )
    try:
        with utils.Watchdog(
            "watchdog",
            initial_timeout=initial_timeout,
        ) as watchdog:
            # Set up the application
            application = PatternTestApplication(
                headless,
                cu_context,
                cu_device_ordinal,
                metadata_left,
                camera_left,
                camera_mode_left,
                metadata_right,
                camera_right,
                camera_mode_right,
                use_roce,
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
            with utils.AbortHook(hololink.reset):
                try:
                    assert camera_left._reset_callbacks == 0
                    assert camera_right._reset_callbacks == 0
                    hololink.reset()
                    assert camera_left._reset_callbacks == 1
                    assert camera_right._reset_callbacks == 1
                    camera_left.configure(camera_mode_left)
                    camera_left.test_pattern(pattern_left)
                    camera_right.configure(camera_mode_right)
                    camera_right.test_pattern(pattern_right)

                    # For testing, make sure we call the get_register method.
                    STANDBY = 0x3000
                    camera_left.get_register(STANDBY)
                    # Configure scheduler.
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
                        # Use the default one.
                        pass
                    else:
                        raise Exception(f"Unexpected {scheduler=}")
                    #
                    application.run()
                finally:
                    hololink.stop()

        # Verify CRCs match. Only the RoCE path delivers a lossless frame
        # whose CRC we can trust; the Linux-socket path expects packet loss,
        # so its image data is not evaluated.
        skip_frames = SKIP_INITIAL_FRAMES
        left_bad_crcs = 0
        left_bad_color_crcs = 0
        left_short_frames = 0
        right_bad_crcs = 0
        right_bad_color_crcs = 0
        right_short_frames = 0
        if use_roce:
            crcs_left = application._record_metadata_left.get_record()
            logging.info(f"Left camera captured {len(crcs_left)} frames")
            for frame_idx, record in enumerate(crcs_left):
                logging.trace(
                    f"{record.left_flags=:#x} "
                    f"{record.left_crc=:#x} {record.left_frame_number=:#x} "
                    f"{record.check_crc_left=:#x} {record.bayer_crc_left=:#x} "
                    f"{record.left_imm_data=:#x} {record.left_bytes_written=:#x}"
                )
                # Skip validation for first N frames (initialization artifacts)
                if frame_idx < skip_frames:
                    continue
                # Don't include the last frame, when the pipeline is shutting down.
                if frame_idx >= (frame_limit - 1):
                    continue
                if record.left_bytes_written != application._frame_size_left:
                    missing_bytes = (
                        application._frame_size_left - record.left_bytes_written
                    )
                    logging.error(
                        f"Left short CSI frame at frame {frame_idx}: "
                        f"frame_number={record.left_frame_number:#x}, "
                        f"flags={record.left_flags:#x}, "
                        f"bytes_written={record.left_bytes_written:#x}, "
                        f"expected={application._frame_size_left:#x}, "
                        f"missing_bytes={missing_bytes}"
                    )
                    left_short_frames += 1
                if record.left_crc != record.check_crc_left:
                    logging.error(
                        f"Left CRC mismatch at frame {frame_idx}: "
                        f"received={record.left_crc:#x}, "
                        f"computed={record.check_crc_left:#x}"
                    )
                    left_bad_crcs += 1
                if record.bayer_crc_left != expected_left:
                    left_bad_color_crcs += 1
            if left_bad_crcs == 0:
                logging.info(f"Validated {len(crcs_left) - skip_frames} left CRCs")

            crcs_right = application._record_metadata_right.get_record()
            logging.info(f"Right camera captured {len(crcs_right)} frames")
            for frame_idx, record in enumerate(crcs_right):
                logging.trace(
                    f"{record.right_flags=:#x} "
                    f"{record.right_crc=:#x} {record.right_frame_number=:#x} "
                    f"{record.check_crc_right=:#x} {record.bayer_crc_right=:#x} "
                    f"{record.right_imm_data=:#x} {record.right_bytes_written=:#x}"
                )
                # Skip validation for first N frames (initialization artifacts)
                if frame_idx < skip_frames:
                    continue
                # Don't include the last frame, when the pipeline is shutting down.
                if frame_idx >= (frame_limit - 1):
                    continue
                if record.right_bytes_written != application._frame_size_right:
                    missing_bytes = (
                        application._frame_size_right - record.right_bytes_written
                    )
                    logging.error(
                        f"Right short CSI frame at frame {frame_idx}: "
                        f"frame_number={record.right_frame_number:#x}, "
                        f"flags={record.right_flags:#x}, "
                        f"bytes_written={record.right_bytes_written:#x}, "
                        f"expected={application._frame_size_right:#x}, "
                        f"missing_bytes={missing_bytes}"
                    )
                    right_short_frames += 1
                if record.right_crc != record.check_crc_right:
                    logging.error(
                        f"Right CRC mismatch at frame {frame_idx}: "
                        f"received={record.right_crc:#x}, "
                        f"computed={record.check_crc_right:#x}"
                    )
                    right_bad_crcs += 1
                if record.bayer_crc_right != expected_right:
                    right_bad_color_crcs += 1
            if right_bad_crcs == 0:
                logging.info(f"Validated {len(crcs_right) - skip_frames} right CRCs")

    finally:
        # Destroy the graph, including allocator-owned tensors left in queues,
        # while the module-scoped primary-context retain is still valid. Native
        # CUDA/GXF teardown may outlive Python collection briefly, so the fixture
        # keeps that context alive across the following parametrized case.
        cleanup_application()

    assert (left_short_frames + right_short_frames) == 0, (
        f"Captured {left_short_frames} short left CSI frames and "
        f"{right_short_frames} short right CSI frames"
    )
    assert (left_bad_crcs + right_bad_crcs) == 0
    # Make sure the colors are as expected.
    assert left_bad_color_crcs == 0
    assert right_bad_color_crcs == 0


def _wait_for_channel(module_dir, peer_ip):
    adapter = hololink_module.Adapter.get_adapter()
    if module_dir:
        adapter.set_module_directory(module_dir)
    return adapter.wait_for_channel(peer_ip, DISCOVERY_TIMEOUT_S)


# Each tuple carries an allowable_dt: the largest skew the frame aligner
# tolerates between the two legs. IMX274 cannot hardware-sync, so the
# acceptance window is one full frame period (here 1/60 s at 60 FPS).
def round_up(f, decimal_places):
    factor = 10**decimal_places
    return math.ceil(f * factor) / factor


ONE_FRAME_AT_60FPS = round_up(1.0 / 60, 3)

expected_4k_results = [
    (
        # left
        imx274.Imx274_Mode.IMX274_MODE_3840X2160_60FPS,
        10,
        0xC361D7D3,
        # right
        imx274.Imx274_Mode.IMX274_MODE_3840X2160_60FPS,
        11,
        0x56CDC730,
        # allowable_dt
        ONE_FRAME_AT_60FPS,
    ),
    (
        # left
        imx274.Imx274_Mode.IMX274_MODE_3840X2160_60FPS,
        11,
        0x56CDC730,
        # right
        imx274.Imx274_Mode.IMX274_MODE_3840X2160_60FPS,
        10,
        0xC361D7D3,
        # allowable_dt
        ONE_FRAME_AT_60FPS,
    ),
]

expected_1080p_results = [
    (
        # left
        imx274.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
        10,
        0xB718A38C,
        # right
        imx274.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
        11,
        0x8F4D79DE,
        # allowable_dt
        ONE_FRAME_AT_60FPS,
    ),
    (
        # left
        imx274.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
        11,
        0x8F4D79DE,
        # right
        imx274.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
        10,
        0xB718A38C,
        # allowable_dt
        ONE_FRAME_AT_60FPS,
    ),
]

# RAW12 exercises Imx274Cam.configure_converter's 210-byte (175 + 35) embedded
# data offset and the packed12bitTo16bit kernel; a wrong start byte shifts the
# image and changes the bayer CRC.
#
# The expected CRCs were captured on hardware with the 210-byte offset in place.
expected_4k_12bit_results = [
    (
        # left
        imx274.Imx274_Mode.IMX274_MODE_3840X2160_30FPS_12BITS,
        10,
        0x9D48B11B,
        # right
        imx274.Imx274_Mode.IMX274_MODE_3840X2160_30FPS_12BITS,
        11,
        0x42559C42,
        # allowable_dt
        ONE_FRAME_AT_60FPS,
    ),
]

expected_results = []
expected_results.extend(expected_4k_results)
expected_results.extend(expected_4k_12bit_results)
expected_results.extend(expected_1080p_results)

mtus = [
    None,  # Use the datachannel default setting
    hololink_module.DEFAULT_MTU,
    330,  # produces a 2-page, 256 byte payload.
    4096,
]

full_mtu_pairs = [(mtu_left, mtu_right) for mtu_left in mtus for mtu_right in mtus]
reduced_mtu_pairs = [
    (None, None),
    (hololink_module.DEFAULT_MTU, hololink_module.DEFAULT_MTU),
    (330, 330),
    (4096, 4096),
    (330, 4096),
    (4096, 330),
]


def pytest_generate_tests(metafunc):
    if "mtu_left" in metafunc.fixturenames and "mtu_right" in metafunc.fixturenames:
        mtu_pairs = (
            reduced_mtu_pairs
            if metafunc.config.getoption("--reduced-mtu-matrix")
            else full_mtu_pairs
        )
        metafunc.parametrize("mtu_left,mtu_right", mtu_pairs)


@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode_left, pattern_left, expected_left, camera_mode_right, pattern_right, expected_right, allowable_dt",  # noqa: E501
    expected_results,
)
def test_imx274_pattern(
    camera_mode_left,
    pattern_left,
    expected_left,
    camera_mode_right,
    pattern_right,
    expected_right,
    allowable_dt,
    headless,
    cuda_primary_context,
    channel_ips,
    scheduler,
    module_dir,
):
    if len(channel_ips) < 2:
        pytest.skip("--channel-ips needs at least two IPs for this test")
    hololink_left, hololink_right = channel_ips[0], channel_ips[1]
    # Get a handle to data sources
    channel_metadata_left = _wait_for_channel(module_dir, hololink_left)
    channel_metadata_right = _wait_for_channel(module_dir, hololink_right)
    run_test(
        headless,
        cuda_primary_context,
        module_dir,
        channel_metadata_left,
        channel_metadata_right,
        True,  # use_roce
        camera_mode_left,
        pattern_left,
        expected_left,
        camera_mode_right,
        pattern_right,
        expected_right,
        allowable_dt,
        scheduler,
    )


@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode_left, pattern_left, expected_left, camera_mode_right, pattern_right, expected_right, allowable_dt",  # noqa: E501
    [expected_4k_results[0]],  # No need to do lots of these
)
def test_imx274_pattern_with_various_mtus(
    camera_mode_left,
    pattern_left,
    expected_left,
    camera_mode_right,
    pattern_right,
    expected_right,
    allowable_dt,
    headless,
    cuda_primary_context,
    channel_ips,
    module_dir,
    mtu_left,
    mtu_right,
):
    if len(channel_ips) < 2:
        pytest.skip("--channel-ips needs at least two IPs for this test")
    hololink_left, hololink_right = channel_ips[0], channel_ips[1]
    adapter = hololink_module.Adapter.get_adapter()
    # Get a handle to data sources
    channel_metadata_left = _wait_for_channel(module_dir, hololink_left)
    if mtu_left is not None:
        adapter.use_mtu(channel_metadata_left, mtu_left)
    channel_metadata_right = _wait_for_channel(module_dir, hololink_right)
    if mtu_right is not None:
        adapter.use_mtu(channel_metadata_right, mtu_right)
    run_test(
        headless,
        cuda_primary_context,
        module_dir,
        channel_metadata_left,
        channel_metadata_right,
        True,  # use_roce
        camera_mode_left,
        pattern_left,
        expected_left,
        camera_mode_right,
        pattern_right,
        expected_right,
        allowable_dt,
        scheduler="default",
    )


@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode_left, pattern_left, expected_left, camera_mode_right, pattern_right, expected_right, allowable_dt",  # noqa: E501
    expected_results,
)
@pytest.mark.parametrize(
    "multicast_left, multicast_left_port, multicast_right, multicast_right_port",  # noqa: E501
    [
        ("224.0.0.228", 4791, "224.0.0.229", 4791),
    ],
)
def test_imx274_multicast(
    camera_mode_left,
    pattern_left,
    expected_left,
    camera_mode_right,
    pattern_right,
    expected_right,
    allowable_dt,
    headless,
    cuda_primary_context,
    channel_ips,
    scheduler,
    module_dir,
    multicast_left,
    multicast_left_port,
    multicast_right,
    multicast_right_port,
):
    if len(channel_ips) < 2:
        pytest.skip("--channel-ips needs at least two IPs for this test")
    hololink_left, hololink_right = channel_ips[0], channel_ips[1]
    adapter = hololink_module.Adapter.get_adapter()
    # Get a handle to data sources, then stamp each channel's metadata
    # with its multicast destination. Adapter.use_multicast records the
    # group address/port that the RoCE data channel programs into the
    # FPGA when it configures the data plane.
    channel_metadata_left = _wait_for_channel(module_dir, hololink_left)
    adapter.use_multicast(channel_metadata_left, multicast_left, multicast_left_port)
    channel_metadata_right = _wait_for_channel(module_dir, hololink_right)
    adapter.use_multicast(channel_metadata_right, multicast_right, multicast_right_port)
    run_test(
        headless,
        cuda_primary_context,
        module_dir,
        channel_metadata_left,
        channel_metadata_right,
        True,  # use_roce
        camera_mode_left,
        pattern_left,
        expected_left,
        camera_mode_right,
        pattern_right,
        expected_right,
        allowable_dt,
        scheduler,
    )


# Test stereo patterns across a single network interface. Run once per board
# (channel_index selects channel_ips[channel_index]) so that each interface is
# verified to carry both of its sensors. The module resolves the IB device
# from the peer metadata, so there's no ibv_name to pair with the index.
@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode_left, pattern_left, expected_left, camera_mode_right, pattern_right, expected_right, allowable_dt",  # noqa: E501
    expected_1080p_results,
)
@pytest.mark.parametrize(
    "channel_index",
    [0, 1],
)
def test_imx274_stereo_single_interface(
    camera_mode_left,
    pattern_left,
    expected_left,
    camera_mode_right,
    pattern_right,
    expected_right,
    allowable_dt,
    headless,
    cuda_primary_context,
    scheduler,
    channel_ips,
    module_dir,
    channel_index,
):
    if channel_index >= len(channel_ips):
        pytest.skip(
            f"--channel-ips has {len(channel_ips)} IP(s); this case needs "
            f"index {channel_index}"
        )
    channel_ip = channel_ips[channel_index]
    # Get a handle to data sources
    adapter = hololink_module.Adapter.get_adapter()
    if module_dir:
        adapter.set_module_directory(module_dir)
    channel_metadata = adapter.wait_for_channel(channel_ip, DISCOVERY_TIMEOUT_S)
    # Now make separate ones for left and right; and set them to
    # use sensor 0 and 1 respectively.
    channel_metadata_left = hololink_module.EnumerationMetadata(channel_metadata)
    adapter.use_sensor(channel_metadata_left, 0)
    channel_metadata_right = hololink_module.EnumerationMetadata(channel_metadata)
    adapter.use_sensor(channel_metadata_right, 1)
    #
    run_test(
        headless,
        cuda_primary_context,
        module_dir,
        channel_metadata_left,
        channel_metadata_right,
        True,  # use_roce
        camera_mode_left,
        pattern_left,
        expected_left,
        camera_mode_right,
        pattern_right,
        expected_right,
        allowable_dt,
        scheduler,
    )


# Test stereo patterns across a single network interface using linux sockets.
# This test doesn't actually evaluate the image data due to expected packet losses.
@pytest.mark.skip_unless_imx274
@pytest.mark.parametrize(
    "camera_mode_left, pattern_left, expected_left, camera_mode_right, pattern_right, expected_right, allowable_dt",  # noqa: E501
    expected_1080p_results,
)
def test_linux_imx274_stereo_single_interface(
    camera_mode_left,
    pattern_left,
    expected_left,
    camera_mode_right,
    pattern_right,
    expected_right,
    allowable_dt,
    headless,
    cuda_primary_context,
    scheduler,
    channel_ip,
    module_dir,
):
    # Get a handle to data sources
    adapter = hololink_module.Adapter.get_adapter()
    if module_dir:
        adapter.set_module_directory(module_dir)
    channel_metadata = adapter.wait_for_channel(channel_ip, DISCOVERY_TIMEOUT_S)
    # Now make separate ones for left and right; and set them to
    # use sensor 0 and 1 respectively.
    channel_metadata_left = hololink_module.EnumerationMetadata(channel_metadata)
    adapter.use_sensor(channel_metadata_left, 0)
    channel_metadata_right = hololink_module.EnumerationMetadata(channel_metadata)
    adapter.use_sensor(channel_metadata_right, 1)
    #
    run_test(
        headless,
        cuda_primary_context,
        module_dir,
        channel_metadata_left,
        channel_metadata_right,
        False,  # use_roce -> Linux sockets; image data not evaluated
        camera_mode_left,
        pattern_left,
        expected_left,  # unused: image data not evaluated over Linux sockets
        camera_mode_right,
        pattern_right,
        expected_right,  # unused: image data not evaluated over Linux sockets
        allowable_dt,
        scheduler,
    )
