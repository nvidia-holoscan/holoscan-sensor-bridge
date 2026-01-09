# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# See README.md for detailed information.

import ctypes
import logging

import cuda.bindings.driver as cuda
import holoscan
import operators
import pytest
import utils

import hololink as hololink_module

# Number of frames to skip during pipeline initialization to avoid artifacts
SKIP_INITIAL_FRAMES = 15


class CameraWrapper(hololink_module.sensors.imx274.dual_imx274.Imx274Cam):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reset_callbacks = 0
        self._hololink.on_reset(self._reset)

    def _reset(self):
        self._reset_callbacks += 1
        logging.info(f"{self._reset_callbacks=}")


class PatternTestApplication(holoscan.core.Application):
    def __init__(
        self,
        headless,
        cuda_context,
        cuda_device_ordinal,
        hololink_channel_left,
        ibv_name_left,
        ibv_port_left,
        camera_left,
        camera_mode_left,
        hololink_channel_right,
        ibv_name_right,
        ibv_port_right,
        camera_right,
        camera_mode_right,
        watchdog,
        frame_limit,
    ):
        logging.info("__init__")
        super().__init__()
        self._headless = headless
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._hololink_channel_left = hololink_channel_left
        self._ibv_name_left = ibv_name_left
        self._ibv_port_left = ibv_port_left
        self._camera_left = camera_left
        self._camera_mode_left = camera_mode_left
        self._hololink_channel_right = hololink_channel_right
        self._ibv_name_right = ibv_name_right
        self._ibv_port_right = ibv_port_right
        self._camera_right = camera_right
        self._camera_mode_right = camera_mode_right
        self._watchdog = watchdog
        self._bucket_count_left = 0
        self._bucket_count_right = 0
        self._frame_limit = frame_limit
        # These are HSDK controls-- because we have stereo
        # camera paths going into the same visualizer, don't
        # raise an error when each path present metadata
        # with the same names.  Because we don't use that metadata,
        # it's easiest to just ignore new items with the same
        # names as existing items.
        self.is_metadata_enabled = True
        self.metadata_policy = holoscan.core.MetadataPolicy.REJECT

    def compose(self):
        logging.info("compose")
        self._condition_left = holoscan.conditions.CountCondition(
            self,
            name="count_left",
            count=self._frame_limit,
        )
        self._condition_right = holoscan.conditions.CountCondition(
            self,
            name="count_right",
            count=self._frame_limit,
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
        logging.info(f"left {frame_size=}")
        frame_context = self._cuda_context
        if self._ibv_name_left:
            receiver_operator_left = hololink_module.operators.RoceReceiverOp(
                self,
                self._condition_left,
                name="receiver_left",
                frame_size=frame_size,
                frame_context=frame_context,
                ibv_name=self._ibv_name_left,
                ibv_port=self._ibv_port_left,
                hololink_channel=self._hololink_channel_left,
                device=self._camera_left,
                rename_metadata=lambda name: f"left_{name}",
            )
        else:
            receiver_operator_left = hololink_module.operators.LinuxReceiverOp(
                self,
                self._condition_left,
                name="receiver_left",
                frame_size=frame_size,
                frame_context=frame_context,
                hololink_channel=self._hololink_channel_left,
                device=self._camera_left,
                rename_metadata=lambda name: f"left_{name}",
            )

        #
        frame_size = csi_to_bayer_operator_right.get_csi_length()
        logging.info(f"right {frame_size=}")
        frame_context = self._cuda_context
        if self._ibv_name_right:
            receiver_operator_right = hololink_module.operators.RoceReceiverOp(
                self,
                self._condition_right,
                name="receiver_right",
                frame_size=frame_size,
                frame_context=frame_context,
                ibv_name=self._ibv_name_right,
                ibv_port=self._ibv_port_right,
                hololink_channel=self._hololink_channel_right,
                device=self._camera_right,
                rename_metadata=lambda name: f"right_{name}",
            )
        else:
            receiver_operator_right = hololink_module.operators.LinuxReceiverOp(
                self,
                self._condition_right,
                name="receiver_right",
                frame_size=frame_size,
                frame_context=frame_context,
                hololink_channel=self._hololink_channel_right,
                device=self._camera_right,
                rename_metadata=lambda name: f"right_{name}",
            )

        #
        # CRC operators
        compute_crc_left = hololink_module.operators.ComputeCrcOp(
            self,
            name="compute_crc_left",
            frame_size=csi_to_bayer_operator_left.get_csi_length(),
        )
        check_crc_left = hololink_module.operators.CheckCrcOp(
            self,
            compute_crc_left,
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
            compute_crc_right,
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
            bayer_crc_left,
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
            bayer_crc_right,
            name="bayer_check_crc_right",
            computed_crc_metadata_name="bayer_crc_right",
        )

        self._record_metadata_left = operators.RecordMetadataOp(
            self,
            name="record_metadata_left",
            metadata_names=[
                "left_crc",
                "check_crc_left",
                "bayer_crc_left",
            ],
        )
        self._record_metadata_right = operators.RecordMetadataOp(
            self,
            name="record_metadata_right",
            metadata_names=[
                "right_crc",
                "check_crc_right",
                "bayer_crc_right",
            ],
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
            window_title="IMX274 pattern test",
            enable_camera_pose_output=True,
            camera_pose_output_type="extrinsics_model",
        )
        #
        watchdog_operator = operators.WatchdogOp(
            self,
            name="watchdog_operator",
            watchdog=self._watchdog,
        )

        # Add CRC checking to the pipeline
        self.add_flow(receiver_operator_left, compute_crc_left, {("output", "input")})
        self.add_flow(compute_crc_left, check_crc_left, {("output", "input")})
        self.add_flow(check_crc_left, csi_to_bayer_operator_left, {("output", "input")})
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

        self.add_flow(receiver_operator_right, compute_crc_right, {("output", "input")})
        self.add_flow(compute_crc_right, check_crc_right, {("output", "input")})
        self.add_flow(
            check_crc_right, csi_to_bayer_operator_right, {("output", "input")}
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

    def left_buckets(self, buckets):
        global actual_left
        actual_left = buckets

    def right_buckets(self, buckets):
        global actual_right
        actual_right = buckets


# This may execute on unaccelerated configurations, where
# there may be any number of infiniband interfaces (but
# most likely zero).  In this case, placate parametrize
# by providing dummy None values in these columns.
sys_ibv_name_left, sys_ibv_name_right = (
    hololink_module.infiniband_devices() + [None, None]
)[:2]


expected_4k_results = [
    (
        # left
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_3840X2160_60FPS,
        10,
        0xC361D7D3,
        # right
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_3840X2160_60FPS,
        11,
        0x56CDC730,
    ),
    (
        # left
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_3840X2160_60FPS,
        11,
        0x56CDC730,
        # right
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_3840X2160_60FPS,
        10,
        0xC361D7D3,
    ),
]

expected_1080p_results = [
    (
        # left
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
        10,
        0xB718A38C,
        # right
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
        11,
        0x8F4D79DE,
    ),
    (
        # left
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
        11,
        0x8F4D79DE,
        # right
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
        10,
        0xB718A38C,
    ),
]

expected_results = []
expected_results.extend(expected_4k_results)
expected_results.extend(expected_1080p_results)


def run_test(
    headless,
    channel_metadata_left,
    ibv_name_left,
    ibv_port_left,
    camera_mode_left,
    pattern_left,
    expected_left,
    channel_metadata_right,
    ibv_name_right,
    ibv_port_right,
    camera_mode_right,
    pattern_right,
    expected_right,
    scheduler,
):
    #
    logging.info("Initializing.")
    #
    # Get a handle to the GPU
    (cu_result,) = cuda.cuInit(0)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_device_ordinal = 0
    cu_result, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_result, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    #
    hololink_channel_left = hololink_module.DataChannel(channel_metadata_left)
    hololink_channel_right = hololink_module.DataChannel(channel_metadata_right)
    # Get a handle to the camera
    camera_left = CameraWrapper(hololink_channel_left, expander_configuration=0)
    camera_right = CameraWrapper(hololink_channel_right, expander_configuration=1)
    frame_limit = 100
    ready_frame = 15  # ignore this many initial frames while the pipeline initializes
    initial_timeout = utils.timeout_sequence(
        [(30, ready_frame), (0.5, frame_limit - ready_frame - 2), (30, 1)]
    )
    with utils.Watchdog(
        "watchdog",
        initial_timeout=initial_timeout,
    ) as watchdog:
        # Set up the application
        application = PatternTestApplication(
            headless,
            cu_context,
            cu_device_ordinal,
            hololink_channel_left,
            ibv_name_left,
            ibv_port_left,
            camera_left,
            camera_mode_left,
            hololink_channel_right,
            ibv_name_right,
            ibv_port_right,
            camera_right,
            camera_mode_right,
            watchdog,
            frame_limit=frame_limit,
        )
        # Run it.
        hololink = hololink_channel_left.hololink()
        assert hololink is hololink_channel_right.hololink()
        hololink.start()
        try:
            assert camera_left._reset_callbacks == 0
            assert camera_right._reset_callbacks == 0
            hololink.reset()
            assert camera_left._reset_callbacks == 1
            assert camera_right._reset_callbacks == 1
            camera_left.setup_clock()  # this also sets camera_right's clock
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

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS

    # Verify CRCs match
    skip_frames = SKIP_INITIAL_FRAMES
    left_bad_crcs = 0
    left_bad_color_crcs = 0
    right_bad_crcs = 0
    right_bad_color_crcs = 0
    if ibv_name_left:
        crcs_left = application._record_metadata_left.get_record()
        logging.info(f"Left camera captured {len(crcs_left)} frames")
        for frame_idx, (left_crc, check_crc_left, bayer_crc_left) in enumerate(
            crcs_left
        ):
            logging.trace(f"{left_crc=:#x} {check_crc_left=:#x} {bayer_crc_left=:#x}")
            # Skip validation for first N frames (initialization artifacts)
            if frame_idx < skip_frames:
                continue
            # Don't include the last frame, when the pipeline is shutting down.
            if frame_idx >= (frame_limit - 1):
                continue
            if left_crc != check_crc_left:
                logging.error(
                    f"Left CRC mismatch at frame {frame_idx}: "
                    f"received={left_crc:#x}, computed={check_crc_left:#x}"
                )
                left_bad_crcs += 1
            if bayer_crc_left != expected_left:
                left_bad_color_crcs += 1
        if left_bad_crcs == 0:
            logging.info(f"Validated {len(crcs_left) - skip_frames} left CRCs")

    if ibv_name_right:
        crcs_right = application._record_metadata_right.get_record()
        logging.info(f"Right camera captured {len(crcs_right)} frames")
        for frame_idx, (right_crc, check_crc_right, bayer_crc_right) in enumerate(
            crcs_right
        ):
            logging.trace(
                f"{right_crc=:#x} {check_crc_right=:#x} {bayer_crc_right=:#x}"
            )
            # Skip validation for first N frames (initialization artifacts)
            if frame_idx < skip_frames:
                continue
            # Don't include the last frame, when the pipeline is shutting down.
            if frame_idx >= (frame_limit - 1):
                continue
            if right_crc != check_crc_right:
                logging.error(
                    f"Right CRC mismatch at frame {frame_idx}: "
                    f"received={right_crc:#x}, computed={check_crc_right:#x}"
                )
                right_bad_crcs += 1
            if bayer_crc_right != expected_right:
                right_bad_color_crcs += 1
        if right_bad_crcs == 0:
            logging.info(f"Validated {len(crcs_right) - skip_frames} right CRCs")
    assert (left_bad_crcs + right_bad_crcs) == 0
    # Make sure the colors are as expected.
    if expected_left is not None:
        assert left_bad_color_crcs == 0
    if expected_right is not None:
        assert right_bad_color_crcs == 0


@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode_left, pattern_left, expected_left, camera_mode_right, pattern_right, expected_right",  # noqa: E501
    expected_results,
)
@pytest.mark.parametrize(
    "ibv_name_left, ibv_name_right",
    [
        (sys_ibv_name_left, sys_ibv_name_right),
    ],
)
@pytest.mark.parametrize(
    "hololink_left, hololink_right",
    [
        ("192.168.0.2", "192.168.0.3"),
    ],
)
def test_imx274_pattern(
    camera_mode_left,
    pattern_left,
    expected_left,
    camera_mode_right,
    pattern_right,
    expected_right,
    headless,
    hololink_left,
    hololink_right,
    scheduler,
    ibv_name_left,
    ibv_name_right,
):
    # Get a handle to data sources
    channel_metadata_left = hololink_module.Enumerator.find_channel(
        channel_ip=hololink_left
    )
    channel_metadata_right = hololink_module.Enumerator.find_channel(
        channel_ip=hololink_right
    )
    ibv_port_left, ibv_port_right = 1, 1
    run_test(
        headless,
        channel_metadata_left,
        ibv_name_left,
        ibv_port_left,
        camera_mode_left,
        pattern_left,
        expected_left,
        channel_metadata_right,
        ibv_name_right,
        ibv_port_right,
        camera_mode_right,
        pattern_right,
        expected_right,
        scheduler,
    )


@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode_left, pattern_left, expected_left, camera_mode_right, pattern_right, expected_right",  # noqa: E501
    expected_results,
)
@pytest.mark.parametrize(
    "multicast_left, multicast_left_port, multicast_right, multicast_right_port",  # noqa: E501
    [
        ("224.0.0.228", 4791, "224.0.0.229", 4791),
    ],
)
@pytest.mark.parametrize(
    "ibv_name_left, ibv_name_right",  # noqa: E501
    [
        (sys_ibv_name_left, sys_ibv_name_right),
    ],
)
@pytest.mark.parametrize(
    "hololink_left, hololink_right",
    [
        ("192.168.0.2", "192.168.0.3"),
    ],
)
def test_imx274_multicast(
    camera_mode_left,
    pattern_left,
    expected_left,
    camera_mode_right,
    pattern_right,
    expected_right,
    headless,
    hololink_left,
    hololink_right,
    scheduler,
    multicast_left,
    multicast_left_port,
    multicast_right,
    multicast_right_port,
    ibv_name_left,
    ibv_name_right,
):
    # Get a handle to data sources
    channel_metadata_left = hololink_module.Enumerator.find_channel(
        channel_ip=hololink_left
    )
    hololink_module.DataChannel.use_multicast(
        channel_metadata_left, multicast_left, multicast_left_port
    )
    channel_metadata_right = hololink_module.Enumerator.find_channel(
        channel_ip=hololink_right
    )
    hololink_module.DataChannel.use_multicast(
        channel_metadata_right, multicast_right, multicast_right_port
    )
    ibv_port_left, ibv_port_right = 1, 1
    run_test(
        headless,
        channel_metadata_left,
        ibv_name_left,
        ibv_port_left,
        camera_mode_left,
        pattern_left,
        expected_left,
        channel_metadata_right,
        ibv_name_right,
        ibv_port_right,
        camera_mode_right,
        pattern_right,
        expected_right,
        scheduler,
    )


# Test stereo patterns across a single network interface.
@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode_left, pattern_left, expected_left, camera_mode_right, pattern_right, expected_right",  # noqa: E501
    expected_1080p_results,
)
@pytest.mark.parametrize(
    "ibv_name, ibv_channel_ip",
    [
        (sys_ibv_name_left, "192.168.0.2"),
        (sys_ibv_name_right, "192.168.0.3"),
    ],
)
def test_imx274_stereo_single_interface(
    camera_mode_left,
    pattern_left,
    expected_left,
    camera_mode_right,
    pattern_right,
    expected_right,
    headless,
    scheduler,
    ibv_name,
    ibv_channel_ip,
):
    # Get a handle to data sources
    channel_metadata = hololink_module.Enumerator.find_channel(
        channel_ip=ibv_channel_ip
    )
    # Now make separate ones for left and right; and set them to
    # use sensor 0 and 1 respectively.
    channel_metadata_left = hololink_module.Metadata(channel_metadata)
    hololink_module.DataChannel.use_sensor(channel_metadata_left, 0)
    channel_metadata_right = hololink_module.Metadata(channel_metadata)
    hololink_module.DataChannel.use_sensor(channel_metadata_right, 1)
    #
    ibv_port = 1
    run_test(
        headless,
        channel_metadata_left,
        ibv_name,
        ibv_port,
        camera_mode_left,
        pattern_left,
        expected_left,
        channel_metadata_right,
        ibv_name,
        ibv_port,
        camera_mode_right,
        pattern_right,
        expected_right,
        scheduler,
    )


# Test stereo patterns across a single network interface using linux sockets.
# This test doesn't actually evaluate the image data due to expected packet losses.
@pytest.mark.skip_unless_imx274
@pytest.mark.parametrize(
    "camera_mode_left, pattern_left, expected_left, camera_mode_right, pattern_right, expected_right",  # noqa: E501
    expected_1080p_results,
)
def test_linux_imx274_stereo_single_interface(
    camera_mode_left,
    pattern_left,
    expected_left,
    camera_mode_right,
    pattern_right,
    expected_right,
    headless,
    scheduler,
    channel_ip,
):
    # Get a handle to data sources
    channel_metadata = hololink_module.Enumerator.find_channel(channel_ip=channel_ip)
    # Now make separate ones for left and right; and set them to
    # use sensor 0 and 1 respectively.
    channel_metadata_left = hololink_module.Metadata(channel_metadata)
    hololink_module.DataChannel.use_sensor(channel_metadata_left, 0)
    channel_metadata_right = hololink_module.Metadata(channel_metadata)
    hololink_module.DataChannel.use_sensor(channel_metadata_right, 1)
    #
    run_test(
        headless,
        channel_metadata_left,
        None,  # ibv_name_left
        None,  # ibv_port_left
        camera_mode_left,
        pattern_left,
        None,  # expected_left
        channel_metadata_right,
        None,  # ibv_name_right
        None,  # ibv_port_right
        camera_mode_right,
        pattern_right,
        None,  # expected_right
        scheduler,
    )
