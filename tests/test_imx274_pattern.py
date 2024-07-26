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
import os

import cupy as cp
import holoscan
import pytest
from cuda import cuda

import hololink as hololink_module


class Profiler(holoscan.core.Operator):
    def __init__(self, *args, callback=None, out_tensor_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._count = 0
        self._callback = callback
        self._out_tensor_name = out_tensor_name

    def setup(self, spec):
        logging.info("setup")
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        self._count += 1
        in_message = op_input.receive("input")
        cp_frame = cp.from_dlpack(in_message.get(""))  # cp_frame.shape is (y,x,4)
        op_output.emit({self._out_tensor_name: cp_frame}, "output")
        # Give it some time to settle
        if self._count < 20:
            return
        # Compute the Y of YCrCb
        r = cp_frame[:, :, 0]
        g = cp_frame[:, :, 1]
        b = cp_frame[:, :, 2]
        y = r * 0.299 + g * 0.587 + b * 0.114
        logging.debug(f"{y=}")
        #
        unique = cp.unique(y)
        logging.info(f"{unique=}")
        buckets, _ = cp.histogram(y, bins=16, range=(0, 65536))
        s = ",".join([f"{x}" for x in buckets])
        logging.info(f"buckets=[{s}]")
        self._callback(buckets)


actual_left = None
actual_right = None


class CameraWrapper(hololink_module.sensors.imx274.dual_imx274.Imx274Cam):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reset_callbacks = 0
        self._hololink.on_reset(self._reset)

    def _reset(self):
        self._reset_callbacks += 1
        logging.info(f"{self._reset_callbacks=}")


def reset_globals():
    global actual_left, actual_right
    actual_left, actual_right = None, None


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

    def compose(self):
        logging.info("compose")
        self._ok = holoscan.conditions.BooleanCondition(
            self, name="ok", enable_tick=True
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
            num_blocks=2,
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
            num_blocks=2,
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
        receiver_operator_left = hololink_module.operators.RoceReceiverOp(
            self,
            self._ok,
            name="receiver_left",
            frame_size=frame_size,
            frame_context=frame_context,
            ibv_name=self._ibv_name_left,
            ibv_port=self._ibv_port_left,
            hololink_channel=self._hololink_channel_left,
            device=self._camera_left,
        )

        #
        frame_size = csi_to_bayer_operator_right.get_csi_length()
        logging.info(f"right {frame_size=}")
        frame_context = self._cuda_context
        receiver_operator_right = hololink_module.operators.RoceReceiverOp(
            self,
            self._ok,
            name="receiver_right",
            frame_size=frame_size,
            frame_context=frame_context,
            ibv_name=self._ibv_name_right,
            ibv_port=self._ibv_port_right,
            hololink_channel=self._hololink_channel_right,
            device=self._camera_right,
        )

        #
        rgba_components_per_pixel = 4
        bayer_pool_left = holoscan.resources.BlockMemoryPool(
            self,
            name="pool",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=self._camera_left._width
            * rgba_components_per_pixel
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera_left._height,
            num_blocks=2,
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
        bayer_pool_right = holoscan.resources.BlockMemoryPool(
            self,
            name="pool",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=self._camera_right._width
            * rgba_components_per_pixel
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera_right._height,
            num_blocks=2,
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

        #
        profiler_left = Profiler(
            self,
            name="profiler_left",
            callback=lambda buckets: self.left_buckets(buckets),
            out_tensor_name="left",
        )
        profiler_right = Profiler(
            self,
            name="profiler_right",
            callback=lambda buckets: self.right_buckets(buckets),
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
        )
        #
        self.add_flow(
            receiver_operator_left, csi_to_bayer_operator_left, {("output", "input")}
        )
        self.add_flow(
            csi_to_bayer_operator_left, demosaic_left, {("output", "receiver")}
        )
        self.add_flow(demosaic_left, profiler_left, {("transmitter", "input")})
        self.add_flow(profiler_left, visualizer, {("output", "receivers")})

        self.add_flow(
            receiver_operator_right, csi_to_bayer_operator_right, {("output", "input")}
        )
        self.add_flow(
            csi_to_bayer_operator_right, demosaic_right, {("output", "receiver")}
        )
        self.add_flow(demosaic_right, profiler_right, {("transmitter", "input")})
        self.add_flow(profiler_right, visualizer, {("output", "receivers")})

    def _check_done(self):
        global actual_left, actual_right
        logging.trace(f"{actual_left=} {actual_right=}")
        if actual_left is None:
            return
        if actual_right is None:
            return
        logging.info("DONE")
        self._ok.disable_tick()

    def left_buckets(self, buckets):
        global actual_left
        if actual_left is None:
            actual_left = buckets
        self._check_done()

    def right_buckets(self, buckets):
        global actual_right
        if actual_right is None:
            actual_right = buckets
        self._check_done()


@pytest.mark.skip_unless_udp_server
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode_left, pattern_left, expected_left, camera_mode_right, pattern_right, expected_right",  # noqa: E501
    [
        (
            # left
            hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_3840X2160_60FPS,
            10,
            # fmt: off
            [1038960, 1032480, 6480, 0, 1028160, 8640, 1032480, 0, 0, 1028160, 8640, 1041120, 0, 0, 1028160, 1041120],
            # fmt: on
            # right
            hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_3840X2160_60FPS,
            11,
            # fmt: off
            [921600, 917760, 7680, 0, 913920, 7680, 917760, 0, 0, 913920, 7680, 925440, 0, 0, 913920, 1847040],
            # fmt: on
        ),
        (
            # left
            hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_3840X2160_60FPS,
            11,
            # fmt: off
            [921600, 917760, 7680, 0, 913920, 7680, 917760, 0, 0, 913920, 7680, 925440, 0, 0, 913920, 1847040],
            # fmt: on
            # right
            hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_3840X2160_60FPS,
            10,
            # fmt: off
            [1038960, 1032480, 6480, 0, 1028160, 8640, 1032480, 0, 0, 1028160, 8640, 1041120, 0, 0, 1028160, 1041120],
            # fmt: on
        ),
        (
            # left
            hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
            10,
            # fmt: off
            [260280, 258120, 1080, 0, 257040, 2160, 258120, 0, 0, 257040, 2160, 260280, 0, 0, 257040, 260280],
            # fmt: on
            # right
            hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
            11,
            # fmt: off
            [0, 0, 0, 0, 0, 1920, 228480, 0, 0, 456960, 3840, 462720, 0, 0, 456960, 462720],
            # fmt: on
        ),
        (
            # left
            hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
            11,
            # fmt: off
            [0, 0, 0, 0, 0, 1920, 228480, 0, 0, 456960, 3840, 462720, 0, 0, 456960, 462720],
            # fmt: on
            # right
            hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
            10,
            # fmt: off
            [260280, 258120, 1080, 0, 257040, 2160, 258120, 0, 0, 257040, 2160, 260280, 0, 0, 257040, 260280],
            # fmt: on
        ),
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
    hololink_address,
    capsys,
):
    #
    logging.info("Initializing.")
    #
    reset_globals()
    # Get a handle to the GPU
    (cu_result,) = cuda.cuInit(0)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_device_ordinal = 0
    cu_result, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_result, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    # Get a handle to data sources
    hololink_left = hololink_address
    channel_metadata_left = hololink_module.Enumerator.find_channel(
        channel_ip=hololink_left
    )
    hololink_channel_left = hololink_module.DataChannel(channel_metadata_left)
    ip = [int(x) for x in hololink_address.split(".")]
    ip[-1] += 1
    hololink_right = ".".join([f"{x}" for x in ip])
    channel_metadata_right = hololink_module.Enumerator.find_channel(
        channel_ip=hololink_right
    )
    hololink_channel_right = hololink_module.DataChannel(channel_metadata_right)
    # Get a handle to the camera
    camera_left = CameraWrapper(hololink_channel_left, expander_configuration=0)
    camera_right = CameraWrapper(hololink_channel_right, expander_configuration=1)
    #
    ibv_name_left, ibv_name_right = sorted(os.listdir("/sys/class/infiniband"))
    ibv_port_left, ibv_port_right = 1, 1
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
    )
    default_configuration = os.path.join(
        os.path.dirname(__file__), "example_configuration.yaml"
    )
    application.config(default_configuration)
    # Run it.
    hololink = hololink_channel_left.hololink()
    assert hololink is hololink_channel_right.hololink()
    hololink.start()
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
    application.run()
    hololink.stop()

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS

    # Now check the buckets.
    global actual_left, actual_right
    #
    logging.info(f"{expected_left=}")
    logging.info(f"{actual_left=}")
    left_diffs = [abs(a - e) for e, a in zip(expected_left, actual_left, strict=True)]
    logging.info(f"{left_diffs=}")
    left_diff = sum(left_diffs)
    logging.info(f"{left_diff=}")
    #
    logging.info(f"{expected_right=}")
    logging.info(f"{actual_right=}")
    right_diffs = [
        abs(a - e) for e, a in zip(expected_right, actual_right, strict=True)
    ]
    logging.info(f"{right_diffs=}")
    right_diff = sum(right_diffs)
    logging.info(f"{right_diff=}")

    assert 0 <= left_diff < 4
    assert 0 <= right_diff < 4
