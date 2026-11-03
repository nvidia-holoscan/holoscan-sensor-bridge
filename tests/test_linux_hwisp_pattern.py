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

# HSDK5 differences from v2.7: the graph follows examples/linux_hwisp_player.py
# (LinuxReceiverOp -> CsiToBayerOp -> ArgusIspOp, and HolovizOp unless
# --headless); the receivers end the run at FRAME_LIMIT frames, and the first
# histogram of each leg is checked, as in v2.7. HSDK5's ArgusIspOp emits 8-bit
# RGB, which operators.luma_histogram() scales to the 16-bit range; the expected
# histograms are v2.7's and weren't measured again on HSDK5, because this test
# is skipped (BAJQ0XTT-173) and Argus couldn't start in the HSDK5 container on
# AGX Thor.

import logging

import applications
import operators
import pytest
import utils
from hololink_module.sensors.imx274.imx274_mode import Imx274_Mode

import hololink as hololink_module

# Frames to receive per leg; the first histogram of each leg is the one checked.
FRAME_LIMIT = 30


def add_hwisp_leg(test_graph, prefix, camera, camera_mode, cu_context, histograms):
    rx = applications.add_receiver(
        test_graph,
        f"{prefix}receiver",
        hololink_module.operators.LinuxReceiverOp,
        camera,
        cu_context,
        FRAME_LIMIT,
    )
    isp = applications.add_isp(
        test_graph, prefix, rx, camera, image_processor=False, demosaic=False
    )
    csi = isp["csi"]
    argus_isp = test_graph.op(
        hololink_module.operators.ArgusIspOp,
        f"{prefix}argus_isp",
        bayer_format=camera.bayer_format,
        # 60fps is 16.67ms
        exposure_time_ms=16.67,
        analog_gain=10.0,
        pixel_bit_depth=10,
        sensor_mode_index=int(camera_mode.value),
    )
    test_graph.add_flow(csi.tensor, argus_isp.in_tensor, queue_depth=4)
    test_graph.add_flow(csi.metadata, argus_isp.in_metadata, queue_depth=4)
    test_graph.bind(f"{prefix}argus_isp", "tensor")
    profiler = test_graph.op(
        operators.ColorProfiler,
        f"{prefix}color_profiler",
        callback_token=operators.register(histograms.append),
    )
    applications.image_flow(test_graph, argus_isp.tensor, profiler.input)
    return applications.Image(
        argus_isp.tensor, camera.width, camera.height, format="rgb8_unorm"
    )


@pytest.mark.skip("https://jirasw.nvidia.com/browse/BAJQ0XTT-173")
@pytest.mark.skip_on_jetpack_7
@pytest.mark.skip_unless_igpu
@pytest.mark.skip_unless_imx274
@pytest.mark.parametrize(
    "camera_mode_left, pattern_left, expected_left, camera_mode_right, pattern_right, expected_right",  # noqa: E501
    [
        (
            # left
            Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
            10,
            # fmt: off
            [260280, 258120, 1080, 0, 257040, 2160, 258120, 0, 0, 257040, 2160, 260280, 0, 0, 257040, 260280],
            # fmt: on
            # right
            Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
            11,
            # fmt: off
            [0, 0, 0, 0, 0, 1920, 228480, 0, 0, 456960, 3840, 462720, 0, 0, 456960, 462720],
            # fmt: on
        ),
        (
            # left
            Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
            11,
            # fmt: off
            [0, 0, 0, 0, 0, 1920, 228480, 0, 0, 456960, 3840, 462720, 0, 0, 456960, 462720],
            # fmt: on
            # right
            Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
            10,
            # fmt: off
            [260280, 258120, 1080, 0, 257040, 2160, 258120, 0, 0, 257040, 2160, 260280, 0, 0, 257040, 260280],
            # fmt: on
        ),
    ],
)
def test_linux_hwisp_pattern(
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
    if not hasattr(hololink_module.operators, "ArgusIspOp"):
        pytest.skip("ArgusIspOp isn't built")
    ip = [int(x) for x in hololink_address.split(".")]
    ip[-1] += 1
    hololink_right = ".".join([f"{x}" for x in ip])
    histograms_left, histograms_right = [], []
    with applications.CudaContext() as cuda_context, applications.imx274_camera(
        hololink_address, camera_mode_left, pattern=pattern_left
    ) as camera_left, utils.RunMonitor() as monitor:
        # camera_left's setup_clock() also sets camera_right's clock.
        camera_right = applications.open_imx274(
            hololink_right,
            camera_mode_right,
            pattern=pattern_right,
            expander_configuration=1,
            reset=False,
            device=camera_left.device,
        )
        assert camera_left.device is camera_right.channel.hololink()
        test_graph = applications.TestGraph("hwisp_pattern")
        left = add_hwisp_leg(
            test_graph,
            "left_",
            camera_left,
            camera_mode_left,
            cuda_context.cu_context,
            histograms_left,
        )
        right = add_hwisp_leg(
            test_graph,
            "right_",
            camera_right,
            camera_mode_right,
            cuda_context.cu_context,
            histograms_right,
        )
        applications.show_images(
            test_graph, "holoviz", [left, right], headless, "IMX274 pattern test"
        )
        test_graph.run()
    monitor.assert_ok()

    # Now check the buckets.
    assert histograms_left and histograms_right, "ColorProfiler got no image"
    actual_left, actual_right = list(histograms_left[0]), list(histograms_right[0])
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
