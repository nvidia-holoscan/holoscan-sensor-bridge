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

# HSDK5 differences from v2.7: the graph is an hs.Graph, and the receiver ends
# the run at FRAME_LIMIT frames; v2.7 stopped at the first histogram. The first
# histogram is checked, as in v2.7.

import logging

import applications
import operators
import pytest
import utils

import hololink as hololink_module

# Frames to receive; the first histogram is the one that's checked.
FRAME_LIMIT = 30


@pytest.mark.skip_unless_imx477
@pytest.mark.parametrize(
    "camera_mode,expected",
    [
        (
            [3840, 2160],
            # fmt: on
            [
                641520,
                4320,
                1090800,
                2160,
                0,
                1088640,
                4320,
                1090800,
                0,
                1088640,
                4320,
                1090800,
                4320,
                0,
                1088640,
                1095120,
            ],  #
        ),
    ],
)
def test_imx477_pattern(
    camera_mode,
    expected,
    headless,
    hololink_address,
):
    logging.info("Initializing.")
    histograms = []
    with applications.CudaContext() as cuda_context, applications.imx477_camera(
        hololink_address, camera_id=0, pattern=True
    ) as camera, utils.RunMonitor() as monitor:
        assert [camera.width, camera.height] == camera_mode
        # Prove that get_register works as expected.
        pattern_register = camera.camera.get_register(0x601)
        assert pattern_register == 0x2
        test_graph = applications.TestGraph("imx477_pattern")
        rx = applications.add_receiver(
            test_graph,
            "receiver",
            hololink_module.operators.LinuxReceiverOp,
            camera,
            cuda_context.cu_context,
            FRAME_LIMIT,
        )
        isp = applications.add_isp(
            test_graph, "", rx, camera, image_processor=False, demosaic=False
        )
        # CsiToBayerOp -> BayerDemosaicOp, without ImageProcessorOp, as in v2.7.
        demosaic = applications.add_demosaic(test_graph, "", isp["csi"], camera)
        profiler = test_graph.op(
            operators.ColorProfiler,
            "profiler",
            callback_token=operators.register(histograms.append),
        )
        applications.image_flow(test_graph, demosaic.output, profiler.input)
        applications.show_images(
            test_graph,
            "holoviz",
            [applications.Image(demosaic.output, camera.width, camera.height)],
            headless,
            "IMX477 pattern test",
        )
        test_graph.run()

    monitor.assert_ok()
    # Now check the buckets.
    assert histograms, "ColorProfiler got no image"
    actual = list(histograms[0])
    logging.info(f"{expected=}")
    logging.info(f"{actual=}")
    diffs = [abs(a - e) for e, a in zip(expected, actual, strict=True)]
    logging.info(f"{diffs=}")
    diff = sum(diffs)
    logging.info(f"{diff=}")

    assert 0 <= diff < 4
