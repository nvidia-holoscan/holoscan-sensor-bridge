# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# A consumer of the receiver stalls for 0.2 s at frame 30. The receiver must drop
# the frames that arrive meanwhile and go on: the frame numbers it publishes have a
# gap right after the stalled frame, frames keep coming after the gap, and the
# watchdog sees a frame at least every 0.5 s after startup.
#
# In v2.7 the stall (0.1 s) and the watchdog were after HolovizOp, and the stall
# held up the whole pipeline (one scheduler thread), so the RoCE receiver dropped
# the frames that arrived meanwhile. HSDK5 runs the operators on several threads and
# doesn't wait for a full queue: a hololink receiver drops a frame that it can't
# publish. So the stalled sink takes the receiver's frames on a blocking edge with
# room for STALL_QUEUE_DEPTH frames, and the receiver drops whole frames, tensor
# and metadata, while the sink sleeps.

import logging
import time

import applications
import holoscan as hs
import operators
import pytest
import utils
from hololink_module.sensors.imx274.imx274_mode import Imx274_Mode

import hololink as hololink_module

# The sink's frame that stalls, and for how long: 12 frames at 60 FPS. At v2.7's
# 0.1 s, only 2 frames dropped (below), the fewest the test accepts.
STALL_FRAME = 30
STALL_S = 0.2

# During the stall, one frame more than the queue depth gets through: frames
# 30, 31 and 32 after the stalled frame 29 on an IGX Thor T7000 (ConnectX,
# 1080p60). The receiver drops the frames that arrive after them (2 there with a
# 0.1 s stall).
STALL_QUEUE_DEPTH = 2

# The frames that must arrive after the dropped ones.
RECOVERY_FRAMES = 10


class StallOp(hs.Operator):
    """Records the frame number of each of the receiver's frames, and sleeps
    STALL_S at the STALL_FRAME-th one. Its edge blocks, so it sees each frame
    that the receiver publishes, in order."""

    input = hs.Input("input", hs.payload.Tensor)

    record_token = hs.Param(0)

    def setup(self, spec):
        spec.input(self.input, queue_depth=STALL_QUEUE_DEPTH)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, context):
        sample = self.input.receive()
        if sample.is_missing:
            return
        frame_numbers = operators.lookup(self.record_token)
        frame_numbers.append(sample.metadata.frame_id)
        if len(frame_numbers) == STALL_FRAME:
            logging.info("STALLING")
            time.sleep(STALL_S)


def dropped_frame_test(
    camera_mode,
    pattern,
    hololink_address,
    frame_limit,
    receiver,
    receiver_kwargs,
    headless,
):
    frame_numbers = []
    with applications.CudaContext() as cuda_context, applications.imx274_camera(
        hololink_address, camera_mode, pattern=pattern
    ) as camera, utils.Watchdog(
        "frame-reception",
        initial_timeout=operators.color_profiler_initial_timeout(frame_limit),
        min_taps=utils.frame_floor(frame_limit),
    ) as watchdog, utils.RunMonitor() as monitor:
        test_graph = applications.TestGraph("dropped_frame")
        rx = applications.add_receiver(
            test_graph,
            "receiver",
            receiver,
            camera,
            cuda_context.cu_context,
            frame_limit,
            **receiver_kwargs,
        )
        isp = applications.add_isp(test_graph, "", rx, camera)
        applications.show_images(
            test_graph,
            "holoviz",
            [applications.Image(isp["demosaic"].output, camera.width, camera.height)],
            headless,
            "IMX274 dropped frame test",
        )
        watchdog_operator = test_graph.op(
            operators.WatchdogOp,
            "watchdog_operator",
            watchdog_token=operators.register(watchdog),
        )
        operators.metadata_flow(
            test_graph, isp["image_processor"].metadata, watchdog_operator.metadata
        )
        stall_operator = test_graph.op(
            StallOp, "stall_operator", record_token=operators.register(frame_numbers)
        )
        # The receiver publishes a frame's tensor before its metadata, so a full
        # queue here makes it drop the whole frame (counted). A sink that took
        # only the metadata would leave CsiToBayerOp tensors without metadata.
        test_graph.add_flow(
            rx.tensor,
            stall_operator.input,
            queue_policy=hs.QueuePolicy.BLOCKING,
            queue_depth=STALL_QUEUE_DEPTH,
        )
        test_graph.run()

    monitor.assert_ok()
    assert len(frame_numbers) > STALL_FRAME, f"Only {len(frame_numbers)} frames"
    stalled = frame_numbers[STALL_FRAME - 1]
    after_stall = frame_numbers[STALL_FRAME - 1 :]
    gaps = [
        (before, after)
        for before, after in zip(after_stall, after_stall[1:])
        if after != before + 1
    ]
    logging.info(
        f"{len(frame_numbers)} frames; stalled at frame {stalled}; gaps after it: {gaps}"
    )
    assert gaps, f"No frame was dropped after the stall at frame {stalled}"
    before, after = gaps[0]
    # The frames in the sink's queue and one more arrive; the receiver drops the
    # next ones.
    assert (
        before <= stalled + STALL_QUEUE_DEPTH + 1
    ), f"The first frames dropped, after {before}, aren't the stall's (frame {stalled})"
    assert after - before > 2, f"Only frames {before + 1}..{after - 1} were dropped"
    recovered = len(after_stall) - after_stall.index(after)
    assert (
        recovered >= RECOVERY_FRAMES
    ), f"Only {recovered} frames arrived after the dropped ones"


@utils.requires_roce
@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode, pattern",
    [
        (
            Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
            10,
        ),
    ],
)
def test_roce_imx274_dropped_frame(
    camera_mode,
    pattern,
    headless,
    hololink_address,
    ibv_name,
    ibv_port,
    frame_limit,
):
    dropped_frame_test(
        camera_mode,
        pattern,
        hololink_address,
        frame_limit,
        hololink_module.operators.RoceReceiverOp,
        {"ibv_name": ibv_name, "ibv_port": ibv_port},
        headless,
    )
