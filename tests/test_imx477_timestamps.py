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

# HSDK5 differences from v2.7: v2.7 ran examples/imx477_player.main() with its
# HoloscanApplication class patched; the HSDK5 example has no such class, so the
# test builds the same pipeline (sensor set-up as in that example, test pattern,
# PTP sync) as an hs.Graph. The receiver ends the run at the frame limit, and
# TimeProfiler takes the metadata after CsiToBayerOp.

import datetime
import logging
import threading

import applications
import operators
import pytest
import utils

import hololink as hololink_module


def run_timestamp_test(
    hololink_address, receiver, receiver_kwargs, frame_limit, headless
):
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
    with applications.CudaContext() as cuda_context, applications.imx477_camera(
        hololink_address, camera_id=0, pattern=True, ptp=True
    ) as camera, utils.Watchdog(
        "watchdog",
        initial_timeout=initial_timeout,
        min_taps=utils.frame_floor(frame_limit),
    ) as watchdog, utils.RunMonitor() as monitor:
        test_graph = applications.TestGraph("imx477_timestamps")
        rx = applications.add_receiver(
            test_graph,
            "receiver",
            receiver,
            camera,
            cuda_context.cu_context,
            frame_limit,
            **receiver_kwargs,
        )
        isp = applications.add_isp(
            test_graph, "", rx, camera, image_processor=False, demosaic=False
        )
        demosaic = applications.add_demosaic(test_graph, "", isp["csi"], camera)
        applications.show_images(
            test_graph,
            "holoviz",
            [applications.Image(demosaic.output, camera.width, camera.height)],
            headless,
            "IMX477 timestamps",
        )
        profiler = test_graph.op(
            operators.TimeProfiler,
            "profiler",
            callback_token=operators.register(time_profile),
        )
        operators.metadata_flow(test_graph, isp["csi"].metadata, profiler.metadata)
        watchdog_operator = test_graph.op(
            operators.WatchdogOp,
            "watchdog_operator",
            watchdog_token=operators.register(watchdog),
        )
        operators.metadata_flow(
            test_graph, isp["csi"].metadata, watchdog_operator.metadata
        )
        test_graph.run()
    monitor.assert_ok()
    return timestamps


# frame_time represents the constant time difference between when the
#   frame-start and frame-end messages arrive at the FPGA.
# time_limit, the acceptable amount of time between when the frame was sent and
#   when we got around to looking at it, is much smaller in the RDMA
#   configuration.
@pytest.mark.skip_unless_ptp
@pytest.mark.skip_unless_imx477
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "roce_mode, frame_time, time_limit",  # noqa: E501
    [
        (
            True,
            0.015,
            0.004,
        ),
        (
            False,
            0.015,
            0.012,
        ),
    ],
)
def test_imx477_timestamps(
    roce_mode,
    frame_time,
    time_limit,
    headless,
    hololink_address,
    ibv_name,
    ibv_port,
    frame_limit,
):
    if roce_mode:
        if not hasattr(hololink_module.operators, "RoceReceiverOp"):
            pytest.skip("RoceReceiverOp isn't built (HOLOLINK_BUILD_ROCE=OFF)")
        receiver = hololink_module.operators.RoceReceiverOp
        receiver_kwargs = {"ibv_name": ibv_name, "ibv_port": ibv_port}
    else:
        receiver = hololink_module.operators.LinuxReceiverOp
        receiver_kwargs = {}
    timestamps = run_timestamp_test(
        hololink_address, receiver, receiver_kwargs, frame_limit, headless
    )

    # check for errors
    pipeline_dts, receiver_dts = [], []
    metadata_receiver_dts = []
    # Allow for startup times to be a bit longer
    settled_timestamps = timestamps[5:-5]
    assert len(settled_timestamps) >= 100
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
        logging.debug(
            f"{image_timestamp=} {received_timestamp=} {receiver_dt=:0.6f} {frame_number=}"
        )
        receiver_dts.append(round(receiver_dt, 4))
        metadata_receiver_dt = received_timestamp_s - metadata_timestamp_s
        logging.debug(
            f"{metadata_timestamp=} {received_timestamp=} {metadata_receiver_dt=:0.6f} {frame_number=}"
        )
        metadata_receiver_dts.append(round(metadata_receiver_dt, 4))

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
    assert (frame_time + 0) <= smallest_time_difference
    assert smallest_time_difference <= largest_time_difference
    assert largest_time_difference <= (frame_time + time_limit)
    #
    smallest_time_difference = min(metadata_receiver_dts)
    largest_time_difference = max(metadata_receiver_dts)
    logging.info(
        f"FPGA to full frame received {smallest_time_difference=} {largest_time_difference=}"
    )
    assert smallest_time_difference <= largest_time_difference
