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

# The data plane's buffer length is set below the frame size, so the FPGA sends
# each sensor frame as two: a "runt" of that length, then the rest. Each receiver
# must report the bytes the FPGA wrote, and with trim, emit a tensor of just that
# size.
#
# HSDK5 differences from v2.7:
# - The buffer length is written from the receiver's device_start callback, which
#   it calls after it has configured the data channel; v2.7 wrote it from an
#   operator's start(). HSDK5 doesn't export DP_BUFFER_LENGTH to Python.
# - The tensor sizes and the metadata are recorded by sinks on the receiver's
#   outputs and matched by frame number after the run, and each frame is checked;
#   v2.7 checked the last frame of each size. The frames are not passed on to
#   CsiToBayerOp, as v2.7 did after a Python operator padded each runt frame to
#   the full frame size.

import dataclasses
import logging

import applications
import holoscan as hs
import operators
import pytest
import utils
from hololink_module.sensors.imx274.imx274_mode import Imx274_Mode

import hololink as hololink_module

# The data plane register that sets where a frame ends (DP_BUFFER_LENGTH in
# src/hololink/core/data_channel.hpp).
DP_BUFFER_LENGTH = 0x18


@dataclasses.dataclass
class FrameRecord:
    frame_number: int
    bytes_written: int


class TensorSizeOp(hs.Operator):
    """Appends (frame_number, tensor size) to the list registered as record_token."""

    input = hs.Input("input", hs.payload.Tensor)

    record_token = hs.Param(0)

    def setup(self, spec):
        spec.input(self.input, queue_depth=4)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, context):
        sample = self.input.receive()
        operators.lookup(self.record_token).append(
            (sample.metadata.frame_id, sample.data.nbytes)
        )


def runt_test(
    camera_mode,
    pattern,
    hololink_address,
    frame_limit,
    trim,
    receiver,
    receiver_kwargs,
):
    frame_records = []
    tensor_sizes = []
    with applications.CudaContext() as cuda_context, applications.imx274_camera(
        hololink_address, camera_mode, pattern=pattern
    ) as camera, utils.Watchdog(
        "frame-reception",
        initial_timeout=operators.color_profiler_initial_timeout(frame_limit),
        min_taps=utils.frame_floor(frame_limit),
    ) as watchdog, utils.RunMonitor() as monitor:
        frame_size = camera.frame_size
        # Our receiver buffer is going to be smaller than
        # the actual frame size; so every frame we get is
        # really one full one and another runt one.
        # (keep our size to an even multiple of 8)
        runt_frame_size = int(frame_size * 0.8) & ~7
        remaining_frame_size = frame_size - runt_frame_size
        expected_frame_sizes = [runt_frame_size, remaining_frame_size]
        vp_address = camera.metadata["vp_address"]

        def device_start():
            camera.device.write_uint32(vp_address + DP_BUFFER_LENGTH, runt_frame_size)
            camera.camera.start()

        test_graph = applications.TestGraph("runt")
        rx = applications.add_receiver(
            test_graph,
            "receiver",
            receiver,
            camera,
            cuda_context.cu_context,
            frame_limit,
            device_start=device_start,
            trim=trim,
            **receiver_kwargs(camera, runt_frame_size),
        )
        size_checker = test_graph.op(
            TensorSizeOp,
            "size_checker",
            record_token=operators.register(tensor_sizes),
        )
        test_graph.add_flow(rx.tensor, size_checker.input, queue_depth=4)
        test_graph.bind("receiver", "tensor")
        record = test_graph.op(
            operators.RecordMetadataOp,
            "record_metadata",
            record_token=operators.register((FrameRecord, frame_records)),
        )
        operators.metadata_flow(test_graph, rx.metadata, record.metadata)
        watchdog_operator = test_graph.op(
            operators.WatchdogOp,
            "watchdog_operator",
            watchdog_token=operators.register(watchdog),
        )
        operators.metadata_flow(test_graph, rx.metadata, watchdog_operator.metadata)
        test_graph.run()

    monitor.assert_ok()
    sizes = dict(tensor_sizes)
    found_sizes = {}
    for frame in frame_records:
        if frame.frame_number not in sizes:
            continue
        bytes_written, num_elements = frame.bytes_written, sizes[frame.frame_number]
        message = f"frame {frame.frame_number}: {bytes_written=:#x} {num_elements=:#x}"
        assert bytes_written <= frame_size, message
        assert bytes_written in expected_frame_sizes, message
        assert num_elements == (bytes_written if trim else frame_size), message
        found_sizes[bytes_written] = num_elements
    for bytes_written, num_elements in found_sizes.items():
        logging.info(f"{bytes_written=:#x} {num_elements=:#x}")
    assert len(found_sizes) == len(expected_frame_sizes)


@pytest.mark.skip_unless_imx274
@pytest.mark.parametrize(
    "camera_mode, pattern",
    [
        (
            Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
            10,
        ),
    ],
)
@pytest.mark.parametrize(
    "trim",
    [
        True,
        False,
    ],
)
def test_linux_imx274_runt(
    camera_mode,
    pattern,
    headless,
    hololink_address,
    frame_limit,
    trim,
):
    runt_test(
        camera_mode,
        pattern,
        hololink_address,
        frame_limit,
        trim,
        hololink_module.operators.LinuxReceiverOp,
        lambda camera, runt_frame_size: {},
    )


def round_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


# The RoCE runt test needs the frame metadata right after the runt data, so it
# gives the receiver the full frame_size and metadata_offset =
# round_up(runt_frame_size, PAGE_SIZE), as v2.7 does. HSDK5's
# RoceReceiverOp::start_receiver() rejects any metadata_offset below the full
# frame ("metadata_offset ... overlaps frame data"), and without the offset the
# receiver doesn't find the metadata (bytes_written=0). Seen on IGX Thor; v2.7
# passes both cases on the same devkit. Remove the skip when the receiver accepts
# an explicit metadata_offset inside the frame buffer.
@pytest.mark.skip(
    reason="HSDK5 RoceReceiverOp rejects a metadata_offset below frame_size, "
    "which the runt layout needs"
)
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
@pytest.mark.parametrize(
    "trim",
    [
        True,
        False,
    ],
)
def test_roce_imx274_runt(
    camera_mode,
    pattern,
    headless,
    hololink_address,
    ibv_name,
    ibv_port,
    frame_limit,
    trim,
):
    PAGE_SIZE = 128

    def receiver_kwargs(camera, runt_frame_size):
        return {
            "ibv_name": ibv_name,
            "ibv_port": ibv_port,
            "metadata_offset": round_up(runt_frame_size, PAGE_SIZE),
        }

    runt_test(
        camera_mode,
        pattern,
        hololink_address,
        frame_limit,
        trim,
        hololink_module.operators.RoceReceiverOp,
        receiver_kwargs,
    )


@pytest.mark.skip_unless_imx274
@pytest.mark.skip_unless_coe
@pytest.mark.parametrize(
    "camera_mode, pattern",
    [
        (
            Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
            10,
        ),
    ],
)
@pytest.mark.parametrize(
    "trim",
    [
        False,
        True,
    ],
)
def test_coe_imx274_runt(
    camera_mode,
    pattern,
    headless,
    hololink_address,
    frame_limit,
    trim,
    coe_interfaces,
):
    def receiver_kwargs(camera, runt_frame_size):
        return {
            "coe_interface": coe_interfaces[0],
            "pixel_width": camera.width,
            "coe_channel": 1,
        }

    runt_test(
        camera_mode,
        pattern,
        hololink_address,
        frame_limit,
        trim,
        hololink_module.operators.LinuxCoeReceiverOp,
        receiver_kwargs,
    )
