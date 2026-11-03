# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# HSDK5 differences from v2.7: the test graphs follow
# examples/linux_coe_imx274_player.py (LinuxCoeReceiverOp -> converter ->
# ImageProcessorOp -> BayerDemosaicOp), with v2.7's converter: the HSB's
# packetizer packs the pixels and PackedFormatConverterOp unpacks them (see
# applications.packed_camera()). The receivers end the run at the frame limit.

import logging
import sys
from unittest import mock

import applications
import operators
import pytest
import utils
from hololink_module.sensors.imx274.imx274_mode import Imx274_Mode

import hololink as hololink_module
from examples import linux_coe_imx274_player

COE_STALL_TIMEOUT_MS = 4500


# Unaccelerated, single camera test
@pytest.mark.skip_unless_coe
@pytest.mark.skip_unless_imx274
@pytest.mark.reset_hsb_before_retry
@pytest.mark.flaky(reruns=2)
def test_coe_linux_player(
    headless, frame_limit, hololink_address, coe_interfaces, capsys
):
    arguments = [
        sys.argv[0],
        "--frame-limit",
        str(frame_limit),
        "--coe-interface",
        coe_interfaces[0],
        "--stall-timeout-ms",
        str(COE_STALL_TIMEOUT_MS),
        "--hololink",
        hololink_address,
    ]
    if headless:
        arguments.extend(["--headless"])

    with utils.RunMonitor() as monitor, mock.patch("sys.argv", arguments):
        result = linux_coe_imx274_player.main()

    assert result == 0
    # The app ends the run with request_stop() at its frame limit, which its
    # receiver doesn't know, so RunMonitor can't relate the floor to it.
    monitor.assert_ok(min_frames=utils.frame_floor(frame_limit), allow_stopped=True)

    # check for errors
    captured = capsys.readouterr()
    assert captured.err == ""


def add_coe_leg(
    test_graph,
    prefix,
    camera,
    cu_context,
    frame_limit,
    coe_interface,
    coe_channel,
    receiver_affinity,
    watchdog_token,
):
    # As in v2.7, the HSB's packetizer packs the pixels.
    camera = applications.packed_camera(camera)
    rx = applications.add_receiver(
        test_graph,
        f"{prefix}receiver",
        hololink_module.operators.LinuxCoeReceiverOp,
        camera,
        cu_context,
        frame_limit,
        coe_interface=coe_interface,
        pixel_width=camera.width,
        # Each camera sharing a network connection must use
        # a unique channel number from 0..63.
        coe_channel=coe_channel,
        stall_timeout_ms=COE_STALL_TIMEOUT_MS,
        **({"receiver_affinity": receiver_affinity} if receiver_affinity else {}),
    )
    isp = applications.add_isp(test_graph, prefix, rx, camera, packed=True)
    watchdog_operator = test_graph.op(
        operators.WatchdogOp, f"{prefix}watchdog", watchdog_token=watchdog_token
    )
    operators.metadata_flow(
        test_graph, isp["image_processor"].metadata, watchdog_operator.metadata
    )
    return applications.Image(isp["demosaic"].output, camera.width, camera.height)


def run_dual_coe_test(
    frame_limit,
    channel_ip_left,
    camera_mode_left,
    pattern_left,
    channel_ip_right,
    camera_mode_right,
    pattern_right,
    coe_interface_left,
    coe_interface_right,
    headless,
):
    #
    logging.info("Initializing.")
    with applications.CudaContext() as cuda_context, applications.imx274_camera(
        channel_ip_left,
        camera_mode_left,
        pattern=pattern_left,
        expander_configuration=0,
    ) as camera_left, utils.leg_watchdogs(
        ("left", "right"),
        initial_timeout=operators.color_profiler_initial_timeout(frame_limit),
        min_taps=utils.frame_floor(frame_limit),
    ) as watchdogs, utils.RunMonitor() as monitor:
        # camera_left's setup_clock() also sets camera_right's clock.
        camera_right = applications.open_imx274(
            channel_ip_right,
            camera_mode_right,
            pattern=pattern_right,
            expander_configuration=1,
            reset=False,
            device=camera_left.device,
        )
        assert camera_left.device is camera_right.channel.hololink()
        test_graph = applications.TestGraph("dual_coe")
        left = add_coe_leg(
            test_graph,
            "left_",
            camera_left,
            cuda_context.cu_context,
            frame_limit,
            coe_interface_left,
            1,
            [2],
            operators.register(watchdogs["left"]),
        )
        right = add_coe_leg(
            test_graph,
            "right_",
            camera_right,
            cuda_context.cu_context,
            frame_limit,
            coe_interface_right,
            2,
            [3],
            operators.register(watchdogs["right"]),
        )
        applications.show_images(
            test_graph, "holoviz", [left, right], headless, "IMX274 pattern test"
        )
        test_graph.run()
    assert len(monitor.frames) == 2, monitor.frames
    monitor.assert_ok()


# Test COE modes.
@pytest.mark.skip_unless_imx274
@pytest.mark.skip_unless_coe
@pytest.mark.parametrize(
    "camera_mode_left, pattern_left, camera_mode_right, pattern_right",  # noqa: E501
    [
        (  # left
            Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
            10,
            # right
            Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
            11,
        ),
    ],
)
def test_dual_imx274_pattern_coe(
    frame_limit,
    camera_mode_left,
    pattern_left,
    camera_mode_right,
    pattern_right,
    headless,
    channel_ips,
    coe_interfaces,
):
    # This only works if there are two interfaces listed in channel_ips.
    if len(channel_ips) < 2:
        pytest.skip(
            "Test is valid only when --channel-ips has more than one interface."
        )
    # This only works if there are two interfaces listed in coe_interfaces.
    if len(coe_interfaces) < 2:
        pytest.skip(
            "Test is valid only with --coe-interface is given with more than one interface."
        )
    # We're unaccelerated; don't test the results.
    run_dual_coe_test(
        frame_limit,
        channel_ips[0],
        camera_mode_left,
        pattern_left,
        channel_ips[1],
        camera_mode_right,
        pattern_right,
        coe_interface_left=coe_interfaces[0],
        coe_interface_right=coe_interfaces[1],
        headless=headless,
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
    # coe_interface is an index to coe_interfaces. The channel IP for
    # each case is read from --channel-ips at index `expander_configuration`
    # (so case 0 uses channel_ips[0], case 1 uses channel_ips[1]) —
    # matching the original hardcoded pairing of 192.168.0.{2,3} with
    # the two camera positions.
    "expander_configuration, coe_interface",
    [
        (0, 0),
        (1, 1),
    ],
)
def test_imx274_pattern_coe(
    camera_mode,
    pattern,
    headless,
    channel_ips,
    expander_configuration,
    coe_interface,
    coe_interfaces,
    frame_limit,
):
    if expander_configuration >= len(channel_ips):
        pytest.skip(
            f"--channel-ips has {len(channel_ips)} IP(s); this case needs "
            f"index {expander_configuration}"
        )
    if coe_interface >= len(coe_interfaces):
        pytest.skip(
            f"--coe-interface has {len(coe_interfaces)} interface(s); this case "
            f"needs index {coe_interface}"
        )
    logging.info("Initializing.")
    with applications.CudaContext() as cuda_context, applications.imx274_camera(
        channel_ips[expander_configuration],
        camera_mode,
        pattern=pattern,
        expander_configuration=expander_configuration,
    ) as camera, utils.Watchdog(
        "frame-reception",
        initial_timeout=operators.color_profiler_initial_timeout(frame_limit),
        min_taps=utils.frame_floor(frame_limit),
    ) as watchdog, utils.RunMonitor() as monitor:
        test_graph = applications.TestGraph("coe")
        image = add_coe_leg(
            test_graph,
            "",
            camera,
            cuda_context.cu_context,
            frame_limit,
            coe_interfaces[coe_interface],
            1,
            None,
            operators.register(watchdog),
        )
        applications.show_images(test_graph, "holoviz", [image], headless, "Visualizer")
        test_graph.run()
    monitor.assert_ok()
