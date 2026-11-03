# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Test IMX274 settings synchronized with frame-end: for each received frame, an
# operator schedules the next test pattern with a frame-end sequencer, and each
# frame must show the pattern that was scheduled two frames before it.
#
# HSDK5 differences from v2.7:
# - The native hl.NativeImx274Sensor configures the camera and starts and stops it
#   for the receiver. It has no synchronized_test_pattern_update(), which v2.7's
#   Python dual_imx274.Imx274Cam had, so the test encodes the same I2C writes
#   into the frame-end sequencer itself.
# - The frame CRC and the CRC of the demosaiced image are computed on the GPU and
#   checked through an hl.CrcChannel (operators.add_crc_check() and
#   operators.add_image_crc_check()), as v2.7 did with ComputeCrcOp and
#   CheckCrcOp. The image is bit-exact with v2.7's, so the expected values are
#   unchanged.
# - Results are matched by frame number instead of by position in one metadata
#   record, because HSDK5 operators can't add fields to the frame metadata.

import logging
import math

import applications
import operators
import pytest
import utils
from hololink_module.sensors.imx274.imx274_mode import Imx274_Mode

import hololink as hololink_module

# The I2C addresses of the camera and of the board's I2C expander, and the expander
# output that selects each camera, as in v2.7's dual_imx274 and li_i2c_expander.
CAM_I2C_ADDRESS = 0x1A
I2C_EXPANDER_ADDRESS = 0x70
EXPANDER_OUTPUTS = {0: 0b0001, 1: 0b0010}
TEST_PATTERN_REGISTER = 0x303D

# fmt: off
imx274_expected_color_profile = {
    10: 0xA9072F9B,
    11: 0xD5A4409B,
}
# fmt: on


def synchronized_test_pattern_update(channel, expander_configuration, pattern):
    """Set the camera's test pattern when the current frame ends, as v2.7's
    dual_imx274.Imx274Cam.synchronized_test_pattern_update() did: select the camera
    on the I2C expander and write the register, both with the channel's frame-end
    sequencer."""
    sequencer = channel.frame_end_sequencer()
    i2c = channel.hololink().get_i2c(hololink_module.CAM_I2C_BUS)
    output = EXPANDER_OUTPUTS[expander_configuration]
    i2c.encode_i2c_request(sequencer, I2C_EXPANDER_ADDRESS, bytes([output]), 0)
    register = TEST_PATTERN_REGISTER
    write_bytes = bytes([register >> 8, register & 0xFF, pattern])
    i2c.encode_i2c_request(sequencer, CAM_I2C_ADDRESS, write_bytes, 0)
    sequencer.enable()


class Leg:
    """One camera, the pattern schedule for it, and what the run records."""

    def __init__(
        self, name, camera, expander_configuration, ibv_name, ibv_port, patterns
    ):
        self.name = name
        self.camera = camera  # applications.SensorCamera
        self.expander_configuration = expander_configuration
        self.ibv_name = ibv_name
        self.ibv_port = ibv_port
        self.patterns = patterns
        # The current received frame was configured two frames ago;
        # pattern_queue keeps track of this.
        pattern = patterns[-1]
        self.pattern_queue = [pattern, pattern]
        self.received_frame_numbers = []
        self.crc_checker = None
        self.image_crc_checker = None

    def update_camera(self, frame_number):
        # Called for each received frame, in order.
        self.received_frame_numbers.append(frame_number)
        n = len(self.received_frame_numbers) % len(self.patterns)
        next_pattern = self.patterns[n]
        synchronized_test_pattern_update(
            self.camera.channel, self.expander_configuration, next_pattern
        )
        self.pattern_queue.append(next_pattern)


class UpdateCameraOp(operators.OnFrameNOperator):
    """Calls the registered callback with each frame's frame number."""

    def compute(self, context):
        metadata = self.metadata.receive_data()
        operators.lookup(self.callback_token)(metadata.frame_number)


def add_leg(test_graph, leg, cu_context, frame_limit, watchdog_token):
    camera = leg.camera
    rx = applications.add_receiver(
        test_graph,
        f"{leg.name}_receiver",
        hololink_module.operators.RoceReceiverOp,
        camera,
        cu_context,
        frame_limit,
        ibv_name=leg.ibv_name,
        ibv_port=leg.ibv_port,
        # Prevent reuse while downstream GPU operators consume a page.
        pages=5,
        # Absorb brief receiver-to-operator scheduling delays.
        queue_size=5,
    )
    update_camera = test_graph.op(
        UpdateCameraOp,
        f"{leg.name}_update_camera",
        callback_token=operators.register(leg.update_camera),
    )
    operators.metadata_flow(test_graph, rx.metadata, update_camera.metadata)
    leg.crc_checker = operators.add_crc_check(
        test_graph,
        f"{leg.name}_crc",
        rx,
        camera.frame_size,
        test_graph.cuda_device_ordinal,
    )
    isp = applications.add_isp(test_graph, f"{leg.name}_", rx, camera)
    rgba_image_size = camera.width * camera.height * 4 * 2  # 16-bit RGBA
    leg.image_crc_checker = operators.add_image_crc_check(
        test_graph,
        f"{leg.name}_image_crc",
        isp["demosaic"].output,
        isp["image_processor"].metadata,
        rgba_image_size,
        test_graph.cuda_device_ordinal,
    )
    watchdog = test_graph.op(
        operators.WatchdogOp, f"{leg.name}_watchdog", watchdog_token=watchdog_token
    )
    operators.metadata_flow(
        test_graph, isp["image_processor"].metadata, watchdog.metadata
    )
    return applications.Image(isp["demosaic"].output, camera.width, camera.height)


def check_records(leg, ready_frame):
    # the pipeline sees frames that we don't due
    # to longer startup for some pipeline elements
    shutdown_frames = len(leg.received_frame_numbers) - 10
    crcs = leg.crc_checker.by_frame()
    image_crcs = {
        frame_number: crc
        for frame_number, (crc, _) in leg.image_crc_checker.by_frame().items()
    }
    bad = False
    image_checked = 0
    for n, frame_number in enumerate(leg.received_frame_numbers):
        pattern = leg.pattern_queue[n]
        expected_image_crc = imx274_expected_color_profile[pattern]
        computed_crc, received_crc = crcs.get(frame_number, (None, None))
        image_crc = image_crcs.get(frame_number)
        message = (
            f"{leg.name=} {n=} {frame_number=} {received_crc=} {computed_crc=} "
            f"{pattern=} {expected_image_crc=:#x} {image_crc=}"
        )
        ok = received_crc is not None and received_crc == computed_crc
        if image_crc is not None:
            ok = ok and image_crc == expected_image_crc
        if ok:
            logging.debug(message)
            if image_crc is not None and ready_frame <= n <= shutdown_frames:
                image_checked += 1
            continue
        if n < ready_frame:
            logging.debug(message)
            logging.info(f"Frame {n} didn't match but is during startup; ignoring.")
            continue
        if n > shutdown_frames:
            logging.debug(message)
            logging.info(f"Frame {n} didn't match but is during shutdown; ignoring.")
            continue
        logging.error(message)
        bad = True
    # v2.7 checked the image of each frame; the GPU check misses only frames it
    # can't keep up with.
    window = max(0, shutdown_frames - ready_frame + 1)
    minimum = math.ceil(window * operators.CRC_COVERAGE)
    logging.info(f"{leg.name}: {image_checked} of {window} image CRCs checked")
    assert (
        image_checked >= minimum
    ), f"{leg.name}: only {image_checked} of {window} image CRCs checked"
    return bad


def run_i2c_test(legs, frame_limit, headless):
    ready_frame = 10  # ignore this many initial frames while the pipeline initializes
    # The last 10 frames have the long timeout: a leg can end a few frames short of
    # frame_limit (frames dropped on the way), and the run's teardown follows.
    initial_timeout = utils.timeout_sequence(
        [(30, ready_frame), (0.5, frame_limit - ready_frame - 10), (30, 1)]
    )
    with applications.CudaContext() as cuda_context, utils.leg_watchdogs(
        [leg.name for leg in legs],
        initial_timeout=initial_timeout,
        min_taps=utils.frame_floor(frame_limit),
    ) as watchdogs, utils.RunMonitor() as monitor:
        test_graph = applications.TestGraph(
            "imx274_i2c", cuda_device_ordinal=cuda_context.device_ordinal
        )
        images = [
            add_leg(
                test_graph,
                leg,
                cuda_context.cu_context,
                frame_limit,
                operators.register(watchdogs[leg.name]),
            )
            for leg in legs
        ]
        applications.show_images(
            test_graph, "holoviz", images, headless, "IMX274 I2C test"
        )
        test_graph.run()
    monitor.assert_ok()
    return [check_records(leg, ready_frame) for leg in legs]


@utils.requires_roce
@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [
        Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
    ],
)
@pytest.mark.parametrize(
    "patterns",
    [
        list(imx274_expected_color_profile.keys()),
    ],
)
def test_imx274_synchronized_i2c_settings(
    camera_mode,
    patterns,
    headless,
    frame_limit,
    hololink_address,
    ibv_name,
    ibv_port,
):
    # As in v2.7, the digital gain isn't set: the expected image CRCs are of the
    # patterns that configure() leaves.
    with applications.imx274_camera(
        hololink_address, camera_mode, pattern=patterns[-1], digital_gain=None
    ) as camera:
        leg = Leg("camera", camera, 0, ibv_name, ibv_port, patterns)
        (bad,) = run_i2c_test([leg], frame_limit, headless)
    assert not bad


@utils.requires_roce
@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode_left, camera_mode_right",  # noqa: E501
    [
        (
            Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
            Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
        ),
    ],
)
@pytest.mark.parametrize(
    "patterns",
    [
        list(imx274_expected_color_profile.keys()),
    ],
)
def test_stereo_imx274_synchronized_i2c_settings(
    camera_mode_left,
    camera_mode_right,
    patterns,
    headless,
    frame_limit,
    channel_ips,
):
    if len(channel_ips) < 2:
        pytest.skip("--channel-ips needs at least two IPs for this test")
    # As in test_imx274_synchronized_i2c_settings, the digital gain isn't set.
    with applications.imx274_camera(
        channel_ips[0],
        camera_mode_left,
        pattern=patterns[-1],
        expander_configuration=0,
        digital_gain=None,
    ) as camera_left:
        # camera_left's setup_clock() also sets camera_right's clock.
        camera_right = applications.open_imx274(
            channel_ips[1],
            camera_mode_right,
            pattern=patterns[-1],
            expander_configuration=1,
            reset=False,
            device=camera_left.device,
            digital_gain=None,
        )
        assert camera_left.device is camera_right.channel.hololink()
        # Each leg's receiver uses the IBV device on its own network interface.
        left = Leg("left", camera_left, 0, "", 1, patterns)
        right = Leg("right", camera_right, 1, "", 1, patterns)
        bad_left, bad_right = run_i2c_test([left, right], frame_limit, headless)
    assert not bad_left and not bad_right
