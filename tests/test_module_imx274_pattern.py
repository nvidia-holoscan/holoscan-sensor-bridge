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
# CsiToBayerOp), plus the shared test helpers in operators.py.
#
# Notable differences from the legacy test, all driven by the module API:
#   * Discovery is Adapter.wait_for_channel(peer_ip, timeout) rather than
#     Enumerator.find_channel; single-interface stereo clones the base
#     EnumerationMetadata and calls Adapter.use_sensor(clone, n).
#   * The module RoceReceiverOp selects its IB device from the peer, so
#     there is no ibv_name parameter; the RoCE-vs-Linux choice is simply
#     which receiver class the application instantiates.
#
# HSDK5 differences:
#   * The frame CRC is checked by hololink's ComputeCrcOp and CheckCrcOp
#     through an hl.CrcChannel (operators.add_crc_check); the module's
#     ComputeCrcOp has no CRC channel and the module has no CheckCrcOp. The
#     bayer (color) CRC of the demosaiced image is checked the same way, by
#     hololink's ComputeCrcImageOp (operators.add_image_crc_check).
#   * There is no FrameAlignerOp in the graph: each leg runs on its own, with its
#     own watchdog, and the legs are paired by timestamp after the run. As v2.7's
#     watchdog after the aligner did, operators.assert_aligned() needs a pair
#     within allowable_dt at least every 0.5 s.
#   * HSDK5 has no selectable schedulers, so the tests no longer take one.
#   * The receivers end the run at the frame limit.

import dataclasses
import logging
import math
import weakref

import applications
import hololink_module.operators
import hololink_module.sensors.imx274 as imx274
import holoscan_vision as hv
import operators
import pytest
import utils

import hololink_module

# Number of frames to skip during pipeline initialization to avoid artifacts
SKIP_INITIAL_FRAMES = 15

# Seconds to wait for each bootp announcement before giving up.
DISCOVERY_TIMEOUT_S = 30.0

# Each run captures this many frames per leg.
FRAME_LIMIT = 100


@dataclasses.dataclass
class FrameRecord:
    frame_number: int
    timestamp_s: int
    timestamp_ns: int
    bytes_written: int


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


class Leg:
    """One camera of the stereo pair, and what the run records for it."""

    def __init__(self, name, metadata, camera, camera_mode, affinity):
        self.name = name
        self.metadata = metadata
        self.camera = camera
        self.camera_mode = camera_mode
        self.affinity = affinity
        self.records = []
        self.crc_checker = None
        self.image_crc_checker = None
        self.frame_size = None


def add_leg(test_graph, leg, cu_context, use_roce, watchdog_token):
    leg.camera.set_mode(leg.camera_mode)
    converter = hololink_module.CsiToBayerConverter()
    leg.camera.configure_converter(converter)
    leg.frame_size = converter.get_csi_length()
    logging.info(f"{leg.name} frame_size={leg.frame_size}")
    bayer_name = leg.camera.bayer_format().name.lower()

    # The module RoceReceiverOp resolves its IB device from the peer in the
    # enumeration metadata, so (unlike the legacy op) there is no
    # ibv_name/ibv_port to pass; device_start/device_stop arm and disarm the
    # sensor at pipeline start/stop.
    MAX_PAGES = 5
    receiver_kwargs = {}
    if use_roce:
        receiver = hololink_module.operators.RoceReceiverOp
    else:
        receiver = hololink_module.operators.LinuxReceiverOp
        receiver_kwargs["receiver_affinity"] = leg.affinity
    rx = test_graph.op(
        receiver,
        f"receiver_{leg.name}",
        metadata=leg.metadata,
        frame_context=int(cu_context),
        frame_size=leg.frame_size,
        device_start=leg.camera.start,
        device_stop=leg.camera.stop,
        pages=MAX_PAGES,
        queue_size=MAX_PAGES,
        frame_limit=FRAME_LIMIT,
        frame_limit_ends_run=True,
        **receiver_kwargs,
    )

    # Compute and capture each raw-frame CRC on a branch from the receiver.
    leg.crc_checker = operators.add_crc_check(
        test_graph, leg.name, rx, leg.frame_size, test_graph.cuda_device_ordinal
    )
    record = test_graph.op(
        operators.RecordMetadataOp,
        f"record_metadata_{leg.name}",
        record_token=operators.register((FrameRecord, leg.records)),
    )
    operators.metadata_flow(test_graph, rx.metadata, record.metadata)

    csi = test_graph.op(
        hololink_module.operators.CsiToBayerOp,
        f"csi_to_bayer_{leg.name}",
        cuda_device_ordinal=test_graph.cuda_device_ordinal,
        geometry=converter.geometry(),
        bayer_format=bayer_name,
    )
    test_graph.add_flow(rx.tensor, csi.in_tensor, queue_depth=4)
    test_graph.add_flow(rx.metadata, csi.in_metadata, queue_depth=4)
    test_graph.bind(f"receiver_{leg.name}", "tensor")
    demosaic = test_graph.op(
        hv.operators.BayerDemosaicOp,
        f"demosaic_{leg.name}",
        bayer_grid_pos=bayer_name,
        generate_alpha=True,
        alpha_value=65535,
    )
    test_graph.add_flow(csi.tensor, demosaic.input, queue_depth=4)
    test_graph.bind(f"csi_to_bayer_{leg.name}", "tensor")
    # Compute the CRC of the bayer image and use that to
    # determine if we're receiving the expected image.
    rgba_image_size = leg.camera._width * leg.camera._height * 4 * 2  # 16-bit RGBA
    leg.image_crc_checker = operators.add_image_crc_check(
        test_graph,
        f"bayer_crc_{leg.name}",
        demosaic.output,
        csi.metadata,
        rgba_image_size,
        test_graph.cuda_device_ordinal,
    )
    test_graph.bind(f"demosaic_{leg.name}", "output")
    watchdog_operator = test_graph.op(
        operators.WatchdogOp,
        f"watchdog_{leg.name}",
        watchdog_token=watchdog_token,
    )
    operators.metadata_flow(test_graph, csi.metadata, watchdog_operator.metadata)
    return applications.Image(demosaic.output, leg.camera._width, leg.camera._height)


def check_leg(leg, expected):
    """Count the short frames, bad CRCs and bad bayer CRCs of a RoCE leg, without
    the first SKIP_INITIAL_FRAMES frames and the last one."""
    records = sorted(leg.records, key=lambda record: record.frame_number)
    logging.info(f"{leg.name} camera captured {len(records)} frames")
    settled = records[SKIP_INITIAL_FRAMES : len(records) - 1]
    short_frames = 0
    for record in settled:
        if record.bytes_written != leg.frame_size:
            logging.error(
                f"{leg.name} short CSI frame: frame_number={record.frame_number:#x}, "
                f"bytes_written={record.bytes_written:#x}, "
                f"expected={leg.frame_size:#x}, "
                f"missing_bytes={leg.frame_size - record.bytes_written}"
            )
            short_frames += 1
    crc_result = leg.crc_checker.result(skip_first=SKIP_INITIAL_FRAMES, skip_last=1)
    for frame_number, reason in crc_result.bad:
        logging.error(f"{leg.name} CRC mismatch at frame {frame_number:#x}: {reason}")
    if settled:
        first, last = settled[0].frame_number, settled[-1].frame_number
        image_crcs = [
            (number, crc)
            for number, (crc, _) in sorted(leg.image_crc_checker.by_frame().items())
            if first <= number <= last
        ]
    else:
        first = last = None
        image_crcs = []
    bad_images = [(number, crc) for number, crc in image_crcs if crc != expected]
    bad_color_crcs = len(bad_images)
    if bad_color_crcs:
        logging.error(
            f"{leg.name}: {bad_color_crcs} of {len(image_crcs)} bayer CRCs aren't "
            f"{expected:#x} (checked frames {first}..{last}); bad (frame_number, "
            f"crc): {[(number, f'{crc:#x}') for number, crc in bad_images]}"
        )
    logging.info(
        f"{leg.name}: {crc_result.checked} frame CRCs and {len(image_crcs)} "
        "bayer CRCs checked"
    )
    minimum = max(11, math.ceil(len(settled) * operators.CRC_COVERAGE))
    assert (
        crc_result.checked >= minimum
    ), f"{leg.name}: only {crc_result.checked} of {len(settled)} frame CRCs checked"
    assert (
        len(image_crcs) >= minimum
    ), f"{leg.name}: only {len(image_crcs)} of {len(settled)} bayer CRCs checked"
    return short_frames, len(crc_result.bad), bad_color_crcs


@pytest.fixture(scope="module")
def cuda_primary_context():
    """Keep one primary-context retain across all parametrized cases."""
    with applications.CudaContext() as cuda_context:
        yield cuda_context.device_ordinal, cuda_context.cu_context


def run_test(
    cuda_primary_context,
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
    headless,
):
    #
    logging.info("Initializing.")
    cu_device_ordinal, cu_context = cuda_primary_context

    #
    # The two sensors share the board's expander; select output 0 for the
    # left sensor and output 1 for the right one (the metadata-based
    # Imx274Cam reads this back when it constructs).
    imx274.Imx274Cam.use_expander_configuration(metadata_left, 0)
    imx274.Imx274Cam.use_expander_configuration(metadata_right, 1)
    camera_left = CameraWrapper(metadata_left)
    camera_right = CameraWrapper(metadata_right)
    left = Leg("left", metadata_left, camera_left, camera_mode_left, [2])
    right = Leg("right", metadata_right, camera_right, camera_mode_right, [3])
    ready_frame = 15  # ignore this many initial frames while the pipeline initializes
    initial_timeout = utils.timeout_sequence(
        # The last 10 frames have the long timeout: a leg can end a few frames
        # short of FRAME_LIMIT (frames dropped on the way), and teardown follows.
        [(30, ready_frame), (0.5, FRAME_LIMIT - ready_frame - 10), (30, 1)]
    )
    with utils.leg_watchdogs(
        (left.name, right.name),
        initial_timeout=initial_timeout,
        min_taps=utils.frame_floor(FRAME_LIMIT),
    ) as watchdogs, utils.RunMonitor() as monitor:
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
                #
                test_graph = applications.TestGraph(
                    "imx274_pattern", cuda_device_ordinal=cu_device_ordinal
                )
                images = [
                    add_leg(
                        test_graph,
                        leg,
                        cu_context,
                        use_roce,
                        operators.register(watchdogs[leg.name]),
                    )
                    for leg in (left, right)
                ]
                applications.show_images(
                    test_graph, "holoviz", images, headless, "IMX274 pattern test"
                )
                test_graph.run()
            finally:
                hololink.stop()

    monitor.assert_ok()
    operators.assert_aligned(
        [(r.timestamp_s + r.timestamp_ns * 1e-9, r) for r in left.records],
        [(r.timestamp_s + r.timestamp_ns * 1e-9, r) for r in right.records],
        allowable_dt,
        min_pairs=utils.frame_floor(FRAME_LIMIT),
        skip_first=ready_frame,
    )

    # Verify CRCs match. Only the RoCE path delivers a lossless frame
    # whose CRC we can trust; the Linux-socket path expects packet loss,
    # so its image data is not evaluated.
    left_short_frames = left_bad_crcs = left_bad_color_crcs = 0
    right_short_frames = right_bad_crcs = right_bad_color_crcs = 0
    if use_roce:
        left_short_frames, left_bad_crcs, left_bad_color_crcs = check_leg(
            left, expected_left
        )
        right_short_frames, right_bad_crcs, right_bad_color_crcs = check_leg(
            right, expected_right
        )

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


# Each tuple carries an allowable_dt: the largest skew between the two legs
# for which their frames are paired. IMX274 cannot hardware-sync, so the
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


@utils.requires_module_roce
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
    module_dir,
):
    if len(channel_ips) < 2:
        pytest.skip("--channel-ips needs at least two IPs for this test")
    hololink_left, hololink_right = channel_ips[0], channel_ips[1]
    # Get a handle to data sources
    channel_metadata_left = _wait_for_channel(module_dir, hololink_left)
    channel_metadata_right = _wait_for_channel(module_dir, hololink_right)
    run_test(
        cuda_primary_context,
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
        headless,
    )


@utils.requires_module_roce
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
        cuda_primary_context,
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
        headless,
    )


@utils.requires_module_roce
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
        cuda_primary_context,
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
        headless,
    )


# Test stereo patterns across a single network interface. Run once per board
# (channel_index selects channel_ips[channel_index]) so that each interface is
# verified to carry both of its sensors. The module resolves the IB device
# from the peer metadata, so there's no ibv_name to pair with the index.
@utils.requires_module_roce
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
        cuda_primary_context,
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
        headless,
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
        cuda_primary_context,
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
        headless,
    )
