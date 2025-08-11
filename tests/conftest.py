# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import logging
import logging.handlers
import os
import socket
import subprocess
import threading
import traceback
from unittest.mock import patch

import pytest

import hololink as hololink_module

# If desired, forward python logging to UDP port 514 in not exactly a SYSLOG
# compatible way but a way that works great with wireshark.  This
# is the same thing we're doing in C++.
if False:

    class UdpWriter:
        def __init__(self, sender_ip, destination_ip="255.255.255.255"):
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self._socket.bind((sender_ip, 0))
            self._socket.connect((destination_ip, 514))

        def write(self, msg):
            self._socket.send(msg.encode())

    udp_writer = UdpWriter(sender_ip="127.0.0.1", destination_ip="127.0.0.1")
    handler = logging.StreamHandler(stream=udp_writer)
    formatter = logging.Formatter(
        fmt="%(levelname)s %(relativeCreated)d %(funcName)s %(filename)s:%(lineno)d tid=%(threadName)s -- %(message)s"
    )
    handler.setFormatter(formatter)
    handler.terminator = ""
    logger = logging.getLogger()
    logger.addHandler(handler)


@pytest.fixture(scope="function", autouse=True)
def forward_exception():
    caught_exception = None

    class ThreadWrapper(threading.Thread):
        def run(self):
            try:
                super().run()
            except BaseException as e:
                logging.error("Caught %s (%s)" % (e, type(e)))
                tb = traceback.format_exc()
                for s in tb.split("\n"):
                    logging.error(s)
                nonlocal caught_exception
                caught_exception = e
                os._exit(1)

    with patch("threading.Thread", ThreadWrapper):
        yield
        if caught_exception:
            raise caught_exception


def _hololink_session_finalizer():
    hololink_module.Hololink.reset_framework()


@pytest.fixture(scope="function", autouse=True)
def hololink_session(request):
    request.addfinalizer(_hololink_session_finalizer)


# Produce a default list of COE interfaces.  For now
# we just find the device names associated with the
# routes to 192.168.0.2 and 192.168.0.3 and provide
# those.  Override those using the "--coe-interface=..."
# switch.
def coe_devices():
    device_addresses = ["192.168.0.2", "192.168.0.3"]
    r = []
    for device_address in device_addresses:
        command = f"ip -j route get {device_address}".split(" ")
        try:
            process = subprocess.run(
                command, capture_output=True, text=True, check=True
            )
            route_info = json.loads(process.stdout)
            dev = route_info[0].get("dev")
            r.append(dev)
        except Exception as e:
            logging.error(
                f'Skipping COE interface detection for "{device_address}" due to {e}.'
            )
    return ",".join(r)


def pytest_addoption(parser):
    parser.addoption(
        "--imx274",
        action="store_true",
        default=False,
        help="Don't skip tests specific to IMX274",
    )
    parser.addoption(
        "--headless",
        action="store_true",
        default=False,
        help="Run holoviz in headless mode",
    )
    parser.addoption(
        "--frame-limit",
        help="For test_imx274_player, stop after this many frames.",
        default=300,
        type=int,
    )
    parser.addoption(
        "--mock-camera",
        help="IP address of MockCamera server; omit to create a temporary one.",
    )
    parser.addoption(
        "--hololink",
        default="192.168.0.2",
        help="IP address of Hololink controller to use.",
    )
    parser.addoption(
        "--igpu",
        action="store_true",
        default=False,
        help="Don't skip igpu based test.",
    )
    parser.addoption(
        "--dgpu",
        action="store_true",
        default=False,
        help="Don't skip dgpu based test.",
    )
    infiniband_interfaces = hololink_module.infiniband_devices()
    parser.addoption(
        "--ibv-name",
        default=infiniband_interfaces[0] if infiniband_interfaces else None,
        help="IBV device to use",
    )
    parser.addoption(
        "--ibv-port",
        type=int,
        default=1,
        help="IBVerbs device port to use",
    )
    parser.addoption(
        "--unaccelerated-only",
        action="store_true",
        default=False,
        help="Don't run tests requiring accelerated networking",
    )
    parser.addoption(
        "--ptp",
        action="store_true",
        default=False,
        help="Don't skip tests requiring PTP support",
    )
    parser.addoption(
        "--imx477",
        action="store_true",
        default=False,
        help="Include tests for IMX477.",
    )
    parser.addoption(
        "--stereo-imx477",
        action="store_true",
        default=False,
        help="Include tests for Stereo IMX477.",
    )
    parser.addoption(
        "--imx715",
        action="store_true",
        default=False,
        help="Include tests for IMX715.",
    )
    parser.addoption(
        "--hsb",
        action="store_true",
        default=False,
        help="Don't skip tests using HSB.",
    )
    parser.addoption(
        "--hsb-nano",
        action="store_true",
        default=False,
        help="Don't skip tests using HSB Nano.",
    )
    parser.addoption(
        "--channel-ips",
        default=["192.168.0.2", "192.168.0.3"],
        nargs="+",
        help="Use these data plane addresses.",
    )
    parser.addoption(
        "--schedulers",
        default=["default", "greedy", "multithread", "event"],
        nargs="+",
        help="Use these schedulers.",
    )
    parser.addoption(
        "--vb1940",
        action="store_true",
        default=False,
        help="Don't skip tests using VB1940.",
    )
    coe_interfaces = coe_devices()
    parser.addoption(
        "--coe-interface",
        default=coe_interfaces,
        help="Run 1722 COE tests using the given interface-name(s).",
    )
    parser.addoption(
        "--ecam0m30tof",
        action="store_true",
        default=False,
        help="Include tests for ECam0M30Tof.",
    )
    parser.addoption(
        "--audio",
        action="store_true",
        default=False,
        help="Include tests for Audio.",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--imx274"):
        skip_imx274 = pytest.mark.skip(reason="Tests only run in --imx274 mode.")
        for item in items:
            if "skip_unless_imx274" in item.keywords:
                item.add_marker(skip_imx274)
    if config.getoption("--unaccelerated-only"):
        skip_accelerated_networking = pytest.mark.skip(
            reason="Don't run network accelerated tests when --unaccelerated-only is specified."
        )
        for item in items:
            if "accelerated_networking" in item.keywords:
                item.add_marker(skip_accelerated_networking)
    if not config.getoption("--igpu"):
        skip_igpu = pytest.mark.skip(reason="Tests only run in --igpu mode.")
        for item in items:
            if "skip_unless_igpu" in item.keywords:
                item.add_marker(skip_igpu)
    if not config.getoption("--dgpu"):
        skip_dgpu = pytest.mark.skip(reason="Tests only run in --dgpu mode.")
        for item in items:
            if "skip_unless_dgpu" in item.keywords:
                item.add_marker(skip_dgpu)
    if not config.getoption("--ptp"):
        skip_ptp = pytest.mark.skip(reason="Tests only run in --ptp mode.")
        for item in items:
            if "skip_unless_ptp" in item.keywords:
                item.add_marker(skip_ptp)
    # --stereo-imx477 implies --imx477; but not the other way around
    if not config.getoption("--imx477") and not config.getoption("--stereo-imx477"):
        skip_imx477 = pytest.mark.skip(
            reason="Tests only run in --imx477 or --stereo-imx477 mode."
        )
        for item in items:
            if "skip_unless_imx477" in item.keywords:
                item.add_marker(skip_imx477)
    # but --imx477 does not imply --stereo-imx477
    if not config.getoption("--stereo-imx477"):
        skip_stereo_imx477 = pytest.mark.skip(
            reason="Tests only run in --stereo-imx477 mode."
        )
        for item in items:
            if "skip_unless_stereo_imx477" in item.keywords:
                item.add_marker(skip_stereo_imx477)
    if not config.getoption("--hsb") and not config.getoption("--imx274"):
        skip_hsb = pytest.mark.skip(reason="Tests only run in --hsb mode.")
        for item in items:
            if "skip_unless_hsb" in item.keywords:
                item.add_marker(skip_hsb)
    if not config.getoption("--hsb-nano"):
        skip_hsb_nano = pytest.mark.skip(reason="Tests only run in --hsb-nano mode.")
        for item in items:
            if "skip_unless_hsb_nano" in item.keywords:
                item.add_marker(skip_hsb_nano)
    if not config.getoption("--vb1940"):
        skip_vb1940 = pytest.mark.skip(reason="Tests only run in --vb1940 mode.")
        for item in items:
            if "skip_unless_vb1940" in item.keywords:
                item.add_marker(skip_vb1940)
    if not config.getoption("--coe-interface"):
        skip_coe = pytest.mark.skip(
            reason="Tests only run when --coe-interface=(interface,...) is given."
        )
        for item in items:
            if "skip_unless_coe" in item.keywords:
                item.add_marker(skip_coe)
    if not config.getoption("--imx715"):
        skip_imx715 = pytest.mark.skip(reason="Tests only run in --imx715 mode.")
        for item in items:
            if "skip_unless_imx715" in item.keywords:
                item.add_marker(skip_imx715)
    if not config.getoption("--ecam0m30tof"):
        skip_ecam0m30tof = pytest.mark.skip(
            reason="Tests only run in --ecam0m30tof mode."
        )
        for item in items:
            if "skip_unless_ecam0m30tof" in item.keywords:
                item.add_marker(skip_ecam0m30tof)
    if not config.getoption("--mock-camera"):
        skip_mock_camera = pytest.mark.skip(
            reason="Tests only run in --mock-camera mode."
        )
        for item in items:
            if "skip_unless_mock_camera" in item.keywords:
                item.add_marker(skip_mock_camera)
    if not config.getoption("--audio"):
        skip_audio = pytest.mark.skip(reason="Tests only run in --audio mode.")
        for item in items:
            if "skip_unless_audio" in item.keywords:
                item.add_marker(skip_audio)


@pytest.fixture
def headless(request):
    return request.config.getoption("--headless")


@pytest.fixture
def frame_limit(request):
    return request.config.getoption("--frame-limit")


@pytest.fixture
def mock_camera_ip(request):
    return request.config.getoption("--mock-camera")


@pytest.fixture
def hololink_address(request):
    return request.config.getoption("--hololink")


@pytest.fixture
def ibv_port(request):
    return request.config.getoption("--ibv-port")


@pytest.fixture
def ibv_name(request):
    return request.config.getoption("--ibv-name")


# If a test has an "channel_ips" (plural) fixture, then pass
# the list from the command line.
@pytest.fixture
def channel_ips(request):
    return request.config.getoption("--channel-ips")


# If a test runs on COE, provide the interface name(s) for that.
# NOTE that this is a list which may be empty.
@pytest.fixture
def coe_interfaces(request):
    s = request.config.getoption("--coe-interface")
    return s.split(",")


# "ptp_enable" is True if the user specified "--ptp" on
# the command line.
@pytest.fixture
def ptp_enable(request):
    return request.config.getoption("--ptp")


@pytest.fixture
def audio_enable(request):
    return request.config.getoption("--audio")


def pytest_generate_tests(metafunc):
    # If a test has an "channel_ip" (singular) fixture, then parameterize
    # from the list given on the command line.
    if "channel_ip" in metafunc.fixturenames:
        channel_ips = metafunc.config.getoption("--channel-ips")
        metafunc.parametrize("channel_ip", channel_ips)
    # If a test has a "scheduler" (singular) fixture, then parameterize
    # from the list given on the command line.
    if "scheduler" in metafunc.fixturenames:
        schedulers = metafunc.config.getoption("--schedulers")
        metafunc.parametrize("scheduler", schedulers)
