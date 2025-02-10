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

import logging
import logging.handlers
import os
import socket
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


def pytest_addoption(parser):
    parser.addoption(
        "--imx274",
        action="store_true",
        default=False,
        help="Don't skip test_udp.",
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
        "--udpcam",
        help="IP address of UDPCam server; omit to create a temporary one.",
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
<<<<<<< HEAD
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
        "--ar0234",
        action="store_true",
        help="Don't skip tests using AR0234.",
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
    if not config.getoption("--imx477"):
        skip_imx477 = pytest.mark.skip(reason="Tests only run in --imx477 mode.")
        for item in items:
            if "skip_unless_imx477" in item.keywords:
                item.add_marker(skip_imx477)
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
    if not config.getoption("--ar0234"):
        skip_ar0234 = pytest.mark.skip(reason="Tests only run in --ar0234 mode.")
        for item in items:
            if "skip_unless_ar0234" in item.keywords:
                item.add_marker(skip_ar0234)


@pytest.fixture
def headless(request):
    return request.config.getoption("--headless")


@pytest.fixture
def frame_limit(request):
    return request.config.getoption("--frame-limit")


@pytest.fixture
def udpcam(request):
    return request.config.getoption("--udpcam")


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
