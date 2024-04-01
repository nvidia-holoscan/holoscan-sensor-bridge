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
import os
import threading
import traceback
from unittest.mock import patch

import pytest

import hololink as hololink_module


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
    hololink_module.Hololink._reset_framework()


@pytest.fixture(scope="function", autouse=True)
def hololink_session(request):
    request.addfinalizer(_hololink_session_finalizer)


def pytest_addoption(parser):
    parser.addoption(
        "--udp-server",
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
    default_infiniband_interface = "mlx5_0"
    try:
        default_infiniband_interface = sorted(os.listdir("/sys/class/infiniband"))[0]
    except (FileNotFoundError, IndexError):
        pass
    parser.addoption(
        "--ibv-name",
        default=default_infiniband_interface,
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


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--udp-server"):
        skip_udp_server = pytest.mark.skip(
            reason="Tests only run in --udp-server mode."
        )
        for item in items:
            if "skip_unless_udp_server" in item.keywords:
                item.add_marker(skip_udp_server)
    if config.getoption("--unaccelerated-only"):
        skip_accelerated_networking = pytest.mark.skip(
            reason="Don't run network accelerated tests when --unaccelerated-only is specified."
        )
        for item in items:
            if "accelerated_networking" in item.keywords:
                item.add_marker(skip_accelerated_networking)


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
