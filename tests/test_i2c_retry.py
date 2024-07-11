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

import logging

import udp_server

import hololink as hololink_module


class MockHololink(hololink_module.Hololink):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._drop = 0
        self._sent = 0

    def drop(self, n):
        self._drop = n

    def send_control(self, request):
        logging.trace(f"{request=}")
        if self._drop > 0:
            self._drop -= 1
            if self._drop == 0:
                logging.debug(f"dropping {request=}")
                return
        super().send_control(request)
        self._sent += 1


class I2cProxy:
    def __init__(self, i2c):
        self._i2c = i2c

    def __getattr__(self, attr):
        return getattr(self._i2c, attr)

    def i2c_transaction(self, *args, **kwargs):
        logging.debug("i2c_transaction(args=%s, kwargs=%s)" % (args, kwargs))
        return self._i2c.i2c_transaction(*args, **kwargs)


def test_i2c_retry(udpcam):
    logging.info("Initializing.")

    with udp_server.TestServer(udpcam) as server:
        channel_metadata = server.channel_metadata()

        def create_hololink(metadata):
            # Workaround for what's probably a pybind11 bug
            # Background:
            # This callback creates a Python instance of type MockHololink which inherits from
            # the C++ Hololink class. pybind11 keeps an internal list to automatic tranlate between C++ and
            # Python instances. The MockHololink instance returned from here is automatically present as a
            # Hololink class in C++ and referenced by DataChannel. When getting back the instance through
            # `hololink_channel.hololink()` pybind11 see that the instance is a MockHololink instance.
            # Problem:
            # The instance we get back from `hololink_channel.hololink()` is not a MockHololink
            # instance, but a Hololink instance. And accessing `hololink._sent` fails. The reason for this
            # is that the MockHololink instance is no longer present in pybind11's internal registry. Breakpoints
            # to the pybind11 `deregister_instance()` are not triggered. It's unclear why the instance is deleted,
            # passing it back to C++ should keep it alive.
            # Workaround:
            # Keep a global reference to the Mockhololink instance to prevent it from beeing destroyed.
            global mh
            mh = MockHololink(
                metadata["peer_ip"], metadata["control_port"], metadata["serial_number"]
            )
            return mh

        hololink_channel = hololink_module.DataChannel(
            channel_metadata, create_hololink
        )
        hololink = hololink_channel.hololink()
        hololink.start()
        try:
            # Camera
            camera = hololink_module.sensors.udp_cam.UdpCam(hololink_channel)
            # monkey patch with our special I2c failure-inducing object.
            camera._i2c = I2cProxy(camera._i2c)

            hololink._sent = 0
            version = camera.get_version()
            logging.info(f"{version=}, {hololink._sent=}")

            # Now do it again but drop each packet and prove it still works.
            n = 0
            while hololink._drop == 0:
                n += 1
                logging.info(f"---- {n=}")
                hololink._drop = n
                hololink._sent = 0
                new_version = camera.get_version()
                logging.info(f"{version=}, {hololink._sent=}")
                assert version == new_version
                camera.tap_watchdog()
        finally:
            hololink.stop()
