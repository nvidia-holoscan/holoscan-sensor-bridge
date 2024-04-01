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

import udp_server

import hololink as hololink_module


class MockHololink(hololink_module.Hololink):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._drop = 0
        self._sent = 0

    def drop(self, n):
        self._drop = n

    def _send_control(self, request):
        logging.trace("request=%s" % (request,))
        if self._drop > 0:
            self._drop -= 1
            if self._drop == 0:
                logging.debug("dropping request=%s" % (request,))
                return
        super()._send_control(request)
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
        channel_metadata["hololink_class"] = MockHololink
        hololink_channel = hololink_module.HololinkDataChannel(channel_metadata)
        hololink = hololink_channel.hololink()
        hololink.start()
        try:
            # Camera
            camera = hololink_module.sensors.udp_cam.UdpCam(hololink_channel)
            # monkey patch with our special I2c failure-inducing object.
            camera._i2c = I2cProxy(camera._i2c)

            hololink._sent = 0
            version = camera.get_version()
            logging.info("version=%s, sent=%s" % (version, hololink._sent))

            # Now do it again but drop each packet and prove it still works.
            n = 0
            while hololink._drop == 0:
                n += 1
                logging.info("---- n=%s" % (n,))
                hololink._drop = n
                hololink._sent = 0
                new_version = camera.get_version()
                logging.info("version=%s, sent=%s" % (version, hololink._sent))
                assert version == new_version
                camera.tap_watchdog()
        finally:
            hololink.stop()
