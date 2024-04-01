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


def test_udpcam_version(udpcam):
    logging.info("Initializing.")

    with udp_server.TestServer(udpcam) as server:
        channel_metadata = server.channel_metadata()
        hololink_channel = hololink_module.HololinkDataChannel(channel_metadata)
        hololink = hololink_channel.hololink()
        hololink.start()

        # Camera
        camera = hololink_module.sensors.udp_cam.UdpCam(hololink_channel)
        camera.reset()

        version = camera.get_version()
        logging.info("version=%s" % (version,))
        hololink.stop()
