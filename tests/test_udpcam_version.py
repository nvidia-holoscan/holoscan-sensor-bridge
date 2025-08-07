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

import mock_camera
import mock_server

import hololink as hololink_module


def test_udpcam_version(mock_camera_ip):
    logging.info("Initializing.")

    with mock_server.TestServer(mock_camera_ip) as server:
        channel_metadata = server.channel_metadata()
        hololink_channel = hololink_module.DataChannel(channel_metadata)
        hololink = hololink_channel.hololink()
        hololink.start()

        # Camera
        camera = mock_camera.MockCamera(hololink_channel)
        camera.reset()

        version = camera.get_version()
        logging.info("version=%s" % (version,))
        hololink.stop()
