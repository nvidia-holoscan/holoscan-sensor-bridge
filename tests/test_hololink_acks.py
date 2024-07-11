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

import pytest

import hololink as hololink_module


class AckTimerHololink(hololink_module.Hololink):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._acks = []

    def executed(self, request_time, request, reply_time, reply):
        self._acks.append((request_time, reply_time))


# This test doesn't actually require accelerated networking,
# but because it tests device functionality and not host
# functionality, we'll skip this for hosts with slower network
# ports.
@pytest.mark.skip_unless_udp_server
@pytest.mark.accelerated_networking
def test_hololink_acks(hololink_address):
    logging.info("Initializing.")
    #
    metadata = hololink_module.Enumerator.find_channel(hololink_address)
    hololink = AckTimerHololink(
        peer_ip=metadata["peer_ip"],
        control_port=metadata["control_port"],
        serial_number=metadata["serial_number"],
    )
    hololink.start()
    for i in range(10):
        hololink.get_fpga_version()
        hololink.get_fpga_date()
    hololink.stop()

    max_dt = None
    for n, (request_time, reply_time) in enumerate(hololink._acks):
        dt_s = reply_time - request_time
        dt_ms = dt_s * 1000.0
        dt_us = dt_ms * 1000.0
        print(f"{n=} {request_time=} {dt_us=:0.3f}")
        if (max_dt is None) or (dt_us > max_dt):
            max_dt = dt_us
    assert max_dt < 500
