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
import utils

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
@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
def test_hololink_acks(hololink_address):
    logging.info("Initializing.")
    #
    metadata = hololink_module.Enumerator.find_channel(hololink_address)
    hololink = AckTimerHololink(
        peer_ip=metadata["peer_ip"],
        control_port=metadata["control_port"],
        serial_number=metadata["serial_number"],
        sequence_number_checking=(
            False if metadata["sequence_number_checking"] == 0 else True
        ),
    )
    with utils.PriorityScheduler():
        hololink.start()
        # Make sure we have PTP sync first.
        logging.debug("Waiting for PTP sync.")
        if not hololink.ptp_synchronize():
            raise ValueError("Failed to synchronize PTP.")
        #
        for i in range(20):
            hololink.get_hsb_ip_version()
            hololink.get_fpga_date()
        hololink.stop()

    max_dt = None
    settled_acks = hololink._acks[5:]
    assert len(settled_acks) > 10
    for n, (request_time, reply_time) in enumerate(settled_acks):
        dt_s = reply_time - request_time
        dt_ms = dt_s * 1000.0
        dt_us = dt_ms * 1000.0
        print(f"{n=} {request_time=} {dt_us=:0.3f}")
        if (max_dt is None) or (dt_us > max_dt):
            max_dt = dt_us
    assert max_dt < 500


class DropRequestHololink(hololink_module.Hololink):
    """
    We hook into the ECB transmitter and use that to selectively
    drop ECB requests; this exercises the retry mechanism in
    write_uint32 and read_uint32.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._drop_send = 0
        self._drop_receive = 0

    def send_control(self, request):
        if self._drop_send > 0:
            self._drop_send -= 1
            if self._drop_send == 0:
                logging.info(f"send_control dropped {request=}")
                return
        logging.info(f"send_control {request=}")
        super().send_control(request)

    def receive_control(self, timeout):
        while True:
            reply = super().receive_control(timeout)
            if self._drop_receive > 0:
                self._drop_receive -= 1
                if self._drop_receive == 0:
                    logging.info(f"receive_control dropped {reply=}")
                    continue
            logging.info(f"receive_control {reply=}")
            return reply


# This test doesn't actually require accelerated networking,
# but because it tests device functionality and not host
# functionality, we'll skip this for hosts with slower network
# ports.
@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
def test_hololink_ecb_retry(hololink_address):
    logging.info("Initializing.")
    #
    metadata = hololink_module.Enumerator.find_channel(hololink_address)
    hololink = DropRequestHololink(
        peer_ip=metadata["peer_ip"],
        control_port=metadata["control_port"],
        serial_number=metadata["serial_number"],
        sequence_number_checking=(
            False if metadata["sequence_number_checking"] == 0 else True
        ),
    )
    with utils.PriorityScheduler():
        hololink.start()

        # Write a known value to a known location.
        value = 0xEEEEAA55
        hololink.write_uint32(hololink_module.APB_RAM, value)
        v = hololink.read_uint32(hololink_module.APB_RAM)
        assert v == value

        # Show that we can retry a lost read request
        hololink._drop_send = 1
        v = hololink.read_uint32(hololink_module.APB_RAM)
        assert v == value

        # Show that we can retry a lost read reply
        hololink._drop_receive = 1
        v = hololink.read_uint32(hololink_module.APB_RAM)
        assert v == value

        # Show that we can retry a lost write request
        value = 0xEEAA55EE
        hololink._drop_send = 1
        hololink.write_uint32(hololink_module.APB_RAM, value)

        v = hololink.read_uint32(hololink_module.APB_RAM)
        assert v == value

        # Show that we can retry a lost write reply
        value = 0xAA55EEEE
        hololink._drop_receive = 1
        hololink.write_uint32(hololink_module.APB_RAM, value)

        v = hololink.read_uint32(hololink_module.APB_RAM)
        assert v == value

        hololink.stop()
