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
import threading

import pytest

import hololink as hololink_module


@pytest.mark.skip_unless_imx274
def test_hololink_enumeration_native(timeout_s=60):
    enumerator = hololink_module.Enumerator()
    devices_found = {}

    def call_back(metadata):
        logging.debug(f"{metadata=}")
        serial_number = metadata.get("serial_number")
        if serial_number is not None:
            if serial_number in devices_found:
                return False
            devices_found[serial_number] = metadata
        return True

    enumerator.enumerated(call_back, hololink_module.Timeout(timeout_s))

    for serial_number, metadata in devices_found.items():
        logging.info(f"Found {serial_number=}")
    assert len(devices_found) >= 1


@pytest.mark.skip_unless_imx274
def test_hololink_enumeration_supplemental(channel_ip, timeout_s=60):
    metadata = hololink_module.Metadata(
        {
            "test-parameter": "HELLO",
        }
    )
    uuid_strategy = hololink_module.BasicEnumerationStrategy(
        metadata,
        total_sensors=3,
        total_dataplanes=3,
        sifs_per_sensor=2,
    )
    uuid = "889b7ce3-65a5-4247-8b05-4ff1904c3359"
    hololink_module.Enumerator.set_uuid_strategy(uuid, uuid_strategy)
    channel_metadata = hololink_module.Enumerator.find_channel(channel_ip=channel_ip)
    logging.debug(f"{channel_metadata=}")
    assert channel_metadata["test-parameter"] == "HELLO"
    hololink_module.DataChannel.use_sensor(channel_metadata, 2)
    hololink_module.DataChannel.use_data_plane_configuration(channel_metadata, 2)


@pytest.mark.skip_unless_imx274
def test_hololink_enumeration_via_reactor(channel_ip, timeout_s=5):
    received_metadata = []
    condition = threading.Condition()

    def callback(metadata):
        logging.debug(f"Received metadata: {metadata}")
        with condition:
            received_metadata.append(metadata)
            condition.notify()

    # Register the callback for the channel IP using C++ Enumerator static methods
    callback_handle = hololink_module.Enumerator.register_ip(channel_ip, callback)

    try:
        # Wait for enumeration to happen
        while True:
            with condition:
                assert condition.wait(timeout_s), "Timed out waiting for messages."
                if len(received_metadata) > 5:
                    break
    finally:
        # Unregister the callback
        hololink_module.Enumerator.unregister_ip(callback_handle)

    # Verify we received metadata
    assert len(received_metadata) > 0, f"No metadata received for {channel_ip}"
    logging.info(
        f"Received {len(received_metadata)} metadata callbacks for {channel_ip}"
    )
