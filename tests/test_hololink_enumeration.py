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


@pytest.mark.skip_unless_udp_server
def test_hololink_enumeration_native(timeout_s=60):
    enumerator = hololink_module.HololinkEnumerator()
    devices_found = {}
    for packet, metadata in enumerator.enumeration_packets(timeout_s):
        logging.debug(f"{metadata=}")
        serial_number = metadata.get("serial_number")
        if serial_number is not None:
            devices_found[serial_number] = metadata
    for serial_number, metadata in devices_found.items():
        logging.info(f"Found {serial_number=}")
    assert len(devices_found) >= 1
