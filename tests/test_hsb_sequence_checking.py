# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


@pytest.mark.skip_unless_hsb
def test_hsb_sequence_checking(hololink_address):
    logging.info("Initializing.")
    #
    metadata = hololink_module.Enumerator.find_channel(hololink_address)

    # Create a Hololink controller.  Note that this approach to creating
    # hololink objects isn't something applications should do.  Calling start()
    # on that will reset the sequence number in the HSB device.
    hololink_a = hololink_module.Hololink(
        peer_ip=metadata["peer_ip"],
        control_port=metadata["control_port"],
        serial_number=metadata["serial_number"],
        sequence_number_checking=True,
    )
    hololink_a.start()

    # Create another Hololink controller for the same HSB device.  Applications
    # should never do this.  Calling start() on that will reset the sequence
    # number in the HSB device.
    hololink_b = hololink_module.Hololink(
        peer_ip=metadata["peer_ip"],
        control_port=metadata["control_port"],
        serial_number=metadata["serial_number"],
        sequence_number_checking=True,
    )
    hololink_b.start()
    # Advance the sequence number by performing a control plane transaction.
    version_b = hololink_b.get_hsb_ip_version()
    logging.info(f"Got {version_b=:#x}")

    # hololink_a's cache of the sequence number is now out-of-date.  Performing
    # a transaction with it should result in an exception.
    try:
        bad_version_a = hololink_a.get_hsb_ip_version()
        logging.info(f"Got {bad_version_a=:#x}")
    except RuntimeError as e:
        logging.info(f"Caught {e=}({type(e)=}) as expected.")
        return

    assert False and "This should have caused an exception."
