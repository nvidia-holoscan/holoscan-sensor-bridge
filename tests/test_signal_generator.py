# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import subprocess
import sys

import pytest


# The app picks the RoCE device on each board's network interface: the first devices the
# system lists may not reach the boards (on a DGX Spark each port shows up as two devices, and
# only one of them has the board's RoCE v2 GID).
@pytest.mark.skip_unless_signal_generator
@pytest.mark.accelerated_networking
def test_signal_generator(tx_hololink_address, hololink_address):
    script = os.path.join(
        os.path.dirname(__file__), "..", "examples", "signal_generator.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            script,
            "--tx-hololink",
            tx_hololink_address,
            "--rx-hololink",
            hololink_address,
            "--frame-limit",
            "1",
            "--no-gui",
        ],
        timeout=120,
    )
    assert result.returncode == 0
