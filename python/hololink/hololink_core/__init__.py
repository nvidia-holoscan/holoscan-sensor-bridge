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

import os

# Load _hololink_core with RTLD_GLOBAL so its symbols are available to other modules
# This prevents duplicate copies of hololink_core code across Python extensions
import sys

# Save the current dlopen flags
if hasattr(sys, "getdlopenflags"):
    old_flags = sys.getdlopenflags()
    # Set RTLD_GLOBAL (0x00100 on Linux) so symbols are available globally
    sys.setdlopenflags(os.RTLD_LAZY | os.RTLD_GLOBAL)

from ._hololink_core import (
    DEFAULT_MTU,
    PAGE_SIZE,
    UDP_PACKET_SIZE,
    ArpWrapper,
    Deserializer,
    Reactor,
    Serializer,
    gettid,
    local_ip_and_mac,
    local_ip_and_mac_from_socket,
    local_mac,
    round_up,
)

# Restore the original dlopen flags
if hasattr(sys, "getdlopenflags"):
    sys.setdlopenflags(old_flags)

__all__ = [
    "ArpWrapper",
    "Deserializer",
    "DEFAULT_MTU",
    "PAGE_SIZE",
    "Reactor",
    "Serializer",
    "UDP_PACKET_SIZE",
    "local_ip_and_mac",
    "local_ip_and_mac_from_socket",
    "local_mac",
    "round_up",
    "gettid",
]
