#!/bin/bash

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

# This is a convenience script used by tests/test_emulator.py to execute a command in the network namespace ns_<if_name>
# For use cases, see https://docs.nvidia/com/holoscan/sensor-bridge/latest/emulation.html#testing

if [ $# -lt 2 ] ; then
	echo "Usage: $0 <if_name> <command> [args...]"
	echo "  Execute a command in network namespace ns_<if_name>"
	exit 1
fi

INTERFACE=$1
shift

ip netns exec ns_$INTERFACE "$@"
