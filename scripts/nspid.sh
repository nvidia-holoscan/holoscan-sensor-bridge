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

# This script is used by the tests/test_emulator.py script to find the pid of the emulator process running in the network namespace so that it can be properly cleaned up.

if [ $# -lt 2 ] ; then
	echo "Usage: $0 <if_name> <command> [args...]"
	echo "  prints out the pid of the <command> with args as running in the network namespace ns_<if_name>. Note that typically would just pass the program name as flags can be problematic to the use of grep"
	exit 1
fi

INTERFACE=$1
shift

ip netns pids ns_$INTERFACE | xargs ps -o pid,command -p | grep "$@" | awk '{print $1}'
