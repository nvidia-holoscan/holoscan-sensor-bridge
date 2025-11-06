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

# This is a convenience script used by pytest hw_loopback session fixture in tests/conftest.py to remove the interface from the network namespace and add an IPV4 address to it
# For use cases, see https://docs.nvidia/com/holoscan/sensor-bridge/latest/emulation.html#testing

if [ $# -lt 1 ] ; then
	echo "Usage: $0 <if_name> [<ipv4>]"
	echo "	remove namespace ns_<if_name> ns_<if_name> and reset <if_name> with ip <ipv4>"
	echo "  <ipv4> - ipv4 address in CIDR notation. Defaults to current address if found"
	exit 1
fi

INTERFACE=$1
IPV4=
if [ $# -lt 2 ] ; then
	IPV4="`ip netns exec ns_$INTERFACE ip addr sh $INTERFACE | grep -o 'inet .*' | awk '{print $2}'`"
else
	IPV4=$2
fi

# ignore stdout if namespace file exist, but keep errors
ip netns del ns_$INTERFACE
MAX_ATTEMPTS=10

ip addr sh $INTERFACE 1>/dev/null 2>&1
while [ $? -ne 0 ] && [ $MAX_ATTEMPTS -gt 0 ] ; do
	sleep 1
	ip addr sh $INTERFACE 1>/dev/null 2>&1
	MAX_ATTEMPTS=$((MAX_ATTEMPTS - 1))
done

ip addr sh $INTERFACE 1>/dev/null 2>&1
if [ $? -eq 0 ] ; then
	ip addr add dev $INTERFACE $IPV4 brd + 1>/dev/null 2>&1
	ip link set dev $INTERFACE down && ip link set dev $INTERFACE up 1>/dev/null 2>&1
else
	echo "failed to find $INTERFACE"
	exit 1
fi
