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

# This is a convenience script used by pytest hw_loopback session fixture in tests/conftest.py to isolate an interface into a network namespace with the provided IPV4 address
# For use cases, see https://docs.nvidia/com/holoscan/sensor-bridge/latest/emulation.html#testing

if [ $# -lt 1 ] ; then
	echo "Usage: $0 <if_name> [<ipv4>]"
	echo "	isolate <if_name> under namespace ns_<if_name> with ip <ipv4>"
	echo "  <ipv4> - ipv4 address in CIDR notation. Defaults to current address if found"
	exit 1
fi

INTERFACE=$1
IPV4=
if [ $# -lt 2 ] ; then
	IPV4=`ip addr sh $INTERFACE | grep 'inet ' | awk '{print $2}'`
else
	IPV4=$2
fi

ip netns add ns_$INTERFACE

ip link set $INTERFACE netns ns_$INTERFACE
ip netns exec ns_$INTERFACE ip addr add dev $INTERFACE $IPV4 brd +
ip netns exec ns_$INTERFACE sh -c "ip link set dev $INTERFACE down && ip link set dev $INTERFACE up"
