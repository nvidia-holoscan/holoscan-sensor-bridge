#!/bin/bash

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
#
# See README.md for detailed information.

set -o errexit
umask 0

f=0
igpu=0
dgpu=0
usage=1

while [ $# -gt 0 ]
do
    case "$1" in
        "-f")
            f=1
            ;;
        "--igpu")
            igpu=1
            ;;
        "--dgpu")
            dgpu=1
            ;;
        *)
            usage=1
            break
            ;;
    esac
    usage=0
    shift
done

if [ $dgpu -eq $igpu ]; then
echo "Exactly one of --igpu or --dgpu must be selected." >&2
usage=1
fi

if [ $usage -ne 0 ]
then
echo "Usage: $0 [-f] --igpu|--dgpu" >&2
exit 1
fi

# Do a bit of environment checking:
# If we're running 'connmand' (e.g. IGX deployment)
# and veth isn't in the blacklist file, then
# warn users about this.  When not blacklisted,
# connmand will add docker's veth* interfaces
# to the default route--which breaks our internet,
# making building the containers fail.
if pidof -q connmand
then
if ! grep NetworkInterfaceBlacklist /etc/connman/main.conf | grep -q veth
then
(
echo WARNING: Docker builds fail when connmand assigns
echo docker virtual network addresses to the default
echo router table.  This failure is indicated by
echo "docker build" getting stuck downloading file\(s\)
echo from the internet.  You can verify this failure
echo mode, when stuck, by running "netstat -rn"
echo and observing a default route over a "veth*" interface.
echo To resolve this issue, add "veth" to the connmand
echo network interface blacklist:
echo \(In /etc/connman/main.conf\)
echo [General]
echo NetworkInterfaceBlacklist=veth
echo \(You may have other network interfaces listed here too\)
echo Run this script with "-f" to install anyway.
) >&2
# Use "-f" to keep going.
if [ "$f" = "0" ]
then
exit 1
fi
fi
fi

set -o xtrace

SCRIPT=`realpath "$0"`
HERE=`dirname "$SCRIPT"`
ROOT=`realpath "$HERE/.."`
VERSION=`cat $ROOT/VERSION`

if [ ! -d "$ROOT/nvidia" ]; then
mkdir $ROOT/nvidia
fi

CONTAINER_TYPE=dgpu
if [ $igpu -ne 0 ]
then
CONTAINER_TYPE=igpu
if [ ! -d "/usr/lib/aarch64-linux-gnu/nvidia" ]
then
echo "Error: Required path and libs are missing. \
      Upgrade the development kit with Jetpack 6 or newer."
exit 1
fi
cp /usr/lib/aarch64-linux-gnu/nvidia/libnvargus.so $ROOT/nvidia
cp /usr/lib/aarch64-linux-gnu/nvidia/libnvargus_socketclient.so $ROOT/nvidia
cp /usr/lib/aarch64-linux-gnu/nvidia/libnvargus_socketserver.so $ROOT/nvidia
cp /usr/lib/aarch64-linux-gnu/nvidia/libnvfusacap.so $ROOT/nvidia
cp /usr/lib/aarch64-linux-gnu/nvidia/libnvodm_imager.so $ROOT/nvidia
cp /usr/lib/aarch64-linux-gnu/nvidia/libnvscf.so $ROOT/nvidia
fi

chmod 755 $ROOT/nvidia

# Build the development container
docker build \
    --network=host \
    --build-arg CONTAINER_TYPE=$CONTAINER_TYPE \
    -t hololink-prototype:$VERSION \
    -f $HERE/Dockerfile \
    $ROOT

# Build a container that has python extensions set up.
docker build \
    --network=host \
    --build-arg CONTAINER_VERSION=hololink-prototype:$VERSION \
    -t hololink-demo:$VERSION \
    -f $HERE/Dockerfile.demo \
    $ROOT

rm -rf nvidia
