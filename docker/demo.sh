#!/bin/sh

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


# This script runs the command given on the command
# line within the hololink demo container.  For example,
#
#   sh docker/demo.sh pytest -s -o log_cli_level=15
#

set -o errexit
set -o xtrace

SCRIPT=`realpath "$0"`
HERE=`dirname "$SCRIPT"`
ROOT=`realpath $HERE/..`
VERSION=`cat $ROOT/VERSION`

NAME=demo

# See if we need to run our container differently.
while [ $# -ge 1 ]
do
case "$1" in
  --name=*)
    NAME="${1#--name=}"
    ;;
  *)
    break
    ;;
esac
shift
done

# enableRawReprocess needs to be set 1 if the frame capture is not \
# done by Argus API and only ISP being used.
# NOTE that we're currently testing with a value of 2 here.

# PulseAudio: mount the host socket + cookie so ALSA device "pulse" works
# inside the container (requires libasound2-plugins in hololink-demo).
DEMO_PULSE_RUNDIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
DEMO_PULSE_NATIVE="${DEMO_PULSE_RUNDIR}/pulse/native"
DEMO_XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${HOME}/.config}"
DEMO_PULSE_COOKIE_DIR="${DEMO_XDG_CONFIG_HOME}/pulse"
DEMO_PULSE_DOCKER=""
if [ -S "$DEMO_PULSE_NATIVE" ]; then
    DEMO_ASOUND_CONF="${HERE}/asound.conf.pulse"
    DEMO_PULSE_DOCKER="-v ${DEMO_PULSE_RUNDIR}/pulse:${DEMO_PULSE_RUNDIR}/pulse \
        -e HOME=${HOME} \
        -e XDG_RUNTIME_DIR=${DEMO_PULSE_RUNDIR} \
        -e XDG_CONFIG_HOME=${DEMO_XDG_CONFIG_HOME} \
        -e PULSE_SERVER=unix:${DEMO_PULSE_NATIVE}"
    if [ -d "$DEMO_PULSE_COOKIE_DIR" ]; then
        DEMO_PULSE_DOCKER="${DEMO_PULSE_DOCKER} -v ${DEMO_PULSE_COOKIE_DIR}:${DEMO_PULSE_COOKIE_DIR}"
    fi
    if [ -f "$DEMO_ASOUND_CONF" ]; then
        DEMO_PULSE_DOCKER="${DEMO_PULSE_DOCKER} -v ${DEMO_ASOUND_CONF}:/etc/asound.conf:ro"
    fi
fi

docker run \
    -it \
    --rm \
    --net host \
    --gpus all \
    --runtime=nvidia \
    --shm-size=1gb \
    --ulimit stack=33554432 \
    --privileged \
    --name "$NAME" \
    -v $PWD:$PWD \
    -v $ROOT:$ROOT \
    -v $HOME:$HOME \
    $(if [ -d /opt/nvidia/pva-sdk-2.9 ]; then echo "-v /opt/nvidia/pva-sdk-2.9:/opt/nvidia/pva-sdk-2.9"; fi) \
    $(if [ -d /etc/pva ]; then echo "-v /etc/pva:/etc/pva"; fi) \
    $(if [ -d /usr/lib/aarch64-linux-gnu/nvidia ]; then echo "-v /usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu/nvidia"; fi) \
    $(if [ -d /usr/lib/aarch64-linux-gnu/tegra ]; then echo "-v /usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra"; fi) \
    -v /sys/bus/pci/devices:/sys/bus/pci/devices \
    -v /sys/kernel/mm/hugepages:/sys/kernel/mm/hugepages \
    -v /dev:/dev \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /tmp/argus_socket:/tmp/argus_socket \
    -v /sys/devices:/sys/devices \
    -v /var/nvidia/nvcam/settings:/var/nvidia/nvcam/settings \
    -w $PWD \
    -e NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility,display \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e DISPLAY=$DISPLAY \
    -e enableRawReprocess=2 \
    -e rawReprocessModulePartName="A6V26" \
    $DEMO_PULSE_DOCKER \
    hololink-demo:$VERSION \
    "$@"
