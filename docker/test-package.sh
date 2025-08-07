#!/bin/sh

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

set -o errexit
set -o xtrace

SCRIPT=`realpath "$0"`
HERE=`dirname "$SCRIPT"`
ROOT=`realpath $HERE/..`
HOLOLINK_VERSION=`cat $ROOT/VERSION`
ARCH=`uname -m`
if [ "$ARCH" = "x86_64" ]; then
    ARCH='amd64'
elif [ "$ARCH" = "aarch64" ]; then
    ARCH='arm64'
fi

NAME=hololink-test-package

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

CONTAINER_VERSION=nvidia/cuda:12.2.0-base-ubuntu22.04

# Build a container that has the packages installed.
DOCKER_BUILDKIT=1 docker build \
    --network=host \
    --build-arg CONTAINER_VERSION=${CONTAINER_VERSION} \
    --build-arg HOLOLINK_VERSION=${HOLOLINK_VERSION} \
    --build-arg ARCH=${ARCH} \
    --build-context hololink-package=$ROOT/package \
    -t $NAME:$HOLOLINK_VERSION \
    -f $HERE/Dockerfile.test-package \
    $ROOT

docker run \
    -it \
    --rm \
    --net host \
    --gpus all \
    --runtime=nvidia \
    --shm-size=1gb \
    --privileged \
    --name "$NAME" \
    -v $PWD:$PWD \
    -v $ROOT:$ROOT \
    -v $HOME:$HOME \
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
    $NAME:$HOLOLINK_VERSION \
    $*
