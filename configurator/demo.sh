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

# This script launches a Hololink demo container along with a virtual
# machine running the HSB SPI Daemon for the sake of device configuration.
#
# Options:
#   --spi-vm [path]
#       Specifies root folder containing configurator VM image files (see options below)
#   --kernel [file]
#       Specifies the kernel image to use. This path is relative to `--spi-vm`.
#       Default: Image
#   --rootfs [file]
#       Specifies the root filesystem to use. This path is relative to `--spi-vm`.
#       Default: core-image-minimal-hsb-ad9986.rootfs.ext4
#   --dtb [file]
#       Specifies the device tree to use. This path is relative to `--spi-vm`.
#       Default: hsb-ad9986.dtb
#
# The PID of the VM that is launched will be written to spi-vm.pid, and
# its console output will be written to spi-vm.log.
#
# See README.md for detailed information.

set -o errexit

SCRIPT=`realpath "$0"`
SCRIPT_DIR=`dirname "$SCRIPT"`
ROOT=`realpath $SCRIPT_DIR/..`
VERSION=`cat $ROOT/VERSION`

SPI_VM_PATH=${SCRIPT_DIR}/build/tmp/deploy/images/hsb-ad9986
KERNEL=Image
ROOTFS=core-image-minimal-hsb-ad9986.rootfs.ext4
DTB=hsb-ad9986.dtb

while [ $# -ge 1 ]; do
case "$1" in
  --spi-vm=*)
    SPI_VM_PATH="${1#--spi-vm=}"
    ;;
  --kernel=*)
    KERNEL="${1#--kernel=}"
    ;;
  --rootfs=*)
    ROOTFS="${1#--rootfs=}"
    ;;
  --dtb=*)
    DTB="${1#--dtb=}"
    ;;
  *)
    break
    ;;
esac
shift
done

# Launch the configurator VM
qemu-system-aarch64 \
    -machine virt \
    -cpu cortex-a57 \
    -m 256 \
    -serial file:spi-vm.log \
    -display none \
    -kernel ${SPI_VM_PATH}/${KERNEL} \
    -dtb ${SPI_VM_PATH}/${DTB} \
    -drive id=disk0,file=${SPI_VM_PATH}/${ROOTFS},if=none,format=raw \
    -device virtio-blk-pci,drive=disk0 \
    -netdev user,id=net0,hostfwd=tcp::2222-:22,hostfwd=tcp::30431-:30431 \
    -device virtio-net-pci,netdev=net0,mac=52:54:00:12:35:02 \
    -append 'root=/dev/vda rw mem=256M ip=dhcp swiotlb=0' \
    -daemonize -pidfile spi-vm.pid
SPI_VM_PID=$(cat spi-vm.pid)
echo "Launched VM with PID ${SPI_VM_PID}"

# Don't exit if docker fails in order to clean up VM after.
set +o errexit

# Launch the Hololink container
docker run \
    -it \
    --rm \
    --net host \
    --gpus all \
    --runtime=nvidia \
    --shm-size=1gb \
    --privileged \
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
    hololink-demo:$VERSION \
    $*

# Kill the configurator VM
if [ -n "${SPI_VM_PID}" ]; then
    kill ${SPI_VM_PID}
    wait ${SPI_VM_PID} 2>/dev/null
fi
