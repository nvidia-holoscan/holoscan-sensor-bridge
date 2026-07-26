#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
GPU_CONFIG=
HOLOLINK_ROCE_USE_GPU_VRAM=ON
# Empty means "not set on the command line"; the default is resolved later
# based on the detected platform (disabled by default on AGX Orin).
HOLOLINK_USE_SCCACHE=
# HSDK follows Major.Minor.Patch versioning scheme
HSDK_VERSION="4.4.0"

# detected and populated in script and used in docker build command
PROTOTYPE_OPTIONS=""
CONTAINER_TYPE=
L4T_VERSION=

while [ $# -gt 0 ]
do
    case "$1" in
        "-f")
            f=1
            ;;
        "--igpu")
            igpu=1
            GPU_CONFIG="igpu"
            ;;
        "--dgpu")
            dgpu=1
            GPU_CONFIG="dgpu"
            ;;
        "--disable-roce-gpu-vram")
            HOLOLINK_ROCE_USE_GPU_VRAM=OFF
            ;;
        --use-sccache=*)
            case "${1#--use-sccache=}" in
                0)
                    HOLOLINK_USE_SCCACHE=OFF
                    ;;
                1)
                    HOLOLINK_USE_SCCACHE=ON
                    ;;
                *)
                    echo "Invalid value for --use-sccache; must be 0 or 1." >&2
                    usage=1
                    break
                    ;;
            esac
            ;;
        --hsdk=*)
            HSDK_VERSION="${1#--hsdk=}"
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
echo "Usage: $0 [-f] --igpu|--dgpu [--disable-roce-gpu-vram] [--use-sccache=0|1]" >&2
exit 1
fi

# get the product name from the system. L4T and DGX OS support Desktop Management Interface (DMI)
# if this file does not exist or PRODUCT_NAME is "Unknown", then we cannot guarantee the current configuration
# of the docker container is compatible
PRODUCT_NAME=
if [ -f /sys/class/dmi/id/product_name ]
then
    PRODUCT_NAME=$(cat /sys/class/dmi/id/product_name)
fi

if [ -z "$PRODUCT_NAME" ] || [ "$PRODUCT_NAME" = "Unknown" ]
then
    echo "Warning: Cannot determine product name from DMI. This system/configuration may not be supported."
fi

# Resolve the value for sccache when not explicitly set on the command line.
# Precedence: --use-sccache command line > HSB_USE_SCCACHE env variable > platform default.
# sccache is enabled by default everywhere except on the AGX Orin platform.
if [ -z "$HOLOLINK_USE_SCCACHE" ]
then
    if [ -n "$HSB_USE_SCCACHE" ]
    then
        case "$HSB_USE_SCCACHE" in
            0)
                HOLOLINK_USE_SCCACHE=OFF
                ;;
            1)
                HOLOLINK_USE_SCCACHE=ON
                ;;
            *)
                echo "Invalid value for HSB_USE_SCCACHE; must be 0 or 1." >&2
                exit 1
                ;;
        esac
    elif echo "$PRODUCT_NAME" | grep -q "AGX Orin"
    then
        HOLOLINK_USE_SCCACHE=OFF
    else
        HOLOLINK_USE_SCCACHE=ON
    fi
fi

# compiles a minimal binary as CUDA program verbosely to detect the native architecture used for the available GPU(s) and captures it in output.
# this is exported to the container as CMAKE_CUDA_ARCHITECTURES so cmake picks it up directly for all CUDA targets (including gpu_roce_transceiver)
# without ever needing nvcc -arch=native inside the container (which has no GPU access during build).
HOLOLINK_CUDA_ARCHS=$(echo -n "int main(void) {return 0; }" | /usr/local/cuda/bin/nvcc -arch=native --verbose -x cu -o /tmp/main - 2>&1 | grep -m 1 -oE "__CUDA_ARCH__=[^-]+" | cut -d= -f2 | tr -d ' ')
if [ -z "$HOLOLINK_CUDA_ARCHS" ]; then
    HOLOLINK_CUDA_ARCHS=800
fi
# The result above will be for example 890 for compute 8.9. Remove the trailing digit to get the CMake form (e.g. 89).
HOLOLINK_CUDA_ARCHS="${HOLOLINK_CUDA_ARCHS%?}"

# detect infiniband devices. HOLOLINK_BUILD_ROCE is set to 1 if any are found
IBV_DEVICES=""
if [ -d /sys/class/infiniband/ ]
then
    IBV_DEVICES=$(ls /sys/class/infiniband/ | LC_COLLATE=C sort | tr '\n' ' ')
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

# put logic for checking l4t configuration and libraries here
check_l4t() {
    if [ "$#" -ne 1 ]; then
        echo "Error: Expected exactly 1 argument <PRODUCT_NAME>, but received $#" >&2
        exit 1
    fi
    if echo $1 | grep -q "IGX Thor"
    then
        echo "Warning: \"$1\" configuration is not officially supported."
    fi
    L4T_CONFIG=$(cat /etc/nv_tegra_release)
    # Derive the L4T version from the BSP release file, which is authoritative.
    # The DMI bios_version reports the UEFI/QSPI firmware version; on IGX that is
    # flashed independently of the rootfs, so it can carry a leading "v"
    # (e.g. "v36.4.3", which breaks integer parsing) or lag the real BSP
    # (e.g. "36.1.0" on a fresh IGX OS 1.1.3 flash whose BSP is actually 36.5.x,
    # which pins a too-old apt repo lacking the CUDA 12.6 dev packages).
    # "# R36 (release), REVISION: 5.1, ..." -> 36.5.1
    L4T_RELEASE_LINE=$(head -n1 /etc/nv_tegra_release)
    L4T_MAJOR=$(echo "$L4T_RELEASE_LINE" | sed -nE 's/^# R([0-9]+).*/\1/p')
    L4T_REVISION=$(echo "$L4T_RELEASE_LINE" | sed -nE 's/.*REVISION:[[:space:]]*([0-9]+\.[0-9]+).*/\1/p')
    if [ -z "$L4T_MAJOR" ] || [ -z "$L4T_REVISION" ]; then
        echo "Error: Unable to parse L4T version from /etc/nv_tegra_release: $L4T_RELEASE_LINE" >&2
        exit 1
    fi
    L4T_VERSION="${L4T_MAJOR}.${L4T_REVISION}"
    L4T_VERSION_MAJOR=${L4T_VERSION%%.*}
    echo "L4T version: $L4T_VERSION"

    # only support Jetpack 6+ or IGX Base OS v1.0+, which are both L4T v36+
    if [ $L4T_VERSION_MAJOR -lt 36 ]
    then
        if echo $1 | grep -q "IGX"
        then
            if echo $1 | grep -q "Thor"
            then
                echo "Error: L4T version $L4T_VERSION_MAJOR is not supported for \"$1\". Upgrade to latest version of IGX BaseOS 2."
            else
                echo "Error: L4T version $L4T_VERSION_MAJOR is not supported for \"$1\". Upgrade to IGX BaseOS v1 or newer."
            fi
        else
            # AGX/Nano/NX configurations
            if echo $1 | grep -q "Thor"
            then
                echo "Error: L4T version $L4T_VERSION_MAJOR is not supported for \"$1\". Upgrade to Jetpack 7."
            else
                echo "Error: L4T version $L4T_VERSION_MAJOR is not supported for \"$1\". Upgrade to Jetpack 6 or newer."
            fi
        fi
        exit 1
    fi

    # assumes the user space libraries are in a path that does not contain spaces
    L4T_LIBRARIES_PATH="/$(echo $L4T_CONFIG | grep -oE 'TARGET_USERSPACE_LIB_DIR_PATH=[^ ]*' | awk -F'=' '{print $2}')"
    if [ ! -d $L4T_LIBRARIES_PATH ]
    then
        echo "Error: L4T libraries not found at $L4T_LIBRARIES_PATH"
        exit 1
    fi

    PROTOTYPE_OPTIONS="$PROTOTYPE_OPTIONS --build-context l4t-libs=$L4T_LIBRARIES_PATH "
    PROTOTYPE_OPTIONS="$PROTOTYPE_OPTIONS --build-context l4t-src=/usr/src "
}

set -o xtrace

SCRIPT=`realpath "$0"`
HERE=`dirname "$SCRIPT"`
ROOT=`realpath "$HERE/.."`
VERSION=`cat $ROOT/VERSION`

# default cuda 13 container type
CONTAINER_TYPE=cuda13
# check if l4t product
if [ -f /etc/nv_tegra_release ] ;
then
    check_l4t "$PRODUCT_NAME"
fi

# PVA SDK build context (only add if directory exists, otherwise use empty context)
# PVA SDK 2.9 (optional for build context)
if [ -d "/opt/nvidia/pva-sdk-2.9" ]; then
    # point directly at pva-sdk-2.9 within /opt/nvidia so that the build context does not pull in everything under /opt/nvidia
    PROTOTYPE_OPTIONS="$PROTOTYPE_OPTIONS --build-context pva-sdk=/opt/nvidia/pva-sdk-2.9 "
else
    # Create empty build context directory to avoid Docker build errors
    # Dockerfile will gracefully skip PVA SDK copy if not present
    mkdir -p /tmp/empty-pva-context
    PROTOTYPE_OPTIONS="$PROTOTYPE_OPTIONS --build-context pva-sdk=/tmp/empty-pva-context "
fi

if [ $igpu -ne 0 ]
then
    if [ -n "$L4T_VERSION_MAJOR" ]
    then
        if [ "$L4T_VERSION_MAJOR" = "36" ]
        then
	    # JP6.X/IGX 1.X X>0
	    CONTAINER_TYPE=cuda12_igpu
        else
            # JP7.2+/IGX 2.1+
            CONTAINER_TYPE=cuda13_igpu
        fi
    fi

    if echo $PRODUCT_NAME | grep -q "Thor"
    then
        if echo $PRODUCT_NAME | grep -qE "AGX|MINI"
        then
            CONTAINER_TYPE=cuda13_sipl
        else
            CONTAINER_TYPE=cuda13_l4t
        fi
    fi
elif [ $dgpu -ne 0 ]
then
    # check for definite invalid dgpu configurations
    if (echo $PRODUCT_NAME | grep -q "Spark") || (echo $PRODUCT_NAME | grep -q "AGX") || (echo $PRODUCT_NAME | grep -q "Nano")
    then
        echo "Error: product \"$PRODUCT_NAME\" does not have a supported dgpu configuration"
        exit 1
    fi

    if echo $PRODUCT_NAME | grep -q "Orin"
    then
        # IGX Orin dgpu
        CONTAINER_TYPE=cuda12_dgpu
    elif echo $PRODUCT_NAME | grep -q "Thor"
    then
        # IGX Thor dgpu
        CONTAINER_TYPE=cuda13_l4t
    fi
fi

# For Jetson Nano devices, which have very limited memory,
# limit the number of CPUs so we don't run out of RAM.
INSTALL_ENVIRONMENT=""
MEM_G=`awk '/MemTotal/ {print int($2 / 1024 / 1024)}' /proc/meminfo`
if [ $MEM_G -lt 10 ]
then
INSTALL_ENVIRONMENT="taskset -c 0-2"
fi

# temporarily disable tracing to output configuration variables
set +x
echo "PRODUCT_NAME: $PRODUCT_NAME"
echo "HSDK_VERSION: $HSDK_VERSION"
echo "CONTAINER_TYPE: $CONTAINER_TYPE"
echo "PROTOTYPE_OPTIONS: $PROTOTYPE_OPTIONS"
echo "INSTALL_ENVIRONMENT: $INSTALL_ENVIRONMENT"
echo "L4T_VERSION: $L4T_VERSION"
echo "HOLOLINK_CUDA_ARCHS: $HOLOLINK_CUDA_ARCHS"
echo "IBV_DEVICES: $IBV_DEVICES"
echo "HOLOLINK_ROCE_USE_GPU_VRAM: $HOLOLINK_ROCE_USE_GPU_VRAM"
echo "HOLOLINK_USE_SCCACHE: $HOLOLINK_USE_SCCACHE"
set -x

# Build the development container.  We specifically rely on buildkit skipping
# the dgpu or igpu stages that aren't included in the final image we're
# creating.
DOCKER_BUILDKIT=1 docker build \
    --network=host \
    --build-arg "CONTAINER_TYPE=$CONTAINER_TYPE" \
    --build-arg "HSDK_VERSION=$HSDK_VERSION" \
    --build-arg "L4T_VERSION=$L4T_VERSION" \
    --build-arg "IBV_DEVICES=$IBV_DEVICES" \
    -t hololink-prototype:$VERSION \
    -f $HERE/Dockerfile \
    $PROTOTYPE_OPTIONS \
    $ROOT

# Build a container that has python extensions set up.
DOCKER_BUILDKIT=1 docker build \
    --network=host \
    --build-arg CONTAINER_VERSION=hololink-prototype:$VERSION \
    --build-arg "INSTALL_ENVIRONMENT=$INSTALL_ENVIRONMENT" \
    --build-arg "CUDAARCHS=$HOLOLINK_CUDA_ARCHS" \
    --build-arg "HOLOLINK_ROCE_USE_GPU_VRAM=$HOLOLINK_ROCE_USE_GPU_VRAM" \
    --build-arg "HOLOLINK_USE_SCCACHE=$HOLOLINK_USE_SCCACHE" \
    -t hololink-demo:$VERSION \
    -f $HERE/Dockerfile.demo \
    $ROOT

