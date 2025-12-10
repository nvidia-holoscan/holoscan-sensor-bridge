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
GPU_CONFIG=
# HSDK follows Major.Minor.Patch versioning scheme
HSDK_VERSION="3.9.0"

# detected and populated in script and used in docker build command
PROTOTYPE_OPTIONS=""
CONTAINER_TYPE=
HSDK_CONTAINER_CONFIG=

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
echo "Usage: $0 [-f] --igpu|--dgpu" >&2
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

# check to make sure user can run nvidia-smi without sudo.
# For example, on AGX Orin running Jetpack 6, the user configured with SDK Manager and the next user will have
# sufficient permissions/group access but any subsequently added users will not be in the proper groups to run
#basic utilities requiring access to the GPU.
if ! nvidia-smi > /dev/null 2>&1 
then
    echo "Error: User \"${USER}\" does not have permission to run nvidia-smi. Please add the user to the appropriate groups to run GPU utilities, e.g. 'video' and 'render' groups on l4t."
    exit 1
fi

# determine the CUDA driver compatibility version (Major)
# CTK may not be installed and nvcc only reports the runtime API version.
# nvidia-smi reports the CUDA display driver version, circa driver version 410.72, so should CUDA version 11+ 
# on arm64 or x86_64 platforms.
DRIVER_CUDA_VERSION_MAJOR=$(nvidia-smi | grep -oE "CUDA Version: [0-9]+" | awk '{print $3}' )

# handle HSDK version
HSDK_MAJOR_VERSION=${HSDK_VERSION%%.*}
HSDK_MINOR_PATCH=${HSDK_VERSION#*.}
HSDK_MINOR_VERSION=${HSDK_MINOR_PATCH%.*}
HSDK_PATCH_VERSION=${HSDK_VERSION##*.}

HSDK_CONTAINER_CONFIG="v$HSDK_VERSION"

# HSDK versions 3.6.1+ currently require differentiation of the cuda version
if [ $HSDK_MAJOR_VERSION -eq 3 ]
then
    if [ $HSDK_MINOR_VERSION -eq 6 ] && [ $HSDK_PATCH_VERSION -gt 0 ] || [ $HSDK_MINOR_VERSION -gt 6 ]
    then
        HSDK_CONTAINER_CONFIG="${HSDK_CONTAINER_CONFIG}-cuda${DRIVER_CUDA_VERSION_MAJOR}"
    fi
fi
# CUDA 13 and HSDK containers based on it do not distinguish jetson (igpu) vs sbsa (dgpu) GPU_CONFIGs anymore
if [ "$HSDK_VERSION" = "3.6.1" ] || [ $DRIVER_CUDA_VERSION_MAJOR -lt 13 ]
then
    HSDK_CONTAINER_CONFIG="${HSDK_CONTAINER_CONFIG}-${GPU_CONFIG}"
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
    # get L4T version from DMI. The dmi bios_version for l4t includes the full l4t release version
    L4T_VERSION=$(cat /sys/class/dmi/id/bios_version | cut -d- -f1)
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

    PROTOTYPE_OPTIONS="$PROTOTYPE_OPTIONS --build-arg L4T_VERSION=$L4T_VERSION "
    PROTOTYPE_OPTIONS="$PROTOTYPE_OPTIONS --build-context l4t-libs=$L4T_LIBRARIES_PATH "
    PROTOTYPE_OPTIONS="$PROTOTYPE_OPTIONS --build-context l4t-src=/usr/src "
}

set -o xtrace

SCRIPT=`realpath "$0"`
HERE=`dirname "$SCRIPT"`
ROOT=`realpath "$HERE/.."`
VERSION=`cat $ROOT/VERSION`

if [ $igpu -ne 0 ]
then
    if echo $PRODUCT_NAME | grep -q "AGX Thor"
    then
        # AGX Thor configuration
        CONTAINER_TYPE=igpu_sipl
        check_l4t "$PRODUCT_NAME"
    elif echo $PRODUCT_NAME | grep -q "Spark"
    then
        # DGX Spark configuration
        CONTAINER_TYPE=igpu_spark
        # no l4t check here -- not l4t
    else
        # other igpu configurations
        CONTAINER_TYPE=igpu
        if (echo $PRODUCT_NAME | grep -q "Orin") || (echo $PRODUCT_NAME | grep -q "Thor")
        then
            check_l4t "$PRODUCT_NAME"
        else
            # build an igpu container as requested, but some of the checks/pre-build configurations may not be present
            echo "WARNING: product \"$PRODUCT_NAME\" not recognized. defaulting to basic igpu container configuration"
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
    CONTAINER_TYPE=dgpu
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
echo "DRIVER_CUDA_VERSION_MAJOR: $DRIVER_CUDA_VERSION_MAJOR"
echo "HSDK_CONTAINER_CONFIG: $HSDK_CONTAINER_CONFIG"
echo "CONTAINER_TYPE: $CONTAINER_TYPE"
echo "PROTOTYPE_OPTIONS: $PROTOTYPE_OPTIONS"
echo "INSTALL_ENVIRONMENT: $INSTALL_ENVIRONMENT"
set -x

# Build the development container.  We specifically rely on buildkit skipping
# the dgpu or igpu stages that aren't included in the final image we're
# creating.
DOCKER_BUILDKIT=1 docker build \
    --network=host \
    --build-arg "CONTAINER_TYPE=$CONTAINER_TYPE" \
    --build-arg "HSDK_CONTAINER_CONFIG=$HSDK_CONTAINER_CONFIG" \
    -t hololink-prototype:$VERSION \
    -f $HERE/Dockerfile \
    $PROTOTYPE_OPTIONS \
    $ROOT

# Build a container that has python extensions set up.
DOCKER_BUILDKIT=1 docker build \
    --network=host \
    --build-arg CONTAINER_VERSION=hololink-prototype:$VERSION \
    --build-arg "INSTALL_ENVIRONMENT=$INSTALL_ENVIRONMENT" \
    -t hololink-demo:$VERSION \
    -f $HERE/Dockerfile.demo \
    $ROOT

