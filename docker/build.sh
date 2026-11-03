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
# --igpu/--dgpu are accepted for backward compatibility only (see below).
igpu=0
dgpu=0
sipl=0
usage=0
# Resolved --no-cache flag passed to docker build. Enabled by the --no-cache
# option or the NO_CACHE=1 environment variable (the option wins).
NO_CACHE_ARG=
# Prune the BuildKit cache records this build creates when disk is tight: between the two
# image builds (the hololink-prototype layer snapshots and the apt/pip/sccache cache mounts)
# and again after hololink-demo (its builder-stage and layer snapshots). Only records that
# were not in the cache when this script started are removed (see BUILD_CACHE_BASELINE), so
# other projects' caches on a shared host or CI runner are left alone. --prune-cache /
# --no-prune-cache win, then PRUNE_CACHE=1|0 in the environment; otherwise the default
# is derived from the free space on the Docker data root (below PRUNE_CACHE_FREE_GIB
# -> prune). Resolved into PRUNE_CACHE_OPT.
PRUNE_CACHE_OPT=
PRUNE_CACHE_FREE_GIB=30
PRUNE_CACHE_REASON=
HOLOLINK_ROCE_USE_GPU_VRAM=ON
# OFF leaves holoscan-inference and TensorRT (~3 GB) out of hololink-demo; see --no-inference.
# hololink-prototype never carries them.
HOLOLINK_INSTALL_INFERENCE=ON
# Empty means "not set on the command line"; the default is resolved later
# based on the detected platform (disabled by default on AGX Orin).
HOLOLINK_USE_SCCACHE=
# HSDK follows Major.Minor.Patch versioning scheme
HSDK_VERSION="5.0.0"
# CMAKE_CUDA_ARCHITECTURES for the wheel and tools builds (semicolon-separated, e.g. "87;110").
# --cuda-archs= sets it; otherwise it is detected from the host GPUs below. Empty is allowed:
# Dockerfile.demo then resolves it against the container's nvcc and falls back to its default.
HOLOLINK_CUDA_ARCHS=

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
            # Unused: all container types share the same CUDA 13 base image and the
            # container type is derived from the host. Tolerated for older callers.
            igpu=1
            ;;
        "--dgpu")
            # Unused, see --igpu.
            dgpu=1
            ;;
        "--sipl")
            # Force the SIPL/FUSA (CoE offload) container type instead of relying
            # on the DMI product name detection below.
            sipl=1
            ;;
        "--disable-roce-gpu-vram")
            HOLOLINK_ROCE_USE_GPU_VRAM=OFF
            ;;
        "--no-inference")
            # Leave holoscan-inference and its TensorRT dependency (libnvinfer10 alone is
            # ~2.9 GB) out of hololink-demo. Only the tao_peoplenet / body_pose_estimation
            # examples need them.
            HOLOLINK_INSTALL_INFERENCE=OFF
            ;;
        "--no-cache")
            NO_CACHE_ARG="--no-cache"
            ;;
        "--prune-cache")
            PRUNE_CACHE_OPT=1
            PRUNE_CACHE_REASON="--prune-cache"
            ;;
        "--no-prune-cache")
            PRUNE_CACHE_OPT=0
            PRUNE_CACHE_REASON="--no-prune-cache"
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
        --cuda-archs=*)
            HOLOLINK_CUDA_ARCHS="${1#--cuda-archs=}"
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage=1
            break
            ;;
    esac
    shift
done

if [ $usage -ne 0 ]
then
echo "Usage: $0 [-f] [--sipl] [--disable-roce-gpu-vram] [--no-inference] [--use-sccache=0|1] [--no-cache] [--prune-cache|--no-prune-cache] [--hsdk=VERSION] [--cuda-archs=LIST]" >&2
echo "       --cuda-archs=LIST sets CMAKE_CUDA_ARCHITECTURES for the builds (e.g. 87 or 87;110) instead of" >&2
echo "       detecting it from the host GPUs (nvidia-smi, then nvcc -arch=native); Dockerfile.demo validates" >&2
echo "       the list against its nvcc and falls back to its own default when nothing is known." >&2
echo "       --no-inference leaves holoscan-inference and TensorRT (~3 GB) out of hololink-demo;" >&2
echo "       the tao_peoplenet / body_pose_estimation examples then cannot run." >&2
echo "       --prune-cache drops the BuildKit cache records this build created, between the two image" >&2
echo "       builds (the hololink-prototype layer snapshots and the apt/pip/sccache cache mounts) and again" >&2
echo "       after hololink-demo; records that existed before the build (other projects, other users) are" >&2
echo "       kept. Lowers the peak and resting disk usage of a build at the cost of a full rebuild next time." >&2
echo "       Default: on when the Docker data root has less than ${PRUNE_CACHE_FREE_GIB} GiB free (AGX Orin), else off;" >&2
echo "       --no-prune-cache (or PRUNE_CACHE=0) keeps the cache regardless." >&2
echo "       --igpu and --dgpu are accepted but ignored; the container type is derived from the host." >&2
exit 1
fi

if [ $igpu -ne 0 ] || [ $dgpu -ne 0 ]
then
    echo "Warning: --igpu/--dgpu are no longer used and will be ignored; the container type is derived from the host."
fi

# Resolve --no-cache: the command-line option wins; otherwise honor the NO_CACHE
# environment variable. This lets callers (e.g. CI) force a cacheless build by
# exporting NO_CACHE=1
if [ -z "$NO_CACHE_ARG" ] && [ "$NO_CACHE" = "1" ]
then
    NO_CACHE_ARG="--no-cache"
fi

# Resolve the cache prunes: option > PRUNE_CACHE=1|0 in the environment > free space on the
# Docker data root. With Docker's containerd image store (the JetPack 7 default) every built
# layer is on disk three times while its image is exported (BuildKit snapshot, compressed blob,
# unpacked image) and 1.6 times at rest (blob + unpacked image). Next to a stock JetPack 7 flash
# on AGX Orin (~24-28 GiB free) a build only has a chance if the cache is dropped as soon as
# each image is in the store; on hosts with room the cache is kept so the next build is fast.
DOCKER_ROOT=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || true)
[ -n "$DOCKER_ROOT" ] && [ -d "$DOCKER_ROOT" ] || DOCKER_ROOT=/var/lib/docker
[ -d "$DOCKER_ROOT" ] || DOCKER_ROOT=/
DOCKER_ROOT_FREE_GIB=$(df --output=avail -k "$DOCKER_ROOT" | tail -n1 | awk '{print int($1 / 1024 / 1024)}')
# Storage driver, plus the snapshotter when Docker runs on the containerd image store
# (e.g. "overlayfs (io.containerd.snapshotter.v1)"); the classic store reports "overlay2".
DOCKER_IMAGE_STORE=$(docker info --format '{{.Driver}}{{range .DriverStatus}}{{if eq (index . 0) "driver-type"}} ({{index . 1}}){{end}}{{end}}' 2>/dev/null || echo unknown)
if [ -z "$PRUNE_CACHE_OPT" ]
then
    case "$PRUNE_CACHE" in
        1)
            PRUNE_CACHE_OPT=1
            PRUNE_CACHE_REASON="PRUNE_CACHE=1 in the environment"
            ;;
        0)
            PRUNE_CACHE_OPT=0
            PRUNE_CACHE_REASON="PRUNE_CACHE=0 in the environment"
            ;;
        *)
            if [ "$DOCKER_ROOT_FREE_GIB" -lt "$PRUNE_CACHE_FREE_GIB" ]
            then
                PRUNE_CACHE_OPT=1
            else
                PRUNE_CACHE_OPT=0
            fi
            PRUNE_CACHE_REASON="${DOCKER_ROOT_FREE_GIB} GiB free on ${DOCKER_ROOT}, threshold ${PRUNE_CACHE_FREE_GIB} GiB"
            ;;
    esac
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

# Detect the compute capability of every GPU in this host for CMAKE_CUDA_ARCHITECTURES, without
# assuming a CUDA compiler on the host: the image build has no GPU access, so the detection has
# to happen here, but the compiler lives in the container. nvidia-smi comes with the driver and
# reports compute_cap per GPU (e.g. "8.7", "11.0", "12.1" -> 87;110;121); a host nvcc, if there
# is one, is the fallback via -arch=native, which prints one __CUDA_ARCH__=<arch>0 per GPU.
# The list (possibly empty) goes to Dockerfile.demo as CUDAARCHS, where it is checked against
# the container's nvcc and replaced by that file's default when nothing usable is known.
if [ -z "$HOLOLINK_CUDA_ARCHS" ] && command -v nvidia-smi >/dev/null 2>&1
then
    HOLOLINK_CUDA_ARCHS=$( (nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true) \
        | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd ';' - || true)
fi
if [ -z "$HOLOLINK_CUDA_ARCHS" ] && [ -x /usr/local/cuda/bin/nvcc ]
then
    NVCC_NATIVE_OUTPUT=$(echo -n "int main(void) {return 0; }" | /usr/local/cuda/bin/nvcc -arch=native --verbose -x cu -o /tmp/main - 2>&1 || true)
    if ! echo "$NVCC_NATIVE_OUTPUT" | grep -qi "Cannot find valid GPU"
    then
        HOLOLINK_CUDA_ARCHS=$(echo "$NVCC_NATIVE_OUTPUT" | grep -oE "__CUDA_ARCH__=[0-9]+" | cut -d= -f2 | sed 's/.$//' | sort -un | paste -sd ';' - || true)
    fi
fi
if [ -z "$HOLOLINK_CUDA_ARCHS" ]
then
    echo "Warning: could not detect the GPU architecture of this host (no nvidia-smi compute_cap, no usable nvcc);"
    echo "         Dockerfile.demo will build for its default CUDA architectures. Use --cuda-archs=LIST to choose."
fi

# The container is built on the CUDA 13 runtime image, so the host must provide CUDA 13 or newer.
# Read the host toolkit version from /usr/local/cuda (version.json, then nvcc).
HOST_CUDA_VERSION=
if [ -f /usr/local/cuda/version.json ]
then
    HOST_CUDA_VERSION=$(grep -A2 '"cuda"' /usr/local/cuda/version.json | grep -oE '"version"[[:space:]]*:[[:space:]]*"[0-9.]+"' | grep -oE '[0-9]+(\.[0-9]+)+' | head -n1)
fi
if [ -z "$HOST_CUDA_VERSION" ] && [ -x /usr/local/cuda/bin/nvcc ]
then
    HOST_CUDA_VERSION=$(/usr/local/cuda/bin/nvcc --version | grep -oE "release [0-9]+\.[0-9]+" | awk '{print $2}')
fi
if [ -z "$HOST_CUDA_VERSION" ]
then
    echo "Warning: Cannot determine the host CUDA version from /usr/local/cuda; the container requires CUDA 13 or newer."
elif [ "${HOST_CUDA_VERSION%%.*}" -lt 13 ]
then
    echo "Error: host CUDA version $HOST_CUDA_VERSION is not supported. CUDA 13 or newer is required." >&2
    exit 1
fi

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
    echo "L4T version: $L4T_VERSION"

    # HSDK5 requires CUDA 13, which needs L4T R38.4.0 or newer (JetPack 7.1+ on Thor,
    # JetPack 7.2+ / R39.2 on AGX Orin). This is a pure version floor: the product name is
    # deliberately not consulted so that any platform running a new enough BSP can build.
    L4T_MIN_VERSION="38.4.0"
    if [ "$(printf '%s\n%s\n' "$L4T_VERSION" "$L4T_MIN_VERSION" | sort -V | head -n1)" != "$L4T_MIN_VERSION" ]
    then
        echo "Error: L4T version $L4T_VERSION is not supported for \"$1\". L4T R${L4T_MIN_VERSION} or newer is required (JetPack 7.1+ on Thor, JetPack 7.2+ on AGX Orin)." >&2
        exit 1
    fi

    # assumes the user space libraries are in a path that does not contain spaces
    L4T_LIBRARIES_PATH="/$(echo $L4T_CONFIG | grep -oE 'TARGET_USERSPACE_LIB_DIR_PATH=[^ ]*' | awk -F'=' '{print $2}')"
    if [ ! -d $L4T_LIBRARIES_PATH ]
    then
        echo "Error: L4T libraries not found at $L4T_LIBRARIES_PATH"
        exit 1
    fi

    # Expose the BSP release file to the image through a dedicated build context
    # (a named context must be a directory) so python/setup.py and the tests can
    # parse the L4T version inside the container the same way this script does.
    L4T_RELEASE_CONTEXT=/tmp/hololink-l4t-release
    mkdir -p $L4T_RELEASE_CONTEXT
    cp /etc/nv_tegra_release $L4T_RELEASE_CONTEXT/nv_tegra_release

    PROTOTYPE_OPTIONS="$PROTOTYPE_OPTIONS --build-context l4t-libs=$L4T_LIBRARIES_PATH "
    PROTOTYPE_OPTIONS="$PROTOTYPE_OPTIONS --build-context l4t-src=/usr/src "
    PROTOTYPE_OPTIONS="$PROTOTYPE_OPTIONS --build-context l4t-release=$L4T_RELEASE_CONTEXT "
}

# Jetson SIPL API headers on the host.
#
# The cuda13_sipl image copies /usr/src/jetson_sipl_api from the host when it exists and
# otherwise downloads Jetson_SIPL_API_R${L4T_VERSION}_aarch64.tbz2 during the build. Not every
# L4T release has that package: IGX SW 2.0 on IGX Thor Mini is L4T R38.5.x and the matching
# JetPack 7.1 package is published as R38.4.0. These helpers try the host's own L4T release
# first and otherwise, on an R38 host, fall back to the known JetPack 7.1 package, and install
# it on the host exactly like the manual step in docs/user_guide/setup.mdx (tar xjf ... -C /),
# which needs sudo when not run as root.
# The package extracts relative to / (usr/src/jetson_sipl_api, usr/share/doc/jetson_sipl_api).
SIPL_API_ROOT=/
SIPL_API_INSTALL_DIR=${SIPL_API_ROOT%/}/usr/src/jetson_sipl_api
# The JetPack 7.1 Jetson SIPL API release, used when the host's own release has no package.
# Only for hosts on the same L4T major: the SIPL API changes between majors
# (cmake/sipl_api.cmake builds against API v1 on L4T 38 and v2 on L4T 39), so these headers
# are of no use to an R39 host.
SIPL_API_FALLBACK_VERSION=38.4.0

# <L4T_VERSION>: download URL of that release's Jetson SIPL API package.
sipl_api_url() {
    echo "https://developer.nvidia.com/downloads/embedded/L4T/r${1%%.*}_Release_v${1#*.}/release/Jetson_SIPL_API_R$1_aarch64.tbz2"
}

# <url>: succeeds when the package exists. developer.nvidia.com answers a HEAD request for a
# missing file with a redirect too, so this follows the redirect and fetches one byte.
sipl_api_available() {
    curl -fsSL --max-time 30 -r 0-0 -o /dev/null "$1" 2>/dev/null
}

# <L4T_VERSION>: the versions find_sipl_api_version tries, in order: the host's own release,
# then $SIPL_API_FALLBACK_VERSION when the host is on its L4T major.
sipl_api_candidates() {
    echo "$1"
    if [ "${1%%.*}" = "${SIPL_API_FALLBACK_VERSION%%.*}" ] && [ "$1" != "$SIPL_API_FALLBACK_VERSION" ]
    then
        echo "$SIPL_API_FALLBACK_VERSION"
    fi
}

# <L4T_VERSION>: prints the Jetson SIPL API release to install: the first of
# sipl_api_candidates for which NVIDIA publishes a package. Fails when none is published.
find_sipl_api_version() {
    local version
    for version in $(sipl_api_candidates "$1")
    do
        if sipl_api_available "$(sipl_api_url "$version")"
        then
            echo "$version"
            return 0
        fi
    done
    return 1
}

# Succeeds when $SIPL_API_INSTALL_DIR holds what the image build takes from it: the SIPL
# sources the Dockerfile builds with cmake, and the NvSci headers the FUSA operators include.
sipl_api_installed() {
    [ -f "$SIPL_API_INSTALL_DIR/sipl/CMakeLists.txt" ] \
        && [ -f "$SIPL_API_INSTALL_DIR/sipl/include/nvsci/nvscibuf.h" ]
}

# <L4T_VERSION>: makes sure a complete $SIPL_API_INSTALL_DIR exists on the host, downloading
# and installing the package for <L4T_VERSION>, or the JetPack 7.1 fallback on an R38 host,
# when the directory does not exist. Never leaves a partial installation behind.
install_sipl_api_headers() {
    local version url tmpdir tarball as_root candidates
    if [ -d "$SIPL_API_INSTALL_DIR" ]
    then
        if sipl_api_installed
        then
            echo "Jetson SIPL API: using the host's $SIPL_API_INSTALL_DIR"
            return 0
        fi
        # Not ours to delete: this function removes what it installed whenever that is
        # incomplete (below), so this directory got there some other way.
        echo "Error: $SIPL_API_INSTALL_DIR exists but is incomplete (sipl/CMakeLists.txt or sipl/include/nvsci/nvscibuf.h is missing); the image build would copy it as is." >&2
        echo "       Remove it (sudo rm -rf $SIPL_API_INSTALL_DIR) and run this script again to reinstall it." >&2
        return 1
    fi
    candidates=$(sipl_api_candidates "$1" | sed 's/^/R/' | paste -sd, | sed 's/,/, then /')
    echo "Jetson SIPL API: $SIPL_API_INSTALL_DIR is not on this host; looking for the L4T $candidates package ..."
    if ! version=$( { set +x; } 2>/dev/null; find_sipl_api_version "$1" )
    then
        echo "Error: no Jetson SIPL API package is published for L4T $candidates." >&2
        echo "       Install the headers by hand into $SIPL_API_INSTALL_DIR (see docs/user_guide/setup.mdx)." >&2
        return 1
    fi
    url=$(sipl_api_url "$version")
    tmpdir=$(mktemp -d)
    tarball="$tmpdir/Jetson_SIPL_API_R${version}_aarch64.tbz2"
    echo "Jetson SIPL API: installing R$version for L4T R$1 from $url"
    if ! curl -fsSL --max-time 900 -o "$tarball" "$url"
    then
        rm -rf "$tmpdir"
        echo "Error: download of $url failed." >&2
        return 1
    fi
    as_root=
    if [ ! -w "$SIPL_API_ROOT" ] || [ ! -w "$SIPL_API_ROOT/usr/src" ]
    then
        as_root=sudo
        echo "Jetson SIPL API: extracting into $SIPL_API_ROOT as root (sudo) ..."
    fi
    # $SIPL_API_INSTALL_DIR did not exist above, so whatever is there after a failed or
    # incomplete extraction is ours to remove: left in place, the next run would take it for
    # a complete installation and the image build would copy it as is.
    if ! $as_root tar xjf "$tarball" -C "$SIPL_API_ROOT"
    then
        $as_root rm -rf "$SIPL_API_INSTALL_DIR"
        echo "Error: could not extract $tarball into $SIPL_API_ROOT (the package installs $SIPL_API_INSTALL_DIR); try: sudo tar xjf $tarball -C $SIPL_API_ROOT" >&2
        return 1
    fi
    rm -rf "$tmpdir"
    if ! sipl_api_installed
    then
        $as_root rm -rf "$SIPL_API_INSTALL_DIR"
        echo "Error: Jetson SIPL API R$version did not install sipl/CMakeLists.txt and sipl/include/nvsci/nvscibuf.h under $SIPL_API_INSTALL_DIR; removed what it did install." >&2
        return 1
    fi
    echo "Jetson SIPL API: R$version installed in $SIPL_API_INSTALL_DIR"
}

set -o xtrace

SCRIPT=`realpath "$0"`
HERE=`dirname "$SCRIPT"`
ROOT=`realpath "$HERE/.."`
VERSION=`cat $ROOT/VERSION`

# default cuda 13 container type (non-L4T hosts, e.g. DGX Spark)
CONTAINER_TYPE=cuda13
# On L4T hosts, check the L4T version and collect the host library paths.
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

# The container type is derived from the host: every type shares the CUDA 13 base image,
# L4T hosts add the Argus/NvSci host libraries, and AGX Thor / Thor Mini add SIPL/FUSA.
if [ -n "$L4T_VERSION" ]
then
    CONTAINER_TYPE=cuda13_l4t
    if echo $PRODUCT_NAME | grep -q "Thor" && echo $PRODUCT_NAME | grep -iqE "AGX|Mini"
    then
        # AGX Thor / Thor Mini: SIPL/FUSA (CoE offload) capture
        CONTAINER_TYPE=cuda13_sipl
    fi
fi

if [ $sipl -ne 0 ]
then
    if [ -z "$L4T_VERSION" ]
    then
        echo "Error: --sipl requires an L4T host (/etc/nv_tegra_release not found)." >&2
        exit 1
    fi
    CONTAINER_TYPE=cuda13_sipl
fi

# IGX Thor Mini on IGX SW 2.0 reports L4T R38.5.x, for which NVIDIA publishes no Jetson SIPL
# API package, so the image's own download (keyed on the exact L4T version) cannot work.
# Put the host's own package, or the JetPack 7.1 R38.4.0 one, on the host instead (the DMI
# product name is "NVIDIA IGX Thor Developer Kit Mini", so match loosely on IGX, Thor and Mini);
# the image then takes the host copy and the Dockerfile does not change.
if [ "$CONTAINER_TYPE" = "cuda13_sipl" ] && echo "$PRODUCT_NAME" | grep -iqE "IGX.*Thor.*Mini"
then
    if ! install_sipl_api_headers "$L4T_VERSION"
    then
        echo "Run this script with \"-f\" to build anyway (the image build will then try the download for L4T R$L4T_VERSION itself, and fail if NVIDIA publishes none)." >&2
        # Use "-f" to keep going.
        if [ "$f" = "0" ]
        then
            exit 1
        fi
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
echo "HOST_CUDA_VERSION: $HOST_CUDA_VERSION"
echo "HSDK_VERSION: $HSDK_VERSION"
echo "CONTAINER_TYPE: $CONTAINER_TYPE"
echo "PROTOTYPE_OPTIONS: $PROTOTYPE_OPTIONS"
echo "INSTALL_ENVIRONMENT: $INSTALL_ENVIRONMENT"
echo "L4T_VERSION: $L4T_VERSION"
echo "HOLOLINK_CUDA_ARCHS: $HOLOLINK_CUDA_ARCHS"
echo "IBV_DEVICES: $IBV_DEVICES"
echo "HOLOLINK_ROCE_USE_GPU_VRAM: $HOLOLINK_ROCE_USE_GPU_VRAM"
echo "HOLOLINK_INSTALL_INFERENCE: $HOLOLINK_INSTALL_INFERENCE"
echo "HOLOLINK_USE_SCCACHE: $HOLOLINK_USE_SCCACHE"
echo "DOCKER_IMAGE_STORE: $DOCKER_IMAGE_STORE, $DOCKER_ROOT_FREE_GIB GiB free on $DOCKER_ROOT"
echo "PRUNE_CACHE: $PRUNE_CACHE_OPT ($PRUNE_CACHE_REASON)"
set -x

# Drop the BuildKit cache: the layer snapshots ("regular" records) and, in a second call because
# several --filter options are ANDed, the apt/pip/sccache cache mounts ("exec.cachemount"
# records). Loud on purpose: the banner names the reason and the free space afterwards.
# IDs of every record in the BuildKit cache of the default builder, sorted.
build_cache_ids()
{
    docker buildx du --verbose 2>/dev/null | awk '/^ID:/ {print $2}' | sort -u
}

# Remove the cache records that are not in BUILD_CACHE_BASELINE, i.e. the ones created since this
# script started: this build's layer snapshots, builder stages and cache mounts. Records that
# existed before (other projects' builds on a shared host or CI runner) are never touched. A
# record BuildKit refuses to drop (in use by a running build) is reported and skipped.
prune_build_cache()
{
    set +x
    NEW_RECORDS=$(mktemp)
    build_cache_ids | comm -13 "$BUILD_CACHE_BASELINE" - > "$NEW_RECORDS"
    NEW_RECORD_COUNT=$(grep -c . "$NEW_RECORDS" || true)
    echo "################################################################################"
    echo "# PRUNE_CACHE: $1"
    echo "# ($PRUNE_CACHE_REASON)"
    echo "# Dropping the $NEW_RECORD_COUNT BuildKit cache record(s) created since this script started;"
    echo "# records that existed before are kept. Opt out with --no-prune-cache or PRUNE_CACHE=0."
    echo "################################################################################"
    while read -r record_id
    do
        [ -n "$record_id" ] || continue
        docker buildx prune --force --filter "id=$record_id" > /dev/null \
            || echo "# PRUNE_CACHE: could not drop cache record $record_id (in use?); skipped"
    done < "$NEW_RECORDS"
    rm -f "$NEW_RECORDS"
    echo "################################################################################"
    echo "# PRUNE_CACHE: done; $(df --output=avail -k "$DOCKER_ROOT" | tail -n1 | awk '{print int($1 / 1024 / 1024)}') GiB now free on ${DOCKER_ROOT}."
    echo "################################################################################"
    set -x
}

# Baseline for the prunes: whatever is in the BuildKit cache now predates this build and is
# never removed. Taken before the first docker build so both prunes measure against it.
BUILD_CACHE_BASELINE=
if [ $PRUNE_CACHE_OPT -ne 0 ]
then
    BUILD_CACHE_BASELINE=$(mktemp)
    build_cache_ids > "$BUILD_CACHE_BASELINE"
fi

# Build the development container.  We specifically rely on buildkit skipping
# the dgpu or igpu stages that aren't included in the final image we're
# creating.
DOCKER_BUILDKIT=1 docker build \
    $NO_CACHE_ARG \
    --network=host \
    --build-arg "CONTAINER_TYPE=$CONTAINER_TYPE" \
    --build-arg "HSDK_VERSION=$HSDK_VERSION" \
    --build-arg "L4T_VERSION=$L4T_VERSION" \
    --build-arg "IBV_DEVICES=$IBV_DEVICES" \
    -t hololink-prototype:$VERSION \
    -f $HERE/Dockerfile \
    $PROTOTYPE_OPTIONS \
    $ROOT

# hololink-prototype is now in the image store, so its BuildKit layer snapshots are only a
# rebuild accelerator. With Docker's containerd image store every layer built above otherwise
# sits on disk three times while hololink-demo builds: as the build snapshot, as the exported
# compressed blob and as the unpacked image layer (~5 GB of snapshots for this image). The
# prune drops those snapshots and the apt/pip/sccache cache mounts (~3-4 GB of debs and wheels
# the demo build does not need: its own downloads, TensorRT above all, come later) -- only the
# records this build created; anything already in the cache when the script started stays.
if [ $PRUNE_CACHE_OPT -ne 0 ]
then
    prune_build_cache "dropping the hololink-prototype layer snapshots and the apt/pip/sccache cache mounts before the hololink-demo build; the next hololink-prototype build re-runs and re-downloads every layer"
fi

# Build a container that has python extensions set up.
DOCKER_BUILDKIT=1 docker build \
    $NO_CACHE_ARG \
    --network=host \
    --build-arg CONTAINER_VERSION=hololink-prototype:$VERSION \
    --build-arg "CONTAINER_TYPE=$CONTAINER_TYPE" \
    --build-arg "HOLOLINK_INSTALL_INFERENCE=$HOLOLINK_INSTALL_INFERENCE" \
    --build-arg "INSTALL_ENVIRONMENT=$INSTALL_ENVIRONMENT" \
    --build-arg "CUDAARCHS=$HOLOLINK_CUDA_ARCHS" \
    --build-arg "HOLOLINK_ROCE_USE_GPU_VRAM=$HOLOLINK_ROCE_USE_GPU_VRAM" \
    --build-arg "HOLOLINK_USE_SCCACHE=$HOLOLINK_USE_SCCACHE" \
    -t hololink-demo:$VERSION \
    -f $HERE/Dockerfile.demo \
    $ROOT

# Both images are in the store; what is left in the BuildKit cache is the hololink-demo build's
# builder stage (~2 GB: source, build tree, wheel), its layer snapshots (another copy of
# TensorRT) and the cache mounts refilled above. On a tight host drop them too, so the images
# themselves are all that stays behind.
if [ $PRUNE_CACHE_OPT -ne 0 ]
then
    prune_build_cache "dropping the hololink-demo builder-stage and layer snapshots and the cache mounts now that both images are in the store"
fi
