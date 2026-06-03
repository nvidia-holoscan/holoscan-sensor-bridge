# syntax=docker/dockerfile:1

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

# Container providing build environment for Holoscan Sensor Bridge.
# Note that some Jetson configurations require building on the target platform
# in order to mount driver and development dependencies

ARG CONTAINER_TYPE=invalid
ARG HSDK_VERSION=4.2.0
ARG DRIVER_CUDA_VERSION_MAJOR=invalid
ARG APT_CACHE_MOUNT="--mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked"

# dgpu (IGX Orin) specific configuration goes here.
# NOTE: if using -runtime- container instead of -devel-, libcufft-dev, libcurand-dev, cuda-cudart-dev, and cuda-driver-dev should also be installed
FROM nvcr.io/nvidia/cuda:12.6.3-runtime-ubuntu22.04 AS build_cuda12_dgpu
ARG L4T_VERSION=invalid
ENV CUDA_VERSION="${CUDA_VERSION:-12.6.3}"
ENV CUDA_VERSION_MAJOR="${CUDA_VERSION%%.*}"
ENV HSDK_PKG_NAME="holoscan"
ENV L4T_MAJ_VER="${L4T_VERSION%%.*}"
ENV L4T_MIN_PATCH_VER="${L4T_VERSION#*.}"
ENV L4T_MIN_VER="${L4T_MIN_PATCH_VER%%.*}"

# Since we use CTK 12.6, the l4t repo versions must be 36.4.1 (IGX 1.1+) to pull compatible dev packages \
# the l4t dgpu repository does not match the L4T version in IGX 1.2+; pinned at 36.4.1
RUN ${APT_CACHE_MOUNT} apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y curl \
    && echo "deb https://repo.download.nvidia.com/jetson/dgpu-rm r36.4.1 main" > /etc/apt/sources.list.d/nvidia-l4t-apt-source.list \
    && curl -Os --output-dir /etc/apt/trusted.gpg.d/ https://repo.download.nvidia.com/jetson/jetson-ota-public.asc

# PVA CRC wheel build needs NvSciBuf (required by cuPVA). Copy from host when building on IGX/L4T.
COPY --from=l4t-libs \
    libnvscibuf.so \
    libnvscicommon.so.1 \
    /usr/lib/aarch64-linux-gnu/nvidia/

# l4t igpu (AGX Orin, Nano, NX, IGX Orin igpu) jetpack 6 specific configuration goes here.
FROM nvcr.io/nvidia/l4t-cuda:12.6.11-runtime AS build_cuda12_igpu
ARG L4T_VERSION=invalid
ARG TARGETARCH
ENV CUDA_VERSION="${CUDA_VERSION:-12.6.11}"
ENV CUDA_VERSION_MAJOR="${CUDA_VERSION%%.*}"
ENV HSDK_PKG_NAME="holoscan-cuda-${CUDA_VERSION_MAJOR}"
ENV L4T_MAJ_VER="${L4T_VERSION%%.*}"
ENV L4T_MIN_PATCH_VER="${L4T_VERSION#*.}"
ENV L4T_MIN_VER="${L4T_MIN_PATCH_VER%%.*}"

RUN ${APT_CACHE_MOUNT} apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y curl \
    # Since we use CTK 12.6, the l4t repo versions must be 36.4 (JP 6.1+) or 36.5 (JP 6.2.2+) to pull compatible dev packages. \
    && echo "deb https://repo.download.nvidia.com/jetson/common r${L4T_MAJ_VER}.${L4T_MIN_VER} main" > /etc/apt/sources.list.d/nvidia-l4t-apt-source.list \
    && echo "deb https://repo.download.nvidia.com/jetson/t234 r${L4T_MAJ_VER}.${L4T_MIN_VER} main" >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list \
    && echo "deb https://repo.download.nvidia.com/jetson/ffmpeg r${L4T_MAJ_VER}.${L4T_MIN_VER} main" >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list \
    && curl -Os --output-dir /etc/apt/trusted.gpg.d/ https://repo.download.nvidia.com/jetson/jetson-ota-public.asc \
    && apt-get update \
    # EGL and libGLESv2 is needed by Argus api otherwise the library throws error \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y -q libgles2-mesa-dev \
    # we do not have the permission to ./mm-api/DEBIAN/postinst \
    # to just install nvidia-l4t-jetson-multimedia-api via apt install \
    && apt-get download nvidia-l4t-jetson-multimedia-api \
    && dpkg-deb -R ./nvidia-l4t-jetson-multimedia-api_*_arm64.deb ./mm-api \
    && cp -r ./mm-api/usr/src/jetson_multimedia_api /usr/src/jetson_multimedia_api \
    && sed -i 's/sudo//' ./mm-api/DEBIAN/postinst \
    && ./mm-api/DEBIAN/postinst \
    && rm -rf ./nvidia-l4t-jetson-multimedia-api_*_arm64.deb ./mm-api

COPY --from=l4t-libs \
    libnvargus.so \
    libnvargus_socketclient.so \
    libnvargus_socketserver.so \
    libnvfusacap.so \
    libnvodm_imager.so \
    libnvscf.so \
    /usr/lib/aarch64-linux-gnu/nvidia/
ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:$LD_LIBRARY_PATH

# DGX Spark and Base for AGX/IGX Thor configurations go here
# NOTE: if using -runtime- container instead of -devel-, libcufft-dev, libcurand-dev, cuda-cudart-dev, and cuda-driver-dev should also be installed
FROM nvcr.io/nvidia/cuda:13.0.0-runtime-ubuntu24.04 AS build_cuda13
ARG TARGETARCH
ENV CUDA_VERSION="${CUDA_VERSION:-13.0.0}"
ENV CUDA_VERSION_MAJOR="${CUDA_VERSION%%.*}"
ENV HSDK_PKG_NAME="holoscan-cuda-${CUDA_VERSION_MAJOR}"
# install holoscan dev dependencies so that holoscan-cuda-13=4.* doesn't pull the incompatible CTK 13.2 components
# Holds on toolkit config packages (including *-13-2-*) so holoscan/HSDK apt deps cannot switch /usr/local/cuda to CTK 13.2.
RUN ${APT_CACHE_MOUNT} apt-get update && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y \
    cuda-cudart-dev-13-0 \
    cuda-driver-dev-13-0 \
    cuda-nvcc-13-0 \
    cuda-nvrtc-dev-13-0 \
    libnpp-dev-13-0

# cuda13 on L4T (e.g. IGX Thor): add NvSciBuf from host for PVA CRC wheel build.
FROM build_cuda13 AS build_cuda13_l4t
ARG L4T_VERSION=invalid
ENV L4T_MAJ_VER="${L4T_VERSION%%.*}"
ENV L4T_MIN_PATCH_VER="${L4T_VERSION#*.}"
ENV L4T_MIN_VER="${L4T_MIN_PATCH_VER%%.*}"

COPY --from=l4t-libs \
    libnvscibuf.so \
    libnvscicommon.so.1 \
    /usr/lib/aarch64-linux-gnu/nvidia/

# cuda13 on L4T (e.g. IGX Thor): add NvSciBuf from host for PVA CRC wheel build.
FROM build_cuda13_l4t AS build_cuda13_igpu
ARG L4T_VERSION=invalid

# EGL and libGLESv2 are needed by Argus api otherwise the library throws error.
RUN ${APT_CACHE_MOUNT} apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y curl \
    && curl -Os --output-dir /etc/apt/trusted.gpg.d/ https://repo.download.nvidia.com/jetson/jetson-ota-public.asc \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y -q libgles2-mesa-dev

COPY --from=l4t-src jetson_multimedia_api* /usr/src/jetson_multimedia_api
RUN if [ ! -d "/usr/src/jetson_multimedia_api" ]; then \
        curl -L -O https://developer.nvidia.com/downloads/embedded/L4T/r${L4T_MAJ_VER}_Release_v${L4T_MIN_PATCH_VER}/release/Jetson_Multimedia_API_R${L4T_VERSION}_aarch64.tbz2 \
        && tar xjf Jetson_Multimedia_API_R${L4T_VERSION}_aarch64.tbz2 -C / \
        && rm Jetson_Multimedia_API_R${L4T_VERSION}_aarch64.tbz2; \
    fi

COPY --from=l4t-libs \
    libnvargus.so \
    libnvargus_socketclient.so \
    libnvargus_socketserver.so \
    libnvfusacap*.so \
    libnvodm_imager.so \
    libnvscf.so \
    /usr/lib/aarch64-linux-gnu/nvidia/
ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:$LD_LIBRARY_PATH

# AGX Thor/SIPL/FUSA specific configuration goes here.
FROM build_cuda13_igpu AS build_cuda13_sipl
ARG L4T_VERSION=invalid
# this is used to identify if the environment supports CoE offload for SIPL/FUSA builds
ENV COE_OFFLOAD=1

# L4T repos
# Copy the Jetson SIPL API from the host (fall back to download if not installed on the host).
# cmake and python for Jetson Multimedia API and SIPL API build.
COPY --from=l4t-src jetson_sipl_api* /usr/src/jetson_sipl_api
RUN ${APT_CACHE_MOUNT} apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y curl \
    && echo "deb https://repo.download.nvidia.com/jetson/common r${L4T_MAJ_VER}.${L4T_MIN_VER} main" > /etc/apt/sources.list.d/nvidia-l4t-apt-source.list \
    && echo "deb https://repo.download.nvidia.com/jetson/som r${L4T_MAJ_VER}.${L4T_MIN_VER} main" >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list \
    && if [ ! -d "/usr/src/jetson_sipl_api" ]; then \
        apt-get update \
        && DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3 cmake \
        && curl -L -O https://developer.nvidia.com/downloads/embedded/L4T/r${L4T_MAJ_VER}_Release_v${L4T_MIN_PATCH_VER}/release/Jetson_SIPL_API_R${L4T_VERSION}_aarch64.tbz2 \
        && tar xjf Jetson_SIPL_API_R${L4T_VERSION}_aarch64.tbz2 -C / \
        && rm Jetson_SIPL_API_R${L4T_VERSION}_aarch64.tbz2 \
        && cd /usr/src/jetson_sipl_api/sipl \
        && rm -rf build && mkdir build && cd build \
        && cmake .. \
        && make -j \
        && make install ; \
    fi

# Copy the L4T libraries into the container.
COPY --from=l4t-libs \
    libgnat-*.so \
    libnvbufsurface.so \
    libnvbuf_fdmap.so.1.0.0 \
    libnvcamerahal.so \
    libnvfusacapinterface.so \
    libnvisppg.so \
    libnvisp.so \
    libnvos.so \
    libnvrm_chip.so \
    libnvrm_gpu.so \
    libnvrm_host1x.so \
    libnvrm_mem.so \
    libnvrm_surface.so \
    libnvrm_sync.so \
    libnvsciipc.so \
    libnvscisync.so \
    libnvsipl_control.so \
    libnvsipl_devblk_cdi.so \
    libnvsipl_devblk_crypto.so \
    libnvsipl_devblk_ddi.so \
    libnvsipl_devblk.so \
    libnvsipl_pipeline.so \
    libnvsipl_query.so \
    libnvsipl.so \
    libnvsocsys.so \
    libnvtegrahv.so \
    /usr/lib/aarch64-linux-gnu/nvidia/

# Common configurations for all supported platforms here
FROM build_${CONTAINER_TYPE}

# ARGs must be redeclared after FROM to be available in this build stage
ARG HSDK_VERSION=invalid
ARG L4T_VERSION=invalid
ARG CUDA_NATIVE_ARCH=invalid
ARG IBV_DEVICES=
ARG TARGETARCH
ARG DOCA_VERSION=3.3.0
ARG DEBIAN_FRONTEND=noninteractive

# controls whether roce is disabled in builds (no IBV_DEVICES-->no roce)
ENV IBV_DEVICES=${IBV_DEVICES}
# the python environment is protected in the cuda containers, so enable breaking system packages for all containers
ENV PIP_BREAK_SYSTEM_PACKAGES=1
# run as root user...ignore warnings about installing as root inside container
ENV PIP_ROOT_USER_ACTION=ignore
ENV CUDA_NATIVE_ARCH=${CUDA_NATIVE_ARCH}
ENV CUDA_VERSION_MINOR_PATCH="${CUDA_VERSION#*.}"
ENV CUDA_VERSION_MINOR="${CUDA_VERSION_MINOR_PATCH%%.*}"
ENV CUDA_PATH=/usr/local/cuda

# add trtexec to path for inference engine building examples. NOTE: for tensorrt 10.3 on JP 6.1+, engine building frequently fails within container with memory limits due to bug
# suggestion is to build engine outside of container: https://forums.developer.nvidia.com/t/tensorrt-has-insufficent-memory-in-container/318948
# because hololink path is mounted in container, the build must happen within hololink/holoscan-sensor-bridge directory and --engine option must be specified to the example application
ENV PATH="/usr/src/tensorrt/bin:$PATH"

# Install dependencies for APT setup
RUN ${APT_CACHE_MOUNT} apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl \
    gnupg

WORKDIR /opt

# Configure the DOCA APT repository
RUN if [ -n "$IBV_DEVICES" ] ; then \
        if [ "${TARGETARCH}" = "amd64" ]; then \
            DOCA_ARCH="amd64"; \
        elif [ "${TARGETARCH}" = "arm64" ]; then \
            DOCA_ARCH="arm64"; \
        else \
            echo "Unknown architecture: $TARGETARCH"; \
            exit 1; \
        fi \
        && apt-get update \
        && UBUNTU_VERSION=$(cat /etc/os-release | grep "VERSION_ID" | awk -F'=' '{print $2}' | sed 's/[."]//g') \
        && curl -fL -O https://www.mellanox.com/downloads/DOCA/DOCA_v${DOCA_VERSION}/host/doca-host_${DOCA_VERSION}-088000-26.01-ubuntu${UBUNTU_VERSION}_${DOCA_ARCH}.deb \
        && apt-get install -o DPkg::Options::="--force-confnew" -y ./doca-host_${DOCA_VERSION}-088000-26.01-ubuntu${UBUNTU_VERSION}_${DOCA_ARCH}.deb \
        && rm -f ./doca-host_${DOCA_VERSION}-088000-26.01-ubuntu${UBUNTU_VERSION}_${DOCA_ARCH}.deb \
        && apt-get update \
    ; fi

# Set up CMake PPA for CMake 3.31
RUN ${APT_CACHE_MOUNT} \
    . /etc/os-release \
    && KW_KEYRING="/usr/share/keyrings/kitware-archive-keyring.gpg" \
    && curl -fsSL "https://apt.kitware.com/keys/kitware-archive-latest.asc" \
        | gpg --dearmor -o "$KW_KEYRING" \
    && echo "deb [signed-by=$KW_KEYRING] https://apt.kitware.com/ubuntu/ $VERSION_CODENAME main" \
        > /etc/apt/sources.list.d/kitware.list \
    && apt-get update

# System package dependencies
#
# net-tools gives us the "arp" command
# iputils-ping gives us the "ping" command
# iproute2 gives us the "ip" command
# For Holoviz: vulkan-tools, glslang-tools, vulkan-validationlayers, libvulkan-dev
#
# Pin to CMake 3.31.11 to work around 3.31.12 installation error (CMake Issue#27775).
RUN ${APT_CACHE_MOUNT} apt-get update \
    && . /etc/os-release \
    && UBUNTU_VERSION=$(echo $VERSION_ID | sed 's/\.//g') \
    && curl -fsSL -O https://developer.download.nvidia.com/compute/nvcomp/5.2.0/local_installers/nvcomp-local-repo-ubuntu${UBUNTU_VERSION}-5.2.0_5.2.0-1_${TARGETARCH}.deb \
    && dpkg -i nvcomp-local-repo-ubuntu${UBUNTU_VERSION}-5.2.0_5.2.0-1_${TARGETARCH}.deb \
    && cp /var/nvcomp-local-repo-ubuntu${UBUNTU_VERSION}-5.2.0/nvcomp-*-keyring.gpg /usr/share/keyrings/ \
    && apt-get update \
    && DEB_DEPENDENCIES="libnvcomp5-cuda-${CUDA_VERSION_MAJOR} libnvcomp5-dev-cuda-${CUDA_VERSION_MAJOR} libnvcomp5-static-cuda-${CUDA_VERSION_MAJOR} nvcomp-cuda-${CUDA_VERSION_MAJOR}" \
    && if [ -n "$IBV_DEVICES" ] ; then \
        DEB_DEPENDENCIES="$DEB_DEPENDENCIES libibverbs-dev libdoca-sdk-gpunetio-dev libdoca-sdk-verbs-dev mlnx-ofed-kernel-utils ibverbs-utils" \
    ; fi \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y -q \
    # Container tools
    git \
    curl \
    lsb-release \
    python3 \
    x11-utils \
    vulkan-tools \
    glslang-tools \
    net-tools \
    iputils-ping \
    iproute2 \
    # Build tools
    build-essential \
    cmake="3.31.11*" \
    cmake-data="3.31.11*" \
    ninja-build \
    pkgconf \
    # Development libraries
    python3-setuptools \
    python3-dev \
    libnuma-dev \
    libelf-dev \
    libegl1 \
    libfmt-dev \
    libyaml-cpp-dev \
    libcurlpp-dev \
    libssl-dev \
    libvulkan-dev \
    vulkan-validationlayers \
    # these should already be installed, but ensure they are present
    libcurand-dev-${CUDA_VERSION_MAJOR}-${CUDA_VERSION_MINOR} \
    libcufft-dev-${CUDA_VERSION_MAJOR}-${CUDA_VERSION_MINOR} \
    cuda-cudart-dev-${CUDA_VERSION_MAJOR}-${CUDA_VERSION_MINOR} \
    cuda-driver-dev-${CUDA_VERSION_MAJOR}-${CUDA_VERSION_MINOR} \
    cuda-nvcc-${CUDA_VERSION_MAJOR}-${CUDA_VERSION_MINOR} \
    cuda-nvrtc-dev-${CUDA_VERSION_MAJOR}-${CUDA_VERSION_MINOR} \
    libnpp-dev-${CUDA_VERSION_MAJOR}-${CUDA_VERSION_MINOR} \
    $DEB_DEPENDENCIES

# python package dependencies
# holoscan has a minimum pip version requirement that cannot be satisfied by the debian package
# bootstrap it, upgrade, then install holoscan and dependencies
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3 get-pip.py && rm -f get-pip.py \
    && python3 -m pip install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir meson ninja pyelftools numpy \
    && pip3 install holoscan-cu${CUDA_VERSION_MAJOR}==${HSDK_VERSION}

RUN ${APT_CACHE_MOUNT} \
    apt-get update \
    ; if ! apt list --installed 2>/dev/null | grep -qi ${HSDK_PKG_NAME} ; then \
        echo "installing ${HSDK_PKG_NAME}=${HSDK_VERSION}*" \
        && apt-get install --no-install-recommends -y ${HSDK_PKG_NAME}=${HSDK_VERSION}* \
    ; fi \
    ; if ! apt list --installed 2>/dev/null | grep -qi "holoscan" ; then \
        . /etc/os-release \
        && UBUNTU_VERSION=$(echo $VERSION_ID | sed 's/\.//g') \
        && if [ "${UBUNTU_VERSION}" = "2204" ]; then \
            curl -fsSL -O https://developer.download.nvidia.com/compute/holoscan/${HSDK_VERSION}/local_installers/holoscan-local-tegra-repo-ubuntu${UBUNTU_VERSION}-${HSDK_VERSION}_${HSDK_VERSION}-1_${TARGETARCH}.deb \
            && dpkg -i holoscan-local-tegra-repo-ubuntu${UBUNTU_VERSION}-${HSDK_VERSION}_${HSDK_VERSION}-1_${TARGETARCH}.deb \
            && cp /var/holoscan-local-tegra-repo-ubuntu${UBUNTU_VERSION}-${HSDK_VERSION}/holoscan-*-keyring.gpg /usr/share/keyrings/     \
        ; else \
            curl -fsSL -O https://developer.download.nvidia.com/compute/holoscan/${HSDK_VERSION}/local_installers/holoscan-local-repo-ubuntu${UBUNTU_VERSION}-${HSDK_VERSION}_${HSDK_VERSION}-1_${TARGETARCH}.deb \
            && dpkg -i holoscan-local-repo-ubuntu${UBUNTU_VERSION}-${HSDK_VERSION}_${HSDK_VERSION}-1_${TARGETARCH}.deb \
            && cp /var/holoscan-local-repo-ubuntu${UBUNTU_VERSION}-${HSDK_VERSION}/holoscan-*-keyring.gpg /usr/share/keyrings/    \
        ; fi \
        && apt-get update \
        && echo "${HSDK_PKG_NAME}=${HSDK_VERSION} not found, trying local CUDA APT repo" \
        && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y --allow-downgrades ${HSDK_PKG_NAME}=${HSDK_VERSION}* \
    ; fi \
    && echo "/opt/nvidia/holoscan/lib" >> /etc/ld.so.conf.d/holoscan.conf \
    && ldconfig \
    && ldconfig -p | grep -q "libholoscan_core.so" || (echo "holoscan libraries not found" && exit 1)

RUN rm -rf /var/lib/apt/lists/*

# Copy PVA SDK from host if present (optional for systems with PVA hardware)
# Build context 'pva-sdk' points to /opt/nvidia (if exists) or /tmp/empty-pva-context
# Use shell to handle missing source gracefully
# Copy PVA SDK 2.9 from host if present (for systems with PVA hardware)
RUN --mount=type=bind,from=pva-sdk,source=.,target=/tmp/pva-sdk-2.9 \
    if [ -d "/tmp/pva-sdk-2.9" ] && [ -n "`ls -A /tmp/pva-sdk-2.9`" ] ; then \
        mkdir -p /opt/nvidia/pva-sdk-2.9 && \
        cp -r /tmp/pva-sdk-2.9/* /opt/nvidia/pva-sdk-2.9/ && \
        echo "PVA SDK 2.9 copied successfully"; \
    else \
        echo "PVA SDK 2.9 not found - PVA CRC operator will be skipped"; \
    fi

# repeated ARG because it has to be inside the FROM above
ARG CONTAINER_TYPE=invalid
ARG HSDK_VERSION=invalid
ARG L4T_VERSION=invalid

