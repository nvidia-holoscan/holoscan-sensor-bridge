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
ARG HSDK_VERSION=5.0.0
ARG NVCOMP_VERSION=invalid
ARG DRIVER_CUDA_VERSION_MAJOR=invalid

# DGX Spark, and the base for the AGX Orin / AGX Thor / IGX Thor configurations, go here
# NOTE: if using -runtime- container instead of -devel-, libcufft-dev, libcurand-dev, cuda-cudart-dev, and cuda-driver-dev should also be installed
FROM nvcr.io/nvidia/cuda:13.0.0-runtime-ubuntu24.04 AS build_cuda13
ARG CONTAINER_TYPE
ARG TARGETARCH
ENV CUDA_VERSION="${CUDA_VERSION:-13.0.0}"
ENV CUDA_VERSION_MAJOR="${CUDA_VERSION%%.*}"
# Install the holoscan dev dependencies first so that holoscan=5.* (installed further down from
# the HSDK local APT repository) doesn't pull the incompatible CTK 13.2 components.
# Holds on toolkit config packages (including *-13-2-*) so holoscan/HSDK apt deps cannot switch /usr/local/cuda to CTK 13.2.
#
# The -dev packages also ship static variants of every CUDA library (nvrtc, nvptxcompiler, npp*:
# ~350 MB here, ~800 MB over the whole image). Neither hololink nor the HSDK links any of them, so
# they are deleted in the same layer that installs them. libcudart_static.a stays because CMake's
# CUDA language links the static runtime by default; libnppicc_static.a stays because
# holoscan-vision-config.cmake names it for static BayerDemosaicOp consumers.
RUN --mount=type=cache,id=hololink-${CONTAINER_TYPE}-${TARGETARCH}-apt-archives,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=hololink-${CONTAINER_TYPE}-${TARGETARCH}-apt-lists,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && printf 'Binary::apt::APT::Keep-Downloaded-Packages "true";\n' > /etc/apt/apt.conf.d/99-keep-downloaded-packages \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    cuda-cudart-dev-13-0 \
    cuda-driver-dev-13-0 \
    cuda-nvcc-13-0 \
    cuda-nvrtc-dev-13-0 \
    libnpp-dev-13-0 \
    && find /usr/local/cuda-*/targets/*/lib -name '*_static.a' ! -name 'libcudart_static.a' ! -name 'libnppicc_static.a' -delete

# cuda13 on L4T (IGX Thor igpu/dgpu, AGX Thor): add NvSciBuf from host for PVA CRC wheel build
# and the Jetson Multimedia API (Argus) plus its host libraries for the Argus ISP operator.
FROM build_cuda13 AS build_cuda13_l4t
ARG L4T_VERSION=invalid
ENV L4T_MAJ_VER="${L4T_VERSION%%.*}"
ENV L4T_MIN_PATCH_VER="${L4T_VERSION#*.}"
ENV L4T_MIN_VER="${L4T_MIN_PATCH_VER%%.*}"

# The host BSP release file ("# R38 (release), REVISION: 2.0, ..."); python/setup.py and
# tests/conftest.py parse it for the L4T version the same way docker/build.sh does.
COPY --from=l4t-release nv_tegra_release /etc/nv_tegra_release

COPY --from=l4t-libs \
    libnvscibuf.so \
    libnvscicommon.so.1 \
    /usr/lib/aarch64-linux-gnu/nvidia/

# EGL and libGLESv2 are needed by Argus api otherwise the library throws error.
RUN --mount=type=cache,id=hololink-${CONTAINER_TYPE}-${TARGETARCH}-apt-archives,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=hololink-${CONTAINER_TYPE}-${TARGETARCH}-apt-lists,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl \
    && curl -Os --output-dir /etc/apt/trusted.gpg.d/ https://repo.download.nvidia.com/jetson/jetson-ota-public.asc \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends libgles2-mesa-dev

# Copy the Jetson Multimedia API from the host (fall back to download if not installed on the host).
# The host's /usr/src is the l4t-src build context; it is bind-mounted rather than COPY'd so the
# shell can check for the directory at build time -- a COPY with no match fails the build before
# any fallback can run. The download is best-effort: not every L4T release publishes a
# Jetson_Multimedia_API tarball, so a missing package is reported as a loud warning rather than
# failing the build. Only the Argus ISP operator (HOLOLINK_BUILD_ARGUS_ISP, OFF by default) needs it.
RUN --mount=type=bind,from=l4t-src,target=/mnt/l4t-src \
    MMAPI_HOST=""; \
    for d in /mnt/l4t-src/jetson_multimedia_api*; do \
        if [ -d "$d" ]; then MMAPI_HOST="$d"; break; fi; \
    done; \
    if [ -n "$MMAPI_HOST" ]; then \
        echo "Jetson Multimedia API: copying $MMAPI_HOST from the host"; \
        cp -a "$MMAPI_HOST" /usr/src/jetson_multimedia_api; \
    else \
        MMAPI_TARBALL="Jetson_Multimedia_API_R${L4T_VERSION}_aarch64.tbz2"; \
        MMAPI_URL="https://developer.nvidia.com/downloads/embedded/L4T/r${L4T_MAJ_VER}_Release_v${L4T_MIN_PATCH_VER}/release/${MMAPI_TARBALL}"; \
        echo "Jetson Multimedia API: not on the host, downloading ${MMAPI_URL}"; \
        if curl -fsSL -o "${MMAPI_TARBALL}" "${MMAPI_URL}" \
           && tar xjf "${MMAPI_TARBALL}" -C /; then \
            rm -f "${MMAPI_TARBALL}"; \
        else \
            rm -f "${MMAPI_TARBALL}"; \
            echo "" >&2; \
            echo "##############################################################################" >&2; \
            echo "# WARNING: Jetson Multimedia API not available for L4T ${L4T_VERSION}" >&2; \
            echo "#   not found on the host under /usr/src/jetson_multimedia_api, and the" >&2; \
            echo "#   download failed:" >&2; \
            echo "#     ${MMAPI_URL}" >&2; \
            echo "#   Continuing without it; the Argus ISP operator (HOLOLINK_BUILD_ARGUS_ISP)" >&2; \
            echo "#   cannot be built in this container." >&2; \
            echo "##############################################################################" >&2; \
            echo "" >&2; \
        fi; \
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
FROM build_cuda13_l4t AS build_cuda13_sipl
ARG L4T_VERSION=invalid
# this is used to identify if the environment supports CoE offload for SIPL/FUSA builds
ENV COE_OFFLOAD=1

# L4T repos
# Copy the Jetson SIPL API from the host (fall back to download if the host has none, or an
# incomplete one without the SIPL sources). A wildcard COPY with no match is not an error.
# cmake and python for Jetson Multimedia API and SIPL API build.
COPY --from=l4t-src jetson_sipl_api* /usr/src/jetson_sipl_api
RUN --mount=type=cache,id=hololink-${CONTAINER_TYPE}-${TARGETARCH}-apt-archives,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=hololink-${CONTAINER_TYPE}-${TARGETARCH}-apt-lists,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl \
    && echo "deb https://repo.download.nvidia.com/jetson/common r${L4T_MAJ_VER}.${L4T_MIN_VER} main" > /etc/apt/sources.list.d/nvidia-l4t-apt-source.list \
    && echo "deb https://repo.download.nvidia.com/jetson/som r${L4T_MAJ_VER}.${L4T_MIN_VER} main" >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list \
    && if [ ! -f "/usr/src/jetson_sipl_api/sipl/CMakeLists.txt" ]; then \
        apt-get update \
        && DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends python3 cmake \
        && curl -fL -O https://developer.nvidia.com/downloads/embedded/L4T/r${L4T_MAJ_VER}_Release_v${L4T_MIN_PATCH_VER}/release/Jetson_SIPL_API_R${L4T_VERSION}_aarch64.tbz2 \
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
ARG CONTAINER_TYPE
ARG HSDK_VERSION=invalid
ARG NVCOMP_VERSION=5.2.0
ARG L4T_VERSION=invalid
ARG IBV_DEVICES=
ARG TARGETARCH
ARG DOCA_VERSION=3.4.0
ARG DEBIAN_FRONTEND=noninteractive
# Record the HSDK release in the image under the name every later layer uses, here and in
# Dockerfile.demo, which builds FROM this image and installs holoscan-inference from the same
# local APT repository (see
# HOLOLINK_INSTALL_INFERENCE there). This image itself never carries holoscan-inference or
# TensorRT (libnvinfer10 alone is ~2.9 GB): only the tao_peoplenet / body_pose_estimation
# examples need them, and they run from hololink-demo.
ENV HOLOLINK_HSDK_VERSION=${HSDK_VERSION}

# controls whether roce is disabled in builds (no IBV_DEVICES-->no roce)
ENV IBV_DEVICES=${IBV_DEVICES}
# the python environment is protected in the cuda containers, so enable breaking system packages for all containers
ENV PIP_BREAK_SYSTEM_PACKAGES=1
# run as root user...ignore warnings about installing as root inside container
ENV PIP_ROOT_USER_ACTION=ignore
ENV CUDA_VERSION_MINOR_PATCH="${CUDA_VERSION#*.}"
ENV CUDA_VERSION_MINOR="${CUDA_VERSION_MINOR_PATCH%%.*}"
ENV CUDA_PATH=/usr/local/cuda

# Install dependencies for APT setup.
# The pinning file makes packages from the local APT repositories set up below (DOCA, nvcomp,
# HSDK) win over the online repositories. It needs to come alphanumerically before any other
# pinning files where the package matches. On L4T hosts, the cuda-repository-pin-600 file is
# present with a "Package: *" and otherwise takes precedence over this
RUN --mount=type=cache,id=hololink-${CONTAINER_TYPE}-${TARGETARCH}-apt-archives,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=hololink-${CONTAINER_TYPE}-${TARGETARCH}-apt-lists,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    && printf 'Package: *\nPin: origin ""\nPin-Priority: 1001\n' > /etc/apt/preferences.d/00-nv-local-repo

WORKDIR /opt

# Configure the DOCA APT repository

RUN --mount=type=cache,id=hololink-${CONTAINER_TYPE}-${TARGETARCH}-apt-archives,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=hololink-${CONTAINER_TYPE}-${TARGETARCH}-apt-lists,target=/var/lib/apt,sharing=locked \
    if [ -n "$IBV_DEVICES" ] ; then \
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
        && curl -fL -O https://www.mellanox.com/downloads/DOCA/DOCA_v${DOCA_VERSION}/host/doca-host_${DOCA_VERSION}-085000-26.04-ubuntu${UBUNTU_VERSION}_${DOCA_ARCH}.deb \
        && apt-get install -o DPkg::Options::="--force-confnew" -y ./doca-host_${DOCA_VERSION}-085000-26.04-ubuntu${UBUNTU_VERSION}_${DOCA_ARCH}.deb \
        && rm -f ./doca-host_${DOCA_VERSION}-085000-26.04-ubuntu${UBUNTU_VERSION}_${DOCA_ARCH}.deb \
        && apt-get update \
        # RoCE / GPUNetIO packages from the DOCA repository
        && DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends \
            libibverbs-dev \
            libdoca-sdk-gpunetio-dev \
            libdoca-sdk-verbs-dev \
            mlnx-ofed-kernel-utils \
            ibverbs-utils \
        # doca-host is only the local APT repository: ~800 MB of .debs (firmware updater, HPC-X, ...)
        # under /usr/share that nothing needs once the packages above are installed. Purge it in
        # this same layer so the pool never lands in the image.
        && apt-get purge -y doca-host \
        && rm -rf /usr/share/doca-host-* /etc/apt/sources.list.d/doca.list \
    ; fi

# Set up CMake PPA for CMake 4.x
RUN --mount=type=cache,id=hololink-${CONTAINER_TYPE}-${TARGETARCH}-apt-archives,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=hololink-${CONTAINER_TYPE}-${TARGETARCH}-apt-lists,target=/var/lib/apt,sharing=locked \
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
# CMake 4.x required by HSDK5's CMake config files.
RUN --mount=type=cache,id=hololink-${CONTAINER_TYPE}-${TARGETARCH}-apt-archives,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=hololink-${CONTAINER_TYPE}-${TARGETARCH}-apt-lists,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && . /etc/os-release \
    && UBUNTU_VERSION=$(echo $VERSION_ID | sed 's/\.//g') \
    && curl -fsSL -O https://developer.download.nvidia.com/compute/nvcomp/${NVCOMP_VERSION}/local_installers/nvcomp-local-repo-ubuntu${UBUNTU_VERSION}-${NVCOMP_VERSION}_${NVCOMP_VERSION}-1_${TARGETARCH}.deb \
    && dpkg -i nvcomp-local-repo-ubuntu${UBUNTU_VERSION}-${NVCOMP_VERSION}_${NVCOMP_VERSION}-1_${TARGETARCH}.deb \
    && cp /var/nvcomp-local-repo-ubuntu${UBUNTU_VERSION}-${NVCOMP_VERSION}/nvcomp-*-keyring.gpg /usr/share/keyrings/ \
    && apt-get update \
    && DEB_DEPENDENCIES="libnvcomp5-cuda-${CUDA_VERSION_MAJOR}=${NVCOMP_VERSION}* libnvcomp5-dev-cuda-${CUDA_VERSION_MAJOR}=${NVCOMP_VERSION}* nvcomp-cuda-${CUDA_VERSION_MAJOR}=${NVCOMP_VERSION}*" \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends \
    # Container tools
    git \
    curl \
    lsb-release \
    python3 \
    x11-utils \
    xvfb \
    vulkan-tools \
    glslang-tools \
    net-tools \
    iputils-ping \
    iproute2 \
    # Build tools
    build-essential \
    cmake="4.*" \
    cmake-data="4.*" \
    ninja-build \
    # lld: fast multi-threaded linker (CMAKE_LINKER_TYPE=LLD). Static-links the
    # many tool/example binaries far faster than the default GNU BFD ld.
    lld \
    pkgconf \
    # Development libraries
    python3-setuptools \
    python3-dev \
    libnuma-dev \
    libelf-dev \
    libegl1 \
    libfmt-dev \
    libexpected-dev \
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
    $DEB_DEPENDENCIES \
    # the nvcomp local repository (~80 MB of .debs) is not needed once nvcomp is installed
    && apt-get purge -y nvcomp-local-repo-ubuntu${UBUNTU_VERSION}-${NVCOMP_VERSION} \
    && rm -rf /var/nvcomp-local-repo-* /etc/apt/sources.list.d/nvcomp-local-*.list \
    # libcufft-dev / libcurand-dev add ~440 MB of static libraries nothing here links (see above)
    && find /usr/local/cuda-*/targets/*/lib -name '*_static.a' ! -name 'libcudart_static.a' ! -name 'libnppicc_static.a' -delete

# sccache: a shared compiler cache so successive image builds skip recompiling
# unchanged C++/CUDA translation units. We use the RAPIDS sccache fork because it
# supports caching nvcc (CUDA) compilations. Matches the approach in holoscan-sdk's
# Dockerfile. This only places the binary on PATH; the compiler-launcher wiring and
# the SCCACHE_DIR cache mount live in Dockerfile.demo's builder stage.
ARG SCCACHE_VERSION=v0.14.0-rapids.19
ARG SCCACHE_BASE_URL=https://github.com/rapidsai/sccache/releases/download
RUN curl -fS -# -L -o /tmp/sccache.tar.gz \
        ${SCCACHE_BASE_URL}/${SCCACHE_VERSION}/sccache-${SCCACHE_VERSION}-$(uname -m)-unknown-linux-musl.tar.gz \
    && tar -xzf /tmp/sccache.tar.gz -C /usr/local/bin --strip-components=1 \
        sccache-${SCCACHE_VERSION}-$(uname -m)-unknown-linux-musl/sccache \
    && rm /tmp/sccache.tar.gz

# Slang shader compiler (build-time tool only). HoloViz-based operators compile their shaders
# to SPIR-V at configure/build time through the SDK's HolovizShaderHeader.cmake helper, which
# locates slangc through find_program() over CMAKE_PREFIX_PATH and PATH; keep this version in
# lockstep with HOLOSCAN_SLANG_VERSION in the holoscan-sdk tree (cmake/deps/slang.cmake).
ARG SLANG_VERSION=2026.14
ARG SLANG_SHA256_AMD64=9445cf0b35332bfda055c9424e307f957ff58befb04c5b45b9f9f29894a41cf8
ARG SLANG_SHA256_ARM64=7acbfe80260009263346f72efaadabfa7666248a045e720b573ba78fb008ac42
RUN case "${TARGETARCH}" in \
        amd64) slang_arch="x86_64"; slang_sha256="${SLANG_SHA256_AMD64}" ;; \
        arm64) slang_arch="aarch64"; slang_sha256="${SLANG_SHA256_ARM64}" ;; \
        *) echo "Unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 1 ;; \
    esac \
    && slang_archive="slang-${SLANG_VERSION}-linux-${slang_arch}.tar.gz" \
    && curl -fsSL --retry 5 --retry-delay 3 \
        "https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/${slang_archive}" \
        -o "/tmp/${slang_archive}" \
    && echo "${slang_sha256}  /tmp/${slang_archive}" | sha256sum -c - \
    && mkdir -p /opt/slang \
    && tar -xzf "/tmp/${slang_archive}" -C /opt/slang bin lib \
    && rm -rf "/tmp/${slang_archive}" /opt/slang/lib/cmake /opt/slang/lib/pkgconfig \
    && chmod -R a+rX /opt/slang \
    && test "$(/opt/slang/bin/slangc -v 2>&1)" = "${SLANG_VERSION}"
ENV PATH=/opt/slang/bin:${PATH}

# python package dependencies
# holoscan has a minimum pip version requirement that cannot be satisfied by the debian package
# bootstrap it, upgrade, then install holoscan and dependencies.
# The HSDK5 wheels come from PyPI, pinned to the HOLOLINK_HSDK_VERSION release.
RUN --mount=type=cache,id=hololink-${CONTAINER_TYPE}-${TARGETARCH}-pip,target=/root/.cache/pip \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3 get-pip.py && rm -f get-pip.py \
    && python3 -m pip install --upgrade pip \
    && pip3 install meson ninja pyelftools numpy \
    # holoscan-cu${CUDA_VERSION_MAJOR} is the metapackage for the four module wheels; this image takes
    # the three without inference (holoscan-inference-cu${CUDA_VERSION_MAJOR} is added by Dockerfile.demo).
    && pip3 install \
        "holoscan-core-cu${CUDA_VERSION_MAJOR}==${HOLOLINK_HSDK_VERSION}" \
        "holoscan-vision-cu${CUDA_VERSION_MAJOR}==${HOLOLINK_HSDK_VERSION}" \
        "holoscan-visualization-cu${CUDA_VERSION_MAJOR}==${HOLOLINK_HSDK_VERSION}"

# Install the HSDK debian packages. HSDK5 ships modular debian packages
# (holoscan-{core,vision,visualization,inference}-cuda-${CUDA_VERSION_MAJOR}); "holoscan" is the
# metapackage that pulls all of them (the local repo only carries the CUDA 13 variants). This
# image takes core, vision and visualization; holoscan-inference, whose deb hard-depends on
# TensorRT (libnvinfer10, ~2.9 GB), is installed by Dockerfile.demo into hololink-demo only.
# They install into the standard system prefix (/usr/lib/<multiarch>, /usr/include,
# /usr/lib/<multiarch>/cmake/holoscan), so no ld.so.conf entry is needed; CV-CUDA is bundled
# with libholoscan-vision5-cuda-${CUDA_VERSION_MAJOR}.
# The public local APT repository (developer.download.nvidia.com local_installers) is set up,
# used and purged in this one layer so its ~130 MB pool of .debs does not stay in the image.
# Its debs carry a build segment behind the release (5.0.0 ships 5.0.0.8.0-1), hence the
# ${HOLOLINK_HSDK_VERSION}.* pins: whatever build the release's repository carries, and
# nothing from a different release.
RUN --mount=type=cache,id=hololink-${CONTAINER_TYPE}-${TARGETARCH}-apt-archives,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=hololink-${CONTAINER_TYPE}-${TARGETARCH}-apt-lists,target=/var/lib/apt,sharing=locked \
    UBUNTU_VERSION=$(cat /etc/os-release | grep "VERSION_ID" | awk -F'=' '{print $2}' | sed 's/[."]//g') \
    && HSDK_LOCAL_REPO="holoscan-local-repo-ubuntu${UBUNTU_VERSION}-${HOLOLINK_HSDK_VERSION}" \
    && HSDK_LOCAL_REPO_URL="https://developer.download.nvidia.com/compute/holoscan/${HOLOLINK_HSDK_VERSION}/local_installers/${HSDK_LOCAL_REPO}_${HOLOLINK_HSDK_VERSION}-1_${TARGETARCH}.deb" \
    && echo "Fetching HSDK local APT repository: ${HSDK_LOCAL_REPO_URL}" \
    && curl -fsSL -O "${HSDK_LOCAL_REPO_URL}" \
    && dpkg -i ${HSDK_LOCAL_REPO}_${HOLOLINK_HSDK_VERSION}-1_${TARGETARCH}.deb \
    && rm -f ${HSDK_LOCAL_REPO}_${HOLOLINK_HSDK_VERSION}-1_${TARGETARCH}.deb \
    && cp /var/${HSDK_LOCAL_REPO}/holoscan-*-keyring.gpg /usr/share/keyrings/ \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y --allow-downgrades \
        "holoscan-core=${HOLOLINK_HSDK_VERSION}.*" \
        "holoscan-vision=${HOLOLINK_HSDK_VERSION}.*" \
        "holoscan-visualization=${HOLOLINK_HSDK_VERSION}.*" \
    && apt-get purge -y ${HSDK_LOCAL_REPO} \
    && rm -rf /var/${HSDK_LOCAL_REPO} /etc/apt/sources.list.d/holoscan-local-*.list \
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
ARG L4T_VERSION=invalid
