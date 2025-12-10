# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

ARG CONTAINER_TYPE=invalid
ARG HSDK_CONTAINER_CONFIG=invalid

# dgpu specific configuration goes here.
FROM nvcr.io/nvidia/clara-holoscan/holoscan:${HSDK_CONTAINER_CONFIG} AS build_dgpu

# generic l4t igpu (Orin/NX/Nano/Thor) specific configuration goes here.
FROM nvcr.io/nvidia/clara-holoscan/holoscan:${HSDK_CONTAINER_CONFIG} AS build_igpu
RUN \
    # EGL and libGLESv2 is needed by Argus api otherwise the library throws error \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y -q libgles2-mesa-dev && \
    echo "deb https://repo.download.nvidia.com/jetson/common r36.3 main" > /etc/apt/sources.list.d/nvidia-l4t-apt-source.list && \
    echo "deb https://repo.download.nvidia.com/jetson/t234 r36.3 main" >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list && \
    wget -O - https://repo.download.nvidia.com/jetson/jetson-ota-public.asc | apt-key add - && \
    apt-get update && \
    # we do not have the permission to ./mm-api/DEBIAN/postinst \
    # to just install nvidia-l4t-jetson-multimedia-api via apt install \
    apt-get download nvidia-l4t-jetson-multimedia-api && \
    dpkg-deb -R ./nvidia-l4t-jetson-multimedia-api_*_arm64.deb ./mm-api && \
    cp -r ./mm-api/usr/src/jetson_multimedia_api /usr/src/jetson_multimedia_api && \
    sed -i 's/sudo//' ./mm-api/DEBIAN/postinst && \
    ./mm-api/DEBIAN/postinst && \
    rm -rf ./nvidia-l4t-jetson-multimedia-api_*_arm64.deb ./mm-api && \
    rm -rf /var/lib/apt/lists/*

COPY --from=l4t-libs \
    libnvargus.so \
    libnvargus_socketclient.so \
    libnvargus_socketserver.so \
    libnvfusacap.so \
    libnvodm_imager.so \
    libnvscf.so \
    /usr/lib/aarch64-linux-gnu/nvidia/
ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:$LD_LIBRARY_PATH

# Thor specific configuration goes here.
FROM nvcr.io/nvidia/clara-holoscan/holoscan:${HSDK_CONTAINER_CONFIG} AS build_igpu_sipl
ARG L4T_VERSION=invalid
ENV L4T_MAJ_VER="${L4T_VERSION%%.*}"
ENV L4T_MIN_VER="${L4T_VERSION#*.}"

# Copy the Jetson Multimedia API from the host (fall back to download if not installed on the host).
COPY --from=l4t-src jetson_multimedia_api* /usr/src/jetson_multimedia_api
RUN if [ ! -d "/usr/src/jetson_multimedia_api" ]; then \
        wget https://developer.nvidia.com/downloads/embedded/L4T/r${L4T_MAJ_VER}_Release_v${L4T_MIN_VER}/release/Jetson_Multimedia_API_R${L4T_VERSION}_aarch64.tbz2 && \
        tar xjf Jetson_Multimedia_API_R${L4T_VERSION}_aarch64.tbz2 -C / && \
        rm Jetson_Multimedia_API_R${L4T_VERSION}_aarch64.tbz2; \
    fi

# Copy the Jetson SIPL API from the host (fall back to download if not installed on the host).
COPY --from=l4t-src jetson_sipl_api* /usr/src/jetson_sipl_api
RUN if [ ! -d "/usr/src/jetson_sipl_api" ]; then \
        wget https://developer.nvidia.com/downloads/embedded/L4T/r${L4T_MAJ_VER}_Release_v${L4T_MIN_VER}/release/Jetson_SIPL_API_R${L4T_VERSION}_aarch64.tbz2 && \
        tar xjf Jetson_SIPL_API_R${L4T_VERSION}_aarch64.tbz2 -C / && \
        rm Jetson_SIPL_API_R${L4T_VERSION}_aarch64.tbz2; \
    fi

# Build and install the UDDF drivers.
RUN cd /usr/src/jetson_sipl_api/sipl && \
    rm -rf build && mkdir build && cd build && \
    cmake .. && \
    make -j && \
    make install

# EGL and libGLESv2 are needed by Argus api otherwise the library throws error.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y -q libgles2-mesa-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy the L4T libraries into the container.
COPY --from=l4t-libs \
    libgnat-24.so \
    libnvargus.so \
    libnvargus_socketclient.so \
    libnvargus_socketserver.so \
    libnvbufsurface.so \
    libnvbuf_fdmap.so.1.0.0 \
    libnvcamerahal.so \
    libnvfusacapinterface.so \
    libnvfusacap*.so \
    libnvisppg.so \
    libnvisp.so \
    libnvodm_imager.so \
    libnvos.so \
    libnvrm_chip.so \
    libnvrm_gpu.so \
    libnvrm_host1x.so \
    libnvrm_mem.so \
    libnvrm_surface.so \
    libnvrm_sync.so \
    libnvscf.so \
    libnvscibuf.so \
    libnvscicommon.so.1 \
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
ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:$LD_LIBRARY_PATH

# DGX Spark specific configuration goes here.
FROM nvcr.io/nvidia/clara-holoscan/holoscan:${HSDK_CONTAINER_CONFIG} AS build_igpu_spark

# configuration for all GPU configurations start here.
FROM build_${CONTAINER_TYPE}

RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y -q \
    python3 \
    python3-dev \
    python3-pip \
  && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y -q \
    wget \
    pkg-config \
    libibverbs-dev \
    libnuma-dev \
    libelf-dev \
    libfmt-dev \
    libyaml-cpp-dev \
    libcurlpp-dev \
  && rm -rf /var/lib/apt/lists/*

RUN pip3 install meson ninja pyelftools
# repeated ARG because it has to be inside the FROM above
ARG CONTAINER_TYPE=invalid
ARG HSDK_CONTAINER_CONFIG=invalid
ENV CONTAINER_TYPE=${CONTAINER_TYPE}
