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

ARG CONTAINER_TYPE=dgpu
FROM nvcr.io/nvidia/clara-holoscan/holoscan:v2.1.0-${CONTAINER_TYPE}

ARG CONTAINER_TYPE

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
  && rm -rf /var/lib/apt/lists/*

RUN pip3 install meson ninja pyelftools

RUN if [ "$CONTAINER_TYPE" = "igpu" ]; then \
    # libgtk is needed by Argus api otherwise the library throughs error \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y -q libgtk-3-dev && \
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
    rm -rf ./nvidia-l4t-jetson-multimedia-api_*_arm64.deb ./mm-api; \
  fi

ADD nvidia /usr/lib/aarch64-linux-gnu/nvidia

ENV CONTAINER_TYPE=$CONTAINER_TYPE
ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:$LD_LIBRARY_PATH
