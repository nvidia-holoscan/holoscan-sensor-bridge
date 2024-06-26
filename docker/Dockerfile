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

ARG CONTAINER_TYPE=dgpu
FROM nvcr.io/nvidia/clara-holoscan/holoscan:v1.0.3-${CONTAINER_TYPE}

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
