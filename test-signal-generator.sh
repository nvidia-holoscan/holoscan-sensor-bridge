#!/usr/bin/env bash
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

SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR"

shopt -s nullglob
LC_COLLATE=C INFINIBANDS=(/sys/class/infiniband/*)
shopt -u nullglob
if [ ${#INFINIBANDS[@]} -lt 2 ]; then
    echo "Error: at least 2 InfiniBand devices required, found ${#INFINIBANDS[@]}" >&2
    exit 1
fi
TX_INTERFACE=$(basename "${INFINIBANDS[1]}")
RX_INTERFACE=$(basename "${INFINIBANDS[0]}")

pytest tests/test_signal_generator.py \
    --signal-generator \
    --tx-hololink=192.168.0.3 \
    --hololink=192.168.0.2 \
    --tx-ibv-name="$TX_INTERFACE" \
    --ibv-name="$RX_INTERFACE"
