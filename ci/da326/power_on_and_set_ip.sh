#!/bin/sh

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -o xtrace
set -o errexit

SCRIPT=$(realpath "$0")
HERE=$(dirname "$SCRIPT")
ROOT=$(realpath "$HERE/../..")
VERSION=$(cat "$ROOT/VERSION")

DA326_MAC="A0:24:90:B0:00:01"
DA326_IP="192.168.0.12"

if [ "$1" = "go" ]; then
    # Runs inside Docker container
    SET_IP="$2"
    ENUMERATE="$3"

    "$SET_IP" "$DA326_MAC" "$DA326_IP" &
    SET_IP_PID=$!
    sleep 1
    if ! kill -0 $SET_IP_PID 2>/dev/null; then
        echo "set-ip failed to start or exited immediately"
        exit 1
    fi

    I=0
    while [ $I -lt 12 ]; do
        ENUM=$("$ENUMERATE" --timeout=5 2>/dev/null) || true
        if echo "$ENUM" | grep -q "mac_id=$DA326_MAC" && echo "$ENUM" | grep -q "ip_address=$DA326_IP"; then
            kill $SET_IP_PID 2>/dev/null || true
            echo "DA326 configured: mac_id=$DA326_MAC ip_address=$DA326_IP"
            exit 0
        fi
        I=$(( I + 1 ))
    done

    kill $SET_IP_PID 2>/dev/null || true
    echo "Validation failed: DA326 not found at mac_id=$DA326_MAC ip_address=$DA326_IP"
    exit 1
fi

# Build Docker image if not already present
if ! docker image inspect "hololink-demo:$VERSION" > /dev/null 2>&1; then
    sh "$HERE/../build.sh" --dgpu
fi

# Temp build directory
BUILD_DIR=$(mktemp -d)
cleanup() {
    docker run --rm \
        -v "$BUILD_DIR:$BUILD_DIR" \
        "hololink-demo:$VERSION" \
        find "$BUILD_DIR" -mindepth 1 -delete
    rmdir "$BUILD_DIR"
}
trap cleanup EXIT

run() {
    docker run \
        --rm \
        --net host \
        --gpus all \
        --cap-add NET_ADMIN \
        -v "$ROOT:$ROOT:ro" \
        -v "$BUILD_DIR:$BUILD_DIR" \
        -w "$ROOT" \
        "hololink-demo:$VERSION" \
        "$@"
}

run cmake -S . -B "$BUILD_DIR" -G Ninja -DHOLOLINK_BUILD_ONLY_NATIVE=ON
run cmake --build "$BUILD_DIR" --target hololink-set-ip hololink-enumerate

SET_IP="$BUILD_DIR/tools/set_ip/hololink-set-ip"
ENUMERATE="$BUILD_DIR/tools/enumerate/hololink-enumerate"

# Power on DA326
sh "$HERE/on.sh"

# Run set-ip and enumerate loop inside a single container session
if ! run sh "$SCRIPT" go "$SET_IP" "$ENUMERATE"; then
    sh "$HERE/off.sh"
    exit 1
fi

