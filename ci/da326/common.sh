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

# Shared scaffolding for the DA326 player CI stages, sourced by the per-test
# wrapper scripts in this directory.

set -o errexit
set -o xtrace

HERE=$(dirname "$(realpath "$0")")
ROOT=$(realpath "$HERE/../..")
VERSION=$(cat "$ROOT/VERSION")

DA326_MAC="A0:24:90:B0:00:01"
DA326_IP="192.168.0.12"
DAEMON="set-ip-da326"

BUILD_DIR=$(mktemp -d)
cleanup() {
    docker stop "$DAEMON" 2>/dev/null || true
    docker run --rm \
        -v "$BUILD_DIR:$BUILD_DIR" \
        "hololink-demo:$VERSION" \
        find "$BUILD_DIR" -mindepth 1 -delete
    rmdir "$BUILD_DIR"
}
trap cleanup EXIT

_run() {
    docker run --rm --net host --gpus all \
        -v "$ROOT:$ROOT:ro" \
        -v "$BUILD_DIR:$BUILD_DIR" \
        -w "$ROOT" \
        "hololink-demo:$VERSION" \
        "$@"
}

# Build hololink-set-ip and launch it as a daemon that holds the DA326 at
# DA326_IP for the duration of the test; fail fast if it doesn't come up.
da326_start_set_ip() {
    _run cmake -S . -B "$BUILD_DIR" -G Ninja -DHOLOLINK_BUILD_ONLY_NATIVE=ON
    _run cmake --build "$BUILD_DIR" --target hololink-set-ip

    docker rm -f "$DAEMON" 2>/dev/null || true
    docker run --detach --rm --name "$DAEMON" \
        --net host --gpus all --cap-add NET_ADMIN \
        -v "$ROOT:$ROOT:ro" \
        -v "$BUILD_DIR:$BUILD_DIR" \
        "hololink-demo:$VERSION" \
        "$BUILD_DIR/tools/set_ip/hololink-set-ip" "$DA326_MAC" "$DA326_IP"

    sleep 1
    if ! docker inspect --format='{{.State.Status}}' "$DAEMON" 2>/dev/null | grep -q running; then
        echo "set-ip daemon failed to start or exited immediately"
        exit 1
    fi
}

# Run a player in the CI container against the DA326. Pass the interpreter and
# script plus any player-specific flags; the common --frame-limit/--hololink
# flags are appended.  The player runs windowed under a throwaway Xvfb (via
# "xvfb-run -a") instead of --headless, to avoid the AGX Thor (L4T R39) Vulkan
# instance leak documented in ci/unit_test.sh.
da326_player() {
    sh "$ROOT/ci/ci.sh" --dgpu \
        sh -c 'PYTHONUNBUFFERED=1 xvfb-run -a "$@"' xvfb-wrap \
        "$@" \
        --frame-limit 300 \
        --hololink "$DA326_IP"
}
