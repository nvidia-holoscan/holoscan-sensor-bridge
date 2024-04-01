#!/bin/sh

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
set -o xtrace

SCRIPT=`realpath "$0"`
HERE=`dirname "$SCRIPT"`
ROOT=`cd $HERE && git rev-parse --show-toplevel`
VERSION=`cat $ROOT/VERSION`

DEFAULT_COMMAND="make -C user_guide clean html"
CONTAINER_NAME="docs"

while [ $# -ge 1 ]
do
case "$1" in
    "--interactive")
        INTERACTIVE="-ti"
        DEFAULT_COMMAND="/bin/bash -l"
        ;;
    "--container-name")
        shift
        CONTAINER_NAME="$1"
        ;;
    *)
        break
        ;;
esac
shift
done

# Build the container with tools for documentation generation.
docker build \
    --network=host \
    --build-arg CONTAINER_VERSION=hololink-prototype:$VERSION \
    -t hololink-docs:$VERSION \
    -f $HERE/Dockerfile.docs \
    $HERE

# Build the documentation itself.
COMMAND=${1-$DEFAULT_COMMAND}
docker run \
    $DOCKER_OPTS \
    --rm \
    --net host \
    --name "$CONTAINER_NAME" \
    --user $(id -u):$(id -g) \
    $INTERACTIVE \
    -v $ROOT:$ROOT \
    -w $HERE \
    hololink-docs:$VERSION \
    $COMMAND
