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
# See docs/user_guide/README.md for detailed information.
#
# Usage:
#   docs/make_docs.sh [--preview|--dev] [--no-docker] [--with-library-mdx] ...
#   DOCKER_OPTS='-e MY_VAR=1' docs/make_docs.sh   # extra docker run flags
#
# DOCKER_OPTS: optional extra flags passed to `docker run` (unset = none).

set -o errexit

usage() {
    cat <<EOF
Usage: docs/make_docs.sh [options]

Build and validate Fern docs (default). Preview only with --preview.

Options:
  --preview, --dev       Run fern docs dev (local preview; uses the docs container by default)
  --login                Log in to Fern (device-code in Docker; browser on host)
  --no-docker            Run on the host instead of the docs container
  --docker               Use the docs container (default)
  --with-library-mdx     Generate the Emulation C++ API MDX (requires Fern auth)
  --skip-library-mdx     Skip C++ API generation
  --skip-fern-check      Pipeline only; skip fern check
  --publish              Publish docs to production (docs.nvidia.com)
  --publish-preview      Publish a remote Fern docs preview (CI)
  --preview-id ID        Preview id for --publish-preview
  --delete-preview-id ID Delete a remote preview before publishing
  --force                Skip overwrite confirmation for remote preview publish (CI only)
  --port PORT            Preview server port (default: 3000; backend uses PORT+1)
  --container-name NAME  Docker container name (default: docs)
  --help                 Show this help

Environment:
  DOCKER_OPTS            Extra flags for docker run (optional; default: empty).
  FERN_TOKEN             Non-expiring Fern org API token from 'fern token', for
                         C++ API generation and publishing (optional). The 'fern login' credential in
                         ~/.fern is shared into the container automatically and
                         is NOT used as FERN_TOKEN.
EOF
}

SCRIPT=`realpath "$0"`
HERE=`dirname "$SCRIPT"`
ROOT=`cd $HERE && git rev-parse --show-toplevel`

ARGS=""
CONTAINER_NAME="docs"

while [ $# -ge 1 ]; do
    case "$1" in
        --preview|--dev)
            ARGS="$ARGS --preview"
            ;;
        --login)
            ARGS="$ARGS --login"
            ;;
        --no-docker)
            ARGS="$ARGS --no-docker"
            ;;
        --docker)
            ;;
        --with-library-mdx)
            ARGS="$ARGS --with-library-mdx"
            ;;
        --skip-library-mdx)
            ARGS="$ARGS --skip-library-mdx"
            ;;
        --skip-fern-check)
            ARGS="$ARGS --skip-fern-check"
            ;;
        --publish)
            ARGS="$ARGS --publish"
            ;;
        --publish-preview)
            ARGS="$ARGS --publish-preview"
            ;;
        --preview-id)
            shift
            ARGS="$ARGS --preview-id $1"
            ;;
        --delete-preview-id)
            shift
            ARGS="$ARGS --delete-preview-id $1"
            ;;
        --force)
            ARGS="$ARGS --force"
            ;;
        --port)
            shift
            ARGS="$ARGS --port $1"
            ;;
        --container-name)
            shift
            CONTAINER_NAME="$1"
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
    shift
done

# Do NOT copy ~/.fern/token into FERN_TOKEN. That file holds the `fern login` OAuth
# credential (identity only; it may lack org permissions), whereas FERN_TOKEN must be a
# non-expiring org API token from `fern token`. Injecting the login token as FERN_TOKEN
# makes Fern reject it (FERN_TOKEN is read first) and fall back to an interactive login
# prompt. The cached login credential is shared into the container via the mounted
# ~/.fern instead; set FERN_TOKEN yourself only when you have a real org token.

DOCKER_OPTS="${DOCKER_OPTS:-}"
if [ -n "$DOCKER_OPTS" ]; then
    echo "warning: DOCKER_OPTS is not used by build_hololink_docs.py; pass --no-docker and run natively if needed." >&2
fi

exec python3 "$HERE/scripts/build_hololink_docs.py" --container-name "$CONTAINER_NAME" $ARGS
