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
#
# See README.md for detailed information.

# Usage:
#   ci/stress.sh [--count N] [--inside|--around] <command> [args...]
#
# Repeats <command> N times to surface flaky failures and performance variance.
#
#   --inside  (default)  one container, N runs.  Fast: avoids the per-iteration
#                        image check + `docker run` startup cost.  <command> must
#                        honor $STRESS_ITERATIONS (unit_test.sh,
#                        imx274_ptp_test.sh, performance.sh do).
#   --around             a fresh invocation (new container) per iteration.  Full
#                        isolation; works with ANY command, including the
#                        destructive hsb_flasher.sh.
#
# Exits nonzero if any iteration failed.
#
# Examples:
#   ci/stress.sh --count 25 ci/unit_test.sh --dgpu
#   ci/stress.sh --count 5 --around ci/hsb_flasher.sh --dgpu

set -o errexit
set -o xtrace

SCRIPT=`realpath "$0"`
HERE=`dirname "$SCRIPT"`

COUNT=10
MODE=inside
while [ $# -ge 1 ]
do
case "$1" in
  "--count")
    if [ "$#" -lt 2 ]; then
      echo "$0: --count requires a value" >&2
      exit 1
    fi
    COUNT="$2"
    shift
    ;;
  --count=*)
    COUNT="${1#--count=}"
    ;;
  "--inside")
    MODE=inside
    ;;
  "--around")
    MODE=around
    ;;
  *)
    break
    ;;
esac
shift
done

case "$COUNT" in
  ''|*[!0-9]*)
    echo "$0: --count must be a non-negative integer, got '$COUNT'" >&2
    exit 1
    ;;
esac

if [ $# -lt 1 ]
then
echo "Usage: $0 [--count N] [--inside|--around] <command> [args...]" >&2
exit 1
fi

if [ "$MODE" = "around" ]
then
    # The outer loop drives the count; force each child container to run exactly
    # once so we get N total runs, not N*N.
    . "$HERE/stress_lib.sh"
    export STRESS_ITERATIONS=1
    stress_run -n "$COUNT" "$@"
else
    # Inner loop: a single container whose go-branch repeats N times.
    STRESS_ITERATIONS="$COUNT" "$@"
fi
