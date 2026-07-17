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

# Shared stress-loop helper.  Source this file, then call:
#     stress_run [-n N] <command> [args...]
#
# The iteration count comes from "-n N", else from $STRESS_ITERATIONS, else 1.
# With a count of 1 the command runs exactly once and its exit status is
# propagated unchanged, so normal CI is byte-for-byte identical.  With N>1 the
# command runs N times; every iteration runs even if one fails (the status is
# captured so the caller's `set -o errexit` does not abort the loop); a
# per-iteration line plus a summary are printed; and the function returns
# nonzero if any iteration failed.

stress_run() {
    iterations="${STRESS_ITERATIONS:-1}"
    if [ "$1" = "-n" ]; then
        if [ "$#" -lt 2 ]; then
            echo "stress_run: -n requires an iteration count" >&2
            return 2
        fi
        iterations="$2"
        shift 2
    fi

    # Reject non-digits before any numeric test, else `[ ... -le ... ]` errors
    # out and the loop is silently skipped.
    case "$iterations" in
        ''|*[!0-9]*)
            echo "stress_run: iteration count must be a positive integer, got '$iterations'" >&2
            return 2
            ;;
    esac

    if [ "$#" -lt 1 ]; then
        echo "stress_run: no command given" >&2
        return 2
    fi

    # Fast path: exact legacy behavior, no loop, no extra output.
    if [ "$iterations" -le 1 ]; then
        "$@"
        return $?
    fi

    passed=0
    failed=0
    i=1
    while [ "$i" -le "$iterations" ]; do
        echo "===== STRESS iteration $i/$iterations ($passed passed, $failed failed so far): $* ====="
        # Testing the status as an `if` condition suppresses `errexit`, so a
        # failing iteration is counted rather than aborting the loop.
        if "$@"; then
            passed=$((passed + 1))
        else
            failed=$((failed + 1))
            echo "===== STRESS iteration $i/$iterations FAILED ====="
        fi
        i=$((i + 1))
    done

    echo "===== STRESS SUMMARY: $passed passed, $failed failed of $iterations ====="
    [ "$failed" -eq 0 ]
}
