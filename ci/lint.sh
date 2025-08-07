#!/bin/bash

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

set -o errexit

# let's make sure we're running under
# bash; our array variables depend on it.
if [ "$BASH" = "" ]
then
exec bash $0 $*
fi

# Usage:
#   lint.sh [--lint]
# or
#   lint.sh --format
#
# This script builds and invokes a container called 'hololink-lint'
# with the tools used for formatting the project source code.  When
# run without a switch, or with "--lint", the source files in the
# project are tested against the project formatting rules, terminating
# with a non-zero error code if any violations are found.
#
# When run with "--format", then source files are run through formatting
# rules and updated accordingly.  This is usually all that's necessary
# to get a subsequent "lint.sh --lint" to succeed.
#

SCRIPT=`realpath "$0"`
HERE=`dirname "$SCRIPT"`
ROOT=`cd $HERE && git rev-parse --show-toplevel`
VERSION=`cat $ROOT/VERSION`

cd $ROOT

SKIP=( \
    docs
    python/hololink/renesas_bajoran_lite_ts1.py
    # black: Cannot parse: 18:7: from ._@MODULE_NAME@ import @MODULE_CLASS_NAME@
    cmake/pybind11/__init__.py
    scripts
    build*
)

# Codespell checks for spelling mistakes.  Don't include PDF files.
CODESPELL_FILES=( `git ls-files | grep -v '.pdf$'` )

C_FILES=( `git ls-files | egrep '.(cpp|hpp)$' ` )
DOCS_FILES=( `git ls-files | egrep '.md$' ` )

# Each command likes its list of inputs slightly different
T=${SKIP[*]}
SKIP_RE="(${T// /|})"
SKIP_COMMAS=${T// /,}
SKIP_ISORT=${SKIP[*]/#/--skip }

# The 88 line length is chosen based on the default
# value used by black.
MDFORMAT="--wrap 88 --end-of-line lf"

# Allow some flake8 violations
# E203 - black puts spaces around the ':' in array slices
# E501 - black uses 88 character lengths; flake8 complains for lines longer than 79
# F824 - Allow the code to say 'global x' then only read from that variable; this
#   helps readers of your code.
# W503 - black formats long expressions with the operator at the beginning
#   of the line
FLAKE8="--ignore=E203,E501,F824,W503"

umask 0

# By default, run lint; specify "--format" to
# rewrite source files with automatic formatting.
MODE="lint"

case "$1" in
    --do-format)
        # We rely on 'set -o errexit' above to terminate on an error
        set -o xtrace
        isort $SKIP_ISORT --profile black .
        black "--extend-exclude=$SKIP_RE" .
        flake8 $FLAKE8 --extend-exclude=$SKIP_COMMAS
        clang-format --style=file:${ROOT}/.clang-format -i ${C_FILES[*]}
        mdformat $MDFORMAT ${DOCS_FILES[*]}
        codespell --ignore-words=$HERE/acceptable_words.txt ${CODESPELL_FILES[*]}
        exit 0
        ;;
    --do-lint)
        # We rely on 'set -o errexit' above to terminate on an error
        set -o xtrace
        isort --check-only $SKIP_ISORT --profile black .
        black --check "--extend-exclude=$SKIP_RE" .
        flake8 $FLAKE8 --extend-exclude=$SKIP_COMMAS
        clang-format --style=file:${ROOT}/.clang-format --dry-run -Werror ${C_FILES[*]}
        mdformat --check $MDFORMAT ${DOCS_FILES[*]}
        codespell --ignore-words=$HERE/acceptable_words.txt ${CODESPELL_FILES[*]}
        exit 0
        ;;
    --do-*)
        # Invalid request.
        echo "Unexpected command \"$1\"; aborting." >&2
        exit 1
        ;;
    --*)
        # Without "--do-format" or "--do-lint" on the command-line,
        # start the CI container and run us again with "--do-<whatever>".
        MODE="${1#--}"
        ;;
esac

# We only get here if we weren't "--do-lint" or "--do-format"; we're going to
# "--do-$MODE".
# Also, we specifically rely on buildkit skipping the dgpu or igpu stages that
# aren't included in the final image we're creating.
DOCKER_BUILDKIT=1 docker build \
    -t hololink-lint:$VERSION \
    -f $HERE/Dockerfile.lint \
    $ROOT

USER=`id -u`:`id -g`
docker run \
    --rm \
    --user $USER \
    -v $ROOT:$ROOT \
    -w $ROOT \
    hololink-lint:$VERSION \
    $0 --do-$MODE
