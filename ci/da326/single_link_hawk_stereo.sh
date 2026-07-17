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

. "$(dirname "$(realpath "$0")")/common.sh"

da326_start_set_ip

if ! OUT=$(da326_player python3 examples/linux_single_network_stereo_hawk_player.py --channel B 2>&1); then
    echo "$OUT"
    exit 1
fi
echo "$OUT"

AVERAGE=$(echo "$OUT" | grep "Stereo skew:" | tail -1 | sed 's/.*average=\([0-9.]*\) us.*/\1/')
if [ -z "$AVERAGE" ]; then
    echo "FAIL: No stereo skew output found"
    exit 1
fi
echo "Stereo skew average: ${AVERAGE} us"
if awk "BEGIN { exit ($AVERAGE < 100) ? 0 : 1 }"; then
    echo "PASS: Stereo skew average=${AVERAGE} us < 100 us"
else
    echo "FAIL: Stereo skew average=${AVERAGE} us >= 100 us"
    exit 1
fi
