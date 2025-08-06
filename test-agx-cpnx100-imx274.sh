#!/bin/bash

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
#

set -o errexit

#
# This script runs `pytest` with the switches appropriate for an AGX
# host connected to a Lattice CPNX100-ETH-SENSOR-BRIDGE with an IMX274 attached;
# the network is connected from the first HSB port to the on-board ethernet.
#
# This test only runs the non-network-accelerated tests and only works with
# the first (192.168.0.2) HSB interface.
#
pytest --imx274 --unaccelerated-only --channel-ips=192.168.0.2 --schedulers=default --ptp
