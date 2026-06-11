# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Validate hololink/metadata.json against the Holoscan module JSON schema.

Requires holoscan-cli from the main branch:
    pip install git+https://github.com/nvidia-holoscan/holoscan-cli.git
"""

import json
from pathlib import Path

import pytest

metadata_validator = pytest.importorskip(
    "holoscan_cli.metadata.metadata_validator",
    reason=(
        "holoscan_cli.metadata not available; install with: "
        "pip install git+https://github.com/nvidia-holoscan/holoscan-cli.git"
    ),
)

METADATA_PATH = Path(__file__).resolve().parent.parent / "metadata.json"


def test_metadata_json_validates_against_module_schema():
    with open(METADATA_PATH) as f:
        data = json.load(f)
    valid, msg = metadata_validator.validate_json(data, METADATA_PATH.parent)
    assert valid, f"metadata.json failed schema validation:\n{msg}"
