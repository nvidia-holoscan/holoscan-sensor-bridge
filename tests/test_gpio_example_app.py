# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# See README.md for detailed information.

import sys
from unittest import mock

import pytest

from examples import gpio_example_app


@pytest.mark.skip_unless_hsb_nano
def test_gpio_example_app(frame_limit, capsys):
    arguments = [
        sys.argv[0],
        "--cycle-limit",
        str(frame_limit),
        "--sleep-time",
        "0",
    ]

    with mock.patch("sys.argv", arguments):
        gpio_example_app.main()

        # check for errors
        captured = capsys.readouterr()
        assert captured.err == ""
