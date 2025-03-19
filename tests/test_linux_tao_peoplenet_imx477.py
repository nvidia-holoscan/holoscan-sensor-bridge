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

# See README.md for detailed information.

import sys
from os.path import exists
from unittest import mock
from urllib.request import urlretrieve

import pytest

from examples import linux_tao_peoplenet_imx477


@pytest.mark.skip_unless_imx477
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [
        [3840, 2160],
    ],
)
def test_linux_tao_peoplenet(
    camera_mode, headless, frame_limit, hololink_address, capsys
):
    # Download the PeopleNet ONNX model
    file_name = "examples/resnet34_peoplenet_int8.onnx"
    if not exists(file_name):
        url = "https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet/pruned_quantized_decrypted_v2.3.3/files?redirect=true&path=resnet34_peoplenet_int8.onnx"
        urlretrieve(url, file_name)

    arguments = [
        sys.argv[0],
        "--frame-limit",
        str(frame_limit),
        "--cam",
        str(0),
    ]
    if headless:
        arguments.extend(["--headless"])

    with mock.patch("sys.argv", arguments):
        linux_tao_peoplenet_imx477.main()

        # check for errors
        captured = capsys.readouterr()
        assert captured.err == ""
