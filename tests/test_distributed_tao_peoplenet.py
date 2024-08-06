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

# See README.md for detailed information.

import sys
from os.path import exists
from unittest import mock
from urllib.request import urlretrieve

import pytest

import hololink as hololink_module
from examples import distributed_tao_peoplenet


@pytest.mark.skip(reason="Disabled, see https://nvbugspro.nvidia.com/bug/4472408")
@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_3840X2160_60FPS,
    ],
)
def test_distributed_tao_peoplenet(
    camera_mode, headless, frame_limit, hololink_address, ibv_name, ibv_port, capfd
):
    # Download the PeopleNet ONNX model
    file_name = "examples/resnet34_peoplenet_int8.onnx"
    if not exists(file_name):
        url = "https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet/pruned_quantized_decrypted_v2.3.3/files?redirect=true&path=resnet34_peoplenet_int8.onnx"
        urlretrieve(url, file_name)

    arguments = [
        sys.argv[0],
        "--camera-mode",
        str(camera_mode.value),
        "--frame-limit",
        str(frame_limit),
        "--hololink",
        hololink_address,
        "--ibv-name",
        ibv_name,
        "--ibv-port",
        str(ibv_port),
    ]
    if headless:
        arguments.extend(["--headless"])

    with mock.patch("sys.argv", arguments):
        distributed_tao_peoplenet.main()

        # check for errors
        captured = capfd.readouterr()
        assert "[error]" not in captured.err
