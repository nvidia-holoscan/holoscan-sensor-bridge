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

import applications
import pytest

import hololink as hololink_module


@pytest.mark.skip_unless_imx274
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
    ],
)
def test_imx274_mono_linux_naive(
    camera_mode, headless, frame_limit, hololink_address, ptp_enable
):
    mono_test = applications.MonoTest(
        camera_mode,
        headless,
        frame_limit,
        hololink_address,
        receiver_factory=applications.linux_receiver_factory,
        isp_factory=applications.naive_isp,
        camera_factory=lambda channel, instance: applications.imx274_camera_factory(
            channel, instance, camera_mode, 10
        ),
        ptp_enable=ptp_enable,
    )
    mono_test.execute()


@pytest.mark.skip_unless_imx274
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
    ],
)
def test_imx274_stereo_linux_naive(
    camera_mode,
    headless,
    frame_limit,
    hololink_address,
    ptp_enable,
):
    stereo_test = applications.StereoTest(
        camera_mode,
        headless,
        frame_limit,
        hololink_address,
        left_receiver_factory=applications.linux_receiver_factory,
        right_receiver_factory=applications.linux_receiver_factory,
        left_isp_factory=applications.naive_isp,
        right_isp_factory=applications.naive_isp,
        left_camera_factory=lambda channel, instance: applications.imx274_camera_factory(
            channel, instance, camera_mode, 10
        ),
        right_camera_factory=lambda channel, instance: applications.imx274_camera_factory(
            channel, instance, camera_mode, 11
        ),
        ptp_enable=ptp_enable,
    )
    stereo_test.execute()


@pytest.mark.skip_unless_coe
@pytest.mark.skip_unless_imx274
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
    ],
)
def test_imx274_stereo_linux_naive_coe(
    camera_mode,
    headless,
    frame_limit,
    hololink_address,
    ptp_enable,
    coe_interfaces,
):
    pixel_width = 1920
    stereo_test = applications.StereoTest(
        camera_mode,
        headless,
        frame_limit,
        hololink_address,
        left_receiver_factory=lambda *args, **kwargs: applications.linux_coe_receiver_factory(
            coe_interfaces[0], 1, pixel_width, *args, **kwargs
        ),
        right_receiver_factory=lambda *args, **kwargs: applications.linux_coe_receiver_factory(
            coe_interfaces[0], 2, pixel_width, *args, **kwargs
        ),
        left_isp_factory=applications.naive_isp,
        right_isp_factory=applications.naive_isp,
        left_camera_factory=lambda channel, instance: applications.imx274_camera_factory(
            channel, instance, camera_mode, 10
        ),
        right_camera_factory=lambda channel, instance: applications.imx274_camera_factory(
            channel, instance, camera_mode, 11
        ),
        ptp_enable=ptp_enable,
    )
    stereo_test.execute()


# NOTE THAT ARGUS ONLY WORKS FOR ONE CHANNEL.
@pytest.mark.skip_unless_igpu
@pytest.mark.skip_unless_imx274
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
    ],
)
def test_imx274_stereo_linux_argus(
    camera_mode,
    headless,
    frame_limit,
    hololink_address,
    ptp_enable,
):
    stereo_test = applications.StereoTest(
        camera_mode,
        headless,
        frame_limit,
        hololink_address,
        left_receiver_factory=applications.linux_receiver_factory,
        right_receiver_factory=applications.linux_receiver_factory,
        left_isp_factory=applications.argus_isp,
        right_isp_factory=applications.argus_isp,
        left_camera_factory=lambda channel, instance: applications.imx274_camera_factory(
            channel, instance, camera_mode, 10
        ),
        right_camera_factory=lambda channel, instance: applications.imx274_camera_factory(
            channel, instance, camera_mode, 11
        ),
        ptp_enable=ptp_enable,
    )
    stereo_test.execute()


# NOTE THAT ARGUS ONLY WORKS FOR ONE CHANNEL.
@pytest.mark.skip_unless_igpu
@pytest.mark.skip_unless_imx274
@pytest.mark.skip_unless_coe
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
    ],
)
def test_imx274_stereo_linux_argus_coe(
    camera_mode,
    headless,
    frame_limit,
    hololink_address,
    ptp_enable,
    coe_interfaces,
):
    pixel_width = 1920
    stereo_test = applications.StereoTest(
        camera_mode,
        headless,
        frame_limit,
        hololink_address,
        left_receiver_factory=lambda *args, **kwargs: applications.linux_coe_receiver_factory(
            coe_interfaces[0], 1, pixel_width, *args, **kwargs
        ),
        right_receiver_factory=lambda *args, **kwargs: applications.linux_coe_receiver_factory(
            coe_interfaces[0], 2, pixel_width, *args, **kwargs
        ),
        left_isp_factory=applications.argus_isp,
        right_isp_factory=applications.argus_isp,
        left_camera_factory=lambda channel, instance: applications.imx274_camera_factory(
            channel, instance, camera_mode, 10
        ),
        right_camera_factory=lambda channel, instance: applications.imx274_camera_factory(
            channel, instance, camera_mode, 11
        ),
        ptp_enable=ptp_enable,
    )
    stereo_test.execute()


@pytest.mark.skip_unless_igpu
@pytest.mark.skip_unless_imx274
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
    ],
)
def test_imx274_mono_linux_argus(
    camera_mode, headless, frame_limit, hololink_address, ptp_enable
):
    mono_test = applications.MonoTest(
        camera_mode,
        headless,
        frame_limit,
        hololink_address,
        receiver_factory=applications.linux_receiver_factory,
        isp_factory=applications.argus_isp,
        camera_factory=lambda channel, instance: applications.imx274_camera_factory(
            channel, instance, camera_mode, 10
        ),
        ptp_enable=ptp_enable,
    )
    mono_test.execute()


@pytest.mark.skip_unless_igpu
@pytest.mark.skip_unless_imx274
@pytest.mark.skip_unless_coe
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
    ],
)
def test_imx274_mono_linux_argus_coe(
    camera_mode,
    headless,
    frame_limit,
    hololink_address,
    ptp_enable,
    coe_interfaces,
):
    pixel_width = 1920
    mono_test = applications.MonoTest(
        camera_mode,
        headless,
        frame_limit,
        hololink_address,
        receiver_factory=lambda *args, **kwargs: applications.linux_coe_receiver_factory(
            coe_interfaces[0], 1, pixel_width, *args, **kwargs
        ),
        isp_factory=applications.argus_isp,
        camera_factory=lambda channel, instance: applications.imx274_camera_factory(
            channel, instance, camera_mode, 10
        ),
        ptp_enable=ptp_enable,
    )
    mono_test.execute()


@pytest.mark.skip_unless_vb1940
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [
        hololink_module.sensors.vb1940.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS,
    ],
)
def test_vb1940_mono_linux_naive(
    camera_mode, headless, frame_limit, hololink_address, ptp_enable
):
    mono_test = applications.MonoTest(
        camera_mode,
        headless,
        frame_limit,
        hololink_address,
        receiver_factory=applications.linux_receiver_factory,
        isp_factory=applications.naive_isp,
        camera_factory=lambda channel, instance: applications.vb1940_camera_factory(
            channel, instance, camera_mode
        ),
        ptp_enable=ptp_enable,
    )
    mono_test.execute()


@pytest.mark.skip_unless_igpu
@pytest.mark.skip_unless_vb1940
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [
        hololink_module.sensors.vb1940.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS,
    ],
)
def test_vb1940_mono_linux_argus(
    camera_mode, headless, frame_limit, hololink_address, ptp_enable
):
    mono_test = applications.MonoTest(
        camera_mode,
        headless,
        frame_limit,
        hololink_address,
        receiver_factory=applications.linux_receiver_factory,
        isp_factory=applications.argus_isp,
        camera_factory=lambda channel, instance: applications.vb1940_camera_factory(
            channel, instance, camera_mode
        ),
        ptp_enable=ptp_enable,
    )
    mono_test.execute()


@pytest.mark.skip_unless_coe
@pytest.mark.skip_unless_vb1940
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [
        hololink_module.sensors.vb1940.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS,
    ],
)
def test_vb1940_mono_linux_naive_coe(
    camera_mode,
    headless,
    frame_limit,
    hololink_address,
    ptp_enable,
    coe_interfaces,
):
    pixel_width = 2560
    mono_test = applications.MonoTest(
        camera_mode,
        headless,
        frame_limit,
        hololink_address,
        receiver_factory=lambda *args, **kwargs: applications.linux_coe_receiver_factory(
            coe_interfaces[0], 1, pixel_width, *args, **kwargs
        ),
        isp_factory=applications.naive_isp,
        camera_factory=lambda channel, instance: applications.vb1940_camera_factory(
            channel, instance, camera_mode
        ),
        ptp_enable=ptp_enable,
    )
    mono_test.execute()


@pytest.mark.skip_unless_vb1940
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [
        hololink_module.sensors.vb1940.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS,
    ],
)
def test_vb1940_stereo_linux_naive(
    camera_mode,
    headless,
    frame_limit,
    hololink_address,
    ptp_enable,
):
    stereo_test = applications.StereoTest(
        camera_mode,
        headless,
        frame_limit,
        hololink_address,
        left_receiver_factory=applications.linux_receiver_factory,
        right_receiver_factory=applications.linux_receiver_factory,
        left_isp_factory=applications.naive_isp,
        right_isp_factory=applications.naive_isp,
        left_camera_factory=lambda channel, instance: applications.vb1940_camera_factory(
            channel, instance, camera_mode
        ),
        right_camera_factory=lambda channel, instance: applications.vb1940_camera_factory(
            channel, instance, camera_mode
        ),
        ptp_enable=ptp_enable,
    )
    stereo_test.execute()


@pytest.mark.skip_unless_coe
@pytest.mark.skip_unless_vb1940
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [
        hololink_module.sensors.vb1940.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS,
    ],
)
def test_vb1940_stereo_linux_naive_coe(
    camera_mode,
    headless,
    frame_limit,
    hololink_address,
    ptp_enable,
    coe_interfaces,
):
    pixel_width = 2560
    stereo_test = applications.StereoTest(
        camera_mode,
        headless,
        frame_limit,
        hololink_address,
        left_receiver_factory=lambda *args, **kwargs: applications.linux_coe_receiver_factory(
            coe_interfaces[0], 1, pixel_width, *args, **kwargs
        ),
        right_receiver_factory=lambda *args, **kwargs: applications.linux_coe_receiver_factory(
            coe_interfaces[0], 2, pixel_width, *args, **kwargs
        ),
        left_isp_factory=applications.naive_isp,
        right_isp_factory=applications.naive_isp,
        left_camera_factory=lambda channel, instance: applications.vb1940_camera_factory(
            channel, instance, camera_mode
        ),
        right_camera_factory=lambda channel, instance: applications.vb1940_camera_factory(
            channel, instance, camera_mode
        ),
        ptp_enable=ptp_enable,
    )
    stereo_test.execute()


# NOTE THAT ARGUS ONLY WORKS FOR ONE CHANNEL.
@pytest.mark.skip_unless_coe
@pytest.mark.skip_unless_igpu
@pytest.mark.skip_unless_vb1940
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [
        hololink_module.sensors.vb1940.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS,
    ],
)
def test_vb1940_stereo_linux_argus_coe(
    camera_mode,
    headless,
    frame_limit,
    hololink_address,
    ptp_enable,
    coe_interfaces,
):
    pixel_width = 2560
    stereo_test = applications.StereoTest(
        camera_mode,
        headless,
        frame_limit,
        hololink_address,
        left_receiver_factory=lambda *args, **kwargs: applications.linux_coe_receiver_factory(
            coe_interfaces[0], 1, pixel_width, *args, **kwargs
        ),
        right_receiver_factory=lambda *args, **kwargs: applications.linux_coe_receiver_factory(
            coe_interfaces[0], 2, pixel_width, *args, **kwargs
        ),
        left_isp_factory=applications.argus_isp,
        right_isp_factory=applications.argus_isp,
        left_camera_factory=lambda channel, instance: applications.vb1940_camera_factory(
            channel, instance, camera_mode
        ),
        right_camera_factory=lambda channel, instance: applications.vb1940_camera_factory(
            channel, instance, camera_mode
        ),
        ptp_enable=ptp_enable,
    )
    stereo_test.execute()
