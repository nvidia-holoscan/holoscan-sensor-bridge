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

import argparse
import ctypes
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import holoscan
from cuda import cuda

import hololink as hololink_module


class HoloscanApplication(holoscan.core.Application):
    def __init__(
        self,
        headless,
        fullscreen,
        cuda_context,
        cuda_device_ordinal,
        hololink_channel,
        camera,
        camera_mode,
        frame_limit,
        stream_id,
    ):
        logging.info("__init__")
        super().__init__()
        self._headless = headless
        self._fullscreen = fullscreen
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._hololink_channel = hololink_channel
        self._camera = camera
        self._camera_mode = camera_mode
        self._frame_limit = frame_limit
        self._stream_id = stream_id

    def compose(self):
        logging.info("compose")
        if self._frame_limit:
            self._count = holoscan.conditions.CountCondition(
                self,
                name="count",
                count=self._frame_limit,
            )
            condition = self._count
        else:
            self._ok = holoscan.conditions.BooleanCondition(
                self, name="ok", enable_tick=True
            )
            condition = self._ok
        self._camera.set_mode(self._camera_mode)

        # image_decoder_allocator_pool = holoscan.resources.BlockMemoryPool(
        #     self,
        #     name="pool",
        #     # storage_type of 1 is device memory
        #     storage_type=1,
        #     block_size=self._camera._width
        #     * ctypes.sizeof(ctypes.c_uint16) * 2
        #     * self._camera._height,
        #     num_blocks=2,
        # )
        image_decoder_allocator_pool = holoscan.resources.UnboundedAllocator(self)

        image_decoder = hololink_module.operators.ImageDecoderOp(
            self,
            name="image_decoder",
            out_tensor_name="output",
            allocator=image_decoder_allocator_pool,
            cuda_device_ordinal=self._cuda_device_ordinal,
        )
        self._camera.configure_converter(image_decoder)

        frame_size = image_decoder.get_csi_length()
        frame_context = self._cuda_context
        receiver_operator = hololink_module.operators.LinuxReceiverOperator(
            self,
            condition,
            name="receiver",
            frame_size=frame_size,
            frame_context=frame_context,
            udp_port=54739 + self._stream_id,
            hololink_channel=self._hololink_channel,
            device=self._camera,
        )

        visualizer = holoscan.operators.HolovizOp(
            self,
            name="holoviz",
            fullscreen=self._fullscreen,
            headless=self._headless,
            framebuffer_srgb=False,
        )

        self.add_flow(receiver_operator, image_decoder, {("output", "input")})
        self.add_flow(image_decoder, visualizer, {("output", "receivers")})


def main():
    parser = argparse.ArgumentParser()
    modes = hololink_module.sensors.d555.d555_mode.RealSense_Mode
    mode_choices = [mode.value for mode in modes]
    mode_help = " ".join([f"{mode.value}:{mode.name}" for mode in modes])
    parser.add_argument(
        "--camera-mode",
        type=int,
        choices=mode_choices,
        default=mode_choices[4],
        help=mode_help,
    )
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument(
        "--fullscreen", action="store_true", help="Run in fullscreen mode"
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Exit after receiving this many frames",
    )
    default_configuration = os.path.join(
        os.path.dirname(__file__), "example_configuration.yaml"
    )
    parser.add_argument(
        "--configuration",
        default=default_configuration,
        help="Configuration file",
    )
    parser.add_argument(
        "--hololink",
        default="192.168.0.2",
        help="IP address of Hololink board",
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level to display",
    )
    parser.add_argument(
        "--expander-configuration",
        type=int,
        default=0,
        choices=(0, 1),
        help="I2C Expander configuration",
    )
    parser.add_argument(
        "--pattern",
        type=int,
        choices=range(12),
        help="Configure to display a test pattern.",
    )
    args = parser.parse_args()
    hololink_module.logging_level(args.log_level)
    logging.getLogger().setLevel(args.log_level)
    logging.info("Initializing.")
    # Get a handle to the GPU
    (cu_result,) = cuda.cuInit(0)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_device_ordinal = 0
    cu_result, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_result, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    # # Get a handle to the Hololink device
    channel_metadata = hololink_module.Enumerator.find_channel(channel_ip=args.hololink)
    overrides = hololink_module.Metadata(
        {
            "vsync_enable": 0,  # or 1
            "block_enable": 0,  # or 1
        }
    )
    channel_metadata.update(overrides)
    hololink_channel = hololink_module.DataChannel(channel_metadata)
    # # Get a handle to the camera
    logging.info("camera mode: %s", args.camera_mode)
    camera_mode = hololink_module.sensors.d555.d555_mode.RealSense_Mode(
        args.camera_mode % hololink_module.sensors.d555.d555_mode.PROFILE_COUNT
    )
    stream_id = (
        hololink_module.sensors.d555.d555_mode.RealSense_StreamId.DEPTH
        if args.camera_mode < hololink_module.sensors.d555.d555_mode.PROFILE_COUNT
        else hololink_module.sensors.d555.d555_mode.RealSense_StreamId.RGB
    )
    camera = hololink_module.sensors.d555.d555.RealsenseCamD555(
        hololink_channel, stream_id
    )

    # Set up the application
    application = HoloscanApplication(
        args.headless,
        args.fullscreen,
        cu_context,
        cu_device_ordinal,
        hololink_channel,
        camera,
        camera_mode,
        args.frame_limit,
        stream_id.value,
    )
    application.config(args.configuration)
    # Run it.
    hololink = hololink_channel.hololink()
    hololink.start()
    camera.configure(camera_mode)
    os.environ["GXF_MEMORY_DEBUG"] = "1"
    application.run()
    hololink.stop()

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS


if __name__ == "__main__":
    main()
