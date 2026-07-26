# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""IMX274 camera player built on the hololink_module API.

Runnable companion to the "IMX274 Module Tutorial" user guide. It plays one
IMX274 camera behind a Holoscan Sensor Bridge board through the hardware (RoCE)
receiver and displays the video with Holoviz. Configuration is intentionally
hard-coded so the tutorial can focus on the API rather than on argument parsing.
"""

import ctypes

import cuda.bindings.driver as cuda
import hololink_module.operators
import hololink_module.sensors.imx274
import holoscan
from hololink_module.sensors.imx274 import imx274_mode as adapter_imx274_mode

import hololink_module

# The board announces itself over bootp; this is the peer IP we wait for.
HOLOLINK_IP = "192.168.0.2"
# Seconds to wait for that announcement before giving up.
DISCOVERY_TIMEOUT = 30.0
CAMERA_MODE = adapter_imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS


class HoloscanApplication(holoscan.core.Application):
    def __init__(self, cuda_context, cuda_device_ordinal, metadata, camera):
        super().__init__()
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._metadata = metadata
        self._camera = camera

    def compose(self):
        self._camera.set_mode(CAMERA_MODE)
        camera_width = self._camera._width
        camera_height = self._camera._height
        bayer_format = self._camera.bayer_format()
        pixel_format = self._camera.pixel_format()

        csi_to_bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="csi_pool",
            storage_type=1,  # device
            block_size=camera_width * ctypes.sizeof(ctypes.c_uint16) * camera_height,
            num_blocks=4,
        )
        csi_op = hololink_module.operators.CsiToBayerOp(
            self,
            name="csi_to_bayer",
            allocator=csi_to_bayer_pool,
            cuda_device_ordinal=self._cuda_device_ordinal,
        )
        # Configuring the converter tells us how big each received frame is.
        self._camera.configure_converter(csi_op)
        frame_size = csi_op.get_csi_length()

        receiver = hololink_module.operators.RoceReceiverOp(
            self,
            name="receiver",
            enumeration_metadata=self._metadata,
            frame_context=self._cuda_context,
            frame_size=frame_size,
            device_start=self._camera.start,
            device_stop=self._camera.stop,
        )

        image_proc = hololink_module.operators.ImageProcessorOp(
            self,
            name="image_processor",
            optical_black=50,  # IMX274 optical black
            bayer_format=bayer_format.value,
            pixel_format=pixel_format.value,
        )

        rgba_components_per_pixel = 4
        bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="bayer_pool",
            storage_type=1,
            block_size=camera_width
            * rgba_components_per_pixel
            * ctypes.sizeof(ctypes.c_uint16)
            * camera_height,
            num_blocks=4,
        )
        demosaic = holoscan.operators.BayerDemosaicOp(
            self,
            name="demosaic",
            pool=bayer_pool,
            generate_alpha=True,
            alpha_value=65535,
            bayer_grid_pos=bayer_format.value,
            interpolation_mode=0,
        )

        visualizer = holoscan.operators.HolovizOp(
            self,
            name="holoviz",
            framebuffer_srgb=True,
        )

        self.add_flow(receiver, csi_op, {("output", "input")})
        self.add_flow(csi_op, image_proc, {("output", "input")})
        self.add_flow(image_proc, demosaic, {("output", "receiver")})
        self.add_flow(demosaic, visualizer, {("transmitter", "receivers")})


def main():
    # CUDA bring-up
    (cu_result,) = cuda.cuInit(0)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_device_ordinal = 0
    cu_result, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_result, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS

    # Find the board via bootp enumeration and get its metadata.
    adapter = hololink_module.Adapter.get_adapter()
    metadata = adapter.wait_for_channel(HOLOLINK_IP, DISCOVERY_TIMEOUT)

    # The sensor driver is linked directly into the application; it is not a
    # service. It resolves the services it needs from the enumeration metadata.
    camera = hololink_module.sensors.imx274.Imx274Cam(metadata)

    # The control plane is a versioned service fetched by metadata.
    hololink = hololink_module.HololinkInterfaceV1.get_service(metadata)
    # start() opens the control-plane socket; without it, no device I/O works.
    hololink.start()
    try:
        # The framework leaves device state alone unless the application asks;
        # reset() drives the board into a known state.
        hololink.reset()
        # configure() writes device registers, so it must run after reset().
        camera.configure(CAMERA_MODE)
        camera.set_digital_gain_reg(0x4)

        app = HoloscanApplication(cu_context, cu_device_ordinal, metadata, camera)
        app.run()
    finally:
        # Stop streaming data and close up sockets.
        hololink.stop()
        (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
        assert cu_result == cuda.CUresult.CUDA_SUCCESS


if __name__ == "__main__":
    main()
