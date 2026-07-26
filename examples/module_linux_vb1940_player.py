# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Single-camera Leopard VB1940 player driving the hololink_module V1
surface through the SOFTWARE (Linux) receiver.

Identical in shape to the C++ examples/module_linux_vb1940_player.cpp;
the receiver is the software LinuxReceiverOp (no ibverbs), so this runs
on hosts with no infiniband device.
"""

import argparse
import ctypes
import logging

import cuda.bindings.driver as cuda
import hololink_module.operators
import hololink_module.sensors.vb1940
import holoscan

import hololink_module


class HoloscanApplication(holoscan.core.Application):
    def __init__(
        self,
        headless,
        cuda_context,
        cuda_device_ordinal,
        metadata,
        camera,
        frame_limit,
    ):
        super().__init__()
        self._headless = headless
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._metadata = metadata
        self._camera = camera
        self._frame_limit = frame_limit
        self.is_metadata_enabled = True
        self.metadata_policy = holoscan.core.MetadataPolicy.REJECT

    def compose(self):
        if self._frame_limit:
            condition = holoscan.conditions.CountCondition(
                self, name="count", count=self._frame_limit
            )
        else:
            condition = holoscan.conditions.BooleanCondition(
                self, name="ok", enable_tick=True
            )

        camera_width = self._camera.width()
        camera_height = self._camera.height()
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
        self._camera.configure_converter(csi_op)
        frame_size = csi_op.get_csi_length()

        receiver = hololink_module.operators.LinuxReceiverOp(
            self,
            condition,
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
            optical_black=8,  # VB1940 optical black (RAW10)
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
            headless=self._headless,
            framebuffer_srgb=True,
        )

        self.add_flow(receiver, csi_op, {("output", "input")})
        self.add_flow(csi_op, image_proc, {("output", "input")})
        self.add_flow(image_proc, demosaic, {("output", "receiver")})
        self.add_flow(demosaic, visualizer, {("transmitter", "receivers")})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera-mode",
        type=int,
        default=hololink_module.sensors.vb1940.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS.value,
        help="VB1940 mode (0=2560x1984 30fps, 1=1920x1080 30fps, "
        "2=2560x1984 30fps 8-bit, 3=2560x1984 60fps)",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Exit after receiving this many frames",
    )
    parser.add_argument(
        "--hololink",
        default="192.168.0.2",
        help="Peer IP of the Leopard VB1940 board (waited for via bootp)",
    )
    parser.add_argument(
        "--discovery-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for the bootp announcement before giving up",
    )
    parser.add_argument(
        "--module-dir",
        default=None,
        help="Override the directory containing hololink_<UUID>.so files",
    )
    parser.add_argument("--log-level", type=int, default=20)
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)

    # CUDA bring-up
    (cu_result,) = cuda.cuInit(0)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_device_ordinal = 0
    cu_result, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_result, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS

    adapter = hololink_module.Adapter.get_adapter()
    if args.module_dir:
        adapter.set_module_directory(args.module_dir)

    metadata = adapter.wait_for_channel(args.hololink, args.discovery_timeout)

    # Get the driver object
    camera = hololink_module.sensors.vb1940.Vb1940Cam(metadata)
    # Connect to the HSB unit
    hololink = hololink_module.HololinkInterfaceV1.get_service(metadata)
    hololink.start()
    try:
        # Drive the unit into a known state
        hololink.reset()
        # Configure the camera (walks the secure-boot FSM the first time
        # after power-up, then applies the per-mode register table).
        camera_mode = hololink_module.sensors.vb1940.Vb1940_Mode(args.camera_mode)
        camera.configure(camera_mode)

        # Run it
        app = HoloscanApplication(
            headless=args.headless,
            cuda_context=cu_context,
            cuda_device_ordinal=cu_device_ordinal,
            metadata=metadata,
            camera=camera,
            frame_limit=args.frame_limit,
        )
        app.run()
    finally:
        # Stop streaming data and close up sockets
        hololink.stop()
        (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
        assert cu_result == cuda.CUresult.CUDA_SUCCESS


if __name__ == "__main__":
    main()
