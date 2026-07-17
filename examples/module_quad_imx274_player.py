# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""4-channel IMX274 player driving the hololink_module V1 surface.

Topology: 4 independent HSB-Lite data planes (each its own IMX274
camera) flowing into a single Holoscan pipeline (RoceReceiverOp →
CsiToBayerOp → ImageProcessorOp → BayerDemosaicOp → HolovizOp).

Default IPs: 192.168.0.200 / .201 / .202 / .203 (devices 0..3). The
example pins the cameras to IMX274_MODE_1920X1080_60FPS (mode 1, 1080p
RAW10) so the four-channel aggregate fits in the data-plane bandwidth
budget; the 4K modes over four channels exceed what the link can
sustain.

Hardware required: 4 HSB-Lite data planes (each driving an IMX274
camera) reachable on the test network.
"""

import argparse
import ctypes
import dataclasses
import logging
import time

import cuda.bindings.driver as cuda
import hololink_module.operators
import hololink_module.sensors.imx274
import holoscan
from hololink_module.sensors.imx274 import imx274_mode as adapter_imx274_mode

import hololink_module


@dataclasses.dataclass
class Channel:
    """Everything the application needs to drive one data plane.

    The enumeration metadata identifies the supplement module; the
    HololinkInterface drives the per-board control plane lifecycle;
    the Imx274Cam drives this data plane's sensor. Two channels that
    share a board hold the same HololinkInterface (the service locator
    returns one instance per (module, instance_id) pair); the
    application treats them independently regardless.
    """

    metadata: object
    hololink: object
    camera: object


class HoloscanApplication(holoscan.core.Application):
    def __init__(
        self,
        headless,
        cuda_context,
        cuda_device_ordinal,
        channels,
        camera_mode,
        frame_limit,
        window_height,
        window_width,
        window_title,
    ):
        super().__init__()
        self._headless = headless
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._channels = channels
        self._camera_mode = camera_mode
        self._frame_limit = frame_limit
        self._window_height = window_height
        self._window_width = window_width
        self._window_title = window_title
        self.is_metadata_enabled = True
        self.metadata_policy = holoscan.core.MetadataPolicy.REJECT

    def compose(self):
        # One condition per data plane.
        conditions = []
        for i, _ in enumerate(self._channels):
            if self._frame_limit:
                cond = holoscan.conditions.CountCondition(
                    self, name=f"count_{i}", count=self._frame_limit
                )
            else:
                cond = holoscan.conditions.BooleanCondition(
                    self, name=f"ok_{i}", enable_tick=True
                )
            conditions.append(cond)

        for ch in self._channels:
            ch.camera.set_mode(self._camera_mode)

        # Sensors + buffers
        camera_width = self._channels[0].camera._width
        camera_height = self._channels[0].camera._height
        bayer_format = self._channels[0].camera.bayer_format()
        pixel_format = self._channels[0].camera.pixel_format()

        csi_to_bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="csi_pool",
            storage_type=1,  # device
            block_size=camera_width * ctypes.sizeof(ctypes.c_uint16) * camera_height,
            num_blocks=4 * len(self._channels),
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
            num_blocks=4 * len(self._channels),
        )

        receivers = []
        csi_ops = []
        image_processors = []
        demosaics = []
        tensor_specs = []

        for i, ch in enumerate(self._channels):
            tensor_name = f"cam{i}"

            csi_op = hololink_module.operators.CsiToBayerOp(
                self,
                name=f"csi_to_bayer_{i}",
                allocator=csi_to_bayer_pool,
                cuda_device_ordinal=self._cuda_device_ordinal,
                out_tensor_name=tensor_name,
            )
            ch.camera.configure_converter(csi_op)
            csi_ops.append(csi_op)

            frame_size = csi_op.get_csi_length()

            receiver = hololink_module.operators.RoceReceiverOp(
                self,
                conditions[i],
                name=f"receiver_{i}",
                enumeration_metadata=ch.metadata,
                frame_context=self._cuda_context,
                frame_size=frame_size,
                device_start=ch.camera.start,
                device_stop=ch.camera.stop,
            )
            receivers.append(receiver)

            image_proc = hololink_module.operators.ImageProcessorOp(
                self,
                name=f"image_processor_{i}",
                optical_black=50,  # IMX274 optical black
                bayer_format=bayer_format.value,
                pixel_format=pixel_format.value,
            )
            image_processors.append(image_proc)

            demosaic = holoscan.operators.BayerDemosaicOp(
                self,
                name=f"demosaic_{i}",
                pool=bayer_pool,
                generate_alpha=True,
                alpha_value=65535,
                bayer_grid_pos=bayer_format.value,
                interpolation_mode=0,
                in_tensor_name=tensor_name,
                out_tensor_name=tensor_name,
            )
            demosaics.append(demosaic)

            # 2x2 grid: cam0 top-left, cam1 top-right, cam2 bottom-left, cam3 bottom-right.
            spec = holoscan.operators.HolovizOp.InputSpec(
                tensor_name, holoscan.operators.HolovizOp.InputType.COLOR
            )
            view = holoscan.operators.HolovizOp.InputSpec.View()
            view.offset_x = 0.5 * (i % 2)
            view.offset_y = 0.5 * (i // 2)
            view.width = 0.5
            view.height = 0.5
            spec.views = [view]
            tensor_specs.append(spec)

        visualizer = holoscan.operators.HolovizOp(
            self,
            name="holoviz",
            headless=self._headless,
            framebuffer_srgb=True,
            tensors=tensor_specs,
            height=self._window_height,
            width=self._window_width,
            window_title=self._window_title,
        )

        for receiver, csi_op, image_proc, demosaic in zip(
            receivers, csi_ops, image_processors, demosaics
        ):
            self.add_flow(receiver, csi_op, {("output", "input")})
            self.add_flow(csi_op, image_proc, {("output", "input")})
            self.add_flow(image_proc, demosaic, {("output", "receiver")})
            self.add_flow(demosaic, visualizer, {("transmitter", "receivers")})


def _enumerate(adapter, expected_peer_ips, timeout_s=30.0):
    """Block until every expected peer IP has been enumerated.

    Each Adapter.wait_for_channel call is bounded by the remaining
    time against the cumulative deadline, so the total time to wait
    for all peers is bounded by ``timeout_s`` rather than
    ``len(expected_peer_ips) * timeout_s``. Returns the metadata list
    in the same order as ``expected_peer_ips``. Raises RuntimeError
    when any peer fails to announce within the remaining budget.
    """
    deadline = time.monotonic() + timeout_s
    found = []
    for ip in expected_peer_ips:
        remaining = max(deadline - time.monotonic(), 0.0)
        found.append(adapter.wait_for_channel(ip, remaining))
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera-mode",
        type=int,
        default=adapter_imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS.value,
        help="IMX274 mode (1=1920x1080, 0=3840x2160 RAW10, 2=3840x2160 RAW12)",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Exit after receiving this many frames per channel",
    )
    parser.add_argument(
        "--device-0",
        default="192.168.0.200",
        help="Peer IP of device 0 (waited for via bootp)",
    )
    parser.add_argument(
        "--device-1",
        default="192.168.0.201",
        help="Peer IP of device 1 (waited for via bootp)",
    )
    parser.add_argument(
        "--device-2",
        default="192.168.0.202",
        help="Peer IP of device 2 (waited for via bootp)",
    )
    parser.add_argument(
        "--device-3",
        default="192.168.0.203",
        help="Peer IP of device 3 (waited for via bootp)",
    )
    parser.add_argument(
        "--discovery-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for all four bootp announcements before giving up",
    )
    parser.add_argument(
        "--module-dir",
        default=None,
        help="Override the directory containing hololink_<UUID>.so files "
        "(env HOLOLINK_MODULE_DIR or /usr/lib/hololink/modules otherwise)",
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
    )
    parser.add_argument("--window-height", type=int, default=2160 // 4)
    parser.add_argument("--window-width", type=int, default=3840 // 3)
    parser.add_argument("--title", default="hololink_module — 4-channel IMX274")
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

    device_ips = [args.device_0, args.device_1, args.device_2, args.device_3]
    metadatas = _enumerate(adapter, device_ips, timeout_s=args.discovery_timeout)

    # Build a channel per device.
    channels = []
    for md in metadatas:
        # Get the driver object
        camera = hololink_module.sensors.imx274.Imx274Cam(md)
        # Connect to the HSB unit
        hololink = hololink_module.HololinkInterfaceV1.get_service(md)
        channels.append(Channel(metadata=md, hololink=hololink, camera=camera))

    # Bring each unit up; channels sharing a board share a HololinkInterface,
    # so start/reset run once per board.
    camera_mode = adapter_imx274_mode.Imx274_Mode(args.camera_mode)
    started = []
    brought_up = set()
    for ch in channels:
        if id(ch.hololink) not in brought_up:
            brought_up.add(id(ch.hololink))
            ch.hololink.start()
            started.append(ch.hololink)
            # Drive the unit into a known state
            ch.hololink.reset()
        # Configure the camera
        ch.camera.configure(camera_mode)
        ch.camera.set_digital_gain_reg(0x4)

    # Run it
    app = HoloscanApplication(
        headless=args.headless,
        cuda_context=cu_context,
        cuda_device_ordinal=cu_device_ordinal,
        channels=channels,
        camera_mode=camera_mode,
        frame_limit=args.frame_limit,
        window_height=args.window_height,
        window_width=args.window_width,
        window_title=args.title,
    )
    app.run()

    # Stop streaming data and close up sockets
    for hololink in started:
        hololink.stop()
    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS


if __name__ == "__main__":
    main()
