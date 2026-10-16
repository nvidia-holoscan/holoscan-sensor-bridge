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

# OX08D10 camera player for the Taurotech HSB DA326 board (GMSL2 via MAX96716A/MAX96719B).
#
# Prerequisites:
#   - Taurotech HSB DA326 board with MAX96716A deserializer
#   - Aumovio SAT301DP17 camera module connected to GMSL link A or B
#   - Board init (release_reset, power_cycle, setup_clock) runs via
#     TauroTechDa326InterfaceV1 before GMSL bring-up.

import argparse
import ctypes
import logging
import os

import cuda.bindings.driver as cuda
import cupy as cp
import hololink_module.operators
import hololink_module.taurotech_da326 as _da326_module
import holoscan
from hololink.sensors.ox08d10 import ox08d10_mode
from hololink.sensors.ox08d10.ox08d10_cam import Ox08d10Cam

import hololink as hololink_legacy
import hololink_module

CHANNEL_TO_SIF = {"A": 0, "B": 1}


class FrameSizeFilterOp(holoscan.core.Operator):
    """Drop frames where bytes_written or tensor size doesn't match expected_frame_size.

    Incomplete frames passed to CsiToBayerOp produce stitched/ghost artifacts.
    Log frame sizes for the first max_log_frames frames to diagnose sizing issues.
    """

    def __init__(self, *args, expected_frame_size, max_log_frames=0, **kwargs):
        super().__init__(*args, **kwargs)
        self._expected_frame_size = expected_frame_size
        self._max_log_frames = max_log_frames
        self._frame_count = 0
        self._dropped_count = 0
        self._sizes = {}

    def setup(self, spec):
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("input")
        cp_frame = cp.asarray(in_message.get(""))
        bytes_written = self.metadata.get("bytes_written")
        tensor_size = cp_frame.size
        self._sizes[bytes_written] = self._sizes.get(bytes_written, 0) + 1
        self._frame_count += 1

        if self._max_log_frames and self._frame_count <= self._max_log_frames:
            logging.info(
                "frame-size-check frame=%d bytes_written=%s tensor_size=%d "
                "expected=%d unique_sizes=%s",
                self._frame_count,
                bytes_written,
                tensor_size,
                self._expected_frame_size,
                dict(sorted(self._sizes.items())),
            )

        if (
            bytes_written != self._expected_frame_size
            or tensor_size != self._expected_frame_size
        ):
            self._dropped_count += 1
            if self._dropped_count == 1 or (
                self._max_log_frames
                and (self._dropped_count <= 10 or self._dropped_count % 60 == 0)
            ):
                logging.warning(
                    "Dropping incomplete OX08D10 frame: frame=%d dropped=%d "
                    "bytes_written=%s tensor_size=%d expected=%d",
                    self._frame_count,
                    self._dropped_count,
                    bytes_written,
                    tensor_size,
                    self._expected_frame_size,
                )
            return

        op_output.emit({"": cp_frame}, "output")


class HoloscanApplication(holoscan.core.Application):
    def __init__(
        self,
        headless,
        fullscreen,
        cuda_context,
        cuda_device_ordinal,
        sif_metadata,
        camera,
        camera_mode,
        frame_limit,
        window_height,
        window_width,
        window_title,
        check_frame_sizes=0,
    ):
        logging.info("__init__")
        super().__init__()
        self._headless = headless
        self._fullscreen = fullscreen
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._sif_metadata = sif_metadata
        self._camera = camera
        self._camera_mode = camera_mode
        self._frame_limit = frame_limit
        self._window_height = window_height
        self._window_width = window_width
        self._window_title = window_title
        self._check_frame_sizes = check_frame_sizes

    def compose(self):
        logging.info("compose")
        if self._frame_limit is not None:
            condition = holoscan.conditions.CountCondition(
                self, name="count", count=self._frame_limit
            )
        else:
            condition = holoscan.conditions.BooleanCondition(
                self, name="ok", enable_tick=True
            )

        self._camera.set_mode(self._camera_mode)

        csi_to_bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="pool",
            storage_type=1,
            block_size=self._camera._width
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera._height,
            num_blocks=2,
        )
        csi_to_bayer_operator = hololink_legacy.operators.CsiToBayerOp(
            self,
            name="csi_to_bayer",
            allocator=csi_to_bayer_pool,
            cuda_device_ordinal=self._cuda_device_ordinal,
        )
        self._camera.configure_converter(csi_to_bayer_operator)

        frame_size = csi_to_bayer_operator.get_csi_length()
        receiver_operator = hololink_module.operators.LinuxReceiverOp(
            self,
            condition,
            name="receiver",
            enumeration_metadata=self._sif_metadata,
            frame_context=self._cuda_context,
            frame_size=frame_size,
            device_start=self._camera.start,
            device_stop=self._camera.stop,
        )

        frame_size_filter = FrameSizeFilterOp(
            self,
            name="frame_size_filter",
            expected_frame_size=frame_size,
            max_log_frames=self._check_frame_sizes,
        )

        pixel_format = self._camera.pixel_format()
        bayer_format = self._camera.bayer_format()
        image_processor_operator = hololink_legacy.operators.ImageProcessorOp(
            self,
            name="image_processor",
            # OX08D10 optical black: datasheet BLKD_SEL default is 0x40 (64 DN at 10-bit).
            # For RAW12 HDR mode this is unconfirmed; adjust from vendor calibration.
            optical_black=64,
            bayer_format=bayer_format.value,
            pixel_format=pixel_format.value,
        )

        rgba_components_per_pixel = 4
        bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="pool",
            storage_type=1,
            block_size=self._camera._width
            * rgba_components_per_pixel
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera._height,
            num_blocks=2,
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
            fullscreen=self._fullscreen,
            headless=self._headless,
            height=self._window_height,
            width=self._window_width,
            window_title=self._window_title,
        )

        self.add_flow(receiver_operator, frame_size_filter, {("output", "input")})
        self.add_flow(frame_size_filter, csi_to_bayer_operator, {("output", "input")})
        self.add_flow(
            csi_to_bayer_operator, image_processor_operator, {("output", "input")}
        )
        self.add_flow(image_processor_operator, demosaic, {("output", "receiver")})
        self.add_flow(demosaic, visualizer, {("transmitter", "receivers")})


def main():
    parser = argparse.ArgumentParser(
        description="OX08D10 player (GMSL2 via Taurotech HSB DA326 board)"
    )
    modes = ox08d10_mode.Ox08d10_Mode
    mode_choices = [modes.OX08D10_MODE_3840X2160_30FPS_RAW12.value]
    parser.add_argument(
        "--camera-mode",
        type=int,
        choices=mode_choices,
        default=modes.OX08D10_MODE_3840X2160_30FPS_RAW12.value,
        help="1:OX08D10_MODE_3840X2160_30FPS_RAW12",
    )
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--frame-limit", type=int, default=None)
    default_configuration = os.path.join(
        os.path.dirname(__file__), "example_configuration.yaml"
    )
    parser.add_argument("--configuration", default=default_configuration)
    parser.add_argument("--hololink", default="192.168.0.2")
    parser.add_argument("--log-level", type=int, default=20)
    parser.add_argument(
        "--channel",
        default="A",
        choices=("A", "B"),
        help="GMSL link the camera is connected to on the Taurotech board",
    )
    parser.add_argument("--window-height", type=int, default=1080)
    parser.add_argument("--window-width", type=int, default=1920)
    parser.add_argument("--title", default="OX08D10")
    parser.add_argument("--skip-reset", action="store_true")
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip board init, GMSL bring-up and sensor init (use if already streaming)",
    )
    parser.add_argument("--exposure", type=lambda x: int(x, 0), default=0x0200)
    parser.add_argument("--pattern", type=int, choices=range(4))
    parser.add_argument(
        "--check-frame-sizes",
        type=int,
        default=0,
        metavar="N",
        help="Log receiver bytes_written/tensor sizes for the first N frames",
    )
    args = parser.parse_args()

    hololink_legacy.logging_level(args.log_level)
    logging.info("Initializing.")

    (cu_result,) = cuda.cuInit(0)
    if cu_result != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuInit failed: {cu_result}")
    cu_device_ordinal = 0
    cu_result, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
    if cu_result != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuDeviceGet failed: {cu_result}")
    cu_result, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
    if cu_result != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuDevicePrimaryCtxRetain failed: {cu_result}")

    adapter = hololink_module.Adapter.get_adapter()
    metadata = adapter.wait_for_channel(args.hololink, 30.0)

    holo_iface = None
    try:
        holo_iface = hololink_module.HololinkInterfaceV1.get_service(metadata)
        holo_iface.start()
        board = _da326_module.TauroTechDa326InterfaceV1.get_service(metadata)

        sif = CHANNEL_TO_SIF[args.channel]
        sif_metadata = hololink_module.EnumerationMetadata(metadata)
        adapter.use_sensor(sif_metadata, sif)

        camera = Ox08d10Cam(
            board.hololink(),
            skip_i2c=args.skip_setup,
        )
        camera_mode = ox08d10_mode.Ox08d10_Mode(args.camera_mode)

        application = HoloscanApplication(
            args.headless,
            args.fullscreen,
            cu_context,
            cu_device_ordinal,
            sif_metadata,
            camera,
            camera_mode,
            args.frame_limit,
            args.window_height,
            args.window_width,
            args.title,
            args.check_frame_sizes,
        )
        application.config(args.configuration)

        str_to_link = {
            "A": ox08d10_mode.GmslLink.LINK_A,
            "B": ox08d10_mode.GmslLink.LINK_B,
        }

        if not args.skip_setup:
            if not args.skip_reset:
                holo_iface.reset()

            # DA326 board init: releases camera reset, power-cycles the module,
            # verifies power, and sets up board clocks. Must run before GMSL bring-up.
            board.release_reset()
            board.power_cycle()
            board.check_power()
            board.setup_clock()

            # GMSL bring-up: configures MAX96716A deserializer, MAX96719B serializer,
            # releases sensor reset, waits for boot, validates sensor ID, writes mode
            # registers, and sets virtual channel mapping.
            camera.configure_vendor_tunnel_path(
                camera_mode,
                str_to_link[args.channel],
            )
            if args.pattern is not None:
                camera.test_pattern(args.pattern)
            camera.set_exposure_reg(args.exposure)

        logging.info("Calling run")
        application.run()
    finally:
        if holo_iface is not None:
            holo_iface.stop()
        (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    # Checked out here so that a teardown failure doesn't displace an exception
    # coming out of application.run().
    if cu_result != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuDevicePrimaryCtxRelease failed: {cu_result}")


if __name__ == "__main__":
    main()
