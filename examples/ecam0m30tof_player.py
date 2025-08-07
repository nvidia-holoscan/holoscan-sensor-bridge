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

import argparse
import ctypes
import logging
import os
import time

import cupy as cp
import holoscan
from cuda import cuda

import hololink as hololink_module


def write_lmmi_register(camera_idx, data, addr, hololink):
    slow_spi = hololink.get_spi(
        hololink_module.CLNX_SPI_BUS,
        chip_select=0,
        cpol=0,
        cpha=1,
        width=1,
        prescaler=0xF,
    )
    request = [0x11, 0x0C]
    while 1:  # Check Ready
        rdval = slow_spi.spi_transaction(bytearray(), bytearray(request), 1)
        rdval_int = int.from_bytes(bytes(rdval), byteorder="little")
        if rdval_int & 0x4:
            break
    LMMI_WRITE = 1
    request = [LMMI_WRITE, 0x0D, data]
    slow_spi.spi_transaction(bytearray(), bytearray(request), 0)
    request = [LMMI_WRITE, 0x0F, addr]
    slow_spi.spi_transaction(bytearray(), bytearray(request), 0)
    request = [LMMI_WRITE, 0x0C, (camera_idx << 4) + 0x1]
    slow_spi.spi_transaction(bytearray(), bytearray(request), 0)


def reconfigurable_mipi(
    camera_idx, lane_prog, clock_lane_settle_cycle, data_lane_settle_cycle, hololink
):
    logging.debug(lane_prog, clock_lane_settle_cycle, data_lane_settle_cycle)

    reg0A = ((clock_lane_settle_cycle & 0x1) << 3) + (lane_prog << 1)
    reg0B = clock_lane_settle_cycle >> 1
    reg0C = (clock_lane_settle_cycle >> 2) & 0x80
    reg0F = (data_lane_settle_cycle & 0x3) << 2
    reg10 = (data_lane_settle_cycle >> 2) & 0xF

    logging.debug(reg0A, reg0B, reg0C, reg0F, reg10)

    write_lmmi_register(camera_idx, reg0A, 0x0A, hololink)
    write_lmmi_register(camera_idx, reg0B, 0x0B, hololink)
    write_lmmi_register(camera_idx, reg0C, 0x0C, hololink)
    write_lmmi_register(camera_idx, reg0F, 0x0F, hololink)
    write_lmmi_register(camera_idx, reg10, 0x10, hololink)


def program_mipi_phy(camera_idx, num_lanes, clock_freq, hololink):

    lane_prog = num_lanes - 1
    if clock_freq >= 1000:
        data_lane_settle_cycle = 6
    elif clock_freq >= 350:
        data_lane_settle_cycle = 7
    elif clock_freq >= 200:
        data_lane_settle_cycle = 8
    elif clock_freq >= 150:
        data_lane_settle_cycle = 9
    else:
        data_lane_settle_cycle = 11
    clock_lane_settle_cycle = 9
    reconfigurable_mipi(
        camera_idx, lane_prog, clock_lane_settle_cycle, data_lane_settle_cycle, hololink
    )


def convert_depth_to_grayscale(depth_image):
    # Normalize the depth image to the range 0 to 255
    # 1. First, normalize the values from 0 to 65535 to 0 to 1
    depth_normalized = depth_image.astype(cp.float32) / 65535.0
    # 2. Scale it to the range 0 to 255 (for 8-bit grayscale)
    depth_grayscale = cp.clip(depth_normalized * 255, 0, 255).astype(cp.uint8)
    return depth_grayscale


class ImageShiftAndProcessingOperator(holoscan.core.Operator):
    def __init__(self, *args, no_of_planes=1, **kwargs):
        super().__init__(*args, **kwargs)
        self._no_of_planes = no_of_planes
        self._color_map = cp.zeros((480, 640, 3), cp.uint8)

    def setup(self, spec):
        logging.info("setup")
        spec.input("input")
        spec.output("output")

    def start(self):
        pass

    def stop(self):
        pass

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message = op_input.receive("input")
        msg = in_message.get("image")
        cp_frame = cp.asarray(msg)
        cp_frame_uint8 = convert_depth_to_grayscale(cp_frame)
        if self._no_of_planes == 1:
            op_output.emit({"image": cp_frame_uint8}, "output")
        elif self._no_of_planes == 2:
            depth = cp_frame_uint8[-1::-2, :, :]
            active_ir = cp_frame_uint8[1::2, :, 0] * 2
            self._color_map[:, :, 0] = cp.squeeze(active_ir)
            self._color_map[:, :, 1] = cp.squeeze(active_ir)
            self._color_map[:, :, 2] = cp.squeeze(active_ir)
            op_output.emit({"depth": depth, "image": self._color_map}, "output")


class HoloscanApplication(holoscan.core.Application):
    def __init__(
        self,
        headless,
        fullscreen,
        cuda_context,
        cuda_device_ordinal,
        hololink_channel,
        ibv_name,
        ibv_port,
        camera,
        camera_mode,
        frame_limit,
    ):
        logging.info("__init__")
        super().__init__()
        self._headless = headless
        self._fullscreen = fullscreen
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._hololink_channel = hololink_channel
        self._ibv_name = ibv_name
        self._ibv_port = ibv_port
        self._camera = camera
        self._camera_mode = camera_mode
        self._frame_limit = frame_limit
        self._no_of_planes = 1
        if (
            camera_mode
            == hololink_module.sensors.ecam0m30tof.ECam0M30Tof_Mode.EDEPTH_MODE_DEPTH_IR
        ):
            self._no_of_planes = 2

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

        csi_to_bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="pool",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=self._camera._width
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera._height,
            num_blocks=2,
        )
        csi_to_bayer_operator = hololink_module.operators.CsiToBayerOp(
            self,
            name="csi_to_bayer",
            allocator=csi_to_bayer_pool,
            cuda_device_ordinal=self._cuda_device_ordinal,
            out_tensor_name="image",
        )
        self._camera.configure_converter(csi_to_bayer_operator)

        frame_size = csi_to_bayer_operator.get_csi_length()
        logging.info(f"{frame_size=}")
        frame_context = self._cuda_context
        receiver_operator = hololink_module.operators.RoceReceiverOp(
            self,
            condition,
            name="receiver",
            frame_size=frame_size,
            frame_context=frame_context,
            ibv_name=self._ibv_name,
            ibv_port=self._ibv_port,
            hololink_channel=self._hololink_channel,
            device=self._camera,
        )

        image_shift = ImageShiftAndProcessingOperator(
            self,
            name="image_shift",
            no_of_planes=self._no_of_planes,
        )

        image_spec = holoscan.operators.HolovizOp.InputSpec(
            "image", holoscan.operators.HolovizOp.InputType.COLOR
        )
        tensor_list = [image_spec]

        if self._no_of_planes == 2:
            view = holoscan.operators.HolovizOp.InputSpec.View()
            view.offset_x = 0.0
            view.offset_y = 0.0
            view.width = 0.5
            view.height = 1
            image_spec.views = [view]

            image2_spec = holoscan.operators.HolovizOp.InputSpec(
                "depth", holoscan.operators.HolovizOp.InputType.DEPTH_MAP
            )
            view = holoscan.operators.HolovizOp.InputSpec.View()
            view.offset_x = 0.5
            view.offset_y = 0.0
            view.width = 0.5
            view.height = 1
            image2_spec.views = [view]
            tensor_list = [image_spec, image2_spec]

        visualizer = holoscan.operators.HolovizOp(
            self,
            name="holoviz",
            fullscreen=self._fullscreen,
            headless=self._headless,
            width=640 * self._no_of_planes,
            height=480,
            tensors=tensor_list,
        )

        self.add_flow(receiver_operator, csi_to_bayer_operator, {("output", "input")})
        self.add_flow(csi_to_bayer_operator, image_shift, {("output", "input")})
        self.add_flow(image_shift, visualizer, {("output", "receivers")})


def main():
    parser = argparse.ArgumentParser()
    modes = hololink_module.sensors.ecam0m30tof.ECam0M30Tof_Mode
    mode_choices = [mode.value for mode in modes]
    mode_help = " ".join([f"{mode.value}:{mode.name}" for mode in modes])
    parser.add_argument(
        "--camera-mode",
        type=int,
        choices=mode_choices,
        default=mode_choices[0],
        help=mode_help,
    )
    parser.add_argument(
        "--depth-range",
        type=int,
        choices=(0, 1),
        default=1,
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
    infiniband_devices = hololink_module.infiniband_devices()
    parser.add_argument(
        "--ibv-name",
        default=infiniband_devices[0],
        help="IBV device to use",
    )
    parser.add_argument(
        "--ibv-port",
        type=int,
        default=1,
        help="Port number of IBV device",
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
    parser.add_argument(
        "--ptp-sync",
        action="store_true",
        help="After reset, wait for PTP time to synchronize.",
    )
    parser.add_argument(
        "--skip-reset",
        action="store_true",
        help="Don't call reset on the hololink device.",
    )
    args = parser.parse_args()
    hololink_module.logging_level(args.log_level)
    logging.info("Initializing.")
    # Get a handle to the GPU
    (cu_result,) = cuda.cuInit(0)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_device_ordinal = 0
    cu_result, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_result, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    # Get a handle to the data source
    channel_metadata = hololink_module.Enumerator.find_channel(channel_ip=args.hololink)
    logging.info(f"{channel_metadata=}")
    hololink_channel = hololink_module.DataChannel(channel_metadata)
    # Get a handle to the camera
    camera = hololink_module.sensors.ecam0m30tof.ECam0M30Tof(
        hololink_channel, expander_configuration=args.expander_configuration
    )
    camera_mode = hololink_module.sensors.ecam0m30tof.ECam0M30Tof_Mode(args.camera_mode)
    # Set up the application
    application = HoloscanApplication(
        args.headless,
        args.fullscreen,
        cu_context,
        cu_device_ordinal,
        hololink_channel,
        args.ibv_name,
        args.ibv_port,
        camera,
        camera_mode,
        args.frame_limit,
    )
    application.config(args.configuration)
    # Run it.
    hololink = hololink_channel.hololink()
    hololink.start()

    if not args.skip_reset:
        hololink.reset()
    if args.ptp_sync:
        logging.debug("Waiting for PTP sync.")
        if not hololink.ptp_synchronize():
            logging.error("Failed to synchronize PTP; ignoring.")
        else:
            logging.debug("PTP synchronized.")

    program_mipi_phy(0, 2, 270, hololink)
    program_mipi_phy(1, 2, 270, hololink)

    camera.setup_clock()
    time.sleep(1)
    camera.configure(camera_mode)
    if args.pattern is not None:
        camera.test_pattern(args.pattern)

    logging.info("Calling run")
    application.run()

    hololink.stop()

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS


if __name__ == "__main__":
    main()
