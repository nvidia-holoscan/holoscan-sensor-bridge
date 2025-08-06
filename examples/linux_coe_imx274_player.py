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

#
# This example displays live IMX274 video from one channel using
# the Linux socket emulation layer with P1722B COE packets.
# This program requires at least one command-line argument, which
# is the name of the network interface to listen to for COE packets.
# Those messages are sent using L2 messages with a VETH tag of 0xCCC,
# so to enable reception of those messages, create a new
# network interface that receives those veth messages:
#
#   N0=enP5p3s0f0np0  # Or whatever network you're attached to
#   # this will be our new VETH specific interface
#   E0=coe.0
#   sudo ip link add link $N0 name $E0 type vlan id 3276
#   sudo ifconfig $E0 up
#
# (Common convention is to name the VETH interface as "(device-name).(vlan-id)"
# but e.g. "enP5p3s0f0np0.2376" exceeds the permissible length of
# the device name, thus the choice of "coe.0" here.)
#
# Once the veth device is running, you can run this example:
#
#   python3 examples/linux_coe_imx274_player.py --coe-interface=$E0
#
#

import argparse
import ctypes
import datetime
import logging
import os

import holoscan
from cuda import cuda

import hololink as hololink_module

MS_PER_SEC = 1000.0
US_PER_SEC = 1000.0 * MS_PER_SEC
NS_PER_SEC = 1000.0 * US_PER_SEC
SEC_PER_NS = 1.0 / NS_PER_SEC


def get_timestamp(metadata, name):
    s = metadata[f"{name}_s"]
    f = metadata[f"{name}_ns"]
    f *= SEC_PER_NS
    return s + f


class TimingReportOperator(holoscan.core.Operator):
    def __init__(
        self,
        *args,
        skip_frames=10,  # Allow the system to settle
        report_frames=10,  # Show this many timing reports
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._skip_frames = skip_frames
        self._report_frames = report_frames
        self._frames = 0

    def setup(self, spec):
        logging.info("setup")
        spec.input("input")

    def compute(self, op_input, op_output, context):
        # What time is it now?
        complete_timestamp = datetime.datetime.now(datetime.UTC)

        #
        _ = op_input.receive("input")
        self._frames += 1
        if self._frames < self._skip_frames:
            return
        if self._frames >= (self._skip_frames + self._report_frames):
            return

        # Show the timing for this frame.
        metadata = self.metadata
        frame_number = metadata.get("frame_number", 0)

        # frame_start_s is the time that the first data arrived at the FPGA;
        # the network receiver calls this "timestamp".
        frame_start_s = get_timestamp(metadata, "timestamp")

        # After the FPGA sends the last sensor data packet for a frame, it follows
        # that with a 128-byte metadata packet.  This timestamp (which the network
        # receiver calls "metadata") is the time at which the FPGA sends that
        # packet; so it's the time immediately after the the last byte of sensor
        # data in this window.  The difference between frame_start_s and frame_end_s
        # is how long it took for the sensor to produce enough data for a complete
        # frame.
        frame_end_s = get_timestamp(metadata, "metadata")

        # frame_time shows how long it takes for the data to transfer
        # from the sensor to the FPGA.
        frame_time = frame_end_s - frame_start_s

        # overall_time is the time the entire pipeline took, from the first
        # byte arriving at FPGA until after visualization is complete.
        overall_time = complete_timestamp.timestamp() - frame_start_s
        logging.info(f"{frame_number=} {frame_time=:.4f} {overall_time=:.4f}")


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
        coe_interface,
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
        self._coe_interface = coe_interface
        # This is a control for HSDK
        self.is_metadata_enabled = True

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
        )
        self._camera.configure_converter(csi_to_bayer_operator)

        frame_size = csi_to_bayer_operator.get_csi_length()
        frame_context = self._cuda_context
        # Each camera sharing a network connection must use
        # a unique channel number from 0..63.
        coe_channel = 1
        pixel_width = self._camera._width
        receiver_operator = hololink_module.operators.LinuxCoeReceiverOp(
            self,
            condition,
            name="receiver",
            frame_size=frame_size,
            frame_context=frame_context,
            hololink_channel=self._hololink_channel,
            device=self._camera,
            coe_interface=self._coe_interface,
            pixel_width=pixel_width,
            coe_channel=coe_channel,
        )

        pixel_format = self._camera.pixel_format()
        bayer_format = self._camera.bayer_format()
        image_processor_operator = hololink_module.operators.ImageProcessorOp(
            self,
            name="image_processor",
            # Optical black value for imx274 is 50
            optical_black=50,
            bayer_format=bayer_format.value,
            pixel_format=pixel_format.value,
        )

        rgba_components_per_pixel = 4
        bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="pool",
            # storage_type of 1 is device memory
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
            fullscreen=self._fullscreen,
            headless=self._headless,
            framebuffer_srgb=True,
            enable_camera_pose_output=True,
            camera_pose_output_type="extrinsics_model",
        )

        timer = TimingReportOperator(
            self,
            name="timer",
        )

        #
        self.add_flow(receiver_operator, csi_to_bayer_operator, {("output", "input")})
        self.add_flow(
            csi_to_bayer_operator, image_processor_operator, {("output", "input")}
        )
        self.add_flow(image_processor_operator, demosaic, {("output", "receiver")})
        self.add_flow(demosaic, visualizer, {("transmitter", "receivers")})
        self.add_flow(visualizer, timer, {("camera_pose_output", "input")})


def main():
    parser = argparse.ArgumentParser()
    modes = hololink_module.sensors.imx274.imx274_mode.Imx274_Mode
    mode_choices = [mode.value for mode in modes]
    mode_help = " ".join([f"{mode.value}:{mode.name}" for mode in modes])
    parser.add_argument(
        "--camera-mode",
        type=int,
        choices=mode_choices,
        default=mode_choices[0],
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
    parser.add_argument(
        "--coe-interface",
        required=True,
        help="Name of interface connected to VETH 3276 (0xCCC).",
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
    # Get a handle to the Hololink device
    channel_metadata = hololink_module.Enumerator.find_channel(channel_ip=args.hololink)
    hololink_channel = hololink_module.DataChannel(channel_metadata)
    # Get a handle to the camera
    camera = hololink_module.sensors.imx274.dual_imx274.Imx274Cam(
        hololink_channel, expander_configuration=args.expander_configuration
    )
    camera_mode = hololink_module.sensors.imx274.imx274_mode.Imx274_Mode(
        args.camera_mode
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
        args.coe_interface,
    )
    application.config(args.configuration)
    # Run it.
    hololink = hololink_channel.hololink()
    hololink.start()
    hololink.reset()
    # Sync PTP; otherwise the timestamps we get from HSB aren't synchronized with us.
    logging.debug("Waiting for PTP sync.")
    if not hololink.ptp_synchronize():
        raise ValueError("Failed to synchronize PTP.")
    else:
        logging.debug("PTP synchronized.")
    camera.setup_clock()
    camera.configure(camera_mode)
    camera.set_digital_gain_reg(0x4)
    if args.pattern is not None:
        camera.test_pattern(args.pattern)
    application.run()
    hololink.stop()

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS


if __name__ == "__main__":
    main()
