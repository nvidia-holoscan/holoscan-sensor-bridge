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
import os
import time

import cupy as cp
import holoscan
from cuda import cuda

import hololink as hololink_module

MS_PER_SEC = 1000.0
US_PER_SEC = 1000.0 * MS_PER_SEC
NS_PER_SEC = 1000.0 * US_PER_SEC
SEC_PER_NS = 1.0 / NS_PER_SEC


class ProfilerOperator(holoscan.core.Operator):
    def __init__(
        self,
        *args,
        samples_per_frame=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._samples_per_frame = samples_per_frame

    def setup(self, spec):
        logging.info("setup")
        spec.input("input")

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("input")
        tensor = in_message.get("")
        data = cp.asarray(tensor, dtype=cp.uint8).get()
        deserializer = hololink_module.Deserializer(data)
        for _ in range(self._samples_per_frame):
            timestamp_ns = deserializer.next_uint32_le()
            timestamp_s = deserializer.next_uint32_le()
            timestamp_s += timestamp_ns * SEC_PER_NS
            tag = deserializer.next_uint8()
            sensor_data = ["%02X" % x for x in deserializer.next_buffer(6)]
            ff = deserializer.next_uint8()
            logging.info(f"{timestamp_s=} {tag=:#x} {sensor_data=}")
            assert ff == 0xFF
        #
        metadata = self.metadata
        imu_frame_number = metadata.get("imu_frame_number", 0)
        imu_image_timestamp_ns = metadata.get("imu_timestamp_ns", 0)
        imu_image_timestamp_s = metadata.get("imu_timestamp_s", 0)
        imu_image_timestamp_s += imu_image_timestamp_ns * SEC_PER_NS
        logging.info(f"{imu_frame_number=} {imu_image_timestamp_s=:0.6f}")


class HoloscanApplication(holoscan.core.Application):
    def __init__(
        self,
        headless,
        cuda_context,
        cuda_device_ordinal,
        ibv_name,
        ibv_port,
        hololink_channel_left,
        camera_left,
        camera_mode_left,
        hololink_channel_right,
        camera_right,
        camera_mode_right,
        hololink_channel_imu,
        imu,
        frame_limit,
    ):
        logging.info("__init__")
        super().__init__()
        self._headless = headless
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._ibv_name = ibv_name
        self._ibv_port = ibv_port
        self._hololink_channel_left = hololink_channel_left
        self._camera_left = camera_left
        self._camera_mode_left = camera_mode_left
        self._hololink_channel_right = hololink_channel_right
        self._camera_right = camera_right
        self._camera_mode_right = camera_mode_right
        self._hololink_channel_imu = hololink_channel_imu
        self._imu = imu
        self._frame_limit = frame_limit
        self.enable_metadata(True)

    def compose(self):
        self._condition_left = holoscan.conditions.CountCondition(
            self,
            name="condition_left",
            count=self._frame_limit,
        )
        self._condition_right = holoscan.conditions.CountCondition(
            self,
            name="condition_right",
            count=self._frame_limit,
        )
        self._condition_imu = holoscan.conditions.CountCondition(
            self,
            name="condition_imu",
            count=self._frame_limit,
        )

        self._camera_left.set_mode(self._camera_mode_left)
        self._camera_right.set_mode(self._camera_mode_right)

        csi_to_bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="csi_to_bayer_pool",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=self._camera_left.width()
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera_left.height(),
            num_blocks=6,
        )
        csi_to_bayer_operator_left = hololink_module.operators.CsiToBayerOp(
            self,
            name="csi_to_bayer_left",
            allocator=csi_to_bayer_pool,
            cuda_device_ordinal=self._cuda_device_ordinal,
            out_tensor_name="left",
        )
        self._camera_left.configure_converter(csi_to_bayer_operator_left)
        csi_to_bayer_operator_right = hololink_module.operators.CsiToBayerOp(
            self,
            name="csi_to_bayer_right",
            allocator=csi_to_bayer_pool,
            cuda_device_ordinal=self._cuda_device_ordinal,
            out_tensor_name="right",
        )
        self._camera_right.configure_converter(csi_to_bayer_operator_right)

        camera_frame_size = csi_to_bayer_operator_left.get_csi_length()
        assert camera_frame_size == csi_to_bayer_operator_right.get_csi_length()

        frame_context = self._cuda_context
        receiver_operator_left = hololink_module.operators.RoceReceiverOp(
            self,
            self._condition_left,
            name="receiver_left",
            frame_size=camera_frame_size,
            frame_context=frame_context,
            ibv_name=self._ibv_name,
            ibv_port=self._ibv_port,
            hololink_channel=self._hololink_channel_left,
            device=self._camera_left,
            rename_metadata=lambda name: f"left_{name}",
        )

        #
        receiver_operator_right = hololink_module.operators.RoceReceiverOp(
            self,
            self._condition_right,
            name="receiver_right",
            frame_size=camera_frame_size,
            frame_context=frame_context,
            ibv_name=self._ibv_name,
            ibv_port=self._ibv_port,
            hololink_channel=self._hololink_channel_right,
            device=self._camera_right,
            rename_metadata=lambda name: f"right_{name}",
        )

        bytes_per_sample = 16
        self._imu_frame_size = self._imu.samples_per_frame() * bytes_per_sample
        imu_receiver_operator = hololink_module.operators.RoceReceiverOp(
            self,
            self._condition_imu,
            name="imu_receiver",
            frame_size=self._imu_frame_size,
            frame_context=self._cuda_context,
            ibv_name=self._ibv_name,
            ibv_port=self._ibv_port,
            hololink_channel=self._hololink_channel_imu,
            device=self._imu,
            rename_metadata=lambda name: f"imu_{name}",
        )

        bayer_format = self._camera_left.bayer_format()
        assert bayer_format == self._camera_right.bayer_format()
        pixel_format = self._camera_left.pixel_format()
        assert pixel_format == self._camera_right.pixel_format()
        image_processor_left = hololink_module.operators.ImageProcessorOp(
            self,
            name="image_processor_left",
            # Optical black value for vb1940 is 8 for raw10
            optical_black=8,
            bayer_format=bayer_format.value,
            pixel_format=pixel_format.value,
        )
        image_processor_right = hololink_module.operators.ImageProcessorOp(
            self,
            name="image_processor_right",
            # Optical black value for vb1940 is 8 for raw10
            optical_black=8,
            bayer_format=bayer_format.value,
            pixel_format=pixel_format.value,
        )

        rgba_components_per_pixel = 4
        bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="bayer_pool",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=self._camera_left.width()
            * rgba_components_per_pixel
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera_left.height(),
            num_blocks=6,
        )
        demosaic_left = holoscan.operators.BayerDemosaicOp(
            self,
            name="demosaic_left",
            pool=bayer_pool,
            generate_alpha=True,
            alpha_value=65535,
            bayer_grid_pos=bayer_format.value,
            interpolation_mode=0,
            in_tensor_name="left",
            out_tensor_name="left",
        )
        demosaic_right = holoscan.operators.BayerDemosaicOp(
            self,
            name="demosaic_right",
            pool=bayer_pool,
            generate_alpha=True,
            alpha_value=65535,
            bayer_grid_pos=bayer_format.value,
            interpolation_mode=0,
            in_tensor_name="right",
            out_tensor_name="right",
        )

        left_spec = holoscan.operators.HolovizOp.InputSpec(
            "left", holoscan.operators.HolovizOp.InputType.COLOR
        )
        left_spec_view = holoscan.operators.HolovizOp.InputSpec.View()
        left_spec_view.offset_x = 0
        left_spec_view.offset_y = 0
        left_spec_view.width = 0.5
        left_spec_view.height = 1
        left_spec.views = [left_spec_view]

        right_spec = holoscan.operators.HolovizOp.InputSpec(
            "right", holoscan.operators.HolovizOp.InputType.COLOR
        )
        right_spec_view = holoscan.operators.HolovizOp.InputSpec.View()
        right_spec_view.offset_x = 0.5
        right_spec_view.offset_y = 0
        right_spec_view.width = 0.5
        right_spec_view.height = 1
        right_spec.views = [right_spec_view]

        window_height = 200
        window_width = 600  # for the pair
        window_title = "VB1940 player"
        visualizer = holoscan.operators.HolovizOp(
            self,
            name="holoviz",
            headless=self._headless,
            framebuffer_srgb=True,
            tensors=[left_spec, right_spec],
            height=window_height,
            width=window_width,
            window_title=window_title,
        )

        profiler = ProfilerOperator(
            self,
            name="profiler",
            samples_per_frame=self._imu.samples_per_frame(),
        )

        self.add_flow(
            receiver_operator_left,
            csi_to_bayer_operator_left,
            {("output", "input")},
        )
        self.add_flow(
            receiver_operator_right,
            csi_to_bayer_operator_right,
            {("output", "input")},
        )
        self.add_flow(
            csi_to_bayer_operator_left, image_processor_left, {("output", "input")}
        )
        self.add_flow(
            csi_to_bayer_operator_right, image_processor_right, {("output", "input")}
        )
        self.add_flow(image_processor_left, demosaic_left, {("output", "receiver")})
        self.add_flow(image_processor_right, demosaic_right, {("output", "receiver")})
        self.add_flow(demosaic_left, visualizer, {("transmitter", "receivers")})
        self.add_flow(demosaic_right, visualizer, {("transmitter", "receivers")})
        self.add_flow(imu_receiver_operator, profiler, {("output", "input")})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--accelerometer-rate",
        type=float,
        choices=[12.5, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0],
        default=400,
        help="Accelerometer data rate",
    )
    parser.add_argument(
        "--gyroscope-rate",
        type=int,
        choices=[100, 200, 400, 1000, 2000],
        default=400,
        help="Gyroscope data rate",
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
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=300,
        help="Stop after receiving this many frames.",
    )
    parser.add_argument(
        "--frequency",
        type=int,
        default=30,
        help="VSYNC frequency in Hz (10, 30, 60, 90, 120). Default is 30Hz",
    )
    parser.add_argument(
        "--camera-mode",
        type=int,
        default=hololink_module.sensors.vb1940.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS.value,
        help="VB1940 mode",
    )
    parser.add_argument(
        "--exp",
        type=int,
        default=256,
        help="set EXPOSURE duration in lines, RANGE(4 to 65535). Default line value is 29.70usec",
    )
    parser.add_argument(
        "--gain",
        type=int,
        default=0,
        help="Set Analog Gain, RANGE(0 to 12). Default is 0. Equation is (16/(16-gain)",
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
    logging.debug(f"{channel_metadata=}")
    # Now make separate connection metadata for left and right; and set them to
    # use sensor 0 and 1 respectively; IMU is on 2.  This will borrow the data plane
    # configuration we found on that interface.
    channel_metadata_left = hololink_module.Metadata(channel_metadata)
    hololink_module.DataChannel.use_sensor(channel_metadata_left, 0)
    channel_metadata_right = hololink_module.Metadata(channel_metadata)
    hololink_module.DataChannel.use_sensor(channel_metadata_right, 1)
    imu_channel_metadata = hololink_module.Metadata(channel_metadata)
    hololink_module.DataChannel.use_sensor(imu_channel_metadata, 2)
    #
    hololink_channel_left = hololink_module.DataChannel(channel_metadata_left)
    hololink_channel_right = hololink_module.DataChannel(channel_metadata_right)
    hololink_channel_imu = hololink_module.DataChannel(imu_channel_metadata)
    hololink = hololink_channel_imu.hololink()
    assert hololink is hololink_channel_left.hololink()
    assert hololink is hololink_channel_right.hololink()
    # Get a handle to cameras
    vsync = hololink.ptp_pps_output(args.frequency)
    camera_left = hololink_module.sensors.vb1940.Vb1940Cam(
        hololink_channel_left,
        vsync=vsync,
    )
    camera_right = hololink_module.sensors.vb1940.Vb1940Cam(
        hololink_channel_right,
        vsync=vsync,
    )
    camera_mode = hololink_module.sensors.vb1940.Vb1940_Mode(args.camera_mode)
    # Get a handle to the IMU
    imu = hololink_module.sensors.vb1940.Imu(hololink)
    # Connect to the Hololink device before configuring sensors
    hololink.start()
    hololink.reset()
    logging.debug("Waiting for PTP sync.")
    if not hololink.ptp_synchronize():
        raise ValueError("Failed to synchronize PTP.")
    logging.debug("PTP synchronized.")
    # Initialize cameras.
    hololink.write_uint32(0x8, 0x0)
    camera_left.setup_clock()  # this also sets camera_right's clock
    hololink.write_uint32(0x8, 0x3)
    time.sleep(100 / 1000)

    camera_left.get_register_32(0x0000)  # DEVICE_MODEL_ID:"S940"(ASCII code:0x53393430)
    camera_left.get_register_32(0x0734)  # EXT_CLOCK(25MHz = 0x017d7840)
    camera_left.configure(camera_mode)
    camera_left.set_analog_gain_reg(args.gain)  # Gain value has to be int
    camera_left.set_exposure_reg(args.exp)  # Exposure value has to be int

    camera_right.get_register_32(
        0x0000
    )  # DEVICE_MODEL_ID:"S940"(ASCII code:0x53393430)
    camera_right.get_register_32(0x0734)  # EXT_CLOCK(25MHz = 0x017d7840)
    camera_right.configure(camera_mode)
    camera_right.set_analog_gain_reg(args.gain)  # Gain value has to be int
    camera_right.set_exposure_reg(args.exp)  # Exposure value has to be int
    #
    imu.configure(
        samples_per_frame=10,
        accelerometer_rate=args.accelerometer_rate,
        gyroscope_rate=args.gyroscope_rate,
    )

    # Set up the application
    application = HoloscanApplication(
        args.headless,
        cu_context,
        cu_device_ordinal,
        args.ibv_name,
        args.ibv_port,
        hololink_channel_left,
        camera_left,
        camera_mode,
        hololink_channel_right,
        camera_right,
        camera_mode,
        hololink_channel_imu,
        imu,
        args.frame_limit,
    )

    logging.info("Calling run")
    application.run()
    hololink.stop()

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS


if __name__ == "__main__":
    main()
