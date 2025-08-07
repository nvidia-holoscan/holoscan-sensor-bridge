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
import logging
import os

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
        frame_number = metadata.get("frame_number", 0)
        image_timestamp_ns = metadata.get("timestamp_ns", 0)
        image_timestamp_s = metadata.get("timestamp_s", 0)
        image_timestamp_s += image_timestamp_ns * SEC_PER_NS
        logging.info(f"{frame_number=} {image_timestamp_s=:0.6f}")


class HoloscanApplication(holoscan.core.Application):
    def __init__(
        self,
        cuda_context,
        cuda_device_ordinal,
        hololink_channel,
        ibv_name,
        ibv_port,
        imu,
        imu_samples_per_frame,
        frame_limit,
    ):
        logging.info("__init__")
        super().__init__()
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._hololink_channel = hololink_channel
        self._ibv_name = ibv_name
        self._ibv_port = ibv_port
        self._frame_limit = frame_limit
        self._imu_samples_per_frame = imu_samples_per_frame
        self._imu = imu

    def compose(self):
        self._count = holoscan.conditions.CountCondition(
            self,
            name="count",
            count=self._frame_limit,
        )

        bytes_per_sample = 16
        self._frame_size = self._imu_samples_per_frame * bytes_per_sample
        receiver_operator = hololink_module.operators.RoceReceiverOp(
            self,
            self._count,
            name="receiver",
            frame_size=self._frame_size,
            frame_context=self._cuda_context,
            ibv_name=self._ibv_name,
            ibv_port=self._ibv_port,
            hololink_channel=self._hololink_channel,
            device=self._imu,
        )
        profiler = ProfilerOperator(
            self,
            name="profiler",
            samples_per_frame=self._imu_samples_per_frame,
        )

        self.add_flow(receiver_operator, profiler, {("output", "input")})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--accelerometer-rate",
        type=float,
        choices=[12.5, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0],
        default=400.0,
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
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=300,
        help="Stop after receiving this many frames.",
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
    # IMU is on channel 2
    channel_metadata_2 = hololink_module.Metadata(channel_metadata)
    hololink_module.DataChannel.use_sensor(channel_metadata_2, 2)
    logging.info(f"{channel_metadata_2=}")
    hololink_channel = hololink_module.DataChannel(channel_metadata_2)
    hololink = hololink_channel.hololink()
    # Get a handle to the IMU
    imu = hololink_module.sensors.vb1940.Imu(hololink)
    imu_samples_per_frame = 10
    # Set up the application
    application = HoloscanApplication(
        cu_context,
        cu_device_ordinal,
        hololink_channel,
        args.ibv_name,
        args.ibv_port,
        imu,
        imu_samples_per_frame,
        args.frame_limit,
    )

    hololink = hololink_channel.hololink()
    hololink.start()  # Start the Hololink device before configuring IMU
    hololink.reset()
    logging.debug("Waiting for PTP sync.")
    if not hololink.ptp_synchronize():
        raise ValueError("Failed to synchronize PTP.")
    logging.debug("PTP synchronized.")
    imu.configure(
        samples_per_frame=imu_samples_per_frame,
        accelerometer_rate=args.accelerometer_rate,
        gyroscope_rate=args.gyroscope_rate,
    )
    logging.info("Calling run")
    application.run()
    hololink.stop()

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS


if __name__ == "__main__":
    main()
