# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# See README.md for detailed information.

# Streams raw accelerometer/gyroscope samples from the VB1940's onboard IMU over RoCE and logs
# each frame's decoded samples. hololink.sensors.vb1940.Imu is a plain Python device-handle
# class (I2C configuration + start()/stop() streaming control, already fully bound) -- used
# exactly like a camera sensor's device object elsewhere in this porting effort: its start()/
# stop() are RoceReceiverOp's device_start/device_stop callbacks, not any kind of new operator.
# No C++ work needed for this app at all.
#
#   RoceReceiverOp -> ProfilerOp
#
# --frame-limit is the receiver's frame_limit parameter: the receiver ends the run after that
# many frames. HSDK5 has no CountCondition; this replaces the CountCondition of the old app.

import argparse
import logging
import sys
from datetime import timedelta

import capture_metadata
import cuda.bindings.driver as cuda_driver
import cupy as cp

# hololink/__init__.py never imports sensors; without this, hl.sensors.vb1940.Imu below raises
# AttributeError ("module 'hololink' has no attribute 'sensors'") despite sensors/__init__.py's
# own lazy-loading __getattr__ supporting "vb1940" -- that __getattr__ only runs once something
# has already bound `sensors` onto the hololink package object, which only a real import does.
import hololink.sensors.vb1940  # noqa: F401
import holoscan as hs

import hololink as hl

MS_PER_SEC = 1000.0
US_PER_SEC = 1000.0 * MS_PER_SEC
NS_PER_SEC = 1000.0 * US_PER_SEC
SEC_PER_NS = 1.0 / NS_PER_SEC

BYTES_PER_SAMPLE = 16

# Matches hololink::operators::kMetadataPairingWindow (capture_metadata_utils.hpp): the
# receiver stamps both ports with an identical capture_time, so this bounded-window
# OnAligned+Exact{} pairs them precisely.
_METADATA_PAIRING_WINDOW = timedelta(microseconds=500)


class ProfilerOp(hs.Operator):
    """Decode and log raw IMU samples, correlated with the receiver's own frame metadata."""

    in_tensor = hs.Input("in_tensor", hs.payload.Tensor)
    in_metadata = hs.Input("in_metadata", capture_metadata.CaptureMetadata)

    def setup(self, spec):
        spec.input(self.in_tensor, queue_depth=4)
        spec.input(self.in_metadata, queue_depth=4)

    def contract(self):
        return (
            hs.Contract()
            .trigger(
                hs.OnAligned(
                    ports=[self.in_tensor, self.in_metadata],
                    window=_METADATA_PAIRING_WINDOW,
                )
            )
            .input(self.in_tensor, hs.Exact())
            .input(self.in_metadata, hs.Exact())
        )

    def compute(self, context):
        tensor_sample = self.in_tensor.receive()
        metadata_sample = self.in_metadata.receive()
        if tensor_sample.is_missing or metadata_sample.is_missing:
            return

        # A received HSDK5 tensor is read-only, and HSDK5 exports it only as versioned DLPack
        # on a real CUDA stream (cupy's default legacy stream is refused with kNotSupported).
        # Import it without a copy on a private non-blocking stream, and synchronize before
        # the array is used.
        with cp.cuda.Stream(non_blocking=True) as stream:
            capsule = tensor_sample.data.__dlpack__(
                stream=stream.ptr,
                max_version=(1, 0),  # Versioned DLPack
                copy=False,  # No pixel-buffer copy
            )
            cp_frame = cp.from_dlpack(capsule)
            stream.synchronize()
        # The HSDK5 Deserializer takes bytes only.
        data = cp_frame.astype(cp.uint8, copy=False).get().tobytes()
        deserializer = hl.Deserializer(data)
        while deserializer.remaining() > 0:
            timestamp_ns = deserializer.next_uint32_le()
            timestamp_s = deserializer.next_uint32_le()
            timestamp_s += timestamp_ns * SEC_PER_NS
            tag = deserializer.next_uint8()
            sensor_data = ["%02X" % x for x in deserializer.next_buffer(6)]
            ff = deserializer.next_uint8()
            logging.info(f"{timestamp_s=} {tag=:#x} {sensor_data=}")
            if ff != 0xFF:
                raise RuntimeError(
                    f"Unexpected IMU frame terminator {ff:#x}; expected 0xFF."
                )

        frame = metadata_sample.data
        image_timestamp_s = frame.timestamp_s + frame.timestamp_ns * SEC_PER_NS
        logging.info(f"frame_number={frame.frame_number} {image_timestamp_s=:0.6f}")


def _cuda_check(result):
    err = result[0]
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        _, description = cuda_driver.cuGetErrorString(err)
        raise RuntimeError(f"CUDA error: {description}")
    if len(result) == 2:
        return result[1]
    return result[1:]


def run(
    hololink_ip, ibv_name, ibv_port, accelerometer_rate, gyroscope_rate, frame_limit
):
    cuda_device_ordinal = 0
    _cuda_check(cuda_driver.cuInit(0))
    cu_device = _cuda_check(cuda_driver.cuDeviceGet(cuda_device_ordinal))
    cu_context = _cuda_check(cuda_driver.cuDevicePrimaryCtxRetain(cu_device))
    cuda_driver.cuCtxSetCurrent(cu_context)

    metadata = hl.find_channel(hololink_ip)
    # IMU is on channel 2.
    hl.use_sensor(metadata, 2)
    channel = hl.DataChannel(metadata)
    hololink_dev = channel.hololink()
    hololink_dev.start()  # Start the Hololink device before configuring the IMU.
    try:
        hololink_dev.reset()
        logging.debug("Waiting for PTP sync.")
        if not hololink_dev.ptp_synchronize():
            raise ValueError("Failed to synchronize PTP.")
        logging.debug("PTP synchronized.")

        imu = hl.sensors.vb1940.Imu(hololink_dev)
        imu_samples_per_frame = 10
        imu.configure(
            samples_per_frame=imu_samples_per_frame,
            accelerometer_rate=accelerometer_rate,
            gyroscope_rate=gyroscope_rate,
        )

        graph = hs.Graph("vb1940_imu_player")
        receiver = graph.op(
            hl.operators.RoceReceiverOp,
            "receiver",
            data_channel=channel,
            frame_context=int(cu_context),
            device_start=imu.start,
            device_stop=imu.stop,
            frame_size=imu_samples_per_frame * BYTES_PER_SAMPLE,
            ibv_name=ibv_name,
            ibv_port=ibv_port,
            frame_limit=frame_limit or 0,
        )
        profiler = graph.op(ProfilerOp, "profiler")
        graph.add_flow(receiver.tensor, profiler.in_tensor, queue_depth=4)
        graph.add_flow(receiver.metadata, profiler.in_metadata, queue_depth=4)

        clock_key = graph.add_clock(hs.RealtimeClock, "clock")
        graph.set_default_clock(clock_key)

        options = hs.CompileOptions()
        options.deployment.bind_tensor_output_device(
            operator_path="receiver",
            output_port="tensor",
            device=hs.execution.DeviceId(cuda_device_ordinal),
        )

        plan = hs.compile(graph, options)
        if not plan.ok:
            print("Graph compilation failed:", file=sys.stderr)
            for diag in plan.diagnostics:
                print(f"  {diag}", file=sys.stderr)
            return 1

        logging.info("Calling run")
        with hs.run_async(plan) as session:
            # With --frame-limit, the receiver ends the run after frame_limit frames.
            session.wait()
    finally:
        hololink_dev.stop()

    _cuda_check(cuda_driver.cuDevicePrimaryCtxRelease(cu_device))
    return 0


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
        "--ibv-name",
        default="",
        help="IBV device to use (default: the device on the board's network interface)",
    )
    parser.add_argument(
        "--ibv-port",
        type=int,
        default=1,
        help="Port number of IBV device (with --ibv-name)",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=300,
        help="Stop after receiving this many frames.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    logging.info("Initializing.")

    sys.exit(
        run(
            args.hololink,
            args.ibv_name,
            args.ibv_port,
            args.accelerometer_rate,
            args.gyroscope_rate,
            args.frame_limit,
        )
    )


if __name__ == "__main__":
    main()
