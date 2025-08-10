# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import datetime
import logging
import math

import cupy as cp
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


def record_times(recorder_queue, metadata):
    #
    now = datetime.datetime.utcnow()
    #
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

    # received_timestamp_s is the host time after the background thread woke up
    # with the nofication that a frame of data was available.  This shows how long
    # it took for the CPU to actually run the backtground user-mode thread where it observes
    # the end-of-frame.  This background thread sets a flag that will wake up
    # the pipeline network receiver operator.
    received_timestamp_s = get_timestamp(metadata, "received")

    # operator_timestamp_s is the time when the next pipeline element woke up--
    # the next operator after the network receiver.  This is used to compute
    # how much time overhead is required for the pipeline to actually receive
    # sensor data.
    operator_timestamp_s = get_timestamp(metadata, "operator_timestamp")

    # complete_timestamp_s is the time when visualization finished.
    complete_timestamp_s = get_timestamp(metadata, "complete_timestamp")

    recorder_queue.append(
        (
            now,
            frame_start_s,
            frame_end_s,
            received_timestamp_s,
            operator_timestamp_s,
            complete_timestamp_s,
            frame_number,
        )
    )


def save_timestamp(metadata, name, timestamp):
    # This method works around the fact that we can't store
    # datetime objects in metadata.
    f, s = math.modf(timestamp.timestamp())
    metadata[f"{name}_s"] = int(s)
    metadata[f"{name}_ns"] = int(f * NS_PER_SEC)


class InstrumentedTimeProfiler(holoscan.core.Operator):
    def __init__(
        self,
        *args,
        recorder_queue=None,
        operator_name="operator",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._recorder_queue = recorder_queue
        self._operator_name = operator_name

    def setup(self, spec):
        logging.info("setup")
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        # What time is it now?
        operator_timestamp = datetime.datetime.utcnow()

        in_message = op_input.receive("input")
        cp_frame = cp.asarray(in_message.get(""))
        #
        save_timestamp(
            self.metadata, self._operator_name + "_timestamp", operator_timestamp
        )
        op_output.emit({"": cp_frame}, "output")


class MonitorOperator(holoscan.core.Operator):
    def __init__(
        self,
        *args,
        recorder_queue=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._recorder_queue = recorder_queue

    def setup(self, spec):
        logging.info("setup")
        spec.input("input")

    def compute(self, op_input, op_output, context):
        # What time is it now?
        complete_timestamp = datetime.datetime.utcnow()

        _ = op_input.receive("input")
        #
        save_timestamp(self.metadata, "complete_timestamp", complete_timestamp)
        record_times(self._recorder_queue, self.metadata)


class MicroApplication(holoscan.core.Application):
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
        frame_limit,
        recorder_queue,
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
        self._frame_limit = frame_limit
        self._recorder_queue = recorder_queue
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
        profiler = InstrumentedTimeProfiler(
            self,
            name="profiler",
            recorder_queue=self._recorder_queue,
        )

        bayer_format = self._camera.bayer_format()
        pixel_format = self._camera.pixel_format()
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
        #
        monitor = MonitorOperator(
            self,
            name="monitor",
            recorder_queue=self._recorder_queue,
        )
        #
        self.add_flow(receiver_operator, profiler, {("output", "input")})
        self.add_flow(profiler, csi_to_bayer_operator, {("output", "input")})
        self.add_flow(
            csi_to_bayer_operator, image_processor_operator, {("output", "input")}
        )
        self.add_flow(image_processor_operator, demosaic, {("output", "receiver")})
        self.add_flow(demosaic, visualizer, {("transmitter", "receivers")})
        self.add_flow(visualizer, monitor, {("camera_pose_output", "input")})

    def _terminate(self, recorded_timestamps):
        self._ok.disable_tick()
        global timestamps
        timestamps = recorded_timestamps


def main():
    parser = argparse.ArgumentParser()
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

    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level to display",
    )
    parser.add_argument(
        "--cam",
        type=int,
        default=0,
        choices=(0, 1),
        help="which camera to stream: 0 to stream camera connected to j14 or 1 to stream camera connected to j17 (default is 0)",
    )
    parser.add_argument(
        "--pattern",
        action="store_true",
        help="Configure to display a test pattern.",
    )
    parser.add_argument(
        "--exposure",
        type=int,
        default=0x05,
        help="Configure exposure.",
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
        "--resolution",
        default="4k",
        help="4k or 1080p",
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
    cu_result, cu_context = cuda.cuCtxCreate(0, cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS

    # Get a handle to the Hololink device
    if args.cam == 0:
        channel_metadata = hololink_module.Enumerator.find_channel(
            channel_ip="192.168.0.2"
        )
    elif args.cam == 1:
        channel_metadata = hololink_module.Enumerator.find_channel(
            channel_ip="192.168.0.3"
        )
    else:
        raise Exception(f"Unexpected camera={args.cam}")

    hololink_channel = hololink_module.DataChannel(channel_metadata)
    # Get a handle to the camera
    camera = hololink_module.sensors.imx477.Imx477(
        hololink_channel, args.cam, args.resolution
    )

    recorder_queue = []

    # Set up the application
    application = MicroApplication(
        args.headless,
        args.fullscreen,
        cu_context,
        cu_device_ordinal,
        hololink_channel,
        args.ibv_name,
        args.ibv_port,
        camera,
        args.frame_limit,
        recorder_queue,
    )

    # Run it.
    hololink = hololink_channel.hololink()
    hololink.start()
    hololink.reset()

    ptp_sync_timeout_s = 10
    ptp_sync_timeout = hololink_module.Timeout(ptp_sync_timeout_s)
    logging.debug("Waiting for PTP sync.")
    if not hololink.ptp_synchronize(ptp_sync_timeout):
        raise ValueError(
            f"Failed to synchronize PTP after {ptp_sync_timeout_s} seconds; ignoring."
        )
    else:
        logging.debug("PTP synchronized.")

    # Configures the camera for 3840x2160, 60fps
    camera.configure()

    # IMX477 Analog gain settings function. Analog gain value range is 0-1023 in decimal (10 bits). Users are free to experiment with the register values.
    camera.set_analog_gain(0x2FF)
    camera.set_exposure_reg(args.exposure)

    if args.pattern:
        camera.set_pattern()
    application.run()
    hololink.stop()

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS

    # Report stats at the end of the application
    frame_time_dts = []
    cpu_latency_dts = []
    operator_latency_dts = []
    processing_time_dts = []
    overall_time_dts = []

    settled_timestamps = recorder_queue[5:-5]
    assert len(settled_timestamps) >= 100
    for (
        now,
        frame_start_s,
        frame_end_s,
        received_timestamp_s,
        operator_timestamp_s,
        complete_timestamp_s,
        frame_number,
    ) in settled_timestamps:

        frame_start = datetime.datetime.fromtimestamp(frame_start_s).isoformat()
        frame_end = datetime.datetime.fromtimestamp(frame_end_s).isoformat()
        received_timestamp = datetime.datetime.fromtimestamp(
            received_timestamp_s
        ).isoformat()
        operator_timestamp = datetime.datetime.fromtimestamp(
            operator_timestamp_s
        ).isoformat()
        complete_timestamp = datetime.datetime.fromtimestamp(
            complete_timestamp_s
        ).isoformat()

        frame_time_dt = frame_end_s - frame_start_s
        frame_time_dts.append(round(frame_time_dt, 4))

        cpu_latency_dt = received_timestamp_s - frame_end_s
        cpu_latency_dts.append(round(cpu_latency_dt, 4))

        operator_latency_dt = operator_timestamp_s - received_timestamp_s
        operator_latency_dts.append(round(operator_latency_dt, 4))

        processing_time_dt = complete_timestamp_s - operator_timestamp_s
        processing_time_dts.append(round(processing_time_dt, 4))

        overall_time_dt = complete_timestamp_s - frame_start_s
        overall_time_dts.append(round(overall_time_dt, 4))
        logging.debug(f"** Frame Information  for Frame Number = {frame_number}**")
        logging.debug(f"Frame Start          : {frame_start}")
        logging.debug(f"Frame End            : {frame_end}")
        logging.debug(f"Received Timestamp   : {received_timestamp}")
        logging.debug(f"Operator Timestamp   : {operator_timestamp}")
        logging.debug(f"Complete Timestamp   : {complete_timestamp}")
        logging.debug(f"Frame Time (dt)      : {frame_time_dt:.6f} s")
        logging.debug(f"CPU Latency (dt)     : {cpu_latency_dt:.6f} s")
        logging.debug(f"Operator Latency (dt): {operator_latency_dt:.6f} s")
        logging.debug(f"Processing Time (dt) : {processing_time_dt:.6f} s")
        logging.debug(f"Overall Time (dt)    : {overall_time_dt:.6f} s")

    logging.info("** Complete report: **")
    logging.info(f"{'Metric':<30}{'Min':<15}{'Max':<15}{'Avg':<15}")
    #
    ft_min_time_difference = min(frame_time_dts)
    ft_max_time_difference = max(frame_time_dts)
    ft_avg_time_difference = sum(frame_time_dts) / len(frame_time_dts)
    logging.info("Frame Time (in sec):")
    logging.info(
        f"{'Frame Time':<30}{ft_min_time_difference:<15}{ft_max_time_difference:<15}{ft_avg_time_difference:<15}"
    )
    #
    cl_min_time_difference = min(cpu_latency_dts)
    cl_max_time_difference = max(cpu_latency_dts)
    cl_avg_time_difference = sum(cpu_latency_dts) / len(cpu_latency_dts)
    logging.info("FGPA frame transfer latency (in sec):")
    logging.info(
        f"{'Frame Transfer Latency':<30}{cl_min_time_difference:<15}{cl_max_time_difference:<15}{cl_avg_time_difference:<15}"
    )
    #
    ol_min_time_difference = min(operator_latency_dts)
    ol_max_time_difference = max(operator_latency_dts)
    ol_avg_time_difference = sum(operator_latency_dts) / len(operator_latency_dts)
    logging.info("FGPA to Operator after network operator latency (in sec):")
    logging.info(
        f"{'Operator Latency':<30}{ol_min_time_difference:<15}{ol_max_time_difference:<15}{ol_avg_time_difference:<15}"
    )
    #
    pt_min_time_difference = min(processing_time_dts)
    pt_max_time_difference = max(processing_time_dts)
    pt_avg_time_difference = sum(processing_time_dts) / len(processing_time_dts)
    logging.info("Processing of frame latency (in sec):")
    logging.info(
        f"{'Processing Latency':<30}{pt_min_time_difference:<15}{pt_max_time_difference:<15}{pt_avg_time_difference:<15}"
    )
    #
    ot_min_time_difference = min(overall_time_dts)
    ot_max_time_difference = max(overall_time_dts)
    ot_avg_time_difference = sum(overall_time_dts) / len(overall_time_dts)
    logging.info("Frame start till end of SW pipeline latency (in sec):")
    logging.info(
        f"{'SW Pipeline Latency':<30}{ot_min_time_difference:<15}{ot_max_time_difference:<15}{ot_avg_time_difference:<15}"
    )


if __name__ == "__main__":
    main()
