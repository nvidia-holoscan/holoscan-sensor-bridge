# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import datetime
import logging
import math

import cupy as cp
import holoscan

COLOR_PROFILER_START_FRAME = 5


class ColorProfiler(holoscan.core.Operator):
    def __init__(self, *args, callback=None, out_tensor_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._count = 0
        self._callback = callback
        self._out_tensor_name = out_tensor_name

    def setup(self, spec):
        logging.info("setup")
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        self._count += 1
        in_message = op_input.receive("input")
        cp_frame = cp.asarray(in_message.get(""))  # cp_frame.shape is (y,x,4)
        op_output.emit({self._out_tensor_name: cp_frame}, "output")
        # Give it some time to settle
        if self._count < COLOR_PROFILER_START_FRAME:
            return
        # Compute the Y of YCrCb
        r = cp_frame[:, :, 0]
        g = cp_frame[:, :, 1]
        b = cp_frame[:, :, 2]
        y = r * 0.299 + g * 0.587 + b * 0.114
        #
        buckets, _ = cp.histogram(y, bins=16, range=(0, 65536))
        self._callback(buckets)


MS_PER_SEC = 1000.0
US_PER_SEC = 1000.0 * MS_PER_SEC
NS_PER_SEC = 1000.0 * US_PER_SEC
SEC_PER_NS = 1.0 / NS_PER_SEC


class TimeProfiler(holoscan.core.Operator):
    def __init__(
        self,
        *args,
        callback=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._count = 0
        self._callback = callback
        self._timestamps = []
        self._packets_dropped = None

    def setup(self, spec):
        logging.info("setup")
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        self._count += 1
        in_message = op_input.receive("input")
        cp_frame = cp.asarray(in_message.get(""))  # cp_frame.shape is (y,x,4)
        op_output.emit({"": cp_frame}, "output")
        #
        metadata = self.metadata
        frame_number = metadata.get("frame_number", 0)
        packets_dropped = metadata.get("packets_dropped", 0)
        if packets_dropped != self._packets_dropped:
            logging.info(f"{packets_dropped=} ({packets_dropped:#X}) {frame_number=}")
            self._packets_dropped = packets_dropped
        image_timestamp_ns = metadata.get("timestamp_ns", 0)
        image_timestamp_s = metadata.get("timestamp_s", 0)
        image_timestamp_s += image_timestamp_ns * SEC_PER_NS
        received_timestamp_s = metadata.get("received_s", 0)
        received_timestamp_ns = metadata.get("received_ns", 0)
        received_timestamp_s = received_timestamp_s + received_timestamp_ns * SEC_PER_NS
        metadata_timestamp_s = metadata.get("metadata_s", 0)
        metadata_timestamp_ns = metadata.get("metadata_ns", 0)
        metadata_timestamp_s = metadata_timestamp_s + metadata_timestamp_ns * SEC_PER_NS
        pipeline_timestamp_s = datetime.datetime.now(datetime.timezone.utc).timestamp()
        self._timestamps.append(
            (
                image_timestamp_s,
                metadata_timestamp_s,
                received_timestamp_s,
                pipeline_timestamp_s,
                frame_number,
            )
        )
        if self._count < 200:
            return
        self._callback(self._timestamps)


class WatchdogOp(holoscan.core.Operator):
    def __init__(
        self,
        *args,
        watchdog=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._watchdog = watchdog

    def setup(self, spec):
        spec.input("input")

    def compute(self, op_input, op_output, context):
        _ = op_input.receive("input")
        self._watchdog.tap()


class MonitorOperator(holoscan.core.Operator):
    def __init__(
        self,
        *args,
        # If you don't give us a callback, do nothing
        callback=lambda op, metadata: None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._callback = callback

    def setup(self, spec):
        logging.info("setup")
        spec.input("input")

    def compute(self, op_input, op_output, context):
        # What time is it now?
        complete = datetime.datetime.utcnow()

        _ = op_input.receive("input")
        #
        self.save_timestamp(self.metadata, "complete", complete)
        self._callback(self, self.metadata)

    def save_timestamp(self, metadata, name, timestamp):
        # This method works around the fact that we can't store
        # datetime objects in metadata.
        f, s = math.modf(timestamp.timestamp())
        metadata[f"{name}_s"] = int(s)
        metadata[f"{name}_ns"] = int(f * NS_PER_SEC)

    def get_timestamp(self, metadata, name):
        s = metadata[f"{name}_s"]
        f = metadata[f"{name}_ns"]
        f *= SEC_PER_NS
        return s + f

    def get_times(self, metadata, rename_metadata=lambda name: name):
        #
        frame_number = metadata.get(rename_metadata("frame_number"), 0)

        # frame_start_s is the time that the first data arrived at the FPGA;
        # the network receiver calls this "timestamp".
        frame_start_s = self.get_timestamp(metadata, rename_metadata("timestamp"))

        # After the FPGA sends the last sensor data packet for a frame, it follows
        # that with a 128-byte metadata packet.  This timestamp (which the network
        # receiver calls "metadata") is the time at which the FPGA sends that
        # packet; so it's the time immediately after the the last byte of sensor
        # data in this window.  The difference between frame_start_s and frame_end_s
        # is how long it took for the sensor to produce enough data for a complete
        # frame.
        frame_end_s = self.get_timestamp(metadata, rename_metadata("metadata"))

        # complete_s is the time when our compute ran, after visualization finished.
        complete_s = self.get_timestamp(metadata, "complete")

        return {
            rename_metadata("frame_number"): frame_number,
            rename_metadata("frame_start"): frame_start_s,
            rename_metadata("frame_end"): frame_end_s,
            "complete": complete_s,
        }
