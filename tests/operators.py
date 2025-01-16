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
        in_message = op_input.receive("input")
        in_message.get("")
        self._watchdog.tap()
