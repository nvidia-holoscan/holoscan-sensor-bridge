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

import dataclasses
import datetime
import logging
import math

import cupy as cp
import holoscan
import utils

import hololink as hololink_module

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
        # Compute the Y of YCrCb
        r = cp_frame[:, :, 0]
        g = cp_frame[:, :, 1]
        b = cp_frame[:, :, 2]
        y = r * 0.299 + g * 0.587 + b * 0.114
        #
        buckets, _ = cp.histogram(y, bins=16, range=(0, 65536))
        self._callback(buckets)


def color_profiler_initial_timeout(frame_limit):
    # Produce a list appropriate for use with Watchdog's initial_timeout
    # parameter when using ColorProfiler.
    start_frames = COLOR_PROFILER_START_FRAME + 2
    working_frames = frame_limit - start_frames - 10
    assert working_frames > 0
    long_timeout = 30  # seconds
    short_timeout = 0.5  # seconds
    timeout_profile = [
        (long_timeout, start_frames),
        (short_timeout, working_frames),
        (long_timeout, 1),
    ]
    return utils.timeout_sequence(timeout_profile)


MS_PER_SEC = 1000.0
US_PER_SEC = 1000.0 * MS_PER_SEC
NS_PER_SEC = 1000.0 * US_PER_SEC
SEC_PER_NS = 1.0 / NS_PER_SEC


class TimeProfiler(holoscan.core.Operator):
    def __init__(
        self,
        *args,
        callback=None,
        rename_metadata=lambda name: name,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._count = 0
        self._callback = callback
        self._packets_dropped = None
        #
        self._frame_number_metadata = rename_metadata("frame_number")
        self._packets_dropped_metadata = rename_metadata("packets_dropped")
        self._timestamp_ns_metadata = rename_metadata("timestamp_ns")
        self._timestamp_s_metadata = rename_metadata("timestamp_s")
        self._received_s_metadata = rename_metadata("received_s")
        self._received_ns_metadata = rename_metadata("received_ns")
        self._metadata_s_metadata = rename_metadata("metadata_s")
        self._metadata_ns_metadata = rename_metadata("metadata_ns")

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
        frame_number = metadata.get(self._frame_number_metadata, 0)
        packets_dropped = metadata.get(self._packets_dropped_metadata, 0)
        if packets_dropped != self._packets_dropped:
            logging.info(f"{packets_dropped=} ({packets_dropped:#X}) {frame_number=}")
            self._packets_dropped = packets_dropped
        image_timestamp_ns = metadata.get(self._timestamp_ns_metadata, 0)
        image_timestamp_s = metadata.get(self._timestamp_s_metadata, 0)
        image_timestamp_s += image_timestamp_ns * SEC_PER_NS
        received_timestamp_s = metadata.get(self._received_s_metadata, 0)
        received_timestamp_ns = metadata.get(self._received_ns_metadata, 0)
        received_timestamp_s = received_timestamp_s + received_timestamp_ns * SEC_PER_NS
        metadata_timestamp_s = metadata.get(self._metadata_s_metadata, 0)
        metadata_timestamp_ns = metadata.get(self._metadata_ns_metadata, 0)
        metadata_timestamp_s = metadata_timestamp_s + metadata_timestamp_ns * SEC_PER_NS
        pipeline_timestamp_s = datetime.datetime.now(datetime.timezone.utc).timestamp()
        self._callback(
            image_timestamp_s,
            metadata_timestamp_s,
            received_timestamp_s,
            pipeline_timestamp_s,
            frame_number,
        )


class WatchdogOp(holoscan.core.Operator):
    def __init__(
        self,
        *args,
        watchdog=None,
        frame_limit=None,
        stop_conditions=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._watchdog = watchdog
        # The frame budget lives at this sink, not on the receivers: a
        # frame-dropping stage (FrameAlignerOp) sits between the receivers
        # and here, so counting produced frames would starve this sink by
        # the drop count. Once frame_limit aligned frames have arrived we
        # disable the source receivers' tick conditions so the application
        # drains and exits cleanly.
        self._frame_limit = frame_limit
        self._stop_conditions = stop_conditions or []
        self._taps = 0

    def setup(self, spec):
        spec.input("input")

    def compute(self, op_input, op_output, context):
        _ = op_input.receive("input")
        self._watchdog.tap()
        self._taps += 1
        if self._frame_limit is not None and self._taps >= self._frame_limit:
            for condition in self._stop_conditions:
                condition.disable_tick()


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
        complete = datetime.datetime.now(datetime.timezone.utc)

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
        frame_number_metadata = rename_metadata("frame_number")
        timestamp_metadata = rename_metadata("timestamp")
        metadata_metadata = rename_metadata("metadata")
        frame_start_metadata = rename_metadata("frame_start")
        frame_end_metadata = rename_metadata("frame_end")

        #
        frame_number = metadata.get(frame_number_metadata, 0)

        # frame_start_s is the time that the first data arrived at the FPGA;
        # the network receiver calls this "timestamp".
        frame_start_s = self.get_timestamp(metadata, timestamp_metadata)

        # After the FPGA sends the last sensor data packet for a frame, it follows
        # that with a 128-byte metadata packet.  This timestamp (which the network
        # receiver calls "metadata") is the time at which the FPGA sends that
        # packet; so it's the time immediately after the the last byte of sensor
        # data in this window.  The difference between frame_start_s and frame_end_s
        # is how long it took for the sensor to produce enough data for a complete
        # frame.
        frame_end_s = self.get_timestamp(metadata, metadata_metadata)

        # complete_s is the time when our compute ran, after visualization finished.
        complete_s = self.get_timestamp(metadata, "complete")

        return {
            frame_number_metadata: frame_number,
            frame_start_metadata: frame_start_s,
            frame_end_metadata: frame_end_s,
            "complete": complete_s,
        }


class PassThroughOperator(holoscan.core.Operator):
    def __init__(self, *args, in_tensor_name="", out_tensor_name="", **kwargs):
        super().__init__(*args, **kwargs)
        self._in_tensor_name = in_tensor_name
        self._out_tensor_name = out_tensor_name

    def setup(self, spec):
        logging.info("setup")
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("input")
        tensor = in_message.get(self._in_tensor_name)
        op_output.emit({self._out_tensor_name: tensor}, "output")


class RecordMetadataOp(PassThroughOperator):
    def __init__(self, *args, metadata_class=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._metadata_class = metadata_class
        self._field_names = [f.name for f in dataclasses.fields(self._metadata_class)]
        self._record = []

    def compute(self, op_input, op_output, context):
        super().compute(op_input, op_output, context)
        metadata = self.metadata
        try:
            values = []
            for name in self._field_names:
                value = metadata[name]
                hololink_module.NvtxTrace.event_u64(
                    f"{self.name} {name=} {value=:#x}", value
                )
                values.append(value)
            self._record.append(self._metadata_class(*values))
        except KeyError as e:
            logging.error(f"Caught {e} ({type(e)=})")
            logging.error("Metadata contents:")
            for name, value in metadata.items():
                logging.error(f"  {name}={value}")
            raise

    def get_record(self):
        return self._record


class FrameAlignerOp(holoscan.core.Operator):
    def __init__(
        self,
        *args,
        name=None,
        allowable_dt=None,
        input_tensors=None,  # this is a list e.g. ["left", "right"]
        rename_metadata=None,  # this is a list: [lambda name: f"left_{name}", ...]
        outputs=None,  # this is a list, e.g. ["left", "right"]
        **kwargs,
    ):
        self._outputs = outputs
        super().__init__(*args, name=name, **kwargs)
        self._name = name
        self._allowable_dt = allowable_dt
        self._input_tensors = input_tensors
        self._count = len(input_tensors)
        if len(rename_metadata) != self._count:
            raise ValueError(
                f"input_tensors has {self._count} items; rename_metadata ({len(rename_metadata)}) must be the same length."
            )
        if len(outputs) != self._count:
            raise ValueError(
                f"input_tensors has {self._count} items; outputs ({len(outputs)}) must be the same length."
            )
        self._timestamp_s_metadata = [
            rename("timestamp_s") for rename in rename_metadata
        ]
        self._timestamp_ns_metadata = [
            rename("timestamp_ns") for rename in rename_metadata
        ]
        self._dropped = 0
        # Diagnostics: per-leg arrival counts, emitted groups, and dt drops.
        # These pinpoint whether a stall is starvation (one leg stops
        # arriving), an alignment drop (dt > allowable_dt), or input-queue
        # POP loss (arrivals lag the receivers' frame numbers).
        self._arrivals = [0] * self._count
        self._emitted = 0
        self._dt_dropped = 0
        self._data_cache = [None] * self._count
        self._metadata_cache = [None] * self._count
        for i in range(self._count):
            metadata_cache = holoscan.core.MetadataDictionary()
            metadata_cache.policy = holoscan.core.MetadataPolicy.UPDATE
            self._metadata_cache[i] = metadata_cache
        self._timestamp = [0] * self._count
        self._cached = 0

    def setup(self, spec):
        logging.info("setup")
        spec.input("input", policy=holoscan.core.IOSpec.QueuePolicy.POP)
        for output in self._outputs:
            spec.output(output)

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("input")
        metadata = self.metadata
        # in_message.get(name) is the only way to find
        # if 'name' is a supplied tensor.
        for index, name in enumerate(self._input_tensors):
            m = in_message.get(name)
            if m is not None:
                self.cache_message(index, name, m, metadata)

        # Do we have all the channels?
        if self._cached < self._count:
            return

        # Is the oldest one outside our allowable dt?
        dt = max(self._timestamp) - min(self._timestamp)
        if self._allowable_dt is not None:
            if dt > self._allowable_dt:
                # Wait for the next frame; eventually
                # we should get sync'd up
                self._dt_dropped += 1
                logging.debug(
                    f"{self._name}: dropping group dt={dt:.6f} > "
                    f"allowable_dt={self._allowable_dt:.6f} "
                    f"arrivals={self._arrivals} emitted={self._emitted} "
                    f"dt_dropped={self._dt_dropped}"
                )
                return

        # So this group of frames should be forwarded.
        for index, output_name in enumerate(self._outputs):
            metadata.swap(self._metadata_cache[index])
            op_output.emit({"": self._data_cache[index]}, output_name)
            # reset
            self._data_cache[index] = None
            self._metadata_cache[index].clear()
            self._timestamp[index] = 0  # causes a huge dt
        self._cached = 0
        self._emitted += 1
        if self._emitted % 30 == 0:
            logging.debug(
                f"{self._name}: emitted={self._emitted} dt={dt:.6f} "
                f"arrivals={self._arrivals} dt_dropped={self._dt_dropped}"
            )

    def cache_message(self, index, name, tensor, metadata):
        # Default a missing timestamp to 0 so a frame without one (e.g. a
        # controller's fallback image, served while disconnected) doesn't crash
        # the aligner; live FPGA timestamps are never 0, so real streams are
        # unaffected.
        timestamp_s = metadata.get(self._timestamp_s_metadata[index], 0)
        timestamp_ns = metadata.get(self._timestamp_ns_metadata[index], 0)
        timestamp_ns *= SEC_PER_NS
        timestamp_s += timestamp_ns
        logging.trace(f"got {index=} {name=} {timestamp_s=}")
        self._arrivals[index] += 1
        # Track occupancy by the data slot (reset to None on emit), not the
        # timestamp: a fallback frame's timestamp is 0, which is also the
        # post-emit reset value, so counting by timestamp would recount the
        # same leg and over-report _cached — emitting before every leg arrives.
        if self._data_cache[index] is None:
            self._cached += 1
        #
        metadata_cache = self._metadata_cache[index]
        metadata_cache.update(metadata)
        self._data_cache[index] = tensor
        self._timestamp[index] = timestamp_s

    def enable(self, allowable_dt):
        self._allowable_dt = allowable_dt


class OnFrameNOperator(holoscan.core.Operator):
    def __init__(
        self,
        *args,
        trigger_frame=50,
        callback=None,
        in_tensor_name="",
        out_tensor_name=None,
        **kwargs,
    ):
        self._in_tensor_name = in_tensor_name
        self._out_tensor_name = out_tensor_name
        super().__init__(*args, **kwargs)
        self._trigger_frame = trigger_frame
        self._callback = callback
        self._count = 0

    def setup(self, spec):
        logging.info("setup")
        spec.input("input")
        if self._out_tensor_name is not None:
            spec.output("output")

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("input")
        if self._out_tensor_name is not None:
            tensor = in_message.get(self._in_tensor_name)
            op_output.emit({self._out_tensor_name: tensor}, "output")
        self.check()

    def check(self):
        self._count += 1
        if self._count != self._trigger_frame:
            return
        self._callback()
