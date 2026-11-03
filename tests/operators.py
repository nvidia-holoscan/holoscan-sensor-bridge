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

# HSDK5 test operators. They differ from the HSDK4 ones in three ways:
# - Each one is a sink on a side branch of the pipeline, not a pass-through, so it
#   never holds up or changes the frames that the pipeline processes (connect the
#   ones that take only metadata with metadata_flow()). v2.7's FrameAlignerOp is
#   pair_by_timestamp(), which pairs the recorded frames after the run.
# - hs.Param values are deep-copied and type-checked, so a callback or a Watchdog
#   can't be a parameter. register() keeps the object and returns an int token,
#   which the operator takes as its parameter.
# - HSDK5 creates a new instance of each Python operator for each run, so nothing
#   that the test reads after the run may be kept on the instance.

import dataclasses
import datetime
import itertools
import logging
import threading

import capture_metadata
import cupy as cp
import holoscan as hs
import utils

import hololink as hololink_module

COLOR_PROFILER_START_FRAME = 5

# The share of a run's frames whose CRCs a test must check: v2.7 checked each
# frame's in the pipeline, and the HSDK5 checks miss only the frames that they
# can't keep up with.
CRC_COVERAGE = 0.9

MS_PER_SEC = 1000.0
US_PER_SEC = 1000.0 * MS_PER_SEC
NS_PER_SEC = 1000.0 * US_PER_SEC
SEC_PER_NS = 1.0 / NS_PER_SEC

_registry = {}
_registry_tokens = itertools.count(1)
_registry_lock = threading.Lock()


def register(obj):
    """Keep obj for an operator, which takes the returned token as a parameter."""
    with _registry_lock:
        token = next(_registry_tokens)
        _registry[token] = obj
    return token


def lookup(token):
    with _registry_lock:
        return _registry[token]


def luma_histogram(cp_frame):
    """The 16-bucket histogram of the Y of YCrCb that v2.7's ColorProfiler made,
    from an RGB(A) image with shape (y, x, 3 or 4). An 8-bit image (ArgusIspOp's
    RGB8) is scaled to the 16-bit range first."""
    if cp_frame.dtype == cp.uint8:
        cp_frame = cp_frame.astype(cp.uint16) * 257
    r = cp_frame[:, :, 0]
    g = cp_frame[:, :, 1]
    b = cp_frame[:, :, 2]
    y = r * 0.299 + g * 0.587 + b * 0.114
    buckets, _ = cp.histogram(y, bins=16, range=(0, 65536))
    return buckets


class ColorProfiler(hs.Operator):
    """Calls callback(buckets) with luma_histogram() of each image (for example
    BayerDemosaicOp's output)."""

    input = hs.Input("input", hs.schema.Image)

    callback_token = hs.Param(0)

    def setup(self, spec):
        spec.input(self.input, queue_depth=4)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, context):
        sample = self.input.receive()
        if sample.is_missing:
            return
        with cp.cuda.Stream(non_blocking=True) as stream:
            capsule = sample.data.data.__dlpack__(
                stream=stream.ptr, max_version=(1, 0), copy=False
            )
            buckets = luma_histogram(cp.from_dlpack(capsule)).get(stream=stream)
            stream.synchronize()
        lookup(self.callback_token)(buckets)


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


class TimeProfiler(hs.Operator):
    """Calls callback(image_timestamp_s, metadata_timestamp_s, received_timestamp_s,
    pipeline_timestamp_s, frame_number) for each frame's metadata, as v2.7 did.

    HSDK5 metadata is a fixed CaptureMetadata struct, so v2.7's rename_metadata has
    no HSDK5 equivalent."""

    metadata = hs.Input("metadata", capture_metadata.CaptureMetadata)

    callback_token = hs.Param(0)

    def setup(self, spec):
        spec.input(self.metadata, queue_depth=4)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.metadata))

    def compute(self, context):
        metadata = self.metadata.receive_data()
        pipeline_timestamp_s = datetime.datetime.now(datetime.timezone.utc).timestamp()
        image_timestamp_s = metadata.timestamp_s + metadata.timestamp_ns * SEC_PER_NS
        metadata_timestamp_s = metadata.metadata_s + metadata.metadata_ns * SEC_PER_NS
        received_timestamp_s = metadata.received_s + metadata.received_ns * SEC_PER_NS
        lookup(self.callback_token)(
            image_timestamp_s,
            metadata_timestamp_s,
            received_timestamp_s,
            pipeline_timestamp_s,
            metadata.frame_number,
        )


class WatchdogOp(hs.Operator):
    """Taps a utils.Watchdog for each frame's metadata. When it stops, the run is
    over, and it finishes every running watchdog (utils.finish_watchdogs()).

    The receivers end the run at their frame_limit, so v2.7's frame_limit and
    stop_conditions parameters have no HSDK5 equivalent."""

    metadata = hs.Input("metadata", capture_metadata.CaptureMetadata)

    watchdog_token = hs.Param(0)

    def setup(self, spec):
        spec.input(self.metadata, queue_depth=4)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.metadata))

    def compute(self, context):
        self.metadata.receive_data()
        lookup(self.watchdog_token).tap()

    def stop(self):
        utils.finish_watchdogs()


class RecordMetadataOp(hs.Operator):
    """Appends metadata_class(*fields) to record for each frame, where the fields are
    the CaptureMetadata fields that metadata_class names. Register (metadata_class,
    record) as the record_token."""

    metadata = hs.Input("metadata", capture_metadata.CaptureMetadata)

    record_token = hs.Param(0)

    def setup(self, spec):
        spec.input(self.metadata, queue_depth=4)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.metadata))

    def compute(self, context):
        metadata = self.metadata.receive_data()
        metadata_class, record = lookup(self.record_token)
        values = [
            getattr(metadata, field.name)
            for field in dataclasses.fields(metadata_class)
        ]
        record.append(metadata_class(*values))


class OnFrameNOperator(hs.Operator):
    """Calls callback() once, when the trigger_frame-th frame's metadata arrives."""

    metadata = hs.Input("metadata", capture_metadata.CaptureMetadata)

    trigger_frame = hs.Param(50)
    callback_token = hs.Param(0)

    def setup(self, spec):
        spec.input(self.metadata, queue_depth=4)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.metadata))

    def start(self):
        self._count = 0

    def compute(self, context):
        self.metadata.receive_data()
        self._count += 1
        if self._count == self.trigger_frame:
            lookup(self.callback_token)()


@dataclasses.dataclass
class CheckResult:
    checked: int
    bad: list  # (frame_number, reason)


class CrcChecker:
    """Records the GPU CRC check of each frame. add_crc_check() and
    add_image_crc_check() connect it."""

    def __init__(self):
        self._lock = threading.Lock()
        self._checks = []  # (frame_number, computed_crc, received_crc)

    def check_crc(self, computed_crc, received_crc, frame_number):
        # Called by CheckCrcOp through hl.CrcChannel.
        with self._lock:
            self._checks.append((frame_number, computed_crc, received_crc))

    def by_frame(self):
        """{frame_number: (computed_crc, received_crc)} for each checked frame."""
        with self._lock:
            return {
                frame_number: (computed, received)
                for frame_number, computed, received in self._checks
            }

    def result(self, skip_first=0, skip_last=0):
        """The check of the frames in order of frame number, without the first
        skip_first and the last skip_last frames."""
        with self._lock:
            checks = sorted(self._checks)
        checks = checks[skip_first : len(checks) - skip_last]
        bad = [
            (frame_number, f"crc {computed:#x} != {received:#x}")
            for frame_number, computed, received in checks
            if computed != received
        ]
        return CheckResult(checked=len(checks), bad=bad)


def add_crc_check(graph, name, receiver, frame_size, cuda_device_ordinal=0):
    """Check the FPGA CRC of each frame that receiver emits, on the GPU.

    ComputeCrcOp is a branch from the receiver, and CheckCrcOp takes ComputeCrcOp's
    metadata, as in examples/imx274_frame_validation.py. ComputeCrcOp's tensor
    output stays unconnected; don't bind a device for it."""
    checker = CrcChecker()
    crc_channel = hololink_module.CrcChannel(check_crc=checker.check_crc)
    compute = graph.op(
        hololink_module.operators.ComputeCrcOp,
        f"{name}_compute_crc",
        crc_channel=crc_channel,
        frame_size=frame_size,
        cuda_device_ordinal=cuda_device_ordinal,
    )
    check = graph.op(
        hololink_module.operators.CheckCrcOp,
        f"{name}_check_crc",
        crc_channel=crc_channel,
    )
    graph.add_flow(receiver.tensor, compute.in_tensor, queue_depth=4)
    graph.add_flow(receiver.metadata, compute.in_metadata, queue_depth=4)
    graph.add_flow(compute.metadata, check.in_metadata, queue_depth=4)
    return checker


def metadata_flow(graph, source, sink, queue_depth=16):
    """Connect a metadata output to an operator that takes only metadata, such as
    WatchdogOp, RecordMetadataOp or TimeProfiler, so that it never holds up the
    pipeline: a full queue drops its oldest sample. On a blocking edge, a sink that
    fell queue_depth frames behind would make the producer publish frames' tensors
    without their metadata, and an operator that pairs the two (CsiToBayerOp, say)
    would stop for good: it retires such a tensor only when newer metadata arrives,
    and the producer, its tensor queue full, sends no more."""
    graph.add_flow(
        source, sink, queue_policy=hs.QueuePolicy.DROP_OLDEST, queue_depth=queue_depth
    )


def add_image_crc_check(
    graph, name, image, metadata, frame_size, cuda_device_ordinal=0
):
    """Compute the CRC of each image, for example BayerDemosaicOp's, on the GPU, as
    v2.7 did with ComputeCrcOp; the returned CrcChecker's by_frame() gives
    {frame_number: (image CRC, the frame's metadata.crc)}. v2.7's expected image
    CRCs still apply: the HSDK5 demosaiced image is bit-exact with v2.7's.

    ComputeCrcImageOp pairs each image with the frame's metadata on their capture
    time, so take metadata from the operator that made the image's input (the
    operators pass the capture time on). frame_size is the image's size in bytes.
    Both are connected with DROP_OLDEST: the check never holds up the pipeline,
    and the frames it can't keep up with go unchecked. The metadata comes first,
    so the image needs a queue of one."""
    checker = CrcChecker()
    crc_channel = hololink_module.CrcChannel(check_crc=checker.check_crc)
    compute = graph.op(
        hololink_module.operators.ComputeCrcImageOp,
        f"{name}_compute_crc",
        crc_channel=crc_channel,
        frame_size=frame_size,
        cuda_device_ordinal=cuda_device_ordinal,
    )
    check = graph.op(
        hololink_module.operators.CheckCrcOp,
        f"{name}_check_crc",
        crc_channel=crc_channel,
    )
    drop_oldest = hs.QueuePolicy.DROP_OLDEST
    graph.add_flow(image, compute.in_tensor, queue_policy=drop_oldest, queue_depth=1)
    graph.add_flow(
        metadata, compute.in_metadata, queue_policy=drop_oldest, queue_depth=4
    )
    graph.add_flow(compute.metadata, check.in_metadata, queue_depth=4)
    return checker


def pair_by_timestamp(left, right, allowable_dt):
    """Pair two lists of (timestamp_s, item), each in time order, where the
    timestamps differ by at most allowable_dt. This replaces v2.7's FrameAlignerOp,
    which paired the legs in the graph."""
    pairs = []
    i = j = 0
    while i < len(left) and j < len(right):
        dt = left[i][0] - right[j][0]
        if abs(dt) <= allowable_dt:
            pairs.append((left[i], right[j]))
            i += 1
            j += 1
        elif dt < 0:
            i += 1
        else:
            j += 1
    logging.debug(f"pair_by_timestamp: {len(pairs)} pairs of {len(left)}, {len(right)}")
    return pairs


def assert_aligned(left, right, allowable_dt, min_pairs, skip_first=15, max_gap_s=0.5):
    """Pair the stereo legs left and right, lists of (timestamp_s, item), with
    pair_by_timestamp(), and fail unless they stayed aligned: at least min_pairs
    pairs, and after the first skip_first pairs, none more than max_gap_s after the
    one before, up to the end of the shorter leg. That is v2.7's check: its
    watchdog after FrameAlignerOp needed an aligned pair at least every 0.5 s once
    the pipeline was running. Returns the pairs."""
    left = sorted(left, key=lambda entry: entry[0])
    right = sorted(right, key=lambda entry: entry[0])
    pairs = pair_by_timestamp(left, right, allowable_dt)
    times = [left_entry[0] for left_entry, _ in pairs[skip_first:]]
    if times:
        # Legs that stop pairing for good leave no later pair to measure from.
        times.append(min(left[-1][0], right[-1][0]))
    gaps = [(later - earlier, earlier) for earlier, later in zip(times, times[1:])]
    max_gap, max_gap_at = max(gaps, default=(0.0, None))
    logging.info(
        f"{len(pairs)} stereo pairs within {allowable_dt} s, of {len(left)} left "
        f"and {len(right)} right frames; longest gap {max_gap:.3f} s"
    )
    assert len(pairs) >= min_pairs, (
        f"Only {len(pairs)} stereo pairs within {allowable_dt} s (need {min_pairs}), "
        f"of {len(left)} left and {len(right)} right frames"
    )
    assert (
        max_gap <= max_gap_s
    ), f"No stereo pair for {max_gap:.3f} s after the one at {max_gap_at:.6f} s"
    return pairs
