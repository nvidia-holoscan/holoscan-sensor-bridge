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

# See README.md for detailed information.

# Checks the HSDK5 test operator helpers in operators.py that need no board.

import dataclasses
import time

import applications
import cupy as cp
import numpy as np
import operators
import pytest
import test_utils_python
import utils


def test_register_lookup():
    first = object()
    second = object()
    first_token = operators.register(first)
    second_token = operators.register(second)
    assert first_token != second_token
    assert operators.lookup(first_token) is first
    assert operators.lookup(second_token) is second


def test_pair_by_timestamp():
    left = [(1.000, "l0"), (1.033, "l1"), (1.066, "l2")]
    right = [(1.0001, "r0"), (1.0335, "r1"), (1.2, "r2")]
    pairs = operators.pair_by_timestamp(left, right, 0.001)
    assert [(lt[1], rt[1]) for lt, rt in pairs] == [("l0", "r0"), ("l1", "r1")]


def test_pair_by_timestamp_skips_a_dropped_frame():
    left = [(1.000, "l0"), (1.033, "l1"), (1.066, "l2")]
    right = [(1.000, "r0"), (1.066, "r2")]
    pairs = operators.pair_by_timestamp(left, right, 0.001)
    assert [(lt[1], rt[1]) for lt, rt in pairs] == [("l0", "r0"), ("l2", "r2")]


def _leg(count, offset_s=0.0, period_s=1 / 60, missing=()):
    return [
        (offset_s + index * period_s, index)
        for index in range(count)
        if index not in missing
    ]


def test_assert_aligned():
    pairs = operators.assert_aligned(
        _leg(100), _leg(100, offset_s=0.002), 1 / 60, min_pairs=90
    )
    assert len(pairs) == 100


def test_assert_aligned_needs_min_pairs():
    with pytest.raises(AssertionError, match="Only 50 stereo pairs"):
        operators.assert_aligned(
            _leg(100), _leg(50, offset_s=0.002), 1 / 60, min_pairs=60
        )


def test_assert_aligned_fails_on_a_gap():
    # 40 frames without a right frame: 0.68 s without a pair.
    with pytest.raises(AssertionError, match="No stereo pair for 0.68"):
        operators.assert_aligned(
            _leg(100),
            _leg(100, missing=range(40, 80)),
            0.001,
            min_pairs=20,
        )


def test_assert_aligned_fails_on_a_lost_tail():
    # The legs pair for 40 frames, then the right one's frames come half a frame
    # late: no pair for the last second, though both legs go on.
    right = [(t + (0.008 if index >= 40 else 0.0), index) for t, index in _leg(100)]
    with pytest.raises(AssertionError, match="No stereo pair for 1.0"):
        operators.assert_aligned(_leg(100), right, 0.001, min_pairs=20)


def test_assert_aligned_skips_the_first_pairs():
    # The gap is between the first pairs, while the pipeline starts.
    operators.assert_aligned(
        _leg(100), _leg(100, missing=range(5, 45)), 0.001, min_pairs=20
    )


def test_watchdog_op_finishes_the_watchdog():
    # MetadataSourceOp's 5 frames each tap the watchdog. WatchdogOp finishes it
    # when the run stops, so the block can outlast the 0.3 s tap timeout.
    with utils.Watchdog("operator", timeout=0.3, min_taps=5) as watchdog:
        test_graph = applications.TestGraph("watchdog")
        source = test_graph.op(test_utils_python.MetadataSourceOp, "source")
        watchdog_operator = test_graph.op(
            operators.WatchdogOp,
            "watchdog",
            watchdog_token=operators.register(watchdog),
        )
        test_graph.add_flow(source.metadata, watchdog_operator.metadata, queue_depth=16)
        test_graph.run(timeout_s=60)
        time.sleep(0.6)
    assert watchdog.count == test_utils_python.MetadataSourceOp.SAMPLES


def test_luma_histogram():
    # Every pixel is white: all of them land in the top bucket.
    frame = cp.full((16, 16, 4), 65535, dtype=cp.uint16)
    buckets = operators.luma_histogram(frame).get()
    assert len(buckets) == 16
    assert buckets[15] == 16 * 16
    assert buckets.sum() == 16 * 16


def test_crc_checker_result_skips_frames():
    checker = operators.CrcChecker()
    # Out of order on purpose; frame 3 is bad, frame 9 (skipped) is bad.
    for frame_number in [2, 0, 1, 3, 4, 5, 6, 7, 8, 9]:
        bad = frame_number in (3, 9)
        checker.check_crc(0x1234, 0x9999 if bad else 0x1234, frame_number)
    result = checker.result(skip_first=1, skip_last=1)
    assert result.checked == 8
    assert [frame_number for frame_number, _ in result.bad] == [3]


def _v27_color_profile(frame):
    """v2.7's ColorProfiler.compute() histogram, in numpy."""
    r = frame[:, :, 0]
    g = frame[:, :, 1]
    b = frame[:, :, 2]
    y = r * 0.299 + g * 0.587 + b * 0.114
    buckets, _ = np.histogram(y, bins=16, range=(0, 65536))
    return buckets


def test_color_histogram_matches_v27():
    frame = np.random.default_rng(1).integers(0, 65536, (64, 48, 4), dtype=np.uint16)
    buckets = operators.luma_histogram(cp.asarray(frame)).get()
    assert list(buckets) == list(_v27_color_profile(frame))


def test_color_histogram_scales_8_bit_images():
    frame = np.random.default_rng(2).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    buckets = operators.luma_histogram(cp.asarray(frame)).get()
    expected = _v27_color_profile(frame.astype(np.uint16) * 257)
    assert list(buckets) == list(expected)


@dataclasses.dataclass
class _FrameNumberRecord:
    frame_number: int


def test_sink_results_survive_new_instance():
    # HSDK5 makes a new instance of each Python operator for a run, so a sink
    # must keep its results outside the instance; RecordMetadataOp uses register().
    record = []
    test_graph = applications.TestGraph("record_metadata")
    source = test_graph.op(test_utils_python.MetadataSourceOp, "source")
    recorder = test_graph.op(
        operators.RecordMetadataOp,
        "recorder",
        record_token=operators.register((_FrameNumberRecord, record)),
    )
    test_graph.add_flow(source.metadata, recorder.metadata, queue_depth=16)
    test_graph.run(timeout_s=60)
    assert [r.frame_number for r in record] == list(
        range(test_utils_python.MetadataSourceOp.SAMPLES)
    )
