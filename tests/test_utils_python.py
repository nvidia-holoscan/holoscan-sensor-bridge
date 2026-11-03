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

# Checks the HSDK5 test helpers in utils.py that need no board.

import logging
import os
import signal
import threading
import time

import applications
import capture_metadata
import holoscan
import pytest
import utils


def test_watchdog_fires_after_timeout():
    with pytest.raises(AssertionError, match="timeout"):
        with utils.Watchdog("fires", timeout=0.1):
            time.sleep(1)


def test_watchdog_tap_keeps_alive():
    with utils.Watchdog("tapped", timeout=0.2) as watchdog:
        for _ in range(10):
            time.sleep(0.05)
            watchdog.tap()


def test_watchdog_fails_without_taps():
    # Whatever taps the watchdog never ran, so the check it makes never happened.
    with pytest.raises(AssertionError, match="tapped 0 times"):
        with utils.Watchdog("untapped", timeout=5):
            pass


def test_watchdog_min_taps():
    with pytest.raises(AssertionError, match="tapped 2 times, not at least 3"):
        with utils.Watchdog("short", timeout=5, min_taps=3) as watchdog:
            watchdog.tap()
            watchdog.tap()
    with utils.Watchdog("enough", timeout=5, min_taps=3) as watchdog:
        for _ in range(3):
            watchdog.tap()
    assert watchdog.count == 3


def test_watchdog_finish_allows_teardown():
    # After finish(), the block may take longer than the tap timeout to end.
    with utils.Watchdog("finished", timeout=0.2) as watchdog:
        watchdog.tap()
        watchdog.finish(timeout=5)
        time.sleep(0.5)


def test_leg_watchdogs_are_independent():
    # One leg's taps don't keep the other leg's watchdog alive.
    with pytest.raises(AssertionError, match="right watchdog"):
        with utils.leg_watchdogs(("left", "right"), timeout=0.3) as watchdogs:
            for _ in range(20):
                time.sleep(0.05)
                watchdogs["left"].tap()


def test_frame_floor():
    assert utils.frame_floor(300) == 60
    assert utils.frame_floor(100) == 20
    assert utils.frame_floor(30) == 10
    # At most half of a short run's frames.
    assert utils.frame_floor(10) == 5
    assert utils.frame_floor(5) == 3


def test_finish_watchdogs():
    # WatchdogOp finishes every running watchdog when it stops: the block may then
    # take longer than the tap timeouts to end.
    with utils.leg_watchdogs(("left", "right"), timeout=0.2) as watchdogs:
        for watchdog in watchdogs.values():
            watchdog.tap()
        utils.finish_watchdogs(timeout=5)
        time.sleep(0.5)


def test_watchdog_initial_timeout_sequence():
    # Two long intervals, then the short one.
    with pytest.raises(AssertionError, match="timeout"):
        with utils.Watchdog(
            "sequence", initial_timeout=utils.timeout_sequence([(0.5, 2), (0.1, 1)])
        ) as watchdog:
            time.sleep(0.3)
            watchdog.tap()  # second interval: 0.5 s
            time.sleep(0.3)
            watchdog.tap()  # third interval: 0.1 s
            time.sleep(1)


class _GatedLock:
    """Stands in for a Watchdog's lock and holds its timer thread back until released."""

    def __init__(self, lock):
        self._lock = lock
        self.waiting = threading.Event()
        self.release = threading.Event()

    def __enter__(self):
        if threading.current_thread() is not threading.main_thread():
            self.waiting.set()
            self.release.wait()
        return self._lock.__enter__()

    def __exit__(self, *args):
        return self._lock.__exit__(*args)


def test_watchdog_stopped_as_it_fires():
    # The timer fires, then stop() runs before the timer commits to the timeout: it
    # must not commit then. The outer watchdog keeps SIGUSR1 handled, so a late one
    # fails this test instead of killing pytest.
    with utils.Watchdog("outer", timeout=30) as outer:
        outer.tap()
        watchdog = utils.Watchdog("stopped", timeout=0.1)
        lock = _GatedLock(watchdog._lock)
        watchdog._lock = lock
        watchdog.start()
        assert lock.waiting.wait(5)
        watchdog.stop()
        lock.release.set()
        time.sleep(0.5)
        assert not watchdog._timed_out


class _HoldTimerAtItsMessage(logging.Handler):
    """Holds a Watchdog's timer at its timeout message, which _expired() logs under the
    watchdog's lock, until released."""

    def __init__(self):
        super().__init__()
        self.held = threading.Event()
        self.resume = threading.Event()  # not "release": Handler has a release()

    def emit(self, record):
        if threading.current_thread() is threading.main_thread():
            return
        if "timed out" in record.getMessage():
            self.held.set()
            self.resume.wait(5)


def test_watchdog_stopped_as_it_times_out():
    # The other order: the timer commits to the timeout while stop() waits for the
    # lock, so its SIGUSR1 is handled in stop(). The watchdog must still be
    # deregistered: else watchdog_failed stays set, so later timeouts fail no test,
    # and their timers abort pytest.
    hold = _HoldTimerAtItsMessage()
    logging.getLogger().addHandler(hold)
    watchdog = utils.Watchdog("committed", timeout=0.1)
    try:
        watchdog.start()
        assert hold.held.wait(5)
        # The timer commits once stop() waits for the lock the timer holds.
        threading.Timer(0.3, hold.resume.set).start()
        with pytest.raises(AssertionError, match="timeout"):
            watchdog.stop()
        assert watchdog not in utils.watchdogs
        assert not utils.watchdog_failed
    finally:
        logging.getLogger().removeHandler(hold)
        hold.resume.set()
        if watchdog in utils.watchdogs:
            watchdog.stop()  # finish the stop() that the assert cut short


def test_watchdog_timeout_in_stop_cleanup(monkeypatch):
    # A SIGUSR1 handled in stop()'s cleanup, after the watchdog left and before
    # SIGUSR2 restores the handlers. Its assert must wait for the end of stop():
    # raised there, it left watchdog_failed set and the handlers installed.
    watchdog = utils.Watchdog("late", timeout=30)
    kill = os.kill

    def kill_after_a_late_timeout(pid, sig):
        if sig == signal.SIGUSR2:  # stop()'s last step
            watchdog._timed_out = True  # as _expired() commits
            kill(pid, signal.SIGUSR1)
        kill(pid, sig)

    monkeypatch.setattr(utils.os, "kill", kill_after_a_late_timeout)
    # Restored after the test, should it fail.
    monkeypatch.setattr(utils, "watchdog_failed", False)
    watchdog.start()
    try:
        with pytest.raises(AssertionError, match="timeout"):
            watchdog.stop()
        assert watchdog not in utils.watchdogs
        assert not utils.watchdog_failed
        assert signal.getsignal(signal.SIGUSR1) == signal.SIG_DFL
    finally:
        watchdog._sigusr2(signal.SIGUSR2, None)  # restores the handlers if it failed


def test_watchdog_times_out_after_another_one_did(monkeypatch):
    # Of two watchdogs (one per leg, say), the first times out in its stop(), and the
    # caller catches its failure and goes on. A timeout of the second must still fail
    # the test: ignored, it would leave its timer to abort pytest.
    first = utils.Watchdog("first", timeout=30)
    second = utils.Watchdog("second", timeout=30)
    kill = os.kill
    injected = []

    def kill_after_a_late_timeout(pid, sig):
        if sig == signal.SIGUSR2 and not injected:  # the first stop()'s last step
            injected.append(sig)
            first._timed_out = True  # as _expired() commits
            kill(pid, signal.SIGUSR1)
        kill(pid, sig)

    monkeypatch.setattr(utils.os, "kill", kill_after_a_late_timeout)
    # Restored after the test, should it fail.
    monkeypatch.setattr(utils, "watchdog_failed", False)
    first.start()
    second.start()
    try:
        with pytest.raises(AssertionError, match="first"):
            first.stop()
        second.tap(0.1)
        with pytest.raises(AssertionError, match="second"):
            time.sleep(1)  # the second's timer aborts 2 s after its timeout
    finally:
        second.stop()


def _monitor(frames, terminations, frame_limits=None):
    monitor = utils.RunMonitor()
    monitor.frames = frames
    monitor.frame_limits = frame_limits or {}
    monitor.terminations = terminations
    return monitor


def test_run_monitor_passes_with_frames():
    _monitor({"rx": 99}, [holoscan.RunTermination.REQUESTED_COMPLETED]).assert_ok()


def test_run_monitor_fails_without_frames():
    monitor = _monitor({"rx": 0}, [holoscan.RunTermination.REQUESTED_COMPLETED])
    with pytest.raises(AssertionError, match="rx"):
        monitor.assert_ok()


def test_run_monitor_floor_follows_frame_limit():
    # frame_limit counts dropped frames too; a fifth of them must arrive.
    done = [holoscan.RunTermination.REQUESTED_COMPLETED]
    with pytest.raises(AssertionError, match="59 frames, need 60"):
        _monitor({"rx": 59}, done, {"rx": 300}).assert_ok()
    _monitor({"rx": 60}, done, {"rx": 300}).assert_ok()
    with pytest.raises(AssertionError, match="9 frames, need 10"):
        _monitor({"rx": 9}, done).assert_ok()


def test_run_monitor_fails_on_failed_run():
    monitor = _monitor({"rx": 50}, [holoscan.RunTermination.FAILED])
    with pytest.raises(AssertionError, match="FAILED"):
        monitor.assert_ok()


def test_run_monitor_fails_without_run():
    with pytest.raises(AssertionError, match="No HSDK5 run"):
        _monitor({}, []).assert_ok()


def test_run_monitor_fails_on_silent_receiver():
    # A receiver that never emits a frame has no FrameCounterOp samples; it must
    # still count, as 0.
    with utils.RunMonitor() as monitor:
        monitor._receivers.update(left=0, right=0)
        utils.FrameCounterOp.counts["left"] = list(range(50))
    assert monitor.frames == {"left": 50, "right": 0}
    monitor.terminations = [holoscan.RunTermination.REQUESTED_COMPLETED]
    with pytest.raises(AssertionError, match="right"):
        monitor.assert_ok()


def test_run_monitor_stopped_needs_allow_stopped():
    monitor = _monitor({"rx": 50}, [holoscan.RunTermination.STOPPED])
    with pytest.raises(AssertionError, match="STOPPED"):
        monitor.assert_ok()
    monitor.assert_ok(allow_stopped=True)


class MetadataSourceOp(holoscan.Operator):
    """Emits SAMPLES CaptureMetadata samples, with frame numbers 0, 1, ..."""

    SAMPLES = 5

    metadata = holoscan.Output("metadata", capture_metadata.CaptureMetadata)

    def setup(self, spec):
        spec.output(self.metadata)

    def contract(self):
        return (
            holoscan.Contract()
            .trigger(holoscan.OnClock(period=holoscan.time.ms(5)))
            .complete_after(self.SAMPLES)
        )

    def start(self):
        self._frame_number = 0

    def compute(self, context):
        self.metadata.emit(
            capture_metadata.CaptureMetadata(frame_number=self._frame_number)
        )
        self._frame_number += 1


def test_frame_counter_counts_only_samples():
    utils.FrameCounterOp.counts.clear()
    test_graph = applications.TestGraph("frame_counter")
    source = test_graph.op(MetadataSourceOp, "source")
    counter = test_graph.op(utils.FrameCounterOp, "counter", key="rx")
    test_graph.add_flow(source.metadata, counter.metadata, queue_depth=16)
    termination = test_graph.run(timeout_s=60)
    assert termination in utils.GOOD_TERMINATIONS
    assert utils.FrameCounterOp.counts["rx"] == list(range(MetadataSourceOp.SAMPLES))


class LinuxReceiverOp(MetadataSourceOp):
    """Named like a receiver, so RunMonitor counts its frames."""

    frame_limit = holoscan.Param(0)
    frame_limit_ends_run = holoscan.Param(False)


@pytest.mark.parametrize("frame_limit, passes", [(5, True), (50, False)])
def test_run_monitor_counts_receiver_frames(frame_limit, passes):
    # RunMonitor adds a FrameCounterOp after each receiver of the graphs built in
    # the block, and takes the receiver's frame_limit; the source always emits
    # MetadataSourceOp.SAMPLES (5) frames.
    with utils.RunMonitor() as monitor:
        test_graph = applications.TestGraph("receiver")
        test_graph.op(LinuxReceiverOp, "rx", frame_limit=frame_limit)
        test_graph.run(timeout_s=60)
    assert monitor.frames == {"rx": MetadataSourceOp.SAMPLES}
    assert monitor.frame_limits == {"rx": frame_limit}
    if passes:
        monitor.assert_ok()
    else:
        with pytest.raises(AssertionError, match="5 frames, need 10"):
            monitor.assert_ok()


def test_run_monitor_keys_receivers_of_two_graphs():
    # A receiver of another graph with the same name gets its own key, so a silent
    # one can't count the first one's frames.
    with utils.RunMonitor() as monitor:
        first = applications.TestGraph("first")
        first.op(LinuxReceiverOp, "rx", frame_limit=5)
        first.run(timeout_s=60)
        second = applications.TestGraph("second")
        second.op(LinuxReceiverOp, "rx", frame_limit=5)
    assert monitor.frames == {"rx": MetadataSourceOp.SAMPLES, "rx#2": 0}
    with pytest.raises(AssertionError, match="rx#2"):
        monitor.assert_ok()


class FailingSourceOp(MetadataSourceOp):
    """Fails its first activation."""

    def compute(self, context):
        raise RuntimeError("source failed")


def test_run_monitor_fails_a_bad_run_on_exit():
    # Ahead of a watchdog around it, which would report only too few taps.
    with pytest.raises(AssertionError, match="Run ended with"):
        with utils.RunMonitor():
            test_graph = applications.TestGraph("failing")
            test_graph.op(FailingSourceOp, "source")
            test_graph.run(timeout_s=60)


def test_run_monitor_restores_holoscan():
    graph, run_async = holoscan.Graph, holoscan.run_async
    with utils.RunMonitor():
        assert holoscan.Graph is not graph
        assert holoscan.run_async is not run_async
    assert holoscan.Graph is graph
    assert holoscan.run_async is run_async
