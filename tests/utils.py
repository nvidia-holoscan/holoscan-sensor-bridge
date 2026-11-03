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

import contextlib
import logging
import os
import signal
import threading
import time
import traceback
from unittest import mock

import capture_metadata
import cuda.bindings.driver as cuda
import cupy as cp
import hololink.sensors.csi  # noqa: F401 -- hololink_module.sensors.csi below
import hololink_module.operators as hololink_module_operators
import holoscan as hs
import numpy as np
import PIL.Image
import pytest

import hololink as hololink_module


def encode_raw_8_bayer_image(bayer_image):
    return bayer_image.astype(np.uint8)


def encode_raw_12_bayer_image(bayer_image):
    """
    Given a uint16 bayer image, encode it
    to a stream of bytes where the format
    matches the RAW12 format published in IMX274 spec;
    given the first two 12-bit color values (c0 and c1),
    the output is
        c0 bits 4..11,
        c1 bits 4..11,
        c0 bits 0..3 | ((c1 bits 0..3) << 4)
    for each uint16 in bayer_image.
    """
    bayer_height, bayer_width = bayer_image.shape
    ri = bayer_image.ravel()
    upper_byte = (ri >> 4).astype(np.uint8)
    lower_byte = (ri & 0xF).astype(np.uint8)
    combined_lower_bytes = lower_byte[::2] | (lower_byte[1::2] << 4)
    raw_12 = np.stack(
        [
            upper_byte[::2],
            upper_byte[1::2],
            combined_lower_bytes,
        ],
        axis=1,
    )
    # we're now 3 bytes per pixel wide instead of 2 half-words
    raw_12 = raw_12.reshape(bayer_height, bayer_width * 3 // 2)
    return raw_12


def encode_raw_10_bayer_image(bayer_image):
    """
    Given a uint16 bayer image, encode it
    to a stream of bytes where the format
    matches the RAW10 format published in IMX274 spec;
    given the first four 10-bit color values (c0, c1, c2, and c3),
    the output is
        c0 bits 2..9,
        c1 bits 2..9,
        c2 bits 2..9,
        c3 bits 2..9,
        (c0 bits 0..1 | ((c1 bits 0..1) << 2)
            | (c2 bits 0..1) << 4 | ((c3 bits 0..1) << 6))
    for each uint16 in bayer_image.
    """
    bayer_height, bayer_width = bayer_image.shape
    ri = bayer_image.ravel()
    upper_byte = (ri >> 2).astype(np.uint8)
    lower_byte = (ri & 0x3).astype(np.uint8)
    combined_lower_bytes = (
        lower_byte[::4]
        | (lower_byte[1::4] << 2)
        | (lower_byte[2::4] << 4)
        | (lower_byte[3::4] << 6)
    )
    raw_10 = np.stack(
        [
            upper_byte[::4],
            upper_byte[1::4],
            upper_byte[2::4],
            upper_byte[3::4],
            combined_lower_bytes,
        ],
        axis=1,
    )
    # we're now 5 bytes per four pixels wide
    raw_10 = raw_10.reshape(bayer_height, bayer_width * 5 // 4)
    return raw_10


def encode_8_bit_image(image):
    return image


def encode_12_bit_image(image):
    return (image >> 4).astype(np.uint8)


def encode_10_bit_image(image):
    return (image >> 2).astype(np.uint8)


def make_image(
    bayer_height,
    bayer_width,
    bayer_format: hololink_module.sensors.csi.BayerFormat,
    pixel_format: hololink_module.sensors.csi.PixelFormat,
):
    """Make a demo video frame."""
    logging.info(
        "Generating a demo image bayer_height=%s bayer_width=%s bayer_format=%s pixel_format=%s"  # noqa: E501
        % (bayer_height, bayer_width, bayer_format, pixel_format)
    )
    if pixel_format == hololink_module.sensors.csi.PixelFormat.RAW_8:
        dtype = np.uint8
        limit = 255
        bayer_encoder = encode_raw_8_bayer_image
        image_encoder = encode_8_bit_image
    elif pixel_format == hololink_module.sensors.csi.PixelFormat.RAW_12:
        dtype = np.uint16
        limit = 4095
        bayer_encoder = encode_raw_12_bayer_image
        image_encoder = encode_12_bit_image
    elif pixel_format == hololink_module.sensors.csi.PixelFormat.RAW_10:
        dtype = np.uint16
        limit = 1023
        bayer_encoder = encode_raw_10_bayer_image
        image_encoder = encode_10_bit_image
    # the colors vary from min to max
    width = bayer_width // 2
    height = bayer_height // 2
    r = np.linspace(0, limit, num=width, dtype=dtype)
    g = np.linspace(0, limit, num=height, dtype=dtype)
    b = np.linspace(limit, 0, num=width, dtype=dtype)
    a = np.full((width,), limit, dtype=dtype)
    # make square arrays, these are one element
    # per pixel for the corresponding color.
    sr = np.tile(r, height).reshape(height, width)
    sg = np.tile(g, width).reshape(width, height).transpose()
    sb = np.tile(b, height).reshape(height, width)
    sa = np.tile(a, height).reshape(height, width)
    # merge into an RGBA frame.
    elements_per_pixel = 4
    image = np.stack([sr, sg, sb, sa], axis=2).reshape(
        height, width, elements_per_pixel
    )
    # Now make the bayer frame.
    if bayer_format == hololink_module.sensors.csi.BayerFormat.RGGB:
        # upper_line is red0, green0, red1, green1, ...
        a, b = sr.ravel(), sg.ravel()
        c = np.empty((a.size + b.size,), dtype=dtype)
        c[0::2] = a
        c[1::2] = b
        upper_line = c.reshape(height, bayer_width)
        # lower_line is green0, blue0, green1, blue1, ...
        a, b = sg.ravel(), sb.ravel()
        c = np.empty((a.size + b.size,), dtype=dtype)
        c[0::2] = a
        c[1::2] = b
        lower_line = c.reshape(height, bayer_width)
        bayer_image = np.stack([upper_line, lower_line], axis=1).reshape(
            bayer_height, bayer_width
        )
    elif bayer_format == hololink_module.sensors.csi.BayerFormat.GBRG:
        # upper_line is green0, blue0, green1, blue1, ...
        a, b = sg.ravel(), sb.ravel()
        c = np.empty((a.size + b.size,), dtype=dtype)
        c[0::2] = a
        c[1::2] = b
        upper_line = c.reshape(height, bayer_width)
        # lower_line is red0, green0, red1, green1, ...
        a, b = sr.ravel(), sg.ravel()
        c = np.empty((a.size + b.size,), dtype=dtype)
        c[0::2] = a
        c[1::2] = b
        lower_line = c.reshape(height, bayer_width)
        bayer_image = np.stack([upper_line, lower_line], axis=1).reshape(
            bayer_height, bayer_width
        )
    else:
        assert False and 'Unexpected image format "%s".' % (bayer_format,)
    #
    image = image_encoder(image)
    bayer_image = bayer_encoder(bayer_image)
    return image, bayer_image


def caller(n=0):
    stack = traceback.extract_stack(limit=3 + n)
    full_filename, line, method, statement = stack[-(3 + n)]
    filename = os.path.basename(full_filename)
    return "%s:%u" % (filename, line)


# The running watchdogs; SIGUSR1 and SIGUSR2 are handled while there are any.
watchdogs = set()
watchdogs_lock = threading.Lock()
# Set once SIGUSR1 has failed the test, so that another watchdog's timeout doesn't
# raise again while the test unwinds to its watchdogs; cleared as each one stops.
watchdog_failed = False
# While the main thread runs Watchdog.stop(), where _sigusr1 leaves a timeout's
# failure for stop() to raise once its cleanup is done: raised from the handler, it
# would cut the cleanup short. None otherwise.
deferred_failures = None

# A watchdog timeout can end the process with SIGABRT, bypassing finally blocks
# and atexit handlers.  Hardware cleanup that must happen before that abort can
# be registered here.  Hooks are best effort: the control plane may itself be
# unhealthy, so no hook may delay the abort indefinitely.
abort_hooks = []
abort_hooks_lock = threading.Lock()
ABORT_HOOK_TIMEOUT_S = 5.0


def register_abort_hook(hook):
    """Register hook() to run immediately before a watchdog SIGABRT."""
    with abort_hooks_lock:
        abort_hooks.append(hook)
    return hook


def unregister_abort_hook(hook):
    with abort_hooks_lock:
        if hook in abort_hooks:
            abort_hooks.remove(hook)


class AbortHook:
    """Register a watchdog abort hook for the duration of a context."""

    def __init__(self, hook):
        self._hook = hook

    def __enter__(self):
        register_abort_hook(self._hook)
        return self

    def __exit__(self, *args):
        unregister_abort_hook(self._hook)


def run_abort_hooks():
    with abort_hooks_lock:
        hooks = list(abort_hooks)
    for hook in hooks:

        def call(hook=hook):
            try:
                hook()
            except Exception as e:
                logging.error(f"Abort hook {hook} failed: {e}")

        thread = threading.Thread(target=call, daemon=True)
        thread.start()
        thread.join(ABORT_HOOK_TIMEOUT_S)
        if thread.is_alive():
            logging.error(
                f"Abort hook {hook} did not finish in {ABORT_HOOK_TIMEOUT_S}s; "
                "aborting anyway."
            )


class Watchdog:
    """When used this way:

        with Watchdog("watchdog-name", timeout=2) as watchdog:
            while True:
                watchdog.tap()
                do_something()

    If do_something takes longer than 2 seconds to execute, we'll
    assert fail with a watchdog timeout.

    Some variations:

    - Allow the first pass to take longer:

        with Watchdog("watchdog-name", initial_timeout=10, timeout=2) as watchdog:
            while True:
                watchdog.tap()
                do_something()

    - allow do_something() to take up to 30 seconds for the first 20 iterations,
      then only allow 2 seconds after that:

        with Watchdog("watchdog-name", initial_timeout=[30]*20, timeout=2) as watchdog:
            while True:
                watchdog.tap()
                do_something()

      This accommodates workflows where initialization may make the first n iterations
      take longer.

    - use a dynamic timeout by passing in a new limit each call to tap:

        with Watchdog("watchdog-name", timeout=10) as watchdog:
            while True:
                watchdog.tap(2)
                do_something()

      Watchdog always prefers to use the value passed to tap, and will fall
      back to the next initial values (if any remain) or finally the value
      passed as timeout to the constructor.  You can specify only the
      initial_timeout, in which case the last value from the initial_timeout
      will continue to be used as the tap value after the list is finished.

    The block fails when the watchdog was tapped fewer than min_taps times:
    whatever taps it stopped early, or never ran.
    """

    def __init__(
        self, name, timeout=None, initial_timeout=None, min_taps=1, location=None
    ):
        self._name = name
        # Where the watchdog was made, for the messages; by default the caller.
        self._caller = location or caller()
        self._str_id = f'watchdog@{id(self):#x} ({self._caller}) "{self._name}"'
        logging.debug(f"Creating {self._str_id}")
        if initial_timeout is None:
            # The dummy math here raises an exception if users didn't specify
            # a timeout parameter.
            initial_timeout = [timeout + 0]
        try:
            self._initial_timeout = iter(initial_timeout)
        except TypeError:
            self._initial_timeout = iter([initial_timeout])
        self._next_timeout = timeout
        self._min_taps = min_taps
        self._count = 0
        self._lock = threading.Lock()
        self._tap_time = None
        self._last_timeout = None
        # HSDK5 doesn't bind hololink's Reactor; a timer thread fires the alarm instead.
        self._alarm = None
        self._running = False
        # Set when _expired fires. The SIGUSR1 -> "assert False" path can be
        # swallowed (e.g. when the timeout lands inside a GXF operator's stop(),
        # GXF catches and logs the AssertionError), which would let a stalled
        # run be reported as PASSED. __exit__ re-checks this flag in the main
        # thread after run() returns so a timeout always fails the test.
        self._timed_out = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
        if args[0] is not None:
            return
        # A watchdog timeout is always a failure. The SIGUSR1 -> assert path can
        # be swallowed when the timeout lands inside a C++ operator's stop()
        # (GXF logs and discards the exception), which would let a stalled run be
        # reported as PASSED. Re-raise here, in the main thread after run() has
        # returned, unless the block is already unwinding another exception.
        if self._timed_out:
            raise AssertionError(f"{self._str_id} timed out (count={self._count})")
        if self._count < self._min_taps:
            raise AssertionError(
                f"{self._str_id} was tapped {self._count} times, "
                f"not at least {self._min_taps}"
            )

    @property
    def count(self):
        """The number of taps so far."""
        return self._count

    def start(self):
        # In order to get pytest to properly fail, we need to call assert False
        # in the main thread-- which must be the thread calling start().  (This
        # is checked by these calls to signal.signal(...) which fail in any
        # other thread except the main one.)  So the idea is, when a background
        # thread (the reactor) discovers a watchdog timeout, it sends SIGUSR1
        # to the foreground process, which will interrupt whatever it's waiting
        # on and trigger the test failure.  When we stop the watchdog, we need
        # to set the SIGUSR1 and SIGUSR2 signals back to SIG_DFL, so we use
        # SIGUSR2 to get the foreground thread to do that.
        with watchdogs_lock:
            watchdogs.add(self)
            if len(watchdogs) == 1:
                signal.signal(signal.SIGUSR1, self._sigusr1)
                signal.signal(signal.SIGUSR2, self._sigusr2)
        self._pid = os.getpid()
        timeout = self._get_next_timeout()
        self._running = True
        with self._lock:
            self._set_alarm(timeout)

    def _set_alarm(self, timeout):
        # Called with self._lock held.
        if self._alarm is not None:
            self._alarm.cancel()
        self._alarm = threading.Timer(timeout, self._expired)
        self._alarm.daemon = True
        self._alarm.start()

    def _sigusr1(self, signal_number, stack_frame):
        global watchdog_failed
        # This handler is the first running watchdog's; name the ones that timed
        # out. No lock: the main thread, which runs this, may hold watchdogs_lock.
        timed_out = [
            watchdog._str_id for watchdog in list(watchdogs) if watchdog._timed_out
        ]
        message = (
            f"Caught SIGUSR1: watchdog timeout: {', '.join(timed_out) or self._str_id}."
        )
        logging.error(message)
        if watchdog_failed:
            return
        watchdog_failed = True
        if deferred_failures is not None:
            deferred_failures.append(message)
            return
        assert False, message

    def _sigusr2(self, signal_number, stack_frame):
        global watchdog_failed
        # Cleared here, after the handler of any SIGUSR1 sent before stop()'s SIGUSR2,
        # which may have set it; also while other watchdogs run, as a caller that
        # caught the failure goes on, and their timeouts must fail it again.
        watchdog_failed = False
        with watchdogs_lock:
            if watchdogs:
                signal.signal(signal.SIGUSR1, self._sigusr1)
                signal.signal(signal.SIGUSR2, self._sigusr2)
            else:
                signal.signal(signal.SIGUSR1, signal.SIG_DFL)
                signal.signal(signal.SIGUSR2, signal.SIG_DFL)

    def stop(self):
        global deferred_failures
        # A timer that committed to its timeout first can have its SIGUSR1 handled in
        # here. _sigusr1 then leaves the failure for this to raise once the watchdog
        # is deregistered and the handlers restored, and clearing _running before
        # taking the lock keeps the timer from aborting.
        deferred_failures = []
        self._running = False
        try:
            with self._lock:
                if self._alarm is not None:
                    self._alarm.cancel()
            with watchdogs_lock:
                watchdogs.discard(self)
            # _sigusr2 restores the handlers. A SIGUSR1 still pending comes to this
            # thread with this signal, and its handler runs first, so it can't meet
            # SIG_DFL.
            os.kill(self._pid, signal.SIGUSR2)
        finally:
            failures, deferred_failures = deferred_failures, None
        if failures:
            raise AssertionError(failures[0])

    def tap(self, timeout=None):
        # It's permissible for users to call stop
        # and then call tap at a later time;
        # just ignore the tap call in that case.
        if not self._running:
            logging.debug(
                f"{self._str_id} Ignoring call to tap because we've been stopped."
            )
            return
        with self._lock:
            self._count += 1
            logging.debug(f"tapping {self._str_id} count={self._count} {timeout=}")
            self._tap_time = time.monotonic()
            self._set_alarm(self._get_next_timeout(timeout))

    def finish(self, timeout=30):
        """The stream that taps this watchdog ended, for example with the run:
        allow timeout seconds until the block ends, rather than the timeout of the
        next tap. This isn't a tap; see finish_watchdogs()."""
        if not self._running:
            return
        with self._lock:
            self._set_alarm(timeout)

    def _get_next_timeout(self, user_value=None):
        if user_value is not None:
            return user_value
        try:
            timeout = next(self._initial_timeout)
            self._last_timeout = timeout
            return timeout
        except StopIteration:
            pass
        if self._next_timeout is None:
            return self._last_timeout
        return self._next_timeout

    def _expired(self):
        # Check and commit under the lock that stop() and tap() take, so that once
        # they return this timer can't fire: a late SIGUSR1 would fail whatever test
        # runs then, or kill pytest after the last watchdog restored SIG_DFL.
        with self._lock:
            # It's possible for users to call stop
            # when we have this call queued up; so
            # don't trigger an error in that case.
            if not self._running:
                logging.debug(
                    f"{self._str_id} Ignoring call to _expired because we've been stopped."
                )
                return
            # A tap can replace the timer after this one started to run.
            if threading.current_thread() is not self._alarm:
                return
            now = time.monotonic()
            dt = "N/A"
            if self._tap_time is not None:
                now = time.monotonic()
                dt = now - self._tap_time
            message = (
                f"{self._str_id} timed out, count={self._count}, "
                f"time since last tap={dt}."
            )
            logging.error(message)
            # Record the timeout so __exit__ fails the test even if the SIGUSR1
            # assert below is swallowed (e.g. inside a GXF operator stop()).
            self._timed_out = True
            os.kill(self._pid, signal.SIGUSR1)
        time.sleep(2)
        if not self._running:
            # _sigusr1's assert unwound the test, and __exit__ stopped us.
            return
        run_abort_hooks()
        logging.error("Sending SIGABRT to the local process group.")
        local_process_group = 0
        os.kill(local_process_group, signal.SIGABRT)

    def update(self, timeout=None, initial_timeout=None, tap=True):
        """Reconfigure how the next tap() works."""
        if initial_timeout is None:
            # The dummy math here raises an exception if users didn't specify
            # a timeout parameter.
            initial_timeout = [timeout + 0]
        with self._lock:
            try:
                self._initial_timeout = iter(initial_timeout)
            except TypeError:
                self._initial_timeout = iter([initial_timeout])
            self._next_timeout = timeout
        if tap:
            self.tap()


def timeout_sequence(profile):
    # Given an input list of [(value,count),...], produce a list approproriate
    # for use with Timeout's initial_timeout parameter.  For example, one test
    # wants to have long timeouts for the first 10 loops (to allow for GPU startup
    # times), short timeouts for the next 200 loops (when the actual test is
    # running), then longer timeouts after that (while the loop shuts down).
    # You can do that this way:
    #
    #   long_timeout = 30 # seconds
    #   short_timeout = 0.5 # seconds
    #   profile=[(long_timeout, 10), (short_timeout, 200), (long_timeout, 1)]
    #   with Watchdog("watchdog-name", initial_timeout=profile):
    #       ...
    #
    # Note that the Watchdog will continue to use the last element in initial_timeout
    # as the tap value after it exhausts the list.
    r = []
    for value, count in profile:
        r.extend([value] * count)
    return r


def finish_watchdogs(timeout=30):
    """finish() every running watchdog. WatchdogOp calls this when it stops: the
    run is over then, and a leg that ended a few frames short of its frame_limit
    must not time out while the run's other operators stop, which can take longer
    than a short timeout (cameras' stop() sleeps, for one)."""
    with watchdogs_lock:
        running = list(watchdogs)
    for watchdog in running:
        watchdog.finish(timeout)


@contextlib.contextmanager
def leg_watchdogs(legs, **kwargs):
    """A Watchdog for each leg of a stereo test, as {leg: watchdog}; kwargs go to
    each one. With one watchdog for both legs, a leg that stalls would hide behind
    the other leg's taps, and the other leg's taps would use up the timeouts of
    initial_timeout meant for this one."""
    location = caller(1)
    with contextlib.ExitStack() as stack:
        yield {
            leg: stack.enter_context(
                Watchdog(f"{leg} watchdog", location=location, **kwargs)
            )
            for leg in legs
        }


def frame_floor(frame_limit):
    """The fewest frames that a receiver with frame_limit must deliver: a fifth of
    frame_limit, at least 10, but at most half of it. frame_limit counts the
    receiver's compute() calls, and a call can end without a frame, which a full
    tensor pool or downstream queue drops."""
    return min(max(10, frame_limit // 5), (frame_limit + 1) // 2)


# RoceReceiverOp exists only in builds with HOLOLINK_BUILD_ROCE=ON.
requires_roce = pytest.mark.skipif(
    not hasattr(hololink_module.operators, "RoceReceiverOp"),
    reason="RoceReceiverOp isn't built (HOLOLINK_BUILD_ROCE=OFF)",
)

# The same, for hololink_module's RoceReceiverOp.
requires_module_roce = pytest.mark.skipif(
    not hasattr(hololink_module_operators, "RoceReceiverOp"),
    reason="RoceReceiverOp isn't built (HOLOLINK_BUILD_ROCE=OFF)",
)

# Receivers whose frames RunMonitor counts, by the last part of their operator type name.
RECEIVER_TYPE_NAMES = frozenset(
    {"LinuxReceiverOp", "RoceReceiverOp", "LinuxCoeReceiverOp", "FusaCoeCaptureOp"}
)

# Terminations of a run that went well. The receivers' frame_limit_ends_run ends
# a run REQUESTED_COMPLETED.
GOOD_TERMINATIONS = (
    hs.RunTermination.COMPLETED,
    hs.RunTermination.REQUESTED_COMPLETED,
)

# Some examples end the run at their frame limit with request_stop(), which ends it
# STOPPED; tests of those examples pass allow_stopped=True to assert_ok().
STOPPED_TERMINATIONS = (hs.RunTermination.STOPPED,)


class FrameCounterOp(hs.Operator):
    """Records the frame number of each frame a receiver emits, under "key".

    HSDK5 creates a new instance of a Python operator for each run, so the frame
    numbers are kept in the class attribute "counts".
    """

    metadata = hs.Input("metadata", capture_metadata.CaptureMetadata)

    key = hs.Param("")

    counts = {}

    def setup(self, spec):
        spec.input(self.metadata, queue_depth=16)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.metadata))

    def compute(self, context):
        metadata = self.metadata.receive_data()
        type(self).counts.setdefault(self.key, []).append(metadata.frame_number)


def describe_diagnostic(diagnostic):
    detail = getattr(diagnostic, "detail", None)
    error = getattr(detail, "error", None) if detail is not None else None
    message = getattr(error, "message", None) if error is not None else None
    return f"{diagnostic}: {message}" if message else str(diagnostic)


class RunMonitor:
    """Proves that data flowed through the HSDK5 runs started in a block:

        with utils.RunMonitor() as monitor, mock.patch("sys.argv", arguments):
            result = example.main()
        monitor.assert_ok()

    In v2.7 the tests counted frames with a subclass of the receiver operator, and
    HSDK5 operators can't be subclassed from Python. Instead, every hs.Graph built in
    the block gets a FrameCounterOp on each receiver's metadata output, and every
    run's session is kept for its termination. The receiver's frame_limit is not
    enough: it counts compute() calls, and a call can end without a frame (see
    frame_floor()).

    Receivers are counted by name; a receiver of another graph in the block with the
    same name gets its own key, e.g. "receiver#2". A run that ended badly (other than
    STOPPED, which assert_ok() may allow) fails the block as it exits, with the run's
    diagnostics, ahead of what a watchdog around it would report.
    """

    def __init__(self):
        self.frames = {}
        self.frame_limits = {}
        self.frame_numbers = {}
        self.terminations = []
        self.diagnostics = []
        self._receivers = {}  # name: frame_limit
        self._sessions = []
        self._patches = []

    def __enter__(self):
        FrameCounterOp.counts.clear()
        sessions = self._sessions
        receivers = self._receivers
        base_graph = hs.Graph
        base_run_async = hs.run_async

        class CountingGraph(base_graph):
            def op(self, factory, name, *args, **kwargs):
                handle = super().op(factory, name, *args, **kwargs)
                type_name = str(getattr(handle, "type_name", ""))
                if type_name.rsplit(".", 1)[-1] in RECEIVER_TYPE_NAMES:
                    # Recorded here, so a receiver that never emits a frame counts 0.
                    key, number = name, 1
                    while key in receivers:
                        number += 1
                        key = f"{name}#{number}"
                    receivers[key] = kwargs.get("frame_limit") or 0
                    counter = super().op(
                        FrameCounterOp, f"{name}_frame_counter", key=key
                    )
                    # Drop rather than block: the counter must never slow the receiver.
                    self.add_flow(
                        handle.metadata,
                        counter.metadata,
                        queue_policy=hs.QueuePolicy.DROP_OLDEST,
                        queue_depth=16,
                    )
                return handle

        def run_async(plan, *args, **kwargs):
            session = base_run_async(plan, *args, **kwargs)
            sessions.append(session)
            return session

        self._patches = [
            mock.patch("holoscan.Graph", CountingGraph),
            mock.patch("holoscan.run_async", run_async),
        ]
        for patch in self._patches:
            patch.start()
        return self

    def __exit__(self, *args):
        for patch in reversed(self._patches):
            patch.stop()
        self._patches = []
        self.terminations = [session.termination for session in self._sessions]
        self.diagnostics = [
            describe_diagnostic(diagnostic)
            for session in self._sessions
            for diagnostic in session.diagnostics
        ]
        self._sessions.clear()
        self.frame_numbers = {name: [] for name in self._receivers}
        self.frame_numbers.update(
            {key: list(numbers) for key, numbers in FrameCounterOp.counts.items()}
        )
        FrameCounterOp.counts.clear()
        self.frame_limits = dict(self._receivers)
        self._receivers.clear()
        self.frames = {key: len(numbers) for key, numbers in self.frame_numbers.items()}
        good = GOOD_TERMINATIONS + STOPPED_TERMINATIONS
        failed = [
            termination for termination in self.terminations if termination not in good
        ]
        if failed and args[0] is None:
            raise AssertionError(
                f"Run ended with {failed}; diagnostics: {self.diagnostics}"
            )

    def assert_ok(self, min_frames=None, allow_stopped=False):
        """Every run ended well, and every receiver delivered at least min_frames
        frames: by default frame_floor() of its frame_limit, or 10 without one.
        With allow_stopped, a run that ended STOPPED (request_stop()) ended well too."""
        assert self.terminations, "No HSDK5 run was started"
        good = GOOD_TERMINATIONS + (STOPPED_TERMINATIONS if allow_stopped else ())
        failed = [
            termination for termination in self.terminations if termination not in good
        ]
        assert not failed, f"Run ended with {failed}; diagnostics: {self.diagnostics}"
        assert self.frames, "No receiver was found in the graph"
        short = {}
        for key, count in self.frames.items():
            frame_limit = self.frame_limits.get(key, 0)
            need = min_frames
            if need is None:
                need = frame_floor(frame_limit) if frame_limit else 10
            logging.info(
                f"{key} delivered {count} frames"
                + (f" of frame_limit {frame_limit}" if frame_limit else "")
            )
            if count < need:
                short[key] = f"{count} frames, need {need}"
        assert not short, f"Too few frames received: {short}"


class CudaContext:
    """Retain the device's CUDA primary context for the duration of the block.

    Yields self exposing ``.context`` (the ``CUcontext``) and
    ``.device_ordinal``. The primary context is released on ``__exit__`` even
    when the body raises, so an early error can't leak the retained context.
    """

    def __init__(self, device_ordinal=0):
        self.device_ordinal = device_ordinal

    def __enter__(self):
        (cu_result,) = cuda.cuInit(0)
        assert cu_result == cuda.CUresult.CUDA_SUCCESS
        cu_result, self._device = cuda.cuDeviceGet(self.device_ordinal)
        assert cu_result == cuda.CUresult.CUDA_SUCCESS
        cu_result, self.context = cuda.cuDevicePrimaryCtxRetain(self._device)
        assert cu_result == cuda.CUresult.CUDA_SUCCESS
        return self

    def __exit__(self, *args):
        (cu_result,) = cuda.cuDevicePrimaryCtxRelease(self._device)
        assert cu_result == cuda.CUresult.CUDA_SUCCESS


def encode_bayer_image(rgb_image, bayer_format):
    sr = rgb_image[:, :, 0]
    sg = rgb_image[:, :, 1]
    sb = rgb_image[:, :, 2]
    dtype = cp.uint16
    height, width, elements_per_pixel = rgb_image.shape
    bayer_height, bayer_width = height, width
    if bayer_format == hololink_module.sensors.csi.BayerFormat.RGGB:
        # upper_line is red0, green0, red1, green1, ...
        r, g = sr.ravel(), sg.ravel()
        assert r.size == g.size
        c = cp.empty(((r.size + g.size) // 2,), dtype=dtype)
        c[0::2] = r[::2]
        c[1::2] = g[::2]
        upper_line = c.reshape(height, bayer_width)
        upper_line = upper_line[::2]  # only take every other line
        # lower_line is green0, blue0, green1, blue1, ...
        g, b = sg.ravel(), sb.ravel()
        c = cp.empty(((g.size + b.size) // 2,), dtype=dtype)
        c[0::2] = g[::2]
        c[1::2] = b[::2]
        lower_line = c.reshape(height, bayer_width)
        lower_line = lower_line[::2]  # only take every other line
        bayer_image = cp.stack([upper_line, lower_line], axis=1).reshape(
            bayer_height, bayer_width
        )
    elif bayer_format == hololink_module.sensors.csi.BayerFormat.GBRG:
        # upper_line is green0, blue0, green1, blue1, ...
        g, b = sg.ravel(), sb.ravel()
        assert g.size == b.size
        c = cp.empty(((g.size + b.size) // 2,), dtype=dtype)
        c[0::2] = g[::2]
        c[1::2] = b[::2]
        upper_line = c.reshape(height, bayer_width)
        upper_line = upper_line[::2]  # only take every other line
        # lower_line is red0, green0, red1, green1, ...
        r, g = sr.ravel(), sg.ravel()
        c = cp.empty(((r.size + g.size) // 2,), dtype=dtype)
        c[0::2] = r[::2]
        c[1::2] = g[::2]
        lower_line = c.reshape(height, bayer_width)
        lower_line = lower_line[::2]  # only take every other line
        bayer_image = cp.stack([upper_line, lower_line], axis=1).reshape(
            bayer_height, bayer_width
        )
    else:
        assert False and 'Unexpected image format "%s".' % (bayer_format,)
    return bayer_image


def bayer_image_to_csi(pixel_format, bayer_image, start_byte, received_line_bytes):
    if pixel_format == hololink_module.sensors.csi.PixelFormat.RAW_8:
        csi_image = encode_raw_8_bayer_image(bayer_image)
    elif pixel_format == hololink_module.sensors.csi.PixelFormat.RAW_12:
        csi_image = encode_raw_12_bayer_image(bayer_image)
    elif pixel_format == hololink_module.sensors.csi.PixelFormat.RAW_10:
        csi_image = encode_raw_10_bayer_image(bayer_image)
    else:
        assert False and f"Unexpected {pixel_format=}"
    rows, cols = csi_image.shape
    expected_lines = cp.resize(csi_image, (rows, received_line_bytes))
    start_metadata = cp.empty((start_byte,), dtype=cp.uint8)
    return cp.concatenate((start_metadata, expected_lines.ravel()))
    # with_prefix = cp.array([start_metadata, expected_lines.ravel()])
    # return with_prefix.ravel()


def make_csi_from_image_file(
    height,
    width,
    pixel_format,
    bayer_format,
    start_byte,
    received_line_bytes,
    filename,
):
    # Workaround for the fact that pytest globally configures the
    # logging line with our `log_timestamp_s` and `tid` values,
    # but PIL uses module-specific loggers (which don't inherit
    # filters from the root logger).  Our workaround isn't great--
    # this will crash the program if PIL logs something at WARN
    # level-- which isn't a problem for now.
    logging.getLogger("PIL").setLevel(logging.WARN)
    image = (
        PIL.Image.open(filename)
        .convert("RGB")
        .resize((width, height), PIL.Image.LANCZOS)
    )
    uint16_image = cp.array(image, dtype=np.uint16)  # only 8 bits are set
    if pixel_format == hololink_module.sensors.csi.PixelFormat.RAW_10:
        uint16_image[:] <<= 2  # use 10 bits
    else:
        assert False and "untested mode"
    rgb_image = uint16_image.reshape(height, width, 3)
    bayer_image = encode_bayer_image(rgb_image, bayer_format)
    csi_bytes = bayer_image_to_csi(
        pixel_format, bayer_image, start_byte, received_line_bytes
    )
    return csi_bytes
