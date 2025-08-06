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

import logging
import os
import queue
import threading
import time
import traceback

import numpy as np

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
    """

    def __init__(self, name, timeout=None, initial_timeout=None):
        self._name = name
        self._caller = caller()
        self._str_id = f'watchdog@{id(self):#x} ({self._caller}) "{self._name}"'
        logging.trace(f"Creating {self._str_id}")
        if initial_timeout is None:
            # The dummy math here raises an exception if users didn't specify
            # a timeout parameter.
            initial_timeout = [timeout + 0]
        try:
            self._initial_timeout = iter(initial_timeout)
        except TypeError:
            self._initial_timeout = iter([initial_timeout])
        self._next_timeout = timeout
        self._q = queue.Queue()
        self._count = 0
        self._lock = threading.Lock()
        self._tap_time = None
        self._last_timeout = None
        self._thread = None
        self._thread_lock = threading.Lock()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def start(self):
        with self._thread_lock:
            assert self._thread is None
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self):
        with self._thread_lock:
            if self._thread is None:
                return
            self._q.put(None)
            self._thread.join()
            self._thread = None

    def tap(self, timeout=None):
        self._count += 1
        logging.trace(f"tapping {self._str_id} count={self._count} {timeout=}")
        self._tap_time = time.monotonic()
        self._q.put(self._get_next_timeout(timeout))

    def _get_next_timeout(self, user_value=None):
        with self._lock:
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

    def _run(self):
        hololink_module.NvtxTrace.setThreadName(self._name)
        logging.trace(f"running {self._str_id}.")
        try:
            timeout = self._get_next_timeout()
            while True:
                timeout = self._q.get(block=True, timeout=timeout)
                if timeout is None:
                    logging.trace(f"closing {self._str_id}.")
                    return
        except queue.Empty:
            pass
        dt = "N/A"
        if self._tap_time is not None:
            now = time.monotonic()
            dt = now - self._tap_time
        message = f"{self._str_id} timed out, count={self._count}, {timeout=}, time since last tap={dt}."
        logging.trace(message)
        raise Exception(message)

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


receiver_count = 0


class MockedLinuxReceiverOperator(hololink_module.operators.LinuxReceiverOperator):
    """
    Use with unittest.mock("hololink.operators.LinuxReceiverOperator")
    to assert fail when the stack doesn't receive a frame in time.
    """

    def __init__(self, *args, **kwargs):
        logging.info("Using MockedLinuxReceiverOperator.")
        super().__init__(*args, **kwargs)
        global receiver_count
        receiver_count = self._count

    def compute(self, op_input, op_output, context):
        r = super().compute(op_input, op_output, context)
        # Allow test fixturing to check the number of times
        # we've been called.
        global receiver_count
        receiver_count = self._count
        return r

    def timeout(self, op_input, op_output, context):
        logging.error(f"Frame reception timeout, {self._count=}.")
        if self._count > 10:
            assert False


class PriorityScheduler:
    def __init__(self):
        self._scheduler = os.sched_getscheduler(0)
        self._params = os.sched_getparam(0)

    def __enter__(self):
        logging.debug("Setting scheduler.")
        sched_priority = self._params.sched_priority + 1
        sched_param = os.sched_param(sched_priority=sched_priority)
        os.sched_setscheduler(0, os.SCHED_FIFO, sched_param)

    def __exit__(self, *args):
        logging.debug("Resetting scheduler.")
        os.sched_setscheduler(0, self._scheduler, self._params)
