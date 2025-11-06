# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import threading
import time

import hololink


def test_reactor_singleton():
    """Test that get_reactor() returns a singleton instance."""
    reactor1 = hololink.Reactor.get_reactor()
    reactor2 = hololink.Reactor.get_reactor()

    assert reactor1 is reactor2
    assert reactor1 is not None


def test_reactor_now():
    """Test that now() returns monotonic time."""
    reactor = hololink.Reactor.get_reactor()

    time1 = reactor.now()
    time.sleep(0.01)  # Sleep 10ms
    time2 = reactor.now()

    assert isinstance(time1, float)
    assert isinstance(time2, float)
    assert time2 > time1
    # Should be roughly 10ms difference (with some tolerance)
    assert 0.008 < (time2 - time1) < 0.02


def test_reactor_add_callback():
    """Test adding immediate callbacks."""
    reactor = hololink.Reactor.get_reactor()

    callback_called = threading.Event()
    callback_result = []

    def test_callback():
        callback_result.append("called")
        callback_called.set()

    reactor.add_callback(test_callback)

    # Wait for callback to be executed (should be immediate)
    assert callback_called.wait(timeout=1.0)
    assert callback_result == ["called"]


def test_reactor_add_alarm_s():
    """Test adding time-based alarms with seconds offset."""
    reactor = hololink.Reactor.get_reactor()

    alarm_called = threading.Event()
    alarm_result = []
    start_time = time.time()

    def alarm_callback():
        alarm_result.append(time.time() - start_time)
        alarm_called.set()

    # Schedule alarm to fire in 0.1 seconds
    alarm_handle = reactor.add_alarm_s(0.1, alarm_callback)  # noqa: F841

    # Wait for alarm to fire
    assert alarm_called.wait(timeout=1.0)

    # Check that it fired roughly at the right time
    assert len(alarm_result) == 1
    assert 0.08 < alarm_result[0] < 0.15  # Some tolerance for timing


def test_reactor_add_alarm():
    """Test adding alarms at specific times."""
    reactor = hololink.Reactor.get_reactor()

    alarm_called = threading.Event()
    alarm_result = []

    def alarm_callback():
        alarm_result.append(reactor.now())
        alarm_called.set()

    # Schedule alarm to fire 0.1 seconds from now
    target_time = reactor.now() + 0.1
    alarm_handle = reactor.add_alarm(target_time, alarm_callback)  # noqa: F841

    # Wait for alarm to fire
    assert alarm_called.wait(timeout=1.0)

    # Check that it fired roughly at the target time
    assert len(alarm_result) == 1
    actual_time = alarm_result[0]
    assert abs(actual_time - target_time) < 0.05  # 50ms tolerance


def test_reactor_cancel_alarm():
    """Test cancelling scheduled alarms."""
    reactor = hololink.Reactor.get_reactor()

    alarm_called = threading.Event()

    def alarm_callback():
        alarm_called.set()

    # Schedule alarm for 0.2 seconds
    alarm_handle = reactor.add_alarm_s(0.2, alarm_callback)

    # Cancel the alarm immediately
    reactor.cancel_alarm(alarm_handle)

    # Wait a bit longer than the alarm would have fired
    time.sleep(0.3)

    # The alarm should not have been called
    assert not alarm_called.is_set()


def test_reactor_multiple_alarms():
    """Test multiple alarms firing in correct order."""
    reactor = hololink.Reactor.get_reactor()

    results = []
    results_lock = threading.Lock()
    all_alarms_done = threading.Event()

    def make_callback(name):
        def callback():
            with results_lock:
                results.append(name)
                if len(results) == 3:
                    all_alarms_done.set()

        return callback

    # Schedule alarms in reverse order
    reactor.add_alarm_s(0.15, make_callback("third"))
    reactor.add_alarm_s(0.05, make_callback("first"))
    reactor.add_alarm_s(0.10, make_callback("second"))

    # Wait for all alarms to complete
    assert all_alarms_done.wait(timeout=1.0)

    # Check they fired in the correct order
    assert results == ["first", "second", "third"]


def test_reactor_fd_callback():
    """Test file descriptor callbacks using a pipe."""
    reactor = hololink.Reactor.get_reactor()

    # Create a pipe for testing
    read_fd, write_fd = os.pipe()

    try:
        callback_called = threading.Event()
        callback_result = []

        def fd_callback(fd, events):
            s = os.read(fd, 2048)
            callback_result.append((fd, events, s))
            callback_called.set()

        # Add file descriptor callback for read events
        reactor.add_fd_callback(read_fd, fd_callback, 0x001)  # POLLIN

        # Write some data to trigger the callback
        message = b"test"
        os.write(write_fd, message)

        # Wait for callback
        assert callback_called.wait(timeout=1.0)

        # Verify callback was called with correct parameters
        assert len(callback_result) == 1
        fd, events, received_message = callback_result[0]
        assert fd == read_fd
        assert events & 0x001  # POLLIN should be set
        assert received_message == message

        # Clean up
        reactor.remove_fd_callback(read_fd)

    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_reactor_remove_fd_callback():
    """Test removing file descriptor callbacks."""
    reactor = hololink.Reactor.get_reactor()

    # Create a pipe for testing
    read_fd, write_fd = os.pipe()

    try:
        callback_called = threading.Event()

        def fd_callback(fd, events):
            callback_called.set()

        # Add and then remove the callback
        reactor.add_fd_callback(read_fd, fd_callback)
        reactor.remove_fd_callback(read_fd)

        # Write data - callback should not be called
        os.write(write_fd, b"test")

        # Wait briefly
        time.sleep(0.1)

        # Callback should not have been called
        assert not callback_called.is_set()

    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_reactor_is_current_thread():
    """Test the is_current_thread method."""
    reactor = hololink.Reactor.get_reactor()

    # From the main thread, should return False (reactor runs in its own thread)
    assert not reactor.is_current_thread()

    # Test from within a reactor callback
    current_thread_result = []
    callback_done = threading.Event()

    def callback_check_thread():
        current_thread_result.append(reactor.is_current_thread())
        callback_done.set()

    reactor.add_callback(callback_check_thread)

    assert callback_done.wait(timeout=1.0)
    assert len(current_thread_result) == 1
    assert current_thread_result[0] is True  # Should be True from within reactor thread


def test_reactor_thread_safety():
    """Test that reactor operations are thread-safe."""
    reactor = hololink.Reactor.get_reactor()

    results = []
    results_lock = threading.Lock()

    def worker_thread(thread_id):
        for i in range(10):
            # Add callbacks from multiple threads
            def callback(tid=thread_id, idx=i):
                with results_lock:
                    results.append(f"thread_{tid}_callback_{idx}")

            reactor.add_callback(callback)

        # Also add alarms from multiple threads
        def alarm_callback(tid=thread_id):
            with results_lock:
                results.append(f"thread_{tid}_alarm")

        reactor.add_alarm_s(0.1, alarm_callback)

    # Start multiple worker threads
    threads = []
    for i in range(3):
        t = threading.Thread(target=worker_thread, args=(i,))
        threads.append(t)
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    # Wait a bit for all callbacks/alarms to execute
    time.sleep(0.2)

    # Check that we got results from all threads
    with results_lock:
        # Should have 30 callbacks (3 threads * 10 callbacks each) + 3 alarms
        assert len(results) == 33

        # Check that all thread IDs are represented
        thread_0_items = [r for r in results if "thread_0" in r]
        thread_1_items = [r for r in results if "thread_1" in r]
        thread_2_items = [r for r in results if "thread_2" in r]

        assert len(thread_0_items) == 11  # 10 callbacks + 1 alarm
        assert len(thread_1_items) == 11  # 10 callbacks + 1 alarm
        assert len(thread_2_items) == 11  # 10 callbacks + 1 alarm
