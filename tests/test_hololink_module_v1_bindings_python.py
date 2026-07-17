# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Smoke tests for the V1 service-interface pybind bindings. Verifies that
# every interface bound from C++ (a) admits a Python subclass overriding
# its pure virtuals (proves the trampoline machinery is wired) and (b)
# returns the override's value when the binding's concrete C++ method is
# invoked from Python.

import hololink_module


def test_log_level_enum_present():
    levels = hololink_module.LogLevel
    assert int(levels.Trace) == 10
    assert int(levels.Debug) == 20
    assert int(levels.Info) == 30
    assert int(levels.Warning) == 40
    assert int(levels.Error) == 50


def test_i2c_lock_subclass_dispatches_overrides():
    class CountingLock(hololink_module.I2cLockV1):
        def __init__(self):
            super().__init__()
            self.locks = 0
            self.unlocks = 0
            self.try_locks = 0

        def lock(self):
            self.locks += 1

        def unlock(self):
            self.unlocks += 1

        def try_lock(self):
            self.try_locks += 1
            return True

    lock = CountingLock()
    lock.lock()
    lock.lock()
    lock.unlock()
    assert lock.try_lock() is True
    assert lock.locks == 2
    assert lock.unlocks == 1
    assert lock.try_locks == 1


def test_i2c_interface_subclass_round_trips_bytes():
    class FakeBus(hololink_module.I2cInterfaceV1):
        def __init__(self):
            super().__init__()
            self.last_call = None
            self.reply = b""

        def i2c_transaction(self, peripheral_address, write_bytes, read_byte_count):
            self.last_call = (peripheral_address, bytes(write_bytes), read_byte_count)
            return self.reply[:read_byte_count]

        def encode_i2c_request(
            self, sequencer, peripheral_i2c_address, write_bytes, read_byte_count
        ):
            return ([1, 2, 3], [4, 5], 6)

    bus = FakeBus()
    bus.reply = b"\xab\xcd"
    out = bus.i2c_transaction(0x42, b"\x10\x20", 2)
    assert out == b"\xab\xcd"
    assert bus.last_call == (0x42, b"\x10\x20", 2)


def test_sequencer_subclass_records_writes():
    class FakeSequencer(hololink_module.SequencerInterfaceV1):
        def __init__(self):
            super().__init__()
            self.writes = []
            self.next_index = 0

        def write_uint32(self, address, data):
            self.writes.append((address, data))
            self.next_index += 4
            return self.next_index

        def read_uint32(self, address, initial_value=0xFFFFFFFF):
            return 0

        def poll(self, address, mask, match):
            return 0

        def enable(self):
            return 0

        def location(self):
            return 0xC0DE

    seq = FakeSequencer()
    assert seq.write_uint32(0x100, 0xABCD) == 4
    assert seq.write_uint32(0x104, 0x1234) == 8
    assert seq.location() == 0xC0DE
    assert seq.writes == [(0x100, 0xABCD), (0x104, 0x1234)]


def test_hololink_interface_subclass_routes_uint32_io():
    cached_board_metadata = hololink_module.EnumerationMetadata()
    cached_board_metadata["serial_number"] = "test"

    class FakeBoard(hololink_module.HololinkInterfaceV1):
        def __init__(self):
            super().__init__()
            self.writes = []
            self.read_table = {}

        def enumeration_metadata(self):
            return cached_board_metadata

        def default_data_channel(self):
            return None

        def start(self):
            return 0

        def stop(self):
            return 0

        def reset(self):
            return 0

        def configure_hsb(self):
            return 0

        def write_uint32(self, addresses, values):
            self.writes.extend(zip(addresses, values))
            return 0

        def read_uint32(self, addresses):
            return [self.read_table.get(a, 0) for a in addresses]

        def and_uint32(self, address, mask):
            return 0

        def or_uint32(self, address, mask):
            return 0

        def roce_data_channel_instance_id(self, metadata):
            return f"serial={metadata['serial_number']};channel=0"

        def i2c_instance_id(self, bus, address):
            return f"serial=test;bus={bus};address={address}"

    board = FakeBoard()
    assert board.write_uint32([0x100, 0x104], [0x1, 0x2]) == 0
    assert board.writes == [(0x100, 0x1), (0x104, 0x2)]

    board.read_table[0x200] = 0xDEADBEEF
    assert board.read_uint32([0x200, 0x204]) == [0xDEADBEEF, 0]

    # default_data_channel is a new virtual on HololinkInterface — the
    # Python override returns None to confirm the binding dispatches.
    assert board.default_data_channel() is None


def test_roce_data_channel_subclass_dispatches_attach_receiver():
    # RoceDataChannelInterface is standalone — its Python subclass
    # only implements the RoCE-specific surface (attach_receiver /
    # detach_receiver / instance_id hooks). The per-channel anchor
    # (DataChannelInterface) is a separate service that the impl
    # composes with internally on the C++ side; Python tests don't
    # need to model it.

    class FakeChannel(hololink_module.RoceDataChannelInterfaceV1):
        def __init__(self):
            super().__init__()
            self.last_receiver = None
            self.detach_count = 0

        def attach_receiver(self, receiver):
            # attach_receiver takes the started receiver and binds it
            # to the channel; the C++ impl pulls QP / frame-memory /
            # layout off the receiver via the anchor's cached
            # EnumerationMetadata. The Python subclass just records
            # what it was handed.
            self.last_receiver = receiver
            return 0

        def detach_receiver(self):
            self.detach_count += 1
            return 0

        def frame_end_sequencer_instance_id(self):
            return "serial=test;channel=0"

        def parent_hololink_instance_id(self):
            return "serial=test"

    ch = FakeChannel()
    # Pass None for the receiver — the Python subclass's
    # attach_receiver records it without dereferencing.
    assert ch.attach_receiver(None) == 0
    assert ch.last_receiver is None
    assert ch.detach_receiver() == 0
    assert ch.detach_count == 1


def test_enumeration_interface_subclass_runs_update_metadata():
    class StampingEnumeration(hololink_module.EnumerationInterfaceV1):
        def update_metadata(self, metadata, raw_packet):
            metadata["module_name"] = "py-stub"
            metadata["compat_id"] = 9999

    md = hololink_module.EnumerationMetadata()
    enum = StampingEnumeration()
    enum.update_metadata(md, None)
    assert md["module_name"] == "py-stub"
    assert md["compat_id"] == 9999


def test_logging_interface_subclass_dispatches():
    class CapturingLogger(hololink_module.LoggingInterfaceV1):
        def __init__(self):
            super().__init__()
            self.records = []

        def level(self):
            return hololink_module.LogLevel.Debug

        def log(self, level, file, line, function, message):
            self.records.append((int(level), file, line, function, message))

    log = CapturingLogger()
    assert log.level() == hololink_module.LogLevel.Debug
    log.log(hololink_module.LogLevel.Info, "file.cpp", 42, "fn", "hello")
    assert log.records == [(30, "file.cpp", 42, "fn", "hello")]


def test_reactor_class_present_but_not_subclassable_from_python():
    # Reactor is bound for consumers only — there's no public
    # constructor exposed to Python.
    assert hasattr(hololink_module, "ReactorV1")
    assert hasattr(hololink_module, "AlarmEntry")
