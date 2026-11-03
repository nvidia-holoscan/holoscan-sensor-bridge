# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Hardware-free checks of hololink.set_uuid_strategy(). Enumerator's strategy registry lives until
# the process ends, so it holds a C++ copy of a Python BasicEnumerationStrategy: holding the Python
# object would keep it past the interpreter, and nanobind would report it leaked at every exit.

import subprocess
import sys

import hololink as hl

# No board enumerates with these UUIDs.
_UUID = "00000000-0000-4000-8000-000000005701"
_EXIT_UUID = "00000000-0000-4000-8000-000000005702"


def _strategy(total_sensors):
    return hl.BasicEnumerationStrategy(
        hl.Metadata(),
        total_sensors=total_sensors,
        total_dataplanes=1,
        sifs_per_sensor=1,
    )


def test_set_uuid_strategy_returns_the_strategy_it_replaces():
    strategy = _strategy(4)
    assert hl.set_uuid_strategy(_UUID, strategy) is None
    previous = hl.set_uuid_strategy(_UUID, _strategy(2))
    # The registry held a copy, not the Python object.
    assert isinstance(previous, hl.BasicEnumerationStrategy)
    assert previous is not strategy
    # A strategy an earlier call returned can be registered again.
    assert isinstance(
        hl.set_uuid_strategy(_UUID, previous), hl.BasicEnumerationStrategy
    )


def test_a_registered_strategy_is_not_reported_leaked_at_exit():
    script = (
        "import hololink as hl\n"
        f"hl.set_uuid_strategy({_EXIT_UUID!r}, hl.BasicEnumerationStrategy(hl.Metadata()))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "leaked" not in result.stderr, result.stderr
