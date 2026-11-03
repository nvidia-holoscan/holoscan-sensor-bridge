# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Hardware-free checks of the board tools against the HSDK5 bindings: the `hololink` command
# (python/tools/hololink.py), python/tools/imx274.py and polarfire_esb.py, and hsb_flasher's
# PolarFire strategy. Every hololink name they use exists, every subcommand starts, and the
# bindings they need behave without a board.

import ast
import pathlib
import subprocess
import sys
import time

import pytest

import hololink as hl

SOURCE = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = [
    "python/tools/hololink.py",
    "python/tools/imx274.py",
    "python/tools/polarfire_esb.py",
    "tools/hsb_flasher/firmware_flash_strategies/hsb_polarfire.py",
]


def _hololink_names(path):
    """(line, dotted name) for each attribute chain on the name the script imports hololink as."""
    tree = ast.parse(path.read_text(), str(path))
    aliases = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "hololink"
    }
    for node in ast.walk(tree):
        names = []
        while isinstance(node, ast.Attribute):
            names.append(node.attr)
            node = node.value
        if names and isinstance(node, ast.Name) and node.id in aliases:
            yield node.lineno, ".".join(reversed(names))


@pytest.mark.parametrize("script", SCRIPTS)
def test_every_hololink_name_resolves(script):
    missing = set()
    for line, name in _hololink_names(SOURCE / script):
        value = hl
        for part in name.split("."):
            if not hasattr(value, part):
                missing.add(f"line {line}: hololink.{name}")
                break
            value = getattr(value, part)
    assert not missing, sorted(missing)


HOLOLINK_SUBCOMMANDS = [
    "hsb_ip_version",
    "read_uint32",
    "write_uint32",
    "reset",
    "ptp-relock",
    "set-ip",
    "enumerate",
    "health",
    "program",
]
IMX274_SUBCOMMANDS = [
    "get_register",
    "set_register",
    "configure",
    "start",
    "reset",
    "pattern",
]


def _tool(*args):
    return subprocess.run(
        [sys.executable, "-m", *args], capture_output=True, text=True, timeout=120
    )


@pytest.mark.parametrize(
    "command",
    [["hololink.tools.hololink"]]
    + [["hololink.tools.hololink", s] for s in HOLOLINK_SUBCOMMANDS]
    + [["hololink.tools.imx274"]]
    + [["hololink.tools.imx274", s] for s in IMX274_SUBCOMMANDS]
    + [["hololink.tools.polarfire_esb"]],
    ids=lambda command: " ".join(command).replace("hololink.tools.", ""),
)
def test_tool_starts(command):
    result = _tool(*command, "--help")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_set_ip_points_to_the_cpp_tool():
    result = _tool(
        "hololink.tools.hololink", "set-ip", "00:11:22:33:44:55", "192.168.0.9"
    )
    assert result.returncode == 1
    assert "hololink-set-ip" in result.stderr


def test_force_needs_an_fpga_uuid():
    result = _tool("hololink.tools.hololink", "--force", "hsb_ip_version")
    assert result.returncode == 2
    assert "--force needs --fpga-uuid" in result.stderr


def test_metadata_from_a_dict():
    metadata = hl.Metadata({"peer_ip": "192.168.0.2", "control_port": 8192})
    assert metadata["peer_ip"] == "192.168.0.2"
    assert metadata["control_port"] == 8192
    assert (
        repr(metadata) == "Metadata({'control_port': 8192, 'peer_ip': '192.168.0.2'})"
    )
    with pytest.raises(TypeError):
        hl.Metadata({"control_port": 8192.0})


def test_hsb_log_level():
    level = hl.get_hsb_log_level()
    try:
        hl.set_hsb_log_level(50)
        assert hl.get_hsb_log_level() == 50
    finally:
        hl.set_hsb_log_level(level)


def test_find_channel_raises_at_its_timeout():
    # Nothing announces itself as 127.0.0.2.
    for find_channel in (hl.find_channel, hl.Enumerator.find_channel):
        start = time.monotonic()
        with pytest.raises(RuntimeError):
            find_channel(channel_ip="127.0.0.2", timeout=hl.Timeout(0.2))
        assert time.monotonic() - start < 5


def test_enumerated_returns_at_its_timeout():
    start = time.monotonic()
    hl.Enumerator.enumerated(lambda metadata: True, hl.Timeout(0.2))
    assert time.monotonic() - start < 5
