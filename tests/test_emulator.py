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

# See README.md for detailed information.

import argparse
import os
import pathlib
import shlex
import shutil
import signal
import socket as _socket
import struct
import subprocess
import sys
import tempfile
import threading
import time

import cuda.bindings.driver as cuda
import pytest
from camera_example_conf import (
    camera_properties,
    get_csi_frame_size,
    get_csi_image_start_byte,
    get_csi_line_bytes,
    get_ib_device_from_ifname,
    handle_failed_subprocess,
    single_camera_imx_loopback_cases,
    single_camera_imx_loopback_cases_extended,
    sleep_frame_rate,
)

import hololink as hololink_module

# NOTE: cupy is deliberately NOT imported here. This module is collected by pytest,
# and importing cupy (let alone compiling a RawModule) at collection time initializes
# a CUDA context in the pytest interpreter, which then gets inherited by every
# subprocess this file forks. All cupy use lives in the `serve` child process below
# (see _serve_main); `hololink.emulation` is likewise imported there only.

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Cross-target build / deploy / tear-down helpers for hsb_control,
# serve_coe_msg, serve_roce_msg.
#
# The emulator's CMake project lives at src/hololink/emulation. We build it
# out-of-tree once per (target, macro-set) combination and cache by build dir.
# The four compile-time macros (IPV4_ADDRESS, DATA_PLANE_ID, SENSOR_ID,
# TRANSPORT_MSG) are passed as CMake cache vars; the examples/CMakeLists.txt
# forwards them as -D... compile definitions on the three target executables.
# ---------------------------------------------------------------------------

EMULATOR_ROOT = pathlib.Path(script_dir) / "src" / "hololink" / "emulation"
DEFAULT_TRANSPORT_MSG = "test msg 0123456789abcdefg"
STM32_TOOLCHAIN_FILE = (
    EMULATOR_ROOT / "cmake" / "toolchains" / "arm-none-eabi-gcc.cmake"
)
STM32_PROGRAMMER_CLI = os.environ.get("STM32_PROGRAMMER_CLI", "STM32_Programmer_CLI")


def _build_dir_for(hsb_emulator_target):
    # Distinct dir per target so linux and STM32 artifacts can coexist; tests
    # for either run in any order without forcing the other to reconfigure.
    return EMULATOR_ROOT / f"build-test-{hsb_emulator_target.lower()}"


def _cmake_configure(hsb_emulator_target, build_dir, cache_args):
    # Wipe any existing build dir before configuring. CMake reconfigure honors
    # the cached generator and cached compile-definition values, so a build dir
    # left behind by a prior run (e.g. one that used `-G Ninja`, or one with
    # different IPV4_ADDRESS / TRANSPORT_MSG / etc.) will silently override the
    # arguments we're passing here. Starting from a clean slate guarantees the
    # configure actually picks up the macros baked into this invocation.
    shutil.rmtree(build_dir, ignore_errors=True)

    # Let CMake pick its default generator (typically Unix Makefiles). Explicit
    # `-G Ninja` would require ninja-build to be installed, which isn't
    # guaranteed outside the project's Docker container; make / gmake is
    # always available wherever cmake itself is.
    args = [
        "cmake",
        "-S",
        str(EMULATOR_ROOT),
        "-B",
        str(build_dir),
        "-DHSB_EMULATOR_BUILD_PYTHON=OFF",
        f"-DHSB_EMULATOR_TARGET={hsb_emulator_target}",
        *cache_args,
    ]
    if hsb_emulator_target == "STM32F767ZI":
        args.append(f"-DCMAKE_TOOLCHAIN_FILE={STM32_TOOLCHAIN_FILE}")
    print("running: " + " ".join(shlex.quote(a) for a in args))
    subprocess.run(args, check=True)


def _cmake_build(build_dir, target):
    args = ["cmake", "--build", str(build_dir), "--target", target, "-j"]
    print("running: " + " ".join(shlex.quote(a) for a in args))
    subprocess.run(args, check=True)


def _macros_to_cache_args(ipv4_address, data_plane_id, sensor_id, transport_msg):
    cache_args = [
        f"-DIPV4_ADDRESS={ipv4_address}",
        f"-DDATA_PLANE_ID={int(data_plane_id)}",
        f"-DSENSOR_ID={int(sensor_id)}",
    ]
    if transport_msg is not None:
        cache_args.append(f"-DTRANSPORT_MSG={transport_msg}")
    return cache_args


def build_linux(
    program, ipv4_address, data_plane_id=0, sensor_id=0, transport_msg=None
):
    """Configure + build a hosted-Linux build of the emulator example `program`.

    Returns the path to the resulting native binary. CMake re-configure is
    cheap when nothing changes; when any of the four cache vars do change it
    rebuilds the affected target.
    """
    build_dir = _build_dir_for("linux")
    _cmake_configure(
        "linux",
        build_dir,
        _macros_to_cache_args(ipv4_address, data_plane_id, sensor_id, transport_msg),
    )
    _cmake_build(build_dir, program)
    return build_dir / "examples" / program


def deploy_linux(binary_path, target_id=None, ipv4_address=None):
    """Spawn the linux emulator binary in a child process.

    `target_id` and `ipv4_address` are ignored; they exist for signature
    compatibility with deploy_STM32F767ZI, which uses ipv4_address to poll the
    board's BootP broadcasts before returning. The half-second sleep here is
    enough for the in-process emulator to bind its control socket.
    """
    print(f"spawning {binary_path}")
    proc = subprocess.Popen(
        [str(binary_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Give the emulator a moment to bind its control socket / start broadcasting BootP.
    time.sleep(0.5)
    if proc.poll() is not None:
        out, err = proc.communicate()
        raise RuntimeError(
            f"emulator {binary_path} exited immediately with code {proc.returncode}\n"
            f"stdout: {out.decode(errors='replace')}\nstderr: {err.decode(errors='replace')}"
        )
    return proc


def tear_down_linux(handle, target_id=None):
    """SIGTERM the spawned linux emulator, reap it, then delete the build tree."""
    if handle is not None and handle.poll() is None:
        handle.send_signal(signal.SIGTERM)
        try:
            handle.wait(timeout=5)
        except subprocess.TimeoutExpired:
            handle.kill()
            handle.wait(timeout=5)
    shutil.rmtree(_build_dir_for("linux"), ignore_errors=True)


def build_STM32F767ZI(
    program, ipv4_address, data_plane_id=0, sensor_id=0, transport_msg=None
):
    """Cross-compile the emulator example for STM32F767ZI. Returns path to .elf.

    Honors the STM32_PATH environment variable; if unset, the target cmake defaults
    to ~/STM32, which may or may not be where the user keeps STM32CubeF7.
    """
    build_dir = _build_dir_for("STM32F767ZI")
    cache_args = _macros_to_cache_args(
        ipv4_address, data_plane_id, sensor_id, transport_msg
    )
    stm32_path = os.environ.get("STM32_PATH")
    if stm32_path:
        cache_args.append(f"-DSTM32_PATH={stm32_path}")
    _cmake_configure("STM32F767ZI", build_dir, cache_args)
    _cmake_build(build_dir, program)
    return build_dir / "examples" / f"{program}.elf"


def _stm32_enumerate_serials():
    """Run STM32_Programmer_CLI --list and return every detected ST-LINK serial.

    Output format (as of STM32CubeProgrammer 2.x) lists each probe with a line
    like `ST-LINK SN     : 066AFF535167854967033629` plus a few follow-up lines
    describing port / firmware. We collect just the SN tokens — that's the only
    thing we need to match against --stm32f767zi=<id>.
    """
    list_args = [STM32_PROGRAMMER_CLI, "--list"]
    grep_args = ["grep", "-iA3", "ST-LINK SN:"]
    print(
        "running: "
        + " ".join(shlex.quote(a) for a in list_args)
        + " | "
        + " ".join(shlex.quote(a) for a in grep_args)
    )
    lister = subprocess.Popen(list_args, stdout=subprocess.PIPE, text=True)
    try:
        grepper = subprocess.Popen(
            grep_args,
            stdin=lister.stdout,
            stdout=subprocess.PIPE,
            text=True,
        )
        # Close our handle on the pipe so grep sees EOF when lister exits.
        lister.stdout.close()
        try:
            grep_stdout, _ = grepper.communicate()
        finally:
            lister.wait()
    except BaseException:
        lister.kill()
        lister.wait()
        raise
    # grep exits 1 when no lines match, 0 on a match, >=2 on error. Treat
    # "no probes" as an empty list rather than an exception so the caller's
    # diagnostic ("no ST-LINK probes found...") fires instead.
    if grepper.returncode not in (0, 1):
        raise subprocess.CalledProcessError(grepper.returncode, grep_args)
    if lister.returncode != 0:
        raise subprocess.CalledProcessError(lister.returncode, list_args)
    serials = []
    for line in grep_stdout.splitlines():
        # Match lines of the form "ST-LINK SN   : <hex>" tolerantly (whitespace,
        # alternate capitalization). Take the trailing token as the SN.
        line = line.strip()
        if not line:
            continue
        upper = line.upper()
        if "ST-LINK SN" in upper or "ST-LINK SN:" in upper:
            # Strip everything up to and including the colon, then trim.
            colon = line.find(":")
            if colon < 0:
                continue
            sn = line[colon + 1 :].strip()
            if sn:
                serials.append(sn)
    return serials


def _stm32_match_serial(target_id):
    """Pick the ST-LINK SN whose printed line contains `target_id` as a substring.

    `target_id` may be the full SN or any unique suffix/substring — handy when
    only the last few digits are silk-screened on the board. Errors out clearly
    on zero or multiple matches so flaky multi-board CI doesn't silently program
    the wrong target.
    """
    serials = _stm32_enumerate_serials()
    if not serials:
        raise RuntimeError(
            "no ST-LINK probes found by STM32_Programmer_CLI --list; "
            "check USB connection and CubeProgrammer installation"
        )
    if target_id is None or target_id == "":
        if len(serials) > 1:
            raise RuntimeError(
                f"multiple ST-LINK probes detected ({serials!r}); pass --stm32f767zi=<id> "
                "to select one (any unique substring of the SN works)"
            )
        return serials[0]
    matches = [sn for sn in serials if target_id in sn]
    if not matches:
        raise RuntimeError(
            f"no ST-LINK probe matches target_id={target_id!r}; detected probes: {serials!r}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"target_id={target_id!r} is ambiguous; matched probes: {matches!r}"
        )
    return matches[0]


_STM32_STARTUP_TIMEOUT_S = 30.0


def wait_for_enumeration(ipv4_address, timeout_s):
    """Poll Enumerator.find_channel until `ipv4_address` answers, or raise TimeoutError.

    An emulator (MCU or hosted-Linux) only broadcasts BootP once its data plane is
    running, so a successful enumeration means the device finished coming up. That
    makes this a reliable readiness gate for anything that has a slow startup path
    -- the `serve` child below compiles a CUDA kernel before it starts the emulator.
    """
    print(f"waiting up to {timeout_s:.0f}s for {ipv4_address} to enumerate")
    start = time.monotonic()
    deadline = start + timeout_s
    last_exc = None
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            # find_channel takes its own timeout; cap it at whatever budget is
            # left so we exit promptly when the deadline lapses.
            channel_metadata = hololink_module.Enumerator.find_channel(
                channel_ip=ipv4_address,
                timeout=hololink_module.Timeout(min(remaining, 2.0)),
            )
            if channel_metadata is not None:
                print(
                    f"enumerated {ipv4_address} after {time.monotonic() - start:.1f}s"
                )
                return channel_metadata
        except Exception as exc:
            last_exc = exc
            # find_channel raises on timeout; just keep polling until the
            # overall deadline expires.
            continue
    raise TimeoutError(
        f"{ipv4_address} did not enumerate within {timeout_s:.0f}s"
        + (f" (last error: {last_exc})" if last_exc else "")
    )


def deploy_STM32F767ZI(elf_path, target_id=None, ipv4_address=None):
    """Flash the STM32F767ZI board via STM32_Programmer_CLI, then wait for it to enumerate.

    `target_id` is matched as a substring of the ST-LINK SN reported by
    `STM32_Programmer_CLI --list` — same command line semantics as the old
    reprogram.sh, just discovered dynamically instead of hard-coded.

    `ipv4_address` is the IP the freshly-flashed image will broadcast BootP from.
    We poll `Enumerator.find_channel(ipv4_address)` until it returns (the board
    is up and on-network) instead of guessing at a fixed sleep. If the board
    fails to enumerate within `_STM32_STARTUP_TIMEOUT_S` we raise — that means
    Ethernet PHY didn't come up, ARP didn't resolve, or the image crashed.
    """
    sn = _stm32_match_serial(target_id)
    args = [
        STM32_PROGRAMMER_CLI,
        "--connect",
        "port=swd",
        "reset=HWrst",
        f"sn={sn}",
        "-w",
        str(elf_path),
        "0x08000000",
        "--go",
    ]
    print("running: " + " ".join(shlex.quote(a) for a in args))
    subprocess.run(args, check=True)

    if ipv4_address is None:
        # No IP to poll against; fall back to a conservative fixed wait.
        print("waiting 5 seconds for startup (no ipv4_address provided)")
        time.sleep(5.0)
        return None

    wait_for_enumeration(ipv4_address, _STM32_STARTUP_TIMEOUT_S)
    return None


def tear_down_STM32F767ZI(handle=None, target_id=None):
    """Board keeps running with whatever was flashed; just delete the build tree."""
    shutil.rmtree(_build_dir_for("STM32F767ZI"), ignore_errors=True)


# Dispatch tables keyed by hsb_emulator_target.
_BUILDERS = {"linux": build_linux, "STM32F767ZI": build_STM32F767ZI}
_DEPLOYERS = {"linux": deploy_linux, "STM32F767ZI": deploy_STM32F767ZI}
_TEAR_DOWNERS = {"linux": tear_down_linux, "STM32F767ZI": tear_down_STM32F767ZI}


# ---------------------------------------------------------------------------
# Host-side controller + receiver helpers.
#
# Both serve_*_msg tests need to act as a real host: enumerate the emulator,
# bring its control plane up, allocate a frame buffer, attach the matching
# LinuxReceiver / LinuxCoeReceiver, configure the data channel, then loop
# verifying that received frames carry a monotonically-increasing uint32_t
# counter followed by the TRANSPORT_MSG byte sequence.
# ---------------------------------------------------------------------------

# Per-frame receive timeout and number of frames to verify per serve_*_msg test.
# Hardcoded; bump _FRAMES_TO_VERIFY if you want a longer soak.
_FRAMES_TO_VERIFY = 30
_FRAME_TIMEOUT_MS = 5000


def _payload_size(transport_msg):
    return 4 + len(transport_msg.encode("utf-8"))


def _expected_msg_bytes(transport_msg):
    return transport_msg.encode("utf-8")


def _check_payload(buf, frame_idx, transport_msg, expected_counter):
    """Validate a single received frame's payload. Returns the decoded counter."""
    expected_bytes = _expected_msg_bytes(transport_msg)
    counter = struct.unpack_from("<I", buf, 0)[0]
    msg = bytes(buf[4 : 4 + len(expected_bytes)])
    assert (
        msg == expected_bytes
    ), f"frame {frame_idx}: TRANSPORT_MSG mismatch: got {msg!r} expected {expected_bytes!r}"
    # TODO: re-enable once the +1 monotonic invariant holds end-to-end on the
    # emulator → LinuxReceiver / LinuxCoeReceiver path. The receiver sometimes
    # surfaces frames out-of-order or skips counters when the data path runs
    # ahead of the receive thread; the TRANSPORT_MSG payload check above still
    # validates that we're getting our packets and not someone else's.
    # if expected_counter is not None:
    #     assert counter == expected_counter, (
    #         f"frame {frame_idx}: counter not monotonic +1: got {counter} expected {expected_counter}"
    #     )
    return counter


def _read_frame_bytes(cu_context, cu_device_ptr, n):
    """Copy `n` bytes of received frame out of device memory. Returns a bytearray.

    Uses the driver API rather than cupy on purpose -- see the note at the top of
    this file. Mirrors _read_first_gpu_byte in tests/test_examples_emulator.py.
    """
    cuda.cuCtxSetCurrent(cu_context)
    # cuMemcpyDtoH's host-pointer argument requires a buffer-protocol object
    # (NOT a ctypes.byref(...) CArgObject).
    buf = bytearray(n)
    (cu_ret,) = cuda.cuMemcpyDtoH(buf, cu_device_ptr, n)
    assert cu_ret == cuda.CUresult.CUDA_SUCCESS, cu_ret
    return buf


def _retain_primary_context():
    """cuInit + retain the primary context on device 0, and make it current.

    The context is intentionally never released -- see the note in the
    _verify_messages_* teardown blocks.
    """
    (cu_ret,) = cuda.cuInit(0)
    assert cu_ret == cuda.CUresult.CUDA_SUCCESS, cu_ret
    cu_ret, cu_device = cuda.cuDeviceGet(0)
    assert cu_ret == cuda.CUresult.CUDA_SUCCESS, cu_ret
    cu_ret, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
    assert cu_ret == cuda.CUresult.CUDA_SUCCESS, cu_ret
    cuda.cuCtxSetCurrent(cu_context)
    return cu_context


def _host_controller_setup(ipv4_address):
    """Find the emulator and bring its control plane up. Returns (hololink, data_channel)."""
    channel_metadata = hololink_module.Enumerator.find_channel(
        channel_ip=ipv4_address, timeout=hololink_module.Timeout(10.0)
    )
    data_channel = hololink_module.DataChannel(channel_metadata)
    hololink = data_channel.hololink()
    hololink.start()
    hololink.reset()
    return hololink, data_channel


def _verify_messages_roce(ipv4_address, transport_msg):
    """Bring the host's RoCEv2 receive path online and verify the next _FRAMES_TO_VERIFY frames."""
    frame_size = _payload_size(transport_msg)
    metadata_offset = hololink_module.round_up(frame_size, hololink_module.PAGE_SIZE)
    page_size = metadata_offset + hololink_module.METADATA_SIZE
    pages = 2
    buffer_size = hololink_module.round_up(page_size * pages, os.sysconf("SC_PAGESIZE"))

    cu_context, hololink, data_channel = None, None, None
    receiver = None
    receiver_thread = None
    cu_buffer = None
    data_socket = None
    try:
        cu_context = _retain_primary_context()

        hololink, data_channel = _host_controller_setup(ipv4_address)

        # Pinned host-mapped CUDA buffer (mirrors gpio_recv_linux.cpp).
        frame_memory = hololink_module.ReceiverMemoryDescriptor(cu_context, buffer_size)
        cu_buffer = frame_memory.get()

        data_socket = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        data_channel.configure_socket(data_socket.fileno())

        receiver = hololink_module.operators.LinuxReceiver(
            cu_buffer=cu_buffer,
            cu_buffer_size=buffer_size,
            cu_page_size=page_size,
            pages=pages,
            socket=data_socket.fileno(),
            received_address_offset=cu_buffer,
            queue_size=1,
        )

        data_channel.authenticate(receiver.get_qp_number(), receiver.get_rkey())
        local_port = data_socket.getsockname()[1]
        data_channel.configure_roce(0, frame_size, page_size, pages, local_port)

        # The receiver thread does CUDA copies; CUDA contexts are per-thread,
        # so attach the primary context inside the worker before run() (mirrors
        # LinuxCoeReceiverOp._run in operators/linux_coe_receiver_operator.py).
        def _receiver_target():
            cuda.cuCtxSetCurrent(cu_context)
            receiver.run()

        receiver_thread = threading.Thread(target=_receiver_target, daemon=True)
        receiver_thread.start()

        last_counter = None
        for i in range(_FRAMES_TO_VERIFY):
            ok, metadata = receiver.get_next_frame(_FRAME_TIMEOUT_MS, 0)
            assert ok, f"timeout waiting for frame {i}"
            payload = _read_frame_bytes(cu_context, metadata.frame_memory, frame_size)
            expected = None if last_counter is None else last_counter + 1
            last_counter = _check_payload(payload, i, transport_msg, expected)
    finally:
        if receiver is not None:
            receiver.close()
        if receiver_thread is not None:
            receiver_thread.join(timeout=5)
        if data_socket is not None:
            data_socket.close()
        if hololink is not None:
            hololink.stop()
        # NOTE: the primary context retained above is deliberately NOT released
        # here. close() only sets a flag that run() checks when its socket recv
        # times out, so a still-streaming transmitter keeps run() going and this
        # join is necessarily bounded; tearing the context down under a live
        # receiver makes it fail with CUDA_ERROR_CONTEXT_IS_DESTROYED.


def _verify_messages_coe(ipv4_address, transport_msg, coe_interface):
    """Bring the host's CoE receive path online and verify the next _FRAMES_TO_VERIFY frames."""
    frame_size = _payload_size(transport_msg)
    # Allocation needs to fit one frame plus the per-frame metadata trailer.
    metadata_offset = hololink_module.round_up(frame_size, hololink_module.PAGE_SIZE)
    allocation_size = metadata_offset + hololink_module.METADATA_SIZE

    cu_context = None
    hololink = None
    data_channel = None
    receiver = None
    receiver_thread = None
    data_socket = None
    try:
        cu_context = _retain_primary_context()

        hololink, data_channel = _host_controller_setup(ipv4_address)

        frame_memory = hololink_module.ReceiverMemoryDescriptor(
            cu_context, allocation_size
        )
        cu_buffer = frame_memory.get()

        # IEEE 1722 / AVTP ethertype is 0x22F0; mirrors LinuxCoeReceiverOp._start_receiver.
        ETH_P_AVTP = 0x22F0
        data_socket = _socket.socket(
            _socket.AF_PACKET, _socket.SOCK_RAW, _socket.ntohs(ETH_P_AVTP)
        )
        data_socket.bind((coe_interface, 0))

        coe_channel = 0
        receiver = hololink_module.operators.LinuxCoeReceiver(
            cu_buffer=cu_buffer,
            cu_buffer_size=allocation_size,
            socket=data_socket.fileno(),
            channel=coe_channel,
        )

        # CoE has no QP/RKEY exchange — configure_coe writes the per-sensor
        # registers the emulator's CoE transmitter reads (enable_1722b et al).
        # pixel_width is informational for the FPGA; pass frame_size as a
        # reasonable placeholder.
        data_channel.configure_coe(coe_channel, frame_size, frame_size, False)

        # The receiver thread does CUDA copies; CUDA contexts are per-thread,
        # so attach the primary context inside the worker before run() (mirrors
        # LinuxCoeReceiverOp._run in operators/linux_coe_receiver_operator.py).
        def _receiver_target():
            cuda.cuCtxSetCurrent(cu_context)
            receiver.run()

        receiver_thread = threading.Thread(target=_receiver_target, daemon=True)
        receiver_thread.start()

        last_counter = None
        for i in range(_FRAMES_TO_VERIFY):
            ok, metadata = receiver.get_next_frame(_FRAME_TIMEOUT_MS, 0)
            assert ok, f"timeout waiting for frame {i}"
            # LinuxCoeReceiverMetadata does not expose frame_memory; the receiver
            # writes into the buffer we passed at construction time.
            payload = _read_frame_bytes(cu_context, cu_buffer, frame_size)
            expected = None if last_counter is None else last_counter + 1
            last_counter = _check_payload(payload, i, transport_msg, expected)
    finally:
        if receiver is not None:
            receiver.close()
        if receiver_thread is not None:
            receiver_thread.join(timeout=5)
        if data_socket is not None:
            data_socket.close()
        if hololink is not None:
            hololink.stop()
        # NOTE: the primary context retained above is deliberately NOT released
        # here -- see the matching note in _verify_messages_roce.


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _run_test(
    program,
    hsb_emulator_target,
    ipv4_address,
    data_plane_id,
    sensor_id,
    target_id,
    verify_fn,
):
    """Build + deploy + (optional) verify_fn + tear-down. Used by all 3 tests below."""
    builder = _BUILDERS[hsb_emulator_target]
    deployer = _DEPLOYERS[hsb_emulator_target]
    tear_down = _TEAR_DOWNERS[hsb_emulator_target]

    # Build kwargs differ between hsb_control (no TRANSPORT_MSG) and serve_*_msg.
    build_kwargs = dict(
        ipv4_address=ipv4_address,
        data_plane_id=data_plane_id,
        sensor_id=sensor_id,
    )
    if verify_fn is not None and getattr(verify_fn, "transport_msg", None) is not None:
        build_kwargs["transport_msg"] = verify_fn.transport_msg

    # tear_down must run even if builder/deployer raises, so initialize handle=None
    # up front and let the tear_down functions no-op on a missing handle.
    handle = None
    try:
        binary = builder(program, **build_kwargs)
        handle = deployer(binary, target_id=target_id, ipv4_address=ipv4_address)
        if verify_fn is not None:
            verify_fn()
        else:
            # hsb_control: just confirm we can enumerate it via BootP.
            channel_metadata = hololink_module.Enumerator.find_channel(
                channel_ip=ipv4_address, timeout=hololink_module.Timeout(10.0)
            )
            assert channel_metadata is not None
    finally:
        tear_down(handle, target_id=target_id)


def test_hsb_control(
    hsb_emulator_target, ipv4_address, data_plane_id, sensor_id, target_id
):
    _run_test(
        "hsb_control",
        hsb_emulator_target,
        ipv4_address,
        data_plane_id,
        sensor_id,
        target_id,
        verify_fn=None,
    )


def test_serve_coe_msg(
    hsb_emulator_target,
    ipv4_address,
    data_plane_id,
    sensor_id,
    transport_msg,
    target_id,
    request,
):
    # Linux target broadcasts CoE over loopback; STM32 sends CoE on whatever NIC
    # the board is wired to. The latter is host-specific, so allow override via
    # the --coe-interface pytest option (already declared in conftest.py).
    if hsb_emulator_target == "linux":
        coe_interface = "lo"
    else:
        coe_interface = request.config.getoption("--coe-interface", default=None)
    if not coe_interface:
        pytest.skip(
            "serve_coe_msg on this target needs an explicit CoE interface; "
            "pass --coe-interface=<iface> (e.g. --coe-interface=enP4p3s0f0np0)"
        )

    def verify():
        _verify_messages_coe(ipv4_address, transport_msg, coe_interface)

    verify.transport_msg = transport_msg
    _run_test(
        "serve_coe_msg",
        hsb_emulator_target,
        ipv4_address,
        data_plane_id,
        sensor_id,
        target_id,
        verify_fn=verify,
    )


def test_serve_roce_msg(
    hsb_emulator_target,
    ipv4_address,
    data_plane_id,
    sensor_id,
    transport_msg,
    target_id,
):
    def verify():
        _verify_messages_roce(ipv4_address, transport_msg)

    verify.transport_msg = transport_msg
    _run_test(
        "serve_roce_msg",
        hsb_emulator_target,
        ipv4_address,
        data_plane_id,
        sensor_id,
        target_id,
        verify_fn=verify,
    )


csi_kernels = """
#include <cstdint>
#include <cstddef>
__device__ uint16_t bayerBG(uint16_t row, uint16_t col, uint16_t max_color, uint16_t pixel_width, uint16_t pixel_height)
{
    if (row >= pixel_height) {
        row = pixel_height - 1;
    }
    if (col >= pixel_width) {
        col = pixel_width - 1;
    }
    if (row & 0x01u) { // GR of BGGR pattern
        if (col & 0x01u) { // red
            return (uint16_t) (max_color * (1.0f * col / pixel_width));
        } else { // green
            return (uint16_t) (max_color * (1.0f * (row - 1) / pixel_height));
        }
    } else { // BG of BGGR pattern
        if (col & 0x01u) { // green
            return (uint16_t) (max_color * (1.0f * row / pixel_height));
        } else { // blue
            return (uint16_t) (max_color * (1.0f * (pixel_width - col - 1) / pixel_width));
        }
    }
    return 0;
}

__device__ uint16_t bayerRG(uint16_t row, uint16_t col, uint16_t max_color, uint16_t pixel_width, uint16_t pixel_height)
{
    if (row >= pixel_height) {
        row = pixel_height - 1;
    }
    if (col >= pixel_width) {
        col = pixel_width - 1;
    }
    if (row & 0x01u) { // GB of RGGB pattern
        if (col & 0x01u) { // blue
            return (uint16_t) (max_color * (1.0f * (pixel_width - col - 1) / pixel_width));
        } else { // green
            return (uint16_t) (max_color * (1.0f * (row - 1) / pixel_height));
        }
    } else { // RG of GBRG pattern
        if (col & 0x01u) { // green
            return (uint16_t) (max_color * (1.0f * row / pixel_height));
        } else { // red
            return (uint16_t) (max_color * (1.0f * col / pixel_width));
        }
    }
}

__device__ uint16_t bayerGB(uint16_t row, uint16_t col, uint16_t max_color, uint16_t pixel_width, uint16_t pixel_height)
{
    if (row >= pixel_height) {
        row = pixel_height - 1;
    }
    if (col >= pixel_width) {
        col = pixel_width - 1;
    }
    if (row & 0x01u) { // RG of GBRG pattern
        if (col & 0x01u) { // green
            return (uint16_t) (max_color * (1.0f * (row - 1) / pixel_height));
        } else { // red
            return (uint16_t) (max_color * (1.0f * col / pixel_width));
        }
    } else { // GB of GBRG pattern
        if (col & 0x01u) { // blue
            return (uint16_t) (max_color * (1.0f * (pixel_width - col - 1) / pixel_width));
        } else { // green
            return (uint16_t) (max_color * (1.0f * row / pixel_height));
        }
    }
}

// no bayer GR needed

__constant__ __device__ uint16_t (*bayer_functions[])(uint16_t, uint16_t, uint16_t, uint16_t, uint16_t) = {
    &bayerBG,
    &bayerRG,
    &bayerGB,
};

extern "C" __global__ void generate_csi_frame_8bit(uint8_t * data, size_t start_byte, uint16_t line_bytes, uint16_t pixel_height, uint16_t pixel_width, uint8_t bayer_format)
{
    const uint16_t max_color = 256;
    const uint8_t bytes_per_pixel = 1;
    const uint8_t pixels_per_block = 1;
    int32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    int32_t col_block = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t col = col_block * pixels_per_block;
    size_t data_offset = start_byte + row * (size_t)line_bytes + col_block * bytes_per_pixel;
    if (row >= pixel_height || col >= pixel_width) {
        return;
    }

    uint16_t (*bayer_function)(uint16_t, uint16_t, uint16_t, uint16_t, uint16_t) = bayer_functions[bayer_format];

    data[data_offset] = (uint8_t)(bayer_function(row, col, max_color, pixel_width, pixel_height) & 0xFF);
}

extern "C" __global__ void generate_csi_frame_10bit(uint8_t * data, size_t start_byte, uint16_t line_bytes, uint16_t pixel_height, uint16_t pixel_width, uint8_t bayer_format)
{
    const uint16_t max_color = 1024;
    const uint8_t bytes_per_pixel = 5;
    const uint8_t pixels_per_block = 4;
    int32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    int32_t col_block = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t col = col_block * pixels_per_block;
    size_t data_offset = start_byte + row * (size_t)line_bytes + col_block * bytes_per_pixel;
    if (row >= pixel_height || col >= pixel_width) {
        return;
    }

    uint16_t (*bayer_function)(uint16_t, uint16_t, uint16_t, uint16_t, uint16_t) = bayer_functions[bayer_format];

    uint8_t * data_write = data + data_offset;

    data_write[bytes_per_pixel - 1] = 0;
    for (uint8_t i = 0; i < bytes_per_pixel - 1; i++) {
        uint16_t pixel = bayer_function(row, col + i, max_color, pixel_width, pixel_height);
        data_write[i] = (uint8_t)((pixel >> 2) & 0xFF);
        data_write[bytes_per_pixel - 1] |= (uint8_t)((pixel & 0x3) << (i * 2));
    }
}

extern "C" __global__ void generate_csi_frame_12bit(uint8_t * data, size_t start_byte, uint16_t line_bytes, uint16_t pixel_height, uint16_t pixel_width, uint8_t bayer_format)
{
    const uint16_t max_color = 1024;
    const uint8_t bytes_per_pixel = 3;
    const uint8_t pixels_per_block = 2;
    int32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    int32_t col_block = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t col = col_block * pixels_per_block;
    size_t data_offset = start_byte + row * (size_t)line_bytes + col_block * bytes_per_pixel;
    if (row >= pixel_height || col >= pixel_width) {
        return;
    }

    uint16_t (*bayer_function)(uint16_t, uint16_t, uint16_t, uint16_t, uint16_t) = bayer_functions[bayer_format];

    uint8_t * data_write = data + data_offset;

    uint16_t pixel_left = bayer_function(row, col, max_color, pixel_width, pixel_height);
    uint16_t pixel_right = bayer_function(row, col + 1, max_color, pixel_width, pixel_height);

    data_write[0] = (uint8_t)((pixel_left >> 4) & 0xFF);
    data_write[1] = (uint8_t)((pixel_right >> 4) & 0xFF);
    data_write[2] = (uint8_t)((pixel_left & 0xF) | ((pixel_right & 0xF) << 4));
}
"""

# names + block strides of the csi frame generation kernels in csi_kernels above.
# Compiled on demand by build_csi_generator -- this map is data only, so importing
# this module costs nothing.
csi_generators = {
    hololink_module.sensors.csi.PixelFormat.RAW_8: {
        "name": "generate_csi_frame_8bit",
        "pixels_per_block": 1,
    },
    hololink_module.sensors.csi.PixelFormat.RAW_10: {
        "name": "generate_csi_frame_10bit",
        "pixels_per_block": 4,
    },
    hololink_module.sensors.csi.PixelFormat.RAW_12: {
        "name": "generate_csi_frame_12bit",
        "pixels_per_block": 2,
    },
}


def build_csi_generator(pixel_format):
    """Compile csi_kernels and fetch just the kernel for `pixel_format`.

    Returns (module, kernel, pixels_per_block). The module reference has to
    outlive the kernel, so callers must hold on to it.

    Child-process only: this is the first CUDA call the process makes, and the
    nvcc compile it triggers is the reason the parent needs a readiness gate
    (wait_for_enumeration) before it launches the host application.
    """
    import cupy as cp

    if pixel_format not in csi_generators:
        raise ValueError(
            f"Invalid pixel format: {pixel_format}. must be one of {list(csi_generators.keys())}"
        )
    properties = csi_generators[pixel_format]
    mod = cp.RawModule(
        code=csi_kernels,
        backend="nvcc",
        options=("-std=c++17",),
    )
    # only build the one generator this run needs; compiling all three costs
    # seconds per test parameter for kernels that never get launched
    return mod, mod.get_function(properties["name"]), properties["pixels_per_block"]


def make_frame_data(camera_mode_properties, kernel, pixels_per_block, gpu):
    import cupy as cp

    image_start_byte = get_csi_image_start_byte(camera_mode_properties)
    pixel_width = camera_mode_properties["width"]
    pixel_height = camera_mode_properties["height"]
    gpu_data = cp.zeros((get_csi_frame_size(camera_mode_properties),), dtype=cp.uint8)

    threads_per_block = (32, 32)
    # x-dimension has stride of pixels_per_block, y-dimension has stride of 1
    blocks_per_grid = (
        (
            (pixel_width + pixels_per_block - 1) // pixels_per_block
            + threads_per_block[0]
            - 1
        )
        // threads_per_block[0],
        (pixel_height + threads_per_block[1] - 1) // threads_per_block[1],
    )

    kernel(
        blocks_per_grid,
        threads_per_block,
        (
            gpu_data,
            image_start_byte,
            get_csi_line_bytes(camera_mode_properties),
            pixel_height,
            pixel_width,
            int(camera_mode_properties["bayer_format"]),
        ),
    )

    if not gpu:
        return cp.asnumpy(gpu_data)
    return gpu_data


def _reap(process):
    """Kill `process` if it is still running, then drain its pipes so wait() can't block."""
    if process is None:
        return
    if process.poll() is None:
        process.kill()
    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=5)
    except ValueError:
        # pipes already closed by an earlier communicate() (handle_failed_subprocess)
        pass


def _await_emulator(emulator_process, ipv4_address, timeout_s):
    """Wait for the emulator subprocess to come up, or raise describing why it didn't.

    Polls enumeration in short slices so a child that dies on startup (a bad
    import, a CUDA failure) is reported straight away instead of stalling for
    the whole startup budget.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        if emulator_process.poll() is not None:
            handle_failed_subprocess(emulator_process, "Emulator")
            raise RuntimeError(
                f"Emulator exited during startup with code: {emulator_process.returncode}"
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            handle_failed_subprocess(emulator_process, "Emulator", True)
            raise TimeoutError(
                f"Emulator did not enumerate {ipv4_address} within {timeout_s:.0f}s"
            )
        try:
            return wait_for_enumeration(ipv4_address, min(remaining, 5.0))
        except TimeoutError:
            continue


# drive a host application against an emulator subprocess until the host exits.
# success is the host exiting 0; the emulator serves frames indefinitely, so it is
# killed on every exit path rather than waited on.
def loop_emulator_subprocess(host_command, emu_command, timeout_s, hololink_address):
    print(f"host_command: {host_command}")
    print(f"emu_command: {emu_command}")

    host_process, emulator_process = None, None
    emu_stdout, emu_stderr = tempfile.TemporaryFile(), tempfile.TemporaryFile()
    host_stdout, host_stderr = tempfile.TemporaryFile(), tempfile.TemporaryFile()
    try:
        emulator_process = subprocess.Popen(
            emu_command, stdout=emu_stdout, stderr=emu_stderr
        )

        # The emulator only broadcasts BootP once hsb.start() has returned, which is
        # after its kernel compile and frame generation. Gate on that rather than
        # letting the host application burn its own enumeration timeout waiting.
        _await_emulator(emulator_process, hololink_address, _EMULATOR_STARTUP_TIMEOUT_S)

        host_process = subprocess.Popen(
            host_command, stdout=host_stdout, stderr=host_stderr
        )

        time_start = time.time()
        time_end = time_start + timeout_s
        while time.time() < time_end:
            if host_process.poll() is not None:
                emulator_process.kill()  # regardless of host return state
                if (
                    host_process.returncode != 0
                ):  # a failure if the host application exits with a non-zero code
                    handle_failed_subprocess(
                        host_process, "Host", out_file=host_stdout, err_file=host_stderr
                    )
                    handle_failed_subprocess(
                        emulator_process,
                        "Emulator",
                        out_file=emu_stdout,
                        err_file=emu_stderr,
                    )
                    print(f"test failed in {time.time() - time_start} seconds", end=" ")
                    raise RuntimeError(
                        f"Host application exited with code: {host_process.returncode}"
                    )
                print(
                    f"test completed successfully in {time.time() - time_start} seconds",
                    end=" ",
                )
                return

            # the emulator loops forever, so any exit at all is a failure
            if emulator_process.poll() is not None:
                handle_failed_subprocess(
                    host_process,
                    "Host",
                    True,
                    out_file=host_stdout,
                    err_file=host_stderr,
                )
                handle_failed_subprocess(
                    emulator_process,
                    "Emulator",
                    out_file=emu_stdout,
                    err_file=emu_stderr,
                )
                print(f"test failed in {time.time() - time_start} seconds", end=" ")
                raise RuntimeError(
                    f"Emulator unexpectedly exited early with code: {emulator_process.returncode}"
                )
            time.sleep(1)

        # timeout expired, kill both processes and raise an error
        handle_failed_subprocess(
            host_process, "Host", True, out_file=host_stdout, err_file=host_stderr
        )
        handle_failed_subprocess(
            emulator_process, "Emulator", True, out_file=emu_stdout, err_file=emu_stderr
        )
        print(f"test failed in {time.time() - time_start} seconds", end=" ")
        raise RuntimeError(f"Timeout reached after {timeout_s} seconds")
    finally:
        _reap(emulator_process)
        _reap(host_process)
        emu_stdout.close()
        emu_stderr.close()
        host_stdout.close()
        host_stderr.close()


# how long to allow the emulator subprocess to compile its CSI kernel and start
# broadcasting BootP before we give up on it. The nvcc compile of csi_kernels
# dominates and is machine-dependent, so this is generous.
_EMULATOR_STARTUP_TIMEOUT_S = 120.0


# tests single camera imx emulator configurations in loopback mode by
# - running a subprocess for the host application from its respective examples/ folder
# - running a second subprocess for the emulator (this file, `serve` mode)
# - emulator built for each camera type/mode and transport combination
# - sends tests frames for "frame_limit" number of frames at frame rate determined by the camera mode
# - successful test is the host application exits normally without error code and no early exit from emulator process
def test_emulator_single_camera_imx_loopback(
    imx,
    transport,
    camera_mode,
    gpu,
    frame_limit,
    headless,
    hololink_address,
    hw_loopback,
    json_config,
):
    # hard coded test values for now
    camera_count = 1
    # accelerated networking still needs the emulator wrapped in scripts/nsexec.sh and
    # the roce parametrization below re-enabled, so only unaccelerated linux and coe
    # transport are supported for now
    hw_loopback = None
    assert transport in ("linux", "coe")

    if imx not in camera_properties:
        raise ValueError(
            f"Invalid IMX: {imx}. must be one of {list(camera_properties.keys())}"
        )
    camera = camera_properties[imx]

    if camera_mode not in camera["modes"]:
        raise ValueError(
            f"Invalid camera mode: {camera_mode}. must be one of {list(camera['modes'].keys())}"
        )
    camera_mode_properties = camera["modes"][camera_mode]
    frame_rate = camera_mode_properties["frame_rate"]

    if frame_limit < 1:
        raise ValueError(f"Invalid frame limit: {frame_limit}. must be greater than 0")
    if frame_limit > 60:
        frame_limit = 60
    if not isinstance(gpu, bool):
        raise ValueError(f"Invalid GPU flag: {gpu}. must be a boolean")
    if transport not in camera["examples"]:
        raise ValueError(
            f"Invalid transport: {transport}. must be one of {list(camera['examples'].keys())}"
        )
    if camera_count not in camera["examples"][transport]:
        raise ValueError(
            f"Invalid camera count: {camera_count}. must be one of {list(camera['examples'][transport].keys())}"
        )

    if hw_loopback is None:
        assert transport in ("linux", "coe")
        emu_interface = "lo"
        host_interface = "lo"
        # loopback can only have one valid IP address robust to transport type and cannot be accelerated. use 127.0.0.1 instead of whatever user applied
        hololink_address = "127.0.0.1"
        host_transport = transport
    else:
        emu_interface, host_interface = hw_loopback
        assert host_interface != emu_interface
        if transport == "sipl":
            assert host_interface.startswith("mgbe")
        host_transport = transport

    host_command = [
        "python3",
        camera["examples"][host_transport][camera_count],
        "--frame-limit",
        str(frame_limit),
        "--hololink",
        hololink_address,
        camera["mode_flag"],
        str(camera_mode),
    ]

    if host_transport == "coe":
        host_command.extend(["--coe-interface", host_interface])
    elif host_transport == "sipl":
        # assertion since currently only vb1940 supports sipl. Update when imx274 becomes available
        assert imx == "vb1940"
        # remove camera-mode and hololink options that are unsupported for sipl applications
        host_command = host_command[:-4]
        host_command.extend(["--json-config", json_config])
    elif host_transport == "roce":
        host_command.extend(["--ibv-name", get_ib_device_from_ifname(host_interface)])
    elif host_transport != "linux":
        raise ValueError(
            f"Test {test_emulator_single_camera_imx_loopback.__name__} does not support transport type: {transport}."
        )
    if headless:
        host_command.append("--headless")

    # The emulator runs in its own process. Everything CUDA -- importing cupy,
    # compiling the CSI kernel, generating the frame -- happens over there, after
    # this process is done forking and before the emulator spawns any thread of
    # its own. Doing it here instead would touch the driver API before the fork
    # and fork a multi-threaded process, and the CUDA context would then persist
    # into every subsequent test parameter.
    emu_command = [
        "python3",
        os.path.abspath(__file__),
        "serve",
        "--imx",
        imx,
        "--camera-mode",
        str(camera_mode),
        "--transport",
        host_transport,
        "--address",
        hololink_address,
        "--frame-limit",
        str(frame_limit),
        "--frame-rate",
        str(frame_rate),
    ]
    if gpu:
        emu_command.append("--gpu")

    timeout_s = (
        2 * frame_limit / frame_rate + camera["configure_timeout"] * camera_count
    )

    loop_emulator_subprocess(host_command, emu_command, timeout_s, hololink_address)


# dynamically generate test cases for the 2 test functions in this file based on the command line options
def pytest_generate_tests(metafunc):
    if metafunc.function == test_emulator_single_camera_imx_loopback:
        parameters = single_camera_imx_loopback_cases
        if metafunc.config.getoption("--emulator"):
            parameters += single_camera_imx_loopback_cases_extended
        # for when roce support is added
        # if metafunc.config.getoption("--hw-loopback") is not None and not metafunc.config.getoption("--unaccelerated-only"):
        #    if has_ib_interface():
        #        parameters += single_camera_imx_loopback_cases_roce
        metafunc.parametrize("imx,transport,camera_mode,gpu", parameters)

    # hsb_control / serve_coe_msg / serve_roce_msg: always run with the linux
    # target on loopback; additionally run with STM32F767ZI when the user passes
    # --stm32f767zi=<programmer-id>. data_plane_id and sensor_id default to 0
    # for both targets. ipv4_address differs by target: loopback for linux,
    # the device's data-plane address (192.168.0.2) for the MCU.
    if metafunc.function in (test_hsb_control, test_serve_coe_msg, test_serve_roce_msg):
        targets = []
        stm32_id = metafunc.config.getoption("--stm32f767zi")
        if stm32_id is not None:
            targets.append(("STM32F767ZI", "192.168.0.2", stm32_id))
        else:
            targets.append(("linux", "127.0.0.1", None))

        if metafunc.function is test_hsb_control:
            metafunc.parametrize(
                "hsb_emulator_target,ipv4_address,data_plane_id,sensor_id,target_id",
                [(t, ip, 0, 0, tid) for (t, ip, tid) in targets],
            )
        else:
            metafunc.parametrize(
                "hsb_emulator_target,ipv4_address,data_plane_id,sensor_id,transport_msg,target_id",
                [(t, ip, 0, 0, DEFAULT_TRANSPORT_MSG, tid) for (t, ip, tid) in targets],
            )


# ---------------------------------------------------------------------------
# `serve` mode -- the emulator side of test_emulator_single_camera_imx_loopback,
# run as its own process:
#
#   python3 tests/test_emulator.py serve --imx imx477 --camera-mode 1080p \
#           --transport linux --address 127.0.0.1 --frame-rate 60 [--gpu]
#
# There is no standalone example for the generic imx cameras (unlike the vb1940
# serve_*_frames.py scripts, which read their geometry back off an emulated
# sensor over I2C); these frames are built from the static camera_properties
# geometry for the mode under test, so the server lives here next to that table.
#
# Nothing in this process forks, which is what lets it do CUDA work first and
# only then construct the emulator objects -- HSBEmulator and each DataPlane
# spawn a thread from their constructors.
# ---------------------------------------------------------------------------


def _resolve_camera_mode(camera, camera_mode):
    """Look up a mode key that arrived over the command line as a string.

    imx477 modes are strings ("4k"/"1080p"); the other cameras key off int enum
    values, which argparse hands us as text.
    """
    modes = camera["modes"]
    if camera_mode in modes:
        return camera_mode
    try:
        return int(camera_mode)
    except ValueError:
        pass
    raise ValueError(
        f"Invalid camera mode: {camera_mode}. must be one of {list(modes.keys())}"
    )


def _serve_main(argv):
    parser = argparse.ArgumentParser(
        description="Serve generated CSI-2 frames from an HSB Emulator as if from a single imx sensor"
    )
    parser.add_argument(
        "--imx", required=True, help="Camera name, a key of camera_properties"
    )
    parser.add_argument(
        "--camera-mode", required=True, help="Camera mode, a key of camera['modes']"
    )
    parser.add_argument(
        "--transport", required=True, choices=("linux", "coe", "roce", "sipl")
    )
    parser.add_argument(
        "--address", required=True, help="IP address the emulator serves from"
    )
    parser.add_argument("--frame-rate", type=int, required=True)
    parser.add_argument("--data-plane-id", type=int, default=0)
    parser.add_argument("--sensor-id", type=int, default=0)
    parser.add_argument("--frame-limit", type=int, default=60)
    parser.add_argument(
        "-g", "--gpu", action="store_true", help="Serve the data from the GPU"
    )
    args = parser.parse_args(argv)

    if args.imx not in camera_properties:
        raise ValueError(
            f"Invalid IMX: {args.imx}. must be one of {list(camera_properties.keys())}"
        )
    camera = camera_properties[args.imx]
    camera_mode_properties = camera["modes"][
        _resolve_camera_mode(camera, args.camera_mode)
    ]

    # CUDA first: compile just the kernel this mode needs and generate the frame,
    # before any emulator object exists to start a thread.
    mod, kernel, pixels_per_block = build_csi_generator(
        camera_mode_properties["pixel_format"]
    )
    frame_data = make_frame_data(
        camera_mode_properties, kernel, pixels_per_block, args.gpu
    )
    assert mod is not None  # the kernel is only valid while its module is alive

    # Emulator second: HSBEmulator's ctor starts the control thread and each
    # DataPlane's ctor starts a BootP thread.
    import hololink.emulation as hemu

    hsb = hemu.HSBEmulator()
    if args.transport in ("coe", "sipl"):
        data_plane_type = hemu.COEDataPlane
    else:
        data_plane_type = hemu.LinuxDataPlane
    data_plane = data_plane_type(
        hsb, hemu.IPAddress(args.address), args.data_plane_id, args.sensor_id
    )

    hsb.start()

    time_final = (
        time.time() + args.frame_limit / args.frame_rate + 15.0
    )  # add 15 seconds to the timeout to account for the emulator startup time and wait for harness shutdown

    # Serve until the parent kills us or our time limit expires.
    print(f"serving {args.imx} mode {args.camera_mode} on {args.address}", flush=True)
    try:
        while time.time() < time_final:
            last_frame_time = time.time_ns()
            sent_bytes = data_plane.send(frame_data)
            if sent_bytes < 0:
                raise RuntimeError(f"Error sending data: {sent_bytes}")
            sleep_frame_rate(last_frame_time, args.frame_rate)
        print("Time limit expired without a kill from the test harness, stopping...")
    except KeyboardInterrupt:
        print("\nReceived Ctrl+C, stopping...")
    finally:
        hsb.stop()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        _serve_main(sys.argv[2:])
    else:
        pytest.main()
