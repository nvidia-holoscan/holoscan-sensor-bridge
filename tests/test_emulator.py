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

import inspect
import json
import os
import signal
import subprocess
import time

import cupy as cp
import pytest

import hololink as hololink_module
import hololink.emulation as hemu

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def has_sipl_interface():
    try:
        coe_channels = os.listdir("/sys/class/capture-coe-channel/")
        return len(coe_channels) > 0
    except FileNotFoundError:
        return False


def has_ib_interface():
    try:
        ib_channels = hololink_module.infiniband_devices()
        return len(ib_channels) > 0
    except FileNotFoundError:
        return False


def get_ib_device_ifname(ib_device):
    try:
        ifname = os.listdir(f"/sys/class/infiniband/{ib_device}/device/net/")[0]
        return ifname
    except FileNotFoundError:
        return None


def get_ib_device_from_ifname(ifname):
    ib_devices = hololink_module.infiniband_devices()
    for ib_device in ib_devices:
        ib_device_ifname = get_ib_device_ifname(ib_device)
        if ib_device_ifname == ifname:
            return ib_device
    return None


def byte_align_8(value):
    return (value + 7) // 8 * 8


def line_bytes_from_pixel_format(pixel_width, pixel_format):
    if pixel_format == hololink_module.sensors.csi.PixelFormat.RAW_8:
        return pixel_width  # 8 bits per pixel, so no padding needed
    elif pixel_format == hololink_module.sensors.csi.PixelFormat.RAW_10:
        return (pixel_width + 3) // 4 * 5  # 10 bits per pixel, so 5 bytes per 4 pixels
    elif pixel_format == hololink_module.sensors.csi.PixelFormat.RAW_12:
        return (pixel_width + 1) // 2 * 3  # 12 bits per pixel, so 3 bytes per 2 pixels
    else:
        raise ValueError(
            f"Invalid pixel format: {pixel_format}. must be one of {list(hololink_module.sensors.csi.PixelFormat)}"
        )


# this maps the transport and camera count settings to a specific vb1940 example application
# which is used by the tests to both test the vb1940 I2CPeripheral/emulated sensor and the
# provided applications
vb1940_emulator_examples = {
    "linux": {
        1: "src/hololink/emulation/examples/serve_linux_single_vb1940_frames.py",
        2: "src/hololink/emulation/examples/serve_linux_stereo_vb1940_frames.py",
    },
    "coe": {
        1: "src/hololink/emulation/examples/serve_coe_single_vb1940_frames.py",
        2: "src/hololink/emulation/examples/serve_coe_stereo_vb1940_frames.py",
    },
}

# python dict of all the shared camera properties used to configure tests and generate
# test data for the HSB Emulator for different camera scenarios
camera_properties = {
    "vb1940": {
        "configure_timeout": 35,
        "examples": {
            "linux": {
                1: "examples/linux_vb1940_player.py",
                2: "examples/linux_single_network_stereo_vb1940_player.py",
            },
            "coe": {
                1: "examples/linux_coe_single_vb1940_player.py",
                2: "examples/linux_coe_stereo_vb1940_player.py",
            },
            "sipl": {
                1: "examples/sipl_player.py",
                2: "examples/sipl_player.py",
            },
            "roce": {
                1: "examples/vb1940_player.py",
                2: "examples/single_network_stereo_vb1940_player.py",
            },
        },
        "mode_flag": "--camera-mode",
        "modes": {
            hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS.value: {
                "start_lines": 1,
                "end_lines": 2,
                "width": 2560,
                "height": 1984,
                "frame_rate": 30,
                "pixel_format": hololink_module.sensors.csi.PixelFormat.RAW_10,
            },
            hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_1920X1080_30FPS.value: {
                "start_lines": 1,
                "end_lines": 2,
                "width": 1920,
                "height": 1080,
                "frame_rate": 30,
                "pixel_format": hololink_module.sensors.csi.PixelFormat.RAW_10,
            },
            hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS_8BIT.value: {
                "start_lines": 1,
                "end_lines": 2,
                "width": 2560,
                "height": 1984,
                "frame_rate": 30,
                "pixel_format": hololink_module.sensors.csi.PixelFormat.RAW_8,
            },
        },
    },
    "imx477": {
        "configure_timeout": 10,
        "examples": {
            "linux": {
                1: "examples/linux_imx477_player.py",
                2: "examples/linux_imx477_stereo_player.py",
            },
            "roce": {
                1: "examples/imx477_player.py",
                2: "examples/single_network_stereo_imx477_player.py",
            },
        },
        "mode_flag": "--resolution",
        "modes": {
            "4k": {
                "width": 3840,
                "height": 2160,
                "frame_rate": 60,
                "pixel_format": hololink_module.sensors.csi.PixelFormat.RAW_8,
                "bayer_format": hololink_module.sensors.csi.BayerFormat.RGGB,
            },
            "1080p": {
                "width": 1920,
                "height": 1080,
                "frame_rate": 60,
                "pixel_format": hololink_module.sensors.csi.PixelFormat.RAW_8,
                "bayer_format": hololink_module.sensors.csi.BayerFormat.RGGB,
            },
        },
    },
    "imx715": {
        "configure_timeout": 15,
        "examples": {
            "linux": {
                1: "examples/linux_imx715_player.py",
            },
            "roce": {
                1: "examples/imx715_player.py",
            },
        },
        "mode_flag": "--camera-mode",
        "modes": {
            hololink_module.sensors.imx715.IMX715_MODE_3840X2160_30FPS_12BPP: {
                "start_lines": 37,
                "width": 3840,
                "height": 2160,
                "frame_rate": 30,
                "pixel_format": hololink_module.sensors.csi.PixelFormat.RAW_12,
                "bayer_format": hololink_module.sensors.csi.BayerFormat.GBRG,
            },
            hololink_module.sensors.imx715.IMX715_MODE_3840X2160_60FPS_10BPP: {
                "start_lines": 41,
                "width": 3840,
                "height": 2160,
                "frame_rate": 60,
                "pixel_format": hololink_module.sensors.csi.PixelFormat.RAW_10,
                "bayer_format": hololink_module.sensors.csi.BayerFormat.GBRG,
            },
            hololink_module.sensors.imx715.IMX715_MODE_1920X1080_60FPS_12BPP: {
                "start_lines": 21,
                "end_lines": 14,
                # this has an unusual line_bytes setting. Hard-coding until a more robust method is needed
                "line_bytes": byte_align_8(
                    line_bytes_from_pixel_format(
                        1920, hololink_module.sensors.csi.PixelFormat.RAW_12
                    )
                    + 38
                ),
                "width": 1920,
                "height": 1080,
                "frame_rate": 60,
                "pixel_format": hololink_module.sensors.csi.PixelFormat.RAW_12,
                "bayer_format": hololink_module.sensors.csi.BayerFormat.GBRG,
            },
        },
    },
    "imx274": {
        "configure_timeout": 10,
        "examples": {
            "linux": {
                1: "examples/linux_imx274_player.py",
            },
            "roce": {
                1: "examples/imx274_player.py",
            },
        },
        "mode_flag": "--camera-mode",
        "modes": {
            hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_3840X2160_60FPS.value: {
                "start_bytes": 175,
                "start_lines": 8,
                "width": 3840,
                "height": 2160,
                "frame_rate": 60,
                "pixel_format": hololink_module.sensors.csi.PixelFormat.RAW_10,
                "bayer_format": hololink_module.sensors.csi.BayerFormat.RGGB,
            },
            hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS.value: {
                "start_bytes": 175,
                "start_lines": 8,
                "width": 1920,
                "height": 1080,
                "frame_rate": 60,
                "pixel_format": hololink_module.sensors.csi.PixelFormat.RAW_10,
                "bayer_format": hololink_module.sensors.csi.BayerFormat.RGGB,
            },
            hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_3840X2160_60FPS_12BITS.value: {
                "start_bytes": 175,
                "start_lines": 16,
                "width": 3840,
                "height": 2160,
                "frame_rate": 60,
                "pixel_format": hololink_module.sensors.csi.PixelFormat.RAW_12,
                "bayer_format": hololink_module.sensors.csi.BayerFormat.RGGB,
            },
        },
    },
}


def initialize_camera_properties():
    # fill in some default values for the camera properties

    for camera in camera_properties:
        modes = camera_properties[camera]["modes"]
        for mode, mode_properties in modes.items():
            # none of the cameras have line_start or line_end bytes,
            # so we can just set the line_bytes based on the pixel format and width
            # only set the values if they are not already set
            if "line_bytes" not in mode_properties:
                mode_properties["line_bytes"] = byte_align_8(
                    line_bytes_from_pixel_format(
                        mode_properties["width"], mode_properties["pixel_format"]
                    )
                )
            if "image_start_byte" not in mode_properties:
                mode_properties["image_start_byte"] = byte_align_8(
                    mode_properties.get("start_bytes", 0)
                ) + mode_properties["line_bytes"] * mode_properties.get(
                    "start_lines", 0
                )
            if "csi_frame_size" not in mode_properties:
                trailing_bytes = (
                    mode_properties.get("end_bytes", 0)
                    + mode_properties.get("end_lines", 0)
                    * mode_properties["line_bytes"]
                )
                mode_properties["csi_frame_size"] = (
                    mode_properties["image_start_byte"]
                    + mode_properties["line_bytes"] * mode_properties["height"]
                    + trailing_bytes
                )


initialize_camera_properties()


def get_csi_image_size(camera_mode):
    return camera_mode["line_bytes"] * camera_mode["height"]


def get_csi_image_start_byte(camera_mode):
    return camera_mode["image_start_byte"]


def get_csi_frame_size(camera_mode):
    return camera_mode["csi_frame_size"]


def get_csi_line_bytes(camera_mode):
    return camera_mode["line_bytes"]


def get_namespace_pid(namespace, command):
    try:
        pid_cmd = [
            os.path.join(script_dir, "scripts", "nspid.sh"),
            namespace,
            command[0],  # only take the running binary name as flags throw off the grep
        ]
        nspid_process = subprocess.run(
            pid_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return int(nspid_process.stdout.decode("utf-8").strip())
    except Exception as e:
        print(f"Error getting namespace PID: {e}")
        return None


def handle_failed_subprocess(subprocess, name, kill=None):
    if kill:
        kill()
    stdout, stderr = subprocess.communicate()
    print(f"{name} subprocess stdout: \n{stdout.decode('utf-8', errors='replace')}")
    print(f"{name} subprocess stderr: \n{stderr.decode('utf-8', errors='replace')}")


def emulator_loopback(host_command, emulator_command, timeout_s, isolated_interface):

    time_start = time.time()
    time_end = time_start + timeout_s
    # start the emulator command and define the appropriate kill function
    emulator_process = None
    if isolated_interface != "lo":
        # if the emulator is running in an isolated namespace, the process that needs to be killed is within the namespace
        # killing the process from the scripts/nsexec.sh basically makes the emulator process an orphan that root/sudo needs to clean up
        emulator_process = subprocess.Popen(
            [os.path.join(script_dir, "scripts", "nsexec.sh"), isolated_interface]
            + emulator_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        nspid = get_namespace_pid(isolated_interface, emulator_command)
        while nspid is None and time.time() < time_end:
            time.sleep(1)
            nspid = get_namespace_pid(isolated_interface, emulator_command)
        if nspid is None:
            raise RuntimeError(
                f"Timeout reached after {timeout_s} seconds while waiting for emulator process to start. cmd: {emulator_command}. May require manual cleanup of namespace: {isolated_interface}"
            )

        # why is this here? because ci/lint.sh says we cannot assign a lambda to a variable
        def nsexec_kill():
            os.kill(nspid, signal.SIGKILL)

        emulator_kill = nsexec_kill
    else:
        emulator_process = subprocess.Popen(
            emulator_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        emulator_kill = emulator_process.kill

    # start the host command
    host_process = subprocess.Popen(
        host_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    host_kill = host_process.kill

    # restart timeout for the main loop
    time_end = time.time() + timeout_s
    while time.time() < time_end:
        # success is host process exiting normally. If it exits with error code, kill the emulator and raise an error for a failed test
        if (
            host_process.poll() is not None
        ):  # only success is if host process exits normally
            emulator_kill()  # regardless of host return state, kill the emulator
            if host_process.returncode != 0:
                handle_failed_subprocess(host_process, "Host", host_kill)
                handle_failed_subprocess(emulator_process, "Emulator")
                print(f"test failed in {time.time() - time_start} seconds", end=" ")
                raise RuntimeError(
                    f"Host subprocess exited early with code: {host_process.returncode}"
                )
            print(
                f"test completed successfully in {time.time() - time_start} seconds",
                end=" ",
            )
            return

        # if emulator process exits early, kill the host process and raise an error
        if emulator_process.poll() is not None:
            handle_failed_subprocess(host_process, "Host", host_kill)
            handle_failed_subprocess(emulator_process, "Emulator")
            print(f"test failed in {time.time() - time_start} seconds", end=" ")
            raise RuntimeError(
                f"Emulator unexpectedly exited early with code: {emulator_process.returncode}"
            )
        time.sleep(1)
    # timeout reached, kill both processes and raise an error
    handle_failed_subprocess(host_process, "Host", host_kill)
    handle_failed_subprocess(emulator_process, "Emulator", emulator_kill)
    print(f"test failed in {time.time() - time_start} seconds", end=" ")
    raise RuntimeError(f"Host subprocess timed out after {timeout_s} seconds")


# test cases covering basic emulator functionality:
# linux and coe transport with and without GPU device inputs and a simulated I2CPeripheral sensor (vb1940) with stateful interactions
# and accessing sensor state to build csi-frames dynamically
vb1940_loopback_cases = [
    (
        "linux",
        1,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS.value,
        True,
    ),
    (
        "coe",
        1,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS.value,
        True,
    ),
    (
        "linux",
        1,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS.value,
        False,
    ),
    (
        "linux",
        1,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_1920X1080_30FPS.value,
        False,
    ),
    (
        "linux",
        1,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS_8BIT.value,
        False,
    ),
]

# covers a wider parameter space but no new functional checks
vb1940_loopback_cases_extended = [
    (
        "linux",
        2,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS.value,
        True,
    ),
    (
        "coe",
        2,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS.value,
        True,
    ),
    (
        "coe",
        1,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS.value,
        False,
    ),
    (
        "coe",
        1,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_1920X1080_30FPS.value,
        False,
    ),
    (
        "coe",
        1,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS_8BIT.value,
        False,
    ),
    (
        "coe",
        2,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS.value,
        False,
    ),
    (
        "coe",
        2,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_1920X1080_30FPS.value,
        False,
    ),
    (
        "coe",
        2,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS_8BIT.value,
        False,
    ),
    (
        "linux",
        2,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS.value,
        False,
    ),
    (
        "linux",
        2,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_1920X1080_30FPS.value,
        False,
    ),
    (
        "linux",
        2,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS_8BIT.value,
        False,
    ),
]

# hardware accelerated tests for sipl transport. requires the "--hw-loopback" option/fixture
# Note: because of the nature of SIPL capture configurations, the actual camera count for the tests
# will be determined by the number of cameras in the JSON configuration file.
vb1940_loopback_cases_sipl = [
    (
        "sipl",
        None,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS.value,
        False,
    ),
]

# hardware accelerated tests for roce transport. requires the "--hw-loopback" option/fixture
vb1940_loopback_cases_roce = [
    (
        "roce",
        1,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS.value,
        False,
    ),
    (
        "roce",
        2,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS.value,
        False,
    ),
]


# tests vb1940 emulator configurations in loopback mode by
# - running a subprocess for both the host and emulator applications from their respective examples/ folders
# - sends tests frames for "frame_limit" number of frames at frame rate determined by the camera mode
# - successful test is the host application exits normally without error code and no early exit from emulator process
def test_emulator_vb1940_loopback(
    transport,
    camera_count,
    camera_mode,
    gpu,
    frame_limit,
    headless,
    hololink_address,
    hw_loopback,
    json_config,
):
    camera = camera_properties["vb1940"]
    if camera_count < 1 or camera_count > 2:
        raise ValueError(f"Invalid camera count: {camera_count}")
    if camera_mode < 0 or camera_mode > 2:
        raise ValueError(f"Invalid camera mode: {camera_mode}. must be 0, 1, or 2")
    frame_rate = 30  # for all vb1940 modes
    if frame_limit < 1:
        raise ValueError(f"Invalid frame limit: {frame_limit}. must be greater than 0")
    if frame_limit > 60:
        frame_limit = 60
    if not isinstance(gpu, bool):
        raise ValueError(f"Invalid GPU flag: {gpu}. must be a boolean")

    if hw_loopback is None:
        assert transport in ("linux", "coe")
        emu_interface = "lo"
        host_interface = "lo"
        # loopback can only have one valid IP address robust to transport type and cannot be accelerated. use 127.0.0.1 instead of whatever user applied
        hololink_address = "127.0.0.1"
        host_transport = transport
        emu_transport = transport
    else:
        emu_interface, host_interface = hw_loopback
        assert host_interface != emu_interface
        if transport == "sipl":
            emu_transport = "coe"
            assert host_interface.startswith("mgbe")
        elif transport == "roce":
            emu_transport = "linux"
        else:
            emu_transport = transport
        host_transport = transport

    host_command = [
        "python3",
        camera["examples"][host_transport][camera_count],
        "--frame-limit",
        str(frame_limit),
        "--camera-mode",
        str(camera_mode),
        "--hololink",
        hololink_address,
    ]

    if host_transport == "coe":
        host_command.extend(["--coe-interface", host_interface])
    elif host_transport == "roce":
        host_command.extend(["--ibv-name", get_ib_device_from_ifname(host_interface)])
    elif host_transport == "sipl":
        # remove cmaera-mode and hololink options that are unsupported
        host_command = host_command[:-4]
        if camera_count == 1:
            host_command.extend(["--json-config", json_config])
        elif camera_count == 2:
            host_command.extend(["--json-config", json_config])
        else:
            raise ValueError(
                f"Invalid camera count: {camera_count}. must be 1 or 2 for sipl transport"
            )

    if headless:
        host_command.append("--headless")
    emu_command = [
        "python3",
        vb1940_emulator_examples[emu_transport][camera_count],
        "--frame-rate",
        str(frame_rate),
    ]
    if gpu:
        emu_command.append("--gpu")
    emu_command.append(hololink_address)

    timeout_s = (
        2 * frame_limit / frame_rate + camera["configure_timeout"] * camera_count
    )
    emulator_loopback(host_command, emu_command, timeout_s, emu_interface)


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

# data structure for holding cuda kernels for csi frame generation
csi_generators = {
    "module": None,  # reserved for module loading
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


# initialize the csi_generator kernels
def initialize_csi_generators():
    mod = cp.RawModule(
        code=csi_kernels,
        backend="nvcc",
        options=("-std=c++17",),
    )

    for key, properties in csi_generators.items():
        if properties is not None:
            properties["kernel"] = mod.get_function(properties["name"])

    csi_generators["module"] = mod  # save a reference to the module


# actually run the initialization
initialize_csi_generators()


def make_frame_data(camera_mode_properties, gpu):
    image_start_byte = get_csi_image_start_byte(camera_mode_properties)
    pixel_format = camera_mode_properties["pixel_format"]
    pixel_width = camera_mode_properties["width"]
    pixel_height = camera_mode_properties["height"]
    generator = csi_generators[pixel_format]
    pixels_per_block = generator["pixels_per_block"]
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

    generator["kernel"](
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


def sleep_frame_rate(last_frame_time, frame_rate_per_second):
    """Sleep the thread to try to match target frame rate"""
    delta_s = (
        1 / frame_rate_per_second - (time.time_ns() - last_frame_time) / 1000000000
    )
    if delta_s > 0:
        time.sleep(delta_s)


# loop a single dataplane by sending frames at the specified frame rate until the timeout is reached or the host process exits with a non-zero code
def loop_single_dataplane(data_plane, frame_data, frame_rate, subproc, timeout_s):
    time_start = time.time()
    time_end = time_start + timeout_s
    while time.time() < time_end:
        last_frame_time = time.time_ns()
        if subproc.poll() is not None:
            if (
                subproc.returncode != 0
            ):  # a failure if the host application exits with a non-zero code
                handle_failed_subprocess(subproc, "Host", False)
                print(f"test failed in {time.time() - time_start} seconds", end=" ")
                raise RuntimeError(
                    f"Host application exited with code: {subproc.returncode}"
                )
            print(
                f"test completed successfully in {time.time() - time_start} seconds",
                end=" ",
            )
            return
        sent_bytes = data_plane.send(frame_data)
        if sent_bytes < 0:
            print(f"test failed in {time.time() - time_start} seconds", end=" ")
            raise RuntimeError(f"Error sending data: {sent_bytes}")

        sleep_frame_rate(last_frame_time, frame_rate)

    # timeout expired, kill the host process and raise an error
    handle_failed_subprocess(subproc, "Host", True)
    print(f"test failed in {time.time() - time_start} seconds", end=" ")
    raise RuntimeError(f"Timeout reached after {timeout_s} seconds")


# test cases for single camera imx loopback mode
# - running a subprocess for the host application from its respective examples/ folder
# - sends tests frames for "frame_limit" number of frames at frame rate determined by the camera mode
# - successful test is the host application exits normally without error code and no early exit from emulator process
single_camera_imx_loopback_cases = [
    ("imx477", "linux", "1080p", True),
    ("imx477", "linux", "1080p", False),
    ("imx477", "linux", "4k", False),
]

single_camera_imx_loopback_cases_extended = [
    (
        "imx715",
        "linux",
        hololink_module.sensors.imx715.IMX715_MODE_3840X2160_30FPS_12BPP,
        False,
    ),
    (
        "imx715",
        "linux",
        hololink_module.sensors.imx715.IMX715_MODE_3840X2160_60FPS_10BPP,
        False,
    ),
    (
        "imx715",
        "linux",
        hololink_module.sensors.imx715.IMX715_MODE_1920X1080_60FPS_12BPP,
        False,
    ),
    (
        "imx274",
        "linux",
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_3840X2160_60FPS.value,
        False,
    ),
    (
        "imx274",
        "linux",
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS.value,
        False,
    ),
    (
        "imx274",
        "linux",
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_3840X2160_60FPS_12BITS.value,
        False,
    ),
]

# unused for now. see comments in test function
single_camera_imx_loopback_cases_roce = [
    ("imx477", "roce", "1080p", False),
    (
        "imx715",
        "roce",
        hololink_module.sensors.imx715.IMX715_MODE_3840X2160_30FPS_12BPP,
        False,
    ),
    (
        "imx274",
        "roce",
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_3840X2160_60FPS.value,
        False,
    ),
]


# tests single camera imx emulator configurations in loopback mode by
# - running a subprocess for the host application from its respective examples/ folder
# - emulator code runs within the function (no separate binary examples yet)
# - emulator built for each camera type/mode and transport combination
# - sends tests frames for "frame_limit" number of frames at frame rate determined by the camera mode
# - successful test is the host application exits normally without error code and no early errors from emulator code
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
    # accelerated networking will require a binary or script that can be separately launched. Until either an example is built for the
    # generic imx cameras or the HSB Emulator portion of this function is built out into a separate callable script, only unaccelerated
    # linux and coe transport are supported for now
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

    # generate the test data
    frame_data = make_frame_data(camera_mode_properties, gpu)

    # build emulator
    hsb = hemu.HSBEmulator()
    data_plane = None
    if host_transport == "coe":
        host_command.extend(["--coe-interface", host_interface])
        data_plane = hemu.COEDataPlane(hsb, hemu.IPAddress(hololink_address), 0, 0)
    elif host_transport == "sipl":
        # assertion since currently only vb1940 supports sipl. Update when imx274 becomes available
        assert imx == "vb1940"
        # remove camera-mode and hololink options that are unsupported for sipl applications
        host_command = host_command[:-4]
        if camera_count == 1:
            host_command.extend(["--json-config", json_config])
        else:
            host_command.extend(["--json-config", json_config])
        data_plane = hemu.COEDataPlane(hsb, hemu.IPAddress(hololink_address), 0, 0)
    elif host_transport == "linux":
        data_plane = hemu.LinuxDataPlane(hsb, hemu.IPAddress(hololink_address), 0, 0)
    elif host_transport == "roce":
        data_plane = hemu.LinuxDataPlane(hsb, hemu.IPAddress(hololink_address), 0, 0)
        host_command.extend(["--ibv-name", get_ib_device_from_ifname(host_interface)])
    else:
        raise ValueError(
            f": Test {inspect.currentframe().f_code.co_name} does not support transport type: {transport}."
        )
    if headless:
        host_command.append("--headless")

    hsb.start()

    # start the host application subprocess
    host_process = subprocess.Popen(
        host_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    timeout_s = (
        2 * frame_limit / frame_rate + camera["configure_timeout"] * camera_count
    )
    loop_single_dataplane(data_plane, frame_data, frame_rate, host_process, timeout_s)
    hsb.stop()


def get_camera_count_from_json_config(json_config):
    with open(json_config, "r") as f:
        config = json.load(f)
    return len(config["cameraConfigs"])


# dynamically generate test cases for the 2 test functions in this file based on the command line options
def pytest_generate_tests(metafunc):
    if metafunc.function == test_emulator_vb1940_loopback:
        parameters = vb1940_loopback_cases
        if metafunc.config.getoption("--emulator"):
            parameters += vb1940_loopback_cases_extended
        if metafunc.config.getoption(
            "--hw-loopback"
        ) is not None and not metafunc.config.getoption("--unaccelerated-only"):
            if (
                has_sipl_interface()
                and metafunc.config.getoption("--json-config") is not None
            ):
                camera_count = get_camera_count_from_json_config(
                    metafunc.config.getoption("--json-config")
                )
                for test_case in vb1940_loopback_cases_sipl:
                    parameters.append(
                        (test_case[0], camera_count, test_case[2], test_case[3])
                    )
            if has_ib_interface():
                parameters += vb1940_loopback_cases_roce
        metafunc.parametrize("transport,camera_count,camera_mode,gpu", parameters)
    elif metafunc.function == test_emulator_single_camera_imx_loopback:
        parameters = single_camera_imx_loopback_cases
        if metafunc.config.getoption("--emulator"):
            parameters += single_camera_imx_loopback_cases_extended
        # for when roce support is added
        # if metafunc.config.getoption("--hw-loopback") is not None and not metafunc.config.getoption("--unaccelerated-only"):
        #    if has_ib_interface():
        #        parameters += single_camera_imx_loopback_cases_roce
        metafunc.parametrize("imx,transport,camera_mode,gpu", parameters)


if __name__ == "__main__":
    pytest.main()
