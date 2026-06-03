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
import os
import subprocess
import time

import cupy as cp
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
import hololink.emulation as hemu

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
                handle_failed_subprocess(subproc, "Host")
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


if __name__ == "__main__":
    pytest.main()
