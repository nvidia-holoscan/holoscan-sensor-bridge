"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
See README.md for detailed information.
"""

import os
import time
from dataclasses import dataclass

import cupy as cp
import numpy as np

kernel_map = {
    "module": None,
    "bayer8p_to_T_X2Rc10Rb10Ra10_kernel": None,
    "bayer8p_to_10p_kernel": None,
    "bayer8p_to_T_R12_PK_ISP_kernel": None,
    "generate_bayerGB8p_kernel": None,
}


# load the kernels from the file emulator_kernels.cu to be shared by both the C++ and Python code
def load_kernels(kernel_map):
    kernel_file = os.path.join(os.path.dirname(__file__), "emulator_kernels.cu")
    with open(kernel_file, "r") as f:
        kernel_code = f.read()

    mod = cp.RawModule(
        code=kernel_code,
        backend="nvcc",
        options=(
            "-I" + os.path.dirname(__file__),
            "-std=c++17",
        ),
    )
    kernel_map["module"] = mod
    for k, v in kernel_map.items():
        if v is None:
            kernel_map[k] = mod.get_function(k)


load_kernels(kernel_map)


# convert 8-bit bayer pattern to 10-bit bayer pattern with T_X2Rc10Rb10Ra10 encoding (3 pixels per 4 bytes)
def bayer8p_to_T_X2Rc10Rb10Ra10(src):
    if src.ndim != 2:
        raise ValueError("src must be a 2D array")
    pixel_height, pixel_width = src.shape
    line_bytes = (pixel_width + 2) // 3 * 4
    line_bytes = ((line_bytes + 63) >> 6) << 6
    dest = cp.zeros((pixel_height, line_bytes), dtype=cp.uint8)
    threads_per_block = (32, 32)
    blocks_per_grid = (
        (pixel_width + 3 * threads_per_block[0] - 1) // (3 * threads_per_block[0]),
        (pixel_height + threads_per_block[1] - 1) // threads_per_block[1],
    )
    kernel_map["bayer8p_to_T_X2Rc10Rb10Ra10_kernel"](
        blocks_per_grid,
        threads_per_block,
        (dest, line_bytes, src, pixel_height, pixel_width),
    )
    return dest


# convert 8-bit bayer pattern to 10-bit packed bayer pattern (4 pixels per 5 bytes)
def bayer8p_to_10p(src):
    if src.ndim != 2:
        raise ValueError("src must be a 2D array")
    pixel_height, pixel_width = src.shape
    line_bytes = (pixel_width + 3) // 4 * 5
    line_bytes = ((line_bytes + 7) >> 3) << 3
    dest = cp.zeros((pixel_height, line_bytes), dtype=cp.uint8)
    threads_per_block = (32, 32)
    blocks_per_grid = (
        (pixel_width + 4 * threads_per_block[0] - 1) // (4 * threads_per_block[0]),
        (pixel_height + threads_per_block[1] - 1) // threads_per_block[1],
    )
    kernel_map["bayer8p_to_10p_kernel"](
        blocks_per_grid,
        threads_per_block,
        (dest, line_bytes, src, pixel_height, pixel_width),
    )
    return dest


# convert 8-bit bayer pattern to 12-bit bayer pattern with T_R12_PK_ISP encoding (2 pixels per 3 bytes)
def bayer8p_to_T_R12_PK_ISP(src):
    if src.ndim != 2:
        raise ValueError("src must be a 2D array")
    pixel_height, pixel_width = src.shape
    line_bytes = (pixel_width + 1) // 2 * 3
    line_bytes = ((line_bytes + 63) >> 6) << 6
    dest = cp.zeros((pixel_height, line_bytes), dtype=cp.uint8)
    threads_per_block = (32, 32)
    blocks_per_grid = (
        (pixel_width + 2 * threads_per_block[0] - 1) // (2 * threads_per_block[0]),
        (pixel_height + threads_per_block[1] - 1) // threads_per_block[1],
    )
    kernel_map["bayer8p_to_T_R12_PK_ISP_kernel"](
        blocks_per_grid,
        threads_per_block,
        (dest, line_bytes, src, pixel_height, pixel_width),
    )
    return dest


# generate 8-bit bayer pattern (GBRG)
def generate_bayerGB8p(pixel_height, pixel_width):
    threads_per_block = (32, 32)
    blocks_per_grid = (
        (pixel_width + threads_per_block[0] - 1) // threads_per_block[0],
        (pixel_height + threads_per_block[1] - 1) // threads_per_block[1],
    )
    dest = cp.zeros((pixel_height, pixel_width), dtype=cp.uint8)
    kernel_map["generate_bayerGB8p_kernel"](
        blocks_per_grid, threads_per_block, (dest, pixel_height, pixel_width)
    )
    return dest


@dataclass
class LoopConfig:
    num_frames: int
    frame_rate_per_second: int


# load data from a file
def load_data(filename, gpu=False):
    if gpu:
        return cp.fromfile(filename, dtype=cp.uint8)
    return np.fromfile(filename, dtype=np.uint8)


# sleep to match the target frame rate
def sleep_frame_rate(last_frame_time, frame_rate_per_second):
    delta_s = (
        1 / frame_rate_per_second - (time.time_ns() - last_frame_time) / 1000000000
    )
    if delta_s > 0:
        time.sleep(delta_s)


# initialize and loop through frames from the file `filename`, loading data into the `tensor`
def loop_single_from_file(loop_config, filename, data_plane, frame_size=0, gpu=False):
    # setup
    data = load_data(filename, gpu)
    if data is None or len(data) == 0:
        raise RuntimeError(f"Failed to load data from {filename}")
    n_bytes = len(data)

    if not frame_size:
        frame_size = n_bytes

    if frame_size > n_bytes:
        raise RuntimeError(
            f"frame size is larger than the file size: {frame_size} > {n_bytes}"
        )

    frame_count = 0
    # main loop
    offset = 0
    while not loop_config.num_frames or (frame_count < loop_config.num_frames):
        # you can slice, but the resulting array must be contiguous (cannot stride without making a copy)
        last_frame_time = time.time_ns()
        tensor_data = data[offset : offset + frame_size]
        sent_bytes = data_plane.send(tensor_data)
        if sent_bytes < 0:
            raise RuntimeError(f"Error sending data: {sent_bytes}")
        if sent_bytes > 0:
            frame_count += 1
            offset += frame_size
            if offset > n_bytes - frame_size:
                offset = 0
        sleep_frame_rate(last_frame_time, loop_config.frame_rate_per_second)


# combined loop function that loads data from file but waits for VB1940 streaming control
def loop_single_vb1940_from_file(
    loop_config, filename, vb1940, data_plane, frame_size=0, gpu=False
):
    # setup
    data = load_data(filename, gpu)
    if data is None:
        raise RuntimeError(f"Failed to load data from {filename}")
    n_bytes = len(data)

    if not frame_size:
        frame_size = n_bytes
    if frame_size > n_bytes:
        raise RuntimeError(
            f"frame size is larger than the file size: {frame_size} > {n_bytes}"
        )

    frame_count = 0
    offset = 0
    # main loop
    while not loop_config.num_frames or (frame_count < loop_config.num_frames):
        # you can slice, but the resulting array must be contiguous (cannot stride without making a copy)
        last_frame_time = time.time_ns()

        if vb1940.is_streaming():
            tensor_data = data[offset : offset + frame_size]
            sent_bytes = data_plane.send(tensor_data)
            if sent_bytes < 0:
                raise RuntimeError(f"Error sending data: {sent_bytes}")
            if sent_bytes > 0:
                frame_count += 1
                offset += frame_size
                if offset > n_bytes - frame_size:
                    offset = 0
        sleep_frame_rate(last_frame_time, loop_config.frame_rate_per_second)


# generate a vb1940 csi-2 format frame with optional packetization (query data_plane.packetizer_enabled())
def generate_vb1940_frame(is_gpu, vb1940, packetize):
    frame_generator = None
    if vb1940.get_pixel_bits() == 10:
        if packetize:
            frame_generator = bayer8p_to_T_X2Rc10Rb10Ra10
        else:
            frame_generator = bayer8p_to_10p

    pixel_height = vb1940.get_pixel_height()
    pixel_width = vb1940.get_pixel_width()
    start_byte = vb1940.get_image_start_byte()
    proto_frame = generate_bayerGB8p(pixel_height, pixel_width)
    if frame_generator:
        proto_frame = frame_generator(proto_frame)
    image_size = proto_frame.shape[1] * proto_frame.shape[0]
    frame = cp.zeros((image_size + proto_frame.shape[1] * 3,), dtype=cp.uint8)
    frame[start_byte : start_byte + image_size] = proto_frame.flatten()

    if is_gpu:
        return frame
    return cp.asnumpy(frame)


# the thread loop for serving a frame (configured on host side) in vb1940 csi-2 format as if from a single vb1940 sensor
def loop_single_vb1940(loop_config, vb1940, data_plane, gpu=False):
    """Initialize and run data serving loop"""
    streaming = False
    frame_count = 0

    while not loop_config.num_frames or (frame_count < loop_config.num_frames):
        last_frame_time = time.time_ns()

        if not streaming and vb1940.is_streaming():
            streaming = True
            frame_data = generate_vb1940_frame(
                gpu, vb1940, data_plane.packetizer_enabled()
            )
        elif not vb1940.is_streaming():
            streaming = False

        if streaming:
            sent_bytes = data_plane.send(frame_data)
            if sent_bytes < 0:
                raise RuntimeError(f"Error sending data: {sent_bytes}")
            if sent_bytes > 0:  # only increment frame_count if data was sent
                frame_count += 1

        sleep_frame_rate(last_frame_time, loop_config.frame_rate_per_second)


# the thread loop for serving two frames simultaneously (configured on host side) in vb1940 csi-2 format as if from stereo vb1940 sensors
def loop_stereo_vb1940(
    loop_config, vb1940_0, data_plane_0, vb1940_1, data_plane_1, gpu=False
):
    """Initialize and run data serving loop"""
    streaming = False
    frame_count = 0

    while not loop_config.num_frames or (frame_count < loop_config.num_frames):
        last_frame_time = time.time_ns()

        if not streaming and vb1940_0.is_streaming() and vb1940_1.is_streaming():
            streaming = True
            tensor_0 = generate_vb1940_frame(
                gpu, vb1940_0, data_plane_0.packetizer_enabled()
            )
            tensor_1 = generate_vb1940_frame(
                gpu, vb1940_1, data_plane_1.packetizer_enabled()
            )
        elif not vb1940_0.is_streaming() and not vb1940_1.is_streaming():
            streaming = False

        if streaming:
            sent_bytes_0 = 0
            sent_bytes_1 = 0

            if vb1940_0.is_streaming():
                sent_bytes_0 = data_plane_0.send(tensor_0)
                if sent_bytes_0 < 0:
                    raise RuntimeError(f"Error sending data: {sent_bytes_0}")

            if vb1940_1.is_streaming():
                sent_bytes_1 = data_plane_1.send(tensor_1)
                if sent_bytes_1 < 0:
                    raise RuntimeError(f"Error sending data: {sent_bytes_1}")

            if sent_bytes_0 + sent_bytes_1 > 0:
                frame_count += 1

        sleep_frame_rate(last_frame_time, loop_config.frame_rate_per_second)
