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

import argparse
import time
from dataclasses import dataclass

import hololink.emulation as hemu


@dataclass
class LoopConfig:
    frame_size: int
    num_cycles: int
    frame_rate_per_second: int


"""
@brief Example program to serve frames from a file to a HSB Emulator as a single sensor

This program is used to serve frames from a source file of raw bytes from an HSB emulator.

The program will read data from a source file, breaking it up into frames of size 'frame_size'
and send the entirety of the file 'num_cycles' times at a rate of 'frame_rate_per_second'
frames per second.

Frames will not start being sent until a connection is established with a receiver (a target
address and port is detected).

default values:
- frame_rate_per_second: 60
- num_cycles: 0 (infinite)
- frame_size: 0 (use the entire size of the source file)

for details of other parameters, see the help message below or running

/path/to/serve_frames --help
"""


def load_data(filename, gpu=False):
    if gpu:
        try:
            import cupy as cp
        except ImportError:
            print("Cupy is not installed. Cannot specify --gpu flag.")
            exit(1)
        return cp.fromfile(filename, dtype=cp.uint8)
    else:
        import numpy as np

        return np.fromfile(filename, dtype=np.uint8)


def sleep_frame_rate(last_frame_time, frame_rate_per_second):
    delta_s = (
        1 / frame_rate_per_second - (time.time_ns() - last_frame_time) / 1000000000
    )
    if delta_s > 0:
        time.sleep(delta_s)


def loop(data_plane, filename, loop_config, gpu=False):
    # setup
    data = load_data(filename, gpu)
    if not len(data):
        print(f"Failed to load data from {filename}")
        return

    if not loop_config.frame_size:
        print("setting frame size to length of data")
        loop_config.frame_size = len(data)

    num_frames = len(data) // loop_config.frame_size
    if not num_frames:
        print("Frame size is larger than the file size")
        return
    cycle_count = 0
    frame_count = 0

    # main loop
    while not loop_config.num_cycles or cycle_count < loop_config.num_cycles:
        # you can slice, but the resulting array must be contiguous (cannot stride without making a copy)
        tensor_data = data[
            frame_count
            * loop_config.frame_size : min(
                (frame_count + 1) * loop_config.frame_size, len(data)
            )
        ]
        last_frame_time = time.time_ns()
        sent_bytes = data_plane.send(tensor_data)
        if sent_bytes < 0:
            print(f"Error sending data: {sent_bytes}")
        sleep_frame_rate(last_frame_time, loop_config.frame_rate_per_second)
        if sent_bytes > 0:
            frame_count += 1
            if frame_count >= num_frames:
                frame_count = 0
                cycle_count += 1


def main():

    parser = argparse.ArgumentParser(description="Serve frames from a file")
    parser.add_argument(
        "ip_address", type=str, help="IP address of the HoloLink device"
    )
    parser.add_argument(
        "filename", type=str, help="Path to the file containing the frames"
    )
    parser.add_argument(
        "--subnet-bits", type=int, default=24, help="Subnet bits for the IP address"
    )
    parser.add_argument(
        "--source-port", type=int, default=12888, help="Source port for the data plane"
    )
    parser.add_argument(
        "--frame-rate", type=int, default=60, help="Frame rate in frames per second"
    )
    parser.add_argument("--frame-size", type=int, default=0, help="Frame size")
    parser.add_argument(
        "--num-cycles", type=int, default=10, help="Number of cycles to run"
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    args = parser.parse_args()

    # check arguments
    if args.subnet_bits < 0 or args.subnet_bits > 32:
        print("Subnet bits must be between 0 and 32")
        return
    if args.source_port < 1024 or args.source_port > 65535:
        print("Source port must be between 1024 and 65535")
        return
    if args.frame_rate <= 0:
        print("Frame rate must be greater than 0")
        return
    if args.num_cycles < 0:
        print("Number of cycles must be greater than 0")
        return
    if args.frame_size < 0:
        print("Frame size must be greater than 0")
        return

    loop_config = LoopConfig(
        frame_size=args.frame_size,
        num_cycles=args.num_cycles,
        frame_rate_per_second=args.frame_rate,
    )

    # build the emulator and data plane(s)
    hsb = hemu.HSBEmulator()
    linux_data_plane = hemu.LinuxDataPlane(
        hsb,
        hemu.IPAddress(args.ip_address, args.subnet_bits),
        args.source_port,
        hemu.DataPlaneID.DATA_PLANE_0,
        hemu.SensorID.SENSOR_0,
    )

    # start the emulator
    hsb.start()

    # run your loop/thread/operator
    print("Running the HSB Emulator... Press Ctrl+C to stop")
    loop(linux_data_plane, args.filename, loop_config, args.gpu)

    # stop the emulator (if cleanup is needed)
    hsb.stop()


if __name__ == "__main__":
    main()
