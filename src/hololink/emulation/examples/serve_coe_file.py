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
import sys

from emulator_utils import LoopConfig, loop_single_from_file

import hololink.emulation as hemu

"""
@brief Example program to serve frames from a file to a HSB Emulator as a single sensor

This program is used to serve frames from a source file of raw bytes from an HSB emulator.

The program will read data from a source file, breaking it up into frames of size 'frame_size'
and send the entirety of the file 'frame_limit' times at a rate of 'frame_rate_per_second'
frames per second using Camera-over-Ethernet (CoE) packets over Linux sockets.

Frames will not start being sent until a connection is established with a receiver (a target
address and port is detected).

default values:
- frame_rate_per_second: 60
- frame_limit: 0 (infinite)
- frame_size: 0 (use the entire size of the source file)

for details of other parameters, see the help message below or running

/path/to/serve_coe_file --help
"""


def main():

    parser = argparse.ArgumentParser(description="Serve frames from a file")
    parser.add_argument(
        "ip_address", type=str, help="IP address of the HoloLink device"
    )
    parser.add_argument(
        "filename", type=str, help="Path to the file containing the frames"
    )
    parser.add_argument(
        "-r",
        "--frame-rate",
        type=int,
        default=60,
        help="Frame rate in frames per second",
    )
    parser.add_argument("-s", "--frame-size", type=int, default=0, help="Frame size")
    parser.add_argument(
        "-l",
        "--frame-limit",
        type=int,
        default=0,
        help="Number of frames to serve (default: 0 - infinite)",
    )
    parser.add_argument(
        "-g", "--gpu", action="store_true", help="Serve the data from the GPU"
    )

    args = parser.parse_args()

    # check arguments
    if args.frame_rate <= 0:
        print("Frame rate must be greater than 0")
        sys.exit(1)
    if args.frame_limit < 0:
        print("Number of frames must be greater than 0")
        sys.exit(1)
    if args.frame_size < 0:
        print("Frame size must be greater than 0")
        sys.exit(1)

    loop_config = LoopConfig(
        num_frames=args.frame_limit,
        frame_rate_per_second=args.frame_rate,
    )

    # build the emulator and data plane(s)
    hsb = hemu.HSBEmulator()
    data_plane_id = 0
    sensor_id = 0
    data_plane = hemu.COEDataPlane(
        hsb,
        hemu.IPAddress(args.ip_address),
        data_plane_id,
        sensor_id,
    )
    # start the emulator
    hsb.start()

    # run your loop/thread/operator
    print("Running the HSB Emulator... Press Ctrl+C to stop")
    loop_single_from_file(
        loop_config, args.filename, data_plane, args.frame_size, args.gpu
    )

    # stop the emulator (if cleanup is needed)
    hsb.stop()


if __name__ == "__main__":
    main()
