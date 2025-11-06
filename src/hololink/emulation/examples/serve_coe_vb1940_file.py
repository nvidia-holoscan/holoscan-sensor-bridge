#
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
#
# See README.md for detailed information.
#

import argparse
import sys

from emulator_utils import LoopConfig, loop_single_vb1940_from_file

import hololink as hololink_module
import hololink.emulation as hemu

"""
@brief Example program to serve frames from a file to a SIPL application (Thor) using the HSB Emulator with VB1940 streaming control

 The program will:
   1) Create a VB1940 emulator instance and attach it to the HSB
   2) Wait for the VB1940 to be set to streaming mode by the host
   3) Load data from a source file
   4) Control streaming based on VB1940 emulator state
   5) Send frames from the loaded file data at the specified rate
   6) Serve the data through the COEDataPlane (Camera-over-Ethernet protocol transmission)

 Default values:
 - frame-rate: 30 (per second)
 - frame-limit: 0 (infinite)
 - frame-size: 0 (use the entire size of the source file)
 - gpu: false (serve data from the CPU)
"""


def main():
    parser = argparse.ArgumentParser(
        description="Serve frames from a file as if from a single vb1940 sensor connected to a SIPL application"
    )
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
        default=30,
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
        print("Frame rate must be a positive integer")
        sys.exit(1)
    if args.frame_limit < 0:
        print("Frame limit must be a non-negative integer")
        sys.exit(1)
    if args.frame_size < 0:
        print("Frame size must be a non-negative integer")
        sys.exit(1)

    loop_config = LoopConfig(
        num_frames=args.frame_limit,
        frame_rate_per_second=args.frame_rate,
    )

    # build the emulator and data plane(s)
    # NOTE: this is the configuration of the Leopard Eagle board,
    # which is currently the only one supported with vb1940 in SIPL applications
    hsb = hemu.HSBEmulator(hemu.HSB_LEOPARD_EAGLE_CONFIG)
    address = hemu.IPAddress(args.ip_address)
    data_plane_id = 0
    sensor_id = 0
    coe_data_plane = hemu.COEDataPlane(hsb, address, data_plane_id, sensor_id)

    # create and attach the vb1940 sensors to the HSB's I2C controller
    vb1940 = hemu.sensors.Vb1940Emulator()
    # On Leopard Eagle, the i2c bus address is the sensor_id offset from CAM_I2C_BUS
    vb1940.attach_to_i2c(
        hsb.get_i2c(hololink_module.I2C_CTRL), hololink_module.CAM_I2C_BUS + sensor_id
    )

    # start the emulator
    hsb.start()

    # run your loop/thread/operator
    print("Running HSB Emulator... Press Ctrl+C to stop")
    try:
        loop_single_vb1940_from_file(
            loop_config,
            args.filename,
            vb1940,
            coe_data_plane,
            args.frame_size,
            args.gpu,
        )
    except KeyboardInterrupt:
        print("\nReceived Ctrl+C, stopping...")
    finally:
        # stop the emulator (if cleanup is needed)
        hsb.stop()


if __name__ == "__main__":
    main()
