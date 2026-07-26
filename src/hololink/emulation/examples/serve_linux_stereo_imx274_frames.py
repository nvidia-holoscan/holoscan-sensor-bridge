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
import copy
import sys

from emulator_utils import LoopConfig, loop_stereo_imx274

import hololink as hololink_module
import hololink.emulation as hemu

"""
@brief Example program to serve frames from a file to a HSB Emulator as if
from stereo imx274 sensors

The program emulates both the HSB and imx274 sensors and uses their interfaces to
   1) generate the frame based on configuration from the host size (frame shape/size)
      data can either be in CPU or GPU memory
   2) when the imx274 are set by the host to streaming mode, the program begins to send
      the data through the LinuxDataPlane (RoCEv2 packets over UDP sockets)
   3) The rate at which it will send data defaults to the program parameters (frame rate).
      Note that ultimately the tuning of the Emulator Device's network and CPU stack will
      determine whether it can reach the target frame rate
   4) The program will run until the number of frames specified by the user is reached
      (infinite for 0, the default) or the user presses Ctrl+C

Default values
- frame-rate: 30 (per second)
- frame-limit: 0 (infinite)
- gpu: false (serve data from the CPU)
"""


def main():
    parser = argparse.ArgumentParser(
        description="Serve CSI-2 data as if from stereo imx274 sensors using Linux data plane"
    )
    parser.add_argument(
        "ip_address",
        type=str,
        help="IP address of the HSB Emulator device. Note: for roce receivers, this should be on the same subnet as the receiver and physically connected",
    )
    parser.add_argument(
        "-r",
        "--frame-rate",
        type=int,
        default=30,
        help="Frame rate in frames per second (default: 30)",
    )
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

    loop_config = LoopConfig(
        num_frames=args.frame_limit,
        frame_rate_per_second=args.frame_rate,
    )

    config = copy.copy(hemu.HSB_EMULATOR_CONFIG)
    config.sensor_count = 2
    config.sifs_per_sensor = 2

    # build the emulator and data plane(s)
    hsb = hemu.HSBEmulator(config)
    address = hemu.IPAddress(args.ip_address)
    data_plane_id = 0
    # NOTE: the sensor_ids connect the data planes to the sensors as configured on the HSB board
    sensor_id_0 = 0
    sensor_id_1 = 1

    linux_data_plane_0 = hemu.LinuxDataPlane(hsb, address, data_plane_id, sensor_id_0)
    linux_data_plane_1 = hemu.LinuxDataPlane(hsb, address, data_plane_id, sensor_id_1)

    # create and attach the imx274 sensors to the HSB's I2C controller
    imx274_0 = hemu.sensors.IMX274Emulator()
    imx274_1 = hemu.sensors.IMX274Emulator()
    # The i2c bus address is the sensor_id offset from CAM_I2C_BUS
    imx274_0.attach_to_i2c(
        hsb.get_i2c(hololink_module.I2C_CTRL), hololink_module.CAM_I2C_BUS + sensor_id_0
    )
    imx274_1.attach_to_i2c(
        hsb.get_i2c(hololink_module.I2C_CTRL), hololink_module.CAM_I2C_BUS + sensor_id_1
    )

    # start the emulator
    hsb.start()
    # stop the bootp broadcast from one of the data planes on the same IPAddress
    # if data planes are on different addresses, do not do this
    linux_data_plane_1.stop_bootp()

    # run your loop/thread/operator
    print("Running HSB Emulator... Press Ctrl+C to stop")
    try:
        loop_stereo_imx274(
            loop_config,
            imx274_0,
            linux_data_plane_0,
            imx274_1,
            linux_data_plane_1,
            args.gpu,
        )
    except KeyboardInterrupt:
        print("\nReceived Ctrl+C, stopping...")
    finally:
        # stop the emulator (if cleanup is needed)
        hsb.stop()


if __name__ == "__main__":
    main()
