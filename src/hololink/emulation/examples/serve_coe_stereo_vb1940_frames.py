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

from emulator_utils import LoopConfig, loop_stereo_vb1940

import hololink as hololink_module
import hololink.emulation as hemu

"""
@brief Example program to serve frames from a file to a HSB Emulator as if
from stereo vb1940 sensors (e.g. Leopard Eagle HSB)

The program emulates both the HSB and vb1940 sensors and uses their interfaces to
   1) generate the frame based on configuration from the host size (frame shape/size)
      data can either be in CPU or GPU memory
   2) when the vb1940 are set by the host to streaming mode, the program begins to send
      the data through the COEDataPlane (Camera-over-Ethernet protocol transmission)
   3) The rate at which it will send data defaults to Leopard Eagle values but can be tuned
      by program parameters (frame rate). Note that ultimately the tuning of the Emulator
      Device's network and CPU stack will determine whether it can reach the target frame rate
   4) The program will run until the number of frames specified by the user is reached
      (infinite for 0, the default) or the user presses Ctrl+C

Default values
- frame-rate: 30 (per second)
- frame-limit: 0 (infinite)
- gpu: false (serve data from the CPU)
"""


def main():
    parser = argparse.ArgumentParser(
        description="Serve CSI-2 data as if from a stereo vb1940 sensor"
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
        print("Frame limit must be a positive integer")
        sys.exit(1)

    loop_config = LoopConfig(
        num_frames=args.frame_limit,
        frame_rate_per_second=args.frame_rate,
    )

    # build the emulator and data plane(s)
    # NOTE: this is the configuration of the Leopard Eagle board,
    # which is currently the only one that will work with vb1940
    hsb = hemu.HSBEmulator(hemu.HSB_LEOPARD_EAGLE_CONFIG)
    address = hemu.IPAddress(args.ip_address)
    data_plane_id = 0
    sensor_id_0 = 0
    sensor_id_1 = 1

    coe_data_plane_0 = hemu.COEDataPlane(hsb, address, data_plane_id, sensor_id_0)
    coe_data_plane_1 = hemu.COEDataPlane(hsb, address, data_plane_id, sensor_id_1)

    # create and attach the vb1940 sensors to the HSB's I2C controller
    vb1940_0 = hemu.sensors.Vb1940Emulator()
    vb1940_1 = hemu.sensors.Vb1940Emulator()
    # On Leopard Eagle, the i2c bus address is the sensor_id offset from CAM_I2C_BUS
    vb1940_0.attach_to_i2c(
        hsb.get_i2c(hololink_module.I2C_CTRL), hololink_module.CAM_I2C_BUS + sensor_id_0
    )
    vb1940_1.attach_to_i2c(
        hsb.get_i2c(hololink_module.I2C_CTRL), hololink_module.CAM_I2C_BUS + sensor_id_1
    )

    # start the emulator
    hsb.start()
    # stop the bootp broadcast from one of the data planes on the same IPAddress
    # if data planes are on different addresses, do not do this
    coe_data_plane_1.stop_bootp()

    # run your loop/thread/operator
    print("Running HSB Emulator... Press Ctrl+C to stop")
    try:
        loop_stereo_vb1940(
            loop_config,
            vb1940_0,
            coe_data_plane_0,
            vb1940_1,
            coe_data_plane_1,
            args.gpu,
        )
    except KeyboardInterrupt:
        print("\nReceived Ctrl+C, stopping...")
    finally:
        # stop the emulator (if cleanup is needed)
        hsb.stop()


if __name__ == "__main__":
    main()
