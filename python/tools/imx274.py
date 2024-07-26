# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import logging

import hololink as hololink_module


def _get_register(args, camera, hololink):
    address = args.address
    value = camera.get_register(address)
    logging.info(f"{address=:#x} {value=:#x}")
    print(hex(value))


def _set_register(args, camera, hololink):
    address = args.address
    value = args.value
    camera.set_register(address, value)


def _configure(args, camera, hololink):
    mode = hololink_module.sensors.imx274.imx274_mode.Imx274_Mode(args.mode)
    camera.setup_clock()
    camera.configure(mode)
    if args.start:
        camera.start()


def _start(args, camera, hololink):
    camera.start()


def _reset(args, camera, hololink):
    hololink.reset()


def _pattern(args, camera, hololink):
    if args.disable:
        camera.set_register(0x303C, 0)
        camera.set_register(0x377F, 0)
        camera.set_register(0x3781, 0)
        camera.set_register(0x370B, 0)
    else:
        camera.set_register(0x303C, 0x11)
        camera.set_register(0x370E, 0x01)
        camera.set_register(0x377F, 0x01)
        camera.set_register(0x3781, 0x01)
        camera.set_register(0x370B, 0x11)
    if args.value:
        camera.set_register(0x303D, args.value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hololink",
        default="192.168.0.2",
        help="IP address of Hololink board",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Configure logging for debug output",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Configure logging for trace output",
    )
    parser.add_argument(
        "--expander-configuration",
        type=int,
        default=0,
        choices=(0, 1),
        help="I2C Expander configuration",
    )
    subparsers = parser.add_subparsers(
        help="Subcommands", dest="command", required=True
    )

    # "get_register": Fetch an IMX274 register
    sp = subparsers.add_parser("get_register", help="Fetch an IMX274 register")
    sp.add_argument(
        "address",
        type=lambda x: int(x, 0),
        help="Address to read; prefix with 0x to interpret address as hex.",
    )
    sp.set_defaults(go=_get_register)

    # "set_register (address) (value)": Write a given value to the given IMX274 register.
    sp = subparsers.add_parser(
        "set_register",
        help="Write a given value to the given register; prefix with 0x to interpret values as hex.",
    )
    sp.add_argument(
        "address",
        type=lambda x: int(x, 0),
        help="Address to write; prefix with 0x to interpret address as hex.",
    )
    sp.add_argument(
        "value",
        type=lambda x: int(x, 0),
        help="Value to write; prefix with 0x to interpret address as hex.",
    )
    sp.set_defaults(go=_set_register)

    # "configure [--mode mode] [--start]": Write configuration to the camera, necessary before start."
    sp = subparsers.add_parser(
        "configure",
        help="Write configuration to the camera",
    )
    sp.add_argument(
        "--mode",
        type=int,
        default=hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_3840X2160_60FPS.value,
        help="IMX274 mode",
    )
    sp.add_argument(
        "--start",
        action="store_true",
        help="Also call camera.start()",
    )
    sp.set_defaults(go=_configure)

    # "start": Enable data plane traffic from the camera; be sure to configure first.
    sp = subparsers.add_parser(
        "start",
        help="Enable data plane transmission from the camera",
    )
    sp.set_defaults(go=_start)

    # "reset": Call hololink.reset() with a camera instantiated.
    sp = subparsers.add_parser(
        "reset",
        help="Call hololink reset to reset the camera device.",
    )
    sp.set_defaults(go=_reset)

    # "pattern" -- Enable test pattern; add "--disable" to disable test pattern
    # Useful values to use with "--value":
    #   - black: 1
    #   - white: 2
    #   - darker gray: 3
    #   - lighter gray: 4
    #   - v color bars: 10
    #   - h color bars: 11
    sp = subparsers.add_parser(
        "pattern",
        help="Configure IMX274 for a test video pattern",
    )
    sp.add_argument(
        "--disable",
        action="store_true",
        help="Unconfigure test pattern",
    )
    sp.add_argument(
        "--value",
        type=int,
        default=10,
        help="Pattern to configure",
    )
    sp.set_defaults(go=_pattern)

    #
    args = parser.parse_args()

    if args.trace:
        logging.basicConfig(level=logging.TRACE)
    elif args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    #
    channel_metadata = hololink_module.Enumerator.find_channel(channel_ip=args.hololink)
    hololink_channel = hololink_module.DataChannel(channel_metadata)
    camera = hololink_module.sensors.imx274.dual_imx274.Imx274Cam(
        hololink_channel, expander_configuration=args.expander_configuration
    )
    hololink = hololink_channel.hololink()
    hololink.start()

    #
    args.go(args, camera, hololink)

    hololink.stop()


if __name__ == "__main__":
    main()
