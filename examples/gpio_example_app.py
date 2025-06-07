# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import time

import holoscan

import hololink as hololink_module

# configurations to test
configs = [
    "ALL_OUT_L",  # All pins output low
    "ALL_OUT_H",  # All pins output high
    "ALL_IN",  # All pins inputs
    "ODD_OUT_H",  # Odd pins output high, even pins input - use jumper to short & test
    "EVEN_OUT_H",  # Even pins output high, Odd pins input - use jumper to short & test
]

# dictionary for print beautification
dir = {0: "output", 1: "input"}


# define custom Operators for use in the demo
class GpioSetOp(holoscan.core.Operator):
    """
    operator to demonstrate different GPIO test configuration settings.
    This operator changes the GPIOs pin by pin according to the test
    configuration it currently runs (specified in the 'configs' variable).
    It sends the latest changed pin + the current test configuration to the
    GpioGetOp for validation purposes.
    once a sweep on all 16 pins is completed, next test configuration is set.
    """

    def __init__(self, fragment, hololink_channel, gpio, *args, **kwargs):
        self._hololink = hololink_channel.hololink()
        self._gpio = gpio
        self.pin = 0
        self.test_config = 0

        # how many pins are supported on the platform running the example
        self._supported_pins_number = self._gpio.get_supported_pin_num()

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: holoscan.core.OperatorSpec):
        spec.output("gpio_changed_out")
        spec.output("test_config_out")

        # set all gpios as output, high - test fast setting via loop
        for i in range(self._supported_pins_number):
            self._gpio.set_direction(i, self._gpio.OUT)
            self._gpio.set_value(i, self._gpio.HIGH)

    def compute(self, op_input, op_output, context):

        logging.info(f"GpioSetOp[{configs[self.test_config]}]")

        # set gpio pins per test configuration 1 pin at a time per current
        # configuration tested
        if configs[self.test_config] == "ALL_OUT_L":
            self._gpio.set_direction(self.pin, self._gpio.OUT)
            self._gpio.set_value(self.pin, self._gpio.LOW)

        elif configs[self.test_config] == "ALL_OUT_H":
            self._gpio.set_direction(self.pin, self._gpio.OUT)
            self._gpio.set_value(self.pin, self._gpio.HIGH)

        elif configs[self.test_config] == "ALL_IN":
            self._gpio.set_direction(self.pin, self._gpio.IN)

        elif configs[self.test_config] == "ODD_OUT_H":
            if (self.pin & 0x1) == 1:  # odd pin
                self._gpio.set_direction(self.pin, self._gpio.OUT)
                self._gpio.set_value(self.pin, self._gpio.HIGH)
            else:  # even pin
                self._gpio.set_direction(self.pin, self._gpio.IN)

        else:  # EVEN_OUT_H
            if (self.pin & 0x1) == 1:  # odd pin
                self._gpio.set_direction(self.pin, self._gpio.IN)
            else:  # even pin
                self._gpio.set_direction(self.pin, self._gpio.OUT)
                self._gpio.set_value(self.pin, self._gpio.HIGH)

        # send current changed pin and tested configuration to
        # second operator to validate value changes
        op_output.emit(self.pin, "gpio_changed_out")
        op_output.emit(self.test_config, "test_config_out")

        # prepare for next gpio to change
        self.pin += 1
        self.pin %= self._supported_pins_number

        # done pins sweep - move to next configuration to test
        if self.pin == 0:
            self.test_config += 1
            self.test_config %= len(configs)


class GpioGetOp(holoscan.core.Operator):
    """
    operator to demonstrate reads from the GPIO bank
    Receives the last changed GPIO number and it is set to input
    reads and prints it value and direction.
    sleeps for the given time to allow physcal board measurements
    """

    def __init__(self, fragment, hololink_channel, gpio, sleep_time, *args, **kwargs):
        self._hololink = hololink_channel.hololink()
        self._gpio = gpio
        self._sleep_time = sleep_time

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: holoscan.core.OperatorSpec):
        spec.input("gpio_changed_in")
        spec.input("test_config_in")

    def compute(self, op_input, op_output, context):

        pin = op_input.receive("gpio_changed_in")
        test_config = op_input.receive("test_config_in")

        logging.info(f"GpioGetOp[{configs[test_config]}]")
        direction = self._gpio.get_direction(pin)
        value = self._gpio.get_value(pin)
        logging.info(f"pin:{pin}, direction:{dir[direction]},value:{value}")

        # sleep to allow time for physical board measurememt
        time.sleep(self._sleep_time)


class HoloscanApplication(holoscan.core.Application):
    def __init__(
        self,
        hololink_channel,
        channel_metadata,
        sleep_time,
        cycle_limit,
    ):
        logging.info("__init__")
        super().__init__()
        self._hololink_channel = hololink_channel
        self._channel_metadata = channel_metadata
        self._hololink = hololink_channel.hololink()
        self._sleep_time = sleep_time
        self._cycle_limit = cycle_limit  # may be None

    def compose(self):
        logging.info("compose")

        # create the GPIO instance to be shared with the two operators
        self._gpio = self._hololink.get_gpio(self._channel_metadata)

        # Conditions support cycle-limit
        if self._cycle_limit:
            self._count = holoscan.conditions.CountCondition(
                self,
                name="count",
                count=self._cycle_limit,
            )
            condition = self._count
        else:
            self._ok = holoscan.conditions.BooleanCondition(
                self, name="ok", enable_tick=True
            )
            condition = self._ok

        # example of operator instantiation
        gpio_read = GpioGetOp(
            self,
            self._hololink_channel,
            self._gpio,
            self._sleep_time,
            condition,
            name="gpio_read",
        )
        gpio_set = GpioSetOp(self, self._hololink_channel, self._gpio, name="gpio_set")
        self.add_flow(
            gpio_set,
            gpio_read,
            {
                ("gpio_changed_out", "gpio_changed_in"),
                ("test_config_out", "test_config_in"),
            },
        )


def main():
    parser = argparse.ArgumentParser()

    default_configuration = os.path.join(
        os.path.dirname(__file__), "example_configuration.yaml"
    )
    parser.add_argument(
        "--configuration",
        default=default_configuration,
        help="Configuration file",
    )
    parser.add_argument(
        "--hololink",
        default="192.168.0.2",
        help="IP address of Hololink board",
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level to display",
    )
    parser.add_argument(
        "--sleep-time",
        type=float,
        default=2.0,
        help="Time to allow for physical signal measurement.",
    )
    parser.add_argument(
        "--cycle-limit",
        type=int,
        help="Limit the number of cycles for the application; by default this runs forever.",
    )

    args = parser.parse_args()
    hololink_module.logging_level(args.log_level)
    logging.info("Initializing.")

    # Get a handle to the Hololink device
    channel_metadata = hololink_module.Enumerator.find_channel(channel_ip=args.hololink)
    hololink_channel = hololink_module.DataChannel(channel_metadata)

    # Set up the application
    application = HoloscanApplication(
        hololink_channel,
        channel_metadata,
        args.sleep_time,
        args.cycle_limit,
    )
    application.config(args.configuration)
    # Run it.
    hololink = hololink_channel.hololink()
    hololink.start()
    hololink.reset()
    application.run()
    hololink.stop()


if __name__ == "__main__":
    main()
