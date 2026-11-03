# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# GPIO pin sweep demo: GpioSetOp cycles every pin through a sequence of test configurations
# (all-output-low, all-output-high, all-input, alternating odd/even -- for use with a jumper
# wire between adjacent pins), GpioSetOp forwards the just-changed pin to GpioGetOp, which reads
# it back and logs direction/value, pausing to let a human probe the physical board with a
# multimeter. No C++ operator involved -- both are thin wrappers around a raw hololink.GPIO
# register handle (hololink_dev.get_gpio(metadata), restored as part of this port; GPIO wasn't
# bound to Python at all before this round).
#
# Pacing: the old app let GpioSetOp tick freely and relied on GpioGetOp's own time.sleep(
# sleep_time) plus a single-slot downstream queue to backpressure the whole pipeline to roughly
# one pin change per sleep_time seconds. Here GpioSetOp's own hs.OnClock(period=sleep_time) is
# paced directly at the intended cadence instead -- same real-world outcome (one change every
# sleep_time seconds), without depending on backpressure semantics to get there; GpioGetOp's own
# sleep is dropped as redundant pacing, not a separate kept-for-fidelity behavior.
#
# --cycle-limit: the old app's CountCondition counted GpioGetOp's own ticks (one per GPIO
# change, not per full configs sweep -- a bit of a misnomer in the old flag's own name, preserved
# as-is). Here GpioSetOp counts its own compute() calls directly and calls
# context.request_stop() once the limit is reached -- simpler than this effort's established
# cross-thread _FrameLimitObserver pattern, which exists specifically for apps where counting
# and stopping happen on different operators; here GpioSetOp is naturally both.

import argparse
import logging
import sys
from datetime import timedelta

import holoscan as hs

import hololink as hl

# configurations to test
CONFIGS = [
    "ALL_OUT_L",  # All pins output low
    "ALL_OUT_H",  # All pins output high
    "ALL_IN",  # All pins inputs
    "ODD_OUT_H",  # Odd pins output high, even pins input - use jumper to short & test
    "EVEN_OUT_H",  # Even pins output high, Odd pins input - use jumper to short & test
]

# dictionary for print beautification
DIRECTION_NAMES = {0: "output", 1: "input"}


class _GpioShared:
    """Shared hololink.GPIO handle: graph.op() can't take a live object as a kwarg, so both
    operators read it from here instead, matching this whole porting effort's established
    class-level-stash pattern for sharing a live object between two pure-Python operators.
    """

    gpio = None


class GpioSetOp(hs.Operator):
    """Sweeps every pin through CONFIGS, one pin change per compute(); forwards the changed pin
    and current test configuration to GpioGetOp for validation.

    The changed pin and test configuration are always produced together (every compute() call
    emits both), so they're sent as one (pin, test_config) tuple through a single port rather
    than two separate ports -- the latter would need OnAligned-style pairing (this effort's
    established mechanism for correlating two ports' data) despite having no natural capture-time
    relationship to align on, since these are plain scalars, not tensor/metadata samples.
    """

    pin_and_config_out = hs.Output("pin_and_config_out", hs.pyobject)

    sleep_time = hs.Param(2.0)
    cycle_limit = hs.Param(0)  # 0 = unlimited

    def setup(self, spec):
        spec.output(self.pin_and_config_out)

    def contract(self):
        return hs.Contract().trigger(
            hs.OnClock(period=timedelta(seconds=self.sleep_time))
        )

    def start(self):
        self._gpio = _GpioShared.gpio
        self._supported_pins_number = self._gpio.get_supported_pin_num()
        self.pin = 0
        self.test_config = 0
        self._count = 0

        # set all gpios as output, high -- test fast setting via loop
        for i in range(self._supported_pins_number):
            self._gpio.set_direction(i, self._gpio.OUT)
            self._gpio.set_value(i, self._gpio.HIGH)

    def compute(self, context):
        # Checked first, before any register write: request_stop() lets the current scheduler
        # generation finish rather than aborting mid-flight (confirmed by testing, not assumed --
        # a fast clock period let a handful of extra compute() calls through after the limit was
        # first reached when this check ran only at the end). Returning here before touching
        # hardware bounds that slop to "a few extra no-op ticks", not "a few extra pin writes".
        if self.cycle_limit and self._count >= self.cycle_limit:
            context.request_stop()
            return

        logging.info(f"GpioSetOp[{CONFIGS[self.test_config]}]")

        # set gpio pins per test configuration 1 pin at a time per current
        # configuration tested
        if CONFIGS[self.test_config] == "ALL_OUT_L":
            self._gpio.set_direction(self.pin, self._gpio.OUT)
            self._gpio.set_value(self.pin, self._gpio.LOW)

        elif CONFIGS[self.test_config] == "ALL_OUT_H":
            self._gpio.set_direction(self.pin, self._gpio.OUT)
            self._gpio.set_value(self.pin, self._gpio.HIGH)

        elif CONFIGS[self.test_config] == "ALL_IN":
            self._gpio.set_direction(self.pin, self._gpio.IN)

        elif CONFIGS[self.test_config] == "ODD_OUT_H":
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

        # send current changed pin and tested configuration to the next operator to
        # validate value changes
        self.pin_and_config_out.emit((self.pin, self.test_config))

        # prepare for next gpio to change
        self.pin += 1
        self.pin %= self._supported_pins_number

        # done pins sweep -- move to next configuration to test
        if self.pin == 0:
            self.test_config += 1
            self.test_config %= len(CONFIGS)

        self._count += 1


class GpioGetOp(hs.Operator):
    """Reads back the pin GpioSetOp just changed and logs its direction/value."""

    pin_and_config_in = hs.Input("pin_and_config_in", hs.pyobject)

    def setup(self, spec):
        spec.input(self.pin_and_config_in)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.pin_and_config_in))

    def start(self):
        self._gpio = _GpioShared.gpio

    def compute(self, context):
        pin, test_config = self.pin_and_config_in.receive_data()

        logging.info(f"GpioGetOp[{CONFIGS[test_config]}]")
        direction = self._gpio.get_direction(pin)
        value = self._gpio.get_value(pin)
        logging.info(f"pin:{pin}, direction:{DIRECTION_NAMES[direction]},value:{value}")


def run(hololink_ip, sleep_time, cycle_limit):
    metadata = hl.find_channel(hololink_ip)
    channel = hl.DataChannel(metadata)
    hololink_dev = channel.hololink()
    hololink_dev.start()
    try:
        hololink_dev.reset()

        _GpioShared.gpio = hololink_dev.get_gpio(metadata)

        graph = hs.Graph("gpio_example_app")
        gpio_set = graph.op(
            GpioSetOp, "gpio_set", sleep_time=sleep_time, cycle_limit=cycle_limit or 0
        )
        gpio_read = graph.op(GpioGetOp, "gpio_read")
        graph.add_flow(gpio_set.pin_and_config_out, gpio_read.pin_and_config_in)

        clock_key = graph.add_clock(hs.RealtimeClock, "clock")
        graph.set_default_clock(clock_key)

        plan = hs.compile(graph, hs.CompileOptions())
        if not plan.ok:
            print("Graph compilation failed:", file=sys.stderr)
            for diag in plan.diagnostics:
                print(f"  {diag}", file=sys.stderr)
            return 1

        with hs.run_async(plan) as session:
            session.wait()
    finally:
        hololink_dev.stop()

    return 0


def main():
    parser = argparse.ArgumentParser()
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

    logging.basicConfig(level=args.log_level)
    logging.info("Initializing.")

    sys.exit(run(args.hololink, args.sleep_time, args.cycle_limit))


if __name__ == "__main__":
    main()
