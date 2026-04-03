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

import enum
import logging
import threading
import time

import hololink as hololink_module

device_by_serial_number = {}
device_lock = threading.Lock()


def reset_device_map():
    global device_by_serial_number
    global device_lock
    with device_lock:
        device_by_serial_number = {}


class Device:
    """Device represents an HSB unit itself which houses any number of
    sensors."""

    @staticmethod
    def get_device(enumeration_metadata):
        global device_lock, device_by_serial_number
        with device_lock:
            serial_number = enumeration_metadata["serial_number"]
            device = device_by_serial_number.get(serial_number)
            if device is None:
                device = Device(enumeration_metadata)
                device_by_serial_number[serial_number] = device
                device.start()
            return device

    # States
    class State(enum.Enum):
        NOT_FOUND = 100  # no I/O is possible to this device
        RESET = 200  # waiting for the device to come out of reset
        CONNECTED = 300  # I/O works but we aren't PTP sync'd yet
        RUN = 400  # PTP is sync'd up so timestamps are valid

    def __init__(self, enumeration_metadata):
        serial_number = enumeration_metadata["serial_number"]
        peer_ip = enumeration_metadata["peer_ip"]
        self._serial_number = serial_number
        # this is the IP address we use for ECB
        self._peer_ip = peer_ip
        #
        self._ptp_poll_time = 0.1  # seconds
        self._ptp_timeout = 30  # seconds
        self._ptp_alarm = None
        #
        self._state = None
        self._reactor = hololink_module.Reactor.get_reactor()
        self._hololink = None
        self._connected = set()

    def goto(self, state):
        logging.info(f"device={self._serial_number} {state=}")
        self._state = state
        for controller in list(self._connected):
            controller.device_state(state)

    def start(self):
        self.goto(Device.State.NOT_FOUND)

    def ready(self):
        logging.debug(f"ip={self._peer_ip} serial_number={self._serial_number} ready.")
        self.goto(Device.State.RUN)

    def enumerated(self, controller, enumeration_metadata):
        logging.trace(f"Received {enumeration_metadata=}")
        assert self._reactor.is_current_thread()
        now = time.monotonic()
        if controller not in self._connected:
            controller.device_state(self._state)
            self._connected.add(controller)
        # Since ECB is directed toward self._peer_ip,
        # don't progress from NOT_FOUND if we're enumerated
        # from another port first.
        peer_ip = enumeration_metadata["peer_ip"]
        primary_controller = peer_ip == self._peer_ip
        #
        if (self._state == Device.State.NOT_FOUND) and primary_controller:
            # Fetching the hololink object and calling start on it are
            # idempotent-- you can do this any number of times and only
            # the first matters-- this happens when Device is lost
            # and found again.
            self._hololink = hololink_module.Hololink.from_enumeration_metadata(
                enumeration_metadata
            )
            self._hololink.start()
            # Reset the sequence number
            get_hsb_ip_version_timeout = hololink_module.Timeout(30, 0.2)
            version = self._hololink.get_hsb_ip_version(
                get_hsb_ip_version_timeout, check_sequence=False
            )
            # Drive a reset on the device
            logging.info(
                f"ip={self._peer_ip} serial_number={self._serial_number} {version=:#x} resetting the device."
            )
            self._hololink.trigger_reset()
            self._reset_time = now
            self.goto(Device.State.RESET)
            return
        if self._state == Device.State.RESET:
            # Don't mistake enumeration messages that were
            # queued up from before we hit reset.
            dt = now - self._reset_time
            if dt < 0.2:
                return
            logging.info(
                f"ip={self._peer_ip} serial_number={self._serial_number} enumerated after reset."
            )
            # Maybe reduce the time wasted in ARP cache updates.
            self._hololink.seed_arp(enumeration_metadata)
            self._hololink.post_reset_configuration()
            self.goto(Device.State.CONNECTED)
            # Poll for PTP synchronization
            self._reset_time = now  # because PTP sync timeout uses this
            logging.info(
                f"ip={self._peer_ip} serial_number={self._serial_number} waiting for PTP sync."
            )
            self.check_ptp()

    def check_ptp(self):
        assert self._reactor.is_current_thread()
        self._ptp_alarm = None
        logging.trace(f"check_ptp {self._state=}")
        if self._state != Device.State.CONNECTED:
            return
        try:
            ptp_synchronized = self._hololink.ptp_synchronized()
        except hololink_module.TransactionError as transaction_error:
            logging.info(f"Discovered {transaction_error=}; device lost.")
            self.lost()
            return
        if ptp_synchronized:
            logging.debug(
                f"ip={self._peer_ip} serial_number={self._serial_number} PTP synchronized."
            )
            self.ready()
            return
        # Not sync'd yet.
        # Should we give up?
        now = time.monotonic()
        dt = now - self._reset_time
        if dt >= self._ptp_timeout:
            logging.error(
                f"ip={self._peer_ip} serial_number={self._serial_number} PTP synchronization timeout; ignoring."
            )
            self.ready()
            return
        # Keep polling.
        self._ptp_alarm = self._reactor.add_alarm_s(self._ptp_poll_time, self.check_ptp)

    def lost(self):
        self._reactor.cancel_alarm(self._ptp_alarm)
        self.goto(Device.State.NOT_FOUND)
        self._connected = set()


class SensorFactory:
    def __init__(self, channel_ip):
        self._channel_ip = channel_ip
        self._watchdog_timeout = 0.5  # seconds
        #
        self._reactor = hololink_module.Reactor.get_reactor()
        self._hololink_channel = None
        self._hololink = None
        self._sensor = None
        self._watchdog = None

    def configure_converter(self, converter):
        raise NotImplementedError()

    def start(self, hsb_controller):
        logging.trace(f"{self} start")
        self._hsb_controller = hsb_controller
        self._enumeration_handle = hololink_module.Enumerator.register_ip(
            self._channel_ip, self._enumerated
        )

    def stop(self):
        logging.trace(f"{self} stop")
        self._reactor.cancel_alarm(self._watchdog)
        hololink_module.Enumerator.unregister_ip(self._enumeration_handle)

    def _enumerated(self, metadata):
        logging.trace(f"{self} enumerated {metadata=}")
        assert self._reactor.is_current_thread()
        self._enumeration_metadata = metadata
        self._device = Device.get_device(metadata)
        self._device.enumerated(self, metadata)

    def device_state(self, device_state):
        logging.info(f"{self} {device_state=}")
        if device_state is Device.State.CONNECTED:
            assert self._sensor is None
            self.found()
        elif device_state is Device.State.RUN:
            assert self._sensor is not None
            self.run()
        elif device_state is Device.State.NOT_FOUND:
            # We can get here while self._sensor is None
            # if we didn't get to found when the device
            # was lost (e.g. during setup)
            self._sensor = None
            self._reactor.cancel_alarm(self._watchdog)

    def found(self):
        logging.trace(f"{self} found.")
        try:
            if self._hololink_channel is None:
                self.new_data_channel()
            self._sensor = self.new_sensor()
        except hololink_module.TransactionError as transaction_error:
            logging.info(f"Caught {transaction_error=} during configuration.")
            self._device.lost()
            return
        self._hsb_controller.found(
            self._enumeration_metadata, self._hololink_channel, self._sensor
        )

    def new_data_channel(self):
        self._hololink_channel = hololink_module.DataChannel(self._enumeration_metadata)
        self._hololink = self._hololink_channel.hololink()

    def new_sensor(self):
        raise NotImplementedError()

    def run(self):
        logging.trace(f"{self} run")
        self._watchdog = self._reactor.add_alarm_s(
            self._watchdog_timeout, self.watchdog_timeout
        )
        self._hsb_controller.run()

    def tap(self):
        logging.trace(f"{self} tap")
        self._reactor.cancel_alarm(self._watchdog)
        self._watchdog = self._reactor.add_alarm_s(
            self._watchdog_timeout, self.watchdog_timeout
        )

    def watchdog_timeout(self):
        self._reactor.cancel_alarm(self._watchdog)
        logging.info(f"{self} device watchdog timeout.")
        self._sensor = None
        self._device.lost()
        self._hsb_controller.lost()


class HsbController:
    def __init__(
        self,
        sensor_factory,
        network_receiver,
    ):
        self._sensor_factory = sensor_factory
        self._network_receiver = network_receiver
        #
        self._connected = False

    def start(self, operator):
        logging.debug("start")
        self._operator = operator
        self._network_receiver.start(operator)
        self._sensor_factory.start(self)

    def stop(self):
        logging.debug("stop")
        self._sensor_factory.stop()
        self._network_receiver.stop()

    def get_next_frame(self):
        r = self._network_receiver.get_next_frame()
        self._sensor_factory.tap()
        return r

    def found(self, metadata, hololink_channel, device):
        self._network_receiver.found(metadata, hololink_channel, device)
        self._connected = True

    def lost(self):
        self._connected = False
        self._network_receiver.lost()
        self._operator.lost()

    def run(self):
        self._network_receiver.run()

    def connected(self):
        return self._connected
