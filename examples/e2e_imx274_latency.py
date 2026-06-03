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

# See README.md for detailed information.

import argparse
import ctypes
import logging
import os

import cuda.bindings.driver as cuda
import holoscan

import hololink as hololink_module
from hololink import AsyncEventListener


class AsyncEventPrinter(AsyncEventListener):
    """Collects async event timing and reports statistics at shutdown.

    We track timing between GPIO0 (start) and GPIO1 (end) events.
    For each pair we record (end - start) and compute min/avg/max.
    """

    def __init__(self):
        super().__init__()
        self._last_start_ts = None
        self._start_ts = []  # list of start timestamps (in nanoseconds)
        self._end_ts = []  # list of end timestamps (in nanoseconds)

    def on_gpio0_event(self, state, timestamp_s, timestamp_ns):
        """Called when GPIO0 state changes. We capture the timestamp on rising edge."""

        # Collect timestamp only when GPIO0 is active and GPIO1 is not active
        if state == 0x40000:
            # record start timestamp (in nanoseconds)
            ts = float(timestamp_s) * 1e9 + float(timestamp_ns)
            self._last_start_ts = ts

    def on_gpio1_event(self, state, timestamp_s, timestamp_ns):
        """Called when GPIO1 state changes. We capture the timestamp on rising edge."""

        # Collect timestamp when both GPIO0 and GPIO1 are active
        if state == 0xC0000:
            # record end timestamp (in nanoseconds)
            ts = float(timestamp_s) * 1e9 + float(timestamp_ns)
            if self._last_start_ts is not None:
                self._start_ts.append(self._last_start_ts)
                self._end_ts.append(ts)

    def report(self):
        # Calculate deltas from paired start and end timestamps
        if not self._start_ts or not self._end_ts:
            logging.error("No async event timing data collected.")
            return

        # Pair up start and end timestamps
        n_pairs = min(len(self._start_ts), len(self._end_ts))
        if n_pairs == 0:
            logging.error("No complete timing pairs collected.")
            return

        deltas = []
        for i in range(n_pairs):
            delta = self._end_ts[i] - self._start_ts[i]
            deltas.append(delta)

        if not deltas:
            logging.error("No deltas calculated.")
            return

        n_total = len(deltas)
        # Discard first 10 and last 1 pair, if possible.
        if n_total <= 11:
            logging.error("Not enough async event timing data.")
            return

        deltas_filtered = deltas[10:-1]
        n = len(deltas_filtered)
        min_delta = min(deltas_filtered)
        max_delta = max(deltas_filtered)
        avg_delta = sum(deltas_filtered) / n

        # Print in milliseconds for readability (deltas are in nanoseconds).
        logging.info(
            f"Async event timing over {n} pairs: "
            f"min={min_delta / 1e6:.3f} ms, "
            f"avg={avg_delta / 1e6:.3f} ms, "
            f"max={max_delta / 1e6:.3f} ms"
        )


class MultiEventAcknowledgmentOperator(holoscan.core.Operator):
    """
    Operator that acknowledges events after a specified frame count.
    """

    def __init__(self, fragment, hololink, frame_threshold=50, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self._hololink = hololink
        self._frame_threshold = frame_threshold
        self._frame_count = 0

    def setup(self, spec):
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, _context):
        in_message = op_input.receive("input")

        self._frame_count += 1

        CTRL_EVT_CLEAR = 0x0000020C

        if self._frame_count >= self._frame_threshold:
            logging.debug(f"Acknowledging events after {self._frame_count} frames")
            self._hololink.write_uint32(CTRL_EVT_CLEAR, 0xFFFFFFFF)  # Set flag
            self._hololink.write_uint32(CTRL_EVT_CLEAR, 0x0)  # Clear flag
            # Reset frame counter for next acknowledgment cycle
            self._frame_count = 0

        op_output.emit(in_message, "output")


class HoloscanApplication(holoscan.core.Application):
    def __init__(
        self,
        headless,
        fullscreen,
        cuda_context,
        cuda_device_ordinal,
        hololink_channel,
        camera,
        camera_mode,
        frame_limit,
        channel_metadata,
    ):
        logging.info("__init__")
        super().__init__()
        self._headless = headless
        self._fullscreen = fullscreen
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._hololink_channel = hololink_channel
        self._camera = camera
        self._camera_mode = camera_mode
        self._frame_limit = frame_limit

        self._channel_metadata = channel_metadata
        self._hololink = hololink_channel.hololink()

    def setup_sequencer(self):
        # GPIO register addresses
        GPIO_DIRECTION_REG_BASE = 0x0000002C
        # GPIO module control registers
        GPIO_DELAYED_IO_OUTPUT_REG = 0x70000014
        GPIO_MODULE_CLEAR_REG = 0x7000080C
        GPIO_MODULE_ENABLE_REG = 0x700008F0
        GPIO_DELAY_REG = 0x70000804
        GPIO_DELAYED_SET_REG = 0x70000814

        self._hololink.write_uint32(
            GPIO_DELAYED_IO_OUTPUT_REG, 0x00000100
        )  # Enable DELAYED_IO output on GPIO

        # GPIO Pin 0 will be set HIGH on Frame Start event
        sequencer = self._hololink.sif0_frame_start_sequencer()
        sequencer.write_uint32(GPIO_MODULE_CLEAR_REG, 0x1)  # Clear all
        sequencer.write_uint32(
            GPIO_MODULE_ENABLE_REG, 0x1
        )  # To enable the whole module
        sequencer.write_uint32(GPIO_DELAY_REG, 0x0)  # 0ms delay
        sequencer.write_uint32(GPIO_DELAYED_SET_REG, 0x1)  # Delayed set
        sequencer.write_uint32(GPIO_DIRECTION_REG_BASE, 0x2)  # Set pin 1 as Input
        sequencer.enable()

        # Setting up sequencer to retrieve timestamp on LED_TRIGGER aka GPIO0
        sequencer = self._hololink.gpio0_sequencer()
        sequencer.enable()

        # When the PD_ACTIVE aka GPIO1 goes HIGH, timestamp is retrieved to calculate end to end measurements.
        # Also LED_TRIGGER aka GPIO0 is set to LOW to ACK that PD_ACTIVE went HIGH.
        sequencer = self._hololink.gpio1_sequencer()
        sequencer.write_uint32(
            GPIO_MODULE_ENABLE_REG, 0x1
        )  # To enable the whole module
        sequencer.write_uint32(GPIO_MODULE_CLEAR_REG, 0x1)  # To clear all
        sequencer.enable()

    def reset_gpio(self):
        self._hololink.write_uint32(
            0x70000014, 0x00000000
        )  # Disable DELAYED_IO output on GPIO

    def compose(self):
        logging.info("compose")
        if self._frame_limit:
            self._count = holoscan.conditions.CountCondition(
                self,
                name="count",
                count=self._frame_limit,
            )
            condition = self._count
        else:
            self._ok = holoscan.conditions.BooleanCondition(
                self, name="ok", enable_tick=True
            )
            condition = self._ok
        self._camera.set_mode(self._camera_mode)

        csi_to_bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="pool",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=self._camera._width
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera._height,
            num_blocks=4,
        )
        csi_to_bayer_operator = hololink_module.operators.CsiToBayerOp(
            self,
            name="csi_to_bayer",
            allocator=csi_to_bayer_pool,
            cuda_device_ordinal=self._cuda_device_ordinal,
        )
        self._camera.configure_converter(csi_to_bayer_operator)

        frame_size = csi_to_bayer_operator.get_csi_length()
        frame_context = self._cuda_context
        receiver_operator = hololink_module.operators.LinuxReceiverOperator(
            self,
            condition,
            name="receiver",
            frame_size=frame_size,
            frame_context=frame_context,
            hololink_channel=self._hololink_channel,
            device=self._camera,
        )

        pixel_format = self._camera.pixel_format()
        bayer_format = self._camera.bayer_format()
        image_processor_operator = hololink_module.operators.ImageProcessorOp(
            self,
            name="image_processor",
            # Optical black value for imx274 is 50
            optical_black=50,
            bayer_format=bayer_format.value,
            pixel_format=pixel_format.value,
        )

        rgba_components_per_pixel = 4
        bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="pool",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=self._camera._width
            * rgba_components_per_pixel
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera._height,
            num_blocks=4,
        )
        demosaic = holoscan.operators.BayerDemosaicOp(
            self,
            name="demosaic",
            pool=bayer_pool,
            generate_alpha=True,
            alpha_value=65535,
            bayer_grid_pos=bayer_format.value,
            interpolation_mode=0,
        )
        event_ack_operator = MultiEventAcknowledgmentOperator(
            self,
            name="event_ack",
            hololink=self._hololink,
            frame_threshold=50,  # Acknowledge every 50 frames
        )
        visualizer = holoscan.operators.HolovizOp(
            self,
            name="holoviz",
            fullscreen=self._fullscreen,
            headless=self._headless,
            framebuffer_srgb=True,
        )

        #
        self.add_flow(receiver_operator, csi_to_bayer_operator, {("output", "input")})
        self.add_flow(
            csi_to_bayer_operator, image_processor_operator, {("output", "input")}
        )
        self.add_flow(image_processor_operator, demosaic, {("output", "receiver")})
        self.add_flow(demosaic, event_ack_operator, {("transmitter", "input")})
        self.add_flow(event_ack_operator, visualizer, {("output", "receivers")})


def main():
    parser = argparse.ArgumentParser()
    modes = hololink_module.sensors.imx274.imx274_mode.Imx274_Mode
    mode_choices = [mode.value for mode in modes]
    mode_help = " ".join([f"{mode.value}:{mode.name}" for mode in modes])
    parser.add_argument(
        "--camera-mode",
        type=int,
        choices=mode_choices,
        default=mode_choices[0],
        help=mode_help,
    )
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument(
        "--fullscreen", action="store_true", help="Run in fullscreen mode"
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Exit after receiving this many frames",
    )
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
        "--expander-configuration",
        type=int,
        default=0,
        choices=(0, 1),
        help="I2C Expander configuration",
    )
    parser.add_argument(
        "--pattern",
        type=int,
        choices=range(12),
        help="Configure to display a test pattern.",
    )
    args = parser.parse_args()
    hololink_module.logging_level(args.log_level)
    logging.info("Initializing.")
    # Get a handle to the GPU
    (cu_result,) = cuda.cuInit(0)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_device_ordinal = 0
    cu_result, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_result, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    # Get a handle to the Hololink device
    channel_metadata = hololink_module.Enumerator.find_channel(channel_ip=args.hololink)
    hololink_channel = hololink_module.DataChannel(channel_metadata)
    # Get a handle to the camera
    camera = hololink_module.sensors.imx274.dual_imx274.Imx274Cam(
        hololink_channel, expander_configuration=args.expander_configuration
    )
    camera_mode = hololink_module.sensors.imx274.imx274_mode.Imx274_Mode(
        args.camera_mode
    )

    # Run it.
    hololink = hololink_channel.hololink()
    # Register async event printer so we collect interrupt/timestamp info from Python.
    async_printer = AsyncEventPrinter()
    hololink.on_async_event(async_printer)
    hololink.start()
    hololink.reset()

    # Set up the application
    application = HoloscanApplication(
        args.headless,
        args.fullscreen,
        cu_context,
        cu_device_ordinal,
        hololink_channel,
        camera,
        camera_mode,
        args.frame_limit,
        channel_metadata,
    )
    application.config(args.configuration)

    camera.setup_clock()
    camera.configure(camera_mode)
    camera.set_digital_gain_reg(0x4)
    if args.pattern is not None:
        camera.test_pattern(args.pattern)

    # Setup the sequencer for all the events for latency measurements
    application.setup_sequencer()
    # Run the app and capture latency measurements
    application.run()
    # Reset the GPIO to default configuration once the measurements are done
    application.reset_gpio()

    hololink.stop()

    # Print async event timing statistics.
    async_printer.report()

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS


if __name__ == "__main__":
    main()
