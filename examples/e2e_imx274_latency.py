# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# End-to-end ("glass-to-glass") latency measurement for a single IMX274 sensor via
# LinuxReceiverOp, using an external hardware trigger loop instead of the internal
# CaptureMetadataT timestamps linux_imx274_latency.py uses: GPIO0 ("LED_TRIGGER") is driven
# high by the FPGA's own sequencer hardware on the sensor's frame-start event; GPIO1
# ("PD_ACTIVE") is an external photodiode input that goes high once the resulting frame has
# actually rendered to the screen. The GPIO1-minus-GPIO0 delta is sensor-trigger-to-photons-on-
# screen -- it includes GPU render/composite/scanout time that CaptureMetadataT's fields (which
# stop at CsiToBayerOp's entry) can't express at all. This needs real external hardware (an
# LED+photodiode loop wired to the board's GPIO pins) to produce any data; it still displays a
# normal live Holoviz feed, so --headless structural verification works the same as any other
# port in this effort, just without real timing data to report.
#
# Python port of this repo's own previous (HSDK4-era) Python version of this app, via the HSDK5
# Python bindings. New Python bindings were needed and added as part of this port (none existed
# before): HololinkDevice.on_async_event()/sif0_frame_start_sequencer()/gpio0_sequencer()/
# gpio1_sequencer(), and the AsyncEventListener/Sequencer classes themselves
# (python/hololink/_hololink.cpp) -- AsyncEventListener needed a nanobind trampoline class since
# Python code subclasses it and overrides its virtual event-callback methods, modeled directly
# on this repo's own pre-nanobind pybind11 binding of the same class (see git history prior to
# commit 1f76a334, which removed it during the nanobind rewrite).
#
#   LinuxReceiverOp -> CsiToBayerOp -> ImageProcessorOp -> BayerDemosaicOp ->
#     MultiEventAcknowledgmentOp -> HolovizOp
#
# MultiEventAcknowledgmentOp passes tensors through unchanged; every 50 frames it pulses the
# FPGA's CTRL_EVT_CLEAR register so the async-event latch doesn't stall. It needs the live
# HololinkDevice object, which can't flow through graph.op()'s kwargs: they carry only plain
# values (confirmed: AUTHORING_PARAMETER_TYPE_MISMATCH on a live-object kwarg), and constructor
# handles -- how the C++-registered receivers take their channel -- exist only for registry
# rows, not for a pure-Python operator like this one. Both it and its caller are under our
# control, so this uses a plain Python-side stash instead -- see _HololinkDeviceStash below.
#
# --frame-limit is the receiver's frame_limit parameter: the receiver ends the run after that
# many frames. HSDK5 has no CountCondition; this replaces the CountCondition of the old app.
#
# Usage:
#   e2e_imx274_latency.py [OPTIONS]
#
# Options:
#   --hololink IP                 IP address of Hololink board (default: 192.168.0.2)
#   --camera-mode N                IMX274 mode (default: 0 = 3840x2160@60fps)
#   --expander-configuration N    I2C expander configuration (0 or 1, default: 0)
#   --pattern N                   Configure a test pattern (0-11) instead of live video
#   --frame-limit N               Number of frames to capture (default: unlimited)
#   --headless                    Run without opening a window (still renders, presents nowhere)
#   --fullscreen                  Run in fullscreen mode
#   --log-level N                 Python logging level (default: 20 = INFO)

import argparse
import logging
import sys

import holoscan as hs
import holoscan_vision as hv
import holoscan_visualization as hz  # noqa: F401 -- registers "holoscan.visualization.HolovizOp"
from cuda.bindings import driver as cuda_driver

import hololink as hl

_BAYER_FORMAT_NAMES = ["bggr", "rggb", "gbrg", "grbg"]
_BAYER_FORMAT_VALUES = {name: index for index, name in enumerate(_BAYER_FORMAT_NAMES)}

CTRL_EVT_CLEAR = 0x0000020C

# GPIO register addresses used by setup_sequencer()/reset_gpio() below.
_GPIO_DIRECTION_REG_BASE = 0x0000002C
_GPIO_DELAYED_IO_OUTPUT_REG = 0x70000014
_GPIO_MODULE_CLEAR_REG = 0x7000080C
_GPIO_MODULE_ENABLE_REG = 0x700008F0
_GPIO_DELAY_REG = 0x70000804
_GPIO_DELAYED_SET_REG = 0x70000814


def _cuda_check(result):
    err = result[0]
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        _, description = cuda_driver.cuGetErrorString(err)
        raise RuntimeError(f"CUDA error: {description}")
    if len(result) == 2:
        return result[1]
    return result[1:]


class AsyncEventPrinter(hl.AsyncEventListener):
    """Collects async event timing and reports statistics at shutdown.

    Tracks timing between GPIO0 (start) and GPIO1 (end) events; for each pair records
    (end - start) and reports min/avg/max once report() is called.
    """

    def __init__(self):
        super().__init__()
        self._last_start_ts = None
        self._start_ts = []  # list of start timestamps (in nanoseconds)
        self._end_ts = []  # list of end timestamps (in nanoseconds)

    def on_gpio0_event(self, state, timestamp_s, timestamp_ns):
        # Collect timestamp only when GPIO0 is active and GPIO1 is not active.
        if state == 0x40000:
            self._last_start_ts = float(timestamp_s) * 1e9 + float(timestamp_ns)

    def on_gpio1_event(self, state, timestamp_s, timestamp_ns):
        # Collect timestamp when both GPIO0 and GPIO1 are active.
        if state == 0xC0000:
            ts = float(timestamp_s) * 1e9 + float(timestamp_ns)
            if self._last_start_ts is not None:
                self._start_ts.append(self._last_start_ts)
                self._end_ts.append(ts)

    def report(self):
        if not self._start_ts or not self._end_ts:
            logging.error("No async event timing data collected.")
            return

        n_pairs = min(len(self._start_ts), len(self._end_ts))
        if n_pairs == 0:
            logging.error("No complete timing pairs collected.")
            return

        deltas = [self._end_ts[i] - self._start_ts[i] for i in range(n_pairs)]

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


class _HololinkDeviceStash:
    """Plain Python-side stash for passing the live HololinkDevice into
    MultiEventAcknowledgmentOp -- see the file-level comment for why."""

    _devices = {}
    _next_token = 0

    @classmethod
    def stash(cls, device):
        token = cls._next_token
        cls._next_token += 1
        cls._devices[token] = device
        return token

    @classmethod
    def redeem(cls, token):
        return cls._devices[token]

    @classmethod
    def release(cls, token):
        cls._devices.pop(token, None)


class MultiEventAcknowledgmentOp(hs.Operator):
    """Counts displayed frames on a side branch; every frame_threshold frames, pulses
    CTRL_EVT_CLEAR so the FPGA's async-event latch doesn't stall. It has no output: a
    Python operator cannot re-emit the demosaiced device Image (BufferError: kNotSupported),
    so HolovizOp takes BayerDemosaicOp's output directly."""

    input = hs.Input("input", hs.schema.Image)

    hololink_device_token = hs.Param(0)
    frame_threshold = hs.Param(50)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._frame_count = 0

    def setup(self, spec):
        spec.input(self.input, queue_depth=4)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, context):
        sample = self.input.receive()
        if sample.is_missing:
            return

        self._frame_count += 1
        if self._frame_count >= self.frame_threshold:
            logging.debug(f"Acknowledging events after {self._frame_count} frames")
            device = _HololinkDeviceStash.redeem(self.hololink_device_token)
            device.write_uint32(CTRL_EVT_CLEAR, 0xFFFFFFFF)  # Set flag
            device.write_uint32(CTRL_EVT_CLEAR, 0x0)  # Clear flag
            self._frame_count = 0


class DiscardImageOp(hs.Operator):
    """With --headless, HolovizOp renders into a CUDA image on its output instead of a window;
    the output must be consumed, and this drops each rendered frame."""

    input = hs.Input("input", hs.schema.Image)

    def setup(self, spec):
        spec.input(self.input, queue_depth=1)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, context):
        self.input.receive_data()


def setup_sequencer(hololink_dev, frame_start_delay):
    """Arm the FPGA's hardware GPIO trigger loop: GPIO0 pulses on frame-start (after
    frame_start_delay sequencer ticks), GPIO1 is the external photodiode input."""
    hololink_dev.write_uint32(
        _GPIO_DELAYED_IO_OUTPUT_REG, 0x00000100
    )  # Enable DELAYED_IO output on GPIO

    # GPIO Pin 0 will be set HIGH on Frame Start event.
    sequencer = hololink_dev.sif0_frame_start_sequencer()
    sequencer.write_uint32(_GPIO_MODULE_CLEAR_REG, 0x1)  # Clear all
    sequencer.write_uint32(_GPIO_MODULE_ENABLE_REG, 0x1)  # Enable the whole module
    sequencer.write_uint32(_GPIO_DELAY_REG, frame_start_delay)
    sequencer.write_uint32(_GPIO_DELAYED_SET_REG, 0x1)  # Delayed set
    sequencer.write_uint32(_GPIO_DIRECTION_REG_BASE, 0x2)  # Set pin 1 as Input
    sequencer.enable()

    # Setting up sequencer to retrieve timestamp on LED_TRIGGER aka GPIO0.
    sequencer = hololink_dev.gpio0_sequencer()
    sequencer.enable()

    # When PD_ACTIVE aka GPIO1 goes HIGH, timestamp is retrieved to calculate end-to-end
    # measurements. Also LED_TRIGGER aka GPIO0 is set LOW to ACK that PD_ACTIVE went HIGH.
    sequencer = hololink_dev.gpio1_sequencer()
    sequencer.write_uint32(_GPIO_MODULE_ENABLE_REG, 0x1)  # Enable the whole module
    sequencer.write_uint32(_GPIO_MODULE_CLEAR_REG, 0x1)  # Clear all
    sequencer.enable()


def reset_gpio(hololink_dev):
    hololink_dev.write_uint32(
        _GPIO_DELAYED_IO_OUTPUT_REG, 0x00000000
    )  # Disable DELAYED_IO


def run(
    hololink_ip,
    camera_mode,
    expander_configuration,
    pattern,
    frame_limit,
    headless,
    fullscreen,
):
    cuda_device_ordinal = 0

    print("Initializing CUDA...")
    _cuda_check(cuda_driver.cuInit(0))
    cu_device = _cuda_check(cuda_driver.cuDeviceGet(cuda_device_ordinal))
    cu_context = _cuda_check(cuda_driver.cuDevicePrimaryCtxRetain(cu_device))
    hololink_dev = None
    async_printer = None
    hololink_device_token = None
    gpio_armed = False
    try:
        cuda_driver.cuCtxSetCurrent(cu_context)

        print(f"Enumerating HSB board at {hololink_ip}...")
        metadata = hl.find_channel(hololink_ip)
        channel = hl.DataChannel(metadata)

        hololink_dev = channel.hololink()
        # Register async event printer so we collect interrupt/timestamp info from Python.
        async_printer = AsyncEventPrinter()
        hololink_dev.on_async_event(async_printer)
        hololink_dev.start()
        hololink_dev.reset()

        camera = hl.NativeImx274Sensor(
            channel, expander_configuration=expander_configuration
        )
        camera.setup_clock()
        camera.configure(camera_mode)
        camera.set_digital_gain_reg(0x4)
        if pattern is not None:
            camera.test_pattern(pattern)

        pixel_format = camera.get_pixel_format()
        bayer_format = camera.get_bayer_format()
        frame_width = camera.get_width()
        frame_height = camera.get_height()

        geometry = hl.resolve_generic_geometry(camera)

        hololink_device_token = _HololinkDeviceStash.stash(hololink_dev)

        # Build the graph.
        graph = hs.Graph("e2e_imx274_latency")

        rx = graph.op(
            hl.operators.LinuxReceiverOp,
            "rx",
            data_channel=channel,
            frame_context=int(cu_context),
            device_start=camera.start,
            device_stop=camera.stop,
            frame_size=(
                geometry["start_byte"]
                + geometry["bytes_per_line"] * geometry["pixel_height"]
                + geometry["trailing_bytes"]
            ),
            frame_limit=frame_limit or 0,
            frame_limit_ends_run=True,
        )

        csi = graph.op(
            hl.operators.CsiToBayerOp,
            "csi_to_bayer",
            cuda_device_ordinal=cuda_device_ordinal,
            geometry=geometry,
            bayer_format=bayer_format,
        )
        graph.add_flow(rx.tensor, csi.in_tensor, queue_depth=4)
        graph.add_flow(rx.metadata, csi.in_metadata, queue_depth=4)

        image_processor = graph.op(
            hl.operators.ImageProcessorOp,
            "image_processor",
            pixel_format=pixel_format,
            pixel_width=geometry["pixel_width"],
            pixel_height=geometry["pixel_height"],
            bayer_format=_BAYER_FORMAT_VALUES[bayer_format],
            # Optical black value for imx274 is 50 (matches this app's own HSDK4-era default).
            optical_black=50,
            cuda_device_ordinal=cuda_device_ordinal,
        )
        graph.add_flow(csi.tensor, image_processor.in_tensor, queue_depth=4)
        graph.add_flow(csi.metadata, image_processor.in_metadata, queue_depth=4)

        # image_processor.metadata is not connected: nothing uses it, and an
        # output that is not connected has no queue that can fill.

        demosaic = graph.op(
            hv.operators.BayerDemosaicOp,
            "bayer_demosaic",
            bayer_grid_pos=bayer_format,
            generate_alpha=True,
            alpha_value=65535,
        )
        graph.add_flow(image_processor.tensor, demosaic.input, queue_depth=4)

        event_ack = graph.op(
            MultiEventAcknowledgmentOp,
            "event_ack",
            hololink_device_token=hololink_device_token,
            frame_threshold=50,  # Acknowledge every 50 frames
        )
        # Drop-oldest: BayerDemosaicOp fails the run when one of its queues is full, and a
        # control-plane retry in event_ack's register writes takes 100 ms.
        graph.add_flow(
            demosaic.output,
            event_ack.input,
            queue_depth=4,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
        )

        if headless:
            # Renders into a CUDA image on HolovizOp's output (consumed by DiscardImageOp). The
            # CUDA sink has no framebuffer_srgb: its sRGB format encodes on write instead.
            sink = {
                "kind": "cuda",
                "width": frame_width,
                "height": frame_height,
                "format": "rgba8_srgb",
            }
        else:
            sink = {
                "kind": "window",
                "framebuffer_srgb": True,
                "title": "E2E IMX274 Latency",
                "width": frame_width,
                "height": frame_height,
                "fullscreen": fullscreen,
                "close_on_escape": True,
            }
        viz = graph.op(
            "holoscan.visualization.HolovizOp",
            "holoviz",
            layers=[
                {
                    "port_name": "video",
                    "kind": "image",
                    "format": "rgba16_unorm",
                    "max_width": frame_width,
                    "max_height": frame_height,
                }
            ],
            sink=sink,
            # One frame per received frame.
            trigger={"kind": "layer"},
        )
        graph.add_flow(
            demosaic.output,
            viz.video,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
            queue_depth=1,
        )
        if headless:
            discard = graph.op(DiscardImageOp, "discard")
            graph.add_flow(
                viz.output,
                discard.input,
                queue_policy=hs.QueuePolicy.DROP_OLDEST,
                queue_depth=1,
            )

        clock_key = graph.add_clock(hs.RealtimeClock, "clock")
        graph.set_default_clock(clock_key)

        options = hs.CompileOptions()
        for operator_path in ("rx", "csi_to_bayer", "image_processor"):
            options.deployment.bind_tensor_output_device(
                operator_path=operator_path,
                output_port="tensor",
                device=hs.execution.DeviceId(cuda_device_ordinal),
            )
        options.deployment.bind_tensor_output_device(
            operator_path="bayer_demosaic",
            output_port="output",
            device=hs.execution.DeviceId(cuda_device_ordinal),
        )

        if headless:
            options.deployment.bind_tensor_output_device(
                operator_path="holoviz",
                output_port="output",
                device=hs.execution.DeviceId(cuda_device_ordinal),
            )

        plan = hs.compile(graph, options)
        if not plan.ok:
            print("Graph compilation failed:", file=sys.stderr)
            for diag in plan.diagnostics:
                print(f"  {diag}", file=sys.stderr)
            return 1

        # Setup the sequencer for all the events for latency measurements.
        gpio_armed = True
        setup_sequencer(hololink_dev, frame_start_delay=0x0)  # 0ms delay

        print(
            f"Capturing mode={camera_mode} {frame_width}x{frame_height} -- running (Ctrl+C to stop)..."
        )

        try:
            with hs.run_async(plan) as session:
                # With --frame-limit, the receiver ends the run after frame_limit frames.
                session.wait()
        except KeyboardInterrupt:
            pass

        # Print async event timing statistics.
        async_printer.report()

        print("Done.")
        return 0
    finally:
        # Each step is attempted even if an earlier one fails. Hololink instances are cached
        # for the life of the process, so the listener (and the stashed device) would
        # otherwise only be freed at interpreter exit.
        try:
            if hololink_dev is not None:
                try:
                    if gpio_armed:
                        # Reset the GPIO to default configuration once the measurements are done.
                        reset_gpio(hololink_dev)
                finally:
                    try:
                        hololink_dev.stop()
                    finally:
                        if async_printer is not None:
                            hololink_dev.remove_async_event(async_printer)
        finally:
            try:
                if hololink_device_token is not None:
                    _HololinkDeviceStash.release(hololink_device_token)
            finally:
                _cuda_check(cuda_driver.cuDevicePrimaryCtxRelease(cu_device))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera-mode",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="IMX274 mode (default: 0 = 3840x2160@60fps)",
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
    parser.add_argument(
        "--hololink", default="192.168.0.2", help="IP address of Hololink board"
    )
    parser.add_argument(
        "--log-level", type=int, default=20, help="Python logging level to display"
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

    logging.basicConfig(level=args.log_level)

    return run(
        args.hololink,
        args.camera_mode,
        args.expander_configuration,
        args.pattern,
        args.frame_limit,
        args.headless,
        args.fullscreen,
    )


if __name__ == "__main__":
    sys.exit(main())
