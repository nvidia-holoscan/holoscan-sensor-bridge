# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# IMX274 mid-capture reconnection test on the hololink_module V1 surface.
#
# A capture pipeline runs while the board is reset mid-stream; the module
# reconnection framework (HsbControllerOp -> HsbController -> SensorFactory +
# NetworkReceiver) tears the device state down, invalidates the board's V1
# handles, rediscovers it on re-announcement, and resumes delivering frames.
#
# Structure:
#   * Discovery is Adapter.wait_for_channel(peer_ip, timeout); the reconnection
#     loop is Adapter.register_ip, owned inside the C++ SensorFactory.
#   * The application subclasses hololink_module.operators.SensorFactory
#     (new_sensor) and .SensorDevice (stop_sensor / fallback_frame). new_sensor
#     builds a fresh Imx274Cam(metadata) on every (re)connect, which
#     re-get_service()s the board's handles — the reconnection "re-fetch" is
#     implicit in that rebuild.
#   * The reset is triggered through the versioned supplement:
#     hololink_module.hsb_lite.HsbLiteInterfaceV1.get_service(metadata)
#         .trigger_reset()
#     deferred onto the reactor.
#   * The data plane is selected by a NetworkReceiverFactory
#     (make_roce_network_receiver_factory).

import contextlib
import ctypes
import dataclasses
import logging
import os
import threading
import weakref

import applications
import cupy as cp
import hololink_module.hsb_lite
import hololink_module.operators
import hololink_module.sensors.imx274 as imx274
import holoscan
import numpy as np
import operators
import PIL.Image
import pytest
import utils
from hololink_module.sensors import csi

import hololink_module

# Seconds to wait for each bootp announcement before giving up.
DISCOVERY_TIMEOUT_S = 30.0

# Board bring-up (start()/reset()) runs once per board per (re)connect. The two
# stereo factories share a board's control plane, so the first to reach
# new_sensor brings it up and the second sees it already up. A reconnect yields
# a fresh HololinkInterfaceV1 instance (device_lost() invalidated the old one),
# so identity membership naturally resets — no explicit clearing needed. Weak so
# the invalidated instance drops out once the old camera is replaced.
_board_lock = threading.Lock()
_boards_up = weakref.WeakSet()


@dataclasses.dataclass
class ComputedCrcRecord:
    computed_crc: int


class Imx274SensorDevice(hololink_module.operators.SensorDevice):
    """Wraps one armed Imx274Cam for the reconnection controller.

    Built already-configured+streaming by Imx274SensorFactory.new_sensor;
    replaced by a fresh one on reconnect.
    """

    def __init__(self, camera):
        super().__init__()
        self._camera = camera

    def stop_sensor(self):
        # Best-effort: on loss the device is usually unreachable, so a
        # control-plane write here may time out. Swallow so teardown still
        # proceeds to device_lost().
        try:
            self._camera.stop()
        except Exception:
            logging.info("stop_sensor: camera.stop() failed (device likely gone).")
        return hololink_module.HOLOLINK_MODULE_OK


class Imx274SensorFactory(hololink_module.operators.SensorFactory):
    """Sensor-side reconnection policy for an IMX274 leg.

    The C++ base owns the watchdog, the register_ip loop, and the
    connect/disconnect callbacks into HsbController; this override supplies
    new_sensor, which brings the board up and arms the camera on every
    (re)connect.
    """

    def __init__(self, camera_factory, instance, camera_mode, pattern):
        super().__init__()
        self._camera_factory = camera_factory
        self._instance = instance
        self._camera_mode = camera_mode
        self._pattern = pattern
        self._sensor_count = 0
        self._fallback_ptr = 0
        self._fallback_size = 0
        self._sensor = None  # keeps the current SensorDevice's Python side alive
        if camera_mode == imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS:
            self._height, self._width = 1080, 1920
            self._pixel_format = csi.PixelFormat.RAW_10
            self._bayer_format = csi.BayerFormat.RGGB
        else:
            raise RuntimeError(f"Unsupported {camera_mode=}")

    # -- geometry, used by compose() to size buffers and the fallback image --

    def image_size(self):
        return self._height, self._width

    def image_format(self):
        return self._pixel_format, self._bayer_format

    def configure_converter(self, converter):
        # Trains either the real CsiToBayerOp or the fallback CsiImage to this
        # sensor's geometry, computing the same offsets
        # Imx274Cam.configure_converter would but without a live camera (no
        # device handles exist yet at compose time). The converter takes the
        # module csi.PixelFormat enumerator value.
        pixel_format = self._pixel_format.value
        start_byte = converter.receiver_start_byte()
        transmitted_line_bytes = converter.transmitted_line_bytes(
            pixel_format, self._width
        )
        received_line_bytes = converter.received_line_bytes(transmitted_line_bytes)
        # 175 bytes of metadata precede the image data.
        start_byte += converter.received_line_bytes(175)
        if self._pixel_format == csi.PixelFormat.RAW_10:
            # 8 lines of optical black before real image data
            start_byte += received_line_bytes * 8
        elif self._pixel_format == csi.PixelFormat.RAW_12:
            # 16 lines of optical black before real image data
            start_byte += received_line_bytes * 16
        else:
            raise Exception(f"Incorrect pixel format={self._pixel_format} for IMX274.")
        converter.configure(
            start_byte,
            received_line_bytes,
            self._width,
            self._height,
            pixel_format,
        )

    def set_fallback_frame(self, ptr, size):
        self._fallback_ptr = ptr
        self._fallback_size = size

    def fallback_frame(self):
        # The test image (SMPTE bars) shown when no live frame is available —
        # at startup before the first connect and during an outage. A device
        # pointer built up front; touches no live device handles.
        return (self._fallback_ptr, self._fallback_size)

    # -- the reconnection override --

    def new_sensor(self, metadata):
        # Called on every (re)connect (reactor thread). Building a fresh
        # Imx274Cam(metadata) re-get_service()s the board's HololinkInterfaceV1
        # / OscillatorInterfaceV1 / I2c handles, so the camera resyncs with the
        # post-reset module state automatically.
        #
        # Bring-up touches the control plane (hololink.start()/reset(),
        # camera.configure()), which fails if the board is gone or resetting —
        # e.g. the reconnect-during-configuration tests reset the board mid
        # camera.configure(), so its register writes NAK. Report that by
        # returning None (the framework's "bring-up failed, retry on
        # re-announce" signal) rather than letting the exception escape the
        # override, where the trampoline would discard it as an unraisable
        # exception (a noisy warning / strict-CI failure). invalidate_board()
        # then clears the board so the retry rebuilds against fresh handles.
        try:
            hololink = hololink_module.HololinkInterfaceV1.get_service(metadata)
            with _board_lock:
                if hololink not in _boards_up:
                    hololink.start()
                    hololink.reset()
                    _boards_up.add(hololink)
            imx274.Imx274Cam.use_expander_configuration(metadata, self._instance)
            camera = self._camera_factory(metadata)
            camera.configure(self._camera_mode)
            camera.set_digital_gain_reg(0x4)
            camera.test_pattern(self._pattern)
            camera.start()
        except Exception as e:
            logging.info(f"new_sensor: bring-up failed ({e}); awaiting re-announce.")
            return None
        self._sensor_count += 1
        logging.info(f"new_sensor: {self._sensor_count=}")
        # Hold a Python reference: the C++ SensorFactory owns the returned
        # SensorDevice through a shared_ptr, but stock pybind11 doesn't keep the
        # Python subclass alive from a C++-only shared_ptr, so without this the
        # object is garbage-collected and its stop_sensor override vanishes
        # (pure-virtual call). Replaced each (re)connect.
        self._sensor = Imx274SensorDevice(camera)
        return self._sensor


class InstrumentedImx274CamContext:
    """Triggers a board reset partway through camera configuration.

    Counts set_register calls across every camera the factory builds and, on
    the Nth call, fires the HSB-Lite trigger_reset() for the board being
    configured — so the device is lost *mid-configuration* rather than during
    steady-state capture. The `==` check fires exactly once: the count only
    grows as cameras are rebuilt on reconnect, so a later (successful)
    configuration never re-triggers it.

    This exercises the control-plane loss path: the reset lands while
    new_sensor is still writing sensor registers, so the remaining writes time
    out, new_sensor fails, and SensorFactory.invalidate_board() clears the
    board before the retry rebuilds it against fresh handles.
    """

    def __init__(self, metadata, set_registers_trigger):
        self._metadata = metadata
        self._set_registers_trigger = set_registers_trigger
        self._set_register_calls = 0
        self._reset_triggered = False

    def set_register(self, camera, register, value):
        self._set_register_calls += 1
        if self._set_register_calls == self._set_registers_trigger:
            logging.info("Triggering reset during configuration.")
            hololink_module.hsb_lite.HsbLiteInterfaceV1.get_service(
                self._metadata
            ).trigger_reset()
            self._reset_triggered = True


class InstrumentedImx274Cam(imx274.Imx274Cam):
    """Imx274Cam that reports each set_register to an instrumentation context."""

    def __init__(self, *args, context=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._context = context

    def set_register(self, register, value):
        self._context.set_register(self, register, value)
        super().set_register(register, value)


# CSI-frame synthesis for the fallback test image: render an image file into a
# sensor-format CSI frame (a device-side cupy array), using the module csi
# enums. Only the IMX274 RAW_10 / RGGB path used here is implemented.


def _encode_raw_10_bayer_image(bayer_image):
    # RAW10 (IMX274 spec): for each four 10-bit values c0..c3, emit their high
    # 8 bits, then a fifth byte packing the four low-2-bit remainders.
    bayer_height, bayer_width = bayer_image.shape
    ri = bayer_image.ravel()
    upper_byte = (ri >> 2).astype(np.uint8)
    lower_byte = (ri & 0x3).astype(np.uint8)
    combined_lower_bytes = (
        lower_byte[::4]
        | (lower_byte[1::4] << 2)
        | (lower_byte[2::4] << 4)
        | (lower_byte[3::4] << 6)
    )
    raw_10 = np.stack(
        [
            upper_byte[::4],
            upper_byte[1::4],
            upper_byte[2::4],
            upper_byte[3::4],
            combined_lower_bytes,
        ],
        axis=1,
    )
    return raw_10.reshape(bayer_height, bayer_width * 5 // 4)


def _encode_bayer_image(rgb_image, bayer_format):
    sr = rgb_image[:, :, 0]
    sg = rgb_image[:, :, 1]
    sb = rgb_image[:, :, 2]
    dtype = cp.uint16
    height, width, _ = rgb_image.shape
    bayer_height, bayer_width = height, width
    if bayer_format != csi.BayerFormat.RGGB:
        raise Exception(f"Unexpected bayer format={bayer_format}")
    # upper_line is red0, green0, red1, green1, ...
    r, g = sr.ravel(), sg.ravel()
    assert r.size == g.size
    c = cp.empty(((r.size + g.size) // 2,), dtype=dtype)
    c[0::2] = r[::2]
    c[1::2] = g[::2]
    upper_line = c.reshape(height, bayer_width)[::2]  # every other line
    # lower_line is green0, blue0, green1, blue1, ...
    g, b = sg.ravel(), sb.ravel()
    c = cp.empty(((g.size + b.size) // 2,), dtype=dtype)
    c[0::2] = g[::2]
    c[1::2] = b[::2]
    lower_line = c.reshape(height, bayer_width)[::2]  # every other line
    return cp.stack([upper_line, lower_line], axis=1).reshape(bayer_height, bayer_width)


def _bayer_image_to_csi(pixel_format, bayer_image, start_byte, received_line_bytes):
    if pixel_format != csi.PixelFormat.RAW_10:
        raise Exception(f"Unexpected {pixel_format=}")
    csi_image = _encode_raw_10_bayer_image(bayer_image)
    rows, _ = csi_image.shape
    expected_lines = cp.resize(csi_image, (rows, received_line_bytes))
    start_metadata = cp.empty((start_byte,), dtype=cp.uint8)
    return cp.concatenate((start_metadata, expected_lines.ravel()))


def _make_csi_from_image_file(
    height, width, pixel_format, bayer_format, start_byte, received_line_bytes, filename
):
    # PIL uses module-specific loggers that don't inherit the root logger's
    # pytest filters; quiet it so a stray WARN doesn't crash the run.
    logging.getLogger("PIL").setLevel(logging.WARN)
    image = (
        PIL.Image.open(filename)
        .convert("RGB")
        .resize((width, height), PIL.Image.LANCZOS)
    )
    uint16_image = cp.array(image, dtype=np.uint16)  # only 8 bits are set
    if pixel_format != csi.PixelFormat.RAW_10:
        raise Exception(f"Unsupported {pixel_format=} for the fallback image")
    uint16_image[:] <<= 2  # use 10 bits
    rgb_image = uint16_image.reshape(height, width, 3)
    bayer_image = _encode_bayer_image(rgb_image, bayer_format)
    return _bayer_image_to_csi(
        pixel_format, bayer_image, start_byte, received_line_bytes
    )


class CsiImage:
    """Duck-typed converter that renders an image file into a CSI frame.

    Handed to Imx274SensorFactory.configure_converter like the real
    CsiToBayerOp; captures the CSI bytes (a cupy device array) so the frame can
    be served as the fallback while the device is disconnected.
    """

    def __init__(self, sensor_factory, filename):
        self._sensor_factory = sensor_factory
        self._filename = filename
        self._csi_image = None

    def receiver_start_byte(self):
        return 0

    def received_line_bytes(self, transmitted_line_bytes):
        return (transmitted_line_bytes + 7) & ~7  # round up to 8

    def transmitted_line_bytes(self, pixel_format, pixel_width):
        if pixel_format == csi.PixelFormat.RAW_8.value:
            return pixel_width
        if pixel_format == csi.PixelFormat.RAW_10.value:
            return pixel_width * 5 // 4
        if pixel_format == csi.PixelFormat.RAW_12.value:
            return pixel_width * 3 // 2
        raise Exception(f"Unexpected {pixel_format=}")

    def configure(self, start_byte, received_line_bytes, width, height, pixel_format):
        pixel_format_enum, bayer_format = self._sensor_factory.image_format()
        self._csi_image = _make_csi_from_image_file(
            height,
            width,
            pixel_format_enum,
            bayer_format,
            start_byte,
            received_line_bytes,
            filename=self._filename,
        )
        logging.info(f"{self._filename=} {self._csi_image.size=}")

    def csi_image(self):
        return self._csi_image


class StatusOp(holoscan.core.Operator):
    def __init__(
        self,
        *args,
        in_tensor_name="",
        out_tensor_name="",
        status_name="status",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._in_tensor_name = in_tensor_name
        self._out_tensor_name = out_tensor_name
        self._status_name = status_name
        # Position (normalized) the Holoviz text spec below reads as its anchor.
        self._status = np.asarray([(0.5, 0.5)])

    def setup(self, spec):
        spec.input("input")
        spec.output("output")
        spec.output("output_specs")

    def start(self):
        self._frame_count = 0

    def compute(self, op_input, op_output, context):
        self._frame_count += 1
        in_message = op_input.receive("input")
        in_tensor = in_message.get(self._in_tensor_name)
        op_output.emit(
            {self._status_name: self._status, self._out_tensor_name: in_tensor},
            "output",
        )
        spec = holoscan.operators.HolovizOp.InputSpec(self._status_name, "text")
        spec.text = [f"{self._frame_count=}"]
        spec.priority = 1
        op_output.emit([spec], "output_specs")


def check_crcs(
    context,
    expected_crc,
    computed_metadata,
    get_computed_crc,
    frame_limit,
    reset_after,
    settle=20,
):
    # Skip a window around each reset while the device drops, the fallback
    # ticks, and the recovered stream settles. Also drop the first/last couple
    # of frames (pipeline start/shutdown).
    skip = set()
    for frame in [1] + list(reset_after) + [frame_limit - 1]:
        for i in range(frame - settle, frame + settle + 1):
            skip.add(i)
    checked = 0
    passed = 0
    for frame, metadata in enumerate(computed_metadata):
        if frame in skip:
            continue
        computed_crc = get_computed_crc(metadata)
        checked += 1
        if computed_crc == expected_crc:
            passed += 1
        else:
            logging.info(f"{context} {frame=} {computed_crc=:#x} {expected_crc=:#x}")
    return checked, passed


@dataclasses.dataclass
class ChannelSpec:
    metadata: object
    sensor_factory: object
    network_receiver_factory: object
    watchdog: object
    expected_crc: object


class ReconnectTestApplication(holoscan.core.Application):
    def __init__(
        self,
        headless,
        cuda_context,
        cuda_device_ordinal,
        frame_limit,
        channels,
        reset_after,
    ):
        logging.info("__init__")
        super().__init__()
        self._headless = headless
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._frame_limit = frame_limit
        self._channels = channels
        self._reset_after = reset_after
        # Two data paths feed the visualizer (status text + image); ignore
        # duplicate metadata keys rather than raising.
        self.is_metadata_enabled = True
        self.metadata_policy = holoscan.core.MetadataPolicy.REJECT
        # Keep the fallback device buffers alive for the run.
        self._fallback_buffers = []
        self.recorders = []

    def _per_channel(self, context, channel):
        condition = holoscan.conditions.CountCondition(
            self,
            name=f"{context}_condition",
            count=self._frame_limit,
        )

        # csi_to_bayer_operator is the first operator that knows the received
        # sensor geometry.
        height, width = channel.sensor_factory.image_size()
        csi_pool_block_size = width * height * ctypes.sizeof(ctypes.c_uint16)
        csi_to_bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name=f"{context}_csi_to_bayer_pool",
            storage_type=1,  # device memory
            block_size=csi_pool_block_size,
            num_blocks=4,
        )
        csi_to_bayer_operator = hololink_module.operators.CsiToBayerOp(
            self,
            name=f"{context}_csi_to_bayer",
            allocator=csi_to_bayer_pool,
            cuda_device_ordinal=self._cuda_device_ordinal,
        )
        channel.sensor_factory.configure_converter(csi_to_bayer_operator)
        frame_size = csi_to_bayer_operator.get_csi_length()

        # Fallback SMPTE bars served while disconnected: render the CSI frame
        # (a cupy device array) once, size it into a frame_size device buffer,
        # and hand its pointer to the factory so every SensorDevice serves it
        # during an outage. Kept alive on the app for the run.
        here = os.path.dirname(__file__)
        fallback = CsiImage(
            channel.sensor_factory,
            filename=os.path.join(here, "SMPTE_Color_Bars.png"),
        )
        channel.sensor_factory.configure_converter(fallback)
        fallback_buffer = cp.zeros((frame_size,), dtype=cp.uint8)
        flat = fallback.csi_image().astype(cp.uint8).ravel()
        n = min(int(flat.size), frame_size)
        fallback_buffer[:n] = flat[:n]
        self._fallback_buffers.append(fallback_buffer)
        channel.sensor_factory.set_fallback_frame(
            int(fallback_buffer.data.ptr), frame_size
        )

        hsb_controller_operator = hololink_module.operators.HsbControllerOp(
            self,
            condition,
            name=f"{context}_controller",
            enumeration_metadata=channel.metadata,
            frame_context=self._cuda_context,
            frame_size=frame_size,
            network_receiver_factory=channel.network_receiver_factory,
            sensor_factory=channel.sensor_factory,
        )

        pixel_format, bayer_format = channel.sensor_factory.image_format()
        rgba_components_per_pixel = 4
        bayer_size = (
            width * rgba_components_per_pixel * ctypes.sizeof(ctypes.c_uint16) * height
        )
        bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name=f"{context}_pool",
            storage_type=1,  # device memory
            block_size=bayer_size,
            num_blocks=4,
        )
        demosaic = holoscan.operators.BayerDemosaicOp(
            self,
            name=f"{context}_demosaic",
            pool=bayer_pool,
            generate_alpha=True,
            alpha_value=65535,
            bayer_grid_pos=bayer_format.value,
            interpolation_mode=0,
        )

        # CRC of the bayer image tells us whether we're receiving the expected
        # test pattern once the device has recovered.
        bayer_crc = hololink_module.operators.ComputeCrcOp(
            self,
            name=f"{context}_bayer_crc",
            frame_size=bayer_size,
        )
        record_bayer_crc = hololink_module.operators.CheckCrcOp(
            self,
            name=f"{context}_record_bayer_crc",
            compute_crc_op=bayer_crc,
        )

        status = StatusOp(self, name=f"{context}_status")

        # One window per channel. Shrink them for a multi-channel (stereo) run
        # so both windows fit on screen and can be observed at once; a lone
        # channel gets a larger window. Title each with its peer IP and channel
        # number so the two stereo windows are easy to tell apart.
        window_height, window_width = (
            (400, 600) if len(self._channels) > 1 else (720, 1080)
        )
        metadata = channel.metadata
        peer_ip = metadata["peer_ip"] if "peer_ip" in metadata else "?"
        if "data_channel" in metadata:
            channel_number = metadata["data_channel"]
        elif "data_plane" in metadata:
            channel_number = metadata["data_plane"]
        else:
            channel_number = "?"
        window_title = f"IMX274 reconnect — {context} {peer_ip} ch{channel_number}"
        visualizer = holoscan.operators.HolovizOp(
            self,
            name=f"{context}_holoviz",
            height=window_height,
            width=window_width,
            window_title=window_title,
            headless=self._headless,
            framebuffer_srgb=True,
            enable_camera_pose_output=True,
            camera_pose_output_type="extrinsics_model",
        )

        watchdog_operator = operators.WatchdogOp(
            self,
            name=f"{context}_watchdog_operator",
            watchdog=channel.watchdog,
        )

        metadata_recorder = operators.RecordMetadataOp(
            self,
            name=f"{context}_metadata_recorder",
            metadata_class=ComputedCrcRecord,
        )
        self.recorders.append(metadata_recorder)

        self.add_flow(
            hsb_controller_operator, csi_to_bayer_operator, {("output", "input")}
        )
        self.add_flow(csi_to_bayer_operator, demosaic, {("output", "receiver")})
        self.add_flow(demosaic, bayer_crc, {("transmitter", "input")})
        self.add_flow(bayer_crc, record_bayer_crc, {("output", "input")})
        self.add_flow(record_bayer_crc, metadata_recorder, {("output", "input")})
        self.add_flow(metadata_recorder, status, {("output", "input")})
        self.add_flow(status, visualizer, {("output", "receivers")})
        self.add_flow(status, visualizer, {("output_specs", "input_specs")})
        self.add_flow(visualizer, watchdog_operator, {("camera_pose_output", "input")})
        return visualizer

    def compose(self):
        logging.info("compose")
        first_visualizer = None
        for index, channel in enumerate(self._channels):
            context = "left" if index == 0 else ("right" if index == 1 else str(index))
            visualizer = self._per_channel(context, channel)
            if first_visualizer is None:
                first_visualizer = visualizer

        # Trigger the board reset off the reactor thread (not the pipeline's
        # compute thread) via the HSB-Lite supplement. Reset the first channel's
        # board only — a single reset exercises the full loss/recovery path.
        reset_metadata = self._channels[0].metadata
        adapter = hololink_module.Adapter.get_adapter()
        reactor = adapter.reactor()

        def reset():
            logging.info("Triggering reset.")
            hololink_module.hsb_lite.HsbLiteInterfaceV1.get_service(
                reset_metadata
            ).trigger_reset()

        for trigger_frame in self._reset_after:
            reset_operator = operators.OnFrameNOperator(
                self,
                name=f"reset_operator_{trigger_frame}",
                trigger_frame=trigger_frame,
                callback=lambda reactor=reactor: reactor.add_callback(reset),
            )
            self.add_flow(
                first_visualizer, reset_operator, {("camera_pose_output", "input")}
            )


def _wait_for_channel(module_dir, peer_ip):
    adapter = hololink_module.Adapter.get_adapter()
    if module_dir:
        adapter.set_module_directory(module_dir)
    return adapter.wait_for_channel(peer_ip, DISCOVERY_TIMEOUT_S)


def reconnect_test(headless, frame_limit, reset_after, channel_inputs):
    logging.info("Initializing.")
    with applications.CudaContext() as (cu_context, cu_device_ordinal):
        channels = []
        # A per-channel frame-reception watchdog with a generous timeout — the
        # outage plus rediscovery can span several seconds.
        with contextlib.ExitStack() as stack:
            for index, ci in enumerate(channel_inputs):
                context = "left" if index == 0 else "right"
                watchdog = stack.enter_context(
                    utils.Watchdog(f"{context}_frame-reception", timeout=30)
                )
                channels.append(
                    ChannelSpec(
                        metadata=ci["metadata"],
                        sensor_factory=ci["sensor_factory"],
                        network_receiver_factory=ci["network_receiver_factory"],
                        watchdog=watchdog,
                        expected_crc=ci["expected_crc"],
                    )
                )
            application = ReconnectTestApplication(
                headless,
                cu_context,
                cu_device_ordinal,
                frame_limit,
                channels,
                reset_after,
            )
            application.run()

    records = [recorder.get_record() for recorder in application.recorders]
    return application, records


mono_modes_and_patterns = [
    (
        imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
        10,
        0xB718A38C,
    ),
]

stereo_modes_and_patterns = [
    (
        imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
        10,
        0xB718A38C,
        imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
        11,
        0x8F4D79DE,
    ),
]


def _mono_channel(metadata, instance, camera_mode, pattern, expected, receiver_factory):
    return {
        "metadata": metadata,
        "sensor_factory": Imx274SensorFactory(
            imx274.Imx274Cam, instance, camera_mode, pattern
        ),
        "network_receiver_factory": receiver_factory(),
        "expected_crc": expected,
    }


# Number of sensor-register writes into camera configuration at which the
# during-configuration tests trigger the board reset.
_RESET_DURING_CONFIGURATION_TRIGGER = 20


def _instrumented_mono_channel(
    metadata, instance, camera_mode, pattern, expected, receiver_factory
):
    # Like _mono_channel, but the camera is an InstrumentedImx274Cam that
    # resets the board mid-configuration. The shared context is returned on the
    # channel so the test can confirm the reset actually fired.
    context = InstrumentedImx274CamContext(
        metadata, set_registers_trigger=_RESET_DURING_CONFIGURATION_TRIGGER
    )

    def camera_factory(md):
        return InstrumentedImx274Cam(md, context=context)

    return {
        "metadata": metadata,
        "sensor_factory": Imx274SensorFactory(
            camera_factory, instance, camera_mode, pattern
        ),
        "network_receiver_factory": receiver_factory(),
        "expected_crc": expected,
        "reset_context": context,
    }


def _clean_tail_length(record, expected_crc):
    # Length of the trailing run of correct-CRC frames. After a reset the
    # pipeline serves the fallback image through reboot + rediscovery +
    # reconfiguration (those frames carry the fallback's CRC, not the
    # pattern's), then streams correctly — so a long clean tail proves the
    # stream recovered. Drops the last couple of frames (pipeline shutdown).
    crcs = [record[i].computed_crc for i in range(len(record) - 2)]
    clean = 0
    for crc in reversed(crcs):
        if crc != expected_crc:
            break
        clean += 1
    return clean


@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
@pytest.mark.parametrize("camera_mode, pattern, expected", mono_modes_and_patterns)
@pytest.mark.parametrize("hololink_sensor_instance", [0, 1])
def test_roce_imx274_reconnect(
    camera_mode,
    pattern,
    expected,
    headless,
    channel_ips,
    hololink_sensor_instance,
    module_dir,
):
    if hololink_sensor_instance >= len(channel_ips):
        pytest.skip(
            f"--channel-ips has {len(channel_ips)} IP(s); this case needs "
            f"index {hololink_sensor_instance}"
        )
    channel_ip = channel_ips[hololink_sensor_instance]
    frame_limit = 300
    reset_after = [150]
    metadata = _wait_for_channel(module_dir, channel_ip)
    channel = _mono_channel(
        metadata,
        hololink_sensor_instance,
        camera_mode,
        pattern,
        expected,
        hololink_module.operators.make_roce_network_receiver_factory,
    )
    _application, records = reconnect_test(
        headless, frame_limit, reset_after, [channel]
    )
    # The device was reset once, so a fresh sensor was built at least twice
    # (initial connect + reconnect).
    assert channel["sensor_factory"]._sensor_count >= 2
    checked, passed = check_crcs(
        "roce-reconnect",
        expected,
        records[0],
        lambda metadata: metadata.computed_crc,
        frame_limit,
        reset_after,
    )
    assert checked > 100
    assert passed == checked


@pytest.mark.skip_unless_imx274
@pytest.mark.parametrize("camera_mode, pattern, expected", mono_modes_and_patterns)
@pytest.mark.parametrize("hololink_sensor_instance", [0, 1])
def test_linux_imx274_reconnect(
    camera_mode,
    pattern,
    expected,
    headless,
    channel_ips,
    hololink_sensor_instance,
    module_dir,
):
    if hololink_sensor_instance >= len(channel_ips):
        pytest.skip(
            f"--channel-ips has {len(channel_ips)} IP(s); this case needs "
            f"index {hololink_sensor_instance}"
        )
    channel_ip = channel_ips[hololink_sensor_instance]
    frame_limit = 300
    reset_after = [150]
    metadata = _wait_for_channel(module_dir, channel_ip)
    channel = _mono_channel(
        metadata,
        hololink_sensor_instance,
        camera_mode,
        pattern,
        expected,
        hololink_module.operators.make_linux_network_receiver_factory,
    )
    _application, records = reconnect_test(
        headless, frame_limit, reset_after, [channel]
    )
    # Linux sockets drop packets, so we validate recovery, not CRCs.
    assert channel["sensor_factory"]._sensor_count >= 2


@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode_left, pattern_left, expected_left, "
    "camera_mode_right, pattern_right, expected_right",
    stereo_modes_and_patterns,
)
def test_stereo_roce_imx274_reconnect(
    camera_mode_left,
    pattern_left,
    expected_left,
    camera_mode_right,
    pattern_right,
    expected_right,
    headless,
    channel_ips,
    module_dir,
):
    if len(channel_ips) < 2:
        pytest.skip("--channel-ips needs at least two IPs for this test")
    frame_limit = 300
    reset_after = [150]
    metadata_left = _wait_for_channel(module_dir, channel_ips[0])
    metadata_right = _wait_for_channel(module_dir, channel_ips[1])
    channel_left = _mono_channel(
        metadata_left,
        0,
        camera_mode_left,
        pattern_left,
        expected_left,
        hololink_module.operators.make_roce_network_receiver_factory,
    )
    channel_right = _mono_channel(
        metadata_right,
        1,
        camera_mode_right,
        pattern_right,
        expected_right,
        hololink_module.operators.make_roce_network_receiver_factory,
    )
    _application, records = reconnect_test(
        headless, frame_limit, reset_after, [channel_left, channel_right]
    )
    assert channel_left["sensor_factory"]._sensor_count >= 2
    for context, channel, record in (
        ("left", channel_left, records[0]),
        ("right", channel_right, records[1]),
    ):
        checked, passed = check_crcs(
            f"stereo-roce-reconnect-{context}",
            channel["expected_crc"],
            record,
            lambda metadata: metadata.computed_crc,
            frame_limit,
            reset_after,
        )
        assert checked > 100
        assert passed == checked


@pytest.mark.skip_unless_imx274
@pytest.mark.parametrize(
    "camera_mode_left, pattern_left, expected_left, "
    "camera_mode_right, pattern_right, expected_right",
    stereo_modes_and_patterns,
)
def test_stereo_linux_imx274_reconnect(
    camera_mode_left,
    pattern_left,
    expected_left,
    camera_mode_right,
    pattern_right,
    expected_right,
    headless,
    channel_ips,
    module_dir,
):
    if len(channel_ips) < 2:
        pytest.skip("--channel-ips needs at least two IPs for this test")
    frame_limit = 300
    reset_after = [150]
    metadata_left = _wait_for_channel(module_dir, channel_ips[0])
    metadata_right = _wait_for_channel(module_dir, channel_ips[1])
    channel_left = _mono_channel(
        metadata_left,
        0,
        camera_mode_left,
        pattern_left,
        expected_left,
        hololink_module.operators.make_linux_network_receiver_factory,
    )
    channel_right = _mono_channel(
        metadata_right,
        1,
        camera_mode_right,
        pattern_right,
        expected_right,
        hololink_module.operators.make_linux_network_receiver_factory,
    )
    _application, records = reconnect_test(
        headless, frame_limit, reset_after, [channel_left, channel_right]
    )
    # Linux sockets drop packets, so we validate recovery, not CRCs.
    assert channel_left["sensor_factory"]._sensor_count >= 2
    assert channel_right["sensor_factory"]._sensor_count >= 1


@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
@pytest.mark.parametrize("camera_mode, pattern, expected", mono_modes_and_patterns)
@pytest.mark.parametrize("hololink_sensor_instance", [0, 1])
def test_roce_imx274_reconnect_during_configuration(
    camera_mode,
    pattern,
    expected,
    headless,
    channel_ips,
    hololink_sensor_instance,
    module_dir,
):
    if hololink_sensor_instance >= len(channel_ips):
        pytest.skip(
            f"--channel-ips has {len(channel_ips)} IP(s); this case needs "
            f"index {hololink_sensor_instance}"
        )
    channel_ip = channel_ips[hololink_sensor_instance]
    frame_limit = 200
    # No frame-based reset: the reset is triggered from inside camera
    # configuration (see InstrumentedImx274Cam), losing the board mid-bring-up.
    reset_after = []
    metadata = _wait_for_channel(module_dir, channel_ip)
    channel = _instrumented_mono_channel(
        metadata,
        hololink_sensor_instance,
        camera_mode,
        pattern,
        expected,
        hololink_module.operators.make_roce_network_receiver_factory,
    )
    _application, records = reconnect_test(
        headless, frame_limit, reset_after, [channel]
    )
    assert channel["reset_context"]._reset_triggered
    clean = _clean_tail_length(records[0], expected)
    logging.info(f"roce-reconnect-during-config: clean tail = {clean} frames")
    assert clean > 50


@pytest.mark.skip_unless_imx274
@pytest.mark.parametrize("camera_mode, pattern, expected", mono_modes_and_patterns)
@pytest.mark.parametrize("hololink_sensor_instance", [0, 1])
def test_linux_imx274_reconnect_during_configuration(
    camera_mode,
    pattern,
    expected,
    headless,
    channel_ips,
    hololink_sensor_instance,
    module_dir,
):
    if hololink_sensor_instance >= len(channel_ips):
        pytest.skip(
            f"--channel-ips has {len(channel_ips)} IP(s); this case needs "
            f"index {hololink_sensor_instance}"
        )
    channel_ip = channel_ips[hololink_sensor_instance]
    frame_limit = 200
    reset_after = []
    metadata = _wait_for_channel(module_dir, channel_ip)
    channel = _instrumented_mono_channel(
        metadata,
        hololink_sensor_instance,
        camera_mode,
        pattern,
        expected,
        hololink_module.operators.make_linux_network_receiver_factory,
    )
    _application, records = reconnect_test(
        headless, frame_limit, reset_after, [channel]
    )
    # Linux sockets drop packets, so we validate that the mid-configuration
    # reset fired and the sensor was (re)built, not CRCs.
    assert channel["reset_context"]._reset_triggered
    assert channel["sensor_factory"]._sensor_count >= 1


@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
@pytest.mark.parametrize(
    "camera_mode_left, pattern_left, expected_left, "
    "camera_mode_right, pattern_right, expected_right",
    stereo_modes_and_patterns,
)
def test_stereo_roce_imx274_reconnect_during_configuration(
    camera_mode_left,
    pattern_left,
    expected_left,
    camera_mode_right,
    pattern_right,
    expected_right,
    headless,
    channel_ips,
    module_dir,
):
    if len(channel_ips) < 2:
        pytest.skip("--channel-ips needs at least two IPs for this test")
    frame_limit = 200
    reset_after = []
    metadata_left = _wait_for_channel(module_dir, channel_ips[0])
    metadata_right = _wait_for_channel(module_dir, channel_ips[1])
    # Only the left leg's camera is instrumented, so the mid-configuration reset
    # hits the left board; the right board streams throughout.
    channel_left = _instrumented_mono_channel(
        metadata_left,
        0,
        camera_mode_left,
        pattern_left,
        expected_left,
        hololink_module.operators.make_roce_network_receiver_factory,
    )
    channel_right = _mono_channel(
        metadata_right,
        1,
        camera_mode_right,
        pattern_right,
        expected_right,
        hololink_module.operators.make_roce_network_receiver_factory,
    )
    _application, records = reconnect_test(
        headless, frame_limit, reset_after, [channel_left, channel_right]
    )
    assert channel_left["reset_context"]._reset_triggered
    for context, expected_crc, record in (
        ("left", expected_left, records[0]),
        ("right", expected_right, records[1]),
    ):
        clean = _clean_tail_length(record, expected_crc)
        logging.info(
            f"stereo-roce-reconnect-during-config-{context}: clean tail = "
            f"{clean} frames"
        )
        assert clean > 50


@pytest.mark.skip_unless_imx274
@pytest.mark.accelerated_networking
def test_stereo_imx274_reconnect_example(headless, channel_ips, module_dir):
    # Smoke-test the minimal stereo reconnection example (single FrameAligner'd
    # Holoviz window): it runs end to end for a short frame budget and both legs
    # connect at least once.
    if len(channel_ips) < 2:
        pytest.skip("--channel-ips needs at least two IPs for this test")
    import reconnecting_imx274_player as example

    application = example.run(
        channel_ips=channel_ips[:2],
        frame_limit=30,
        headless=headless,
        transport="roce",
        module_dir=module_dir,
    )
    for channel in application._channels:
        assert channel["sensor_factory"]._sensor_count >= 1
