# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Minimal stereo IMX274 player on the hololink_module V1 surface, using the
reconnection framework.

Each IMX274 leg runs a self-healing capture pipeline
(HsbControllerOp -> CsiToBayerOp -> BayerDemosaicOp); a FrameAlignerOp pairs the
two free-running legs frame-for-frame, and both are shown side by side in a
single Holoviz window. The player starts immediately without waiting for any
device: it shows the SMPTE test pattern and connects each leg when its device
announces itself. If a leg's device is later lost, its device state is
invalidated, the test pattern returns, and it reconnects on re-announcement —
the two legs recover independently.

The two legs are the two data channels of one stereo (dual-sensor) board — each
data channel enumerates under its own peer IP. Run it with those two IPs:

    python reconnecting_imx274_player.py \\
        --channel-ips 192.168.0.2,192.168.0.3

Add --frame-limit N to exit after N frames (otherwise it runs until Ctrl-C),
and --transport linux to use the software receiver instead of RoCE.
"""

import argparse
import ctypes
import logging
import os
import sys
import threading
import weakref

import cuda.bindings.driver as cuda
import cupy as cp
import hololink_module.operators
import hololink_module.sensors.imx274 as imx274
import holoscan
import numpy as np
import PIL.Image
from hololink_module.sensors import csi
from hololink_module.sensors.imx274 import imx274_mode

import hololink_module

# FrameAlignerOp and PassThroughOperator are shared test operators; add the
# tests directory to the path so this runs standalone as well as under pytest.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import operators  # noqa: E402

# Largest skew the frame aligner tolerates between the two legs. IMX274 cannot
# hardware-sync, so the acceptance window is one full frame period.
ALLOWABLE_DT_S = 1.0 / 60

# The two legs are two data channels of one board, so they share a control
# plane: board bring-up (start()/reset()) must run once per board, not once per
# leg, or the second leg's reset would disrupt the first mid-stream. The first
# leg to reach new_sensor brings the board up; the second sees it already up. A
# reconnect yields a fresh HololinkInterfaceV1 (device_lost() invalidated the
# old one), so identity membership resets on its own; weak so invalidated
# instances drop out.
_board_lock = threading.Lock()
_boards_up = weakref.WeakSet()


class Imx274SensorDevice(hololink_module.operators.SensorDevice):
    """Wraps one armed Imx274Cam; replaced by a fresh one on reconnect."""

    def __init__(self, camera):
        super().__init__()
        self._camera = camera

    def stop_sensor(self):
        try:
            self._camera.stop()
        except Exception:
            logging.info("stop_sensor: camera.stop() failed (device likely gone).")
        return hololink_module.HOLOLINK_MODULE_OK


class Imx274SensorFactory(hololink_module.operators.SensorFactory):
    """Sensor-side reconnection policy for one IMX274 leg. new_sensor brings the
    board up and arms a fresh camera on every (re)connect."""

    def __init__(self, instance, camera_mode):
        super().__init__()
        self._instance = instance
        self._camera_mode = camera_mode
        self._sensor = None  # keeps the current SensorDevice's Python side alive
        self._sensor_count = 0
        self._fallback_ptr = 0
        self._fallback_size = 0
        if camera_mode == imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS:
            self._height, self._width = 1080, 1920
            self._pixel_format = csi.PixelFormat.RAW_10
            self._bayer_format = csi.BayerFormat.RGGB
        else:
            raise RuntimeError(f"Unsupported {camera_mode=}")

    def image_size(self):
        return self._height, self._width

    def image_format(self):
        return self._pixel_format, self._bayer_format

    def set_fallback_frame(self, ptr, size):
        self._fallback_ptr = ptr
        self._fallback_size = size

    def fallback_frame(self):
        # The image shown when no live frame is available — the SMPTE test bars
        # at startup (before the first connect) and during an outage. A device
        # pointer built up front; touches no live device handles.
        return (self._fallback_ptr, self._fallback_size)

    def configure_converter(self, converter):
        # Train the CsiToBayerOp to this sensor's geometry without a live camera
        # (none exists at compose time — the board comes up in new_sensor).
        pixel_format = self._pixel_format.value
        start_byte = converter.receiver_start_byte()
        transmitted_line_bytes = converter.transmitted_line_bytes(
            pixel_format, self._width
        )
        received_line_bytes = converter.received_line_bytes(transmitted_line_bytes)
        start_byte += converter.received_line_bytes(175)  # metadata preamble
        start_byte += received_line_bytes * 8  # RAW_10 optical-black lines
        converter.configure(
            start_byte, received_line_bytes, self._width, self._height, pixel_format
        )

    def new_sensor(self, metadata):
        # Reactor thread. A fresh Imx274Cam(metadata) re-get_service()s the
        # board's V1 handles, so it resyncs with post-reset module state. Report
        # a failed bring-up (e.g. the board is gone) by returning None so the
        # framework invalidates the board and retries on re-announce.
        try:
            hololink = hololink_module.HololinkInterfaceV1.get_service(metadata)
            with _board_lock:
                if hololink not in _boards_up:
                    hololink.start()
                    hololink.reset()
                    _boards_up.add(hololink)
            imx274.Imx274Cam.use_expander_configuration(metadata, self._instance)
            camera = imx274.Imx274Cam(metadata)
            camera.configure(self._camera_mode)
            camera.set_digital_gain_reg(0x4)
            camera.start()
        except Exception as e:
            logging.info(f"new_sensor: bring-up failed ({e}); awaiting re-announce.")
            return None
        self._sensor_count += 1
        self._sensor = Imx274SensorDevice(camera)
        return self._sensor


# CSI-frame synthesis for the fallback image: render an image file into a
# sensor-format CSI frame (a device-side cupy array). Only the IMX274 RAW_10 /
# RGGB path is implemented.


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

    Handed to Imx274SensorFactory.configure_converter like the real CsiToBayerOp;
    captures the CSI bytes (a cupy device array) so the frame can be served as
    the fallback while the device is disconnected.
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

    def csi_image(self):
        return self._csi_image


class StereoReconnectApplication(holoscan.core.Application):
    def __init__(
        self, headless, cuda_context, cuda_device_ordinal, channels, frame_limit
    ):
        super().__init__()
        self._headless = headless
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._channels = channels
        self._frame_limit = frame_limit
        # Keep the fallback device buffers alive for the run.
        self._fallback_buffers = []
        self.is_metadata_enabled = True
        # The two legs stamp distinct (left_/right_) metadata keys, so there are
        # no collisions when both feed the aligner.
        self.metadata_policy = holoscan.core.MetadataPolicy.REJECT

    def compose(self):
        # The two free-running legs are paired by the aligner and shown side by
        # side; each leg is renamed "left"/"right" for its pane.
        frame_aligner = operators.FrameAlignerOp(
            self,
            name="frame_aligner",
            allowable_dt=ALLOWABLE_DT_S,
            input_tensors=["left", "right"],
            rename_metadata=[lambda name: f"left_{name}", lambda name: f"right_{name}"],
            outputs=["output_left", "output_right"],
        )

        specs = []
        for index, channel in enumerate(self._channels):
            name = "left" if index == 0 else "right"
            factory = channel["sensor_factory"]

            height, width = factory.image_size()
            _, bayer_format = factory.image_format()
            csi_to_bayer_pool = holoscan.resources.BlockMemoryPool(
                self,
                name=f"{name}_csi_pool",
                storage_type=1,  # device memory
                block_size=width * height * ctypes.sizeof(ctypes.c_uint16),
                num_blocks=4,
            )
            csi_to_bayer = hololink_module.operators.CsiToBayerOp(
                self,
                name=f"{name}_csi_to_bayer",
                allocator=csi_to_bayer_pool,
                cuda_device_ordinal=self._cuda_device_ordinal,
            )
            factory.configure_converter(csi_to_bayer)
            frame_size = csi_to_bayer.get_csi_length()

            # Fallback SMPTE bars shown while disconnected (at startup before the
            # first connect and during an outage): render the CSI frame once,
            # size it into a frame_size device buffer, and hand its pointer to
            # the factory. HsbControllerOp serves it whenever no live frame is
            # available. Kept alive on the app for the run.
            here = os.path.dirname(__file__)
            fallback = CsiImage(
                factory, filename=os.path.join(here, "SMPTE_Color_Bars.png")
            )
            factory.configure_converter(fallback)
            fallback_buffer = cp.zeros((frame_size,), dtype=cp.uint8)
            flat = fallback.csi_image().astype(cp.uint8).ravel()
            n = min(int(flat.size), frame_size)
            fallback_buffer[:n] = flat[:n]
            self._fallback_buffers.append(fallback_buffer)
            factory.set_fallback_frame(int(fallback_buffer.data.ptr), frame_size)

            # Frame budget lives on each controller (like the reconnect tests) so
            # a player is bounded when --frame-limit is given, and free-runs
            # otherwise.
            if self._frame_limit:
                condition = holoscan.conditions.CountCondition(
                    self, name=f"{name}_count", count=self._frame_limit
                )
            else:
                condition = holoscan.conditions.BooleanCondition(
                    self, name=f"{name}_ok", enable_tick=True
                )

            controller = hololink_module.operators.HsbControllerOp(
                self,
                condition,
                name=f"{name}_controller",
                enumeration_metadata=channel["metadata"],
                frame_context=self._cuda_context,
                frame_size=frame_size,
                network_receiver_factory=channel["network_receiver_factory"],
                sensor_factory=factory,
                out_tensor_name=name,
                rename_metadata=(lambda prefix: lambda key: f"{prefix}_{key}")(name),
            )

            rgba_components_per_pixel = 4
            bayer_pool = holoscan.resources.BlockMemoryPool(
                self,
                name=f"{name}_bayer_pool",
                storage_type=1,
                block_size=width
                * rgba_components_per_pixel
                * ctypes.sizeof(ctypes.c_uint16)
                * height,
                num_blocks=4,
            )
            demosaic = holoscan.operators.BayerDemosaicOp(
                self,
                name=f"{name}_demosaic",
                pool=bayer_pool,
                generate_alpha=True,
                alpha_value=65535,
                bayer_grid_pos=bayer_format.value,
                interpolation_mode=0,
            )
            # BayerDemosaicOp emits an unnamed tensor; name it for its pane.
            rename = operators.PassThroughOperator(
                self, name=f"{name}_rename", out_tensor_name=name
            )

            # controller -> aligner; aligner.output_<name> -> csi -> demosaic ->
            # rename -> the shared visualizer.
            self.add_flow(controller, frame_aligner, {("output", "input")})
            self.add_flow(frame_aligner, csi_to_bayer, {(f"output_{name}", "input")})
            self.add_flow(csi_to_bayer, demosaic, {("output", "receiver")})
            self.add_flow(demosaic, rename, {("transmitter", "input")})

            view = holoscan.operators.HolovizOp.InputSpec.View()
            view.offset_x = 0.0 if index == 0 else 0.5
            view.offset_y = 0.0
            view.width = 0.5
            view.height = 1.0
            spec = holoscan.operators.HolovizOp.InputSpec(
                name, holoscan.operators.HolovizOp.InputType.COLOR
            )
            spec.views = [view]
            specs.append((rename, spec))

        visualizer = holoscan.operators.HolovizOp(
            self,
            name="holoviz",
            headless=self._headless,
            framebuffer_srgb=True,
            tensors=[spec for _rename, spec in specs],
            width=1280,
            height=400,
            window_title="Stereo IMX274 reconnection (module)",
        )
        for rename, _spec in specs:
            self.add_flow(rename, visualizer, {("output", "receivers")})


def _network_receiver_factory(transport):
    if transport == "roce":
        return hololink_module.operators.make_roce_network_receiver_factory()
    if transport == "linux":
        return hololink_module.operators.make_linux_network_receiver_factory()
    raise ValueError(f"Unknown transport {transport!r} (expected 'roce' or 'linux')")


def run(
    channel_ips,
    camera_mode=imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS,
    frame_limit=None,
    headless=False,
    transport="roce",
    module_dir=None,
):
    """Build and run the stereo reconnection player; returns the application (its
    per-leg Imx274SensorFactory objects carry a _sensor_count)."""
    if len(channel_ips) < 2:
        raise ValueError("stereo needs two channel IPs")

    (cu_result,) = cuda.cuInit(0)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_device_ordinal = 0
    cu_result, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_result, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS

    adapter = hololink_module.Adapter.get_adapter()
    if module_dir:
        adapter.set_module_directory(module_dir)

    try:
        channels = []
        for index, channel_ip in enumerate(channel_ips[:2]):
            # Don't block on discovery. The reconnection framework registers the
            # peer IP and connects whenever the device announces itself, so the
            # player starts immediately — showing the SMPTE test pattern until a
            # device appears (and again during any later outage). The startup
            # metadata only needs the peer IP; every other field comes from the
            # device's announcement.
            metadata = hololink_module.EnumerationMetadata()
            metadata["peer_ip"] = channel_ip
            channels.append(
                {
                    "metadata": metadata,
                    "sensor_factory": Imx274SensorFactory(index, camera_mode),
                    "network_receiver_factory": _network_receiver_factory(transport),
                }
            )
        application = StereoReconnectApplication(
            headless, cu_context, cu_device_ordinal, channels, frame_limit
        )
        application.run()
        return application
    finally:
        (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
        assert cu_result == cuda.CUresult.CUDA_SUCCESS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--channel-ips",
        default="192.168.0.2,192.168.0.3",
        help="Comma-separated peer IPs of the two data channels (typically the "
        "two channels of one stereo board)",
    )
    parser.add_argument(
        "--camera-mode",
        type=int,
        default=imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS.value,
        help="IMX274 mode (1 = 1920x1080 60fps)",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Exit after this many frames (default: run until Ctrl-C)",
    )
    parser.add_argument(
        "--transport",
        choices=("roce", "linux"),
        default="roce",
        help="Data-plane transport (default: roce)",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--module-dir",
        default=None,
        help="Override the directory containing hololink_<UUID>.so files",
    )
    parser.add_argument("--log-level", type=int, default=20)
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)

    run(
        channel_ips=[ip.strip() for ip in args.channel_ips.split(",") if ip.strip()],
        camera_mode=imx274_mode.Imx274_Mode(args.camera_mode),
        frame_limit=args.frame_limit,
        headless=args.headless,
        transport=args.transport,
        module_dir=args.module_dir,
    )


if __name__ == "__main__":
    main()
