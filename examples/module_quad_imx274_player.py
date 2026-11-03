# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# 4-channel IMX274 player driving the hololink_module V1 surface. Python port of this repo's
# own previous (HSDK4-era) Python version of this app, via the HSDK5 Python bindings -- the
# sensor bring-up sequence below matches that original exactly; the receiver/pipeline
# construction (including the 2x2 HolovizOp grid layout) instead mirrors this repo's own
# examples/module_quad_imx274_player.cpp. The per-board construction below is otherwise
# identical to examples/module_linux_imx274_player.py -- this file just composes it four times
# into one 2x2-layout window instead of one window per board, and uses RoceReceiverOp(module)
# (hardware RDMA) instead of LinuxReceiverOp(module), matching both the HSDK4-era original and
# the C++ reference: four independent data planes aggregated over one data-plane bandwidth
# budget need RoCE, not a software socket per channel.
#
# Topology: 4 independent HSB-Lite data planes (each its own IMX274 camera) flowing into a
# single Holoscan pipeline (RoceReceiverOp(module) -> CsiToBayerOp(module) ->
# ImageProcessorOp(module) -> BayerDemosaicOp -> HolovizOp), with all four legs tiled into one
# window as a 2x2 grid (one HolovizOp, four image layers, each with its own "view" offset/scale
# -- cam0 top-left, cam1 top-right, cam2 bottom-left, cam3 bottom-right). Each camera trains
# its own CsiToBayerConverter and HololinkInterface independently; channels sharing a physical board
# (none do by default, but nothing assumes they can't) dedup their start()/reset() by
# HololinkInterface identity.
#
# Default IPs: 192.168.0.200 / .201 / .202 / .203 (devices 0..3). The example pins the cameras
# to IMX274_MODE_1920X1080_60FPS (mode 1, 1080p RAW10) so the four-channel aggregate fits in the
# data-plane bandwidth budget; the 4K modes over four channels exceed what the link can sustain.
#
# Unlike the C++ reference (which still carries a local BayerImageExtractorOp 16-bit->8-bit
# narrowing shim), this skips that stage -- same simplification already applied to this repo's
# other ported Python examples: BayerDemosaicOp accepts ImageProcessorOp's 16-bit schema::ImageT
# output directly.
#
# --frame-limit has no HSDK5 CountCondition equivalent, so it's implemented with a small native
# Python operator, FrameLimitOp, tapping each channel's receiver tensor output and counting
# activations independently per channel (matches the HSDK4-era original's own per-channel
# CountCondition); the one that brings the last channel to the limit stops the run from the
# graph, so a run that ends first, for example by failing, doesn't leave the app waiting.
#
# Hardware required: 4 HSB-Lite data planes (each driving an IMX274 camera) reachable on the
# test network.
#
# Usage:
#   module_quad_imx274_player.py [OPTIONS]
#
# Options:
#   --camera-mode N        IMX274 mode (1=1920x1080, 0=3840x2160 RAW10, 2=3840x2160 RAW12)
#   --headless              Run without opening a window (still renders, presents nowhere)
#   --frame-limit N        Exit after receiving this many frames per channel
#   --device-0 IP           Peer IP of device 0 (waited for via bootp)
#   --device-1 IP           Peer IP of device 1 (waited for via bootp)
#   --device-2 IP           Peer IP of device 2 (waited for via bootp)
#   --device-3 IP           Peer IP of device 3 (waited for via bootp)
#   --discovery-timeout S  Seconds to wait for all four bootp announcements before giving up
#   --module-dir PATH      Override the directory containing hololink_<UUID>.so files
#   --log-level N           Python logging level (default: 20 = INFO)
#   --window-height N      Total window height in pixels (default: 2160 // 4)
#   --window-width N       Total window width in pixels (default: 3840 // 3)
#   --title TITLE           Window title

import argparse
import dataclasses
import logging
import signal
import sys
import threading
import time

import hololink_module.operators
import hololink_module.sensors.imx274
import holoscan as hs
import holoscan_vision as hv
import holoscan_visualization as hz  # noqa: F401 -- registers "holoscan.visualization.HolovizOp"
from cuda.bindings import driver as cuda_driver
from hololink_module.sensors.imx274 import imx274_mode as adapter_imx274_mode

import hololink_module


def _cuda_check(result):
    err = result[0]
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        _, description = cuda_driver.cuGetErrorString(err)
        raise RuntimeError(f"CUDA error: {description}")
    if len(result) == 2:
        return result[1]
    return result[1:]


@dataclasses.dataclass
class _Channel:
    """Everything the application needs to drive one data plane.

    The enumeration metadata identifies the supplement module; the HololinkInterface drives
    the per-board control plane lifecycle; the Imx274Cam drives this data plane's sensor. Two
    channels that share a board hold the same HololinkInterface (the service locator returns
    one instance per (module, instance_id) pair); the application treats them independently
    regardless.
    """

    metadata: object
    hololink: object
    camera: object


def _enumerate(adapter, expected_peer_ips, timeout_s=30.0):
    """Block until every expected peer IP has been enumerated.

    Each Adapter.wait_for_channel call is bounded by the remaining time against the
    cumulative deadline, so the total time to wait for all peers is bounded by ``timeout_s``
    rather than ``len(expected_peer_ips) * timeout_s``. Returns the metadata list in the same
    order as ``expected_peer_ips``.
    """
    deadline = time.monotonic() + timeout_s
    found = []
    for ip in expected_peer_ips:
        remaining = max(deadline - time.monotonic(), 0.0)
        found.append(adapter.wait_for_channel(ip, remaining))
    return found


class _FrameLimit:
    """Shared state for --frame-limit: one counter per data plane, matching the HSDK4-era
    original's own per-channel CountCondition (every channel independently gated on its own
    frame count; FrameLimitOp stops the run once all four have reached the limit)."""

    lock = threading.Lock()
    counts = []
    limit = 0

    @classmethod
    def reset(cls, limit, channel_count):
        with cls.lock:
            cls.counts = [0] * channel_count
            cls.limit = limit

    @classmethod
    def record(cls, channel_index):
        """Counts a frame; True once every counter has reached the limit."""
        with cls.lock:
            cls.counts[channel_index] += 1
            return cls.limit > 0 and all(count >= cls.limit for count in cls.counts)


class FrameLimitOp(hs.Operator):
    """Count one channel's captured frames toward --frame-limit; see _FrameLimit."""

    input = hs.Input("input", hs.payload.Tensor)

    channel_index = hs.Param(0)

    def setup(self, spec):
        spec.input(self.input, queue_depth=4)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, context):
        self.input.receive_data()
        if _FrameLimit.record(self.channel_index):
            context.request_stop()


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


def run(
    device_ips,
    camera_mode,
    frame_limit,
    headless,
    discovery_timeout,
    module_dir,
    window_height,
    window_width,
    window_title,
):
    cuda_device_ordinal = 0

    print("Initializing CUDA...")
    _cuda_check(cuda_driver.cuInit(0))
    cu_device = _cuda_check(cuda_driver.cuDeviceGet(cuda_device_ordinal))
    cu_context = _cuda_check(cuda_driver.cuDevicePrimaryCtxRetain(cu_device))
    cuda_driver.cuCtxSetCurrent(cu_context)

    adapter = hololink_module.Adapter.get_adapter()
    if module_dir:
        adapter.set_module_directory(module_dir)

    print(f"Enumerating {len(device_ips)} HSB boards: {', '.join(device_ips)}...")
    metadatas = _enumerate(adapter, device_ips, timeout_s=discovery_timeout)

    # Build a channel per device.
    channels = []
    for md in metadatas:
        # Get the driver object.
        camera = hololink_module.sensors.imx274.Imx274Cam(md)
        # Connect to the HSB unit.
        hololink_dev = hololink_module.HololinkInterfaceV1.get_service(md)
        channels.append(_Channel(metadata=md, hololink=hololink_dev, camera=camera))

    # Bring each unit up; channels sharing a board share a HololinkInterface, so start/reset
    # run once per board.
    camera_mode_enum = adapter_imx274_mode.Imx274_Mode(camera_mode)
    started = []
    brought_up = set()
    for ch in channels:
        if id(ch.hololink) not in brought_up:
            brought_up.add(id(ch.hololink))
            ch.hololink.start()
            started.append(ch.hololink)
            # Drive the unit into a known state.
            ch.hololink.reset()
        # Configure the camera.
        ch.camera.configure(camera_mode_enum)
        ch.camera.set_digital_gain_reg(0x4)
        # Re-assert the mode so geometry resolution below reads freshly-updated
        # _width/_height/_pixel_format state (mirrors the HSDK4-era original's own compose()
        # set_mode() call, made right before configure_converter() there too -- preserved by
        # examples/module_quad_imx274_player.cpp's own port for the same reason).
        ch.camera.set_mode(camera_mode_enum)

    try:
        if frame_limit:
            _FrameLimit.reset(frame_limit, len(channels))

        # Build the graph.
        graph = hs.Graph("module_quad_imx274_player")

        # Pass 1: declare one image layer per channel, 2x2 grid (cam0 top-left, cam1
        # top-right, cam2 bottom-left, cam3 bottom-right) -- the layer list must be complete
        # before HolovizOp is constructed.
        layers = []
        for i, ch in enumerate(channels):
            layers.append(
                {
                    "port_name": f"video_{i}",
                    "kind": "image",
                    "format": "rgba16_unorm",
                    "max_width": ch.camera._width,
                    "max_height": ch.camera._height,
                    "view": {
                        "offset_x": 0.5 * (i % 2),
                        "offset_y": 0.5 * (i // 2),
                        "scale_x": 0.5,
                        "scale_y": 0.5,
                    },
                }
            )

        if headless:
            # Renders into a CUDA image on HolovizOp's output (consumed by DiscardImageOp). The
            # CUDA sink has no framebuffer_srgb: its sRGB format encodes on write instead.
            sink = {
                "kind": "cuda",
                "width": window_width,
                "height": window_height,
                "format": "rgba8_srgb",
            }
        else:
            sink = {
                "kind": "window",
                # The demosaiced data is linear sensor output: encode to sRGB on write.
                "framebuffer_srgb": True,
                "title": window_title,
                "width": window_width,
                "height": window_height,
                "close_on_escape": True,
            }
        viz = graph.op(
            "holoscan.visualization.HolovizOp",
            "holoviz",
            layers=layers,
            sink=sink,
            trigger={
                "kind": "clock",
                "period_ns": round(1e9 / 60.0),
            },
        )
        if headless:
            discard = graph.op(DiscardImageOp, "discard")
            graph.add_flow(
                viz.output,
                discard.input,
                queue_policy=hs.QueuePolicy.DROP_OLDEST,
                queue_depth=1,
            )

        # Pass 2: build each camera's receive/convert/process/demosaic chain and wire it into
        # viz.
        tensor_output_operators = []
        for i, ch in enumerate(channels):
            name = str(i)

            # ch.camera.pixel_format()/bayer_format() return hololink_module.sensors.csi's
            # plain enum.Enum members (same as module_linux_imx274_player.py) -- use .value.
            pixel_format = ch.camera.pixel_format()
            bayer_format = ch.camera.bayer_format()
            bayer_name = bayer_format.name.lower()
            # The sensor describes its CSI layout to the converter; the converter then reports
            # the receiver's frame size.
            csi_to_bayer = hololink_module.CsiToBayerConverter()
            ch.camera.configure_converter(csi_to_bayer)

            receiver = graph.op(
                hololink_module.operators.RoceReceiverOp,
                f"receiver_{name}",
                metadata=ch.metadata,
                frame_context=int(cu_context),
                device_start=ch.camera.start,
                device_stop=ch.camera.stop,
                frame_size=csi_to_bayer.get_csi_length(),
            )

            if frame_limit:
                frame_limit_op = graph.op(
                    FrameLimitOp, f"frame_limit_{name}", channel_index=i
                )
                graph.add_flow(receiver.tensor, frame_limit_op.input, queue_depth=4)

            geometry = csi_to_bayer.geometry()
            csi = graph.op(
                hololink_module.operators.CsiToBayerOp,
                f"csi_to_bayer_{name}",
                cuda_device_ordinal=cuda_device_ordinal,
                geometry=geometry,
                bayer_format=bayer_name,
            )
            graph.add_flow(receiver.tensor, csi.in_tensor, queue_depth=4)
            graph.add_flow(receiver.metadata, csi.in_metadata, queue_depth=4)

            image_processor = graph.op(
                hololink_module.operators.ImageProcessorOp,
                f"image_processor_{name}",
                pixel_format=pixel_format.value,
                pixel_width=geometry["pixel_width"],
                pixel_height=geometry["pixel_height"],
                bayer_format=bayer_format.value,
                optical_black=50,  # IMX274 optical black
                cuda_device_ordinal=cuda_device_ordinal,
            )
            graph.add_flow(csi.tensor, image_processor.in_tensor, queue_depth=4)
            graph.add_flow(csi.metadata, image_processor.in_metadata, queue_depth=4)

            demosaic = graph.op(
                hv.operators.BayerDemosaicOp,
                f"bayer_demosaic_{name}",
                bayer_grid_pos=bayer_name,
                generate_alpha=True,
                alpha_value=65535,
            )
            graph.add_flow(image_processor.tensor, demosaic.input, queue_depth=4)
            graph.add_flow(
                demosaic.output,
                getattr(viz, f"video_{name}"),
                queue_policy=hs.QueuePolicy.DROP_OLDEST,
                queue_depth=1,
            )

            tensor_output_operators.append(receiver.name)
            tensor_output_operators.append(csi.name)
            tensor_output_operators.append(image_processor.name)

        clock_key = graph.add_clock(hs.RealtimeClock, "clock")
        graph.set_default_clock(clock_key)

        options = hs.CompileOptions()
        for operator_path in tensor_output_operators:
            options.deployment.bind_tensor_output_device(
                operator_path=operator_path,
                output_port="tensor",
                device=hs.execution.DeviceId(cuda_device_ordinal),
            )
        for i in range(len(channels)):
            options.deployment.bind_tensor_output_device(
                operator_path=f"bayer_demosaic_{i}",
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

        print(
            f"Capturing {len(channels)} sensors, mode={camera_mode_enum.value} "
            "-- running (Ctrl+C to stop)..."
        )

        with hs.run_async(plan) as session:
            try:
                session.wait()
            except KeyboardInterrupt:
                # Ctrl+C or SIGTERM: leaving the block stops the session cooperatively, which
                # stops the camera.
                pass
        if session.diagnostics:
            print(session.summary(), file=sys.stderr)
        failed = session.termination == hs.RunTermination.FAILED
    finally:
        # Stop streaming data and close up sockets.
        for hololink_dev in started:
            hololink_dev.stop()

    _cuda_check(cuda_driver.cuDevicePrimaryCtxRelease(cu_device))

    if failed:
        return 1
    print("Done.")
    return 0


def main():
    # Like Ctrl+C, SIGTERM (kill, docker stop) raises KeyboardInterrupt, so the session stops
    # cooperatively and the camera and the board are stopped on the way out.
    signal.signal(signal.SIGTERM, signal.default_int_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera-mode",
        type=int,
        default=adapter_imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS.value,
        help="IMX274 mode (1=1920x1080, 0=3840x2160 RAW10, 2=3840x2160 RAW12)",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Exit after receiving this many frames per channel",
    )
    parser.add_argument(
        "--device-0",
        default="192.168.0.200",
        help="Peer IP of device 0 (waited for via bootp)",
    )
    parser.add_argument(
        "--device-1",
        default="192.168.0.201",
        help="Peer IP of device 1 (waited for via bootp)",
    )
    parser.add_argument(
        "--device-2",
        default="192.168.0.202",
        help="Peer IP of device 2 (waited for via bootp)",
    )
    parser.add_argument(
        "--device-3",
        default="192.168.0.203",
        help="Peer IP of device 3 (waited for via bootp)",
    )
    parser.add_argument(
        "--discovery-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for all four bootp announcements before giving up",
    )
    parser.add_argument(
        "--module-dir",
        default=None,
        help="Override the directory containing hololink_<UUID>.so files "
        "(env HOLOLINK_MODULE_DIR or /usr/lib/hololink/modules otherwise)",
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
    )
    parser.add_argument("--window-height", type=int, default=2160 // 4)
    parser.add_argument("--window-width", type=int, default=3840 // 3)
    parser.add_argument("--title", default="hololink_module — 4-channel IMX274")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)

    device_ips = [args.device_0, args.device_1, args.device_2, args.device_3]

    return run(
        device_ips,
        args.camera_mode,
        args.frame_limit,
        args.headless,
        args.discovery_timeout,
        args.module_dir,
        args.window_height,
        args.window_width,
        args.title,
    )


if __name__ == "__main__":
    sys.exit(main())
