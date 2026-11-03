# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Single-camera Leopard VB1940 player driving the hololink_module V1 surface through the
# SOFTWARE (Linux) receiver. Python port of this repo's own previous (HSDK4-era) Python version
# of this app, via the HSDK5 Python bindings -- the sensor bring-up sequence below matches that
# original exactly; the receiver/pipeline construction instead mirrors this repo's own
# examples/module_linux_vb1940_player.cpp. Runs on hosts with no infiniband device (the receiver
# is the software LinuxReceiverOp, no ibverbs).
#
#   LinuxReceiverOp(module) -> CsiToBayerOp(module) -> ImageProcessorOp(module) ->
#       BayerDemosaicOp -> HolovizOp
#
# Unlike the C++ reference (which still carries a local BayerImageExtractorOp 16-bit->8-bit
# narrowing shim), this skips that stage -- same simplification already applied to this repo's
# other ported Python examples (e.g. vb1940_player.py, linux_hawk_dual_link_player.py):
# BayerDemosaicOp accepts ImageProcessorOp's 16-bit schema::ImageT output directly.
#
# hololink_module's Vb1940Cam sensor driver (nanobind-bound C++, not pure Python) is unchanged;
# only the receiver (hololink::module::operators::LinuxReceiverOp, ported to HSDK5's
# Operator<Params> shape) and the downstream converters (CsiToBayerOp/ImageProcessorOp, both
# auto-registered) changed.
#
# --frame-limit has no HSDK5 CountCondition equivalent, so it's implemented with a small native
# Python operator, FrameLimitOp, tapping the receiver's tensor output and counting activations
# (matches the HSDK4-era original's own single CountCondition -- this app is single-sensor).
#
# Usage:
#   module_linux_vb1940_player.py [OPTIONS]
#
# Options:
#   --camera-mode N        VB1940 mode (0=2560x1984 30fps, 1=1920x1080 30fps,
#                           2=2560x1984 30fps 8-bit, 3=2560x1984 60fps)
#   --headless              Run without opening a window (still renders, presents nowhere)
#   --fullscreen            Run in fullscreen mode
#   --frame-limit N        Exit after receiving this many frames
#   --hololink IP           Peer IP of the Leopard VB1940 board (waited for via bootp)
#   --discovery-timeout S  Seconds to wait for the bootp announcement before giving up
#   --module-dir PATH      Override the directory containing hololink_<UUID>.so files
#   --log-level N           Python logging level (default: 20 = INFO)

import argparse
import logging
import signal
import sys
import threading

import hololink_module.operators
import hololink_module.sensors.vb1940
import holoscan as hs
import holoscan_vision as hv
import holoscan_visualization as hz  # noqa: F401 -- registers "holoscan.visualization.HolovizOp"
from cuda.bindings import driver as cuda_driver

import hololink_module


def _cuda_check(result):
    err = result[0]
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        _, description = cuda_driver.cuGetErrorString(err)
        raise RuntimeError(f"CUDA error: {description}")
    if len(result) == 2:
        return result[1]
    return result[1:]


class _FrameLimit:
    """Shared state for --frame-limit: a single counter (this app is single-sensor)."""

    lock = threading.Lock()
    count = 0
    limit = 0

    @classmethod
    def reset(cls, limit):
        with cls.lock:
            cls.count = 0
            cls.limit = limit

    @classmethod
    def record(cls):
        """Counts a frame; True once the limit is reached."""
        with cls.lock:
            cls.count += 1
            return cls.limit > 0 and cls.count >= cls.limit


class FrameLimitOp(hs.Operator):
    """Count captured frames toward --frame-limit; see _FrameLimit."""

    input = hs.Input("input", hs.payload.Tensor)

    def setup(self, spec):
        spec.input(self.input, queue_depth=4)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, context):
        self.input.receive_data()
        if _FrameLimit.record():
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
    hololink_ip,
    camera_mode,
    frame_limit,
    headless,
    fullscreen,
    discovery_timeout,
    module_dir,
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

    print(f"Enumerating HSB board at {hololink_ip}...")
    metadata = adapter.wait_for_channel(hololink_ip, discovery_timeout)

    # Get the driver object.
    camera = hololink_module.sensors.vb1940.Vb1940Cam(metadata)
    # Connect to the HSB unit.
    hololink_dev = hololink_module.HololinkInterfaceV1.get_service(metadata)
    hololink_dev.start()
    try:
        # Drive the unit into a known state.
        hololink_dev.reset()
        # Configure the camera (walks the secure-boot FSM the first time after power-up, then
        # applies the per-mode register table).
        camera_mode_enum = hololink_module.sensors.vb1940.Vb1940_Mode(camera_mode)
        camera.configure(camera_mode_enum)

        # camera.pixel_format()/bayer_format() are nanobind-bound with nb::is_arithmetic()
        # (confirmed by reading python/hololink_module/_hololink_module.cpp's bind_vb1940_cam()
        # directly, not assumed) -- int() would work directly here, unlike IMX274's plain
        # enum.Enum, but .value/.name.lower() work uniformly across both, so this app uses the
        # same accessors as every other module-tree Python player for consistency.
        pixel_format = camera.pixel_format()
        bayer_format = camera.bayer_format()
        bayer_name = bayer_format.name.lower()
        # The sensor describes its CSI layout to the converter; the converter then reports
        # the receiver's frame size.
        csi_to_bayer = hololink_module.CsiToBayerConverter()
        camera.configure_converter(csi_to_bayer)

        if frame_limit:
            _FrameLimit.reset(frame_limit)

        # Build the graph.
        graph = hs.Graph("module_linux_vb1940_player")

        receiver = graph.op(
            hololink_module.operators.LinuxReceiverOp,
            "receiver",
            metadata=metadata,
            frame_context=int(cu_context),
            device_start=camera.start,
            device_stop=camera.stop,
            frame_size=csi_to_bayer.get_csi_length(),
        )

        if frame_limit:
            frame_limit_op = graph.op(FrameLimitOp, "frame_limit")
            graph.add_flow(receiver.tensor, frame_limit_op.input, queue_depth=4)

        geometry = csi_to_bayer.geometry()
        csi = graph.op(
            hololink_module.operators.CsiToBayerOp,
            "csi_to_bayer",
            cuda_device_ordinal=cuda_device_ordinal,
            geometry=geometry,
            bayer_format=bayer_name,
        )
        graph.add_flow(receiver.tensor, csi.in_tensor, queue_depth=4)
        graph.add_flow(receiver.metadata, csi.in_metadata, queue_depth=4)

        image_processor = graph.op(
            hololink_module.operators.ImageProcessorOp,
            "image_processor",
            pixel_format=pixel_format.value,
            pixel_width=geometry["pixel_width"],
            pixel_height=geometry["pixel_height"],
            bayer_format=bayer_format.value,
            optical_black=8,  # VB1940 optical black (RAW10)
            cuda_device_ordinal=cuda_device_ordinal,
        )
        graph.add_flow(csi.tensor, image_processor.in_tensor, queue_depth=4)
        graph.add_flow(csi.metadata, image_processor.in_metadata, queue_depth=4)

        demosaic = graph.op(
            hv.operators.BayerDemosaicOp,
            "bayer_demosaic",
            bayer_grid_pos=bayer_name,
            generate_alpha=True,
            alpha_value=65535,
        )
        graph.add_flow(image_processor.tensor, demosaic.input, queue_depth=4)

        if headless:
            # Renders into a CUDA image on HolovizOp's output (consumed by DiscardImageOp). The
            # CUDA sink has no framebuffer_srgb: its sRGB format encodes on write instead.
            sink = {
                "kind": "cuda",
                "width": camera.width(),
                "height": camera.height(),
                "format": "rgba8_srgb",
            }
        else:
            sink = {
                "kind": "window",
                # The demosaiced data is linear sensor output: encode to sRGB on write.
                "framebuffer_srgb": True,
                "title": f"Holoviz - {hololink_ip}",
                "width": camera.width(),
                "height": camera.height(),
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
                    "max_width": camera.width(),
                    "max_height": camera.height(),
                }
            ],
            sink=sink,
            trigger={
                "kind": "clock",
                "period_ns": round(1e9 / 60.0),
            },
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
        for operator_path in ("receiver", "csi_to_bayer", "image_processor"):
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

        print(
            f"Capturing, mode={camera_mode_enum.value} "
            f"{camera.width()}x{camera.height()} -- running (Ctrl+C to stop)..."
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
        default=hololink_module.sensors.vb1940.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS.value,
        help="VB1940 mode (0=2560x1984 30fps, 1=1920x1080 30fps, "
        "2=2560x1984 30fps 8-bit, 3=2560x1984 60fps)",
    )
    parser.add_argument("--headless", action="store_true")
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
        "--hololink",
        default="192.168.0.2",
        help="Peer IP of the Leopard VB1940 board (waited for via bootp)",
    )
    parser.add_argument(
        "--discovery-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for the bootp announcement before giving up",
    )
    parser.add_argument(
        "--module-dir",
        default=None,
        help="Override the directory containing hololink_<UUID>.so files",
    )
    parser.add_argument("--log-level", type=int, default=20)
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)

    return run(
        args.hololink,
        args.camera_mode,
        args.frame_limit,
        args.headless,
        args.fullscreen,
        args.discovery_timeout,
        args.module_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
