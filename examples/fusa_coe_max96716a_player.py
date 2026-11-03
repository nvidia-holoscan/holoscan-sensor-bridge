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

# MAX96716A internal video pattern generator (VPG) live viewer over the FuSa CoE capture
# path. Python port of this repo's own previous (HSDK4-era) Python version of this app, via
# the HSDK5 Python bindings -- the deserializer/VPG bring-up sequence below matches that
# original exactly; the receiver/pipeline construction instead mirrors this repo's own
# examples/fusa_coe_max96716a_player.cpp.
#
#   FusaCoeCaptureOp(module) -> RgbReshapeOp -> HolovizOp
#
# Near-identical to the already-ported examples/max96716a_player.py (RoCE sibling) -- same
# RgbReshapeOp, same VPG/TauroTechDa326 setup -- with hololink_module.operators.
# FusaCoeCaptureOp swapped in for the receiver. The VPG has no CSI/Bayer stage (the
# deserializer's built-in test pattern is RGB888 already); FusaCoeCaptureOp's Python
# authoring row has no exposed equivalent of the opaque byte-blob capture mode the .cpp
# reference uses instead (frame_size_bytes -- "has no Python-facing caller in either tree's
# registration row currently", per module_operator_catalog.cpp's own comment), so
# `geometry` is instead synthesized by describing the RGB888 byte stream as a
# one-byte-per-pixel (RAW_8) image whose "pixel width" is the per-row byte count -- see the
# geometry comment in run() below.
#
# --frame-limit has no HSDK5 CountCondition equivalent, so it's implemented with a small
# native Python operator, FrameLimitOp, tapping the receiver's tensor output and counting
# activations (same pattern as every other *_player.py port in this repo, including
# examples/max96716a_player.py/examples/linux_max96716a_player.py above) -- simpler than,
# but equivalent in effect to, examples/fusa_coe_max96716a_player.cpp's own
# metadata-tapping FrameLimitOp/FrameLimitStatus pair.
#
# Dropped, matching established precedent: --configuration (application.config(), no HSDK5
# equivalent, every port in this repo already drops this). --fullscreen was never part of
# this particular app's CLI (neither the HSDK4-era original nor
# examples/fusa_coe_max96716a_player.cpp have it) and isn't added here either.
#
# Usage:
#   fusa_coe_max96716a_player.py [OPTIONS]
#
# Options:
#   --headless             Run without opening a window (still renders, presents nowhere)
#   --frame-limit N        Number of frames to capture (default: unlimited)
#   --hololink IP          IP address of Hololink board (default: 192.168.0.2)
#   --log-level N          Python logging level (default: 20 = INFO)
#   --window-height N      Window height in pixels (default: 1080)
#   --window-width N       Window width in pixels (default: 1920)
#   --title TITLE          Window title prefix (default: "MAX96716A VPG (FuSa CoE)")
#   --width N              VPG output width in pixels (default: 1920)
#   --height N             VPG output height in pixels (default: 1080)
#   --pattern {checkerboard,gradient}  VPG test pattern (default: checkerboard)
#   --sensor {left,right}  Which SIF to capture from (left=SIF 0/Port A, right=SIF 1/Port B)
#   --timeout N            Per-frame capture timeout, in milliseconds (default: 1500)
#   --interface NAME       Network interface to capture on (default: from channel metadata)
#   --skip-setup           Skip board/deserializer bring-up (assume already configured)
#   --skip-reset           Skip the initial board reset (only meaningful without --skip-setup)

import argparse
import logging
import signal
import sys
import threading

import hololink_module.operators
import hololink_module.sensors.csi as csi_module
import hololink_module.sensors.deserializers.max96716a as max96716a_module
import hololink_module.taurotech_da326 as taurotech_da326_module
import holoscan as hs
import holoscan_visualization as hz  # noqa: F401 -- registers "holoscan.visualization.HolovizOp"
from cuda.bindings import driver as cuda_driver

import hololink_module

# Mirrors hololink::module::operators::FusaCoeCaptureOp's own kMaxFrameBytes (64 MiB) -- see
# fusa_coe_capture_op.hpp -- not Python-visible, so duplicated here the same way
# frame_size's own formula below is (there is no live operator instance to query before the
# graph compiles).
_MAX_FRAME_BYTES = 64 * 1024 * 1024

SENSOR_ARG_TO_SIF = {"left": 0, "right": 1}


def _cuda_check(result):
    err = result[0]
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        _, description = cuda_driver.cuGetErrorString(err)
        raise RuntimeError(f"CUDA error: {description}")
    if len(result) == 2:
        return result[1]
    return result[1:]


def _parse_mac(mac_str):
    return [int(x, 16) for x in mac_str.split(":")]


def _write24(deserializer, base, value):
    deserializer.set_register(base + 0, (value >> 16) & 0xFF)
    deserializer.set_register(base + 1, (value >> 8) & 0xFF)
    deserializer.set_register(base + 2, value & 0xFF)


def _write16(deserializer, base, value):
    deserializer.set_register(base + 0, (value >> 8) & 0xFF)
    deserializer.set_register(base + 1, value & 0xFF)


def configure_vpg(deserializer, width, height, pattern):
    h_active = width
    h_blank = 280
    h_total = h_active + h_blank
    v_active = height
    v_blank = 45
    v_total = v_active + v_blank

    vs_high = v_active * h_total
    vs_low = v_blank * h_total

    hs_high = 44
    hs_low = h_total - hs_high
    hs_cnt = v_total

    de_high = h_active
    de_low = h_blank
    de_cnt = v_active

    _write24(deserializer, 0x0245, vs_high)
    _write24(deserializer, 0x0248, vs_low)
    _write16(deserializer, 0x024E, hs_high)
    _write16(deserializer, 0x0250, hs_low)
    _write16(deserializer, 0x0252, hs_cnt)
    _write16(deserializer, 0x0257, de_high)
    _write16(deserializer, 0x0259, de_low)
    _write16(deserializer, 0x025B, de_cnt)

    if pattern == "checkerboard":
        deserializer.set_register(0x025E, 0x00)
        deserializer.set_register(0x025F, 0x00)
        deserializer.set_register(0x0260, 0xFF)
        deserializer.set_register(0x0261, 0xFF)
        deserializer.set_register(0x0262, 0x00)
        deserializer.set_register(0x0263, 0x00)
        deserializer.set_register(0x0264, 16)
        deserializer.set_register(0x0265, 16)
        deserializer.set_register(0x0266, 8)
        patgen_mode = 0b01
    elif pattern == "gradient":
        deserializer.set_register(0x025D, 0x01)
        patgen_mode = 0b10
    else:
        raise RuntimeError(f"Unknown VPG pattern {pattern!r}")

    deserializer.set_register(0x0240, 0xE3)
    deserializer.set_register(0x0241, patgen_mode << 4)


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


class RgbReshapeOp(hs.Operator):
    """Reshapes the flat RGB888 receiver buffer into a (H, W, 3) holoscan.schema.Image
    Holoviz can render directly. No Bayer/CSI conversion is needed -- the VPG already emits
    interleaved RGB. Mirrors examples/fusa_coe_max96716a_player.cpp's own RgbReshapeOp,
    including its drop-(rather than pad-)short-frame behavior. Copies into SDK-owned
    image buffers like examples/max96716a_player.py and linux_max96716a_player.py.
    """

    input = hs.Input("input", hs.payload.Tensor)
    output = hs.Output("output", hs.schema.Image)

    width = hs.Param(0)
    height = hs.Param(0)

    def setup(self, spec):
        spec.input(self.input, queue_depth=4)
        spec.output(self.output).produces_tensor(
            hs.tensor.OutputSpec(
                storage="runtime_pool",
                memory_kind=hs.tensor.MemoryKind.CUDA_DEVICE,
                dtype=hs.tensor.DLDataType(1, 8, 1),
                rank=3,
            ),
            hs.tensor.byte_span(self.width * self.height * 3),
            field="data",
        )

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, context):
        sample = self.input.receive()
        if sample.is_missing:
            return
        pixels = self.width * self.height * 3
        if sample.data.nbytes < pixels:
            return  # Short frame -- drop it. Holoviz can't render a partial image.
        capture_time = sample.metadata.capture_time
        if not capture_time.valid:
            capture_time = context.activation_time
        descriptor = hs.schema.Image(
            height=self.height,
            width=self.width,
            encoding=hs.schema.image.ImageEncoding.RGB8,
        )
        request = hs.tensor.LoanRequest(
            shape=(self.height, self.width, 3), dtype=hs.tensor.DLDataType(1, 8, 1)
        )
        options = hs.EmitOptions(
            capture_time=capture_time, frame_id=sample.metadata.frame_id
        )
        stream = context.stream()
        try:
            with stream:
                with self.output.allocate_tensors(
                    descriptor, {"data": request}, options=options
                ) as loan:
                    _cuda_check(
                        cuda_driver.cuMemcpyDtoDAsync(
                            loan.tensors["data"].data,
                            sample.data.data,
                            pixels,
                            stream.__cuda_stream__()[1],
                        )
                    )
        except (hs.errors.ResourceExhaustedError, hs.errors.BackpressuredError):
            return  # Display is behind; drop this frame, matching the other VPG players.


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
    headless,
    frame_limit,
    window_height,
    window_width,
    title,
    vpg_width,
    vpg_height,
    pattern,
    sensor_arg,
    timeout,
    interface_arg,
    skip_setup,
    skip_reset,
):
    cuda_device_ordinal = 0

    # Fail fast, before touching any hardware, rather than letting a bad resolution surface
    # as a cryptic error (or garbage register writes in configure_vpg()) deep inside the app.
    if vpg_width <= 0 or vpg_height <= 0:
        print(
            f"Invalid resolution {vpg_width}x{vpg_height}: --width/--height must both be "
            "positive.",
            file=sys.stderr,
        )
        return 1
    # RGB888 = 3 bytes/pixel; pad generously for CSI-2 per-line framing overhead.
    frame_size = vpg_width * 3 * vpg_height + 16 * vpg_height + 4096
    if frame_size > _MAX_FRAME_BYTES:
        print(
            f"Requested resolution {vpg_width}x{vpg_height} ({frame_size} bytes/frame) "
            f"exceeds the maximum supported frame size ({_MAX_FRAME_BYTES} bytes). Reduce "
            "--width/--height.",
            file=sys.stderr,
        )
        return 1

    print("Initializing CUDA...")
    _cuda_check(cuda_driver.cuInit(0))
    cu_device = _cuda_check(cuda_driver.cuDeviceGet(cuda_device_ordinal))
    cu_context = _cuda_check(cuda_driver.cuDevicePrimaryCtxRetain(cu_device))
    cuda_driver.cuCtxSetCurrent(cu_context)

    sif = SENSOR_ARG_TO_SIF[sensor_arg]

    print(f"Enumerating HSB board at {hololink_ip}...")
    adapter = hololink_module.Adapter.get_adapter()
    adapter_metadata = adapter.wait_for_channel(hololink_ip, 30.0)

    sif_metadata = hololink_module.EnumerationMetadata(adapter_metadata)
    adapter.use_sensor(sif_metadata, sif)

    holo_iface = hololink_module.HololinkInterfaceV1.get_service(adapter_metadata)
    holo_iface.start()
    board = taurotech_da326_module.TauroTechDa326InterfaceV1.get_service(
        adapter_metadata
    )
    sensor_holo = board.hololink()
    deserializer = max96716a_module.Max96716a(sensor_holo)

    try:
        if not skip_setup:
            if not skip_reset:
                holo_iface.reset()

            board.release_reset()
            board.power_cycle()
            board.check_power()
            board.setup_clock()

            dev_id = deserializer.get_register(deserializer.DEV_ID_REG)
            expected_dev_id = deserializer.DEV_ID
            if dev_id != expected_dev_id:
                raise RuntimeError(
                    f"Deserializer mismatch: expected {expected_dev_id}, got {dev_id}"
                )

            deserializer.configure_video_pipe()
            deserializer.set_register(0x0330, 0x84)  # force_csi_out
            configure_vpg(deserializer, vpg_width, vpg_height, pattern)

        window_title = f"{title} - {sensor_arg}"

        if frame_limit:
            _FrameLimit.reset(frame_limit)

        interface = (
            interface_arg if interface_arg else sif_metadata.get("interface", "")
        )

        # The VPG emits an opaque RGB888 byte stream, not real CSI/Bayer sensor data -- but
        # FusaCoeCaptureOp's Python authoring row unconditionally requires a `geometry=`
        # (see the file-level comment above). Describe the byte stream as a synthetic
        # one-byte-per-pixel (RAW_8) image whose "pixel width" is the per-row byte count (3
        # bytes/pixel * width); this gets the 64-byte HSB line-alignment padding right
        # without claiming real CSI/Bayer structure -- RgbReshapeOp above reshapes the flat
        # byte tensor directly and ignores this framing.
        geometry = hololink_module.resolve_fusa_coe_geometry_from_dimensions(
            csi_module.PixelFormat.RAW_8.value, vpg_width * 3, vpg_height, False
        )

        # Build the graph.
        graph = hs.Graph("fusa_coe_max96716a_player")

        capture = graph.op(
            hololink_module.operators.FusaCoeCaptureOp,
            "capture",
            # The VPG is configured above; capture still requires lifecycle callbacks.
            device_start=lambda: None,
            device_stop=lambda: None,
            metadata=sif_metadata,
            interface=interface,
            mac_addr=_parse_mac(sif_metadata.get("mac_id", "00:00:00:00:00:00")),
            timeout=timeout,
            pixel_format="raw_8",
            bayer_format="rggb",
            geometry=geometry,
        )

        if frame_limit:
            frame_limit_op = graph.op(FrameLimitOp, "frame_limit")
            graph.add_flow(capture.tensor, frame_limit_op.input, queue_depth=4)

        reshape = graph.op(RgbReshapeOp, "reshape", width=vpg_width, height=vpg_height)
        graph.add_flow(capture.tensor, reshape.input, queue_depth=4)

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
                "title": window_title,
                "width": window_width,
                "height": window_height,
                "close_on_escape": True,
                "framebuffer_srgb": True,
            }
        viz = graph.op(
            "holoscan.visualization.HolovizOp",
            "holoviz",
            layers=[
                {
                    "port_name": "video",
                    "kind": "image",
                    "format": "rgb8_unorm",
                    "max_width": vpg_width,
                    "max_height": vpg_height,
                }
            ],
            sink=sink,
            trigger={
                "kind": "clock",
                "period_ns": round(1e9 / 60.0),
            },
        )
        graph.add_flow(
            reshape.output,
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
        options.deployment.bind_tensor_output_device(
            operator_path="capture",
            output_port="tensor",
            device=hs.execution.DeviceId(cuda_device_ordinal),
        )
        options.deployment.bind_tensor_output_device(
            operator_path="reshape",
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
            f"Capturing {vpg_width}x{vpg_height} pattern={pattern} -- running "
            "(Ctrl+C to stop)..."
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
        holo_iface.stop()

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
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frame-limit", type=int, default=None)
    parser.add_argument("--hololink", default="192.168.0.2")
    parser.add_argument("--log-level", type=int, default=20)
    parser.add_argument("--window-height", type=int, default=1080)
    parser.add_argument("--window-width", type=int, default=1920)
    parser.add_argument("--title", default="MAX96716A VPG (FuSa CoE)")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument(
        "--pattern", choices=("checkerboard", "gradient"), default="checkerboard"
    )
    parser.add_argument(
        "--sensor", choices=tuple(SENSOR_ARG_TO_SIF.keys()), default="left"
    )
    parser.add_argument("--timeout", type=int, default=1500)
    parser.add_argument("--interface", default=None)
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--skip-reset", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    return run(
        args.hololink,
        args.headless,
        args.frame_limit,
        args.window_height,
        args.window_width,
        args.title,
        args.width,
        args.height,
        args.pattern,
        args.sensor,
        args.timeout,
        args.interface,
        args.skip_setup,
        args.skip_reset,
    )


if __name__ == "__main__":
    sys.exit(main())
