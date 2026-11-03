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

# Two Hawk modules (one per GMSL link) live viewer over RoCE, one HolovizOp window per link.
# Python port of this repo's own previous (HSDK4-era) Python version of this app, via the HSDK5
# Python bindings -- the sensor bring-up sequence below matches that original exactly. The
# receiver construction mirrors this repo's own examples/hawk_dual_link_player.cpp (the RoCE
# C++ sibling); everything downstream of the receiver instead mirrors examples/
# linux_hawk_dual_link_player.py (this app's software-transport sibling, ported in the same
# effort) -- same CsiToBayerOp(module)/ImageProcessorOp(module) wiring, CLI surface, and
# FrameLimitOp pattern, just with the receiver swapped. Like that file (and unlike
# hawk_dual_link_player.cpp, which still carries a local BayerImageExtractorOp 16-bit->8-bit
# narrowing shim), this skips that stage -- same simplification already applied to this repo's
# other ported Python examples (e.g. vb1940_player.py): BayerDemosaicOp accepts
# ImageProcessorOp's 16-bit schema::ImageT output directly.
#
#   RoceReceiverOp(module) -> CsiToBayerOp(module) -> ImageProcessorOp(module) ->
#       BayerDemosaicOp -> HolovizOp   (x2, one per SIF/GMSL link)
#
# hololink_module's Hawk/Ar0234/Max9295d/Max96716a/TauroTechDa326InterfaceV1 sensor driver is
# unchanged; only the receiver (hololink::module::operators::RoceReceiverOp, ported to HSDK5's
# Operator<Params> shape) and the downstream converters (hololink::module::operators::
# CsiToBayerOp/ImageProcessorOp, both auto-registered in Phase 1 of the same effort) changed.
# Unlike native's RoceReceiverParams, module's has no ibv_name/ibv_port fields at all -- the
# ibverbs device is resolved internally via the V1 service locator from the channel's
# enumeration metadata, not a per-operator parameter (confirmed by reading
# roce_receiver_op.hpp directly, not assumed).
#
# --frame-limit has no HSDK5 CountCondition equivalent, so it's implemented with a small native
# Python operator, FrameLimitOp, tapping each receiver's tensor output and counting activations
# independently per link (matching the HSDK4-era original's own per-receiver CountCondition).
#
# Usage:
#   hawk_dual_link_player.py [OPTIONS]
#
# Options:
#   --camera-mode N       AR0234 mode (default: 0 = 1920x1200@60fps)
#   --headless            Run without opening a window (still renders, presents nowhere)
#   --fullscreen          Run in fullscreen mode
#   --frame-limit N       Number of frames to capture per link (default: unlimited)
#   --hololink IP         IP address of Hololink board (default: 192.168.0.2)
#   --log-level N         Python logging level (default: 20 = INFO)
#   --window-height N     Total window height in pixels, halved per link (default: 1080)
#   --window-width N      Total window width in pixels, halved per link (default: 1920)
#   --title TITLE         Unused; preserved for CLI-surface fidelity with the HSDK4-era
#                          original and examples/hawk_dual_link_player.cpp (neither uses it
#                          either).
#   --sensor {left,right} Which AR0234 (left or right) to stream from each Hawk module
#   --exposure N          Sensor exposure register value (default: 0x02DC)
#   --pattern N           Test-pattern register value (default: sensor's own default)
#   --skip-setup          Skip board/deserializer/sensor bring-up (assume already configured)
#   --skip-reset          Skip the initial board reset (only meaningful without --skip-setup)

import argparse
import logging
import signal
import sys
import threading

import hololink_module.operators
import hololink_module.sensors.ar0234.ar0234_mode as ar0234_mode
import hololink_module.sensors.deserializers.max96716a as max96716a_module
import hololink_module.sensors.hawk as hawk_module
import hololink_module.taurotech_da326 as taurotech_da326_module
import holoscan as hs
import holoscan_vision as hv
import holoscan_visualization as hz  # noqa: F401 -- registers "holoscan.visualization.HolovizOp"
from cuda.bindings import driver as cuda_driver

import hololink as hl  # Only for resolve_generic_geometry_from_dimensions (see its own comment

# in python/hololink/_hololink.cpp: the generic geometry helper for pure-Python sensor drivers
# with no nanobind-bound camera type -- Hawk/Ar0234 here, ECam0M30Tof elsewhere).
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
    """Shared state for --frame-limit: one counter per SIF/link, matching the HSDK4-era
    original's own per-receiver CountCondition (each link independently gated on its own
    frame count, not a combined total)."""

    lock = threading.Lock()
    counts = []
    limit = 0

    @classmethod
    def reset(cls, limit, link_count):
        with cls.lock:
            cls.counts = [0] * link_count
            cls.limit = limit

    @classmethod
    def record(cls, idx):
        """Counts a frame; True once every counter has reached the limit."""
        with cls.lock:
            cls.counts[idx] += 1
            return cls.limit > 0 and all(count >= cls.limit for count in cls.counts)


class FrameLimitOp(hs.Operator):
    """Count captured frames toward --frame-limit for one link; see _FrameLimit."""

    input = hs.Input("input", hs.payload.Tensor)

    link_index = hs.Param(0)

    def setup(self, spec):
        spec.input(self.input, queue_depth=4)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, context):
        self.input.receive_data()
        if _FrameLimit.record(self.link_index):
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


def _compose_pipeline(
    graph,
    idx,
    sif_metadata,
    hawk,
    cu_context,
    cuda_device_ordinal,
    window_title,
    per_window_width,
    per_window_height,
    headless,
    fullscreen,
    frame_limit,
):
    suffix = str(idx)

    # hawk.pixel_format()/bayer_format() return hololink_module.sensors.csi's plain
    # enum.Enum members (confirmed by testing, not assumed): unlike native's own nanobind-bound,
    # nb::is_arithmetic()-enabled PixelFormat/BayerFormat, int() doesn't work on these directly
    # -- use .value.
    pixel_format = hawk.pixel_format()
    bayer_format = hawk.bayer_format()
    bayer_name = bayer_format.name.lower()
    geometry = hl.resolve_generic_geometry_from_dimensions(
        pixel_format.value, hawk._width, hawk._height
    )
    frame_size = (
        geometry["start_byte"]
        + geometry["bytes_per_line"] * geometry["pixel_height"]
        + geometry["trailing_bytes"]
    )

    receiver = graph.op(
        hololink_module.operators.RoceReceiverOp,
        f"receiver_{suffix}",
        metadata=sif_metadata,
        frame_context=int(cu_context),
        device_start=hawk.start,
        device_stop=hawk.stop,
        frame_size=frame_size,
    )

    if frame_limit:
        frame_limit_op = graph.op(FrameLimitOp, f"frame_limit_{suffix}", link_index=idx)
        graph.add_flow(receiver.tensor, frame_limit_op.input, queue_depth=4)

    csi = graph.op(
        hl.operators.CsiToBayerOp,
        f"csi_to_bayer_{suffix}",
        cuda_device_ordinal=cuda_device_ordinal,
        geometry=geometry,
        bayer_format=bayer_name,
    )
    graph.add_flow(receiver.tensor, csi.in_tensor, queue_depth=4)
    graph.add_flow(receiver.metadata, csi.in_metadata, queue_depth=4)

    image_processor = graph.op(
        hl.operators.ImageProcessorOp,
        f"image_processor_{suffix}",
        pixel_format=pixel_format.value,
        pixel_width=geometry["pixel_width"],
        pixel_height=geometry["pixel_height"],
        bayer_format=bayer_format.value,
        optical_black=50,
        cuda_device_ordinal=cuda_device_ordinal,
    )
    graph.add_flow(csi.tensor, image_processor.in_tensor, queue_depth=4)
    graph.add_flow(csi.metadata, image_processor.in_metadata, queue_depth=4)

    demosaic = graph.op(
        hv.operators.BayerDemosaicOp,
        f"bayer_demosaic_{suffix}",
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
            "width": per_window_width,
            "height": per_window_height,
            "format": "rgba8_srgb",
        }
    else:
        sink = {
            "kind": "window",
            "framebuffer_srgb": True,
            "title": f"Holoviz - {window_title}",
            "width": per_window_width,
            "height": per_window_height,
            "fullscreen": fullscreen,
            "close_on_escape": True,
        }
    viz = graph.op(
        "holoscan.visualization.HolovizOp",
        f"holoviz_{suffix}",
        layers=[
            {
                "port_name": "video",
                "kind": "image",
                "format": "rgba16_unorm",
                "max_width": geometry["pixel_width"],
                "max_height": geometry["pixel_height"],
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
        discard = graph.op(DiscardImageOp, f"discard_{suffix}")
        graph.add_flow(
            viz.output,
            discard.input,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
            queue_depth=1,
        )

    tensor_output_paths = [
        f"receiver_{suffix}",
        f"csi_to_bayer_{suffix}",
        f"image_processor_{suffix}",
    ]
    image_output_paths = [f"bayer_demosaic_{suffix}"]
    if headless:
        image_output_paths.append(f"holoviz_{suffix}")
    return tensor_output_paths, image_output_paths


def run(
    hololink_ip,
    camera_mode_int,
    headless,
    fullscreen,
    frame_limit,
    window_height,
    window_width,
    sensor_arg,
    exposure,
    pattern,
    pattern_set,
    skip_setup,
    skip_reset,
):
    cuda_device_ordinal = 0

    print("Initializing CUDA...")
    _cuda_check(cuda_driver.cuInit(0))
    cu_device = _cuda_check(cuda_driver.cuDeviceGet(cuda_device_ordinal))
    cu_context = _cuda_check(cuda_driver.cuDevicePrimaryCtxRetain(cu_device))
    cuda_driver.cuCtxSetCurrent(cu_context)

    # hololink channel(s) setup -- one SIF metadata per link (SIF 0 and SIF 1).
    sif_list = [0, 1]

    print(f"Enumerating HSB board at {hololink_ip}...")
    adapter = hololink_module.Adapter.get_adapter()
    metadata = adapter.wait_for_channel(hololink_ip, 30.0)

    sif_metadatas = []
    for sif in sif_list:
        sif_meta = hololink_module.EnumerationMetadata(metadata)
        adapter.use_sensor(sif_meta, sif)
        sif_metadatas.append(sif_meta)

    holo_iface = hololink_module.HololinkInterfaceV1.get_service(metadata)
    holo_iface.start()
    board = taurotech_da326_module.TauroTechDa326InterfaceV1.get_service(metadata)

    legacy_holo = board.hololink()
    deserializer = max96716a_module.Max96716a(legacy_holo)
    hawk_a = hawk_module.Hawk(legacy_holo, skip_i2c=skip_setup)
    hawk_b = hawk_module.Hawk(legacy_holo, skip_i2c=skip_setup)
    hawk_modules = [hawk_a, hawk_b]
    camera_mode = ar0234_mode.Ar0234_Mode(camera_mode_int)
    for hawk in hawk_modules:
        hawk.set_mode(camera_mode)

    try:
        if not skip_setup:
            if not skip_reset:
                holo_iface.reset()

            # Board Init
            board.release_reset()
            board.power_cycle()
            board.check_power()
            board.setup_clock()

            # Deser sanity check
            dev_id = deserializer.get_register(deserializer.DEV_ID_REG)
            expected_dev_id = deserializer.DEV_ID
            if dev_id != expected_dev_id:
                raise RuntimeError(
                    f"Deserializer mismatch: expected {expected_dev_id}, got {dev_id}"
                )

            # Both of the serializers are at 0x40. Remap link B serializer to 0x41.
            deserializer.enable_link_exclusive(deserializer.GmslLink.LINK_B)
            hawk_b.set_serializer_i2c_address(hawk_b.get_serializer_i2c_address() + 1)
            deserializer.enable_both_links()

            # Serializer sanity check
            for hawk in hawk_modules:
                dev_id = hawk.serializer.get_register(hawk.serializer.DEV_ID_REG)
                expected_dev_id = hawk.serializer.DEV_ID
                if dev_id != expected_dev_id:
                    raise RuntimeError(
                        f"Serializer mismatch: expected {expected_dev_id}, got {dev_id}"
                    )

            # Both of the sensors are at 0x10 and 0x18. Remap hawk_b sensors to 0x11 and 0x1A.
            hawk_b.remap_sensor_addresses(0x11, 0x1A)

            # Sensor sanity check
            for hawk in hawk_modules:
                for sensor in hawk.sensors:
                    dev_id = sensor.get_register(sensor.DEV_ID_REG)
                    if dev_id != sensor.DEV_ID:
                        raise RuntimeError(
                            f"Sensor mismatch at i2c=0x{sensor.get_i2c_address():02X}: "
                            f"expected 0x{sensor.DEV_ID:04X}, got 0x{dev_id:04X}"
                        )

            # Deserializer settings
            deserializer.configure_video_pipe()
            if sensor_arg == "left":
                left = deserializer.stream_id_to_pipe_mapping(
                    deserializer.GmslLink.LINK_A, 0, deserializer.VideoPipe.PIPE_Y
                )
                right = deserializer.stream_id_to_pipe_mapping(
                    deserializer.GmslLink.LINK_B, 0, deserializer.VideoPipe.PIPE_Z
                )
            elif sensor_arg == "right":
                left = deserializer.stream_id_to_pipe_mapping(
                    deserializer.GmslLink.LINK_A, 2, deserializer.VideoPipe.PIPE_Y
                )
                right = deserializer.stream_id_to_pipe_mapping(
                    deserializer.GmslLink.LINK_B, 2, deserializer.VideoPipe.PIPE_Z
                )
            else:
                raise RuntimeError(f"Non-existent config {sensor_arg}")
            deserializer.set_register(deserializer.VIDEO_PIPE_SEL, left | right)

            # Sensor programming
            for hawk in hawk_modules:
                hawk.configure(camera_mode)
                if pattern_set:
                    hawk.test_pattern(pattern)
                hawk.set_exposure_reg(exposure)

        # A first then B, label by sensor side.
        sensor_label = sensor_arg.capitalize()
        window_titles = [f"{sensor_label} A", f"{sensor_label} B"]

        per_window_height = window_height // 2
        per_window_width = window_width // 2

        if frame_limit:
            _FrameLimit.reset(frame_limit, len(sif_list))

        # Build the graph.
        graph = hs.Graph("hawk_dual_link_player")

        tensor_output_paths = []
        image_output_paths = []
        for idx, (sif_metadata, hawk, title) in enumerate(
            zip(sif_metadatas, hawk_modules, window_titles)
        ):
            t, i = _compose_pipeline(
                graph,
                idx,
                sif_metadata,
                hawk,
                cu_context,
                cuda_device_ordinal,
                title,
                per_window_width,
                per_window_height,
                headless,
                fullscreen,
                frame_limit,
            )
            tensor_output_paths += t
            image_output_paths += i

        clock_key = graph.add_clock(hs.RealtimeClock, "clock")
        graph.set_default_clock(clock_key)

        options = hs.CompileOptions()
        for operator_path in tensor_output_paths:
            options.deployment.bind_tensor_output_device(
                operator_path=operator_path,
                output_port="tensor",
                device=hs.execution.DeviceId(cuda_device_ordinal),
            )
        for operator_path in image_output_paths:
            options.deployment.bind_tensor_output_device(
                operator_path=operator_path,
                output_port="output",
                device=hs.execution.DeviceId(cuda_device_ordinal),
            )

        plan = hs.compile(graph, options)
        if not plan.ok:
            print("Graph compilation failed:", file=sys.stderr)
            for diag in plan.diagnostics:
                print(f"  {diag}", file=sys.stderr)
            return 1

        print(f"Capturing, mode={camera_mode.value} -- running (Ctrl+C to stop)...")

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

    sensor_options = {"left", "right"}
    parser = argparse.ArgumentParser()
    modes = ar0234_mode.Ar0234_Mode
    mode_choices = [mode.value for mode in modes]
    mode_help = " ".join([f"{mode.value}:{mode.name}" for mode in modes])
    parser.add_argument(
        "--camera-mode",
        type=int,
        choices=mode_choices,
        default=mode_choices[0],
        help=mode_help,
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--frame-limit", type=int, default=None)
    parser.add_argument("--hololink", default="192.168.0.2")
    parser.add_argument("--log-level", type=int, default=20)
    parser.add_argument("--window-height", type=int, default=1080)
    parser.add_argument("--window-width", type=int, default=1920)
    parser.add_argument(
        "--title",
        default="AR0234",
        help="Unused; preserved for CLI-surface fidelity with the HSDK4-era original.",
    )
    parser.add_argument(
        "--sensor",
        choices=tuple(sensor_options),
        default="left",
        help="Which AR0234 (left or right) to stream from each Hawk module; one window per Hawk.",
    )
    parser.add_argument("--exposure", type=lambda x: int(x, 0), default=0x02DC)
    parser.add_argument("--pattern", type=lambda x: int(x, 0))
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--skip-reset", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)

    return run(
        args.hololink,
        args.camera_mode,
        args.headless,
        args.fullscreen,
        args.frame_limit,
        args.window_height,
        args.window_width,
        args.sensor,
        args.exposure,
        args.pattern,
        args.pattern is not None,
        args.skip_setup,
        args.skip_reset,
    )


if __name__ == "__main__":
    sys.exit(main())
