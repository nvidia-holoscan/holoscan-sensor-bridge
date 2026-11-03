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

# One Hawk module (one GMSL link, --sensor left|right|both) live viewer over FuSa CoE. Python
# port of this repo's own previous (HSDK4-era) Python version of this app, via the HSDK5 Python
# bindings -- the sensor bring-up sequence below matches that original exactly; the receiver/
# pipeline construction instead mirrors this repo's own
# examples/fusa_coe_hawk_single_link_player.cpp. Unlike that C++ file (which still carries a
# local BayerImageExtractorOp 16-bit->8-bit narrowing shim), this skips that stage -- same
# simplification already applied to this repo's other ported Python examples (e.g.
# linux_hawk_dual_link_player.py, vb1940_player.py): BayerDemosaicOp accepts ImageProcessorOp's
# 16-bit schema::ImageT output directly.
#
#   FusaCoeCaptureOp(module) -> [PackedFormatConverterOp(module) | CsiToBayerOp(module)] ->
#       ImageProcessorOp(module) -> BayerDemosaicOp -> HolovizOp
#       (x1, or x2 if --sensor both -- one window per sensor)
#
# The unpack stage depends on the sensor mode's pixel format: RAW_8 modes are always captured
# unpacketized (the HSB packetizer only has programs for RAW_10/RAW_12), so those use
# CsiToBayerOp(module); RAW_10/RAW_12 modes default to packetized capture (override with
# --no-packetizer), which uses PackedFormatConverterOp(module) instead.
#
# hololink_module's Hawk/Ar0234/Max9295d/Max96716a/TauroTechDa326InterfaceV1 sensor driver is
# unchanged; only the receiver (hololink::module::operators::FusaCoeCaptureOp, ported to HSDK5's
# Operator<Params> shape) and the downstream converters (hololink::module::operators::
# {CsiToBayerOp,PackedFormatConverterOp,ImageProcessorOp}, all auto-registered in Phase 1 of the
# same effort) changed. Geometry is resolved via hololink_module.
# resolve_fusa_coe_geometry_from_dimensions() (64-byte HSB line alignment) -- the module-tree,
# scalar-dimensioned equivalent of the C++ reference's FusaCoeCaptureOp::make_geometry_adapter()
# dance, needing no live camera.configure_converter() call.
#
# The C++ reference also programs the CPNX MIPI D-PHY lanes directly (hololink::module::
# MipiDphyInterfaceV1), a step the HSDK4-era original never had and that has no Python binding
# (MipiDphyInterfaceV1 is a C++-only service interface -- confirmed by grep, every usage of it
# in this repo is in a .cpp file). Omitted here, matching the HSDK4-era original's own scope.
#
# Like the HSDK4-era original (and unlike this app's RoCE/Linux siblings), there is no
# --skip-setup here -- board/deserializer/sensor bring-up always runs; only the initial
# Hololink reset is optional (--skip-reset). There is also no CUDA context to retain here:
# unlike LinuxReceiverOp/RoceReceiverOp(module), FusaCoeCaptureParams(module) carries no
# frame_context field at all (confirmed by reading fusa_coe_capture_op.hpp) -- matching both
# the HSDK4-era original and the C++ sibling, neither of which calls cuInit()/
# cuDevicePrimaryCtxRetain() either.
#
# --interface was accepted but never actually applied in the HSDK4-era original (always
# overridden by sif_metadata.get("interface", "")); the C++ sibling fixes this (uses
# --interface when given, falls back to metadata otherwise), and this port follows the C++
# sibling's fix so the two language ports of this app agree.
#
# --frame-limit has no HSDK5 CountCondition equivalent, so it's implemented with a small native
# Python operator, FrameLimitOp, tapping each capture's tensor output and counting activations
# independently per sensor (matching the HSDK4-era original's own per-receiver CountCondition).
#
# Usage:
#   fusa_coe_hawk_single_link_player.py [OPTIONS]
#
# Options:
#   --camera-mode N             AR0234 mode (default: 0 = 1920x1200@60fps)
#   --headless                  Run without opening a window (still renders, presents nowhere)
#   --fullscreen                Run in fullscreen mode
#   --frame-limit N             Number of frames to capture per sensor (default: unlimited)
#   --hololink IP                IP address of Hololink board (default: 192.168.0.2)
#   --log-level N                Python logging level (default: 20 = INFO)
#   --window-height N            Window height in pixels, halved if --sensor both (default: 1080)
#   --window-width N             Window width in pixels, halved if --sensor both (default: 1920)
#   --title TITLE                Window title prefix (default: "AR0234 (FuSa CoE)")
#   --sensor {left,right,both}   Which sensor(s) to stream; 'both' opens one window per sensor
#   --channel {A,B}              GMSL link to enable (default: A)
#   --exposure N                 Sensor exposure register value (default: 0x02DC)
#   --pattern N                  Test-pattern register value (default: sensor's own default)
#   --timeout N                  Capture request timeout in milliseconds (default: 1500)
#   --interface NAME              Ethernet interface (default: auto, from board metadata)
#   --skip-reset                  Skip the initial board reset
#   --no-packetizer                Capture unpacketized CSI-2 data instead of HSB-packed data

import argparse
import logging
import signal
import sys
import threading

import hololink_module.operators
import hololink_module.sensors.ar0234.ar0234_mode as ar0234_mode
import hololink_module.sensors.csi as csi_module
import hololink_module.sensors.deserializers.max96716a as max96716a_module
import hololink_module.sensors.hawk as hawk_module
import hololink_module.taurotech_da326 as taurotech_da326_module
import holoscan as hs
import holoscan_vision as hv
import holoscan_visualization as hz  # noqa: F401 -- registers "holoscan.visualization.HolovizOp"

import hololink as hl  # Native downstream operators.
import hololink_module

# Intend to map sensor left to SIF 0 and sensor right to SIF 1 regardless of link chosen.
SENSOR_ARG_TO_SIF = {"left": (0,), "right": (1,), "both": (0, 1)}
SIF_TO_NAME = {0: "left", 1: "right"}


def _parse_mac(mac_str):
    return [int(x, 16) for x in mac_str.split(":")]


class _FrameLimit:
    """Shared state for --frame-limit: one counter per sensor, matching the HSDK4-era
    original's own per-receiver CountCondition (each sensor independently gated on its own
    frame count, not a combined total)."""

    lock = threading.Lock()
    counts = []
    limit = 0

    @classmethod
    def reset(cls, limit, sensor_count):
        with cls.lock:
            cls.counts = [0] * sensor_count
            cls.limit = limit

    @classmethod
    def record(cls, idx):
        """Counts a frame; True once every counter has reached the limit."""
        with cls.lock:
            cls.counts[idx] += 1
            return cls.limit > 0 and all(count >= cls.limit for count in cls.counts)


class FrameLimitOp(hs.Operator):
    """Count captured frames toward --frame-limit for one sensor; see _FrameLimit."""

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
    camera,
    hawk,
    cuda_device_ordinal,
    window_title,
    per_window_width,
    per_window_height,
    headless,
    fullscreen,
    frame_limit,
    timeout,
    interface_override,
    no_packetizer,
):
    suffix = str(idx)

    # camera.pixel_format()/bayer_format() return hololink_module.sensors.csi's plain
    # enum.Enum members (confirmed by testing, not assumed): unlike native's own nanobind-bound,
    # nb::is_arithmetic()-enabled PixelFormat/BayerFormat, int() doesn't work on these directly
    # -- use .value for ImageProcessorOp, but FusaCoeCaptureOp's own pixel_format/bayer_format
    # kwargs take string names instead (.name.lower()).
    pixel_format = camera.pixel_format()
    bayer_format = camera.bayer_format()
    bayer_name = bayer_format.name.lower()

    # The packetizer only has programs for RAW_10 and RAW_12, so an 8-bit mode is always
    # captured unpacketized.
    packetizer_enabled = (
        not no_packetizer and pixel_format != csi_module.PixelFormat.RAW_8
    )

    geometry = hololink_module.resolve_fusa_coe_geometry_from_dimensions(
        pixel_format.value, camera._width, camera._height, packetizer_enabled
    )

    # Same physical NIC interface/MAC for every sensor on this board.
    interface = (
        interface_override if interface_override else sif_metadata.get("interface", "")
    )
    mac_addr = _parse_mac(sif_metadata.get("mac_id", "00:00:00:00:00:00"))

    # device start/stop go through the shared hawk (both AR0234s live on it), matching the
    # HSDK4-era original's own device=self._hawk -- not the per-sensor camera used above for
    # format/geometry queries.

    capture = graph.op(
        hololink_module.operators.FusaCoeCaptureOp,
        f"fusa_coe_capture_{suffix}",
        metadata=sif_metadata,
        device_start=hawk.start,
        device_stop=hawk.stop,
        geometry=geometry,
        interface=interface,
        mac_addr=mac_addr,
        timeout=timeout,
        pixel_format=pixel_format.name.lower(),
        bayer_format=bayer_name,
        packetizer_enabled=packetizer_enabled,
    )

    if frame_limit:
        frame_limit_op = graph.op(FrameLimitOp, f"frame_limit_{suffix}", link_index=idx)
        graph.add_flow(capture.tensor, frame_limit_op.input, queue_depth=4)

    # Which converter is correct depends on the packetizer: enabled, the buffer holds
    # HSB-packed data; disabled, it is still CSI-2 exactly as the sensor sent it (minus the
    # header, which FusaCoeCaptureOp already strips before emitting -- hence header_included
    # =False below, unlike this app's RoCE/Linux siblings' CsiToBayerOp, which still needs it).
    if packetizer_enabled:
        converter = graph.op(
            hl.operators.PackedFormatConverterOp,
            f"packed_format_converter_{suffix}",
            cuda_device_ordinal=cuda_device_ordinal,
            bayer_format=bayer_name,
            geometry=geometry,
        )
        converter_name = f"packed_format_converter_{suffix}"
    else:
        converter = graph.op(
            hl.operators.CsiToBayerOp,
            f"csi_to_bayer_{suffix}",
            cuda_device_ordinal=cuda_device_ordinal,
            geometry=geometry,
            bayer_format=bayer_name,
            header_included=False,
        )
        converter_name = f"csi_to_bayer_{suffix}"
    graph.add_flow(capture.tensor, converter.in_tensor, queue_depth=4)
    graph.add_flow(capture.metadata, converter.in_metadata, queue_depth=4)

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
    graph.add_flow(converter.tensor, image_processor.in_tensor, queue_depth=4)
    graph.add_flow(converter.metadata, image_processor.in_metadata, queue_depth=4)

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
                # Layer limits describe the sensor image, independently of the window size.
                "max_width": camera._width,
                "max_height": camera._height,
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
        f"fusa_coe_capture_{suffix}",
        converter_name,
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
    title,
    sensor_arg,
    channel_arg,
    exposure,
    pattern,
    timeout,
    interface_override,
    skip_reset,
    no_packetizer,
):
    cuda_device_ordinal = 0

    # hololink channel(s) setup
    sif_list = SENSOR_ARG_TO_SIF[sensor_arg]

    print(f"Enumerating HSB board at {hololink_ip}...")
    adapter = hololink_module.Adapter.get_adapter()
    adapter_metadata = adapter.wait_for_channel(hololink_ip, 30.0)

    sif_metadatas = []
    for sif in sif_list:
        sif_meta = hololink_module.EnumerationMetadata(adapter_metadata)
        adapter.use_sensor(sif_meta, sif)
        sif_metadatas.append(sif_meta)

    holo_iface = hololink_module.HololinkInterfaceV1.get_service(adapter_metadata)
    holo_iface.start()
    board = taurotech_da326_module.TauroTechDa326InterfaceV1.get_service(
        adapter_metadata
    )
    legacy_holo = board.hololink()
    deserializer = max96716a_module.Max96716a(legacy_holo)
    hawk = hawk_module.Hawk(legacy_holo)
    camera_mode = ar0234_mode.Ar0234_Mode(camera_mode_int)
    hawk.set_mode(camera_mode)

    cameras = [hawk.sensors[sif] for sif in sif_list]

    try:
        if not skip_reset:
            holo_iface.reset()

        # Board Init
        board.release_reset()
        board.power_cycle()
        board.check_power()
        board.setup_clock()

        # Deser sanity check
        dev_id = deserializer.get_register(deserializer.DEV_ID_REG)
        if dev_id != deserializer.DEV_ID:
            raise RuntimeError(
                f"Deserializer mismatch: expected {deserializer.DEV_ID}, got {dev_id}"
            )

        # Only enable the link that we care about
        link = deserializer.GmslLink[f"LINK_{channel_arg}"]
        deserializer.enable_link_exclusive(link)

        # Serializer sanity check
        dev_id = hawk.serializer.get_register(hawk.serializer.DEV_ID_REG)
        if dev_id != hawk.serializer.DEV_ID:
            raise RuntimeError(
                f"Serializer mismatch: expected {hawk.serializer.DEV_ID}, got {dev_id}"
            )

        # Sensor sanity check
        for sensor in hawk.sensors:
            dev_id = sensor.get_register(sensor.DEV_ID_REG)
            if dev_id != sensor.DEV_ID:
                raise RuntimeError(
                    f"Sensor mismatch at i2c=0x{sensor.get_i2c_address():02X}: "
                    f"expected 0x{sensor.DEV_ID:04X}, got 0x{dev_id:04X}"
                )

        # Deserializer settings
        deserializer.configure_video_pipe()
        left_map = deserializer.stream_id_to_pipe_mapping(
            link, 0, deserializer.VideoPipe.PIPE_Y
        )
        right_map = deserializer.stream_id_to_pipe_mapping(
            link, 2, deserializer.VideoPipe.PIPE_Z
        )
        deserializer.set_register(deserializer.VIDEO_PIPE_SEL, left_map | right_map)

        # Sensor programming
        hawk.configure(camera_mode)
        if pattern is not None:
            hawk.test_pattern(pattern)
        hawk.set_exposure_reg(exposure)

        window_titles = [f"{title} - {SIF_TO_NAME[sif]}" for sif in sif_list]

        per_window_width = window_width
        per_window_height = window_height
        if len(sif_list) > 1:
            per_window_width //= 2
            per_window_height //= 2

        if frame_limit:
            _FrameLimit.reset(frame_limit, len(sif_list))

        # Build the graph.
        graph = hs.Graph("fusa_coe_hawk_single_link_player")

        tensor_output_paths = []
        image_output_paths = []
        for idx, (sif_metadata, camera, win_title) in enumerate(
            zip(sif_metadatas, cameras, window_titles, strict=True)
        ):
            t, i = _compose_pipeline(
                graph,
                idx,
                sif_metadata,
                camera,
                hawk,
                cuda_device_ordinal,
                win_title,
                per_window_width,
                per_window_height,
                headless,
                fullscreen,
                frame_limit,
                timeout,
                interface_override,
                no_packetizer,
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

        print(
            f"Capturing {len(cameras)} sensor(s), mode={camera_mode.value} "
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
        holo_iface.stop()

    if failed:
        return 1
    print("Done.")
    return 0


def main():
    # Like Ctrl+C, SIGTERM (kill, docker stop) raises KeyboardInterrupt, so the session stops
    # cooperatively and the camera and the board are stopped on the way out.
    signal.signal(signal.SIGTERM, signal.default_int_handler)

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
    parser.add_argument("--title", default="AR0234 (FuSa CoE)")
    parser.add_argument(
        "--sensor",
        choices=tuple(SENSOR_ARG_TO_SIF.keys()),
        default="left",
        help="Which sensor(s) to stream. 'both' opens one window per sensor.",
    )
    parser.add_argument("--channel", default="A", choices=("A", "B"))
    parser.add_argument("--exposure", type=lambda x: int(x, 0), default=0x02DC)
    parser.add_argument("--pattern", type=lambda x: int(x, 0))
    parser.add_argument("--timeout", type=int, default=1500)
    parser.add_argument("--interface", default=None)
    parser.add_argument("--skip-reset", action="store_true")
    parser.add_argument(
        "--no-packetizer",
        action="store_true",
        help="Capture unpacketized CSI-2 data instead of HSB-packed data",
    )
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
        args.title,
        args.sensor,
        args.channel,
        args.exposure,
        args.pattern,
        args.timeout,
        args.interface,
        args.skip_reset,
        args.no_packetizer,
    )


if __name__ == "__main__":
    sys.exit(main())
