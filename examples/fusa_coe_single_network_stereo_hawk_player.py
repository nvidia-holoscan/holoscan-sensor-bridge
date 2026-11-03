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

# Hawk module stereo (both AR0234 sensors of one GMSL link, FSYNC-synced) live viewer over FuSa
# CoE, one HolovizOp window per sensor. Python port of this repo's own previous (HSDK4-era)
# Python version of this app, via the HSDK5 Python bindings -- the sensor bring-up sequence
# below matches that original exactly; the receiver/pipeline construction instead mirrors this
# repo's own examples/fusa_coe_single_network_stereo_hawk_player.cpp (FusaCoeCaptureOp(module),
# a hololink_module manual-row operator). Python-API mechanics (graph.op()/add_flow(),
# CompileOptions, hs.run_async, constructor-handle kwargs, geometry helper calls) mirror
# examples/linux_hawk_dual_link_player.py.
#
#   FusaCoeCaptureOp(module) -> SkewProbeOp -> [PackedFormatConverterOp | CsiToBayerOp] ->
#       ImageProcessorOp -> BayerDemosaicOp -> HolovizOp
#       (x2: one pipeline per SIF -- left/right AR0234 of the one shared Hawk module/GMSL link,
#       NOT two independent Hawk modules like linux_hawk_dual_link_player.py's two separate
#       links -- both streams share a single FuSa CoE network path, hence "single network
#       stereo")
#
# Only the receiver (FusaCoeCaptureOp) is a hololink_module.operators.* (manual-row) operator;
# the downstream PackedFormatConverterOp/CsiToBayerOp/ImageProcessorOp are the NATIVE
# hololink.operators.* operators (same family fusa_coe_imx274_player.py already uses) --
# confirmed directly against this app's own C++ reference
# (`graph.op<hololink::operators::{PackedFormatConverterOp,CsiToBayerOp,ImageProcessorOp}>`, not
# `hololink::module::operators::`) and against fusa_coe_imx274_player.py's own already-verified
# usage, not assumed from the receiver's own namespace.
#
# SkewProbeOp is a native-Python pass-through operator (tensor+metadata aligned receive -- see
# linux_imx274_frame_validation.py's own FrameValidationOp for the same OnAligned+Exact{}
# pattern) that records each frame's FPGA-receive timestamp (CaptureMetadata.timestamp_s/
# timestamp_ns) to measure left/right stereo skew -- matches the HSDK4-era original's own
# SkewState/SkewProbe and examples/fusa_coe_single_network_stereo_hawk_player.cpp's own
# SkewState/SkewProbeOp.
#
# Unlike examples/fusa_coe_single_network_stereo_hawk_player.cpp (which still carries a local
# BayerImageExtractorOp 16-bit->8-bit narrowing shim), this skips that stage -- same
# simplification already applied to this repo's other ported Python examples (e.g.
# vb1940_player.py): BayerDemosaicOp accepts ImageProcessorOp's 16-bit schema::ImageT output
# directly.
#
# hololink_module's Hawk/Ar0234/Max9295d/Max96716a/TauroTechDa326InterfaceV1 sensor driver is
# unchanged; only the receiver (hololink::module::operators::FusaCoeCaptureOp, ported to
# HSDK5's Operator<Params> shape) and the downstream converters
# (hololink::module::operators::{CsiToBayerOp,PackedFormatConverterOp,ImageProcessorOp}, all
# auto-registered) changed.
#
# KNOWN GAP vs examples/fusa_coe_single_network_stereo_hawk_player.cpp: that C++ reference (and
# its single-link sibling, fusa_coe_hawk_single_link_player.cpp) additionally programs the
# FPGA's MIPI D-PHY lane count/line rate per SIF via hololink::module::MipiDphyInterfaceV1
# before sensor programming -- a step absent from this app's own HSDK4-era Python original (and
# from every other not-yet-ported Hawk/FUSA Python example in this tree, confirmed by grep).
# MipiDphyInterfaceV1 has no Python binding at all (confirmed: no reference anywhere under
# python/hololink_module/), and adding one is out of scope for a Python-only port, so this step
# is NOT reproduced here. Flagged, not silently dropped: if FUSA CoE capture doesn't lock
# without it on real hardware, this binding gap is why.
#
# --frame-limit has no HSDK5 CountCondition equivalent, so it's implemented with a small native
# Python operator, FrameLimitOp, tapping each capture's tensor output and counting activations
# independently per SIF (matching the HSDK4-era original's own per-receiver CountCondition).
#
# The HSDK4-era original prints the stereo-skew summary via atexit + a SIGINT handler (not just
# a try/finally) because the FuSa worker thread can abort() the process before Python's
# `finally` block runs -- preserved here exactly, with the same multiple-safety-nets rationale.
#
# Usage:
#   fusa_coe_single_network_stereo_hawk_player.py [OPTIONS]
#
# Options:
#   --camera-mode N        AR0234 mode (default: 0 = 1920x1200@60fps)
#   --headless             Run without opening a window (still renders, presents nowhere)
#   --frame-limit N        Number of frames to capture per SIF (default: unlimited)
#   --hololink IP          IP address of Hololink board (default: 192.168.0.2)
#   --log-level N          Python logging level (default: 20 = INFO)
#   --width N              Total window width in pixels, halved per sensor (default: 1920)
#   --height N             Total window height in pixels, halved per sensor (default: 1080)
#   --title TEXT           Window title prefix (default: "Hawk stereo (FuSa CoE)")
#   --channel {A,B}        GMSL link to enable (default: A)
#   --exposure N           Sensor exposure register value (default: 0x02DC)
#   --pattern N            Test-pattern register value (default: sensor's own default)
#   --timeout N            Capture request timeout in milliseconds (default: 1500)
#   --interface NAME       Ethernet interface (default: auto, from enumeration metadata)
#   --skip-reset           Skip the initial board reset
#   --no-packetizer        Capture unpacketized CSI-2 data instead of HSB-packed data
#   --disable-sync         Skip VSYNC/FSYNC setup; sensors free-run.

import argparse
import atexit
import logging
import signal
import sys
import threading
from datetime import timedelta

import capture_metadata
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

SIF_LIST = (0, 1)
SIF_NAME = ("left", "right")

# Matches hololink::operators::kMetadataPairingWindow (capture_metadata_utils.hpp): the
# receiver stamps both ports with an identical capture_time, so this bounded-window
# OnAligned+Exact{} pairs them precisely.
_METADATA_PAIRING_WINDOW = timedelta(microseconds=500)


def _parse_mac(mac_str):
    return [int(x, 16) for x in mac_str.split(":")]


class _FrameLimit:
    """Shared state for --frame-limit: one counter per SIF, matching the HSDK4-era original's
    own per-receiver CountCondition (each SIF independently gated on its own frame count, not
    a combined total)."""

    lock = threading.Lock()
    counts = []
    limit = 0

    @classmethod
    def reset(cls, limit, sif_count):
        with cls.lock:
            cls.counts = [0] * sif_count
            cls.limit = limit

    @classmethod
    def record(cls, idx):
        """Counts a frame; True once every counter has reached the limit."""
        with cls.lock:
            cls.counts[idx] += 1
            return cls.limit > 0 and all(count >= cls.limit for count in cls.counts)


class FrameLimitOp(hs.Operator):
    """Count captured frames toward --frame-limit for one SIF; see _FrameLimit."""

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


class _SkewObserver:
    """Shared state for stereo skew measurement: pairs left/right frames by nearest
    FPGA-receive timestamp (the receiver metadata timestamp_s/timestamp_ns); see SkewProbeOp.
    Classmethod singleton (same shape as _FrameLimit above), since skew_probe_left and
    skew_probe_right computes may run concurrently. Algorithm matches the HSDK4-era original's
    own SkewState and examples/fusa_coe_single_network_stereo_hawk_player.cpp's own SkewState
    exactly."""

    lock = threading.Lock()
    left = []
    right = []
    deltas_ns = []

    HALF_FRAME_NS = 8_333_333  # half of 1/60s

    @classmethod
    def record(cls, label, ts_ns):
        with cls.lock:
            own = cls.left if label == "left" else cls.right
            other = cls.right if label == "left" else cls.left
            if other:
                best_idx = 0
                best_diff = abs(other[0] - ts_ns)
                for i in range(1, len(other)):
                    d = abs(other[i] - ts_ns)
                    if d < best_diff:
                        best_diff = d
                        best_idx = i
                if best_diff <= cls.HALF_FRAME_NS:
                    other_ts = other.pop(best_idx)
                    if label == "right":
                        cls.deltas_ns.append(ts_ns - other_ts)
                    else:
                        cls.deltas_ns.append(other_ts - ts_ns)
                    cutoff = min(ts_ns, other_ts)
                    cls.left[:] = [t for t in cls.left if t >= cutoff]
                    cls.right[:] = [t for t in cls.right if t >= cutoff]
                    return
            own.append(ts_ns)
            if len(own) > 16:
                del own[0]

    @classmethod
    def summary(cls):
        with cls.lock:
            n = len(cls.deltas_ns)
            if n == 0:
                return "no paired frames"
            avg_us = abs(sum(cls.deltas_ns) / n) / 1000.0
            return f"frames={n}, average={avg_us:.2f} us"


class SkewProbeOp(hs.Operator):
    """Pass-through: records each frame's FPGA-receive timestamp into _SkewObserver, then
    forwards tensor+metadata unchanged. Mirrors the C++ reference's own SkewProbeOp (aligned
    tensor+metadata receive, same pairing window as FrameValidationOp elsewhere in this repo).
    """

    in_tensor = hs.Input("in_tensor", hs.payload.Tensor)
    in_metadata = hs.Input("in_metadata", capture_metadata.CaptureMetadata)
    tensor = hs.Output("tensor", hs.payload.Tensor)
    metadata = hs.Output("metadata", capture_metadata.CaptureMetadata)

    sif_index = hs.Param(0)

    def setup(self, spec):
        spec.input(self.in_tensor, queue_depth=4)
        spec.input(self.in_metadata, queue_depth=4)
        spec.output(self.tensor)
        spec.output(self.metadata)

    def contract(self):
        return (
            hs.Contract()
            .trigger(
                hs.OnAligned(
                    ports=[self.in_tensor, self.in_metadata],
                    window=_METADATA_PAIRING_WINDOW,
                )
            )
            .input(self.in_tensor, hs.Exact())
            .input(self.in_metadata, hs.Exact())
        )

    def compute(self, context):
        tensor_sample = self.in_tensor.receive()
        metadata_sample = self.in_metadata.receive()
        if tensor_sample.is_missing or metadata_sample.is_missing:
            return
        frame = metadata_sample.data
        ts_ns = int(frame.timestamp_s) * 1_000_000_000 + int(frame.timestamp_ns)
        _SkewObserver.record(SIF_NAME[self.sif_index], ts_ns)
        self.tensor.forward(tensor_sample)
        self.metadata.forward(metadata_sample)


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
    camera,
    interface_arg,
    timeout_ms,
    no_packetizer,
    window_title,
    per_window_width,
    per_window_height,
    headless,
    frame_limit,
):
    suffix = str(idx)
    label = SIF_NAME[idx]

    # camera.pixel_format()/bayer_format() return hololink_module.sensors.csi's plain
    # enum.Enum members: int() doesn't work on these directly -- use .value for
    # ImageProcessorOp's plain-int fields, .name.lower() for the string-enum fields
    # FusaCoeCaptureOp/CsiToBayerOp/PackedFormatConverterOp expect.
    pixel_format = camera.pixel_format()
    bayer_format = camera.bayer_format()
    bayer_name = bayer_format.name.lower()

    # The packetizer only has programs for RAW_10 and RAW_12, so an 8-bit mode is always
    # captured unpacketized -- matches the HSDK4-era original exactly.
    packetizer_enabled = (
        not no_packetizer and pixel_format != csi_module.PixelFormat.RAW_8
    )

    geometry = hololink_module.resolve_fusa_coe_geometry_from_dimensions(
        pixel_format.value, camera._width, camera._height, packetizer_enabled
    )

    interface = interface_arg or sif_metadata.get("interface", "")
    mac_addr = _parse_mac(sif_metadata.get("mac_id", "00:00:00:00:00:00"))

    # device=hawk (not the per-sensor camera): the whole Hawk module's start()/stop() is
    # shared across both SIFs' captures, matching both the HSDK4-era original's own
    # `device=self._hawk` and the C++ reference's `[hawk] { hawk->start(); }` capture.

    capture = graph.op(
        hololink_module.operators.FusaCoeCaptureOp,
        f"fusa_coe_capture_{suffix}",
        metadata=sif_metadata,
        device_start=hawk.start,
        device_stop=hawk.stop,
        interface=interface,
        mac_addr=mac_addr,
        timeout=timeout_ms,
        pixel_format=pixel_format.name.lower(),
        bayer_format=bayer_name,
        packetizer_enabled=packetizer_enabled,
        geometry=geometry,
    )

    if frame_limit:
        frame_limit_op = graph.op(FrameLimitOp, f"frame_limit_{suffix}", link_index=idx)
        graph.add_flow(capture.tensor, frame_limit_op.input, queue_depth=4)

    skew_probe = graph.op(SkewProbeOp, f"skew_probe_{label}", sif_index=idx)
    graph.add_flow(capture.tensor, skew_probe.in_tensor, queue_depth=4)
    graph.add_flow(capture.metadata, skew_probe.in_metadata, queue_depth=4)

    # Which converter is correct depends on the packetizer: enabled, the buffer holds
    # HSB-packed data; disabled, it is still CSI-2 as the sensor sent it (minus the header,
    # which FusaCoeCaptureOp always strips -- hence header_included=False below). Both are the
    # NATIVE hololink.operators (not module) -- see this file's header comment.
    if packetizer_enabled:
        converter_name = f"packed_format_converter_{suffix}"
        converter = graph.op(
            hl.operators.PackedFormatConverterOp,
            converter_name,
            cuda_device_ordinal=0,
            bayer_format=bayer_name,
            geometry=geometry,
        )
    else:
        converter_name = f"csi_to_bayer_{suffix}"
        converter = graph.op(
            hl.operators.CsiToBayerOp,
            converter_name,
            cuda_device_ordinal=0,
            geometry=geometry,
            bayer_format=bayer_name,
            header_included=False,
        )
    graph.add_flow(skew_probe.tensor, converter.in_tensor, queue_depth=4)
    graph.add_flow(skew_probe.metadata, converter.in_metadata, queue_depth=4)

    image_processor = graph.op(
        hl.operators.ImageProcessorOp,
        f"image_processor_{suffix}",
        pixel_format=pixel_format.value,
        pixel_width=geometry["pixel_width"],
        pixel_height=geometry["pixel_height"],
        bayer_format=bayer_format.value,
        optical_black=50,
        cuda_device_ordinal=0,
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
            # The HSDK4-era original has no --fullscreen flag for this app; matches the C++
            # reference's own hardcoded .fullscreen = false.
            "fullscreen": False,
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
    frame_limit,
    width,
    height,
    title,
    channel_arg,
    exposure,
    pattern,
    pattern_set,
    timeout_ms,
    interface_arg,
    skip_reset,
    no_packetizer,
    disable_sync,
):
    print(f"Enumerating HSB board at {hololink_ip}...")
    adapter = hololink_module.Adapter.get_adapter()
    adapter_metadata = adapter.wait_for_channel(hololink_ip, 30)

    sif_metadatas = []
    for sif in SIF_LIST:
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

    cameras = list(hawk.sensors)

    # Always print the skew on exit, regardless of how the process winds down (Ctrl+C,
    # frame-limit, FuSa worker-thread shutdown error). Multiple safety nets because the C++
    # worker thread can abort() the process before Python's `finally` block runs -- matches
    # the HSDK4-era original exactly.
    _printed = [False]

    def _print_skew():
        if _printed[0]:
            return
        _printed[0] = True
        print(f"Stereo skew: {_SkewObserver.summary()}", flush=True)

    atexit.register(_print_skew)

    def _sigint_handler(_signum, _frame):
        _print_skew()
        # Re-raise default so Holoscan's own interrupt handler still runs.
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        sys.exit(130)

    signal.signal(signal.SIGINT, _sigint_handler)
    # SIGTERM (kill, docker stop) the same way, so the skew is printed and the board is stopped.
    signal.signal(signal.SIGTERM, _sigint_handler)

    try:
        if not skip_reset:
            holo_iface.reset()

        board.release_reset()
        board.power_cycle()
        board.check_power()
        board.setup_clock()

        dev_id = deserializer.get_register(deserializer.DEV_ID_REG)
        if dev_id != deserializer.DEV_ID:
            raise RuntimeError(
                f"Deserializer mismatch: expected {deserializer.DEV_ID}, got {dev_id}"
            )

        link = deserializer.GmslLink[f"LINK_{channel_arg}"]
        deserializer.enable_link_exclusive(link)

        dev_id = hawk.serializer.get_register(hawk.serializer.DEV_ID_REG)
        if dev_id != hawk.serializer.DEV_ID:
            raise RuntimeError(
                f"Serializer mismatch: expected {hawk.serializer.DEV_ID}, got {dev_id}"
            )

        for sensor in hawk.sensors:
            dev_id = sensor.get_register(sensor.DEV_ID_REG)
            if dev_id != sensor.DEV_ID:
                raise RuntimeError(
                    f"Sensor mismatch at i2c=0x{sensor.get_i2c_address():02X}: "
                    f"expected 0x{sensor.DEV_ID:04X}, got 0x{dev_id:04X}"
                )

        deserializer.configure_video_pipe()
        left_map = deserializer.stream_id_to_pipe_mapping(
            link, 0, deserializer.VideoPipe.PIPE_Y
        )
        right_map = deserializer.stream_id_to_pipe_mapping(
            link, 2, deserializer.VideoPipe.PIPE_Z
        )
        deserializer.set_register(deserializer.VIDEO_PIPE_SEL, left_map | right_map)

        # Sensor programming. hawk.configure() calls each sensor's configure_camera(), which
        # itself calls set_mode() first -- so pixel_format()/_width/_height below are already
        # correct once this returns; no separate set_mode() call needed (matches both the
        # HSDK4-era original, whose own explicit per-camera set_mode() call in compose() is
        # redundant with this, and the C++ reference, which has no explicit call at all).
        hawk.configure(camera_mode, fsync=not disable_sync)
        if pattern_set:
            hawk.test_pattern(pattern)
        hawk.set_exposure_reg(exposure)

        if not disable_sync:
            hawk.serializer.set_register(0x02D6, 0x10)
            holo_iface.ptp_pps_output().enable(60)
            holo_iface.ptp_pps_output().start()
            holo_iface.or_uint32(0x70000014, 1 << 4)
            holo_iface.and_uint32(0x0000002C, 0xFFFFFEFF)
            deserializer.route_pin_to_gmsl_gpio(link=link, pin=8, tx_id=0x01)
            hawk.serializer.route_gmsl_gpio_to_pin(pin=9, rx_id=0x01)
            hawk.serializer.route_gmsl_gpio_to_pin(pin=10, rx_id=0x01)
        else:
            logging.info("Sync disabled (--disable-sync); sensors free-run.")

        window_titles = [f"{title} - left", f"{title} - right"]
        per_w = width // 2
        per_h = height // 2

        if frame_limit:
            _FrameLimit.reset(frame_limit, len(SIF_LIST))

        # Build the graph.
        graph = hs.Graph("fusa_coe_single_network_stereo_hawk_player")

        tensor_output_paths = []
        image_output_paths = []
        for idx, (sif_metadata, camera, win_title) in enumerate(
            zip(sif_metadatas, cameras, window_titles, strict=True)
        ):
            t, i = _compose_pipeline(
                graph,
                idx,
                sif_metadata,
                hawk,
                camera,
                interface_arg,
                timeout_ms,
                no_packetizer,
                win_title,
                per_w,
                per_h,
                headless,
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
                device=hs.execution.DeviceId(0),
            )
        for operator_path in image_output_paths:
            options.deployment.bind_tensor_output_device(
                operator_path=operator_path,
                output_port="output",
                device=hs.execution.DeviceId(0),
            )

        plan = hs.compile(graph, options)
        if not plan.ok:
            print("Graph compilation failed:", file=sys.stderr)
            for diag in plan.diagnostics:
                print(f"  {diag}", file=sys.stderr)
            return 1

        print(f"Capturing, mode={camera_mode.value} -- running (Ctrl+C to stop)...")

        with hs.run_async(plan) as session:
            # Ctrl+C and SIGTERM go through _sigint_handler, whose sys.exit() leaves this block
            # and so stops the session cooperatively, which stops the camera.
            session.wait()
        if session.diagnostics:
            print(session.summary(), file=sys.stderr)
        failed = session.termination == hs.RunTermination.FAILED
    finally:
        holo_iface.stop()
        _print_skew()

    if failed:
        return 1
    print("Done.")
    return 0


def main():
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
    parser.add_argument("--frame-limit", type=int, default=None)
    parser.add_argument("--hololink", default="192.168.0.2")
    parser.add_argument("--log-level", type=int, default=20)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--title", default="Hawk stereo (FuSa CoE)")
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
    parser.add_argument(
        "--disable-sync",
        action="store_true",
        help="Skip VSYNC/FSYNC setup; sensors free-run.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    return run(
        args.hololink,
        args.camera_mode,
        args.headless,
        args.frame_limit,
        args.width,
        args.height,
        args.title,
        args.channel,
        args.exposure,
        args.pattern,
        args.pattern is not None,
        args.timeout,
        args.interface,
        args.skip_reset,
        args.no_packetizer,
        args.disable_sync,
    )


if __name__ == "__main__":
    sys.exit(main())
