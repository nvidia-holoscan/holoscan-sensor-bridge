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

# Hawk module stereo (both AR0234 sensors of one GMSL link, FSYNC-synced) live viewer over a
# single RoCE network, one HolovizOp window per sensor. Python port of this repo's own previous
# (HSDK4-era) Python version of this app, via the HSDK5 Python bindings -- the sensor bring-up
# sequence below matches that original exactly; the receiver/pipeline construction instead
# mirrors this repo's own examples/single_network_stereo_hawk_player.cpp (RoceReceiverOp(module),
# a hololink_module manual-row operator, like linux_hawk_dual_link_player.py's LinuxReceiverOp
# sibling). Python-API mechanics (graph.op()/add_flow(), CompileOptions, hs.run_async,
# constructor-handle kwargs, geometry helper calls) mirror examples/linux_hawk_dual_link_player.py.
#
#   RoceReceiverOp(module) -> SkewProbeOp -> CsiToBayerOp -> ImageProcessorOp -> BayerDemosaicOp
#       -> HolovizOp   (x2: one pipeline per SIF -- left/right AR0234 of the one shared Hawk
#       module/GMSL link, NOT two independent Hawk modules like linux_hawk_dual_link_player.py's
#       two separate links -- both streams share a single RoCE network/IB device, hence "single
#       network stereo")
#
# Only the receiver is a hololink_module.operators.* (manual-row) operator; the downstream
# CsiToBayerOp/ImageProcessorOp are the NATIVE hololink.operators.* operators (same family
# vb1940_player.py/imx274_player.py already use) -- confirmed directly against this app's own
# C++ reference (`graph.op<hololink::operators::CsiToBayerOp>`/`ImageProcessorOp`, not
# `hololink::module::operators::`), not assumed from the receiver's own namespace.
#
# SkewProbeOp is a native-Python pass-through operator (tensor+metadata aligned receive -- see
# linux_imx274_frame_validation.py's own FrameValidationOp for the same OnAligned+Exact{}
# pattern) that records each frame's FPGA-receive timestamp (CaptureMetadata.timestamp_s/
# timestamp_ns) to measure left/right stereo skew -- matches the HSDK4-era original's own
# SkewState/SkewProbe and examples/single_network_stereo_hawk_player.cpp's own SkewState/
# SkewProbeOp.
#
# Unlike examples/single_network_stereo_hawk_player.cpp (which still carries a local
# BayerImageExtractorOp 16-bit->8-bit narrowing shim), this skips that stage -- same
# simplification already applied to this repo's other ported Python examples (e.g.
# vb1940_player.py): BayerDemosaicOp accepts ImageProcessorOp's 16-bit schema::ImageT output
# directly.
#
# hololink_module's Hawk/Ar0234/Max9295d/Max96716a/TauroTechDa326InterfaceV1 sensor driver is
# unchanged; only the receiver (hololink::module::operators::RoceReceiverOp, ported to HSDK5's
# Operator<Params> shape) and the downstream converters (hololink::module::operators::
# CsiToBayerOp/ImageProcessorOp, both auto-registered) changed.
#
# --frame-limit has no HSDK5 CountCondition equivalent, so it's implemented with a small native
# Python operator, FrameLimitOp, tapping each receiver's tensor output and counting activations
# independently per SIF (matching the HSDK4-era original's own per-receiver CountCondition).
#
# Dropped, matching established precedent: --configuration (application.config(), no HSDK5
# equivalent -- examples/example_configuration.yaml is an empty YAML document, so this was
# already a no-op; every port in this repo already drops this flag, e.g. ecam0m30tof_player.py).
#
# NOT verified against real hardware in this environment: RoceReceiverOp(module) needs ibverbs,
# which this verification container doesn't have (HOLOLINK_BUILD_ROCE=OFF).
#
# Usage:
#   single_network_stereo_hawk_player.py [OPTIONS]
#
# Options:
#   --camera-mode N       AR0234 mode (default: 0 = 1920x1200@60fps)
#   --headless            Run without opening a window (still renders, presents nowhere)
#   --fullscreen          Run in fullscreen mode
#   --frame-limit N       Number of frames to capture per SIF (default: unlimited)
#   --hololink IP         IP address of Hololink board (default: 192.168.0.2)
#   --log-level N         Python logging level (default: 20 = INFO)
#   --window-height N     Total window height in pixels, halved per sensor (default: 1080)
#   --window-width N      Total window width in pixels, halved per sensor (default: 1920)
#   --title TEXT          Window title prefix (default: "Hawk Stereo")
#   --channel {A,B}       GMSL link to enable (default: A)
#   --exposure N          Sensor exposure register value (default: 0x02DC)
#   --pattern N           Test-pattern register value (default: sensor's own default)
#   --skip-setup          Skip board/deserializer/sensor bring-up (assume already configured)
#   --skip-reset          Skip the initial board reset (only meaningful without --skip-setup)
#   --disable-sync        Skip VSYNC/FSYNC setup; sensors free-run.

import argparse
import logging
import signal
import sys
import threading
from datetime import timedelta

import capture_metadata
import hololink_module.operators
import hololink_module.sensors.ar0234.ar0234_mode as ar0234_mode
import hololink_module.sensors.deserializers.max96716a as max96716a_module
import hololink_module.sensors.hawk as hawk_module
import hololink_module.taurotech_da326 as taurotech_da326_module
import holoscan as hs
import holoscan_vision as hv
import holoscan_visualization as hz  # noqa: F401 -- registers "holoscan.visualization.HolovizOp"
from cuda.bindings import driver as cuda_driver

import hololink as hl  # Only for resolve_generic_geometry_from_dimensions.
import hololink_module

SIF_LIST = (0, 1)
SIF_NAME = ("left", "right")

# Matches hololink::operators::kMetadataPairingWindow (capture_metadata_utils.hpp): the
# receiver stamps both ports with an identical capture_time, so this bounded-window
# OnAligned+Exact{} pairs them precisely.
_METADATA_PAIRING_WINDOW = timedelta(microseconds=500)


def _cuda_check(result):
    err = result[0]
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        _, description = cuda_driver.cuGetErrorString(err)
        raise RuntimeError(f"CUDA error: {description}")
    if len(result) == 2:
        return result[1]
    return result[1:]


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
    own SkewState and examples/single_network_stereo_hawk_player.cpp's own SkewState exactly.
    """

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
    cu_context,
    cuda_device_ordinal,
    window_title,
    per_window_width,
    per_window_height,
    headless,
    fullscreen,
    frame_limit,
    pixel_format,
    bayer_format,
    geometry,
    frame_size,
):
    suffix = str(idx)
    bayer_name = bayer_format.name.lower()

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

    skew_probe = graph.op(SkewProbeOp, f"skew_probe_{SIF_NAME[idx]}", sif_index=idx)
    graph.add_flow(receiver.tensor, skew_probe.in_tensor, queue_depth=4)
    graph.add_flow(receiver.metadata, skew_probe.in_metadata, queue_depth=4)

    # Native (hololink.operators), not module -- see this file's header comment.
    csi = graph.op(
        hl.operators.CsiToBayerOp,
        f"csi_to_bayer_{suffix}",
        cuda_device_ordinal=cuda_device_ordinal,
        geometry=geometry,
        bayer_format=bayer_name,
    )
    graph.add_flow(skew_probe.tensor, csi.in_tensor, queue_depth=4)
    graph.add_flow(skew_probe.metadata, csi.in_metadata, queue_depth=4)

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
    title,
    channel_arg,
    exposure,
    pattern,
    pattern_set,
    skip_setup,
    skip_reset,
    disable_sync,
):
    cuda_device_ordinal = 0

    print("Initializing CUDA...")
    _cuda_check(cuda_driver.cuInit(0))
    cu_device = _cuda_check(cuda_driver.cuDeviceGet(cuda_device_ordinal))
    cu_context = _cuda_check(cuda_driver.cuDevicePrimaryCtxRetain(cu_device))
    cuda_driver.cuCtxSetCurrent(cu_context)

    print(f"Enumerating HSB board at {hololink_ip}...")
    adapter = hololink_module.Adapter.get_adapter()
    adapter_metadata = adapter.wait_for_channel(hololink_ip, 30)

    holo_iface = hololink_module.HololinkInterfaceV1.get_service(adapter_metadata)
    holo_iface.start()
    board = taurotech_da326_module.TauroTechDa326InterfaceV1.get_service(
        adapter_metadata
    )
    # sensor_holo provides I2C access for sensor/deserializer setup; sensors will be migrated
    # to HololinkInterfaceV1.get_i2c() separately.
    sensor_holo = board.hololink()

    per_channel_metadatas = []
    for sif in SIF_LIST:
        sif_meta = hololink_module.EnumerationMetadata(adapter_metadata)
        adapter.use_sensor(sif_meta, sif)
        per_channel_metadatas.append(sif_meta)

    deserializer = max96716a_module.Max96716a(sensor_holo)
    hawk = hawk_module.Hawk(sensor_holo, skip_i2c=skip_setup)
    camera_mode = ar0234_mode.Ar0234_Mode(camera_mode_int)
    # set_mode() is pure Python-side bookkeeping (dimensions/pixel format for geometry
    # resolution below) -- safe to call unconditionally, even with --skip-setup, matching
    # examples/single_network_stereo_hawk_player.cpp's own unconditional hawk->set_mode() call.
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

            # Deserializer sanity check
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
            left = deserializer.stream_id_to_pipe_mapping(
                link, 0, deserializer.VideoPipe.PIPE_Y
            )
            right = deserializer.stream_id_to_pipe_mapping(
                link, 2, deserializer.VideoPipe.PIPE_Z
            )
            deserializer.set_register(deserializer.VIDEO_PIPE_SEL, left | right)

            # Sensor programming
            hawk.configure(camera_mode, fsync=not disable_sync)

            if pattern_set:
                hawk.test_pattern(pattern)

            hawk.set_exposure_reg(exposure)

            if not disable_sync:
                # For GPIO Pin 8, disable it, and hold it high.
                hawk.serializer.set_register(0x02D6, 0x10)

                # Vsync settings setup
                frequency_hz = 60
                holo_iface.ptp_pps_output().enable(frequency_hz)
                holo_iface.ptp_pps_output().start()

                # todo: refactor ptp_pps_output to do this, instead of having to do it here
                # Enable VSYNC. On the DA326, GPIO Pin 8 is used to route the VSYNC signal.
                holo_iface.or_uint32(0x70000014, 1 << 4)
                holo_iface.and_uint32(0x0000002C, 0xFFFFFEFF)

                # Forward FPGA VSYNC arriving on MFP8 onto GMSL channel 8 for this link.
                deserializer.route_pin_to_gmsl_gpio(link=link, pin=8, tx_id=0x01)

                # Fan FSYNC (GMSL channel 8) out to both AR0234 TRIGGER pins (MFP9, MFP10).
                hawk.serializer.route_gmsl_gpio_to_pin(pin=9, rx_id=0x01)
                hawk.serializer.route_gmsl_gpio_to_pin(pin=10, rx_id=0x01)
            else:
                logging.info("Sync disabled (--disable-sync); sensors free-run.")

        # hawk.pixel_format()/bayer_format() return hololink_module.sensors.csi's plain
        # enum.Enum members (confirmed by testing, not assumed): unlike native's own
        # nanobind-bound, nb::is_arithmetic()-enabled PixelFormat/BayerFormat, int() doesn't
        # work on these directly -- use .value. Both AR0234s are identical and configured
        # identically, so geometry/format are resolved once (matches the C++ reference).
        pixel_format = hawk.pixel_format()
        bayer_format = hawk.bayer_format()
        geometry = hl.resolve_generic_geometry_from_dimensions(
            pixel_format.value, hawk._width, hawk._height
        )
        frame_size = (
            geometry["start_byte"]
            + geometry["bytes_per_line"] * geometry["pixel_height"]
            + geometry["trailing_bytes"]
        )

        window_titles = [f"{title} - {SIF_NAME[idx]}" for idx in SIF_LIST]
        per_window_width = window_width // 2
        per_window_height = window_height // 2

        if frame_limit:
            _FrameLimit.reset(frame_limit, len(SIF_LIST))

        # Build the graph.
        graph = hs.Graph("single_network_stereo_hawk_player")

        tensor_output_paths = []
        image_output_paths = []
        for idx, (sif_metadata, win_title) in enumerate(
            zip(per_channel_metadatas, window_titles, strict=True)
        ):
            t, i = _compose_pipeline(
                graph,
                idx,
                sif_metadata,
                hawk,
                cu_context,
                cuda_device_ordinal,
                win_title,
                per_window_width,
                per_window_height,
                headless,
                fullscreen,
                frame_limit,
                pixel_format,
                bayer_format,
                geometry,
                frame_size,
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
        logging.info("Stereo skew: %s", _SkewObserver.summary())

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
    parser.add_argument("--title", default="Hawk Stereo")
    parser.add_argument("--channel", default="A", choices=("A", "B"))
    parser.add_argument("--exposure", type=lambda x: int(x, 0), default=0x02DC)
    parser.add_argument("--pattern", type=lambda x: int(x, 0))
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--skip-reset", action="store_true")
    parser.add_argument(
        "--disable-sync",
        action="store_true",
        help="Skip VSYNC/FSYNC setup; sensors free-run. For comparison with default enabled.",
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
        args.channel,
        args.exposure,
        args.pattern,
        args.pattern is not None,
        args.skip_setup,
        args.skip_reset,
        args.disable_sync,
    )


if __name__ == "__main__":
    sys.exit(main())
