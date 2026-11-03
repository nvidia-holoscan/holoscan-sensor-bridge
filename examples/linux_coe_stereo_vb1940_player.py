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

# Stereo VB1940 sensors (always both) sharing one board/IP, each received over IEEE 1722/AVTP
# Class-Oriented-Encapsulation (COE) packets via LinuxCoeReceiverOp -- a raw AF_PACKET socket
# bound to one shared network interface, demuxed by per-leg coe_channel (0/1), not UDP/RoCE like
# examples/linux_single_network_stereo_vb1940_player.py. Python port of this repo's own previous
# (HSDK4-era) Python version of this app, via the HSDK5 Python bindings -- there is no .cpp
# sibling. Combines that already-ported sibling's single-network-stereo wiring (per-leg
# data_channel/camera callbacks, one-board hl.use_sensor(0|1) demux, tiled Holoviz layers,
# FrameLimitOp) with
# examples/linux_coe_imx274_player.py's own LinuxCoeReceiverOp usage (see that file's header
# comment for the full rationale behind the operator itself, new in this branch).
#
#   LinuxCoeReceiverOp -> CsiToBayerOp -> ImageProcessorOp -> BayerDemosaicOp -> HolovizOp
#
# The old app's --trigger/--frequency flags are parsed but were never actually wired to
# anything (no Synchronizer/ptp_pps_output call anywhere in its own source, unlike the sibling
# UDP-transport stereo app, which does wire them) -- preserved here exactly as dead, not
# "completed" to match that sibling's different, real feature scope.
#
# Fixed one real bug found while porting: the old app accepted --fullscreen and stored it, but
# never actually passed it to HolovizOp (always rendered windowed) -- now wired into sink.
#
# To receive COE packets, create a VLAN interface tagged for VETH 0xCCC (3276) on top of your
# real network interface, then point --coe-interface at it -- see
# examples/linux_coe_imx274_player.py's header comment for the exact `ip link add` recipe.
#
# --frame-limit has no HSDK5 CountCondition equivalent, so it's implemented with a small native
# Python operator, FrameLimitOp, tapping each leg's receiver tensor output and counting
# activations; the one that brings the last leg to the limit stops the run from the graph, so a
# run that ends first, for example by failing, doesn't leave the app waiting.
#
# Usage:
#   linux_coe_stereo_vb1940_player.py [OPTIONS]
#
# Options:
#   --hololink IP          IP address of Hololink board (default: 192.168.0.2)
#   --camera-mode N         VB1940 mode (default: 0 = 2560x1984@30fps)
#   --coe-interface NAME    Network interface receiving VETH 3276 (0xCCC) COE packets (required)
#   --trigger               Accepted for CLI compatibility; unused (matches the old app)
#   --frequency N           Accepted for CLI compatibility; unused (matches the old app)
#   --exp N                 Exposure duration in lines, 4-65535 (default: 256)
#   --gain N                Analog gain, 0-12 (default: 0)
#   --window-height N      Displayed window height (default: 2160 // 2)
#   --window-width N       Displayed window width (default: int(3840 / 1.5))
#   --title TITLE           Window title (default: "Holoviz - <hololink IP>")
#   --frame-limit N        Number of frames to capture per sensor (default: unlimited)
#   --headless              Run without opening a window (still renders, presents nowhere)
#   --fullscreen            Run in fullscreen mode
#   --log-level N           Python logging level (default: 20 = INFO)

import argparse
import logging
import signal
import sys
import threading
import time

import holoscan as hs
import holoscan_vision as hv
import holoscan_visualization as hz  # noqa: F401 -- registers "holoscan.visualization.HolovizOp"
from cuda.bindings import driver as cuda_driver

import hololink as hl

_BAYER_FORMAT_NAMES = ["bggr", "rggb", "gbrg", "grbg"]
_BAYER_FORMAT_VALUES = {name: index for index, name in enumerate(_BAYER_FORMAT_NAMES)}


def _cuda_check(result):
    err = result[0]
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        _, description = cuda_driver.cuGetErrorString(err)
        raise RuntimeError(f"CUDA error: {description}")
    if len(result) == 2:
        return result[1]
    return result[1:]


class _FrameLimit:
    """Shared state for --frame-limit: every leg must reach the limit before FrameLimitOp stops
    the run."""

    lock = threading.Lock()
    counts = {}
    limit = 0

    @classmethod
    def reset(cls, leg_count, limit):
        with cls.lock:
            cls.counts = {i: 0 for i in range(leg_count)}
            cls.limit = limit

    @classmethod
    def record(cls, leg):
        """Counts a frame; True once every counter has reached the limit."""
        with cls.lock:
            cls.counts[leg] += 1
            return cls._done_locked()

    @classmethod
    def _done_locked(cls):
        return cls.limit > 0 and all(
            count >= cls.limit for count in cls.counts.values()
        )


class FrameLimitOp(hs.Operator):
    """Count one leg's captured frames toward --frame-limit; see _FrameLimit."""

    input = hs.Input("input", hs.payload.Tensor)
    leg = hs.Param(0)

    def setup(self, spec):
        spec.input(self.input, queue_depth=4)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, context):
        self.input.receive_data()
        if _FrameLimit.record(self.leg):
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
    coe_interface,
    exposure,
    gain,
    window_height,
    window_width,
    title,
    frame_limit,
    headless,
    fullscreen,
):
    cuda_device_ordinal = 0
    tensor_names = ["left", "right"]

    print("Initializing CUDA...")
    _cuda_check(cuda_driver.cuInit(0))
    cu_device = _cuda_check(cuda_driver.cuDeviceGet(cuda_device_ordinal))
    cu_context = _cuda_check(cuda_driver.cuDevicePrimaryCtxRetain(cu_device))
    cuda_driver.cuCtxSetCurrent(cu_context)

    print(f"Enumerating HSB board at {hololink_ip}...")
    channels = []
    for sensor_id in (0, 1):
        metadata = hl.find_channel(hololink_ip)
        hl.use_sensor(metadata, sensor_id)
        channels.append(hl.DataChannel(metadata))

    hololink_dev = channels[0].hololink()
    if hololink_dev is not channels[1].hololink():
        raise RuntimeError("Data channels are on different HSB devices.")
    hololink_dev.start()

    cameras = [hl.NativeVb1940Sensor(channel) for channel in channels]

    # Manual sensor-reset sequence -- matches this app's own HSDK4-era source and
    # linux_single_network_stereo_vb1940_player.py's identical bring-up: both sensors share one
    # external clock, so their reset pins are released together (0x3 = both bits set).
    hololink_dev.write_uint32(0x8, 0x0)  # Keep both sensors' RESET at low
    cameras[0].setup_clock()  # this also sets the other camera's clock
    hololink_dev.write_uint32(0x8, 0x3)  # Release RESET for both sensors
    time.sleep(100 / 1000)

    for camera in cameras:
        camera.get_register_32(0x0000)  # DEVICE_MODEL_ID:"S940"(ASCII code:0x53393430)
        camera.get_register_32(0x0734)  # EXT_CLOCK(25MHz = 0x017d7840)
        camera.configure(camera_mode)
        camera.set_analog_gain_reg(gain)
        camera.set_exposure_reg(exposure)

    try:
        cal_eeprom = cameras[0].get_rgb_calibration_data_string()
        logging.info(f"Camera calibration data retrieved successfully: \n{cal_eeprom}")
    except Exception as e:
        logging.warning(f"Failed to read camera calibration data: {e}")

    geometries = [hl.resolve_generic_geometry(camera) for camera in cameras]
    frame_sizes = [
        geometry["start_byte"]
        + geometry["bytes_per_line"] * geometry["pixel_height"]
        + geometry["trailing_bytes"]
        for geometry in geometries
    ]
    if frame_sizes[0] != frame_sizes[1]:
        raise RuntimeError("Both cameras must have the same frame size")

    if frame_limit:
        _FrameLimit.reset(len(cameras), frame_limit)

    # Build the graph.
    graph = hs.Graph("linux_coe_stereo_vb1940_player")

    # Pass 1: declare one image layer per camera, tiled side by side -- the layer list must be
    # complete before HolovizOp is constructed.
    layers = []
    for i, camera in enumerate(cameras):
        layers.append(
            {
                "port_name": f"video_{tensor_names[i]}",
                "kind": "image",
                "format": "rgba16_unorm",
                "max_width": camera.get_width(),
                "max_height": camera.get_height(),
                "view": {
                    "offset_x": 0.5 * i,
                    "offset_y": 0.0,
                    "scale_x": 0.5,
                    "scale_y": 1.0,
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
            "framebuffer_srgb": True,
            "title": title or f"Holoviz - {hololink_ip}",
            "width": window_width,
            "height": window_height,
            "fullscreen": fullscreen,
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

    tensor_output_operators = []

    for i, camera in enumerate(cameras):
        name = tensor_names[i]
        geometry = geometries[i]
        bayer_format = camera.get_bayer_format()
        pixel_format = camera.get_pixel_format()

        rx = graph.op(
            hl.operators.LinuxCoeReceiverOp,
            f"rx_{name}",
            data_channel=channels[i],
            frame_context=int(cu_context),
            device_start=camera.start,
            device_stop=camera.stop,
            frame_size=frame_sizes[i],
            coe_interface=coe_interface,
            pixel_width=camera.get_width(),
            coe_channel=i,
        )

        if frame_limit:
            frame_limit_op = graph.op(FrameLimitOp, f"frame_limit_{name}", leg=i)
            graph.add_flow(rx.tensor, frame_limit_op.input, queue_depth=4)

        csi = graph.op(
            hl.operators.CsiToBayerOp,
            f"csi_to_bayer_{name}",
            cuda_device_ordinal=cuda_device_ordinal,
            geometry=geometry,
            bayer_format=bayer_format,
        )
        graph.add_flow(rx.tensor, csi.in_tensor, queue_depth=4)
        graph.add_flow(rx.metadata, csi.in_metadata, queue_depth=4)

        image_processor = graph.op(
            hl.operators.ImageProcessorOp,
            f"image_processor_{name}",
            pixel_format=pixel_format,
            pixel_width=geometry["pixel_width"],
            pixel_height=geometry["pixel_height"],
            bayer_format=_BAYER_FORMAT_VALUES[bayer_format],
            # Optical black value for vb1940 is 8 for RAW10 (matches this app's own
            # HSDK4-era default).
            optical_black=8,
            cuda_device_ordinal=cuda_device_ordinal,
        )
        graph.add_flow(csi.tensor, image_processor.in_tensor, queue_depth=4)
        graph.add_flow(csi.metadata, image_processor.in_metadata, queue_depth=4)

        demosaic = graph.op(
            hv.operators.BayerDemosaicOp,
            f"bayer_demosaic_{name}",
            bayer_grid_pos=bayer_format,
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

        tensor_output_operators.append(rx.name)
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
    for name in tensor_names:
        options.deployment.bind_tensor_output_device(
            operator_path=f"bayer_demosaic_{name}",
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
        hololink_dev.stop()
        _cuda_check(cuda_driver.cuDevicePrimaryCtxRelease(cu_device))
        return 1

    print(f"Capturing 2 sensors, mode={camera_mode} -- running (Ctrl+C to stop)...")

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
        default=0,
        help="VB1940 mode (default: 0 = 2560x1984@30fps)",
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
        "--hololink",
        default="192.168.0.2",
        help="IP address of Hololink board",
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level to display",
    )
    parser.add_argument(
        "--coe-interface",
        required=True,
        help="Name of interface connected to VETH 3276 (0xCCC).",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=2160 // 2,  # arbitrary default
        help="Set the height of the displayed window",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=int(3840 / 1.5),  # arbitrary default
        help="Set the width of the displayed window",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Set the window title",
    )
    parser.add_argument(
        "--trigger",
        action="store_true",
        help="Run in trigger mode",
    )
    parser.add_argument(
        "--exp",
        type=int,
        default=256,
        help="set EXPOSURE duration in lines, RANGE(4 to 65535). Default line value is 29.70usec",
    )
    parser.add_argument(
        "--gain",
        type=int,
        default=0,
        help="Set Analog Gain, RANGE(0 to 12). Default is 0. Equation is (16/(16-gain)",
    )
    parser.add_argument(
        "--frequency",
        type=int,
        default=30,
        help="VSYNC frequency in Hz (10, 30, 60, 90, 120). Default is 30Hz",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    return run(
        args.hololink,
        args.camera_mode,
        args.coe_interface,
        args.exp,
        args.gain,
        args.window_height,
        args.window_width,
        args.title,
        args.frame_limit,
        args.headless,
        args.fullscreen,
    )


if __name__ == "__main__":
    sys.exit(main())
