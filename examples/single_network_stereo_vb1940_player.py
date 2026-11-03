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

# Stereo VB1940 sensors (always both -- this app has no single-sensor mode) via
# RoceReceiverOp, tiled side by side in one Holoviz window. Python port of this repo's own
# previous (HSDK4-era) Python version of this app, via the HSDK5 Python bindings; the receiver
# construction pattern, VSYNC trigger-mode wiring, and manual sensor-reset sequence below also
# match examples/single_network_stereo_vb1940_player.cpp. Unlike that C++ file (which still
# carries a local BayerImageExtractorOp 16-bit->8-bit narrowing shim), this skips that stage --
# same simplification already applied to this repo's other ported Python examples:
# BayerDemosaicOp accepts ImageProcessorOp's 16-bit schema::ImageT output directly.
#
#   RoceReceiverOp -> CsiToBayerOp -> ImageProcessorOp -> BayerDemosaicOp -> HolovizOp
#
# --print-time (both the old Python app and the C++ sibling have a real, working version of
# this) is NOT ported: it needs a native Python operator with a port typed
# hololink::CaptureMetadataT, and that FlatBuffers schema has no Python-importable stub
# generated anywhere in this build -- confirmed by checking the build tree, not assumed. Adding
# one is real, standalone infrastructure work out of scope for this port; flagged here rather
# than silently dropped. Every other CLI option both sources agree on (--trigger/--exp/--gain
# included) is ported.
#
# --frame-limit is the frame_limit parameter of each leg's receiver: the first receiver that
# reaches the limit ends the run. HSDK5 has no CountCondition; this replaces the two
# CountConditions of the old app.
#
# Usage:
#   single_network_stereo_vb1940_player.py [OPTIONS]
#
# Options:
#   --hololink IP          IP address of Hololink board (default: 192.168.0.2)
#   --camera-mode N         VB1940 mode (default: 0 = 2560x1984@30fps)
#   --ibv-name NAME         IBV device to use (default: the device on the board's network interface)
#   --ibv-port N            Port number of IBV device (default: 1)
#   --trigger               Run in trigger mode for camera synchronization
#   --frequency N           VSYNC frequency in Hz when --trigger is set (default: 30)
#   --exp N                 Exposure duration in lines, 4-65535 (default: 256)
#   --gain N                Analog gain, 0-12 (default: 0)
#   --window-height N      Displayed window height (default: 2160 // 4)
#   --window-width N       Displayed window width (default: 3840 // 3)
#   --title TITLE           Window title (default: "Holoviz - <hololink IP>")
#   --frame-limit N        Number of frames to capture per sensor (default: unlimited)
#   --headless              Run without opening a window (still renders, presents nowhere)
#   --log-level N           Python logging level (default: 20 = INFO)

import argparse
import logging
import sys
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
    ibv_name,
    ibv_port,
    trigger,
    frequency,
    exposure,
    gain,
    window_height,
    window_width,
    title,
    frame_limit,
    headless,
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
    hololink_dev.reset()

    vsync = hl.Synchronizer.null_synchronizer()
    if trigger:
        vsync = hololink_dev.ptp_pps_output(frequency)

    cameras = [hl.NativeVb1940Sensor(channel, vsync) for channel in channels]

    # Manual sensor-reset sequence -- matches single_network_stereo_vb1940_player.cpp: both
    # sensors share one external clock, so their reset pins are released together (0x3 = both
    # bits set).
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

    # Build the graph.
    graph = hs.Graph("single_network_stereo_vb1940_player")

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
            "title": title or f"Holoviz - {hololink_ip}",
            "width": window_width,
            "height": window_height,
            "fullscreen": False,
            "close_on_escape": True,
            "framebuffer_srgb": True,
        }
    viz = graph.op(
        "holoscan.visualization.HolovizOp",
        "holoviz",
        layers=layers,
        sink=sink,
        # One frame per received frame.
        trigger={"kind": "layer"},
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
            hl.operators.RoceReceiverOp,
            f"rx_{name}",
            data_channel=channels[i],
            frame_context=int(cu_context),
            device_start=camera.start,
            device_stop=camera.stop,
            frame_size=frame_sizes[i],
            ibv_name=ibv_name,
            ibv_port=ibv_port,
            frame_limit=frame_limit or 0,
            frame_limit_ends_run=True,
        )

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

        # image_processor.metadata is not connected: nothing uses it, and an
        # output that is not connected has no queue that can fill.

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
        # With --frame-limit, the receiver ends the run after frame_limit frames.
        session.wait()

    hololink_dev.stop()
    _cuda_check(cuda_driver.cuDevicePrimaryCtxRelease(cu_device))

    print("Done.")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera-mode",
        type=int,
        default=0,
        help="VB1940 mode (default: 0 = 2560x1984@30fps)",
    )
    parser.add_argument(
        "--trigger",
        action="store_true",
        help="Run in trigger mode for camera synchronization",
    )
    parser.add_argument(
        "--exp",
        type=int,
        default=256,
        help="Set EXPOSURE duration in lines, RANGE(4 to 65535). Default line value is 29.70usec",
    )
    parser.add_argument(
        "--gain",
        type=int,
        default=0,
        help="Set Analog Gain, RANGE(0 to 12). Default is 0. Equation is (16/(16-gain))",
    )
    parser.add_argument(
        "--frequency",
        type=int,
        default=30,
        help="VSYNC frequency in Hz (10, 30, 60, 90, 120). Default is 30Hz",
    )
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Exit after receiving this many frames",
    )
    parser.add_argument(
        "--hololink", default="192.168.0.2", help="IP address of Hololink board"
    )
    parser.add_argument(
        "--log-level", type=int, default=20, help="Python logging level to display"
    )
    parser.add_argument(
        "--ibv-name",
        default="",
        help="IBV device to use (default: the device on the board's network interface)",
    )
    parser.add_argument(
        "--ibv-port",
        type=int,
        default=1,
        help="Port number of IBV device (with --ibv-name)",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=2160 // 4,  # arbitrary default
        help="Set the height of the displayed window",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=3840 // 3,  # arbitrary default
        help="Set the width of the displayed window",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Set the window title",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    return run(
        args.hololink,
        args.camera_mode,
        args.ibv_name,
        args.ibv_port,
        args.trigger,
        args.frequency,
        args.exp,
        args.gain,
        args.window_height,
        args.window_width,
        args.title,
        args.frame_limit,
        args.headless,
    )


if __name__ == "__main__":
    sys.exit(main())
