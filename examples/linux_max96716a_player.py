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

import argparse
import os

import cuda.bindings.driver as cuda
import cupy as cp
import hololink_module.operators
import hololink_module.sensors.deserializers.max96716a as _max96716a_module
import hololink_module.taurotech_da326 as _da326_module
import holoscan

import hololink as hololink_legacy
import hololink_module


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

    _write24(deserializer, 0x0245, vs_high)  # VS_HIGH_2
    _write24(deserializer, 0x0248, vs_low)  # VS_LOW_2
    _write16(deserializer, 0x024E, hs_high)  # HS_HIGH_1
    _write16(deserializer, 0x0250, hs_low)  # HS_LOW_1
    _write16(deserializer, 0x0252, hs_cnt)  # HS_CNT_1
    _write16(deserializer, 0x0257, de_high)  # DE_HIGH_1
    _write16(deserializer, 0x0259, de_low)  # DE_LOW_1
    _write16(deserializer, 0x025B, de_cnt)  # DE_CNT_1

    if pattern == "checkerboard":
        deserializer.set_register(0x025E, 0x00)  # CHKR_COLOR_A_L
        deserializer.set_register(0x025F, 0x00)  # CHKR_COLOR_A_M
        deserializer.set_register(0x0260, 0xFF)  # CHKR_COLOR_A_M
        deserializer.set_register(0x0261, 0xFF)  # CHKR_COLOR_B_L
        deserializer.set_register(0x0262, 0x00)  # CHKR_COLOR_B_M
        deserializer.set_register(0x0263, 0x00)  # CHKR_COLOR_B_H
        deserializer.set_register(0x0264, 16)  # CHKR_RPT_A
        deserializer.set_register(0x0265, 16)  # CHKR_RPT_B
        deserializer.set_register(0x0266, 8)  # CHKR_ALT
        patgen_mode = 0b01
    elif pattern == "gradient":
        deserializer.set_register(0x025D, 0x01)  # GRAD_INCR
        patgen_mode = 0b10
    else:
        raise Exception(f"Unknown VPG pattern {pattern!r}")

    deserializer.set_register(0x0240, 0xE3)  # PATGEN_0
    deserializer.set_register(0x0241, patgen_mode << 4)  # PATGEN_1


class RgbReshapeOp(holoscan.core.Operator):
    def __init__(self, *args, width, height, **kwargs):
        self._width = width
        self._height = height
        self._pixels = width * height * 3
        super().__init__(*args, **kwargs)

    def setup(self, spec):
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("input")
        cp_frame = cp.asarray(in_message.get(""))

        # Reshape flat receiver buffer into an RGB image so Holoviz can detect it.
        # Pad short frames with zeros; truncate over-long frames.
        if cp_frame.size >= self._pixels:
            img = cp_frame[: self._pixels].reshape(self._height, self._width, 3)
        else:
            usable_rows = cp_frame.size // (self._width * 3)
            if usable_rows == 0:
                return
            img = cp.zeros((self._height, self._width, 3), dtype=cp.uint8)
            img[:usable_rows] = cp_frame[: usable_rows * self._width * 3].reshape(
                usable_rows, self._width, 3
            )
        op_output.emit({"": img}, "output")


class HoloscanApplication(holoscan.core.Application):
    def __init__(
        self,
        headless,
        fullscreen,
        cuda_context,
        cuda_device_ordinal,
        sif_metadata,
        frame_size,
        frame_limit,
        window_height,
        window_width,
        window_title,
        vpg_width,
        vpg_height,
    ):
        super().__init__()
        self._headless = headless
        self._fullscreen = fullscreen
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._sif_metadata = sif_metadata
        self._frame_size = frame_size
        self._frame_limit = frame_limit
        self._window_height = window_height
        self._window_width = window_width
        self._window_title = window_title
        self._vpg_width = vpg_width
        self._vpg_height = vpg_height

    def compose(self):
        if self._frame_limit:
            condition = holoscan.conditions.CountCondition(
                self, name="count", count=self._frame_limit
            )
        else:
            condition = holoscan.conditions.BooleanCondition(
                self, name="ok", enable_tick=True
            )

        # VPG has no per-stream device to start/stop. Pass no-op callables.
        receiver_operator = hololink_module.operators.LinuxReceiverOp(
            self,
            condition,
            name="receiver",
            enumeration_metadata=self._sif_metadata,
            frame_context=self._cuda_context,
            frame_size=self._frame_size,
            device_start=lambda: None,
            device_stop=lambda: None,
        )

        profiler = RgbReshapeOp(
            self, name="profiler", width=self._vpg_width, height=self._vpg_height
        )

        visualizer = holoscan.operators.HolovizOp(
            self,
            name="holoviz",
            framebuffer_srgb=True,
            fullscreen=self._fullscreen,
            headless=self._headless,
            height=self._window_height,
            width=self._window_width,
            window_title=self._window_title,
            tensors=[dict(name="", type="color")],
        )

        self.add_flow(receiver_operator, profiler, {("output", "input")})
        self.add_flow(profiler, visualizer, {("output", "receivers")})


def main():
    SENSOR_ARG_TO_SIF = {"left": 0, "right": 1}

    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--frame-limit", type=int, default=None)
    parser.add_argument(
        "--configuration",
        default=os.path.join(os.path.dirname(__file__), "example_configuration.yaml"),
    )
    parser.add_argument("--hololink", default="192.168.0.2")
    parser.add_argument("--log-level", type=int, default=20)
    parser.add_argument("--window-height", type=int, default=1080)
    parser.add_argument("--window-width", type=int, default=1920)
    parser.add_argument("--title", default="MAX96716A VPG")
    parser.add_argument(
        "--width", type=int, default=1920, help="VPG output width in pixels"
    )
    parser.add_argument(
        "--height", type=int, default=1080, help="VPG output height in pixels"
    )
    parser.add_argument(
        "--pattern", choices=("checkerboard", "gradient"), default="checkerboard"
    )
    parser.add_argument(
        "--sensor",
        choices=tuple(SENSOR_ARG_TO_SIF.keys()),
        default="left",
        help="Which SIF to capture from (left=SIF 0/Port A, right=SIF 1/Port B).",
    )
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--skip-reset", action="store_true")
    args = parser.parse_args()
    hololink_legacy.logging_level(args.log_level)

    # CUDA section
    (cu_result,) = cuda.cuInit(0)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_device_ordinal = 0
    cu_result, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_result, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS

    # hololink channel setup — single channel pinned to the requested SIF.
    sif = SENSOR_ARG_TO_SIF[args.sensor]

    adapter = hololink_module.Adapter.get_adapter()
    metadata = adapter.wait_for_channel(args.hololink, 30.0)
    sif_metadata = hololink_module.EnumerationMetadata(metadata)
    adapter.use_sensor(sif_metadata, sif)

    holo_iface = hololink_module.HololinkInterfaceV1.get_service(metadata)
    holo_iface.start()
    board = _da326_module.TauroTechDa326InterfaceV1.get_service(metadata)

    legacy_holo = board.hololink()
    deserializer = _max96716a_module.Max96716a(legacy_holo)

    # RGB888 = 3 bytes/pixel; pad generously for CSI-2 per-line framing overhead.
    frame_size = args.width * 3 * args.height + 16 * args.height + 4096

    window_title = f"{args.title} - {args.sensor}"

    application = HoloscanApplication(
        args.headless,
        args.fullscreen,
        cu_context,
        cu_device_ordinal,
        sif_metadata,
        frame_size,
        args.frame_limit,
        args.window_height,
        args.window_width,
        window_title,
        args.width,
        args.height,
    )
    application.config(args.configuration)

    try:
        if not args.skip_setup:
            if not args.skip_reset:
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
                raise Exception(
                    f"Deserializer mismatch: expected {expected_dev_id}, got {dev_id}"
                )

            # MIPI output configuration — phy_2x4, MIPI_TX1/2 lane counts.
            deserializer.configure_video_pipe()

            # force_csi_out
            deserializer.set_register(0x0330, 0x84)
            configure_vpg(deserializer, args.width, args.height, args.pattern)

        application.run()
    finally:
        holo_iface.stop()

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS


if __name__ == "__main__":
    main()
