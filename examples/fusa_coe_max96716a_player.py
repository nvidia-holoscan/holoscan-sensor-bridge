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

import cupy as cp
import hololink_module.operators
import hololink_module.sensors.deserializers.max96716a as _max96716a_module
import hololink_module.taurotech_da326 as _da326_module
import holoscan

import hololink as hololink_legacy
import hololink_module


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
        raise Exception(f"Unknown VPG pattern {pattern!r}")

    deserializer.set_register(0x0240, 0xE3)
    deserializer.set_register(0x0241, patgen_mode << 4)


class _VpgDevice:
    """FuSa CoE capture requires a device with start()/stop(); the VPG has none."""

    def start(self):
        pass

    def stop(self):
        pass


class _RgbReshape(holoscan.core.Operator):
    """Reshapes the flat uint8 receiver buffer into (H, W, 3) so Holoviz recognises it."""

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
        if cp_frame.size < self._pixels:
            return
        img = cp_frame[: self._pixels].reshape(self._height, self._width, 3)
        op_output.emit({"": img}, "output")


class App(holoscan.core.Application):
    def __init__(
        self,
        sif_metadata,
        timeout,
        headless,
        window_width,
        window_height,
        frame_size,
        frame_limit,
        vpg_width,
        vpg_height,
        window_title,
    ):
        super().__init__()
        self._sif_metadata = sif_metadata
        self._timeout = timeout
        self._headless = headless
        self._window_width = window_width
        self._window_height = window_height
        self._frame_size = frame_size
        self._frame_limit = frame_limit
        self._vpg_width = vpg_width
        self._vpg_height = vpg_height
        self._window_title = window_title
        self._device = _VpgDevice()

    def compose(self):
        if self._frame_limit:
            condition = holoscan.conditions.CountCondition(
                self, name="count", count=self._frame_limit
            )
        else:
            condition = holoscan.conditions.BooleanCondition(
                self, name="ok", enable_tick=True
            )

        # Capture raw bytes from the VPG. Bypass the CSI/Bayer plumbing — the VPG
        # emits RGB888 which we treat as an opaque byte stream (configure_frame_size).
        fusa_coe_capture = hololink_module.operators.FusaCoeCaptureOp(
            self,
            condition,
            name="fusa_coe_capture",
            enumeration_metadata=self._sif_metadata,
            interface=self._sif_metadata.get("interface", ""),
            mac_addr=_parse_mac(self._sif_metadata.get("mac_id", "00:00:00:00:00:00")),
            timeout=self._timeout,
            device=self._device,
        )
        fusa_coe_capture.configure_frame_size(self._frame_size)

        reshape = _RgbReshape(
            self, name="reshape", width=self._vpg_width, height=self._vpg_height
        )

        visualizer = holoscan.operators.HolovizOp(
            self,
            name="holoviz",
            framebuffer_srgb=True,
            headless=self._headless,
            height=self._window_height,
            width=self._window_width,
            window_title=self._window_title,
            tensors=[dict(name="", type="color")],
        )

        self.add_flow(fusa_coe_capture, reshape, {("output", "input")})
        self.add_flow(reshape, visualizer, {("output", "receivers")})


def main():
    SENSOR_ARG_TO_SIF = {"left": 0, "right": 1}

    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frame-limit", type=int, default=None)
    parser.add_argument(
        "--configuration",
        default=os.path.join(os.path.dirname(__file__), "example_configuration.yaml"),
    )
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
    hololink_legacy.logging_level(args.log_level)

    sif = SENSOR_ARG_TO_SIF[args.sensor]
    _adapter = hololink_module.Adapter.get_adapter()
    adapter_metadata = _adapter.wait_for_channel(args.hololink, 30)

    sif_metadata = hololink_module.EnumerationMetadata(adapter_metadata)
    _adapter.use_sensor(sif_metadata, sif)

    holo_iface = hololink_module.HololinkInterfaceV1.get_service(adapter_metadata)
    holo_iface.start()
    board = _da326_module.TauroTechDa326InterfaceV1.get_service(adapter_metadata)
    legacy_holo = board.hololink()
    deserializer = _max96716a_module.Max96716a(legacy_holo)

    # RGB888 = 3 bytes/pixel; pad generously for CSI-2 per-line framing overhead.
    frame_size = args.width * 3 * args.height + 16 * args.height + 4096
    window_title = f"{args.title} - {args.sensor}"

    application = App(
        sif_metadata,
        args.timeout,
        args.headless,
        args.window_width,
        args.window_height,
        frame_size,
        args.frame_limit,
        args.width,
        args.height,
        window_title,
    )
    application.config(args.configuration)

    try:
        if not args.skip_setup:
            if not args.skip_reset:
                holo_iface.reset()

            board.release_reset()
            board.power_cycle()
            board.check_power()
            board.setup_clock()

            dev_id = deserializer.get_register(deserializer.DEV_ID_REG)
            if dev_id != deserializer.DEV_ID:
                raise Exception(
                    f"Deserializer mismatch: expected {deserializer.DEV_ID}, got {dev_id}"
                )

            deserializer.configure_video_pipe()
            deserializer.set_register(0x0330, 0x84)  # force_csi_out
            configure_vpg(deserializer, args.width, args.height, args.pattern)

        application.run()
    finally:
        holo_iface.stop()


if __name__ == "__main__":
    main()
