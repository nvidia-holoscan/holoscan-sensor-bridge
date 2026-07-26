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
import atexit
import ctypes
import logging
import signal
import sys

import hololink_module.operators
import hololink_module.sensors.ar0234.ar0234_mode as _ar0234_mode_module
import hololink_module.sensors.deserializers.max96716a as _max96716a_module
import hololink_module.sensors.hawk as _hawk_module
import hololink_module.taurotech_da326 as _da326_module
import holoscan

import hololink as hololink_legacy
import hololink_module


def _parse_mac(mac_str):
    return [int(x, 16) for x in mac_str.split(":")]


class SkewState:
    """Pairs left/right frames by nearest FPGA-receive timestamp (the receiver
    metadata `timestamp_s/timestamp_ns`)."""

    def __init__(self):
        self._left = []
        self._right = []
        self._deltas_ns = []

    def record(self, label, ts_ns):
        own = self._left if label == "left" else self._right
        other = self._right if label == "left" else self._left
        if other:
            best_idx = 0
            best_diff = abs(other[0] - ts_ns)
            for i in range(1, len(other)):
                d = abs(other[i] - ts_ns)
                if d < best_diff:
                    best_diff = d
                    best_idx = i
            HALF_FRAME_NS = 8_333_333  # half of 1/60s
            if best_diff <= HALF_FRAME_NS:
                other_ts = other.pop(best_idx)
                if label == "right":
                    self._deltas_ns.append(ts_ns - other_ts)
                else:
                    self._deltas_ns.append(other_ts - ts_ns)
                cutoff = min(ts_ns, other_ts)
                self._left[:] = [t for t in self._left if t >= cutoff]
                self._right[:] = [t for t in self._right if t >= cutoff]
                return
        own.append(ts_ns)
        if len(own) > 16:
            del own[0]

    def summary(self):
        n = len(self._deltas_ns)
        if n == 0:
            return "no paired frames"
        avg_us = abs(sum(self._deltas_ns) / n) / 1000.0
        return f"frames={n}, average={avg_us:.2f} us"


class SkewProbe(holoscan.core.Operator):
    def __init__(self, *args, label, state, **kwargs):
        self._label = label
        self._state = state
        super().__init__(*args, **kwargs)

    def setup(self, spec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        msg = op_input.receive("in")
        m = self.metadata
        ts_ns = int(m["timestamp_s"]) * 1_000_000_000 + int(m["timestamp_ns"])
        self._state.record(self._label, ts_ns)
        op_output.emit(msg, "out")


class App(holoscan.core.Application):
    def __init__(
        self,
        sif_metadatas,
        hawk,
        cameras,
        interface,
        camera_mode,
        timeout,
        headless,
        width,
        height,
        frame_limit,
        window_titles,
        skew_state,
    ):
        super().__init__()
        self._sif_metadatas = sif_metadatas
        self._interface = interface or ""
        self._hawk = hawk
        self._cameras = cameras
        self._camera_mode = camera_mode
        self._timeout = timeout
        self._headless = headless
        self._width = width
        self._height = height
        self._frame_limit = frame_limit
        self._window_titles = window_titles
        self._skew_state = skew_state

        self.enable_metadata(True)
        self.metadata_policy = holoscan.core.MetadataPolicy.REJECT

    def compose(self):
        for i in range(len(self._cameras)):
            label = "left" if i == 0 else "right"
            if self._frame_limit:
                condition = holoscan.conditions.CountCondition(
                    self, name=f"count_{i}", count=self._frame_limit
                )
            else:
                condition = holoscan.conditions.BooleanCondition(
                    self, name=f"ok_{i}", enable_tick=True
                )

            self._cameras[i].set_mode(self._camera_mode)
            pixel_format = self._cameras[i].pixel_format()
            bayer_format = self._cameras[i].bayer_format()
            name = f"{i}"

            fusa_coe_capture = hololink_module.operators.FusaCoeCaptureOp(
                self,
                condition,
                name=f"fusa_coe_capture_{name}",
                enumeration_metadata=self._sif_metadatas[i],
                interface=self._interface
                or self._sif_metadatas[i].get("interface", ""),
                mac_addr=_parse_mac(
                    self._sif_metadatas[i].get("mac_id", "00:00:00:00:00:00")
                ),
                timeout=self._timeout,
                device=self._hawk,
                out_tensor_name=name,
            )
            self._cameras[i].configure_converter(fusa_coe_capture)

            skew_probe = SkewProbe(
                self,
                name=f"skew_probe_{label}",
                label=label,
                state=self._skew_state,
            )

            packed_format_converter_pool = holoscan.resources.BlockMemoryPool(
                self,
                name=f"packed_format_converter_pool_{name}",
                storage_type=1,
                block_size=self._cameras[i]._width
                * ctypes.sizeof(ctypes.c_uint16)
                * self._cameras[i]._height,
                num_blocks=4,
            )
            packed_format_converter = hololink_module.operators.PackedFormatConverterOp(
                self,
                name=f"packed_format_converter_{name}",
                allocator=packed_format_converter_pool,
                in_tensor_name=name,
            )
            fusa_coe_capture.configure_converter(packed_format_converter)

            image_processor = hololink_legacy.operators.ImageProcessorOp(
                self,
                name=f"image_processor_{name}",
                optical_black=50,
                bayer_format=bayer_format.value,
                pixel_format=pixel_format.value,
            )

            rgba_components_per_pixel = 4
            bayer_demosaic_pool = holoscan.resources.BlockMemoryPool(
                self,
                name=f"bayer_demosaic_pool_{name}",
                storage_type=1,
                block_size=self._cameras[i]._width
                * rgba_components_per_pixel
                * ctypes.sizeof(ctypes.c_uint16)
                * self._cameras[i]._height,
                num_blocks=4,
            )
            bayer_demosaic = holoscan.operators.BayerDemosaicOp(
                self,
                name=f"bayer_demosaic_{name}",
                pool=bayer_demosaic_pool,
                generate_alpha=True,
                alpha_value=65535,
                bayer_grid_pos=bayer_format.value,
                interpolation_mode=0,
            )

            visualizer = holoscan.operators.HolovizOp(
                self,
                name=f"holoviz_{name}",
                framebuffer_srgb=True,
                headless=self._headless,
                height=self._height,
                width=self._width,
                window_title=self._window_titles[i],
            )

            self.add_flow(fusa_coe_capture, skew_probe, {("output", "in")})
            self.add_flow(skew_probe, packed_format_converter, {("out", "input")})
            self.add_flow(
                packed_format_converter, image_processor, {("output", "input")}
            )
            self.add_flow(image_processor, bayer_demosaic, {("output", "receiver")})
            self.add_flow(bayer_demosaic, visualizer, {("transmitter", "receivers")})


def main():
    parser = argparse.ArgumentParser()
    modes = _ar0234_mode_module.Ar0234_Mode
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
        "--disable-sync",
        action="store_true",
        help="Skip VSYNC/FSYNC setup; sensors free-run.",
    )
    args = parser.parse_args()

    hololink_legacy.logging_level(args.log_level)

    sif_list = (0, 1)
    _adapter = hololink_module.Adapter.get_adapter()
    adapter_metadata = _adapter.wait_for_channel(args.hololink, 30)

    sif_metadatas = []
    for sif in sif_list:
        sif_meta = hololink_module.EnumerationMetadata(adapter_metadata)
        _adapter.use_sensor(sif_meta, sif)
        sif_metadatas.append(sif_meta)

    holo_iface = hololink_module.HololinkInterfaceV1.get_service(adapter_metadata)
    holo_iface.start()
    board = _da326_module.TauroTechDa326InterfaceV1.get_service(adapter_metadata)
    legacy_holo = board.hololink()
    deserializer = _max96716a_module.Max96716a(legacy_holo)
    hawk = _hawk_module.Hawk(legacy_holo)
    camera_mode = _ar0234_mode_module.Ar0234_Mode(args.camera_mode)

    cameras = list(hawk.sensors)
    window_titles = [f"{args.title} - left", f"{args.title} - right"]
    per_w = args.width // 2
    per_h = args.height // 2
    skew_state = SkewState()

    # Always print the skew on exit, regardless of how the process winds down
    # (Ctrl+C, frame-limit, FuSa worker-thread shutdown error). Multiple safety
    # nets because the C++ worker thread can abort() the process before
    # Python's `finally` block runs.
    _printed = [False]

    def _print_skew():
        if _printed[0]:
            return
        _printed[0] = True
        print(f"Stereo skew: {skew_state.summary()}", flush=True)

    atexit.register(_print_skew)

    def _sigint_handler(_signum, _frame):
        _print_skew()
        # Re-raise default so Holoscan's own interrupt handler still runs.
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        sys.exit(130)

    signal.signal(signal.SIGINT, _sigint_handler)

    application = App(
        sif_metadatas,
        hawk,
        cameras,
        args.interface,
        camera_mode,
        args.timeout,
        args.headless,
        per_w,
        per_h,
        args.frame_limit,
        window_titles,
        skew_state,
    )

    try:
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

        link = deserializer.GmslLink[f"LINK_{args.channel}"]
        deserializer.enable_link_exclusive(link)

        dev_id = hawk.serializer.get_register(hawk.serializer.DEV_ID_REG)
        if dev_id != hawk.serializer.DEV_ID:
            raise Exception(
                f"Serializer mismatch: expected {hawk.serializer.DEV_ID}, got {dev_id}"
            )

        for sensor in hawk.sensors:
            dev_id = sensor.get_register(sensor.DEV_ID_REG)
            if dev_id != sensor.DEV_ID:
                raise Exception(
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

        hawk.configure(camera_mode, fsync=not args.disable_sync)
        if args.pattern is not None:
            hawk.test_pattern(args.pattern)
        hawk.set_exposure_reg(args.exposure)

        if not args.disable_sync:
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

        application.run()
    finally:
        holo_iface.stop()
        _print_skew()


if __name__ == "__main__":
    main()
