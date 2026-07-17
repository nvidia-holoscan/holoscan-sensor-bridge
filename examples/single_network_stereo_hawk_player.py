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
import ctypes
import logging
import os

import cuda.bindings.driver as cuda
import hololink_module.operators
import hololink_module.sensors.ar0234.ar0234_mode as _ar0234_mode_module
import hololink_module.sensors.deserializers.max96716a as _max96716a_module
import hololink_module.sensors.hawk as _hawk_module
import hololink_module.taurotech_da326 as _da326_module
import holoscan

import hololink as hololink_legacy
import hololink_module


class SkewState:
    """Pairs left/right frames by nearest FPGA-receive timestamp (the receiver
    metadata `timestamp_s/timestamp_ns`"""

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

    @property
    def paired_count(self):
        return len(self._deltas_ns)

    def summary(self):
        n = self.paired_count
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


class HoloscanApplication(holoscan.core.Application):
    def __init__(
        self,
        headless,
        fullscreen,
        cuda_context,
        cuda_device_ordinal,
        per_channel_metadatas,
        camera,
        camera_mode,
        frame_limit,
        window_height,
        window_width,
        window_titles,
        skew_state,
    ):
        super().__init__()
        self._headless = headless
        self._fullscreen = fullscreen
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._per_channel_metadatas = per_channel_metadatas
        self._camera = camera
        self._camera_mode = camera_mode
        self._frame_limit = frame_limit
        self._skew_state = skew_state
        self._window_height = window_height
        self._window_width = window_width
        self._window_titles = window_titles
        self.is_metadata_enabled = True
        self.metadata_policy = holoscan.core.MetadataPolicy.REJECT

    def compose(self):
        self._camera.set_mode(self._camera_mode)
        for idx, (sif_metadata, title) in enumerate(
            zip(self._per_channel_metadatas, self._window_titles, strict=True)
        ):
            self._compose_pipeline(idx, sif_metadata, title)

    def _compose_pipeline(self, idx, sif_metadata, window_title):
        label = "left" if idx == 0 else "right"
        if self._frame_limit is not None:
            condition = holoscan.conditions.CountCondition(
                self, name=f"count_{idx}", count=self._frame_limit
            )
        else:
            condition = holoscan.conditions.BooleanCondition(
                self, name=f"ok_{idx}", enable_tick=True
            )

        csi_to_bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name=f"csi_pool_{idx}",
            storage_type=1,
            block_size=self._camera._width
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera._height,
            num_blocks=2,
        )
        csi_to_bayer_operator = hololink_legacy.operators.CsiToBayerOp(
            self,
            name=f"csi_to_bayer_{idx}",
            allocator=csi_to_bayer_pool,
            cuda_device_ordinal=self._cuda_device_ordinal,
        )
        self._camera.configure_converter(csi_to_bayer_operator)

        receiver_operator = hololink_module.operators.RoceReceiverOp(
            self,
            condition,
            name=f"receiver_{idx}",
            enumeration_metadata=sif_metadata,
            frame_context=self._cuda_context,
            frame_size=csi_to_bayer_operator.get_csi_length(),
            device_start=lambda: self._camera.start(),
            device_stop=lambda: self._camera.stop(),
        )

        pixel_format = self._camera.pixel_format()
        bayer_format = self._camera.bayer_format()
        image_processor_operator = hololink_legacy.operators.ImageProcessorOp(
            self,
            name=f"image_processor_{idx}",
            optical_black=50,
            bayer_format=bayer_format.value,
            pixel_format=pixel_format.value,
        )

        bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name=f"bayer_pool_{idx}",
            storage_type=1,
            block_size=self._camera._width
            * 4
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera._height,
            num_blocks=2,
        )
        demosaic = holoscan.operators.BayerDemosaicOp(
            self,
            name=f"demosaic_{idx}",
            pool=bayer_pool,
            generate_alpha=True,
            alpha_value=65535,
            bayer_grid_pos=bayer_format.value,
            interpolation_mode=0,
        )
        visualizer = holoscan.operators.HolovizOp(
            self,
            name=f"holoviz_{idx}",
            framebuffer_srgb=True,
            fullscreen=self._fullscreen,
            headless=self._headless,
            height=self._window_height,
            width=self._window_width,
            window_title=window_title,
        )

        skew_probe = SkewProbe(
            self,
            name=f"skew_probe_{label}",
            label=label,
            state=self._skew_state,
        )

        self.add_flow(receiver_operator, skew_probe, {("output", "in")})
        self.add_flow(skew_probe, csi_to_bayer_operator, {("out", "input")})
        self.add_flow(
            csi_to_bayer_operator, image_processor_operator, {("output", "input")}
        )
        self.add_flow(image_processor_operator, demosaic, {("output", "receiver")})
        self.add_flow(demosaic, visualizer, {("transmitter", "receivers")})


def main():
    SIF_LIST = (0, 1)
    SIF_TO_NAME = {0: "left", 1: "right"}

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
    hololink_legacy.logging_level(args.log_level)

    (cu_result,) = cuda.cuInit(0)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_device_ordinal = 0
    cu_result, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_result, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS

    _adapter = hololink_module.Adapter.get_adapter()
    adapter_metadata = _adapter.wait_for_channel(args.hololink, 30)

    holo_iface = hololink_module.HololinkInterfaceV1.get_service(adapter_metadata)
    holo_iface.start()
    board = _da326_module.TauroTechDa326InterfaceV1.get_service(adapter_metadata)
    # sensor_holo provides I2C access for sensor/deserializer setup;
    # sensors will be migrated to HololinkInterfaceV1.get_i2c() separately.
    sensor_holo = board.hololink()

    per_channel_metadatas = []
    for sif in SIF_LIST:
        sif_meta = hololink_module.EnumerationMetadata(adapter_metadata)
        _adapter.use_sensor(sif_meta, sif)
        per_channel_metadatas.append(sif_meta)

    deserializer = _max96716a_module.Max96716a(sensor_holo)
    hawk = _hawk_module.Hawk(
        sensor_holo,
        skip_i2c=args.skip_setup,
    )
    camera_mode = _ar0234_mode_module.Ar0234_Mode(args.camera_mode)

    window_titles = [f"{args.title} - {SIF_TO_NAME[idx]}" for idx in SIF_LIST]
    per_window_width = args.window_width // 2
    per_window_height = args.window_height // 2

    skew_state = SkewState()

    application = HoloscanApplication(
        args.headless,
        args.fullscreen,
        cu_context,
        cu_device_ordinal,
        per_channel_metadatas,
        hawk,
        camera_mode,
        args.frame_limit,
        per_window_height,
        per_window_width,
        window_titles,
        skew_state,
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

            # Deserializer sanity check
            dev_id = deserializer.get_register(deserializer.DEV_ID_REG)
            if dev_id != deserializer.DEV_ID:
                raise Exception(
                    f"Deserializer mismatch: expected {deserializer.DEV_ID}, "
                    f"got {dev_id}"
                )

            # Only enable the link that we care about
            link = deserializer.GmslLink[f"LINK_{args.channel}"]
            deserializer.enable_link_exclusive(link)

            # Serializer sanity check
            dev_id = hawk.serializer.get_register(hawk.serializer.DEV_ID_REG)
            if dev_id != hawk.serializer.DEV_ID:
                raise Exception(
                    f"Serializer mismatch: expected {hawk.serializer.DEV_ID}, "
                    f"got {dev_id}"
                )

            # Sensor sanity check
            for sensor in hawk.sensors:
                dev_id = sensor.get_register(sensor.DEV_ID_REG)
                if dev_id != sensor.DEV_ID:
                    raise Exception(
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
            hawk.configure(camera_mode, fsync=not args.disable_sync)

            if args.pattern is not None:
                hawk.test_pattern(args.pattern)

            hawk.set_exposure_reg(args.exposure)

            if not args.disable_sync:
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

        application.run()
    finally:
        holo_iface.stop()
        logging.info("Stereo skew: %s", skew_state.summary())

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS


if __name__ == "__main__":
    main()
