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


class App(holoscan.core.Application):
    def __init__(
        self,
        sif_metadatas,
        hawks,
        cameras,
        camera_mode,
        timeout,
        headless,
        width,
        height,
        frame_limit,
        window_titles,
    ):
        super().__init__()
        self._sif_metadatas = sif_metadatas
        self._hawks = hawks
        self._cameras = cameras
        self._camera_mode = camera_mode
        self._timeout = timeout
        self._headless = headless
        self._width = width
        self._height = height
        self._frame_limit = frame_limit
        self._window_titles = window_titles

        self.enable_metadata(True)
        self.metadata_policy = holoscan.core.MetadataPolicy.REJECT

    def compose(self):
        for i in range(len(self._cameras)):
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
            hawk_i = self._hawks[i]

            fusa_coe_capture = hololink_module.operators.FusaCoeCaptureOp(
                self,
                condition,
                name=f"fusa_coe_capture_{name}",
                enumeration_metadata=self._sif_metadatas[i],
                interface=self._sif_metadatas[i].get("interface", ""),
                mac_addr=_parse_mac(
                    self._sif_metadatas[i].get("mac_id", "00:00:00:00:00:00")
                ),
                timeout=self._timeout,
                device=hawk_i,
                out_tensor_name=name,
            )
            self._cameras[i].configure_converter(fusa_coe_capture)

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

            self.add_flow(
                fusa_coe_capture, packed_format_converter, {("output", "input")}
            )
            self.add_flow(
                packed_format_converter, image_processor, {("output", "input")}
            )
            self.add_flow(image_processor, bayer_demosaic, {("output", "receiver")})
            self.add_flow(bayer_demosaic, visualizer, {("transmitter", "receivers")})


def main():
    SENSOR_ARG_TO_SIF = {"left": 0, "right": 1}

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
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--title", default="AR0234 dual-link (FuSa CoE)")
    parser.add_argument(
        "--sensor",
        choices=tuple(SENSOR_ARG_TO_SIF.keys()),
        default="left",
        help="Which AR0234 (left or right) to stream from each Hawk module; one window per Hawk.",
    )
    parser.add_argument("--exposure", type=lambda x: int(x, 0), default=0x02DC)
    parser.add_argument("--pattern", type=lambda x: int(x, 0))
    parser.add_argument("--timeout", type=int, default=1500)
    parser.add_argument("--interface", default=None)
    parser.add_argument("--skip-reset", action="store_true")
    args = parser.parse_args()

    hololink_legacy.logging_level(args.log_level)

    # One SIF per Hawk — same sensor index from each Hawk.
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
    hawk_a = _hawk_module.Hawk(legacy_holo)
    hawk_b = _hawk_module.Hawk(legacy_holo)
    hawk_modules = (hawk_a, hawk_b)
    camera_mode = _ar0234_mode_module.Ar0234_Mode(args.camera_mode)

    sensor_index = SENSOR_ARG_TO_SIF[args.sensor]
    cameras = [hawk.sensors[sensor_index] for hawk in hawk_modules]
    hawk_label = ("A", "B")
    window_titles = [
        f"{args.title} - {args.sensor} {hawk_label[i]}"
        for i in range(len(hawk_modules))
    ]

    application = App(
        sif_metadatas,
        hawk_modules,
        cameras,
        camera_mode,
        args.timeout,
        args.headless,
        args.width,
        args.height,
        args.frame_limit,
        window_titles,
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

        # Both serializers default to 0x40. Remap link B serializer to 0x41.
        deserializer.enable_link_exclusive(deserializer.GmslLink.LINK_B)
        hawk_b.set_serializer_i2c_address(hawk_b.get_serializer_i2c_address() + 1)
        deserializer.enable_both_links()

        for hawk in hawk_modules:
            dev_id = hawk.serializer.get_register(hawk.serializer.DEV_ID_REG)
            if dev_id != hawk.serializer.DEV_ID:
                raise Exception(
                    f"Serializer mismatch: expected {hawk.serializer.DEV_ID}, got {dev_id}"
                )

        # Remap hawk_b sensors from 0x10/0x18 to 0x11/0x1A.
        hawk_b.remap_sensor_addresses(0x11, 0x1A)

        for hawk in hawk_modules:
            for sensor in hawk.sensors:
                dev_id = sensor.get_register(sensor.DEV_ID_REG)
                if dev_id != sensor.DEV_ID:
                    raise Exception(
                        f"Sensor mismatch at i2c=0x{sensor.get_i2c_address():02X}: "
                        f"expected 0x{sensor.DEV_ID:04X}, got 0x{dev_id:04X}"
                    )

        deserializer.configure_video_pipe()
        if args.sensor == "left":
            left_map = deserializer.stream_id_to_pipe_mapping(
                deserializer.GmslLink.LINK_A, 0, deserializer.VideoPipe.PIPE_Y
            )
            right_map = deserializer.stream_id_to_pipe_mapping(
                deserializer.GmslLink.LINK_B, 0, deserializer.VideoPipe.PIPE_Z
            )
        else:
            left_map = deserializer.stream_id_to_pipe_mapping(
                deserializer.GmslLink.LINK_A, 2, deserializer.VideoPipe.PIPE_Y
            )
            right_map = deserializer.stream_id_to_pipe_mapping(
                deserializer.GmslLink.LINK_B, 2, deserializer.VideoPipe.PIPE_Z
            )
        deserializer.set_register(deserializer.VIDEO_PIPE_SEL, left_map | right_map)

        for hawk in hawk_modules:
            hawk.configure(camera_mode)
            if args.pattern is not None:
                hawk.test_pattern(args.pattern)
            hawk.set_exposure_reg(args.exposure)

        application.run()
    finally:
        holo_iface.stop()


if __name__ == "__main__":
    main()
