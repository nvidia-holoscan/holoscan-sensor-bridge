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
        hawk,
        cameras,
        camera_mode,
        timeout,
        headless,
        fullscreen,
        window_width,
        window_height,
        frame_limit,
        window_titles,
    ):
        super().__init__()
        self._hololink_channels = sif_metadatas
        self._hawk = hawk
        self._cameras = cameras
        self._camera_mode = camera_mode
        self._timeout = timeout
        self._headless = headless
        self._fullscreen = fullscreen
        self._window_width = window_width
        self._window_height = window_height
        self._frame_limit = frame_limit
        self._window_titles = window_titles

        self.enable_metadata(True)
        self.metadata_policy = holoscan.core.MetadataPolicy.REJECT

    def compose(self):
        for i, (sif_metadata, camera, title) in enumerate(
            zip(
                self._hololink_channels, self._cameras, self._window_titles, strict=True
            )
        ):
            self._compose_pipeline(i, sif_metadata, camera, title)

    def _compose_pipeline(self, i, sif_metadata, camera, window_title):
        if self._frame_limit:
            condition = holoscan.conditions.CountCondition(
                self, name=f"count_{i}", count=self._frame_limit
            )
        else:
            condition = holoscan.conditions.BooleanCondition(
                self, name=f"ok_{i}", enable_tick=True
            )

        camera.set_mode(self._camera_mode)
        pixel_format = camera.pixel_format()
        bayer_format = camera.bayer_format()
        name = f"{i}"

        fusa_coe_capture = hololink_module.operators.FusaCoeCaptureOp(
            self,
            condition,
            name=f"fusa_coe_capture_{name}",
            enumeration_metadata=sif_metadata,
            interface=sif_metadata.get("interface", ""),
            mac_addr=_parse_mac(sif_metadata.get("mac_id", "00:00:00:00:00:00")),
            timeout=self._timeout,
            device=self._hawk,
            out_tensor_name=name,
        )
        camera.configure_converter(fusa_coe_capture)

        packed_format_converter_pool = holoscan.resources.BlockMemoryPool(
            self,
            name=f"packed_format_converter_pool_{name}",
            storage_type=1,
            block_size=camera._width * ctypes.sizeof(ctypes.c_uint16) * camera._height,
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
            block_size=camera._width
            * rgba_components_per_pixel
            * ctypes.sizeof(ctypes.c_uint16)
            * camera._height,
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
            fullscreen=self._fullscreen,
            headless=self._headless,
            height=self._window_height,
            width=self._window_width,
            window_title=window_title,
        )

        self.add_flow(fusa_coe_capture, packed_format_converter, {("output", "input")})
        self.add_flow(packed_format_converter, image_processor, {("output", "input")})
        self.add_flow(image_processor, bayer_demosaic, {("output", "receiver")})
        self.add_flow(bayer_demosaic, visualizer, {("transmitter", "receivers")})


def main():
    SENSOR_ARG_TO_SIF = {"left": (0,), "right": (1,), "both": (0, 1)}
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
    args = parser.parse_args()

    hololink_legacy.logging_level(args.log_level)

    sif_list = SENSOR_ARG_TO_SIF[args.sensor]
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

    cameras = [hawk.sensors[sif] for sif in sif_list]
    window_titles = [f"{args.title} - {SIF_TO_NAME[sif]}" for sif in sif_list]
    per_window_width = args.window_width
    per_window_height = args.window_height
    if len(sif_list) > 1:
        per_window_width //= 2
        per_window_height //= 2

    application = App(
        sif_metadatas,
        hawk,
        cameras,
        camera_mode,
        args.timeout,
        args.headless,
        args.fullscreen,
        per_window_width,
        per_window_height,
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

        hawk.configure(camera_mode)
        if args.pattern is not None:
            hawk.test_pattern(args.pattern)
        hawk.set_exposure_reg(args.exposure)

        application.run()
    finally:
        holo_iface.stop()


if __name__ == "__main__":
    main()
