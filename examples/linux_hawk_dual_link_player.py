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
import os

import cuda.bindings.driver as cuda
import cupy as cp
import hololink_module.operators
import hololink_module.sensors.ar0234.ar0234_mode as _ar0234_mode_module
import hololink_module.sensors.deserializers.max96716a as _max96716a_module
import hololink_module.sensors.hawk as _hawk_module
import hololink_module.taurotech_da326 as _da326_module
import holoscan

import hololink as hololink_legacy
import hololink_module


class InstrumentedTimeProfiler(holoscan.core.Operator):
    def setup(self, spec):
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("input")
        cp_frame = cp.asarray(in_message.get(""))
        op_output.emit({"": cp_frame}, "output")


class HoloscanApplication(holoscan.core.Application):
    def __init__(
        self,
        headless,
        fullscreen,
        cuda_context,
        cuda_device_ordinal,
        sif_metadatas,
        cameras,
        camera_mode,
        frame_limit,
        window_height,
        window_width,
        window_titles,
    ):
        super().__init__()
        self._headless = headless
        self._fullscreen = fullscreen
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._sif_metadatas = sif_metadatas
        self._cameras = cameras
        self._camera_mode = camera_mode
        self._frame_limit = frame_limit
        self._window_height = window_height
        self._window_width = window_width
        self._window_titles = window_titles

    def compose(self):
        for camera in self._cameras:
            camera.set_mode(self._camera_mode)
        for idx, (sif_metadata, camera, title) in enumerate(
            zip(self._sif_metadatas, self._cameras, self._window_titles)
        ):
            self._compose_pipeline(idx, sif_metadata, camera, title)

    def _compose_pipeline(self, idx, sif_metadata, camera, window_title):
        if self._frame_limit:
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
            block_size=camera._width * ctypes.sizeof(ctypes.c_uint16) * camera._height,
            num_blocks=2,
        )
        csi_to_bayer_operator = hololink_legacy.operators.CsiToBayerOp(
            self,
            name=f"csi_to_bayer_{idx}",
            allocator=csi_to_bayer_pool,
            cuda_device_ordinal=self._cuda_device_ordinal,
        )
        camera.configure_converter(csi_to_bayer_operator)

        receiver_operator = hololink_module.operators.LinuxReceiverOp(
            self,
            condition,
            name=f"receiver_{idx}",
            enumeration_metadata=sif_metadata,
            frame_context=self._cuda_context,
            frame_size=csi_to_bayer_operator.get_csi_length(),
            device_start=camera.start,
            device_stop=camera.stop,
        )

        profiler = InstrumentedTimeProfiler(self, name=f"profiler_{idx}")
        pixel_format = camera.pixel_format()
        bayer_format = camera.bayer_format()
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
            block_size=camera._width
            * 4
            * ctypes.sizeof(ctypes.c_uint16)
            * camera._height,
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

        self.add_flow(receiver_operator, profiler, {("output", "input")})
        self.add_flow(profiler, csi_to_bayer_operator, {("output", "input")})
        self.add_flow(
            csi_to_bayer_operator, image_processor_operator, {("output", "input")}
        )
        self.add_flow(image_processor_operator, demosaic, {("output", "receiver")})
        self.add_flow(demosaic, visualizer, {("transmitter", "receivers")})


def main():
    sensor_options = {"left", "right"}
    # CLI arguments section
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
    parser.add_argument("--title", default="AR0234")
    parser.add_argument(
        "--sensor",
        choices=tuple(sensor_options),
        default="left",
        help="Which AR0234 (left or right) to stream from each Hawk module; one window per Hawk.",
    )
    parser.add_argument("--exposure", type=lambda x: int(x, 0), default=0x02DC)
    parser.add_argument("--pattern", type=lambda x: int(x, 0))
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

    # hololink channel(s) setup — one SIF metadata per link (SIF 0 and SIF 1).
    sif_list = [0, 1]

    adapter = hololink_module.Adapter.get_adapter()
    metadata = adapter.wait_for_channel(args.hololink, 30.0)

    sif_metadatas = []
    for sif in sif_list:
        sif_meta = hololink_module.EnumerationMetadata(metadata)
        adapter.use_sensor(sif_meta, sif)
        sif_metadatas.append(sif_meta)

    holo_iface = hololink_module.HololinkInterfaceV1.get_service(metadata)
    holo_iface.start()
    board = _da326_module.TauroTechDa326InterfaceV1.get_service(metadata)

    legacy_holo = board.hololink()
    deserializer = _max96716a_module.Max96716a(legacy_holo)
    hawk_a = _hawk_module.Hawk(
        legacy_holo,
        skip_i2c=args.skip_setup,
    )
    hawk_b = _hawk_module.Hawk(
        legacy_holo,
        skip_i2c=args.skip_setup,
    )
    hawk_modules = [hawk_a, hawk_b]
    camera_mode = _ar0234_mode_module.Ar0234_Mode(args.camera_mode)

    # A first then B
    sensor_label = args.sensor.capitalize()
    window_titles = [f"{sensor_label} A", f"{sensor_label} B"]

    # application section
    per_window_height = args.window_height // 2
    per_window_width = args.window_width // 2

    application = HoloscanApplication(
        args.headless,
        args.fullscreen,
        cu_context,
        cu_device_ordinal,
        sif_metadatas,
        hawk_modules,
        camera_mode,
        args.frame_limit,
        per_window_height,
        per_window_width,
        window_titles,
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

            # Both of the serializers are at 0x40. Remap link B serializer to 0x41.
            deserializer.enable_link_exclusive(deserializer.GmslLink.LINK_B)
            hawk_b.set_serializer_i2c_address(hawk_b.get_serializer_i2c_address() + 1)
            deserializer.enable_both_links()

            # Serializer sanity check
            for hawk in hawk_modules:
                dev_id = hawk.serializer.get_register(hawk.serializer.DEV_ID_REG)
                expected_dev_id = hawk.serializer.DEV_ID
                if dev_id != expected_dev_id:
                    raise Exception(
                        f"Serializer mismatch: expected {expected_dev_id}, got {dev_id}"
                    )

            # Both of the sensors are at 0x10 and 0x18. Remap sensor hawk_b sensors to 0x11 and 0x1A
            hawk_b.remap_sensor_addresses(0x11, 0x1A)

            # Sensor sanity check
            for hawk in hawk_modules:
                for sensor in hawk.sensors:
                    dev_id = sensor.get_register(sensor.DEV_ID_REG)
                    if dev_id != sensor.DEV_ID:
                        raise Exception(
                            f"Sensor mismatch at i2c=0x{sensor.get_i2c_address():02X}: "
                            f"expected 0x{sensor.DEV_ID:04X}, got 0x{dev_id:04X}"
                        )

            # Deserializer settings
            deserializer.configure_video_pipe()

            if args.sensor == "left":
                left = deserializer.stream_id_to_pipe_mapping(
                    deserializer.GmslLink.LINK_A, 0, deserializer.VideoPipe.PIPE_Y
                )
                right = deserializer.stream_id_to_pipe_mapping(
                    deserializer.GmslLink.LINK_B, 0, deserializer.VideoPipe.PIPE_Z
                )
            elif args.sensor == "right":
                left = deserializer.stream_id_to_pipe_mapping(
                    deserializer.GmslLink.LINK_A, 2, deserializer.VideoPipe.PIPE_Y
                )
                right = deserializer.stream_id_to_pipe_mapping(
                    deserializer.GmslLink.LINK_B, 2, deserializer.VideoPipe.PIPE_Z
                )
            else:
                raise Exception(f"Non-existent config {args.sensor}")
            deserializer.set_register(deserializer.VIDEO_PIPE_SEL, left | right)

            # Sensor programming
            for hawk in hawk_modules:
                hawk.configure(camera_mode)
                if args.pattern is not None:
                    hawk.test_pattern(args.pattern)
                hawk.set_exposure_reg(args.exposure)

        application.run()
    finally:
        holo_iface.stop()

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS


if __name__ == "__main__":
    main()
