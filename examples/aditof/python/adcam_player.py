#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved.
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

# More details about the chip and eval kit can be found on https://www.analog.com/en/products/adtf3175.html

import argparse
import ctypes
import hashlib
import logging
import os
import pydoc
import sys

import adcam
import cuda.bindings.driver as cuda
import cupy as cp
import holoscan
import requests
import yaml

import hololink as hololink_module

# import time


class ADTFUnpackOp(holoscan.core.Operator):

    JET_LUT_U8 = bytes(
        [
            0x00,
            0x00,
            0x7F,
            0x00,
            0x00,
            0x84,
            0x00,
            0x00,
            0x88,
            0x00,
            0x00,
            0x8D,
            0x00,
            0x00,
            0x91,
            0x00,
            0x00,
            0x96,
            0x00,
            0x00,
            0x9A,
            0x00,
            0x00,
            0x9F,
            0x00,
            0x00,
            0xA3,
            0x00,
            0x00,
            0xA8,
            0x00,
            0x00,
            0xAC,
            0x00,
            0x00,
            0xB1,
            0x00,
            0x00,
            0xB6,
            0x00,
            0x00,
            0xBA,
            0x00,
            0x00,
            0xBF,
            0x00,
            0x00,
            0xC3,
            0x00,
            0x00,
            0xC8,
            0x00,
            0x00,
            0xCC,
            0x00,
            0x00,
            0xD1,
            0x00,
            0x00,
            0xD5,
            0x00,
            0x00,
            0xDA,
            0x00,
            0x00,
            0xDE,
            0x00,
            0x00,
            0xE3,
            0x00,
            0x00,
            0xE8,
            0x00,
            0x00,
            0xEC,
            0x00,
            0x00,
            0xF1,
            0x00,
            0x00,
            0xF5,
            0x00,
            0x00,
            0xFA,
            0x00,
            0x00,
            0xFE,
            0x00,
            0x00,
            0xFF,
            0x00,
            0x00,
            0xFF,
            0x00,
            0x00,
            0xFF,
            0x00,
            0x00,
            0xFF,
            0x00,
            0x04,
            0xFF,
            0x00,
            0x08,
            0xFF,
            0x00,
            0x0C,
            0xFF,
            0x00,
            0x10,
            0xFF,
            0x00,
            0x14,
            0xFF,
            0x00,
            0x18,
            0xFF,
            0x00,
            0x1C,
            0xFF,
            0x00,
            0x20,
            0xFF,
            0x00,
            0x24,
            0xFF,
            0x00,
            0x28,
            0xFF,
            0x00,
            0x2C,
            0xFF,
            0x00,
            0x30,
            0xFF,
            0x00,
            0x34,
            0xFF,
            0x00,
            0x38,
            0xFF,
            0x00,
            0x3C,
            0xFF,
            0x00,
            0x40,
            0xFF,
            0x00,
            0x44,
            0xFF,
            0x00,
            0x48,
            0xFF,
            0x00,
            0x4C,
            0xFF,
            0x00,
            0x50,
            0xFF,
            0x00,
            0x54,
            0xFF,
            0x00,
            0x58,
            0xFF,
            0x00,
            0x5C,
            0xFF,
            0x00,
            0x60,
            0xFF,
            0x00,
            0x64,
            0xFF,
            0x00,
            0x68,
            0xFF,
            0x00,
            0x6C,
            0xFF,
            0x00,
            0x70,
            0xFF,
            0x00,
            0x74,
            0xFF,
            0x00,
            0x78,
            0xFF,
            0x00,
            0x7C,
            0xFF,
            0x00,
            0x80,
            0xFF,
            0x00,
            0x84,
            0xFF,
            0x00,
            0x88,
            0xFF,
            0x00,
            0x8C,
            0xFF,
            0x00,
            0x90,
            0xFF,
            0x00,
            0x94,
            0xFF,
            0x00,
            0x98,
            0xFF,
            0x00,
            0x9C,
            0xFF,
            0x00,
            0xA0,
            0xFF,
            0x00,
            0xA4,
            0xFF,
            0x00,
            0xA8,
            0xFF,
            0x00,
            0xAC,
            0xFF,
            0x00,
            0xB0,
            0xFF,
            0x00,
            0xB4,
            0xFF,
            0x00,
            0xB8,
            0xFF,
            0x00,
            0xBC,
            0xFF,
            0x00,
            0xC0,
            0xFF,
            0x00,
            0xC4,
            0xFF,
            0x00,
            0xC8,
            0xFF,
            0x00,
            0xCC,
            0xFF,
            0x00,
            0xD0,
            0xFF,
            0x00,
            0xD4,
            0xFF,
            0x00,
            0xD8,
            0xFF,
            0x00,
            0xDC,
            0xFE,
            0x00,
            0xE0,
            0xFA,
            0x00,
            0xE4,
            0xF7,
            0x02,
            0xE8,
            0xF4,
            0x05,
            0xEC,
            0xF1,
            0x08,
            0xF0,
            0xED,
            0x0C,
            0xF4,
            0xEA,
            0x0F,
            0xF8,
            0xE7,
            0x12,
            0xFC,
            0xE4,
            0x15,
            0xFF,
            0xE1,
            0x18,
            0xFF,
            0xDD,
            0x1C,
            0xFF,
            0xDA,
            0x1F,
            0xFF,
            0xD7,
            0x22,
            0xFF,
            0xD4,
            0x25,
            0xFF,
            0xD0,
            0x29,
            0xFF,
            0xCD,
            0x2C,
            0xFF,
            0xCA,
            0x2F,
            0xFF,
            0xC7,
            0x32,
            0xFF,
            0xC3,
            0x36,
            0xFF,
            0xC0,
            0x39,
            0xFF,
            0xBD,
            0x3C,
            0xFF,
            0xBA,
            0x3F,
            0xFF,
            0xB7,
            0x42,
            0xFF,
            0xB3,
            0x46,
            0xFF,
            0xB0,
            0x49,
            0xFF,
            0xAD,
            0x4C,
            0xFF,
            0xAA,
            0x4F,
            0xFF,
            0xA6,
            0x53,
            0xFF,
            0xA3,
            0x56,
            0xFF,
            0xA0,
            0x59,
            0xFF,
            0x9D,
            0x5C,
            0xFF,
            0x9A,
            0x5F,
            0xFF,
            0x96,
            0x63,
            0xFF,
            0x93,
            0x66,
            0xFF,
            0x90,
            0x69,
            0xFF,
            0x8D,
            0x6C,
            0xFF,
            0x89,
            0x70,
            0xFF,
            0x86,
            0x73,
            0xFF,
            0x83,
            0x76,
            0xFF,
            0x80,
            0x79,
            0xFF,
            0x7D,
            0x7C,
            0xFF,
            0x79,
            0x80,
            0xFF,
            0x76,
            0x83,
            0xFF,
            0x73,
            0x86,
            0xFF,
            0x70,
            0x89,
            0xFF,
            0x6C,
            0x8D,
            0xFF,
            0x69,
            0x90,
            0xFF,
            0x66,
            0x93,
            0xFF,
            0x63,
            0x96,
            0xFF,
            0x5F,
            0x9A,
            0xFF,
            0x5C,
            0x9D,
            0xFF,
            0x59,
            0xA0,
            0xFF,
            0x56,
            0xA3,
            0xFF,
            0x53,
            0xA6,
            0xFF,
            0x4F,
            0xAA,
            0xFF,
            0x4C,
            0xAD,
            0xFF,
            0x49,
            0xB0,
            0xFF,
            0x46,
            0xB3,
            0xFF,
            0x42,
            0xB7,
            0xFF,
            0x3F,
            0xBA,
            0xFF,
            0x3C,
            0xBD,
            0xFF,
            0x39,
            0xC0,
            0xFF,
            0x36,
            0xC3,
            0xFF,
            0x32,
            0xC7,
            0xFF,
            0x2F,
            0xCA,
            0xFF,
            0x2C,
            0xCD,
            0xFF,
            0x29,
            0xD0,
            0xFF,
            0x25,
            0xD4,
            0xFF,
            0x22,
            0xD7,
            0xFF,
            0x1F,
            0xDA,
            0xFF,
            0x1C,
            0xDD,
            0xFF,
            0x18,
            0xE0,
            0xFF,
            0x15,
            0xE4,
            0xFF,
            0x12,
            0xE7,
            0xFF,
            0x0F,
            0xEA,
            0xFF,
            0x0C,
            0xED,
            0xFF,
            0x08,
            0xF1,
            0xFC,
            0x05,
            0xF4,
            0xF8,
            0x02,
            0xF7,
            0xF4,
            0x00,
            0xFA,
            0xF0,
            0x00,
            0xFE,
            0xED,
            0x00,
            0xFF,
            0xE9,
            0x00,
            0xFF,
            0xE5,
            0x00,
            0xFF,
            0xE2,
            0x00,
            0xFF,
            0xDE,
            0x00,
            0xFF,
            0xDA,
            0x00,
            0xFF,
            0xD7,
            0x00,
            0xFF,
            0xD3,
            0x00,
            0xFF,
            0xCF,
            0x00,
            0xFF,
            0xCB,
            0x00,
            0xFF,
            0xC8,
            0x00,
            0xFF,
            0xC4,
            0x00,
            0xFF,
            0xC0,
            0x00,
            0xFF,
            0xBD,
            0x00,
            0xFF,
            0xB9,
            0x00,
            0xFF,
            0xB5,
            0x00,
            0xFF,
            0xB1,
            0x00,
            0xFF,
            0xAE,
            0x00,
            0xFF,
            0xAA,
            0x00,
            0xFF,
            0xA6,
            0x00,
            0xFF,
            0xA3,
            0x00,
            0xFF,
            0x9F,
            0x00,
            0xFF,
            0x9B,
            0x00,
            0xFF,
            0x98,
            0x00,
            0xFF,
            0x94,
            0x00,
            0xFF,
            0x90,
            0x00,
            0xFF,
            0x8C,
            0x00,
            0xFF,
            0x89,
            0x00,
            0xFF,
            0x85,
            0x00,
            0xFF,
            0x81,
            0x00,
            0xFF,
            0x7E,
            0x00,
            0xFF,
            0x7A,
            0x00,
            0xFF,
            0x76,
            0x00,
            0xFF,
            0x73,
            0x00,
            0xFF,
            0x6F,
            0x00,
            0xFF,
            0x6B,
            0x00,
            0xFF,
            0x67,
            0x00,
            0xFF,
            0x64,
            0x00,
            0xFF,
            0x60,
            0x00,
            0xFF,
            0x5C,
            0x00,
            0xFF,
            0x59,
            0x00,
            0xFF,
            0x55,
            0x00,
            0xFF,
            0x51,
            0x00,
            0xFF,
            0x4D,
            0x00,
            0xFF,
            0x4A,
            0x00,
            0xFF,
            0x46,
            0x00,
            0xFF,
            0x42,
            0x00,
            0xFF,
            0x3F,
            0x00,
            0xFF,
            0x3B,
            0x00,
            0xFF,
            0x37,
            0x00,
            0xFF,
            0x34,
            0x00,
            0xFF,
            0x30,
            0x00,
            0xFF,
            0x2C,
            0x00,
            0xFF,
            0x28,
            0x00,
            0xFF,
            0x25,
            0x00,
            0xFF,
            0x21,
            0x00,
            0xFF,
            0x1D,
            0x00,
            0xFF,
            0x1A,
            0x00,
            0xFF,
            0x16,
            0x00,
            0xFE,
            0x12,
            0x00,
            0xFA,
            0x0F,
            0x00,
            0xF5,
            0x0B,
            0x00,
            0xF1,
            0x07,
            0x00,
            0xEC,
            0x03,
            0x00,
            0xE8,
            0x00,
            0x00,
            0xE3,
            0x00,
            0x00,
            0xDE,
            0x00,
            0x00,
            0xDA,
            0x00,
            0x00,
            0xD5,
            0x00,
            0x00,
            0xD1,
            0x00,
            0x00,
            0xCC,
            0x00,
            0x00,
            0xC8,
            0x00,
            0x00,
            0xC3,
            0x00,
            0x00,
            0xBF,
            0x00,
            0x00,
            0xBA,
            0x00,
            0x00,
            0xB6,
            0x00,
            0x00,
            0xB1,
            0x00,
            0x00,
            0xAC,
            0x00,
            0x00,
            0xA8,
            0x00,
            0x00,
            0xA3,
            0x00,
            0x00,
            0x9F,
            0x00,
            0x00,
            0x9A,
            0x00,
            0x00,
            0x96,
            0x00,
            0x00,
            0x91,
            0x00,
            0x00,
            0x8D,
            0x00,
            0x00,
            0x88,
            0x00,
            0x00,
            0x84,
            0x00,
            0x00,
            0x7F,
            0x00,
            0x00,
        ]
    )

    def __init__(self, *args, no_of_planes, width, height, size, **kwargs):
        super().__init__(*args, **kwargs)
        self._no_of_planes = no_of_planes
        lut_np = cp.frombuffer(self.JET_LUT_U8, dtype=cp.uint8).reshape(256, 3)
        self._jet_lut = cp.asarray(lut_np)  # GPU resident

        self._width = width
        self._height = height
        self._size = size
        logging.info(
            f"ADTFUnpackOp init width X height X size = {self._width} X {self._height} X {self._size}"
        )
        self._save = 1

    def setup(self, spec):
        logging.info("ADTFUnpackOp setup")
        spec.input("input")
        spec.output("output")

    def start(self):
        pass

    def stop(self):
        pass

    def converttojetimage(self, depth_u16):
        # depth_u16: (H, W), uint16

        depth_norm = cp.clip(
            (depth_u16.astype(cp.float32) / 4000.0) * 255, 0, 255
        ).astype(cp.uint8)

        rgb = self._jet_lut[depth_norm]
        # shape: (H, W, 3), dtype=uint8
        return rgb

    def convert_to_grayscale(self, image, max_val=4096.0):
        # Normalize to 0-255 using the given full-scale value
        image_normalized = cp.clip(
            image.astype(cp.float32) * 255.0 / max_val, 0, 255
        ).astype(cp.uint8)
        image_grayscale = cp.repeat(image_normalized[:, :, None], 3, axis=2)
        return image_grayscale

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message = op_input.receive("input")
        msg = in_message.get("")
        cp_frame = cp.asarray(msg)
        cp_frame_u8 = (cp_frame >> 8).astype(cp.uint8)

        # logging.info("compute")
        # Extract  the data from the stream
        if self._no_of_planes == 3:

            # -----------------------------------------------------------------------
            # Unpack ADI ToF subframe-planar layout (mirrors unpack_kernel in .cu):
            #
            #   Subframe 1 (N×3 bytes): [D_L][D_H][C] for each of N = H×W pixels
            #   Subframe 2 (N×2 bytes): [AB_L][AB_H] for each of N pixels
            #
            # Total = N×5 bytes (NOT 5 interleaved bytes per pixel).
            # -----------------------------------------------------------------------
            N = self._height * self._width
            raw_flat = cp_frame_u8.reshape(-1)  # flatten to 1D (N*5 bytes)

            # Subframe 1: depth (uint16 LE) + confidence (uint8) — 3 bytes per pixel
            sf1 = raw_flat[: N * 3].reshape(N, 3)
            depth = (
                sf1[:, 0].astype(cp.uint16) | (sf1[:, 1].astype(cp.uint16) << 8)
            ).reshape(self._height, self._width)
            conf = sf1[:, 2].astype(cp.uint16).reshape(self._height, self._width)

            # Subframe 2: active brightness (uint16 LE) — 2 bytes per pixel
            sf2 = raw_flat[N * 3 : N * 5].reshape(N, 2)
            active_brightness = (
                sf2[:, 0].astype(cp.uint16) | (sf2[:, 1].astype(cp.uint16) << 8)
            ).reshape(self._height, self._width)

        else:
            raw = cp_frame_u8.reshape(-1)  # flatten to 1D (N*5 bytes)
            depth = raw[:, :, 0].astype(cp.uint16) | (
                raw[:, :, 1].astype(cp.uint16) << 8
            )
            active_brightness = raw[:, :, 2].astype(cp.uint16) | (
                raw[:, :, 3].astype(cp.uint16) << 8
            )
        if self._save == 1:
            # dump or save once frame of data, executed only once
            cp_frame_u8.astype("uint8").tofile("dump.bin")
            depth.astype("uint16").tofile("depth.bin")
            if self._no_of_planes == 3:
                conf.astype("uint16").tofile("conf.bin")

            active_brightness.astype("uint16").tofile("ab.bin")
            self._save = 0

        depth_c = self.converttojetimage(depth)
        active_brightness_c = self.convert_to_grayscale(active_brightness)
        if self._no_of_planes == 3:
            conf_c = self.convert_to_grayscale(conf)

        if self._no_of_planes == 1:
            op_output.emit(
                {"Depth": cp_frame_u8}, "output"
            )  # CHECK: This is for raw data passing
        elif self._no_of_planes == 2:
            op_output.emit(
                {"Depth": depth_c, "ActiveBrightness": active_brightness_c}, "output"
            )
        elif self._no_of_planes == 3:
            op_output.emit(
                {
                    "Depth": depth_c,
                    "ActiveBrightness": active_brightness_c,
                    "Conf": conf_c,
                },
                "output",
            )


class HoloscanApplication(holoscan.core.Application):
    def __init__(
        self,
        headless,
        fullscreen,
        cuda_context,
        cuda_device_ordinal,
        hololink_channel,
        ibv_name,
        ibv_port,
        adcam_inst,
        frame_limit,
        # mode,
    ):
        logging.info("__init__")
        super().__init__()
        self._headless = headless
        self._fullscreen = fullscreen
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._hololink_channel = hololink_channel
        self._ibv_name = ibv_name
        self._ibv_port = ibv_port
        self._adcam_inst = adcam_inst
        self._frame_limit = frame_limit
        self._mode = 6

    def compose(self):
        if self._frame_limit:
            self._count = holoscan.conditions.CountCondition(
                self,
                name="count",
                count=self._frame_limit,
            )
            condition = self._count
        else:
            self._ok = holoscan.conditions.BooleanCondition(
                self, name="ok", enable_tick=True
            )
            condition = self._ok

        # self._adcam_inst.set_param(self._mode)
        self._adcam_inst.set_mipi()
        self._adcam_inst.set_mode()
        csi_to_bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="pool",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=self._adcam_inst._width
            * self._adcam_inst._byteperpixel
            * ctypes.sizeof(ctypes.c_uint16)
            * self._adcam_inst._height,
            num_blocks=2,
        )
        csi_to_bayer_operator = hololink_module.operators.CsiToBayerOp(
            self,
            name="csi_to_bayer",
            allocator=csi_to_bayer_pool,
            cuda_device_ordinal=self._cuda_device_ordinal,
        )
        self._adcam_inst.configure_converter(csi_to_bayer_operator)

        frame_size = csi_to_bayer_operator.get_csi_length()
        logging.info(f"{frame_size=}")
        frame_context = self._cuda_context

        if self._ibv_name is not None:
            receiver_operator = hololink_module.operators.RoceReceiverOp(
                self,
                condition,
                name="receiver",
                frame_size=frame_size,
                frame_context=frame_context,
                ibv_name=self._ibv_name,
                ibv_port=self._ibv_port,
                hololink_channel=self._hololink_channel,
                device=self._adcam_inst,
            )
        else:
            receiver_operator = hololink_module.operators.LinuxReceiverOperator(
                self,
                condition,
                name="receiver",
                frame_size=frame_size,
                frame_context=frame_context,
                hololink_channel=self._hololink_channel,
                device=self._adcam_inst,
            )

        num_planes = self._adcam_inst.get_num_planes()

        ADIToF_data = ADTFUnpackOp(
            self,
            name="ADIToF_data",
            no_of_planes=num_planes,
            width=self._adcam_inst.get_pixel_width(),
            height=self._adcam_inst.get_pixel_height(),
            size=self._adcam_inst.get_pixelsize(),
        )

        if num_planes == 2:
            left_spec = holoscan.operators.HolovizOp.InputSpec(
                "Depth", holoscan.operators.HolovizOp.InputType.COLOR
            )
            left_spec_view = holoscan.operators.HolovizOp.InputSpec.View()
            left_spec_view.offset_x = 0
            left_spec_view.offset_y = 0
            left_spec_view.width = 0.50
            left_spec_view.height = 1
            left_spec.views = [left_spec_view]

            center_spec = holoscan.operators.HolovizOp.InputSpec(
                "ActiveBrightness", holoscan.operators.HolovizOp.InputType.COLOR
            )
            center_spec_view = holoscan.operators.HolovizOp.InputSpec.View()
            center_spec_view.offset_x = 0.50
            center_spec_view.offset_y = 0
            center_spec_view.width = 0.51
            center_spec_view.height = 1
            center_spec.views = [center_spec_view]

            window_height = 1920
            window_width = 2048  # for the pair
            window_title = "ADI ToF Player"
            visualizer = holoscan.operators.HolovizOp(
                self,
                name="holoviz",
                headless=self._headless,
                framebuffer_srgb=True,
                # tensors=[left_spec],
                # tensors=[left_spec, center_spec],
                tensors=[left_spec, center_spec],
                height=window_height,
                width=window_width,
                window_title=window_title,
            )

        else:

            left_spec = holoscan.operators.HolovizOp.InputSpec(
                "Depth", holoscan.operators.HolovizOp.InputType.COLOR
            )
            left_spec_view = holoscan.operators.HolovizOp.InputSpec.View()
            left_spec_view.offset_x = 0
            left_spec_view.offset_y = 0
            left_spec_view.width = 0.33
            left_spec_view.height = 1
            left_spec.views = [left_spec_view]

            center_spec = holoscan.operators.HolovizOp.InputSpec(
                "ActiveBrightness", holoscan.operators.HolovizOp.InputType.COLOR
            )
            center_spec_view = holoscan.operators.HolovizOp.InputSpec.View()
            center_spec_view.offset_x = 0.33
            center_spec_view.offset_y = 0
            center_spec_view.width = 0.33
            center_spec_view.height = 1
            center_spec.views = [center_spec_view]

            right_spec = holoscan.operators.HolovizOp.InputSpec(
                "Conf", holoscan.operators.HolovizOp.InputType.COLOR
            )
            right_spec_view = holoscan.operators.HolovizOp.InputSpec.View()
            right_spec_view.offset_x = 0.66
            right_spec_view.offset_y = 0
            right_spec_view.width = 0.34
            right_spec_view.height = 1
            right_spec.views = [right_spec_view]

            window_height = 1920
            window_width = 2048  # for the pair
            window_title = "ADI ToF Player"
            visualizer = holoscan.operators.HolovizOp(
                self,
                name="holoviz",
                headless=self._headless,
                framebuffer_srgb=True,
                # tensors=[left_spec],
                # tensors=[left_spec, center_spec],
                tensors=[left_spec, center_spec, right_spec],
                height=window_height,
                width=window_width,
                window_title=window_title,
            )

        self.add_flow(receiver_operator, csi_to_bayer_operator, {("output", "input")})
        self.add_flow(csi_to_bayer_operator, ADIToF_data, {("output", "input")})
        self.add_flow(ADIToF_data, visualizer, {("output", "receivers")})


def int_or_none(value):
    if value == "None":
        return None
    return int(value)


def main():
    # Get a handle to the Hololink port we're connected to.
    parser = argparse.ArgumentParser(
        description="ADITOF Holoscan application parseing arguments"
    )

    # Define arguments
    parser.add_argument(
        "--resetAdcam",
        "-r",
        type=int,
        default=0,
        required=False,
        help="Power on Reset ADCAM module",
    )
    parser.add_argument(
        "--capture",
        "-c",
        type=int,
        default=-1,
        required=False,
        help="Capture ADCAM streams",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose mode"
    )
    parser.add_argument(
        "--resetOnly",
        "-RO",
        type=int,
        default=0,
        required=False,
        help="Soft Reset ADCAM module",
    )
    parser.add_argument(
        "--getStatus",
        "-gs",
        type=int,
        default=0,
        required=False,
        help="Get status part of debug",
    )

    parser.add_argument(
        "--captureMode",
        type=int,
        default=6,
        required=False,
        help="Capture mode index (0-9, default 6)",
    )

    parser.add_argument(
        "--resetPin",
        type=int,
        default=0,
        required=False,
        help="GPIO reset pin number (0-31, default 0)",
    )

    parser.add_argument(
        "--numPlanes",
        type=int,
        default=3,
        required=False,
        help="numPlanes (2 (Depth + AB) or 3 (Depth + AB + Conf))",
    )

    parser.add_argument(
        "--captureFps",
        type=int,
        default=30,
        required=False,
        help="Adcam Capture FPS (some FPS may not work), default 30",
    )

    parser.add_argument(
        "--metadata",
        type=int,
        default=0,
        required=False,
        help="Metadata to be removed from MIPI receive, refer readme, default 0",
    )

    parser.add_argument(
        "--maxMipi",
        type=int,
        default=1,
        required=False,
        help="Max supported Mipi per lane speed (default 1=2.5Gbps,2=2Gbps, 3= 1.5Gbps, 4=1Gbps supported), refer readme",
    )

    parser.add_argument(
        "--log-level",
        default="info",
        required=False,
        help="Logging level: trace/debug/info/warn/error (default info)",
    )

    parser.add_argument(
        "--firmwareUpdate",
        default=None,
        required=False,
        metavar="MANIFEST.yaml",
        help="Path to firmware manifest YAML file (e.g. adi_manifest.yaml)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Allow firmware downgrade (requires --firmwareUpdate)",
    )

    # Parse arguments
    # args = parser.parse_args()

    parser.add_argument(
        "--frame-limit",
        type=int_or_none,
        default=300,
        help="Exit after receiving this many frames",
    )

    infiniband_devices = hololink_module.infiniband_devices()

    #  check if we have infiniband_devices or regular linux network
    if infiniband_devices and len(infiniband_devices) > 0:

        parser.add_argument(
            "--ibv-name",
            default=infiniband_devices[0],
            help="IBV device to use",
        )
        parser.add_argument(
            "--ibv-port",
            type=int,
            default=1,
            help="Port number of IBV device",
        )
    else:
        logging.info("No Infiniband devices found, using linux player")
        parser.add_argument(
            "--ibv-name",
            default=None,
            help="Network device to use",
        )
        parser.add_argument(
            "--ibv-port",
            type=int,
            default=0,
            help="Port number of device",
        )

    args = parser.parse_args()

    hololink_module.logging_level(2)
    logging.info("Initializing.")

    # Apply log level — use hololink's logging_level() instead of basicConfig()
    # to avoid overriding the hololink formatter that injects log_timestamp_s.
    _log_level_map = {
        "trace": logging.DEBUG,
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warn": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    logging.getLogger().setLevel(
        _log_level_map.get(args.log_level.lower(), logging.INFO)
    )

    # Validate captureMode range
    if not (0 <= args.captureMode <= 9):
        print(f"Error: --captureMode must be 0-9, got {args.captureMode}")
        sys.exit(1)
    # Validate resetPin range
    if not (0 <= args.resetPin <= 31):
        print(f"Error: --resetPin must be 0-31, got {args.resetPin}")
        sys.exit(1)

    (cu_result,) = cuda.cuInit(0)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_device_ordinal = 0
    cu_result, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_result, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS

    logging.info("starting")
    channel_metadata = hololink_module.Enumerator.find_channel(channel_ip="192.168.0.2")
    logging.info(f"{channel_metadata=}")
    hololink_channel = hololink_module.DataChannel(channel_metadata)
    # Instantiate the adcam_inst itself; CAM_I2C_BUS is the appropriate bus enable setting
    # for the I2C controller our adcam_inst is attached to
    adcam_inst = adcam.adcam(
        hololink_channel,
        hololink_module.CAM_I2C_BUS,
        channel_metadata,
        adcam_mode=args.captureMode,
        reset_pin=args.resetPin,
        num_planes=args.numPlanes,
        tof_fps=args.captureFps,
        metadata_sz=args.metadata,
        mipi_lane_speed=args.maxMipi,
    )

    # Establish a connection to the hololink device
    hololink = hololink_channel.hololink()
    hololink.start()

    if args.resetAdcam == 1:
        logging.info("Doing the full Reset including power on sequence")
        adcam_inst.adcam_reset_power_on()

    if args.resetOnly == 1:
        logging.info("Performing ONLY Reset - NOT doing FULL Power on reset")
        adcam_inst.adcam_Only_reset()

    # check if the chip exists (0x0112) before querying imager type (0x0032)
    if adcam_inst.probe_adcam_adtf3175() != 1:
        logging.warning(
            "ADCAM not responding; performing automatic power-on reset and retrying..."
        )
        adcam_inst.adcam_reset_power_on()
        if adcam_inst.probe_adcam_adtf3175() != 1:
            logging.error(
                "No ADCAM ADTF3175 found after reset, connect ADCAM and try again"
            )
        hololink.stop()
        exit()

    # Detect imager type after the device is confirmed present
    adcam_inst.get_imager_type_and_ccb_version()

    # Fetch the device version.
    if args.getStatus == 1:
        logging.debug("Getting only status")
        adcam_inst.get_status()

    # ---------------------------------------------------------------------------
    # Firmware update via YAML manifest (mirrors C++ Programmer flow)
    # ---------------------------------------------------------------------------
    if args.firmwareUpdate is not None:
        manifest_path = args.firmwareUpdate
        if not os.path.exists(manifest_path):
            print(f"Error: manifest file not found: {manifest_path}")
            sys.exit(1)

        print(f"Loading firmware manifest: {manifest_path}")
        with open(manifest_path, "rt") as f:
            manifest = yaml.safe_load(f)
        section = manifest.get("hololink")
        if section is None:
            print("Error: manifest missing 'hololink' section")
            sys.exit(1)

        def _fetch_content(content_name):
            """Download or read a content entry, verify md5+size, return bytes."""
            meta = section["content"][content_name]
            expected_md5 = meta["md5"]
            expected_size = meta["size"]
            if "url" in meta:
                url = meta["url"]
                print(f"Downloading {content_name} from {url} ...")
                resp = requests.get(
                    url,
                    headers={"Content-Type": "binary/octet-stream"},
                    timeout=120,
                )
                if resp.status_code != 200:
                    raise RuntimeError(
                        f'Unable to fetch "{url}"; HTTP {resp.status_code}'
                    )
                data = resp.content
            elif "filename" in meta:
                with open(meta["filename"], "rb") as fh:
                    data = fh.read()
            else:
                raise RuntimeError(
                    f"No source for content '{content_name}' in manifest"
                )
            if len(data) != expected_size:
                raise RuntimeError(
                    f"{content_name}: expected {expected_size} bytes, got {len(data)}"
                )
            actual_md5 = hashlib.md5(data).hexdigest()
            if actual_md5.lower() != expected_md5.lower():
                raise RuntimeError(
                    f"{content_name}: MD5 mismatch (expected {expected_md5}, got {actual_md5})"
                )
            return data

        # EULA check
        licenses = section.get("licenses")
        if licenses and not getattr(args, "accept_eula", False):
            print("You must accept EULA terms in order to continue.")
            print("For each document, press <Space> to see the next page;")
            print("At the end of the document, enter <Q> to continue.")
            input("To continue, press <Enter>: ")
            for lic_name in licenses:
                lic_text = _fetch_content(lic_name).decode(errors="replace")
                pydoc.pager(lic_text)
                answer = input(
                    "Press 'y' or 'Y' to accept this end user license agreement: "
                )
                if not answer.strip().upper().startswith("Y"):
                    print("EULA not accepted. Aborting.")
                    hololink.stop()
                    sys.exit(1)

        # Fetch firmware images
        content = {}
        for img in section.get("images", []):
            ctx = img["context"]
            cname = img["content"]
            print(f"Fetching image: context={ctx} content={cname}")
            content[ctx] = _fetch_content(cname)

        fw_bin = content.get("adcam")
        if fw_bin is None:
            print("Error: manifest has no 'adcam' context image")
            sys.exit(1)
        print(f"Firmware binary: {len(fw_bin)} bytes")

        result = adcam_inst.adsd3500_flash(fw_bin, force=args.force)
        if result:
            print("Firmware update successful!")
        else:
            print("Firmware update failed.")
        hololink.stop()
        sys.exit(0)

    # Read master and slave firmware versions in a single burst session
    adcam_inst.switch_from_standard_to_burst()
    version = adcam_inst.get_fw_version_burst_mode(adcam.GET_MASTER_FIRMWARE_COMMAND)
    logging.info(f"{version=}")
    adcam_inst.get_fw_version_burst_mode(adcam.GET_SLAVE_FIRMWARE_COMMAND)
    adcam_inst.switch_from_burst_to_standard()

    if args.capture == 1:
        # Set up the application
        application = HoloscanApplication(
            False,
            True,
            cu_context,
            cu_device_ordinal,
            hololink_channel,
            args.ibv_name,
            args.ibv_port,
            adcam_inst,
            args.frame_limit,
        )
        application.run()
    elif args.capture == 2:
        logging.debug("Force stop capture..")
        adcam_inst.stream_off()

    hololink.stop()


if __name__ == "__main__":
    main()
