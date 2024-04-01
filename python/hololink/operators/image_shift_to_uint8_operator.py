# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# See README.md for detailed information.

import logging

import cupy as cp
import holoscan


@cp.fuse()
def shift_to_uint8(img, shift):
    img = img >> shift
    return img.astype(cp.uint8)


class ImageShiftToUint8Operator(holoscan.core.Operator):
    def __init__(self, *args, shift=0, **kwargs):
        super().__init__(*args, **kwargs)
        self._shift = shift

    def setup(self, spec):
        logging.info("setup")
        spec.input("input")
        spec.output("output")

    def start(self):
        pass

    def stop(self):
        pass

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message = op_input.receive("input")
        cp_frame = cp.from_dlpack(in_message.get(""))
        cp_frame_uint8 = shift_to_uint8(cp_frame, self._shift)
        op_output.emit({"": cp_frame_uint8}, "output")
