# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import importlib
import sys

_OBJECTS = {
    "ArgusIspOp": "argus_isp",
    "AudioPacketizerOp": "audio_packetizer",
    "BaseReceiverOp": "base_receiver_op",
    "CheckCrcOp": "compute_crc",
    "CheckPvaCrcOp": "pva_crc",
    "ComputeCrcOp": "compute_crc",
    "ComputePvaCrcOp": "pva_crc",
    "CsiToBayerOp": "csi_to_bayer",
    "FusaCoeCaptureOp": "fusa_coe_capture",
    "HsbControllerOp": "hsb_controller_op",
    "ImageProcessorOp": "image_processor",
    "ImageShiftToUint8Operator": "image_shift_to_uint8_operator",
    "IQDecoderOp": "iq_dec",
    "IQEncoderOp": "iq_enc",
    "LinuxCoeReceiver": "linux_coe_receiver",
    "LinuxCoeReceiverOp": "linux_coe_receiver_operator",
    "LinuxReceiver": "linux_receiver",
    "LinuxReceiverOp": "linux_receiver_op",
    "LinuxReceiverOperator": "linux_receiver_operator",
    "PackedFormatConverterOp": "packed_format_converter",
    "Rational": "sig_gen",
    "RoceReceiver": "roce_receiver",
    "RoceReceiverOp": "roce_receiver",
    "RoceTransmitterOp": "roce_transmitter",
    "GpuRoceTransceiverOp": "gpu_roce_transceiver",
    "SignalGeneratorOp": "sig_gen",
    "SignalViewerOp": "sig_viewer",
    "SIPLCaptureService": "sipl_capture",
    "SIPLCameraOutputOp": "sipl_capture",
    "SubFrameCombinerOp": "sub_frame_combiner",
    "UdpTransmitterOp": "udp_transmitter",
}

_MODULES = [
    "linux_controller_receiver",
    "roce_controller_receiver",
]


__all__ = []

__all__.extend(_MODULES)
__all__.extend(_OBJECTS.keys())


# Autocomplete
def __dir__():
    return __all__


# Lazily load modules and classes
def __getattr__(attr):
    if attr in _OBJECTS:
        module_name = ".".join([__name__, _OBJECTS[attr]])
        if module_name in sys.modules:  # cached
            module = sys.modules[module_name]
        else:
            module = importlib.import_module(module_name)  # import
            sys.modules[module_name] = module  # cache
        return getattr(module, attr)
    if attr in _MODULES:
        module_name = f"{__name__}.{attr}"
        if module_name in sys.modules:  # cached
            module = sys.modules[module_name]
        else:
            module = importlib.import_module(module_name)  # import
            sys.modules[module_name] = module  # cache
        return module
    raise AttributeError(f"module {__name__} has no attribute {attr}")
