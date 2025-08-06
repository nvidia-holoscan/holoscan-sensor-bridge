# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

_MODULES = {
    "ArgusIspOp": "argus_isp",
    "AudioPacketizerOp": "audio_packetizer",
    "BaseReceiverOp": "base_receiver_op",
    "CsiToBayerOp": "csi_to_bayer",
    "ImageProcessorOp": "image_processor",
    "ImageShiftToUint8Operator": "image_shift_to_uint8_operator",
    "IQDecoderOp": "iq_dec",
    "IQEncoderOp": "iq_enc",
    "LinuxCoeReceiver": "linux_coe_receiver",
    "LinuxCoeReceiverOp": "linux_coe_receiver_operator",
    "LinuxReceiver": "linux_receiver",
    "LinuxReceiverOperator": "linux_receiver_operator",
    "PackedFormatConverterOp": "packed_format_converter",
    "Rational": "sig_gen",
    "RoceReceiverOp": "roce_receiver",
    "RoceTransmitterOp": "roce_transmitter",
    "SignalGeneratorOp": "sig_gen",
    "SignalViewerOp": "sig_viewer",
    "SIPLCaptureOp": "sipl_capture",
    "UdpTransmitterOp": "udp_transmitter",
}

__all__ = []

__all__.extend(_MODULES.keys())


# Autocomplete
def __dir__():
    return __all__


# Lazily load modules and classes
def __getattr__(attr):
    if attr in _MODULES:
        module_name = ".".join([__name__, _MODULES[attr]])
        if module_name in sys.modules:  # cached
            module = sys.modules[module_name]
        else:
            module = importlib.import_module(module_name)  # import
            sys.modules[module_name] = module  # cache
        return getattr(module, attr)
    raise AttributeError(f"module {__name__} has no attribute {attr}")
