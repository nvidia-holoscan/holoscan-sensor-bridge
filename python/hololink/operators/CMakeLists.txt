# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

copy_hololink_python_files(
    TARGET_NAME copy_hololink_operators_python
    DESTINATION operators
    FILES
        __init__.py
        base_receiver_op.py
        hsb_controller_op.py
        image_shift_to_uint8_operator.py
        linux_coe_receiver_operator.py
        linux_controller_receiver.py
        linux_receiver_operator.py
        roce_controller_receiver.py
)

# include the operators
if(HOLOLINK_BUILD_ARGUS_ISP)
  add_subdirectory(argus_isp)
endif()
add_subdirectory(audio_packetizer)
add_subdirectory(compute_crc)
add_subdirectory(csi_to_bayer)
if(HOLOLINK_BUILD_FUSA)
  add_subdirectory(fusa_coe_capture)
endif()
add_subdirectory(image_processor)
add_subdirectory(iq_dec)
add_subdirectory(iq_enc)
add_subdirectory(linux_receiver)
add_subdirectory(linux_receiver_op)
add_subdirectory(linux_coe_receiver)
add_subdirectory(packed_format_converter)
if(HOLOLINK_BUILD_PVA)
  add_subdirectory(pva_crc)
endif()
if (HOLOLINK_BUILD_ROCE)
  add_subdirectory(roce_receiver)
  add_subdirectory(roce_transmitter)
  add_subdirectory(gpu_roce_transceiver)
  add_subdirectory(sig_gen)
  add_subdirectory(sig_viewer)
endif() # HOLOLINK_BUILD_ROCE
if(HOLOLINK_BUILD_SIPL)
  add_subdirectory(sipl_capture)
endif()
add_subdirectory(sub_frame_combiner)
add_subdirectory(udp_transmitter)
