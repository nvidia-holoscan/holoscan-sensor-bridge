# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# include the operators
if(DEFINED ENV{CONTAINER_TYPE} AND "$ENV{CONTAINER_TYPE}" STREQUAL "igpu")
  add_subdirectory(argus_isp)
endif()
add_subdirectory(audio_packetizer)
add_subdirectory(csi_to_bayer)
add_subdirectory(image_processor)
add_subdirectory(iq_dec)
add_subdirectory(iq_enc)
add_subdirectory(linux_receiver)
add_subdirectory(linux_coe_receiver)
add_subdirectory(packed_format_converter)
add_subdirectory(roce_receiver)
add_subdirectory(roce_transmitter)
add_subdirectory(sig_gen)
add_subdirectory(sig_viewer)
if(HOLOLINK_BUILD_SIPL)
  add_subdirectory(sipl_capture)
endif()
add_subdirectory(udp_transmitter)
