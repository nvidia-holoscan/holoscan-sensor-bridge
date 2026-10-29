# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Install the tools
install(FILES
        hsb-ptp.conf
        generate_manifest.py
        manifest.yaml
        manifest_leopard_cpnx100.yaml
        DESTINATION scripts
        COMPONENT hololink-scripts)

# Installs/reloads the hololink_pcie kernel driver (locally or over ssh) and makes it
# load at boot; see src/ehsb/kmod/README.md. Part of the PCIe transport.
if(HOLOLINK_BUILD_PCIE)
  install(PROGRAMS
          deploy_pcie_kmod.sh
          DESTINATION scripts
          COMPONENT hololink-scripts)
endif()
