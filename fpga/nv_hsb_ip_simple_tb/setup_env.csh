#!/bin/csh -f
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

# Common NV HSB IP Simple Testbench environment.
# Source this file from the testbench directory.

if (! $?HOLOLINK_TB_DIR) then
    setenv HOLOLINK_TB_DIR "$cwd"
endif

######################################################
# HSB IP RTL and HOLOLINK_DEF directory
# Update this to correct directory and uncomment
#setenv HOLOLINK_PATH "${HOLOLINK_TB_DIR}/"
#setenv HOLOLINK_DEF_INCLUDE "${HOLOLINK_TB_DIR}/"
######################################################

# FSDB DUMP
setenv DUMP 1

# Simulator selection: VCS or XCELIUM.
setenv SIMULATOR "VCS"

# Synopsys VCS/Verdi environment.
setenv VCS_HOME "/home/tools/vcs/2023.03-SP2-8"
setenv DESIGNWARE_HOME "/home/tools/synopsys/designware_home"
setenv VERDI_HOME "/home/tools/debussy/verdi3_2023.03-SP2-8"

# Set this when SIMULATOR is XCELIUM.
setenv XCELIUM_HOME ""

setenv PATH "${VCS_HOME}/bin:${VERDI_HOME}/bin:${PATH}"
if ("${XCELIUM_HOME}" != "") then
    setenv PATH "${XCELIUM_HOME}/tools/bin:${PATH}"
endif
