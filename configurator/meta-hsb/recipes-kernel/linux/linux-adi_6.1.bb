# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

SUMMARY = "Analog Devices Linux kernel"
DESCRIPTION = "Linux kernel with ADI drivers and patches"
LICENSE = "GPL-2.0-only"
LIC_FILES_CHKSUM = "file://COPYING;md5=6bc538ed5bd9a7fc9398086aedcd7e46"

inherit kernel
require recipes-kernel/linux/linux-yocto.inc

LINUX_VERSION ?= "6.1"
LINUX_VERSION_EXTENSION = "-adi"
PV = "${LINUX_VERSION}+git${SRCPV}"

KERNELURI = "git://github.com/analogdevicesinc/linux.git;protocol=https;branch=${SRCBRANCH}"
YOCTO_META = "git://git.yoctoproject.org/yocto-kernel-cache;type=kmeta;name=meta;branch=yocto-6.1;destsuffix=yocto-meta"
SRC_URI = "${KERNELURI} ${YOCTO_META}"

SRCREV = "cf811cc22806b2c42783bf2ff32449bf19ffa237"
SRCREV_meta = "39618187603db317c764f9749bde644d7bc1a0b1"
SRCBRANCH = "adi-6.1.0"

KCONFIG_MODE = "alldefconfig"
KBUILD_DEFCONFIG = "defconfig"

SRC_URI += " \
    file://disable-broken-drivers.cfg \
    file://compile-drivers-as-modules.cfg \
"

# Copy and add all dts files to the build.
SRC_URI += "file://dts"
do_configure:append() {
    cp ${WORKDIR}/dts/*.dtsi ${S}/arch/arm64/boot/dts/
    for dts_file in ${WORKDIR}/dts/*.dts; do
        if [ -f "$dts_file" ]; then
            cp $dts_file ${S}/arch/arm64/boot/dts/
            dts_name=$(basename $dts_file)
            config_name=$(echo $dts_name | sed 's/\.dts$//' | tr '-' '_' | tr '[:lower:]' '[:upper:]')
            if ! grep -q "$dts_name" ${S}/arch/arm64/boot/dts/Makefile; then
                echo "dtb-\$(CONFIG_${config_name}) += ${dts_name%.dts}.dtb" >> ${S}/arch/arm64/boot/dts/Makefile
            fi
        fi
    done
}
