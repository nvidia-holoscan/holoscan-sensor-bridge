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

SUMMARY = "HSB SPI Daemon"
LICENSE = "Apache-2.0"
LIC_FILES_CHKSUM = "file://${WORKDIR}/hsb-spi-daemon.py;endline=16;md5=5b972fff81771bec23c8ba38b7531983"

PACKAGES = "${PN}"

inherit update-rc.d
INITSCRIPT_NAME = "hsb-spi-daemon"
INITSCRIPT_PARAMS = "defaults 99"

SRC_URI = " \
    file://hsb.conf \
    file://hsb-spi-daemon.py \
    file://hsb-spi-daemon \
"

do_install () {
    install -d ${D}/etc/modprobe.d
    install -m 0644 ${WORKDIR}/hsb.conf ${D}/etc/modprobe.d
    install -d ${D}/opt/nvidia
    install -m 0755 ${WORKDIR}/hsb-spi-daemon.py ${D}/opt/nvidia
    install -d ${D}/${sysconfdir}/init.d
    install -m 0755 ${WORKDIR}/hsb-spi-daemon ${D}/${sysconfdir}/init.d
}

FILES:${PN} += " \
    /etc/modprobe.d \
    /opt/nvidia \
"

RDEPENDS:${PN} += " \
    bash \
    hsb-spi \
    python3-core \
    python3-logging \
"
