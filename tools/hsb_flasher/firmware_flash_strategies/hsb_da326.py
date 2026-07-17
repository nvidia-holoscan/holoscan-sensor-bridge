#!/usr/bin/env python3
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

"""
HSB_Da326 Flash Strategy Module

This module routes a device firmware version to a flash strategy identified by
its MIN_VERSION (the lowest version that strategy covers). A device at version V
is dispatched to the strategy with the largest MIN_VERSION <= V.
"""

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


def _setup_flash_strategies():
    """Add hsb_da326_flash_strategies to path and import."""
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    import hsb_da326_flash_strategies

    return hsb_da326_flash_strategies


_flash_strategies = _setup_flash_strategies()


class HSBDa326FlasherBase(ABC):
    """Base class for HSB_Da326 flashers."""

    FPGA_UUID = "9957a9ac-36b5-4518-83ec-5d514aecb750"

    MIN_VERSION: int = 0

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "__abstractmethods__", None) and not isinstance(
            cls.__dict__.get("MIN_VERSION"), int
        ):
            raise TypeError(f"{cls.__name__} must define a MIN_VERSION class attribute")

    def __init__(self, ip_address: str, mac_address: str, version: int):
        self.ip_address = ip_address
        self.mac_address = mac_address
        self.version = version

    @classmethod
    def supports(cls, fpga_uuid: str, version: int) -> bool:
        return fpga_uuid == cls.FPGA_UUID and version >= cls.MIN_VERSION

    @abstractmethod
    def flash(self, clnx_path: str, cpnx_path: str) -> bool:
        """Flash the device. Override in subclass."""
        pass

    def establish_board_state(self):
        pass


class HSBDa326Flasher(HSBDa326FlasherBase):
    """Modern connection-based flash for DA326 (hsb_ip_version >= 0x2511)."""

    MIN_VERSION = 0x2511

    def flash(self, _clnx_path: str, cpnx_path: str) -> bool:
        return _flash_strategies.hsb_da326_flash_2511(self.ip_address, cpnx_path)


FLASH_STRATEGIES = sorted(
    [HSBDa326Flasher],
    key=lambda c: c.MIN_VERSION,
    reverse=True,
)


def get_flasher(
    fpga_uuid: str, version: int, ip_address: str, mac_address: str
) -> Optional[HSBDa326FlasherBase]:
    for flasher_cls in FLASH_STRATEGIES:
        if flasher_cls.supports(fpga_uuid, version):
            return flasher_cls(
                ip_address=ip_address, mac_address=mac_address, version=version
            )
    return None


def supports(fpga_uuid: str, version: int) -> bool:
    return any(cls.supports(fpga_uuid, version) for cls in FLASH_STRATEGIES)


def do_flash(
    fpga_uuid: str,
    version: int,
    mac_address: str,
    ip_address: str,
    clnx_path: str,
    cpnx_path: str,
) -> bool:
    flasher = get_flasher(fpga_uuid, version, ip_address, mac_address)
    if flasher is None:
        print(f"[hsb_da326] No flasher for uuid={fpga_uuid} version=0x{version:04x}")
        return False

    print(f"[hsb_da326] Using {flasher.__class__.__name__}")
    print(f"  FPGA UUID: {fpga_uuid}")
    print(f"  Version: 0x{version:04x}")
    print(f"  MAC: {mac_address}")
    print(f"  IP: {ip_address}")
    print(f"  CLNX: {clnx_path}")
    print(f"  CPNX: {cpnx_path}")

    try:
        flasher.establish_board_state()
        success = flasher.flash(clnx_path, cpnx_path)
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as e:
        print(f"[hsb_da326] ERROR: {e}")
        return False
    else:
        if success:
            print("[hsb_da326] Flash completed successfully")
        else:
            print("[hsb_da326] Flash failed")
        return success
