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
HSB_Leopard Flash Strategy Module

This module contains flasher classes for the HSB_Leopard.
Each class handles exactly one version. The base class handles the fpga_uuid.
The arbiter function `get_flasher()` returns the appropriate class instance
for a given version.
"""

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

# =============================================================================
# Setup flash_strategies path
# =============================================================================


def _setup_flash_strategies():
    """Add hsb_leopard_flash_strategies to path and import."""
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    import hsb_leopard_flash_strategies

    return hsb_leopard_flash_strategies


_flash_strategies = _setup_flash_strategies()


# =============================================================================
# Base Class
# =============================================================================


class HSBLeopardFlasherBase(ABC):
    """Base class for HSB_Leopard flashers."""

    FPGA_UUID = "f1627640-b4dc-48af-a360-c55b09b3d230"

    VERSION: int

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "__abstractmethods__", None) and not isinstance(
            cls.__dict__.get("VERSION"), int
        ):
            raise TypeError(f"{cls.__name__} must define a VERSION class attribute")

    def __init__(self, ip_address: str, mac_address: str):
        self.ip_address = ip_address
        self.mac_address = mac_address

    @classmethod
    def supports(cls, fpga_uuid: str, version: int) -> bool:
        """Check if this flasher supports the given fpga_uuid and version."""
        return fpga_uuid == cls.FPGA_UUID and version == cls.VERSION

    @abstractmethod
    def flash(self, _clnx_path: str, cpnx_path: str) -> bool:
        """Flash the device. Override in subclass."""
        pass


# =============================================================================
# Flasher Implementations
# =============================================================================


class HSBLeopardFlasher2603(HSBLeopardFlasherBase):
    """Flasher for HSB_Leopard version 0x2603."""

    VERSION = 0x2603

    def flash(self, _clnx_path: str, cpnx_path: str) -> bool:
        return _flash_strategies.hsb_leopard_flash_2507(self.ip_address, cpnx_path)


class HSBLeopardFlasher2510(HSBLeopardFlasherBase):
    """Flasher for HSB_Leopard version 0x2510."""

    VERSION = 0x2510

    def flash(self, _clnx_path: str, cpnx_path: str) -> bool:
        return _flash_strategies.hsb_leopard_flash_2507(self.ip_address, cpnx_path)


class HSBLeopardFlasher2507(HSBLeopardFlasherBase):
    """Flasher for HSB_Leopard version 0x2507."""

    VERSION = 0x2507

    def flash(self, _clnx_path: str, cpnx_path: str) -> bool:
        return _flash_strategies.hsb_leopard_flash_2507(self.ip_address, cpnx_path)


# =============================================================================
# Registry of all flashers in this module
# =============================================================================

FLASH_STRATEGIES = [
    HSBLeopardFlasher2603,
    HSBLeopardFlasher2510,
    HSBLeopardFlasher2507,
]


# =============================================================================
# Arbiter - called by C++ to get the right flasher
# =============================================================================


def get_flasher(
    fpga_uuid: str, version: int, ip_address: str, mac_address: str
) -> Optional[HSBLeopardFlasherBase]:
    """
    Arbiter function: returns the appropriate flasher instance for the given
    fpga_uuid and version, or None if not supported.
    """
    for flasher_cls in FLASH_STRATEGIES:
        if flasher_cls.supports(fpga_uuid, version):
            return flasher_cls(ip_address=ip_address, mac_address=mac_address)
    return None


def supports(fpga_uuid: str, version: int) -> bool:
    """
    Check if this module has a flasher for the given fpga_uuid and version.
    Called by C++ for discovery.
    """
    return any(cls.supports(fpga_uuid, version) for cls in FLASH_STRATEGIES)


def do_flash(
    fpga_uuid: str,
    version: int,
    mac_address: str,
    ip_address: str,
    clnx_path: str,
    cpnx_path: str,
) -> bool:
    """
    Main entry point called by C++.
    Gets the appropriate flasher and executes the flash.
    """
    flasher = get_flasher(fpga_uuid, version, ip_address, mac_address)
    if flasher is None:
        print(f"[hsb_leopard] No flasher for uuid={fpga_uuid} version=0x{version:04x}")
        return False

    print(f"[hsb_leopard] Using {flasher.__class__.__name__}")
    print(f"  FPGA UUID: {fpga_uuid}")
    print(f"  Version: 0x{version:04x}")
    print(f"  MAC: {mac_address}")
    print(f"  IP: {ip_address}")
    print(f"  CPNX: {cpnx_path}")

    try:
        success = flasher.flash(clnx_path, cpnx_path)
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as e:
        print(f"[hsb_leopard] ERROR: {e}")
        return False
    else:
        if success:
            print("[hsb_leopard] Flash completed successfully")
        else:
            print("[hsb_leopard] Flash failed")
        return success
