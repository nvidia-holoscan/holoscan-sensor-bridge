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
HSB_Lite Flash Strategy Module

This module contains flasher classes for the HSB_Lite.
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
    """Add flash_strategies to path and import."""
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    import flash_strategies

    return flash_strategies


_flash_strategies = _setup_flash_strategies()


# =============================================================================
# Base Class
# =============================================================================


class HSBLiteFlasherBase(ABC):
    """Base class for HSB_Lite flashers."""

    # All HSB_Lite flashers share this UUID
    FPGA_UUID = "889b7ce3-65a5-4247-8b05-4ff1904c3359"

    # Subclasses must define this
    VERSION: int = 0

    def __init__(self, ip_address: str, mac_address: str):
        self.ip_address = ip_address
        self.mac_address = mac_address

    @classmethod
    def supports(cls, fpga_uuid: str, version: int) -> bool:
        """Check if this flasher supports the given fpga_uuid and version."""
        return fpga_uuid == cls.FPGA_UUID and version == cls.VERSION

    @abstractmethod
    def flash(self, clnx_path: str, cpnx_path: str) -> bool:
        """Flash the device. Override in subclass."""
        pass


# =============================================================================
# Flasher Implementations
# =============================================================================


class HSBLiteFlasher2603(HSBLiteFlasherBase):
    """Flasher for HSB_Lite version 0x2603."""

    VERSION = 0x2603

    def flash(self, clnx_path: str, cpnx_path: str) -> bool:
        return _flash_strategies.hsb_lite_flash_2507(
            self.ip_address, clnx_path, cpnx_path
        )


class HSBLiteFlasher2601(HSBLiteFlasherBase):
    """Flasher for HSB_Lite version 0x2601."""

    VERSION = 0x2601

    def flash(self, clnx_path: str, cpnx_path: str) -> bool:
        return _flash_strategies.hsb_lite_flash_2507(
            self.ip_address, clnx_path, cpnx_path
        )


class HSBLiteFlasher2512(HSBLiteFlasherBase):
    """Flasher for HSB_Lite version 0x2512."""

    VERSION = 0x2512

    def flash(self, clnx_path: str, cpnx_path: str) -> bool:
        return _flash_strategies.hsb_lite_flash_2507(
            self.ip_address, clnx_path, cpnx_path
        )


class HSBLiteFlasher2511(HSBLiteFlasherBase):
    """Flasher for HSB_Lite version 0x2511."""

    VERSION = 0x2511

    def flash(self, clnx_path: str, cpnx_path: str) -> bool:
        return _flash_strategies.hsb_lite_flash_2507(
            self.ip_address, clnx_path, cpnx_path
        )


class HSBLiteFlasher2510(HSBLiteFlasherBase):
    """Flasher for HSB_Lite version 0x2510."""

    VERSION = 0x2510

    def flash(self, clnx_path: str, cpnx_path: str) -> bool:
        return _flash_strategies.hsb_lite_flash_2507(
            self.ip_address, clnx_path, cpnx_path
        )


class HSBLiteFlasher2508(HSBLiteFlasherBase):
    """Flasher for HSB_Lite version 0x2508."""

    VERSION = 0x2508

    def flash(self, clnx_path: str, cpnx_path: str) -> bool:
        return _flash_strategies.hsb_lite_flash_2507(
            self.ip_address, clnx_path, cpnx_path
        )


class HSBLiteFlasher2507(HSBLiteFlasherBase):
    """Flasher for HSB_Lite version 0x2507."""

    VERSION = 0x2507

    def flash(self, clnx_path: str, cpnx_path: str) -> bool:
        return _flash_strategies.hsb_lite_flash_2507(
            self.ip_address, clnx_path, cpnx_path
        )


# =============================================================================
# Legacy Flasher Implementations
# =============================================================================


class HSBLiteFlasher2506(HSBLiteFlasherBase):
    """Flasher for HSB_Lite version 0x2506."""

    VERSION = 0x2506

    def flash(self, clnx_path: str, cpnx_path: str) -> bool:
        return _flash_strategies.hsb_lite_flash_2504(
            self.ip_address,
            clnx_path,
            cpnx_path,
            self.VERSION,
            self.FPGA_UUID,
            self.mac_address,
        )


class HSBLiteFlasher2505(HSBLiteFlasherBase):
    """Flasher for HSB_Lite version 0x2505."""

    VERSION = 0x2505

    def flash(self, clnx_path: str, cpnx_path: str) -> bool:
        return _flash_strategies.hsb_lite_flash_2504(
            self.ip_address,
            clnx_path,
            cpnx_path,
            self.VERSION,
            self.FPGA_UUID,
            self.mac_address,
        )


class HSBLiteFlasher2504(HSBLiteFlasherBase):
    """Flasher for HSB_Lite version 0x2504."""

    VERSION = 0x2504

    def flash(self, clnx_path: str, cpnx_path: str) -> bool:
        return _flash_strategies.hsb_lite_flash_2504(
            self.ip_address,
            clnx_path,
            cpnx_path,
            self.VERSION,
            self.FPGA_UUID,
            self.mac_address,
        )


# =============================================================================
# UDP Flasher Implementations
# =============================================================================


class HSBLiteFlasher2502(HSBLiteFlasherBase):
    """Flasher for HSB_Lite version 0x2502."""

    VERSION = 0x2502

    def flash(self, clnx_path: str, cpnx_path: str) -> bool:
        return _flash_strategies.hsb_lite_flash_2407(
            self.ip_address, clnx_path, cpnx_path
        )


class HSBLiteFlasher2412(HSBLiteFlasherBase):
    """Flasher for HSB_Lite version 0x2412."""

    VERSION = 0x2412

    def flash(self, clnx_path: str, cpnx_path: str) -> bool:
        return _flash_strategies.hsb_lite_flash_2407(
            self.ip_address, clnx_path, cpnx_path
        )


class HSBLiteFlasher2410(HSBLiteFlasherBase):
    """Flasher for HSB_Lite version 0x2410."""

    VERSION = 0x2410

    def flash(self, clnx_path: str, cpnx_path: str) -> bool:
        return _flash_strategies.hsb_lite_flash_2407(
            self.ip_address, clnx_path, cpnx_path
        )


class HSBLiteFlasher2407(HSBLiteFlasherBase):
    """Flasher for HSB_Lite version 0x2407."""

    VERSION = 0x2407

    def flash(self, clnx_path: str, cpnx_path: str) -> bool:
        return _flash_strategies.hsb_lite_flash_2407(
            self.ip_address, clnx_path, cpnx_path
        )


# =============================================================================
# Registry of all flashers in this module
# =============================================================================

FLASH_STRATEGIES = [
    # Modern
    HSBLiteFlasher2603,
    HSBLiteFlasher2601,
    HSBLiteFlasher2512,
    HSBLiteFlasher2511,
    HSBLiteFlasher2510,
    HSBLiteFlasher2508,
    HSBLiteFlasher2507,
    # Legacy
    HSBLiteFlasher2506,
    HSBLiteFlasher2505,
    HSBLiteFlasher2504,
    # UDP
    HSBLiteFlasher2502,
    HSBLiteFlasher2412,
    HSBLiteFlasher2410,
    HSBLiteFlasher2407,
]


# =============================================================================
# Arbiter - called by C++ to get the right flasher
# =============================================================================


def get_flasher(
    fpga_uuid: str, version: int, ip_address: str, mac_address: str
) -> Optional[HSBLiteFlasherBase]:
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
        print(f"[hsb_lite] No flasher for uuid={fpga_uuid} version=0x{version:04x}")
        return False

    print(f"[hsb_lite] Using {flasher.__class__.__name__}")
    print(f"  FPGA UUID: {fpga_uuid}")
    print(f"  Version: 0x{version:04x}")
    print(f"  MAC: {mac_address}")
    print(f"  IP: {ip_address}")
    print(f"  CLNX: {clnx_path}")
    print(f"  CPNX: {cpnx_path}")

    try:
        success = flasher.flash(clnx_path, cpnx_path)
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as e:
        print(f"[hsb_lite] ERROR: {e}")
        return False
    else:
        if success:
            print("[hsb_lite] Flash completed successfully")
        else:
            print("[hsb_lite] Flash failed")
        return success
