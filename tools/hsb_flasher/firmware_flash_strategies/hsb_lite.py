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

This module routes a device firmware version to a flash strategy identified by
its MIN_VERSION (the lowest version that strategy covers). A device at version V
is dispatched to the strategy with the largest MIN_VERSION <= V.
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

    FPGA_UUID = "889b7ce3-65a5-4247-8b05-4ff1904c3359"

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


# =============================================================================
# Flasher Implementations
# =============================================================================


class HSBLiteFlasherUDP(HSBLiteFlasherBase):
    """Raw UDP"""

    MIN_VERSION = 0x2407

    def flash(self, clnx_path: str, cpnx_path: str) -> bool:
        return _flash_strategies.hsb_lite_flash_2407(
            self.ip_address, clnx_path, cpnx_path
        )


class HSBLiteFlasherLegacy(HSBLiteFlasherBase):
    """Legacy connection based"""

    MIN_VERSION = 0x2504

    def flash(self, clnx_path: str, cpnx_path: str) -> bool:
        return _flash_strategies.hsb_lite_flash_2504(
            self.ip_address,
            clnx_path,
            cpnx_path,
            self.version,
            self.FPGA_UUID,
            self.mac_address,
        )


class HSBLiteFlasherModern(HSBLiteFlasherBase):
    """Current connection based flash"""

    MIN_VERSION = 0x2507

    def flash(self, clnx_path: str, cpnx_path: str) -> bool:
        return _flash_strategies.hsb_lite_flash_2507(
            self.ip_address, clnx_path, cpnx_path
        )


# =============================================================================
# Registry of all flashers in this module
# =============================================================================

# Sorted descending by MIN_VERSION
FLASH_STRATEGIES = sorted(
    [HSBLiteFlasherModern, HSBLiteFlasherLegacy, HSBLiteFlasherUDP],
    key=lambda c: c.MIN_VERSION,
    reverse=True,
)


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
            return flasher_cls(
                ip_address=ip_address, mac_address=mac_address, version=version
            )
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
