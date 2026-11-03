# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PVA CRC native extension.

Houses the compiled `pva_crc` nanobind extension (built under cpp/, see cpp/CMakeLists.txt)
wrapping hololink::PvaCrc (src/hololink/operators/pva_crc/pva_crc.hpp) -- hardware-accelerated
CRC32 computation on NVIDIA PVA (Programmable Vision Accelerator). Only built when
HOLOLINK_BUILD_PVA is on and /opt/nvidia/pva-sdk-2.9 is present (Tegra/aarch64 only).

No Python operator classes live here: ComputePvaCrcOp/CheckPvaCrcOp (the Holoscan operators
driving this extension) are defined directly in the example apps that use them, matching every
other custom per-app operator in this porting effort -- see e.g.
examples/linux_imx274_frame_validation.py's FrameValidationOp. `import pva_crc` itself is
deferred inside each app's own operator `start()` (not here, not at module level) so the rest of
the `hololink` package stays importable even where this extension didn't build.
"""
