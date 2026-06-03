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

"""
PVA CRC Operators

Async pattern for PVA hardware-accelerated CRC computation:
- ComputePvaCrcOp: Launches non-blocking PVA computation
- CheckPvaCrcOp: Retrieves results (may block if not ready)

Follows the same pattern as ComputeCrcOp/CheckCrcOp for nvCOMP CRC.
"""

import logging
import os

import cupy as cp
import holoscan
import nvtx

# PVA CRC timing constants (in microseconds)
PVA_EXPECTED_TIME_US = 1_000  # ~1ms - Typical PVA execution time
PVA_CHECK_TIMEOUT_US = 2_000  # 2ms - Timeout for checking PVA results (provides margin for timing variance)


class ComputePvaCrcOp(holoscan.core.Operator):
    """
    Launch PVA CRC computation (non-blocking, 1D API)

    This operator submits the frame to PVA hardware and returns immediately,
    allowing the pipeline to continue processing while PVA computes in the background.

    Args:
        data_size: Size of buffer in bytes (must be multiple of 4, 32KB-12MB)
        remaining_size: Bytes remaining after this chunk (for parallel computation), default 0
        is_first_chunk: True to apply preconditioning (for first chunk), default True
        use_dual_vpu: Enable dual VPU parallel processing (auto-splits across VPU0+VPU1), default True
        start_byte: Offset to skip CSI headers (0 for full frame)

    Raises:
        ValueError: If data_size is not provided or doesn't meet PVA constraints
    """

    def __init__(
        self,
        *args,
        data_size=None,
        remaining_size=0,
        is_first_chunk=True,
        use_dual_vpu=True,
        start_byte=0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Validate data_size is provided
        if data_size is None:
            raise ValueError(
                "data_size is required and must be specified explicitly. "
                "Set data_size to your frame size in bytes (e.g., from CsiToBayerOp.get_csi_length()). "
                "PVA constraints: multiple of 4, min 32KB, max 12MB."
            )

        self._data_size = data_size
        self._remaining_size = remaining_size
        self._is_first_chunk = is_first_chunk
        self._use_dual_vpu = use_dual_vpu
        self._start_byte = start_byte  # Offset to skip CSI headers (0 for full frame)
        self._pva_crc = None
        self._initialized = False
        self._frame_count = 0

    def setup(self, spec):
        spec.input("input")
        spec.output("output")

    def start(self):
        """Initialize PVA CRC library"""
        try:
            # Import the pybind11 module
            import sys

            # Get the directory where this operator file is located
            operator_dir = os.path.dirname(os.path.abspath(__file__))
            module_path = os.path.join(operator_dir, "cpp")

            # Add installed module path if not already present
            if module_path not in sys.path:
                sys.path.insert(0, module_path)

            # Set PVA_CRC_DEV_BUILD=/path/to/build to override the default dev path
            # Default to project build directory (accessible on host via volume mount)
            dev_build_path = os.environ.get(
                "PVA_CRC_DEV_BUILD", os.path.join(os.getcwd(), "build/pva_crc/cpp")
            )
            if os.path.exists(dev_build_path) and dev_build_path not in sys.path:
                sys.path.insert(0, dev_build_path)
                if "PVA_CRC_DEV_BUILD" not in os.environ:
                    logging.info(f"PVA CRC: Loading from build path {dev_build_path}")
                else:
                    logging.info(
                        f"PVA CRC: Using development build from {dev_build_path}"
                    )

            import pva_crc

            # Create PVA CRC instance
            self._pva_crc = pva_crc.PvaCrc()

            # Initialize with 1D buffer size (NEW API with dual VPU support)
            self._pva_crc.initialize(
                data_size=self._data_size,
                remaining_size=self._remaining_size,
                is_first_chunk=self._is_first_chunk,
                use_dual_vpu=self._use_dual_vpu,
            )

            logging.info(
                f"ComputePvaCrcOp initialized: data_size={self._data_size:,} bytes, "
                f"remaining_size={self._remaining_size}, is_first_chunk={self._is_first_chunk}, "
                f"use_dual_vpu={self._use_dual_vpu}, start_byte={self._start_byte}"
            )

            self._initialized = True

        except ImportError as e:
            logging.error(f"FATAL: Failed to import pva_crc module: {e}")
            logging.error("Make sure the module is built and in the Python path")
            raise
        except Exception as e:
            logging.error(f"FATAL: Failed to initialize ComputePvaCrcOp: {e}")
            raise

    def compute(self, op_input, op_output, context):
        """Launch PVA computation and pass through immediately (non-blocking)"""
        # Validate initialization before processing
        if not self._initialized:
            raise RuntimeError(
                "ComputePvaCrcOp not initialized. "
                "This indicates start() failed or wasn't called."
            )

        # Receive input
        in_message = op_input.receive("input")
        if not in_message:
            return

        # Get input tensor (GPU memory)
        input_data = in_message.get("")
        if input_data is None:
            raise ValueError(
                f"Input message does not contain expected tensor data. "
                f"Frame {self._frame_count} has no data payload."
            )
        input_tensor = cp.asarray(input_data)
        if input_tensor.size == 0:
            raise ValueError(
                f"Input tensor is empty. Frame {self._frame_count} has zero size."
            )

        # Validate start_byte and data_size against tensor size
        tensor_size = input_tensor.nbytes
        if self._start_byte >= tensor_size:
            raise ValueError(
                f"start_byte ({self._start_byte}) exceeds tensor size ({tensor_size} bytes). "
                f"Frame {self._frame_count} has insufficient data."
            )
        if self._start_byte + self._data_size > tensor_size:
            raise ValueError(
                f"start_byte ({self._start_byte}) + data_size ({self._data_size}) = "
                f"{self._start_byte + self._data_size} exceeds tensor size ({tensor_size} bytes). "
                f"Frame {self._frame_count} has insufficient data."
            )

        # Debug: Log CRC from metadata (first 10 frames, then every 100th frame)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            crc_value = self.metadata.get("crc")
            if self._frame_count < 10 or self._frame_count % 100 == 0:
                if crc_value is not None:
                    logging.debug(
                        f"ComputePvaCrcOp [frame {self._frame_count}]: CRC from metadata = {crc_value} (0x{crc_value:x})"
                    )
                else:
                    logging.debug(
                        f"ComputePvaCrcOp [frame {self._frame_count}]: CRC from metadata = NOT FOUND"
                    )
                logging.debug(
                    f"ComputePvaCrcOp [frame {self._frame_count}]: All metadata keys = {list(self.metadata.keys())}"
                )

        # Get GPU pointer with offset
        input_ptr = input_tensor.data.ptr + self._start_byte

        # NVTX marker: Launch PVA computation
        nvtx.push_range("PVA_CRC_Launch", color="green")
        nvtx.mark(f"PVA_CRC_Launch_frame={self._frame_count}")

        try:
            # Launch async computation (non-blocking, returns fence handle)
            fence_handle = self._pva_crc.launch_compute(input_ptr)

            # Store fence in metadata for CheckPvaCrcOp
            self.metadata["pva_fence"] = fence_handle

        except RuntimeError as e:
            logging.error(f"PVA CRC launch failed: {e}")
            raise
        finally:
            # Always pop NVTX range, regardless of exception type
            nvtx.pop_range()

        self._frame_count += 1

        # Pass through the tensor immediately (PVA computes in background)
        op_output.emit({"": input_tensor}, "output")

    def stop(self):
        """Cleanup PVA CRC resources"""
        self._pva_crc = None

    def get_pva_crc_handle(self):
        """Get PVA CRC handle for CheckPvaCrcOp"""
        if not self._initialized:
            raise RuntimeError(
                "Cannot get PVA CRC handle: ComputePvaCrcOp not initialized"
            )
        return self._pva_crc


class CheckPvaCrcOp(holoscan.core.Operator):
    """
    Check PVA CRC computation results (may block if not ready)

    This operator retrieves the PVA results and stores them in metadata.
    It should be placed later in the pipeline after other operators that can
    run in parallel with PVA computation.

    Args:
        compute_pva_crc_op: The ComputePvaCrcOp instance to retrieve results from
        computed_crc_metadata_name: Name to use for computed CRC in metadata (default: "pva_crc")

    Note: frame_height parameter is deprecated in 1D API (single CRC output)
    """

    def __init__(
        self,
        *args,
        compute_pva_crc_op,
        frame_height=None,  # Deprecated but kept for compatibility
        computed_crc_metadata_name="pva_crc",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._compute_op = compute_pva_crc_op
        self._computed_crc_metadata_name = computed_crc_metadata_name
        self._frame_count = 0
        # Pre-allocate GPU memory buffer (memory pool pattern)
        self._output_gpu = None

        # Warn if frame_height was provided (deprecated in 1D API)
        if frame_height is not None:
            logging.warning(
                "frame_height parameter is deprecated in 1D API and will be ignored"
            )

    def setup(self, spec):
        spec.input("input")
        spec.output("output")

    def start(self):
        """Pre-allocate GPU memory buffer once (avoids per-frame allocation)"""
        # Allocate output for dual VPU (2x uint32 - library needs space for both VPU outputs)
        self._output_gpu = cp.zeros(2, dtype=cp.uint32)
        logging.info(
            "CheckPvaCrcOp initialized: Pre-allocated 2 CRCs for dual VPU (8 bytes GPU memory)"
        )

    def compute(self, op_input, op_output, context):
        """Retrieve PVA results (blocks if not ready, but usually ready by now)"""
        # Receive input
        in_message = op_input.receive("input")
        if not in_message:
            return

        # Get input tensor (GPU memory)
        input_data = in_message.get("")
        if input_data is None:
            raise ValueError(
                f"Input message does not contain expected tensor data. "
                f"Frame {self._frame_count} has no data payload."
            )
        input_tensor = cp.asarray(input_data)
        if input_tensor.size == 0:
            raise ValueError(
                f"Input tensor is empty. Frame {self._frame_count} has zero size."
            )

        fence_handle = self.metadata.get("pva_fence")

        if not fence_handle:
            raise RuntimeError(
                "No PVA fence found in metadata. Pipeline configuration error: "
                "CheckPvaCrcOp must receive frames from ComputePvaCrcOp, and "
                "metadata must be properly passed between operators. "
                f"Available metadata keys: {list(self.metadata.keys())}"
            )

        # Use pre-allocated GPU buffer (no per-frame allocation!)
        output_ptr = self._output_gpu.data.ptr

        # Initialize to None to ensure safe cleanup in finally block
        pva_crc = None

        # Use try-finally to guarantee fence cleanup and NVTX pop in all code paths
        try:
            # NVTX marker: Start waiting for PVA results
            nvtx.push_range("PVA_CRC_Wait", color="orange")
            nvtx.mark(f"PVA_CRC_Wait_frame={self._frame_count}")

            # Get PVA CRC handle (inside try to ensure cleanup if this fails)
            pva_crc = self._compute_op.get_pva_crc_handle()

            # Wait for PVA computation completion with small timeout
            # PVA typically completes in ~0.4ms (dual VPU) or ~0.8ms (single VPU)
            # 2ms timeout provides margin for timing variance while still catching real issues
            result = pva_crc.check_results(
                fence_handle, output_ptr, timeout_us=PVA_CHECK_TIMEOUT_US
            )

            if result < 0:
                logging.error(
                    f"PVA CRC check failed with error code {result} on frame {self._frame_count}. "
                    "This indicates PVA hardware failure or invalid operation."
                )
                raise RuntimeError(
                    f"PVA CRC check failed with error code {result}. "
                    "Check PVA hardware status and ensure valid parameters."
                )
            elif result == 1:
                timeout_ms = PVA_CHECK_TIMEOUT_US / 1000
                logging.error(
                    f"PVA CRC timeout after {timeout_ms}ms on frame {self._frame_count}. "
                    "This indicates the async pipeline is not properly paced (frames arriving faster than PVA can compute)."
                )
                raise RuntimeError(
                    f"PVA CRC computation timeout after {timeout_ms}ms. "
                    "Check pipeline scheduling or increase frame interval."
                )

            # Success path: result == 0
            # Get frame CRC (single value in 1D API)
            # PVA returns standard CRC32, but camera uses JAMCRC (bit-inverted CRC32)
            pva_crc32 = int(self._output_gpu[0].get())
            frame_crc = ~pva_crc32 & 0xFFFFFFFF  # Invert to get JAMCRC

            # Add PVA CRC to metadata (for RecordMetadataOp or other operators to use)
            self.metadata[self._computed_crc_metadata_name] = frame_crc
            self._frame_count += 1

            # Pass through the tensor
            op_output.emit({"": input_tensor}, "output")

        finally:
            nvtx.pop_range()  # Always pop exactly once
            # Free fence only if both pva_crc handle and fence_handle are valid
            if pva_crc is not None and fence_handle is not None:
                try:
                    pva_crc.free_fence(fence_handle)
                except Exception as e:
                    # Log warning but don't raise - fence cleanup failure shouldn't crash the pipeline
                    logging.warning(
                        f"Error freeing PVA fence on frame {self._frame_count}: {e}"
                    )

    def stop(self):
        """Cleanup GPU memory and clear references"""
        self._compute_op = None
        self._output_gpu = None
