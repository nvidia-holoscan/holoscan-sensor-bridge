# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# See README.md for detailed information.

import logging
import socket

import cupy as cp
import holoscan
from cuda import cuda


class BaseReceiverOperator(holoscan.core.Operator):
    def __init__(
        self,
        *args,
        hololink_channel=None,
        device=None,
        frame_memory=None,
        frame_size=None,
        frame_context=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._hololink_channel = hololink_channel
        self._device = device
        self._frame_size = frame_size
        self._frame_memory = frame_memory
        self._frame_context = frame_context
        self._ok = False
        self._metadata = None
        # We'll allocate this for you if you like.
        if self._frame_memory is None:
            self._frame_memory = self._allocate(self._frame_size)

    def setup(self, spec):
        logging.info("setup")
        spec.output("output")

    def start(self):
        unowned_memory = cp.cuda.UnownedMemory(
            self._frame_memory, self._frame_size, self
        )
        self._cp_frame = cp.ndarray(
            (self._frame_size,),
            dtype=cp.uint8,
            memptr=cp.cuda.MemoryPointer(unowned_memory, 0),
        )
        self._cp_frame[:] = 0xFF
        logging.info(f"frame_size={self._frame_size} frame={self._frame_memory}")
        #
        self._data_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._start_receiver()
        local_ip, local_port = self._local_ip_and_port()
        logging.info(f"{local_ip=} {local_port=}")
        self._hololink_channel.configure(
            self._frame_memory, self._frame_size, local_port
        )
        self._device.start()

    def _start_receiver(self):
        raise NotImplementedError()

    def _local_ip_and_port(self):
        local_ip, local_port = self._data_socket.getsockname()
        return local_ip, local_port

    def stop(self):
        self._device.stop()
        self._stop()
        self._frame_size = 0
        del self._cp_frame
        self._cp_frame = None

    def _stop(self):
        raise NotImplementedError()

    def compute(self, op_input, op_output, context):
        timeout_ms = 1000
        # metadata is a dict or None
        self._metadata = self._get_next_frame(timeout_ms)
        if not self._metadata:
            if self._ok:
                self._ok = False
                logging.error("Ingress frame timeout; ignoring.")
        else:
            self._ok = True
        op_output.emit({"": self._cp_frame}, "output")

    def _get_next_frame(self, timeout_ms):
        """Returns metadata: dict or None"""
        raise NotImplementedError()

    def metadata(self):
        return self._metadata

    def _allocate(self, size, flags=0):
        (cu_result,) = cuda.cuInit(0)
        assert cu_result == cuda.CUresult.CUDA_SUCCESS
        (cu_result,) = cuda.cuCtxSetCurrent(self._frame_context)
        assert cu_result == cuda.CUresult.CUDA_SUCCESS
        cu_result, cu_device = cuda.cuCtxGetDevice()
        assert cu_result == cuda.CUresult.CUDA_SUCCESS
        cu_result, integrated = cuda.cuDeviceGetAttribute(
            cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_INTEGRATED, cu_device
        )
        assert cu_result == cuda.CUresult.CUDA_SUCCESS
        logging.trace(f"{integrated=}")
        if integrated == 0:
            # We're a discrete GPU device; so allocate using cuMemAlloc/cuMemFree
            cu_result, device_deviceptr = cuda.cuMemAlloc(size)
            assert cu_result == cuda.CUresult.CUDA_SUCCESS
            return int(device_deviceptr)
        # We're an integrated device (e.g. Tegra) so we must allocate
        # using cuMemHostAlloc/cuMemFreeHost
        cu_result, host_deviceptr = cuda.cuMemHostAlloc(size, flags)
        assert cu_result == cuda.CUresult.CUDA_SUCCESS
        cu_result, device_deviceptr = cuda.cuMemHostGetDevicePointer(host_deviceptr, 0)
        assert cu_result == cuda.CUresult.CUDA_SUCCESS
        return int(device_deviceptr)
