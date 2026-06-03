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

import os
import resource
import socket
import threading

import cuda.bindings.driver as cuda
import cupy as cp

import hololink as hololink_module


class LinuxControllerReceiver:
    def __init__(self, cu_context, cu_device, frame_size):
        self._cu_context = cu_context
        self._frame_size = frame_size
        self._pages = 1
        metadata_address = hololink_module.round_up(
            self._frame_size, hololink_module.PAGE_SIZE
        )
        self._page_size = metadata_address + hololink_module.METADATA_SIZE
        allocation_size = hololink_module.round_up(
            self._page_size * self._pages, resource.getpagesize()
        )
        self._receiver_memory_descriptor = hololink_module.ReceiverMemoryDescriptor(
            cu_context, allocation_size
        )
        self._receiver_affinity = None
        # By default, run us on the third core in the system;
        # run with HOLOLINK_AFFINITY=<n> to use a different core or
        # set HOLOLINK_AFFINITY="" to avoid affinity configuration.
        affinity = os.getenv("HOLOLINK_AFFINITY", "2")
        # The len(affinity) supports this command
        #   HOLOLINK_AFFINITY= python3 ...
        # to avoid affinity settings.
        if (affinity is not None) and (len(affinity) > 0):
            self._receiver_affinity = {int(affinity)}
        self._device = None
        self._running = False
        self._receiver = None

    def found(self, metadata, hololink_channel, device):
        cuda.cuCtxSetCurrent(self._cu_context)
        self._metadata = metadata
        self._data_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        hololink_channel.configure_socket(self._data_socket.fileno())
        self._receiver = hololink_module.operators.LinuxReceiver(
            self._receiver_memory_descriptor.get(),
            self._frame_size,
            self._frame_size,
            1,
            self._data_socket.fileno(),
            self._receiver_memory_descriptor.get(),
        )
        self._receiver.set_frame_ready(self._ready)
        self._receiver_thread = threading.Thread(
            daemon=True, name="linux_receiver", target=self._run
        )
        self._receiver_thread.start()
        hololink_channel.authenticate(
            self._receiver.get_qp_number(),
            self._receiver.get_rkey(),
        )
        distal_memory_address_start = 0
        _, local_port = self._data_socket.getsockname()
        hololink_channel.configure_roce(
            distal_memory_address_start,
            self._frame_size,
            self._page_size,
            self._pages,
            local_port,
        )
        self._device = device

    def lost(self):
        if self._running:
            self._receiver.close()
            self._receiver = None
            self._running = False

    def start(self, operator):
        self._operator = operator

    def _run(self):
        cuda.cuCtxSetCurrent(self._cu_context)
        if self._receiver_affinity:
            os.sched_setaffinity(0, self._receiver_affinity)
        self._receiver.run()

    def _ready(self, receiver):
        self._operator.frame_ready()

    def stop(self):
        if self._running:
            self._device.stop()
            self._device = None
            if self._receiver:
                self._receiver.close()
                self._receiver = None
            self._running = False

    def get_next_frame(self):
        if not self._running:
            return None
        timeout_ms = 1000
        cuda_stream = 0
        ok, receiver_metadata = self._receiver.get_next_frame(timeout_ms, cuda_stream)
        if not ok:
            return None
        #
        frame_memory = receiver_metadata.frame_memory
        unowned_memory = cp.cuda.UnownedMemory(frame_memory, self._page_size, self)
        cp_frame = cp.ndarray(
            (self._frame_size,),
            dtype=cp.uint8,
            memptr=cp.cuda.MemoryPointer(unowned_memory, 0),
        )
        return cp_frame

    def run(self):
        self._device.start()
        self._running = True
