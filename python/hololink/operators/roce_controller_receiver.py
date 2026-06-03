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
import os
import resource
import socket
import threading

import cuda.bindings.driver as cuda
import cupy as cp

import hololink as hololink_module


class RoceControllerReceiver:
    def __init__(self, cu_context, cu_device, frame_size, ibv_name, ibv_port):
        self._cu_context = cu_context
        self._ibv_name = ibv_name
        self._ibv_port = ibv_port
        self._frame_size = frame_size
        self._pages = 2
        #
        self._metadata_address = hololink_module.round_up(
            frame_size, hololink_module.PAGE_SIZE
        )
        self._page_size = self._metadata_address + hololink_module.METADATA_SIZE
        self._allocation_size = hololink_module.round_up(
            self._page_size * self._pages, resource.getpagesize()
        )
        self._receiver_memory_descriptor = hololink_module.ReceiverMemoryDescriptor(
            self._cu_context, self._allocation_size
        )
        #
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
        logging.info(f"{id(self)=} found")
        cuda.cuCtxSetCurrent(self._cu_context)
        self._metadata = metadata
        self._data_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        hololink_channel.configure_socket(self._data_socket.fileno())
        peer_ip = hololink_channel.peer_ip()
        logging.info(f"{self._ibv_name=} {self._ibv_port=} {peer_ip=}")
        self._receiver = hololink_module.operators.RoceReceiver(
            self._ibv_name,
            self._ibv_port,
            self._receiver_memory_descriptor.get(),
            self._allocation_size,
            self._frame_size,
            self._page_size,
            self._pages,
            self._metadata_address,
            peer_ip,
        )
        self._receiver.set_frame_ready(self._ready)
        # Set up ibverbs
        if not self._receiver.start():
            raise Exception("Failed to start RoceReceiver.")
        self._receiver_thread = threading.Thread(
            daemon=True, name="roce_receiver", target=self._run
        )
        self._receiver_thread.start()
        hololink_channel.authenticate(
            self._receiver.get_qp_number(),
            self._receiver.get_rkey(),
        )
        # This is the well-defined UDP port for ROCE traffic.
        local_port = 4791
        hololink_channel.configure_roce(
            self._receiver.external_frame_memory(),
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
        self._receiver.blocking_monitor()

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
        timeout_ms = 1000
        ok, receiver_metadata = self._receiver.get_next_frame(timeout_ms)
        if not ok:
            return None
        #
        unowned_memory = cp.cuda.UnownedMemory(
            receiver_metadata.frame_memory, self._page_size, self
        )
        cp_frame = cp.ndarray(
            (self._frame_size,),
            dtype=cp.uint8,
            memptr=cp.cuda.MemoryPointer(unowned_memory, 0),
        )
        return cp_frame

    def run(self):
        logging.info(f"{id(self)=} run")
        self._device.start()
        self._running = True
