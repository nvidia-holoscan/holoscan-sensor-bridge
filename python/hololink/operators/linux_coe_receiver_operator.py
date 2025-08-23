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

# See README.md for detailed information.

import logging
import os
import socket
import threading

import cupy as cp
import holoscan
from cuda import cuda

import hololink as hololink_module


class LinuxCoeReceiverOp(hololink_module.operators.BaseReceiverOp):
    def __init__(
        self,
        *args,
        receiver_affinity=None,
        coe_interface=None,
        pixel_width=None,
        coe_channel=0,
        rename_metadata=lambda original_name: original_name,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._receiver_affinity = receiver_affinity
        self._coe_interface = coe_interface
        self._pixel_width = pixel_width
        self._coe_channel = coe_channel
        self._vlan_enabled = False
        if self._receiver_affinity is None:
            # By default, run us on the third core in the system;
            # run with HOLOLINK_AFFINITY=<n> to use a different core or
            # set HOLOLINK_AFFINITY="" to avoid affinity configuration.
            affinity = os.getenv("HOLOLINK_AFFINITY", "2")
            # The len(affinity) supports this command
            #   HOLOLINK_AFFINITY= python3 ...
            # to avoid affinity settings.
            if (affinity is not None) and (len(affinity) > 0):
                self._receiver_affinity = {int(affinity)}
        self._frame_packets_received_metadata = rename_metadata(
            "frame_packets_received"
        )
        self._frame_bytes_received_metadata = rename_metadata("frame_bytes_received")
        self._received_frame_number_metadata = rename_metadata("received_frame_number")
        self._frame_number_metadata = rename_metadata("frame_number")
        self._frame_start_s_metadata = rename_metadata("frame_start_s")
        self._frame_start_ns_metadata = rename_metadata("frame_start_ns")
        self._frame_end_s_metadata = rename_metadata("frame_end_s")
        self._frame_end_ns_metadata = rename_metadata("frame_end_ns")
        self._received_s_metadata = rename_metadata("received_s")
        self._received_ns_metadata = rename_metadata("received_ns")
        self._timestamp_s_metadata = rename_metadata("timestamp_s")
        self._timestamp_ns_metadata = rename_metadata("timestamp_ns")
        self._metadata_s_metadata = rename_metadata("metadata_s")
        self._metadata_ns_metadata = rename_metadata("metadata_ns")
        self._crc_metadata = rename_metadata("crc")
        self._psn_metadata = rename_metadata("psn")
        self._bytes_written_metadata = rename_metadata("bytes_written")

    def start(self):
        unowned_memory = cp.cuda.UnownedMemory(
            self._frame_memory, self._allocation_size, self
        )
        self._cp_frame = cp.ndarray(
            (self._frame_size,),
            dtype=cp.uint8,
            memptr=cp.cuda.MemoryPointer(unowned_memory, 0),
        )
        self._cp_frame[:] = 0xFF
        logging.info(f"frame_size={self._frame_size} frame={self._frame_memory}")
        self._start_receiver()
        self._hololink_channel.configure_coe(
            self._coe_channel, self._frame_size, self._pixel_width, self._vlan_enabled
        )
        self._frame_ready_condition.event_state = (
            holoscan.conditions.AsynchronousEventState.EVENT_WAITING
        )
        self._device.start()

    def _start_receiver(self):
        # We receive all packets on the interface.
        ETH_P_AVTP = 0x22F0
        self._data_socket = socket.socket(
            socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(ETH_P_AVTP)
        )
        self._data_socket.bind((self._coe_interface, 0))
        self._receiver = hololink_module.operators.LinuxCoeReceiver(
            self._frame_memory,
            self._allocation_size,
            self._data_socket.fileno(),
            self._coe_channel,
        )

        def _ready(receiver):
            self.frame_ready()

        self._receiver.set_frame_ready(_ready)
        self._receiver_thread = threading.Thread(
            daemon=True, name=self.name, target=self._run
        )
        self._receiver_thread.start()

    def _run(self):
        cuda.cuCtxSetCurrent(self._frame_context)
        if self._receiver_affinity:
            os.sched_setaffinity(0, self._receiver_affinity)
        self._receiver.run()

    def _stop(self):
        self._data_socket.close()
        self._receiver.close()
        self._receiver_thread.join()

    def _get_next_frame(self, timeout_ms):
        ok, receiver_metadata = self._receiver.get_next_frame(timeout_ms)
        if not ok:
            return None
        application_metadata = {
            self._frame_packets_received_metadata: receiver_metadata.frame_packets_received,
            self._frame_bytes_received_metadata: receiver_metadata.frame_bytes_received,
            self._received_frame_number_metadata: receiver_metadata.received_frame_number,
            self._frame_number_metadata: receiver_metadata.frame_number,
            self._frame_start_s_metadata: receiver_metadata.frame_start_s,
            self._frame_start_ns_metadata: receiver_metadata.frame_start_ns,
            self._frame_end_s_metadata: receiver_metadata.frame_end_s,
            self._frame_end_ns_metadata: receiver_metadata.frame_end_ns,
            self._received_s_metadata: receiver_metadata.received_s,
            self._received_ns_metadata: receiver_metadata.received_ns,
            self._timestamp_s_metadata: receiver_metadata.timestamp_s,
            self._timestamp_ns_metadata: receiver_metadata.timestamp_ns,
            self._metadata_s_metadata: receiver_metadata.metadata_s,
            self._metadata_ns_metadata: receiver_metadata.metadata_ns,
            self._crc_metadata: receiver_metadata.crc,
            self._psn_metadata: receiver_metadata.psn,
            self._bytes_written_metadata: receiver_metadata.bytes_written,
        }
        return application_metadata
