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
import threading

from cuda import cuda

import hololink as hololink_module

MS_PER_SEC = 1000.0
US_PER_SEC = 1000.0 * MS_PER_SEC
NS_PER_SEC = 1000.0 * US_PER_SEC
SEC_PER_NS = 1.0 / NS_PER_SEC


class RoceReceiverOperator(hololink_module.operators.BaseReceiverOperator):
    def __init__(self, *args, ibv_name=None, ibv_port=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._ibv_name = ibv_name
        self._ibv_port = ibv_port

    def _start_receiver(self):
        peer_ip = self._hololink_channel.peer_ip()
        logging.info(f"{self._ibv_name=} {self._ibv_port=} {peer_ip=}")
        self._receiver = hololink_module.operators.RoceReceiver(
            self._ibv_name,
            self._ibv_port,
            self._frame_memory,
            self._frame_size,
            peer_ip,
        )
        self._receiver.start()
        self._hololink_channel.authenticate(
            self._receiver.get_qp_number(), self._receiver.get_rkey()
        )
        self._data_socket.bind(
            ("", 0)
        )  # we don't actually receive anything here because CX7 hides it.
        self._receiver_thread = threading.Thread(
            daemon=True, name="receiver_thread", target=self._run
        )
        self._receiver_thread.start()

    def _run(self):
        cuda.cuCtxSetCurrent(self._frame_context)
        self._receiver.blocking_monitor()

    def _stop(self):
        self._data_socket.close()
        self._receiver.close()
        self._receiver_thread.join()

    def _get_next_frame(self, timeout_ms):
        ok, metadata = self._receiver.get_next_frame(timeout_ms)
        if not ok:
            return None
        frame_end = metadata.frame_end_s + (metadata.frame_end_ns * SEC_PER_NS)
        metadata = {
            "frame_number": metadata.frame_number,
            "rx_write_requests": metadata.rx_write_requests,
            "frame_end": frame_end,
        }
        return metadata

    def _local_ip_and_port(self):
        local_ip, local_port = self._data_socket.getsockname()
        return local_ip, 4791
