#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved.
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

import argparse
import logging
import time

import holoscan

import hololink as hololink_module

SLEEP_TIME = 1


def enable_i2s(hololink):
    hololink.write_uint32(0x70000014, 0x0000000F)
    time.sleep(SLEEP_TIME)
    hololink.write_uint32(0x80000010, 0x00000004)
    time.sleep(SLEEP_TIME)
    hololink.write_uint32(0x8000000C, 0x00000002)
    time.sleep(SLEEP_TIME)
    hololink.write_uint32(0x80000008, 0x00000003)
    time.sleep(SLEEP_TIME)
    hololink.write_uint32(0x80000004, 0x00000002)
    time.sleep(SLEEP_TIME)
    hololink.write_uint32(0x80000000, 0x00000003)
    logging.info("Finished setting I2S")


def set_tx_af_ae(hololink, af=0x00001600, ae=0x00001000):
    # set sensor tx out
    hololink.write_uint32(0x01200004, af)
    time.sleep(SLEEP_TIME)
    hololink.write_uint32(0x01200008, ae)
    time.sleep(SLEEP_TIME)
    hololink.write_uint32(0x01200000, 0x00000004)
    logging.info("Finished setting TX almost full & empty")


def set_tx_pause(hololink):
    # set sensor tx 0 out
    hololink.write_uint32(0x01200000, 0x00000003)
    time.sleep(SLEEP_TIME)
    # set tx 0 host pause mapping
    hololink.write_uint32(0x0120000C, 0x00000001)
    time.sleep(SLEEP_TIME)
    logging.info("Finished setting TX and pause")


def en_i2s_tx(hololink):
    enable_i2s(hololink)
    set_tx_af_ae(hololink, 0x00000800, 0x00000400)
    set_tx_pause(hololink)


class AudioRoceTransmitApp(holoscan.core.Application):
    def __init__(
        self,
        wav_file,
        chunk_size,
        hololink_ip,
        ibv_name,
        ibv_port,
        ibv_qp,
        queue_size,
    ):
        super().__init__()
        self.wav_file = wav_file
        self.chunk_size = chunk_size
        self.hololink_ip = hololink_ip
        self.ibv_name = ibv_name
        self.ibv_port = ibv_port
        self.ibv_qp = ibv_qp
        self.queue_size = queue_size

    def compose(self):
        # Create allocator for GPU memory
        allocator = holoscan.resources.UnboundedAllocator(
            self,
            name="allocator",
        )

        # Create audio packetizer operator
        audio_packetizer = hololink_module.operators.AudioPacketizerOp(
            self,
            wav_file=self.wav_file,
            chunk_size=self.chunk_size,
            is_udp_tx=False,
            pool=allocator,
            name="audio_packetizer",
        )

        # Use RoCE transmitter operator instead of UDP transmitter
        # buffer_size is the maximum size of a single transmitted buffer.
        # With 24-bit stereo at 48 kHz, a chunk of 192 samples corresponds to:
        # 192 frames * 2 channels * 4 bytes per sample (padded to 32-bit) = 1536 bytes.
        # We keep buffer_size configurable at the operator level; the operator will
        # derive sizes from the incoming tensor.
        buffer_size = 12 + self.chunk_size + 4
        roce_transmitter = hololink_module.operators.RoceTransmitterOp(
            self,
            name="roce_transmitter",
            ibv_name=self.ibv_name,
            ibv_port=self.ibv_port,
            hololink_ip=self.hololink_ip,
            ibv_qp=self.ibv_qp,
            buffer_size=buffer_size,
            queue_size=self.queue_size,
        )

        # Connect operators
        self.add_flow(audio_packetizer, roce_transmitter, {("out", "input")})


def main():
    rx_default_infiniband_interface = "mlx5_0"
    try:
        infiniband_interfaces = hololink_module.infiniband_devices()
        if infiniband_interfaces:
            rx_default_infiniband_interface = infiniband_interfaces[0]
    except FileNotFoundError:
        pass

    parser = argparse.ArgumentParser(description="Audio RoCE Transmission Application")
    parser.add_argument(
        "--wav-file", required=True, help="Path to WAV file to transmit"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=192, help="Size of audio chunks to transmit"
    )
    parser.add_argument("--hololink", required=True, help="Hololink IP address")
    parser.add_argument(
        "--ibv-name",
        type=str,
        default=rx_default_infiniband_interface,
        help="IBV device name used for transmission",
    )
    parser.add_argument(
        "--ibv-port",
        type=int,
        default=1,
        help="Port number of IBV device used for transmission",
    )
    parser.add_argument(
        "--ibv-qp",
        type=int,
        default=2,
        help="QP number for the IBV stream that the data is transmitted to",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=3,
        help="The number of buffers that can wait to be transmitted",
    )

    args = parser.parse_args()

    channel_metadata = hololink_module.Enumerator.find_channel(channel_ip=args.hololink)
    logging.info(f"{channel_metadata=}")
    hololink_channel = hololink_module.DataChannel(channel_metadata)

    app = AudioRoceTransmitApp(
        wav_file=args.wav_file,
        chunk_size=args.chunk_size,
        hololink_ip=args.hololink,
        ibv_name=args.ibv_name,
        ibv_port=args.ibv_port,
        ibv_qp=args.ibv_qp,
        queue_size=args.queue_size,
    )

    hololink = hololink_channel.hololink()
    hololink.start()
    en_i2s_tx(hololink)

    app.run()


if __name__ == "__main__":
    main()
