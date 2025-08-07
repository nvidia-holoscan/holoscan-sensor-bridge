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

import argparse
import logging
import time

import holoscan

import hololink as hololink_module

SLEEP_TIME = 1


def enable_i2s(hololink):
    hololink.write_uint32(0x70000014, 0x0000000F)
    time.sleep(SLEEP_TIME)
    hololink.write_uint32(0x80000100, 0x00000001)
    time.sleep(SLEEP_TIME)
    hololink.write_uint32(0x80000000, 0x00000001)
    time.sleep(SLEEP_TIME)
    hololink.write_uint32(0x0000002C, 0x00001000)  # dmic gpio in
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
    # set sensor tx 1 out
    hololink.write_uint32(0x01210000, 0x00000003)
    time.sleep(SLEEP_TIME)
    # set tx 0 host pause mapping
    hololink.write_uint32(0x0120000C, 0x00000001)
    time.sleep(SLEEP_TIME)
    # set tx 1 host pause mapping
    hololink.write_uint32(0x0120000C, 0x00000001)
    logging.info("Finished setting TX and pause")


def en_i2s_tx(hololink):
    enable_i2s(hololink)
    set_tx_af_ae(hololink, 0x00000800, 0x00000400)
    set_tx_pause(hololink)


class AudioTransmitApp(holoscan.core.Application):
    def __init__(self, wav_file, chunk_size, hololink_ip):
        super().__init__()
        self.wav_file = wav_file
        self.chunk_size = chunk_size
        self.hololink_ip = hololink_ip

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
            pool=allocator,
            name="audio_packetizer",
        )

        # Create UDP transmitter operator
        udp_transmitter = hololink_module.operators.UdpTransmitterOp(
            self,
            ip=self.hololink_ip,
            port=4791,
            name="udp_transmitter",
            lossy=False,
        )

        # Connect operators
        self.add_flow(audio_packetizer, udp_transmitter, {("out", "input")})


def main():
    parser = argparse.ArgumentParser(description="Audio Transmission Application")
    parser.add_argument(
        "--wav-file", required=True, help="Path to WAV file to transmit"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=192, help="Size of audio chunks to transmit"
    )
    parser.add_argument("--hololink", required=True, help="Hololink IP address")

    args = parser.parse_args()

    channel_metadata = hololink_module.Enumerator.find_channel(channel_ip=args.hololink)
    logging.info(f"{channel_metadata=}")
    hololink_channel = hololink_module.DataChannel(channel_metadata)

    app = AudioTransmitApp(
        wav_file=args.wav_file, chunk_size=args.chunk_size, hololink_ip=args.hololink
    )

    hololink = hololink_channel.hololink()
    hololink.start()
    en_i2s_tx(hololink)

    app.run()


if __name__ == "__main__":
    main()
