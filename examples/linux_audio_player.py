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
#
# See README.md for detailed information.

# Plays a WAV file to a Hololink board's I2S output over UDP. en_i2s_tx() (a direct
# register-poke sequence, no Holoscan dependency) configures the I2S TX path once at startup;
# AudioUdpPacketizerOp then reads the WAV file and emits IB-header+CRC-framed chunks to
# UdpTransmitterOp (already HSDK5-native; no C++ work needed for this app).
#
# AudioPacketizerOp (src/hololink/operators/audio_packetizer/) is NOT reused here -- it's still
# HSDK4-era GXF/Entity code (HOLOSCAN_OPERATOR_FORWARD_ARGS, 3-arg compute()), and its own logic
# has no hardware/driver dependency at all (reads a file host-side, uploads chunks to the GPU).
# Matches the user's own correction earlier in this porting effort ("HSDK supports native Python
# operators, not every Python operator needs an equivalent C++ one"): AudioUdpPacketizerOp below
# reproduces its exact byte layout and CRC logic in pure Python instead of rewriting the C++
# operator to HSDK5's CRTP/tensor-pool shape (which would mean designing a new periodic-source
# operator pattern with no close precedent in this codebase -- every other source this whole
# effort has ported is a receiver driven by hardware notification, not a file-chunk timer).
#
# chunk_size is a **byte** count (used directly as a slice length into the expanded WAV sample
# buffer), despite the old app's own CLI help text and inline comments implying "samples" --
# confirmed from audio_packetizer.cpp's own compute(), which is what actually ran.
#
# Realtime pacing is a deliberate behavior improvement over the old app, not just a port
# artifact: the old AudioPacketizerOp had no explicit pacing at all, ticking as fast as the
# HSDK4 greedy scheduler allowed and relying entirely on UdpTransmitterOp's own queue/
# backpressure to avoid flooding the network. HSDK5 requires an explicit contract() trigger for
# a custom operator; hs.OnClock(period=...) paced to the WAV's actual playback rate is the
# correct behavior for "play this audio file" -- not merely a mechanical substitute. The WAV
# header is read once in main(), before the graph is built, specifically so chunk_period_s is
# known before AudioUdpPacketizerOp.contract() (evaluated at compile time, before start()) needs
# it -- start() re-reads the WAV file for its full sample data, matching the old app's own
# split between construction-time knowledge and start()-time file loading.
#
# Only 24-bit WAV input is supported, matching the old C++ operator's own hardcoded assumption
# (it unconditionally treats each sample as 3 bytes, zero-padded to 4) -- not a new restriction
# introduced by this port.

import argparse
import logging
import struct
import sys
import time
import wave
import zlib
from datetime import timedelta

import cupy as cp
import holoscan as hs
import numpy as np

import hololink as hl

SLEEP_TIME = 1

IB_HEADER_BYTES = 12
CRC_BYTES = 4


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
    hololink.write_uint32(0x01200004, af)
    time.sleep(SLEEP_TIME)
    hololink.write_uint32(0x01200008, ae)
    time.sleep(SLEEP_TIME)
    hololink.write_uint32(0x01200000, 0x00000004)
    logging.info("Finished setting TX almost full & empty")


def set_tx_pause(hololink):
    hololink.write_uint32(0x01200000, 0x00000003)
    time.sleep(SLEEP_TIME)
    hololink.write_uint32(0x0120000C, 0x00000001)
    time.sleep(SLEEP_TIME)
    logging.info("Finished setting TX and pause")


def en_i2s_tx(hololink):
    enable_i2s(hololink)
    set_tx_af_ae(hololink, 0x00000800, 0x00000400)
    set_tx_pause(hololink)


def _read_wav_header(wav_file):
    """Return (channels, sample_rate) after validating 24-bit sample width."""
    with wave.open(wav_file, "rb") as wf:
        sampwidth = wf.getsampwidth()
        if sampwidth != 3:
            raise ValueError(
                f"{wav_file}: only 24-bit WAV files are supported (got {sampwidth * 8}-bit)"
            )
        return wf.getnchannels(), wf.getframerate()


def _load_wav_samples(wav_file):
    """Read raw interleaved 24-bit samples and zero-pad each to 4 bytes (matches the old
    AudioPacketizerOp::load_wav_file()'s own fixed 24-bit-only expansion)."""
    with wave.open(wav_file, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    samples = len(raw) // 3
    trimmed = np.frombuffer(raw[: samples * 3], dtype=np.uint8).reshape(-1, 3)
    padded = np.zeros((samples, 4), dtype=np.uint8)
    padded[:, :3] = trimmed
    return padded.reshape(-1).tobytes()


class AudioUdpPacketizerOp(hs.Operator):
    """Reads a 24-bit WAV file and emits IB-header+CRC-framed chunks for UdpTransmitterOp.

    Byte layout matches audio_packetizer.cpp's compute() exactly: a 12-byte fabricated IB
    header, chunk_size bytes of payload (zero-padded at end of file), and a 4-byte big-endian
    CRC32 (zlib seed 0xdebb20e3) of the payload alone.
    """

    out = hs.Output("out", hs.payload.Tensor)

    wav_file = hs.Param("")
    chunk_size = hs.Param(192)
    chunk_period_s = hs.Param(0.004)

    def setup(self, spec):
        spec.output(self.out).produces_tensor(
            hs.tensor.OutputSpec(
                storage="runtime_pool",
                memory_kind=hs.tensor.MemoryKind.CUDA_DEVICE,
                dtype=hs.tensor.DLDataType(1, 8, 1),
                rank=1,
            ),
            hs.tensor.byte_span(IB_HEADER_BYTES + self.chunk_size + CRC_BYTES),
        )

    def contract(self):
        return hs.Contract().trigger(
            hs.OnClock(period=timedelta(seconds=self.chunk_period_s))
        )

    def start(self):
        self._audio_data = _load_wav_samples(self.wav_file)
        self._pos = 0
        self._num_packets = 0
        logging.info("AudioUdpPacketizerOp started with WAV file: %s", self.wav_file)

    def compute(self, context):
        chunk = self.chunk_size
        payload_size = min(chunk, len(self._audio_data) - self._pos)

        packet = bytearray(IB_HEADER_BYTES + chunk + CRC_BYTES)
        struct.pack_into(">I", packet, 0, (0x24 << 24) | 0x40FFFF)
        struct.pack_into(">I", packet, 4, 2 & 0xFFFFFF)
        struct.pack_into(">I", packet, 8, self._num_packets & 0xFFFFFF)

        packet[IB_HEADER_BYTES : IB_HEADER_BYTES + payload_size] = self._audio_data[
            self._pos : self._pos + payload_size
        ]
        # Tail beyond payload_size (short final chunk) stays zero -- bytearray() is zero-filled.

        payload = bytes(packet[IB_HEADER_BYTES : IB_HEADER_BYTES + chunk])
        crc = zlib.crc32(payload, 0xDEBB20E3) & 0xFFFFFFFF
        struct.pack_into(">I", packet, IB_HEADER_BYTES + chunk, crc)

        packet_np = np.frombuffer(bytes(packet), dtype=np.uint8)
        with cp.cuda.Stream(non_blocking=True) as stream:
            with self.out.allocate_tensor(shape=packet_np.shape, dtype="uint8") as loan:
                memory = cp.cuda.UnownedMemory(
                    loan.tensor.data, packet_np.nbytes, loan.tensor
                )
                destination = cp.ndarray(
                    packet_np.shape,
                    dtype=cp.uint8,
                    memptr=cp.cuda.MemoryPointer(memory, 0),
                )
                destination.set(packet_np)
                # Publish only after the packet is ready for the native transmitter.
                stream.synchronize()

        self._pos += payload_size
        if self._pos >= len(self._audio_data):
            self._pos = 0
        self._num_packets += 1


def run(wav_file, chunk_size, hololink_ip):
    channels, sample_rate = _read_wav_header(wav_file)
    frames_per_chunk = chunk_size / (
        4 * channels
    )  # 4 bytes/sample after 24->32-bit padding
    chunk_period_s = frames_per_chunk / sample_rate

    metadata = hl.find_channel(hololink_ip)
    channel = hl.DataChannel(metadata)
    hololink_dev = channel.hololink()
    try:
        hololink_dev.start()
        en_i2s_tx(hololink_dev)

        graph = hs.Graph("linux_audio_player")
        packetizer = graph.op(
            AudioUdpPacketizerOp,
            "audio_packetizer",
            wav_file=wav_file,
            chunk_size=chunk_size,
            chunk_period_s=chunk_period_s,
        )
        transmitter = graph.op(
            hl.operators.UdpTransmitterOp,
            "udp_transmitter",
            ip=hololink_ip,
            port=4791,
            lossy=False,
        )
        # One packet in the worker, one queued, and one in compute/enqueue.
        graph.add_flow(packetizer.out, transmitter.input, queue_depth=3)

        clock_key = graph.add_clock(hs.RealtimeClock, "clock")
        graph.set_default_clock(clock_key)

        options = hs.CompileOptions()
        options.deployment.bind_tensor_output_device(
            operator_path="audio_packetizer",
            output_port="out",
            device=hs.execution.DeviceId(0),
        )
        plan = hs.compile(graph, options)
        if not plan.ok:
            print("Graph compilation failed:", file=sys.stderr)
            for diag in plan.diagnostics:
                print(f"  {diag}", file=sys.stderr)
            return 1

        print(
            f"Playing {wav_file} ({sample_rate} Hz, {channels} ch) -- Ctrl+C to stop..."
        )
        with hs.run_async(plan) as session:
            try:
                session.wait()
            except KeyboardInterrupt:
                session.request_stop()
                session.wait()
            failed = any(
                diag.severity == hs.diagnostics.DiagnosticSeverity.ERROR
                for diag in session.diagnostics
            )
        if failed:
            print("Run failed; see runtime diagnostics above.", file=sys.stderr)
            return 1
        print("Done.")
        return 0
    finally:
        hololink_dev.stop()


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

    return run(args.wav_file, args.chunk_size, args.hololink)


if __name__ == "__main__":
    sys.exit(main())
