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

# Plays a WAV file to a Hololink board's I2S output over RoCE. hololink.get_i2s() configures
# the I2S TX path once at startup (defaults: mclk_div=16, no loopback, 24-bit I2S stereo, TX/RX
# enabled); AudioRocePacketizerOp then reads the WAV file and emits raw (no IB header/CRC)
# chunks to RoceTransmitterOp.
#
# See linux_audio_player.py's module docstring for the full rationale behind porting
# AudioPacketizerOp to pure Python instead of HSDK5 C++ (no hardware/driver dependency; matches
# the user's "not every Python operator needs a C++ one" correction), the realtime hs.OnClock
# pacing design (a deliberate improvement over the old app's unpaced scheduler-rate ticking),
# and the 24-bit-WAV-only restriction (matches the old C++ operator's own hardcoded assumption).
# AudioRocePacketizerOp below is this app's own copy, with RoCE's simpler no-header/no-CRC
# framing (is_udp_tx=False in the old app) rather than duplicating one dual-mode class with a
# branch neither app ever takes both sides of.
#
# get_i2s() wasn't exposed to Python at all before this port (confirmed by grepping
# _hololink.cpp) -- added as part of this round. RoceTransmitterOp likewise needed a
# HOLOSCAN_PARAM_SCHEMA + registration call it was missing (it was already a modern
# Operator<RoceTransmitterParams>, just never wired into the catalog).

import argparse
import logging
import sys
import wave
from datetime import timedelta

import cupy as cp
import holoscan as hs
import numpy as np

import hololink as hl


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


class AudioRocePacketizerOp(hs.Operator):
    """Reads a 24-bit WAV file and emits raw chunk_size-byte payloads for RoceTransmitterOp.

    No IB header or CRC -- RoceTransmitterOp handles its own framing on the wire (is_udp_tx
    False in the old app's C++ operator).
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
            hs.tensor.byte_span(self.chunk_size),
        )

    def contract(self):
        return hs.Contract().trigger(
            hs.OnClock(period=timedelta(seconds=self.chunk_period_s))
        )

    def start(self):
        self._audio_data = _load_wav_samples(self.wav_file)
        self._pos = 0
        logging.info("AudioRocePacketizerOp started with WAV file: %s", self.wav_file)

    def compute(self, context):
        chunk = self.chunk_size
        payload_size = min(chunk, len(self._audio_data) - self._pos)

        packet = bytearray(chunk)
        packet[:payload_size] = self._audio_data[self._pos : self._pos + payload_size]
        # Tail beyond payload_size (short final chunk) stays zero -- bytearray() is zero-filled.

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
                # The native transmitter must see a completed upload on publication.
                stream.synchronize()

        self._pos += payload_size
        if self._pos >= len(self._audio_data):
            self._pos = 0


def run(
    wav_file,
    chunk_size,
    hololink_ip,
    ibv_name,
    ibv_port,
    ibv_qp,
    queue_size,
):
    channels, sample_rate = _read_wav_header(wav_file)
    frames_per_chunk = chunk_size / (
        4 * channels
    )  # 4 bytes/sample after 24->32-bit padding
    chunk_period_s = frames_per_chunk / sample_rate

    metadata = hl.find_channel(hololink_ip)
    if not ibv_name:
        # The device on the interface the board enumerated on: RoceTransmitterOp, which
        # has no DataChannel, would take the routing table's.
        ibv_name, ibv_port = hl.infiniband_device_for_peer(
            hololink_ip, metadata.get_string("interface") or ""
        )
        logging.info(
            f"Using IB device {ibv_name} port {ibv_port} for the board at {hololink_ip}"
        )
    channel = hl.DataChannel(metadata)
    hololink_dev = channel.hololink()
    try:
        hololink_dev.start()
        # Configure I2S TX: mclk_div=16, no loopback, 24-bit stereo, TX/RX enabled.
        hololink_dev.get_i2s(metadata)

        # Preserve the old app's 16-byte staging-buffer headroom. The transmitter
        # derives each send's actual size from the incoming tensor.
        buffer_size = chunk_size + 16

        graph = hs.Graph("audio_player")
        packetizer = graph.op(
            AudioRocePacketizerOp,
            "audio_packetizer",
            wav_file=wav_file,
            chunk_size=chunk_size,
            chunk_period_s=chunk_period_s,
        )
        transmitter = graph.op(
            hl.operators.RoceTransmitterOp,
            "roce_transmitter",
            ibv_name=ibv_name,
            ibv_port=ibv_port,
            hololink_ip=hololink_ip,
            ibv_qp=ibv_qp,
            buffer_size=buffer_size,
            queue_size=queue_size,
        )
        # Allow retained packets plus the next compute/enqueue borrow.
        graph.add_flow(packetizer.out, transmitter.input, queue_depth=queue_size + 1)

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
        default="",
        help="IBV device name used for transmission "
        "(default: the device on the board's network interface)",
    )
    parser.add_argument(
        "--ibv-port",
        type=int,
        default=1,
        help="Port number of IBV device used for transmission (with --ibv-name)",
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

    return run(
        args.wav_file,
        args.chunk_size,
        args.hololink,
        args.ibv_name,
        args.ibv_port,
        args.ibv_qp,
        args.queue_size,
    )


if __name__ == "__main__":
    sys.exit(main())
