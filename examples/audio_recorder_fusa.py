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

"""
Hololink live audio capture: I2S RX -> FuSa CoE receive -> ALSA playback (optional WAV).

Captures I2S audio via FuSa CoE and plays it back through ALSA (optionally
writing a WAV file).

The FuSa CoE wire format delivers a 256 B payload of 32 interleaved stereo S32 pairs
(slot 0 -> left, slot 1 -> right), identical in layout to the RoCE transport.

``FusaCoeCaptureOp`` is configured with ``cpu_output=True`` so the output tensor is
backed by the CPU-accessible NvSciBuf VA (kHost memory).  ``bytes_written`` and the
PCM payload are read directly from that buffer — no CUDA driver involvement.  Keepalive
frames (164 B) are skipped.
"""

import argparse
import ctypes
import logging
import os
import queue
import struct
import threading
import time
import wave

import cuda.bindings.driver as cuda
import holoscan
import numpy as np

import hololink as hololink_module

# FuSa capture page (4096 B). Do not use 256 — NvSciBuf MGBE pitch is 4 KiB.
FUSA_CAPTURE_PAGE_BYTES = 4096

PCM_BYTES = 256
KEEPALIVE_BYTES = 164
SAMPLES_PER_FRAME = PCM_BYTES // 4 // 2  # 32 stereo pairs

DEFAULT_FRAME_SIZE = FUSA_CAPTURE_PAGE_BYTES
DEFAULT_TIMEOUT_MS = 1500

_HSB_METADATA_BYTES_WRITTEN_OFFSET = 24

ALSA_CHANNELS = 2
ALSA_SAMPLE_BYTES = 4
# 512 frames (~11 ms) avoids PulseAudio underruns; writing one 32-sample FuSa
# frame per write (32 frames) causes ~1500 IPC calls/sec and scratchiness.
ALSA_PERIOD_FRAMES = 512
DEFAULT_SAMPLE_RATE = 48000

PLAYBACK_QUEUE_CHUNKS = 256
WAV_QUEUE_CHUNKS = 4096
WAV_HEADER_REFRESH_FRAMES = DEFAULT_SAMPLE_RATE // SAMPLES_PER_FRAME


def _i32_wrap(v: int) -> int:
    """Wrap a Python int into the signed 32-bit range (matches C int32_t overflow)."""
    v = int(v)
    return ((v + 0x80000000) & 0xFFFFFFFF) - 0x80000000


def _default_alsa_device() -> str:
    """Return ALSA device: ALSA_PCM_DEVICE env override, pulse if PULSE_SERVER is set, else default."""
    if os.environ.get("ALSA_PCM_DEVICE"):
        return os.environ["ALSA_PCM_DEVICE"]
    if os.environ.get("PULSE_SERVER"):
        return "pulse"
    return "default"


def _read_bytes_written_from_cpu(cpu_ptr: int, capture_frame_bytes: int) -> int:
    """Read bytes_written from the HSB metadata block appended after the capture payload."""
    offset = (
        hololink_module.round_up(capture_frame_bytes, 128)
        + _HSB_METADATA_BYTES_WRITTEN_OFFSET
    )
    raw = bytes((ctypes.c_uint8 * 8).from_address(cpu_ptr + offset))
    (bytes_written,) = struct.unpack(">Q", raw)
    return int(bytes_written)


def _extract_pcm_payload(raw: np.ndarray, valid_bytes: int) -> np.ndarray | None:
    """Return 256 B PCM payload, or None for keepalive/invalid frames."""
    if valid_bytes == KEEPALIVE_BYTES or valid_bytes <= 0:
        return None
    if valid_bytes < PCM_BYTES:
        logging.debug("skip frame: valid_bytes=%d too short for PCM", valid_bytes)
        return None
    return raw[:PCM_BYTES]


def _pcm_to_alsa_bytes(
    pcm: np.ndarray,
    *,
    dc_x_prev: list[int],
    dc_y_prev: list[int],
    alsa_endianness: str = "BE",
) -> bytes:
    """Decode one 256 B CoE PCM block and apply the DC blocker."""
    x_arr = np.frombuffer(pcm, dtype=">i4").reshape(-1, ALSA_CHANNELS)
    y_arr = np.empty_like(x_arr)
    for ch in range(ALSA_CHANNELS):
        x_prev = dc_x_prev[ch]
        y_prev = dc_y_prev[ch]
        yc = [0] * x_arr.shape[0]
        for i, cur in enumerate(x_arr[:, ch].tolist()):
            yi = _i32_wrap(cur - x_prev + y_prev - (y_prev >> 8))
            yc[i] = yi
            x_prev = cur
            y_prev = yi
        y_arr[:, ch] = yc
        dc_x_prev[ch] = x_prev
        dc_y_prev[ch] = y_prev

    dtype = "<i4" if alsa_endianness == "LE" else ">i4"
    return y_arr.astype(dtype, copy=False).tobytes()


class AudioRxDevice:
    """Minimal device handle for ``FusaCoeCaptureOp``; no per-frame state needed."""

    def __init__(
        self,
        hololink: hololink_module.Hololink,
        enumeration_metadata: hololink_module.Metadata,
    ):
        self._hololink = hololink
        self._enumeration_metadata = enumeration_metadata

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class AudioAlsaPlaybackOp(holoscan.core.Operator):
    """Decode FuSa CoE I2S audio frames and forward to ALSA and/or a WAV file.

    Decoding runs inline in ``compute()``.  Two worker threads decouple ALSA
    writes and WAV I/O from the capture path.
    """

    def __init__(
        self,
        fragment,
        *,
        alsa_device: str,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        alsa_endianness: str = "BE",
        output_path: str | None = None,
        no_playback: bool = False,
        frame_size: int = DEFAULT_FRAME_SIZE,
        log_every: int = 128,
        **kwargs,
    ):
        super().__init__(fragment, **kwargs)
        self._frame_size = frame_size
        self._alsa_device = alsa_device
        self._sample_rate = sample_rate
        self._output_path = output_path
        self._no_playback = no_playback
        if self._no_playback and not self._output_path:
            raise ValueError("--no-playback requires --output")
        self._alsa_endianness = alsa_endianness.upper()
        if self._alsa_endianness not in ("BE", "LE"):
            raise ValueError("alsa_endianness must be BE or LE")
        self._log_every = max(1, log_every)
        self._frames = 0
        self._skipped = 0
        self._first_frame_time: float | None = None
        self._dc_x_prev = [0, 0]
        self._dc_y_prev = [0, 0]
        self._alsa_pcm = None
        self._alsa_period_frames = ALSA_PERIOD_FRAMES
        self._alsa_write_bytes = (
            self._alsa_period_frames * ALSA_CHANNELS * ALSA_SAMPLE_BYTES
        )
        self._play_queue: queue.Queue[bytes | None] = queue.Queue(
            maxsize=PLAYBACK_QUEUE_CHUNKS
        )
        self._play_thread: threading.Thread | None = None
        self._dropped = 0
        self._wav_queue: queue.Queue[bytes | None] = queue.Queue(
            maxsize=WAV_QUEUE_CHUNKS
        )
        self._wav_thread: threading.Thread | None = None
        self._wav_writes = 0
        self._wav_dropped = 0
        self._wav_file: wave.Wave_write | None = None
        self._alsa_error_cls: type[Exception] = Exception
        self._alsaaudio = None
        self._alsa_fail_streak = 0
        self._alsa_writes = 0

    def setup(self, spec):
        spec.input("input")

    def _create_alsa_pcm(self):
        """Open the playback PCM handle."""
        assert self._alsaaudio is not None
        fmt = (
            self._alsaaudio.PCM_FORMAT_S32_LE
            if self._alsa_endianness == "LE"
            else self._alsaaudio.PCM_FORMAT_S32_BE
        )
        return self._alsaaudio.PCM(
            self._alsaaudio.PCM_PLAYBACK,
            self._alsaaudio.PCM_NORMAL,
            device=self._alsa_device,
            channels=ALSA_CHANNELS,
            rate=self._sample_rate,
            format=fmt,
            periodsize=self._alsa_period_frames,
        )

    def start(self):
        try:
            if self._output_path:
                self._wav_file = wave.open(self._output_path, "wb")
                self._wav_file.setnchannels(ALSA_CHANNELS)
                self._wav_file.setsampwidth(ALSA_SAMPLE_BYTES)
                self._wav_file.setframerate(self._sample_rate)
                logging.info(
                    "WAV open: %s rate=%d channels=%d sampwidth=%d",
                    self._output_path,
                    self._sample_rate,
                    ALSA_CHANNELS,
                    ALSA_SAMPLE_BYTES,
                )

            if self._no_playback:
                logging.info("ALSA playback disabled (--no-playback)")
            else:
                try:
                    import alsaaudio
                except ImportError as e:
                    raise RuntimeError(
                        "python3-alsaaudio required (installed in hololink-demo image)"
                    ) from e
                self._alsaaudio = alsaaudio
                self._alsa_error_cls = alsaaudio.ALSAAudioError
                self._alsa_pcm = self._create_alsa_pcm()
                logging.info(
                    "ALSA open: device=%s rate=%d period_frames=%d (%d B/write) "
                    "stereo S32 %s",
                    self._alsa_device,
                    self._sample_rate,
                    self._alsa_period_frames,
                    self._alsa_write_bytes,
                    self._alsa_endianness,
                )
        except Exception:
            if self._wav_file is not None:
                self._wav_file.close()
                self._wav_file = None
            if self._alsa_pcm is not None:
                self._alsa_pcm.close()
                self._alsa_pcm = None
            raise

        if self._wav_file is not None:
            self._wav_thread = threading.Thread(
                target=self._wav_writer, name="wav-writer", daemon=True
            )
            self._wav_thread.start()
        if self._alsa_pcm is not None:
            self._play_thread = threading.Thread(
                target=self._playback_worker, name="alsa-playback", daemon=True
            )
            self._play_thread.start()

    def _process_host_page(
        self, host_page: np.ndarray, bytes_written: int | None
    ) -> None:
        valid = len(host_page) if bytes_written is None else int(bytes_written)
        pcm_block = _extract_pcm_payload(host_page, valid)
        if pcm_block is None:
            self._skipped += 1
            return

        try:
            pcm = _pcm_to_alsa_bytes(
                pcm_block,
                dc_x_prev=self._dc_x_prev,
                dc_y_prev=self._dc_y_prev,
                alsa_endianness=self._alsa_endianness,
            )
        except ValueError as e:
            logging.debug("skip frame: %s", e)
            self._skipped += 1
            return

        if not self._no_playback and self._enqueue_drop_oldest(self._play_queue, pcm):
            self._dropped += 1
        if self._wav_file is not None and self._enqueue_drop_oldest(
            self._wav_queue, pcm
        ):
            self._wav_dropped += 1
        self._frames += 1
        now = time.monotonic()
        if self._first_frame_time is None:
            self._first_frame_time = now

        if self._frames % self._log_every == 0:
            elapsed = now - self._first_frame_time
            rate = self._frames / elapsed if elapsed > 0 else 0.0
            logging.debug(
                "fusa_frame=%d bytes_written=%s rate=%.0f/s "
                "play_q=%d play_drop=%d wav_q=%d wav_drop=%d",
                self._frames,
                bytes_written,
                rate,
                self._play_queue.qsize(),
                self._dropped,
                self._wav_queue.qsize(),
                self._wav_dropped,
            )

    def _stop_worker(
        self, thread: threading.Thread, q: queue.Queue, timeout: float | None = 2.0
    ) -> None:
        # Send None sentinel so the worker drains and exits, then join.
        try:
            q.put(None, timeout=timeout)  # sentinel: drain remainder and exit
        except queue.Full:
            # Finite timeout expired with the worker stalled and the queue full.
            # Drop one item so the sentinel still lands (single producer => fits),
            # letting the worker exit cleanly once it resumes.
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            q.put_nowait(None)
        thread.join(timeout=timeout)

    def stop(self):
        # ALSA: bounded wait; losing a few buffered samples at exit is fine.
        if self._play_thread is not None:
            self._stop_worker(self._play_thread, self._play_queue)
            self._play_thread = None
        # WAV: wait for a full drain so every queued frame is written before
        # close(), even on slow storage (recording must not lose the tail).
        if self._wav_thread is not None:
            self._stop_worker(self._wav_thread, self._wav_queue, timeout=None)
            self._wav_thread = None
        if self._wav_file is not None:
            self._wav_file.close()
            self._wav_file = None
            samples = self._wav_writes * SAMPLES_PER_FRAME
            logging.info(
                "WAV closed: %s (%d frames written, %d samples/channel, %d dropped)",
                self._output_path,
                self._wav_writes,
                samples,
                self._wav_dropped,
            )
        if self._alsa_pcm is not None:
            try:
                self._alsa_pcm.close()
            except Exception as e:
                logging.warning("ALSA close: %s", e)
            self._alsa_pcm = None
        elapsed = (
            time.monotonic() - self._first_frame_time if self._first_frame_time else 0.0
        )
        rate = self._frames / elapsed if elapsed > 0 else 0.0
        logging.info(
            "Audio sink stopped after %d FuSa frames (%.0f frames/sec avg, "
            "%d keepalive/non-audio skipped, %d playback queue drops)",
            self._frames,
            rate,
            self._skipped,
            self._dropped,
        )

    def _playback_worker(self) -> None:
        pending = bytearray()
        while True:
            item = self._play_queue.get()
            if item is None:
                if pending:
                    self._alsa_write(bytes(pending))
                break
            try:
                pending += item
                while len(pending) >= self._alsa_write_bytes:
                    self._alsa_write(bytes(pending[: self._alsa_write_bytes]))
                    del pending[: self._alsa_write_bytes]
            except Exception as e:
                logging.warning("ALSA playback: %s", e)

    def _reopen_alsa(self) -> None:
        if self._alsaaudio is None:
            return
        if self._alsa_pcm is not None:
            try:
                self._alsa_pcm.close()
            except Exception:
                pass
        try:
            self._alsa_pcm = self._create_alsa_pcm()
            logging.info("ALSA reopened: device=%s", self._alsa_device)
        except Exception as e:
            logging.error("ALSA reopen failed: %s", e)
            self._alsa_pcm = None

    def _enqueue_drop_oldest(self, q: queue.Queue, pcm: bytes) -> bool:
        try:
            q.put_nowait(pcm)
            return False
        except queue.Full:
            pass
        dropped = False
        try:
            q.get_nowait()
            dropped = True
        except queue.Empty:
            pass
        q.put_nowait(pcm)
        return dropped

    def _pcm_for_wav(self, pcm: bytes) -> bytes:
        """Return PCM bytes in WAV's required little-endian layout."""
        if self._alsa_endianness == "LE":
            return pcm
        return np.frombuffer(pcm, dtype=">i4").astype("<i4", copy=False).tobytes()

    def _wav_writer(self) -> None:
        while True:
            item = self._wav_queue.get()
            if item is None:
                break
            try:
                self._wav_file.writeframesraw(self._pcm_for_wav(item))
                self._wav_writes += 1
                if self._wav_writes % WAV_HEADER_REFRESH_FRAMES == 0:
                    self._wav_file.writeframes(b"")
            except Exception as e:
                logging.warning("WAV write: %s", e)

    def _alsa_write(self, data: bytes) -> None:
        if self._alsa_pcm is None:
            return
        try:
            self._alsa_pcm.write(data)
            self._alsa_fail_streak = 0
            if self._alsa_writes == 0:
                samples = np.frombuffer(data, dtype="<i4")
                peak = int(np.max(np.abs(samples))) if len(samples) else 0
                logging.info(
                    "ALSA first write: %d B, %d samples, peak=%d",
                    len(data),
                    len(samples),
                    peak,
                )
            self._alsa_writes += 1
            return
        except self._alsa_error_cls:
            pass
        try:
            self._alsa_pcm.drop()
            self._alsa_pcm.write(data)
            self._alsa_fail_streak = 0
            return
        except self._alsa_error_cls as e:
            self._alsa_fail_streak += 1
            if self._alsa_fail_streak <= 3 or self._alsa_fail_streak % 128 == 0:
                logging.warning(
                    "ALSA write failed (%d consecutive): %s",
                    self._alsa_fail_streak,
                    e,
                )
            if self._alsa_fail_streak in (4, 32):
                self._reopen_alsa()
                if self._alsa_pcm is not None:
                    try:
                        self._alsa_pcm.write(data)
                        self._alsa_fail_streak = 0
                    except self._alsa_error_cls as e2:
                        logging.warning("ALSA write after reopen failed: %s", e2)
            if self._alsa_fail_streak >= 256 and self._alsa_pcm is not None:
                logging.error(
                    "Disabling ALSA playback after %d consecutive write failures "
                    "(try --alsa-device pulse or ALSA_PCM_DEVICE=default)",
                    self._alsa_fail_streak,
                )
                try:
                    self._alsa_pcm.close()
                except Exception:
                    pass
                self._alsa_pcm = None

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("input")
        # FusaCoeCaptureOp is configured with cpu_output=True, so the tensor is
        # backed by the CPU-accessible NvSciBuf VA (kHost).  np.asarray() gives a
        # zero-copy view; .ctypes.data is the raw address for HSB metadata reading.
        host_buf = np.asarray(in_message.get(""), dtype=np.uint8)
        cpu_ptr = host_buf.ctypes.data
        raw_bw = _read_bytes_written_from_cpu(cpu_ptr, self._frame_size)
        bytes_written = 0 if raw_bw is None else raw_bw
        copy_len = max(0, min(int(bytes_written), PCM_BYTES))
        host_page = host_buf[:copy_len].copy()
        del host_buf, in_message  # release FuSa buffer pool entry

        self._process_host_page(host_page, bytes_written)


class AudioFusaPlaybackApp(holoscan.core.Application):
    def __init__(
        self,
        hololink_channel: hololink_module.DataChannel,
        *,
        frame_size: int,
        interface: str | None,
        timeout_ms: int,
        frame_limit: int | None,
        alsa_device: str,
        sample_rate: int,
        alsa_endianness: str,
        output_path: str | None,
        no_playback: bool,
        enumeration_metadata: hololink_module.Metadata,
        hololink: hololink_module.Hololink,
    ):
        logging.info("__init__")
        super().__init__()
        self.enable_metadata(True)
        self._hololink_channel = hololink_channel
        self._frame_size = frame_size
        self._interface = interface
        self._timeout_ms = timeout_ms
        self._frame_limit = frame_limit
        self._alsa_device = alsa_device
        self._sample_rate = sample_rate
        self._alsa_endianness = alsa_endianness
        self._output_path = output_path
        self._no_playback = no_playback
        self._enumeration_metadata = enumeration_metadata
        self._hololink = hololink

    def compose(self):
        logging.info("compose")
        if self._frame_limit:
            condition = holoscan.conditions.CountCondition(
                self, name="rx_count", count=self._frame_limit
            )
        else:
            condition = holoscan.conditions.BooleanCondition(
                self, name="rx_run", enable_tick=True
            )

        interface = self._interface or self._enumeration_metadata.get("interface")
        hsb_mac = self._enumeration_metadata.get("mac_id")
        mac_addr = list(bytes.fromhex(hsb_mac.replace(":", "")))

        device = AudioRxDevice(self._hololink, self._enumeration_metadata)
        receiver = hololink_module.operators.FusaCoeCaptureOp(
            self,
            condition,
            name="fusa_receiver",
            interface=interface,
            mac_addr=mac_addr,
            hololink_channel=self._hololink_channel,
            timeout=self._timeout_ms,
            device=device,
            cpu_output=True,
        )
        receiver.configure_frame_size(self._frame_size)
        playback = AudioAlsaPlaybackOp(
            self,
            name="alsa_playback",
            alsa_device=self._alsa_device,
            sample_rate=self._sample_rate,
            alsa_endianness=self._alsa_endianness,
            output_path=self._output_path,
            no_playback=self._no_playback,
            frame_size=self._frame_size,
        )
        self.add_flow(receiver, playback, {("output", "input")})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hololink I2S audio -> FuSa CoE -> ALSA live playback (optional WAV).",
    )
    parser.add_argument(
        "--hololink", default="192.168.0.2", help="IP address of Hololink board"
    )
    parser.add_argument(
        "--sensor",
        type=int,
        default=2,
        help="Sensor index of the I2S audio interface (default: 2 on hololink-lite)",
    )
    parser.add_argument(
        "--skip-reset",
        action="store_true",
        help="Skip the Hololink reset before bring-up",
    )
    parser.add_argument(
        "--log-level", type=int, default=20, help="Logging level to display"
    )
    parser.add_argument(
        "--frame-size",
        type=int,
        default=DEFAULT_FRAME_SIZE,
        help="FuSa CoE capture page size in bytes (default: 4096; do not use 256)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_MS,
        help="FuSa capture request timeout in milliseconds",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Exit after receiving this many frames",
    )
    parser.add_argument(
        "--interface",
        default=None,
        help="Host Ethernet interface (default: from enumeration metadata)",
    )
    parser.add_argument(
        "--alsa-device",
        default=_default_alsa_device(),
        help="ALSA PCM device (default: pulse when PULSE_SERVER is set, else default)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        choices=[DEFAULT_SAMPLE_RATE],
        default=DEFAULT_SAMPLE_RATE,
        help="Capture sample rate (Hz). Only 48000 is supported today.",
    )
    parser.add_argument(
        "--endianness",
        choices=["BE", "LE"],
        default="BE",
        help="Sample endianness presented to ALSA; the output WAV is always little-endian.",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="Write decoded stereo S32 PCM to a little-endian WAV file at PATH.",
    )
    parser.add_argument(
        "--no-playback",
        action="store_true",
        help="Capture-only mode; skip ALSA playback. Requires --output.",
    )
    args = parser.parse_args()
    hololink_module.logging_level(args.log_level)
    logging.info("Initializing.")

    channel_metadata = hololink_module.Enumerator.find_channel(channel_ip=args.hololink)
    logging.info(f"{channel_metadata=}")

    hololink_module.DataChannel.use_sensor(channel_metadata, args.sensor)
    hololink_channel = hololink_module.DataChannel(channel_metadata)
    hololink = hololink_channel.hololink()
    hololink.start()
    cu_dev = None
    cu_ctx = None
    try:
        (r,) = cuda.cuInit(0)
        if r != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuInit failed: {r}")
        r, cu_dev = cuda.cuDeviceGet(0)
        if r != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuDeviceGet failed: {r}")
        r, cu_ctx = cuda.cuDevicePrimaryCtxRetain(cu_dev)
        if r != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuDevicePrimaryCtxRetain failed: {r}")

        if not args.skip_reset:
            hololink.reset()

        logging.info(
            "HSB version=%#x FPGA datecode=%#x",
            hololink.get_hsb_ip_version(),
            hololink.get_fpga_date(),
        )

        hololink.get_i2s(channel_metadata, enable_tx=False)
        logging.info("I2S audio path configured")

        app = AudioFusaPlaybackApp(
            hololink_channel=hololink_channel,
            frame_size=args.frame_size,
            interface=args.interface,
            timeout_ms=args.timeout,
            frame_limit=args.frame_limit,
            alsa_device=args.alsa_device,
            sample_rate=args.sample_rate,
            alsa_endianness=args.endianness,
            output_path=args.output,
            no_playback=args.no_playback,
            enumeration_metadata=channel_metadata,
            hololink=hololink,
        )
        config = os.path.join(os.path.dirname(__file__), "example_configuration.yaml")
        if os.path.isfile(config):
            app.config(config)

        logging.info(
            "Starting: alsa=%s output=%s no_playback=%s frame_limit=%s interface=%s "
            "endianness=%s",
            args.alsa_device if not args.no_playback else "(disabled)",
            args.output or "(none)",
            args.no_playback,
            args.frame_limit or "unlimited",
            args.interface or channel_metadata.get("interface"),
            args.endianness,
        )
        logging.info("Calling run")
        app.run()
    finally:
        hololink.stop()
        if cu_ctx is not None:
            (r,) = cuda.cuDevicePrimaryCtxRelease(cu_dev)
            if r != cuda.CUresult.CUDA_SUCCESS:
                logging.warning("cuDevicePrimaryCtxRelease failed: %s", r)


if __name__ == "__main__":
    main()
