#!/usr/bin/env python3
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

"""
Hololink live audio capture: I2S RX -> FuSa CoE receive -> ALSA playback (optional WAV).

Captures I2S audio via FuSa CoE and plays it back through ALSA (optionally
writing a WAV file).

The FuSa CoE wire format delivers a 256 B payload of 32 interleaved stereo S32 pairs
(slot 0 -> left, slot 1 -> right), identical in layout to the RoCE transport.

``FusaCoeCaptureOp`` is configured with ``cpu_output=True`` so the output tensor is
backed by an SDK-owned host buffer. ``bytes_written`` arrives on the metadata
port, while PCM is read from the tensor. Keepalive
frames (164 B) are skipped.

Three real gaps this port needed, all genuinely missing rather than renamed (confirmed by
grepping the current tree, not assumed):
  - FusaCoeCaptureOp's own cpu_output support was a real, working feature on this repo's
    `internal` branch and was dropped during the HSDK5 rewrite (setup()/compute() hardcoded
    kCudaDevice) -- restored as part of this port, along with wiring frame_size_bytes (an
    already-present C++ Params field the catalog row never parsed) so this app can skip CSI
    geometry entirely, matching its own raw-byte-capture need.
  - hololink.get_i2s()/get_hsb_ip_version()/get_fpga_date()/hololink.round_up() were not
    exposed to Python at all -- added as part of this port.
  - FusaCoeCaptureOp takes its device start/stop as callbacks, so a plain Python device object
    like AudioRxDevice below passes its own methods (device_start=device.start).

Valid payload length comes from the capture metadata port, paired with the copied
host tensor. The SDK tensor does not include the NvSciBuf metadata trailer.

See audio_recorder.py's module docstring for the shared rationale (both apps' AudioAlsaPlaybackOp
is independently duplicated, matching this whole porting effort's established convention) behind
porting AudioAlsaPlaybackOp mechanically to hs.Operator's declarative style.
"""

import argparse
import logging
import os
import queue
import sys
import threading
import time
import wave
from datetime import timedelta

import capture_metadata
import cuda.bindings.driver as cuda_driver
import holoscan as hs
import numpy as np

import hololink as hl

# FuSa capture page (4096 B). Do not use 256 -- NvSciBuf MGBE pitch is 4 KiB.
FUSA_CAPTURE_PAGE_BYTES = 4096

PCM_BYTES = 256
KEEPALIVE_BYTES = 164
SAMPLES_PER_FRAME = PCM_BYTES // 4 // 2  # 32 stereo pairs

DEFAULT_FRAME_SIZE = FUSA_CAPTURE_PAGE_BYTES
DEFAULT_TIMEOUT_MS = 1500

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


def _extract_pcm_payload(raw, valid_bytes):
    """Return 256 B PCM payload, or None for keepalive/invalid frames."""
    if valid_bytes == KEEPALIVE_BYTES or valid_bytes <= 0:
        return None
    if valid_bytes < PCM_BYTES or len(raw) < PCM_BYTES:
        logging.debug("skip frame: valid_bytes=%d too short for PCM", valid_bytes)
        return None
    return raw[:PCM_BYTES]


def _pcm_to_alsa_bytes(pcm, *, dc_x_prev, dc_y_prev, alsa_endianness="BE"):
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

    def __init__(self, hololink, enumeration_metadata):
        self._hololink = hololink
        self._enumeration_metadata = enumeration_metadata

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class AudioAlsaPlaybackOp(hs.Operator):
    """Decode FuSa CoE I2S audio frames and forward to ALSA and/or a WAV file.

    Decoding runs inline in ``compute()``.  Two worker threads decouple ALSA
    writes and WAV I/O from the capture path.
    """

    in_tensor = hs.Input("in_tensor", hs.payload.Tensor)
    in_metadata = hs.Input("in_metadata", capture_metadata.CaptureMetadata)

    alsa_device = hs.Param("default")
    sample_rate = hs.Param(DEFAULT_SAMPLE_RATE)
    alsa_endianness = hs.Param("BE")
    output_path = hs.Param("")
    no_playback = hs.Param(False)
    log_every = hs.Param(128)

    def setup(self, spec):
        spec.input(self.in_tensor, queue_depth=4)
        spec.input(self.in_metadata, queue_depth=4)

    def contract(self):
        return (
            hs.Contract()
            .trigger(
                hs.OnAligned(
                    ports=[self.in_tensor, self.in_metadata],
                    window=timedelta(microseconds=500),
                )
            )
            .input(self.in_tensor, hs.Exact())
            .input(self.in_metadata, hs.Exact())
        )

    def start(self):
        if self.no_playback and not self.output_path:
            raise ValueError("--no-playback requires --output")
        self._alsa_endianness = self.alsa_endianness.upper()
        if self._alsa_endianness not in ("BE", "LE"):
            raise ValueError("alsa_endianness must be BE or LE")
        self._log_every = max(1, self.log_every)
        self._frames = 0
        self._skipped = 0
        self._first_frame_time = None
        self._dc_x_prev = [0, 0]
        self._dc_y_prev = [0, 0]
        self._alsa_pcm = None
        self._alsa_period_frames = ALSA_PERIOD_FRAMES
        self._alsa_write_bytes = (
            self._alsa_period_frames * ALSA_CHANNELS * ALSA_SAMPLE_BYTES
        )
        self._play_queue = queue.Queue(maxsize=PLAYBACK_QUEUE_CHUNKS)
        self._play_thread = None
        self._dropped = 0
        self._wav_queue = queue.Queue(maxsize=WAV_QUEUE_CHUNKS)
        self._wav_thread = None
        self._wav_writes = 0
        self._wav_dropped = 0
        self._wav_file = None
        self._alsa_error_cls = Exception
        self._alsaaudio = None
        self._alsa_fail_streak = 0
        self._alsa_writes = 0

        try:
            if self.output_path:
                self._wav_file = wave.open(self.output_path, "wb")
                self._wav_file.setnchannels(ALSA_CHANNELS)
                self._wav_file.setsampwidth(ALSA_SAMPLE_BYTES)
                self._wav_file.setframerate(self.sample_rate)
                logging.info(
                    "WAV open: %s rate=%d channels=%d sampwidth=%d",
                    self.output_path,
                    self.sample_rate,
                    ALSA_CHANNELS,
                    ALSA_SAMPLE_BYTES,
                )

            if self.no_playback:
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
                    self.alsa_device,
                    self.sample_rate,
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

    def _create_alsa_pcm(self):
        """Open the playback PCM handle."""
        if self._alsaaudio is None:
            raise RuntimeError(
                "alsaaudio is unavailable; cannot open the playback device."
            )
        fmt = (
            self._alsaaudio.PCM_FORMAT_S32_LE
            if self._alsa_endianness == "LE"
            else self._alsaaudio.PCM_FORMAT_S32_BE
        )
        return self._alsaaudio.PCM(
            self._alsaaudio.PCM_PLAYBACK,
            self._alsaaudio.PCM_NORMAL,
            device=self.alsa_device,
            channels=ALSA_CHANNELS,
            rate=self.sample_rate,
            format=fmt,
            periodsize=self._alsa_period_frames,
        )

    def _process_host_page(self, host_page, bytes_written):
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

        if not self.no_playback and self._enqueue_drop_oldest(self._play_queue, pcm):
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

    def _stop_worker(self, thread, q, timeout=2.0):
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
                self.output_path,
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

    def _playback_worker(self):
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

    def _reopen_alsa(self):
        if self._alsaaudio is None:
            return
        if self._alsa_pcm is not None:
            try:
                self._alsa_pcm.close()
            except Exception:
                pass
        try:
            self._alsa_pcm = self._create_alsa_pcm()
            logging.info("ALSA reopened: device=%s", self.alsa_device)
        except Exception as e:
            logging.error("ALSA reopen failed: %s", e)
            self._alsa_pcm = None

    def _enqueue_drop_oldest(self, q, pcm):
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

    def _pcm_for_wav(self, pcm):
        """Return PCM bytes in WAV's required little-endian layout."""
        if self._alsa_endianness == "LE":
            return pcm
        return np.frombuffer(pcm, dtype=">i4").astype("<i4", copy=False).tobytes()

    def _wav_writer(self):
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

    def _alsa_write(self, data):
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

    def compute(self, context):
        tensor = self.in_tensor.receive_data()
        metadata = self.in_metadata.receive_data()
        bytes_written = metadata.bytes_written
        # Only the payload is copied into the SDK tensor; metadata is on its own port.
        host_buf = np.asarray(tensor, dtype=np.uint8)
        copy_len = max(0, min(int(bytes_written), PCM_BYTES, host_buf.size))
        host_page = host_buf[:copy_len].copy()

        self._process_host_page(host_page, bytes_written)


def _cuda_check(result):
    err = result[0]
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        _, description = cuda_driver.cuGetErrorString(err)
        raise RuntimeError(f"CUDA error: {description}")
    if len(result) == 2:
        return result[1]
    return result[1:]


def run(
    hololink_ip,
    sensor,
    skip_reset,
    frame_size,
    timeout_ms,
    frame_limit,
    interface_override,
    alsa_device,
    sample_rate,
    alsa_endianness,
    output_path,
    no_playback,
):
    output_path = (
        output_path or ""
    )  # hs.Param can't default/bind to None; "" means "unset"

    _cuda_check(cuda_driver.cuInit(0))
    cu_device = _cuda_check(cuda_driver.cuDeviceGet(0))
    cu_context = _cuda_check(cuda_driver.cuDevicePrimaryCtxRetain(cu_device))
    hololink_dev = None
    try:
        cuda_driver.cuCtxSetCurrent(cu_context)

        metadata = hl.find_channel(hololink_ip)
        hl.use_sensor(metadata, sensor)
        channel = hl.DataChannel(metadata)
        hololink_dev = channel.hololink()
        hololink_dev.start()
        if not skip_reset:
            hololink_dev.reset()

        logging.info(
            "HSB version=%#x FPGA datecode=%#x",
            hololink_dev.get_hsb_ip_version(),
            hololink_dev.get_fpga_date(),
        )

        hololink_dev.get_i2s(metadata, enable_tx=False)
        logging.info("I2S audio path configured")

        interface = interface_override or metadata.get_string("interface")
        hsb_mac = metadata.get_string("mac_id")
        mac_addr = list(bytes.fromhex(hsb_mac.replace(":", "")))

        device = AudioRxDevice(hololink_dev, metadata)

        graph = hs.Graph("audio_recorder_fusa")
        receiver = graph.op(
            hl.operators.FusaCoeCaptureOp,
            "receiver",
            data_channel=channel,
            device_start=device.start,
            device_stop=device.stop,
            interface=interface,
            mac_addr=mac_addr,
            timeout=timeout_ms,
            cpu_output=True,
            frame_size_bytes=frame_size,
            frame_limit=frame_limit or 0,
        )
        playback = graph.op(
            AudioAlsaPlaybackOp,
            "alsa_playback",
            alsa_device=alsa_device,
            sample_rate=sample_rate,
            alsa_endianness=alsa_endianness,
            output_path=output_path,
            no_playback=no_playback,
        )
        graph.add_flow(receiver.tensor, playback.in_tensor, queue_depth=4)

        graph.add_flow(receiver.metadata, playback.in_metadata, queue_depth=4)

        clock_key = graph.add_clock(hs.RealtimeClock, "clock")
        graph.set_default_clock(clock_key)

        # receiver's "tensor" output is cpu_output=True (kHost) -- no CUDA device binding needed or
        # wanted, unlike every kCudaDevice receiver elsewhere in this porting effort.
        plan = hs.compile(graph, hs.CompileOptions())
        if not plan.ok:
            print("Graph compilation failed:", file=sys.stderr)
            for diag in plan.diagnostics:
                print(f"  {diag}", file=sys.stderr)
            return 1

        logging.info(
            "Starting: alsa=%s output=%s no_playback=%s frame_limit=%s interface=%s "
            "endianness=%s",
            alsa_device if not no_playback else "(disabled)",
            output_path or "(none)",
            no_playback,
            frame_limit or "unlimited",
            interface,
            alsa_endianness,
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
        try:
            if hololink_dev is not None:
                hololink_dev.stop()
        finally:
            _cuda_check(cuda_driver.cuDevicePrimaryCtxRelease(cu_device))


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

    logging.basicConfig(level=args.log_level)
    logging.info("Initializing.")

    sys.exit(
        run(
            args.hololink,
            args.sensor,
            args.skip_reset,
            args.frame_size,
            args.timeout,
            args.frame_limit,
            args.interface,
            args.alsa_device,
            args.sample_rate,
            args.endianness,
            args.output,
            args.no_playback,
        )
    )


if __name__ == "__main__":
    main()
