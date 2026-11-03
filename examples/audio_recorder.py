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
Hololink live audio capture: I2S RX -> RoCE receive -> ALSA playback (optional WAV).

Reads two-channel S32 audio streamed by the Hololink FPGA over RoCE, applies a
DC-blocking IIR filter per channel, and plays the result through ALSA. With
``--output`` the same decoded PCM is also written to a stereo WAV file.

Both I2S slots are forwarded as a stereo pair (slot 0 -> L, slot 1 -> R). When
the board's DMIC drives only one slot, the other slot carries whatever is on
the bus and will appear in the corresponding speaker channel.

RoceReceiverOp is already HSDK5-native; this port's only new
C++ work was exposing hololink.get_i2s() to Python (unbound before this round, confirmed
by grepping _hololink.cpp). AudioAlsaPlaybackOp below is the old app's own pure-Python
operator (no GXF/C++ dependency), ported mechanically to hs.Operator's declarative port/
contract() style -- its ALSA/WAV-writer-thread/DC-blocker logic is unchanged.

Unused receiver metadata is left unconnected. AudioAlsaPlaybackOp keeps its own
frame counter for periodic diagnostics.
"""

import argparse
import logging
import os
import queue
import sys
import threading
import wave

import cuda.bindings.driver as cuda_driver
import cupy as cp
import holoscan as hs
import numpy as np

import hololink as hl

DEFAULT_FRAME_SIZE = 4096
DEFAULT_PAGES = 4
DEFAULT_QUEUE_SIZE = 3

# Trimmed RoCE audio payload: 32 stereo S32 frames per completion.
ROCE_PCM_BYTES = 256
ROCE_SAMPLES_PER_FRAME = ROCE_PCM_BYTES // 4 // 2

# Keepalive / control frames have this length; the decoder drops them.
ROCE_NON_AUDIO_LEN = 164

ALSA_CHANNELS = 2
ALSA_SAMPLE_BYTES = 4
# ALSA playback period in frames (~11 ms at 48 kHz). Too small a period starves
# low-latency sinks (e.g. the analog headphone jack) into underrun noise.
ALSA_PERIOD_FRAMES = 512
DEFAULT_SAMPLE_RATE = 48000

# Playback queue depth (receiver thread -> ALSA writer thread). 256 buffers is
# ~170 ms; bounds live latency. On overflow the oldest buffer is dropped.
PLAYBACK_QUEUE_CHUNKS = 256

# WAV queue depth (receiver thread -> WAV writer thread). Larger than the
# playback queue (~2.7 s) to absorb disk write bursts before dropping a frame.
WAV_QUEUE_CHUNKS = 4096

# Rewrite the WAV header every ~1 s while streaming so the file stays playable
# if the process is killed before close() finalizes it.
WAV_HEADER_REFRESH_FRAMES = DEFAULT_SAMPLE_RATE // ROCE_SAMPLES_PER_FRAME


def _i32_wrap(v: int) -> int:
    """Wrap a Python int into the signed 32-bit range (matches C int32_t overflow)."""
    v = int(v)
    return ((v + 0x80000000) & 0xFFFFFFFF) - 0x80000000


def _default_alsa_device() -> str:
    """Pick a sensible ALSA device name based on the runtime environment.

    Honors an explicit ``ALSA_PCM_DEVICE`` override, falls back to PulseAudio
    when a Pulse server is exported (typical inside the demo container), and
    otherwise uses the host's default ALSA sink.
    """
    if os.environ.get("ALSA_PCM_DEVICE"):
        return os.environ["ALSA_PCM_DEVICE"]
    if os.environ.get("PULSE_SERVER"):
        return "pulse"
    return "default"


class AudioRxDevice:
    """Minimal device handle for ``RoceReceiverOp``; no per-frame state needed."""

    def __init__(self, hololink, enumeration_metadata):
        self._hololink = hololink
        self._enumeration_metadata = enumeration_metadata

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class AudioAlsaPlaybackOp(hs.Operator):
    """Decode RoCE-delivered I2S audio and forward it to ALSA and/or a WAV file.

    Each RoCE completion carries a 256 B trimmed buffer of interleaved stereo
    S32 samples (ch0=L, ch1=R, big-endian on the wire). A single-pole DC
    blocker is applied independently to each channel, then the decoded PCM is
    enqueued for ALSA playback and, when ``output_path`` is set, appended to
    the output WAV file.
    """

    in_tensor = hs.Input("in_tensor", hs.payload.Tensor)

    alsa_device = hs.Param("default")
    sample_rate = hs.Param(DEFAULT_SAMPLE_RATE)
    alsa_endianness = hs.Param("BE")
    output_path = hs.Param("")
    no_playback = hs.Param(False)
    log_every = hs.Param(128)

    def setup(self, spec):
        spec.input(self.in_tensor, queue_depth=4)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.in_tensor))

    def start(self):
        if self.no_playback and not self.output_path:
            raise ValueError("--no-playback requires --output")
        self._alsa_endianness = self.alsa_endianness.upper()
        if self._alsa_endianness not in ("BE", "LE"):
            raise ValueError("alsa_endianness must be BE or LE")
        self._log_every = max(1, self.log_every)
        self._frames = 0
        # Per-channel DC-blocker IIR state carried across frames: previous input
        # x[n-1] and previous output y[n-1] for [left, right]; start at rest (0).
        self._dc_x_prev = [0, 0]
        self._dc_y_prev = [0, 0]
        self._alsa_pcm = None
        # Drain to ALSA in full periods so the sink buffer never starves.
        self._alsa_write_bytes = ALSA_PERIOD_FRAMES * ALSA_CHANNELS * ALSA_SAMPLE_BYTES
        # Playback runs on its own thread; a blocking ALSA write on the receiver
        # thread would stall intake and drop frames.
        self._play_queue = queue.Queue(maxsize=PLAYBACK_QUEUE_CHUNKS)
        self._play_thread = None
        self._dropped = 0
        # WAV writing also runs on its own thread so disk stalls don't drop frames.
        self._wav_queue = queue.Queue(maxsize=WAV_QUEUE_CHUNKS)
        self._wav_thread = None
        self._wav_writes = 0
        self._wav_dropped = 0
        self._wav_file = None
        self._alsa_error_cls = Exception

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
                self._alsa_error_cls = alsaaudio.ALSAAudioError

                self._alsa_pcm = alsaaudio.PCM(
                    alsaaudio.PCM_PLAYBACK,
                    alsaaudio.PCM_NORMAL,
                    device=self.alsa_device,
                )
                self._alsa_pcm.setchannels(ALSA_CHANNELS)
                self._alsa_pcm.setrate(self.sample_rate)
                fmt = (
                    alsaaudio.PCM_FORMAT_S32_LE
                    if self._alsa_endianness == "LE"
                    else alsaaudio.PCM_FORMAT_S32_BE
                )
                self._alsa_pcm.setformat(fmt)
                self._alsa_pcm.setperiodsize(ALSA_PERIOD_FRAMES)
                logging.info(
                    "ALSA open: device=%s rate=%d period_frames=%d (%d B) stereo S32 %s",
                    self.alsa_device,
                    self.sample_rate,
                    ALSA_PERIOD_FRAMES,
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

        # Start workers only after setup succeeds, so a failure leaves no orphan thread.
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

    def _stop_worker(self, thread, q, timeout=2.0):
        # Deliver the sentinel so the worker exits cleanly, then wait for it.
        # timeout=None blocks until both (full drain, no drop); a finite timeout
        # bounds shutdown.
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
            samples = self._wav_writes * ROCE_SAMPLES_PER_FRAME
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
        logging.info(
            "Audio sink stopped after %d RoCE frames (%d dropped at playback queue)",
            self._frames,
            self._dropped,
        )

    def _roce_i2s_to_alsa_pcm(self, raw):
        """Decode one RoCE buffer to S32 PCM, applying a per-channel DC blocker.

        The DC blocker is a single-pole IIR with cutoff ~12 Hz at 48 kHz:

            y[n] = x[n] - x[n-1] + y[n-1] - (y[n-1] >> 8)

        Returns ``None`` for non-audio control frames and for buffers whose
        length is not a multiple of the stereo S32 frame size.
        """
        n = len(raw)
        if n == ROCE_NON_AUDIO_LEN:
            return None
        if n % (ALSA_CHANNELS * ALSA_SAMPLE_BYTES) != 0:
            logging.warning(
                "skip frame: length %d not multiple of %d",
                n,
                ALSA_CHANNELS * ALSA_SAMPLE_BYTES,
            )
            return None

        # Decode BE S32 with NumPy; IIR is recursive so the inner step stays in Python.
        x_arr = np.frombuffer(raw, dtype=">i4").reshape(-1, ALSA_CHANNELS)
        y_arr = np.empty_like(x_arr)
        for ch in range(ALSA_CHANNELS):
            x_prev = self._dc_x_prev[ch]
            y_prev = self._dc_y_prev[ch]
            yc = [0] * x_arr.shape[0]
            for i, cur in enumerate(x_arr[:, ch].tolist()):
                yi = _i32_wrap(cur - x_prev + y_prev - (y_prev >> 8))
                yc[i] = yi
                x_prev = cur
                y_prev = yi
            y_arr[:, ch] = yc
            self._dc_x_prev[ch] = x_prev
            self._dc_y_prev[ch] = y_prev

        target_dtype = "<i4" if self._alsa_endianness == "LE" else ">i4"
        return y_arr.astype(target_dtype, copy=False).tobytes()

    def _playback_worker(self):
        """Drain decoded PCM to ALSA in full periods on a dedicated thread.

        The blocking write() paces output without stalling the receiver; exits on
        the None sentinel.
        """
        pending = bytearray()
        while True:
            item = self._play_queue.get()
            if item is None:
                break
            try:
                pending += item
                while len(pending) >= self._alsa_write_bytes:
                    self._alsa_write(bytes(pending[: self._alsa_write_bytes]))
                    del pending[: self._alsa_write_bytes]
            except Exception as e:
                logging.warning("ALSA playback: %s", e)

    def _enqueue_drop_oldest(self, q, pcm):
        """Put pcm on q without blocking, dropping the oldest item if it is full.

        Returns True if a drop occurred. Keeps a slow consumer from blocking the
        receiver.
        """
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
        """Drain decoded PCM to the WAV file on a dedicated thread.

        Keeps the byteswap and disk I/O off the receiver; exits on the None
        sentinel.
        """
        while True:
            item = self._wav_queue.get()
            if item is None:
                break
            try:
                # writeframesraw appends without rewriting the header, which would
                # otherwise seek and flush on every frame.
                self._wav_file.writeframesraw(self._pcm_for_wav(item))
                self._wav_writes += 1
                # Patch the header periodically so a killed process still leaves a
                # playable file.
                if self._wav_writes % WAV_HEADER_REFRESH_FRAMES == 0:
                    self._wav_file.writeframes(b"")
            except Exception as e:  # keep the writer alive on I/O errors
                logging.warning("WAV write: %s", e)

    def _alsa_write(self, data):
        if self._alsa_pcm is None:
            return
        try:
            self._alsa_pcm.write(data)
        except self._alsa_error_cls as e:
            logging.warning("ALSA write: %s; drop and retry", e)
            try:
                self._alsa_pcm.drop()
                self._alsa_pcm.write(data)
            except self._alsa_error_cls as e2:
                logging.error("ALSA retry failed: %s", e2)

    def compute(self, context):
        tensor = self.in_tensor.receive_data()
        # RoCE publishes a CUDA tensor, already trimmed to bytes_written. Export
        # versioned DLPack explicitly; the legacy array interface is unsupported.
        with cp.cuda.Stream(non_blocking=True) as stream:
            raw = cp.from_dlpack(
                tensor.__dlpack__(stream=stream.ptr, max_version=(1, 0), copy=False)
            ).get()
        pcm = self._roce_i2s_to_alsa_pcm(raw)
        if pcm is None:
            return
        if self._frames == 0 and len(raw) != ROCE_PCM_BYTES:
            logging.warning(
                "first RoCE buffer %d B (expected %d for I2S trim); decode may be wrong",
                len(raw),
                ROCE_PCM_BYTES,
            )
        # Hand off to the worker threads (non-blocking); playback first.
        if not self.no_playback and self._enqueue_drop_oldest(self._play_queue, pcm):
            self._dropped += 1
        if self._wav_file is not None and self._enqueue_drop_oldest(
            self._wav_queue, pcm
        ):
            self._wav_dropped += 1
        self._frames += 1
        if self._frames % self._log_every == 0:
            logging.debug(
                "roce_frame=%d raw=%d B play_q=%d play_drop=%d wav_q=%d wav_drop=%d",
                self._frames,
                len(raw),
                self._play_queue.qsize(),
                self._dropped,
                self._wav_queue.qsize(),
                self._wav_dropped,
            )


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
    pages,
    queue_size,
    frame_limit,
    ibv_name,
    ibv_port,
    alsa_device,
    sample_rate,
    alsa_endianness,
    output_path,
    no_playback,
):
    if queue_size > pages:
        raise ValueError("queue_size must be <= pages")
    output_path = (
        output_path or ""
    )  # hs.Param can't default/bind to None; "" means "unset"

    cuda_device_ordinal = 0
    _cuda_check(cuda_driver.cuInit(0))
    cu_device = _cuda_check(cuda_driver.cuDeviceGet(cuda_device_ordinal))
    cu_context = _cuda_check(cuda_driver.cuDevicePrimaryCtxRetain(cu_device))
    hololink_dev = None
    try:
        _cuda_check(cuda_driver.cuCtxSetCurrent(cu_context))

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

        # Bring up I2S for RX-only capture; RoCE delivers samples.
        hololink_dev.get_i2s(metadata, enable_tx=False)
        logging.info("I2S audio path configured")

        device = AudioRxDevice(hololink_dev, metadata)

        graph = hs.Graph("audio_recorder")
        receiver = graph.op(
            hl.operators.RoceReceiverOp,
            "receiver",
            data_channel=channel,
            frame_context=int(cu_context),
            device_start=device.start,
            device_stop=device.stop,
            frame_size=frame_size,
            ibv_name=ibv_name,
            ibv_port=ibv_port,
            pages=pages,
            queue_size=queue_size,
            trim=True,
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

        clock_key = graph.add_clock(hs.RealtimeClock, "clock")
        graph.set_default_clock(clock_key)

        options = hs.CompileOptions()
        options.deployment.bind_tensor_output_device(
            operator_path="receiver",
            output_port="tensor",
            device=hs.execution.DeviceId(cuda_device_ordinal),
        )

        plan = hs.compile(graph, options)
        if not plan.ok:
            print("Graph compilation failed:", file=sys.stderr)
            for diag in plan.diagnostics:
                print(f"  {diag}", file=sys.stderr)
            return 1

        logging.info(
            "Starting: alsa=%s output=%s no_playback=%s frame_limit=%s",
            alsa_device if not no_playback else "(disabled)",
            output_path or "(none)",
            no_playback,
            frame_limit or "unlimited",
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
        description="Hololink I2S audio -> RoCE -> ALSA live playback (optional WAV).",
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
        help="RoCE receive frame size in bytes",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=DEFAULT_PAGES,
        help="Number of receive page buffers",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=DEFAULT_QUEUE_SIZE,
        help="Receive queue depth (must be <= pages)",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Exit after receiving this many frames",
    )
    parser.add_argument(
        "--ibv-name",
        default="",
        help="IBV device to use (default: the device on the board's network interface)",
    )
    parser.add_argument(
        "--ibv-port",
        type=int,
        default=1,
        help="Port number of IBV device (with --ibv-name)",
    )
    parser.add_argument(
        "--alsa-device",
        default=_default_alsa_device(),
        help="ALSA PCM device (default: pulse in demo container, else default)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        choices=[DEFAULT_SAMPLE_RATE],
        default=DEFAULT_SAMPLE_RATE,
        help="Capture sample rate (Hz). Only 48000 is supported today; the I2S "
        "capture path and ALSA period sizing are hardcoded to 48 kHz.",
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
            args.pages,
            args.queue_size,
            args.frame_limit,
            args.ibv_name,
            args.ibv_port,
            args.alsa_device,
            args.sample_rate,
            args.endianness,
            args.output,
            args.no_playback,
        )
    )


if __name__ == "__main__":
    main()
