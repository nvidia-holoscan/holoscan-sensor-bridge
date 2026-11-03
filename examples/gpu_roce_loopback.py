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

# GPU RoCE Transceiver loopback test. Assumes the hardware includes a data generator IP (the
# ram_player module) for each of the NUM_SENSORS sensors, on an RFSoC (PYNQ) board. A single
# GpuRoceTransceiverOp serves the first --num-qps sensors (all of them by default), with one QP
# per sensor, in single-kernel loopback mode: a GPU-resident persistent kernel owns the entire
# data path (DOCA GPUNetIO/Verbs, CQ/QP management, doorbell rings against mlx5 CQEs/WQEs all
# happen on the GPU) and sends the received data back. Per its own C++ implementation,
# get_next_frame() always returns {} -- the inherited BaseReceiverOp::compute() therefore never
# emits a tensor or metadata sample through the operator's declared ports. The `sink` operator
# below exists only to keep those two declared ports connected, matching every other receiver's
# own downstream-wiring convention in this porting effort; it never actually receives anything
# at runtime.
#
# DataGen configures and triggers each sensor's data generator, and IlaCapture (--ila-capture)
# reads each sensor's timestamp ILA, via direct register access (write_uint32/read_uint32). Both
# have no Holoscan dependency of their own and are unchanged from this repo's previous
# (pre-HSDK5) version of this app.
#
# One operator, several sensors: tx_ibv_qp lists one remote QP per sensor (sensor N uses
# --tx-ibv-qp + 2 * N); the first sensor's DataChannel is the operator's data_channel, and the
# others' go in an hl.DataChannels as extra_data_channels, in the same order. Its device_start
# and device_stop callbacks start the DataGens in order and stop them all, as the old app's list
# of devices did.
#
# Two architectural simplifications versus the old app, both because its actual job -- keeping
# the process alive while the GPU kernel runs autonomously, and tearing the receiver down cleanly
# on exit -- is now handled by mechanisms this whole porting effort already relies on for every
# other receiver, not by bespoke per-app code:
#   - The old app's KeepAlive operator (a dummy operator ticking on a manually-armed
#     BooleanCondition, whose own stop() forwarded to the receiver's stop()) has no port here.
#     hs.run_async(plan)'s session already keeps the process alive via session.wait() regardless
#     of whether any operator has pending trigger work, and already calls every operator's own
#     stop() (here, GpuRoceTransceiverOp's inherited BaseReceiverOp::stop(), which tears down the
#     GPU monitor thread and the transceiver itself) automatically on session teardown. KeepAlive's
#     other job, running the --ila-capture captures, is IlaCaptureOp's: its first compute() runs
#     them, then stops the DataGens and the run, as KeepAlive's first tick did. The old app also
#     ran the captures on the thread that drove the DataGens, because HSB rejects a control-plane
#     transaction whose sequence number is out of order. That isn't needed for correctness:
#     Hololink takes each sequence number and completes its transaction under one lock. HSDK5
#     stops a run only after its running compute() returns, so on Ctrl+C the captures end early
#     (context.stop_requested), keeping the finished ones, which are still saved.
#   - The old app's DataGen.start()/stop() were the receiver's device callbacks, given as a list.
#     Here GpuRoceTransceiverOp's device_start/device_stop callbacks start and stop them -- the
#     same mechanism every other receiver's sensor-device object (NativeImx274Sensor,
#     NativeVb1940Sensor, ...) already uses to get its own start()/stop() called at the right
#     points in BaseReceiverOp::start()/stop(). DataGen.set_receiver_operator() is dropped as a
#     result: it only existed to give KeepAlive a handle to call receiver.stop() from.
#
# Two genuine, narrow fidelity gaps versus the old app:
#   - PTP sync uses the already-established no-arg hololink.ptp_synchronize() (30 s default,
#     matching every other app in this porting effort) instead of a custom 10 s Timeout: Timeout
#     and the Timeout-taking ptp_synchronize() overload remain unbound, and the old app only
#     logged a failure and went on.
#   - --configuration is not accepted: HSDK5 has no application configuration file to load, and
#     the old app read no key from it.
#
# GpuRoceTransceiverOp is built only with DOCA GPUNetIO, which docker/build.sh installs when the
# host that builds the image has a ConnectX NIC (it checks /sys/class/infiniband/); running this
# app also needs the RFSoC board. This port has not been run against hardware.

import argparse
import csv
import json
import logging
import statistics
import sys
import time
from datetime import timedelta
from pathlib import Path

import capture_metadata
import holoscan as hs
from cuda.bindings import driver as cuda_driver

import hololink as hl

DATA_GEN_SPACING_US = 10
NUM_SENSORS = 4
SENSOR_BANK_STRIDE = 0x0020_0000
SENSOR_TX_STRIDE = 0x0001_0000

PLAYER_BASES = tuple(
    0x5000_0000 + sensor * SENSOR_BANK_STRIDE for sensor in range(NUM_SENSORS)
)
TX_PKT_PROC_BASES = tuple(
    0x0120_0000 + sensor * SENSOR_TX_STRIDE for sensor in range(NUM_SENSORS)
)
# The board's own UUID must be registered against the four-sensor strategy;
# otherwise enumeration falls back to a two-sensor default and use_sensor(2)
# is rejected as out-of-range.
# PYNQ / RFSoC user drops: [2] timestamp ILA, [3] SIF ILA, [4] RAM player.
BOARD_UUID = "5f544c06-c28f-4883-ae4a-e3710f783a6f"
TIMESTAMP_ILA_BASE = 0x3000_0000
SIF_ILA_BASE = 0x4000_0000
TIMER_TICKS_PER_US = 322

# PYNQ / RFSoC timestamp ILA is DEPTH=16384, W_DATA=256 (8 dwords), so the sample
# RAM spans base+0x0008_0000 to base+0x0010_0000.
# RAM window = 1 << (log2(depth) + 2 + log2(words)); word stride = 1 << (log2(depth) + 2).
ILA_DEPTH = 16384
ILA_WORDS = 8
ILA_RAM_WINDOW_OFFSET = 0x0008_0000
ILA_RAM_WORD_SHIFT = 16
ILA_CAPTURE_TIMES_S = (0, 5, 10)
ILA_TIMEOUT_S = 10
ILA_OUTPUT = Path("output/gpu_roce_loopback")


def _cuda_check(result):
    err = result[0]
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        _, description = cuda_driver.cuGetErrorString(err)
        raise RuntimeError(f"CUDA error: {description}")
    if len(result) == 2:
        return result[1]
    return result[1:]


def percentile(values, percent):
    position = (len(values) - 1) * percent / 100
    lower = int(position)
    fraction = position - lower
    if lower + 1 == len(values):
        return values[lower]
    return values[lower] + (values[lower + 1] - values[lower]) * fraction


class _CaptureStopped(Exception):
    """The run is stopping: IlaCapture.run() ends early."""


class IlaCapture:
    """Single-threaded capture of the timestamp ILAs of the sensors in use.

    Every register access below runs on IlaCaptureOp's compute(); no capture
    thread is created.
    """

    def __init__(self, hololink, ila_bases):
        self._hololink = hololink
        self._ila_bases = ila_bases
        self._captures = []
        self._start = None
        self._started_sensors = set()
        self._stop_requested = lambda: False

    def _write(self, base, value):
        self._hololink.write_uint32(base, value)

    def _read(self, address):
        return self._hololink.read_uint32(address)

    def arm(self):
        for base in self._ila_bases:
            self._write(base, 0)
            self._write(base, 2)
        time.sleep(1)
        for base in self._ila_bases:
            self._write(base, 0)
            self._write(base, 1)

    def sensor_started(self, sensor_index):
        self._started_sensors.add(sensor_index)
        if len(self._started_sensors) == len(self._ila_bases):
            self._start = time.monotonic()

    def all_started(self):
        return self._start is not None

    def _check_stop(self):
        if self._stop_requested():
            raise _CaptureStopped()

    def _wait_complete(self, base, sensor_index, deadline):
        while time.monotonic() < deadline:
            self._check_stop()
            if self._read(base + 0x80) & 2:
                return
            time.sleep(0.1)
        # samples==0 means the ILA never saw a single SIF TX beat, which points
        # upstream at the player or TX path rather than at the ILA itself.
        status = self._read(base + 0x80)
        samples = self._read(base + 0x84)
        raise TimeoutError(
            f"sensor {sensor_index} ILA capture timed out at base {base:#010x} "
            f"(stat={status:#010x} busy={status & 1} done={(status >> 1) & 1} "
            f"trigger={(status >> 2) & 1} samples={samples}/{ILA_DEPTH})"
        )

    def _read_rows(self, base):
        rows = []
        for sample_index in range(ILA_DEPTH):
            if sample_index % 1024 == 0:
                self._check_stop()
            sample = 0
            for ram_index in range(ILA_WORDS):
                address = (
                    base
                    + ILA_RAM_WINDOW_OFFSET
                    + (ram_index << ILA_RAM_WORD_SHIFT)
                    + sample_index * 4
                )
                sample |= self._read(address) << (ram_index * 32)
            rows.append(
                {
                    "Sample": sample_index,
                    "ptp_sif_nsec": sample & 0xFFFF_FFFF,
                    "ptp_sif_sec": (sample >> 32) & 0xFFFF_FFFF,
                    "ptp_sync_nsec": (sample >> 64) & 0xFFFF_FFFF,
                    "ptp_sync_sec": (sample >> 96) & 0xFFFF_FFFF,
                    "frame_cnt": (sample >> 128) & 0xFFF,
                    "sof": (sample >> 140) & 1,
                    "eof": (sample >> 141) & 1,
                }
            )
        return rows

    def _save_results(self, rows, capture_index, sensor_index):
        output = (
            ILA_OUTPUT / f"capture_{capture_index:02d}" / f"sensor_{sensor_index:02d}"
        )
        output.mkdir(parents=True, exist_ok=True)
        fields = list(rows[0])
        with (output / "samples.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows(rows)

        valid = [
            row
            for row in rows
            if row["ptp_sif_sec"]
            and row["ptp_sif_nsec"]
            and row["ptp_sync_sec"]
            and row["ptp_sync_nsec"]
        ]
        latencies = [
            (row["ptp_sync_sec"] - row["ptp_sif_sec"]) * 1_000_000_000
            + row["ptp_sync_nsec"]
            - row["ptp_sif_nsec"]
            for row in valid
        ]
        with (output / "latency.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("sample", "latency_us"))
            writer.writerows(
                (row["Sample"], latency / 1000)
                for row, latency in zip(valid, latencies)
            )

        ordered = sorted(latencies)
        frame_counts = [row["frame_cnt"] for row in rows]
        frame_pass = all(
            (current - previous) % 4096 == 1
            for previous, current in zip(frame_counts, frame_counts[1:])
        )
        stats = {
            "total_samples": len(rows),
            "valid_samples": len(latencies),
            "frame_count_pass": frame_pass,
        }
        if ordered:
            stats.update(
                {
                    "mean_latency_us": statistics.fmean(ordered) / 1000,
                    "median_latency_us": statistics.median(ordered) / 1000,
                    "std_latency_us": statistics.pstdev(ordered) / 1000,
                    "min_latency_us": ordered[0] / 1000,
                    "max_latency_us": ordered[-1] / 1000,
                    "p95_latency_us": percentile(ordered, 95) / 1000,
                    "p99_latency_us": percentile(ordered, 99) / 1000,
                }
            )
        with (output / "stats.json").open("w") as stream:
            json.dump(stats, stream, indent=2)
            stream.write("\n")
        logging.info(
            "ILA capture %d sensor %d results: %s",
            capture_index,
            sensor_index,
            stats,
        )

    def run(self, stop_requested=lambda: False):
        """Take every capture in sequence; the first is pre-armed. Once stop_requested() returns
        True, end early, keeping the captures already finished: HSDK5 stops the run only after
        the compute() running this returns."""
        if self._start is None:
            raise RuntimeError("not all sensor traffic generators started")
        self._stop_requested = stop_requested
        try:
            for index, capture_time in enumerate(ILA_CAPTURE_TIMES_S):
                # Wait for the capture time in steps, so that a stop ends the wait.
                while (delay := self._start + capture_time - time.monotonic()) > 0:
                    self._check_stop()
                    time.sleep(min(delay, 0.1))
                if index:
                    self.arm()
                # One deadline for the whole window, so a dead sensor doesn't
                # multiply the wait by the number of sensors.
                deadline = time.monotonic() + ILA_TIMEOUT_S
                completed = set()
                for sensor_index, base in enumerate(self._ila_bases):
                    try:
                        self._wait_complete(base, sensor_index, deadline)
                        completed.add(sensor_index)
                    except TimeoutError as e:
                        logging.error("%s; keeping the sensors that finished.", e)
                for base in self._ila_bases:
                    self._write(base, 0)
                self._captures.append(
                    [
                        self._read_rows(base) if sensor_index in completed else None
                        for sensor_index, base in enumerate(self._ila_bases)
                    ]
                )
        except _CaptureStopped:
            logging.info(
                "ILA capture stopped early; keeping %d finished captures.",
                len(self._captures),
            )
        finally:
            self.stop()

    def stop(self):
        for base in self._ila_bases:
            self._write(base, 0)

    def save(self):
        for capture_index, sensor_rows in enumerate(self._captures):
            for sensor_index, rows in enumerate(sensor_rows):
                if rows is None:
                    continue
                self._save_results(rows, capture_index, sensor_index)


class DataGen:
    PLAYER_TIMER_OFFSET = 0x0008
    PLAYER_WINDOW_SIZE_OFFSET = 0x000C
    PLAYER_WINDOW_NUMBER_OFFSET = 0x0010
    PLAYER_ENABLE_OFFSET = 0x0004
    PLAYER_PAYLOAD_RAM_OFFSET = 0x0010_0000
    # Metadata packet control lives in the sensor's virtual port block
    # (vp_address = 0x1000 + 0x40 * sensor), not in the TX packetizer block.
    METADATA_PACKET_OFFSET = 0x2C

    def __init__(
        self,
        hololink,
        rate_mbps,
        sensor_index,
        player_base,
        tx_pkt_proc_base,
        vp_base,
        timer_ticks_per_us,
        data_gen_mode=1,
        ila=None,
    ):
        self._hololink = hololink
        self.rate_mbps = rate_mbps
        self._sensor_index = sensor_index
        self._player_base = player_base
        self._tx_pkt_proc_base = tx_pkt_proc_base
        self._vp_base = vp_base
        self._timer_ticks_per_us = timer_ticks_per_us
        self._data_gen_mode = data_gen_mode
        self._frame_count = 0
        self._stopping = False
        self._ila = ila

    def _write(self, address, value):
        return self._hololink.write_uint32(address, value)

    def _read(self, address):
        return self._hololink.read_uint32(address)

    def _enable_sif_tx(self):
        self._write(self._tx_pkt_proc_base, 0x00000005)

    def _configure_player(self):
        if self.bytes_per_window < 64:
            self._write(
                self._player_base + self.PLAYER_WINDOW_SIZE_OFFSET, 64
            )  # Window Size
        else:
            self._write(
                self._player_base + self.PLAYER_WINDOW_SIZE_OFFSET,
                self.bytes_per_window,
            )  # Window Size
        self._write(
            self._player_base + self.PLAYER_WINDOW_NUMBER_OFFSET,
            self.window_number,
        )  # Window Number
        self._write(
            self._player_base + self.PLAYER_TIMER_OFFSET,
            round(self._timer_ticks_per_us * DATA_GEN_SPACING_US),
        )  # data_gen     , TMR Value

    def _enable_loopback_test(self):
        self.window_number = 4096
        self._configure_player()
        self._write(
            self._player_base + self.PLAYER_ENABLE_OFFSET, 0x00000003
        )  # data_gen     , Enable

    def _disable_data_gen(self):
        self._write(
            self._player_base + self.PLAYER_ENABLE_OFFSET, 0x00000000
        )  # data_gen     , Enable

    def _disable_metadata_packet(self):
        address = self._vp_base + self.METADATA_PACKET_OFFSET
        val = self._read(address)
        self._write(address, val | (1 << 16))  # Disable Metadata Packet via RMW

    def _stream_test(self):
        # Enable the data gen based on test mode
        self._disable_data_gen()
        self._disable_metadata_packet()
        self._enable_sif_tx()
        self._enable_loopback_test()

    def start(self):
        # Called from GpuRoceTransceiverOp's own start() (through its device_start callback),
        # after the receiver is constructed and listening.
        # Start Packet Gen after system is in working condition
        time.sleep(1)
        self._stream_test()
        if self._ila is not None:
            self._ila.sensor_started(self._sensor_index)

    def stop(self):
        # Called from GpuRoceTransceiverOp's own stop() (device-stop callback), before the
        # receiver itself tears down, and by IlaCaptureOp after the captures. Guard against
        # multiple calls during shutdown.
        if self._stopping:
            return
        self._stopping = True
        self._disable_data_gen()


class SinkOp(hs.Operator):
    """Discards GpuRoceTransceiverOp's declared tensor/metadata outputs.

    The receiver's own compute() never actually emits through them (see module docstring) --
    this operator exists only so neither declared output port is left unconnected.
    """

    in_tensor = hs.Input("in_tensor", hs.payload.Tensor)
    in_metadata = hs.Input("in_metadata", capture_metadata.CaptureMetadata)

    def setup(self, spec):
        spec.input(self.in_tensor)
        spec.input(self.in_metadata)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.in_tensor))

    def compute(self, context):
        self.in_tensor.receive()
        self.in_metadata.receive()


class _IlaShared:
    """What IlaCaptureOp works on, set by run() before the graph is compiled, and whether the
    captures failed."""

    ila = None
    data_gens = []
    failed = False


class IlaCaptureOp(hs.Operator):
    """Runs the --ila-capture captures once the DataGens have started, then stops the DataGens
    and the run."""

    def setup(self, spec):
        pass

    def contract(self):
        return hs.Contract().trigger(hs.OnClock(period=timedelta(milliseconds=100)))

    def start(self):
        self._done = False

    def compute(self, context):
        # Every operator, so every DataGen, starts before the first compute(); IlaCapture.run()
        # needs them all started, so wait for that anyway.
        if self._done or not _IlaShared.ila.all_started():
            return
        self._done = True
        try:
            _IlaShared.ila.run(stop_requested=lambda: context.stop_requested)
        except Exception as e:
            logging.error(f"ILA capture failed: {e}")
            _IlaShared.failed = True
        finally:
            for data_gen in _IlaShared.data_gens:
                data_gen.stop()
            context.request_stop()


def run(
    hololink_ip,
    rate_mbps,
    frame_size,
    rx_ibv_name,
    tx_ibv_name,
    ibv_port,
    tx_ibv_qp,
    num_qps,
    gpu_id,
    forward,
    mtu,
    ila_capture,
):
    del tx_ibv_name  # Parsed for CLI compatibility; unused by GpuRoceTransceiverOp.

    cuda_device_ordinal = 0
    _cuda_check(cuda_driver.cuInit(0))
    cu_device = _cuda_check(cuda_driver.cuDeviceGet(cuda_device_ordinal))
    cu_context = _cuda_check(cuda_driver.cuDevicePrimaryCtxRetain(cu_device))
    cuda_driver.cuCtxSetCurrent(cu_context)

    # This board enumerates with a UUID whose topology Enumerator doesn't know by default -- it's
    # a data-generator board, not a camera, with its own sensor/dataplane/SIF counts. "We're
    # special" per the old app's own comment.
    uuid_strategy = hl.BasicEnumerationStrategy(
        hl.Metadata(),
        total_sensors=NUM_SENSORS,
        total_dataplanes=1,
        sifs_per_sensor=1,
    )
    uuid = "889b7ce3-65a5-4247-8b05-4ff1904c3359"
    hl.set_uuid_strategy(uuid, uuid_strategy)
    hl.set_uuid_strategy(BOARD_UUID, uuid_strategy)

    print(f"Enumerating HSB board at {hololink_ip}...")
    channel_metadata = hl.find_channel(hololink_ip)
    # A board whose UUID wasn't registered above enumerates with the default
    # sensor count, so fail loudly here instead of deep inside use_sensor.
    discovered_uuid = channel_metadata["fpga_uuid"]
    logging.info(f"fpga_uuid={discovered_uuid}")
    if discovered_uuid != BOARD_UUID:
        logging.warning(
            "fpga_uuid=%s does not match BOARD_UUID=%s; "
            "check the board selection at the top of this file.",
            discovered_uuid,
            BOARD_UUID,
        )
        hl.set_uuid_strategy(discovered_uuid, uuid_strategy)
        channel_metadata = hl.find_channel(hololink_ip)

    channels = []
    vp_bases = []
    for sensor_index in range(num_qps):
        sensor_metadata = hl.Metadata(channel_metadata)
        hl.use_sensor(sensor_metadata, sensor_index)
        hl.use_mtu(sensor_metadata, mtu)
        vp_bases.append(sensor_metadata["vp_address"])
        channels.append(hl.DataChannel(sensor_metadata))

    hololink_dev = channels[0].hololink()
    timestamp_ila_bases = tuple(
        TIMESTAMP_ILA_BASE + sensor * SENSOR_BANK_STRIDE for sensor in range(num_qps)
    )
    sif_ila_bases = tuple(
        SIF_ILA_BASE + sensor * SENSOR_BANK_STRIDE for sensor in range(num_qps)
    )
    logging.info(
        "player_bases=%s timestamp_ila_bases=%s sif_ila_bases=%s "
        "tx_pkt_proc_bases=%s vp_bases=%s",
        [f"{address:#010x}" for address in PLAYER_BASES[:num_qps]],
        [f"{address:#010x}" for address in timestamp_ila_bases],
        [f"{address:#010x}" for address in sif_ila_bases],
        [f"{address:#010x}" for address in TX_PKT_PROC_BASES[:num_qps]],
        [f"{address:#06x}" for address in vp_bases],
    )
    ila = IlaCapture(hololink_dev, timestamp_ila_bases) if ila_capture else None

    failed = False
    hololink_dev.start()
    try:
        hololink_dev.reset()
        logging.debug("Waiting for PTP sync.")
        if not hololink_dev.ptp_synchronize():
            logging.error("Failed to synchronize PTP; ignoring.")
        else:
            logging.debug("PTP synchronized.")

        data_gens = []
        for sensor_index in range(num_qps):
            data_gen = DataGen(
                hololink_dev,
                rate_mbps,
                sensor_index,
                PLAYER_BASES[sensor_index],
                TX_PKT_PROC_BASES[sensor_index],
                vp_bases[sensor_index],
                TIMER_TICKS_PER_US,
                data_gen_mode=0,
                ila=ila,
            )
            data_gen.bytes_per_window = frame_size
            data_gens.append(data_gen)

        # HSB demuxes SIF TX by dest_qp[7:1]-1 (see roce_ack.sv /
        # sensor_tx_pkt_proc.sv dest_info[56:49]==sensor+1). Sequential
        # QPs 2,3,4,5 collide onto sensors 0 and 1; even QPs 2,4,6,8
        # map to sensors 0..3.
        tx_ibv_qps = ",".join(
            str(tx_ibv_qp + 2 * sensor_index) for sensor_index in range(num_qps)
        )
        # A single operator serves every sensor in use (at most 4): the QP numbers in
        # tx_ibv_qps, the DataChannels and the DataGens are matched by position.
        extra_channels = {}
        if len(channels) > 1:
            extra_channels["extra_data_channels"] = hl.DataChannels(channels[1:])

        def start_data_gens():
            # A start() that fails gets no stop(): stop the DataGens already started, then
            # report the failure.
            for started, data_gen in enumerate(data_gens):
                try:
                    data_gen.start()
                except Exception:
                    for earlier in reversed(data_gens[:started]):
                        try:
                            earlier.stop()
                        except Exception:
                            pass
                    raise

        def stop_data_gens():
            # Stop every DataGen even if one fails, then report the first failure.
            failure = None
            for data_gen in data_gens:
                try:
                    data_gen.stop()
                except Exception as e:
                    failure = failure or e
            if failure is not None:
                raise failure

        graph = hs.Graph("gpu_roce_loopback")
        receiver = graph.op(
            hl.operators.GpuRoceTransceiverOp,
            "receiver",
            data_channel=channels[0],
            frame_context=int(cu_context),
            device_start=start_data_gens,
            device_stop=stop_data_gens,
            frame_size=frame_size,
            ibv_name=rx_ibv_name,
            ibv_port=ibv_port,
            tx_ibv_qp=tx_ibv_qps,
            gpu_id=gpu_id,
            forward=forward,
            **extra_channels,
        )
        sink = graph.op(SinkOp, "sink")
        graph.add_flow(receiver.tensor, sink.in_tensor)
        graph.add_flow(receiver.metadata, sink.in_metadata)

        if ila is not None:
            _IlaShared.ila = ila
            _IlaShared.data_gens = data_gens
            graph.op(IlaCaptureOp, "ila_capture")
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

        if ila is not None:
            # Every timestamp ILA in use must be armed before any player starts.
            ila.arm()
        print(
            "GPU RoCE Transceiver loopback test running -- DataGen is driving the on-board data "
            "generators directly via register access; GpuRoceTransceiverOp's GPU-resident kernel "
            "owns the data path end to end (Ctrl+C to stop)..."
        )
        with hs.run_async(plan) as session:
            try:
                session.wait()
            except KeyboardInterrupt:
                # Ctrl+C: leaving the block stops the session; the captures taken so far are
                # still saved below, as the old app did.
                pass
        if session.diagnostics:
            print(session.summary(), file=sys.stderr)
        failed = session.termination == hs.RunTermination.FAILED
    finally:
        if ila is not None:
            ila.stop()
        hololink_dev.stop()
        (cu_result,) = cuda_driver.cuDevicePrimaryCtxRelease(cu_device)
        if cu_result != cuda_driver.CUresult.CUDA_SUCCESS:
            logging.error(f"Failed to release CUDA primary context: {cu_result}")

    # Processing happens after the run so it never delays a capture window.
    if ila is not None:
        ila.save()

    if failed or _IlaShared.failed:
        return 1
    print("Done.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="GPU RoCE Transceiver loopback test")
    parser.add_argument("--rate-mbps", type=int, default=1000, help="Rate Mbps")
    parser.add_argument(
        "--frame-size",
        type=int,
        default=4096,
        help="Frame/message size in bytes for benchmarking",
    )
    parser.add_argument(
        "--hololink", default="192.168.0.2", help="IP address of Hololink board"
    )
    parser.add_argument(
        "--log-level", type=int, default=20, help="Python logging level to display"
    )

    parser.add_argument(
        "--rx-ibv-name",
        default="",
        help="IBV device name for receiver "
        "(default: the device on the board's network interface)",
    )
    parser.add_argument(
        "--tx-ibv-name", default="", help="IBV device name for transmitter (unused)"
    )
    parser.add_argument(
        "--ibv-port",
        type=int,
        default=1,
        help="Port number of IBV device (with --rx-ibv-name)",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device id")
    parser.add_argument("--forward", type=int, default=1, help="Enable forward")
    parser.add_argument(
        "--tx-ibv-qp",
        type=int,
        default=2,
        help="Transmitter QP number of sensor 0; sensor N uses this value + 2 * N",
    )
    parser.add_argument(
        "--num-qps",
        type=int,
        default=NUM_SENSORS,
        choices=range(1, NUM_SENSORS + 1),
        help="Number of QPs to create, one per sensor starting at sensor 0",
    )
    parser.add_argument(
        "--mtu",
        type=int,
        default=1472,
        help="Maximum Transmission Unit (MTU) size in bytes. Default: 1472 (standard), use 9000+ for jumbo frames",
    )
    parser.add_argument(
        "--ila-capture",
        action="store_true",
        help="Capture, analyze, and stop automatically during the loopback run",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    logging.info("Initializing.")

    return run(
        args.hololink,
        args.rate_mbps,
        args.frame_size,
        args.rx_ibv_name,
        args.tx_ibv_name,
        args.ibv_port,
        args.tx_ibv_qp,
        args.num_qps,
        args.gpu_id,
        args.forward,
        args.mtu,
        args.ila_capture,
    )


if __name__ == "__main__":
    sys.exit(main())
