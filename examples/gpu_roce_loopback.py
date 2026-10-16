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

# This example assumes the hardware includes a data generator IP (the ram_player module)
# for each of the NUM_SENSORS sensors. It instantiates a single GPU RoCE Transceiver
# operator for the first --num-qps sensors (all of them by default), with one QP per
# sensor, and configures it to run in single-kernel loopback mode, which sends received
# data back to the output port. The example also configures and triggers the data
# generators (via DataGen) to start receiving packets. Since the GPU RoCE Transceiver
# operator runs on the GPU, a KeepAlive operator is used to receive a termination signal
# from the user and forward it to the transceiver operator.

import argparse
import csv
import json
import logging
import os
import statistics
import time
from pathlib import Path

import cuda.bindings.driver as cuda
import holoscan

import hololink as hololink_module

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


def percentile(values, percent):
    position = (len(values) - 1) * percent / 100
    lower = int(position)
    fraction = position - lower
    if lower + 1 == len(values):
        return values[lower]
    return values[lower] + (values[lower + 1] - values[lower]) * fraction


class IlaCapture:
    """Single-threaded capture of the timestamp ILAs of the sensors in use.

    HSB rejects an ECB transaction whose sequence number is out of order, so
    every register access below runs on the Holoscan scheduler thread that
    also drives DataGen; no capture thread is created.
    """

    def __init__(self, hololink, ila_bases):
        self._hololink = hololink
        self._ila_bases = ila_bases
        self._captures = []
        self._start = None
        self._started_sensors = set()

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

    def _wait_complete(self, base, sensor_index, deadline):
        while time.monotonic() < deadline:
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

    def run(self):
        """Take every capture in sequence; the first is pre-armed."""
        if self._start is None:
            raise RuntimeError("not all sensor traffic generators started")
        try:
            for index, capture_time in enumerate(ILA_CAPTURE_TIMES_S):
                delay = self._start + capture_time - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
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
        # Start Packet Gen after system is in working condition
        time.sleep(1)
        self._stream_test()
        if self._ila is not None:
            self._ila.sensor_started(self._sensor_index)

    def set_receiver_operator(self, receiver_op):
        """Store reference to receiver operator for shutdown coordination"""
        self._receiver_operator = receiver_op

    def stop(self):
        # Guard against multiple calls during shutdown
        if self._stopping:
            return
        self._stopping = True
        self._disable_data_gen()


class SinkOperator(holoscan.core.Operator):
    def setup(self, spec):
        spec.input("input")

    def compute(self, op_input, op_output, context):
        _ = op_input.receive("input")
        return None


class KeepAlive(holoscan.core.Operator):
    """Dummy operator that ticks periodically to keep the scheduler active"""

    def __init__(
        self,
        fragment,
        *args,
        receiver_op=None,
        ila=None,
        tick_condition=None,
        data_gens=None,
        **kwargs,
    ):
        self._receiver_op = receiver_op
        self._ila = ila
        self._tick_condition = tick_condition
        self._data_gens = data_gens or []
        self._ila_done = False
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec):
        pass

    def compute(self, op_input, op_output, context):
        # Captures run here so that ILA and DataGen register access share this
        # thread; the GPU receiver only touches the control plane at start/stop.
        if self._ila is None or self._ila_done:
            return None
        self._ila_done = True
        try:
            self._ila.run()
        except Exception as e:
            logging.error(f"ILA capture failed: {e}")
        finally:
            for data_gen in self._data_gens:
                data_gen.stop()
            self._tick_condition.disable_tick()
        return None

    def stop(self):
        if self._receiver_op is not None:
            try:
                self._receiver_op.stop()
            except Exception as e:
                logging.error(f"KeepAlive failed to stop receiver: {e}")
        super().stop()


class HoloscanApplication(holoscan.core.Application):
    def __init__(
        self,
        cuda_context,
        cuda_device_ordinal,
        hololink_channels,
        hololink,
        rate_mbps,
        hololink_ip,
        rx_ibv_name,
        tx_ibv_name,
        tx_ibv_qp,
        ibv_port,
        frame_size,
        mtu=1472,
        gpu_id=0,
        forward=1,
        ila=None,
        timer_ticks_per_us=201,
        vp_bases=None,
    ):
        logging.info("__init__")
        super().__init__()
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._hololink_channels = hololink_channels
        self._hololink = hololink
        self._rate_mbps = rate_mbps
        self._hololink_ip = hololink_ip
        self._rx_ibv_name = rx_ibv_name
        self._tx_ibv_name = tx_ibv_name
        self._tx_ibv_qp = tx_ibv_qp
        self._ibv_port = ibv_port
        self._frame_size = frame_size
        self._mtu = mtu
        self._gpu_id = gpu_id
        self._forward = forward
        self._ila = ila
        self._timer_ticks_per_us = timer_ticks_per_us
        self._vp_bases = vp_bases
        #
        self.metadata_policy = holoscan.core.MetadataPolicy.REJECT

    def compose(self):
        logging.info("compose")
        frame_size = self._frame_size
        frame_context = self._cuda_context

        sensor_indexes = range(len(self._hololink_channels))
        self._data_gens = []
        for sensor_index in sensor_indexes:
            data_gen = DataGen(
                self._hololink,
                self._rate_mbps,
                sensor_index,
                PLAYER_BASES[sensor_index],
                TX_PKT_PROC_BASES[sensor_index],
                self._vp_bases[sensor_index],
                self._timer_ticks_per_us,
                data_gen_mode=0,
                ila=self._ila,
            )
            data_gen.bytes_per_window = frame_size
            self._data_gens.append(data_gen)

        # HSB demuxes SIF TX by dest_qp[7:1]-1 (see roce_ack.sv /
        # sensor_tx_pkt_proc.sv dest_info[56:49]==sensor+1). Sequential
        # QPs 2,3,4,5 collide onto sensors 0 and 1; even QPs 2,4,6,8
        # map to sensors 0..3.
        tx_ibv_qp = ",".join(
            str(self._tx_ibv_qp + 2 * sensor_index) for sensor_index in sensor_indexes
        )
        # A single operator serves every sensor in use (at most 4): the QP numbers in
        # tx_ibv_qp, the DataChannels and the DataGens are matched by position.
        self._receiver_operator = hololink_module.operators.GpuRoceTransceiverOp(
            self,
            name="Gpu Roce Transceiver",
            frame_size=frame_size,
            frame_context=frame_context,
            ibv_name=self._rx_ibv_name,
            tx_ibv_qp=tx_ibv_qp,
            ibv_port=self._ibv_port,
            hololink_channel=self._hololink_channels,
            device=self._data_gens,
            gpu_id=self._gpu_id,
            forward=self._forward,
        )
        for data_gen in self._data_gens:
            data_gen.set_receiver_operator(self._receiver_operator)
        sink_operator = SinkOperator(self, name="sink")
        self.add_flow(self._receiver_operator, sink_operator, {("output", "input")})

        # Keepalive ticker to keep the app running while GPU receiver runs autonomously
        # Pass receiver_op so KeepAlive can stop it when KeepAlive.stop() is called
        self._keepalive_condition = holoscan.conditions.BooleanCondition(
            self, name="keepalive_tick", enable_tick=True
        )
        keepalive = KeepAlive(
            self,
            self._keepalive_condition,
            receiver_op=self._receiver_operator,
            ila=self._ila,
            tick_condition=self._keepalive_condition,
            data_gens=self._data_gens,
            name="keepalive",
        )
        self.add_operator(keepalive)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rate-mbps",
        type=int,
        default=1000,
        help="Rate Mbps",
    )
    parser.add_argument(
        "--frame-size",
        type=int,
        default=4096,
        help="Frame/message size in bytes for benchmarking",
    )

    default_configuration = os.path.join(
        os.path.dirname(__file__), "example_configuration.yaml"
    )
    parser.add_argument(
        "--configuration",
        default=default_configuration,
        help="Configuration file",
    )
    parser.add_argument(
        "--hololink",
        default="192.168.0.2",
        help="IP address of Hololink board",
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level to display",
    )
    try:
        infiniband_devices = hololink_module.infiniband_devices()
    except (FileNotFoundError, Exception):
        infiniband_devices = []

    rx_default_ibv = infiniband_devices[0] if infiniband_devices else "roceP5p3s0f0"
    tx_default_ibv = (
        infiniband_devices[1]
        if len(infiniband_devices) > 1
        else infiniband_devices[0] if infiniband_devices else "roceP5p3s0f0"
    )

    parser.add_argument(
        "--rx-ibv-name",
        default=rx_default_ibv,
        help="IBV device name for receiver",
    )
    parser.add_argument(
        "--tx-ibv-name",
        default=tx_default_ibv,
        help="IBV device name for transmitter",
    )
    parser.add_argument(
        "--ibv-port",
        type=int,
        default=1,
        help="Port number of IBV device",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU device id",
    )
    parser.add_argument(
        "--forward",
        type=int,
        default=1,
        help="Enable forward",
    )
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
    hololink_module.logging_level(args.log_level)
    logging.info("Initializing.")
    # Get a handle to the GPU
    (cu_result,) = cuda.cuInit(0)
    if cu_result != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA initialization failed with error: {cu_result}")
    cu_device_ordinal = 0
    cu_result, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
    if cu_result != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to get CUDA device: {cu_result}")
    cu_result, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
    if cu_result != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to retain CUDA primary context: {cu_result}")

    # We're special.
    uuid_strategy = hololink_module.BasicEnumerationStrategy(
        total_sensors=NUM_SENSORS,
        total_dataplanes=1,
        sifs_per_sensor=1,
    )
    uuid = "889b7ce3-65a5-4247-8b05-4ff1904c3359"
    hololink_module.Enumerator.set_uuid_strategy(uuid, uuid_strategy)
    hololink_module.Enumerator.set_uuid_strategy(BOARD_UUID, uuid_strategy)
    # Get a handle to the data source
    channel_metadata = hololink_module.Enumerator.find_channel(channel_ip=args.hololink)
    logging.info(f"{channel_metadata=}")
    # A board whose UUID wasn't registered above enumerates with the default
    # sensor count, so fail loudly here instead of deep inside use_sensor.
    discovered_uuid = channel_metadata["fpga_uuid"]
    if discovered_uuid != BOARD_UUID:
        logging.warning(
            "fpga_uuid=%s does not match BOARD_UUID=%s; "
            "check the board selection at the top of this file.",
            discovered_uuid,
            BOARD_UUID,
        )
        hololink_module.Enumerator.set_uuid_strategy(discovered_uuid, uuid_strategy)
        channel_metadata = hololink_module.Enumerator.find_channel(
            channel_ip=args.hololink
        )

    hololink_channels = []
    vp_bases = []
    for sensor_index in range(args.num_qps):
        sensor_metadata = hololink_module.Metadata(channel_metadata)
        hololink_module.DataChannel.use_sensor(sensor_metadata, sensor_index)
        hololink_module.DataChannel.use_mtu(sensor_metadata, args.mtu)
        vp_bases.append(sensor_metadata["vp_address"])
        hololink_channels.append(hololink_module.DataChannel(sensor_metadata))

    hololink = hololink_channels[0].hololink()
    timestamp_ila_bases = tuple(
        TIMESTAMP_ILA_BASE + sensor * SENSOR_BANK_STRIDE
        for sensor in range(args.num_qps)
    )
    sif_ila_bases = tuple(
        SIF_ILA_BASE + sensor * SENSOR_BANK_STRIDE for sensor in range(args.num_qps)
    )
    logging.info(
        "player_bases=%s timestamp_ila_bases=%s sif_ila_bases=%s "
        "tx_pkt_proc_bases=%s vp_bases=%s",
        [f"{address:#010x}" for address in PLAYER_BASES[: args.num_qps]],
        [f"{address:#010x}" for address in timestamp_ila_bases],
        [f"{address:#010x}" for address in sif_ila_bases],
        [f"{address:#010x}" for address in TX_PKT_PROC_BASES[: args.num_qps]],
        [f"{address:#06x}" for address in vp_bases],
    )
    ila = IlaCapture(hololink, timestamp_ila_bases) if args.ila_capture else None
    # Set up the application
    application = HoloscanApplication(
        cu_context,
        cu_device_ordinal,
        hololink_channels,
        hololink,
        args.rate_mbps,
        args.hololink,
        args.rx_ibv_name,
        args.tx_ibv_name,
        args.tx_ibv_qp,
        args.ibv_port,
        args.frame_size,
        args.mtu,
        args.gpu_id,
        args.forward,
        ila,
        TIMER_TICKS_PER_US,
        vp_bases,
    )
    application.config(args.configuration)
    # Set up the application
    hololink.start()  # Start the Hololink device before configuring IMU
    hololink.reset()
    ptp_sync_timeout_s = 10
    logging.debug("Waiting for PTP sync.")
    if not hololink.ptp_synchronize(hololink_module.Timeout(ptp_sync_timeout_s)):
        logging.error(
            f"Failed to synchronize PTP after {ptp_sync_timeout_s} seconds; ignoring."
        )
    else:
        logging.debug("PTP synchronized.")

    try:
        if ila is not None:
            # Every timestamp ILA in use must be armed before any player starts.
            ila.arm()
        logging.info("Calling run")
        application.run()
    finally:
        if ila is not None:
            ila.stop()
        hololink.stop()
        (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
        if cu_result != cuda.CUresult.CUDA_SUCCESS:
            logging.error(f"Failed to release CUDA primary context: {cu_result}")

    # Processing happens after the run so it never delays a capture window.
    if ila is not None:
        ila.save()


if __name__ == "__main__":
    main()
