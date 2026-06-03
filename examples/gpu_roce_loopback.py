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

# This example assumes the hardware includes a data generator IP (the ram_player module).
# It instantiates the GPU RoCE Transceiver operator and configures it to run in single-kernel
# loopback mode, which sends received data back to the output port. The example also
# configures and triggers the data generator (via DataGen) to start receiving packets.
# Since the GPU RoCE Transceiver operator runs on the GPU, a KeepAlive operator is used to
# receive a termination signal from the user and forward it to the transceiver operator.

import argparse
import logging
import os
import time

import cuda.bindings.driver as cuda
import holoscan

import hololink as hololink_module

DATA_GEN_SPACING_US = 10
BOARD = "RFSoC"


class DataGen:
    PLAYER_ADDR = 0x5000_0000
    PLAYER_TIMER_OFFSET = 0x0008
    PLAYER_WINDOW_SIZE_OFFSET = 0x000C
    PLAYER_WINDOW_NUMBER_OFFSET = 0x0010
    PLAYER_ENABLE_OFFSET = 0x0004
    METADATA_PACKET_OFFSET = 0x102C
    TX_PKT_PROC_ADDR = 0x0120_0000

    def __init__(self, hololink, rate_mbps, sif, data_gen_mode=1):
        self._hololink = hololink
        self.rate_mbps = rate_mbps
        self._sif = sif
        self._data_gen_mode = data_gen_mode
        self._frame_count = 0
        self._stopping = False

    def _enable_sif_tx(self):
        self._hololink.write_uint32(self.TX_PKT_PROC_ADDR, 0x00000005)

    def _configure_player(self):
        if self.bytes_per_window < 64:
            self._hololink.write_uint32(
                self.PLAYER_ADDR + self.PLAYER_WINDOW_SIZE_OFFSET, 64
            )  # Window Size
        else:
            self._hololink.write_uint32(
                self.PLAYER_ADDR + self.PLAYER_WINDOW_SIZE_OFFSET, self.bytes_per_window
            )  # Window Size
        self._hololink.write_uint32(
            self.PLAYER_ADDR + self.PLAYER_WINDOW_NUMBER_OFFSET, self.window_number
        )  # Window Number
        self._hololink.write_uint32(
            self.PLAYER_ADDR + self.PLAYER_TIMER_OFFSET,
            round(
                322 * DATA_GEN_SPACING_US
                if BOARD == "RFSoC"
                else 201 * DATA_GEN_SPACING_US
            ),
        )  # data_gen     , TMR Value

    def _enable_loopback_test(self):
        self.window_number = 4096
        self._configure_player()
        self._hololink.write_uint32(
            self.PLAYER_ADDR + self.PLAYER_ENABLE_OFFSET, 0x00000003
        )  # data_gen     , Enable

    def _disable_data_gen(self):
        self._hololink.write_uint32(
            self.PLAYER_ADDR + self.PLAYER_ENABLE_OFFSET, 0x00000000
        )  # data_gen     , Enable

    def _disable_metadata_packet(self):
        val = self._hololink.read_uint32(self.METADATA_PACKET_OFFSET)
        self._hololink.write_uint32(
            self.METADATA_PACKET_OFFSET, val | (1 << 16)
        )  # Disable Metadata Packet via RMW

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
        pass

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

    def __init__(self, fragment, *args, receiver_op=None, **kwargs):
        self._receiver_op = receiver_op
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec):
        pass

    def compute(self, op_input, op_output, context):
        return None

    def stop(self):
        if self._receiver_op:
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
        hololink_channel,
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
    ):
        logging.info("__init__")
        super().__init__()
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._hololink_channel = hololink_channel
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
        #
        self.metadata_policy = holoscan.core.MetadataPolicy.REJECT

    def compose(self):
        logging.info("compose")
        frame_size = self._frame_size
        frame_context = self._cuda_context

        self._data_gen = DataGen(self._hololink, self._rate_mbps, 0, data_gen_mode=0)
        self._data_gen.bytes_per_window = frame_size

        self._receiver_operator = hololink_module.operators.GpuRoceTransceiverOp(
            self,
            name="Gpu Roce Transceiver",
            frame_size=frame_size,
            frame_context=frame_context,
            ibv_name=self._rx_ibv_name,
            tx_ibv_qp=self._tx_ibv_qp,
            ibv_port=self._ibv_port,
            hololink_channel=self._hololink_channel,
            device=self._data_gen,
            gpu_id=self._gpu_id,
            forward=self._forward,
        )

        # Store reference for shutdown coordination (even though not directly used)
        self._data_gen.set_receiver_operator(self._receiver_operator)

        # Dummy sink operator
        sink_operator = SinkOperator(self, name="sink")
        self.add_flow(self._receiver_operator, sink_operator, {("output", "input")})

        # Keepalive ticker to keep the app running while GPU receiver runs autonomously
        # Pass receiver_op so KeepAlive can stop it when KeepAlive.stop() is called
        keepalive = KeepAlive(
            self,
            holoscan.conditions.BooleanCondition(
                self, name="keepalive_tick", enable_tick=True
            ),
            receiver_op=self._receiver_operator,
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
        help="QP number for the transmitter IBV stream",
    )
    parser.add_argument(
        "--mtu",
        type=int,
        default=1472,
        help="Maximum Transmission Unit (MTU) size in bytes. Default: 1472 (standard), use 9000+ for jumbo frames",
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
        total_sensors=1,
        total_dataplanes=1,
        sifs_per_sensor=2,
    )
    uuid = "889b7ce3-65a5-4247-8b05-4ff1904c3359"
    hololink_module.Enumerator.set_uuid_strategy(uuid, uuid_strategy)
    # Get a handle to the data source
    channel_metadata = hololink_module.Enumerator.find_channel(channel_ip=args.hololink)
    logging.info(f"{channel_metadata=}")
    #
    sensor_metadata = hololink_module.Metadata(channel_metadata)
    hololink_module.DataChannel.use_sensor(sensor_metadata, 0)
    # Add MTU to metadata using update method
    hololink_module.DataChannel.use_mtu(sensor_metadata, args.mtu)
    #
    hololink_channel = hololink_module.DataChannel(sensor_metadata)
    hololink = hololink_channel.hololink()
    # Set up the application
    application = HoloscanApplication(
        cu_context,
        cu_device_ordinal,
        hololink_channel,
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
    )
    application.config(args.configuration)
    # Set up the application
    hololink.start()  # Start the Hololink device before configuring IMU
    hololink.reset()
    ptp_sync_timeout_s = 10
    ptp_sync_timeout = hololink_module.Timeout(ptp_sync_timeout_s)
    logging.debug("Waiting for PTP sync.")
    if not hololink.ptp_synchronize(ptp_sync_timeout):
        logging.error(
            f"Failed to synchronize PTP after {ptp_sync_timeout_s} seconds; ignoring."
        )
    else:
        logging.debug("PTP synchronized.")

    logging.info("Calling run")
    application.run()

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    if cu_result != cuda.CUresult.CUDA_SUCCESS:
        logging.error(f"Failed to release CUDA primary context: {cu_result}")


if __name__ == "__main__":
    main()
