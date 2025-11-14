# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# See README.md for detailed information.

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
    DATA_GEN_BASE_ADDR = 0x01000100
    DATA_GEN_ENABLE = 0x04
    DATA_GEN_MODE = 0x08
    DATA_GEN_SIZE = 0x0C
    DATA_GEN_OUTPUT_RATE = 0x10
    DATA_GEN_TMR_VAL = 0x14
    ETH_DATA_RATE_MBPS = 10000

    def __init__(self, hololink, rate_mbps, sif, data_gen_mode=1):
        self._hololink = hololink
        #       self.bytes_per_window = bytes_per_window
        self.rate_mbps = rate_mbps
        self._sif = sif
        self._data_gen_mode = data_gen_mode
        self._frame_count = 0
        self._stopping = False

    def _configure_data_gen(self):
        # output_div = int((self.rate_mbps / self.ETH_DATA_RATE_MBPS) * 65535)
        sif_mask = 0x10000 * self._sif
        self._hololink.write_uint32(
            self.DATA_GEN_BASE_ADDR + sif_mask + self.DATA_GEN_ENABLE, 0x00000000
        )  # data_gen     , Enable
        self._hololink.write_uint32(
            self.DATA_GEN_BASE_ADDR + sif_mask + self.DATA_GEN_MODE, self._data_gen_mode
        )  # data_gen     , Mode - Count
        self._hololink.write_uint32(
            self.DATA_GEN_BASE_ADDR + sif_mask + self.DATA_GEN_SIZE,
            self.bytes_per_window,
        )  # data_gen     , Size
        self._hololink.write_uint32(
            # self.DATA_GEN_BASE_ADDR + sif_mask + self.DATA_GEN_OUTPUT_RATE, output_div
            self.DATA_GEN_BASE_ADDR + sif_mask + self.DATA_GEN_OUTPUT_RATE,
            0xFFFF,
        )  # data_gen     , Output Rate
        self._hololink.write_uint32(
            self.DATA_GEN_BASE_ADDR + sif_mask + self.DATA_GEN_TMR_VAL,
            round(
                322 * DATA_GEN_SPACING_US
                if BOARD == "RFSoC"
                else 201 * DATA_GEN_SPACING_US
            ),
        )  # data_gen     , TMR Value

    def _enable_data_gen(self):
        sif_mask = 0x10000 * self._sif
        time.sleep(0.1)
        self._hololink.write_uint32(
            self.DATA_GEN_BASE_ADDR + sif_mask + self.DATA_GEN_ENABLE, 0x00000003
        )  # data_gen     , Enable

    def _disable_data_gen(self):
        sif_mask = 0x10000 * self._sif
        time.sleep(0.1)
        self._hololink.write_uint32(
            self.DATA_GEN_BASE_ADDR + sif_mask + self.DATA_GEN_ENABLE, 0x00000000
        )  # data_gen     , Enable

    def _vp_default(self):
        vp_mask = 0x40 * self.vp_index
        self._hololink.write_uint32(
            0x00001000 + vp_mask, 0x00000000
        )  # vp     , Destination QP
        self._hololink.write_uint32(
            0x00001004 + vp_mask, 0x0000F00D
        )  # vp     , Remote Key
        self._hololink.write_uint32(
            0x00001008 + vp_mask, 0x00000000
        )  # vp     , Buffer 0 Virtual Address
        self._hololink.write_uint32(
            0x0000100C + vp_mask, 0x00000000
        )  # vp     , Buffer 1 Virtual Address
        self._hololink.write_uint32(
            0x00001010 + vp_mask, 0x00000000
        )  # vp     , Buffer 2 Virtual Address
        self._hololink.write_uint32(
            0x00001014 + vp_mask, 0x00000000
        )  # vp     , Buffer 3 Virtual Address
        self._hololink.write_uint32(
            0x00001018 + vp_mask, self.bytes_per_window
        )  # vp     , Bytes per Window
        self._hololink.write_uint32(
            0x0000101C + vp_mask, 0x00000001
        )  # vp     , Buffer enable (0x1)
        self._hololink.write_uint32(
            0x00001020 + vp_mask, 0x0000FFFF
        )  # vp     , dp_pkt_mac_addr_lo
        self._hololink.write_uint32(
            0x00001024 + vp_mask, 0xFFFFFFFF
        )  # vp     , dp_pkt_mac_addr_hi
        self._hololink.write_uint32(
            0x00001028 + vp_mask, 0x0000BEEF
        )  # vp     , dp_pkt_ip_addr
        self._hololink.write_uint32(
            0x0000102C + vp_mask, 0x00002000
        )  # vp     , dp_pkt_fpga_udp_port

    def _hif_default(self):
        hif_mask = 0x10000 * self.hif_index
        vp_mask = 1 << self.vp_index
        self._hololink.write_uint32(
            0x02000304 + hif_mask, 0x0000000B
        )  # hif     , Eth pkt length - 1408 Bytes or 11 Pages
        self._hololink.write_uint32(
            0x02000308 + hif_mask, 0x0000DEAD
        )  # hif     , Fpga UDP port
        self._hololink.write_uint32(
            0x0200030C + hif_mask, vp_mask
        )  # hif     , VP mask    - Enable SIF

    def _stream_test(self, vp_index, hif_index):
        # Default Register setup
        self.vp_index = vp_index
        self.hif_index = hif_index

        #       self._vp_default()
        #       self._hif_default()
        # Configure the data gen
        self._configure_data_gen()
        # Enable the data gen
        self._enable_data_gen()

    def start(self):
        # Start Packet Gen after system is in working condition
        time.sleep(1)
        self._stream_test(0, 1)  # Stream (SIF 0,VP 0) to HIF 1
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
            ibv_port=self._ibv_port,
            hololink_channel=self._hololink_channel,
            device=self._data_gen,
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
        "--ptp-sync",
        action="store_true",
        help="After reset, wait for PTP time to synchronize.",
    )
    parser.add_argument(
        "--skip-reset",
        action="store_true",
        help="Don't call reset on the hololink device.",
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
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_device_ordinal = 0
    cu_result, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_result, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
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
    assert cu_result == cuda.CUresult.CUDA_SUCCESS


if __name__ == "__main__":
    main()
