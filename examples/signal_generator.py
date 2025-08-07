# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import threading

import holoscan
from cuda import cuda

import hololink as hololink_module


def get_infiniband_interface_index(interface_name):
    """Get the index of an infiniband interface by name."""
    try:
        infiniband_devices = hololink_module.infiniband_devices()
        return infiniband_devices.index(interface_name)
    except (FileNotFoundError, ValueError):
        return -1


class HololinkDevice:
    def __init__(self, hololink, frame_size, host_pause_mapping):
        logging.info("HololinkDevice __init__")
        self._hololink = hololink
        self._frame_size = frame_size
        self._host_pause_mapping = host_pause_mapping
        logging.info(f"HololinkDevice {self._frame_size=:#X}")

    def start(self):
        logging.info("HololinkDevice start")

        self._mxfe_config = hololink_module.AD9986Config(self._hololink)
        self._mxfe_config.host_pause_mapping(self._host_pause_mapping)
        self._mxfe_config.apply()

    def stop(self):
        logging.info("HololinkDevice stop")


class RoceReceiverInitializer:
    """
    Helper class to handle initialization and resource management for RoCE receiver setup.
    This class encapsulates the complex initialization logic while providing proper error handling
    and resource cleanup. It follows the same pattern as other examples in the codebase while
    maintaining clean separation of concerns.
    """

    def __init__(self, hololink_ip, frame_size, host_pause_mapping):
        self._hololink_ip = hololink_ip
        self._frame_size = frame_size
        self._host_pause_mapping = host_pause_mapping
        self._cu_context = None
        self._hololink_channel = None
        self._hololink = None
        self._hololink_device = None
        self._initialized = False

    def initialize(self):
        """Initialize CUDA context and Hololink resources."""
        try:
            # Initialize CUDA
            (cu_result,) = cuda.cuInit(0)
            if cu_result != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError("Failed to initialize CUDA")

            cu_device_ordinal = 0
            cu_result, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
            if cu_result != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError("Failed to get CUDA device")

            cu_result, self._cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
            if cu_result != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError("Failed to retain CUDA context")

            # Initialize Hololink
            channel_metadata = hololink_module.Enumerator.find_channel(
                channel_ip=self._hololink_ip
            )
            hololink_module.DataChannel.use_sensor(
                channel_metadata, 0
            )  # Support Tx/Rx on the same interface
            logging.info(f"{channel_metadata=}")

            self._hololink_channel = hololink_module.DataChannel(channel_metadata)
            self._hololink = self._hololink_channel.hololink()
            self._hololink_device = HololinkDevice(
                self._hololink, self._frame_size, self._host_pause_mapping
            )

            self._hololink.start()
            self._hololink.reset()
            self._initialized = True

        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Failed to initialize RoCE receiver: {str(e)}")

    def cleanup(self):
        """Clean up resources in reverse order of initialization."""
        if self._hololink:
            try:
                self._hololink.reset()
            except Exception as e:
                logging.error(f"Error during hololink reset: {e}")

        if self._hololink_device:
            try:
                self._hololink_device.stop()
            except Exception as e:
                logging.error(f"Error during device stop: {e}")

        self._hololink = None
        self._hololink_channel = None
        self._hololink_device = None
        self._cu_context = None
        self._initialized = False

    def create_receiver_op(self, fragment, name, ibv_name, ibv_port):
        """Create and return a configured RoceReceiverOp."""
        if not self._initialized:
            raise RuntimeError("RoceReceiverInitializer not initialized")

        return hololink_module.operators.RoceReceiverOp(
            fragment,
            name=name,
            frame_size=self._frame_size,
            frame_context=self._cu_context,
            ibv_name=ibv_name,
            ibv_port=ibv_port,
            hololink_channel=self._hololink_channel,
            device=self._hololink_device,
        )

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# The SignalGeneratorTxApp is a sample app that does the following:
# 1. Generates a signal with IQ components.
# 2. The signal is encoded into an IQ buffer
# 3. The IQ buffer is transmitted by a RoCE transmitter.
class SignalGeneratorTxApp(holoscan.core.Application):
    def __init__(
        self,
        args,
        renderer,
    ):
        logging.info("__init__")
        super().__init__()
        self._args = args
        self._renderer = renderer

    def compose(self):
        logging.info("compose")
        samples_count = self._args.samples_count
        sampling_interval = self._args.sampling_interval
        buffer_size = samples_count * 4

        self._signal_generator = hololink_module.operators.SignalGeneratorOp(
            self,
            name="Signal Generator",
            renderer=self._renderer,
            samples_count=samples_count,
            sampling_interval=sampling_interval,
            in_phase=self._args.expression_i,
            quadrature=self._args.expression_q,
        )
        self._iq_encoder = hololink_module.operators.IQEncoderOp(
            self,
            name="IQ Encoder",
            renderer=self._renderer,
        )
        self._roce_transmitter = hololink_module.operators.RoceTransmitterOp(
            self,
            name="Roce Transmitter",
            ibv_name=self._args.tx_ibv_name,
            ibv_port=self._args.tx_ibv_port,
            hololink_ip=self._args.tx_hololink,
            ibv_qp=self._args.tx_ibv_qp,
            buffer_size=buffer_size,
            queue_size=self._args.tx_queue_size,
        )

        self.add_flow(self._signal_generator, self._iq_encoder, {("output", "input")})
        self.add_flow(self._iq_encoder, self._roce_transmitter, {("output", "input")})


# The SignalGeneratorRxApp is a sample app that does the following:
# 1. Receives an IQ buffer from a RoCE receiver.
# 2. The IQ buffer is decoded back into a signal (with IQ components).
# 3. The signal is viewed by the SignalViewerOp.
# 4. The signal is transmitted to a UDP port and can be viewed by the GNU Radio app.
class SignalGeneratorRxApp(holoscan.core.Application):
    def __init__(
        self,
        args,
        renderer,
    ):
        logging.info("__init__")
        super().__init__()
        self._args = args
        self._renderer = renderer
        self._receiver_initializer = None

    def compose(self):
        logging.info("compose")
        udp_enabled = len(self._args.udp_ip) > 0
        samples_count = self._args.samples_count
        buffer_size = samples_count * 4

        host_pause_mapping = 1 << get_infiniband_interface_index(self._args.tx_ibv_name)

        # Initialize receiver resources
        self._receiver_initializer = RoceReceiverInitializer(
            self._args.rx_hololink, buffer_size, host_pause_mapping
        )
        self._receiver_initializer.initialize()

        try:
            self._roce_receiver = self._receiver_initializer.create_receiver_op(
                self,
                name="Roce Receiver",
                ibv_name=self._args.rx_ibv_name,
                ibv_port=self._args.rx_ibv_port,
            )

            self._iq_decoder = hololink_module.operators.IQDecoderOp(
                self,
                renderer=self._renderer,
                name="IQ Decoder",
            )
            self._signal_viewer = hololink_module.operators.SignalViewerOp(
                self,
                renderer=self._renderer,
                name="Signal Viewer",
            )

            self.add_flow(self._roce_receiver, self._iq_decoder, {("output", "input")})
            self.add_flow(self._iq_decoder, self._signal_viewer, {("output", "input")})

            if udp_enabled:
                self._udp_transmitter = hololink_module.operators.UdpTransmitterOp(
                    self,
                    ip=self._args.udp_ip,
                    port=self._args.udp_port,
                    # The max_buffer_size value should match the gnu radio configuration file
                    max_buffer_size=8192 * 4,  # 8192 * sizeof(float)
                    name="Udp Transmitter",
                )
                self.add_flow(
                    self._iq_decoder, self._udp_transmitter, {("output", "input")}
                )
        except Exception as e:
            if self._receiver_initializer:
                self._receiver_initializer.cleanup()
            raise e

    def stop(self):
        """Clean up resources when the application stops."""
        if self._receiver_initializer:
            self._receiver_initializer.cleanup()
        super().stop()


def parse_rational(rational_str):
    return hololink_module.operators.Rational(rational_str)


def main():
    rx_default_infiniband_interface = "mlx5_0"
    tx_default_infiniband_interface = "mlx5_1"
    try:
        infiniband_interfaces = hololink_module.infiniband_devices()
        rx_default_infiniband_interface = infiniband_interfaces[0]
        tx_default_infiniband_interface = infiniband_interfaces[1]
    except FileNotFoundError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level to display",
    )
    parser.add_argument(
        "--real-time",
        action="store_true",
        help="Set the process to real-time priority",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable the graphic user interface",
    )
    parser.add_argument(
        "--expression-i",
        type=str,
        default="cos(2*PI*x)",
        help="Signal expression for the in-phase component. Use 'x' as the expression's variable",
    )
    parser.add_argument(
        "--expression-q",
        type=str,
        default="sin(2*PI*x)",
        help="Signal expression for the quadrature component. Use 'x' as the expression's variable",
    )
    parser.add_argument(
        "--samples-count",
        type=int,
        default=4096 * 1000 * 3,
        help="Number of samples to be generated",
    )
    default_sampling_interval = hololink_module.operators.Rational("1/128")
    parser.add_argument(
        "--sampling-interval",
        type=parse_rational,
        default=default_sampling_interval,
        help='The interval between sequential samples, must be in the following format: "num/den"',
    )

    parser.add_argument(
        "--tx-hololink",
        type=str,
        default="",
        help="IP address of Hololink board that the data is transitted to",
    )
    parser.add_argument(
        "--tx-ibv-name",
        type=str,
        default=tx_default_infiniband_interface,
        help="IBV device name used for transmission",
    )
    parser.add_argument(
        "--tx-ibv-port",
        type=int,
        default=1,
        help="Port number of IBV device used for transmission",
    )
    parser.add_argument(
        "--tx-ibv-qp",
        type=int,
        default=2,
        help="QP number for the IBV stream that the data is transmitted to",
    )
    parser.add_argument(
        "--tx-queue-size",
        type=int,
        default=3,
        help="The number of buffers that can wait to be transmitted",
    )

    parser.add_argument(
        "--rx-hololink",
        default="",
        help="IP address of Hololink board that the data is received from",
    )
    parser.add_argument(
        "--rx-ibv-name",
        type=str,
        default=rx_default_infiniband_interface,
        help="Local IB device name used for reception",
    )
    parser.add_argument(
        "--rx-ibv-port",
        type=int,
        default=1,
        help="Port of the local IB device used for reception",
    )

    parser.add_argument(
        "--udp-ip",
        type=str,
        default="",
        help="IP to transmit the data to",
    )
    parser.add_argument(
        "--udp-port",
        type=int,
        default=5000,
        help="Port to transmit the data to",
    )

    args = parser.parse_args()

    # Set the process to real-time priority
    if args.real_time:
        sched_param = os.sched_param(sched_priority=99)  # Maximum priority
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, sched_param)
            logging.info(f"Set PID={os.getpid()} priority to real-time")
        except PermissionError:
            logging.error(
                f"Failed to set scheduler for PID={os.getpid()}. Need root privileges for real-time scheduling."
            )

    hololink_module.logging_level(args.log_level)
    logging.info("Initializing")

    renderer = None
    if not args.no_gui:
        renderer = hololink_module.ImGuiRenderer()

    tx_app = None
    rx_app = None
    threads = []

    if args.tx_hololink:
        tx_app = SignalGeneratorTxApp(
            args,
            renderer,
        )
        tx_thread = threading.Thread(target=tx_app.run)
        threads.append(tx_thread)
        tx_thread.start()

    if args.rx_hololink:
        rx_app = SignalGeneratorRxApp(
            args,
            renderer,
        )
        rx_thread = threading.Thread(target=rx_app.run)
        threads.append(rx_thread)
        rx_thread.start()

    if not threads:
        logging.error("Either tx-hololink or rx-hololink must be provided")
        return 1

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    return 0


if __name__ == "__main__":
    main()
