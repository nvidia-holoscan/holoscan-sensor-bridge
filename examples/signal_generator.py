# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Signal generator over an HSB100G (AD9986 MxFE):
#
#   TX (--tx-hololink): SignalGeneratorOp -> IQEncoderOp -> RoceTransmitterOp
#   RX (--rx-hololink): RoceReceiverOp -> IQDecoderOp -> SignalViewerOp
#                                                     -> UdpTransmitterOp (--udp-ip, e.g. for
#                                                        the GNU Radio client in
#                                                        signal_generator_client.grc)
#
# Either leg or both run in one graph. The operators' GUIs share one ImGui window (--no-gui to
# disable); closing it stops the app.

import argparse
import logging
import os
import re
import signal
import sys
import threading

import capture_metadata
import cuda.bindings.driver as cuda
import holoscan as hs

import hololink as hl

CUDA_DEVICE_ORDINAL = 0
# The receiver's ceiling for one frame (BaseReceiverOp's kMaxFrameBytes).
MAX_FRAME_BYTES = 64 * 1024 * 1024
# Each IQ sample travels as two int16 components.
ENCODED_BYTES_PER_SAMPLE = 4
# Should match the GNU Radio client's configuration: 8192 floats.
UDP_MAX_BUFFER_SIZE = 8192 * 4
# The operators' GUI windows, top to bottom: the pipeline's order. The operators add them as
# they start, in an order the runtime picks.
WINDOW_ORDER = ["Signal Generator", "IQ Encoder", "IQ Decoder", "Signal Viewer"]

# Set by FrameLimitOp: hosted operators are constructed without arguments, so they can't be
# handed a live object.
_frame_limit_reached = threading.Event()


class FrameLimitOp(hs.Operator):
    """Counts received frames toward --frame-limit (draining the receiver's metadata)."""

    metadata = hs.Input("metadata", capture_metadata.CaptureMetadata)

    limit = hs.Param(1)

    def setup(self, spec):
        spec.input(self.metadata, queue_depth=4)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.metadata))

    def start(self):
        self._frames = 0

    def compute(self, context):
        self.metadata.receive_data()
        self._frames += 1
        if self._frames >= self.limit:
            _frame_limit_reached.set()


def _cuda_check(result):
    err = result[0]
    if err != cuda.CUresult.CUDA_SUCCESS:
        _, description = cuda.cuGetErrorString(err)
        raise RuntimeError(f"CUDA error: {description}")
    if len(result) == 2:
        return result[1]
    return result[1:]


def bind_output_device(options, operator_path, output_port):
    options.deployment.bind_tensor_output_device(
        operator_path=operator_path,
        output_port=output_port,
        device=hs.execution.DeviceId(CUDA_DEVICE_ORDINAL),
    )


def host_pause_mapping(tx_hololink_ip):
    """The data plane the TX board is enumerated on picks the RX MxFE's host-pause input."""
    tx_data_plane = hl.find_channel(tx_hololink_ip).get_int("data_plane")
    if tx_data_plane not in (0, 1):
        logging.error(
            "HSB100G host pause mapping must have tx-hololink mapped to data plane 0 or 1. "
            f"tx-hololink ({tx_hololink_ip}) is mapped to data plane {tx_data_plane}"
        )
        raise ValueError("TX data plane must be 0 or 1")
    logging.info(f"TX data plane={tx_data_plane}")
    return 1 << tx_data_plane


def ibv_device(board_ip, ibv_name, ibv_port, channel_metadata=None):
    """The IB device and port for the board at board_ip: ibv_name and ibv_port if ibv_name is
    given, else the device on the network interface the board's DataChannel uses. The first
    devices the system lists may not reach the board: on a DGX Spark each port shows up as two
    devices, and only one of them has the board's RoCE v2 GID."""
    if ibv_name:
        return ibv_name, ibv_port
    if channel_metadata is None:
        channel_metadata = hl.find_channel(board_ip)
    interface = channel_metadata.get_string("interface") or ""
    ibv_name, ibv_port = hl.infiniband_device_for_peer(board_ip, interface)
    logging.info(
        f"Using IB device {ibv_name} port {ibv_port} for the board at {board_ip}"
    )
    return ibv_name, ibv_port


def add_transmitter(graph, args):
    """Adds the RoCE transmitter that sends the encoded signal to --tx-hololink."""
    ibv_name, ibv_port = ibv_device(
        args.tx_hololink, args.tx_ibv_name, args.tx_ibv_port
    )
    return graph.op(
        hl.operators.RoceTransmitterOp,
        "roce_transmitter",
        ibv_name=ibv_name,
        ibv_port=ibv_port,
        hololink_ip=args.tx_hololink,
        ibv_qp=args.tx_ibv_qp,
        buffer_size=args.samples_count * ENCODED_BYTES_PER_SAMPLE,
        queue_size=args.tx_queue_size,
    )


def add_receiver(graph, options, args, cu_context):
    """Adds the RoCE receiver for --rx-hololink; its start configures the board's MxFE.

    Returns the receiver and the board's HololinkDevice.
    """
    channel_metadata = hl.find_channel(args.rx_hololink)
    hl.use_sensor(channel_metadata, 0)  # Support Tx/Rx on the same interface
    ibv_name, ibv_port = ibv_device(
        args.rx_hololink, args.rx_ibv_name, args.rx_ibv_port, channel_metadata
    )
    channel = hl.DataChannel(channel_metadata)
    hololink = channel.hololink()
    mxfe = hl.AD9986Config(hololink)
    mxfe.host_pause_mapping(host_pause_mapping(args.tx_hololink))

    def start_mxfe():
        # The receiver's device start: configure the MxFE, and fail unless the JESD link comes
        # up. A failed start gets no device stop, so stop the SPI daemon thread here.
        try:
            mxfe.apply()
            if not mxfe.verify_link_status():
                raise RuntimeError("JESD link status check failed")
        except BaseException:
            mxfe.stop()
            raise

    receiver = graph.op(
        hl.operators.RoceReceiverOp,
        "roce_receiver",
        data_channel=channel,
        device_start=start_mxfe,
        device_stop=mxfe.stop,
        frame_context=int(cu_context),
        frame_size=args.samples_count * ENCODED_BYTES_PER_SAMPLE,
        ibv_name=ibv_name,
        ibv_port=ibv_port,
    )
    bind_output_device(options, "roce_receiver", "tensor")
    return receiver, hololink


def run(args):
    buffer_size = args.samples_count * ENCODED_BYTES_PER_SAMPLE
    if not (args.tx_hololink or args.rx_hololink):
        logging.error("Either tx-hololink or rx-hololink must be provided")
        return 1
    if args.rx_hololink and not args.tx_hololink:
        # The RX board's MxFE configuration depends on the TX board's data plane.
        logging.error("rx-hololink also needs tx-hololink")
        return 1
    if args.samples_count <= 0 or args.samples_count % 4:
        logging.error("samples-count must be a positive multiple of 4")
        return 1
    if args.frame_limit is not None and args.frame_limit < 0:
        logging.error("frame-limit must not be negative")
        return 1
    if args.rx_hololink and buffer_size > MAX_FRAME_BYTES:
        logging.error(
            f"samples-count {args.samples_count} makes {buffer_size}-byte frames, "
            f"more than the receiver's {MAX_FRAME_BYTES}"
        )
        return 1

    # Set the process to real-time priority
    if args.real_time:
        sched_param = os.sched_param(sched_priority=99)  # Maximum priority
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, sched_param)
            logging.info(f"Set PID={os.getpid()} priority to real-time")
        except PermissionError:
            logging.error(
                f"Failed to set scheduler for PID={os.getpid()}. "
                "Need root privileges for real-time scheduling."
            )

    logging.info("Initializing")
    _cuda_check(cuda.cuInit(0))
    cu_device = _cuda_check(cuda.cuDeviceGet(CUDA_DEVICE_ORDINAL))
    cu_context = _cuda_check(cuda.cuDevicePrimaryCtxRetain(cu_device))
    _cuda_check(cuda.cuCtxSetCurrent(cu_context))
    try:
        return run_graph(args, cu_context)
    finally:
        _cuda_check(cuda.cuDevicePrimaryCtxRelease(cu_device))


def run_graph(args, cu_context):
    # HolovizOp has no Dear ImGui support, so the operators' GUIs go to a separate renderer,
    # with its own window and thread.
    renderer = None if args.no_gui else hl.ImGuiRenderer.shared()
    gui = renderer is not None
    if renderer is not None:
        renderer.set_window_order(WINDOW_ORDER)

    graph = hs.Graph("signal_generator")
    options = hs.CompileOptions()

    transmitter = None
    if args.tx_hololink:
        signal_generator = graph.op(
            hl.operators.SignalGeneratorOp,
            "signal_generator",
            samples_count=args.samples_count,
            sampling_interval=args.sampling_interval,
            in_phase=args.expression_i,
            quadrature=args.expression_q,
            gui=gui,
        )
        iq_encoder = graph.op(hl.operators.IQEncoderOp, "iq_encoder", gui=gui)
        transmitter = add_transmitter(graph, args)

        graph.add_flow(signal_generator.output, iq_encoder.input)
        # The transmitter holds up to tx-queue-size encoded buffers while they're sent; size
        # the queue (and so the encoder's buffer pool) beyond that.
        graph.add_flow(
            iq_encoder.output, transmitter.input, queue_depth=args.tx_queue_size + 2
        )
        bind_output_device(options, "signal_generator", "output")
        bind_output_device(options, "iq_encoder", "output")

    rx_hololink = None
    if args.rx_hololink:
        receiver, rx_hololink = add_receiver(graph, options, args, cu_context)
        iq_decoder = graph.op(
            hl.operators.IQDecoderOp,
            "iq_decoder",
            max_samples=args.samples_count,
            gui=gui,
        )
        signal_viewer = graph.op(hl.operators.SignalViewerOp, "signal_viewer", gui=gui)

        # The receiver fails the run if an emit is backpressured, so its consumers drop the
        # oldest frames instead (a live stream can't be paused).
        receiver_edge = dict(queue_policy=hs.QueuePolicy.DROP_OLDEST, queue_depth=4)
        graph.add_flow(receiver.tensor, iq_decoder.input, **receiver_edge)
        graph.add_flow(
            iq_decoder.output,
            signal_viewer.input,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
            queue_depth=1,
        )
        if args.frame_limit:
            frame_limit = graph.op(FrameLimitOp, "frame_limit", limit=args.frame_limit)
            graph.add_flow(receiver.metadata, frame_limit.metadata, **receiver_edge)
        else:
            discard = graph.op(hl.operators.DiscardMetadataOp, "discard_metadata")
            graph.add_flow(receiver.metadata, discard.metadata, **receiver_edge)

        if args.udp_ip:
            udp_transmitter = graph.op(
                hl.operators.UdpTransmitterOp,
                "udp_transmitter",
                ip=args.udp_ip,
                port=args.udp_port,
                max_buffer_size=UDP_MAX_BUFFER_SIZE,
            )
            graph.add_flow(iq_decoder.output, udp_transmitter.input, queue_depth=4)
        bind_output_device(options, "iq_decoder", "output")

        # Configure the MxFE (in the receiver's start) before the transmitter connects.
        if transmitter is not None:
            graph.lifecycle().before(
                receiver, transmitter, hs.lifecycle.LifecycleStage.START
            )
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))

    plan = hs.compile(graph, options)
    if not plan.ok:
        logging.error(plan.summary())
        logging.error("Graph compilation failed")
        return 1

    _frame_limit_reached.clear()
    # Like the C++ app, SIGTERM stops the app the way Ctrl-C does.
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        if rx_hololink is not None:
            rx_hololink.start()
            rx_hololink.reset()
        with hs.run_async(plan) as session:
            try:
                while not session.wait(timeout=0.1):
                    if _frame_limit_reached.is_set():
                        logging.info(f"Received {args.frame_limit} frames")
                        break
                    if renderer is not None and renderer.window_closed():
                        break
            except KeyboardInterrupt:
                pass
            finally:
                # Stopping: from here on Ctrl-C or SIGTERM kills the app, in case an operator
                # doesn't stop (leaving the with block waits without watching for signals).
                signal.signal(signal.SIGINT, signal.SIG_DFL)
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
            session.request_stop()
            session.wait()
            failed = session.termination == hs.RunTermination.FAILED
    finally:
        if rx_hololink is not None:
            try:
                rx_hololink.reset()
            except Exception as e:
                logging.error(f"Error during hololink reset: {e}")
            try:
                rx_hololink.stop()
            except Exception as e:
                logging.error(f"Error during hololink stop: {e}")
    return 1 if failed else 0


def parse_rational(rational_str):
    match = re.fullmatch(r"(\d+)/(\d+)", rational_str)
    if not match or int(match.group(2)) == 0:
        raise argparse.ArgumentTypeError(
            f'"{rational_str}" is not a "num/den" ratio with a positive denominator'
        )
    return rational_str


def main():
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
        "--frame-limit",
        type=int,
        default=None,
        help="Stop after receiving this many frames (useful for CI)",
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
    parser.add_argument(
        "--sampling-interval",
        type=parse_rational,
        default="1/128",
        help='The interval between sequential samples, must be in the following format: "num/den"',
    )

    parser.add_argument(
        "--tx-hololink",
        type=str,
        default="",
        help="IP address of Hololink board that the data is transmitted to",
    )
    parser.add_argument(
        "--tx-ibv-name",
        type=str,
        default="",
        help="IBV device name used for transmission "
        "(default: the device on the TX board's network interface)",
    )
    parser.add_argument(
        "--tx-ibv-port",
        type=int,
        default=1,
        help="Port number of IBV device used for transmission (with --tx-ibv-name)",
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
        default="",
        help="Local IB device name used for reception "
        "(default: the device on the RX board's network interface)",
    )
    parser.add_argument(
        "--rx-ibv-port",
        type=int,
        default=1,
        help="Port of the local IB device used for reception (with --rx-ibv-name)",
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
    logging.basicConfig(level=args.log_level)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
