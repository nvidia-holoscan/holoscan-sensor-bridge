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

# Single IMX274 sub-frame live viewer over RoCE (RDMA/InfiniBand). Python port of
# examples/sub_frame_imx274_player.cpp, via the HSDK5 Python bindings; see
# linux_sub_frame_imx274_player.py for the same player over the Linux socket receiver:
#
#   RoceReceiverOp -> CsiToBayerOp (sub-frame) -> ImageProcessorOp -> SubFrameCombinerOp ->
#     BayerDemosaicOp -> HolovizOp
#
# The sensor frame travels and is processed as horizontal bands of --sub-frame-rows rows: the
# receiver delivers one band at a time, and CsiToBayerOp and ImageProcessorOp each run on a band
# as soon as it lands. SubFrameCombinerOp reassembles the bands into a full frame, which is
# demosaiced once into RGBA16 and displayed. Combining before demosaicing (the HSDK4 version of
# this player demosaiced each band) avoids seams at the band boundaries.
#
# Usage:
#   sub_frame_imx274_player.py [OPTIONS]
#
# Options:
#   --hololink IP                 IP address of Hololink board (default: 192.168.0.2)
#   --camera-mode N               IMX274 mode (default: 0 = 3840x2160@60fps)
#   --sub-frame-rows N            Rows per sub-frame (default: 216; even, and must divide the
#                                 camera height evenly; 0 = whole frames)
#   --frame-limit N               Stop after N frames (default: run until the window closes)
#   --ibv-name NAME               IBV device to use (default: the device on the board's network interface)
#   --ibv-port N                  Port number of the IBV device (default: 1)
#   --headless                    Render into a CUDA image instead of a window
#   --fullscreen                  Run in fullscreen mode
#   --expander-configuration N    I2C expander configuration (0 or 1, default: 0)
#   --pattern N                   Configure a test pattern (0-11) instead of live video
#   --log-level N                 Python logging level (default: 20 = INFO)

import argparse
import logging
import signal
import sys

import capture_metadata
import holoscan as hs
import holoscan_vision as hv
import holoscan_visualization as hz  # noqa: F401 -- registers "holoscan.visualization.HolovizOp"
from cuda.bindings import driver as cuda_driver

import hololink as hl

_BAYER_FORMAT_NAMES = ["bggr", "rggb", "gbrg", "grbg"]
_BAYER_FORMAT_VALUES = {name: index for index, name in enumerate(_BAYER_FORMAT_NAMES)}

# Receiver buffers: six frames of sub-frame chunks, at most this many (a 4K band page is about
# 1.25 MB, so 66 of them take about 83 MB of GPU memory). See receiver_queue_size() for why the
# ring is this deep.
_MAX_PAGES = 66
# The bands run through the capture lane in step, so the queues up to the combiner seldom hold
# more than one or two; keep them shallow, since every queue entry costs a 64 MiB output pool slot
# in the receiver, CsiToBayerOp and ImageProcessorOp.
_SUB_FRAME_QUEUE_DEPTH = 4


def _cuda_check(result):
    err = result[0]
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        _, description = cuda_driver.cuGetErrorString(err)
        raise RuntimeError(f"CUDA error: {description}")
    if len(result) == 2:
        return result[1]
    return result[1:]


def resolve_sub_frame_rows(sub_frame_rows, height):
    """Check --sub-frame-rows against the camera height; 0 selects whole frames."""
    if sub_frame_rows == 0:
        return height
    if sub_frame_rows > height:
        raise ValueError(
            f"Sub frame rows of {sub_frame_rows} is greater than camera height of {height}"
        )
    if height % sub_frame_rows != 0:
        raise ValueError(
            f"Camera height of {height} is not evenly divisible by sub frame rows of "
            f"{sub_frame_rows}"
        )
    if sub_frame_rows % 2 != 0:
        # ImageProcessorOp takes the color filter phase from each band's own first row, so every
        # band must start on an even image row.
        raise ValueError(f"Sub frame rows of {sub_frame_rows} is odd; it must be even")
    return sub_frame_rows


def chunks_per_frame(geometry, sub_frame_size):
    """Chunks the HSB sends per frame: one more than the frame has bands when the CSI data
    carries a header."""
    csi_length = (
        geometry["start_byte"]
        + geometry["bytes_per_line"] * geometry["pixel_height"]
        + geometry["trailing_bytes"]
    )
    return -(-csi_length // sub_frame_size)


def receiver_pages(geometry, sub_frame_size):
    """Receiver buffers for six frames of chunks, at most _MAX_PAGES."""
    return min(chunks_per_frame(geometry, sub_frame_size) * 6, _MAX_PAGES)


def receiver_queue_size(geometry, sub_frame_size, pages):
    """The receiver's queue_size: one frame of chunks short of pages.

    The board writes the pages round-robin and reuses a page `pages` chunks after it wrote it,
    whether or not the host is done with it: RoceReceiverOp copies a page out on its CUDA
    stream some time after taking it, and nothing holds the board off meanwhile. queue_size caps
    how far behind the newest chunk a page can be when the receiver hands it out, so this many
    chunks always remain before the board writes that page again. With queue_size equal to
    pages, the oldest page handed out could already be being rewritten: a band of a later frame
    then shows up in the assembled frame (a tear that drifts down the image)."""
    margin = chunks_per_frame(geometry, sub_frame_size)
    return pages - margin if pages > margin else 1


class FrameLimitOp(hs.Operator):
    """Stops the session once the combiner has assembled --frame-limit frames."""

    metadata = hs.Input("metadata", capture_metadata.CaptureMetadata)

    frame_limit = hs.Param(0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._frames_seen = 0

    def setup(self, spec):
        spec.input(self.metadata, queue_depth=4)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.metadata))

    def compute(self, context):
        self.metadata.receive_data()
        self._frames_seen += 1
        if self.frame_limit > 0 and self._frames_seen >= self.frame_limit:
            context.request_stop()


class DiscardImageOp(hs.Operator):
    """With --headless, HolovizOp renders into a CUDA image on its output instead of a window;
    the output must be consumed, and this drops each rendered frame."""

    input = hs.Input("input", hs.schema.Image)

    def setup(self, spec):
        spec.input(self.input, queue_depth=1)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, context):
        self.input.receive_data()


def run(
    hololink_ip,
    camera_mode,
    sub_frame_rows,
    ibv_name,
    ibv_port,
    expander_configuration,
    pattern,
    frame_limit,
    headless,
    fullscreen,
):
    cuda_device_ordinal = 0

    print("Initializing.")
    _cuda_check(cuda_driver.cuInit(0))
    cu_device = _cuda_check(cuda_driver.cuDeviceGet(cuda_device_ordinal))
    cu_context = _cuda_check(cuda_driver.cuDevicePrimaryCtxRetain(cu_device))
    try:
        cuda_driver.cuCtxSetCurrent(cu_context)

        # Get a handle to the data source
        channel_metadata = hl.find_channel(hololink_ip)
        channel = hl.DataChannel(channel_metadata)

        camera = hl.NativeImx274Sensor(
            channel, expander_configuration=expander_configuration
        )
        camera.set_mode(camera_mode)
        width = camera.get_width()
        height = camera.get_height()
        if width == 0 or height == 0:
            print(f"Unsupported --camera-mode {camera_mode}", file=sys.stderr)
            return 1
        try:
            sub_frame_rows = resolve_sub_frame_rows(sub_frame_rows, height)
        except ValueError as e:
            print(e, file=sys.stderr)
            return 1
        pixel_format = camera.get_pixel_format()
        bayer_format = camera.get_bayer_format()
        geometry = hl.resolve_generic_geometry(camera)

        # The receiver delivers one band at a time: what CsiToBayerOp computes for its
        # sub-frame size.
        sub_frame_size = geometry["bytes_per_line"] * sub_frame_rows
        pages = receiver_pages(geometry, sub_frame_size)
        queue_size = receiver_queue_size(geometry, sub_frame_size, pages)

        graph = hs.Graph("sub_frame_imx274_player")

        rx = graph.op(
            hl.operators.RoceReceiverOp,
            "receiver",
            data_channel=channel,
            frame_context=int(cu_context),
            device_start=camera.start,
            device_stop=camera.stop,
            frame_size=sub_frame_size,
            # Trim each band to the bytes received: the last band of a frame is shorter when
            # the CSI data carries a header.
            trim=True,
            pages=pages,
            # One frame of chunks short of pages: the margin before the board reuses a page.
            queue_size=queue_size,
            # Report "Ingress frame timeout" when no data comes, as HSDK4's receiver did here
            # (use_frame_ready_condition=False).
            watch_ingress=True,
            ibv_name=ibv_name,
            ibv_port=ibv_port,
        )

        csi = graph.op(
            hl.operators.CsiToBayerOp,
            "csi_to_bayer",
            cuda_device_ordinal=cuda_device_ordinal,
            sub_frame_rows=sub_frame_rows,
            geometry=geometry,
            bayer_format=bayer_format,
        )

        image_processor = graph.op(
            hl.operators.ImageProcessorOp,
            "image_processor",
            pixel_format=pixel_format,
            pixel_width=geometry["pixel_width"],
            pixel_height=geometry["pixel_height"],
            bayer_format=_BAYER_FORMAT_VALUES[bayer_format],
            # Optical black value for imx274 is 50
            optical_black=50,
            cuda_device_ordinal=cuda_device_ordinal,
        )

        combiner = graph.op(
            hl.operators.SubFrameCombinerOp,
            "sub_frame_combiner",
            cuda_device_ordinal=cuda_device_ordinal,
            height=height,
        )

        demosaic = graph.op(
            hv.operators.BayerDemosaicOp,
            "bayer_demosaic",
            bayer_grid_pos=bayer_format,
            generate_alpha=True,
            alpha_value=65535,
        )

        frame_limiter = graph.op(FrameLimitOp, "frame_limit", frame_limit=frame_limit)

        if headless:
            # Renders into a CUDA image on HolovizOp's output (consumed by DiscardImageOp). The
            # CUDA sink has no framebuffer_srgb: its sRGB format encodes on write instead.
            sink = {
                "kind": "cuda",
                "width": width,
                "height": height,
                "format": "rgba8_srgb",
            }
        else:
            sink = {
                "kind": "window",
                "title": f"Holoviz - {hololink_ip}",
                "width": width,
                "height": height,
                "fullscreen": fullscreen,
                "close_on_escape": True,
                # The demosaiced data is linear sensor output: encode to sRGB on write.
                "framebuffer_srgb": True,
            }
        viz = graph.op(
            "holoscan.visualization.HolovizOp",
            "holoviz",
            layers=[
                {
                    "port_name": "video",
                    "kind": "image",
                    "format": "rgba16_unorm",
                    "max_width": width,
                    "max_height": height,
                }
            ],
            sink=sink,
            # Render one frame per assembled frame.
            trigger={"kind": "layer"},
        )

        graph.add_flow(rx.tensor, csi.in_tensor, queue_depth=_SUB_FRAME_QUEUE_DEPTH)
        graph.add_flow(rx.metadata, csi.in_metadata, queue_depth=_SUB_FRAME_QUEUE_DEPTH)
        graph.add_flow(
            csi.tensor, image_processor.in_tensor, queue_depth=_SUB_FRAME_QUEUE_DEPTH
        )
        graph.add_flow(
            csi.metadata,
            image_processor.in_metadata,
            queue_depth=_SUB_FRAME_QUEUE_DEPTH,
        )
        graph.add_flow(
            image_processor.tensor,
            combiner.in_tensor,
            queue_depth=_SUB_FRAME_QUEUE_DEPTH,
        )
        graph.add_flow(
            image_processor.metadata,
            combiner.in_metadata,
            queue_depth=_SUB_FRAME_QUEUE_DEPTH,
        )
        graph.add_flow(combiner.tensor, demosaic.input, queue_depth=4)
        # The counter must never hold the combiner back: shed rather than fill up.
        graph.add_flow(
            combiner.metadata,
            frame_limiter.metadata,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
            queue_depth=4,
        )
        # A depth-1 mailbox: if the display stalls, it skips to the latest frame.
        graph.add_flow(
            demosaic.output,
            viz.video,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
            queue_depth=1,
        )
        if headless:
            discard = graph.op(DiscardImageOp, "discard")
            graph.add_flow(
                viz.output,
                discard.input,
                queue_policy=hs.QueuePolicy.DROP_OLDEST,
                queue_depth=1,
            )

        # HolovizOp blocks on the swapchain at the display refresh rate, and band processing can
        # hold up whatever shares its lane: either stall on the receiver's lane is long enough to
        # miss bands. The receiver runs on a lane of its own, which also gives it a CUDA stream of
        # its own (one per partition by default): its copy out of the board's page then no longer
        # queues behind the band kernels, which matters here because the board reuses the page on
        # its own schedule. Everything up to the combiner runs at band rate on another lane; the
        # rest runs at frame rate.
        graph.partition("receiver").add(rx)
        graph.partition("capture").add(csi, image_processor, combiner)
        graph.auto_partition_remaining()

        graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))

        options = hs.CompileOptions()
        tensor_outputs = [
            ("receiver", "tensor"),
            ("csi_to_bayer", "tensor"),
            ("image_processor", "tensor"),
            ("sub_frame_combiner", "tensor"),
            ("bayer_demosaic", "output"),
        ]
        if headless:
            tensor_outputs.append(("holoviz", "output"))
        for operator_path, output_port in tensor_outputs:
            options.deployment.bind_tensor_output_device(
                operator_path=operator_path,
                output_port=output_port,
                device=hs.execution.DeviceId(cuda_device_ordinal),
            )

        plan = hs.compile(graph, options)
        if not plan.ok:
            print("Graph compilation failed:", file=sys.stderr)
            for diag in plan.diagnostics:
                print(f"  {diag}", file=sys.stderr)
            return 1

        print(
            f"Sub-frame mode: {height // sub_frame_rows} bands of {sub_frame_rows} rows, "
            f"{pages} receiver pages, queue_size {queue_size}"
        )

        hololink_dev = channel.hololink()
        hololink_dev.start()
        try:
            hololink_dev.reset()
            camera.setup_clock()
            camera.configure(camera_mode)
            camera.set_digital_gain_reg(0x4)
            if pattern is not None:
                camera.test_pattern(pattern)

            print("Calling run")
            with hs.run_async(plan) as session:
                try:
                    session.wait()
                except KeyboardInterrupt:
                    # Ctrl+C or SIGTERM: leaving the block stops the session cooperatively,
                    # which stops the camera.
                    pass
            if session.diagnostics:
                print(session.summary(), file=sys.stderr)
            failed = session.termination == hs.RunTermination.FAILED
        finally:
            hololink_dev.stop()
    finally:
        _cuda_check(cuda_driver.cuDevicePrimaryCtxRelease(cu_device))

    return 1 if failed else 0


def main():
    # Like Ctrl+C, SIGTERM (kill, docker stop) raises KeyboardInterrupt, so the session stops
    # cooperatively and the camera and the board are stopped on the way out.
    signal.signal(signal.SIGTERM, signal.default_int_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera-mode",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="IMX274 mode (default: 0 = 3840x2160@60fps)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Render into a CUDA image instead of a window",
    )
    parser.add_argument(
        "--fullscreen", action="store_true", help="Run in fullscreen mode"
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=0,
        help="Stop after this many frames (0 = run until the window closes)",
    )
    parser.add_argument(
        "--sub-frame-rows",
        type=int,
        # the camera height must be evenly divisible by the sub frame rows, by choosing a default
        # of 216, this is the case for the modes supported by the IMX274 sensor.
        default=216,
        help="Rows per sub-frame (even, and must divide the camera height evenly; "
        "0 = whole frames)",
    )
    parser.add_argument(
        "--hololink", default="192.168.0.2", help="IP address of Hololink board"
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
        help="Port number of the IBV device (with --ibv-name)",
    )
    parser.add_argument(
        "--log-level", type=int, default=20, help="Python logging level to display"
    )
    parser.add_argument(
        "--expander-configuration",
        type=int,
        default=0,
        choices=(0, 1),
        help="I2C Expander configuration",
    )
    parser.add_argument(
        "--pattern",
        type=int,
        choices=range(12),
        help="Configure to display a test pattern.",
    )
    args = parser.parse_args()
    if args.sub_frame_rows < 0:
        parser.error(
            f"--sub-frame-rows must be non-negative, got {args.sub_frame_rows}"
        )

    logging.basicConfig(level=args.log_level)

    return run(
        args.hololink,
        args.camera_mode,
        args.sub_frame_rows,
        args.ibv_name,
        args.ibv_port,
        args.expander_configuration,
        args.pattern,
        args.frame_limit,
        args.headless,
        args.fullscreen,
    )


if __name__ == "__main__":
    sys.exit(main())
