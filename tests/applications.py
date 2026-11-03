# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# HSDK5 graphs for the tests that build their own pipeline. In HSDK4 these tests
# composed a holoscan.core.Application; in HSDK5 they build an hs.Graph, compile it
# and run it. The pipeline is the one in examples/linux_imx274_player.py. The
# checks that v2.7 did on displayed frames are Python sinks (operators.py) on side
# branches; show_images() still shows the images in a HolovizOp window unless
# --headless is given.

import contextlib
import dataclasses
import functools
import logging
import time

import cuda.bindings.driver as cuda
import hololink.sensors.imx477  # noqa: F401 -- hololink_module.sensors.imx477 below
import holoscan as hs
import holoscan_vision as hv
import holoscan_visualization  # noqa: F401 -- registers "holoscan.visualization.HolovizOp"

import hololink as hololink_module

_BAYER_FORMAT_NAMES = ["bggr", "rggb", "gbrg", "grbg"]
_BAYER_FORMAT_VALUES = {name: index for index, name in enumerate(_BAYER_FORMAT_NAMES)}

# Time for a run to end after its receivers reach their frame limit.
RUN_TIMEOUT_S = 300
# How long a run that overran RUN_TIMEOUT_S gets to stop after request_stop().
STOP_TIMEOUT_S = 30


class CudaContext:
    """Retain the device's CUDA primary context for the duration of the block."""

    def __init__(self, device_ordinal=0):
        self.device_ordinal = device_ordinal
        self.cu_device = None
        self.cu_context = None

    def __enter__(self):
        (cu_result,) = cuda.cuInit(0)
        assert cu_result == cuda.CUresult.CUDA_SUCCESS
        cu_result, self.cu_device = cuda.cuDeviceGet(self.device_ordinal)
        assert cu_result == cuda.CUresult.CUDA_SUCCESS
        cu_result, self.cu_context = cuda.cuDevicePrimaryCtxRetain(self.cu_device)
        assert cu_result == cuda.CUresult.CUDA_SUCCESS
        (cu_result,) = cuda.cuCtxSetCurrent(self.cu_context)
        assert cu_result == cuda.CUresult.CUDA_SUCCESS
        return self

    def __exit__(self, *args):
        (cu_result,) = cuda.cuDevicePrimaryCtxRelease(self.cu_device)
        assert cu_result == cuda.CUresult.CUDA_SUCCESS


class TestGraph:
    """An hs.Graph and the device bindings that its tensor outputs need.

    The compiler needs a device for each connected tensor output, and refuses one
    for an output that isn't connected (PHYSICAL_TENSOR_DEVICE_BINDING_UNUSED); so
    bind each output when its flow is added."""

    __test__ = False  # Not a pytest test class.

    def __init__(self, name, cuda_device_ordinal=0):
        self.graph = hs.Graph(name)
        self.cuda_device_ordinal = cuda_device_ordinal
        self._bindings = []

    def op(self, *args, **kwargs):
        return self.graph.op(*args, **kwargs)

    def add_flow(self, *args, **kwargs):
        return self.graph.add_flow(*args, **kwargs)

    def bind(self, operator_path, output_port):
        self._bindings.append((operator_path, output_port))

    def compile(self):
        """Compile the graph; return the plan."""
        self.graph.set_default_clock(self.graph.add_clock(hs.RealtimeClock, "clock"))
        options = hs.CompileOptions()
        for operator_path, output_port in self._bindings:
            options.deployment.bind_tensor_output_device(
                operator_path=operator_path,
                output_port=output_port,
                device=hs.execution.DeviceId(self.cuda_device_ordinal),
            )
        plan = hs.compile(self.graph, options)
        assert plan.ok, f"Graph compilation failed: {list(plan.diagnostics)}"
        return plan

    def run(self, timeout_s=RUN_TIMEOUT_S):
        """Compile and run the graph; return the session's termination."""
        plan = self.compile()
        # Not `with run_async(plan) as session`: its __exit__ requests stop and then waits
        # without a timeout, so a run that doesn't stop would hang the test process there.
        # This keeps every wait bounded and raises instead.
        session = hs.run_async(plan)
        try:
            ended = session.wait(timeout=timeout_s)
        except BaseException:
            session.request_stop()
            session.wait(timeout=STOP_TIMEOUT_S)
            raise
        if not ended:
            session.request_stop()
            if not session.wait(timeout=STOP_TIMEOUT_S):
                raise AssertionError(
                    f"The run didn't end in {timeout_s} s, and didn't stop within "
                    f"{STOP_TIMEOUT_S} s of request_stop()"
                )
            raise AssertionError(f"The run didn't end in {timeout_s} s")
        # The session is terminal; an unbounded wait now returns at once and joins its
        # worker, as the context manager's exit would have.
        session.wait()
        diagnostics = [str(diagnostic) for diagnostic in session.diagnostics]
        logging.info(f"Run ended: {session.termination} {diagnostics}")
        return session.termination


class DiscardImageOp(hs.Operator):
    """Takes each image (for example from ImageProcessorOp or BayerDemosaicOp), so
    that the operators before it run for each frame."""

    input = hs.Input("input", hs.schema.Image)

    def setup(self, spec):
        spec.input(self.input, queue_depth=1)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, context):
        self.input.receive()


@dataclasses.dataclass
class SensorCamera:
    metadata: object  # hl.Metadata from hl.find_channel
    channel: object  # hl.DataChannel
    device: object  # hl.HololinkDevice
    camera: object  # hl.NativeImx274Sensor or hl.sensors.imx477.Imx477
    geometry: dict
    frame_size: int
    pixel_format: object
    bayer_format: str  # for example "rggb"
    width: int
    height: int


def _stop_started_device_on_failure(open_function):
    """If open_function started the device and then fails, stop the device: the
    caller gets no camera to stop it with."""

    @functools.wraps(open_function)
    def wrapper(*args, **kwargs):
        started = []
        try:
            return open_function(*args, _started=started, **kwargs)
        except BaseException:
            for device in started:
                device.stop()
            raise

    return wrapper


@_stop_started_device_on_failure
def open_imx274(
    channel_ip,
    camera_mode,
    pattern=None,
    expander_configuration=0,
    reset=True,
    device=None,
    ptp=False,
    digital_gain=0x4,
    _started=None,
):
    """Find the channel at channel_ip and configure its IMX274. Pass the device
    of the other channel of the same board as device, to start it only once.
    With ptp, wait for PTP to synchronize after the reset, and fail if it doesn't.
    With digital_gain None, the sensor's digital gain is left as configure() sets
    it."""
    metadata = hololink_module.find_channel(channel_ip)
    channel = hololink_module.DataChannel(metadata)
    if device is None:
        device = channel.hololink()
        device.start()
        _started.append(device)
        if reset:
            device.reset()
        if ptp and not device.ptp_synchronize():
            raise ValueError("Failed to synchronize PTP.")
    camera = hololink_module.NativeImx274Sensor(
        channel, expander_configuration=expander_configuration
    )
    if reset:
        camera.setup_clock()
    camera.configure(int(camera_mode.value))
    if digital_gain is not None:
        camera.set_digital_gain_reg(digital_gain)
    if pattern is not None:
        camera.test_pattern(pattern)
    geometry = hololink_module.resolve_generic_geometry(camera)
    frame_size = (
        geometry["start_byte"]
        + geometry["bytes_per_line"] * geometry["pixel_height"]
        + geometry["trailing_bytes"]
    )
    return SensorCamera(
        metadata=metadata,
        channel=channel,
        device=device,
        camera=camera,
        geometry=geometry,
        frame_size=frame_size,
        pixel_format=camera.get_pixel_format(),
        bayer_format=camera.get_bayer_format(),
        width=camera.get_width(),
        height=camera.get_height(),
    )


@_stop_started_device_on_failure
def open_imx477(channel_ip, camera_id=0, pattern=True, ptp=False, _started=None):
    """Find the channel at channel_ip and configure its IMX477 (sensor camera_id),
    as examples/imx477_player.py does. With pattern, show its test pattern."""
    metadata = hololink_module.find_channel(channel_ip)
    hololink_module.use_sensor(metadata, camera_id)
    channel = hololink_module.DataChannel(metadata)
    device = channel.hololink()
    device.start()
    _started.append(device)
    device.reset()
    if ptp and not device.ptp_synchronize():
        raise ValueError("Failed to synchronize PTP.")
    camera = hololink_module.sensors.imx477.Imx477(channel, camera_id=camera_id)
    camera.configure()
    if pattern:
        camera.set_pattern()
    pixel_format = camera.pixel_format()
    # Imx477.configure_converter() adds no start-byte offset or trailing bytes.
    geometry = hololink_module.resolve_generic_geometry_from_dimensions(
        int(pixel_format), camera._width, camera._height
    )
    frame_size = (
        geometry["start_byte"]
        + geometry["bytes_per_line"] * geometry["pixel_height"]
        + geometry["trailing_bytes"]
    )
    return SensorCamera(
        metadata=metadata,
        channel=channel,
        device=device,
        camera=camera,
        geometry=geometry,
        frame_size=frame_size,
        pixel_format=pixel_format,
        bayer_format=camera.bayer_format().name.lower(),
        width=camera._width,
        height=camera._height,
    )


@_stop_started_device_on_failure
def open_vb1940(
    channel_ip,
    camera_mode,
    sensor=0,
    vsync=None,
    device=None,
    ptp=False,
    _started=None,
):
    """Find sensor `sensor` of the board at channel_ip and configure its VB1940,
    as v2.7's vb1940_camera_factory did. Pass the device of the other sensor of the
    same board as device, to start and reset it only once; then this sensor's
    clock is not set up again (sensor 0's setup_clock() sets both). With ptp,
    wait for PTP to synchronize after the reset, and fail if it doesn't. With
    vsync, the board's PTP PPS output at that frequency (0: its default) drives
    the sensor's vsync."""
    metadata = hololink_module.find_channel(channel_ip)
    hololink_module.use_sensor(metadata, sensor)
    channel = hololink_module.DataChannel(metadata)
    first = device is None
    if first:
        device = channel.hololink()
        device.start()
        _started.append(device)
        device.reset()
        if ptp and not device.ptp_synchronize():
            raise ValueError("Failed to synchronize PTP.")
    if vsync is None:
        camera = hololink_module.NativeVb1940Sensor(channel)
    else:
        synchronizer = device.ptp_pps_output(frequency=vsync)
        camera = hololink_module.NativeVb1940Sensor(channel, synchronizer)
    if first:
        camera.setup_clock()  # also configures other cameras' clocks on this board
    device.write_uint32(0x8, 0x3)  # Release the sensor RESET to high
    time.sleep(100 / 1000)
    camera.get_register_32(0x0000)  # DEVICE_MODEL_ID:"S940"(ASCII code:0x53393430)
    camera.get_register_32(0x0734)  # EXT_CLOCK(25MHz = 0x017d7840)
    camera.configure(int(camera_mode.value))
    geometry = hololink_module.resolve_generic_geometry(camera)
    frame_size = (
        geometry["start_byte"]
        + geometry["bytes_per_line"] * geometry["pixel_height"]
        + geometry["trailing_bytes"]
    )
    return SensorCamera(
        metadata=metadata,
        channel=channel,
        device=device,
        camera=camera,
        geometry=geometry,
        frame_size=frame_size,
        pixel_format=camera.get_pixel_format(),
        bayer_format=camera.get_bayer_format(),
        width=camera.get_width(),
        height=camera.get_height(),
    )


@contextlib.contextmanager
def imx477_camera(channel_ip, **kwargs):
    """open_imx477() for the block; the device is stopped at the end."""
    camera = open_imx477(channel_ip, **kwargs)
    try:
        yield camera
    finally:
        camera.device.stop()


@contextlib.contextmanager
def imx274_camera(channel_ip, camera_mode, **kwargs):
    """open_imx274() for the block; the device is stopped at the end."""
    camera = open_imx274(channel_ip, camera_mode, **kwargs)
    try:
        yield camera
    finally:
        camera.device.stop()


def packed_camera(camera):
    """camera, with the HSB's packetizer packing its RAW_10 or RAW_12 pixels: the
    channel enables the program when the receiver configures it, and the frames
    have the packed geometry, for add_isp(packed=True)."""
    program = hololink_module.get_packetizer_program(camera.pixel_format)
    camera.channel.set_packetizer_program(program)
    geometry = hololink_module.resolve_packed_geometry(camera.camera)
    frame_size = (
        geometry["start_byte"]
        + geometry["bytes_per_line"] * geometry["pixel_height"]
        + geometry["trailing_bytes"]
    )
    return dataclasses.replace(camera, geometry=geometry, frame_size=frame_size)


def add_receiver(
    test_graph,
    name,
    receiver,
    camera,
    cu_context,
    frame_limit,
    device_start=None,
    **kwargs,
):
    """Add a receiver for camera. It ends the run at frame_limit, also when it is
    one of several receivers (stereo): then the last one to reach it ends the run.
    The receiver calls device_start (default: the camera's start()) after it has
    configured the data channel."""
    return test_graph.op(
        receiver,
        name,
        data_channel=camera.channel,
        frame_context=int(cu_context),
        device_start=device_start or camera.camera.start,
        device_stop=camera.camera.stop,
        frame_size=camera.frame_size,
        frame_limit=frame_limit,
        frame_limit_ends_run=True,
        **kwargs,
    )


def add_isp(
    test_graph,
    prefix,
    receiver,
    camera,
    image_processor=True,
    demosaic=True,
    packed=False,
):
    """CsiToBayerOp, then ImageProcessorOp and BayerDemosaicOp, after receiver.
    With packed, PackedFormatConverterOp takes CsiToBayerOp's place, for frames
    that the HSB's packetizer packed (see packed_camera()).

    Returns a dict of the operator handles ("csi", the converter;
    "image_processor"; "demosaic"). The last image output (demosaic.output, or
    the one before it) is bound to the device; connect it to a sink with
    image_flow(): BayerDemosaicOp fails the run (kBackpressured) when one of its
    output queues is full."""
    handles = {}
    if packed:
        csi_name = f"{prefix}packed_format_converter"
        csi = test_graph.op(
            hololink_module.operators.PackedFormatConverterOp,
            csi_name,
            cuda_device_ordinal=test_graph.cuda_device_ordinal,
            geometry=camera.geometry,
            bayer_format=camera.bayer_format,
            # The receivers emit the whole frame, from its start_byte header on.
            header_included=True,
        )
    else:
        csi_name = f"{prefix}csi_to_bayer"
        csi = test_graph.op(
            hololink_module.operators.CsiToBayerOp,
            csi_name,
            cuda_device_ordinal=test_graph.cuda_device_ordinal,
            geometry=camera.geometry,
            bayer_format=camera.bayer_format,
        )
    test_graph.add_flow(receiver.tensor, csi.in_tensor, queue_depth=4)
    test_graph.add_flow(receiver.metadata, csi.in_metadata, queue_depth=4)
    test_graph.bind(str(receiver.name), "tensor")
    test_graph.bind(csi_name, "tensor")
    handles["csi"] = csi
    if not image_processor:
        return handles
    isp = test_graph.op(
        hololink_module.operators.ImageProcessorOp,
        f"{prefix}image_processor",
        pixel_format=camera.pixel_format,
        bayer_format=_BAYER_FORMAT_VALUES[camera.bayer_format],
        # Optical black value for imx274 is 50
        optical_black=50,
        cuda_device_ordinal=test_graph.cuda_device_ordinal,
        # Sizes the output tensor pool's slots to the image, not to the 64 MiB ceiling.
        pixel_width=camera.width,
        pixel_height=camera.height,
    )
    test_graph.add_flow(csi.tensor, isp.in_tensor, queue_depth=4)
    test_graph.add_flow(csi.metadata, isp.in_metadata, queue_depth=4)
    test_graph.bind(f"{prefix}image_processor", "tensor")
    handles["image_processor"] = isp
    if not demosaic:
        return handles
    handles["demosaic"] = add_demosaic(test_graph, prefix, isp, camera)
    return handles


def add_demosaic(test_graph, prefix, source, camera):
    """BayerDemosaicOp on source's image output (ImageProcessorOp or CsiToBayerOp,
    whose output add_isp has bound). Its output is bound to the device; connect it
    to a sink with image_flow()."""
    bayer_demosaic = test_graph.op(
        hv.operators.BayerDemosaicOp,
        f"{prefix}bayer_demosaic",
        bayer_grid_pos=camera.bayer_format,
        generate_alpha=True,
        alpha_value=65535,
    )
    test_graph.add_flow(source.tensor, bayer_demosaic.input, queue_depth=4)
    test_graph.bind(f"{prefix}bayer_demosaic", "output")
    return bayer_demosaic


@dataclasses.dataclass
class Image:
    """An image output to show: the output port, its size and its HolovizOp format
    (BayerDemosaicOp: "rgba16_unorm"; ArgusIspOp: "rgb8_unorm")."""

    port: object
    width: int
    height: int
    format: str = "rgba16_unorm"


def show_images(test_graph, name, images, headless, title):
    """Show images side by side in a HolovizOp window, as v2.7's tests did; with
    headless, take them with DiscardImageOp instead (HSDK5's HolovizOp renders
    without a window only into CUDA memory, for another operator to take). Either
    way the images are connected with image_flow(), so the display never holds up
    the pipeline."""
    if headless:
        for index, image in enumerate(images):
            discard = test_graph.op(DiscardImageOp, f"{name}_discard_{index}")
            image_flow(test_graph, image.port, discard.input)
        return
    count = len(images)
    layers = [
        {
            "port_name": f"video_{index}",
            "kind": "image",
            "format": image.format,
            "max_width": image.width,
            "max_height": image.height,
            "view": {
                "offset_x": index / count,
                "offset_y": 0.0,
                "scale_x": 1.0 / count,
                "scale_y": 1.0,
            },
        }
        for index, image in enumerate(images)
    ]
    # A window no wider than 1280 pixels, with the images' aspect ratio.
    total_width = sum(image.width for image in images)
    height = max(image.height for image in images)
    window_width = min(1280, total_width)
    window_height = max(1, round(height * window_width / total_width))
    visualizer = test_graph.op(
        "holoscan.visualization.HolovizOp",
        name,
        layers=layers,
        sink={
            "kind": "window",
            "framebuffer_srgb": True,
            "title": title,
            "width": window_width,
            "height": window_height,
            "fullscreen": False,
            "close_on_escape": True,
        },
        # One frame per received frame.
        trigger={"kind": "layer"},
    )
    for index, image in enumerate(images):
        image_flow(test_graph, image.port, getattr(visualizer, f"video_{index}"))


def image_flow(test_graph, source, sink, queue_depth=1):
    """Connect an image output to a check sink. The sink gets the newest images
    and the pipeline never waits for it: a full queue drops its oldest image."""
    test_graph.add_flow(
        source,
        sink,
        queue_policy=hs.QueuePolicy.DROP_OLDEST,
        queue_depth=queue_depth,
    )
