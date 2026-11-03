# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Hardware-free checks that the hololink and hololink_module receivers take their live objects
# as Holoscan constructor handles from Python: the operator rows declare them, the bound objects
# export them, any callable serves as a device callback, and the graph compiles. No board is
# needed: compiling replays the operator constructors, which never touch a device.

import holoscan as hs
import pytest

import hololink as hl


class _Camera:
    """A camera stand-in whose start/stop calls are recorded."""

    def __init__(self):
        self.calls = []

    def start(self):
        self.calls.append("start")

    def stop(self):
        self.calls.append("stop")


def _data_channel():
    # Enough enumeration metadata for DataChannel's constructor; nothing contacts the board.
    metadata = hl.Metadata()
    for name, value in {
        "serial_number": "constructor-handle-test",
        "peer_ip": "127.0.0.1",
        "control_port": 8192,
        "vp_mask": 0,
        "data_plane": 0,
        "sensor": 0,
        "sif_address": 0,
        "vp_address": 0,
        "hif_address": 0,
        "fpga_uuid": "00000000-0000-0000-0000-000000000000",
    }.items():
        metadata[name] = value
    return hl.DataChannel(metadata)


def _compile(graph, tensor_ports):
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    options = hs.CompileOptions()
    for operator_path, output_port in tensor_ports:
        options.deployment.bind_tensor_output_device(
            operator_path=operator_path,
            output_port=output_port,
            device=hs.execution.DeviceId(0),
        )
    plan = hs.compile(graph, options)
    assert plan.ok, plan
    return plan


def test_receiver_rows_declare_their_constructor_handles():
    handles = hl.operators.LinuxReceiverOp.constructor_handles
    assert handles["data_channel"].type_name == "hololink.DataChannel"
    assert handles["data_channel"].required
    for callback in ("device_start", "device_stop"):
        assert handles[callback].type_name == "holoscan.VoidCallback"
        assert not handles[callback].required


def test_receiver_takes_the_channel_and_camera_callbacks():
    graph = hs.Graph("receiver-constructor-handles")
    camera = _Camera()
    receiver = graph.op(
        hl.operators.LinuxReceiverOp,
        "receiver",
        data_channel=_data_channel(),
        device_start=camera.start,
        device_stop=camera.stop,
        frame_context=0,
        frame_size=1024,
    )
    assert "data_channel=<hololink.DataChannel: process-local>" in repr(receiver)
    _compile(graph, [("receiver", "tensor")])
    # Compiling builds the operator but never starts it.
    assert camera.calls == []


def test_receiver_takes_frame_limit_ends_run():
    graph = hs.Graph("receiver-frame-limit-ends-run")
    camera = _Camera()
    graph.op(
        hl.operators.LinuxReceiverOp,
        "receiver",
        data_channel=_data_channel(),
        device_start=camera.start,
        device_stop=camera.stop,
        frame_context=0,
        frame_size=1024,
        frame_limit=3,
        frame_limit_ends_run=True,
    )
    _compile(graph, [("receiver", "tensor")])


def test_receiver_rejects_a_plain_object_as_its_channel():
    graph = hs.Graph("receiver-plain-object-channel")
    with pytest.raises(hs.errors.IncompatibleError, match="PROTOCOL_MISSING"):
        graph.op(
            hl.operators.LinuxReceiverOp,
            "receiver",
            data_channel=object(),
            frame_context=0,
            frame_size=1024,
        )


def test_receiver_rejects_data_channels_as_its_channel():
    # hololink.DataChannels is its own handle type (GpuRoceTransceiverOp's extra_data_channels),
    # not a DataChannel.
    graph = hs.Graph("receiver-data-channels-as-channel")
    with pytest.raises(hs.errors.IncompatibleError):
        graph.op(
            hl.operators.LinuxReceiverOp,
            "receiver",
            data_channel=hl.DataChannels([_data_channel()]),
            frame_context=0,
            frame_size=1024,
        )


@pytest.mark.skipif(
    not hasattr(hl.operators, "GpuRoceTransceiverOp"),
    reason="GpuRoceTransceiverOp is built only with DOCA GPUNetIO",
)
def test_gpu_roce_transceiver_takes_the_extra_data_channels():
    handles = hl.operators.GpuRoceTransceiverOp.constructor_handles
    assert handles["extra_data_channels"].type_name == "hololink.DataChannels"
    assert not handles["extra_data_channels"].required
    graph = hs.Graph("gpu-roce-transceiver-extra-data-channels")
    receiver = graph.op(
        hl.operators.GpuRoceTransceiverOp,
        "receiver",
        data_channel=_data_channel(),
        extra_data_channels=hl.DataChannels([_data_channel()]),
        frame_context=0,
        frame_size=1024,
        tx_ibv_qp="2,4",
    )
    assert "extra_data_channels=<hololink.DataChannels" in repr(receiver)


def test_receiver_requires_its_channel():
    graph = hs.Graph("receiver-without-channel")
    with pytest.raises(hs.errors.InvalidArgumentError, match="REQUIRED"):
        graph.op(
            hl.operators.LinuxReceiverOp, "receiver", frame_context=0, frame_size=1024
        )


def test_module_receiver_takes_the_metadata_and_camera_callbacks():
    hololink_module = pytest.importorskip("hololink_module")
    operators = pytest.importorskip("hololink_module.operators")
    graph = hs.Graph("module-receiver-constructor-handles")
    camera = _Camera()
    graph.op(
        operators.LinuxReceiverOp,
        "receiver",
        metadata=hololink_module.EnumerationMetadata(),
        device_start=camera.start,
        device_stop=camera.stop,
        frame_context=0,
        frame_size=1024,
    )
    _compile(graph, [("receiver", "tensor")])
    assert camera.calls == []


def test_hsb_controller_takes_its_metadata_and_factories():
    hololink_module = pytest.importorskip("hololink_module")
    operators = pytest.importorskip("hololink_module.operators")

    class _NoSensorFactory(hololink_module.SensorFactory):
        def new_sensor(self, metadata):
            return None

    graph = hs.Graph("hsb-controller-constructor-handles")
    graph.op(
        operators.HsbControllerOp,
        "controller",
        metadata=hololink_module.EnumerationMetadata(),
        sensor_factory=_NoSensorFactory(),
        network_receiver_factory=hololink_module.make_linux_network_receiver_factory(),
        frame_context=0,
        frame_size=1024,
    )
    _compile(graph, [("controller", "tensor")])


def test_crc_rows_declare_their_crc_channel():
    for row in (
        hl.operators.ComputeCrcOp,
        hl.operators.ComputeCrcImageOp,
        hl.operators.CheckCrcOp,
    ):
        handle = row.constructor_handles["crc_channel"]
        assert handle.type_name == "hololink.CrcChannel"
        assert handle.required


def test_crc_operators_share_a_crc_channel():
    graph = hs.Graph("crc-channel")
    checked = []
    crc_channel = hl.CrcChannel(
        check_crc=lambda computed, received, frame: checked.append(frame)
    )
    receiver = graph.op(
        hl.operators.LinuxReceiverOp,
        "receiver",
        data_channel=_data_channel(),
        frame_context=0,
        frame_size=1024,
    )
    compute_crc = graph.op(
        hl.operators.ComputeCrcOp,
        "compute_crc",
        crc_channel=crc_channel,
        frame_size=1024,
    )
    check_crc = graph.op(hl.operators.CheckCrcOp, "check_crc", crc_channel=crc_channel)
    graph.add_flow(receiver.tensor, compute_crc.in_tensor, queue_depth=4)
    graph.add_flow(receiver.metadata, compute_crc.in_metadata, queue_depth=4)
    graph.add_flow(
        compute_crc.metadata,
        check_crc.in_metadata,
        queue_depth=4,
        queue_policy=hs.QueuePolicy.DROP_OLDEST,
    )
    assert "crc_channel=<hololink.CrcChannel: process-local>" in repr(check_crc)
    _compile(graph, [("receiver", "tensor")])
    # Compiling builds the operators but never runs them.
    assert checked == []


def test_packed_converter_takes_the_packed_geometry():
    # The CoE tests' legs with the packetizer (applications.packed_camera()):
    # LinuxCoeReceiverOp, then PackedFormatConverterOp in add_isp(packed=True).
    # Imported here: applications needs CUDA and the vision and visualization
    # modules, which the other tests in this file don't.
    applications = pytest.importorskip("applications")
    width, height = 32, 16
    bytes_per_line = 64  # 32 RAW_10 pixels packed 3 to 4 bytes, on 64-byte lines
    camera = applications.SensorCamera(
        metadata=None,
        channel=_data_channel(),
        device=None,
        camera=_Camera(),
        geometry={
            "start_byte": bytes_per_line,
            "bytes_per_line": bytes_per_line,
            "pixel_width": width,
            "pixel_height": height,
            "pixel_format": "raw_10",
            "trailing_bytes": 0,
        },
        frame_size=bytes_per_line * (height + 1),
        pixel_format=int(hl.PixelFormat.RAW_10),
        bayer_format="rggb",
        width=width,
        height=height,
    )
    test_graph = applications.TestGraph("packed")
    receiver = applications.add_receiver(
        test_graph,
        "receiver",
        hl.operators.LinuxCoeReceiverOp,
        camera,
        0,
        10,
        coe_interface="lo",
        pixel_width=width,
        coe_channel=1,
    )
    isp = applications.add_isp(test_graph, "", receiver, camera, packed=True)
    discard = test_graph.op(applications.DiscardImageOp, "discard")
    applications.image_flow(test_graph, isp["demosaic"].output, discard.input)
    test_graph.compile()


def test_image_crc_takes_an_image_and_its_metadata():
    # ComputeCrcImageOp takes an image (here CsiToBayerOp's) and the frame's
    # metadata, as tests/operators.py's add_image_crc_check() connects it.
    graph = hs.Graph("image-crc")
    crc_channel = hl.CrcChannel(check_crc=lambda computed, received, frame: None)
    receiver = graph.op(
        hl.operators.LinuxReceiverOp,
        "receiver",
        data_channel=_data_channel(),
        frame_context=0,
        frame_size=1024,
    )
    csi = graph.op(
        hl.operators.CsiToBayerOp,
        "csi_to_bayer",
        geometry={
            "start_byte": 0,
            "bytes_per_line": 40,
            "pixel_width": 32,
            "pixel_height": 16,
            "pixel_format": "raw_10",
            "trailing_bytes": 0,
        },
        bayer_format="rggb",
    )
    compute_crc = graph.op(
        hl.operators.ComputeCrcImageOp,
        "compute_crc",
        crc_channel=crc_channel,
        frame_size=32 * 16 * 2,
    )
    check_crc = graph.op(hl.operators.CheckCrcOp, "check_crc", crc_channel=crc_channel)
    graph.add_flow(receiver.tensor, csi.in_tensor, queue_depth=4)
    graph.add_flow(receiver.metadata, csi.in_metadata, queue_depth=4)
    graph.add_flow(
        csi.tensor,
        compute_crc.in_tensor,
        queue_policy=hs.QueuePolicy.DROP_OLDEST,
        queue_depth=2,
    )
    graph.add_flow(
        csi.metadata,
        compute_crc.in_metadata,
        queue_policy=hs.QueuePolicy.DROP_OLDEST,
        queue_depth=4,
    )
    graph.add_flow(compute_crc.metadata, check_crc.in_metadata, queue_depth=4)
    _compile(graph, [("receiver", "tensor"), ("csi_to_bayer", "tensor")])
