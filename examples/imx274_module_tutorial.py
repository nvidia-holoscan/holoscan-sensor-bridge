# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""IMX274 camera player built on the hololink_module API.

Runnable companion to the "IMX274 Module Tutorial" user guide. It plays one
IMX274 camera behind a Holoscan Sensor Bridge board through the hardware (RoCE)
receiver and displays the video with Holoviz. Configuration is intentionally
hard-coded so the tutorial can focus on the API rather than on argument parsing.
The C++ version is examples/imx274_module_tutorial.cpp; the CLI-driven siblings
are examples/module_imx274_player.cpp and examples/module_quad_imx274_player.py.
"""

import signal
import sys

import hololink_module.operators
import hololink_module.sensors.imx274
import holoscan as hs
import holoscan_vision as hv
import holoscan_visualization as hz  # noqa: F401 -- registers "holoscan.visualization.HolovizOp"
from cuda.bindings import driver as cuda_driver
from hololink_module.sensors.imx274 import imx274_mode as adapter_imx274_mode

import hololink_module

# The board announces itself over bootp; this is the peer IP we wait for.
HOLOLINK_IP = "192.168.0.2"
# Seconds to wait for that announcement before giving up.
DISCOVERY_TIMEOUT = 30.0
CAMERA_MODE = adapter_imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS


def _cuda_check(result):
    err = result[0]
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        _, description = cuda_driver.cuGetErrorString(err)
        raise RuntimeError(f"CUDA error: {description}")
    if len(result) == 2:
        return result[1]
    return result[1:]


def build_graph(cu_context, cu_device_ordinal, metadata, camera):
    """Returns the player's graph and the compile options that bind its outputs to the GPU."""
    camera.set_mode(CAMERA_MODE)
    # The sensor's CSI-2 pixel and Bayer formats are plain enum.Enum members: use .value.
    pixel_format = camera.pixel_format()
    bayer_format = camera.bayer_format()
    bayer_name = bayer_format.name.lower()

    graph = hs.Graph("imx274_module_tutorial")

    converter = hololink_module.CsiToBayerConverter()
    # Configuring the converter tells us how big each received frame is.
    camera.configure_converter(converter)
    frame_size = converter.get_csi_length()
    geometry = converter.geometry()
    csi = graph.op(
        hololink_module.operators.CsiToBayerOp,
        "csi_to_bayer",
        cuda_device_ordinal=cu_device_ordinal,
        geometry=geometry,
        bayer_format=bayer_name,
    )

    receiver = graph.op(
        hololink_module.operators.RoceReceiverOp,
        "receiver",
        metadata=metadata,
        frame_context=int(cu_context),
        frame_size=frame_size,
        device_start=camera.start,
        device_stop=camera.stop,
    )

    image_processor = graph.op(
        hololink_module.operators.ImageProcessorOp,
        "image_processor",
        pixel_format=pixel_format.value,
        bayer_format=bayer_format.value,
        optical_black=50,  # IMX274 optical black
        cuda_device_ordinal=cu_device_ordinal,
        pixel_width=geometry["pixel_width"],
        pixel_height=geometry["pixel_height"],
    )

    demosaic = graph.op(
        hv.operators.BayerDemosaicOp,
        "bayer_demosaic",
        bayer_grid_pos=bayer_name,
        generate_alpha=True,
        alpha_value=65535,
    )

    # A 1920x1080 window titled "Holoviz" with one image layer for the camera's 16-bit RGBA
    # frames, drawn 60 times a second.
    visualizer = graph.op(
        "holoscan.visualization.HolovizOp",
        "holoviz",
        layers=[
            {
                "port_name": "video",
                "kind": "image",
                "format": "rgba16_unorm",
                "max_width": camera._width,
                "max_height": camera._height,
            }
        ],
        sink={
            "kind": "window",
            # The demosaiced data is linear sensor output: encode to sRGB on write.
            "framebuffer_srgb": True,
            "title": "Holoviz",
            "width": 1920,
            "height": 1080,
            "close_on_escape": True,
        },
        trigger={"kind": "clock", "period_ns": round(1e9 / 60.0)},
    )

    graph.add_flow(receiver.tensor, csi.in_tensor, queue_depth=2)
    graph.add_flow(receiver.metadata, csi.in_metadata, queue_depth=2)
    graph.add_flow(csi.tensor, image_processor.in_tensor, queue_depth=2)
    graph.add_flow(csi.metadata, image_processor.in_metadata, queue_depth=2)
    graph.add_flow(image_processor.tensor, demosaic.input, queue_depth=2)
    graph.add_flow(
        demosaic.output,
        visualizer.video,
        queue_policy=hs.QueuePolicy.DROP_OLDEST,
        queue_depth=1,
    )

    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))

    options = hs.CompileOptions()
    for operator_path in ("receiver", "csi_to_bayer", "image_processor"):
        options.deployment.bind_tensor_output_device(
            operator_path=operator_path,
            output_port="tensor",
            device=hs.execution.DeviceId(cu_device_ordinal),
        )
    options.deployment.bind_tensor_output_device(
        operator_path="bayer_demosaic",
        output_port="output",
        device=hs.execution.DeviceId(cu_device_ordinal),
    )
    return graph, options


def main():
    # Like Ctrl+C, SIGTERM (kill, docker stop) raises KeyboardInterrupt, so the session stops
    # cooperatively and the camera and the board are stopped on the way out.
    signal.signal(signal.SIGTERM, signal.default_int_handler)

    # CUDA bring-up
    _cuda_check(cuda_driver.cuInit(0))
    cu_device_ordinal = 0
    cu_device = _cuda_check(cuda_driver.cuDeviceGet(cu_device_ordinal))
    cu_context = _cuda_check(cuda_driver.cuDevicePrimaryCtxRetain(cu_device))
    _cuda_check(cuda_driver.cuCtxSetCurrent(cu_context))

    # Find the board via bootp enumeration and get its metadata.
    adapter = hololink_module.Adapter.get_adapter()
    metadata = adapter.wait_for_channel(HOLOLINK_IP, DISCOVERY_TIMEOUT)

    # The sensor driver is linked directly into the application; it is not a
    # service. It resolves the services it needs from the enumeration metadata.
    camera = hololink_module.sensors.imx274.Imx274Cam(metadata)

    # The control plane is a versioned service fetched by metadata.
    hololink = hololink_module.HololinkInterfaceV1.get_service(metadata)
    # start() opens the control-plane socket; without it, no device I/O works.
    hololink.start()
    try:
        # The framework leaves device state alone unless the application asks;
        # reset() drives the board into a known state.
        hololink.reset()
        # configure() writes device registers, so it must run after reset().
        camera.configure(CAMERA_MODE)
        camera.set_digital_gain_reg(0x4)

        graph, options = build_graph(cu_context, cu_device_ordinal, metadata, camera)
        plan = hs.compile(graph, options)
        if not plan.ok:
            print("Graph compilation failed:", file=sys.stderr)
            for diag in plan.diagnostics:
                print(f"  {diag}", file=sys.stderr)
            return 1

        # Run until the window closes or Ctrl+C, then report how the run ended.
        with hs.run_async(plan) as session:
            try:
                session.wait()
            except KeyboardInterrupt:
                # Leaving the block stops the session cooperatively, which stops the camera.
                pass
        if session.diagnostics:
            print(session.summary(), file=sys.stderr)
        failed = session.termination == hs.RunTermination.FAILED
    finally:
        # Stop streaming data and close up sockets.
        hololink.stop()
        release_result = cuda_driver.cuDevicePrimaryCtxRelease(cu_device)
    # Checked out here so that a teardown failure doesn't displace an exception
    # coming out of the run.
    _cuda_check(release_result)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
