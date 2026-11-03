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
#
# Stereo VB1940 cameras (left/right, sensors 0/1) plus the onboard IMU (sensor 2), all via
# RoceReceiverOp, synchronized by one PTP-PPS VSYNC. Camera/CSI/demosaic/Holoviz wiring matches
# the already-proven examples/single_network_stereo_vb1940_player.py template exactly
# (NativeVb1940Sensor, not the old app's own pure-Python Vb1940Cam driver -- NativeVb1940Sensor
# is the camera driver every other VB1940 app this porting effort uses, already handles VSYNC
# attachment in its own C++ binding with no Python-side Synchronizable subclassing needed). The
# IMU leg reuses vb1940_imu_player.py's own ProfilerOp/pairing design.
#
#   RoceReceiverOp(left)  -> CsiToBayerOp -> ImageProcessorOp -> BayerDemosaicOp -> HolovizOp
#   RoceReceiverOp(right) -> CsiToBayerOp -> ImageProcessorOp -> BayerDemosaicOp -> HolovizOp
#   RoceReceiverOp(imu)   -> ProfilerOp
#
# Unlike single_network_stereo_vb1940_player.py's own --trigger flag (VSYNC only when passed),
# this app's old version always runs in VSYNC trigger mode unconditionally (no --trigger flag
# exists in the old app at all) -- preserved as-is, not unified with the sibling's own default.
#
# hololink.sensors.vb1940.Imu's own rename_metadata=lambda kwarg (used by the old app so all
# three legs' per-frame metadata could coexist in one shared dict) has no HSDK5 equivalent
# needed: Sample.metadata is per-port, not a shared pipeline-wide dict, so there's no key
# collision to avoid between legs in the first place -- dropped, not replaced.
#
# --frame-limit is the frame_limit parameter of all three receivers, as the old app's three
# CountConditions were (HSDK5 has none). The camera receivers also set frame_limit_ends_run, so
# the run ends once both have delivered that many frames: the IMU, which sends about 80 frames/s
# against the cameras' 30, has usually stopped by then; with slower IMU rates, its last frames
# are cut.

import argparse
import logging
import sys
import time
from datetime import timedelta

import capture_metadata
import cuda.bindings.driver as cuda_driver
import cupy as cp

# Binds hl.sensors.vb1940 as an attribute; see vb1940_imu_player.py's own identical import for
# why this is needed.
import hololink.sensors.vb1940  # noqa: F401
import holoscan as hs
import holoscan_vision as hv
import holoscan_visualization as hz  # noqa: F401 -- registers "holoscan.visualization.HolovizOp"

import hololink as hl

MS_PER_SEC = 1000.0
US_PER_SEC = 1000.0 * MS_PER_SEC
NS_PER_SEC = 1000.0 * US_PER_SEC
SEC_PER_NS = 1.0 / NS_PER_SEC

BYTES_PER_SAMPLE = 16

_BAYER_FORMAT_NAMES = ["bggr", "rggb", "gbrg", "grbg"]
_BAYER_FORMAT_VALUES = {name: index for index, name in enumerate(_BAYER_FORMAT_NAMES)}

# Matches hololink::operators::kMetadataPairingWindow (capture_metadata_utils.hpp): the
# receiver stamps both ports with an identical capture_time, so this bounded-window
# OnAligned+Exact{} pairs them precisely.
_METADATA_PAIRING_WINDOW = timedelta(microseconds=500)


class ProfilerOp(hs.Operator):
    """Decode and log raw IMU samples, correlated with the receiver's own frame metadata."""

    in_tensor = hs.Input("in_tensor", hs.payload.Tensor)
    in_metadata = hs.Input("in_metadata", capture_metadata.CaptureMetadata)
    samples_per_frame = hs.Param(10)

    def setup(self, spec):
        spec.input(self.in_tensor, queue_depth=4)
        spec.input(self.in_metadata, queue_depth=4)

    def contract(self):
        return (
            hs.Contract()
            .trigger(
                hs.OnAligned(
                    ports=[self.in_tensor, self.in_metadata],
                    window=_METADATA_PAIRING_WINDOW,
                )
            )
            .input(self.in_tensor, hs.Exact())
            .input(self.in_metadata, hs.Exact())
        )

    def compute(self, context):
        tensor_sample = self.in_tensor.receive()
        metadata_sample = self.in_metadata.receive()
        if tensor_sample.is_missing or metadata_sample.is_missing:
            return

        # A received HSDK5 tensor is read-only, and HSDK5 exports it only as versioned DLPack
        # on a real CUDA stream (cupy's default legacy stream is refused with kNotSupported).
        # Import it without a copy on a private non-blocking stream, and synchronize before
        # the array is used.
        with cp.cuda.Stream(non_blocking=True) as stream:
            capsule = tensor_sample.data.__dlpack__(
                stream=stream.ptr,
                max_version=(1, 0),  # Versioned DLPack
                copy=False,  # No pixel-buffer copy
            )
            cp_frame = cp.from_dlpack(capsule)
            stream.synchronize()
        # The HSDK5 Deserializer takes bytes only.
        data = cp_frame.astype(cp.uint8, copy=False).get().tobytes()
        deserializer = hl.Deserializer(data)
        for _ in range(self.samples_per_frame):
            timestamp_ns = deserializer.next_uint32_le()
            timestamp_s = deserializer.next_uint32_le()
            timestamp_s += timestamp_ns * SEC_PER_NS
            tag = deserializer.next_uint8()
            sensor_data = ["%02X" % x for x in deserializer.next_buffer(6)]
            ff = deserializer.next_uint8()
            logging.info(f"{timestamp_s=} {tag=:#x} {sensor_data=}")
            if ff != 0xFF:
                raise RuntimeError(
                    f"Unexpected IMU frame terminator {ff:#x}; expected 0xFF."
                )

        frame = metadata_sample.data
        imu_image_timestamp_s = frame.timestamp_s + frame.timestamp_ns * SEC_PER_NS
        logging.info(
            f"imu_frame_number={frame.frame_number} {imu_image_timestamp_s=:0.6f}"
        )


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


def _cuda_check(result):
    err = result[0]
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        _, description = cuda_driver.cuGetErrorString(err)
        raise RuntimeError(f"CUDA error: {description}")
    if len(result) == 2:
        return result[1]
    return result[1:]


def run(
    hololink_ip,
    camera_mode,
    ibv_name,
    ibv_port,
    frequency,
    exposure,
    gain,
    accelerometer_rate,
    gyroscope_rate,
    frame_limit,
    headless,
):
    cuda_device_ordinal = 0
    tensor_names = ["left", "right"]

    _cuda_check(cuda_driver.cuInit(0))
    cu_device = _cuda_check(cuda_driver.cuDeviceGet(cuda_device_ordinal))
    cu_context = _cuda_check(cuda_driver.cuDevicePrimaryCtxRetain(cu_device))
    cuda_driver.cuCtxSetCurrent(cu_context)

    # Separate connection metadata for left/right/imu (sensors 0/1/2), each borrowing the
    # data-plane configuration found on the shared interface.
    channels = []
    for sensor_id in (0, 1, 2):
        metadata = hl.find_channel(hololink_ip)
        hl.use_sensor(metadata, sensor_id)
        channels.append(hl.DataChannel(metadata))
    channel_left, channel_right, channel_imu = channels

    hololink_dev = channel_imu.hololink()
    if hololink_dev is not channel_left.hololink():
        raise RuntimeError("Data channels are on different HSB devices.")
    if hololink_dev is not channel_right.hololink():
        raise RuntimeError("Data channels are on different HSB devices.")
    hololink_dev.start()

    try:
        hololink_dev.reset()
        logging.debug("Waiting for PTP sync.")
        if not hololink_dev.ptp_synchronize():
            raise ValueError("Failed to synchronize PTP.")
        logging.debug("PTP synchronized.")

        # Unlike single_network_stereo_vb1940_player.py's --trigger flag, this app always runs
        # in VSYNC trigger mode -- matches the old app's own unconditional behavior.
        vsync = hololink_dev.ptp_pps_output(frequency)
        cameras = [
            hl.NativeVb1940Sensor(channel_left, vsync),
            hl.NativeVb1940Sensor(channel_right, vsync),
        ]

        # Manual sensor-reset sequence -- both sensors share one external clock, so their reset
        # pins are released together (0x3 = both bits set).
        hololink_dev.write_uint32(0x8, 0x0)
        cameras[0].setup_clock()  # this also sets the other camera's clock
        hololink_dev.write_uint32(0x8, 0x3)
        time.sleep(100 / 1000)

        for camera in cameras:
            camera.get_register_32(
                0x0000
            )  # DEVICE_MODEL_ID:"S940"(ASCII code:0x53393430)
            camera.get_register_32(0x0734)  # EXT_CLOCK(25MHz = 0x017d7840)
            camera.configure(camera_mode)
            camera.set_analog_gain_reg(gain)
            camera.set_exposure_reg(exposure)

        geometries = [hl.resolve_generic_geometry(camera) for camera in cameras]
        frame_sizes = [
            geometry["start_byte"]
            + geometry["bytes_per_line"] * geometry["pixel_height"]
            + geometry["trailing_bytes"]
            for geometry in geometries
        ]
        if frame_sizes[0] != frame_sizes[1]:
            raise RuntimeError("Both cameras must have the same frame size")

        imu = hl.sensors.vb1940.Imu(hololink_dev)
        imu_samples_per_frame = 10
        imu.configure(
            samples_per_frame=imu_samples_per_frame,
            accelerometer_rate=accelerometer_rate,
            gyroscope_rate=gyroscope_rate,
        )

        graph = hs.Graph("vb1940_stereo_imu_player")

        layers = []
        for i, camera in enumerate(cameras):
            layers.append(
                {
                    "port_name": f"video_{tensor_names[i]}",
                    "kind": "image",
                    "format": "rgba16_unorm",
                    "max_width": camera.get_width(),
                    "max_height": camera.get_height(),
                    "view": {
                        "offset_x": 0.5 * i,
                        "offset_y": 0.0,
                        "scale_x": 0.5,
                        "scale_y": 1.0,
                    },
                }
            )

        if headless:
            # Renders into a CUDA image on HolovizOp's output (consumed by DiscardImageOp). The
            # CUDA sink has no framebuffer_srgb: its sRGB format encodes on write instead.
            sink = {
                "kind": "cuda",
                "width": 600,
                "height": 200,
                "format": "rgba8_srgb",
            }
        else:
            sink = {
                "kind": "window",
                "title": "VB1940 player",
                "width": 600,
                "height": 200,
                "fullscreen": False,
                "close_on_escape": True,
                "framebuffer_srgb": True,
            }
        viz = graph.op(
            "holoscan.visualization.HolovizOp",
            "holoviz",
            layers=layers,
            sink=sink,
            # One frame per received frame.
            trigger={"kind": "layer"},
        )
        if headless:
            discard = graph.op(DiscardImageOp, "discard")
            graph.add_flow(
                viz.output,
                discard.input,
                queue_policy=hs.QueuePolicy.DROP_OLDEST,
                queue_depth=1,
            )

        tensor_output_operators = []

        for i, camera in enumerate(cameras):
            name = tensor_names[i]
            geometry = geometries[i]
            bayer_format = camera.get_bayer_format()
            pixel_format = camera.get_pixel_format()

            rx = graph.op(
                hl.operators.RoceReceiverOp,
                f"rx_{name}",
                data_channel=channels[i],
                frame_context=int(cu_context),
                device_start=camera.start,
                device_stop=camera.stop,
                frame_size=frame_sizes[i],
                ibv_name=ibv_name,
                ibv_port=ibv_port,
                frame_limit=frame_limit or 0,
                frame_limit_ends_run=True,
            )

            csi = graph.op(
                hl.operators.CsiToBayerOp,
                f"csi_to_bayer_{name}",
                cuda_device_ordinal=cuda_device_ordinal,
                geometry=geometry,
                bayer_format=bayer_format,
            )
            graph.add_flow(rx.tensor, csi.in_tensor, queue_depth=4)
            graph.add_flow(rx.metadata, csi.in_metadata, queue_depth=4)

            image_processor = graph.op(
                hl.operators.ImageProcessorOp,
                f"image_processor_{name}",
                pixel_format=pixel_format,
                pixel_width=geometry["pixel_width"],
                pixel_height=geometry["pixel_height"],
                bayer_format=_BAYER_FORMAT_VALUES[bayer_format],
                # Optical black value for vb1940 is 8 for RAW10 (matches this app's own
                # HSDK4-era default).
                optical_black=8,
                cuda_device_ordinal=cuda_device_ordinal,
            )
            graph.add_flow(csi.tensor, image_processor.in_tensor, queue_depth=4)
            graph.add_flow(csi.metadata, image_processor.in_metadata, queue_depth=4)

            # image_processor.metadata is not connected: nothing uses it, and an
            # output that is not connected has no queue that can fill.

            demosaic = graph.op(
                hv.operators.BayerDemosaicOp,
                f"bayer_demosaic_{name}",
                bayer_grid_pos=bayer_format,
                generate_alpha=True,
                alpha_value=65535,
            )
            graph.add_flow(image_processor.tensor, demosaic.input, queue_depth=4)
            graph.add_flow(
                demosaic.output,
                getattr(viz, f"video_{name}"),
                queue_policy=hs.QueuePolicy.DROP_OLDEST,
                queue_depth=1,
            )

            tensor_output_operators.append(rx.name)
            tensor_output_operators.append(csi.name)
            tensor_output_operators.append(image_processor.name)

        imu_rx = graph.op(
            hl.operators.RoceReceiverOp,
            "imu_receiver",
            data_channel=channel_imu,
            frame_context=int(cu_context),
            device_start=imu.start,
            device_stop=imu.stop,
            frame_size=imu_samples_per_frame * BYTES_PER_SAMPLE,
            ibv_name=ibv_name,
            ibv_port=ibv_port,
            frame_limit=frame_limit or 0,
        )
        profiler = graph.op(
            ProfilerOp, "profiler", samples_per_frame=imu_samples_per_frame
        )
        graph.add_flow(imu_rx.tensor, profiler.in_tensor, queue_depth=4)
        graph.add_flow(imu_rx.metadata, profiler.in_metadata, queue_depth=4)
        tensor_output_operators.append(imu_rx.name)

        clock_key = graph.add_clock(hs.RealtimeClock, "clock")
        graph.set_default_clock(clock_key)

        options = hs.CompileOptions()
        for operator_path in tensor_output_operators:
            options.deployment.bind_tensor_output_device(
                operator_path=operator_path,
                output_port="tensor",
                device=hs.execution.DeviceId(cuda_device_ordinal),
            )
        for name in tensor_names:
            options.deployment.bind_tensor_output_device(
                operator_path=f"bayer_demosaic_{name}",
                output_port="output",
                device=hs.execution.DeviceId(cuda_device_ordinal),
            )

        if headless:
            options.deployment.bind_tensor_output_device(
                operator_path="holoviz",
                output_port="output",
                device=hs.execution.DeviceId(cuda_device_ordinal),
            )

        plan = hs.compile(graph, options)
        if not plan.ok:
            print("Graph compilation failed:", file=sys.stderr)
            for diag in plan.diagnostics:
                print(f"  {diag}", file=sys.stderr)
            return 1

        print(
            f"Capturing 2 sensors + IMU, mode={camera_mode} -- running (Ctrl+C to stop)..."
        )
        with hs.run_async(plan) as session:
            # With --frame-limit, the first camera receiver that reaches frame_limit ends the run.
            session.wait()
    finally:
        hololink_dev.stop()

    _cuda_check(cuda_driver.cuDevicePrimaryCtxRelease(cu_device))
    print("Done.")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--accelerometer-rate",
        type=float,
        choices=[12.5, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0],
        default=400.0,
        help="Accelerometer data rate",
    )
    parser.add_argument(
        "--gyroscope-rate",
        type=int,
        choices=[100, 200, 400, 1000, 2000],
        default=400,
        help="Gyroscope data rate",
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
    parser.add_argument(
        "--ibv-name",
        default="",
        help="IBV device to use (default: the device on the board's network interface)",
    )
    parser.add_argument(
        "--ibv-port",
        type=int,
        default=1,
        help="Port number of IBV device (with --ibv-name)",
    )
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=300,
        help="Stop after receiving this many frames.",
    )
    parser.add_argument(
        "--frequency",
        type=int,
        default=30,
        help="VSYNC frequency in Hz (10, 30, 60, 90, 120). Default is 30Hz",
    )
    parser.add_argument(
        "--camera-mode",
        type=int,
        default=0,
        help="VB1940 mode",
    )
    parser.add_argument(
        "--exp",
        type=int,
        default=256,
        help="set EXPOSURE duration in lines, RANGE(4 to 65535). Default line value is 29.70usec",
    )
    parser.add_argument(
        "--gain",
        type=int,
        default=0,
        help="Set Analog Gain, RANGE(0 to 12). Default is 0. Equation is (16/(16-gain)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    logging.info("Initializing.")

    sys.exit(
        run(
            args.hololink,
            args.camera_mode,
            args.ibv_name,
            args.ibv_port,
            args.frequency,
            args.exp,
            args.gain,
            args.accelerometer_rate,
            args.gyroscope_rate,
            args.frame_limit,
            args.headless,
        )
    )


if __name__ == "__main__":
    main()
