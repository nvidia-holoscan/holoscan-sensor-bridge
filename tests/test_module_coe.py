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
# Adapter (hololink_module) port of tests/test_coe.py.
#
# The capture path is built entirely on the hololink_module V1 surface and
# self-composes the pipeline in this file rather than reusing the shared
# tests/applications.py (MonoTest/StereoTest), matching the earlier module
# ports (test_module_imx274_pattern.py, test_module_imx274_timestamps.py).
#
# Notable differences from the legacy test, all driven by the module API:
#   * Discovery is Adapter.wait_for_channel(peer_ip, timeout) rather than
#     Enumerator.find_channel; stereo clones the base EnumerationMetadata and
#     calls Adapter.use_sensor(clone, n).
#   * Two capture front-ends share one ISP tail (ImageProcessorOp naive ISP ->
#     BayerDemosaicOp -> HolovizOp):
#       - non-CoE:  LinuxReceiverOp -> CsiToBayerOp
#       - CoE:      FusaCoeCaptureOp -> PackedFormatConverterOp
#     The module has no Linux-socket CoE receiver (the legacy LinuxCoeReceiverOp
#     has no module twin); the module's CoE consumer is the FUSA hardware
#     capture op, so the CoE cases run FusaCoeCaptureOp (built only with
#     -DHOLOLINK_BUILD_FUSA, hence a runtime skip when it is absent).
#   * The two ArgusIspOp cases from the legacy file are dropped (ArgusIspOp is
#     deprecated and has no module equivalent).
#   * Multi-sensor (stereo) tests insert a FrameAlignerOp: the two sensors
#     free-run (no hardware frame-sync), so the aligner pairs them frame-for-
#     frame and drops a group whenever the legs' timestamps differ by more than
#     one frame period. Both capture ops (Linux and, since the rename_metadata
#     addition, FUSA CoE) stamp per-leg-prefixed timestamp metadata natively,
#     so no test-side prefixing shim is needed.
#   * VB1940 bring-up folds setup_clock / register pokes into configure(), so
#     (unlike the legacy vb1940_camera_factory) there is no write_uint32 /
#     get_register_32 dance; the naive tests run the sensor free-running
#     (NullVsync), matching the legacy null-synchronizer default.
#
# As in the legacy Linux-socket tests, the pipeline is validated by running to
# frame_limit without a watchdog timeout; image data is not evaluated (the
# Linux and CoE paths tolerate packet loss). ptp_synchronize() runs when --ptp
# is given so the FPGA timestamp clock shares the host's PTP domain.

import ctypes
import logging
import math

import hololink_module.operators
import hololink_module.sensors.imx274 as imx274
import hololink_module.sensors.vb1940 as vb1940
import holoscan
import operators
import pytest
import utils

import hololink_module

# Seconds to wait for each bootp announcement before giving up.
DISCOVERY_TIMEOUT_S = 30.0

# FUSA CoE capture-request timeout, in milliseconds (matches the fusa_coe_*
# example players).
COE_TIMEOUT_MS = 1500

# Fallback when the enumeration metadata carries no board MAC.
DEFAULT_MAC = "00:00:00:00:00:00"

# ImageProcessorOp optical-black level (matches the legacy naive_isp in
# tests/applications.py).
OPTICAL_BLACK = 50

# CoE vs. non-CoE selection is expressed as which capture op the application
# instantiates.
COE = True
LINUX = False


def _wait_for_channel(module_dir, peer_ip):
    adapter = hololink_module.Adapter.get_adapter()
    if module_dir:
        adapter.set_module_directory(module_dir)
    return adapter.wait_for_channel(peer_ip, DISCOVERY_TIMEOUT_S)


def _parse_mac(mac_str):
    return [int(x, 16) for x in mac_str.split(":")]


def _dims(camera):
    """Return (width, height).

    The module IMX274 driver exposes geometry as ``_width`` / ``_height``
    attributes (set by ``set_mode``); the module VB1940 driver exposes it as
    ``width()`` / ``height()`` methods (valid after ``configure``).
    """
    if hasattr(camera, "_width"):
        return camera._width, camera._height
    return camera.width(), camera.height()


def _initial_timeout(frame_limit):
    # Long timeout while the pipeline warms up, short timeout in steady state,
    # then long again for drain — the same shape the sibling module ports use.
    ready = 15
    return utils.timeout_sequence(
        [(30, ready), (0.5, max(1, frame_limit - ready - 2)), (30, 1)]
    )


def _fusa_coe_available():
    return hasattr(hololink_module.operators, "FusaCoeCaptureOp")


# Round up so the frame aligner's acceptance window is never smaller than the
# intended tolerance.
def round_up(f, decimal_places):
    factor = 10**decimal_places
    return math.ceil(f * factor) / factor


# The two sensors free-run (no hardware frame-sync), so the frame aligner needs
# a window wider than a single frame period to pair the legs; use twice the
# frame time for every stereo case (both Linux and CoE).
TWO_FRAMES_AT_60FPS = round_up(2.0 / 60, 3)
TWO_FRAMES_AT_30FPS = round_up(2.0 / 30, 3)


def _build_leg(
    app,
    side,
    camera,
    metadata,
    condition,
    rename_metadata,
    capture_out_tensor_name,
    unpack_out_tensor_name,
    affinity,
):
    """Build one sensor's capture + unpack path on ``app``.

    The capture op emits its frame under ``capture_out_tensor_name`` (a per-leg
    name so a downstream FrameAlignerOp can tell legs apart) and stamps
    per-leg-prefixed metadata via ``rename_metadata``. The unpack op
    (CsiToBayerOp for the Linux path, PackedFormatConverterOp for the CoE path)
    reads the "" tensor — the capture op emits "" directly in the mono case,
    and the aligner re-emits "" in the stereo case — and writes
    ``unpack_out_tensor_name`` for the ISP tail. Returns (capture, unpack); the
    caller wires capture -> [aligner ->] unpack.
    """
    width, height = _dims(camera)
    if app._use_coe:
        # FUSA hardware capture. The channel + frame-metadata services are
        # resolved from the enumeration metadata; device.start()/stop() arm
        # and disarm the sensor at pipeline start/stop. rename_metadata adds
        # the per-leg prefix so two legs' metadata don't collide.
        capture = hololink_module.operators.FusaCoeCaptureOp(
            app,
            condition,
            name=f"fusa_coe_capture_{side}",
            enumeration_metadata=metadata,
            interface=app._coe_interface,
            mac_addr=_parse_mac(metadata.get("mac_id", DEFAULT_MAC)),
            timeout=COE_TIMEOUT_MS,
            device=camera,
            rename_metadata=rename_metadata,
            out_tensor_name=capture_out_tensor_name,
        )
        camera.configure_converter(capture)
        unpack_pool = holoscan.resources.BlockMemoryPool(
            app,
            name=f"unpack_pool_{side}",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=width * ctypes.sizeof(ctypes.c_uint16) * height,
            num_blocks=4,
        )
        unpack = hololink_module.operators.PackedFormatConverterOp(
            app,
            name=f"packed_format_converter_{side}",
            allocator=unpack_pool,
            cuda_device_ordinal=app._cuda_device_ordinal,
            out_tensor_name=unpack_out_tensor_name,
        )
        capture.configure_converter(unpack)
        return capture, unpack

    # Linux-socket receiver. The module op resolves its data channel from
    # the enumeration metadata (no ibv_name / coe_channel / pixel_width);
    # its frame_size comes from the CSI converter's geometry.
    unpack_pool = holoscan.resources.BlockMemoryPool(
        app,
        name=f"unpack_pool_{side}",
        # storage_type of 1 is device memory
        storage_type=1,
        block_size=width * ctypes.sizeof(ctypes.c_uint16) * height,
        num_blocks=4,
    )
    unpack = hololink_module.operators.CsiToBayerOp(
        app,
        name=f"csi_to_bayer_{side}",
        allocator=unpack_pool,
        cuda_device_ordinal=app._cuda_device_ordinal,
        out_tensor_name=unpack_out_tensor_name,
    )
    camera.configure_converter(unpack)
    capture = hololink_module.operators.LinuxReceiverOp(
        app,
        condition,
        name=f"receiver_{side}",
        enumeration_metadata=metadata,
        frame_context=app._cuda_context,
        frame_size=unpack.get_csi_length(),
        receiver_affinity=affinity,
        device_start=camera.start,
        device_stop=camera.stop,
        rename_metadata=rename_metadata,
        out_tensor_name=capture_out_tensor_name,
    )
    return capture, unpack


def _build_isp(app, side, camera, unpack, tensor_name):
    """Naive ISP tail on ``app``: ImageProcessorOp -> BayerDemosaicOp.

    ImageProcessorOp preserves the tensor name, so the demosaic reads/writes
    the same per-leg name. Returns the demosaic op (its "transmitter" output
    carries ``tensor_name``).
    """
    width, height = _dims(camera)
    bayer_format = camera.bayer_format()
    image_processor = hololink_module.operators.ImageProcessorOp(
        app,
        name=f"image_processor_{side}",
        optical_black=OPTICAL_BLACK,
        bayer_format=bayer_format.value,
        pixel_format=camera.pixel_format().value,
    )
    rgba_components_per_pixel = 4
    bayer_pool = holoscan.resources.BlockMemoryPool(
        app,
        name=f"bayer_pool_{side}",
        # storage_type of 1 is device memory
        storage_type=1,
        block_size=width
        * rgba_components_per_pixel
        * ctypes.sizeof(ctypes.c_uint16)
        * height,
        num_blocks=4,
    )
    demosaic = holoscan.operators.BayerDemosaicOp(
        app,
        name=f"demosaic_{side}",
        pool=bayer_pool,
        generate_alpha=True,
        alpha_value=65535,
        bayer_grid_pos=bayer_format.value,
        interpolation_mode=0,
        in_tensor_name=tensor_name,
        out_tensor_name=tensor_name,
    )
    app.add_flow(unpack, image_processor, {("output", "input")})
    app.add_flow(image_processor, demosaic, {("output", "receiver")})
    return demosaic


class MonoCaptureApplication(holoscan.core.Application):
    """Single-sensor capture pipeline.

    capture -> unpack -> image_processor -> demosaic -> visualizer -> watchdog

    The capture op ticks frame_limit times (CountCondition) and the pipeline
    drains; the watchdog sink guards against a stall.
    """

    def __init__(
        self,
        headless,
        cuda_context,
        cuda_device_ordinal,
        metadata,
        camera,
        frame_limit,
        watchdog,
        use_coe,
        coe_interface,
    ):
        logging.info("__init__")
        super().__init__()
        self._headless = headless
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._metadata = metadata
        self._camera = camera
        self._frame_limit = frame_limit
        self._watchdog = watchdog
        self._use_coe = use_coe
        self._coe_interface = coe_interface
        self.is_metadata_enabled = True

    def compose(self):
        logging.info("compose")
        condition = holoscan.conditions.CountCondition(
            self, name="count", count=self._frame_limit
        )
        capture, unpack = _build_leg(
            self,
            "mono",
            self._camera,
            self._metadata,
            condition,
            rename_metadata=lambda name: name,
            capture_out_tensor_name="",
            unpack_out_tensor_name="",
            affinity=[2],
        )
        self.add_flow(capture, unpack, {("output", "input")})
        demosaic = _build_isp(self, "mono", self._camera, unpack, "")
        visualizer = holoscan.operators.HolovizOp(
            self,
            name="holoviz",
            headless=self._headless,
            enable_camera_pose_output=True,
            camera_pose_output_type="extrinsics_model",
        )
        watchdog_operator = operators.WatchdogOp(
            self,
            name="watchdog_operator",
            watchdog=self._watchdog,
        )
        self.add_flow(demosaic, visualizer, {("transmitter", "receivers")})
        self.add_flow(visualizer, watchdog_operator, {("camera_pose_output", "input")})


class StereoCaptureApplication(holoscan.core.Application):
    """Two free-running sensors, aligned, into a side-by-side visualizer.

    capture_{left,right} -> frame_aligner -> unpack_{left,right}
        -> image_processor_{left,right} -> demosaic_{left,right} -> visualizer
        -> watchdog

    The two legs free-run on BooleanConditions; the frame aligner drops
    unpaired / misaligned frames, so the watchdog sink owns the frame budget
    (counting on the sources would starve the sink by the drop count).
    """

    def __init__(
        self,
        headless,
        cuda_context,
        cuda_device_ordinal,
        metadata_left,
        camera_left,
        metadata_right,
        camera_right,
        allowable_dt,
        frame_limit,
        watchdog,
        use_coe,
        coe_interface,
    ):
        logging.info("__init__")
        super().__init__()
        self._headless = headless
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._metadata_left = metadata_left
        self._camera_left = camera_left
        self._metadata_right = metadata_right
        self._camera_right = camera_right
        self._allowable_dt = allowable_dt
        self._frame_limit = frame_limit
        self._watchdog = watchdog
        self._use_coe = use_coe
        self._coe_interface = coe_interface
        # Each leg stamps per-leg-prefixed metadata (left_/right_), so the keys
        # never collide; UPDATE merges everything the two aligned messages carry.
        self.is_metadata_enabled = True
        self.metadata_policy = holoscan.core.MetadataPolicy.UPDATE

    def _pane(self, name, offset_x):
        spec = holoscan.operators.HolovizOp.InputSpec(
            name, holoscan.operators.HolovizOp.InputType.COLOR
        )
        view = holoscan.operators.HolovizOp.InputSpec.View()
        view.offset_x = offset_x
        view.offset_y = 0
        view.width = 0.5
        view.height = 1
        spec.views = [view]
        return spec

    def compose(self):
        logging.info("compose")
        # The legs free-run; the watchdog sink owns the frame budget and
        # disables these tick conditions once frame_limit aligned pairs have
        # arrived. Counting on the sources would starve the sink because the
        # aligner drops unpaired / misaligned frames upstream of it.
        condition_left = holoscan.conditions.BooleanCondition(
            self, name="enable_left", enable_tick=True
        )
        condition_right = holoscan.conditions.BooleanCondition(
            self, name="enable_right", enable_tick=True
        )
        capture_left, unpack_left = _build_leg(
            self,
            "left",
            self._camera_left,
            self._metadata_left,
            condition_left,
            rename_metadata=lambda name: f"left_{name}",
            capture_out_tensor_name="left",
            unpack_out_tensor_name="left",
            affinity=[2],
        )
        capture_right, unpack_right = _build_leg(
            self,
            "right",
            self._camera_right,
            self._metadata_right,
            condition_right,
            rename_metadata=lambda name: f"right_{name}",
            capture_out_tensor_name="right",
            unpack_out_tensor_name="right",
            affinity=[3],
        )

        # Frame aligner. Both legs feed its single input; it pairs them by
        # tensor name ("left"/"right") and reads the per-leg timestamp metadata
        # (left_/right_timestamp_s, left_/right_timestamp_ns), dropping a group
        # whenever the two legs' timestamps differ by more than allowable_dt
        # (one full frame period). It fans out paired frames to each leg's
        # unpack op.
        frame_aligner = operators.FrameAlignerOp(
            self,
            name="frame_aligner",
            allowable_dt=self._allowable_dt,
            input_tensors=["left", "right"],
            rename_metadata=[
                lambda name: f"left_{name}",
                lambda name: f"right_{name}",
            ],
            outputs=["output_left", "output_right"],
        )

        demosaic_left = _build_isp(self, "left", self._camera_left, unpack_left, "left")
        demosaic_right = _build_isp(
            self, "right", self._camera_right, unpack_right, "right"
        )

        visualizer = holoscan.operators.HolovizOp(
            self,
            name="holoviz",
            headless=self._headless,
            framebuffer_srgb=True,
            tensors=[self._pane("left", 0), self._pane("right", 0.5)],
            enable_camera_pose_output=True,
            camera_pose_output_type="extrinsics_model",
        )
        # The sink ticks once per aligned pair reaching the visualizer and,
        # after frame_limit pairs, disables the sources so the app drains.
        watchdog_count = holoscan.conditions.CountCondition(
            self, name="watchdog_count", count=self._frame_limit
        )
        watchdog_operator = operators.WatchdogOp(
            self,
            watchdog_count,
            name="watchdog_operator",
            watchdog=self._watchdog,
            frame_limit=self._frame_limit,
            stop_conditions=[condition_left, condition_right],
        )

        self.add_flow(capture_left, frame_aligner, {("output", "input")})
        self.add_flow(capture_right, frame_aligner, {("output", "input")})
        self.add_flow(frame_aligner, unpack_left, {("output_left", "input")})
        self.add_flow(frame_aligner, unpack_right, {("output_right", "input")})
        self.add_flow(demosaic_left, visualizer, {("transmitter", "receivers")})
        self.add_flow(demosaic_right, visualizer, {("transmitter", "receivers")})
        self.add_flow(visualizer, watchdog_operator, {("camera_pose_output", "input")})


def _synchronize_ptp(hololink, ptp_enable):
    if not ptp_enable:
        return
    logging.debug("Waiting for PTP sync.")
    if not hololink.ptp_synchronize():
        logging.error("Failed to synchronize PTP; ignoring.")
    else:
        logging.debug("PTP synchronized.")


def run_mono(
    headless,
    metadata,
    frame_limit,
    use_coe,
    coe_interface,
    ptp_enable,
    prepare_camera,
):
    """Bring the board up and run a single-sensor capture.

    ``prepare_camera()`` constructs and configures the camera; it runs after
    reset (and PTP sync) because sensor construction touches the control plane,
    which must be reset first — this is the ordering the module VB1940 test
    uses, and IMX274 tolerates it too.
    """
    logging.info("Initializing.")
    hololink = hololink_module.HololinkInterfaceV1.get_service(metadata)
    # CudaContext releases the primary context on exit even if the body raises.
    with utils.CudaContext() as cuda_context:
        with utils.Watchdog(
            "watchdog", initial_timeout=_initial_timeout(frame_limit)
        ) as watchdog:
            hololink.start()
            try:
                hololink.reset()
                _synchronize_ptp(hololink, ptp_enable)
                camera = prepare_camera()
                application = MonoCaptureApplication(
                    headless,
                    cuda_context.context,
                    cuda_context.device_ordinal,
                    metadata,
                    camera,
                    frame_limit,
                    watchdog,
                    use_coe,
                    coe_interface,
                )
                logging.info("Calling run")
                application.run()
            finally:
                hololink.stop()


def run_stereo(
    headless,
    metadata_left,
    metadata_right,
    allowable_dt,
    frame_limit,
    use_coe,
    coe_interface,
    ptp_enable,
    prepare_cameras,
):
    """Bring the board up and run an aligned two-sensor capture.

    ``prepare_cameras()`` returns (camera_left, camera_right), constructed and
    configured after reset for the same reason as run_mono.
    """
    logging.info("Initializing.")
    # Both sensors on a stereo board share the same control-plane
    # HololinkInterface, so bring the unit up just once.
    hololink = hololink_module.HololinkInterfaceV1.get_service(metadata_left)
    assert hololink is hololink_module.HololinkInterfaceV1.get_service(metadata_right)
    with utils.CudaContext() as cuda_context:
        with utils.Watchdog(
            "watchdog", initial_timeout=_initial_timeout(frame_limit)
        ) as watchdog:
            hololink.start()
            try:
                hololink.reset()
                _synchronize_ptp(hololink, ptp_enable)
                camera_left, camera_right = prepare_cameras()
                application = StereoCaptureApplication(
                    headless,
                    cuda_context.context,
                    cuda_context.device_ordinal,
                    metadata_left,
                    camera_left,
                    metadata_right,
                    camera_right,
                    allowable_dt,
                    frame_limit,
                    watchdog,
                    use_coe,
                    coe_interface,
                )
                logging.info("Calling run")
                application.run()
            finally:
                hololink.stop()


# --- IMX274 sensor bring-up ------------------------------------------------


def _prepare_imx274_mono(metadata, camera_mode, pattern):
    camera = imx274.Imx274Cam(metadata)
    _imx274_bring_up(camera, camera_mode, pattern)
    return camera


def _prepare_imx274_stereo(metadata_left, metadata_right, camera_mode):
    # The two sensors share the board's expander; select output 0 for the left
    # sensor and output 1 for the right one before constructing each camera.
    imx274.Imx274Cam.use_expander_configuration(metadata_left, 0)
    imx274.Imx274Cam.use_expander_configuration(metadata_right, 1)
    camera_left = imx274.Imx274Cam(metadata_left)
    camera_right = imx274.Imx274Cam(metadata_right)
    # Left = pattern 10, right = pattern 11 (as in the legacy test).
    _imx274_bring_up(camera_left, camera_mode, 10)
    _imx274_bring_up(camera_right, camera_mode, 11)
    return camera_left, camera_right


def _imx274_bring_up(camera, camera_mode, pattern):
    # set_mode populates the driver's geometry (_width/_height) that compose()
    # reads; configure() programs the sensor (folding in the reference-clock
    # setup, so there is no separate setup_clock() call). The test pattern
    # gives deterministic pixels since image data isn't evaluated here.
    camera.set_mode(camera_mode)
    camera.configure(camera_mode)
    camera.set_digital_gain_reg(0x4)
    camera.test_pattern(pattern)


# --- VB1940 sensor bring-up ------------------------------------------------


def _prepare_vb1940_mono(metadata, camera_mode):
    camera = vb1940.Vb1940Cam(metadata)
    _vb1940_bring_up(camera, camera_mode)
    return camera


def _prepare_vb1940_stereo(metadata_left, metadata_right, camera_mode):
    camera_left = vb1940.Vb1940Cam(metadata_left)
    camera_right = vb1940.Vb1940Cam(metadata_right)
    _vb1940_bring_up(camera_left, camera_mode)
    _vb1940_bring_up(camera_right, camera_mode)
    return camera_left, camera_right


def _vb1940_bring_up(camera, camera_mode):
    # configure() programs the sensor and populates width()/height(); the
    # module folds setup_clock and the sensor-enable register pokes into it, so
    # (unlike the legacy vb1940_camera_factory) there is no write_uint32 /
    # get_register_32 sequence. NullVsync (the constructor default) leaves the
    # sensor free-running, matching the legacy null-synchronizer default.
    camera.configure(camera_mode)


# --- IMX274 tests ----------------------------------------------------------

IMX274_MODE = imx274.Imx274_Mode.IMX274_MODE_1920X1080_60FPS


def _stereo_metadata(module_dir, hololink_address):
    # A board that hosts a stereo pair exposes the two sensors as two data
    # planes that share one control-plane HololinkInterface; discover it once
    # and clone the metadata per sensor.
    channel_metadata = _wait_for_channel(module_dir, hololink_address)
    adapter = hololink_module.Adapter.get_adapter()
    metadata_left = hololink_module.EnumerationMetadata(channel_metadata)
    adapter.use_sensor(metadata_left, 0)
    metadata_right = hololink_module.EnumerationMetadata(channel_metadata)
    adapter.use_sensor(metadata_right, 1)
    return metadata_left, metadata_right


@pytest.mark.skip_unless_imx274
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [IMX274_MODE],
)
def test_imx274_mono_linux_naive(
    camera_mode, headless, frame_limit, hololink_address, module_dir, ptp_enable
):
    metadata = _wait_for_channel(module_dir, hololink_address)
    with utils.PriorityScheduler():
        run_mono(
            headless,
            metadata,
            frame_limit,
            LINUX,
            None,
            ptp_enable,
            prepare_camera=lambda: _prepare_imx274_mono(metadata, camera_mode, 10),
        )


@pytest.mark.skip_unless_imx274
@pytest.mark.skip_unless_ptp
@pytest.mark.parametrize(
    "camera_mode, allowable_dt",  # noqa: E501
    [(IMX274_MODE, TWO_FRAMES_AT_60FPS)],
)
def test_imx274_stereo_linux_naive(
    camera_mode,
    allowable_dt,
    headless,
    frame_limit,
    hololink_address,
    module_dir,
    ptp_enable,
):
    metadata_left, metadata_right = _stereo_metadata(module_dir, hololink_address)
    with utils.PriorityScheduler():
        run_stereo(
            headless,
            metadata_left,
            metadata_right,
            allowable_dt,
            frame_limit,
            LINUX,
            None,
            ptp_enable,
            prepare_cameras=lambda: _prepare_imx274_stereo(
                metadata_left, metadata_right, camera_mode
            ),
        )


@pytest.mark.skip_unless_coe
@pytest.mark.skip_unless_imx274
@pytest.mark.skip_unless_ptp
@pytest.mark.parametrize(
    "camera_mode, allowable_dt",  # noqa: E501
    [(IMX274_MODE, TWO_FRAMES_AT_60FPS)],
)
def test_imx274_stereo_fusa_naive_coe(
    camera_mode,
    allowable_dt,
    headless,
    frame_limit,
    hololink_address,
    module_dir,
    ptp_enable,
    coe_interfaces,
):
    if not _fusa_coe_available():
        pytest.skip("FusaCoeCaptureOp not built (needs -DHOLOLINK_BUILD_FUSA).")
    metadata_left, metadata_right = _stereo_metadata(module_dir, hololink_address)
    with utils.PriorityScheduler():
        run_stereo(
            headless,
            metadata_left,
            metadata_right,
            allowable_dt,
            frame_limit,
            COE,
            coe_interfaces[0],
            ptp_enable,
            prepare_cameras=lambda: _prepare_imx274_stereo(
                metadata_left, metadata_right, camera_mode
            ),
        )


# --- VB1940 tests ----------------------------------------------------------

VB1940_MODE = vb1940.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS


@pytest.mark.skip_unless_vb1940
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [VB1940_MODE],
)
def test_vb1940_mono_linux_naive(
    camera_mode, headless, frame_limit, hololink_address, module_dir, ptp_enable
):
    metadata = _wait_for_channel(module_dir, hololink_address)
    with utils.PriorityScheduler():
        run_mono(
            headless,
            metadata,
            frame_limit,
            LINUX,
            None,
            ptp_enable,
            prepare_camera=lambda: _prepare_vb1940_mono(metadata, camera_mode),
        )


@pytest.mark.skip_unless_coe
@pytest.mark.skip_unless_vb1940
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [VB1940_MODE],
)
def test_vb1940_mono_fusa_naive_coe(
    camera_mode,
    headless,
    frame_limit,
    hololink_address,
    module_dir,
    ptp_enable,
    coe_interfaces,
):
    if not _fusa_coe_available():
        pytest.skip("FusaCoeCaptureOp not built (needs -DHOLOLINK_BUILD_FUSA).")
    metadata = _wait_for_channel(module_dir, hololink_address)
    adapter = hololink_module.Adapter.get_adapter()
    adapter.use_sensor(metadata, 0)
    with utils.PriorityScheduler():
        run_mono(
            headless,
            metadata,
            frame_limit,
            COE,
            coe_interfaces[0],
            ptp_enable,
            prepare_camera=lambda: _prepare_vb1940_mono(metadata, camera_mode),
        )


@pytest.mark.skip_unless_vb1940
@pytest.mark.skip_unless_ptp
@pytest.mark.parametrize(
    "camera_mode, allowable_dt",  # noqa: E501
    [(VB1940_MODE, TWO_FRAMES_AT_30FPS)],
)
def test_vb1940_stereo_linux_naive(
    camera_mode,
    allowable_dt,
    headless,
    frame_limit,
    hololink_address,
    module_dir,
    ptp_enable,
):
    metadata_left, metadata_right = _stereo_metadata(module_dir, hololink_address)
    with utils.PriorityScheduler():
        run_stereo(
            headless,
            metadata_left,
            metadata_right,
            allowable_dt,
            frame_limit,
            LINUX,
            None,
            ptp_enable,
            prepare_cameras=lambda: _prepare_vb1940_stereo(
                metadata_left, metadata_right, camera_mode
            ),
        )


@pytest.mark.skip_unless_coe
@pytest.mark.skip_unless_vb1940
@pytest.mark.skip_unless_ptp
@pytest.mark.parametrize(
    "camera_mode, allowable_dt",  # noqa: E501
    [(VB1940_MODE, TWO_FRAMES_AT_30FPS)],
)
def test_vb1940_stereo_fusa_naive_coe(
    camera_mode,
    allowable_dt,
    headless,
    frame_limit,
    hololink_address,
    module_dir,
    ptp_enable,
    coe_interfaces,
):
    if not _fusa_coe_available():
        pytest.skip("FusaCoeCaptureOp not built (needs -DHOLOLINK_BUILD_FUSA).")
    metadata_left, metadata_right = _stereo_metadata(module_dir, hololink_address)
    with utils.PriorityScheduler():
        run_stereo(
            headless,
            metadata_left,
            metadata_right,
            allowable_dt,
            frame_limit,
            COE,
            coe_interfaces[0],
            ptp_enable,
            prepare_cameras=lambda: _prepare_vb1940_stereo(
                metadata_left, metadata_right, camera_mode
            ),
        )
