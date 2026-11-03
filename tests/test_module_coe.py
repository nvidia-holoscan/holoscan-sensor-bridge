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
#
# HSDK5 differences:
#   * The tests build an hs.Graph; each leg's capture ends the run at
#     frame_limit (frame_limit_ends_run).
#   * FusaCoeCaptureOp takes its geometry from
#     resolve_fusa_coe_geometry_from_dimensions() and passes it on to
#     PackedFormatConverterOp (or CsiToBayerOp without the packetizer).
#   * A CoE capture that stalls for stall_timeout_ms fails the run, which
#     RunMonitor reports (v2.7 had only the watchdog for a stall).
#   * There is no FrameAlignerOp in the graph: each stereo leg has its own
#     watchdog, and the tests pair the two legs' frame-start times after the
#     run. As v2.7's watchdog after the aligner did, operators.assert_aligned()
#     needs a pair within allowable_dt at least every 0.5 s.

import contextlib
import errno
import logging
import math
import os
import sys

import applications
import hololink_module.operators
import hololink_module.sensors.imx274 as imx274
import hololink_module.sensors.vb1940 as vb1940
import holoscan_vision as hv
import operators
import pytest
import utils

import hololink_module

# Seconds to wait for each bootp announcement before giving up.
DISCOVERY_TIMEOUT_S = 30.0

# FUSA CoE capture-request timeout, in milliseconds (matches the fusa_coe_*
# example players).
COE_TIMEOUT_MS = 1500

# A single empty capture request can be a routine startup hiccup. Three in a
# row is 4.5 seconds without a frame on a 30fps sensor and is a real stall.
COE_STALL_TIMEOUT_MS = 3 * COE_TIMEOUT_MS

# Once the receiver identifies a stall it stops the graph and tears down the
# capture stack. Keep the hard-abort watchdog behind that clean failure path,
# while retaining it as a backstop if graph shutdown itself hangs.
COE_STALL_CLEANUP_GRACE_S = 5

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
    # then long again for the last 10 frames: a leg can end a few frames short
    # of frame_limit (frames dropped on the way), and teardown follows.
    ready = 15
    return utils.timeout_sequence(
        [(30, ready), (0.5, max(1, frame_limit - ready - 10)), (30, 1)]
    )


def _fusa_coe_available():
    return hasattr(hololink_module.operators, "FusaCoeCaptureOp")


# FusaCoeCaptureCore::stop() runs its control-plane teardown (channel
# unconfigure / device stop) through guard_teardown(), which logs and swallows
# whatever is thrown: stop() is called from GXF's noexcept codelet-stop path, so
# letting an exception escape there calls std::terminate(). Exactly one such
# failure is expected and comes from outside our code -- libnvfusacap's
# closeHandler() does an unconditional close(0) on CoE handler teardown, which
# destroys a control socket that happened to land on a stdio fd, so the next
# control-plane sendto() fails with EBADF. Anything else swallowed there is an
# unexpected failure the swallow would otherwise hide, so scan the run's log and
# fail the test on it.
_TEARDOWN_FAILURE = "during teardown failed (ignored"
_EXPECTED_TEARDOWN_FAILURE = (f"errno={errno.EBADF}:", os.strerror(errno.EBADF))


def _unexpected_teardown_failures(stderr):
    return [
        line
        for line in stderr.splitlines()
        if _TEARDOWN_FAILURE in line
        and not any(known in line for known in _EXPECTED_TEARDOWN_FAILURE)
    ]


@contextlib.contextmanager
def check_teardown_failures(capfd):
    """Fail the test if teardown swallowed anything other than the known EBADF."""
    try:
        yield
    finally:
        captured = capfd.readouterr()
        # readouterr() drains the capture buffers; write them back so the run's
        # output still reaches pytest's report.
        sys.stdout.write(captured.out)
        sys.stderr.write(captured.err)
        unexpected = _unexpected_teardown_failures(captured.err)
    # Outside the finally: when the body itself raised, that failure is the one
    # worth reporting, so don't displace it with this check.
    assert (
        not unexpected
    ), "Unexpected exception(s) swallowed during teardown:\n" + "\n".join(unexpected)


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
    test_graph,
    side,
    camera,
    metadata,
    use_coe,
    coe_interface,
    coe_stall_timeout_ms,
    cu_context,
    frame_limit,
    affinity,
):
    """Build one sensor's capture + unpack path; return (capture, unpack).

    The capture ends the run at frame_limit (frame_limit_ends_run), together
    with the other leg's capture in a stereo run."""
    width, height = _dims(camera)
    bayer_name = camera.bayer_format().name.lower()
    if use_coe:
        # FUSA hardware capture. The channel + frame-metadata services are
        # resolved from the enumeration metadata; device_start/device_stop arm
        # and disarm the sensor at pipeline start/stop.
        packetizer_enabled = (
            camera.pixel_format() != hololink_module.sensors.csi.PixelFormat.RAW_8
        )
        geometry = hololink_module.resolve_fusa_coe_geometry_from_dimensions(
            int(camera.pixel_format().value), width, height, packetizer_enabled
        )
        capture = test_graph.op(
            hololink_module.operators.FusaCoeCaptureOp,
            f"fusa_coe_capture_{side}",
            metadata=metadata,
            device_start=camera.start,
            device_stop=camera.stop,
            interface=coe_interface,
            mac_addr=_parse_mac(metadata.get("mac_id", DEFAULT_MAC)),
            timeout=COE_TIMEOUT_MS,
            stall_timeout_ms=coe_stall_timeout_ms,
            pixel_format=camera.pixel_format().name.lower(),
            bayer_format=bayer_name,
            packetizer_enabled=packetizer_enabled,
            geometry=geometry,
            frame_limit=frame_limit,
            frame_limit_ends_run=True,
        )
        if packetizer_enabled:
            unpack = test_graph.op(
                hololink_module.operators.PackedFormatConverterOp,
                f"packed_format_converter_{side}",
                cuda_device_ordinal=test_graph.cuda_device_ordinal,
                bayer_format=bayer_name,
                geometry=geometry,
            )
        else:
            unpack = test_graph.op(
                hololink_module.operators.CsiToBayerOp,
                f"csi_to_bayer_{side}",
                cuda_device_ordinal=test_graph.cuda_device_ordinal,
                geometry=geometry,
                bayer_format=bayer_name,
                # FusaCoeCaptureOp strips the CSI header before emitting.
                header_included=False,
            )
        test_graph.bind(f"fusa_coe_capture_{side}", "tensor")
    else:
        # Linux-socket receiver. The module op resolves its data channel from
        # the enumeration metadata (no ibv_name / coe_channel / pixel_width);
        # its frame_size comes from the CSI converter's geometry.
        converter = hololink_module.CsiToBayerConverter()
        camera.configure_converter(converter)
        capture = test_graph.op(
            hololink_module.operators.LinuxReceiverOp,
            f"receiver_{side}",
            metadata=metadata,
            frame_context=int(cu_context),
            frame_size=converter.get_csi_length(),
            receiver_affinity=affinity,
            device_start=camera.start,
            device_stop=camera.stop,
            frame_limit=frame_limit,
            frame_limit_ends_run=True,
        )
        unpack = test_graph.op(
            hololink_module.operators.CsiToBayerOp,
            f"csi_to_bayer_{side}",
            cuda_device_ordinal=test_graph.cuda_device_ordinal,
            geometry=converter.geometry(),
            bayer_format=bayer_name,
        )
        test_graph.bind(f"receiver_{side}", "tensor")
    test_graph.add_flow(capture.tensor, unpack.in_tensor, queue_depth=4)
    test_graph.add_flow(capture.metadata, unpack.in_metadata, queue_depth=4)
    return capture, unpack


def _build_isp(test_graph, side, camera, unpack, watchdog_token):
    """Naive ISP tail: ImageProcessorOp -> BayerDemosaicOp, and the watchdog;
    return the demosaiced image to show."""
    bayer_format = camera.bayer_format()
    image_processor = test_graph.op(
        hololink_module.operators.ImageProcessorOp,
        f"image_processor_{side}",
        optical_black=OPTICAL_BLACK,
        bayer_format=bayer_format.value,
        pixel_format=camera.pixel_format().value,
        cuda_device_ordinal=test_graph.cuda_device_ordinal,
    )
    test_graph.add_flow(unpack.tensor, image_processor.in_tensor, queue_depth=4)
    test_graph.add_flow(unpack.metadata, image_processor.in_metadata, queue_depth=4)
    test_graph.bind(str(unpack.name), "tensor")
    demosaic = test_graph.op(
        hv.operators.BayerDemosaicOp,
        f"demosaic_{side}",
        bayer_grid_pos=bayer_format.name.lower(),
        generate_alpha=True,
        alpha_value=65535,
    )
    test_graph.add_flow(image_processor.tensor, demosaic.input, queue_depth=4)
    test_graph.bind(f"image_processor_{side}", "tensor")
    test_graph.bind(f"demosaic_{side}", "output")
    watchdog = test_graph.op(
        operators.WatchdogOp, f"watchdog_{side}", watchdog_token=watchdog_token
    )
    operators.metadata_flow(test_graph, image_processor.metadata, watchdog.metadata)
    width, height = _dims(camera)
    return applications.Image(demosaic.output, width, height)


def _watchdog_timeouts(frame_limit, coe_stall_timeout_ms):
    watchdog_timeouts = _initial_timeout(frame_limit)
    if coe_stall_timeout_ms:
        clean_failure_timeout_s = (
            coe_stall_timeout_ms / 1000 + COE_STALL_CLEANUP_GRACE_S
        )
        watchdog_timeouts = [
            max(timeout, clean_failure_timeout_s) for timeout in watchdog_timeouts
        ]
    return watchdog_timeouts


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
    coe_stall_timeout_ms=0,
):
    """Bring the board up and run a single-sensor capture.

    ``prepare_camera()`` constructs and configures the camera; it runs after
    reset (and PTP sync) because sensor construction touches the control plane,
    which must be reset first — this is the ordering the module VB1940 test
    uses, and IMX274 tolerates it too.

    In HSDK5 a CoE capture that stalls for coe_stall_timeout_ms fails the run,
    which RunMonitor reports (v2.7 had only the watchdog for a stall).
    """
    logging.info("Initializing.")
    hololink = hololink_module.HololinkInterfaceV1.get_service(metadata)
    watchdog_timeouts = _watchdog_timeouts(frame_limit, coe_stall_timeout_ms)
    # CudaContext releases the primary context on exit even if the body raises.
    with applications.CudaContext() as cuda_context, utils.Watchdog(
        "watchdog",
        initial_timeout=watchdog_timeouts,
        min_taps=utils.frame_floor(frame_limit),
    ) as watchdog, utils.RunMonitor() as monitor:
        hololink.start()
        with utils.AbortHook(hololink.reset):
            try:
                hololink.reset()
                _synchronize_ptp(hololink, ptp_enable)
                camera = prepare_camera()
                test_graph = applications.TestGraph(
                    "mono_capture", cuda_device_ordinal=cuda_context.device_ordinal
                )
                _, unpack = _build_leg(
                    test_graph,
                    "mono",
                    camera,
                    metadata,
                    use_coe,
                    coe_interface,
                    coe_stall_timeout_ms,
                    cuda_context.cu_context,
                    frame_limit,
                    [2],
                )
                image = _build_isp(
                    test_graph, "mono", camera, unpack, operators.register(watchdog)
                )
                applications.show_images(
                    test_graph, "holoviz", [image], headless, "Mono capture"
                )
                logging.info("Calling run")
                test_graph.run()
            finally:
                hololink.stop()
    monitor.assert_ok()


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
    coe_stall_timeout_ms=0,
):
    """Bring the board up and run a two-sensor capture, then pair the frames.

    ``prepare_cameras()`` returns (camera_left, camera_right), constructed and
    configured after reset for the same reason as run_mono. The two sensors
    free-run (no hardware frame-sync); v2.7 paired them in the graph with
    FrameAlignerOp, and this test pairs their frame-start times after the run
    (operators.assert_aligned()).
    """
    logging.info("Initializing.")
    # Both sensors on a stereo board share the same control-plane
    # HololinkInterface, so bring the unit up just once.
    hololink = hololink_module.HololinkInterfaceV1.get_service(metadata_left)
    assert hololink is hololink_module.HololinkInterfaceV1.get_service(metadata_right)
    watchdog_timeouts = _watchdog_timeouts(frame_limit, coe_stall_timeout_ms)
    frame_starts = {"left": [], "right": []}
    with applications.CudaContext() as cuda_context, utils.leg_watchdogs(
        ("left", "right"),
        initial_timeout=watchdog_timeouts,
        min_taps=utils.frame_floor(frame_limit),
    ) as watchdogs, utils.RunMonitor() as monitor:
        hololink.start()
        with utils.AbortHook(hololink.reset):
            try:
                hololink.reset()
                _synchronize_ptp(hololink, ptp_enable)
                camera_left, camera_right = prepare_cameras()
                test_graph = applications.TestGraph(
                    "stereo_capture", cuda_device_ordinal=cuda_context.device_ordinal
                )
                images = []
                for side, camera, metadata, affinity in (
                    ("left", camera_left, metadata_left, [2]),
                    ("right", camera_right, metadata_right, [3]),
                ):
                    capture, unpack = _build_leg(
                        test_graph,
                        side,
                        camera,
                        metadata,
                        use_coe,
                        coe_interface,
                        coe_stall_timeout_ms,
                        cuda_context.cu_context,
                        frame_limit,
                        affinity,
                    )
                    images.append(
                        _build_isp(
                            test_graph,
                            side,
                            camera,
                            unpack,
                            operators.register(watchdogs[side]),
                        )
                    )

                    def time_profile(
                        image_timestamp_s, *args, starts=frame_starts[side]
                    ):
                        starts.append((image_timestamp_s, args[-1]))

                    profiler = test_graph.op(
                        operators.TimeProfiler,
                        f"profiler_{side}",
                        callback_token=operators.register(time_profile),
                    )
                    operators.metadata_flow(
                        test_graph, capture.metadata, profiler.metadata
                    )
                applications.show_images(
                    test_graph, "holoviz", images, headless, "Stereo capture"
                )
                logging.info("Calling run")
                test_graph.run()
            finally:
                hololink.stop()
    monitor.assert_ok()
    operators.assert_aligned(
        frame_starts["left"],
        frame_starts["right"],
        allowable_dt,
        min_pairs=utils.frame_floor(frame_limit),
    )


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
    capfd,
):
    if not _fusa_coe_available():
        pytest.skip("FusaCoeCaptureOp not built (needs -DHOLOLINK_BUILD_FUSA).")
    metadata_left, metadata_right = _stereo_metadata(module_dir, hololink_address)
    with check_teardown_failures(capfd):
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
@pytest.mark.reset_hsb_before_retry
@pytest.mark.parametrize(
    "camera_mode",  # noqa: E501
    [
        # Thor RCE can intermittently lose its CoE frame-slot phase. The test
        # turns the resulting ingress stall into a normal failure before retrying.
        pytest.param(VB1940_MODE, marks=pytest.mark.flaky(reruns=2)),
    ],
)
def test_vb1940_mono_fusa_naive_coe(
    camera_mode,
    headless,
    frame_limit,
    hololink_address,
    module_dir,
    ptp_enable,
    coe_interfaces,
    capfd,
):
    if not _fusa_coe_available():
        pytest.skip("FusaCoeCaptureOp not built (needs -DHOLOLINK_BUILD_FUSA).")
    metadata = _wait_for_channel(module_dir, hololink_address)
    adapter = hololink_module.Adapter.get_adapter()
    adapter.use_sensor(metadata, 0)
    with check_teardown_failures(capfd):
        run_mono(
            headless,
            metadata,
            frame_limit,
            COE,
            coe_interfaces[0],
            ptp_enable,
            prepare_camera=lambda: _prepare_vb1940_mono(metadata, camera_mode),
            coe_stall_timeout_ms=COE_STALL_TIMEOUT_MS,
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
@pytest.mark.reset_hsb_before_retry
@pytest.mark.parametrize(
    "camera_mode, allowable_dt",  # noqa: E501
    [
        # The same Thor RCE frame-slot stall can affect either CoE leg. Stop
        # cleanly, reset the board, and retry the complete stereo test.
        pytest.param(
            VB1940_MODE,
            TWO_FRAMES_AT_30FPS,
            marks=pytest.mark.flaky(reruns=2),
        ),
    ],
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
    capfd,
):
    if not _fusa_coe_available():
        pytest.skip("FusaCoeCaptureOp not built (needs -DHOLOLINK_BUILD_FUSA).")
    metadata_left, metadata_right = _stereo_metadata(module_dir, hololink_address)
    with check_teardown_failures(capfd):
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
            coe_stall_timeout_ms=COE_STALL_TIMEOUT_MS,
        )
