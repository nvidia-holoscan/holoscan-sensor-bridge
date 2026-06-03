import json
import os
import time

import hololink as hololink_module


def byte_align_8(value):
    return (value + 7) // 8 * 8


def get_csi_image_size(camera_mode):
    return camera_mode["line_bytes"] * camera_mode["height"]


def get_csi_image_start_byte(camera_mode):
    return camera_mode["image_start_byte"]


def get_csi_frame_size(camera_mode):
    return camera_mode["csi_frame_size"]


def get_csi_line_bytes(camera_mode):
    return camera_mode["line_bytes"]


def line_bytes_from_pixel_format(pixel_width, pixel_format):
    if pixel_format == hololink_module.sensors.csi.PixelFormat.RAW_8:
        return pixel_width  # 8 bits per pixel, so no padding needed
    elif pixel_format == hololink_module.sensors.csi.PixelFormat.RAW_10:
        return (pixel_width + 3) // 4 * 5  # 10 bits per pixel, so 5 bytes per 4 pixels
    elif pixel_format == hololink_module.sensors.csi.PixelFormat.RAW_12:
        return (pixel_width + 1) // 2 * 3  # 12 bits per pixel, so 3 bytes per 2 pixels
    else:
        raise ValueError(
            f"Invalid pixel format: {pixel_format}. must be one of {list(hololink_module.sensors.csi.PixelFormat)}"
        )


# python dict of all the shared camera properties used to configure tests and generate
# test data for the HSB Emulator for different camera scenarios
camera_properties = {
    "vb1940": {
        "configure_timeout": 35,
        "examples": {
            "linux": {
                1: "examples/linux_vb1940_player.py",
                2: "examples/linux_single_network_stereo_vb1940_player.py",
            },
            "coe": {
                1: "examples/linux_coe_single_vb1940_player.py",
                2: "examples/linux_coe_stereo_vb1940_player.py",
            },
            "sipl": {
                1: "examples/sipl_player.py",
                2: "examples/sipl_player.py",
            },
            "roce": {
                1: "examples/vb1940_player.py",
                2: "examples/single_network_stereo_vb1940_player.py",
            },
        },
        "mode_flag": "--camera-mode",
        "modes": {
            hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS.value: {
                "start_lines": 1,
                "end_lines": 2,
                "width": 2560,
                "height": 1984,
                "frame_rate": 30,
                "pixel_format": hololink_module.sensors.csi.PixelFormat.RAW_10,
            },
            hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_1920X1080_30FPS.value: {
                "start_lines": 1,
                "end_lines": 2,
                "width": 1920,
                "height": 1080,
                "frame_rate": 30,
                "pixel_format": hololink_module.sensors.csi.PixelFormat.RAW_10,
            },
            hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS_8BIT.value: {
                "start_lines": 1,
                "end_lines": 2,
                "width": 2560,
                "height": 1984,
                "frame_rate": 30,
                "pixel_format": hololink_module.sensors.csi.PixelFormat.RAW_8,
            },
        },
    },
    "imx477": {
        "configure_timeout": 10,
        "examples": {
            "linux": {
                1: "examples/linux_imx477_player.py",
                2: "examples/linux_imx477_stereo_player.py",
            },
            "roce": {
                1: "examples/imx477_player.py",
                2: "examples/single_network_stereo_imx477_player.py",
            },
        },
        "mode_flag": "--resolution",
        "modes": {
            "4k": {
                "width": 3840,
                "height": 2160,
                "frame_rate": 60,
                "pixel_format": hololink_module.sensors.csi.PixelFormat.RAW_8,
                "bayer_format": hololink_module.sensors.csi.BayerFormat.RGGB,
            },
            "1080p": {
                "width": 1920,
                "height": 1080,
                "frame_rate": 60,
                "pixel_format": hololink_module.sensors.csi.PixelFormat.RAW_8,
                "bayer_format": hololink_module.sensors.csi.BayerFormat.RGGB,
            },
        },
    },
    "imx715": {
        "configure_timeout": 15,
        "examples": {
            "linux": {
                1: "examples/linux_imx715_player.py",
            },
            "roce": {
                1: "examples/imx715_player.py",
            },
        },
        "mode_flag": "--camera-mode",
        "modes": {
            hololink_module.sensors.imx715.IMX715_MODE_3840X2160_30FPS_12BPP: {
                "start_lines": 37,
                "width": 3840,
                "height": 2160,
                "frame_rate": 30,
                "pixel_format": hololink_module.sensors.csi.PixelFormat.RAW_12,
                "bayer_format": hololink_module.sensors.csi.BayerFormat.GBRG,
            },
            hololink_module.sensors.imx715.IMX715_MODE_3840X2160_60FPS_10BPP: {
                "start_lines": 41,
                "width": 3840,
                "height": 2160,
                "frame_rate": 60,
                "pixel_format": hololink_module.sensors.csi.PixelFormat.RAW_10,
                "bayer_format": hololink_module.sensors.csi.BayerFormat.GBRG,
            },
            hololink_module.sensors.imx715.IMX715_MODE_1920X1080_60FPS_12BPP: {
                "start_lines": 21,
                "end_lines": 14,
                # this has an unusual line_bytes setting. Hard-coding until a more robust method is needed
                "line_bytes": byte_align_8(
                    line_bytes_from_pixel_format(
                        1920, hololink_module.sensors.csi.PixelFormat.RAW_12
                    )
                    + 38
                ),
                "width": 1920,
                "height": 1080,
                "frame_rate": 60,
                "pixel_format": hololink_module.sensors.csi.PixelFormat.RAW_12,
                "bayer_format": hololink_module.sensors.csi.BayerFormat.GBRG,
            },
        },
    },
    "imx274": {
        "configure_timeout": 10,
        "examples": {
            "linux": {
                1: "examples/linux_imx274_player.py",
            },
            "roce": {
                1: "examples/imx274_player.py",
            },
        },
        "mode_flag": "--camera-mode",
        "modes": {
            hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_3840X2160_60FPS.value: {
                "start_bytes": 175,
                "start_lines": 8,
                "width": 3840,
                "height": 2160,
                "frame_rate": 60,
                "pixel_format": hololink_module.sensors.csi.PixelFormat.RAW_10,
                "bayer_format": hololink_module.sensors.csi.BayerFormat.RGGB,
            },
            hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS.value: {
                "start_bytes": 175,
                "start_lines": 8,
                "width": 1920,
                "height": 1080,
                "frame_rate": 60,
                "pixel_format": hololink_module.sensors.csi.PixelFormat.RAW_10,
                "bayer_format": hololink_module.sensors.csi.BayerFormat.RGGB,
            },
            hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_3840X2160_60FPS_12BITS.value: {
                "start_bytes": 175,
                "start_lines": 16,
                "width": 3840,
                "height": 2160,
                "frame_rate": 60,
                "pixel_format": hololink_module.sensors.csi.PixelFormat.RAW_12,
                "bayer_format": hololink_module.sensors.csi.BayerFormat.RGGB,
            },
        },
    },
}


def initialize_camera_properties():
    # fill in some default values for the camera properties that cannot/should not be calculated in the file scope maps

    for camera in camera_properties:
        modes = camera_properties[camera]["modes"]
        for mode, mode_properties in modes.items():
            # none of the cameras have line_start or line_end bytes,
            # so we can just set the line_bytes based on the pixel format and width
            # only set the values if they are not already set
            if "line_bytes" not in mode_properties:
                mode_properties["line_bytes"] = byte_align_8(
                    line_bytes_from_pixel_format(
                        mode_properties["width"], mode_properties["pixel_format"]
                    )
                )
            if "image_start_byte" not in mode_properties:
                mode_properties["image_start_byte"] = byte_align_8(
                    mode_properties.get("start_bytes", 0)
                ) + mode_properties["line_bytes"] * mode_properties.get(
                    "start_lines", 0
                )
            if "csi_frame_size" not in mode_properties:
                trailing_bytes = (
                    mode_properties.get("end_bytes", 0)
                    + mode_properties.get("end_lines", 0)
                    * mode_properties["line_bytes"]
                )
                mode_properties["csi_frame_size"] = (
                    mode_properties["image_start_byte"]
                    + mode_properties["line_bytes"] * mode_properties["height"]
                    + trailing_bytes
                )


initialize_camera_properties()


def sleep_frame_rate(last_frame_time, frame_rate_per_second):
    """Sleep the thread to try to match target frame rate"""
    delta_s = (
        1 / frame_rate_per_second - (time.time_ns() - last_frame_time) / 1000000000
    )
    if delta_s > 0:
        time.sleep(delta_s)


def handle_failed_subprocess(subprocess, name, kill=None):
    if kill:
        kill()
    stdout, stderr = subprocess.communicate()
    print(f"{name} subprocess stdout: \n{stdout.decode('utf-8', errors='replace')}")
    print(f"{name} subprocess stderr: \n{stderr.decode('utf-8', errors='replace')}")


# this maps the transport and camera count settings to a specific vb1940 example application
# which is used by the tests to both test the vb1940 I2CPeripheral/emulated sensor and the
# provided applications
vb1940_emulator_examples = {
    "linux": {
        1: "src/hololink/emulation/examples/serve_linux_single_vb1940_frames.py",
        2: "src/hololink/emulation/examples/serve_linux_stereo_vb1940_frames.py",
    },
    "coe": {
        1: "src/hololink/emulation/examples/serve_coe_single_vb1940_frames.py",
        2: "src/hololink/emulation/examples/serve_coe_stereo_vb1940_frames.py",
    },
}

# test cases covering basic emulator functionality:
# linux and coe transport with and without GPU device inputs and a simulated I2CPeripheral sensor (vb1940) with stateful interactions
# and accessing sensor state to build csi-frames dynamically
vb1940_loopback_cases = [
    (
        "linux",
        1,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS.value,
        True,
    ),
    (
        "coe",
        1,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS.value,
        True,
    ),
    (
        "linux",
        1,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS.value,
        False,
    ),
    (
        "linux",
        1,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_1920X1080_30FPS.value,
        False,
    ),
    (
        "linux",
        1,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS_8BIT.value,
        False,
    ),
]

# covers a wider parameter space but no new functional checks
vb1940_loopback_cases_extended = [
    (
        "linux",
        2,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS.value,
        True,
    ),
    (
        "coe",
        2,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS.value,
        True,
    ),
    (
        "coe",
        1,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS.value,
        False,
    ),
    (
        "coe",
        1,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_1920X1080_30FPS.value,
        False,
    ),
    (
        "coe",
        1,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS_8BIT.value,
        False,
    ),
    (
        "coe",
        2,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS.value,
        False,
    ),
    (
        "coe",
        2,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_1920X1080_30FPS.value,
        False,
    ),
    (
        "coe",
        2,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS_8BIT.value,
        False,
    ),
    (
        "linux",
        2,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS.value,
        False,
    ),
    (
        "linux",
        2,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_1920X1080_30FPS.value,
        False,
    ),
    (
        "linux",
        2,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS_8BIT.value,
        False,
    ),
]

# test cases for single camera imx loopback mode
# - running a subprocess for the host application from its respective examples/ folder
# - sends tests frames for "frame_limit" number of frames at frame rate determined by the camera mode
# - successful test is the host application exits normally without error code and no early exit from emulator process
single_camera_imx_loopback_cases = [
    ("imx477", "linux", "1080p", True),
    ("imx477", "linux", "1080p", False),
    ("imx477", "linux", "4k", False),
]

single_camera_imx_loopback_cases_extended = [
    (
        "imx715",
        "linux",
        hololink_module.sensors.imx715.IMX715_MODE_3840X2160_30FPS_12BPP,
        False,
    ),
    (
        "imx715",
        "linux",
        hololink_module.sensors.imx715.IMX715_MODE_3840X2160_60FPS_10BPP,
        False,
    ),
    (
        "imx715",
        "linux",
        hololink_module.sensors.imx715.IMX715_MODE_1920X1080_60FPS_12BPP,
        False,
    ),
    (
        "imx274",
        "linux",
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_3840X2160_60FPS.value,
        False,
    ),
    (
        "imx274",
        "linux",
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS.value,
        False,
    ),
    (
        "imx274",
        "linux",
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_3840X2160_60FPS_12BITS.value,
        False,
    ),
]

# SIPL example utilities


def get_camera_count_from_json_config(json_config):
    with open(json_config, "r") as f:
        config = json.load(f)
    return len(config["cameraConfigs"])


def has_sipl_interface():
    try:
        coe_channels = os.listdir("/sys/class/capture-coe-channel/")
        return len(coe_channels) > 0
    except FileNotFoundError:
        return False


# hardware accelerated tests for sipl transport. requires the "--hw-loopback" option/fixture
# Note: because of the nature of SIPL capture configurations, the actual camera count for the tests
# will be determined by the number of cameras in the JSON configuration file.
vb1940_loopback_cases_sipl = [
    (
        "sipl",
        None,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS.value,
        False,
    ),
]

# Infiniband example utilities


def has_ib_interface():
    try:
        ib_channels = hololink_module.infiniband_devices()
        return len(ib_channels) > 0
    except FileNotFoundError:
        return False


def get_ib_device_ifname(ib_device):
    try:
        ifname = os.listdir(f"/sys/class/infiniband/{ib_device}/device/net/")[0]
        return ifname
    except FileNotFoundError:
        return None


def get_ib_device_from_ifname(ifname):
    ib_devices = hololink_module.infiniband_devices()
    for ib_device in ib_devices:
        ib_device_ifname = get_ib_device_ifname(ib_device)
        if ib_device_ifname == ifname:
            return ib_device
    return None


# unused for now. see comments in test function
single_camera_imx_loopback_cases_roce = [
    ("imx477", "roce", "1080p", False),
    (
        "imx715",
        "roce",
        hololink_module.sensors.imx715.IMX715_MODE_3840X2160_30FPS_12BPP,
        False,
    ),
    (
        "imx274",
        "roce",
        hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_3840X2160_60FPS.value,
        False,
    ),
]

# hardware accelerated tests for roce transport. requires the "--hw-loopback" option/fixture
vb1940_loopback_cases_roce = [
    (
        "roce",
        1,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS.value,
        False,
    ),
    (
        "roce",
        2,
        hololink_module.sensors.vb1940.vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS.value,
        False,
    ),
]
