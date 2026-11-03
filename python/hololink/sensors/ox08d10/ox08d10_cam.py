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

import logging
import time

import hololink as hololink_module

from . import ox08d10_mode

DRIVER_NAME = "OX08D10"
VERSION = 1

# 7-bit I2C address
CAM_I2C_ADDRESS = 0x36

# 24-bit chip ID spread across three 8-bit registers; 0x580844 identifies the OX08D10 model
REG_SENSOR_ID_H = 0x300A
REG_SENSOR_ID_M = 0x300B
REG_SENSOR_ID_L = 0x300C
EXPECTED_SENSOR_ID = (0x58, 0x08, 0x44)

# HCG coarse exposure registers (units: rows)
REG_EXPOSURE_H = 0x3501
REG_EXPOSURE_L = 0x3502

# Streaming control
REG_STREAM = 0x0100


class Ox08d10Cam:
    # Active image height (lines). MIPI output includes 16 additional trailing
    # rows from the serializer; those are handled in configure_converter via
    # trailing_bytes, not counted as active image data.
    ACTIVE_HEIGHT = 2160
    FRONT_EMBEDDED_LINES = 2  # EBD lines prepended to image data

    # ---------------------------------------------------------------------------
    # GMSL NOTE -- SKILL BOUNDARY
    #
    # The OX08D10 is connected via GMSL2 (MAX96719B serializer at 0x40) on the
    # Aumovio SAT301DP17 camera module. I2C to the sensor is tunneled through
    # the GMSL link via the MAX96716A deserializer at 0x28 on the Taurotech HSB
    # board. The full bring-up order is:
    #
    #   1. Configure MAX96716A deserializer (0x28) for 6Gbps GMSL2 mode and
    #      select the correct GMSL link (A or B).
    #   2. Wait for GMSL link lock (poll deserializer lock bit with timeout).
    #   3. Configure MAX96716A deserializer video pipe for tunnel mode.
    #   4. Configure MAX96719B serializer (0x40): MIPI PHY, RCLKOUT, translator.
    #   5. Release sensor reset via serializer GPIO3 and wait 500ms for boot.
    #   6. Configure deserializer D-PHY output path.
    #   7. Run sensor initialization sequences.
    #   8. Set virtual channel mapping.
    #   9. Validate sensor identity via now-live tunneled I2C.
    #
    # Unlike IMX274/IMX681, GMSL cameras do NOT use the LI I2C expander.
    # I2C reaches the sensor through the GMSL link; the expander is bypassed.
    # ---------------------------------------------------------------------------

    def __init__(
        self,
        hololink,
        i2c_bus=hololink_module.CAM_I2C_BUS,
        i2c_id=0,
        camera_i2c_address=CAM_I2C_ADDRESS,
        skip_i2c=False,
    ):
        # hololink is HololinkInterfaceV1 (from TauroTechDa326InterfaceV1.hololink()
        # or HololinkInterfaceV1.get_service()).  One I2c client per device on the bus.
        i2c_bus_n = i2c_bus + i2c_id
        self._camera_i2c_address = camera_i2c_address
        self._i2c_des = hololink.get_i2c(i2c_bus_n, ox08d10_mode.MAX96716A_DESER_ADDR)
        self._i2c_ser = hololink.get_i2c(i2c_bus_n, ox08d10_mode.MAX96719_SER_ADDR)
        self._i2c_cam = hololink.get_i2c(i2c_bus_n, camera_i2c_address)
        self._mode = ox08d10_mode.Ox08d10_Mode.Unknown
        self._received_height = None
        self._width = None
        self._height = None
        self._pixel_format = None
        self._num_lanes = None
        self.skip_i2c = skip_i2c

    # ------------------------------------------------------------------
    # GMSL bring-up: the main entry point for sensor initialization
    # ------------------------------------------------------------------

    def configure_vendor_tunnel_path(
        self,
        ox08d10_mode_set=ox08d10_mode.Ox08d10_Mode.OX08D10_MODE_3840X2160_30FPS_RAW12,
        link=ox08d10_mode.GmslLink.LINK_A,
    ):
        """Configure the full OX08D10/MAX96719/MAX96716 GMSL + sensor path.

        1. Configure MAX96716A deserializer control path (6Gbps, link select)
        2. Wait for GMSL link lock (deserializer LOCKED bit)
        3. Configure MAX96716A deserializer video pipe (tunnel mode, RAW12 DT)
        4. Configure MAX96719B serializer (D-PHY 4-lane, RCLKOUT, I2C translator)
        5. Release sensor reset via serializer GPIO3, wait 500ms for boot
        6. Configure deserializer D-PHY output path
        7. Run sensor initialization sequences
        8. Set virtual channel mapping based on GMSL link
        9. Validate sensor identity (0x300A/0x300B/0x300C)
        """
        self.set_mode(ox08d10_mode_set)
        ox08d10_mode.set_gmsl_link(link)

        link_name = "A" if link == ox08d10_mode.GmslLink.LINK_A else "B"
        pipe_sel, dphy_dst = self._deserializer_route(link_name)

        # Step 1+2+3: deserializer control path, link lock, video pipe
        self._configure_deserializer_tunnel(link_name, pipe_sel, dphy_dst)

        # Step 4: serializer init (PHY, RCLKOUT, I2C translator; reset NOT released yet)
        self._configure_serializer_vendor()

        # Step 5: release sensor reset and wait for boot
        self._release_ox08d_reset()

        # Step 6: deserializer D-PHY output path
        self._configure_deserializer_dphy_output(link_name, pipe_sel, dphy_dst)

        # Step 7: sensor register init
        self._init_sensor(ox08d10_mode_set)

        # Step 8: virtual channel mapping
        self._set_virtual_channel(link)

        # Step 9: validate sensor identity (after all config)
        self.validate_sensor_identity()

    def _deserializer_route(self, link_name):
        """Return (pipe_sel, dphy_dst) for the given GMSL link."""
        routes = {
            "A": (0x12, 0x09),
            "B": (0x36, 0x09),
        }
        try:
            return routes[link_name]
        except KeyError as e:
            raise ValueError("link_name must be 'A' or 'B'") from e

    def _configure_deserializer_tunnel(self, link_name, pipe_sel, dphy_dst):
        """Set MAX96716A to 6Gbps mode, select link, wait for GMSL lock, configure video pipe."""
        des = ox08d10_mode.MAX96716A_DESER_ADDR
        is_link_a = link_name == "A"
        link_enable = 0x01 if is_link_a else 0x02
        ctrl0_6gbps = 0x32 if is_link_a else 0x23
        reset_register = 0x0010 if is_link_a else 0x0012
        reset_value = ctrl0_6gbps if is_link_a else 0x20
        lock_register = 0x0013 if is_link_a else 0x5009

        self.set_max929x_register(des, 0x0010, ctrl0_6gbps)
        self.set_max929x_register(des, 0x0F00, link_enable)
        self.set_max929x_register(des, reset_register, reset_value)
        time.sleep(0.2)

        # Poll for GMSL link lock (bit 3 of lock register)
        locked = False
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            status = self.get_max929x_register(des, lock_register)
            logging.debug(
                "GMSL link %s lock poll: reg=0x%04X val=0x%02X",
                link_name,
                lock_register,
                status,
            )
            if status & 0x08:
                locked = True
                break
            time.sleep(0.05)
        if not locked:
            status = self.get_max929x_register(des, lock_register)
            raise RuntimeError(
                f"GMSL link {link_name} did not lock within 10s -- "
                f"deserializer 0x{des:02X} status reg 0x{lock_register:04X}=0x{status:02X}. "
                "Check camera cable and power."
            )
        logging.info(
            "GMSL link %s locked (reg 0x%04X=0x%02X)", link_name, lock_register, status
        )
        time.sleep(0.1)

        for register, value in (
            (0x0F00, link_enable),  # LINK_EN: enable GMSL link A (0x01) or B (0x02)
            (0x0330, 0x04),  # MIPI_PHY0: 2x4 D-PHY configuration
            (0x0333, 0x4E),  # MIPI_PHY3: D-PHY lane mapping phy1/phy0
            (0x0334, 0xE4),  # MIPI_PHY4: D-PHY lane mapping phy3/phy2
            (0x040A, 0x00),  # MIPI_TX0: 4-lane CSI-2 output
            (0x044A, 0xD0),  # MIPI_TX1: pipe A mapping enable
            (0x048A, 0xD0),  # MIPI_TX2: pipe B mapping enable
            (0x04CA, 0x00),  # MIPI_TX3: pipe C disabled
            (0x0474, dphy_dst),  # MIPI_TX25: pipe A D-PHY destination
            (0x04B4, dphy_dst),  # MIPI_TX...: pipe B D-PHY destination
            (0x031D, 0x00),  # MIPI_PHY13: D-PHY pre-begin timing = 0
            (0x0320, 0x36),  # BACKTOP25: D-PHY timing calibration
            (0x0323, 0x36),  # BACKTOP28: D-PHY timing calibration
            (0x0326, 0x00),  # BACKTOP31: D-PHY timing calibration
            (0x0161, pipe_sel),  # VIDEO_PIPE_SEL: video pipe enable and select
            (0x0332, 0xF4),  # MIPI_PHY2: release D-PHY standby, 4 lanes active
            (0x044B, 0x03),  # MIPI_TX11: enable data type maps 0 and 1
            (0x044D, 0x2C),  # MIPI_TX13: MAP_SRC_0 = 0x2C (RAW12 CSI-2 data type)
            (0x044E, 0x2C),  # MIPI_TX14: MAP_DST_0 = 0x2C (output as RAW12)
            (0x044F, 0xAC),  # MIPI_TX15: MAP_SRC_1 = 0xAC (RAW12 + embedded data flag)
            (0x0450, 0xAC),  # MIPI_TX16: MAP_DST_1 = 0xAC (output embedded)
            (
                0x046D,
                dphy_dst,
            ),  # MIPI_TX45: MAP_DPHY_DEST_3, D-PHY destination for maps
            (0x0313, 0x02),  # BACKTOP12: enable CSI-2 output
        ):
            self.set_max929x_register(des, register, value)

        # Clear MAX96719B loss-of-lock flag
        self.set_max929x_register(
            ox08d10_mode.MAX96719_SER_ADDR, 0x1024, 0x20
        )  # DEC_ERR_FLAG: clear loss-of-lock error flag
        logging.debug("Cleared MAX96719 loss-of-lock flag")
        time.sleep(0.05)

    def _configure_serializer_vendor(self):
        """Configure MAX96719B: default mode, RCLKOUT, I2C address translator, PHY.

        Does NOT release sensor reset -- call _release_ox08d_reset() separately.
        """
        ser = ox08d10_mode.MAX96719_SER_ADDR

        # Default mode
        for register, value, mask in (
            (0x350F, 0x05, 0x37),  # MIPI_RX_0: I2C port A mode configuration
            (0x3511, 0x46, 0x77),  # MIPI_RX_1: I2C port A timing configuration
            (0x358F, 0x05, 0x37),  # MIPI_RX_8: I2C port B mode configuration
            (0x3591, 0x46, 0x77),  # MIPI_RX_9: I2C port B timing configuration
            (0x3013, 0x00, 0x80),  # FSYNC_CFG: disable external frame sync
            (0x0101, 0x00, 0x06),  # REG1: disable wake-up feature
            (0xA005, 0x01, 0x01),  # MIPI_RX_5: assert MIPI reset
            (0x2003, 0x20, 0xE1),  # WM3: enable video line CRC
        ):
            current = self.get_max929x_register(ser, register)
            self.set_max929x_register(ser, register, (current & ~mask) | (value & mask))

        # Enable RCLKOUT (25 MHz reference clock to sensor XVCLK)
        current = self.get_max929x_register(ser, 0x0014)
        self.set_max929x_register(ser, 0x0014, (current & ~0x01) | 0x01)

        # I2C address translator: map sensor virtual address to physical address.
        # The MAX96719B uses 8-bit format (7-bit address left-shifted by 1).
        # Required for proper I2C tunneling through the GMSL link.
        virtual_addr = self._camera_i2c_address << 1
        physical_addr = CAM_I2C_ADDRESS << 1
        for register, value in (
            (0x351A, virtual_addr),  # virtual address (host-side)
            (0x351B, physical_addr),  # physical address (sensor-side)
            (0x351C, 0x00),  # disable second translation entry
            (0x351D, 0x00),
        ):
            self.set_max929x_register(ser, register, value)

        # C-PHY/D-PHY: assert MIPI reset with lane config, set pkt-detect timer,
        # then release MIPI reset and enable video port.
        current = self.get_max929x_register(ser, 0xA005)
        self.set_max929x_register(
            ser, 0xA005, (current & ~0xE1) | 0xE1
        )  # MIPI_RX_5: assert MIPI reset + enable deskew, 4-lane
        current = self.get_max929x_register(ser, 0xA113)
        self.set_max929x_register(
            ser, 0xA113, (current & ~0x02) | 0x02
        )  # MIPI_RX_113: set packet-detect timer to 33 ms
        time.sleep(0.05)
        current = self.get_max929x_register(ser, 0xA005)
        self.set_max929x_register(
            ser, 0xA005, (current & ~0xE1) | 0xE0
        )  # MIPI_RX_5: release MIPI reset, enable deskew + 4-lane
        self.set_max929x_register(ser, 0x2003, 0x21)  # WM3: enable video port
        self.set_max929x_register(ser, 0x0020, 0x9F)  # REG20: ERRB TX enable
        time.sleep(0.05)

    def _release_ox08d_reset(self):
        """Release sensor reset via MAX96719B GPIO3 and wait 500ms for boot."""
        ser = ox08d10_mode.MAX96719_SER_ADDR
        self.set_max929x_register(ser, 0x3049, 0x30)  # assert reset (GPIO3 low)
        time.sleep(0.05)
        self.set_max929x_register(ser, 0x3049, 0x38)  # release reset (GPIO3 high)
        time.sleep(0.5)  # 500ms sensor boot time

    def _configure_deserializer_dphy_output(self, link_name, pipe_sel, dphy_dst):
        """Set MAX96716A D-PHY output configuration."""
        des = ox08d10_mode.MAX96716A_DESER_ADDR
        for register, value in (
            (0x0330, 0x04),  # MIPI_PHY0: 2x4 D-PHY configuration
            (0x0333, 0x4E),  # MIPI_PHY3: D-PHY lane mapping phy1/phy0
            (0x0334, 0xE4),  # MIPI_PHY4: D-PHY lane mapping phy3/phy2
            (0x040A, 0x00),  # MIPI_TX0: 4-lane CSI-2 output
            (0x044A, 0xD0),  # MIPI_TX1: pipe A mapping enable
            (0x048A, 0xD0),  # MIPI_TX2: pipe B mapping enable
            (0x04CA, 0x00),  # MIPI_TX3: pipe C disabled
            (0x0474, dphy_dst),  # MIPI_TX25: pipe A D-PHY destination
            (0x04B4, dphy_dst),  # MIPI_TX...: pipe B D-PHY destination
            (0x031D, 0x00),  # MIPI_PHY13: D-PHY pre-begin timing = 0
            (0x0320, 0x36),  # BACKTOP25: D-PHY timing calibration
            (0x0323, 0x36),  # BACKTOP28: D-PHY timing calibration
            (0x0326, 0x00),  # BACKTOP31: D-PHY timing calibration
            (0x0161, pipe_sel),  # VIDEO_PIPE_SEL: video pipe enable and select
            (0x0332, 0xF4),  # MIPI_PHY2: release D-PHY standby, 4 lanes active
            (0x0005, 0x00),  # REG5: disable LOCK_EN and ERRB_EN
            (0x03EF, 0xC0),  # FSYNC_15F: FSYNC timing configuration
            (
                0x03E2,
                0x00,
            ),  # FSYNC_0E2: trigger FSYNC calibration (250 ms wait required)
            (0x03EA, 0x00),  # FSYNC_0EA: FSYNC timing register
            (0x03EB, 0x00),  # FSYNC_0EB: FSYNC timing register
            (0x03E5, 0xD4),  # FSYNC_0E5: FSYNC period high byte
            (0x03E6, 0xDC),  # FSYNC_0E6: FSYNC period mid byte
            (0x03E7, 0xCB),  # FSYNC_0E7: FSYNC period low byte
            (0x03F1, 0x40),  # FSYNC_0F1: FSYNC enable and output configuration
            (0x02BC, 0x04),  # BACKTOP area: high-speed I/O calibration control
            (0x02BE, 0x11),  # BACKTOP area: high-speed I/O calibration control
            (0x0003, 0x43),  # REG3: software reset and configuration
            (0x02C1, 0x12),  # BACKTOP area: high-speed I/O calibration control
        ):
            self.set_max929x_register(des, register, value)
            if register == 0x03E2:
                time.sleep(0.25)

        for register, value in (
            (0x0161, pipe_sel),  # VIDEO_PIPE_SEL: re-apply pipe enable and select
            (0x044B, 0x03),  # MIPI_TX11: enable data type maps 0 and 1
            (0x044D, 0x2C),  # MIPI_TX13: MAP_SRC_0 = 0x2C (RAW12 CSI-2 data type)
            (0x044E, 0x2C),  # MIPI_TX14: MAP_DST_0 = 0x2C (output as RAW12)
            (0x044F, 0xAC),  # MIPI_TX15: MAP_SRC_1 = 0xAC (RAW12 + embedded data flag)
            (0x0450, 0x2C),  # MIPI_TX16: MAP_DST_1 = 0x2C (output embedded as RAW12)
            (
                0x046D,
                dphy_dst,
            ),  # MIPI_TX45: MAP_DPHY_DEST_3, D-PHY destination for maps
        ):
            self.set_max929x_register(des, register, value)

        self.set_max929x_register(
            des, 0x0332, 0x00
        )  # MIPI_PHY2: hold all lanes in standby
        time.sleep(0.2)
        self.set_max929x_register(
            des, 0x0332, 0xF0
        )  # MIPI_PHY2: release standby, 4 lanes active
        time.sleep(0.05)

    def _set_virtual_channel(self, link):
        """Set OX08D10 virtual channel mapping based on GMSL link.

        Register 0x4813 controls VC mapping; base value 0xE4 is fixed.
        """
        link_index = 1 if link == ox08d10_mode.GmslLink.LINK_B else 0
        vc_id = 0xE4
        vc_id &= ~(0x3 | (0x3 << (link_index * 2)))
        vc_id |= link_index
        self.set_register(0x4813, vc_id)
        logging.info("OX08D10 VC mapping=0x%02X", vc_id)

    # ------------------------------------------------------------------
    # Sensor initialization
    # ------------------------------------------------------------------

    def _init_sensor(self, mode):
        """Initialize OX08D10 for the selected HDR4 PWL12 mode."""
        for table in (
            ox08d10_mode.ox08d10_software_reset,
            ox08d10_mode.ox08d10_system_defaults,
            self._mode_table(mode),
            ox08d10_mode.ox08d10_raw12_output_groups,
            ox08d10_mode.ox08d10_disable_ebd,
            ox08d10_mode.ox08d10_pwl12_blc4,
            ox08d10_mode.ox08d10_hdr_combine_table,
            ox08d10_mode.ox08d10_fsync_internal,
            ox08d10_mode.ox08d10_flip_adjust,
            ox08d10_mode.ox08d10_safety_config_flip,
        ):
            for reg, val in table:
                self._write_table_entry(reg, val)

        current = self.get_register(0x500D)
        self.set_register(0x500D, current & ~0x06)

        for reg, val in ox08d10_mode.ox08d10_isp_clear:
            self._write_table_entry(reg, val)

    # ------------------------------------------------------------------
    # Sensor identity validation
    # ------------------------------------------------------------------

    def get_sensor_id(self):
        try:
            return tuple(
                self.get_register(r)
                for r in (REG_SENSOR_ID_H, REG_SENSOR_ID_M, REG_SENSOR_ID_L)
            )
        except RuntimeError as e:
            logging.warning("Unable to read OX08D10 sensor ID: %s", e)
            return None

    def validate_sensor_identity(self):
        if self.skip_i2c:
            return
        sensor_id = self.get_sensor_id()
        if sensor_id == EXPECTED_SENSOR_ID:
            logging.info("OX08D10 sensor ID confirmed: %02X %02X %02X", *sensor_id)
            return
        raise RuntimeError(
            "OX08D10 sensor ID mismatch at addr=0x%02X: expected %02X %02X %02X, read %s"
            % (
                self._camera_i2c_address,
                *EXPECTED_SENSOR_ID,
                "NAK" if sensor_id is None else "%02X %02X %02X" % sensor_id,
            )
        )

    # ------------------------------------------------------------------
    # Mode selection
    # ------------------------------------------------------------------

    def set_mode(self, ox08d10_mode_set):
        if ox08d10_mode_set.value < ox08d10_mode.Ox08d10_Mode.Unknown.value:
            self._mode = ox08d10_mode_set
            fmt = ox08d10_mode.ox08d10_frame_format[self._mode.value]
            self._received_height = fmt.height  # includes trailing rows
            self._height = self.ACTIVE_HEIGHT  # active image rows only
            self._width = fmt.width
            self._framerate = fmt.framerate
            self._pixel_format = fmt.pixel_format
            self._num_lanes = fmt.num_lanes
            logging.debug(
                "OX08D10 mode set: %dx%d @ %dfps (received_height=%d)",
                self._width,
                self._height,
                self._framerate,
                self._received_height,
            )
        else:
            raise ValueError(f"Unsupported OX08D10 mode: {ox08d10_mode_set}")

    def _mode_table(self, mode):
        if mode == ox08d10_mode.Ox08d10_Mode.OX08D10_MODE_3840X2160_45FPS_RAW12:
            return ox08d10_mode.ox08d10_mode_3840x2160_45fps_raw12
        if mode == ox08d10_mode.Ox08d10_Mode.OX08D10_MODE_3840X2160_30FPS_RAW12:
            return ox08d10_mode.ox08d10_mode_3840x2160_30fps_raw12
        raise ValueError(f"No register table for mode {mode}")

    # ------------------------------------------------------------------
    # Format accessors
    # ------------------------------------------------------------------

    def pixel_format(self):
        return self._pixel_format

    def bayer_format(self):
        # RGGB Bayer order
        return hololink_module.sensors.csi.BayerFormat.RGGB

    def num_lanes(self):
        return self._num_lanes

    # ------------------------------------------------------------------
    # Converter configuration
    # ------------------------------------------------------------------

    def configure_converter(self, converter):
        start_byte = converter.receiver_start_byte()
        transmitted_line_bytes = converter.transmitted_line_bytes(
            self._pixel_format, self._width
        )
        received_line_bytes = converter.received_line_bytes(transmitted_line_bytes)

        # Skip 2 front EBD lines (present even after seq_disable_embedded_data)
        front_embedded_bytes = received_line_bytes * self.FRONT_EMBEDDED_LINES

        # Trailing non-image rows: received_height - EBD - active_height
        trailing_lines = (
            self._received_height - self.FRONT_EMBEDDED_LINES - self._height
        )
        trailing_bytes = received_line_bytes * trailing_lines

        converter.configure(
            start_byte + front_embedded_bytes,
            received_line_bytes,
            self._width,
            self._height,
            self._pixel_format,
            trailing_bytes,
        )

    # ------------------------------------------------------------------
    # Streaming control
    # ------------------------------------------------------------------

    def start(self):
        for reg, val in ox08d10_mode.ox08d10_start:
            self._write_table_entry(reg, val)

    def stop(self):
        for reg, val in ox08d10_mode.ox08d10_stop:
            self._write_table_entry(reg, val)

    # ------------------------------------------------------------------
    # Exposure / gain control
    # ------------------------------------------------------------------

    def set_exposure_reg(
        self, value=0x001E
    ):  # 0x001E = 30 rows of HCG coarse exposure (sensor default)
        value = max(0, min(value, 0xFFFF))
        self.set_register(REG_EXPOSURE_H, (value >> 8) & 0xFF)
        self.set_register(REG_EXPOSURE_L, value & 0xFF)
        time.sleep(ox08d10_mode.OX08D10_WAIT_MS / 1000)

    # ------------------------------------------------------------------
    # Test pattern (ISP pre-process registers)
    # ------------------------------------------------------------------

    REG_ISP_CTRL = 0x5000
    REG_PRE_ISP_ENABLE_START = 0x5004
    REG_TEST_PATTERN_HCG = 0x5240
    REG_TEST_PATTERN_LCG = 0x5440
    REG_TEST_PATTERN_LOFIC = 0x5640
    REG_TEST_PATTERN_VS = 0x5840

    def test_pattern(self, pattern=1):
        pattern_value = ((pattern & 0x3) << 4) | 0x01
        current = self.get_register(self.REG_ISP_CTRL)
        self.set_register(self.REG_ISP_CTRL, (current or 0) | 0x01)
        for offset in range(4):
            reg = self.REG_PRE_ISP_ENABLE_START + offset
            current = self.get_register(reg)
            self.set_register(reg, (current or 0) | 0x01)
        for reg in (
            self.REG_TEST_PATTERN_HCG,
            self.REG_TEST_PATTERN_LCG,
            self.REG_TEST_PATTERN_LOFIC,
            self.REG_TEST_PATTERN_VS,
        ):
            self.set_register(reg, pattern_value)

    # ------------------------------------------------------------------
    # Register I/O -- sensor (16-bit addr, 8-bit data, SCCB/I2C)
    # I2C reaches the sensor via GMSL tunnel through MAX96719B.
    # ------------------------------------------------------------------

    def get_register(self, register, timeout=None):
        if self.skip_i2c:
            logging.debug("Skip i2c")
            return 0
        write_bytes = bytes([register >> 8, register & 0xFF])
        reply = self._i2c_cam.i2c_transaction(self._camera_i2c_address, write_bytes, 1)
        value = reply[0]
        logging.debug(
            "get_register(i2c_addr=0x%02X, reg=0x%04X)=0x%02X",
            self._camera_i2c_address,
            register,
            value,
        )
        return value

    def set_register(self, register, value, timeout=None):
        if self.skip_i2c:
            logging.debug("Skip i2c")
            return
        logging.debug(
            "set_register(i2c_addr=0x%02X, reg=0x%04X, val=0x%02X)",
            self._camera_i2c_address,
            register,
            value,
        )
        write_bytes = bytes([register >> 8, register & 0xFF, value & 0xFF])
        self._i2c_cam.i2c_transaction(self._camera_i2c_address, write_bytes, 0)

    # ------------------------------------------------------------------
    # Register I/O -- serializer / deserializer (MAX96719B, MAX96716A)
    # Each SerDes chip has its own I2cInterfaceV1 client.
    # ------------------------------------------------------------------

    def _i2c_for_serdes(self, i2c_address):
        if i2c_address == ox08d10_mode.MAX96716A_DESER_ADDR:
            return self._i2c_des
        if i2c_address == ox08d10_mode.MAX96719_SER_ADDR:
            return self._i2c_ser
        raise ValueError(f"Unknown SerDes I2C address 0x{i2c_address:02X}")

    def get_max929x_register(self, i2c_address, register, timeout=None):
        if self.skip_i2c:
            logging.debug("Skip i2c")
            return 0
        i2c = self._i2c_for_serdes(i2c_address)
        write_bytes = bytes([register >> 8, register & 0xFF])
        reply = i2c.i2c_transaction(i2c_address, write_bytes, 1)
        value = reply[0]
        logging.debug(
            "get_max929x(i2c_addr=0x%02X, reg=0x%04X)=0x%02X",
            i2c_address,
            register,
            value,
        )
        return value

    def set_max929x_register(self, i2c_address, register, value, timeout=None):
        if self.skip_i2c:
            logging.debug("Skip i2c")
            return
        logging.debug(
            "set_max929x(i2c_addr=0x%02X, reg=0x%04X, val=0x%02X)",
            i2c_address,
            register,
            value,
        )
        i2c = self._i2c_for_serdes(i2c_address)
        write_bytes = bytes([register >> 8, register & 0xFF, value & 0xFF])
        try:
            i2c.i2c_transaction(i2c_address, write_bytes, 0)
        except RuntimeError as e:
            logging.error(
                "%s - Failed: addr=0x%02X reg=0x%04X val=0x%02X",
                e,
                i2c_address,
                register,
                value,
            )
            raise

    def _write_table_entry(self, reg, val):
        if reg == ox08d10_mode.OX08D10_TABLE_WAIT_MS:
            logging.debug("sleep %s ms", val)
            time.sleep(val / 1000)
        elif reg == ox08d10_mode.OX08D10_TABLE_WRITE_MASKED:
            actual_reg, value, mask = val
            current = self.get_register(actual_reg)
            self.set_register(actual_reg, (current & ~mask) | (value & mask))
        else:
            self.set_register(reg, val)

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def log_video_status(self, label):
        status_registers = (
            ("OX08D10_STREAM", self._camera_i2c_address, REG_STREAM),
            ("OX08D10_FORMAT_1F", self._camera_i2c_address, 0x431F),
            ("OX08D10_MIPI_CTRL02", self._camera_i2c_address, 0x4802),
            ("MAX96716_CTRL0", ox08d10_mode.MAX96716A_DESER_ADDR, 0x0010),
            ("MAX96716_CTRL2", ox08d10_mode.MAX96716A_DESER_ADDR, 0x0012),
            ("MAX96716_STATUS", ox08d10_mode.MAX96716A_DESER_ADDR, 0x0013),
            ("MAX96716_LINK_CFG", ox08d10_mode.MAX96716A_DESER_ADDR, 0x0F00),
            ("MAX96716_PIPE_Y", ox08d10_mode.MAX96716A_DESER_ADDR, 0x044B),
            ("MAX96716_PIPE_Y_DT0", ox08d10_mode.MAX96716A_DESER_ADDR, 0x044D),
            ("MAX96719_CC_CTRL", ox08d10_mode.MAX96719_SER_ADDR, 0x0001),
            ("MAX96719_STATUS", ox08d10_mode.MAX96719_SER_ADDR, 0x0013),
            ("MAX96719_RCLKOUT", ox08d10_mode.MAX96719_SER_ADDR, 0x0014),
            ("MAX96719_GPIO3_RST", ox08d10_mode.MAX96719_SER_ADDR, 0x3049),
            ("MAX96719_PHY_LANES", ox08d10_mode.MAX96719_SER_ADDR, 0xA005),
            ("MAX96719_VIDEO_TX", ox08d10_mode.MAX96719_SER_ADDR, 0x2003),
            ("MAX96719_TUN_PKT", ox08d10_mode.MAX96719_SER_ADDR, 0xA113),
        )
        for name, i2c_address, register in status_registers:
            try:
                if i2c_address == self._camera_i2c_address:
                    value = self.get_register(register)
                else:
                    value = self.get_max929x_register(i2c_address, register)
            except RuntimeError as e:
                logging.warning(
                    "%s: unable to read %s reg=0x%04X: %s", label, name, register, e
                )
                continue
            logging.info(
                "%s: %s addr=0x%02X reg=0x%04X val=0x%02X",
                label,
                name,
                i2c_address,
                register,
                value,
            )
