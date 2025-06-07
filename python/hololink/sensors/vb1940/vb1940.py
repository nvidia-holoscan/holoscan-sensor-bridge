"""
SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import time
from collections import OrderedDict
from enum import Enum

import hololink as hololink_module

from . import li_i2c_expander, vb1940_mode

# Camera info
DRIVER_NAME = "VB1940"
VERSION = 1

# Camera I2C address.
CAM_I2C_ADDRESS = 0x10

# register map
DEVICE_REVISION_REG = 0x0004
SYSTEM_UP_REG = 0x0514
BOOT_REG = 0x0515
SW_STBY_REG = 0x0516
STREAMING_REG = 0x0517
SYSTEM_FSM_STATE_REG = 0x0044
BOOT_FSM_REG = 0x0200


class system_fsm_state(Enum):
    HW_STBY = 0x0
    SYSTEM_UP = 0x1
    BOOT = 0x2
    SW_STBY = 0x3
    STREAMING = 0x4
    STALL = 0x5
    HALT = 0x6


class boot_fsm_state(Enum):
    HW_STBY = 0x00
    COLD_BOOT = 0x01
    CLOCK_INIT = 0x02
    NVM_DWLD = 0x10
    NVM_UNPACK = 0x11
    SYSTEM_BOOT = 0x12
    NONCE_GRNERATION = 0x20
    EPH_KEYS_GENERATION = 0x21
    WAIT_CERTIFICATE = 0x22
    CERTIFICATE_PARSING = 0x23
    CERIFICATE_VERIF_ROOT = 0x24
    CERTIFICATE_VERIF_USER = 0x25
    CERTIFICATE_CHECK_FIELDS = 0x26
    ECDH = 0x30
    ECDH_SS_GEN = 0x31
    ECDH_MASTER_KEY_GEN = 0x32
    ECDH_SESSION_KEY_GEN = 0x33
    AUTHENTICATION = 0x40
    AUTHENTICATION_MSG_CREATE = 0x41
    AUTHENTICATION_MSG_SIGN = 0x42
    PROVISIONING = 0x50
    PROVISIONING_UID = 0x51
    PROVISIONING_EC_PRIV_KEY = 0x52
    PROVISIONING_EC_PUB_KEY = 0x53
    CTM_PROVISIONING_CID = 0x54
    CTM_PROVISIONING_OEM_ROOT_KEY = 0x54
    PAIRING = 0x55
    FWP_SETUP = 0x60
    VTP_SETUP = 0x61
    FA_RETURN = 0x70
    BOOT_WAITING_CMD = 0x80
    BOOT_COMPLETED = 0xBC


class Vb1940Cam:
    def __init__(
        self,
        hololink_channel,
        i2c_bus=hololink_module.CAM_I2C_BUS,
        expander_configuration=0,
    ):
        self._hololink = hololink_channel.hololink()
        i2c_bus = hololink_module.CAM_I2C_BUS + expander_configuration
        self._i2c = self._hololink.get_i2c(i2c_bus)
        self._mode = vb1940_mode.Vb1940_Mode.Unknown
        # Configure i2c expander on the Leopard board
        self._i2c_expander = li_i2c_expander.LII2CExpander(self._hololink, i2c_bus)
        if expander_configuration == 1:
            self._i2c_expander_configuration = (
                li_i2c_expander.I2C_Expander_Output_EN.OUTPUT_2
            )
        else:
            self._i2c_expander_configuration = (
                li_i2c_expander.I2C_Expander_Output_EN.OUTPUT_1
            )

    def get_device_id(self):
        ret = self.get_register_32(DEVICE_REVISION_REG)
        logging.debug("Device ID:0x%X" % ret)

    def status_check(self):
        count = 0
        while True:
            time.sleep(10 / 1000)
            ret = self.get_register(SYSTEM_FSM_STATE_REG)
            count += 1
            assert count < 30, (
                "Incorrect system fsm state:0x%X. Please reconnect camera" % ret
            )
            logging.debug("system fsm state:0x%X" % ret)
            if ret == system_fsm_state.SYSTEM_UP.value:
                break
        time.sleep(10 / 1000)
        self.set_register_8(SYSTEM_UP_REG, 0x01)
        time.sleep(10 / 1000)
        while True:
            time.sleep(10 / 1000)
            ret = self.get_register(SYSTEM_FSM_STATE_REG)
            logging.debug("system fsm state:0x%X" % ret)
            if ret == system_fsm_state.BOOT.value:
                break
        time.sleep(10 / 1000)
        while True:
            time.sleep(10 / 1000)
            ret = self.get_register(BOOT_FSM_REG)
            logging.debug("boot fsm state:0x%X" % ret)
            if ret == boot_fsm_state.WAIT_CERTIFICATE.value:
                break

    def do_secure_boot(self):
        # load certificate
        logging.debug("##secure boot-load certificate")
        self.write_certificate()
        self.set_register_8(BOOT_REG, 0x01)
        while True:
            time.sleep(10 / 1000)
            ret = self.get_register(BOOT_FSM_REG)
            logging.debug("boot fsm state:0x%X" % ret)
            if ret == boot_fsm_state.BOOT_WAITING_CMD.value:
                break
        # load fwp
        logging.debug("##secure boot-load FWP")
        self.write_fw()
        self.set_register_8(BOOT_REG, 0x02)
        while True:
            time.sleep(10 / 1000)
            ret = self.get_register(BOOT_FSM_REG)
            logging.debug("boot fsm state:0x%X" % ret)
            if ret == boot_fsm_state.FWP_SETUP.value:
                break
        # end boot
        logging.debug("##secure boot-end boot")
        self.set_register_8(BOOT_REG, 0x10)

    def write_certificate(self):
        start_addr = vb1940_mode.VB1940_CERTIFICATE_START_ADDR
        data_size = len(vb1940_mode.VB1940_CERTIFICATE)
        page_size = vb1940_mode.VB1940_PAGE_SIZE
        page_count = (data_size + page_size - 1) // page_size
        for id in range(page_count):
            offset = id * page_size
            register = start_addr + offset
            data_buffer = vb1940_mode.VB1940_CERTIFICATE[
                offset : offset + min(data_size - offset, page_size)
            ]
            if data_buffer:
                self.set_register_buffer(register, data_buffer)

    def write_fw(self):
        start_addr = vb1940_mode.VB1940_FWP_START_ADDR
        data_size = len(vb1940_mode.VB1940_FW)
        page_size = vb1940_mode.VB1940_PAGE_SIZE
        page_count = (data_size + page_size - 1) // page_size
        for id in range(page_count):
            offset = id * page_size
            register = start_addr + offset
            data_buffer = vb1940_mode.VB1940_FW[
                offset : offset + min(data_size - offset, page_size)
            ]
            if data_buffer:
                self.set_register_buffer(register, data_buffer)

    def write_vt_patch(self):
        page_size = vb1940_mode.VB1940_PAGE_SIZE
        logging.debug("VT_PATCH: LEDC_RAM")
        start_addr = vb1940_mode.LDEC_RAM_CONTENT_START_ADDR
        data_size = len(vb1940_mode.LDEC_RAM_CONTENT)
        page_count = (data_size + page_size - 1) // page_size
        for id in range(page_count):
            offset = id * page_size
            register = start_addr + offset
            data_buffer = vb1940_mode.LDEC_RAM_CONTENT[
                offset : offset + min(data_size - offset, page_size)
            ]
            if data_buffer:
                self.set_register_buffer(register, data_buffer)

        logging.debug("VT_PATCH: RD_RAM_SEQ_1")
        start_addr = vb1940_mode.RD_RAM_SEQ_1_CONTENT_START_ADDR
        data_size = len(vb1940_mode.RD_RAM_SEQ_1_CONTENT)
        page_count = (data_size + page_size - 1) // page_size
        for id in range(page_count):
            offset = id * page_size
            register = start_addr + offset
            data_buffer = vb1940_mode.RD_RAM_SEQ_1_CONTENT[
                offset : offset + min(data_size - offset, page_size)
            ]
            if data_buffer:
                self.set_register_buffer(register, data_buffer)

        logging.debug("VT_PATCH: GT_RAM_PAT")
        start_addr = vb1940_mode.GT_RAM_PAT_CONTENT_START_ADDR
        data_size = len(vb1940_mode.GT_RAM_PAT_CONTENT)
        page_count = (data_size + page_size - 1) // page_size
        for id in range(page_count):
            offset = id * page_size
            register = start_addr + offset
            data_buffer = vb1940_mode.GT_RAM_PAT_CONTENT[
                offset : offset + min(data_size - offset, page_size)
            ]
            if data_buffer:
                self.set_register_buffer(register, data_buffer)

        logging.debug("VT_PATCH: GT_RAM_SQE_1")
        start_addr = vb1940_mode.GT_RAM_SEQ_1_CONTENT_START_ADDR
        data_size = len(vb1940_mode.GT_RAM_SEQ_1_CONTENT)
        page_count = (data_size + page_size - 1) // page_size
        for id in range(page_count):
            offset = id * page_size
            register = start_addr + offset
            data_buffer = vb1940_mode.GT_RAM_SEQ_1_CONTENT[
                offset : offset + min(data_size - offset, page_size)
            ]
            if data_buffer:
                self.set_register_buffer(register, data_buffer)

        logging.debug("VT_PATCH: GT_RAM_SQE_2")
        start_addr = vb1940_mode.GT_RAM_SEQ_2_CONTENT_START_ADDR
        data_size = len(vb1940_mode.GT_RAM_SEQ_2_CONTENT)
        page_count = (data_size + page_size - 1) // page_size
        for id in range(page_count):
            offset = id * page_size
            register = start_addr + offset
            data_buffer = vb1940_mode.GT_RAM_SEQ_2_CONTENT[
                offset : offset + min(data_size - offset, page_size)
            ]
            if data_buffer:
                self.set_register_buffer(register, data_buffer)

        logging.debug("VT_PATCH: GT_RAM_SQE_3")
        start_addr = vb1940_mode.GT_RAM_SEQ_3_CONTENT_START_ADDR
        data_size = len(vb1940_mode.GT_RAM_SEQ_3_CONTENT)
        page_count = (data_size + page_size - 1) // page_size
        for id in range(page_count):
            offset = id * page_size
            register = start_addr + offset
            data_buffer = vb1940_mode.GT_RAM_SEQ_3_CONTENT[
                offset : offset + min(data_size - offset, page_size)
            ]
            if data_buffer:
                self.set_register_buffer(register, data_buffer)

        logging.debug("VT_PATCH: GT_RAM_SQE_4")
        start_addr = vb1940_mode.GT_RAM_SEQ_4_CONTENT_START_ADDR
        data_size = len(vb1940_mode.GT_RAM_SEQ_4_CONTENT)
        page_count = (data_size + page_size - 1) // page_size
        for id in range(page_count):
            offset = id * page_size
            register = start_addr + offset
            data_buffer = vb1940_mode.GT_RAM_SEQ_4_CONTENT[
                offset : offset + min(data_size - offset, page_size)
            ]
            if data_buffer:
                self.set_register_buffer(register, data_buffer)

        logging.debug("VT_PATCH: RD_RAM_PAT")
        start_addr = vb1940_mode.RD_RAM_PAT_CONTENT_START_ADDR
        data_size = len(vb1940_mode.RD_RAM_PAT_CONTENT)
        page_count = (data_size + page_size - 1) // page_size
        for id in range(page_count):
            offset = id * page_size
            register = start_addr + offset
            data_buffer = vb1940_mode.RD_RAM_PAT_CONTENT[
                offset : offset + min(data_size - offset, page_size)
            ]
            if data_buffer:
                self.set_register_buffer(register, data_buffer)

    def setup_clock(self):
        # set the clock driver.
        self._hololink.setup_clock(
            hololink_module.renesas_bajoran_lite_ts2.device_configuration()
        )

    def configure(self, mode_set):
        # Make sure this is a version we know about.
        version = self.get_version()
        logging.info("version=%s" % (version,))
        assert version == VERSION

        # get device id
        logging.debug("##1.get device id")
        self.get_device_id()
        time.sleep(10 / 1000)

        status = self.get_register(SYSTEM_FSM_STATE_REG)
        logging.debug("system fsm state:0x%X" % status)
        if status == system_fsm_state.SW_STBY.value:
            # configure the camera based on the mode
            logging.debug("##sensor configuration")
            self.configure_camera(mode_set)
        else:
            logging.debug("##2.status check")
            self.status_check()
            time.sleep(10 / 1000)
            # start sensor
            logging.debug("##3.start sensor")
            self.set_register_8(SYSTEM_UP_REG, 0x01)

            while True:
                time.sleep(10 / 1000)
                ret = self.get_register(SYSTEM_FSM_STATE_REG)
                logging.debug("system fsm state:0x%X" % ret)
                if ret == system_fsm_state.BOOT.value:
                    break

            while True:
                time.sleep(10 / 1000)
                ret = self.get_register(BOOT_FSM_REG)
                logging.debug("boot fsm state:0x%X" % ret)
                if ret == boot_fsm_state.WAIT_CERTIFICATE.value:
                    break

            # secure boot
            logging.debug("##4.secure boot")
            self.do_secure_boot()
            time.sleep(10 / 1000)

            ret = self.get_register(SYSTEM_FSM_STATE_REG)
            logging.debug("system fsm state:0x%X" % ret)
            time.sleep(10 / 1000)
            ret = self.get_register(SYSTEM_FSM_STATE_REG)
            logging.debug("system fsm state:0x%X" % ret)

            # update VT PATCH to RAM
            logging.debug("##5.VT_PATCH")
            self.write_vt_patch()

            # configure the camera based on the mode
            logging.debug("##6.sensor configuration")
            self.configure_camera(mode_set)

    def start(self):
        """Start Streaming"""
        self._running = True
        #
        # Setting these register is time-consuming.
        for reg, val in vb1940_mode.vb1940_start:
            if reg == vb1940_mode.VB1940_TABLE_WAIT_MS:
                time.sleep(val / 1000)  # the val is in ms
            else:
                self.set_register_8(reg, val)
        count = 0
        while True:
            time.sleep(vb1940_mode.VB1940_WAIT_MS_START / 1000)
            ret = self.get_register(SW_STBY_REG)
            logging.debug("SW_STBY state:0x%X" % ret)
            count += 1
            if count == 30:
                break
            if ret == 0x00 or ret == 0x01:
                break
        time.sleep(10 / 1000)
        count = 0
        while True:
            time.sleep(vb1940_mode.VB1940_WAIT_MS_START / 1000)
            ret = self.get_register(SYSTEM_FSM_STATE_REG)
            logging.debug("system fsm state:0x%X" % ret)
            count += 1
            if count == 30:
                break
            if ret == system_fsm_state.STREAMING.value:
                break

    def stop(self):
        """Stop Streaming"""
        for reg, val in vb1940_mode.vb1940_stop:
            if reg == vb1940_mode.VB1940_TABLE_WAIT_MS:
                time.sleep(val / 1000)  # the val is in ms
            else:
                self.set_register_8(reg, val)
        count = 0
        while True:
            time.sleep(vb1940_mode.VB1940_WAIT_MS_START / 1000)
            ret = self.get_register(STREAMING_REG)
            logging.debug("SW_STBY state:0x%X" % ret)
            count += 1
            if count == 30:
                break
            if ret == 0x00 or ret == 0x01:
                break
        time.sleep(10 / 1000)
        count = 0
        while True:
            time.sleep(vb1940_mode.VB1940_WAIT_MS_START / 1000)
            ret = self.get_register(SYSTEM_FSM_STATE_REG)
            logging.debug("system fsm state:0x%X" % ret)
            count += 1
            if count == 30:
                break
            if ret == system_fsm_state.SW_STBY.value:
                break
        self._running = False

    def get_version(self):
        # TODO: get the version or the name of the sensor from the sensor
        return VERSION

    def get_register(self, register):
        logging.debug("get_register(register=%d(0x%X))" % (register, register))
        self._i2c_expander.configure(self._i2c_expander_configuration.value)
        write_bytes = bytearray(100)
        serializer = hololink_module.Serializer(write_bytes)
        serializer.append_uint16_be(register)
        read_byte_count = 1
        reply = self._i2c.i2c_transaction(
            CAM_I2C_ADDRESS, write_bytes[: serializer.length()], read_byte_count
        )
        deserializer = hololink_module.Deserializer(reply)
        r = deserializer.next_uint8()
        logging.debug(
            "get_register(register=%d(0x%X))=%d(0x%X)" % (register, register, r, r)
        )
        return r

    def get_register_32(self, register):
        logging.debug("get_register(register=%d(0x%X))" % (register, register))
        self._i2c_expander.configure(self._i2c_expander_configuration.value)
        write_bytes = bytearray(100)
        serializer = hololink_module.Serializer(write_bytes)
        serializer.append_uint16_be(register)
        read_byte_count = 4
        reply = self._i2c.i2c_transaction(
            CAM_I2C_ADDRESS, write_bytes[: serializer.length()], read_byte_count
        )
        deserializer = hololink_module.Deserializer(reply)
        r = deserializer.next_uint32_be()
        logging.debug(
            "get_register(register=%d(0x%X))=%d(0x%X)" % (register, register, r, r)
        )
        return r

    def set_register_8(self, register, value, timeout=None):
        logging.debug(
            "set_register_8(register=%d(0x%X), value=%d(0x%X))"
            % (register, register, value, value)
        )
        self._i2c_expander.configure(self._i2c_expander_configuration.value)
        write_bytes = bytearray(100)
        serializer = hololink_module.Serializer(write_bytes)
        serializer.append_uint16_be(register)
        serializer.append_uint8(value)
        read_byte_count = 0
        self._i2c.i2c_transaction(
            CAM_I2C_ADDRESS,
            write_bytes[: serializer.length()],
            read_byte_count,
            timeout=timeout,
        )

    def set_register_16(self, register, value, timeout=None):
        logging.debug(
            "set_register_16(register=%d(0x%X), value=%d(0x%X))"
            % (register, register, value, value)
        )
        self._i2c_expander.configure(self._i2c_expander_configuration.value)
        write_bytes = bytearray(100)
        serializer = hololink_module.Serializer(write_bytes)
        serializer.append_uint16_be(register)
        serializer.append_uint16_be(value)
        read_byte_count = 0
        self._i2c.i2c_transaction(
            CAM_I2C_ADDRESS,
            write_bytes[: serializer.length()],
            read_byte_count,
            timeout=timeout,
        )

    def set_register_32(self, register, value, timeout=None):
        logging.debug(
            "set_register_16(register=%d(0x%X), value=%d(0x%X))"
            % (register, register, value, value)
        )
        self._i2c_expander.configure(self._i2c_expander_configuration.value)
        write_bytes = bytearray(100)
        serializer = hololink_module.Serializer(write_bytes)
        serializer.append_uint16_be(register)
        serializer.append_uint32_be(value)
        read_byte_count = 0
        self._i2c.i2c_transaction(
            CAM_I2C_ADDRESS,
            write_bytes[: serializer.length()],
            read_byte_count,
            timeout=timeout,
        )

    def set_register_buffer(self, register, data_buffer, timeout=None):
        logging.debug(
            "set_register_buffer(register=%d(0x%X), data size=%d)"
            % (register, register, len(data_buffer))
        )
        self._i2c_expander.configure(self._i2c_expander_configuration.value)
        write_bytes = bytearray(256)
        serializer = hololink_module.Serializer(write_bytes)
        serializer.append_uint16_be(register)
        serializer.append_buffer(bytearray(data_buffer))
        read_byte_count = 0
        self._i2c.i2c_transaction(
            CAM_I2C_ADDRESS,
            write_bytes[: serializer.length()],
            read_byte_count,
            timeout=timeout,
        )

    def configure_camera(self, mode_set):
        self.set_mode(mode_set)

        mode_list = OrderedDict()

        if mode_set.value == vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS.value:
            mode_list = vb1940_mode.vb1940_mode_2560X1984_30fps
        else:
            logging.error(f"{mode_set} mode is not present.")

        for reg, val in mode_list:
            if reg == vb1940_mode.VB1940_TABLE_WAIT_MS:
                logging.debug(f"sleep {val} ms")
                time.sleep(val / 1000)  # the val is in ms
            else:
                self.set_register_8(reg, val)

    def set_exposure_reg(self, value=0x0014):
        """
        The minimum integration time is 30us(4lines).
        """
        if value < 0x0004:
            logging.warn(f"Exposure value {value} is lower than the minimum.")
            value = 0x0004

        if value > 0xFFFF:
            logging.warn(f"Exposure value {value} is higher than the maximum.")
            value = 0xFFFF
        self.set_register_16(vb1940_mode.REG_EXP, value)
        time.sleep(vb1940_mode.VB1940_WAIT_MS / 1000)

    def set_analog_gain_reg(self, value=0x00):
        if value < 0x00:
            logging.warn(f"Gain value {value} is lower than the minimum.")
            value = 0x00

        if value > 0x18:
            logging.warn(f"Gain value {value} is more than maximum.")
            value = 0x18

        self.set_register_8(vb1940_mode.REG_AG, value)
        time.sleep(vb1940_mode.VB1940_WAIT_MS / 1000)

    def set_mode(self, mode_set):
        if mode_set.value < len(vb1940_mode.Vb1940_Mode):
            self._mode = mode_set
            mode = vb1940_mode.vb1940_frame_format[self._mode.value]
            self._height = mode.height
            self._width = mode.width
            self._pixel_format = mode.pixel_format
        else:
            logging.error("Incorrect mode for VB1940")
            self._mode = -1

    def configure_converter(self, converter):
        # where do we find the first received byte?
        start_byte = converter.receiver_start_byte()
        transmitted_line_bytes = converter.transmitted_line_bytes(
            self._pixel_format, self._width
        )
        received_line_bytes = converter.received_line_bytes(transmitted_line_bytes)
        assert self._pixel_format == hololink_module.sensors.csi.PixelFormat.RAW_10
        # sensor has 1 line of status before the real image data starts
        start_byte += received_line_bytes
        # sensor has 2 line of status after the real image data is complete
        trailing_bytes = received_line_bytes * 2
        converter.configure(
            start_byte,
            received_line_bytes,
            self._width,
            self._height,
            self._pixel_format,
            trailing_bytes,
        )

    def pixel_format(self):
        return self._pixel_format

    def bayer_format(self):
        return hololink_module.sensors.csi.BayerFormat.GBRG
