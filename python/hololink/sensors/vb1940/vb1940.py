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
import struct
import time
from collections import OrderedDict
from enum import Enum

import hololink as hololink_module

from . import vb1940_mode

# Camera info
DRIVER_NAME = "VB1940"
VERSION = 1

# Camera I2C address.
CAM_I2C_ADDRESS = 0x10
EEPROM_I2C_ADDRESS = 0x51
EEPROM_MAX_PAGE_NUM = 256
EEPROM_PAGE_SIZE = 64
CALIB_SIZE = 256
VCL_EN_I2C_ADDRESS_1 = 0x70
VCL_EN_I2C_ADDRESS_2 = 0x71
VCL_PWM_I2C_ADDRESS = 0x21

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


class Vb1940Cam(hololink_module.Synchronizable):
    def __init__(
        self,
        hololink_channel,
        vsync=hololink_module.Synchronizer.null_synchronizer(),
    ):
        super().__init__()
        self._hololink = hololink_channel.hololink()
        enumeration_metadata = hololink_channel.enumeration_metadata()
        i2c_bus = enumeration_metadata["i2c_bus"]
        self._i2c = self._hololink.get_i2c(i2c_bus)
        self._mode = vb1940_mode.Vb1940_Mode.Unknown
        self._vsync = vsync
        self._width = None
        self._height = None

    def parse_calibration_data_dict(self, data):
        if len(data) < 32:
            logging.error(
                "Incomplete calibration block ‑ expected 32 doubles, got %d", len(data)
            )
            return {}

        data_parsed_dict = {}
        left_intrinsic_parameter = [
            [data[0], 0, data[2]],
            [0, data[1], data[3]],
            [0, 0, 1],
        ]
        data_parsed_dict.update({"left_intrinsic_parameter": left_intrinsic_parameter})
        left_distortion_parameters = data[4 : 4 + 8] + list([0] * 6)
        data_parsed_dict.update(
            {"left_distortion_parameters": left_distortion_parameters}
        )
        right_intrinsic_parameter = [
            [data[12], 0, data[14]],
            [0, data[13], data[15]],
            [0, 0, 1],
        ]
        data_parsed_dict.update(
            {"right_intrinsic_parameter": right_intrinsic_parameter}
        )
        right_distortion_parameters = data[16 : 16 + 8] + list([0] * 6)
        data_parsed_dict.update(
            {"right_distortion_parameters": right_distortion_parameters}
        )
        R = data[24 : 24 + 3]
        data_parsed_dict.update({"R": R})
        T = data[27 : 27 + 3]
        data_parsed_dict.update({"T": T})
        sn = data[31]
        data_parsed_dict.update({"sn": sn})
        return data_parsed_dict

    def get_calibration_data(self, part=0):
        """
        Read the calibration data from EEPROM.

        Following is EEPROM layout of every part of
        calibration data. Each part occupies 256 bytes.
        The prefixs 'L' and 'R' denote left and right,
        respectively.
        |-8----8-----8-----8-----8-----8-----8-----8-|
        L_fx  L_fy  L_cx  L_cy  L_k1  L_k2  L_p1  L_p2
        L_k3  L_k4  L_k5  L_k6  R_fx  R_fy  R_cx  R_cy
        R_k1  R_k2  R_p1  R_p2  R_k3  R_k4  R_k5  R_k6
        Rx    Ry    Rz    Tx    Ty    Tz    sn

        part(int): indicates which part to be read.
                   0: data only for RGB mode
                   1: data onlyfor IR mode
                   2: both data for RGB and IR mode
        """
        calib_data_dict = {"RGB": {}, "IR": {}}
        calib_pages = CALIB_SIZE // EEPROM_PAGE_SIZE
        step = 8
        # RGB
        if part == 0 or part == 2:
            # get raw data
            rgb_pages = range(calib_pages)
            rgb_calib_data = []
            for page in rgb_pages:
                data = self.get_eeprom_page(page)
                rgb_calib_data.extend(data)
            # convert into double
            rgb_calib_data_parsed = []
            for i in range(CALIB_SIZE // step):
                try:
                    parsed_data = struct.unpack(
                        "!d", bytearray(rgb_calib_data[i * step : (i + 1) * step])
                    )[0]
                    rgb_calib_data_parsed.append(parsed_data)
                except Exception as exception:
                    logging.error(
                        f"Error in parsing RGB calibration data, {exception=}"
                    )
            #
            rgb_calib_data_dict = self.parse_calibration_data_dict(
                rgb_calib_data_parsed
            )
            calib_data_dict["RGB"] = rgb_calib_data_dict
        # IR
        if part == 1 or part == 2:
            # get raw data
            ir_pages = range(calib_pages, calib_pages * 2)
            ir_calib_data = []
            for page in ir_pages:
                data = self.get_eeprom_page(page)
                ir_calib_data.extend(data)
            # convert into double
            ir_calib_data_parsed = []
            for i in range(CALIB_SIZE // step):
                try:
                    parsed_data = struct.unpack(
                        "!d", bytearray(ir_calib_data[i * step : (i + 1) * step])
                    )[0]
                    ir_calib_data_parsed.append(parsed_data)
                except Exception as exception:
                    logging.error(f"Error in parsing IR calibration data, {exception=}")
            #
            ir_calib_data_dict = self.parse_calibration_data_dict(ir_calib_data_parsed)
            calib_data_dict["IR"] = ir_calib_data_dict
        return calib_data_dict

    def set_eeprom_register(self, register, value, timeout=None):
        logging.debug(
            "set_eeprom_register(register=%d(0x%X), value=%d(0x%X))"
            % (register, register, value, value)
        )
        write_bytes = bytearray(100)
        serializer = hololink_module.Serializer(write_bytes)
        serializer.append_uint16_be(register)
        serializer.append_uint8(value)
        read_byte_count = 0
        self._i2c.i2c_transaction(
            EEPROM_I2C_ADDRESS,
            write_bytes[: serializer.length()],
            read_byte_count,
            timeout=timeout,
        )

    def set_eeprom_page(self, page_num, page_offset, data_buffer, timeout=None):
        """
        Page write of EEPROM.
        Up to 64 bytes can be written in one Write cycle. The internal byte address counter is
        automatically incremented after each data byte is loaded. If more than 64 data bytes
        to be transmitted, then earlier bytes within the selected page will be overwritten by
        later bytes. To avoid this "wrap−around", check the size of 'data_buffer' to be written
        firstly.

        page_num: the number of page to be written, range from 0 to 255.
        page_offset: the first byte to be written in the selected page, range from 0 to 63
        data_buffer: the buffer of data to be written
        """
        logging.debug(
            "set_eeprom_page(page_num=%d(0x%X), page_offset=%d(0x%X), data_len=%d)"
            % (page_num, page_num, page_offset, page_offset, len(data_buffer))
        )
        assert (
            page_num < EEPROM_MAX_PAGE_NUM
        ), "The number of page should be in a range from 0 to 255."
        assert (
            page_offset + len(data_buffer) <= EEPROM_PAGE_SIZE
        ), "page_offset(%d) + data_len(%d) should not be greater than 64." % (
            page_offset,
            len(data_buffer),
        )
        register = (page_num << 6) + page_offset
        write_bytes = bytearray(66)
        serializer = hololink_module.Serializer(write_bytes)
        serializer.append_uint16_be(register)
        serializer.append_buffer(bytearray(data_buffer))
        read_byte_count = 0
        self._i2c.i2c_transaction(
            EEPROM_I2C_ADDRESS,
            write_bytes[: serializer.length()],
            read_byte_count,
            timeout=timeout,
        )

    def get_eeprom_register(self, register):
        write_bytes = bytearray(100)
        serializer = hololink_module.Serializer(write_bytes)
        serializer.append_uint16_be(register)
        read_byte_count = 1
        reply = self._i2c.i2c_transaction(
            EEPROM_I2C_ADDRESS, write_bytes[: serializer.length()], read_byte_count
        )
        deserializer = hololink_module.Deserializer(reply)
        r = deserializer.next_uint8()
        logging.debug(
            "get_eeprom_register(register=%d(0x%X),value=%d(0x%X))"
            % (register, register, r, r)
        )
        return r

    def get_eeprom_page(self, page_num=0, page_offset=0, data_len=64):
        assert (
            page_num < EEPROM_MAX_PAGE_NUM
        ), "The number of page should be in a range from 0 to 255."
        assert (
            page_offset + data_len
        ) <= EEPROM_PAGE_SIZE, f"{page_offset=} + {data_len=} should not be greater than {EEPROM_PAGE_SIZE=}."
        register = (page_num << 6) + page_offset
        write_bytes = bytearray(100)
        serializer = hololink_module.Serializer(write_bytes)
        serializer.append_uint16_be(register)
        read_byte_count = data_len
        reply = self._i2c.i2c_transaction(
            EEPROM_I2C_ADDRESS, write_bytes[: serializer.length()], read_byte_count
        )
        deserializer = hololink_module.Deserializer(reply)
        r = deserializer.next_buffer(read_byte_count)
        assert len(r) == read_byte_count, "read %d != %d" % (len(r), read_byte_count)
        logging.debug(
            "get_eeprom_page(register=%d(0x%X),buffer=%s)"
            % (register, register, bytearray(r))
        )
        return list(bytearray(r))

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
        self._vsync.attach(self)
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
        self._vsync.detach(self)
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
        elif (
            mode_set.value == vb1940_mode.Vb1940_Mode.VB1940_MODE_1920X1080_30FPS.value
        ):
            mode_list = vb1940_mode.vb1940_mode_1920X1080_30fps
        elif (
            mode_set.value
            == vb1940_mode.Vb1940_Mode.VB1940_MODE_2560X1984_30FPS_8BIT.value
        ):
            mode_list = vb1940_mode.vb1940_mode_2560X1984_30fps_8bit
        else:
            logging.error(f"{mode_set} mode is not present.")

        for reg, val in mode_list:
            if reg == vb1940_mode.VB1940_TABLE_WAIT_MS:
                logging.debug(f"sleep {val} ms")
                time.sleep(val / 1000)  # the val is in ms
            else:
                if reg == 0xAC6 and (self._vsync.is_enabled()):
                    val = 0x01
                self.set_register_8(reg, val)

    def set_exposure_reg(self, value=0x0014):
        # The minimum integration time is 30us(4lines).
        # value: integration time in lines, in little endian.
        if value < 0x0004:
            logging.warn(f"Exposure value {value} is lower than the minimum.")
            value = 0x0004

        if value > 0xFFFF:
            logging.warn(f"Exposure value {value} is higher than the maximum.")
            value = 0xFFFF
        # if set_register_16 is used to set exposure, change the value passed in into big endian
        self.set_register_8(vb1940_mode.REG_EXP, value & 0xFF)
        self.set_register_8(vb1940_mode.REG_EXP + 1, (value >> 8) & 0xFF)
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
        # status lines are not converted
        status_line_bytes = 0
        if self._pixel_format == hololink_module.sensors.csi.PixelFormat.RAW_8:
            status_line_bytes = int(self._width)
        elif self._pixel_format == hololink_module.sensors.csi.PixelFormat.RAW_10:
            status_line_bytes = int(self._width * 10 / 8)
        elif self._pixel_format == hololink_module.sensors.csi.PixelFormat.RAW_12:
            status_line_bytes = int(self._width * 12 / 8)
        status_line_bytes = converter.received_line_bytes(status_line_bytes)
        # sensor has 1 line of status before the real image data starts
        start_byte += status_line_bytes
        # sensor has 2 line of status after the real image data is complete
        trailing_bytes = status_line_bytes * 2
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

    def width(self):
        if self._width is None:
            raise RuntimeError(
                "Image width is unavailable; call configure_camera first."
            )
        return self._width

    def height(self):
        if self._height is None:
            raise RuntimeError(
                "Image height is unavailable; call configure_camera first."
            )
        return self._height
