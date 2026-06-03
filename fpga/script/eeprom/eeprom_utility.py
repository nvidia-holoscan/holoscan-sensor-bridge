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

import argparse
import math
import socket
import sys
import time

# Could add checking for 0b, 0x, etc, and convert to 2 character hex
# print(f"{write_byte_addr}, {write_byte_data}")
parser = argparse.ArgumentParser(
    description="EEPROM Utility with configurable APB clock frequency"
)
parser.add_argument(
    "--apb_clk_freq",
    type=float,
    default=156.25e6 / 8,
    help="APB clock frequency in Hz (used to calculate clk_cnt) [default: 156.25e6/8 = 19531250 Hz]",
)
parser.add_argument(
    "--host",
    default="192.168.0.101",
    help="Local bind address for UDP socket (ECB source) [default: 192.168.0.101]",
)
parser.add_argument(
    "--dest",
    default="192.168.0.2",
    help="Holoscan Sensor Bridge / board IP (ECB destination) [default: 192.168.0.2]",
)
parser.add_argument(
    "--eeprom-reg-addr-bits",
    type=int,
    choices=(8, 16),
    default=8,
    help="EEPROM register address width in bits (matches EEPROM_REG_ADDR_BITS / HSB build) [default: 8]",
)
args = parser.parse_args()

# Host/Board Info
HOST = args.host
DEST = args.dest
PORT = 8192  # 0x2000

# User Config
I2C_CORE_POST_2504 = 1  # 0=OLD CORE, 1=NEW CORE
EEPROM_REG_ADDR_BITS = args.eeprom_reg_addr_bits  # 8 or 16

# Specific info for EEPROM
eeprom_dev_addr = "50"
EEPROM_REG_ADDR_BYTES = int(math.ceil(EEPROM_REG_ADDR_BITS / 8))

if I2C_CORE_POST_2504 == 1:
    i2c_ctrl_addr = "030002"
    i2c_data_addr = "030003"
    clk_cnt_calculated = int(((args.apb_clk_freq - 1) / 400e3 / 2) + 1)
else:
    i2c_ctrl_addr = "040000"
    if (args.apb_clk_freq / 50) < 400e3:
        clk_cnt_calculated = 10
    else:
        clk_cnt_calculated = int(((args.apb_clk_freq - 1) / 400e3 / 5) + 1)

clk_cnt = format(clk_cnt_calculated, "X")  # Convert to hex string

# Temp data for testing
eeprom_test_data = [
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
]

#
# EEPROM Device Data stored only in this script once the EEPROM is read
#
eeprom_device_data_loaded = 0
eeprom_device_data = [
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
    "FF",
]

# Turn off ethernet transactions and use database eeprom_test_data
debug = 0
# Display data
display_data = 0
# Display I2C commands
display_cmds = 1

if not debug:
    print(f"UDP target {DEST}:{PORT}")
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, 11, 1)
    s.settimeout(1)
    s.bind((HOST, PORT))


seq = 0


def write_dword(data, addr):
    global seq
    flags = "01"
    cmd = "04"
    packet = cmd + flags + str(seq).zfill(4) + "0000" + addr + data
    # print("WR ADDR: 0x" + addr + " DATA: 0x" + data)
    if debug:
        if display_cmds:
            new_p = packet[::-1]
            p = ""
            for y in range(0, len(new_p), 8):
                if y + 8 < len(new_p):
                    p = p + new_p[y : y + 8] + "_"
                else:
                    p = p + new_p[y::]
            new_p = p[::-1]
            print(f"Write Dword: {new_p}")
    else:
        for x in range(1):
            try:
                s.sendto(bytes.fromhex(packet), (DEST, PORT))
                ack, ip_addr = s.recvfrom(1024)
                # print("ack")
                # print(ack)
                # print("ip_addr")
                # print(ip_addr)
                time.sleep(0.05)
                # if (ack[4:]) != b'\x00':
                #     print("ERROR: invalid address")
                #     sys.exit()
            except (OSError, ValueError) as e:
                print("ERROR: failed to write to EEPROM", e)
                sys.exit(1)


def read_dword(addr):
    global seq
    flags = "00"
    cmd = "14"
    packet = cmd + flags + str(seq).zfill(4) + "0000" + addr
    # print("RD ADDR: 0x" + addr)

    if debug:
        if display_cmds:
            new_p = packet[::-1]
            p = ""
            for y in range(0, len(new_p), 8):
                if y + 8 < len(new_p):
                    p = p + new_p[y : y + 8] + "_"
                else:
                    p = p + new_p[y::]
            new_p = p[::-1]
            print(f"Read Dword: {new_p}")
        # Dummy return which is not used in debug mode
        data_hex = "11121314"
        ls = data_hex[6:8] + data_hex[4:6] + data_hex[2:4] + data_hex[0:2]
        return ls
    else:
        #        for x in range(3):
        #            try:
        s.sendto(bytes.fromhex(packet), (DEST, PORT))
        data, addr = s.recvfrom(1024)
        data_hex = str(data[10:].hex()).upper()
        ls = data_hex[6:8] + data_hex[4:6] + data_hex[2:4] + data_hex[0:2]
        # time.sleep(0.1)
        return ls


#            except:
#                pass


def eeprom_init():
    if I2C_CORE_POST_2504 == 0:
        write_dword(
            "00" + eeprom_dev_addr + "0002", i2c_ctrl_addr + "00"
        )  # Enable core, set dev addr
        write_dword("000000" + clk_cnt, i2c_ctrl_addr + "08")  # CLK_CNT==5, 390KHz
    else:
        write_dword("00000000", i2c_ctrl_addr + "00")  # Status Clear
        write_dword("000000" + clk_cnt, i2c_ctrl_addr + "0C")  # CLK_CNT==5, 390KHz


def write_eeprom_page(addr, data):
    if EEPROM_REG_ADDR_BYTES == 2:
        addr_end = addr[2:4] + addr[:2]
    else:
        addr_end = addr

    if I2C_CORE_POST_2504 == 0:
        if EEPROM_REG_ADDR_BYTES == 2:
            write_dword(
                "0000000A", i2c_ctrl_addr + "04"
            )  # NUM_CMD=0x02, NUM_RD=0x0, NUM_WR=0x2
            write_dword(data[1] + data[0] + addr_end, i2c_ctrl_addr + "10")  # WR DATA
            write_dword(
                data[5] + data[4] + data[3] + data[2], i2c_ctrl_addr + "14"
            )  # WR DATA
            write_dword("0000" + data[7] + data[6], i2c_ctrl_addr + "18")  # WR DATA
        else:
            write_dword(
                "00000009", i2c_ctrl_addr + "04"
            )  # NUM_CMD=0x02, NUM_RD=0x0, NUM_WR=0x2
            write_dword(
                data[2] + data[1] + data[0] + addr_end, i2c_ctrl_addr + "10"
            )  # WR DATA
            write_dword(
                data[6] + data[5] + data[4] + data[3], i2c_ctrl_addr + "14"
            )  # WR DATA
            write_dword("000000" + data[7], i2c_ctrl_addr + "18")  # WR DATA
        write_dword(
            "00" + eeprom_dev_addr + "0003", i2c_ctrl_addr + "00"
        )  # Start I2C Transaction
    else:
        write_dword("00000000", i2c_ctrl_addr + "00")  # Status Clear
        if EEPROM_REG_ADDR_BYTES == 2:
            write_dword(
                "0000000A", i2c_ctrl_addr + "08"
            )  # NUM_CMD=0x02, NUM_RD=0x0, NUM_WR=0x2
            write_dword(data[1] + data[0] + addr_end, i2c_data_addr + "00")  # WR DATA
            write_dword(
                data[5] + data[4] + data[3] + data[2], i2c_data_addr + "04"
            )  # WR DATA
            write_dword("0000" + data[7] + data[6], i2c_data_addr + "08")  # WR DATA
        else:
            write_dword(
                "00000009", i2c_ctrl_addr + "08"
            )  # NUM_CMD=0x02, NUM_RD=0x0, NUM_WR=0x2
            write_dword(
                data[2] + data[1] + data[0] + addr_end, i2c_data_addr + "00"
            )  # WR DATA
            write_dword(
                data[6] + data[5] + data[4] + data[3], i2c_data_addr + "04"
            )  # WR DATA
            write_dword("000000" + data[7], i2c_data_addr + "08")  # WR DATA
        write_dword("004C4B40", i2c_ctrl_addr + "10")  # Status Clear
        write_dword(
            "00" + eeprom_dev_addr + "0001", i2c_ctrl_addr + "00"
        )  # Start I2C Transaction

    #
    # Save the write to EEPROM_DEVICE_DATA
    #
    addr_int = int(addr, 16)
    eeprom_device_data[addr_int] = data[0]
    eeprom_device_data[addr_int + 1] = data[1]
    eeprom_device_data[addr_int + 2] = data[2]
    eeprom_device_data[addr_int + 3] = data[3]
    eeprom_device_data[addr_int + 4] = data[4]
    eeprom_device_data[addr_int + 5] = data[5]
    eeprom_device_data[addr_int + 6] = data[6]
    eeprom_device_data[addr_int + 7] = data[7]
    #
    # If debug then use EEPROM_TEST_DATA as the Device
    #
    if debug:
        eeprom_test_data[addr_int] = data[0]
        eeprom_test_data[addr_int + 1] = data[1]
        eeprom_test_data[addr_int + 2] = data[2]
        eeprom_test_data[addr_int + 3] = data[3]
        eeprom_test_data[addr_int + 4] = data[4]
        eeprom_test_data[addr_int + 5] = data[5]
        eeprom_test_data[addr_int + 6] = data[6]
        eeprom_test_data[addr_int + 7] = data[7]


def read_eeprom_page(addr):
    if EEPROM_REG_ADDR_BYTES == 2:
        addr_end = addr[2:4] + addr[:2]
    else:
        addr_end = addr
    # print(addr_end)

    if I2C_CORE_POST_2504 == 1:
        write_dword("00000000", i2c_ctrl_addr + "00")  # Status Clear
        write_dword("000000" + clk_cnt, i2c_ctrl_addr + "0C")  # CLK_CNT==5, 390KHz
        if EEPROM_REG_ADDR_BYTES == 2:
            write_dword("00080002", i2c_ctrl_addr + "08")  # NUM_RD=0x0, NUM_WR=0x3
            write_dword("0000" + addr_end, i2c_data_addr + "00")  # WR DATA
        else:
            write_dword("00080001", i2c_ctrl_addr + "08")  # NUM_RD=0x0, NUM_WR=0x2
            write_dword("000000" + addr_end, i2c_data_addr + "00")  # WR DATA
        write_dword("004C4B40", i2c_ctrl_addr + "10")  # Timeout
        write_dword(
            "00" + eeprom_dev_addr + "0001", i2c_ctrl_addr + "00"
        )  # Start I2C Transaction
        # Read I2C Done
        # d0 = read_dword(i2c_ctrl_addr + "80")
        # print(d0)
        time.sleep(0.1)
        d1 = read_dword(i2c_data_addr + "00")
        d2 = read_dword(i2c_data_addr + "04")
    else:
        write_dword(
            "00" + eeprom_dev_addr + "0002", i2c_ctrl_addr + "00"
        )  # Enable core, set dev addr
        if EEPROM_REG_ADDR_BYTES == 2:
            write_dword("00000802", i2c_ctrl_addr + "04")  # NUM_RD=0x8, NUM_WR=0x2
            write_dword("0000" + addr_end, i2c_ctrl_addr + "10")  # WR DATA
        else:
            write_dword("00000801", i2c_ctrl_addr + "04")  # NUM_RD=0x0, NUM_WR=0x1
            write_dword("000000" + addr_end, i2c_ctrl_addr + "10")  # WR DATA
        write_dword("000000" + clk_cnt, i2c_ctrl_addr + "08")  # CLK_CNT==5, 390KHz
        write_dword(
            "00" + eeprom_dev_addr + "0003", i2c_ctrl_addr + "00"
        )  # Start I2C Transaction
        d1 = read_dword(i2c_ctrl_addr + "10")
        d2 = read_dword(i2c_ctrl_addr + "14")
    addr_int = int(addr, 16)
    if debug:
        data = [
            eeprom_test_data[addr_int],
            eeprom_test_data[addr_int + 1],
            eeprom_test_data[addr_int + 2],
            eeprom_test_data[addr_int + 3],
            eeprom_test_data[addr_int + 4],
            eeprom_test_data[addr_int + 5],
            eeprom_test_data[addr_int + 6],
            eeprom_test_data[addr_int + 7],
        ]
        eeprom_device_data[addr_int] = eeprom_test_data[addr_int]
        eeprom_device_data[addr_int + 1] = eeprom_test_data[addr_int + 1]
        eeprom_device_data[addr_int + 2] = eeprom_test_data[addr_int + 2]
        eeprom_device_data[addr_int + 3] = eeprom_test_data[addr_int + 3]
        eeprom_device_data[addr_int + 4] = eeprom_test_data[addr_int + 4]
        eeprom_device_data[addr_int + 5] = eeprom_test_data[addr_int + 5]
        eeprom_device_data[addr_int + 6] = eeprom_test_data[addr_int + 6]
        eeprom_device_data[addr_int + 7] = eeprom_test_data[addr_int + 7]
        return data
    else:
        eeprom_device_data[addr_int] = d1[0:2]
        eeprom_device_data[addr_int + 1] = d1[2:4]
        eeprom_device_data[addr_int + 2] = d1[4:6]
        eeprom_device_data[addr_int + 3] = d1[6:8]
        eeprom_device_data[addr_int + 4] = d2[0:2]
        eeprom_device_data[addr_int + 5] = d2[2:4]
        eeprom_device_data[addr_int + 6] = d2[4:6]
        eeprom_device_data[addr_int + 7] = d2[6:8]
        data = [d1[0:2], d1[2:4], d1[4:6], d1[6:8], d2[0:2], d2[2:4], d2[4:6], d2[6:8]]
        return data


def eeprom_write_all():
    eeprom_init()
    for x in range(32):
        addr = str(hex(x * 8)).replace("0x", "").upper().zfill(2)
        write_eeprom_page(addr, ["A1", "B2", "C3", "D4", "E5", "F6", "97", "88"])
        time.sleep(0.5)
        print(read_eeprom_page(addr))
        time.sleep(0.5)


def write_eeprom_byte(addr, data):
    if EEPROM_REG_ADDR_BYTES == 2:
        addr = addr[2:4] + addr[:2]
        # print("ADDR")
        # print(addr)
    if I2C_CORE_POST_2504 == 1:
        write_dword("00000000", i2c_ctrl_addr + "00")  # Status Clear
        write_dword("000000" + clk_cnt, i2c_ctrl_addr + "0C")  # CLK_CNT==5, 390KHz
        if EEPROM_REG_ADDR_BYTES == 2:
            write_dword("00000003", i2c_ctrl_addr + "08")  # NUM_RD=0x0, NUM_WR=0x3
            write_dword("00" + data + addr, i2c_data_addr + "00")  # WR DATA
        else:
            write_dword("00000002", i2c_ctrl_addr + "08")  # NUM_RD=0x0, NUM_WR=0x2
            write_dword("0000" + data + addr, i2c_data_addr + "00")  # WR DATA
        write_dword("004C4B40", i2c_ctrl_addr + "10")  # Timeout
        write_dword(
            "00" + eeprom_dev_addr + "0001", i2c_ctrl_addr + "00"
        )  # Start I2C Transaction
    else:
        write_dword(
            "00" + eeprom_dev_addr + "0002", i2c_ctrl_addr + "00"
        )  # Enable core, set dev addr
        write_dword("000000" + clk_cnt, i2c_ctrl_addr + "08")  # CLK_CNT==5, 390KHz
        if EEPROM_REG_ADDR_BYTES == 2:
            write_dword("00000003", i2c_ctrl_addr + "04")  # NUM_RD=0x0, NUM_WR=0x3
            write_dword("00" + data + addr, i2c_ctrl_addr + "10")  # WR DATA
        else:
            write_dword("00000002", i2c_ctrl_addr + "04")  # NUM_RD=0x0, NUM_WR=0x2
            write_dword("0000" + data + addr, i2c_ctrl_addr + "10")  # WR DATA
        write_dword("00" + data + addr, i2c_ctrl_addr + "10")  # WR DATA
        write_dword(
            "00" + eeprom_dev_addr + "0003", i2c_ctrl_addr + "00"
        )  # Start I2C Transaction
    addr_int = int(addr, 16)
    # eeprom_device_data[addr_int] = data
    if debug:
        eeprom_test_data[addr_int] = data
    if display_data:
        print(f"addr: {addr_int} = {data}")


#
# Does a random byte read by setting the address (dev_addr, byte_addr(NUM_WR=1)) then
# (dev_addr, read_byte(NUM_RD=1)).
#
def read_eeprom_byte(addr):
    if EEPROM_REG_ADDR_BYTES == 2:
        addr_end = addr[2:4] + addr[:2]
    else:
        addr_end = addr

    # print("ADDR 0x"+addr)
    if I2C_CORE_POST_2504 == 1:
        write_dword("00000000", i2c_ctrl_addr + "00")  # Status Clear
        write_dword("000000" + clk_cnt, i2c_ctrl_addr + "0C")  # CLK_CNT==5, 390KHz
        if EEPROM_REG_ADDR_BYTES == 2:
            write_dword("00010002", i2c_ctrl_addr + "08")  # NUM_RD=0x0, NUM_WR=0x3
            write_dword("0000" + addr_end, i2c_data_addr + "00")  # WR DATA
        else:
            write_dword("00010001", i2c_ctrl_addr + "08")  # NUM_RD=0x0, NUM_WR=0x2
            write_dword("000000" + addr_end, i2c_data_addr + "00")  # WR DATA
        write_dword("004C4B40", i2c_ctrl_addr + "10")  # Timeout
        write_dword(
            "00" + eeprom_dev_addr + "0001", i2c_ctrl_addr + "00"
        )  # Start I2C Transaction
        d1 = read_dword(i2c_data_addr + "00")  # ,dword_read(i2c_ctrl_addr + "14")
    else:
        write_dword(
            "00" + eeprom_dev_addr + "0002", i2c_ctrl_addr + "00"
        )  # Enable core, set dev addr
        write_dword("000000" + clk_cnt, i2c_ctrl_addr + "08")  # CLK_CNT==5, 390KHz
        if EEPROM_REG_ADDR_BYTES == 2:
            write_dword("00000102", i2c_ctrl_addr + "04")  # NUM_RD=0x1, NUM_WR=0x1
            write_dword("0000" + addr_end, i2c_ctrl_addr + "10")  # WR DATA
        else:
            write_dword("00000101", i2c_ctrl_addr + "04")  # NUM_RD=0x1, NUM_WR=0x1
            write_dword("000000" + addr_end, i2c_ctrl_addr + "10")  # WR DATA
        write_dword(
            "00" + eeprom_dev_addr + "0003", i2c_ctrl_addr + "00"
        )  # Start I2C Transaction
        d1 = read_dword(i2c_ctrl_addr + "10")  # ,dword_read(i2c_ctrl_addr + "14")
    if debug:
        addr_int = int(addr, 16)
        data = eeprom_test_data[addr_int]
        if display_data:
            print(f"addr: {addr_int} = {data}")
        return data
    else:
        data = d1[0:2]
        return data


def AddToCRC(b, crc):
    b2 = b
    if b < 0:
        b2 = b + 256
    for i in range(8):
        odd = ((b2 ^ crc) & 1) == 1
        crc >>= 1
        b2 >>= 1
        if odd:
            crc ^= 0x8C  # This means crc ^= 140.
    return crc


def compute_crc(eeprom_data):
    crc = 0
    for i in range(255):
        crc = AddToCRC(int(eeprom_data[i], 16), crc)
    return crc


#
# Main
#

#
# Set up I2C controller for EEPROM access
#
# eeprom_init()
# write_eeprom_byte("00", "11")
# write_eeprom_byte("01", "22")
# write_eeprom_byte("02", "33")
# write_eeprom_byte("03", "44")
# write_eeprom_byte("04", "55")
# write_eeprom_byte("05", "66")
# write_eeprom_byte("06", "77")
# write_eeprom_byte("07", "88")
# print(read_eeprom_page("00"))

# quit()

while True:
    print("")
    print("EEPROM Commands:")
    print("  q                   - Quit")
    print("  d                   - Dump the EEPROM to the screen")
    print("  e                   - Erase the EEPROM setting the CRC correctly")
    print("  w <file>            - Read <file> and write to EEPROM")
    print(
        "  wrc <file>          - Read <file> and write to EEPROM, read EEPROM, and Compare"
    )
    print("  r <file>            - Read EEPROM and write to <file>")
    print("  wb <addr> <byte>    - Write EEPROM Address <0x??> with Byte <0x??>")
    print(
        "  wbc <addr> <byte>   - Write EEPROM Address <0x??> with Byte <0x??> fixing the CRC"
    )
    print("  rb <addr>           - Read EEPROM Address <0x??>")
    print("")
    reply = input("Enter Command: ")
    fields = reply.split()
    fields[0] = fields[0].lower()
    if fields[0] == "q":
        break
    #
    # Dump EEPROM data to the screen
    #
    elif fields[0] == "d":
        crc = 0
        for x in range(32):
            addr = (
                str(hex(x * 8))
                .replace("0x", "")
                .upper()
                .zfill(EEPROM_REG_ADDR_BYTES * 2)
            )
            # print(addr)
            data = read_eeprom_page(addr)
            # time.sleep(0.5)
            print(data)
            addr_int = x * 8
            for byte in data:
                addr_str = str(hex(addr_int)).replace("0x", "").upper().zfill(2)
                if addr_str == "FF":
                    crc_str = str(hex(crc)).replace("0x", "").upper().zfill(2)
                    if byte != crc_str:
                        print(f"\nCRC Mismatch: EEPROM({byte}) Computed({crc_str})\n")
                        reply = input("Fix CRC byte (y or n): ")
                        reply = reply.lower()
                        if reply == "y":
                            write_eeprom_byte(addr_str, crc_str)
                            eeprom_device_data[addr_int] = crc_str
                            time.sleep(0.1)
                else:
                    crc = AddToCRC(int(byte, 16), crc)
                addr_int = addr_int + 1
        eeprom_device_data_loaded = 1
    #
    # Erase EEPROM
    #
    elif fields[0] == "e":
        crc = 0
        data_array = []
        count = 0
        reply = input("Really Erase the EEPROM (y or n): ")
        reply = reply.lower()
        if reply == "y":
            for addr in range(256):
                addr_str = (
                    str(hex(addr))
                    .replace("0x", "")
                    .upper()
                    .zfill(EEPROM_REG_ADDR_BYTES * 2)
                )
                data = "FF"
                data_array.append(data)
                if addr != 255:
                    crc = AddToCRC(int(data, 16), crc)
                # Write every 8 bytes to EEPROM
                if count == 0:
                    addr_save = addr_str
                if count == 7:
                    if addr_save == "F8":
                        data_array[7] = str(hex(crc)).replace("0x", "").upper().zfill(2)
                    if display_data:
                        print(f"{addr_save} {data_array}")
                    write_eeprom_page(addr_save, data_array)
                    count = 0
                    data_array = []
                else:
                    count = count + 1
            eeprom_device_data_loaded = 1
    #
    # Read <addr>
    #
    elif fields[0] == "rb":
        if len(fields) < 2:
            print("Error(rb): Need an address\n")
        else:
            fields[1] = (
                fields[1].replace("0x", "").upper().zfill(EEPROM_REG_ADDR_BYTES * 2)
            )
            byte = read_eeprom_byte(fields[1])
            print(f"\nAddress 0x{fields[1]} ({int(fields[1], 16)}) = 0x{byte}")
    #
    # Write <addr> <byte>
    #
    elif fields[0] == "wb":
        if len(fields) < 3:
            print("Error(wb): Need both an address and data\n")
        else:
            fields[1] = (
                fields[1].replace("0x", "").upper().zfill(EEPROM_REG_ADDR_BYTES * 2)
            )
            fields[2] = fields[2].replace("0x", "").upper().zfill(2)
            print("\nWriting byte without CRC fixing!!\n")
            reply = input("Continue? (y or n): ")
            reply = reply.lower()
            if reply == "y":
                write_eeprom_byte(fields[1], fields[2])
                addr_int = int(fields[1], 16)
                eeprom_device_data[addr_int] = fields[2]
                time.sleep(0.5)
    #
    # Write <addr> <byte> with updated CRC
    #
    elif fields[0] == "wbc":
        if len(fields) < 3:
            print("Error(wbc): Need both an address and data\n")
        else:
            if eeprom_device_data_loaded:
                fields[1] = (
                    fields[1].replace("0x", "").upper().zfill(EEPROM_REG_ADDR_BYTES * 2)
                )
                fields[2] = fields[2].replace("0x", "").upper().zfill(2)
                # update eeprom_device_data
                addr_int = int(fields[1], 16)
                eeprom_device_data[addr_int] = fields[2]
                # compute CRC
                crc = 0
                for x in range(255):
                    crc = AddToCRC(int(eeprom_device_data[x], 16), crc)
                # write byte
                write_eeprom_byte(fields[1], fields[2])
                time.sleep(0.5)
                # write CRC
                crc_str = str(hex(crc)).replace("0x", "").upper().zfill(2)
                write_eeprom_byte("FF", crc_str)
                time.sleep(0.5)
            else:
                print(
                    "\nPlease dump the contents of the EEPROM before running this command!"
                )
                print(
                    "Or write the contents of the EEPROM to a file before running this command!"
                )
                print("Or load a file into the EEPROM before running this command!\n")
    #
    # Read EEPROM data and write to a file checking CRC
    #
    elif fields[0] == "r":
        if len(fields) < 2:
            print("Error(r): Need a file name\n")
        else:
            crc = 0
            f = open(fields[1], "w")
            for x in range(32):
                addr = (
                    str(hex(x * 8))
                    .replace("0x", "")
                    .upper()
                    .zfill(EEPROM_REG_ADDR_BYTES * 2)
                )
                data = read_eeprom_page(addr)
                time.sleep(0.5)
                addr_int = x * 8
                for byte in data:
                    addr_str = str(hex(addr_int)).replace("0x", "").upper().zfill(2)
                    if addr_str == "FF":
                        crc_str = str(hex(crc)).replace("0x", "").upper().zfill(2)
                        if byte != crc_str:
                            print(f"\nCRC Mismatch: EEPROM({byte}) Computed({crc_str})")
                            print(
                                "CRC Mismatch: Writing correct CRC byte to the file..."
                            )
                            byte = crc_str
                            # eeprom_test_data[255] = crc_str
                    else:
                        crc = AddToCRC(int(byte, 16), crc)
                    if display_data:
                        print(f"@{addr_str}  {byte}")
                    f.write(f"@{addr_str}  {byte}\n")
                    addr_int = addr_int + 1
            f.close
            eeprom_device_data_loaded = 1
    #
    # Read <file> and write to EEPROM fixing the CRC byte
    #
    elif fields[0] == "w":
        if len(fields) < 2:
            print("Error(w): Need a file name\n")
        else:
            crc = 0
            data_array = []
            count = 0
            with open(fields[1], "r") as file:
                while True:
                    line = file.readline()
                    if not line:
                        break
                    fields = line.split()
                    addr = fields[0].replace("@", "")
                    data = fields[1]
                    data_array.append(data)
                    if addr != "FF":
                        crc = AddToCRC(int(data, 16), crc)
                    # Write every 8 bytes to EEPROM
                    if count == 0:
                        addr_save = addr
                    if count == 7:
                        if addr_save == "F8":
                            data_array[7] = (
                                str(hex(crc)).replace("0x", "").upper().zfill(2)
                            )
                        if display_data:
                            print(f"{addr_save} {data_array}")
                        write_eeprom_page(addr_save, data_array)
                        # time.sleep(0.5)
                        count = 0
                        data_array = []
                    else:
                        count = count + 1
            file.close
            eeprom_device_data_loaded = 1
    #
    # Read <file> and write to EEPROM, read EEPROM, and Compare
    #
    elif fields[0] == "wrc":
        if len(fields) < 2:
            print("Error(wrc): Need a file name\n")
        else:
            crc = 0
            data_array = []
            count = 0
            print(f"\nReading from {fields[1]} and writing EEPROM", end="", flush=True)
            with open(fields[1], "r") as file:
                while True:
                    line = file.readline()
                    if not line:
                        break
                    addr = line[1:3]
                    data = line[5:7]
                    data_array.append(data)
                    if addr != "FF":
                        crc = AddToCRC(int(data, 16), crc)
                    # Write every 8 bytes to EEPROM
                    if count == 0:
                        addr_save = addr
                    if count == 7:
                        if addr_save == "F8":
                            data_array[7] = (
                                str(hex(crc)).replace("0x", "").upper().zfill(2)
                            )
                        if display_data:
                            print(f"{addr_save} {data_array}")
                        write_eeprom_page(addr_save, data_array)
                        print(".", end="", flush=True)
                        # time.sleep(0.5)
                        count = 0
                        data_array = []
                    else:
                        count = count + 1
            file.close
            eeprom_device_data_loaded = 1
            print("")
            print("Reading EEPROM and comparing", end="", flush=True)
            crc = 0
            for x in range(32):
                addr = (
                    str(hex(x * 8))
                    .replace("0x", "")
                    .upper()
                    .zfill(EEPROM_REG_ADDR_BYTES * 2)
                )
                data = read_eeprom_page(addr)
                print(".", end="", flush=True)
                # time.sleep(0.5)
                # print(data)
                addr_int = x * 8
                for byte in data:
                    addr_str = str(hex(addr_int)).replace("0x", "").upper().zfill(2)
                    if eeprom_device_data[addr_int] != byte:
                        print(
                            f"\nData Mismatch: Address({addr_str}) EEPROM({byte}) Expected({eeprom_device_data[addr_int]})"
                        )
                    if addr_str == "FF":
                        crc_str = str(hex(crc)).replace("0x", "").upper().zfill(2)
                        if byte != crc_str:
                            print(
                                f"\nCRC Mismatch: EEPROM({byte}) Computed({crc_str})\n"
                            )
                    else:
                        crc = AddToCRC(int(byte, 16), crc)
                    addr_int = addr_int + 1
            print("")
