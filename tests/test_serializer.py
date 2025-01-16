# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# See README.md for detailed information.

import hololink


def test_serialize_uint32_le():
    b = bytearray(50)
    serializer = hololink.Serializer(b)
    assert serializer.append_uint32_le(0x12345678)
    assert b[0] == 0x78
    assert b[1] == 0x56
    assert b[2] == 0x34
    assert b[3] == 0x12
    assert serializer.length() == 4


def test_serialize_uint16_le():
    b = bytearray(50)
    serializer = hololink.Serializer(b)
    assert serializer.append_uint16_le(0xA99C)
    assert b[0] == 0x9C
    assert b[1] == 0xA9
    assert serializer.length() == 2


def test_serialize_uint8():
    b = bytearray(50)
    serializer = hololink.Serializer(b)
    assert serializer.append_uint8(0x99)
    assert b[0] == 0x99
    assert serializer.length() == 1


def test_deserialize_uint32_le():
    b = bytearray(50)
    b[0] = 0x5A
    b[1] = 0x6B
    b[2] = 0x7C
    b[3] = 0x8A
    deserializer = hololink.Deserializer(b[:4])
    assert deserializer.next_uint32_le() == 0x8A7C6B5A


def test_deserialize_uint16_le():
    b = bytearray(50)
    b[0] = 0x30
    b[1] = 0x89
    deserializer = hololink.Deserializer(b, length=2)
    assert deserializer.next_uint16_le() == 0x8930


def test_deserialize_length():
    b = bytearray(50)
    b[0] = 0xAA
    b[1] = 0xBB
    b[2] = 0xCC
    b[3] = 0xDD
    deserializer = hololink.Deserializer(b, 4)
    assert deserializer.next_uint32_le() == 0xDDCCBBAA


def test_serialize_buffer():
    b = bytearray(50)
    serializer = hololink.Serializer(b)
    assert serializer.append_buffer(b"Hello")
    assert b[0] == ord("H")
    assert b[1] == ord("e")
    assert b[2] == ord("l")
    assert b[3] == ord("l")
    assert b[4] == ord("o")
    assert serializer.length() == 5


def test_deserialize_buffer():
    b = bytearray(50)
    b[0] = ord("G")
    b[1] = ord("o")
    b[2] = ord("o")
    b[3] = ord("d")
    b[4] = ord("b")
    b[5] = ord("y")
    b[6] = ord("e")
    deserializer = hololink.Deserializer(b)
    length = 7
    r = deserializer.next_buffer(length)
    assert len(r) == length
    assert r == b"Goodbye"


def test_serializer_integration():
    b = bytearray(50)
    serializer = hololink.Serializer(b)
    s = b"Test string"
    assert serializer.append_uint32_le(len(s))
    assert serializer.append_buffer(s)
    # ensure that the next u32 is misaligned
    assert (serializer.length() & 1) != 0
    signature = 0x1234_5678
    assert serializer.append_uint32_le(signature)
    assert serializer.length() == (8 + len(s))
    deserializer = hololink.Deserializer(b, serializer.length())
    data_length = deserializer.next_uint32_le()
    assert data_length == len(s)
    r = deserializer.next_buffer(data_length)
    assert len(r) == data_length
    assert r == s
    assert deserializer.next_uint32_le() == signature


def test_serialize_uint16_be():
    b = bytearray(50)
    serializer = hololink.Serializer(b)
    assert serializer.append_uint16_be(0xA99C)
    assert b[0] == 0xA9
    assert b[1] == 0x9C
    assert serializer.length() == 2


def test_deserialize_uint32_be():
    b = bytearray(50)
    b[0] = 0x8A
    b[1] = 0x7C
    b[2] = 0x6B
    b[3] = 0x5A
    deserializer = hololink.Deserializer(b[:4])
    assert deserializer.next_uint32_be() == 0x8A7C6B5A


def test_serialize_uint32_be():
    b = bytearray(50)
    serializer = hololink.Serializer(b)
    assert serializer.append_uint32_be(0x12345678)
    assert b[0] == 0x12
    assert b[1] == 0x34
    assert b[2] == 0x56
    assert b[3] == 0x78
    assert serializer.length() == 4


def test_deserialize_uint16_be():
    b = bytearray(50)
    b[0] = 0x89
    b[1] = 0x30
    deserializer = hololink.Deserializer(b, length=2)
    assert deserializer.next_uint16_be() == 0x8930


def test_deserialize_uint64_be():
    b = bytes(
        [
            0x8A,
            0x7C,
            0x6B,
            0x5A,
            0x8C,
            0x7B,
            0x6A,
            0x53,
        ]
    )
    deserializer = hololink.Deserializer(b)
    assert deserializer.next_uint64_be() == 0x8A7C6B5A8C7B6A53


def test_deserialize_uint64_le():
    b = bytes(
        [
            0x8A,
            0x7C,
            0x6B,
            0x5A,
            0x8C,
            0x7B,
            0x6A,
            0x53,
        ]
    )
    deserializer = hololink.Deserializer(b)
    assert deserializer.next_uint64_le() == 0x536A7B8C5A6B7C8A
