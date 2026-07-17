/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "max96716a.hpp"

#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>

#include <fmt/format.h>

#include <hololink/core/logging_internal.hpp>

namespace hololink::module::sensors::max96716a {

Max96716a::Max96716a(std::shared_ptr<HololinkInterfaceV1> hololink)
{
    i2c_ = hololink->get_i2c<>(i2c_bus_, i2c_address_);
}

uint8_t Max96716a::get_register(uint16_t reg)
{
    std::vector<uint8_t> write_bytes = { uint8_t(reg >> 8), uint8_t(reg & 0xFF) };
    std::vector<uint8_t> read_bytes(1);
    const auto s = i2c_->i2c_transaction(i2c_address_, write_bytes, read_bytes);
    if (s != HOLOLINK_MODULE_OK) {
        throw std::runtime_error(fmt::format(
            "While reading MAX96716A register: i2c_addr=0x{:02X}, register=0x{:04X}, status={}",
            i2c_address_, reg, s));
    }
    return read_bytes[0];
}

void Max96716a::set_register(uint16_t reg, uint8_t value)
{
    std::vector<uint8_t> write_bytes = { uint8_t(reg >> 8), uint8_t(reg & 0xFF), value };
    std::vector<uint8_t> read_bytes;
    const auto s = i2c_->i2c_transaction(i2c_address_, write_bytes, read_bytes);
    if (s != HOLOLINK_MODULE_OK) {
        HSB_LOG_ERROR("Failed: i2c_addr=0x{:02X}, register=0x{:04X}, value=0x{:02X}, status={}",
            i2c_address_, reg, value, s);
        throw std::runtime_error(fmt::format(
            "While writing MAX96716A register: i2c_addr=0x{:02X}, register=0x{:04X}, status={}",
            i2c_address_, reg, s));
    }
}

void Max96716a::enable_link_exclusive(GmslLink link)
{
    set_register(0x0F00, (link == GmslLink::LINK_A) ? 0x01 : 0x02);
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
}

void Max96716a::enable_both_links()
{
    set_register(0x0F00, 0x03);
    set_register(0x0010, 0x03); // Reverse splitter mode. Both links enabled.
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
}

uint8_t Max96716a::stream_id_to_pipe_mapping(GmslLink link, int stream_id, VideoPipe pipe)
{
    if (stream_id < 0 || stream_id >= 4) {
        throw std::runtime_error(fmt::format("stream_id must be in 0..3, got {}", stream_id));
    }
    int sid = (link == GmslLink::LINK_B) ? stream_id + 4 : stream_id;
    int pipe_value = (pipe == VideoPipe::PIPE_Z) ? (sid << 3) : sid;
    return static_cast<uint8_t>(pipe_value);
}

void Max96716a::configure_video_pipe()
{
    set_register(0x0330, 0x04); // MIPI 2x4: Port A, PHY0/1; Port B, PHY2/3.
    set_register(0x0474, 0x08); // MIPI TX1: 2 data lanes
    set_register(0x04B4, 0x08); // MIPI TX2: 2 data lanes
}

void Max96716a::route_pin_to_gmsl_gpio(GmslLink link, int pin, uint8_t tx_id)
{
    uint16_t page_offset = (link == GmslLink::LINK_A) ? uint16_t(0) : GPIO_ALT_PAGE_OFFSET;
    uint16_t base = page_offset + GPIO_PIN_BASE + 3 * static_cast<uint16_t>(pin);
    set_register(base, 0x02); // GPIO_A: GPIO_TX_EN=1
    set_register(base + 1, static_cast<uint8_t>(0x60 | (tx_id & 0x1F))); // GPIO_B: push-pull, pull-up, GPIO_TX_ID
    set_register(base + 2, static_cast<uint8_t>(tx_id & 0x1F)); // GPIO_C: GPIO_RX_ID
}

} // namespace hololink::module::sensors::max96716a
