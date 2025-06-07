/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "li_i2c_expander.hpp"

#include <hololink/core/serializer.hpp>

namespace hololink::sensors {
LII2CExpander::LII2CExpander(Hololink& hololink, uint32_t i2c_bus)
    : i2c_(hololink.get_i2c(i2c_bus))
{
}

void LII2CExpander::configure(I2CExpanderOutputEN output_en)
{
    std::vector<uint8_t> write_bytes(1);
    core::Serializer serializer(write_bytes.data(), write_bytes.size());
    serializer.append_uint8(static_cast<uint8_t>(output_en));

    uint32_t read_byte_count = 0;
    i2c_->i2c_transaction(I2C_EXPANDER_ADDRESS, write_bytes, read_byte_count);
}

} // namespace hololink::sensors
