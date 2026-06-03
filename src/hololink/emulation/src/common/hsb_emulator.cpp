/**
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
 *
 * See README.md for detailed information.
 */

#include <chrono>
#include <climits>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include "core/deserializer.hpp"
#include "core/serializer.hpp"

#include "address_memory.hpp"
#include "data_plane.hpp"
#include "hsb_emulator.hpp"
#include "i2c_interface.hpp"
#include "net.hpp"
#include "utils.hpp"

namespace hololink::emulation {

I2CController::~I2CController()
{
    stop();
}

HSBEmulator::HSBEmulator()
    : HSBEmulator(HSB_EMULATOR_CONFIG)
{
}

HSBEmulator::~HSBEmulator()
{
}

int HSBEmulator::read(uint32_t address, uint32_t& value)
{
    struct AddressValuePair address_value = { address, 0 };
    registers_->read(address_value);
    value = address_value.value;
    return 0;
}

}
