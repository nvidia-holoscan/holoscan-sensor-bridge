/**
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
 *
 * See README.md for detailed information.
 */

#ifndef STM32_HSB_EMULATOR_HPP
#define STM32_HSB_EMULATOR_HPP

#include "../../hsb_emulator.hpp"
#include <cstdint>

#include "STM32/address_map.hpp"
#include "STM32/apb_events.hpp"
#include "STM32/i2c.hpp"
#include "STM32/net.hpp"
#include "STM32/spi.hpp"

#define CONTROL_PACKET_SIZE TX_BUFFER_SIZE

#ifndef COUNTOF
#define COUNTOF(array) (sizeof(array) / sizeof(array[0]))
#endif

#ifndef CP_WRITE_HANDLERS_MAX_NUM
#define CP_WRITE_HANDLERS_MAX_NUM 20
#endif
#ifndef CP_READ_HANDLERS_MAX_NUM
#define CP_READ_HANDLERS_MAX_NUM 20
#endif

#define CHECK_CP_MAP_SET(expr) \
    do {                       \
        int status = (expr);   \
        if (status) {          \
            Error_Handler();   \
        }                      \
    } while (0)

namespace hololink::emulation {

struct ControlPlaneCallback {
    ControlPlaneCallback_f callback;
    void* ctxt;
};

struct AsyncEventCtxt {
    // these are the registers themselves, held in an array for simpler access
    uint32_t data[(CTRL_EVT_SW_EVENT - CTRL_EVENT) / REGISTER_SIZE];
    uint32_t status;
};

class RegisterMemory : public AddressMemory {
public:
    RegisterMemory() = default;
    ~RegisterMemory() override = default;

    // see address_memory.hpp for documentation
    int write(AddressValuePair& address_value) override;
    int read(AddressValuePair& address_value) override;
    int write_many(AddressValuePair* address_values, int num_addresses) override;
    int read_many(AddressValuePair* address_values, int num_addresses) override;
    int write_range(uint32_t start_address, uint32_t* values, int num_addresses, int stride = 2) override;
    int read_range(uint32_t start_address, uint32_t* values, int num_addresses, int stride = 2) override;

private:
    friend class HSBEmulator;
    void set_ctxt(HSBEmulatorCtxt* ctxt) { ctxt_ = ctxt; }
    struct HSBEmulatorCtxt* ctxt_ { nullptr };
};

struct HSBEmulatorCtxt {
    RegisterMemory register_memory;
    DataPlane* data_plane_list[MAX_DATA_PLANES];
    HSBEmulator* hsb_emulator;
    ETH_HandleTypeDef eth_handle;
    SpiControllerCtxt* spi_ctxt;
    struct PTPConfig ptp_config;
    struct AsyncEventCtxt async_event_ctxt;

    AddressMap<ControlPlaneCallback, CP_WRITE_HANDLERS_MAX_NUM> cp_write_map;
    AddressMap<ControlPlaneCallback, CP_READ_HANDLERS_MAX_NUM> cp_read_map;
    uint32_t apb_ram_data[APB_RAM_DATA_SIZE / REGISTER_SIZE];

    // TODO: generalize to an event loop mechanism
    uint32_t up_time_msec;
    uint32_t next_bootp_time_msec;
    unsigned short data_plane_count;
    bool running;
};

}

#endif