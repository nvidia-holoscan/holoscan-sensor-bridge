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

#include "STM32/apb_events.hpp"
#include "STM32/hsb_emulator.hpp"
#include "STM32/stm32_system.h"
#include "hsb_config.hpp"
#include <limits.h>
#include <stddef.h>
#include <stdint.h>

#define SEQUENCE_COMMAND_SIZE_BITS 2
#define SEQUENCE_COMMAND_COUNT (1 << SEQUENCE_COMMAND_SIZE_BITS)
#define SEQUENCE_COMMAND_MASK (SEQUENCE_COMMAND_COUNT - 1)
#define DONE_COMMAND_ADDRESS 0xFFFFFFFF
#define APB_EVENT_BUFFER_SIZE 0x100

enum SequenceCommand {
    SEQ_POLL = 0,
    SEQ_WR = 1,
    SEQ_RD = 2,
    SEQ_ERROR = 3,
};

namespace hololink::emulation {

int apb_ram_read(void* ctxt, struct AddressValuePair* addr_val, int max_count)
{
    uint32_t* apb_ram_data = (uint32_t*)ctxt;
    static const uint32_t max_address = APB_RAM + APB_RAM_DATA_SIZE;
    struct AddressValuePair* cur = addr_val;
    int n = 0;
    while (n < max_count) {
        uint32_t address = AVP_GET_ADDRESS(cur);
        if (address >= max_address || address < APB_RAM) {
            return n;
        }
        AVP_SET_VALUE(cur, apb_ram_data[((address - APB_RAM) / REGISTER_SIZE)]);
        cur++;
        n++;
    }
    return n;
}

int apb_ram_write(void* ctxt, struct AddressValuePair* addr_val, int max_count)
{
    uint32_t* apb_ram_data = (uint32_t*)ctxt;
    static uint32_t max_address = APB_RAM + APB_RAM_DATA_SIZE;
    struct AddressValuePair* cur = addr_val;
    int n = 0;
    while (n < max_count) {
        uint32_t address = AVP_GET_ADDRESS(cur);
        if (address >= max_address || address < APB_RAM) {
            return n;
        }
        apb_ram_data[((address - APB_RAM) / REGISTER_SIZE)] = AVP_GET_VALUE(cur);
        cur++;
        n++;
    }
    return n;
}

int async_event_readback_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count)
{
    AsyncEventCtxt* async_event_ctxt = (AsyncEventCtxt*)ctxt;
    struct AddressValuePair* cur = addr_val;
    static const int max_index = COUNTOF(async_event_ctxt->data);
    int n = 0;
    while (n < max_count) {
        uint32_t address = AVP_GET_ADDRESS(cur);
        if (CTRL_EVT_STAT == address) {
            AVP_SET_VALUE(cur, async_event_ctxt->status);
        } else {
            int index = ((address - CTRL_EVENT) / sizeof(uint32_t));
            if (index < max_index) {
                AVP_SET_VALUE(cur, async_event_ctxt->data[index]);
            } else {
                return n;
            }
        }
        cur++;
        n++;
    }
    return n;
}

int async_event_configure_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count)
{
    AsyncEventCtxt* async_event_ctxt = (AsyncEventCtxt*)ctxt;
    struct AddressValuePair* cur = addr_val;
    static const int max_index = COUNTOF(async_event_ctxt->data);
    int n = 0;
    while (n < max_count) {
        uint32_t address = AVP_GET_ADDRESS(cur);
        int index = ((address - CTRL_EVENT) / sizeof(uint32_t));
        if (index < max_index) {
            async_event_ctxt->data[index] = AVP_GET_VALUE(cur);
        } else {
            return n;
        }
        cur++;
        n++;
    }
    return n;
}

// returns the value at the address and advances the address by the size of the address
uint32_t next_value(uint32_t* values, uint32_t* address)
{
    uint32_t value = values[(*address - APB_RAM) / REGISTER_SIZE];
    (*address) += REGISTER_SIZE; // advance next address
    return value;
}

// seq_index, cmd_index, and cmd_bit are all the next values to consume from the sequence_values vector
// returns the next command code and if necessary, updates the indices to the next value to consume
uint8_t next_cmd(uint32_t* values, uint32_t* cmd_word, uint32_t* address, uint8_t* cmd_bit)
{
    if (*cmd_bit >= REGISTER_SIZE * CHAR_BIT) {
        *cmd_bit = 0;
        *cmd_word = next_value(values, address);
    }

    uint8_t cmd = (*cmd_word >> *cmd_bit) & SEQUENCE_COMMAND_MASK;
    cmd = (cmd > (uint8_t)SEQ_ERROR) ? (uint8_t)SEQ_ERROR : cmd;
    *cmd_bit += SEQUENCE_COMMAND_SIZE_BITS;
    return cmd;
}

// WRITE CMD callback. seq_address is where the written value is read from. seq_address is advanced by the size of the address
void sequence_write(struct HSBEmulatorCtxt* hsb_emulator_ctxt, uint32_t* seq_address, uint32_t register_address)
{
    uint32_t value = next_value(hsb_emulator_ctxt->apb_ram_data, seq_address);
    struct AddressValuePair pair = { register_address, value };
    hsb_emulator_ctxt->register_memory.write(pair);
}
// READ CMD callback. seq_address is where the read value will be stored. seq_address is advanced by the size of the address
void sequence_read(struct HSBEmulatorCtxt* hsb_emulator_ctxt, uint32_t* seq_address, uint32_t register_address)
{
    struct AddressValuePair pair = { register_address, 0 };
    hsb_emulator_ctxt->register_memory.read(pair);
    pair.address = *seq_address;
    hsb_emulator_ctxt->register_memory.write(pair);
    (void)next_value(hsb_emulator_ctxt->apb_ram_data, seq_address); // mostly to advance the seq_address. discard result for now
}

// POLL CMD callback. seq_address is where the match value is read from. seq_address is advanced by the size of the address
void sequence_poll(struct HSBEmulatorCtxt* hsb_emulator_ctxt, uint32_t* seq_address, uint32_t register_address)
{
    uint32_t match = next_value(hsb_emulator_ctxt->apb_ram_data, seq_address);
    uint32_t mask = next_value(hsb_emulator_ctxt->apb_ram_data, seq_address);
    struct AddressValuePair pair = { register_address, 0 };

    // no support for non-blocking callbacks yet so this must match immediately
    hsb_emulator_ctxt->register_memory.read(pair);
    if ((pair.value & mask) == match) {
        return;
    }
    // TODO: write to a static register that there is a failure
}

// ERROR CMD callback. seq_address is where the error value is read from. seq_address is advanced by the size of the address
void sequence_error(struct HSBEmulatorCtxt* hsb_emulator_ctxt, uint32_t* seq_address, uint32_t register_address)
{
    (void)hsb_emulator_ctxt;
    (void)seq_address;
    (void)register_address;
    // TODO: write to a static register that there is a failure
    Error_Handler();
}
static void (*APB_EVENT_CMD_CALLBACKS[SEQUENCE_COMMAND_COUNT])(struct HSBEmulatorCtxt* hsb_emulator_ctxt, uint32_t* seq_address, uint32_t register_address) = {
    [SEQ_POLL] = &sequence_poll,
    [SEQ_WR] = &sequence_write,
    [SEQ_RD] = &sequence_read,
    [SEQ_ERROR] = &sequence_error,
};

int handle_apb_sw_event(void* ctxt, struct AddressValuePair* addr_val, int max_count)
{
    (void)max_count;
    struct HSBEmulatorCtxt* hsb_emulator_ctxt = ((struct HSBEmulatorCtxt*)ctxt);

    uint32_t* values = &hsb_emulator_ctxt->apb_ram_data[0];

    if (AVP_GET_VALUE(addr_val)) {
        // sequence values is a byte stream of 32-bit values of format:
        // CMD = 2-bit command code
        // ADDR = 32-bit address.
        // DATA = 32-bit data
        // CMD=WR consumes 2 values (ADDR, DATA), CMD=RD consumes 2 values (ADDR, DATA), CMD=POLL consumes 3 values (ADDR, DATA, DATA)
        // CMD=0 with data to consume is an error, CMD=RD with only one DATA value is a DONE command
        // ADDR/DATA prefixed by (CMD) shows the command that owns it
        // with "[]" delineating 32-bit boundaries, an example sequence will look like:
        // [RWRWRRPWRWWWWWRR]
        //          [(R)ADDR][(R)DATA][(R)ADDR][(R)DATA][(W)ADDR][(W)DATA][(W)ADDR][(W)DATA]
        //          [(W)ADDR][(W)DATA][(W)ADDR][(W)DATA][(W)ADDR][(W)DATA][(R)ADDR][(R)DATA]
        //          [(W)ADDR][(W)DATA][(P)ADDR][(P)DATA][(P)DATA][(R)ADDR][(R)DATA][(R)ADDR][(R)DATA]
        //          [(W)ADDR][(W)DATA][(R)ADDR][(R)DATA][(W)ADDR][(W)DATA][(R)ADDR][(R)DATA]
        // [00000000000000RW]
        //          [(W)ADDR][(W)DATA][(R)DATA] // terminal DONE command only has a single value to read
        //
        uint32_t seq_address = APB_SW_EVENT_START_ADDRESS;
        uint8_t cmd_bit = 0;
        uint32_t cmd_word = next_value(values, &seq_address);
        int count = 0;
        uint8_t cmd_code = next_cmd(values, &cmd_word, &seq_address, &cmd_bit);
        // the next value is already an address
        uint32_t address = next_value(values, &seq_address);

        // check for a DONE command identified as the -1/0xFFFF'FFFF address
        while (DONE_COMMAND_ADDRESS != address) {
            auto callback = APB_EVENT_CMD_CALLBACKS[cmd_code];
            // all 3 commands + error have a valid callback. Any code not 00/01/10 results in an exception
            callback(hsb_emulator_ctxt, &seq_address, address);
            count++;
            cmd_code = next_cmd(values, &cmd_word, &seq_address, &cmd_bit);
            // the next value is already an address
            address = next_value(values, &seq_address);
        }
    }

    return 1;
}

}
