/*
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
See README.md for detailed information.
*/

#include "STM32/gpio.hpp"
#include "STM32/tim.h"
#include "hsb_emulator.hpp"
#include "net.hpp"
#include "rocev2_data_plane.hpp"

/*
@brief Example gpio status send over RoCEv2 from MCU to host that is triggered by an internal timer.

The timer is set to 1 second by default, but can be updated from host via the SEND_DELAY_ADDRESS register.
For example on host:
    hololink write_uint32 SEND_DELAY_ADDRESS 10000

    will set the timer to 10ms updates.

Time and GPIO status are target dependent, but the GPIO status follows the same format:

Every 2 bits (BitPair) corresponds to a GPIO pin with the msb being the error bit and the lsb being the value of the pin.

For STM32F7 targets:
The timer is TIM7 set up for 96 MHz clock.
The GPIO status is a 32-bit value with the following bit pairs:
- BitPair 0: LED_GREEN
- BitPair 1: LED_BLUE
- BitPair 2: LED_RED
- BitPair 3: I/O pin PE14, which is set up for slow input.
*/

using namespace hololink::emulation;

// default send delay is 1 second
uint32_t send_delay_us = 1000000;
volatile bool send_flag = false;
// address to update the send delay from host
static const uint32_t SEND_DELAY_ADDRESS = 0x88u;

#ifdef STM32F7
// TIM7 periodic interrupt (e.g. for send_delay_us)
#define TIM7_CLOCK_HZ 96000000U
#define TIM7_ARR_MAX 65535U
static TIM_HandleTypeDef htim7;

static void compute_prescaler_period(uint32_t period_us, uint32_t* prescaler, uint32_t* period)
{
    // (psc+1)*(arr+1) = period_us * TIM7_CLOCK_HZ / 1e6
    uint64_t period_ticks = (uint64_t)period_us * (TIM7_CLOCK_HZ / 1000000U);
    if (period_ticks <= TIM7_ARR_MAX + 1U) {
        *prescaler = 0;
        *period = (period_ticks > 0) ? (uint32_t)(period_ticks - 1) : 0;
    } else {
        *prescaler = (uint32_t)((period_ticks - 1) / (TIM7_ARR_MAX + 1U));
        *period = (uint32_t)(period_ticks / (*prescaler + 1U)) - 1U;
        if (*period > TIM7_ARR_MAX) {
            *period = TIM7_ARR_MAX;
        }
    }
}

int timer_setup(uint32_t period_us)
{
    uint32_t prescaler, period;
    compute_prescaler_period(period_us, &prescaler, &period);

    htim7.Instance = TIM7;
    htim7.Init.Prescaler = prescaler;
    htim7.Init.CounterMode = TIM_COUNTERMODE_UP;
    htim7.Init.Period = period;
    htim7.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
    htim7.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;

    if (timer_init(&htim7, 5, 0) != 0) {
        return -1;
    }
    return timer_start(&htim7);
}

int timer_update(uint32_t period_us)
{
    if (timer_stop(&htim7) != 0) {
        return -1;
    }
    uint32_t psc, arr;
    compute_prescaler_period(period_us, &psc, &arr);
    htim7.Init.Prescaler = psc;
    htim7.Init.Period = arr;
    if (HAL_TIM_Base_Init(&htim7) != HAL_OK) {
        return -1;
    }
    return timer_start(&htim7);
}

extern "C" void TIM7_IRQHandler(void)
{
    HAL_TIM_IRQHandler(&htim7);
}

// HAL callback for TIM7 period elapsed (C linkage so HAL C code calls this)
extern "C" void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef* htim)
{
    (void)htim;
    if (htim->Instance == TIM7) {
        send_flag = true;
    }
}

bool get_send_flag(void)
{
    // check if the GPIO pin PF15 is high to trigger a send
    bool send_frame = false;
    HAL_NVIC_DisableIRQ(TIM7_IRQn);
    if (send_flag) {
        send_frame = true;
        // clear the trigger flag
        send_flag = false;
    }
    HAL_NVIC_EnableIRQ(TIM7_IRQn);
    return send_frame;
}

// nucleo-f767zi specific function to get the status of the LED GPIO pins and input pin PE14
uint32_t get_gpio_status(HSBEmulator& hsb)
{
    uint32_t led_status;
    uint32_t gpio_status = 0;
    // leds are output GPIO pins PB0, PB7, and PB14 in bank 0xC. So to turn on all leds, hololink write_uint32 0xC 0x40810000
    if (hsb.read(GPIO_STATUS_BASE_REGISTER, led_status)) {
        gpio_status |= 0x3F; // set error bits
    } else {
        gpio_status |= (led_status & 0x00010000) >> 15; // put red led in bit 1
        gpio_status |= (led_status & 0x00800000) >> 20; // put green led in bit 3
        gpio_status |= (led_status & 0x40000000) >> 25; // put blue led in bit 5
    }
    // input pin PE14 which is bit 14 in third bank of GPIO_STATUS_BASE_REGISTER
    uint32_t input_status;
    if (hsb.read(GPIO_STATUS_BASE_REGISTER + 2 * REGISTER_SIZE, input_status)) {
        gpio_status |= 0xC0; // set error bit
    } else {
        gpio_status |= (input_status & 0x00004000) >> 7; // put input pin in bit 7
    }
    return gpio_status;
}
#endif

// callback to update the send delay from host on SEND_DELAY_ADDRESS register
int update_send_delay(void* ctxt, struct AddressValuePair* addr_vals, int max_count)
{
    (void)ctxt;
    int i = 0;
    while (i < max_count) {
        uint32_t address = AVP_GET_ADDRESS(addr_vals + i);
        if (address != SEND_DELAY_ADDRESS) {
            break;
        }
        send_delay_us = AVP_GET_VALUE(addr_vals + i);
        i++;
    }
    if (i) {
        if (timer_update(send_delay_us)) {
            Error_Handler();
        }
    }
    return i;
}

int main(void)
{
    HSBEmulator hsb;
    // data plane to emit Bootp and send over RoCEv2
    RoCEv2DataPlane data_plane(hsb, IPAddress_from_string("192.168.0.2"), 0, 0);

    if (hsb.register_write_callback(SEND_DELAY_ADDRESS, SEND_DELAY_ADDRESS + REGISTER_SIZE, update_send_delay, nullptr)) {
        Error_Handler();
    }

    // initialize components and modules of the HSBEmulator
    hsb.start();

    timer_setup(send_delay_us);
    while (true) {
        // explicitly call handle_msgs() so that program is compatible with both MCU and hosted Linux targets where it is a no-op
        hsb.handle_msgs();
        if (get_send_flag()) {
            uint32_t gpio_status = get_gpio_status(hsb);
            data_plane.send((uint8_t*)&gpio_status, sizeof(gpio_status), DEFAULT_FRAME_METADATA); // pass frame metadata to ensure a full frame is written
        }
    }

    // any cleanup that is required. User must handle breaking out of the loop above
    hsb.stop();

    return 0;
}
