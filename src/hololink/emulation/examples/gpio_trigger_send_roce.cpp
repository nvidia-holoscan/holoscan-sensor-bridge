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

#include "hsb_emulator.hpp"
#include "net.hpp"
#include "rocev2_data_plane.hpp"

/*
@brief Example sending gpio trigger status over RoCEv2 from MCU to host.

When a designated GPIO pin interrupt is triggered, the MCU will send a timestamp over RoCEv2 to the host.
The GPIO pin and handlers are target dependent.

For STM32F7 targets:
The GPIO pin is PF15.
The interrupt is triggered on the rising edge of the GPIO pin.

*/

volatile bool trigger_send = false;

using namespace hololink::emulation;

#if defined(STM32F7)
#include "STM32/gpio.hpp"
#include "STM32/stm32_system.h"
#include "STM32/tim.h"

// trigger callback on PF15 rising edge to trigger roce send
extern "C" void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
    if (GPIO_Pin == GPIO_PIN_15) {
        trigger_send = true;
    }
}

// send timestamp roce packet to destination
bool get_send_flag(void)
{
    // check if the GPIO pin PF15 is high to trigger a send
    bool send_frame = false;
    HAL_NVIC_DisableIRQ(EXTI15_10_IRQn);
    if (trigger_send) {
        send_frame = true;
        // clear the trigger flag
        trigger_send = false;
    }
    HAL_NVIC_EnableIRQ(EXTI15_10_IRQn);

    return send_frame;
}
#endif

int main(void)
{
    HSBEmulator hsb;
    // data plane to emit Bootp and send over RoCEv2
    RoCEv2DataPlane data_plane(hsb, IPAddress_from_string("192.168.0.2"), 0, 0);

    // initialize components and modules of the HSBEmulator
    hsb.start();

    while (true) {
        // explicitly call handle_msgs() so that program is compatible with both MCU and hosted Linux targets where it is a no-op
        hsb.handle_msgs();
        if (get_send_flag()) {
            struct timespec current_time;
            if (clock_gettime(CLOCK_REALTIME, &current_time) != 0) {
                Error_Handler();
            }
            data_plane.send((uint8_t*)&current_time, sizeof(current_time), DEFAULT_FRAME_METADATA); // pass frame metadata to ensure a full frame is written
        }
    }

    // any cleanup that is required. User must handle breaking out of the loop above
    hsb.stop();

    return 0;
}
