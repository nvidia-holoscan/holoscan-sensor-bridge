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

#include <cstdint>
#include <cstring>

#include "coe_data_plane.hpp"
#include "hsb_emulator.hpp"
// this should be the target-specific net.hpp. make sure to use the correct include order
#include "net.hpp"

/*
@brief Cross-platform HSB Emulator that emits a message buffer over the COE data plane every loop iteration.

Each transmitted buffer is the concatenation of a 4-byte little-endian uint32_t loop counter followed by
the bytes of the TRANSPORT_MSG compile-time macro (no NUL terminator). The counter increments by 1 each
iteration so receivers can verify ordering.

Compile-time configuration:
  -DIPV4_ADDRESS=\"<a.b.c.d>\"   source IP for this emulator (default "192.168.0.2")
  -DDATA_PLANE_ID=<n>             data-plane id (default 0)
  -DSENSOR_ID=<n>                 sensor id (default 0)
  -DTRANSPORT_MSG=\"...\"         payload string appended after the counter
                                  (default "test msg 0123456789abcdefg")
  -DFRAME_RATE=<n>                send rate in frames per second; the loop sleeps
                                  msleep(1000 / FRAME_RATE) between sends (default 30)
*/

#ifndef IPV4_ADDRESS
#define IPV4_ADDRESS "192.168.0.2"
#endif
#ifndef DATA_PLANE_ID
#define DATA_PLANE_ID 0
#endif
#ifndef SENSOR_ID
#define SENSOR_ID 0
#endif
#ifndef TRANSPORT_MSG
#define TRANSPORT_MSG "test msg 0123456789abcdefg"
#endif
#define DEFAULT_FRAME_RATE 30
#ifndef FRAME_RATE
#define FRAME_RATE DEFAULT_FRAME_RATE
#endif

using namespace hololink::emulation;

int main(void)
{
    HSBEmulator hsb;
    COEDataPlane data_plane(hsb, IPAddress_from_string(IPV4_ADDRESS), DATA_PLANE_ID, SENSOR_ID);

    // Build the per-iteration buffer once: [uint32_t counter][TRANSPORT_MSG bytes].
    // sizeof drops the implicit NUL terminator on the macro string literal.
    static constexpr size_t kMsgLen = sizeof(TRANSPORT_MSG) - 1;
    static constexpr size_t kBufLen = sizeof(uint32_t) + kMsgLen;
    uint8_t buffer[kBufLen];
    std::memcpy(buffer + sizeof(uint32_t), TRANSPORT_MSG, kMsgLen);

    hsb.start();

    static constexpr unsigned kFrameIntervalMs = (FRAME_RATE > 0) ? (1000u / FRAME_RATE) : (1000u / DEFAULT_FRAME_RATE);
    uint32_t n_counter = 0;
    uint32_t h_counter = 0;
    while (true) {
        hsb.handle_msgs();
        n_counter = htonl(h_counter);
        std::memcpy(buffer, &n_counter, sizeof(uint32_t));
        if (data_plane.send(buffer, kBufLen, DEFAULT_FRAME_METADATA) < 0) {
            Error_Handler("Failed to send buffer");
            break;
        }
        h_counter++;
        msleep(kFrameIntervalMs);
    }

    hsb.stop();
    return 0;
}
