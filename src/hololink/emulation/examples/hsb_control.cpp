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

#include "data_plane.hpp"
#include "hsb_emulator.hpp"
#include "net.hpp"

/*
@brief Example HSB Emulator instance that is cross-platform in that it will compile for all targets

So that it is cross-platform, it does not use any target-specific features and no input arguments are available.

It will create a default HSBEmulator instance and a default DataPlane instance that responds to control plane messages
directed at the address configured via the IPV4_ADDRESS compile-time macro (default "192.168.0.2").

Compile-time configuration:
  -DIPV4_ADDRESS=\"<a.b.c.d>\"  source IP for this emulator (default "192.168.0.2")
  -DDATA_PLANE_ID=<n>            data-plane id (default 0)
  -DSENSOR_ID=<n>                sensor id (default 0)
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

using namespace hololink::emulation;

int main(void)
{
    HSBEmulator hsb;
    // data plane to emit Bootp. Attempting to send will fail since no transmitter is attached
    // to the default DataPlane
    DataPlane data_plane(hsb, IPAddress_from_string(IPV4_ADDRESS), DATA_PLANE_ID, SENSOR_ID);

    // initialize components and modules of the HSBEmulator
    hsb.start();

    while (true) {
        // explicitly call handle_msgs() so that program is compatible with both MCU and hosted Linux targets where it is a no-op
        hsb.handle_msgs();
    }

    // any cleanup that is required. User must handle breaking out of the loop above
    hsb.stop();

    return 0;
}
