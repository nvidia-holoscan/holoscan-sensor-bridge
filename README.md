# Holoscan Sensor Bridge

## Introduction

Holoscan Sensor Bridge provides a FPGA based interface for low-latency sensor data
processing using GPUs. Peripheral device data is acquired by the FPGA and sent via UDP
to the host system where ConnectX devices can write that UDP data directly into GPU
memory. This software package supports integrating that equipment into Holoscan
pipelines and provides several examples showing video processing and inference using an
IMX274 camera with
[Lattice Holoscan Sensor Bridge device](https://www.latticesemi.com/products/developmentboardsandkits/certuspro-nx-sensor-to-ethernet-bridge-board)
or an IMX477 camera with
[Microchip Holoscan Sensor Bridge](https://www.microchip.com/en-us/products/fpgas-and-plds/boards-and-kits/ethernet-sensor-bridge).

## Setup

Holoscan sensor bridge software comes with an
[extensive user guide](https://docs.nvidia.com/holoscan/sensor-bridge/index.html),
including instructions for setup on
[NVIDIA IGX](https://www.nvidia.com/en-us/edge-computing/products/igx/) and
[NVIDIA AGX](https://developer.nvidia.com/embedded/learn/jetson-agx-orin-devkit-user-guide/index.html)
configurations. Please see the user guide for host configuration and instructions on
running unit tests.

## FPGA IP Source Files

The FPGA IP source code for the Holoscan sensor bridge is available at the following
link:
[FPGA IP Source](https://edge.urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/hsb/fpga_ip/)

## Troubleshooting

Be sure and check the
[release notes](https://docs.nvidia.com/holoscan/sensor-bridge/support/release-notes)
for frequently asked questions and troubleshooting tips.

## Submitting changes

This software is published under the Apache 2.0 license, allowing unrestricted
commercial and noncommercial use. Please consider submitting your changes to this
framework by consulting the instructions in the [CONTRIBUTING.md](CONTRIBUTING.md) file.

Note that all code submissions must pass the rules enforced by running
`scripts/lint/lint.sh`. You can run the source code formatter by executing
`scripts/lint/lint.sh --format` -- this will run
the formatter on all C++, Python, and markdown files in the project, and is usually all
that's necessary to get `scripts/lint/lint.sh` to pass.

## Third-party licensing

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for copied components,
their included paths, and upstream terms. Retained notices are available for
[Analog Devices MIT](third_party_licenses/meta-adi-MIT.txt),
[Microchip device-restricted RTL](third_party_licenses/Microchip-RTL.txt) and
[STM32CubeMX BSD-3-Clause](third_party_licenses/STM32CubeMX-BSD.txt).
These components retain their own license scopes. The inventory also identifies
unresolved provenance and license-distribution findings.
When distributing these components, include the inventory and their corresponding
retained copyright and license notices in or alongside the product.
