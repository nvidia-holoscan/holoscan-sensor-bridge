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

## Building the Docker image (HSDK 5.x)

The image installs the public HSDK 5.0.0 release: the debian packages from the
Holoscan local apt repository installer on developer.download.nvidia.com and the
wheels from PyPI. No NVIDIA-internal repositories are needed.

Hololink needs Holoscan SDK 5.0.0 or later (see `MIGRATION.md`).

### 1. Build the hololink-prototype and hololink-demo images

The container type is derived from the host, so no GPU selection is needed. The
host must provide CUDA 13 or newer; L4T hosts (AGX Orin, AGX Thor, IGX Thor, Thor
Mini) must run L4T R38.4.0 or newer (JetPack 7.1+ on Thor, JetPack 7.2+ / R39.2 on
AGX Orin).

```bash
docker/build.sh

# Force the SIPL/FUSA (CoE offload) container type regardless of the DMI
# product name detection (L4T hosts only)
docker/build.sh --sipl
```

`docker/build.sh --hsdk=VERSION` selects a different HSDK release (default `5.0.0`).
The legacy `--igpu`/`--dgpu` options are accepted but ignored.

Disk usage on small targets (an AGX Orin flash leaves 24 to 28 GB free):

- `hololink-prototype` never carries holoscan-inference or TensorRT (`libnvinfer10` alone is
  about 2.9 GB); they are installed into `hololink-demo` only, where the `tao_peoplenet` and
  `body_pose_estimation` examples run. `docker/build.sh --no-inference` leaves them out of
  `hololink-demo` as well.
- Between the two image builds, and again after `hololink-demo`, `docker/build.sh` can drop
  the BuildKit cache records the build created: the layer snapshots and the apt/pip/sccache
  cache mounts. Records that were already in the cache when the script started (other
  projects' builds on a shared host or CI runner) are never removed. With Docker's
  containerd image store (the JetPack 7 default) every built layer is on disk three times while
  its image is exported (build snapshot, compressed blob, unpacked image) and 1.6 times at rest;
  pruning as soon as each image is in the store lowers the peak by roughly the size of that
  image's own layers, at the cost of a full rebuild and re-download next time. The default is
  derived from the free space on the Docker data root: below 30 GiB (a stock AGX Orin flash)
  the prune is on, otherwise off. `--prune-cache` / `--no-prune-cache` (or `PRUNE_CACHE=1|0`
  in the environment) override it, and the script prints a banner whenever it prunes.
- Even so, `hololink-prototype` (8.9 GB) plus a `hololink-demo` with TensorRT (15.4 GB) occupy
  about 25 GB at rest in the containerd image store, more than a full AGX Orin flash leaves
  free. On such a host either build with `--no-inference` (`hololink-demo` is then about 10 GB
  and the pair about 16 GB at rest), or do not install TensorRT on the host when flashing: the
  container carries its own copy. To remove an already installed host TensorRT, first mark the
  other JetPack components as manually installed so that removing the `nvidia-jetpack`
  meta-packages does not make `apt autoremove` take them along, preview, then purge:

  ```sh
  apt-mark showauto | grep -E 'cuda|cudnn|nvidia|libnv|vpi|nvsci' | xargs -r sudo apt-mark manual
  sudo apt-get -s purge '^libnvinfer' '^libnvonnxparsers' '^tensorrt' '^python3-libnvinfer' '^nvidia-tensorrt'
  sudo apt-get purge '^libnvinfer' '^libnvonnxparsers' '^tensorrt' '^python3-libnvinfer' '^nvidia-tensorrt'
  sudo apt-get clean
  ```

### 2. Enter the container

```bash
docker/demo.sh
```

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
