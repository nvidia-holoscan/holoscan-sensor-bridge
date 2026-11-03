# Examples

## Getting the source

```bash

  git clone https://github.com/nvidia-holoscan/holoscan-sensor-bridge.git

```

## Building the Hololink source and starting the Hololink container

```bash

  sh ./docker/build.sh

  export DISPLAY=:0  # or your X display, e.g. :1

  xhost +

  sh ./docker/demo.sh

```

## Running the SignalGenerator Python app

The SignalGeneratorApp is a sample app that does the following:

1. Generates two signals. One represents the signal's I component and the other
   represents the signal's Q components.
1. The two signal components are encoded into an IQ buffer
1. The encoded buffer is transmitted to the FPGA using RoCE

Use the following command to get the most up-to-date program options:

**Python:**

```bash
  python3 ./examples/signal_generator.py --help
```

**C++:**

```bash
  ./signal_generator --help
```

## Using locally built holoscan sdk image as base image

1. Build latest Holoscan SDK
1. Modify ./docker/Dockerfile to use latest SDK image as a base image. Image name would
   be holoscan-sdk-build-<arch>-<dgpu or igpu>
1. Build the source and restart container using the steps mentioned above.
1. Also mount SDK install directory and set PYTHONPATH to
   <holoscan-sdk>/public/install-<arch>-<dgpu or igpu>/python/lib/ inside the container.
1. Run the apps following above mentioned steps.
