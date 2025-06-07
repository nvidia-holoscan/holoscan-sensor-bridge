# Running distributed apps

There are two distributed examples available:

- `distributed_imx274_player.py` is the distributed version of `imx274_player.py`
  application.
- `distributed_tao_peoplenet.py` is the distributed version of `tao_peoplenet.py`
  application.

## Getting the source

```bash

  git clone https://github.com/nvidia-holoscan/holoscan-sensor-bridge.git

```

## Building the Hololink source and starting the Hololink container

```bash

  sh ./docker/build.sh

  export DISPLAY=`(DISPLAY)`

  xhost +

  sh ./docker/demo.sh

```

## Running the distributed IMX player application

### Running the distributed IMX player application on a single node

```bash

  python3 ./examples/distributed_imx274_player.py --driver --worker --fragments all

```

### Running the distributed IMX player application on multiple nodes

The distributed IMX player application can be run on two nodes. One node with the
Hololink board connected can run the source fragment and the other node with the display
can run the visualizer fragment.

On a node having Hololink board connected, run:

```bash

  python3 ./examples/distributed_imx274_player.py --driver --worker --address `(Host IP: Port (example: 5555))` --fragments src_fragment

```

On a node having display connected, run:

```bash

  python3 ./examples/distributed_imx274_player.py --worker --address `(Node 1 IP: Node 1 driver port (example: 5555))` --fragments visualizer_fragment

```

This should open Holoviz window and render camera images on the second node.

## Running the distributed TAO PeopleNet application

Prerequisite: Download the PeopleNet ONNX model from the NGC website:

```bash

wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet/pruned_quantized_decrypted_v2.3.3/files?redirect=true&path=resnet34_peoplenet_int8.onnx' -O examples/resnet34_peoplenet_int8.onnx

```

### Running the distributed TAO PeopleNet application on single node

```bash

  python3 ./examples/distributed_tao_peoplenet.py --driver --worker --fragments all

```

### Running the distributed TAO PeopleNet application on multiple nodes

The distributed TAO PeopleNet application can be run on two nodes. The application
consists of two fragments. The `src_fragment` contains operators related to input
handling and visualization. This fragment needs to be run on the node having Hololink
and display connected to it.

The `inference_fragment` consists of operators related to inferencing and postprocessing
and it should be run on a node with inferencing capabilities.

On a node having Hololink board and display connected, run:

```bash

  python3 ./examples/distributed_tao_peoplenet.py --driver --worker --address `(Host IP: Port (example: 5555))` --fragments src_fragment

```

On a node having inferencing capabilities, run:

```bash

  python3 ./examples/distributed_tao_peoplenet.py --worker --address `(Node 1 IP: Node 1 driver port (example: 5555))` --fragments visualizer_fragment

```

This should open Holoviz window and render the output on the first node.

To know more about the optional parameters like `--driver`, `--worker`, `--fragments`,
and `--address`, refer Holoscan SDK user guide about running Holoscan distributed
applications -
https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_create_distributed_app.html.

### Running the SignalGenerator Python app

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
