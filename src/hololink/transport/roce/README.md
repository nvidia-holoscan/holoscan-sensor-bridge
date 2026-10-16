# CPU RoCE transport

`CpuRoceTransceiver` is a reusable, CPU-polled RoCE v2 transport for
applications that need direct control over receive and transmit rings. It is
designed for CUDA-Q to replace its legacy CPU RoCE backend. The library has no
CUDA or Holoscan dependency; it can be built inside Hololink or as a
standalone CMake project.

```text
 application / CUDA-Q
          |
          | ring flags, payload slots, optional Unified callback
          v
 CpuRoceTransceiver
   +-------------------+       +-------------------+
   | RX ring + RX CQ   |       | TX ring + TX CQ   |
   +-------------------+       +-------------------+
              \                 /
               +---- UC QP ----+  libibverbs
                         |
                         v
               HSB or RoCE peer
```

## Capabilities

- `Rx`, `Tx`, `Duplex`, `Forward`, `Unified`, and `CallerDriven` modes.
  `CallerDriven` starts no polling thread: the caller serially drives
  `rx_poll()` and `tx_publish()`. Calls must not overlap, but successive calls
  may run on different threads; the caller orders them, for example by joining
  the previous thread or holding a mutex. That is the shape CUDA-Q's unified
  `cpu_dataplane` dispatcher calls.
- Ordinary Send and RDMA Write-with-Immediate transmit operations.
- Independent RX and TX ring geometry.
- Transceiver-owned, caller-owned host, caller-owned GPU pointer, or
  borrowed DMA-BUF payload memory.
- Optional HSB RX metadata trailers.
- Explicit CPU placement for one-poller modes and for both Duplex pollers.
- Standalone build, install, and `find_package(HololinkRoce)` consumption.

The transport uses a UC QP and busy-polls completion queues. It does not add
retransmission or delivery acknowledgement; applications must provide any
required sequencing, timeout, and peer-credit protocol.

The private `ibv.*` wrapper does not log. `RoceTransmitterOp` compiles this
same `ibv.cpp`. Logging stays out of this standalone package.

## Requirements

- Linux.
- A RoCE-capable libibverbs provider.
- rdma-core 34 or newer. `ibv_reg_dmabuf_mr` is an unconditional link-time
  requirement, including host-memory-only configurations.
- HSB IP 0x2602 or newer for RDMA Write-with-Immediate interoperability
  (12-bit slot field). The older 0x2510 layout is not supported for that
  operation.
- CMake 3.24 or newer and a C++17 compiler.

DMA-BUF additionally requires a compatible exporter and GPUDirect RDMA setup.
The caller owns exporter-specific synchronization.

## Build

From the Hololink umbrella project, the target is built when
`HOLOLINK_BUILD_ROCE=ON`:

```sh
cmake -S . -B build -DHOLOLINK_BUILD_ROCE=ON
cmake --build build --target hololink_transport_roce
```

For a CUDA- and Holoscan-free standalone build:

```sh
cmake -S src/hololink/transport/roce -B roce-build
cmake --build roce-build
cmake --install roce-build --prefix "$PWD/roce-install"
```

A consuming CMake project can use:

```cmake
find_package(HololinkRoce REQUIRED CONFIG COMPONENTS transport)
target_link_libraries(my_target PRIVATE hololink::transport::roce)
```

Point `CMAKE_PREFIX_PATH` at the install prefix, or set
`HololinkRoce_DIR` to the generated package directory.

## Lifecycle

Configure the rings and peer, then call `start()`, run
`blocking_monitor()` on the polling thread, and call `close()` from a control
thread:

```cpp
#include <hololink/transport/roce/cpu_roce_transceiver.hpp>

using hololink::transport::roce::CpuRoceTransceiver;

CpuRoceTransceiver::Config config;
config.ibv_name = "mlx5_0";
config.ibv_port = 1;
config.mode = CpuRoceTransceiver::Mode::Rx;
config.local_ip = "192.0.2.10";
config.rx.stride = 4096;
config.rx.slot_count = 8;
config.rx.frame_layout.payload_size = 4096;
config.peer.ip = "192.0.2.20";

CpuRoceTransceiver transceiver { config };
transceiver.start();
transceiver.blocking_monitor(); // another thread calls close()
```

For two-phase peer setup, call `setup()`, exchange
`get_qp_number()`, `get_rkey()`, and ring geometry with the peer, then call
`connect(peer)`. `close()` is idempotent and terminal for that instance.

The full API contract, including ring publication semantics, metadata layout,
DMA-BUF synchronization, mode-specific validation, and concurrency rules, is
documented in `cpu_roce_transceiver.hpp`.

## Memory and placement

A payload may be host memory, a caller-owned GPU pointer, or a borrowed
DMA-BUF. GPU-pointer registration uses `ibv_reg_mr_iova` and needs
nvidia-peermem. DMA-BUF registration does not. `poll_cpu_best_effort`
leaves the current CPU mask in place after a failed affinity read or pin
and does not throw.

Host-trailer receive builds an mlx5 indirect memory key when
`infiniband/mlx5dv.h` is present. Configuration, the runtime library, and
NIC acceptance still decide whether that key is used. Without the header,
or when the key is not used, the trailer is copied from device memory.

`get_path_mtu()` reports the payload bytes of the path MTU programmed on
the queue pair. It is zero until connect reaches RTS. `request_exit()` asks
`blocking_monitor()` to return without releasing the queue pair. In
CallerDriven mode `request_exit()` has no effect, because that mode has no
monitor thread. Stop calling `rx_poll()` and `tx_publish()`, then call
`close()`.

The frame receiver and frame transmitter under
`operators/cpu_roce_transceiver/` are an experimental layer above this
transport. Their headers say so. The stable contract is this transport.

## Tests

Hololink's test suite includes:

- ring, lifecycle, mode, metadata, DMA-BUF validation, and monitor tests;
- private libibverbs wrapper ABI and helper tests;
- build-tree, standalone, and installed-package consumer tests.

Hardware qualification is maintained separately from the product sources.
