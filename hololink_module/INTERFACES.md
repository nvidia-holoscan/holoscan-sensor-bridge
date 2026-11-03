# Released `hololink_module` Interfaces

Application code reaches the services on an HSB device through version-tagged interface
objects — `HololinkInterfaceV1`, `I2cInterfaceV1`, and so on — resolved by name through
the service locator. This file records which of those interfaces have been released, and
which release froze each one.

A released interface is frozen: its methods are never added, removed, reordered, or
re-signatured, and its `type_id` keeps resolving to an implementation that honours the
same contract. New capability arrives as a *new* versioned class — typically deriving
from the released one — published under its own `type_id` alongside the old. An
application that doesn't need the new capability keeps naming the released version and
keeps working; one that does names the newer version explicitly and fails fast against a
module that doesn't publish it.

Freezing is per interface, not per release. A release is simply the point at which a
particular interface version was frozen, and only the interfaces that actually changed
carry a newer version. Each interface declares the release that froze it as a
`static constexpr const char* release_version` member next to its `type_id`; an interface
no release has published yet declares `UNRELEASED` instead, and an implementation inherits
the marker of the interface it implements. A fixture that exists only to exercise the
framework declares `TEST_ONLY` and never appears in this file — it is not for application
use. See `plans/hololink_module.md` § Versioning for the full rationale.

Sections below are ordered by `VERSION`, newest first, and each section lists only the
interfaces frozen at that release. The current API surface is the accumulation of every
section.

This file is also intended as an input to automated API documentation generation; that
generation is not wired up yet.

## 2.7.0

The first released interface set. Header paths are relative to
`hololink_module/host/include/` unless noted otherwise.

### `ReactorV1`

`hololink/module/reactor.hpp` · `reactor.v1`

Process-wide event-loop singleton. Every callback it dispatches runs sequentially on a
single thread, so handlers never observe concurrent execution of one another and require
no internal locking against each other.

### `LoggingInterfaceV1`

`hololink/module/logging.hpp` · `logging.v1`

Process-wide log sink. The host registers exactly one implementation; modules cache the
looked-up pointer for the life of the `.so` and the `HSB_LOG_*` macros dispatch through
it.

### `EnumerationInterfaceV1`

`hololink/module/enumeration.hpp` · `enumeration.v1`

Per-module discovery hook. The Adapter calls `update_metadata` after a bootp packet
arrives (or after a manual `enumerate()` call) and the module enriches the metadata with
the version-specific fields it knows about — VP/SIF/HIF addresses, sensor counts,
board-specific descriptors.

### `ChannelConfigurationInterfaceV1`

`hololink/module/channel_configuration.hpp` · `channel_configuration.v1`

Per-module helper for the module-aware `EnumerationMetadata` edits an application makes
before handing the metadata to a per-channel service constructor: selecting a sensor on
the board, recording the MTU the RoCE data plane should honour, and recording a multicast
destination.

### `HololinkInterfaceV1`

`hololink/module/hololink.hpp` · `hololink.v1`

Per-board control plane, keyed by `"serial=<serial_number>"`. Carries the per-board
register access, reset and PTP surface, the per-board enumeration metadata snapshot every
per-board service reads through, and the factory methods for the board's I2C buses,
sequencers, data channels and VSYNC sources.

### `I2cInterfaceV1`

`hololink/module/i2c.hpp` · `i2c.v1`

A single I2C bus + peripheral pair on the board. Runs synchronous I2C transactions, and
encodes transactions into a `SequencerInterfaceV1` for deferred or event-triggered
execution.

### `I2cLockV1`

`hololink/module/i2c_lock.hpp` · not a locator service

Mutex-like handle for the per-board I2C bus, satisfying the BasicLockable + Lockable
concepts so callers can drive it through `std::lock_guard` / `std::unique_lock` /
`std::scoped_lock`. Returned unlocked from `HololinkInterfaceV1::i2c_lock`.

### `SequencerInterfaceV1`

`hololink/module/sequencer.hpp` · `sequencer.v1`

A pre-programmed command sequence (reads, writes, polls) that executes atomically on the
FPGA, triggered by a hardware event — frame end, frame start, GPIO — or by software.
`location()` returns the base address in FPGA memory so callers can read results back
after execution.

### `DataChannelInterfaceV1`

`hololink/module/data_channel.hpp` · `data_channel.v1`

Per-channel anchor, transport-agnostic, keyed by
`"serial=<serial_number>;data_channel=<n>"`. Caches the two values every per-channel
service reads off: the channel's enumeration metadata, and the parent per-board
`HololinkInterfaceV1`.

### `RoceDataChannelInterfaceV1`

`hololink/module/roce_data_channel.hpp` · `roce_data_channel.v1`

RoCE transport view of a per-channel `DataChannelInterfaceV1`. Adds the RoCE-specific
transport surface — `attach_receiver` / `detach_receiver` / `frame_end_sequencer` — and
programs the on-device packetizer to deliver frames into receiver-managed memory.

### `CoeDataChannelInterfaceV1`

`hololink/module/coe_data_channel.hpp` · `coe_data_channel.v1`

CoE transport view of a per-channel `DataChannelInterfaceV1`, adding the IEEE-1722
transport surface (`configure_coe` / `unconfigure`) and packetizer selection for the
stream's CSI pixel format.

### `LinuxDataChannelInterfaceV1`

`hololink/module/linux_data_channel.hpp` · `linux_data_channel.v1`

Software (Linux) transport view of a per-channel `DataChannelInterfaceV1`. The wire
protocol is still RoCEv2, but the receiver lives in user space, so the channel binds a
host socket (`configure_socket`) and programs the FPGA with that socket's local port.

### `RoceReceiverInterfaceV1`

`hololink/module/roce_receiver.hpp` · `roce_receiver.v1`

Per-channel ibverbs receiver. Brought up by the operator with `start(...)`, handed to
`RoceDataChannelInterfaceV1::attach_receiver` so the FPGA can be programmed from it, then
run on a dedicated thread via `blocking_monitor()` while `get_next_frame()` pulls frames.

### `RoceReceiverFrameInfoV1`

`hololink/module/roce_receiver.hpp` · not a locator service

Per-frame info populated by `RoceReceiverInterfaceV1::get_next_frame` — the frame's
device pointers plus per-frame statistics.

### `LinuxReceiverInterfaceV1`

`hololink/module/linux_receiver.hpp` · `linux_receiver.v1`

Per-channel software receiver. Where `RoceReceiverInterfaceV1` fronts a hardware ibverbs
QP, this reassembles HSB's RoCEv2 UDP packets in user space over an ordinary datagram
socket, so it needs no infiniband device and links no ibverbs.

### `LinuxReceiverFrameInfoV1`

`hololink/module/linux_receiver.hpp` · not a locator service

Per-frame info populated by `LinuxReceiverInterfaceV1::get_next_frame`. Unlike the
hardware path, the software receiver decodes the per-frame metadata as it reassembles the
stream, so those fields are returned here directly.

### `FrameMetadataInterfaceV1`

`hololink/module/frame_metadata.hpp` · `frame_metadata.v1`

Singleton service that decodes the fixed-size end-of-frame metadata block the device
deposits into the receiver's host buffer. Receivers pass a pointer + size and consume the
structured fields.

### `OscillatorInterfaceV1`

`hololink/module/oscillator.hpp` · `oscillator.v1`

Per-data-plane oscillator the supplement programs to drive a downstream sensor's pixel
clock. Exposes the requested rate through `enable`, plus implementation-defined tunables
through `get_caps` / `set_caps`.

### `VsyncInterfaceV1`

`hololink/module/vsync.hpp` · `vsync.v1`

Abstract base for any source that drives a board's VSYNC trigger line. Consumers — camera
drivers — only need to know whether some source is currently driving the trigger so they
can switch the sensor's trigger-input mode on or off; they do not configure the source.

### `PtpPpsOutputInterfaceV1`

`hololink/module/ptp_pps_output.hpp` · `ptp_pps_output.v1` (also published as `vsync.v1`)

PTP-PPS-driven VSYNC source, one per board. A PTP-locked pulse-per-second train drives the
trigger line connected to the cameras' frame-start input, at a caller-programmed rate.

### `CsiConverterV1`

`hololink/module/csi_converter.hpp` · not a locator service

Interface for an object that interprets received CSI image data. A sensor driver calls
these methods to train the converter on how to read the block of memory the device
delivers, and reads the receiver-side geometry back off it.

### `HsbLiteInterfaceV1`

`hololink/module/hsb_lite/hsb_lite.hpp` (in `hololink_module/module/hsb_lite/include/`) ·
`hsb_lite.v1`

HSB-Lite board supplement, one instance per board. Exposes the board-specific surface on
top of the canonical services — notably programming the on-board Renesas Bajoran Lite TS1
clock generator.

### `LeopardVb1940InterfaceV1`

`hololink/module/leopard_vb1940/leopard_vb1940.hpp` (in
`hololink_module/module/leopard_vb1940/include/`) · `leopard_vb1940.v1`

Leopard VB1940 board supplement, one instance per board. Board-level bring-up primitives
the VB1940 sensor driver reaches for; each is idempotent, so a camera driver can invoke it
from `configure()` without coordinating with the board's other channels.

### `TauroTechDa326InterfaceV1`

`hololink/module/taurotech_da326/taurotech_da326.hpp` (in
`hololink_module/module/taurotech_da326/include/`) · `taurotech_da326.v1`

TauroTech DA326 board supplement. Exposes the board-specific ops: camera reset release,
deserializer power cycle, power check, and I2C clock setup.

### `TutorialDeviceInterfaceV1`

`hololink/module/tutorial/tutorial_device.hpp` (in
`hololink_module/module/tutorial/include/`) · `tutorial_device.v1`

Tutorial device status service — the one board-specific capability the Tutorial device
adds on top of the canonical HSB services: a status LED.
