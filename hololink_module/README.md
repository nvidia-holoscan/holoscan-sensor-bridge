# hololink_module

The hololink_module library is a standalone, modular replacement for the
register/peripheral surface in `src/hololink/core/`. It is built as a sequence of phases
described in `plans/hololink_module.md`. This README tracks what each phase implemented
and how to build / test that phase's deliverables.

## Phase 1 — Framework skeleton

**Implemented.** The minimal scaffolding the rest of the module builds on: the C ABI
(`hololink_module_status_t`, `hololink_module_init_t`, `hololink_module_services_t` with
its `status` field, plus `hololink_module_get_abi_check`), the `ServiceLocatable<T>`
CRTP, the abstract `Module` class with the concrete `LoadedModule` subclass that wraps a
peer's callback pair (and optionally a dlopen-handle keeper), the `Publisher` class that
owns the per-binary registry and exposes `self_module()` for in-binary lookups, the
`Adapter` process-wide singleton with a long-lived host `Publisher`, the
`dlopen(RTLD_LOCAL)` + ABI check load path, and the `add_hololink_module()` CMake
helper. `hololink_module/` is wired into the top-level build via `add_subdirectory`.
Three gtests under `tests/` exercise the full chain: a stub module .so built with
`add_hololink_module()` loads, exchanges a service in both directions (host publishes a
value, the stub reads it back and republishes it), and a deliberately-mismatched ABI
stub is rejected before init.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD"
cmake --build "$BUILD" --target hololink_module
cmake --build "$BUILD" --target hololink_module_runtime
cmake --build "$BUILD" --target hololink_module_framework_test
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" -R hololink_module_framework_test --output-on-failure
```

## Phase 2 — Host singletons (Reactor + Logging)

**Implemented.** Two abstract V1 service interfaces — `ReactorV1` and
`LoggingInterfaceV1` — plus their concrete, module-owned implementations (no dependency
on legacy `src/hololink/core`; see **Host isolation**). `Reactor`
(`host/src/reactor_impl.cpp`) is the `ReactorV1` service backed by its own dedicated
poll thread; it is a process singleton, created once and never freed (so the thread is
never joined during static destruction). `Logging` is the host's own stderr sink, owning
its log level (parsed from the same `HOLOSCAN_LOG_LEVEL` / `HOLOLINK_LOG_LEVEL` env
vars) and applying `gettid()` + monotonic timestamp decoration. The `Adapter`
constructor publishes both singletons on the host `Publisher` (instance_id `""`) and
seeds the host's per-binary `hsb_logger_cache`. The
`HSB_LOG_{TRACE,DEBUG,INFO,WARN,ERROR}` macros in `logging.hpp` short-circuit on the
cached logger's `level()` then dispatch through it; modules populate their own copy of
the cache during `hololink_module_init` after fetching the logger via
`LoggingInterfaceV1::get_service`. A new gtest under `tests/` loads a stub that fetches
both singletons, schedules a Reactor callback, and from inside the callback emits an
`HSB_LOG_INFO` line — the test verifies the log record arrives at the host-installed
sink with intact file/line/function/level metadata.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD"
cmake --build "$BUILD" --target hololink_module_singletons_test
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" -R hololink_module_singletons_test --output-on-failure
```

## Phase 3 — Enumeration

**Implemented.** The discovery surface a board's bootp metadata flows through. New
public types: `EnumerationMetadata` (a
`std::map<std::string, std::variant<int64_t, std::string, std::vector<uint8_t>>>`
subclass with a typed `get<T>` helper) and `EnumerationInterfaceV1` (per-module abstract
interface with `update_metadata(metadata, raw_packet, raw_packet_len)`). New `Adapter`
API: `enumerate(EnumerationMetadata)` looks up `metadata["fpga_uuid"]`, optionally
`metadata["compat_id"]`, finds the matching `hololink_<uuid>[_<compat>].so` under the
configured module directory, calls its `EnumerationInterfaceV1::update_metadata` to
enrich the metadata, and stores the result; `find_channel(channel_ip)` returns the
enriched metadata for a given peer IP; `set_module_directory()` overrides the search
directory (env var `HOLOLINK_MODULE_DIR` and the default `/usr/lib/hololink/modules`
otherwise). The ABI check struct gained `size_of_enumeration_metadata`,
`align_of_enumeration_metadata`, `size_of_std_string`, and `align_of_std_string` fields
so toolchain / STL drift between host and module is rejected before either side
constructs a metadata object that would otherwise differ in layout. Four gtests under
`tests/` cover the manual-enumerate happy path (load stub by UUID → `update_metadata`
runs across the .so boundary → `find_channel` observes both original and enriched
fields) plus rejection of missing UUID, unknown UUID, and unknown channel IP.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD"
cmake --build "$BUILD" --target hololink_module_enumeration_test
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" -R hololink_module_enumeration_test --output-on-failure
```

## Phase 4 — V1 service surface + `module/core/` wrappers (partial)

**Implemented.** Every V1 abstract interface header that the rework's example will
exercise is now in place under `host/include/hololink/module/`: `hololink.hpp`
(`HololinkInterfaceV1`), `roce_data_channel.hpp` (`RoceDataChannelInterfaceV1`),
`i2c.hpp` (`I2cInterfaceV1`), `i2c_lock.hpp` (`I2cLockV1`), `sequencer.hpp`
(`SequencerInterfaceV1`), `frame_metadata.hpp` (`FrameMetadataInterfaceV1`). Each
interface follows the project conventions — `MixedCaseV1` + alias, singleton or
`name=value;...` instance_id pattern, virtual-only surface — and the cross-references
between them (child accessors on `HololinkInterfaceV1` for I2C / RoCE / sequencer;
`get_hololink` and `frame_end_sequencer` on `RoceDataChannelInterfaceV1`) compile
through forward declarations so the headers stay independently includable.

A new `module/core/` source tree hosts the default implementations. Two are landed and
unit-tested in this phase:

- **`I2cLockImplV1`** — wraps a caller-supplied `std::shared_ptr<std::mutex>`,
  implementing `lock` / `unlock` / `try_lock` faithfully so the handle plugs into
  `std::lock_guard` / `std::unique_lock` / `std::scoped_lock`.
- **`FrameMetadataV1`** — owns the layout of the device's 48-byte end-of-frame metadata
  block (per the plan's "implementation in `module/core/` owns the layout" wording) and
  decodes it directly into the V1 struct. Rejects undersized buffers and null pointers
  up-front.

The `hololink::module_core` static archive (emitted from `module/core/CMakeLists.txt`)
is wired into the module tree's top-level CMake and carries the head-of-tree
`DEFAULT_COMPAT 2603` CMake property, ready for `add_hololink_module()` to read in Phase
5+. The ABI check struct gained `size_of_enumeration_metadata`,
`align_of_enumeration_metadata`, `size_of_std_string`, and `align_of_std_string` fields
in Phase 3; this phase tests that every stub built with the current `module_runtime`
archive still passes the host's checks.

The four other default wrappers (`HololinkV1`, `RoceDataChannelV1`, `I2cV1`,
`SequencerV1`) are deferred to **Phase 5**, where the HSB-Lite module brings the wrapper
plus a published instance through one CMake target and the integration test exercises
the whole chain end-to-end. Splitting the work this way keeps each phase's "Done when"
criterion verifiable with the tests landed in that same phase.

Five gtests cover the implemented wrappers: `I2cLockImpl` BasicLockable compliance +
thread-serialization invariant + `try_lock`; `FrameMetadataImpl` all-fields decode
against a synthetic block + undersized-buffer rejection + null-pointer rejection.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD"
cmake --build "$BUILD" --target hololink_module
cmake --build "$BUILD" --target hololink_module_core_test
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" -R hololink_module_core_test --output-on-failure
```

## Phase 5 — HSB-Lite module (C++ side; pybind / Python deferred)

**Implemented.** The HSB-Lite module tree under `hololink_module/module/hsb_lite/`.
Claims the real HSB-Lite UUID (`889b7ce3-65a5-4247-8b05-4ff1904c3359`); a per-module
`add_hololink_module()` invocation produces
`hololink_889b7ce3-65a5-4247-8b05-4ff1904c3359.so` under the build's module dir. The
module's `module_entry.cpp` constructs a `hololink::module::Publisher`, publishes the
`FrameMetadataV1` singleton from `module/core/`, and publishes its own
`HsbLiteEnumerationV1` as the `EnumerationInterfaceV1` override. The override stamps
`module_name=hsb_lite` on every metadata it sees and backfills `compat_id` with the
head-of-tree default `2603` when the bootp payload didn't carry one. Two new
public-facing types: the abstract `HsbLiteInterfaceV1` (under
`module/hsb_lite/include/hololink/module/hsb_lite/hsb_lite.hpp`) with a
`setup_clock(profile)` member, a fire-and-forget `trigger_reset()` member (writes the
board's reset registers and returns without waiting for re-enumeration or reconfiguring
HSB — recovery is the pipeline's job; used to induce a mid-capture loss in reconnection
tests), and a static `get_hsb_lite(module, metadata)` convenience, plus the new
`hololink::hsb_lite::headers` INTERFACE CMake target applications and per-module Python
sub-packages link to reach the type. Three gtests under `tests/` cover:
`Adapter::enumerate` loads the HSB-Lite .so by UUID and the override's enrichment is
observed through `Adapter::find_channel`; a bootp-supplied `compat_id` is preserved (not
overwritten by the default backfill); and the FrameMetadata singleton fetched through
the loaded `Module` decodes a known 48-byte block correctly.

**Out of scope this phase.** The four hardware-coupled wrappers (`HololinkV1`,
`RoceDataChannelV1`, `I2cV1`, `SequencerV1`), `HsbLiteV1` (the concrete supplement that
holds a `HololinkInterface` and drives the Renesas Bajoran clock through it), the
`_hololink_py_module.so` pybind extension, the `hololink_module.hsb_lite` Python
sub-package, and the Python end-to-end script. Those land in a follow-up phase that
pairs them with an HSB emulator integration test so each wrapper commits together with
its first real run-through.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD"
cmake --build "$BUILD" --target hsb_lite
cmake --build "$BUILD" --target hololink_module_hsb_lite_test
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" -R hololink_module_hsb_lite_test --output-on-failure
```

## Phase 5b — Core pybind11 bindings

**Implemented.** A `_hololink_py_module` pybind11 extension under
`host/python/hololink_module_py.cpp` that exposes the slice of the core surface needed
by Python applications driving the module: `Adapter` (singleton accessor +
`set_module_directory` / `enumerate` / `find_channel` / `load_module`), `Module` (opaque
holder for the Python-facing `T.get_service(module, ...)` calls), `EnumerationMetadata`
(with `__getitem__` / `__setitem__` / `__contains__` / `__len__` / `get(key, default)`
mapping the `int64 / str / bytes` variant alternatives onto Python types), the
`FrameMetadata` plain-data struct, and `FrameMetadataInterface` (static
`get_service(module)` + `decode(host_memory)` accepting any buffer-protocol object). The
accompanying `__init__.py` exports the unversioned aliases (`FrameMetadataInterface`)
and pins the V1 names on top of them (`FrameMetadataInterfaceV1`) per the project
convention. The CMake plumbing ships the `.so` and `__init__.py` together under
`${BUILD}/python/hololink_module/` so
`PYTHONPATH=${BUILD}/python python3 -c "import hololink_module"` just works. Three
pytest tests under `tests/` cover the full path: manual-enumerate driving the HSB-Lite
`.so` from Python and observing the enriched metadata; the unversioned-to-V1 alias
identity; and the `EnumerationMetadata.get(key, default)` accessor.

**Out of scope this phase.** Bindings for the V1 interfaces deferred from Phase 5
(`HololinkInterface`, `RoceDataChannelInterface`, `I2cInterface`, `I2cLockV1`,
`SequencerInterface`, `ReactorV1`, `LoggingInterface`, `EnumerationInterface`) and the
per-module `hololink_module.hsb_lite` Python sub-package; those land alongside the
deferred concrete impls when the HSB emulator integration arrives. Holoscan-coupled
operator bindings (Phase 6) ship in their own `_hololink_module_operators.so` extension.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD" -DHOLOLINK_BUILD_PYTHON=ON
cmake --build "$BUILD" --target _hololink_py_module
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" -R hololink_module_python_smoke_test --output-on-failure
```

## Phase 5 extension — module/core V1 wrappers

**Implemented.** The four module-side V1 wrappers Phase 4 deferred now ship under
`hololink_module/module/core/` alongside the existing `FrameMetadataV1` /
`I2cLockImplV1`. Each is a header-only thin delegating wrapper over the matching
`hololink::core` legacy type:

- `HololinkV1` (`hololink_default.hpp`) — wraps a
  `std::shared_ptr<LegacyHololinkAccess>`, where `LegacyHololinkAccess` is a tiny
  subclass of `hololink::Hololink` that re-exposes the legacy class's protected
  `configure_hsb` / `and_uint32` / `or_uint32` (those three are protected on the legacy
  class but published on V1's surface). Implements the V1 lifecycle (`start` / `stop` /
  `reset` / `configure_hsb`), the batched register I/O (`write_uint32` / `read_uint32` /
  `and_uint32` / `or_uint32`), and `i2c_lock` returning a `unique_ptr<I2cLockNamedV1>`
  over the legacy `NamedLock&`. The vector-form `write_uint32` / `read_uint32` take
  parallel address/value arrays and issue them as a **single control-plane UDP
  message**, not one round-trip per register: writes batch every pair into a
  `Hololink::WriteData` (one `WR_BLOCK`), and a read of consecutive registers (stride 4)
  goes out as one `RD_BLOCK` — only genuinely scattered read addresses fall back to
  per-address requests. Both pass the backing Hololink's sequence-number-checking
  setting so a batched read stays on the same setting as every other request. The
  protected V1 instance-id hooks build `serial=<n>;data_plane=<n>` and
  `serial=<n>;bus=<n>;address=<n>` so the supplement-published per-board children (I2c /
  RoCE / sequencers) resolve correctly through the templated factories.
- `I2cV1` (`i2c_default.hpp`) — wraps `std::shared_ptr<hololink::Hololink::I2c>`.
  `i2c_transaction` delegates directly. `encode_i2c_request` returns `INVALID_PARAMETER`
  until a virtual hook lands that lets the wrapper recover the legacy
  `Hololink::Sequencer&` from an arbitrary `SequencerInterfaceV1` (the `-fno-rtti` build
  forbids `dynamic_cast`). The IMX274 driver reaches `encode_i2c_request` only from
  synchronized test-pattern paths that aren't on Phase 8's critical path; the bridging
  hook is deferred to integration.
- `SequencerV1` (`sequencer_default.hpp`) — wraps
  `std::shared_ptr<hololink::Hololink::Sequencer>`. Direct delegation for every V1
  method.
- `RoceDataChannelV1` (`roce_data_channel_default.hpp`) — wraps
  `std::shared_ptr<hololink::DataChannel>` plus a copy of the `EnumerationMetadata` it
  represents. `configure(...)` calls legacy `authenticate(qp_number, rkey)` then
  `configure_roce(frame_memory, frame_size, page_size, pages, /*local_data_port=*/0)`.
  The V1 instance-id hooks compose `serial=<n>;data_plane=<n>;kind=frame_end` and
  `serial=<n>` so the matching sequencer + parent lookups resolve.

`I2cLockNamedV1` lives next to `HololinkV1` in the same header. NamedLock has no native
`try_lock`, so this wrapper's `try_lock` falls back to a blocking `lock()` and returns
`true` — adequate for V1's BasicLockable contract; callers that genuinely need
non-blocking acquisition use the `std::mutex`-backed `I2cLockImplV1` from Phase 4
instead.

**Out of scope this phase.** No new tests landed in this slice (this phase's end-to-end
proof is the Phase 8 hardware integration). The deferred Phase 5 `HsbLiteV1` supplement
and the `hololink_module.hsb_lite` per-module Python sub-package are still pending;
they're what wires these wrappers into a real publication chain. The bridging hook for
`I2cV1::encode_i2c_request` lands alongside the supplement integration.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD"
cmake --build "$BUILD" --target hololink_module
```

(Header-only; the `hololink::module_core` static archive that existed since Phase 4
picks them up automatically when consumed by a module .so via `add_hololink_module()`.)

## Phase 5 extension — HSB-Lite supplement + Python sub-package

**Implemented.** The deferred concrete supplement and its per-module Python sub-package
land alongside the module/core wrappers from the previous extension. Specifically:

- `module/hsb_lite/hsb_lite_impl.hpp` — `HsbLiteV1`, the concrete `HsbLiteInterfaceV1`
  supplement. Holds a `shared_ptr<LegacyHololinkAccess>` (the same legacy class
  `HololinkV1` wraps) and implements `setup_clock(profile)` by delegating to the legacy
  `Hololink::setup_clock`, which drives the on-board Renesas Bajoran Lite TS1 clock
  generator over I2C.
- `module/hsb_lite/hsb_lite_enumeration.cpp` — the enumeration override now constructs
  the per-board publication chain. After stamping `module_name=hsb_lite` /
  `compat_id=2603` and backfilling `control_port=8192` (matching the legacy enumerator's
  default), if `peer_ip` and `serial_number` are present it builds
  `LegacyHololinkAccess` once per serial, wraps it in `HololinkV1`, constructs
  `HsbLiteV1` over the same legacy class, and publishes both under
  `serial=<serial_number>` via the same Publisher the module already uses for the
  `FrameMetadataInterfaceV1` singleton. A per-supplement `unordered_map<string, Board>`
  keys the cache by serial — repeated enumerate calls for the same serial reuse the
  cached pair.
- `module/hsb_lite/include/hololink/module/hsb_lite/hsb_lite.hpp` —
  `HsbLiteInterfaceV1::get_hsb_lite` is now an inline header function (was a separate
  `.cpp`), so the per-module Python pybind extension can call it without linking the
  module's `.so`. The standalone `hsb_lite_get_hsb_lite.cpp` is removed.
- `module/hsb_lite/python/hsb_lite_py.cpp` + `__init__.py` — per-module pybind extension
  `_hololink_module_hsb_lite` wrapping `HsbLiteInterface` (the unversioned alias) with
  the static `get_hsb_lite(module, metadata, allow_null)` factory and the
  `setup_clock(clock_profile)` method. The extension first imports
  `hololink_module._hololink_py_module` so the V1 base types it depends on (`Module`,
  `EnumerationMetadata`) are registered with pybind11. The `__init__.py` exposes both
  `HsbLiteInterface` and the pinned `HsbLiteInterfaceV1` alias.
- `module/hsb_lite/CMakeLists.txt` — drops the deleted `hsb_lite_get_hsb_lite.cpp`, adds
  the `pybind11_add_module(_hololink_module_hsb_lite ...)` block staging the `.so` +
  `__init__.py` together under `${BUILD}/python/hololink_module/hsb_lite/` so
  `PYTHONPATH=${BUILD}/python python3 -c "import hololink_module.hsb_lite"` works
  directly out of the build tree.

The supplement is scoped to a single `HololinkInterface` per the per-board convention
(one supplement per HSB-Lite serial). Per-channel `RoceDataChannelV1` publication and
per-(bus, address) `I2cV1` publication remain deferred — Phase 8's hardware integration
is what wires them in.

**Out of scope this phase.** Per-channel RoCE wrappers and the IMX274 I2C path (both
still pending for Phase 8). The existing `tests/hololink_module_hsb_lite_test` gtest
still proves the metadata-stamping path; no new tests for this slice (the end-to-end
proof is Phase 8). The bridging hook for `I2cV1::encode_i2c_request` lands alongside
Phase 8 integration.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD" -DHOLOLINK_BUILD_PYTHON=ON
cmake --build "$BUILD" --target hsb_lite
cmake --build "$BUILD" --target _hololink_module_hsb_lite
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" -R hololink_module_hsb_lite_test --output-on-failure
```

## Phase 5 extension — `hsb_lite_2510` per-compat-id module + naming convention

**Implemented.** A second HSB-Lite module pinned to compat-id `2510` lands under
`hololink_module/module/hsb_lite_2510/`. It shares the FPGA UUID with `module/hsb_lite/`
and the public `HsbLiteInterfaceV1` supplement type (no second Python sub-package —
applications keep importing `hololink_module.hsb_lite`); the two `.so` files are
distinguished by their filenames. `module/hsb_lite/` ships as the compat-suffixed
`hololink_<UUID>_2603.so`; `module/hsb_lite_2510/` ships as the bare
`hololink_<UUID>.so` (via `add_hololink_module(... NO_COMPAT_SUFFIX ...)`), which is the
Adapter loader's no-compat-id fallback. Boards reporting `compat_id=0x2603` over bootp
load `hololink_<UUID>_2603.so`; everything else — `compat_id=0x2510`, any other
compat-id without a dedicated `.so`, or no compat_id at all — falls through to
`hololink_<UUID>.so` and lands on the 2510 module. The new module is a near-clone of
`module/hsb_lite/` differing only in the fallback `DEFAULT_COMPAT_ID=0x2510` constant in
its enumeration override.

Compat-id values across the module follow a single convention. The **numeric form**
(stored in `EnumerationMetadata["compat_id"]` as an `int64`, parsed from the
little-endian uint16 on the wire) is the FPGA's 16-bit IP version (e.g. `0x2603`,
`0x2510`); the **string form** (used in the CMake `COMPAT` / `DEFAULT_COMPAT` property,
in `.so` filenames, and in human-facing logs) is the 4-digit lowercase hex rendering of
that number (`"2603"`, `"2510"`). `Adapter::load_module_for` formats `compat_id` as
`%04x` when composing the filename it dlopens; examples and tests do the same when
constructing the path to pass to `Adapter::load_module` directly.

Alongside the new module, `add_hololink_module()` resolves the per-module `.so` filename
in this order:

1. The `NO_COMPAT_SUFFIX` option, if supplied → emits the bare `hololink_<UUID>.so`. The
   resulting `.so` is the Adapter loader's catch-all when enumeration metadata's
   `compat_id` has no dedicated compat-suffixed `.so`. Used by `module/hsb_lite_2510/`.
1. The `COMPAT` argument, if supplied → emits `hololink_<UUID>_<compat>.so`. Used by the
   bootp-stub test module (`COMPAT 2603`).
1. The `DEFAULT_COMPAT` target property on `hololink::module_core` (currently `2603`,
   set by `module/core/CMakeLists.txt`). This is what `module/hsb_lite/` consumes
   implicitly — it ships no `COMPAT` argument and inherits `2603` from
   `hololink::module_core`.

`NO_COMPAT_SUFFIX` and `COMPAT` are mutually exclusive. Specifying neither (and having
no `DEFAULT_COMPAT` property on `hololink::module_core`) is a configure error.

To insulate applications from the filename convention entirely, `Adapter` gains a public
`get_module(const EnumerationMetadata&)` (Python: `adapter.get_module(metadata)`) that
reads `fpga_uuid` + `compat_id` from the metadata and returns the same cached `Module`
the loader resolved during `enumerate()`. Applications no longer compose the `.so` path
themselves — both C++ examples (`examples/module_imx274_player.cpp`,
`examples/module_quad_imx274_player.cpp`) and the Python example
(`examples/module_quad_imx274_player.py`) now call `adapter.get_module(metadata)`
directly.

Existing tests are updated accordingly:

- `tests/hololink_module_hsb_lite_test.cpp` passes `compat_id` explicitly on every
  `enumerate()` call; the two enumerate-path tests now cover both compat-id `2603` (load
  `module/hsb_lite/`) and `2510` (load `module/hsb_lite_2510/`).
- `tests/hololink_module_bootp_e2e_test.cpp` updates the hardcoded stub `.so` path to
  the new `_2603.so` suffix.
- `tests/test_hololink_module_python.py` passes `metadata["compat_id"] = 0x2603` before
  driving `Adapter.enumerate`.

**Out of scope this phase.** Per-compat-id behavioral divergence inside
`module/hsb_lite_2510/` (addresses, clock profile, supplement implementation) — the
slice landed today is structural only, with the same enumeration logic as
`module/hsb_lite/`. Real behavior deltas land when 2510 hardware verification surfaces
the differences.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD" -DHOLOLINK_BUILD_PYTHON=ON
cmake --build "$BUILD" --target hsb_lite
cmake --build "$BUILD" --target hsb_lite_2510
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" -R hololink_module_hsb_lite_test --output-on-failure
ctest --test-dir "$BUILD" -R hololink_module_bootp_e2e_test --output-on-failure
ctest --test-dir "$BUILD" -R hololink_module_python_smoke_test --output-on-failure
```

## Phase 8 — module_imx274_player.py example (structural)

**Implemented.** `examples/module_imx274_player.py` is the first end-to-end example
driving the rework. Topology: 2 HSB-Lite boards × 2 IMX274 cameras = 4 RoCE data
channels feeding a single Holoscan pipeline (`RoceReceiverOp` → `CsiToBayerOp` →
`ImageProcessorOp` → `BayerDemosaicOp` → `HolovizOp` arranged as a 2×2 visualization
grid). All control flow goes through the V1 surface:

- `hololink_module.Adapter.start_bootp_listener()` — opens the bootp UDP socket on the
  host `ReactorV1`. The four HSB-Lite data planes announce themselves; each arriving
  packet drives the supplement's `update_metadata` override. The example polls
  `Adapter.find_channel(peer_ip)` for each expected IP until all four resolve (default
  30s timeout via `--discovery-timeout`).
- `Adapter.load_module(path/hololink_<UUID>.so)` once — handles the Module that
  `FrameMetadataInterface.get_service`,
  `HololinkInterface.get_service(module, "serial=<n>")`, and
  `HsbLiteInterface.get_hsb_lite(module, metadata)` go through.
- `HsbLiteInterface.setup_clock(...)` per board, fed the same
  `hololink.renesas_bajoran_lite_ts1.device_configuration()` profile the legacy driver
  uses.
- `HololinkInterface.get_roce_data_channel(metadata)` per channel returns the V1 channel
  handed to `RoceReceiverOp`'s `channel` parameter.
- `hololink_module.sensors.imx274.Imx274Cam(hololink_v1, i2c_bus=1, expander_configuration=channel)`
  — one IMX274 driver per camera, calling `cam.configure(IMX274_MODE_1920X1080_60FPS)`
  (mode 1, 1080p RAW10) so the four-channel aggregate fits the data-plane bandwidth
  budget. The 4K modes over four channels exceed what the link can sustain.
- Default IPs: board 1 at `192.168.0.200/.201`, board 2 at `192.168.0.202/.203` — each
  pair is one HSB-Lite's two RoCE data channels.

A C++ sibling lives at `examples/module_imx274_player.cpp`. It mirrors the Python
example one-for-one: same default IPs, same camera mode, same lifecycle, same Holoscan
pipeline. Camera control goes through a V1-native C++ IMX274 driver under
`hololink_module/host/sensors/imx274/` (`Imx274Cam`, `LII2CExpander`) that takes
`shared_ptr<HololinkInterfaceV1>` directly. The driver re-uses the legacy IMX274
register tables from `src/hololink/sensors/camera/imx274/imx274_mode.hpp` (the data is
byte-for-byte identical to the Python version and the module rework's policy allows
reading legacy headers, just not extending or rewriting them). The CMake target is
`hololink::module::sensors::imx274`.

`examples/CMakeLists.txt` appends `module_imx274_player.py` to `EXAMPLE_INSTALL_FILES`
so it ships next to the other example scripts.

The HSB-Lite supplement now publishes the full per-board + per-channel V1 service chain
the example exercises:

- Per first-seen serial: `HololinkV1` + `HsbLiteV1` (as before), plus `I2cV1` for
  `(bus=1, address=0x1A)` (IMX274) and `(bus=1, address=0x70)` (LI I2C expander) so the
  IMX274 driver's `Imx274Cam` constructor resolves its I2C handles via
  `HololinkInterface.get_i2c(...)`.
- Per first-seen `(serial, channel)`: backfills HSB-Lite-specific channel addresses
  derived from the channel index (`vp_mask`, `sif_address`, `vp_address`, `hif_address`,
  `sensor`, `data_plane`, `frame_end_event`), converts the V1 `EnumerationMetadata` to a
  legacy `hololink::Metadata`, constructs a legacy `hololink::DataChannel` whose
  `create_hololink` lambda returns the cached `LegacyHololinkAccess`, wraps it in
  `RoceDataChannelV1` and publishes under `serial=<n>;data_plane=<n>`. The legacy
  `frame_end_sequencer()` from the data channel is wrapped in `SequencerV1` and
  published under `serial=<n>;data_plane=<n>;kind=frame_end`.

Repeat enumerate calls reuse the cached entries — the publication chain runs exactly
once per identifier.

The `add_hololink_module()` CMake helper stages every `hololink_<UUID>.so` into a
canonical `${BUILD}/lib/hololink/modules/` directory so the example's `--module-dir`
argument points at one consistent location regardless of which module subdirectory
contributed the source.

A ctest entry `module_imx274_player_test` is always registered when the build includes
Python + operators (`HOLOLINK_BUILD_PYTHON=ON` + `HOLOLINK_MODULE_BUILD_OPERATORS=ON`,
both default ON when their respective dependencies are available). The test SKIPS by
default and runs only when the runtime opt-in env var `HOLOLINK_TEST_IMX274=1` is set —
its command is wrapped to exit 77 (the autotools "skip" code) when the env var is
absent, and `SKIP_RETURN_CODE 77` makes ctest report SKIPPED. The test is also tagged
with `LABELS imx274` so `ctest -L imx274` filters in just the hardware-dependent slice.

**Hardware.** Real HSB-Lite hardware (2 boards × 2 IMX274 cameras), no HSB emulator (the
emulator does not model IMX274 on the I2C bus), Holoscan SDK, CUDA toolkit, RoCE-capable
IB device(s). The example accepts `--ibv-name` multiple times so each receiver can be
pinned to a separate IB device.

**Run command (manual).**

```bash
PYTHONPATH=$BUILD/python python3 examples/module_imx274_player.py \
    --hololink-board1 192.168.0.200 \
    --hololink-board1-channel-2 192.168.0.201 \
    --hololink-board2 192.168.0.202 \
    --hololink-board2-channel-2 192.168.0.203 \
    --module-dir $BUILD/lib/hololink/modules \
    --frame-limit=10 --headless
```

**Run command (ctest, hardware enabled at runtime).**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD"
cmake --build "$BUILD"
HOLOLINK_TEST_IMX274=1 ctest --test-dir "$BUILD" -R module_imx274_player_test --output-on-failure
```

Without `HOLOLINK_TEST_IMX274=1`, the test reports SKIPPED and ctest exits clean — the
same single build serves both hardware and non-hardware runners.

## Phase 6 extension — RoCE receiver thread + Holoscan emit

**Implemented.** The deferred ibverbs receiver thread that Phase 6 promised now wires
in. Rather than reimplement the QP/MR/CQ machinery, the V1 `RoceReceiverOp` delegates to
the existing tree's `hololink::operators::RoceReceiver`, which already handles device +
memory region setup, QP transitions, receive WR posting, completion-queue polling,
frame-end PSN matching, and the host-staging copy of the per-frame metadata block. The
new operator parameters are: `peer_ip` (the HSB sender IP the QP path authenticates
against), `queue_size` (in-flight WR depth), and `metadata_offset` (in-page offset of
the 48-byte EOF metadata block).

`start()` allocates a `hololink::ReceiverMemoryDescriptor` of `pages × page_size` (GPU
memory tied to `frame_context`), constructs a `RoceReceiver` over the buffer, calls
`RoceReceiver::start()` to bring the QP up, then calls
`channel_->configure(metadata, qp_number, rkey, frame_memory, frame_size, page_size, pages)`
with the `qp_number` / `rkey` / `external_frame_memory` the receiver came up with. A
monitor thread is spawned to drive `RoceReceiver::blocking_monitor()` — which polls the
CQ and signals frames to the consumer side.

`compute()` calls `RoceReceiver::get_next_frame(timeout=1000ms, ...)`. On a frame:

- Synchronously copies the 48-byte EOF block from device memory to a stack buffer via
  `cuMemcpyDtoH` and feeds it to `frame_metadata_->decode(...)` so the V1
  `FrameMetadataInterfaceV1` owns the layout (the legacy receiver also decodes
  internally for its own bookkeeping; this redundant copy keeps modules in charge of
  metadata format).
- Wraps the just-received frame buffer in a `nvidia::gxf::Tensor` (uint8 vector of
  `frame_size` bytes, `MemoryStorageType::kDevice`, `wrapMemory` over the existing CUDA
  pointer) inside a freshly-created `gxf::Entity`. The `wrapMemory` release callback
  captures a copy of the `frame_buffer_` `shared_ptr` so the device buffer outlives any
  in-flight tensor even after `stop()` drops the operator's own reference (see the
  use-after-free fix below).
- Stamps the operator's metadata map with the V1 `FrameMetadata` fields by name
  (`flags`, `psn`, `crc`, `frame_number`, `timestamp_s` / `_ns`, `bytes_written`,
  `metadata_s` / `_ns`) so downstream operators read by key.
- Emits the entity over `"output"`. The shape matches what `BaseReceiverOp` from the
  legacy tree emits, so it drops into the existing CsiToBayer → ImageProcessor →
  BayerDemosaic → HolovizOp pipeline.

`stop()` calls `channel_->unconfigure()`, `RoceReceiver::close()`, and joins the monitor
thread.

`hololink_module/host/operators/CMakeLists.txt` adds private `target_link_libraries` on
`hololink::core` (for `ReceiverMemoryDescriptor`) and
`hololink::operators::roce_receiver` (for the receiver class). The pybind extension's
contract is unchanged; only the operator's runtime body grew.

**Out of scope this phase.** The C++ gtest from Phase 6 still constructs the operator
with mock V1 service handles in a `holoscan::Application` and verifies `compose_graph()`
succeeds; it does not run the operator's actual `start()` / `compute()` / `stop()`
lifecycle (that needs a real ibverbs device and HSB peer). End-to-end exercise lands at
Phase 8.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD" -DHOLOLINK_MODULE_BUILD_OPERATORS=ON
cmake --build "$BUILD" --target hololink_operators
cmake --build "$BUILD" --target _hololink_module_operators
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" -R hololink_module_roce_receiver_op_test --output-on-failure
ctest --test-dir "$BUILD" -R hololink_module_operators_python_smoke_test --output-on-failure
```

## Phase 6 extension — `RoceReceiverV1` abstraction + per-supplement override

**Implemented.** The Phase 6 operator used to construct
`hololink::operators::RoceReceiver` directly from `RoceReceiverOp::start()`. That left
no override seam for a per-FPGA-revision behavior diff. This change wraps the legacy
receiver behind a new `RoceReceiverInterfaceV1` service so per-board overrides happen at
the standard service-locator level.

- `host/include/hololink/module/roce_receiver.hpp` — new abstract
  `RoceReceiverInterfaceV1` (alias `RoceReceiverV1`), a
  `ServiceLocatable<RoceReceiverInterfaceV1>` with `type_id = "roce_receiver.v1"`. Its
  `start(...)` takes the ten runtime parameters individually (ibv_name, ibv_port,
  cu_buffer, cu_buffer_size, cu_frame_size, cu_page_size, pages, metadata_offset,
  peer_ip, queue_size) rather than via a config struct so the compiler enforces each
  one. The header also defines `RoceReceiverFrameInfoV1`, the per-frame output struct
  mirroring the subset of the legacy `RoceReceiverMetadata` the operator consumes
  (`frame_memory`, `metadata_memory`, frame counters / timestamps, drop counter).
- `host/include/hololink/module/roce_data_channel.hpp` —
  `RoceDataChannelInterfaceV1::configure` shrinks from seven args to two:
  `configure(metadata, shared_ptr<RoceReceiverInterfaceV1>)`. The channel reads
  qp_number / rkey / external_frame_memory / frame_size / page_size / pages off the
  passed-in receiver.
- `module/core/roce_receiver_default.hpp` — new `RoceReceiverV1` wraps a
  (initially-null) `shared_ptr<hololink::operators::RoceReceiver>`. `start(...)`
  constructs the legacy receiver via a protected virtual `make_receiver(...)` hook and
  brings the QP up; every V1 method delegates. `get_next_frame` translates the legacy
  struct into `RoceReceiverFrameInfoV1` field-for-field.
- `module/core/roce_data_channel_default.hpp` — `RoceDataChannelV1::configure` rewritten
  to the 2-arg form: reads the QP / frame-memory / frame-layout values off the receiver
  and feeds them to `DataChannel::configure_roce`.
- `module/core/CMakeLists.txt` — `hololink::module_core` now PUBLIC-links
  `hololink::operators::roce_receiver` so every per-board module .so absorbs the legacy
  receiver class through its private copy of the wrapper.
- `module/hsb_lite/hsb_lite_enumeration.cpp` — publishes one `RoceReceiverV1` per
  `(serial, data_plane)` under the same `"serial=…;data_plane=…"` instance_id as the
  matching `RoceDataChannelV1`. The shell is constructed at enumerate time; the
  underlying ibverbs receiver comes up inside `start(...)`.
- `module/hsb_lite_2510/hsb_lite_2510_roce_receiver.hpp` — new
  `HsbLite2510RoceReceiverV1` (subclasses `RoceReceiverV1`, overrides `make_receiver` to
  construct a `HsbLite2510LegacyRoceReceiver`) and `HsbLite2510LegacyRoceReceiver`
  (subclasses `hololink::operators::RoceReceiver`, currently a marker — 2510-specific
  overrides of `copy_metadata_to_host` / `get_frame_metadata` land here when hardware
  verification surfaces what differs).
- `src/hololink/core/data_channel.hpp` — `hololink::DataChannel::configure_roce` is now
  `virtual` and the class gains a `virtual ~DataChannel() = default` so subclasses
  destruct properly through a base pointer. This is the override seam the 2510 data
  channel hooks into; it mirrors how the legacy `RoceReceiver` already declares its hook
  methods `virtual`.
- `module/hsb_lite_2510/hsb_lite_2510_data_channel.hpp` — new `HsbLite2510DataChannel`
  subclass of `hololink::DataChannel`, currently a marker. Concrete `configure_roce`
  override (and any other channel-side behavior the 2510 FPGA requires) lands here when
  hardware verification surfaces the diff.
- `module/hsb_lite_2510/hsb_lite_enumeration.cpp` — constructs `HsbLite2510DataChannel`
  (instead of the bare `hololink::DataChannel`) and wraps it in the default
  `RoceDataChannelV1`. The V1 impl's `backing_->configure_roce(...)` call now
  virtual-dispatches through the 2510 subclass. Also publishes the
  `HsbLite2510RoceReceiverV1` under the same `(serial, data_plane)` instance_id the
  default would have used, so `RoceReceiverV1::get_service` resolves to the override
  automatically when the loader picked `module/hsb_lite_2510/`.
- `host/operators/roce_receiver_op.{hpp,cpp}` — operator no longer includes
  `hololink/operators/roce_receiver.hpp`. `start()` looks up the receiver via
  `RoceReceiverInterfaceV1::get_service(channel_->module(), "serial=<n>;data_plane=<n>")`,
  calls `receiver_->start(ibv_name, ibv_port, cu_buffer, …, queue_size)` with the ten
  parameters individually, then calls `channel_->configure(metadata, receiver_)`. The
  monitor thread runs `receiver_->blocking_monitor()`; `compute()` reads
  `frame_info.frame_memory` / `frame_info.metadata_memory` from the new V1 struct.
- `tests/hololink_module_roce_receiver_op_test.cpp` — the existing `MockRoceDataChannel`
  is updated to the new `configure(metadata, receiver)` signature and now captures the
  passed-in `shared_ptr<RoceReceiverInterfaceV1>`. The test still only asserts
  `app->op() != nullptr` after compose; no receiver mock is needed because the test
  never runs `start()`.

**Out of scope this phase.** Behavioral content of the 2510 overrides — the
`HsbLite2510LegacyRoceReceiver` and `HsbLite2510DataChannel` subclasses are both markers
today, with no virtual overrides. Concrete 2510 behavior (RoceReceiver's
`copy_metadata_to_host` / `get_frame_metadata`, DataChannel's `configure_roce`, or
whatever the FPGA actually requires) lands when hardware verification surfaces the diff.
Python bindings for `RoceReceiverV1` itself are also out of scope — the operator keeps
the same pybind argument list, so example code is unchanged.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD" -DHOLOLINK_MODULE_BUILD_OPERATORS=ON
cmake --build "$BUILD" --target hsb_lite
cmake --build "$BUILD" --target hsb_lite_2510
cmake --build "$BUILD" --target hololink_operators
cmake --build "$BUILD" --target hololink_module_roce_receiver_op_test
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" -R hololink_module_roce_receiver_op_test --output-on-failure
ctest --test-dir "$BUILD" -R hololink_module_hsb_lite_test --output-on-failure
```

## Phase 3 extension — bootp listener

**Implemented.** The bootp UDP socket Phase 3 originally deferred now hangs off the host
`ReactorV1` so the Adapter discovers boards on its own:

- `Adapter::start_bootp_listener(port=12267)` opens an `AF_INET` / `SOCK_DGRAM` socket
  via `hololink::Enumerator::configure_socket`, registers an `FdCallback` with the host
  `ReactorV1` (`POLLIN`), and on each event calls
  `hololink::Enumerator::handle_bootp_fd(fd)` to extract the legacy `Metadata`
  - raw packet bytes. The legacy metadata converts to V1 `EnumerationMetadata`
    member-wise (same map shape) and feeds through
    `Adapter::enumerate(metadata, raw_packet, raw_packet_len)`. Modules that want to
    read vendor-specific bytes get them through the existing V1
    `EnumerationInterfaceV1::update_metadata(metadata, raw_packet, raw_packet_len)`
    contract.
- `Adapter::enumerate` gained a raw-packet overload; the original `enumerate(metadata)`
  form forwards to it with `nullptr` / `0`.
- `Adapter::stop_bootp_listener()` removes the fd callback and closes the socket. The
  destructor calls it for clean shutdown.
- The pybind binding exposes both methods so applications use
  `hololink_module.Adapter.get_adapter().start_bootp_listener(port)` instead of (or in
  addition to) manual `enumerate(metadata)` calls.

Two gtests cover the listener:

- `hololink_module_bootp_listener_test` exercises start/stop lifecycle (idempotency,
  port cleanup) using `start_bootp_listener(0)` so the kernel assigns an ephemeral port
  — sidesteps any collision with whatever else might happen to be bound on the test box.
- `hololink_module_bootp_e2e_test` is the end-to-end exercise. It hand-crafts a bootp v2
  packet (~277 bytes — fixed BOOTP header + 16-byte hardware address + 64-byte
  server_hostname + 128-byte boot_filename + 8-byte vendor section (`0xE0` tag, `'NVDA'`
  id) + 33-byte v2 fields (compat_id, fpga_uuid, transmitted packet count, 7-byte
  serial, hsb_ip_version, fpga_crc)) and sends it via `sendto(127.0.0.1, port)`. A stub
  module (`hololink_module_bootp_stub_module.cpp`) publishes both an
  `EnumerationInterfaceV1` (records what `update_metadata` saw) and a test-only
  `BootpCaptureV1` service the host queries to verify the parsed UUID, serial,
  compat_id, data_plane, peer_ip, and the round-tripped raw packet bytes.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD"
cmake --build "$BUILD" --target hololink_module_bootp_listener_test
cmake --build "$BUILD" --target hololink_module_bootp_e2e_test
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" -R hololink_module_bootp_listener_test --output-on-failure
ctest --test-dir "$BUILD" -R hololink_module_bootp_e2e_test --output-on-failure
```

## Phase 5b extension — V1 service-interface bindings

**Implemented.** The `_hololink_py_module` pybind extension grew bindings for every
abstract V1 service interface the IMX274 driver and per-module Python sub-packages will
consume: `I2cLock`, `I2cInterface`, `SequencerInterface`, `RoceDataChannelInterface`,
`HololinkInterface`, `EnumerationInterface`, `LoggingInterface`, plus a consumer-only
`Reactor` class, the `LogLevel` enum, and the opaque `AlarmEntry` token type. Each
abstract interface (other than `Reactor`) gets a pybind trampoline so Python can
subclass it: pure-virtual methods dispatch through `PYBIND11_OVERRIDE_PURE`, and where
the C++ ABI uses in/out vector parameters or other shapes that don't translate directly,
the trampoline reshapes the call (e.g. `I2cInterface.i2c_transaction` is exposed as
`(peripheral_address, write_bytes, read_byte_count) -> bytes` on both sides, with the
trampoline copying Python's bytes return into the C++ `read_bytes` vector). The
`__init__.py` exports both the unversioned aliases (`HololinkInterface`, `I2cInterface`,
…) and the pinned V1 names (`HololinkInterfaceV1`, `I2cInterfaceV1`, …) per the project
convention.

`Reactor` is bound for consumers only — the host publishes the C++ `Reactor` impl and
Python only ever observes it; there is no public Python constructor and no trampoline.
`HololinkInterface.i2c_lock` is bound for callers (returns a Python-owned `I2cLock`) but
the trampoline rejects Python overrides — the V1 contract returns
`std::unique_ptr<I2cLock>` whose default-deleter ownership transfer cannot be bridged
cleanly from a Python-owned override return; real `HololinkInterface` wrappers implement
`i2c_lock` in C++.

A pytest under `tests/test_hololink_module_v1_bindings_python.py` verifies that for each
interface the unversioned alias is `is`-identical to the V1 alias, that a Python
subclass with trivial pure-virtual overrides constructs cleanly, and that calling a
representative method dispatches through to the Python override.

The IMX274 sensor driver was tightened to match the V1 surface — its `set_register` no
longer plumbs a `timeout=` kwarg (V1 `i2c_transaction` doesn't have one).

**Out of scope this phase.** The deferred Phase 5 concrete implementations (`HololinkV1`
/ `RoceDataChannelV1` / `I2cV1` / `I2cLockImplV1` / `SequencerV1`, the `HsbLiteV1`
supplement, the `hololink_module.hsb_lite` per-module Python sub-package), the deferred
Phase 6 ibverbs receiver thread, and real-hardware verification of the IMX274 driver —
each unblocked by but separate from this binding slice.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD" -DHOLOLINK_BUILD_PYTHON=ON
cmake --build "$BUILD" --target _hololink_py_module
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" -R hololink_module_v1_bindings_python_test --output-on-failure
```

## Phase 6 — RoCE receiver operator (Holoscan-coupled, structural)

**Implemented.** A Holoscan-coupled `RoceReceiverOp` operator under
`hololink_module/host/operators/`. The class is a `holoscan::Operator` subclass that
takes a `shared_ptr<RoceDataChannelInterfaceV1>` plus a
`shared_ptr<FrameMetadataInterfaceV1>` and the supporting parameters (enumeration
metadata, CUDA frame context, frame size, page size, pages, ibv name + port). The
Holoscan lifecycle is wired:

- `setup()` declares every parameter on the `OperatorSpec`.
- `start()` calls
  `channel->configure(metadata, qp_number, rkey, frame_memory, frame_size, page_size, pages)`
  — the structural channel-V1 contract — and tracks the `configured_` flag.
- `stop()` calls `channel->unconfigure()` when the start path succeeded.

The `_hololink_module_operators.so` pybind extension wraps the operator under the
per-module `hololink_module.operators` Python sub-package. The package's `__init__.py`
exports `RoceReceiverOp` under the unversioned alias name. The pybind entry-point's
first act is `py::module_::import("hololink_module._hololink_py_module")` so the V1
service types the operator's parameters reference are registered before introspection
runs.

The CMake plumbing under `hololink_module/host/operators/CMakeLists.txt` is gated by the
top-level `HOLOLINK_MODULE_BUILD_OPERATORS` option (defaults ON, matching the parent
project's Holoscan-required defaults). It finds Holoscan via
`find_package(holoscan 4.0)`, builds `hololink::operators` static archive, and (when
`HOLOLINK_BUILD_PYTHON=ON`) the operators pybind extension. Build environments without
Holoscan configure with `-DHOLOLINK_MODULE_BUILD_OPERATORS=OFF` and skip this directory.

A C++ gtest under `tests/` constructs the operator via a `MockRoceDataChannel` +
`StubFrameMetadata` test fixture inside a real `holoscan::Application` and verifies
`compose_graph()` yields a non-null operator with the V1 service handles bound through
the args list. Per the project's testing rule, the mock classes are ordinary subclasses
of the public V1 interfaces — no test-only hooks were added to the framework. A Python
smoke test asserts the `hololink_module.operators` sub-package imports and the
`RoceReceiverOp` symbol resolves through it. Asserting that `channel->configure(...)` is
called when the operator runs needs `Fragment::run_async()` (driving the full Holoscan
lifecycle through parameter binding); that's a follow-up that pairs with the ibverbs
receiver-thread integration where the run can actually deliver frames.

**Out of scope this phase.** The actual ibverbs receiver thread that allocates the host
frame buffer, registers it with the local IB device, creates / binds the QP (yielding
the real `qp_number` and `rkey`), posts receive work-requests, and consumes completion
events. The current `start()` calls `channel->configure(...)` with zero values for
`qp_number` / `rkey` / `frame_memory` since the ibverbs work hasn't run; `compute()` is
a placeholder that emits no output. The full data path — including the per-frame
`FrameMetadataInterfaceV1::decode` call and the downstream emit — lands together with
hardware integration in a follow-up so the receive loop can be exercised against real
RoCE traffic.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD" -DHOLOLINK_MODULE_BUILD_OPERATORS=ON
cmake --build "$BUILD" --target hololink_operators
cmake --build "$BUILD" --target _hololink_module_operators
cmake --build "$BUILD" --target hololink_module_roce_receiver_op_test
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" -R hololink_module_roce_receiver_op_test --output-on-failure
ctest --test-dir "$BUILD" -R hololink_module_operators_python_smoke_test --output-on-failure
```

## Phase 7 — IMX274 sensor driver (structural)

**Implemented.** A Python IMX274 driver bound to V1 module handles directly, under
`hololink_module/python/sensors/imx274/`. The driver mirrors the legacy
`hololink.sensors.imx274.dual_imx274.Imx274Cam` register-table flow but takes a
`HololinkInterfaceV1` as the constructor handle (no `DataChannel` indirection) and
fetches `I2cInterfaceV1` via `hololink.get_i2c(bus=..., address=...)`. The HSB-Lite I2C
expander (LII2CExpander) sits at its own bus address and gates the shared bus to one of
two cameras; the per-camera driver instance is constructed with
`expander_configuration=0` (OUTPUT_1) or `=1` (OUTPUT_2) and the expander write is
issued synchronously before each camera transaction under the per-board `I2cLockV1`. The
pure-Python module ships under the `hololink_module.sensors.imx274` sub-package —
`Imx274Cam`, `Imx274_Mode`, and `LII2CExpander` are exposed through the package
`__init__.py`. Pixel- and Bayer-format enums for module-side sensor drivers live
alongside as `hololink_module.sensors.csi`.

CMake plumbing under `hololink_module/python/CMakeLists.txt` (and the `sensors/` +
`sensors/imx274/` subtrees) stages the .py files into
`${BUILD}/python/hololink_module/sensors/imx274/` next to the existing
`_hololink_py_module` extension, so
`PYTHONPATH=${BUILD}/python python3 -c "import hololink_module.sensors.imx274"` works
directly out of the build tree.

A Mock-based pytest under `tests/test_hololink_module_imx274_python.py` stands in fakes
for `HololinkInterfaceV1` / `I2cInterfaceV1` / `I2cLockV1` and verifies the driver's
emitted bus traffic: the expander selects the right output mask (`OUTPUT_1` for
`expander_configuration=0`, `OUTPUT_2` for `=1`), each `set_register` call writes the
correct big-endian 16-bit register address followed by the value byte to
`CAM_I2C_ADDRESS`, `get_register` round-trips through the same expander gate, the
per-board lock is acquired and released around every transaction (including when the
underlying I2C call raises), and `set_mode` reports the right geometry + pixel format
for both 4K-RAW10 and 4K-RAW12.

**Out of scope this phase.** Real-hardware verification — driving an actual IMX274 on an
HSB-Lite carrier through clock setup, mode programming, and a streaming `start()` /
`stop()` round-trip. That work pairs with the deferred `HololinkV1` / `I2cV1` /
`I2cLockImplV1` / `SequencerV1` wrappers (Phase 5 follow-up) and the per-module
`hololink_module.hsb_lite` Python sub-package, since the driver needs module Python
bindings for those types before it can be instantiated against a live board. IMX274
verification will run on real hardware (no HSB emulator) once those pieces are in place.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD"
cmake --build "$BUILD" --target hololink_module
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" -R hololink_module_imx274_python_test --output-on-failure
```

## Phase 6 extension — `RoceReceiverOp` derives ibv_name / ibv_port from peer_ip

**Implemented.** `RoceReceiverOp` no longer takes `ibv_name` / `ibv_port` as constructor
arguments. `peer_ip` was already required to bring the QP up; the operator's `start()`
now calls `hololink::module::ibv_device_for_peer(peer_ip)` inline and feeds the resolved
`(ibv_name, ibv_port)` pair to `RoceReceiverInterfaceV1::start(...)`. Applications no
longer repeat the route-resolution choice the module already encapsulates.

- `host/operators/include/hololink/module/operators/roce_receiver_op.hpp` — drops the
  `ibv_name_` / `ibv_port_` `holoscan::Parameter` members.
- `host/operators/roce_receiver_op.cpp` — drops the matching `spec.param(...)`
  declarations and includes `hololink/module/ibv_device.hpp`; `start()` calls
  `ibv_device_for_peer(peer_ip_.get())` immediately before forwarding to
  `receiver_->start(...)`.
- `host/operators/python/operators_py.cpp` — removes `ibv_name` / `ibv_port` from the
  `PyRoceReceiverOp` constructor signature, the `holoscan::ArgList`, and the pybind
  `py::init<>` template + `_a` kwarg defaults.
- `examples/module_imx274_player.cpp` — drops the `--ibv-name` / `--ibv-port` CLI
  options, the `hololink::infiniband_devices()` fallback, the corresponding
  `HoloscanApplication` constructor params + members, and the
  `holoscan::Arg("ibv_name", …) / holoscan::Arg("ibv_port", …)` lines.
- `examples/module_quad_imx274_player.{cpp,py}` — drops the inline
  `ibv_device_for_peer(peer_ip)` call inside the per-channel compose loop and the
  corresponding `holoscan::Arg` / kwargs. The C++ variant also drops the now-unused
  `#include "hololink/module/ibv_device.hpp"`.
- `tests/hololink_module_roce_receiver_op_test.cpp` — drops the placeholder
  `Arg("ibv_name", "test")` / `Arg("ibv_port", 1)` lines from the compose-only smoke
  test; the test still asserts `op() != nullptr` after `compose_graph()`.

The `RoceReceiverInterfaceV1::start(...)` contract is unchanged — it still takes
`ibv_name` / `ibv_port` as the first two of its ten arguments, because the V1 receiver
service intentionally pins which device the QP is bound to. This change only relocates
*where the lookup happens*: the operator now does it once, internally, from `peer_ip`.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD" -DHOLOLINK_MODULE_BUILD_OPERATORS=ON
cmake --build "$BUILD" --target hololink_operators
cmake --build "$BUILD" --target _hololink_module_operators
cmake --build "$BUILD" --target hololink_module_roce_receiver_op_test
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" -R hololink_module_roce_receiver_op_test --output-on-failure
ctest --test-dir "$BUILD" -R hololink_module_operators_python_smoke_test --output-on-failure
```

## Phase 6 extension — `RoceReceiverOp` resolves channel + frame_metadata from metadata

**Implemented.** `RoceReceiverOp` no longer takes `channel` or `frame_metadata` as
constructor Args. The operator already accepted the `enumeration_metadata` that
identifies the supplement module; `start()` now uses it directly to look up the matching
V1 services from the loader-resolved module:

- `Adapter::get_adapter().get_module(metadata)` returns the supplement Module the bootp
  listener cached for this (fpga_uuid, compat_id) pair.
- `RoceDataChannelInterfaceV1::get_service(module, "serial=<n>;data_plane=<n>")` returns
  the per-data-plane channel — the same instance_id format the receiver uses, so the
  per-supplement subclass on `hsb_lite_2510` is picked up automatically.
- `FrameMetadataInterfaceV1::get_service(module)` returns the per-module singleton.

Applications now hand the op nothing more than the metadata; the data-channel and
frame-metadata services are an internal lookup. `enumeration_metadata` is the only
V1-shaped Arg the op accepts.

- `host/operators/include/hololink/module/operators/roce_receiver_op.hpp` — drops the
  `channel_` / `frame_metadata_` `holoscan::Parameter` members; adds plain
  `std::shared_ptr<…>` fields populated in `start()` and released in `stop()`.
- `host/operators/roce_receiver_op.cpp` — removes the matching `spec.param(...)`
  declarations, the YAML converter registrations for the two shared_ptr types, and the
  corresponding `HOLOLINK_MODULE_YAML_CONVERTER_UNSUPPORTED` macro expansions. Includes
  `hololink/module/adapter.hpp` and `module.hpp`. `start()` parses `serial_number` /
  `data_plane`, calls `Adapter::get_adapter().get_module(metadata)`, then fetches the
  channel, frame-metadata service, and receiver from that module. `stop()` resets both
  service handles alongside the existing teardown.
- `host/operators/python/operators_py.cpp` — drops `channel` and `frame_metadata` from
  the `PyRoceReceiverOp` constructor signature, the `holoscan::ArgList`, and the pybind
  `py::init<>` template + `_a` kwargs.
- `examples/module_imx274_player.cpp` — drops the `channel` / `frame_metadata_service`
  lookup from `main()`, the `HoloscanApplication` constructor params + member fields,
  and the `holoscan::Arg("channel", …)` / `holoscan::Arg("frame_metadata", …)` lines
  from the receiver `make_operator<>` call. The example also drops the now-unused
  `<hololink/module/frame_metadata.hpp>` and `<hololink/module/roce_data_channel.hpp>`
  includes.
- `examples/module_quad_imx274_player.{cpp,py}` — drops the per-data-plane `channels` +
  `frame_metadata_services` vectors / list (C++ and Python), the `HoloscanApplication`
  channel-list parameter, and the corresponding Args. The Python `HoloscanApplication`
  now takes `metadatas` (one per data plane) in place of the 4-tuple list.
- `tests/hololink_module_roce_receiver_op_test.cpp` — drops the `MockRoceDataChannel` +
  `StubFrameMetadata` fixtures (no longer reachable through the op's Args) and
  constructs with the minimal `enumeration_metadata`
  - frame_size + page_size + pages Args. Still asserts `op() != nullptr` after
    `compose_graph()`.

**Out of scope this phase.** Hardware-side verification of the full lookup chain —
exercising `start()` requires a loaded supplement module that publishes the channel +
receiver + frame-metadata services under the expected instance_ids. That path is covered
end-to-end by the integration test in Phase 8.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD" -DHOLOLINK_MODULE_BUILD_OPERATORS=ON
cmake --build "$BUILD" --target hololink_operators
cmake --build "$BUILD" --target _hololink_module_operators
cmake --build "$BUILD" --target hololink_module_roce_receiver_op_test
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" -R hololink_module_roce_receiver_op_test --output-on-failure
ctest --test-dir "$BUILD" -R hololink_module_operators_python_smoke_test --output-on-failure
```

## Phase 5 extension — `OscillatorInterfaceV1` + clock-rate cache on `HololinkImpl`

**Implemented.** New per-data-plane V1 service that lets the sensor driver drive the
on-board reference clock during sensor configuration — applications no longer call
`hsb_lite->setup_clock(...)` themselves.

- `host/include/hololink/module/oscillator.hpp` — `OscillatorInterfaceV1`
  (`type_id = "oscillator.v1"`, `ServiceLocatable<...>`) with
  `enable(uint64_t clocks_per_second)` returning `bool` plus `get_caps()` / `set_caps()`
  for implementation-defined tunable knobs. Instance_id matches the per-data-plane
  `"serial=<serial_number>;data_plane=<n>"` form so the oscillator is reachable
  alongside the channel and receiver.
- `module/hsb_lite/oscillator_impl.{hpp,cpp}` — `OscillatorV1` holding the per-board
  `shared_ptr<module_core::HololinkV1>` + data-plane index. `enable(...)` just delegates
  to `HololinkV1::enable_clock(...)`.
- `module/hsb_lite_2510/oscillator_impl.{hpp,cpp}` — `HsbLite2510OscillatorV1`, same
  shape (the 2510 may later diverge).
- `module/core/hololink_default.{hpp,cpp}` — new module-private method
  `HololinkV1::enable_clock(uint64_t)`. First call programs the clock generator via
  `setup_clock(hololink::renesas::DEVICE_CONFIGURATION)`, caches the rate, returns
  `true`. Subsequent same-rate calls are no-op `true`; different-rate calls return
  `false`. The constructor registers a private nested `Hololink::ResetController` via
  `backing_->on_reset(...)` so the cache is dropped on hololink reset; the next
  `enable_clock(rate)` re-programs from scratch. The cache lives on `HololinkImpl` so
  the two oscillators per board (one per data plane) coordinate through one shared
  state.
- Both modules' `HsbLiteEnumerationV1` publish one oscillator instance per data plane
  under `data_plane_instance_id`, constructed with the board's `HololinkV1`.
- `host/sensors/imx274/imx274_cam.{hpp,cpp}` and `python/sensors/imx274/imx274_cam.py` —
  `Imx274Cam` constructor takes a required `OscillatorInterfaceV1` after the
  `HololinkInterface`. `configure(mode)` calls `oscillator->enable(25'000'000)` first
  and throws / raises if `enable` returns false.
- `host/python/hololink_module_py.cpp` + `python/__init__.py` — `OscillatorInterface`
  bound for Python with `get_service` / `enable` / `get_caps` / `set_caps`;
  `OscillatorInterfaceV1` aliased alongside the unversioned name.
- `examples/module_imx274_player.cpp` + `examples/module_quad_imx274_player.{cpp,py}` —
  fetch the per-data-plane oscillator from the locator and pass it to `Imx274Cam`. The
  explicit `hsb_lite->setup_clock(...)` call is removed from every example's bring-up
  loop; the clock is now programmed inside `Imx274Cam::configure` via the oscillator.
  The `HsbLiteInterface` lookup and the corresponding `Channel` struct field are dropped
  from each example since clock setup was the only consumer.
- `tests/test_hololink_module_imx274_python.py` — the existing `Imx274Cam` fixtures pass
  a `MagicMock()` for the oscillator slot (the tests exercise register I/O paths that
  don't reach `configure()`).

**Out of scope this phase.** Behavioral content of `get_caps` / `set_caps` — both
modules return `{}` / `false` today. Validating the requested rate against the
hardware's actual capability is also deferred: `enable_clock(rate)` accepts whatever
rate the first caller asks for and programs the chip via `DEVICE_CONFIGURATION`, so
asking for a non-25-MHz rate first commits the chip to that label with the wrong
register sequence. The IMX274 driver only ever asks for 25 MHz; once a different sensor
lands, the impl will need to validate against the chip's actual range.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD" -DHOLOLINK_MODULE_BUILD_OPERATORS=ON
cmake --build "$BUILD" --target hsb_lite
cmake --build "$BUILD" --target hsb_lite_2510
cmake --build "$BUILD" --target hololink_module_sensors_imx274
cmake --build "$BUILD" --target _hololink_py_module
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" -R hololink_module_core_test --output-on-failure
ctest --test-dir "$BUILD" -R hololink_module_imx274_python_test --output-on-failure
```

## Phase 5 extension — metadata-only service lookups + `Imx274Cam(metadata)` + TOCTOU fix on module loader

**Implemented.** Ergonomic + safety follow-ups on top of the oscillator + clock-cache
work:

- `host/include/hololink/module/hololink.hpp` + `host/src/hololink.cpp` — new static
  `HololinkInterfaceV1::get_service(metadata, allow_null = false)`. Resolves the
  supplement module by calling `Adapter::get_adapter().get_module(metadata)` itself,
  builds the per-board `"serial=<serial_number>"` instance_id from
  `metadata.get<std::string>("serial_number")`, and delegates to the base
  `ServiceLocatable<HololinkInterfaceV1>::get_service`. Body is out-of-line so the
  header doesn't drag `adapter.hpp` into every consumer. The base
  `(module, instance_id, allow_null)` form is kept reachable through
  `using ServiceLocatable<…>::get_service;`.
- `host/include/hololink/module/oscillator.hpp` + `host/src/oscillator.cpp` — same
  pattern for `OscillatorInterfaceV1::get_service(metadata, allow_null = false)`. Builds
  the per-data-plane `"serial=<serial_number>;data_plane=<data_plane>"` instance_id.
- `host/sensors/imx274/imx274_cam.{hpp,cpp}` + `python/sensors/imx274/imx274_cam.py` —
  new `Imx274Cam(metadata, i2c_bus = DEFAULT_CAM_I2C_BUS)` constructor (Python
  dispatches on the first arg's type). Fetches HololinkInterface + OscillatorInterface
  from the metadata using the overloads above. `expander_configuration` is derived from
  metadata: prefers `metadata["expander_configuration"]` when set, falls back to
  `metadata["data_plane"]`.
- `host/sensors/imx274/imx274_cam.{hpp,cpp}` + `python/sensors/imx274/imx274_cam.py` —
  new static `Imx274Cam::use_expander_configuration(metadata, value)` (Python
  `@staticmethod`). Application code uses this to stamp an override into
  `metadata["expander_configuration"]` instead of mutating the field by string key.
- `host/python/hololink_module_py.cpp` — pybind bindings for the metadata-taking
  `get_service` overloads (no `module` parameter). The two overloads coexist on each
  class via pybind11 overload resolution.
- `host/include/hololink/module/adapter.hpp` + `host/src/adapter.cpp` —
  `Adapter::load_module_for` rewritten to fix a check-then-use TOCTOU. New private
  helper `Adapter::try_load_module(path)` collapses `std::filesystem::exists(path)` +
  `load_module(path)` into one `load_so` attempt; on `std::runtime_error` it stats the
  path and either returns an empty `shared_ptr` (file missing — caller falls back) or
  re-throws (real load failure: dlopen / ABI mismatch / init failure). `load_module_for`
  now walks the compat-suffixed and bare candidates via `try_load_module` and only
  throws "no module .so for fpga_uuid='…'" when both come back empty.
- `examples/module_imx274_player.cpp` + `examples/module_quad_imx274_player.{cpp,py}` —
  `module_handle` / `module` locals dropped; service lookups + `Imx274Cam` constructor
  take metadata directly. Single-board example stamps `--expander-configuration` via
  `Imx274Cam::use_expander_configuration(...)` only when the CLI flag was actually
  passed. Quad C++ + Python use the default fallback path (data_plane).

**Out of scope this phase.** Race window inside `try_load_module` (deletion between the
`load_so` throw and the post-failure `std::filesystem::exists` check) misclassifies a
real load error as ENOENT and falls back. The window is far narrower than the previous
exists-then-load pattern. `Imx274Cam::use_expander_configuration` is the only
application-facing helper today; other metadata fields stay supplement-internal.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD" -DHOLOLINK_MODULE_BUILD_OPERATORS=ON
cmake --build "$BUILD" --target hololink_module
cmake --build "$BUILD" --target _hololink_py_module
cmake --build "$BUILD" --target module_imx274_player
cmake --build "$BUILD" --target module_quad_imx274_player
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" -R hololink_module_framework_test --output-on-failure
ctest --test-dir "$BUILD" -R hololink_module_enumeration_test --output-on-failure
ctest --test-dir "$BUILD" -R hololink_module_imx274_python_test --output-on-failure
```

## Phase 3 extension — `Adapter::wait_for_channel` replaces the `enumerated_` cache + `find_channel`

**Implemented.** Replaced the unbounded `enumerated_` vector + `find_channel(peer_ip)`
lookup with a one-shot per-peer-IP slot map + condition variable + blocking
`wait_for_channel(peer_ip, timeout)` API. Eliminates the second source of truth that
duplicated the service locator and bounds the cache size to one entry per peer.

- `host/include/hololink/module/adapter.hpp` — `enumerated_` (vector) gone; replaced by
  `pending_by_peer_ip_` (unordered_map) + `enumeration_cv_`. Declares
  `EnumerationMetadata Adapter::wait_for_channel(const std::string& peer_ip, std::chrono::milliseconds timeout)`.
  Drops the `find_channel` declaration.
- `host/src/adapter.cpp` — `enumerate()` writes the (enriched-if-known-UUID) metadata
  into the `peer_ip` slot (overwriting any prior unconsumed entry) and notifies the cv.
  `wait_for_channel(peer_ip, timeout)` *first erases any cached slot for that peer*,
  then `cv.wait_for`s, consumes on return, and throws on timeout. The pre-erase means
  `wait_for_channel` always returns an announcement that arrived *after* the call
  started waiting — a stale cached one from minutes ago won't satisfy it. `find_channel`
  is gone.
- `host/python/hololink_module_py.cpp` — pybind for
  `wait_for_channel(peer_ip, timeout_s)` takes seconds as a Python `float`; the lambda
  converts to `std::chrono::milliseconds`. `find_channel` binding removed.
- `examples/module_imx274_player.cpp` — dropped the local `wait_for_channel` polling
  helper and the `<thread>` include; `main()` calls
  `adapter.wait_for_channel(hololink_ip, discovery_timeout)` directly.
- `examples/module_quad_imx274_player.cpp` — local `enumerate` helper now walks
  `device_ips` sequentially calling `adapter.wait_for_channel(ip, remaining)` against a
  cumulative deadline, so total wait time is bounded by `discovery_timeout` rather than
  `4 × discovery_timeout`. `<thread>` include dropped.
- `examples/module_quad_imx274_player.py` — same simplification on the Python side:
  `_enumerate` does a sequential `adapter.wait_for_channel(ip, remaining)` loop.
- `tests/hololink_module_enumeration_test.cpp` — `ManualEnumerateLoadsModuleAndEnriches`
  and `EnumerateStoresUnknownUuidUnenriched` spawn a helper thread that sleeps 50ms then
  calls `enumerate(...)` while the test thread is blocked in
  `wait_for_channel(peer_ip, 1s)`. A synchronous enumerate-then-wait would have the
  pre-erase wipe the just-posted entry. `FindChannelThrowsOnUnknownIp` becomes
  `WaitForChannelTimesOutOnUnknownIp` with a 50ms timeout.
- `tests/test_hololink_module_python.py` — same pattern: a `threading.Thread` enumerates
  after a 50ms sleep while the test thread blocks in
  `wait_for_channel(peer_ip, timeout_s=1.0)`.

**Out of scope this phase.** Multi-waiter is "first consumer wins" — the slot is
single-shot, no broadcast semantics. Callback-style `register_ip` / `register_all`
remain unimplemented; `wait_for_channel` is the only sync primitive today.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD"
cmake --build "$BUILD" --target hololink_module
cmake --build "$BUILD" --target _hololink_py_module
cmake --build "$BUILD" --target module_imx274_player
cmake --build "$BUILD" --target module_quad_imx274_player
cmake --build "$BUILD" --target hololink_module_enumeration_test
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" -R hololink_module_enumeration_test --output-on-failure
ctest --test-dir "$BUILD" -R hololink_module_python_test --output-on-failure
```

## Phase 3 extension — callback-based `register_ip` / `register_all`; `wait_for_channel` built on top

**Implemented.** Adapter gains the plan's callback-style enumeration delivery API
(`register_ip` / `register_all` / `unregister`), and `wait_for_channel` is rewritten to
build on `register_ip` with per-call state. The `pending_by_peer_ip_` map and the shared
`enumeration_cv_` are removed — no global cache lives in the Adapter for this anymore.

- `host/include/hololink/module/adapter.hpp` — abstract `EnumerationCallback` with a
  single pure-virtual `handle_metadata(EnumerationMetadata&)` plus
  `EnumerationCallbackHandle = std::shared_ptr<EnumerationCallback>`. Adapter declares
  `register_ip(peer_ip, callback)` / `register_all(callback)` returning the handle, and
  `unregister(handle)`. `pending_by_peer_ip_` / `enumeration_cv_` are replaced by
  `registrations_` (vector of handles) + `registrations_mutex_`. No public state or
  filter conditional in the header — the per-subclass filter behavior lives in
  adapter.cpp.
- `host/src/adapter.cpp` — anonymous namespace defines two `EnumerationCallback`
  subclasses: `PeerIpEnumerationCallback` (stores the wrapped function + a `peer_ip`;
  invokes the function only when `metadata["peer_ip"]` matches) and
  `AllPeersEnumerationCallback` (unconditionally invokes the wrapped function).
  `register_ip` / `register_all` construct the appropriate subclass. `enumerate()`
  snapshots `registrations_` under the mutex, then unlocks and calls `handle_metadata`
  on every snapshot entry — each registration decides whether to fire. The snapshot
  pattern lets callbacks re-enter the registration API and lets a callback unregister
  itself without invalidating the dispatch loop. `wait_for_channel(peer_ip, timeout)`
  builds a per-call `State { mutex, cv, optional<metadata> }` shared with a closure
  passed to `register_ip`: the closure stamps `state->received` on first fire and
  notifies; the caller `cv.wait_for`s on it, `unregister`s, then returns the metadata
  (or throws on timeout).
- `host/python/hololink_module_py.cpp` — pybind binds `EnumerationCallback` as the
  opaque `EnumerationCallbackHandle` class (no methods). Adapter gets `register_ip` /
  `register_all` / `unregister` bindings; `wait_for_channel` signature stays the same.

**Out of scope this phase.** Callback ordering is registration order — no
specificity-based prioritization between `register_ip` and `register_all`. Exceptions
thrown from a callback abort the dispatch loop for that announcement; applications that
want fault isolation between callbacks wrap their bodies in their own try / catch.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD"
cmake --build "$BUILD" --target hololink_module
cmake --build "$BUILD" --target _hololink_py_module
cmake --build "$BUILD" --target hololink_module_enumeration_test
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" -R hololink_module_enumeration_test --output-on-failure
ctest --test-dir "$BUILD" -R hololink_module_python_test --output-on-failure
```

## Phase 3 extension — `adapter_enumerator` tool + raw subscriber API

**Implemented.** A new `hololink_module/tools/` directory houses board-agnostic
command-line tools that ship with the framework, starting with `adapter_enumerator` — a
real-time bootp inspector. To support its `--raw` mode the Adapter gains a separate pair
of "raw" registration entry points that fire before any module .so is loaded.

- `host/include/hololink/module/adapter.hpp` — adds `register_raw_ip(peer_ip, callback)`
  and `register_raw_all(callback)` alongside the existing `register_ip` /
  `register_all`. Both return the same opaque `EnumerationCallbackHandle`;
  `unregister(handle)` removes from whichever list it was added to without the caller
  having to track which. A second private vector `raw_registrations_` lives next to
  `registrations_` under the same mutex.
- `host/src/adapter.cpp` — `enumerate()` now snapshots both lists at the top. Raw
  subscribers fire first; then, only when at least one post-enrichment subscriber is
  registered, the module-load + `update_metadata` path runs and fires the regular
  subscribers. A process built around `register_raw_*` only (e.g. the new tool in
  `--raw` mode) does no dlopen, so no module .so files need to be present on the host.
  `EnumerationCallback::handle_metadata` takes a `const EnumerationMetadata&` (and the
  four `register_*` entry points take `std::function<void(const EnumerationMetadata&)>`)
  so callbacks are observers, not mutators — the raw dispatch does not need a defensive
  copy to protect the enrichment path. As part of the same change, `load_module_for` now
  returns both the loaded `Module` and the absolute path it loaded from; `enumerate()`
  stamps that path into `metadata["module_filename"]` before invoking the module's
  `update_metadata`, fulfilling the plan's "the module still publishes the bootp
  metadata without `module_filename` or enrichment" contract — post-enrichment
  subscribers see `module_filename`, raw subscribers don't. `enumerate()` also trims the
  bootp deserializer's fixed-width 16-byte `hardware_address` blob down to
  `hardware_address_length` before any subscriber fires, so callers (including raw
  subscribers) only see the meaningful prefix.
- `tools/adapter_enumerator.cpp` — new executable. Parses `--raw` and
  `--hololink=<peer-ip>` (mutually exclusive; `--raw` shows pre-enrichment fields
  without loading modules, `--hololink` filters the enriched stream). Registers the
  appropriate callback and blocks on `pause()` so the default SIGINT / SIGTERM handlers
  terminate cleanly — the Adapter already started the bootp listener in its constructor.
  Each announcement renders as a timestamped block with one `key = value` per line;
  int64 values print as decimal, strings are quoted with non-printable bytes escaped,
  and byte blobs render as `<N bytes: aa bb …>`.
- `tools/CMakeLists.txt` — `add_executable(adapter_enumerator ...)` linking
  `hololink::module` only; installed under `${CMAKE_INSTALL_BINDIR}` in the
  `hololink-module-tools` install component.
- `hololink_module/CMakeLists.txt` — `add_subdirectory(tools)`.

**Out of scope this phase.** No Python bindings for `register_raw_ip` /
`register_raw_all` yet; the tool is C++-only. No structured-output mode (JSON / TSV) —
the renderer is human-readable only. No filtering by UUID, serial, or module name.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD"
cmake --build "$BUILD" --target hololink_module
cmake --build "$BUILD" --target adapter_enumerator
```

**Run commands.**

```bash
BUILD=/tmp/build-hololink
# Show every device, enriched with its module's update_metadata.
"$BUILD/hololink_module/tools/adapter_enumerator"

# Show only what bootp carried; do not load any module .so.
"$BUILD/hololink_module/tools/adapter_enumerator" --raw

# Filter to a single peer.
"$BUILD/hololink_module/tools/adapter_enumerator" --hololink=192.168.0.2
```

## Phase 3 extension — bootp listener auto-starts in `Adapter::Adapter()`

**Implemented.** The Adapter's constructor calls `start_bootp_listener()` itself, so
application code, examples, and tools no longer need an explicit call. The underlying
socket has `SO_REUSEADDR` and `SO_REUSEPORT` enabled so a process that constructs the
Adapter singleton coexists with other module-using processes on the host (each receives
every broadcast bootp announcement from the kernel).

- `host/src/adapter.cpp` — `Adapter::Adapter()` calls `start_bootp_listener()` after
  publishing the Logging / Reactor services. The socket setup sets both reuse flags
  before delegating to `Enumerator::configure_socket` (which performs the `bind`); if
  either `setsockopt` fails the socket is closed and the error propagates.
  `start_bootp_listener` remains idempotent — a second call while the listener is up is
  a no-op — and `stop_bootp_listener` is still available for tests / tools that need to
  rebind on a custom port.
- `host/include/hololink/module/adapter.hpp` — doc comment updated; the public API
  signature is unchanged.
- Examples and the new `adapter_enumerator` tool — explicit
  `adapter.start_bootp_listener()` calls removed in favor of the auto-start.
- Tests that bind a non-default port (`bootp_listener_test` exercising port 0;
  `bootp_e2e_test` exercising `TEST_PORT`) now call `stop_bootp_listener()` first to
  release the default-port socket the constructor opened, then
  `start_bootp_listener (custom_port)`.

**Out of scope this phase.** No lazy-start mode (the listener always runs from
construction). No way to disable bootp entirely — applications that don't want it can
ignore the open socket, but the FD is always held.

## Phase 5 extension — `ConfigurableService<T>` + lazy module-side construction

**Implemented.** Service interfaces that need metadata-driven setup gain a
`ConfigurableService<T>` CRTP layer on top of `Service<T>` (renamed from
`ServiceLocatable<T>`). The metadata-form `T::get_service(EnumerationMetadata)` walks
the host through the module-side Publisher; on a cache miss the module
default-constructs the impl, registers it, and the host calls `configure(metadata)` on
the typed result to materialize backing resources. `HololinkInterfaceV1` is the first
interface migrated to this pattern.

- `host/include/hololink/module/service.hpp` (renamed from `service_locatable.hpp`) —
  `Service<T>` is the renamed CRTP base. `ConfigurableService<T>` is a new derived layer
  adding the metadata-form static `get_service(metadata, allow_null)` and a virtual
  `configure(metadata)` (default no-op). The metadata-form template delegates to a
  host-only helper defined in `host/src/service.cpp` so the header does not drag
  `adapter.hpp` into module-side TUs that include it.

- `host/include/hololink/module/service_locator.h` — the `hololink_module_get_service`
  function pointer does cache-or-construct internally on the module side; no parallel
  "get_or_construct" function pointer is added. For reconnection, `release_service` is
  now **refcounted** (was a no-op) so cross-`.so` host handles are owning — a behavioral
  change only; the C-ABI structs are unchanged (invalidation is driven by a
  `device_lost()` virtual on the interfaces, not a new callback). See the `Publisher`
  entry.

- `host/include/hololink/module/module.hpp` + `host/src_module/module_base.cpp` —
  `Module`'s sole `get_service` virtual carries the cache-or-construct semantics: a
  registry miss invokes the registered service constructor (if any), which publishes the
  new impl back through the same Publisher; the lookup then re-reads and returns.
  `Publisher::SelfModule` follows the same path for in-binary lookups.

- `host/include/hololink/module/publisher.hpp` + `module_base.cpp` — `Publisher` is now
  abstract. `construct_service(instance_id, type_id)` is pure virtual; every binary that
  creates a Publisher declares its own subclass that overrides it (returning `{}` when
  there are no lazy services). There's no factory method — callers build the subclass
  directly with `std::make_shared<MySubclass>()`, and the `Publisher` base constructor
  stamps `current_` (throwing if a Publisher already exists in this binary). The
  constructor is protected and the destructor is virtual. `Publisher::lookup` does the
  cache-or-construct flow: registry hit → return; miss → call the virtual; re-read the
  registry. `Publisher::get_service_thunk` calls `lookup` directly.

  - **Refcounted handles + `device_lost()` (reconnection support).** The C-ABI hands the
    host a raw pointer into a registry-owned instance, so eviction would otherwise
    dangle outstanding handles. `get_service_thunk` now records a strong reference per
    handout in an `outstanding_` side-table (keyed by the raw pointer, with a
    live-handle count); `release_service_thunk` — previously a no-op — drops one
    reference (erasing at count 0). An instance therefore stays alive while `registry_`
    caches it **or** any host handle is outstanding, making host handles owning.
    Invalidation is driven through the device-state objects; the object graph (which
    services form a board) is known by the objects, never by the `Publisher`.
    `HololinkInterfaceV1` and `DataChannelInterfaceV1` gain a `device_lost()` virtual.
    `HololinkV1` is the board aggregation point: per-board services register with it via
    `register_associated(this)` — in their `configure` if they have one, else in their
    constructor (taking the owning `HololinkV1`, passed by the matching `construct_*`) —
    so `HololinkV1::device_lost()` invalidates every associated service and itself.
    `DataChannelV1::device_lost()` delegates to `hololink_->device_lost()` (the cascade
    covers the channel, which registered). `Publisher::invalidate(const void* service)`
    is a dumb, identity-keyed static: it removes the `registry_` entries whose stored
    pointer equals `service` (a service is published under its most-derived type, so
    `this` matches `.get()`), resolving this binary's `Publisher` via `current_` like
    the get_service / release_service thunks — it has **no** notion of device, serial,
    or grouping (it's compare-only, never dereferences). Outstanding handles keep the
    instance alive until released; the next `lookup` reconstructs a fresh one via
    `construct_service`. Conceptually the module analogue of the legacy
    `Hololink::reset_framework()`, but object-driven rather than a central serial-keyed
    wipe. `device_lost()` is bound in Python on both interfaces.

  - **Control-plane loss detection.** Detected at the reconnection boundary, not inside
    the wrappers. `HololinkV1`'s register methods do not catch backing exceptions — a
    control-plane `hololink::TransactionError` / `TimeoutError` propagates just as it
    does out of every sibling wrapper (`i2c` / `data_channel` / `sequencer`). The one
    place that owns loss recovery, `SensorFactory::on_enumerated`, wraps the whole
    (re)connect bring-up (`new_sensor` + `on_connect`) in a single sanctioned
    recovery-boundary `catch (const std::exception&)`: a bring-up that throws, or a
    Python `new_sensor` that reports failure by returning null, routes into
    `invalidate_board(metadata)`, which resolves the board's `HololinkInterfaceV1` and
    calls `device_lost()` (the same per-board cascade above) so the next announcement
    re-materializes fresh device state instead of the stale handles that just failed.
    The data-plane watchdog covers steady-state loss independently.

  - **Host side** — `Adapter::Adapter()` builds the host's publisher via
    `std::make_shared<HostPublisher>()`; `HostPublisher::construct_service` is a no-op
    (host services Reactor / Logging are eagerly published).

  - **Modules** — each module defines a per-module subclass with a real override;
    `hsb_lite` / `hsb_lite_2510` ship `HsbLitePublisher`.

  - **Test stubs** — the four stub modules each define a `TestPublisher` whose override
    is also a no-op (every test service is eagerly published in `module_init`).

- `host/include/hololink/module/hololink.hpp` — `HololinkInterfaceV1` now derives from
  `ConfigurableService<HololinkInterfaceV1>` and exposes
  `static std::string instance_id_for(const EnumerationMetadata&)`; the hand-written
  metadata-form `get_service` overload is gone (the template provides it).
  `host/src/hololink.cpp` is reduced to a placeholder.

- `module/core/hololink_default.{hpp,cpp}` — `HololinkV1` has a cheap default
  constructor; `configure(metadata)` materializes `backing_` (the `LegacyHololinkAccess`
  plus the on-reset clock-rate listener) on first call and is idempotent thereafter.

- `module/hsb_lite/module_entry.cpp` + `module/hsb_lite_2510/module_entry.cpp` — define
  a file-local `HsbLitePublisher : public Publisher` whose `construct_service` override
  switches on `type_id` and produces a default `HololinkV1` for `HololinkInterfaceV1`
  (publishing it under the supplied `instance_id` via
  `ServicePublisher<HololinkInterfaceV1>`). The module's init builds it via
  `std::make_shared<HsbLitePublisher>()`. Other type_ids return `{}` for now (other
  impls remain on the eager path until they migrate).

- `module/hsb_lite/hsb_lite_enumeration.{hpp,cpp}` +
  `module/hsb_lite_2510/hsb_lite_enumeration.{hpp,cpp}` — `update_metadata` no longer
  constructs or publishes `HololinkV1`; the lazy path handles it. The `Board.hololink`
  field is removed. The board's `LegacyHololinkAccess` is still constructed here for
  HsbLite / per-data-plane services (RoceDataChannel, Sequencer, RoceReceiver,
  Oscillator, I2c) that have not yet migrated.

**Out of scope this phase.** Only HololinkInterface migrates this turn. HsbLite,
RoceDataChannel, Sequencer, RoceReceiver, Oscillator, and I2c remain eagerly published
with the per-board `LegacyHololinkAccess` constructed inside `update_metadata` — folding
them onto a single `LegacyHololinkAccess` shared with the lazy Hololink (or onto
`ConfigurableService` themselves) is a follow-up. `update_metadata` therefore is not yet
fully write-only; the HsbLite / per-data-plane construction code there will move next.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD"
cmake --build "$BUILD" --target hololink_module
cmake --build "$BUILD" --target hololink_889b7ce3-65a5-4247-8b05-4ff1904c3359_2603
cmake --build "$BUILD" --target hololink_889b7ce3-65a5-4247-8b05-4ff1904c3359
cmake --build "$BUILD" --target hololink_module_enumeration_test
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" -R hololink_module_enumeration_test --output-on-failure
ctest --test-dir "$BUILD" -R hololink_module_hsb_lite_test --output-on-failure
```

## Phase 5 extension — `RoceDataChannelInterfaceV1` migrates to `ConfigurableService`

**Implemented.** `RoceDataChannelInterfaceV1` joins `HololinkInterfaceV1` and
`OscillatorInterfaceV1` on the `ConfigurableService` lazy-construct path. The existing
operator-facing pair `configure(metadata, receiver)` / `unconfigure()` was renamed to
`attach_receiver(receiver)` / `detach_receiver()` to free up the `configure(metadata)`
name for the one-shot lazy materialization contract.

- `host/include/hololink/module/roce_data_channel.hpp` — base class change to
  `ConfigurableService<RoceDataChannelInterfaceV1>`. Adds
  `static std::string instance_id_for(const EnumerationMetadata&)` returning
  `"serial=<n>;data_plane=<p>"`. The operator-facing methods are renamed to
  `attach_receiver` / `detach_receiver`.
- `module/core/roce_data_channel_default.hpp` — `RoceDataChannelV1` has a cheap default
  constructor. `configure(metadata)` is wrapped in `std::call_once`: fetches the
  per-board `HololinkInterfaceV1` via the module's own Publisher (cache-or-construct),
  casts to `module_core::HololinkV1`, calls `legacy_access()` (throws if the Hololink
  hasn't been configured yet), and builds the backing `hololink::DataChannel` against
  that legacy. The DataChannel type is produced via a new protected
  `virtual make_backing(legacy_metadata, create_hololink)` hook — default returns
  `hololink::DataChannel`, FPGA-revision subclasses override.
- `module/hsb_lite_2510/hsb_lite_2510_roce_data_channel.hpp` (new) —
  `HsbLite2510RoceDataChannelV1 : public RoceDataChannelV1` overrides `make_backing` to
  construct `HsbLite2510DataChannel` so the 2510 FPGA's `configure_roce(...)` override
  is dispatched through.
- `module/hsb_lite/module_entry.cpp` + `module/hsb_lite_2510/module_entry.cpp` —
  `HsbLitePublisher::construct_service` adds the `RoceDataChannelInterfaceV1::type_id`
  branch (2603 uses `module_core::RoceDataChannelV1`; 2510 uses
  `hsb_lite::HsbLite2510RoceDataChannelV1`).
- `module/hsb_lite/hsb_lite_enumeration.cpp` +
  `module/hsb_lite_2510/hsb_lite_enumeration.cpp` — `update_metadata` drops the eager
  RoceDataChannelImpl construction + publish; `DataPlane.data_channel` removed. A local
  `legacy_data_channel` is still built per data plane because Sequencer (still eagerly
  published) needs it for `frame_end_sequencer()`. Two `hololink::DataChannel` objects
  per `(serial, data_plane)` in the transitional state — they share the same per-board
  `LegacyHololinkAccess`. Folds together when Sequencer migrates.
- `host/operators/roce_receiver_op.cpp` — switches from the instance-id form to
  `RoceDataChannelInterfaceV1::get_service(metadata)` so the channel is configured
  before `attach_receiver(receiver_)` runs. Renamed calls: `attach_receiver` /
  `detach_receiver` (error messages updated to match).
- `host/include/hololink/module/hololink.hpp` — `get_roce_data_channel(md)` factory now
  calls `T::get_service(md)` (metadata form) so callers reach a fully-configured channel
  through the standard helper.
- `host/python/hololink_module_py.cpp` — `PyRoceDataChannel` trampoline overrides
  renamed; binding adds the metadata-form `get_service` static and renames the bound
  methods to `attach_receiver` / `detach_receiver`.
- `tests/test_hololink_module_v1_bindings_python.py` — Python subclass test updated to
  override `attach_receiver` / `detach_receiver`.

**Contract.** Fetching `RoceDataChannelInterfaceV1::get_service(metadata)` requires the
per-board Hololink to have been configured first (same shape as I2c). Applications that
drive the locator through the metadata form for everything chain the configuration
automatically — Hololink first, then RoceDataChannel.

**Out of scope this phase.** Sequencer and the per-data-plane RoceReceiver remain
eagerly published; Sequencer's eager `legacy_data_channel` is the reason
`update_metadata` still constructs one. Migrating Sequencer collapses the
two-DataChannel transitional state.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD"
cmake --build "$BUILD" --target hololink_module
cmake --build "$BUILD" --target hololink_889b7ce3-65a5-4247-8b05-4ff1904c3359_2603
cmake --build "$BUILD" --target hololink_889b7ce3-65a5-4247-8b05-4ff1904c3359
cmake --build "$BUILD" --target _hololink_py_module
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" -R hololink_module_v1_bindings_python --output-on-failure
ctest --test-dir "$BUILD" -R hololink_module_hsb_lite_test --output-on-failure
```

## Phase 5 extension — `RoceReceiverInterfaceV1` migrates to `ConfigurableService`

**Implemented.** `RoceReceiverInterfaceV1` joins Hololink / Oscillator / RoceDataChannel
on the lazy `ConfigurableService` path. The impl is a default- constructible shell (just
like before) — the operator's `start(...)` is what materializes ibverbs state.
`configure(metadata)` is the inherited no-op default. With this, every per-data-plane V1
service is lazy; `update_metadata` no longer publishes anything per data plane, the
`DataPlane` struct and `by_serial_data_plane_` map are gone, and the per-data-plane
block in `update_metadata` collapses to a one-liner comment.

- `host/include/hololink/module/roce_receiver.hpp` — derives
  `ConfigurableService<RoceReceiverInterfaceV1>`. Adds
  `static std::string instance_id_for(const EnumerationMetadata&)` returning
  `"serial=X;data_plane=N"`. Class-doc updated to note the shell pattern
  (default-constructed, `start(...)` does the real work).
- `module/hsb_lite/module_entry.cpp` + `module/hsb_lite_2510/module_entry.cpp` —
  `HsbLitePublisher::construct_service` gains the `RoceReceiverInterfaceV1::type_id`
  branch: `make_shared<…>` (2603: `module_core::RoceReceiverV1`; 2510:
  `hsb_lite::HsbLite2510RoceReceiverV1`), publish, return.
- `module/hsb_lite/hsb_lite_enumeration.{hpp,cpp}` +
  `module/hsb_lite_2510/hsb_lite_enumeration.{hpp,cpp}` — eager RoceReceiver
  construction + publish gone. `DataPlane` struct and `by_serial_data_plane_` map
  removed. `data_plane_instance_id` helper gone. `roce_receiver.hpp` /
  `roce_receiver_default.hpp` / `hsb_lite_2510_roce_receiver.hpp` /
  `oscillator_impl.hpp` / `<map>` includes dropped. `update_metadata` reduces to the
  metadata-stamping work plus the per-board `(legacy, HsbLite)` publish on first
  encounter of a serial.
- `host/operators/roce_receiver_op.cpp` — fetches the receiver via the metadata-form
  `RoceReceiverInterfaceV1::get_service(metadata_.get())` instead of the instance-id
  form, matching how it already resolves `RoceDataChannelInterfaceV1`.
- `host/python/hololink_module_py.cpp` — adds the metadata-form `get_service` static
  binding to `RoceReceiverInterface`.

**Out of scope this phase.** `HsbLiteInterfaceV1` is still eagerly published in
`update_metadata` — its public surface uses the `get_hsb_lite(module, metadata)` factory
rather than `ConfigurableService::get_service(metadata)`, so migrating it would change a
public helper signature. Folding HsbLite onto ConfigurableService (and finally
collapsing `update_metadata` to a pure metadata-stamper) is a follow-up.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD"
cmake --build "$BUILD" --target hololink_module
cmake --build "$BUILD" --target hololink_889b7ce3-65a5-4247-8b05-4ff1904c3359_2603
cmake --build "$BUILD" --target hololink_889b7ce3-65a5-4247-8b05-4ff1904c3359
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" -R hololink_module_hsb_lite_test --output-on-failure
ctest --test-dir "$BUILD" -R hololink_module_enumeration_test --output-on-failure
```

## Phase 5 extension — `HsbLiteInterfaceV1` migrates; `update_metadata` is now write-only

**Implemented.** `HsbLiteInterfaceV1` joins every other V1 service on the lazy
`ConfigurableService` path. With this final migration, the HsbLite supplements'
`update_metadata` collapses to pure metadata stamping — no publishing, no per-board
state, no constructor argument. The `HsbLiteEnumerationV1` class is now a
default-constructible stamper.

- `module/hsb_lite/include/hololink/module/hsb_lite/hsb_lite.hpp` — `HsbLiteInterfaceV1`
  derives `ConfigurableService<HsbLiteInterfaceV1>` and exposes
  `static std::string instance_id_for(const EnumerationMetadata&)` returning
  `"serial=<n>"`. The hand-written `get_hsb_lite(module, metadata)` factory is gone (the
  metadata-form `get_service` template provides it).
- `module/hsb_lite/hsb_lite_impl.hpp` (2603) + `module/hsb_lite_2510/hsb_lite_impl.hpp`
  (2510) — `HsbLiteV1` has a cheap default constructor. `configure(metadata)` is wrapped
  in `std::call_once`: resolves the per-board Hololink via
  `HololinkInterfaceV1::get_service(this->module(), "serial=<n>")`, casts to
  `module_core::HololinkV1`, calls its `configure(metadata)` (idempotent), and stashes
  the resulting `legacy_access()` for `setup_clock` to drive the I2C path against.
- `module/hsb_lite/module_entry.cpp` + `module/hsb_lite_2510/module_entry.cpp` —
  `HsbLitePublisher::construct_service` adds the `HsbLiteInterfaceV1::type_id` branch:
  default-construct, publish, return.
- `module/hsb_lite/hsb_lite_enumeration.{hpp,cpp}` +
  `module/hsb_lite_2510/hsb_lite_enumeration.{hpp,cpp}` — gutted. `HsbLiteEnumerationV1`
  keeps only its `update_metadata` override (which stamps `module_name`, backfills
  `control_port` and per-data-plane addresses, and returns). The class is now
  default-constructible: the `Publisher`, `Board`, and `by_serial_` members are all
  gone. The constructor in each `module_entry.cpp` calls
  `std::make_shared<HsbLiteEnumerationV1>()` with no argument.
- `module/hsb_lite/python/hsb_lite_py.cpp` — `HsbLiteInterface` Python binding swaps the
  `get_hsb_lite` static for both forms of `get_service` (instance-id + metadata),
  matching the surface every other V1 interface exposes.

**End state.** Every V1 service is now constructed lazily on first lookup;
`update_metadata` does what the directive originally specified ("only writes to the
enumeration metadata structure"). The full V1 chain (Hololink → HsbLite, RoceDataChannel
→ Sequencer, I2c → Hololink) cascades through `Publisher::lookup` + the per-impl
`configure(metadata)`; the user-visible contract is "fetch what you want via the
metadata form, in any order — its chain of prerequisites resolves itself."

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD"
cmake --build "$BUILD" --target hololink_module
cmake --build "$BUILD" --target hololink_889b7ce3-65a5-4247-8b05-4ff1904c3359_2603
cmake --build "$BUILD" --target hololink_889b7ce3-65a5-4247-8b05-4ff1904c3359
cmake --build "$BUILD" --target _hololink_py_module
cmake --build "$BUILD" --target _hololink_module_hsb_lite
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" -R hololink_module_hsb_lite_test --output-on-failure
ctest --test-dir "$BUILD" -R hololink_module_enumeration_test --output-on-failure
ctest --test-dir "$BUILD" -R hololink_module_v1_bindings_python --output-on-failure
```

### Phase 5 extension — `DataChannelInterfaceV1` anchor + per-board / per-channel dependent rule

**Goal.** Pull the per-channel anchor surface into a dedicated service
`DataChannelInterfaceV1` so sibling per-channel services route through *one* canonical
place for metadata and Hololink. RoCE-specific behaviour lives on a separate service
that composes with the anchor (has-a, not is-a) so a channel without RoCE — or with some
future non-RoCE transport — is still a valid `DataChannelInterfaceV1`.

**End state.**

- `host/include/hololink/module/data_channel.hpp` declares `DataChannelInterfaceV1` as
  the per-channel anchor — `ConfigurableService<DataChannelInterfaceV1>` with
  `type_id = "data_channel.v1"`. Transport-agnostic surface only:
  `enumeration_metadata()` and `hololink()`. No mention of RoCE.
- `host/include/hololink/module/roce_data_channel.hpp` declares
  `RoceDataChannelInterfaceV1` as its own
  `ConfigurableService<RoceDataChannelInterfaceV1>` with
  `type_id = "roce_data_channel.v1"`. *Not* derived from `DataChannelInterfaceV1`. Adds
  the RoCE-specific transport surface (`attach_receiver` / `detach_receiver`) plus the
  legacy `frame_end_sequencer` / `get_hololink` template helpers that the RoCE flow
  uses.
- `module/core/data_channel_default.hpp` defines `DataChannelV1` — the anchor impl.
  `configure(metadata)` stashes metadata, resolves the per-board Hololink via the
  `(module, instance_id)` form, and calls `hololink->ensure_configured(metadata)` so the
  framework latch materializes the backing.
- `module/core/roce_data_channel_default.hpp` updates `RoceDataChannelV1` to hold a
  `shared_ptr<DataChannelInterfaceV1>` (anchor) handed in at construction.
  `configure(metadata)` drives `anchor_->ensure_configured(metadata)` before building
  the legacy `hololink::DataChannel`, so an application that calls
  `RoceDataChannelInterfaceV1::get_service(metadata)` without first touching the anchor
  still works — the anchor materializes transitively. The `hsb_lite_2510` subclass
  inherits the anchor- taking constructor unchanged.
- `HololinkInterfaceV1` exposes `enumeration_metadata()`. `HololinkV1::configure`
  stashes metadata so per-board sibling services (`I2cInterfaceV1`,
  `OscillatorInterfaceV1`) read off the snapshot rather than carrying their own copy.
- `ConfigurableService<Derived>` owns the configure-once latch. Public
  `ensure_configured(metadata)` wraps the virtual `configure(metadata)` in
  `std::call_once` on a per-instance `configure_once_` flag. The framework's
  metadata-form `get_service` routes through `ensure_configured`, so cache-hit lookups
  don't re-run `configure`. Impls don't need their own `std::call_once`.
- Per-instance flavor (`I2c` bus, `Sequencer` kind) is parsed from the instance_id in
  the supplement's `Publisher::construct_service` switch and passed to the impl's
  constructor.
- Module entries (`module/hsb_lite/module_entry.cpp`,
  `module/hsb_lite_2510/module_entry.cpp`) publish two cache slots per channel:
  `DataChannelInterfaceV1::type_id` → `DataChannelV1`, and
  `RoceDataChannelInterfaceV1::type_id` → `RoceDataChannelV1` (with anchor resolved
  through the same Publisher). Sibling `Sequencer` construct resolves the legacy
  frame_end_sequencer through `RoceDataChannelInterfaceV1::get_service` (RoCE-specific
  resource).
- Python bindings expose `DataChannelInterface` and `RoceDataChannelInterface` as
  separate `py::class_` declarations with no inheritance between them. `__init__.py`
  exposes both unversioned + V1-pinned aliases.
- Construction contract for application code: an application that wants a RoCE channel
  calls `RoceDataChannelInterfaceV1::get_service(metadata)` and the anchor materializes
  transitively. An application that only wants the anchor calls
  `DataChannelInterfaceV1::get_service(metadata)`. Sibling per-channel / per-board
  services that need a *cached* anchor (Sequencer, RoceReceiver, I2c, Oscillator) do a
  cache-only fetch and fail fast with a clear message if the anchor wasn't built yet.

**Why this shape.** Composition keeps `DataChannelInterfaceV1` transport-agnostic — a
CoE channel (or any future variant) can implement just the anchor without inheriting
from anything RoCE-specific. The framework-owned configure-once latch means the "don't
re-run configure on cache hits" semantics are one rule in one place, not duplicated
`std::call_once` in every impl.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD"
cmake --build "$BUILD" --target hololink_module
cmake --build "$BUILD" --target hololink_889b7ce3-65a5-4247-8b05-4ff1904c3359_2603
cmake --build "$BUILD" --target hololink_889b7ce3-65a5-4247-8b05-4ff1904c3359
cmake --build "$BUILD" --target _hololink_py_module
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" --output-on-failure -LE hardware
```

## Phase 5 extension — `EnumerationMetadata` cache rule + `attach_receiver` drops redundant parameter

**Implemented.** Pinned the convention that `DataChannelInterfaceV1` is the per-channel
`EnumerationMetadata` cache and `HololinkInterfaceV1` is the per-board cache (exempt).
Every other V1 service that needs metadata reads through one of the two — per-channel
services hold a `shared_ptr<DataChannelInterfaceV1>` and read
`anchor_->enumeration_metadata()`, per-board services hold a
`shared_ptr<HololinkInterfaceV1>` and read `hololink_->enumeration_metadata()`. No other
service caches its own metadata. Caching of derived scalars (e.g.
`OscillatorImpl::data_plane_`) is fine — the rule is about not holding a second copy of
the metadata blob.

The codebase was already largely compliant; the single drift was the
`RoceDataChannelInterfaceV1::attach_receiver(metadata, receiver)` signature — the impl
ignored the `metadata` parameter (`module/core/roce_data_channel_default.hpp` marked it
`/*metadata*/`) and pulls anything it needs through `anchor_->enumeration_metadata()`
already. The parameter is removed.

- `host/include/hololink/module/roce_data_channel.hpp` — `attach_receiver(receiver)`
  (the `metadata` parameter is gone). Doc comment notes that per-channel enumeration
  fields are on the anchor.
- `module/core/roce_data_channel_default.hpp` — `RoceDataChannelV1::attach_receiver`
  signature matches; body unchanged.
- `host/operators/roce_receiver_op.cpp` — caller drops the metadata argument.
- `host/operators/include/hololink/module/operators/roce_receiver_op.hpp` — lifecycle
  comment updated.
- `host/include/hololink/module/roce_receiver.hpp` — class doc + accessor doc updated to
  spell `attach_receiver(receiver)`.
- `host/python/hololink_module_py.cpp` — `PyRoceDataChannel` trampoline and the
  `def("attach_receiver", ...)` binding drop the metadata argument.
- `host/include/hololink/module/data_channel.hpp` — class doc spells out that this is
  the primary `EnumerationMetadata` cache.
- `host/include/hololink/module/hololink.hpp` — class doc + `enumeration_metadata()`
  accessor doc spell out that per-board services read through this accessor rather than
  holding their own copy.
- `tests/test_hololink_module_v1_bindings_python.py` — Python subclass test updated to
  the one-argument signature.

**Constraint for deferred Phase 5 wrappers.** When the deferred wrappers land
(`HsbLiteV1`, any future per-channel/per-board service), they follow the rule:
per-channel impls hold a `DataChannelInterfaceV1` anchor and read through its accessor;
per-board impls hold a `HololinkInterfaceV1` and read through its accessor. No new
service stores `EnumerationMetadata` directly.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD"
cmake --build "$BUILD" --target hololink_module
cmake --build "$BUILD" --target hololink_889b7ce3-65a5-4247-8b05-4ff1904c3359_2603
cmake --build "$BUILD" --target hololink_889b7ce3-65a5-4247-8b05-4ff1904c3359
cmake --build "$BUILD" --target _hololink_py_module
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" --output-on-failure -LE hardware
```

## Phase 5 extension — `HololinkImpl::configure` builds the default `DataChannelInterfaceV1`

**Implemented.** `HololinkImpl::configure(metadata)` now constructs a single
`DataChannelInterfaceV1` alongside the Hololink and exposes it via a new
`HololinkInterfaceV1::default_data_channel()` virtual. The default channel is built from
the same `EnumerationMetadata` blob the Hololink was configured with, so its per-board
fields cannot drift onto inconsistent state — the two are bound at construction.
Board-level code that wants a channel handle reads this accessor; per-data-plane
channels beyond the default are still reached through
`DataChannelInterfaceV1::get_service(metadata)`.

- `host/include/hololink/module/hololink.hpp` — forward-declares
  `DataChannelInterfaceV1`; `HololinkInterfaceV1` now also inherits
  `std::enable_shared_from_this<HololinkInterfaceV1>` so the impl can hand
  `shared_from_this()` to the channel; new virtual `default_data_channel() const`.
- `module/core/data_channel_default.hpp` — `DataChannelV1` gains an eager 2-arg
  constructor `(shared_ptr<HololinkInterfaceV1>, EnumerationMetadata)` that stashes both
  directly. No parent lookup, no `ensure_configured` call — the Hololink is in the
  middle of its own `configure`, so a recursive `ensure_configured` would deadlock the
  once_flag. The default channel is held on the Hololink (not in the Publisher's cache),
  so the framework's `configure(metadata)` is not expected to run on it. The 0-arg ctor
  \+ `configure(metadata)` path is unchanged for Publisher-driven channels.
- `module/core/hololink_default.hpp` — new `default_data_channel_` member and the
  corresponding accessor override. Includes `hololink/module/data_channel.hpp` for the
  abstract type.
- `module/core/hololink_default.cpp` — `configure` constructs `default_data_channel_`
  after `backing_` is materialized, only when `metadata` carries `data_plane`
  (control-plane-only Hololinks leave it null).
- `host/python/hololink_module_py.cpp` — `PyHololinkInterface` trampoline adds
  `default_data_channel()` via `PYBIND11_OVERRIDE_PURE`; binding adds
  `def("default_data_channel", ...)`.
- `tests/test_hololink_module_v1_bindings_python.py` — `FakeBoard` picks up the new
  override; one extra assertion proves the binding dispatches.

**Why this shape.** A second top-level `enable_shared_from_this` on
`HololinkInterfaceV1` (alongside the framework's `ConfigurableService`) is the simplest
way for `HololinkImpl::configure` to hand a typed `shared_ptr<HololinkInterfaceV1>` to
the channel without re-deriving it through the locator. The framework's
`weak_ptr<Module>` on `Service` covers a different need (recovering the Module from a
typed pointer) and doesn't expose `shared_from_this()` on the interface type.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD"
cmake --build "$BUILD" --target hololink_module
cmake --build "$BUILD" --target hololink_889b7ce3-65a5-4247-8b05-4ff1904c3359_2603
cmake --build "$BUILD" --target hololink_889b7ce3-65a5-4247-8b05-4ff1904c3359
cmake --build "$BUILD" --target _hololink_py_module
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" --output-on-failure -LE hardware
```

## Phase 5 extension — module-side HSB_LOG bridge wired in HSB-Lite supplements

**Implemented.** The `HSB_LOG_*` macros in `host/include/hololink/module/logging.hpp`
dispatch through a per-binary cache pointer (`hololink::module::hsb_logger_cache`).
Because the host loads each module under `RTLD_LOCAL`, every .so gets its own copy of
that pointer — the host's `Adapter` sets the host pointer at startup, but supplement .so
files have to set their own. Prior to this change neither HSB-Lite supplement did, so
`HSB_LOG_*` calls inside supplement code silently no-op'd.

- `module/hsb_lite/module_entry.cpp` and `module/hsb_lite_2510/module_entry.cpp` —
  `hololink_module_init` now fetches `LoggingInterfaceV1` via the locator immediately
  after the `LoadedModule` is built and calls
  `hololink::module::set_hsb_logger_cache(g_logger.get())` to wire that .so's cache. The
  pattern mirrors the host bootstrap in `host/src/adapter.cpp:267` and the test fixture
  in `tests/hololink_module_singletons_stub_module.cpp:62-63`. A module-scope
  `static std::shared_ptr<LoggingInterfaceV1> g_logger` keeps the impl alive for the
  life of the .so.
- `module/hsb_lite_2510/hsb_lite_2510_roce_data_channel.hpp` — adds the missing
  `hololink/module/logging.hpp` include and fixes a placeholder-without-argument call:
  `HSB_LOG_INFO("hsb_ip_version={:#x}", hsb_ip_version);` (previously `HSB_LOG_INFO` was
  called with a format string and no argument, which would have thrown at runtime even
  if the cache had been wired). Logs which `hsb_ip_version` the supplement saw when
  selecting between the legacy and 2510 RoCE backing.

**Why each supplement does this itself.** The cache symbol lives in
`hololink::module_runtime` (the static archive every supplement links via
`add_hololink_module()` in `cmake/HololinkModule.cmake:100`). With `RTLD_LOCAL`, symbol
lookups don't cross .so boundaries — so the host's setter affects only the host binary,
and each module must call its own copy of `set_hsb_logger_cache` to make `HSB_LOG_*`
reach the host's logger.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD"
cmake --build "$BUILD" --target hololink_889b7ce3-65a5-4247-8b05-4ff1904c3359_2603
cmake --build "$BUILD" --target hololink_889b7ce3-65a5-4247-8b05-4ff1904c3359
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" --output-on-failure -LE hardware
```

## Phase 5 extension — `instance_id_for(metadata)` renamed to `locator_id(metadata)`

**Implemented.** The static helper every `ConfigurableService<T>` exposes for deriving
its locator key from `EnumerationMetadata` was named `instance_id_for(metadata)`. That
name read as "compute an instance_id for this metadata" but the term "instance_id" is
also the name of the local variable the framework stores after the derivation, and the
noun was being overloaded with a potential future instance-bound accessor (e.g. asking a
configured service "what's your key?"). Renaming the static to `locator_id` describes
what it actually returns (the locator's id for this service shape) and leaves
`instance_id` free for the local-variable / locator-API parameter sense.

- Renamed in declarations on every V1 interface:
  `host/include/hololink/module/{hololink, data_channel, roce_data_channel, roce_receiver, oscillator}.hpp`
  and `module/hsb_lite/include/.../hsb_lite.hpp`.
- Renamed in the framework's `Service<T>::get_service(module, metadata)` trampoline at
  `host/include/hololink/module/service.hpp` — the inline call becomes
  `Derived::locator_id(metadata)`.
- Renamed at every call site that did the manual anchor-fetch dance:
  `module/core/{data_channel_default, roce_receiver_default}.hpp` and both supplement
  `module_entry.cpp` files.
- Pre-rename phase descriptions earlier in this README still mention `instance_id_for` —
  they document what each phase shipped at the time and were intentionally left alone.

No behavior change. Build / test commands unchanged.

## Phase 5 extension — HSB-Lite supplements compacted to fewer files

**Implemented.** Each HSB-Lite supplement was a constellation of short
one-class-per-file headers (`hsb_lite_impl.hpp`, `oscillator_impl.{hpp,cpp}`,
`hsb_lite_enumeration.{hpp,cpp}`, plus three thin override headers in 2510) that only
`module_entry.cpp` ever included. They've been folded into `module_entry.cpp` in each
supplement directory.

- `module/hsb_lite/` — went from 6 internal source files to 1 (`module_entry.cpp`).
  `HsbLiteEnumerationV1`, `HsbLiteV1`, and `OscillatorV1` are now defined inline at the
  top of the file in `namespace hololink::module::hsb_lite` before the
  anonymous-namespace `HsbLitePublisher`.
- `module/hsb_lite_2510/` — went from 9 internal source files to 2 (`module_entry.cpp`
  plus `hsb_lite_2510_data_channel.hpp`). The legacy `hololink::DataChannel` subclass
  stays in its own header because of its size (~140 LOC of register manipulation) and
  its distinct layer; everything else — the stock services,
  `HsbLite2510LegacyRoceReceiver`, and the two `module_core` override wrappers
  (`HsbLite2510RoceReceiverV1`, `HsbLite2510RoceDataChannelV1`) — is inline.
- Both `CMakeLists.txt` files now list only `module_entry.cpp` under `SOURCES`.
- No cross-directory `#include`s — the two supplements stay independent of each other.
  No public surface, python sub-package, or `module/core/` wrapper was touched.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD"
cmake --build "$BUILD" --target hololink_889b7ce3-65a5-4247-8b05-4ff1904c3359_2603
cmake --build "$BUILD" --target hololink_889b7ce3-65a5-4247-8b05-4ff1904c3359
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" --output-on-failure -LE hardware
```

## Phase 5 extension — virtual `ServiceBase` + per-impl `type_id` so supplements lose their `static_pointer_cast`s

**Implemented.** Sibling impls inside a supplement need to call impl-only surface
(`HololinkV1::legacy_access()`, `RoceDataChannelV1::legacy_data_channel()`), which isn't
on the V1 interface. The old pattern was: fetch the V1, then
`std::static_pointer_cast<HololinkV1>(iface)`. That cast trusted an unwritten contract
("whoever publishes `hololink.v1` publishes a `module_core::HololinkV1`") — and any
supplement that broke the contract would `static_cast` to the wrong type silently.

Two changes make the typed lookup safe and cast-free:

1. **`service.hpp` — virtual `ServiceBase`.** `Service<T>`'s instance state (`module_`
   weak_ptr + `module()` accessor) moves into a non-template `ServiceBase`; `Service<T>`
   inherits it virtually. With virtual inheritance, any class that picks up multiple
   `Service<T>` bases up its hierarchy ends up with a single `ServiceBase` subobject —
   one `module_`, one stamping site. The friend declaration moves to `ServiceBase` so
   `ServicePublisher<T>::publish` stamps the shared `module_` regardless of which `T`
   it's parameterized on. `publisher.hpp`'s stamping line is now
   `impl->ServiceBase::module_ = ...`.

1. **Per-impl `type_id` + `Service<Impl>` second base.** Each impl gets its own
   namespace-flavored `type_id` distinct from its V1 interface's:

   - `module_core::HololinkV1` → `"hololink.module_core.v1"`
   - `module_core::DataChannelV1` → `"data_channel.module_core.v1"`
   - `module_core::RoceDataChannelV1` → `"roce_data_channel.module_core.v1"`
   - `module_core::RoceReceiverV1` → `"roce_receiver.module_core.v1"`
   - `hsb_lite::HsbLiteV1` (in `module/hsb_lite/`) → `"hsb_lite.hsb_lite.v1"`
   - `hsb_lite::OscillatorV1` (in `module/hsb_lite/`) → `"oscillator.hsb_lite.v1"`
   - `hsb_lite::HsbLiteV1` (in `module/hsb_lite_2510/`) → `"hsb_lite.hsb_lite_2510.v1"`
   - `hsb_lite::HsbLite2510OscillatorV1` → `"oscillator.hsb_lite_2510.v1"`
   - `hsb_lite::HsbLite2510RoceReceiverV1` → `"roce_receiver.hsb_lite_2510.v1"`
   - `hsb_lite::HsbLite2510RoceDataChannelV1` → `"roce_data_channel.hsb_lite_2510.v1"`

   Each impl picks up `Service<Impl>` as a second base and declares
   `using Service<Impl>::get_service;` in its class body. The using-declaration hides
   the inherited V1-side `get_service` overloads via C++ name-hiding (any declaration of
   the same name in the derived class hides all inherited declarations of that name,
   regardless of signature), so `HololinkV1::get_service(module, instance_id)` resolves
   to `Service<HololinkV1>::get_service` unambiguously. The framework's
   `static_pointer_cast` inside that call is provably safe because the impl `type_id` is
   only published with `Impl` instances. Virtual inheritance of `ServiceBase` ensures
   the impl's `Service<Impl>` base and the V1 chain's `Service<V1>` base share a single
   `module_` weak_ptr — no diamond.

**Each supplement's `construct_service` now publishes under every relevant `type_id`.**
The same `shared_ptr` is registered N times (interface + impl + any board-specific
subclass) — each `ServicePublisher<T>::publish` call costs one extra map entry in the
locator and is idempotent on the `ServiceBase::module_` stamp. For a 2510 RoCE channel
that's three keys: `RoceDataChannelInterfaceV1::type_id`, `RoceDataChannelV1::type_id`,
`HsbLite2510RoceDataChannelV1::type_id`.

**Cast sites eliminated.** All ten `std::static_pointer_cast<*ImplV1>(...)` call sites
in `module/core/{data_channel_default,roce_data_channel_default}.hpp` and both
supplement `module_entry.cpp` files are gone. Each is replaced with
`HololinkV1::get_service(...)` (or `RoceDataChannelV1::get_service(...)`).

**Files modified.**

- `host/include/hololink/module/service.hpp` — split `Service<T>` into `ServiceBase` +
  `Service<T> : virtual ServiceBase`.
- `host/include/hololink/module/publisher.hpp` — `ServicePublisher<T>::publish` stamps
  `impl->ServiceBase::module_`.
- `module/core/{hololink_default,data_channel_default,roce_data_channel_default,roce_receiver_default}.hpp`
  — per-impl `type_id` + `Service<Impl>` second base +
  `using Service<Impl>::get_service;`; cast sites removed.
- `module/hsb_lite/module_entry.cpp` and `module/hsb_lite_2510/module_entry.cpp` — same
  pattern on every impl/subclass; each `construct_service` branch publishes under all
  relevant `type_id`s; four `static_pointer_cast` sites replaced.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD"
cmake --build "$BUILD" --target hololink_889b7ce3-65a5-4247-8b05-4ff1904c3359_2603
cmake --build "$BUILD" --target hololink_889b7ce3-65a5-4247-8b05-4ff1904c3359
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" --output-on-failure -LE hardware
```

## Phase 5 extension — `EnumerationMetadata::get` returns `T` directly + default-value overload

**Implemented.** The single `get<T>(key) → std::optional<T>` method was replaced by two
overloads that both return `T`:

```cpp
template <typename T>
T get(const std::string& key) const;
    // throws std::runtime_error on miss-or-wrong-type

template <typename T>
T get(const std::string& key, const T& default_value) const;
    // returns default_value on miss-or-wrong-type
```

A new `contains(const std::string&) const → bool` member was added alongside (the
project builds against C++17; `std::map::contains` is C++20-only so we provide it
directly via `find(key) != cend()`).

Why drop the optional return:

- **Required-field callers** went from
  `auto opt = md.get<T>(k); if (!opt) throw "..."; *opt` → `md.get<T>(k)`. The
  framework's throw text identifies the missing key.
- **Default-value callers** went from `md.get<T>(k).value_or(d)` → `md.get<T>(k, d)`.
- **Presence-checking callers** went from `auto opt = md.get<T>(k); if (opt) {...}` →
  `if (md.contains(k)) { ... md.get<T>(k) ... }`. Behavior change to flag: the old
  optional-return silently treated "key present, wrong variant" the same as "missing";
  `contains(k)` returns true for "present with wrong variant" and the subsequent
  `get<T>(k)` throws. At every site the variant is fixed by convention, so a type
  mismatch is a programmer bug — surfacing it is preferable.

**Files modified.**

- `host/include/hololink/module/enumeration_metadata.hpp` — the two `get` overloads +
  `contains`. `<optional>` include dropped; `<stdexcept>` added.
- All six V1 `locator_id` definitions (`hololink.hpp`, `data_channel.hpp`,
  `roce_data_channel.hpp`, `roce_receiver.hpp`, `oscillator.hpp`,
  `module/hsb_lite/include/.../hsb_lite.hpp`) collapsed to one-liners.
- `module/core/{hololink_default.cpp,.hpp, roce_receiver_default.hpp, roce_data_channel_default.hpp, data_plane_metadata.hpp}`
  — categories A/B/C.
- `host/src/adapter.cpp` — five sites: two `fpga_uuid` throw blocks dropped; three
  presence-checks switched to `contains`; the `compat_id` callers build the
  `std::optional<int64_t>` locally (`Adapter::load_module_for`'s signature stays as-is).
- `host/sensors/imx274/imx274_cam.cpp` — `expander_configuration_from` simplified.
- `host/operators/roce_receiver_op.cpp` — `peer_ip` miss-check dropped; uses the result
  directly (no `*`).
- `tools/adapter_enumerator.cpp` — `peer_ip` default-value (1 site).
- `tests/hololink_module_bootp_stub_module.cpp` — five test-snapshot fields switched to
  default-value form.
- `tests/hololink_module_enumeration_test.cpp` and
  `tests/hololink_module_hsb_lite_test.cpp` — `has_value`/`*opt` patterns switched to
  `contains` + direct comparison; `.value_or("")` switched to the default-value
  overload.
- Both supplements' `module_entry.cpp` — Oscillator `configure(metadata)`'s data_plane
  miss-check dropped.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD"
cmake --build "$BUILD"
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" --output-on-failure -LE hardware
```

## Phase 5 extension — drop `Impl` from V1 implementation class names

**Implemented.** The convention `FooInterfaceV1` (interface) + `FooImplV1` (impl) is
redundant once the impl lives in a clearly implementation-oriented namespace. Renamed:

- **Module impls** (in `module_core::` or `hsb_lite::`) — drop `Impl`, keep `V1`. Module
  source code always uses `V*` for both imports and exports; the same impl object may or
  may not implement multiple V_n interfaces, but the version tag stays in the name.

  - `module_core::{Hololink,DataChannel,RoceDataChannel,RoceReceiver,I2c,Sequencer,FrameMetadata}V1`
  - `hsb_lite::{HsbLite, HsbLiteEnumeration, HsbLiteOscillator}V1` (hsb_lite supplement)
  - `hsb_lite::HsbLite2510{,Enumeration,Oscillator,RoceReceiver,RoceDataChannel}V1`
    (hsb_lite_2510 supplement)

  Supplement impl class names carry the supplement-name prefix (`HsbLite…` or
  `HsbLite2510…`) so each impl is searchable by name alone. Module_core impls
  (`HololinkV1`, `DataChannelV1`, …) stay prefix-free — they're the canonical default
  implementations.

- **Host impls** (`ReactorImplV1`, `LoggingImplV1`) — drop **both** `Impl` and `V1`. The
  classes implement the latest `*V_n` and, when V2 lands, the same instance is published
  under every prior version's type_id by inheritance. A version-tagged class name would
  lie at that point.

  - `ReactorImplV1` → `hololink::module::Reactor`
  - `LoggingImplV1` → `hololink::module::Logging`

  Both are `.cpp`-local classes in `hololink::module` (defined in `reactor_impl.cpp` /
  `logging_impl.cpp`); the leaf names `Reactor` / `Logging` sit alongside — and don't
  collide with — the `ReactorV1` / `LoggingInterfaceV1` interfaces they implement.

- **Kept as-is** — `module_core::I2cLockImplV1` (interface is `I2cLockV1` without the
  `Interface` suffix, so dropping `Impl` collides visually) and
  `module_core::I2cLockNamedV1` (sibling).

**`type_id` strings updated** — every impl's runtime key drops `_impl`:

| Old                                         | New                                    |
| ------------------------------------------- | -------------------------------------- |
| `"hololink_impl.module_core.v1"`            | `"hololink.module_core.v1"`            |
| `"data_channel_impl.module_core.v1"`        | `"data_channel.module_core.v1"`        |
| `"roce_data_channel_impl.module_core.v1"`   | `"roce_data_channel.module_core.v1"`   |
| `"roce_receiver_impl.module_core.v1"`       | `"roce_receiver.module_core.v1"`       |
| `"hsb_lite_impl.hsb_lite.v1"`               | `"hsb_lite.hsb_lite.v1"`               |
| `"oscillator_impl.hsb_lite.v1"`             | `"oscillator.hsb_lite.v1"`             |
| `"hsb_lite_impl.hsb_lite_2510.v1"`          | `"hsb_lite.hsb_lite_2510.v1"`          |
| `"oscillator_impl.hsb_lite_2510.v1"`        | `"oscillator.hsb_lite_2510.v1"`        |
| `"roce_receiver_impl.hsb_lite_2510.v1"`     | `"roce_receiver.hsb_lite_2510.v1"`     |
| `"roce_data_channel_impl.hsb_lite_2510.v1"` | `"roce_data_channel.hsb_lite_2510.v1"` |

Interface `type_id`s (`"hololink.v1"` etc.) are unchanged. The locator's
duplicate-registration check in `Publisher::register_service` still distinguishes the
shorter impl strings from their interface counterparts.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD"
cmake --build "$BUILD"
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" --output-on-failure -LE hardware
```

## Phase 5 extension — `for_each_type_id` moves the hierarchy chain into the classes

**Implemented.** Supplement `construct_service` bodies used to hard-code each impl's
type_id chain — every branch listed the interface `type_id`, the impl `type_id`, and
(for per-board subclasses) the parent impl `type_id`, then called one
`ServicePublisher<T>` per key. The chain is a property of each class's inheritance, but
the supplement carried the knowledge.

Each interface and impl class now declares a single static method:

```cpp
template <typename Callback>
static bool for_each_type_id(Callback&& cb)
{
    return cb(type_id);                              // interface root
    // -- or --
    return Parent::for_each_type_id(cb) && cb(type_id);  // impl / subclass
}
```

The callback signature is `bool(const char*)`. Returning `false` halts the walk;
returning `true` continues. Chained calls propagate via `&&` short-circuit.

**Framework additions (`host/include/hololink/module/publisher.hpp`).**

- `ServicePublisher<T>::publish(instance_id, impl)` reshaped — the single-arg form
  iterates `T::for_each_type_id` and delegates to a new
  `publish(instance_id, type_id, impl)` overload for each key. Interfaces (one key) are
  unchanged in behavior; impls (chain) now publish under every key in one call.
- `Publisher::has_type_id<T>(type_id)` added as a static template query — true iff
  `type_id` appears anywhere in `T::for_each_type_id`'s emissions. Used by
  `construct_service` to gate each branch.

**Supplement `construct_service` simplification.** Each branch is now four lines (five
for the one needing an anchor lookup):

```cpp
if (Publisher::has_type_id<HololinkV1>(type_id)) {
    auto impl = std::make_shared<HololinkV1>();
    ServicePublisher<HololinkV1>(shared_from_this()).publish(instance_id, impl);
    return impl;
}
```

`SequencerInterfaceV1` and `I2cInterfaceV1` keep their bespoke construction paths (parse
`instance_id`, fetch a sibling impl, build the V1 wrapper) since their factories don't
fit the `make_shared<T>()` shape.

**Files modified.**

- `host/include/hololink/module/publisher.hpp` — reshaped `ServicePublisher<T>::publish`
  (now two overloads); added `Publisher::has_type_id`.
- All twelve V1 interface headers — added `for_each_type_id` emitting just the
  interface's own `type_id`.
- `module/core/{hololink_default,data_channel_default,roce_data_channel_default,roce_receiver_default}.hpp`
  — added `for_each_type_id` chaining to the V1 interface.
- `module/hsb_lite/module_entry.cpp` and `module/hsb_lite_2510/module_entry.cpp` — added
  `for_each_type_id` on the supplement-local impls; rewrote each
  `HsbLitePublisher::construct_service` body. 2510 per-board subclasses
  (`HsbLite2510RoceReceiverV1`, `HsbLite2510RoceDataChannelV1`) chain through their
  module_core parent impl, so a single `publish` call registers under all three keys
  (interface + module_core impl + 2510 subclass).

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD"
cmake --build "$BUILD"
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" --output-on-failure -LE hardware
```

## Phase 5 extension — `for_each_type_id` moves into `Service<X>` via `ServiceAlias` typedef

**Implemented.** The per-class `for_each_type_id` bodies (added in the previous
extension) collapse into one default in `Service<X>`. Each class declares the chain
parent via a single `using ServiceAlias = …;` typedef, and the framework's
`Service<X>::for_each_type_id` walks via SFINAE on that typedef.

`Service<X>` gains:

```cpp
template <typename Cb>
static bool for_each_type_id(Cb&& cb)
{
    if constexpr (has_service_alias<X>::value) {
        if (!X::ServiceAlias::for_each_type_id(cb)) return false;
    }
    return cb(X::type_id);
}
```

A class with no `ServiceAlias` (every V1 interface) emits just its own `type_id` and
terminates the chain — no per-class boilerplate. A class with
`using ServiceAlias = ParentClass;` walks the parent's chain first, then emits its own
`type_id`.

Per impl class, the two new lines:

```cpp
class HololinkV1 : public HololinkInterfaceV1,
                   public Service<HololinkV1> {
public:
    using ServiceAlias = HololinkInterfaceV1;       // <-- new
    static constexpr const char* type_id = "hololink.module_core.v1";
    using Service<HololinkV1>::get_service;
    using Service<HololinkV1>::for_each_type_id;     // <-- new (disambiguates)
    // ... existing methods; no for_each_type_id body
};
```

`using Service<HololinkV1>::for_each_type_id;` mirrors the existing
`using Service<HololinkV1>::get_service;` — both name-hide the inherited V1 chain
overloads that would otherwise be ambiguous against the impl's direct `Service<X>` base.

**Per-board subclasses** chain through their parent impl instead of the interface:

```cpp
class HsbLite2510RoceDataChannelV1
    : public module_core::RoceDataChannelV1,
      public Service<HsbLite2510RoceDataChannelV1> {
public:
    using ServiceAlias = module_core::RoceDataChannelV1;
    // ...
};
```

The chain walks: `RoceDataChannelInterfaceV1` → `RoceDataChannelV1` →
`HsbLite2510RoceDataChannelV1`. One typedef per class makes the whole walk declarative.

**Files modified.**

- `host/include/hololink/module/service.hpp` — added the SFINAE detector
  (`has_service_alias<U>`) and the `for_each_type_id` static template in `Service<X>`.
- All twelve V1 interface headers — dropped their five-line `for_each_type_id` bodies;
  framework now provides them.
- `module/core/{hololink_default,data_channel_default,roce_data_channel_default,roce_receiver_default}.hpp`
  — dropped the chain-walking bodies; added `using ServiceAlias = <V1>;` and
  `using Service<X>::for_each_type_id;`.
- `module/hsb_lite/module_entry.cpp` and `module/hsb_lite_2510/module_entry.cpp` — same
  shape for `HsbLiteV1`, `HsbLiteOscillatorV1`, `HsbLite2510V1`,
  `HsbLite2510OscillatorV1`, and (for the per-board subclasses)
  `HsbLite2510RoceReceiverV1`, `HsbLite2510RoceDataChannelV1`.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD"
cmake --build "$BUILD"
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" --output-on-failure -LE hardware
```

## Phase 5 extension — `module_core::HsbLitePublisher` as the canonical Publisher base

**Implemented.** The canonical HSB-Lite configuration (HsbLite, Oscillator, Hololink,
DataChannel, RoCE data channel, RoCE receiver, Sequencer, I2c, enumeration) lives in
`module/core/hsb_lite_publisher.hpp` + `module/core/hsb_lite_default.hpp`. Both
supplements (`hsb_lite` and `hsb_lite_2510`) reuse it: the canonical supplement uses it
unchanged; the 2510 supplement subclasses it and overrides only the two RoCE seams.

`Publisher::construct_service` now returns `bool` instead of `std::shared_ptr<void>`.
The framework (`Publisher::lookup` in `host/src_module/module_base.cpp`) already
discarded the returned pointer — it checked truthiness and then re-looked-up the cache
that `ServicePublisher::publish` had populated during the call. The signature is now
honest about that.

### `module_core::HsbLitePublisher`

`construct_service` is a short-circuit OR over one branch per canonical service, with
`construct_overrides` at the head as the board-extension hook:

```cpp
bool construct_service(
    const std::string& instance_id,
    const std::string& type_id) override
{
    return construct_overrides(instance_id, type_id)
        || construct_hololink(instance_id, type_id)
        || construct_data_channel(instance_id, type_id)
        || construct_roce_data_channel(instance_id, type_id)
        || construct_roce_receiver(instance_id, type_id)
        || construct_hsb_lite(instance_id, type_id)
        || construct_oscillator(instance_id, type_id)
        || construct_sequencer(instance_id, type_id)
        || construct_i2c(instance_id, type_id);
}
```

Each `construct_<service>(instance_id, type_id)` method encapsulates the `has_type_id`
check + impl construction + `ServicePublisher::publish`

- `return true` pattern. The base's `construct_overrides` returns `false`; a board
  subclass overrides it to handle anything ahead of the canonical chain (substituting
  impl classes on canonical type_ids, or adding bespoke type_ids like
  SPI/UART/board-specific services).

### Board override paths

Two equivalent ways to extend:

1. **Override `construct_overrides`** — centralized: substitutions on canonical type_ids
   and/or bespoke type_ids in one method. Runs first, so anything it handles preempts
   the canonical chain.
1. **Override individual `construct_<service>` methods** — targeted substitution of one
   canonical branch.

Both methods are virtual; boards pick whichever shape reads best.

### Supplements after the lift

- `module/hsb_lite/module_entry.cpp`: ~315 lines → ~76. Just `hololink_module_init`
  constructing the canonical Publisher + EnumerationV1; no supplement-local classes.
- `module/hsb_lite_2510/module_entry.cpp`: ~479 lines → ~239. Keeps the 2510-specific
  impl subclasses (`HsbLite2510LegacyRoceReceiver`, `HsbLite2510RoceReceiverV1`,
  `HsbLite2510RoceDataChannelV1`, `HsbLite2510DataChannel`); adds `HsbLite2510Publisher`
  overriding only `construct_roce_data_channel` + `construct_roce_receiver`.

The 2510 publisher in full:

```cpp
class HsbLite2510Publisher : public module_core::HsbLitePublisher {
protected:
    bool construct_roce_data_channel(
        const std::string& instance_id,
        const std::string& type_id) override
    {
        if (!Publisher::has_type_id<HsbLite2510RoceDataChannelV1>(type_id))
            return false;
        auto anchor = DataChannelInterfaceV1::get_service(
            this->self_module(), instance_id.c_str());
        auto impl = std::make_shared<HsbLite2510RoceDataChannelV1>(
            std::move(anchor));
        ServicePublisher<HsbLite2510RoceDataChannelV1>(shared_from_this())
            .publish(instance_id, impl);
        return true;
    }

    bool construct_roce_receiver(
        const std::string& instance_id,
        const std::string& type_id) override
    {
        if (!Publisher::has_type_id<HsbLite2510RoceReceiverV1>(type_id))
            return false;
        auto impl = std::make_shared<HsbLite2510RoceReceiverV1>();
        ServicePublisher<HsbLite2510RoceReceiverV1>(shared_from_this())
            .publish(instance_id, impl);
        return true;
    }
};
```

### Type_ids

Canonical impls use `module_core` form: `"hsb_lite.module_core.v1"`,
`"oscillator.module_core.v1"`. Per-supplement type_ids (`*.hsb_lite_2510.v1`) survive
only on classes that genuinely override behavior — the RoCE channel + receiver
subclasses.

### Build commands

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD"
cmake --build "$BUILD"
```

### Test-run commands

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" --output-on-failure -LE hardware
```

## Phase 5 extension — Publisher absorbs `EnumerationInterfaceV1`

**Implemented.** `module_core::HsbLiteEnumerationV1` is gone as a standalone class. The
Publisher itself implements `EnumerationInterfaceV1` — `HsbLitePublisher` multi-inherits
both `Publisher` and `EnumerationInterfaceV1`, and the enumeration body moves into
`HsbLitePublisher::update_metadata`. The standalone helper
`module_core::set_data_plane_metadata` (previously in
`module/core/data_plane_metadata.hpp`) also goes — its body is now the default impl of
`HsbLitePublisher::set_data_plane_metadata`, a protected virtual method.

### Override surface

Three protected virtuals on `HsbLitePublisher`:

- `std::string module_name() const = 0` — **pure**. Every supplement names itself. The
  canonical Publisher does not declare a default; every supplement subclass — including
  the canonical `hsb_lite` supplement — provides this string.
- `void set_data_plane_metadata(metadata, total_sensors, total_dataplanes, sifs_per_sensor)`
  — virtual. Default body is the lifted free-function body (computes sensor / vp_mask /
  sif_address / vp_address / hif_address / frame_end_event from the `data_plane` index).
  Override to apply a different per-data-plane address layout.
- `void publish_enumeration()` — virtual. Default registers `this` Publisher as the
  `EnumerationInterfaceV1` singleton in its own registry via the `shared_ptr` aliasing
  constructor (no RTTI). Override to register a different `EnumerationInterfaceV1` impl
  entirely.

`update_metadata` ties them together: stamps `metadata["module_name"] = module_name()`,
then calls `set_data_plane_metadata(metadata, 2, 2, 2)` with the HSB-Lite defaults. A
supplement with different inventory counts overrides `update_metadata` and passes its
own.

### Supplements after the absorption

Each supplement gains a tiny subclass — usually just a `module_name()` override. The
canonical `hsb_lite` supplement:

```cpp
class HsbLitePublisher : public module_core::HsbLitePublisher {
protected:
    std::string module_name() const override { return "hsb_lite"; }
};
```

The 2510 supplement's existing `HsbLite2510Publisher` picks up a matching
`module_name()` override (also returning `"hsb_lite"` — 2510's enumeration currently
stamps the same identifier as the canonical, and a future divergence is one line away).

Each `hololink_module_init` drops the `make_shared<HsbLiteEnumerationV1>` block and
replaces it with one call:

```cpp
publisher->publish_enumeration();
```

### Why no diamond

`Publisher` inherits `std::enable_shared_from_this<Publisher>`. `EnumerationInterfaceV1`
inherits `Service<EnumerationInterfaceV1>` → `ServiceBase` (virtually). The two bases
are disjoint — neither common ancestor is shared — so `HsbLitePublisher` picks up a
clean multi-inherited vtable with no diamond.

### Build / test

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD"
cmake --build "$BUILD"
ctest --test-dir "$BUILD" --output-on-failure -LE hardware
```

## Phase 5 extension — Publisher bootstrap absorbed into the constructor

**Implemented.** `Publisher`'s new init-taking constructor builds the host `Module` from
`init->{get_service,release_service}`, fetches `LoggingInterfaceV1` (wiring the
per-binary `HSB_LOG` cache), and fetches `ReactorV1`. Three accessors expose the
bootstrap state: `host_module()`, `logger()`, `reactor()`. Module-side classes inherit
the constructor via `using Publisher::Publisher;` so
`std::make_shared<MyPublisher>(init)` runs the bootstrap during construction.

The constructor validates `init` inline — null pointer, mismatched `api_version`, or
missing `get_service`/`release_service` callbacks all throw `std::runtime_error`.
Supplements lose the inline `if (!init || …) return INVALID_PARAMETER;` block; all
init-time failures now route through the existing `catch (...)` and map to
`MODULE_INIT_FAILED`. The host-side `adapter.cpp` doesn't switch on these codes, so
collapsing two init-failure status codes into one is observable but inert.

The default `Publisher()` constructor stays — it's the construction path for the
host-side `HostPublisher` in `host/src/adapter.cpp`, which lives in the host process
itself (no `hololink_module_init_t` to bootstrap from). Module-side Publishers use the
init-taking ctor.

### `HsbLitePublisher::publish_frame_metadata`

Parallel to `publish_enumeration()`. Default constructs `module_core::FrameMetadataV1`
and registers it as `FrameMetadataInterfaceV1` in the Publisher's own registry. A
supplement subclass overrides to publish a different impl.

### Supplements + test stubs after the lift

Each `hololink_module_init` shrinks. Take `hsb_lite`'s:

```cpp
extern "C" hololink_module_services_t
hololink_module_init(const hololink_module_init_t* init)
{
    hololink_module_services_t result {};
    try {
        auto publisher = std::make_shared<HsbLitePublisher>(init);
        g_publisher = publisher;
        publisher->publish_frame_metadata();
        publisher->publish_enumeration();
        result = g_publisher->callbacks();
    } catch (const std::exception&) {
        result.status = HOLOLINK_MODULE_INIT_FAILED;
    }
    return result;
}
```

`g_host_module`, `g_logger`, `g_frame_metadata` statics gone. The `hsb_lite_2510`
supplement is the same shape. The four test stubs shed the bootstrap as well; the
singletons stub also drops its explicit `ReactorV1::get_service` call — it uses
`publisher->reactor()->add_callback(...)` instead.

### Build / test

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD"
cmake --build "$BUILD"
ctest --test-dir "$BUILD" --output-on-failure -LE hardware
```

## Phase 5 extension — HsbLite2510 variations promoted into `module/core/`

**Implemented.** The HsbLite2510-specific impls move from the `hsb_lite_2510` supplement
into `hololink_module/module/core/` under the `hololink::module::module_core` namespace.
Future partner supplements targeting pre-0x2603 / pre-0x2602 FPGA revisions reuse this
dispatch directly.

`Legacy` dropped from the operator subclass name — `HsbLite2510LegacyRoceReceiver` is
now just `module_core::HsbLite2510RoceReceiver`. Per-impl type_ids
(`roce_receiver.hsb_lite_2510.v1`, `roce_data_channel.hsb_lite_2510.v1`) preserved
unchanged — the type_id describes the impl variant, not the C++ namespace.

### New headers in `module/core/`

- `hsb_lite_2510_roce_receiver.hpp` — `module_core::HsbLite2510RoceReceiver` (subclass
  of `hololink::operators::RoceReceiver` with the pre-0x2603 PSN-decode overrides) +
  `module_core::HsbLite2510RoceReceiverV1` (subclass of `module_core::RoceReceiverV1`
  whose `make_receiver` dispatches on `hsb_ip_version`).
- `hsb_lite_2510_data_channel.hpp` — `module_core::HsbLite2510DataChannel` (subclass of
  `hololink::DataChannel` with the pre-0x2602 RDMA-page register layout) +
  `module_core::HsbLite2510RoceDataChannelV1` (subclass of
  `module_core::RoceDataChannelV1` whose `make_backing` dispatches on `hsb_ip_version`).
  Carries direct includes for `<thread>`, `<chrono>`, `<limits>`, `<stdexcept>`,
  `<fmt/format.h>` — the original supplement header relied on transitives.
- `hsb_lite_2510_publisher.hpp` — `module_core::HsbLite2510Publisher` (concrete subclass
  of `module_core::HsbLitePublisher` with `module_name() = "hsb_lite_2510"` and
  `construct_roce_*` overrides that route to the two V1 wrappers above).

### Supplement shrinkage

`module/hsb_lite_2510/module_entry.cpp` is now ~33 lines — just the static `g_publisher`
and the C-ABI `hololink_module_init` that instantiates
`module_core::HsbLite2510Publisher` and calls `setup()`. No supplement-local class
definitions remain.

The companion file `module/hsb_lite_2510/hsb_lite_2510_data_channel.hpp` is deleted; its
content moved to `module/core/hsb_lite_2510_data_channel.hpp`.

### Build / test

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD"
cmake --build "$BUILD"
ctest --test-dir "$BUILD" --output-on-failure -LE hardware
```

## Phase 6 extension — `LinuxReceiverOp` module operator + `LinuxReceiverV1` / `LinuxDataChannelInterfaceV1`

**Implemented.** The module previously shipped only the hardware `RoceReceiverOp`. This
change ports the legacy `hololink::operators::LinuxReceiverOp` — a software receiver
that reassembles HSB's RoCEv2 UDP stream in user space — to the module V1 surface, so
the operators tree has a receiver that runs on hosts with no infiniband device
(`HOLOLINK_BUILD_ROCE=OFF`). The wire protocol is still RoCEv2; the host side is a
datagram socket rather than an ibverbs QP, which drives two transport differences from
the hardware path (device frame-memory address `0` + the host socket's `local_port`,
versus the receiver's device address + the fixed RoCEv2 port).

- `host/include/hololink/module/linux_receiver.hpp` — new abstract
  `LinuxReceiverInterfaceV1` (alias `LinuxReceiverV1`), a
  `ConfigurableService<LinuxReceiverInterfaceV1>` with `type_id = "linux_receiver.v1"`.
  `start(...)` takes the bound data-socket fd + CUDA-buffer / frame-layout parameters
  individually (no ibv_name / ibv_port / peer_ip — the socket is the transport, and
  `received_address_offset` is `cu_buffer` since HSB writes from address 0).
  `get_next_frame(timeout_ms, info, cuda_stream)` carries the stream as an opaque
  `void*` so the header keeps no `<cuda.h>` dependency. The getter surface adds
  `local_port()` and drops `external_frame_memory()`. `LinuxReceiverFrameInfoV1` carries
  the software-decoded per-frame metadata directly (timestamps, crc, psn, bytes_written,
  …) plus the receiver's packet / byte / drop counters — there is **no** device-side EOF
  block to copy back, unlike the RoCE path.
- `host/include/hololink/module/linux_data_channel.hpp` — new
  `LinuxDataChannelInterfaceV1` (`type_id = "linux_data_channel.v1"`), the
  software-transport sibling of `RoceDataChannelInterfaceV1` over the same
  `DataChannelInterfaceV1` anchor. Adds `configure_socket(fd)`; `attach_receiver` issues
  `authenticate(qp, rkey)` then
  `configure_roce(0, frame_size, page_size, pages, local_port)`.
- `module/core/linux_receiver_default.hpp` — `LinuxReceiverV1` wraps a (initially-null)
  `shared_ptr<hololink::operators::LinuxReceiver>` constructed via a protected virtual
  `make_receiver(...)` hook. `get_next_frame` translates the legacy
  `LinuxReceiverMetadata` (including its already-decoded `frame_metadata`) into
  `LinuxReceiverFrameInfoV1` and forwards the `CUstream`; `local_port()` reads the bound
  socket via `getsockname`.
- `module/core/linux_data_channel_default.hpp` — `LinuxDataChannelV1` composes the
  anchor and a backing legacy `hololink::DataChannel` (same `make_backing` hook as the
  RoCE channel, so the 2510 `HsbLite2510DataChannel` override dispatches here too).
- `module/core/hsb_lite_publisher.hpp` — `construct_service` chain grows
  `construct_linux_data_channel` (inline, ungated) and `construct_linux_receiver`
  (declared here, defined out-of-line). `module/core/linux_receiver_construct.cpp` —
  out-of-line `HsbLitePublisher::construct_linux_receiver`, compiled per-board like the
  RoCE construct TU but **not** gated on `HOLOLINK_BUILD_ROCE` (the software receiver
  needs no ibverbs and always publishes).
- `module/core/CMakeLists.txt` — `hololink::module_core` now compiles
  `src/hololink/operators/linux_receiver/linux_receiver.cpp` directly (and PUBLIC-links
  `CUDA::cuda_driver`) rather than linking the legacy
  `hololink::operators::linux_receiver` target, which would drag `base_receiver_op` →
  Holoscan into every module `.so`. The per-board module CMakeLists (`hsb_lite`,
  `hsb_lite_2510`, `leopard_vb1940`) add `../core/linux_receiver_construct.cpp` to their
  `SOURCES`.
- `host/operators/linux_receiver_op.{hpp,cpp}` — `LinuxReceiverOp`, a
  `holoscan::Operator` mirroring `RoceReceiverOp`. `start()` resolves the channel +
  receiver (+ `FrameMetadataInterfaceV1` for `block_size()` only) from the metadata,
  allocates the frame buffer, creates a datagram socket and binds it via
  `channel->configure_socket(fd)`, starts the receiver, then
  `channel->attach_receiver(receiver)`, and spawns the worker thread (running
  `blocking_monitor()` with the configured CPU affinity). `compute()` polls
  `get_next_frame` on a Holoscan-allocated stream and stamps the decoded metadata fields
  directly (no `FrameMetadataInterfaceV1::decode`). A `receiver_affinity` parameter
  defaults from `HOLOLINK_AFFINITY`.
- `host/operators/CMakeLists.txt` — `hololink_operators` is now built whenever
  `HOLOLINK_MODULE_BUILD_OPERATORS` is on (always carries `linux_receiver_op.cpp`); the
  RoCE operator TU is added only with `HOLOLINK_BUILD_ROCE`.
- `host/operators/python/operators_py.cpp` — `LinuxReceiverOp` pybind binding registered
  unconditionally (the RoCE binding stays `#ifdef HOLOLINK_BUILD_ROCE`); the operators
  package re-export already surfaces whatever the extension registered.
- `host/python/hololink_module_py.cpp` — binds the abstract `csi::CsiConverterV1` (with
  a trampoline so Python can subclass it), a pure module type so the core pybind gains
  no legacy dependency. The `csi::PixelFormat` enum it references is *not* re-registered
  here — the per-sensor modules (e.g. `sensors/vb1940`) already register it, and pybind
  keys types by C++ type, so a second registration would fail; the trampoline marshals
  it via that registration at call time. The C++ `Vb1940Cam::configure_converter` takes
  a `CsiConverterV1`, so a Python caller must pass a subclass; the pure-Python
  `Imx274Cam` duck-types and accepts the same object.
- `examples/legacy_csi_converter.py` — Python sibling of
  `examples/legacy_csi_converter.hpp`: a
  `LegacyCsiConverter(hololink_module.CsiConverterV1)` that wraps the legacy
  `CsiToBayerOp`, forwards the four converter methods, and translates `PixelFormat` by
  value. The application-layer bridge keeps the legacy CSI reference out of the module
  and its sensor packages; the module example players construct it and hand it to
  `configure_converter` instead of the raw `CsiToBayerOp`.
- `examples/module_linux_imx274_player.{cpp,py}` +
  `examples/module_linux_vb1940_player.{cpp,py}` — the IMX274 and single-camera VB1940
  pipelines driven through `LinuxReceiverOp` (drop-in swap for `RoceReceiverOp`); the
  Python players feed `configure_converter` through `examples/legacy_csi_converter.py`.
  `examples/CMakeLists.txt` registers the C++ players under
  `HOLOLINK_MODULE_BUILD_OPERATORS` (not `HOLOLINK_BUILD_ROCE`).
- `tests/CMakeLists.txt` — `module_linux_imx274_player_test` /
  `module_linux_vb1940_player_test` C++ smoke tests (gated on
  `HOLOLINK_MODULE_BUILD_OPERATORS`, runtime-gated on `HOLOLINK_TEST_IMX274` /
  `HOLOLINK_TEST_VB1940`, labels `imx274` / `vb1940`).
  `tests/test_module_linux_imx274_player.py` (`@pytest.mark.skip_unless_imx274`) and
  `tests/test_module_linux_vb1940_player.py` (`@pytest.mark.skip_unless_vb1940`) drive
  the Python players — neither carries `@pytest.mark.accelerated_networking`, so they
  run in RoCE-off configurations.

### Build / test

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD"
cmake --build "$BUILD" --target module_linux_imx274_player module_linux_vb1940_player
# Software-receiver smoke tests run only with hardware + the per-sensor switch:
HOLOLINK_TEST_IMX274=1 ctest --test-dir "$BUILD" -R module_linux_imx274_player_test
HOLOLINK_TEST_VB1940=1 ctest --test-dir "$BUILD" -R module_linux_vb1940_player_test
```

## Phase 6 extension — native module `CsiToBayerOp` (retires the `LegacyCsiConverter` shim)

**Implemented.** The module sensors interpret received CSI data through a
`CsiConverterV1`; until now the only implementation was the legacy
`hololink::operators::CsiToBayerOp`, bridged into every example through the
`examples/legacy_csi_converter.{hpp,py}` shim (`LegacyCsiConverter`). The module now
ships a native converter operator, so the examples drop the shim and hand the operator
straight to `configure_converter()`.

- `host/operators/include/hololink/module/operators/csi_to_bayer_op.hpp` +
  `host/operators/csi_to_bayer_op.cpp` — `hololink::module::operators::CsiToBayerOp`,
  both a `holoscan::Operator` and a `hololink::module::csi::CsiConverterV1` (the
  dual-role pattern `FusaCoeCaptureOp` uses). It resolves no module service (no
  `enumeration_metadata`); the constructor mirrors the legacy operator (`allocator`,
  `cuda_device_ordinal`, `out_tensor_name`, `sub_frame_rows`) and exposes
  `get_csi_length()` / `get_sub_frame_size()`. The CSI→Bayer engine (the NVRTC
  `frameReconstruction8/10/12` kernels, the sub-frame accumulation in `compute()`, and
  the four geometry helpers) is ported from the legacy operator but expressed against
  `hololink::module::csi::PixelFormat`, so no legacy CSI type appears in its API — the
  two copies are independent (the legacy operator is untouched). Its CUDA helpers are
  module-owned too: `host/operators/cuda_function_launcher.{hpp,cpp}` vendors
  `hololink::module::operators::CudaFunctionLauncher` / `CudaContextScopedPush`
  (behavior-identical ports of the legacy `hololink::common` helpers), error checking
  goes through `HOLOLINK_MODULE_CUDA_CHECK` and the `UniqueCUdeviceptr` alias comes from
  `cuda_unique.hpp`, and it uses the module `round_up` from `page_size.hpp`. The
  operator therefore carries no source or link dependency on legacy `hololink::core` /
  `hololink::common`.
- `host/operators/CMakeLists.txt` — `csi_to_bayer_op.cpp` and the vendored
  `cuda_function_launcher.cpp` join the always-built `hololink_operators` library
  (alongside `linux_receiver_op.cpp`; no capability gate), which links `CUDA::nvrtc`
  (for NVRTC compilation) in addition to `CUDA::cuda_driver`. Legacy `hololink::core` is
  now linked only when FUSA is built (its operator header pulls in
  `hololink/core/csi_controller.hpp`); the framework-only configuration of
  `hololink_operators` / `_hololink_module_operators` links no legacy core.
- `host/operators/python/operators_py.cpp` — `CsiToBayerOp` pybind binding registered
  unconditionally, listing both `holoscan::Operator` and `csi::CsiConverterV1` as bases
  so a Python-constructed op converts to `shared_ptr<CsiConverterV1>` for
  `configure_converter`. Its `configure` / `transmitted_line_bytes` take the pixel
  format as a plain integer (the enumerator value) — not the bound
  `hololink_module.sensors.vb1940.PixelFormat` enum — so the pure-Python `Imx274Cam` can
  pass `csi.PixelFormat.value` at the boundary without importing the C++ enum.
- `python/sensors/imx274/imx274_cam.py::configure_converter` drops the legacy
  `hololink.PixelFormat` translation and passes the module pixel-format value directly
  to the converter. The C++ `Imx274Cam` / `Vb1940Cam` already pass the module
  `csi::PixelFormat`, so they are unchanged.
- Example players migrated from `LegacyCsiConverter(legacy CsiToBayerOp)` to the native
  operator: Python `module_linux_imx274_player.py`, `module_linux_vb1940_player.py`,
  `module_quad_imx274_player.py`; C++ `module_imx274_player.cpp`,
  `module_linux_imx274_player.cpp`, `module_quad_imx274_player.cpp`,
  `module_vb1940_player.cpp`, `module_linux_vb1940_player.cpp`,
  `module_stereo_vb1940_player.cpp`. `examples/legacy_csi_converter.{hpp,py}` and its
  `examples/CMakeLists.txt` entry are removed. (The Phase-6-extension `LinuxReceiverOp`
  section above still describes the shim as it stood then — this section supersedes it.)
- The existing `tests/test_module_linux_imx274_player.py` /
  `tests/test_module_linux_vb1940_player.py` need no change: they drive the example
  `main()` and pass no converter-specific arguments, so they exercise the native
  operator once the examples migrate.

### Build / test

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD" -DHOLOLINK_MODULE_BUILD_OPERATORS=ON
cmake --build "$BUILD" --target hololink_operators _hololink_module_operators
python3 -c "import hololink_module.operators as o; assert hasattr(o, 'CsiToBayerOp')"
```

## Phase 6 extension — native module `ImageProcessorOp`

**Implemented.** The module example players had been migrated onto the native
`CsiToBayerOp`, but still reached into the legacy tree for the next pipeline stage —
`hololink::operators::ImageProcessorOp` (optical-black correction + Grey-World auto
white-balance). It was the only remaining legacy operator those players used; in the
three Python players it was the only consumer of `import hololink`. The module now ships
a native sibling, so every module example runs an all-module pipeline.

- `host/operators/include/hololink/module/operators/image_processor_op.hpp` +
  `host/operators/image_processor_op.cpp` —
  `hololink::module::operators::ImageProcessorOp`, a **plain** `holoscan::Operator`
  (unlike `CsiToBayerOp` it implements no module interface — it is not a converter). The
  constructor mirrors the legacy operator exactly (`pixel_format`, `bayer_format`,
  `optical_black`, `cuda_device_ordinal`). The engine (the NVRTC `applyBlackLevel` /
  `histogram` / `calcWBGains` / `applyOperations` kernels, the histogram/shared-memory
  sizing in `start()`, and the sub-frame-aware white-balance accumulation in
  `compute()`) is ported verbatim from the legacy operator. The `pixel_format` /
  `bayer_format` params stay `int` but are interpreted against
  `hololink::module::csi::PixelFormat` / `BayerFormat` (identical enumerator values), so
  no legacy CSI type appears in the operator — the two copies are independent (the
  legacy operator is untouched). It reuses the module-owned CUDA helpers
  (`CudaFunctionLauncher` / `CudaContextScopedPush` from
  `hololink/module/operators/cuda_function_launcher.hpp`, and `UniqueCUdeviceptr` /
  `HOLOLINK_MODULE_CUDA_CHECK` from `hololink/module/cuda_unique.hpp`) and the module
  `HSB_LOG_*` logging — so it links no legacy `hololink::core`.
- `host/operators/CMakeLists.txt` — `image_processor_op.cpp` joins the always-built
  `hololink_operators` library (alongside `csi_to_bayer_op.cpp` /
  `linux_receiver_op.cpp`; no capability gate). No new link deps or compile definitions
  are needed.
- `host/operators/python/operators_py.cpp` — `ImageProcessorOp` pybind binding
  registered unconditionally, listing only `holoscan::Operator` as a base (no converter
  interface). Because the public API is plain `int` params (no enum), the binding needs
  no int-coercion lambda (unlike `CsiToBayerOp.configure`).
- Example players migrated from the legacy `ImageProcessorOp` to the native operator
  (constructor arguments unchanged): Python `module_linux_imx274_player.py`,
  `module_linux_vb1940_player.py`, `module_quad_imx274_player.py` — each also drops its
  now-unused `import hololink as hololink_module`, leaving the Python module players
  with no legacy `import hololink`; C++ `module_imx274_player.cpp`,
  `module_linux_imx274_player.cpp`, `module_quad_imx274_player.cpp`,
  `module_vb1940_player.cpp`, `module_linux_vb1940_player.cpp`,
  `module_stereo_vb1940_player.cpp`, and `module_fusa_coe_vb1940_player.cpp` (the FUSA
  player runs `ImageProcessorOp` after `FusaCoeCaptureOp`, so it was outside the
  `CsiToBayerOp` migration but in scope here).
- The existing `tests/test_module_linux_imx274_player.py` /
  `tests/test_module_linux_vb1940_player.py` need no change: they drive the example
  `main()` and pass no operator-specific arguments, so they exercise the native operator
  once the examples migrate.

### Build / test

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD" -DHOLOLINK_MODULE_BUILD_OPERATORS=ON
cmake --build "$BUILD" --target hololink_operators _hololink_module_operators
python3 -c "import hololink_module.operators as o; assert hasattr(o, 'ImageProcessorOp')"
```

## Phase 6 extension — native module `PackedFormatConverterOp` + module utilities

**Implemented.** Closes the remaining legacy `hololink` references in the module example
players and tests: the FUSA example's packed-CSI converter, the legacy `CudaCheck` /
`env_hololink_ip` / `MacAddress` host utilities, and legacy operators + `hsb_log_*` in
`tests/test_module_vb1940.py`. The C++ module players now carry no legacy `hololink`
includes.

- `host/operators/include/hololink/module/operators/packed_format_converter_op.hpp` +
  `host/operators/packed_format_converter_op.cpp` —
  `hololink::module::operators::PackedFormatConverterOp`, both a `holoscan::Operator`
  and a `hololink::module::csi::CsiConverterV1` (dual-role, like `CsiToBayerOp`). The
  unpack engine (the NVRTC `packed8bitTo16bit` / `packed10bitTo16bit` /
  `packed12bitTo16bit` kernels) is ported verbatim from the legacy
  `hololink::operators::PackedFormatConverterOp` but expressed against
  `hololink::module::csi::PixelFormat`. Constructor
  `allocator, cuda_device_ordinal, in_tensor_name, out_tensor_name`; public
  `get_frame_size()` + the four `CsiConverterV1` methods (`received_line_bytes` =
  `hololink::module::round_up(…, 64)`). It reuses the module-owned CUDA helpers
  (`cuda_function_launcher.hpp`, `cuda_unique.hpp` / `HOLOLINK_MODULE_CUDA_CHECK`) — no
  legacy `hololink::core` / `hololink::common`.
- `host/operators/CMakeLists.txt` — `packed_format_converter_op.cpp` joins the
  always-built `hololink_operators` library (no capability gate).
  `host/operators/python/operators_py.cpp` registers it unconditionally, listing both
  `holoscan::Operator` and `csi::CsiConverterV1` as bases, with the plain-integer
  pixel-format lambdas `CsiToBayerOp` uses.
- `FusaCoeCaptureOp::configure_converter` now takes
  `hololink::module::csi::CsiConverterV1&` (was the legacy
  `hololink::csi::CsiConverter&`), so the native converter is handed to it directly;
  `core_.pixel_format()` (legacy enum) is cast across to the module enum (identical
  values).
- New public module utility headers: `host/include/hololink/module/tools.hpp`
  (header-only inline `env_hololink_ip`) and
  `host/include/hololink/module/networking.hpp` (`MacAddress` alias and the module-owned
  `DEFAULT_MTU` constant — a mirror of the legacy `hololink::core` value so callers name
  no legacy constant).
- Module Python convenience bindings in `host/python/hololink_module_py.cpp`:
  module-level `hsb_log_trace/debug/info/warn/error` (route through the registered HSB
  logger; no-op when unset), the module-owned `DEFAULT_MTU` constant (mirror of the
  legacy `hololink::core` value), and a RoCE-gated `infiniband_devices()` (beside
  `ibv_device_for_peer`). Exported from `host/python/__init__.py`.
- Example migration: all 7 C++ `module_*` players swap `<hololink/common/tools.hpp>` →
  `"hololink/module/tools.hpp"` (`env_hololink_ip` →
  `hololink::module::env_hololink_ip`); the 6 non-FUSA players swap
  `<hololink/common/cuda_helper.hpp>` → `"hololink/module/cuda_unique.hpp"` (`CudaCheck`
  → `HOLOLINK_MODULE_CUDA_CHECK`); the FUSA player swaps
  `<hololink/core/networking.hpp>` → `"hololink/module/networking.hpp"`
  (`hololink::core::MacAddress` → `hololink::module::MacAddress`) and the legacy
  `PackedFormatConverterOp` for the native one. `examples/CMakeLists.txt` drops the
  legacy operator links (`hololink::operators::csi_to_bayer` / `::image_processor` /
  `::packed_format_converter`) from the migrated `module_*` targets, which now resolve
  the native operators via the `hololink::operators` module alias. The same targets also
  drop the now-stale legacy `hololink` (core) and
  `hololink::sensors::native_*_camera_sensor` links — they were leftovers from the
  pre-module players (used only in a doc comment, and not required transitively since
  the module sensor libs are standalone) — so each `module_*` player links only
  module-defined libraries plus the Holoscan SDK. The legacy example players
  (`imx274_player`, `vb1940_player`, `fusa_coe_*`, …) keep their legacy links unchanged.
- `tests/test_module_vb1940.py` moves onto `hololink_module.operators.CsiToBayerOp` /
  `ImageProcessorOp` and `hololink_module.hsb_log_*`, dropping `import hololink`.
  `tests/conftest.py` additionally logs through `hololink_module.hsb_log_info` in
  `report_test_name`. Its legacy `Reactor`/`Hololink.reset_framework` finalizers and
  `infiniband_devices` collection logic stay legacy — the module's service-based
  architecture has no global framework-reset to call, and those paths must stay
  authoritative for the legacy tests conftest also serves.

### Build / test

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD" -DHOLOLINK_MODULE_BUILD_OPERATORS=ON
cmake --build "$BUILD" --target hololink_operators _hololink_module_operators _hololink_py_module
python3 -c "import hololink_module.operators as o; assert hasattr(o, 'PackedFormatConverterOp')"
python3 -c "import hololink_module as h; assert hasattr(h, 'hsb_log_info')"
```

## Phase 6 extension — namespace-mirroring sensor header layout

The C++ sensor drivers now expose their public headers through a namespaced `include/`
root, matching the convention the operators library already uses
(`include/hololink/module/operators/…`). Previously each sensor target put its leaf
directory directly on the consumer include path, so callers wrote a bare
`#include "imx274_cam.hpp"`. Now the path mirrors the C++ namespace
(`hololink::module::sensors::imx274`), so it reads
`#include "hololink/module/sensors/imx274/imx274_cam.hpp"` — self-describing and
collision-proof across sensors.

- `host/sensors/imx274/` — `imx274_cam.hpp`, `imx274_mode.hpp`, and
  `li_i2c_expander.hpp` move under `include/hololink/module/sensors/imx274/`.
  `imx274_cam.cpp` stays put. `host/sensors/imx274/CMakeLists.txt` switches the target's
  `$<BUILD_INTERFACE>` include root from `${CMAKE_CURRENT_SOURCE_DIR}` to
  `${CMAKE_CURRENT_SOURCE_DIR}/include`.
- `host/sensors/vb1940/` — `vb1940_cam.hpp` moves under
  `include/hololink/module/sensors/vb1940/`; same one-line CMake include-root change.
  `vb1940_cam.cpp` and `python/vb1940_py.cpp` stay put.
- Include updates: each driver's `.cpp`, the cross-header includes inside
  `imx274_cam.hpp` (its `imx274_mode.hpp` / `li_i2c_expander.hpp` pulls), the `vb1940`
  pybind TU, and the 7 `module_*` example players adopt the prefixed path. The legacy
  `src/hololink/sensors/…` drivers keep their own leaf-dir convention, untouched.

### Installing the module C++ surface for external applications

The sensor drivers and the runtime they need are now exported so an out-of-tree project
can build against them:

```cmake
find_package(Hololink REQUIRED)
# the consumer finds CUDA / fmt itself — the package config carries no
# find_dependency calls (project-wide convention; see HololinkConfig.cmake.in)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(my_app PRIVATE
    hololink::module                  # host runtime (Adapter, services, HSB logger)
    hololink::module::sensors::imx274 # or ::vb1940
)
```

The module targets join the project-wide `HololinkTargets` export set
(`cmake/HololinkExport.cmake`, written when `-DHOLOLINK_BUILD_EXPORT=ON`), namespaced
like the legacy operator/sensor targets:

- `host/CMakeLists.txt` — exports `hololink::module_headers` (the public
  `hololink/module/**` header tree), `hololink::module_runtime` (defines the
  `hsb_logger_cache` symbol the `HSB_LOG_*` macros resolve against — so the sensor
  archives are not independently linkable without it), and `hololink::module` (host
  runtime; its static-propagated `$<LINK_ONLY>` deps are `CUDA::cuda_driver` +
  `hololink::module_runtime`).
- `host/sensors/{imx274,vb1940}/CMakeLists.txt` — export
  `hololink::module::sensors::{imx274,vb1940}` and install their `include/hololink/…`
  trees.
- `module/leopard_vb1940/CMakeLists.txt` — exports `hololink::leopard_vb1940::headers`
  (a PUBLIC usage requirement of the VB1940 driver) and installs its header tree.
- No `find_dependency` is added to `HololinkConfig.cmake.in`: the existing package
  already exports imported-target references (`holoscan::core`, `CUDA::*`,
  `fmt::fmt-header-only`) without them, leaving the consumer responsible for those
  `find_package` calls. The module follows suit.
- Not (yet) exported: the Holoscan-coupled operators (`hololink::operators`) and
  per-board module `.so`s — out of scope for the sensor-driver surface.

### Build / test

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD" -DHOLOLINK_BUILD_EXPORT=ON
cmake --build "$BUILD" --target hololink_module_sensors_imx274 hololink_module_sensors_vb1940
# one component installs the whole module graph (headers, runtime, supplements,
# sensors) so a single-component install yields a complete dependency graph:
DESTDIR=/tmp/stage cmake --install "$BUILD" --component hololink-module
# verify the namespaced headers + package config landed:
test -f /tmp/stage/usr/local/include/hololink/module/sensors/imx274/imx274_cam.hpp
test -f /tmp/stage/usr/local/lib/cmake/hololink/HololinkConfig.cmake
```

## Phase 5 extension — `hsb_lite_2510` declines unsupported FPGA IP versions

**Problem.** `module/hsb_lite_2510/` ships as the bare `hololink_<UUID>.so`, so the
Adapter loader routes every HSB-Lite board without a dedicated compat-suffixed `.so`
there — including newer silicon this build predates. But that module only drives FPGAs
whose IP version is `0x2510` through `0x2603` (inclusive); it cannot drive anything
newer (or older). A board reporting no compat-id but newer silicon would previously land
on the 2510 module and be mis-driven. (Returning a hard error from `update_metadata` was
not an option: in the bootp-listener path that exception escapes the Reactor thread and
calls `std::terminate`, killing the whole host.)

**Implemented.** `update_metadata` may now return a third outcome,
`HOLOLINK_MODULE_ENUMERATION_SKIPPED` (`status.h`), meaning "the module recognizes the
device but declines to drive it." It is neither success nor a hard error:

- `Adapter::enumerate` (`host/src/adapter.cpp`) treats the skip status as "suppress the
  announcement" — post-enrichment subscribers (`register_ip` / `register_all`,
  `wait_for_channel`) are *not* notified, so the application never sees the unsupported
  device, and nothing throws. (Raw subscribers still fire, since they run before the
  module is consulted.) It logs one `HSB_LOG_INFO` line noting which module skipped
  which `fpga_uuid`, capped at 4 emissions total so a device re-announcing on every
  bootp broadcast doesn't flood the log; on the 4th it emits a final notice that further
  skip messages are suppressed.
- `HsbLite2510Publisher::update_metadata` (`module/core/hsb_lite_2510_publisher.hpp`)
  gates on `metadata["hsb_ip_version"]`: a version outside `0x2510`–`0x2603` (newer or
  older) logs one `HSB_LOG_INFO` line reporting the device serial number and the
  unsupported IP version, then returns `HOLOLINK_MODULE_ENUMERATION_SKIPPED`. Supported
  versions delegate to `HsbLitePublisher::update_metadata` unchanged. The notice is
  capped at 4 emissions *per serial number* (the 4th followed by a final "further
  notices suppressed" line), so each distinct unsupported board is reported a few times
  and then goes quiet. When `hsb_ip_version` is absent from the metadata the module
  makes no judgment and delegates to the base (preserving prior behavior).

The skip path is distinct from the existing "no module matched" path (where the bootp
metadata is still dispatched un-enriched): a skip means a module *was* loaded and
explicitly declined, so the announcement is withheld entirely.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD" -DHOLOLINK_BUILD_PYTHON=ON
cmake --build "$BUILD" --target hsb_lite_2510
```

## Phase 5 extension — one per-sensor formula: `update_metadata` delegates to `use_sensor`

**Problem.** Per-sensor addressing was computed two different ways. Enumerate-time
enrichment (`HsbLitePublisher::update_metadata`) called the protected
`set_data_plane_metadata`, which stamped `hif_address`. The application path
(`HsbLiteChannelConfigurationV1::use_sensor`) instead round-tripped through the legacy
`hololink::DataChannel::use_sensor` strategy dispatch, which does **not** stamp
`hif_address` (legacy splits that into a separate `use_data_plane_configuration` call
the module never made). The two paths agreed on every field they both wrote — but
`use_sensor` left `hif_address` at its previously-enumerated value. Selecting a sensor
on a different data plane (`Adapter::use_sensor(metadata, n)`) therefore re-pointed
`sensor` / `sif_address` / `vp_address` at sensor *n* while `hif_address` stayed stale —
a latent mis-addressing bug on HSB-Lite (1:1 sensor↔data-plane). Leopard had already
worked around the duplication with its own `configure_sensor_metadata` + a
back-pointer'd channel-config; canonical HSB-Lite had not.

**Implemented.** A single formula, one chokepoint:

- `module_core::hsb_lite_sensor_metadata(metadata, sensor_number, data_plane, sifs_per_sensor)`
  (`module/core/hsb_lite_publisher.hpp`) is now a free function — the single source of
  truth for the HSB-shape address math (`sensor`, `vp_mask`, `sif_address`,
  `vp_address`, `hif_address`, `data_channel`, `frame_end_event`). It replaces the
  protected `HsbLitePublisher::set_data_plane_metadata` virtual, which is removed.
- `HsbLiteChannelConfigurationV1::use_sensor` calls the formula directly with the
  HSB-Lite layout (`data_plane == sensor_number`, two SIFs per sensor) after a `[0, 3)`
  range check, so **`hif_address` is recomputed on every sensor selection**. It no
  longer touches legacy `DataChannel::use_sensor`; `use_mtu` stamps `mtu` inline. The
  class stays stateless. Consequently `setup()` no longer calls
  `Enumerator::configure_default_enumeration_strategies()` (that existed only to feed
  the legacy `use_sensor` dispatch), and the
  `hololink/core/{data_channel,enumerator,metadata}` includes are dropped.
- `HsbLitePublisher::update_metadata` stamps `module_name` and then **delegates to its
  module's `ChannelConfigurationInterfaceV1::use_sensor`** for the default (bootp)
  sensor — `metadata["data_plane"]`. Because `use_sensor` is virtual and each board
  publishes its own channel-config, the board's own per-sensor formula runs, and the
  enumerate and application paths can no longer drift. The channel-config is published
  in `setup()` before any enumerate, so the lookup always resolves.
- `module/leopard_vb1940/` collapses accordingly: `LeopardVb1940ChannelConfigurationV1`
  loses its publisher back-pointer (and the forward declaration / `friend`), and its
  `use_sensor` stamps the Leopard layout (`data_plane=0`, one SIF per sensor, camera
  `i2c_bus`) via the shared formula. `LeopardVb1940Publisher` drops both its
  `configure_sensor_metadata` and its `update_metadata` override — the inherited base
  `update_metadata` delegates to Leopard's `use_sensor` automatically.

A side effect of routing enumerate through `use_sensor` is that the sensor index is now
range-checked at enumerate time (`[0, 3)` for HSB-Lite); the old
`set_data_plane_metadata` path did no validation. Bootp data planes are always in range,
so this only rejects malformed input. The `hsb_lite_2510` module is unchanged: its
`update_metadata` still gates on `hsb_ip_version`, then calls the base, which now
delegates — so it inherits the `hif_address` fix for free.

Tests (`tests/hololink_module_hsb_lite_test.cpp`): `UseSensorRecomputesHifAddress`
guards the bug, `EnumerateMatchesUseSensorFieldForField` pins the no-drift invariant,
and `UseSensorRejectsOutOfRange` covers the validation.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD" -DHOLOLINK_BUILD_PYTHON=ON
cmake --build "$BUILD" --target hsb_lite_2510
cmake --build "$BUILD" --target leopard_vb1940
ctest --test-dir "$BUILD" --output-on-failure -R HololinkAdapterHsbLite
```

## Phase 6 extension — receiver frame buffer outlives in-flight tensors

**Implemented.** Ports the legacy fix (`53a63b34`, "Fix use-after-free of receiver frame
buffer for in-flight frames") into the module receiver operators. Stopping a receiver
freed its frame buffer while in-flight frames still pointed into it, so the GPU pipeline
could read freed device memory and fail with `CUDA_ERROR_ILLEGAL_ADDRESS`.

In `RoceReceiverOp::stop()` / `LinuxReceiverOp::stop()` the `Cleanup` RAII guard calls
`frame_buffer_.reset()`, releasing the `ReceiverMemoryDescriptor` (and its GPU memory).
But `compute()` wraps that buffer in a `nvidia::gxf::Tensor` with `wrapMemory`, and
those tensors can still be in flight downstream when `stop()` runs. The legacy tree
solved this in `BaseReceiverOp::compute()` by capturing an owning handle from a new
virtual `frame_memory_owner()` in the wrapped tensor's release callback.

The module has no `BaseReceiverOp` (each operator owns its `compute()`), and
`frame_buffer_` is already a `std::shared_ptr<ReceiverMemoryDescriptor>` (the legacy
`unique_ptr`→`shared_ptr` change was unnecessary here). So the port is just the capture:
each `compute()` copies the `frame_buffer_` `shared_ptr` into the `wrapMemory` release
lambda, so the buffer is freed only after the last referencing tensor is released — even
though `stop()` has already dropped the operator's own reference.

Files: `host/operators/roce_receiver_op.cpp`, `host/operators/linux_receiver_op.cpp`
(the `wrapMemory` release callback in `compute()`). No header, pybind, or build changes
— the operator API is unchanged.

## Phase 8 extension — framework support to complete the stereo IMX274 pattern test

**Implemented.** `tests/test_module_imx274_pattern.py` (the module port of the legacy
`tests/test_imx274_pattern.py`) needed five capabilities that the module lacked or only
partially provided; it previously worked around them or skipped variants. All five are
now on the V1 surface (added directly to the existing V1 interfaces — none have shipped,
so no `V2` was introduced).

- **`Adapter.use_mtu` Python binding.** The C++ `Adapter::use_mtu` →
  `ChannelConfigurationInterfaceV1::use_mtu` → `HsbLiteChannelConfigurationV1::use_mtu`
  (stamps `metadata["mtu"]`) path already existed; only the Python binding was missing.
  Added in `host/python/hololink_module_py.cpp` (`bind_adapter`), mirroring the
  `use_sensor` lambda. The MTU test now calls `adapter.use_mtu(metadata, mtu)` instead
  of stamping the key.

- **Multicast (`use_multicast`).** New `ChannelConfigurationInterfaceV1::use_multicast`
  (`host/include/hololink/module/channel_configuration.hpp`), implemented in
  `HsbLiteChannelConfigurationV1` (`module/core/hsb_lite_publisher.hpp`) to stamp
  `metadata["multicast"]` / `metadata["multicast_port"]` (the keys the legacy
  `DataChannel` reads). New `Adapter::use_multicast` (`host/include/.../adapter.hpp`,
  `host/src/adapter.cpp`)

  - Python binding. No receiver change is needed: `RoceDataChannelV1::configure` copies
    every metadata key onto the legacy `DataChannel`, which programs the FPGA's
    multicast MAC/IP/port. `test_imx274_multicast` is no longer skipped.

- **`rename_metadata` on the receiver operators.** `RoceReceiverOp` and
  `LinuxReceiverOp` take a new `rename_metadata`
  `std::function<std::string(const std::string&)>` parameter (defaults to identity). The
  renamed key names are cached in `start()` so `compute()` does no per-frame string
  work. Bound through the `PyRoceReceiverOp` / `PyLinuxReceiverOp` trampolines in
  `host/operators/python/operators_py.cpp`. `RoceReceiverOp` additionally emits
  `imm_data` / `page_number` (from `RoceReceiverFrameInfoV1::imm_data`), matching
  `LinuxReceiverOp`. Stereo flows pass `lambda name: f"left_"/"right_" + name` so the
  two legs' metadata don't collide.

- **`on_reset` on `HololinkInterfaceV1`.** New
  `virtual std::shared_ptr<ResetRegistration> on_reset(std::function<void()> callback)`
  (added to V1). It returns an RAII handle: holding it keeps the callback registered,
  dropping it unregisters. `HololinkV1` (`module/core/hololink_default.hpp`) registers a
  single aggregating `ResetCallbackController` with the backing legacy `Hololink` (whose
  `reset_controllers_` list is append-only) and keeps the per-caller callbacks in its
  own id-keyed registry; the handle erases its entry on destruction. This matters
  because the per-board `HololinkInterfaceV1` is a process-lifetime singleton —
  registering directly would accumulate callbacks (and pin their owners) for the whole
  session. Bound via the `PyHololinkInterface` trampoline + `bind_hololink_interface`
  (the handle is exposed as an opaque `ResetRegistration`). The test's `CameraWrapper`
  registers a **weak** callback (so the registry doesn't pin the camera) and keeps the
  handle, so the registration is released when the camera is garbage-collected; it
  asserts the callback fires once per `hololink.reset()`.

- **CRC operators in `hololink_module.operators`.** `ComputeCrcOp` / `CheckCrcOp` are
  ported from the legacy `hololink::operators` into `hololink::module::operators`
  (`host/operators/include/hololink/module/operators/compute_crc_op.hpp`,
  `host/operators/compute_crc_op.cpp`) — self-contained Holoscan operators using nvcomp
  CRC32 + CUDA, with the module logging / page-size / CUDA-check helpers. Gated by the
  new `HOLOLINK_MODULE_BUILD_CRC` CMake option (default ON when nvcomp is found; cleanly
  disabled otherwise). Bound in `operators_py.cpp`. The test now uses
  `hololink_module.operators.ComputeCrcOp` / `CheckCrcOp`; the CRC operators run in both
  the RoCE and Linux pipelines, with the strict received-vs-computed assertion kept on
  the lossless RoCE path (matching the legacy test, which only validates CRCs on the
  RoCE legs).

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD" -DHOLOLINK_MODULE_BUILD_OPERATORS=ON \
    -DHOLOLINK_BUILD_PYTHON=ON -DHOLOLINK_BUILD_ROCE=ON \
    -DHOLOLINK_MODULE_BUILD_CRC=ON
cmake --build "$BUILD"
```

**Test-run commands.**

```bash
python3 -m py_compile tests/test_module_imx274_pattern.py
pytest tests/test_module_imx274_pattern.py \
    --channel-ips <ip0> <ip1> --schedulers default -v
```

## Phase 8 extension — stereo frame alignment in the IMX274 pattern test

**Implemented.** IMX274 has no hardware frame-sync, so the two legs of
`tests/test_module_imx274_pattern.py` free-run and drift apart. The test now pairs the
two streams frame-for-frame with a pure-Python `FrameAlignerOp` (copied from legacy
commit `b80c5ea2` into the shared `tests/operators.py`). The aligner takes one `input`
port (`QueuePolicy.POP`) fed by both legs, distinguishes them by tensor name, reads each
leg's `{prefix}timestamp_s` / `{prefix}timestamp_ns` metadata, and forwards a paired
group only when the legs' timestamps are within `allowable_dt`. The acceptance window is
one full frame period (the largest skew that still guarantees adjacent-frame pairing
without hardware sync) — passed per-case as `allowable_dt` (`1/60 s` for the 60-FPS
modes) through the parametrized result tuples.

- **`out_tensor_name` on the receiver operators.** `RoceReceiverOp` and
  `LinuxReceiverOp` emitted an unnamed tensor; the aligner needs named tensors to tell
  the legs apart. Both receivers now take a
  `holoscan::Parameter<std::string> out_tensor_name_` (default `""`, preserving existing
  behavior), used in the `add<nvidia::gxf::Tensor>(...)` call and bound through the
  `PyRoceReceiverOp` / `PyLinuxReceiverOp` trampolines + `py::init` lists in
  `host/operators/python/operators_py.cpp`. This matches the existing `out_tensor_name`
  convention on `CsiToBayerOp` / `PackedFormatConverterOp` / `FusaCoeCaptureOp`. The
  receivers already emit `timestamp_s` / `timestamp_ns` under the per-leg renamed keys,
  so no other receiver change was needed.

- **Pipeline placement.** The aligner is the first hop: both receivers feed it and it
  fans paired frames out to each leg's `ComputeCrcOp`, so CRC and everything downstream
  run only on aligned frames — unaligned frames are dropped before any CRC work.

- **Single-interface stereo: `use_sensor` no longer bumps the data plane.** The core
  `HsbLiteChannelConfigurationV1::use_sensor` previously stamped `hif_address` per
  sensor (`data_plane = sensor_number`), assuming a 1:1 sensor↔data-plane layout.
  hololink-lite (and the `hsb_lite_2510` variant) actually multiplex multiple sensors
  onto **one** data plane, differentiated by virtual pipeline (VP) — matching the legacy
  `BasicEnumerationStrategy::use_sensor`, which re-stamps
  `vp_address`/`vp_mask`/`sif_address` but never touches `hif_address`. The old behavior
  OR'd sensor 1's `vp_mask` into the wrong data plane's `DP_VP_MASK`, so the FPGA never
  forwarded sensor 1's frames and the right leg of `test_imx274_stereo_single_interface`
  received nothing. `use_sensor` now keeps the enumerated data plane
  (`metadata["data_plane"]`) and only re-points the per-sensor VP/SIF. (The two-IP
  `test_imx274_pattern` path was unaffected — it enumerates each data plane directly
  rather than going through `use_sensor`.)

- **Frame-ready async scheduling on the receivers (required for the join).** The module
  receivers previously block-polled `get_next_frame` (up to 1 s) inside `compute()`.
  Feeding two such receivers into one aligner input under the greedy scheduler
  deadlocks: a leg that isn't delivering blocks the single scheduler thread, so the
  aligner only ever sees the other leg and never pairs. Both `RoceReceiverOp` and
  `LinuxReceiverOp` now always create a `holoscan::AsynchronousCondition` in `setup()`
  that the receiver's monitor thread signals via a new
  `RoceReceiverInterfaceV1::set_frame_ready` /
  `LinuxReceiverInterfaceV1::set_frame_ready` callback. `compute()` then runs only when
  a frame is actually ready and never blocks, so multiple receivers feeding a join
  schedule correctly. (Unlike legacy, there is no blocking-mode toggle — the blocking
  path is strictly a footgun that deadlocks joins for a negligible latency win.) The
  callback is forwarded to the legacy backing receiver in
  `module/core/{roce,linux}_receiver_default.hpp`.

**Test-run commands.**

```bash
python3 -m py_compile tests/operators.py tests/test_module_imx274_pattern.py
pytest tests/test_module_imx274_pattern.py \
    --channel-ips <ip0> <ip1> --schedulers default -v
```

## Phase 8 extension — data channels apply the application's enumeration metadata

**Implemented.** A data channel now uses the enumeration metadata the application hands
it at `get_service(metadata)` on **every** resolution, not just the first.

`RoceDataChannelInterfaceV1` (and the Linux/CoE channels) is a
per-`(serial,data_channel)` `ConfigurableService` whose `configure()` runs once
(`std::call_once`). That froze the metadata-derived host binding: the first test in a
process to resolve `data_channel=1` (e.g. the 2-IP `test_imx274_pattern` using the
board's native `192.168.0.3`) pinned the legacy `DataChannel` to `peer_ip=.3` →
`DP_HOST_IP=.102`. A later `test_imx274_stereo_single_interface` resolves the same
service with its `192.168.0.2` clone, but inherited the cached `.3`, so the FPGA wrote
to `.102` while the receiver's QP listened on `.101` → intermittent, order-dependent
data loss (`count=0`).

- `ConfigurableService::ensure_configured` (`host/include/hololink/module/service.hpp`)
  is now **virtual**, broadening its contract from "configure exactly once" to "ensure
  the configuration reflects this metadata." The default still runs `configure()` once
  via `call_once` (correct when a key's metadata is stable across resolutions).
- `RoceDataChannelV1` / `LinuxDataChannelV1` / `CoeDataChannelV1`
  (`module/core/*_data_channel_default.hpp`) **override `ensure_configured`** to re-run
  `configure(metadata)` (which rebuilds the backing legacy `DataChannel`) whenever the
  supplied metadata differs from what they last built from (cached in
  `applied_metadata_`); `configure()` is otherwise unchanged.

This is transport-agnostic (not RoCE/QP-specific), makes the channel deterministically
reflect the application's metadata regardless of test order, and retires the cross-test
process-singleton-state-leak for data channels. `local_ip_and_mac` and the QP path are
unchanged; the anchor `DataChannelV1` stays once-configured (only
`serial`/`data_channel` and the serial-keyed hololink are read from it). No legacy
`src/hololink` edits.

## Phase 8 extension — stereo IMX274 test-harness corrections (frame budget, watchdog, per-board coverage)

**Implemented.** Bringing the frame aligner up on hardware surfaced three defects in the
test harness (the data path itself was correct). All are test-side (`tests/`).

- **Frame budget moved to the sink.** `frame_limit` was a `CountCondition` on the two
  receivers, but `FrameAlignerOp` drops frames between the receivers and the visualizer
  (intentional `dt > allowable_dt` drops, plus occasional `QueuePolicy.POP` loss on its
  shared input). So the watchdog sink only ever saw `frame_limit − drops` taps; when the
  receivers hit their count and stopped, the watchdog was left waiting for taps that
  never arrive and timed out (`count=97` of 100). The budget now lives at the sink: the
  receivers free-run on a `holoscan.conditions.BooleanCondition`, `watchdog_operator`
  carries the `CountCondition(frame_limit)`, and `WatchdogOp` (`tests/operators.py`, now
  taking `frame_limit` + `stop_conditions`) disables the receivers' tick conditions once
  it has tapped `frame_limit` *aligned* frames. "100 frames" now means 100 aligned
  frames reaching the visualizer; the source over-produces to cover drops; the app
  drains cleanly. This is loss-rate-independent and also closes a latent shortfall on
  the lossless RoCE path (where even the intentional `dt`-drops would have left the sink
  short).

- **Watchdog timeouts fail deterministically.** `utils.Watchdog` (shared
  `tests/utils.py`) signalled failure with `assert False` in its `SIGUSR1` handler. When
  the timeout lands while the main thread is inside a GXF operator's `stop()`, GXF
  catches and discards the `AssertionError` (`gxf_wrapper.cpp:291`), so `run()` returns
  and a *stalled* run was reported `PASSED`. The watchdog now records `_timed_out` in
  `_expired` and re-raises in `__exit__` (main thread, after `run()` returns, unless the
  block is already unwinding another exception), so a timeout always fails the test.

- **Per-board single-interface coverage.** `test_imx274_stereo_single_interface` now
  parametrizes `channel_index` `[0, 1]` (cases beyond `--channel-ips` `pytest.skip`),
  running the single-interface scenario once per board — restoring the legacy
  `test_imx274_pattern.py` `(ibv_index, ibv_name)` two-interface coverage. The module
  resolves the IB device from the peer metadata, so only the channel index (which board)
  is parametrized; no `ibv_name` pairing is needed.

- **Aligner diagnostics.** `FrameAlignerOp` logs per-leg `arrivals`, `emitted`, and
  `dt_dropped` (on each `dt`-drop and every 30th emit) so a stall can be attributed to
  starvation (one leg stops arriving), alignment (`dt > allowable_dt`), or input-queue
  POP loss.

## Phase 8 extension — module port of the IMX274 timestamps test

**Implemented.** `tests/test_module_imx274_timestamps.py` is the module port of the
legacy `tests/test_imx274_timestamps.py` (PTP-synchronized frame-timestamp validation).
The legacy test drove `examples/imx274_player.main()` via `mock.patch`; there is no
Python `module_imx274_player`, so the module port self-composes the pipeline and
replicates `main()`'s bring-up (discovery, camera, reset, PTP sync, configure) inline —
the same self-composition the pattern-test port uses. It reuses the shared
`tests/operators.py` helpers (`TimeProfiler`, `WatchdogOp`, `FrameAlignerOp`) unchanged.
The legacy `tests/test_imx274_timestamps.py` is removed (replaced by this port), and
`ci/imx274_ptp_test.sh` now runs the module test. Unlike the legacy test, the
timing-bound assertions are enforced only on the RoCE path; the Linux-socket path has no
reliable timing (kernel scheduling, packet loss, userspace reassembly), so its timings
are logged but not asserted (frame collection is still checked on both paths).

Two capabilities were missing on the V1 surface and were added directly to the existing
V1 interfaces (nothing has shipped, so no `V2`):

- **`received_s` / `received_ns` on the module `RoceReceiverOp`.** `TimeProfiler`
  computes FPGA-to-host latency from the receiver's host-side reception time, which the
  module `LinuxReceiverOp` already emitted but `RoceReceiverOp` did not. The value
  already travels on `RoceReceiverFrameInfoV1` (populated from the backing legacy
  receiver); the op just wasn't stamping it. Added the two `rename("received_s")` /
  `received_ns` keys in `start()` and the matching `meta_map->set(...)` in `compute()`
  (`hololink_module/host/operators/roce_receiver_op.cpp` + its header). Without this the
  RoCE `receiver_dt` assertions would read a zero reception time and fail.

- **`ptp_synchronize()` on `HololinkInterfaceV1`.** New
  `virtual bool ptp_synchronize() = 0` (`host/include/hololink/module/hololink.hpp`),
  implemented in `HololinkV1` (`module/core/hololink_default.hpp`) by delegating to the
  backing legacy `Hololink`'s default-timeout overload, and bound through the
  `PyHololinkInterface` trampoline + `bind_hololink_interface`
  (`host/python/hololink_module_py.cpp`, with `gil_scoped_release` since it blocks up to
  ~20 s). The test calls it after `start()`/`reset()` so the FPGA timestamp clock shares
  the host's PTP domain. (The module folds the sensor reference-clock setup into
  `Imx274Cam.configure()` via the per-data-plane oscillator, so — unlike the legacy path
  — there is no separate `setup_clock()` call.)

The stereo case (`test_imx274_stereo_roce_naive`) is self-composed rather than built on
the legacy `applications.StereoTest`: the two free-running legs feed a `FrameAlignerOp`
(IMX274 has no hardware frame-sync, so they drift at startup), which forwards only pairs
within one frame period; a `StereoMonitorOp` sink reads each leg's `left_`/`right_`
`timestamp_s`/`_ns` (the frame-start time) to log the inter-leg skew and owns the frame
budget (disabling the receivers' tick conditions once `frame_limit` aligned pairs
arrive).

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD" -DHOLOLINK_MODULE_BUILD_OPERATORS=ON \
    -DHOLOLINK_BUILD_PYTHON=ON -DHOLOLINK_BUILD_ROCE=ON
cmake --build "$BUILD"
```

**Test-run commands.**

```bash
python3 -m py_compile tests/test_module_imx274_timestamps.py
pytest tests/test_module_imx274_timestamps.py --imx274 --ptp --headless -vv
```

## Phase 8 extension — `rename_metadata` + `received_s`/`received_ns` on `FusaCoeCaptureOp`

**Implemented.** `FusaCoeCaptureOp` gained an optional `rename_metadata` parameter
(default identity), matching the convention `RoceReceiverOp` / `LinuxReceiverOp` already
follow. The op already accepted `out_tensor_name`; `rename_metadata` closes the gap so a
FUSA CoE leg can be told apart *and* stamp per-leg-prefixed frame metadata. Without it,
two capture legs feeding one consumer (a stereo pair through a `FrameAlignerOp`) would
both write the same unprefixed keys (`timestamp_s`, `timestamp_ns`, `metadata_s`,
`metadata_ns`, `crc`, `frame_number`) and collide on merge.

The op also now emits **`received_s` / `received_ns`**, matching the RoCE/Linux
receivers (and the legacy `LinuxCoeReceiverOp`). This is the host-side reception time —
the `CLOCK_REALTIME` wall-clock instant the CPU woke up with the frame-ready
notification, captured in `compute()` immediately after `wait_for_acquired_buffer`
returns and before the metadata decode so it stays close to that wakeup. The FUSA
capture core's `CoeFrameMetadata` carries no reception time, so it's sampled in the op
rather than decoded from the EOF block. `CLOCK_REALTIME` shares the PTP-disciplined
domain of the FPGA `timestamp_s`, so a downstream `received - timestamp` latency is
meaningful (this is what `TimeProfiler` computes).

The op caches the (possibly renamed) key names once in `start()` — defaulting to
identity when no function is supplied — and `compute()` stamps through the cached names,
so there is no per-frame string work
(`hololink_module/host/operators/fusa_coe_capture_op.cpp` + its header). The pybind
trampoline `PyFusaCoeCaptureOp` forwards the function through a `holoscan::Arg`, and the
binding exposes it as a keyword defaulting to the identity `std::function` — inserted
before `out_tensor_name` in the signature, but since every caller passes
`out_tensor_name` / `name` by keyword, existing call sites are unaffected
(`host/operators/python/operators_py.cpp`). This lets a stereo FUSA CoE pipeline align
its legs natively (per-leg `rename_metadata` + `out_tensor_name`) exactly as the
RoCE/Linux receiver legs do, with no test-side metadata-prefixing shim.

The binding also now lists `hololink::module::csi::CsiConverterV1` as a pybind base of
`FusaCoeCaptureOp` (matching `CsiToBayerOp` / `PackedFormatConverterOp`). The op has
always *been* a `CsiConverterV1` in C++ — a sensor trains it via
`camera.configure_converter(fusa)` — but without the base declared, pybind could not
upcast it and that call raised
`TypeError: incompatible function arguments ... converter: CsiConverterV1`.
`CsiConverterV1` is registered by the imported core extension (`_hololink_py_module`),
so it resolves as a base here.

**`FusaCoeCaptureOp` is now asynchronous.**
`FusaCoeCaptureCore::wait_for_acquired_buffer` blocks the caller until a frame lands;
previously `compute()` called it directly, so the op blocked the scheduler thread for up
to a capture timeout. That's fine for a single op on the default (greedy, single-thread)
scheduler, but two blocking capture legs feeding one `FrameAlignerOp` (a stereo CoE
pair) serialize: the legs arrive ~a capture-timeout apart, always beyond the aligner's
one-frame window, so it never emits a pair. Rather than push a `MultiThreadScheduler`
onto every caller, the op now follows the same async pattern as `RoceReceiverOp` /
`LinuxReceiverOp` / the legacy `BaseReceiverOp`: a background monitor thread owns the
blocking `wait_for_acquired_buffer` and signals an `holoscan::AsynchronousCondition`;
`compute()` runs on the scheduler thread only when a frame is ready and does just the
(fast) GXF wrap + emit. The monitor and `compute()` hand off exactly one frame at a time
(a mutex/condition-variable gate), which preserves the core's single
`acquire -> register_pending_output -> release` slot (`buffer_in_compute_` is one-deep).
`received_s`/`received_ns` are sampled in the monitor right after the wait returns — the
true CPU-wakeup instant. Net effect: mono and stereo CoE both run correctly on the
default scheduler, matching the receiver operators and the legacy CoE path (which was
async via `BaseReceiverOp`), and the test needs no scheduler override
(`hololink_module/host/operators/fusa_coe_capture_op.cpp` + its header).

`FusaCoeCaptureCore::wait_for_acquired_buffer` now has an explicit three-way contract:
it returns `true` (a frame was acquired; the out-param `BufferView` is populated),
returns `false` (a plain per-request timeout — no frame within the window, which is
normal at startup and between frames), or **throws** (a genuine capture failure — any
state that is neither OK nor a timeout). The monitor retries on `false` and deliberately
does **not** catch the throw: a real capture failure propagates out of the monitor
thread and terminates the process, surfacing the fault loudly (and landing directly on
it under a debugger) rather than being swallowed into a silently stalled pipeline. The
legacy `FusaCoeCaptureOp::compute()` (`src/hololink/operators/fusa_coe_capture/`)
consumes the same contract — it returns early (skip-and-retry) on `false` and lets a
failure throw out of `compute()` (where GXF tears the graph down).

## Phase 8 extension — module port of the CoE test

**Implemented.** `tests/test_module_coe.py` is the module port of the legacy
`tests/test_coe.py`. Like the earlier ports it self-composes the pipeline in the test
file rather than reusing the shared `tests/applications.py` (`MonoTest` / `StereoTest`),
and it covers IMX274 and VB1940 across mono / stereo and non-CoE / CoE — seven test
functions mirroring the legacy file's non-Argus rows.

Two capture front-ends share one naive-ISP tail (`ImageProcessorOp` → `BayerDemosaicOp`
→ `HolovizOp`):

- **non-CoE:** `LinuxReceiverOp` → `CsiToBayerOp`.
- **CoE:** `FusaCoeCaptureOp` → `PackedFormatConverterOp`. The module has no
  Linux-socket CoE receiver (the legacy `LinuxCoeReceiverOp` has no module twin); the
  module's CoE consumer is the FUSA hardware capture op, so the CoE cases run
  `FusaCoeCaptureOp` (built only with `-DHOLOLINK_BUILD_FUSA`, hence a runtime
  `pytest.skip` when it is absent). Discovery/interface/MAC follow the adapter FUSA
  convention from `examples/fusa_coe_single_network_stereo_hawk_player.py`: `interface`
  from the `--coe-interface` fixture, `mac_addr` from `metadata.get("mac_id")`.

Design notes:

- **Argus dropped.** The legacy file's `*_argus*` rows are omitted — `ArgusIspOp` is
  deprecated and has no module equivalent.
- **FrameAlignerOp on every stereo test.** The two sensors free-run (no hardware
  frame-sync), so the aligner pairs them frame-for-frame and drops a group whose legs'
  timestamps differ by more than the tolerance. Every stereo case (both Linux and CoE)
  uses **twice the frame time** as `allowable_dt` (`TWO_FRAMES_AT_60FPS` /
  `TWO_FRAMES_AT_30FPS`) — a single frame period is too tight to reliably pair two
  free-running legs. Both capture ops stamp per-leg-prefixed timestamp metadata natively
  (the Linux receiver always did; `FusaCoeCaptureOp` does since the `rename_metadata`
  addition above), so no test-side prefixing shim is needed. Because the aligner pairs
  legs by their PTP-disciplined FPGA timestamps, the stereo tests **require PTP** — each
  is marked `@pytest.mark.skip_unless_ptp` and so runs only when `--ptp` is on the
  command line (the mono tests, which have no aligner, are not gated).
  `ptp_synchronize()` therefore always runs at startup in the stereo cases.
- **No special scheduler.** All cases run on the default scheduler. `FusaCoeCaptureOp`
  is asynchronous (see the op section below), so — like `LinuxReceiverOp` — two CoE
  capture legs feeding one `FrameAlignerOp` deliver frames concurrently rather than
  serializing.
- **Naming: `linux` vs `fusa`.** The non-CoE cases capture through `LinuxReceiverOp` and
  are named `*_linux_naive`; the CoE cases capture through `FusaCoeCaptureOp` and are
  named `*_fusa_naive_coe` (there is no Linux-socket CoE receiver in the module, so
  `linux` would misname them).
- **VB1940 bring-up** is `Vb1940Cam(metadata)` + `configure(mode)` — the module folds
  `setup_clock` and the sensor-enable register pokes into `configure()`, so there is no
  legacy `write_uint32` / `get_register_32` dance; NullVsync (the constructor default)
  leaves the sensor free-running, matching the legacy null-synchronizer default. Cameras
  are constructed *after* `reset()` (the ordering the module VB1940 test uses; IMX274
  tolerates it too).
- **Validation** matches the legacy Linux-socket tests: run to `frame_limit` without a
  watchdog timeout; image data is not evaluated (the Linux and CoE paths tolerate packet
  loss). `ptp_synchronize()` runs when `--ptp` is given. The legacy `tests/test_coe.py`
  is removed — this port replaces it.

**Test-run commands.**

```bash
python3 -m py_compile tests/test_module_coe.py
pytest tests/test_module_coe.py --imx274 --vb1940 \
    --coe-interface=<iface> --headless -vv
```

## Phase 9 — module reconnection

**Implemented; RoCE mono + stereo verified on hardware (Linux path built, pending a
hardware run).** A capture pipeline survives a board reset — whether mid-stream or
mid-configuration: the device state is invalidated, the board is rediscovered on
re-announcement, fresh V1 handles are fetched, and frames resume. The work splits into
framework support (`hololink_module/host`) and an application-supplied sensor/camera
policy (the test), keeping device-specific behavior out of the framework. The test
(`tests/test_module_imx274_reconnect.py`) supersedes the legacy reconnection test and
its `hololink.hsb_controller` / `*_controller_receiver` implementation, both now
removed.

- **`trigger_reset()` on `HsbLiteInterfaceV1`** (`module/hsb_lite/.../hsb_lite.hpp`,
  impl `module/core/hsb_lite_default.hpp` forwarding through `legacy_access()`, bound in
  `hsb_lite_py.cpp`). Fire-and-forget FPGA reset used to induce the loss.

- **Refcounted ownership + per-device invalidation.** `Publisher` refcounts host handles
  in an `outstanding_` side-table (`get_service`/`release_service`), so cross-`.so`
  handles are owning and old handles stay valid until dropped (no UAF). `device_lost()`
  on `HololinkInterfaceV1` / `DataChannelInterfaceV1` cascades to every per-board
  service that registered via `register_associated(this)`, then
  `Publisher::invalidate(this)` removes the identity-matching `registry_` entries — no
  device/graph knowledge in the `Publisher`. A re-`get_service` after loss returns fresh
  instances that re-run `configure()`. One subtlety: `HsbLiteOscillatorV1` is a
  per-board survivor (shared by serial across data planes, cached outside the registry),
  so it outlives the `HololinkV1`/legacy it configured against. Its clock-rate cache is
  therefore keyed to the `LegacyHololinkAccess` it programmed — a reconnect (or a fresh
  pipeline run on the same board) has a new legacy Hololink, which `enable()` sees as a
  cache miss and reprograms, instead of no-op'ing on a stale rate and leaving the
  clock-less IMX274 to NAK every I2C.

- **Reusable `Watchdog`** (`host/include/hololink/module/watchdog.hpp`) — a `ReactorV1`
  deadline timer (mutex + generation counter + shared-state alive flag) tapped on each
  received frame; its timeout is the data-plane loss detector.

- **Decoupled controller stack** (`host/operators`): `NetworkReceiver` (transport seam;
  `RoceNetworkReceiver` + `make_roce_network_receiver_factory()` for the hardware path,
  `LinuxNetworkReceiver` + `make_linux_network_receiver_factory()` for the software
  path), `SensorFactory` (owns the watchdog, the `register_ip` reconnection loop, and
  `new_sensor`), `SensorDevice` (wraps one armed camera; `stop_sensor` /
  `fallback_frame`), and `HsbController` (orchestrator cycling receiver
  `construct`→`run` / `destruct` and calling `device_lost()` on loss). `HsbControllerOp`
  is a thin `holoscan::Operator` adapter over `HsbController`; its `compute()` allocates
  the pipeline CUDA stream from the Holoscan execution context, threads it through
  `NetworkReceiver::get_next_frame`, and sets it on the emitted tensor — the software
  receiver copies each frame host→device on that stream (so it overlaps downstream
  work), and the RoCE receiver, which DMAs straight to device memory, ignores it. All
  bound in `operators_py.cpp` (`PyHsbControllerOp`, `SensorFactory` / `SensorDevice`
  trampolines, opaque `NetworkReceiver` for the factory round-trip). Transport-agnostic
  pieces and `linux_network_receiver.cpp` build always (the software path links no
  ibverbs); `roce_network_receiver.cpp` is gated on `HOLOLINK_BUILD_ROCE`, and its
  factory binding likewise.

- **The test** (`tests/test_module_imx274_reconnect.py`).
  `Imx274SensorFactory.new_sensor` brings the board up (`start()`/`reset()`, guarded
  once per board per connect) and builds a fresh `Imx274Cam(metadata)` — that rebuild is
  the reconnection re-fetch. The reset fires at frame 150 via
  `HsbLiteInterfaceV1.get_service(metadata).trigger_reset()` deferred onto the reactor
  (`Adapter.reactor().add_callback(...)` — a new binding returning the host `ReactorV1`,
  which is published on the host module rather than a device module); RoCE CRCs are
  checked outside a settle window around the reset, and `_sensor_count >= 2` asserts the
  reconnect happened. A SMPTE-bars test image is shown whenever no live frame is
  available — at startup before the first connect and during a later outage — via the
  app's `SensorFactory.fallback_frame()` override (a `virtual` returning `{0,0}` by
  default, not tied to a `SensorDevice` so it works before any sensor exists);
  `HsbControllerOp::start()` primes one `compute()` so the test image appears at
  startup. (`make_csi_from_image_file` is copied into the test and retargeted to the
  module `csi` enums — `utils`'s copy is bound to the legacy `hololink` enum type.) The
  Linux cases select the software transport via `make_linux_network_receiver_factory()`
  and validate recovery (`_sensor_count`) rather than CRCs, since software sockets drop
  packets. The file covers all seven legacy reconnect cases: RoCE/Linux mono, RoCE/Linux
  stereo, and three **reconnect-during-configuration** variants that trigger the reset
  from *inside* camera configuration (an `InstrumentedImx274Cam.set_register` fires
  `trigger_reset()` on the Nth write), losing the board mid-bring-up to exercise the
  control-plane loss path (`new_sensor` fails → `invalidate_board` → retry); those
  assert the reset fired and the recovered stream ends in a long clean run of
  correct-CRC frames.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD" -DHOLOLINK_MODULE_BUILD_OPERATORS=ON \
    -DHOLOLINK_BUILD_PYTHON=ON -DHOLOLINK_BUILD_ROCE=ON
cmake --build "$BUILD"
```

**Test-run commands.**

```bash
python3 -m py_compile tests/test_module_imx274_reconnect.py
pytest tests/test_module_imx274_reconnect.py --imx274 --headless -vv
```

## Phase 9 extension — Linux software transport on pre-0x2602 (0x2510) FPGAs

**Implemented; pending a hardware run.** The Linux software-socket receiver now works on
0x2510 boards, mirroring the existing RoCE 2510 support. The CoE test's `*_linux_naive`
cases previously failed there with
`UnsupportedVersion: hsb_ip_version=0X2510; minimum supported version=0X2602`.

The block was a **misplaced version gate**: the legacy `DataChannel::configure_socket`
hard-checked `hsb_ip_version >= 0x2602`. The RoCE path never calls `configure_socket`
and has no such gate — it stays correct on 0x2510 purely by *type selection*
(`make_backing` builds `HsbLite2510DataChannel`, whose virtual `configure_roce` speaks
the older memory map). The Linux path lacked the equivalent data-channel specialization
and, being the only caller of the gate, hit it.

- **The version check moved to where the memory map is actually programmed.**
  `configure_roce` (called by *both* transports) now runs it, driven by a new
  `virtual int64_t DataChannel::minimum_hsb_ip_version()` (base `0x2602`); the hardcoded
  check leaves `configure_socket`. Every path that was gated before (legacy RoCE /
  legacy Linux / module Linux — all call `configure_roce`) stays gated; COE (which never
  called `configure_socket`) is unaffected. The net is a *stronger* guard — it now
  covers the RoCE data plane too, and fails loudly before any traffic instead of
  silently mis-programming.

- **`HsbLite2510DataChannel`** overrides `minimum_hsb_ip_version()` to `0x2510` and
  calls the shared check at the top of its `configure_roce`.

- **`HsbLite2510LinuxDataChannelV1`** is the software-transport sibling of
  `HsbLite2510RoceDataChannelV1`: same `make_backing` override selecting
  `HsbLite2510DataChannel` below 0x2602, plugged in by a
  `HsbLite2510Publisher::construct_linux_data_channel` override that mirrors
  `construct_roce_data_channel`.

- **`HsbLite2510LinuxReceiverV1` + `HsbLite2510LinuxReceiver`** mirror the RoCE receiver
  specialization. The software receiver derives the frame's page index from the RDMA
  immediate-data field, whose bit layout changed at 0x2603 (base masks `0xFFF`, the
  pre-0x2603 layout masks `0xFF`) — so on a 0x2510 board the modern mask picks up PSN
  bits and yields a bogus page (e.g. `3840`), throwing `Invalid page`.
  `LinuxReceiver::page_from_imm` is now `virtual` (matching
  `RoceReceiver::page_from_imm`), `LinuxReceiverV1` gained the `make_receiver` seam
  override, and the publisher's `construct_linux_receiver` override (out-of-line in
  `linux_receiver_construct_2510.cpp`) publishes the 2510 subclass. The `>= 0x2603`
  boundary matches the RoCE receiver's — distinct from the data channel's 0x2602
  memory-map boundary.

**Test-run commands.**

```bash
pytest tests/test_module_coe.py::test_imx274_mono_linux_naive --imx274 --headless -vv
```
