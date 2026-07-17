# Hololink Adapter

## Problem

An application may need to communicate with multiple hololink devices that require
different versions of the hololink library. Because C++ symbol names collide when two
versions of the same library are loaded into one process, there is no safe way to link
multiple hololink versions into a single binary.

## Solution

A thin adapter layer that isolates each hololink library version inside its own shared
object, loaded with `dlopen(RTLD_LOCAL)`. Because all modules are built with the same
toolchain as the adapter, the boundary uses versioned C++ abstract interfaces rather
than C function-pointer tables. `extern "C"` linkage is used only for the `dlsym` entry
points (`hololink_adapter_init`, `hololink_adapter_get_abi_check`). A symmetric service
locator pattern lets the host and modules discover each other's capabilities without
coupling.

Interfaces are **not frozen during this project** — each V1 interface may be edited in
place as implementations surface requirements, with all callers updated in the same
commit. A hard freeze is adopted only when the project is complete, after which new
capabilities are added through new versioned interfaces (`V2`, …) rather than changes to
existing ones.

## Scope

### Singleton adapter (host process)

The adapter is a single instance in the host application:

| Responsibility        | Details                                                                                    |
| --------------------- | ------------------------------------------------------------------------------------------ |
| Reactor               | Singleton event loop shared by the application and all modules                             |
| Service locator       | Provides host services (Reactor, logging, …) to modules                                    |
| Device enumeration    | Bootp v2 listener on Reactor thread; `wait_for_channel`, `register_ip`, manual `enumerate` |
| UUID → module mapping | Loads and caches one module per `(uuid, compat-id)` key                                    |
| Module lifecycle      | `dlopen`, ABI check, init, service injection                                               |

### Loadable module (per UUID)

Each module is a `.so` loaded with `dlopen(RTLD_LOCAL)`. Source code for a loaded module
is under `module/` — a single shared **`module/core/`** carries the common backend code
and the default service implementations, and a thin per-device **`module/<name>/`** that
names the device UUID, inherits or overrides the compat-id, and adds whatever
device-specific supplements and overrides are needed. Both compile into the single `.so`
that gets shipped. The interfaces a module exposes are:

| Interface                  | Responsibility                                                                                                                                                                              |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `EnumerationInterface`     | `update_metadata` — version-specific enumeration enrichment                                                                                                                                 |
| `HololinkInterface`        | Device control plane: `start`/`stop`/`reset`, `write_uint32`/`read_uint32`, `configure_hsb`, `i2c_lock`, plus `get_*` factories for bus and data-channel interfaces and sequencer factories |
| `SequencerInterface`       | Build pre-programmed command sequences (`write_uint32`, `read_uint32`, `poll`, `enable`, `location`)                                                                                        |
| `RoceDataChannelInterface` | Per-board RoCE data channel: `configure(metadata, qp, rkey, frame_memory, frame_size, page_size, pages)`, `unconfigure`, `frame_end_sequencer`, `get_hololink`                              |
| `I2cInterface`             | `i2c_transaction`, `encode_i2c_request` (encode into a Sequencer for deferred execution)                                                                                                    |
| `FrameMetadataInterface`   | `decode(host_memory, size, out_metadata)` — parse the end-of-frame metadata block deposited by the device into structured fields                                                            |

Board-specific primitives that are not generic (e.g. HSB-Lite's clock setup) live on a
separate board-specific interface (`HsbLiteInterfaceV1`, `LeopardVb1940InterfaceV1`, …)
contributed by the module tree, not by core.

### Core

A single `module/core/` directory holds the V1-shaped wrappers over the existing
`src/hololink/core/` backend (register I/O, data-channel configure, I2C encoder,
sequencer building blocks, the end-of-frame metadata decoder), exposing the default V1
service implementations: `HololinkInterfaceV1`, `RoceDataChannelInterfaceV1`,
`I2cInterfaceV1`, `SequencerInterfaceV1`, `EnumerationInterfaceV1`, and
`FrameMetadataInterfaceV1`. The existing `src/hololink/core/` is unchanged; the wrappers
delegate into it. `module/core/` also names a default compat-id (e.g. `2603`) — the
IP-block revision the wrappers track at head-of-tree.

There is **one** wrapper archive. Every first-party module — and every partner module
that targets the same family of devices — links the same `hololink::module` archive and
absorbs a private copy of it (and the `src/hololink/core/` symbols it pulls in) via
`RTLD_LOCAL`. Cross-module differences live in the module tree, not in the wrappers.

### First-party modules

| Module directory         | Output `.so`                                            | Target device                                                          |
| ------------------------ | ------------------------------------------------------- | ---------------------------------------------------------------------- |
| `module/hsb_lite/`       | `hololink_889b7ce3-65a5-4247-8b05-4ff1904c3359_2603.so` | HSB-Lite, current FPGA (`0x2603`)                                      |
| `module/hsb_lite_2510/`  | `hololink_889b7ce3-65a5-4247-8b05-4ff1904c3359.so`      | HSB-Lite fallback (`0x2510` + any compat without a dedicated `.so`)    |
| `module/leopard_vb1940/` | `hololink_f1627640-b4dc-48af-a360-c55b09b3d230.so`      | Leopard VB1940 (bare-UUID exception: FPGA doesn't publish a compat-id) |

Each module directory carries:

- The device UUID — the value the bootp payload publishes for that board.
- A choice of filename suffix at `add_hololink_module()` time. By default the helper
  emits `hololink_<UUID>_<compat>.so` where the compat-id is the module's `COMPAT`
  argument or, if absent, the `DEFAULT_COMPAT` target property on `hololink::module`
  (currently `2603`). Passing `NO_COMPAT_SUFFIX` instead emits the bare
  `hololink_<UUID>.so` — the Adapter loader's catch-all when enumeration metadata's
  `compat_id` has no dedicated compat-suffixed `.so` (or carries no `compat_id` at all).
  `NO_COMPAT_SUFFIX` and `COMPAT` are mutually exclusive.
- **Supplements** — additional V1 services that core does not publish because they are
  board-specific. `HsbLiteInterfaceV1` (HSB-Lite's clock setup),
  `LeopardVb1940InterfaceV1`, a partner-specific `PartnerXyzInterfaceV1`, a
  board-specific power sequencer all land here. They are registered under new
  `(instance_id, type_id)` pairs alongside the core defaults.
- **Overrides** — replacement implementations for V1 services where the module's
  hardware diverges from what core assumes. The override is registered under the same
  `(instance_id, type_id)` as the core default, so the module's implementation shadows
  it. `module/hsb_lite_2510/` is the canonical override case: the older FPGA's register
  map differs enough from current HSB-Lite that affected V1 services ship as fallback
  replacements in `module/hsb_lite_2510/`. The first concrete example is the per-channel
  `RoceReceiverInterfaceV1` (see "RoCE receiver" under C++ Interfaces): both modules
  publish a receiver per `(serial, data_plane)` under the same locator key, but the 2510
  module's instance is a subclass of the legacy `hololink::operators::RoceReceiver` that
  changes the behavior the 2510 FPGA requires, while `module/hsb_lite/` (compat
  `0x2603`) publishes the unmodified default. `RoceReceiverV1::get_service(module, …)`
  returns whichever the loaded module published, transparently to `RoceReceiverOp`.
  Parts of the surface that haven't changed continue to come from the shared core.
- The `module_entry.cpp` that wires core defaults + supplements + overrides into
  `hololink_adapter_init`.

HSB-Lite's two FPGA revisions are disambiguated by compat-id rather than UUID, and they
ship under different filenames so the Adapter loader's `(uuid, compat_id) → .so` lookup
routes each board to the right module:

- `module/hsb_lite/` is the current-FPGA-specific module. It ships as the
  compat-suffixed `hololink_<UUID>_2603.so`; only boards whose bootp payload carries
  `compat_id=0x2603` resolve to it.
- `module/hsb_lite_2510/` is the catch-all. It ships as the bare `hololink_<UUID>.so`
  (built with `add_hololink_module(... NO_COMPAT_SUFFIX ...)`) and serves every HSB-Lite
  board the loader doesn't already have a dedicated compat-suffixed `.so` for —
  including `compat_id=0x2510`, any future compat-id without its own module, and
  payloads that carry no `compat_id` at all.

Both module trees share core; the differences live in `module/hsb_lite_2510/` as
overrides. The two HSB-Lite *module* trees (`module/hsb_lite/`, `module/hsb_lite_2510/`)
carry **no shared module-specific source** — anything 2510 needs that core doesn't
already provide is contributed from inside `module/hsb_lite_2510/`. The shared code that
does live in core is shared because core represents the single current backend; if a
future revision diverges so far from current that no shared backend is meaningful, the
resolution is to advance core and push everything older into the older module's
overrides, not to fork core.

`module/leopard_vb1940/` follows the same pattern — UUID, optional compat override, and
however many supplements / overrides Leopard's hardware needs on top of the shared core.

Two hardware demos validate multi-module cohabitation:

- **Same UUID, different compat-ids.** A system with both `0x2603` and `0x2510` HSB-Lite
  hardware attached. Both `.so` files load into the same process under distinct
  `(uuid, compat)` cache keys, each carries its own private backend via `RTLD_LOCAL`,
  and the two boards are driven simultaneously through the same adapter. Validates that
  the same UUID can resolve to different modules within one process.
- **Different UUIDs.** A system with HSB-Lite hardware (either `0x2603` or `0x2510`) and
  a Leopard VB1940 board attached. Two modules with distinct UUIDs load into the same
  process; the adapter routes each enumeration result to the matching module by UUID,
  and the application drives both kinds of board concurrently. Validates the cross-UUID
  case independent of the HSB-Lite compat-id story.

Note that it is unusual for the tree to support older device revisions; the 2510 version
of HSB-Lite is included here for testing and demonstration purpose. Partners normally
would tag an entire implementation tree and use that tag to fetch and build the module
that supports that device.

### Out of scope

Full camera/sensor drivers (`Imx274Cam`, `Vb1940Cam`, etc.) and higher-level operators
(`ImageProcessor`, `RoceReceiver`, etc.) are not part of the adapter and are not edited
as part of this work — they remain in the existing tree. Per-frame data flow (acquiring
the frame buffer, signalling completion, scheduling work) belongs to those receivers;
the adapter contributes only the **decode** step through `FrameMetadataInterface`.

## Architecture

```
Host Application
  │
  ├─ Adapter Enumerator       (simplified, runs on Reactor thread)
  │    listens for bootp ──► extracts UUID ──► loads/finds module
  │                                                │
  │                          ┌─────────────────────┘
  │                          ▼
  │              Module for UUID X                Module for UUID Y
  │              (.so, RTLD_LOCAL)                 (.so, RTLD_LOCAL)
  │   ┌──────────────────────────────┐  ┌──────────────────────────────┐
  │   │ module/X/                    │  │ module/Y/                    │
  │   │   UUID, optional compat      │  │   UUID, optional compat      │
  │   │   supplements + overrides    │  │   supplements + overrides    │
  │   │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │  │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
  │   │ hololink::module (private)   │  │ hololink::module (private)   │
  │   │   default V1 services        │  │   default V1 services        │
  │   │   wraps src/hololink/core/   │  │   wraps src/hololink/core/   │
  │   └──────────────────────────────┘  └──────────────────────────────┘
```

Each module's `.so` statically absorbs `hololink::module` plus its own supplements and
overrides. `RTLD_LOCAL` keeps those symbols private to the `.so`, so two modules
carrying their own private copy of the wrapper layer — same source, separately compiled
into each `.so` — coexist safely.

## Python Example

```python
import hololink_adapter

class Imx274Driver:
    def __init__(self, i2c):
        self._i2c = i2c
    def read_chip_id(self):
        data = self._i2c.i2c_transaction(0x1A, [0x30, 0x49])
        return (data[0] << 8) | data[1]

adapter = hololink_adapter.Adapter.get_adapter()
metadata = adapter.wait_for_channel("192.168.0.100", timeout_s=30.0)
module = adapter.get_module(metadata)

hololink = hololink_adapter.HololinkInterface.get_hololink(module, metadata)
hololink.start()

i2c = hololink.get_i2c(bus=1, address=0x1A)
sensor = Imx274Driver(i2c)
chip_id = sensor.read_chip_id()
```

## Enumeration

### Flow

1. **Bootp socket** — UDP socket registered as an FD callback on the shared Reactor,
   configured with `SO_REUSEPORT` so other processes on the host can listen on the same
   port independently. Only bootp v2 is supported.
1. **Packet parsing** — The adapter deserializes **every known field** from the bootp
   message into the returned `EnumerationMetadata`. The decoder does not pre-filter to a
   hand-picked subset; if the parser knows how to decode a field, the value is written
   into the metadata so downstream code (the module's `update_metadata`, application
   callbacks, log output, diagnostic dumps) can read it directly. That includes the
   standard bootp header fields (`peer_ip`, `control_port`, `serial_number`,
   `fpga_uuid`, `hsb_ip_version`, hardware address, board fields, …) and every
   recognized field inside the NVDA vendor section — most notably the 16-bit `compat_id`
   when present, but also any other vendor TLV the parser supports. Unrecognized vendor
   TLVs are skipped. New fields added to the bootp payload are added to the parser and
   immediately become available to every consumer with no API change.
1. **Module resolution** — The adapter maintains a `(uuid, compat-id-or-none) → Module`
   map. On first request for a key it searches the module directory for a `.so`,
   `dlopen(RTLD_LOCAL)` loads it, runs the ABI check, and calls `hololink_adapter_init`.
1. **Metadata enrichment** — The adapter calls the module's
   `EnumerationInterfaceV1::update_metadata` (on the Reactor thread) to add
   version-specific fields (VP/SIF/HIF addresses, sensor counts, board fields).
1. **Application callback** — The enriched `EnumerationMetadata` is delivered through
   `find_channel` / `register_ip` / `register_all`.

Manual enumeration: applications construct `EnumerationMetadata` directly and call
`Adapter::enumerate(metadata)`. From that point the pipeline is identical to the
bootp-driven path.

### Metadata types

- **`EnumerationMetadata`** —
  `std::map<std::string, std::variant<int64_t, std::string, std::vector<uint8_t>>>`
  carrying discovery information (peer IP, control port, UUID, `hsb_ip_version`,
  `compat_id`, sensor indices, board description, etc.). This is what flows through
  enumeration.
- **`FrameMetadataInterface::FrameMetadata`** — the structured per-frame fields decoded
  from the end-of-frame metadata block deposited by the device into the receiver's host
  buffer. The adapter ships the decoder (`FrameMetadataInterface::decode`); the receiver
  supplies the raw block.

### Module directory and filename scheme

Default directory `/usr/lib/hololink/modules/`, overridden by `HOLOLINK_MODULE_DIR`.
Filenames:

```
hololink_<uuid>.so                 # serves every device with this UUID that has no
                                   # dedicated compat-suffixed .so (or whose bootp
                                   # payload carries no compat-id)
hololink_<uuid>_<compat>.so        # serves only devices advertising this compat
```

Compat-id has two representations that must not be confused:

- **Numeric form** — the FPGA's 16-bit IP version field. Parsed from the bootp payload's
  little-endian uint16 and stored in `EnumerationMetadata["compat_id"]` as an `int64`
  (e.g. `0x2603`, `0x2510`). Code that compares against the wire value uses this form
  (`metadata["compat_id"] == 0x2603`).
- **String form** — the 4-digit lowercase hex rendering of the numeric value (`"2603"`,
  `"2510"`). Used in the CMake `COMPAT` / `DEFAULT_COMPAT` properties, in `.so`
  filenames, and in human-facing logs. `Adapter::load_module_for` formats `compat_id` as
  `%04x` when composing the filename it dlopens; out-of-tree code that composes the same
  path uses the same `%04x` rendering.

Example:

```
hololink_f1627640-b4dc-48af-a360-c55b09b3d230.so           # Leopard VB1940
                                                           # (bare-UUID exception:
                                                           # FPGA doesn't publish a
                                                           # compat-id)
hololink_889b7ce3-65a5-4247-8b05-4ff1904c3359_2603.so      # HSB-Lite, compat 0x2603
hololink_889b7ce3-65a5-4247-8b05-4ff1904c3359.so           # HSB-Lite fallback (catches
                                                           # 0x2510 + any compat-id
                                                           # without a dedicated .so)
```

Lookup order for a device advertising compat-id C with UUID U:

1. `hololink_<U>_<hex(C)>.so` if present (hex(C) = 4-digit lowercase hex of the numeric
   `compat_id`)
1. `hololink_<U>.so` otherwise (or when the bootp payload carried no compat-id)

If neither exists, the adapter still publishes the bootp metadata without
`module_filename` or enrichment. The `compat_id` metadata entry is always populated when
the bootp payload carried one.

`889b7ce3-65a5-4247-8b05-4ff1904c3359` is the real HSB-Lite device UUID. Test fixtures
that need a synthetic UUID (stub modules, cohabitation tests, negative cases) use the
reserved test UUID `01020304-0506-0708-090a-0b0c0d0e0f00` so tests never match a real
device's `.so`.

### ABI check at load time

This framework depends on the application and loadable module being built by the same
toolchain. To mitigate problems caused by violations of this, a test is performed when a
module is loaded. Before init, the adapter calls a C function the module exports and
compares a header sentinel pair against every C++ type that crosses the boundary:

```c
typedef struct hololink_adapter_abi_check {
    uint32_t magic;        /* HOLOLINK_ADAPTER_ABI_MAGIC */
    uint32_t api_version;  /* HOLOLINK_ADAPTER_API_VERSION */
    uint32_t struct_size;  /* sizeof(hololink_adapter_abi_check) */
    uint32_t size_of_enumeration_metadata;
    uint32_t align_of_enumeration_metadata;
    uint32_t size_of_std_string;
    uint32_t align_of_std_string;
} hololink_adapter_abi_check_t;

extern "C" hololink_adapter_abi_check_t hololink_adapter_get_abi_check(void);
```

The leading `magic` / `api_version` / `struct_size` triple is the sentinel — any module
whose header doesn't match the host's expected magic, advertised API version, or
declared struct size is rejected before any C++ type's layout is even read. The
remaining fields catch toolchain / STL drift on the C++ types that flow through the C
ABI (`EnumerationMetadata`, `std::string`); a mismatch on any of them rejects the module
with a diagnostic identifying which type differs. Additional fields may be appended in
future API versions; consumers must use `struct_size` to detect older modules whose
payload does not include the newer fields.

## Service Locator

Both host→module and module→host communication use the same symmetric locator pattern.
Services are identified by a `(instance_id, type_id)` pair:

- `type_id` is the versioned interface name (`"reactor.v1"`, `"i2c.v1"`).
- `instance_id` distinguishes multiple instances of the same type (e.g. I2C buses,
  per-channel data channels). The format is `"name=value;name=value;..."` —
  semicolon-separated `name=value` pairs (e.g. `"serial=abc123;data_plane=0"`,
  `"serial=abc123;bus=1;address=0x42"`). Use `""` for anonymous / singleton services.
  Values are bare — never repeat the key as a prefix on the value (`channel=0`, never
  `channel=channel_0`; `bus=1`, never `bus=bus_1`; `kind=software`, never
  `kind=kind_software`). Keys are short, lowercase, ASCII (`serial`, `channel`, `bus`,
  `address`, `kind`, …).

### C ABI

```c
typedef const void* hololink_adapter_service_t;

typedef hololink_adapter_service_t (*hololink_adapter_get_service)(
    const char* instance_id, const char* type_id);

/* Fires once per matching get_service call when the last shared_ptr the
   host handed out drops. Required (must be non-NULL): even services
   that have no per-instance teardown still receive the notification,
   so they can observe when their reference count returns to its
   "no consumers attached" baseline (e.g. a ReactorV1 with refcount 1
   means no modules currently hold a reference). Implementations that
   genuinely have nothing to do supply an empty function. */
typedef void (*hololink_adapter_release_service)(
    hololink_adapter_service_t instance);
```

### ServiceLocatable CRTP

Every interface inherits from `ServiceLocatable<Derived>` and supplies a
`static constexpr const char* type_id`. The CRTP provides a three-arg
`static std::shared_ptr<Derived> get_service(module, instance_id, allow_null = false)`
that looks up and aliases the typed pointer:

```cpp
template <typename Derived>
class ServiceLocatable {
public:
    static std::shared_ptr<Derived> get_service(
        std::shared_ptr<Module> module,
        const char* instance_id,
        bool allow_null = false);

    // Locked to a shared_ptr<Module> by ServicePublisher at publication
    // time so child accessors (e.g. HololinkInterface::get_i2c) can do
    // T_child::get_service(module(), ...) without threading the Module
    // through every call. Public so callers holding a
    // shared_ptr<Derived> can recover the Module directly.
    std::shared_ptr<Module> module() const { return module_.lock(); }

private:
    // Stamped at publication time (NOT at lookup) by ServicePublisher
    // with the publishing Publisher's self_module(). The wrapped
    // callbacks on that Module loop back into the same binary's
    // Publisher registry, so child lookups resolve in the same binary
    // the parent service was published in. Cross-binary shared_ptr<Module>
    // pinning is structurally impossible — module_ always refers to a
    // Module in the same binary as Derived.
    std::weak_ptr<Module> module_;

    template <typename T> friend class ServicePublisher;
};

class I2cInterfaceV1 : public ServiceLocatable<I2cInterfaceV1> {
    static constexpr const char* type_id = "i2c.v1";
    // ...
};
```

Singleton interfaces (`ReactorV1`, `LoggingInterfaceV1`) hide the inherited three-arg
form with a two-arg `get_service(module, allow_null = false)` overload that passes `""`
internally, so callers cannot supply an `instance_id` the locator would not recognize.

Callers always pass `instance_id` explicitly to the three-arg form — `""` for anonymous,
a named string otherwise. The returned service caches the `shared_ptr<Module>`, so child
accessors on the service reuse the same locator without extra plumbing. Do not fall back
to `module->get_service("", "…")` + `static_cast` in application code — that bypasses
the typed CRTP path everything else in the system relies on.

### Cross-binary `shared_ptr` safety

`std::shared_ptr<Module>` control blocks are allocated in whichever binary constructs
them. `ServiceLocatable<T>::module_` is a `std::weak_ptr<Module>` set at **publication
time** by `ServicePublisher` to the publishing `Publisher`'s `self_module()` — and the
self-Module lives in the same binary as the publisher, which lives in the same binary as
the Derived class. So the weak_ptr always refers to a Module in the same binary as
Derived — no cross-binary pinning is possible. Lookup never writes `module_`. This keeps
both directions safe: a module-resident service's `module_` points at a module-resident
Module; a host-resident service's `module_` points at the host-resident Module.

### Module initialization

```c
typedef struct hololink_adapter_init {
    uint32_t api_version;
    uint32_t reserved_;
    hololink_adapter_get_service get_service;          /* host → module */
    hololink_adapter_release_service release_service;  /* host → module */
} hololink_adapter_init_t;

typedef struct hololink_adapter_module_services {
    hololink_adapter_status_t status;                  /* HOLOLINK_ADAPTER_OK or an error */
    hololink_adapter_get_service get_service;
    hololink_adapter_release_service release_service;
} hololink_adapter_module_services_t;

extern "C" hololink_adapter_module_services_t
hololink_adapter_init(const hololink_adapter_init_t* init);
```

Two cooperating C++ types implement the locator: **`Module`** and **`Publisher`**.

- **`Module`** wraps a `(get_service, release_service)` callback pair coming from a peer
  binary and uses it to look services up over there. It is the lookup-capable type;
  consumers always call `T::get_service(shared_ptr<Module>, instance_id)`. There are
  many Module instances in a process — one per peer relationship.
- **`Publisher`** owns the local-binary registry that services get registered into. It
  exports the static C-ABI thunks the peer binary calls back through, plus
  `register_service()` and `callbacks()`. It does **not** itself perform lookups — it
  exposes `self_module()`, a `shared_ptr<Module>` whose wrapped callbacks are the
  publisher's own thunks, for any code in the same binary that needs to look up
  locally-registered services. There is exactly one `Publisher` per binary, enforced by
  `Publisher::create()`.

Inside `hololink_adapter_init`, the module:

1. Calls `Publisher::create()` and registers its services through `ServicePublisher<T>`
   helpers; `publish()` stamps each service's inherited `ServiceLocatable<T>::module_`
   with `publisher.self_module()` so child accessors later look children up in the same
   registry.
1. Constructs a `shared_ptr<Module>` over the host's callback pair (from
   `init->{get,release}_service`) and reaches host services through the typed factory
   methods (`LoggingInterfaceV1::get_service(host_module)`,
   `ReactorV1::get_service(host_module)`, …).
1. Returns `publisher.callbacks()` as the `hololink_adapter_module_services_t` payload.

The host inspects the returned `status` first: anything other than `HOLOLINK_ADAPTER_OK`
signals init failure (the module reporting it could not bring itself up — missing
dependency, ABI mismatch, hardware unreachable, etc.) and the host rejects the module
with a diagnostic including the status code. When `status` is `HOLOLINK_ADAPTER_OK`,
both callback pointers must be non-NULL — `release_service` is **required** in both
directions (host→module via the `init` struct, module→host via the returned
`hololink_adapter_module_services_t`); it must never be NULL. Services without any
per-instance teardown supply an empty function and still receive notifications, because
the refcount-back-to-baseline event is itself meaningful — for example, a `ReactorV1`
whose reference count drops to 1 (the host's own copy) is observing that no modules
currently hold a reference, which is information a future unload path or a diagnostic
API can act on.

The host follows the same shape: `Adapter::get_adapter()` constructs the host's single
`Publisher` at process startup; host-provided services (Reactor, Logging, …) register
themselves through `ServicePublisher<T>` against that Publisher. When a module is
loaded, `LoadedModule` is constructed as a `Module` whose wrapped callbacks are the
.so's returned `(get_service, release_service)` pair, and the host hands
`host_publisher->callbacks()` into `hololink_adapter_init`.

### Adding a new service

1. Define the interface class inheriting from `ServiceLocatable`; supply `type_id` and
   (for singletons) a two-arg `get_service` overload.
1. Register the concrete implementation from the service owner's side (typically in
   `hololink_adapter_init` for module-provided services, at adapter startup for
   host-provided services).
1. Consumers call `Interface::get_service(module, instance_id)` to look it up.

Both the host and the module can be service providers and consumers — the locator is
symmetric.

### Version naming

There is no unversioned interface alias. Every interface is named by its explicit
version (`I2cInterfaceV1`, `HololinkInterfaceV1`, …) everywhere it is used — interface
declarations, implementations, publish sites, consumer code, and application code. The
version an application depends on is therefore a visible decision at each call site, not
a consequence of which headers it happened to compile against.

This is a deliberate departure from a "track the latest version" alias. An alias like
`using HololinkInterface = HololinkInterfaceV1;` meant to re-point to `V2` later would
let a plain recompile silently raise an application's interface requirement: the
consumer-side `Interface::get_service` call resolves the wire `type_id` from
`Derived::type_id`, so bumping the alias to `V2` changes the requested `type_id` to
`"hololink.v2"`. Against a separately-built module that only publishes `"hololink.v1"`,
the application then fails at runtime (`get_service` throws "no matching service is
registered") with no compile-time signal and no load-time gate — the ABI check verifies
toolchain/STL layout, not interface versions. Naming the version explicitly removes that
recompile-drift; adopting a new version is an intentional edit (`HololinkInterfaceV1` →
`HololinkInterfaceV2`) reviewed at the call site.

This does **not** add load-time interface-version negotiation. An application built
against `V2` and run against a module that only publishes `V1` still fails at runtime
via the same `get_service` throw — the intended "fail fast against an unsupported
configuration" behavior. Explicit naming only guarantees the version requirement cannot
change without a source edit.

Implementation classes derive from the explicitly-versioned interface their name claims:
`HololinkImplV1`, `I2cImplV1`, `SequencerImplV1`, `FrameMetadataImplV1`,
`ReactorImplV1`, `LoggingImplV1`, … each derive from the matching `…V1` interface
(`HololinkInterfaceV1`, `ReactorV1`, `LoggingInterfaceV1`, …). When a `V2` interface is
introduced, an implementation that must support it grows explicitly to satisfy the new
pure virtuals — there is no alias re-point that does it implicitly.

The version-pinned name is the only form used. It appears in:

- Interface declaration lines (`class FooInterfaceV1 : public Service<FooInterfaceV1>`).
- Out-of-line definitions of an interface's own static methods
  (`LoggingInterfaceV1::get_logger() noexcept` in `host/src_module/logging.cpp`).
- Implementation class names and their bases
  (`class HololinkImplV1 : public HololinkInterfaceV1`).
- Consumer code: local variables, member fields, storage types, free-function and method
  declarations, static-method call sites (`HololinkInterfaceV1::get_service(metadata)`,
  `Interface::get_service(module, …)`), helper return types, callback wrappers, and test
  fakes (`class FakeReactor : public ReactorV1`). Fakes track a specific version
  deliberately: when `V2` ships and adds pure virtuals, a `V1`-based fake still compiles
  unchanged and a `V2`-based one is written explicitly.
- `type_id` strings (`"hololink.v1"`, `"reactor.v1"`, …) — wire-format identifiers.
- Tests that verify the V1 contract or its ABI (e.g.
  `tests/hololink_adapter_framework_test.cpp`,
  `tests/hololink_adapter_singletons_test.cpp`).

In Python the same rule holds: the package exports `HololinkInterfaceV1` etc. and no
unversioned name, so an application imports the version it targets.

Template-default factory arguments name the explicit current version
(`get_i2c<T = I2cInterfaceV1>`, `frame_end_sequencer<T = SequencerInterfaceV1>`, …). The
default is frozen and never moves on its own; a caller that wants a future version
writes it out (`get_i2c<I2cInterfaceV2>()`).

#### Cross-version ABI hazard

Because no unversioned alias exists, the principal leak channel — an alias silently
re-pointing under a cross-binary contract — is closed by construction. Two residual
rules keep versioned ABI surfaces honest across separately-built binaries.

##### Rule A — override declarations spell out the base's explicit version

When a derived class overrides a virtual on a versioned base, the override's parameter
types, return type, and any base-class types referenced in the signature must spell out
the same explicit version the base declares (e.g. `SequencerInterfaceV1&`,
`std::unique_ptr<I2cLockV1>&`). The `override` keyword is mandatory on every override:
it turns any mismatch between a derived signature and the base's slot into a compile
error, rather than a silently-shadowed method that leaves the base's pure virtual
unimplemented and traps as a pure-virtual call at runtime.

##### Rule B — `publish<T>(...)` is keyed on the explicit version

Every `ServicePublisher::publish<T>(instance_id, impl)` call spells `T` as the
explicitly-versioned name (`I2cInterfaceV1`, `HololinkInterfaceV1`, …). The published
`type_id` is `T::type_id`, the wire string consumers resolve against, so the explicit
name keeps it stable. The same rule applies to type-trait introspection at publish sites
(e.g. `std::is_base_of<XInterfaceV1, Leaf>::value` static_asserts that gate which `Leaf`
types a publish helper accepts).

A module that intentionally publishes one impl under both V1 and V2 type_ids — so
V1-only and V2-aware consumers both reach the same instance — does so by issuing two
explicit publish calls, one per version. The `publish_*_default` helpers in
`module/core/include/hololink/module/core/publish.hpp` follow this rule: each publishes
under the explicit version it is named for, and a "publish under both" helper composes
them.

### Reactor

The Reactor's primary purpose is to provide **a location where callbacks run with a
thread-safety guarantee**: every callback the Reactor dispatches runs sequentially on a
single thread, so handlers never observe concurrent execution of one another and require
no internal locking against each other. Code that needs "this work must not overlap with
that other work" registers both pieces with the Reactor and gets that serialization for
free.

The familiar entry points — FD callbacks, one-shot alarms, timers, and direct
`add_callback` queueing — are convenient consumers of that facility. They are not the
purpose of the Reactor; they are the surface through which different kinds of work
arrive at the single sequencing thread.

The Reactor is a process singleton — exactly one instance, owned by the adapter, shared
by the application and every loaded module. If each module created its own Reactor,
callbacks from different modules would run on independent threads and lose the
serialization guarantee. `ReactorV1` is how everyone reaches the singleton.

```cpp
class ReactorV1 : public ServiceLocatable<ReactorV1> {
public:
    static constexpr const char* type_id = "reactor.v1";

    static std::shared_ptr<ReactorV1> get_service(
        std::shared_ptr<Module> module, bool allow_null = false);
    static ReactorV1* get_reactor();  // adapter-owned singleton; no refcount

    using Callback   = std::function<void()>;
    using FdCallback = std::function<void(int fd, short events)>;
    struct AlarmEntry { /* opaque */ };
    using AlarmHandle = std::shared_ptr<AlarmEntry>;

    virtual struct timespec now() const = 0;
    virtual void add_callback(std::shared_ptr<Callback> callback) = 0;
    virtual void add_fd_callback(int fd, std::shared_ptr<FdCallback> callback,
                                 short events) = 0;
    virtual void remove_fd_callback(int fd) = 0;
    virtual AlarmHandle add_alarm_s(float seconds,
                                    std::shared_ptr<Callback> callback) = 0;
    virtual AlarmHandle add_alarm(const struct timespec& when,
                                  std::shared_ptr<Callback> callback) = 0;
    virtual void cancel_alarm(AlarmHandle handle) = 0;
    virtual bool is_current_thread() const = 0;
};
```

**Callback lifetime.** Every `add_*` takes its callback as a `std::shared_ptr`. The
Reactor stores the `shared_ptr` for as long as the callback could still be invoked — a
one-shot alarm until dispatch returns (or `cancel_alarm` completes), an FD callback
until `remove_fd_callback` returns *and* any in-flight dispatch finishes. Modules that
bind state into a callback capture whatever `shared_ptr` must outlive any pending
invocation (typically the module's `shared_ptr<Module>`), so while the Reactor holds the
callback, it transitively holds the module. Module unload is safe — wait until the
Reactor has released every callback referencing the module, then proceed.

The Reactor is a synchronization mechanism, not a low-latency event dispatcher. Handlers
run sequentially (no locking required within a handler), but with arbitrary other
handlers queued ahead of them; there are no latency guarantees. If a handler blocks then
no reactor handlers can execute-- which would be considered a bug in that handler, and
Reactor currently has no mitigation for this.

### Logging

A process-wide sink reached through `LoggingInterfaceV1`. The host owns the
implementation (`host/src/logging_impl.cpp`): a self-contained stderr console sink that
parses `HOLOSCAN_LOG_LEVEL` / `HOLOLINK_LOG_LEVEL` for its level and decorates each line
with `gettid()` + a monotonic timestamp. It is **not** a wrapper over
`src/hololink/core/logging.cpp` — the host carries no dependency on legacy core (see
**Host isolation** under Library Layout). Without this, each module would maintain its
own log level and sink, and module log lines would bypass whatever sink the application
installed.

```cpp
enum class LogLevel : int {
    Trace = 10, Debug = 20, Info = 30, Warning = 40, Error = 50,
};

class LoggingInterfaceV1 : public ServiceLocatable<LoggingInterfaceV1> {
public:
    static constexpr const char* type_id = "logging.v1";

    static std::shared_ptr<LoggingInterfaceV1> get_service(
        std::shared_ptr<Module> module, bool allow_null = false);
    static LoggingInterfaceV1* get_logger();  // adapter-owned singleton

    virtual LogLevel level() const = 0;  // caller-side filter
    virtual void log(LogLevel level,
                     const char* file, unsigned line, const char* function,
                     const char* message) = 0;
};
```

The `HSB_LOG_{TRACE,DEBUG,INFO,WARN,ERROR}` macros compare against `level()` before
doing any `fmt::format` work (caller-side short-circuit), then call `log(...)`. The same
macros work for host code and module code; modules cache the `LoggingInterface` pointer
once during `hololink_adapter_init` via `get_service` and log through it for the life of
the `.so`.

The Python package mirrors these as `hsb_log_{trace,debug,info,warn,error}` helpers that
accept a brace-style format string plus `*args`/`**kwargs`, short-circuit on level, and
capture the caller's Python frame for file/line/function.

## Library Layout

The adapter is **always statically linked** into whichever binary consumes it — a C++
application, the `_hololink_adapter.so` pybind extension, or a test executable. There is
no `libhololink_adapter.so`. The framework ships four static targets, plus an optional
fifth for Holoscan-coupled operators:

- **`hololink::adapter_headers`** (`INTERFACE`). Public declarations only
  (`service_locator.h`, `service_locatable.hpp`, `module.hpp`, `publisher.hpp`,
  `logging.hpp`, `reactor.hpp`, `hololink_interface.hpp`, …) plus the transitive
  `fmt::fmt-header-only` include. Consumed by every other target and by every loaded
  module.

- **`hololink::adapter_module`** (`STATIC`). Module-side framework glue — the `Module`
  class, the `Publisher` class with its registry + static C-ABI thunks +
  per-binary-singleton enforcement, `hololink_adapter_get_abi_check`, and the
  `ServicePublisher` helper that stamps `module_` with the publisher's `self_module()`
  at publication time. Every module absorbs its own private copy.

- **`hololink::adapter`** (`STATIC`). Host-side adapter implementation: `Adapter`,
  `ReactorImplV1`, `LoggingImplV1`, the bootp-v2 listener/parser, the `Module` subclass
  that wraps loaded `.so` files, host-side `get_service` / `release_service` callbacks,
  ABI-check wiring. The reactor poll loop, logging sink, bootp parser, and route lookup
  are all adapter-owned (under `host/src/`); this archive **does not** link
  `hololink::core`. Depends only on `hololink::adapter_module`. See **Host isolation**
  below.

- **`hololink::module`** (`STATIC`). Default V1 service implementations, written as
  wrappers over the existing `src/hololink/core/` backend. Emitted from `module/core/`.
  Carries a default-compat-id property that `add_hololink_module()` reads. Depends on
  `hololink::adapter_headers` + `hololink::adapter_module` and on the existing
  `hololink::core` library that `src/hololink/core/` produces. Absorbed privately into
  every module `.so` (one copy per `.so`, isolated by `RTLD_LOCAL`); not consumed
  directly by host applications. Exported through `find_package(hololink_adapter)` so
  out-of-tree partner modules can link it.

- **`hololink::operators`** (`STATIC`, optional). Holoscan-coupled operators that take
  adapter interfaces directly — currently the RoCE receiver operator (consuming a
  `RoceDataChannelInterfaceV1*`) and the Linux software receiver operator (a
  `LinuxDataChannelInterfaceV1*`). The whole operators *tree* (the static lib + the
  `_hololink_adapter_operators` pybind extension) is gated by
  `HOLOLINK_ADAPTER_BUILD_OPERATORS=ON` (the default for the wheel build) — Holoscan is
  the tree gate, since every operator derives from `holoscan::Operator`. *Individual*
  operators inside the tree are then enabled per environment capability: the RoCE
  receiver operator is built only when `HOLOLINK_BUILD_ROCE` is also on, while the Linux
  software receiver operator (no ibverbs dependency) builds whenever the tree does.
  Depends on `hololink::adapter_headers` and Holoscan; not depended on by
  `hololink::adapter` or `hololink::module`, so the framework still builds for partners
  that don't have Holoscan. See **Environment capability gating** below.

All archives compile with `CXX_VISIBILITY_PRESET hidden`,
`VISIBILITY_INLINES_HIDDEN ON`, and `POSITION_INDEPENDENT_CODE ON`, so consumers that
are themselves `.so` files keep adapter symbols private. `HOLOLINK_ADAPTER_EXPORT` marks
types whose vtables / RTTI must be visible across a `.so` boundary. Adapter symbols are
never exported from a consumer binary for third-party `dlsym` — modules talk to the host
only through the C-ABI callback pair passed to `hololink_adapter_init`.

### Host isolation

The host tree (`hololink_adapter/host/` — the `hololink::adapter` archive, its public
headers under `host/include/hololink/adapter/`, and `hololink::adapter_module`) carries
**no** source or link dependency on legacy `src/hololink/{core,common,sensors,csi}`. An
application that links `hololink::adapter` (directly, or via `_hololink_adapter.so`)
therefore pulls in no legacy core; legacy core reaches the process only inside loaded
module `.so` files. This is what keeps an application from being statically pinned to a
single core version — the same coupling the adapter exists to remove.

Concretely, the infrastructure the host needs is adapter-owned, under `host/src/`: the
reactor poll loop (`reactor_impl.cpp` — the `ReactorV1` impl owns its poll thread
directly, as a process singleton that is never freed), the bootp-v2 listener/parser
(`bootp.{hpp,cpp}` + a private `deserializer.hpp`), the general route / MAC lookup
(`networking.{hpp,cpp}`, compiled unconditionally — not RoCE-specific, though its only
current caller is the RoCE-gated `ibv_device.cpp`), the logging sink
(`logging_impl.cpp`), and the CUDA RAII helpers backing `ReceiverMemoryDescriptor`
(`hololink/adapter/cuda_unique.hpp`, replacing `hololink/common/cuda_helper.hpp`). The
bootp parser deliberately does **no** device-specific enrichment — sensor / data-plane
register layout is the loaded module's `EnumerationInterfaceV1::update_metadata`, keyed
on the `data_plane` the parser records.

### Sensor isolation

The sensor drivers (`host/sensors/imx274`, `host/sensors/vb1940`) are isolated by the
same rule: they reference no legacy `hololink::*` and link no `hololink::core` /
`hololink::sensors::*`. They drive hardware only through adapter V1 interfaces
(`HololinkInterfaceV1`, `I2cInterfaceV1`, `SequencerInterfaceV1`,
`OscillatorInterfaceV1`, `VsyncInterfaceV1`). Two pieces make that possible:

- **Adapter-owned CSI types + converter interface.**
  `hololink/adapter/csi_converter.hpp` defines, in namespace `hololink::adapter::csi`
  (mirroring the legacy `hololink::csi`, since these are CSI-domain types, not sensor
  types), the adapter `PixelFormat` / `BayerFormat` (values byte-identical to the legacy
  CSI enums) together with `CsiConverterV1` — a plain abstract base (not a locator
  service, passed by `shared_ptr`) mirroring the legacy 4-method CSI converter contract.
  `Imx274Cam` / `Vb1940Cam` expose
  `configure_converter(std::shared_ptr<CsiConverterV1>)`; no legacy CSI type appears in
  their API.
- **Vendored device data.** The IMX274 register-sequence tables and the VB1940 mode
  sequences + firmware / RAM / certificate blobs are copied into `host/sensors/` as
  adapter-owned data (byte-identical to the legacy source), so the sensors borrow no
  legacy data symbols and link no legacy sensor library.

The adapter ships a native converter implementation, the
`host/operators/csi_to_bayer_op` operator (`hololink::adapter::operators::CsiToBayerOp`)
— both a `holoscan::Operator` and a `CsiConverterV1` — so the example players plug it
straight into the pipeline and hand it to `configure_converter()` with no
application-layer shim. (The earlier `examples/legacy_csi_converter.{hpp,py}` bridge
over the legacy `CsiToBayerOp` has been retired; see the operators follow-up below.)

Known remaining exception, tracked as a follow-up: when FUSA is built, the
`host/operators/` tree links `hololink::core` — the FUSA operator header pulls in
`hololink/core/csi_controller.hpp`, and the RoCE receiver operator wraps the legacy
ibverbs receiver (via `hololink::adapter`). The `CsiToBayerOp` CUDA-helper coupling is
resolved: its `CudaFunctionLauncher` / `CudaContextScopedPush` are now adapter-owned in
`host/operators/cuda_function_launcher.{hpp,cpp}`, so the always-built operators link no
legacy core.

From the adapter's targets a loaded module `.so` links **only**
`hololink::adapter_headers` + `hololink::adapter_module` + `hololink::module` — never
`hololink::adapter`. The module's own `module/<name>/` sources compile on top of those,
and a final `add_hololink_module()` invocation links the combined object set into
`hololink_<uuid>[_<compat>].so`.

Cores absorb privately, never as a shared library — every module links the same
`hololink::module` static archive but each `.so` ends up with its own copy of the
symbols, isolated from other modules' copies by `RTLD_LOCAL`. There is no
"libhololink_module.so".

Modules are otherwise free to link whatever implementation-dependent libraries their
core or supplements need (ibverbs / `librdmacm` for RoCE, CUDA libraries, vendor SDKs,
system services, etc.); those dependencies show up in `ldd` on the module as expected.
The constraint is about adapter coupling, not about what other system / vendor libraries
the code may use.

Module-class symbols come from the module's own absorbed copy of
`hololink_adapter_module.a` plus its absorbed copy of `hololink_module.a`, isolated from
the host's copies via `RTLD_LOCAL` + hidden visibility.

### Environment capability gating

Some functionality depends on a build-time environment capability — most notably RoCE,
which needs ibverbs. On a host with no infiniband devices the build sets
`HOLOLINK_BUILD_ROCE=OFF`, and in that configuration `libibverbs-dev` is not installed,
so `<infiniband/verbs.h>` and the legacy `hololink::operators::RoceReceiver` are absent.
The adapter still builds completely; capability-dependent pieces degrade rather than
break the build. Four rules keep this honest:

1. **Headers and call sites stay capability-neutral.** Publisher headers
   (`hsb_lite_publisher.hpp`, …) *declare* `construct_roce_receiver` (and the 2510
   override) and the `construct_service` chain always *calls* it — neither is
   `#ifdef`'d, and neither names the ibverbs receiver. The method is *defined*
   out-of-line in a build-gated `.cpp` (`module/core/roce_receiver_construct.cpp` for
   the base, `roce_receiver_construct_2510.cpp` for the override): with
   `HOLOLINK_BUILD_ROCE` the body constructs and publishes the functional
   `RoceReceiverV1` / `HsbLite2510RoceReceiverV1` (these TUs are the only place in the
   module tree that includes the legacy receiver, so they alone pull in
   `<infiniband/verbs.h>`); without it the method is a no-op that returns `false`,
   publishing no service. Because defining a publisher method needs the full publisher
   class — which transitively includes a board supplement header (`hsb_lite.hpp`) that
   core's include path lacks — these `.cpp` files are listed in each **per-board
   module**'s `SOURCES` (which carries the supplement path), not compiled into
   `hololink::module`. Each module compiles the construct TU for every publisher class
   in its hierarchy (the base for all; the 2510 override only for
   `module/hsb_lite_2510`). The build selector reaches those TUs the same way it reaches
   the host side (rule 4): a single `PUBLIC` `HOLOLINK_BUILD_ROCE` definition on
   `hololink::module` — inherited by every per-board module through the link, and baked
   into the PCH the modules reuse, so the flag and the precompiled header stay
   consistent without per-source overrides. **Conditional compilation lives only in
   build-gated `.cpp` files and in CMake — never in headers or call sites.**

1. **Don't publish a do-nothing service.** When a capability is absent the method is a
   no-op that returns `false`, so the service's `(instance_id, type_id)` simply isn't
   registered and `get_service` reports a clean miss — there is no stub that pretends to
   be a receiver.

1. **The operators tree always builds; individual operators are enabled per
   capability.** `HOLOLINK_ADAPTER_BUILD_OPERATORS` (≈ Holoscan present) gates the whole
   tree; within it, each operator opts in on its own. The `_hololink_adapter_operators`
   pybind extension and its `hololink_adapter.operators` package **always build and
   import** (when Python + operators are enabled), regardless of RoCE; an operator
   appears as an attribute only when its capability was built (the package `__init__.py`
   re-exports whatever the extension registered). The RoCE receiver operator's binding
   is the only `#ifdef HOLOLINK_BUILD_ROCE` region in `operators_py.cpp`, and only the
   RoCE operator's TU within `hololink::operators` is RoCE-gated; the library itself
   builds whenever `HOLOLINK_ADAPTER_BUILD_OPERATORS` is on, since it also carries the
   Linux software receiver operator, which has no ibverbs dependency and builds
   regardless of `HOLOLINK_BUILD_ROCE`. A new operator adds its own block independently
   and gates (or doesn't gate) its own TU on the capability it needs.

1. **Gate, don't stub, what nothing calls.** `ibv_device_for_peer` (`ibv_device.cpp`,
   links ibverbs) is built only with `HOLOLINK_BUILD_ROCE`, and its sole caller is the
   RoCE receiver operator — itself built only with RoCE. So in a non-RoCE build
   *nothing* calls it; rather than a do-nothing stub, the function and its Python
   binding are simply omitted. The binding in `hololink_adapter_py.cpp` is
   `#ifdef HOLOLINK_BUILD_ROCE`, and the package re-export is guarded by `hasattr`. The
   macro reaches the binding (and the operators extension, and the framework test that
   shares the host PCH) from a single `PUBLIC` definition on `hololink::adapter` — every
   adapter consumer that links it sees the same flag, so the precompiled header stays
   consistent. (Contrast rule 1, where `construct_roce_receiver` *is* always called from
   the dispatch chain and so keeps a real definition that returns `false` without
   publishing — there is no caller to satisfy here.)

**Testing.** Tests that exercise a capability-gated operator carry the existing
`@pytest.mark.accelerated_networking` marker; `tests/conftest.py` skips them when no
infiniband device is present (or `--unaccelerated-only` is passed) — the same condition
that produced `HOLOLINK_BUILD_ROCE=OFF`. The operators-subpackage import test stays
unmarked so the always-importable contract is verified in RoCE-off runs. Compiled and
hardware-binary tests that link the RoCE operator
(`hololink_adapter_roce_receiver_op_test`, the `module_vb1940_player` /
`module_quad_imx274_player` smoke tests) stay CMake-gated on `HOLOLINK_BUILD_ROCE` —
they are not built in that configuration. The Linux software-receiver tests are the
counterexample: `module_linux_imx274_player_test` / `module_linux_vb1940_player_test`
and their Python integration tests (`@pytest.mark.skip_unless_imx274` /
`@pytest.mark.skip_unless_vb1940`, never `@pytest.mark.accelerated_networking`) are
gated only on the per-sensor hardware switch, not on `HOLOLINK_BUILD_ROCE`, so they run
in RoCE-off configurations (see the Linux receiver phase).

### Single-adapter-owner rule

Host-side adapter state (`Adapter` singleton, `ReactorImplV1` poll thread,
`LoggingImplV1` sink) has exactly one instance per process. That follows from the two
supported application shapes:

- **C++ application.** The application binary statically links `hololink::adapter`
  exactly once.
- **Python application.** `import hololink_adapter` loads `_hololink_adapter.so`, which
  has `hololink::adapter` absorbed. Python's import semantics guarantee a single load
  per interpreter.

A configuration where a C++ program both directly links `hololink::adapter` **and**
loads `_hololink_adapter.so` (e.g. a C++ app embedding a Python interpreter that imports
`hololink_adapter`) is not supported. C++ programs that need Python interop use the
Python extension and do not statically link the adapter themselves.

## Versioning

`type_id` strings include a version suffix (`"hololink.v1"`, `"hololink.v2"`, …). A
single implementation may register under multiple type ids for backward compatibility.

**During this project** — while the adapter and first-party modules are under
development — V1 interfaces are not frozen. Implementations may surface method
signatures, parameters, or return-type requirements not yet captured; the V1 class
definition is edited in place and every caller in the tree is updated in the same
commit. Exactly one V1 of each interface exists across the whole project at any moment.

**After the project is complete** — every first-party module builds against the current
V1 surface and passes its hardware tests — each V1 interface is declared frozen. Methods
are never added, removed, reordered, or re-signatured thereafter. New capabilities
become V2 classes, typically inheriting from V1 and adding methods; older consumers
continue to resolve the V1 type id.

### Versioning example — updating `RoceDataChannelInterface`

Two distinct kinds of change show up around an interface like
`RoceDataChannelInterface`. They have very different consequences for versioning.

**Register-map / implementation change (no interface change).** Suppose a new FPGA
revision changes how the receiver-page registers are laid out — e.g. raising the
supported page count from 4 to 4096 by reshaping the register map. In this case, the
FPGA would change its compat-id (or UUID). The interface contract
`RoceDataChannelInterfaceV1::configure(... unsigned pages)` doesn't need to change,
because the new configuration can be configured to work in a specification-compatible
way. In this case, the build tools are configured with the new device ID (e.g.
compat-id) and the implementation of RoceDataChannelInterfaceV1 is updated; after
building, existing applications would continue to work as before.

**API / contract change (new interface version).** The safest implementation of the V1
interface would not allow the application to access more than 4 buffers. To accommodate
those applications that intend to use more, a new RoceDataChannelInterfaceV2 could be
added. Applications needing the larger buffer API would be updated to use this new
interface, and the implementation (probably as a subclass of RoceDataChannelInterfaceV1)
would be written. This new implementation is now registered under both
`"roce_data_channel.v1"` and `"roce_data_channel.v2"`. Applications that don't need the
new capability keep using V1 and continue to work unchanged on every module.
Applications that want the new capability request V2 explicitly; that lookup succeeds
only on newer modules that publish the V2 type id. Because a newer application
explicitly needs the V2 interface, it will fail quickly when run against a configuration
not supporting it.

Version selection — C++ call sites name the version explicitly; the template default is
the frozen current version:

```cpp
std::shared_ptr<RoceDataChannelInterfaceV1> v1 =
    hololink->get_roce_data_channel(md);                      // default == V1, frozen
std::shared_ptr<RoceDataChannelInterfaceV2> v2 =
    hololink->get_roce_data_channel<RoceDataChannelInterfaceV2>(md);  // opt into V2
```

Python mirrors this through the import name: an application imports
`RoceDataChannelInterfaceV1` (or `RoceDataChannelInterfaceV2` once it exists) — there is
no unversioned name to track the latest.

(Note: per-FPGA-revision implementation selection — for the register-map kind of change
above — uses **compat-id**. Devices don't typically publish a new UUID per FPGA
revision; the UUID identifies the device, and compat-id discriminates between revisions
of that device. The `(uuid, compat)` lookup picks up the right module automatically.
HSB-Lite's `module/hsb_lite/` and `module/hsb_lite_2510/` are the example: same UUID,
disambiguated by filename. `module/hsb_lite/` ships as the compat-suffixed
`hololink_<UUID>_2603.so` and serves boards reporting `compat_id=0x2603`;
`module/hsb_lite_2510/` ships as the bare `hololink_<UUID>.so` (via `NO_COMPAT_SUFFIX`)
and serves every other HSB-Lite board the loader doesn't already have a dedicated
compat-suffixed `.so` for. Both link the same `hololink::module`, with `2510`'s
overrides in its module tree handling whatever differs from current core.)

## Adding a partner device

This section walks through bringing a single partner device — one UUID, one piece of
hardware — onto the adapter. The same recipe applies to every other device the partner
ships (each gets its own UUID and its own `module/<device_name>/` tree) and to devices
from other partners. There is no shared "partner" abstraction in the build; each device
is just another module that links `hololink::module`.

The device's module does **not** copy or edit core. It creates a new tree that links
`hololink::module` and contributes whatever the device's board adds (supplements) or
differs in (overrides) on top of core's defaults. The expected workflow:

1. **Create the module tree.** A new `module/<device_name>/` directory in the partner's
   own repo or worktree, modelled on `module/hsb_lite/`. It contains the device's
   `module_entry.cpp` plus any supplement / override sources, and a `CMakeLists.txt`
   that calls `find_package(hololink_adapter)` (or sits in-tree if the device is
   contributed upstream) and invokes
   `add_hololink_module(NAME <device_name> UUID <device-uuid> ...)`. The resulting `.so`
   is named `hololink_<device-uuid>_<compat>.so` by default — the device's UUID, plus
   either the explicit `COMPAT` argument or the `DEFAULT_COMPAT` inherited from
   `hololink::module` rendered as 4-digit lowercase hex.

1. **Contribute supplements.** Anything the device's board has that core's defaults
   don't cover — a device-specific board interface analogous to `HsbLiteInterfaceV1`,
   custom peripherals, board-specific power sequencing — is added as an additional V1
   service registered under a new `(instance_id, type_id)` from the device's
   `module_entry.cpp`.

1. **Contribute overrides.** Where a default V1 service from core doesn't match the
   device's hardware (e.g. a different I2C bus topology, a different data-channel
   register layout), the device's module registers its own implementation under the same
   `(instance_id, type_id)` and shadows core's default at locator-lookup time.

1. **Optionally override the filename suffix.** The compat-id baked into core's CMake
   property is what most devices want — it represents the IP-block revision core tracks
   at head-of-tree. Two opt-outs exist for cases where the default isn't right:

   - Pass `COMPAT <override>` to `add_hololink_module()` when the device's hardware is
     at a different IP-block revision than core, and the device's overrides are doing
     the work of bridging the difference. The explicit compat distinguishes the `.so` so
     the `(uuid, compat)` lookup routes the right device to it. `<override>` is the
     4-digit lowercase hex string form of the compat-id (e.g. `COMPAT 2510`).
   - Pass `NO_COMPAT_SUFFIX` to `add_hololink_module()` when the module is the catch-all
     for its UUID — it should serve every device that the loader doesn't already have a
     dedicated compat-suffixed `.so` for. The resulting `.so` is named bare
     `hololink_<device-uuid>.so` and sits on the loader's fallback path.
     `module/hsb_lite_2510/` uses this — it's the HSB-Lite fallback that catches
     `compat_id=0x2510`, any other compat-id without a dedicated `.so`, and payloads
     that carry no compat-id at all. `COMPAT` and `NO_COMPAT_SUFFIX` are mutually
     exclusive.

1. **Test and publish.** Build, test on the device's hardware, and publish the resulting
   tree (typically an out-of-tree repo or vendor SDK) so users can install the resulting
   `.so` alongside other modules.

**Adding more devices.** A second device from the same partner — different hardware,
different UUID — follows the same five steps in a new `module/<other_device_name>/`
tree. Devices from a third party do the same. Nothing in the build couples one device's
module to another's; they coexist at runtime under distinct `(uuid, compat)` keys, each
absorbing its own private copy of `hololink::module` via `RTLD_LOCAL`.

**Updating to a newer core.** When core advances at head-of-tree and ships against a new
compat-id, the device's module **rebuilds unchanged against the new release** — no
patches to re-apply, because the module never patched core. The `.so` picks up the new
compat-id automatically, sitting alongside earlier compat-id binaries (built from
earlier core releases) so customers with either revision of the hardware can run. A
device that needs to keep shipping for an older revision of its hardware keeps building
against the older core release in parallel.

Out-of-tree builds consume the framework through the installed CMake package —
`find_package(hololink_adapter)` exports `hololink::adapter_headers`,
`hololink::adapter_module`, `hololink::module`, and the `add_hololink_module()` helper —
so a device's `CMakeLists.txt` is roughly the size of `module/hsb_lite/CMakeLists.txt`
with the UUID swapped, plus whatever target the device's supplements / overrides need.

## Constraints

1. **Same toolchain.** All modules and the adapter are built with the same compiler and
   STL implementation. This lets C++ types (e.g. `EnumerationMetadata`) cross the
   boundary directly.
1. **ABI check at load time.** Before init, the adapter verifies the module's C++ type
   layouts match its own; mismatches reject the module with a diagnostic.
1. **Symbol isolation.** Modules load with `dlopen(RTLD_LOCAL)`. No hololink symbols
   leak from a module into the host or other modules.
1. **Unload-safe design; no unload today.** Modules are loaded once and cached for the
   process lifetime. Every Reactor callback is held as a `std::shared_ptr` and captures
   whatever `shared_ptr` keeps module-side state alive while the Reactor holds it, so a
   future unload path is safe — wait for the Reactor to release every callback
   referencing a module before unloading.
1. **Singleton Adapter, single Reactor.** `Adapter::get_adapter()` returns the one
   process-wide Adapter; all modules share one Reactor.
1. **Per-board data channels.** Data-channel instances are specific to a channel on a
   specific board (not singletons). `EnumerationMetadata` identifies both.
1. **One module per `(uuid, compat-id-or-none)`.** The cache key is that tuple; a single
   module instance serves all devices matching it.
1. **Status codes, not exceptions.** Interface methods for retryable operations return
   `hololink_adapter_status_t`. Exceptions are reserved for programming errors and
   unrecoverable conditions. Don't hide exceptions behind `catch (...)`.
1. **Reactor-thread safety.** Enumeration callbacks and anything else dispatched on the
   Reactor thread must not block.
1. **Python GIL discipline on the Reactor thread.** Reactor-thread code that enters
   Python acquires the GIL (`pybind11::gil_scoped_acquire`); host code that blocks on
   Reactor work while holding the GIL releases it first
   (`pybind11::gil_scoped_release`).
1. **Symmetric service locator.** Both directions use the same `(instance_id, type_id)`
   pattern. Singletons use `""` as `instance_id`.
1. **Manual and bootp enumeration share one path.** `Adapter::enumerate()` feeds the
   same pipeline as bootp decoding; no separate object-creation API.
1. **Bootp v2 only.** Bootp v1 (board-ID-to-UUID mapping) is not supported.
1. **`SO_REUSEPORT`** on the bootp socket so other processes on the host can listen on
   the same port independently.

## Code Generation Guidelines

Rules the adapter's source code must follow. These apply to LLM-assisted edits and
hand-written code alike.

### C++ style

- **Prefer explicit types over `auto`.** Write the type name out by default — explicit
  types make signatures locally legible and help reviewers spot upcasts / conversions.
  Two narrow exceptions: (a) the type name already appears on the right-hand side of the
  same statement (`auto loaded = LoadedModule::create(...)`,
  `auto svc = std::make_shared<HsbLiteImpl>(...)`) — repeating it on the left is
  mechanical noise; (b) the deduced type is genuinely un-nameable (lambdas, iterator
  types from local scope).
- **`static` at namespace / file scope, not anonymous namespaces.** File-local helpers
  use `static`. The one exception is file-local *types*, where an anonymous namespace or
  function-local class are the only tools that give internal linkage.
- **Don't hide exceptions.** Avoid `catch (...)` unless required. Let unhandled
  exceptions propagate — `std::terminate` produces a useful message and backtrace.
  Narrow handlers (`catch (const SpecificError&)`) are fine.
- **No plan-phase references anywhere in the codebase.** Comments in C++, CMake, Python,
  and shell sources describe *what* the code does, not *when* (which phase) it was
  added. The same applies to file headers, doc comments, type / target / option names,
  and commit-time scaffolding. Plan-phase references belong in
  `plans/hololink_module.md`, `hololink_module/README.md`, and PR descriptions only.
- **`class` for types with methods, `struct` for plain data.**
- **Prefer named functions over repurposed operators.** E.g. `p.append(segment)` rather
  than `p / segment` for `std::filesystem::path`.

### Service interface convention

Every interface reachable through `get_service` follows the same shape:

```cpp
class FooInterfaceV1 : public ServiceLocatable<FooInterfaceV1> {
public:
    virtual ~FooInterfaceV1() = default;
    static constexpr const char* type_id = "foo.v1";
    // ... pure virtual methods ...
};
```

Consumer code reaches the type through its explicit version name (`FooInterfaceV1`).
There is no unversioned alias — see [Version naming](#version-naming).

#### Per-board supplements are scoped to a HololinkInterface

A board-specific supplement (`HsbLiteInterface`, `LeopardVb1940Interface`, every future
partner-module supplement) is intrinsically scoped to the HololinkInterface it operates
on. The convention is to publish one supplement instance per board, keyed by
`metadata["serial_number"]` — the same key shape `HololinkInterface` uses — so two
boards in one process drive their own hardware through their own supplement instances.

The interface header lives in the module's own include tree, not in the host's:

- File path: `module/<name>/include/hololink/adapter/<name>/<name>.hpp`. Example:
  `module/hsb_lite/include/hololink/adapter/hsb_lite/hsb_lite.hpp`,
  `module/leopard_vb1940/include/hololink/adapter/leopard_vb1940/leopard_vb1940.hpp`.
- Namespace: `hololink::adapter::<name>` — the device-specific name slots into the
  `hololink::adapter` umbrella. So `hololink::adapter::hsb_lite::HsbLiteInterface`,
  `hololink::adapter::leopard_vb1940::LeopardVb1940Interface`.
- The host's `host/include/hololink/adapter/` tree carries no per-board knowledge —
  neither the headers nor the namespaces.

CMake exposes the header through a per-module INTERFACE-library target named
`hololink::<name>::headers` so applications opt in explicitly:

```cmake
add_library(hololink_<name>_headers INTERFACE)
add_library(hololink::<name>::headers ALIAS hololink_<name>_headers)
target_include_directories(hololink_<name>_headers
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(hololink_<name>_headers
    INTERFACE hololink::adapter_headers)
install(DIRECTORY include/hololink/adapter/<name>
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hololink/adapter
    COMPONENT hololink-<name>-headers)
```

Application code that drives the board does
`target_link_libraries(my_app PRIVATE hololink::<name>::headers)` and
`#include "hololink/adapter/<name>/<name>.hpp"`. Applications that do not drive that
board never link the headers target and never see the include directory. Sibling modules
that share a supplement type (e.g. HSB-Lite compat-2510 reusing `HsbLiteInterface` from
compat-2603) link the same INTERFACE target rather than shipping their own header copy.

The supplement's pybind extension links the same INTERFACE header target so its binding
source compiles against the per-module include path.

Construction and lifetime:

- The supplement impl takes the per-board `HololinkInterface` (and any peripheral
  controllers it operates on, e.g. an I2C bus) as constructor arguments and holds them
  as `shared_ptr` members. It does not look up the Hololink lazily on each method call.
- The module's `EnumerationInterface` override constructs the per-board supplement at
  the same point it constructs the per-board Hololink and peripherals (on first sight of
  a `serial_number`) and publishes it under that serial alongside everything else.
- `module_entry.cpp` does **not** publish a singleton supplement at module load — that
  pattern is wrong because a singleton has no Hololink to bind to.

Surface shape for callers:

- The interface header omits the singleton-style two-arg `get_service` override and
  inherits the multi-instance three-arg form from `ServiceLocatable<>`.
- The interface header adds a `get_<name>(module, metadata)` static convenience that
  reads `serial_number` and delegates — same shape `HololinkInterface::get_hololink`
  uses. Throws `std::runtime_error` when the metadata is missing or malformed.
- Supplement methods do not take a metadata argument — the impl already knows its
  Hololink. Methods with a natural per-call payload (e.g. `setup_clock(profile)`,
  `set_camera_power(enabled)`) take only that payload.

Python bindings follow the C++ shape: bind through `bind_multi_instance_get_service<>`
and add a `get_<name>(module, metadata)` static factory that mirrors the C++ helper.

### Testing

- **Tests prove added code paths; nothing more.** A change ships with the tests needed
  to prove the new code paths it added. Don't write exhaustive coverage of pre-existing
  behavior, hypothetical regressions, or every corner of the API just because the test
  framework allows it. The goal is "this change works", not "this whole subsystem is
  exhaustively tested."
- **Framework code does not bend for tests.** No test-only accessors, no
  `friend class XTest`, no `#ifdef TESTING` branches, no public methods added "so a test
  can reach private state", no injectable singletons that have no non-test caller, no
  service-locator backdoors that exist only because a test would otherwise have nothing
  to grip onto. If the only caller of a hook is a test, the hook should not exist;
  rework the test to use the public surface, accept that the path is exercised
  end-to-end by an integration / hardware test rather than a fine-grained unit test, or
  drop the test.
- **Test fixtures are normal users of the public API.** A stub module .so that a test
  loads is built by the same `add_hololink_module()` helper that ships per-board
  modules; its `hololink_adapter_init` constructs a `Publisher`, publishes services
  through `ServicePublisher<T>`, and returns its callbacks the same way a production
  module does. Tests prove the public surface works *because* the fixtures are ordinary
  clients of that surface.
- **Hardware / SDK gates.** Tests that need real hardware, the HSB emulator, the
  Holoscan SDK, or CUDA are gated by the same CMake options that gate the code under
  test (`HOLOLINK_BUILD_TESTS`, `HOLOLINK_ADAPTER_BUILD_OPERATORS`, …). A
  hardware-dependent test does not silently skip when the dependency is missing — it
  doesn't get built at all in that configuration.

### Logging and error messages

- **Write for application developers.** Every string that reaches a log or exception
  reads as an English sentence framed as `"While …: …"`. Example:
  `"While loading hololink module '{}': entry point '{}' could not be resolved ({})."`.
- **Omit internal identifiers.** Don't put C++ method names (`~ReactorImplV1`), syscalls
  (`dlopen`), or init-function names into user-visible strings. Translate to user
  concepts ("entry point", "initialization", "load").
- **Don't use "landed".** Say "implemented", "added", "shipped".

### Python bindings

- **`get_service` takes a Module + `instance_id`.** Non-singleton interfaces take both;
  singletons bind their in-class two-arg override. No `instance_id` default.
- **GIL discipline for callbacks.** Python callables registered on the reactor are
  wrapped in a `shared_ptr` whose deleter reacquires the GIL before releasing the
  underlying `py::object`.
- **Bind the explicit V1 type under its V1 name.** Every `py::class_<...>` for a V1
  service uses the explicitly-versioned type as both the C++ template parameter AND the
  Python-visible class name:
  `py::class_<hololink::adapter::ReactorV1, std::shared_ptr<hololink::adapter::ReactorV1>>(m, "ReactorV1", ...)`.
  Method references follow (`&hololink::adapter::ReactorV1::start`,
  `ReactorV1::AlarmEntry`, `ReactorV1::Callback`). There is no unversioned Python name —
  `__init__.py` re-exports `ReactorV1` only, so a Python application imports the version
  it targets and a rebuild never silently moves it to a newer interface. When a `V2`
  interface ships it is bound and exported as `ReactorV2` alongside `ReactorV1`.
- **GIL release on binding lambdas.** Use `py::call_guard<py::gil_scoped_release>` only
  on direct method delegations whose body returns a primitive (`int`, `bool`, `void`).
  Any lambda that calls `py::make_tuple`, `py::bytes(...)`, `vector_to_bytes`,
  `bytes_to_vector`, or otherwise constructs / decomposes Python objects in the body
  scopes a manual `{ py::gil_scoped_release release; ... }` around just the C++ call —
  `call_guard` releases the GIL for the entire body, which crashes on the surrounding
  Python-object work.
- **Top-level qualified imports in test / application Python.** Use
  `import hololink_adapter` and `hololink_adapter.X` rather than
  `from hololink_adapter import X`. Test extensions are aliased on import:
  `import hololink_adapter._hololink_adapter_test_support as test_support`. The
  package's own `__init__.py` is the legitimate place to re-export with
  `from ._submodule import X`; that re-export is what makes the rule work for everyone
  else.
- **Per-module Python sub-packages.** Per-board supplement bindings ship as separate
  pybind extensions under `hololink_adapter.<module_name>` rather than bundling into the
  core `_hololink_adapter` extension. The C++ source for the binding lives next to the
  module's C++ source — `module/<name>/python/` — and the build stages the resulting
  `_hololink_adapter_<name>.so` plus its `__init__.py` into
  `<build>/python/hololink_adapter/<name>/`. Each sub-package's `__init__.py` imports
  its interface from the local extension and exports it under its explicit V1 name. This
  isolates dependencies the same way the C++ modules do — an application that does not
  need a supplement never loads its Python extension. The PYBIND11_MODULE entry of each
  sub-package extension calls
  `py::module_::import("hololink_adapter._hololink_adapter")` so the core's `Module` /
  `EnumerationMetadata` / etc. type registrations are visible. Shared binding helpers —
  `bind_singleton_get_service<>`, byte-buffer conversion, enumeration-metadata
  conversion — live in `python/include/hololink/adapter/python/` as inline-only headers
  so each per-module extension picks them up without cross-extension link dependencies.

### CMake + build

- **No phase numbering in target or option names.** Option names and install
  destinations are permanent.
- **Project-relative source paths.** Code that embeds a source path in a string emits
  `HOLOLINK_ADAPTER_FILE` (defined in `hololink/adapter/logging.hpp`), never `__FILE__`
  directly. `HOLOLINK_ADAPTER_FILE` defaults to `__FILE__` (the `#define` is guarded by
  `#ifndef`, so a build-supplied value wins), and the host CMake always adds
  `-fmacro-prefix-map=<root>/=` as an INTERFACE compile option on
  `hololink::adapter_headers`, so `__FILE__` renders relative to the project root for
  every consumer. The flag is applied uniformly (each target's precompiled header and
  its sources see the same option), so the PCH stays valid — unlike a per-source
  `-DHOLOLINK_ADAPTER_FILE="…"`, which diverges from the PCH's preprocessor state and is
  dropped with `-Winvalid-pch`. Requires GCC ≥ 8 / Clang ≥ 10. Current call sites: the
  `HSB_LOG_*` macros (`logging.hpp`) and `HOLOLINK_ADAPTER_CUDA_CHECK`
  (`cuda_unique.hpp`).

### Parsing and protocol decoding

- **Chain decode-then-save through `&&`.** Parsers expose file-local
  `save_field(metadata, "key", value)` helpers that return `true` so they slot into the
  same `&&` chain as `Deserializer::next_*`. Every decoded field lands in the metadata
  on the line that consumed it, and the chain short-circuits on first failure.
- **Save decoded values before validity checks.** When a predicate rejects a packet, the
  offending value is already in the metadata so the caller can diagnose what actually
  arrived.
- **Don't duplicate sizes of values being deserialized.** The `Deserializer::next_*`
  bounds check is authoritative — don't wrap it in an external length check.
- **Sub-Deserializer for bounded sections.** Tag-length-value protocols are parsed with
  a fresh `Deserializer` windowed over each section's declared length.
- **Parsers don't clear caller-supplied metadata.** A parser writes only the keys it
  decoded; callers pre-populate context (e.g. the bootp listener writes `interface`,
  `interface_address` from `IP_PKTINFO`) before the call.

### Metadata conventions

- **Opaque identifiers are lowercase-hex strings, not integers.** `compat_id`,
  `hsb_ip_version`, FPGA UUID — published as the exact hex form downstream code uses. A
  16-bit identifier lands as 4 lowercase-hex characters (`"2603"`, not `0x2603`).
  Reserve `int64_t` entries for values that participate in arithmetic.
- **Variable-length fields are trimmed to their valid byte count.** Fixed-layout byte
  arrays with a length field (bootp's 16-byte `chaddr` qualified by
  `hardware_address_length`) are truncated before being saved.
- **Enumeration callbacks have filter-agnostic names.** Every subscriber receives the
  same `EnumerationMetadata` regardless of filter. Use `EnumerationCallback`,
  `EnumerationCallbackHandle`, `unregister(handle)`; the filter appears only in the
  registration method name (`register_ip`, `register_all`).

### Lifetime and singletons

- **Leaky process-wide singletons for cross-library safety.** Singletons that hold raw
  pointers to other singletons (Adapter registers an FD callback on the Reactor) must
  not be stored in a `std::unique_ptr` at static scope — the unique_ptr's destructor
  races with the other singletons' destruction. Store in a `T*` never deleted on the
  exit path; the OS reclaims the memory at teardown.
- **`reset_for_testing()` for deterministic teardown.** Singletons expose a static
  `reset_for_testing()` tests call between cases to start fresh; production never calls
  it.

## C++ Interfaces

All interface methods for retryable operations return `hololink_adapter_status_t`.
Status values:

```c
typedef enum {
    HOLOLINK_ADAPTER_OK = 0,
    HOLOLINK_ADAPTER_NETWORK_ERROR,
    HOLOLINK_ADAPTER_DEVICE_ERROR,
    HOLOLINK_ADAPTER_INVALID_PARAMETER,
    HOLOLINK_ADAPTER_UNSUPPORTED,
    HOLOLINK_ADAPTER_NOT_READY,
    HOLOLINK_ADAPTER_MODULE_ERROR,    /* dlopen / dlsym failure */
    HOLOLINK_ADAPTER_TIMEOUT,
} hololink_adapter_status_t;
```

### Enumeration

```cpp
class EnumerationInterfaceV1 : public ServiceLocatable<EnumerationInterfaceV1> {
public:
    static constexpr const char* type_id = "enumeration.v1";

    // Called on the Reactor thread when a bootp packet arrives for a
    // UUID this module handles. raw_packet may be NULL for manual
    // enumeration paths.
    virtual hololink_adapter_status_t update_metadata(
        EnumerationMetadata& metadata,
        const uint8_t* raw_packet, size_t raw_packet_len) = 0;
};
```

### Hololink control

```cpp
class HololinkInterfaceV1 : public ServiceLocatable<HololinkInterfaceV1> {
public:
    static constexpr const char* type_id = "hololink.v1";

    // The base ServiceLocatable form — (module, instance_id, allow_null).
    using ServiceLocatable<HololinkInterfaceV1>::get_service;

    // Convenience overload: resolves the supplement module through
    // Adapter::get_module(metadata) and builds the per-board
    // "serial=<serial_number>" instance_id from the metadata so
    // application code holding only an EnumerationMetadata doesn't
    // thread a Module pointer through every call site.
    static std::shared_ptr<HololinkInterfaceV1> get_service(
        const EnumerationMetadata& metadata,
        bool allow_null = false);

    // --- Device lifecycle ---
    virtual hololink_adapter_status_t start() = 0;
    virtual hololink_adapter_status_t stop() = 0;
    virtual hololink_adapter_status_t reset() = 0;
    virtual hololink_adapter_status_t configure_hsb() = 0;

    // --- Control plane ---
    virtual hololink_adapter_status_t write_uint32(
        const std::vector<uint32_t>& addresses,
        const std::vector<uint32_t>& values) = 0;
    virtual hololink_adapter_status_t read_uint32(
        const std::vector<uint32_t>& addresses,
        std::vector<uint32_t>& out_values) = 0;
    virtual hololink_adapter_status_t and_uint32(uint32_t address, uint32_t mask) = 0;
    virtual hololink_adapter_status_t or_uint32 (uint32_t address, uint32_t mask) = 0;

    // --- Peripheral bus locking ---
    // Returns an unlocked BasicLockable/Lockable handle compatible with
    // std::lock_guard / std::unique_lock / std::scoped_lock.
    virtual hololink_adapter_status_t i2c_lock(
        std::unique_ptr<I2cLockV1>& out_lock) = 0;

    // --- Child-object factories ---
    // Each factory is a non-virtual template. Default T is the explicit
    // current version (frozen; it never moves on its own). Callers that
    // want a future version spell it out (get_i2c<I2cInterfaceV2>()).
    // allow_null = false throws when the module does not publish the
    // requested (instance_id, type_id); true returns an empty
    // shared_ptr. Instance-id strings come from the protected
    // *_instance_id hooks, overridden per module.
    template <typename T = RoceDataChannelInterfaceV1>
    std::shared_ptr<T> get_roce_data_channel(
        const EnumerationMetadata& md, bool allow_null = false);
    template <typename T = I2cInterfaceV1>
    std::shared_ptr<T> get_i2c(uint32_t bus, uint32_t address,
                               bool allow_null = false);

    // --- Sequencer factories (fixed instance ids) ---
    template <typename T = SequencerInterfaceV1>
    std::shared_ptr<T> software_sequencer(bool allow_null = false);
    template <typename T = SequencerInterfaceV1>
    std::shared_ptr<T> gpio0_sequencer(bool allow_null = false);
    template <typename T = SequencerInterfaceV1>
    std::shared_ptr<T> gpio1_sequencer(bool allow_null = false);
    template <typename T = SequencerInterfaceV1>
    std::shared_ptr<T> sif0_frame_start_sequencer(bool allow_null = false);
    template <typename T = SequencerInterfaceV1>
    std::shared_ptr<T> sif1_frame_start_sequencer(bool allow_null = false);

protected:
    // Per-factory instance-id hooks. Each derived class overrides these
    // to translate factory arguments into instance-id strings its
    // module-side get_service dispatch recognizes. Every hook returns an
    // id in "name=value;name=value;..." form. Every id includes a
    // serial=<serial_number> pair — the per-board scoping that
    // disambiguates services published by two boards sharing the same
    // module. The data-channel hook returns
    // "serial=<serial_number>;data_plane=<n>"; the I2C hook returns
    // "serial=<serial_number>;bus=<bus>;address=<address>", so two
    // boards' I2C bus 1 / 0x42 do not collide.
    virtual std::string roce_data_channel_instance_id(
        const EnumerationMetadata& md) = 0;
    virtual std::string i2c_instance_id(uint32_t bus, uint32_t address) = 0;
};
```

### Sequencer

A Sequencer is a pre-programmed command sequence (reads, writes, polls) that executes
atomically on the FPGA, triggered by a hardware event (frame end, frame start, GPIO) or
software. The `location` method returns the base address in FPGA memory so callers can
read back results after execution.

```cpp
class SequencerInterfaceV1 : public ServiceLocatable<SequencerInterfaceV1> {
public:
    static constexpr const char* type_id = "sequencer.v1";

    virtual unsigned write_uint32(uint32_t address, uint32_t data) = 0;
    virtual unsigned read_uint32(uint32_t address,
                                 uint32_t initial_value = 0xFFFFFFFF) = 0;
    virtual unsigned poll(uint32_t address, uint32_t mask, uint32_t match) = 0;

    virtual hololink_adapter_status_t enable() = 0;
    virtual uint32_t location() = 0;
};
```

Instances come from factory methods on `HololinkInterface` and from data channel
interfaces' `frame_end_sequencer`.

### Data channels

Each data-channel instance is specific to a channel on a specific board (identified by
`EnumerationMetadata`). The adapter exposes RoCE channels through
`RoceDataChannelInterfaceV1`; future transports add their own peer interfaces.

The `instance_id` for a channel is `"serial=<serial_number>;data_plane=<n>"`, where
`<serial_number>` comes from `EnumerationMetadata["serial_number"]` and `<n>` is the
channel number on that board (e.g. `0`, `1`, …). The `serial=` pair scopes the id to one
specific board so two boards in the same process don't collide on `channel=0`; the
`channel=` pair selects which channel on that board. Consumers reach the channel via
`RoceDataChannelInterface::get_service(module, "serial=" + serial + ";data_plane=0")`.
In practice application code rarely builds the id string by hand — the
`HololinkInterface::get_roce_data_channel(metadata)` factory builds it from the metadata
it was handed. The hand-built form above is what the factory produces internally, and it
is what test code and direct `get_service` callers spell out.

```cpp
class RoceDataChannelInterfaceV1
    : public ServiceLocatable<RoceDataChannelInterfaceV1> {
public:
    virtual ~RoceDataChannelInterfaceV1() = default;
    static constexpr const char* type_id = "roce_data_channel.v1";

    // configure() programs the FPGA's RoCE peer with the receiver-side
    // QP number, rkey, frame-memory pointer, and frame layout. Rather
    // than taking each value as a separate argument, configure() takes
    // the receiver object the operator already holds — the channel
    // pulls qp_number / rkey / external_frame_memory / frame_size /
    // page_size / pages from it. This means the operator doesn't have
    // to forward seven independent values, and it lets a per-board
    // RoceDataChannelImplV1 subclass coordinate channel + receiver
    // setup as a single unit if it ever needs to.
    virtual hololink_adapter_status_t configure(
        const EnumerationMetadata& metadata,
        std::shared_ptr<RoceReceiverInterfaceV1> receiver) = 0;
    virtual hololink_adapter_status_t unconfigure() = 0;

    template <typename T = SequencerInterfaceV1>
    std::shared_ptr<T> frame_end_sequencer(bool allow_null = false);
    template <typename T = HololinkInterfaceV1>
    std::shared_ptr<T> get_hololink(bool allow_null = false);

protected:
    virtual std::string frame_end_sequencer_instance_id() = 0;
    virtual std::string parent_hololink_instance_id() = 0;
};
```

### RoCE receiver

`RoceReceiverInterfaceV1` (alias `RoceReceiverV1`) is the per-channel ibverbs receiver
the V1 RoCE operator drives. It wraps the legacy `hololink::operators::RoceReceiver`'s
public surface so per-board / per-FPGA-revision behavior can be overridden through the
adapter's standard service-override mechanism: the default implementation under
`module/core/` delegates to the legacy class verbatim, and a per-supplement override
(e.g. `module/hsb_lite_2510/`) ships a subclass that changes only the behavior its
hardware needs.

`RoceReceiverV1` follows the same service-locator pattern as every other per-channel V1
service. The supplement publishes one receiver instance per `(serial, data_plane)`
alongside the matching `RoceDataChannelInterfaceV1` — the receiver and the channel share
an `instance_id` (`"serial=<serial_number>;data_plane=<n>"`), and
`module/hsb_lite_2510/` overrides by publishing a 2510-specific receiver subclass under
the same id so the locator returns the override automatically.

`RoceReceiverOp` looks up the receiver through the standard `get_service` path keyed by
the enumeration metadata it was handed:

```cpp
auto receiver = hololink::adapter::RoceReceiverV1::get_service(
    module, "serial=" + serial + ";data_plane=" + std::to_string(data_plane));
receiver->start(                               // bring up the QP with runtime params
    ibv_name, ibv_port,
    cu_buffer, cu_buffer_size,
    cu_frame_size, cu_page_size, pages,
    metadata_offset, peer_ip, queue_size);
channel->configure(metadata, receiver);        // channel reads qp_number / rkey /
                                                //   external_frame_memory / frame_size /
                                                //   page_size / pages directly off receiver
// monitor thread runs receiver->blocking_monitor()
// compute() calls receiver->get_next_frame(...)
// shutdown: channel->unconfigure(); receiver->close()
```

The receiver's runtime parameters (ibverbs device name, the CUDA buffer the operator
allocated, frame layout, queue depth, peer IP, …) are application-side choices, so they
flow into the receiver through `start(...)` as individual arguments — never bundled into
a config struct, because the compiler can't force the caller to populate every field of
a struct but it does force them to supply every argument of a function. The supplement
constructs the receiver as a shell at enumerate time; `start()` is when the underlying
`hololink::operators::RoceReceiver` actually comes up.

```cpp
// Subset of the legacy hololink::operators::RoceReceiverMetadata the V1
// operator consumes. The struct is fine here because it is an output
// value the receiver writes — the compiler cannot enforce per-field
// initialization on either input or output structs, but for outputs
// the cost of a missing field shows up at the consumer (read of a
// silently-zero value) the same way an extra return-by-tuple would.
struct RoceReceiverFrameInfoV1 {
    std::uint64_t frame_memory;     // CUdeviceptr to the frame's pixel data
    std::uint64_t metadata_memory;  // CUdeviceptr to the 48-byte EOF block
    std::uint64_t received_frame_number;
    std::uint32_t frame_number;
    std::uint32_t imm_data;
    std::uint64_t received_s;
    std::uint64_t received_ns;
    std::uint64_t rx_write_requests;
    std::uint32_t dropped;
};

class RoceReceiverInterfaceV1
    : public ServiceLocatable<RoceReceiverInterfaceV1> {
public:
    virtual ~RoceReceiverInterfaceV1() = default;
    static constexpr const char* type_id = "roce_receiver.v1";

    // Lifecycle. start(...) constructs the underlying ibverbs
    // receiver from the runtime parameters, brings up the QP, and
    // posts the initial WRs; close() signals the monitor thread to
    // exit; the operator's worker thread runs blocking_monitor()
    // until close() fires. Parameters are passed individually rather
    // than bundled into a struct so the compiler forces the caller to
    // supply every one — a missing struct field is silently
    // zero-initialized and shows up only at runtime.
    virtual hololink_adapter_status_t start(
        const std::string& ibv_name,
        unsigned ibv_port,
        std::uint64_t cu_buffer,
        std::size_t cu_buffer_size,
        std::size_t cu_frame_size,
        std::size_t cu_page_size,
        unsigned pages,
        std::size_t metadata_offset,
        const std::string& peer_ip,
        unsigned queue_size) = 0;
    virtual void close() = 0;
    virtual void blocking_monitor() = 0;

    // Per-frame receive. Returns false on timeout (no frame within
    // `timeout_ms`); on success, fills `info` with the frame's host /
    // device pointers and per-frame statistics the operator stamps
    // onto its outbound metadata map.
    virtual bool get_next_frame(unsigned timeout_ms,
        RoceReceiverFrameInfoV1& info) = 0;
    virtual bool frames_ready() = 0;

    // QP wire-up values + frame-layout the channel reads in
    // configure(metadata, receiver) to program the FPGA. Populated
    // after start(...) returns.
    virtual std::uint32_t get_qp_number() = 0;
    virtual std::uint32_t get_rkey() = 0;
    virtual std::uint64_t external_frame_memory() = 0;
    virtual std::size_t frame_size() = 0;
    virtual std::size_t page_size() = 0;
    virtual unsigned pages() = 0;
};
```

Defaults under `module/core/`:

- `module/core/roce_receiver_default.hpp` — `RoceReceiverImplV1` holds a
  (initially-null) `std::shared_ptr<hololink::operators::RoceReceiver>` and the runtime
  parameters `start(...)` was called with. `start(...)` constructs the legacy receiver
  from those parameters and calls its `start()`; every subsequent V1 method delegates to
  it. `get_next_frame` translates the legacy `RoceReceiverMetadata` into
  `RoceReceiverFrameInfoV1` field-for-field.
- `RoceDataChannelImplV1::configure(metadata, receiver)` reads
  `receiver->get_qp_number()`, `receiver->get_rkey()`,
  `receiver->external_frame_memory()`, `receiver->frame_size()`,
  `receiver->page_size()`, and `receiver->pages()` and feeds them to the existing
  `hololink::DataChannel::configure_roce` call.
- The default `HsbLiteEnumerationImplV1::update_metadata` publishes a
  `RoceReceiverImplV1` per `(serial, data_plane)` under `"serial=…;data_plane=…"`,
  alongside the matching `RoceDataChannelImplV1` it already publishes.

`module/hsb_lite_2510/`'s `HsbLiteEnumerationImplV1::update_metadata` publishes a
`HsbLite2510RoceReceiverImplV1` under the same `(serial, data_plane)` instance_id. The
2510 subclass derives from the legacy `hololink::operators::RoceReceiver` (the legacy
class already has virtual hooks for metadata copy / decode) and overrides only the
specific behavior the 2510 FPGA needs. Because both modules publish under the same
locator key, `RoceReceiverV1::get_service(module, instance_id)` returns the override
automatically when the loader resolved to `module/hsb_lite_2510/`'s `.so`. The public V1
surface that `RoceReceiverOp` consumes stays unchanged.

### Linux receiver

`LinuxReceiverInterfaceV1` (alias `LinuxReceiverV1`) is the per-channel **software**
receiver the V1 Linux operator drives. It wraps the legacy
`hololink::operators::LinuxReceiver`'s public surface. Where `RoceReceiverInterfaceV1`
fronts a hardware ibverbs QP, the Linux receiver reassembles HSB's RoCEv2 UDP packets in
user space over an ordinary datagram socket — so it needs no infiniband device and links
no ibverbs (see the capability-gating note in its phase). It still produces a
software-emulated `qp_number` / `rkey` and still drives the FPGA through
`configure_roce`, so it sits behind its own transport view of the channel exactly as
RoCE does.

Two differences from the hardware path drive the separate `LinuxDataChannelInterfaceV1`
view:

- The FPGA is configured with `distal_memory_address_start = 0`; the software receiver
  adds the local buffer address (`received_address_offset`, which equals the `cu_buffer`
  it was started with) itself. So the channel passes `0` as the device frame-memory
  address, not `external_frame_memory()`.
- The destination UDP port is the host socket's bound `local_port` (read via
  `getsockname` after `configure_socket` binds it), not the fixed `4791` the hardware
  RoCE channel uses.

Service-locator shape mirrors RoCE exactly: the supplement publishes one
`LinuxReceiverInterfaceV1` and one `LinuxDataChannelInterfaceV1` per
`(serial, data_channel)` under the same `"serial=<serial_number>;data_channel=<n>"`
instance_id, alongside the matching RoCE pair and the shared `DataChannelInterfaceV1`
anchor. An application picks software vs hardware transport purely by which operator it
instantiates — `LinuxReceiverOp` resolves the Linux pair, `RoceReceiverOp` the RoCE pair
— with no enumeration-time choice. `module/hsb_lite_2510/` overrides the same way: a
2510-specific subclass published under the same locator key is returned automatically.

`LinuxReceiverOp` looks the services up through the metadata-form `get_service`:

```cpp
auto channel  = LinuxDataChannelInterfaceV1::get_service(metadata);
auto receiver = LinuxReceiverInterfaceV1::get_service(metadata);

int data_socket = ::socket(AF_INET, SOCK_DGRAM, 0);
channel->configure_socket(data_socket);     // bind to the data plane's local interface
receiver->start(                            // construct + run the software receiver
    data_socket,
    cu_buffer, cu_buffer_size,
    cu_frame_size, cu_page_size, pages,
    metadata_offset, queue_size);
channel->attach_receiver(receiver);          // authenticate(qp, rkey) +
                                             //   configure_roce(0, …, local_port)
// monitor thread runs receiver->blocking_monitor()  (legacy LinuxReceiver::run())
// compute() polls receiver->get_next_frame(timeout, info, cuda_stream)
// shutdown: channel->detach_receiver(); receiver->close()
```

As with RoCE, the receiver's runtime parameters flow into `start(...)` as individual
arguments rather than a config struct, so the compiler forces every one to be supplied.
The supplement constructs the receiver as a shell at publish time; `start()` is when the
underlying `hololink::operators::LinuxReceiver` is actually constructed (over the
operator's CUDA buffer and the bound socket) and its `run()` thread begins.

```cpp
// Per-frame info the Linux operator reads after each frame. A superset
// of RoceReceiverFrameInfoV1 reflecting the software receiver's richer
// per-frame accounting (packet / byte counts the kernel-bypass hardware
// path doesn't surface). It is an output struct the receiver fills, so
// — like RoceReceiverFrameInfoV1 — the struct form is acceptable: a
// missing field shows up at the consumer the same way a missing tuple
// element would.
struct LinuxReceiverFrameInfoV1 {
    std::uint64_t frame_memory;     // CUdeviceptr to the frame's pixel data
    std::uint64_t metadata_memory;  // CUdeviceptr to the EOF metadata block
    std::uint64_t received_frame_number;
    std::uint32_t frame_number;
    std::uint32_t imm_data;
    std::int64_t  received_s;
    std::int64_t  received_ns;
    std::uint32_t frame_packets_received;
    std::uint32_t frame_bytes_received;
    std::uint64_t packets_dropped;
};

class LinuxReceiverInterfaceV1
    : public ConfigurableService<LinuxReceiverInterfaceV1> {
public:
    static constexpr const char* type_id = "linux_receiver.v1";

    // serial=<serial_number>;data_channel=<n> — same key as the channel.
    static std::string locator_id(const EnumerationMetadata& metadata);

    virtual ~LinuxReceiverInterfaceV1() = default;

    // Construct the underlying software receiver over `data_socket`
    // (already bound by LinuxDataChannelInterfaceV1::configure_socket)
    // and the CUDA buffer, then start its run() thread. No ibv_name /
    // ibv_port / peer_ip — the socket is the transport, and HSB targets
    // the socket's bound local_port. received_address_offset is
    // cu_buffer (HSB writes from address 0).
    virtual hololink_adapter_status_t start(
        int data_socket,
        std::uint64_t cu_buffer,
        std::size_t cu_buffer_size,
        std::size_t cu_frame_size,
        std::size_t cu_page_size,
        unsigned pages,
        std::size_t metadata_offset,
        unsigned queue_size) = 0;
    virtual void close() = 0;
    virtual void blocking_monitor() = 0;   // legacy LinuxReceiver::run()

    // Per-frame receive. Takes a CUstream — the software receiver does
    // the device-side copy on it (the hardware RoCE path writes via
    // RDMA and needs no stream, hence RoceReceiverInterfaceV1 omits it).
    virtual bool get_next_frame(unsigned timeout_ms,
        LinuxReceiverFrameInfoV1& info, CUstream cuda_stream) = 0;
    virtual bool frames_ready() = 0;

    // QP wire-up + frame-layout the channel reads in
    // attach_receiver(receiver). local_port is the bound socket's port;
    // there is no external_frame_memory() — the Linux channel always
    // passes 0 for the device frame-memory address.
    virtual std::uint32_t get_qp_number() = 0;
    virtual std::uint32_t get_rkey() = 0;
    virtual std::uint32_t local_port() = 0;
    virtual std::size_t frame_size() = 0;
    virtual std::size_t page_size() = 0;
    virtual unsigned pages() = 0;
};
```

`LinuxDataChannelInterfaceV1` is the software-transport view, a sibling of
`RoceDataChannelInterfaceV1` composing the same `DataChannelInterfaceV1` anchor:

```cpp
class LinuxDataChannelInterfaceV1
    : public ConfigurableService<LinuxDataChannelInterfaceV1> {
public:
    static constexpr const char* type_id = "linux_data_channel.v1";

    // Same key as the anchor it wraps — (serial, data_channel).
    static std::string locator_id(const EnumerationMetadata& metadata);

    virtual ~LinuxDataChannelInterfaceV1() = default;

    // Bind the operator-created datagram socket to this channel's data
    // plane (delegates to the backing legacy
    // DataChannel::configure_socket). Must be called before
    // LinuxReceiverInterfaceV1::start(socket, …).
    virtual hololink_adapter_status_t configure_socket(int data_socket) = 0;

    // authenticate(get_qp_number, get_rkey) then
    // configure_roce(0, frame_size, page_size, pages, local_port) — the
    // device frame-memory address is 0 (the receiver adds the local
    // offset) and the destination port is the receiver's local_port.
    virtual hololink_adapter_status_t attach_receiver(
        std::shared_ptr<LinuxReceiverInterfaceV1> receiver) = 0;
    virtual hololink_adapter_status_t detach_receiver() = 0;

    template <typename T = SequencerInterfaceV1>
    std::shared_ptr<T> frame_end_sequencer(bool allow_null = false);
    template <typename T = HololinkInterfaceV1>
    std::shared_ptr<T> get_hololink(bool allow_null = false);
};
```

Defaults under `module/core/`:

- `module/core/linux_receiver_default.hpp` — `LinuxReceiverImplV1` holds an
  (initially-null) `std::shared_ptr<hololink::operators::LinuxReceiver>` plus the
  parameters `start(...)` was called with. `start(...)` constructs the legacy receiver
  from those parameters (passing `cu_buffer` as both the buffer and the
  `received_address_offset`) and spawns its `run()` thread; every subsequent V1 method
  delegates. `get_next_frame` translates the legacy `LinuxReceiverMetadata` into
  `LinuxReceiverFrameInfoV1` field-for-field and forwards the `CUstream`. `local_port()`
  returns the value read from the bound socket.
- `module/core/linux_data_channel_default.hpp` — `LinuxDataChannelImplV1` holds the
  shared `DataChannelInterfaceV1` anchor and the backing legacy `hololink::DataChannel`.
  `configure_socket` delegates to `backing_->configure_socket`; `attach_receiver` issues
  `backing_->authenticate(receiver->get_qp_number(), receiver->get_rkey())` then
  `backing_->configure_roce(0, receiver->frame_size(), receiver->page_size(), receiver->pages(), receiver->local_port())`;
  `detach_receiver` calls `backing_->unconfigure()`. It uses the same `make_backing`
  hook as the RoCE channel, so `module/hsb_lite_2510/`'s `HsbLite2510DataChannel`
  subclass dispatches its `configure_roce` override here too.
- `HsbLiteEnumerationImplV1::update_metadata` publishes a `LinuxReceiverImplV1` and a
  `LinuxDataChannelImplV1` per `(serial, data_channel)` under the same instance_id it
  already uses for the RoCE pair and the anchor.

The legacy `LinuxReceiver`'s metadata handling already routes the EOF block through the
shared decode path, so per-FPGA-revision differences land in the module's
`FrameMetadataInterfaceV1::decode` and the `HsbLite2510DataChannel` `configure_roce`
override (both already present for RoCE) rather than in a Linux-receiver subclass. A
`HsbLite2510LinuxReceiverImplV1` override is added under the same locator key only if
2510 hardware verification surfaces a software-receiver behavior diff; the override seam
exists by construction because both modules publish under the same
`(instance_id, type_id)`.

### I2C lock (BasicLockable / Lockable)

The handle returned by `HololinkInterfaceV1::i2c_lock()` is a mutex-like object
satisfying the standard `BasicLockable` and `Lockable` concepts:

```cpp
class I2cLockV1 {
public:
    virtual ~I2cLockV1() = default;
    virtual void lock() = 0;
    virtual void unlock() = 0;
    virtual bool try_lock() = 0;
};
```

Comes back unlocked; callers acquire and release through standard-library primitives
(`std::lock_guard`, `std::unique_lock`, `std::scoped_lock`). Not a `ServiceLocatable` —
a fresh handle comes back per call. Python binds `__enter__` / `__exit__` to `lock()` /
`unlock()`, plus explicit `.lock()` / `.unlock()` / `.try_lock()`. The handle's
destructor refuses to release a still-held lock (assert/log) — correct code always pairs
`lock`/`unlock`.

### I2C interface

```cpp
class I2cInterfaceV1 : public ServiceLocatable<I2cInterfaceV1> {
public:
    static constexpr const char* type_id = "i2c.v1";

    virtual hololink_adapter_status_t i2c_transaction(
        uint32_t peripheral_address,
        const std::vector<uint8_t>& write_bytes,
        std::vector<uint8_t>& read_bytes) = 0;

    // Encode an I2C transaction into a Sequencer for deferred /
    // event-triggered execution. Indexes identify locations in the
    // sequencer buffer where results can be read back.
    virtual hololink_adapter_status_t encode_i2c_request(
        SequencerInterfaceV1& sequencer,
        uint32_t peripheral_i2c_address,
        const std::vector<uint8_t>& write_bytes,
        uint32_t read_byte_count,
        std::vector<unsigned>& out_write_indexes,
        std::vector<unsigned>& out_read_indexes,
        unsigned& out_status_index) = 0;
};
```

### Frame metadata

Receivers (`RoceReceiver`, future transports) deposit each frame's data directly into a
host-visible buffer. At a known offset within that buffer the device also writes a
fixed-size end-of-frame metadata block — flags, sequence numbers, frame number,
timestamps, byte counts. `FrameMetadataInterfaceV1` is the singleton service that turns
that block into structured fields. Receivers pass a pointer + size, ignoring how the
bytes are laid out; the implementation living in `module/core/` owns the layout.

```cpp
class FrameMetadataInterfaceV1 : public ServiceLocatable<FrameMetadataInterfaceV1> {
public:
    virtual ~FrameMetadataInterfaceV1() = default;
    static constexpr const char* type_id = "frame_metadata.v1";

    static std::shared_ptr<FrameMetadataInterfaceV1> get_service(
        std::shared_ptr<Module> module, bool allow_null = false);

    struct FrameMetadata {
        uint32_t flags;
        uint32_t psn;
        uint32_t crc;
        uint16_t frame_number;
        // Time when the first sample data for the frame was received.
        uint64_t timestamp_s;
        uint32_t timestamp_ns;
        uint64_t bytes_written;
        // Time the metadata block itself was emitted.
        uint64_t metadata_s;
        uint32_t metadata_ns;
    };

    // Size in bytes of the end-of-frame metadata block this
    // implementation decodes. Callers use it to size the host
    // buffer they hand to decode(...); constant for the lifetime
    // of the implementation.
    virtual size_t block_size() const = 0;

    // Decode the end-of-frame metadata block at host_memory into
    // out_metadata. host_memory_size_bytes is the size of the block
    // available at host_memory; INVALID_PARAMETER is returned when
    // it is too small to contain a valid block.
    virtual hololink_adapter_status_t decode(
        const void* host_memory,
        size_t host_memory_size_bytes,
        FrameMetadata& out_metadata) const = 0;
};
```

`decode` is `const` and stateless, so the singleton is safe to call concurrently from
receiver threads. It does not touch the device — the metadata block is already in host
memory by the time `decode` is invoked. `block_size()` reports the device-defined block
layout the implementation owns so receivers don't hard-code the byte count themselves.

### Oscillator

Each data plane needs an on-board reference clock (typically the IMX274 / equivalent
sensor's pixel clock). `OscillatorInterfaceV1` is the per-data-plane V1 service the
supplement publishes for that — one instance per `(serial, data_plane)`, keyed by the
same `"serial=<serial_number>;data_plane=<n>"` instance_id the matching
`RoceDataChannelInterfaceV1` uses, so applications fetch the oscillator alongside the
channel through the standard locator.

```cpp
class OscillatorInterfaceV1 : public ServiceLocatable<OscillatorInterfaceV1> {
public:
    static constexpr const char* type_id = "oscillator.v1";

    // The base ServiceLocatable form — (module, instance_id, allow_null).
    using ServiceLocatable<OscillatorInterfaceV1>::get_service;

    // Convenience overload: resolves the supplement module through
    // Adapter::get_module(metadata) and builds the per-data-plane
    // "serial=<serial_number>;data_plane=<data_plane>" instance_id
    // from the metadata, so application code holding only an
    // EnumerationMetadata doesn't thread a Module pointer through.
    static std::shared_ptr<OscillatorInterfaceV1> get_service(
        const EnumerationMetadata& metadata,
        bool allow_null = false);

    virtual ~OscillatorInterfaceV1() = default;

    // Program the oscillator for `clocks_per_second`. Returns true
    // if the requested rate is achievable on this hardware, false
    // if the configuration is not possible.
    virtual bool enable(uint64_t clocks_per_second) = 0;

    // Tunable knobs the oscillator exposes — implementation-defined
    // keys (e.g. "spread_spectrum", "output_swing") mapped to their
    // current values. Both keys and values are strings so the
    // interface stays free of implementation-specific types.
    virtual std::map<std::string, std::string> get_caps() = 0;

    // Apply a set of cap updates. All-or-nothing — returns true
    // only if every requested key was recognized and accepted.
    virtual bool set_caps(const std::map<std::string, std::string>& caps) = 0;
};
```

Two data planes can share a single on-board clock generator chip; the per-data-plane
oscillators are responsible for coordinating that. HSB-Lite's default implementation
delegates to `HololinkImplV1::enable_clock(...)` (module-private helper) which caches
the rate the first caller commits and rejects subsequent calls that ask for a different
rate. The same wrapper registers an `on_reset` listener so the cached rate is dropped
when the underlying hololink is reset, and the next `enable_clock(rate)` re-programs the
generator from scratch.

The sensor driver invokes the oscillator during configuration — e.g. `Imx274Cam` takes
an `OscillatorInterfaceV1` at construction and calls `enable(25_000_000)` from inside
`configure(...)` before touching the sensor's registers. Applications never call
`enable` directly.

## Host-Side Classes

`Module` is a thin C++ wrapper around a
`(hololink_adapter_get_service, hololink_adapter_release_service)` callback pair. It
doesn't know whether those callbacks live in a `.so` the adapter just loaded, in the
host's own service table, or in a test fake — all locators present the same
`get_service(instance_id, type_id)` contract. Callers pass Modules around as
`shared_ptr<Module>` so a service fetched through one keeps it alive long enough to
reach the locator.

The `.so`-loaded subclass that owns the dl handle is a private implementation detail;
`Adapter::get_module` returns a `shared_ptr<Module>`.

```cpp
namespace hololink::adapter {

class Module {
public:
    Module(hololink_adapter_get_service get_service_cb,
           hololink_adapter_release_service release_service_cb);
    virtual ~Module() = default;

    // Query the locator for a service instance. Throws std::runtime_error
    // when not found unless allow_null = true. Returned shared_ptr's
    // deleter invokes Module::release_service when the last reference
    // drops — higher-level callers go through ServiceLocatable<T>::get_service
    // which aliases into a typed shared_ptr<T>.
    std::shared_ptr<const void> get_service(
        const char* instance_id, const char* type_id,
        bool allow_null = false) const;

    void release_service(
        hololink_adapter_service_t instance) const noexcept;

private:
    hololink_adapter_get_service get_service_;
    hololink_adapter_release_service release_service_;
};

class Adapter {
public:
    static Adapter& get_adapter();
    ~Adapter();

    std::shared_ptr<Module> get_module(
        const EnumerationMetadata& metadata) const;

    // Bootp-driven: blocks for up to `timeout` until enumerate()
    // observes an announcement for `peer_ip` *after this call has
    // started waiting*, consumes the slot, and returns the
    // (enriched-if-known-UUID) metadata. Any pending announcement
    // observed before this call is discarded — wait_for_channel
    // always returns a fresh bootp message, never a stale cached
    // one. Throws on timeout.
    EnumerationMetadata wait_for_channel(
        const std::string& peer_ip,
        std::chrono::milliseconds timeout);

    // Bootp-driven async. Callback receives enriched metadata
    // regardless of which register_* method produced the registration.
    EnumerationCallbackHandle register_ip(
        const std::string& ip,
        std::function<void(EnumerationMetadata&)> callback);
    EnumerationCallbackHandle register_all(
        std::function<void(EnumerationMetadata&)> callback);
    void unregister(const EnumerationCallbackHandle& handle);

    // Manual: submit metadata through the same pipeline bootp decoding uses.
    EnumerationMetadata enumerate(const EnumerationMetadata& metadata);

private:
    Adapter();
    std::map<std::string, std::shared_ptr<Module>> uuid_to_module_;
};

} // namespace hololink::adapter
```

### Application usage

```cpp
Adapter& adapter = Adapter::get_adapter();
EnumerationMetadata metadata = adapter.wait_for_channel(
    "192.168.0.100", std::chrono::seconds(30));
std::shared_ptr<Module> module = adapter.get_module(metadata);

std::shared_ptr<HololinkInterface> hololink =
    HololinkInterface::get_hololink(module, metadata);
hololink->start();

std::shared_ptr<RoceDataChannelInterface> roce_channel =
    hololink->get_roce_data_channel(metadata);
roce_channel->configure(metadata, qp_number, rkey,
                        frame_memory, frame_size, page_size, pages);

std::shared_ptr<I2cInterface> i2c = hololink->get_i2c(bus, address);
i2c->i2c_transaction(peripheral, write_bytes, read_bytes);

// Pin explicitly when the app must guarantee behaviour against a
// specific version:
std::shared_ptr<RoceDataChannelInterfaceV1> pinned =
    hololink->get_roce_data_channel<RoceDataChannelInterfaceV1>(metadata);
```

Or start from a data channel directly — it caches its `shared_ptr<Module>`, so
`get_hololink()` needs no extra plumbing. The instance_id for a channel is
`"serial=<serial_number>;data_plane=<n>"`, where `<serial_number>` comes from
`EnumerationMetadata["serial_number"]` and `<n>` is the channel number on that board:

```cpp
const std::string& serial =
    std::get<std::string>(metadata.at("serial_number"));
std::shared_ptr<RoceDataChannelInterface> roce_channel =
    RoceDataChannelInterface::get_service(
        module, "serial=" + serial + ";data_plane=0");
std::shared_ptr<HololinkInterface> hololink = roce_channel->get_hololink();
```

### Python error model

The pybind11 wrappers translate the C++ conventions to idiomatic Python:

- **Output parameters become return values.** `read_uint32(addresses, out_values)` →
  `values = hololink.read_uint32(addresses)`;
  `i2c_transaction(peripheral, write_bytes, read_bytes)` →
  `read_bytes = i2c.i2c_transaction(peripheral, write_bytes)`; etc.
- **Errors raise Python exceptions.** Non-OK status codes translate to exceptions.
- **`get_service` is called transparently.** Factory methods (`get_i2c`,
  `get_roce_data_channel`, …) are wrapped so Python sees the typed interface object
  directly — `i2c = hololink.get_i2c(bus=1, address=0x1A)` with no explicit
  `get_service` step.

## Module-Side Implementation

Each module `.so` is built from two cooperating source trees:

- **`module/core/`** — repo-wide shared wrapper layer. Holds default V1 implementations
  of every generic interface (`HololinkInterfaceV1`, `RoceDataChannelInterfaceV1`,
  `I2cInterfaceV1`, `SequencerInterfaceV1`, `EnumerationInterfaceV1`,
  `FrameMetadataInterfaceV1`) implemented as wrappers over the existing
  `src/hololink/core/` backend, which is unchanged. Compiles to a static archive
  `hololink::module` and declares the head-of-tree default compat-id as a CMake property
  on that target.

- **`module/<name>/`** — per-device tree. Names the device UUID, optionally overrides
  the compat-id, contributes any board-specific supplements (additional V1 services that
  core doesn't provide) and overrides (replacement implementations of core's defaults
  where the device's hardware diverges), and emits the `module_entry.cpp` that wires the
  registered set into `hololink_adapter_init`. Always small; the size of a module tree
  scales with how far the device diverges from core.

Adding a new device is done by **creating `module/<name>/`** that links
`hololink::module`, declares the UUID, contributes whatever supplements / overrides the
device requires, and emits its `module_entry.cpp`. No core copy. This is the only
mechanism by which a new module — first-party or partner — is created.

The only framework targets a module's CMake reaches outside its own tree are:

- `hololink::adapter_headers` — V1 interface declarations, service locator types, status
  codes, the `fmt` transitive include for `HSB_LOG_*`.
- `hololink::adapter_module` — `Module` and `Publisher` classes, the
  `hololink_adapter_get_abi_check` symbol, and the `ServicePublisher` helper. Absorbed
  privately via `RTLD_LOCAL` so each module carries its own copy.
- `hololink::module` — the shared backend + default V1 services. Absorbed privately the
  same way.
- The `add_hololink_module()` CMake helper (a reusable function, not a library).

The default V1 implementations live in core. They follow the plain-delegation pattern —
wrapping core's backend types and translating exceptions to status codes:

```cpp
// In module/core/, used by every module that links hololink::module.
class HololinkImplV1 : public HololinkInterfaceV1 {
public:
    HololinkImplV1(std::shared_ptr<backend::Hololink> hololink)
        : hololink_(std::move(hololink)) {}

    hololink_adapter_status_t start() override {
        hololink_->start();
        return HOLOLINK_ADAPTER_OK;
    }
    hololink_adapter_status_t write_uint32(
            const std::vector<uint32_t>& addresses,
            const std::vector<uint32_t>& values) override {
        if (!hololink_->write_uint32(/* ... */)) {
            return HOLOLINK_ADAPTER_NETWORK_ERROR;
        }
        return HOLOLINK_ADAPTER_OK;
    }
    // ...
private:
    std::shared_ptr<backend::Hololink> hololink_;
};
```

A module-side **supplement** is a fresh V1 service registered under a new
`(instance_id, type_id)` pair — for example, HSB-Lite's `HsbLiteInterfaceV1` is added
from `module/hsb_lite/` because it represents the HSB-Lite *board's* clock setup, which
core knows nothing about. A partner module contributes a partner-specific board
interface from `module/<device_name>/` the same way.

A module-side **override** is a V1 service registered under the same
`(instance_id, type_id)` as one of core's defaults — registration order in
`module_entry.cpp` arranges for the module's implementation to win the locator lookup,
so consumers reach the override transparently. `module/hsb_lite_2510/` uses this for the
V1 services whose backing register layout differs from current HSB-Lite; the unchanged
services continue to come from core.

## Directory Layout

```
hololink_adapter/
├── host/
│   ├── include/hololink/adapter/       # Public headers (adapter consumers + modules)
│   │     hololink_interface.hpp        #   V1 interfaces
│   │     status.h                      #   hololink_adapter_status_t
│   │     service_locator.h             #   C ABI typedefs + init struct
│   │     service_locatable.hpp         #   CRTP
│   │     module.hpp                    #   Module class
│   │     publisher.hpp                  #   Publisher + ServicePublisher
│   │     adapter.hpp                   #   Adapter class
│   │     reactor.hpp                   #   ReactorV1
│   │     logging.hpp                   #   LoggingInterfaceV1 + HSB_LOG_* macros
│   │     enumeration_metadata.hpp
│   │     serializer.hpp / deserializer.hpp
│   ├── src/                            # hololink::adapter
│   │     adapter.cpp                   #   bootp listener, UUID→Module map
│   │     module.cpp                    #   dlopen + ABI check + init
│   │     reactor_impl.cpp              #   ReactorImplV1 (single poll thread)
│   │     logging_impl.cpp              #   LoggingImplV1 (console sink)
│   │     bootp_parser.cpp
│   ├── src_module/                     # hololink::adapter_module
│   │     module_base.cpp               #   Module + Publisher classes,
│   │                                   #   abi_check, ServicePublisher
│   ├── operators/                      # hololink::operators (optional —
│   │   │                               #   HOLOLINK_ADAPTER_BUILD_OPERATORS)
│   │   ├── roce_receiver_op.hpp/.cpp   #   RoCE receiver: takes
│   │   │                               #   RoceDataChannelInterfaceV1*
│   │   └── python/
│   │         operators_py.cpp          # _hololink_adapter_operators.so
│   ├── python/
│   │     hololink_adapter_py.cpp       # _hololink_adapter.so — core native ext
│   ├── sensors/                        # Per-camera C++ drivers over adapter handles
│   │   ├── imx274/
│   │   │     imx274_cam.hpp/.cpp       #   Imx274Cam over HololinkInterfaceV1
│   │   │     imx274_mode.hpp           #   re-exports legacy register tables
│   │   │     csi.hpp                   #   adapter-side PixelFormat / BayerFormat
│   │   │     li_i2c_expander.hpp       #   I2C expander for HSB-Lite carrier
│   │   │     CMakeLists.txt
│   │   └── vb1940/
│   │         vb1940_cam.hpp/.cpp       #   Vb1940Cam + Vb1940_Mode enum +
│   │                                   #   re-exports of legacy register / firmware
│   │                                   #   tables (all in vb1940_cam.hpp)
│   │         csi.hpp
│   │         CMakeLists.txt
│   └── CMakeLists.txt
│
├── python/                             # Pure-Python sub-packages, ship in wheel
│   └── sensors/
│       └── imx274/
│             __init__.py
│             imx274_cam.py             #   Imx274Cam over adapter handles
│             imx274_mode.py
│             li_i2c_expander.py        #   I2C expander helper for HSB-Lite
│
├── cmake/
│   └── HololinkModule.cmake            # add_hololink_module() — wraps add_library
│                                       #   to emit a module .so whose OUTPUT_NAME
│                                       #   is built from UUID and the resolved
│                                       #   compat-id (core default or per-module
│                                       #   override).
│
└── module/                                    # Shared core + per-device modules
    ├── core/                                  # Shared static archive (hololink::module).
    │   │                                      #   V1-shaped wrappers over the existing
    │   │                                      #   src/hololink/core/ backend (which is
    │   │                                      #   unchanged); only the default V1
    │   │                                      #   implementations live here.
    │   ├── enumeration_default.hpp/.cpp       #   Default EnumerationInterfaceV1
    │   ├── hololink_default.hpp/.cpp          #   Default HololinkInterfaceV1
    │   ├── roce_data_channel_default.*
    │   ├── i2c_default.*
    │   ├── sequencer_default.*
    │   ├── frame_metadata_default.hpp/.cpp    #   Default FrameMetadataInterfaceV1
    │   │                                      #   (decodes end-of-frame block)
    │   └── CMakeLists.txt                     #   add_library(hololink_module STATIC ...)
    │                                          #   sets DEFAULT_COMPAT property = 2603
    │
    ├── hsb_lite/                              # HSB-Lite reference module — minimal
    │   ├── hsb_lite_impl.hpp/.cpp             #   HsbLiteInterfaceV1 (clock setup) —
    │   │                                      #   supplement, new (instance, type)
    │   ├── module_entry.cpp                   #   extern "C" hololink_adapter_init —
    │   │                                      #   registers core defaults + supplement
    │   └── CMakeLists.txt                     #   add_hololink_module(NAME hsb_lite
    │                                          #                       UUID 889b7ce3-...)
    │                                          #   compat inherited from core default
    │
    ├── hsb_lite_2510/                         # Same UUID, ships under the bare
    │                                          #   `hololink_<UUID>.so` filename via
    │                                          #   add_hololink_module(... NO_COMPAT_SUFFIX
    │                                          #   ...). Catches compat_id=0x2510, any
    │                                          #   other compat without a dedicated .so,
    │                                          #   and payloads with no compat-id. Carries
    │                                          #   override impls for the V1 services
    │                                          #   whose register map differs from current
    │                                          #   HSB-Lite; otherwise uses the same core
    │                                          #   defaults.
    │
    ├── leopard_vb1940/                        # Leopard VB1940 module — bare-UUID .so
    │   ├── module_entry.cpp                   #   defines LeopardVb1940Publisher +
    │   │                                      #   LeopardVb1940V1 + LeopardVb1940-
    │   │                                      #   OscillatorV1 inline (all module-
    │   │                                      #   private) and instantiates the
    │   │                                      #   publisher in hololink_adapter_init
    │   ├── include/hololink/adapter/leopard_vb1940/
    │   │     leopard_vb1940.hpp               #   LeopardVb1940InterfaceV1 declaration
    │   │                                      #   (public — apps + Python sub-package
    │   │                                      #   import it)
    │   └── CMakeLists.txt                     #   add_hololink_module(NAME leopard_vb1940
    │                                          #     UUID f1627640-… NO_COMPAT_SUFFIX ...)
    │                                          #   bare-UUID exception: the Leopard
    │                                          #   FPGA doesn't publish a compat-id
    │                                          #   over bootp today.
    │
    └── <device_name>/                        # Out-of-tree partner device — UUID,
                                                #   optional compat override,
                                                #   device-specific supplements/
                                                #   overrides. No core copy. Same
                                                #   pattern repeats per device.
```

New examples and tests for `hololink_adapter` live in the **existing top-level
directories**, alongside their pre-rework siblings — no new file is created under any
existing path other than `examples/` and `tests/`:

- `examples/module_imx274_player.py` — the 4-channel IMX274 example using the new
  adapter APIs (sits next to the existing `imx274_player.py` /
  `stereo_imx274_player.py`).
- `tests/<adapter test sources>` — gtests and Python tests that exercise the adapter
  framework, the wrapper layer, the new operator, and the new sensor driver.

Adding files there does require small edits to `examples/CMakeLists.txt` (append the new
`.py` to `EXAMPLE_INSTALL_FILES`) and `tests/CMakeLists.txt` (register the new tests via
the existing `ConfigureTest()` helper). These are the only files in the existing tree
this plan modifies, alongside the top-level `CMakeLists.txt` adding
`add_subdirectory(hololink_adapter)`.

## Build Integration

- **`hololink::adapter`** (static archive) — Adapter + Module loader + Reactor + Logging
  \+ bootp parser, all adapter-owned. This archive carries no dependency on legacy
  `hololink::core` (see **Host isolation**); it depends only on
  `hololink::adapter_module`. Does not reference `hololink::module`.
- **`hololink::adapter_module`** (static archive) — module-side framework glue. Every
  loaded module absorbs its own private copy via `RTLD_LOCAL`.
- **`hololink::module`** (static archive) — default V1 implementations as wrappers over
  the existing `src/hololink/core/` backend, plus the default-compat-id CMake property.
  Emitted by `module/core/CMakeLists.txt`. Depends on the existing `hololink::core`
  library. Absorbed privately into every module `.so`.
- **`hololink::operators`** (static archive, optional) — Holoscan-coupled operators that
  take adapter interfaces directly. Built only when
  `HOLOLINK_ADAPTER_BUILD_OPERATORS=ON` (default ON for the wheel). Depends on Holoscan;
  the framework targets above do not.
- **`_hololink_adapter.so`** (pybind extension) — statically absorbs `hololink::adapter`
  \+ `hololink::adapter_module`.
- **`_hololink_adapter_operators.so`** (pybind extension, optional) — wraps
  `hololink::operators`. Built only with `HOLOLINK_ADAPTER_BUILD_OPERATORS=ON`.
- **`hololink_adapter.sensors.imx274`** (pure-Python sub-package) — IMX274 sensor driver
  and I2C expander helper, takes adapter handles. No native extension. Ships in the
  wheel under `site-packages/hololink_adapter/sensors/imx274/`.
- **Each module `.so`** — links `hololink::adapter_module` + `hololink::module`
  privately, compiles its own supplement / override sources, does **not** link
  `hololink::adapter`. Reaches host services only through the callback pair received in
  `hololink_adapter_init`. The compat-id baked into the output filename comes from
  core's default-compat property unless the module's `add_hololink_module()` invocation
  passes `COMPAT <override>`.

## Distribution and Install Layout

**CMake install (`cmake --install`):**

| Path                                                                                          | Artifact                                                                                                                                           |
| --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `<prefix>/lib/libhololink_adapter.a`                                                          | `hololink::adapter` static archive                                                                                                                 |
| `<prefix>/lib/libhololink_adapter_module.a`                                                   | `hololink::adapter_module` static archive                                                                                                          |
| `<prefix>/lib/libhololink_module.a`                                                           | `hololink::module` static archive                                                                                                                  |
| `<prefix>/lib/libhololink_operators.a` (optional)                                             | `hololink::operators` static archive — present when `HOLOLINK_ADAPTER_BUILD_OPERATORS=ON`                                                          |
| `<prefix>/lib/hololink/modules/hololink_<uuid>[_<compat>].so`                                 | Each first-party module's loadable `.so`                                                                                                           |
| `<prefix>/include/hololink/adapter/*`                                                         | Adapter public headers                                                                                                                             |
| `<prefix>/lib/cmake/hololink_adapter/HololinkModule.cmake`                                    | `add_hololink_module()` helper for out-of-tree module builds                                                                                       |
| `<prefix>/lib/cmake/hololink_adapter/hololink_adapterConfig.cmake` (+ `-version`, `-targets`) | `find_package(hololink_adapter)` — exports `hololink::adapter`, `hololink::adapter_module`, `hololink::module`, and `hololink::operators` if built |

**Python wheel (`pip install hololink_adapter`):**

```
site-packages/hololink_adapter/
├── __init__.py
├── _hololink_adapter.so                # core native extension
├── _hololink_adapter_operators.so      # operators native extension (when built)
├── sensors/
│   └── imx274/                         # pure-Python IMX274 sensor driver
│         __init__.py
│         imx274_cam.py
│         imx274_mode.py
│         li_i2c_expander.py
├── modules/
│   ├── hololink_889b7ce3-…_2603.so     # hsb_lite module (compat 0x2603)
│   ├── hololink_889b7ce3-….so          # hsb_lite_2510 fallback module
│   ├── hololink_f1627640-….so          # leopard_vb1940 module (bare-UUID
│   │                                   # exception: FPGA doesn't publish compat)
│   └── …
├── lib/                                 # static archives for out-of-tree module
│   ├── libhololink_adapter_module.a     #   builds against the pip-installed package
│   ├── libhololink_module.a
│   └── cmake/hololink_adapter/          #   find_package(hololink_adapter) reaches these
└── include/hololink/adapter/*           # adapter headers (reachable via
                                         # hololink_adapter.get_include())
```

A C++ app linked against the CMake install carries one copy of `hololink::adapter`. A
Python app has `_hololink_adapter.so` as its core native extension, with the optional
`_hololink_adapter_operators.so` alongside it when operators are built.

## Build Phases

The work is split into deliverable-oriented phases. Each phase produces a coherent slice
of the system that builds and is tested on its own. The phases below are intentionally
coarse — each will be broken into smaller pieces (commits, sub-phases) during
implementation.

**Completion checklist (per phase and per sub-phase).** When a phase — or any of its
sub-phases — is finished, append a section to `hololink_adapter/README.md` that:

1. Names the phase.
1. Summarizes what was implemented in one paragraph. Plan-phase references belong in the
   README and PR descriptions only — never in source code, CMake, or any other file in
   the codebase (see "No plan-phase references anywhere in the codebase" under Code
   Generation Guidelines).
1. Lists the build commands needed to produce the artifacts from that phase.
1. Lists the test-run commands that exercise the phase's deliverables (unit tests,
   integration tests, hardware demos as applicable).

Phases run in order — later phases depend on the artifacts of earlier ones. Phases 6 and
7 are independent — either can land in either order — and both gate Phase 8. Phase 5b
sits between Phase 5 and Phase 7 (Phase 7's IMX274 driver needs the adapter Python
bindings Phase 5b establishes).

A phase may ship in slices when verification of part of its deliverable depends on
infrastructure (CUDA, the Holoscan SDK, the HSB emulator, real hardware) that's only
available in some build environments. In those cases the structural / interface work
lands in the originally-numbered phase and the hardware- or external-SDK-coupled
follow-up lands later. The per-phase sections in `hololink_adapter/README.md` are the
authoritative record of what was actually delivered when; this document records the
intended end-state.

**Current status.** All eight phases have implementations or structural skeletons. The
Phase 8 smoke test is always registered in ctest when the build includes Python +
operators; it skips at runtime by default (returns 77 / SKIPPED) and runs only when the
user opts in with `HOLOLINK_TEST_IMX274=1` against a runner with 2 HSB-Lite boards × 2
IMX274 cameras. Every code path the example exercises is now in place; what remains is
execution against the hardware. Each phase section below has an "As-built status" note
recording what was actually delivered in slices; the per-phase sections in
`hololink_adapter/README.md` carry the corresponding build / test-run commands.

### Phase 1 — Framework skeleton

**Deliverable.** The minimal scaffolding everything else builds against. Wire
`hololink_adapter/` into the top-level build via `add_subdirectory(hololink_adapter)`.
Land the C ABI (`hololink_adapter_status_t`, `hololink_adapter_init_t`,
`hololink_adapter_module_services_t` with the `status` field), the `ServiceLocatable<T>`
CRTP, the `Module` class (callback-pair locator), the `Publisher` class (per-binary
registry + static C-ABI thunks + `self_module()`), the `Adapter` singleton,
`dlopen(RTLD_LOCAL)` with the ABI check, and the `add_hololink_module()` CMake helper.

**Done when.** A stub `.so` built with `add_hololink_module()` loads, exchanges a no-op
service through the symmetric locator (host → module and module → host), and unloads
cleanly. The ABI check rejects a deliberately-mismatched stub.

### Phase 2 — Host singletons (Reactor + Logging)

**Deliverable.** `ReactorV1` (single poll thread, FD callbacks, alarms, single-threaded
callback contract) and `LoggingInterfaceV1`, both backed by adapter-owned
implementations under `host/src/` (no dependency on `src/hololink/core`; see **Host
isolation**). Both reachable from a module via the typed `T::get_service(host_module)`
path; both registered on the host side at adapter startup.

**Done when.** The Phase 1 stub schedules a Reactor callback that runs and emits an
`HSB_LOG_*` line through the host sink (`HOLOSCAN_LOG_LEVEL` / `HOLOLINK_LOG_LEVEL` env
vars honored).

### Phase 3 — Enumeration

**Deliverable.** `EnumerationInterfaceV1`, `EnumerationMetadata` (with a typed
`get<T>(key)` helper), and the manual `Adapter::enumerate(metadata)` /
`Adapter::find_channel(channel_ip)` / `Adapter::set_module_directory(dir)` API. UUID →
cached Module mapping keyed by `(uuid, compat-id)` with the
`hololink_<uuid>[_<compat>].so` lookup against the configured module directory
(`HOLOLINK_MODULE_DIR` env var, falling back to `/usr/lib/hololink/modules/`).

**Out of scope (originally).** The bootp v2 parser and the bootp UDP socket attached to
the Reactor.

**As-built status.** `Adapter::start_bootp_listener(port=12267)` opens a UDP socket
(with `SO_REUSEADDR` / `SO_REUSEPORT`), calls the adapter-owned `configure_bootp_socket`
(IP_PKTINFO + bind), and registers an `FdCallback` against the host `ReactorV1`. Parsing
is the adapter-owned `receive_bootp(fd)` (`host/src/bootp.{hpp,cpp}` + a private
`deserializer.hpp`): it `recvmsg`s the datagram and writes every known bootp header /
NVDA-vendor / v2-body field — including `compat_id` and `fpga_uuid` — directly into a V1
`EnumerationMetadata`, then hands it plus the raw packet bytes to
`Adapter::enumerate(metadata, raw_packet, raw_packet_len)` so modules see the original
bytes via `EnumerationInterfaceV1::update_metadata`. The parser does **no**
device-specific (sensor / data-plane) enrichment — that is the module's
`update_metadata`, keyed on the `data_plane` the parser records. The host bootp path no
longer calls the legacy `hololink::Enumerator` (see **Host isolation**).

**Done when.** A synthetic bootp packet (or a manual `enumerate()` call) finds a stub
module that claims the UUID, and `Adapter::find_channel(channel_ip=…)` returns the
channel's `EnumerationMetadata`.

### Phase 4 — V1 service surface + `module/core/` wrappers (interface headers + 2 of 6 default impls)

**Deliverable.** Every V1 abstract interface header under
`host/include/hololink/adapter/` — `HololinkInterfaceV1`, `RoceDataChannelInterfaceV1`,
`I2cInterfaceV1`, `I2cLockV1`, `SequencerInterfaceV1`, `FrameMetadataInterfaceV1`. The
`module/core/` source tree + `hololink::module` static archive + head-of-tree
`DEFAULT_COMPAT 2603` CMake property. Two default implementations:

- **`I2cLockImplV1`** — backed by a caller-supplied `std::shared_ptr<std::mutex>`,
  faithfully implementing `lock` / `unlock` / `try_lock` so the handle plugs into
  `std::lock_guard` / `std::unique_lock` / `std::scoped_lock`.
- **`FrameMetadataImplV1`** — owns the layout of the device's 48-byte end-of-frame
  metadata block and decodes it directly into the V1 `FrameMetadata` struct, rejecting
  undersized buffers and null pointers up-front.

**Out of scope (originally).** The four hardware-coupled default impls.

**As-built status.** All four landed under `module/core/` as header-only delegating
wrappers in a follow-up:

- `HololinkImplV1` (`hololink_default.hpp`) wraps a `LegacyHololinkAccess` — a tiny
  subclass of `hololink::Hololink` that re-exposes the legacy class's protected
  `configure_hsb` / `and_uint32` / `or_uint32` (those three are protected on the legacy
  class but published on V1's surface). `i2c_lock` returns an `I2cLockNamedV1` over the
  legacy `NamedLock&` (also lives in the same header).
- `I2cImplV1` (`i2c_default.hpp`) — `i2c_transaction` delegates directly;
  `encode_i2c_request` returns `INVALID_PARAMETER` until a virtual hook lands that
  recovers the legacy `Hololink::Sequencer&` from an arbitrary `SequencerInterfaceV1`
  (the `-fno-rtti` build forbids `dynamic_cast`).
- `SequencerImplV1` (`sequencer_default.hpp`) — direct delegation of every V1 method.
- `RoceDataChannelImplV1` (`roce_data_channel_default.hpp`) — wraps
  `std::shared_ptr<hololink::DataChannel>` plus a copy of the `EnumerationMetadata`;
  `configure(...)` calls legacy `authenticate` then `configure_roce(local_data_port=0)`.

These wrappers have no new tests in this slice — Phase 8's hardware integration is the
canonical proof of correctness.

**Done when.** Unit tests prove `I2cLockImplV1` follows the BasicLockable / Lockable
concept rules and serializes threads against a shared mutex, and prove
`FrameMetadataImplV1` decodes a synthetic 48-byte block field-for-field while rejecting
undersized buffers and null pointers.

### Phase 5 — HSB-Lite module (structure)

**Deliverable.** `module/hsb_lite/` claiming the real HSB-Lite UUID
`889b7ce3-65a5-4247-8b05-4ff1904c3359`, built via `add_hololink_module()` so the .so
emits as `hololink_<UUID>_2603.so` (compat-suffixed; the helper inherits
`DEFAULT_COMPAT=2603` from `hololink::module` when no `COMPAT` argument is given).
Public abstract `HsbLiteInterfaceV1` under the per-module include path
(`module/hsb_lite/include/hololink/adapter/hsb_lite/hsb_lite.hpp`) with a static
`get_hsb_lite(module, metadata)` convenience and a `setup_clock(profile)` virtual. The
module's `module_entry.cpp` constructs a `Publisher`, publishes the
`FrameMetadataImplV1` singleton (carried over from `module/core/`), and publishes its
own `HsbLiteEnumerationImplV1` as the `EnumerationInterfaceV1` override that stamps
`module_name=hsb_lite` and backfills `compat_id=0x2603` when the bootp payload didn't
carry one. `hololink::hsb_lite::headers` is exposed as a CMake INTERFACE target so
applications and per-module Python sub-packages can link the supplement type.

**Out of scope (originally).** The four hardware-coupled wrappers from Phase 4; the
concrete `HsbLiteImplV1` supplement; the `hololink_adapter.hsb_lite` per-module Python
sub-package; the per-board + per-channel publication chain; the end-to-end Python
script.

**As-built status.** All landed except real-hardware verification:

- The four hardware-coupled wrappers ship under `module/core/` (see Phase 4 as-built
  block).
- `HsbLiteImplV1` (`module/hsb_lite/hsb_lite_impl.hpp`) holds a `LegacyHololinkAccess`
  and implements `setup_clock(profile)` by delegating to the legacy
  `Hololink::setup_clock`.
- `HsbLiteEnumerationImplV1::update_metadata` now publishes the full per-board +
  per-channel V1 service chain. Per first-seen serial: `HololinkImplV1` +
  `HsbLiteImplV1` (under `serial=<n>`), plus `I2cImplV1` for `(bus=1, address=0x1A)`
  (IMX274) and `(bus=1, address=0x70)` (LI I2C expander). Per first-seen
  `(serial, channel)`: HSB-Lite-specific channel addresses backfilled from the channel
  index, legacy `hololink::DataChannel` constructed against the cached
  `LegacyHololinkAccess`, wrapped in `RoceDataChannelImplV1` (under
  `serial=<n>;data_plane=<n>`), and the legacy `frame_end_sequencer()` wrapped in
  `SequencerImplV1` (under `serial=<n>;data_plane=<n>;kind=frame_end`).
- The `hololink_adapter.hsb_lite` per-module Python sub-package
  (`module/hsb_lite/python/`) builds `_hololink_adapter_hsb_lite.so` via
  `pybind11_add_module`, wraps `HsbLiteInterface` with the static
  `get_hsb_lite(module, metadata)` factory and the `setup_clock(clock_profile)` method,
  and stages alongside `__init__.py` under `${BUILD}/python/hololink_adapter/hsb_lite/`.
  `HsbLiteInterfaceV1::get_hsb_lite` was moved to an inline header function so the
  pybind extension doesn't need to link the module's `.so`.
- `add_hololink_module()` was updated to stage every `hololink_<UUID>.so` into
  `${BUILD}/lib/hololink/modules/` so applications point `--module-dir` at one
  consistent location.

The end-to-end Python script lands as Phase 8's `module_imx274_player.py`. The
hardware-equipped smoke test that drives it is the only remaining outstanding piece.

**Done when.** A manual `Adapter.enumerate(metadata)` loads the HSB-Lite .so by UUID,
the `HsbLiteEnumerationImplV1` override stamps the metadata,
`Adapter.find_channel(peer_ip)` returns the enriched result, and the
`FrameMetadataInterface` singleton fetched through the loaded `Module` decodes a known
48-byte block correctly.

### Phase 5 extension — `hsb_lite_2510` per-compat-id module + naming convention

**Deliverable.** A second HSB-Lite module under `module/hsb_lite_2510/` that shares the
FPGA UUID with `module/hsb_lite/`, reuses the public `HsbLiteInterfaceV1` supplement
type (no second Python sub-package), and ships as the bare `hololink_<UUID>.so` so it
acts as the Adapter loader's catch-all for every HSB-Lite board that doesn't have a
dedicated compat-suffixed `.so`. Internally a near-clone of `module/hsb_lite/` with
`DEFAULT_COMPAT_ID=0x2510` as the fallback for missing-from-metadata cases.

Alongside the module, `add_hololink_module()` is updated to be the single source of
truth for `.so` filename composition. Resolution order:

1. `NO_COMPAT_SUFFIX` option → bare `hololink_<UUID>.so`. Used by
   `module/hsb_lite_2510/`.
1. `COMPAT <hex>` argument → `hololink_<UUID>_<hex>.so`. `<hex>` is the 4-digit
   lowercase hex string form of the compat-id.
1. `DEFAULT_COMPAT` target property on `hololink::module` → same compat-suffixed
   filename, inheriting the head-of-tree value (`2603`). Used by `module/hsb_lite/` and
   the bootp-stub test module (which passes `COMPAT 2603` explicitly).

`NO_COMPAT_SUFFIX` and `COMPAT` are mutually exclusive; specifying neither (and having
no `DEFAULT_COMPAT` set) is a configure error.

Compat-id values across the adapter follow a single convention: **numeric form** (stored
in `EnumerationMetadata["compat_id"]` as `int64`, parsed from the wire as a
little-endian uint16; e.g. `0x2603`, `0x2510`) versus **string form** (used in CMake
properties, `.so` filenames, and human-facing logs; the 4-digit lowercase hex rendering
of the numeric value: `"2603"`, `"2510"`). `Adapter::load_module_for` formats
`compat_id` as `%04x` when composing the filename it dlopens; this is the only place the
conversion happens, because applications no longer compose `.so` paths themselves.

To insulate applications from the filename convention entirely, `Adapter` exposes a
public `get_module(const EnumerationMetadata&)` (Python: `adapter.get_module(metadata)`)
that reads `fpga_uuid` + `compat_id` from the metadata and returns the same cached
`Module` the loader resolved during `enumerate()`. All example code uses this API; the
manual `adapter.load_module(path)` overload remains available for tests that want to
verify the path-keyed cache behavior directly.

**Out of scope.** Per-compat-id behavioral divergence inside `module/hsb_lite_2510/`
(addresses, clock profile, supplement implementation) — the slice landed is structural
only, with the same enumeration logic as `module/hsb_lite/`. Real behavior deltas land
when 2510 hardware verification surfaces the differences. The first known divergence — a
behavior change in `hololink::operators::RoceReceiver` that the 2510 FPGA requires — is
captured under "Phase 6 extension — `RoceReceiverV1` abstraction + per-supplement
override" below, which introduces the override seam (a `RoceReceiverInterfaceV1` service
the V1 operator fetches via `get_service`, published per-`(serial, data_plane)` by each
supplement so `module/hsb_lite_2510/` shadows the default at locator-lookup time).

**Done when.** `module/hsb_lite_2510/` builds as `hololink_<UUID>.so`,
`module/hsb_lite/` builds as `hololink_<UUID>_2603.so`, the hsb_lite gtest
(`tests/hololink_adapter_hsb_lite_test.cpp`) covers both compat-id `0x2603` (loads
`hololink_<UUID>_2603.so`) and `0x2510` (falls through the loader to
`hololink_<UUID>.so`) end-to-end with explicit `compat_id` in the metadata, and the
Python smoke test still loads the compat-`0x2603` module through the `hololink_adapter`
extension.

### Phase 5b — Core pybind bindings

**Deliverable.** A `_hololink_adapter` pybind11 extension under
`host/python/hololink_adapter_py.cpp` exposing the slice of the core surface needed by
Python applications driving the adapter: `Adapter` (singleton accessor +
`set_module_directory` / `enumerate` / `find_channel` / `load_module`), `Module` (opaque
holder for the typed `T.get_service(module, ...)` calls), `EnumerationMetadata` with
full `__getitem__` / `__setitem__` / `__contains__` / `__len__` / `get(key, default)`
mapping the `int64 / str / bytes` variant alternatives onto Python types, the
`FrameMetadata` plain-data struct, and `FrameMetadataInterfaceV1` (static
`get_service(module)` + `decode(host_memory)` accepting any buffer-protocol object). The
accompanying `__init__.py` exports the version-unspecific names (`Adapter`,
`EnumerationMetadata`, `FrameMetadata`, …) plus the explicit V1 interface names
(`FrameMetadataInterfaceV1`, …) — there is no unversioned interface alias. CMake
plumbing under `host/CMakeLists.txt` builds the extension via
`include(hololink_deps/pybind11)` (which finds an installed pybind11 or falls back to
FetchContent at v2.13.6) and stages the .so + `__init__.py` together under
`${BUILD}/python/hololink_adapter/` so
`PYTHONPATH=${BUILD}/python python3 -c "import hololink_adapter"` resolves the package.

**Out of scope (originally).** Bindings for the V1 interfaces that had no concrete
implementation yet (`HololinkInterfaceV1`, `RoceDataChannelInterfaceV1`,
`I2cInterfaceV1`, `I2cLockV1`, `SequencerInterfaceV1`, `ReactorV1`,
`LoggingInterfaceV1`, `EnumerationInterfaceV1`); the per-module
`hololink_adapter.hsb_lite` Python sub-package; the Holoscan-coupled operator pybind
extension (Phase 6).

**As-built status.** The `_hololink_adapter` pybind extension grew bindings for every
abstract V1 service interface — `I2cLockV1`, `I2cInterfaceV1`, `SequencerInterfaceV1`,
`RoceDataChannelInterfaceV1`, `HololinkInterfaceV1`, `EnumerationInterfaceV1`,
`LoggingInterfaceV1` (with the `LogLevel` enum), plus a consumer-only `ReactorV1` class
and the opaque `AlarmEntry` token type. Each abstract interface (other than `ReactorV1`)
gets a pybind trampoline so Python can subclass it. Where the C++ ABI uses in/out vector
parameters that don't translate directly, the trampoline reshapes the call (e.g.
`I2cInterfaceV1.i2c_transaction` is exposed as
`(peripheral_address, write_bytes, read_byte_count) -> bytes` on both sides).
`__init__.py` exports each interface under its explicit V1 name only. `ReactorV1` is
consumer-only (host publishes the C++ `ReactorImplV1`); `HololinkInterfaceV1.i2c_lock`
is callable from Python but the trampoline rejects Python overrides — V1's
`unique_ptr<I2cLockV1>` ownership transfer doesn't bridge cleanly from a Python-owned
override return.

The `hololink_adapter.hsb_lite` and `hololink_adapter.operators` per-module Python
sub-packages also landed (in their respective phases).

**Done when.** A pytest under `tests/` imports `hololink_adapter`, drives
`Adapter.enumerate` against the HSB-Lite .so from Phase 5, observes the enriched
metadata fields back across the binding, and confirms the explicit V1 interface names
(e.g. `FrameMetadataInterfaceV1`) are importable while the unversioned names are not.

### Phase 6 — RoCE receiver operator

**Deliverable.** `host/operators/roce_receiver_op.{hpp,cpp}` — a `holoscan::Operator`
subclass that takes `RoceDataChannelInterfaceV1*`, configures the QP via the channel's
`configure(...)`, posts receive buffers, and decodes per-frame metadata via
`FrameMetadataInterfaceV1`. Built only when `HOLOLINK_ADAPTER_BUILD_OPERATORS=ON`. The
`_hololink_adapter_operators.so` pybind extension wraps the operator for Python
consumers.

**Done when.** A unit test constructs the operator against a
`RoceDataChannelInterfaceV1` mock that satisfies its public surface (no test-only hooks
added to the framework to make this possible), and the operator's pybind class
instantiates from Python with adapter-flavored arguments.

**As-built status.** The structural operator landed first (parameters declared,
`start()` calling `channel->configure(...)` with placeholder zeros, `compute()` a
no-op). The ibverbs receiver thread landed as a follow-up: `start()` allocates a
`hololink::ReceiverMemoryDescriptor`, constructs a legacy
`hololink::operators::RoceReceiver` over the buffer, calls `receiver->start()` to bring
up the QP + post WRs, then calls
`channel_->configure(metadata, qp_number, rkey, frame_memory, ...)` with the values the
receiver came up with. A monitor thread runs `RoceReceiver::blocking_monitor()`.
`compute()` calls `get_next_frame(timeout=1000ms)`, copies the 48-byte EOF block from
device memory to a stack buffer via `cuMemcpyDtoH`, decodes via the V1
`FrameMetadataInterfaceV1::decode` (so modules own the metadata layout), wraps the
just-received frame buffer in a `nvidia::gxf::Tensor` (uint8 vector,
`MemoryStorageType::kDevice`, `wrapMemory` with a no-op release), stamps the V1
`FrameMetadata` fields onto the operator's metadata map by name, and emits the entity
over `"output"`. The shape matches legacy `BaseReceiverOp` so it drops into the existing
CsiToBayer → ImageProcessor → BayerDemosaic → HolovizOp pipeline.

**Infrastructure.** The operator subclasses `holoscan::Operator` and so requires the
Holoscan SDK headers and libraries to build at all.
`HOLOLINK_ADAPTER_BUILD_OPERATORS=ON` gates the operator targets in CMake; build
environments without Holoscan can configure with the option off and skip this phase's
targets entirely.

### Phase 6 extension — `RoceReceiverV1` abstraction + per-supplement override (planned)

**Motivation.** `module/hsb_lite_2510/` needs to change a behavior in
`hololink::operators::RoceReceiver` that the 2510 FPGA requires. The structural Phase 6
operator constructs `hololink::operators::RoceReceiver` directly from
`RoceReceiverOp::start()` — there is no override seam, so a per-board behavior diff has
nowhere to live. The fix is to introduce a `RoceReceiverInterfaceV1` abstraction (see
"RoCE receiver" under C++ Interfaces) that wraps the legacy class's public surface, and
have `RoceReceiverOp` consume the V1 interface through the standard service locator
instead.

**Deliverable.**

- `host/include/hololink/adapter/roce_receiver.hpp` — abstract `RoceReceiverInterfaceV1`
  as a `ServiceLocatable<RoceReceiverInterfaceV1>` with `type_id = "roce_receiver.v1"`.
  Its `start(...)` takes the runtime parameters individually (ibv_name, ibv_port,
  cu_buffer, cu_buffer_size, cu_frame_size, cu_page_size, pages, metadata_offset,
  peer_ip, queue_size) rather than via a config struct — the compiler enforces each one.
  The header also defines the `RoceReceiverFrameInfoV1` output struct that mirrors the
  subset of the legacy `RoceReceiverMetadata` the V1 operator reads (`frame_memory`,
  `metadata_memory`, frame counters / timestamps, drop counter).
- `RoceDataChannelInterfaceV1::configure` simplifies from seven args
  (`metadata, qp_number, rkey, frame_memory, frame_size, page_size, pages`) to two —
  `configure(metadata, shared_ptr<RoceReceiverInterfaceV1> receiver)` — and reads every
  value off the passed-in receiver.
- `module/core/roce_receiver_default.hpp` — `RoceReceiverImplV1` holds a
  (initially-null) `shared_ptr<hololink::operators::RoceReceiver>` plus the runtime
  parameters `start(...)` was called with. `start(...)` constructs the legacy receiver
  from those parameters and calls its `start()`; every subsequent V1 method delegates.
  `get_next_frame` translates the legacy `RoceReceiverMetadata` into
  `RoceReceiverFrameInfoV1` field-for-field. The default
  `HsbLiteEnumerationImplV1::update_metadata` publishes one `RoceReceiverImplV1` per
  `(serial, data_plane)` under the same `"serial=…;data_plane=…"` instance_id it already
  uses for the matching `RoceDataChannelImplV1`.
- `module/hsb_lite_2510/` — its `HsbLiteEnumerationImplV1::update_metadata` publishes a
  `HsbLite2510RoceReceiverImplV1` under the same `(serial, data_plane)` instance_id the
  default would have used. The 2510 subclass derives from the legacy
  `hololink::operators::RoceReceiver` and overrides only the specific behavior the 2510
  FPGA needs (e.g. `copy_metadata_to_host` / `get_frame_metadata`, which are already
  virtual on the legacy class for exactly this purpose). Service-locator lookup resolves
  to the override automatically because both modules publish under the same
  `(instance_id, type_id)`. The supplement also constructs the legacy
  `hololink::DataChannel` it wraps in `RoceDataChannelImplV1` as a
  `HsbLite2510DataChannel` (a subclass of the legacy class), so
  `RoceDataChannelImplV1::configure(metadata, receiver)`'s call to
  `backing_->configure_roce(...)` virtual-dispatches through the 2510 subclass — the
  hook point for channel-side behavior diffs. This requires marking
  `hololink::DataChannel::configure_roce` `virtual` and adding a
  `virtual ~DataChannel()` (the only changes to `src/hololink/core/`); it mirrors the
  receiver pattern, where the legacy class's hookable methods are already virtual.
- `host/operators/roce_receiver_op.cpp` — replace the direct
  `std::make_shared<hololink::operators::RoceReceiver>(...)` call in `start()` with a
  `RoceReceiverV1::get_service(module, instance_id)` lookup followed by
  `receiver->start(ibv_name, ibv_port, cu_buffer, …, queue_size)` (the same ten args the
  legacy constructor used to take, passed positionally). The `instance_id` is built from
  `metadata["serial_number"]` + `metadata["data_plane"]`, matching the channel's id.
  Then call `channel_->configure(metadata, receiver_)` — no more separate qp_number /
  rkey / frame_memory args. The op holds `std::shared_ptr<RoceReceiverV1>` and calls
  only V1 methods. The monitor thread body becomes `receiver_->blocking_monitor()`;
  `compute()` calls `receiver_->get_next_frame(timeout, info)` and reads
  `info.frame_memory` / `info.metadata_memory` instead of the legacy struct's fields.

**Out of scope.**

- Behavioral content of the 2510 override (which legacy virtual the subclass actually
  overrides, what it returns). This phase wires up the override seam; the specific 2510
  behavior lands when 2510 hardware verification surfaces what the diff needs to be.
- Mocking out `RoceReceiverV1` for unit testing. The `module/core/` default still goes
  through the legacy `hololink::operators::RoceReceiver`, which needs ibverbs hardware
  to exercise — same as today. A pure-software mock subclass for unit tests is a
  follow-up if it becomes useful.
- Python bindings for `RoceReceiverInterfaceV1` itself. The receiver is an internal
  detail of `RoceReceiverOp` — applications drive it indirectly by constructing the
  operator with the existing parameter surface (channel, frame_metadata,
  enumeration_metadata, ibv_name, ibv_port, peer_ip, frame_size, page_size, pages,
  queue_size, …). The op forwards those values into `receiver->start(...)` internally,
  so the pybind operator class's argument list and the example call sites stay
  unchanged. A direct Python binding for `RoceReceiverV1` can be added if a future use
  case wants one.

**Done when.**

- `RoceReceiverInterfaceV1` and `RoceReceiverFrameInfoV1` are declared in
  `host/include/hololink/adapter/roce_receiver.hpp` and consumed only through that
  header. `host/operators/roce_receiver_op.cpp` no longer includes
  `hololink/operators/roce_receiver.hpp`; the legacy receiver type only enters the build
  through `module/core/`'s default impl.
- `RoceDataChannelInterfaceV1::configure` is the 2-arg form
  `configure(metadata, shared_ptr<RoceReceiverInterfaceV1>)` everywhere it's declared,
  implemented, or called (header, `module/core/` default impl, `module/hsb_lite_2510/`
  override, operator call site). No call site in the tree still passes `qp_number` /
  `rkey` / `frame_memory` as separate arguments.
- `RoceReceiverOp::start()` looks up the receiver via
  `RoceReceiverV1::get_service(module, instance_id)` where the `instance_id` matches the
  channel's (`"serial=<serial>;data_plane=<data_plane>"`), then calls
  `receiver->start(...)` with the operator's ten individual parameters.
- `module/core/`'s default supplement publishes a `RoceReceiverImplV1` reachable via
  `RoceReceiverV1::get_service(module, "serial=…;data_plane=…")`; the hsb_lite (compat
  `0x2603`) end-to-end path through `module_imx274_player.cpp` still runs against real
  hardware and decodes frames through the V1 surface.
- `module/hsb_lite_2510/`'s supplement publishes a `HsbLite2510RoceReceiverImplV1` under
  the same locator key, whose 2510-specific behavior is observable (the behavior itself
  is per the "Out of scope" note above — a follow-up).
- The existing Phase 6 operator unit test (constructs the op against a
  `RoceDataChannelInterfaceV1` mock) is updated to also satisfy a `RoceReceiverV1` mock
  reachable via `get_service`, without adding test-only hooks to the framework.

**Infrastructure.** Same as Phase 6 (Holoscan SDK to build the operator; ibverbs + real
/ emulated HSB hardware to run the end-to-end test).

### Phase 6 extension — `LinuxReceiverOp` adapter operator + `LinuxReceiverV1` / `LinuxDataChannelInterfaceV1`

**Motivation.** The adapter ships a hardware RoCE receiver operator (Phase 6 + its
extension), but the legacy tree also has `hololink::operators::LinuxReceiverOp` — a
software receiver that reassembles HSB's RoCEv2 UDP stream in user space and needs no
infiniband device. Porting it to the adapter V1 surface gives the operators tree a
receiver that runs on hosts where `HOLOLINK_BUILD_ROCE=OFF`, and exercises the
framework's per-transport "view" composition with a second transport.

**Deliverable.**

- `host/include/hololink/adapter/linux_receiver.hpp` — `LinuxReceiverInterfaceV1`
  (`type_id = "linux_receiver.v1"`) and `LinuxReceiverFrameInfoV1`, per "Linux receiver"
  under C++ Interfaces. `start(...)` takes the data-socket fd + the CUDA-buffer /
  frame-layout parameters individually (no `ibv_name` / `ibv_port` / `peer_ip`);
  `get_next_frame` carries a `CUstream`; the getter surface adds `local_port()` and
  drops `external_frame_memory()`.
- `host/include/hololink/adapter/linux_data_channel.hpp` — `LinuxDataChannelInterfaceV1`
  (`type_id = "linux_data_channel.v1"`), the software-transport sibling of
  `RoceDataChannelInterfaceV1` over the same `DataChannelInterfaceV1` anchor, adding
  `configure_socket(fd)` and an `attach_receiver(shared_ptr<LinuxReceiverInterfaceV1>)`
  that programs `configure_roce(0, …, local_port)`.
- `module/core/linux_receiver_default.hpp` (`LinuxReceiverImplV1`) and
  `module/core/linux_data_channel_default.hpp` (`LinuxDataChannelImplV1`), wrapping
  `hololink::operators::LinuxReceiver` and the legacy `hololink::DataChannel`. Both
  published per `(serial, data_channel)` by `HsbLiteEnumerationImplV1::update_metadata`,
  alongside the existing RoCE pair and the shared anchor. The publisher /
  `construct_service` chain grows a `construct_linux_receiver` +
  `construct_linux_data_channel` entry; unlike `construct_roce_receiver`, these are
  **not** build-gated (no ibverbs dependency) and are always defined and always publish.
- `host/operators/linux_receiver_op.{hpp,cpp}` +
  `host/operators/include/hololink/adapter/operators/linux_receiver_op.hpp` — a
  `holoscan::Operator` mirroring the adapter `RoceReceiverOp`. `start()` resolves
  `LinuxDataChannelInterfaceV1` + `LinuxReceiverInterfaceV1` +
  `FrameMetadataInterfaceV1` from `enumeration_metadata`, allocates the frame buffer,
  creates the datagram socket and calls `channel_->configure_socket(fd)`, then
  `receiver_->start(...)` and `channel_->attach_receiver(receiver_)`, and spawns the
  monitor thread running `receiver_->blocking_monitor()`. `compute()` polls
  `receiver_->get_next_frame(1000ms, info, stream)` (returning on timeout — same poll
  model as `RoceReceiverOp`, no frame-ready async condition), decodes the EOF block via
  the resolved `FrameMetadataInterfaceV1::decode`, wraps the frame in a
  `nvidia::gxf::Tensor`, stamps the V1 `FrameMetadata` fields **and** the Linux-specific
  `LinuxReceiverFrameInfoV1` stats (`frame_packets_received`, `frame_bytes_received`,
  `packets_dropped`) onto the operator's metadata map, and emits over `"output"`. A
  `receiver_affinity` operator parameter (defaulting from `HOLOLINK_AFFINITY`, as in the
  legacy op) sets the monitor thread's CPU affinity.
- `_hololink_adapter_operators` pybind binding for `LinuxReceiverOp`, mirroring the
  `RoceReceiverOp` binding (same `enumeration_metadata` / `frame_context` / `frame_size`
  / `page_size` / `pages` / `queue_size` / `metadata_offset` / `device_start` /
  `device_stop` argument surface, plus `receiver_affinity`).
- Example players, one per sensor family — each the software-receiver parallel of an
  existing RoCE adapter player, structurally identical except the receiver operator is
  `LinuxReceiverOp` instead of `RoceReceiverOp` (pipeline shape
  `LinuxReceiverOp → CsiToBayerOp → ImageProcessorOp → BayerDemosaicOp → HolovizOp` is
  unchanged):
  - `examples/module_linux_imx274_player.{py,cpp}` — IMX274 (`Imx274Cam`), parallel to
    `examples/module_imx274_player`.
  - `examples/module_linux_vb1940_player.{py,cpp}` — single-camera Leopard VB1940
    (`Vb1940Cam` + the `LeopardVb1940InterfaceV1` / `OscillatorInterfaceV1` bring-up),
    parallel to `examples/module_vb1940_player.cpp`. Reuses the Phase-5-extension VB1940
    driver unchanged — only the receiver operator differs.
  - `examples/CMakeLists.txt` registers `module_linux_imx274_player` and
    `module_linux_vb1940_player` the same way their RoCE counterparts are, but **not**
    under the `HOLOLINK_BUILD_ROCE` gate — they build whenever the adapter examples /
    operators do.
- `hololink_adapter/README.md` — a new as-built "Phase 6 extension — `LinuxReceiverOp`
  …" section appended **as this code lands**, in the same form as the existing
  `RoceReceiverOp` README sections (`Phase 6 extension — RoceReceiverV1 abstraction …`,
  `Phase 6 — RoCE receiver operator …`). The README is the as-built record and lags the
  plan; it is updated in the same commit(s) that apply the code, never ahead of it.
  Subsequent commits that touch the Linux receiver surface (interface edits, the 2510
  override, the frame-ready follow-up) likewise update their README section as they
  land.

**Capability gating.** This refines the Environment-capability-gating rules: the adapter
`hololink::operators` library and the `_hololink_adapter_operators` extension already
build whenever `HOLOLINK_ADAPTER_BUILD_OPERATORS=ON` (Holoscan present). The RoCE
*operator* TU stays `HOLOLINK_BUILD_ROCE`-gated; the **Linux operator TU and its
`module/core/` default impl build unconditionally within the operators tree**, because
the legacy `LinuxReceiver` links only `hololink::core` + CUDA (its lone ibverbs touch is
a compile-time opcode `static_assert` guarded by `HOLOLINK_HAVE_IBV_OPCODE`, with a
hardcoded fallback when RoCE is off). So in a `HOLOLINK_BUILD_ROCE=OFF` build the
operators package imports and exposes `LinuxReceiverOp` while `RoceReceiverOp` is absent
— `module_linux_imx274_player` is runnable where `module_imx274_player` is not.

**Testing.** Both sensor families get a C++ smoke test and a Python integration test,
each parallel to its existing RoCE counterpart but **without** the
`@pytest.mark.accelerated_networking` marker — the software receiver needs no infiniband
device, so these run in environments where the RoCE tests skip.

- **C++ player smoke tests.** `tests/CMakeLists.txt` registers
  `module_linux_imx274_player_test` and `module_linux_vb1940_player_test` alongside the
  existing `module_*_player_test` targets — each runs its example with
  `--frame-limit=10 --headless` and is CMake-gated on the per-sensor hardware env
  (`HOLOLINK_TEST_IMX274=1` / `HOLOLINK_TEST_VB1940=1`, `SKIP_RETURN_CODE 77` so ctest
  reports SKIPPED otherwise), carrying the matching `imx274` / `vb1940` ctest label.
  Unlike `module_vb1940_player_test` / `module_quad_imx274_player`, they are **not**
  gated on `HOLOLINK_BUILD_ROCE`.
- **Python integration tests.** `tests/test_module_linux_imx274_player.py` and
  `tests/test_module_linux_vb1940_player.py` drive the corresponding example's `main()`
  (mocking the receiver operator for frame-count assertions, the same pattern as the
  legacy `tests/test_linux_imx274_player.py` and the adapter
  `tests/test_module_vb1940.py`). The IMX274 test carries
  `@pytest.mark.skip_unless_imx274`; the VB1940 test carries
  `@pytest.mark.skip_unless_vb1940`. Neither carries
  `@pytest.mark.accelerated_networking` — that marker (and `HOLOLINK_BUILD_ROCE`) stays
  exclusive to the hardware RoCE path. These markers gate on the `--imx274` / `--vb1940`
  switches via `tests/conftest.py`.

**Out of scope.**

- Behavioral content of any 2510 software-receiver override. The metadata-layout diff
  already routes through `FrameMetadataInterfaceV1::decode` and the
  `HsbLite2510DataChannel` `configure_roce` override; a `HsbLite2510LinuxReceiverImplV1`
  is added under the same locator key only if 2510 verification surfaces a
  software-receiver-specific need. This phase wires the seam (both modules publish under
  the same `(instance_id, type_id)`); it does not author a behavior diff.
- A frame-ready / `AsynchronousCondition` low-latency path. The operator polls
  `get_next_frame` with a timeout, matching the adapter `RoceReceiverOp`. Reinstating
  the legacy Linux op's `set_frame_ready` condition is a follow-up if profiling shows
  the poll model costs too much.
- Python bindings for `LinuxReceiverInterfaceV1` / `LinuxDataChannelInterfaceV1`
  themselves — internal details of `LinuxReceiverOp`, reached only through the
  operator's parameter surface, same as the RoCE receiver.

**Done when.**

- `LinuxReceiverInterfaceV1`, `LinuxReceiverFrameInfoV1`, and
  `LinuxDataChannelInterfaceV1` are declared in their headers and consumed only through
  them; `host/operators/linux_receiver_op.cpp` includes no legacy
  `hololink/operators/linux_receiver*` header — the legacy receiver type enters the
  build only through `module/core/`.
- `module/core/`'s default supplement publishes a `LinuxReceiverImplV1` +
  `LinuxDataChannelImplV1` reachable via `…V1::get_service(metadata)` keyed by
  `"serial=…;data_channel=…"`; the hsb_lite (compat `0x2603`) end-to-end path runs
  `module_linux_imx274_player` against real IMX274 hardware, and the
  `module/leopard_vb1940/` path runs `module_linux_vb1940_player` against real VB1940
  hardware — both decode frames through the V1 surface.
- `LinuxReceiverOp` builds and its pybind class instantiates from Python in a
  `HOLOLINK_BUILD_ROCE=OFF` configuration (where `RoceReceiverOp` is not built).
- A unit test constructs the operator against `LinuxDataChannelInterfaceV1` +
  `LinuxReceiverInterfaceV1` mocks reachable via `get_service`, without adding test-only
  hooks to the framework (mirroring `hololink_adapter_roce_receiver_op_test`).
- The IMX274 and VB1940 examples each have a CMake-gated C++ smoke test
  (`module_linux_imx274_player_test`, `module_linux_vb1940_player_test`) and a Python
  integration test (`tests/test_module_linux_imx274_player.py` carrying
  `@pytest.mark.skip_unless_imx274`, `tests/test_module_linux_vb1940_player.py` carrying
  `@pytest.mark.skip_unless_vb1940`); none is gated on `HOLOLINK_BUILD_ROCE` or carries
  `@pytest.mark.accelerated_networking`.
- `hololink_adapter/README.md` carries an as-built section for the `LinuxReceiverOp`
  work, added in the same commit(s) that landed the code (not ahead of it).

**Infrastructure.** Holoscan SDK to build the operator; real / emulated HSB hardware to
run the end-to-end test. No ibverbs / infiniband requirement — that is the point of this
phase.

### Phase 6 extension — native adapter `CsiToBayerOp` (retires the `LegacyCsiConverter` shim)

**Motivation.** The adapter sensor drivers (`Imx274Cam` / `Vb1940Cam`) interpret
received CSI data through a `CsiConverterV1`. Until now the only implementation was the
legacy `hololink::operators::CsiToBayerOp`, so every adapter example bridged it through
`examples/legacy_csi_converter.{hpp,py}` (`LegacyCsiConverter`) — a shim that wraps the
legacy operator and casts the adapter `PixelFormat` to the legacy one. A native
converter operator lets the examples drop the shim entirely.

**Deliverable.**

- `host/operators/csi_to_bayer_op.{cpp}` +
  `host/operators/include/hololink/adapter/operators/csi_to_bayer_op.hpp` —
  `hololink::adapter::operators::CsiToBayerOp`, both a `holoscan::Operator` and a
  `hololink::adapter::csi::CsiConverterV1` (the same dual-role pattern
  `FusaCoeCaptureOp` uses). It plugs into the pipeline and is handed straight to a
  sensor's `configure_converter()`. It resolves no adapter service (no
  `enumeration_metadata`); the constructor mirrors the legacy operator (`allocator`,
  `cuda_device_ordinal`, `out_tensor_name`, `sub_frame_rows`) and adds
  `get_csi_length()` / `get_sub_frame_size()`.
- **Self-contained engine.** The CSI→Bayer engine — the NVRTC
  `frameReconstruction8/10/12` kernels, the sub-frame accumulation in `compute()`, and
  the four geometry helpers — is ported from the legacy operator, but expressed against
  `hololink::adapter::csi::PixelFormat`, so no legacy CSI type appears in the operator's
  API (the legacy `CsiToBayerOp` is left untouched; the two copies are independent). Its
  CUDA helpers are adapter-owned too: `host/operators/cuda_function_launcher.{hpp,cpp}`
  vendors `CudaFunctionLauncher` / `CudaContextScopedPush` (behavior-identical ports of
  the legacy `hololink::common` helpers), error checking goes through
  `HOLOLINK_ADAPTER_CUDA_CHECK` and `UniqueCUdeviceptr` comes from `cuda_unique.hpp`,
  and it uses the adapter `round_up` from `page_size.hpp` — so the operator carries no
  source or link dependency on legacy `hololink::core` / `hololink::common`.
- `_hololink_adapter_operators` pybind binding for `CsiToBayerOp`, **ungated** (no
  ibverbs / FUSA dependency) — listing both `holoscan::Operator` and `CsiConverterV1` as
  bases so a Python-constructed op converts to `shared_ptr<CsiConverterV1>` for
  `configure_converter`. The pixel-format arguments are bound as plain integers (the
  enumerator value) so the pure-Python `Imx274Cam` can pass its `csi.PixelFormat.value`
  without importing the bound C++ enum. `host/operators/CMakeLists.txt` adds the source
  (and the vendored `cuda_function_launcher.cpp`) to the always-built
  `hololink_operators` library, which links `CUDA::nvrtc` + `CUDA::cuda_driver`; legacy
  `hololink::core` is linked only for FUSA now (the always-built operators no longer
  need it).
- `python/sensors/imx274/imx274_cam.py::configure_converter` drops the legacy
  `hololink.PixelFormat` translation and passes the adapter pixel-format value directly.
  The C++ sensors already pass the adapter `csi::PixelFormat`, so they are unchanged.
- The example players migrate from `LegacyCsiConverter(legacy CsiToBayerOp)` to the
  native operator (construct it, hand it to `configure_converter`, size the frame via
  `get_csi_length()`): Python `module_linux_imx274_player.py`,
  `module_linux_vb1940_player.py`, `module_quad_imx274_player.py`; C++
  `module_imx274_player.cpp`, `module_linux_imx274_player.cpp`,
  `module_quad_imx274_player.cpp`, `module_vb1940_player.cpp`,
  `module_linux_vb1940_player.cpp`, `module_stereo_vb1940_player.cpp`. The
  `examples/legacy_csi_converter.{hpp,py}` shim and its `examples/CMakeLists.txt` entry
  are removed.
- No new tests: the existing `tests/test_module_linux_imx274_player.py` /
  `tests/test_module_linux_vb1940_player.py` (gated `@pytest.mark.skip_unless_imx274` /
  `skip_unless_vb1940`) drive the example `main()` directly and pass no
  converter-specific arguments, so migrating the examples carries them onto the native
  operator.
- `hololink_adapter/README.md` carries an as-built section for this work, added in the
  same commit(s) that landed the code.

**Infrastructure.** Holoscan SDK + CUDA to build and run the operator; real / emulated
HSB hardware (IMX274 / VB1940 cameras) for the end-to-end example runs.

### Phase 6 extension — native adapter `ImageProcessorOp`

**Motivation.** After the CsiToBayerOp migration, the adapter example players still
reached into the legacy tree for the next pipeline stage —
`hololink::operators::ImageProcessorOp` (optical-black correction + Grey-World auto
white-balance). It was the only remaining legacy operator those players used; in the
three Python players it was in fact the only reason `import hololink` survived. A native
sibling completes the de-legacying of the adapter pipeline.

**Deliverable.**

- `host/operators/image_processor_op.{cpp}` +
  `host/operators/include/hololink/adapter/operators/image_processor_op.hpp` —
  `hololink::adapter::operators::ImageProcessorOp`, a **plain** `holoscan::Operator`
  (unlike `CsiToBayerOp` it implements no adapter interface — it is not a converter).
  The constructor mirrors the legacy operator exactly (`pixel_format`, `bayer_format`,
  `optical_black`, `cuda_device_ordinal`).
- **Self-contained engine.** The engine — the NVRTC `applyBlackLevel` / `histogram` /
  `calcWBGains` / `applyOperations` kernels, the histogram/shared-memory sizing in
  `start()`, and the sub-frame-aware white-balance accumulation in `compute()` — is
  ported verbatim from the legacy operator. The `pixel_format` / `bayer_format` params
  stay `int` but are interpreted against `hololink::adapter::csi::PixelFormat` /
  `BayerFormat` (identical enumerator values), so no legacy CSI type appears in the
  operator. It reuses the adapter-owned CUDA helpers (`CudaFunctionLauncher` /
  `CudaContextScopedPush` from `cuda_function_launcher.hpp`, `UniqueCUdeviceptr` /
  `HOLOLINK_ADAPTER_CUDA_CHECK` from `cuda_unique.hpp`) and the adapter `HSB_LOG_*`
  logging, so it links no legacy `hololink::core`.
- `_hololink_adapter_operators` pybind binding for `ImageProcessorOp`, **ungated** (no
  ibverbs / FUSA dependency). Because the public API is plain `int` params (no enum),
  the binding needs no int-coercion lambda (unlike `CsiToBayerOp.configure`).
  `host/operators/CMakeLists.txt` adds the source to the always-built
  `hololink_operators` library; no new link deps or capability define are required.
- The example players migrate from the legacy `ImageProcessorOp` to the native operator
  (constructor arguments unchanged): Python `module_linux_imx274_player.py`,
  `module_linux_vb1940_player.py`, `module_quad_imx274_player.py` (each also drops its
  now-unused `import hololink`, leaving the Python adapter players with no legacy
  import); C++ `module_imx274_player.cpp`, `module_linux_imx274_player.cpp`,
  `module_quad_imx274_player.cpp`, `module_vb1940_player.cpp`,
  `module_linux_vb1940_player.cpp`, `module_stereo_vb1940_player.cpp`, and
  `module_fusa_coe_vb1940_player.cpp` (the FUSA player runs `ImageProcessorOp` after
  `FusaCoeCaptureOp`, so it was outside the CsiToBayerOp migration but in scope here).
- No new tests: the existing `tests/test_module_linux_imx274_player.py` /
  `tests/test_module_linux_vb1940_player.py` drive the example `main()` directly and
  pass no operator-specific arguments, so migrating the examples carries them onto the
  native operator.
- `hololink_adapter/README.md` carries an as-built section for this work, added in the
  same commit(s) that landed the code.

**Infrastructure.** Holoscan SDK + CUDA to build and run the operator; real / emulated
HSB hardware (IMX274 / VB1940 cameras) for the end-to-end example runs.

### Phase 6 extension — native adapter `PackedFormatConverterOp` + adapter utilities

**Motivation.** After the `CsiToBayerOp` / `ImageProcessorOp` migrations, the adapter
example players and tests had a few remaining legacy `hololink` references: the FUSA
example's `hololink::operators::PackedFormatConverterOp` (its packed-CSI→16-bit
converter), the legacy `CudaCheck` macro and `env_hololink_ip` / `MacAddress` host
utilities, and legacy operators + `hsb_log_*` in `tests/test_module_vb1940.py`. This
extension closes those, so the C++ adapter players carry no legacy `hololink` includes.

**Deliverable.**

- `host/operators/packed_format_converter_op.{cpp}` +
  `host/operators/include/hololink/adapter/operators/packed_format_converter_op.hpp` —
  `hololink::adapter::operators::PackedFormatConverterOp`, both a `holoscan::Operator`
  and a `hololink::adapter::csi::CsiConverterV1` (dual-role, like `CsiToBayerOp`). The
  unpack engine (NVRTC `packed8bitTo16bit` / `packed10bitTo16bit` /
  `packed12bitTo16bit`) is ported from the legacy operator, expressed against
  `hololink::adapter::csi::PixelFormat`. It uses the adapter-owned CUDA helpers
  (`cuda_function_launcher.hpp`, `cuda_unique.hpp`) and links no legacy
  `hololink::core`. Constructor
  `allocator, cuda_device_ordinal, in_tensor_name, out_tensor_name`; public
  `get_frame_size()` + the 4 `CsiConverterV1` methods. Ungated (always built); ungated
  pybind binding registered next to `CsiToBayerOp`.
- `FusaCoeCaptureOp::configure_converter` flips from `hololink::csi::CsiConverter&` to
  `hololink::adapter::csi::CsiConverterV1&` (header + cpp), so the native converter is
  handed to it directly — removing a legacy CSI type from the adapter FUSA op's surface.
- New public adapter utility headers: `hololink/adapter/tools.hpp` (header-only inline
  `env_hololink_ip`) and `hololink/adapter/networking.hpp` (`MacAddress` alias). The C++
  example players replace `<hololink/common/tools.hpp>` /
  `<hololink/common/cuda_helper.hpp>` / `<hololink/core/networking.hpp>` with the
  adapter equivalents (`env_hololink_ip`, `HOLOLINK_ADAPTER_CUDA_CHECK`, `MacAddress`)
  and the legacy `PackedFormatConverterOp`; the migrated `module_*` targets in
  `examples/CMakeLists.txt` drop their legacy operator links as well as the now-stale
  legacy `hololink` (core) and `hololink::sensors::native_*_camera_sensor` links, so
  each links only adapter-defined libraries plus the Holoscan SDK (the legacy example
  players keep theirs).
- Adapter Python convenience bindings: module-level `hololink_adapter.hsb_log_*` (route
  through the registered HSB logger) and `hololink_adapter.infiniband_devices()`
  (RoCE-gated). `tests/test_module_vb1940.py` moves onto the native operators + adapter
  logging (dropping `import hololink`); `tests/conftest.py` additionally logs through
  the adapter logger in `report_test_name` (keeping its legacy reset/infiniband paths —
  the adapter has no global framework-reset, so those stay legacy).
- `hololink_adapter/README.md` carries an as-built section for this work.

**Infrastructure.** Holoscan SDK + CUDA to build and run the operator; ibverbs for the
RoCE-gated `infiniband_devices` binding; real / emulated HSB hardware for the example
runs.

### Phase 7 — IMX274 sensor driver

**Deliverable.** `python/sensors/imx274/` shipping `Imx274Cam`, `Imx274_Mode`, and
`LII2CExpander` written against adapter handles (`HololinkInterfaceV1`,
`I2cInterfaceV1`, `I2cLockV1`). Mirrors the legacy `dual_imx274.py`'s register sequences
and clock setup but without coupling to the legacy `DataChannel` / `Hololink` types.

**Done when.** Unit tests construct an `Imx274Cam` against a real HSB-Lite board with
IMX274 cameras attached, run `setup_clock`, and round-trip `set_register` /
`get_register` through the I2C expander (both expander outputs exercised).

**Infrastructure.** The driver depends on the Phase 5 / Phase 5b deferrals being in
place: the `HololinkInterface`, `I2cInterface`, and `I2cLockV1` Python bindings (so
`Imx274Cam` can take adapter handles), the concrete `HsbLiteImplV1` supplement (or a
direct `HololinkInterface` accessor for board-specific clock setup), and the four
hardware-coupled wrappers from Phase 4. IMX274 tests run against **real HSB-Lite
hardware with IMX274 cameras attached** — the HSB emulator is not used for the IMX274
test path because the emulator does not model an IMX274 camera on the I2C bus.

### Phase 8 — `module_imx274_player.py` example + integration test

**Deliverable.** `examples/module_imx274_player.py` wiring 2 HSB-Lite boards × 2 IMX274
cameras each = 4 RoCE data channels into a Holoscan pipeline (CsiToBayer →
ImageProcessor → BayerDemosaic → HolovizOp). The example's default channel-IP arguments
are `192.168.0.200`, `192.168.0.201`, `192.168.0.202`, `192.168.0.203` — two consecutive
IPs per board (board 1 at `.200/.201`, board 2 at `.202/.203`), each pair representing
the two RoCE data channels on a single HSB-Lite. The example configures every camera to
`Imx274_Mode.IMX274_MODE_1920X1080_60FPS` (mode 1, 1080p RAW10) so the four-channel
aggregate fits inside the data-plane network's bandwidth budget; the 4K modes (modes 0
and 2) over four channels exceed what the link can sustain. The new file is appended to
`EXAMPLE_INSTALL_FILES` in `examples/CMakeLists.txt`. `tests/CMakeLists.txt` registers a
`--frame-limit=10 --headless` smoke test that runs against real HSB-Lite hardware with
IMX274 cameras attached. `hololink_adapter/README.md` carries a section per phase
summarizing what was implemented and listing the build / test-run commands.

**Done when.** The smoke test passes against real HSB-Lite hardware (2 boards × 2 IMX274
cameras = 4 channels) and the README covers all phases per the per-phase update rule.

**As-built status.** `examples/module_imx274_player.py` ships the 4-channel pipeline
described above and is appended to `EXAMPLE_INSTALL_FILES`. Discovery is bootp-driven:
`adapter.start_bootp_listener()` opens the UDP socket on the host `ReactorV1`; the
example then polls `adapter.find_channel(peer_ip)` for each of the four expected peer
IPs until all four boards have announced themselves (default 30s timeout via
`--discovery-timeout`). The HSB-Lite supplement's publication chain (described under
Phase 5's as-built status) covers every V1 lookup the example makes —
`HololinkInterface`, `HsbLiteInterface`, `RoceDataChannelInterface`,
`SequencerInterface`, and `I2cInterface` (for the IMX274 + LI expander addresses). The
example reuses `hololink_module.renesas_bajoran_lite_ts1.device_configuration()` for the
clock profile so no profile data is duplicated. The ctest registration
(`module_imx274_player_test`) is wired in `tests/CMakeLists.txt` and runs the example
with `--frame-limit=10 --headless` against the boards on the test network. The same
build serves hardware and non-hardware runners — the test always registers when the
build includes Python + operators, and skips at runtime unless the runtime env var
`HOLOLINK_TEST_IMX274=1` is set (`SKIP_RETURN_CODE 77` makes ctest report SKIPPED in the
unset case). The test also carries `LABELS "imx274"` so `ctest -L imx274` runs the
hardware-dependent slice.

**Infrastructure.** This phase exercises the full stack and so needs every external
dependency simultaneously: **real HSB-Lite hardware with IMX274 cameras attached** (2
boards × 2 cameras for the 4-channel topology) reachable on the test network, the
Holoscan SDK (for the receiver operator + the rest of the Holoscan pipeline), the CUDA
toolkit (transitively via `hololink::core` and Holoscan), and the Phase 5 / Phase 5b /
Phase 6 / Phase 7 deferrals all delivered. The HSB emulator is not used for this phase
because it does not model IMX274 cameras on the I2C bus. The smoke test runs with
`--frame-limit=10 --headless` so a display is not required, but a real GPU and
RoCE-capable network stack are.

### Phase 5 extension — `OscillatorInterfaceV1` + clock-rate cache on `HololinkImpl`

**Deliverable.** New per-data-plane V1 service `OscillatorInterfaceV1` published by both
HSB-Lite modules (`hsb_lite` for compat-id 2603 and `hsb_lite_2510` for the 2510 FPGA
revision). The service lets the sensor driver — not the application — drive the on-board
reference clock during sensor configuration.

- `host/include/hololink/adapter/oscillator.hpp` — new abstract `OscillatorInterfaceV1`
  with `enable(uint64_t clocks_per_second)` returning `bool` (true: rate programmed or
  already programmed; false: configuration not achievable on this hardware), plus
  `get_caps()` / `set_caps()` for implementation-defined tunable knobs. Service is a
  `ServiceLocatable<OscillatorInterfaceV1>` with `type_id = "oscillator.v1"`;
  instance_id matches the per-data-plane `"serial=<serial_number>;data_plane=<n>"` form
  so the oscillator is reachable alongside the channel and receiver.
- `module/hsb_lite/oscillator_impl.{hpp,cpp}` — `OscillatorImplV1`, holds a
  `shared_ptr<module_core::HololinkImplV1>` and the data-plane index. `enable(...)`
  delegates to `HololinkImplV1::enable_clock(...)` so two data planes on the same board
  coordinate through the per-board cache.
- `module/hsb_lite_2510/oscillator_impl.{hpp,cpp}` — same shape,
  `HsbLite2510OscillatorImplV1` following the 2510-override naming convention.
- `module/core/hololink_default.{hpp,cpp}` — new module-private method
  `HololinkImplV1::enable_clock(uint64_t)`. First call programs the on-board clock
  generator via `setup_clock(hololink::renesas::DEVICE_CONFIGURATION)`, caches the rate,
  and returns true. Subsequent calls with the same rate are no-ops that return true;
  subsequent calls with a different rate return false (the on-board generator can serve
  only one rate at a time). The constructor registers a private nested
  `Hololink::ResetController` via `backing_->on_reset(...)` so the cached rate is
  dropped whenever the underlying hololink is reset; the next `enable_clock(rate)`
  re-programs the generator. The cache is shared across the two `OscillatorImplV1`
  instances published per board because both delegate through the same `HololinkImplV1`.
- Both modules' `HsbLiteEnumerationImplV1` publish one oscillator instance per data
  plane under the `data_plane_instance_id`, constructed with the board's
  `shared_ptr<HololinkImplV1>` (not the bare `LegacyHololinkAccess` — the oscillator
  needs the wrapper for `enable_clock` access).
- `host/sensors/imx274/imx274_cam.{hpp,cpp}` and `python/sensors/imx274/imx274_cam.py` —
  `Imx274Cam` constructor takes a required `OscillatorInterfaceV1` (after the
  HololinkInterface). `configure(mode)` calls `oscillator->enable(25'000'000)` before
  touching the sensor's registers; if `enable` returns false, throws / raises
  "oscillator does not support the IMX274's 25 MHz reference clock".
- `host/python/hololink_adapter_py.cpp` + `python/__init__.py` — `OscillatorInterfaceV1`
  bound for Python with `get_service` / `enable` / `get_caps` / `set_caps`; exported
  under that explicit V1 name.
- `examples/module_imx274_player.cpp` + `examples/module_quad_imx274_player.{cpp,py}` —
  fetch the per-data-plane `OscillatorInterface` from the locator and pass it to
  `Imx274Cam`. The explicit `hsb_lite->setup_clock(...)` lifecycle call is removed from
  the bring-up loop in every example; the clock is programmed from inside
  `Imx274Cam::configure` via the oscillator, and the per-board cache + reset listener on
  `HololinkImplV1` ensure two cameras sharing a board don't double-program the generator
  (the second `enable(25 MHz)` is a cache hit returning true; a board `reset()` between
  cameras drops the cache and the next `enable` re-programs). The `HsbLiteInterface`
  lookup is dropped from each example since clock setup is the only thing they used it
  for; the `Channel` struct in the quad C++ + Python examples also drops its `hsb_lite`
  field.

**Out of scope this phase.** Behavioral content of the oscillator's `get_caps` /
`set_caps` — both modules return `{}` / `false` today. A non-25-MHz rate isn't a
supported request: `enable_clock(rate)` accepts whatever rate the first caller asks for
and programs the chip via the `DEVICE_CONFIGURATION` table, so calling
`enable(other_rate)` first commits the chip to "other_rate" with the wrong register
sequence. Validating the requested rate against the hardware's actual capability is left
for follow-up; the IMX274 driver only ever asks for 25 MHz, and the cache rejects any
later mismatched request.

**Done when.** Both modules build, the example pytest + ctest smoke tests pass with the
oscillator wired in, and `camera->configure(mode)` succeeds against either the emulator
(for the structural tests that don't exercise the clock chip) or real hardware (where
`setup_clock(DEVICE_CONFIGURATION)` actually programs the Renesas Bajoran Lite TS1).

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD" -DHOLOLINK_ADAPTER_BUILD_OPERATORS=ON
cmake --build "$BUILD" --target hsb_lite
cmake --build "$BUILD" --target hsb_lite_2510
cmake --build "$BUILD" --target hololink_adapter_sensors_imx274
cmake --build "$BUILD" --target _hololink_adapter
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" -R hololink_adapter_module_core_test --output-on-failure
ctest --test-dir "$BUILD" -R hololink_adapter_imx274_python_test --output-on-failure
```

### Phase 5 extension — metadata-only service lookups + `Imx274Cam(metadata)` + TOCTOU fix on module loader

**Deliverable.** Ergonomic + safety follow-ups now that the V1 surface is in place:

- `host/include/hololink/adapter/hololink.hpp` + `host/src/hololink.cpp` — new static
  `HololinkInterfaceV1::get_service(metadata, allow_null = false)`. Resolves the
  supplement module by calling `Adapter::get_adapter().get_module(metadata)` itself,
  derives the per-board `"serial=<serial_number>"` instance_id from
  `metadata.get<std::string>("serial_number")`, and delegates to the base
  `ServiceLocatable<HololinkInterfaceV1>::get_service`. Throws if `serial_number` is
  missing; `allow_null` continues to govern whether an unpublished service produces an
  empty pointer or a throw. The base `(module, instance_id, allow_null)` form is kept
  reachable through a `using ServiceLocatable<…>::get_service;` declaration. Body moved
  out-of-line so the header doesn't have to drag `adapter.hpp` into every consumer.
- `host/include/hololink/adapter/oscillator.hpp` + `host/src/oscillator.cpp` — same
  pattern for `OscillatorInterfaceV1::get_service(metadata, allow_null = false)`. Builds
  the per-data-plane `"serial=<serial_number>;data_plane=<data_plane>"` instance_id.
- `host/sensors/imx274/imx274_cam.{hpp,cpp}` + `python/sensors/imx274/imx274_cam.py` —
  new `Imx274Cam(const EnumerationMetadata& metadata, uint32_t i2c_bus = …)` constructor
  (Python: `Imx274Cam(metadata, i2c_bus=…)` via first-arg type dispatch in `__init__`).
  Fetches HololinkInterface + OscillatorInterface from the metadata using the overloads
  above. `expander_configuration` is derived from metadata: prefer
  `metadata["expander_configuration"]` when present, else fall back to
  `metadata["data_plane"]` (the HSB-Lite carrier conflates the two — the data_plane
  index doubles as the I2C-expander output selector). The original
  `(hololink, oscillator, i2c_bus, expander_configuration)` ctor stays for code paths
  that already have the services in hand.
- `host/sensors/imx274/imx274_cam.{hpp,cpp}` + `python/sensors/imx274/imx274_cam.py` —
  new static `Imx274Cam::use_expander_configuration(metadata, value)` /
  `Imx274Cam.use_expander_configuration(metadata, expander_configuration)`. Stamps an
  override into `metadata["expander_configuration"]` so application code doesn't mutate
  the EnumerationMetadata fields by string key. Keeps the metadata layout an `Imx274Cam`
  concern.
- `host/python/hololink_adapter_py.cpp` + `python/__init__.py` — pybind bindings for the
  metadata-taking `get_service` overloads (no `module` parameter). The instance_id +
  metadata forms coexist on each class via pybind11 overload resolution;
  `OscillatorInterfaceV1` was already exported from `__init__.py`.
- `host/include/hololink/adapter/adapter.hpp` + `host/src/adapter.cpp` —
  `Adapter::load_module_for` had a check-then-use TOCTOU:
  `std::filesystem::exists(path)` before `load_module(path)` meant a file deleted in the
  gap surfaced as a confusing dlopen error instead of falling through to the next
  candidate. New private helper `Adapter::try_load_module(path)` collapses the check and
  the load into a single
  `try { load_so(...); } catch (const std::runtime_error&) { stat-after-failure; }`
  step: file missing → return empty `shared_ptr` (caller falls back); file present → re-
  throw so real load failures (dlopen / ABI / init) still surface. `load_module_for`
  walks the compat-suffixed candidate then the bare candidate via `try_load_module`,
  with a final "no module .so for fpga_uuid" throw when both come back empty.
- `examples/module_imx274_player.cpp` + `examples/module_quad_imx274_player.{cpp,py}` —
  `module_handle` / `module` locals dropped now that the metadata-taking lookups resolve
  the supplement module themselves. Single-board variant calls
  `Imx274Cam::use_expander_configuration(metadata, n)` (only when the
  `--expander- configuration` CLI flag was actually passed) instead of mutating
  `metadata["expander_configuration"]` directly. Quad variants construct cameras as
  `Imx274Cam(md)` / `hololink_adapter.sensors.imx274.Imx274Cam(md)` and rely on the
  metadata fallback to pick up `data_plane`. `i2c_bus` argument is dropped from the ctor
  calls when it equals `DEFAULT_CAM_I2C_BUS`.

**Out of scope this phase.** The residual race window in `try_load_module` (file deleted
between the `load_so` throw and the `std::filesystem::exists` check inside the `catch`)
misclassifies a real load error as ENOENT and falls back. The window is microscopic
compared to the prior pattern and the wrong action is "fall back to a different .so"
rather than "crash" — acceptable for now. The `Imx274Cam::use_expander_ configuration`
helper is the only application-facing way to seed `metadata["expander_configuration"]`;
other metadata fields (sensor / vp_address / hif_address / …) remain a supplement-only
concern and have no equivalent helpers yet.

**Done when.** Examples build and run with the new shapes against the existing test
hardware path (no behavioral change vs the pre-refactor path; the metadata-taking
overloads are pure ergonomic shorthand), and `try_load_module`'s fallback semantics
match the previous code for the happy path (compat-suffixed .so picked when present,
bare .so otherwise, throw when neither exists).

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD" -DHOLOLINK_ADAPTER_BUILD_OPERATORS=ON
cmake --build "$BUILD" --target hololink_adapter
cmake --build "$BUILD" --target _hololink_adapter
cmake --build "$BUILD" --target module_imx274_player
cmake --build "$BUILD" --target module_quad_imx274_player
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" -R hololink_adapter_framework_test --output-on-failure
ctest --test-dir "$BUILD" -R hololink_adapter_enumeration_test --output-on-failure
ctest --test-dir "$BUILD" -R hololink_adapter_imx274_python_test --output-on-failure
```

### Phase 3 extension — `Adapter::wait_for_channel` replaces the `enumerated_` cache + `find_channel`

**Deliverable.** Trade the unbounded `enumerated_` vector + `find_channel(peer_ip)`
lookup for a one-shot per-peer-IP slot map + condition variable + blocking
`wait_for_channel(peer_ip, timeout)` API. Removes the second source of truth that
duplicated the locator state and bounded the cache size.

- `host/include/hololink/adapter/adapter.hpp` — declares
  `EnumerationMetadata Adapter::wait_for_channel(const std::string& peer_ip, std::chrono::milliseconds timeout)`.
  Replaces the `enumerated_` vector with a
  `std::unordered_map<std::string, EnumerationMetadata> pending_by_peer_ip_` plus a
  `std::condition_variable enumeration_cv_`.
- `host/src/adapter.cpp` — `enumerate()` writes the metadata into the slot keyed by
  `peer_ip` (overwriting any prior unconsumed entry — the most recent announcement wins)
  and notifies the cv. `wait_for_channel(peer_ip, timeout)` first erases any slot the
  peer might already have (so a cached announcement minutes old never masquerades as a
  fresh one), then does a `cv.wait_for` on the slot's presence, consumes (`map::erase`)
  on return, and throws
  `std::runtime_error("…peer IP '…' did not announce itself within the timeout")` on
  timeout. `find_channel` is removed. The pre-erase means `wait_for_channel` always
  blocks until a bootp announcement that arrives *after* the call started waiting —
  tests therefore have to drive `enumerate` from a helper thread (a synchronous
  `enumerate()` followed by `wait_for_channel()` would have the pre-erase discard the
  just-posted entry).
- `host/python/hololink_adapter_py.cpp` — pybind binding for `wait_for_channel` takes
  `(peer_ip, timeout_s)` with `timeout_s` as a Python `float` (seconds), converted to
  `std::chrono::milliseconds` inside the lambda. The `find_channel` binding is dropped.
- `examples/module_imx274_player.cpp` — the local `wait_for_channel` polling helper is
  gone; `main()` calls `adapter.wait_for_channel(hololink_ip, discovery_timeout)`
  directly. The `<thread>` include is dropped (no more sleep loop).
- `examples/module_quad_imx274_player.cpp` — the local `enumerate` polling helper
  shrinks to a sequential
  `for ip : device_ips { adapter.wait_for_channel(ip, remaining); }` loop with a
  cumulative deadline so the total wait is bounded by the configured discovery timeout
  rather than `4 × timeout`. The `<thread>` include is dropped.
- `examples/module_quad_imx274_player.py` — same restructuring on the Python side; the
  local `_enumerate` helper does a sequential `adapter.wait_for_channel(ip, remaining)`
  walk with a cumulative deadline.
- `tests/hololink_adapter_enumeration_test.cpp` —
  `ManualEnumerateLoadsModuleAndEnriches` and `EnumerateStoresUnknownUuidUnenriched`
  both spawn a helper thread that sleeps 50ms then calls `enumerate(...)` while the main
  test thread is blocked in `wait_for_channel(peer_ip, 1s)`. (A synchronous
  enumerate-then-wait would have the pre-erase wipe the slot.)
  `FindChannelThrowsOnUnknownIp` becomes `WaitForChannelTimesOutOnUnknownIp` with a 50ms
  timeout.
- `tests/test_hololink_adapter_python.py` — `find_channel` call swapped for the same
  pattern: a `threading.Thread` enumerates after a 50ms sleep while the test thread
  blocks in `wait_for_channel(peer_ip, timeout_s=1.0)`.

**Out of scope this phase.** The bootp listener's callback-style `register_ip` /
`register_all` APIs sketched elsewhere in the plan have not landed; `wait_for_channel`
is the only synchronization primitive today. Multi-waiter behavior is "first consumer
wins" — adequate for the current single-threaded discovery flows in the examples but not
a general broadcast-style primitive.

**Done when.** `EnumerateStoresUnknownUuidUnenriched` +
`WaitForChannelTimesOutOnUnknownIp` pass; example smoke tests still discover the
expected boards within their configured discovery timeouts; no `find_channel` references
remain in adapter source, pybind, or example code.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD"
cmake --build "$BUILD" --target hololink_adapter
cmake --build "$BUILD" --target _hololink_adapter
cmake --build "$BUILD" --target module_imx274_player
cmake --build "$BUILD" --target module_quad_imx274_player
cmake --build "$BUILD" --target hololink_adapter_enumeration_test
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" -R hololink_adapter_enumeration_test --output-on-failure
ctest --test-dir "$BUILD" -R hololink_adapter_python_test --output-on-failure
```

### Phase 3 extension — callback-based `register_ip` / `register_all`; `wait_for_channel` built on top

**Deliverable.** The plan calls for callback-style enumeration delivery to the
application; this extension lands `register_ip` / `register_all` / `unregister` on
Adapter and rewrites `wait_for_channel` to build on `register_ip` rather than a global
slot map. The `pending_by_peer_ip_` / `enumeration_cv_` members are removed — each
`wait_for_channel` call carries its own per-call state, scoped to the registration's
lifetime.

- `host/include/hololink/adapter/adapter.hpp` — public abstract `EnumerationCallback`
  with a single pure-virtual `handle_metadata(EnumerationMetadata&)` plus
  `EnumerationCallbackHandle = std::shared_ptr<EnumerationCallback>`. Adapter gains
  `register_ip(peer_ip, callback)` / `register_all(callback)` returning the handle, plus
  `unregister(handle)`. The `pending_by_peer_ip_` map and `enumeration_cv_` are removed;
  `registrations_` (vector of handles) + `registrations_mutex_` replace them. The actual
  filter behavior lives in private subclasses in adapter.cpp — the header carries no
  state and no conditional logic about it.
- `host/src/adapter.cpp` — anonymous namespace defines two `EnumerationCallback`
  subclasses: `PeerIpEnumerationCallback` stores the wrapped `std::function` plus a
  `peer_ip` to match against (it only invokes the function when `metadata["peer_ip"]`
  matches), and `AllPeersEnumerationCallback` unconditionally invokes the wrapped
  function. `register_ip` / `register_all` each construct the appropriate subclass and
  push it onto `registrations_`. `enumerate()` snapshots `registrations_` under the
  mutex, then unlocks and calls `handle_metadata` on every snapshot entry — each
  registration decides for itself whether to fire. Snapshotting lets callbacks re-enter
  the registration API and lets a callback unregister itself without invalidating the
  dispatch loop. `wait_for_channel(peer_ip, timeout)` creates a per-call
  `State { mutex, cv, optional<metadata> }` shared with a closure passed to
  `register_ip`. The closure stamps `state->received` and notifies on the first fire
  (later fires before unregister are silently ignored). The caller `cv.wait_for`s on
  `state->received`, then `unregister`s and either returns the metadata or throws on
  timeout.
- `host/python/hololink_adapter_py.cpp` — pybind binds `EnumerationCallback` as the
  opaque `EnumerationCallbackHandle` class (no methods). Adapter gains
  `register_ip(peer_ip, callback)` / `register_all(callback)` returning the handle and
  `unregister(handle)`. The `wait_for_channel` binding is unchanged in signature; its
  docstring is updated to mention the register_ip-based semantics.

**Why this shape.** The plan's design intent is that the adapter doesn't own application
state — it routes enumeration events to whatever code wants them. The callback API gives
applications the raw routing primitive. `wait_for_channel` becomes a convenience built
from `register_ip`, not the only entry point. An app that wants per-peer continuous
callbacks (e.g., to log every re-announcement, or to update a UI) calls `register_ip`
directly.

**Out of scope this phase.** Callback ordering is "registration order" (older first);
the adapter doesn't sort by specificity (register_ip vs register_all). Exceptions thrown
from a callback propagate up through `enumerate()` — they abort the dispatch loop, so
subsequent callbacks for the same announcement don't fire. Applications that need fault
isolation between callbacks wrap their callback bodies in their own try / catch.

**Done when.** The existing `wait_for_channel` enumeration tests still pass (they
exercise `register_ip` indirectly through `wait_for_channel`). Adapter source contains
no `pending_by_peer_ip_` / `enumeration_cv_` references; the new fields are
`registrations_` / `registrations_mutex_`.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD"
cmake --build "$BUILD" --target hololink_adapter
cmake --build "$BUILD" --target _hololink_adapter
cmake --build "$BUILD" --target hololink_adapter_enumeration_test
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" -R hololink_adapter_enumeration_test --output-on-failure
ctest --test-dir "$BUILD" -R hololink_adapter_python_test --output-on-failure
```

### Phase 5 extension — `DataChannelInterfaceV1` anchor base + per-board / per-channel dependent rule

**Context.** As the V1 service surface grew, every `ConfigurableService` ended up
independently reaching into the Publisher to look up its parent state (Hololink for
I2c/Sequencer/etc.), each one parsing its own slice of the enumeration metadata at
configure time. That produced two problems: (1) it was easy for two services on the same
channel to disagree on a metadata field if they were constructed against different
snapshots, and (2) the construct-order expectations were implicit. Pull the per-channel
state into a single anchor — `DataChannelInterfaceV1` — and make every other service on
the channel route through it.

`HololinkInterfaceV1` already plays the equivalent role for per-board state (it dedupes
by serial), so the rule is symmetric across two anchor scopes:

| Scope                                     | Anchor                   | Dependents                                        |
| ----------------------------------------- | ------------------------ | ------------------------------------------------- |
| **Per-board** (`serial=X`)                | `HololinkInterfaceV1`    | `I2cInterfaceV1`, `OscillatorInterfaceV1`         |
| **Per-channel** (`serial=X;data_plane=Y`) | `DataChannelInterfaceV1` | `SequencerInterfaceV1`, `RoceReceiverInterfaceV1` |

A dependent's `configure(metadata)` does a *cache-only* lookup of its anchor and throws
if the anchor isn't yet in the cache. Anchors must be explicitly constructed by the
application before dependents are requested.
`RoceDataChannelInterfaceV1::get_service(metadata)` (which drives
`DataChannelInterfaceV1::get_service(metadata)`) is allowed to construct its own anchor
(Hololink) transitively — that's the one exception, since a DataChannel cannot exist
without its Hololink.

**Deliverable.**

- New abstract base `DataChannelInterfaceV1` in
  `host/include/hololink/adapter/data_channel.hpp`. It owns the per-channel anchor
  surface:

  ```cpp
  class DataChannelInterfaceV1
      : public ConfigurableService<DataChannelInterfaceV1> {
  public:
      virtual const EnumerationMetadata& enumeration_metadata() const = 0;
      virtual std::shared_ptr<HololinkInterfaceV1> hololink() const = 0;

      static std::string instance_id_for(const EnumerationMetadata& metadata);

      // RTTI-free downcast hook (hololink_adapter builds with -fno-rtti).
      // Default returns nullptr; the RoCE specialization overrides.
      virtual RoceDataChannelInterfaceV1* as_roce_channel() { return nullptr; }
  };
  ```

  The `ConfigurableService<DataChannelInterfaceV1>` cache slot is what every per-channel
  dependent looks up against.

- `RoceDataChannelInterfaceV1` (in
  `host/include/hololink/adapter/roce_data_channel.hpp`) becomes a pure specialization
  that derives from `DataChannelInterfaceV1`, overrides `as_roce_channel()` to return
  `this`, and declares only the RoCE-specific surface (`attach_receiver`,
  `detach_receiver`). It is **not** an independent `ConfigurableService` — the cache
  slot belongs to the base.

  Both `RoceDataChannelInterfaceV1::get_service(metadata)` and
  `RoceDataChannelInterfaceV1::get_service(module, instance_id)` are kept as convenience
  entry points. Each routes through `DataChannelInterfaceV1::get_service(...)` and
  downcasts via `as_roce_channel()`, returning a
  `shared_ptr<RoceDataChannelInterfaceV1>` whose ownership shares with the base via the
  aliasing-constructor share. Callers that want a RoCE-typed pointer ask through this
  API rather than casting from the base by hand.

- `HololinkInterfaceV1` gains
  `virtual const EnumerationMetadata& enumeration_metadata() const = 0;`, symmetric with
  the new `DataChannelInterfaceV1`. `HololinkImplV1::configure` stashes the metadata it
  was configured with. Per-board dependents read fields off
  `hololink->enumeration_metadata()` rather than holding their own copy.

- `RoceDataChannelImplV1::configure(metadata)` is the one place that performs a
  cache-or-construct Hololink lookup:
  `HololinkInterfaceV1::get_service(module, metadata)`. The result is stashed in
  `hololink_`, the legacy `hololink::DataChannel` is built against
  `hololink_->legacy_access()`, and `hololink()` returns `hololink_` to sibling
  services.

- Per-instance flavor (I2c `bus`, Sequencer `kind`) is parsed in the supplement's
  `Publisher::construct_service` switch and passed to the impl's constructor;
  `configure()` reads it off member state instead of re-parsing the instance_id.

  ```cpp
  // module_entry.cpp
  if (type_id == typeid(I2cInterfaceV1)) {
      const auto pairs = parse_name_value_pairs(instance_id);
      const uint32_t bus = std::stoul(pairs.at("bus"));
      return std::make_shared<I2cImplV1>(bus);
  }
  ```

- Per-board dependents (`I2cImplV1`, `OscillatorImplV1`):

  ```cpp
  void configure(const EnumerationMetadata& metadata) override {
      std::call_once(once_, [&]() {
          const std::string hololink_id
              = HololinkInterfaceV1::instance_id_for(metadata);
          auto hololink_iface = HololinkInterfaceV1::get_service(
              this->module(), hololink_id.c_str());
          if (!hololink_iface) {
              throw std::runtime_error(
                  "While configuring I2cInterface: parent HololinkInterface "
                  "has not been constructed for " + hololink_id
                  + " — call HololinkInterfaceV1::get_service(metadata) first");
          }
          auto hololink_impl
              = std::static_pointer_cast<HololinkImplV1>(hololink_iface);
          backing_ = hololink_impl->legacy_access()->get_i2c(bus_);
      });
  }
  ```

- Per-channel dependents (`SequencerImplV1`, `RoceReceiverImplV1`) do the same
  cache-only fetch against `DataChannelInterfaceV1`, throw on miss, and read every
  per-channel field from `channel->enumeration_metadata()`. The `channel->hololink()`
  accessor is the canonical Hololink path for these services — they never re-lookup it
  through the Publisher.

- `RoceReceiverOp` calls the kept-for-convenience
  `RoceDataChannelInterfaceV1::get_service(metadata_.get())`. The convenience form
  routes through the base cache and downcasts internally, so the operator never sees the
  anchor split.

**Python bindings.**

- `DataChannelInterfaceV1` binds with `enumeration_metadata` / `hololink` /
  `as_roce_channel` on a base `py::class_<DataChannelInterfaceV1>`.
- `RoceDataChannelInterfaceV1` binds as a derived `py::class_` (using pybind11's
  base-class declaration) so `attach_receiver` / `detach_receiver` are available on the
  RoCE-typed view. The `__init__.py` exports both under their explicit V1 names
  (`DataChannelInterfaceV1`, `RoceDataChannelInterfaceV1`) — no unversioned alias.

**Out of scope.** No change to the C ABI (`HOLOLINK_ADAPTER_API_VERSION` stays at 1 — no
new ABI surface, the existing `get_service` ABI form covers both anchors via their
respective `type_id` keys). No change to the Publisher / `construct_service` signature.

**Why this shape.** The anchor pattern makes construct-order an explicit contract:
anchors are application-driven, dependents fail fast if their anchor isn't there.
Splitting `DataChannelInterfaceV1` from `RoceDataChannelInterfaceV1` lets sibling
services be transport-agnostic (they don't care whether the channel is RoCE or some
future COE / 1722 variant — they only ask the channel for its metadata and Hololink).
The RTTI-free downcast via `as_roce_channel()` keeps `-fno-rtti` intact while still
letting transport-specific code surface the typed view.

**Done when.** Unit tests prove:

- A per-channel dependent's `configure` throws when its `DataChannel` isn't yet in the
  cache; succeeds once it is.
- A per-board dependent's `configure` throws when its `Hololink` isn't yet in the cache;
  succeeds once it is.
- `RoceDataChannelInterfaceV1::get_service(metadata)` and
  `DataChannelInterfaceV1::get_service(metadata)` return the same cached object (the
  RoCE form just downcasts).
- All existing service tests pass under the new contract once they're updated to
  construct anchors first.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD"
cmake --build "$BUILD" --target hololink_adapter
cmake --build "$BUILD" --target hsb_lite
cmake --build "$BUILD" --target hsb_lite_2510
cmake --build "$BUILD" --target _hololink_adapter
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" --output-on-failure -LE hardware
```

### Phase 5 extension — `Service<X>::for_each_type_id` via `ServiceAlias` typedef

**Context.** As the supplements grew, `HsbLitePublisher::construct_service` (in both
`module/hsb_lite/module_entry.cpp` and `module/hsb_lite_2510/module_entry.cpp`) ended up
hard-coding the `type_id` hierarchy of each impl class. Every branch listed the V1
interface `type_id`, the impl `type_id`, and (for per-board subclasses) the parent impl
`type_id`, then called one `ServicePublisher<T>::publish` per key. The chain is a
property of each class's inheritance, but the supplement carried the knowledge — adding
a fourth published key per class meant editing three or four places.

Move the chain knowledge into the classes themselves, but without making every class
carry a five-line `for_each_type_id` body. Provide `for_each_type_id` once on
`Service<X>` and let each class declare its chain parent via a single
`using ServiceAlias = …;` typedef.

**Deliverable.**

- `host/include/hololink/adapter/service.hpp` grows a SFINAE detector
  (`has_service_alias<U>`) plus a static `for_each_type_id` template on `Service<X>`:

  ```cpp
  template <typename Callback>
  static bool for_each_type_id(Callback&& cb)
  {
      if constexpr (has_service_alias<X>::value) {
          if (!X::ServiceAlias::for_each_type_id(cb)) return false;
      }
      return cb(X::type_id);
  }
  ```

  Callback signature `bool(const char*)`: return `false` to halt iteration, `true` to
  continue. The `&&` short-circuit propagates the halt up the chain.

- Every V1 interface inherits this default automatically through its
  `Service<InterfaceV1>` base — with no `ServiceAlias` declared, the chain terminates at
  the interface's own `type_id`. **Zero** lines per interface.

- Every impl class declares two `using` lines:

  ```cpp
  class HololinkV1 : public HololinkInterfaceV1,
                     public Service<HololinkV1> {
  public:
      static constexpr const char* type_id = "hololink.module_core.v1";
      using Service<HololinkV1>::get_service;
      using Service<HololinkV1>::for_each_type_id;   // disambiguates the inherited
                                                     // chain walker
      using ServiceAlias = HololinkInterfaceV1;      // declares the chain parent
      // ... existing methods; no for_each_type_id body
  };
  ```

  `using Service<HololinkV1>::for_each_type_id;` mirrors the existing
  `using Service<HololinkV1>::get_service;` — both name-hide the inherited V1 chain
  overloads that would otherwise be ambiguous against the impl's direct `Service<X>`
  base.

- Per-board subclasses chain through their parent **impl**, not the interface, so the
  chain emits three keys:

  ```cpp
  class HsbLite2510RoceDataChannelV1
      : public module_core::RoceDataChannelV1,
        public Service<HsbLite2510RoceDataChannelV1> {
  public:
      using ServiceAlias = module_core::RoceDataChannelV1;
      // ...
  };
  ```

  Iteration yields
  `{RoceDataChannelInterfaceV1::type_id, RoceDataChannelV1::type_id, HsbLite2510RoceDataChannelV1::type_id}`.

- `ServicePublisher<T>::publish(instance_id, impl)` walks `T::for_each_type_id` and
  registers `impl` under every emitted key. The explicit-`type_id` overload
  (`publish(instance_id, type_id, impl)`) remains for the few callers that already know
  the key.

- `Publisher::has_type_id<T>(type_id)` returns `true` iff `type_id` appears in `T`'s
  reported set:

  ```cpp
  template <typename T>
  static bool has_type_id(const std::string& type_id) {
      bool found = false;
      T::for_each_type_id([&](const char* tid) {
          if (type_id == tid) { found = true; return false; }
          return true;
      });
      return found;
  }
  ```

- `HsbLitePublisher::construct_service` branches collapse to a uniform shape — gate with
  `Publisher::has_type_id<T>(type_id)`, construct inline, publish with one call:

  ```cpp
  if (Publisher::has_type_id<HsbLite2510RoceDataChannelV1>(type_id)) {
      auto anchor = DataChannelInterfaceV1::get_service(
          this->self_module(), instance_id.c_str());
      auto impl = std::make_shared<HsbLite2510RoceDataChannelV1>(std::move(anchor));
      ServicePublisher<HsbLite2510RoceDataChannelV1>(shared_from_this())
          .publish(instance_id, impl);
      return impl;
  }
  ```

  No `||`-chains over `type_id` strings, no per-impl `ServicePublisher<X>` repetition.

**Out of scope.** `SequencerInterfaceV1` / `I2cInterfaceV1` keep their bespoke
construction paths — their factory bodies do significant work (parse `instance_id`,
fetch a sibling impl) and don't fit the `make_shared<T>()` shape. They register under a
single key, so the manual
`ServicePublisher<InterfaceV1>::publish(instance_id, key, impl)` form is already
correct. `host::Reactor` / `host::Logging` need no annotation — they sit on the default
chain (V1 interface with no `ServiceAlias`).

**Why this shape.** Two `using` lines per impl is the minimum that lets the framework
walk the chain. The first (`using Service<X>::for_each_type_id;`) is structural — it
matches the existing `using Service<X>::get_service;` line and resolves the same name
collision against the inherited interface. The second (`using ServiceAlias = Parent;`)
is the only *new* piece of knowledge each class contributes. SFINAE detects the typedef,
so V1 interfaces — which have no parent in the chain — need nothing at all.

Naming alternatives considered for the typedef: `PublishAlias`, `ParentImpl`, `Parent`.
`ServiceAlias` matched the surrounding `ServiceLocatable` / `ServicePublisher` /
`Service<X>` vocabulary best.

**Done when.**

- Both supplement `module_entry.cpp` files have no `||`-chains over `type_id`
  comparisons and no per-impl `ServicePublisher<X>` repetition.
- `g++ -std=c++17 -fno-rtti -fsyntax-only` passes on both supplement `module_entry.cpp`
  files and on the module_core impl headers.
- Full out-of-tree cmake build produces both supplement `.so`s.
- `ctest -LE hardware` is green.

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

### Phase 5 extension — `module_core::HsbLitePublisher` as the canonical Publisher base

**Context.** HSB-Lite is the canonical reference configuration: a `Publisher` that
publishes the standard set of services (HololinkV1, DataChannelV1, RoceDataChannelV1,
RoceReceiverV1, HsbLiteV1, HsbLiteOscillatorV1, plus peripherals — I2c today — and
Sequencer) against a board with HSB-Lite's peripheral inventory. Both supplements
(`hsb_lite`, `hsb_lite_2510`) duplicated ~85% of `construct_service` between them, plus
identical `HsbLiteV1` / `HsbLiteOscillatorV1` / `HsbLiteEnumerationV1` classes that
differed only in type_id strings. The user expects more future supplements to follow
this pattern, so the canonical configuration belongs in `module/core/` rather than
per-supplement.

**Deliverable.**

- **New** `module/core/hsb_lite_default.hpp` — `module_core::HsbLiteV1` (type_id
  `"hsb_lite.module_core.v1"`) + `module_core::HsbLiteOscillatorV1` (type_id
  `"oscillator.module_core.v1"`). Implementations lifted verbatim from the hsb_lite
  supplement, namespace-moved.

- **New** `module/core/hsb_lite_publisher.hpp` — `module_core::HsbLitePublisher`. Each
  canonical service's construction logic is a protected virtual method following the "if
  `has_type_id<X>`, make X, publish X, return true; else return false" pattern.
  `construct_service` is a short-circuit OR over each branch with `construct_overrides`
  at the head as the board-extension hook:

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

  `construct_overrides` returns `false` in the base; a board subclass overrides it to
  handle one or more type_ids ahead of the canonical chain (substituting impl classes on
  canonical type_ids, or adding bespoke type_ids like SPI). Anything
  `construct_overrides` handles preempts the canonical chain. The per-service virtuals
  (`construct_hololink`, …, `construct_i2c`) are also overridable for targeted
  substitution of one canonical branch — boards pick whichever shape (centralized in
  `construct_overrides` or one targeted override) reads best.

- **Framework signature change.** `Publisher::construct_service` returns `bool` instead
  of `std::shared_ptr<void>`. The framework (`host/src_module/module_base.cpp`
  `Publisher::lookup`) already discarded the returned pointer — checked truthiness, then
  re-looked-up the cache that `ServicePublisher::publish` had populated during the call.
  The signature is now honest about that. All `Publisher` subclasses (host-side
  `HostPublisher`, every supplement, every test stub) updated.

- **Supplements thin out.** `module/hsb_lite/module_entry.cpp` drops to ~76 lines (was
  ~315): a local `HsbLitePublisher : module_core::HsbLitePublisher` subclass that
  overrides `module_name()` to return `"hsb_lite"`, plus the C-ABI entry.
  `module/hsb_lite_2510/module_entry.cpp` drops to ~239 lines (was ~422): keeps the
  2510-specific impl subclasses + a `HsbLite2510Publisher` subclass that overrides only
  `construct_roce_data_channel` and `construct_roce_receiver`.

**Why this shape.** Two equivalent extension paths reflect two different intents —
`construct_overrides` for "I want to redirect a few type_ids in one place, including
bespoke ones the base doesn't know about," and per-service overrides for "I want to
substitute exactly one canonical branch." Both methods are virtual; boards pick the
shape that reads best for them. The rejected alternative — a
`std::map<type_id, std::function<...>>` registry populated by `register_factory(...)`
calls — introduces lifetime/ordering concerns the virtual approach avoids, and fragments
dispatch logic across the base's `construct_service` and the subclass's constructor.

**Done when.**

- Both supplements + every test stub use `Publisher::construct_service` returning
  `bool`.
- `module/hsb_lite/module_entry.cpp` carries no canonical-impl classes; just a
  `module_name()` override and the C-ABI entry.
- `module/hsb_lite_2510/module_entry.cpp` carries only 2510-specific impl subclasses + a
  publisher subclass with two construct-method overrides.
- `g++ -std=c++17 -fno-rtti -fsyntax-only` passes on framework + supplements
  - test stubs.
- Full out-of-tree cmake build produces both supplement `.so`s.

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

### Phase 5 extension — Publisher absorbs `EnumerationInterfaceV1`

**Context.** `module_core::HsbLiteEnumerationV1` was a tiny standalone class that
stamped `metadata["module_name"]` and called `set_data_plane_metadata(2, 2, 2)`. Each
supplement's `hololink_adapter_init` constructed it separately from the Publisher and
published it onto the Publisher — artificial split, since the enumeration is a
per-supplement property that belongs to the Publisher.

**Deliverable.**

- `module_core::HsbLitePublisher` multi-inherits `EnumerationInterfaceV1`. `Publisher`
  and `EnumerationInterfaceV1` are disjoint bases (no diamond; `Publisher` derives from
  `enable_shared_from_this<Publisher>`, while `EnumerationInterfaceV1` derives from
  `Service<EnumerationInterfaceV1>` → `ServiceBase` virtually).

- The Publisher's `update_metadata` is the absorbed body — calls
  `metadata["module_name"] = module_name();` followed by
  `set_data_plane_metadata(metadata, 2, 2, 2);`.

- Two new protected virtuals:

  - `virtual std::string module_name() const = 0` — pure. Every supplement declares its
    own identity. The canonical `hsb_lite` and `hsb_lite_2510` supplements each gain a
    5-line subclass to provide their `module_name()`.
  - `virtual void set_data_plane_metadata(metadata, total_sensors, total_dataplanes, sifs_per_sensor)`
    — virtual. Default body is the former free-function body (lifted from
    `module/core/data_plane_metadata.hpp`). Override for a different per-data-plane
    address layout.

- New `virtual void publish_enumeration()`. Registers `this` Publisher as the
  `EnumerationInterfaceV1` singleton in its own registry. Uses the `shared_ptr` aliasing
  constructor — `std::shared_ptr<EnumerationInterfaceV1>(self, this)` — to materialize
  an `EnumerationInterfaceV1`-typed shared_ptr that shares ownership with the
  Publisher's `shared_ptr<Publisher>`. No RTTI; works under `-fno-rtti`. Override to
  publish a different `EnumerationInterfaceV1` impl entirely.

- `module_core::HsbLiteEnumerationV1` class deleted.

- `module/core/data_plane_metadata.hpp` deleted. Its body now lives as the default of
  `HsbLitePublisher::set_data_plane_metadata`.

**Out of scope.** `Publisher` (the host-side base) doesn't change. The host-side
`HostPublisher` in `adapter.cpp` doesn't inherit `EnumerationInterfaceV1` — only
`HsbLitePublisher` does.

**Done when.**

- Repo grep for `HsbLiteEnumerationV1` and `set_data_plane_metadata` (free function)
  returns no live code references — only historical README mentions.
- Both supplements drop `make_shared<HsbLiteEnumerationV1>` /
  `ServicePublisher<EnumerationInterfaceV1>::publish(...)` from `hololink_adapter_init`;
  replaced with `publisher->publish_enumeration()`.
- Enumeration flow stamps the same metadata it did before.
- `g++ -std=c++17 -fno-rtti -fsyntax-only` clean on framework + supplements + stubs.

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

### Phase 5 extension — Publisher absorbs host Module/logger/reactor bootstrap + `publish_frame_metadata`

**Context.** Every module-side `hololink_adapter_init` opened with the same bootstrap
dance — construct a `LoadedModule` from `init->{get_service,release_service}`, fetch
`LoggingInterfaceV1`, wire the per-binary `HSB_LOG` cache, sometimes fetch `ReactorV1`.
Each HSB-Lite-shaped supplement then made a `FrameMetadataV1` and published it as
`FrameMetadataInterfaceV1`. All identical, all per-supplement statics.

**Deliverable.**

- New `Publisher` constructor `Publisher(const hololink_adapter_init_t* init)` validates
  `init` inline and performs the bootstrap: builds the host `Module`, fetches
  `LoggingInterfaceV1` (wires the HSB_LOG cache), fetches `ReactorV1`. All three stored
  as private members; exposed via three new accessors `host_module()` / `logger()` /
  `reactor()`.

- The constructor is delegating: `Publisher(init) : Publisher() { … }`. The default
  `Publisher()` ctor stays — it's the host-side `HostPublisher`'s construction path (the
  host process has no `init` to bootstrap from). The init-taking ctor is `public`
  (despite `Publisher` being abstract via the pure-virtual `construct_service`) so
  derived classes can inherit it via `using Publisher::Publisher;` and have
  `make_shared<MyPublisher>(init)` work end-to-end.

- New `HsbLitePublisher::publish_frame_metadata()` virtual. Default constructs
  `module_core::FrameMetadataV1` and publishes it as `FrameMetadataInterfaceV1` in this
  Publisher's registry. Parallel to `publish_enumeration()`. A supplement subclass
  overrides to publish a different `FrameMetadataInterfaceV1` impl.

- Both supplements + four test stubs drop their `g_host_module` / `g_logger` / (where
  present) `g_frame_metadata` statics. Each test stub now instantiates
  `make_shared<TestPublisher>(init)` and uses `publisher->host_module()` /
  `publisher->reactor()` where it needs them; the singletons stub specifically drops its
  explicit `ReactorV1::get_service(g_host_module)` lookup.

**Out of scope.** Status-code reporting for init failure stays via the same mechanism
the supplements used (catch the throw, set `HOLOLINK_ADAPTER_MODULE_INIT_FAILED`).
That's revisited next.

**Done when.**

- Repo grep across supplements + test stubs for `LoadedModule::create`,
  `LoggingInterfaceV1::get_service`, `set_hsb_logger_cache`, `ReactorV1::get_service`,
  `ServicePublisher<FrameMetadataInterfaceV1>` — no remaining live references.
- Framework files, supplements, and test stubs syntax-check clean together.

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

### Phase 5 extension — `Publisher::setup(init)` returns status; no `try`/`catch` in `hololink_adapter_init`

**Context.** With the bootstrap inside the constructor, init validation was still
throwing on a malformed `init` from the host. The supplement caught all exceptions and
translated to `HOLOLINK_ADAPTER_MODULE_INIT_FAILED`, collapsing the only realistic
external error path (init validation → `HOLOLINK_ADAPTER_INVALID_PARAMETER`) into the
generic catch-all. Goal: route the validation through a status return so the supplement
doesn't need a catch.

**Deliverable.**

- The init-taking `Publisher` constructor is removed. `Publisher` has just the default
  ctor now. The init parameter moves to a new method
  `Publisher::setup(const hololink_adapter_init_t* init)` — virtual, returns
  `hololink_adapter_module_services_t` (the C-ABI struct), defined in
  `host/src_module/module_base.cpp`. Its body:

  ```cpp
  hololink_adapter_module_services_t Publisher::setup(
      const hololink_adapter_init_t* init)
  {
      hololink_adapter_module_services_t result {};
      if (!init || init->api_version != HOLOLINK_ADAPTER_API_VERSION
          || !init->get_service || !init->release_service) {
          result.status = HOLOLINK_ADAPTER_INVALID_PARAMETER;
          return result;
      }
      host_module_ = LoadedModule::create(
          init->get_service, init->release_service);
      logger_ = LoggingInterfaceV1::get_service(host_module_);
      set_hsb_logger_cache(logger_.get());
      reactor_ = ReactorV1::get_service(host_module_);
      return callbacks();
  }
  ```

  Bad init returns a status; no throw. Good init runs the same bootstrap + `callbacks()`
  that the constructor used to.

- `HsbLitePublisher::setup(init)` forwards to `Publisher::setup(init)`, early-returns on
  non-OK status, then calls `publish_frame_metadata()` and `publish_enumeration()`
  before returning the result:

  ```cpp
  hololink_adapter_module_services_t setup(
      const hololink_adapter_init_t* init) override
  {
      auto result = Publisher::setup(init);
      if (result.status != HOLOLINK_ADAPTER_OK) {
          return result;
      }
      publish_frame_metadata();
      publish_enumeration();
      return result;
  }
  ```

- Every supplement's `hololink_adapter_init` becomes three lines, no try/catch:

  ```cpp
  auto publisher = std::make_shared<HsbLitePublisher>();
  g_publisher = publisher;
  return publisher->setup(init);
  ```

- Every test stub follows the same shape. The stub that reads from
  `publisher->host_module()` (to fetch `HostTestServiceV1`) reorders to call
  `setup(init)` first, return early on validation failure, then use the now-populated
  `host_module()` before publishing its own service into the registry.

- `using Publisher::Publisher;` (and the parallel
  `using module_core::HsbLitePublisher::HsbLitePublisher;`) declarations in subclasses
  are removed — no init-taking ctor to inherit anymore.

**Why this shape.** The supplement's `hololink_adapter_init` is the C-ABI boundary;
exceptions can't propagate across `extern "C"` without UB. The previous design relied on
a catch as the firewall. Returning a status from `setup(init)` puts the only realistic
external error (host-malformed init) onto an explicit return path, leaving no exception
sources the supplement needs to translate. Remaining throw sources inside `setup` (host
singleton missing — host bug; `std::bad_alloc` — system failure; duplicate registration
— programming error in a subclass) escape and terminate the process; that's the right
outcome for each of those.

**Done when.**

- Repo grep across supplements + test stubs for `try {` or `catch (` in
  `hololink_adapter_init` bodies — no remaining matches.
- Repo grep for `make_shared<…Publisher>(init)` — no remaining matches; the init
  parameter goes to `setup`, not the constructor.
- `g++ -std=c++17 -fno-rtti -fsyntax-only` clean across framework + supplements + four
  test stubs (8 TUs together).
- Test stubs that use `host_module()`/`reactor()` work — they call `setup(init)` first
  and use the populated accessors afterwards.

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

### Phase 5 extension — HsbLite2510 implementations promoted to `module/core/`

**Context.** The 2510 supplement carried five HsbLite2510-specific classes — four in
`module/hsb_lite_2510/module_entry.cpp` plus
`module/hsb_lite_2510/hsb_lite_2510_data_channel.hpp`. They sat in the supplement
because that was the only consumer. Future partner supplements targeting pre-0x2603 /
pre-0x2602 FPGA revisions are expected to reuse the same dispatch — which makes them
canonical-but-versioned, not supplement-private.

**Deliverable.**

- **New** `module/core/hsb_lite_2510_roce_receiver.hpp` —
  `module_core::HsbLite2510RoceReceiver` (renamed from `HsbLite2510LegacyRoceReceiver`;
  the "Legacy" prefix dropped) + `module_core::HsbLite2510RoceReceiverV1`. Identical
  implementation.

- **New** `module/core/hsb_lite_2510_data_channel.hpp` —
  `module_core::HsbLite2510DataChannel` (relocated from `hsb_lite` namespace)

  - `module_core::HsbLite2510RoceDataChannelV1`. Now carries direct includes for
    `<thread>`, `<chrono>`, `<limits>`, `<stdexcept>`, `<fmt/format.h>` — the original
    supplement header relied on transitive includes from `data_channel.hpp`.

- **New** `module/core/hsb_lite_2510_publisher.hpp` —
  `module_core::HsbLite2510Publisher`. Concrete subclass of
  `module_core::HsbLitePublisher`;
  `module_name() const override { return "hsb_lite_2510"; }`; overrides
  `construct_roce_data_channel` and `construct_roce_receiver` to construct the 2510 V1
  wrappers above. Future supplements targeting the same FPGA revision instantiate this
  publisher directly, or subclass it further to override `module_name`.

- `module/hsb_lite_2510/module_entry.cpp` shrinks to ~33 lines: just the static
  `g_publisher` and a `hololink_adapter_init` that instantiates
  `module_core::HsbLite2510Publisher` and calls `setup(init)`. No supplement-local class
  definitions remain.

- `module/hsb_lite_2510/hsb_lite_2510_data_channel.hpp` deleted.

- Per-impl type_ids preserved unchanged (`roce_receiver.hsb_lite_2510.v1`,
  `roce_data_channel.hsb_lite_2510.v1`) — the type_id describes the impl variant, not
  the C++ namespace.

**Out of scope.** No CMakeLists change needed. Header-only additions ride along with
`hololink::module`'s `target_include_directories` PUBLIC entry, which already exposes
the whole `module/core/` directory.

**Done when.**

- Repo grep for `class HsbLite2510*` definitions in the 2510 supplement — none remain.
- Repo grep for `HsbLite2510LegacyRoceReceiver` — no live references; the class is now
  `module_core::HsbLite2510RoceReceiver`.
- `g++ -std=c++17 -fno-rtti -fsyntax-only` clean on framework + supplements + four test
  stubs.

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

### Phase 5 extension — Leopard VB1940 module + `Vb1940Cam` driver + single + stereo `module_vb1940_player` examples

**Context.** Leopard VB1940 is the second device family targeted by the adapter (the
first being HSB-Lite, covered by Phase 5 and its extensions). `module/leopard_vb1940/`
was named in the directory layout and the first-party module table but never built; no
host-side sensor driver existed for VB1940 under the adapter; and the only VB1940
example ran against the legacy `DataChannel` / `Hololink` types. This extension stands
up all three pieces together so a Leopard VB1940 board enumerates, loads its own `.so`,
brings up its camera through the adapter, and streams frames through a Holoscan pipeline
whose shape matches `module_imx274_player.cpp`. It validates the cross-UUID cohabitation
demo described under "First-party modules": a Leopard VB1940 board and an HSB-Lite board
can now be driven concurrently in one process.

**Deliverable.**

- **New module `module/leopard_vb1940/`** — second first-party module. Ships as the
  bare-UUID `hololink_f1627640-b4dc-48af-a360-c55b09b3d230.so` via
  `add_hololink_module(NAME leopard_vb1940 UUID f1627640-b4dc-48af-a360-c55b09b3d230 NO_COMPAT_SUFFIX ...)`.
  This is a documented exception to the "modules should not use `NO_COMPAT_SUFFIX`"
  rule: the Leopard VB1940 FPGA does not publish a compat-id over bootp today, so the
  bare-UUID filename is what `Adapter::load_module_for` resolves to via its no-compat-id
  fallback path. `module/hsb_lite_2510/` is the other documented exception. Switch to
  the compat-suffixed form once the FPGA reports a compat-id.

  - `module/leopard_vb1940/module_entry.cpp` — defines every module-private type inline
    (file-scope; no separate per-impl headers) plus `hololink_adapter_init`:
    - `LeopardVb1940V1` — the `LeopardVb1940InterfaceV1` impl. Owns the per-board
      sensor-reset register state via two idempotent primitives, both reachable from
      `Vb1940Cam`'s lifecycle:
      - `expect_sensor(int64_t sensor_number)` — called from
        `Vb1940Cam::configure(mode)`. Adds the bit to `expected_sensors_` (32-bit
        cumulative mask of channels the application has declared). No hardware write.
      - `enable_sensor(int64_t sensor_number)` — called from `Vb1940Cam::start()`.
        Validates the sensor is in `expected_sensors_`, then commits the cumulative mask
        to the FPGA sensor-reset register `0x8` if the hardware doesn't already match.
        First commit after a fresh board (or after the on_reset listener cleared
        `committed_mask_`) writes `0x8 = 0x0` defensively then writes the cumulative
        `expected_sensors_`, sleeps 100 ms. Subsequent commits where the mask matches
        are no-ops. The two-phase split (expect → enable) ensures the supplement sees
        every channel's expectation before any single channel commits — without it, the
        first stereo channel would write a partial mask (e.g. `0x1`) and we can't assume
        hardware honours partial-mask writes (legacy only transitions `0x0 -> full_mask`
        atomically). `expected_sensors_` persists across board reset (application
        intent, not hardware state); `committed_mask_` is cleared by an `on_reset`
        listener so the next `enable_sensor` re-commits.
    - `LeopardVb1940OscillatorV1` — the `OscillatorInterfaceV1` impl backed by Bajoran
      TS2. Published per board under `OscillatorInterfaceV1`'s `serial=...;data_plane=N`
      instance_id — `data_plane` is the physical HIF index (always `0` for Leopard's
      single-HIF carrier), so both stereo channels resolve to the same
      `OscillatorImplV1` instance, matching the physical one-chip-per-board reality.
      `enable(uint64_t)` programs the on-board Renesas Bajoran Lite TS2 on first call,
      caches the rate, returns true on same-rate calls and false on mismatched rates. An
      `on_reset` listener clears the cache so a post-reset `enable()` reprograms. The
      TS2 `DEVICE_CONFIGURATION` table lives in the existing legacy header
      `src/hololink/sensors/camera/vb1940/renesas_bajoran_lite_ts2.hpp` and is reused
      verbatim.
    - `LeopardVb1940Publisher` — an `HsbLitePublisher` subclass that overrides:
      - `module_name() const override { return "leopard_vb1940"; }`.
      - `update_metadata(...)` — Leopard's per-board enumeration configuration.
        `metadata["data_plane"]` is the logical channel index (the discriminator every
        per-channel service locator keys on; `Vb1940Cam::use_sensor` stamps it per
        camera for stereo). The override passes that value as `sensor_number` to the
        inherited `set_data_plane_metadata` and hardcodes the physical `data_plane`
        parameter to `0` — Leopard VB1940-AIO has one physical data plane shared by
        every sensor, so both stereo channels share `hif_address = 0x02000300`. The
        per-sensor SIF / VP / `vp_mask` / `frame_end_event` fields come out of
        `sensor_number`.
      - `construct_hsb_lite` (return false — Leopard isn't HSB-Lite),
      - `construct_oscillator` (publish `LeopardVb1940OscillatorV1`),
      - `construct_overrides` (publish `LeopardVb1940V1` on the `leopard_vb1940.impl.v1`
        type_id).
    - `hololink_adapter_init` instantiates `LeopardVb1940Publisher` and calls
      `setup(init)`. Same shape `module/hsb_lite/module_entry.cpp` uses — every
      module-private type lives in `module_entry.cpp` rather than its own header. (The
      `module/core/hsb_lite_2510_publisher.hpp` + `hsb_lite_2510_*` impl headers case is
      the documented exception: future partner supplements targeting that FPGA revision
      are expected to instantiate them directly, so they live in `module/core/`.)
  - `module/leopard_vb1940/include/hololink/adapter/leopard_vb1940/leopard_vb1940.hpp` —
    the `LeopardVb1940InterfaceV1` declaration in namespace
    `hololink::adapter::leopard_vb1940` per "Per-board supplements are scoped to a
    HololinkInterface". Declares the `expect_sensor` / `enable_sensor` virtuals
    (impl-side contract documented above). INTERFACE-library target
    `hololink::leopard_vb1940::headers` exposes the header to applications that opt in
    and to the per-sensor `Vb1940Cam` driver that consumes it.
  - `module/leopard_vb1940/CMakeLists.txt` — wires `add_hololink_module(...)` and the
    headers INTERFACE library. Picks up `hololink::hsb_lite::headers` (transitively
    required by `HsbLitePublisher`) and `hololink::module`'s `INCLUDE_DIRECTORIES`.

- **New driver `hololink_adapter/host/sensors/vb1940/`** — mirrors
  `hololink_adapter/host/sensors/imx274/` file-for-file:

  - `vb1940_cam.hpp` / `vb1940_cam.cpp` —
    `hololink::adapter::sensors::vb1940::Vb1940Cam`. Two constructors:
    - `Vb1940Cam(shared_ptr<HololinkInterfaceV1>, shared_ptr<OscillatorInterfaceV1>, shared_ptr<LeopardVb1940InterfaceV1>, int64_t sensor_number, uint32_t i2c_bus = DEFAULT_CAM_I2C_BUS)`
      — explicit form. Each constructor argument is one shared resource the driver
      coordinates with (Hololink for register / I2C transactions, Oscillator for the
      board clock, LeopardVb1940 for sensor-reset bring-up); `sensor_number` identifies
      which channel on the board this driver instance represents.
    - `Vb1940Cam(const EnumerationMetadata&, uint32_t i2c_bus = DEFAULT_CAM_I2C_BUS)` —
      metadata-driven form. Resolves `HololinkInterfaceV1`, `OscillatorInterfaceV1`, and
      `LeopardVb1940InterfaceV1` via the metadata-taking `get_service` overloads, and
      reads `sensor_number` from `metadata["sensor_number"]` (falling back to
      `metadata["data_plane"]`).
    - No `use_expander_configuration` static — Leopard VB1940 has no per-board I2C
      expander; the LII2CExpander class stays IMX274-specific.
  - Public surface is minimal — just what callers actually need. Everything else
    (register-level I/O, EEPROM, FSM-walking helpers, firmware-loading sequence,
    calibration parsing) lives as private members in the class or as file-local
    constants in `vb1940_cam.cpp`:
    - Lifecycle: `configure(Vb1940_Mode)` and `start()` split deliberately. The hardware
      writes that configure the sensor (clock, sensor-reset register, FSM walk, mode
      register table) live in `start()` so they happen *after* `hololink->start()` /
      `hololink->reset()` — fresh-from-reset state — and so the supplement can collect
      every channel's `expect_sensor` registration before any one of them commits.
      - `configure(mode)` is lightweight: caches `mode_` / `width_` / `height_` /
        `pixel_format_` via `set_mode(mode)` so the example's `compose()` can query
        them; then calls `leopard_->expect_sensor(sensor_number_)` to register this
        channel with the supplement. No hardware writes, no FSM walk.
      - `start()` does the bring-up + FSM + streaming start:
        - `setup_clock()` (delegates to `oscillator_->enable(25_000_000)`; idempotent
          via the supplement's rate-commit cache).
        - `leopard_->enable_sensor(sensor_number_)` (commits the cumulative expected
          mask to the FPGA sensor-reset register; idempotent via the supplement's
          committed-mask cache).
        - FSM walk: `get_device_id` → if not in `SW_STBY`, `status_check` → `SYSTEM_UP`
          → `BOOT` → `WAIT_CERTIFICATE` → `do_secure_boot` → `write_vt_patch`.
        - `configure_camera(mode_)` (applies the per-mode register table from the cached
          mode).
        - `VB1940_START_SEQUENCE` (streaming start command) + wait for
          `SYSTEM_FSM_STATE_REG == STREAMING`.
      - `stop()` — applies `VB1940_STOP_SEQUENCE` and waits for `SW_STBY`. Unchanged
        from the original port.
    - Exposure + gain tuning: `set_exposure_reg(value = 0x0014)`,
      `set_analog_gain_reg(value = 0x00)`.
    - Format accessors: `pixel_format()`, `bayer_format()`, `width()`, `height()`.
    - Converter setup: `configure_converter(shared_ptr<hololink::csi::CsiConverter>)` —
      status-line trailers (1 prepended, 2 trailing) preserved per legacy
      `native_vb1940_sensor.cpp:516-550`.
    - Calibration: `get_rgb_calibration_data()`, `get_ir_calibration_data()` (return the
      public `CalibrationData` struct). RGB pages 0-3, IR pages 4-7; each block is 30
      big-endian doubles.
  - Register / EEPROM addresses, wait-time constants, the `FrameFormat` table, the
    `Vb1940_Mode` enum (public), the per-mode `_SEQUENCE` references, and the
    firmware-blob start addresses all live in `vb1940_cam.cpp` at file scope — only the
    public-surface declarations and the `Vb1940_Mode` / `CalibrationData` types are
    visible in the header. Internal methods (`get_register`, `set_register_8`,
    `set_register_buffer`, `do_secure_boot`, `write_certificate`, `write_fw`,
    `write_vt_patch`, `read_eeprom_page`, `parse_calibration_data`,
    `read_calibration_from_eeprom`, etc.) are private members of `Vb1940Cam`. Dead
    legacy methods that no caller invoked (`set_register_16`, `set_register_32`,
    `get_eeprom_register`, `set_eeprom_register`, `set_eeprom_page`) are dropped rather
    than ported — the V1 driver can add them back if a real consumer needs them. The
    legacy `<hololink/sensors/camera/vb1940/vb1940_mode.hpp>` is included only by the
    `.cpp`; the register / firmware data is reached through
    `::hololink::sensors::vb1940_mode::*` qualified names so no aliases live in the
    header.
  - `csi.hpp` — adapter-side `PixelFormat` / `BayerFormat` enums whose integer values
    are byte-compatible with `hololink::csi`, mirroring `imx274/csi.hpp`. Each sensor
    driver carries its own copy so the per-driver header set stays self-contained.
  - `CMakeLists.txt` — STATIC target `hololink_adapter_sensors_vb1940`, alias
    `hololink::adapter::sensors::vb1940`. Public deps: `hololink::adapter_headers` and
    `hololink::leopard_vb1940::headers` (the latter is **PUBLIC** because
    `vb1940_cam.hpp` includes the supplement's public header for the
    `LeopardVb1940InterfaceV1` shared_ptr member, so the include path must propagate to
    every consumer — the examples and any future tests). Private deps: `hololink::core`
    and `hololink::sensors::native_vb1940_camera_sensor` (for the firmware-blob arrays +
    register tables — no public-API leakage). Same hidden-visibility + PIC properties
    `imx274/CMakeLists.txt` sets.
  - `hololink_adapter/host/sensors/CMakeLists.txt` gains `add_subdirectory(vb1940)`.
  - `hololink_adapter/CMakeLists.txt` — `add_subdirectory` order updated so every
    `module/*` directory is processed before `host/sensors`, since
    `hololink::leopard_vb1940::headers` is defined in `module/leopard_vb1940/` and
    consumed by the sensor target. The framework target in `host/` still comes first
    (everything below depends on `hololink::adapter_headers`).

- **New example `examples/module_vb1940_player.cpp`** — single-camera, mirrors
  `examples/module_imx274_player.cpp` shape:

  - CLI flags: `--hololink <ip>` (default
    `hololink::env_hololink_ip(0, "192.168.0.2")`), `--camera-mode <int>` (default
    matches the existing legacy `examples/vb1940_player.cpp` default), `--module-dir`,
    `--frame-limit`, `--headless`, `--fullscreen`, `--discovery-timeout`, `--log-level`.
    The IMX274-specific `--expander-configuration` and `--pattern` flags are dropped (no
    equivalent on Leopard).
  - `HoloscanApplication::compose()` wires the same pipeline shape: `RoceReceiverOp` →
    `CsiToBayerOp` → `ImageProcessorOp` → `BayerDemosaicOp` → `HolovizOp`.
    `device_start` / `device_stop` callbacks on the receiver invoke `camera_->start()` /
    `camera_->stop()`.
  - `main()` follows the same flow as `module_imx274_player.cpp`: CUDA init,
    `Adapter::get_adapter()`, optional module-dir override,
    `adapter.wait_for_channel(hololink_ip, discovery_timeout)`, construct
    `Vb1940Cam(metadata)`, fetch `HololinkInterface::get_service(metadata)`,
    `hololink->start()` + `hololink->reset()`, `camera->configure(camera_mode)`, build +
    run app, `hololink->stop()`.
  - `examples/CMakeLists.txt` — new target `module_vb1940_player` registered the same
    way `module_imx274_player` is. Links: `hololink`, `hololink::adapter`,
    `hololink::leopard_vb1940::headers`, `hololink::operators::csi_to_bayer`,
    `hololink::operators::image_processor`, `hololink::operators`,
    `hololink::adapter::sensors::vb1940`,
    `hololink::sensors::native_vb1940_camera_sensor`, Holoscan libs. Appended to
    `EXAMPLE_INSTALL_FILES`.
  - `tests/CMakeLists.txt` — new `module_vb1940_player_test` registered alongside
    `module_imx274_player_test`. Runs `module_vb1940_player --frame-limit=10 --headless`
    with `LABELS "vb1940"`; skips at runtime unless `HOLOLINK_TEST_VB1940=1` is set
    (`SKIP_RETURN_CODE 77` so ctest reports SKIPPED otherwise).

- **New example `examples/module_stereo_vb1940_player.cpp`** — 2-camera VB1940-AIO
  player. Single `wait_for_channel(ip)` (one HSB hosts both data planes);
  `Vb1940Cam::use_sensor(base_metadata, 0/1)` produces a per-sensor metadata clone with
  `sensor_number` stamped and the supplement's per-sensor address fields (`sensor`,
  `vp_mask`, `sif_address`, `vp_address`, `hif_address`, `frame_end_event`) re-stamped
  by the supplement's `update_metadata` (invoked through
  `EnumerationInterfaceV1::get_service(module)->update_metadata(clone)`). The
  per-channel I2C bus is passed to `Vb1940Cam` explicitly (`i2c_bus=1` for sensor 0,
  `i2c_bus=2` for sensor 1; Leopard wires the two cameras onto consecutive buses,
  mirroring `CAM_I2C_BUS + sensor_number` in the legacy strategy). Both channels share
  one `HololinkInterface` (same serial); `hololink->start()` / `reset()` runs once for
  the board. `HoloscanStereoApplication` composes two parallel
  `RoceReceiverOp → CsiToBayerOp → ImageProcessorOp → BayerDemosaicOp` legs into a
  single `HolovizOp` with the standard left/right side-by-side view (`offset_x` 0 and
  0.5, `width` 0.5 each).

  - **No application-side board configuration.** The example body is
    `wait_for_channel → use_sensor → construct Vb1940Cam per channel → hololink start/reset → camera->configure(mode) per channel → app->run()`.
    Every register-level write that brings the board up (clock-chip programming, sensor
    reset/release) lives behind `Vb1940Cam::configure` / `start` and the
    `LeopardVb1940InterfaceV1` primitives — the example never names a FPGA address.
  - `tests/CMakeLists.txt` registers `module_stereo_vb1940_player_test` under the same
    `HOLOLINK_TEST_VB1940=1` gate and `vb1940` label as the single-camera test.
    `examples/CMakeLists.txt` adds the target alongside `module_vb1940_player` (same
    link set).

- **New `Vb1940Cam::use_sensor(metadata, sensor_number)` static** — addition to the
  driver's public surface (otherwise deliberately minimal). Clones the base metadata,
  stamps `metadata["data_plane"] = sensor_number` (the per-channel locator
  discriminator), and calls the supplement's `EnumerationInterfaceV1::update_metadata`
  so the supplement re-derives every per-channel field (sensor, vp_mask, sif_address,
  vp_address, hif_address, frame_end_event) under its own formula. `data_plane` is the
  only metadata key the per-channel locators (`DataChannelInterfaceV1`,
  `RoceDataChannelInterfaceV1`, `RoceReceiverInterfaceV1`,
  `HololinkV1::roce_data_channel_instance_id`, `RoceDataChannelV1::data_plane()`)
  consult — the contract is "enumeration metadata always has data_plane", no fallback to
  anywhere else. Bridging logical channel index ↔ physical HIF index is the supplement's
  responsibility: `LeopardVb1940Publisher::update_metadata` reads the channel index from
  `metadata["data_plane"]` and hardcodes the physical `data_plane` parameter to 0 in its
  `set_data_plane_metadata` call. Single-camera `module_vb1940_player.cpp` is unaffected
  — it uses the as-enumerated metadata directly.

**Out of scope this phase.** Python sub-package (`hololink_adapter.sensors.vb1940`) and
Python example (`examples/module_vb1940_player.py`) — deferred to a follow-up extension
once the C++ end-to-end path is proven. `LeopardVb1940InterfaceV1`'s surface beyond
`expect_sensor` / `enable_sensor` (additional power-sequencing knobs, VCL enable on
`0x70` / `0x71`, VCL PWM on `0x21`, board-level diagnostics) is filled in as call sites
in `Vb1940Cam` or other consumers demand them — each follows the same idempotent
first-call-configures / cache-clears-on-reset contract.

**Done when.**

- `hololink_f1627640-b4dc-48af-a360-c55b09b3d230.so` builds and the adapter loader
  resolves it for a real Leopard VB1940 board's bootp announcement via the no-compat-id
  fallback path (bare-UUID exception — see Deliverable).
- `module_vb1940_player --frame-limit=10 --headless` runs end-to-end against a real
  Leopard VB1940 board: enumerates, loads the module, brings up the camera via
  `Vb1940Cam` (firmware load + clock + sensor configuration), drains 10 frames through
  the Holoscan pipeline.
- `ctest -R module_vb1940_player_test` reports SKIPPED (exit 77) without
  `HOLOLINK_TEST_VB1940=1`, PASSED with it set against reachable hardware.
- `g++ -std=c++17 -fno-rtti -fsyntax-only` is clean on the new sources.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD" -DHOLOLINK_ADAPTER_BUILD_OPERATORS=ON
cmake --build "$BUILD" --target leopard_vb1940
cmake --build "$BUILD" --target hololink_adapter_sensors_vb1940
cmake --build "$BUILD" --target module_vb1940_player
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
HOLOLINK_TEST_VB1940=1 \
    ctest --test-dir "$BUILD" -R module_vb1940_player_test --output-on-failure
```

### Phase 5 extension — `set_data_plane_metadata` takes explicit (sensor_number, data_plane) so supplements can map logical channel ↔ physical HIF

**Context.** Adding Leopard VB1940 stereo support surfaced an ambiguity in the
supplement's metadata-stamping path: HSB-Lite's 1:1 sensor↔data-plane mapping let the
single `metadata["data_plane"]` field stand in for both the per-channel SIF / VP
addressing inputs and the per-HIF address input. Leopard VB1940-AIO has two sensors
sharing one physical HIF, so the supplement needs separate inputs for "which logical
channel is this" (drives SIF / VP / vp_mask / frame_end_event) and "which physical data
plane sits behind it" (drives `hif_address`). Without that split, the second stereo
channel reads a non-existent HIF at `0x02010300` (`RESPONSE_INVALID_ADDR`).

**Deliverable.**

- `module/core/hsb_lite_publisher.hpp` — `HsbLitePublisher::set_data_plane_metadata`'s
  signature changes from `(metadata, total_sensors, total_dataplanes, sifs_per_sensor)`
  to `(metadata, int64_t sensor_number, int64_t data_plane, uint32_t sifs_per_sensor)`.
  `sensor_number` drives the SIF / VP / vp_mask / frame_end_event arithmetic;
  `data_plane` drives the HIF arithmetic. Each supplement's `update_metadata` override
  is responsible for translating its board's metadata into the two parameters.

  - `HsbLitePublisher::update_metadata` reads `metadata["data_plane"]` (the per-channel
    locator discriminator) and passes the same value as both `sensor_number` and
    `data_plane` — HSB-Lite has a 1:1 mapping. `sifs_per_sensor=2`.

  - `LeopardVb1940Publisher::update_metadata` (in
    `module/leopard_vb1940/module_entry.cpp`) reads `metadata["data_plane"]` as the
    `sensor_number` argument and hardcodes the `data_plane` argument to `0` — Leopard
    has one physical HIF shared by every sensor. `sifs_per_sensor=1`. It does **not**
    override `set_data_plane_metadata` itself; the canonical formula on
    `HsbLitePublisher` produces the right addresses once the parameters are right.

- The per-channel service locators (`DataChannelInterfaceV1`,
  `RoceDataChannelInterfaceV1`, `RoceReceiverInterfaceV1`,
  `HololinkV1::roce_data_channel_instance_id`, `RoceDataChannelV1::data_plane()`)
  continue to key on `metadata["data_plane"]` directly per the existing contract
  ("enumeration metadata always has data_plane"). The bootp parser stamps it; per-
  camera helpers like `Vb1940Cam::use_sensor` stamp it per channel so stereo channels
  resolve to distinct cached instances. No `sensor_number` metadata key; no fallback.
  The bridging from logical channel index (in metadata) to physical HIF index (in the
  `set_data_plane_metadata` arg) lives entirely in the supplement's `update_metadata`
  override.

- `host/include/hololink/adapter/oscillator.hpp:OscillatorInterfaceV1::locator_id` also
  continues to key on `metadata["data_plane"]`. On Leopard stereo the two channels'
  metadata each have a distinct `data_plane` (0 / 1) so the locator returns distinct
  `OscillatorImplV1` instances; both instances program the same physical Bajoran TS2
  chip on first `enable()` and idempotency-cache hits on subsequent calls, which is
  correct (idempotent writes to the same chip).

**Out of scope.** No change to the bootp parser, the host-side Adapter, or any
service-locator key shape. Backward compatibility for hand-coded
`"serial=...;data_plane=N"` test strings is preserved.

**Done when.**

- `g++ -std=c++17 -fno-rtti -fsyntax-only` clean on framework + supplements + sensor
  drivers.
- For HSB-Lite single-camera and quad-board tests: behaviour unchanged (1:1 mapping
  reduces to the prior single-`data_plane` call).
- For Leopard VB1940 stereo: `module_stereo_vb1940_player_test` against real hardware
  brings both cameras up and drains 10 frames per channel without
  `RESPONSE_INVALID_ADDR` on the right receiver.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD" -DHOLOLINK_ADAPTER_BUILD_OPERATORS=ON
cmake --build "$BUILD"
```

**Test-run commands.**

```bash
BUILD=/tmp/build-hololink
ctest --test-dir "$BUILD" --output-on-failure -LE hardware
HOLOLINK_TEST_VB1940=1 \
    ctest --test-dir "$BUILD" -L vb1940 --output-on-failure
```

### Phase 8 extension — framework support to complete the stereo IMX274 pattern test

**Context.** `tests/test_module_imx274_pattern.py` is the adapter-native port of the
legacy `tests/test_imx274_pattern.py` (stereo dual-IMX274 CRC pattern test). The port is
*incomplete*: five legacy capabilities have no adapter equivalent, so the test either
worked around them (MTU stamped directly onto metadata, per-leg metadata recording) or
skipped them (`test_imx274_multicast`, the reset-callback assertions). This extension
adds those five capabilities so the adapter test is a faithful equivalent — no
workarounds, nothing skipped. None of the V1 interfaces touched here have shipped, so
new methods/fields are added **directly to the existing V1 surface** (no `V2`).

**Deliverable.**

1. **`Adapter::use_mtu` Python binding.** The C++ path already exists
   (`Adapter::use_mtu` → `ChannelConfigurationInterfaceV1::use_mtu` →
   `HsbLiteChannelConfigurationV1::use_mtu` stamps `metadata["mtu"]`). Add the missing
   binding in `host/python/hololink_adapter_py.cpp` (`bind_adapter`), mirroring the
   `use_sensor` lambda (`py::arg("metadata"), py::arg("mtu")`). The test stops stamping
   `metadata["mtu"]` and calls `adapter.use_mtu(metadata, mtu)`.

1. **Multicast (`use_multicast`).** The RoCE data plane is already wired automatically:
   `RoceDataChannelV1::configure` (`module/core/roce_data_channel_default.hpp`) copies
   every `EnumerationMetadata` key into the legacy `hololink::DataChannel`, whose ctor
   reads `multicast` / `multicast_port` and whose `configure_roce` programs the FPGA. So
   stamping those two keys is sufficient — no adapter receiver change.

   - `host/include/hololink/adapter/channel_configuration.hpp` — add pure virtual to
     `ChannelConfigurationInterfaceV1` (after `use_mtu`):
     `virtual void use_multicast(EnumerationMetadata& metadata, std::string address, uint16_t port) = 0;`
   - `module/core/hsb_lite_publisher.hpp` — implement in
     `HsbLiteChannelConfigurationV1`, stamping `metadata["multicast"] = address;` and
     `metadata["multicast_port"] = static_cast<int64_t>(port);` (mirrors legacy
     `DataChannel::use_multicast`). Leopard inherits the base impl.
   - `host/include/hololink/adapter/adapter.hpp` + `host/src/adapter.cpp` —
     `Adapter::use_multicast` mirroring `use_mtu`.
   - `host/python/hololink_adapter_py.cpp` — bind `use_multicast` (`metadata`,
     `address`, `port`).
   - Test: replace the `pytest.skip(...)` in `test_imx274_multicast` with the real RoCE
     flow, parametrized on the legacy group addresses
     `("224.0.0.228", 4791, "224.0.0.229", 4791)`.

1. **`rename_metadata` on the receiver operators.** Adapter receivers emit hardcoded,
   unprefixed metadata keys (`RoceReceiverOp::compute` in
   `host/operators/roce_receiver_op.cpp`; `LinuxReceiverOp::compute` in
   `host/operators/linux_receiver_op.cpp`). Add a
   `holoscan::Parameter<std::function<std::string(const std::string&)>> rename_metadata_`
   to **both** ops (the adapter already binds `std::function` callbacks
   `device_start`/`device_stop` via `HOLOLINK_ADAPTER_YAML_CONVERTER_UNSUPPORTED` +
   `register_converter<T>()`, so a param — not a legacy-style setter — fits the
   convention). Cache the renamed key names in `start()` (default to identity) and use
   them when stamping. Expose the kwarg through the `PyRoceReceiverOp` /
   `PyLinuxReceiverOp` trampolines and `.def(py::init<...>)` lists in
   `host/operators/python/operators_py.cpp`, mirroring `device_start`/`device_stop`.
   Additionally, `RoceReceiverOp` now emits `imm_data` / `page_number` from `frame_info`
   (`RoceReceiverFrameInfoV1::imm_data` already exists), matching `LinuxReceiverOp`
   (`page_number = imm_data & 0xFFF`). The test passes
   `rename_metadata=lambda name: f"left_{name}"` / `f"right_{name}"` and restores the
   legacy prefixed-field metadata records.

1. **`on_reset` on `HololinkInterfaceV1`.** Add (directly to V1)
   `virtual std::shared_ptr<ResetRegistration> on_reset(std::function<void()> callback) = 0;`
   plus a nested RAII `ResetRegistration` handle to
   `host/include/hololink/adapter/hololink.hpp` (after `configure_hsb()`). `HololinkV1`
   (`module/core/hololink_default.hpp`) registers ONE aggregating
   `ResetCallbackController : public hololink::Hololink::ResetController` with the
   backing `Hololink` (whose `reset_controllers_` is append-only with no removal) and
   keeps the per-caller callbacks in its own id-keyed registry; the returned handle
   erases its entry on destruction. This is required because the per-board
   `HololinkInterfaceV1` is a process-lifetime singleton — forwarding each callback
   straight to the legacy list would accumulate registrations (and pin their owners) for
   the whole session. Bind it on the `PyHololinkInterface` trampoline
   (`PYBIND11_OVERRIDE_PURE`) and in `bind_hololink_interface` (the handle is an opaque
   `ResetRegistration` class). The test's `CameraWrapper(imx274.Imx274Cam)` registers a
   **weak** callback (so the registry doesn't pin the camera) and keeps the handle, so
   the registration is dropped when the camera is garbage-collected; it asserts the
   callback fires once per `hololink.reset()`.

1. **CRC operators in `hololink_adapter.operators`.** Port the self-contained legacy
   operators (`src/hololink/operators/compute_crc/compute_crc.{hpp,cpp}`) into the
   adapter tree as
   `host/operators/include/hololink/adapter/operators/compute_crc_op.hpp` +
   `host/operators/compute_crc_op.cpp` — namespace `hololink::adapter::operators`,
   adapter include guard/SPDX header, `hololink/adapter/logging.hpp` (`HSB_LOG_*`) in
   place of the legacy core logger, and `HOLOLINK_ADAPTER_YAML_CONVERTER_UNSUPPORTED` +
   `register_converter` for the `std::shared_ptr<ComputeCrcOp>` param on `CheckCrcOp`.
   `host/operators/CMakeLists.txt` adds a `HOLOLINK_ADAPTER_BUILD_CRC` option (default
   ON when nvcomp is found), a guarded
   `find_package(nvcomp REQUIRED PATHS /usr/lib/sbsa-linux-gnu/cmake/nvcomp)`, the new
   source, the nvcomp/CUDA link deps, and a `HOLOLINK_ADAPTER_BUILD_CRC` compile
   definition on `hololink_operators` and the `_hololink_adapter_operators` pybind
   module (gated like `HOLOLINK_BUILD_ROCE` / `HOLOLINK_BUILD_FUSA`). Add the
   `PyComputeCrcOp` / `PyCheckCrcOp` trampolines + bindings in `operators_py.cpp`. The
   test switches its CRC operators from `hololink.operators.*` to
   `hololink_adapter.operators.*` and runs CRC validation on both the RoCE and Linux
   paths.

`hololink_adapter/README.md` is updated in the same change (per repo convention): the
new `use_mtu` / `use_multicast` / `on_reset` surface, the `rename_metadata` receiver
param and RoCE `imm_data` emission, and the `hololink_adapter.operators` CRC operators +
`HOLOLINK_ADAPTER_BUILD_CRC` option.

**Out of scope.** No `V2` interfaces (V1 is unreleased). No edits to the legacy
`src/hololink` tree. No multicast receive-side group-join in the adapter (the RoCE FPGA
targets the group; parity with the legacy RoCE multicast test). No new metadata
service-locator key shapes.

**Done when.**

- `g++ -std=c++17 -fno-rtti -fsyntax-only` clean on framework + operators; the
  `_hololink_adapter` and `_hololink_adapter_operators` pybind modules build with
  `HOLOLINK_ADAPTER_BUILD_OPERATORS=ON` and `HOLOLINK_ADAPTER_BUILD_CRC=ON`.
- `python -m py_compile tests/test_module_imx274_pattern.py` passes; an import smoke
  test confirms `Adapter.use_mtu` / `Adapter.use_multicast`,
  `HololinkInterfaceV1.on_reset`, and
  `hololink_adapter.operators.{ComputeCrcOp,CheckCrcOp}` exist.
- On hardware (≥2 channel IPs, IMX274 + RoCE): `test_module_imx274_pattern.py` passes
  the stereo, various-MTU, multicast, and single-interface variants with the same
  recorded CRCs as the legacy `tests/test_imx274_pattern.py`; `test_imx274_multicast` is
  no longer skipped.

**Build commands.**

```bash
BUILD=/tmp/build-hololink
cmake -S . -B "$BUILD" -DHOLOLINK_ADAPTER_BUILD_OPERATORS=ON \
    -DHOLOLINK_BUILD_PYTHON=ON -DHOLOLINK_BUILD_ROCE=ON \
    -DHOLOLINK_ADAPTER_BUILD_CRC=ON
cmake --build "$BUILD"
```

**Test-run commands.**

```bash
python -m py_compile tests/test_module_imx274_pattern.py
pytest tests/test_module_imx274_pattern.py \
    --channel-ips <ip0> <ip1> --schedulers default -v
```

### Phase 8 extension — stereo frame alignment in the IMX274 pattern test

**Context.** IMX274 has no hardware frame-sync, so the two legs of
`tests/test_module_imx274_pattern.py` free-run and their frames drift apart over time.
The legacy stereo app (commit `b80c5ea2`) handled this with a pure-Python
`FrameAlignerOp` (`tests/operators.py`) that pairs the two streams frame-for-frame and
drops a group when the legs' timestamps differ by more than an allowable skew. This
extension brings that operator into the adapter test. The acceptance window is **one
full frame period** — the largest skew that still guarantees adjacent-frame pairing on a
non-hardware-synced sensor.

**Deliverable.**

1. **`FrameAlignerOp` (pure Python).** Copied verbatim from `b80c5ea2` into the shared
   `tests/operators.py` (it depends only on `holoscan`, `logging`, and the existing
   `SEC_PER_NS`). One `input` port (`QueuePolicy.POP`) fed by both legs; it tells the
   legs apart by tensor name (`input_tensors=["left","right"]`), reads each leg's
   `{prefix}timestamp_s` / `{prefix}timestamp_ns` metadata, and emits paired frames on
   per-leg `outputs` ports as the unnamed tensor.

1. **`out_tensor_name` on the receiver operators.** The adapter receivers emitted an
   unnamed tensor (`add<nvidia::gxf::Tensor>("")`), but the aligner needs named tensors
   to distinguish legs. Add a `holoscan::Parameter<std::string> out_tensor_name_`
   (default `""`) to **both** `RoceReceiverOp` and `LinuxReceiverOp`
   (`host/operators/include/hololink/adapter/operators/{roce,linux}_receiver_op.hpp` +
   `host/operators/{roce,linux}_receiver_op.cpp`), used in the `add<Tensor>(...)` call,
   and expose the kwarg through the `PyRoceReceiverOp` / `PyLinuxReceiverOp` trampolines
   and `.def(py::init<...>)` lists in `host/operators/python/operators_py.cpp` (placed
   right before `name`). This matches the existing `out_tensor_name` convention on
   `CsiToBayerOp` / `PackedFormatConverterOp` / `FusaCoeCaptureOp`; the `""` default
   preserves today's behavior for every other call site. The receivers already emit
   `timestamp_s` / `timestamp_ns` under the per-leg renamed keys, so no other receiver
   change is needed.

1. **Frame-ready async scheduling on the receivers (required for the join).** The
   adapter receivers block-polled `get_next_frame` (up to 1 s) in `compute()`; feeding
   two of them into one aligner input under the greedy scheduler deadlocks (a
   non-delivering leg blocks the single thread, so the aligner only ever sees one leg
   and never pairs). Port the legacy `BaseReceiverOp` frame-ready model to **both**
   `RoceReceiverOp` and `LinuxReceiverOp`: `setup()` always builds a
   `holoscan::AsynchronousCondition`; the monitor thread wakes it through a new
   `set_frame_ready(std::function<void()>)` on `RoceReceiverInterfaceV1` /
   `LinuxReceiverInterfaceV1` (forwarded to the legacy backing receiver in
   `module/core/{roce,linux}_receiver_default.hpp`). `compute()` re-arms the condition
   (`EVENT_WAITING`, then `EVENT_DONE` if `frames_ready()`); `start()` / `stop()` arm
   and retire it. `compute()` then never blocks, so multiple receivers feeding a join
   schedule correctly under any scheduler. (Unlike legacy there is no blocking-mode
   toggle — the blocking path only deadlocks joins for a negligible latency win.)

1. **Single-interface stereo data-plane fix.** Core
   `HsbLiteChannelConfigurationV1::use_sensor` (`module/core/hsb_lite_publisher.hpp`)
   stamped `hif_address` per sensor (`data_plane = sensor_number`), assuming 1:1
   sensor↔data-plane. hololink-lite multiplexes multiple sensors onto one data plane,
   differentiated by VP — matching legacy `BasicEnumerationStrategy::use_sensor`
   (`src/hololink/core/enumerator.cpp`), which re-stamps
   `vp_address`/`vp_mask`/`sif_address` but leaves `hif_address` alone. The old behavior
   OR'd sensor 1's `vp_mask` into the wrong data plane's `DP_VP_MASK`, so the FPGA never
   forwarded sensor 1 and the right leg of `test_imx274_stereo_single_interface` got no
   frames. Fixed by passing the enumerated `metadata["data_plane"]` (not
   `sensor_number`) to `hsb_lite_sensor_metadata`. (Two-IP `test_imx274_pattern` was
   unaffected — it enumerates each data plane directly.)

1. **Test wiring.** `tests/test_module_imx274_pattern.py` threads a per-case
   `allowable_dt` value through the parametrized result tuples (`expected_4k_results` /
   `expected_1080p_results`, all `1/60 s`), the shared `@pytest.mark.parametrize` arg
   string, every test signature, `run_test`, and `PatternTestApplication`.
   `_make_receiver` passes `out_tensor_name="left"`/`"right"`. The aligner is the first
   hop — both receivers feed it; it fans paired frames out to each leg's `ComputeCrcOp`
   — so CRC and everything downstream run only on aligned frames.

`hololink_adapter/README.md` is updated in the same change: the receiver
`out_tensor_name` parameter, the frame-ready async-condition scheduling, and the stereo
test's frame-alignment stage.

**Out of scope.** No `V2` interfaces. No edits to the legacy `src/hololink` tree. No
hardware-level synchronization (IMX274 doesn't support it) — alignment is purely
timestamp-based, software-side dropping.

**Done when.**

- The `_hololink_adapter_operators` pybind module builds with the new `out_tensor_name`
  receiver kwarg;
  `python -m py_compile tests/operators.py tests/test_module_imx274_pattern.py` passes.
- On hardware (≥2 channel IPs, IMX274 + RoCE): the stereo legs pair frame-for-frame at
  the record stage and the existing CRC assertions still pass.

### Phase 8 extension — data channels apply the application's enumeration metadata each resolution

**Context.** `test_imx274_stereo_single_interface` failed intermittently with `count=0`
(the right leg received nothing). Root cause, traced from `passed`/`failed` logs: the
per-`(serial,data_channel)` `RoceDataChannelInterfaceV1` is a process-lifetime
`ConfigurableService` whose `configure()` runs **once** (`std::call_once`), freezing the
metadata-derived host binding. An earlier test in the same process (the 2-IP
`test_imx274_pattern`, whose right leg uses the board's native `.3` enumeration)
configured `data_channel=1` first → the legacy `DataChannel` was pinned to `peer_ip=.3`,
so `configure_roce` programmed `DP_HOST_IP=.102`. The later single-interface test
resolves the same service with its `.2` clone, but the cached `.3` config wins, so the
FPGA writes to `.102` while the receiver's QP (resolved per-test from the `.2` metadata)
listens on `.101` → loss. Whichever metadata configures the singleton first wins for the
whole process → order-dependent, intermittent, cross-test. `local_ip_and_mac` is
deterministic; the QP is not at fault.

**Fix (transport-agnostic, at the service layer).** A data channel must use the
enumeration metadata the application hands it at `get_service(metadata)` on **every**
resolution, not just the first. Make `ConfigurableService::ensure_configured`
(`host/include/hololink/adapter/service.hpp`) **virtual**, broadening its contract from
"configure exactly once" to "ensure the configuration reflects this metadata" — the
default still runs `configure()` once via `call_once` (correct when a key's metadata is
stable). The data-channel impls (`RoceDataChannelV1`, `LinuxDataChannelV1`,
`CoeDataChannelV1` in `module/core/*_data_channel_default.hpp`) **override
`ensure_configured`** to re-run `configure(metadata)` (which rebuilds the backing legacy
`DataChannel`) whenever the supplied metadata differs from what they last built from
(cached in `applied_metadata_`); `configure()` itself is unchanged. No new method and no
`get_service` change — one virtual + three small overrides.

**Why this level.** Fixes RoCE, Linux, and CoE identically (not QP/RoCE-specific); makes
the channel deterministically reflect the application's metadata regardless of test
order; and retires this cross-test process-singleton-state-leak class for data channels
(same family as the `on_reset` accumulation). The anchor `DataChannelV1` is left
once-configured: only invariant fields (`serial_number`, `data_channel`) and the
serial-keyed hololink are read from it, so it never needs refreshing. No `src/hololink`
edits.

**Done when.** With the single-interface test run after a 2-IP test in one process, the
right data channel rebuilds from the `.2` metadata each resolution → `DP_HOST_IP=.101`,
matching the receiver → both legs receive, deterministically across test order; the
intermittent `count=0` is gone.

### Phase 8 extension — stereo IMX274 test-harness corrections (frame budget, watchdog, per-board coverage)

**Context.** Running the frame-aligned stereo test on hardware surfaced three defects in
the test harness rather than the data path: the aligner-equipped pipeline timed out at
`count=97`/100, and the failure was intermittently reported as `PASSED`. All fixes are
test-side (`tests/`); no adapter or `src/hololink` code is involved.

**Fixes.**

1. **Frame budget at the sink, not the source.** `frame_limit` was a `CountCondition` on
   both receivers. Because `FrameAlignerOp` drops frames between the receivers and the
   visualizer (intentional `dt > allowable_dt` drops + occasional `QueuePolicy.POP` loss
   on its shared input), the watchdog sink saw only `frame_limit − drops` taps; once the
   receivers hit their count and stopped, the watchdog waited for taps that never came
   and timed out. The budget now lives at the sink: receivers free-run on a
   `BooleanCondition`, `watchdog_operator` carries `CountCondition(frame_limit)`, and
   `WatchdogOp` (`tests/operators.py`) takes `frame_limit` + `stop_conditions` and
   disables the receivers' tick conditions after it has tapped `frame_limit` aligned
   frames. "100 frames" now means 100 *aligned* frames reaching the visualizer; the
   source over-produces to cover drops; the app drains cleanly. Loss-rate-independent,
   and it also closes a latent shortfall on the lossless RoCE path (the intentional
   `dt`-drops alone would otherwise leave the sink short).

1. **Watchdog timeouts fail the test deterministically.** `utils.Watchdog` (shared
   `tests/utils.py`) signalled failure via `assert False` in its `SIGUSR1` handler. When
   the timeout fires while the main thread is inside a GXF operator's `stop()`, GXF
   catches and discards the `AssertionError` (logged at `gxf_wrapper.cpp:291`), so
   `run()` returns and the stalled run is reported `PASSED` (a false pass; the SIGABRT
   escalation makes it inconsistent run-to-run). The watchdog now sets `_timed_out` in
   `_expired` and re-raises in `__exit__` (main thread, after `run()` returns, unless
   another exception is already propagating).

1. **Per-board single-interface coverage.** `test_imx274_stereo_single_interface` was
   reduced to `channel_ips[0]` only when ported from legacy. It now parametrizes
   `channel_index` `[0, 1]` (indices beyond `--channel-ips` `pytest.skip`), running the
   single-interface scenario once per board — matching legacy `test_imx274_pattern.py`'s
   `(ibv_index, ibv_name)` two-interface coverage. The adapter resolves the IB device
   from peer metadata, so only the channel index is parametrized.

1. **Aligner diagnostics.** `FrameAlignerOp` logs per-leg `arrivals`, `emitted`, and
   `dt_dropped` (on each `dt`-drop and every 30th emit), distinguishing a stall caused
   by starvation, alignment, or input-queue POP loss.

**Done when.**
`python -m py_compile tests/operators.py tests/test_module_imx274_pattern.py tests/utils.py`
passes; on hardware the aligned stereo test reaches `frame_limit` aligned frames at the
visualizer and exits without a watchdog timeout, and a genuine stall now fails the test
(rather than false-passing).

### Rename — `hololink_adapter` → `hololink_module`

**Context.** The top-level directory holding all code developed for this project,
`hololink_adapter/`, was named at the start of the project before the design settled.
The name is now considered incorrect: the design does not follow the adapter pattern, so
"adapter" is misleading. `hololink_module` is the better name and is the target for a
rename. The `hololink_adapter` identifier currently appears throughout the codebase and
this plan (~220 references) as the directory name, the `hololink::adapter` namespace,
library/target names, the `hololink_adapter` / `_hololink_adapter.so` pybind module, and
the `hololink_adapter_*` C-ABI entry points (`hololink_adapter_init`,
`hololink_adapter_get_abi_check`).

**Scope analysis.** Measured against the tracked tree (excluding `plans/` and the stray
working-dir junk). The string `hololink_adapter` is not one identifier but a family of
derived names, each with its own blast radius and its own rename decision:

| #   | Category                        | Form                                                                                              | Count                                      | Notes                                                                                                                                                                                                                                                                                                                                                         |
| --- | ------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Top-level directory             | `hololink_adapter/`                                                                               | 1 dir, 131 files                           | `git mv` to `hololink_module/`. Single source of truth for everything below.                                                                                                                                                                                                                                                                                  |
| 2   | Include-path component          | `#include <hololink/adapter/…>`                                                                   | ~440 lines                                 | The public include tree is `host/include/hololink/adapter/…` (+ `operators/`, `sensors/`). Renaming the namespace implies renaming this dir to `hololink/module/`.                                                                                                                                                                                            |
| 3   | C++ namespace                   | `hololink::adapter`                                                                               | 745                                        | Largest category. Note this is **distinct** from the `Adapter` host-singleton class — see open questions.                                                                                                                                                                                                                                                     |
| 4   | Macro / include-guard prefix    | `HOLOLINK_ADAPTER_*`                                                                              | 536                                        | Mixed: most are include guards (mechanically derivable from path), but real macros too — `HOLOLINK_ADAPTER_OK` (102), `HOLOLINK_ADAPTER_CUDA_CHECK` (84), status codes, `HOLOLINK_ADAPTER_BUILD_OPERATORS`/`_CRC` build options, `HOLOLINK_ADAPTER_EXPORT`, `HOLOLINK_ADAPTER_API_VERSION`, `HOLOLINK_ADAPTER_ABI_MAGIC`, `HOLOLINK_ADAPTER_TEST_MODULE_DIR`. |
| 5   | C-ABI entry points / types      | `hololink_adapter_init`, `hololink_adapter_get_abi_check`, `hololink_adapter_*_t`                 | ~46                                        | The `dlsym` symbols + their structs (`hololink_adapter_init_t`, `_module_services_t`, `_abi_check_t`, `_status_t`, `_service_t`, the `get_service`/`release_service` fn typedefs). Renaming is an ABI break — acceptable since interfaces are not frozen, but every module's `module_entry.cpp` and the host loader must change together.                     |
| 6   | Python import package           | `import hololink_adapter`                                                                         | 38                                         | Installed package `python/hololink_adapter/`; build copies to `${CMAKE_BINARY_DIR}/python/hololink_adapter/` and installs to `…/hololink_adapter`.                                                                                                                                                                                                            |
| 7   | pybind extension module         | `_hololink_adapter` / `_hololink_adapter.so`                                                      | 92                                         | Core ext `hololink_adapter_py.cpp` → `_hololink_adapter.so`; also `hololink_adapter.operators` + `_hololink_adapter_operators`, and per-sensor exts.                                                                                                                                                                                                          |
| 8   | CMake targets + alias namespace | `hololink_adapter_*` targets, `hololink::adapter*` aliases, `find_package(hololink_adapter)`      | ~65 target refs                            | Archives `hololink_adapter_headers`/`hololink::adapter_headers`, `hololink_adapter_module`/`hololink::adapter`, `hololink_adapter_operators`, sensor targets `hololink::adapter::sensors::{imx274,vb1940}`, the export/package name, plus ~25 test targets (`hololink_adapter_*_test`, `*_stub_module`).                                                      |
| 9   | File names                      | `hololink_adapter_*.{cpp,hpp,py}`                                                                 | ~20 test files + `hololink_adapter_py.cpp` | `tests/hololink_adapter_*` and `tests/test_hololink_adapter_*` need `git mv`.                                                                                                                                                                                                                                                                                 |
| 10  | README                          | `hololink_adapter/README.md`                                                                      | 192 KB                                     | Moves with the dir; must be re-swept for the string (\[[keep-adapter-readme-in-sync]\]).                                                                                                                                                                                                                                                                      |
| 11  | Build/install path strings      | `python/hololink_adapter/`, install `DESTINATION`, `HOLOLINK_PYTHON_INSTALL_DIR/hololink_adapter` | a handful                                  | In `hololink_adapter/CMakeLists.txt`.                                                                                                                                                                                                                                                                                                                         |

**Cross-tree fan-out (references outside `hololink_adapter/`).** The rename is not
self-contained — 28 files outside the directory reference it: top-level `CMakeLists.txt`
(3), `examples/` (10 files: every `module_*_player.{cpp,py}` plus `vsync_start_op.hpp`),
`tests/` (the C++/Python adapter tests + `CMakeLists.txt` with 87 refs, `conftest.py`,
`test_module_*` players), `python/setup.py` (24), and `docker/Dockerfile.demo`.

**Naming collisions / decisions to resolve (open questions).**

1. **Target-name collisions — RESOLVED (see "Resolved naming scheme" below).** A blind
   `hololink_adapter` → `hololink_module` substitution breaks the CMake target graph in
   two ways. `module_module` is explicitly **not acceptable**, so the rename needs a
   deliberate target-naming scheme rather than a sed pass. Existing target landscape:

   | Existing target                                          | Role                                                                                                       | Naive result                           |
   | -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | -------------------------------------- |
   | `hololink_adapter_headers` (`hololink::adapter_headers`) | INTERFACE public headers                                                                                   | `hololink_module_headers` ✓            |
   | `hololink_adapter_module` (`hololink::adapter_module`)   | module-side framework glue (Module + Publisher + `abi_check` + logging cache, absorbed by each module .so) | `hololink_module_module` ✗ stutter     |
   | `hololink_adapter` (`hololink::adapter`)                 | host-side adapter impl archive                                                                             | `hololink_module` ✗ **hard collision** |
   | `hololink_module` (`hololink::module`)                   | module/core V1-wrapper archive — **already named this** (`module/core/CMakeLists.txt:13`)                  | —                                      |

   The exact `module_module` sites (three independent families):

   - **A. CMake target `adapter_module`** — `host/CMakeLists.txt:60` (`add_library`),
     `:63` (alias), `:79` (`EXPORT_NAME adapter_module`), `:64/65/68/78/80/88/142`
     (uses); `module/core/CMakeLists.txt:25` and `cmake/HololinkModule.cmake:100` (link
     uses); comments at `HololinkModule.cmake:31`, `host/CMakeLists.txt:57/159`,
     `tests/CMakeLists.txt:116`.
   - **B. C-ABI struct `hololink_adapter_module_services_t`** (the `init` return
     payload) — 21 occurrences in 13 files: defn in `service_locator.h` (×3, incl. the
     `struct hololink_adapter_module_services` tag), `publisher.hpp` (×2),
     `src/adapter.cpp` (×2), `src_module/module_base.cpp` (×4),
     `module/core/hsb_lite_publisher.hpp`, all three `module/**/module_entry.cpp`, and 5
     `tests/*_module.cpp` stubs.
   - **C. Test target `hololink_adapter_module_core_test`** —
     `tests/CMakeLists.txt:217/219/223` (+ 3 README refs).

   The former **hard collision (D)**: the host archive `hololink_adapter` wants to
   become `hololink_module`, which was originally the module/core archive's name — CMake
   would error on the duplicate. **Resolved by renaming the *other* side of the
   collision:** the module/core archive became `hololink_module_core` /
   `hololink::module_core` and the glue archive became `hololink_module_runtime`, so no
   bare `hololink_module` target remains to collide with. The host archive is therefore
   renamed to `hololink_module` / `hololink::module` (EXPORT `module`) like every other
   target — it no longer needs to keep the `adapter` name. (It still builds the
   `Adapter` host-singleton class, whose *type* name is kept per Q2.) See the "Resolved
   naming scheme" table below.

1. **Namespace vs. the `Adapter` class — RESOLVED: keep the `Adapter` host-singleton.**
   `hololink::adapter` (the project namespace) is separate from the `Adapter`
   host-singleton class (`Adapter::get_adapter()`, `host/src/adapter.cpp`,
   `host/include/hololink/adapter/adapter.hpp`, `tools/adapter_enumerator.cpp`).
   Decision: the rename touches the project/directory/namespace, **but the `Adapter`
   class keeps its name** (and `Adapter::get_adapter()` is unchanged). Consequences:

   - The class becomes `hololink::module::Adapter` (namespace renamed, type name kept).
   - The basenames `adapter.hpp` / `adapter.cpp` / `adapter_enumerator.cpp` stay as-is;
     only their path component moves (`hololink/adapter/adapter.hpp` →
     `hololink/module/adapter.hpp`).
   - "Adapter" remains a legitimate term for the host singleton; only the
     *project/library* identity ("the hololink adapter") becomes "the hololink module".

1. **C-ABI break coordination — RESOLVED: no ABI-check change needed.** None of this
   interface has been released publicly, so there are no external modules to stay
   compatible with. The `dlsym` entry points and structs (category 5) are still renamed
   as part of the sweep, and the host loader + every `module_entry.cpp` change together
   in the same commit — but the `HOLOLINK_ADAPTER_ABI_MAGIC` value and the version check
   do **not** need to bump (only renamed to the new prefix). Host and modules are always
   built together from this tree, so no compatibility shim or deprecation window is
   required.

1. **Mechanical vs. semantic — RESOLVED by the scheme below.** Everything except the
   four hand-named cases (host archive kept; glue archive; module/core archive; the
   services struct) is a safe prefix substitution: `hololink_adapter` →
   `hololink_module`, `hololink::adapter` → `hololink::module`, `hololink/adapter/` →
   `hololink/module/`, `HOLOLINK_ADAPTER_` → `HOLOLINK_MODULE_`, `hololink_adapter_` →
   `hololink_module_`. The four exceptions are applied by hand (or as targeted, ordered
   replacements run *before* the blanket prefix pass so they don't first turn into the
   stutter).

**Rough magnitude.** ~830 textual occurrences across ~105 tracked files, ~131 files
moved by directory rename, ~20 additional files renamed, spanning C++, Python, CMake,
and Docker. This is a mechanical-but-wide change best done as a scripted sweep with the
collision cases handled by hand, then a full clean build + test pass.

### Resolved naming scheme

All four open questions are resolved. The project, directory, C++ namespace, include
path, macro/ABI prefixes, Python package, and pybind extension move from `adapter` →
`module`; the `Adapter` host-singleton class is kept; and four names are hand-chosen to
avoid the `module_module` stutter and the `hololink_module` target collision.

**Global (safe prefix) substitutions:**

| Category                      | Old                                                         | New                                                       |
| ----------------------------- | ----------------------------------------------------------- | --------------------------------------------------------- |
| Top-level directory           | `hololink_adapter/`                                         | `hololink_module/`                                        |
| C++ namespace                 | `hololink::adapter` (incl. `…::sensors::*`)                 | `hololink::module`                                        |
| Host-singleton class          | `hololink::adapter::Adapter`                                | `hololink::module::Adapter` (type name kept)              |
| Public include path           | `<hololink/adapter/…>`                                      | `<hololink/module/…>`                                     |
| Macro / include-guard prefix  | `HOLOLINK_ADAPTER_*`                                        | `HOLOLINK_MODULE_*` (`…_ABI_MAGIC` value unchanged — §3)  |
| C-ABI symbol prefix           | `hololink_adapter_*`                                        | `hololink_module_*`                                       |
| — init entry point            | `hololink_adapter_init`                                     | `hololink_module_init`                                    |
| — abi-check entry point       | `hololink_adapter_get_abi_check`                            | `hololink_module_get_abi_check`                           |
| Python package                | `import hololink_adapter`                                   | `import hololink_module`                                  |
| pybind core extension         | `_hololink_adapter[.so]` (`hololink_adapter_py.cpp`)        | `_hololink_module[.so]` (`hololink_module_py.cpp`)        |
| pybind operators ext / subpkg | `_hololink_adapter_operators`, `hololink_adapter.operators` | `_hololink_module_operators`, `hololink_module.operators` |
| Install component             | `hololink-adapter`                                          | `hololink-module`                                         |

**Hand-chosen names (the four resolved collisions):**

| What                               | Old                                                                                 | New                                                                              | Q        |
| ---------------------------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | -------- |
| Host-side impl archive             | `hololink_adapter` / `hololink::adapter` (EXPORT `adapter`)                         | `hololink_module` / `hololink::module` (EXPORT `module`)                         | D2       |
| Module-side framework-glue archive | `hololink_adapter_module` / `hololink::adapter_module` (EXPORT `adapter_module`)    | `hololink_module_runtime` / `hololink::module_runtime` (EXPORT `module_runtime`) | A2       |
| Module/core V1-wrapper archive     | `hololink_module` / `hololink::module`                                              | `hololink_module_core` / `hololink::module_core`                                 | core     |
| C-ABI services struct              | `hololink_adapter_module_services_t` (tag `hololink_adapter_module_services`)       | `hololink_module_services_t`                                                     | B1       |
| Public-headers archive             | `hololink_adapter_headers` / `hololink::adapter_headers` (EXPORT `adapter_headers`) | `hololink_module_headers` / `hololink::module_headers` (EXPORT `module_headers`) | (prefix) |
| Operators archive                  | `hololink_adapter_operators`                                                        | `hololink_module_operators`                                                      | (prefix) |
| Sensor targets                     | `hololink::adapter::sensors::{imx274,vb1940}`                                       | `hololink::module::sensors::{imx274,vb1940}`                                     | (prefix) |
| module/core test target            | `hololink_adapter_module_core_test`                                                 | `hololink_module_core_test`                                                      | C1       |
| Other test / stub targets          | `hololink_adapter_*_test`, `*_stub_module`                                          | `hololink_module_*_test`, `*_stub_module`                                        | (prefix) |

**Resolved to full consistency (supersedes the earlier D2 "kept" trade).** The host impl
archive is renamed to `hololink_module` / `hololink::module` (EXPORT `module`), matching
its siblings `hololink::module_headers`, `hololink::module_runtime`,
`hololink::module_core`, and `hololink::module::sensors::*`, the surrounding C++
namespace `hololink::module`, and the `hololink_module_*` C-ABI entry points. The name
was free once the module/core archive moved to `hololink_module_core`, so keeping
"adapter" bought nothing; every target now shares the `module` identity. The `Adapter`
host-singleton *class* it builds keeps its type name (Q2).

**Application-facing impact.** Consumers change `import hololink_adapter` →
`import hololink_module`, `#include <hololink/adapter/…>` → `<hololink/module/…>`, and
`target_link_libraries(app hololink::adapter)` → `hololink::module` for the host
library, switching to `hololink::module_core` / `hololink::module::sensors::*` for those
pieces.

### Execution plan

Done on branch `pogrady/hololink_module_name` from the repo root, as a **single commit**
(the C-ABI symbol rename means the host loader, every `module_entry.cpp`, and the test
stubs must change together — §3 — and the README moves with the code,
\[[keep-adapter-readme-in-sync]\]). The stray untracked working-dir files (`log*`,
`*.out`, `failed*`, `id_rsa*`, …) are left alone. Steps are ordered so the hand-named
cases are rewritten *before* the blanket prefix pass — otherwise they first decay into
the `module_module` stutter or the `hololink_module` duplicate.

**Step 0 — Legacy `hololink_module` alias clash (RESOLVED).** The legacy `hololink`
package is aliased as `hololink_module` (`import hololink as hololink_module`) in ~150
files (`examples/`, `tests/`, `ci/`, `docs/`). Renaming our Python package to
`hololink_module` shadows that alias in any file importing both. Only **two** of our
files import both: `tests/conftest.py` and `tests/test_module_imx274_pattern.py` (no
install-level conflict — legacy installs as `hololink`, ours as `hololink_module`; the
clash is purely in-file name shadowing). Resolution, applied *before* the rename passes
so no `hololink_module` legacy token survives to collide:

- `tests/conftest.py` — legacy import re-aliased: `import hololink as hololink_module` →
  `import hololink as hololink_legacy` (and its 5 uses → `hololink_legacy.*`).
- `tests/test_module_imx274_pattern.py` — legacy import dropped entirely; its only use
  was `hololink_module.DEFAULT_MTU` (the import comment's "IB device list" use was
  already stale). Rather than inline a literal, `DEFAULT_MTU` is now **adapter-owned**,
  in the same manner as `local_ip_and_mac`/`MacAddress`: a
  `constexpr uint32_t DEFAULT_MTU = 1500` in
  `host/include/hololink/module/networking.hpp` (`hololink::module`), exposed via the
  adapter pybind (`m.attr("DEFAULT_MTU") = ad::DEFAULT_MTU`) and re-exported from
  `host/python/__init__.py`, so the test uses `hololink_module.DEFAULT_MTU` (no legacy
  dependency, no magic number). README networking.hpp / pybind bullets updated to match
  (\[[keep-adapter-readme-in-sync]\]).

All other ~148 files keep their `import hololink as hololink_module` alias untouched —
they never import our package, so the rename passes (which only match
`hololink_adapter*` tokens) leave them alone, and the module/core target rename
(`hololink_module` → `hololink_module_core`) is scoped to CMake files only.

**Step 1 — Directory and file moves (`git mv`, so history follows).**

- `git mv hololink_adapter hololink_module`.
- Include-path component, every tree: `…/include/hololink/adapter/` →
  `…/include/hololink/module/` (host public headers,
  `host/operators/include/hololink/adapter/operators/`,
  `host/sensors/*/include/hololink/adapter/sensors/…`, and the per-board
  `module/*/include/hololink/adapter/<board>/`).
- `host/python/hololink_adapter_py.cpp` → `host/python/hololink_module_py.cpp`.
- `tests/hololink_adapter_*.{cpp,hpp}` → `tests/hololink_module_*`;
  `tests/test_hololink_adapter_*.py` → `tests/test_hololink_module_*.py` (and rename
  `tests/hololink_adapter_module_core_test.cpp` →
  `tests/hololink_module_core_test.cpp`).

**Step 2 — Ordered token replacements (most-specific first).** Applied across tracked
text files excluding `plans/`. Run the pre-pass before the blanket pass.

*Pre-pass (frees/renames the collision names while their tokens are still unambiguous):*

1. **module/core archive** — `hololink::module` → `hololink::module_core`; bare
   `hololink_module` (word boundary) → `hololink_module_core`. Safe to do first: today
   these tokens refer *only* to the module/core target (the C++ namespace is still
   `adapter` at this point).
1. **services struct (B)** — `hololink_adapter_module_services` →
   `hololink_module_services` (covers the tag and the `_t`).
1. **module/core test (C)** — `hololink_adapter_module_core_test` →
   `hololink_module_core_test`.
1. **glue archive (A)** — `hololink_adapter_module` → `hololink_module_runtime`,
   `hololink::adapter_module` → `hololink::module_runtime`, and the bare
   `EXPORT_NAME adapter_module` / comment token `adapter_module` → `module_runtime`.
   (Steps 2–3 ran first so these `hololink_adapter_module…` longer tokens are already
   gone.)

*Blanket prefix pass (file-type aware — this is where D2 lives):* 5. **C++ / Python /
headers / prose** (`*.cpp *.cc *.hpp *.h *.py *.md`): `hololink::adapter` →
`hololink::module`; `hololink/adapter/` → `hololink/module/`; `HOLOLINK_ADAPTER_` →
`HOLOLINK_MODULE_`; `hololink_adapter_` → `hololink_module_`; `_hololink_adapter` →
`_hololink_module`; remaining bare `hololink_adapter` (package / dir / pybind refs) →
`hololink_module`. (The host-archive *target* token never appears in these files.) 6.
**CMake** (`CMakeLists.txt *.cmake`): substitute only the **suffixed** forms —
`hololink_adapter_` → `hololink_module_`, `hololink::adapter_` → `hololink::module_`,
`hololink::adapter::` → `hololink::module::`, `hololink-adapter` → `hololink-module`,
and path strings `python/hololink_adapter` → `python/hololink_module`. The host-archive
tokens are now renamed like everything else: `add_library(hololink_adapter …)` →
`hololink_module`, the bare alias `hololink::adapter` → `hololink::module`,
`EXPORT_NAME adapter` → `module`, and every `target_*(hololink_adapter …)` / link use
(examples, tests, operators, tools) → `hololink_module` / `hololink::module`. Bare
`hololink_adapter` in *prose comments* (e.g. "the hololink_adapter tree") is fixed by
hand (Step 3).

Note: `HOLOLINK_ADAPTER_ABI_MAGIC` → `HOLOLINK_MODULE_ABI_MAGIC` is a name-only change
(value unchanged, §3). The user/CI-facing cache options
`HOLOLINK_ADAPTER_BUILD_OPERATORS` and `HOLOLINK_ADAPTER_BUILD_CRC` become
`HOLOLINK_MODULE_BUILD_*` — any `-D…` flags, CI scripts, or `docker/Dockerfile.demo`
invocations must update too.

**Step 3 — Manual fix-ups and review.**

- CMake bare-token disambiguation: host-archive target renamed to `hololink_module`;
  prose/dir comments updated.
- `EXPORT_NAME` edits: `adapter_headers` → `module_headers`, `adapter_module` →
  `module_runtime`, and the host archive's `EXPORT_NAME adapter` → `module`. Verify the
  `HololinkTargets` export set / `HololinkConfig.cmake` consumers still resolve.
- Cross-tree references (outside the renamed dir): top-level `CMakeLists.txt`,
  `examples/*` (`module_*_player.{cpp,py}`, `vsync_start_op.hpp`),
  `tests/CMakeLists.txt`, `tests/conftest.py`, the `tests/test_module_*` players,
  `python/setup.py`, `docker/Dockerfile.demo`.
- README: re-sweep `hololink_module/README.md` for any residual `hololink_adapter`
  (\[[keep-adapter-readme-in-sync]\]).

**Step 4 — Verification gate.**

- `git grep -n 'module_module'` → empty.
- `git grep -n 'hololink_adapter'` (and `hololink::adapter`) returns **nothing** in
  tracked code — the host archive is now `hololink_module` / `hololink::module` like
  every other target. Only `plans/` prose may still reference the old name historically.
- Clean configure + build (`cmake -B build …; cmake --build build`) with the renamed
  `-DHOLOLINK_MODULE_BUILD_OPERATORS=ON` (and RoCE per environment).
- `ctest --test-dir build` full suite green; `python -c "import hololink_module"` and
  the `*_python_smoke_test`s pass; `python -m py_compile` the touched
  `examples/`/`tests/` `.py`.
- Hardware-only tests (IMX274 stereo, VB1940) are out of band here — note they must be
  re-run on a rig before merge.

**Step 5 — Commit.** One commit titled for the rename; every target, including the host
archive, moves to the `hololink_module` / `hololink::module` identity (no kept
exception).

### Execution status (in progress)

The identifier rename has been applied and statically verified; not yet committed.

**Done and verified:**

- Steps 0–2 complete: directory + include-path + file `git mv`s; legacy-alias clash
  fixed (conftest → `hololink_legacy`; pattern test inlines `DEFAULT_MTU = 1500`);
  pre-passes (services struct → `hololink_module_services_t`; module/core →
  `hololink_module_core`; glue → `hololink_module_runtime`) and the file-type-aware
  blanket passes.
- A **5th stutter** not caught in analysis surfaced and was fixed: the status code
  `HOLOLINK_ADAPTER_MODULE_INIT_FAILED` → `HOLOLINK_MODULE_INIT_FAILED` (dropped the
  redundant inner word, like the services struct), updated in `status.h` and the
  abi-mismatch test.
- A test filename picked up the stutter during the bulk `git mv`
  (`hololink_module_module_core_test.cpp`) and was corrected to
  `hololink_module_core_test.cpp` to match CMake.
- EXPORT_NAMEs realigned: `module_headers`, `module_runtime`, `module::sensors::*`, and
  the host archive `module` (see follow-up below).
- **Follow-up (host archive no longer an exception):** the host impl archive was renamed
  from `hololink_adapter` / `hololink::adapter` (EXPORT `adapter`) to `hololink_module`
  / `hololink::module` (EXPORT `module`), with all link consumers updated —
  `hololink_module/host/CMakeLists.txt`, `examples/CMakeLists.txt`,
  `tests/CMakeLists.txt`, `hololink_module/host/operators/CMakeLists.txt`,
  `hololink_module/tools/CMakeLists.txt`, `hololink_module/module/core/CMakeLists.txt`,
  and the README. This was safe because the module/core archive had already moved to
  `hololink_module_core`, freeing the `hololink_module` target name.
- Verification: no lowercase `module_module`; no residual `hololink_adapter`,
  `hololink::adapter`, `hololink/adapter`, `HOLOLINK_ADAPTER`, or `EXPORT_NAME adapter`
  anywhere in tracked code; pybind module names, CMake link graph, and all `.cpp` source
  refs resolve; all 78 touched Python files `py_compile` clean.

**Could not run here:** the C++ configure + build + `ctest` gate — this sandbox has no
cmake/g++/CUDA/Holoscan. Must run on a build host (alongside the hardware tests).

**Open judgment calls (not mechanical bugs) — awaiting decision:**

1. **Include-guard stutter** `HOLOLINK_MODULE_MODULE_*` — RESOLVED: deduped. All 57
   occurrences across 19 module-side headers collapsed via `HOLOLINK_MODULE_MODULE_` →
   `HOLOLINK_MODULE_` (so `…MODULE_CORE_*` → `…CORE_*`). No `MODULE_MODULE` (any case)
   remains anywhere; all guards verified unique except one pre-existing duplicate noted
   below.
   - `module.hpp`'s guard is `HOLOLINK_MODULE_BASE_HPP` (two-segment, matching
     `module_base.cpp` which implements the `Module` base class) rather than the bare
     project-root `HOLOLINK_MODULE_HPP` — deduped (no stutter), unique, and consistent
     with the sibling two-segment guards.
   - **Build "redefinition of class Module" errors (`log`) were NOT a source bug — stale
     PCH from sccache.** A Docker build (`log`) showed only `module.hpp` redefining
     `class Module`/`LoadedModule` under the precompiled header. Investigation: the
     source guard is correct, unique, and well-formed (one `module.hpp`, standard PCH
     pattern), and renaming the guard repeatedly changed nothing — proving the compiler
     was reading a **cached `.gch`**, not the current source. The build's
     `SCCACHE_DIR=/sccache` is a persistent BuildKit cache mount, so a `.gch` compiled
     before the rename (old guard macro) is served back; the consumer's current-source
     re-include then redefines the class. Fix is environmental: rebuild with
     `--use-sccache=0`, or clear the `/sccache` cache mount
     (`docker builder prune --filter type=exec.cachemount`) — no source change.
   - **Pre-existing duplicate guard — FIXED:** the public
     `host/include/hololink/module/networking.hpp` and the private
     `host/src/networking.hpp` shared guard `HOLOLINK_MODULE_NETWORKING_HPP` (they
     shared `HOLOLINK_ADAPTER_NETWORKING_HPP` at HEAD too — the rename carried it
     forward). The private header's guard is now `HOLOLINK_MODULE_SRC_NETWORKING_HPP`;
     the public header keeps the canonical name. All include guards verified unique.
1. **Prose "adapter" as an English word** — README RESOLVED; code comments still
   pending.
   - **README done:** every framework-concept use of the word "adapter" reworded to
     "module" (~66 prose lowercase uses + the one capitalized "Adapter Python
     convenience bindings" → "Module …"). Preserved, by design: the `Adapter` class (47
     uses), the `adapter` Python-variable convention
     (`adapter = …Adapter.get_adapter()`, 27 uses), and the kept file identifiers
     (`adapter.cpp/.hpp`, `src/adapter`). (The host-archive target/namespace tokens
     `hololink_adapter` / `hololink::adapter` / `EXPORT_NAME adapter` were *not* kept —
     see the follow-up above; they are renamed to `hololink_module` / `hololink::module`
     / `module`.) No double-word artifacts; section headers reworded.
   - **Code comments/strings — DONE.** Framework-concept "adapter" reworded to "module"
     across `.cpp/.hpp/.h/.py` comments/docstrings/strings *and* CMake comments (incl.
     the `window_title="… (adapter)"` label → `(module)`, and capitalized framework uses
     like "Adapter port of" / "Adapter equivalent of" → "Module …"). A prose-only regex
     (`adapter` followed by space+letter / `-word` / `'` / EOL / backtick, with
     lookbehind excluding `:`/`/`/word/`(`/`& `) protected every code form. Verified
     intact: the `adapter` C++/Python variable
     (`auto& adapter = …Adapter::get_adapter()`, `adapter.method()`, params/args), the
     `Adapter` class, and the kept file identifiers (`src/adapter.cpp`, `adapter.hpp`).
     (The target/namespace tokens `hololink::adapter` / `hololink_adapter` /
     `EXPORT_NAME adapter` are renamed to the `module` forms, per the follow-up above.)
     No `an module`/`module module` artifacts; all touched Python `py_compile`s.

### Port `test_imx274_reconnect.py` → `test_module_imx274_reconnect.py`

**Status: DONE.** The port is complete and its source has been retired — both
`tests/test_imx274_reconnect.py` and the legacy reconnection implementation it exercised
(`hololink.hsb_controller` with `HsbController` / `SensorFactory` / `reset_device_map`,
`hololink.operators.HsbControllerOp`, and
`linux_controller_receiver.LinuxControllerReceiver` /
`roce_controller_receiver.RoceControllerReceiver`) have been removed, since
`tests/test_module_imx274_reconnect.py` and the `hololink_module` framework fully
replace them. The description below records what was ported *from* for historical
context.

**Goal.** Port the legacy device-reset/reconnect test `tests/test_imx274_reconnect.py`
to a new `tests/test_module_imx274_reconnect.py` built entirely on the `hololink_module`
V1 surface — the same treatment already applied to the pattern test, where
`tests/test_imx274_pattern.py` became `tests/test_module_imx274_pattern.py` (see the
"Phase 8 extension" sections above). The module pattern port is the reference for the
API shape and conventions to follow.

**Legacy surface that was replaced** (now removed). `tests/test_imx274_reconnect.py` was
written against the pre-module API and exercised FPGA reset while a capture pipeline
runs:

- **Discovery / control:** `hololink.hsb_controller.SensorFactory`
  (`Imx274SensorFactory` subclass), `HsbControllerOp`,
  `hsb_controller.reset_device_map()`, and the legacy
  `linux_controller_receiver.LinuxControllerReceiver` /
  `roce_controller_receiver.RoceControllerReceiver` (RoCE passes explicit
  `ibv_name`/`ibv_port`).
- **Reset mechanics under test:** `hololink.trigger_reset()` fired from a
  `Reactor.add_callback` after a chosen frame (`OnFrameNOperator`, `reset_after=[150]`),
  and `hololink.on_reset(cb)` re-arming shared board state (the `hololink_state` /
  `CLOCK_UP` / `UNKNOWN` clock-setup guard). A second scenario resets *during sensor
  configuration* via `InstrumentedImx274Cam` / `InstrumentedImx274CamContext`, which
  triggers the reset on the Nth `set_register` call (`set_registers_trigger=20`).
- **Validation / plumbing:** `CsiToBayerOp`, `BayerDemosaicOp`, `ComputeCrcOp` /
  `CheckCrcOp`, `StatusOp`, `operators.WatchdogOp`, `operators.RecordMetadataOp`,
  `CsiImage` fallback (SMPTE bars). CRCs are only checked on the lossless RoCE path.
- **Test matrix (8 tests):** mono + stereo × Linux + RoCE for the frame-triggered reset,
  plus reconnect-during-configuration variants (mono Linux, mono RoCE, stereo RoCE).

**Target surface (from the module pattern port).**
`Adapter.wait_for_channel(peer_ip, timeout)` for discovery; `EnumerationMetadata` clone
\+ `Adapter.use_sensor` for single-interface stereo; `Imx274Cam(metadata)` with
`Imx274Cam.use_expander_configuration`; `RoceReceiverOp` / `LinuxReceiverOp` (IB device
resolved from the peer — no `ibv_name`; `device_start`/`device_stop` arm the sensor);
`HololinkInterfaceV1.get_service(metadata)` with `.start()` / `.reset()` / `.stop()`;
`on_reset` registered as a weak method holding the RAII handle (the `CameraWrapper`
pattern); `FrameAlignerOp` + sink-owned frame budget for the stereo legs; module
`ComputeCrcOp` / `CheckCrcOp`.

**Design principle — invalidate on loss, re-fetch on rediscovery.** When a device is
lost (reset / disconnect), the objects that represent that device's state must be
**invalidated** — they no longer describe live module state and must not be used to
drive the device. When the device is **rediscovered**, the application is responsible
for **fetching fresh pointers** (re-resolving the module's services / handles for that
device) so that its view is back in sync with the module's state. The application does
not silently keep reusing stale handles across a loss/rediscovery cycle; the reconnect
test must exercise exactly this fetch-again-after-rediscovery path. This is the module
analogue of the legacy `on_reset` re-arm (which reset shared board state so the next
frame re-ran clock/sensor setup) — but expressed as invalidate-then-re-fetch of the
device-state objects rather than mutating a shared `hololink_state` dict in place.

**How loss happens and how it's detected.** Device loss is typically caused by a random
disconnection or by the distal device losing power — i.e. the loss is not a clean,
locally-initiated teardown but an abrupt disappearance of the peer. It is normally
*discovered* by a timeout: either a **data-plane timeout** (frames stop arriving at the
receiver) or a **timeout on a control-plane transaction** (a register read/write to the
device fails to complete). Either timeout is the trigger that invalidates the
device-state objects; rediscovery (a fresh enumeration/announcement from the peer once
it returns) is what lets the application re-fetch. The reconnect test's deliberate
`trigger_reset` stands in for this abrupt loss so the timeout → invalidate → rediscover
→ re-fetch path is exercised deterministically.

**Where the framework support lives — `hololink_module/host`.** The framework code
needed to *support* reconnection (loss detection via the data-plane / control-plane
timeouts above, invalidation of device-state objects, and the hooks that let an
application re-fetch on rediscovery) is expected to live under `hololink_module/host` —
the host-side framework tree — **not** in the per-board / per-device trees under
`hololink_module/module/*` (e.g. `hsb_lite`, `hsb_lite_2510`, `leopard_vb1940`,
`taurotech_da326`) or the sensor drivers. Reconnection is a generic capability of the
host framework; only the parts that are genuinely board- or sensor-specific belong in a
module.

Because of this split, **any framework code carried over from the legacy reconnect path
must be scrutinized for device-specific behavior before it lands in
`hololink_module/host`.** The legacy `Imx274SensorFactory` mixed generic reset re-arm
(the `on_reset` → invalidate shared state) with IMX274-specific setup (clock bring-up,
`configure`, `test_pattern`, expander selection); the port must keep the generic
reconnection machinery in the host framework and leave the sensor-specific work in the
`Imx274Cam` driver / module. When reviewing ported framework code, separate "what any
device needs on loss/rediscovery" (→ `hololink_module/host`) from "what this sensor or
board needs" (→ module / sensor driver).

#### Framework landscape (verified against `hololink_module/host` + `module/core`)

Three areas were mapped to establish what the module surface already provides and where
the gaps are. Findings, with the load-bearing references:

**Control plane — imperative reset works; there is no invalidation.**

- `HololinkInterfaceV1` (`host/include/hololink/module/hololink.hpp:54-237`, impl
  `module/core/hololink_default.{hpp,cpp}`) exposes `start()` / `stop()` / `reset()` /
  `configure_hsb()` / `ptp_synchronize()` and `on_reset(cb) → ResetRegistration` (RAII;
  registry-backed, aggregates onto the legacy append-only list — this is the
  `CameraWrapper` weak-method pattern). `reset()` is **blocking**: it forwards to legacy
  `Hololink::reset()` = `trigger_reset()` → `find_channel` (wait ≤30 s for re-announce)
  → `seed_arp` → `post_reset_configuration()` (`configure_hsb` + fire reset
  controllers).
- **There is no `trigger_reset` (fire-and-forget) and no `is_reset` / lost-state query
  on the V1 surface** — only the blocking `reset()` and the `on_reset` notification.
- Services resolved via `get_service(metadata)` are **process-lifetime singletons**: the
  module-side `Publisher::registry_` caches every service strongly, keyed by
  `(instance_id, type_id)` (serial-scoped), with `try_emplace`-only insertion and a
  no-op `release_service_thunk` — **nothing ever evicts**. `Adapter::modules_` (the
  `.so` handles) is likewise never evicted.
- `HololinkInterfaceV1` uses the base `ensure_configured` = `std::call_once`
  (`host/include/hololink/module/service.hpp:174-188`), so `configure(metadata)` runs
  **once ever**; a re-fetch after loss returns the same object with the original
  addressing. Only the data channels (`RoceDataChannelV1`, …) override
  `ensure_configured` to rebuild their backing `DataChannel` when the supplied metadata
  differs (`module/core/roce_data_channel_default.hpp` — the "apply metadata each
  resolution" extension above).
- The legacy tree has the missing piece: `Hololink::reset_framework()`
  (`src/hololink/core/hololink.cpp:322-331`) erases the serial-keyed singleton map so
  the next lookup re-materializes. **No module-surface equivalent exists.**

**Data plane — a stall is invisible today.**

- The module `RoceReceiverOp` / `LinuxReceiverOp` (`host/operators/*_receiver_op.cpp`)
  run an `AsynchronousCondition` + monitor-thread + `set_frame_ready` model (the
  frame-ready async scheduling from the stereo pattern extension). `compute()` calls
  `receiver_→get_next_frame(GET_NEXT_FRAME_TIMEOUT_MS=1000, …)`; **on timeout it does a
  bare `return`** — no counter, no deadline, no error, no signal.
- The RoCE monitor thread `poll()`s the completion channel with `timeout=-1` (blocks
  forever); ibverbs async events and non-success CQ completions that would indicate
  link/QP loss are **logged and dropped**
  (`src/hololink/operators/roce_receiver/ roce_receiver.cpp`). `FrameInfoV1` carries no
  error/loss field. The legacy `BaseReceiverOp::timeout()` at most logs once.
- `device_start` / `device_stop` (bound to `camera.start` / `camera.stop`) only arm /
  disarm the sensor at pipeline start/stop; they are not part of any loss detection.

**Discovery / rediscovery — the Adapter layer already re-delivers.**

- `Adapter` is a Meyers singleton with the bootp listener always running.
  `wait_for_channel(peer_ip, timeout)` holds **no cache** — it registers a one-shot
  `register_ip` closure and blocks until the *next* announcement, returning a fresh
  `EnumerationMetadata` **by value**. `register_ip` callbacks fire on **every** matching
  announcement, including re-announcements after a device returns.
- So rediscovery is already supported *at the Adapter layer*: after a loss, a new
  `wait_for_channel(peer_ip, …)` blocks and returns fresh metadata for the same peer.
  The only thing that does not refresh is the **service cache** above.

**Net gap for "invalidate on loss → re-fetch on rediscovery":** the three missing pieces
are (1) a **loss signal** — turn a data-plane stall (consecutive `get_next_frame`
timeouts / a deadline) and/or a failed control-plane transaction into an actionable
event, ideally also surfacing the ibverbs async/CQ errors that are currently dropped;
(2) an **invalidation path** — a module-surface analogue of `reset_framework()` that
evicts (or forces reconfigure of) the serial-keyed services in `Publisher::registry_`
(and possibly `Adapter::modules_`) so a re-fetch yields genuinely fresh, re-configured
device handles; and (3) the **application-visible re-fetch flow** — a notification (or
polled state) that tells the app "this device was lost/returned," after which it re-runs
`wait_for_channel` → `get_service(fresh_metadata)` → rebuild the affected handles
(camera, receiver op). All three belong in `hololink_module/host` per the split above.

#### Design forks (need a decision before writing the plan)

The gap can be closed several ways with materially different scope. Rather than guess,
the plan pauses here on these forks:

1. **How the test induces loss.** Legacy fired `hololink.trigger_reset()`
   (fire-and-forget) mid-capture from a reactor callback so the *pipeline* had to detect
   and recover. The module surface only has the blocking `reset()`. Either add a
   fire-and-forget `trigger_reset()` to `HololinkInterfaceV1` (faithful to the real
   "abrupt loss" the design principle targets), or drive the test with the blocking
   `reset()` (simpler, but the app initiates the recovery so it exercises less of the
   async loss path).

1. **What detects loss.** Data-plane timeout in the receiver op (matches the "frames
   stop arriving" case), control-plane transaction timeout (matches "register op
   fails"), or both. This decides where the new signal originates.

1. **Invalidate vs reconfigure.** Add a real eviction API to `Publisher` / `Module`
   (true `reset_framework()` analogue — services destroyed and rebuilt), or extend the
   existing `ensure_configured`-reconfigure-on-changed-metadata pattern to
   `HololinkInterfaceV1` and friends (lighter, but keeps object identity). This is the
   single biggest scope driver.

1. **Test coverage.** Which of the 8 legacy cases to port (mono/stereo × Linux/RoCE
   frame-triggered reset + the three reconnect-during-configuration variants), and
   whether the pattern-port harness corrections (sink-owned frame budget, watchdog
   re-raise, `FrameAlignerOp` for stereo) carry over.

#### Decisions

1. **Induce loss — add fire-and-forget `trigger_reset()`.** The test induces an *abrupt*
   loss (not an app-initiated clean `reset()`), so the pipeline must detect and recover
   on its own. Faithful to the "random disconnect / distal power loss" scenario.
1. **Invalidation — real eviction (a `reset_framework()` analogue).** On loss the
   serial-keyed device-state services are **destroyed/evicted** from the caches, so a
   re-fetch yields genuinely new objects. Matches the principle literally: "objects …
   invalidated" and "fetch **new pointers**."
1. **Detection — both data-plane and control-plane timeouts.** A receiver frame-stall
   deadline *and* a failed control-plane register transaction both signal loss.
1. **Coverage — all seven legacy cases.** Mono + stereo × Linux × RoCE frame-triggered
   reset, reusing the pattern-port harness (sink-owned frame budget, watchdog re-raise),
   plus the three reconnect-during-configuration variants (mono Linux, mono RoCE, stereo
   RoCE) — **all implemented**. The during-configuration cases map the "reset on the Nth
   `set_register`" instrumentation to the module surface via `InstrumentedImx274Cam`,
   whose `set_register` override fires `HsbLiteInterfaceV1.trigger_reset()`.
1. **Reconnection-loop ownership — a controller op wraps the receiver.** A new
   module-surface controller operator wraps the receiver and handles its reconnection
   needs (loss response, invalidate + re-fetch, fallback frame during downtime). The
   receiver op stays a "dumb" data-plane operator; the reconnection **policy** lives in
   the controller. The controller op wraps an `HsbController` built from a
   `sensor_factory` + `network_receiver`, owns the `frame_ready` async condition, and in
   `compute()` emits a **fallback SMPTE image** whenever `controller.connected()` is
   false so the pipeline keeps producing frames while the link is down. The pattern port
   wired the receiver op directly (no reconnection, so no controller); the reconnect
   pipeline reintroduces a controller precisely because recovery needs a stateful head
   operator.

#### Proposed design

All framework pieces land under `hololink_module/host` (+ their `module/core` impls);
the sensor-specific re-commit stays in the `Imx274Cam` driver.

**1. Fire-and-forget `trigger_reset()` — on the HSB-Lite board supplement.**
`trigger_reset` is the board-specific "poke the reset registers" operation, so it lives
on the board supplement interface `HsbLiteInterfaceV1`
(`module/hsb_lite/include/hololink/module/hsb_lite/hsb_lite.hpp`, alongside
`setup_clock`), **not** on the generic `HololinkInterfaceV1`. Add
`virtual hololink_module_status_t trigger_reset() = 0;`, implemented in `HsbLiteV1`
(`module/core/hsb_lite_default.hpp`) forwarding through `hololink_->legacy_access()` to
legacy `Hololink::trigger_reset()` (`src/hololink/core/hololink.cpp:399-413` — writes
the reset registers and returns; device I/O then fails until re-announce). Bind it in
`module/hsb_lite/python/hsb_lite_py.cpp` alongside `setup_clock`. Unlike
`HololinkInterfaceV1::reset()` it does **not** wait for re-enumeration or reconfigure —
that is the pipeline's job. (The test drives it from an `OnFrameNOperator` +
`Reactor.add_callback`, as legacy did, on the board's `HsbLiteInterfaceV1` supplement.)
The generic reconnection machinery (loss detection, eviction) still lives in
`hololink_module/host`; only this board-register poke is board-specific.

**2. Data-plane loss detection — a Reactor-alarm watchdog.** Loss detection can **not**
live in `RoceReceiverOp` / `LinuxReceiverOp::compute()`: the receivers are event-gated
by the frame-ready `AsynchronousCondition` (the stereo pattern port), so during a *full
stall* no frame arrives → the monitor never signals → `compute()` is never scheduled → a
deadline check inside it never runs. (The receiver's blocking monitor thread waits
indefinitely, so it doesn't notice either.) Instead, use a **Reactor-alarm** timer,
armed via `ReactorV1::add_alarm_s(watchdog_timeout, cb)` (the module `ReactorV1` already
exposes `add_alarm_s` / `add_alarm` / `cancel_alarm` / `AlarmHandle`), **tapped**
(cancel + re-arm) on **every received frame**. On timeout the alarm callback (reactor
thread) **calls the controller** — it is a pure detector; it does **not** invoke
`device_lost()` itself. The **controller owns the response**: it calls `device_lost()`
(the R2 invalidation) and drives recovery/fallback. (Detector/policy split — the
detector reports loss and never touches device state directly.) No new thread, no
`compute()` deadline, no `src/hololink` (legacy core) change. The optional ibverbs
`set_fault` (a link/QP-error callback from the RoCE monitor) remains a later enhancement
for *immediate* hard-error detection, but it needs legacy edits and is not required —
the watchdog covers silent stalls and errors alike.

**Reusable `Watchdog` object.** The watchdog is **not** hand-rolled in the controller —
it is a reusable `hololink::module::Watchdog`
(`host/include/hololink/module/watchdog.hpp`) over `ReactorV1::add_alarm_s`: `arm()` /
`tap()` (re)start the deadline, and on timeout it fires an `on_timeout` callback once on
the reactor thread. The controller constructs one in `start()` with
`on_timeout = [this]{ on_loss(); }` and calls `watchdog_->tap()` per frame. Its
thread-safety, resolved against the real reactor semantics, lives inside the object:

- A `mutex` guards the `AlarmHandle`; **deadlock-free** because the reactor's
  `cancel_alarm` only erases the pending entry (never waits on a running callback) and
  callbacks dispatch *outside* the reactor lock (`host/src/reactor_impl.cpp:187-199`,
  `:310-324`) — so `arm`/`cancel` take `{mutex → reactor lock}` while the timeout takes
  only `{mutex}`, no reverse edge.
- A **generation counter** drops a timeout that fired just as a concurrent
  `tap`/`cancel` superseded it (it slipped past `cancel_alarm` into the dispatch batch),
  so a frame landing exactly at the deadline doesn't cause a spurious loss.
- The alarm callback holds the watchdog's **shared state** (not the owner) and checks an
  `alive` flag, so a timeout in flight when the `Watchdog` is destroyed is a no-op, not
  a use-after-free — a lifetime hole the bespoke version had. The controller still
  `cancel()`s (via `stop_receiver`) and destroys the watchdog before its captured fields
  in `stop()`, so `on_loss` can't run against a torn-down controller.

> **Merges with R5.** The watchdog is owned by the sensor/controller layer (the legacy
> `SensorFactory` is the sensor-side controller), it taps on frames the receiver
> produces, and its timeout notifies the controller — so it is built as part of the
> controller op (R5), not as a standalone receiver-op change.

**3. Control-plane loss detection.** Control-plane register transactions already throw
on timeout (legacy read/write). The framework treats a control-plane timeout during
capture as a loss trigger feeding the same invalidation path (below), rather than
letting it escape as an unhandled error.

**4. Invalidation via `device_lost()` on the device-state interfaces — refcounted,
per-device.** Resolved design (given the hazard below): first make service ownership
**real** so eviction is safe, then invalidate a single device by dropping its cached
services through a method on the device-state object itself — **not** a host-side
`Adapter` call or a new C-ABI callback (a virtual on the interface dispatches into the
module exactly like `start()`/`reset()` already do, so no ABI layout change is needed).

- **Refcount the thunks (no ABI change).** `Publisher::get_service_thunk` records an
  outstanding strong `shared_ptr` per returned instance in a new `outstanding_`
  side-table (`unordered_map<const void*, {shared_ptr<void> sp; size_t count}>`, its own
  mutex); `release_service_thunk` — was a **no-op** — drops one outstanding ref (erasing
  at count 0). Host handles thus become owning: an instance stays alive while
  `registry_` caches it **or** any host handle is outstanding. (The in-binary
  `SelfModule` path already returns an aliasing `shared_ptr`, so it was always safe;
  only the cross-`.so` `LoadedModule` path needed this.) This is a behavioral change to
  `release_service`, not a struct/layout change — the C-ABI is untouched.
- **`device_lost()` on the interfaces; the Hololink invalidates its whole board.**
  Invalidation is driven through the device-state objects, and the object graph (which
  services belong to a board) is known by the objects, **never by the `Publisher`**.
  `HololinkInterfaceV1` and `DataChannelInterfaceV1` each gain
  `virtual hololink_module_status_t device_lost() = 0`. **`HololinkV1` is the
  aggregation point**: per-board services register themselves with it at configure via a
  `register_associated(this)` hook, so `HololinkV1::device_lost()` invalidates every
  associated service (each via `Publisher::invalidate(ptr)`) **and** itself — i.e.
  `device_lost` on the Hololink drops the board's whole object set.
  `DataChannelV1::device_lost()` simply delegates to `hololink_->device_lost()` (the
  child→parent edge lives in the data channel); the cascade then covers the channel
  because it registered. Both bound in Python (+ the `PyHololinkInterface` /
  `PyDataChannel` trampolines). Association is stored on `HololinkV1` as a
  `vector<const void*>` under a mutex; `Publisher::invalidate` is compare-only (never
  dereferences), and within the reconnect lifecycle every registered service is alive
  during the cascade and the whole Hololink is replaced afterward, so a raw-pointer list
  is safe (no ABA).
- **`Publisher::invalidate(const void* service)` — a dumb, identity-keyed primitive.**
  It removes every `registry_` entry whose stored pointer equals `service` (a service is
  published under its most-derived type via `ServicePublisher<T>`, so `this` inside its
  `device_lost()` equals the stored `.get()`; multiple `type_id` entries for the same
  instance are all removed). It carries **no notion of "device", serial, or grouping** —
  it just forgets the one object it's handed. A module cannot reach the host's `Adapter`
  singleton (RTLD_LOCAL), so the static resolves this binary's `Publisher` via
  `current_`, mirroring the `get_service` / `release_service` thunks. Outstanding host
  handles keep the instance alive until released; the **next** `get_service` misses the
  cache and reconstructs a **fresh** instance via `construct_service` (a compile-time
  per-`type_id` virtual — code, not registry data — so reconstruction always survives
  eviction), which is what makes re-fetch yield genuinely new pointers without dangling
  the old ones.
- **Registration coverage — all per-board services wired.** Each registers with its
  owning `HololinkV1` by one of two shapes: (i) services with a `configure(metadata)`
  resolve the Hololink and register there — `DataChannelV1` (anchor),
  `RoceDataChannelV1` / `LinuxDataChannelV1` / `CoeDataChannelV1` (transport),
  `HsbLiteV1`, `HsbLiteOscillatorV1`, and the per-channel `RoceReceiverV1` /
  `LinuxReceiverV1`; (ii) services built eagerly with backings take their owning
  `HololinkV1` in the constructor (passed by the matching
  `HsbLitePublisher::construct_*`), hold it, and register there — `I2cV1`,
  `PtpPpsOutputV1`, `SequencerV1`, `NullVsyncV1`. (`I2cV1` holding its Hololink also
  retires the old "caller must keep the parent alive" caveat.) The `SequencerV1` built
  here is the **frame-end** sequencer, keyed `serial=…;data_channel=…` — genuinely
  per-board; the separate `kind=software` / `kind=gpio0` factory sequencers are a
  different, unhandled path. Every registrant is a single-inheritance impl (or
  impl-published), so the registered `this` matches the pointer `Publisher::invalidate`
  compares against.
- Conceptually mirrors legacy `Hololink::reset_framework()`
  (`src/hololink/core/hololink.cpp:322-331`), but where the legacy code wiped a whole
  serial-keyed map from one place, invalidation here is per-object and driven by the
  objects, keeping device-graph knowledge out of the `Publisher`.

> **⚠ Ownership hazard found while implementing R2 (must resolve before coding).** The
> module ownership model makes naive eviction a **use-after-free**.
> `Publisher::registry_` holds the **only** strong `shared_ptr` to each service; the
> C-ABI `get_service_thunk` returns the **raw** `stored.get()` pointer
> (`module_base.cpp:236`), and the host wraps it in a `shared_ptr` whose deleter calls
> `release_service_thunk` — **a no-op** (`module_base.cpp:239-244`). So host handles are
> **non-owning raw pointers into registry-owned objects**. Clearing `registry_`
> therefore *destroys* the object immediately, and every outstanding host handle (the
> camera's `HololinkInterfaceV1`, the receiver op's data channel, …) dangles. My earlier
> note that "holders keep their objects alive until they drop them" was wrong — they do
> not. Also note the codebase's existing reset story is **reconfigure-in-place**, not
> eviction: `HsbLiteOscillatorV1` drops its rate cache via an `on_reset` listener
> (`hsb_lite_default.hpp:73-94`) rather than being re-created. Safe eviction needs one
> of: (a) a strict **drop-all-handles-then-evict-then- re-fetch** contract the
> controller/camera obey, never dereferencing across the eviction (works, but fragile —
> a stray held handle is a UAF); or (b) making the C-ABI ownership **real** —
> `get_service`/`release_service` refcount per instance so host handles keep the object
> alive and eviction just drops the registry's ref (robust, bigger change). This
> reopens, in practice, the eviction-vs-reconfigure decision, since reconfigure-in-place
> (the path the codebase already uses) sidesteps the hazard entirely by preserving
> object identity. **Flagged for decision below.**

**5. Controller op owns the reconnection loop; holders re-fetch on rediscovery.**
Rediscovery already works at the Adapter layer: after the reset, the device re-announces
and a fresh `wait_for_channel(peer_ip, …)` returns new `EnumerationMetadata`. The
persistent Holoscan operators are **not** rebuilt (the graph is fixed during `run()`);
instead the controller op — the head operator, wrapping the receiver — drives recovery
and each holder re-fetches its device-state objects in place. Concretely, the
controller:

- sits at the head of the pipeline and, per the legacy pattern, has a `connected()`
  state, a `frame_ready` async condition it owns, and a `fallback_image` (SMPTE bars) it
  emits downstream while disconnected so the CSI→bayer→CRC→visualizer chain keeps
  ticking;
- owns the **Reactor-alarm watchdog** (§2): it `tap()`s (cancel + re-arm `add_alarm_s`)
  on every frame the receiver delivers; the alarm's timeout callback (reactor thread)
  calls back into the controller's loss handler (the watchdog itself does no
  invalidation). That handler flips `connected()` false, calls `device_lost()`, runs
  `device_stop` → `detach_receiver`, and starts recovery. (A control-plane transaction
  timeout from step 3 feeds the same handler.)
- **recovers** (after the loss handler's `device_lost()` has evicted the stale
  serial-keyed services — via the data channel it holds, which delegates to the
  Hololink, step 4): `wait_for_channel` on its peer for the re-announcement,
  re-`get_service` the data channel + receiver from the fresh metadata,
  re-`attach_receiver`, `device_start`, re-arms the watchdog, and flips `connected()`
  true so real frames resume;
- while recovering, `compute()` emits the fallback image (mirrors legacy
  `HsbControllerOp.compute` / `.lost()`), so `frame_limit` still advances and the
  watchdog does not trip during the outage.

The **camera** participates in the re-fetch: its `HololinkInterfaceV1` /
`OscillatorInterfaceV1` handles are evicted by the `device_lost()` invalidation, so on
recovery it must drop and re-`get_service` them and re-commit its registers. This is
where the legacy `on_reset` re-arm maps — the `CameraWrapper` weak-method `on_reset`
handle from the pattern port is extended to additionally re-fetch the evicted handles
(the sensor-specific re-commit stays in `Imx274Cam`; the generic invalidate/notify stays
in the framework).

**6. New module-surface controller operator (wraps the receiver).** Add a module
controller op (host operator, e.g.
`host/operators/{include/hololink/module/operators/,}hsb_controller_op.*` + a
`PyHsbControllerOp` trampoline in `host/operators/python/operators_py.cpp`) that owns
the loop in step 5. It is constructed with the receiver (or the receiver's construction
inputs — `enumeration_metadata`, `frame_context`, `frame_size`,
`device_start`/`device_stop`) and a `fallback_image`, presents a single `output` port,
and is wired at the head of the pipeline feeding `CsiToBayerOp` (as legacy
`HsbControllerOp` fed `csi_to_bayer`). The receiver op's data-plane mechanics
(attach/detach, monitor thread, `get_next_frame`, `device_start/stop`) are what the
controller wraps and re-drives on recovery. Keep the op generic — no IMX274/board
specifics in the controller.

#### Test structure (`tests/test_module_imx274_reconnect.py`)

Port from the pattern-port harness, using the new controller op at the head:

- Discovery/build follows `test_module_imx274_pattern.py` (`Adapter.wait_for_channel`,
  `Imx274Cam(metadata)` / `CameraWrapper`, module CRC ops, `FrameAlignerOp` + sink-owned
  frame budget for stereo), but the head of the pipeline is the **controller op wrapping
  the receiver** (+ `fallback_image` SMPTE bars) rather than the receiver op wired
  directly.
- Add an `OnFrameNOperator` (or equivalent) that, at the chosen frame(s) (`reset_after`,
  legacy used `[150]`), fires the board's
  `HsbLiteInterfaceV1.get_service(metadata).trigger_reset()` via `Reactor.add_callback`.
- `frame_limit` extended past the reset (legacy `300`) so recovered frames are observed;
  the fallback-image ticks during the outage keep the frame budget advancing. CRCs
  validated only on the lossless RoCE path (as legacy), and only on frames after
  recovery settles (skip a window around the reset).
- Assert the camera's reset/re-fetch fired (the `on_reset`/re-fetch counter) and that
  real frames resume post-reset.

**Done when.** `py_compile` passes for the new test + touched `tests/` helpers; the
`_hololink_module_operators` pybind builds with the new `trigger_reset` and receiver
loss-detection params; and on hardware (IMX274 + RoCE, ≥1 channel IP) the pipeline
survives a mid-capture `trigger_reset`, re-fetches fresh device handles on rediscovery,
and resumes delivering correct-CRC frames.

#### Execution plan

Done on a feature branch off `internal` (current tree is `internal`; the reconnect work
is not yet committed). Ordered so each phase builds and is independently verifiable —
walking-skeleton framework pieces first, controller + camera + test last. Each phase
that touches the module API updates `hololink_module/README.md` in the same commit
(\[[keep-adapter-readme-in-sync]\]). Constants are `ALL_CAPS`
(\[[constants-use-all-caps]\]); no anonymous namespaces in `.cpp`
(\[[no-anonymous-namespace-in-cpp]\]).

- **R1 — fire-and-forget `trigger_reset()` on the HSB-Lite supplement (no deps). DONE
  (code; awaiting Docker build).** Added pure-virtual `trigger_reset()` to
  `HsbLiteInterfaceV1` (`module/hsb_lite/include/hololink/module/hsb_lite/hsb_lite.hpp`,
  next to `setup_clock`), impl in `HsbLiteV1` (`module/core/hsb_lite_default.hpp`)
  forwarding through `hololink_->legacy_access()` to legacy `Hololink::trigger_reset()`,
  bound in `module/hsb_lite/python/hsb_lite_py.cpp` next to `setup_clock` (raises on
  non-OK status). `HsbLiteV1` is the only `HsbLiteInterfaceV1` impl. *Verify:* framework
  builds (Docker: `docker/build.sh` / `ci/build.sh` — no local C++ toolchain in this
  sandbox); `HsbLiteInterfaceV1.get_service(metadata).trigger_reset()` callable from
  Python; the pattern test is unaffected (new method, no existing-path change).

- **R2 — refcounted ownership + per-device `device_lost()` invalidation (no deps). DONE
  (code; awaiting Docker build).** Implemented per §4, **no ABI layout change**: (a)
  `Publisher` refcounts host handles in a new `outstanding_` side-table —
  `get_service_thunk` records a strong ref, `release_service_thunk` (was a no-op) drops
  one — making cross-`.so` host handles owning (`host/src_module/module_base.cpp`,
  `publisher.hpp`); this is a behavioral change to `release_service` only, the C-ABI
  structs are untouched. (b) `HololinkInterfaceV1::device_lost()` + `HololinkV1`
  cascades: it invalidates every service that registered via `register_associated(this)`
  (a `vector<const void*>` under a mutex) plus itself.
  `DataChannelInterfaceV1::device_lost()` + `DataChannelV1` delegates to
  `hololink_->device_lost()` (the cascade covers the channel). Both bound in Python (+
  the `PyHololinkInterface` / `PyDataChannel` trampolines). (c)
  `Publisher::invalidate(const void* service)` is a dumb, identity-keyed, compare-only
  static — resolves this binary's `Publisher` via `current_` and removes the `registry_`
  entries whose stored pointer equals `service`. **No device/serial/graph knowledge in
  the `Publisher`.** No `Module::reset_framework`, no C-ABI callback, no
  `Adapter::reset_framework`, no per-module edits; no C++ test stub touches the C-ABI.
  (d) `register_associated` wired for **all** per-board services: those with a
  `configure` register there (data-channel family, `HsbLiteV1`, `HsbLiteOscillatorV1`,
  `Roce`/`LinuxReceiverV1`); the eagerly-built ones take their owning `HololinkV1` via
  constructor (passed by the matching `construct_*`) and register there (`I2cV1`,
  `PtpPpsOutputV1`, `SequencerV1`, `NullVsyncV1`). *Verify:* framework builds (Docker —
  no local toolchain); after `hololink.device_lost()`, a re-`get_service` on that
  board's Hololink **and** its registered services returns **new** instances that re-run
  `configure()`, still-held old handles stay valid until dropped (no UAF), and a second
  board is unaffected.

- **R3 — data-plane loss detection via a Reactor-alarm watchdog (deps: R2; merges into
  R5).** Per §2: a sensor/controller-owned watchdog on `ReactorV1::add_alarm_s`, tapped
  (cancel + re-arm) on every received frame. Its timeout callback **calls the
  controller's loss handler** (pure detector — it does not invalidate); the controller
  then calls `device_lost()` and recovers. **Not** a `compute()` deadline (the receivers
  are event-gated, so `compute()` never runs during a full stall) and **not** a new
  thread (the Reactor's alarm timer fires it). Because the watchdog is owned by the
  controller layer, taps on the receiver's frames, and calls back into the controller,
  it is implemented as part of the controller op — see R5. Optional later enhancement:
  ibverbs `set_fault` for immediate hard-error detection (needs legacy edits). *Verify:*
  pattern test unaffected (no watchdog wired there); on a forced mid-capture stall the
  timeout calls the controller, which calls `device_lost()`, and frames resume after
  rediscovery.

- **R4 — control-plane loss detection (deps: R2). DONE (code).** Control-plane failures
  are detected at the **reconnection boundary**, not inside the wrappers. The
  `HololinkV1` register methods do **not** catch backing exceptions — a control-plane
  `hololink::TransactionError` / `TimeoutError` propagates like it does out of every
  other wrapper (`i2c` / `data_channel` / `sequencer`), keeping the "don't swallow
  backing exceptions" rule (Code Generation Guidelines) uniform across `module/core/`.
  Instead, `SensorFactory::on_enumerated` (`host/operators/sensor_factory.cpp`) — the
  one place that owns loss recovery — wraps the whole (re)connect bring-up (`new_sensor`
  \+ `on_connect`) in a single sanctioned recovery-boundary
  `catch (const std::exception&)`. A bring-up that throws, or a Python `new_sensor` that
  reports failure by returning null (its trampoline having swallowed the exception),
  routes into `invalidate_board(metadata)`, which resolves the board's
  `HololinkInterfaceV1` and calls `device_lost()` — the same R2 per-board cascade — so
  the next announcement re-materializes fresh device state instead of the stale handles
  that just failed; the factory then stays disconnected and retries on re-announce.
  Re-resolving to reach the Hololink is cheap (its constructor opens no socket and does
  no I/O). The R3 data-plane watchdog still detects steady-state loss independently.
  *Verify:* builds; a control-plane transaction that fails during bring-up invalidates
  the board and the next announcement reconnects with fresh handles.

- **R5 — decoupled `SensorFactory` / `NetworkReceiver` under an `HsbController`
  orchestrator (deps: R1–R4). DONE — RoCE mono + stereo verified on hardware; Linux
  transport built.** Five collaborating pieces; the orchestrator owns the lifecycle and
  the sensor and data-plane halves are decoupled and never see each other:

  - **`NetworkReceiver`** (`.../operators/network_receiver.hpp`) — the transport seam
    (the transports have **no common base**), built by an app-supplied
    `NetworkReceiverFactory`. Surface is `construct(metadata, Config)` / `run()` /
    `destruct()` plus `get_next_frame(timeout, cuda_stream)` / `frames_ready` /
    `frame_memory` / `frame_buffer_owner` / `stamp_metadata` / `data_channel`.
    `get_next_frame` takes the operator's per-compute pipeline stream (allocated from
    the Holoscan execution context and set on the emitted tensor): a software transport
    places its host→device frame copy on it so the copy overlaps downstream work; a
    hardware transport that DMAs straight to device memory ignores it.
    **`RoceNetworkReceiver`** (`roce_network_receiver.{hpp,cpp}`) absorbs
    `RoceReceiverOp`'s resolve/attach/monitor/`get_next_frame`/metadata behind it, built
    by `make_roce_network_receiver_factory()` (gated on `HOLOLINK_BUILD_ROCE`).
    **`LinuxNetworkReceiver`** (`linux_network_receiver.{hpp,cpp}`) is the software
    sibling: it absorbs `LinuxReceiverOp`'s datagram-socket + user-space RoCEv2
    reassembly path
    (create/`configure_socket`/`start`/`attach`/monitor-thread-with-affinity, per-frame
    metadata straight off the receiver's frame-info struct), built by
    `make_linux_network_receiver_factory()`. It links no ibverbs, so it is built
    **unconditionally** and its factory is bound unconditionally.
  - **`Watchdog`** (`host/include/hololink/module/watchdog.hpp`) — a reusable
    `ReactorV1` deadline timer (mutex + generation + shared-state alive flag; §2).
  - **`SensorDevice`** (`.../operators/sensor_device.hpp`) — a **wrapper over one sensor
    driver** (the camera, e.g. `Imx274Cam`), created already-configured+armed by
    `SensorFactory::new_sensor`. App override (Python trampoline): `stop_sensor()` and
    `fallback_frame(ptr, size)` (the pipeline data emitted while disconnected — sensor
    knowledge, no live device handles). Kept across an outage so fallback still works;
    replaced on reconnect.
  - **`SensorFactory`** (`.../operators/sensor_factory.{hpp,cpp}`) — a C++ base (Python
    trampoline) that **owns the watchdog, the reconnection policy, and sensor
    bring-up**, and is **decoupled from the data plane** (it reports connect/disconnect
    to the controller through two callbacks, never touching a receiver). Pure-virtual
    app hook: `new_sensor(metadata) -> SensorDevice`. Base-implemented (but `virtual`)
    lifecycle: `start(on_connect, on_disconnect, metadata, watchdog_timeout_s)`
    registers via `Adapter::register_ip` and stays registered for the run, so **one**
    `on_enumerated` path drives both the initial connect and every reconnect (unified);
    `tap()` refreshes the watchdog; `on_loss()` (watchdog timeout) stops the sensor +
    fires `on_disconnect`, and the still-registered handler reconnects on re-announce.
  - **`HsbController`** (`hsb_controller.{hpp,cpp}`) — the **orchestrator** (not a
    Holoscan type). Owns one `NetworkReceiver` and the `SensorFactory`; `found`/`lost`
    (the factory's callbacks, on the reactor thread) cycle the receiver
    `construct`→`run` / `destruct` and call `data_channel->device_lost()` on loss;
    `get_next_frame` (scheduler thread) pulls a frame + taps the factory. The receiver
    is guarded by `receiver_mutex_` (reactor vs scheduler thread).
  - **`HsbControllerOp`** (`hsb_controller_op.{hpp,cpp}`) — a **thin** HSDK adapter that
    only implements `holoscan::Operator`: owns the frame-ready `AsynchronousCondition`
    and the GXF tensor plumbing, and forwards `start`/`stop`/`compute` to a
    `unique_ptr<HsbController>`. `compute()` allocates the pipeline CUDA stream from the
    execution context, threads it through `HsbController::get_next_frame` to the
    receiver, and sets it on the emitted tensor; it emits the controller's next frame
    while connected, else the controller's fallback frame.

  Op params: `enumeration_metadata`, `frame_context`, `frame_size`,
  pages/queue/page_size, `metadata_offset`, `out_tensor_name`, `rename_metadata`,
  `network_receiver_factory`, `sensor_factory`, `watchdog_timeout_s`. Wired ahead of
  `CsiToBayerOp`. pybind (`PyHsbControllerOp`, `SensorFactory` trampoline with
  `new_sensor` + `fallback_frame`, `SensorDevice` trampoline with `stop_sensor`, opaque
  `NetworkReceiver` for factory round-trip, `make_linux_network_receiver_factory` always
  bound + `make_roce_network_receiver_factory` gated) and CMake (transport-agnostic
  pieces and `linux_network_receiver.cpp` always built; `roce_network_receiver.cpp`
  gated on `HOLOLINK_BUILD_ROCE`) are in place; superseded `receiver_transport.hpp` /
  `roce_transport.{hpp,cpp}` removed. Both transports (`RoceNetworkReceiver` +
  `LinuxNetworkReceiver`) are implemented. **Next:** Docker build. *Verify:* builds;
  `py_compile`; a mono pipeline emits fallback on a forced stall and real frames after
  rediscovery.

- **R6 — camera re-fetch on reset (deps: R2). DONE — absorbed into `new_sensor`.**
  Rather than an `on_reset` re-`get_service` handler on a long-lived camera, the new
  architecture rebuilds the sensor from scratch on every (re)connect:
  `SensorFactory.new_sensor(metadata)` constructs a fresh `Imx274Cam(metadata)`, whose
  constructor re-`get_service`s the board's `HololinkInterfaceV1` /
  `OscillatorInterfaceV1` / `I2c` handles — so the camera resyncs with the post-reset
  module state implicitly, no explicit re-fetch/notify path needed. Board bring-up
  (`hololink.start()` + `reset()`) lives in the app's `new_sensor`, guarded once per
  board per (re)connect (a reconnect yields a fresh, thus not-yet-brought-up,
  `HololinkInterfaceV1` instance). Keeps the `SensorFactory` base generic; board/device
  specifics stay in the app. *Verify:* on rediscovery the rebuilt camera holds fresh
  handles (the `_sensor_count` advances past 1).

- **R7 — the test (deps: R5, R6). RoCE mono + stereo DONE and verified on hardware;
  Linux cases un-gated now that `LinuxNetworkReceiver` is built (pending a hardware
  run).** `tests/test_module_imx274_reconnect.py`: the app subclasses
  `hololink_module.operators.SensorFactory` (`new_sensor` + `fallback_frame`) and
  `.SensorDevice` (`stop_sensor`), wires `HsbControllerOp` (with
  `make_roce_network_receiver_factory()` for RoCE or
  `make_linux_network_receiver_factory()` for the software path —
  `_linux_receiver_factory` resolves the binding, no longer skipping) at the pipeline
  head feeding `CsiToBayerOp`, and triggers the reset at frame 150 via
  `hsb_lite.HsbLiteInterfaceV1.get_service(metadata).trigger_reset()` deferred onto the
  reactor (`OnFrameNOperator` → `Adapter.reactor().add_callback`; the new
  `Adapter.reactor()` binding returns the host `ReactorV1`, which lives on the host
  module — not a device module). `frame_limit=300`; the fallback SMPTE-bars CSI frame is
  rendered up front into a `frame_size` cupy device buffer and served by the
  `SensorDevice` during the outage. (`make_csi_from_image_file` is copied into the test
  and retargeted to the module `csi` enums, since `utils`'s copy is bound to the legacy
  `hololink` enum type.) RoCE CRCs are validated outside a settle window around the
  reset; the test asserts `_sensor_count >= 2` (initial + reconnect). The Linux cases
  (`test_linux_imx274_reconnect`, `test_stereo_linux_imx274_reconnect`) validate
  recovery only (`_sensor_count`), not CRCs, since software sockets drop packets. All
  seven legacy cases are now ported: the four remaining ones —
  `test_stereo_linux_imx274_reconnect` and the three **reconnect-during-configuration**
  variants (Linux mono, RoCE mono, RoCE stereo) — are in place. The during-configuration
  cases trigger the reset from *inside* camera configuration via an
  `InstrumentedImx274Cam` whose `set_register` override fires
  `HsbLiteInterfaceV1.trigger_reset()` on the Nth write (exactly once), so the board is
  lost mid-bring-up — directly exercising the R4 control-plane path (`new_sensor` fails
  → `SensorFactory::invalidate_board` → retry). They assert the reset fired and, for
  RoCE, that the recovered stream ends in a long clean run of correct-CRC frames
  (`_clean_tail_length`), which is robust to the fallback-frame prefix the module
  pipeline emits during the outage (unlike the legacy, which recorded no fallback).
  *Verify:* `py_compile` + `black` + `flake8` clean; on hardware, survives mid-capture
  `trigger_reset` and resumes (RoCE: correct-CRC frames; Linux: reconnects).

Walking-skeleton milestone: **R1 + R2 + R3 + R5 + R6 + a mono-RoCE R7 case** proves the
loss→invalidate→rediscover→re-fetch→resume path end-to-end. **Achieved and verified on
hardware:** `test_roce_imx274_reconnect` (mono, instances 0 and 1) and
`test_stereo_roce_imx274_reconnect` both pass — a mid-capture `trigger_reset` at frame
150 is detected by the R3 watchdog, drives `device_lost()`, and frames resume with
correct CRCs after rediscovery. R4 (control-plane timeout routing) is now implemented as
a complementary detector. Remaining: `LinuxNetworkReceiver` (the Linux reconnect cases
skip until it lands) and a consolidated Docker/CI build of all the C++ changes.
