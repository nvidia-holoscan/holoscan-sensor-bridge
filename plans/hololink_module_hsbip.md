# `HsbIp` base publisher + a 2510-FPGA `leopard_vb1940` module

## Context

`plans/hololink_module.md` describes the module framework and the first-party modules
built on it. Two facts drive this work:

1. **A 2510-FPGA Leopard VB1940 exists and has no module.** The delta it needs is the
   same delta that separates `module/hsb_lite_2510/` from `module/hsb_lite/`: the
   pre-0x2602 data-plane memory map (`HsbLite2510DataChannel`), the pre-0x2603 RDMA
   immediate-data layout on both receivers, and an FPGA-version gate at enumeration.
   Those implementations already live in `module/core/`, promoted there precisely so
   "future partner supplements targeting pre-0x2603 / pre-0x2602 FPGA revisions" could
   reuse them — but they are reachable today only through `HsbLite2510Publisher`, a
   subclass of `HsbLitePublisher`.

2. **`module_core::HsbLitePublisher` is doing two jobs.** It is both the canonical
   HSB-IP-shaped publisher (control plane, data channels, receivers, sequencer, I2C,
   PTP/PPS, enumeration body) and the HSB-Lite board definition (CrossLink-NX D-PHY,
   the `HsbLiteInterfaceV1` supplement, the Bajoran TS1 oscillator, the 3-sensor /
   2-SIF channel layout). Every non-HSB-Lite board inherits it and then overrides the
   HSB-Lite parts back out — `leopard_vb1940` overrides `construct_hsb_lite` to return
   `false` purely to keep the `hsb_lite.v1` type_id from resolving on a Leopard board,
   and `taurotech_da326` does not, so a DA326 today publishes an `HsbLiteV1` supplement
   it has no use for.

This work splits the generic half out as `module_core::HsbIpPublisher`, re-parents the
non-HSB-Lite boards onto it, and adds `module/leopard_vb1940_2510/`.

The 2510 VB1940 usually reports `compat_id = 0`, so for that board the `.so` filename
cannot select the module. The FPGA version is tested instead — the same approach
`hsb_lite_2510` follows: an `hsb_ip_version` range gate in `update_metadata`, and an
`hsb_ip_version` comparison in each data channel's / receiver's factory hook that picks
the pre-0x2602 backing class.

The current 2603 Leopard FPGA does report `compat_id = 0x2603`, so the split maps
cleanly onto the loader: a 2603 board resolves the compat-suffixed `.so` and a 2510
board (`compat_id = 0`, no dedicated `.so`) falls through to the bare one. The
`NO_COMPAT_SUFFIX` note in `module/leopard_vb1940/CMakeLists.txt` — "the Leopard VB1940
implementation does not currently publish a compat-id over bootp" — is stale.

## Decisions taken

| Question | Decision |
| --- | --- |
| Loader selection | Mirror the `hsb_lite` pair: `module/leopard_vb1940_2510/` takes the bare `hololink_<UUID>.so` (the catch-all for `compat_id=0`, absent, or any compat without a dedicated `.so`); `module/leopard_vb1940/` moves to the compat-suffixed `hololink_<UUID>_2603.so`. |
| `HsbIp` scope | The CLNX D-PHY default, the `HsbLiteInterfaceV1` supplement, the Bajoran TS1 oscillator, and the 3-sensor / 2-SIF channel configuration all stay behind in `HsbLitePublisher`. `HsbIpPublisher` keeps the generic chain, the shared addressing formula, and a share-by-serial oscillator helper. |
| 2510 reuse across branches | Subclass and duplicate. No template mixin: `LeopardVb1940_2510_Publisher` derives from `LeopardVb1940Publisher` and carries its own copies of the four `construct_*` overrides and the version gate. |
| 2510 naming | Implementations move to `HsbIp2510*` in `hsb_ip_2510_*.hpp`. The `HsbLite2510*` names and their `*.hsb_lite_2510.v1` type_ids retire: nothing in-tree resolved those strings, and `plans/hololink_module_interface_versions.md` lists them as explicitly unfrozen internal override targets. |

## Deliverables

### 1. New `module/core/hsb_ip_publisher.hpp` — `HsbIpPublisher`

The generic HSB-IP-shaped publisher, moved out of `hsb_lite_publisher.hpp` unchanged
except as noted. Carries:

- `construct_service` — the short-circuit OR chain, with two edits: `construct_hsb_lite`
  and `construct_mipi_dphy` are both gone from it, and it gains a
  `construct_board_services(instance_id, type_id)` tail hook (default `false`) so a
  subclass can append a branch after the canonical ones. `construct_overrides` stays at
  the head but narrows to its actual job — substituting a board's impl for a service the
  chain would otherwise build. A service the chain knows nothing about has nothing to
  preempt, so every in-tree board supplement (`LeopardVb1940V1`, `TauroTechDa326V1`,
  `TutorialDeviceV1`, `HsbLiteV1`) publishes from the tail hook instead; no board
  overrides `construct_overrides` any more.
- `update_metadata`, `publish_enumeration`, `publish_frame_metadata`, `setup(init)`,
  `module_name()` (pure virtual) — unchanged bodies.
- `construct_hololink`, `construct_ptp_pps_output`, `construct_null_vsync`,
  `construct_data_channel`, `construct_roce_data_channel`, `construct_coe_data_channel`,
  `construct_roce_receiver` (declared; defined out-of-line),
  `construct_linux_data_channel`, `construct_linux_receiver` (declared; defined
  out-of-line), `construct_sequencer`, `construct_i2c` — unchanged bodies.
- `set_mipi_dphy_metadata` — **removed**, replaced by a generic
  `update_board_metadata(metadata)` tail hook called from the same point in
  `update_metadata`, whose return value becomes `update_metadata`'s. A board stamps
  whatever its own hardware needs there (the D-PHY register shape is what the in-tree
  boards use it for) and returns `HOLOLINK_MODULE_OK`, or declines a device it must not
  drive with `HOLOLINK_MODULE_ENUMERATION_SKIPPED`. The base stamps nothing and
  accepts. The existing contract still
  covers a board that stamps nothing: "a design that stamps none is inert: `configure()`
  leaves it unconfigured and every `program()` call reports
  `HOLOLINK_MODULE_NOT_FOUND`."
- `construct_mipi_dphy` — **removed entirely**, not defaulted to `false`. The D-PHY
  register layout belongs to the FPGA design on the carrier, not to the HSB IP, so the
  base declares no branch for it; a board publishes its own from
  `construct_board_services`. The two publish helpers leave the publisher as free
  functions `construct_clnx_mipi_dphy(publisher, instance_id, type_id)` and
  `construct_cpnx_mipi_dphy(...)`, sitting next to the impls they publish in
  `module/core/mipi_dphy/clnx.hpp` and `cpnx.hpp`. `hsb_ip_publisher.hpp` then includes
  neither D-PHY header, in either direction: the stamping half moves out too, behind
  the generic `update_board_metadata` hook above.
- `mipi_dphy_parent(instance_id)` — renamed `configured_hololink(instance_id)`, since
  nothing about resolving a configured per-board `HololinkV1` is D-PHY-specific. Body
  and error text unchanged apart from the "While constructing MipiDphy" prefix, which
  becomes a caller-supplied `what` string.
- `construct_oscillator(instance_id, type_id)` — returns `false` by default (the base
  names no clock generator), plus a new protected
  `template <typename Impl> bool construct_shared_oscillator(instance_id, type_id)`
  holding the share-by-serial logic and the `serial -> shared_ptr<void>` cache that
  `HsbLitePublisher::construct_oscillator` and Leopard's copy both open-code today. The
  comment explaining why share-by-serial is required for the impl's rate cache moves
  onto the helper.
- `hsb_ip_sensor_metadata(...)` — renamed from `hsb_lite_sensor_metadata`. Body
  unchanged; it is the HSB IP register-map formula, not an HSB-Lite one.
- `HsbIpChannelConfigurationV1` — new base for the per-board channel configurations,
  carrying `type_id = "channel_configuration.module_core.v1"`, `use_mtu`,
  `use_multicast`, and abstract `use_sensor` / `is_sensor_valid`. The type_id string is
  unchanged, so every board's `ServicePublisher<...>(…).publish("", impl)` keeps
  registering under the same key; only the C++ type named in the template argument
  changes.

### 2. `module/core/hsb_lite_publisher.hpp` — `HsbLitePublisher` reduced to HSB-Lite

`class HsbLitePublisher : public HsbIpPublisher`, holding only:

- `construct_board_services` → `construct_hsb_lite` (publishes `HsbLiteV1` on
  `hsb_lite.v1`) then the CLNX D-PHY helper; both moved here from the base chain.
- `construct_oscillator` → `construct_shared_oscillator<HsbLiteOscillatorV1>` (Bajoran
  TS1).
- `update_board_metadata` → `stamp_clnx_mipi_dphy_metadata`, paired with the CLNX
  helper its `construct_board_services` calls. The existing comment explaining that the
  two must agree moves here with them.
- `publish_channel_configuration` → `HsbLiteChannelConfigurationV1`.
- `HsbLiteChannelConfigurationV1 : HsbIpChannelConfigurationV1` — keeps `TOTAL_SENSORS
  = 3`, `is_sensor_valid`, and the 2-SIFs-per-sensor `use_sensor` that now calls
  `hsb_ip_sensor_metadata`.

`module_name()` stays pure virtual; `module/hsb_lite/module_entry.cpp` is unchanged.

### 3. Boards re-parented onto `HsbIpPublisher`

- **`module/leopard_vb1940/`** — `LeopardVb1940Publisher : HsbIpPublisher`. Deletes the
  `construct_hsb_lite` override (nothing to suppress any more) and replaces the
  hand-rolled `construct_oscillator` + `oscillator_by_serial_` map with
  `construct_shared_oscillator<LeopardVb1940OscillatorV1>`. Keeps the CPNX D-PHY, now as
  a `construct_board_services` override paired with an `update_board_metadata` that
  stamps the CPNX shape at 60 MHz.
  `LeopardVb1940ChannelConfigurationV1` re-parents to `HsbIpChannelConfigurationV1` and
  gains an `is_sensor_valid` override sharing its own `TOTAL_SENSORS = 3` with
  `use_sensor` (today it inherits HSB-Lite's).
- **`module/taurotech_da326/`** — `TauroTechDa326Publisher : HsbIpPublisher`. Adds an
  explicit `construct_oscillator` → `construct_shared_oscillator<HsbLiteOscillatorV1>`
  so today's TS1 behavior is preserved rather than silently inherited.
  `TauroTechDa326ChannelConfigurationV1` re-parents to `HsbIpChannelConfigurationV1` and
  gains `is_sensor_valid` with `TOTAL_SENSORS = 2` — **a fix**: it currently inherits
  HSB-Lite's 3-sensor bound, so `update_metadata` accepts a bootp `data_plane` of 2 on a
  2-sensor board and then throws out of `use_sensor` in the reactor's bootp callback.
  A DA326 also stops publishing `HsbLiteV1` / the `hsb_lite.v1` type_id — **the point of
  the re-parenting, not a regression**: the DA326 is not expected to implement
  `hsb_lite.v1`, its own supplement is `TauroTechDa326V1`, and the type_id resolves
  today only as an accident of inheriting `HsbLitePublisher`.
- **`module/tutorial/` stays on `HsbLitePublisher`.** It is an HSB-Lite-shaped device
  (CLNX D-PHY, TS1 oscillator, HSB-Lite supplement), and
  `docs/user_guide/module_device_tutorial.mdx` documents that inheritance. Flagging it
  here so the exclusion is a decision rather than an oversight.

### 4. 2510 implementations renamed to `HsbIp2510*`

New files under `module/core/`, holding the current bodies verbatim under the new names:

| New file | Types |
| --- | --- |
| `hsb_ip_2510_data_channel.hpp` | `HsbIp2510DataChannel`, `HsbIp2510RoceDataChannelV1` (`"roce_data_channel.hsb_ip_2510.v1"`), `HsbIp2510LinuxDataChannelV1` (`"linux_data_channel.hsb_ip_2510.v1"`) |
| `hsb_ip_2510_roce_receiver.hpp` | `HsbIp2510RoceReceiver`, `HsbIp2510RoceReceiverV1` (`"roce_receiver.hsb_ip_2510.v1"`) |
| `hsb_ip_2510_linux_receiver.hpp` | `HsbIp2510LinuxReceiver`, `HsbIp2510LinuxReceiverV1` (`"linux_receiver.hsb_ip_2510.v1"`) |

`hsb_lite_2510_{data_channel,roce_receiver,linux_receiver}.hpp` are **deleted**, along
with every `HsbLite2510*` transport name and the four `*.hsb_lite_2510.v1` type_ids.
Nothing in-tree — application, example, test or Python binding — resolved those strings,
and `plans/hololink_module_interface_versions.md` records them as internal override
targets that stay editable. Keeping them would have meant a second name for each impl
and a fourth key on every registration chain.

`HsbLite2510Publisher` and its two out-of-line receiver-construct TUs construct the
`HsbIp2510*` types directly, exactly as the Leopard 2510 publisher does, so both
branches expose one set of type_ids.

### 5. New `module/leopard_vb1940_2510/`

Prerequisite: **`LeopardVb1940Publisher` and the three module-private impl classes move
out of `module/leopard_vb1940/module_entry.cpp`** into a module-private
`module/leopard_vb1940/leopard_vb1940_publisher.hpp` (same placement rationale as
`module/core/hsb_lite_publisher.hpp` — the header both module dirs include, while the
public supplement type stays in `include/hololink/module/leopard_vb1940/`).
`module/leopard_vb1940/module_entry.cpp` shrinks to the `g_publisher` static plus
`hololink_module_init`, matching `module/hsb_lite/module_entry.cpp`.

New files:

- `leopard_vb1940_2510_publisher.hpp` — `LeopardVb1940_2510_Publisher :
  LeopardVb1940Publisher`. The 2510 board's D-PHY configuration (CPNX, 60 MHz settle
  clock) and Bajoran TS2 oscillator are identical to the 2603 board's, so both are
  inherited with no override; the subclass carries only the four transport branches and
  the version gate:
  - `module_name()` → `"leopard_vb1940_2510"`.
  - `update_metadata` — the `hsb_ip_version` gate, duplicated from
    `HsbLite2510Publisher` (per-serial log cap at 4 notices,
    `HOLOLINK_MODULE_ENUMERATION_SKIPPED` on an unsupported version, delegate to the
    base otherwise), with **an upper bound only**: decline `hsb_ip_version > 0x2603`,
    accept everything at or below it. `hsb_lite_2510`'s lower bound of `0x2510` is not
    copied. The 2510 Leopard is known only to report *some* value below `0x2603`; a
    lower bound guessed wrong would decline the one board this module exists for, and
    because this module is the bare catch-all it would decline every Leopard board
    reporting a low or absent version. The upper bound is the part that earns its keep —
    it keeps newer silicon this build predates from being mis-driven. Add a lower bound
    once the exact reported value is known.
  - `construct_roce_data_channel` / `construct_linux_data_channel` → publish
    `HsbIp2510RoceDataChannelV1` / `HsbIp2510LinuxDataChannelV1`. Their `make_backing`
    picks `HsbIp2510DataChannel` below 0x2602 and the stock `hololink::DataChannel` at
    or above it, so one module drives both FPGA revisions.
  - `construct_roce_receiver` / `construct_linux_receiver` — declared here, defined
    out-of-line (the ibverbs and `<cuda.h>` dependencies stay out of the header),
    publishing `HsbIp2510RoceReceiverV1` / `HsbIp2510LinuxReceiverV1`.
- `roce_receiver_construct_2510.cpp`, `linux_receiver_construct_2510.cpp` — the two
  out-of-line definitions, same `HOLOLINK_BUILD_ROCE` shape as the identically named
  pair `module/hsb_lite_2510/` compiles out of `module/core/`. The `_2510` suffix keeps
  them from colliding with the base pair `add_hololink_module()` injects.
- `module_entry.cpp` — `g_publisher` + `hololink_module_init`.
- `CMakeLists.txt` — `add_hololink_module(NAME leopard_vb1940_2510 UUID
  f1627640-b4dc-48af-a360-c55b09b3d230 NO_COMPAT_SUFFIX SOURCES module_entry.cpp
  roce_receiver_construct_2510.cpp linux_receiver_construct_2510.cpp)`, with
  `module/leopard_vb1940/` and `module/leopard_vb1940/include` on the include path and
  `hololink::module_core` linked. No new headers target and no Python sub-package: the
  module publishes the existing `LeopardVb1940InterfaceV1`, so `Vb1940Cam` reaches it
  unchanged.

`hololink_module/CMakeLists.txt` gains `add_subdirectory(module/leopard_vb1940_2510)`.

Note that the injected `module/core/roce_receiver_construct.cpp` /
`linux_receiver_construct.cpp` TUs now define `HsbIpPublisher::construct_roce_receiver`
/ `construct_linux_receiver` and include `hsb_ip_publisher.hpp`, which no longer pulls in
a board supplement header. Both Leopard modules can therefore drop
`hololink::hsb_lite::headers` from their include paths.

### 6. `module/leopard_vb1940/CMakeLists.txt` — compat-suffixed

Drop `NO_COMPAT_SUFFIX`, so the module builds as
`hololink_f1627640-b4dc-48af-a360-c55b09b3d230_2603.so` from `DEFAULT_COMPAT`, which is
what a 2603 Leopard board's `compat_id` now resolves to. Rewrite the leading comment: it
still claims the FPGA publishes no compat-id, and the bare-UUID exception now belongs to
`module/leopard_vb1940_2510/`. The two documented `NO_COMPAT_SUFFIX` users are
`module/hsb_lite_2510/` and `module/leopard_vb1940_2510/`.

### 7. Tests

- **New `tests/hololink_module_leopard_vb1940_test.cpp`**, modelled on
  `tests/hololink_module_hsb_lite_test.cpp` (same TEST-NET-1 peer IPs + ephemeral-port
  listener guards):
  - `compat_id = 0x2603` loads `hololink_<UUID>_2603.so`; enriched metadata carries
    `module_name == "leopard_vb1940"`.
  - `compat_id = 0` and metadata with no `compat_id` both fall through to the bare
    `hololink_<UUID>.so`; `module_name == "leopard_vb1940_2510"`.
  - An `hsb_ip_version` above `0x2603` is declined: `enumerate` fires no
    post-enrichment callback and does not throw. A low version (below `0x2510`, where
    `hsb_lite_2510` would decline) is *accepted* — pinning the deliberate absence of a
    lower bound.
  - `update_metadata` and `Adapter::use_sensor` agree field-for-field on the Leopard
    layout (`data_plane` 0 for every sensor, one SIF per sensor, `i2c_bus = 1 +
    sensor_number`), and an out-of-range sensor is rejected.
  - Wire into `tests/CMakeLists.txt` with `add_dependencies(… leopard_vb1940
    leopard_vb1940_2510)` and `HOLOLINK_MODULE_TEST_MODULE_DIR=$<TARGET_FILE_DIR:leopard_vb1940>`.
- **`tests/hololink_module_hsb_lite_test.cpp` unchanged and still green.** It drives
  both HSB-Lite modules end-to-end without naming an impl type_id, so it is the
  regression gate for deliverables 1, 2 and 4 — the split and the 2510 rename must be
  invisible to it.
- **Hardware runs.** `ctest -L vb1940` (`module_vb1940_player_test`,
  `module_stereo_vb1940_player_test`, `module_linux_vb1940_player_test`) needs no edit,
  but it now exercises whichever Leopard module the attached board's `compat_id` selects
  — worth running against both a 2510 board and a current board.

### 8. Documentation

- `hololink_module/README.md` — a new phase-log section in the established style
  (**Implemented** / **Out of scope** / **Build commands** / **Test-run commands**)
  covering the `HsbIp` split, the `HsbIp2510*` rename, and the new module. Include the
  build targets `leopard_vb1940` and `leopard_vb1940_2510`, and the resulting `.so`
  names.
- `plans/hololink_module.md` — the module tables and directory listings that name the
  Leopard `.so` as the bare-UUID exception: the first-party module table (~line 84), the
  module-directory example (~line 290), the install-layout tree (~line 2486), and the
  Phase 5 Leopard extension's packaging paragraph (~line 4422). Add a Phase 5 extension
  entry for this work.

## Open item

**What exact `hsb_ip_version` does a 2510 VB1940 report?** Known to be below `0x2603`,
which is all the dispatch in the data channels and receivers needs — they compare
against `0x2602` / `0x2603` and pick the pre-0x2602 backing. The exact value matters
only for adding a lower bound to the enumeration gate, which deliverable 5 deliberately
omits until then. Log it during bring-up: the `HSB_LOG_INFO "hsb_ip_version={:#x}"` in
`make_backing` prints it on the first channel configuration.

## Out of scope

- Any 2510-specific behavior beyond what `hsb_lite_2510` already encodes (register
  addresses, clock profile, sensor bring-up). Real deltas land when 2510 VB1940 hardware
  verification surfaces them; the module structure is what this work delivers.
- Moving `module/core/roce_receiver_construct.cpp` / `linux_receiver_construct.cpp` into
  `hololink_module_core` proper. The split removes the reason they must be compiled
  per-module (`HsbIpPublisher` no longer includes a supplement header), but retiring
  `add_hololink_module`'s injection is a separate change.
- A Python sub-package or public header target for `leopard_vb1940_2510`.
- `module/tutorial/` re-parenting, and the tutorial docs that describe it.

## Verification

1. `cmake -S . -B "$BUILD" && cmake --build "$BUILD"` clean.
2. Filenames — `ls $BUILD/lib/hololink/modules/` shows exactly:
   ```text
   hololink_889b7ce3-65a5-4247-8b05-4ff1904c3359_2603.so   # hsb_lite
   hololink_889b7ce3-65a5-4247-8b05-4ff1904c3359.so        # hsb_lite_2510
   hololink_f1627640-b4dc-48af-a360-c55b09b3d230_2603.so   # leopard_vb1940
   hololink_f1627640-b4dc-48af-a360-c55b09b3d230.so        # leopard_vb1940_2510
   ```
   plus the DA326 and tutorial modules.
3. `ctest --test-dir "$BUILD" --output-on-failure -LE hardware` — in particular
   `hololink_module_hsb_lite_test` (unchanged, proves the split and the 2510 rename are
   invisible from outside) and the new `hololink_module_leopard_vb1940_test`.
4. `grep -rn "hsb_lite\|HsbLite" hololink_module/module/core/hsb_ip_publisher.hpp`
   returns only prose comments pointing at the subclass — no HSB-Lite type, include or
   type_id survives in the base.
5. `grep -rn "HsbLitePublisher" hololink_module/module/leopard_vb1940*
   hololink_module/module/taurotech_da326` is empty — the boards are re-parented.
6. Hardware: run `module_vb1940_player` against a 2510 board (expect the bare `.so` in
   the `module_filename` metadata and `HSB_LOG` reporting the 2510 backing class) and
   against a current board, single and stereo.
7. `pytest tests/` for the Python suite — no binding changes are expected, so any
   failure is a regression.
