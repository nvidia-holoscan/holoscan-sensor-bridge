# Release the V1 `hololink_module` versioned interfaces

## Context

`plans/hololink_module.md` describes a framework in which application code reaches HSB
device services through version-tagged interface objects (`HololinkInterfaceV1`,
`I2cInterfaceV1`, …) resolved by a service locator. Throughout development those
interfaces were explicitly *not* frozen: a V1 class could be edited in place with every
caller updated in the same commit.

Commit `2622ed9b` set `VERSION` to `2.7.0`, a public release of HSB software. That
release is the point at which the versioned interfaces present in the tree stop being
editable and become a contract: applications built against them keep working against
later modules, and new capability arrives as a new version of the affected interface
only.

Freezing is **per interface, not per release**. A release is simply the point at which a
particular interface version was frozen. Several releases from now the API will be a mix
of V1, V2, and later — only the interfaces that actually changed will carry a newer
version, and each will name the release that froze it. `2.7.0` is the first such point,
so every versioned class currently in the public headers freezes there.

Nothing about the interfaces themselves changes: `git diff 2622ed9b HEAD --
hololink_module/host/include hololink_module/module/*/include` is empty, so the tree's
current definitions *are* the 2.7.0 definitions. This work records the freeze.

## Deliverables

### 1. `Frozen:` tag on each released class declaration

Append a single tag line to the existing block comment above each versioned class:

```cpp
/* Per-board control plane. Reached via HololinkInterfaceV1::get_service
 * with the per-board instance_id "serial=<serial_number>".
 * ...
 *
 * Frozen: 2.7.0 */
class HololinkInterfaceV1
```

The tag is a bare marker — the rule it implies (methods are never added, removed,
reordered, or re-signatured; new capability becomes a new version published alongside
the old type_id) is stated once in `INTERFACES.md` rather than repeated 25 times.
Every class below already has a leading block comment, so this is an added line in an
existing comment, never a new comment block.

**21 classes in `hololink_module/host/include/hololink/module/`:**

| Header | Class(es) |
| --- | --- |
| `reactor.hpp` | `ReactorV1` |
| `logging.hpp` | `LoggingInterfaceV1` |
| `enumeration.hpp` | `EnumerationInterfaceV1` |
| `channel_configuration.hpp` | `ChannelConfigurationInterfaceV1` |
| `hololink.hpp` | `HololinkInterfaceV1` |
| `i2c.hpp` | `I2cInterfaceV1` |
| `i2c_lock.hpp` | `I2cLockV1` |
| `sequencer.hpp` | `SequencerInterfaceV1` |
| `data_channel.hpp` | `DataChannelInterfaceV1` |
| `roce_data_channel.hpp` | `RoceDataChannelInterfaceV1` |
| `coe_data_channel.hpp` | `CoeDataChannelInterfaceV1` |
| `linux_data_channel.hpp` | `LinuxDataChannelInterfaceV1` |
| `roce_receiver.hpp` | `RoceReceiverInterfaceV1`, `RoceReceiverFrameInfoV1` |
| `linux_receiver.hpp` | `LinuxReceiverInterfaceV1`, `LinuxReceiverFrameInfoV1` |
| `frame_metadata.hpp` | `FrameMetadataInterfaceV1` |
| `oscillator.hpp` | `OscillatorInterfaceV1` |
| `vsync.hpp` | `VsyncInterfaceV1` |
| `ptp_pps_output.hpp` | `PtpPpsOutputInterfaceV1` |
| `csi_converter.hpp` | `CsiConverterV1` |

**4 per-device module interfaces** (each under `hololink_module/module/<device>/include/hololink/module/<device>/`):
`HsbLiteInterfaceV1`, `LeopardVb1940InterfaceV1`, `TauroTechDa326InterfaceV1`,
`TutorialDeviceInterfaceV1`.

**Not frozen, deliberately:** implementation classes under `hololink_module/module/core/`
(`HsbLiteV1`, `I2cV1`, `RoceDataChannelV1`, `HsbLite2510RoceReceiverV1`, …) and their
`*.module_core.v1` / `*.hsb_lite_2510.v1` type_ids. Those are internal override targets,
not the application boundary, and stay editable.

### 2. New `hololink_module/INTERFACES.md`

The list of released interfaces, ordered by `VERSION`, newest first. Each section is a
**delta**: the versioned classes frozen at that release. `2.7.0` lists all 25 because it
is the first; a later release lists only what changed (e.g. just
`RoceDataChannelInterfaceV2`). The current API surface is what a reader accumulates
reading down the file.

Structure:

- Short preamble stating the freeze rule once — a listed class's definition never
  changes; new capability arrives as a new versioned class, typically deriving from the
  old one, whose implementation publishes under both type_ids so existing consumers keep
  resolving the older one. Point at
  `plans/hololink_module.md` § Versioning for the full rationale.
- A note that this file is intended as input to automated API-doc generation. Wiring
  that up is **not** in scope here; the file just needs to be machine-readable enough
  (stable heading levels, one row/entry per class, `type_id` recorded) that the
  generator has something to key on later.
- `## 2.7.0` — one entry per class: name, header path, `type_id` (or "not a locator
  service" for `I2cLockV1`, `CsiConverterV1`, and the two `…FrameInfoV1` types), and a
  one-to-two sentence description of the service.

Descriptions should be drawn from each class's existing header comment rather than
freshly invented — e.g. `ReactorV1`: "Process-wide event-loop singleton; every callback
it dispatches runs sequentially on one thread, so handlers need no locking against each
other." `SequencerInterfaceV1`: "A pre-programmed command sequence (reads, writes, polls)
that executes atomically on the FPGA, triggered by a hardware event or by software."

### 3. `plans/hololink_module.md` — versioning prose

Three spots currently assert the interfaces are unfrozen and that a freeze happens "when
the project is complete". Reword to the per-interface rule, and record that 2.7.0 froze
the V1 set:

- **~line 20** (overview): "Interfaces are **not frozen during this project** … A hard
  freeze is adopted only when the project is complete".
- **`## Versioning`, ~lines 947–963**: the "During this project" / "After the project is
  complete" pair. Replace with: each versioned interface is frozen by the release that
  publishes it; the release recorded in the class's `Frozen:` tag and in
  `INTERFACES.md` is that interface's freeze point, not a whole-API version. The V1 set
  froze at 2.7.0. A later release freezes only the interfaces whose version it bumps.
- **~line 5103** (table aside): "acceptable since interfaces are not frozen" — the C-ABI
  entry points are now released too; correct or scope the claim.

Leave the `### Versioning example — updating RoceDataChannelInterface` walkthrough alone;
it already describes exactly the per-interface model.

### 4. Python binding docstrings

The 17 `py::class_` sites that bind a released class currently pass no docstring. Add
`"Frozen: 2.7.0"` as the docstring argument so it surfaces in `help()` and in generated
Python API docs:

- `hololink_module/host/python/hololink_module_py.cpp` (15): `CsiConverterV1`,
  `FrameMetadataInterfaceV1`, `I2cLockV1`, `I2cInterfaceV1`, `SequencerInterfaceV1`,
  `OscillatorInterfaceV1`, `RoceReceiverInterfaceV1`, `DataChannelInterfaceV1`,
  `RoceDataChannelInterfaceV1`, `VsyncInterfaceV1`, `PtpPpsOutputInterfaceV1`,
  `HololinkInterfaceV1`, `EnumerationInterfaceV1`, `LoggingInterfaceV1`, `ReactorV1`.
- `hololink_module/module/hsb_lite/python/hsb_lite_py.cpp`: `HsbLiteInterfaceV1`.
- `hololink_module/module/taurotech_da326/python/taurotech_da326_py.cpp`:
  `TauroTechDa326InterfaceV1`.

The remaining released classes have no Python binding (`ChannelConfigurationInterfaceV1`,
`CoeDataChannelInterfaceV1`, `LinuxDataChannelInterfaceV1`, `LinuxReceiverInterfaceV1`,
both `…FrameInfoV1` types, `LeopardVb1940InterfaceV1`, `TutorialDeviceInterfaceV1`) —
nothing to do for them.

### 5. `RELEASE_NOTES.md`

Add a bullet under `## 2.7-GA, July 2026` → `### Updates from 2.6-GA`, next to the
existing **hololink_module** bullet: the versioned application interfaces shipped in this
release are frozen; their definitions will not change, and new capability will arrive as
new interface versions. Point at `hololink_module/INTERFACES.md`.

## Out of scope

- Automated API documentation generation from `INTERFACES.md`.
- `docs/user_guide/module_device_tutorial.mdx` and `module_imx274_tutorial.mdx` mention
  `…InterfaceV1` names but say nothing about freezing; no edit needed. Worth a skim
  during implementation in case either repeats the "not frozen" language.

## Verification

1. `git diff 2622ed9b HEAD -- hololink_module/host/include hololink_module/module/*/include`
   is empty *before* starting — confirms the headers still hold the 2.7.0 definitions
   being frozen. (Verified at planning time.)
2. Coverage check — every versioned class in the public include trees carries a tag and
   appears in `INTERFACES.md`:
   ```
   grep -rn "^class .*V1" hololink_module/host/include/hololink/module/ \
       hololink_module/module/*/include/
   grep -c "Frozen: 2.7.0" hololink_module/host/include/hololink/module/*.hpp \
       hololink_module/module/*/include/hololink/module/*/*.hpp
   ```
   Cross-check the 25 names against the `## 2.7.0` section; confirm nothing under
   `hololink_module/module/core/` picked up a tag.
3. Build: `cmake` + `make`. The C++ change is comment-only, so a clean build is the bar.
   The pybind docstring arguments are real code — confirm the Python extension compiles.
4. `ctest` for the C++ suite and `pytest` for the Python suite — no behavior change is
   expected, so any failure is a regression from the pybind edits.
5. `python3 -c "import hololink_module; help(hololink_module.HololinkInterfaceV1)"` shows
   `Frozen: 2.7.0`.
6. Read `INTERFACES.md` end to end: descriptions match the header comments they came
   from, `type_id` strings match the headers verbatim, and the newest-first ordering
   leaves an obvious place to add the next release's section.
