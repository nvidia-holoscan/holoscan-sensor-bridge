# hololink_module Tutorial — Documentation Outline

> This outline is the **single source of truth** for the tutorial. The rendered
> `docs/user_guide/module_device_tutorial.mdx` is a build artifact generated from
> it. All changes — structure *and* wording — are made here, then the `.mdx` is
> regenerated. See [Build artifact & regeneration](#build-artifact--regeneration).

## Goal

A tutorial-format user guide that walks a partner device developer through the
process to develop a new HSB device (a "hololink module") that applications can load and
use.  Include C++ code examples to clarify the actual required code.  Specifically,

- `hololink_module/module/tutorial` is a new module which defines an HsbLite derivative.
  If the existing `examples/imx274_module_tutorial.py` was executed on a system with our
  Tutorial device connected, it should run as expected with no source code changes.
  Likewise, the C++ version `examples/imx274_module_tutorial.cpp` should run on the new
  configuration without recompilation.
- `examples/example_module_tutorial.cpp` controls a status LED that is present on this
  specific implementation.  It has no data plane traffic; instead a device-specific
  interface object (e.g. "TutorialDeviceInterfaceV1") has a `set_status_led` method that
  this application calls.  The application should use the ReactorV1 interface to set an
  alarm that toggles the LED once per second.

## Build artifact & regeneration

Artifacts generated from this outline and must stay in sync; changes here should always
update the tutorial code examples.

The doc's snippets are copied verbatim from the example code, so the chain is
**outline → example code → doc snippets**.

- All are **build artifacts**. Treat them as disposable.
- **Never hand-edit an artifact.** Anything you dislike is fixed by editing this
  outline — wording as a rule under [Voice & wording rules](#voice--wording-rules),
  a term under [Preferred terminology](#preferred-terminology), or an exact sentence
  under [Pinned phrasings](#pinned-phrasings); example-code behavior as a rule under
  [Authoring notes](#authoring-notes--mechanics) — never by patching the artifact.
- On **any** change to this outline, **regenerate all artifacts from scratch**: first
  the module source and the example program, then the `.mdx` (copying its snippets
  verbatim from the freshly regenerated code). Do not diff-patch a previous render.
- After regenerating, re-verify: the example parses, the module builds, `<Tabs>`/`<Tab>`
  balanced, page registered in `fern/index.yml`, doc snippets match their source, and no
  rule is violated (e.g. grep the `.mdx` for banned legacy symbols — expect none).
- Rationale: so anyone can reproduce the doc *and* its code, and learn the method, from
  the outline alone — without the chat history that produced it.

### Generated artifacts

| Artifact | Role |
| --- | --- |
| `hololink_module/module/tutorial/include/hololink/module/tutorial/tutorial_device.hpp` | Public bespoke-service interface (`TutorialDeviceInterfaceV1`) |
| `hololink_module/module/tutorial/module_entry.cpp` | Device service impl + channel configuration + `Publisher` + `hololink_module_init` |
| `hololink_module/module/tutorial/CMakeLists.txt` | `add_hololink_module` + header target |
| `hololink_module/CMakeLists.txt` | `add_subdirectory(module/tutorial)` registration |
| `examples/example_module_tutorial.cpp` | Status-LED application (Reactor 1 Hz toggle) |
| `examples/CMakeLists.txt` | Install entry for the example |
| `docs/user_guide/module_device_tutorial.mdx` | The rendered tutorial |
| `docs/user_guide/fern/index.yml` | Nav registration |

The existing `examples/imx274_module_tutorial.{py,cpp}` are **not** artifacts of this
outline and must not be modified — the tutorial's claim is that they run unchanged.

## Authoring notes — mechanics

- Target file: `docs/user_guide/module_device_tutorial.mdx`
- Nav registration: add an entry to `docs/user_guide/fern/index.yml`, under the
  `Applications` section (after `New Sensors`).
- MDX conventions, from existing `docs/user_guide/*.mdx`:
  - No YAML front-matter, no license/SPDX header. Start with a `##` heading (H1
    comes from nav).
  - The tutorial's own code is C++; its snippets are all C++, so there are no `<Tabs>`
    (one language). The doc may still reference existing Python applications — e.g. a
    `python3 examples/...py` launch line — to show they run unchanged on the new device.
    Fence by language: ` ```cpp ` for code, ` ```sh ` for shell commands, ` ```text `
    for output/trees. No `title=` on fences.
  - Callouts are **bold inline text** (`**NOTE:** ...`), not components. The MDX system
    also offers `<Tabs>`/`<Tab>`, `<Anchor>`, and raw `<img>`, but this tutorial uses
    none of them.
  - Don't bold prose lead-in lines. The Introduction's two workflow lead-ins
    ("Instantiate the HSB-IP block within your device", "To build a driver, the typical
    workflow includes") are normal paragraphs, not bold headers.
  - Images live beside the mdx: `![alt](file.png)` or `<img src="file.png"
    alt="..." width="100%"/>`. Cross-link siblings as `[text](architecture.mdx#anchor)`.
  - Skeleton templates to mirror: `examples.mdx` (task-oriented) and
    `architecture.mdx` (code walk-through with Python/C++ tabs).
- Ship the actual working code in the locations listed above; guarantee the tutorial
  stays runnable and in sync. Every code block in the doc is copied verbatim from a
  contiguous region of a generated file.
- The module code follows the existing board modules' patterns exactly: it subclasses
  `module_core::HsbLitePublisher` and specializes only what the Tutorial device changes.
  Mirror `module/hsb_lite`, `module/taurotech_da326`, and `module/leopard_vb1940`.
- **Project code preferences win over sibling-file style.** When a sibling module's style
  conflicts with a project preference, follow the preference — never copy the sibling's
  deviation. In particular: no anonymous namespaces in `.cpp` (use `static` for
  file-scope constants/functions); ALL_CAPS for constants. (For example, `taurotech_da326`
  wraps its register constants in an anonymous namespace; the Tutorial module uses
  `static constexpr` instead.)
- Examples use current hololink_module APIs, never deprecated ones.
- The tutorial module ships no Python interfaces (no pybind bindings). Existing Python
  applications are expected to run unchanged, but we don't clutter the tutorial module
  itself with Python; its bespoke service is C++-only.
- The status-LED example is intentionally free of configuration: no command-line
  options; the board IP and discovery timeout are named constants. It runs forever (no
  signal handling or shutdown); the Reactor thread drives the toggling.

## Voice & wording rules

Generative, checkable rules that produce the prose. Add a rule here whenever a
correction generalizes; this section doubles as a reusable style guide.

- **No legacy symbols.** Never write a legacy symbol name in the doc — including
  `Enumerator`, `DataChannel`, `Hololink`, `NativeImx274Sensor`, or any legacy
  `hololink::` class. Exactly one generic statement is allowed: that
  `hololink_module` is the current API and the legacy API is deprecated. Never
  enumerate what it replaces. *Check:* grep the rendered `.mdx` for those names —
  expect zero matches. (The framework types `HsbLitePublisher`, `HololinkInterfaceV1`,
  `ReactorV1`, `Service`, `ServicePublisher`, `ConfigurableService` are current API, not
  legacy, and are fine to name.)
- **Don't explain external concepts.** Assume Holoscan (HSDK) and CUDA are
  documented elsewhere. Do not explain Holoscan operators/graphs generally, CUDA
  init/shutdown, or network protocols. Reference them; don't teach them.
- **Prefer the wording in the outline.** When the Page structure (or a pinned phrasing)
  supplies prose, use its wording rather than paraphrasing it. This is a preference, not
  a mandate: adjust only with good reason — doc flow, terminology consistency, or fixing
  a clear typo/error (e.g. `1hz` → `1 Hz`). Do not reword faithful prose for its own sake.
- **Revise in place; prefer cutting to adding.** When wording changes, replace or
  delete — do not append. Shorter is better; if a sentence can go, cut it.
- **Plain and direct.** Second person, imperative when instructing the reader
  ("subclass the publisher", "publish the service"). One idea per sentence; keep
  sentences short. State the page's purpose in one plain sentence — no throat-clearing.
- **Show, don't narrate.** Prefer a short code snippet over prose describing code.
  Explain *why* a step exists, not what each line mechanically does.
- **Specialization is the method.** Frame building a device as *specializing* the
  canonical `HsbLite` implementation — overriding only what differs — not writing a
  device from scratch. Name the specific hooks a step overrides.

## Preferred terminology

| Use | Avoid | Note |
| --- | --- | --- |
| `hololink_module` (code font) | "the adapter library", generic "the API" | the API's name |
| Holoscan Sensor Bridge board; then "the board" | "unit", "device" (used loosely) | full name on first mention, "the board" after |
| module | "plugin", "driver" (for the `.so`) | the loadable per-device shared object (`hololink_<uuid>.so`) |
| service | "interface object", "handle" | an app-facing capability fetched via `get_service` |
| fetch a service | "create"/"instantiate"/"construct" a service | services are fetched, not constructed |
| publish a service | "register", "expose" | what a module's `Publisher` does so apps can fetch it |
| specialize `HsbLite` | "extend", "fork", "write from scratch" | subclass `HsbLitePublisher`, override only what differs |
| bespoke service | "custom API", "extra interface" | a device-specific service (`TutorialDeviceInterfaceV1`) |
| the Tutorial device | "our device", "the demo board" | the example board this tutorial builds |
| the adapter | "the manager", "the loader" | the discovery/entry object (`Adapter`) |
| the reactor | "the timer", "the scheduler" | the host event loop (`ReactorV1`) that runs alarms |

## Pinned phrasings

Exact sentences where wording is load-bearing. Regenerate them verbatim; if one
should change, change it here first.

- (none currently.)

## Reference code the samples are drawn from

- Minimal module (identity-only specialization): `hololink_module/module/hsb_lite/module_entry.cpp`
- Bespoke service + channel-config override: `hololink_module/module/taurotech_da326/` (`module_entry.cpp`, `include/.../taurotech_da326.hpp`, `CMakeLists.txt`)
- Three-sensor channel configuration (data_plane 0, one SIF per sensor): `hololink_module/module/leopard_vb1940/module_entry.cpp`
- Canonical publisher + channel configuration + per-sensor addressing: `hololink_module/module/core/hsb_lite_publisher.hpp`, `hsb_lite_default.hpp`
- Register I/O + `locator_id` + `start`/`stop`/`reset`: `hololink_module/host/include/hololink/module/hololink.hpp`
- Service / publisher framework: `hololink_module/host/include/hololink/module/service.hpp`, `publisher.hpp`, `module.hpp`, `adapter.hpp`
- Reactor (alarms): `hololink_module/host/include/hololink/module/reactor.hpp`; C++ `ReactorV1::get_service(adapter.host_publisher()->self_module())`
- Module build helper: `hololink_module/cmake/HololinkModule.cmake` (`add_hololink_module`)
- The existing application the module runs unchanged: `examples/imx274_module_tutorial.cpp`

The Tutorial device's fixed FPGA UUID is `d3061b3b-85b0-4096-ba57-296d2418477f`. Like the
canonical `hsb_lite` module, it uses the framework's default compat-id (do **not** pass
`NO_COMPAT_SUFFIX` — that is a documented exception for firmware that doesn't advertise a
compat-id), so the module builds as `hololink_<uuid>_<compat>.so`. The compat-id is the
FPGA's IP version; the host matches both UUID and compat-id when loading the module.

---

## Page structure

- Introduction
  - Users building their own HSB-IP based devices can use this tutorial to learn how to
    build an `hololink_module` driver.  Applications use these drivers to access the
    functionality on that board, both for generic services and for things that are only
    present on your specific device.  Typical device development workflow follows.

    Instantiate the HSB-IP block within your device
      - Generate a new UUID for your device (e.g. `uuidgen`)
      - Update configuration parameters, e.g. to indicate how many sensor interfaces and
        network ports your device has
      - Add or adjust any peripherals that are specific to your device
      - Ensure that per-device parameters, like MAC IDs and serial numbers are
        appropriately configured
      - Once the device is deployed, and the host is properly connected, you should see
        reasonable data using the `hololink-enumerate` command.  For example, an HsbLite
        device produces these messages, one for each network port:

```sh
# hololink-enumerate
mac_id=3A:31:1D:1E:24:AA hsb_ip_version=0x2606 fpga_crc=0x0 ip_address=192.168.0.2 fpga_uuid=889b7ce3-65a5-4247-8b05-4ff1904c3359 serial_number=10060032828115 interface=enP5p3s0f0np0 board=hololink-lite
mac_id=3A:31:1D:1E:24:AB hsb_ip_version=0x2606 fpga_crc=0x0 ip_address=192.168.0.3 fpga_uuid=889b7ce3-65a5-4247-8b05-4ff1904c3359 serial_number=10060032828115 interface=enP5p3s0f1np1 board=hololink-lite
```

    To build a driver, typical workflow includes
      - Identify the variations in your device from the standard HsbLite configuration
        - You'll leverage the existing HsbLite implementation for everything that
          doesn't change
      - Copy from a module template into a new directory for your device
        - `hololink_module/module/<your-device>`
        - Use the module we're building in this tutorial for your template, `hololink_module/module/tutorial`
      - Update the build instructions with the name and UUID for your device
      - Update the software with your configuration parameters (e.g. number of sensor
        interfaces and network ports)
      - Override any `HsbLite` service implementations as necessary
      - Create services for any new features
      - Build and install your module
      - If your device supports a sensor configuration that's already
        supported by an example application, you probably can use it without
        modification.  You shouldn't even have to recompile that application code.
      - For your new sensors or features, build an example application that demonstrates
        that feature.  Follow the [IMX274 Module Tutorial](module_imx274_tutorial.mdx) for instructions
        on this.
      - Add testing for your device to the `tests` directory; this way your users can
        continue to access your validation suite
    - Once your device is validated, you can publicly publish your updates for customers
      to use.

    As new versions of HSB-IP and host software are released, your module driver should
    continue to work as is.  Guidelines on what needs to be updated and when are
    provided below.

- `hololink_module/module/tutorial`
  - Imaginary implementation
  - Same as `HsbLite`, with these variations:
    - UUID: d3061b3b-85b0-4096-ba57-296d2418477f
    - 3 sensor ports: 0 and 1 are camera and 2 is IMU
    - One network interface
    - A single register (TUTORIAL_DEVICE_STATUS, 0x42348000) with a bit (STATUS_LED_BIT)
      that turns an LED on or off
  - We'll show how you build a module based on `HsbLite`, apply sensor port
    configurations, and provide an API for setting that on-board LED.

- TutorialDevice module
  - This module is an example you can copy to build your own device: **replace the word
    "tutorial" with your device's name throughout.** (Render this instruction in bold.)
  - Module source code is in `hololink_module/module/tutorial` directory.
    `hololink_module/CMakeLists.txt` says `add_subdirectory(module/tutorial)` to find
    this code.  Convention is to keep those `add_subdirectory` calls in alphabetic
    order.
  - `hololink_module/module/tutorial/CMakeLists.txt` includes a call to
    `add_hololink_module`.  Show how this trains the build system to build a shared
    object e.g. `hololink_<UUID>_<compat-id>.so`.  Note that the compat-id, by default,
    comes from the library--you're always building a module for that specific version of
    HSB-IP.  We'll include compatibility guidelines later.
  - `hololink_module/module/tutorial/module_entry.cpp` — refer to the in-tree file for
    complete details; the key parts follow, top-down.
    - hololink_module_init.  This is the module's initialization entry point — the host
      calls it to initialize the library when the application loads the module.  Explain
      that TutorialDevicePublisher knows how to satisfy application requests for services.
      The `TutorialDevice*` types should be renamed for your device.
    - Explain that TutorialDevicePublisher handles all application `get_service`
      requests, normally just passing them to the HsbLite implementation (our
      superclass).  Our device specific behavior-- the overrides from normal HsbLite
      behavior-- are handled by this object.  For the basic module, show only
      `module_name` and `publish_channel_configuration`; `construct_overrides` comes
      later, under board-specific extensions.
    - Explain TutorialDevicePublisher::module_name; this is for documentation and logging
      only
    - Explain TutorialDevicePublisher::publish_channel_configuration.  Show how
      `TutorialDeviceChannelConfigurationV1` is used to redefine `use_sensor`, which is
      how we pass the sensor and network configuration information back to the
      framework.  Note that publish_channel_configuration is not unique to a specific
      device--the same `TutorialDeviceChannelConfigurationV1` instance is used for every
      device found.
    - Because we don't have any additional application-visible variations from HsbLite --
      the functionality is all based on HSB-IP -- no further driver software work is
      necessary for the basic module.  Build the module with cmake into a build directory
      under `/tmp`; the module `.so` is staged under `$BUILD_DIR/lib/hololink/modules/`,
      where applications find it:

```sh
BUILD_DIR=/tmp/hololink-build
cmake -S . -B "$BUILD_DIR" -G Ninja
cmake --build "$BUILD_DIR" --target tutorial
```
    - Applications dynamically load hololink module drivers, so you won't have
      to recompile those in order to work with your new device.  The application would
      observe our new UUID during enumeration and use that to find your new module driver.
      In other words, at this point running `module_imx274_player` or `python3
      examples/module_linux_imx274_player.py` should work unmodified.

- Board-specific extensions
  - Now that existing applications run, add functionality that is specific to your device.
    For the Tutorial device that's the status LED, exposed through a bespoke service.
  - Show `TutorialDeviceInterfaceV1` — these are the APIs that applications can call — and
    the `TutorialDeviceV1` implementation of `set_status_led`.  State that the interface
    goes in the module's public header
    `hololink_module/module/tutorial/include/hololink/module/tutorial/tutorial_device.hpp`
    so applications can include it.
    - For `TutorialDeviceV1`: prefix the walkthrough by noting the transition from the
      public interface `TutorialDeviceInterfaceV1` — visible to any application that includes
      the header — to its implementation `TutorialDeviceV1`, which is only visible within the
      module itself.  Then: `configure()` sets the instance up for a specific device —
      the framework keeps a separate instance per serial number, so different devices each
      get their own; once set up, calls to `set_status_led` control the on-board LED
      (pull-down: clearing bit 0 of `TUTORIAL_DEVICE_STATUS` turns it on, setting it turns
      it off).  In the shown snippet, omit the `using` declarations — framework boilerplate
      with no tutorial value (keep them in the source file).
    - Below the interface, briefly note `type_id` (the unique-per-class identity
      `get_service` uses for type safety) and `locator_id` (builds the cache key that
      finds a device's object — a `;`-separated `name=value` string, here
      `serial=<serial_number>`).
  - `TutorialDevicePublisher::construct_overrides` is the right place to construct
    instances of `TutorialDeviceV1`, where the implementation of `set_status_led` goes.
    Show it framed inside `class TutorialDevicePublisher ... { ... };` (with a `// ...`
    elision for the base methods) so it's clear it's a method of that class.
    - Explain how `Publisher::has_type_id<TutorialDeviceV1>(type_id)` checks to see if
      this is the object type we know about
    - Explain how `ServicePublisher<TutorialDeviceV1>(shared_from_this())
      .publish(instance_id, impl)` works--we have an individual instance for each
      distinct instance_id (which is usually just the device serial number).  For
      example, the same HsbLite device can appear on multiple network ports, so this
      convention allows us to map both of those interfaces back to the same object
      here.
    - Explain how once an object is sent to `publish`, we won't be asked to create it
      again (unless that object is invalidated, e.g. device disconnection).  Later
      calls start with the cache that publish maintains.
    - Returning `true` means that the object is correctly constructed and searching
      for any more handlers is unnecessary.
  - If your application needs specific functions on your board, like our `set_status_led`
    method above, then it must fetch a handle to the `TutorialDeviceInterfaceV1` instance.
    In this case, your application is now tightly coupled to your device-- running that
    application with another device would trigger an exception when it tries to get that
    unsupported `TutorialDeviceInterfaceV1` implementation.

- Status-LED application (C++)
  - Intro: a small application that has no data plane at all — it just drives the LED via
    the bespoke service and the reactor.
  - Snippet: discover the board (`Adapter::get_adapter()`, `wait_for_channel`), bring up
    the control plane (`HololinkInterfaceV1::get_service`, `start()`, `reset()`), fetch
    `TutorialDeviceInterfaceV1`.
  - Snippet: get the reactor (`ReactorV1::get_service(adapter.host_publisher()->self_module())`),
    define the 1 Hz `tick` callback that flips the LED and re-arms itself with
    `add_alarm_s(BLINK_PERIOD_S, tick)`, arm the first fire, then run.
    - Explain: reactor alarms are one-shot, so a periodic timer re-arms from inside its
      own callback; the reactor runs the callback on its own thread.
  - Snippet: block the main thread forever so the reactor keeps blinking — a `sleep(60)`
    in a `for (;;)` loop. Assume the program runs forever — no signal handling and no
    shutdown.

- Background.  Briefly explain
  - the protocol around hololink_module_init
  - the module-specific service cache and how some objects are singleton and some are
    device specific (e.g. by serial number)
  - HsbLite is a core implementation because it implements the host side behavior for
    HSB-IP.  Virtually any implementation there can be overridden in a device specific module
  - Functional perspective
    - When an HSB device is powered on, it periodically publishes an enumeration
      message.  These are typically sent at 1 Hz.  This enumeration message includes a
      UUID that identifies the device and a `compat-id` field, providing specific
      information about version compatibility.
    - The application uses `hololink_module::Adapter` to listen to these bootp messages.
      Each enumeration message is deserialized into a structure which is referred to as
      enumeration metadata.
    - The received UUID and `compat-id` fields are used to locate the hololink module
      associated with this device.  This module is a driver (as a shared object) and is
      typically named `hololink_<UUID>_<compat-id>.so`.  When found, it's loaded and
      initialized, and the received enumeration metadata is passed to it.  At this
      point, the module can adjust or enhance enumeration data as necessary.
    - The resulting enumeration metadata is passed back to the application.  If the host
      doesn't have a compatible module that it can load, the enumeration metadata from
      the deserialized network message is sent (unmodified) to the application.
    - The application uses this enumeration metadata to fetch specific services.  For
      example, to reset the board, the application calls `HololinkInterfaceV1::reset`.
      To get a pointer to the `HololinkInterfaceV1` object, use
      `HololinkInterfaceV1::get_service(enumeration_metadata)`.  (Insert code sample
      here.)  This call to `get_service` fetches the possibly customized
      implementation from the device module-- so the call to `reset` does what's
      necessary for that specific device.  (Note that not all services are found using
      enumeration metadata; some services are found with other criteria.)
    - The publisher's `construct_service` chain: services are constructed lazily on first
      fetch; override any of the methods it calls to substitute your own implementation
      instead of the HsbLite implementation.  You can even override `construct_service`
      itself if necessary.
    - All parts of the application depend on these services.  For example, to configure
      the data plane for the device to use RoCE messaging, RoceReceiverOp fetches
      `RoceDataChannelInterfaceV1` and calls its configuration methods.  An application
      that does not use RoCE signalling wouldn't fetch this object; so not all devices
      need to support it.  If an application asks for a service that the module doesn't
      support, the program would terminate with an error message showing that the
      service isn't supported by that device.
    - Some APIs are device specific, e.g. activating a feature or component that is only
      instantiated on that device.  The device module can define a device specific
      interface (e.g. `HsbLiteInterfaceV1`) with APIs that access and control those
      functions.  Applications that want to use those features can get this
      device-specific service and call its methods-- with the recognition that those
      calls tightly couple the application to that device.
  - Version management
    - The specification of `compat-id` is a specific set of registers, peripherals, and their
      behaviors; any HSB-IP with the same `compat-id` is register compatible.  If a
      breaking change occurs in HSB-IP registers, the `compat-id` is updated, which guarantees
      that existing deployed systems will not try to use the incompatible version.
    - Explain how services exported by the module are versioned; explain how this allows
      us to build new modules later on (e.g. to support new features), and as long as
      the module still produces services matching the interface version that the
      application requires, the application will continue to work.

