# hololink_module Tutorial — Documentation Outline

> This outline is the **single source of truth** for the tutorial. The rendered
> `docs/user_guide/module_imx274_tutorial.mdx` is a build artifact generated from
> it. All changes — structure *and* wording — are made here, then the `.mdx` is
> regenerated. See [Build artifact & regeneration](#build-artifact--regeneration).

## Goal

A tutorial-format user guide that walks an application developer through a
single-channel IMX274 player built on `hololink_module`. Include a Python / C++
toggle showing code samples for the reader's chosen environment. The doc is based
on `hololink_module` APIs.

## Build artifact & regeneration

Three artifacts are generated from this outline and must stay in sync:
the rendered doc, and the two example programs
(`examples/imx274_module_tutorial.py`, `examples/imx274_module_tutorial.cpp`).
The doc's snippets are copied verbatim from the example programs, so the chain is
**outline → example programs → doc snippets**.

- All three are **build artifacts**. Treat them as disposable.
- **Never hand-edit an artifact.** Anything you dislike is fixed by editing this
  outline — wording as a rule under [Voice & wording rules](#voice--wording-rules),
  a term under [Preferred terminology](#preferred-terminology), or an exact sentence
  under [Pinned phrasings](#pinned-phrasings); example-code behavior as a rule under
  [Authoring notes](#authoring-notes--mechanics) — never by patching the artifact.
- On **any** change to this outline, **regenerate all three from scratch**: first
  the two example programs, then the `.mdx` (copying its snippets verbatim from the
  freshly regenerated examples). Do not diff-patch a previous render. The two
  examples must mirror each other one-for-one so the Python/C++ tabs line up.
- After regenerating, re-verify: examples parse/build, `<Tabs>`/`<Tab>` balanced,
  page registered in `fern/index.yml`, doc snippets match the examples, and no rule
  is violated (e.g. grep the `.mdx` for banned legacy symbols — expect none).
- Rationale: so anyone can reproduce the doc *and* its examples, and learn the
  method, from the outline alone — without the chat history that produced it.

## Authoring notes — mechanics

- Target file: `docs/user_guide/module_imx274_tutorial.mdx`
- Nav registration: a `- page:` / `path: ../module_imx274_tutorial.mdx` entry in
  `docs/user_guide/fern/index.yml`, under `Getting Started` (after `Examples`).
- MDX conventions, from existing `docs/user_guide/*.mdx`:
  - No YAML front-matter, no license/SPDX header. Start with a `##` heading (H1
    comes from nav).
  - Dual-language via `<Tabs>` + `<Tab title="Python" language="python">` /
    `<Tab title="C++" language="cpp">`; blank line inside the tab around fenced
    code. Fence by language: ` ```python `, ` ```cpp `, ` ```sh `. No `title=` on
    fences.
  - Callouts are **bold inline text** (`**NOTE:** ...`), not components. Only
    components in use are `<Tabs>`, `<Tab>`, `<Anchor>`, raw `<img>`.
  - Images live beside the mdx: `![alt](file.png)` or `<img src="file.png"
    alt="..." width="100%"/>`. Cross-link siblings as `[text](architecture.mdx#anchor)`.
  - Skeleton templates to mirror: `examples.mdx` (task-oriented) and
    `architecture.mdx` (code walk-through with Python/C++ tabs).
- Ship the actual working example in `examples/`: `imx274_module_tutorial.py` and
  `imx274_module_tutorial.cpp`. The doc's snippets are copied verbatim from these,
  so the tutorial stays runnable and in sync.
- Use `hololink_module.operators.RoceReceiverOp`.
- The example is intentionally free of configuration: no command-line options; a
  single hard-coded camera mode, board IP, and discovery timeout as named constants.
- Don't include unused controls (e.g. "condition" in the pipeline)
- Examples use current Holoscan APIs, never deprecated ones. In particular, a
  single-channel display pipeline consumes no frame metadata, so do not enable it
  (no `enable_metadata` / `metadata_policy`, and never the deprecated
  `is_metadata_enabled` property setter). `metadata_policy` only matters when
  multiple sources feed one operator.
- Note that the IMX274 camera is a stereo pair device; phrases like "a single IMX274
  camera" are confusing; instead say "an IMX274 camera" and be clear that for IP
  address 192.168.0.2, we'll be accessing the first camera on the device.
- Omit the "Next steps" section for now — its candidate links all support legacy
  approaches.

## Voice & wording rules

Generative, checkable rules that produce the prose. Add a rule here whenever a
correction generalizes; this section doubles as a reusable style guide.

- **No legacy symbols.** Never write a legacy symbol name in the doc — including
  `Enumerator`, `DataChannel`, `Hololink`, `NativeImx274Sensor`, or any legacy
  `hololink::` class. Exactly one generic statement is allowed: that
  `hololink_module` is the current API and the legacy API is deprecated. Never
  enumerate what it replaces. *Check:* grep the rendered `.mdx` for those names —
  expect zero matches.
- **Don't explain external concepts.** Assume Holoscan (HSDK) and CUDA are
  documented elsewhere. Do not explain Holoscan operators/graphs generally, CUDA
  init/shutdown, or network protocols. Reference them; don't teach them.
- **Revise in place; prefer cutting to adding.** When wording changes, replace or
  delete — do not append. Shorter is better; if a sentence can go, cut it.
- **Plain and direct.** Second person, imperative when instructing the reader
  ("fetch the service", "run the example"). One idea per sentence; keep sentences
  short. State the page's purpose in one plain sentence — no throat-clearing.
- **Show, don't narrate.** Prefer a short code snippet over prose describing code.
  Explain *why* a step exists, not what each line mechanically does.

## Preferred terminology

| Use | Avoid | Note |
| --- | --- | --- |
| `hololink_module` (code font) | "the adapter library", generic "the API" | the API's name |
| Holoscan Sensor Bridge board; then "the board" | "unit", "device" (used loosely) | full name on first mention, "the board" after |
| service | "interface object", "handle" | an app-facing capability fetched via `get_service` |
| fetch a service | "create"/"instantiate"/"construct" a service | services are fetched, not constructed |
| the adapter | "the manager", "the loader" | the discovery/entry object (`Adapter`) |
| camera driver / sensor driver | "the camera service" | `Imx274Cam`; it is **not** a service |
| control plane | "the connection" | what `HololinkInterfaceV1` provides |

## Pinned phrasings

Exact sentences where wording is load-bearing. Regenerate them verbatim; if one
should change, change it here first.

- **Positioning (Overview):** "Applications using the new `hololink_module` can talk
  to different HSB devices transparently, unlike those applications using the legacy
  API." Attach it to the opening purpose paragraph — not a standalone line.
- **Services model (Overview):** "An application accesses the capabilities it needs
  as *services* instead of constructing device classes directly."
- **Sensor-driver caveat (Application initialization):** "The sensor driver is not
  a service: it is linked directly into your application, which is tightly coupled
  to this sensor."
- **Pipeline ordering (Application pipeline):** "While data flow starts with the
  receiver operator, we can't construct it without knowing some configuration
  (specifically the receiver buffer size) that comes from `csi_to_bayer`."

## Reference code the samples are drawn from

- **C++ single-channel RoCE player (canonical):** `examples/module_imx274_player.cpp`
- **Python single-channel player (software receiver):** `examples/module_linux_imx274_player.py`
- **Python RoCE call shape (from quad example):** `examples/module_quad_imx274_player.py:148-157`
- Camera driver: `hololink_module/host/sensors/imx274/imx274_cam.cpp`, `hololink_module/python/sensors/imx274/imx274_cam.py`
- RoCE receiver op: `hololink_module/host/operators/roce_receiver_op.cpp`
- Service locator / versioning: `hololink_module/host/include/hololink/module/service.hpp`, `hololink.hpp`, `roce_receiver.hpp`

---

## Page structure

- Introduction
  - Explain the purpose and goal of this page; end that paragraph with the pinned
    Positioning sentence, connected to the purpose (no standalone deprecation line)
  - Services model
    - supports applications working with different implementations and
      versions of HSB devices, no recompilation necessary
    - application APIs are presented as services
    - app fetches these via `Interface::get_service`; they aren't constructed directly by
      application code
      - get_service often uses enumeration metadata to identify the specific instance

- IMX274 player tutorial
  - Configuration
    - Very brief review of the equipment in use
    - Assume that user has built and run the demo docker container per the user guide
  - Application pipeline
    - Open with the pinned Pipeline-ordering sentence, then the actionable lead-in:
      build/configure csi_to_bayer, read the frame size, then build the receiver.
      State this ordering rationale once here; don't repeat it on the receiver
      operator.
    - HSB specific operators
      - Image conversion (CsiToBayerOp)
        - unpacks the board's CSI-formatted bayer video into 16-bit samples
        - configuring it against the camera driver fixes the per-frame buffer size
      - Receiver operator
        - built after the buffer size is known
        - RoceReceiverOp "configures itself to listen
          for data on that specifically enumerated channel"; don't enumerate which sub-services it
          resolves internally
      - Naive image processing (ImageProcessorOp)
        - a hololink_module operator, NOT a standard Holoscan operator
        - provides a naive ISP of bayer data (e.g. RGGB) using the GPU
      - Demosaic (standard Holoscan BayerDemosaicOp)
        - Converts e.g. RGGB to RGBA
      - Visualization (standard Holoscan HolovizOp)
  - Application initialization
    - Open with: the main program is responsible for finding the board, configuring
      it, then running the pipeline described above.
    - Adapter object
    - Device enumeration
    - Device setup
      - start
        - establishes a socket for control plane communication
        - include a comment to clarify that this is necessary for any I/O to the device
          to work
      - reset
        - The framework does not affect the state of the device without a request from the
          application
        - so the application must request `reset` in order to ensure the device is in a
          known state
    - Sensor instantiation
      - sensor configuration
      - Note that the sensor driver is not a service and is linked directly to the
        application code.  The application is tightly coupled to this sensor
        device.
      - include a comment to clarify that camera configuration calls probably write to
        device registers and therefore must follow the hololink.reset call
    - Run the HSDK application
  - Build and run the application
- Background
  - Services allow application to work with different HSB devices and versions without
    recompiling
    - services are versioned, e.g. `HololinkInterfaceV1`.  Once an interface is
      published, it is not allowed to change.  Instead, new APIs would be specified in a
      e.g. V2 instance (where the V2 interface usually just extends the V1 interface).
      Modules normally continue to publish older supported versions of services in order
      to support already deployed applications
    - services are implemented in device-specific shared libraries
    - device enumeration (bootp) provides a device UUID
    - that UUID and "compat-id" are used to locate the right shared object file
    - the shared object provides the implementation of the requested interface
      - this is done by a runtime binding for the virtual methods in the interface
        object
    - requesting an interface that isn't implemented on this device results in an exception
      - unless "allow_null=true" which indicates the application has error handling for
        this case
- (Next steps section omitted for now — its candidate links all support legacy
  approaches.)
