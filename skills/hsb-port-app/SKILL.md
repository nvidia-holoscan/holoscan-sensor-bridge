---
name: hsb-port-app
description: Create or port Hololink example apps across receiver types, cameras, and pipelines. Not for running apps (/hsb-app), FPGA flashing (/hsb-flash), or hardware setup (/hsb-setup).
license: Apache-2.0
allowed-tools: Read,Write,Edit,MultiEdit,Grep,Glob
metadata:
  author: "Holoscan Team <holoscan-team@nvidia.com>"
  version: "1.0.0"
  team: holoscan
  tags:
    - holoscan
    - holoscan-sensor-bridge
    - hsb
    - example-porting
    - app-porting
---

# Hololink Example App Builder

**EXECUTION ORDER: Entry point → Step 0 → Step 1 → Step 2 → Step 3 → Step 4 (4.1→4.2→4.3→4.4) → Step 5. Do not reorder or skip steps.**

## Entry point

**FIRST ACTION: Begin here. Execute A, B, C, D in order — do not skip.**

**A — BRANCH (help):** If `$ARGUMENTS` contains `--help` or `-h` → print the block below and stop. This is a help exit, not success.

```
USAGE:  /hsb-port-app [DESCRIPTION] [FLAGS]
FLAGS:  --allow-unvalidated-layout  Continue when CSI layout cannot be derived
        --dry-run                   Show synthesized content without writing to disk
        --help                      Show this message
EXAMPLES:
  /hsb-port-app "port imx274_frame_validation.py to FuSa CoE"
  /hsb-port-app "make a linux version of the vb1940 PVA CRC validation app"
  /hsb-port-app --dry-run "create an imx274 FuSa CoE nvcomp CRC validation app"
```

Otherwise → proceed immediately to B.

**B — Extract:** If `$ARGUMENTS` is present, extract from it; otherwise ask: "What would you like to port or create?" Record: source app (optional), target camera, target receiver, target pipeline, and flags (`--allow-unvalidated-layout`, `--dry-run`). Flags added during Step 1 clarification are also accepted. Then proceed immediately to C.

**C — BRANCH (axis gate):**
- If target camera, receiver, AND pipeline are all known → proceed immediately to D.
- If any of the three is missing → ask once (this is round 1), wait for response, record it, then proceed to D. Do not guess.

**D — Proceed to Step 0.**

---

## Step 0 — Discover what exists

Use the Glob tool (`examples/*.py`) to build a live map of (camera, receiver, pipeline) triples using the naming convention below. Do not use Bash for this step.

| Prefix | Receiver |
|--------|----------|
| *(none)* | RoCE (default) |
| `linux_` or `linux_coe_` | Linux UDP (`linux_` is default; `linux_coe_` for CoE variants) |
| `fusa_coe_` | FuSa CoE |
| `gpu_roce_` | GPU RoCE |

- **BRANCH (Glob result) — next step is always Step 1 in both non-stop cases:**
  - Glob returns files → live map ready → **go to Step 1**.
  - Glob returns nothing → check `references/receiver-reference.md` (must have receiver table) and `references/pipeline-reference.md` (must contain `compute_imx274_csi_layout` pattern). If valid → **go to Step 1** (Step 4.1 will draw wiring from reference file code blocks directly — no example app). Otherwise → **stop exit**.

---

## Step 1 — Decompose the target

Record all of the following explicitly:
1. Source app name (if porting — needed for Step 4.2c; absent for create-from-scratch)
2. Target camera
3. Target receiver
4. Target pipeline concern (e.g. player, frame_validation, pva_crc_validation)
5. Pipeline type: **derived from item 4** — audio if the pipeline name is `audio_recorder`, `audio_player`, or starts with `audio_`; video otherwise.

> **BRANCH — REQUIRED GATE (exit of Step 1, entry to Step 2):**
> - If items 2, 3, and 4 are all known → derive item 5 from item 4, then **go to Step 2**.
> - If any of items 2, 3, or 4 is missing → ask to clarify, update, re-evaluate. Budget: 4 prompts total — Entry C is prompt 1; up to 2 more individual prompts here; then 1 combined prompt for all still-missing axes. If still incomplete → **terminate the skill**: "Unable to determine target — re-invoke `/hsb-port-app`." Do not guess.

For multi-concern targets: split pipeline name by known keywords (`player`, `latency`, `frame_validation`, `crc_validation`, `pva`, `nvcomp`, `loopback`, `audio`) — each match is a concern (e.g. `nvcomp_crc_validation` → {nvcomp, crc_validation}). Run Step 2 for each, merge in Step 4.2e. See `references/worked-example.md`. Proceed to Step 2.

---

## Step 2 — Find the best reference per concern

*(Empty live map: skip priority matching — use receiver-reference.md for wiring and pipeline-reference.md for operators; stop if operators absent.)*

Evaluate priorities in order; use the **first match** and stop. Camera+receiver: one selection total. Pipeline: once per distinct concern. A single file may match both — read it once.

**Camera + receiver setup** (select a reference app — prefer player-type for simplicity):
1. Same camera + same receiver
2. Same camera + different receiver — receiver wiring differs, camera init reusable
3. Different camera + same receiver — camera init differs, receiver wiring reusable

**Processing pipeline** (for each distinct concern):
1. Same pipeline + same receiver + same camera
2. Same pipeline + same receiver + different camera
3. Same pipeline + different receiver + same camera
4. Same pipeline + different receiver + different camera — structural template only

**MUST read both** selected references before synthesis. *(Skip if renamed-copy — no synthesis.)*

**BRANCH (exact match):** If an exact match exists in the live map — same camera, receiver, AND pipeline as all three target axes — ask: "An exact match exists at `examples/<file>.py`. Do you want to: (1) use the existing file, (2) create a renamed copy, or (3) create a new variant with different behavior?" Then:
- (1) Use existing → output its path and stop (exact-match exit — not an error).
- (2) Renamed copy → ask: "What would you like to name the new file?" Validate: matches `^[A-Za-z_][A-Za-z0-9_]*\.py$`, no path separators or `..`. Glob-check `examples/<user_name>` — if exists: "Overwrite?" If `--dry-run`: display copy content with new name and stop — no file written. Otherwise: copy (Write tool). Skips Steps 3–4. Proceed to **Pre-Step 5** and Step 5 (Check 4 skips; keep `crc_validation` if copying CRC app).
- (3) New variant → ask for a distinct filename. Validate: matches `^[A-Za-z_][A-Za-z0-9_]*\.py$`, no path separators or `..`. Continue to Step 3.

**If no match at any priority:** use closest structural template (prefer: same receiver > same camera > same pipeline > simplest; ties: most matching axes, then alphabetical). Document choice. Flag hardware validation. Any absent operators must be sourced from `references/pipeline-reference.md`.

**Proceed to Step 3.**

---

## Step 3 — Identify the seam (analysis only — no writes)

Record the seam boundaries:
- **CSI layout, memory pool sizes, camera mode enum** — camera-specific. Note whether camera likely exposes `get_csi_frame_layout()` (vb1940: yes; imx274: no); Step 4.3 confirms.
- **Operator classes, flow graph, post-run analysis** — pipeline-generic.

Record: (1) which boundaries apply, (2) `get_csi_frame_layout()` availability, (3) which values are camera vs pipeline.

**Proceed to Step 4** (reads required reference files, then executes 4.1 → 4.2 → 4.3 → 4.4 in order — do not skip).

---

## Allowed commands

**DO:**
- Read files: use Read, Grep, Glob tools
- Write/Edit output file: Write to create `examples/<new_file>.py` or copy (renamed-copy); Edit to fix during Pre-Step 5 and Step 5
- Discover examples: use Glob(`examples/*.py`) in Step 0
- Run sanity checks: use Grep tool in Step 5

**DO NOT:**
- Run Bash, docker, git, or any shell command
- Execute, import, or test the synthesized file (`python3`, `exec()`, `subprocess`, hololink APIs)
- Write files outside `examples/`

---

## Step 4 — Synthesize

Read reference files first:
- `references/receiver-reference.md` — **BRANCH: missing → stop, report which file is missing; present → continue**
- `references/pipeline-reference.md` — **BRANCH: missing → stop, report which file is missing; present → continue**
- `references/worked-example.md` — optional; read if present for 3-concern guidance
- **Both required files present → proceed to Step 4.1**

**Pre-write check (before Step 4.1):** If `--dry-run` was set, skip the overwrite check and proceed to Step 4.1. Otherwise: use the filename set in Step 2 option (3) if established; otherwise construct from Step 1 items via the File naming convention (e.g. `fusa_coe_imx274_frame_validation.py`). Glob-check `examples/<new_file>.py`.
- If file exists: ask the user "This will overwrite `examples/<new_file>.py`. Continue?" If no → **stop exit**. If yes → proceed to Step 4.1.
- Otherwise: proceed to Step 4.1.

**Before writing any code — restate the target:** Confirm: "I am synthesizing `examples/<new_file>.py` for camera=`<camera>`, receiver=`<receiver>`, pipeline=`<pipeline>`."

**Step 4.1** — **MUST copy** `__init__`, `compose()` wiring, and `main()` setup from the camera+receiver reference. If no example app exists, use receiver-reference.md code blocks for the target receiver instead. Proceed to Step 4.2.

---

**Step 4.2** — **MUST insert** pipeline operators and flow graph from the pipeline reference:

- 2a. Copy all pipeline operator classes and flow graph wiring.
- 2b. Replace camera-specific sizes/formats with target camera values. Priorities 1–2: read from the reference. Priority 3 (different-camera): find target camera app in live map for init values; if live map empty, read from camera Python class constructor. CSI layout only in Step 4.3.
- 2c. Run two separate searches:
  - **Search 1**: Strip receiver prefix, find the **longest known camera name** from the live map matching a prefix of the remaining filename (e.g. `imx274` from `imx274_nvcomp_crc_validation`). If no match: use first `_`-segment. If extracted token = target camera (Step 1 item 2): record "source camera equals target" — no rename needed. For create-from-scratch: apply to pipeline reference filename. Otherwise: grep and replace extracted token with target camera name.
  - **Search 2**: *(Skip if no source app, or source app not found/unreadable.)* Grep for hardcoded numeric values from the source app's `__init__` and camera setup blocks. Cross-check target camera+receiver reference — replace only values that differ.
- 2d. *(FuSa CoE video only — skip if pipeline type is audio)* Insert `PackedFormatConverterOp` between capture and downstream operators.
- 2e. *(Multi-concern only)* Merge operator sets: shared operators include once. Detect conflicts: list `add_flow(src, dst)` calls (or equivalent connection patterns) from each reference's compose(); a conflict is same `src`, different `dst` across references. Resolve by concern priority (CRC > PVA > frame_validation > latency > player; unlisted rank below player). On tie: keep first-selected wiring. Document each conflict as inline comment. Verify all ports connect.

**BRANCH — Step 4.2 routing:**

| FuSa CoE video? | Multi-concern? | Order | Next |
|---|---|---|---|
| Yes | Yes | 2a→2b→2c→2d→2e | Step 4.3 |
| Yes | No  | 2a→2b→2c→2d    | Step 4.3 |
| No  | Yes | 2a→2b→2c→2e    | Step 4.3 |
| No  | No  | 2a→2b→2c       | Step 4.3 |

---

**Step 4.3** — Resolve CSI layout seam. *(Skip to Step 4.4 if pipeline concern does not contain `validation`. Runs for frame_validation and crc_validation; Checks 5/6 scope to filename containing `crc_validation` only — FullFrameAccessOp is CRC-specific.)*

Read the camera's Python class (`python/hololink/sensors/<camera>/<camera>.py`). If not found, skip to 3b-STEP-1. Execute exactly one branch:

- **BRANCH:** `get_csi_frame_layout()` exists → 3a; does not exist → 3b.
- 3a. Write the call in `compose()` (after camera setup) to get `start_byte` and `full_frame_size`. Proceed to Step 4.4.
- 3b. **BRANCH (header accessible):** If C++ header not accessible → skip to 3b-FALLBACK; otherwise → proceed to 3b-STEP-1.

**3b-STEP-1**: Open `hololink_module/host/sensors/<camera>/include/hololink/module/sensors/<camera>/<camera>_cam.hpp`. If not found, try `src/hololink/sensors/camera/<camera>/`; if still missing, skip to 3b-FALLBACK.

**3b-STEP-2**: Find `configure_converter()`. Extract: `start_byte` (required), `line_bytes` (width×bpp/8 rounded to 8), `embedded_data_bytes` (constant+format adj), all pixel-format branches (translate every case, not only RAW_10/RAW_12). start_byte found → 3b-STEP-3; start_byte absent → 3b-FALLBACK.

**3b-STEP-3**: Write `compute_<camera>_csi_layout(camera, converter)` following the `compute_imx274_csi_layout()` pattern in `references/pipeline-reference.md`. If camera lacks `_pixel_format`/`_width`/`_height` or converter methods, adapt from C++ header fields. Proceed to Step 4.4.

**3b-FALLBACK**: CSI layout cannot be derived.
- **Default:** abort, report summary, stop.
- **Override** (if `--allow-unvalidated-layout` was set in `$ARGUMENTS` or during Step 1 clarification): continue to Step 4.4. Prepend `# WARNING: CSI layout unvalidated — do not use for CRC validation without hardware verification.` before writing. Record `LAYOUT_UNVALIDATED = True` (used in success exit item 5).

---

**Step 4.4** — **MUST copy** post-run analysis from the pipeline reference verbatim. Adjust field names only. Multi-concern conflicts: use highest-priority concern's analysis (CRC > PVA > frame_validation > latency > player); append unique metrics from lower-priority concerns.

**BRANCH (after Step 4.4):**
- If `--dry-run` was set → display synthesized content to user and stop (**dry-run exit — no file is written**). Skip Step 5.
- Otherwise → **use the Write tool to create `examples/<new_file>.py`**. Proceed to Pre-Step 5.

---

## Pre-Step 5 — Verify output matches target

Read the output file and confirm: (1) correct receiver class, (2) target camera present and source camera absent, (3) key pipeline operators present. Fix in place and re-verify once. If receiver class or camera mismatch persists → stop (blocking). If operators absent → re-run Step 4.2 once; if still absent → stop (unresolvable).

## Step 5 — Sanity check

Use the Grep tool. After each failing check: fix and re-run **once** (bounded — one retry per check, then escalate).
- Passes on retry → continue to next check.
- Fails again (code issue) → ask user: "Skip check N and continue, or stop?" Skip → app may be structurally incomplete; add "**WARNING: Check N skipped — review manually.**" to summary. Stop → unresolvable exit.
- Fails (hardware/SDK/registers) → **unresolvable** → go to unresolvable exit immediately.

**Check 1 — Correct receiver class:**
Grep for `RoceReceiverOp`, `LinuxReceiverOperator`, `FusaCoeCaptureOp`, `GpuRoceTransceiverOp`.
- PASS: expected class present at least once AND none of the other three appear.
- FAIL: expected class absent OR any other class present.

**Check 2 — No stale transport args:**
- `ibv_name` and `ibv_port` are required for **RoCE only** (player pattern). GPU RoCE player apps use `GpuRoceTransceiverOp` without ibv_name/ibv_port. Note: GPU RoCE loopback apps (`gpu_roce_loopback.py` pattern) do use rx/tx ibv names — skip this check for loopback pipelines.
- Skip if target receiver is RoCE.
- Skip if target receiver is GPU RoCE AND pipeline concern (Step 1 item 4) contains `loopback`.
- Otherwise (Linux, FuSa CoE, GPU RoCE player): grep for `ibv_name` or `ibv_port`. FAIL if either is present — they are stale from a RoCE source and must be removed.

**Check 3 — No stale FuSa CoE audio args:**
Skip if pipeline type (Step 1 item 5) is audio. Otherwise: grep for `cpu_output = True` or `configure_frame_size(`. FAIL if present.

**Check 4 — No stale source-camera name:**
Skip if: no source app; OR renamed-copy path was taken; OR Search 1 recorded "source camera equals target." Otherwise: grep for the **source camera token** from Step 4.2c Search 1. FAIL if found.

**Check 5 — FullFrameAccessOp (CRC only):**
Skip if pipeline concern (Step 1 item 4) does not contain `crc_validation`. Otherwise: grep for `FullFrameAccessOp`. FAIL if absent.

**Check 6 — CSI layout resolution (CRC only):**
Skip if pipeline concern (Step 1 item 4) does not contain `crc_validation`. Otherwise: grep for `get_csi_frame_layout` or `compute_.*csi_layout`. FAIL if absent. Ask user to type: **"CSI layout unvalidated — hardware verification required."** Exact match → continue to success exit with WARNING. Any other response → unresolvable exit.

**Common fixes:** 1 — patch receiver class in `__init__`/`compose()`, preserve operators. 2 — remove ibv_name/ibv_port. 3 — remove cpu_output/configure_frame_size. 4 — replace stale camera token. 5 — add FullFrameAccessOp. 6 — add get_csi_frame_layout() or compute_helper.

**Unresolvable exit:** Preserve file, report structured summary (checks passed, check unresolvable, what is needed), stop. Do not declare app ready.

**Success exit:**
1. Output path: `examples/<new_file>.py`
2. Show file content.
3. Show diff summary: receiver class, camera values, operators, seam method.
4. If checks were skipped: "**WARNING: Skipped checks require manual review before use.**"
5. If `LAYOUT_UNVALIDATED = True`: "**WARNING: CSI layout unvalidated — hardware verification required. Run `/hsb-app` to validate.**" Otherwise: "The new app is ready at `examples/<new_file>.py`. To run on hardware, use `/hsb-app`."

---

## Receiver reference

See [references/receiver-reference.md](references/receiver-reference.md) for constructor args, FuSa CoE `compose()` setup, channel setup, reset/PTP sequence, and CLI args by receiver.

## Pipeline reference

See [references/pipeline-reference.md](references/pipeline-reference.md) for FuSa CoE pipeline shape, CSI layout helpers, and audio pipeline notes.

---

## File naming

`<receiver_prefix><camera>_<pipeline_concern1>[_<pipeline_concern2>...].py`

Single concern: `<receiver_prefix><camera>_<pipeline>.py`. Multi-concern: concatenate with `_`.

Examples:
- `fusa_coe_imx274_frame_validation.py` — single concern
- `fusa_coe_imx274_nvcomp_crc_validation.py` — multi-concern (nvcomp + CRC)
- `linux_vb1940_player.py`
- `imx274_pva_crc_validation.py` (RoCE — no prefix)

---

## Examples

```
/hsb-port-app
/hsb-port-app "port imx274_frame_validation.py to FuSa CoE"
/hsb-port-app "make a linux version of the vb1940 PVA CRC validation app"
/hsb-port-app "create an imx274 FuSa CoE nvcomp CRC validation app"
/hsb-port-app --help
```

---

## Instructions

- Use Grep tool for Step 5 sanity checks. Use Write tool to create `examples/<new_file>.py`.
- Pass `--dry-run` to preview without writing. Pass `--allow-unvalidated-layout` to continue when CSI layout is unavailable.

## Purpose

Create or port Hololink example Python apps combining receiver, camera, and pipeline axes. Synthesizes from existing reference apps.

## Prerequisites

- Hololink/HSB repo root with `examples/` directory; apps follow `<receiver_prefix><camera>_<pipeline>.py` naming.
- `references/receiver-reference.md` and `references/pipeline-reference.md` exist alongside the skill.

## Inputs

No inputs are required — all information can be gathered interactively.

- `$ARGUMENTS` *(optional)*: source — slash-command argument string. Free-form port/create description. If absent or too vague, the skill asks the user. Takes precedence over interactive prompting when present.
- `--allow-unvalidated-layout` *(optional flag in `$ARGUMENTS`)*: source — `$ARGUMENTS` string. Allows synthesis to continue when CSI layout cannot be derived. Default behavior is to abort.
- `--dry-run` *(optional flag in `$ARGUMENTS`)*: source — `$ARGUMENTS` string. Displays synthesized content without writing to disk; skips Step 5.

## Limitations

- Writes `examples/<new_file>.py` to disk; prompts before overwrite. Use `--dry-run` to preview.
- Cannot run on hardware — use `/hsb-app`. CSI layout requires C++ header access.

## Troubleshooting

- **Glob returns no files**: ensure working directory is Hololink repo root.
- **C++ header not found**: proceeds to 3b-FALLBACK; validate on hardware.
- **Check 1 fails**: patch receiver class in `__init__`/`compose()` in-place.
