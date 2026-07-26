# CI stress testing

The CI scripts in this folder each run their test suite **once** per pipeline. The
stress-test wrapper repeats one of those scripts many times in a row so that *flaky*
failures and performance variance surface, on a nightly or manual basis.

## Quick start

```sh
# Run the unit tests 25 times in a single container (fast, the default):
ci/stress.sh --count 25 ci/unit_test.sh --dgpu

# Run them with a fresh container per iteration (full isolation):
ci/stress.sh --count 25 --around ci/unit_test.sh --dgpu

# Stress the destructive flasher (around mode only -- see below):
ci/stress.sh --count 5 --around ci/hsb_flasher.sh --dgpu
```

Every iteration runs even if one fails; a per-iteration line and a summary are printed,
and the command exits nonzero if **any** iteration failed:

```
===== STRESS iteration 1/25 (0 passed, 0 failed so far): one_pass --dgpu =====
...
===== STRESS iteration 25/25 (23 passed, 1 failed so far): one_pass --dgpu =====
===== STRESS SUMMARY: 24 passed, 1 failed of 25 =====
```

## Two loop locations, one knob

The wrapped scripts already use a two-level self-invocation: e.g.
`ci/unit_test.sh --dgpu` calls `ci.sh --dgpu <script> go ...`, and `ci.sh` does
`docker run ... hololink-ci <script> go ...`; inside the container the same script
re-runs under its `if [ "$1" = "go" ]` branch where the real `pytest`/`cmake` work
happens. That seam gives two natural places to loop, both driven by the single
environment variable **`STRESS_ITERATIONS`** (default `1`):

| Mode       | Flag                 | What happens                                                                                                     | Use when                                                                                                                      |
| ---------- | -------------------- | ---------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **inside** | `--inside` (default) | One container, N test runs. `ci.sh` forwards `STRESS_ITERATIONS=N` into the container and the `go` branch loops. | Speed — avoids the per-iteration image check + `docker run` startup cost.                                                     |
| **around** | `--around`           | A fresh invocation (new container) per iteration.                                                                | Full isolation between runs; the only correct mode for scripts that do not honor `STRESS_ITERATIONS` (e.g. `hsb_flasher.sh`). |

There is no double counting: in `--around`, `stress.sh` exports `STRESS_ITERATIONS=1` so
each child container runs exactly once while the outer loop drives the count — N total
runs, not N×N.

When `STRESS_ITERATIONS` is unset or `1`, the wrapped scripts behave exactly as before —
the helper runs the command once and propagates its exit status, with no extra output.
Normal CI is unaffected.

## Which scripts support which mode

| Script               | inside (fast) | around (isolated) |
| -------------------- | :-----------: | :---------------: |
| `unit_test.sh`       |      ✅       |        ✅         |
| `imx274_ptp_test.sh` |      ✅       |        ✅         |
| `performance.sh`     |      ✅       |        ✅         |
| `hsb_flasher.sh`     |       —       |        ✅         |

`hsb_flasher.sh` is host orchestration (it builds the flasher in its own container, then
flashes and power-cycles the board on the host) with no `go` branch, so looping inside a
single container is meaningless. It is also destructive, so a fresh invocation per
iteration is exactly what you want — run it with `--around`.

## How it is built

- **`stress_lib.sh`** — sourced helper providing `stress_run [-n N] <command>`. The
  single place the loop, per-iteration logging, summary, and pass/fail semantics live.
  The count comes from `-n N`, else `$STRESS_ITERATIONS`, else 1.
- **`stress.sh`** — the entry point. Parses `--count N` and `--inside`/`--around` and
  dispatches: inside mode runs the command once with `STRESS_ITERATIONS=N`; around mode
  loops the command N times with `stress_run`.
- **`ci.sh`** — forwards `-e STRESS_ITERATIONS` into the container so the `go` branch
  can see it.
- **`unit_test.sh`, `imx274_ptp_test.sh`, `performance.sh`** — their `go` branch wraps
  the existing work in `stress_run` (multi-command branches collect the steps into a
  `one_pass` function).

### A note on `set -o errexit`

Every script runs under `set -o errexit`, which would normally abort the whole script on
the first failing iteration. Two rules keep all N iterations running:

1. `stress_run` invokes the command as an `if` condition (`if "$@"; then ...`). POSIX
   `sh` suppresses `errexit` for a command whose status is tested, so a failing
   iteration is *counted* rather than fatal.
1. In a multi-command `go` branch, each step in `one_pass` ends with `|| return 1` so a
   failure stops that iteration at the failing step (without running later steps),
   independently of the ambient `errexit` state.

## Running it from GitLab CI

`.gitlab-ci.yml` defines seven stress jobs, each in its own stage, that run **only**
when the variable `RUN_STRESS_TEST` is `"true"`. They come in two tiers:

- **Default-on** (`extends: .stress-test`) — `stress-hsb-flasher-job`,
  `stress-dgpu-unit-test-job`, `stress-dgpu-imx274-performance-test-job`, and
  `stress-dgpu-imx274-ptp-test-job`. These run automatically in any triggered stress
  pipeline.
- **Default-off** (`extends: .stress-test-optional`) — `stress-igpu-unit-test-job`,
  `stress-dgpu-imx274-unit-test-job`, and `stress-dgpu-vb1940-unit-test-job`. These show
  up as manual (play-button) jobs you opt into, but run automatically on a nightly
  schedule or when `RUN_ALL_STRESS == "true"`.

```yaml
.stress-test:                  # default-on
  variables:
    STRESS_COUNT: "25"   # iteration count; override per-schedule or per-run
    STRESS_MODE: ""      # "" => --inside (fast); "--around" => fresh container per run
  rules:
    - if: '$RUN_STRESS_TEST == "true"'
  needs: []

.stress-test-optional:         # default-off
  extends: .stress-test
  rules:
    - if: '$RUN_STRESS_TEST == "true" && $CI_PIPELINE_SOURCE == "schedule"'
    - if: '$RUN_STRESS_TEST == "true" && $RUN_ALL_STRESS == "true"'
    - if: '$RUN_STRESS_TEST == "true"'
      when: manual
      allow_failure: true
```

So on a normal merge-request pipeline `RUN_STRESS_TEST` is unset and no stress jobs
appear. They are triggered two ways:

### Nightly (pipeline schedule)

In GitLab: **Build → Pipeline schedules → New schedule**.

- **Interval pattern:** custom cron, e.g. `0 2 * * *` (02:00 daily); pick a timezone.
- **Target branch:** the branch to test (e.g. `internal`).
- **Variables:** add `RUN_STRESS_TEST = true`. Optionally `STRESS_COUNT = 50` and
  `STRESS_MODE = --around` to override the defaults.

On a schedule, **all** stress jobs — including the default-off ones — run automatically.

### Manual

In GitLab: **Build → Pipelines → Run pipeline**. Pick the branch, add the variable
`RUN_STRESS_TEST = true` (and optionally `STRESS_COUNT` / `STRESS_MODE`), then **Run
pipeline**. The default-on jobs run; the default-off jobs appear with a play button you
click to run them. To run everything automatically without clicking, also set
`RUN_ALL_STRESS = true`.

> The scheduled/manual variables (`STRESS_COUNT`, `STRESS_MODE`, `RUN_ALL_STRESS`) take
> precedence over the template defaults. `STRESS_COUNT` is deliberately a different name
> from the script-level `STRESS_ITERATIONS` so it never leaks into the normal jobs and
> make them loop.

### Only the stress jobs run

A scheduled/manual stress pipeline runs **only** the stress jobs — the normal jobs are
excluded from it. Each normal job extends a shared `.not-on-stress` template that skips
it when `RUN_STRESS_TEST` is set:

```yaml
.not-on-stress:
  rules:
    - if: '$RUN_STRESS_TEST == "true"'
      when: never
    - when: on_success

dgpu-unit-test-job:
  extends: .not-on-stress
  # ...rest unchanged...
```

The catch-all `- when: on_success` preserves the normal jobs' existing behavior on every
other pipeline; only the `RUN_STRESS_TEST == "true"` case is changed.

### Other schedulers

The wrapper is just a shell entry point, so it works outside GitLab too — e.g. a Jenkins
"Build periodically" job or a `cron` entry running
`xvfb-run -a ci/stress.sh --count 25 ci/unit_test.sh --dgpu`.
