# Contributing to Holoscan Sensor Bridge

## Table of Contents

- [Introduction](#introduction)
- [Where to Contribute](#where-to-contribute)
- [Bug Fixes to Holoscan Sensor Bridge](#bug-fixes-to-holoscan-sensor-bridge)
- [Signing Your Contribution](#signing-your-contribution)
- [Reporting issues](#reporting-issues)

## Introduction

Welcome to Holoscan Sensor Bridge! Please read our [README](./README.md) document for an
overview of the project.

Please read this guide if you are interested in contributing open source code to
Holoscan Sensor Bridge.

## Where to Contribute

There are two repositories for Holoscan Sensor Bridge (HSB) contributions:

| Contribution                                                                                      | Repository                                                                                            |
| ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| A bug fix for the HSB code                                                                        | [Holoscan Sensor Bridge](https://github.com/nvidia-holoscan/holoscan-sensor-bridge) (this repository) |
| A new sensor, application, operator, FPGA design, demo, tutorial, utility, AI skill, or benchmark | [Holoscan Sensor Bridge Hub](https://github.com/nvidia-holoscan/holoscan-sensor-bridge-hub/tree/main) |

The HSB repository accepts a pull request only for a bug fix. Send all other
contributions to the Holoscan Sensor Bridge Hub, see the
[Hub contributing instructions](https://github.com/nvidia-holoscan/holoscan-sensor-bridge-hub/tree/main?tab=contributing-ov-file).

## Bug Fixes to Holoscan Sensor Bridge

A bug-fix pull request must obey these rules:

- The pull request fixes a bug in the HSB code. It does not add a feature, a sensor, or
  an application.
- Each commit has a DCO sign-off. See
  [Signing Your Contribution](#signing-your-contribution).
- The code passes the lint checks in `scripts/lint/lint.sh`.

### Requirements

We recommend that new developers review GitHub's
[starting documentation](https://docs.github.com/en/get-started/start-your-journey)
before making their first contribution.

### Workflow

1. Developers must first [fork](https://help.github.com/en/articles/fork-a-repo) the
   [upstream](https://github.com/nvidia-holoscan/holoscan-sensor-bridge) Holoscan Sensor
   Bridge repository.

1. Git clone the forked repository and push changes to the personal fork.

   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_FORK.git
   # Checkout the targeted branch and commit changes
   # Push the commits to a branch on the fork (remote).
   git push -u origin <local-branch>:<remote-branch>
   ```

1. All code submissions must be formatted according to the rules in
   `scripts/lint/lint.sh`. You can run `scripts/lint/lint.sh --format` to automatically
   format all C++, Python, and Markdown files-- usually that's enough to get
   `scripts/lint/lint.sh` to pass.

1. Make sure that the lint checks pass: `scripts/lint/lint.sh --lint`.

1. Sign off each commit:

   ```bash
   git commit -s
   ```

   The `-s` option adds the DCO sign-off line (see below). We also recommend that you
   sign each commit with your GPG or SSH key (`git commit -s -S`). Add the key to your
   GitHub account, so that GitHub shows the commit as "Verified".

1. Once the code changes are staged on the fork and ready for review, please
   [submit](https://help.github.com/en/articles/creating-a-pull-request) a
   [Pull Request](https://help.github.com/en/articles/about-pull-requests) (PR) to merge
   the changes from a branch of the fork into a selected branch of upstream.

   - Exercise caution when selecting the source and target branches for the PR.
   - In the PR description, describe the bug, the fix, and how you tested the fix.

1. Holoscan Sensor Bridge maintainers will review the PR and accept the proposal if
   changes meet Holoscan Sensor Bridge standards.

Thanks in advance for your patience as we review your contributions. We do appreciate
them!

### License Guidelines

- Make sure that you can contribute your work to open source. Verify that no license
  and/or patent conflict is introduced by your code. NVIDIA is not responsible for
  conflicts resulting from community contributions.

- We encourage community submissions under the Apache 2.0 permissive open source
  license, which is the [Holoscan Sensor Bridge License](./LICENSE).

- We require that members [sign](#signing-your-contribution) their contributions to
  certify their work.

### Coding Guidelines

- All source code contributions must strictly adhere to the Holoscan Sensor Bridge
  coding style.

## Signing Your Contribution

- We require that all contributors "sign-off" on their commits. This certifies that the
  contribution is your original work, or you have rights to submit it under the same
  license, or a compatible license.

- Any contribution which contains commits that are not Signed-Off will not be accepted.

- To sign off on a commit you simply use the `--signoff` (or `-s`) option when
  committing your changes:

  ```bash
  $ git commit -s -m "Add cool feature."
  ```

  This will append the following to your commit message:

  ```
  Signed-off-by: Your Name <your@email.com>
  ```

- Full text of the DCO:

  ```
    Developer Certificate of Origin
    Version 1.1
    
    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129
    
    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  ```

  ```
    Developer's Certificate of Origin 1.1
    
    By making a contribution to this project, I certify that:
    
    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or
    
    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or
    
    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.
    
    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```

## Reporting issues

Please open a
[Holoscan Sensor Bridge Issue Request](https://github.com/nvidia-holoscan/holoscan-sensor-bridge/issues)
to request an enhancement, bug fix, or other change in Holoscan Sensor Bridge. Report a
problem in a Hub project on the
[Hub issue tracker](https://github.com/nvidia-holoscan/holoscan-sensor-bridge-hub/issues).
