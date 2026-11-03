# Holoscan Sensor Bridge documentation archives

This branch stores prepared documentation snapshots. Assemble a selected immutable
commit with the current user guide to publish multiple versions in one Fern site.
Retain existing snapshots unchanged when adding a new release.

## Version 2.7

`v2.7/` preserves the documentation served at
https://docs.nvidia.com/holoscan/sensor-bridge on November 3, 2026: 32 guide pages,
42 emulator API pages, and 39 served assets. Two additional VSYNC images repair
broken published filesystem URLs using the corresponding files from public commit
`e4e3042d014c89ed14f92669f6b8f28045d96a25`.

The source is an explicit publication capture. The original deployment commit and
API generator source revision are unknown; the historical `2.7.0` tag does not
reproduce the published site. Do not interpret the observed public repository
commit as the original API generation input.

`v2.7/provenance.json` records the captured page URLs, raw hashes, capture time,
and final snapshot file inventory. `v2.7/publication-evidence.json` records asset
hashes and the limited transformations: remove the generated Latest/AI delivery
preamble, restore API metadata and escaped heading anchors, mirror assets,
normalize version links, and repair three legacy anchors and two VSYNC image URLs.
Two of those anchors exist in the rendered site but are omitted by its Markdown
exporter; the GPUDirect anchor repairs an existing broken link at the matching
public-source section.

The prepared tree is licensed under Apache 2.0, like its source repository. See
`LICENSE`. Retain provenance and hashes when mirroring this branch publicly.
