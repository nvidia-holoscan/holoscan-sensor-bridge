# Worked Example — 3-concern target (nvcomp + CRC + FuSa CoE, imx274)

Target: `fusa_coe_imx274_nvcomp_crc_validation.py`

1. **Camera+receiver reference**: Search for FuSa CoE + imx274. Best match: `fusa_coe_imx274_player.py` (priority 1 — same camera, same receiver, simplest pipeline). Read this for `__init__`, `compose()` wiring, and `main()`.
2. **Pipeline reference (nvcomp+CRC)**: This is two concerns — nvcomp compression and CRC validation. Run pipeline reference selection independently for each:
   - CRC concern: best match is `vb1940_fusa_nvcomp_crc_validation.py` (same pipeline + same receiver, different camera — priority 2). Provides `FullFrameAccessOp`, `ComputeCrcOp`, `CheckCrcOp`.
   - nvcomp concern: same file covers nvcomp compression operators.
   - Since both concerns are covered by the same reference, use it once.
3. **Seam**: imx274 has no `get_csi_frame_layout()` → write `compute_imx274_csi_layout()`. Replace vb1940-specific frame sizes with imx274 values.
4. **Merge**: `FullFrameAccessOp` appears once (shared by CRC and nvcomp concerns) — include it once. No wiring conflicts.
5. **Output**: `examples/fusa_coe_imx274_nvcomp_crc_validation.py`
