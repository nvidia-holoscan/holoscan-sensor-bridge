# Pipeline Reference

## FuSa CoE video pipeline shape

```
fusa_coe_capture → [FullFrameAccessOp] → packed_format_converter
                                       → image_processor → demosaic → visualizer
```

`FullFrameAccessOp` is only needed when the pipeline must access the full CSI buffer (header + pixels) — i.e. CRC validation. Player apps go directly `fusa_coe_capture → packed_format_converter`.

## CSI layout for CRC validation

**vb1940** — use the camera method:
```python
layout = self._camera.get_csi_frame_layout(fusa_coe_capture)
start_byte, full_frame_size = layout.start_byte, layout.full_frame_size
```

**imx274** — `Imx274Cam` has no `get_csi_frame_layout()`; write a standalone helper:
```python
def compute_imx274_csi_layout(camera, converter):
    start_byte = converter.receiver_start_byte()
    transmitted = converter.transmitted_line_bytes(camera._pixel_format, camera._width)
    line_bytes = converter.received_line_bytes(transmitted)
    embedded_data_bytes = 175
    if camera._pixel_format == hololink_module.sensors.csi.PixelFormat.RAW_10:
        start_byte += line_bytes * 8
    elif camera._pixel_format == hololink_module.sensors.csi.PixelFormat.RAW_12:
        embedded_data_bytes += 35
        start_byte += line_bytes * 16
    start_byte += converter.received_line_bytes(embedded_data_bytes)
    pixel_data_size = line_bytes * camera._height
    full_frame_size = start_byte + pixel_data_size
    return start_byte, full_frame_size
```

**Other cameras** — check whether the camera class exposes `get_csi_frame_layout()`; if not, follow these steps:
1. Locate the camera's C++ header at `hololink_module/host/sensors/<camera>/include/hololink/module/sensors/<camera>/<camera>_cam.hpp`.
2. Find `configure_converter()` and read the `start_byte`, `line_bytes`, `embedded_data_bytes`, and pixel-format branch logic.
3. Translate that logic into a Python helper following the `compute_imx274_csi_layout()` pattern above.
4. If the header is not accessible, flag the layout as unvalidated and note that hardware validation is required before the app can be trusted for CRC.

## Audio pipeline

Payload decoding is receiver-specific — do not copy between RoCE and FuSa CoE variants. Use:
- RoCE audio: `audio_recorder.py`
- FuSa CoE audio: `audio_recorder_fusa.py`
