# Receiver Reference

## Constructor args by receiver

| Receiver | Class | Key args |
|----------|-------|----------|
| RoCE | `hololink_module.operators.RoceReceiverOp` | `ibv_name`, `ibv_port`, `hololink_channel`, `device`, `frame_context` |
| Linux | `hololink_module.operators.LinuxReceiverOperator` | `hololink_channel` |
| FuSa CoE | `hololink_module.operators.FusaCoeCaptureOp` | `interface`, `mac_addr`, `hololink_channel`, `timeout`, `device`; audio also needs `cpu_output=True` + `receiver.configure_frame_size(...)` |
| GPU RoCE | `hololink_module.operators.GpuRoceTransceiverOp` | `hololink_channel` (player pattern — no ibv_name/ibv_port; see `gpu_roce_loopback.py` for the loopback pattern which adds rx/tx ibv names) |

## FuSa CoE — additional setup in `compose()`

```python
metadata = self._hololink_channel.enumeration_metadata()
interface = self._interface or metadata.get("interface")
hsb_mac = metadata.get("mac_id")
hsb_mac_bytes = list(bytes.fromhex(hsb_mac.replace(":", "")))

fusa_coe_capture = hololink_module.operators.FusaCoeCaptureOp(
    self, condition,
    name="fusa_coe_capture",
    interface=interface,
    mac_addr=hsb_mac_bytes,
    hololink_channel=self._hololink_channel,
    timeout=self._timeout,
    device=self._camera,
)
self._camera.configure_converter(fusa_coe_capture)
```

Video pipelines: also insert `PackedFormatConverterOp` between capture and downstream.

## Channel setup in `main()`

- RoCE / Linux: `hololink_module.DataChannel(channel_metadata)`
- FuSa CoE (multi-sensor): `DataChannel.use_sensor(metadata_obj, sensor_idx)` then `DataChannel(metadata_obj)`

## Reset / PTP sequence

- RoCE: optional `hololink.ptp_synchronize()` (flag-gated)
- Linux: `hololink.reset()` + `camera.setup_clock()`
- FuSa CoE player: `hololink.reset()` + `camera.setup_clock()`
- FuSa CoE latency/validation: same + **`hololink.ptp_synchronize()`** required for cross-domain timestamps

## CLI args by receiver

| Arg | RoCE | Linux | FuSa CoE |
|-----|------|-------|----------|
| `--ibv-name` / `--ibv-port` | yes | no | no |
| `--interface` | no | no | yes |
| `--timeout` | no | no | yes |
| `--ptp-sync` / `--skip-reset` | yes | no | yes (`--skip-reset` only) |
