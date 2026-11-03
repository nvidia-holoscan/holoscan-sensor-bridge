# Hololink HSDK5 Migration Guide

This document tracks all breaking API and behavioral changes introduced when
porting hololink operators from HSDK4 to HSDK5.  Each operator section
documents what changed, what was intentionally dropped, and how to update
calling code.

These operators and examples need Holoscan SDK 5.0.0.  Its pre-release builds before
5.0.0.7 under-size the tensor pool of an operator whose output crosses to another thread
through a queue that drops the oldest sample, so the multi-camera players fail now and then
with `kResourceExhausted` from `BayerDemosaicOp`.  Every 5.0.0.x build reports version 5.0.0
to CMake, so `find_package(holoscan)` can't tell them apart: use the released packages.

---

## Receiver Operators

Covers `BaseReceiverOp`, `LinuxReceiverOp`, `RoceReceiverOp`, `GpuRoceTransceiverOp`, and
`CpuRoceTransceiverOp`.

### 1. Operator construction and parameters

**HSDK4:** operators were constructed with `holoscan::ArgList{holoscan::Arg{"param", value}, ...}`
or YAML-driven parameter registration via `spec.param(...)`.

**HSDK5:** pass the shared data channel and optional device lifecycle callbacks to the
operator constructor. Pass buffer and receiver configuration through the dedicated flat
aggregate using `set_params()`.

```cpp
// HSDK4
auto rx = graph.add_operator<hololink::operators::LinuxReceiverOp>("rx",
    holoscan::Arg("hololink_channel", &channel),
    holoscan::Arg("frame_size", frame_size),
    holoscan::Arg("pages", 2u),
    holoscan::Arg("queue_size", 1u));

// HSDK5
auto channel = std::make_shared<hololink::DataChannel>(channel_metadata);
auto rx = graph.op<hololink::operators::LinuxReceiverOp>(
    "rx", channel,
    [camera] { camera->start(); },
    [camera] { camera->stop(); });
rx.set_params(hololink::operators::LinuxReceiverParams {
    .frame_context    = cu_context,
    .frame_size       = frame_size,
    .pages            = 2,
    .queue_size       = 1,
});
```

The same applies to `RoceReceiverParams`, `GpuRoceTransceiverParams` and
`CpuRoceTransceiverParams`. The constructor
rejects a null data channel. Capture callback dependencies by value so graph replays do not
refer to local variables that have gone out of scope.

`frame_context` remains a parameter because `CUcontext` is a borrowed opaque pointer, which
HSDK5 intentionally rejects as a replayable constructor capture. The application must keep
that CUDA context valid for every receiver execution that uses it.

---

### 2. Output ports — `tensor` and `metadata` replace the GXF Entity

**HSDK4:** operators emitted a `gxf::Entity` carrying a `nvidia::gxf::Tensor` that wrapped
the RDMA buffer in-place (zero-copy).  Metadata was a GXF metadata dictionary attached to
the same entity.

**HSDK5:** `LinuxReceiverOp` and `RoceReceiverOp` publish each frame on two output ports:

- `tensor` (`Output<holoscan::Tensor>`): the frame's bytes, a 1-D `uint8` tensor in a CUDA
  device pool slot (trimmed to `bytes_written` when `trim` is set).
- `metadata` (`Output<hololink::CaptureMetadataT>`): the per-frame hardware metadata, a
  FlatBuffer table defined in `src/hololink/operators/capture_metadata.fbs` (see §3).

Both samples are emitted with the same `capture_time` (when the receiver got the frame) and
`frame_id` (the frame's `frame_number`).  `FusaCoeCaptureOp` publishes the same pair, and
every operator that passes frames on receives the pair on two inputs and publishes it on its
own `tensor` and `metadata` outputs.  The converters (`CsiToBayerOp`,
`PackedFormatConverterOp`) receive a `holoscan::Tensor` and publish a
`holoscan::schema::ImageT`; `ImageProcessorOp`, `SubFrameCombinerOp` and `ArgusIspOp` receive
and publish `schema::ImageT`.

**Exception:** `GpuRoceTransceiverOp` and `CpuRoceTransceiverOp` emit nothing on these ports;
`get_next_frame()` always returns `{}`.  `GpuRoceTransceiverOp`'s GPU persistent kernel owns
the data path and writes directly into a ring buffer; downstream operators access frame data
via the direct-access API (`frame_memory()`, `get_ring_addr()`, `get_ring_stride_sz()`, etc.).
`CpuRoceTransceiverOp`'s CPU-polled completion queues own its data path; in Duplex mode, the
application reaches its rings through `get_ring_addr(is_rx)` and the other ring accessors —
see §11.

```cpp
// HSDK4 sink
auto sample = op_input.receive<gxf::Entity>("output");
auto tensor  = sample.get<nvidia::gxf::Tensor>(""); // zero-copy RDMA pointer
int64_t crc  = sample.get<int64_t>("crc");          // GXF metadata dict

// HSDK5 sink: one input per port
holoscan::Input<holoscan::Tensor> input_tensor { "in_tensor" };
holoscan::Input<hololink::CaptureMetadataT> input_metadata { "in_metadata" };

// contract(): activate once both inputs hold the samples of one frame
holoscan::Contract c;
c.trigger(holoscan::OnAligned {
     .ports = holoscan::ports(input_tensor, input_metadata),
     .window = hololink::operators::kMetadataPairingWindow,  // 0: same capture_time
 })
    .input(input_tensor, holoscan::Exact {})
    .input(input_metadata, holoscan::Exact {});

// compute()
auto maybe_paired = hololink::operators::receive_paired_tensor_metadata(input_tensor, input_metadata);
if (!maybe_paired) {
    return holoscan::make_unexpected(std::move(maybe_paired).error());  // a receive() error
}
if (!*maybe_paired) {
    // frame_id and frame_number differ: count the drop (DropCounter), skip the frame
    return {};
}
auto& paired = **maybe_paired;
const holoscan::Tensor& pixels = paired.tensor.data;        // copied pool tensor
int64_t crc                    = paired.metadata.data.crc;  // typed schema field

// Wiring
graph.add_flow(rx->tensor, sink->input_tensor);
graph.add_flow(rx->metadata, sink->input_metadata);
```

- `kMetadataPairingWindow`, `receive_paired_tensor_metadata()` and the helpers below are in
  `src/hololink/operators/capture_metadata_utils.hpp`.
- `kMetadataPairingWindow` is 0: a tensor sample pairs only with the metadata sample
  that has the identical `capture_time`.  The producer stamps both samples of a frame
  (or of a sub-frame band) with the time of the activation that emits them, and every
  later hop passes that time on unchanged (`propagate_emit_options()` below; the SDK's
  operators keep it too), so samples of different frames never align.  A sample whose
  partner never comes, such as a tensor whose metadata was dropped, is retired once the
  other input holds a newer sample, and the next frame pairs as usual.
- `receive_paired_tensor_metadata()` also checks that the tensor sample's `frame_id`
  matches the metadata's `frame_number`.  When they differ, it returns an empty
  `std::optional` instead of a pair: skip the frame and count the drop (`DropCounter`
  reports drops at most once per second); the run goes on.  It returns an error only
  when receiving a sample fails.
- A consumer must connect both inputs: its `OnAligned` trigger fires only when both hold a
  sample.  A producer's `metadata` output may be left unconnected.
- Connect a consumer that takes only the `metadata` output, such as a printer or a frame
  counter, with `holoscan::QueuePolicy::kDropOldest`, so that it never fills up.  When a
  `metadata` queue is full, the producer drops the frame's metadata after its tensor
  went out, and consumers that pair the two lose that frame too; Holoscan SDK builds before
  5.0.0.7 also fail the run when the stream ends after such a partial publication.
- Operators that pass frames on publish the pair with `emit_tensor_and_metadata()`, using
  `EmitOptions` from `propagate_emit_options()`, which keeps the received sample's
  `capture_time` and flags (such as `holoscan::SampleFlags::kDegraded`) and sets `frame_id`.
  When a downstream queue or the output tensor pool is full, the frame is dropped instead of
  failing the run, so a slow consumer costs frames, not the run; the drops are logged at most
  once per second.

**Important:** HSDK5 no longer wraps the RDMA buffer directly.  One `cudaMemcpyAsync`
(Device-to-Device) is performed per frame to copy into the HSDK5 tensor pool.

---

### 3. Frame metadata — GXF metadata dict replaced by `CaptureMetadataT` fields

**HSDK4:** all metadata was accessible from the GXF metadata dictionary via string keys.

**HSDK5:** metadata is carried in the fields of `hololink::CaptureMetadataT`, the sample
on the `metadata` port.  Below, `sample` is `input_metadata.receive()`; after
`receive_paired_tensor_metadata()` (§2), the same fields are
`paired.metadata.data.<field>`.  Every field is an `int64` except `sub_frame_rows`
(`uint32`); a field the producer doesn't fill is 0.

| HSDK4 metadata key | HSDK5 field | Notes |
|---|---|---|
| `"frame_number"` | `sample->data.frame_number` | In sub-frame mode, `CsiToBayerOp` replaces the receiver's chunk number with the sensor frame number |
| `"received_frame_number"` | `sample->data.received_frame_number` | The receiver's own frame count; filled by the Linux and RoCE receivers (native and `hololink_module`) |
| `"timestamp_s"` | `sample->data.timestamp_s` | |
| `"timestamp_ns"` | `sample->data.timestamp_ns` | |
| `"metadata_s"` | `sample->data.metadata_s` | |
| `"metadata_ns"` | `sample->data.metadata_ns` | |
| `"received_s"` | `sample->data.received_s` | |
| `"received_ns"` | `sample->data.received_ns` | |
| `"crc"` | `sample->data.crc` | |
| `"bytes_written"` | `sample->data.bytes_written` | |
| `"packets_dropped"` | `sample->data.packets_dropped` | The Linux receivers' "packets_dropped" plus the RoCE receivers' "dropped" (native and `hololink_module`) |
| `"psn"` | `sample->data.psn` | The Linux receivers (native and `hololink_module`) and the `hololink_module` RoCE receiver; the native `RoceReceiverOp` leaves it 0 |
| `"imm_data"` | `sample->data.imm_data` | |
| `"page_number"` | `sample->data.page_number` | `imm_data & 0xFFF` |
| `"frame_packets_received"` | `sample->data.frame_packets_received` | The Linux receivers only (native and `hololink_module`) |
| `"frame_bytes_received"` | `sample->data.frame_bytes_received` | The Linux receivers only (native and `hololink_module`) |
| `"rx_write_requests"` | `sample->data.rx_write_requests` | Always 0: reading the NIC's counter cost a sysfs read per frame; read `hw_counters/rx_write_requests` under `/sys/class/infiniband` instead |
| `"sub_frame_offset"` | `sample->data.sub_frame_offset` | First image row of the band, set by `CsiToBayerOp` in sub-frame mode; 0 for full frames |
| — | `sample->data.sub_frame_rows` | Rows per band, set by `CsiToBayerOp`; 0 for full frames |
| — | `sample->data.operator_entry_ns` | `CLOCK_REALTIME` time (ns) at which `CsiToBayerOp` started processing the frame; 0 if not measured |
| `"frame_memory"` | **Removed** — use the received tensor's `data()` | See §9 |
| `"frame_id"` (implicit) | `sample->metadata.frame_id` | Set from `frame_number` via HSDK5 `EmitOptions`, on both the tensor and the metadata sample |

`LinuxReceiverOp` and `RoceReceiverOp`, native and `hololink_module`, all fill
`frame_number`, `received_frame_number`, the `timestamp_*`, `metadata_*` and
`received_*` times, `crc`, `bytes_written`, `packets_dropped`, `imm_data` and
`page_number`; the notes above say which of them fill the others.  The
`hololink_module` `HsbControllerOp` fills what its Linux or RoCE network receiver fills.
`FusaCoeCaptureOp` fills only `frame_number`, `timestamp_*`, `metadata_*`, `crc` and
`received_*` (FuSa CoE Capture Operator §4).

---

### 4. CUDA device placement — `CompileOptions` replaces `cuda_device_id` param

**HSDK4:** `cuda_device_id` was passed as a named argument to control the `DLDevice` field
of the output tensor.

**HSDK5:** the `cuda_device_id` field has been **removed** from all params structs.  CUDA
device placement is declared at graph compile time:

```cpp
holoscan::CompileOptions options;
options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement {
    .operator_path = "rx",
    .output_port   = "tensor",
    .device        = holoscan::DeviceId { 0 },  // GPU 0
});
auto plan = holoscan::compile(graph, std::move(options));
```

This must be called for every receiver operator whose `tensor` output is consumed
downstream.

---

### 5. `set_rename_metadata()` removed

**HSDK4:** callers could pass a rename function to redirect metadata keys, e.g. for stereo
pipelines where two receivers shared a downstream GXF metadata dict.

**HSDK5:** the function has been **removed**.  `CaptureMetadataT` fields are fixed by the
`.fbs` schema and cannot be renamed at runtime.  Each receiver publishes its metadata as its
own sample on its own `metadata` port, so for multi-camera scenarios, keep each receiver's
`tensor`/`metadata` pair on separate typed inputs and correlate them via `OnAligned`
contracts:

```cpp
// HSDK5 stereo pattern
holoscan::Input<holoscan::Tensor> left_tensor;                // left receiver's "tensor"
holoscan::Input<hololink::CaptureMetadataT> left_metadata;    // left receiver's "metadata"
holoscan::Input<holoscan::Tensor> right_tensor;               // right receiver's "tensor"
holoscan::Input<hololink::CaptureMetadataT> right_metadata;   // right receiver's "metadata"
// In contract(): trigger(OnAligned{.ports = ports(left_tensor, left_metadata, right_tensor,
//     right_metadata), .window = 2ms}), with Exact{} on each input
```

Pair each camera's tensor and metadata with `receive_paired_tensor_metadata()` (§2).

---

### 6. `use_frame_ready_condition` removed

**HSDK4:** setting `use_frame_ready_condition=false` disabled the `AsynchronousCondition`
and caused `compute()` to block inside `get_next_frame()`.  This was used in sub-frame
processing examples for minimum wakeup latency.

**HSDK5:** the parameter has been **removed**.  `OnNotified` scheduling is always used.
HSDK5's notification path is more efficient than HSDK4's `AsynchronousCondition`, reducing
the latency concern.

For what `use_frame_ready_condition=false` also did, running `compute()` while no frame
arrives, set `watch_ingress = true` in the native `LinuxReceiverParams` or
`RoceReceiverParams` (default `false`).  The receiver then also runs `compute()` after
each second without a frame, logs the error "Ingress frame timeout; ignoring." once per
stall, and counts each such run toward `frame_limit` (§13).  With the default, as with
HSDK4's frame-ready condition, a receiver that gets no frames doesn't run `compute()`,
logs nothing and counts nothing.  The `hololink_module` receivers have no
`watch_ingress`: one that gets no frames behaves like that default, and HSDK4's warnings
"Ingress stalled: ..." and "Ingress resumed after N empty polls" are gone.

The sub-frame players (`sub_frame_imx274_player`, `linux_sub_frame_imx274_player`,
`sub_frame_vb1940_player`), the HSDK4 users of `use_frame_ready_condition=false`, set
`watch_ingress`.  They also run the receiver and the operators that process each band
(`CsiToBayerOp`, `ImageProcessorOp` and, in the IMX274 players, `SubFrameCombinerOp`) in
a graph partition of their own, so the display's swapchain waits never hold up the
receiver.

---

### 7. `device_start` / `device_stop` are optional constructor arguments

**HSDK4:** not providing `device_start` caused a null-function-call crash.

**HSDK5:** both callbacks are optional constructor arguments. Omitting them silently skips
camera start/stop. If your sensor requires explicit
start/stop, **always provide both callbacks**.

---

### 8. `queue_size` must be ≤ `pages`

Unchanged from HSDK4: `queue_size` and `pages` keep their HSDK4 defaults
(`LinuxReceiverParams`: `pages=1, queue_size=1`; `RoceReceiverParams`: `pages=2,
queue_size=1`), and `start_receiver()` still throws if `queue_size > pages`.

---

### 9. `"frame_memory"` metadata key removed (RoCE)

**HSDK4:** the raw `CUdeviceptr` of the RDMA receive buffer was published to the GXF
metadata dict as `"frame_memory"`.

**HSDK5:** this key is not present in `CaptureMetadataT`.  The pixel data pointer is the
received tensor's `data()` (`paired.tensor.data.data()`), which points at the copy
in the receiver's tensor pool (see §2).  The DMA buffer slot index is available via
`sample->data.page_number` (`imm_data & 0xFFF`).

---

### 10. GXF-style operator construction removed

The following HSDK4 patterns no longer compile against the HSDK5 base:

| Removed | Replacement |
|---|---|
| `HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(Op, Base)` | Constructor for runtime dependencies + `set_params()` for configuration |
| `spec.param(x_, "x", ...)` | Field in `XxxParams` struct |
| `register_converter<DataChannel*>()` | Pass `std::shared_ptr<DataChannel>` to the constructor |
| `x_.get()` | `this->params().x` |
| `BaseReceiverOp::setup(spec)` call in derived `setup()` | Derived `setup()` removed entirely |
| `BaseReceiverOp::start()` / `stop()` explicit calls in overrides | Overrides removed; base handles lifecycle |
| `void initialize()` override | Removed; no per-type YAML parsing needed |

---

### 11. `GpuRoceTransceiverOp` and `CpuRoceTransceiverOp` — same changes plus CRTP conversion

All the above apply.  Additionally:

- The direct `frame_ready_condition_->event_state(EVENT_WAITING/DONE)` calls in
  `start()`/`stop()` are gone — the `NotificationSender` lifecycle handles this.
- `GpuRoceTransceiverOp` now inherits `BaseReceiverOp<GpuRoceTransceiverOp, GpuRoceTransceiverParams>`.
- `get_next_frame()` still always returns `{}` (GPU persistent kernel owns the data path).
  Downstream operators access frame data via `frame_memory()`, `get_ring_addr()`, etc.
  directly — this design is unchanged from HSDK4.

`CpuRoceTransceiverOp` changes the same way: it inherits
`BaseReceiverOp<CpuRoceTransceiverOp, CpuRoceTransceiverParams>`, its HSDK4 parameters keep
their names and defaults in `CpuRoceTransceiverParams`, and its `start()`/`stop()` overrides
are gone (`BaseReceiverOp` stops the receiver when a start fails).  Its ring accessors keep
their HSDK4 signatures (`get_ring_addr(bool is_rx)` and the others, without a sensor index);
it has no `frame_memory()`.  One behavior differs: a CQ monitor failure ends the run, as a
fatal receive-path event does in `RoceReceiverOp`, and a failure that comes while the operator
starts fails its start.  HSDK4 logged it and left the datapath inactive while the application
ran on.

---

### 12. Python bindings

The Python bindings are ported as well.  A Python application adds a receiver with
`graph.op()`, which takes the receiver's parameters (the fields of its `...Params`) as
keyword arguments, and its live objects as constructor handles: the `hololink.DataChannel`
(`data_channel`; the `hololink_module` receivers take their enumeration `metadata`) and,
optionally, the camera's start and stop callables (`device_start`, `device_stop`; any
Python callable will do).  Its outputs are `tensor` and `metadata`, and the tensor output
needs a device binding, as in C++ (section 4):

```python
rx = graph.op(
    hl.operators.LinuxReceiverOp,
    "rx",
    data_channel=channel,
    frame_context=int(cu_context),
    device_start=camera.start,
    device_stop=camera.stop,
    frame_size=frame_size,
    frame_limit=frame_limit or 0,
)
graph.add_flow(rx.tensor, csi.in_tensor)
graph.add_flow(rx.metadata, csi.in_metadata)

options = hs.CompileOptions()
options.deployment.bind_tensor_output_device(
    operator_path="rx", output_port="tensor", device=hs.execution.DeviceId(0)
)
```

---

### 13. `frame_limit` replaces a `CountCondition` on the receiver

**HSDK4:** an application stopped a receiver after a number of frames by attaching a
`CountCondition` to it; the run ended once the rest of the pipeline had nothing left to do.

**HSDK5:** set `frame_limit` in the receiver's parameters.  The receiver completes after
that many `compute()` calls (0, the default, means no limit), its outputs close, and the
run completes (`RunTermination::kCompleted`) once the rest of the graph has drained.
The `hololink_module` receivers (`HsbControllerOp` included) and both
`FusaCoeCaptureOp`s (`FusaCoeCaptureParams`) take the same field, and `frame_limit_ends_run`
(below).

```cpp
rx.set_params(hololink::operators::LinuxReceiverParams {
    .frame_context    = cu_context,
    .frame_size       = frame_size,
    .frame_limit      = 100,
});
```

With `watch_ingress = true` (§6), each `compute()` that runs after a second without a
frame counts too, as the timeouts of HSDK4's blocking receivers counted toward their
`CountCondition`.  Without it, a receiver that gets no frames counts nothing, and the
run goes on until it is stopped.

A RoCE receiver whose receive path fails for good (a fatal completion-queue or queue-pair
event, after which no frame can arrive) fails the run with `ErrorCode::kFailure`, so that a
frame limit doesn't leave it waiting forever; HSDK4's receiver kept running `compute()`
until its `CountCondition` was used up.  The `hololink_module` `RoceReceiverOp` learns of
it through `RoceReceiverInterfaceV2::receive_path_failed()`, which the module's RoCE
receivers publish next to `RoceReceiverInterfaceV1`; with a module that publishes only V1,
the run goes on as before.

The run completes this way only when every source in the graph completes: a second
receiver without a limit, or any other operator that keeps running on its own (on a
clock, for instance), keeps the run going until it is stopped.  Holoscan SDK builds before
5.0.0.7 also don't complete a run by itself through an operator output that feeds more
than one consumer.  The latency tools' `DisplayHandoffOp` (`examples/imx274_latency.hpp`),
written for them, counts the frames after the fan-out instead and calls
`ExecutionContext::request_completion()` from its `compute()` at the limit; the run then
ends `RunTermination::kRequestedCompleted`, without waiting for the frames still in the
pipeline.

Nor does it complete while an operator triggered by one input holds, on another input, a
sample that came after the triggering input's stream ended.  A `HolovizOp` that renders on
the left camera's layer keeps a stereo run going when the right camera's last frame reaches
it after the left camera's; one that draws inference overlays does the same with an overlay
that comes after the video's last frame.  For such a run, set `frame_limit_ends_run = true`
next to `frame_limit` on its receivers: once the last of them has reached its limit, that
`compute()` call ends the whole run (`request_completion()`,
`RunTermination::kRequestedCompleted`; frames still in the pipeline are dropped).  A receiver
that never reaches its limit, such as a camera that stops sending, still keeps the run going.
HSDK5 gives an operator nothing that names its run, so the receivers that set it form one
group per executable, or per Python extension module (`hololink` and `hololink_module` each
have their own): run one frame-limited graph at a time, and in Python take its receivers
from one of the two packages.  The example apps that give their receivers a `frame_limit`
and display the frames set it; the audio recorders, which must write every frame, do not.

The sub-frame players (`sub_frame_imx274_player`, `linux_sub_frame_imx274_player`,
`sub_frame_vb1940_player`) leave `frame_limit` unset, because their receivers deliver
each frame in several chunks.  Their `--frame-limit` counts whole frames instead, with
`FrameLimitOp` (`hx::add_frame_limit()` in `examples/player_support.hpp`), which reads a
`metadata` output through a queue that drops the oldest sample and calls
`request_completion()` at the limit:

- The IMX274 players count the frames `SubFrameCombinerOp` emits.
- `sub_frame_vb1940_player` counts on `ImageProcessorOp`'s per-band metadata: a frame
  counts when a band of a later frame arrives, and a late band of an earlier frame
  doesn't count.

Only frames count, so a sub-frame player whose camera sends nothing logs "Ingress frame
timeout" and runs until it is stopped; HSDK4's counted its receiver's one-second timeouts
too.

When `FrameLimitOp` falls behind, the queue sheds samples from the count, so the run
ends later rather than holding back the pipeline.

---

### 14. `ibv_name` defaults to the device that reaches the board (RoCE)

**HSDK4:** `RoceReceiverOp`, `GpuRoceTransceiverOp` and `CpuRoceTransceiverOp` defaulted
`ibv_name` to `"roceP5p3s0f0"`, an IGX's first port, and the example applications passed the
first device the host lists.  Neither reaches every board: a board on an IGX's second port
needs the other device, and on a DGX Spark each port shows up as two devices, of which only
the one with the IP address has the board's RoCE v2 GID.

**HSDK5:** `ibv_name` defaults to empty in `RoceReceiverParams`, `GpuRoceTransceiverParams`
and `CpuRoceTransceiverParams`.  An empty name makes the receiver use the device, and its
port, on the network interface that `configure_roce()` programs the board to send to: the
one the board enumerated on (`DataChannel::infiniband_device()`).  It logs the device it
picked, and its start fails, naming that interface, when no device is bound to it.  A
name given explicitly is used as before, with `ibv_port`.  With no `ibv_name`,
`RoceTransmitterOp`, which has no `DataChannel`, takes the device on the interface that the
routing table uses for `hololink_ip` instead of the first one.  The example applications'
`--ibv-name` options default to empty too.

---

### Summary of removed parameters

| Parameter | Operator(s) | Reason removed |
|---|---|---|
| `cuda_device_id` | All receivers | Device placement via `CompileOptions`; no longer a per-call setting |
| `hololink_channel` | All receivers | Required shared runtime dependency passed to the constructor |
| `device_start` / `device_stop` | All receivers | Optional lifecycle behavior passed to the constructor |
| `use_frame_ready_condition` | Linux, RoCE | HSDK5 always uses notification scheduling; for `use_frame_ready_condition=false`, set `watch_ingress = true` (native receivers, §6) |
| `rename_metadata` / `set_rename_metadata()` | Linux, RoCE | Schema field names are fixed at compile time |

---

<!-- Add new operator sections below this line -->

## FuSa CoE Capture Operator

Covers `FusaCoeCaptureOp`, native and `hololink_module`.  It publishes frames as the
receivers do (Receiver Operators §2), but it is configured differently.

### 1. Construction and `FusaCoeCaptureParams`

**HSDK4:** the operator took `holoscan::Arg`s (`interface`, `mac_addr`, `timeout`,
`hololink_channel` or, in `hololink_module`, `metadata`, `device_start`, `device_stop`,
`out_tensor_name`, ...), and the application then called `configure_format()` and
`configure_converter()` on the operator instance.

**HSDK5:** `graph.op<>()` takes only the operator's name; everything else is a field of
`FusaCoeCaptureParams`, passed with `set_params()`.  Unlike the receivers, the data
channel and the device callbacks are fields too, not constructor arguments:

```cpp
using hololink::operators::FusaCoeCaptureOp;
// The HSB packetizer has programs only for RAW_10 and RAW_12.
const bool packetizer_enabled
    = camera->get_pixel_format() != hololink::csi::PixelFormat::RAW_8;
hololink::csi::CsiGeometry geometry;
camera->configure_converter(
    FusaCoeCaptureOp::make_geometry_adapter(geometry, packetizer_enabled));

auto capture = graph.op<FusaCoeCaptureOp>("capture");
capture.set_params(hololink::operators::FusaCoeCaptureParams {
    .hololink_channel   = &hololink_channel,  // a DataChannel that outlives the run
    .device_start       = [camera] { camera->start(); },
    .device_stop        = [camera] { camera->stop(); },
    .interface          = interface,
    .mac_addr           = mac_addr,           // 6 bytes
    .timeout            = 1500,               // per capture request, in ms
    .pixel_format       = camera->get_pixel_format(),
    .bayer_format       = camera->get_bayer_format(),
    .packetizer_enabled = packetizer_enabled,
    .geometry           = geometry,
    .name               = std::string(capture.name()),  // for its log messages, §5
});
```

- The `hololink_module` operator takes the channel's `EnumerationMetadata` (`metadata`)
  instead of `hololink_channel`.
- `start()` throws when `device_start` or `device_stop` is missing, when `mac_addr`
  doesn't hold 6 bytes, when `timeout` is 0, or, in the native operator, when
  `hololink_channel` is null.
- `out_tensor_name` is gone (the frame has a port of its own, §4), and so are the native
  operator's `cpu_output` and the `hololink_module` operator's `rename_metadata`
  (Receiver Operators §5).

---

### 2. The capture format is set in the parameters

**HSDK4:** `configure_format(pixel_format, bayer_format, packetizer_enabled)` had to be
called on the operator before `configure_converter()` and `start()`, and
`configure_frame_size(frame_size_bytes)` set up a capture of opaque byte blobs instead.

**HSDK5:** set `pixel_format`, `bayer_format` and `packetizer_enabled` in
`FusaCoeCaptureParams`; their defaults are placeholders, so always set all three.
Together they select one NvSciBuf color format.  The packetizer can be enabled only for
RAW_10 and RAW_12, and `start()` throws for a combination that has no format.  For
opaque byte blobs, set `frame_size_bytes` instead; the format and `geometry` fields are
then ignored.

---

### 3. `make_geometry_adapter()` replaces the live `CsiConverter`

**HSDK4:** the operator was a `CsiConverter` (`CsiConverterV1` in `hololink_module`).
The camera driver configured the live operator (`camera->configure_converter(capture)`),
and the operator then configured the converter after it
(`capture->configure_converter(*converter)`).

**HSDK5:** there is no operator instance to configure before the graph runs.  As in §1,
pass `FusaCoeCaptureOp::make_geometry_adapter(geometry, packetizer_enabled)` to the
camera driver's `configure_converter()` before declaring the operator, then copy
`geometry` into `FusaCoeCaptureParams::geometry` and into the parameters of the
converter after it: `PackedFormatConverterParams::geometry` for packetized data (Image
and Converter Operators §3), or `CsiToBayerParams::geometry`, with
`header_included = false`, for unpacketized data.  The adapter lays out lines the way
the FuSa capture receives them, which with the packetizer differs from `CsiToBayerOp`'s
layout.  Pass the same `packetizer_enabled` to `make_geometry_adapter()` and to
`FusaCoeCaptureParams`: `start()` throws when `geometry.bytes_per_line` doesn't match
the format.

---

### 4. Output ports — `tensor` and `metadata` replace the GXF Entity

**HSDK4:** the operator emitted a GXF entity on `output`: a tensor named
`out_tensor_name` that wrapped the capture buffer in place, with the frame's metadata in
the GXF metadata dictionary.

**HSDK5:** like the receivers (Receiver Operators §2), the operator publishes each frame
on `tensor` (`Output<holoscan::Tensor>`, a 1-D `uint8` CUDA tensor) and `metadata`
(`Output<hololink::CaptureMetadataT>`).

- The tensor holds only the image lines,
  `geometry.bytes_per_line * geometry.pixel_height` bytes; the bytes before and after
  them are left out, which is why a `CsiToBayerOp` after the operator needs
  `header_included = false`.  It is a copy in the runtime tensor pool (one
  `cudaMemcpyAsync` per frame), and the capture buffer is released once the copy is
  done.  Bind the `tensor` output to the GPU at compile time, as for the receivers
  (Receiver Operators §4).
- The metadata holds `frame_number`, `timestamp_s`/`timestamp_ns`,
  `metadata_s`/`metadata_ns` and `crc` from the HSB, and `received_s`/`received_ns`, the
  host time at which `compute()` took the frame; the other fields are 0.
- When the tensor pool or a downstream queue is full, the frame is dropped instead of
  failing the run; the drops are logged at most once per second.
- The operator holds only the newest frame, and `compute()` runs once for each frame it
  can take.  When `compute()` has not taken a frame by the time the next one arrives, as
  when the operators that share its thread are busy, the newer frame replaces it, and
  the replaced frame is logged as dropped too ("replaced by a newer frame").  Frames
  replaced before the run's first `compute()` only appear in one start-up message.

---

### 5. `stall_timeout_ms`

A warning is always logged when frames stop arriving.  Set `stall_timeout_ms` to also
fail the run after that many milliseconds without a frame (0, the default, disables
this): `compute()` returns `ErrorCode::kTimeout`, the run ends
`RunTermination::kFailed`, and its diagnostic names the capture operator and says how
long it stalled.  This replaces the `hololink_module` operator's `capture_stalled()` and
`capture_stall_duration_ms()`, which HSDK4 applications checked to fail the run; the
native operator gains the field.

Set `name`, the last field of `FusaCoeCaptureParams`, to the operator's graph name
(`std::string(capture.name())`).  The stall, resume and failure messages use it to say
which capture stalled, as HSDK4's `out_tensor_name` did.

---

### 6. `frame_limit` replaces a `CountCondition`

Set `frame_limit` instead of attaching a `CountCondition` to the operator.  It completes
after that many `compute()` calls, as the receivers do (Receiver Operators §13, which
also says when the run then completes by itself, and when to set `frame_limit_ends_run`).

Only frames count.  HSDK4's native capture operator also counted each capture request that
timed out, so a camera that sent nothing ended a `--frame-limit` run after `frame_limit`
timeouts; in HSDK5 such a run goes on.  Set `stall_timeout_ms` (§5) to end it.  The FuSa
players set it to `frame_limit` times the capture timeout when `--frame-limit` is given
(`hx::frame_limit_stall_timeout_ms()`), so a camera that sends nothing fails the run after
about as long as HSDK4's run took to end.

---

## Image and Converter Operators

Covers `CsiToBayerOp`, `PackedFormatConverterOp` and `SubFrameCombinerOp`, and the
`hololink_module` `CsiToBayerOp` and `PackedFormatConverterOp` where noted.  Their ports
follow the receivers' `tensor`/`metadata` pattern (Receiver Operators §2).

### 1. Frames with lost data are passed on, marked degraded

`CsiToBayerParams::drop_incomplete` and `SubFrameCombinerParams::drop_incomplete` (default
`false`) decide what happens to data that was partly lost upstream:

- `CsiToBayerOp`, full frames: of a frame the HSB wrote only part of (the receiver trims it
  to `bytes_written`), the rows that arrived are converted and the rest of the image is
  zeroed; a frame that lost packets on the way (the Linux receiver counts fewer bytes
  received than written) is converted as received.
- `CsiToBayerOp`, sub-frame mode: a band missing data from a holed receiver chunk is
  emitted with its lost rows holding stale data.  A band whose last chunk was dropped, or
  was too short to finish it, is never emitted, and the band after it goes out marked
  degraded.
- `SubFrameCombinerOp`: a frame missing a band (including one whose next frame starts before
  all of its bands arrived), or holding a band marked degraded, is emitted with the missing
  rows holding an earlier frame's content.

By default, both samples of such a frame or band carry `holoscan::SampleFlags::kDegraded`,
which `propagate_emit_options()` carries through the operators downstream; check
`sample->metadata.flags` to tell.  HSDK4 passed such data on unmarked.  With
`drop_incomplete = true`, `CsiToBayerOp` withholds it and `SubFrameCombinerOp` discards it.
The `hololink_module` `CsiToBayerParams` has the same field.

---

### 2. `bayer_format` is optional

`CsiToBayerParams::bayer_format` and `PackedFormatConverterParams::bayer_format` (native and
`hololink_module`) are `std::optional`.  When set, typically from
`camera->get_bayer_format()`, it selects the Bayer encoding (`BAYER_RGGB` and so on) stamped
on the emitted `holoscan::schema::ImageT`.  When unset, the image is stamped `CUSTOM`, so a
downstream demosaic uses its own grid position, as in HSDK4.

---

### 3. `PackedFormatConverterOp` is configured through `PackedFormatConverterParams`

**HSDK4:** the native operator took `allocator` and `cuda_device_ordinal` arguments, and a
sensor driver configured its geometry through the `CsiConverter` interface of the live
instance.

**HSDK5:** `graph.op<>()` defers construction, so there is no live instance to configure.
Like its `hololink_module` counterpart, the native operator is configured entirely through
`PackedFormatConverterParams` and is no longer a `CsiConverter`:

```cpp
hololink::csi::CsiGeometry geometry;
camera->configure_converter(hololink::operators::FusaCoeCaptureOp::make_geometry_adapter(
    geometry, /*packetizer_enabled=*/true));

auto converter = graph.op<hololink::operators::PackedFormatConverterOp>("converter");
converter.set_params(hololink::operators::PackedFormatConverterParams {
    .cuda_device_ordinal = cuda_device_ordinal,
    .bayer_format        = camera->get_bayer_format(),
    .geometry            = geometry,
});
```

`geometry` is required.  Its `start_byte` applies only with `header_included = true`, for a
producer whose frames still start with the header, such as `LinuxCoeReceiverOp`;
`FusaCoeCaptureOp` emits only the image lines, and `trailing_bytes` is never part of the
image.  Such a receiver's `DataChannel` needs the packetizer program
(`set_packetizer_program(csi::get_packetizer_program(pixel_format))`), which
`FusaCoeCaptureOp` sets itself.  In Python, `hololink.get_packetizer_program()` returns the
program (v2.7's `hololink.csi.get_packetizer_program()`), and
`hololink.resolve_packed_geometry(camera)` resolves the packed geometry in any build.  The
output comes from the runtime tensor pool, so there is no allocator; a frame is dropped when
the pool or a downstream queue is full.

---

## CRC Operators

Covers `ComputeCrcOp` and `CheckCrcOp`, native and `hololink_module`.

### 1. `ComputeCrcOp` and `CheckCrcOp` share a `CrcChannel`

**HSDK4:** `CheckCrcOp` was given the `ComputeCrcOp` instance (`compute_crc_op`), passed the
tensor through, and validated the CRC that operator had computed last.

**HSDK5:** `graph.op<>()` builds operators from replayable factories, so one operator can't
be handed another.  The application creates one `hololink::operators::CrcChannel`
(`src/hololink/operators/compute_crc/crc_channel.hpp`) and passes it to both `graph.op<>()`
calls:

```cpp
auto crc_channel = std::make_shared<hololink::operators::CrcChannel>();
auto compute_crc = graph.op<hololink::operators::ComputeCrcTensorOp>(
    "compute_crc", crc_channel, cuda_device_ordinal, frame_size);
auto check_crc = graph.op<MyCheckCrcOp>("check_crc", crc_channel);  // derived from CheckCrcOp
```

`ComputeCrcOp` publishes each frame's CRC keyed by its `frame_number`, and `CheckCrcOp`
takes the CRC of the frame it receives, waiting for the GPU only for that frame.  A checker
that runs behind, or a frame dropped between the two operators, therefore never validates a
frame against another frame's CRC.

- `ComputeCrcOp` takes a tensor/metadata pair and forwards it unchanged.  It is a template on
  the payload type: `ComputeCrcTensorOp` (`holoscan::Tensor`) or `ComputeCrcImageOp`
  (`holoscan::schema::ImageT`).  In Python they are `hololink.operators.ComputeCrcOp` and
  `hololink.operators.ComputeCrcImageOp`.
- Send `ComputeCrcOp`'s `tensor` output to an operator that allocates its own tensors, such
  as a converter.  HSDK5 gives an operator a CUDA stream only when it allocates tensors or
  receives them from an operator that does, so an operator that reads the forwarded tensor
  without allocating its own fails the run.
- `CheckCrcOp` takes only metadata (`in_metadata` in, `metadata` out, which may be left
  unconnected) and is no longer a template.  Feed it from a point that keeps the
  `frame_number` `ComputeCrcOp` saw: the converter after it, for instance, but not a
  sub-frame `CsiToBayerOp`, which renumbers frames; take the metadata ahead of that one.
  Where it branches off the chain, connect it with `holoscan::QueuePolicy::kDropOldest`,
  as any consumer that takes only metadata (Receiver Operators §2).
- Override `check_crc(computed_crc, metadata)` to compare the CRC with the HSB's
  `metadata.crc`; `check_crc(computed_crc)` and `get_last_crc()` remain.  The computed CRC is
  no longer written to the metadata (HSDK4's `computed_crc` key).
- `ComputeCrcOp::get_computed_crc()` (native and `hololink_module`) and the `hololink_module`
  `CheckCrcParams` are removed; the `hololink_module` operators take the channel as their
  constructor argument too.

---

## Example applications

The C++ example players take the options of their HSDK4 versions, with the same defaults
except `--ibv-name`'s: the device on the board's network interface rather than the host's
first one (Receiver Operators §14).  Two options are new in HSDK5:

- `fusa_coe_vb1940_player --bayer-grid-pos N` overrides the Bayer phase that the converter,
  `ImageProcessorOp` and `BayerDemosaicOp` use (0 BGGR, 1 RGGB, 2 GBRG, 3 GRBG).  The
  default, -1, keeps the sensor's GBRG.
- `fusa_coe_test_sensor_receiver --timeout-s S` ends the run after `S` seconds when
  `--frame-limit` frames haven't arrived by then, and exits 1.  The default, 0, sets no
  deadline; a sensor that sends nothing still fails the run after `--frame-limit` capture
  timeouts (15 s by default), about when HSDK4's run ended.

The latency tools (`linux_imx274_latency`, `roce_imx274_latency`) are C++ versions of the
HSDK4 Python tools.  `--hololink` defaults to the first address in `HOLOLINK_IPS`, else
192.168.0.2, and `--log-interval N` is new: it logs a progress line every `N` frames (0,
the default, logs none).

`--configuration` is still accepted, but HSDK5 applications read no YAML configuration: a
player logs that it ignores the file, and fails, as HSDK4's `Application::config()` did,
when the file doesn't exist.
