# In-process PCIe hololink_module (`hsb_pcie`): design

The `hsb_pcie` module (`hololink_module/module/hsb_pcie/`, `.so`
`hololink_58c1419b-1e7a-4120-95fd-4efc156187c1.so`) carries the HSB control plane over a
Hololink PCIe endpoint without the `hsb_pcie` emulator daemon. The SDK keeps speaking HSB
register addresses; the ECB frames it builds are decoded in-process, dispatched onto BAR0
through the `hololink_pcie` kernel driver, and answered in-process. On top of it sits a
module-native receive path (`hololink_module/host/pcie/`, operator
`hololink::module::operators::PcieReceiverOp`) that programs the FPGA's DMA registers with
plain `HololinkInterfaceV1::write_uint32` / `read_uint32`.

Paths below are relative to the `hololink` repo unless they start with `hsb_pcie/`, which is
the sibling repo holding the kernel driver, the emulator daemon and the reference bring-up
tree `hsb_pcie/hololink_pcie_bringup/` (its `driver/`, `include/hololink_pcie_uapi.h` and
`include/hololink_pcie_regs.h` are byte-identical to `hsb_pcie/hsdk5_pcie_demo/`; the
reference C++ receiver lives in `hsb_pcie/hsdk5_pcie_demo/src/pcie/`).

## 1. Transport model

### HSB-address interface over an in-process bridge

`hololink::Hololink` funnels every control-plane access through `Hololink::execute()`,
which serializes on `execute_mutex_` and calls the virtual pair `send_control()` /
`receive_control()` (`src/hololink/core/hololink.hpp`). `PcieHololink`
(`hololink_module/module/hsb_pcie/pcie_hololink.{hpp,cpp}`) subclasses
`module_core::LegacyHololinkAccess` (itself a `hololink::Hololink`) and overrides that pair:

- `send_control(request)` hands the frame to `EcbBridge::process()` and parks the reply in
  `pending_reply_`.
- `receive_control(timeout)` returns and clears `pending_reply_`; the timeout is ignored
  because nothing is in flight.

Because `execute()` serializes requests, one pending-reply slot is enough. Everything the
legacy class builds on top of `execute()` -- `WriteData` batching, block reads,
`Hololink::I2c`, `Hololink::Spi`, `Hololink::Sequencer`, `configure_hsb()` -- works
unchanged, as do module_core's `HololinkV1`, `I2cV1`, `SequencerV1`, `DataChannelV1` and
`PtpPpsOutputV1`, which reach the board through `HololinkV1::legacy_access()`.

### ECB request/reply layout (`ecb_bridge.{hpp,cpp}`)

`EcbBridge` decodes exactly what `hololink.cpp` encodes and synthesizes what its readers
parse (constants from `hololink.hpp`):

| Field | Request | Reply |
|---|---|---|
| byte 0 | `cmd`: `WR_DWORD` 0x04, `WR_BLOCK` 0x09, `RD_DWORD` 0x14, `RD_BLOCK` 0x19 | `cmd \| 0x80` (`EcbBridge::REPLY_BIT`) |
| byte 1 | `flags` (`REQUEST_FLAGS_SEQUENCE_CHECK` is accepted and ignored: no lossy carrier) | `flags` echoed |
| bytes 2-3 | `sequence` u16 BE | echoed |
| byte 4 | reserved | `response_code` |
| byte 5 | reserved | 0 |
| body | BE `(address[, value])` items; `RD_DWORD` carries a bare address, the others 8-byte pairs; `n = ((len - 6) + 4) / 8` (`EcbBridge::pair_count`) | one BE `(address, value)` pair per request item, in order |
| trailer | -- | `latched_sequence` u16 BE (the most recent request's sequence) + 2 zero pad bytes |

Dispatch semantics: items are dispatched in order through the `AddressPolicy`; the first
failure stops dispatch, keeps the earlier writes, and sets `response_code`
(`RESPONSE_INVALID_ADDR` = 3 for rejected/unmapped/misaligned addresses,
`RESPONSE_ERROR_GENERAL` = 2 when the endpoint failed the access). Items not reached echo the
request value (0 for reads). An unknown `cmd` yields `RESPONSE_INVALID_CMD`; a frame too
short to decode yields `RESPONSE_INVALID_PKT_LENGTH`; both carry no pairs. An empty request
yields an empty reply, which the SDK treats as a timeout and retries. A software-sequencer
trigger armed by any item of the request runs once after the whole item loop and before the
reply is built (see "The software APB sequencer" below).

### Why `Hololink::start/stop/reset` became virtual, and what the overrides do

`Hololink::start()`, `stop()` and `reset()` were non-virtual; `post_reset_configuration()`
and `started_` were private. The PCIe carrier cannot reuse their bodies: `start()` opens the
UDP control socket and the async-event socket/thread, `reset()` waits up to 30 s for a bootp
announcement via `Enumerator::find_channel`. The core change (`git diff
src/hololink/core/hololink.hpp`) makes the three methods `virtual`, moves `started_` to a
`protected:` block and documents that a transport subclass overriding `reset()` must call
`post_reset_configuration()`. No new interface was added; behaviour is changed by
subclassing.

`PcieHololink` overrides:

- `start()`: idempotent on `started_` (same compare-exchange latch as the base). Opens the
  endpoint if it is not already open, captures BDF/UUID (`remember_identity()`), programs the
  I2C block's timing directly on BAR0 (`program_i2c_timing()`: `I2C_CTRL_BASE + I2C_CLK_CNT`
  = 0x4020C <- `AddressPolicy::i2c_clk_cnt()`, 400 kHz SCL at the CertusPro-NX Versa's 60 MHz
  APB clock, i.e. 75;
  `I2C_CTRL_BASE + I2C_TIMEOUT` = 0x40210 <- `I2C_TIMEOUT_VALUE` 5000000), releases the
  sensor resets (`release_sensor_resets()`:
  read-modify-write of `HLNK_CTRL_1[1:0]` to `0x3`, followed by `SENSOR_RESET_SETTLE_MS` =
  10 ms only when a bit was actually held -- on the CertusPro-NX Versa bit 0 is the IMX258's
  hardware reset on J37 and a freshly configured design holds it, so without this the
  camera's first I2C transaction NAKs), then runs `configure_hsb()` through the
  bridge. **No sockets are opened and no thread is started**; `hololink_module_hsb_pcie_hololink_test`
  (`StartOpensNoSocketsAndConfiguresHsb`) asserts the `/proc/self/fd` count is unchanged.
  On failure the object is left not-started and an endpoint opened by this call is closed.
- `stop()`: clears `started_` and closes the endpoint. The destructor calls `stop()` first so
  `Hololink::~Hololink` never sees `started_` set with no sockets behind it.
- `reset()`: calls the inherited `trigger_reset()` (0x08 <- 0, 0x08 <- 0x3, 0x04 <- 0x8).
  All three are forwarded to the FPGA (see the policy table), so the endpoint genuinely
  resets under the driver. The override then waits for the design to come back by issuing
  *live* BAR0 reads of `HSB_VER` and `HSB_UUID_3..0` (`probe_live_identity()`, re-opening the
  endpoint if the reset dropped it) until `HSB_VER` is neither 0 nor all-ones and the UUID
  equals the one captured at construction/`start()`, bounded by
  `reset_identity_timeout()` (default `RESET_IDENTITY_TIMEOUT_S` = 5 s, polled every
  `RESET_POLL_INTERVAL_MS` = 50 ms). The driver's IDENTITY ioctl is deliberately not used
  for this step: it serves `hsb_version` and `VERSION_VALID` from its probe-time cache and
  only re-reads the UUID, so it says "valid" while the FPGA is still mid-reset. Once live,
  the BDF reported by the endpoint must equal the captured one (a changed BDF or a live but
  different UUID throws), the I2C timing is re-programmed, `SW_SEN_RST` is restored to the
  end state `trigger_reset()` intends (`release_sensor_resets()`: read-modify-write of
  `HLNK_CTRL_1[1:0]` to `0x3` -- `trigger_reset()` writes `0x3` there *before* `SYS_RST`, and the
  design reset returns the word to `0`; Ethernet parity, not a streaming fix: the reference
  PCIe tooling drives these bits as the host-display framer reset and streams the camera over
  Sensor RX with the word at `0`), and `post_reset_configuration()` runs. No bootp
  `find_channel`. `start()` deliberately does not touch the register: driving the design into
  a known state is `reset()`'s job. **The PCIe examples do not call `reset()` by default**
  (`--reset` opts in): on the bitfiles seen so far the sensor receive path does not come back
  from `SYS_RST` until the FPGA is reconfigured. Established on an AGX Orin with bitfile
  `0x5c694fdc`: the reference player streamed, the module player then ran with `reset()` and
  received nothing while the sensor demonstrably kept emitting frames, and from then on the
  reference player received nothing either, until the design was reloaded. The reference PCIe
  tooling never writes `SYS_RST`. `PcieHololink::reset()` logs a warning to that effect every
  time it runs (section 6).

### Address policy (`address_policy.{hpp,cpp}`)

The module's address space is BAR0 offsets (0x00000-0xFFFFF) plus the HSB aliases the SDK
still emits. `AddressPolicy` is an ordered range table (first match wins) with actions
`ForwardIdentity`, `ForwardRebase`, `Shadow`, `ReadOnlyConstant`, `ApbSequencer`, `Reject`;
an unmatched address takes `default_action()`, `Shadow` unless switched to `Reject` with
`set_default_action()`. Misaligned addresses are `RESPONSE_INVALID_ADDR` before lookup. The
table as built by `AddressPolicy::build_ranges()`:

| Range (name) | Action | Why |
|---|---|---|
| `0x0100-0x0187` (`ptp`) | Shadow | the SDK's Ethernet PTP words; the endpoint's own PTP block at the same BAR0 offsets is the kernel driver's (`bootp_ptp` / `ptp_clock`), never the SDK's |
| `0x0188` (`ptp_sync_stat`) | ReadOnlyConstant `0xF` (`PTP_SYNC_STAT_LOCKED`) | `FPGA_PTP_SYNC_STAT` reads "locked" so `ptp_synchronize()` returns instead of waiting 30 s |
| `0x018C-0x01FF` (`ptp`) | Shadow | rest of the PTP block |
| `0x0200-0x020F` (`ctrl_evt`) | Shadow | CTRL_EVT scratch / rising / falling / clear: the endpoint bitfile does not execute CTRL_BUS_EVT programs, so these stay host-side for the software sequencer |
| `0x0210-0x0223` (`ctrl_evt_host`) | Shadow | host MAC/IP/UDP ports and FPGA UDP port: Ethernet-only async-event delivery |
| `0x0224-0x022B` (`ctrl_evt_apb`) | Shadow | `CTRL_EVT_APB_INTERRUPT_EN` 0x224, `CTRL_EVT_APB_TIMEOUT` 0x228: read by the software sequencer (the timeout bounds POLL) |
| `0x022C` (`ctrl_evt_sw_event`) | ApbSequencer | `CTRL_EVT_SW_EVENT`: shadowed, and a write of 1 arms the software sequencer's deferred trigger |
| `0x0230-0x0233` (`ctrl_evt_vlan`) | Shadow | event-channel VLAN word: Ethernet-only |
| `0x0280-0x0287` (`ctrl_evt_stat`) | Shadow | `CTRL_EVT_STAT` 0x280, `CTRL_EVT_FSM_STAT` 0x284: written by the software sequencer (error bits) |
| `0x2000-0x2FFF` (`apb_ram`) | Shadow | vector table + sequencer programs, executed from the shadow copy by the software sequencer; never reach the FPGA's EVT_RAM |
| `0x00000-0xFFFFF` (`bar0`) | ForwardIdentity | everything else inside BAR0, **including `0x04` SYS_RST**; the module-native ring programs PCIE_DP / DP_PKT / MSI-X / DATA_GEN / GPIO here |
| `0x01000000-0x01FFFFFF` (`sif`) | Shadow | Ethernet SIF packetizer block, incl. the I2S FIFO at 0x01200000 |
| `0x02000000-0x02FFFFFF` (`hif`) | Shadow | Ethernet HIF block; RoCE/Linux/CoE data-channel configuration is unsupported on this transport |
| `0x03000000-0x030001FF` (`spi_alias`) | ForwardRebase -> `0x40000` (`SPI_BAR0_BASE`) | SDK `SPI_CTRL` is outside BAR0 (next subsection) |
| `0x03000200-0x030003FF` (`i2c_alias`) | ForwardRebase -> `0x40200` (`I2C_BAR0_BASE`); write hook on `+0x0C` | SDK `I2C_CTRL` is outside BAR0; the hook pins CLK_CNT |
| `0x40000000-0x6FFFFFFF` (`jesd`) | Reject | absent |
| `0x70000000-0x7FFFFFFF` (`vsync`) | Shadow | VSYNC / GPIO_MUX_EN word: Ethernet-only |
| `0x80000000-0xFFFFFFFF` (`i2s_apb`) | Reject | absent |
| anything else (e.g. `0x03000400` UART alias, `0x00100000-0x00FFFFFF`) | `default_action()` = Shadow | not mapped by the daemon either |

Shadow words read 0 until written. `ReadOnlyConstant` accepts and drops writes, as a
read-only hardware register would. The endpoint's own `Status::INVALID_ADDRESS` (misaligned
or outside `PcieEndpointInterface::BAR0_LEN`) maps to `RESPONSE_INVALID_ADDR`;
`READ_FAILED` / `WRITE_FAILED` / `NOT_ATTACHED` map to `RESPONSE_ERROR_GENERAL`.

### Why the I2C/SPI alias exists

BAR0 is 1 MiB: `USR_MODULE_END = 0x000fffff` (`hsb_pcie/hololink_pcie_bringup/lib/
hololink_pcie/regs.py`, mirrored as `hololink::pcie::USR_MODULE_END` in
`hololink_module/host/pcie/include/hololink/module/pcie/pcie_regs.hpp`). The SDK addresses the
I2C controller at HSB `I2C_CTRL = 0x03000200` and the SPI controller at `SPI_CTRL =
0x03000000` (`src/hololink/core/hololink.hpp`). On the FPGA those same nv_hsb_ip blocks
sit inside BAR0 at `I2C_CTRL_BASE = 0x40200` and `SPI_CTRL_BASE = 0x40000`
(`pcie_regs.hpp`), with byte-identical register offsets and bit fields. `0x03000200` is
outside the BAR window, so a write to it cannot reach the FPGA unrebased.

The emulator daemon already solves this BAR0 1 MiB issue entirely:
`HSBPCIEEmulator::add_default_forwarding_ranges()`
(`hsb_pcie/driver/src/hsb_pcie/hsb_pcie_emulator.cpp`) registers exactly two
`ForwardingRange`s, `I2C_CTRL -> I2C_CTRL_BASE` and `SPI_CTRL -> SPI_CTRL_BASE`, each
0x200 bytes, and `translate()` rebases by base-address subtraction. The module keeps
exactly that translation -- and only that one -- in-process: the `spi_alias` and `i2c_alias`
rows above are the sole `ForwardRebase` entries in the policy. Every other address is either
forwarded as-is, shadowed, or rejected; nothing else is re-based.

Two details the alias carries that the daemon did not:

- **CLK_CNT write hook.** `Hololink::I2c::set_i2c_clock()` (`hololink.cpp`) writes
  `0x19` to `I2C_CTRL + 0x0C`, which on this block's divider (`SCL ~= apb_clk / (2 *
  clk_cnt)`) is 2 MHz at a 100 MHz APB clock, not the 400 kHz the comment intends. The
  `AddressPolicy` constructor takes the board's APB clock (`apb_clk_hz`, default 100 MHz)
  and installs a `WriteHook` on `I2C_ALIAS_CLK_CNT` (0x0300020C) that rewrites whatever
  value arrives to `i2c_clk_cnt()` = `clk_cnt_for(I2C_SCL_HZ, apb_clk_hz)` (same arithmetic
  as `pcie_regs.hpp` `clk_cnt_for`): **75** at the CertusPro-NX Versa's 60 MHz, the
  policy's default and the only clock `create_backing()` builds it with.
- **I2C_TIMEOUT has no HSB register.** `I2C_TIMEOUT = 0x010` (BAR0 0x40210) exists in the
  shared RTL but the SDK never writes it (there is no HSB-side name for it). The module's
  `PcieHololink::start()` programs it directly on the endpoint, together with CLK_CNT, in
  `program_i2c_timing()`; `reset()` re-programs both after SYS_RST.

### The software APB sequencer

APB sequencer programs are **always** executed in-process; there is no engine selection.
Hardware finding (IGX, endpoint bitfile, `hololink_pcie_identity --peek/--poke` and the
IMX258 player): the `CTRL_BUS_EVT` / `EVT_RAM` blocks on the endpoint bitfile accept and
read back writes over PCIe but do not execute the programs written to them -- forwarding
`CTRL_EVT` + `APB_RAM` to BAR0 made the SDK's very first `I2c::i2c_transaction()` time out
waiting for the pulse on `CTRL_EVT_SW_EVENT` to complete. Running the same SDK programs
through the in-process sequencer instead brought the IMX258 up and verified all 82 sensor
registers (`--verify-registers`, zero mismatches). This part of the Hololink IP will not
change, so the module ships only the software engine and the address policy has no mode.

`CTRL_EVT` (except the Ethernet-only host/VLAN words, which are shadowed anyway) and
`APB_RAM` are `Shadow`, `0x22C` is `ApbSequencer`, and `ApbSequencer`
(`apb_sequencer.{hpp,cpp}`) -- a port of `handle_apb_sw_event` / `sequence_poll` in
`src/hololink/emulation/src/common/apb_events.cpp` -- executes the program the
`SW_EVENT_VECTOR` (`0x2000 + 4 * SW_EVENT`, `SW_EVENT` = 2) points at, over the shadow copy
of APB RAM. Opcodes are `Hololink::Sequencer::Op` (`POLL` 0, `WR` 1, `RD` 2), decoded
exactly as `add_command`/`append` encode them (2-bit commands, a fresh command word every 16
commands, `DONE_COMMAND_ADDRESS` 0xFFFFFFFF terminator). Every WR/RD/POLL address is
dispatched back through the policy, so an embedded `0x03000200` reaches BAR0 `0x40200` just
like a direct SDK write. POLL is a timed loop bounded by shadow `CTRL_EVT_APB_TIMEOUT` in
`APB_TIMEOUT_UNIT_S` = 51.2 ns units (`DEFAULT_TIMEOUT_S` = 1 s when unset); errors set
`FSM_ERROR_MASK` (bit 28) in shadow `0x284` and `STAT_APB_ERROR_MASK` (bit 31) in shadow
`0x280`; `MAX_COMMANDS` = 4096 guards a program that never reaches DONE.

**Deferred SW_EVENT trigger.** `Hololink::I2c::i2c_transaction()` queues `SW_EVENT=1`,
`SW_EVENT=0` and only then `CTRL_EVT_APB_TIMEOUT` in one `WR_BLOCK`. On an Ethernet HSB the
sequencer runs asynchronously, so the later timeout write lands before a long POLL expires.
In-process that cannot happen, so `AddressPolicy::dispatch_write()` only records a pending
trigger (`sequencer_trigger_pending()`) and `EcbBridge::process()` fires it
(`fire_pending_sequencer_trigger()`) once every item of the request has been applied, even
when a later item failed. A trigger re-armed from inside a running program is dropped with a
warning, not re-entered. `PcieHololink` installs the sequencer (`ApbSequencer::install()`)
for its whole lifetime and uninstalls it in its destructor.

`hololink_pcie_identity --bdf <bdf> --peek OFF` / `--poke OFF VAL`
(`hololink_module/tools/hololink_pcie_identity.cpp`) remains the general BAR0 bring-up aid
(e.g. `--peek 0x0` for `GLB_SCRATCH`, `--peek 0x284` for `CTRL_EVT_FSM_STAT`).

## 2. Enumeration standard

### Discovery: the kernel driver announces PCIe endpoints over bootp

The `Adapter` discovers devices through exactly one mechanism, its bootp listener on UDP 12267
(`Adapter::start_bootp_listener`, `hololink_module/host/src/bootp.cpp`), as it did before the
PCIe work; `adapter.hpp` carries nothing PCIe-specific. PCIe endpoints reach that listener the
way Ethernet boards do: the `hololink_pcie` kernel driver publishes a bootp announcement for
every bound endpoint.

- **Publisher** (`src/ehsb/kmod/hololink_pcie_drv.c`, `hl_pcie_bootp_work_fn`): one delayed
  work on the shared system workqueue, scheduled at probe (first announcement immediately, so a
  hot-plugged board is visible within a second) and re-armed every `HZ` while any device is
  bound. Under `hl_pcie_devices_lock` it walks the device array and, for each endpoint whose
  UUID is valid (`HL_PCIE_ID_F_UUID_VALID`; an all-ones or absent identity names no module and
  is not announced), fills a 300-byte RFC 951 payload and injects it into the loopback
  interface with `netif_rx()`: Ethernet header (all-zero addresses, so `eth_type_trans()` on
  `lo` yields `PACKET_HOST`), IPv4 `<packed BDF> -> 127.255.255.255` -- the **destination** is
  the loopback subnet broadcast, so the announcement is delivered to **every** socket bound to
  port 12267 on `INADDR_ANY`, as an Ethernet board's broadcast is (a unicast to `127.0.0.1`
  reaches exactly one of the sockets sharing the port, most specific bind wins, so a process
  binding `0.0.0.0:12267` after the discovery daemon would take every announcement away from
  it; a socket bound to `127.0.0.1` itself does not match the broadcast and never sees it); the
  **source** is the BDF packed as `(domain << 16) | (bus << 8) | devfn`, the same value as
  `ciaddr`, so `0005:01:00.0` sends from `0.5.1.0` -- UDP `12268 -> 12267` with checksum 0, and
  -- the load-bearing detail -- the output route from `ip_route_output_key()` attached with
  `skb_dst_set()`. Without it `ip_route_input_slow()` drops a hand-built loopback frame as a
  martian destination (`route_localnet=0`); with it no input routing runs and nothing depends
  on a sysctl. The route is the local table's `broadcast 127.255.255.255 dev lo` entry, which
  the kernel adds alongside `127.0.0.1/8`: `RTN_BROADCAST`, flagged `RTCF_BROADCAST|RTCF_LOCAL`
  with `ip_local_deliver()` as its input hook, so `__udp4_lib_rcv()` takes the
  `__udp4_lib_mcast_deliver()` fan-out rather than the single-socket unicast lookup. It is
  looked up with the loopback source, not the packed BDF: a
  `0.0.0.0/8` source is refused by `ip_route_output_key()` (zeronet) and would be a martian
  source on input, and with the dst pre-attached neither check is reached. `hl_pcie_remove` calls
  `cancel_delayed_work_sync()` first, outside the device lock, then unlinks the device and
  re-arms for the ones that remain; `hl_pcie_exit` cancels once more after
  `pci_unregister_driver`. `bootp_announce=0` disables publishing. Only the initial network
  namespace sees the announcements.
- **Payload** (`src/ehsb/kmod/include/hololink_pcie_bootp.h`, plain C over `__u8` so the same
  inline filler compiles in the kmod and in the userspace test; mirrored to
  `hololink_module/host/pcie/include/hololink/module/pcie/hololink_pcie_bootp.h` and checked
  byte-identical at configure time): `op=1`, **`htype=0x50`** (`HL_PCIE_BOOTP_HTYPE_PCIE`,
  ASCII 'P'; not an IANA hardware type, which stop at 37 -- Ethernet boards send 1),
  **`hlen=4`**, `xid` = `(domain << 16) | requester_id`, `secs` since bind, **`ciaddr` =
  `(domain << 16) | (bus << 8) | devfn`**, **`chaddr[0..3]`** = the same BDF big-endian
  (`dom_hi, dom_lo, bus, devfn`), and the HSB vendor extension laid out exactly as the Ethernet
  firmware's (`docs/user_guide/ip_integration.mdx`, Table 2): tag `0xE0`, **len 56**, `'NVDA'`,
  `data_plane 0`, `enum_version 2`, `compat_id 0`, the 16-byte board UUID in the driver's byte
  order (below), packet count, **`serial[7]`** = the endpoint's PCIe Device Serial Number
  (`pci_get_dsn()`, EUI-64: its 7 least-significant bytes LSB first, the firmware's little-endian
  serial order, dropping the top OUI byte) or, without a DSN capability (0 or all-ones), the
  placeholder `"PCIE"` + domain low byte, bus, devfn; `hsb_ip_version` = `HSB_VER & 0xFFFF`,
  `fpga_crc 0`; vendor 41..44 reserved (0); **45..50 PTP seconds BE48, 51..54 PTP nanoseconds
  BE32** -- a SNAPSHOT of the endpoint's `bar_ptp` clock taken just before the send when the
  driver runs with `bootp_ptp=1`, zero otherwise; **55..56 PSN** BE16 (`pkt_count & 0xFFFF`);
  **57 status**: bit0 `PTP_VALID` (the timestamp is live; the Ethernet firmware's bit0 means PTP
  enabled and a SYNC received), bit1 `SERIAL_DSN` (the serial is the DSN, not the placeholder).
- **Host parser** (`bootp.cpp`): transport-agnostic and untouched for PCIe. Whatever the
  `htype`, `mac_id` is the hardware address colon-hex (`"00:05:01:00"` for the driver's record)
  and `peer_ip` the datagram's IPv4 source; `hardware_type` and `hardware_address_length` are
  reported as sent; `ciaddr` becomes `client_ip_address` rendered dotted-quad;
  `hardware_address` keeps the 16 raw bytes (trimmed to `hlen` by the Adapter); decoding of the
  vendor area stops at `fpga_crc`. (The legacy `src/hololink/core/enumerator.cpp` behind
  `hololink-enumerate`, which has no module to defer to, renders the BDF and decodes the tail
  itself.) `xid` becomes
  `transaction_id`; `IP_PKTINFO` gives `interface = "lo"` and `destination_address =
  "127.255.255.255"`; `interface_address` is `ipi_spec_dst`, which for a broadcast from a
  `0.0.0.0/8` source `fib_compute_spec_dst()` resolves through `inet_select_addr()` to a
  non-loopback address of the host (informational; nothing consumes it. A userspace sender
  from `127.0.0.1` still yields `127.0.0.1`).
  The parser also stamps `control_port = 8192` and `sequence_number_checking = 1`.
- **`hsb_pcie` module, `update_metadata`** (`hsb_pcie_publisher.hpp`; runs after the UUID
  match, before the shared `HsbIpPublisher` path and before the Adapter matches `register_ip`
  / `wait_for_channel` subscribers on `peer_ip`): for **`htype 0x50`** with a 4-byte `chaddr`
  the bytes are rendered as the textual BDF (`"0005:01:00.0"`, `pcie::bdf32_to_string`) into
  **both `mac_id`** (the hardware address in its htype's convention; also stamped as
  `hardware_address_text`) **and `peer_ip`** -- the channel key applications wait on. The
  datagram's IPv4 source it replaces is not kept separately: the driver sources from the
  packed BDF (`"0.5.1.0"`), the same value `ciaddr` carries, so `client_ip_address` already
  stands for it (and keeps an htype-agnostic consumer at one peer per endpoint). When the
  vendor length says the tail is present (56; older PCIe drivers said 39, the emulator 4) the
  module also decodes, from the raw packet it is handed, `ptp_timestamp_s`,
  `ptp_timestamp_ns`, `bootp_psn` and `bootp_status` (the raw status byte: bit0 the PTP
  timestamp is live, bit1 the serial is the DSN; no derived flag keys are published, consumers
  test the bits). Raw subscribers (`register_raw_*`, `adapter_enumerator
  --raw`) therefore see the parser's view of a PCIe endpoint -- `peer_ip` the source address,
  `mac_id` `"00:05:01:00"`, no tail keys -- and every post-enrichment subscriber the BDF.
- **Module stamps the BDF and decides the serial** (`HsbPciePublisher::update_board_metadata`,
  `module/hsb_pcie/hsb_pcie_publisher.hpp`): the `.so` is selected by the board UUID like any
  module, and `EnumerationInterfaceV1::update_metadata` runs on a mutable record **after** the
  raw subscribers and **before** `register_ip` / `wait_for_channel` matching
  (`Adapter::enumerate`, `adapter.cpp`). Shape 1a (current driver, `hardware_type == 0x50`):
  `peer_ip` is already the BDF; the module stamps `pcie_bdf`, `pcie_domain/bus/device/function`
  (`pcie::bdf_string_to_u32`) and `transport = "pcie"`, and logs a `client_ip_address` that
  disagrees with `chaddr` (chaddr wins). Shape 1b (older driver,
  dotted-quad `peer_ip` + parseable `client_ip_address`): `peer_ip` is rewritten to
  `DDDD:BB:DD.F` (`pcie::dotted_quad_to_u32`, `pcie::bdf32_to_string`) with the same stamps and
  `pcie_source_ip` = the source; `127.0.0.1` (userspace senders) is accepted silently, any
  other source mismatch is logged and `ciaddr` wins. **Serial rule** for both bootp shapes: the
  announced 14-hex serial is kept when `bootp_status` bit1 says it is the Device Serial Number
  (`pcie_serial_source = "dsn"`, unique per board), otherwise the placeholder is replaced by
  `serial_number = "pcie-<bdf>"` (`pcie_serial_source = "bdf"`, unique per slot); both are
  filesystem-safe for `NamedLock` and are the `serial=<...>` instance key. Applications
  therefore keep addressing the endpoint by BDF: `wait_for_channel("0005:01:00.0")` resolves on
  the first announcement after the wait started (within one cadence). A dotted-quad `peer_ip`
  **without** a parseable client address -- an Ethernet-shaped record -- is declined
  (`HOLOLINK_MODULE_ENUMERATION_SKIPPED`). Hand-built metadata with `peer_ip` already a BDF
  (no `hardware_type`; keeps its serial), or a `/dev/hololink_pcieN` / `by-bdf` alias with
  `transport = "pcie"`, is accepted as before. `mac_id` is not consulted.
  `PcieHololinkV1::create_backing()` then resolves the `/dev/hololink_pcieN` minor from the BDF
  through sysfs (`pcie::minor_for_bdf`, via the class-device link under
  `/sys/bus/pci/devices/<bdf>/hololink_pcie/`), independent of the custom sysfs attributes.

Raw subscribers (`register_raw_ip` / `register_raw_all`, and the legacy `hololink-enumerate`)
see the pre-module record, which since the htype-0x50 announcement already has `peer_ip =
mac_id = "0005:01:00.0"` and the tail fields; what they lack is only the module's `pcie_*`
stamps and the serial decision. `Adapter::enumerate` skips module loading when no
post-enrichment subscriber is registered. There is no "gone" notification (as for Ethernet). The examples simply
`wait_for_channel(bdf)` (`discover_device()` in `examples/module_pcie_imx258_player.cpp`; `--hololink`
may be omitted when `scan_pcie_devices()` -- a sysfs read, not discovery -- lists exactly one
endpoint). `adapter_enumerator` prints every announcement, or once per `peer_ip` with
`-d/--dedup` (the BDF for PCIe, the IPv4 address for Ethernet); `--raw` shows the parser's
output without loading modules.

### Metadata a PCIe endpoint ends up with

Integer values are `int64_t`. Origin: **P** = bootp parser, **M** = `hsb_pcie` module,
**A** = `Adapter::use_sensor`.

| Key | Value | Origin |
|---|---|---|
| `peer_ip` | BDF `"dddd:bb:dd.f"` (lowercase) -- **the channel key**. Rendered from the 4-byte `chaddr` by the module's `update_metadata` for `htype 0x50`; restored from `client_ip_address` by the module for an older driver's `htype 1` record. The parser delivers the datagram's IPv4 source | M (P) |
| `hardware_type`, `hardware_address_length` | `0x50`, `4` | P |
| `client_ip_address` | the BDF as the parser rendered `ciaddr`, e.g. `"0.5.1.0"`; the driver sources the datagram from the same packed value, so this also stands for the IPv4 source | P |
| `pcie_source_ip` | only on an older driver's `htype 1` record: its pre-rewrite `peer_ip` (the IPv4 source) | M |
| `pcie_bdf` | same as `peer_ip` | M |
| `pcie_domain`, `pcie_bus`, `pcie_device`, `pcie_function` | ints from the BDF | M |
| `mac_id`, `hardware_address_text` | the BDF text (the hardware address in its htype's convention); informational. The parser delivers `mac_id` as the colon-hex bytes (`"00:05:01:00"`) and no `hardware_address_text` | M (P) |
| `ptp_timestamp_s`, `ptp_timestamp_ns` | the endpoint's PTP clock as snapshotted for this announcement (`bootp_ptp=1` and a bitfile with `bar_ptp`), else 0 / 0; `bootp_status` bit0 says which; decoded from the raw packet | M |
| `bootp_psn`, `bootp_status` | announcement sequence number; the raw status byte (bit0 PTP timestamp live, bit1 serial is the DSN) -- the only flag published, consumers test its bits | M |
| `transaction_id` | `(domain << 16) \| requester_id` | P |
| `interface`, `destination_address`, `interface_address` | `"lo"`, `"127.255.255.255"`, and `ipi_spec_dst` -- for the driver's broadcast from a `0.0.0.0/8` source a non-loopback host address chosen by `inet_select_addr()`, `"127.0.0.1"` for a userspace sender (`IP_PKTINFO`; informational) | P |
| `fpga_uuid` | canonical lowercase hyphenated text of the board UUID | P |
| `serial_number` | the 14-hex Device Serial Number when the driver announced one (`pcie_serial_source = "dsn"`, unique per board), else `"pcie-" + bdf` (`"bdf"`, unique per slot); both filesystem-safe for `NamedLock` and the `serial=<...>` instance key | M (P) |
| `pcie_serial_source` | `"dsn"` or `"bdf"` | M |
| `hsb_ip_version` | `HSB_VER & 0xFFFF` | P |
| `transport` | `"pcie"` | M |
| `data_plane` | `0` (required by `HsbIpPublisher`) | P |
| `enum_version`, `compat_id`, `transmitted_packet_count`, `seconds`, `flags`, `hardware_address_length`, ... | as parsed | P |
| `control_port` | `8192` (parser constant; read only by the Ethernet default `HololinkV1::create_backing()`, which `PcieHololinkV1` overrides -- unused) | P |
| `sensor`, `vp_address` (`0x1000 + 0x40 * n`), `vp_mask`, `pcie_vector = n` | `PcieChannelConfigurationV1::use_sensor` (`hsb_ip_sensor_metadata(md, n, data_plane=0, sifs_per_sensor=1)`); `TOTAL_SENSORS` is 2 | A |

Optional app-supplied override read by the module (cannot travel in bootp; hand-built
metadata and tests only): `pcie_backend` (`device` default, or `loopback` with `pcie_minor`).
The removed APB engine-selection key, if still present in app metadata, is ignored with a
one-line warning.

**`fpga_uuid` byte order.** The FPGA holds the UUID in four registers, `HSB_UUID_3` (0xB8)
most significant. `uuid[0..3]` = `HSB_UUID_3` big-endian, `uuid[4..7]` = `HSB_UUID_2`
(0xB4), `uuid[8..11]` = `HSB_UUID_1` (0xB0), `uuid[12..15]` = `HSB_UUID_0` (0xAC); printing
the bytes in array order as 8-4-4-4-12 hex gives the canonical text. This is the order of
`hl_pcie_identity_info::uuid`, of the `uuid` sysfs attribute (`%pUb`), of the bootp vendor
`uuid[16]` (`hl_pcie_identity_uuid_bytes`), and of `HSB_PCIE_MODULE_UUID` in
`loopback_pcie_endpoint.hpp`. `uuid_from_text()` accepts hyphenated or bare 32-digit hex,
any case.

The low-level sysfs/ioctl helpers (`pcie_enumerator.{hpp,cpp}`: `scan_pcie_devices()`,
`read_device_info()`, identity precedence IDENTITY ioctl -> sysfs attributes -> raw BAR0 reads,
`device_node_open_error()`) remain for `hololink_pcie_identity` and the examples' single-endpoint
convenience; they are not a discovery path.

### Why the BDF, not the character device, is the `peer_ip` analog

On Ethernet the channel key is the address the control plane is sent to: the board's IP. On
PCIe that role is played by the PCI function -- its BAR0 is where every ECB-decoded register
access lands -- so the BDF (`"DDDD:BB:DD.F"`) is the key. `/dev/hololink_pcieN` is only the
driver's handle on that function: minors are assigned in probe order and move when a second
board is added, a slot is re-enumerated or the driver is reloaded, so keying on them would
make the channel identity depend on boot history. Everything that needs the node derives it
from the BDF at the moment of use, through the header-only resolvers in
`hololink_module/host/pcie/include/hololink/module/pcie/pcie_sysfs.hpp`: the class device is a
child of the PCI device (`device_create(class, &pdev->dev, ...)`), so
`/sys/bus/pci/devices/<bdf>/hololink_pcie/hololink_pcieN` names the minor directly, and a
walk of `/sys/class/hololink_pcie/*/{bdf,device}` is the fallback. The header is shared by
the enumerator (`hololink::module`), the module `.so` (which must not link the host library)
and the receive path; `pcie::bdf_for_peer()` also maps the convenience spellings
(`/dev/hololink_pcieN`, `/dev/hololink/by-bdf/<bdf>`) back to the BDF.

**Optional udev stable aliases.** The driver exports `bdf` and `uuid` as class attributes, so
udev can publish stable names next to the probe-ordered node:

```
SUBSYSTEM=="hololink_pcie", ATTR{bdf}=="?*", SYMLINK+="hololink/by-bdf/$attr{bdf}"
SUBSYSTEM=="hololink_pcie", ATTR{uuid}=="?*", SYMLINK+="hololink/by-uuid/$attr{uuid}"
```

(e.g. in `/etc/udev/rules.d/99-hololink-pcie.rules` next to the permission rule of section
4). They are a convenience for shells and scripts; the framework never needs them, and
`bdf_for_peer()` resolves a `by-bdf` alias back to the BDF when one is passed as
`--hololink`.

### Board UUID and the identity tool

The framework selects a board's module by the UUID the board reports. The hsb_pcie endpoint
bitfile reports **`58c1419b-1e7a-4120-95fd-4efc156187c1`** in BAR0 `HSB_UUID_3..0` and holds
those registers read-only (writes are refused), so that value is the board identity and the
module is built as `hololink_58c1419b-1e7a-4120-95fd-4efc156187c1.so`
(`pcie::HSB_PCIE_MODULE_UUID`, `hsb_pcie::HSB_PCIE_MODULE_UUID`). A future bitfile that
reports a different UUID needs either a second `add_hololink_module(... UUID ...)` entry or a
`hololink_<uuid>.so` symlink to this module in the module directory; nothing is written to
the board. The daemon-era UUID programming (`MSIXPCIE::program_uuid`) is therefore not part
of the module path; `pcie::program_uuid()` remains in the enumerator library for boards
whose registers are writable, but no tool calls it.

`hololink_pcie_identity` (`hololink_module/tools/hololink_pcie_identity.cpp`) is the
read-only companion for bring-up:

| Option | Effect |
|---|---|
| `--bdf DDDD:BB:DD.F` | select the device by PCI address, resolved to the minor via `minor_for_bdf()` (`/sys/class/hololink_pcie/*/{bdf,device}`) |
| `--minor N` | select `/dev/hololink_pcieN` directly (mutually exclusive with `--bdf`) |
| `--show` | print the identity (`read_device_info()`); the default action |
| `--peek OFFSET` | one 32-bit BAR0 read via `HL_PCIE_IOC_READ` (`pcie::bar_read32`) |
| `--poke OFFSET VALUE` | one 32-bit BAR0 write via `HL_PCIE_IOC_WRITE`, then readback |
| `--list` | `scan_pcie_devices()` table of every endpoint |

## 3. Receiver delta vs the reference design

Reference: `hsb_pcie/hololink_pcie_bringup/` (driver, uapi and regs byte-identical to
`hsb_pcie/hsdk5_pcie_demo/`); the reference C++ receiver is
`hsb_pcie/hsdk5_pcie_demo/src/pcie/pcie_rx_ring.cpp`, `pcie_device.cpp`,
`pcie_receiver.cuh`. Module path, split like the RoCE path into a **channel** the module
publishes and a **receiver** the host owns:

- **Channel (module side, owns every register):**
  `hololink_module/module/hsb_pcie/pcie_data_channel_v1.{hpp,cpp}` (`PcieDataChannelV1`),
  the impl of `PcieDataChannelInterfaceV1` declared in the module-owned public header
  `hololink_module/module/hsb_pcie/include/hololink/module/hsb_pcie/pcie_data_channel.hpp`
  (INTERFACE target `hololink::hsb_pcie::headers`, namespace `hololink::module::hsb_pcie`,
  section 5). It is the PCIe counterpart of `module_core::RoceDataChannelV1`: has-a with the
  per-channel `DataChannelInterfaceV1` anchor, keyed `serial=<n>;data_channel=<k>`, and its
  `attach_receiver(receiver, chunk_pages)` / `detach_receiver()` / `arm()` / `disarm()` /
  `diagnostics()` issue every PCIE_DP, MSI-X and DP_PKT access through the parent
  `HololinkInterfaceV1` (`write_uint32` / `read_uint32` / `or_uint32` / `and_uint32`), so the
  address policy applies to the channel like to any client.
- **Receiver (host side, owns memory and the interrupt):** `hololink_module/host/pcie/`
  (target `hololink::module::pcie_receiver`: `include/hololink/module/pcie/pcie_rx_geometry.hpp`
  -- geometry, `RxRingLayout` with `validate()` / `validate_buffer()`, `RxRingConfig` --
  `pcie_device_access.{hpp,cpp}`, `pcie_receiver.{hpp,cpp}` -- `PcieReceiver`,
  `parse_rx_metadata`). `PcieReceiver` holds no
  `HololinkInterfaceV1` and writes no register: it implements
  `hsb_pcie::PcieReceiverAttachmentV1` (`iova`, `window_bytes`, `buf_inc_pages`,
  `num_buffers`, `requester_id` from the device identity, `vector` = sensor,
  `disable_metadata`) and the channel reads those to program the endpoint. The receiver owns
  the interrupt side entirely: it enables its vector through the driver
  (`HL_PCIE_IOC_MSIX_CTRL`) in `start()`, waits on it (`HL_PCIE_IOC_MSIX_WAIT`), and disables
  it in `close()`; no userspace code writes the MSI-X table. The former host-side ring class
  is gone; its arithmetic moved into `pcie_rx_geometry.hpp` and its HSB register sequence into
  the channel.
- **Operator:** `hololink_module/host/operators/pcie_receiver_op.cpp` (header
  `hololink_module/host/operators/include/hololink/module/operators/pcie_receiver_op.hpp`),
  the mirror of `roce_receiver_op.cpp`: `channel_ = PcieDataChannelInterfaceV1::get_service(md)`,
  `receiver_->start()`, `channel_->attach_receiver(receiver_, chunk_pages)`, monitor thread,
  `device_start`; `stop()` = `device_stop`, `channel_->detach_receiver()`, `receiver_->close()`,
  join.


### Carried over unchanged

| Item | Reference | Module |
|---|---|---|
| Geometry arithmetic: RAW10 `width*5/4` per line padded to `line_alignment`, `frame_dma_bytes = lines*height + raw_start_byte`, metadata block at `align_up(frame, 128)`, slot stride `align_up(metadata_offset + 128, 4096)`, `BUF_INC = slot_stride/128` (0 for one buffer) | `pcie_rx_ring.cpp` (`Raw10Geometry`, constructor) | `pcie_rx_geometry.hpp` (`RxFrameGeometry`, `RxRingLayout::compute`); parity pinned by `hololink_module_pcie_receiver_test` (`Raw10ParityWithLegacyPath`: `640 + 2400*1080`) |
| HSB register sequence: disarm this sensor's `DP_PKT_VIP_MASK` bit (the CertusPro-NX Versa has no data generator, so the reference's `DATA_GEN_ENA = 0` has nothing to address) -> `DP_PKT_LEN = chunk_pages` -> `PCIE_DP` words `BUF_BASE_HI` (`iova >> 32`), `REQUESTER_ID`, `BUF_INC`, `BUF_START` (`iova & 0xFFFFFFFF`), `BUF_LEN`, `BUF_PTR_RANGE` (`num_buffers - 1`), `METADATA_CTRL` (0) -> arm | reference ring `program()` + `start()` | `PcieDataChannelV1::attach_receiver()` (`pcie_data_channel_v1.cpp`), verbatim minus the reference's MSI-X table writes (see "Changed"); asserted by `hololink_module_hsb_pcie_test` `PcieDataChannelProgramsAndArmsPerSensor` over the loopback register file, which also pins every MSI-X entry word to its pre-attach value |
| Arm/disarm edges on `DP_PKT_VIP_MASK` (0x3030C) are **posted**, never read back | `start()` / `stop()` | `PcieDataChannelV1::arm()` / `disarm()` / the edges inside attach / detach (now per-bit RMW, below) |
| `HL_PCIE_IOC_MSIX_INFO` with `reset_count` zeroes the vector's completion counter before the first wait | `pcie_device.cpp` | `PcieReceiver::start()` (the routing fields the ioctl also returns are no longer used for programming; see "Changed") |
| Slot metadata layouts: current -- `flags` @0x00 BE32, `valid_bytes` @0x18 BE64, `frame_number` @0x22 BE16, zero-padded from 0x30; legacy WR-IMM -- @0x50/@0x68/@0x72 with little-endian fallbacks at @0x00/@0x18/@0x20; sentinel-filled block = not written | `read_metadata()` | `parse_rx_metadata()` (`RxMetadataLayout::CURRENT` / `LEGACY` / `NONE`) |
| Limits: `num_buffers` in [1, 4096] (12-bit `BUF_PTR_RANGE`), `chunk_pages` in [1, 255] (8-bit `DP_PKT_LEN`), 128-byte IOVA alignment, no 4 GiB straddle | constructor / `validate_buffer` | `RxRingLayout::validate()` / `validate_buffer()` in the receiver's `start()` (`RX_MAX_BUFFERS`), and again in `PcieDataChannelV1::validate()` on the attachment (`chunk_pages`, vector match, 4 GiB straddle) -- a rejected attachment returns `HOLOLINK_MODULE_INVALID_PARAMETER` before any register is written |
| Completion accounting: `delta = count - last`, `delta > 1` counts `delta - 1` dropped | `wait_irq()` | `blocking_monitor()` |

### Changed

| Item | Reference | Module | Why |
|---|---|---|---|
| Who programs the registers | the receive ring, on its own fd | the **module's** `PcieDataChannelV1` (`PcieDataChannelInterfaceV1`), keyed per `(serial, data_channel)` like `RoceDataChannelV1`; the receiver exposes what to program (`PcieReceiverAttachmentV1`) and never touches a register. `HsbPciePublisher::construct_board_services` publishes it next to the `DataChannelInterfaceV1` anchor | the same channel / receiver split as RoCE: one place programs the data plane on every transport, and the receive library needs no control plane at all |
| Register access | `PcieDevice::read32/write32` -> `HL_PCIE_IOC_READ/WRITE` on the ring's own fd | `HololinkInterfaceV1::write_uint32({bar0_offset}, {value})` / `read_uint32` on the per-board service the channel's anchor resolves (`PcieDataChannelV1::write_posted` / `read_reg`); the `hsb_pcie` module executes them against BAR0 through its address policy | one control plane for SDK and receiver; no `pcie_transaction`, no relay, no bypass of the address policy |
| Verified writes | plain writes | every configuration register is write + readback, mismatch throws naming the register (`write_verified`), after the channel disarmed itself; only the `VIP_MASK` edges stay posted | catches a BAR path returning all-ones or a register that ignores the write |
| Arm / disarm | whole-word posted writes of `1 << sensor` / `0` to `DP_PKT_VIP_MASK` | **read-modify-write of this sensor's bit only**: `arm()` = `HololinkInterfaceV1::or_uint32(DP_PKT_BASE + DP_PKT_VIP_MASK, vp_mask)`, `disarm()` = `and_uint32(..., ~vp_mask)` (`PcieDataChannelV1::set_vip_bit`, `vp_mask` from the metadata = `1 << sensor`); `attach_receiver()`'s initial quiesce and `detach_receiver()` are the same `and_uint32`; `PcieDataGenerator::quiesce_rx()` and `select_sensor_rx_source()` clear only their sensor's bit too. The module serializes the read and the write (`Hololink::execute`); the write itself is still posted. Asserted by `PcieDataChannelProgramsAndArmsPerSensor` (VIP_MASK 0x1 after sensor 0 attaches -> 0x3 after sensor 1 -> 0x1 after sensor 1 detaches -> 0x0) | the word is shared by both sensors; a whole-word write from one ring stopped (or started) the other sensor's stream |
| Device fd | one fd for everything, `flock(LOCK_EX)` on it | `PcieDeviceAccess` (`pcie_device_access.cpp`) opens `/dev/hololink_pcieN` -- derived from `peer_ip` (the BDF) via `pcie::device_node_for_bdf()` in `PcieReceiverOp::device_node_from_metadata()`; a literal node is accepted as a convenience -- **only** for dmabuf import / release, `HL_PCIE_IOC_DMA_ALLOC` / `HL_PCIE_IOC_DMA_FREE` + `mmap` (host-staging opt-in only), `HL_PCIE_IOC_MSIX_INFO`, `HL_PCIE_IOC_MSIX_CTRL`, `HL_PCIE_IOC_MSIX_WAIT`, `HL_PCIE_IOC_IDENTITY`; no flock | registers go through the module's channel; the fd is receiver-owned so the kernel's `quiesce_on_release` is scoped to the DMA owner |
| Frame memory | `HL_PCIE_IOC_DMA_ALLOC` coherent host memory, `mmap`, `memset` sentinel, then `cudaMemcpyAsync` H2D of a staging copy per frame | `cuMemGetHandleForAddressRange(CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD)` on the caller's VRAM + `HL_PCIE_IOC_DMABUF_IMPORT` (`import_dmabuf`); the slot is wrapped zero-copy in the output tensor. Host staging survives as an explicit opt-in (`alloc_host_staged`, `FrameBufferSource::DeviceAccess`, `frame_buffer_source: device`) | frames land in VRAM directly; the driver's `hl_pcie_dmabuf_check` enforces the address-generator constraints |
| MSI-X table ownership | `initialize_msix_table()` masks **and programs all** vectors on every ring construction from `HL_PCIE_IOC_MSIX_INFO`, then unmasks `vector = sensor`; the ring throws if `msg_addr > UINT32_MAX` | **the kernel owns the table; no userspace write to `0x8000-0x83FF` at all.** The driver requests every vector disabled (`IRQF_NO_AUTOEN`, eager `IRQ_DISABLE_UNLAZY`) and flips one with `enable_irq()` / `disable_irq()` on `HL_PCIE_IOC_MSIX_CTRL` (ioctl 14, per fd, refcounted across fds) or implicitly on the first `HL_PCIE_IOC_MSIX_WAIT`; on the 0 -> 1 transition it first restores the entry's address/data from the kernel's cached message (`get_cached_msi_msg`), so an entry wiped by `SYS_RST` comes back. `PcieReceiver::start()` enables its vector after the counter baseline; `close()` disables it after the monitor has drained and before the mapping is released; the fd's close drops any holds left. `PcieDataChannelV1` neither programs nor masks anything in the table -- the channel's surface is the HSB blocks only (section 5). Asserted by `PcieDataChannelProgramsAndArmsPerSensor` (every entry word identical before and after attach / detach) and by the receiver test (`ctrl_calls` = enable on start, disable on close; `UNSUPPORTED` from a driver without the ioctl fails `start()`) | the table format and its location are PCI-spec facts, not HSB registers, and the kernel already programs them; a userspace mask write behind the kernel's back is undone by the next affinity change, and a mask-all from one ring silenced the sibling's live vector. What remains board-specific about interrupts is the `vector == sensor` convention, stamped once by `PcieChannelConfigurationV1` |
| Completion wait | poll `HL_PCIE_IOC_MSIX_INFO` every 1 ms until `irq_count` changes | `HL_PCIE_IOC_MSIX_WAIT` (`wait_frame`), blocking with a 100 ms bound (`MONITOR_WAIT_MS`) so `stop()` / `close()` are noticed; `PcieReceiver::start()` reads `MSIX_INFO` with `reset_count`, takes a baseline `wait_frame(vector, 0, 0, last)`, then enables the vector (`MSIX_CTRL`) before the channel arms -- a completion raised between arm and the monitor's first wait sits in the pending-bit array and is delivered on the spot, so the wait itself is unchanged by the kernel-owned table. `stop()` only stops the monitor and wakes `get_next_frame()` waiters (disarming is the channel's `detach_receiver()`); `close()` additionally waits for `blocking_monitor()` to return before releasing the mapping, so a frame completed during the disarm is still decoded on live memory (host-staged path) | no polling floor on latency, no ioctl per poll |
| Slot selection | scan every slot for `valid_bytes == frame_bytes` and the newest unseen `frame_number` | same rule (`PcieReceiver::find_newest_frame()`): scan every slot's metadata block, skip blocks with zero bytes or a `(slot, frame_number)` already delivered, pick the frame furthest ahead of the last delivered one (mod 2^16), with a 50 ms settle window after the interrupt; metadata blocks are zeroed at `start()` so a reused buffer cannot masquerade as this run | carried over deliberately: the endpoint's write pointer survives re-programming of `BUF_PTR_RANGE` (on hardware a run after a 30-frame player run began at slot 3), so a pointer-tracking receiver reports empty slots; the scan follows the data |
| Frame number | 16-bit wire value | extended across wraparound to a monotonic 64-bit counter (`extended_frame_number_`) | downstream sees a monotonic counter |
| Metadata decode | local parser only | `FrameMetadataInterfaceV1::decode` when the module publishes the service and the block is `RxMetadataLayout::CURRENT` (the 48-byte HSB EOF block + zero padding), else the local parser (`PcieReceiverOp::compute`) | psn / crc / timestamps come from the shared decoder |
| Quiesce / GPIO in `program()` | zeroes `MEMRD_FETCH` and `TX_STRM_CTRL`, RMWs `GPIO_O_0` / `GPIO_T_0` (HDMI select, RAW10 select), checks `HSB_VER == 0x2606` and `GLB_SCRATCH != 0xffffffff` | only this sensor's `VIP_MASK` bit cleared (in `attach_receiver()`); the CertusPro-NX Versa has neither the reference board's data generator nor its ISP/HDMI GPIO mux, so there is nothing else to quiesce or select; liveness is the endpoint's concern (`PcieEndpoint::open()` validates `HSB_VER`) | the channel programs the data plane; board setup is not a receiver property |
| Operator | `PcieReceiverOp` in `pcie_receiver.cuh`: `(minor, sensor, num_buffers, geometry, fps, frame_limit)`, `OnClock` trigger, H2D copy per compute | `hololink::module::operators::PcieReceiverOp`: params `enumeration_metadata`, `frame_context`, `frame_size` (0 = derive), `device_start`, `device_stop`, `out_tensor_name`, `rename_metadata`, `sensor` (-1 = metadata `sensor` key; an explicit value must match the channel's), `num_buffers`, `chunk_pages` (passed to `attach_receiver`), `window_bytes`, `width`, `height`, `stride_pixels`, `crop_x`, `line_alignment`, `raw_start_byte`, `frame_buffer_source` (packed RAW10 only); device node derived from `peer_ip` (the BDF) through sysfs; `AsynchronousCondition` driven by the monitor thread; the data channel is `PcieDataChannelInterfaceV1::get_service(metadata)` -- `start()` = receiver `start()`, `channel->attach_receiver(receiver, chunk_pages)`, monitor thread, `device_start`; `stop()` = `device_stop`, `channel->detach_receiver()`, receiver `close()`, join | mirrors `RoceReceiverOp` one for one; enumeration linkage replaces a bare `minor` argument |
| Emitted metadata | `frames`, `frame_number`, `slot`, `irq_delta` on stderr | `flags`, `psn`, `crc`, `frame_number`, `timestamp_s/ns`, `bytes_written`, `metadata_s/ns`, `received_s/ns`, `frame_memory`, `page_number`, `dropped`, `frames` in the operator metadata map (renamable) | same key set as the other module receivers |
| Diagnostics | Python `pcie_rx_player.py` dump | `PcieDataChannelInterfaceV1::diagnostics(PcieChannelDiagnostics&)` reads back `GLB_SCRATCH`, `PACK_CTRL`, `DP_PKT_*`, `PCIE_DP_*`, the MSI-X entry and PBA; `PcieReceiverOp` logs it on a stall | one call to compare against a legacy run |

### VIP_MASK vs VP_MASK

They are the same register. The HSB host-interface packet block has a per-virtual-port
enable word at `+0x0C`: on Ethernet the SDK names it `DP_VP_MASK` at `hif_address + 0x0C`
(`0x0200030C` for data plane 0), on PCIe the bring-up register map names it
`DP_PKT_VIP_MASK` at `DP_PKT_BASE + 0x0C` (`0x3030C`). Bit `n` starts / stops streaming of
virtual port `n`, and the word is shared by every sensor on the plane. On Ethernet the SDK's
`DataChannel` read-modify-writes it when a channel is configured or unconfigured; on this
module the PCIe data channel does the RMW (`PcieDataChannelV1::arm()` / `disarm()` and the
edges inside `attach_receiver()` / `detach_receiver()`, via `or_uint32` / `and_uint32`), and
`DataChannel`'s RoCE configure -- which would program the same block's `VP` words as a RoCE
descriptor -- is disabled (`construct_roce_data_channel` returns false; section 5). So there
is exactly one writer of this word per sensor bit on either transport, and on both it is the
transport's data-channel service.

## 4. Driver delta vs `hsb_pcie/hololink_pcie_bringup/driver`

The driver now lives in this tree at `src/ehsb/kmod/hololink_pcie_drv.c` (uapi
`src/ehsb/kmod/include/hololink_pcie_uapi.h`; the installed userspace copy
`hololink_module/host/pcie/include/hololink/module/pcie/hololink_pcie_uapi.h` and the legacy
operator's copy are checked byte-identical at configure time), built by the top-level CMake
as target `hololink_pcie_drv` into `<build>/ehsb/kmod/`. `hsb_pcie/driver/` keeps a copy for
the emulator daemon build. It is a superset of the reference driver. **Existing ioctl numbers, structs, module parameters and
semantics are unchanged:** `HL_PCIE_IOC_READ` 2, `HL_PCIE_IOC_WRITE` 3,
`HL_PCIE_IOC_DMA_ALLOC` 4, `HL_PCIE_IOC_DMA_FREE` 5, `HL_PCIE_IOC_MSIX_INFO` 9 keep their
numbers and argument structs; `keep_pci_enabled`, `skip_identity_check`,
`require_identity_magic` and `expected_hsb_version` (default `HL_PCIE_REG_MAP_VERSION`
0x2606, 0 accepts any live version) exist in both drivers with the same meaning. Additions:

| Addition | Where | Behaviour |
|---|---|---|
| `HL_PCIE_IOC_DMABUF_IMPORT` (10, `struct hl_pcie_dmabuf`) / `HL_PCIE_IOC_DMABUF_RELEASE` (11) | `hl_pcie_ioctl_dmabuf_import` / `hl_pcie_dmabuf_check` | attaches a dma-buf fd to the PCI device and returns `iova`/`size`/`handle`. `hl_pcie_dmabuf_check` walks the sg list and requires every segment to start where the previous ended (**contiguity by adjacency**, not `nents == 1`: a VRAM export is chunked into hundreds of entries while describing one range) -> `-EOPNOTSUPP`; 128-byte alignment (`HL_PCIE_DMABUF_ALIGNMENT`) -> `-EINVAL`; any address above 4 GiB when `dma_address_bits <= 32` -> `-ERANGE`; straddling a 4 GiB boundary -> `-ERANGE`. Imports are owned by the fd and released on close |
| `HL_PCIE_IOC_MSIX_WAIT` (12, `struct hl_pcie_msix_wait`) | `hl_pcie_ioctl_msix_wait`; per-vector `wait_queue_head_t` in `struct hl_pcie_vector`, `wake_up_interruptible` from `hl_pcie_irq_handler` | blocks until the vector's `count != last_count`; `timeout_ms = 0` polls, `HL_PCIE_MSIX_WAIT_FOREVER` blocks indefinitely; returns `-ETIMEDOUT` / `-ERESTARTSYS`; `irq_count` is reported even on timeout so the caller can resynchronize. **Side effect since the kernel-owned table:** waiting enables the vector for the calling fd as if by `MSIX_CTRL`, so a caller that only waits (the reference bring-up tree's Python player, which still writes the table itself) keeps receiving completions. `HL_PCIE_IOC_MSIX_INFO` with `reset_count != 0` does the same, for callers that poll `irq_count` instead of waiting |
| Kernel-owned MSI-X table | `hl_pcie_setup_irqs`: `irq_set_status_flags(irq, IRQ_DISABLE_UNLAZY)` + `request_irq(..., IRQF_NO_AUTOEN, ...)` (fallback `disable_irq` after `request_irq` on kernels before 5.13); `hl_pcie_vector_enable` / `hl_pcie_vector_disable` under `hdev->lock`; `enable_refs` in `struct hl_pcie_vector`, `enabled_vectors` bitmap in `struct hl_pcie_file_ctx` | every vector starts disabled in both the kernel's shadow and the hardware; a vector is enabled while at least one fd holds it (an fd holds it at most once). The 0 -> 1 transition first rewrites the entry's `MESSAGE_ADDR_LO/HI` / `DATA` from `get_cached_msi_msg()` (the kernel's own values, so nothing disagrees) and then `enable_irq()`s, which performs the deferred startup / unmask; the 1 -> 0 transition `disable_irq()`s, which masks eagerly and waits for a running handler. `hl_pcie_mask_all_fpga_msix` still runs at probe (before MSI-X is enabled on the function) and at remove (after it is torn down), when the kernel holds no table state |
| `HL_PCIE_IOC_MSIX_CTRL` (14, `_IOWR('H', 14, struct hl_pcie_msix_ctrl)`, 32 bytes, `BUILD_BUG_ON`) | `hl_pcie_ioctl_msix_ctrl` | `vector`, `enable` (0/1), `flags` and `reserved[4]` must be 0; out `enabled_refs` = fds holding the vector after the call. `-EINVAL` for a vector beyond `num_vectors`. Idempotent per fd |
| `quiesce_on_release` (bool, default 1) | `hl_pcie_quiesce` from `hl_pcie_release` when the closing context owns DMA memory (`hl_pcie_ctx_owns_dma`) | writes `DP_PKT_VIP_MASK = 0` and flush-reads it, **before** unmapping/freeing. The driver's only unsolicited BAR0 write; set 0 to isolate |
| vector release on close | `hl_pcie_release_vectors` from `hl_pcie_release`, right after the quiesce and before any memory is freed; unconditional (it only touches IRQ state this fd created) | drops every vector hold of the closing fd (`disable_irq` when it was the last holder, which waits for a handler in flight), so a receiver that dies leaves its vector masked and no wakeup can reach reclaimed memory |
| bootp publisher | `hl_pcie_bootp_work_fn` on one `DECLARE_DELAYED_WORK`, `hl_pcie_bootp_send` (`#if IS_ENABLED(CONFIG_INET)`), payload `include/hololink_pcie_bootp.h`; scheduled from probe, `cancel_delayed_work_sync` in remove (before any lock) and exit; `bootp_announce` (bool, default 1) | announces every bound, UUID-valid endpoint once per second on `lo` with `netif_rx()` -- `<packed BDF>:12268 -> 127.255.255.255:12267` (the loopback subnet broadcast, delivered to every `INADDR_ANY` socket on the port instead of the single sharer a unicast to `127.0.0.1` would pick; the IP source is the BDF packed as in `ciaddr`, so even raw listeners see one peer per endpoint), the local table's `broadcast 127.255.255.255 dev lo` route (`RTCF_BROADCAST|RTCF_LOCAL`) attached via `ip_route_output_key` + `skb_dst_set` so the martian filter passes with no sysctl; BDF in `ciaddr`, BDF-derived MAC in `chaddr`, HSB vendor extension v2 with the board UUID and `HSB_VER`. Section 2 |
| `dma_address_bits` (uint, default 64) | `dma_set_mask_and_coherent(DMA_BIT_MASK(dma_address_bits))` at probe; gates the `-ERANGE` check above | 64 allows VRAM above 4 GiB; 32 for bitfiles that ignore `BUF_BASE_HI` (FIXED_MSB) and truncate |
| `HL_PCIE_IOC_IDENTITY` (13, `_IOR('H', 13, struct hl_pcie_identity_info)`) | `hl_pcie_ioctl_identity` | see below |
| six read-only sysfs attributes | `hl_pcie_dev_attrs[]` via `hl_pcie_class->dev_groups` (`ATTRIBUTE_GROUPS(hl_pcie_dev)`) | see below |
| `HL_PCIE_ID_F_*` flag macros | moved from `hololink_pcie_drv.c` to the uapi header, values unchanged (`VERSION_VALID` 0x1, `EXPECTED_VERSION_MATCH` 0x2, `CHECK_SKIPPED` 0x4, `MAGIC_VALID` 0x8, `UUID_VALID` 0x10) | shared with userspace |
| UUID refresh | `hl_pcie_refresh_uuid` from `hl_pcie_open`, from the IDENTITY ioctl and from `uuid_show` | re-reads only `HL_PCIE_REG_HSB_UUID0`..`HL_PCIE_REG_HSB_UUID3`; version/magic stay as taken at probe (they gate binding) |

**`struct hl_pcie_identity_info`** (exactly 64 bytes, `BUILD_BUG_ON` in `hl_pcie_init` and
`static_assert`s in `pcie_endpoint.cpp` / `pcie_device_access.cpp`; the size is part of the
ioctl number, so future fields go into `reserved[]`, never a resize):

| Offset | Field | Content |
|---|---|---|
| 0 | `__u16 domain` | `pci_domain_nr` |
| 2 | `__u8 bus` | bus number |
| 3 | `__u8 devfn` | `PCI_DEVFN(slot, func)` |
| 4 | `__u16 requester_id` | `(bus << 8) \| devfn`, what `PCIE_DP_REQUESTER_ID` takes |
| 6 | `__u16 minor` | `N` of this `/dev/hololink_pcieN` |
| 8 | `__u32 hsb_version` | `HSB_VER` as read at probe |
| 12 | `__u32 hsb_date` | `HSB_DATE` as read at probe |
| 16 | `__u32 hsb_status` | `HSB_STAT` as read at probe |
| 20 | `__u32 flags` | `HL_PCIE_ID_F_*` |
| 24 | `__u8 uuid[16]` | `put_unaligned_be64(uuid_high)` then `put_unaligned_be64(uuid_low)`, i.e. `HSB_UUID3` first, each register big-endian -- the byte order of section 2 |
| 40 | `__u32 num_msix_vectors` | vectors allocated |
| 44 | `__u32 dma_address_bits` | module parameter value |
| 48 | `__u64 bar0_len` | BAR0 length |
| 56 | `__u32 reserved[2]` | zero |

**sysfs attributes** on `/sys/class/hololink_pcie/hololink_pcieN/`, readable without opening
the node (same sources and formats as the ioctl): `bdf` (`%04x:%02x:%02x.%d`), `uuid`
(`%pUb`: canonical lowercase, refreshed from BAR0 on read), `hsb_version` (`0x%08x`),
`hsb_date` (`0x%08x`), `requester_id` (`0x%04x`), `msix_vectors` (`%d`).

**udev rule.** `device_create()` is called with no devnode callback, so the node lands as
`0600 root:root`. The durable fix (also in `hsb_pcie/docs/testing.md`):

```
KERNEL=="hololink_pcie[0-9]*", MODE="0660", GROUP="hololink"
```

installed as `/etc/udev/rules.d/99-hololink-pcie.rules` after `groupadd -f hololink` and
`usermod -aG hololink $USER`, then `udevadm control --reload && udevadm trigger
--subsystem-match=hololink_pcie`. The sysfs attributes are world-readable regardless, so
enumeration keyed on `bdf`/`uuid` works before the rule is in place; opening the node does
not. The optional `by-bdf` / `by-uuid` alias rules of section 2 go in the same file.

## 5. Core / module_core changes and framework notes

### `hololink::Hololink` virtuals (`src/hololink/core/hololink.hpp`)

`start()`, `stop()`, `reset()` are now `virtual`; `started_` moved into a `protected:`
block; `post_reset_configuration()` is documented as the hook a transport subclass calls
once the device is reachable again. `send_control` / `receive_control` were already
virtual. No signature changed; existing callers are unaffected.

### `HololinkV1::create_backing()` (`hololink_module/module/core/hololink_default.{hpp,cpp}`)

`HololinkV1::configure()` used to construct its private `LegacyHololinkAccess backing_`
inline from `peer_ip` / `control_port`. It now calls a new `protected: virtual
std::shared_ptr<LegacyHololinkAccess> create_backing(const EnumerationMetadata&)` whose
default body is exactly the old inline construction. `PcieHololinkV1 : HololinkV1`
(`pcie_hololink_v1.{hpp,cpp}`) overrides only this: it resolves the `/dev/hololink_pcieN`
minor from `peer_ip` -- the BDF -- through sysfs (`resolve_minor` -> `pcie::minor_for_bdf`
from the header-only `pcie/pcie_sysfs.hpp`; a `/dev/hololink_pcieN` or `/dev/hololink/by-bdf`
spelling is accepted too), reads `serial_number` and `pcie_backend` (`device` ->
`PcieEndpoint`, `loopback` -> `LoopbackPcieEndpoint` with the minor from `pcie_minor` and the
`peer_ip` BDF as its identity), builds a default `AddressPolicy` (software sequencer only),
and returns a `PcieHololink`. Because
`PcieHololink` IS-A `LegacyHololinkAccess`, `HololinkV1::start/stop/reset/configure_hsb/
ptp_synchronize/write_uint32/read_uint32/...` delegate to it unchanged and the virtual
overrides do the PCIe work. The frozen V1 interfaces (`HololinkInterfaceV1` and siblings)
are untouched: the extension point is on module_core's implementation class.

`HsbPciePublisher` (`hsb_pcie_publisher.hpp`, `module_name()` = `"hsb_pcie"`) is an
`HsbIpPublisher` that publishes `PcieHololinkV1` under the same ids the base uses for
`HololinkV1` (`construct_hololink`), publishes `PcieChannelConfigurationV1`, publishes the
PCIe data channel (next section), and gates enumeration as described in section 2.
`module_entry.cpp` is hsb_lite's entry with this publisher.

### Services the module publishes; the `hololink::hsb_pcie::headers` target

| Service | type_id(s) | instance_id | Where |
|---|---|---|---|
| `EnumerationInterfaceV1`, `FrameMetadataInterfaceV1`, `ChannelConfigurationInterfaceV1` (`PcieChannelConfigurationV1`) | canonical | `""` | `HsbIpPublisher::setup` / `publish_channel_configuration` |
| `HololinkInterfaceV1` (`PcieHololinkV1`, IS-A `HololinkV1`) | `hololink.v1`, `hololink.module_core.v1` | `serial=<n>` | `construct_hololink` override |
| `DataChannelInterfaceV1` anchor (`module_core::DataChannelV1`) | `data_channel.v1`, `data_channel.module_core.v1` | `serial=<n>;data_channel=<k>` | inherited `construct_data_channel` |
| **`PcieDataChannelInterfaceV1`** (`PcieDataChannelV1`) | `pcie_data_channel.v1`, `pcie_data_channel.hsb_pcie.v1` | `serial=<n>;data_channel=<k>` | `construct_board_services` |
| `I2cInterfaceV1`, `SequencerInterfaceV1`, `PtpPpsOutputInterfaceV1`, null `VsyncInterfaceV1` | canonical | canonical | inherited |
| `RoceDataChannelInterfaceV1`, `LinuxDataChannelInterfaceV1`, `CoeDataChannelInterfaceV1`, `OscillatorInterfaceV1` | -- | -- | **declined** (warning, `false`) |

`PcieDataChannelInterfaceV1` (`release_version = UNRELEASED`, so not yet in
`hololink_module/INTERFACES.md`, which lists released interfaces only) is declared in the
module-owned public header
`hololink_module/module/hsb_pcie/include/hololink/module/hsb_pcie/pcie_data_channel.hpp`,
together with `PcieReceiverAttachmentV1` (what a receiver hands the channel: ring geometry,
requester id and vector; no MSI-X message, the table is the kernel's) and
`PcieChannelDiagnostics` (whose MSI-X words are read-only diagnostics). The header ships through the INTERFACE target
`hololink::hsb_pcie::headers` -- the pattern of `hololink::hsb_lite::headers`: an include
directory plus `hololink::module_headers`, no module `.so`, installed with the framework
headers and exported as `hsb_pcie_headers`. `hololink::module::pcie_receiver` links it
`PUBLIC` (the receiver implements the attachment), `hololink::operators` links it `PUBLIC`
(the operator header names the channel type), the two module examples and
`hololink_module_hsb_pcie_test` link it directly. `hololink_module/CMakeLists.txt` processes
`module/hsb_pcie` before `host/pcie` so the target exists when the receiver library is
declared.

Namespace: `hololink::module::hsb_pcie`, the module's own, so directory, namespace and module
name coincide as they do for hsb_lite. The host receive path keeps `hololink::module::pcie`
for the transport-generic host pieces (register map, sysfs, receiver, geometry);
`PcieReceiver` names the attachment interface fully qualified.

Anchor arrangement: the transport-agnostic `DataChannelInterfaceV1` for a PCIe channel is
the module_core `DataChannelV1` that `construct_data_channel` publishes (the one
`HololinkV1::configure` also builds as `default_data_channel()` is the same class, held on the
Hololink, not published). `PcieDataChannelV1` is has-a with that anchor and is published only
under its own ids -- exactly how `RoceDataChannelV1` sits next to, not instead of, the anchor.
Publishing the PCIe impl under `data_channel.v1` too would make it is-a
`DataChannelInterfaceV1`, which the anchor design deliberately avoids, and would race
`construct_data_channel` for the key. `PcieDataChannelV1::configure(metadata)` drives the
anchor's `ensure_configured`, resolves the per-board `HololinkV1` and registers with it (so
`device_lost()` invalidates the channel), then reads `sensor`, `vp_mask` and `pcie_vector` --
the values `PcieChannelConfigurationV1::use_sensor` stamps.

### The `NO_RECEIVER_CONSTRUCTS` break

`add_hololink_module()` (`hololink_module/cmake/HololinkModule.cmake`) appends
`HOLOLINK_MODULE_RECEIVER_CONSTRUCT_SOURCES` (`module/core/roce_receiver_construct.cpp` and
`linux_receiver_construct.cpp`, set in `module/core/CMakeLists.txt`; the `_2510` variants
are added explicitly by the 2510 modules, not injected) to every module's sources unless `NO_RECEIVER_CONSTRUCTS` is
passed. Those TUs define `HsbIpPublisher::construct_roce_receiver` /
`construct_linux_receiver`, which the header declares `virtual` and calls from
`construct_service()` but does not define inline (to keep the header free of ibverbs and
`<cuda.h>`). Any `HsbIpPublisher` subclass built with `NO_RECEIVER_CONSTRUCTS` therefore has
undefined vtable entries and **fails at `dlopen`**. `NO_RECEIVER_CONSTRUCTS` is only usable
for a minimal, non-`HsbIpPublisher` module (the test stubs in `tests/CMakeLists.txt`). The
`hsb_pcie` module keeps the injected TUs: `RoceReceiverV1` / `LinuxReceiverV1` remain
constructible in principle, but the data-channel services they would pair with are declined
(next paragraph), so an Ethernet-style pipeline fails at the data-channel lookup, not at
load.

### `NO_COMPAT_SUFFIX`, no `compat_id`

`Adapter::load_module_for(uuid, compat_id)` (`hololink_module/host/src/adapter.cpp`) tries
`hololink_<uuid>_<compat>.so` when the metadata carries `compat_id`, falling back to the
bare `hololink_<uuid>.so`. The PCIe enumerator stamps no `compat_id`, so the module is built
with `NO_COMPAT_SUFFIX` and is exactly `hololink_58c1419b-1e7a-4120-95fd-4efc156187c1.so`.
Module directories come from `Adapter::set_module_directory()`, the `HOLOLINK_MODULE_DIR`
environment variable, or `<prefix>/lib/hololink/modules` with `<prefix>` derived from the
framework library's location (`dladdr`) and from `/proc/self/exe`
(`resolved_module_directories()`).

### Ethernet data channels disabled on this module

`HsbPciePublisher::construct_roce_data_channel`, `construct_linux_data_channel`,
`construct_coe_data_channel` and `construct_oscillator` return `false` after a warning when
asked for their type id, so `RoceDataChannelV1` / `LinuxDataChannelV1` / `CoeDataChannelV1`
/ `OscillatorInterfaceV1` lookups fail fast. The address policy backs this up: the SIF/HIF
blocks such a configure would write are `Shadow`. The transport view of a channel on this
module is `PcieDataChannelInterfaceV1` (above); the per-channel `DataChannelV1` anchor
(`construct_data_channel`, and the `default_data_channel()` `HololinkV1::configure` builds
because the metadata carries `data_plane`) is transport-agnostic and programs nothing.

### The daemon-era path is gone

Earlier on this branch a second PCIe path existed: an `hsb_pcie` emulator daemon fronting the
endpoint over UDP loopback, a `Hololink::PCIe` / `pcie_transaction` relay block in the core
(`0x03000400`), a `src/hololink/operators/dma_receiver/` receiver, and the legacy
`pcie_datagen_smoke` / `pcie_imx258_player` examples. All of it has been removed; the core is
back to `internal` apart from the three virtuals in section 5. The module path documented here
is the only PCIe path. The reference bring-up tree (`hsb_pcie/hololink_pcie_bringup`) remains
the external point of comparison for register sequences.

## 6. Hazards and bring-up checklist

### Hazards

- **SYS_RST is forwarded, and on current bitfiles it kills the sensor receive path until
  the FPGA is reconfigured.** `Hololink::reset()` -> `trigger_reset()` writes `0x04 <- 0x8`
  and the FPGA really resets under the driver: in-flight DMA stops, the MSI-X table
  entries, `PCIE_DP` words and I2C timing programmed by software must be assumed lost, and
  BAR0 reads return 0 / all-ones until the design is back. The data generator, the DMA
  engine and MSI-X all recover; the CSI receive block does not (`PACK_SIF_TVLD_STAT` stays
  0 with the sensor emitting), presumably because its initialization runs only at bitfile
  load. Observed on an AGX Orin, bitfile `0x5c694fdc`, where one resetting run left the
  reference player capturing nothing too until the board was reloaded. So on this transport
  `reset()` means "reload the design afterwards"; the examples default to not calling it,
  `PcieHololink::reset()` warns whenever it is called, and no reference PCIe flow writes the
  bit. `PcieHololink::reset()` handles
  the control side (live re-identify, I2C timing, `SW_SEN_RST` release,
  `post_reset_configuration()`), but a
  running receiver must be detached (`PcieDataChannelInterfaceV1::detach_receiver()`) and
  closed before, and re-`start()`ed and re-attached after. The kernel does not know the
  table was wiped; the receiver's re-`start()` covers it because the driver rewrites the
  entry's address/data from its cached message on the first enable of a vector nobody holds.
  The driver's cached `hsb_version`/`flags` are not affected. Watch `dmesg` for
  `arm-smmu`, AER and DPC events around a reset. <!-- codespell:ignore aer -->
- **VP / PCIE_DP overlap at `0x1000 + 0x40*n`.** On Ethernet the SDK's `DataChannel`
  programs this block as a RoCE descriptor (QP, RKEY, host MAC/IP/UDP); on this FPGA the
  same words are `PCIE_DP_BUF_BASE_HI`, `REQUESTER_ID`, `BUF_INC`, `BUF_START`, `BUF_LEN`,
  `METADATA_CTRL`, `BUF_PTR_RANGE`. The block is `ForwardIdentity`, so an Ethernet
  `DataChannel` configure would program a live DMA destination with garbage. Only
  `PcieDataChannelV1` may write there (the Ethernet channels are declined, section 5); the
  `DataChannelV1` anchor programs nothing.
- **`DP_PKT_VIP_MASK` is one word for both sensors, now read-modify-written.**
  `PcieDataChannelV1::arm()` / `disarm()`, the quiesce inside `attach_receiver()` and
  `detach_receiver()` all
  touch only their sensor's bit through `or_uint32` / `and_uint32`. The module serializes the
  read and the write, but nothing else may write the whole word: the driver's
  `quiesce_on_release` still writes `VIP_MASK = 0` when the closing fd owns DMA memory (it
  stops *both* sensors -- intended, that process is going away), and an Ethernet
  `DataChannel` configure would as well (disabled on this module, section 5).
- **The MSI-X table is the kernel's.** Nothing in userspace writes `0x8000-0x83FF`: the
  receiver enables and disables its vector through `HL_PCIE_IOC_MSIX_CTRL`, the driver does
  it with `enable_irq()` / `disable_irq()`, and the channel only reads the entry back for
  `diagnostics()`. Two consequences to keep in mind. A receiver that still writes the table
  itself (the reference bring-up tree's Python player) stays alive only through the
  implicit enable in `MSIX_WAIT` and in `MSIX_INFO` with `reset_count` -- found the hard way on
  an AGX Orin, where the reference player timed out under this driver until that was added; a
  new receiver on a driver without ioctl 14 fails `start()` rather than waiting on a vector
  that can never fire (`PcieDeviceAccess::msix_ctrl` maps `ENOTTY` to `UNSUPPORTED`). And a
  raw `write_uint32` to a `VECTOR_CONTROL` word (from a tool, or `hololink_pcie_identity
  --poke`) desynchronizes the kernel's shadow until the next enable/disable of that vector.
- **The frame-number reference can go stale.** `find_newest_frame()` rejects a decoded
  block whose 16-bit frame number is not ahead of the last delivered one (distance 0 or
  >= 0x8000), the reference design's rule. If the endpoint's counter restarts (sensor or
  packetizer reset, design reset) or the first pick came from a block that was not this
  run's, every subsequent frame looks "older" and nothing is delivered although completions
  keep arriving. The monitor therefore resynchronizes after `RESYNC_AFTER_MISSES` (8)
  consecutive completions in which at least one decoded, unseen block was rejected only by
  that rule: it drops the reference (keeping the delivered `(slot, frame)` keys) and takes
  the newest block present; the extended frame number restarts from the wire value. The
  first three misses and every 100th log a per-slot dump -- layout, `valid_bytes`, frame
  number, flags, the rejecting rule and the block's first 0x30 bytes -- so a wedged run
  explains itself (`PcieReceiver::describe_slots`).
- **`DP_PKT_LEN` is still global.** The data-TLP length at `DP_PKT_BASE + DP_PKT_LEN` is
  one word for the plane; each channel's `attach_receiver()` writes its own `chunk_pages`
  there, so two sensors must use the same `--chunk-pages`. The channel reads the word back
  first and logs a warning when it retargets a non-zero value another channel left there
  (the second attach silently changes the first's TLP size otherwise).
- **PTP and VSYNC are shadowed on the module path.** `ptp_synchronize()` returns immediately
  because `0x188` reads `0xF`. The endpoint's time base is not the SDK's Ethernet PTP words
  (whose addresses collide with the `bar_ptp` block) but the kernel driver's opt-in PTP
  support: `ptp_clock=1` registers `/dev/ptpN` for phc2sys and `bootp_ptp=1` snapshots the
  clock into every announcement (`src/ehsb/kmod/README.md`, "PTP: clock and time sync").
  Frame timestamps decoded from the metadata block (`timestamp_s/ns`, `metadata_s/ns`) come
  from that same FPGA timer, so once phc2sys has converged they are directly comparable with
  `received_s/ns`, which are host `CLOCK_REALTIME`. Writes to `0x70000000+` succeed silently
  without effect.
- **Two fds on one node.** `PcieEndpoint` holds one fd (plus a `resource0` mmap fd when
  permitted) for the control plane; `PcieDeviceAccess` holds another for DMA/MSI-X; the
  enumerator and the identity tool open the node briefly. `hl_pcie_open` has no
  exclusivity. `quiesce_on_release` fires only when the closing context owns DMA memory, so
  closing the control fd does not disarm the endpoint and closing the receiver fd does.
- **Permissions.** Without the udev rule the node is `0600 root:root`, so enumeration
  works for any user (sysfs identity) while every `open()` of the node fails with EACCES
  -- seen on an AGX Orin as "could not open PCIe endpoint" right after a successful
  listing. All three open sites (`PcieEndpoint`, `PcieDeviceAccess`, the enumerator)
  now diagnose the errno through `pcie::device_node_open_error()` and name the udev rule
  (`src/ehsb/kmod/README.md`, "Device node permissions"). BAR0 mmap of
  `/sys/bus/pci/devices/<bdf>/resource0` needs write access to that sysfs file (normally
  root); `PcieEndpoint::map_bar0()` falls back to the 32-bit ioctls otherwise (correct,
  slower).
- **Shared IOMMU group and `dma_address_bits`** (from `hsb_pcie/docs/testing.md`): on IGX
  the endpoint shares a translation domain with the NIC and the GPU; a bitfile that ignores
  `BUF_BASE_HI` with `dma_address_bits=64` truncates VRAM addresses and writes into other
  devices' memory. `cat /sys/module/hololink_pcie_drv/parameters/dma_address_bits` must
  match the bitfile before every DMA run; `sudo dmesg -C` first, stop on any `arm-smmu ...
  Unhandled context fault`.

### Hardware checklist (bound endpoint, extended driver loaded, daemon stopped)

1. Identity without opening the node, then enumerate:
   `cat /sys/class/hololink_pcie/hololink_pcie0/{bdf,uuid,hsb_version}` must show the
   read-only UUID `58c1419b-1e7a-4120-95fd-4efc156187c1`; `hololink_pcie_identity --bdf
   <bdf>` shows the same; `sudo tcpdump -i lo -n udp port 12267` shows one 300-byte datagram
   per endpoint per second; `adapter_enumerator -d` prints the record with `peer_ip` = the
   BDF and `interface = lo` (unbind/rebind the endpoint and it prints again within a second);
   `dmesg` clean.
2. IMX258 player over the module path (`module_pcie_imx258_player --hololink <bdf>
   [--module-dir <dir>]`; `<bdf>` as `lspci -D` / `cat /sys/class/hololink_pcie/*/bdf` print
   it, e.g. `0005:06:00.0`; omit `--hololink` when only one endpoint is bound; add
   `--module-dir <dir>` or set `HOLOLINK_MODULE_DIR` when the module `.so` is not under the
   framework's `<prefix>/lib/hololink/modules`; `--discovery-timeout S` only re-scans sysfs
   for an endpoint whose node is still appearing, a device that is not listed fails
   immediately): first `--skip-camera --frame-limit 4 --headless` (the channel is attached
   and armed but no I2C is issued; feed it with a sensor that is already streaming), then
   `--verify-registers --frame-limit 30 --headless` (model ID `0x0258`, zero register
   mismatches, `bytes_written == 2592640` = `640 + 2400*1080`, `dropped == 0`), then a
   windowed image check (drop `--headless`). `Imx258Cam` lives in
   `hololink_module/host/sensors/imx258/`.
4. `hololink->reset()`: the endpoint resets (SYS_RST forwarded), re-identify succeeds within
   `RESET_IDENTITY_TIMEOUT_S`, `configure_hsb` re-runs; `sudo dmesg | grep -iE
   "smmu|AER|DPC"` clean. <!-- codespell:ignore aer -->

### Software-only tests (ctest, no hardware)

| Test | Covers |
|---|---|
| `hololink_module_hsb_pcie_codec_test` | `EcbBridge` over `LoopbackPcieEndpoint`: `RD_DWORD 0x80` -> 0x2606, identity forward incl. `0x04`, I2C/SPI alias rebase, CLK_CNT hook -> 125, `RD_BLOCK` address echo, `WR_BLOCK` stops at reject keeping earlier writes, misaligned/unknown-cmd codes, PTP constant, shadowed host words, CTRL_EVT / APB RAM shadowed and `0x22C` arms the trigger; `ApbSequencer`: program execution, embedded alias reaches BAR0, POLL timeout per `APB_TIMEOUT`, deferred trigger semantics, FSM_ERROR on bad opcode |
| `hololink_module_hsb_pcie_hololink_test` | `PcieHololink` over loopback: `start()` opens no fds and runs `configure_hsb`, scalar/batched/block access, `and_uint32`/`or_uint32`, `reset()` re-identifies / rejects a mismatch / waits / times out; `start()` releases a held `SW_SEN_RST` keeping unrelated bits and `reset()` re-releases it after a modelled `SYS_RST` zeroes it (`DesignResetEndpoint`), failed `start()` cleanup, stop-restart, destruction order |
| `hololink_module_hsb_pcie_test` | loads the `.so` through `Adapter::enumerate` with `peer_ip=0000:00:00.0` (the BDF), `mac_id=00:00:00:00:00:00`, `pcie_minor=0`, `pcie_backend=loopback`: `module_name`, `use_sensor(1)` -> `vp_address 0x1040` / `pcie_vector 1`, sensor 2 rejected, `HololinkInterfaceV1` `start()` + `read_uint32({0x80}) == 0x2606`, `I2cInterfaceV1` fetch, RoCE data channel declined, the removed APB engine-selection key tolerated; declined: an IPv4 peer with no (or a malformed) `client_ip_address` even with `transport=pcie`, and a `/dev/hololink_pcie0` peer without `transport=pcie`; accepted: a BDF peer without `mac_id`, a `/dev/hololink_pcie0` peer with `transport=pcie`, and the kernel-bootp shape (`KernelBootpAnnouncementIsRestoredToTheBdf`: `peer_ip=127.0.0.1` + `client_ip_address=0.5.1.0` -> `peer_ip=0005:01:00.0`, `serial_number=pcie-0005:01:00.0`, `pcie_source_ip`, a working loopback Hololink behind it). **`PcieDataChannelInterfaceV1`** (`PcieDataChannelProgramsAndArmsPerSensor`): channels for sensor 0 and 1 resolve per key with `sensor` / `vector` / `vp_mask` from the metadata, a `FakeAttachment` (iova `0x2800e00000`, window `0x278F80`, 3 buffers, requester `0x0600`, MSI-X `{0xFEE00000, 0x4021}`) attached with `chunk_pages=2` reads back through `hololink->read_uint32` as the PCIE_DP words at `0x1000 + 0x40*sensor`, `DP_PKT_LEN=2`, the MSI-X entry `0x8000 + 0x10*sensor` (address, data, unmasked), `VIP_MASK` `0x1 -> 0x3` after both attach, `0x1` after sensor 1 detaches (its entry re-masked, sensor 0's untouched), `0x0` after both; wrong-vector and `chunk_pages=256` attachments rejected before any write, double attach rejected, `arm()` refused when detached, `disarm()` / `arm()` toggle one bit, `diagnostics()` returns the programmed values, re-attach after detach works |
| `hololink_module_pcie_enumerator_test` | `uuid_to_text` / `uuid_from_text` round trip, `scan_pcie_devices` against a fake sysfs tree (BDF from attribute and from the `device` link, broken entries skipped), `minor_for_bdf` / `device_node_for_bdf` over the `/sys/bus/pci/devices/<bdf>/hololink_pcie/` fast path and the class-walk fallback, `bdf_for_peer` for the BDF / node / `by-bdf` alias spellings, device helpers fail cleanly without a node |
| `hololink_module_pcie_bootp_test` | the kernel driver's announcement without a kernel: `hl_pcie_bootp_fill` from the mirrored header yields the 300-byte wire layout (every offset asserted), `dotted_quad_to_u32` / `bdf32_to_string` round-trip; the payload sent over loopback to the Adapter's `INADDR_ANY` listener as the driver sends it -- broadcast to `127.255.255.255` with `SO_BROADCAST` -- parses raw to `peer_ip = 127.0.0.1` (the test's userspace sender; the driver sources from the packed BDF, which userspace cannot), `client_ip_address = 0.5.1.0`, `mac_id = 02:50:00:05:01:00`, `transaction_id`, `fpga_uuid`, `hsb_ip_version 0x2606`, `enum_version 2`, `serial_number` hex, `interface = lo`; and through the real `hsb_pcie` `.so` `wait_for_channel("0005:01:00.0")` resolves with `peer_ip`, `pcie_bdf`, `pcie_source_ip`, `serial_number = pcie-<bdf>`, `transport` restored |
| `hololink_module_pcie_receiver_test` | geometry parity with the legacy path, `RxRingLayout::validate()` / `validate_buffer()` limits (buffer count, window, alignment, 4 GiB straddle), both metadata layouts, `PcieReceiver` against a fake `PcieDeviceAccessInterface` (no `HololinkInterfaceV1` anywhere): `start()` imports, reads `MSIX_INFO` for the sensor's vector with `reset_count` and baselines `wait_frame`; the `PcieReceiverAttachmentV1` values (`iova`, `window_bytes`, `buf_inc_pages`, `num_buffers`, `requester_id` from the identity, `vector`, `msix_message`, `disable_metadata`); `start()` enables the vector through `msix_ctrl` after the baseline and `close()` disables it once before releasing the mapping; a driver answering `UNSUPPORTED` (no `HL_PCIE_IOC_MSIX_CTRL`) fails `start()` with the mapping released; a vector held by another fd is still released; `start()` fails cleanly (mapping released) on a too-small or misaligned buffer, a vector beyond the allocation or MSI-X disabled; start/stop/close (a parked `get_next_frame()` is woken by `close()`; `stop()` keeps the mapping and `close()` waits for the monitor before releasing it) and accounting (`{0,1,2,0}` slots, batched completions -> dropped, frame-number wraparound); a frame-counter restart (frame 5 after 30000) is rejected as older for 8 completions and then delivered by the resynchronization, with the already-delivered block still withheld; `PcieReceiverOp` construct + `compose_graph`, missing `peer_ip` error, unbound-BDF error naming the BDF, literal node accepted (when operators are built) |
| `hololink_module_imx258_test` | IMX258 constants, frame-format table, register-table invariants (49 fps table, 60 fps differs only at `0x0307`), CSI-converter geometry (start byte 640, 2400-byte lines) |

```bash
ctest --test-dir <build> -R 'hololink_module_(hsb_pcie|pcie_enumerator|pcie_receiver|imx258)' --output-on-failure
```

## 7. Change summary and compatibility

This section is the high-level ledger of the branch: what was added, which pre-existing
interfaces were touched and how, which models of the legacy `hololink` did not carry over
to a PCIe control plane and what replaced them, what this means for
`hololink_module/INTERFACES.md`, and where forward/backward compatibility is preserved or
broken. Sections 1-6 hold the detail; this one only points at them.

### 7.1 Footprint

Everything under `hololink_module/module/hsb_pcie/`, `hololink_module/host/pcie/`,
`hololink_module/host/sensors/imx258/` (the PCIe enumeration helpers and the PCIe headers
live under `host/pcie/`; `hololink_module/host/src` and `host/include` carry no PCIe code), the
`examples/module_pcie_imx258_player` program, the `hololink_pcie_identity` tool, the new tests and this
document are **new files**. Pre-existing files were modified as follows; nothing else in the
tree changed.

| Pre-existing file | Change | Kind |
|---|---|---|
| `src/hololink/core/hololink.hpp` | `Hololink::start/stop/reset` made `virtual`; `started_` moved to `protected:`; doc comments | legacy core, source-compatible |
| `hololink_module/module/core/hololink_default.{hpp,cpp}` | `HololinkV1::create_backing()` protected virtual factory; `configure()` calls it | module_core implementation class |
| `hololink_module/host/include/hololink/module/adapter.hpp`, `host/src/adapter.cpp` | **unchanged from `internal`** (a `DiscoveryProvider` abstraction and `rescan()` existed on the branch for a while and were reverted; discovery is bootp-only) | host API, unchanged |
| `hololink_module/host/src/bootp.cpp` | `recvmsg` uses `MSG_DONTWAIT`; `EAGAIN` returns an empty record | bug fix (pre-existing hang) |
| `hololink_module/host/src/reactor_impl.cpp` | `POLLNVAL` fds are skipped instead of dispatched | bug fix (pre-existing) |
| `hololink_module/host/python/hololink_module_py.cpp` | unchanged from `internal` | Python, unchanged |
| `hololink_module/tools/adapter_enumerator.cpp` | `-d/--dedup` keyed on `peer_ip` | tool, additive |
| `hololink_module/CMakeLists.txt`, `host/CMakeLists.txt`, `host/operators/CMakeLists.txt`, `host/sensors/CMakeLists.txt`, `tools/CMakeLists.txt`, `examples/CMakeLists.txt`, `tests/CMakeLists.txt` | new subdirectories, sources and targets; `hololink::operators` additionally links `hololink::hsb_pcie::headers` and `hololink::module::pcie_receiver` (both core-free) | build |
| `hololink_module/README.md` | discovery entry (bootp-only; kernel driver announces PCIe) | docs |
| `hsb_pcie/driver/hololink_pcie_drv.c`, `hsb_pcie/driver/include/hololink_pcie_uapi.h` (now also `src/ehsb/kmod/` in this tree, built by the top-level CMake) | `HL_PCIE_IOC_IDENTITY` (ioctl 13), `HL_PCIE_IOC_MSIX_CTRL` (ioctl 14), six read-only sysfs attributes, UUID re-read on `open()`, kernel-owned MSI-X mask; ioctls 2-12 untouched | kernel driver, additive |

Not touched: the frozen V1 interface headers under `hololink_module/host/include/hololink/module/`
(every class listed in `INTERFACES.md`), `abi_check.h` and the `hololink_module_init` /
`hololink_module_get_abi_check` C handshake, and every other module under
`hololink_module/module/`. The daemon-era pieces (`Hololink::PCIe` relay, `dma_receiver`
operator, legacy `pcie_*` examples) that once lived on this branch are deleted, so the core
differs from `internal` only by the three virtuals.

### 7.2 Interfaces touched, by layer

| Layer | Interface | What changed | Compatibility |
|---|---|---|---|
| Frozen module interfaces (`INTERFACES.md` 2.7.0 set) | `HololinkInterfaceV1`, `DataChannelInterfaceV1`, `ChannelConfigurationInterfaceV1`, `EnumerationInterfaceV1`, ... | **nothing** -- the module implements them (`PcieHololinkV1`, module_core `DataChannelV1`, `PcieChannelConfigurationV1`, `HsbIpPublisher::update_metadata`) and declines the Ethernet transport views | unchanged |
| New module interface (UNRELEASED) | `PcieDataChannelInterfaceV1` (`pcie_data_channel.v1`), `PcieReceiverAttachmentV1`, `PcieMsixMessage`, `PcieChannelDiagnostics` in `hololink/module/hsb_pcie/pcie_data_channel.hpp` | new, published only by this module under its own ids; may still change until a release freezes it | additive |
| Module C ABI | `hololink_module_init`, `hololink_module_get_abi_check`, `hololink_module_abi_check_t` | nothing | unchanged |
| module_core implementation class (explicitly not frozen, `plans/hololink_module_interface_versions.md` "Not frozen, deliberately") | `HololinkV1` | `create_backing()` virtual factory; default body is the previous inline construction | source-compatible for other modules; each module statically links its own module_core, so no cross-`.so` effect |
| Legacy core | `hololink::Hololink` | three methods made virtual, one member made protected | source-compatible; **vtable layout of `Hololink` changes** (7.5) |
| Host framework API (not a frozen V1 interface, never crossed by a module `.so`) | `Adapter` | nothing: discovery stays bootp-only, the bootp listener is unchanged | unchanged |
| Kernel driver, new behaviour | bootp publisher (`hl_pcie_bootp_work_fn`, `include/hololink_pcie_bootp.h`) | one announcement per bound endpoint per second on `lo`, broadcast `<packed BDF>:12268 -> 127.255.255.255:12267` so every `INADDR_ANY` listener on the port receives it; `bootp_announce` parameter | additive; the host sees one more bootp source |
| Host framework API, new | `pcie_enumerator.hpp` (sysfs/ioctl helpers for the identity tool and examples), `pcie/pcie_sysfs.hpp` (+ `dotted_quad_to_u32`, `bdf32_to_string`), `pcie/pcie_regs.hpp`, `pcie/hololink_pcie_uapi.h`, `pcie/hololink_pcie_bootp.h` | new installed headers; none is a discovery path | additive |
| Host receive path, new | `hololink::module::pcie::PcieReceiver`, `PcieDeviceAccess`, `RxRingLayout`; `hololink::module::operators::PcieReceiverOp` | new; `PcieReceiverOp` mirrors `RoceReceiverOp`'s parameter shape | additive |
| Python | -- | unchanged | no PCIe-specific Python bindings yet |
| Kernel uapi | `hololink_pcie_uapi.h` (driver and the two hololink mirrors) | ioctl 13 (`IDENTITY`, 64-byte struct) and ioctl 14 (`MSIX_CTRL`, 32-byte struct) appended; flags | additive; ioctls 2-12 and their structs unchanged. `MSIX_WAIT` keeps its struct but gains a side effect (enables the vector for the fd) |
| Kernel IRQ setup | `hl_pcie_setup_irqs` | vectors requested disabled (`IRQF_NO_AUTOEN`, `IRQ_DISABLE_UNLAZY`); enable/disable via the kernel IRQ API, refcounted per fd | behaviour change: a vector fires only for an fd that enabled it or waited on it |
| Kernel sysfs | `/sys/class/hololink_pcie/hololink_pcieN/{bdf,uuid,hsb_version,hsb_date,requester_id,msix_vectors}` | new read-only attributes | additive |
| Enumeration metadata | bootp-parsed keys as for any board; `transport`, `pcie_*`, `pcie_source_ip` added by the module | the *meaning* of `peer_ip` is widened post-enrichment: an IPv4 dotted quad for Ethernet, a `DDDD:BB:DD.F` BDF for PCIe (section 2); the raw record is the packed BDF as a dotted quad (`0.5.1.0`) | new keys additive; the widened `peer_ip` is the one thing an existing consumer can observe |

### 7.3 Legacy `hololink` models that did not fit a PCIe control plane

Each row is an assumption baked into the Ethernet-era SDK or framework that the PCIe
endpoint violates, and what the module does instead. The first eight are inherent to the
transport; the rest are framework or hardware findings made during bring-up.

| # | Legacy model | Why it does not hold on PCIe | Replacement | Where |
|---|---|---|---|---|
| 1 | Discovery is a bootp broadcast the board sends; the `Adapter` owns one UDP listener | the endpoint sends nothing | the kernel driver sends it instead: one bootp announcement per bound endpoint per second, injected into `lo` with `netif_rx()` as a broadcast on the loopback subnet (`127.255.255.255:12267`, so every `INADDR_ANY` listener receives it, as with an Ethernet broadcast); the Adapter is unchanged. The BDF rides in `ciaddr` because the parser takes `peer_ip` from the IP source, and the module restores `peer_ip` before subscribers are matched | 2 |
| 2 | The channel key `peer_ip` is an IPv4 address and `mac_id` the hardware address | there is no IP; the stable name of the function is its PCI address | the announcement says so in its header (`htype 0x50`, `chaddr` = the BDF), and the `hsb_pcie` module's `update_metadata` (the legacy enumerator behind `hololink-enumerate` on its own) renders `peer_ip` = `mac_id` = the BDF text; `interface` = `lo`, char device resolved from the BDF through sysfs; `wait_for_channel(peer_ip)` is a string compare and works as-is | 2 "Why the BDF" |
| 3 | The control plane is a UDP socket pair (`send_control`/`receive_control`), plus an async-event socket and thread, opened by `Hololink::start()` | no carrier; a register access is a BAR0 MMIO or ioctl | `PcieHololink` overrides `send_control`/`receive_control` with the in-process `EcbBridge`; `start()` opens no sockets and only programs I2C timing before `configure_hsb()` | 1 |
| 4 | `Hololink::reset()` waits up to 30 s for the board's next bootp announcement (`find_channel`) | nothing is announced after `SYS_RST` | `reset()` override: `trigger_reset()`, poll live `HSB_VER`/UUID over BAR0 until they match the metadata, then `post_reset_configuration()` | 1 "Why start/stop/reset became virtual" |
| 5 | `ptp_synchronize()` polls PTP status at `0x188` until locked | the endpoint's PTP block (`bar_ptp`, BAR0 0x100..0x1FF) has a different layout at the same addresses and is owned by the kernel driver (opt-in `ptp_clock` / `bootp_ptp`) | `0x188` is a read-only constant `0xF` in the address policy; other PTP words are shadowed; time sync is phc2sys against the driver's `/dev/ptpN` | 1 "Address policy", kmod README |
| 6 | I2C and SPI live at HSB `0x03000200` / `0x03000000` | BAR0 is 1 MiB; the blocks sit at `0x40200` / `0x40000` | alias rebase, the one address translation kept; `CLK_CNT` writes rewritten to 125 | 1 "Why the I2C/SPI alias exists" |
| 7 | SDK I2C runs as a program on the FPGA's APB sequencer (`CTRL_EVT` + `APB_RAM`) | the FPGA does not execute sequencer programs written over PCIe (hardware finding, 2026-09-10) | in-process software `ApbSequencer` over shadowed `APB_RAM`, deferred trigger on `0x22C`; no engine switch exists | 1 "The software APB sequencer" |
| 8 | A data channel is configured by writing SIF/HIF (RoCE/Linux/CoE) blocks; the receiver is handed a `DataChannel` | the endpoint DMA engine is programmed through PCIE_DP / DP_PKT words instead (the interrupt side is generic PCI, row 16) | `PcieDataChannelInterfaceV1` (`PcieDataChannelV1`) programs the channel and arms/disarms its `VIP_MASK` bit via `or_uint32`/`and_uint32`; RoCE/Linux/CoE/oscillator constructs return `false`; `PcieReceiverOp` takes `enumeration_metadata` and resolves the PCIe channel itself | 5 "Services the module publishes", 3 |
| 9 | module_core `HololinkV1::configure()` hard-wires `LegacyHololinkAccess(peer_ip, control_port)` | the backing must be the PCIe subclass | `create_backing()` virtual factory; `PcieHololinkV1` overrides only that | 5 |
| 10 | `Hololink::start/stop/reset` are non-virtual; `started_` private | the subclass must replace them | made virtual / protected in core | 5 |
| 11 | A board's `fpga_uuid` is chosen to match a module; a provisioning step writes it | the endpoint's UUID registers are read-only (writes refused) | the module is named after the board's UUID `58c1419b-...`; `hololink_pcie_identity` is read-only | 2 "Board UUID and the identity tool" |
| 12 | `add_hololink_module(... NO_RECEIVER_CONSTRUCTS)` is how a module opts out of Ethernet receivers | any `HsbIpPublisher` subclass built that way has undefined vtable entries and fails at `dlopen` | keep the injected receiver TUs; decline at the data-channel lookup instead | 5 "The NO_RECEIVER_CONSTRUCTS break" |
| 13 | Module lookup appends `_<compat_id>` | the enumerator stamps no `compat_id` | `NO_COMPAT_SUFFIX` | 5 |
| 14 | `FrameMetadataInterfaceV1` decodes the HSB end-of-frame block | the endpoint deposits a different slot-metadata layout (two variants) | local `parse_rx_metadata` in the receiver; the service is not used on this path | 3 |
| 15 | Receiver slot accounting follows a write pointer assumed to restart at slot 0 when `BUF_PTR_RANGE` is programmed (reference design) | the endpoint's write pointer persists across runs, so a fresh receiver starts mid-ring | `PcieReceiver::find_newest_frame()` scans slot metadata, keyed on `(slot, frame_number)`; the reference receiver kept the pointer model | 3 "Changed" |
| 16 | Userspace programs and masks the MSI-X table (mask-all, then its own entry) | the table is generic PCI the kernel already owns: a mask-all from one ring silenced the sibling, and any userspace mask write is undone by the next kernel rewrite of the entry (affinity change) | no userspace table write at all: the receiver enables/disables its vector through `HL_PCIE_IOC_MSIX_CTRL`, the driver uses `enable_irq` / `disable_irq`, the channel programs only HSB blocks; `VIP_MASK` (== Ethernet `DP_VP_MASK`) is touched one bit at a time | 3 "MSI-X table ownership", 4 |
| 17 | `Hololink::PCIe` / `pcie_transaction` wrap a PCIe access in a relay ECB block through `PCIE_CTRL` | the module owns BAR0 directly; every access is already a plain `read_uint32`/`write_uint32` | not used on the module path; the API is left in core untouched | 5 |
| 18 | A board that enumerates is a board that answers | a bound-but-non-decoding BAR (lspci `Mem-`) reads all-ones | the driver announces only endpoints whose UUID is valid (`HL_PCIE_ID_F_UUID_VALID`), so a dead endpoint is simply absent from discovery; `hololink_pcie_identity` still reports it | 2 |
| 19 | Framework: reactor dispatches any fd with `revents`; `receive_bootp` blocks in `recvmsg` | a stale `POLLNVAL` fd and a spurious readiness wedge the single reactor thread (surfaced by stop/start of the bootp listener) | skip `POLLNVAL`; `MSG_DONTWAIT` | 7.1 |

### 7.4 Impact on `hololink_module/INTERFACES.md`

`INTERFACES.md` lists released, frozen interfaces only. **No entry in its 2.7.0 section
changes**: none of the 25 frozen classes was edited, and every one the module publishes is
served by an implementation of the frozen definition.

Additions the release that ships this module would have to record, under its own version
heading (both currently declare `release_version = UNRELEASED` and must be bumped to that
release's version string at freeze time):

- **`PcieDataChannelInterfaceV1`** -- `hololink/module/hsb_pcie/pcie_data_channel.hpp` (in
  `hololink_module/module/hsb_pcie/include/`) · `pcie_data_channel.v1`. PCIe transport view of
  a per-channel `DataChannelInterfaceV1`, keyed `"serial=<n>;data_channel=<k>"`. Programs the
  endpoint's PCIE_DP / DP_PKT_LEN words from a receiver attachment and arms or disarms the
  channel's `VIP_MASK` bit; the MSI-X table is not part of its surface. Sits beside the
  transport-agnostic anchor, as `RoceDataChannelInterfaceV1` does.
- **`PcieReceiverAttachmentV1`** -- same header · not a locator service. What a receiver hands
  `PcieDataChannelInterfaceV1::attach_receiver`: the DMA window (iova, window bytes, buffer
  count), requester id and MSI-X vector. Freezes together with the channel interface because
  it is a parameter type of it. It deliberately carries no MSI-X message: the receiver enables
  its own vector through the driver, so a frozen V1 never has to drop a dead accessor.

Not candidates for `INTERFACES.md`, by the file's own rule: `PcieHololinkV1` /
`PcieChannelConfigurationV1` / `PcieDataChannelV1` (implementations of frozen or new
interfaces), `PcieReceiver`, `PcieReceiverOp`, the enumerator, and `Imx258Cam`.

Description drift worth fixing when that section is written: the `EnumerationInterfaceV1`
entry says the Adapter calls `update_metadata` "after a bootp packet arrives (or after a
manual `enumerate()` call)"; it is now also called after a PCIe provider announcement. The
contract is unchanged, only the description is narrower than the behaviour.

Behaviour of frozen interfaces on this module that a caller written against the Ethernet
modules will notice, all within the frozen contracts (no method was added, removed or
re-signatured):

- `HololinkInterfaceV1::ptp_synchronize()` returns immediately (shadowed status).
- `HololinkInterfaceV1::reset()` really resets the FPGA (`SYS_RST` forwarded) and returns
  once BAR0 re-identifies; no bootp wait.
- `get_service<RoceDataChannelInterfaceV1 | LinuxDataChannelInterfaceV1 |
  CoeDataChannelInterfaceV1 | OscillatorInterfaceV1>` fails through the normal
  service-not-found path.
- `ChannelConfigurationInterfaceV1::use_sensor(n)` accepts `n < 2` and stamps `pcie_vector`.

### 7.5 Forward / backward compatibility

**Preserved.**

- *Previously built module `.so`s load on the new host.* The C handshake (`abi_check.h`,
  `hololink_module_init`, `hololink_module_get_abi_check`, the `sizeof`/`alignof` checks of
  `EnumerationMetadata` and `std::string`) is unchanged, and `Adapter` is a host-side class
  no module calls. Hsb-Lite, Leopard, DA326 and Tutorial modules are untouched.
- *The new `hsb_pcie` module `.so` loads on a 2.7.0 host* in principle: it needs no new host
  service, only the kernel driver's bootp announcement, which a 2.7.0 host's unchanged bootp
  listener parses identically; it would lack only the host-side `PcieReceiver` library and the
  `pcie_sysfs.hpp` helpers the module header-includes. There is no supported reason to do
  this.
- *Applications compiled against the frozen 2.7.0 headers* are unaffected; those headers did
  not change.
- *Driver ioctls 2-12 behave as before*; the reference bring-up tree's tooling still works
  against this driver.
- *Ethernet-only hosts.* Nothing changes for them: the Adapter has no PCIe code; without the
  kernel driver loaded no PCIe announcement is ever produced.
- *Driver uapi forward room*: `hl_pcie_identity_info` keeps `reserved[2]` and its size is part
  of the ioctl number; new fields go into the reserved words, never beyond 64 bytes. Sysfs
  attributes are additive by nature.

**Broken or changed -- read before upgrading.**

1. **`libhololink` core binary ABI.** Making `Hololink::start/stop/reset` virtual adds three
   vtable slots to `hololink::Hololink`. Source is compatible (no signature changed; callers
   recompile without edits), but any binary that was compiled against the previous header and
   subclasses `Hololink` (the pybind trampoline `PyHololink`, any out-of-tree subclass) or
   allocates it must be rebuilt against the new header. In this tree the Python extension and
   every consumer are rebuilt together; `ihsb` compiles the core in-tree, so for it this is a
   rebuild, not a code change. `started_` moving to `protected:` does not change the layout.
2. **`register_all` subscribers see a new record shape.** An application that registers for
   every announcement now also receives PCIe records whose `peer_ip` and `mac_id` are the
   BDF string. Code that parses `peer_ip` as an IPv4 address or `mac_id` as a MAC must first
   filter on `hardware_type` (`0x50` for PCIe, `1` for Ethernet) or, after enrichment, on
   `transport == "pcie"`. `register_ip(<ipv4>)` and `wait_for_channel(<ipv4>)` subscribers are
   unaffected because the BDF never string-matches an address.
3. **`Adapter` construction has new side effects**: one sysfs walk and one netlink socket
   per process, and a reactor alarm on hotplug bursts. Cost is negligible; the observable part
   is a `discovery:` log line and, in tools, a `pcie (...)` provider entry.
4. **`Adapter` private layout changed** (bootp fd/callback/mutex replaced by the provider
   vector). The host library is `STATIC`, so every consumer relinks anyway; there is no shared
   `Adapter` object to mismatch.
5. **Kernel driver `open()` now reads the four UUID registers** on every open, including
   opens by any consumer. This is a new side effect on an existing entry point; it
   is four MMIO reads and cannot fail, but it is the one place the "existing APIs unchanged"
   rule is bent rather than purely appended to. `HL_PCIE_IOC_IDENTITY` also re-reads the UUID,
   while `hsb_version/hsb_date/hsb_status` remain the probe-time values -- which is also why
   the bootp publisher gates on the UUID's validity.
   The second bend is `HL_PCIE_IOC_MSIX_WAIT` (and `HL_PCIE_IOC_MSIX_INFO` with
   `reset_count`): their structs and return values are unchanged,
   but waiting now enables the vector for the calling fd. This is what keeps a receiver that
   still writes the FPGA table itself (the reference bring-up tree's Python player) receiving
   completions once vectors start disabled in the kernel. Conversely the new `PcieReceiver`
   refuses to start on a driver without ioctl 14.
6. **One mirror of the driver uapi header** exists in `hololink`
   (`hololink_module/host/pcie/include/hololink/module/pcie/`) and must stay byte-identical to the
   canonical `src/ehsb/kmod/include/hololink_pcie_uapi.h`; likewise the bootp layout header.
   `src/ehsb/kmod/CMakeLists.txt` fails the configure when either drifts.
7. **`PcieDataChannelInterfaceV1` and `PcieReceiverAttachmentV1` are `UNRELEASED`.** Until a
   release freezes them their definitions may change in place; an application that names them
   opts into that. Everything else the module publishes is a frozen interface.
8. **`hololink::operators` gains two link dependencies** (`hololink::hsb_pcie::headers`,
   `hololink::module::pcie_receiver`). Both are core-free, so the "operators link no legacy
   core" property holds; a downstream build that consumed the exported target set must pick up
   the new exports (`hsb_pcie_headers`, the receiver library).
9. **Module `.so` name is the board's UUID.** A second PCIe bitfile reporting a different
    UUID needs its own module entry or a `hololink_<uuid>.so` symlink; two boards reporting
    the same UUID are told apart by `serial_number = pcie-<BDF>`.

**Who must rebuild after pulling this branch:** everything that links `libhololink` core or
the static host library -- i.e. the whole tree and `ihsb`. Nothing has to change in source
outside the two items in "Broken or changed" that an application might actually hit
(items 1 and 2).

**Not done, by ruling or scope:** Python bindings for the PCIe receiver, channel and
enumerator; `nvidia_peermem` (dmabuf is the only import path).
