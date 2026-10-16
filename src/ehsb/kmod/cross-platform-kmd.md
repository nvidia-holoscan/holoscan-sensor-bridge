# `hololink_pcie` kernel driver: cross-platform risks

Target matrix this document is written against:

| Platform | CPU | GPU | Kernel (Ubuntu) | Status |
|---|---|---|---|---|
| IGX Orin developer kit | aarch64 (Tegra234) | dGPU, RTX 6000 Ada | `7.0.0-1010-nvidia` | **the only platform the driver has run on** |
| Jetson (Orin, Thor) with L4T | aarch64 | iGPU | the L4T kernel of the release (5.15- and 6.x-based lines have shipped) | untested |
| IGX with L4T and iGPU only | aarch64 | iGPU | as above | untested |
| DGX | x86_64 or Grace (aarch64) | dGPU | Ubuntu stock or HWE 6.x; 7.x possible | untested |

Everything below is derived from `src/ehsb/kmod/hololink_pcie_drv.c` as of the kernel-owned
MSI-X and bootp-publisher changes and from its userspace consumer, the `hololink_module` PCIe
receive path (`hololink_module/host/pcie/`). Each section states what the driver does today, where that
assumption stops holding, how it would fail, and what to do about it. Severity is the
consequence when the risk materializes: **blocks** (no receive path on that platform),
**intermittent** (works sometimes), **degraded** (works, slower or noisier), **safety**
(silent memory corruption possible).

The short version: the kernel-API surface is small and already version-guarded, so a 6.x
or 7.x Ubuntu kernel is a low risk. The real risks are in what the driver assumes about the
platform around it: an IOMMU that coalesces mappings, a dma-buf exporter that can hand out
GPU memory to a peer, MSI-X with a generous vector budget, and a GPU whose memory the FPGA
can write into coherently. Those assumptions hold on the IGX dGPU and are unverified or
false elsewhere, and the integrated-GPU case is known to break before the driver is even
reached.

## 1. Kernel API surface and version drift

The driver uses no architecture-specific code (no `CONFIG_*`, `__aarch64__` or x86 guards
anywhere) and touches a small set of kernel APIs. Version guards present today:

| API | Introduced / changed | Guard in the driver | Risk |
|---|---|---|---|
| `class_create()` lost its `owner` argument | 6.4 | `#if LINUX_VERSION_CODE >= 6.4.0` | none |
| `<linux/unaligned.h>` replaced `<asm/unaligned.h>` | 6.12 | `#if >= 6.12.0` | none |
| `MODULE_IMPORT_NS("DMA_BUF")` takes a string | 6.13 | `#if >= 6.13.0` | none. On kernels before the `DMA_BUF` namespace existed (pre-5.16) the macro is a harmless modinfo tag |
| `IRQF_NO_AUTOEN` | 5.13 | `#ifdef`, fallback `request_irq` + `disable_irq` | none; the fallback leaves a brief unmasked window at probe |
| `IRQ_DISABLE_UNLAZY` via `irq_set_status_flags` | 4.x | none needed | none |
| `dma_buf_dynamic_attach(dmabuf, dev, importer_ops, priv)`, `dma_buf_pin/unpin` | 5.7-5.8 | none | fine on every target kernel; the driver holds `dma_resv_lock` around map/pin/unmap as the dynamic-importer contract requires |
| `dma_buf_map_attachment()` locked variant | `_unlocked` siblings added in 6.2; the locked one is what a dynamic importer must call | none | a future kernel could rename; compile break, not a silent one |
| `get_cached_msi_msg()` | long-standing, `EXPORT_SYMBOL_GPL` | none | requires `MODULE_LICENSE("GPL")`, which the driver declares |
| `pci_alloc_irq_vectors(PCI_IRQ_MSIX)`, `pci_irq_vector` | 4.8 | none | none |
| `dma_alloc_attrs` / `dma_mmap_attrs`, `dma_set_mask_and_coherent` | long-standing | none | none |
| `struct class::dev_groups`, `device_create` | long-standing | none | `device_create` with no `devnode` callback yields `0600 root:root`; udev rule required everywhere |

**Where 7.x can still break the build.** Anything above without a guard is stable API, but
7.x kernels have not been compiled against except the one IGX kernel. The cheap insurance is
a compile matrix: every kernel headers package the deployment set uses, built with `W=1` and
warnings treated as failures. `src/ehsb/kmod/CMakeLists.txt` resolves `KERNEL_DIR` from `uname -r`
only; a matrix needs `-DKERNEL_DIR=<headers>` per kernel.

**Compiler mismatch.** The IGX build already warns "the compiler differs from the one used
to build the kernel" and skips BTF generation for lack of `vmlinux`. Harmless today; a kernel
built with a different major GCC or with Rust/LLVM (Ubuntu ships some) may refuse to load a
module built with a mismatched toolchain when `CONFIG_MODVERSIONS` disagrees. DKMS (section 6)
removes this class of problem by building on the target.

## 2. Interrupts

**What the driver does.** `hl_pcie_setup_irqs` requires MSI-X only (`PCI_IRQ_MSIX`), refuses
to bind if the function advertises fewer than `HL_PCIE_REQUIRED_MSIX_VECTORS` (33: the
transmit engine uses vector `32 + sensor`), and asks for up to 64. Every vector is requested
disabled and enabled per fd through `HL_PCIE_IOC_MSIX_CTRL` or implicitly by
`HL_PCIE_IOC_MSIX_WAIT`. On the first enable of a vector the driver rewrites that table
entry's address/data from `get_cached_msi_msg()` (the kernel's own values) before
`enable_irq()`.

| Risk | Platforms | Failure mode | Severity | Mitigation |
|---|---|---|---|---|
| No MSI or INTx fallback | any host where MSI-X allocation fails | probe fails with "MSI-X allocation failed" | blocks | acceptable by design (the FPGA's completion model is per-vector), but log the platform's vector budget |
| 33-vector minimum on a shared MSI budget | Tegra root ports: the DesignWare host controller draws MSI vectors from a per-controller pool shared by every device behind that port | `pci_alloc_irq_vectors(33, 64)` returns `-ENOSPC` when the pool is partly used by a NIC or NVMe on the same controller; probe fails | blocks | **the receive path needs 2 vectors**: one per sensor, `vector == sensor`, for the two sensors the module exposes (`PcieChannelConfigurationV1::TOTAL_SENSORS`); `MSIX_INFO(0)` is only a query. The 33 exists for the transmit engine's `32 + sensor` convention. Make the minimum a module parameter defaulting to 2 and let the transmit path check `num_msix_vectors` at runtime |
| MSI-X through a DesignWare MSI catcher | Tegra (Orin) | works; all vectors target one doorbell, so message *address* is identical per vector and only *data* differs -- the entry-restore code handles this because it copies both fields verbatim | none | none, verify once |
| GIC ITS vs x86 APIC / interrupt remapping | Grace and Tegra vs x86 DGX | different `msi_msg` encodings; the driver never interprets them, only echoes the kernel's cache | none | none |
| Kernel-side entry rewrite on affinity change | any, with `irqbalance` | now consistent: the kernel owns the mask and the driver's restore writes the kernel's values | none (fixed by the kernel-owned change) | keep userspace out of `0x8000-0x83FF` |
| Legacy receiver still writes the table | any | works only through the implicit enable in `MSIX_WAIT` and in `MSIX_INFO` with `reset_count` (the reference Python player polls `MSIX_INFO` and never waits; it timed out under this driver on the Orin until that was added); a userspace `VECTOR_CONTROL` write desynchronizes the kernel's shadow until the next enable/disable | degraded | retire the legacy table writes or gate them on an old driver |

## 3. DMA addressing and the IOMMU

This is where the driver is most platform-dependent. Three properties of the current code
matter: `hl_pcie_dmabuf_check` requires the dma-buf mapping to be **one contiguous DMA
range** (segments adjacent, checked by walking the sg list); the whole ring must sit in
**one 4 GiB window** because `PCIE_DP_BUF_BASE_HI` is a fixed upper word; and with
`dma_address_bits=32` anything above 4 GiB is rejected.

### 3.1 Contiguity depends on an IOMMU translating for the device

With an IOMMU domain attached (arm-smmu on IGX, `intel-iommu` / `amd-iommu` on x86, the
Tegra SMMU on Jetson), the DMA API maps a scatter list into a single IOVA range, so the
adjacency check passes even when the exporter reports hundreds of segments (317 for a 20 MiB
VRAM export was measured). **Without translation** -- `iommu.passthrough=1`,
`intel_iommu=off`, a Tegra PCIe controller whose SMMU stream is bypassed, or a kernel built
without the IOMMU driver -- `sg_dma_address` is the physical address: VRAM BAR1 chunks or
system pages that are not adjacent. The import fails with `-EOPNOTSUPP` and the receive path
has nothing to fall back on.

| Platform | Likelihood | Severity |
|---|---|---|
| DGX | high: passthrough is a common tuning for GPU and NIC throughput | blocks |
| Jetson | medium: SMMU is on by default for PCIe but per-controller device-tree settings vary | blocks |
| IGX dGPU | measured working (group 22 shared with GPU and NIC) | -- |

Mitigations, in order of cost: document IOMMU translation as a hard prerequisite and have
the enumerator warn when `/sys/bus/pci/devices/<bdf>/iommu_group` is absent; make the
host-staged path (`dma_alloc_attrs`, always contiguous) selectable per platform; or add
scatter-gather to the FPGA address generator, which the fixed `BUF_START/BUF_LEN` pair
cannot express today.

### 3.2 The 4 GiB window is decided by the IOVA allocator

The dma-iommu allocator hands out IOVAs top-down below the device's DMA mask; where a given
ring lands is not under the driver's control (IGX rings land around `0x28_0100_0000`). A
ring that happens to straddle a 4 GiB boundary is rejected with `-ERANGE` at import, and a
retry with a fresh allocation may succeed. Physical addresses in passthrough mode have the
same property. This is **intermittent** on every platform, and more likely the larger the
ring and the busier the IOVA space (DGX with many GPUs and NICs). Mitigation: the receiver
should retry the CUDA allocation a bounded number of times on `INVALID_ADDRESS`; the driver
could also export the failing range so the retry can steer.

### 3.3 `dma_address_bits=32` and platforms with everything above 4 GiB

The 32-bit mode exists for bitfiles that truncate addresses. In that mode `dma_alloc_attrs`
honours the mask (the kernel allocates below 4 GiB, using swiotlb or a DMA32 zone), so the
host-staged path keeps working, but every VRAM import is rejected because dGPU BAR1
apertures sit far above 4 GiB on every platform. This is by design; note it in the
per-platform bring-up so the failure is not mistaken for a driver bug.

### 3.4 Isolation: who the FPGA can hurt

On IGX the endpoint shares one IOMMU group (and domain) with the GPU and the ConnectX NIC,
so a stray DMA write faults only if the IOVA is unmapped and corrupts silently if it is
mapped for a neighbour; the driver's `quiesce_on_release` exists because that happened
during bring-up. On DGX with ACS-capable switches each function typically gets its own
group, so a stray write faults instead of corrupting -- better isolation, but the same
`arm-smmu`/`DMAR`/`AMD-Vi` fault lines must be watched in `dmesg`. On Jetson the grouping
depends on the SMMU stream-ID assignment per controller; check `iommu_group` before
trusting the quiesce to protect anything.

### 3.5 Peer-to-peer writes into GPU memory (dGPU)

The receive path is GPUDirect: the FPGA writes into VRAM through the GPU's BAR1, using a
dma-buf the GPU driver exports with `allow_peer2peer`. On IGX both devices hang off the same
Tegra root complex and it works. Elsewhere:

| Risk | Platforms | Failure mode | Severity | Mitigation |
|---|---|---|---|---|
| Exporter refuses P2P | DGX with the proprietary (non-open) NVIDIA kernel module, or an open module older than the dma-buf P2P support | `cuMemGetHandleForAddressRange(DMA_BUF)` fails, or `dma_buf_map_attachment` returns an error; no import | blocks | require the open kernel modules; the enumerator can check `/sys/module/nvidia/version` and the `dma-buf` capability |
| ACS on the switch path | DGX: PCIe switches between the FPGA slot and the GPU have Source Validation / Request Redirect enabled | with IOMMU on, TLPs detour through the root complex and are translated: works, lower bandwidth and more latency; with IOMMU off, the kernel's `pci_p2pdma` distance check makes the exporter refuse, or writes are dropped | degraded / blocks | keep the IOMMU on; check `lspci -vvv` for `ACSCtl` on every bridge between the two; `pci=disable_acs_redir=<bdf list>` only after the security implications are accepted |
| Cross-root-complex P2P | DGX with the FPGA and GPU on different sockets' root complexes | root-complex-to-root-complex forwarding is platform-dependent; may be unsupported or slow | intermittent / degraded | place the FPGA under the same root complex as the target GPU; the enumerator already publishes `interface = pciDDDD:BB` for this comparison |
| Write ordering across the fabric | any multi-hop path | the FPGA writes payload, then the metadata block, then raises MSI-X; PCIe ordering guarantees posted writes arrive in order on one path, but the receiver's 50 ms metadata settle loop exists because the interrupt can overtake visibility on IGX | degraded | keep the settle loop; measure how often it engages per platform (the per-slot dump reports it) |

`gpu_p2p_contiguity_probe/` in the hsb_pcie repo is the tool for measuring whether a given
GPU/FPGA pair produces contiguous, importable P2P mappings; run it before anything else on a
new platform.

### 3.6 Integrated GPU: the receive path does not reach the driver

On Tegra the GPU has no VRAM; CUDA device memory is system memory. Two things break before
the driver is involved and two more need verification once it is:

1. **The allocator picks host memory.** `ReceiverMemoryDescriptor` allocates with
   `cuMemHostAlloc(DEVICEMAP)` when `CU_DEVICE_ATTRIBUTE_INTEGRATED` is set.
   `cuMemGetHandleForAddressRange(CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD)` is defined for
   device allocations, so the export fails and `import_dmabuf` never issues the ioctl.
   **Blocks.** Fix on the hololink side: allocate with `cuMemAlloc`/`cuMemCreate` on
   integrated devices when the buffer is destined for dma-buf export, or route integrated
   platforms to the host-staged source.
2. **dma-buf export of CUDA memory on Tegra** must be confirmed for the installed CUDA and
   L4T release; support arrived later than on dGPU and depends on the nvgpu/nvmap stack.
   **Blocks** until verified.
3. **GPU cache coherence.** The FPGA's PCIe writes land in DRAM. On the integrated GPU the
   L2 caches system memory, and a kernel that read slot N last frame may hold stale lines
   when the FPGA rewrites slot N. On dGPU this cannot happen because writes enter through
   the GPU's own memory path. **Safety** (silent stale pixels). Verify with the data
   generator's byte-for-byte check over many frames; if it fails, allocate the ring as
   uncached/write-combined or issue an explicit cache invalidate per frame.
4. **Host-staged registration.** `alloc_host_staged` maps the driver's `dma_alloc_attrs`
   memory with `dma_mmap_attrs` and registers it with `cuMemHostRegister(DEVICEMAP)`. On
   arm64 a coherent DMA mapping for a non-`dma-coherent` device is non-cacheable memory,
   and CUDA rejects registering some non-cacheable or non-page-backed mappings. Tegra's
   PCIe is `dma-coherent`, so this probably works there and probably fails on some x86
   configurations where the coherent mapping is uncached. **Blocks** the fallback path on
   the platforms where it is most needed; verify per platform. The comment in
   `alloc_host_staged` also gates this path on the bitfile emitting 3DW TLP headers
   correctly for sub-4 GiB targets -- a bitfile property, not a platform one, but it has
   to be re-confirmed with the bitfile shipped to each platform.

### 3.7 Page size

Some arm64 Ubuntu kernels use 64 KiB pages (`CONFIG_ARM64_64K_PAGES`). The driver
`PAGE_ALIGN`s host allocations and the mmap offset scheme `HL_PCIE_DMA_MMAP_BASE_PGOFF +
handle` is in page units. Both userspace consumers (`pcie_device_access.cpp` and the legacy
`dma_device_access.cpp`) multiply by `getpagesize()`, so this is correct on 64 KiB kernels;
the only effect is that a host-staged allocation rounds up to 64 KiB. The FPGA's 128-byte
and 4 KiB slot alignments are unaffected.

### 3.8 Board and bitfile state the software must not assume

Lessons from bring-up that still apply to the CertusPro-NX Versa. The driver writes nothing
on the sensor path (its only unsolicited BAR0 writes are the MSI-X entry restore, the
release-time quiesce and the identity reads), so when the module player stalls while the
reference Python player streams on the same board, the difference is on the application
side: the I2C mechanism (software APB sequencer through the FPGA's I2C block vs direct MMIO)
and the order of operations (the reference turns streaming on as the last register of its
table, before any ring exists; the module player turns it on after the receiver is armed).
`module_pcie_imx258_player --camera-start-early` reproduces the reference order, and its
streaming check (`Imx258Cam::streaming_status`: `MODE_SELECT` plus the SMIA frame counter at
`0x0005` sampled twice) says whether the sensor is emitting frames independently of the FPGA.
The reference player also needs `MSIX_INFO` with `reset_count` to enable its vector: it polls
`MSIX_INFO` and unmasks the table entry itself, never calling `MSIX_WAIT` or `MSIX_CTRL`. On
any new board or bitfile, compare `HSB_DATE` against the last known-good bitfile *and* run the
reference player before concluding anything about the software.

- **`HLNK_CTRL_1` (`SW_SEN_RST`) reset value is 0 and does not gate the receive path**: the
  reference player streams the camera with the word at `0`; its tooling drives these bits as
  the host-display framer reset (Sensor TX). An earlier version of this note blamed the
  frozen valid-data status on it; that was wrong. `PcieHololink::reset()` still restores the
  `0x3` end state `trigger_reset()` intends, for Ethernet parity only.
- **`SYS_RST` kills the sensor receive path until the FPGA is reconfigured.** Confirmed on
  the Orin with the IGX's bitfile (`0x5c694fdc`): the reference player streamed; the module
  player then ran with `reset()` (a `SYS_RST` pulse) and received nothing while the sensor
  demonstrably kept emitting (`FRM_CNT` advancing) and every channel register matched the
  reference's; from then on the reference player received nothing either, and a
  run without `reset()` could not undo it. The generator, DMA engine and MSI-X all survive a `SYS_RST`; the CSI
  receive block does not, so its initialization must run only at bitfile load. No reference
  PCIe flow writes `SYS_RST`. Consequences: the PCIe examples no longer call `reset()` by
  default (`--reset` opts in), `PcieHololink::reset()` warns every time, and on this transport
  `HololinkInterfaceV1::reset()` has to be read as "reload the design afterwards". Recovery is
  a power cycle or FPGA reprogram; nothing over PCIe brings the path back.
- **A marginal link at Gen3 floods the host with correctable AER interrupts and can wedge <!-- codespell:ignore aer -->
  the GPU.** The Orin's root port trained the FPGA link at 8 GT/s x8 (`LnkSta: Speed 8GT/s
  (downgraded)` from a 16 GT/s capability) and reported physical-layer `RxErr` correctable
  errors at several per millisecond -- `dmesg` shows only a rate-limited trickle, but the
  controller's line in `/proc/interrupts` climbed by thousands per second on one CPU. Every
  error is an interrupt to `aerdrv`; with that CPU saturated, `nvgpu`'s power-on path blocked
  in `synchronize_irq` and the CUDA-initializing process sat unkillable in D state, so the
  module player hung at "Initializing." before touching the device. Forcing the root port's
  target link speed to Gen2 (`setpci CAP_EXP+0x30.w`, bits 3:0 = 2, then Retrain Link,
  `CAP_EXP+0x10.w` bit 5) stopped the errors and the player streamed. 5 GT/s x8 is ample for
  the sensor data rate. The change does not survive a reboot; the durable form is the PCIe
  controller's `max-link-speed` device-tree property or a script before the driver loads.
  Check `/proc/interrupts` for the controller's line before loading anything on a new board.
- **Not every bitfile has two sensor receive instances.** A single-sensor bitfile leaves
  sensor 1's status block (`0x10280`) unmapped, reading all-ones. The module's channel
  configuration advertises two sensors regardless, which is harmless but not *checked*. A
  probe of each sensor's `PACK_CTRL` block for all-ones at enumeration would let
  `use_sensor(1)` fail early instead of arming a channel that can never complete.

## 4. PCIe enumeration and platform firmware

| Risk | Platforms | Failure mode | Severity | Mitigation |
|---|---|---|---|---|
| BAR not assigned or memory decode off after FPGA reconfiguration | Tegra root ports (seen on IGX: `Control: Mem-`, BAR0 all-ones) | probe binds but every BAR read is `0xffffffff`; identity invalid; the enumerator now flags the endpoint "not responding" | blocks until rescan | keep the hsb_pcie repo's `hololink_pcie_bringup/scripts/{rescan_subtree,fix_bar_assignment}.sh`; `pci_enable_device_mem` does turn decode on, so a reload usually fixes it |
| Non-zero PCI domain | IGX (`0005:`), Grace, some Jetson controllers | none: the driver reports `domain` in the identity and userspace keys on the full BDF | none | none |
| Hotplug / `remove` + `rescan` while a receiver is open | any | `hl_pcie_remove` masks the table and frees IRQs while an fd may be blocked in `MSIX_WAIT`; the driver wakes waiters, but userspace must expect `-ENODEV`-class errors | degraded | the receiver treats a failed wait as fatal and exits its monitor; acceptable |
| `keep_pci_enabled` default | any | remove leaves memory decoding on so a subsequent reload sees a live BAR; on platforms that power-manage slots this may be undesirable | none | parameter exists |
| Vendor/device ID overlap | any | the ID table claims two vendor IDs (`0x1204`, `0x19aa`) with several device IDs; another driver claiming the same IDs would race for the function | unlikely | none |
| Secure Boot / lockdown | DGX and any UEFI Ubuntu host with Secure Boot on | an unsigned `.ko` is refused (`Key was rejected by service`); with kernel lockdown, `resource0` mmap from userspace is also refused, so the control plane silently uses the ioctl fallback (already implemented) | blocks (load) | sign the module through DKMS with an enrolled MOK, or ship it in a signed package; document the ioctl fallback's lower BAR throughput |

## 5. Assumptions userspace makes about the driver

These are not driver bugs, but they are what breaks first when the driver runs on a new
platform:

- Sysfs paths: `/sys/class/hololink_pcie/hololink_pcieN/{bdf,uuid,...}`,
  `/sys/bus/pci/devices/<bdf>/hololink_pcie/`, and `/sys/bus/pci/devices/<bdf>/resource0`.
  Stable across kernels; the `resource0` mmap needs root or a udev rule and falls back to
  ioctls otherwise.
- Discovery is the driver's bootp announcement broadcast on the loopback subnet
  (`netif_rx()` into `init_net.loopback_dev`, `<packed BDF>:12268 -> 127.255.255.255:12267`,
  the local table's broadcast route attached, so every `INADDR_ANY` listener on the port
  receives it). A listener in another network namespace does not see it, nor does one bound
  to `127.0.0.1` rather than `INADDR_ANY`; `lo` down withdraws the `127.255.255.255` route
  and the tick is skipped. The Adapter itself has no PCIe code.
- The identity ioctl's `hsb_version/hsb_date/hsb_status` are probe-time values; only the
  UUID is live. The "not responding" decision keys on the UUID for that reason.
- `HL_PCIE_IOC_MSIX_CTRL` is required by the module receive path; an older driver fails the
  receiver's `start()` with a clear message rather than hanging.

## 6. Build, packaging and deployment

- **Out-of-tree kbuild via CMake.** `src/ehsb/kmod/CMakeLists.txt` copies the source into a
  staging directory and runs `make -C $KERNEL_DIR M=...`; `KERNEL_DIR` defaults to
  `/lib/modules/$(uname -r)/build`. On Ubuntu that is `linux-headers-$(uname -r)`; on L4T
  the headers package is `nvidia-l4t-kernel-headers` and its `build` symlink has at times
  pointed at a directory needing `make modules_prepare`. Verify per L4T release.
- **DKMS** is the practical answer to kernel updates on a fleet: a `dkms.conf` with
  `MAKE="make -C ${kernel_source_dir} M=${dkms_tree}/..."` rebuilds on every kernel install
  and integrates with Ubuntu's MOK signing. There is no `dkms.conf` in the repo today.
- **Module signing** (section 4) on Secure Boot hosts.
- **The uapi mirror** (`hololink_module/host/pcie/include/hololink/module/pcie/`) must match
  `src/ehsb/kmod/include/hololink_pcie_uapi.h` byte for byte; `src/ehsb/kmod/CMakeLists.txt`
  fails the configure when it drifts.
- **`MODULE_VERSION`** is not declared, so `modinfo` cannot tell the kernel-owned-MSI-X
  driver from its predecessor; the userspace probe of `HL_PCIE_IOC_MSIX_CTRL` is the only
  version test. Declare a version and expose it in the identity `reserved[]` words.

## 7. Recommended validation per platform

Run in this order on every new platform before any application work. Each step isolates
one of the risks above.

1. `modinfo build/ehsb/kmod/hololink_pcie_drv.ko` loads (signing, vermagic), then `insmod`;
   `dmesg` shows `allocated N MSI-X vectors` with N >= 33 today, >= 2 once the minimum is
   a parameter (section 2).
2. `cat /sys/bus/pci/devices/<bdf>/iommu_group/type` (or the group directory's existence)
   confirms translation; note which other devices share the group (3.1, 3.4).
3. `hololink_pcie_identity --bdf <bdf> --show`: identity valid, `responding` true (4).
4. `lspci -vvv` on every bridge between the FPGA and the GPU: `ACSCtl` flags, and confirm
   both sit under the same root complex (3.5).
5. `gpu_p2p_contiguity_probe`: importable, contiguous, single 4 GiB window (3.1, 3.2, 3.5).
6. `module_pcie_imx258_player --verify-registers --frame-limit 200 --headless` with a
   streaming camera: proves DMA, MSI-X, ordering and, on iGPU, cache coherence (3.6.3).
   Watch `dmesg` for `arm-smmu`/`DMAR`/`AMD-Vi` faults and `move_notify`.
7. Kill the player with `SIGKILL` mid-stream and confirm `dmesg` stays quiet and the next
   run starts clean (quiesce on release, vector release).
8. On iGPU only: run step 6 with the host-staged source as well (3.6.4).

## 8. Summary of the highest risks

| # | Risk | Platforms | Severity |
|---|---|---|---|
| 1 | Integrated GPU allocates host memory that cannot be dma-buf exported; the module receive path never reaches the driver | Jetson, IGX iGPU | blocks |
| 2 | No IOMMU translation (passthrough) breaks the single-contiguous-range requirement | DGX (likely tuning), Jetson (config-dependent) | blocks |
| 3 | dma-buf P2P export requires the open NVIDIA kernel modules and a P2P-capable fabric path (ACS, same root complex) | DGX | blocks / degraded |
| 4 | 33-vector MSI-X minimum on a shared Tegra MSI pool (receive needs only 2) | Jetson, IGX iGPU | blocks |
| 5 | GPU L2 staleness for FPGA writes into system memory | Jetson, IGX iGPU | safety |
| 6 | Ring straddling a 4 GiB IOVA boundary | all | intermittent |
| 7 | Unsigned module on Secure Boot hosts; no DKMS | DGX, any UEFI Ubuntu | blocks (load) |
| 8 | Host-staged fallback's `cuMemHostRegister` of coherent DMA memory | x86 DGX, verify on Tegra | blocks the fallback |
| 9 | Kernel API drift on 7.x beyond the guarded set | all, over time | build break, visible |

Items 1, 4 and 5 are specific to the integrated-GPU platforms and are the reason those
should be treated as a separate bring-up rather than a redeploy of the IGX dGPU result.
