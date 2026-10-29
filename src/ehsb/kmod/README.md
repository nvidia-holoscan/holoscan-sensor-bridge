# `hololink_pcie` kernel driver

Out-of-tree PCIe character driver for the Hololink PCIe endpoint: BAR0 register access
(`HL_PCIE_IOC_READ/WRITE`), coherent host DMA allocations, dma-buf import of GPU memory
for GPUDirect receive, per-vector MSI-X wait and enable (`HL_PCIE_IOC_MSIX_WAIT`,
`HL_PCIE_IOC_MSIX_CTRL`), and the device identity (`HL_PCIE_IOC_IDENTITY` plus read-only
sysfs attributes under `/sys/class/hololink_pcie/hololink_pcieN/`).

Layout:

| File | Role |
|---|---|
| `hololink_pcie_drv.c` | the driver |
| `include/hololink_pcie_uapi.h` | ioctl ABI shared with userspace. The userspace copy, `hololink_module/module/hsb_pcie/hololink_pcie_uapi.h`, must be byte-identical to this one; `CMakeLists.txt` fails the configure otherwise |
| `include/hololink_pcie_regs.h` | BAR0 register offsets the driver uses |
| `include/hololink_pcie_bootp.h` | the 300-byte bootp announcement the driver publishes per bound endpoint (layout, byte helpers, inline filler). Plain C over `__u8`, so the userspace copy at `hololink_module/module/hsb_pcie/hololink_pcie_bootp.h` (checked byte-identical at configure time) pins the layout to the host parser in a test |
| `Makefile` | kbuild entry point; `make -C src/ehsb/kmod` builds in place |
| `CMakeLists.txt` | stages the sources into `<build>/ehsb/kmod/` and runs kbuild there, as target `hololink_pcie_drv`, so the source tree stays clean |
| `cross-platform-kmd.md` | risk assessment for L4T iGPU, L4T dGPU and DGX platforms on 6.x / 7.x kernels |

## Supported kernels

Designed and tested against Linux 6.x (IGX Orin 5.15-era images are no longer a target). The
hard minimum is **5.18**: `netif_rx()` callable from process context (the bootp publisher runs on
a workqueue), `IRQF_NO_AUTOEN`, and the dma-buf dynamic-attach API. `hololink_pcie_drv.c` refuses
to compile on anything older (`#error`), so an unsupported tree fails at build time rather than
misbehaving at load. The remaining version guards (`class_create` 6.4, `linux/unaligned.h` 6.12,
`MODULE_IMPORT_NS` 6.13, `kernel_bind` 7.0 if the socket path is ever used) cover the 6.x line.

## Device discovery: bootp on loopback

The hololink_module `Adapter` discovers devices only through bootp on UDP 12267, as Ethernet
boards announce themselves. The driver plays that role for PCIe endpoints: a single delayed work
on the shared system workqueue runs once per second and, for every bound endpoint whose UUID is
valid, injects one 300-byte bootp datagram into the loopback interface with `netif_rx()`
(`<packed BDF>:12268 -> 127.255.255.255:12267` -- the destination is the loopback subnet
broadcast, so exactly like an Ethernet board's broadcast the announcement is delivered to every
socket bound to port 12267 on `INADDR_ANY`, where a unicast to `127.0.0.1` would reach only one
of the sockets sharing the port (most specific bind wins) and a later `INADDR_ANY` binder would
take it away from the discovery daemon; the IP source is the BDF packed as `(domain << 16) |
(bus << 8) | devfn`, the same value as `ciaddr`, so even a raw listener such as
`hololink-enumerate` sees one peer per endpoint; the local table's `broadcast 127.255.255.255
dev lo` route is attached to the skb so the frame passes with no sysctl changes). The first announcement goes out at probe, so a hot-plugged board
is visible within a second. The header says what the carrier is: `htype = 0x50` ('P', not an
IANA hardware type; Ethernet boards send 1) and `hlen = 4` with `chaddr` = the BDF big-endian
(`dom_hi, dom_lo, bus, devfn`), which the `hsb_pcie` module renders as the textual BDF
(`0005:01:00.0`) into `peer_ip` and `mac_id` in its `update_metadata` -- the hololink_module
bootp parser is transport-agnostic and reports the source address and the colon-hex bytes, so
`adapter_enumerator` shows the BDF and `adapter_enumerator --raw` the parser's view; the legacy
enumerator behind `hololink-enumerate` renders the BDF itself. The same BDF travels packed in `ciaddr`
(`(domain << 16) | (bus << 8) | devfn`, rendered `0.5.1.0` in `client_ip_address`), which is
also the datagram's IPv4 source, so older parsers derive the BDF from it and nothing about the
source is lost. The vendor area follows the Ethernet firmware's
layout end to end (tag length 56): the board UUID, the 7-byte serial -- the endpoint's PCIe
Device Serial Number when it has the capability (its low 7 bytes, LSB first; status bit1 set),
else the placeholder `"PCIE"` + BDF bytes -- and, when loaded with `bootp_ptp=1`, a snapshot of
the endpoint's PTP clock (bytes 45..54, status bit0 set). The `hsb_pcie` module, selected by the
UUID, stamps `pcie_bdf` / `pcie_domain` / ... and keeps a Device Serial Number as `serial_number`
(the placeholder becomes `pcie-<bdf>`). `remove` cancels the work synchronously
before tearing the device down and re-arms it for the devices that remain; module exit cancels
it once more. `bootp_announce=0` disables publishing. Only the initial network namespace sees the
announcements. Check with `sudo tcpdump -i lo -n udp port 12267`. A raw listener must bind
`INADDR_ANY` (or `127.255.255.255`): a socket bound to `127.0.0.1` does not match the broadcast
destination and never receives an announcement. A userspace sender that mimics the driver (as
`hololink_module_pcie_bootp_test` does) sends to `127.255.255.255` with `SO_BROADCAST` set; it
cannot reproduce the packed-BDF source, which lies in `0.0.0.0/8`.

Build knobs: the whole PCIe transport (this module, the hsb_pcie hololink_module, the
receiver, tools, tests and examples) is behind `-DHOLOLINK_BUILD_PCIE=ON` (default OFF);
within that, `-DHOLOLINK_BUILD_KERNEL_MODULE=OFF` skips the kernel module; `-DKERNEL_DIR=<headers>` points
kbuild at a kernel other than the running one (defaults to `/lib/modules/$(uname -r)/build`,
and the module is skipped with a warning when that directory is absent).

The host1x syncpoint ioctls import `host1x_syncpt_*` from the kernel's `host1x` module, and
with `CONFIG_MODVERSIONS` the `.ko` records a CRC per imported symbol that must be the one the
loaded host1x exports; a module built with the wrong ones compiles cleanly and `insmod` then
refuses it with `Invalid parameters` (`dmesg`: `disagrees about version of symbol
host1x_syncpt_...`). Which file holds the right CRCs depends on who provides host1x: on an IGX
Orin (`7.0.0-nvidia`) host1x is in-tree and the base kernel's `Module.symvers`, which kbuild
uses by itself, is correct; on an IGX Thor (R38) or a Jetson it is the out-of-tree `host1x(O)`,
whose own `Module.symvers` must be handed to kbuild -- the base file there still lists the
in-tree CRCs, which is exactly the mismatch. The Makefile looks for the out-of-tree file at
`/usr/src/nvidia/nvidia-public/`, `/usr/src/nvidia/nvidia-oot/` and
`/usr/src/linux-headers-tegra-oot-*-<release>/nvidia-oot/` and uses it when present; when none
is present it checks what `modinfo -k <release> -n host1x` resolves to and warns if that is an
out-of-tree module. Override with `NVIDIA_PUBLIC_SYMVERS=<file>` in the environment of
`cmake --build` (or on the `make` line), and `NVIDIA_SYMVERS_ROOT=<dir>` to search under a
sysroot. `scripts/deploy_pcie_kmod.sh` repeats the comparison on the target before `insmod`
(step 1, "host1x CRCs") and names the rebuild command on a mismatch.

## PTP: clock and time sync

The endpoint has a PTP clock, the `bar_ptp` block at BAR0 `0x100..0x1FF` (bringup register map
`fpga/registers/hololink_pcie/2606/bar_ptp.csv`; commands are edge-triggered handshakes against
`STATUS.COMMAND_ACK`, and SNAPSHOT latches a coherent seconds/nanoseconds pair at `0x180..0x188`).
The driver offers it two ways, **both off by default**: current bitfiles without the block stub
that window with `pready=0`, and a read there can hang on the interconnect watchdog rather than
return all-ones. Enable them only on a bitfile that implements `bar_ptp` (a STATUS read at
`0x19C`, e.g. `hololink_pcie_identity --bdf $BDF --peek 0x19c`, shows bit0 PRESENT set, not
`0xffffffff`).

| Parameter | Default | Meaning |
|---|---|---|
| `ptp_clock` | `0` | Register the clock as a PTP hardware clock, `/dev/ptpN`, one per endpoint (`ptp_clock_register`, needs `CONFIG_PTP_1588_CLOCK`). `gettimex64` / `settime64` / `adjtime` / `adjfine`; no external timestamps. |
| `bootp_ptp` | `0` | SNAPSHOT the clock before every bootp announcement and carry it in vendor bytes 45..54 (`ptp_timestamp_s/ns`, status bit0), as the Ethernet firmware does. |
| `ptp_clk_hz` | `125000000` | The PTP timer clock; only scales `adjfine` (`default_inc = 1e9 * 2^24 / Hz`). 125 MHz on the CertusPro-NX Versa. |
| `ptp_max_adj_ppb` | `1000000` | `adjfine` pull range. The timer applies `inc = default_inc + FREQ_ADJ` unsaturated, so an over-wide range lets a servo stall the counter. |

`scripts/deploy_pcie_kmod.sh --ptp[=CLK_HZ]` loads with `bootp_ptp=1 ptp_clock=1
ptp_clk_hz=CLK_HZ` (default 125000000), persists the same in `/etc/modprobe.d/hololink_pcie_drv.conf`
for the boot-time load, and prints the `/dev/ptpN` it registered; a deploy without `--ptp`
removes that file again. To find the node by hand: `readlink -f /sys/class/ptp/ptp*/device`
names the PCI function each PHC belongs to (a ConnectX usually holds `ptp0`).

Time sync is **phc2sys** (PTM is not an option: Orin has none without a CX7), with the HSB
clock as the slave (`-c`) and either the host clock or the CX7's PHC as the source:

```bash
sudo phc2sys -s CLOCK_REALTIME -c /dev/ptpN -O 0 -S 0.001 -m   # Orin -> HSB
sudo phc2sys -s /dev/ptp0      -c /dev/ptpN -O 0 -S 0.001 -m   # CX7  -> HSB
```

`-O 0`: no fixed offset; `-S 0.001`: step (via `settime64` / `adjtime`, the first one typically
whole seconds) when the offset exceeds 1 ms, otherwise steer the frequency (`adjfine`); `-m`:
print each measurement. Expect an initial step, then `freq` corrections settling to
microsecond-class offsets (Orin-to-CX7 class, not the nanoseconds of Ethernet HSB 1588). The
FPGA writes frame metadata timestamps from the same timer, so once phc2sys has converged they
are directly comparable with host `CLOCK_REALTIME`, and with `bootp_ptp=1` every announcement's
`ptp_timestamp_s` matches the host's `date +%s` within the announcement latency.

Loading is a privileged, explicit step and is not part of `install`:

```
sudo rmmod hololink_pcie_drv 2>/dev/null; sudo insmod <build>/ehsb/kmod/hololink_pcie_drv.ko
```

## Cold-boot procedure

The complete sequence from power-off to a listed endpoint, with the commands for each step.
Steps 3 to 6 assume the hololink tree is built (`cmake -S . -B build && cmake --build build`)
and that you are in its root. `$BDF` is the endpoint's PCI address, `$RP` the root port it hangs
off; both are derived in step 3.

**1. Power on the FPGA board.** It must be up and its PCIe link ready before the host
enumerates the bus, or the host will not see it at all (a hot-plugged endpoint needs
`echo 1 | sudo tee /sys/bus/pci/rescan` and, on Tegra root ports, sometimes a BAR reassignment;
the reference tree's `hololink_pcie_bringup/scripts/rescan_subtree.sh` covers that case).

**2. Power on the AGX / IGX.** Once logged in, confirm the endpoint enumerated with both BARs
assigned. Nothing is bound yet, so `Mem-` and `[disabled]` are expected here:

```
lspci -Dnn -d 1204:
lspci -Dnn -d 1204: | awk '{print $1}' | head -1 | xargs -I{} lspci -s {} -vv | grep -E 'Control:|Region [01]'
```

**3. Set the link to 5 GT/s (Gen2).** On the AGX Orin the link trains at 8 GT/s and the
FPGA's transmitter is marginal there: the root port reports physical-layer `RxErr`
correctable errors at several per millisecond, each an interrupt, which saturates a CPU and
has left CUDA initialization stuck inside the GPU driver. Force the root port's target
speed to Gen2 and retrain, then confirm the error interrupts have stopped:

```
BDF=$(lspci -Dnd 1204: | awk '{print $1}' | head -1)
RP=$(basename "$(readlink -f /sys/bus/pci/devices/$BDF/..)")
echo "endpoint $BDF behind root port $RP"

# Link Control 2 (CAP_EXP+0x30): target link speed in bits 3:0 -- 1=2.5, 2=5, 3=8, 4=16 GT/s
old=$(sudo setpci -s $RP CAP_EXP+0x30.w)
sudo setpci -s $RP CAP_EXP+0x30.w=$(printf '%04x' $(( 0x$old & ~0xF | 0x2 )))
# Link Control (CAP_EXP+0x10): bit 5 = Retrain Link
old=$(sudo setpci -s $RP CAP_EXP+0x10.w)
sudo setpci -s $RP CAP_EXP+0x10.w=$(printf '%04x' $(( 0x$old | 0x20 )))
sleep 1
sudo lspci -s $BDF -vv | grep 'LnkSta:'          # expect: Speed 5GT/s, Width x8

# The PCIe controller's interrupt line must be quiet: read it twice, the count must not climb.
grep -E 'tegra-pcie-intr|aerdrv' /proc/interrupts; sleep 3; grep -E 'tegra-pcie-intr|aerdrv' /proc/interrupts
```

If the count still climbs at Gen2, try `0x1` (Gen1) in the first `setpci`; if it climbs even
there, the problem is mechanical (seating, cable, riser). This setting does not survive a
reboot; the durable form is the `max-link-speed` property of that PCIe controller in the
device tree, or this block in a boot-time script that runs before step 4.

**4. Load the kernel module and bind the endpoint.**

```
lsmod | grep hololink_pcie_drv && sudo rmmod hololink_pcie_drv     # e.g. the reference tree's copy
sudo insmod build/ehsb/kmod/hololink_pcie_drv.ko

# Bound? The driver's ID table claims 1204:9c25 / 9c1d / 0e00c and 19aa:9c25 by itself.
lspci -s $BDF -k | grep 'Kernel driver in use'
```

If that line is missing (a bitfile with an ID not in the table), bind it explicitly:

```
echo hololink_pcie | sudo tee /sys/bus/pci/devices/$BDF/driver_override
echo $BDF | sudo tee /sys/bus/pci/drivers_probe
```

Then verify: memory decode on, a class device present, the probe in `dmesg`, and the loaded
module being the file you built:

```
sudo lspci -s $BDF -vv | grep -E 'Control:'                              # Mem+ BusMaster+
ls /sys/class/hololink_pcie/; cat /sys/class/hololink_pcie/hololink_pcie0/{bdf,uuid,hsb_version,msix_vectors}
sudo dmesg | grep -i hololink_pcie | tail -n 5                          # "allocated 64 MSI-X vectors", "ready"
[ "$(cat /sys/module/hololink_pcie_drv/srcversion)" = "$(modinfo -F srcversion build/ehsb/kmod/hololink_pcie_drv.ko)" ] && echo "loaded module matches the build"
```

(`srcversion` exists only when the kernel was built with `CONFIG_MODULE_SRCVERSION_ALL`; the
Tegra 6.8 kernels were not. The GNU build-id works everywhere: compare `readelf -n` of the
`.ko` with `/sys/module/hololink_pcie_drv/notes/.note.gnu.build-id`, which is what the script
below does.)

`scripts/deploy_pcie_kmod.sh` does the reload and the build-id check for you and also
installs the module under `/lib/modules/$(uname -r)/updates/` with a
`/etc/modules-load.d/hololink_pcie_drv.conf` entry, so it loads at boot. Run it on the
target (`scripts/deploy_pcie_kmod.sh --path build/ehsb/kmod`) or from the build host, which
copies the script and the `.ko` over and runs it there:

```bash
scripts/deploy_pcie_kmod.sh --path build/ehsb/kmod nvidia@10.137.145.93:~/Documents/jbonde
```

It refuses a `.ko` whose vermagic is not the running kernel, names any process holding
`/dev/hololink_pcie*` instead of failing on `rmmod`, and finishes by confirming that the
loaded module, the inserted file and the installed copy are the same build. The remote form
removes the two staged files from the target once the install succeeds (the copy under
`/lib/modules` stays); if it fails, it leaves them there and prints where they are.

**5. Device-node permissions.** The node is created `0600 root:root`. Either the permanent
group setup (once per machine, then log in again or `newgrp hololink`):

```
sudo groupadd -f hololink
sudo usermod -aG hololink $USER
echo 'KERNEL=="hololink_pcie[0-9]*", MODE="0660", GROUP="hololink"' | sudo tee /etc/udev/rules.d/99-hololink-pcie.rules
sudo udevadm control --reload && sudo udevadm trigger --subsystem-match=hololink_pcie
sudo chgrp hololink /dev/hololink_pcie*; sudo chmod 660 /dev/hololink_pcie*   # nodes that predate the rule
ls -l /dev/hololink_pcie*                                                     # crw-rw---- root hololink
```

or, for one session, `sudo chmod 666 /dev/hololink_pcie*` (lost on the next driver reload).
Running the applications under `sudo -E` also works; `-E` keeps `DISPLAY`/`XAUTHORITY` for
Holoviz.

**6. Enumerate.**

```
timeout 3 build/hololink_module/tools/adapter_enumerator --dedup
```

Expected within a second: one record with `peer_ip = mac_id = "$BDF"` (rendered from `chaddr`
by the parser), `hardware_type = 80`, `client_ip_address = "0.5.1.0"`, `interface = "lo"`, `fpga_uuid =
58c1419b-1e7a-4120-95fd-4efc156187c1`, `serial_number` = the 14-hex Device Serial Number with
`pcie_serial_source = dsn` (compare `sudo lspci -vv -s $BDF | grep -A1 'Device Serial'`: the last
seven groups, reversed) or `pcie-$BDF` with `pcie_serial_source = bdf` when the endpoint has no
DSN capability, `bootp_psn` counting up, `bootp_status`. Nothing listed: check `sudo tcpdump -i lo -n
udp port 12267` -- datagrams present means the module `.so` is not on the module path; none
means the driver is not announcing (`cat /sys/module/hololink_pcie_drv/parameters/bootp_announce`,
and the endpoint's UUID must be valid, i.e. step 4's `Mem+`). From here `module_pcie_imx258_player --hololink $BDF --frame-limit
300` streams the camera. It does not issue `SYS_RST` unless given `--reset`, which on current
bitfiles leaves the sensor receive path dead until the FPGA is reconfigured.

## Start-up script

Everything from a fresh build to a first enumeration, in order, with a check after each
step. Save it as `hololink_pcie_start.sh` (or run the steps by hand; each is one or two
commands) and run it from the hololink repo root. It needs `sudo` for the module and
device-node steps only; the build and the enumeration run as you.

```bash
#!/usr/bin/env bash
# hololink_pcie_start.sh -- build, (re)load, verify, open up and enumerate the
# hololink_pcie kernel driver. Run from the hololink repo root.
#
#   hololink_pcie_start.sh [-b BUILD_DIR] [-d DDDD:BB:DD.F] [-p "param=value ..."]
#                          [--no-build] [--world]
#
#   -b   configured CMake build tree (default: build)
#   -d   PCI address to bind explicitly if the driver did not claim it at insert
#   -p   module parameters for insmod, e.g. -p "dma_address_bits=32"
#   --no-build   skip step 1 (use the .ko already in the build tree)
#   --world      step 6 does chmod 666 instead of installing the udev group rule
set -euo pipefail

BUILD=build; BDF=""; PARAMS=""; DO_BUILD=1; WORLD=0
GROUP=hololink; MOD=hololink_pcie_drv
while [ $# -gt 0 ]; do
    case "$1" in
        -b) BUILD=$2; shift 2 ;;
        -d) BDF=$2; shift 2 ;;
        -p) PARAMS=$2; shift 2 ;;
        --no-build) DO_BUILD=0; shift ;;
        --world) WORLD=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
KO="$BUILD/ehsb/kmod/$MOD.ko"
step() { printf '\n== %s\n' "$*"; }
die() { echo "FAIL: $*" >&2; exit 1; }

step "1. Build the driver"
if [ "$DO_BUILD" = 1 ]; then
    [ -f "$BUILD/CMakeCache.txt" ] || die "$BUILD is not a configured build tree (cmake -S . -B $BUILD first)"
    cmake --build "$BUILD" --target hololink_pcie_drv
fi
[ -f "$KO" ] || die "$KO not found"

step "2. Check the module was built for this kernel"
VERMAGIC=$(modinfo -F vermagic "$KO" | awk '{print $1}')
[ "$VERMAGIC" = "$(uname -r)" ] \
    || die "vermagic '$VERMAGIC' is not the running kernel '$(uname -r)'; rebuild with -DKERNEL_DIR=/lib/modules/$(uname -r)/build"
SRCVERSION=$(modinfo -F srcversion "$KO" || true)
echo "  $KO: vermagic $VERMAGIC srcversion ${SRCVERSION:-n/a}"

step "3. Remove the old module"
if lsmod | grep -q "^$MOD "; then
    # An open /dev/hololink_pcieN (a receiver, hololink_pcie_identity) keeps the
    # module busy and rmmod would fail; name the holders instead of failing obscurely.
    if command -v fuser >/dev/null 2>&1 && fuser -s /dev/hololink_pcie* 2>/dev/null; then
        fuser -v /dev/hololink_pcie* 2>&1 | sed 's/^/  /' >&2 || true
        die "/dev/hololink_pcie* is open by the processes above; stop them first"
    fi
    sudo rmmod "$MOD"
    echo "  removed the previously loaded $MOD"
else
    echo "  not loaded"
fi

step "4. Insert the new module"
# shellcheck disable=SC2086  # PARAMS is a deliberately word-split list
sudo insmod "$KO" $PARAMS
echo "  inserted $KO ${PARAMS:+with $PARAMS}"

step "5. Verify it is loaded, is this exact build, and has bound the endpoint"
lsmod | grep -q "^$MOD " || die "$MOD is not in lsmod after insmod"
if [ -n "$SRCVERSION" ] && [ -r "/sys/module/$MOD/srcversion" ]; then
    LOADED=$(cat "/sys/module/$MOD/srcversion")
    [ "$LOADED" = "$SRCVERSION" ] \
        || die "the loaded module's srcversion $LOADED differs from $KO ($SRCVERSION): a different build is loaded"
    echo "  loaded srcversion matches the built file"
fi
if [ -n "$BDF" ] && [ ! -e "/sys/bus/pci/devices/$BDF/driver" ]; then
    # The driver's ID table did not claim it (a bitfile with an unlisted device ID);
    # bind it explicitly.
    echo hololink_pcie | sudo tee "/sys/bus/pci/devices/$BDF/driver_override" >/dev/null
    echo "$BDF" | sudo tee /sys/bus/pci/drivers_probe >/dev/null
fi
shopt -s nullglob
NODES=(/sys/class/hololink_pcie/hololink_pcie*)
shopt -u nullglob
[ ${#NODES[@]} -gt 0 ] \
    || die "no /sys/class/hololink_pcie/hololink_pcieN: the endpoint is not bound (try -d <bdf>; check 'lspci -d 1204:' and 'sudo dmesg | tail')"
for n in "${NODES[@]}"; do
    bdf=$(basename "$(readlink -f "$n/device")")
    printf '  %s -> %s  uuid=%s  hsb_version=%s  msix_vectors=%s\n' \
        "$(basename "$n")" "$bdf" "$(cat "$n/uuid")" "$(cat "$n/hsb_version")" "$(cat "$n/msix_vectors")"
    # Memory decode off means every BAR read is all-ones and the endpoint will enumerate
    # as "not responding" (this is the usual state after an FPGA reconfiguration).
    if ! lspci -s "$bdf" -vv 2>/dev/null | grep -E '^\s*Control:' | grep -q 'Mem+'; then
        echo "  WARNING: $bdf has memory decode off (lspci 'Control: Mem-'); rescan/reassign BARs, then rerun"
    fi
done
sudo dmesg | grep -iE "$MOD|hololink_pcie" | tail -n 5 | sed 's/^/  dmesg: /'

step "6. Device node permissions"
if [ "$WORLD" = 1 ]; then
    sudo chmod 666 /dev/hololink_pcie*
else
    sudo groupadd -f "$GROUP"
    RULE=/etc/udev/rules.d/99-hololink-pcie.rules
    if [ ! -f "$RULE" ]; then
        echo "KERNEL==\"hololink_pcie[0-9]*\", MODE=\"0660\", GROUP=\"$GROUP\"" | sudo tee "$RULE" >/dev/null
        sudo udevadm control --reload
        echo "  installed $RULE"
    fi
    sudo udevadm trigger --subsystem-match=hololink_pcie
    # Nodes that existed before the rule are not always re-permissioned by the trigger.
    sudo chgrp "$GROUP" /dev/hololink_pcie*
    sudo chmod 660 /dev/hololink_pcie*
    if ! id -nG "$USER" | tr ' ' '\n' | grep -qx "$GROUP"; then
        sudo usermod -aG "$GROUP" "$USER"
        echo "  added $USER to group $GROUP: log in again (or run 'newgrp $GROUP') before opening the node"
    fi
fi
ls -l /dev/hololink_pcie*

step "7. Enumerate"
# adapter_enumerator listens until interrupted; PCIe endpoints are announced as soon as it
# starts, so a few seconds is enough. A '!! DEVICE NODE NOT OPENABLE FROM THIS PROCESS'
# line here means step 6's group change has not reached this shell yet.
timeout 3 "$BUILD/hololink_module/tools/adapter_enumerator" --dedup || [ $? -eq 124 ]
```

Before step 1 on a board you have not run before, look at the link once: `sudo lspci -s <bdf>
-vv | grep LnkSta` and the PCIe controller's line in `/proc/interrupts` read twice a few
seconds apart. A count climbing by thousands is a correctable-error storm from a marginal
link (seen on an AGX Orin at 8 GT/s); it saturates a CPU and can leave CUDA initialization
stuck in the GPU driver before this software is even reached. Force the root port to Gen2
and retrain (`cross-platform-kmd.md`, section 3.8) and re-check before continuing.

What each step checks and what its failure means:

| Step | Check | If it fails |
|---|---|---|
| 1 | `cmake --build --target hololink_pcie_drv` | kernel headers missing for the running kernel: install `linux-headers-$(uname -r)` (L4T: `nvidia-l4t-kernel-headers`) or point `-DKERNEL_DIR` at them |
| 2 | `modinfo -F vermagic` equals `uname -r` | the build tree was configured against another kernel; reconfigure with `-DKERNEL_DIR` |
| 3 | `rmmod` of the previous module | a process still holds `/dev/hololink_pcieN`; `fuser -v` names it |
| 4 | `insmod` | `Key was rejected by service`: Secure Boot wants a signed module (see `cross-platform-kmd.md` section 4); `Invalid module format`: step 2 was skipped |
| 5 | `lsmod`, `/sys/module/.../srcversion` equals the file's, a class device exists, `lspci` shows `Mem+` | not bound: the device ID is not in the driver's table, use `-d <bdf>`; `Mem-`: BAR decode off after an FPGA reload, rescan the slot; `dmesg` has the probe's reason (`allocated N MSI-X vectors` on success) |
| 6 | `ls -l` shows `crw-rw---- root hololink` (or `crw-rw-rw-` with `--world`) | a freshly added group membership needs a new login or `newgrp hololink` |
| 7 | the endpoint appears with its UUID, `pcie_identity_valid = 1` and no `!!` lines | `NOT RESPONDING`: step 5's `Mem-`; `DEVICE NODE NOT OPENABLE`: step 6's group has not reached this shell |

## Device node permissions

`device_create()` is called with no devnode callback, so `/dev/hololink_pcieN` appears as
`0600 root:root`. Enumeration still works for any user because the identity attributes under
`/sys/class/hololink_pcie/hololink_pcieN/` are world-readable (`adapter_enumerator` then
prints `!! DEVICE NODE NOT OPENABLE FROM THIS PROCESS` and leaves `pcie_bar0_len` /
`pcie_dma_address_bits` at 0, the two fields only the ioctl provides), but nothing that has to
open the node -- the `hsb_pcie` module's control plane, the receiver, `hololink_pcie_identity`
-- can run as a non-root user until the node is made accessible. `scripts/deploy_pcie_kmod.sh`
does this in its step 3 on every deploy: it creates the `hololink` group, installs
`/etc/udev/rules.d/99-hololink-pcie.rules`, reloads udev and adds the invoking user
(`$SUDO_USER`) to the group, then checks in step 5 that each node reads `crw-rw---- root
hololink`. By hand, once per machine, it is:

```
sudo groupadd -f hololink
sudo usermod -aG hololink $USER
echo 'KERNEL=="hololink_pcie[0-9]*", MODE="0660", GROUP="hololink"' \
    | sudo tee /etc/udev/rules.d/99-hololink-pcie.rules
sudo udevadm control --reload && sudo udevadm trigger --subsystem-match=hololink_pcie
```

Either way the new group membership applies to new logins only: log in again (or `newgrp
hololink` in an existing shell), after which `ls -l /dev/hololink_pcie0` shows `crw-rw----
root hololink`. A `sudo chmod 666 /dev/hololink_pcie0` works until the driver is next
reloaded. BAR0 access through
`/sys/bus/pci/devices/<bdf>/resource0` needs write permission on that sysfs file (root); the
control plane falls back to the driver's 32-bit ioctls without it, which is correct and
slower.

The in-process `hsb_pcie` hololink_module and the module-native `PcieReceiver` are the
consumers; `docs/pcie_hololink_module_design.md` section 4 lists the driver's additions over
the reference bring-up driver.
