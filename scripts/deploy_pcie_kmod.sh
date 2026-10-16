#!/usr/bin/env bash
# deploy_pcie_kmod.sh -- install the hololink_pcie kernel driver so it loads now and at boot,
# locally or on a remote target.
#
#   deploy_pcie_kmod.sh [--path DIR] [--ptp[=CLK_HZ]] [--dry-run]
#   deploy_pcie_kmod.sh [--path DIR] [--ptp[=CLK_HZ]] [--dry-run] user@host:path
#
#   --path DIR      directory holding hololink_pcie_drv.ko (default: the current directory)
#   --ptp[=CLK_HZ]  enable the endpoint's PTP clock: load with bootp_ptp=1 (the bootp
#                   announcement carries a PTP snapshot) and ptp_clock=1 (a /dev/ptpN hardware
#                   clock for phc2sys), with ptp_clk_hz=CLK_HZ (default 125000000, the
#                   CertusPro-NX Versa's host/PTP clock), and
#                   persist the same parameters for the boot-time load in
#                   /etc/modprobe.d/hololink_pcie_drv.conf. Without --ptp that file is removed,
#                   so a plain deploy turns PTP back off. Only for bitfiles that implement the
#                   bar_ptp block: on one that stubs BAR0 0x100..0x1FF a read there can stall
#                   the interconnect (the driver defaults both parameters off for that reason).
#   --dry-run       print the privileged commands instead of running them (local mode)
#
# Without a positional argument (local mode), on this machine:
#   1. check DIR/hololink_pcie_drv.ko was built for the running kernel (vermagic), and that
#      the host1x_syncpt_* symbol CRCs it imports are the ones this kernel's host1x exports:
#      the in-tree host1x's (base Module.symvers) on an IGX Orin, the out-of-tree host1x(O)'s
#      (that tree's Module.symvers, e.g. /usr/src/linux-headers-tegra-oot-*/nvidia-oot/) on
#      an IGX Thor. A mismatch is reported here with the rebuild command instead of as
#      insmod's bare "Invalid parameters";
#   2. install it as /lib/modules/$(uname -r)/updates/hololink_pcie_drv.ko (replacing any
#      copy already there), run depmod, and write /etc/modules-load.d/hololink_pcie_drv.conf
#      so it loads at boot;
#   3. make /dev/hololink_pcie* usable without root: create the `hololink` group, install
#      /etc/udev/rules.d/99-hololink-pcie.rules (MODE 0660, GROUP hololink), reload udev,
#      and add the invoking user ($SUDO_USER) to the group. A membership added here applies
#      to new logins only (or `newgrp hololink` in an existing shell); the script says so;
#   4. rmmod the loaded hololink_pcie_drv (naming any process holding /dev/hololink_pcie*
#      instead of failing obscurely) and insmod DIR/hololink_pcie_drv.ko;
#   5. confirm the loaded module, the file just inserted and the installed copy are the
#      same build, and that modprobe resolves to the installed copy -- i.e. the module
#      running now is the one the next boot will load. Loaded-vs-file identity is the GNU
#      build-id (the .ko's .note.gnu.build-id against /sys/module/<name>/notes/), which
#      every module carries; srcversion is only a fallback, since the kernel emits it only
#      with CONFIG_MODULE_SRCVERSION_ALL or a MODULE_VERSION() in the driver (the Tegra
#      6.8 kernels have neither). File-vs-installed identity is a sha256; the device nodes
#      must read crw-rw---- root hololink.
#
# With user@host:path (remote mode): scp this script and DIR/hololink_pcie_drv.ko to that
# directory on the target, then run the script there in local mode over ssh -t:
#
#   sudo scripts/deploy_pcie_kmod.sh --path build/ehsb/kmod nvidia@10.137.145.93:~/Documents/jbonde
#
# copies scripts/deploy_pcie_kmod.sh and build/ehsb/kmod/hololink_pcie_drv.ko to
# ~/Documents/jbonde on 10.137.145.93 and executes
#
#   ssh -t nvidia@10.137.145.93 "~/Documents/jbonde/deploy_pcie_kmod.sh --path ~/Documents/jbonde"
#
# --ptp is forwarded to the remote run verbatim.
# The remote run uses sudo itself for the privileged steps (hence -t: it may prompt).
# When it succeeds, the two staged files are removed from the target again (the installed
# copy under /lib/modules stays); when it fails, they are left in place and their location
# is printed so the run can be repeated or inspected on the target.
# Neither mode needs to be run as root; sudo is used where required.
set -euo pipefail

MOD=hololink_pcie_drv
KO_NAME="$MOD.ko"
# Device-node access: the group the udev rule hands /dev/hololink_pcie* to.
GROUP=hololink
UDEV_RULE=/etc/udev/rules.d/99-hololink-pcie.rules
UDEV_RULE_LINE='KERNEL=="hololink_pcie[0-9]*", MODE="0660", GROUP="hololink"'
DIR="$PWD"
DRY_RUN=0
REMOTE=""
PTP=0
PTP_CLK_HZ=125000000
PTP_ARG=""

usage() { sed -n '2,42p' "$0" | sed 's/^# \{0,1\}//'; }
die() { echo "FAIL: $*" >&2; exit 1; }
step() { printf '\n== %s\n' "$*"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --path)
            [ $# -ge 2 ] || die "--path needs a directory"
            DIR=$2
            shift 2
            ;;
        --path=*) DIR=${1#--path=}; shift ;;
        --ptp) PTP=1; PTP_ARG="--ptp"; shift ;;
        --ptp=*)
            PTP=1
            PTP_CLK_HZ=${1#--ptp=}
            case "$PTP_CLK_HZ" in
                '' | *[!0-9]*) die "--ptp=CLK_HZ needs a positive integer, got '$PTP_CLK_HZ'" ;;
            esac
            PTP_ARG="--ptp=$PTP_CLK_HZ"
            shift
            ;;
        --dry-run | -n) DRY_RUN=1; shift ;;
        -h | --help) usage; exit 0 ;;
        -*) die "unknown option: $1 (see --help)" ;;
        *)
            [ -z "$REMOTE" ] || die "only one user@host:path may be given"
            REMOTE=$1
            shift
            ;;
    esac
done

KO="$DIR/$KO_NAME"
[ -f "$KO" ] || die "$KO not found (build the hololink_pcie_drv target, or pass --path DIR)"

# ---------------------------------------------------------------------------------------
# Remote mode: ship this script and the .ko, then run local mode there.
# ---------------------------------------------------------------------------------------
if [ -n "$REMOTE" ]; then
    case "$REMOTE" in
        *@*:*) ;;
        *) die "positional argument must be user@host:path, got '$REMOTE'" ;;
    esac
    HOSTSPEC=${REMOTE%%:*}
    REMOTE_DIR=${REMOTE#*:}
    [ -n "$REMOTE_DIR" ] || die "user@host:path needs a non-empty path"
    SCRIPT=$(readlink -f "$0")
    # The remote shell expands a leading ~ in the ssh command lines, but scp in its SFTP
    # mode (OpenSSH 9+) does not always, so the copy uses the same directory spelled
    # relative to the remote home. The directory is created first so both files land
    # inside it rather than scp creating a file of that name.
    case "$REMOTE_DIR" in
        '~') SCP_DIR="." ;;
        '~/'*) SCP_DIR=${REMOTE_DIR#\~/} ;;
        *) SCP_DIR=$REMOTE_DIR ;;
    esac
    REMOTE_SCRIPT="$REMOTE_DIR/$(basename "$SCRIPT")"
    REMOTE_KO="$REMOTE_DIR/$KO_NAME"
    step "Copying $(basename "$SCRIPT") and $KO to $HOSTSPEC:$REMOTE_DIR"
    if [ "$DRY_RUN" = 1 ]; then
        echo "  ssh $HOSTSPEC mkdir -p $REMOTE_DIR"
        echo "  scp $SCRIPT $KO $HOSTSPEC:$SCP_DIR/"
        echo "  ssh -t $HOSTSPEC $REMOTE_SCRIPT --path $REMOTE_DIR${PTP_ARG:+ $PTP_ARG}"
        echo "  on success: ssh $HOSTSPEC rm -f $REMOTE_SCRIPT $REMOTE_KO"
        echo "  on failure: leave both files on $HOSTSPEC and say where they are"
        exit 0
    fi
    ssh "$HOSTSPEC" "mkdir -p $REMOTE_DIR"
    scp -q "$SCRIPT" "$KO" "$HOSTSPEC:$SCP_DIR/"
    step "Installing on $HOSTSPEC"
    # Not exec'd: the exit status decides whether the staged files are cleaned up. Captured
    # explicitly: `$?` after an `if` whose condition failed is the if's own status (0).
    STATUS=0
    ssh -t "$HOSTSPEC" "$REMOTE_SCRIPT --path $REMOTE_DIR${PTP_ARG:+ $PTP_ARG}" || STATUS=$?
    if [ "$STATUS" -eq 0 ]; then
        step "Cleaning up the staged files on $HOSTSPEC"
        if ssh "$HOSTSPEC" "rm -f $REMOTE_SCRIPT $REMOTE_KO"; then
            echo "  removed $REMOTE_SCRIPT and $REMOTE_KO (the installed copy under /lib/modules stays)"
        else
            echo "  WARNING: could not remove $REMOTE_SCRIPT and $REMOTE_KO on $HOSTSPEC; remove them by hand" >&2
        fi
        exit 0
    fi
    # Red only when stderr is a terminal, so logs stay free of escape codes.
    RED=""; RESET=""
    if [ -t 2 ]; then RED=$'\033[1;31m'; RESET=$'\033[0m'; fi
    {
        echo
        echo "${RED}FAIL: installing $MOD on $HOSTSPEC failed (exit status $STATUS).${RESET}"
        echo "${RED}The staged files were left on $HOSTSPEC for inspection or a retry:${RESET}"
        echo "${RED}  $REMOTE_SCRIPT${RESET}"
        echo "${RED}  $REMOTE_KO${RESET}"
        echo "${RED}Retry there with: ssh -t $HOSTSPEC $REMOTE_SCRIPT --path $REMOTE_DIR${PTP_ARG:+ $PTP_ARG}${RESET}"
    } >&2
    exit "$STATUS"
fi

# ---------------------------------------------------------------------------------------
# Local mode.
# ---------------------------------------------------------------------------------------
SUDO=""
if [ "$(id -u)" != 0 ]; then
    command -v sudo >/dev/null 2>&1 || die "not root and sudo is not available"
    SUDO=sudo
fi
# Run a privileged command: under sudo when not root, printed instead when --dry-run.
run() {
    if [ "$DRY_RUN" = 1 ]; then
        echo "  [dry-run] ${SUDO:+$SUDO }$*"
    elif [ -n "$SUDO" ]; then
        "$SUDO" "$@"
    else
        "$@"
    fi
}
for tool in modinfo depmod lsmod insmod rmmod sha256sum od; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
# The device-node step is skipped, not fatal, when the group/udev tooling is absent.
NODE_ACCESS_TOOLS=1
for tool in groupadd usermod udevadm getent; do
    command -v "$tool" >/dev/null 2>&1 || NODE_ACCESS_TOOLS=0
done

# True when the module is loaded and initialised. Read from sysfs rather than
# `lsmod | grep -q`: with pipefail, grep -q exiting on the first match sends lsmod a
# SIGPIPE on a long module list, and the pipeline reports the module as absent
# exactly when it is present.
module_loaded() {
    [ "$(cat "/sys/module/$MOD/initstate" 2>/dev/null)" = "live" ]
}
# GNU build-id of a .ko (hex), empty when readelf is missing or the file carries none.
file_build_id() {
    command -v readelf >/dev/null 2>&1 || return 0
    readelf -n "$1" 2>/dev/null | awk '/Build ID/ {print $NF; exit}'
}
# GNU build-id of the loaded module, from the ELF note the kernel exposes in sysfs:
# namesz(4) descsz(4) type(4) name padded to 4, then descsz bytes of id.
loaded_build_id() {
    local note="/sys/module/$MOD/notes/.note.gnu.build-id" namesz descsz offset
    [ -r "$note" ] || return 0
    namesz=$(od -An -tu4 -N4 "$note" | tr -d ' ')
    descsz=$(od -An -tu4 -j4 -N4 "$note" | tr -d ' ')
    offset=$((12 + ((namesz + 3) & ~3)))
    od -An -tx1 -j "$offset" -N "$descsz" "$note" | tr -d ' \n'
}
# One-line identity of a .ko for logs: build-id, srcversion when present, short sha256.
describe_ko() {
    local id sv
    id=$(file_build_id "$1")
    sv=$(modinfo -F srcversion "$1" 2>/dev/null || true)
    printf 'build-id %s  srcversion %s  sha256 %s...' \
        "${id:-n/a}" "${sv:-n/a}" "$(sha256sum "$1" | cut -c1-16)"
}
# The symbol versions a .ko imports, one "0x<crc> <symbol>" per line (CONFIG_MODVERSIONS).
# `modprobe --dump-modversions` first; kmod cannot read every kernel's __versions layout (it
# fails with "Invalid argument" on the Tegra 6.8 kernels), so fall back to reading the ELF
# section directly: 64-byte records of a u64 CRC and a NUL-padded 56-byte name.
ko_modversions() {
    local out
    if out=$(modprobe --dump-modversions "$1" 2>/dev/null) && [ -n "$out" ]; then
        printf '%s\n' "$out"
        return 0
    fi
    command -v python3 >/dev/null 2>&1 || return 1
    python3 - "$1" <<'PY'
import struct, sys
data = open(sys.argv[1], "rb").read()
if data[:4] != b"\x7fELF" or data[4] != 2:
    sys.exit(1)
e = "<" if data[5] == 1 else ">"
shoff = struct.unpack_from(e + "Q", data, 0x28)[0]
shentsize, shnum, shstrndx = struct.unpack_from(e + "HHH", data, 0x3A)
def section(i):
    name, _type, _flags, _addr, offset, size = struct.unpack_from(e + "IIQQQQ", data, shoff + i * shentsize)
    return name, offset, size
_, stroff, _ = section(shstrndx)
for i in range(shnum):
    name, off, size = section(i)
    end = data.index(b"\0", stroff + name)
    if data[stroff + name:end] != b"__versions":
        continue
    for r in range(0, size - size % 64, 64):
        crc = struct.unpack_from(e + "Q", data, off + r)[0]
        sym = data[off + r + 8:off + r + 64].split(b"\0", 1)[0].decode()
        if sym:
            print("0x%08x %s" % (crc & 0xFFFFFFFF, sym))
    sys.exit(0)
sys.exit(1)
PY
}
# Who provides host1x for the running kernel: "in-tree <path>", "out-of-tree <path>",
# "built-in" (symbols in vmlinux, no module) or "" (none installed).
host1x_provider() {
    local ko
    ko=$(modinfo -n host1x 2>/dev/null || true)
    if [ -n "$ko" ]; then
        case "$ko" in
            */kernel/*) echo "in-tree $ko" ;;
            *) echo "out-of-tree $ko" ;;
        esac
    elif grep -qE ' host1x_syncpt_alloc$' /proc/kallsyms 2>/dev/null; then
        echo "built-in"
    fi
}
# The out-of-tree Module.symvers for this kernel, if one is installed: the same candidates
# the kmod Makefile searches, NVIDIA_PUBLIC_SYMVERS first.
find_oot_symvers() {
    local f
    for f in "${NVIDIA_PUBLIC_SYMVERS:-}" \
             /usr/src/nvidia/nvidia-public/Module.symvers \
             /usr/src/nvidia/nvidia-oot/Module.symvers \
             /usr/src/linux-headers-tegra-oot-*-"$KERNEL"/nvidia-oot/Module.symvers; do
        if [ -n "$f" ] && [ -r "$f" ]; then
            echo "$f"
            return 0
        fi
    done
    return 1
}
# Compare the host1x_syncpt_* CRCs recorded in the .ko with the ones the host1x this kernel
# loads was built with. Dies on a mismatch, naming the symbols and the rebuild command;
# says why when it cannot check, and says nothing is at stake when the driver imports no
# host1x symbol at all (an older driver).
check_host1x_crcs() {
    local ko=$1 imports provider kind path symvers reason="" mismatches="" line crc sym want
    if ! imports=$(ko_modversions "$ko"); then
        echo "  host1x CRCs: not checked (neither modprobe --dump-modversions nor python3 can read the module's symbol versions)"
        return 0
    fi
    imports=$(printf '%s\n' "$imports" | awk '$2 ~ /^host1x_/')
    if [ -z "$imports" ]; then
        echo "  host1x CRCs: the driver imports no host1x symbol (or this kernel has no CONFIG_MODVERSIONS); nothing to check"
        return 0
    fi
    provider=$(host1x_provider)
    kind=${provider%% *}
    path=${provider#* }
    [ "$kind" = built-in ] && path=""
    case "$kind" in
        in-tree | built-in)
            symvers="$MODDIR/build/Module.symvers"
            [ -r "$symvers" ] || { symvers=""; reason="$MODDIR/build/Module.symvers is not readable (kernel headers not installed?)"; }
            ;;
        out-of-tree)
            if ! symvers=$(find_oot_symvers); then
                symvers=""
                reason="no out-of-tree Module.symvers found for $KERNEL (looked under /usr/src/nvidia/{nvidia-public,nvidia-oot} and /usr/src/linux-headers-tegra-oot-*-$KERNEL/nvidia-oot; set NVIDIA_PUBLIC_SYMVERS=<file>)"
            fi
            ;;
        *)
            die "the driver imports host1x_syncpt_* but this kernel ($KERNEL) has no host1x module or built-in host1x; insmod would fail with 'Unknown symbol'"
            ;;
    esac
    if [ -n "${NVIDIA_PUBLIC_SYMVERS:-}" ] && [ -r "$NVIDIA_PUBLIC_SYMVERS" ]; then
        symvers=$NVIDIA_PUBLIC_SYMVERS # an explicit file wins, whatever the provider
    fi
    if [ -z "$symvers" ]; then
        echo "  host1x CRCs: not checked against the $kind host1x: $reason; insmod will be the test"
        return 0
    fi
    while read -r crc sym; do
        want=$(awk -v s="$sym" '$2 == s { print $1; exit }' "$symvers")
        [ -n "$want" ] || continue
        if [ $((crc)) -ne $((want)) ]; then
            mismatches="$mismatches
    $sym: module $crc, host1x $want"
        fi
    done <<< "$imports"
    if [ -n "$mismatches" ]; then
        local hint
        case "$kind" in
            out-of-tree) hint="NVIDIA_PUBLIC_SYMVERS=$symvers cmake --build <build-dir> --target hololink_pcie_drv  (then deploy again)" ;;
            *) hint="rebuild against this kernel's headers without NVIDIA_PUBLIC_SYMVERS: cmake --build <build-dir> --target hololink_pcie_drv" ;;
        esac
        die "host1x symbol CRCs in $KO are not the ones this kernel's $kind host1x exports ($symvers); insmod would refuse the module ('Invalid parameters', dmesg 'disagrees about version of symbol'):$mismatches
  Rebuild it: $hint"
    fi
    echo "  host1x CRCs: match the $kind host1x${path:+ ($path)} per $symvers"
}
# After a failed insmod: the kernel said why in dmesg; translate the common cases.
explain_insmod_failure() {
    local log
    log=$($SUDO dmesg 2>/dev/null | tail -n 40 || true)
    if printf '%s\n' "$log" | grep -q 'disagrees about version of symbol'; then
        printf '%s\n' "$log" | grep 'disagrees about version of symbol' | tail -n 3 | sed 's/^/  dmesg: /' >&2
        echo "  The module was built with symbol CRCs from a different host1x than this kernel loads;" \
             "rebuild it with NVIDIA_PUBLIC_SYMVERS pointed at this kernel's out-of-tree Module.symvers" \
             "(see src/ehsb/kmod/Makefile), or without it when host1x is in-tree." >&2
    elif printf '%s\n' "$log" | grep -q "$MOD: Unknown symbol"; then
        printf '%s\n' "$log" | grep "$MOD: Unknown symbol" | tail -n 3 | sed 's/^/  dmesg: /' >&2
        echo "  A symbol the driver imports is not exported by this kernel or its loaded modules (is host1x loaded?)." >&2
    elif printf '%s\n' "$log" | grep -qi 'Key was rejected by service'; then
        echo "  Secure Boot rejected the unsigned module (see src/ehsb/kmod/cross-platform-kmd.md)." >&2
    else
        printf '%s\n' "$log" | tail -n 5 | sed 's/^/  dmesg: /' >&2
    fi
}

KERNEL=$(uname -r)
MODDIR="/lib/modules/$KERNEL"
INSTALLED="$MODDIR/updates/$KO_NAME"
LOAD_CONF="/etc/modules-load.d/$MOD.conf"
OPTIONS_CONF="/etc/modprobe.d/$MOD.conf"
# Module parameters for this deployment, as one string: insmod gets them on its command line,
# the boot-time modprobe from OPTIONS_CONF.
PTP_PARAMS=""
if [ "$PTP" = 1 ]; then
    PTP_PARAMS="bootp_ptp=1 ptp_clock=1 ptp_clk_hz=$PTP_CLK_HZ"
fi

step "1. Check $KO was built for this kernel ($KERNEL)"
VERMAGIC=$(modinfo -F vermagic "$KO" | awk '{print $1}')
[ "$VERMAGIC" = "$KERNEL" ] \
    || die "vermagic '$VERMAGIC' is not the running kernel '$KERNEL'; rebuild against /lib/modules/$KERNEL/build"
BUILD_ID=$(file_build_id "$KO")
SRCVERSION=$(modinfo -F srcversion "$KO" 2>/dev/null || true)
KO_SHA=$(sha256sum "$KO" | awk '{print $1}')
echo "  vermagic $VERMAGIC  $(describe_ko "$KO")"
if [ -z "$BUILD_ID" ]; then
    if [ -n "$SRCVERSION" ]; then
        echo "  no build-id readable (readelf missing?); the loaded module will be matched by srcversion"
    else
        echo "  WARNING: neither a build-id nor a srcversion is available for $KO; the loaded" \
             "module cannot be matched to it (install binutils for readelf)"
    fi
elif [ -z "$SRCVERSION" ]; then
    echo "  (no srcversion: this kernel lacks CONFIG_MODULE_SRCVERSION_ALL and the driver has no MODULE_VERSION; build-id is used instead)"
fi
check_host1x_crcs "$KO"
if [ "$PTP" = 1 ]; then
    # A .ko built before the PTP parameters existed would make insmod fail with the opaque
    # "Unknown parameter"; say what is missing instead.
    for param in bootp_ptp ptp_clock ptp_clk_hz; do
        modinfo -F parm "$KO" 2>/dev/null | grep -q "^$param:" \
            || die "--ptp needs a driver with the '$param' parameter; $KO has none (rebuild from a tree that has it)"
    done
    echo "  PTP requested: $PTP_PARAMS"
fi
if command -v mokutil >/dev/null 2>&1 && mokutil --sb-state 2>/dev/null | grep -q enabled; then
    if ! modinfo -F sig_id "$KO" 2>/dev/null | grep -q .; then
        echo "  WARNING: Secure Boot is enabled and $KO is unsigned; insmod will report" \
             "'Key was rejected by service' (see src/ehsb/kmod/cross-platform-kmd.md)"
    fi
fi

step "2. Install as $INSTALLED and enable loading at boot"
if [ -f "$INSTALLED" ]; then
    echo "  replacing the installed copy ($(describe_ko "$INSTALLED"))"
fi
run install -D -m 0644 "$KO" "$INSTALLED"
run depmod -a "$KERNEL"
if [ "$DRY_RUN" = 1 ]; then
    echo "  [dry-run] write '$MOD' to $LOAD_CONF"
else
    printf '# Installed by deploy_pcie_kmod.sh: load the Hololink PCIe endpoint driver at boot.\n%s\n' "$MOD" \
        | $SUDO tee "$LOAD_CONF" >/dev/null
    RESOLVED=$(modinfo -n "$MOD" 2>/dev/null || true)
    [ "$RESOLVED" = "$INSTALLED" ] \
        || die "modprobe resolves $MOD to '${RESOLVED:-nothing}', not $INSTALLED; the next boot would load a different copy"
    echo "  $LOAD_CONF -> $MOD -> $INSTALLED"
fi
# The boot-time load's parameters: written with --ptp, removed without, so the persistent
# state always mirrors this deployment.
if [ "$PTP" = 1 ]; then
    if [ "$DRY_RUN" = 1 ]; then
        echo "  [dry-run] write 'options $MOD $PTP_PARAMS' to $OPTIONS_CONF"
    else
        printf '# Installed by deploy_pcie_kmod.sh --ptp: the endpoint PTP clock, for phc2sys.\noptions %s %s\n' \
            "$MOD" "$PTP_PARAMS" | $SUDO tee "$OPTIONS_CONF" >/dev/null
        echo "  $OPTIONS_CONF -> options $MOD $PTP_PARAMS"
    fi
elif [ -f "$OPTIONS_CONF" ]; then
    run rm -f "$OPTIONS_CONF"
    echo "  removed $OPTIONS_CONF (PTP off for the boot-time load)"
fi

step "3. Make /dev/hololink_pcie* usable without root (group $GROUP, $UDEV_RULE)"
# device_create() in the driver sets no devnode callback, so the node appears 0600 root:root
# and nothing that opens it (the hsb_pcie module's control plane, the receiver,
# hololink_pcie_identity) runs as a plain user until udev re-owns it. Group + rule are
# system-wide and idempotent; the user added is the one who invoked the script, which in
# remote mode is the ssh user on the target.
NEEDS_RELOGIN=0
TARGET_USER=${SUDO_USER:-}
if [ -z "$TARGET_USER" ] && [ "$(id -u)" != 0 ]; then
    TARGET_USER=$(id -un)
fi
if [ "$NODE_ACCESS_TOOLS" = 0 ]; then
    echo "  skipped: groupadd/usermod/udevadm/getent not all available; see src/ehsb/kmod/README.md 'Device node permissions'"
else
    if getent group "$GROUP" >/dev/null 2>&1; then
        echo "  group $GROUP exists"
    else
        run groupadd -f "$GROUP"
        echo "  created group $GROUP"
    fi
    if [ -f "$UDEV_RULE" ] && [ "$(grep -v '^#' "$UDEV_RULE" 2>/dev/null | grep -v '^$')" = "$UDEV_RULE_LINE" ]; then
        echo "  $UDEV_RULE already in place"
    elif [ "$DRY_RUN" = 1 ]; then
        echo "  [dry-run] write '$UDEV_RULE_LINE' to $UDEV_RULE"
    else
        printf '# Installed by deploy_pcie_kmod.sh: hand the Hololink PCIe endpoint nodes to group %s.\n%s\n' \
            "$GROUP" "$UDEV_RULE_LINE" | $SUDO tee "$UDEV_RULE" >/dev/null
        echo "  wrote $UDEV_RULE"
    fi
    run udevadm control --reload
    # Re-apply to nodes that already exist; the ones insmod creates below get the rule anyway.
    run udevadm trigger --subsystem-match=hololink_pcie
    if [ -z "$TARGET_USER" ] || [ "$TARGET_USER" = root ]; then
        echo "  no non-root invoking user to add to $GROUP (run via sudo from that user's account, or: sudo usermod -aG $GROUP <user>)"
    elif id -nG "$TARGET_USER" 2>/dev/null | tr ' ' '\n' | grep -qx "$GROUP"; then
        echo "  $TARGET_USER is already a member of $GROUP"
    else
        run usermod -aG "$GROUP" "$TARGET_USER"
        NEEDS_RELOGIN=1
        echo "  added $TARGET_USER to $GROUP"
    fi
fi

step "4. Reload the module from $KO"
if module_loaded; then
    # An open /dev/hololink_pcieN (a receiver, hololink_pcie_identity, a player) keeps the
    # module busy and rmmod would fail; name the holders instead of failing obscurely.
    if command -v fuser >/dev/null 2>&1 && $SUDO fuser -s /dev/hololink_pcie* 2>/dev/null; then
        $SUDO fuser -v /dev/hololink_pcie* 2>&1 | sed 's/^/  /' >&2 || true
        die "/dev/hololink_pcie* is open by the processes above; stop them first"
    fi
    LOADED_BEFORE=$(loaded_build_id)
    [ -n "$LOADED_BEFORE" ] || LOADED_BEFORE=$(cat "/sys/module/$MOD/srcversion" 2>/dev/null || echo "unknown")
    run rmmod "$MOD"
    echo "  removed the loaded $MOD ($LOADED_BEFORE)"
else
    echo "  not loaded"
fi
# shellcheck disable=SC2086  # PTP_PARAMS is a deliberately word-split parameter list
if ! run insmod "$KO" $PTP_PARAMS; then
    explain_insmod_failure
    die "insmod $KO failed"
fi
echo "  inserted $KO${PTP_PARAMS:+ $PTP_PARAMS}"

step "5. Confirm the loaded module is the installed one"
if [ "$DRY_RUN" = 1 ]; then
    echo "  [dry-run] compare the loaded module's build-id (or srcversion) with $KO, and $INSTALLED with $KO by sha256"
    echo "  [dry-run] check /dev/hololink_pcie* is crw-rw---- root $GROUP"
    exit 0
fi
module_loaded || die "$MOD is not loaded after insmod (see 'sudo dmesg | tail')"
# Loaded module vs the file that was inserted.
if [ -n "$BUILD_ID" ]; then
    LOADED=$(loaded_build_id)
    [ -n "$LOADED" ] || die "the kernel exposes no /sys/module/$MOD/notes/.note.gnu.build-id; cannot match the loaded module"
    [ "$LOADED" = "$BUILD_ID" ] \
        || die "loaded build-id $LOADED differs from $KO ($BUILD_ID): a different build is loaded"
    IDENTITY="build-id $LOADED"
elif [ -n "$SRCVERSION" ]; then
    LOADED=$(cat "/sys/module/$MOD/srcversion" 2>/dev/null || true)
    [ "$LOADED" = "$SRCVERSION" ] \
        || die "loaded srcversion ${LOADED:-none} differs from $KO ($SRCVERSION): a different build is loaded"
    IDENTITY="srcversion $LOADED"
else
    IDENTITY="(loaded module not verifiable: no build-id or srcversion)"
fi
# Installed copy vs the file that was inserted: byte-identical.
INSTALLED_SHA=$(sha256sum "$INSTALLED" | awk '{print $1}')
[ "$INSTALLED_SHA" = "$KO_SHA" ] \
    || die "installed copy $INSTALLED differs from $KO (sha256 ${INSTALLED_SHA:0:16} vs ${KO_SHA:0:16})"
echo "  loaded module == inserted file ($IDENTITY); installed copy == inserted file (sha256)"
# The PTP parameters took effect as requested, and the PHC exists when asked for.
for param in bootp_ptp ptp_clock; do
    if [ -r "/sys/module/$MOD/parameters/$param" ]; then
        want=N; [ "$PTP" = 1 ] && want=Y
        got=$(cat "/sys/module/$MOD/parameters/$param")
        [ "$got" = "$want" ] || die "$param reads $got after insmod, expected $want"
    fi
done
if [ "$PTP" = 1 ]; then
    PHC=""
    for dev in /sys/class/ptp/ptp*; do
        [ -e "$dev/device" ] || continue
        case "$(readlink -f "$dev/device")" in
            */[0-9a-f][0-9a-f][0-9a-f][0-9a-f]:[0-9a-f][0-9a-f]:[0-9a-f][0-9a-f].[0-7])
                # A PCI function: ours if the hololink_pcie class device hangs off the same one.
                bdf=$(basename "$(readlink -f "$dev/device")")
                if [ -d "/sys/bus/pci/devices/$bdf/hololink_pcie" ]; then
                    PHC="$PHC /dev/$(basename "$dev") ($bdf)"
                fi
                ;;
        esac
    done
    if [ -n "$PHC" ]; then
        echo "  PTP hardware clock:$PHC"
        echo "  sync it with: sudo phc2sys -s CLOCK_REALTIME -c /dev/ptpN -O 0 -S 0.001 -m   (or -s /dev/ptp0 for a ConnectX source)"
    else
        echo "  WARNING: --ptp was given but no PTP hardware clock was registered for a hololink_pcie endpoint;" \
             "the bitfile probably does not implement the bar_ptp block (see 'sudo dmesg | grep -i ptp')"
    fi
fi
echo "  $(lsmod | awk -v m="$MOD" '$1 == m')"
shopt -s nullglob
NODES=(/sys/class/hololink_pcie/hololink_pcie*)
shopt -u nullglob
if [ ${#NODES[@]} -gt 0 ]; then
    for n in "${NODES[@]}"; do
        bdf=$(basename "$(readlink -f "$n/device")")
        printf '  %s -> %s  uuid=%s  hsb_version=%s\n' \
            "$(basename "$n")" "$bdf" "$(cat "$n/uuid" 2>/dev/null || echo '?')" \
            "$(cat "$n/hsb_version" 2>/dev/null || echo '?')"
    done
    # The udev rule took: 0660 root:hololink on every node. udev applies it asynchronously
    # to the nodes insmod just created, so give it a moment before calling it a failure.
    if [ "$NODE_ACCESS_TOOLS" = 1 ]; then
        $SUDO udevadm settle --timeout=5 2>/dev/null || true
        for n in "${NODES[@]}"; do
            dev="/dev/$(basename "$n")"
            perms=$(stat -c '%a %U %G' "$dev" 2>/dev/null || echo "missing")
            if [ "$perms" = "660 root $GROUP" ]; then
                echo "  $dev is crw-rw---- root $GROUP"
            else
                echo "  WARNING: $dev is '$perms', not 660 root $GROUP; the udev rule did not apply" \
                     "(check 'udevadm test /sys/class/hololink_pcie/$(basename "$n")')" >&2
            fi
        done
    fi
else
    echo "  no endpoint bound yet (no /sys/class/hololink_pcie/*); check 'lspci -d 1204:' and 'sudo dmesg | tail'"
fi
if [ "$NEEDS_RELOGIN" = 1 ]; then
    echo "  NOTE: $TARGET_USER was just added to $GROUP; that applies to new logins only --" \
         "log out and back in, or run 'newgrp $GROUP' in each shell that should open /dev/hololink_pcie*"
fi
echo
echo "OK: $MOD (${BUILD_ID:-$SRCVERSION}) is loaded now and will load at boot from $INSTALLED"
