/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * hololink_pcie_identity -- identity and raw BAR0 access for Hololink PCIe endpoints.
 *
 * Shows a board's identity (BDF, root complex, HSB_VER/DATE/STAT, UUID, MSI-X vectors) as
 * hololink_module enumeration will see it, lists every bound endpoint, and offers raw BAR0
 * peek/poke through the hololink_pcie kernel driver for bring-up probes such as the APB
 * sequencer check. The sysfs / ioctl reads behind it (PcieDeviceInfo, pcie_device_info.hpp)
 * are private to this tool: applications discover endpoints through the driver's bootp
 * announcement and Adapter::wait_for_channel. The board UUID (HSB_UUID_3..0) is read-only on
 * the endpoint bitfile and names the module .so, so this tool never writes it.
 */

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "hololink/module/adapter.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink_pcie_uapi.h"
#include "pcie_device_info.hpp"

namespace pcie = hololink::module::pcie;

namespace {

void print_usage(const char* program)
{
    std::fprintf(stderr,
        "Usage: %s (--bdf DDDD:BB:DD.F | --minor N) [--show]\n"
        "       %s (--bdf DDDD:BB:DD.F | --minor N) --peek OFFSET\n"
        "       %s (--bdf DDDD:BB:DD.F | --minor N) --poke OFFSET VALUE\n"
        "       %s --list\n"
        "\n"
        "Identity and raw BAR0 access for Hololink PCIe endpoints. hololink_module selects a\n"
        "board's module by the UUID the board reports in BAR0 HSB_UUID_3..0 (0xB8, 0xB4, 0xB0,\n"
        "0xAC, most significant word first); the endpoint bitfile holds it read-only, so this\n"
        "tool only reads it. The hsb_pcie module answers to %s.\n"
        "\n"
        "Device selection (one required unless --list):\n"
        "  --bdf DDDD:BB:DD.F   PCI address, resolved to /dev/hololink_pcieN via\n"
        "                       /sys/class/hololink_pcie/*/{bdf,device}.\n"
        "  --minor N            Use /dev/hololink_pcieN directly.\n"
        "\n"
        "Actions (default: --show):\n"
        "  --show               Print the board identity.\n"
        "  --peek OFFSET        Read one 32-bit BAR0 register (offset in hex or decimal).\n"
        "  --poke OFFSET VALUE  Write one 32-bit BAR0 register, then read it back.\n"
        "  --list               Scan /sys/class/hololink_pcie and print every endpoint.\n"
        "  -h, --help           Print this message and exit.\n"
        "\n"
        "Exits non-zero on any failure.\n",
        program, program, program, program, pcie::HSB_PCIE_MODULE_UUID);
}

bool parse_u32(const std::string& text, uint32_t& value)
{
    if (text.empty()) {
        return false;
    }
    char* end = nullptr;
    errno = 0;
    const unsigned long long v = std::strtoull(text.c_str(), &end, 0);
    if (errno != 0 || end == nullptr || *end != '\0' || v > 0xFFFFFFFFull) {
        return false;
    }
    value = static_cast<uint32_t>(v);
    return true;
}

void print_info(const pcie::PcieDeviceInfo& info)
{
    std::printf("device_node        %s\n", info.device_node.c_str());
    std::printf("minor              %u\n", info.minor);
    std::printf("bdf                %s\n", info.bdf.c_str());
    std::printf("root_complex       %s\n", info.root_complex.c_str());
    std::printf("sysfs_path         %s\n", info.sysfs_path.c_str());
    std::printf("requester_id       0x%04x\n", info.requester_id);
    std::printf("hsb_version        0x%08x\n", info.hsb_version);
    std::printf("hsb_date           0x%08x\n", info.hsb_date);
    std::printf("hsb_status         0x%08x\n", info.hsb_status);
    std::printf("flags              0x%08x%s%s%s\n", info.flags,
        (info.flags & HL_PCIE_ID_F_VERSION_VALID) ? " VERSION_VALID" : "",
        (info.flags & HL_PCIE_ID_F_MAGIC_VALID) ? " MAGIC_VALID" : "",
        (info.flags & HL_PCIE_ID_F_UUID_VALID) ? " UUID_VALID" : "");
    std::printf("msix_vectors       %u\n", info.msix_vectors);
    std::printf("dma_address_bits   %u\n", info.dma_address_bits);
    std::printf("bar0_len           0x%llx\n", static_cast<unsigned long long>(info.bar0_len));
    std::printf("uuid               %s%s\n", pcie::uuid_to_text(info.uuid).c_str(),
        pcie::uuid_to_text(info.uuid) == pcie::HSB_PCIE_MODULE_UUID ? " (hsb_pcie module)" : "");
    std::printf("responding         %s%s%s\n", info.responding ? "yes" : "NO",
        info.responding ? "" : " -- ", info.responding ? "" : info.identity_error.c_str());
}

int do_list()
{
    const std::vector<pcie::PcieDeviceInfo> devices = pcie::scan_pcie_devices();
    if (devices.empty()) {
        std::printf("no hololink_pcie devices found under %s\n", pcie::DEFAULT_SYSFS_CLASS);
        return 0;
    }
    std::printf("%-22s %-13s %-11s %-10s %-10s %-36s %s\n",
        "device", "bdf", "root", "hsb_ver", "requester", "uuid", "sysfs_path");
    for (const auto& d : devices) {
        std::printf("%-22s %-13s %-11s 0x%08x 0x%04x     %-36s %s\n",
            d.device_node.c_str(), d.bdf.c_str(), d.root_complex.c_str(), d.hsb_version,
            d.requester_id, pcie::uuid_to_text(d.uuid).c_str(), d.sysfs_path.c_str());
    }
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    std::optional<std::string> bdf;
    std::optional<uint16_t> minor;
    bool show = false;
    bool list = false;
    std::optional<uint32_t> peek_offset;
    std::optional<uint32_t> poke_offset;
    uint32_t poke_value = 0;

    auto need_arg = [&](int i, const char* opt) -> bool {
        if (i + 1 >= argc) {
            std::fprintf(stderr, "%s requires an argument\n", opt);
            return false;
        }
        return true;
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--bdf") {
            if (!need_arg(i, "--bdf")) {
                return 2;
            }
            bdf = argv[++i];
        } else if (arg == "--minor") {
            if (!need_arg(i, "--minor")) {
                return 2;
            }
            uint32_t m = 0;
            if (!parse_u32(argv[++i], m) || m > 0xFFFF) {
                std::fprintf(stderr, "--minor: invalid value \"%s\"\n", argv[i]);
                return 2;
            }
            minor = static_cast<uint16_t>(m);
        } else if (arg == "--show") {
            show = true;
        } else if (arg == "--list") {
            list = true;
        } else if (arg == "--peek") {
            if (!need_arg(i, "--peek")) {
                return 2;
            }
            uint32_t off = 0;
            if (!parse_u32(argv[++i], off)) {
                std::fprintf(stderr, "--peek: invalid offset \"%s\"\n", argv[i]);
                return 2;
            }
            peek_offset = off;
        } else if (arg == "--poke") {
            if (i + 2 >= argc) {
                std::fprintf(stderr, "--poke requires OFFSET and VALUE\n");
                return 2;
            }
            uint32_t off = 0;
            if (!parse_u32(argv[++i], off)) {
                std::fprintf(stderr, "--poke: invalid offset \"%s\"\n", argv[i]);
                return 2;
            }
            if (!parse_u32(argv[++i], poke_value)) {
                std::fprintf(stderr, "--poke: invalid value \"%s\"\n", argv[i]);
                return 2;
            }
            poke_offset = off;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 2;
        }
    }

    if (list) {
        return do_list();
    }

    if (!bdf && !minor) {
        std::fprintf(stderr, "one of --bdf, --minor or --list is required\n");
        print_usage(argv[0]);
        return 2;
    }
    if (bdf && minor) {
        std::fprintf(stderr, "--bdf and --minor are mutually exclusive\n");
        return 2;
    }

    if (bdf) {
        minor = pcie::minor_for_bdf(*bdf);
        if (!minor) {
            std::fprintf(stderr, "no hololink_pcie device with BDF %s under %s\n",
                bdf->c_str(), pcie::DEFAULT_SYSFS_CLASS);
            return 1;
        }
    }

    std::string error;

    if (peek_offset) {
        uint32_t value = 0;
        if (!pcie::bar_read32(*minor, *peek_offset, value, error)) {
            std::fprintf(stderr, "peek failed: %s\n", error.c_str());
            return 1;
        }
        std::printf("0x%05x = 0x%08x\n", *peek_offset, value);
        return 0;
    }

    if (poke_offset) {
        if (!pcie::bar_write32(*minor, *poke_offset, poke_value, error)) {
            std::fprintf(stderr, "poke failed: %s\n", error.c_str());
            return 1;
        }
        uint32_t value = 0;
        if (!pcie::bar_read32(*minor, *poke_offset, value, error)) {
            std::fprintf(stderr, "poke readback failed: %s\n", error.c_str());
            return 1;
        }
        std::printf("0x%05x <- 0x%08x (reads back 0x%08x)\n", *poke_offset, poke_value, value);
        return 0;
    }

    pcie::PcieDeviceInfo info;
    if (!pcie::read_device_info(*minor, info, error)) {
        std::fprintf(stderr, "cannot read identity of /dev/hololink_pcie%u: %s\n", *minor, error.c_str());
        return 1;
    }

    (void)show; // --show is the default action; kept for command-line compatibility.
    print_info(info);
    return 0;
}
