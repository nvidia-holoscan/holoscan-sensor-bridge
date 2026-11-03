/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include "hololink/module/adapter.hpp"
#include "hololink/module/enumeration_metadata.hpp"

using hololink::module::Adapter;
using hololink::module::EnumerationMetadata;

static std::string current_timestamp()
{
    using clock = std::chrono::system_clock;
    const clock::time_point now = clock::now();
    const std::time_t secs = clock::to_time_t(now);
    const long usec = std::chrono::duration_cast<std::chrono::microseconds>(
                          now.time_since_epoch())
                          .count()
        % 1000000;
    std::tm tm_buf {};
    ::localtime_r(&secs, &tm_buf);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm_buf);
    char out[48];
    std::snprintf(out, sizeof(out), "%s.%06ld", buf, usec);
    return out;
}

static std::string render_string_value(const std::string& s)
{
    std::ostringstream oss;
    oss << '"';
    for (char c : s) {
        if (c == '\\' || c == '"') {
            oss << '\\' << c;
        } else if (c >= 0x20 && c < 0x7f) {
            oss << c;
        } else {
            oss << "\\x" << std::hex << std::setw(2) << std::setfill('0')
                << (static_cast<unsigned>(c) & 0xff) << std::dec;
        }
    }
    oss << '"';
    return oss.str();
}

static std::string render_bytes_value(const std::vector<uint8_t>& bytes)
{
    std::ostringstream oss;
    oss << "<" << bytes.size() << " bytes:";
    for (uint8_t b : bytes) {
        oss << ' ' << std::hex << std::setw(2) << std::setfill('0')
            << static_cast<unsigned>(b);
    }
    oss << ">";
    return oss.str();
}

static void print_metadata(const EnumerationMetadata& metadata)
{
    const std::string peer
        = metadata.get<std::string>("peer_ip", "<no peer_ip>");
    std::cout << "--- " << current_timestamp() << "  " << peer << " ---\n";
    if (metadata.get<int64_t>("pcie_identity_valid", 1) == 0) {
        std::cout << "  !! NOT RESPONDING: "
                  << metadata.get<std::string>("pcie_identity_error", "identity unreadable") << "\n";
    }
    if (const auto node_error = metadata.find("pcie_node_error");
        node_error != metadata.end() && std::holds_alternative<std::string>(node_error->second)) {
        std::cout << "  !! DEVICE NODE NOT OPENABLE FROM THIS PROCESS: "
                  << std::get<std::string>(node_error->second) << "\n";
    }
    for (const auto& [key, value] : metadata) {
        std::cout << "  " << key << " = ";
        std::visit([](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, int64_t>) {
                std::cout << v;
            } else if constexpr (std::is_same_v<T, std::string>) {
                std::cout << render_string_value(v);
            } else if constexpr (std::is_same_v<T, std::vector<uint8_t>>) {
                std::cout << render_bytes_value(v);
            }
        },
            value);
        std::cout << '\n';
    }
    std::cout.flush();
}

/* The identity a device is deduplicated on with -d: peer_ip, which on
 * both transports is the address control-plane messages are directed
 * to -- the IPv4 address of an Ethernet HSB, the BDF of a PCIe endpoint.
 * One entry per addressable control plane; a board that is renumbered
 * shows up again under its new address. */
static std::string identity_key(const EnumerationMetadata& metadata)
{
    std::string key = metadata.get<std::string>("peer_ip", "<no peer_ip>");
    // A dead endpoint and the same endpoint once it answers are different records.
    if (metadata.get<int64_t>("pcie_identity_valid", 1) == 0) {
        key += "#not-responding";
    }
    return key;
}

static void print_usage(const char* program)
{
    std::fprintf(stderr,
        "Usage: %s [--raw | --hololink=<peer>] [-d|--dedup]\n"
        "\n"
        "Listens for hololink bootp announcements and prints every\n"
        "name/value pair from the enumeration metadata of each\n"
        "device, in real time. Hololink PCIe endpoints are announced\n"
        "over bootp too, by the hololink_pcie kernel driver, once per\n"
        "second on the loopback interface with htype 0x50; once the\n"
        "hsb_pcie module has completed the record they appear in the\n"
        "same stream with peer_ip = mac_id = BDF DDDD:BB:DD.F,\n"
        "client_ip_address = the packed BDF, transport = \"pcie\",\n"
        "interface = lo, plus ptp_timestamp_s/ns, bootp_psn, bootp_status.\n"
        "\n"
        "Without -d every announcement prints: boards and endpoints\n"
        "re-announce over bootp about once a second, so records\n"
        "repeat. With -d each control-plane address (peer_ip: the IPv4\n"
        "address of an Ethernet HSB, the BDF of a PCIe endpoint) prints\n"
        "the first time it is seen only.\n"
        "\n"
        "Options:\n"
        "  --raw             Show only the fields decoded from the\n"
        "                    bootp packet (or synthesized from sysfs\n"
        "                    for PCIe). Module .so files are not\n"
        "                    loaded, so no module_name / enriched\n"
        "                    fields appear; a PCIe endpoint then shows\n"
        "                    peer_ip = the packed source address\n"
        "                    (0.5.1.0), mac_id = 00:05:01:00 and no\n"
        "                    vendor tail.\n"
        "  --hololink=<peer> Only print metadata from the named peer (an\n"
        "                    IP, or the BDF DDDD:BB:DD.F of a PCIe\n"
        "                    endpoint). Loads the matching module to\n"
        "                    enrich the output. Mutually exclusive\n"
        "                    with --raw.\n"
        "  -d, --dedup       Print each control-plane address once: keyed\n"
        "                    on peer_ip, i.e. the IPv4 address of an Ethernet\n"
        "                    HSB or the BDF of a PCIe endpoint.\n"
        "  -h, --help        Print this message and exit.\n",
        program);
}

int main(int argc, char** argv)
{
    bool raw = false;
    bool dedup = false;
    std::string filter_peer_ip;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--raw") {
            raw = true;
        } else if (arg == "-d" || arg == "--dedup") {
            dedup = true;
        } else if (arg.rfind("--hololink=", 0) == 0) {
            filter_peer_ip = arg.substr(std::strlen("--hololink="));
            if (filter_peer_ip.empty()) {
                std::fprintf(stderr, "--hololink= requires a peer IP or BDF\n");
                return 1;
            }
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    if (raw && !filter_peer_ip.empty()) {
        std::fprintf(stderr,
            "--raw and --hololink are mutually exclusive: --raw shows the\n"
            "pre-enrichment bootp fields only, while --hololink filters the\n"
            "post-enrichment stream.\n");
        return 1;
    }

    // Constructing the Adapter starts its bootp listener on UDP 12267.
    // Ethernet boards broadcast there; the hololink_pcie kernel driver
    // announces PCIe endpoints there too, on the loopback interface.
    Adapter& adapter = Adapter::get_adapter();

    // Callbacks fire on the reactor thread; print only when the
    // identity is new -- or always without -d.
    struct Seen {
        std::mutex mutex;
        std::set<std::string> keys;
    };
    auto seen = std::make_shared<Seen>();
    auto callback = [dedup, seen](const EnumerationMetadata& metadata) {
        if (dedup) {
            std::lock_guard<std::mutex> guard(seen->mutex);
            if (!seen->keys.insert(identity_key(metadata)).second) {
                return;
            }
        }
        print_metadata(metadata);
    };

    if (raw) {
        adapter.register_raw_all(callback);
    } else if (!filter_peer_ip.empty()) {
        adapter.register_ip(filter_peer_ip, callback);
    } else {
        adapter.register_all(callback);
    }

    if (raw) {
        std::cerr << "Listening for bootp announcements (raw, modules not loaded)..."
                  << std::endl;
    } else if (!filter_peer_ip.empty()) {
        std::cerr << "Listening for bootp announcements from " << filter_peer_ip
                  << "..." << std::endl;
    } else {
        std::cerr << "Listening for bootp announcements..." << std::endl;
    }

    // Default SIGINT / SIGTERM handlers terminate the process; the
    // Adapter destructor closes the bootp socket on the way out.
    while (true) {
        ::pause();
    }
    return 0;
}
