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

static void print_usage(const char* program)
{
    std::fprintf(stderr,
        "Usage: %s [--raw | --hololink=<peer-ip>]\n"
        "\n"
        "Listens for hololink bootp announcements and prints every\n"
        "name/value pair from the enumeration metadata of each\n"
        "device, in real time.\n"
        "\n"
        "Options:\n"
        "  --raw             Show only the fields decoded from the\n"
        "                    bootp packet. Module .so files are not\n"
        "                    loaded, so no module_name / enriched\n"
        "                    fields appear.\n"
        "  --hololink=<ip>   Only print metadata from the named peer.\n"
        "                    Loads the matching module to enrich the\n"
        "                    output. Mutually exclusive with --raw.\n"
        "  -h, --help        Print this message and exit.\n",
        program);
}

int main(int argc, char** argv)
{
    bool raw = false;
    std::string filter_peer_ip;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--raw") {
            raw = true;
        } else if (arg.rfind("--hololink=", 0) == 0) {
            filter_peer_ip = arg.substr(std::strlen("--hololink="));
            if (filter_peer_ip.empty()) {
                std::fprintf(stderr, "--hololink= requires a peer IP\n");
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

    Adapter& adapter = Adapter::get_adapter();

    auto callback = [](const EnumerationMetadata& metadata) {
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
