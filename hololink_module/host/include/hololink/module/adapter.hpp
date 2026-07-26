/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_ADAPTER_HPP
#define HOLOLINK_MODULE_ADAPTER_HPP

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "enumeration_metadata.hpp"
#include "module.hpp"
#include "publisher.hpp"
#include "reactor.hpp"

namespace hololink::module {

/* Abstract registration object returned by Adapter::register_ip /
 * register_all. Applications hold the handle only long enough to
 * pass it to Adapter::unregister — the registration's actual filter
 * behavior (per-peer match vs match-all) lives in private
 * subclasses in adapter.cpp. enumerate() invokes handle_metadata on
 * every registration; each subclass decides whether to actually
 * call through to its wrapped function.
 *
 * Callbacks observe metadata; they do not mutate it. The const
 * reference is the type-level guarantee: a raw subscriber cannot
 * accidentally bleed mutations into the enrichment path, and a
 * post-enrichment subscriber cannot rewrite the announcement
 * other subscribers also receive. */
class EnumerationCallback {
public:
    virtual ~EnumerationCallback() = default;
    virtual void handle_metadata(const EnumerationMetadata& metadata) = 0;
};
using EnumerationCallbackHandle = std::shared_ptr<EnumerationCallback>;

/* Process-wide singleton owning the host-side module state.
 *
 * Loads and caches modules by .so path, forwards the host->module
 * init handshake (including the ABI check), and owns the long-lived
 * host publisher Module that exposes host-provided services to loaded
 * modules. */
class Adapter {
public:
    static Adapter& get_adapter();

    Adapter(const Adapter&) = delete;
    Adapter& operator=(const Adapter&) = delete;

    /* Load a module .so by path. Performs the ABI check, calls
     * hololink_module_init, and caches the resulting Module by path.
     * Subsequent calls with the same path return the cached Module.
     *
     * Throws std::runtime_error on dlopen failure, missing required
     * symbols, ABI mismatch, or non-OK status from
     * hololink_module_init. */
    std::shared_ptr<Module> load_module(const std::filesystem::path& so_path);

    /* The single host-side Publisher. Host-provided services (Reactor,
     * Logging, etc.) register themselves here; loaded modules reach
     * those services via a Module wrapping this Publisher's callbacks.
     */
    std::shared_ptr<Publisher> host_publisher() const { return host_publisher_; }

    /* Feed an EnumerationMetadata blob through the discovery pipeline.
     * The caller fills in the bootp-decoded fields ("fpga_uuid",
     * optional "compat_id", "peer_ip", …); enumerate() loads the
     * matching module .so by UUID/compat, calls the module's
     * EnumerationInterfaceV1::update_metadata to enrich it, and stores
     * the result in a per-peer-IP slot that subsequent
     * wait_for_channel() calls consume.
     *
     * The raw_packet form passes the original bootp bytes through to
     * the module's update_metadata override — modules that need to
     * inspect vendor-specific bytes use it. The single-arg form
     * passes nullptr / 0.
     *
     * Unknown UUIDs are not an error: when no module .so matches, the
     * unenriched metadata is stored as-is so applications can still
     * receive the announcement via wait_for_channel. A matching module
     * may also return HOLOLINK_MODULE_ENUMERATION_SKIPPED to decline a
     * device it recognizes but cannot drive (e.g. an unsupported FPGA
     * version); the announcement is then suppressed (post-enrichment
     * subscribers are not notified) without throwing. Throws
     * std::runtime_error only on malformed input (missing "fpga_uuid")
     * or when a matching module's update_metadata returns any other
     * non-OK status. */
    void enumerate(EnumerationMetadata metadata);
    void enumerate(EnumerationMetadata metadata,
        const uint8_t* raw_packet, size_t raw_packet_len);

    /* Open a UDP socket on the bootp request port and attach it to
     * the host ReactorV1; each arriving bootp packet is parsed,
     * converted to an EnumerationMetadata, and fed through enumerate()
     * automatically. The default port (12267) matches the legacy
     * enumerator's bootp_request_port.
     *
     * The Adapter calls this from its constructor with the default
     * port; the underlying socket has SO_REUSEADDR / SO_REUSEPORT set
     * so the call coexists with other module-using processes on the
     * host. Application code does not need to invoke it. Tests or
     * tools that need a non-default port call stop_bootp_listener()
     * first and then start_bootp_listener(custom_port).
     *
     * Idempotent — a second call while the listener is already
     * running is a no-op. */
    void start_bootp_listener(uint32_t port = 12267);

    /* Detach the bootp listener from the reactor and close the socket.
     * Called from the destructor at process exit; application code
     * does not need to invoke it. Used by tests / tools that want to
     * rebind on a custom port. */
    void stop_bootp_listener();

    /* Block until enumerate() observes an announcement for `peer_ip`
     * *after* this call has started waiting, and return the
     * (enriched-if-known-UUID) metadata. Built on top of
     * register_ip — any announcement that fired before this call
     * doesn't count. Throws std::runtime_error when the timeout
     * elapses without an announcement. */
    EnumerationMetadata wait_for_channel(
        const std::string& peer_ip,
        std::chrono::milliseconds timeout);

    /* Register a callback that fires when enumerate() observes an
     * announcement matching the registration. The callback runs on
     * whatever thread invoked enumerate() (the bootp listener's
     * Reactor thread for the bootp-driven path; the caller's thread
     * for manual enumerate). Returns an opaque handle the caller
     * later passes to unregister(); the Adapter holds its own
     * reference to the registration so a dropped handle on the
     * caller side does not deregister the callback.
     *
     * register_ip / register_all fire AFTER enumerate() has loaded
     * the matching module and run its update_metadata enrichment —
     * the callback sees the enriched metadata (with module_name,
     * supplement-stamped fields, etc.). register_raw_ip /
     * register_raw_all fire BEFORE module loading, with only the
     * fields the bootp deserializer wrote; module .so files are not
     * loaded for a raw-only subscriber. Both pairs may coexist; the
     * raw callbacks always fire first. */
    EnumerationCallbackHandle register_ip(
        std::string peer_ip,
        std::function<void(const EnumerationMetadata&)> callback);
    EnumerationCallbackHandle register_all(
        std::function<void(const EnumerationMetadata&)> callback);
    EnumerationCallbackHandle register_raw_ip(
        std::string peer_ip,
        std::function<void(const EnumerationMetadata&)> callback);
    EnumerationCallbackHandle register_raw_all(
        std::function<void(const EnumerationMetadata&)> callback);
    void unregister(const EnumerationCallbackHandle& handle);

    /* Resolve and return the Module that serves the given enumeration
     * metadata. Reads "fpga_uuid" and (optionally) "compat_id" from
     * the metadata and performs the same lookup as enumerate(): tries
     * the compat-suffixed `hololink_<uuid>_<hex>.so` first, falls
     * back to the bare `hololink_<uuid>.so` (or uses the bare path
     * directly when no `compat_id` is present). Returns the same
     * cached Module a prior enumerate() call has already loaded.
     *
     * Applications that drove discovery through bootp + wait_for_channel
     * call this to obtain the Module handle for service lookups
     * (FrameMetadataInterface, HololinkInterface, HsbLiteInterface,
     * etc.) without composing the `.so` path themselves.
     *
     * Throws std::runtime_error when the metadata is missing
     * "fpga_uuid" or no matching `.so` exists under the configured
     * module directory. */
    std::shared_ptr<Module> get_module(const EnumerationMetadata& metadata);

    /* Apply the supplement-owned per-sensor stamping to `metadata`.
     * Resolves the module the metadata identifies, fetches its
     * ChannelConfigurationInterfaceV1 singleton, and dispatches —
     * the supplement edits whichever per-board fields move when
     * sensor_number changes (typically data_plane plus the SIF / VP /
     * HIF / frame_end address fields). Applications call this once
     * per sensor on a multi-camera board (instead of stamping the
     * fields themselves) before constructing per-channel services
     * against the resulting metadata. */
    void use_sensor(EnumerationMetadata& metadata, int64_t sensor_number);

    /* Apply the supplement-owned MTU stamping to `metadata`. Resolves
     * the same module the metadata identifies and dispatches to its
     * ChannelConfigurationInterfaceV1. Applications call this when
     * they want a non-default MTU on the RoCE data plane; the
     * per-channel constructors read the stamped value when sizing
     * packets. */
    void use_mtu(EnumerationMetadata& metadata, uint32_t mtu);

    /* Apply the supplement-owned multicast stamping to `metadata`.
     * Resolves the same module the metadata identifies and dispatches
     * to its ChannelConfigurationInterfaceV1. Applications call this
     * when they want the RoCE data plane to target a multicast group
     * (address + port) instead of a unicast peer; the per-channel
     * constructors program the FPGA from the stamped values. */
    void use_multicast(
        EnumerationMetadata& metadata, std::string address, uint16_t port);

    /* Override the module search directory used by enumerate(). When
     * unset, enumerate() reads the HOLOLINK_MODULE_DIR environment
     * variable, then falls back to two auto-discovered candidates:
     * `<host-adapter-.so dir>/../lib/hololink/modules` (covers the
     * pip-wheel install where the host module ships inside the
     * Python package) and `<exe_dir>/../lib/hololink/modules` (covers
     * installed C++ tools). Both are tried in order so a single
     * `cmake --install` and a `pip3 install ./python/` both Just
     * Work without configuration. */
    void set_module_directory(std::filesystem::path dir);

private:
    Adapter();
    ~Adapter();

    // The .so that load_module_for resolved, together with the
    // absolute path it loaded from. The path is what enumerate()
    // stamps into the metadata as "module_filename". Empty module
    // means no .so existed for the (uuid, compat_id) pair.
    struct ResolvedModule {
        std::shared_ptr<Module> module;
        std::filesystem::path path;
    };

    std::vector<std::filesystem::path> resolved_module_directories() const;
    ResolvedModule load_module_for(
        const std::string& uuid, std::optional<int64_t> compat_id);
    /* Like load_module but returns an empty shared_ptr instead of
     * throwing when the .so doesn't exist; any other load failure
     * (dlopen error, ABI mismatch, module init failure) still
     * throws. Used by load_module_for so the
     * compat-suffix-then-fall-back walk doesn't need a separate
     * filesystem::exists probe that races dlopen. */
    std::shared_ptr<Module> try_load_module(
        const std::filesystem::path& so_path);

    static void on_bootp_fd_event(int fd, short events);

    std::shared_ptr<Publisher> host_publisher_;
    std::mutex modules_mutex_;
    std::unordered_map<std::string, std::shared_ptr<Module>> modules_;

    mutable std::mutex enumeration_mutex_;
    std::optional<std::filesystem::path> module_directory_override_;

    // Application-registered enumeration callbacks. enumerate() walks
    // the list (under registrations_mutex_), collects every match for
    // the announced peer_ip (and every register_all entry), then
    // releases the mutex before invoking the callbacks so a callback
    // can re-enter register_ip / register_all / unregister without
    // deadlocking. raw_registrations_ fire before enrichment with the
    // bootp-decoded fields only; registrations_ fire after.
    std::mutex registrations_mutex_;
    std::vector<EnumerationCallbackHandle> raw_registrations_;
    std::vector<EnumerationCallbackHandle> registrations_;

    // Bootp listener state. The Reactor holds the FdCallback's
    // shared_ptr until remove_fd_callback completes, which is what
    // stop_bootp_listener calls before closing the socket.
    std::mutex bootp_mutex_;
    int bootp_socket_fd_ = -1;
    std::shared_ptr<ReactorV1::FdCallback> bootp_callback_;
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_ADAPTER_HPP
