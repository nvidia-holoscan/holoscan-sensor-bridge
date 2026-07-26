/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hololink/module/adapter.hpp"

#include <dlfcn.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hololink/module/abi_check.h"
#include "hololink/module/channel_configuration.hpp"
#include "hololink/module/enumeration.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/logging.hpp"
#include "hololink/module/module.hpp"
#include "hololink/module/publisher.hpp"
#include "hololink/module/reactor.hpp"
#include "hololink/module/service_locator.h"

#include "bootp.hpp"

namespace hololink::module {

// Registration that fires only when the announced peer_ip matches.
class PeerIpEnumerationCallback : public EnumerationCallback {
public:
    PeerIpEnumerationCallback(std::string peer_ip,
        std::function<void(const EnumerationMetadata&)> callback)
        : peer_ip_(std::move(peer_ip))
        , callback_(std::move(callback))
    {
    }

    void handle_metadata(const EnumerationMetadata& metadata) override
    {
        if (metadata.contains("peer_ip")
            && metadata.get<std::string>("peer_ip") == peer_ip_
            && callback_) {
            callback_(metadata);
        }
    }

private:
    std::string peer_ip_;
    std::function<void(const EnumerationMetadata&)> callback_;
};

// Registration that fires for every announcement (register_all).
class AllPeersEnumerationCallback : public EnumerationCallback {
public:
    explicit AllPeersEnumerationCallback(
        std::function<void(const EnumerationMetadata&)> callback)
        : callback_(std::move(callback))
    {
    }

    void handle_metadata(const EnumerationMetadata& metadata) override
    {
        if (callback_) {
            callback_(metadata);
        }
    }

private:
    std::function<void(const EnumerationMetadata&)> callback_;
};

static void verify_abi(const std::string& path,
    hololink_module_abi_check_t reported)
{
    if (reported.magic != HOLOLINK_MODULE_ABI_MAGIC) {
        throw std::runtime_error(std::string(
                                     "While loading hololink_module module '")
            + path
            + "': ABI check magic mismatch (got 0x" + std::to_string(reported.magic)
            + ", expected 0x" + std::to_string(HOLOLINK_MODULE_ABI_MAGIC) + ")");
    }
    if (reported.api_version != HOLOLINK_MODULE_API_VERSION) {
        throw std::runtime_error(std::string(
                                     "While loading hololink_module module '")
            + path
            + "': ABI check api_version mismatch (module reported "
            + std::to_string(reported.api_version) + ", host expects "
            + std::to_string(HOLOLINK_MODULE_API_VERSION) + ")");
    }
    if (reported.struct_size != sizeof(hololink_module_abi_check_t)) {
        throw std::runtime_error(std::string(
                                     "While loading hololink_module module '")
            + path
            + "': ABI check struct_size mismatch (module reported "
            + std::to_string(reported.struct_size) + ", host expects "
            + std::to_string(sizeof(hololink_module_abi_check_t)) + ")");
    }
    if (reported.size_of_enumeration_metadata != sizeof(EnumerationMetadata)
        || reported.align_of_enumeration_metadata != alignof(EnumerationMetadata)) {
        throw std::runtime_error(std::string(
                                     "While loading hololink_module module '")
            + path
            + "': EnumerationMetadata sizeof/alignof mismatch (module reported "
            + std::to_string(reported.size_of_enumeration_metadata) + "/"
            + std::to_string(reported.align_of_enumeration_metadata)
            + ", host expects " + std::to_string(sizeof(EnumerationMetadata)) + "/"
            + std::to_string(alignof(EnumerationMetadata)) + ")");
    }
    if (reported.size_of_std_string != sizeof(std::string)
        || reported.align_of_std_string != alignof(std::string)) {
        throw std::runtime_error(std::string(
                                     "While loading hololink_module module '")
            + path
            + "': std::string sizeof/alignof mismatch (module reported "
            + std::to_string(reported.size_of_std_string) + "/"
            + std::to_string(reported.align_of_std_string)
            + ", host expects " + std::to_string(sizeof(std::string)) + "/"
            + std::to_string(alignof(std::string)) + ")");
    }
}

template <typename Sym>
static Sym dlsym_required(void* handle, const std::string& path, const char* symbol)
{
    ::dlerror();
    void* p = ::dlsym(handle, symbol);
    const char* err = ::dlerror();
    if (err) {
        throw std::runtime_error(std::string(
                                     "While loading hololink_module module '")
            + path
            + "': dlsym('" + symbol + "') failed: " + err);
    }
    if (!p) {
        throw std::runtime_error(std::string(
                                     "While loading hololink_module module '")
            + path
            + "': dlsym('" + symbol + "') returned NULL");
    }
    return reinterpret_cast<Sym>(p);
}

/* RAII wrapper around a dlopen handle. The shared_ptr<DloadHandle>
 * passed to LoadedModule::create as its "keeper" is destroyed when
 * the LoadedModule (and any service handle obtained through it) is
 * released, at which point dlclose drops the .so. */
class DloadHandle {
public:
    explicit DloadHandle(void* handle)
        : handle_(handle)
    {
    }
    ~DloadHandle()
    {
        if (handle_) {
            ::dlclose(handle_);
        }
    }
    DloadHandle(const DloadHandle&) = delete;
    DloadHandle& operator=(const DloadHandle&) = delete;
    void* get() const noexcept { return handle_; }

private:
    void* handle_;
};

static std::shared_ptr<Module> load_so(
    const std::filesystem::path& so_path,
    const std::shared_ptr<Publisher>& host_publisher)
{
    const std::string path = so_path.string();

    ::dlerror();
    auto handle = std::make_shared<DloadHandle>(
        ::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL));
    if (!handle->get()) {
        const char* err = ::dlerror();
        throw std::runtime_error(std::string(
                                     "While loading hololink_module module '")
            + path
            + "': dlopen failed: " + (err ? err : "unknown error"));
    }

    using GetAbiFn = hololink_module_abi_check_t (*)(void);
    GetAbiFn get_abi = dlsym_required<GetAbiFn>(handle->get(), path,
        "hololink_module_get_abi_check");
    verify_abi(path, get_abi());

    using InitFn = hololink_module_services_t (*)(const hololink_module_init_t*);
    InitFn init = dlsym_required<InitFn>(handle->get(), path, "hololink_module_init");

    auto host_callbacks = host_publisher->callbacks();

    hololink_module_init_t init_payload;
    init_payload.api_version = HOLOLINK_MODULE_API_VERSION;
    init_payload.reserved_ = 0;
    init_payload.get_service = host_callbacks.get_service;
    init_payload.release_service = host_callbacks.release_service;

    hololink_module_services_t services = init(&init_payload);

    if (services.status != HOLOLINK_MODULE_OK) {
        throw std::runtime_error(std::string(
                                     "While loading hololink_module module '")
            + path
            + "': hololink_module_init returned status "
            + std::to_string(services.status));
    }
    if (!services.get_service || !services.release_service) {
        throw std::runtime_error(std::string(
                                     "While loading hololink_module module '")
            + path
            + "': hololink_module_init returned NULL callback(s)");
    }

    return LoadedModule::create(services.get_service, services.release_service,
        std::move(handle));
}

// Defined in logging_impl.cpp / reactor_impl.cpp.
std::shared_ptr<LoggingInterfaceV1> make_logging_impl();
std::shared_ptr<ReactorV1> make_reactor_impl();
// Stops and joins the reactor's poll thread (the reactor object itself is a
// leaked singleton and is never destructed).
void shutdown_reactor_impl();

/* Host-side Publisher. The host eagerly publishes Reactor and Logging
 * at Adapter construction time and has no lazy-constructed services;
 * construct_service is a no-op. */
class HostPublisher : public Publisher {
public:
    bool construct_service(
        const std::string& /*instance_id*/,
        const std::string& /*type_id*/) override
    {
        return false;
    }
};

Adapter::Adapter()
    : host_publisher_(std::make_shared<HostPublisher>())
{
    auto logger = make_logging_impl();
    auto reactor = make_reactor_impl();
    ServicePublisher<LoggingInterfaceV1>(host_publisher_).publish("", logger);
    ServicePublisher<ReactorV1>(host_publisher_).publish("", reactor);
    set_hsb_logger_cache(logger.get());

    // The listener runs for the lifetime of the Adapter singleton.
    // SO_REUSEPORT on the underlying socket lets multiple processes
    // on the host coexist, so an unconditional start here does not
    // race with sibling tools. Tests that need a non-default port
    // call stop_bootp_listener() followed by start_bootp_listener(p).
    start_bootp_listener();
}

Adapter::~Adapter()
{
    // Order matters: drop the bootp fd callback, then stop and join the
    // reactor's poll thread while host_publisher_ (and the Logging service
    // the thread logs through) is still alive — otherwise the thread could
    // run callbacks or log through objects freed as this destructor's
    // members tear down.
    stop_bootp_listener();
    shutdown_reactor_impl();
}

Adapter& Adapter::get_adapter()
{
    static Adapter instance;
    return instance;
}

std::shared_ptr<Module> Adapter::load_module(const std::filesystem::path& so_path)
{
    const std::string key = std::filesystem::absolute(so_path).string();

    // Hold the mutex across the entire load so two threads racing on
    // the same path serialize. Loading the same .so twice would
    // dlopen succeed (refcount bumped) but the second
    // hololink_module_init would fail inside the module's
    // Publisher constructor ("a Publisher already exists in this
    // binary"), surfacing as a confusing MODULE_INIT_FAILED.
    std::lock_guard<std::mutex> guard(modules_mutex_);
    const auto it = modules_.find(key);
    if (it != modules_.end()) {
        return it->second;
    }

    auto loaded = load_so(so_path, host_publisher_);
    return modules_.emplace(key, loaded).first->second;
}

void Adapter::set_module_directory(std::filesystem::path dir)
{
    std::lock_guard<std::mutex> guard(enumeration_mutex_);
    module_directory_override_ = std::move(dir);
}

// dladdr anchor: a TU-local symbol whose address dladdr can use to
// recover the path of the .so (or executable) that contains the host
// module code. Lives in this translation unit so it gets linked into
// whatever binary/.so the host module is built into.
static void module_dir_dladdr_anchor() { }

std::vector<std::filesystem::path> Adapter::resolved_module_directories() const
{
    {
        std::lock_guard<std::mutex> guard(enumeration_mutex_);
        if (module_directory_override_) {
            return { *module_directory_override_ };
        }
    }
    if (const char* env = std::getenv("HOLOLINK_MODULE_DIR")) {
        if (env[0] != '\0') {
            return { std::filesystem::path(env) };
        }
    }

    // The install layout drops modules at <prefix>/lib/hololink/modules
    // and the host module library at <prefix>/<something>; the "where
    // is <prefix>" answer depends on what's actually running:
    //
    //   * pip3-installed wheel: <prefix> = site-packages, host module
    //     ships as site-packages/hololink_module/_hololink_py_module*.so
    //     so dladdr resolves to that .so and prefix = parent.parent.
    //
    //   * cmake --install of a C++ tool: <prefix>/bin/<tool> is the
    //     running binary, host module is statically linked, so
    //     /proc/self/exe resolves to that binary and prefix is
    //     /proc/self/exe's parent.parent.
    //
    // We assemble both candidates (deduplicated) so a single resolver
    // works for both deployment shapes; load_module_for tries each in
    // order.
    std::vector<std::filesystem::path> candidates;
    auto add_candidate = [&](std::filesystem::path base) {
        base.append("lib");
        base.append("hololink");
        base.append("modules");
        for (const auto& existing : candidates) {
            if (existing == base) {
                return;
            }
        }
        candidates.push_back(std::move(base));
    };

    Dl_info dl_info;
    if (dladdr(reinterpret_cast<void*>(&module_dir_dladdr_anchor), &dl_info) != 0
        && dl_info.dli_fname != nullptr && dl_info.dli_fname[0] != '\0') {
        const std::filesystem::path lib_path(dl_info.dli_fname);
        if (lib_path.has_parent_path()) {
            add_candidate(lib_path.parent_path().parent_path());
        }
    }

    char self_path[PATH_MAX];
    const ssize_t len = readlink("/proc/self/exe", self_path,
        sizeof(self_path) - 1);
    if (len > 0) {
        self_path[len] = '\0';
        const std::filesystem::path exe(self_path);
        add_candidate(exe.parent_path().parent_path());
    }

    // Both readlink and dladdr should succeed on Linux; the fallback
    // here only fires if both fail (which shouldn't happen in
    // practice).
    if (candidates.empty()) {
        candidates.emplace_back("/usr/lib/hololink/modules");
    }
    return candidates;
}

// compat-id is the FPGA's 16-bit IP version field. Throughout the
// adapter — CMake COMPAT/DEFAULT_COMPAT properties, .so filenames,
// and human-readable metadata renders — it is represented as a
// 4-digit lowercase hex string (e.g. the wire value 0x2603 / 9731
// decimal renders as "2603"). EnumerationMetadata stores the numeric
// value; this helper produces the string form callers compose into
// filenames and diagnostics.
static std::string compat_id_hex(int64_t compat_id)
{
    char buf[8];
    std::snprintf(buf, sizeof(buf), "%04x",
        static_cast<unsigned>(compat_id) & 0xFFFFu);
    return buf;
}

std::shared_ptr<Module> Adapter::try_load_module(
    const std::filesystem::path& so_path)
{
    const std::string key = std::filesystem::absolute(so_path).string();

    // Same locking discipline as load_module — two threads racing on
    // the same path serialize so we don't double-dlopen.
    std::lock_guard<std::mutex> guard(modules_mutex_);
    const auto it = modules_.find(key);
    if (it != modules_.end()) {
        return it->second;
    }
    try {
        auto loaded = load_so(so_path, host_publisher_);
        return modules_.emplace(key, loaded).first->second;
    } catch (const std::runtime_error&) {
        // Distinguish "file isn't there, fall back" from "file is
        // there but failed to load, surface the error". The stat
        // here narrows the race window the previous exists-then-load
        // pattern had: it only matters if the file is deleted in
        // the gap between the dlopen attempt and this check, which
        // is much smaller than the prior check-then-dlopen gap. A
        // file that's present here propagates the load error.
        std::error_code ec;
        if (!std::filesystem::exists(so_path, ec)) {
            return {};
        }
        throw;
    }
}

Adapter::ResolvedModule Adapter::load_module_for(
    const std::string& uuid, std::optional<int64_t> compat_id)
{
    if (uuid.empty()) {
        throw std::runtime_error(
            "While resolving an enumeration: metadata is missing 'fpga_uuid'");
    }

    // Lookup order, per candidate directory: <uuid>_<compat>.so when
    // compat is present, then the bare <uuid>.so. try_load_module
    // returns an empty pointer when the .so doesn't exist so the
    // fallback walk doesn't need a separate filesystem::exists probe
    // that would race the dlopen. No match in any directory → empty
    // pointer; the caller decides whether that's an error (e.g. an
    // explicit get_module(metadata) request) or a valid pass-through
    // (enumeration of a UUID we don't have a module for — the
    // unenriched metadata is still stored).
    //
    // The absolute path is recomputed here (rather than threaded back
    // out of try_load_module) so the cache key and the path stamped
    // into metadata as "module_filename" are produced the same way.
    for (const std::filesystem::path& dir : resolved_module_directories()) {
        if (compat_id) {
            std::filesystem::path with_compat = dir;
            with_compat.append("hololink_" + uuid + "_" + compat_id_hex(*compat_id) + ".so");
            if (auto module = try_load_module(with_compat)) {
                return { std::move(module), std::filesystem::absolute(with_compat) };
            }
        }
        std::filesystem::path bare = dir;
        bare.append("hololink_" + uuid + ".so");
        if (auto module = try_load_module(bare)) {
            return { std::move(module), std::filesystem::absolute(bare) };
        }
    }
    return {};
}

void Adapter::enumerate(EnumerationMetadata metadata)
{
    enumerate(std::move(metadata), nullptr, 0);
}

// The bootp deserializer always writes a 16-byte "hardware_address"
// blob (the on-wire field is fixed-width) and the meaningful prefix
// length in "hardware_address_length". Subscribers want the
// truncated form, so we normalize before any callback fires. A
// length larger than the blob (corrupt packet) or a non-byte-vector
// alternative leaves the entry alone — defensive, but harmless.
static void trim_hardware_address(EnumerationMetadata& metadata)
{
    auto it = metadata.find("hardware_address");
    if (it == metadata.end()) {
        return;
    }
    std::vector<uint8_t>* bytes = std::get_if<std::vector<uint8_t>>(&it->second);
    if (!bytes) {
        return;
    }
    if (!metadata.contains("hardware_address_length")) {
        return;
    }
    const int64_t length = metadata.get<int64_t>("hardware_address_length");
    if (length < 0) {
        return;
    }
    const size_t n = static_cast<size_t>(length);
    if (n < bytes->size()) {
        bytes->resize(n);
    }
}

void Adapter::enumerate(EnumerationMetadata metadata,
    const uint8_t* raw_packet, size_t raw_packet_len)
{
    const std::string uuid = metadata.get<std::string>("fpga_uuid");
    std::optional<int64_t> compat_id;
    if (metadata.contains("compat_id")) {
        compat_id = metadata.get<int64_t>("compat_id");
    }

    trim_hardware_address(metadata);

    // Snapshot the registrations under the mutex so callbacks can
    // re-enter register_ip / register_all / unregister without
    // deadlocking, and so a callback that unregisters itself during
    // dispatch doesn't invalidate our iterator. Holding shared_ptrs
    // in the snapshot also keeps each callback alive for its own
    // handle_metadata call even if it's just been unregistered.
    std::vector<EnumerationCallbackHandle> raw_snapshot;
    std::vector<EnumerationCallbackHandle> snapshot;
    {
        std::lock_guard<std::mutex> guard(registrations_mutex_);
        raw_snapshot = raw_registrations_;
        snapshot = registrations_;
    }

    // Raw subscribers see only the bootp-decoded fields, before any
    // module .so is touched. handle_metadata takes a const reference
    // so a raw callback cannot mutate metadata before it reaches the
    // enrichment path or the post-enrichment subscribers.
    for (const auto& reg : raw_snapshot) {
        reg->handle_metadata(metadata);
    }

    // When no post-enrichment subscriber is registered, nobody would
    // ever see an enriched field. Skipping the .so load here lets a
    // process built around register_raw_* (e.g. the adapter_enumerator
    // tool in --raw mode) run without any module .so files installed.
    //
    // When no .so matches this (uuid, compat_id) the metadata still
    // gets dispatched to registered callbacks unenriched — per the
    // plan, applications observe the announcement and decide what
    // to do; we just skip the module-side update_metadata
    // enrichment since there's no module to run it.
    if (!snapshot.empty()) {
        ResolvedModule resolved = load_module_for(uuid, compat_id);
        if (resolved.module) {
            // Record which .so the module actually loaded so
            // post-enrichment subscribers and get_module callers can
            // identify it without re-running the UUID/compat walk.
            // Stamped before update_metadata so the module's enrichment
            // can read it if it needs to; modules that overwrite this
            // key are doing something unusual.
            metadata["module_filename"] = resolved.path.string();
            auto enumeration = EnumerationInterfaceV1::get_service(resolved.module);
            const hololink_module_status_t status = enumeration->update_metadata(
                metadata, raw_packet, raw_packet_len);
            if (status == HOLOLINK_MODULE_ENUMERATION_SKIPPED) {
                // The module recognized the device but declined to drive
                // it (e.g. an unsupported FPGA version). Don't surface the
                // announcement to post-enrichment subscribers, and don't
                // treat it as a host error. The module logs the specifics;
                // we just note the skip, capped so a device re-announcing
                // on every bootp broadcast doesn't flood the log.
                constexpr unsigned MAX_SKIP_REPORTS = 4;
                static unsigned skip_reports = 0;
                if (skip_reports < MAX_SKIP_REPORTS) {
                    HSB_LOG_INFO(
                        "Module '{}' is skipping the enumeration message for "
                        "fpga_uuid='{}'.",
                        resolved.path.string(), uuid);
                    if (++skip_reports == MAX_SKIP_REPORTS) {
                        HSB_LOG_INFO(
                            "Further enumeration-skip notices will be "
                            "suppressed.");
                    }
                }
                return;
            }
            if (status != HOLOLINK_MODULE_OK) {
                throw std::runtime_error(std::string(
                                             "While enriching enumeration metadata for fpga_uuid='")
                    + uuid + "': update_metadata returned status "
                    + std::to_string(status));
            }
        }
        for (const auto& reg : snapshot) {
            reg->handle_metadata(metadata);
        }
    }
}

EnumerationCallbackHandle Adapter::register_ip(
    std::string peer_ip, std::function<void(const EnumerationMetadata&)> callback)
{
    auto entry = std::make_shared<PeerIpEnumerationCallback>(
        std::move(peer_ip), std::move(callback));
    std::lock_guard<std::mutex> guard(registrations_mutex_);
    registrations_.push_back(entry);
    return entry;
}

EnumerationCallbackHandle Adapter::register_all(
    std::function<void(const EnumerationMetadata&)> callback)
{
    auto entry = std::make_shared<AllPeersEnumerationCallback>(
        std::move(callback));
    std::lock_guard<std::mutex> guard(registrations_mutex_);
    registrations_.push_back(entry);
    return entry;
}

EnumerationCallbackHandle Adapter::register_raw_ip(
    std::string peer_ip, std::function<void(const EnumerationMetadata&)> callback)
{
    auto entry = std::make_shared<PeerIpEnumerationCallback>(
        std::move(peer_ip), std::move(callback));
    std::lock_guard<std::mutex> guard(registrations_mutex_);
    raw_registrations_.push_back(entry);
    return entry;
}

EnumerationCallbackHandle Adapter::register_raw_all(
    std::function<void(const EnumerationMetadata&)> callback)
{
    auto entry = std::make_shared<AllPeersEnumerationCallback>(
        std::move(callback));
    std::lock_guard<std::mutex> guard(registrations_mutex_);
    raw_registrations_.push_back(entry);
    return entry;
}

void Adapter::unregister(const EnumerationCallbackHandle& handle)
{
    // The handle could be in either list; the caller doesn't have to
    // track which register_* call produced it.
    std::lock_guard<std::mutex> guard(registrations_mutex_);
    registrations_.erase(
        std::remove(registrations_.begin(), registrations_.end(), handle),
        registrations_.end());
    raw_registrations_.erase(
        std::remove(raw_registrations_.begin(), raw_registrations_.end(), handle),
        raw_registrations_.end());
}

std::shared_ptr<Module> Adapter::get_module(const EnumerationMetadata& metadata)
{
    const std::string uuid = metadata.get<std::string>("fpga_uuid");
    std::optional<int64_t> compat_id;
    if (metadata.contains("compat_id")) {
        compat_id = metadata.get<int64_t>("compat_id");
    }
    ResolvedModule resolved = load_module_for(uuid, compat_id);
    if (!resolved.module) {
        std::string searched;
        bool first = true;
        for (const auto& dir : resolved_module_directories()) {
            if (!first) {
                searched += ", ";
            }
            searched += "'" + dir.string() + "'";
            first = false;
        }
        throw std::runtime_error(std::string(
                                     "While resolving the module for an enumeration: no module .so "
                                     "for fpga_uuid='")
            + uuid + "' under " + searched);
    }
    return resolved.module;
}

void Adapter::use_sensor(EnumerationMetadata& metadata, int64_t sensor_number)
{
    auto module = get_module(metadata);
    auto config = ChannelConfigurationInterfaceV1::get_service(module);
    config->use_sensor(metadata, sensor_number);
}

void Adapter::use_mtu(EnumerationMetadata& metadata, uint32_t mtu)
{
    auto module = get_module(metadata);
    auto config = ChannelConfigurationInterfaceV1::get_service(module);
    config->use_mtu(metadata, mtu);
}

void Adapter::use_multicast(
    EnumerationMetadata& metadata, std::string address, uint16_t port)
{
    auto module = get_module(metadata);
    auto config = ChannelConfigurationInterfaceV1::get_service(module);
    config->use_multicast(metadata, std::move(address), port);
}

void Adapter::start_bootp_listener(uint32_t port)
{
    std::lock_guard<std::mutex> guard(bootp_mutex_);
    if (bootp_socket_fd_ >= 0) {
        return;
    }

    auto reactor = ReactorV1::get_service(host_publisher_->self_module());

    int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
        throw std::runtime_error(std::string(
                                     "While starting the bootp listener: socket() failed: ")
            + std::strerror(errno));
    }

    // Cooperate with other processes that also want bootp on this
    // host (e.g. the adapter_enumerator tool running alongside a
    // player). SO_REUSEADDR allows binding while a prior socket sits
    // in TIME_WAIT; SO_REUSEPORT lets multiple sockets coexist on
    // the same port and the kernel delivers each broadcast bootp
    // datagram to every bound socket. Both must be set before bind,
    // which happens inside configure_socket.
    const int one = 1;
    if (::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) < 0) {
        const std::string err = std::strerror(errno);
        ::close(fd);
        throw std::runtime_error(
            "While starting the bootp listener: enabling SO_REUSEADDR failed: " + err);
    }
    if (::setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &one, sizeof(one)) < 0) {
        const std::string err = std::strerror(errno);
        ::close(fd);
        throw std::runtime_error(
            "While starting the bootp listener: enabling SO_REUSEPORT failed: " + err);
    }

    if (!configure_bootp_socket(fd, port)) {
        ::close(fd);
        throw std::runtime_error(
            "While starting the bootp listener: configure_bootp_socket failed");
    }

    bootp_socket_fd_ = fd;
    bootp_callback_ = std::make_shared<ReactorV1::FdCallback>(&Adapter::on_bootp_fd_event);
    reactor->add_fd_callback(fd, bootp_callback_, POLLIN);
}

void Adapter::stop_bootp_listener()
{
    std::lock_guard<std::mutex> guard(bootp_mutex_);
    if (bootp_socket_fd_ < 0) {
        return;
    }
    auto reactor = ReactorV1::get_service(
        host_publisher_->self_module(), /*allow_null=*/true);
    if (reactor) {
        reactor->remove_fd_callback(bootp_socket_fd_);
    }
    ::close(bootp_socket_fd_);
    bootp_socket_fd_ = -1;
    bootp_callback_.reset();
}

void Adapter::on_bootp_fd_event(int fd, short /*events*/)
{
    auto [metadata, raw_packet] = receive_bootp(fd);

    if (!metadata.contains("fpga_uuid")) {
        // Not all bootp packets carry a UUID we can dispatch on.
        return;
    }
    get_adapter().enumerate(std::move(metadata), raw_packet.data(), raw_packet.size());
}

EnumerationMetadata Adapter::wait_for_channel(
    const std::string& peer_ip, std::chrono::milliseconds timeout)
{
    // Per-call state owned jointly by this caller and the
    // enumeration-callback closure below. Each wait_for_channel
    // call carries its own state — no global cache lives in the
    // Adapter for this. The shared_ptr keeps the state alive even
    // if the registration outlasts the caller (it shouldn't, but
    // the lifetime is defensively independent).
    struct State {
        std::mutex mutex;
        std::condition_variable cv;
        std::optional<EnumerationMetadata> received;
    };
    auto state = std::make_shared<State>();

    auto handle = register_ip(peer_ip,
        [state](const EnumerationMetadata& metadata) {
            std::lock_guard<std::mutex> guard(state->mutex);
            // First fire after register_ip wins; later fires
            // before unregister are ignored.
            if (!state->received.has_value()) {
                state->received = metadata;
                state->cv.notify_all();
            }
        });

    bool ready;
    {
        std::unique_lock<std::mutex> lock(state->mutex);
        ready = state->cv.wait_for(lock, timeout,
            [&]() { return state->received.has_value(); });
    }
    unregister(handle);

    if (!ready) {
        throw std::runtime_error(std::string(
                                     "While waiting for channel: peer IP '")
            + peer_ip + "' did not announce itself within the timeout");
    }
    return std::move(*state->received);
}

} // namespace hololink::module
