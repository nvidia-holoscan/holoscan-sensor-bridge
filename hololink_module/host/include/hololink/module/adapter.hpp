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
#include "logging.hpp"
#include "module.hpp"
#include "publisher.hpp"
#include "reactor.hpp"

namespace hololink::module {

// An opaque registration. Applications hold only the handle below.
class EnumerationCallback;

/**
 * @brief Identifies a registration made by one of the Adapter::register_*
 * calls; pass it to Adapter::unregister to cancel that registration.
 */
using EnumerationCallbackHandle = std::shared_ptr<EnumerationCallback>;

/**
 * @brief Process-wide entry point to the module system.
 *
 * Discovers devices on the network, resolves each to the module that drives
 * it, and hands the application the Module from which it resolves that
 * device's versioned service interfaces. One singleton instance serves the
 * process.
 *
 * Use @ref Adapter::get_adapter to get a reference to this singleton object.
 */
class Adapter {
public:
    /**
     * @brief Return the process-wide Adapter singleton.
     * @return The Adapter, valid for the lifetime of the process.
     */
    static Adapter& get_adapter();

    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    Adapter(const Adapter&) = delete;
    Adapter& operator=(const Adapter&) = delete;
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    /**
     * @brief Load the Module contained in a named shared object.
     *
     * This method loads up the given shared object and returns the Module
     * instance associated with it.  Module instances are cached, so subsequent
     * calls with the same so_path always return the same Module instance.
     *
     * Normally applications fetch Module instances by calling get_module.
     *
     * @param so_path Path to the module's shared object.
     * @return The module it contains.
     * @throws std::runtime_error if the shared object cannot be loaded, does
     * not contain a module, was built against an incompatible ABI, or fails
     * to initialize.
     */
    std::shared_ptr<Module> load_module(const std::filesystem::path& so_path);
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief Set the threshold for log messages from the host and every loaded module.
     *
     * Overrides the level chosen at startup from the `HOLOLINK_LOG_LEVEL` /
     * `HOLOSCAN_LOG_LEVEL` environment variables (default Info). Takes effect
     * immediately and may be called from any thread.
     *
     * @param level Messages below this level are discarded.
     */
    void set_log_level(LogLevel level);

    /**
     * @brief The Publisher through which the host offers services.
     *
     * Services which are instantiated by the host application include
     *
     * - ReactorV1 provides a runtime environment for callback execution execution
     * - LoggingInterfaceV1 allows loaded modules to forward their log messages to the application
     * - Module provides get_service access for these application level services
     *
     * @return The host publisher.
     */
    std::shared_ptr<Publisher> host_publisher() const { return host_publisher_; }

    /**
     * @brief Application specific device announcement to the discovery pipeline.
     *
     * Ordinarily, applications use Adapter::register_* in order to enumerate
     * devices; as those devices are discovered (e.g. via received bootp
     * messages), EnumerationMetadata is built and used to load up the driver
     * module.  If your application enumerates a device in a different way, you
     * can create your own EnumerationMetadata instance and pass it in to this
     * method in order to trigger loading of the appropriate driver module.  A
     * call here results in execution of the ordinary enumeration callbacks
     * registered by the application.  In other words, you can supply your own
     * EnumerationMetadata if the ordinary bootp-based process doesn't work.
     *
     * Note that applications do not normally call this - devices are announced
     * as their bootp packets arrive; this API is appropriate for special
     * devices and test fixtures.
     *
     * @param metadata The device to announce.
     * @param raw_packet The bootp packet `metadata` was decoded from, handed
     * on to the claiming module: one whose devices carry vendor-specific
     * bytes needs it to complete the metadata. Omit it, or pass in nullptr, when the announcement
     * did not come from a packet.
     * @param raw_packet_len Length of `raw_packet`, in bytes.
     * @throws std::runtime_error if `metadata` carries no "fpga_uuid", or if
     * the claiming module rejects it for any reason other than declining the
     * device.
     */
    void enumerate(EnumerationMetadata metadata,
        const uint8_t* raw_packet = nullptr, size_t raw_packet_len = 0);

    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    /**
     * @brief Start listening for BOOTP messages.
     *
     * This call enables the framework that listens for bootp messages and
     * passes them to enumerate.  This call is executed automatically when
     * the application first fetches an Adapter instance.
     *
     * Calling this while the listener is already running has no effect.
     *
     * @param port UDP port to listen on. The default is the port devices
     * announce to.
     */
    void start_bootp_listener(uint32_t port = 12267);
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    /**
     * @brief Stop listening for BOOTP messages.
     *
     * Applications do not need to call this; it is for callers that want to
     * listen on a different port. Devices announced through enumerate()
     * directly are unaffected.
     *
     * This does not wait for an announcement that is being handled, so it
     * may be called from an enumeration callback. The socket is closed once
     * that handling finishes, which can be after this call returns.
     */
    void stop_bootp_listener();
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief Wait for a device to announce itself, and return its enumeration metadata.
     *
     * @param peer_ip Address of the device to wait for.
     * @param timeout How long to wait before giving up.
     * @return The device's metadata (possibly supplemented by its driver module).
     * @throws std::runtime_error if the timeout elapses first.
     */
    EnumerationMetadata wait_for_channel(
        const std::string& peer_ip,
        std::chrono::milliseconds timeout);

    /**
     * @brief Subscribe to announcements from one device.
     *
     * The callback is given each announcement from `peer_ip` after the
     * claiming module has completed it, so it sees what wait_for_channel()
     * would return. It observes the announcement — a subscriber cannot change
     * what other subscribers, or wait_for_channel(), are given.  The callback
     * is invoked on the Reactor thread, so any blocking behavior in that
     * callback will delay any other callbacks scheduled there.
     *
     * Call unregister with the handle returned here to release the reference
     * to your callback function.
     *
     * @param peer_ip Address of the device to subscribe to.
     * @param callback Invoked with each completed announcement, executed within the Reactor thread.
     * @return Handle identifying the registration to unregister().
     */
    EnumerationCallbackHandle register_ip(
        std::string peer_ip,
        std::function<void(const EnumerationMetadata&)> callback);

    /**
     * @brief Subscribe to announcements from every device.
     *
     * As register_ip(), but without the address filter: the callback is given
     * every announcement, each one completed by the module that claims it.
     *
     * @param callback Invoked with each completed announcement, executed within the Reactor thread.
     * @return Handle identifying the registration to unregister().
     */
    EnumerationCallbackHandle register_all(
        std::function<void(const EnumerationMetadata&)> callback);

    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    /**
     * @brief Subscribe to announcements from one device, before any module
     * completes them.
     *
     * The callback is given only the fields decoded from the device's bootp
     * announcement; no module is consulted, and none is loaded on a raw
     * subscriber's behalf. Where a device has both raw and completed
     * subscribers, the raw ones are called first.
     *
     * @param peer_ip Address of the device to subscribe to.
     * @param callback Invoked with each raw announcement.
     * @return Handle identifying the registration to unregister().
     */
    EnumerationCallbackHandle register_raw_ip(
        std::string peer_ip,
        std::function<void(const EnumerationMetadata&)> callback);
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    /**
     * @brief Subscribe to announcements from every device, before any module
     * completes them.
     *
     * As register_raw_ip(), but without the address filter.
     *
     * @param callback Invoked with each received announcement.
     * @return Handle identifying the registration to unregister().
     */
    EnumerationCallbackHandle register_raw_all(
        std::function<void(const EnumerationMetadata&)> callback);
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief Cancel a registration.
     *
     * No announcement observed after this returns is delivered to the
     * registration. A callback already running is not interrupted.
     *
     * @param handle A handle returned by any of the register_* calls.
     * Cancelling a registration that is not currently registered has no
     * effect.
     */
    void unregister(const EnumerationCallbackHandle& handle);

    /**
     * @brief Return the Module that provides services for an announced device.
     *
     * The device is identified by the "fpga_uuid" and, where present,
     * "compat_id" fields of its announcement.
     *
     * @param metadata An announcement, e.g. returned from wait_for_channel().
     * @return The module that claims the device.
     * @throws std::runtime_error if `metadata` carries no "fpga_uuid", or if
     * no installed module claims the device.
     */
    std::shared_ptr<Module> get_module(const EnumerationMetadata& metadata);

    /**
     * @brief Select which sensor of a device to use.
     *
     * Enumeration provides information about a data path available with
     * a device, and by default, that data path is usually associated
     * with a specific sensor on that device.  This API allows the application
     * to indicate that it wants that data path to connect with a different
     * sensor.
     *
     * @code
     *  // Ordinary data-plane discovery.
     *  auto& adapter = hololink::module::Adapter::get_adapter();
     *  std::string hololink_ip = "192.168.0.2";
     *  std::chrono::seconds discovery_timeout(30);
     *  hololink::module::EnumerationMetadata metadata =
     *      adapter.wait_for_channel(hololink_ip, discovery_timeout);
     *  // Use this specific sensor port on the device.
     *  int64_t sensor_number = 1;
     *  adapter.use_sensor(metadata, sensor_number);
     *  // Now use this configured metadata with e.g. RoceReceiver
     *  // or sensor objects
     *  // ...
     * @endcode
     *
     * @param[in,out] metadata Announcement to rewrite in place.
     * @param sensor_number Zero-based index of the sensor to describe.
     * @throws std::runtime_error if no installed module claims the device,
     *  the module does not support selecting a sensor, or if the sensor_number
     *  value is invalid.
     */
    void use_sensor(EnumerationMetadata& metadata, int64_t sensor_number);

    /**
     * @brief Set the data-plane MTU.
     *
     * Similar to use_sensor, this setting configures the MTU to be used in
     * data-plane packets transmitted by HSB.  Note that setting a value larger
     * than what can be supported through all links in the network will likely
     * result in dropped packets and that the host system likely needs to have
     * a larger MTU configured in the receiver network interface.
     *
     * @param[in,out] metadata Announcement to rewrite in place.
     * @param mtu Data-plane MTU, in bytes.
     * @throws std::runtime_error if no installed module claims the device, or
     * if the module does not support setting the MTU.
     */
    void use_mtu(EnumerationMetadata& metadata, uint32_t mtu);

    /**
     * @brief Configure the data-plane to transmit to a multicast IP destination address.
     *
     * This setting directs the data plane to transmit to the given destination
     * address; this also enables the host data-plane receiver with calls to
     * IP_ADD_MEMBERSHIP.
     *
     * Note that for ROCE RDMA traffic, QP and RKEY mismatches will cause
     * received RDMA traffic to be rejected; so multicast is only expected to
     * work for a single RDMA host.  Non-RDMA hosts (e.g. LinuxReceiver)
     * currently do not check QP or RKEY values, so any number of LinuxReceiver
     * hosts can receive traffic sent this way.
     *
     * @param[in,out] metadata Announcement to rewrite in place.
     * @param address Multicast group address.
     * @param port UDP port of the multicast group.
     * @throws std::runtime_error if no installed module claims the device, or
     * if the module does not support multicast.
     */
    void use_multicast(
        EnumerationMetadata& metadata, std::string address, uint16_t port);

    /**
     * @brief Configure the directory where hololink_module shared objects are located.
     *
     * Applies to every lookup that follows, so call it before the device is
     * announced — in practice, before wait_for_channel(). With no directory
     * set, HOLOLINK_MODULE_DIR names one; failing that, the modules installed
     * alongside the host are used, so an application running against an
     * installed build needs neither.
     *
     * @param dir Directory holding the modules to load.
     */
    void set_module_directory(std::filesystem::path dir);

    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
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

    // Bootp listener state. The FdCallback owns the socket and closes it
    // when the last reference to the callback is dropped; the Reactor holds
    // a reference while a call to the callback is running.
    // bootp_socket_fd_ is only the fd number for remove_fd_callback.
    std::mutex bootp_mutex_;
    int bootp_socket_fd_ = -1;
    std::shared_ptr<ReactorV1::FdCallback> bootp_callback_;
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_ADAPTER_HPP
