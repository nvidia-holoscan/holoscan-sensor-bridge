/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hololink/module/operators/sensor_factory.hpp"

#include <memory>
#include <string>
#include <utility>

#include "hololink/module/adapter.hpp"
#include "hololink/module/hololink.hpp"
#include "hololink/module/logging.hpp"

namespace hololink::module::operators {

SensorFactory::~SensorFactory()
{
    // No virtual calls here (a Python override may be gone). The controller
    // calls stop() for the orderly teardown; this just drops the watchdog +
    // registration so no callback fires against a torn-down factory.
    if (registration_) {
        hololink::module::Adapter::get_adapter().unregister(registration_);
    }
    watchdog_.reset();
}

void SensorFactory::start(
    std::function<void(const hololink::module::EnumerationMetadata&)> on_connect,
    std::function<void()> on_disconnect,
    const hololink::module::EnumerationMetadata& metadata,
    double watchdog_timeout_s)
{
    const std::string peer_ip = metadata.get<std::string>("peer_ip");
    auto& adapter = hololink::module::Adapter::get_adapter();
    auto reactor
        = ReactorV1::get_service(adapter.host_publisher()->self_module());
    auto watchdog = std::make_shared<hololink::module::Watchdog>(
        reactor, watchdog_timeout_s, [this]() { on_loss(); });
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        on_connect_ = std::move(on_connect);
        on_disconnect_ = std::move(on_disconnect);
        peer_ip_ = peer_ip;
        reactor_ = std::move(reactor);
        watchdog_ = std::move(watchdog);
        stopped_ = false;
    }
    // Register last: once registered, on_enumerated can fire on the reactor
    // thread, and it reads the state published above.
    auto registration = adapter.register_ip(
        peer_ip,
        [this](const hololink::module::EnumerationMetadata& md) {
            on_enumerated(md);
        });
    std::lock_guard<std::mutex> lock(state_mutex_);
    registration_ = std::move(registration);
}

void SensorFactory::stop()
{
    // Take ownership of every handle under the lock, then act on the locals
    // outside it. unregister()/cancel() don't wait on an in-flight
    // on_enumerated/on_loss, so one may still be running: it now finds the
    // members cleared (and stopped_ set) and becomes a no-op.
    std::shared_ptr<hololink::module::EnumerationCallback> registration;
    std::shared_ptr<hololink::module::Watchdog> watchdog;
    std::shared_ptr<SensorDevice> sensor;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        stopped_ = true;
        registration = std::move(registration_);
        watchdog = std::move(watchdog_);
        sensor = std::move(sensor_);
        on_connect_ = nullptr;
        on_disconnect_ = nullptr;
    }
    if (registration) {
        hololink::module::Adapter::get_adapter().unregister(registration);
    }
    if (watchdog) {
        watchdog->cancel();
    }
    connected_.store(false);
    // Whoever moves sensor_ out under the lock owns the single stop_sensor()
    // call: an in-flight on_loss that lost the race for the lock finds sensor_
    // already null and skips, so the sensor is stopped exactly once (never
    // twice, never zero times) regardless of how stop()/on_loss() interleave.
    if (sensor) {
        sensor->stop_sensor();
    }
}

void SensorFactory::tap()
{
    std::shared_ptr<hololink::module::Watchdog> watchdog;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        watchdog = watchdog_;
    }
    if (watchdog) {
        watchdog->tap();
    }
}

void SensorFactory::fallback_frame(CUdeviceptr& out_ptr, size_t& out_size)
{
    // Base: no fallback. Applications override to supply the test image.
    out_ptr = 0;
    out_size = 0;
}

void SensorFactory::on_enumerated(
    const hololink::module::EnumerationMetadata& metadata)
{
    // Reactor thread. Ignore announcements while already connected.
    if (connected_.load()) {
        return;
    }
    // Bring the board up first (new_sensor opens the control plane via
    // hololink.start()); the receiver's construct() then programs the FPGA
    // data plane over that same control channel, so it must run after.
    //
    // Reconnection-boundary exception handling — a deliberate, sanctioned
    // catch (the one place that owns loss recovery, the counterpart to the
    // "don't swallow backing exceptions in the wrappers" rule): a control-plane
    // transaction that fails during (re)connect bring-up propagates out of
    // new_sensor / on_connect as an exception. Treat it as control-plane loss:
    // invalidate the board so the next announcement re-materializes fresh
    // device state (the device_lost cascade) instead of re-resolving the stale
    // handles that just failed, then stay disconnected and retry on
    // re-announce. The data-plane watchdog still detects steady-state loss
    // independently. Replaces any prior sensor.
    std::function<void(const hololink::module::EnumerationMetadata&)> on_connect;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        on_connect = on_connect_;
    }
    std::shared_ptr<SensorDevice> sensor;
    try {
        sensor = new_sensor(metadata);
        if (!sensor) {
            // A Python new_sensor override reports failure by returning null;
            // its trampoline has already swallowed the underlying exception.
            HSB_LOG_WARN("SensorFactory: sensor bring-up failed; invalidating "
                         "board and awaiting re-announce.");
            invalidate_board(metadata);
            return;
        }
        if (on_connect) {
            on_connect(metadata); // controller builds + runs the receiver
        }
    } catch (const std::exception& e) {
        HSB_LOG_WARN("SensorFactory: (re)connect bring-up failed ({}); "
                     "invalidating board and awaiting re-announce.",
            e.what());
        invalidate_board(metadata);
        return;
    }
    // Publish the sensor and re-arm the watchdog under the lock, unless stop()
    // ran during bring-up — then drop the freshly built sensor (let it
    // destruct) rather than leave a live one behind a torn-down factory.
    bool published = false;
    std::shared_ptr<hololink::module::Watchdog> watchdog;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (!stopped_) {
            sensor_ = std::move(sensor);
            watchdog = watchdog_;
            connected_.store(true);
            published = true;
        }
    }
    if (!published) {
        HSB_LOG_WARN(
            "SensorFactory: torn down during bring-up; dropping sensor.");
        return;
    }
    if (watchdog) {
        watchdog->arm();
    }
}

void SensorFactory::invalidate_board(
    const hololink::module::EnumerationMetadata& metadata)
{
    // Evict the board's device-state objects so the next (re)connect resolves
    // fresh handles rather than the stale ones a control-plane failure just
    // exposed. device_lost() on the Hololink cascades to everything it
    // aggregates (data channel, oscillator, receiver, ...). Re-resolving to
    // reach it is cheap: the Hololink constructor opens no socket and does no
    // I/O (start() does), so a re-materialized-then-invalidated handle costs
    // nothing on the wire.
    std::shared_ptr<hololink::module::HololinkInterfaceV1> hololink
        = hololink::module::HololinkInterfaceV1::get_service(
            metadata, /*allow_null=*/true);
    if (hololink) {
        hololink->device_lost();
    }
}

void SensorFactory::on_loss()
{
    // Reactor thread (watchdog timeout). Disarm the sensor and tell the
    // controller to tear down + invalidate; the still-registered
    // on_enumerated reconnects on re-announce. The fallback image shown during
    // the outage comes from the factory's fallback_frame, not the sensor.
    if (!connected_.exchange(false)) {
        return;
    }
    HSB_LOG_INFO("SensorFactory: device lost; awaiting rediscovery.");
    std::shared_ptr<SensorDevice> sensor;
    std::function<void()> on_disconnect;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        // Move (not copy) so teardown ownership is exclusive: a concurrent
        // stop() then finds sensor_ null and won't stop the same sensor again.
        sensor = std::move(sensor_);
        on_disconnect = on_disconnect_;
    }
    if (sensor) {
        sensor->stop_sensor();
    }
    if (on_disconnect) {
        on_disconnect();
    }
}

} // namespace hololink::module::operators
