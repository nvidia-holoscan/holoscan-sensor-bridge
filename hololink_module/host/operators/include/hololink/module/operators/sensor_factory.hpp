/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_OPERATORS_SENSOR_FACTORY_HPP
#define HOLOLINK_MODULE_OPERATORS_SENSOR_FACTORY_HPP

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include <cuda.h>

#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/reactor.hpp"
#include "hololink/module/status.h"
#include "hololink/module/watchdog.hpp"

#include "hololink/module/operators/sensor_device.hpp"

namespace hololink::module {
class EnumerationCallback;
}

namespace hololink::module::operators {

/* Sensor-side half of the reconnection design: it creates SensorDevices
 * and owns the frame-reception watchdog + the reconnection policy. It is
 * decoupled from the NetworkReceiver — it reports connect/disconnect to
 * HsbController through the two callbacks it is started with, and never
 * touches the data plane directly. The application subclasses it (in
 * Python via the trampoline) to supply new_sensor for its camera.
 *
 * Unified connect/reconnect: start() registers for the peer's bootp
 * announcements and stays registered for the run, so each announcement
 * while disconnected (re)connects — the first connect and every reconnect
 * share one path. */
class SensorFactory {
public:
    virtual ~SensorFactory();

    /* Create the sensor for a freshly (re)connected channel — bring the board
     * up (open the control plane), build, configure, and arm the camera,
     * returning its wrapper. Called before the receiver's construct() (which
     * programs the FPGA data plane over the control channel this opens).
     * Application override. shared_ptr (not unique_ptr) so a Python override's
     * return loads across the pybind boundary. A Python override MUST keep a
     * reference to the returned SensorDevice (e.g. store it on self): this
     * factory owns it only through a C++ shared_ptr, which does not keep the
     * Python subclass alive, so an unreferenced one is collected and its
     * overrides vanish. */
    virtual std::shared_ptr<SensorDevice> new_sensor(
        const hololink::module::EnumerationMetadata& metadata)
        = 0;

    /* Begin. `on_connect(metadata)` and `on_disconnect()` are HsbController
     * hooks: the factory calls on_connect when a device announces (so the
     * controller builds + runs the receiver) and on_disconnect on loss (so
     * the controller tears it down + invalidates). Base is virtual so a
     * sensor may change the reconnection policy. */
    virtual void start(
        std::function<void(const hololink::module::EnumerationMetadata&)> on_connect,
        std::function<void()> on_disconnect,
        const hololink::module::EnumerationMetadata& metadata,
        double watchdog_timeout_s);

    virtual void stop();

    /* A frame was delivered — refresh the watchdog. Called by HsbController
     * from get_next_frame. */
    void tap();

    /* The frame to emit while no live frame is available — the test image
     * shown at startup (before the first connect) and during an outage. The
     * base returns {0, 0} (nothing); the application overrides it to supply its
     * image. Not tied to a SensorDevice, so it works before any sensor exists. */
    virtual void fallback_frame(CUdeviceptr& out_ptr, size_t& out_size);

protected:
    /* An announcement arrived: (re)connect if not already. */
    virtual void on_enumerated(
        const hololink::module::EnumerationMetadata& metadata);

    /* The watchdog fired: the device is lost. The still-registered
     * on_enumerated reconnects when it re-announces. */
    virtual void on_loss();

    /* Evict the board's device-state objects (Hololink + everything it
     * aggregates, via the device_lost cascade) so the next (re)connect resolves
     * fresh handles. Called when a (re)connect bring-up fails on a control-plane
     * transaction — the control-plane loss path (the watchdog covers the data
     * plane). */
    void invalidate_board(const hololink::module::EnumerationMetadata& metadata);

private:
    // Guards the members below against the three threads that touch them: the
    // reactor thread (on_enumerated / on_loss), the compute thread (tap), and
    // the teardown thread (stop). Held only to copy/publish handles — never
    // across an external call (new_sensor, on_connect_/on_disconnect_, or a
    // watchdog op), so it can't cycle with HsbController's receiver_mutex_.
    std::mutex state_mutex_;
    std::function<void(const hololink::module::EnumerationMetadata&)> on_connect_;
    std::function<void()> on_disconnect_;
    std::string peer_ip_;
    std::atomic<bool> connected_ { false };
    // Set by stop() so a connect racing teardown can't re-publish state.
    bool stopped_ = false;
    // The current sensor; replaced on (re)connect.
    std::shared_ptr<SensorDevice> sensor_;
    std::shared_ptr<hololink::module::ReactorV1> reactor_;
    // shared_ptr (not unique_ptr) so tap() can hold a reference across
    // watchdog_->tap() even if stop() resets the member concurrently.
    std::shared_ptr<hololink::module::Watchdog> watchdog_;
    std::shared_ptr<hololink::module::EnumerationCallback> registration_;
};

} // namespace hololink::module::operators

#endif // HOLOLINK_MODULE_OPERATORS_SENSOR_FACTORY_HPP
