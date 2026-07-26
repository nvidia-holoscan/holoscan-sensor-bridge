/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_OPERATORS_HSB_CONTROLLER_HPP
#define HOLOLINK_MODULE_OPERATORS_HSB_CONTROLLER_HPP

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>

#include <cuda.h>
#include <holoscan/holoscan.hpp>

#include "hololink/module/enumeration_metadata.hpp"

#include "hololink/module/operators/network_receiver.hpp"
#include "hololink/module/operators/sensor_factory.hpp"

namespace hololink::module::operators {

/* Orchestrator between the SensorFactory (sensor + watchdog +
 * reconnection) and the NetworkReceiver (data plane), which are decoupled
 * and never see each other. HsbControllerOp is a thin HSDK adapter over
 * this. Roles: the factory drives the connection lifecycle and calls back
 * (found/lost) into this controller, which cycles the receiver's
 * construct→run / destruct and invalidates the board; the operator pulls
 * frames through get_next_frame (which taps the factory) or emits the
 * factory's fallback while disconnected.
 *
 * Sequence (App → Operator → Controller → {Factory, Receiver}):
 *
 *   App.start
 *     Op.start
 *       Ctrl.start(wake)
 *         build empty Receiver (via factory)
 *         Factory.start(on_connect=found, on_disconnect=lost, md, timeout)
 *           register_ip(peer); new Watchdog
 *   · · · device announces (reactor thread) · · ·
 *     Factory.on_enumerated(md)
 *       sensor = Factory.new_sensor(md)     // board up (control plane) + camera
 *       on_connect(md)  ->  Ctrl.found(md)  // receiver needs the control plane
 *                             Receiver.construct(md); Receiver.run()
 *                             connected = true
 *       Watchdog.arm()
 *   · · · steady state (scheduler thread) · · ·
 *     App.compute -> Op.compute
 *       Ctrl.get_next_frame(...)
 *         Receiver.get_next_frame; Factory.tap()  // refresh watchdog
 *         Receiver.stamp_metadata; frame_memory; frame_buffer_owner
 *       Op wraps the buffer as the output tensor and emits
 *   · · · stall: Watchdog fires (reactor thread) · · ·
 *     Factory.on_loss
 *       sensor.stop_sensor()
 *       on_disconnect()  ->  Ctrl.lost()
 *                              Receiver.destruct()
 *                              data_channel.device_lost()   // invalidate
 *                              wake()                        // one fallback tick
 *     Op.compute -> Ctrl.fallback_frame -> Factory.fallback_frame (app override)
 *   · · · device re-announces -> Factory.on_enumerated -> found (reconnect) · · ·
 *   App.stop -> Op.stop -> Ctrl.stop -> Factory.stop; Receiver.destruct
 *
 * Threading: found/lost run on the reactor thread, get_next_frame on the
 * scheduler thread, so the NetworkReceiver is guarded by receiver_mutex_. */
class HsbController {
public:
    HsbController(std::shared_ptr<SensorFactory> sensor_factory,
        std::shared_ptr<NetworkReceiverFactory> network_receiver_factory,
        NetworkReceiver::Config config,
        hololink::module::EnumerationMetadata metadata,
        double watchdog_timeout_s);
    ~HsbController();

    /* `wake` schedules the operator's compute() — used both as the
     * receiver's frame-ready signal and to emit one fallback on loss. */
    void start(std::function<void()> wake);
    void stop();

    /* Pull the next frame (scheduler thread). On true, `metadata` is
     * stamped and out_frame/out_size/out_owner describe the buffer to wrap
     * as the output tensor; the factory watchdog is tapped. `cuda_stream` is
     * the operator's per-compute pipeline stream, passed to the receiver so a
     * software transport places its host->device copy on it. */
    bool get_next_frame(unsigned timeout_ms,
        holoscan::MetadataDictionary& metadata, CUstream cuda_stream,
        CUdeviceptr& out_frame, size_t& out_size,
        std::shared_ptr<void>& out_owner);
    bool frames_ready();

    bool connected() const { return connected_.load(); }

    /* The current sensor's fallback frame (or {0, 0}). */
    void fallback_frame(CUdeviceptr& out_ptr, size_t& out_size);

private:
    void found(const hololink::module::EnumerationMetadata& metadata);
    void lost();

    std::shared_ptr<SensorFactory> sensor_factory_;
    std::shared_ptr<NetworkReceiverFactory> network_receiver_factory_;
    NetworkReceiver::Config config_;
    hololink::module::EnumerationMetadata metadata_;
    double watchdog_timeout_s_;
    std::function<void()> wake_;

    std::mutex receiver_mutex_;
    std::shared_ptr<NetworkReceiver> network_receiver_;
    std::atomic<bool> connected_ { false };
};

} // namespace hololink::module::operators

#endif // HOLOLINK_MODULE_OPERATORS_HSB_CONTROLLER_HPP
